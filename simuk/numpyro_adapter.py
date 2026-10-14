import inspect
import logging
from typing import NamedTuple

import jax
import numpy as np
import xarray as xr
from arviz_base import dict_to_dataset, extract, from_numpyro
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, Predictive

from simuk.backend_adapter import BackendAdapter, get_simulator_data

log = logging.getLogger(__name__)


def _infer_dims_from_trace(tr):
    """Derive {var_name: [dim_names, ...]} from a numpyro site trace.

    Mirrors what ``arviz_base.io_numpyro.infer_dims`` computes from numpyro
    plates, but works off a trace we already have instead of re-tracing the
    model through ``numpyro.infer.initialization.init_to_sample`` -- an
    attribute-chained lookup that is not reliably populated across
    numpyro/arviz_base version combinations (see AttributeError: module
    'numpyro.infer' has no attribute 'initialization').
    """
    dims = {}
    for name, site in tr.items():
        if site["type"] not in ("sample", "deterministic"):
            continue
        batch_dims = [
            frame.name for frame in sorted(site["cond_indep_stack"], key=lambda frame: frame.dim)
        ]
        event_dims = list(site.get("infer", {}).get("event_dims", []))
        if batch_dims or event_dims:
            dims[name] = batch_dims + event_dims
    return dims


class NumpyroAdapter(BackendAdapter):
    def __init__(
        self,
        data_dir,
        numpyro_model,
        model,
        simulator,
        single_seed,
        augment_observed,
        update_data,
        trace=None,
    ):
        self.data_dir = data_dir
        self.numpyro_model = numpyro_model
        self.model = model
        self.simulator = simulator
        self.augment_observed = augment_observed
        if trace is not None:
            self.posterior = extract(trace, group="posterior", keep_dataset=True)
            self.observed_data = trace["observed_data"]
            self.data_dir = {
                **{v: self.observed_data[v].values for v in self.observed_data.data_vars},
                **data_dir,
            }
        self._extract_model_info(single_seed)

        if update_data is not None:
            raise NotImplementedError("update_data hasn't been implemented for numpyro yet")

    def compute_single_rank(self, transform, name, posterior, simulation_idx, ref_params):
        transformed_posterior = np.array(
            [
                transform(name, posterior[name].isel(sample=i).values)
                for i in range(posterior[name].sizes["sample"])
            ]
        )
        return (
            transformed_posterior
            < transform(name, ref_params[name].isel(sample=simulation_idx).values)
        ).sum(axis=0)

    def get_posterior_predictive_samples(self, num_simulations, seeds, progress_bar):
        num_draws = self.posterior.sizes["sample"]
        draw_indices = np.linspace(0, num_draws - 1, num_simulations, dtype=int)
        posterior = self.posterior.isel(sample=draw_indices)

        if self.simulator is None:
            predictive = Predictive(
                self.numpyro_model.model, posterior_samples=to_numpyro_trace(posterior)
            )

            samples = predictive(jax.random.PRNGKey(seeds[0]))
            posterior_predictive = {
                k: v for k, v in samples.items() if k in self.observed_model_vars
            }

            posterior_predictive = dict_to_dataset(posterior_predictive, sample_dims=["sample"])
            return posterior, posterior_predictive
        else:
            posterior_predictive = get_simulator_data(self.simulator, posterior, seeds)
            return posterior, posterior_predictive

    def get_prior_predictive_samples(self, num_samples, seeds):
        """Generate samples to use for the simulations using numpyro."""
        predictive = Predictive(self.model, num_samples=num_samples)
        free_vars_data = {
            k: v
            for k, v in self.data_dir.items()
            if k not in self.observed_vars and k in self.model_params
        }
        samples = predictive(jax.random.PRNGKey(seeds[0]), **free_vars_data)
        prior = {k: v for k, v in samples.items() if k not in self.observed_vars}
        if self.simulator:
            results = []
            for i, vals in enumerate(zip(*prior.values())):
                params = dict(zip(prior.keys(), vals))
                params["seed"] = seeds[i]
                results.append(self.simulator(**params))
            prior_pred = {
                key: np.asarray([result[key] for result in results]) for key in results[0]
            }
        else:
            prior_pred = {k: v for k, v in samples.items() if k in self.observed_model_vars}

        prior = dict_to_dataset(prior, sample_dims=["sample"])
        prior_pred = dict_to_dataset(prior_pred, sample_dims=["sample"])

        return prior, prior_pred

    def _extract_model_info(self, single_seed):
        self.model_params = set(inspect.signature(self.model).parameters.keys())
        with trace() as tr:
            with seed(rng_seed=int(single_seed)):
                self.numpyro_model.model(**self.data_dir)
        self.dims = _infer_dims_from_trace(tr)
        self.var_names = [
            name
            for name, site in tr.items()
            if site["type"] == "sample" and not site.get("is_observed", True)
        ]

        self.observed_vars = [
            name
            for name, site in tr.items()
            if site["type"] == "sample" and site.get("is_observed", True)
        ]

        # Observed model variables are those that are marked as observed
        # and are also model function parameters in order to be able to condition on them.
        # For instance, this is used to filter out factor variables that are marked as observed
        # but cannot be conditioned on.
        self.observed_model_vars = [
            name for name in self.observed_vars if name in self.model_params
        ]

    def simulation_params_from_simulator(self, ref_params, predictive):
        observed_vars = list(predictive.keys())
        observed_model_vars = [name for name in observed_vars if name in self.model_params]
        if not observed_model_vars:
            raise ValueError("No observed variables to condition on")

        return NumpyroSimulationParams(
            observed_vars=observed_vars,
            observed_model_vars=observed_model_vars,
            var_names=list(
                filter(
                    lambda var_name: var_name not in observed_vars,
                    list(ref_params.keys()),
                )
            ),
            ref_params=ref_params,
        )

    def simulation_params_no_simulator(self, ref_params, predictive):
        return NumpyroSimulationParams(
            observed_vars=self.observed_vars,
            observed_model_vars=self.observed_model_vars,
            var_names=self.var_names,
            ref_params=ref_params,
        )

    def get_posterior_samples(
        self, simulation_parameters, replicated_data, sample_kwargs, seed, method, simulation_idx
    ):
        """Generate posterior samples using numpyro conditioned to a predictive sample.

        Predictive sample can be either prior predictive or posterior predictive.
        """
        mcmc = MCMC(self.numpyro_model, **sample_kwargs)
        rng_seed = jax.random.PRNGKey(seed)

        if method == "posterior":
            observed_data = self.observed_data

            if self.augment_observed is not None:
                augmented_data = self.augment_observed(
                    self.model, observed_data, replicated_data, simulation_idx
                )
            else:
                # Default: stack original and replicated observations along a new
                # leading replication axis, (1, ...) + (n_rep, ...) -> (1 + n_rep, ...).
                # NumPyro broadcasts the likelihood over this axis, so each row is
                # scored as an independent dataset given the same parameters.
                def default_merge_with_logging(var_name):
                    left = observed_data[var_name].values
                    # This is to support (n_obs,) shaped arrays
                    if left.ndim == 1:
                        left = np.expand_dims(left, axis=0)
                    right = replicated_data[var_name]
                    try:
                        return np.concatenate([left, right])
                    except ValueError:
                        log.warning(
                            "Can't concatenate. Observed data dimensions %s vs"
                            " replicated data dimensions %s",
                            left.shape,
                            right.shape,
                        )
                        raise

                augmented_data = {
                    var_name: default_merge_with_logging(var_name)
                    for var_name in simulation_parameters.observed_vars
                }

            vars_to_observations = augmented_data
        else:
            # Prior  SBC simply uses the generated prior predictive replicated data
            vars_to_observations = replicated_data

        free_vars_data = {
            k: v
            for k, v in self.data_dir.items()
            if k not in simulation_parameters.observed_model_vars and k in self.model_params
        }
        prior_predictive_args = {
            k: v
            for k, v in vars_to_observations.items()
            if k in simulation_parameters.observed_model_vars
        }
        mcmc.run(rng_seed, **free_vars_data, **prior_predictive_args)

        extracted = extract(
            from_numpyro(mcmc, dims=self.dims, pred_dims=self.dims),
            group="posterior",
            keep_dataset=True,
        )
        return extracted

    def subsample(self, ref_params, predictive, seed, size):
        log.info("Slicing isn't implemented for numpyro, skipping it.")
        return ref_params, predictive

    def replicate(self, predictive, idx, simulation_params):
        """Return a dictionary of (replication_size,trailing dims) arrays.

        For now replication_size is fixed to one.
        """
        return {
            var_name: np.expand_dims(predictive[var_name].isel(sample=idx).values, axis=0)
            for var_name in simulation_params.observed_vars
        }

    def stop_if_cant_run_without_simulator(self):
        if not self.observed_model_vars:
            raise ValueError(
                "There are no observed variables we can condition on, and NumPyro "
                "will not generate prior predictive samples. Either change the model "
                "or specify a simulator with the `simulator` argument."
            )
        missing = [name for name in self.observed_model_vars if name not in self.data_dir]
        if missing:
            raise ValueError(
                "The following model parameters are missing from data_dir: "
                + ", ".join(sorted(missing))
            )


class NumpyroSimulationParams(NamedTuple):
    observed_vars: list[str]
    observed_model_vars: list[str]
    var_names: list[str]
    ref_params: xr.Dataset


def to_numpyro_trace(posterior):
    reconstructed_trace = {
        var: posterior[var].transpose("sample", ...).values for var in posterior.data_vars
    }
    return reconstructed_trace
