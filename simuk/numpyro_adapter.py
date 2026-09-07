import inspect
import logging
from typing import NamedTuple

import jax
import numpy as np
import xarray as xr
from arviz_base import dict_to_dataset, extract, from_numpyro
from arviz_base.io_numpyro import infer_dims
from numpyro.handlers import seed, trace
from numpyro.infer import MCMC, Predictive

from simuk.backend_adapter import BackendAdapter

log = logging.getLogger(__name__)


class NumpyroAdapter(BackendAdapter):
    def __init__(self, data_dir, numpyro_model, model, simulator, single_seed):
        self.data_dir = data_dir
        self.numpyro_model = numpyro_model
        self.model = model
        self.simulator = simulator
        self._extract_model_info(single_seed)

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
        raise NotImplementedError("Posterior SBC is not implemented for numpyro")

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
            prior_pred = dict_to_dataset(prior_pred, sample_dims=["sample"])
        else:
            prior_pred = {k: v for k, v in samples.items() if k in self.observed_model_vars}
            prior_pred = dict_to_dataset(prior_pred, sample_dims=["sample"], dims=self.dims_by_site)

        prior = dict_to_dataset(prior, sample_dims=["sample"], dims=self.dims_by_site)

        return prior, prior_pred

    def _extract_model_info(self, single_seed):
        self.model_params = set(inspect.signature(self.model).parameters.keys())
        with trace() as tr:
            with seed(rng_seed=int(single_seed)):
                self.numpyro_model.model(**self.data_dir)
        self.var_names = [
            name
            for name, site in tr.items()
            if site["type"] == "sample" and not site.get("is_observed", False)
        ]
        self.observed_vars = [
            name
            for name, site in tr.items()
            if site["type"] == "sample" and site.get("is_observed", False)
        ]
        # Observed model variables are those that are marked as observed
        # and are also model function parameters in order to be able to condition on them.
        # For instance, this is used to filter out factor variables that are marked as observed
        # but cannot be conditioned on.
        self.observed_model_vars = [
            name for name in self.observed_vars if name in self.model_params
        ]

        # loop through the trace and pull the batch dim and event dim names
        # This is needed such that the prior and prior predictive samples have
        # the same dim names as the posterior samples. The prior samples from Predictive
        # does not infer dim names.
        # This uses the same helper ``from_numpyro`` applies to the posterior (via ``infer_dims``)
        self.dims_by_site = infer_dims(self.numpyro_model.model, model_kwargs=self.data_dir)

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
        """Generate posterior samples using numpyro conditioned to a prior predictive sample."""
        if method == "posterior":
            raise NotImplementedError("Posterior SBC not implemented for numpyro")

        mcmc = MCMC(self.numpyro_model, **sample_kwargs)
        rng_seed = jax.random.PRNGKey(seed)

        free_vars_data = {
            k: v
            for k, v in self.data_dir.items()
            if k not in simulation_parameters.observed_model_vars and k in self.model_params
        }
        prior_predictive_args = {
            k: v
            for k, v in replicated_data.items()
            if k in simulation_parameters.observed_model_vars
        }
        mcmc.run(rng_seed, **free_vars_data, **prior_predictive_args)
        return extract(from_numpyro(mcmc), group="posterior", keep_dataset=True)

    def subsample(self, ref_params, predictive, seed, size):
        log.info("Slicing isn't implemented for numpyro, skipping it.")
        return ref_params, predictive

    def replicate(self, predictive, idx, simulation_params):
        return {k: v.isel(sample=idx).values for k, v in predictive.items()}

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
