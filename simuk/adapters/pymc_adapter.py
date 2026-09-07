import traceback
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np
import pymc as pm
import xarray as xr
from arviz_base import dict_to_dataset, extract

from simuk.adapters.backend_adapter import BackendAdapter


class PymcAdapter(BackendAdapter):
    def __init__(self, model, simulator, trace, augment_observed, update_data):
        self.model = model
        self.simulator = simulator
        self.trace = trace
        self.augment_observed = augment_observed
        self.update_data = update_data
        self._extract_model_info()

    def get_posterior_predictive_samples(self, num_simulations, seeds, progress_bar):
        with self.model:
            num_draws = self.trace["posterior"].sizes["draw"]
            draw_indices = np.linspace(0, num_draws - 1, num_simulations, dtype=int)
            thinned_idata = self.trace.isel(draw=draw_indices)
            posterior = extract(thinned_idata, group="posterior", keep_dataset=True)

            if self.simulator is None:
                pm.sample_posterior_predictive(
                    thinned_idata,
                    extend_inferencedata=True,
                    random_seed=seeds[0],
                    progressbar=progress_bar,
                )
                posterior_pred = extract(
                    thinned_idata, group="posterior_predictive", keep_dataset=True
                )
                return posterior, posterior_pred
            else:
                posterior_pred = self._get_simulator_data(posterior, seeds)

            return posterior, posterior_pred

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

    def get_prior_predictive_samples(self, num_samples, seeds):
        """Generate samples to use for the simulations."""
        with self.model:
            idata = pm.sample_prior_predictive(draws=num_samples, random_seed=seeds[0])
            prior = extract(idata, group="prior", keep_dataset=True)

            if self.simulator is None:
                prior_pred = extract(idata, group="prior_predictive", keep_dataset=True)
                return prior, prior_pred

            prior_pred = self._get_simulator_data(prior, seeds)

        return prior, prior_pred

    def _get_simulator_data(self, free_rv_samples, seeds):
        """Run the user-defined simulator to obtain predictive samples.

        These samples can be generated from either prior or posterior samples.
        """
        # Deal with custom simulator
        pred = []
        for i in range(free_rv_samples.sizes["sample"]):
            params = {
                var: free_rv_samples[var].isel(sample=i).values for var in free_rv_samples.data_vars
            }
            params["seed"] = seeds[i]
            try:
                res = self.simulator(**params)
            except Exception as e:
                raise ValueError(
                    f"Error generating prior predictive sample with parameters {params}: {e}."
                ) from e

            if not isinstance(res, Mapping):
                raise TypeError(f"Simulator must return a dictionary, got {type(res)}")

            pred.append(res)

        pred = dict_to_dataset(
            {key: np.stack([pp[key] for pp in pred]) for key in pred[0]},
            sample_dims=["sample"],
            coords={**free_rv_samples.coords},
        )

        return pred

    def _extract_model_info(self):
        """Extract observed and free variables from the model.

        Also records the baseline state for Posterior SBC.
        """
        observed_var_nodes = [obs_rv for obs_rv in self.model.observed_RVs]
        self.observed_vars = [obs.name for obs in observed_var_nodes]
        self.var_names = [v.name for v in self.model.free_RVs]
        # Stores what observed values are given by pm.Data
        self.observed_rvs_to_pm_data = {
            var.name: (
                self.model.rvs_to_values[var].name
                if hasattr(self.model.rvs_to_values[var], "get_value")
                else None
            )
            for var in observed_var_nodes
        }
        self.model_baseline_state = self._get_baseline_state(self.model)

    def _get_baseline_state(self, model):
        """Extract the current mutable data and coordinates from a PyMC model."""
        baseline_data = {}

        # Extract Mutable Data
        for var in model.data_vars:
            if hasattr(var, "get_value"):
                baseline_data[var.name] = var.get_value(borrow=False)

        # Extract Coordinates
        # Convert the internal PyMC coordinate object to a standard dictionary
        baseline_coords = dict(model.coords)

        return {"data": baseline_data, "coords": baseline_coords}

    def simulation_params_no_simulator(self, ref_params, predictive):
        return PymcSimulationParams(
            observed_vars=self.observed_vars, var_names=self.var_names, ref_params=ref_params
        )

    def simulation_params_from_simulator(self, ref_params, predictive):
        observed_vars = list(predictive.data_vars)
        return PymcSimulationParams(
            observed_vars=observed_vars,
            var_names=list(
                filter(
                    lambda var_name: var_name not in observed_vars,
                    list(ref_params.data_vars),
                )
            ),
            ref_params=ref_params,
        )

    def subsample(self, ref_params, predictive, seed, size):
        rng = np.random.default_rng(seed)
        sample_indices = rng.choice(ref_params.sizes["sample"], size=size, replace=False)
        ref_params = ref_params.isel(sample=sample_indices)
        predictive = predictive.isel(sample=sample_indices)
        return ref_params, predictive

    def replicate(self, predictive, idx, simulation_params):
        return {
            var_name: predictive[var_name].isel(sample=idx).values
            for var_name in simulation_params.observed_vars
        }

    def get_posterior_samples(
        self, simulation_parameters, replicated_data, sample_kwargs, seed, method, simulation_idx
    ):
        """Fit the model and return posterior draws for one SBC iteration.

        For **Prior SBC** the model is conditioned on the replicated data
        alone. For **Posterior SBC** the original observed data and the
        replicated data are combined (via ``augment_observed`` or the default
        simple concatenation) and the model is conditioned on the augmented
        dataset.

        Parameters
        ----------
        replicated_data : dict[str, np.ndarray]
            Simulated observations for the current iteration, keyed by
            observed-variable name.

        Returns
        -------
        xarray.Dataset
            Posterior draws from the (augmented) model.
        """
        if method == "posterior":
            observed_data = self.trace["observed_data"]

            if self.augment_observed is not None:
                augmented_data = self.augment_observed(
                    self.model, observed_data, replicated_data, simulation_idx
                )
            else:
                # Default: concatenate original and replicated observations
                augmented_data = {
                    var_name: np.concatenate(
                        [observed_data[var_name].values, replicated_data[var_name]]
                    )
                    for var_name in simulation_parameters.observed_vars
                }

            if self.update_data is not None:
                with self.model:
                    self.update_data(self.model, augmented_data, simulation_idx)

            vars_to_observations = augmented_data
        else:
            # Prior SBC simply uses the generated prior predictive replicated data
            vars_to_observations = replicated_data

        # Set observed data that are pm.Data objects if the user hasn't modified them yet.
        # We enforce an np.array_equal check against the baseline to prevent PyMC size mismatch
        # ValueErrors when the user's `update_data` hook or `pm.observe` already updated it.
        with self.model:
            for rv, data_node in self.observed_rvs_to_pm_data.items():
                if data_node is not None and np.array_equal(
                    self.model.named_vars[data_node].get_value(),
                    self.model_baseline_state["data"][data_node],
                ):
                    pm.set_data(new_data={data_node: vars_to_observations[rv]})

        try:
            new_model = pm.observe(self.model, vars_to_observations=vars_to_observations)
            with new_model:
                check = pm.sample(**sample_kwargs, random_seed=seed)

            posterior = extract(check, group="posterior", keep_dataset=True)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            # Always ensure the model is reset to its un-augmented baseline state
            # so the next simulation iteration isn't corrupted by the previous loop's augmented data
            self._reset_model_state(self.model, self.model_baseline_state)

        return posterior

    def _reset_model_state(self, model, model_state):
        """Reset the state of PyMC model."""
        with model:
            pm.set_data(model_state["data"], coords=model_state["coords"])

    def stop_if_cant_run_without_simulator(self):
        if not self.observed_vars:
            # Ideally, we could raise an error early for `numpyro` also,
            # but `factor` also produces 'observed_vars'
            raise ValueError(
                "There are no observed variables, and PyMC will not generate predictive "
                "samples for both Prior and Posterior SBC. Either change the model or "
                "specify a simulator with the `simulator` argument."
            )


class PymcSimulationParams(NamedTuple):
    observed_vars: list[str]
    var_names: list[str]
    ref_params: xr.Dataset
