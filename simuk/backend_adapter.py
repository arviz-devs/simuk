from abc import ABC, abstractmethod
from collections.abc import Mapping

import numpy as np
from arviz_base import dict_to_dataset


class BackendAdapter(ABC):
    """Interface every inference-backend adapter must implement for SBC.

    Besides the abstract methods below, implementations must expose two
    attributes once constructed:

    Attributes
    ----------
    var_names : list[str]
        Names of the model's free (unobserved) variables. Rank statistics
        are computed for these.
    observed_vars : list[str]
        Names of the model's observed variables. Replicated data is
        generated for, and the model re-conditioned on, these.
    """

    var_names: list[str]
    observed_vars: list[str]

    @abstractmethod
    def compute_single_rank(self, transform, name, posterior, simulation_idx, ref_params):
        pass

    @abstractmethod
    def get_posterior_predictive_samples(self, num_simulations, seeds, progress_bar):
        pass

    @abstractmethod
    def get_prior_predictive_samples(self, num_samples, seeds):
        pass

    @abstractmethod
    def simulation_params_no_simulator(self, ref_params, predictive):
        pass

    @abstractmethod
    def simulation_params_from_simulator(self, ref_params, predictive):
        pass

    @abstractmethod
    def get_posterior_samples(
        self, simulation_parameters, replicated_data, sample_kwargs, seed, method, simulation_idx
    ):
        pass

    @abstractmethod
    def subsample(self, ref_params, predictive, seed, size):
        pass

    @abstractmethod
    def replicate(self, predictive, idx, simulation_params):
        pass

    @abstractmethod
    def stop_if_cant_run_without_simulator(self):
        pass


def get_simulator_data(simulator, free_rv_samples, seeds):
    """Run the user-defined simulator to obtain predictive samples.

    These samples can be generated from either prior or posterior samples.
    Precondition: the number of samples in free_rv_samples must equal the length of seeds.
    """
    # Deal with custom simulator
    pred = []
    for i in range(free_rv_samples.sizes["sample"]):
        params = {
            var: free_rv_samples[var].isel(sample=i).values for var in free_rv_samples.data_vars
        }
        params["seed"] = seeds[i]
        try:
            res = simulator(**params)
        except Exception as e:
            raise ValueError(
                f"Error generating predictive samples with parameters {params}: {e}."
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
