from abc import ABC, abstractmethod


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
