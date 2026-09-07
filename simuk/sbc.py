"""Simulation-based calibration checking (SBC) for PyMC, Bambi, and NumPyro.

Implements both Prior SBC (Talts et al., 2020) and Posterior SBC
(Säilynoja et al., 2025).

References
----------
.. [1] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2020).
   Validating Bayesian Inference Algorithms with Simulation-Based Calibration.
   arXiv:1804.06788.
.. [2] Säilynoja, T., Schmitt, M., Bürkner, P.-C., & Vehtari, A. (2025).
   Posterior SBC: Simulation-Based Calibration Checking Conditional on Data.
   arXiv:2502.03279.
"""

import logging
from copy import copy
from importlib.metadata import version

# Both backends are optional: these imports only provide the names used by the
# engine-detection checks in SBC.__init__, which short-circuit before touching
# a name whose backend is not installed.
try:
    import pymc as pm
except ImportError:
    pass
try:
    from numpyro.infer.mcmc import MCMCKernel
except ImportError:
    pass

import numpy as np
import xarray as xr
from arviz_base import from_dict
from tqdm import tqdm

_log = logging.getLogger(__name__)


class quiet_logging:
    """Turn off logging for PyMC, Bambi and PyTensor."""

    def __init__(self, *libraries):
        self.loggers = [logging.getLogger(library) for library in libraries]

    def __call__(self, func):
        def wrapped(cls, *args, **kwargs):
            levels = []
            for logger in self.loggers:
                levels.append(logger.level)
                logger.setLevel(logging.CRITICAL)
            res = func(cls, *args, **kwargs)
            for logger, level in zip(self.loggers, levels):
                logger.setLevel(level)
            return res

        return wrapped


class SBC:
    r"""Simulation-based calibration checking (SBC).

    Supports two modes of operation:

    - **Prior SBC** (``method="prior"``, default): validates that the inference
      algorithm across the prior. Reference draws come from the prior and replicated data
      from the prior predictive (Talts et al., 2020 [1]_).
    - **Posterior SBC** (``method="posterior"``): validates that the inference
      algorithm across the posterior. Reference draws come from the original posterior
      and replicated data from the posterior predictive. The model is then re-fit on the
      concatenation of the original observations and the replicated data
      (Säilynoja et al., 2025 [2]_).

    Parameters
    ----------
    model : pymc.Model, bambi.Model or numpyro.infer.mcmc.MCMCKernel
        A PyMC, Bambi model or NumPyro MCMC kernel. If a PyMC model the
        data needs to be defined as mutable data.
    method : {"prior", "posterior"}, default "prior"
        Which variant of SBC to perform.
    num_simulations : int, default 1000
        How many SBC iterations to run.
    sample_kwargs : dict, optional
        Keyword arguments forwarded to ``pymc.sample`` (or
        ``bambi.Model.fit`` / ``numpyro.infer.MCMC``).
    seed : int, optional
        Random seed. This persists even if running the simulations is
        paused for whatever reason.
    data_dir : dict, optional
        Keyword arguments passed to numpyro model, intended for use when providing
        an MCMC Kernel model.
    simulator : callable, optional
        A custom data-generating function. It receives the model
        parameter values as keyword arguments plus a ``seed`` integer,
        and must return a ``dict`` mapping observed-variable names to
        numpy arrays.
    trace : arviz.InferenceData, optional
        Required for ``method="posterior"``. An InferenceData object that
        contains both the ``posterior`` and ``observed_data`` groups.
        The number of posterior draws per chain must be at least ``num_simulations``.
    augment_observed : callable, optional
        *Posterior SBC only.* Signature:
        ``(model, observed_data, replicated_data, simulation_idx) -> dict``.
        Builds the augmented observed data that the model will be
        conditioned on. ``observed_data`` is the xarray Dataset from
        ``trace["observed_data"]``, and ``replicated_data`` is a
        ``dict[str, np.ndarray]`` of the simulated observations from the
        original posterior predictive for the current iteration.
        The returned ``dict`` maps variable names to the augmented data.

        The **default** behaviour concatenates the original and replicated
        observations along the first axis for each variable. Provide
        this callback when simple concatenation is not valid, e.g. for
        structured data.
    update_data : callable, optional
        *Posterior SBC only.* Signature:
        ``(model, augmented_data, simulation_idx) -> None``.
        Called *before* conditioning the model on the augmented data.
        Use this to resize covariates, coordinate labels, or other
        ``pm.Data`` containers so that the model is consistent with the
        augmented dataset.
    transform : callable, optional
        A transform applied to both the reference draw and the posterior
        draws before computing the rank statistic. Signature:
        ``(param_name, param_value) -> transformed_value``.
        Useful for defining scalar test quantities (e.g.
        ``lambda param_name, param_value: np.mean(param_value)`` to test the mean
        of a vector parameter). The return values must be comparable with the ``<``
        operator. The default is the identity (rank on the raw parameter values).
    keep_fits : bool, default True
        Whether to store posteriors to allow re-evaluation of rank statistics using
        a different quantity (``compute_rank_statistics``) without needing to run the
        simulations again. ``self.posteriors`` will be an xarray Dataset with a
        ``simulation`` dimension of length ``num_simulations``, and a ``sample`` dimension
        of the same length as the number of posterior samples in each simulation.

    Notes
    -----
    **Prior SBC** exploits the self-consistency of Bayesian updating:
    if :math:`\theta' \sim \pi(\theta)` and
    :math:`y' \sim \pi(y \mid \theta')`, then :math:`\theta'` is also
    a draw from :math:`\pi(\theta \mid y')`.  See Talts et al., 2020 [1]_.

    **Posterior SBC** uses the same self-consistency after conditioning
    on observed data :math:`y_{\text{obs}}`.  A draw
    :math:`\theta'_i \sim \pi(\theta \mid y_{\text{obs}})` and a
    replicated dataset :math:`y_i \sim \pi(y \mid \theta'_i)` are
    combined so that :math:`\theta'_i` is also a draw from
    :math:`\pi(\theta \mid y_i, y_{\text{obs}})`.  The rank of
    :math:`\theta'_i` among augmented-posterior draws should be
    uniformly distributed if the inference is calibrated.
    See Säilynoja et al., 2025 [2]_.

    References
    ----------
    .. [1] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A.
       (2020). Validating Bayesian Inference Algorithms with Simulation-Based
       Calibration. arXiv:1804.06788.
    .. [2] Säilynoja, T., Schmitt, M., Bürkner, P.-C., & Vehtari, A. (2025).
       Posterior SBC: Simulation-Based Calibration Checking Conditional on
       Data. arXiv:2502.03279.

    Examples
    --------
    **Prior SBC** (default):

    .. code-block:: python

        import pymc as pm
        import simuk

        with pm.Model() as model:
            x = pm.Normal('x')
            y = pm.Normal('y', mu=2 * x, observed=obs)

        sbc = simuk.SBC(model, num_simulations=200)
        sbc.run_simulations()

    **Posterior SBC** – validate inference conditional on observed data:

    .. code-block:: python

        import pymc as pm
        import simuk

        with pm.Model() as model:
            x = pm.Normal('x')
            y = pm.Normal('y', mu=2 * x, observed=obs)

            # 1. Obtain posterior samples from the real data
            trace = pm.sample()

        # 2. Run posterior SBC
        sbc = simuk.SBC(
            model,
            method="posterior",
            trace=trace,
            num_simulations=200,
        )
        sbc.run_simulations()
    """

    def __init__(
        self,
        model,
        method="prior",
        num_simulations=1000,
        sample_kwargs=None,
        seed=None,
        data_dir=None,
        simulator=None,
        trace=None,
        augment_observed=None,
        update_data=None,
        transform=None,
        keep_fits=True,
        progress_bar=True,
    ):
        self.num_simulations = num_simulations
        self.seed = seed
        self._seeds = self._get_seeds()

        if hasattr(model, "basic_RVs") and isinstance(model, pm.Model):
            from simuk.pymc_adapter import PymcAdapter  # noqa: PLC0415

            self.engine = "pymc"
            self.model = model
            self.adapter = PymcAdapter(self.model, simulator, trace, augment_observed, update_data)
        elif hasattr(model, "formula"):
            from simuk.pymc_adapter import PymcAdapter  # noqa: PLC0415

            self.engine = "bambi"
            model.build()
            self.bambi_model = model
            self.model = model.backend.model
            self.formula = model.formula
            self.new_data = copy(model.data)
            self.adapter = PymcAdapter(self.model, simulator, trace, augment_observed, update_data)
        elif isinstance(model, MCMCKernel):
            # runtime import so an environment with only Pymc can run SBC over Pymc models.
            from simuk.numpyro_adapter import NumpyroAdapter  # noqa: PLC0415

            self.engine = "numpyro"
            self.numpyro_model = model
            self.model = self.numpyro_model.model
            self.data_dir = data_dir if data_dir is not None else {}
            self.adapter = NumpyroAdapter(
                self.data_dir, self.numpyro_model, self.model, simulator, self._seeds[0]
            )
        else:
            raise ValueError(
                "model should be one of pymc.Model, bambi.Model, or numpyro.infer.mcmc.MCMCKernel"
            )

        if method == "posterior" and self.engine != "pymc":
            raise NotImplementedError("Currently, Posterior SBC is only implemented for PyMC")

        self.progress_bar = progress_bar

        if sample_kwargs is None:
            sample_kwargs = {}
        if self.engine == "numpyro":
            sample_kwargs.setdefault("num_warmup", 1000)
            sample_kwargs.setdefault("num_samples", 1000)
            sample_kwargs.setdefault("progress_bar", False)
        else:
            sample_kwargs.setdefault("progressbar", False)
            sample_kwargs.setdefault("compute_convergence_checks", False)
        self.sample_kwargs = sample_kwargs
        self.simulations = {name: [] for name in self.adapter.var_names}
        self._simulations_complete = 0
        self.posteriors: xr.Dataset | None = None
        self.keep_fits = keep_fits

        if simulator is not None and not callable(simulator):
            raise ValueError("simulator should be a function or None")
        if simulator is not None and self.adapter.observed_vars:
            logging.warning(
                "Provided model contains both observed variables and a simulator. "
                "Ignoring observed variables and using the simulator instead."
            )
        if simulator is None:
            self.adapter.stop_if_cant_run_without_simulator()

        self.simulator = simulator

        self._transform = lambda param_name, param_value: param_value
        if transform is not None:
            if not callable(transform):
                raise ValueError("`transform` should be a function or None")
            self._transform = transform

        self.method = method.lower()
        if self.method == "posterior":
            if trace is None:
                raise ValueError(
                    "When performing Posterior SBC, posterior samples from the "
                    "original posterior are required to generate replicate datasets"
                )
            if "posterior" not in trace:
                raise ValueError("`trace` should contain 'posterior' group")
            if "observed_data" not in trace:
                raise ValueError("`trace` should contain 'observed_data' group")
            if self.num_simulations > trace["posterior"].sizes["draw"]:
                raise ValueError(
                    "posterior samples in `trace` should have more draws per "
                    "chain than `num_simulations`. This is required to obtain enough "
                    "posterior predictive samples"
                )
            self.trace = trace

            if augment_observed is not None and not callable(augment_observed):
                raise ValueError("`augment_observed` should be a function or None")
            self.augment_observed = augment_observed

            if update_data is not None and not callable(update_data):
                raise ValueError("`update_data` should be a function or None")
            self.update_data = update_data

        else:
            if update_data is not None:
                logging.warning(
                    "`update_data` is only supported for Posterior SBC. Ignoring...\n"
                    "Prior SBC does not augment observations, so there is no need to "
                    "update model data."
                )
            if augment_observed is not None:
                logging.warning(
                    "`augment_observed` is only supported for Posterior SBC. Ignoring...\n"
                    "Prior SBC does not augment observations, so there is no need to "
                    "augment observed data and replicated data"
                )
            if trace is not None:
                logging.warning("`trace` is only used for Posterior SBC. Ignoring...")

    def _get_seeds(self):
        """Set the random seed, and generate seeds for all the simulations."""
        rng = np.random.default_rng(self.seed)
        return rng.integers(0, 2**30, size=self.num_simulations)

    def _convert_to_datatree(self):
        """Pack the rank-statistic arrays into an xarray DataTree.

        Creates a group named ``"prior_sbc"`` or ``"posterior_sbc"``
        (depending on ``self.method``) inside ``self.simulations``.
        """
        if self.method == "prior":
            group_name = "prior_sbc"
        else:
            group_name = "posterior_sbc"

        self.simulations = from_dict(
            {group_name: self.simulations},
            attrs={
                "/": {
                    "inferece_library": self.engine,
                    "inferece_library_version": version(self.engine),
                    "modeling_interface": "simuk",
                    "modeling_interface_version": version("simuk"),
                }
            },
        )

    def compute_rank_statistics(self, transform=None):
        """Compute the rank statistic for the reference parameters.

        This method computes the rank of each reference parameter value
        relative to the newly sampled posterior draws for each simulation.

        This allows users to recompute rank statistics rapidly using a
        different parameter transformation without needing to rerun the simulations.

        Parameters
        ----------
        transform : callable, optional
            A function that accepts two arguments: `(param_name, param_value)`.
            This function is applied to both the posterior draws and the
            reference parameter draws before computing the rank. For instance,
            it can be used to take the mean over a vectorized parameter grouping.
            If None, defaults to the `transform` passed during class
            initialization.

        Returns
        -------
        xarray.DataTree
            An xarray.DataTree containing the computed rank statistics, matching
            the output structure generated by `run_simulations`.
        """
        if not self.keep_fits:
            raise ValueError("calling `compute_rank_statistics` requires `keep_fits` to be True")
        if transform is None:
            transform = self._transform
        elif not callable(transform):
            raise ValueError("`transform` should be a function or None")

        self.simulations = {name: [] for name in self.kept_simulation_params.var_names}

        for idx in range(self.posteriors.sizes["simulation"]):
            posterior = self.posteriors.isel(simulation=idx)
            self._compute_single_rank(idx, posterior, transform, self.kept_simulation_params)

        self.simulations = {k: np.stack(v)[None, :] for k, v in self.simulations.items()}
        self._convert_to_datatree()
        return self.simulations

    def _compute_single_rank(self, simulation_idx, posterior, transform, simulation_params):
        for name in simulation_params.var_names:
            self.simulations[name].append(
                self.adapter.compute_single_rank(
                    transform, name, posterior, simulation_idx, simulation_params.ref_params
                )
            )

    @quiet_logging("pymc", "pytensor.gof.compilelock", "bambi", "numpyro")
    def run_simulations(self):
        """Run all SBC iterations (Prior or Posterior SBC).

        For each iteration the method:

        1. Draws a reference parameter vector and a replicated dataset
           (from the prior / prior-predictive for Prior SBC, or from the
           original posterior / posterior-predictive for Posterior SBC).
        2. Fits the model to the (possibly augmented) replicated data.
        3. Computes the rank of the reference draw among the new
           (augmented) posterior draws.

        The results are stored in ``self.simulations`` as an ArviZ
        DataTree with group ``"prior_sbc"`` or ``"posterior_sbc"``.

        This method can be stopped and restarted on the same instance:
        you can keyboard-interrupt part way through, inspect the partial
        results, and then call ``run_simulations()`` again to continue.
        If a seed was passed at init, reproducibility is preserved.

        If an error occurs during a simulation, it is logged (with its
        traceback) rather than raised, and the run finalizes with the
        rank statistics of the iterations completed so far.
        """
        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
            disable=not self.progress_bar,
        )

        if self.method == "prior":
            # In Prior SBC, the reference parameter draws are from the prior,
            # the predictive samples are from the prior predictive
            ref_params, predictive = self.adapter.get_prior_predictive_samples(
                self.num_simulations, self._seeds
            )
        else:
            # In Posterior SBC, the reference parameter draws are from the original posterior,
            # the predictive samples are from the original posterior predictive
            ref_params, predictive = self.adapter.get_posterior_predictive_samples(
                self.num_simulations, self._seeds, self.progress_bar
            )

        ref_params, predictive = self.adapter.subsample(
            ref_params, predictive, self.seed, self.num_simulations
        )

        if self.simulator is not None:
            # if simulator is used, ignore observed_vars
            simulation_params = self.adapter.simulation_params_from_simulator(
                ref_params, predictive
            )
            self.simulations = {var_name: [] for var_name in simulation_params.var_names}
        else:
            simulation_params = self.adapter.simulation_params_no_simulator(ref_params, predictive)

        self._simulation_loop(progress, simulation_params, predictive)

    def _simulation_loop(self, progress, simulation_params, predictive):
        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete

                replicated_data = self.adapter.replicate(predictive, idx, simulation_params)
                posterior = self.adapter.get_posterior_samples(
                    simulation_params,
                    replicated_data,
                    self.sample_kwargs,
                    self._seeds[self._simulations_complete],
                    self.method,
                    self._simulations_complete,
                )
                if self.keep_fits:
                    posterior = posterior.expand_dims({"simulation": [idx]})
                    if self.posteriors is None:
                        self.posteriors = posterior
                    else:
                        self.posteriors = xr.concat([self.posteriors, posterior], dim="simulation")

                    self.kept_simulation_params = simulation_params
                else:
                    self._compute_single_rank(idx, posterior, self._transform, simulation_params)

                self._simulations_complete += 1
                progress.update()
        except Exception:
            _log.exception("Stopping simulation. An error occurred during simulations:")
        finally:
            if self._simulations_complete:
                if self.keep_fits:
                    self.compute_rank_statistics()
                else:
                    self.simulations = {
                        k: np.stack(v)[None, :] for k, v in self.simulations.items()
                    }
                    self._convert_to_datatree()

            progress.close()
