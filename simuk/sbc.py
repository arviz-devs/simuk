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
import traceback
from copy import copy
from importlib.metadata import version

try:
    import pymc as pm
except ImportError:
    pass
try:
    import jax
    from numpyro.handlers import seed, trace
    from numpyro.infer import MCMC, Predictive
    from numpyro.infer.mcmc import MCMCKernel
except ImportError:
    pass

import inspect
from collections.abc import Mapping

import numpy as np
from arviz_base import dict_to_dataset, extract, from_dict, from_numpyro
from tqdm import tqdm


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
      from the prior predictive (Talts et al.,` 2020 [1]_).
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
        simulations again.

    Notes
    -----
    **Prior SBC** exploits the self-consistency of Bayesian updating:
    if :math:`\theta' \sim \pi(\theta)` and
    :math:`y' \sim \pi(y \mid \theta')`, then :math:`\theta'` is also
    a draw from :math:`\pi(\theta \mid y')`.  See Talts et al. (2020).

    **Posterior SBC** uses the same self-consistency after conditioning
    on observed data :math:`y_{\text{obs}}`.  A draw
    :math:`\theta'_i \sim \pi(\theta \mid y_{\text{obs}})` and a
    replicated dataset :math:`y_i \sim \pi(y \mid \theta'_i)` are
    combined so that :math:`\theta'_i` is also a draw from
    :math:`\pi(\theta \mid y_i, y_{\text{obs}})`.  The rank of
    :math:`\theta'_i` among augmented-posterior draws should be
    uniformly distributed if the inference is calibrated.
    See Säilynoja et al. (2025).

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
        if hasattr(model, "basic_RVs") and isinstance(model, pm.Model):
            self.engine = "pymc"
            self.model = model
        elif hasattr(model, "formula"):
            self.engine = "bambi"
            model.build()
            self.bambi_model = model
            self.model = model.backend.model
            self.formula = model.formula
            self.new_data = copy(model.data)
        elif isinstance(model, MCMCKernel):
            self.engine = "numpyro"
            self.numpyro_model = model
            self.model = self.numpyro_model.model
            self.run_simulations = self._run_simulations_numpyro
            self.data_dir = data_dir if data_dir is not None else {}
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

        self.num_simulations = num_simulations
        self.seed = seed
        self._seeds = self._get_seeds()

        self._extract_model_info()
        self.simulations = {name: [] for name in self.var_names}
        self._simulations_complete = 0
        self.posteriors = []
        self.keep_fits = keep_fits
        self.ref_params = None

        if simulator is not None and not callable(simulator):
            raise ValueError("simulator should be a function or None")
        if simulator is not None and self.observed_vars:
            logging.warning(
                "Provided model contains both observed variables and a simulator. "
                "Ignoring observed variables and using the simulator instead."
            )
        if simulator is None and not self.observed_vars and self.engine == "pymc":
            # Ideally, we could raise an error early for `numpyro` also,
            # but `factor` also produces 'observed_vars'
            raise ValueError(
                "There are no observed variables, and PyMC will not generate predictive "
                "samples for both Prior and Posterior SBC. Either change the model or "
                "specify a simulator with the `simulator` argument."
            )

        if simulator is None and self.engine == "numpyro":
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
        self.simulator = simulator

        self._transform = lambda param_name, param_value: param_value
        if transform is not None:
            if not callable(transform):
                raise ValueError("`param_transform` should be a function or None")
            self._transform = transform

        self.method = method.lower()
        if self.method == "posterior":
            if trace is None:
                raise ValueError(
                    "When performing Posterior SBC, posterior samples from the "
                    "original posterior are required to generate replicate datasets"
                )
            if "posterior" not in trace.groups():
                raise ValueError("`trace` should contain 'posterior' group")
            if "observed_data" not in trace.groups():
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

    def _extract_model_info(self):
        """Extract observed and free variables from the model.

        Also records the baseline state for Posterior SBC.
        """
        if self.engine == "numpyro":
            self.model_params = set(inspect.signature(self.model).parameters.keys())
            with trace() as tr:
                with seed(rng_seed=int(self._seeds[0])):
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

        else:
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

    def _reset_model_state(self, model, model_state):
        """Reset the state of PyMC model."""
        with model:
            pm.set_data(model_state["data"], coords=model_state["coords"])

    def _get_seeds(self):
        """Set the random seed, and generate seeds for all the simulations."""
        rng = np.random.default_rng(self.seed)
        return rng.integers(0, 2**30, size=self.num_simulations)

    def _get_simulator_data(self, free_rv_samples):
        """Run the user-defined simulator to obtain predictive samples.

        These samples can be generated from either prior or posterior samples.
        """
        # Deal with custom simulator
        pred = []
        for i in range(free_rv_samples.sizes["sample"]):
            params = {
                var: free_rv_samples[var].isel(sample=i).values for var in free_rv_samples.data_vars
            }
            params["seed"] = self._seeds[i]
            try:
                res = self.simulator(**params)
                assert isinstance(res, Mapping), (
                    f"Simulator must return a dictionary, got {type(res)}"
                )
                pred.append(res)
            except Exception as e:
                raise ValueError(
                    f"Error generating prior predictive sample with parameters {params}: {e}."
                )
        pred = dict_to_dataset(
            {key: np.stack([pp[key] for pp in pred]) for key in pred[0]},
            sample_dims=["sample"],
            coords={**free_rv_samples.coords},
        )

        return pred

    def _get_prior_predictive_samples(self):
        """Generate samples to use for the simulations."""
        with self.model:
            idata = pm.sample_prior_predictive(
                draws=self.num_simulations, random_seed=self._seeds[0]
            )
            prior = extract(idata, group="prior", keep_dataset=True)

            if self.simulator is None:
                prior_pred = extract(idata, group="prior_predictive", keep_dataset=True)
                return prior, prior_pred

            prior_pred = self._get_simulator_data(prior)

        return prior, prior_pred

    def _get_prior_predictive_samples_numpyro(self):
        """Generate samples to use for the simulations using numpyro."""
        predictive = Predictive(self.model, num_samples=self.num_simulations)
        free_vars_data = {
            k: v
            for k, v in self.data_dir.items()
            if k not in self.observed_vars and k in self.model_params
        }
        samples = predictive(jax.random.PRNGKey(self._seeds[0]), **free_vars_data)
        prior = {k: v for k, v in samples.items() if k not in self.observed_vars}
        if self.simulator:
            results = []
            for i, vals in enumerate(zip(*prior.values())):
                params = dict(zip(prior.keys(), vals))
                params["seed"] = self._seeds[i]
                results.append(self.simulator(**params))
            prior_pred = {key: [result[key] for result in results] for key in results[0]}
        else:
            prior_pred = {k: v for k, v in samples.items() if k in self.observed_model_vars}
        return prior, prior_pred

    def _get_posterior_samples(self, replicated_data):
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
        if self.method == "posterior":
            observed_data = self.trace["observed_data"]

            if self.augment_observed is not None:
                augmented_data = self.augment_observed(
                    self.model, observed_data, replicated_data, self._simulations_complete
                )
            else:
                # Default: concatenate original and replicated observations
                augmented_data = {
                    var_name: np.concatenate(
                        [observed_data[var_name].values, replicated_data[var_name]]
                    )
                    for var_name in self.observed_vars
                }

            if self.update_data is not None:
                with self.model:
                    self.update_data(self.model, augmented_data, self._simulations_complete)

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
                check = pm.sample(
                    **self.sample_kwargs, random_seed=self._seeds[self._simulations_complete]
                )

            posterior = extract(check, group="posterior", keep_dataset=True)
        except Exception:
            traceback.print_exc()
            raise
        finally:
            # Always ensure the model is reset to its un-augmented baseline state
            # so the next simulation iteration isn't corrupted by the previous loop's augmented data
            self._reset_model_state(self.model, self.model_baseline_state)

        return posterior

    def _get_posterior_samples_numpyro(self, prior_predictive_draw):
        """Generate posterior samples using numpyro conditioned to a prior predictive sample."""
        mcmc = MCMC(self.numpyro_model, **self.sample_kwargs)
        rng_seed = jax.random.PRNGKey(self._seeds[self._simulations_complete])

        free_vars_data = {
            k: v
            for k, v in self.data_dir.items()
            if k not in self.observed_model_vars and k in self.model_params
        }
        prior_predictive_args = {
            k: v for k, v in prior_predictive_draw.items() if k in self.observed_model_vars
        }
        mcmc.run(rng_seed, **free_vars_data, **prior_predictive_args)
        return from_numpyro(mcmc)["posterior"]

    def _get_posterior_predictive_samples(self):
        with self.model:
            num_draws = self.trace["posterior"].sizes["draw"]
            draw_indices = np.linspace(0, num_draws - 1, self.num_simulations, dtype=int)
            thinned_idata = self.trace.isel(draw=draw_indices)
            posterior = extract(thinned_idata, group="posterior", keep_dataset=True)

            if self.simulator is None:
                pm.sample_posterior_predictive(
                    thinned_idata,
                    extend_inferencedata=True,
                    random_seed=self._seeds[0],
                    progressbar=self.progress_bar,
                )
                posterior_pred = extract(
                    thinned_idata, group="posterior_predictive", keep_dataset=True
                )
                return posterior, posterior_pred
            else:
                posterior_pred = self._get_simulator_data(posterior)

            return posterior, posterior_pred

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
            If None, defaults to the `param_transform` passed during class
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

        self.simulations = {name: [] for name in self.var_names}

        for idx, posterior in enumerate(self.posteriors):
            self._compute_single_rank(idx, posterior, transform)

        self.simulations = {k: np.stack(v)[None, :] for k, v in self.simulations.items()}
        self._convert_to_datatree()
        return self.simulations

    def _compute_single_rank(self, simulation_idx, posterior, transform):
        for name in self.var_names:
            if self.engine == "numpyro":
                transformed_posterior = np.array(
                    [
                        transform(name, posterior[name].sel(chain=0).isel(draw=i).values)
                        for i in range(posterior[name].sizes["draw"])
                    ]
                )
                self.simulations[name].append(
                    (
                        transformed_posterior
                        < transform(name, self.ref_params[name][simulation_idx])
                    ).sum(axis=0)
                )
            elif self.engine in ["bambi", "pymc"]:
                transformed_posterior = np.array(
                    [
                        transform(name, posterior[name].isel(sample=i).values)
                        for i in range(posterior[name].sizes["sample"])
                    ]
                )
                self.simulations[name].append(
                    (
                        transformed_posterior
                        < transform(name, self.ref_params[name].isel(sample=simulation_idx).values)
                    ).sum(axis=0)
                )

    @quiet_logging("pymc", "pytensor.gof.compilelock", "bambi")
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
        """
        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
            disable=not self.progress_bar,
        )

        if self.method == "prior":
            # In Prior SBC, the reference parameter draws are from the prior,
            # the predictive samples are from the prior predictive
            ref_params, predictive = self._get_prior_predictive_samples()
        else:
            # In Posterior SBC, the reference parameter draws are from the original posterior,
            # the predictive samples are from the original posterior predictive
            ref_params, predictive = self._get_posterior_predictive_samples()

        rng = np.random.default_rng(self.seed)
        sample_indices = rng.choice(
            ref_params.sizes["sample"], size=self.num_simulations, replace=False
        )
        self.ref_params = ref_params.isel(sample=sample_indices)
        predictive = predictive.isel(sample=sample_indices)

        # if simulator is used, ignore observed_vars
        if self.simulator is not None:
            self.observed_vars = list(predictive.data_vars)
            self.var_names = list(
                filter(
                    lambda var_name: var_name not in self.observed_vars,
                    list(ref_params.data_vars),
                )
            )
            self.simulations = {var_name: [] for var_name in self.var_names}

        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete

                replicated_data = {
                    var_name: predictive[var_name].isel(sample=idx).values
                    for var_name in self.observed_vars
                }

                posterior = self._get_posterior_samples(replicated_data)
                if self.keep_fits:
                    self.posteriors.append(posterior)
                else:
                    self._compute_single_rank(idx, posterior, self._transform)

                self._simulations_complete += 1
                progress.update()
        except Exception:
            logging.error("Stopping simulation. An error occurred during simulations:")
            traceback.print_exc()
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

    @quiet_logging("numpyro")
    def _run_simulations_numpyro(self):
        """Run all the simulations for Numpyro Model."""
        prior, prior_pred = self._get_prior_predictive_samples_numpyro()
        self.ref_params = prior
        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
        )
        # if simulator is used, ignore observed_vars
        if self.simulator is not None:
            self.observed_vars = list(prior_pred.keys())
            self.observed_model_vars = [
                name for name in self.observed_vars if name in self.model_params
            ]
            if not self.observed_model_vars:
                raise ValueError("No observed variables to condition on")

            self.var_names = list(
                filter(
                    lambda var_name: var_name not in self.observed_vars,
                    list(prior.keys()),
                )
            )
            self.simulations = {var_name: [] for var_name in self.var_names}
        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete
                prior_predictive_draw = {k: v[idx] for k, v in prior_pred.items()}
                posterior = self._get_posterior_samples_numpyro(prior_predictive_draw)
                if self.keep_fits:
                    self.posteriors.append(posterior)
                else:
                    self._compute_single_rank(idx, posterior, self._transform)

                self._simulations_complete += 1
                progress.update()
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
