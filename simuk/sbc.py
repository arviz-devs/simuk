"""Simulation based calibration (Talts et. al. 2018) in PyMC."""

import logging
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

import numpy as np
from arviz_base import extract, from_dict, from_numpyro
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
    """Set up class for doing SBC.

    Parameters
    ----------
    model : pymc.Model, bambi.Model or numpyro.infer.mcmc.MCMCKernel
        A PyMC, Bambi model or Numpyro MCMC kernel. If a PyMC model the data needs to be defined as
        mutable data.
    num_simulations : int
        How many simulations to run
    sample_kwargs : dict[str] -> Any
        Arguments passed to pymc.sample or bambi.Model.fit
    seed : int (optional)
        Random seed. This persists even if running the simulations is
        paused for whatever reason.
    data_dir : dict
        Keyword arguments passed to numpyro model, intended for use when providing
        an MCMC Kernel model.

    Example
    -------

    .. code-block :: python

        with pm.Model() as model:
            x = pm.Normal('x')
            y = pm.Normal('y', mu=2 * x, observed=obs)

        sbc = SBC(model)
        sbc.run_simulations()
        sbc.plot_results()

    """

    def __init__(self, model, num_simulations=1000, sample_kwargs=None, seed=None, data_dir=None):
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
            self.data_dir = data_dir
        else:
            raise ValueError(
                "model should be one of pymc.Model, bambi.Model, or numpyro.infer.mcmc.MCMCKernel"
            )
        self.num_simulations = num_simulations
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
        self.seed = seed
        self._seeds = self._get_seeds()
        self._extract_variable_names()
        self.simulations = {name: [] for name in self.var_names}
        self._simulations_complete = 0

    def _extract_variable_names(self):
        """Extract observed and free variables from the model."""
        if self.engine == "numpyro":
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
        else:
            self.observed_vars = [obs.name for obs in self.model.observed_RVs]
            self.var_names = [v.name for v in self.model.free_RVs]

    def _get_seeds(self):
        """Set the random seed, and generate seeds for all the simulations."""
        rng = np.random.default_rng(self.seed)
        return rng.integers(0, 2**30, size=self.num_simulations)

    def _get_prior_predictive_samples(self):
        """Generate samples to use for the simulations."""
        with self.model:
            idata = pm.sample_prior_predictive(
                samples=self.num_simulations, random_seed=self._seeds[0]
            )
            prior_pred = extract(idata, group="prior_predictive", keep_dataset=True)
            prior = extract(idata, group="prior", keep_dataset=True)
        return prior, prior_pred

    def _get_prior_predictive_samples_numpyro(self):
        """Generate samples to use for the simulations using numpyro."""
        predictive = Predictive(self.model, num_samples=self.num_simulations)
        free_vars_data = {k: v for k, v in self.data_dir.items() if k not in self.observed_vars}
        samples = predictive(jax.random.PRNGKey(self._seeds[0]), **free_vars_data)
        prior = {k: v for k, v in samples.items() if k not in self.observed_vars}
        prior_pred = {k: v for k, v in samples.items() if k in self.observed_vars}
        return prior, prior_pred

    def _get_posterior_samples(self, prior_predictive_draw):
        """Generate posterior samples conditioned to a prior predictive sample."""
        new_model = pm.observe(self.model, prior_predictive_draw)
        with new_model:
            check = pm.sample(
                **self.sample_kwargs, random_seed=self._seeds[self._simulations_complete]
            )

        posterior = extract(check, group="posterior", keep_dataset=True)
        return posterior

    def _get_posterior_samples_numpyro(self, prior_predictive_draw):
        """Generate posterior samples using numpyro conditioned to a prior predictive sample."""
        mcmc = MCMC(self.numpyro_model, **self.sample_kwargs)
        rng_seed = jax.random.PRNGKey(self._seeds[self._simulations_complete])
        free_vars_data = {k: v for k, v in self.data_dir.items() if k not in self.observed_vars}
        mcmc.run(rng_seed, **free_vars_data, **prior_predictive_draw)
        return from_numpyro(mcmc)["posterior"]

    def _convert_to_datatree(self):
        self.simulations = from_dict(
            {"prior_sbc": self.simulations},
            attrs={
                "/": {
                    "inferece_library": self.engine,
                    "inferece_library_version": version(self.engine),
                    "modeling_interface": "simuk",
                    "modeling_interface_version": version("simuk"),
                }
            },
        )

    @quiet_logging("pymc", "pytensor.gof.compilelock", "bambi")
    def run_simulations(self):
        """Run all the simulations.

        This function can be stopped and restarted on the same instance, so you can
        keyboard interrupt part way through, look at the plot, and then resume. If a
        seed was passed initially, it will still be respected (that is, the resulting
        simulations will be identical to running without pausing in the middle).
        """
        prior, prior_pred = self._get_prior_predictive_samples()

        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
        )
        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete
                prior_predictive_draw = {
                    var_name: prior_pred[var_name].sel(chain=0, draw=idx).values
                    for var_name in self.observed_vars
                }

                posterior = self._get_posterior_samples(prior_predictive_draw)
                for name in self.var_names:
                    self.simulations[name].append(
                        (posterior[name] < prior[name].sel(chain=0, draw=idx)).sum("sample").values
                    )
                self._simulations_complete += 1
                progress.update()
        finally:
            self.simulations = {
                k: np.stack(v[: self._simulations_complete])[None, :]
                for k, v in self.simulations.items()
            }
            self._convert_to_datatree()
            progress.close()

    @quiet_logging("numpyro")
    def _run_simulations_numpyro(self):
        """Run all the simulations for Numpyro Model."""
        prior, prior_pred = self._get_prior_predictive_samples_numpyro()
        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
        )
        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete
                prior_predictive_draw = {k: v[idx] for k, v in prior_pred.items()}
                posterior = self._get_posterior_samples_numpyro(prior_predictive_draw)
                for name in self.var_names:
                    self.simulations[name].append(
                        (posterior[name].sel(chain=0) < prior[name][idx]).sum(axis=0).values
                    )
                self._simulations_complete += 1
                progress.update()
        finally:
            self.simulations = {
                k: np.stack(v[: self._simulations_complete])[None, :]
                for k, v in self.simulations.items()
            }
            self._convert_to_datatree()
            progress.close()
