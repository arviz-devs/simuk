"""Simulation based calibration (Talts et. al. 2018) in PyMC."""

import logging
from copy import copy
from importlib.metadata import version

import numpy as np
import pymc as pm
from arviz_base import extract, from_dict
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
    model : function
        A PyMC or Bambi model. If a PyMC model the data needs to be defined as
        mutable data.
    num_simulations : int
        How many simulations to run
    sample_kwargs : dict[str] -> Any
        Arguments passed to pymc.sample or bambi.Model.fit
    seed : int (optional)
        Random seed. This persists even if running the simulations is
        paused for whatever reason.

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

    def __init__(
        self,
        model,
        num_simulations=1000,
        sample_kwargs=None,
        seed=None,
    ):
        if isinstance(model, pm.Model):
            self.engine = "pymc"
            self.model = model
        else:
            self.engine = "bambi"
            model.build()
            self.bambi_model = model
            self.model = model.backend.model
            self.formula = model.formula
            self.new_data = copy(model.data)

        self.observed_vars = [obs_rvs.name for obs_rvs in self.model.observed_RVs]
        self.num_simulations = num_simulations

        self.var_names = [v.name for v in self.model.free_RVs]

        if sample_kwargs is None:
            sample_kwargs = {}
        sample_kwargs.setdefault("progressbar", False)
        sample_kwargs.setdefault("compute_convergence_checks", False)
        self.sample_kwargs = sample_kwargs

        self.simulations = {name: [] for name in self.var_names}
        self._simulations_complete = 0
        self.seed = seed
        self._seeds = self._get_seeds()

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

    def _get_posterior_samples(self, prior_predictive_draw):
        """Generate posterior samples conditioned to a prior predictive sample."""
        new_model = pm.observe(self.model, prior_predictive_draw)
        with new_model:
            check = pm.sample(
                **self.sample_kwargs, random_seed=self._seeds[self._simulations_complete]
            )

        posterior = extract(check, group="posterior", keep_dataset=True)
        return posterior

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
