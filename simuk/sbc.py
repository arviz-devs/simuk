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
    simulator : callable
        A custom simulator function that takes as input the model parameters and
        a int parameter named `seed`, and must return a dictionary of named observations.
    param_transform : callable, optional
        A transform applied to both the reference draw and the posterior
        draws before computing the rank statistic. Signature:
        ``(param_name, param_value) -> transformed_value``.
        Useful for defining scalar test quantities (e.g.
        ``lambda param_name, param_value: np.mean(param_value)`` to test the mean
        of a vector parameter). The return values must be comparable with the ``<``
        operator. The default is the identity (rank on the raw parameter values).

    Example
    -------

    .. code-block :: python

        with pm.Model() as model:
            x = pm.Normal('x')
            y = pm.Normal('y', mu=2 * x, observed=obs)

        sbc = SBC(model)
        sbc.run_simulations()
    """

    def __init__(
        self,
        model,
        num_simulations=1000,
        sample_kwargs=None,
        seed=None,
        data_dir=None,
        simulator=None,
        param_transform=None,
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
        self.posteriors = []
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
                "There are no observed variables, and PyMC will not generate prior "
                "predictive samples. Either change the model or specify a simulator "
                "with the `simulator` argument."
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

        self._param_transform = lambda param_name, param_value: param_value
        if param_transform is not None:
            if not callable(param_transform):
                raise ValueError("`param_transform` should be a function or None")
            self._param_transform = param_transform

    def _extract_variable_names(self):
        """Extract observed and free variables from the model."""
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
                draws=self.num_simulations, random_seed=self._seeds[0]
            )
            prior = extract(idata, group="prior", keep_dataset=True)
            if self.simulator is None:
                prior_pred = extract(idata, group="prior_predictive", keep_dataset=True)
                return prior, prior_pred
            # Deal with custom simulator
            prior_pred = []
            for i in range(prior.sizes["sample"]):
                params = {var: prior[var].isel(sample=i).values for var in prior.data_vars}
                params["seed"] = self._seeds[i]
                try:
                    res = self.simulator(**params)
                    assert isinstance(res, Mapping), (
                        f"Simulator must return a dictionary, got {type(res)}"
                    )
                    prior_pred.append(res)
                except Exception as e:
                    raise ValueError(
                        f"Error generating prior predictive sample with parameters {params}: {e}."
                    )
            prior_pred = dict_to_dataset(
                {key: np.stack([pp[key] for pp in prior_pred]) for key in prior_pred[0]},
                sample_dims=["sample"],
                coords={**prior.coords},
            )
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

    def _get_posterior_samples(self, prior_predictive_draw):
        """Generate posterior samples conditioned to a prior predictive sample."""
        new_model = pm.observe(self.model, prior_predictive_draw)
        with new_model:
            check = pm.sample(
                **self.sample_kwargs,
                random_seed=self._seeds[self._simulations_complete],
            )

        posterior = extract(check, group="posterior", keep_dataset=True)
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
    def compute_rank_statistics(self, param_transform=None):
        """Compute the rank statistic for the reference parameters.

        This method computes the rank of each reference parameter value
        relative to the newly sampled posterior draws for each simulation.

        This allows users to recompute rank statistics rapidly using a
        different parameter transformation without needing to rerun the simulations.

        Parameters
        ----------
        param_transform : callable, optional
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
        if param_transform is None:
            param_transform = self._param_transform
        elif not callable(param_transform):
            raise ValueError("`param_transform` should be a function or None")

        simulations = {name: [] for name in self.var_names}

        for idx, posterior in enumerate(self.posteriors):
            for name in self.var_names:
                if self.engine == "numpyro":
                    transformed_posterior = np.array(
                        [
                            param_transform(name, posterior[name].sel(chain=0).isel(draw=i).values)
                            for i in range(posterior[name].sizes["draw"])
                        ]
                    )
                    simulations[name].append(
                        (
                            transformed_posterior
                            < param_transform(name, self.ref_params[name][idx])
                        ).sum(axis=0)
                    )
                else:
                    transformed_posterior = np.array(
                        [
                            param_transform(name, posterior[name].isel(sample=i).values)
                            for i in range(posterior[name].sizes["sample"])
                        ]
                    )
                    simulations[name].append(
                        (
                            transformed_posterior
                            < param_transform(name, self.ref_params[name].isel(sample=idx).values)
                        ).sum(axis=0)
                    )

        self.simulations = {
            k: np.stack(v)[None, :]
            for k, v in simulations.items()
        }
        self._convert_to_datatree()
        return self.simulations

    @quiet_logging("pymc", "pytensor.gof.compilelock", "bambi")
    def run_simulations(self):
        """Run all the simulations.

        This function can be stopped and restarted on the same instance, so you can
        keyboard interrupt part way through, look at the plot, and then resume. If a
        seed was passed initially, it will still be respected (that is, the resulting
        simulations will be identical to running without pausing in the middle).
        """
        prior, prior_pred = self._get_prior_predictive_samples()
        self.ref_params = prior

        progress = tqdm(
            initial=self._simulations_complete,
            total=self.num_simulations,
        )

        # if simulator is used, ignore observed_vars
        if self.simulator is not None:
            self.observed_vars = list(prior_pred.data_vars)
            self.var_names = list(
                filter(
                    lambda var_name: var_name not in self.observed_vars,
                    list(prior.data_vars),
                )
            )
            self.simulations = {var_name: [] for var_name in self.var_names}

        try:
            while self._simulations_complete < self.num_simulations:
                idx = self._simulations_complete
                prior_predictive_draw = {
                    var_name: prior_pred[var_name].sel(chain=0, draw=idx).values
                    for var_name in self.observed_vars
                }

                posterior = self._get_posterior_samples(prior_predictive_draw)
                self.posteriors.append(posterior)

                self._simulations_complete += 1
                progress.update()
        finally:
            if self._simulations_complete:
                self.compute_rank_statistics()
            
            progress.close()

    @quiet_logging("numpyro")
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
                self.posteriors.append(posterior)

                self._simulations_complete += 1
                progress.update()
        finally:
            if self._simulations_complete > 0:
                self.compute_rank_statistics()
            progress.close()
