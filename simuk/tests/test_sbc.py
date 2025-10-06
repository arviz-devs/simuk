import bambi as bmb
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pymc as pm
import pytest
from jax import random
from numba import njit
from numpyro.infer import MCMC, NUTS

import simuk

np.random.seed(1234)

# Test data
data = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])

# PyMC models
with pm.Model() as centered_eight:
    mu = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
    y_obs = pm.Normal("y", mu=theta, sigma=sigma, observed=data)

with pm.Model() as centered_eight_no_observed:
    mu = pm.Normal("mu", mu=0, sigma=5)
    tau = pm.HalfCauchy("tau", beta=5)
    theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)

    def log_likelihood(theta, observed):
        return pm.math.sum(pm.logp(pm.Normal.dist(mu=theta, sigma=sigma), observed))

    pm.Potential("y_loglike", log_likelihood(mu, data))

# Bambi model
x = np.random.normal(0, 1, 20)
y = 2 + np.random.normal(x, 1)
df = pd.DataFrame({"x": x, "y": y})
bmb_model = bmb.Model("y ~ x", df)


# NumPyro models
def eight_schools_cauchy_prior(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
    numpyro.sample("y", dist.Normal(theta, sigma), obs=y)


def eight_schools_cauchy_prior_no_observed(J, sigma, y=None):
    mu = numpyro.sample("mu", dist.Normal(0, 5))
    tau = numpyro.sample("tau", dist.HalfCauchy(5))
    with numpyro.plate("J", J):
        theta = numpyro.sample("theta", dist.Normal(mu, tau))
    if y is not None:
        log_likelihood = jnp.sum(dist.Normal(theta, sigma).log_prob(y))
        numpyro.factor("custom_likelihood", log_likelihood)


# Custom simulator functions
def centered_eight_simulator(theta, seed, **kwargs):
    rng = np.random.default_rng(seed)
    return {"y": rng.normal(theta, sigma)}


@njit
def centered_eight_jitted_simulator(tau, mu, theta, seed):
    # Some expensive computation
    n = theta.shape[0]
    y = np.zeros(n)
    for i in range(n):
        y[i] = theta[i]
    return {"y": y}


def bmb_simulator(mu, sigma, seed, **kwargs):
    rng = np.random.default_rng(seed)
    return {"y": rng.normal(mu, sigma)}


# --- Tests with observed variables ---
@pytest.mark.parametrize("model", [centered_eight, bmb_model])
def test_sbc_with_observed_data(model):
    sbc = simuk.SBC(
        model,
        num_simulations=10,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


def test_sbc_numpyro_with_observed_data():
    sbc = simuk.SBC(
        NUTS(eight_schools_cauchy_prior),
        data_dir={"J": 8, "sigma": sigma, "y": data},
        num_simulations=10,
        sample_kwargs={"num_warmup": 50, "num_samples": 25},
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


# --- Tests with custom simulators ---
@pytest.mark.parametrize(
    "model,simulator",
    [
        # Case 1: Both simulator function and observed variables present
        (centered_eight, centered_eight_simulator),
        # Case 2: Only simulator function present
        (centered_eight_no_observed, centered_eight_simulator),
    ],
)
def test_sbc_with_custom_simulator(model, simulator):
    sbc = simuk.SBC(
        model, num_simulations=10, sample_kwargs={"draws": 5, "tune": 5}, simulator=simulator
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


@pytest.mark.skipif(
    hasattr(bmb, "__version__") and tuple(map(int, bmb.__version__.split("."))) <= (0, 14),
    reason="requires bambi version > 0.14",
)
def test_sbc_bambi_with_custom_simulator():
    sbc = simuk.SBC(
        bmb_model,
        num_simulations=10,
        sample_kwargs={"draws": 5, "tune": 5},
        simulator=bmb_simulator,
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


@pytest.mark.parametrize(
    "model,simulator",
    [
        # Case 1: Both simulator function and observed variables present
        (eight_schools_cauchy_prior, centered_eight_simulator),
        # Case 2: Only simulator function present
        (eight_schools_cauchy_prior_no_observed, centered_eight_simulator),
    ],
)
def test_sbc_numpyro_with_custom_simulator(model, simulator):
    sbc = simuk.SBC(
        NUTS(model),
        data_dir={"J": 8, "sigma": sigma, "y": data},
        num_simulations=10,
        sample_kwargs={"num_warmup": 50, "num_samples": 25},
        simulator=simulator,
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


# --- Error handling tests with custom simulators ---
def test_sbc_fail_no_observed_variable():
    with pytest.raises(ValueError, match="no observed variables"):
        simuk.SBC(
            centered_eight_no_observed,
            num_simulations=10,
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_sbc_numpyro_fail_no_observed_variable():
    # Note: factor variables are catalogued as 'observed_vars' in NumPyro
    # therefore, we cannot raise an early exception with an informative message
    with pytest.raises(ValueError):
        sbc = simuk.SBC(
            NUTS(eight_schools_cauchy_prior_no_observed),
            data_dir={"J": 8, "sigma": sigma, "y": data},
            num_simulations=10,
            sample_kwargs={"num_warmup": 50, "num_samples": 25},
        )
        sbc.run_simulations()


# Test posterior SBC
def test_posterior_sbc_pymc_with_observed_variables():
    with centered_eight:
        trace = pm.sample(draws=100, tune=100, chains=4)
    sbc = simuk.SBC(
        centered_eight,
        trace=trace,
        num_simulations=10,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


def test_posterior_sbc_pymc_with_custom_simulator():
    with centered_eight:
        trace = pm.sample(draws=100, tune=100, chains=4)
    sbc = simuk.SBC(
        centered_eight,
        trace=trace,
        num_simulations=10,
        simulator=centered_eight_simulator,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


def test_posterior_sbc_bambi_with_observed_variables():
    trace = bmb_model.fit(num_samples=25, tune=50)
    sbc = simuk.SBC(
        bmb_model,
        trace=trace,
        num_simulations=10,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


@pytest.mark.skipif(
    hasattr(bmb, "__version__") and tuple(map(int, bmb.__version__.split("."))) <= (0, 14),
    reason="requires bambi version > 0.14",
)
def test_posterior_sbc_bambi_with_custom_simulator():
    # TODO: The names of the parameters drawn using `bmb_model.prior_predictive`
    # or `bmb_model.fit` are different from the ones you get if you access the
    # `pymc` backend model. Eventually, we should decide how to handle this.
    bmb_model.build()
    model = bmb_model.backend.model
    with model:
        trace = pm.sample(draws=100, tune=100, chains=4)
    sbc = simuk.SBC(
        model,
        trace=trace,
        num_simulations=10,
        sample_kwargs={"draws": 5, "tune": 5},
        simulator=bmb_simulator,
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


def test_posterior_sbc_numpyro_with_observed_variables():
    nuts_kernel = NUTS(eight_schools_cauchy_prior)
    mcmc = MCMC(nuts_kernel, num_warmup=25, num_samples=50)
    data_dir = {"J": 8, "sigma": sigma, "y": data}
    mcmc.run(random.PRNGKey(0), **data_dir)
    sbc = simuk.SBC(
        nuts_kernel,
        trace=mcmc,
        num_simulations=10,
        sample_kwargs={"num_warmup": 50, "num_samples": 25},
        data_dir=data_dir,
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


def test_posterior_sbc_numpyro_with_custom_simulator():
    nuts_kernel = NUTS(eight_schools_cauchy_prior)
    mcmc = MCMC(nuts_kernel, num_warmup=25, num_samples=50)
    data_dir = {"J": 8, "sigma": sigma, "y": data}
    mcmc.run(random.PRNGKey(0), **data_dir)
    sbc = simuk.SBC(
        nuts_kernel,
        trace=mcmc,
        num_simulations=10,
        sample_kwargs={"num_warmup": 50, "num_samples": 25},
        data_dir=data_dir,
        simulator=centered_eight_simulator,
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


# --- Error handling tests posterior SBC ---
def test_posterior_sbc_fail_if_not_enough_samples():
    with pytest.raises(ValueError, match="does not contain enough samples"):
        with centered_eight:
            # Trace is too short
            trace = pm.sample(draws=10, tune=10, chains=4)
            _ = simuk.SBC(
                centered_eight,
                trace=trace,
                num_simulations=1000,
                sample_kwargs={"draws": 5, "tune": 5},
            )


def test_posterior_sbc_fail_if_wrong_trace():
    with pytest.raises(ValueError, match="does not contain a `posterior`"):
        with centered_eight:
            _ = simuk.SBC(
                centered_eight,
                trace={},
                num_simulations=100,
            )
