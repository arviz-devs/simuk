import bambi as bmb
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pymc as pm
import pytest
from numpyro.infer import NUTS

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
def centered_eight_simulator(theta, **kwargs):
    return {"y": np.random.normal(theta, sigma)}


def bmb_simulator(mu, sigma, **kwargs):
    return {"y": np.random.normal(mu, sigma)}


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
        # Case 3: bambi model with custom simulator
        (bmb_model, bmb_simulator),
    ],
)
def test_sbc_with_custom_simulator(model, simulator):
    sbc = simuk.SBC(
        model, num_simulations=10, sample_kwargs={"draws": 5, "tune": 5}, simulator=simulator
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
