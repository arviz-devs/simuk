import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pymc as pm
import pytest

default_rng = np.random.default_rng(1234)


@pytest.fixture(scope="session")
def general_obs_data():
    y = np.array([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0])
    sigma = np.array([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0])
    return y, sigma


# ---------------------------------------------------------------------------
# PyMC fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def pm_simple_obs():
    return default_rng.normal(2.0, 1.0, size=20)


@pytest.fixture(scope="session")
def pm_centered_eight_model(general_obs_data):
    """Fixture for the centered eight schools model."""
    y, sigma = general_obs_data

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("y", mu=theta, sigma=sigma, observed=y)

    return model


@pytest.fixture(scope="session")
def pm_centered_eight_no_observed_model(general_obs_data):
    """Fixture for the centered eight schools model."""
    _, sigma = general_obs_data

    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        tau = pm.HalfCauchy("tau", beta=5)
        theta = pm.Normal("theta", mu=mu, sigma=tau, shape=8)
        pm.Normal("y", mu=theta, sigma=sigma)

    return model


@pytest.fixture(scope="session")
def pm_simple_model(pm_simple_obs):
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0, sigma=5)
        sigma = pm.HalfNormal("sigma", sigma=2)
        y_data = pm.Data("y_data", pm_simple_obs)
        pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

    return model


@pytest.fixture(scope="session")
def pm_simple_model_trace(pm_simple_model):
    with pm_simple_model:
        trace_simple = pm.sample(
            draws=30,
            tune=30,
            chains=1,
            random_seed=123,
            progressbar=False,
            compute_convergence_checks=False,
        )

    return trace_simple


# ---------------------------------------------------------------------------
# Numpyro fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def numpyro_eight_schools_cauchy_prior():
    def _numpyro_eight_schools_cauchy_prior(J, sigma, y=None):
        mu = numpyro.sample("mu", dist.Normal(0, 5))
        tau = numpyro.sample("tau", dist.HalfCauchy(5))
        with numpyro.plate("J", J):
            theta = numpyro.sample("theta", dist.Normal(mu, tau))
        numpyro.sample("y", dist.Normal(theta, sigma), obs=y)

    return _numpyro_eight_schools_cauchy_prior


@pytest.fixture(scope="session")
def numpyro_eight_schools_cauchy_prior_no_observed():
    def _numpyro_eight_schools_cauchy_prior_no_observed(J, sigma, y=None):
        mu = numpyro.sample("mu", dist.Normal(0, 5))
        tau = numpyro.sample("tau", dist.HalfCauchy(5))
        with numpyro.plate("J", J):
            theta = numpyro.sample("theta", dist.Normal(mu, tau))
        if y is not None:
            log_likelihood = jnp.sum(dist.Normal(theta, sigma).log_prob(y))
            numpyro.factor("custom_likelihood", log_likelihood)

    return _numpyro_eight_schools_cauchy_prior_no_observed


@pytest.fixture(scope="session")
def numpyro_eight_schools_cauchy_prior_data(general_obs_data):
    y, sigma = general_obs_data
    return {"J": 8, "sigma": sigma, "y": y}


@pytest.fixture(scope="session")
def numpyro_eight_schools_cauchy_prior_no_observed_data(general_obs_data):
    _, sigma = general_obs_data
    return {"J": 8, "sigma": sigma}
