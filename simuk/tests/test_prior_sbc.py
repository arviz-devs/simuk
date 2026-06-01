import bambi as bmb
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pymc as pm
import pytest
from numba import njit
from numpyro.infer import NUTS

import simuk

default_rng = np.random.default_rng(1234)

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
    y_obs = pm.Normal("y", mu=theta, sigma=sigma)

# Bambi model
x = default_rng.normal(0, 1, 20)
y = 2 + default_rng.normal(x, 1)
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


def numpyro_model_double_observed(y1=jnp.array([0.0]), y2=jnp.array([0.0])):
    numpyro.sample("y1", dist.Normal(0, 1), obs=y1)
    numpyro.sample("y2", dist.Normal(0, 1), obs=y2)


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


def test_sbc_numpyro_missing_observed_data():
    with pytest.raises(ValueError, match="missing from data_dir"):
        simuk.SBC(
            NUTS(numpyro_model_double_observed),
            data_dir={"y2": [0.0]},
            num_simulations=10,
            sample_kwargs={"num_warmup": 10, "num_samples": 5},
        )


def test_sbc_numpyro_empty_observed_data():
    with pytest.raises(ValueError, match="no observed variables"):
        simuk.SBC(
            NUTS(eight_schools_cauchy_prior),
            data_dir={"J": 8, "sigma": sigma},
            num_simulations=10,
            sample_kwargs={"num_warmup": 10, "num_samples": 5},
        )


def test_sbc_numpyro_simulator_no_conditionable_observed():
    def bad_simulator(**kwargs):
        return {"not_a_param": np.array([0.0])}

    sbc = simuk.SBC(
        NUTS(eight_schools_cauchy_prior),
        data_dir={"J": 8, "sigma": sigma, "y": data},
        num_simulations=5,
        sample_kwargs={"num_warmup": 10, "num_samples": 5},
        simulator=bad_simulator,
    )
    with pytest.raises(ValueError, match="No observed variables to condition on"):
        sbc.run_simulations()


# --- Initialization and transform validation tests ---
def test_sbc_invalid_model_type():
    with pytest.raises(ValueError, match="model should be one of"):
        simuk.SBC(object())


def test_sbc_simulator_not_callable():
    with pytest.raises(ValueError, match="simulator should be a function or None"):
        simuk.SBC(centered_eight, simulator=123)


def test_sbc_transform_not_callable_init():
    with pytest.raises(ValueError, match="`transform` should be a function or None"):
        simuk.SBC(centered_eight, transform="not callable")


def test_compute_rank_statistics_requires_keep_fits():
    sbc = simuk.SBC(
        centered_eight,
        num_simulations=1,
        sample_kwargs={"draws": 5, "tune": 5},
        keep_fits=False,
    )
    with pytest.raises(ValueError, match="requires `keep_fits` to be True"):
        sbc.compute_rank_statistics()


def test_compute_rank_statistics_transform_not_callable():
    sbc = simuk.SBC(
        centered_eight,
        num_simulations=1,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    with pytest.raises(ValueError, match="`transform` should be a function or None"):
        sbc.compute_rank_statistics(transform=123)


def test_sbc_run_simulations_keep_fits_false():
    sbc = simuk.SBC(
        centered_eight,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
        keep_fits=False,
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations


def test_sbc_numpyro_run_simulations_keep_fits_false():
    sbc = simuk.SBC(
        NUTS(eight_schools_cauchy_prior),
        data_dir={"J": 8, "sigma": sigma, "y": data},
        num_simulations=2,
        sample_kwargs={"num_warmup": 10, "num_samples": 5},
        keep_fits=False,
    )
    sbc.run_simulations()
    assert "prior_sbc" in sbc.simulations
