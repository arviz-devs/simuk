"""Tests for Posterior SBC (method='posterior')."""

import arviz as az
import jax
import numpy as np
import numpyro as npo
import numpyro.distributions as dist
import pymc as pm
import pytest
from numpyro.infer import MCMC, NUTS

default_rng = np.random.default_rng(1234)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

obs_data = default_rng.normal(2.0, 1.0, size=20)  # (20,)
x_obs = np.linspace(0, 1, 20)  # (20,)
y_obs_reg = 1.5 * x_obs + default_rng.normal(0, 0.5, size=20)  # (20,)

# ---------------------------------------------------------------------------
# PyMC models and traces
# ---------------------------------------------------------------------------

with pm.Model() as simple_model:
    mu = pm.Normal("mu", mu=0, sigma=5)
    sigma = pm.HalfNormal("sigma", sigma=2)
    y_data = pm.Data("y_data", obs_data)
    pm.Normal("y", mu=mu, sigma=sigma, observed=y_data)

with simple_model:
    trace_simple = pm.sample(
        draws=30,
        tune=30,
        chains=1,
        random_seed=123,
        progressbar=False,
        compute_convergence_checks=False,
    )

coords = {"obs_id": np.arange(len(y_obs_reg))}
with pm.Model(coords=coords) as reg_model:
    x = pm.Data("x", x_obs, dims="obs_id")
    y_data = pm.Data("y_data", y_obs_reg, dims="obs_id")
    slope = pm.Normal("slope", mu=0, sigma=5)
    sigma_reg = pm.HalfNormal("sigma", sigma=2)
    pm.Normal("y", mu=slope * x, sigma=sigma_reg, observed=y_data, dims="obs_id")

with reg_model:
    trace_reg = pm.sample(
        draws=30,
        tune=30,
        chains=1,
        random_seed=123,
        progressbar=False,
        compute_convergence_checks=False,
    )


# ---------------------------------------------------------------------------
# Custom simulator and callback functions
# ---------------------------------------------------------------------------


def custom_simulator(mu, sigma, seed, **kwargs):
    rng = np.random.default_rng(seed)
    return {"y": rng.normal(mu, sigma, size=20)}


def custom_augment_observed(model, observed_data, replicated_data, idx):
    # Custom: only keep the last 10 original obs + all replicated
    return {
        var: np.concatenate([observed_data[var].values[-10:], replicated_data[var]])
        for var in replicated_data
    }


def keep_only_replicated_data_augment(model, observed_data, replicated_data, idx):
    # Custom: keep only the replicated data
    return {var: replicated_data[var] for var in replicated_data}


def update_data_reg(model, augmented_data, idx):
    """Resize covariates and coords to match augmented data."""
    n_aug = len(augmented_data["y"])
    x_aug = np.tile(x_obs, n_aug // len(x_obs) + 1)[:n_aug]
    pm.set_data(
        {"x": x_aug, "y_data": augmented_data["y"]},
        coords={"obs_id": np.arange(n_aug)},
    )


def custom_transform(param_name, param_value):
    return param_value**2


def simple_model_numpyro(y=None):
    """
    analogous to simple_model, but implemented
    using Numpyro
    """

    mu = npo.sample("mu", dist.Normal(0, 5))
    sigma = npo.sample("sigma", dist.HalfNormal(2))
    y = npo.sample("y", dist.Normal(mu, sigma).expand([obs_data.shape[0]]), obs=y)


@pytest.fixture(scope="session")
def numpyro_nuts_kernel():
    yield NUTS(simple_model_numpyro)


@pytest.fixture(scope="session")
def simple_model_numpyro_trace(numpyro_nuts_kernel):

    rng_key = jax.random.PRNGKey(0)

    mcmc = MCMC(numpyro_nuts_kernel, num_warmup=100, num_samples=30)

    mcmc.run(rng_key=rng_key, y=obs_data)

    trace = mcmc.get_samples(group_by_chain=True)
    simple_model_numpyro_trace = az.from_dict(
        {
            "posterior": trace,
            # "observed_data": {"y": np.expand_dims(obs_data, 0)},
            "observed_data": {"y": obs_data},
        },
        dims={"mu": ["chain", "draw"], "y": ["observation_idx"]},
        coords={"y": x_obs},
        sample_dims=["chain", "draw"],
    )

    yield simple_model_numpyro_trace
