"""Tests for Posterior SBC (method='posterior')."""

import logging

import numpy as np
import pymc as pm
import pytest

import simuk

np.random.seed(42)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

obs_data = np.random.normal(2.0, 1.0, size=20)
x_obs = np.linspace(0, 1, 20)
y_obs_reg = 1.5 * x_obs + np.random.normal(0, 0.5, size=20)

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


def update_data_reg(model, augmented_data, idx):
    """Resize covariates and coords to match augmented data."""
    n_aug = len(augmented_data["y"])
    x_aug = np.tile(x_obs, n_aug // len(x_obs) + 1)[:n_aug]
    pm.set_data(
        {"x": x_aug, "y_data": augmented_data["y"]},
        coords={"obs_id": np.arange(n_aug)},
    )


def custom_param_transform(param_name, param_value):
    return param_value**2


# ---------------------------------------------------------------------------
# Tests with observed variables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model,trace", [(simple_model, trace_simple)])
def test_posterior_sbc_with_observed_data(model, trace):
    """Basic posterior SBC with a PyMC model."""
    sbc = simuk.SBC(
        model,
        method="posterior",
        trace=trace,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    assert "posterior_sbc" in sbc.simulations


@pytest.mark.parametrize(
    "model,trace,update_data", [(reg_model, trace_reg, update_data_reg)]
)
def test_posterior_sbc_with_update_data(model, trace, update_data):
    """Posterior SBC with dims/coords and update_data callback."""
    sbc = simuk.SBC(
        model,
        method="posterior",
        trace=trace,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
        update_data=update_data,
    )
    sbc.run_simulations()
    assert "posterior_sbc" in sbc.simulations


# ---------------------------------------------------------------------------
# Tests with custom simulator and callbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model,trace,simulator", [(simple_model, trace_simple, custom_simulator)]
)
def test_posterior_sbc_with_custom_simulator(model, trace, simulator):
    """Posterior SBC using a custom simulator function."""
    sbc = simuk.SBC(
        model,
        method="posterior",
        trace=trace,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
        simulator=simulator,
    )
    sbc.run_simulations()
    assert "posterior_sbc" in sbc.simulations


@pytest.mark.parametrize(
    "model,trace,augment_observed",
    [(simple_model, trace_simple, custom_augment_observed)],
)
def test_posterior_sbc_with_augment_observed(model, trace, augment_observed):
    """Posterior SBC with a custom augment_observed callback."""
    sbc = simuk.SBC(
        model,
        method="posterior",
        trace=trace,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
        augment_observed=augment_observed,
    )
    sbc.run_simulations()
    assert "posterior_sbc" in sbc.simulations


@pytest.mark.parametrize(
    "model,trace,param_transform",
    [(simple_model, trace_simple, custom_param_transform)],
)
def test_posterior_sbc_with_param_transform(model, trace, param_transform):
    """Posterior SBC with a param_transform(name, value) function."""
    sbc = simuk.SBC(
        model,
        method="posterior",
        trace=trace,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
        param_transform=param_transform,
    )
    sbc.run_simulations()
    assert "posterior_sbc" in sbc.simulations


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------


def test_posterior_sbc_no_trace():
    """method='posterior' without trace should raise ValueError."""
    with pytest.raises(ValueError, match="posterior samples from the"):
        simuk.SBC(
            simple_model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_posterior_sbc_trace_missing_posterior():
    """trace without 'posterior' group should raise ValueError."""
    trace_missing = trace_simple.copy()
    del trace_missing.posterior
    with pytest.raises(ValueError, match="posterior"):
        simuk.SBC(
            simple_model,
            method="posterior",
            trace=trace_missing,
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_posterior_sbc_trace_missing_observed_data():
    """trace without 'observed_data' group should raise ValueError."""
    trace_missing = trace_simple.copy()
    del trace_missing.observed_data
    with pytest.raises(ValueError, match="observed_data"):
        simuk.SBC(
            simple_model,
            method="posterior",
            trace=trace_missing,
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_posterior_sbc_too_many_simulations():
    """num_simulations > draws should raise ValueError."""
    with pytest.raises(ValueError, match="more draws per"):
        simuk.SBC(
            simple_model,
            method="posterior",
            trace=trace_simple,
            num_simulations=100,  # trace_simple only has 30 draws
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_posterior_sbc_numpyro_not_implemented():
    """Posterior SBC is not yet implemented for NumPyro."""
    numpyro = pytest.importorskip("numpyro")
    import numpyro.distributions as dist
    from numpyro.infer import NUTS

    def numpyro_model(y=None):
        mu = numpyro.sample("mu", dist.Normal(0, 5))
        numpyro.sample("y", dist.Normal(mu, 1), obs=y)

    with pytest.raises(NotImplementedError, match="only implemented for PyMC"):
        simuk.SBC(
            NUTS(numpyro_model),
            method="posterior",
            trace=trace_simple,
            data_dir={"y": obs_data},
            num_simulations=5,
        )


def test_posterior_sbc_warnings_for_prior(caplog):
    """Passing posterior-only args with method='prior' should emit warnings."""
    with caplog.at_level(logging.WARNING):
        simuk.SBC(
            simple_model,
            method="prior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace_simple,
            augment_observed=lambda *a: {},
            update_data=lambda *a: None,
        )

    messages = caplog.text
    assert "update_data" in messages
    assert "augment_observed" in messages
    assert "trace" in messages
