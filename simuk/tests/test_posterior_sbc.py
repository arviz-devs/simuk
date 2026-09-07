"""Tests for Posterior SBC (method='posterior')."""

import logging

import numpy as np
import numpyro.distributions as dist
import pymc as pm
import pytest
from numpyro.infer import NUTS

import simuk

default_rng = np.random.default_rng(1234)

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
x_obs_reg = np.linspace(0, 1, 20)
y_obs_reg = 1.5 * x_obs_reg + default_rng.normal(0, 0.5, size=20)


# ---------------------------------------------------------------------------
# PyMC models and traces
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def pm_reg_model():
    coords = {"obs_id": np.arange(len(y_obs_reg))}
    with pm.Model(coords=coords) as reg_model:
        x = pm.Data("x", x_obs_reg, dims="obs_id")
        y_data = pm.Data("y_data", y_obs_reg, dims="obs_id")
        slope = pm.Normal("slope", mu=0, sigma=5)
        sigma_reg = pm.HalfNormal("sigma", sigma=2)
        pm.Normal("y", mu=slope * x, sigma=sigma_reg, observed=y_data, dims="obs_id")
    return reg_model


@pytest.fixture(scope="module")
def pm_reg_model_trace(pm_reg_model):
    with pm_reg_model:
        trace_reg = pm.sample(
            draws=30,
            tune=30,
            chains=1,
            random_seed=123,
            progressbar=False,
            compute_convergence_checks=False,
        )
    return trace_reg


# ---------------------------------------------------------------------------
# Custom simulator and callback functions
# ---------------------------------------------------------------------------
def simulator_simple(mu, sigma, seed, **kwargs):
    rng = np.random.default_rng(seed)
    return {"y": rng.normal(mu, sigma, size=20)}


def augment_observed_general(model, observed_data, replicated_data, idx):
    # Custom: only keep the last 10 original obs + all replicated
    return {
        var: np.concatenate([observed_data[var].values[-10:], replicated_data[var]])
        for var in replicated_data
    }


def transform_general(param_name, param_value):
    return param_value**2


def update_data_reg(model, augmented_data, idx):
    """Resize covariates and coords to match augmented data."""
    n_aug = len(augmented_data["y"])
    x_aug = np.tile(x_obs_reg, n_aug // len(x_obs_reg) + 1)[:n_aug]
    pm.set_data(
        {"x": x_aug, "y_data": augmented_data["y"]},
        coords={"obs_id": np.arange(n_aug)},
    )


# ---------------------------------------------------------------------------
# Tests with observed variables
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("model_name,trace_name", [("pm_simple_model", "pm_simple_model_trace")])
def test_posterior_sbc_with_observed_data(model_name, trace_name, request):
    """Basic posterior SBC with a PyMC model."""
    model = request.getfixturevalue(model_name)
    trace = request.getfixturevalue(trace_name)

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
    "model_name,trace_name,update_data", [("pm_reg_model", "pm_reg_model_trace", update_data_reg)]
)
def test_posterior_sbc_with_update_data(model_name, trace_name, update_data, request):
    """Posterior SBC with dims/coords and update_data callback."""
    model = request.getfixturevalue(model_name)
    trace = request.getfixturevalue(trace_name)

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
    "model_name,trace_name,simulator",
    [("pm_simple_model", "pm_simple_model_trace", simulator_simple)],
)
def test_posterior_sbc_with_custom_simulator(model_name, trace_name, simulator, request):
    """Posterior SBC using a custom simulator function."""
    model = request.getfixturevalue(model_name)
    trace = request.getfixturevalue(trace_name)

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
    "model_name,trace_name,augment_observed",
    [("pm_simple_model", "pm_simple_model_trace", augment_observed_general)],
)
def test_posterior_sbc_with_augment_observed(model_name, trace_name, augment_observed, request):
    """Posterior SBC with a custom augment_observed callback."""
    model = request.getfixturevalue(model_name)
    trace = request.getfixturevalue(trace_name)

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
    "model_name,trace_name,transform",
    [("pm_simple_model", "pm_simple_model_trace", transform_general)],
)
def test_posterior_sbc_with_transform(model_name, trace_name, transform, request):
    """Posterior SBC with a transform(name, value) function."""
    model = request.getfixturevalue(model_name)
    trace = request.getfixturevalue(trace_name)

    sbc = simuk.SBC(
        model,
        method="posterior",
        trace=trace,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
        transform=transform,
    )
    sbc.run_simulations()
    assert "posterior_sbc" in sbc.simulations


# ---------------------------------------------------------------------------
# Error-handling tests
# ---------------------------------------------------------------------------
def test_posterior_sbc_no_trace(request):
    """method='posterior' without trace should raise ValueError."""
    model = request.getfixturevalue("pm_simple_model")
    with pytest.raises(ValueError, match="posterior samples from the"):
        simuk.SBC(
            model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_posterior_sbc_trace_missing_posterior(request):
    """trace without 'posterior' group should raise ValueError."""
    model = request.getfixturevalue("pm_simple_model")
    trace_missing = request.getfixturevalue("pm_simple_model_trace").copy()
    del trace_missing["posterior"]
    with pytest.raises(ValueError, match="posterior"):
        simuk.SBC(
            model,
            method="posterior",
            trace=trace_missing,
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_posterior_sbc_trace_missing_observed_data(request):
    """trace without 'observed_data' group should raise ValueError."""
    model = request.getfixturevalue("pm_simple_model")
    trace_missing = request.getfixturevalue("pm_simple_model_trace").copy()
    del trace_missing["observed_data"]
    with pytest.raises(ValueError, match="observed_data"):
        simuk.SBC(
            model,
            method="posterior",
            trace=trace_missing,
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_posterior_sbc_too_many_simulations(request):
    """num_simulations > draws should raise ValueError."""
    model = request.getfixturevalue("pm_simple_model")
    trace = request.getfixturevalue("pm_simple_model_trace")
    with pytest.raises(ValueError, match="more draws per"):
        simuk.SBC(
            model,
            method="posterior",
            trace=trace,
            num_simulations=100,  # trace only has 30 draws
            sample_kwargs={"draws": 5, "tune": 5},
        )


def test_posterior_sbc_numpyro_not_implemented(request):
    """Posterior SBC is not yet implemented for NumPyro."""
    numpyro = pytest.importorskip("numpyro")
    trace = request.getfixturevalue("pm_simple_model_trace")

    def numpyro_model(y=None):
        mu = numpyro.sample("mu", dist.Normal(0, 5))
        numpyro.sample("y", dist.Normal(mu, 1), obs=y)

    with pytest.raises(NotImplementedError, match="only implemented for PyMC"):
        simuk.SBC(
            NUTS(numpyro_model),
            method="posterior",
            trace=trace,
            data_dir={"y": np.array([0.0])},
            num_simulations=5,
        )


def test_posterior_sbc_warnings_for_prior(caplog, request):
    """Passing posterior-only args with method='prior' should emit warnings."""
    model = request.getfixturevalue("pm_simple_model")
    trace = request.getfixturevalue("pm_simple_model_trace")
    with caplog.at_level(logging.WARNING):
        simuk.SBC(
            model,
            method="prior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace,
            augment_observed=lambda *a: {},
            update_data=lambda *a: None,
        )

    messages = caplog.text
    assert "update_data" in messages
    assert "augment_observed" in messages
    assert "trace" in messages


def test_posterior_sbc_update_data_not_callable(request):
    """Passing a non-callable update_data should raise ValueError."""
    model = request.getfixturevalue("pm_simple_model")
    trace = request.getfixturevalue("pm_simple_model_trace")
    with pytest.raises(ValueError, match="`update_data` should be a function or None"):
        simuk.SBC(
            model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace,
            update_data="not a function",
        )


def test_posterior_sbc_augment_observed_not_callable(request):
    """Passing a non-callable augment_observed should raise ValueError."""
    model = request.getfixturevalue("pm_simple_model")
    trace = request.getfixturevalue("pm_simple_model_trace")
    with pytest.raises(ValueError, match="`augment_observed` should be a function or None"):
        simuk.SBC(
            model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace,
            augment_observed="not a function",
        )


def test_posterior_sbc_bad_simulator_args(request):
    """Any fault in executing the simulator should raise a general ValueError."""
    model = request.getfixturevalue("pm_simple_model")
    trace = request.getfixturevalue("pm_simple_model_trace")
    with pytest.raises(ValueError, match="Error generating prior predictive sample"):
        sbc = simuk.SBC(
            model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace,
            # bad function args
            simulator=lambda: None,
        )
        sbc.run_simulations()


def test_posterior_sbc_bad_simulator_return(request):
    """Simulator should return a dictionary, otherwise raise a TypeError."""
    model = request.getfixturevalue("pm_simple_model")
    trace = request.getfixturevalue("pm_simple_model_trace")
    with pytest.raises(TypeError, match="Simulator must return a dictionary"):
        sbc = simuk.SBC(
            model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace,
            simulator=lambda *args, **kwargs: None,
        )

        sbc.run_simulations()
