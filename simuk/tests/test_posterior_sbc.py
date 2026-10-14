"""Tests for Posterior SBC (method='posterior')."""

import logging

import pytest

import simuk
from simuk.tests.conftest import (
    custom_augment_observed,
    custom_simulator,
    custom_transform,
    reg_model,
    simple_model,
    trace_reg,
    trace_simple,
    update_data_reg,
)


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
    assert sbc._simulations_complete == 2


@pytest.mark.parametrize("model,trace,update_data", [(reg_model, trace_reg, update_data_reg)])
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
    assert sbc._simulations_complete == 2


# ---------------------------------------------------------------------------
# Tests with custom simulator and callbacks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model,trace,simulator", [(simple_model, trace_simple, custom_simulator)])
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
    assert sbc._simulations_complete == 2


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
    assert sbc._simulations_complete == 2


@pytest.mark.parametrize(
    "model,trace,transform",
    [(simple_model, trace_simple, custom_transform)],
)
def test_posterior_sbc_with_transform(model, trace, transform):
    """Posterior SBC with a transform(name, value) function."""
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
    assert sbc._simulations_complete == 2


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
    del trace_missing["posterior"]
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
    del trace_missing["observed_data"]
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


def test_posterior_sbc_update_data_not_callable():
    """Passing a non-callable update_data should raise ValueError."""
    with pytest.raises(ValueError, match="`update_data` should be a function or None"):
        simuk.SBC(
            simple_model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace_simple,
            update_data="not a function",
        )


def test_posterior_sbc_augment_observed_not_callable():
    """Passing a non-callable augment_observed should raise ValueError."""
    with pytest.raises(ValueError, match="`augment_observed` should be a function or None"):
        simuk.SBC(
            simple_model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace_simple,
            augment_observed="not a function",
        )


def test_posterior_sbc_bad_simulator_args():
    """Any fault in executing the simulator should raise a general ValueError."""
    with pytest.raises(ValueError, match="Error generating predictive samples"):
        sbc = simuk.SBC(
            simple_model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace_simple,
            # bad function args
            simulator=lambda: None,
        )
        sbc.run_simulations()


def test_posterior_sbc_bad_simulator_return():
    """Simulator should return a dictionary, otherwise raise a TypeError."""
    with pytest.raises(TypeError, match="Simulator must return a dictionary"):
        sbc = simuk.SBC(
            simple_model,
            method="posterior",
            num_simulations=5,
            sample_kwargs={"draws": 5, "tune": 5},
            trace=trace_simple,
            simulator=lambda *args, **kwargs: None,
        )

        sbc.run_simulations()
