from collections import ChainMap

import numpy as np
import pytest

import simuk
from simuk.tests.conftest import (
    custom_simulator,
    custom_transform,
    keep_only_replicated_data_augment,
    obs_data,
)


def test_posterior_sbc_with_update_data(simple_model_numpyro_trace, numpyro_nuts_kernel):
    with pytest.raises(NotImplementedError):
        simuk.SBC(
            numpyro_nuts_kernel,
            method="posterior",
            trace=simple_model_numpyro_trace,
            num_simulations=2,
            data_dir={"y": obs_data},
            update_data=lambda x: x,
        )


def test_posterior_sbc_with_augment_observed(simple_model_numpyro_trace, numpyro_nuts_kernel):
    """Posterior SBC with a custom augment_observed callback."""
    sbc = simuk.SBC(
        numpyro_nuts_kernel,
        method="posterior",
        trace=simple_model_numpyro_trace,
        num_simulations=2,
        data_dir={"y": obs_data},
        augment_observed=keep_only_replicated_data_augment,
    )
    sbc.run_simulations()
    assert "posterior_sbc" in sbc.simulations
    assert sbc._simulations_complete == 2


def test_posterior_sbc_with_custom_simulator(simple_model_numpyro_trace, numpyro_nuts_kernel):
    """Posterior SBC using a custom simulator function."""
    sbc = simuk.SBC(
        numpyro_nuts_kernel,
        method="posterior",
        trace=simple_model_numpyro_trace,
        num_simulations=2,
        data_dir={"y": obs_data},
        simulator=custom_simulator,
    )
    sbc.run_simulations()
    assert "posterior_sbc" in sbc.simulations
    assert sbc._simulations_complete == 2


def test_posterior_sbc_numpyro(simple_model_numpyro_trace, numpyro_nuts_kernel):
    sbc = simuk.SBC(
        numpyro_nuts_kernel,
        method="posterior",
        trace=simple_model_numpyro_trace,
        data_dir={"y": obs_data},
        num_simulations=2,
    )
    sbc.run_simulations()
    assert sbc.engine == "numpyro"
    assert "posterior_sbc" in sbc.simulations
    assert sbc.simulations.attrs["inference_library"] == "numpyro"

    posterior_sbc = sbc.simulations["posterior_sbc"]
    assert posterior_sbc.creation_library == "ArviZ"
    assert posterior_sbc.dims == ChainMap({"chain": 1, "draw": 2})


def test_posterior_sbc_with_transform(simple_model_numpyro_trace, numpyro_nuts_kernel):
    """Posterior SBC with a transform(name, value) function."""
    sbc_with_transforms = simuk.SBC(
        numpyro_nuts_kernel,
        method="posterior",
        seed=1234,
        trace=simple_model_numpyro_trace,
        num_simulations=2,
        data_dir={"y": obs_data},
        transform=custom_transform,
    )
    sbc_with_transforms.run_simulations()
    sbc_without_transforms = simuk.SBC(
        numpyro_nuts_kernel,
        method="posterior",
        seed=1234,
        trace=simple_model_numpyro_trace,
        num_simulations=2,
        data_dir={"y": obs_data},
    )
    sbc_without_transforms.run_simulations()

    # given a linear transform, ranks are preserved
    np.testing.assert_allclose(
        sbc_with_transforms.simulations["posterior_sbc"]["mu"],
        sbc_without_transforms.simulations["posterior_sbc"]["mu"],
    )
    assert sbc_without_transforms._simulations_complete == 2
    assert sbc_with_transforms._simulations_complete == 2


def test_posterior_sbc_no_trace(numpyro_nuts_kernel):
    """method='posterior' without trace should raise ValueError."""
    with pytest.raises(ValueError, match="posterior samples from the"):
        simuk.SBC(
            numpyro_nuts_kernel,
            method="posterior",
        )


@pytest.mark.parametrize("missing_group", ["posterior", "observed_data"])
def test_posterior_sbc_trace_missing_group(
    missing_group, simple_model_numpyro_trace, numpyro_nuts_kernel
):
    """trace without 'posterior' group should raise ValueError."""
    trace_missing = simple_model_numpyro_trace.copy()
    del trace_missing[missing_group]
    with pytest.raises(ValueError, match=missing_group):
        simuk.SBC(
            numpyro_nuts_kernel,
            method="posterior",
            trace=trace_missing,
        )


def test_posterior_sbc_too_many_simulations(simple_model_numpyro_trace, numpyro_nuts_kernel):
    """num_simulations > draws should raise ValueError."""
    assert simple_model_numpyro_trace["posterior"].dims == ChainMap({"chain": 1, "draw": 30})
    with pytest.raises(ValueError, match="more draws per"):
        simuk.SBC(
            numpyro_nuts_kernel,
            method="posterior",
            trace=simple_model_numpyro_trace,
            num_simulations=100,
        )
