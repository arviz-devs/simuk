import matplotlib

matplotlib.use("Agg")  # noqa: E402  must be set before any figure is created

import arviz_plots as azp
import numpy as np
import pytest
import xarray as xr
from arviz_base.labels import BaseLabeller
from arviz_stats import eti as azs_eti
from numpyro.infer import NUTS

import simuk
from simuk.plots import (
    _apply_flexible_transform,
    _build_plot_assets,
    _build_recovery_dataset,
    _get_labels,
)


@pytest.fixture(scope="module")
def sbc_with_fits(pm_centered_eight_model):
    sbc = simuk.SBC(
        pm_centered_eight_model,
        num_simulations=10,
        sample_kwargs={"draws": 10, "tune": 10},
        seed=42,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_with_fits_numpyro(
    numpyro_eight_schools_cauchy_prior, numpyro_eight_schools_cauchy_prior_data
):
    sbc = simuk.SBC(
        NUTS(numpyro_eight_schools_cauchy_prior),
        data_dir=numpyro_eight_schools_cauchy_prior_data,
        num_simulations=10,
        sample_kwargs={"num_warmup": 10, "num_samples": 10},
        seed=42,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_no_fits(pm_centered_eight_model):
    sbc = simuk.SBC(
        pm_centered_eight_model,
        num_simulations=10,
        sample_kwargs={"draws": 10, "tune": 10},
        keep_fits=False,
        seed=42,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_posterior_with_fits(pm_simple_model, pm_simple_model_trace):
    sbc = simuk.SBC(
        pm_simple_model,
        trace=pm_simple_model_trace,
        method="posterior",
        num_simulations=10,
        seed=42,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_posterior_no_fits(pm_simple_model, pm_simple_model_trace):
    sbc = simuk.SBC(
        pm_simple_model,
        trace=pm_simple_model_trace,
        method="posterior",
        num_simulations=10,
        sample_kwargs={"draws": 5, "tune": 5},
        seed=42,
        keep_fits=False,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def sbc_collapsing_transform(pm_centered_eight_model):
    """SBC with a vector -> scalar transform (theta collapsed to its mean)."""
    sbc = simuk.SBC(
        pm_centered_eight_model,
        num_simulations=3,
        sample_kwargs={"draws": 20, "tune": 20},
        seed=7,
        progress_bar=False,
        transform=lambda name, value: np.mean(value) if name == "theta" else value,
    )
    sbc.run_simulations()
    return sbc


@pytest.fixture(scope="module")
def recovery_ds(sbc_with_fits):
    return _build_recovery_dataset(
        sbc_with_fits,
        0.89,
        "eti",
        "mean",
        sbc_with_fits._transform,
        sbc_with_fits.kept_simulation_params.var_names,
    )


# ---------------------------------------------------------------------------
# _apply_flexible_transform
# ---------------------------------------------------------------------------
def test_apply_flexible_transform_identity():
    da = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=["simulation", "sample", "j"],
    )
    out = _apply_flexible_transform(da, lambda name, x: x)
    assert out.shape == da.shape
    np.testing.assert_allclose(out.values, da.values)


def test_apply_flexible_transform_scalar_output():
    # vector -> scalar (e.g. mean over the parameter axis)
    da = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=["simulation", "sample", "j"],
    )
    out = _apply_flexible_transform(da, lambda name, x: np.mean(x))
    assert out.dims == ("simulation", "sample")
    np.testing.assert_allclose(out.values, da.mean(axis=-1).values)


def test_apply_flexible_transform_vector_output():
    # vector -> vector of different dims
    da = xr.DataArray(
        np.arange(24.0).reshape(2, 3, 4),
        dims=["simulation", "sample", "j"],
        name="theta",
    )
    new_values = [1, 2, 3]
    new_arr = np.broadcast_to(
        new_values, (da.sizes["simulation"], da.sizes["sample"], len(new_values))
    )

    with pytest.warns(UserWarning, match="Spawned dimensions:"):
        out = _apply_flexible_transform(da, lambda name, x: new_values)

    assert out.dims == ("simulation", "sample", "theta_dim_0")
    np.testing.assert_allclose(out.values, new_arr)


def test_apply_flexible_transform_loop_dims_sample_only():
    # ref_params have dims (sample, *param_dims); loop only over "sample"
    da = xr.DataArray(np.arange(12.0).reshape(4, 3), dims=["sample", "j"])
    out = _apply_flexible_transform(da, lambda name, x: x, loop_dims=("sample",))
    assert out.shape == da.shape
    np.testing.assert_allclose(out.values, da.values)


# ---------------------------------------------------------------------------
# get_labels
# ---------------------------------------------------------------------------
def test_get_labels_scalar_vars():
    ds = xr.Dataset(
        {
            "mu": (("simulation",), np.arange(4.0)),
            "tau": (("simulation",), np.arange(4.0)),
        }
    )
    assert _get_labels(ds) == ["mu", "tau"]


def test_get_labels_vector_vars():
    da = xr.DataArray(np.zeros((4, 3)), dims=["simulation", "j"], coords={"j": [10, 20, 30]})
    ds = xr.Dataset({"theta": da})
    # labels use positional indices from np.ndindex, not coordinate values
    assert _get_labels(ds) == ["theta['j=0']", "theta['j=1']", "theta['j=2']"]


def test_get_labels_mixed():
    ds = xr.Dataset(
        {
            "mu": (("simulation",), np.zeros(4)),
            "theta": (("simulation", "j"), np.zeros((4, 2))),
        }
    )
    assert _get_labels(ds) == ["mu", "theta['j=0']", "theta['j=1']"]


# ---------------------------------------------------------------------------
# _build_recovery_dataset
# ---------------------------------------------------------------------------
def test_build_recovery_dataset_basic(sbc_with_fits):
    ds = _build_recovery_dataset(
        sbc_with_fits,
        0.89,
        "eti",
        "mean",
        sbc_with_fits._transform,
        sbc_with_fits.kept_simulation_params.var_names,
    )
    assert set(ds.data_vars) == {"recovery", "covered"}
    assert ds["recovery"].dims == ("simulation", "parameter", "quantity")
    assert ds["covered"].dims == ("simulation", "parameter")
    assert list(ds["recovery"].coords["quantity"].values) == [
        "true",
        "estimate",
        "ci_low",
        "ci_high",
    ]
    # mu(1) + tau(1) + theta(8) = 10 parameters
    assert ds.sizes["parameter"] == 10
    assert ds.sizes["simulation"] == 10


def test_build_recovery_dataset_true_values(sbc_with_fits):
    # the "true" quantity must match the reference params used to simulate
    ds = _build_recovery_dataset(
        sbc_with_fits, 0.89, "eti", "mean", sbc_with_fits._transform, ["mu", "tau"]
    )
    ref = sbc_with_fits.kept_simulation_params.ref_params
    n_sims = ds.sizes["simulation"]
    for p in ["mu", "tau"]:
        expected = np.array([ref[p].isel(sample=j).values for j in range(n_sims)])
        actual = ds["recovery"].sel(quantity="true", parameter=p).values
        np.testing.assert_allclose(expected, actual)


def test_build_recovery_dataset_ci_consistency(sbc_with_fits):
    ds = _build_recovery_dataset(
        sbc_with_fits,
        0.89,
        "eti",
        "mean",
        sbc_with_fits._transform,
        sbc_with_fits.kept_simulation_params.var_names,
    )
    recovery = ds["recovery"]
    assert (recovery.sel(quantity="ci_low") <= recovery.sel(quantity="ci_high")).all()
    covered_manual = (recovery.sel(quantity="ci_low") <= recovery.sel(quantity="true")) & (
        recovery.sel(quantity="true") <= recovery.sel(quantity="ci_high")
    )
    np.testing.assert_array_equal(covered_manual.values, ds["covered"].values)


def test_build_recovery_dataset_hdi_median(sbc_with_fits):
    ds = _build_recovery_dataset(
        sbc_with_fits, 0.89, "hdi", "median", sbc_with_fits._transform, ["mu"]
    )
    recovery = ds["recovery"]
    assert (recovery.sel(quantity="ci_low") <= recovery.sel(quantity="ci_high")).all()


def test_build_recovery_dataset_eti_width(sbc_with_fits):
    # eti bounds must match a direct arviz_stats.eti call on the stacked draws
    ds = _build_recovery_dataset(
        sbc_with_fits, 0.89, "eti", "mean", sbc_with_fits._transform, ["mu"]
    )
    post_ds = sbc_with_fits.posteriors[["mu"]]
    ci = azs_eti(post_ds, prob=0.89, dim="sample")["mu"]
    np.testing.assert_allclose(
        ds["recovery"].sel(quantity="ci_low", parameter="mu").values,
        ci.sel(ci_bound="lower").values,
    )
    np.testing.assert_allclose(
        ds["recovery"].sel(quantity="ci_high", parameter="mu").values,
        ci.sel(ci_bound="upper").values,
    )


def test_build_recovery_dataset_collapsing_transform(sbc_collapsing_transform):
    sbc = sbc_collapsing_transform
    ds = _build_recovery_dataset(
        sbc, 0.89, "eti", "mean", sbc._transform, sbc.kept_simulation_params.var_names
    )
    # theta collapses to its mean: 3 labels instead of 10
    assert list(ds.coords["parameter"].values) == ["mu", "tau", "theta"]
    ref = sbc.kept_simulation_params.ref_params
    expected_theta = np.array(
        [ref["theta"].isel(sample=j).mean().item() for j in range(ds.sizes["simulation"])]
    )
    actual = ds["recovery"].sel(quantity="true", parameter="theta").values
    np.testing.assert_allclose(expected_theta, actual)


# ---------------------------------------------------------------------------
# _build_plot_assets
# ---------------------------------------------------------------------------
def test_build_plot_assets_shapes(recovery_ds):
    ref_line, points, intervals, titles = _build_plot_assets(recovery_ds, 0.89, BaseLabeller())
    n_params = recovery_ds.sizes["parameter"]
    assert ref_line.dims == ("plot_axis", "endpoint", "parameter")
    assert ref_line.sizes["parameter"] == n_params
    assert points.dims == ("plot_axis", "simulation", "parameter")
    assert intervals.dims == ("plot_axis", "endpoint", "simulation", "parameter")
    assert titles.dims == ("parameter",)
    assert list(ref_line.coords["plot_axis"].values) == ["x", "y"]
    assert list(points.coords["plot_axis"].values) == ["x", "y"]
    assert list(intervals.coords["plot_axis"].values) == ["x", "y"]


def test_build_plot_assets_ref_line_diagonal(recovery_ds):
    ref_line, *_ = _build_plot_assets(recovery_ds, 0.89, BaseLabeller())
    x = ref_line.sel(plot_axis="x")
    y = ref_line.sel(plot_axis="y")
    np.testing.assert_allclose(x.values, y.values)
    assert (x.sel(endpoint=1) >= x.sel(endpoint=0)).all()


def test_build_plot_assets_intervals(recovery_ds):
    *_, intervals, _ = _build_plot_assets(recovery_ds, 0.89, BaseLabeller())
    # x coordinate is the true value at both endpoints
    x = intervals.sel(plot_axis="x")
    np.testing.assert_allclose(x.sel(endpoint=0).values, x.sel(endpoint=1).values)
    np.testing.assert_allclose(
        x.sel(endpoint=0).values,
        recovery_ds["recovery"].sel(quantity="true").transpose("simulation", "parameter").values,
    )
    # y goes lower -> upper
    y0 = intervals.sel(plot_axis="y", endpoint=0)
    y1 = intervals.sel(plot_axis="y", endpoint=1)
    np.testing.assert_allclose(
        y0.values,
        recovery_ds["recovery"].sel(quantity="ci_low").transpose("simulation", "parameter").values,
    )
    np.testing.assert_allclose(
        y1.values,
        recovery_ds["recovery"].sel(quantity="ci_high").transpose("simulation", "parameter").values,
    )


def test_build_plot_assets_titles(recovery_ds):
    ci_prob = 0.89
    *_, titles = _build_plot_assets(recovery_ds, ci_prob, BaseLabeller())
    params = recovery_ds.coords["parameter"].values
    expected_coverages = recovery_ds["covered"].mean("simulation").values
    for title, param, coverage in zip(titles.values, params, expected_coverages):
        assert title.startswith(f"{param}\ncoverage: ")
        assert f"{coverage:.1%}" in title
        assert f"{ci_prob:.1%}" in title


# ---------------------------------------------------------------------------
# plot_parameter_recovery
# ---------------------------------------------------------------------------
def test_ppr_requires_keep_fits(sbc_no_fits):
    with pytest.raises(ValueError, match="keep_fits"):
        simuk.plot_parameter_recovery(sbc_no_fits)


def test_ppr_requires_keep_fits_posterior(sbc_posterior_no_fits):
    with pytest.raises(ValueError, match="keep_fits"):
        simuk.plot_parameter_recovery(sbc_posterior_no_fits)


def test_ppr_requires_completed_simulations(pm_centered_eight_model):
    sbc = simuk.SBC(
        pm_centered_eight_model,
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    with pytest.raises(ValueError, match="No posteriors"):
        simuk.plot_parameter_recovery(sbc)


def test_ppr_requires_completed_simulations_posterior(pm_simple_model, pm_simple_model_trace):
    sbc = simuk.SBC(
        pm_simple_model,
        trace=pm_simple_model_trace,
        method="posterior",
        num_simulations=2,
        sample_kwargs={"draws": 5, "tune": 5},
    )
    with pytest.raises(ValueError, match="No posteriors"):
        simuk.plot_parameter_recovery(sbc)


def test_ppr_basic(sbc_with_fits):
    pc = simuk.plot_parameter_recovery(sbc_with_fits)
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_plot_ppr_basic_numpyro(sbc_with_fits_numpyro):
    pc = simuk.plot_parameter_recovery(sbc_with_fits_numpyro)
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_ppr_numpyro_plate_dim_labels(sbc_with_fits_numpyro):
    """Plate dims must be named after the model plate, not auto-named."""
    sbc = sbc_with_fits_numpyro
    ds = _build_recovery_dataset(
        sbc, 0.89, "eti", "mean", sbc._transform, sbc.kept_simulation_params.var_names
    )
    labels = list(ds.coords["parameter"].values)
    # mu, tau are scalar; theta lives under plate J (size 8)
    assert labels[0] == "mu"
    assert labels[1] == "tau"
    assert labels[2] == "theta['J=0']"
    assert labels[-1] == "theta['J=7']"


def test_ppr_numpyro_true_values_match_ref(sbc_with_fits_numpyro):
    """'true' values must equal the reference params used to simulate."""
    sbc = sbc_with_fits_numpyro
    ds = _build_recovery_dataset(
        sbc, 0.89, "eti", "mean", sbc._transform, sbc.kept_simulation_params.var_names
    )
    ref = sbc.kept_simulation_params.ref_params
    labels = list(ds.coords["parameter"].values)
    trues = ds["recovery"].sel(quantity="true")
    for var in ref.data_vars:
        for j in range(ds.sizes["simulation"]):
            expected = np.atleast_1d(ref[var].isel(sample=j).values).ravel()
            actual = trues.sel(
                parameter=[p for p in labels if p == var or p.startswith(var + "[")], simulation=j
            ).values
            np.testing.assert_allclose(expected, actual.ravel())


def test_ppr_basic_posterior(sbc_posterior_with_fits):
    pc = simuk.plot_parameter_recovery(sbc_posterior_with_fits)
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_ppr_var_names_filter(sbc_with_fits):
    pc = simuk.plot_parameter_recovery(sbc_with_fits, var_names=["mu"])
    used_params = len(pc.data.coords["parameter"])
    assert used_params == 1


def test_ppr_var_names_filter_posterior(sbc_posterior_with_fits):
    pc = simuk.plot_parameter_recovery(sbc_posterior_with_fits, var_names=["mu"])
    used_params = len(pc.data.coords["parameter"])
    assert used_params == 1


def test_ppr_unknown_var(sbc_with_fits):
    with pytest.raises(ValueError, match="unknown variables"):
        simuk.plot_parameter_recovery(sbc_with_fits, var_names=["nope"])


def test_ppr_custom_ci_prob(sbc_with_fits):
    pc = simuk.plot_parameter_recovery(sbc_with_fits, ci_prob=0.5)
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_ppr_custom_ci_prob_posterior(sbc_posterior_with_fits):
    pc = simuk.plot_parameter_recovery(sbc_posterior_with_fits, ci_prob=0.5)
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_ppr_ci_kind_hdi(sbc_with_fits):
    pc = simuk.plot_parameter_recovery(sbc_with_fits, ci_kind="hdi")
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_ppr_invalid_point_estimate(sbc_with_fits):
    with pytest.raises(ValueError, match="is not one of"):
        simuk.plot_parameter_recovery(sbc_with_fits, point_estimate="modddd")


def test_ppr_invalid_ci_kind(sbc_with_fits):
    with pytest.raises(ValueError, match="is not one of"):
        simuk.plot_parameter_recovery(sbc_with_fits, ci_kind="modddd")


def test_ppr_with_preexisting_plot_collection(sbc_with_fits):
    ds = _build_recovery_dataset(
        sbc_with_fits, 0.89, "eti", "mean", sbc_with_fits._transform, ["mu", "tau", "theta"]
    )

    pc = azp.plot_collection.PlotCollection.wrap(
        ds[["recovery"]],
        cols=["parameter"],
        col_wrap=4,
        backend="matplotlib",
    )
    # mu(1) + tau(1) + theta(8) = 10 subplots needed
    returned_pc = simuk.plot_parameter_recovery(sbc_with_fits, plot_collection=pc)
    assert returned_pc is pc


def test_ppr_visuals_disable(sbc_with_fits):
    pc = simuk.plot_parameter_recovery(
        sbc_with_fits,
        visuals={"reference_line": False, "legend": False},
    )
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_ppr_invalid_visual_key(sbc_with_fits):
    with pytest.raises(ValueError, match="Found keys"):
        simuk.plot_parameter_recovery(sbc_with_fits, visuals={"nope": {}})


def test_ppr_pc_kwargs(sbc_with_fits):
    pc = simuk.plot_parameter_recovery(
        sbc_with_fits, pc_kwargs={"col_wrap": 5, "figure_kwargs": {"figsize": (800, 800)}}
    )
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_plot_ecdf_basic(sbc_with_fits, sbc_no_fits):
    pc = simuk.plot_ecdf(sbc_with_fits)
    assert isinstance(pc, azp.plot_collection.PlotCollection)

    pc = simuk.plot_ecdf(sbc_no_fits)
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_plot_ecdf_posterior(sbc_posterior_with_fits, sbc_posterior_no_fits):
    pc = simuk.plot_ecdf(sbc_posterior_with_fits)
    assert isinstance(pc, azp.plot_collection.PlotCollection)

    pc = simuk.plot_ecdf(sbc_posterior_no_fits)
    assert isinstance(pc, azp.plot_collection.PlotCollection)


def test_plot_ecdf_kwargs_forwarded(sbc_with_fits):
    pc = simuk.plot_ecdf(sbc_with_fits, coverage=True)
    assert isinstance(pc, azp.plot_collection.PlotCollection)
