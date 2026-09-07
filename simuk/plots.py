"""Parameter recovery plotting for simulation-based calibration."""

import warnings
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any, Literal

import arviz_plots as azp
import numpy as np
import xarray as xr
from arviz_base import rcParams
from arviz_base.labels import BaseLabeller
from arviz_base.validate import validate_dict_argument, validate_or_use_rcparam
from arviz_plots import plot_ecdf_pit, style
from arviz_plots.plot_collection import PlotCollection
from arviz_plots.plots.utils import filter_aes, get_visual_kwargs, set_wrap_layout
from arviz_stats import eti, hdi

style.use("arviz-variat")


def plot_ecdf(sbc, **kwargs):
    """Plot the empirical cumulative distribution function (ECDF) of SBC rank statistics.

    Determines the target group ("posterior_sbc" or "prior_sbc") based on the calibration
    method and visualizes the uniformity of probability integral transform (PIT) / rank
    values.

    Parameters
    ----------
    sbc : simuk.SBC
        Fitted SBC object containing simulation results in ``sbc.simulations``
        and the evaluation mode in ``sbc.method``.
    **kwargs : dict, optional
        Additional keyword arguments forwarded to ``arviz_plots.plot_ecdf_pit``.

    Returns
    -------
    arviz_plots.PlotCollection
        The PlotCollection wrapping the ECDF plot.
    """
    group = "posterior_sbc" if sbc.method == "posterior" else "prior_sbc"
    pc = plot_ecdf_pit(
        sbc.simulations,
        group=group,
        visuals={"xlabel": False},
        **kwargs,
    )
    return pc


def _apply_flexible_transform(
    da: xr.DataArray, transform: Callable[[str, Any], Any], loop_dims=("simulation", "sample")
) -> xr.DataArray:
    """Apply a transformation to each (simulation, sample) slice of a Dataset.

    Iterates over each data variable in the dataset, applies `transform(name, value)`
    to each slice indexed by `simulation` and `sample`, and reconstructs an xarray
    Dataset. Automatically detects output dimensionality and assigns new dimension
    names if the shape changes.

    Assumes that the transform output shape is consistent per data variable.

    Parameters
    ----------
    da : xr.DataArray
        Input data array containing at least 'simulation' and 'sample' dimensions.
    transform : callable
        Function with signature ``transform(name: str, value: np.ndarray) -> array_like``
        applied to each slice. Must return consistent shapes across all slices
        for a given variable.
    loop_dims : tuple of str, optional
        Dimensions to loop over. Defaults to ('simulation', 'sample').

    Returns
    -------
    xr.DataArray
        Transformed DataArray preserving ``loop_dims`` coordinates.
    """
    in_core_dims = [d for d in da.dims if d not in loop_dims]

    # Because we allow transform to output arbitrary dimensions
    # we determine the output shape and core dimensions dynamically
    sample_indexers = {d: 0 for d in loop_dims if d in da.dims}
    sample_in = da.isel(sample_indexers).values
    sample_out = np.asarray(transform(da.name, sample_in))
    out_ndim = sample_out.ndim

    # Match core dims only if both and AND axis lengths match
    in_core_shape = tuple(da.sizes[d] for d in in_core_dims)
    # transform outputs scalar
    if out_ndim == 0:
        out_core_dims = []
    # transform outputs array with same shape as input
    elif sample_out.shape == in_core_shape:
        out_core_dims = in_core_dims
    # transform outputs array with different shape
    else:
        out_core_dims = [f"{da.name}_dim_{i}" for i in range(out_ndim)]
        warnings.warn(
            f"Variable '{da.name}' transformed from {len(in_core_dims)}D to {out_ndim}D. "
            f"Spawned dimensions: {out_core_dims}",
            UserWarning,
        )

    return xr.apply_ufunc(
        lambda arr: np.asarray(transform(da.name, arr)),
        da,
        input_core_dims=[in_core_dims],
        output_core_dims=[out_core_dims],
        vectorize=True,
    )


def _get_labels(ds: xr.Dataset, loop_dims=("simulation",)) -> list[str]:
    all_labels = []
    for name, da in ds.data_vars.items():
        param_dims = [d for d in da.dims if d not in loop_dims]
        param_shape = tuple(da.sizes[d] for d in param_dims)

        if len(param_shape) == 0:
            all_labels.extend([name])
        else:
            all_labels.extend(
                [
                    f"{name}{[f'{dim}={idx[i]}' for i, dim in enumerate(param_dims)]}"
                    for idx in np.ndindex(param_shape)
                ]
            )

    return all_labels


def _build_recovery_dataset(sbc, ci_prob, ci_kind, point_estimate, transform, var_names):
    """Compute parameter recovery metrics and coverage intervals across simulations.

    Transforms true parameter values and posterior samples, estimates credible
    intervals and central tendencies, and stacks all variables into unified
    coordinate arrays for downstream SBC validation.

    Parameters
    ----------
    sbc : SBC
        Simulation-based calibration results container containing ``posteriors``
        and ``kept_simulation_params``.
    ci_prob : float
        Credibility mass for interval estimation (e.g., 0.95 for 95% intervals).
    ci_kind : {"eti", "hdi"}
        Interval type: equal-tailed interval ("eti") or highest density interval ("hdi").
    point_estimate : {"mean", "median"}
        Summary statistic to evaluate posterior central tendency.
    transform : callable
        Transformation function with signature ``transform(name: str, value: np.ndarray)``
        applied to each simulation/sample slice.
    var_names : list of str
        Subset of variable names to extract and process.

    Returns
    -------
    xr.Dataset
        Dataset containing:
        - ``recovery`` : DataArray of shape (simulation, parameter, quantity) containing
          "true", "estimate", "ci_low", and "ci_high" values.
        - ``covered`` : Boolean DataArray of shape (simulation, parameter) indicating whether
          the true parameter lies within the credible interval.
    """
    # compute transformed posteriors and ref
    post_ds = sbc.posteriors[var_names]
    transformed_post_ds = post_ds.map(_apply_flexible_transform, transform=transform)
    ref_params = sbc.kept_simulation_params.ref_params[var_names]
    transformed_trues = ref_params.map(
        _apply_flexible_transform, transform=transform, loop_dims=("sample",)
    ).rename_dims({"sample": "simulation"})

    # compute credible intervals
    ci_func = {"eti": eti, "hdi": hdi}[ci_kind]
    ci_bounds = ci_func(transformed_post_ds, prob=ci_prob, dim="sample")

    # compute point estimate of transformed posterior samples
    estimates = (
        transformed_post_ds.mean(dim="sample")
        if point_estimate == "mean"
        else transformed_post_ds.median(dim="sample")
    )

    # labels
    labels = _get_labels(transformed_trues, loop_dims=("simulation",))

    # covered dataarray
    covered = (ci_bounds.sel(ci_bound="lower") <= transformed_trues) & (
        transformed_trues <= ci_bounds.sel(ci_bound="upper")
    )

    def _ds_to_param_da(ds):
        da = ds.to_stacked_array(
            new_dim="parameter",
            sample_dims=["simulation"],
        )
        return da.reset_index("parameter", drop=True).assign_coords({"parameter": labels})

    covered_da = _ds_to_param_da(covered)
    estimates_da = _ds_to_param_da(estimates)
    transformed_trues_da = _ds_to_param_da(transformed_trues)
    ci_low_da = _ds_to_param_da(ci_bounds.sel(ci_bound="lower", drop=True))
    ci_high_da = _ds_to_param_da(ci_bounds.sel(ci_bound="upper", drop=True))

    recovery = xr.concat(
        [transformed_trues_da, estimates_da, ci_low_da, ci_high_da],
        dim="quantity",
    ).assign_coords({"quantity": ["true", "estimate", "ci_low", "ci_high"]})

    recovery = recovery.transpose("simulation", "parameter", "quantity")
    covered_da = covered_da.transpose("simulation", "parameter")

    return xr.Dataset({"recovery": recovery, "covered": covered_da})


def _build_plot_assets(ds, ci_prob, labeller):
    """Construct plot assets parameter recovery plots.

    Computes coordinates for identity reference lines, scatter points (true vs.
    estimate), and vertical credible interval segments. Formats subplot titles
    with observed empirical coverage and theoretical binomial confidence bounds.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing recovery variables:
        - ``recovery`` : shape (simulation, parameter, quantity)
        - ``covered`` : shape (simulation, parameter)
    ci_prob : float
        Nominal credible interval mass (between 0 and 1).
    labeller : labeller
        ArviZ labeller used to format parameter coordinate names.

    Returns
    -------
    ref_line : xr.DataArray
        Reference line endpoints with dimensions ``("plot_axis", "endpoint", "parameter")``.
    points : xr.DataArray
        Scatter plot coordinates (true, estimate) with dimensions
        ``("plot_axis", "simulation", "parameter")``.
    intervals : xr.DataArray
        Vertical credible interval segment endpoints with dimensions
        ``("plot_axis", "endpoint", "simulation", "parameter")``.
    titles : xr.DataArray
        Formatted string titles with dimension ``("parameter",)`` reporting
        observed coverage and binomial error margins.
    """
    recovery = ds["recovery"]
    line_min = np.minimum(
        recovery.sel(quantity="true").min("simulation"),
        recovery.sel(quantity="ci_low").min("simulation"),
    )
    line_max = np.maximum(
        recovery.sel(quantity="true").max("simulation"),
        recovery.sel(quantity="ci_high").max("simulation"),
    )

    ref_line = (
        xr.concat(
            [
                xr.concat([line_min, line_max], dim="endpoint"),
                xr.concat([line_min, line_max], dim="endpoint"),
            ],
            dim="plot_axis",
        )
        .assign_coords(plot_axis=["x", "y"], endpoint=[0, 1])
        .rename("reference_line")
    )

    points = (
        xr.concat(
            [
                recovery.sel(quantity="true"),
                recovery.sel(quantity="estimate"),
            ],
            dim="plot_axis",
        )
        .assign_coords(plot_axis=["x", "y"])
        .rename("points")
    )

    # vertical segments: x = true at both endpoints, y goes lower -> upper
    true_vals = recovery.sel(quantity="true")
    ci_low = recovery.sel(quantity="ci_low")
    ci_high = recovery.sel(quantity="ci_high")
    intervals = (
        xr.concat(
            [
                xr.concat([true_vals, true_vals], dim="endpoint"),
                xr.concat([ci_low, ci_high], dim="endpoint"),
            ],
            dim="plot_axis",
        )
        .assign_coords(plot_axis=["x", "y"], endpoint=[0, 1])
        .rename("intervals")
    )

    n_sims = ds["covered"].sizes["simulation"]
    coverages = ds["covered"].mean("simulation")
    margin = 2 * np.sqrt(ci_prob * (1 - ci_prob) / n_sims)
    labels = [
        labeller.make_label_vert(param, {}, {})
        for param in ds["covered"].coords["parameter"].values
    ]
    titles = xr.DataArray(
        [
            f"{label}\ncoverage: {coverage:.1%}\n(expected: {ci_prob:.1%} ± {margin:.1%})"
            for label, coverage in zip(labels, coverages.values)
        ],
        dims=["parameter"],
        coords={"parameter": ds["covered"].coords["parameter"]},
    ).rename("titles")

    return ref_line, points, intervals, titles


def plot_parameter_recovery(
    sbc,
    *,
    var_names=None,
    point_estimate=None,
    ci_prob=None,
    ci_kind=None,
    plot_collection=None,
    backend=None,
    labeller=None,
    aes_by_visuals: Mapping[
        Literal["errorbar", "reference_line", "title", "labels", "ticklabels"],
        Any,
    ] = None,
    visuals: Mapping[
        Literal[
            "labels",
            "reference_line",
            "errorbar",
            "title",
            "legend",
            "ticklabels",
        ],
        Mapping[str, Any] | bool,
    ] = None,
    pc_kwargs=None,
):
    """Create a parameter recovery plot for an SBC object.

    For each parameter (scalar or flattened vector element), the plot shows the
    posterior point estimate against the true value, a vertical credible interval,
    and a 45-degree reference line.  Subplot titles report the observed coverage
    and its expected value (plus/minus two binomial standard errors).

    Parameters
    ----------
    sbc : simuk.SBC
        Fitted SBC object with posteriors and kept simulation parameters.
    var_names : str or list of str, optional
        Variable names to include in the plot.  If None, all parameters in
        ``sbc.kept_simulation_params.var_names`` are used.
    point_estimate : {"mean", "median"}, optional
        Point estimate to plot on the y-axis.  Defaults to
        ``rcParams["stats.point_estimate"]`` ("mean").
    ci_prob : float, optional
        Credible interval probability.  Defaults to ``rcParams["stats.ci_prob"]``
        (0.89).
    ci_kind : {"eti", "hdi"}, optional
        Which credible interval to use. Defaults to ``rcParams["stats.ci_kind"]``
    plot_collection : arviz_plots.PlotCollection, optional
        Existing plot collection to add to.  If None, a new one is created.
    backend : str, optional
        Plotting backend.  Defaults to ``rcParams["plot.backend"]``.
    labeller : labeller, optional
        Labeller used to label each subplot. Defaults to ``BaseLabeller``.
    aes_by_visuals : mapping, optional
        Mapping of visuals to aesthetics that should use their mapping in the
        ``PlotCollection`` when plotted. Valid keys are the same as for
        ``visuals``.
    visuals : mapping of {str : mapping or bool}, optional
        Visuals configuration.  Keys are visual names, values are kwargs dicts.
        Use ``False`` to disable a visual.  Default is ``{}``.

        Supported visuals
        -----------------
        - ``"labels"``: kwargs for ``labelled_x`` / ``labelled_y``
        - ``"reference_line"``: kwargs for the 45-degree line (``line_xy``)
        - ``"errorbar"``: kwargs for the per-simulation scatter + CI line
        - ``"title"``: kwargs for the subplot title
        - ``"legend"``: kwargs for the figure-level legend
        - ``"ticklabels"``: kwargs for :func:`~.visuals.ticklabel_props`
            (e.g. ``size``, ``color``); applied to both axes
    pc_kwargs : dict, optional
        Additional kwargs for ``PlotCollection.wrap``, such as ``cols``,
        ``col_wrap`` or ``figure_kwargs``.  When no ``figure_kwargs`` are given,
        the figure size is scaled automatically from the number of subplots.

    Returns
    -------
    arviz_plots.PlotCollection
        The PlotCollection wrapping the recovery plot.
    """
    if not getattr(sbc, "keep_fits", False):
        raise ValueError(
            "plot_parameter_recovery requires keep_fits=True. Re-run SBC with keep_fits=True."
        )

    if not sbc.posteriors:
        raise ValueError(
            "No posteriors found. Run sbc.run_simulations() with keep_fits=True first."
        )

    # Validate and set defaults
    ci_prob = validate_or_use_rcparam(ci_prob, "stats.ci_prob")
    ci_kind = validate_or_use_rcparam(ci_kind, "stats.ci_kind")
    point_estimate = validate_or_use_rcparam(point_estimate, "stats.point_estimate")
    if backend is None:
        if plot_collection is None:
            backend = rcParams["plot.backend"]
        else:
            backend = plot_collection.backend
    if labeller is None:
        labeller = BaseLabeller()
    pc_kwargs = pc_kwargs or {}
    visuals = validate_dict_argument(visuals, (plot_parameter_recovery, "visuals"))
    aes_by_visuals = validate_dict_argument(
        aes_by_visuals, (plot_parameter_recovery, "aes_by_visuals")
    )

    # Resolve variable names
    all_var_names = sbc.kept_simulation_params.var_names
    if var_names is None:
        resolved_var_names = all_var_names
    elif isinstance(var_names, str):
        resolved_var_names = [var_names]
    else:
        resolved_var_names = list(var_names)
    # Validate that all requested names exist
    missing = set(resolved_var_names) - set(all_var_names)
    if missing:
        raise ValueError(
            f"var_names contains unknown variables: {sorted(missing)}. "
            f"Available: {list(all_var_names)}"
        )

    # Get backend module
    plot_bknd = import_module(f"arviz_plots.backend.{backend}")

    # Build recovery dataset
    ds = _build_recovery_dataset(
        sbc,
        ci_prob=ci_prob,
        ci_kind=ci_kind,
        point_estimate=point_estimate,
        transform=sbc._transform,
        var_names=resolved_var_names,
    )

    ref_line, points, intervals, titles = _build_plot_assets(ds, ci_prob, labeller)

    # Setup plot collection
    if plot_collection is None:
        pc_kwargs["figure_kwargs"] = pc_kwargs.get("figure_kwargs", {}).copy()
        pc_kwargs["figure_kwargs"].setdefault("dpi", 200)
        pc_kwargs["figure_kwargs"].setdefault("figsize", (16, 12))
        pc_kwargs.setdefault("cols", ["parameter"])
        pc_kwargs.setdefault("col_wrap", 4)
        pc_kwargs = set_wrap_layout(pc_kwargs, plot_bknd, ds[["recovery"]])
        pc = PlotCollection.wrap(ds, backend=backend, **pc_kwargs)
    else:
        pc = plot_collection

    # Apply visuals
    # Axis labels
    visuals_labels = get_visual_kwargs(visuals, "labels", {})
    visuals_labels.setdefault("size", 10)
    if visuals_labels is not False:
        pc.map(azp.visuals.labelled_x, text="True Value", **visuals_labels)
        pc.map(azp.visuals.labelled_y, text="Posterior Mean", **visuals_labels)

    # Reference line
    visuals_ref_line = get_visual_kwargs(visuals, "reference_line", {})
    if visuals_ref_line is not False:
        _, _, ref_line_ignore = filter_aes(pc, aes_by_visuals, "reference_line", ["simulation"])
        visuals_ref_line.setdefault("color", "B3")
        visuals_ref_line.setdefault("linestyle", "--")
        pc.map(
            azp.visuals.line_xy,
            "reference_line",
            data=ref_line,
            ignore_aes=ref_line_ignore,
            **visuals_ref_line,
        )

    # Errorbars
    visuals_errorbar = get_visual_kwargs(visuals, "errorbar", {})
    if visuals_errorbar is not False:
        _, _, errorbar_ignore = filter_aes(pc, aes_by_visuals, "errorbar", ["simulation"])
        covered_kwargs = visuals_errorbar.copy()
        uncovered_kwargs = visuals_errorbar.copy()
        covered_kwargs.setdefault("color", "C0")
        uncovered_kwargs.setdefault("color", "C3")
        pc.map(
            azp.visuals.scatter_xy,
            "points_covered",
            data=points,
            mask=ds["covered"],
            ignore_aes=errorbar_ignore,
            **covered_kwargs,
        )
        pc.map(
            azp.visuals.scatter_xy,
            "points_uncovered",
            data=points,
            mask=~ds["covered"],
            ignore_aes=errorbar_ignore,
            **uncovered_kwargs,
        )
        pc.map(
            azp.visuals.line_xy,
            "intervals_covered",
            data=intervals.where(ds["covered"]),
            ignore_aes=errorbar_ignore,
            **covered_kwargs,
        )
        pc.map(
            azp.visuals.line_xy,
            "intervals_uncovered",
            data=intervals.where(~ds["covered"]),
            ignore_aes=errorbar_ignore,
            **uncovered_kwargs,
        )

    # Tick labels
    ticklabels_kwargs = get_visual_kwargs(visuals, "ticklabels")
    if ticklabels_kwargs is not False:
        _, _, ticklabels_ignore = filter_aes(pc, aes_by_visuals, "ticklabels", ["simulation"])
        ticklabels_kwargs.setdefault("size", 10)
        pc.map(
            azp.visuals.ticklabel_props,
            "ticklabels",
            ignore_aes=ticklabels_ignore,
            axis="both",
            store_artist=backend == "none",
            **ticklabels_kwargs,
        )

    # Title
    visuals_title = get_visual_kwargs(visuals, "title", {})
    if visuals_title is not False:
        _, _, title_ignore = filter_aes(pc, aes_by_visuals, "title", ["simulation"])
        visuals_title.setdefault("color", "B1")
        visuals_title.setdefault("size", "small")

        def _labelled_title(da, target, text=None, **kwargs):
            """Stock labelled_title with scalar extraction from subsetted text."""
            if text is not None and hasattr(text, "item"):
                text = text.item()
            return azp.visuals.labelled_title(da, target, text=text, **kwargs)

        pc.map(
            _labelled_title,
            "title",
            data=ds["covered"],
            text=titles,
            ignore_aes=title_ignore,
            **visuals_title,
        )

    # Legend (figure-level)
    visuals_legend = get_visual_kwargs(visuals, "legend", {})
    if visuals_legend is not False:
        plot_bknd.legend(
            pc,
            kwarg_list=[
                {"color": "C0", "linestyle": "-", "width": 2},
                {"color": "C3", "linestyle": "-", "width": 2},
            ],
            label_list=["Covered", "Not covered"],
            title="Credible interval",
            **visuals_legend,
        )

    return pc
