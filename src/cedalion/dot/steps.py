"""Workflow-callable DOT processing steps."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import statsmodels.api as sm
import xarray as xr
from statsmodels.stats.multitest import fdrcorrection

import cedalion.io
from cedalion.dot.forward_model import ForwardModel
from cedalion.dot.head_model import (
    TwoSurfaceHeadModel,
    get_standard_headmodel,
)
from cedalion.dot.image_recon import (
    GaussianSpatialBasisFunctions,
    ImageRecon,
    estimate_alpha_meas,
)
from cedalion.io.forward_model import load_Adot
from cedalion.sigproc.quality import measurement_variance
from cedalion.vis.blocks import plot_surface
from cedalion.physunits import parse_quantity


def _load_head_model(
    head_model: str | dict,
) -> TwoSurfaceHeadModel:
    """Load a standard or construct a custom two-surface head model."""
    if isinstance(head_model, str):
        return get_standard_headmodel(head_model)

    if not isinstance(head_model, dict):
        raise TypeError(
            "head_model must be a standard model name or configuration dict."
        )

    config = head_model.copy()
    model_type = config.pop("type", None)

    if model_type == "surfaces":
        return TwoSurfaceHeadModel.from_surfaces(**config)

    if model_type == "segmentation":
        return TwoSurfaceHeadModel.from_segmentation(**config)

    raise ValueError(
        "Unsupported head model type: "
        f"{model_type!r}. Supported custom types are "
        "'surfaces' and 'segmentation'."
    )


def sensitivity(
    input_snirf: str | Path,
    output_fluence: str | Path,
    output_sensitivity: str | Path,
    head_model: str | dict = "colin27",
) -> None:
    """Compute fluence and sensitivity for a two-surface head model.

    ``head_model`` may be the name of a bundled standard head model or a
    configuration dictionary for constructing a custom model from surfaces or
    segmentation. The measurement geometry and measurement list are taken from
    the input SNIRF recording. The probe is aligned and snapped to the scalp
    before constructing the forward model.
    """
    output_fluence = Path(output_fluence)
    output_sensitivity = Path(output_sensitivity)
    output_fluence.parent.mkdir(parents=True, exist_ok=True)
    output_sensitivity.parent.mkdir(parents=True, exist_ok=True)

    rec = cedalion.io.read_snirf(input_snirf)[0]

    head = _load_head_model(head_model)
    geo3d_snapped = head.align_and_snap_to_scalp(rec.geo3d)

    fwm = ForwardModel(
        head,
        geo3d_snapped,
        rec._measurement_lists["amp"],
    )

    fwm.compute_fluence_nirfaster(output_fluence)
    fwm.compute_sensitivity(output_fluence, output_sensitivity)


def image_reconstruction(
    input_snirf: str | Path,
    input_sensitivity: str | Path,
    output_image: str | Path,
    timeseries: str = "od",
    alpha_meas: float | None = None,
    alpha_meas_k: float = 0.01,
    alpha_spatial: float | None = 0.001,
    lambda_R_conc: float | None = None,
    apply_c_meas: bool = True,
    recon_mode: str = "mua2conc",
    brain_only: bool = False,
    spatial_basis_functions: dict | None = None,
    dOD_thresh: float = 0.001,
) -> None:
    """Reconstruct a brain time series in image space.

    Measurement variance is estimated from the optical-density time series.
    Channels removed during preprocessing are also removed from the sensitivity
    matrix used for reconstruction and are treated as dropped when determining
    spatial sensitivity. If parcel labels are present, insufficiently sensitive
    parcels are removed. Head models without parcel labels are reconstructed in
    vertex space without parcel filtering.
    """
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    rec = cedalion.io.read_snirf(input_snirf)[0]
    y = rec[timeseries]

    sensitivity = load_Adot(input_sensitivity)

    # Preserve stable head-model vertex identifiers through reconstruction and
    # subsequent brain/parcel filtering.
    if "vertex" not in sensitivity.coords:
        sensitivity = sensitivity.assign_coords(
            vertex=range(sensitivity.sizes["vertex"])
        )

    # Preprocessing may physically remove bad channels. Determine those channels
    # from the difference between the forward model and the retained OD data.
    retained_channels = set(y.channel.values)
    dropped_channels = [
        channel
        for channel in sensitivity.channel.values
        if channel not in retained_channels
    ]

    # Determine which parcels remain sufficiently sensitive after channel
    # pruning when the head model provides a parcellation. Individualized head
    # models without parcel labels can still be reconstructed in vertex space.
    sensitive_parcels = None
    if "parcel" in sensitivity.coords:
        _, parcel_mask = ForwardModel.parcel_sensitivity(
            sensitivity,
            chan_droplist=dropped_channels,
            dOD_thresh=dOD_thresh,
        )
        sensitive_parcels = (
            parcel_mask.where(parcel_mask, drop=True).parcel.values.tolist()
        )

        if not sensitive_parcels:
            raise ValueError("No sufficiently sensitive parcels remain.")

    # Keep the sensitivity matrix and measurement data on exactly the same
    # channel axis. The sensitivity ordering is used for both.
    common_channels = [
        channel
        for channel in sensitivity.channel.values
        if channel in retained_channels
    ]

    if not common_channels:
        raise ValueError(
            "No channels are shared by the optical-density data and "
            "the sensitivity matrix."
        )

    sensitivity_recon = sensitivity.sel(channel=common_channels)
    y = y.sel(channel=common_channels)

    # ImageRecon expects the diagonal measurement variance for this path.
    c_meas = measurement_variance(y, calc_covariance=False)

    if alpha_meas is None:
        alpha_meas = float(
            estimate_alpha_meas(
                c_meas.pint.dequantify().values,
                K=alpha_meas_k,
            )
        )

    sbf = None
    if spatial_basis_functions is not None:
        sbf_config = spatial_basis_functions.copy()
        sbf_type = sbf_config.pop("type", None)

        if sbf_type != "gaussian":
            raise ValueError(
                "Unsupported spatial basis function type: "
                f"{sbf_type!r}. Currently supported: 'gaussian'."
            )

        head_model = sbf_config.pop("head_model")
        head_ijk = _load_head_model(head_model)
        head = head_ijk.apply_transform(head_ijk.t_ijk2ras)

        for key in (
            "threshold_brain",
            "threshold_scalp",
            "sigma_brain",
            "sigma_scalp",
        ):
            sbf_config[key] = parse_quantity(sbf_config[key])

        sbf = GaussianSpatialBasisFunctions(
            head,
            sensitivity_recon,
            **sbf_config,
        )

    recon = ImageRecon(
        sensitivity_recon,
        brain_only=brain_only,
        recon_mode=recon_mode,
        spatial_basis_functions=sbf,
        alpha_meas=alpha_meas,
        alpha_spatial=alpha_spatial,
        lambda_R_conc=lambda_R_conc,
        apply_c_meas=apply_c_meas,
    )

    result = recon.reconstruct(y, c_meas=c_meas)

    result = result.where(result.is_brain, drop=True)
    if sensitive_parcels is not None:
        result = result.where(
            result.parcel.isin(sensitive_parcels),
            drop=True,
        )

    result = result.pint.dequantify()
    if "units" in y.time.attrs:
        result.time.attrs["units"] = str(y.time.attrs["units"])
    result.to_netcdf(output_image)

def image_blockaverage(
    input_image: str | Path,
    input_snirf: str | Path,
    output_image: str | Path,
    t_pre,
    t_post,
    trial_types: list[str] | None = None,
) -> None:
    """Calculate a baseline-corrected block average in DOT image space.

    The reconstructed image supplies the chromophore-by-vertex time series.
    Stimulus timing is read from the corresponding SNIRF recording.

    Args:
        input_image: Reconstructed image-space NetCDF file.
        input_snirf: SNIRF file containing stimulus timing.
        output_image: Output NetCDF containing the image-space block average.
        t_pre: Time before stimulus onset included in each epoch.
        t_post: Time after stimulus onset included in each epoch.
        trial_types: Trial types to include. If None, use all available types.
    """
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    t_pre = parse_quantity(t_pre)
    t_post = parse_quantity(t_post)

    rec = cedalion.io.read_snirf(input_snirf)[0]
    image = xr.load_dataarray(input_image)

    time_units = image.time.attrs.get("units")
    if time_units is None:
        time_units = rec["amp"].time.attrs.get("units")
        if time_units is not None:
            image.time.attrs["units"] = str(time_units)

    if time_units is None:
        raise ValueError(
            "No time units found in either the reconstructed image "
            "or the input SNIRF recording."
        )

    image = image.pint.quantify()

    available_trial_types = set(rec.stim.trial_type)

    if trial_types is None:
        selected_trials = sorted(available_trial_types)
    else:
        missing_trial_types = sorted(
            set(trial_types) - available_trial_types
        )
        if missing_trial_types:
            raise ValueError(
                "Requested trial types not found in input recording: "
                f"{missing_trial_types}"
            )
        selected_trials = sorted(set(trial_types))

    epochs = image.cd.to_epochs(
        rec.stim,
        selected_trials,
        before=t_pre,
        after=t_post,
    )

    baseline = epochs.sel(
        reltime=(epochs.reltime < 0)
    ).mean("reltime")
    epochs = epochs - baseline

    blockaverage = epochs.groupby("trial_type").mean("epoch")

    blockaverage = blockaverage.pint.dequantify()
    blockaverage.reltime.attrs["units"] = str(time_units)
    blockaverage.to_netcdf(output_image)



def image_parcel_average(
    input_image: str | Path,
    output_image: str | Path,
) -> None:
    """Average image-space block averages within anatomical parcels.

    Vertices are grouped by their ``parcel`` coordinate. All other dimensions,
    such as trial type, chromophore, and relative time, are preserved.

    Args:
        input_image: Image-space block-average NetCDF file.
        output_image: Output NetCDF containing parcel-averaged time courses.
    """
    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    image = xr.load_dataarray(input_image)

    if "vertex" not in image.dims:
        raise ValueError("Input image does not contain a 'vertex' dimension.")

    if "parcel" not in image.coords:
        raise ValueError("Input image does not contain a 'parcel' coordinate.")

    parcel_average = image.groupby("parcel").mean("vertex")
    parcel_average.to_netcdf(output_image)


def image_group_average(
    input_images: list[str | Path],
    output_image: str | Path,
) -> None:
    """Average parcel-level image-space responses across input images.

    Parcel coordinates are aligned by label using their union. Parcels missing
    from an input are represented as NaN and therefore do not contribute to
    that parcel's mean. The output also stores the number of contributing
    inputs for every data point.

    Args:
        input_images: Parcel-average NetCDF files to combine.
        output_image: Output NetCDF containing group mean and sample counts.
    """
    if not input_images:
        raise ValueError("At least one input image is required.")

    output_image = Path(output_image)
    output_image.parent.mkdir(parents=True, exist_ok=True)

    images = [xr.load_dataarray(fname) for fname in input_images]

    for image in images:
        if "parcel" not in image.dims:
            raise ValueError(
                "All input images must contain a 'parcel' dimension."
            )

    reference = images[0]
    reference_dims = tuple(
        dim for dim in reference.dims if dim != "parcel"
    )

    for image in images[1:]:
        image_dims = tuple(
            dim for dim in image.dims if dim != "parcel"
        )
        if image_dims != reference_dims:
            raise ValueError(
                "All input images must have the same non-parcel dimensions."
            )

        for dim in reference_dims:
            if not image[dim].equals(reference[dim]):
                raise ValueError(
                    f"Input images have mismatched '{dim}' coordinates."
                )

    aligned = xr.align(*images, join="outer")
    stacked = xr.concat(aligned, dim="run")

    group_mean = stacked.mean("run", skipna=True)
    sample_count = stacked.notnull().sum("run")

    squared_deviation_sum = (
        (stacked - group_mean) ** 2
    ).sum("run", skipna=True)

    group_sem = (
        squared_deviation_sum
        / (sample_count * (sample_count - 1))
    ) ** 0.5
    group_sem = group_sem.where(sample_count >= 2)

    group_mean.attrs = images[0].attrs.copy()
    group_sem.attrs = images[0].attrs.copy()

    if "reltime" in group_mean.coords:
        group_mean.reltime.attrs = images[0].reltime.attrs.copy()

    result = xr.Dataset(
        {
            "mean": group_mean,
            "sem": group_sem,
            "n": sample_count,
        }
    )

    result.to_netcdf(output_image)



def image_group_average_visualization(
    input_image: str | Path,
    output_figure: str | Path,
    trial_type: str,
) -> None:
    """Plot group-average parcel responses as compact heatmaps.

    Parcels are sorted by peak absolute post-stimulus HbO response. Individual
    parcel labels are hidden so the figure remains readable when many parcels
    are present.

    Args:
        input_image: Group-average NetCDF file produced by
            :func:`image_group_average`.
        output_figure: Output PNG filename.
        trial_type: Trial type to visualize.
    """
    output_figure = Path(output_figure)
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    group = xr.load_dataset(input_image)

    if "mean" not in group:
        raise ValueError("Group-average input must contain a 'mean' variable.")

    mean = group["mean"]

    required_dims = {"trial_type", "chromo", "parcel", "reltime"}
    missing_dims = required_dims - set(mean.dims)
    if missing_dims:
        raise ValueError(
            "Group-average mean is missing required dimensions: "
            f"{sorted(missing_dims)}"
        )

    if trial_type not in mean.trial_type.values:
        raise ValueError(
            f"Trial type {trial_type!r} is not present in group average."
        )

    mean = mean.sel(trial_type=trial_type)

    available_chromos = mean.chromo.values.tolist()
    chromos = [
        chromo
        for chromo in ("HbO", "HbR")
        if chromo in available_chromos
    ]
    if not chromos:
        chromos = available_chromos

    sort_chromo = "HbO" if "HbO" in available_chromos else chromos[0]
    sort_data = mean.sel(chromo=sort_chromo).transpose(
        "parcel",
        "reltime",
    )

    if np.any(sort_data.reltime.values >= 0):
        sort_data = sort_data.sel(reltime=slice(0, None))

    sort_metric = np.nanmax(
        np.abs(sort_data.values),
        axis=1,
    )
    sort_metric = np.where(
        np.isfinite(sort_metric),
        sort_metric,
        -np.inf,
    )
    order = np.argsort(sort_metric)[::-1]

    mean = mean.isel(parcel=order)
    n_parcels = mean.sizes["parcel"]

    finite = mean.values[np.isfinite(mean.values)]
    if finite.size:
        limit = float(np.nanpercentile(np.abs(finite), 99))
        if not np.isfinite(limit) or limit == 0:
            limit = float(np.nanmax(np.abs(finite)))
    else:
        limit = 1.0

    if not np.isfinite(limit) or limit == 0:
        limit = 1.0

    reltime = np.asarray(mean.reltime.values, dtype=float)

    fig, axes = plt.subplots(
        len(chromos),
        1,
        figsize=(10, max(5.0, 2.5 * len(chromos))),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )

    image_artist = None

    for ax, chromo in zip(axes[:, 0], chromos, strict=True):
        values = np.asarray(
            mean.sel(chromo=chromo)
            .transpose("parcel", "reltime")
            .values,
            dtype=float,
        )

        image_artist = ax.imshow(
            values,
            aspect="auto",
            origin="upper",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            extent=(
                float(reltime[0]),
                float(reltime[-1]),
                n_parcels,
                0,
            ),
        )

        ax.axvline(0.0, linestyle="--", linewidth=1.5)
        ax.set_title(str(chromo))
        ax.set_ylabel(f"Parcels (n={n_parcels})")
        ax.set_yticks([])

    time_units = mean.reltime.attrs.get("units")
    xlabel = "Relative time"
    if time_units is not None:
        xlabel += f" ({time_units})"
    axes[-1, 0].set_xlabel(xlabel)

    fig.suptitle(f"Group average: {trial_type}")

    if image_artist is not None:
        data_units = mean.attrs.get("units")
        colorbar = fig.colorbar(
            image_artist,
            ax=axes[:, 0].tolist(),
            shrink=0.9,
            pad=0.02,
        )

        label = "Group mean"
        if data_units is not None:
            label += f" ({data_units})"
        colorbar.set_label(label)

    fig.savefig(output_figure, dpi=150, bbox_inches="tight")
    plt.close(fig)



def image_feature(
    input_image: str | Path,
    input_snirf: str | Path,
    output_feature: str | Path,
    t_pre,
    t_post,
    feature: str,
    baseline_window: list,
    activity_window: list,
    trial_types: list[str] | None = None,
) -> None:
    """Extract epoch-level image-space parcel features.

    Epochs are baseline-corrected before feature extraction. Individual epochs
    are preserved so downstream statistical tests can operate across trials.

    Currently supported features:
        ``auc_diff``: AUC(HbO - HbR) in the activity window minus the
        corresponding AUC in the baseline window.
    """
    output_feature = Path(output_feature)
    output_feature.parent.mkdir(parents=True, exist_ok=True)

    t_pre = parse_quantity(t_pre)
    t_post = parse_quantity(t_post)

    rec = cedalion.io.read_snirf(input_snirf)[0]
    image = xr.load_dataarray(input_image)

    time_units = image.time.attrs.get("units")
    if time_units is None:
        time_units = rec["amp"].time.attrs.get("units")
        if time_units is not None:
            image.time.attrs["units"] = str(time_units)

    if time_units is None:
        raise ValueError(
            "No time units found in either the reconstructed image "
            "or the input SNIRF recording."
        )

    if "vertex" not in image.dims:
        raise ValueError("Input image does not contain a 'vertex' dimension.")

    if "parcel" not in image.coords:
        raise ValueError("Input image does not contain a 'parcel' coordinate.")

    image = image.pint.quantify()

    # Reduce to parcels before epoching to avoid constructing very large
    # epoch-by-vertex arrays.
    image = image.groupby("parcel").mean("vertex")

    available_trial_types = set(rec.stim.trial_type)

    if trial_types is None:
        selected_trials = sorted(available_trial_types)
    else:
        missing_trial_types = sorted(
            set(trial_types) - available_trial_types
        )
        if missing_trial_types:
            raise ValueError(
                "Requested trial types not found in input recording: "
                f"{missing_trial_types}"
            )
        selected_trials = sorted(set(trial_types))

    epochs = image.cd.to_epochs(
        rec.stim,
        selected_trials,
        before=t_pre,
        after=t_post,
    )

    # Baseline-correct each epoch using the pre-stimulus interval.
    baseline = epochs.sel(
        reltime=(epochs.reltime < 0)
    ).mean("reltime")
    epochs = epochs - baseline

    baseline_start = parse_quantity(baseline_window[0]).to(time_units).magnitude
    baseline_stop = parse_quantity(baseline_window[1]).to(time_units).magnitude
    activity_start = parse_quantity(activity_window[0]).to(time_units).magnitude
    activity_stop = parse_quantity(activity_window[1]).to(time_units).magnitude

    if feature != "auc_diff":
        raise ValueError(
            f"Unsupported image feature: {feature!r}. "
            "Currently supported: 'auc_diff'."
        )

    required_chromos = {"HbO", "HbR"}
    if "chromo" not in epochs.dims or not required_chromos.issubset(
        set(epochs.chromo.values)
    ):
        raise ValueError(
            "Feature 'auc_diff' requires HbO and HbR chromophores."
        )

    chromo_diff = epochs.sel(chromo="HbO") - epochs.sel(chromo="HbR")
    chromo_diff = chromo_diff.pint.dequantify()

    baseline_auc = chromo_diff.sel(
        reltime=slice(baseline_start, baseline_stop)
    ).integrate("reltime")

    activity_auc = chromo_diff.sel(
        reltime=slice(activity_start, activity_stop)
    ).integrate("reltime")

    result = activity_auc - baseline_auc

    data_units = chromo_diff.attrs.get("units")
    if data_units is not None:
        result.attrs["units"] = f"{data_units} * {time_units}"

    result.to_netcdf(output_feature)



def image_statistics(
    input_feature: str | Path,
    output_statistics: str | Path,
    alpha: float = 0.05,
    fdr_method: str = "indep",
) -> None:
    """Run epoch-level one-sample tests and parcel-wise FDR correction.

    For every trial type and parcel, an intercept-only OLS model tests whether
    the mean epoch-level feature differs from zero. Benjamini-Hochberg FDR
    correction is then applied across parcels separately for each trial type.

    Args:
        input_feature: Epoch-level parcel feature NetCDF file.
        output_statistics: Output NetCDF containing t-values, raw p-values,
            FDR-adjusted p-values, rejection masks, and valid epoch counts.
        alpha: Target false-discovery rate.
        fdr_method: Method passed to ``statsmodels.stats.multitest.fdrcorrection``.
            ``"indep"`` gives Benjamini-Hochberg correction and ``"negcorr"``
            gives Benjamini-Yekutieli correction.
    """
    output_statistics = Path(output_statistics)
    output_statistics.parent.mkdir(parents=True, exist_ok=True)

    feature = xr.load_dataarray(input_feature)

    if "epoch" not in feature.dims:
        raise ValueError("Input feature must contain an 'epoch' dimension.")

    if "parcel" not in feature.dims:
        raise ValueError("Input feature must contain a 'parcel' dimension.")

    if (
        "trial_type" not in feature.coords
        or feature.trial_type.dims != ("epoch",)
    ):
        raise ValueError(
            "Input feature must contain a 'trial_type' coordinate "
            "on the 'epoch' dimension."
        )

    if fdr_method not in {"indep", "negcorr"}:
        raise ValueError(
            "fdr_method must be either 'indep' or 'negcorr'."
        )

    # Preserve the order in which conditions occur in the input.
    trial_types = list(
        dict.fromkeys(feature.trial_type.values.tolist())
    )

    coords = {
        "trial_type": trial_types,
        "parcel": feature.parcel.values,
    }
    shape = (len(trial_types), feature.sizes["parcel"])

    tvals = xr.DataArray(
        np.full(shape, np.nan, dtype=float),
        dims=("trial_type", "parcel"),
        coords=coords,
        name="t",
    )
    pvals = xr.DataArray(
        np.full(shape, np.nan, dtype=float),
        dims=("trial_type", "parcel"),
        coords=coords,
        name="p",
    )
    sample_count = xr.DataArray(
        np.zeros(shape, dtype=int),
        dims=("trial_type", "parcel"),
        coords=coords,
        name="n",
    )

    for trial_type in trial_types:
        condition = feature.where(
            feature.trial_type == trial_type,
            drop=True,
        )

        for parcel in feature.parcel.values:
            values = (
                condition.sel(parcel=parcel)
                .values
                .astype(float)
            )
            values = values[np.isfinite(values)]

            n_valid = values.size
            sample_count.loc[
                dict(trial_type=trial_type, parcel=parcel)
            ] = n_valid

            if n_valid < 2:
                continue

            # Perform an intercept-only OLS one-sample test:
            # feature_i = beta_0 + error, H0: beta_0 == 0.
            design = np.ones((n_valid, 1), dtype=float)
            fit = sm.OLS(values, design).fit()

            tvals.loc[
                dict(trial_type=trial_type, parcel=parcel)
            ] = fit.tvalues[0]
            pvals.loc[
                dict(trial_type=trial_type, parcel=parcel)
            ] = fit.pvalues[0]

    pvals_fdr = xr.full_like(pvals, np.nan, dtype=float)
    rejected = xr.full_like(pvals, False, dtype=bool)
    pvals_fdr.name = "p_fdr"
    rejected.name = "rejected"

    # Treat each trial type as a separate family of parcel-wise tests.
    for trial_type in trial_types:
        values = pvals.sel(trial_type=trial_type).values
        valid = np.isfinite(values)

        if not np.any(valid):
            continue

        reject_valid, adjusted_valid = fdrcorrection(
            values[valid],
            alpha=alpha,
            method=fdr_method,
        )

        pvals_fdr.loc[
            dict(trial_type=trial_type)
        ].values[valid] = adjusted_valid
        rejected.loc[
            dict(trial_type=trial_type)
        ].values[valid] = reject_valid

    result = xr.Dataset(
        {
            "t": tvals,
            "p": pvals,
            "p_fdr": pvals_fdr,
            "rejected": rejected,
            "n": sample_count,
        }
    )
    result.attrs["alpha"] = alpha
    result.attrs["fdr_method"] = fdr_method

    result.to_netcdf(output_statistics)


def _parcel_statistics_to_vertex_map(
    statistics: xr.Dataset,
    image: xr.DataArray,
    nvertices: int,
    trial_type: str,
) -> np.ndarray:
    """Map significant parcel t-values onto reconstructed image vertices.

    Vertices that were not reconstructed, belong to non-significant parcels,
    or have non-finite t-values remain NaN.
    """
    if "t" not in statistics or "rejected" not in statistics:
        raise ValueError(
            "Statistics must contain 't' and 'rejected' variables."
        )

    if "trial_type" not in statistics.coords:
        raise ValueError(
            "Statistics must contain a 'trial_type' coordinate."
        )

    if trial_type not in statistics.trial_type.values:
        raise ValueError(
            f"Trial type {trial_type!r} is not present in statistics."
        )

    if "vertex" not in image.dims or "parcel" not in image.coords:
        raise ValueError(
            "Image must contain a 'vertex' dimension and parcel coordinate."
        )

    tvals = statistics["t"].sel(trial_type=trial_type)
    rejected = statistics["rejected"].sel(trial_type=trial_type)

    significant = {}
    for parcel in statistics.parcel.values:
        tval = float(tvals.sel(parcel=parcel))
        is_rejected = bool(rejected.sel(parcel=parcel))

        if is_rejected and np.isfinite(tval):
            significant[parcel] = tval

    vertex_map = np.full(nvertices, np.nan, dtype=float)

    for vertex, parcel in zip(
        image.vertex.values,
        image.parcel.values,
        strict=True,
    ):
        vertex = int(vertex)

        if vertex < 0 or vertex >= nvertices:
            raise ValueError(
                f"Vertex index {vertex} is outside the surface range "
                f"0..{nvertices - 1}."
            )

        if parcel in significant:
            vertex_map[vertex] = significant[parcel]

    return vertex_map


def image_visualization(
    input_statistics: str | Path,
    input_image: str | Path,
    output_figure: str | Path,
    head_model: str | dict = "colin27",
    trial_type: str = "motor",
) -> None:
    """Plot FDR-significant parcel t-values on the reconstruction surface.

    Parcel-level t-statistics are expanded onto the reconstructed vertices
    using the parcel coordinate stored in the image reconstruction. Vertices
    outside the reconstruction and vertices belonging to parcels that do not
    survive FDR correction are shown as NaN.

    Args:
        input_statistics: Parcel-level statistics NetCDF file.
        input_image: Image-reconstruction NetCDF file containing vertex IDs
            and parcel labels.
        output_figure: Output PNG filename.
        head_model: Standard model name or custom head-model configuration
            used for reconstruction.
        trial_type: Trial type to visualize.
    """
    output_figure = Path(output_figure)
    output_figure.parent.mkdir(parents=True, exist_ok=True)

    statistics = xr.load_dataset(input_statistics)
    image = xr.load_dataarray(input_image)
    head = _load_head_model(head_model)

    vertex_map = _parcel_statistics_to_vertex_map(
        statistics=statistics,
        image=image,
        nvertices=head.brain.nvertices,
        trial_type=trial_type,
    )

    finite = vertex_map[np.isfinite(vertex_map)]
    if finite.size:
        limit = float(np.max(np.abs(finite)))
        if limit == 0:
            limit = 1.0
    else:
        limit = 1.0

    plotter = pv.Plotter(
        shape=(2, 3),
        off_screen=True,
        window_size=(1200, 800),
    )

    vertices = np.asarray(
        head.brain.vertices.pint.dequantify().values,
        dtype=float,
    )
    centroid = vertices.mean(axis=0)
    extent = np.ptp(vertices, axis=0)
    distance = max(float(extent.max()) * 4.0, 1.0)

    views = {
        "left": ((0, 0), np.array([-1.0, 0.0, 0.0])),
        "superior": ((0, 1), np.array([0.0, 0.0, 1.0])),
        "right": ((0, 2), np.array([1.0, 0.0, 0.0])),
        "anterior": ((1, 0), np.array([0.0, 1.0, 0.0])),
        "posterior": ((1, 2), np.array([0.0, -1.0, 0.0])),
    }

    for view, (subplot, direction) in views.items():
        plotter.subplot(*subplot)

        plot_surface(
            plotter,
            head.brain,
            color=vertex_map,
            cmap="seismic",
            clim=(-limit, limit),
            nan_color=(0.9, 0.9, 0.9),
            show_scalar_bar=(view == "superior"),
            scalar_bar_args={"title": "t-value"},
            pickable=False,
        )

        view_up = (
            np.array([0.0, 1.0, 0.0])
            if view == "superior"
            else np.array([0.0, 0.0, 1.0])
        )
        plotter.camera_position = [
            centroid + direction * distance,
            centroid,
            view_up,
        ]
        plotter.add_text(view, position="lower_left", font_size=10)

    plotter.subplot(1, 1)
    alpha = statistics.attrs.get("alpha")
    title = f"{trial_type}: FDR-significant parcel t-values"
    if alpha is not None:
        title += f"\nalpha={alpha}"
    plotter.add_text(title, position="upper_left", font_size=12)

    plotter.screenshot(str(output_figure))
    plotter.close()
