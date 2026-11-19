# src/cedalion/vis/dqr.py

import json
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.axes import Axes

import cedalion.io
import cedalion.typing as cdt
from cedalion import Quantity
from cedalion.geometry.landmarks import normalize_landmarks_labels
from cedalion.vis.anatomy.scalp_plot import scalp_plot
from cedalion.vis.blocks import plot_stim_markers


# Keep histogram binning consistent with _get_gvtd_threshold.
_MIN_COUNTS_PER_BIN = 5


_PRUNE_COLORS = [
    "cyan",
    "blue",
    (1.0, 0.9, 0.4),
    (0.3, 1.0, 0.3),
    "magenta",
    "red",
]

_PRUNE_TICKS = [
    (0.08, "SDS"),
    (0.24, "Low Signal"),
    (0.40, "Poor SNR"),
    (0.58, "Good SNR"),
    (0.76, "SCI/PSP"),
    (0.92, "Saturated"),
]

# Sidecar pruning reason codes:
#
# 0 = good
# 1 = source-detector distance
# 2 = low signal
# 3 = poor SNR
# 4 = SCI/PSP
# 5 = saturated
#
# These display values reproduce the categorical locations used by
# the legacy DQR colorbar.
_PRUNE_DISPLAY_VALUE = {
    0: 0.58,
    1: 0.08,
    2: 0.24,
    3: 0.40,
    4: 0.76,
    5: 0.92,
}


def _as_magnitude(
    value: float | Quantity | xr.DataArray | None,
    units: str = "1/s",
) -> float | None:
    """Return a plain scalar magnitude in the requested units."""
    if value is None:
        return None

    if isinstance(value, xr.DataArray):
        if value.size != 1:
            raise ValueError(f"value must be scalar, got shape {value.shape}.")

        value = value.item()

    if isinstance(value, Quantity):
        return float(value.to(units).magnitude)

    return float(value)


def plot_gvtd(
    gvtd: cdt.NDTimeSeries,
    ax: Axes,
    threshold: float | Quantity | xr.DataArray | None = None,
    label: str = "GVTD",
    ylim_factor: float | None = None,
    threshold_color: str = "C1",
    color: str | None = None,
) -> None:
    """Plot a GVTD time trace on an existing axes.

    Args:
        gvtd: One-dimensional GVTD time series.
        ax: Matplotlib axes on which to draw the GVTD trace.
        threshold: Optional GVTD threshold. May be a float or a unit-aware quantity.
        label: Label for the GVTD trace.
        ylim_factor: Optional factor used to set the upper y-axis limit relative
        to the threshold.
        threshold_color: Color used for the threshold line.
        color: Optional color used for the GVTD trace.
    """
    if "time" not in gvtd.dims:
        raise ValueError(f"gvtd must have a 'time' dimension, got dims {gvtd.dims}.")

    if gvtd.ndim != 1:
        raise ValueError(f"gvtd must be one-dimensional, got dims {gvtd.dims}.")

    time = gvtd.coords["time"]

    ax.plot(time, gvtd, label=label, color=color)

    if threshold is not None:
        threshold = _as_magnitude(threshold)

        ax.axhline(
            threshold,
            linestyle="--",
            color=threshold_color,
            label=f"{label} threshold",
        )

        if ylim_factor is not None:
            ax.set_ylim(0, ylim_factor * threshold)

    # TODO: derive axis-label units from GVTD/time metadata instead of hard-coding them.
    ax.set_xlabel("time / s")
    ax.set_ylabel("GVTD / (1/s)")


def plot_gvtd_histogram(
    gvtd: cdt.NDTimeSeries,
    ax: Axes,
    bins: int | None = None,
    threshold: float | Quantity | xr.DataArray | None = None,
    label: str = "GVTD",
    threshold_color: str = "C1",
) -> None:
    """Plot a histogram of GVTD values.

    Args:
        gvtd: One-dimensional GVTD time series.
        ax: Matplotlib axes on which to draw the histogram.
        bins: Number of histogram bins. If None, use the legacy adaptive rule.
        threshold: Optional GVTD threshold. May be a float or a unit-aware quantity.
        label: Label used for the GVTD threshold line.
        threshold_color: Color used for the threshold line.
    """
    if "time" not in gvtd.dims:
        raise ValueError(f"gvtd must have a 'time' dimension, got dims {gvtd.dims}.")

    if gvtd.ndim != 1:
        raise ValueError(f"gvtd must be one-dimensional, got dims {gvtd.dims}.")

    values = gvtd.pint.dequantify().values
    finite_values = values[np.isfinite(values)]

    if bins is None:
        n_bins = max(
            1,
            round(len(values) / _MIN_COUNTS_PER_BIN),
        )

        if finite_values.size == 0:
            bins = np.array([0.0, 1.0])
        else:
            max_value = np.max(finite_values)

            if max_value > 0:
                bin_size = max_value / n_bins
                bins = np.arange(
                    0,
                    max_value + bin_size,
                    bin_size,
                )
            else:
                bins = np.array([0.0, 1.0])

    ax.hist(
        finite_values,
        bins=bins,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.3,
    )

    if threshold is not None:
        threshold = _as_magnitude(threshold)

        ax.axvline(
            threshold,
            linestyle="--",
            color=threshold_color,
            label=f"{label} threshold",
        )

    # TODO: derive axis-label units from GVTD/time metadata instead of hard-coding them.
    ax.set_xlabel("GVTD / (1/s)")
    ax.set_ylabel("count")


def plot_channel_metric(
    ts: cdt.NDTimeSeries,
    geo3d: cdt.LabeledPoints,
    metric: xr.DataArray,
    ax: Axes,
    wavelength: float | None = None,
    title: str | None = None,
    **kwargs,
) -> None:
    """Plot a channel-wise metric on the scalp.

    Args:
        ts: Time series providing channel definitions.
        geo3d: Probe geometry.
        metric: Channel-wise metric to plot.
        ax: Matplotlib axes on which to draw the scalp plot.
        wavelength: Optional wavelength to select from the metric.
        title: Optional axes title.
        **kwargs: Additional keyword arguments passed to scalp_plot.
    """
    if "channel" not in metric.dims:
        raise ValueError(
            f"metric must have a 'channel' dimension, got dims {metric.dims}."
        )

    if wavelength is not None:
        if "wavelength" not in metric.dims:
            raise ValueError(
                "wavelength was provided, but metric has no 'wavelength' dimension."
            )

        if wavelength not in metric.wavelength.values:
            raise ValueError(
                f"wavelength {wavelength} not in metric, "
                f"available: {list(metric.wavelength.values)}."
            )

        metric = metric.sel(wavelength=wavelength)

    if metric.ndim != 1:
        raise ValueError(
            "metric must be one-dimensional after wavelength selection, "
            f"got dims {metric.dims}."
        )

    scalp_plot(
        ts,
        geo3d,
        metric,
        ax,
        title=title,
        **kwargs,
    )





def _require_variables(
    sidecar: xr.Dataset,
    names: list[str],
) -> None:
    """Require variables needed to construct the DQR."""
    missing = [name for name in names if name not in sidecar]

    if missing:
        raise ValueError(
            "DQR sidecar is missing required variables: " + ", ".join(missing)
        )


def _reference_timeseries(rec):
    """Return a time series carrying the full channel definition."""
    if "amp" in rec.timeseries:
        return rec["amp"]

    for ts in rec.timeseries.values():
        if "channel" in ts.dims and "wavelength" in ts.dims:
            return ts

    raise ValueError(
        "The DQR input recording contains no time series "
        "with both 'channel' and 'wavelength' dimensions."
    )


def _prune_display_metric(
    prune_reason: xr.DataArray,
) -> xr.DataArray:
    """Convert pruning reason codes to the legacy display scale."""
    if prune_reason.dims != ("channel",):
        raise ValueError(
            "prune_reason must have exactly the "
            "('channel',) dimension, "
            f"got {prune_reason.dims}."
        )

    reason_values = np.asarray(prune_reason.values)

    values = np.full(
        prune_reason.shape,
        np.nan,
        dtype=float,
    )

    for code, display_value in _PRUNE_DISPLAY_VALUE.items():
        values[reason_values == code] = display_value

    unknown = np.isfinite(reason_values) & ~np.isin(
        reason_values,
        list(_PRUNE_DISPLAY_VALUE),
    )

    if np.any(unknown):
        codes = np.unique(reason_values[unknown])

        raise ValueError(f"Unknown pruning reason codes: {codes.tolist()}.")

    return xr.DataArray(
        values,
        dims=("channel",),
        coords={
            "channel": prune_reason.channel,
        },
        name="pruning_status",
    )


def _finite_range(
    metric: xr.DataArray,
) -> tuple[float, float]:
    """Return the finite global minimum and maximum of a metric."""
    values = np.asarray(
        metric.values,
        dtype=float,
    )

    finite = values[np.isfinite(values)]

    if finite.size == 0:
        raise ValueError(f"Metric {metric.name!r} contains no finite values.")

    vmin = float(finite.min())
    vmax = float(finite.max())

    if vmin == vmax:
        delta = 1.0 if vmin == 0 else abs(vmin) * 0.01

        vmin -= delta
        vmax += delta

    return vmin, vmax


def _preprocess_config_from_sidecar(
    sidecar: xr.Dataset,
) -> dict:
    """Return preprocessing configuration stored in sidecar provenance."""
    preprocess_json = sidecar.attrs.get("preprocess")

    if preprocess_json is None:
        raise ValueError(
            "DQR sidecar is missing preprocessing configuration metadata."
        )

    try:
        preprocess_config = json.loads(preprocess_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "DQR sidecar contains invalid preprocessing configuration metadata."
        ) from exc

    if not isinstance(preprocess_config, dict):
        raise ValueError(
            "DQR sidecar preprocessing configuration metadata must be a JSON object."
        )

    return preprocess_config


def _snr_threshold_from_preprocess_config(
    preprocess_config: dict,
) -> float | None:
    """Return the configured prune SNR threshold."""
    prune_steps = [
        step
        for step in preprocess_config.get("steps", [])
        if step.get("method") == "prune"
    ]

    if len(prune_steps) != 1:
        raise ValueError(
            "DQR requires exactly one prune step in the sidecar preprocessing "
            f"configuration, found {len(prune_steps)}."
        )

    snr_thresh = prune_steps[0].get("params", {}).get("snr_thresh")

    if snr_thresh is None:
        return None

    return float(snr_thresh)


def generate_dqr_plots(
    input_snirf: Path | str,
    input_sidecar: Path | str,
    output_dqr: Path | str,
    output_gvtd_histogram: Path | str,
    *,
    output_channel_quality: Path | str | None = None,
    gvtd_before: str = "gvtd_before",
    gvtd_after: str = "gvtd_after",
    prune_reason: str = "amp_pruned_reason",
    snr: str = "amp_pruned_snr",
    snr_mask: str = "amp_pruned_snr_mask",
    od_variance: str = "od_variance_corrected",
    sd_dist: str = "amp_pruned_sd_dist",
    time_clean_fraction: str = "amp_pruned_time_clean_fraction",
    mean_amp: str = "amp_pruned_mean_amp",
    min_dist: Quantity | None = None,
    title: str | None = None,
) -> None:
    """Generate the Cedalion data-quality report.

    The report combines recording metadata and geometry with
    diagnostic quantities stored during preprocessing.

    Args:
        input_snirf: SNIRF file providing channel definitions,
            geometry, wavelengths, and stimulus information.
        input_sidecar: Preprocessing sidecar NetCDF file.
        output_dqr: Output path for the main DQR figure.
        output_gvtd_histogram: Output path for the GVTD
            histogram comparison.
        output_channel_quality: Optional output path for the
            channel-quality diagnostics figure.
        gvtd_before: Sidecar variable containing pre-correction
            GVTD.
        gvtd_after: Sidecar variable containing post-correction
            GVTD.
        prune_reason: Sidecar variable containing pruning reason
            codes.
        snr: Sidecar variable containing channel SNR.
        snr_mask: Sidecar variable containing the SNR mask.
        od_variance: Sidecar variable containing corrected OD
            log variance.
        sd_dist: Sidecar variable containing source-detector
            distance.
        time_clean_fraction: Sidecar variable containing the
            fraction of clean coupling windows per channel.
        mean_amp: Sidecar variable containing mean amplitude.
        min_dist: Optional minimum source-detector distance to
            display.
        title: Optional report title.
    """
    input_snirf = Path(input_snirf)
    input_sidecar = Path(input_sidecar)
    output_dqr = Path(output_dqr)
    output_gvtd_histogram = Path(output_gvtd_histogram)

    if output_channel_quality is not None:
        output_channel_quality = Path(output_channel_quality)

    records = cedalion.io.read_snirf(
        input_snirf,
        time_units="second",
    )

    if not records:
        raise ValueError(f"No recordings found in {input_snirf}.")

    rec = records[0]

    reference_ts = _reference_timeseries(rec)

    with xr.open_dataset(input_sidecar) as opened:
        sidecar = opened.load()

    preprocess_config = _preprocess_config_from_sidecar(sidecar)

    if preprocess_config.get("normalize_landmarks", False):
        rec.geo3d = normalize_landmarks_labels(rec.geo3d)

    snr_thresh = _snr_threshold_from_preprocess_config(preprocess_config)

    gvtd_before_threshold = gvtd_before + "_threshold"
    gvtd_after_threshold = gvtd_after + "_threshold"

    required_variables = [
        gvtd_before,
        gvtd_before_threshold,
        gvtd_after,
        gvtd_after_threshold,
        prune_reason,
        snr,
        snr_mask,
        od_variance,
    ]

    if output_channel_quality is not None:
        required_variables.extend(
            [
                sd_dist,
                time_clean_fraction,
                mean_amp,
            ]
        )

    _require_variables(
        sidecar,
        required_variables,
    )

    gvtd_before_values = sidecar[gvtd_before]
    gvtd_after_values = sidecar[gvtd_after]

    threshold_before = sidecar[gvtd_before_threshold]
    threshold_after = sidecar[gvtd_after_threshold]

    prune_reason_values = sidecar[prune_reason]
    snr_values = sidecar[snr]
    snr_mask_values = sidecar[snr_mask]
    variance_values = sidecar[od_variance]

    if output_channel_quality is not None:
        sd_dist_values = sidecar[sd_dist]
        time_clean_fraction_values = sidecar[time_clean_fraction]
        mean_amp_values = sidecar[mean_amp]

        if "wavelength" not in mean_amp_values.dims:
            raise ValueError(f"{mean_amp!r} must contain a wavelength dimension.")

    if "wavelength" not in snr_values.dims:
        raise ValueError(f"{snr!r} must contain a wavelength dimension.")

    if "wavelength" not in variance_values.dims:
        raise ValueError(f"{od_variance!r} must contain a wavelength dimension.")

    wavelengths = list(snr_values.wavelength.values)

    if not wavelengths:
        raise ValueError("SNR metric contains no wavelengths.")

    variance_wavelengths = set(variance_values.wavelength.values.tolist())

    missing_wavelengths = [
        wavelength
        for wavelength in wavelengths
        if wavelength not in variance_wavelengths
    ]

    if missing_wavelengths:
        raise ValueError(
            f"OD variance is missing wavelengths present in SNR: {missing_wavelengths}."
        )

    prune_display = _prune_display_metric(prune_reason_values)

    (
        variance_vmin,
        variance_vmax,
    ) = _finite_range(variance_values)

    ncols = max(
        2,
        len(wavelengths),
    )

    fig, axes = plt.subplots(
        3,
        ncols,
        figsize=(
            5.5 * ncols,
            14,
        ),
        squeeze=False,
    )

    #
    # GVTD
    #
    gvtd_ax = axes[
        0,
        0,
    ]

    plot_gvtd(
        gvtd_before_values,
        gvtd_ax,
        threshold=threshold_before,
        label="GVTD",
        ylim_factor=3.0,
        threshold_color="blue",
        color="blue",
    )

    plot_gvtd(
        gvtd_after_values,
        gvtd_ax,
        threshold=threshold_after,
        label="GVTD corrected",
        threshold_color="#ff4500",
        color="#ff4500",
    )

    if rec.stim is not None and not rec.stim.empty:
        plot_stim_markers(
            gvtd_ax,
            rec.stim,
            y=1.0,
        )

    gvtd_ax.legend()
    gvtd_ax.set_title("GVTD")

    #
    # Pruning
    #
    prune_ax = axes[
        0,
        1,
    ]

    prune_cmap = mcolors.ListedColormap(_PRUNE_COLORS)

    reason_array = np.asarray(prune_reason_values.values)

    n_channels = prune_reason_values.sizes["channel"]

    n_pruned = int(np.count_nonzero(np.isfinite(reason_array) & (reason_array != 0)))

    percent_pruned = 100.0 * n_pruned / n_channels

    plot_channel_metric(
        reference_ts,
        rec.geo3d,
        prune_display,
        prune_ax,
        title=(f"Pruned Channels {percent_pruned:.1f}%"),
        min_dist=min_dist,
        cmap=prune_cmap,
        vmin=0,
        vmax=1,
        optode_labels=False,
        optode_size=6,
        cb_ticks_labels=_PRUNE_TICKS,
    )

    #
    # OD variance and SNR
    #
    for (
        column,
        wavelength,
    ) in enumerate(wavelengths):
        plot_channel_metric(
            reference_ts,
            rec.geo3d,
            variance_values,
            axes[
                1,
                column,
            ],
            wavelength=wavelength,
            title=(f"OD Variance - {wavelength:g} nm"),
            min_dist=min_dist,
            cmap="jet",
            vmin=variance_vmin,
            vmax=variance_vmax,
            optode_labels=False,
            optode_size=6,
        )

        wavelength_mask = snr_mask_values.sel(wavelength=wavelength)

        passing = int(
            np.asarray(
                wavelength_mask.values,
                dtype=bool,
            ).sum()
        )

        percent_passing = 100.0 * passing / n_channels

        if snr_thresh is None:
            snr_title = (
                f"SNR - {wavelength:g} nm ({percent_passing:.1f}% pass SNR criterion)"
            )
        else:
            snr_title = (
                f"SNR - {wavelength:g} nm "
                f"({percent_passing:.1f}% "
                f"pass SNR > {snr_thresh:g})"
            )

        plot_channel_metric(
            reference_ts,
            rec.geo3d,
            snr_values,
            axes[
                2,
                column,
            ],
            wavelength=wavelength,
            title=snr_title,
            min_dist=min_dist,
            cmap="jet",
            vmin=0,
            vmax=25,
            optode_labels=False,
            optode_size=6,
        )

    #
    # Hide unused axes for datasets with an unusual
    # number of wavelengths.
    #
    for column in range(
        len(wavelengths),
        ncols,
    ):
        axes[
            1,
            column,
        ].set_axis_off()

        axes[
            2,
            column,
        ].set_axis_off()

    for column in range(
        2,
        ncols,
    ):
        axes[
            0,
            column,
        ].set_axis_off()

    if title is None:
        title = input_snirf.stem

    fig.suptitle(title + "_pruned")

    fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.98,
        )
    )

    output_dqr.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_dqr,
        bbox_inches="tight",
    )

    plt.close(fig)

    #
    # Channel-quality diagnostics
    #
    if output_channel_quality is not None:
        (
            sd_vmin,
            sd_vmax,
        ) = _finite_range(sd_dist_values)

        (
            mean_amp_vmin,
            mean_amp_vmax,
        ) = _finite_range(mean_amp_values)

        quality_fig, quality_axes = plt.subplots(
            2,
            ncols,
            figsize=(
                5.5 * ncols,
                9.5,
            ),
            squeeze=False,
        )

        plot_channel_metric(
            reference_ts,
            rec.geo3d,
            sd_dist_values,
            quality_axes[
                0,
                0,
            ],
            title="Source-Detector Distance",
            min_dist=min_dist,
            cmap="jet",
            vmin=sd_vmin,
            vmax=sd_vmax,
            optode_labels=False,
            optode_size=6,
        )

        plot_channel_metric(
            reference_ts,
            rec.geo3d,
            time_clean_fraction_values,
            quality_axes[
                0,
                1,
            ],
            title="Clean Time Fraction",
            min_dist=min_dist,
            cmap="jet",
            vmin=0,
            vmax=1,
            optode_labels=False,
            optode_size=6,
        )

        for column in range(
            2,
            ncols,
        ):
            quality_axes[
                0,
                column,
            ].set_axis_off()

        for (
            column,
            wavelength,
        ) in enumerate(wavelengths):
            plot_channel_metric(
                reference_ts,
                rec.geo3d,
                mean_amp_values,
                quality_axes[
                    1,
                    column,
                ],
                wavelength=wavelength,
                title=f"Mean Amplitude - {wavelength:g} nm",
                min_dist=min_dist,
                cmap="jet",
                vmin=mean_amp_vmin,
                vmax=mean_amp_vmax,
                optode_labels=False,
                optode_size=6,
            )

        for column in range(
            len(wavelengths),
            ncols,
        ):
            quality_axes[
                1,
                column,
            ].set_axis_off()

        quality_fig.suptitle(title)

        quality_fig.tight_layout(
            rect=(
                0,
                0,
                1,
                0.95,
            )
        )

        output_channel_quality.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        quality_fig.savefig(
            output_channel_quality,
            bbox_inches="tight",
        )

        plt.close(quality_fig)

    #
    # GVTD histograms
    #
    hist_fig, hist_axes = plt.subplots(
        1,
        2,
        figsize=(
            11,
            5,
        ),
        squeeze=False,
    )

    hist_axes = hist_axes[0]

    plot_gvtd_histogram(
        gvtd_before_values,
        hist_axes[0],
        threshold=threshold_before,
        label="GVTD",
        threshold_color="red",
    )

    hist_axes[0].set_title("GVTD Histogram")

    hist_axes[0].legend()

    plot_gvtd_histogram(
        gvtd_after_values,
        hist_axes[1],
        threshold=threshold_after,
        label="GVTD corrected",
        threshold_color="red",
    )

    hist_axes[1].set_title("GVTD Histogram - corrected")

    hist_axes[1].legend()

    hist_fig.suptitle(title)

    hist_fig.tight_layout(
        rect=(
            0,
            0,
            1,
            0.95,
        )
    )

    output_gvtd_histogram.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    hist_fig.savefig(
        output_gvtd_histogram,
        bbox_inches="tight",
    )

    plt.close(hist_fig)
