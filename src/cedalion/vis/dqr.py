# src/cedalion/vis/dqr.py

import numpy as np
import xarray as xr
from matplotlib.axes import Axes

import cedalion.typing as cdt
from cedalion import Quantity
from cedalion.vis.anatomy.scalp_plot import scalp_plot


# Keep histogram binning consistent with _get_gvtd_threshold.
_MIN_COUNTS_PER_BIN = 5


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
