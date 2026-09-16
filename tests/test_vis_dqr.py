import matplotlib.pyplot as plt
import xarray as xr
import pytest

from cedalion import units
import cedalion.vis.dqr as dqr
from cedalion.vis.dqr import plot_gvtd, plot_gvtd_histogram


def test_plot_gvtd():
    """Test that GVTD is drawn on the given axes."""

    gvtd = xr.DataArray(
        [0.1, 0.3, 0.2],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    )

    fig, ax = plt.subplots()

    plot_gvtd(gvtd, ax)

    assert len(ax.lines) == 1

    line = ax.lines[0]

    assert list(line.get_xdata()) == [0, 1, 2]
    assert list(line.get_ydata()) == [0.1, 0.3, 0.2]

    assert ax.get_xlabel() == "time / s"
    assert ax.get_ylabel() == "GVTD / (1/s)"

    plt.close(fig)


def test_plot_gvtd_with_threshold():
    gvtd = xr.DataArray(
        [0.1, 0.3, 0.2],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    )

    fig, ax = plt.subplots()

    plot_gvtd(gvtd, ax, threshold=0.25)

    assert len(ax.lines) == 2
    threshold_line = ax.lines[1]

    assert list(threshold_line.get_ydata()) == [0.25, 0.25]
    assert threshold_line.get_linestyle() == "--"
    assert ax.lines[0].get_label() == "GVTD"
    assert ax.lines[1].get_label() == "GVTD threshold"

    plt.close(fig)


def test_plot_gvtd_requires_time():
    gvtd = xr.DataArray(
        [0.1, 0.3, 0.2],
        dims=["sample"],
    )

    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="time"):
        plot_gvtd(gvtd, ax)

    plt.close(fig)


def test_plot_gvtd_requires_one_dimension():
    gvtd = xr.DataArray(
        [[0.1, 0.3, 0.2], [0.2, 0.4, 0.3]],
        dims=["wavelength", "time"],
        coords={"time": [0, 1, 2]},
    )

    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="one-dimensional"):
        plot_gvtd(gvtd, ax)

    plt.close(fig)


def test_plot_gvtd_allows_some_nan():
    gvtd = xr.DataArray(
        [0.1, float("nan"), 0.2],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    )

    fig, ax = plt.subplots()

    plot_gvtd(gvtd, ax)

    assert len(ax.lines) == 1

    plt.close(fig)


def test_plot_gvtd_with_units():
    gvtd = xr.DataArray(
        [0.1, 0.3, 0.2],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    )

    gvtd = gvtd.pint.quantify("1/s")

    fig, ax = plt.subplots()

    plot_gvtd(gvtd, ax)

    assert len(ax.lines) == 1

    plt.close(fig)


def test_plot_gvtd_does_not_modify_input():
    gvtd = xr.DataArray(
        [0.1, 0.3, 0.2],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    ).pint.quantify("1/s")

    original = gvtd.copy(deep=True)

    fig, ax = plt.subplots()

    plot_gvtd(gvtd, ax)

    xr.testing.assert_identical(gvtd, original)

    plt.close(fig)


def test_plot_gvtd_with_ylim_factor():
    gvtd = xr.DataArray(
        [0.1, 0.3, 0.2],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    )

    fig, ax = plt.subplots()

    plot_gvtd(
        gvtd,
        ax,
        threshold=0.25,
        ylim_factor=3.0,
    )

    assert ax.get_ylim() == (0.0, 0.75)

    plt.close(fig)


def test_plot_gvtd_with_unit_aware_threshold():
    gvtd = xr.DataArray(
        [0.1, 0.3, 0.2],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    ).pint.quantify("1/s")

    threshold = 0.25 / units.s

    fig, ax = plt.subplots()

    plot_gvtd(
        gvtd,
        ax,
        threshold=threshold,
        ylim_factor=3.0,
    )

    assert len(ax.lines) == 2
    assert list(ax.lines[1].get_ydata()) == [0.25, 0.25]
    assert ax.get_ylim() == (0.0, 0.75)

    plt.close(fig)


def test_plot_gvtd_histogram():
    gvtd = xr.DataArray(
        [0.1, 0.1, 0.2, 0.3, 0.3, 0.3],
        dims=["time"],
        coords={"time": [0, 1, 2, 3, 4, 5]},
    )

    fig, ax = plt.subplots()

    plot_gvtd_histogram(gvtd, ax, bins=3)

    assert len(ax.patches) == 3
    assert ax.get_xlabel() == "GVTD / (1/s)"
    assert ax.get_ylabel() == "count"

    plt.close(fig)


def test_plot_gvtd_histogram_with_units():
    gvtd = xr.DataArray(
        [0.1, 0.1, 0.2, 0.3, 0.3, 0.3],
        dims=["time"],
        coords={"time": [0, 1, 2, 3, 4, 5]},
    ).pint.quantify("1/s")

    fig, ax = plt.subplots()

    plot_gvtd_histogram(gvtd, ax, bins=3)

    assert len(ax.patches) == 3

    plt.close(fig)


def test_plot_gvtd_histogram_with_threshold():
    gvtd = xr.DataArray(
        [0.1, 0.1, 0.2, 0.3, 0.3, 0.3],
        dims=["time"],
        coords={"time": [0, 1, 2, 3, 4, 5]},
    ).pint.quantify("1/s")

    threshold = 0.25 / units.s

    fig, ax = plt.subplots()

    plot_gvtd_histogram(
        gvtd,
        ax,
        bins=3,
        threshold=threshold,
    )

    assert len(ax.lines) == 1
    assert list(ax.lines[0].get_xdata()) == [0.25, 0.25]
    assert ax.lines[0].get_linestyle() == "--"
    assert ax.lines[0].get_label() == "GVTD threshold"

    plt.close(fig)


def test_plot_gvtd_histogram_allows_some_nan():
    gvtd = xr.DataArray(
        [0.1, float("nan"), 0.2, 0.3],
        dims=["time"],
        coords={"time": [0, 1, 2, 3]},
    )

    fig, ax = plt.subplots()

    plot_gvtd_histogram(gvtd, ax, bins=3)

    assert len(ax.patches) == 3

    plt.close(fig)


def test_plot_gvtd_histogram_requires_time():
    gvtd = xr.DataArray(
        [0.1, 0.2, 0.3],
        dims=["sample"],
    )

    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="time"):
        plot_gvtd_histogram(gvtd, ax)

    plt.close(fig)


def test_plot_gvtd_histogram_requires_one_dimension():
    gvtd = xr.DataArray(
        [[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]],
        dims=["wavelength", "time"],
        coords={"time": [0, 1, 2]},
    )

    fig, ax = plt.subplots()

    with pytest.raises(ValueError, match="one-dimensional"):
        plot_gvtd_histogram(gvtd, ax)

    plt.close(fig)


def test_plot_gvtd_histogram_does_not_modify_input():
    gvtd = xr.DataArray(
        [0.1, 0.2, 0.3],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    ).pint.quantify("1/s")

    original = gvtd.copy(deep=True)

    fig, ax = plt.subplots()

    plot_gvtd_histogram(gvtd, ax)

    xr.testing.assert_identical(gvtd, original)

    plt.close(fig)


def test_plot_gvtd_histogram_uses_adaptive_bins_by_default():
    gvtd = xr.DataArray(
        [0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
        dims=["time"],
        coords={"time": range(10)},
    )

    fig, ax = plt.subplots()

    plot_gvtd_histogram(gvtd, ax)

    # Legacy rule: round(10 / 5) = 2 bins.
    assert len(ax.patches) == 2

    # GVTD histogram is anchored at zero.
    assert ax.patches[0].get_x() == pytest.approx(0.0)

    plt.close(fig)


def test_plot_gvtd_histogram_allows_all_nan():
    gvtd = xr.DataArray(
        [float("nan"), float("nan"), float("nan")],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    )

    fig, ax = plt.subplots()

    plot_gvtd_histogram(gvtd, ax)

    plt.close(fig)


def test_plot_channel_metric_selects_wavelength(monkeypatch):
    metric = xr.DataArray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dims=["channel", "wavelength"],
        coords={
            "channel": ["ch1", "ch2"],
            "wavelength": [760, 850],
        },
    )

    captured = {}

    def fake_scalp_plot(ts, geo3d, metric, ax, title=None, **kwargs):
        captured["metric"] = metric
        captured["title"] = title

    monkeypatch.setattr(dqr, "scalp_plot", fake_scalp_plot)

    dqr.plot_channel_metric(
        ts=None,
        geo3d=None,
        metric=metric,
        ax=None,
        wavelength=760,
        title="SNR - 760 nm",
    )

    selected = captured["metric"]

    assert selected.dims == ("channel",)
    assert list(selected.values) == [1.0, 3.0]
    assert captured["title"] == "SNR - 760 nm"


def test_plot_channel_metric_requires_one_dimension():
    metric = xr.DataArray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dims=["channel", "wavelength"],
        coords={
            "channel": ["ch1", "ch2"],
            "wavelength": [760, 850],
        },
    )

    with pytest.raises(ValueError, match="one-dimensional"):
        dqr.plot_channel_metric(
            ts=None,
            geo3d=None,
            metric=metric,
            ax=None,
        )


def test_plot_channel_metric_requires_channel():
    metric = xr.DataArray(
        [1.0, 2.0],
        dims=["wavelength"],
        coords={"wavelength": [760, 850]},
    )

    with pytest.raises(ValueError, match="channel"):
        dqr.plot_channel_metric(
            ts=None,
            geo3d=None,
            metric=metric,
            ax=None,
        )


def test_plot_channel_metric_rejects_wavelength_without_dimension():
    metric = xr.DataArray(
        [1.0, 2.0],
        dims=["channel"],
        coords={"channel": ["ch1", "ch2"]},
    )

    with pytest.raises(ValueError, match="no 'wavelength' dimension"):
        dqr.plot_channel_metric(
            ts=None,
            geo3d=None,
            metric=metric,
            ax=None,
            wavelength=760,
        )


def test_plot_channel_metric_does_not_modify_input(monkeypatch):
    metric = xr.DataArray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dims=["channel", "wavelength"],
        coords={
            "channel": ["ch1", "ch2"],
            "wavelength": [760, 850],
        },
    )

    original = metric.copy(deep=True)

    def fake_scalp_plot(ts, geo3d, metric, ax, title=None, **kwargs):
        pass

    monkeypatch.setattr(dqr, "scalp_plot", fake_scalp_plot)

    dqr.plot_channel_metric(
        ts=None,
        geo3d=None,
        metric=metric,
        ax=None,
        wavelength=760,
    )

    xr.testing.assert_identical(metric, original)


def test_plot_channel_metric_rejects_unknown_wavelength():
    metric = xr.DataArray(
        [
            [1.0, 2.0],
            [3.0, 4.0],
        ],
        dims=["channel", "wavelength"],
        coords={
            "channel": ["ch1", "ch2"],
            "wavelength": [760, 850],
        },
    )

    with pytest.raises(ValueError, match="wavelength 999 not in metric"):
        dqr.plot_channel_metric(
            ts=None,
            geo3d=None,
            metric=metric,
            ax=None,
            wavelength=999,
        )


def test_plot_channel_metric_selects_middle_of_three_wavelengths(monkeypatch):
    metric = xr.DataArray(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dims=["channel", "wavelength"],
        coords={
            "channel": ["ch1", "ch2"],
            "wavelength": [760, 800, 850],
        },
    )

    captured = {}

    def fake_scalp_plot(ts, geo3d, metric, ax, title=None, **kwargs):
        captured["metric"] = metric

    monkeypatch.setattr(dqr, "scalp_plot", fake_scalp_plot)

    dqr.plot_channel_metric(
        ts=None,
        geo3d=None,
        metric=metric,
        ax=None,
        wavelength=800,
    )

    selected = captured["metric"]

    assert selected.dims == ("channel",)
    assert list(selected.values) == [2.0, 5.0]


def test_plot_gvtd_with_dataarray_threshold():
    gvtd = xr.DataArray(
        [0.1, 0.3, 0.2],
        dims=["time"],
        coords={"time": [0, 1, 2]},
    ).pint.quantify("1/s")

    threshold = xr.DataArray(0.25).pint.quantify("1/s")

    fig, ax = plt.subplots()

    plot_gvtd(
        gvtd,
        ax,
        threshold=threshold,
    )

    assert len(ax.lines) == 2
    assert list(ax.lines[1].get_ydata()) == [0.25, 0.25]

    plt.close(fig)
