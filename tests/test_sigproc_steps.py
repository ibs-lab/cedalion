"""Tests for preprocessing step adapters."""

import numpy as np
import pytest
import xarray as xr

from cedalion import units
import cedalion.sigproc.steps as steps


class DummyRecording(dict):
    """Minimal recording-like object needed by step adapter tests."""

    def __init__(self):
        super().__init__()
        self.geo3d = object()


def _make_timeseries(
    channels=("ch1", "ch2", "ch3"),
    wavelengths=(760.0, 850.0),
    times=(0.0, 1.0, 2.0, 3.0),
):
    """Create a small deterministic amplitude time series."""
    values = np.ones(
        (len(channels), len(wavelengths), len(times)),
        dtype=float,
    )

    return xr.DataArray(
        values,
        dims=("channel", "wavelength", "time"),
        coords={
            "channel": list(channels),
            "wavelength": list(wavelengths),
            "time": list(times),
        },
    )


def _make_context(ts, step_name):
    """Create a preprocessing context with lightweight test containers."""
    return steps.Context(
        rec=DummyRecording(),
        sidecar={},
        ts=ts,
        step_name=step_name,
    )


def test_collapse_channel_mask_collapses_non_channel_dimensions():
    """Require all values across non-channel dimensions to be clean."""
    mask = xr.DataArray(
        [
            [True, True],
            [True, False],
            [False, False],
        ],
        dims=("channel", "wavelength"),
        coords={
            "channel": ["ch1", "ch2", "ch3"],
            "wavelength": [760.0, 850.0],
        },
    )

    result = steps._collapse_channel_mask(mask)

    expected = xr.DataArray(
        [True, False, False],
        dims=("channel",),
        coords={"channel": ["ch1", "ch2", "ch3"]},
    )

    xr.testing.assert_equal(result, expected)


def test_collapse_channel_mask_requires_channel_dimension():
    """Reject masks that cannot be reduced to channel quality."""
    mask = xr.DataArray(
        [True, False],
        dims=("wavelength",),
        coords={"wavelength": [760.0, 850.0]},
    )

    with pytest.raises(
        ValueError,
        match="must contain a channel dimension",
    ):
        steps._collapse_channel_mask(mask)


def test_snr_adapter_stores_metric_and_mask(monkeypatch):
    """Store SNR diagnostics without modifying the current time series."""
    ts = _make_timeseries()
    ctx = _make_context(ts, "snr")

    snr_values = xr.DataArray(
        [
            [10.0, 11.0],
            [4.0, 3.0],
            [8.0, 9.0],
        ],
        dims=("channel", "wavelength"),
        coords={
            "channel": ts.channel,
            "wavelength": ts.wavelength,
        },
    )

    snr_mask = snr_values > 5.0

    def fake_snr(amplitudes, *, snr_thresh):
        assert amplitudes is ts
        assert snr_thresh == 5.0
        return snr_values, snr_mask

    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "snr",
        fake_snr,
    )

    steps._snr(ctx, snr_thresh=5.0)

    xr.testing.assert_equal(ctx.sidecar["snr"], snr_values)
    xr.testing.assert_equal(ctx.sidecar["snr_mask"], snr_mask)

    assert ctx.rec == {}


def test_gvtd_adapter_stores_trace_mask_and_threshold(monkeypatch):
    """Store all information required to reproduce the GVTD report."""
    ts = _make_timeseries()
    ctx = _make_context(ts, "gvtd_before")

    gvtd_values = xr.DataArray(
        [0.0, 0.1, 0.4, 0.2],
        dims=("time",),
        coords={"time": ts.time},
    )
    gvtd_mask = gvtd_values < 0.3
    gvtd_threshold = xr.DataArray(0.3)

    def fake_gvtd(amplitudes, *, stat_type, n_std):
        assert amplitudes is ts
        assert stat_type == "histogram_mode"
        assert n_std == 10
        return gvtd_values, gvtd_mask

    def fake_threshold(values, *, stat_type, n_std):
        assert values is gvtd_values
        assert stat_type == "histogram_mode"
        assert n_std == 10
        return gvtd_threshold

    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "gvtd",
        fake_gvtd,
    )
    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "_get_gvtd_threshold",
        fake_threshold,
    )

    steps._gvtd(
        ctx,
        stat_type="histogram_mode",
        n_std=10,
    )

    xr.testing.assert_equal(
        ctx.sidecar["gvtd_before"],
        gvtd_values,
    )
    xr.testing.assert_equal(
        ctx.sidecar["gvtd_before_mask"],
        gvtd_mask,
    )
    xr.testing.assert_equal(
        ctx.sidecar["gvtd_before_threshold"],
        gvtd_threshold,
    )


def test_gvtd_from_od_reconstructs_relative_amplitude(monkeypatch):
    """Convert OD back to relative amplitude before calculating GVTD."""
    od_values = np.array(
        [
            [
                [0.0, 0.1, 0.2, 0.3],
                [0.2, 0.3, 0.4, 0.5],
            ],
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.3, 0.4, 0.5, 0.6],
            ],
            [
                [0.2, 0.3, 0.4, 0.5],
                [0.4, 0.5, 0.6, 0.7],
            ],
        ]
    )

    od = xr.DataArray(
        od_values,
        dims=("channel", "wavelength", "time"),
        coords={
            "channel": ["ch1", "ch2", "ch3"],
            "wavelength": [760.0, 850.0],
            "time": [0.0, 1.0, 2.0, 3.0],
        },
    ).pint.quantify("dimensionless")

    ctx = _make_context(od, "gvtd_after")

    gvtd_values = xr.DataArray(
        [0.0, 0.1, 0.2, 0.1],
        dims=("time",),
        coords={"time": od.time},
    )
    gvtd_mask = xr.DataArray(
        [True, True, False, True],
        dims=("time",),
        coords={"time": od.time},
    )
    gvtd_threshold = xr.DataArray(0.15)

    captured = {}

    def fake_gvtd(amplitudes, *, stat_type, n_std):
        captured["amplitudes"] = amplitudes
        assert stat_type == "histogram_mode"
        assert n_std == 10
        return gvtd_values, gvtd_mask

    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "gvtd",
        fake_gvtd,
    )
    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "_get_gvtd_threshold",
        lambda values, *, stat_type, n_std: gvtd_threshold,
    )

    steps._gvtd_from_od(
        ctx,
        stat_type="histogram_mode",
        n_std=10,
    )

    relative_amp = captured["amplitudes"]

    assert relative_amp.pint.units == units.dimensionless

    np.testing.assert_allclose(
        relative_amp.pint.dequantify().values,
        np.exp(-od.pint.dequantify().values),
    )

    xr.testing.assert_equal(
        ctx.sidecar["gvtd_after"],
        gvtd_values,
    )
    xr.testing.assert_equal(
        ctx.sidecar["gvtd_after_mask"],
        gvtd_mask,
    )
    xr.testing.assert_equal(
        ctx.sidecar["gvtd_after_threshold"],
        gvtd_threshold,
    )


def test_log_variance_calculates_log10_temporal_variance():
    """Calculate variance over time and convert nonfinite values to NaN."""
    values = np.array(
        [
            [
                [1.0, 2.0, 3.0, 4.0],
                [5.0, 5.0, 5.0, 5.0],
            ],
            [
                [2.0, 4.0, 6.0, 8.0],
                [1.0, 3.0, 5.0, 7.0],
            ],
        ]
    )

    ts = xr.DataArray(
        values,
        dims=("channel", "wavelength", "time"),
        coords={
            "channel": ["ch1", "ch2"],
            "wavelength": [760.0, 850.0],
            "time": [0.0, 1.0, 2.0, 3.0],
        },
    )

    ctx = _make_context(ts, "od_variance_corrected")

    steps._log_variance(ctx)

    expected_variance = values.var(axis=2)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        expected = np.log10(expected_variance)

    expected[~np.isfinite(expected)] = np.nan

    result = ctx.sidecar["od_variance_corrected"]

    assert result.dims == ("channel", "wavelength")

    np.testing.assert_allclose(
        result.values,
        expected,
        equal_nan=True,
    )


def test_prune_stores_diagnostics_and_drops_bad_channels(monkeypatch):
    """Drop rejected channels while retaining diagnostics for all channels."""
    ts = _make_timeseries()
    ctx = _make_context(ts, "amp_pruned")

    channels = ts.channel.values
    wavelengths = ts.wavelength.values
    times = ts.time.values

    snr_values = xr.DataArray(
        [
            [10.0, 10.0],
            [2.0, 2.0],
            [10.0, 10.0],
        ],
        dims=("channel", "wavelength"),
        coords={
            "channel": channels,
            "wavelength": wavelengths,
        },
    )
    snr_mask = snr_values > 5.0

    sd_dist = xr.DataArray(
        [20.0, 20.0, 20.0],
        dims=("channel",),
        coords={"channel": channels},
    )
    sd_mask = xr.DataArray(
        [True, True, True],
        dims=("channel",),
        coords={"channel": channels},
    )

    mean_amp = xr.DataArray(
        np.full((3, 2), 0.5),
        dims=("channel", "wavelength"),
        coords={
            "channel": channels,
            "wavelength": wavelengths,
        },
    )
    amp_mask = xr.DataArray(
        np.ones((3, 2), dtype=bool),
        dims=("channel", "wavelength"),
        coords={
            "channel": channels,
            "wavelength": wavelengths,
        },
    )

    sci_values = xr.DataArray(
        np.ones((3, 4)),
        dims=("channel", "time"),
        coords={
            "channel": channels,
            "time": times,
        },
    )
    sci_mask = xr.DataArray(
        [
            [True, True, True, True],
            [True, True, True, True],
            [True, False, False, False],
        ],
        dims=("channel", "time"),
        coords={
            "channel": channels,
            "time": times,
        },
    )

    psp_values = xr.DataArray(
        np.ones((3, 4)),
        dims=("channel", "time"),
        coords={
            "channel": channels,
            "time": times,
        },
    )
    psp_mask = xr.DataArray(
        np.ones((3, 4), dtype=bool),
        dims=("channel", "time"),
        coords={
            "channel": channels,
            "time": times,
        },
    )

    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "snr",
        lambda amplitudes, *, snr_thresh: (
            snr_values,
            snr_mask,
        ),
    )
    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "sd_dist",
        lambda amplitudes, geo3d, *, sd_range: (
            sd_dist,
            sd_mask,
        ),
    )
    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "mean_amp",
        lambda amplitudes, *, amp_range: (
            mean_amp,
            amp_mask,
        ),
    )
    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "sci",
        lambda amplitudes, window_length, sci_thresh: (
            sci_values,
            sci_mask,
        ),
    )
    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "psp",
        lambda amplitudes, window_length, psp_thresh: (
            psp_values,
            psp_mask,
        ),
    )

    captured = {}

    def fake_prune_ch(
        amplitudes,
        masks,
        operator,
        flag_drop,
    ):
        assert amplitudes is ts
        assert operator == "all"
        assert flag_drop is True

        captured["mask"] = masks[0]

        return (
            amplitudes.sel(channel=["ch1"]),
            ["ch2", "ch3"],
        )

    monkeypatch.setattr(
        steps.cedalion.sigproc.quality,
        "prune_ch",
        fake_prune_ch,
    )

    steps._prune(
        ctx,
        snr_thresh=5.0,
        sd_thresh_min=1 * units.mm,
        sd_thresh_max=45 * units.mm,
        amp_thresh_min=0.001,
        amp_thresh_max=0.84,
        window_length=5 * units.s,
        sci_thresh=0.6,
        psp_thresh=0.1,
        perc_time_clean_thresh=0.6,
        use_sci=True,
        use_psp=False,
    )

    np.testing.assert_array_equal(
        captured["mask"].values,
        [True, False, False],
    )

    assert ctx.rec["amp_pruned"].channel.values.tolist() == ["ch1"]

    expected_sidecar_entries = {
        "amp_pruned_snr",
        "amp_pruned_snr_mask",
        "amp_pruned_sd_dist",
        "amp_pruned_sd_mask",
        "amp_pruned_mean_amp",
        "amp_pruned_amp_mask",
        "amp_pruned_sci",
        "amp_pruned_sci_mask",
        "amp_pruned_psp",
        "amp_pruned_psp_mask",
        "amp_pruned_time_clean_fraction",
        "amp_pruned_time_clean_mask",
        "amp_pruned_initial_mask",
        "amp_pruned_mask",
        "amp_pruned_reason",
    }

    assert expected_sidecar_entries <= set(ctx.sidecar)

    np.testing.assert_array_equal(
        ctx.sidecar["amp_pruned_mask"].values,
        [True, False, False],
    )

    reason = ctx.sidecar["amp_pruned_reason"]

    assert reason.dims == ("channel",)
    assert reason.sel(channel="ch1").item() == 0

    assert set(np.unique(reason.values)) <= {
        0,
        1,
        2,
        3,
        4,
        5,
    }


def test_prune_rejects_invalid_clean_fraction():
    """Reject a clean-time threshold outside the valid probability range."""
    ts = _make_timeseries()
    ctx = _make_context(ts, "amp_pruned")

    with pytest.raises(
        ValueError,
        match="perc_time_clean_thresh",
    ):
        steps._prune(
            ctx,
            snr_thresh=5.0,
            sd_thresh_min=1 * units.mm,
            sd_thresh_max=45 * units.mm,
            amp_thresh_min=0.001,
            amp_thresh_max=0.84,
            window_length=5 * units.s,
            sci_thresh=0.6,
            psp_thresh=0.1,
            perc_time_clean_thresh=1.1,
            use_sci=True,
            use_psp=False,
        )
