import numpy as np
import pytest

import cedalion.dataclasses as cdc
import cedalion.sim.synthetic_artifact as sa

CHANNELS = ["S1D1", "S1D2", "S2D1"]
TIME = np.arange(0, 100, 0.1)

OTHER_DIMS = pytest.mark.parametrize(
    "other_dim,other_values,units",
    [
        ("wavelength", [760.0, 850.0], "1"),
        ("chromo", ["HbO", "HbR"], "micromolar"),
    ],
)


def _build_ts(other_dim: str, other_values: list, units: str, data=None):
    if data is None:
        rng = np.random.default_rng(42)
        data = rng.normal(size=(len(CHANNELS), len(other_values), len(TIME)))
    return cdc.build_timeseries(
        data,
        ["channel", other_dim, "time"],
        TIME,
        CHANNELS,
        units,
        "s",
        {other_dim: other_values},
    )


def _added(result, ts):
    """Return what add_artifacts added to ts, without units."""
    assert result.dims == ts.dims
    assert result.pint.units == ts.pint.units
    return (result - ts).pint.dequantify()


@OTHER_DIMS
def test_add_artifacts_manual_mode(other_dim, other_values, units):
    ts = _build_ts(other_dim, other_values, units)
    ts_orig_values = ts.pint.dequantify().values.copy()
    onset, duration, scale = 50.0, 0.3, 0.4
    timing = sa.add_event_timing([(onset, duration)], "spike", ["S1D1"])

    result = sa.add_artifacts(
        ts, timing, {"spike": sa.gen_spike}, mode="manual", scale=scale
    )

    diff = _added(result, ts)
    expected = scale * sa.gen_spike(ts.time, onset, duration)

    # selected channel: exactly scale * artifact in every wavelength/chromophore
    for value in other_values:
        np.testing.assert_allclose(
            diff.sel(channel="S1D1", **{other_dim: value}).values,
            expected.values,
            atol=1e-12,
        )

    # other channels are left unchanged
    np.testing.assert_array_equal(diff.sel(channel=["S1D2", "S2D1"]).values, 0.0)

    # input is not modified in place
    np.testing.assert_array_equal(ts.pint.dequantify().values, ts_orig_values)


@OTHER_DIMS
def test_add_artifacts_manual_mode_all_channels(other_dim, other_values, units):
    ts = _build_ts(other_dim, other_values, units)
    onset, duration, scale = 50.0, 0.3, 0.4
    # no channel list: the event applies to all channels
    timing = sa.add_event_timing([(onset, duration)], "spike")

    result = sa.add_artifacts(
        ts, timing, {"spike": sa.gen_spike}, mode="manual", scale=scale
    )

    diff = _added(result, ts)
    expected = scale * sa.gen_spike(ts.time, onset, duration)

    for channel in CHANNELS:
        for value in other_values:
            np.testing.assert_allclose(
                diff.sel(channel=channel, **{other_dim: value}).values,
                expected.values,
                atol=1e-12,
            )


@OTHER_DIMS
def test_add_artifacts_auto_mode(other_dim, other_values, units):
    # Square wave alternating between +a and -a on every sample, with a different a
    # for each channel and wavelength/chromophore. Every full sliding window then
    # spans max - min = 2a, so the auto-mode scaling factor is exactly 2a.
    amplitudes = np.arange(1.0, len(CHANNELS) * len(other_values) + 1).reshape(
        len(CHANNELS), len(other_values)
    )
    square_wave = (-1.0) ** np.arange(len(TIME))
    data = amplitudes[:, :, None] * square_wave[None, None, :]
    ts = _build_ts(other_dim, other_values, units, data=data)

    onset, duration, scale = 50.0, 0.3, 0.4
    timing = sa.add_event_timing([(onset, duration)], "spike", ["S1D2"])

    result = sa.add_artifacts(
        ts, timing, {"spike": sa.gen_spike}, mode="auto", scale=scale
    )

    diff = _added(result, ts)
    spike = sa.gen_spike(ts.time, onset, duration)

    i_ch = CHANNELS.index("S1D2")
    for i_val, value in enumerate(other_values):
        alpha = 2 * amplitudes[i_ch, i_val]
        np.testing.assert_allclose(
            diff.sel(channel="S1D2", **{other_dim: value}).values,
            (alpha * scale * spike).values,
            atol=1e-12,
        )

    np.testing.assert_array_equal(diff.sel(channel=["S1D1", "S2D1"]).values, 0.0)


def test_add_artifacts_invalid_mode():
    ts = _build_ts("wavelength", [760.0, 850.0], "1")
    timing = sa.add_event_timing([(50.0, 0.3)], "spike", ["S1D1"])

    with pytest.raises(ValueError, match="Invalid mode"):
        sa.add_artifacts(ts, timing, {"spike": sa.gen_spike}, mode="foo")
