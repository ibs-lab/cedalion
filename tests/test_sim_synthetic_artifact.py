import numpy as np
import pytest

import cedalion.dataclasses as cdc
import cedalion.sim.synthetic_artifact as sa

CHANNELS = ["S1D1", "S1D2", "S2D1"]


def _build_ts(other_dim: str, other_values: list, units: str):
    rng = np.random.default_rng(42)
    time = np.arange(0, 100, 0.1)
    data = rng.normal(size=(len(CHANNELS), len(other_values), len(time)))
    return cdc.build_timeseries(
        data,
        ["channel", other_dim, "time"],
        time,
        CHANNELS,
        units,
        "s",
        {other_dim: other_values},
    )


@pytest.mark.parametrize(
    "other_dim,other_values,units",
    [
        ("wavelength", [760.0, 850.0], "1"),
        ("chromo", ["HbO", "HbR"], "micromolar"),
    ],
)
def test_add_artifacts_manual_mode(other_dim, other_values, units):
    ts = _build_ts(other_dim, other_values, units)
    ts_orig_values = ts.pint.dequantify().values.copy()
    onset, duration, scale = 50.0, 0.3, 0.4
    timing = sa.add_event_timing([(onset, duration)], "spike", ["S1D1"])

    result = sa.add_artifacts(
        ts, timing, {"spike": sa.gen_spike}, mode="manual", scale=scale
    )

    assert result.dims == ts.dims
    assert result.pint.units == ts.pint.units

    diff = (result - ts).pint.dequantify()
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


def test_add_artifacts_invalid_mode():
    ts = _build_ts("wavelength", [760.0, 850.0], "1")
    timing = sa.add_event_timing([(50.0, 0.3)], "spike", ["S1D1"])

    with pytest.raises(ValueError, match="Invalid mode"):
        sa.add_artifacts(ts, timing, {"spike": sa.gen_spike}, mode="foo")
