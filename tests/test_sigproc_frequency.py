import pytest
import numpy as np
from cedalion.dataclasses.xrschemas import build_timeseries
from cedalion.sigproc.frequency import freq_filter, sampling_rate
from cedalion import units


@pytest.fixture
def timeseries():
    sampling_rate = 3.14
    t = np.arange(1000) / sampling_rate

    f1 = 0.1
    f2 = 1.0

    y1 = 10 * np.sin(2 * np.pi * f1 * t)
    y2 = 10 * np.sin(2 * np.pi * f2 * t)

    return build_timeseries(
        np.vstack((y1, y2, y1 + y2)),
        ["channel", "time"],
        t,
        ["y1", "y2", "y1+y2"],
        "V",
        "s",
    )


def test_fixture(timeseries):
    assert all(timeseries.channel.values == ["y1", "y2", "y1+y2"])


def test_sampling_rate(timeseries):
    q = sampling_rate(timeseries)

    assert q.magnitude == pytest.approx(3.14)
    assert q.units == units.Hz


def test_freq_filter(timeseries):
    filtered = freq_filter(timeseries, 0.8 * units.Hz, 1.2 * units.Hz)
    filtered = filtered.pint.dequantify()
    timeseries = timeseries.pint.dequantify()

    def proj(a, b):
        return np.dot(a, b)

    before_y1 = proj(timeseries.loc["y1"], timeseries.loc["y1"])
    before_y2 = proj(timeseries.loc["y2"], timeseries.loc["y2"])

    after_y1 = proj(timeseries.loc["y1"], filtered.loc["y1"])
    after_y2 = proj(timeseries.loc["y2"], filtered.loc["y2"])
    after_y12_1 = proj(timeseries.loc["y1"], filtered.loc["y1+y2"])
    after_y12_2 = proj(timeseries.loc["y2"], filtered.loc["y1+y2"])

    assert after_y1 < (before_y1 / 100)  # f1 got filtered.
    assert after_y2 == pytest.approx(before_y2, rel=0.005)  #  f2 remains intact
    assert after_y12_1 < (before_y1 / 100)
    assert after_y12_2 == pytest.approx(before_y2, rel=0.005)
