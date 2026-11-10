import numpy as np
import pytest
import xarray as xr

import cedalion.data
import cedalion.dataclasses as cdc
import cedalion.nirs
import cedalion.sigproc.motion as mc
import cedalion.sim.synthetic_artifact as synthetic_artifact
from cedalion import units


@pytest.fixture
def rec():
    rec = cedalion.data.get_snirf_test_data()[0]
    rec["amp"] = rec["amp"].pint.dequantify().pint.quantify(units.V)
    rec["od"] = cedalion.nirs.cw.int2od(rec["amp"])

    # Add some synthetic spikes and baseline shifts
    artifacts = {"spike": synthetic_artifact.gen_spike}
    timing = synthetic_artifact.random_events_perc(rec["od"].time, 0.05, ["spike"])
    rec["od"] = synthetic_artifact.add_artifacts(
        rec["od"], timing, artifacts, window_size=10
    )

    return rec


def test_motion_splineSG_default_param(rec):
    mc.spline_sg(rec["od"], p=1.0)


def test_motion_splineSG_custom_param(rec):
    mc.spline_sg(rec["od"], p=1.0, frame_size=1 * units.s)


def test_motion_tddr(rec):
    mc.tddr(rec["od"])


def test_motion_wavelets(rec):
    mc.wavelet(rec["od"], iqr=1.5, wavelet='db2', level=4)


def test_motion_pca_recurse(rec):
    mc.pca_recurse(
        rec["od"],
        t_motion=0.5,
        t_mask=1,
        stdev_thresh=20,
        amp_thresh=5,
        n_sv=0.97,
        max_iter=5,
    )


def _tddr_test_data():
    rng = np.random.default_rng(seed=0)
    fs = 10.0
    t = np.arange(0, 120, 1 / fs)
    nch = 2
    data = 0.01 * np.sin(2 * np.pi * 0.05 * t) + 0.001 * rng.standard_normal(
        (nch, 2, len(t))
    )
    # motion spikes
    data[:, :, 400:405] += 0.05
    data[:, :, 800:803] -= 0.05

    return cdc.build_timeseries(
        data,
        dims=["channel", "wavelength", "time"],
        time=t,
        channel=["S1D1", "S1D2"],
        value_units="1",
        time_units="s",
        other_coords={
            "wavelength": [760.0, 850.0],
            "source": ("channel", ["S1", "S1"]),
            "detector": ("channel", ["D1", "D2"]),
        },
    )


def test_tddr_dimension_order_independence():
    """TDDR must not depend on the dimension order of the input.

    Regression test: with time not being the last dimension, np.diff ran along a
    size-1 axis and the whole low-frequency content was silently removed.
    """
    ts = _tddr_test_data()

    ref = mc.tddr(ts).pint.dequantify()
    assert ref.transpose("channel", "wavelength", "time").dims == (
        "channel",
        "wavelength",
        "time",
    )

    for dims in [("time", "channel", "wavelength"), ("wavelength", "time", "channel")]:
        out = mc.tddr(ts.transpose(*dims)).pint.dequantify()
        xr.testing.assert_allclose(
            out.transpose("channel", "wavelength", "time"),
            ref.transpose("channel", "wavelength", "time"),
        )

    # the slow sinusoid must survive the correction
    slow = 0.01 * np.sin(2 * np.pi * 0.05 * ts.time.values)
    corrected = ref.transpose("channel", "wavelength", "time").values[0, 0]
    assert np.corrcoef(corrected, slow)[0, 1] > 0.7
