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


def _spline_test_data():
    """A slow sinusoid with a 3 s bump artifact and the matching artifact mask."""
    fs = 10.0
    t = np.arange(0, 60, 1 / fs)
    signal = 0.01 * np.sin(2 * np.pi * 0.05 * t)
    artifact = np.zeros_like(t)
    seg = slice(300, 330)
    artifact[seg] = 0.05 * np.hanning(30)
    data = signal + artifact

    ts = cdc.build_timeseries(
        np.stack([data, data])[None, :, :],
        dims=["channel", "wavelength", "time"],
        time=t,
        channel=["S1D1"],
        value_units="1",
        time_units="s",
        other_coords={
            "wavelength": [760.0, 850.0],
            "source": ("channel", ["S1"]),
            "detector": ("channel", ["D1"]),
        },
    )
    mask = xr.ones_like(ts, dtype=bool)
    mask[:, :, seg] = False
    return ts, mask, seg, signal


def test_spline_smoothing_parameter_follows_csaps():
    """The smoothing parameter p behaves like csaps(t, y, p) in Homer3.

    Regression test: p was passed as scipy's UnivariateSpline `s` (an upper bound
    on the residual sum of squares), which degenerated the fit to a single cubic
    polynomial and made the result independent of p.
    """
    ts, mask, seg, _ = _spline_test_data()

    corrected = {p: mc.spline(ts, mask, p) for p in [0.01, 0.99, 1.0]}

    # spline() corrects the samples between the CLEAN->TAINTED and the
    # TAINTED->CLEAN transition, i.e. [seg.start - 1, seg.stop - 1)
    corrected_seg = slice(seg.start, seg.stop - 1)

    def segment_std(p):
        return np.std(corrected[p].pint.dequantify().values[0, 0, corrected_seg])

    # p=1 interpolates the data: the artifact segment is removed entirely and only
    # a constant (the shifted baseline) remains
    assert segment_std(1.0) < 1e-9

    # a smoother spline follows the bump less closely, so more of it remains
    assert segment_std(0.01) > 10 * segment_std(0.99)

    # p must have an effect
    assert not np.allclose(corrected[0.01].values, corrected[0.99].values)


def test_spline_rejects_invalid_smoothing_parameter():
    ts, mask, _, _ = _spline_test_data()
    with pytest.raises(ValueError):
        mc.spline(ts, mask, 0.0)
    with pytest.raises(ValueError):
        mc.spline(ts, mask, 1.5)
