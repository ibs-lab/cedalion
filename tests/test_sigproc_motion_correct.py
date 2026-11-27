import numpy as np
import pytest

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


def _wavelet_test_data(n_samples=4096, fs=10.0, seed=0):
    rng = np.random.default_rng(seed=seed)
    t = np.arange(n_samples) / fs
    clean = 0.01 * np.sin(2 * np.pi * 0.05 * t) + 0.001 * rng.standard_normal(
        n_samples
    )
    artifact = np.zeros(n_samples)
    artifact[1500:1560] = 0.05 * np.hanning(60)  # 6 s bump
    artifact[3000:3003] = 0.05  # short spike
    data = clean + artifact

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
    return ts, clean, artifact


def test_wavelet_removes_bump_with_default_level():
    """With Homer3's default decomposition depth a 6 s bump is strongly attenuated.

    Regression test: the default of 4 decomposition levels only covered scales
    below 1.6 s at 10 Hz and the IQR statistics were computed on short time blocks,
    so the correction left the bump almost untouched.
    """
    ts, clean, artifact = _wavelet_test_data()

    corrected = mc.wavelet(ts, iqr=1.5).pint.dequantify().values[0, 0]
    uncorrected = ts.pint.dequantify().values[0, 0]

    bump = slice(1480, 1580)
    spike = slice(2990, 3013)

    def residual(x, window):
        return np.std((x - clean)[window])

    assert residual(corrected, bump) < 0.6 * residual(uncorrected, bump)
    assert residual(corrected, spike) < 0.2 * residual(uncorrected, spike)

    # with only 4 decomposition levels (scales up to 1.6 s at 10 Hz) the bump is
    # not corrected at all
    corrected4 = mc.wavelet(ts, iqr=1.5, level=4).pint.dequantify().values[0, 0]
    assert residual(corrected4, bump) > 0.9 * residual(uncorrected, bump)


def test_wavelet_is_shift_invariant():
    """A circular shift of the input shifts the output (record-wide statistics).

    The stationary wavelet transform is shift invariant and the outlier statistics
    are computed over the whole (unpadded) record for each shifted transform. For a
    signal whose length is a power of two the correction therefore commutes with a
    circular shift. Regression test: statistics on contiguous time blocks broke this.
    """
    ts, _, _ = _wavelet_test_data()
    shift = 256

    ts_shifted = ts.copy(data=np.roll(ts.values, shift, axis=-1))

    corrected = mc.wavelet(ts, iqr=1.5).pint.dequantify().values
    corrected_shifted = mc.wavelet(ts_shifted, iqr=1.5).pint.dequantify().values

    np.testing.assert_allclose(
        np.roll(corrected, shift, axis=-1), corrected_shifted, atol=1e-12
    )
