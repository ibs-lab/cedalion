import pytest

import cedalion.datasets
import cedalion.nirs
import cedalion.sigproc.motion_correct as mc
import cedalion.sim.synthetic_artifact as synthetic_artifact
from cedalion import units


@pytest.fixture
def rec():
    rec = cedalion.datasets.get_snirf_test_data()[0]
    rec["amp"] = rec["amp"].pint.dequantify().pint.quantify(units.V)
    rec["od"] = cedalion.nirs.int2od(rec["amp"])

    # Add some synthetic spikes and baseline shifts
    artifacts = {"spike": synthetic_artifact.gen_spike}
    timing = synthetic_artifact.random_events_perc(rec["od"].time, 0.05, ["spike"])
    rec["od"] = synthetic_artifact.add_artifacts(rec["od"], timing, artifacts)

    return rec


def test_motion_correct_splineSG_default_param(rec):
    mc.spline_sg(rec["od"], p=1.0)


def test_motion_correct_splineSG_custom_param(rec):
    mc.spline_sg(rec["od"], p=1.0, frame_size=1 * units.s)


def test_motion_correct_tddr(rec):
    mc.tddr(rec["od"])


def test_motion_correct_wavelets(rec):
    mc.wavelet(rec["od"], iqr=1.5, wavelet='db2', level=4)


def test_motion_correct_pca_recurse(rec):
    mc.pca_recurse(
        rec["od"],
        t_motion=0.5,
        t_mask=1,
        stdev_thresh=20,
        amp_thresh=5,
        n_sv=0.97,
        max_iter=5,
    )
