import pytest

import cedalion.datasets
import cedalion.nirs
import cedalion.sigproc.motion_correct as mc
from cedalion import units


@pytest.fixture
def rec():
    rec = cedalion.datasets.get_snirf_test_data()[0]
    rec["amp"] = rec["amp"].pint.dequantify().pint.quantify(units.V)
    rec["od"] = cedalion.nirs.int2od(rec["amp"])
    return rec


def test_motion_correct_splineSG_default_param(rec):
    mc.motion_correct_splineSG(rec["od"], p=1.0)

def test_motion_correct_splineSG_custom_param(rec):
    mc.motion_correct_splineSG(rec["od"], p=1.0, frame_size=3 * units.s)

def test_motion_correct_tddr(rec):
    mc.tddr(rec["od"])

def test_motion_correct_wavelets(rec):
    mc.motion_correct_wavelet(rec["od"], iqr=1.5, wavelet='db2', level=4)