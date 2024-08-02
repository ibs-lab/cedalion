import pytest
import numpy as np
import cedalion.sigproc.quality
import cedalion.datasets
from cedalion import units


def test_sci():
    recordings = cedalion.datasets.get_snirf_test_data()
    amp = recordings[0].timeseries["amp"]

    sci, sci_mask = cedalion.sigproc.quality.sci(amp, 5 * units.s, 0.7)
