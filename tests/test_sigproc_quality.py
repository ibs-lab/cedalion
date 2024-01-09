import pytest
import numpy as np
import cedalion.sigproc.quality
import cedalion.datasets
from cedalion import units


def test_sci():
    elements = cedalion.datasets.get_snirf_test_data()
    amp = elements[0].data[0]

    sci = cedalion.sigproc.quality.sci(amp, 5 * units.s)
