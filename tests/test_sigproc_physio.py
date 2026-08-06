import pytest
import cedalion.sigproc.physio as physio
import cedalion.data
from cedalion import units
import numpy as np

np.random.seed(42)


@pytest.fixture
def rec():
    rec = cedalion.data.get_snirf_test_data()[0]
    return rec


def test_global_component_subtract(rec):
    ts = rec["amp"]

    ts_weights = ts.sum("time")
    ts_weights[:,:] = np.random.uniform(
        0.01, 1, size=(ts.sizes["channel"], ts.sizes["wavelength"])
    )

    for k in [0, 1, 2]:
        correct, global_comp = physio.global_component_subtract(
            ts, ts_weights=None, k=k
        )

        correct, global_comp = physio.global_component_subtract(
            ts, ts_weights=ts_weights, k=k
        )
