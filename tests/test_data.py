import pytest

import cedalion.data
import cedalion.dataclasses as cdc

REGISTRATION_LANDMARKS = {"Nz", "Iz", "LPA", "RPA", "Cz"}


# restricted to datasets that scripts/prefetch_docs_datasets.py downloads in CI
@pytest.mark.parametrize(
    "name",
    [
        "fingertapping",
        "fingertappingDOT",
        "nn22_resting",
        "ninja_cap_56x144",
        "ninja_uhd_cap_164x496",
    ],
)
def test_get_example_probe(name):
    geo3d, meas_list, amp = cedalion.data.get_example_probe(name)

    is_optode = (geo3d.type == cdc.PointType.SOURCE) | (
        geo3d.type == cdc.PointType.DETECTOR
    )
    assert is_optode.sum() > 0
    # at least three landmarks are needed for the initial alignment
    assert len(REGISTRATION_LANDMARKS & set(geo3d.label.values)) >= 3

    channels = set(meas_list.source.astype(str) + meas_list.detector.astype(str))
    assert channels == set(amp.channel.values)


def test_get_example_probe_unknown():
    with pytest.raises(ValueError, match="Unknown example probe"):
        cedalion.data.get_example_probe("nonexistent")
