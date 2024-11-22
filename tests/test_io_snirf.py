"""Tests for cedalion.io.read_snirf."""

import pytest
import os
from pathlib import Path
import cedalion.io
import cedalion.io.snirf
import cedalion.datasets
from tempfile import TemporaryDirectory

# Edge cases in the handling of snirf files are often discovered in files provided
# by users. Ideally, we assemble a zoo of such edge case files and regularly test
# against them. But we won't get permission to share all of these files. Hence, this
# test looks for an environment variable "SNIRF_ZOO" that points to a local directory.
# The test tries to read all snirf files in it. The test is skipped if the directory is
# not available.

skip_if_snirf_zoo_unavailable = pytest.mark.skipif(
    "SNIRF_ZOO" not in os.environ, reason="snirf zoo not available"
)

testfiles = []

if "SNIRF_ZOO" in os.environ:
    snirf_zoo_dir = Path(os.environ["SNIRF_ZOO"])
    testfiles.extend(sorted(map(str, snirf_zoo_dir.glob("**/*.snirf"))))


@skip_if_snirf_zoo_unavailable
@pytest.mark.parametrize("fname", testfiles)
def test_read_snirf(fname):
    cedalion.io.read_snirf(fname)


@skip_if_snirf_zoo_unavailable
@pytest.mark.parametrize("fname", testfiles)
def test_write_snirf(fname):
    recs = cedalion.io.read_snirf(fname)

    with TemporaryDirectory() as tmpdirname:
        fname = Path(tmpdirname) / "test.snirf"
        cedalion.io.snirf.write_snirf(fname, recs)


def test_add_number_to_name():
    keys = ["amp"]
    assert cedalion.io.snirf.add_number_to_name("amp", keys) == "amp_02"

    keys = ["amp", "amp_02"]
    assert cedalion.io.snirf.add_number_to_name("amp", keys) == "amp_03"

    keys = ["amp", "od", "od_02", "od_03", "amp_02"]
    assert cedalion.io.snirf.add_number_to_name("amp", keys) == "amp_03"
    assert cedalion.io.snirf.add_number_to_name("od", keys) == "od_04"


def test_read_snirf_crs():
    path = cedalion.datasets.get_fingertapping_snirf_path()

    rec = cedalion.io.read_snirf(path)[0]
    assert rec.geo3d.points.crs == "pos"

    rec = cedalion.io.read_snirf(path, crs="another_crs")[0]
    assert rec.geo3d.points.crs == "another_crs"
