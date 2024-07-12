"""Tests for cedalion.io.read_snirf."""

import pytest
import os
from pathlib import Path
import cedalion.io


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
