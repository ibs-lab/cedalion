import os
from pathlib import Path
import pytest

import cedalion.io.nirs

skip_if_snirf_zoo_unavailable = pytest.mark.skipif(
    "SNIRF_ZOO" not in os.environ, reason="snirf zoo not available"
)

testfiles = []

if "SNIRF_ZOO" in os.environ:
    snirf_zoo_dir = Path(os.environ["SNIRF_ZOO"])
    testfiles.extend(sorted(map(str, snirf_zoo_dir.glob("**/*.nirs"))))


@skip_if_snirf_zoo_unavailable
@pytest.mark.parametrize("fname", testfiles)
def test_read_nirs(fname: str):
    cedalion.io.nirs.read_nirs(fname)
