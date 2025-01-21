import os
import tempfile
import numpy as np
import xarray as xr
from numpy.testing import assert_array_almost_equal

import cedalion.dataclasses as cdc
from cedalion.io import load_tsv


def test_load_tsv():
    # prepare test data
    num = 10
    pos = np.random.rand(num, 3)
    labels = [(np.random.choice(["S%d", "D%d"])) % (i+1) for i in range(num)]

    # write test data to a file
    dirpath = tempfile.mkdtemp()
    with open(os.path.join(dirpath, "optodes.tsv"), "w") as f:
        for l, p in zip(labels, pos):
            f.write("%s\t%f\t%f\t%f\n" % (l, p[0], p[1], p[2]))

    # call load_tsv to read test data
    optodes = load_tsv(os.path.join(dirpath, "optodes.tsv"))
    assert isinstance(optodes, xr.DataArray)
    assert_array_almost_equal(optodes.pint.dequantify().values, pos)
    assert sum(optodes.type == cdc.PointType.SOURCE) + \
           sum(optodes.type == cdc.PointType.DETECTOR) == num



