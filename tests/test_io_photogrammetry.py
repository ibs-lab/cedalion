import os, tempfile
import numpy as np
import xarray as xr
from collections import OrderedDict
import cedalion
import cedalion.dataclasses as cdc
from cedalion.io import read_photogrammetry_einstar, read_einstar, opt_fid_to_xr
from numpy.testing import assert_array_almost_equal


def test_read_einstar():
    # prepare test data
    fiducials = np.random.rand(5, 3)
    optodes = np.random.rand(np.random.randint(100), 3)
    tmp_fn = write_test_photo_fn(fiducials, optodes)
    # call read_einstar to read test data
    fid, opt = cedalion.io.read_einstar(tmp_fn)
    # evaluate
    assert list(fid.keys()) == ["Nz", "Iz", "Rpa", "Lpa", "Cz"]
    assert_array_almost_equal(np.array(list(fid.values())), fiducials)
    assert_array_almost_equal(np.array(list(opt.values())), optodes)


def test_opt_fid_to_xr():
    for n_fidu in range(10):
        for n_opts in range(10):
            fiducials = np.random.rand(n_fidu, 3)
            fiducials = {"F" + str(i): f for i, f in enumerate(fiducials)}
            optodes = np.random.rand(n_opts, 3)

            optodes = {"S" + str(i): o for i, o in enumerate(optodes)}
            fid, opt = opt_fid_to_xr(fiducials, optodes)
            assert isinstance(fid, xr.DataArray)
            assert isinstance(opt, xr.DataArray)

            if n_fidu > 0:
                assert_array_almost_equal(fid.values, list(fiducials.values()))
            else:
                assert fid.shape == (0, 3)

            if n_opts > 0:
                assert_array_almost_equal(opt.values, list(optodes.values()))
            else:
                assert opt.shape == (0, 3)

            assert (opt.type == cdc.PointType.SOURCE).all()
            optodes = {l.replace("S", "D"): o for l, o in optodes.items()}
            fid, opt = opt_fid_to_xr(fiducials, optodes)
            assert (opt.type == cdc.PointType.DETECTOR).all()
            assert list(opt.label.values) == ["D%d" % i for i in range(len(opt))]


def test_read_photogrammetry_einstar():
    # prepare test data
    fiducials = np.random.rand(5, 3)
    optodes = np.random.rand(100, 3)
    tmp_fn = write_test_photo_fn(fiducials, optodes)
    fid, opt = read_photogrammetry_einstar(tmp_fn)
    assert isinstance(fid, xr.DataArray)
    assert isinstance(opt, xr.DataArray)
    assert_array_almost_equal(fid.values, fiducials)
    assert_array_almost_equal(opt.values, optodes)
    assert sum(opt.type == cdc.PointType.SOURCE) == 10
    assert sum(opt.type == cdc.PointType.DETECTOR) == 90


def write_test_photo_fn(fid, opt):
    """Write a test file for photogrammetry data"""
    dirpath = tempfile.mkdtemp()
    tmp_fn = os.path.join(dirpath, "tmp.txt")
    with open(tmp_fn, "w") as f:
        f.write("Nz,%f, %f, %f\n" % (fid[0][0], fid[0][1], fid[0][2]))
        f.write("Iz,%f, %f, %f\n" % (fid[1][0], fid[1][1], fid[1][2]))
        f.write("Rpa,%f, %f, %f\n" % (fid[2][0], fid[2][1], fid[2][2]))
        f.write("Lpa,%f, %f, %f\n" % (fid[3][0], fid[3][1], fid[3][2]))
        f.write("Cz,%f, %f, %f\n" % (fid[4][0], fid[4][1], fid[4][2]))
        for i, v in enumerate(opt):
            if i < 10:
                label = "S%d" % i
            else:
                label = "D%d" % i
            f.write("%s, %f, %f, %f\n" % (label, v[0], v[1], v[2]))
    return tmp_fn
