import pytest
import cedalion.xrutils as xrutils
import pint
import numpy as np
import xarray as xr


def test_pinv():
    a = np.asarray([[1, 2], [3, 4]])
    ainv = np.asarray([[-2, 1], [1.5, -0.5]])

    A = xr.DataArray(a, dims=["x", "y"])
    A = A.pint.quantify("kg")

    Ainv = xrutils.pinv(A)

    # units get inverted
    assert Ainv.pint.units == pint.Unit("kg^-1")

    Ainv = Ainv.pint.dequantify()
    A = A.pint.dequantify()

    assert ((Ainv.values - ainv).round(14) == np.zeros((2, 2))).all()

    # matrix product of DataArray.values yields identity
    assert ((Ainv.values @ A.values).round(14) == np.eye(2)).all()

    # matrix product of DataArrays contracts over
    # both dimensions:
    assert Ainv @ A == pytest.approx(2.0)
