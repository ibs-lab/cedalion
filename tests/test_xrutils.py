import numpy as np
import pint
import pytest
import xarray as xr
from scipy.sparse import csr_matrix

import cedalion
import cedalion.xrutils as xrutils


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


def make_random_da(shape: tuple, dims: list[str], seed=0):
    assert len(shape) == len(dims)

    rng = np.random.default_rng(seed)
    data = rng.normal(size=shape)
    coords = {d: [f"{d}{i}" for i in range(s)] for d, s in zip(dims, shape)}

    da = xr.DataArray(data, dims=dims, coords=coords)
    da.attrs["source"] = "unit-test"
    return da


@pytest.mark.parametrize("common_axis", [0, 1])
def test_dot_dataarray_csr(common_axis):
    array = make_random_da((2, 3, 4), ("time", "vertex", "wavelength"), seed=1)

    # build a sparse matrix
    nvertex = array.sizes["vertex"]
    nkernel = 5
    if common_axis == 0:
        bdims = ("vertex", "kernel")
        B_dense = np.arange(nvertex * nkernel, dtype=float).reshape(nvertex, nkernel)
    else:
        bdims = ("kernel", "vertex")
        B_dense = np.arange(nvertex * nkernel, dtype=float).reshape(nkernel, nvertex)

    B = csr_matrix(B_dense)

    result = xrutils.dot_dataarray_csr(array, B, bdims)

    # check dimension names
    expected_dims = ("time", "kernel", "wavelength")
    assert tuple(result.dims) == expected_dims

    # attrs preserved
    assert result.attrs.get("source") == array.attrs.get("source")

    # coords preserved
    for d in ("time", "wavelength"):
        np.testing.assert_array_equal(result.coords[d].values, array.coords[d].values)

    # test matrix multiplication result:

    for i_t in range(2):
        for i_wl in range(4):
            if common_axis == 0:
                assert np.allclose(
                    array[i_t, :, i_wl].values @ B, result[i_t, :, i_wl].values
                )
            else:
                assert np.allclose(
                    array[i_t, :, i_wl].values @ B.T, result[i_t, :, i_wl].values
                )


def test_transpose_like():
    seed = 42
    rng = np.random.default_rng(seed)

    for n_dims in range(1, 5):
        # create a random array
        orig = make_random_da(
            shape=(2,) * n_dims, dims=[f"d{i}" for i in range(n_dims)]
        )

        # shuffle dimensions
        transp = orig.transpose(*rng.permuted(orig.dims))

        # reorder to match orig and test equality
        transp2 = xrutils.transpose_like(transp, orig)
        assert transp2.dims == orig.dims

        # select one dimension to be renamed
        old_dim = rng.choice(orig.dims, 1)[0]
        new_dim = "new"

        # shuffle again and rename
        transp = orig.transpose(*rng.permuted(orig.dims)).rename({old_dim: new_dim})

        transp2 = xrutils.transpose_like(transp, orig, dim_map={new_dim : old_dim})

        for d2, d1 in zip(transp2.dims, orig.dims):
            if d2 == new_dim:
                assert d1 == old_dim
            else:
                assert d2 == d1

def test_check_units():
    x_quant_time = xr.DataArray([1,2,3], dims="x").pint.quantify("ps")
    x_dequant_time = x_quant_time.pint.dequantify()
    x_quant_length = xr.DataArray([1,2,3], dims="x").pint.quantify("ly")
    x_dequant_length = x_quant_length.pint.dequantify()
    x_none = xr.DataArray([1,2,3], dims="x")

    assert xrutils.check_units(x_quant_time, "[time]")
    assert xrutils.check_units(x_dequant_time, "[time]")

    assert not xrutils.check_units(x_dequant_time, "[volume]")
    assert not xrutils.check_units(x_quant_time, "[volume]")

    assert xrutils.check_units(x_quant_length, "[length]")
    assert xrutils.check_units(x_dequant_length, "[length]")

    assert not xrutils.check_units(x_dequant_length, "[volume]")
    assert not xrutils.check_units(x_quant_length, "[volume]")


    for dim in cedalion.units._dimensions.keys(): # ["[time]", "[length]", ...]
        assert not xrutils.check_units(x_none, dim)


def test_unit_stripping_is_error():

    a = xr.DataArray([1,2,3]).pint.quantify("m")

    with pytest.warns(pint.errors.UnitStrippedWarning):
        a.values

    xrutils.unit_stripping_is_error(True)

    with pytest.raises(pint.errors.UnitStrippedWarning):
        a.values

    xrutils.unit_stripping_is_error(False)

    with pytest.warns(pint.errors.UnitStrippedWarning):
        a.values


def test_unit_stripping_is_quiet(recwarn):

    a = xr.DataArray([1,2,3]).pint.quantify("m")

    with pytest.warns(pint.errors.UnitStrippedWarning):
        a.values

    xrutils.unit_stripping_is_quiet(True)
    a.values

    assert not any(
        isinstance(w.message, pint.errors.UnitStrippedWarning) for w in recwarn
    )

    xrutils.unit_stripping_is_quiet(False)

    with pytest.warns(pint.errors.UnitStrippedWarning):
        a.values
