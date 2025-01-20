import pytest
import cedalion.datasets
import cedalion.geometry.utils as geoutils
from cedalion.dataclasses.geometry import affine_transform_from_numpy
from cedalion.errors import CRSMismatchError

@pytest.fixture
def geo3d():
    recordings = cedalion.datasets.get_snirf_test_data()
    geo3d = recordings[0].geo3d
    geo3d = geo3d.rename({"pos": "digitized"})

    return geo3d


@pytest.fixture
def np_transform():
    return geoutils.m_trans([1, 2, 3])


@pytest.fixture
def xr_transform(np_transform):
    return affine_transform_from_numpy(np_transform, "digitized", "coreg", "m", "m")


def test_crs(geo3d):
    assert geo3d.dims == ("label", "digitized")
    assert geo3d.points.crs == "digitized"


def test_apply_transform(geo3d, xr_transform):
    to_crs = "coreg"
    from_crs = "digitized"

    assert geo3d.points.crs == from_crs
    assert xr_transform.dims == (to_crs, from_crs)

    transformed = geo3d.points.apply_transform(xr_transform)
    assert transformed.points.crs == to_crs


def test_apply_transform_crs_mismatch(geo3d, xr_transform):
    geo3d = geo3d.rename({"digitized" : "some_other_crs"})
    with pytest.raises(CRSMismatchError):
        geo3d.points.apply_transform(xr_transform)


def test_apply_transform_numpy(geo3d, np_transform):
    transformed = geo3d.points.apply_transform(np_transform)

    orig_crs = geo3d.points.crs
    transformed_crs = transformed.points.crs

    assert orig_crs == transformed_crs # numpy transforms don't change the crs
    assert geo3d.pint.units == transformed.pint.units # same for units
