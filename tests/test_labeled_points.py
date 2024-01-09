import pytest
import xarray as xr
import numpy as np
from cedalion.dataclasses.xrschemas import LabeledPointCloudSchema
from cedalion.dataclasses import PointType, affine_transform_from_numpy
from cedalion import units


@pytest.fixture
def labeled_points():
    return xr.DataArray(
        np.arange(12).reshape(4, 3),
        dims=["label", "mni"],
        coords={
            "label": ("label", ["S1", "D1", "Nz", "Iz"]),
            "type": (
                "label",
                [
                    PointType.SOURCE,
                    PointType.DETECTOR,
                    PointType.LANDMARK,
                    PointType.LANDMARK,
                ],
            ),
        },
        attrs={"units": "mm"},
    ).pint.quantify()


def test_schema_validate(labeled_points):
    LabeledPointCloudSchema.validate(labeled_points)


def test_points_add_single(labeled_points):
    result = labeled_points.points.add("Cz", [1.0, 2.0, 3.0], PointType.LANDMARK)

    assert result.pint.units == units.Unit("mm")
    assert len(result) == 5
    assert "Cz" in result.label
    assert result.loc["Cz"].type.item() == PointType.LANDMARK

    all(result.loc["Cz"] == units.Quantity([1.0, 2.0, 3.0], "mm"))


def test_points_add_multiple(labeled_points):
    result = labeled_points.points.add(
        ["S5", "D5"], np.arange(6).reshape(2, 3), [PointType.SOURCE, PointType.DETECTOR]
    )

    assert result.pint.units == units.Unit("mm")
    assert len(result) == 6
    assert "S5" in result.label
    assert "D5" in result.label

    assert result.loc["S5"].type.item() == PointType.SOURCE
    assert result.loc["D5"].type.item() == PointType.DETECTOR

    all(result.loc["S5"] == units.Quantity([0.0, 1.0, 2.0], "mm"))
    all(result.loc["D5"] == units.Quantity([3.0, 4.0, 5.0], "mm"))


def test_transform_numpy(labeled_points):
    trafo = np.array(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transformed = labeled_points.points.apply_transform(trafo)

    assert all(labeled_points[0, :] == units.Quantity([0.0, 1.0, 2.0], "mm"))
    assert all(transformed[0, :] == units.Quantity([0.0, 2.0, 4.0], "mm"))
    assert transformed.dims == labeled_points.dims
    assert transformed.pint.units == labeled_points.pint.units


def test_transform_AffineTransform(labeled_points):
    trafo = affine_transform_from_numpy(
        [
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        from_crs="mni",
        to_crs="other_crs",
        from_units="mm",
        to_units="cm",
    )

    transformed = labeled_points.points.apply_transform(trafo)

    assert all(labeled_points[0, :] == units.Quantity([0.0, 1.0, 2.0], "mm"))
    assert all(transformed[0, :] == units.Quantity([0.0, 2.0, 4.0], "cm"))
    assert transformed.dims != labeled_points.dims
    assert labeled_points.dims[1] == "mni"
    assert transformed.dims[1] == "other_crs"
    assert transformed.pint.units == units.cm
