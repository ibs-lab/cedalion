"""Tests for geometry registration helpers."""

import numpy as np

import cedalion.dataclasses as cdc
from cedalion.geometry.registration import simple_scalp_projection


def _make_projection_geometry(nasion_label: str):
    """Build a minimal geometry for testing simple scalp projection."""
    return cdc.build_labeled_points(
        coordinates=np.array(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        crs="pos",
        units="mm",
        labels=[
            "LPA",
            "RPA",
            nasion_label,
            "S1",
        ],
        types=[
            cdc.PointType.LANDMARK,
            cdc.PointType.LANDMARK,
            cdc.PointType.LANDMARK,
            cdc.PointType.SOURCE,
        ],
    )


def test_simple_scalp_projection_accepts_nasion_alias():
    """Treat NASION as an alias for the Nz landmark."""
    projection_nz = simple_scalp_projection(_make_projection_geometry("Nz"))
    projection_nasion = simple_scalp_projection(_make_projection_geometry("NASION"))

    np.testing.assert_allclose(
        projection_nasion.values,
        projection_nz.values,
    )

    assert projection_nasion.label.values.tolist() == [
        "LPA",
        "RPA",
        "NASION",
        "S1",
    ]
