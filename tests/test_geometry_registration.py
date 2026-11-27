import numpy as np
import pytest
from scipy.spatial.transform import Rotation

import cedalion.dataclasses as cdc
import cedalion.geometry.registration as reg


def _fiducials(crs="pos", units="mm"):
    """Five head fiducials in mm."""
    points = np.array(
        [
            [0.0, 95.0, -20.0],  # Nz
            [0.0, -95.0, -10.0],  # Iz
            [-75.0, 0.0, -30.0],  # LPA
            [75.0, 0.0, -30.0],  # RPA
            [0.0, 0.0, 95.0],  # Cz
        ]
    )
    return cdc.build_labeled_points(
        points,
        labels=["Nz", "Iz", "LPA", "RPA", "Cz"],
        types=[cdc.PointType.LANDMARK] * 5,
        crs=crs,
        units=units,
    )


def _rms(a, b):
    a = a.pint.dequantify().values
    b = b.pint.dequantify().values
    return np.sqrt(((a - b) ** 2).sum(axis=1).mean())


@pytest.mark.parametrize("scale", [1.0, 1.15])
@pytest.mark.parametrize(
    "rotvec",
    [
        [0.0, 0.0, np.pi],  # 180 deg about z
        [np.pi, 0.0, 0.0],  # 180 deg about x
        [0.5, -2.5, 1.5],
        [0.1, 0.2, 0.3],
    ],
)
def test_register_trans_rot_isoscale_large_rotations(rotvec, scale):
    """The similarity transform is recovered exactly for any rotation.

    Regression test: the optimizer started at zero rotation and converged to a
    wrong local minimum for rotations above ~135 degrees (rms errors of 50-90 mm for
    180 degree flips, which are common between digitizer and head coordinate
    systems).
    """
    source = _fiducials(crs="dig")
    R = Rotation.from_rotvec(rotvec).as_matrix()
    t = np.array([10.0, -20.0, 30.0])

    target_values = scale * (source.pint.dequantify().values @ R.T) + t
    target = cdc.build_labeled_points(
        target_values,
        labels=source.label.values,
        types=[cdc.PointType.LANDMARK] * 5,
        crs="pos",
        units="mm",
    )

    trafo = reg.register_trans_rot_isoscale(target, source)
    assert _rms(source.points.apply_transform(trafo), target) < 1e-8

    M = trafo.pint.dequantify().values
    np.testing.assert_allclose(np.linalg.det(M[:3, :3]), scale**3, rtol=1e-9)

    if scale == 1.0:
        trafo = reg.register_trans_rot(target, source)
        assert _rms(source.points.apply_transform(trafo), target) < 1e-8


def test_register_trans_rot_converts_units():
    source = _fiducials(crs="dig", units="mm")
    target = _fiducials(crs="pos", units="cm")
    target = target.pint.dequantify() * 0.1
    target = target.pint.quantify("cm")

    trafo = reg.register_trans_rot(target, source)
    aligned = source.points.apply_transform(trafo)
    assert aligned.pint.units == target.pint.units
    assert _rms(aligned, target) < 1e-9
