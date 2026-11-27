import os
import tempfile

import nibabel as nib
import numpy as np
import pytest

import cedalion.dataclasses as cdc
from cedalion.io.anatomy import read_segmentation_masks


def test_read_segmentation_masks_rejects_mismatching_affines():
    """Masks with different affine transforms raise a ValueError.

    Regression test: the consistency check compared each affine with itself.
    """
    mask = np.zeros((5, 5, 5), np.uint8)
    mask[1:4, 1:4, 1:4] = 1
    affine_a = np.eye(4)
    affine_b = np.eye(4)
    affine_b[0, 3] = 10.0

    with tempfile.TemporaryDirectory() as tmp_dir:
        nib.save(nib.Nifti1Image(mask, affine_a), os.path.join(tmp_dir, "a.nii"))
        nib.save(nib.Nifti1Image(mask, affine_a), os.path.join(tmp_dir, "a2.nii"))
        nib.save(nib.Nifti1Image(mask, affine_b), os.path.join(tmp_dir, "b.nii"))

        # identical affines are fine (overlapping masks are not, hence disjoint masks)
        mask2 = np.zeros((5, 5, 5), np.uint8)
        mask2[0, 0, 0] = 1
        nib.save(nib.Nifti1Image(mask2, affine_a), os.path.join(tmp_dir, "c.nii"))
        read_segmentation_masks(tmp_dir, {"gm": "a.nii", "csf": "c.nii"})

        with pytest.raises(ValueError):
            read_segmentation_masks(tmp_dir, {"gm": "a.nii", "csf": "b.nii"})


def test_points_apply_transform_keeps_unit_attribute():
    """Dequantified points with a units attribute keep it through apply_transform.

    Regression test: an operator-precedence bug in the walrus expression stored
    True/False as the units attribute.
    """
    points = cdc.build_labeled_points(
        np.array([[1.0, 2.0, 3.0]]),
        labels=["A"],
        types=[cdc.PointType.LANDMARK],
        crs="pos",
        units="mm",
    ).pint.dequantify()
    assert points.attrs["units"] == "millimeter"

    transformed = points.points.apply_transform(np.eye(4))
    assert transformed.attrs["units"] == "millimeter"

    without_units = points.copy()
    del without_units.attrs["units"]
    transformed = without_units.points.apply_transform(np.eye(4))
    assert "units" not in transformed.attrs
