"""Tests for VoxelHeadModel and the fluence-based voxel reducer."""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pytest
from scipy.sparse import find

import cedalion
import cedalion.data
import cedalion.dataclasses as cdc
from cedalion import units
import cedalion.dot as cdot
from cedalion.dot.voxel_head_model import VoxelHeadModel
from cedalion.io.forward_model import FluenceFile


def _allclose_sparse(A, B, atol=1e-8):
    if A.shape != B.shape:
        return False
    r1, c1, v1 = find(A)
    r2, c2, v2 = find(B)
    if not (np.array_equal(r1, r2) and np.array_equal(c1, c2)):
        return False
    return np.allclose(v1, v2, atol=atol)


@pytest.fixture(scope="module")
def colin27_voxel_head():
    """Build a VoxelHeadModel from the downsampled colin27 segmentation once."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        seg_dir, mask_files, landmarks = cedalion.data.get_colin27_segmentation(
            downsampled=True
        )
    return VoxelHeadModel.from_segmentation(
        segmentation_dir=seg_dir,
        mask_files=mask_files,
        landmarks_ras_file=landmarks,
        smoothing=0,
        scalp_face_count=None,
        max_dist=10 * units.mm,
    )


def test_from_segmentation_smoke(colin27_voxel_head):
    head = colin27_voxel_head

    assert head.crs == "ijk"
    assert isinstance(head.brain, cdc.Voxels)
    assert head.brain.nvertices > 0
    assert head.scalp.nvertices > 0
    assert head.brain_mask.dtype == bool
    assert head.brain_mask.ndim == 3
    seg_shape = tuple(head.segmentation_masks.shape[-3:])
    assert head.brain_mask.shape == seg_shape

    n_brain_voxels = head.brain.nvertices
    assert head.voxel_to_vertex_brain.shape == (
        int(np.prod(seg_shape)),
        n_brain_voxels,
    )
    # the number of nonzero entries equals the number of kept brain voxels
    assert head.voxel_to_vertex_brain.nnz == n_brain_voxels
    assert head.brain_mask.sum() == n_brain_voxels

    # alias property returns the same matrix object
    assert head.voxel_to_voxel_brain is head.voxel_to_vertex_brain


def test_save_load_round_trip(colin27_voxel_head):
    head = colin27_voxel_head

    def iu(x):
        return x.pint.dequantify().values

    with tempfile.TemporaryDirectory() as dirpath:
        tmp_folder = os.path.join(dirpath, "voxel_head")
        head.save(tmp_folder)
        head2 = VoxelHeadModel.load(tmp_folder)

        assert (head.landmarks == head2.landmarks).all()
        assert (head.segmentation_masks == head2.segmentation_masks).all()
        assert np.array_equal(head.brain.voxels, head2.brain.voxels)
        assert head.brain.crs == head2.brain.crs
        assert head.brain.units == head2.brain.units
        assert (head.scalp.mesh.vertices == head2.scalp.mesh.vertices).all()
        assert (head.scalp.mesh.faces == head2.scalp.mesh.faces).all()
        assert (iu(head.t_ijk2ras) == iu(head2.t_ijk2ras)).all()
        assert (iu(head.t_ras2ijk) == iu(head2.t_ras2ijk)).all()
        assert _allclose_sparse(
            head.voxel_to_vertex_brain, head2.voxel_to_vertex_brain
        )
        assert _allclose_sparse(
            head.voxel_to_vertex_scalp, head2.voxel_to_vertex_scalp
        )
        assert np.array_equal(head.brain_mask, head2.brain_mask)


def test_apply_transform_round_trip(colin27_voxel_head):
    head = colin27_voxel_head

    head_ras = head.apply_transform(head.t_ijk2ras)
    assert head_ras.brain.crs == head.t_ijk2ras.dims[0]

    head_back = head_ras.apply_transform(head.t_ras2ijk)
    np.testing.assert_allclose(
        head.brain.voxels, head_back.brain.voxels, atol=1e-6
    )
    np.testing.assert_allclose(
        head.scalp.mesh.vertices, head_back.scalp.mesh.vertices, atol=1e-6
    )


def test_reduce_voxels_to_probe(colin27_voxel_head):
    head = colin27_voxel_head

    # build a tiny "probe": a single point at the centre of the brain bbox in ijk
    centre = head.brain.voxels.mean(axis=0)
    geo3d = cdc.build_labeled_points(
        [centre.tolist()], crs="ijk", units="1"
    )
    geo3d = geo3d.assign_coords(
        type=("label", [cdc.PointType.SOURCE])
    )

    reduced = head.reduce_voxels_to_probe(geo3d, max_dist=20 * units.mm)
    assert reduced.brain.nvertices < head.brain.nvertices
    assert reduced.brain.nvertices > 0
    assert reduced.voxel_to_vertex_brain.shape[1] == reduced.brain.nvertices
    assert reduced.voxel_to_vertex_brain.nnz == reduced.brain.nvertices
    assert reduced.brain_mask.sum() == reduced.brain.nvertices

    # ensure the reduced 3D mask is a strict subset of the original
    assert np.all(head.brain_mask | ~reduced.brain_mask)


def test_reduce_voxels_to_sensitivity(colin27_voxel_head):
    head = colin27_voxel_head
    n_brain = head.brain.nvertices
    n_scalp = head.scalp.nvertices

    rng = np.random.default_rng(0)
    # construct a synthetic Adot where the first half of brain voxels is bright
    Adot_brain = np.zeros((2, n_brain, 1))
    bright = rng.choice(n_brain, size=n_brain // 2, replace=False)
    Adot_brain[:, bright, 0] = 1.0
    Adot_scalp = np.zeros((2, n_scalp, 1))

    is_brain = np.zeros(n_brain + n_scalp, dtype=bool)
    is_brain[:n_brain] = True

    import xarray as xr

    Adot = xr.DataArray(
        np.concatenate([Adot_brain, Adot_scalp], axis=1).astype(np.float32),
        dims=["channel", "vertex", "wavelength"],
        coords={
            "channel": ("channel", ["S1D1", "S1D2"]),
            "wavelength": ("wavelength", [760.0]),
            "is_brain": ("vertex", is_brain),
        },
    )

    reduced, Adot_reduced = head.reduce_voxels_to_sensitivity(
        Adot, sensitivity_threshold=0.5
    )
    assert reduced.brain.nvertices == len(bright)
    assert int(Adot_reduced.is_brain.sum()) == reduced.brain.nvertices
    assert int((~Adot_reduced.is_brain).sum()) == int((~Adot.is_brain).sum())
    assert Adot_reduced.sizes["vertex"] == reduced.brain.nvertices + n_scalp


def test_reduce_voxels_by_fluence(colin27_voxel_head, tmp_path):
    head = colin27_voxel_head
    seg_shape = tuple(head.segmentation_masks.shape[-3:])

    # build a synthetic fluence volume that is 1.0 in a sub-bounding-box of
    # the brain voxels and 0 elsewhere
    flat_indices = np.flatnonzero(head.brain_mask)
    keep_n = max(1, len(flat_indices) // 4)
    keep_idx = flat_indices[:keep_n]

    fluence_vol = np.zeros(seg_shape, dtype=np.float32)
    fluence_vol.flat[keep_idx] = 1.0

    optode_labels = np.array(["S1", "D1"])
    wavelengths = np.array([760.0])

    fluence_path = tmp_path / "fluence.h5"
    import xarray as xr
    optode_pos = xr.DataArray(
        np.zeros((2, 3)),
        dims=["label", "ijk"],
        coords={
            "label": ("label", optode_labels),
            "type": ("label", [cdc.PointType.SOURCE, cdc.PointType.DETECTOR]),
        },
        attrs={"units": "1"},
    ).pint.quantify()

    with FluenceFile(fluence_path, "w") as ff:
        ff.create_fluence_dataset(
            optode_pos, wavelengths, seg_shape, "1 / millimeter ** 2"
        )
        ff.set_fluence_by_index(0, 0, fluence_vol)
        ff.set_fluence_by_index(1, 0, fluence_vol)
        # fluence_at_optodes is required for the file structure but not used here
        fa = xr.DataArray(
            np.ones((2, 2, 1)),
            dims=["optode1", "optode2", "wavelength"],
            coords={
                "optode1": optode_labels,
                "optode2": optode_labels,
                "wavelength": wavelengths,
            },
        )
        ff.set_fluence_at_optodes(fa)

    reduced = head.reduce_voxels_by_fluence(fluence_path, rel_threshold=0.5)
    assert reduced.brain.nvertices == keep_n
    # row indices in the new mapping correspond to the kept voxels
    new_flat = np.flatnonzero(reduced.brain_mask)
    np.testing.assert_array_equal(np.sort(new_flat), np.sort(keep_idx))


def test_get_standard_headmodel_voxel_kind():
    """``kind="voxel"`` returns a VoxelHeadModel built from the colin27 atlas."""
    head = cdot.get_standard_headmodel("colin27", kind="voxel")
    assert isinstance(head, VoxelHeadModel)
    assert head.brain.nvertices > 0
    assert head.scalp.nvertices > 0


def test_get_standard_headmodel_invalid_kind():
    with pytest.raises(ValueError, match="Unknown kind"):
        cdot.get_standard_headmodel("colin27", kind="bogus")
