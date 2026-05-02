"""Utility functions for image reconstruction."""

import numpy as np
import scipy.stats
import xarray as xr
from scipy.sparse import coo_array

import cedalion
import cedalion.dataclasses as cdc
import cedalion.geometry.segmentation as segm
import cedalion.typing as cdt
from cedalion import units, xrutils

# FIXME right location?
def map_segmentation_mask_to_surface(
    segmentation_mask: xr.DataArray,
    transform_vox2ras: cdt.AffineTransform,  # FIXME
    surface: cdc.Surface,
):
    """Find for each voxel the closest vertex on the surface.

    Args:
        segmentation_mask (xr.DataArray): A binary mask of shape (segmentation_type, i,
            j, k).
        transform_vox2ras (xr.DataArray): The affine transformation from voxel to RAS
            space.
        surface (cedalion.dataclasses.Surface): The surface to map the voxels to.

    Returns:
        coo_array: A sparse matrix of shape (ncells, nvertices) that maps voxels to
            cells.
    """

    assert surface.crs == transform_vox2ras.dims[0]

    cell_coords = segm.cell_coordinates(segmentation_mask, flat=True)
    cell_coords = cell_coords.points.apply_transform(transform_vox2ras)

    cell_coords = cell_coords.pint.to(surface.units).pint.dequantify()

    ncells = cell_coords.sizes["label"]
    nvertices = len(surface.vertices)

    # find indices of cells that belong to the mask
    cell_indices = np.flatnonzero(segmentation_mask.values)

    # for each cell query the closests vertex on the surface
    dists, vertex_indices = surface.kdtree.query(
        cell_coords.values[cell_indices, :], workers=-1
    )

    # construct a sparse matrix of shape (ncells, nvertices)
    # that maps voxels to cells
    map_voxel_to_vertex = coo_array(
        (np.ones(len(cell_indices)), (cell_indices, vertex_indices)),
        shape=(ncells, nvertices),
    )

    return map_voxel_to_vertex


def reduce_and_map_brain_voxels(
    brain_mask: xr.DataArray,
    transform_vox2ras: cdt.AffineTransform,
    scalp_surface: cdc.Surface,
    brain_volume: cdc.Voxels,
    max_dist: cdt.QLength = 10 * units.mm,
) -> tuple[coo_array, cdc.Voxels, np.ndarray]:
    """Filter brain voxels by distance to the scalp and build a sparse mapping.

    For each brain voxel, the distance to the closest vertex of the scalp
    surface is computed.  Voxels farther away than ``max_dist`` are dropped.
    The remaining voxels form the reduced brain volume; a sparse matrix maps
    every cell of the original segmentation grid to its index in that
    reduced volume (or to nothing, if the voxel was dropped).

    Args:
        brain_mask: 3D boolean (or 0/1) ``xr.DataArray`` of the brain
            segmentation in voxel space. Shape ``(i, j, k)``.
        transform_vox2ras: Affine transform from voxel (ijk) space to the
            CRS in which ``scalp_surface`` and ``brain_volume`` live.
        scalp_surface: Scalp surface mesh, in the target CRS.
        brain_volume: Brain voxels, in the target CRS, in the same row order
            as ``np.argwhere(brain_mask.values)``.
        max_dist: Maximum allowed scalp-to-voxel distance.

    Returns:
        Tuple ``(map_voxel_to_voxel, reduced_volume, kept_mask_3d)``:

        - **map_voxel_to_voxel**: ``coo_array`` of shape
          ``(ncells, nvoxels_reduced)`` mapping flat segmentation cells to
          reduced voxel indices.
        - **reduced_volume**: ``cdc.Voxels`` containing only kept brain
          voxels, in the same CRS/units as the input ``brain_volume``.
        - **kept_mask_3d**: 3D boolean ``np.ndarray`` aligned to
          ``brain_mask.shape``; True where a voxel is kept.
    """

    assert scalp_surface.crs == transform_vox2ras.dims[0]
    assert brain_volume.crs == transform_vox2ras.dims[0]
    assert scalp_surface.units == brain_volume.units

    max_dist_in_surface_units = float(max_dist.to(scalp_surface.units).magnitude)

    # voxel-centre coordinates of every cell in the segmentation grid,
    # transformed to the scalp/brain CRS and dequantified to the surface units
    cell_coords = segm.cell_coordinates(brain_mask, flat=True)
    cell_coords = cell_coords.points.apply_transform(transform_vox2ras)
    cell_coords = cell_coords.pint.to(scalp_surface.units).pint.dequantify()

    ncells = cell_coords.sizes["label"]

    # flat indices of cells that belong to the brain mask; this is the same
    # ordering used by `voxels_from_segmentation` (np.argwhere is row-major)
    cell_indices_brain = np.flatnonzero(brain_mask.values)

    # for each brain cell, distance to the closest scalp surface vertex
    dists, _ = scalp_surface.kdtree.query(
        cell_coords.values[cell_indices_brain, :], workers=-1
    )

    kept = dists < max_dist_in_surface_units
    nvoxels_reduced = int(kept.sum())

    reduced_volume = cdc.Voxels(
        brain_volume.voxels[kept], brain_volume.crs, brain_volume.units
    )

    # build map: each kept brain cell -> its index in the reduced volume.
    # The kdtree query is a sanity check (each kept voxel should match itself).
    cell_indices_kept = cell_indices_brain[kept]
    dists_self, voxel_indices = reduced_volume.kdtree.query(
        cell_coords.values[cell_indices_kept, :], workers=-1
    )
    assert (dists_self.round(9) == 0.0).all()

    map_voxel_to_voxel = coo_array(
        (np.ones(len(cell_indices_kept)), (cell_indices_kept, voxel_indices)),
        shape=(ncells, nvoxels_reduced),
    )

    kept_mask_3d = np.zeros(brain_mask.shape, dtype=bool)
    kept_mask_3d.flat[cell_indices_kept] = True

    return map_voxel_to_voxel, reduced_volume, kept_mask_3d


def normal_hrf(t, t_peak, t_std, vmax):
    """Create a normal hrf.

    Args:
        t (np.ndarray): The time points.
        t_peak (float): The peak time.
        t_std (float): The standard deviation.
        vmax (float): The maximum value of the HRF.

    Returns:
        np.ndarray: The HRF.
    """
    hrf = scipy.stats.norm.pdf(t, loc=t_peak, scale=t_std)
    hrf *= vmax / hrf.max()
    return hrf


def create_mock_activation_below_point(
    head_model: "cedalion.dot.TwoSurfaceHeadModel",
    point: cdt.LabeledPoints,
    time_length: cdt.QTime,
    sampling_rate: cdt.QFrequency,
    spatial_size: cdt.QLength,
    vmax: float,
):
    """Create a mock activation below a point.

    Args:
        head_model: The head model.
        point: The point below which to create the activation.
        time_length: The length of the activation.
        sampling_rate: The sampling rate.
        spatial_size: The spatial size of the activation.
        vmax: The maximum value of the activation.

    Returns:
        xr.DataArray: The activation.
    """
    # assert head_model.crs == point.points.crs

    _, vidx = head_model.brain.kdtree.query(point)

    # FIXME for simplicity use the euclidean distance here whilw the geodesic distance
    # would be the correct choice
    dists = xrutils.norm(
        head_model.brain.vertices - head_model.brain.vertices[vidx, :],
        head_model.brain.crs,
    )

    nsamples = int((time_length * sampling_rate).to_reduced_units().magnitude.item())
    t = np.arange(nsamples) / sampling_rate

    func_spat = np.exp(-((dists / spatial_size) ** 2)).rename({"label": "vertex"})
    func_temp = xr.DataArray(normal_hrf(t, 10, 3, vmax), dims="time")

    activation = func_temp * func_spat
    activation = activation.assign_coords({"time": t})
    return activation



# affine transformation from MNI305 (Colin27, fsaverage) to MNI152 (ICBM-152)
# see 8. in https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems
mni305_to_mni152 = cdc.affine_transform_from_numpy(
    np.array([
        [ 0.9975, -0.0073,  0.0176, -0.0429],
        [ 0.0146,  1.0009, -0.0024,  1.5496],
        [-0.0130, -0.0093,  0.9971,  1.1840],
        [ 0,       0,       0,       1     ]
    ]),
    from_crs ="mni305",
    to_crs="mni152",
    from_units="mm",
    to_units="mm"
)

