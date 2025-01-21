"""Utility functions for image reconstruction."""

import xarray as xr
import numpy as np
import cedalion
from cedalion import units
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.geometry.segmentation as segm
from scipy.sparse import coo_array
import scipy.stats

from cedalion import xrutils


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
    head_model: "cedalion.imagereco.forward_model.TwoSurfaceHeadModel",
    point: cdt.LabeledPointCloud,
    time_length: units.Quantity,
    sampling_rate: units.Quantity,
    spatial_size: units.Quantity,
    vmax: units.Quantity,
):
    """Create a mock activation below a point.

    Args:
        head_model (cedalion.imagereco.forward_model.TwoSurfaceHeadModel): The head
            model.
        point (cdt.LabeledPointCloud): The point below which to create the activation.
        time_length (units.Quantity): The length of the activation.
        sampling_rate (units.Quantity): The sampling rate.
        spatial_size (units.Quantity): The spatial size of the activation.
        vmax (units.Quantity): The maximum value of the activation.

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
