"""Funtionality to work with segmented MRI scans."""

from typing import List

import numpy as np
import trimesh
import xarray as xr
from scipy.ndimage import binary_fill_holes
from skimage.measure import marching_cubes

import cedalion
import cedalion.dataclasses as cdc


def surface_from_segmentation(
    segmentation_mask: xr.DataArray,
    segmentation_types: List[str],
    isovalue=0.9,
    fill_holes_in_mask=False,
) -> cdc.Surface:
    """Create a surface from a segmentation mask.

    Args:
        segmentation_mask (xr.DataArray): Segmentation mask with dimensions segmentation
            type, i, j, k.
        segmentation_types (List[str]): A list of segmentation types to include in the
            surface.
        isovalue (Float): The isovalue to use for the marching cubes algorithm.
        fill_holes_in_mask (Bool): Whether to fill holes in the mask before creating the
            surface.

    Returns:
        A cedalion.Surface object.
    """
    combined_mask = (
        segmentation_mask.sel(segmentation_type=segmentation_types)
        .any("segmentation_type")
        .values
    )

    if fill_holes_in_mask:
        combined_mask = binary_fill_holes(combined_mask).astype(combined_mask.dtype)

    vertices, faces, normals, values = marching_cubes(combined_mask, isovalue)
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    mesh.fill_holes()
    mesh.fix_normals()

    return cdc.TrimeshSurface(mesh, "ijk", cedalion.units.Unit("1"))


def cell_coordinates(volume, flat: bool = False):
    """Create a DataArray with the coordinates of the cells in a volume.

    Args:
        volume (xr.DataArray): The volume to get the cell coordinates from.
        flat (bool): Whether to flatten the coordinates.

    Returns:
        xr.DataArray: A DataArray with the coordinates of the cells in the volume.
    """
    # coordinates in voxel space
    i = np.arange(volume.shape[0])
    j = np.arange(volume.shape[1])
    k = np.arange(volume.shape[2])

    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")

    coords = np.stack((ii, jj, kk), -1)  # shape (ni,nj,nk,3)

    if flat:
        iif = ii.flatten()
        jjf = jj.flatten()
        kkf = kk.flatten()
        coords = coords.reshape(-1, 3)

        coords = xr.DataArray(
            coords,
            dims=["label", "ijk"],
            coords={
                "label": ("label", np.arange(len(coords))),
                "i": ("label", iif),
                "j": ("label", jjf),
                "k": ("label", kkf),
            },
        )
        coords = coords.pint.quantify("1")
    else:
        coords = xr.DataArray(
            coords,
            dims=["i", "j", "k", "ijk"],
            coords={"i": i, "j": j, "k": k},
        )
        coords = coords.pint.quantify("1")

    return coords
