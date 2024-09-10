"""Funtionality to work with segmented MRI scans."""

from typing import List

import numpy as np
import trimesh
import xarray as xr
from scipy.ndimage import binary_fill_holes
from skimage.measure import marching_cubes

import cedalion
import cedalion.dataclasses as cdc


def voxels_from_segmentation(
    segmentation_mask: xr.DataArray,
    segmentation_types: List[str],
    isovalue=0.9,
    fill_holes_in_mask=False,
) -> cdc.Voxels:
    """ Generate voxels from a segmentation mask.
    
    Parameters
    ----------
    segmentation_mask : xr.DataArray
        Segmentation mask.
    segmentation_types : List[str]
        List of segmentation types.
    isovalue : float, optional
        Isovalue for marching cubes, by default 0.9.
    fill_holes_in_mask : bool, optional
        Fill holes in the mask, by default False.
        
    Returns
    -------
    cdc.Voxels  
        Voxels in voxel space.
    """
    combined_mask = (
        segmentation_mask.sel(segmentation_type=segmentation_types)
        .any("segmentation_type")
        .values
    )

    if fill_holes_in_mask:
        combined_mask = binary_fill_holes(combined_mask).astype(combined_mask.dtype)

    voxels = np.argwhere(combined_mask)

    return cdc.Voxels(voxels, "ijk", cedalion.units.Unit("1"))


def surface_from_segmentation(
    segmentation_mask: xr.DataArray,
    segmentation_types: List[str],
    isovalue=0.9,
    fill_holes_in_mask=False,
) -> cdc.Surface:
	""" Generate a surface from a segmentation mask.
    Parameters
    ----------
    segmentation_mask : xr.DataArray
        Segmentation mask.
    segmentation_types : List[str]
        List of segmentation types.
    isovalue : float, optional
        Isovalue for marching cubes, by default 0.9.
    fill_holes_in_mask : bool, optional
        Fill holes in the mask, by default False.
    
    Returns
    -------
    cdc.Surface
        Surface in voxel space.
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
	"""Generate cell coordinates from a volume.
    Parameters
    ----------
    volume : np.ndarray
        Volume.
    flat : bool, optional
        Flatten the coordinates, by default False.
    Returns
    -------
    xr.DataArray
        Cell coordinates.
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
