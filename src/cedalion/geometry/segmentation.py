"""Funtionality to work with segmented MRI scans."""

from typing import List

import os
import numpy as np
import nibabel as nib
import trimesh
import xarray as xr
import scipy.ndimage
import scipy.signal
from skimage.measure import marching_cubes, label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from functools import reduce

import cedalion
import cedalion.dataclasses as cdc


def voxels_from_segmentation(
    segmentation_mask: xr.DataArray,
    segmentation_types: List[str],
    isovalue=0.9,
    fill_holes_in_mask=False,
) -> cdc.Voxels:
    """Generate voxels from a segmentation mask.

    Args:
        segmentation_mask : xr.DataArray
            Segmentation mask.
        segmentation_types : List[str]
            List of segmentation types.
        isovalue : float, optional
            Isovalue for marching cubes, by default 0.9.
        fill_holes_in_mask : bool, optional
            Fill holes in the mask, by default False.

    Returns:
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
    cdc.TrimeshSurface
        Surface in voxel space.

    """
    if fill_holes_in_mask:
        combined_mask = scipy.ndimage.binary_fill_holes(combined_mask).astype(combined_mask.dtype)

    vertices, faces, normals, values = marching_cubes(combined_mask, isovalue)
    mesh = trimesh.Trimesh(vertices, faces, vertex_normals=normals)
    mesh.fill_holes()
    mesh.fix_normals()

    return cdc.TrimeshSurface(mesh, "ijk", cedalion.units.Unit("1"))


def cell_coordinates(volume, flat: bool = False):
    """Generate cell coordinates from a 3D volume.

    Parameters
    ----------
    volume : np.ndarray
        3D volume.
    flat : bool, optional
        If True, return coordinates as a flat array, by default False.

    Returns
    -------
    xr.DataArray
        Cell coordinates in voxel space.
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






def segmentation_postprocessing(
    segmentation_dir: str,
    mask_files: dict[str, str] = {
        "gray": "c1.nii",
        "white": "c2.nii",
        "csf": "c3.nii",
        "bone": "c4.nii",
        "skin": "c5.nii",
        "air": "c6.nii",
    },
    isSmooth: bool = True,
    fixCSF: bool = True,
    removeDisconnected: bool = True,
    labelUnassigned: bool = True,
    removeAir: bool = True,
    subtractTissues: bool = True
    ) -> dict:
    """ Postprocessing of the segmented SPM12 MRI segmentation files. 

    Parameters
    ----------
    segmentation_dir : str
        Directory where the segmented files are stored.
    mask_files : dict[str, str], optional   
        Dictionary containing the filenames of the segmented tissues.
    isSmooth : bool, optional
        Smooth the segmented tissues using Gaussian filter.
    fixCSF : bool, optional
        Fix the CSF continuity.
    removeDisconnected : bool, optional
        Remove disconnected voxels.
    labelUnassigned : bool, optional
        Label empty voxels to the nearest tissue type.
    removeAir : bool, optional
        Remove air cavities.
    subtractTissues : bool, optional
        Subtract tissues from each others

            
    Returns
    -------
    mask_files : dict
        Dictionary containing the filenames of the postprocessed masks.

    
    References
    ----------
    This whole postprocessing is based on the following references:
    :cite:t:`Huang2013`
    :cite:t:`Harmening2022`
    """

    # Load segmented spm output files
    tissues = mask_files.keys()
    img = {tissue: nib.load(os.path.join(segmentation_dir, mask_files[tissue]))
           for tissue in tissues}

    # Store affine transformations
    affine = {tissue: img[tissue].affine for tissue in tissues}

    # Load segmented image data
    img = {tissue: img[tissue].get_fdata() for tissue in tissues}

    # Smooth the segmented tissues by Gaussian filter
    if isSmooth:
        smt_fil = scipy.ndimage.gaussian_filter
        filter_size = 5
    
        for brain_tissue in ["gray", "whitegray", "cortex"]:
            if brain_tissue in tissues:
                print("Smoothing %s ..." % brain_tissue)
                sigma = 0.2
                truncate = ((filter_size - 1) / 2 - 0.5) / sigma
                for i in range(img[brain_tissue].shape[2]):
                    img[brain_tissue][:, :, i] = smt_fil(img[brain_tissue][:, :, i], sigma=sigma, truncate=truncate)

        if "white" in tissues:
            print("Smoothing WM ...")
            sigma = 0.1
            truncate = ((filter_size - 1) / 2 - 0.5) / sigma
            for i in range(img["white"].shape[2]):
                img["white"][:, :, i] = smt_fil(img["white"][:, :, i], sigma=sigma, truncate=truncate)

        for csf_tissue in ["csf", "brain"]:
            if csf_tissue in tissues:
                print("Smoothing %s ..." % csf_tissue)
                sigma = 0.1
                truncate = ((filter_size - 1) / 2 - 0.5) / sigma
                for i in range(img[csf_tissue].shape[2]):
                    img[csf_tissue][:, :, i] = smt_fil(img[csf_tissue][:, :, i], sigma=sigma, truncate=truncate)

        print("Smoothing bone ...")
        sigma = 0.4
        truncate = ((filter_size - 1) / 2 - 0.5) / sigma
        for i in range(img["bone"].shape[2]):
            img["bone"][:, :, i] = smt_fil(img["bone"][:, :, i], sigma=sigma, truncate=truncate)


        print("Smoothing skin ...")
        sigma = 1
        truncate = ((filter_size - 1) / 2 - 0.5) / sigma
        for i in range(img["skin"].shape[2]):
            img["skin"][:, :, i] = smt_fil(img["skin"][:, :, i], sigma=sigma, truncate=truncate)

        if "air" in tissues:
            print("Smoothing air ...")
            sigma = 1
            truncate = ((filter_size - 1) / 2 - 0.5) / sigma
            for i in range(img["air"].shape[2]):
                img["air"][:, :, i] = smt_fil(img["air"][:, :, i], sigma=sigma, truncate=truncate)

   
    print('Creating binary masks...')
    mask = binaryMaskGenerator(np.stack([img[tissue] for tissue in tissues]))
    img["empt"] = mask[0].astype(bool)
    for i, tissue in enumerate(tissues):
        img[tissue] = mask[i+1].astype(bool)

    # Fix CSF continuity
    if fixCSF: 
        print('Fixing CSF continuity...')
        assert "csf" in tissues, "CSF mask is not available"
        se = np.ones((3, 3, 3))
        dcsf = scipy.ndimage.binary_dilation(img["csf"], structure=se).astype(bool)
        dbone = scipy.ndimage.binary_dilation(img["bone"], structure=se).astype(bool)

        if "gray" in tissues:
            brain_tissue = "gray"
        elif "whitegray" in tissues:
            brain_tissue = "whitegray"
        elif "cortex" in tissues:
            brain_tissue = "cortex"
        elif "brain" in tissues:
            brain_tissue = "brain"
        else:
            raise ValueError("No brain tissue mask is available")

        contin = (img["empt"] & dcsf) | (dbone & img[brain_tissue])
        img["csf"] = img["csf"] | contin

        mask = binaryMaskGenerator(np.stack((img["csf"], img["bone"], img[brain_tissue])))
        img["csf"] = mask[1].astype(bool)
        img["bone"] = mask[2].astype(bool)
        img[brain_tissue] = mask[3].astype(bool)


    # Remove disconnected voxels
    if removeDisconnected:
        for brain_tissue in ["gray", "whitegray", "cortex"]:
            if brain_tissue in tissues:
                print('Removing disconnected voxels for %s...' % brain_tissue)
                thres = 30
                img[brain_tissue] = remove_small_objects(img[brain_tissue], min_size=thres)

        if "white" in tissues:
            print('Removing disconnected voxels for WM...')
            thres = 40
            img["white"] = remove_small_objects(img["white"], min_size=thres)

        for csf_tissue in ["csf", "brain"]:
            if csf_tissue in tissues:
                print('Removing disconnected voxels for %s...' % csf_tissue)
                siz, _ = sizeOfObject(img[csf_tissue])
                try:
                    thres = siz[3]+1
                except:
                    thres = 3
                img[csf_tissue] = remove_small_objects(img[csf_tissue], min_size=thres)

        print('Removing disconnected voxels for Bone...')
        thres = 300
        img["bone"] = remove_small_objects(img["bone"], min_size=thres)

        print('Removing disconnected voxels for Skin...')
        siz, _ = sizeOfObject(img["skin"])
        try:
            thres = siz[1]+1
        except:
            thres = 3
        img["skin"] = remove_small_objects(img["skin"], min_size=thres)

        if "air" in tissues:
            print('Removing disconnected voxels for Air...')
            thres = 20
            img["air"] = remove_small_objects(img["air"], min_size=thres)

    # Label unassigned voxels to the nearest tissue type
    if labelUnassigned:
        print('Generating and labeling empty voxels...')
        mask = binaryMaskGenerator(np.stack([img[tissue] for tissue in tissues]))
        img["empt"] = mask[0].astype(bool)

        # Generate unassigned voxels (empty voxels)
        # usually all empty voxels will be labelled in two loops
        for i in range(2):
            img_fil = {tissue: img[tissue].astype(float)*255 for tissue in tissues if tissue != "empt"}
            smt_fil = scipy.ndimage.gaussian_filter
            filter_size = 5
            sigma = 1
            truncate = ((filter_size - 1) / 2 - 0.5) / sigma

            for tissue in tissues:
                for i in range(img_fil[tissue].shape[2]):
                    img_fil[tissue][:, :, i] = smt_fil(img_fil[tissue][:, :, i], sigma=sigma, truncate=truncate)

            
            mask = binaryMaskGenerator(np.stack([img_fil[tissue] for tissue in tissues]))
            img_fil = {tissue: mask[i+1].astype(bool) for i, tissue in enumerate(tissues)}

            for tissue in tissues:
                img[tissue] = (img["empt"] & img[tissue]) | img_fil[tissue]
            
            empt_and_fil = [(img["empt"] & img_fil[tissue]) for tissue in tissues]
            img["empt"] = np.logical_xor(img["empt"], 
                                         reduce(np.logical_or, empt_and_fil))

            # Relabel each empty voxel to its nearest tissue type
            # The Gaussian filter is used to calculate distances, and max operation
            # relabels each empty voxel based on the distances.
    
    # remoe air cavities
    if removeAir:
        print('Removing outside air...')
        assert "air" in tissues, "Air mask is not available"

        # Logical XOR operation
        temp = np.logical_xor(img["air"], np.ones(img["air"].shape, dtype=bool))

        # Morphological closing
        se_close = np.ones((10, 10, 10), dtype=bool)
        temp = scipy.ndimage.binary_closing(temp, structure=se_close)

        # Fill holes
        temp = clear_border(temp)  # Clears objects connected to the border

        # Morphological erosion
        se_erode = np.ones((12, 12, 12), dtype=bool)
        temp = scipy.ndimage.binary_erosion(temp, structure=se_erode)
        img["air"] = np.logical_and(img["air"], temp)

    # Subtract tissues from each others 
    if subtractTissues:
        print('Subtracting tissues from each others...')
        # from in to outside
        if "gray" in tissues and "white" in tissues:
            img["gray"] = img["gray"] & (~img["white"])
            brain_tissue = "gray"
        elif "whitegray" in tissues:
            brain_tissue = "whitegray"
        elif "cortex" in tissues:
            brain_tissue = "cortex"
        elif "brain" in tissues:
            brain_tissue = "brain"
        else:
            raise ValueError("No brain tissue mask is available")
        img["csf"] = img["csf"] & (~img[brain_tissue])
        img["bone"] = img["bone"] & (~img["csf"])
        img["skin"] = img["skin"] & (~img["bone"])


    # Save masks
    for tissue in tissues:
        img[tissue] = nib.Nifti1Image(img[tissue].astype(float), affine[tissue])
        nib.save(img[tissue], os.path.join(segmentation_dir, ''.join(['mask_', tissue, '.nii'])))


    mask_files = {tissue: ''.join(['mask_', tissue, '.nii']) for tissue in tissues}
    return mask_files


def binaryMaskGenerator(d):
    d0 = np.zeros(np.concatenate(([1], d.shape[1:])))
    data = np.concatenate((d0, d), axis=0)
    max_ind = np.argmax(data,0)
    mask = np.zeros(shape=data.shape)
    for i in range(data.shape[0]):
        mask[i, :, :, :] = np.where(max_ind == i, 1, 0)
    return mask

def sizeOfObject(img, conn = None):
    if conn is None:
        if len(img.shape) == 2:
            conn = 8
        elif len(img.shape) == 3:
            conn = 26

    CC = label(img)  # Find connected components (equivalent to bwconncomp)

    size_obj = []
    for region in regionprops(CC):
        size_obj.append(len(region.coords))

    # Sort sizes in descending order
    size_descend = sorted(size_obj, reverse=True)
    ind = sorted(range(len(size_obj)), key=lambda k: size_obj[k], reverse=True)
    return size_descend, ind
