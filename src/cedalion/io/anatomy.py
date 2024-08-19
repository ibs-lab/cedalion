import nibabel
import xarray as xr
import os
from typing import Dict
import numpy as np
from cedalion.dataclasses import affine_transform_from_numpy

# FIXME
AFFINE_CODES = {
    0: "unknown",  # sform not defined
    1: "scanner",  # RAS+ in scanner coordinates
    2: "aligned",  # RAS+ aligned to some other scan
    3: "talairach",  # RAS+ in Talairach atlas space
    4: "mni",  # RAS+ in MNI atlas space
}


def _get_affine_from_niftii(image: nibabel.nifti1.Nifti1Image):
    transform, code = image.get_sform(coded=True)
    if code != 0:
        return affine_transform_from_numpy(
            transform, "ijk", AFFINE_CODES[code], "1", "mm"
        )

    transform, code = image.get_qform(coded=True)
    if code != 0:
        return affine_transform_from_numpy(
            transform, "ijk", AFFINE_CODES[code], "1", "mm"
        )

    transform = image.get_best_affine()

    return affine_transform_from_numpy(transform, "ijk", AFFINE_CODES[0], "1", "mm")


def read_segmentation_masks(
    basedir: str,
    mask_files: Dict[str, str] = {
        "csf": "csf.nii",
        "gm": "gm.nii",
        "scalp": "scalp.nii",
        "skull": "skull.nii",
        "wm": "wm.nii",
    },
) -> xr.DataArray:
    mask_ids = {seg_type: i + 1 for i, seg_type in enumerate(mask_files.keys())}
    masks = []
    affines = []

    for i, (seg_type, fname) in enumerate(mask_files.items()):
        fpath = os.path.join(basedir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"file '{fpath}' does not exist.")

        f = nibabel.load(fpath)

        volume = f.get_fdata()

        if len(np.unique(volume)) > 2:
            raise ValueError(
                "expected binary mask but found more than two unique "
                f"values in '{fpath}'"
            )

        # mask volume should contain integers stored as floating point numbers.
        # Operations like resampling can introduce small deviations and non-integer
        # mask ids -> round them.
        volume = volume.round(6).astype(np.uint8)

        volume[volume != 0] = mask_ids[seg_type]

        masks.append(
            xr.DataArray(
                volume,
                dims=["i", "j", "k"],
                coords={"segmentation_type": seg_type},
            )
        )

        affine = _get_affine_from_niftii(f)
        affines.append(affine)

    # check that sforms match for all masks
    for i in range(1, len(affines)):
        assert np.all(affines[i] == affines[i])

    masks = xr.concat(masks, dim="segmentation_type")

    # check for voxel that belong to more than one mask # FIXME too strict?
    if (masks > 0).sum("segmentation_type").max() > 1:
        raise ValueError("found voxels with positive entries in more than one mask.")

    affine = affines[0]

    return masks, affine

def read_parcellations(
    basedir: str, mask_file: str, filter_noncortical=False, gray_matter_seg = None
) -> xr.DataArray:
    
    parcellation_path = os.path.join(basedir, mask_file)
    if not os.path.exists(parcellation_path):
        raise FileNotFoundError(f"file '{parcellation_path}' does not exist.")
    
    if filter_noncortical and gray_matter_seg is None:
        raise TypeError("please provide your gray matter mask as input")

    f = nibabel.load(parcellation_path)

    volume = f.get_fdata()
    if filter_noncortical:
        # remove noncortical parcels
        gray_matter_seg = np.where(gray_matter_seg > 0, 1, 0)
        volume = volume*gray_matter_seg

    parcellation_labels = np.unique(volume)

    masks = []
    for i in parcellation_labels[1:]:
        p_volume = np.where(volume == i, 1, 0)
        p_volume = p_volume.round(6).astype(np.uint8)
        if np.any(p_volume):
            masks.append(
                xr.DataArray(
                    p_volume,
                    dims=["i", "j", "k"],
                    coords={"parcel_label": i},
                )
            )

    affine = _get_affine_from_niftii(f)
    masks = xr.concat(masks, dim="parcel_label")

    return masks, affine

def cell_coordinates(mask, affine, units="mm"):
    # coordinates in voxel space
    i = np.arange(mask.shape[0])
    j = np.arange(mask.shape[1])
    k = np.arange(mask.shape[2])

    ii, jj, kk = np.meshgrid(i, j, k, indexing="ij")

    coords = np.stack((ii, jj, kk), -1)  # shape (ni,nj,nk,3)
    transformed = xr.DataArray(
        nibabel.affines.apply_affine(affine, coords),
        dims=["i", "j", "k", "pos"],
        coords={"i": i, "j": j, "k": k},
        attrs={"units": units},
    )

    transformed = transformed.pint.quantify()

    return transformed
