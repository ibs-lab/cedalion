"""Functions for reading and processing anatomical data."""

import gzip
import os
from dataclasses import dataclass
from typing import Dict, Tuple
from pathlib import Path
import json

import nibabel
import nibabel.freesurfer
import numpy as np
import pandas as pd
import scipy.io
import xarray as xr
import trimesh

import cedalion
from cedalion.dataclasses import affine_transform_from_numpy, TrimeshSurface

# FIXME
AFFINE_CODES = {
    0: "unknown",  # sform not defined
    1: "scanner",  # RAS+ in scanner coordinates
    2: "aligned",  # RAS+ aligned to some other scan
    3: "talairach",  # RAS+ in Talairach atlas space
    4: "mni",  # RAS+ in MNI atlas space
}


def _get_affine_from_niftii(image: nibabel.nifti1.Nifti1Image, crs: str | None = None):
    """Get affine transformation matrix from NIFTI image.

    Args:
        image (nibabel.nifti1.Nifti1Image): NIFTI image object
        crs: If provided this sets the the name of the CRS. Otherwise it is inferred
            from AFFINE_CODES.

    Returns:
        xr.DataArray: Affine transformation matrix
    """
    transform, code = image.get_sform(coded=True)
    if code != 0:
        crs_ = AFFINE_CODES[code] if crs is None else crs
        return affine_transform_from_numpy(transform, "ijk", crs_, "1", "mm")

    transform, code = image.get_qform(coded=True)
    if code != 0:
        crs_ = AFFINE_CODES[code] if crs is None else crs
        return affine_transform_from_numpy(transform, "ijk", crs_, "1", "mm")

    transform = image.get_best_affine()

    crs_ = AFFINE_CODES[0] if crs is None else crs
    return affine_transform_from_numpy(transform, "ijk", crs_, "1", "mm")


HEADMODEL_MANIFEST = "headmodel_manifest.json"


@dataclass
class HeadModelFiles:
    """Files of a head model, as listed in its ``headmodel_manifest.json``.

    File names are relative to ``basedir``, the folder containing the manifest.
    Apart from the segmentation masks all files are optional. Missing surfaces and
    voxel-to-vertex mappings are derived from the segmentation masks when the head
    model is constructed (see ``TwoSurfaceHeadModel.from_manifest``).

    Attributes:
        basedir: Folder containing the head model files.
        mask_files: Mapping of segmentation types to NIfTI mask files.
        name: Name of the head model.
        landmarks_ras_file: Landmarks in RAS space (Slicer ``.mrk.json``).
        brain_vertex_coordinates: CSV with additional brain vertex coordinates,
            e.g. parcel labels.
        scalp_surface_obj: Scalp surface in voxel coordinates.
        brain_surface_obj: Brain surface in voxel coordinates.
        freesurfer_surface_obj: High-resolution pial surface.
        inflated_surface_obj: Inflated pial surface.
        parcel_colors: JSON file mapping parcel names to colors.
        voxel_to_vertex_mapping: Precomputed brain voxel-to-vertex mapping.
        surfaces_are_zero_based: Whether the voxel coordinates of the surfaces are
            zero-based (see ``TwoSurfaceHeadModel.from_surfaces``).
        ras_crs: If given, the name of the RAS coordinate system, replacing the name
            derived from the NIfTI header.
    """

    basedir : Path
    mask_files : dict[str, str]
    name : str | None = None
    landmarks_ras_file : str | None = None
    brain_vertex_coordinates : str | None = None
    scalp_surface_obj : str | None = None
    brain_surface_obj : str | None = None
    freesurfer_surface_obj : str | None = None
    inflated_surface_obj : str | None = None
    parcel_colors : str | None = None
    voxel_to_vertex_mapping : str | None = None
    surfaces_are_zero_based : bool = False
    ras_crs : str | None = None

    @classmethod
    def from_manifest(cls, fname: str | Path) -> "HeadModelFiles":
        """Read a head model manifest.

        Args:
            fname: Path to a ``headmodel_manifest.json``.

        Returns:
            The head model files with ``basedir`` set to the manifest's folder.
        """
        fname = Path(fname)
        with fname.open("r") as fin:
            manifest = json.load(fin)

        format_version = manifest.pop("format_version", None)
        if format_version != 1:
            raise ValueError(
                f"{fname}: unsupported manifest format_version {format_version!r}."
            )

        return cls(basedir=fname.parent, **manifest)

    def path(self, attr: str) -> Path | None:
        """Return the full path of a file entry or None if it is not set."""
        fname = getattr(self, attr)
        return None if fname is None else self.basedir / fname

    def load_parcel_colors(self):
        with (self.basedir / self.parcel_colors).open("r") as fin:
            return json.load(fin)

    def load_voxel_to_vertex_mapping(self):
        with gzip.GzipFile(self.basedir / self.voxel_to_vertex_mapping) as fin:
            return scipy.io.mmread(fin)

    def load_brain_vertex_coordinates(self):
        return pd.read_csv(self.basedir / self.brain_vertex_coordinates)


def read_segmentation_masks(
    basedir: str,
    mask_files: Dict[str, str] = {
        "csf": "csf.nii",
        "gm": "gm.nii",
        "scalp": "scalp.nii",
        "skull": "skull.nii",
        "wm": "wm.nii",
    },
) -> Tuple[xr.DataArray, np.ndarray]:
    """Read segmentation masks from NIFTI files.

    Args:
        basedir (str): Directory containing the mask files
        mask_files (Dict[str, str]): Dictionary mapping segmentation types to filenames

    Returns:
        Tuple[xr.DataArray, np.ndarray]:
            - masks (xr.DataArray): Concatenated segmentation masks with a new
              dimension `segmentation_type`.
            - affine (np.ndarray): Affine transformation matrix associated with the
              NIFTI files.
    """
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


def cell_coordinates(mask, affine, units="mm"):
    """Get the coordinates of each voxel centre in the transformed mask.

    Following NIfTI convention, voxel coordinates in ijk-space denote the centre of
    that voxel and the affine maps it to the centre's world coordinates. (See also
    nifti1.h: "The (x,y,z) coordinates refer to the CENTER of a voxel.")

    Args:
        mask (xr.DataArray): A binary mask of shape (i, j, k).
        affine (np.ndarray): Affine transformation matrix.
        units (str): Units of the output coordinates.

    Returns:
        xr.DataArray: Coordinates of the center of each voxel in the mask.
    """
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


def read_parcellations(parcel_file: str | Path) -> pd.DataFrame:
    """Read parcellation labels from a json file.

    Args:
        parcel_file: The parcels file name

    Returns:
        pd.DataFrame: Contains vertices' labels, their appropriate colors
    """
    parcels = pd.read_json(parcel_file)

    parcels = parcels.explode("Vertices")
    parcels["Vertices"] = parcels["Vertices"].astype(int)
    parcels = parcels.sort_values("Vertices")

    parcels["Label"] = parcels["Label"].apply(lambda x: "_".join(x.split(" ")) + "H")

    return parcels

def read_parcel_colors(parcel_colors_file : str | Path) -> dict[str, list]:
    """Read parcel RGB colors from a JSON file.

    Args:
        parcel_colors_file: Path to the JSON file mapping parcel names to
            RGB color lists.

    Returns:
        Dictionary mapping parcel label strings to RGB color lists.
    """
    with open(parcel_colors_file) as fin:
        return json.load(fin)


def trimesh_from_freesurfer(
    lh: str | Path,
    rh: str | Path,
    crs: str,
    units: str,
    offset_hemispheres: bool = True,
) -> TrimeshSurface:
    """Load freesurfer surfaces as a cdc.TrimeshSurface object.

    Args:
        lh: path of the left hemisphere surface file.
        rh: path of the right hemisphere surface file.
        crs: label of the coordinate reference system
        units: units of vertex coordinates
        offset_hemispheres: If True, offset the hemispheres along the x-axis such that
            the left hemisphere's maximum x-coordinate aligns with the right
            hemisphere's minimum x-coordinate, removing the overlap that occurs with
            inflated surfaces.

    Returns:
        A single TrimeshSurface containing both hemispheres, with left-hemisphere
        vertices preceding right-hemisphere vertices in the vertex array.
    """

    units = cedalion.units.Unit(units)

    lh_v, lh_e = nibabel.freesurfer.read_geometry(lh)
    rh_v, rh_e = nibabel.freesurfer.read_geometry(rh)
    lh_v[:,0] -= lh_v[:,0].max()
    rh_v[:,0] -= rh_v[:,0].min()
    v = np.vstack((lh_v, rh_v))
    e = np.vstack( (lh_e, rh_e+len(lh_v)))

    return TrimeshSurface(trimesh.Trimesh(v, e), crs, units)
