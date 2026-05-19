"""Two-surface head model and standard atlas loading for DOT forward modelling."""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import nibabel
import numpy as np
import pandas as pd
import scipy
import scipy.sparse
import trimesh
import xarray as xr
from scipy.spatial import KDTree

import cedalion
import cedalion.data
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import xrutils
from cedalion.dot.utils import map_segmentation_mask_to_surface
from cedalion.geometry.ellipsoid import get_landmarks_for_headsize
from cedalion.geometry.registration import (
    register_general_affine,
    register_trans_rot_isoscale,
    register_optodes_spring_icp,
    register_identity
)
from cedalion.geometry.segmentation import (
    surface_from_segmentation,
    voxels_from_segmentation,
)
from cedalion.io import read_mrk_json, read_segmentation_masks


@dataclass
class TwoSurfaceHeadModel:
    """Head Model class to represent a segmented head.

    Its main functions are reduced to work on voxel projections to scalp and cortex
    surfaces.

    Attributes:
        segmentation_masks : xr.DataArray
            Segmentation masks of the head for each tissue type.
        brain : cdc.Surface
            Surface of the brain.
        scalp : cdc.Surface
            Surface of the scalp.
        landmarks : cdt.LabeledPoints
            Anatomical landmarks in RAS space.
        t_ijk2ras : cdt.AffineTransform
            Affine transformation from ijk to RAS space.
        t_ras2ijk : cdt.AffineTransform
            Affine transformation from RAS to ijk space.
        voxel_to_vertex_brain : scipy.sparse.spmatrix
            Mapping from voxel to brain vertices.
        voxel_to_vertex_scalp : scipy.sparse.spmatrix
            Mapping from voxel to scalp vertices.
        crs : str
            Coordinate reference system of the head model.

    Methods:
        from_segmentation(cls, segmentation_dir, mask_files, landmarks_ras_file,
            brain_seg_types, scalp_seg_types, smoothing, brain_face_count,
            scalp_face_count): Construct instance from segmentation masks in NIfTI
            format.
        apply_transform(transform)
            Apply a coordinate transformation to the head model.
        save(foldername)
            Save the head model to a folder.
        load(foldername)
            Load the head model from a folder.
        align_and_snap_to_scalp(points)
            Align and snap optodes or points to the scalp surface.
    """

    segmentation_masks: xr.DataArray
    brain: cdc.Surface
    scalp: cdc.Surface
    landmarks: cdt.LabeledPoints
    t_ijk2ras: cdt.AffineTransform
    t_ras2ijk: cdt.AffineTransform
    voxel_to_vertex_brain: scipy.sparse.spmatrix
    voxel_to_vertex_scalp: scipy.sparse.spmatrix

    # FIXME need to distinguish between ijk,  ijk+units == aligned == ras

    @classmethod
    def from_segmentation(
        cls,
        segmentation_dir: str,
        mask_files: dict[str, str] = {
            "csf": "csf.nii",
            "gm": "gm.nii",
            "scalp": "scalp.nii",
            "skull": "skull.nii",
            "wm": "wm.nii",
        },
        landmarks_ras_file: str | None = None,
        brain_seg_types: list[str] = ["gm", "wm"],
        scalp_seg_types: list[str] = ["scalp"],
        smoothing: float = 0.,
        brain_face_count: int | None = 180000,
        scalp_face_count: int | None = 60000,
        fill_holes: bool = True,
    ) -> "TwoSurfaceHeadModel":
        """Constructor from binary masks as gained from segmented MRI scans.

        Args:
            segmentation_dir: Folder containing the segmentation masks in NIFTI
                format.
            mask_files: Dictionary mapping segmentation types to NIFTI filenames.
            landmarks_ras_file: Filename of the landmarks in RAS space.
            brain_seg_types: List of segmentation types to be included in
                the brain surface.
            scalp_seg_types: List of segmentation types to be included in
                the scalp surface.
            smoothing: Smoothing factor for the brain and scalp surfaces.
            brain_face_count: Number of faces for the brain surface.
            scalp_face_count: Number of faces for the scalp surface.
            fill_holes: Whether to fill holes in the segmentation masks.
        """

        # load segmentation mask
        segmentation_masks, t_ijk2ras = read_segmentation_masks(
            segmentation_dir, mask_files
        )

        # inspect and invert ijk-to-ras transformation
        t_ras2ijk = xrutils.pinv(t_ijk2ras)

        # crs_ijk = t_ijk2ras.dims[1]
        crs_ras = t_ijk2ras.dims[0]

        # load landmarks. Other than the segmentation masks which are in voxel (ijk)
        # space, these are already in RAS space.
        if landmarks_ras_file is not None:
            if not os.path.isabs(landmarks_ras_file):
                landmarks_ras_file = os.path.join(segmentation_dir, landmarks_ras_file)

            landmarks_ras = read_mrk_json(landmarks_ras_file, crs=crs_ras)
            landmarks_ijk = landmarks_ras.points.apply_transform(t_ras2ijk)
        else:
            landmarks_ijk = None

        # derive surfaces from segmentation masks
        brain_ijk = surface_from_segmentation(
            segmentation_masks, brain_seg_types, fill_holes_in_mask=fill_holes
        )

        # we need the single outer surface from the scalp. The inner border between
        # scalp and skull is not interesting here. Hence, all segmentation types are
        # grouped together, yielding a uniformly filled head volume.
        all_seg_types = segmentation_masks.segmentation_type.values
        scalp_ijk = surface_from_segmentation(
            segmentation_masks, all_seg_types, fill_holes_in_mask=fill_holes
        )

        # smooth surfaces
        if smoothing > 0:
            brain_ijk = brain_ijk.smooth(smoothing)
            scalp_ijk = scalp_ijk.smooth(smoothing)

        # reduce surface face counts
        # use VTK's decimate_pro algorith as MNE's (VTK's) quadric decimation produced
        # meshes on which Pycortex geodesic distance function failed.
        if brain_face_count is not None:
            # brain_ijk = brain_ijk.decimate(brain_face_count)
            vtk_brain_ijk = cdc.VTKSurface.from_trimeshsurface(brain_ijk)
            reduction = 1.0 - brain_face_count / brain_ijk.nfaces
            vtk_brain_ijk = vtk_brain_ijk.decimate(reduction)
            brain_ijk = cdc.TrimeshSurface.from_vtksurface(vtk_brain_ijk)

        if scalp_face_count is not None:
            # scalp_ijk = scalp_ijk.decimate(scalp_face_count)
            vtk_scalp_ijk = cdc.VTKSurface.from_trimeshsurface(scalp_ijk)
            reduction = 1.0 - scalp_face_count / scalp_ijk.nfaces
            vtk_scalp_ijk = vtk_scalp_ijk.decimate(reduction)
            scalp_ijk = cdc.TrimeshSurface.from_vtksurface(vtk_scalp_ijk)

        brain_ijk = brain_ijk.fix_vertex_normals()
        scalp_ijk = scalp_ijk.fix_vertex_normals()

        brain_mask = segmentation_masks.sel(segmentation_type=brain_seg_types).any(
            "segmentation_type"
        )
        scalp_mask = segmentation_masks.sel(segmentation_type=scalp_seg_types).any(
            "segmentation_type"
        )

        voxel_to_vertex_brain = map_segmentation_mask_to_surface(
            brain_mask, t_ijk2ras, brain_ijk.apply_transform(t_ijk2ras)
        )
        voxel_to_vertex_scalp = map_segmentation_mask_to_surface(
            scalp_mask, t_ijk2ras, scalp_ijk.apply_transform(t_ijk2ras)
        )

        return cls(
            segmentation_masks=segmentation_masks,
            brain=brain_ijk,
            scalp=scalp_ijk,
            landmarks=landmarks_ijk,
            t_ijk2ras=t_ijk2ras,
            t_ras2ijk=t_ras2ijk,
            voxel_to_vertex_brain=voxel_to_vertex_brain,
            voxel_to_vertex_scalp=voxel_to_vertex_scalp,
        )

    @classmethod
    def from_surfaces(
        cls,
        segmentation_dir: str,
        mask_files: dict[str, str] = {
            "csf": "csf.nii",
            "gm": "gm.nii",
            "scalp": "scalp.nii",
            "skull": "skull.nii",
            "wm": "wm.nii",
        },
        brain_surface_file: str = None,
        scalp_surface_file: str = None,
        landmarks_ras_file: str | None = None,
        brain_seg_types: list[str] = ["gm", "wm"],
        scalp_seg_types: list[str] = ["scalp"],
        smoothing: float = 0.0,
        brain_face_count: int | None = None,
        scalp_face_count: int | None = None,
        fill_holes: bool = False,
        coordinates_file: Path | str | None = None,
        voxel_to_vertex_mapping_file_brain : Path | None = None,

    ) -> "TwoSurfaceHeadModel":
        """Constructor from seg.masks, brain and head surfaces as gained from MRI scans.

        Args:
            segmentation_dir: Folder containing the segmentation masks in NIFTI
                format.
            mask_files: Dictionary mapping segmentation types to NIFTI
                filenames.
            brain_surface_file: Path to the brain surface.
            scalp_surface_file: Path to the scalp surface.
            landmarks_ras_file: Filename of the landmarks in RAS space.
            brain_seg_types: List of segmentation types to be included in
                the brain surface.
            scalp_seg_types: List of segmentation types to be included in
                the scalp surface.
            smoothing: Smoothing factor for the brain and scalp surfaces.
            brain_face_count: Number of faces for the brain surface.
            scalp_face_count: Number of faces for the scalp surface.
            fill_holes: Whether to fill holes in the segmentation masks.
            coordinates_file: path to file with additional vertex coordinates.
            voxel_to_vertex_mapping_file_brain : precomputed voxel to vertex mapping

        Returns:
            TwoSurfaceHeadModel: An instance of the TwoSurfaceHeadModel class.
        """

        # load segmentation mask
        segmentation_masks, t_ijk2ras = read_segmentation_masks(
            segmentation_dir, mask_files
        )

        # inspect and invert ijk-to-ras transformation
        t_ras2ijk = xrutils.pinv(t_ijk2ras)

        # crs_ijk = t_ijk2ras.dims[1]
        crs_ras = t_ijk2ras.dims[0]


        # load landmarks. Other than the segmentation masks which are in voxel (ijk)
        # space, these are already in RAS space.
        if landmarks_ras_file is not None:
            if not os.path.isabs(landmarks_ras_file):
                landmarks_ras_file = os.path.join(segmentation_dir, landmarks_ras_file)

            landmarks_ras = read_mrk_json(landmarks_ras_file, crs=crs_ras)
            landmarks_ijk = landmarks_ras.points.apply_transform(t_ras2ijk)
        else:
            landmarks_ijk = None

        # derive surfaces from segmentation masks
        if brain_surface_file is not None:
            brain_ijk = trimesh.load(brain_surface_file)
            brain_ijk = cdc.TrimeshSurface(brain_ijk, 'ijk', cedalion.units.Unit("1"))
        else:
            brain_ijk = surface_from_segmentation(
                segmentation_masks, brain_seg_types, fill_holes_in_mask=fill_holes
            )
        # we need the single outer surface from the scalp. The inner border between
        # scalp and skull is not interesting here. Hence, all segmentation types are
        # grouped together, yielding a uniformly filled head volume.

        if scalp_surface_file is not None:
            scalp_ijk = trimesh.load(scalp_surface_file)
            scalp_ijk = cdc.TrimeshSurface(scalp_ijk, 'ijk', cedalion.units.Unit("1"))
        else:
            all_seg_types = segmentation_masks.segmentation_type.values
            scalp_ijk = surface_from_segmentation(
                segmentation_masks, all_seg_types, fill_holes_in_mask=fill_holes
            )

        # smooth surfaces
        if smoothing > 0:
            brain_ijk = brain_ijk.smooth(smoothing)
            scalp_ijk = scalp_ijk.smooth(smoothing)

        # reduce surface face counts
        # use VTK's decimate_pro algorith as MNE's (VTK's) quadric decimation produced
        # meshes on which Pycortex geodesic distance function failed.
        if brain_face_count is not None:
            # brain_ijk = brain_ijk.decimate(brain_face_count)
            vtk_brain_ijk = cdc.VTKSurface.from_trimeshsurface(brain_ijk)
            reduction = 1.0 - brain_face_count / brain_ijk.nfaces
            vtk_brain_ijk = vtk_brain_ijk.decimate(reduction)
            brain_ijk = cdc.TrimeshSurface.from_vtksurface(vtk_brain_ijk)

        if scalp_face_count is not None:
            # scalp_ijk = scalp_ijk.decimate(scalp_face_count)
            vtk_scalp_ijk = cdc.VTKSurface.from_trimeshsurface(scalp_ijk)
            reduction = 1.0 - scalp_face_count / scalp_ijk.nfaces
            vtk_scalp_ijk = vtk_scalp_ijk.decimate(reduction)
            scalp_ijk = cdc.TrimeshSurface.from_vtksurface(vtk_scalp_ijk)

        brain_ijk = brain_ijk.fix_vertex_normals()
        scalp_ijk = scalp_ijk.fix_vertex_normals()

        brain_mask = segmentation_masks.sel(segmentation_type=brain_seg_types).any(
            "segmentation_type"
        )
        scalp_mask = segmentation_masks.sel(segmentation_type=scalp_seg_types).any(
            "segmentation_type"
        )

        if voxel_to_vertex_mapping_file_brain is not None:
            with gzip.GzipFile(voxel_to_vertex_mapping_file_brain) as fin:
                voxel_to_vertex_brain = scipy.io.mmread(fin)
        else:
            voxel_to_vertex_brain = map_segmentation_mask_to_surface(
                brain_mask, t_ijk2ras, brain_ijk.apply_transform(t_ijk2ras)
            )

        voxel_to_vertex_scalp = map_segmentation_mask_to_surface(
            scalp_mask, t_ijk2ras, scalp_ijk.apply_transform(t_ijk2ras)
        )

        if coordinates_file is not None:
            df = pd.read_csv(coordinates_file)
            assert len(df) == brain_ijk.nvertices
            for col in [i for i in df.columns if i != "vertex"]:
                brain_ijk.vertex_coords[col] = np.asarray(df[col])


        return cls(
            segmentation_masks=segmentation_masks,
            brain=brain_ijk,
            scalp=scalp_ijk,
            landmarks=landmarks_ijk,
            t_ijk2ras=t_ijk2ras,
            t_ras2ijk=t_ras2ijk,
            voxel_to_vertex_brain=voxel_to_vertex_brain,
            voxel_to_vertex_scalp=voxel_to_vertex_scalp,
        )


    def __repr__(self):
        tissue_types = ", ".join(self.segmentation_masks.segmentation_type.values)
        return (
            f"TwoSurfaceHeadModel(\n"
            f"  crs: {self.crs}\n"
            f"  tissue_types: {tissue_types}\n"
            f"  brain faces: {self.brain.nfaces} vertices: {self.brain.nvertices} "
            f"units: {self.brain.units}\n"
            f"  scalp faces: {self.scalp.nfaces} vertices: {self.scalp.nvertices} "
            f"units: {self.scalp.units}\n"
            "  landmarks: "
            f"{len(self.landmarks) if self.landmarks is not None else 'None'}\n"
            ")"
        )


    @property
    def crs(self):
        """Coordinate reference system of the head model."""
        assert self.brain.crs == self.scalp.crs
        if self.landmarks is not None:
            assert self.scalp.crs == self.landmarks.points.crs
        return self.brain.crs

    def apply_transform(self, transform: cdt.AffineTransform) -> "TwoSurfaceHeadModel":
        """Apply a coordinate transformation to the head model.

        Args:
            transform : Affine transformation matrix (4x4) to be applied.

        Returns:
            Transformed head model.
        """

        brain = self.brain.apply_transform(transform)
        scalp = self.scalp.apply_transform(transform)
        landmarks = self.landmarks.points.apply_transform(transform) \
                    if self.landmarks is not None else None

        return TwoSurfaceHeadModel(
            segmentation_masks=self.segmentation_masks,
            brain=brain,
            scalp=scalp,
            landmarks=landmarks,
            t_ijk2ras=self.t_ijk2ras,
            t_ras2ijk=self.t_ras2ijk,
            voxel_to_vertex_brain=self.voxel_to_vertex_brain,
            voxel_to_vertex_scalp=self.voxel_to_vertex_scalp,
        )

    def copy(self, copy_segmasks=False):
        """Create a copy of this head model.

        Args:
            copy_segmasks: if False, the new instance gets a reference to the existing
                segmentation masks instead of a full copy.
        """

        if copy_segmasks:
            segmentation_masks = self.segmentation_masks.copy()
        else:
            segmentation_masks = self.segmentation_masks

        return TwoSurfaceHeadModel(
            segmentation_masks=segmentation_masks,
            brain=self.brain.copy(),
            scalp=self.scalp.copy(),
            landmarks=self.landmarks.copy(),
            t_ijk2ras=self.t_ijk2ras.copy(),
            t_ras2ijk=self.t_ras2ijk.copy(),
            voxel_to_vertex_brain=self.voxel_to_vertex_brain.copy(),
            voxel_to_vertex_scalp=self.voxel_to_vertex_scalp.copy(),
        )

    def save(self, foldername: str):
        """Save the head model to a folder.

        Args:
            foldername (str): Folder to save the head model into.

        Returns:
            None
        """

        # Add foldername if not existing
        if ((not os.path.exists(foldername)) or \
            (not os.path.isdir(foldername))):
            os.mkdir(foldername)

        # Save all head model attributes to folder
        self.segmentation_masks.to_netcdf(os.path.join(foldername,
                                                       "segmentation_masks.nc"))
        self.brain.mesh.export(os.path.join(foldername, "brain.ply"),
                                            file_type="ply")
        self.scalp.mesh.export(os.path.join(foldername, "scalp.ply"),
                                            file_type="ply")
        if self.landmarks is not None:
            self.landmarks.drop_vars("type").to_netcdf(
                os.path.join(foldername, "landmarks.nc")
            )
        self.t_ijk2ras.to_netcdf(os.path.join(foldername, "t_ijk2ras.nc"))
        scipy.sparse.save_npz(os.path.join(foldername, "voxel_to_vertex_brain.npz"),
                                           self.voxel_to_vertex_brain)
        scipy.sparse.save_npz(os.path.join(foldername, "voxel_to_vertex_scalp.npz"),
                                           self.voxel_to_vertex_scalp)
        return

    @classmethod
    def load(cls, foldername: str):
        """Load the head model from a folder.

        Args:
            foldername (str): Folder to load the head model from.

        Returns:
            TwoSurfaceHeadModel: Loaded head model.
        """

        # Check if all files exist
        for fn in ["segmentation_masks.nc", "brain.ply", "scalp.ply",
                   "t_ijk2ras.nc", "voxel_to_vertex_brain.npz",
                   "voxel_to_vertex_scalp.npz"]:
            if not os.path.exists(os.path.join(foldername, fn)):
                raise ValueError("%s does not exist." % os.path.join(foldername, fn))

        # Load all attributes from folder
        segmentation_masks = xr.load_dataarray(
            os.path.join(foldername, "segmentation_masks.nc")
        )
        brain =  trimesh.load(os.path.join(foldername, 'brain.ply'), process=False)
        scalp =  trimesh.load(os.path.join(foldername, 'scalp.ply'), process=False)
        if os.path.exists(os.path.join(foldername, 'landmarks.nc')):
            landmarks_ijk = xr.load_dataset(os.path.join(foldername, 'landmarks.nc'))
            landmarks_ijk = xr.DataArray(
                landmarks_ijk.to_array()[0],
                coords={
                    "label": ("label", landmarks_ijk.label.values),
                    "type": (
                        "label",
                        [cdc.PointType.LANDMARK] * len(landmarks_ijk.label),
                    ),
                },
            )
        else:
            landmarks_ijk = None
        t_ijk2ras = xr.load_dataarray(os.path.join(foldername, 't_ijk2ras.nc'))
        voxel_to_vertex_brain = scipy.sparse.load_npz(os.path.join(foldername,
                                                     'voxel_to_vertex_brain.npz'))
        voxel_to_vertex_scalp = scipy.sparse.load_npz(os.path.join(foldername,
                                                      'voxel_to_vertex_scalp.npz'))

        # Construct TwoSurfaceHeadModel
        brain_ijk = cdc.TrimeshSurface(brain, 'ijk', cedalion.units.Unit("1"))
        scalp_ijk = cdc.TrimeshSurface(scalp, 'ijk', cedalion.units.Unit("1"))
        t_ijk2ras = cdc.affine_transform_from_numpy(
            np.array(t_ijk2ras), "ijk", "unknown", "1", "mm"
        )
        t_ras2ijk = xrutils.pinv(t_ijk2ras)

        return cls(
            segmentation_masks=segmentation_masks,
            brain=brain_ijk,
            scalp=scalp_ijk,
            landmarks=landmarks_ijk,
            t_ijk2ras=t_ijk2ras,
            t_ras2ijk=t_ras2ijk,
            voxel_to_vertex_brain=voxel_to_vertex_brain,
            voxel_to_vertex_scalp=voxel_to_vertex_scalp,
        )


    # FIXME maybe this should not be in this class, especially since the
    # algorithm is not good.
    @cdc.validate_schemas
    def align_and_snap_to_scalp(
        self,
        points: cdt.LabeledPoints,
        mode: str = "general",
    ) -> cdt.LabeledPoints:
        """Align and snap optodes or points to the scalp surface.

        Args:
            points (cdt.LabeledPoints): Points to be aligned and snapped to the
                scalp surface.
            mode: method to derive the affine transform. Could be either
                'trans_rot_isoscale' or 'general'. See cedalion.geometry.registraion
                for details.

        Returns:
            cdt.LabeledPoints: Points aligned and snapped to the scalp surface.
        """

        assert self.landmarks is not None, (
            "Please add landmarks in RAS to head instance."
        )
        if mode == "trans_rot_isoscale":
            t = register_trans_rot_isoscale(self.landmarks, points)
        elif mode == "general":
            t = register_general_affine(self.landmarks, points)
        elif mode == "identity":
            t = register_identity(self.landmarks, points)
        else:
            raise ValueError(f"unexpected mode '{mode}'")

        transformed = points.points.apply_transform(t)
        snapped = self.scalp.snap(transformed)
        return snapped

    def align_and_relax_to_scalp(
        self,
        points: cdt.LabeledPoints,
        channels: list[tuple[str, str]] | cdt.NDTimeSeries | pd.DataFrame,
        nominal_distances: dict[tuple[str, str], float] | None = None,
        n_iter: int = 400,
        k_spring: float = 1.0,
        k_anchor: float = 10.0,
        step_size: float = 0.1,
        convergence_tol: float = 0.01,
        initial_align_mode: str = "general",
    ):
        return register_optodes_spring_icp(
            self.scalp,
            points,
            channels,
            self.landmarks,
            nominal_distances=nominal_distances,
            n_iter=n_iter,
            k_spring=k_spring,
            k_anchor=k_anchor,
            step_size=step_size,
            convergence_tol=convergence_tol,
            initial_align_mode=initial_align_mode
        )

    # FIXME then maybe this should also not be in this class
    @cdc.validate_schemas
    def snap_to_scalp_voxels(
        self, points: cdt.LabeledPoints
    ) -> cdt.LabeledPoints:
        """Snap optodes or points to the closest scalp voxel.

        Args:
            points (cdt.LabeledPoints): Points to be snapped to the closest scalp
                voxel.

        Returns:
            cdt.LabeledPoints: Points aligned and snapped to the closest scalp
                voxel.
        """
        # Align to scalp surface
        aligned = self.scalp.snap(points)

        # Snap to closest scalp voxel
        snapped = np.zeros(points.shape)
        for i, a in enumerate(aligned):

            # Get index of scalp surface vertex "a"
            idx = np.argwhere(self.scalp.mesh.vertices == \
                              np.array(a.pint.dequantify()))

            # Reduce to indices with repitition of 3 (so all coordinates match)
            if len(idx) > 3:
                r = [rep[n] for rep in [{}] for i,n in enumerate(idx[:,0]) \
                           if rep.setdefault(n,[]).append(i) or len(rep[n])==3]
                idx = idx[r[0]]

            # Make sure only one vertex is found
            assert len(idx) == 3
            assert idx[0,0] == idx[1,0] == idx[2,0]

            # Get voxel indices mapping to this scalp vertex
            vec = np.zeros(self.scalp.nvertices)
            vec[idx[0,0]] = 1
            voxel_idx = np.argwhere(self.voxel_to_vertex_scalp @ vec == 1)[:,0]

            if len(voxel_idx) > 0:
                # Get voxel coordinates from voxel indices
                try:
                    shape = self.segmentation_masks.shape[-3:]
                except AttributeError: # FIXME should not be handled here
                    shape = self.segmentation_masks.to_dataarray().shape[-3:]
                voxels = np.array(np.unravel_index(voxel_idx, shape)).T

                # Choose the closest voxel
                dist = np.linalg.norm(voxels - np.array(a.pint.dequantify()), axis=1)
                voxel_idx = np.argmin(dist)

            else:
                # If no voxel maps to that scalp surface vertex,
                # simply choose the closest of all scalp voxels

                sm = self.segmentation_masks

                voxels = voxels_from_segmentation(sm, ["scalp"]).voxels
                if len(voxels) == 0:
                    try:
                        scalp_mask = sm.sel(segmentation_type="scalp").to_dataarray()
                    except AttributeError: # FIXME same as above
                        scalp_mask = sm.sel(segmentation_type="scalp")
                    voxels = np.argwhere(np.array(scalp_mask)[0] > 0.99)

                kdtree = KDTree(voxels)
                dist, voxel_idx = kdtree.query(self.scalp.mesh.vertices[idx[0,0]],
                                               workers=-1)

            # Snap to closest scalp voxel
            snapped[i] = voxels[voxel_idx]

        return points.copy(data=snapped)


    def scale_to_landmarks(
        self,
        target_landmarks : cdt.LabeledPoints,
        mode = "general",
    ) -> "TwoSurfaceHeadModel":
        """Scale the head model to match a set of anatomical landmarks.

        Computes a general affine transform that maps the head model's RAS
        landmarks to ``target_landmarks`` and applies it to both surfaces
        and the stored affine transforms.

        Args:
            target_landmarks: Target landmark positions (e.g. from a digitizer)
                in any CRS.  Must contain the same label subset as the model's
                landmarks.
            mode: method to derive the affine transform. Could be either
                'trans_rot_isoscale' or 'general'. See cedalion.geometry.registraion
                for details.

        Returns:
            New :class:`TwoSurfaceHeadModel` scaled and aligned to
            ``target_landmarks``.
        """
        if self.crs == "ijk":
            landmarks_ras = self.landmarks.points.apply_transform(self.t_ijk2ras)
        else:
            landmarks_ras = self.landmarks

        if mode == "trans_rot_isoscale":
            t_ras2scaled = register_trans_rot_isoscale(target_landmarks, landmarks_ras)
        elif mode == "general":
            t_ras2scaled = register_general_affine(target_landmarks, landmarks_ras)
        else:
            raise ValueError(f"unexpected mode '{mode}'")


        t_ijk2scaled = t_ras2scaled @ self.t_ijk2ras
        t_scaled2ijk = xrutils.pinv(t_ijk2scaled)

        if self.crs == "ijk":
            result = self.apply_transform(t_ijk2scaled)
        else:
            result = self.apply_transform(t_ras2scaled)

        result.t_ijk2ras = t_ijk2scaled
        result.t_ras2ijk = t_scaled2ijk

        return result


    def scale_to_headsize(
        self, circumference: cdt.QLength, nz_cz_iz: cdt.QLength, lpa_cz_rpa: cdt.QLength
    ) -> "TwoSurfaceHeadModel":
        """Scale the head model to match anthropometric head-size measurements.

        Fits a tri-axial ellipsoid model to the three measurements, derives
        corresponding landmark positions, and delegates to :meth:`scale_to_landmarks`.

        Args:
            circumference: Head circumference.
            nz_cz_iz: Nasion–Cz–Inion arc length.
            lpa_cz_rpa: Left preauricular–Cz–right preauricular arc length.

        Returns:
            New :class:`TwoSurfaceHeadModel` scaled to the specified head size.
        """
        ellipsoid_landmarks = get_landmarks_for_headsize(
            circumference, nz_cz_iz, lpa_cz_rpa
        )

        return self.scale_to_landmarks(ellipsoid_landmarks)


    def get_brain_mni152_coords(self) -> xr.DataArray:
        """Return MNI152 vertex coordinates of the brain surface.

        Returns:
            xr.DataArray with dimensions ``(label, mni152)`` containing the
            ``mni152_r``, ``mni152_a``, and ``mni152_s`` vertex coordinates.

        Raises:
            ValueError: If the brain surface does not have ``mni152_r/a/s``
                vertex coordinates (i.e. not a standard atlas head model).
        """
        v = self.brain.vertices
        if not all([f"mni152_{i}" in v.coords for i in ["r", "a", "s"]]):
            raise NotImplementedError(
                "The cortex surface has no mni vertex coordinates"
            )

        mni_coords = (
            xr.concat([v.mni152_r, v.mni152_a, v.mni152_s], dim="mni152")
            .transpose("label", ...)
            .rename("mni152")
        )

        keep_coords = ["label", "fsaverage_vertex"]

        for c in [i for i in mni_coords.coords.keys() if i not in keep_coords]:
            mni_coords = mni_coords.drop_vars(c)

        return mni_coords

    def assign_parcels_via_mni_coords(
        self,
        coordinate_label: str,
        label_mapping: dict[int, str],
        voxel_label_niftii: Path,
        voxel_label_crs: str = "mni152",
        background_label = "Background",
        mni_eps=5.
    ) -> "TwoSurfaceHeadModel":
        """Assign parcel labels to brain vertices using a volumetric atlas in MNI space.

        For each brain vertex, the nearest non-background voxel within ``mni_eps``
        in MNI152 space is found and its string label is assigned to the veretx.
        Vertices with no labeled neighbor within the search radius receive
        ``background_label``.

        Args:
            coordinate_label: Name of the new vertex coordinate.
            label_mapping: A dictionary that maps numeric voxel labels in the NIfTI file
                to string parcel names.
            voxel_label_niftii: Path to the NIfTI atlas file whose voxels carry
                numeric region labels.
            voxel_label_crs: Coordinate reference system of the NIfTI file,
                either ``"mni152"`` or ``"mni305"``. Defaults to ``"mni152"``.
            background_label: String label used for unlabeled vertices and, if
                present in ``label_mapping``, to identify background voxels.
                Defaults to ``"Background"``.
            mni_eps: Radius of the ball search around each vertex.
                Defaults to ``5.0``. Increase this value to find non-background
                voxels for each vertex.

        Returns:
            A copy of the head model with parcel labels assigned as a new
            coordinate to brain vertices.
        """

        if voxel_label_crs not in ["mni152", "mni305"]:
            raise ValueError(
                "The niftii file should be either in the 'mni152' or 'mni305' space."
            )

        vertex_mni_coords = self.get_brain_mni152_coords()

        # load the niftii file which contains for each voxel a numeric label
        img = nibabel.load(voxel_label_niftii)
        flat_voxel_numlabels = img.get_fdata().flatten().astype(int)

        # figure out numeric label of background:
        if background_label in label_mapping.values():
            # background is specified in label_mapping
            bg_numlabel = [k for k, v in label_mapping.items() if v == background_label]
        else:
            # background is not specified in label mapping. assume there is one numeric
            # label used in the niftii, that is not in the mapping and that denotes
            # background (often 0).
            used_numlabels = np.unique(flat_voxel_numlabels)
            bg_numlabel = [i for i in used_numlabels if i not in label_mapping.keys()]

        assert len(bg_numlabel) == 1
        bg_numlabel = bg_numlabel[0]

        # obtain cell coordinates in the RAS space of the niftii file
        ijk2crs = cedalion.io.anatomy._get_affine_from_niftii(img, crs=voxel_label_crs)

        if voxel_label_crs == "mni152":
            # the niftii provides already mni152 coordinates. just use its transform
            ijk2mni152 = ijk2crs
        elif voxel_label_crs == "mni305":
            # if the niftii is in mni305 coordiantes apply additionally the transform
            # that bring us from mni305 to mni152
            ijk2mni152 = cedalion.dot.utils.mni305_to_mni152 @ ijk2crs

        ijk2mni152 = ijk2mni152.pint.dequantify()  # FIXME?

        voxel_coords = cedalion.io.anatomy.cell_coordinates(
            img, ijk2mni152, center=True
        )
        voxel_coords = voxel_coords.pint.dequantify()

        flat_voxel_coords = voxel_coords.values.reshape(-1, 3)

        vertex_strlabels = []

        # for all voxels within a ball of radius mni_eps around a vertice's MNI coord...
        t = KDTree(flat_voxel_coords)
        neighbor_list = t.query_ball_point(vertex_mni_coords.values, mni_eps)

        for i, neighbors_flat_idx in enumerate(neighbor_list):
            # ... retrieve their numeric label and distance to the MNI coord
            vortex_numlabels = flat_voxel_numlabels[neighbors_flat_idx]
            vortex_coords = flat_voxel_coords[neighbors_flat_idx]
            dists = np.linalg.norm(vortex_coords - vertex_mni_coords.values[i], axis=1)

            # ... create a mask of voxels that are not background
            notbg_mask = vortex_numlabels != bg_numlabel

            # .. if there are labeled voxels nearby assign the closest one
            if np.any(notbg_mask):
                masked_dists = np.where(notbg_mask, dists, np.inf)
                this_vortex_numlabel = vortex_numlabels[np.argmin(masked_dists)]
                vertex_strlabels.append(label_mapping[this_vortex_numlabel])
            # otherwise assign background
            else:
                vertex_strlabels.append(background_label)

        hm = self.copy()
        hm.brain.vertex_coords[coordinate_label] = vertex_strlabels

        return hm


@lru_cache
def get_standard_headmodel(model : str) -> TwoSurfaceHeadModel:
    """Create a TwoSurfaceHeadmodel for common atlases.

    Once created, this function caches a head model.

    Args:
        model: either colin27 or icbm152
    Returns:
        The loaded head model with 1010-landmarks and parcel labels assigned.
    """

    AVAILABLE_MODELS = ["colin27", "icbm152"]

    if model == "colin27":
        f = cedalion.data.get_colin27_headmodel_files()

        head_ijk = TwoSurfaceHeadModel.from_surfaces(
            segmentation_dir=f.basedir,
            mask_files=f.mask_files,
            brain_surface_file=f.basedir / f.brain_surface_obj,
            scalp_surface_file=f.basedir / f.scalp_surface_obj,
            landmarks_ras_file=f.basedir / f.landmarks_ras_file,
            coordinates_file=f.basedir / f.brain_vertex_coordinates,
            voxel_to_vertex_mapping_file_brain= f.basedir / f.voxel_to_vertex_mapping,
            brain_face_count=None,
            scalp_face_count=None,
            smoothing=0,
        )

        head_ijk.t_ijk2ras = head_ijk.t_ijk2ras.rename({"aligned" : "mni"})
        head_ijk.t_ras2ijk = head_ijk.t_ras2ijk.rename({"aligned" : "mni"})


    elif model == "icbm152":
        f = cedalion.data.get_icbm152_headmodel_files()

        head_ijk = TwoSurfaceHeadModel.from_surfaces(
            segmentation_dir=f.basedir,
            mask_files=f.mask_files,
            brain_surface_file=f.basedir / f.brain_surface_obj,
            scalp_surface_file=f.basedir / f.scalp_surface_obj,
            landmarks_ras_file=f.basedir / f.landmarks_ras_file,
            coordinates_file=f.basedir / f.brain_vertex_coordinates,
            voxel_to_vertex_mapping_file_brain= f.basedir / f.voxel_to_vertex_mapping,
            brain_face_count=None,
            scalp_face_count=None,
            smoothing=0,
        )

        head_ijk.t_ijk2ras = head_ijk.t_ijk2ras.rename({"aligned" : "mni"})
        head_ijk.t_ras2ijk = head_ijk.t_ras2ijk.rename({"aligned" : "mni"})

    else:
        raise ValueError(
            "Unknown head model. Available models are: " + ", ".join(AVAILABLE_MODELS)
        )

    return head_ijk


@lru_cache
def get_inflated_cortex_surface(model : str) -> cdc.TrimeshSurface:
    """Load the inflated cortex surface for a standard atlas head model.

    The result is cached after the first call.

    Args:
        model: Atlas name — one of ``"colin27"`` or ``"icbm152"``.

    Returns:
        :class:`~cedalion.dataclasses.TrimeshSurface` in the ``"inflated"`` CRS.

    Raises:
        ValueError: If ``model`` is not one of the supported atlases.
    """
    AVAILABLE_MODELS = ["colin27", "icbm152"]

    if model == "colin27":
        f = cedalion.data.get_colin27_headmodel_files()
    elif model == "icbm152":
        f = cedalion.data.get_icbm152_headmodel_files()
    else:
        raise ValueError(
            "Unknown head model. Available models are: " + ", ".join(AVAILABLE_MODELS)
        )

    return cdc.TrimeshSurface(
        trimesh.load(f.basedir / "cortex_pial_inflated.obj"),
        crs="inflated",
        units=cedalion.units(None).units,
    )
