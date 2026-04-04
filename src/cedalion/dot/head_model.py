
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

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
        brain_face_count: int | None = 180000,
        scalp_face_count: int | None = 60000,
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
            voxel_to_vertex_brain = scipy.sparse.load_npz(
                voxel_to_vertex_mapping_file_brain
            )
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
            f"  landmarks: {len(self.landmarks)}\n"
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
        self.t_ras2ijk.to_netcdf(os.path.join(foldername, "t_ras2ijk.nc"))
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
                   "t_ijk2ras.nc", "t_ras2ijk.nc", "voxel_to_vertex_brain.npz",
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
        t_ras2ijk = xr.load_dataarray(os.path.join(foldername, 't_ras2ijk.nc'))
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
        else:
            raise ValueError(f"unexpected mode '{mode}'")

        transformed = points.points.apply_transform(t)
        snapped = self.scalp.snap(transformed)
        return snapped


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

        points.values = snapped
        return points


    def scale_to_landmarks(
        self,
        target_landmarks : cdt.LabeledPoints
    ) -> "TwoSurfaceHeadModel":
        if self.crs == "ijk":
            landmarks_ras = self.landmarks.points.apply_transform(self.t_ijk2ras)
        else:
            landmarks_ras = self.landmarks

        t_ras2scaled = register_general_affine(target_landmarks, landmarks_ras)

        t_ijk2scaled = t_ras2scaled @ self.t_ijk2ras
        t_scaled2ijk = xrutils.pinv(t_ijk2scaled)

        if self.crs == "ijk":
            result = self.apply_transform(t_ijk2scaled)
        else:
            result = self.apply_transform(t_ijk2scaled)


        result.t_ijk2ras = t_ijk2scaled
        result.t_ras2ijk = t_scaled2ijk

        return result


    def scale_to_headsize(
        self, circumference: cdt.QLength, nz_cz_iz: cdt.QLength, lpa_cz_rpa: cdt.QLength
    ) -> "TwoSurfaceHeadModel":
        ellipsoid_landmarks = get_landmarks_for_headsize(
            circumference, nz_cz_iz, lpa_cz_rpa
        )

        return self.scale_to_landmarks(ellipsoid_landmarks)




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

