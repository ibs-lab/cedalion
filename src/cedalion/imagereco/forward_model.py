"""Forward model for simulating light transport in the head.
NOTE: Cedalion currently supports two ways to compute fluence:
1) via monte-carlo simulation using the MonteCarloXtreme (MCX) package, and
2) via the finite element method (FEM) using the NIRFASTer package.
While MCX is automatically installed using pip, NIRFASTER has to be manually installed
runnning <$ bash install_nirfaster.sh CPU # or GPU> from a within your cedalion root directory. """

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Optional
import os.path
import warnings
import sys

import numpy as np
import pandas as pd
import pint
import scipy.sparse
from scipy.spatial import KDTree
import trimesh
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
from cedalion.geometry.registration import register_trans_rot_isoscale
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion.geometry.segmentation import (
    surface_from_segmentation,
    voxels_from_segmentation,
)
from cedalion.imagereco.utils import map_segmentation_mask_to_surface

from .tissue_properties import get_tissue_properties

logger = logging.getLogger("cedalion")


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
        landmarks : cdt.LabeledPointCloud
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
    landmarks: cdt.LabeledPointCloud
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
        landmarks_ras_file: Optional[str] = None,
        brain_seg_types: list[str] = ["gm", "wm"],
        scalp_seg_types: list[str] = ["scalp"],
        smoothing: float = 0.5,
        brain_face_count: Optional[int] = 180000,
        scalp_face_count: Optional[int] = 60000,
        fill_holes: bool = True,
    ) -> "TwoSurfaceHeadModel":
        """Constructor from binary masks as gained from segmented MRI scans.

        Args:
            segmentation_dir (str): Folder containing the segmentation masks in NIFTI
                format.
            mask_files (Dict[str, str]): Dictionary mapping segmentation types to NIFTI
                filenames.
            landmarks_ras_file (Optional[str]): Filename of the landmarks in RAS space.
            brain_seg_types (list[str]): List of segmentation types to be included in
                the brain surface.
            scalp_seg_types (list[str]): List of segmentation types to be included in
                the scalp surface.
            smoothing(float): Smoothing factor for the brain and scalp surfaces.
            brain_face_count (Optional[int]): Number of faces for the brain surface.
            scalp_face_count (Optional[int]): Number of faces for the scalp surface.
            fill_holes (bool): Whether to fill holes in the segmentation masks.
        """

        # load segmentation mask
        segmentation_masks, t_ijk2ras = cedalion.io.read_segmentation_masks(
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

            landmarks_ras = cedalion.io.read_mrk_json(landmarks_ras_file, crs=crs_ras)
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
        landmarks_ras_file: Optional[str] = None,
        brain_seg_types: list[str] = ["gm", "wm"],
        scalp_seg_types: list[str] = ["scalp"],
        smoothing: float = 0.5,
        brain_face_count: Optional[int] = 180000,
        scalp_face_count: Optional[int] = 60000,
        fill_holes: bool = False,
    ) -> "TwoSurfaceHeadModel":
        """Constructor from seg.masks, brain and head surfaces as gained from MRI scans.

        Args:
            segmentation_dir (str): Folder containing the segmentation masks in NIFTI
                format.
            mask_files (dict[str, str]): Dictionary mapping segmentation types to NIFTI
                filenames.
            brain_surface_file (str): Path to the brain surface.
            scalp_surface_file (str): Path to the scalp surface.
            landmarks_ras_file (Optional[str]): Filename of the landmarks in RAS space.
            brain_seg_types (list[str]): List of segmentation types to be included in
                the brain surface.
            scalp_seg_types (list[str]): List of segmentation types to be included in
                the scalp surface.
            smoothing (float): Smoothing factor for the brain and scalp surfaces.
            brain_face_count (Optional[int]): Number of faces for the brain surface.
            scalp_face_count (Optional[int]): Number of faces for the scalp surface.
            fill_holes (bool): Whether to fill holes in the segmentation masks.

        Returns:
            TwoSurfaceHeadModel: An instance of the TwoSurfaceHeadModel class.
        """

        # load segmentation mask
        segmentation_masks, t_ijk2ras = cedalion.io.read_segmentation_masks(
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

            landmarks_ras = cedalion.io.read_mrk_json(landmarks_ras_file, crs=crs_ras)
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
        self, points: cdt.LabeledPointCloud
    ) -> cdt.LabeledPointCloud:
        """Align and snap optodes or points to the scalp surface.

        Args:
            points (cdt.LabeledPointCloud): Points to be aligned and snapped to the
                scalp surface.

        Returns:
            cdt.LabeledPointCloud: Points aligned and snapped to the scalp surface.
        """

        assert self.landmarks is not None, "Please add landmarks in RAS to head \
                                            instance."
        t = register_trans_rot_isoscale(self.landmarks, points)
        transformed = points.points.apply_transform(t)
        snapped = self.scalp.snap(transformed)
        return snapped


    # FIXME then maybe this should also not be in this class
    @cdc.validate_schemas
    def snap_to_scalp_voxels(
        self, points: cdt.LabeledPointCloud
    ) -> cdt.LabeledPointCloud:
        """Snap optodes or points to the closest scalp voxel.

        Args:
            points (cdt.LabeledPointCloud): Points to be snapped to the closest scalp
                voxel.

        Returns:
            cdt.LabeledPointCloud: Points aligned and snapped to the closest scalp
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


class ForwardModel:
    """Forward model for simulating light transport in the head.

    ...

    Args:
    head_model (TwoSurfaceHeadModel): Head model containing voxel projections to brain
        and scalp surfaces.
    optode_pos (cdt.LabeledPointCloud): Optode positions.
    optode_dir (xr.DataArray): Optode orientations (directions of light beams).
    tissue_properties (xr.DataArray): Tissue properties for each tissue type.
    volume (xr.DataArray): Voxelated head volume from segmentation masks.
    unitinmm (float): Unit of head model, optodes expressed in mm.
    measurement_list (pd.DataFrame): List of measurements of experiment with source,
        detector, channel, and wavelength.

    Methods:
        compute_fluence(nphoton):
            Compute fluence for each channel and wavelength from photon simulation.
        compute_sensitivity(fluence_all, fluence_at_optodes):
            Compute sensitivity matrix from fluence.
    """

    def __init__(
        self,
        head_model: TwoSurfaceHeadModel,
        geo3d: cdt.LabeledPointCloud,
        measurement_list: pd.DataFrame,
    ):
        """Constructor for the forward model.

        Args:
            head_model (TwoSurfaceHeadModel): Head model containing voxel projections to
                brain and scalp surfaces.
            geo3d (cdt.LabeledPointCloud): Optode positions and directions.
            measurement_list (pd.DataFrame): List of measurements of experiment with
                source, detector, channel and wavelength.
        """

        assert head_model.crs == "ijk"  # FIXME
        assert head_model.crs == geo3d.points.crs

        self.head_model = head_model

        self.optode_pos = geo3d[
            geo3d.type.isin([cdc.PointType.SOURCE, cdc.PointType.DETECTOR])
        ]

        # Comppute the direction of the light beam from the surface normals
        # pmcx fails if directions are not normalized
        self.optode_dir = -head_model.scalp.get_vertex_normals(
            self.optode_pos,
            normalized=True,
        )

        # Slightly realign the optode positions to the closest scalp voxel
        self.optode_pos = head_model.snap_to_scalp_voxels(self.optode_pos)


        self.optode_pos = self.optode_pos.pint.dequantify()
        self.optode_dir = self.optode_dir.pint.dequantify()

        self.tissue_properties = get_tissue_properties(
            self.head_model.segmentation_masks
        )

        self.volume = self.head_model.segmentation_masks.sum("segmentation_type")
        self.volume = self.volume.values.astype(np.uint8)
        self.unitinmm = self._get_unitinmm()

        self.measurement_list = measurement_list

    def _get_unitinmm(self):
        """Calculate length of volume grid cells.

        The forward model operates in ijk-space, in which each cell has unit length. To
        relate to physical distances pmcx needs the 'unitinmm' parameter.
        """

        pts = cdc.build_labeled_points([[0, 0, 0], [0, 0, 1]], crs="ijk", units="1")
        pts_ras = pts.points.apply_transform(self.head_model.t_ijk2ras)
        length = xrutils.norm(pts_ras[1] - pts_ras[0], pts_ras.points.crs)
        return length.pint.magnitude.item()

    def _get_fluence_from_mcx(self, i_optode: int, **kwargs) -> np.ndarray:
        """Run MCX simulation to get fluence for one optode.

        Args:
            i_optode: Index of the optode.
            **kwargs: Additional keywords are passed to MCX's configuration dict.

        Returns:
            np.ndarray: Fluence in each voxel.
        """

        kwargs.setdefault("nphoton", 1e8)

        cfg = {
            "nphoton": kwargs['nphoton'],
            "vol": self.volume,
            "tstart": 0,
            "tend": 5e-9,
            "tstep": 5e-9,
            "srcpos": self.optode_pos.values[i_optode],
            "srcdir": self.optode_dir.values[i_optode],
            "prop": self.tissue_properties,
            "issrcfrom0": 1,
            "isnormalized": 1,
            "outputtype": "fluence", # units: 1/mm^2
            "issavedet": 0,
            "unitinmm": self.unitinmm,
        }

        # merging default cfg with additional positional arguments

        cfg = { **cfg, **kwargs }

        # if pmcx fails, try pmcxcl

        if "cuda" in cfg and cfg["cuda"]:
            import pmcx
            result = pmcx.run(cfg)
        else:
            import pmcxcl
            result = pmcxcl.run(cfg)

        fluence = result["flux"][:, :, :, 0]  # there is only one time bin

        return fluence

    def _fluence_at_optodes(self, fluence, emitting_opt):
        """Fluence caused by one optode at the positions of all other optodes.

        Args:
            fluence (np.ndarray): Fluence in each voxel.
            emitting_opt (int): Index of the emitting optode.

        Returns:
            np.ndarray: Fluence at all optode positions.
        """

        n_optodes = len(self.optode_pos)

        # The fluence in the voxel of the current optode can be zero if
        # the optode position is outside the scalp. In this case move up to
        # a specified distance from the optode position into the optode direction
        # until the fluence becomes positive
        MAX_DISTANCE_IN_MM = 50
        MAX_STEPS = int(np.ceil(MAX_DISTANCE_IN_MM / self.unitinmm))

        result = np.zeros(n_optodes)
        for i_opt in range(n_optodes):
            for i_step in range(MAX_STEPS):
                pos = self.optode_pos[i_opt] + i_step * self.optode_dir[i_opt]
                i, j, k = np.floor(pos.values).astype(int)

                if fluence[i, j, k] > 0:
                    result[i_opt] = fluence[i, j, k]
                    break
            else:
                l_emit = self.optode_pos.label.values[emitting_opt]
                l_rcv = self.optode_pos.label.values[i_opt]
                logger.info(
                    f"fluence from {l_emit} to optode {l_rcv} "
                    f"is zero within {MAX_DISTANCE_IN_MM} mm."
                )

        return result

    def compute_fluence_mcx(self, **kwargs):
        """Compute fluence for each channel and wavelength using MCX package.

        Args:
            kwargs: key-value pairs are passed to MCX's configuration dict. For example
                nphoton (int) to control the number of photons to simulate.
                See https://pypi.org/project/pmcx for further options.

        Returns:
            xr.DataArray: Fluence in each voxel for each channel and wavelength.

        References:
            (:cite:t:`Fang2009`) Qianqian Fang and David A. Boas, "Monte Carlo
            Simulation of Photon Migration in 3D Turbid Media Accelerated by Graphics
            Processing Units," Optics Express, vol.17, issue 22, pp. 20178-20190 (2009).

            (:cite:t:`Yu2018`) Leiming Yu, Fanny Nina-Paravecino, David Kaeli,
            Qianqian Fang, “Scalable and massively parallel Monte Carlo photon transport
            simulations for heterogeneous computing platforms,”
            J. Biomed. Opt. 23(1), 010504 (2018).

            (:cite:t:`Yan2020`) Shijie Yan and Qianqian Fang* (2020),
            "Hybrid mesh and voxel based Monte Carlo algorithm for accurate and
            efficient photon transport modeling in complex bio-tissues,"
            Biomed. Opt. Express, 11(11) pp. 6262-6270.
            https://www.osapublishing.org/boe/abstract.cfm?uri=boe-11-11-6262

        """

        wavelengths = self.measurement_list.wavelength.unique()
        n_wavelength = len(wavelengths)
        n_optodes = len(self.optode_pos)

        fluence_at_optodes = np.zeros((n_optodes, n_optodes, n_wavelength))

        # the fluence per voxel, wavelength and optode position
        # FIXME this may become large. eventually cache on disk?
        fluence_all = np.zeros((n_optodes, n_wavelength) + self.volume.shape)

        for i_opt in range(n_optodes):
            label = self.optode_pos.label.values[i_opt]
            print(f"simulating fluence for {label}. {i_opt+1} / {n_optodes}")

            # run MCX or MCXCL
            # shape: [i,j,k]
            fluence = self._get_fluence_from_mcx(i_opt, **kwargs)

            # FIXME shortcut: currently tissue props are wavelength independent -> copy
            for i_wl in range(n_wavelength):
                # calculate fluence at all optode positions for normalization purposes
                fluence_at_optodes[i_opt, :, i_wl] = self._fluence_at_optodes(
                    fluence, i_opt
                )

                fluence_all[i_opt, i_wl, :, :, :] = fluence

            # accumulate brain and scalp voxels
            # flux = flux.flatten()
            # flux_brain[:, i_opt, i_wl] = flux @ self.head_model.voxel_to_vertex_brain
            # flux_scalp[:, i_opt, i_wl] = flux @ self.head_model.voxel_to_vertex_scalp

        # convert to DataArray
        fluence_all = xr.DataArray(
            fluence_all,
            dims=["label", "wavelength", "i", "j", "k"],
            coords={
                "label": ("label", self.optode_pos.label.values),
                "type": ("label", self.optode_pos.type.values),
                "wavelength": ("wavelength", wavelengths),
            },
            attrs={"units": "1 / millimeter ** 2"},
        )

        fluence_at_optodes = xr.DataArray(
            fluence_at_optodes,
            dims=["optode1", "optode2", "wavelength"],
            coords={
                "optode1": self.optode_pos.label.values,
                "optode2": self.optode_pos.label.values,
                "wavelength": wavelengths,
            },
            attrs={"units": "1 / millimeter ** 2"},
        )

        return fluence_all, fluence_at_optodes

    def compute_fluence_nirfaster(self, meshingparam=None):
        """Compute fluence for each channel and wavelength using NIRFASTer package.

        Args:
            meshingparam (ff.utils.MeshingParam): Parameters to be used by the CGAL
                mesher. Note: they should all be double

        Returns:
        xr.DataArray: Fluence in each voxel for each channel and wavelength.

        References:
            (:cite:t:`Dehghani2009`) Dehghani, Hamid, et al. "Near infrared optical
            tomography using NIRFAST: Algorithm for numerical model and image
            reconstruction."
            Communications in numerical methods in engineering 25.6 (2009): 711-732.
        """

        # FIXME
        src_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../plugins/nirfaster-uFF",
            )
        )
        if src_path not in sys.path:
            sys.path.append(src_path)

        import nirfasteruff as ff

        # Choose between 'CPU' or 'GPU' solver (case insensitive). Automatically
        # determined (GPU prioritized) if not specified
        solver = ff.utils.get_solver()
        # Contains the parameters used by the FEM solvers, Equivalent to
        # 'solver_options' in the Matlab version
        solver_opt = ff.utils.SolverOptions()

        if meshingparam is None:
            # meshing parameters; should be adjusted depending on the user's need
            meshingparam = ff.utils.MeshingParams(
                facet_distance=1.0,
                facet_size=1.0,
                general_cell_size=2.0,
                lloyd_smooth=0,
            )

        # create a nirfaster mesh
        mesh = ff.base.stndmesh()
        # make the optical property matrix; unit in mm-1
        tissueprop = np.zeros((self.tissue_properties.shape[0]-1, 4))
        for i in range(tissueprop.shape[0]):
            tissueprop[i,0] = i+1
            tissueprop[i,1] = self.tissue_properties[i+1, 0]
            tissueprop[i,2] = self.tissue_properties[i+1, 1] * (1-self.tissue_properties[i+1, 2]) # noqa: E501
            tissueprop[i,3] = self.tissue_properties[i+1, 3]

        # all optodes x all optodes
        sources = ff.base.optode(coord=self.optode_pos.data)
        detectors = ff.base.optode(coord=self.optode_pos.data)
        n_optodes = self.optode_pos.data.shape[0]
        link = np.zeros((n_optodes*n_optodes,3), dtype=np.int32)
        ch = 0
        for i in range(n_optodes):
            for j in range(n_optodes):
                link[ch, 0] = i+1
                link[ch, 1] = j+1
                link[ch, 2] = 1
                ch += 1

        # construct the mesh
        mesh.from_volume(
            self.volume,
            param=meshingparam,
            prop=tissueprop,
            src=sources,
            det=detectors,
            link=link,
        )
        # calculate the interpolation functions to and from voxel space
        igrid = np.arange(self.volume.shape[0])
        jgrid = np.arange(self.volume.shape[1])
        kgrid = np.arange(self.volume.shape[2])
        mesh.gen_intmat(igrid, jgrid, kgrid)
        # calculate fluence
        data,_ = mesh.femdata(0, solver=solver, opt=solver_opt)
        amplitude_optode = np.reshape(data.amplitude, (n_optodes,-1))

        wavelengths = self.measurement_list.wavelength.unique()
        n_wavelength = len(wavelengths)
        fluence_all = np.zeros((n_optodes, n_wavelength) + self.volume.shape)
        fluence_at_optodes = np.zeros((n_optodes, n_optodes, n_wavelength))

        for i_wl in range(n_wavelength):
            # PLACEHOLDER: set new property and repeat
            # This way we can void the expensive meshing
            # newprop = []
            # mesh.set_prop(newprop)
            # newdata,_=femdata(0)
            for i_opt in range(n_optodes):
                fluence_all[i_opt, i_wl, :, :, :] = np.transpose(
                    data.phi[:, :, :, i_opt], (1, 0, 2)
                )  # xyz to ijk
                fluence_at_optodes[i_opt, :, i_wl] = amplitude_optode[:,i_opt]

        # convert to DataArray; copied from foward_model
        fluence_all = xr.DataArray(
            fluence_all,
            dims=["label", "wavelength", "i", "j", "k"],
            coords={
                "label": ("label", self.optode_pos.label.values),
                "type": ("label", self.optode_pos.type.values),
                "wavelength": ("wavelength", wavelengths),
            },
            attrs={"units": "1 / millimeter ** 2"},
        )

        fluence_at_optodes = xr.DataArray(
            fluence_at_optodes,
            dims=["optode1", "optode2", "wavelength"],
            coords={
                "optode1": self.optode_pos.label.values,
                "optode2": self.optode_pos.label.values,
                "wavelength": wavelengths,
            },
            attrs={"units": "1 / millimeter ** 2"},
        )

        return fluence_all, fluence_at_optodes


    def compute_sensitivity(self, fluence_all, fluence_at_optodes):
        """Compute sensitivity matrix from fluence.

        Args:
            fluence_all (xr.DataArray): Fluence in each voxel for each wavelength.
            fluence_at_optodes (xr.DataArray): Fluence at all optode positions for each
                wavelength.

        Returns:
            xr.DataArray: Sensitivity matrix for each channel, vertex and wavelength.
        """

        unique_channels = self.measurement_list[
            ["channel", "source", "detector"]
        ].drop_duplicates()

        channels = unique_channels["channel"].tolist()
        source = unique_channels["source"].tolist()
        detector = unique_channels["detector"].tolist()

        n_channel = len(channels)
        wavelengths = self.measurement_list.wavelength.unique().tolist()
        n_wavelength = len(wavelengths)

        n_brain = self.head_model.brain.nvertices
        n_scalp = self.head_model.scalp.nvertices
        Adot_brain = np.zeros((n_channel, n_brain, n_wavelength))
        Adot_scalp = np.zeros((n_channel, n_scalp, n_wavelength))

        for _, r in self.measurement_list.iterrows():
            # using the adjoint monte carlo method
            # see YaoIntesFang2018 and BoasDale2005

            pertubation = (
                fluence_all.loc[r.source, r.wavelength]
                * fluence_all.loc[r.detector, r.wavelength]
            )
            pertubation = pertubation.values.flatten()
            normfactor = (
                fluence_at_optodes.loc[r.source, r.detector, r.wavelength].values
                + fluence_at_optodes.loc[r.detector, r.source, r.wavelength].values
            ) / 2

            i_wl = wavelengths.index(r.wavelength)
            i_ch = channels.index(r.channel)

            Adot_brain[i_ch, :, i_wl] = (
                pertubation @ self.head_model.voxel_to_vertex_brain / normfactor
            )
            Adot_scalp[i_ch, :, i_wl] = (
                pertubation @ self.head_model.voxel_to_vertex_scalp / normfactor
            )

        is_brain = np.zeros((n_brain + n_scalp), dtype=bool)
        is_brain[:n_brain] = True

        # shape [nchannel, nvertices, nwavelength]
        Adot = np.concatenate([Adot_brain, Adot_scalp], axis=1)

        # Adot calculated from fluence has units 1/mm^2. Multiplied with
        # the voxel volume (mm^3) and the change in absorption coefficient (1/mm)
        # this yields optical density (1). For the standard head models with 1mm^3 voxel
        # size, multiplying with the voxel volume is numerically inconsequential.
        # However, this part of the computation and the fluence normalization in the
        # different forward models need further testing. Hence, for the moment and for
        # different voxel sizes a warning is issued.
        Adot *= self._get_unitinmm()**3

        if self._get_unitinmm() != 1:
            warnings.warn("voxel size is not 1 mm^3. Check Adot normalization.")

        return xr.DataArray(
            Adot,
            dims=["channel", "vertex", "wavelength"],
            coords={
                "channel": ("channel", channels),
                "source" : ("channel", source),
                "detector" : ("channel", detector),
                "wavelength": ("wavelength", wavelengths),
                "is_brain": ("vertex", is_brain),
            },
            attrs={"units": "mm"},
        )

    # FIXME: better name for Adot * ext. coeffs
    # FIXME: hardcoded for 2 chromophores (HbO and HbR) and wavelengths
    @staticmethod
    def compute_stacked_sensitivity(sensitivity: xr.DataArray):
        """Compute stacked HbO and HbR sensitivity matrices from fluence.

        Args:
            sensitivity (xr.DataArray): Sensitivity matrix for each vertex and
                wavelength.

        Returns:
            xr.DataArray: Stacked sensitivity matrix for each channel and vertex.
        """

        assert "wavelength" in sensitivity.dims
        wavelengths = sensitivity.wavelength.values
        assert len(wavelengths) == 2

        if "units" in sensitivity.attrs:
            units_sens = pint.Unit(sensitivity.attrs["units"])
        else:
            units_sens = pint.Unit("mm")

        ec = cedalion.nirs.get_extinction_coefficients("prahl", wavelengths)

        units_ec = ec.pint.units
        ec = ec.pint.dequantify()

        units_A = units_sens * units_ec

        nchannel = sensitivity.sizes["channel"]
        nvertices = sensitivity.sizes["vertex"]
        A = np.zeros((2 * nchannel, 2 * nvertices))

        wl1, wl2 = wavelengths
        # fmt: off
        A[:nchannel, :nvertices] = ec.sel(chromo="HbO", wavelength=wl1).values * sensitivity.sel(wavelength=wl1) # noqa: E501
        A[:nchannel, nvertices:] = ec.sel(chromo="HbR", wavelength=wl1).values * sensitivity.sel(wavelength=wl1) # noqa: E501
        A[nchannel:, :nvertices] = ec.sel(chromo="HbO", wavelength=wl2).values * sensitivity.sel(wavelength=wl2) # noqa: E501
        A[nchannel:, nvertices:] = ec.sel(chromo="HbR", wavelength=wl2).values * sensitivity.sel(wavelength=wl2) # noqa: E501
        # fmt: on

        is_brain = np.hstack([sensitivity.is_brain, sensitivity.is_brain])
        flat_chromo = ["HbO"] * nvertices + ["HbR"] * nvertices
        flat_wavelength = [wl1] * nchannel + [wl2] * nchannel
        channel = sensitivity.channel.values
        source = sensitivity.source.values
        detector = sensitivity.detector.values
        flat_channel = np.hstack((channel, channel))
        flat_source = np.hstack((source, source))
        flat_detector = np.hstack((detector, detector))
        vertex = np.hstack([np.arange(nvertices), np.arange(nvertices),])

        A = xr.DataArray(
            A,
            dims=("flat_channel", "flat_vertex"),
            coords={
                "is_brain": ("flat_vertex", is_brain),
                "chromo": ("flat_vertex", flat_chromo),
                "vertex": ("flat_vertex", vertex),
                "wavelength": ("flat_channel", flat_wavelength),
                "channel": ("flat_channel", flat_channel),
                "source": ("flat_channel", flat_source),
                "detector": ("flat_channel", flat_detector),
            },
            attrs={"units": str(units_A)},
        )

        return A


class ForwardModelEEG:
    """EEG forward model for simulating the propagation of electromagnetic
    signals in the head.

    ...

    Args:
    head_model (TwoSurfaceHeadModel): Head model containing segmentation masks
        and scalp surface.
    electrode_pos (cdt.LabeledPointCloud): Electrode positions.
    tissue_properties (xr.DataArray): Tissue properties for each tissue type.
    mesh (np.ndarray): Meshed head volume from segmentation masks.
    unitinmm (float): Unit of head model, electrodes expressed in mm.
    dipoles (np.ndarray): Positions (and orientations) of Equivalent Current
        Dipole sources

    Methods:
        generate_BEM_mesh():
            Contruct nested surface meshes for BEM simulatoin.
        compute_leadfields_BEM():
            Compute the leadfields of the dipoles at the electrodes.
        generate_FEM_mesh():
            Contruct mesh for FEM simulatoin.
        compute_leadfields_FEM():
            Compute the leadfields of the dipoles at the electrodes.
    """

    def __init__(
        self,
        head_model: TwoSurfaceHeadModel,
        elec3d: cdt.LabeledPointCloud,
        dipoles: np.ndarray,
        orientations: np.ndarray,
    ):
        """Constructor for the forward model.

        Args:
            head_model (TwoSurfaceHeadModel): Head model containing segmentation masks
                and scalp surface.
            elec3d (cdt.LabeledPointCloud): Optode positions and directions.
        """

        assert head_model.crs == "ijk"  # FIXME
        assert head_model.crs == elec3d.points.crs

        self.head_model = head_model

        self.electrode_pos = elec3d[
            elec3d.type.isin([cdc.PointType.ELECTRODE])
        ]

        # Slightly realign the optode positions to the closest scalp voxel
        self.electrode_pos = head_model.snap_to_scalp_voxels(self.electrode_pos)

        self.electrode_pos = self.electrode_pos.pint.dequantify()

        #self.tissue_properties = get_tissue_properties(
        #    self.head_model.segmentation_masks
        #)

        self.volume = self.head_model.segmentation_masks.sum("segmentation_type")
        self.volume = self.volume.values.astype(np.uint8)
        #self.mesh = self.generate_FEM_mesh()
        #self.mesh = self.generate_BEM_mesh()
        self.unitinmm = self._get_unitinmm()
        self.dipoles = dipoles
        self.orientations = orientations

    def _get_unitinmm(self):
        """Calculate length of volume grid cells.

        The forward model operates in ijk-space, in which each cell has unit length. To
        relate to physical distances pmcx needs the 'unitinmm' parameter.
        """

        pts = cdc.build_labeled_points([[0, 0, 0], [0, 0, 1]], crs="ijk", units="1")
        pts_ras = pts.points.apply_transform(self.head_model.t_ijk2ras)
        length = xrutils.norm(pts_ras[1] - pts_ras[0], pts_ras.points.crs)
        return length.pint.magnitude.item()


    def generate_BEM_mesh(self, npnt=[1922]*4, meshingparam=None):
        mesh = None
        #def projectmesh():
        cfg_tissue = self.head_model.segmentation_masks.segmentation_type.values #['wm', 'gm', 'csf', 'skull', 'scalp']
        #cfg_tissue = [t for t in cfg_tissue if t != 'air']
        segmentedmri = {k: self.head_model.segmentation_masks.sel({'segmentation_type': k}).values for k in cfg_tissue}

        from scipy.ndimage import binary_dilation

        # Combine gray & white
        segmentedmri['whitegray'] = (segmentedmri['gm'] > 0) | (segmentedmri['wm'] > 0)
        segmentedmri['whitegray'] = segmentedmri['whitegray']

        # Brain = whitegray + csf
        segmentedmri['brain'] = segmentedmri['whitegray'] | (segmentedmri['csf'] > 0)

        # Structuring element (same as ones(3,3,3))
        se = np.ones((3,3,3), dtype=bool)

        # Dilate csf with brain dilation
        segmentedmri['csf'] = (segmentedmri['csf'] > 0) | binary_dilation(segmentedmri['brain'].astype(bool), structure=se)
        segmentedmri['csf'] = segmentedmri['csf']

        # Skull = dilated csf OR skull (AND NOT csf)
        dilated_csf = binary_dilation(segmentedmri['csf'].astype(bool), structure=se)
        segmentedmri['skull'] = (dilated_csf | ((segmentedmri['skull'] > 0) & ~(segmentedmri['csf'] > 0)))

        # Scalp = scalp OR dilated skull
        dilated_skull = binary_dilation(segmentedmri['skull'].astype(bool), structure=se)
        segmentedmri['scalp'] = (segmentedmri['scalp'] > 0) | dilated_skull
        segmentedmri['scalp'] = segmentedmri['scalp']

        # Zero out boundary slices (same as setting z=1 or z=1:2 to zero)
        segmentedmri['skull'][:,:,0] = 0
        segmentedmri['csf'][:,:,0:2] = 0
        segmentedmri['whitegray'][:,:,0:3] = 0

        # Remove fields 'air' and 'brain' like rmfield
        segmentedmri.pop('air')
        segmentedmri.pop('brain') 
        segmentedmri.pop('gm') 
        segmentedmri.pop('wm') 
       
        # also remove whitegray for 3-shell BEMs
        segmentedmri.pop('whitegray') 

        mri = segmentedmri.copy()
        # update tissue types
        cfg_tissue = list(mri.keys())



        for i in range(len(cfg_tissue)):
            tissue_name = cfg_tissue[i]
            seg = mri[tissue_name]
            seglabel = cfg_tissue[i]


            # Apply thresholding, filling holes, and padding
            #seg = volumethreshold(seg, 0.5, seglabel)  # You need a Python version of volumethreshold
            seg = volumefillholes(seg)                # And volumefillholes
            seg = volumepad(seg)                      # And volumepad

            # Update dimensions and transformation
            dim = seg.shape

            #transform = mri['transform'].copy()
            #shift = ft_warp_apply(transform, np.array([[1, 1, 1]])) - ft_warp_apply(transform, np.array([[0, 0, 0]]))
            #transform[0, 3] -= shift[0, 0]
            #transform[1, 3] -= shift[0, 1]
            #transform[2, 3] -= shift[0, 2]

    
            #if cfg['method'] == 'projectmesh':
            mrix, mriy, mriz = np.meshgrid(np.arange(1, dim[0]+1),
                                           np.arange(1, dim[1]+1),
                                           np.arange(1, dim[2]+1), indexing='ij')
            ori = [
                np.mean(mrix[seg]),
                np.mean(mriy[seg]),
                np.mean(mriz[seg])
            ]

            pos, tri = triangulate_seg(seg, npnt[i], ori)  # Needs a Python version
            mesh_i = (pos, tri)
            # 'projectmesh' end



            numvoxels = np.sum(seg)
            
            ## Apply transformation
            #mesh_i = {
            #    'pos': ft_warp_apply(transform, pos),
            #    'tri': tri
            #}
            
            if i == 0:
                mesh = {tissue_name: mesh_i}
                numvoxels_list = [numvoxels]
            else:
                mesh[tissue_name] = mesh_i
                numvoxels_list.append(numvoxels) 

        
        error_threshold = 25;
        smooth_meshes = ['scalp', 'skull', 'csf', 'cortex'] #all
        correct_bnd_errors(mesh, error_threshold, smooth_meshes) # FIXME: smooth only scalp


        # Call pyiso2mesh (vol2surf())
        return mesh






    def compute_leadfields_BEM(self, conductivity=(0.3, 0.006, 0.3)):
        import mne
        import tempfile
        from os.path import join as pth
        # create tempdir for mne
        tmpdir = tempfile.TemporaryDirectory()
        subjects_dir = tmpdir.name
        os.mkdir(pth(subjects_dir, 'sample'))
        os.mkdir(pth(subjects_dir, 'sample', 'bem'))
        print('subjects_dir: ', subjects_dir)

        # export meshes in mne/freesurfer format
        names = {'csf': 'inner_skull.surf',
                 'skull': 'outer_skull.surf',
                 'scalp': 'outer_skin.surf'}
        print(pth(subjects_dir, 'sample', 'bem'))
        for k, surf in self.mesh.items():
            name = names[k]
            pos, tri = surf
            self._write_surf_nibabel(pth(subjects_dir, 'sample', 'bem', name), pos, tri)

        # create bem model
        model = mne.make_bem_model(subject='sample', ico=None,
                                   conductivity=conductivity,
                                   subjects_dir=subjects_dir)
        bem = mne.make_bem_solution(model)

        # setup source space from dipoles + orientations
        src = mne.setup_volume_source_space(subject='sample',
                                            pos={'rr': self.dipoles / 1000.0,
                                                 'nn': self.orientations},
                                            subjects_dir=subjects_dir)

        # the raw file containing the channel location + types
        raw_fname = pth(subjects_dir, 'sample', 'geo3d.fif')
        ch_names = list(self.electrode_pos.label.values)
        info = mne.create_info(ch_names, 1000., 'eeg')
        raw = mne.io.RawArray(np.zeros((len(ch_names), 1)), info)
        chan_pos = self.electrode_pos.values / 1000.0
        chans = {k: v for k, v in zip(ch_names, chan_pos)}
        montage = mne.channels.make_dig_montage(ch_pos=chans)#, nasion=None, lpa=None, rpa=None,
        raw.set_montage(montage)
        raw.save(raw_fname, overwrite=True)
        info = mne.io.read_info(raw_fname)

        # The transformation file obtained by coregistration
        trans = None #identity matrix

        # compute forward soltions
        fwd = mne.make_forward_solution(info, trans=trans, src=src, bem=bem,
                                        meg=False, # include MEG channels
                                        eeg=True, # include EEG channels
                                        mindist=1.0, # ignore sources <= 5mm from inner skull
                                        n_jobs=1) # number of jobs to run in parallel

        # convert solution (no idea why!)
        fwd = mne.convert_forward_solution(fwd, surf_ori=True)

        # extract leadfields
        leadfields = fwd['sol']['data'].T

        #return xr.DataArray(
        #    leadfields,
        #    dims=["channel", "vertex", "wavelength"],
        #    coords={
        #        "channel": ("channel", channels),
        #        "wavelength": ("wavelength", wavelengths),
        #        "is_brain": ("vertex", is_brain),
        #    },
        #    attrs={"units": "mm"},
        #)

        return leadfields


    def generate_FEM_mesh(self, meshingparam=None):

        # use CGAL mesher from nirfaster to generate FEM mesh
        # FIXME
        src_path = os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "../../../plugins/nirfaster-uFF",
            )
        )
        if src_path not in sys.path:
            sys.path.append(src_path)

        import nirfasteruff as ff

        if meshingparam is None:
            # meshing parameters; should be adjusted depending on the user's need
            meshingparam = ff.utils.MeshingParams(
                facet_distance=1.0,
                facet_size=1.0,
                general_cell_size=2.0,
                lloyd_smooth=0,
            )

        # create a nirfaster mesh
        self.mesh = ff.base.stndmesh()


        """
        # make the optical property matrix; unit in mm-1
        tissueprop = np.zeros((self.tissue_properties.shape[0]-1, 4))
        for i in range(tissueprop.shape[0]):
            tissueprop[i,0] = i+1
            tissueprop[i,1] = self.tissue_properties[i+1, 0]
            tissueprop[i,2] = self.tissue_properties[i+1, 1] * (1-self.tissue_properties[i+1, 2]) # noqa: E501
            tissueprop[i,3] = self.tissue_properties[i+1, 3]

        # all optodes x all optodes
        sources = ff.base.optode(coord=self.optode_pos.data)
        detectors = ff.base.optode(coord=self.optode_pos.data)
        n_optodes = self.optode_pos.data.shape[0]
        link = np.zeros((n_optodes*n_optodes,3), dtype=np.int32)
        ch = 0
        for i in range(n_optodes):
            for j in range(n_optodes):
                link[ch, 0] = i+1
                link[ch, 1] = j+1
                link[ch, 2] = 1
                ch += 1
        """

        # construct the mesh
        self.mesh.from_volume(
            self.volume,
            param=meshingparam,
            #prop=tissueprop,
            #src=sources,
            #det=detectors,
            #link=link,
        )
        return self.mesh

    def compute_leadfields_FEM(self):
        import duneuropy as dp
        leadfields = None
        # tbd
        return leadfields

    """
    def extract_surf_from_volume(self, segmentation_masks, seg_types, fill_holes=True):
        #### NOT WORKING PROPERLY ####
        # derive surfaces from segmentation masks
        surf = surface_from_segmentation(
            segmentation_masks, seg_types, fill_holes_in_mask=fill_holes
        )
        #all_seg_types = segmentation_masks.segmentation_type.values
        return surf

    def extract_labeled_surface_meshes(self, nodes, elements, labels):
        #### NOT WORKING PROPERLY ####
        #Extract boundary surface meshes grouped by label from a tetrahedral mesh.

        #Parameters:
        #nodes : (N, 3) ndarray
        #    Coordinates of all mesh vertices.
        #elements : (M, 4) ndarray
        #    Tetrahedral elements (indices into `nodes`, 0-based).
        #labels : (M,) ndarray
        #    Label per tetrahedron.

        #Returns:
        #List of tuples:
        #    [(surf_nodes_label0, surf_faces_label0),
        #     (surf_nodes_label1, surf_faces_label1),
        #     ...]
        #Each face group corresponds to the external surface of tetrahedra of that label.

        from collections import defaultdict
        label_to_faces = defaultdict(list)
        tet_faces = np.array([
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3]
        ])

        # Map each face to (sorted_vertex_tuple, label)
        face_count = defaultdict(list)

        for elem, label in zip(elements, labels):
            for face in tet_faces:
                tri = tuple(sorted(elem[face]))
                face_count[tri].append(label)

        # Faces on the external surface: only belong to one tetrahedron
        for face, label_list in face_count.items():
            if len(label_list) == 1:
                label_to_faces[label_list[0]].append(face)

        result = []

        for label, faces in label_to_faces.items():
            faces = np.array(faces)

            # Reindex vertices to a local set
            unique_node_indices, inverse_indices = np.unique(faces.flatten(), return_inverse=True)
            surf_nodes = nodes[unique_node_indices]
            surf_faces = inverse_indices.reshape((-1, 3))

            result.append((label, surf_nodes, surf_faces))

        return result
    """

    def _read_surf_mne(self, fname):
        import mne
        return mne.surface.read_surface(fname)

    def _write_surf_nibabel(self, fname, vert, face):
        """
        Write a FreeSurfer surface file using nibabel.

        Parameters:
        fname : str
            Output filename (usually ends in .surf).
        vert : ndarray
            Nx3 array of vertex coordinates.
        face : ndarray
            Mx3 array of triangle indices (MATLAB-style 1-based).
        """
        import nibabel as nib
        if vert.shape[1] != 3:
            raise ValueError("vert must be an Nx3 matrix")
        if face.shape[1] != 3:
            raise ValueError("face must be an Mx3 matrix")

        # Convert MATLAB-style 1-based indices to Python-style 0-based
        #face = face - 1

        # Use nibabel to write the surface
        nib.freesurfer.io.write_geometry(fname, vert, face)
        return




def apply_inv_sensitivity(
    od: cdt.NDTimeSeries, inv_sens: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply the inverted sensitivity matrix to optical density data.

    Args:
        od: time series of optical density data
        inv_sens: the inverted sensitivity matrix

    Returns:
        Two DataArrays for the brain and scalp with the reconcstructed time series per
        vertex and chromophore.
    """

    units_str = inv_sens.attrs.get("units", None)

    od_stacked = od.stack({"flat_channel": ["wavelength", "channel"]})
    od_stacked = od_stacked.pint.dequantify()

    delta_conc = inv_sens @ od_stacked

    # Construct a multiindex for dimension flat_vertex from chromo and vertex.
    # Afterwards use this multiindex to unstack flat_vertex. The resulting array
    # has again dimensions vertex and chromo.
    delta_conc = delta_conc.set_xindex(["chromo", "vertex"])
    delta_conc = delta_conc.unstack("flat_vertex")

    # unstacking flat_vertex makes is_brain 2D. is_brain[0,:] == is_brain[1,:]
    is_brain = delta_conc.is_brain[0, :].values

    delta_conc_brain = delta_conc.sel(vertex=is_brain)
    delta_conc_scalp = delta_conc.sel(vertex=~is_brain)

    if units_str is not None:
        delta_conc_brain.attrs["units"] = units_str
        delta_conc_scalp.attrs["units"] = units_str

    return delta_conc_brain, delta_conc_scalp



def stack_flat_vertex(array: xr.DataArray):
    dims = ("chromo", "vertex")

    for dim in dims:
        if dim not in array.dims:
            raise ValueError(f"cannot stack missing dimension {dim}")

    return array.stack({"flat_vertex": dims})


def unstack_flat_vertex(array: xr.DataArray):
    if "flat_vertex" not in array.dims:
        raise ValueError("array misses dimension 'flat_vertex'.")

    coords = ("chromo", "vertex")
    for coord in coords:
        if coord not in array.coords:
            raise ValueError(f"array misses coordinate '{coord}'.")

    return array.set_xindex(coords).unstack("flat_vertex")


def stack_flat_channel(array: xr.DataArray):
    dims = ("wavelength", "channel")

    for dim in dims:
        if dim not in array.dims:
            raise ValueError(f"cannot stack missing dimension {dim}")

    return array.stack({"flat_channel": dims})


def unstack_flat_channel(array: xr.DataArray):
    if "flat_channel" not in array.dims:
        raise ValueError("array misses dimension 'flat_channel'.")

    coords = ("wavelength", "channel")
    for coord in coords:
        if coord not in array.coords:
            raise ValueError(f"array misses coordinate '{coord}'.")

    unstacked = array.set_xindex(coords).unstack("flat_channel")

    # source and detector are unstacked into 2D arrays with dims channel and wavelength.
    # Assert that these coordinates do not vary along the wavelength dimension and
    # then reduce them to channel-only coordinates.

    for coord_name in ["source", "detector"]:
        c = unstacked.coords[coord_name]
        c_wl0 = (
            c[{"wavelength": 0}].copy().drop_vars(["wavelength", "source", "detector"])
        )
        if not (c_wl0 == c).all().item():
            raise ValueError(
                f"coord {coord_name} varies over wavelength after unstacking."
            )
        #unstacked = unstacked.drop_vars(coord_name)
        unstacked = unstacked.assign_coords({coord_name: c_wl0})

    return unstacked




### projectmesh
import numpy as np
from scipy import ndimage
import scipy.ndimage
from scipy.ndimage import binary_fill_holes
from skimage.measure import label
from skimage import measure

def sub2ind(dim, i, j, k):
    return i + (j - 1) * dim[0] + (k - 1) * dim[0] * dim[1]

def volumefillholes(seg):
    return binary_fill_holes(seg)

#def mesh_sphere(npnt):
#    # Generates a sphere using icosphere method
#    # You may use trimesh or other libs
#    import trimesh
#    sphere = trimesh.creation.icosphere(subdivisions=3, radius=1)#int(np.log2(npnt / 42)), radius=1)
#    # subdivisions==5 -> 10242
#    # subdivisions==4 -> 2562
#    # subdivisions==3 -> 642
#    #pos, tri = mesh_sphere(npnt, 'ksphere');
#    return sphere.vertices, sphere.faces


from scipy.spatial import ConvexHull
from math import sqrt, pi, cos, sin

# --- ksphere subfunction ---
def ksphere(N):
    h_list = -1 + 2 * np.arange(N) / (N - 1)
    theta_list = np.arccos(h_list)
    phi_list = np.zeros_like(theta_list)

    for k in range(N):
        h = h_list[k]
        if k == 0 or k == N - 1:
            phi_list[k] = 0
        else:
            phi_list[k] = (phi_list[k - 1] + 3.6 / sqrt(N * (1 - h ** 2))) % (2 * pi)

    az = phi_list
    el = theta_list - pi / 2
    x = np.cos(az) * np.cos(el)
    y = np.sin(az) * np.cos(el)
    z = np.sin(el)

    pos = np.column_stack((x, y, z))
    tri = ConvexHull(pos).simplices
    return pos, tri






def volumepad(input_array, n=1):
    """
    Adds a layer of padding around a 3D volume to ensure the tissue 
    can be meshed up to the edges.

    Parameters:
    - input_array: 3D numpy array (bool or numeric)
    - n: number of padding layers (default is 1)

    Returns:
    - output: padded 3D numpy array
    """
    dim = input_array.shape

    # Determine the dtype and create the padded output
    if input_array.dtype == bool:
        output = np.zeros((dim[0] + 2*n, dim[1] + 2*n, dim[2] + 2*n), dtype=bool)
    else:
        output = np.zeros((dim[0] + 2*n, dim[1] + 2*n, dim[2] + 2*n), dtype=input_array.dtype)

    # Insert the original data into the padded volume
    output[n:dim[0]+n, n:dim[1]+n, n:dim[2]+n] = input_array

    return output


from scipy.ndimage import label, binary_fill_holes
from skimage.morphology import remove_small_holes

def volumefillholes(input_array, along=None):
    """
    Fill holes in a 3D segmented volume.

    Parameters:
    - input_array: 3D numpy array (binary mask)
    - along: axis along which to perform 2D hole filling (1, 2, or 3). 
             If None, performs 3D hole filling.

    Returns:
    - output: 3D numpy array with holes filled
    """

    input_array = input_array.astype(bool)  # Ensure binary
    output = input_array.copy()

    if along is None:
        # 3D hole filling (analogous to SPM's spm_bwlabel + inversion trick)

        # Pad the volume
        inflate = volumepad(input_array, 1)

        # Label connected components in the inverted (background) space
        structure = np.array([[[0,1,0],
                               [1,1,1],
                               [0,1,0]],
                              [[1,1,1],
                               [1,1,1],
                               [1,1,1]],
                              [[0,1,0],
                               [1,1,1],
                               [0,1,0]]], dtype=bool)  # 18-connectivity, matching MATLAB
        lab, num = label(~inflate, structure=structure)

        if num > 1:
            # Keep only the background connected to the corner (lab==lab[0,0,0])
            inflate[lab != lab[0,0,0]] = True
            # Remove the padding
            output = inflate[1:-1, 1:-1, 1:-1]
        else:
            output = input_array

    else:
        dim = input_array.shape
        if along == 1:
            for i in range(dim[0]):
                slice_2d = input_array[i, :, :]
                output[i, :, :] = binary_fill_holes(slice_2d)
        elif along == 2:
            for i in range(dim[1]):
                slice_2d = input_array[:, i, :]
                output[:, i, :] = binary_fill_holes(slice_2d)
        elif along == 3:
            for i in range(dim[2]):
                slice_2d = input_array[:, :, i]
                output[:, :, i] = binary_fill_holes(slice_2d)
        else:
            raise ValueError(f'Invalid dimension {along} to slice the volume')

    return output


from scipy.ndimage import label

def volumethreshold(input_array, threshold=0, tissuelabel='volume'):
    """
    Applies a threshold and keeps the largest connected component 
    to clean up small blobs (e.g., vitamin E capsules).

    Parameters:
    - input_array: 3D numpy array (probabilistic or binary mask)
    - threshold: relative threshold (default 0)
    - tissuelabel: label string (for printing)

    Returns:
    - output: binary 3D numpy array after threshold + largest cluster selection
    """

    # Ensure input is numeric or boolean
    if not (np.issubdtype(input_array.dtype, np.floating) or np.issubdtype(input_array.dtype, np.bool_)):
        input_array = input_array.astype(np.float64)

    # Apply threshold if input is not already logical
    if not np.issubdtype(input_array.dtype, np.bool_):
        if threshold is None:
            raise ValueError('If the input volume is not boolean, you need to define a threshold value.')
        print(f"Thresholding {tissuelabel} at a relative threshold of {threshold:.3f}")
        max_val = np.max(input_array)
        output = (input_array > (threshold * max_val)).astype(np.float64)
    else:
        # If already boolean, no thresholding is needed
        output = input_array.astype(np.float64)

    # Cluster the connected tissue (6-connectivity, like MATLAB's spm_bwlabel(...,6))
    structure = np.array([[[0,0,0],
                           [0,1,0],
                           [0,0,0]],
                          [[0,1,0],
                           [1,1,1],
                           [0,1,0]],
                          [[0,0,0],
                           [0,1,0],
                           [0,0,0]]], dtype=bool)  # 6-connectivity
    cluster, n = label(output, structure=structure)

    if n > 1:
        # Count voxel count for each cluster (skip label 0)
        counts = np.bincount(cluster.flat)
        counts[0] = 0  # background should be zero
        largest_cluster = np.argmax(counts)
        output = (cluster == largest_cluster)
    else:
        # Only one cluster, keep it
        output = (cluster == 1)

    return output


def triangulate_seg(seg, npnt, origin=None):
    """
    seg    = 3D-matrix (boolean) containing the segmented volume
    npnt   = requested number of vertices
    origin = 1x3 vector specifying the location of the origin of the sphere
             in voxel indices. This argument is optional. If undefined, the
             origin of the sphere will be in the centre of the volume.
    """
    seg = (seg != 0)
    dim = seg.shape
    len_ = int(np.ceil(np.sqrt(np.sum(np.array(dim) ** 2)) / 2))

    if not np.any(seg):
        raise ValueError('The segmentation is empty')

    # define the origin if not provided
    if origin is None:
        origin = [dim[0] / 2, dim[1] / 2, dim[2] / 2]

    # fill holes
    seg = volumefillholes(seg)

    # label connected components
    lab, num = label(seg)#, connectivity=3, return_num=True)

    if num > 1:
        print('Warning: multiple blobs detected, using only the largest')
        n = np.bincount(lab.ravel())[1:]  # exclude background
        ix = np.argmax(n) + 1
        seg = lab == ix

    # unit sphere
    #pnt, tri = mesh_sphere(npnt)
    pnt, tri = ksphere(npnt)
    
    ishollow = False

    for i in range(npnt):
        lin = np.outer(np.arange(0, len_ + 0.5, 0.5), pnt[i])
        lin[:, 0] += origin[0]
        lin[:, 1] += origin[1]
        lin[:, 2] += origin[2]

        lin_rounded = np.round(lin).astype(int)

        # valid mask
        valid = (
            (lin_rounded[:, 0] >= 0) & (lin_rounded[:, 0] < dim[0]) &
            (lin_rounded[:, 1] >= 0) & (lin_rounded[:, 1] < dim[1]) &
            (lin_rounded[:, 2] >= 0) & (lin_rounded[:, 2] < dim[2])
        )

        lin_valid = lin_rounded[valid]

        indices = (
            lin_valid[:, 0],
            lin_valid[:, 1],
            lin_valid[:, 2]
        )

        int_vals = seg[indices]

        if np.any(np.diff(int_vals) == 1):
            ishollow = True

        sel = np.where(int_vals)[0]

        if sel.size > 0:
            idx = sel[-1]
            pnt[i, :] = lin_valid[idx, :]
        else:
            pnt[i, :] = lin_valid[-1, :]

    if ishollow:
        print('Warning: the segmentation is not star-shaped, please check the surface mesh')

    return pnt, tri







def correct_bnd_errors(bnd, errorthreshold, smooth):
    """
    Correction of triangular surface meshes for abnormal vertex outliers and
    application of smoothing routines.
    """
    num_corrected = 0
    for ii in bnd.keys():
        for j in range(bnd[ii][1].shape[0]): 
            pos = bnd[ii][0]
            tri = bnd[ii][1]
            v1 = tri[j, 0]
            v2 = tri[j, 1]
            v3 = tri[j, 2]

            if np.linalg.norm(pos[v1, :] - pos[v2, :]) > errorthreshold:
                if np.linalg.norm(pos[v1, :] - pos[v3, :]) > errorthreshold:
                    nb_row = np.where(tri == v1)
                    nbs = np.unique(tri[nb_row[0], :])
                    nbs = nbs[nbs != v1]
                    pos[v1, :] = np.sum(pos[nbs, :], axis=0) / len(nbs)
                    num_corrected += 1
                elif np.linalg.norm(pos[v2, :] - pos[v3, :]) > errorthreshold:
                    nb_row = np.where(tri == v2)
                    nbs = np.unique(tri[nb_row[0], :])
                    nbs = nbs[nbs != v2]
                    pos[v2, :] = np.sum(pos[nbs, :], axis=0) / len(nbs)
                    num_corrected += 1
            elif (np.linalg.norm(pos[v2, :] - pos[v3, :]) > errorthreshold and
                  np.linalg.norm(pos[v1, :] - pos[v3, :]) > errorthreshold):
                nb_row = np.where(tri == v3)
                nbs = np.unique(tri[nb_row[0], :])
                nbs = nbs[nbs != v3]
                pos[v3, :] = np.sum(pos[nbs, :], axis=0) / len(nbs)
                num_corrected += 1

        if ii in smooth:
            pos = lpflow_trismooth(pos, tri)
        bnd[ii] = (pos, tri)

    print(f"Corrected {num_corrected} vertex positions.")
    return bnd


def lpflow_trismooth(xyz, t):
    """
    Laplace flow mesh smoothing for vertex ring.
    """
    if t.shape[1] != 3:
        raise ValueError('Triangle element matrix should be mx3!')
    if xyz.shape[1] != 3:
        raise ValueError('Vertices should be nx3!')

    conn = neighborelem(t, np.max(t))
    xyzn = xyz.copy()

    for k in range(len(xyz)):
        indt01 = conn[k]
        indv01 = np.unique(t[indt01, :].flatten())
        vdist = xyz[indv01, :] - xyz[k, :]
        dist = np.sqrt(np.sum(vdist * vdist, axis=1))
        indaux1 = np.where(dist == 0)[0]
        vdist = np.delete(vdist, indaux1, axis=0)
        if len(dist) == 0:
            xyzn[k, :] = np.nan
        else:
            d = len(vdist)
            vcorr = np.sum(vdist / d, axis=0)
            xyzn[k, :] = xyz[k, :] + vcorr
    return xyzn


def neighborelem(tri, n_vertices):
    """
    Find neighboring elements for each vertex.
    """
    conn = [[] for _ in range(n_vertices + 1)]
    for i in range(tri.shape[0]):
        for j in range(3):
            conn[tri[i, j]].append(i)
    return conn
 


