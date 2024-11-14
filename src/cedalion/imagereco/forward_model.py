from dataclasses import dataclass
import logging
from typing import Optional
import os.path
import sys

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.spatial import KDTree
import trimesh
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
from cedalion.geometry.registration import register_trans_rot_isoscale
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion.geometry.segmentation import surface_from_segmentation
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
        fill_holes: bool = False,
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
        """Constructor from binary masks, brain and head surfaces as gained from MRI scans.

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
        segmentation_masks = xr.load_dataset(
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
        t_ijk2ras = xr.load_dataset(os.path.join(foldername, 't_ijk2ras.nc'))
        t_ras2ijk = xr.load_dataset(os.path.join(foldername, 't_ras2ijk.nc'))
        voxel_to_vertex_brain = scipy.sparse.load_npz(os.path.join(foldername,
                                                     'voxel_to_vertex_brain.npz'))
        voxel_to_vertex_scalp = scipy.sparse.load_npz(os.path.join(foldername,
                                                      'voxel_to_vertex_scalp.npz'))

        # Construct TwoSurfaceHeadModel
        brain_ijk = cdc.TrimeshSurface(brain, 'ijk', cedalion.units.Unit("1"))
        scalp_ijk = cdc.TrimeshSurface(scalp, 'ijk', cedalion.units.Unit("1"))
        t_ijk2ras = cdc.affine_transform_from_numpy(
            np.array(t_ijk2ras.to_dataarray()[0]), "ijk", "unknown", "1", "mm"
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
                except:
                    shape = self.segmentation_masks.to_dataarray().shape[-3:]
                voxels = np.array(np.unravel_index(voxel_idx, shape)).T

                # Choose the closest voxel
                dist = np.linalg.norm(voxels - np.array(a.pint.dequantify()), axis=1)
                voxel_idx = np.argmin(dist)

            else:
                # If no voxel maps to that scalp surface vertex, 
                # simply choose the closest of all scalp voxels
                voxels = voxels_from_segmentation(self.segmentation_masks, ["scalp"]).voxels
                if len(voxels) == 0:
                    try:
                        scalp_mask = self.segmentation_masks.sel(segmentation_type="scalp").to_dataarray()
                    except:
                        scalp_mask = self.segmentation_masks.sel(segmentation_type="scalp")
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
        self.optode_dir = -head_model.scalp.get_vertex_normals(self.optode_pos)
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

    def _get_fluence_from_mcx(self, i_optode: int, **kwargs):
        """Run MCX simulation to get fluence for one optode.

        Args:
            i_optode (int): Index of the optode.
            nphoton  (int): Number of photons to simulate.

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
            "outputtype": "fluence",
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
            nphoton (int): Number of photons to simulate.
            along with other pmcx/pmcxcl accepted input fields,
            see https://pypi.org/project/pmcx/

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
        )

        fluence_at_optodes = xr.DataArray(
            fluence_at_optodes,
            dims=["optode1", "optode2", "wavelength"],
            coords={
                "optode1": self.optode_pos.label.values,
                "optode2": self.optode_pos.label.values,
                "wavelength": wavelengths,
            },
        )

        return fluence_all, fluence_at_optodes


    def compute_fluence_nirfaster(
            self, meshingparam = None
            ):
        """Compute fluence for each channel and wavelength using NIRFASTer package.

        Args:
            meshingparam (ff.utils.MeshingParam) Parameters to be used by the CGAL
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
        )

        fluence_at_optodes = xr.DataArray(
            fluence_at_optodes,
            dims=["optode1", "optode2", "wavelength"],
            coords={
                "optode1": self.optode_pos.label.values,
                "optode2": self.optode_pos.label.values,
                "wavelength": wavelengths,
            },
        )

        return fluence_all, fluence_at_optodes


    def compute_sensitivity_all(self, fluence_all, fluence_at_optodes):
        """Compute sensitivity matrix from fluence.

        Args:
            fluence_all (xr.DataArray): Fluence in each voxel for each wavelength.
            fluence_at_optodes (xr.DataArray): Fluence at all optode positions for each
                wavelength.

        Returns:
            xr.DataArray: Sensitivity matrix for each channel, vertex and wavelength.
        """

        channels = self.measurement_list.channel.unique().tolist()
        n_channel = len(channels)
        wavelengths = self.measurement_list.wavelength.unique().tolist()
        n_wavelength = len(wavelengths)

        n_brain = self.head_model.brain.nvertices
        n_scalp = self.head_model.scalp.nvertices
        Adot_brain = np.zeros((n_channel, n_brain, n_wavelength))
        Adot_scalp = np.zeros((n_channel, n_scalp, n_wavelength))
        # Adot = np.zeros((n_channel, n_voxels, n_wavelength)) # FIXME?

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

        return xr.DataArray(
            Adot,
            dims=["channel", "vertex", "wavelength"],
            coords={
                "channel": ("channel", channels),
                "wavelength": ("wavelength", wavelengths),
                "is_brain": ("vertex", is_brain),
            },
        )


    def compute_sensitivity(self, fluence_all, fluence_at_optodes):
        """Compute sensitivity matrix from fluence.

        Args:
            fluence_all (xr.DataArray): Fluence in each voxel for each wavelength.
            fluence_at_optodes (xr.DataArray): Fluence at all optode positions for each
                wavelength.

        Returns:
            xr.DataArray: Sensitivity matrix for each channel, vertex and wavelength.
        """

        channels = self.measurement_list.channel.unique().tolist()
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

        return xr.DataArray(
            Adot,
            dims=["channel", "vertex", "wavelength"],
            coords={
                "channel": ("channel", channels),
                "wavelength": ("wavelength", wavelengths),
                "is_brain": ("vertex", is_brain),
            },
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

        ec = cedalion.nirs.get_extinction_coefficients("prahl", wavelengths)

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

        A = xr.DataArray(A, dims=("flat_channel", "flat_vertex"))

        return A
