from dataclasses import dataclass
import logging
from typing import Optional
import os.path

import numpy as np
import pandas as pd
import scipy.sparse
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
    """Head Model class to represent a segmented head. Its main functions are
    reduced to work on voxel projections to scalp and cortex surfaces.

    ...

    Attributes
    ----------
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

    Methods
    -------
    from_segmentation(cls, segmentation_dir, mask_files, landmarks_ras_file, brain_seg_types, scalp_seg_types, smoothing, brain_face_count, scalp_face_count)
        Construct instance from segmentation masks in NIfTI format.
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
        brain_face_count: Optional[int] = 60000,
        scalp_face_count: Optional[int] = 60000,
        fill_holes: bool = False,
    ) -> "TwoSurfaceHeadModel":
        """Constructor from binary masks as gained from segmented MRI scans.

        Parameters
        ----------
        segmentation_dir : str
            Folder containing the segmentation masks in NIFTI format.
        mask_files : dict[str, str]
            Dictionary mapping segmentation types to NIFTI filenames.
        landmarks_ras_file : Optional[str]
            Filename of the landmarks in RAS space.
        brain_seg_types : list[str]
            List of segmentation types to be included in the brain surface.
        scalp_seg_types : list[str]
            List of segmentation types to be included in the scalp surface.
        smoothing : float
            Smoothing factor for the brain and scalp surfaces.
        brain_face_count : Optional[int]
            Number of faces for the brain surface.
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
        brain_ijk = surface_from_segmentation(segmentation_masks, brain_seg_types, fill_holes_in_mask=fill_holes)

        # we need the single outer surface from the scalp. The inner border between
        # scalp and skull is not interesting here. Hence, all segmentation types are
        # grouped together, yielding a uniformly filled head volume.
        all_seg_types = segmentation_masks.segmentation_type.values
        scalp_ijk = surface_from_segmentation(segmentation_masks, all_seg_types, fill_holes_in_mask=fill_holes)

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

        Parameters
        ----------
        transform : cdt.AffineTransform
            Affine transformation matrix (4x4) to be applied.

        Returns
        -------
        TwoSurfaceHeadModel
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

        Parameters
        ----------
        foldername : str
            Folder to save the head model into.

        Returns
        -------
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
            self.landmarks.drop_vars('type').to_netcdf(os.path.join(foldername, "landmarks.nc"))
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

        Parameters
        ----------
        foldername : str
            Folder to load the head model from.

        Returns
        -------
        TwoSurfaceHeadModel
            Loaded head model.
        """

        # Check if all files exist
        for fn in ["segmentation_masks.nc", "brain.ply", "scalp.ply",
                   "t_ijk2ras.nc", "t_ras2ijk.nc", "voxel_to_vertex_brain.npz",
                   "voxel_to_vertex_scalp.npz"]:
            if not os.path.exists(os.path.join(foldername, fn)):
                raise ValueError("%s does not exist." % os.path.join(foldername, fn))

        # Load all attributes from folder
        segmentation_masks = xr.load_dataset(os.path.join(foldername, 'segmentation_masks.nc'))
        brain =  trimesh.load(os.path.join(foldername, 'brain.ply'), process=False)
        scalp =  trimesh.load(os.path.join(foldername, 'scalp.ply'), process=False)
        if os.path.exists(os.path.join(foldername, 'landmarks.nc')):
            landmarks_ijk = xr.load_dataset(os.path.join(foldername, 'landmarks.nc'))
            landmarks_ijk = xr.DataArray(
                    landmarks_ijk.to_array()[0],
				    coords={
					    "label": ("label", landmarks_ijk.label.values),
					    "type": ("label", [cdc.PointType.LANDMARK] * len(landmarks_ijk.label)),
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
        t_ijk2ras = cdc.affine_transform_from_numpy(np.array(t_ijk2ras.to_dataarray()[0]), "ijk",
                                                    "unknown", "1", "mm")
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

        Parameters
        ----------
        points : cdt.LabeledPointCloud
            Points to be aligned and snapped to the scalp surface.

        Returns
        -------
        cdt.LabeledPointCloud
            Points aligned and snapped to the scalp surface.
        """

        assert self.landmarks is not None, "Please add landmarks in RAS to head \
                                            instance."
        t = register_trans_rot_isoscale(self.landmarks, points)
        transformed = points.points.apply_transform(t)
        snapped = self.scalp.snap(transformed)
        return snapped


class ForwardModel:
    """Forward model for simulating light transport in the head.

    ...

    Attributes
    ----------
    head_model : TwoSurfaceHeadModel
        Head model containing voxel projections to brain and scalp surfaces.
    optode_pos : cdt.LabeledPointCloud
        Optode positions.
    optode_dir : xr.DataArray
        Optode orientations (directions of light beams).
    tissue_properties : xr.DataArray
        Tissue properties for each tissue type.
    volume : xr.DataArray
        Voxelated head volume from segmentation masks.
    unitinmm : float
        Unit of head model, optodes expressed in mm.
    measurement_list : pd.DataFrame
        List of measurements of experiment with source, detector, channel and wavelength.

    Methods
    -------
    compute_fluence(nphoton)
        Compute fluence for each channel and wavelength from photon simulation.
    compute_sensitivity(fluence_all, fluence_at_optodes)
        Compute sensitivity matrix from fluence.
    """

    def __init__(
        self,
        head_model: TwoSurfaceHeadModel,
        geo3d: cdt.LabeledPointCloud,
        measurement_list: pd.DataFrame,
    ):
        """Constructor for the forward model.

        Parameters
        ----------
        head_model : TwoSurfaceHeadModel
            Head model containing voxel projections to brain and scalp surfaces.
        geo3d : cdt.LabeledPointCloud
            Optode positions and directions.
        measurement_list : pd.DataFrame
            List of measurements of experiment with source, detector, channel and wavelength.
        """

        assert head_model.crs == "ijk"  # FIXME
        assert head_model.crs == geo3d.points.crs

        self.head_model = head_model

        self.optode_pos = geo3d[
            geo3d.type.isin([cdc.PointType.SOURCE, cdc.PointType.DETECTOR])
        ]

        #FIXME make sure that optode is in scalp voxel
        self.optode_pos = self.optode_pos.round()

        self.optode_dir = -head_model.scalp.get_vertex_normals(self.optode_pos)

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

    def _get_fluence_from_mcx(self, i_optode: int, nphoton: int):
        """Run MCX simulation to get fluence for one optode.

        Parameters
        ----------
        i_optode : int
            Index of the optode.
        nphoton : int
            Number of photons to simulate.

        Returns
        -------
        np.ndarray
            Fluence in each voxel.
        """

        cfg = {
            "nphoton": nphoton,
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
            "seed": int(np.floor(np.random.rand() * 1e7)),
            "issavedet": 1,
            "unitinmm": self.unitinmm,
        }

        import pmcx
        result = pmcx.run(cfg)

        fluence = result["flux"][:, :, :, 0]  # there is only one time bin
        fluence = fluence * cfg["tstep"] / result["stat"]["normalizer"]

        return fluence

    def _fluence_at_optodes(self, fluence, emitting_opt):
        """Fluence caused by one optode at the positions of all other optodes.

        Parameters
        ----------
        fluence : np.ndarray
            Fluence in each voxel.
        emitting_opt : int
            Index of the emitting optode.

        Returns
        -------
        np.ndarray
            Fluence at all optode positions.
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

    def compute_fluence(self, nphoton: int = 1e8):
        """Compute fluence for each channel and wavelength from photon simulation.

        Parameters
        ----------
        nphoton : int
            Number of photons to simulate.

        Returns
        -------
        xr.DataArray
            Fluence in each voxel for each channel and wavelength.
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

            # run MCX
            # shape: [i,j,k]
            fluence = self._get_fluence_from_mcx(i_opt, nphoton=nphoton)

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


    def compute_sensitivity_all(self, fluence_all, fluence_at_optodes):
        """Compute sensitivity matrix from fluence.

        Parameters
        ----------
        fluence_all : xr.DataArray
            Fluence in each voxel for each wavelength.
        fluence_at_optodes : xr.DataArray
            Fluence at all optode positions for each wavelength.

        Returns
        -------
        xr.DataArray
            Sensitivity matrix for each channel, vertex and wavelength.
        """

        channels = self.measurement_list.channel.unique().tolist()
        n_channel = len(channels)
        wavelengths = self.measurement_list.wavelength.unique().tolist()
        n_wavelength = len(wavelengths)

        n_brain = self.head_model.brain.nvertices
        n_scalp = self.head_model.scalp.nvertices
        Adot_brain = np.zeros((n_channel, n_brain, n_wavelength))
        Adot_scalp = np.zeros((n_channel, n_scalp, n_wavelength))
        Adot = np.zeros((n_channel, n_voxels, n_wavelength))

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

        Parameters
        ----------
        fluence_all : xr.DataArray
            Fluence in each voxel for each wavelength.
        fluence_at_optodes : xr.DataArray
            Fluence at all optode positions for each wavelength.

        Returns
        -------
        xr.DataArray
            Sensitivity matrix for each channel, vertex and wavelength.
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
    def compute_stacked_sensitivity(self, sensitivity: xr.DataArray):
        """Compute stacked HbO and HbR sensitivity matrices from fluence.

        Parameters
        ----------
        sensitivity : xr.DataArray
            Sensitivity matrix for each vertex and wavelength.

        Returns
        -------
        xr.DataArray
            Stacked sensitivity matrix for each channel and vertex.
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
