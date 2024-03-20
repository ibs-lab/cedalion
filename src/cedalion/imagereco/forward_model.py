from dataclasses import dataclass
import logging
from typing import Optional
import os.path

import numpy as np
import pandas as pd
import pmcx
import scipy.sparse
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
    segmentation_masks: xr.DataArray
    brain: cdc.Surface
    scalp: cdc.Surface
    landmarks: Optional[cdt.LabeledPointCloud]
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
    ) -> "TwoSurfaceHeadModel":
        """Model of a segmented head.

        Based on a segmented MRI scan two surfaces are estimated for the brain
        and the scalp.
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
        brain_ijk = surface_from_segmentation(segmentation_masks, brain_seg_types)

        # we need the single outer surface from the scalp. The inner border between
        # scalp and skull is not interesting here. Hence, all segmentation types are
        # grouped together, yielding a uniformly filled head volume.
        all_seg_types = segmentation_masks.segmentation_type.values
        scalp_ijk = surface_from_segmentation(segmentation_masks, all_seg_types)

        # smooth surfaces
        if smoothing > 0:
            brain_ijk = brain_ijk.smooth(smoothing)
            scalp_ijk = scalp_ijk.smooth(smoothing)

        # reduce surface face counts
        if brain_face_count is not None:
            brain_ijk = brain_ijk.decimate(brain_face_count)

        if scalp_face_count is not None:
            scalp_ijk = scalp_ijk.decimate(scalp_face_count)

        #
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
        assert self.brain.crs == self.scalp.crs
        if self.landmarks is not None:
            assert self.scalp.crs == self.landmarks.points.crs
        return self.brain.crs

    def apply_transform(self, transform: cdt.AffineTransform) -> "TwoSurfaceHeadModel":
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

    # FIXME maybe this should not be in this class, especially since the
    # algorithm is not good.
    @cdc.validate_schemas
    def align_and_snap_to_scalp(
        self, points: cdt.LabeledPointCloud
    ) -> cdt.LabeledPointCloud:
        assert self.landmarks is not None, "Please add landmarks in RAS to head \
                                            instance."
        t = register_trans_rot_isoscale(self.landmarks, points)
        transformed = points.points.apply_transform(t)
        snapped = self.scalp.snap(transformed)
        return snapped


class ForwardModel:
    def __init__(
        self,
        head_model: TwoSurfaceHeadModel,
        geo3d: cdt.LabeledPointCloud,
        measurement_list: pd.DataFrame,
    ):
        assert head_model.crs == "ijk"  # FIXME
        assert head_model.crs == geo3d.points.crs

        self.head_model = head_model

        self.optode_pos = geo3d[
            geo3d.type.isin([cdc.PointType.SOURCE, cdc.PointType.DETECTOR])
        ]

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

        result = pmcx.run(cfg)

        fluence = result["flux"][:, :, :, 0]  # there is only one time bin
        fluence = fluence * cfg["tstep"] / result["stat"]["normalizer"]

        return fluence

    def _fluence_at_optodes(self, fluence, emitting_opt):
        """Fluence caused by one optode at the positions of all other optodes."""
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

    def compute_sensitivity(self, fluence_all, fluence_at_optodes):
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
        wavelengths = self.measurement_list.wavelength.unique()
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
