"""Forward model for simulating light transport in the head.

NOTE: Cedalion currently supports two ways to compute fluence:
1) via monte-carlo simulation using the MonteCarloXtreme (MCX) package, and
2) via the finite element method (FEM) using the NIRFASTer package.
While MCX is automatically installed using pip, NIRFASTER has to be manually installed
runnning <$ bash install_nirfaster.sh CPU # or GPU> from a within your cedalion root
directory.
"""

from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Optional
import os.path
import warnings
import sys
from pathlib import Path

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
from cedalion.io.forward_model import FluenceFile, save_Adot

from .tissue_properties import get_tissue_properties

logger = logging.getLogger("cedalion")



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
        self.measurement_list = measurement_list

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
                                                        self.head_model.segmentation_masks, 
                                                        self.measurement_list.wavelength.unique()
                                                    )

        self.volume = self.head_model.segmentation_masks.sum("segmentation_type")
        self.volume = self.volume.values.astype(np.uint8)
        self.unitinmm = self._get_unitinmm()


    def _get_unitinmm(self):
        """Calculate length of volume grid cells.

        The forward model operates in ijk-space, in which each cell has unit length. To
        relate to physical distances pmcx needs the 'unitinmm' parameter.
        """

        pts = cdc.build_labeled_points([[0, 0, 0], [0, 0, 1]], crs="ijk", units="1")
        pts_ras = pts.points.apply_transform(self.head_model.t_ijk2ras)
        length = xrutils.norm(pts_ras[1] - pts_ras[0], pts_ras.points.crs)
        return length.pint.magnitude.item()

    def _get_fluence_from_mcx(self, i_optode: int, i_wl: int, **kwargs) -> np.ndarray:
        """Run MCX simulation to get fluence for one optode.

        Args:
            i_optode: Index of the optode.
            **kwargs: Additional keywords are passed to MCX's configuration dict.

        Returns:
            np.ndarray: Fluence in each voxel.
        """

        kwargs.setdefault("nphoton", 1e8)
        kwargs.setdefault("cuda", True)

        cfg = {
            "nphoton": kwargs['nphoton'],
            "vol": self.volume,
            "tstart": 0,
            "tend": 5e-9,
            "tstep": 5e-9,
            "srcpos": self.optode_pos.values[i_optode],
            "srcdir": self.optode_dir.values[i_optode],
            "prop": self.tissue_properties[:,:,i_wl],
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

    def compute_fluence_mcx(self, fluence_fname : str | Path, **kwargs):
        """Compute fluence for each channel and wavelength using MCX package.

        Args:
            fluence_fname : the output hdf5 file to store the fluence
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

        units = "1 / millimeter ** 2"

        fluence_at_optodes = xr.DataArray(
            dims=["optode1", "optode2", "wavelength"],
            coords={
                "optode1": self.optode_pos.label.values,
                "optode2": self.optode_pos.label.values,
                "wavelength": wavelengths,
            },
            attrs={"units": "1 / millimeter ** 2"},
        )

        with FluenceFile(fluence_fname, "w") as fluence_file:
            fluence_file.create_fluence_dataset(
                self.optode_pos,
                wavelengths,
                self.volume.shape,
                units
            )

            for i_opt in range(n_optodes):
                label = self.optode_pos.label.values[i_opt]
                print(f"simulating fluence for {label}. {i_opt+1} / {n_optodes}")

                # run MCX or MCXCL
                # shape: [i,j,k]
                fluence = self._get_fluence_from_mcx(i_opt, **kwargs)

                # FIXME shortcut:
                # currently tissue props are wavelength independent -> copy
                for i_wl in range(n_wavelength):
                    # calculate fluence at all optode positions. used for normalization
                    fluence_at_optodes[i_opt, :, i_wl] = self._fluence_at_optodes(
                        fluence, i_opt
                    )

                    fluence_file.set_fluence_by_index(i_opt,i_wl, fluence)

            fluence_file.set_fluence_at_optodes(fluence_at_optodes)


    def compute_fluence_nirfaster(self, fluence_fname : str | Path, meshingparam=None):
        """Compute fluence for each channel and wavelength using NIRFASTer package.

        Args:
            fluence_fname : the output hdf5 file to store the fluence
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

        units = "1 / millimeter ** 2"

        fluence_at_optodes = xr.DataArray(
            dims=["optode1", "optode2", "wavelength"],
            coords={
                "optode1": self.optode_pos.label.values,
                "optode2": self.optode_pos.label.values,
                "wavelength": wavelengths,
            },
            attrs={"units": units},
        )

        with FluenceFile(fluence_fname, "w") as fluence_file:
            fluence_file.create_fluence_dataset(
                self.optode_pos,
                wavelengths,
                self.volume.shape,
                units
            )

            for i_wl in range(n_wavelength):
                # PLACEHOLDER: set new property and repeat
                # This way we can void the expensive meshing
                # newprop = []
                # mesh.set_prop(newprop)
                # newdata,_=femdata(0)
                for i_opt in range(n_optodes):
                    logger.debug(
                        f"computing wl {i_wl + 1}/{n_wavelength} "
                        f"optode {i_opt + 1} / {n_optodes}"
                    )
                    fluence = np.transpose(
                        data.phi[:, :, :, i_opt], (1, 0, 2)
                    )  # xyz to ijk

                    fluence_file.set_fluence_by_index(i_opt,i_wl, fluence)

                    fluence_at_optodes[i_opt, :, i_wl] = amplitude_optode[:,i_opt]

            fluence_file.set_fluence_at_optodes(fluence_at_optodes)

    def compute_sensitivity(
        self,
        fluence_fname: str | Path,
        sensitivity_fname: str | Path,
    ):
        """Compute sensitivity matrix from fluence.

        Args:
            fluence_fname : the input hdf5 file to store the fluence
            sensitivity_fname : the output netcdf file for the sensitivity
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

        # fluence_all: (label, wavelength, i, j, k)
        # fluence_at_optodes: (optode1, optode2, wavelength)

        with FluenceFile(fluence_fname, "r") as fluence_file:
            fluence_at_optodes = fluence_file.get_fluence_at_optodes()


            for _, r in self.measurement_list.iterrows():
                # using the adjoint monte carlo method
                # see YaoIntesFang2018 and BoasDale2005

                f_s = fluence_file.get_fluence(r.source, r.wavelength)
                f_d = fluence_file.get_fluence(r.detector, r.wavelength)

                pertubation = (f_s * f_d).flatten()

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

        Adot = xr.DataArray(
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

        save_Adot(sensitivity_fname, Adot)

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

    # perform the matrix multiplication on numpy arrays for speed
    inv_sens = inv_sens.transpose("flat_channel", "flat_vertex")
    od_stacked = od_stacked.transpose(..., "flat_channel")

    delta_conc = od_stacked.values @ inv_sens.values

    # repackage result as an DataArray
    delta_conc_dims = od_stacked.dims[:-1] + ("flat_vertex",)

    delta_conc = xr.DataArray(
        delta_conc,
        dims=delta_conc_dims,
        coords=(
            xrutils.coords_from_other(od_stacked, dims=delta_conc_dims)
            | xrutils.coords_from_other(inv_sens, dims=delta_conc_dims)
        ),
    )

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
