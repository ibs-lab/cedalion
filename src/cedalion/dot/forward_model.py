"""Forward model for simulating light transport in the head.

NOTE: Cedalion currently supports two ways to compute fluence:
1) via monte-carlo simulation using the MonteCarloXtreme (MCX) package, and
2) via the finite element method (FEM) using the NIRFASTer package.
While MCX is automatically installed using pip, NIRFASTER has to be manually installed
runnning <$ bash install_nirfaster.sh CPU # or GPU> from a within your cedalion root
directory.
"""

from __future__ import annotations
import logging
import os.path
import sys
from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd
import pint
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc

import cedalion.typing as cdt
import cedalion.xrutils as xrutils
import cedalion.io
from cedalion.dot.head_model import TwoSurfaceHeadModel

from cedalion.io.forward_model import FluenceFile, save_Adot

from .tissue_properties import get_tissue_properties

logger = logging.getLogger("cedalion")



class ForwardModel:
    """Forward model for simulating light transport in the head.

    ...

    Args:
    head_model (TwoSurfaceHeadModel): Head model containing voxel projections to brain
        and scalp surfaces.
    optode_pos (cdt.LabeledPoints): Optode positions.
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
        geo3d: cdt.LabeledPoints,
        measurement_list: pd.DataFrame,
    ):
        """Constructor for the forward model.

        Args:
            head_model (TwoSurfaceHeadModel): Head model containing voxel projections to
                brain and scalp surfaces.
            geo3d (cdt.LabeledPoints): Optode positions and directions.
            measurement_list (pd.DataFrame): List of measurements of experiment with
                source, detector, channel and wavelength.
        """

        assert head_model.crs == geo3d.points.crs

        # the forward model operates in voxel space. If the provided head model
        # is in scanner space, transform it back to voxel space.
        if head_model.crs != "ijk":
            head_model = head_model.apply_transform(head_model.t_ras2ijk)
            geo3d = geo3d.points.apply_transform(head_model.t_ras2ijk)

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
            self.measurement_list.wavelength.unique(),
        )

        self.volume = self.head_model.segmentation_masks.sum("segmentation_type")
        self.volume = self.volume.values.astype(np.uint8)
        self.unitinmm = self._get_unitinmm()

    def _get_voxel_dimensions(self) -> pint.Quantity:
        """Calculate the x,y and z voxel dimensions in scanner space.

        Returns:
            A quantified array [dx,dy,dz].
        """

        pts = cdc.build_labeled_points(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], crs="ijk", units="1"
        )
        pts_ras = pts.points.apply_transform(self.head_model.t_ijk2ras)
        return xrutils.norm(pts_ras[1:] - pts_ras[0], pts_ras.points.crs).data


    def _get_unitinmm(self):
        """Calculate length of volume grid cells.

        The forward model operates in ijk-space, in which each cell has unit length. To
        relate to physical distances pmcx needs the 'unitinmm' parameter.
        """

        # FIXME when scaling head models voxels may have different lengths in different
        # dimensions. MCX assumes cubical voxels with dimension unitinmm.
        # Report an edge length  that matches the voxel volume when cubed.
        voxel_dims = self._get_voxel_dimensions().to("mm").magnitude
        return np.prod(voxel_dims)**(1/3)


    def _get_fluence_from_mcx(self, i_optode: int, i_wl: int=0, **kwargs) -> np.ndarray:
        """Run MCX simulation to get fluence for one optode.

        Args:
            i_optode: Index of the optode.
            i_wl: Index of the wavelength.
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

        if self._get_unitinmm() != 1.:
            warn(
                "The current NIRFASTer implementation assumes a voxel volume of 1mm^3, "
                "but the voxel size of this head model is "
                + f"{self._get_voxel_dimensions().to('mm').magnitude} mm."
            )

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
        i_wl = 0 # FIXME
        for i in range(tissueprop.shape[0]):
            tissueprop[i,0] = i+1
            tissueprop[i,1] = self.tissue_properties[i+1, 0, i_wl]
            tissueprop[i,2] = self.tissue_properties[i+1, 1, i_wl] * (1-self.tissue_properties[i+1, 2, i_wl]) # noqa: E501
            tissueprop[i,3] = self.tissue_properties[i+1, 3, i_wl]

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
        """Compute sensitivity matrix from fluence via the adjoint Monte Carlo method.

        The sensitivity matrix (Jacobian) maps absorption changes in each voxel/vertex
        to changes in the measured optical density. It is computed using the adjoint
        Monte Carlo approach (:cite:t:`Boas2005`, :cite:t:`Yao2016`): the fluence
        from source and detector optodes is multiplied element-wise and projected onto
        the head surface.

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

                pertubation = (f_s * f_d).flatten() # shape (nvoxel,)

                normfactor = (
                    fluence_at_optodes.loc[r.source, r.detector, r.wavelength].values
                    + fluence_at_optodes.loc[r.detector, r.source, r.wavelength].values
                ) / 2

                i_wl = wavelengths.index(r.wavelength)
                i_ch = channels.index(r.channel)

                if normfactor > 0:
                    Adot_brain[i_ch, :, i_wl] = (
                        pertubation @ self.head_model.voxel_to_vertex_brain / normfactor
                    )
                    Adot_scalp[i_ch, :, i_wl] = (
                        pertubation @ self.head_model.voxel_to_vertex_scalp / normfactor
                    )
                else:
                    warn(
                        f"Observed zero fluence at optodes for channel {r.channel}. "
                        "Check the montage!"
                    )
                    Adot_brain[i_ch, :, i_wl] = 0.
                    Adot_scalp[i_ch, :, i_wl] = 0.

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

        Adot *= np.prod(self._get_voxel_dimensions().to("mm")).magnitude

        if self._get_unitinmm() != 1:
            warn("voxel size is not 1 mm^3. Check Adot normalization.")

        Adot = xr.DataArray(
            Adot.astype(np.float32),
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

        if "parcel" in self.head_model.brain.vertices.coords:
            parcels = np.concatenate(
                (
                    self.head_model.brain.vertices.coords["parcel"].values,
                    n_scalp * ["scalp"],
                )
            )
            Adot = Adot.assign_coords(parcel = ("vertex", parcels))

        save_Adot(sensitivity_fname, Adot)


    def compute_stacked_sensitivity(
        sensitivity: xr.DataArray, spectrum: str = "prahl"
    ) -> xr.DataArray:
        """Stack sensitivity matrices and incorporate extinction coefficients.

        For image reconstruction the 3D sensitivity arrays must be transformed into
        invertible 2D matrices and extinction coefficients must be incorporated.
        This function accepts sensitivities both in image space (Adot, vertices) or in
        spatial basis function space (H, kernels). The dimensions get transformed from
        (channel, vertex) to (flat_channel, flat_vertex) or (channel, kernel) to
        (flat_channel, flat_kernel), respectively.

        Args:
            sensitivity: Sensitivity matrix (Adot/H ) for each vertex/kernel and
                wavelength.
            spectrum: name of the extinction coefficient spectrum

        Returns:
            Stacked sensitivity matrix for each channel and vertex.
            shape = (flat_channel, flat_vertex)
        """

        assert "wavelength" in sensitivity.dims
        wavelengths = sensitivity.wavelength.values

        if "units" in sensitivity.attrs:
            units_sens = pint.Unit(sensitivity.attrs["units"])
        else:
            units_sens = pint.Unit("mm")

        ec = cedalion.nirs.get_extinction_coefficients(spectrum, wavelengths)

        units_ec = ec.pint.units
        ec = ec.pint.dequantify()

        units_A = units_sens * units_ec

        if "vertex" in sensitivity.dims:
            vertex_dim = "vertex"
            flat_vertex_dim = "flat_vertex"
            vertex_coords = sensitivity.vertex.values
        elif "kernel" in sensitivity.dims:
            vertex_dim = "kernel"
            flat_vertex_dim = "flat_kernel"
            vertex_coords = sensitivity.kernel.values
        else:
            raise ValueError("sensitivity must have vertex or kernel dimension.")


        nchannel = sensitivity.sizes["channel"]
        nvertices = sensitivity.sizes[vertex_dim]
        nwavelengths = len(wavelengths)
        chromos = ec.chromo.values
        nchromos = len(chromos)

        A = np.zeros((nwavelengths * nchannel, nchromos * nvertices))

        # fmt: off
        for i_wl, wl in enumerate(wavelengths):
            for i_ch, chromo in enumerate(chromos):
                c0 = i_wl * nchannel
                c1 = (i_wl + 1) * nchannel
                v0 = i_ch * nvertices
                v1 = (i_ch + 1) * nvertices

                A[c0:c1, v0:v1] = ec.sel(chromo=chromo, wavelength=wl).values * sensitivity.sel(wavelength=wl)  # noqa: E501
        # fmt: on

        is_brain = np.hstack([sensitivity.is_brain] * nchromos)
        flat_chromo = [ch for ch in chromos for _ in range(nvertices)]
        flat_wavelength = [wl for wl in wavelengths for _ in range(nchannel)]
        channel = sensitivity.channel.values
        source = sensitivity.source.values
        detector = sensitivity.detector.values
        flat_channel = np.hstack([channel] * nwavelengths)
        flat_source = np.hstack([source] * nwavelengths)
        flat_detector = np.hstack([detector] * nwavelengths)
        flat_vertex_coords = np.hstack([vertex_coords] * nchromos)

        coords = {
            "is_brain": (flat_vertex_dim, is_brain),
            "chromo": (flat_vertex_dim, flat_chromo),
            vertex_dim: (flat_vertex_dim, flat_vertex_coords),
            "wavelength": ("flat_channel", flat_wavelength),
            "channel": ("flat_channel", flat_channel),
            "source": ("flat_channel", flat_source),
            "detector": ("flat_channel", flat_detector),
        }

        if "parcel" in sensitivity.coords:
            flat_parcels = np.hstack([sensitivity.parcel.values] * nchromos)
            coords["parcel"] = (flat_vertex_dim, flat_parcels)

        A = xr.DataArray(
            A,
            dims=("flat_channel", flat_vertex_dim),
            coords=coords,
            attrs={"units": str(units_A)},
        )

        return A


    @staticmethod
    def parcel_sensitivity(
        Adot: xr.DataArray,
        chan_droplist: list = None,
        dOD_thresh: float = 0.001,
        minCh: int = 1,
        dHbO: float = 10,
        dHbR: float = -3,
    ):
        """Calculate a mask for parcels based on their effective cortex sensitivity.

        Parcels are considered good, if a change in HbO and HbR [µM] in the parcel leads
        to an observable change of at least dOD in at least one wavelength of one
        channel. Sensitivities of all vertices in the parcel are summed up in the
        sensitivity matrix Adot. Bad channels in an actual measurement that are pruned
        can be considered by providing a boolean channel_mask, where False indicates bad
        channels that are dropped and not considered for parcel sensitivity. Requires
        headmodel with parcelation coordinates.

        Args:
            Adot (channel, vertex, wavelength)): Sensitivity matrix with parcel
                coordinate belonging to each vertex
            chan_droplist: list of channel names to be dropped from consideration of
                sensitivity (e.g. pruned channels due to bad signal quality)
            dOD_thresh: threshold for minimum dOD change in a channel that should be
                observed from a hemodynamic change in a parcel
            minCh: minimum number of channels per parcel that should see a change above
                dOD_thresh
            dHbO: change in HbO conc. in the parcel in [µM] used to calculate dOD
            dHbR: change in HbR conc. in the parcel in [µM] used to calculate dOD

        Returns:
            A tuple (parcel_dOD, parcel_mask), where parcel_dOD (channel, parcel,
            wavelength) contains the delta OD observed in a channel for each wavelength
            given the assumed dHb change in a parcel, and parcel_mask is a boolean
            DataArray with parcel coords from Adot that is true for parcels for which
            dOD_thresh is met.

        Initial Contributors:
            - Alexander von Lühmann | vonluehmann@tu-berlin.de | 2025
        """

        # set up xarray with chromophore changes according to user input
        dHb = xr.DataArray(
            [dHbO*1e-6, dHbR*1e-6],
            dims=["chromo"],
            coords={"chromo": ["HbO", "HbR"]},
            attrs={"units": "M"},
            )
        dHb = dHb.pint.quantify()

        # calculate the constant nu/D where nu = c/n the speed of light in biological
        # tissue and D= 1/3(mu_a + mu_s') the photon diffusion coefficient
        # using constants from Wheelock et al 2019
        """ D = 1.03*100 * units("mm²/ns") # 1.03 cm²/ns
        nu = 21.4*10 * units("mm/ns" )# 21.4 cm/ns
        const = nu/D #/10 # convert to mm """
        const = 1

        # if chan_droplist is not None, set values in Adot to zero for all channels in
        # the list
        if chan_droplist is not None:
            Adot_mod = Adot.where(~Adot.channel.isin(chan_droplist), other=0)
        else:
            Adot_mod = Adot

        Adot_stacked = ForwardModel.compute_stacked_sensitivity(Adot_mod)

        # copies Adot and keeps only those vertices whose is_brain coordinate is true
        Adots_brain = Adot_stacked.sel(flat_vertex=Adot_stacked.coords['is_brain'])

        # index wavelength coordinate
        Adots_brain = Adots_brain.set_index(flat_channel='wavelength')
        # index chromo coordinate
        Adots_brain = Adots_brain.set_index(flat_vertex='chromo')

        # get unique wavelengths in wavelength coordinate
        wavelengths = Adots_brain.indexes['flat_channel'].unique()
        chromos = Adots_brain.indexes['flat_vertex'].unique()

        # Loop over both wavelengths and chromos, group vertices by parcels and multiply
        # by dHb change to get the dOD contribution for each channel and parcel per
        # wavelength
        dOD = {}
        for wl in wavelengths:
            for chromo in chromos:
                dOD[wl, chromo] = (
                    Adots_brain.sel(flat_channel=wl)
                    .sel(flat_vertex=chromo)
                    .groupby("parcel")
                    .sum("flat_vertex")
                    * dHb.sel(chromo=chromo)
                )

        coords = {
            "channel": ("channel", Adot.coords["channel"].values),
            "parcel": (
                "parcel",
                dOD[wavelengths[0], chromos[0]].coords["parcel"].values,
            ),
        }

        # sum values in dOD across chromophores to get the total dOD for each parcel and
        # channel per wavelength
        dOD_tot = {}
        for wl in wavelengths:
            dOD_tot[wl] = xr.DataArray(
                dOD[wl, chromos[0]].pint.magnitude + dOD[wl, chromos[1]].pint.magnitude,
                dims=["channel", "parcel"],
                coords=coords,
            )

        # Combine into a single dataarray with a wavelength coordinate
        parcel_dOD = xr.concat(
            [dOD_tot[wl] for wl in wavelengths],
            dim=pd.Index(wavelengths.values, name="wavelength")
        )

        # multiply with constant # FIXME: check the units a last time
        parcel_dOD = parcel_dOD * const

        # calculate mask
        parcel_mask = xrutils.mask(parcel_dOD, True)
        # check where dOD is greater than dOD_thresh
        parcel_mask = parcel_mask.where(parcel_dOD.values >= dOD_thresh, other = False)
        # check whether threshold is passed for either wavelengths
        parcel_mask = parcel_mask.sum("wavelength") >= 1
        # check whether threshold is passed for the minimum number of channels
        parcel_mask = parcel_mask.sum("channel") >= minCh


        return parcel_dOD, parcel_mask


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
    """Stack ``chromo`` and ``vertex`` dimensions into a single ``flat_vertex`` dim.

    Args:
        array: DataArray with ``"chromo"`` and ``"vertex"`` dimensions.

    Returns:
        DataArray with a new stacked ``"flat_vertex"`` dimension.

    Raises:
        ValueError: If ``array`` is missing either the ``"chromo"`` or
            ``"vertex"`` dimension.
    """
    dims = ("chromo", "vertex")

    for dim in dims:
        if dim not in array.dims:
            raise ValueError(f"cannot stack missing dimension {dim}")

    return array.stack({"flat_vertex": dims})


def unstack_flat_vertex(array: xr.DataArray):
    """Unstack the ``flat_vertex`` dimension back into ``chromo`` and ``vertex``.

    Args:
        array: DataArray with a ``"flat_vertex"`` multi-index dimension.

    Returns:
        DataArray with separate ``"chromo"`` and ``"vertex"`` dimensions.
    """
    return xrutils.unstack(array, "flat_vertex", ("chromo", "vertex"))


def stack_flat_channel(array: xr.DataArray):
    """Stack ``wavelength`` and ``channel`` dims into a single ``flat_channel`` dim.

    Args:
        array: DataArray with ``"wavelength"`` and ``"channel"`` dimensions.

    Returns:
        DataArray with a new stacked ``"flat_channel"`` dimension.

    Raises:
        ValueError: If ``array`` is missing either the ``"wavelength"`` or
            ``"channel"`` dimension.
    """
    dims = ("wavelength", "channel")

    for dim in dims:
        if dim not in array.dims:
            raise ValueError(f"cannot stack missing dimension {dim}")

    return array.stack({"flat_channel": dims})


def unstack_flat_channel(array: xr.DataArray):
    """Unstack the ``flat_channel`` dimension back into ``wavelength`` and ``channel``.

    Args:
        array: DataArray with a ``"flat_channel"`` multi-index dimension.

    Returns:
        DataArray with separate ``"wavelength"`` and ``"channel"`` dimensions.
    """
    return xrutils.unstack(array, "flat_channel", ("wavelength", "channel"))


def image_to_channel_space(
    Adot: xr.DataArray, img: xr.DataArray, spectrum: str | None = None
):
    """Project an image-space quantity into channel (measurement) space via Adot.

    Performs the forward projection ``y = Adot @ img`` where the shared vertex
    dimension is contracted.  If ``img`` is in concentration units, the
    chromophore-stacked sensitivity matrix is used; if it is in absorption
    units (1/length), the raw Adot is used directly.

    Args:
        Adot: Sensitivity matrix with a ``"vertex"`` dimension.
        img: Image DataArray with a ``"vertex"`` dimension.  Must be quantified
            in either concentration (``"[concentration]"``) or absorption
            (``"[1/length]"``) units.
        spectrum: Extinction coefficient spectrum (e.g. ``"prahl"``). Required
            when ``img`` has concentration units.

    Returns:
        DataArray in channel space with ``"wavelength"`` and ``"channel"``
        dimensions.

    Raises:
        ValueError: If ``img`` has incompatible units or ``spectrum`` is ``None``
            when concentration units are detected.
    """

    common_dim = "vertex"
    assert (common_dim in Adot.dims) and (common_dim in img.dims)

    if xrutils.check_units(img, "[concentration]"):
        if spectrum is None:
            raise ValueError("You must specify 'spectrum' if img is a concentration.")

        img = img.pint.quantify()

        Adot_stacked = ForwardModel.compute_stacked_sensitivity(Adot, spectrum)
        Adot_stacked = Adot_stacked.pint.quantify()
        img_stacked = stack_flat_vertex(img)

        return unstack_flat_channel(
            xrutils.contract(
                Adot_stacked, img_stacked, dim="flat_vertex"
            )  # FIXME generalize?
        )
    elif xrutils.check_units(img, "1/[length]"):
        Adot = Adot.pint.quantify()
        img = img.pint.quantify()

        return xrutils.contract(Adot, img, dim=common_dim)
    else:
        raise ValueError("img must be a quantified concentration ")
