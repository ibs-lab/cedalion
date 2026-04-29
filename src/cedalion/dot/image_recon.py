"""Solver for the image reconstruction problem."""

import hashlib
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import pint
import scipy.stats
import xarray as xr
from scipy.sparse import csr_array
from scipy.spatial import KDTree
from tqdm import tqdm

import cedalion.dot.forward_model as fwm
import cedalion.io.utils as ioutils
import cedalion.typing as cdt
import cedalion.utils
import cedalion.xrutils as xrutils
from cedalion import nirs, units
from cedalion.dot.head_model import TwoSurfaceHeadModel

logger = logging.getLogger("cedalion")

ReconMode = Literal["conc", "mua", "mua2conc"]

# predefined parameter sets

# we could define constants of parameters that work well together, e.g. based on
# Laura's parameter scans:
REG_TIKHONOV_ONLY = dict(
    alpha_meas=0.001,
    alpha_spatial=None,
    apply_c_meas=False,
)

REG_PAPER_MUA_SBF = dict(
    alpha_meas=1e4,
    alpha_spatial=1e-2,
    apply_c_meas=True,
    lambda_R_conc=1e-6
)
"""Optimal set of regularization parameters according to an optimization study for a
ball squeezing dataset. :cite:t:`Carlton2026`.
"""

SBF_GAUSSIANS_DENSE = dict(
    mask_threshold=-2,
    threshold_brain=1 * units.mm,
    threshold_scalp=5 * units.mm,
    sigma_brain=1 * units.mm,
    sigma_scalp=5 * units.mm,
)
"""Optimal set of Gaussian SBF parameters according to an optimization study for a
ball squeezing dataset. :cite:t:`Carlton2026`.
"""

SBF_GAUSSIANS_SPARSE = dict(
    mask_threshold=-2,
    threshold_brain=5 * units.mm,
    threshold_scalp=20 * units.mm,
    sigma_brain=5 * units.mm,
    sigma_scalp=20 * units.mm,
)
"""A sparse set of gaussians SBFs."""


def estimate_alpha_meas(C_meas, K=0.01):
    """Implements a heuristic for choosing alpha_meas.

    The strength of the regularization is determined by the relative scale of the C and
    R regularization matrices, which is encoded in the ratio:

        K = median(eig( lambda_meas C )) / max(eig( lambda_R A R A'))

    In past analyses K was about 0.01 .. 0.1. With this a heuristic for choosing
    alpha_meas can be formed:

        alpha_meas = K / median(eig(C_meas))

    Args:
        C_meas: diagonal values of C_meas matrix
        K : relative scale of C and R regularization matrices
    """

    return K / np.median(C_meas)


class SpatialBasisFunctions(ABC):
    """Base for SBF implementations."""

    @property
    @abstractmethod
    def H(self) -> xr.DataArray:
        """The sensitivity in kernel space."""
        pass

    @abstractmethod
    def kernel_to_image_space_conc(self, conc_img: xr.DataArray) -> xr.DataArray:
        """Transform an image from kernel to image space in recon. mode 'conc'."""
        pass

    @abstractmethod
    def kernel_to_image_space_mua(self, mua_img: xr.DataArray) -> xr.DataArray:
        """Transform an image from kernel to image space in recon. mode 'mua'."""

    @abstractmethod
    def to_file(self, fname : Path | str):
        """Serialize prepared spatial basis functions into a file.

        Args:
            fname: path of the output file.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, fname : Path | str) -> "SpatialBasisFunctions":
        """Load prepared spatial basis functions from a file.

        Args:
            fname: path of the file to read from.

        Returns:
            SpatialBasisFunctions: Loaded spatial basis functions instance.
        """
        pass


class OriginalGaussianSpatialBasisFunctions:
    def __init__(
        self,
        head_model: TwoSurfaceHeadModel,
        Adot: xr.DataArray,
        threshold_brain: cdt.QLength,
        threshold_scalp: cdt.QLength,
        sigma_brain: cdt.QLength,
        sigma_scalp: cdt.QLength,
        mask_threshold: float,
    ):
        """Gaussian Spatial Basis Functions for DOT image reconstruction.

        Represents the unknown absorption image as a weighted sum of Gaussian
        kernels placed on the head surface, reducing the ill-posed inverse problem
        to a lower-dimensional one. Optimal parameter sets for this implementation
        are given in :cite:t:`Carlton2026`.

        Args:
            head_model: TwoSurfaceHeadModel providing brain and scalp surfaces.
            Adot: Sensitivity matrix (channel × vertex × wavelength).
            threshold_brain: Maximum distance from a sensitivity vertex for a brain
                kernel centre to be included (log-scale units).
            threshold_scalp: Maximum distance from a sensitivity vertex for a scalp
                kernel centre to be included (log-scale units).
            sigma_brain: Spatial width of brain Gaussian kernels.
            sigma_scalp: Spatial width of scalp Gaussian kernels.
            mask_threshold: Log-sensitivity threshold; vertices below this value are
                excluded from the kernel support.
        """

        cedalion.utils.deprecated_api(
            "This implementation of gaussian basis functions will be replaced."
        )

        self.threshold_brain = threshold_brain
        self.threshold_scalp = threshold_scalp
        self.sigma_brain = sigma_brain
        self.sigma_scalp = sigma_scalp
        self.mask_threshold = mask_threshold

        self._mask : xr.DataArray = None # shape (vertex)
        self.G_brain : xr.DataArray = None # shape (vertex, kernel)
        self.G_scalp : xr.DataArray = None  # shape (vertex, kernel)
        self.H: xr.DataArray = None  # shape(channel, kernel, wavelength)

        # compute _G
        self._compute_sensitivity_mask(Adot)
        self._compute_G_gaussian_kernels(head_model)
        self._compute_H(Adot)


    def _compute_sensitivity_mask(self, Adot, wavelength_idx: int = 0):
        """Compute sensitivity mask based on intensity threshold.

        The mask selects those vertices whose summed contribution to the sensitivity
        of all channels is above 10^(mask_threshold).

        Args:
            Adot: Sensitivity matrix.
            wavelength_idx: Index of wavelength to use for mask computation.
        """

        intensity = np.log10(
            Adot.isel(wavelength=wavelength_idx)
            .sum("channel")
            .clip(min=10 ** (self.mask_threshold - 1))  # avoid log10(0)
        )

        mask = intensity > self.mask_threshold
        mask = mask.drop_vars("wavelength")  # keep the is_brain coordinate
        self._mask = mask

    def _downsample_mesh(
        self, mesh: xr.DataArray, threshold: cdt.QLength, mask: np.ndarray
    ) -> xr.DataArray:
        """Downsample the mesh to get seeds of spatial bases.

        Args:
            mesh: vertices of either the brain or scalp surface.
            threshold: distance between vertices in downsampled mesh.
            mask: boolean mask to select mesh vertices

        Returns:
            downsampled mesh vertices as a xr.DataArray

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        # Downsample the mesh using the specified method
        mesh_units = mesh.pint.units
        threshold = threshold.to(mesh_units).magnitude

        mesh = mesh.rename({"label": "vertex"}).pint.dequantify()
        mesh_masked = mesh[mask, :]
        mesh_new = []

        for vv in tqdm(mesh_masked):
            if len(mesh_new) == 0:
                mesh_new.append(vv)
                tree = KDTree(mesh_new)  # Build KDTree for the first point
                continue

            # Query the nearest neighbor within the threshold
            distance, _ = tree.query(vv, distance_upper_bound=threshold)

            # If no point is within the threshold, append the new point
            if distance == float("inf"):
                mesh_new.append(vv)
                tree = KDTree(mesh_new)  # Rebuild the KDTree with the new point

        mesh_new_xr = xr.DataArray(
            mesh_new,
            dims=mesh.dims,
            coords={"vertex": np.arange(len(mesh_new))},
            attrs={"units": mesh_units},
        )

        mesh_new_xr = mesh_new_xr.pint.quantify()

        return mesh_new_xr


    def _get_gaussian_kernels_on_mesh(
        self, mesh_downsampled: xr.DataArray, mesh: xr.DataArray, sigma: cdt.QLength
    ):
        """Compute the matrix containing the spatial bases.

        Args:
            mesh_downsampled: vertices of either the downsampled brain or scalp surface.
                This is used to define the centers of the spatial bases.
            mesh: the original fully sampeld mesh vertices of the brain or scalp.
            sigma: standard deviation used for defining the Gaussian kernel.

        Returns:
            xr.DataArray: matrix containing the spatial bases

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        # Create Gaussian kernels based on the mesh and parameters
        assert mesh.pint.units == mesh_downsampled.pint.units

        mesh_units = mesh.pint.units
        sigma = sigma.to(mesh_units).magnitude

        # Covariance matrix
        cov_matrix = (sigma**2) * np.eye(3)
        inv_cov = np.linalg.inv(cov_matrix)  # Inverse of Cov_matrix
        det_cov = np.linalg.det(cov_matrix)  # Determinant of Cov_matrix
        denominator = np.sqrt((2 * np.pi) ** 3 * det_cov)  # Pre-calculate denominator

        mesh_downsampled = mesh_downsampled.pint.dequantify().values
        mesh = mesh.pint.dequantify().values

        diffs = mesh_downsampled[:, None, :] - mesh[None, :, :]

        # Efficient matrix multiplication using np.einsum to compute
        # (x-mu)' * inv_cov * (x-mu) for all pairs
        exponents = -0.5 * np.einsum("ijk,kl,ijl->ij", diffs, inv_cov, diffs)

        # Compute the kernel matrix
        kernel_matrix = np.exp(exponents) / denominator
        n_vertex = mesh.shape[0]

        dimensions = kernel_matrix.shape

        if dimensions[0] != n_vertex:
            dims = ["kernel", "vertex"]
            #n_kernel = dimensions[0]
        else:
            dims = ["vertex", "kernel"]
            #n_kernel = dimensions[1]

        kernel_matrix_xr = xr.DataArray(
            kernel_matrix,
            dims=dims,
            #coords={"vertex": np.arange(n_vertex), "kernel": np.arange(n_kernel)},
        )

        kernel_matrix_xr = kernel_matrix_xr.transpose("vertex", "kernel")
        return kernel_matrix_xr


    def _compute_G_gaussian_kernels(self, head_model : TwoSurfaceHeadModel):
        """Compute the G matrix which contains all the information of the spatial basis.

        Args:
            head_model: Head model with brain and scalp surfaces.

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024
        """
        brain_downsampled = self._downsample_mesh(
            head_model.brain.vertices,
            self.threshold_brain,
            self._mask.sel(vertex=self._mask.is_brain),
        )
        scalp_downsampled = self._downsample_mesh(
            head_model.scalp.vertices,
            self.threshold_scalp,
            self._mask.sel(vertex=~self._mask.is_brain),
        )

        self.G_brain = self._get_gaussian_kernels_on_mesh(
            brain_downsampled, head_model.brain.vertices, self.sigma_brain
        )
        self.G_scalp = self._get_gaussian_kernels_on_mesh(
            scalp_downsampled, head_model.scalp.vertices, self.sigma_scalp
        )

        vertices = np.arange(head_model.brain.nvertices + head_model.scalp.nvertices)
        kernel = np.arange(self.G_brain.sizes["kernel"] + self.G_scalp.sizes["kernel"])

        self.G_brain = self.G_brain.assign_coords(
            {
                "vertex": vertices[: head_model.brain.nvertices],
                "kernel": kernel[: self.G_brain.sizes["kernel"]],
            }
        )

        self.G_scalp = self.G_scalp.assign_coords(
            {
                "vertex": vertices[head_model.brain.nvertices:],
                "kernel": kernel[self.G_brain.sizes["kernel"]:],
            }
        )


    def _compute_H(self, Adot : xr.DataArray):
        """Compute the H matrix for spatial basis functions.

        Transforms the sensitivity matrix into the spatial basis space.

        Args:
            Adot: Sensitivity matrix shape=(channel, vertex, wavelength)
        """
        assert Adot.dims == ("channel", "vertex", "wavelength")

        n_channel = len(Adot.channel)
        n_wavelength = len(Adot.wavelength)

        # number of kernels
        n_k_brain = self.G_brain.sizes["kernel"]
        n_k = self.G_brain.sizes["kernel"] + self.G_scalp.sizes["kernel"]

        H = np.zeros((n_channel, n_k, n_wavelength))
        for w in range(n_wavelength):
            Adot_w = Adot.isel(wavelength=w).values
            H[:, :n_k_brain, w] = Adot_w[:, Adot.is_brain] @ self.G_brain.values
            H[:, n_k_brain:, w] = Adot_w[:, ~Adot.is_brain] @ self.G_scalp.values

        is_brain = np.ones(n_k, dtype=np.bool_)
        is_brain[n_k_brain:] = False

        H = xr.DataArray(H, dims=("channel", "kernel", "wavelength"))

        H = H.assign_coords(
            {
                "channel": Adot.channel,
                "wavelength": Adot.wavelength,
                "kernel": np.concatenate(
                    [self.G_brain.kernel.values, self.G_scalp.kernel.values]
                ),
                "is_brain": ("kernel", is_brain),
            }
        )
        self.H = H


    def kernel_to_image_space_mua(self, X : np.ndarray) -> np.ndarray:
        """Convert kernel space reconstructions to image space for mua.

        Args:
            X: Reconstruction values in kernel space. shape (kernel, ...)

        Returns:
            np.ndarray: Reconstruction values in image space.
        """

        nkernels_brain = self.G_brain.sizes["kernel"]
        has_scalp = X.sizes["kernel"] > nkernels_brain

        sb_X_brain = X[{"kernel":slice(0,nkernels_brain)}]

        X_brain = xrutils.contract(self.G_brain, sb_X_brain, dim="kernel")

        if has_scalp:
            sb_X_scalp = X[{"kernel": slice(nkernels_brain, None)}]
            X_scalp = xrutils.contract(self.G_scalp, sb_X_scalp, dim="kernel")

        if has_scalp:
            is_brain = np.zeros(
                X_brain.sizes["vertex"] + X_scalp.sizes["vertex"], dtype=bool
            )
            is_brain[:X_brain.sizes["vertex"]] = True
            X = xr.concat([X_brain, X_scalp], dim="vertex").assign_coords(
                {"is_brain": ("vertex", is_brain)}
            )
        else:
            is_brain = np.ones(X_brain.sizes["vertex"], dtype=bool)
            X = X_brain.assign_coords({"is_brain": ("vertex", is_brain)})

        return X



    def kernel_to_image_space_conc(self, X) -> np.ndarray:
        """Convert kernel space reconstructions to image space for concentration.

        Args:
            X: Reconstruction values in kernel space.

        Returns:
            np.ndarray: Reconstruction values in image space with HbO/HbR split.
        """

        assert "flat_kernel" in X.dims


        X = xrutils.unstack(X, "flat_kernel", ("chromo", "kernel"))

        # FIXME limited to two chromophores
        #split = X.shape[0]//2
        nkernels_brain = self.G_brain.sizes["kernel"]
        has_scalp = X.sizes["kernel"] > nkernels_brain

        sb_X_brain_hbo = X[{"kernel":slice(0,nkernels_brain)}].loc[{"chromo" : "HbO"}]
        sb_X_brain_hbr = X[{"kernel":slice(0,nkernels_brain)}].loc[{"chromo" : "HbR"}]

        X_hbo_brain = xrutils.contract(self.G_brain, sb_X_brain_hbo, dim="kernel")
        X_hbr_brain = xrutils.contract(self.G_brain, sb_X_brain_hbr, dim="kernel")

        X_brain = xr.concat(
            [X_hbo_brain, X_hbr_brain], dim="chromo", coords={"chromo": ["HbO", "HbR"]}
        )

        if has_scalp:
            sb_X_scalp_hbo = X[{"kernel": slice(nkernels_brain, None)}].loc[
                {"chromo": "HbO"}
            ]
            sb_X_scalp_hbr = X[{"kernel": slice(nkernels_brain, None)}].loc[
                {"chromo": "HbR"}
            ]

            X_hbo_scalp = xrutils.contract(self.G_scalp, sb_X_scalp_hbo, dim="kernel")
            X_hbr_scalp = xrutils.contract(self.G_scalp, sb_X_scalp_hbr, dim="kernel")

            X_scalp = xr.concat([X_hbo_scalp, X_hbr_scalp], dim="chromo")
            X_scalp = X_scalp.assign_coords({"chromo": ["HbO", "HbR"]})
        if has_scalp:
            is_brain = np.zeros(
                X_brain.sizes["vertex"] + X_scalp.sizes["vertex"], dtype=bool
            )
            is_brain[:X_brain.sizes["vertex"]] = True
            X = xr.concat([X_brain, X_scalp], dim="vertex").assign_coords(
                {"is_brain": ("vertex", is_brain)}
            )
        else:
            is_brain = np.ones(X_brain.sizes["vertex"], dtype=bool)
            X = X_brain.assign_coords({"is_brain": ("vertex", is_brain)})

        X = X.transpose("chromo", "vertex", ...)
        return X


    def to_file(self, fname : Path | str):
        """Serialize prepared Gaussian spatial basis functions to HDF5 file.

        Args:
            fname: path of the output file.
        """

        with h5py.File(fname, "w") as fout:
            ioutils.xarray_to_hdfgroup(fout, self.H, "H")
            ioutils.xarray_to_hdfgroup(fout, self.G_brain, "G_brain")
            ioutils.xarray_to_hdfgroup(fout, self.G_scalp, "G_scalp")
            ioutils.xarray_to_hdfgroup(fout, self._mask, "_mask")

            for name in [
                "threshold_brain",
                "threshold_scalp",
                "sigma_brain",
                "sigma_scalp",
                "mask_threshold",
            ]:
                fout["/"].attrs[name] = str(getattr(self, name))



    @classmethod
    def from_file(cls, fname : Path | str) -> "OriginalGaussianSpatialBasisFunctions":
        """Load prepared Gaussian spatial basis functions from HDF5 group.

        Args:
            fname: path of the file to read from.

        Returns:
            GaussianSpatialBasisFunctions: Loaded instance.
        """

        sbf = cls.__new__(cls)
        with h5py.File(fname, "r") as f:
            sbf.H = ioutils.xarray_from_hdfgroup(f, "H")
            sbf.G_brain = ioutils.xarray_from_hdfgroup(f, "G_brain")
            sbf.G_scalp = ioutils.xarray_from_hdfgroup(f, "G_scalp")
            sbf._mask = ioutils.xarray_from_hdfgroup(f, "_mask")

            for name in [
                "threshold_brain",
                "threshold_scalp",
                "sigma_brain",
                "sigma_scalp",
            ]:
                setattr(sbf, name, pint.Quantity(f["/"].attrs[name]))

            setattr(sbf, "mask_threshold", float(f["/"].attrs["mask_threshold"]))

        return sbf

class GaussianSpatialBasisFunctions(SpatialBasisFunctions):
    """Gaussian Spatial Basis Functions.

    Args:
        head_model: a TwoSurfaceHeadModel with brain and scalp surfaces
        Adot : the sensitivity matrix
        threshold_brain: the distance between kernel centers on the brain
        threshold_scalp: the distance between kernel centers on scalp
        sigma_brain: the width of the gaussians on the brain
        sigma_scalp : the width of the gaussians on the scalp
        mask_threshold: log10(sensitivity) threshold for vertices to be considered
        verbose: controls visibility of status messages and progress bar
    """

    def __init__(
        self,
        head_model: TwoSurfaceHeadModel,
        Adot: xr.DataArray,
        threshold_brain: cdt.QLength,
        threshold_scalp: cdt.QLength,
        sigma_brain: cdt.QLength,
        sigma_scalp: cdt.QLength,
        mask_threshold: float,
        verbose: bool = True,
    ):
        self.threshold_brain = threshold_brain
        self.threshold_scalp = threshold_scalp
        self.sigma_brain = sigma_brain
        self.sigma_scalp = sigma_scalp
        self.mask_threshold = mask_threshold
        self.verbose = verbose

        self._mask: xr.DataArray = None  # shape (vertex)

        # H integrates Adot, i.e. it describes each kernel's influence on each channel
        self._H: xr.DataArray = None  # shape(channel, kernel, wavelength)

        # G contains the kernel's value for each vertex
        self._G: csr_array = None  # shape (kernel, vertex)

        # coordinate arrays of G. Have to keep them separate since G is not a DataArray
        self._G_kernel : np.ndarray = None
        self._G_kernel_is_brain: np.ndarray = None
        self._G_vertex_is_brain: np.ndarray = None
        self._G_vertex_parcel: np.ndarray = None

        self.nkernel_brain : int = None
        self.nvertices_brain : int = None

        # compute _G
        self._compute_sensitivity_mask(Adot)
        self._compute_G_gaussian_kernels(head_model)
        self._compute_H(Adot)

    @property
    def H(self):
        return self._H


    def _compute_sensitivity_mask(self, Adot, wavelength_idx: int = 0):
        """Compute sensitivity mask based on intensity threshold.

        The mask selects those vertices whose summed contribution to the sensitivity
        of all channels is above 10^(mask_threshold).

        Args:
            Adot: Sensitivity matrix.
            wavelength_idx: Index of wavelength to use for mask computation.
        """

        intensity = np.log10(
            Adot.isel(wavelength=wavelength_idx) # FIXME maybe min over wavelengths?
            .sum("channel")
            .clip(min=10 ** (self.mask_threshold - 1))  # avoid log10(0)
        )

        mask = intensity > self.mask_threshold
        mask = mask.drop_vars("wavelength")  # but keep the is_brain coordinate
        self._mask = mask


    def _downsample_mesh(
        self, mesh: xr.DataArray, threshold: cdt.QLength, mask: np.ndarray
    ) -> xr.DataArray:
        """Downsample the mesh to get seeds of spatial bases.

        Args:
            mesh: vertices of either the brain or scalp surface.
            threshold: distance between vertices in downsampled mesh.
            mask: boolean mask to select mesh vertices

        Returns:
            downsampled mesh vertices as a xr.DataArray

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        # Downsample the mesh using the specified method
        mesh_units = mesh.pint.units
        threshold = threshold.to(mesh_units).magnitude

        mesh = mesh.rename({"label": "vertex"}).pint.dequantify()
        mesh_masked = mesh[mask, :].values
        nmasked = mesh_masked.shape[0]

        sel_indices = [0]

        for vertex_index in tqdm(np.arange(1, nmasked), disable=not self.verbose):
            vv = mesh_masked[vertex_index, :]

            dists = np.linalg.norm(mesh_masked[sel_indices, :] - vv[None, :], axis=1)
            if not np.any(dists <= threshold):
                sel_indices.append(vertex_index)

        mesh_new = mesh_masked[sel_indices]

        mesh_new_xr = xr.DataArray(
            mesh_new,
            dims=mesh.dims,
            coords={"vertex": np.arange(len(mesh_new))},
            attrs={"units": mesh_units},
        )

        mesh_new_xr = mesh_new_xr.pint.quantify()

        return mesh_new_xr


    def _get_gaussian_kernels_on_mesh(
        self,
        mesh_downsampled: xr.DataArray,
        mesh: xr.DataArray,
        sigma: cdt.QLength,
        vertex_indices: np.ndarray,
        G_shape : tuple[int,int]
    ):
        """Compute the matrix containing the spatial bases.

        Args:
            mesh_downsampled: vertices of either the downsampled brain or scalp surface.
                This is used to define the centers of the spatial bases.
            mesh: the original fully sampeld mesh vertices of the brain or scalp.
            sigma: standard deviation used for defining the Gaussian kernel.
            vertex_indices: The column indices in the resulting sparse G matrix.
            G_shape: the shape of the resulting G matrix

        Returns:
            xr.DataArray: matrix containing the spatial bases

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        # Create Gaussian kernels based on the mesh and parameters
        assert mesh.pint.units == mesh_downsampled.pint.units

        mesh_units = mesh.pint.units
        sigma = sigma.to(mesh_units).magnitude

        mesh_downsampled = mesh_downsampled.pint.dequantify().values
        mesh = mesh.pint.dequantify().values

        n_kernel = len(mesh_downsampled)
        #n_vertex = len(mesh)

        norm_pdf = scipy.stats.norm(scale=sigma).pdf

        # csr data structures
        csr_data = []
        csr_indices = []
        csr_indptr = [0]
        csr_ndata = 0

        for i_kernel in tqdm(np.arange(n_kernel), disable=not self.verbose):
            dists = np.linalg.norm(mesh_downsampled[[i_kernel],:] - mesh, axis=1)
            kernel_values = norm_pdf(dists)

            # change kernel normalization to match original implementation
            kernel_values /= (2 * np.pi * sigma**2)

            indices : np.ndarray = np.flatnonzero(kernel_values >= 1e-16)
            csr_indices.append( vertex_indices[indices] )
            csr_data.append(kernel_values[indices])
            csr_ndata += len(indices)
            csr_indptr.append(csr_ndata)

        csr_indices = np.hstack(csr_indices)
        csr_data = np.hstack(csr_data)

        kernel_matrix = csr_array(
            (csr_data, csr_indices, csr_indptr), shape=(n_kernel, G_shape[1])
        )

        return kernel_matrix


    def _compute_G_gaussian_kernels(self, head_model : TwoSurfaceHeadModel):
        """Compute the G matrix which contains all the information of the spatial basis.

        Args:
            head_model: Head model with brain and scalp surfaces.

        Initial Contributors:
            - Yuanyuan Gao
            - Laura Carlton | lcarlton@bu.edu | 2024
        """

        # downsample brain and scalp meshes. These vertices will become the centers
        # of the spatial basis funct
        brain_downsampled = self._downsample_mesh(
            head_model.brain.vertices,
            self.threshold_brain,
            self._mask.sel(vertex=self._mask.is_brain),
        )
        scalp_downsampled = self._downsample_mesh(
            head_model.scalp.vertices,
            self.threshold_scalp,
            self._mask.sel(vertex=~self._mask.is_brain),
        )

        # unique vertex indices of the brain and scalp surfaces
        vidx_brain = np.arange(head_model.brain.nvertices)
        vidx_scalp = np.arange(
            head_model.brain.nvertices,
            head_model.brain.nvertices + head_model.scalp.nvertices,
        )
        n_kernel = len(brain_downsampled) + len(scalp_downsampled)
        n_vertex = head_model.brain.nvertices + head_model.scalp.nvertices

        self.nkernel_brain = len(brain_downsampled)
        self.nvertices_brain = head_model.brain.nvertices

        G_shape = (n_kernel, n_vertex)

        G_brain = self._get_gaussian_kernels_on_mesh(
            brain_downsampled,
            head_model.brain.vertices,
            self.sigma_brain,
            vidx_brain,
            G_shape,
        )

        G_scalp = self._get_gaussian_kernels_on_mesh(
            scalp_downsampled,
            head_model.scalp.vertices,
            self.sigma_scalp,
            vidx_scalp,
            G_shape,
        )

        self._G = scipy.sparse.vstack((G_brain, G_scalp))

        self._G_kernel = np.arange(n_kernel)
        self._G_kernel_is_brain = np.zeros(n_kernel, dtype=bool)
        self._G_kernel_is_brain[:self.nkernel_brain] = True
        self._G_vertex_is_brain = np.zeros(n_vertex, dtype=bool)
        self._G_vertex_is_brain[:self.nvertices_brain] = True

        if "parcel" in head_model.brain.vertex_coords:
            self._G_vertex_parcel = np.hstack(
                (
                    head_model.brain.vertex_coords["parcel"],
                    [None] * head_model.scalp.nvertices,
                )
            )


    def _compute_H(self, Adot : xr.DataArray):
        """Compute the H matrix for spatial basis functions.

        Transforms the sensitivity matrix into the spatial basis space.

        Args:
            Adot: Sensitivity matrix shape=(channel, vertex, wavelength)
        """
        assert Adot.dims == ("channel", "vertex", "wavelength")

        H = xrutils.dot_dataarray_csr(Adot, self._G, ["kernel", "vertex"])

        H = H.assign_coords(
            {
                "kernel": self._G_kernel,
                "is_brain": ("kernel", self._G_kernel_is_brain),
            }
        )

        self._H = H


    def kernel_to_image_space_mua(self, X : np.ndarray) -> np.ndarray:
        """Convert kernel space reconstructions to image space for mua.

        Args:
            X: Reconstruction values in kernel space. shape (kernel, ...)

        Returns:
            np.ndarray: Reconstruction values in image space.
        """

        n_kernels = len(self._G_kernel)
        n_kernels_brain = self._G_kernel_is_brain.sum()

        coords = {}

        if X.sizes["kernel"] == n_kernels:
            img = xrutils.dot_dataarray_csr(X, self._G, ["kernel", "vertex"])

            coords["is_brain"] = ("vertex", self._G_vertex_is_brain)
            if self._G_vertex_parcel is not None:
                coords["parcel"] = ("vertex", self._G_vertex_parcel)
        elif X.sizes["kernel"] == n_kernels_brain:  # brain_only == True
            img = xrutils.dot_dataarray_csr(
                X, self._G[:n_kernels_brain, :], ["kernel", "vertex"]
            )
            # FIXME even if only kernels on the brain are provided in X, G contains
            # vertices of the scalp which we have to select away afterwards.
            # It would be more efficient if these vertices would be cut from G.
            img = img.sel(vertex=self._G_vertex_is_brain)

            coords["is_brain"] = ("vertex", np.ones(img.sizes["vertex"], dtype=bool))
            if self._G_vertex_parcel is not None:
                coords["parcel"] = (
                    "vertex",
                    self._G_vertex_parcel[self._G_vertex_is_brain],
                )

        img = img.assign_coords(coords)

        return img


    def kernel_to_image_space_conc(self, X) -> np.ndarray:
        """Convert kernel space reconstructions to image space for concentration.

        Args:
            X: Reconstruction values in kernel space.

        Returns:
            np.ndarray: Reconstruction values in image space with HbO/HbR split.
        """

        assert "flat_kernel" in X.dims


        X = xrutils.unstack(X, "flat_kernel", ("chromo", "kernel"))

        n_kernels = len(self._G_kernel)
        n_kernels_brain = self._G_kernel_is_brain.sum()

        coords = {}

        if X.sizes["kernel"] == n_kernels:
            img = xrutils.dot_dataarray_csr(X, self._G, ["kernel", "vertex"])
            coords["is_brain"] = ("vertex", self._G_vertex_is_brain)
            if self._G_vertex_parcel is not None:
                coords["parcel"] = ("vertex", self._G_vertex_parcel)
        elif X.sizes["kernel"] == n_kernels_brain:  # brain_only == True
            img = xrutils.dot_dataarray_csr(
                X, self._G[:n_kernels_brain, :], ["kernel", "vertex"]
            )
            # FIXME even if only kernels on the brain are provided in X, G contains
            # vertices of the scalp which we have to select away afterwards.
            # It would be more efficient if these vertices would be cut from G.
            img = img.sel(vertex=self._G_vertex_is_brain)

            coords["is_brain"] = ("vertex", np.ones(img.sizes["vertex"], dtype=bool))
            if self._G_vertex_parcel is not None:
                coords["parcel"] = (
                    "vertex",
                    self._G_vertex_parcel[self._G_vertex_is_brain],
                )

        img = img.assign_coords(coords)

        return img


    def to_file(self, fname : Path | str):
        """Serialize prepared Gaussian spatial basis functions to HDF5 file.

        Args:
            fname: path of the output file.
        """
        raise NotImplementedError()


    @classmethod
    def from_file(cls, fname : Path | str) -> "GaussianSpatialBasisFunctions":
        """Load prepared Gaussian spatial basis functions from HDF5 group.

        Args:
            fname: path of the file to read from.

        Returns:
            GaussianSpatialBasisFunctions: Loaded instance.
        """

        raise NotImplementedError()


class ImageRecon:
    """Implements image reconstruction methods for diffuse optical tomography.

    Args:
        Adot: the sensitivity matrix
        recon_mode: select reconstruction method

            - 'conc': directly reconstruct hemoglobin concentrations from OD
                measurements at different wavelengths
            - 'mua': reconstruct absorption changes from OD measurements for each
                wavelength separately.
            - 'mua2conc': reconstruct absorption changes for each wavelength separately.
                Afterwards transform these to hemoglobin concentration changes.

        brain_only: if set to true, scalp vertices in Adot are ignored and the
            reconstruction is constrained to brain vertices

        alpha_meas: regularization parameter to adjust the balance between image
            noise and spatial resolution.

        alpha_spatial: regularization parameter that controls the effective depth of the
            reconstruction by controlling how strongly the vertex sensitivities are
            rescaled. A smaller alpha_spatial will more strongly suppress activation
            that is reconstructed on the scalp.

        lambda_R_conc: regularization parameter that sets the expected magnitude of the
            image covariance.

        apply_c_meas: controls whether the provided measurement covariance should be
            used for measurement regularization.

        spatial_basis_functions: if given reconstruct in the kernel space defined by the
            provided spatial-basis-function implementation. The result is returned in
            image space.
    """
    def __init__(
        self,
        Adot,
        *,
        alpha_meas: float = 0.001,
        alpha_spatial: float | None = None,
        lambda_R_conc: float | None = None,
        apply_c_meas: bool = False,
        recon_mode: ReconMode = "mua",
        brain_only: bool = False,
        spatial_basis_functions: SpatialBasisFunctions | None = None,
    ):
        if recon_mode not in ["conc", "mua", "mua2conc"]:
            raise ValueError(
                "recon_mode must be set to either 'conc', 'mua' or 'mua2conc'!"
            )
        # error handling of invalid params

        self.recon_mode = recon_mode

        # regularization parameters
        self.alpha_meas = alpha_meas
        self.alpha_spatial = alpha_spatial
        self.apply_c_meas = apply_c_meas
        self.lambda_R_conc = lambda_R_conc

        self.sbf = spatial_basis_functions
        self.Adot = Adot # FIXME can we remove this?
        self.brain_only = brain_only

        # cache intermediate matrices to avoid recomputations

        # These would invalidate when Adot or reg./sbf. params. change. Changing
        # these requires a new instance of ImageRecon, so they are considered constants
        # here. Depending on recon_mode they have different shapes.
        self._D: xr.DataArray = None  # R * A.T
        self._F: xr.DataArray = None  # A @ R @ A.T

        # The matrix to transform from absorption changes to concentrations
        self._mua2conc : xr.DataArray = None

        # W invalidates when C_meas changes
        self._W: xr.DataArray = None  # the pseudo_inverse (W=D@inv(F + lambda_meas C))
        self._W_input_hash: str = None  # a hash of C_meas. recompute W on C_meas change

        if self.sbf is not None:
            self._prepare(self.sbf.H)
        else:
            self._prepare(self.Adot)


    def reconstruct(
        self,
        y: cdt.NDTimeSeries,
        c_meas: xr.DataArray | None = None,
    ) -> cdt.NDTimeSeries:
        """Reconstruct images from measurement data.

        Args:
            y: optical density time series or time point data.
            c_meas: Diagonal elements of the measurement covariance matrix (optional).
                dims: wavelength x channel.

        Returns:
            Reconstructed images.
        """

        # y is optical density and dimensionless. Dequantify.
        y = y.pint.dequantify()

        # check if c_meas changed
        c_meas, new_W_input_hash = self._update_and_hash_cmeas(c_meas)

        # (re-)calculate W when C_meas is new
        if (self._W_input_hash is None) or (new_W_input_hash != self._W_input_hash):
            self._W_input_hash = new_W_input_hash
            self._W = self._get_W(c_meas)


        if self.recon_mode == "conc":
            conc_img = self._get_image_conc(y)
            conc_img = conc_img.pint.quantify("M").pint.to("uM")
            return conc_img

        mua_img = self._get_image_mua(y)
        mua_img = mua_img.pint.quantify("1/mm")

        if self.recon_mode == "mua":
            return mua_img
        elif self.recon_mode == "mua2conc":
            conc_img = xrutils.contract(self._mua2conc, mua_img, dim=["wavelength"])
            return conc_img.pint.to("uM")
        else:
            raise ValueError()  # unreachable


    def get_image_noise(self, c_meas: xr.DataArray):
        """Compute image noise/variance estimates.

        Args:
            c_meas: Measurement covariance matrix.

        Returns:
            xr.DataArray: Image noise estimates.
        """

        c_meas, new_W_input_hash = self._update_and_hash_cmeas(c_meas)

        # (re-)calculate W when C_meas is new
        if (self._W_input_hash is None) or (new_W_input_hash != self._W_input_hash):
            self._W_input_hash = new_W_input_hash
            self._W = self._get_W(c_meas)

        if self.recon_mode == "conc":
            c_meas = fwm.stack_flat_channel(c_meas)
            conc_img = self._get_image_noise_conc(c_meas)
            return conc_img

        elif self.recon_mode in ["mua", "mua2conc"]:
            mua_img = self._get_image_noise_mua(c_meas)

            if self.recon_mode == "mua":
                return mua_img
            else:
                return xrutils.contract(
                    self._mua2conc**2, mua_img / units.mm**2, "wavelength"
                )
        else:
            raise ValueError()  # unreachable


    # --- PREPARATION METHODS ---

    def _update_and_hash_cmeas(self, c_meas):
        if self.apply_c_meas:
            if c_meas is None:
                raise NotImplementedError(
                    "c_meas must be provided when apply_c_meas is set."
                )
            else:
                c_meas = c_meas.pint.dequantify()
        else:
            # override any provided c_meas if apply_c_meas == False
            c_meas = None

        if c_meas is not None:
            # average over time if c_meas should still have a time dimension
            time_dim = self._get_time_dimension(c_meas)
            if time_dim is not None:
                c_meas = c_meas.mean(time_dim)

            # calculate a hash value for c_meas
            new_W_input_hash = hashlib.blake2b(
                c_meas.pint.dequantify().values.tobytes()
            ).hexdigest()
        else:
            new_W_input_hash = "no_c_meas"

        return c_meas, new_W_input_hash



    def _prepare(self, Adot):
        """Precompute everything that depends only on inputs in the constructor."""

        if self.brain_only:
            Adot = self.Adot.sel(vertex=self.Adot.is_brain.values)

        # calculate D and F for the selected choice of recon_mode and sbf.
        if self.recon_mode == "conc":
            #Adot_stacked = get_stacked_sensitivity(Adot)
            Adot_stacked = fwm.ForwardModel.compute_stacked_sensitivity(Adot)
            self._D, self._F = self._calculate_DF_conc(Adot_stacked)

        elif self.recon_mode in ["mua", "mua2conc"]:
            self._D, self._F = self._calculate_DF_mua(Adot)

        else:
            raise ValueError()  # unreachable

        if self.recon_mode == "mua2conc":
            # calculate _mua2conc, which transforms absorption to concentration
            # changes
            E = nirs.get_extinction_coefficients("prahl", Adot.wavelength)
            self._mua2conc = xrutils.pinv(E)


    def _get_W(self, C_meas=None):
        """Get the pseudoinverse matrix W for reconstruction.

        Args:
            C_meas: Measurement covariance matrix (optional).

        Returns:
            xr.DataArray: pseudoinverse matrix W.
        """

        D = None

        # without spatial regularization:
        if self.alpha_spatial is None:
            # with spatial basis functions:
            if self.sbf is not None:
                if self.recon_mode == "conc":
                    if self.brain_only:
                        D = fwm.ForwardModel.compute_stacked_sensitivity(
                            self.sbf.H.sel(kernel=self.sbf.H.is_brain.values)
                        ).T
                    else:
                        D = fwm.ForwardModel.compute_stacked_sensitivity(
                            self.sbf.H
                        ).T
                else:
                    if self.brain_only:
                        D = self.sbf.H.sel(kernel=self.sbf.H.is_brain.values)
                    else:
                        D = self.sbf.H
                    D = D.transpose('kernel', 'channel', 'wavelength')
            # without spatial basis functions:
            else:
                if self.recon_mode == "conc":
                    if self.brain_only:
                        D = fwm.ForwardModel.compute_stacked_sensitivity(
                            self.Adot.sel(vertex=self.Adot.is_brain.values)
                        ).T
                    else:
                        D = fwm.ForwardModel.compute_stacked_sensitivity(
                            self.Adot
                        ).T
                else:
                    if self.brain_only:
                        D = self.Adot.sel(vertex=self.Adot.is_brain.values)
                    else:
                        D = self.Adot

                    D = D.transpose('vertex', 'channel', 'wavelength')
        # with spatial regularization:
        else:
            D = self._D

        if self.recon_mode == "conc":
            return self._calculate_W_conc(D, C_meas)
        if self.recon_mode in ["mua", "mua2conc"]:
            return self._calculate_W_mua(D, C_meas)


    # --- MATRIX COMPUTATION METHODS ---
    def _calculate_prior_R(self, A: xr.DataArray):
        """Compute spatial regularization prior (column scaling matrix).

        Calculates diagonal regularization matrix based on forward model sensitivity:
        R_j = 1 / (sum_i A_ij^2 + λ_spatial) where λ_spatial is scaled by max
        sensitivity. Vertices with high sensitivity get less regularization; low
        sensitivity vertices are smoothed more heavily.

        Parameters:
        A : numpy.ndarray or xr.DataArray
            Forward model matrix with shape (n_channels, n_vertices) or similar.
        alpha_spatial : float
            Spatial regularization weight controlling smoothness strength.

        Returns:
        R : numpy.ndarray or xr.DataArray
            Diagonal regularization matrix (as 1D array of diagonal elements)
            with same shape as columns of A.
        """

        B = np.sum((A**2), axis=0)
        b = B.max()

        if self.alpha_spatial is None:
            lambda_spatial = 1.
        else:
            lambda_spatial = self.alpha_spatial * b

        L = np.sqrt(B + lambda_spatial)
        Linv = 1 / L
        R = Linv**2

        return R

    def _calculate_DF(self, A: xr.DataArray):
        """Calculate intermediate D and F matrices for regularization.

        Args:
            A: Sensitivity matrix.

        Returns:
            D matrix as xr.DataArray
            F matrix as xr.DataArray

        """

        if self.alpha_spatial is None:
            dim = A.dims[0]
            F = A.values @ A.values.T
            F_xr = xr.DataArray(F, dims=(f"{dim}_1", f"{dim}_2"))
            D_xr = None
        else:
            #% GET spatial prior R
            R = self._calculate_prior_R(A)
            AR = A * R
            dim = AR.dims[0]

            #% GET F and D
            F = AR.values @ A.values.T
            D = R.values[:, np.newaxis] * A.values.T

            if self.sbf:
                if self.recon_mode in ["mua", "mua2conc"]:
                    vertex_dim = "kernel"
                    channel_dim = "channel"
                else:
                    vertex_dim = "flat_kernel"
                    channel_dim = "flat_channel"
            else:
                if self.recon_mode in ["mua", "mua2conc"]:
                    vertex_dim = "vertex"
                    channel_dim = "channel"
                else:
                    vertex_dim = "flat_vertex"
                    channel_dim = "flat_channel"


            #D_xr = xr.DataArray(D, dims=("flat_vertex", "flat_channel"))
            D_xr = xr.DataArray(D, dims=(vertex_dim, channel_dim))
            D_xr = D_xr.assign_coords(xrutils.coords_from_other(A,dims=D_xr.dims))
            F_xr = xr.DataArray(F, dims=(f"{dim}_1", f"{dim}_2"))

        return D_xr, F_xr


    def _calculate_DF_conc(self, Adot):
        """Calculate D and F matrices for concentration reconstruction.

        Args:
            Adot: Stacked sensitivity matrix for concentration.

        Returns:
            D matrix as xr.DataArray
            F matrix as xr.DataArray
        """
        return self._calculate_DF(Adot)


    def _calculate_DF_mua(self, Adot):
        """Calculate D and F matrices for mua reconstruction.

        Args:
            Adot: Sensitivity matrix with wavelength dimension.

        Returns:
            D matrix as xr.DataArray
            F matrix as xr.DataArray
        """

        D_lst = []
        F_lst = []
        for w in Adot.wavelength:
            D, F = self._calculate_DF(Adot.sel(wavelength=w.values))
            D_lst.append(D)
            F_lst.append(F)

        if all(d is not None for d in D_lst):
            D = xr.concat(D_lst, dim="wavelength")
            D = D.assign_coords({"wavelength": Adot.wavelength})
        else:
            D = None

        F = xr.concat(F_lst, dim="wavelength")
        F = F.assign_coords({"wavelength": Adot.wavelength})

        return D, F


    def _calculate_W(self, A, F, lambda_R, c_meas=None):
        """Calculate pseudoinverse W from sensitivity and regularization.

        Args:
            A: Sensitivity matrix.
            F: Regularization matrix F.
            c_meas: Measurement covariance matrix (optional).
            lambda_R: sets the expected magnitude of the image covariance

        Returns:
            xr.DataArray: pseudoinverse W.
        """

        # FIXME: lambda_R cancels out in the calculation of W. It could be removed here.
        if lambda_R is None:
            lambda_R = 1.

        lambda_meas = lambda_R * self.alpha_meas * np.max(np.linalg.eigh(F)[0])

        # A is 2D. Either (vertex x channel) or (kernel x channel)
        if c_meas is not None:
            W = lambda_R * A.values @ np.linalg.inv(lambda_R * F.values + lambda_meas * c_meas) # noqa:E501
        else:
            W = lambda_R * A.values @ np.linalg.inv(lambda_R * F.values + lambda_meas * np.eye(A.shape[1])) # noqa:E501

        vertex_dim = A.dims[0] # flat_vertex, flat_kernel, kernel
        channel_dim = A.dims[1] # flat_channel, channel

        W_xr = xr.DataArray(W, dims=(vertex_dim, channel_dim))

        W_xr = W_xr.assign_coords(
            xrutils.coords_from_other(A, dims=W_xr.dims)
        )

        return W_xr


    def _calculate_W_conc(self, A, c_meas=None):
        """Calculate pseudoinverse W for concentration reconstruction.

        Args:
            A: Stacked sensitivity matrix.
            c_meas: Measurement covariance matrix (optional).

        Returns:
            xr.DataArray: Pseudoinverse matrix for concentration reconstruction.
        """
        if c_meas is not None:
            c_meas = fwm.stack_flat_channel(c_meas)
            c_meas = np.diag(c_meas)

        return self._calculate_W(A, self._F, self.lambda_R_conc, c_meas)


    def _calculate_W_mua(self, A, c_meas=None):
        """Calculate pseudoinverse W for mua reconstruction.

        Args:
            A: Sensitivity matrix with wavelength dimension.
            c_meas: Measurement covariance matrix (optional).

        Returns:
            xr.DataArray: Pseudoinverse matrix for mua reconstruction with wavelength
                dimension.
        """
        W = []
        lambda_R_indirect = self.compute_lambda_R_indirect()
        for wavelength in A.wavelength:
            if c_meas is not None:
                c_meas_w = c_meas.sel(wavelength=wavelength)
                c_meas_w = np.diag(c_meas_w)
            else:
                c_meas_w = None

            W_xr = self._calculate_W(
                A.sel(wavelength=wavelength),
                self._F.sel(wavelength=wavelength),
                lambda_R_indirect.sel(wavelength=wavelength).values,
                c_meas_w,
            )
            W.append(W_xr)

        W_xr = xr.concat(W, dim="wavelength")
        W_xr = W_xr.assign_coords({"wavelength": A.wavelength})

        return W_xr

    # --- IMAGE RECONSTRUCTION METHODS ---
    def _get_image_conc(self, y: cdt.NDTimeSeries) -> cdt.NDTimeSeries:
        y = fwm.stack_flat_channel(y)
        y = y.reset_index("flat_channel")

        # make sure that ordering of channels is consistent
        # y may contain less channels then W due to pruning
        try:
            sel_channels = [
                i for i in self._W.flat_channel.values if i in y.flat_channel.values
            ]
            y = y.sel(flat_channel = sel_channels)
            W = self._W.sel(flat_channel = sel_channels)
        except KeyError:
            raise ValueError(
                "This time series contains channel(s) which is/are not in the "
                "sensitivity matrix!"
            )

        conc_img = xrutils.contract(W, y, "flat_channel")

        if self.sbf is None:
            conc_img = fwm.unstack_flat_vertex(conc_img)
        else:
            # direct recon with spatial basis
            conc_img = self.sbf.kernel_to_image_space_conc(conc_img)

        return conc_img


    def _get_image_mua(self, y):
        """Compute absorption coefficient image from measurements.

        Args:
            y: Optical density measurement data with wavelength dimension.

        Returns:
            xr.DataArray: Absorption coefficient image.
        """

        # make sure that ordering of channels is consistent y may contain less channels
        # then W due to pruning.
        try:
            sel_channels = [i for i in self._W.channel.values if i in y.channel.values]
            y = y.sel(channel=sel_channels)
            W = self._W.sel(channel=sel_channels)
        except KeyError:
            raise ValueError(
                "This time series contains channel(s) which is/are not in the "
                "sensitivity matrix!"
            )

        mua_img = xrutils.contract(W, y, dim="channel")

        if self.sbf is not None:
            mua_img = self.sbf.kernel_to_image_space_mua(mua_img)

        return mua_img


    def _get_image_noise_conc(self, c_meas: xr.DataArray | None = None):
        """Compute concentration image noise/variance.

        Args:
            c_meas: Measurement covariance matrix

        Returns:
            xr.DataArray: Image noise with proper dimensions and coordinates
        """
        if c_meas is None:
            raise ValueError("c_meas cannot be None for noise computation")

        # Detect time dimension
        time_dim = self._get_time_dimension(c_meas)
        has_time = time_dim is not None

        # Compute noise variance: diag(W @ C_meas @ W.T)
        if has_time:
            # Vectorized computation over time
            c_meas = c_meas.transpose('flat_channel', time_dim)
            noise_var = self._compute_time_varying_noise(self._W, c_meas)
        else:
            # Single timepoint
            noise_var = self._compute_single_noise(self._W, c_meas)

        # Apply spatial basis transformation if needed
        if self.sbf is not None:
            noise_var = self.sbf.kernel_to_image_space_conc(noise_var)
            # if has_time:
            #     noise_var = noise_var  # Transpose back to (vertex, time)
        else:
            # Reshape for HbO/HbR concentration
            # = self._reshape_conc(noise_var, has_time)
            noise_var = fwm.unstack_flat_vertex(noise_var)

        return noise_var
        # Create properly formatted xarray
        #return self._create_conc_dataarray(noise_var, c_meas, time_dim)


    def _get_image_noise_mua(self, c_meas: xr.DataArray | None = None):
        """Compute absorption coefficient image noise/variance.

        Args:
            c_meas: Measurement covariance matrix

        Returns:
            xr.DataArray: Image noise with proper dimensions and coordinates
        """
        if c_meas is None:
            raise ValueError("c_meas cannot be None for noise computation")

        # Detect time dimension
        time_dim = self._get_time_dimension(c_meas)
        has_time = time_dim is not None

        noise_var_list = []

        # Process each wavelength separately
        for wavelength in self._W.wavelength:
            W_wl = self._W.sel(wavelength=wavelength)
            c_wl = c_meas.sel(wavelength=wavelength)

            # Compute noise for this wavelength
            if has_time:
                noise_wl = self._compute_time_varying_noise(W_wl, c_wl)
            else:
                noise_wl = self._compute_single_noise(W_wl, c_wl)

            # Apply spatial basis transformation if needed
            if self.sbf is not None:
                noise_wl = self.sbf.kernel_to_image_space_mua(noise_wl)

            noise_var_list.append(noise_wl)

        # Combine wavelengths
        #noise_var = np.stack(noise_var_list, axis=0)  # (wavelength, vertex, time)
        noise_var = xr.concat(noise_var_list, dim="wavelength")
        noise_var = noise_var.assign_coords({
            "wavelength" : self._W.wavelength.values
        })

        return noise_var
        # Create properly formatted xarray
        #return self._create_mua_dataarray(noise_var, c_meas, time_dim)

    def get_image_noise_posterior(self, c_meas : xr.DataArray | None = None):
        """Compute posterior variance of reconstructed images.

        Calculates the diagonal of the posterior covariance matrix:
        Cov(X|y) = R - R * A^T @ (F + λ*C)^(-1) @ A * R
        where R is the spatial prior. Returns only the diagonal (variance at each
        vertex).

        Parameters:
            c_meas : Measurement covariance matrix.

        Returns:
            xr.DataArray: Posterior variance of reconstructed images.
        """

        c_meas, new_W_input_hash = self._update_and_hash_cmeas(c_meas)

        # (re-)calculate W when C_meas is new
        if (self._W_input_hash is None) or (new_W_input_hash != self._W_input_hash):
            self._W_input_hash = new_W_input_hash
            self._W = self._get_W(c_meas)

        if self.recon_mode == "conc":
            conc_img = self._get_posterior_cov_conc()
            return conc_img

        elif self.recon_mode in ["mua", "mua2conc"]:
            mua_img = self._get_posterior_cov_mua()
            if self.recon_mode == "mua":
                return mua_img
            else:
                return xrutils.contract(
                    self._mua2conc**2, mua_img / units.mm**2, "wavelength"
                )
        else:
            raise ValueError()  # unreachable

    def _get_posterior_cov_conc(self):
        if self.sbf is not None:
            A = self.sbf.H
            Adot_stacked = fwm.ForwardModel.compute_stacked_sensitivity(A)
        else:
            A = self.Adot
            Adot_stacked = fwm.ForwardModel.compute_stacked_sensitivity(A)

        W = self._W
        R = self._calculate_prior_R(Adot_stacked)
        R = self.lambda_R_conc * R

        # ---------------------------------------------------------
        # Posterior variance (diagonal only)
        # mse_post(j) = R_j * (1 - (W A^T)_{jj})
        # ---------------------------------------------------------
        s = np.sum(W * Adot_stacked.T, axis=1)  # elementwise multiply row i with col. i
        mse_post = R * (1.0 - s)

        if self.sbf is not None:
            mse_post = self.sbf.kernel_to_image_space_conc(mse_post).T
        else:
            mse_post = fwm.unstack_flat_vertex(mse_post)

        # FIXME should probably not assume units here
        mse_post = mse_post.pint.quantify("molar**2")

        return mse_post

    def _get_posterior_cov_mua(self):
        """Compute W and mse_posterior for a given wavelength.

        It use spatial regularization (via column scaling) and measurement
        regularization in data space.
        """
        if self.sbf is not None:
            A = self.sbf.H
        else:
            A = self.Adot

        lambda_R_indirect = self.compute_lambda_R_indirect()
        mse_lst = []
        W = self._W

        for wl in A.wavelength:

            lambda_R_wl = lambda_R_indirect.sel(wavelength=wl).values

            A_wl = A.sel(wavelength=wl)
            W_wl = W.sel(wavelength=wl)

            R = self._calculate_prior_R(A_wl)
            R = lambda_R_wl * R

            # ---------------------------------------------------------
            # Posterior variance (diagonal only)
            # mse_post(j) = R_j * (1 - (W A^T)_{jj})
            # ---------------------------------------------------------
            s = np.sum(W_wl * A_wl.T, axis=1)  # elementwise multiply row i with col. i
            mse_post = R * (1.0 - s)
            mse_lst.append(mse_post)

        mse_post_xr = xr.concat(mse_lst, dim='wavelength')

        if self.sbf is not None:
            mse_post_xr = self.sbf.kernel_to_image_space_mua(mse_post_xr)

        return mse_post_xr

    # --- HELPER METHODS FOR IMAGE COMPUTATION ---

    def _get_time_dimension(self, data: xr.DataArray) -> str | None:
        """Detect time dimension in data."""
        for dim in ['time', 'reltime']:
            if dim in data.dims:
                return dim
        return None

    def _get_spatial_dimension(self, data: xr.DataArray) -> str | None:
        for dim in ['vertex', 'kernel', 'parcel', 'channel']:
            if dim in data.dims:
                return dim
        return None

    def _compute_single_noise(
        self, W: xr.DataArray, c_meas: xr.DataArray
    ) -> xr.DataArray:
        """Compute noise for single timepoint: diag(W @ C @ W.T)."""
        return ((W * np.sqrt(c_meas))**2).sum(axis=1)

    def _compute_time_varying_noise(
        self, W: xr.DataArray, c_meas: xr.DataArray
    ) -> np.ndarray:
        """Compute noise for multiple timepoints efficiently.

        This unified method works for both:
        - Full multi-wavelength case (concentration reconstruction)
        - Single wavelength case (mua reconstruction)

        Args:
            W: Weight matrix with dims (vertex, flat_channel) or (vertex, channel)
            c_meas: Covariance with dims (time, flat_channel) or (time, channel)

        Returns:
            np.ndarray: Noise variance with dims (vertex, time)
        """
        # Vectorized computation over time
        c_sqrt = np.sqrt(c_meas.values)  # (time, flat_channel) or (time, channel)
        W_expanded = W.values[:, :, np.newaxis]  # (vertex, flat_channel/channel, 1)

        # Broadcasting: (vertex, flat_channel/channel, time)
        weighted = W_expanded * c_sqrt
        return np.nansum(weighted**2, axis=1)  # (vertex, time)

    def compute_lambda_R_indirect(self):
        """Compute wavelength-specific prior scaling parameter for indirect recon.

        Scales lambda_R to ensure consistency between direct
        (chromophore space) and indirect (wavelength space) methods. Uses extinction
        coefficients to relate chromophore regularization strength to OD regularization.

        Returns:
            lambda_R_indirect : xr.DataArray
            Wavelength-specific parameter with dimension (wavelength,).
            Scaled to match direct method's effective regularization strength.

        """

        # FIXME catch earlier?
        if self.lambda_R_conc is None:
            return xr.DataArray(
                [1.0, 1.0],
                dims=["wavelength"],
                coords={"wavelength": self.Adot.wavelength},
            )

        conc2mua = nirs.get_extinction_coefficients("prahl", self.Adot.wavelength)

        A_stacked = fwm.ForwardModel.compute_stacked_sensitivity(self.Adot)
        nV_brain = self.Adot.is_brain.sum().values
        nV_head = self.Adot.shape[1]

        R_direct = self._calculate_prior_R(A_stacked)
        R_direct = self.lambda_R_conc * R_direct

        R_direct_max = [
            R_direct[:nV_brain].max().values,
            R_direct[nV_head : nV_head + nV_brain].max().values,
        ]

        # Convert direct prior to indirect (OD space)
        R_indirect_wl1 = self._calculate_prior_R(self.Adot.isel(wavelength=0))
        R_indirect_wl2 = self._calculate_prior_R(self.Adot.isel(wavelength=1))

        conc2mua = conc2mua.pint.dequantify()  # FIXME: check units
        R_indirect_converted = conc2mua.values**2 @ R_direct_max  # / units.mm**2

        lambda_wl1 = R_indirect_converted[0] / R_indirect_wl1[:nV_brain].max()
        lambda_wl2 = R_indirect_converted[1] / R_indirect_wl2[:nV_brain].max()

        lambda_R_indirect = xr.DataArray(
            [lambda_wl1, lambda_wl2],
            dims=["wavelength"],
            coords={"wavelength": self.Adot.wavelength},
        )

        return lambda_R_indirect
