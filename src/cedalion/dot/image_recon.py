"""Solver for the image reconstruction problem."""

import numpy as np
import pint
import xarray as xr
from numpy.typing import ArrayLike
from pathlib import Path
from abc import ABC, abstractmethod
import h5py
from tqdm import tqdm 
from typing import Optional, Union, Dict, Any, Tuple
from cedalion.dot.utils import get_stacked_sensitivity
from scipy.spatial import KDTree
import cedalion.xrutils as xrutils
import cedalion.typing as cdt
from dataclasses import dataclass

from cedalion import units, nirs


@dataclass
class RegularizationParams:
    """Parameters controlling the regularization of the inverse problem.

    Args:
        alpha_meas: ...
        ...
    """

    alpha_meas: float
    alpha_spatial: None | float
    apply_c_meas: bool  # FIXME better name


class SpatialBasisFunctions(ABC):
    """Parameters controlling the spatial basis functions.

    Args:
        threshold_brain: ...
        ...
    """

    @abstractmethod
    def prepare(self, head_model, Adot) -> xr.DataArray:
        """Setup internal state and return reduced Adot."""
        pass

    # @abstractmethod
    # def kernel_to_image_space(self, X):
    #     pass

    @abstractmethod
    def to_hdf5_group(self, group: h5py.Group):
        """Serialize spatial basis functions to HDF5 group.
        
        Args:
            group: HDF5 group to write data to.
        """
        pass

    @classmethod
    @abstractmethod
    def from_hdf5_group(cls, group: h5py.Group) -> "SpatialBasisFunctions":
        """Load spatial basis functions from HDF5 group.
        
        Args:
            group: HDF5 group to read data from.
            
        Returns:
            SpatialBasisFunctions: Loaded spatial basis functions instance.
        """
        pass


class GaussianSpatialBasisFunctions(SpatialBasisFunctions):
    def __init__(
        self,
        threshold_brain: cdt.QLength,
        threshold_scalp: cdt.QLength,
        sigma_brain: cdt.QLength,
        sigma_scalp: cdt.QLength,
        mask_threshold: float,
    ):
        self.threshold_brain = threshold_brain
        self.threshold_scalp = threshold_scalp
        self.sigma_brain = sigma_brain
        self.sigma_scalp = sigma_scalp
        self.mask_threshold = mask_threshold

        self._G = None

    def prepare(self, head_model, Adot) -> xr.DataArray:
        """Prepare the spatial basis functions by computing sensitivity mask and Gaussian kernels.
        
        Args:
            head_model: Head model containing brain and scalp surfaces.
            Adot: Sensitivity matrix.
        """
        # compute _G
        self._get_sensitivity_mask(Adot)
        self._generate_G_gaussian_kernels(head_model)
        self.get_H(Adot)

    def _get_sensitivity_mask(self, Adot, wavelength_idx: int =0):
        """Compute sensitivity mask based on intensity threshold.
        
        Args:
            Adot: Sensitivity matrix.
            wavelength_idx: Index of wavelength to use for mask computation.
        """
        intensity = np.log10(Adot.isel(wavelength=wavelength_idx).sum('channel'))
        mask = intensity > self.mask_threshold
        mask = mask.drop_vars('wavelength')
        self.mask = mask


    def _downsample_mesh(self, mesh, threshold, mask):
        """Downsample the mesh to get seeds of spatial bases.

        Args:
            mesh (xr.DataArray): mesh of either the brain or scalp surface.
            threshold (Quantity): distance between vertices in downsampled mesh.
        
        Returns:
            xr.DataArray: downsampled mesh
        
        Initial Contributors:
            - Yuanyuan Gao 
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        # Downsample the mesh using the specified method
        mesh_units = mesh.pint.units
        threshold = threshold.to(mesh_units)
    
        mesh = mesh.rename({'label':'vertex'}).pint.dequantify()
        mesh_masked = mesh[mask,:]
        mesh_new = []

        for vv in tqdm(mesh_masked):
            if len(mesh_new) == 0: 
                mesh_new.append(vv)
                tree = KDTree(mesh_new)  # Build KDTree for the first point
                continue
            
            # Query the nearest neighbor within the threshold
            distance, _ = tree.query(vv, distance_upper_bound=threshold.magnitude)

            
            # If no point is within the threshold, append the new point
            if distance == float('inf'):
                mesh_new.append(vv)
                tree = KDTree(mesh_new)  # Rebuild the KDTree with the new point

        mesh_new_xr = xr.DataArray(mesh_new,
                                dims = mesh.dims,
                                coords = {'vertex':np.arange(len(mesh_new))},
                                attrs = {'units': mesh_units }
            )
        
        mesh_new_xr = mesh_new_xr.pint.quantify()
        
        return mesh_new_xr

    def _get_gaussian_kernels(self, mesh_downsampled, mesh, sigma):
        """Get the matrix containing the spatial bases.

        Args:
            mesh_downsampled (xr.DataArray): mesh of either the downsampled brain or scalp surface.
                This is used to define the centers of the spatial bases.
            mesh (xr.DataArray): the original fully sampeld mesh of the brain or scalp. 
            sigma (Quantity): standard deviation used for defining the Gaussian kernel.
            
        Returns:
            xr.DataArray: matrix containing the spatial bases

        Initial Contributors:
            - Yuanyuan Gao 
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        # Create Gaussian kernels based on the mesh and parameters
        assert mesh.pint.units == mesh_downsampled.pint.units
    
        mesh_units = mesh.pint.units
        sigma = sigma.to(mesh_units)
    
        # Covariance matrix
        cov_matrix = (sigma.magnitude **2) * np.eye(3)
        inv_cov = np.linalg.inv(cov_matrix)  # Inverse of Cov_matrix
        det_cov = np.linalg.det(cov_matrix)  # Determinant of Cov_matrix
        denominator = np.sqrt((2 * np.pi) ** 3 * det_cov)  # Pre-calculate denominator

        mesh_downsampled = mesh_downsampled.pint.dequantify().values
        mesh = mesh.pint.dequantify().values
        
        diffs = mesh_downsampled[:, None, :] - mesh[None, :, :]

        # Efficient matrix multiplication using np.einsum to compute (x-mu)' * inv_cov * (x-mu) for all pairs
        exponents = -0.5 * np.einsum('ijk,kl,ijl->ij', diffs, inv_cov, diffs)

        # Compute the kernel matrix
        kernel_matrix = np.exp(exponents) / denominator
        n_vertex = mesh.shape[0]
        
        dimensions = kernel_matrix.shape
        
        if dimensions[0] != n_vertex:
            dims = ["kernel", "vertex"]
            n_kernel = dimensions[0]
        else:
            dims = ["vertex", "kernel"]
            n_kernel = dimensions[1]
        
        kernel_matrix_xr = xr.DataArray(kernel_matrix, 
                                        dims = dims,
                                        coords = {'vertex': np.arange(n_vertex),
                                                'kernel': np.arange(n_kernel)}
                                        )
        
        kernel_matrix_xr = kernel_matrix_xr.transpose('vertex', 'kernel')
        return kernel_matrix_xr
    
    def _generate_G_gaussian_kernels(self, head_model):
        """Get the G matrix which contains all the information of the spatial basis

        Args:
            head (cfm.TwoSurfaceHeadModel): Head model with brain and scalp surfaces.

        Initial Contributors:
            - Yuanyuan Gao 
            - Laura Carlton | lcarlton@bu.edu | 2024

        """
        brain_downsampled = self._downsample_mesh(head_model.brain.vertices, self.threshold_brain, self.mask.sel(vertex=self.mask.is_brain))
        scalp_downsampled = self._downsample_mesh(head_model.scalp.vertices, self.threshold_scalp, self.mask.sel(vertex=~self.mask.is_brain))
    
        G_brain = self._get_gaussian_kernels(
            brain_downsampled, head_model.brain.vertices, self.sigma_brain
        )
        G_scalp = self._get_gaussian_kernels(
            scalp_downsampled, head_model.scalp.vertices, self.sigma_scalp
        )

        G = {'G_brain': G_brain, 'G_scalp': G_scalp}
        self._G = G

    def get_H(self, Adot):
        """Compute the H matrix for spatial basis functions. Transforms the sensitivity matrix into the spatial basis space.

        Args:
            Adot: Sensitivity matrix.
        """
        n_channel = len(Adot.channel)
        n_wavlength = len(Adot.wavelength)
        n_kernel_brain = len(self._G['G_brain'].kernel)
        n_kernel = len(self._G['G_brain'].kernel) + len(self._G['G_scalp'].kernel)

        H = np.zeros((n_channel, n_kernel, n_wavlength))
        for w in range(n_wavlength):
            Adot_w = Adot.isel(wavelength=w).values
            H[:, :n_kernel_brain, w] = Adot_w[:, Adot.is_brain] @ self._G['G_brain'].values
            H[:, n_kernel_brain:, w] = Adot_w[:, ~Adot.is_brain] @ self._G['G_scalp'].values

        is_brain = np.ones(n_kernel, dtype=np.bool_)
        is_brain[n_kernel_brain:] = False

        H = xr.DataArray(H, dims=("channel", "kernel", "wavelength"))

        H = H.assign_coords({'channel': Adot.channel,
                            'wavelength': Adot.wavelength,
                            'kernel': np.arange(n_kernel),
                            'is_brain': ('kernel', is_brain)})
        self._H = H

    def kernel_to_image_space_mua(self, X):
        """Convert kernel space reconstructions to image space for mua.
        
        Args:
            X: Reconstruction values in kernel space.
            
        Returns:
            np.ndarray: Reconstruction values in image space.
        """
        nkernels_brain = self._G['G_brain'].kernel.shape[0]
        has_scalp = len(X) > nkernels_brain if len(X.shape) < 2 else X.shape[0] > nkernels_brain
        
        if len(X.shape) < 2:
            sb_X_brain = X[:nkernels_brain]
            if has_scalp:
                sb_X_scalp = X[nkernels_brain:]
        else:
            sb_X_brain = X[:nkernels_brain,:]
            if has_scalp:
                sb_X_scalp = X[nkernels_brain:,:]
        
        #% PROJECT BACK TO SURFACE SPACE 
        X_brain = self._G['G_brain'].values @ sb_X_brain
        
        if has_scalp:
            X_scalp = self._G['G_scalp'].values @ sb_X_scalp
            # concatenate them back together
            X = np.concatenate([X_brain, X_scalp])
        else:
            # Only brain vertices
            X = X_brain
        return X

    def kernel_to_image_space_conc(self, X):
        """Convert kernel space reconstructions to image space for concentration.
        
        Args:
            X: Reconstruction values in kernel space.
            
        Returns:
            np.ndarray: Reconstruction values in image space with HbO/HbR split.
        """

        split = X.shape[0]//2
        nkernels_brain = self._G['G_brain'].kernel.shape[0]
        has_scalp = split > nkernels_brain
        
        if len(X.shape) > 1:
            X_hbo = X[:split,:]
            X_hbr = X[split:,:]
            sb_X_brain_hbo = X_hbo[:nkernels_brain,:]
            sb_X_brain_hbr = X_hbr[:nkernels_brain,:]
            
            if has_scalp:
                sb_X_scalp_hbo = X_hbo[nkernels_brain:,:]
                sb_X_scalp_hbr = X_hbr[nkernels_brain:,:]
        else:
            X_hbo = X[:split]
            X_hbr = X[split:]
            sb_X_brain_hbo = X_hbo[:nkernels_brain]
            sb_X_brain_hbr = X_hbr[:nkernels_brain]
            
            if has_scalp:
                sb_X_scalp_hbo = X_hbo[nkernels_brain:]
                sb_X_scalp_hbr = X_hbr[nkernels_brain:]
            
        #% PROJECT BACK TO SURFACE SPACE 
        X_hbo_brain = self._G['G_brain'].values @ sb_X_brain_hbo
        X_hbr_brain = self._G['G_brain'].values @ sb_X_brain_hbr
        
        if has_scalp:
            X_hbo_scalp = self._G['G_scalp'].values @ sb_X_scalp_hbo
            X_hbr_scalp = self._G['G_scalp'].values @ sb_X_scalp_hbr

        # concatenate them back together
        if len(X.shape) == 1:
            if has_scalp:
                X = np.stack([np.concatenate([X_hbo_brain, X_hbo_scalp]),np.concatenate([ X_hbr_brain, X_hbr_scalp])], axis=0)
            else:
                X = np.stack([X_hbo_brain, X_hbr_brain], axis=0)
        else:
            if has_scalp:
                X = np.stack([np.vstack([X_hbo_brain, X_hbo_scalp]), np.vstack([X_hbr_brain, X_hbr_scalp])], axis =0)
            else:
                X = np.stack([X_hbo_brain, X_hbr_brain], axis=0)
        return X

    def to_hdf5_group(self, group):
        """Serialize Gaussian spatial basis functions to HDF5 group.
        
        Args:
            group: HDF5 group to write data to.
        """
        pass

    @classmethod
    def from_hdf5_group(self, group):
        """Load Gaussian spatial basis functions from HDF5 group.
        
        Args:
            group: HDF5 group to read data from.
            
        Returns:
            GaussianSpatialBasisFunctions: Loaded instance.
        """
        pass


class ParcellationBasisFunctions(SpatialBasisFunctions):
    pass


# we could define constants of parameters that work well together, e.g. based on
# Laura's parameter scans:
REG_TIKHONOV_ONLY = RegularizationParams(
    alpha_meas=0.001, alpha_spatial=None, apply_c_meas=False
)

REG_TIKHONOV_SPATIAL = RegularizationParams(
    alpha_meas=0.01, alpha_spatial=0.001, apply_c_meas=False
)

SBF_GAUSSIANS_DENSE = GaussianSpatialBasisFunctions(
    mask_threshold=-2,
    threshold_brain=1 * units.mm,
    threshold_scalp=5 * units.mm,
    sigma_brain=1 * units.mm,
    sigma_scalp=5 * units.mm,
)

SBF_GAUSSIANS_SPARSE = GaussianSpatialBasisFunctions(mask_threshold=-2,
                                                    threshold_brain=5 * units.mm,
                                                    threshold_scalp=20 * units.mm,
                                                    sigma_brain=5 * units.mm,
                                                    sigma_scalp=20 * units.mm,
                                                )


# likewise, if there are any heuristics how to set these parameters, we could offer
# functions to compute them
def estimate_reg_params(*args) -> RegularizationParams:
    """Estimate regularization parameters from data.
    
    Args:
        *args: Variable arguments for parameter estimation.
        
    Returns:
        RegularizationParams: Estimated regularization parameters.
    """
    pass


class ImageRecon:

    def __init__(
        self,
        head_model,
        Adot,
        recon_mode: str = "mua",
        regularization_params: RegularizationParams = REG_TIKHONOV_ONLY,
        spatial_basis_functions: None | SpatialBasisFunctions = None,
    ):
        # error handling of invalid params

        self.recon_mode = recon_mode
        self.reg_params = regularization_params
        self.sbf = spatial_basis_functions
        self.Adot = Adot

        # cache intermediate matrices to avoid recomputations

        # These invalidate when Adot or reg./sbf. params. change.
        # Depending on recon_mode they have different shapes.
        self._D = None  # Linv^2 * A.T
        self._F = None  # A_hat * A_hat.T

        # self._G = None  # SBF kernel matrices for brain and scalp

        # self._mua2conc = 

        # this invalidates when C_meas changes
        self._W = None  # the pseudo_inverse (W=D@inv(F+ lambda_meas C))
        # self._W_input_hash  # a hash of C_meas. if C_meas changes, recompute W

        if self.sbf is not None:
            self.sbf.prepare(head_model, self.Adot)
            self._prepare(self.sbf._H)
        else:
            self._prepare(self.Adot)

    def reconstruct(
        self,
        y: cdt.NDTimeSeries,
        c_meas: xr.DataArray | None = None,
    ) -> cdt.NDTimeSeries:
        """Reconstruct images from measurement data.
        
        Args:
            y: optical density measurement time series or time point data.
            c_meas: Measurement covariance matrix (optional).
            
        Returns:
            cdt.NDTimeSeries: Reconstructed images.
        """
        if (c_meas is None) and self.reg_params.apply_c_meas:
            # estimate c_meas from time_series
            c_meas = ...

        # calculate hash(c_meas) and (re)compute W if necessary
        # W_input_hash = hash(tuple(c_meas))

        if (self._W is None): #or (W_input_hash != self._W_input_hash):
            if c_meas is not None:
                time_dim = self._get_time_dimension(c_meas)
                if time_dim is not None:
                    c_meas = c_meas.mean(time_dim)
                else:
                    c_meas = c_meas

            if self.reg_params.alpha_spatial is None:
                if self.sbf is not None:
                    if self.recon_mode == "conc":
                        D = get_stacked_sensitivity(self.sbf._H.sel(kernel=self.sbf._H.is_brain.values)).T
                    else:
                        D = self.sbf._H.sel(kernel=self.sbf._H.is_brain.values).transpose('kernel', 'channel', 'wavelength')
                    self._W = self._get_W(D, c_meas)
                else:
                    # Need to store original Adot for this case
                    if self.recon_mode == "conc":
                        D = get_stacked_sensitivity(self.Adot.sel(vertex=self.Adot.is_brain.values)).T
                    else:
                        D = self.Adot.sel(vertex=self.Adot.is_brain.values).transpose('vertex', 'channel', 'wavelength')
                    self._W = self._get_W(D, c_meas)
            else:
                self._W = self._get_W(self._D, c_meas)

            # self._W_input_hash = W_input_hash

        if self.recon_mode == "conc":
            y = y.stack(measurement=("wavelength", "channel")).sortby('wavelength')
            conc_img = self._get_image_conc(y)
            return conc_img
        
        elif self.recon_mode in ["mua", "mua*mua2conc"]:
            mua_img = self._get_image_mua(y)

            if self.recon_mode == "mua":
                return mua_img
            else:
                return xr.dot(self._mua2conc, mua_img/units.mm, dims=["wavelength"])
        else:
            raise ValueError()  # unreachable
        
    def get_image_noise(
                        self, c_meas: xr.DataArray
                    ):
        """Compute image noise/variance estimates.
        
        Args:
            c_meas: Measurement covariance matrix.
            
        Returns:
            xr.DataArray: Image noise estimates.
        """
        
        if (c_meas is None) and self.reg_params.apply_c_meas:
            # estimate c_meas from time_series
            c_meas = ...

        # calculate hash(c_meas) and (re)compute W if necessary
        # W_input_hash = hash(tuple(c_meas))

        if (self._W is None): #or (W_input_hash != self._W_input_hash):
            if c_meas is not None:
                time_dim = self._get_time_dimension(c_meas)
                if time_dim is not None:
                    c_meas_tmp = c_meas.mean(time_dim)
                else:
                    c_meas_tmp = c_meas

            if self.reg_params.alpha_spatial is None:
                if self.sbf is not None:
                    if self.recon_mode == "conc":
                        D = get_stacked_sensitivity(self.sbf._H.sel(kernel=self.sbf._H.is_brain.values)).T
                    else:
                        D = self.sbf._H.sel(kernel=self.sbf._H.is_brain.values).transpose('kernel', 'channel', 'wavelength')
                    self._W = self._get_W(D, c_meas_tmp)
                else:
                    # Need to store original Adot for this case
                    if self.recon_mode == "conc":
                        D = get_stacked_sensitivity(self.Adot.sel(vertex=self.Adot.is_brain.values)).T
                    else:
                        D = self.Adot.sel(vertex=self.Adot.is_brain.values).transpose('vertex', 'channel', 'wavelength')
                    self._W = self._get_W(D, c_meas_tmp)
            else:
                self._W = self._get_W(self._D, c_meas_tmp)


            # self._W_input_hash = W_input_hash

        if self.recon_mode == "conc":
            c_meas = c_meas.stack(measurement=("wavelength", "channel")).sortby('channel')
            conc_img = self._get_image_noise_conc(c_meas)
            return conc_img
        
        elif self.recon_mode in ["mua", "mua*mua2conc"]:
            mua_img = self._get_image_noise_mua(c_meas)

            if self.recon_mode == "mua":
                return mua_img
            else:
                return self._mua2conc**2 @ mua_img/units.mm**2  
        else:
            raise ValueError()  # unreachable
        
    def get_image_noise_tstat(
        self, time_series: cdt.NDTimeSeries, c_meas: xr.DataArray | None = None
    ):
        """Compute t-statistic images from noise estimates.
        
        Args:
            time_series: Time series data for statistics computation.
            c_meas: Measurement covariance matrix (optional).
            
        Returns:
            xr.DataArray: T-statistic images.
        """
        # FIXME is this not already images ? so just X_image / X_noise?
        # not sure what time_series and C_meas would be here 
        pass

    def to_file(self, fname: Path | str):
        """Serialize to disk."""

        with h5py.File(fname, "w") as f:
            # store params
            # store D,F,W, mua2conc

            sbf_group = f.create_group("sbf")

            if self.sbf:
                self.sbf.to_hdf5_group(sbf_group)

    @classmethod
    def from_file(cls, fname: str | Path) -> "ImageRecon":
        """Load saved instance from disk.
        
        Args:
            fname: Path to the saved file.
            
        Returns:
            ImageReco: Loaded ImageReco instance.
        """
        pass

    # --- PREPARATION METHODS --- 
    def _prepare(self, Adot):
        """Precompute everything that depends only on inputs in the constructor."""
        if self.reg_params.alpha_spatial is None:
            self.Adot = self.Adot.sel(vertex=self.Adot.is_brain.values)
            
        # calculate D and F for the selected choice of recon_mode and sbf.
        if self.recon_mode == "conc":
            Adot_stacked = get_stacked_sensitivity(Adot)
            self._D, self._F = self._calculate_DF_conc(Adot_stacked)
            
        elif self.recon_mode in ["mua", "mua*mua2conc"]:
            self._D, self._F = self._calculate_DF_mua(Adot)

        else:
            raise ValueError()  # unreachable

        if self.recon_mode == "mua*mua2conc":
            # calculate _mua2conc # FIXME not sure what this is ?
            E = nirs.get_extinction_coefficients('prahl', Adot.wavelength)
            self._mua2conc = xrutils.pinv(E)


    def _get_W(self, A, C_meas=None):
        """Get the pseudoinverse matrix W for reconstruction.
        
        Args:
            A: Sensitivity matrix.
            C_meas: Measurement covariance matrix (optional).
            
        Returns:
            xr.DataArray: pseudoinverse matrix W.
        """
        if self.recon_mode == "conc":
            # A = get_stacked_sensitivity(A)
            return self._calculate_W_conc(A, C_meas)
        if self.recon_mode in ["mua", "mua*mua2conc"]:
            return self._calculate_W_mua(A, C_meas)

    # --- MATRIX COMPUTATION METHODS ---
    def _calculate_DF(self, A):
        """Calculate intermediate D and F matrices for regularization.
        
        Args:
            A: Sensitivity matrix.
            
        Returns:
            D matrix as xr.DataArray
            F matrix as xr.DataArray

        """

        if self.reg_params.alpha_spatial is None:
            F = A.values @ A.values.T
            F_xr = xr.DataArray(F, dims=("measurement1", "measurement2"))
            D_xr = None
        else:
            B = np.sum((A ** 2), axis=0)
            b = B.max()
            
            # GET A_HAT
            lambda_spatial = self.reg_params.alpha_spatial * b

            L = np.sqrt(B + lambda_spatial)
            Linv = 1/L.values
            # Linv = np.diag(Linv)

            A_hat = A * Linv

            #% GET F and D
            F = A_hat.values @ A_hat.values.T
            D = Linv[:, np.newaxis]**2 * A.values.T

            D_xr = xr.DataArray(D, dims=("flat_vertex", "measurement"))
            if 'parcel' in A.coords:
                D_xr = D_xr.assign_coords({"parcel" : ("flat_vertex", A.coords['parcel'].values)})
            if 'is_brain' in A.coords:
                D_xr = D_xr.assign_coords({"is_brain": ("flat_vertex", A.coords['is_brain'].values)})

            F_xr = xr.DataArray(F, dims=("measurement1", "measurement2"))

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
            D,F = self._calculate_DF(Adot.sel(wavelength=w.values))
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

    def _calculate_W(self, A, F, c_meas=None):
        """Calculate pseudoinverse W from sensitivity and regularization.
        
        Args:
            A: Sensitivity matrix.
            F: Regularization matrix F.
            c_meas: Measurement covariance matrix (optional).
            
        Returns:
            xr.DataArray: pseudoinverse W.
        """
        max_eig = np.max(np.linalg.eigvals(F))
        lambda_meas = self.reg_params.alpha_meas * max_eig
  
        if c_meas is not None:
            
            W = A.values @ np.linalg.inv(F  + lambda_meas * c_meas)
        else:
            W = A.values @ np.linalg.inv(F  + lambda_meas * np.eye(A.shape[1]) )

        W_xr = xr.DataArray(W, dims=("flat_vertex", "measurement"))

        if 'parcel' in A.coords:
            W_xr = W_xr.assign_coords({"parcel" : ("flat_vertex", A.coords['parcel'].values)})
        if 'is_brain' in A.coords:
            W_xr = W_xr.assign_coords({"is_brain": ("flat_vertex", A.coords['is_brain'].values)})

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
            c_meas = c_meas.stack(measurement=("wavelength", "channel")).sortby('wavelength')
            c_meas = np.diag(c_meas)
        return self._calculate_W(A, self._F, c_meas)

    def _calculate_W_mua(self, A, c_meas=None):
        """Calculate pseudoinverse W for mua reconstruction.

        Args:
            A: Sensitivity matrix with wavelength dimension.
            c_meas: Measurement covariance matrix (optional).
            
        Returns:
            xr.DataArray: Pseudoinverse matrix for mua reconstruction with wavelength dimension.
        """
        W = []
        for wavelength in A.wavelength:
            if c_meas is not None:
                c_meas_w = c_meas.sel(wavelength=wavelength)
                c_meas_w = np.diag(c_meas_w)
            else:
                c_meas_w = None

            W_xr = self._calculate_W(A.sel(wavelength=wavelength), self._F.sel(wavelength=wavelength), c_meas_w)
            W.append(W_xr)

        W_xr = xr.concat(W, dim="wavelength")
        W_xr = W_xr.assign_coords({"wavelength": A.wavelength})

        return W_xr
    
    # --- IMAGE RECONSTRUCTION METHODS ---

    def _get_image_conc(self, y: cdt.NDTimeSeries) -> cdt.NDTimeSeries:

        # Detect time dimension
        time_dim = self._get_time_dimension(y)
        has_time = time_dim is not None
        if has_time:
            y = y.transpose('measurement', 'time')

        conc_img = self._W.values @ y.values

        if self.sbf is None:
            # direct recon without spatial basis
            split = len(self._W.flat_vertex)//2
            if has_time:
                conc_img = conc_img.reshape([2, split, conc_img.shape[1]])
            else:                             
                conc_img = conc_img.reshape([2, split])            
        else:
            # direct recon with spatial basis
            conc_img = self.sbf.kernel_to_image_space_conc(conc_img)

        return self._create_conc_dataarray(conc_img, y, time_dim)

    def _get_image_mua(self, y):
        """Compute absorption coefficient image from measurements.
        
        Args:
            y: Optical density measurement data with wavelength dimension.

        Returns:
            xr.DataArray: Absorption coefficient image.
        """

        time_dim = self._get_time_dimension(y)
        
        mua_results = []
        for w in y.wavelength:
            W_w = self._W.sel(wavelength=w)
            y_w = y.sel(wavelength=w) 
            X_w = W_w.values @ y_w.values
            if self.sbf is not None:
                X_w = self.sbf.kernel_to_image_space_mua(X_w)

            mua_results.append(X_w)

        # Combine wavelengths: stack along wavelength axis
        mua_img = np.stack(mua_results, axis=0)  # (wavelength, vertex, [time])

        # Create properly formatted DataArray
        return self._create_mua_dataarray(mua_img, y, time_dim)
    

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
            c_meas = c_meas.transpose('measurement', 'time')
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
            noise_var = self._reshape_conc(noise_var, has_time)
        
        # Create properly formatted xarray
        return self._create_conc_dataarray(noise_var, c_meas, time_dim)
    
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
        noise_var = np.stack(noise_var_list, axis=0)  # (wavelength, vertex, time)

        # Create properly formatted xarray
        return self._create_mua_dataarray(noise_var, c_meas, time_dim)
    
    # --- HELPER METHODS FOR IMAGE COMPUTATION ---
    
    def _get_time_dimension(self, data: xr.DataArray) -> str | None:
        """Detect time dimension in data."""
        for dim in ['time', 'reltime']:
            if dim in data.dims:
                return dim
        return None
    
    def _compute_single_noise(self, W: xr.DataArray, c_meas: xr.DataArray) -> np.ndarray:
        """Compute noise for single timepoint: diag(W @ C @ W.T)."""
        c_diag = np.sqrt(c_meas.values)
        return np.nansum((W.values * c_diag)**2, axis=1)

    def _compute_time_varying_noise(self, W: xr.DataArray, c_meas: xr.DataArray) -> np.ndarray:
        """Compute noise for multiple timepoints efficiently.
        
        This unified method works for both:
        - Full multi-wavelength case (concentration reconstruction)
        - Single wavelength case (mua reconstruction)
        
        Args:
            W: Weight matrix with dims (vertex, measurement) or (vertex, channel)
            c_meas: Covariance with dims (time, measurement) or (time, channel)
            
        Returns:
            np.ndarray: Noise variance with dims (vertex, time)
        """
        # Vectorized computation over time
        c_sqrt = np.sqrt(c_meas.values)  # (time, measurement) or (time, channel)
        W_expanded = W.values[:, :, np.newaxis]  # (vertex, measurement/channel, 1)

        # Broadcasting: (vertex, measurement/channel, time)
        weighted = W_expanded * c_sqrt
        return np.nansum(weighted**2, axis=1)  # (vertex, time)
    
    def _reshape_conc(self, noise_var: np.ndarray, has_time: bool) -> np.ndarray:
        """Reshape noise variance for concentration (HbO/HbR split).
        
        Args:
            noise_var: Noise variance array.
            has_time: Whether time dimension is present.
            
        Returns:
            np.ndarray: Reshaped noise variance with HbO/HbR separation.
        """
        if has_time:
            # Split vertex dimension for HbO/HbR: (vertex, time) -> (2, vertex//2, time)
            n_vertex_half = noise_var.shape[0] // 2
            hbo = noise_var[:n_vertex_half, :]
            hbr = noise_var[n_vertex_half:, :]
            return np.stack([hbo, hbr], axis=0)  # (2, vertex, time)
        else:
            # Split vertex dimension: (vertex,) -> (2, vertex//2)
            n_vertex_half = len(noise_var) // 2
            return noise_var.reshape(2, n_vertex_half)
    
    def _create_conc_dataarray(self, X: np.ndarray, c_meas: xr.DataArray, time_dim: str | None) -> xr.DataArray:
        """Create properly formatted concentration DataArray.
        
        Args:
            X: Concentration data array.
            c_meas: Measurement data for coordinate extraction.
            time_dim: Name of time dimension (if present).
            
        Returns:
            xr.DataArray: Formatted concentration DataArray with proper coordinates.
        """
        if time_dim is not None:
            # (2, vertex, time) 
            dims = ('chromo', 'vertex', time_dim)
            coords = {
                'chromo': ['HbO', 'HbR'],
                time_dim: c_meas[time_dim],
                'samples': (time_dim, np.arange(len(c_meas[time_dim]))),
                'vertex': np.arange(X.shape[1])
            }
        else:
            dims = ('chromo', 'vertex')
            coords = {'chromo': ['HbO', 'HbR'], 
                      'vertex': np.arange(X.shape[1])}

        X = xr.DataArray(X, dims=dims, coords=coords)
        return self._add_spatial_coordinates(X)
    
    def _create_mua_dataarray(self, X: np.ndarray, c_meas: xr.DataArray, time_dim: str | None) -> xr.DataArray:
        """Create properly formatted mua DataArray.
        
        Args:
            X: Mua data array.
            c_meas: Measurement data for coordinate extraction.
            time_dim: Name of time dimension (if present).
            
        Returns:
            xr.DataArray: Formatted mua DataArray with proper coordinates.
        """
        if time_dim is not None:
            dims = ('wavelength', 'vertex', time_dim)
            coords = {
                'wavelength': c_meas.wavelength,
                time_dim: c_meas[time_dim],
                'samples': (time_dim, np.arange(len(c_meas[time_dim]))),
                'vertex': np.arange(X.shape[1])
            }
        else:
            dims = ('wavelength', 'vertex')
            coords = {'wavelength': c_meas.wavelength, 
                      'vertex': np.arange(X.shape[1])}
        
        noise_da = xr.DataArray(X, dims=dims, coords=coords)
        return self._add_spatial_coordinates(noise_da)

    def _add_spatial_coordinates(self, X: xr.DataArray) -> xr.DataArray:
        """Add parcel and is_brain coordinates if available in Adot."""
        if 'parcel' in self.Adot.coords:
            X = X.assign_coords({"parcel": ("vertex", self.Adot.coords['parcel'].values)})
        if 'is_brain' in self.Adot.coords:
            X = X.assign_coords({"is_brain": ("vertex", self.Adot.coords['is_brain'].values)})
        return X



#%%
### use cases:
if __name__ == "__main__":
    import cedalion.dot as dot
    head_model = ...
    Adot = ...
    od = ...

    reco = dot.ImageReco(
        head_model,
        Adot,
        recon_mode="mua*mua2conc",
        regularization_params=dot.REG_TIKHONOV_SPATIAL,
        spatial_basis_functions=dot.SBF_GAUSSIANS_DENSE,
    )

    # or for fine-grained control:

    reco = dot.ImageReco(
        head_model,
        Adot,
        recon_mode="conc",
        regularization_params=dot.RegularizationParams(
            alpha_meas=0.001, alpha_spatial=None, apply_c_meas=True
        ),
        spatial_basis_functions=dot.GaussianSpatialBasisFunctions(
            mask_threshold=-2,
            threshold_brain=1 * units.mm,
            threshold_scalp=5 * units.mm,
            sigma_brain=1 * units.mm,
            sigma_scalp=5 * units.mm,
        ),
    )

    # saving prepared reconstruction object to disk
    reco.to_file("/path/to/cached_reco.h5")

    # loading prepared reconstruction object from disk
    reco = dot.ImageReco.from_file("/path/to/cached_reco.h5")

    conc = reco.reconstruct(od)  # c_meas is internally computed from od

    # or:

    c_meas = ...

    od1 = ...
    od2 = ...
    od3 = ...

    conc = reco.reconstruct(od1, c_meas)  # recomputes W
    conc = reco.reconstruct(od2, c_meas)  # uses cached W
    conc = reco.reconstruct(od3, c_meas)  # uses cached W

    # testing:

    for sbf in [None, SBF_GAUSSIANS_DENSE]:
        for reg_params in [REG_TIKHONOV_ONLY, REG_TIKHONOV_SPATIAL, ...]:
            for recon_mode in ["conc", "mua", "mua*mua2conc"]:
                reco = ImageReco(
                    head_model,
                    Adot,
                    recon_mode=recon_mode,
                    regularization_params=reg_params,
                    spatial_basis_functions=sbf,
                )
                c_meas = ...
                result = reco.reconstruct(od, c_meas)

