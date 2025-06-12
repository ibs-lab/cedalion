"""Solver for the image reconstruction problem."""

import numpy as np
import pint
import xarray as xr
from numpy.typing import ArrayLike
from pathlib import Path
from abc import ABC, abstractmethod
import h5py

import cedalion.xrutils as xrutils
import cedalion.typing as cdt
from dataclasses import dataclass

from cedalion import units


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

    @abstractmethod
    def kernel_to_image_space(self, X):
        pass

    @abstractmethod
    def to_hdf5_group(self, group: h5py.Group):
        pass

    @abstractmethod
    @classmethod
    def from_hdf5_group(cls, group: h5py.Group) -> "SpatialBasisFunctions":
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
        # compute _G
        pass

    def kernel_to_image_space(self, X):
        pass

    def to_hdf5_group(self, group):
        pass

    @classmethod
    def from_hdf5_group(self, group):
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

SBF_GAUSSIANS_SPARSE = GaussianSpatialBasisFunctions(...)


# likewise, if there are any heuristics how to set these parameters, we could offer
# functions to compute them
def estimate_reg_params(*args) -> RegularizationParams:
    pass


class ImageReco:
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

        # cache intermediate matrices to avoid recomputations

        # These invalidate when Adot or reg./sbf. params. change.
        # Depending on recon_mode they have different shapes.
        self._D = None  # Linv^2 * A.T
        self._F = None  # A_hat * A_hat.T

        # self._G = None  # SBF kernel matrices for brain and scalp

        self._mua2conc

        # this invalidates when C_meas changes
        self._W = None  # the pseudo_inverse (W=D@inv(F+ lambda_meas C))
        self._W_input_hash  # a hash of C_meas. if C_meas changes, recompute W

        if self.sbf is not None:
            reduced_Adot = self.sbf.prepare(head_model, Adot)
            self._prepare(head_model, reduced_Adot)
        else:
            self._prepare(head_model, Adot)

    def to_file(self, fname: Path | str):
        """Serialize to disk."""

        with h5py.File(fname, "w") as f:
            # store params
            # store D,F,W, mua2conc

            sbf_group = f.create_group("sbf")

            if self.sbf:
                self.sbf.to_hdf5_group(sbf_group)

    @classmethod
    def from_file(cls, fname: str | Path) -> "ImageReco":
        """Load saved instance from disk."""
        pass

    def _prepare(self, head_model, Adot):
        """Precompute everything that depends only on inputs in the constructor."""

        # calculate D and F for the selected choice of recon_mode and sbf.

        if self.recon_mode == "conc":
            self._D, self._F = self._calculate_DF_conc(Adot)

        elif self.recon_mode in ["mua", "mua*mua2conc"]:
            self._D, self._F = self._calculate_DF_mua(Adot)
        else:
            raise ValueError()  # unreachable

        if self.recon_mode == "mua*mua2conc":
            # calculate _mua2conc
            self._mua2conc = ...

    def _calculate_DF_conc(self):
        pass

    def _calculate_W_conc(self):
        pass

    def _calculate_DF_mua(self):
        pass

    def _calculate_W_mua(self):
        pass

    def reconstruct(
        self,
        time_series: cdt.NDTimeSeries,
        c_meas: xr.DataArray | None = None,
    ) -> cdt.NDTimeSeries:
        if (c_meas is None) and self.reg_params.apply_c_meas:
            # estimate c_meas from time_series
            c_meas = ...

        # calculate hash(c_meas) and (re)compute W if necessary
        W_input_hash = hash(c_meas)

        if (self._W is None) or (W_input_hash != self._W_input_hash):
            self._W = ...  # compute pseudo_inverse
            self._W_input_hash = W_input_hash

        if self.recon_mode == "conc":
            if self.sbf is None:
                # direct recon without spatial basis
                conc_img = ...

            else:
                # direct recon with spatial basis
                conc_kernel = ...
                conc_img = self.sbf.kernel_to_image_space(conc_kernel)

            return conc_img
        elif self.recon_mode in ["mua", "mua*mua2conc"]:
            if self.sbf is None:
                # indirect recon without spatial basis
                mua_img = ...
            else:
                # indirect recon with spatial basis
                mua_kernel = ...
                mua_img = self.sbf.kernel_to_image_space(mua_kernel)

            if self.recon_mode == "mua":
                return mua_img
            else:
                return mua_img @ self.mua2conc
        else:
            raise ValueError()  # unreachable

    def get_image_noise(
        self, time_series: cdt.NDTimeSeries, c_meas: xr.DataArray | None = None
    ):
        pass

    def get_image_noise_tstat(
        self, time_series: cdt.NDTimeSeries, c_meas: xr.DataArray | None = None
    ):
        pass


### use cases:

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


# def pseudo_inverse_stacked(
#    Adot: xr.DataArray,
#    alpha: float = 0.01,
#    Cmeas: ArrayLike | None = None,
#    alpha_spatial: float | None = None,
# ):
#    """Calculate the pseudo-inverse of a stacked sensitivity matrix.
#
#    Args:
#        Adot: Stacked matrix
#        alpha: Thikonov regularization parameter
#        Cmeas: Optional measurement regularization parameters. If specified this can
#         be either a vector of size nchannel or a matrix of size nchannelxnchannel.
#        alpha_spatial: Optional spatial regularization parameter.
#         Suggested default is 1e-3, or 1e-2 when spatial basis functions are used.
#
#    Returns:
#        xr.DataArray: Pseudo-inverse of the stacked matrix.
#    """
#
#    if "units" in Adot.attrs:
#        units = pint.Unit(Adot.attrs["units"])
#        inv_units = (1 / units).units
#    elif Adot.pint.units is not None:
#        inv_units = (1 / Adot.pint.units).units
#        Adot = Adot.pint.dequantify()
#    else:
#        inv_units = pint.Unit("1")
#
#    # do spatial regularization
#    if alpha_spatial is not None:
#        AAtdiag = np.sum((Adot.values**2), axis=0)
#
#        b = AAtdiag.max()
#        lambda_spatial = alpha_spatial * b
#
#        L = np.sqrt(AAtdiag + lambda_spatial)
#        Linv = 1 / L
#        A_hat = Adot.values * Linv[np.newaxis, :]
#        AAt = A_hat @ A_hat.T
#        At = (Linv[:, np.newaxis] ** 2) * Adot.values.T
#    else:  # no spatial regularization
#        AAt = Adot.values @ Adot.values.T
#        AAt = Adot.values @ Adot.values.T
#        At = Adot.values.T
#
#    highest_eigenvalue = np.linalg.eig(AAt)[0][0].real
#    lambda_meas = alpha * highest_eigenvalue
#    if Cmeas is None:
#        B = At @ np.linalg.pinv(AAt + lambda_meas * np.eye(AAt.shape[0]))
#    elif len(Cmeas.shape) == 2:
#        B = At @ np.linalg.inv(AAt + lambda_meas * Cmeas)
#    else:
#        B = At @ np.linalg.inv(AAt + lambda_meas * np.diag(Cmeas))
#
#    coords = xrutils.coords_from_other(Adot)
#
#    # don't copy the MultiIndexes
#    for k in ["flat_channel", "flat_vertex"]:
#        if k in coords:
#            del coords[k]
#
#    B = xr.DataArray(
#        B,
#        dims=("flat_vertex", "flat_channel"),
#        coords=coords,
#        attrs={"units": str(inv_units)},
#    )
#
#    return B
