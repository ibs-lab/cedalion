import numpy as np
import xarray as xr
from numpy.typing import ArrayLike
from scipy.interpolate import interp1d

import cedalion.validators as validators
import cedalion.xrutils as xrutils
from importlib import resources


def get_extinction_coefficients(spectrum: str, wavelengths: ArrayLike):
    """Provide a matrix of extinction coefficients from tabulated data.

    Args:
        spectrum (str): The type of spectrum to use. Currently supported options are:
            - "prahl": Extinction coefficients based on the Prahl absorption spectrum.
        wavelengths (ArrayLike): An array-like object containing the wavelengths at which
            to calculate the extinction coefficients.

    Returns:
        xr.DataArray: A matrix of extinction coefficients with dimensions "chromo"
        (chromophore) and "wavelength", where "chromo" represents the type of chromophore
        (e.g., "HbO", "HbR") and "wavelength" represents the wavelengths at which the
        extinction coefficients are calculated. The returned data array is annotated
        with units of "mm^-1 / M".

    References:
        (:cite:t:`Prahl1998`)
    """
    if spectrum == "prahl":
        with resources.open_text(
            "cedalion.data", "prahl_absorption_spectrum.tsv"
        ) as fin:
            coeffs = np.loadtxt(fin, comments="#")

        chromophores = ["HbO", "HbR"]
        spectra = [
            interp1d(coeffs[:, 0], np.log(10) * coeffs[:, i] / 10) for i in [1, 2]
        ]  # convert units from cm^-1/ M to mm^-1 / M

        E = np.array([spec(wl) for spec in spectra for wl in wavelengths]).reshape(
            len(spectra), len(wavelengths)
        )

        E = xr.DataArray(
            E,
            dims=["chromo", "wavelength"],
            coords={"chromo": chromophores, "wavelength": wavelengths},
            attrs={"units": "mm^-1 / M"},
        )
        E = E.pint.quantify()
        return E
    else:
        raise ValueError(f"unsupported spectrum '{spectrum}'")


def channel_distances(amplitudes: xr.DataArray, geo3d: xr.DataArray):
    """Calculate distances between channels.

    Args:
        amplitudes (xr.DataArray): A DataArray representing the amplitudes with dimensions (channel, *).
        geo3d (xr.DataArray): A DataArray containing the 3D coordinates of the channels with dimensions (channel, pos).

    Returns:
        dists (xr.DataArray): A DataArray containing the calculated distances between source and detector channels. 
            The resulting DataArray has the dimension 'channel'.
    """
    validators.has_channel(amplitudes)
    validators.has_positions(geo3d, npos=3)
    validators.is_quantified(geo3d)

    diff = geo3d.loc[amplitudes.source] - geo3d.loc[amplitudes.detector]
    dists = xrutils.norm(diff, "pos")
    dists = dists.rename("dists")

    return dists


def int2od(amplitudes: xr.DataArray):
    """Calculate optical density from intensity amplitude  data.

    Args:
        amplitudes (xr.DataArray, (time, channel, *)): amplitude data.

    Returns:
        od: (xr.DataArray, (time, channel,*): The optical density data.
    """
    od = - np.log( amplitudes / amplitudes.mean("time") )
    return od


def od2conc(
    od: xr.DataArray,
    geo3d: xr.DataArray,
    dpf: xr.DataArray,
    spectrum: str = "prahl",
):
    """Calculate concentration changes from optical density data.

    Args:
        od (xr.DataArray, (channel, wavelength, *)): The optical density data array
        geo3d (xr.DataArray): The 3D coordinates of the optodes.
        dpf (xr.DataArray, (wavelength, *)): The differential pathlength factor data
        spectrum (str, optional): The type of spectrum to use for calculating extinction
            coefficients. Defaults to "prahl".

    Returns:
        conc: A data array containing concentration changes with dimensions
        "channel" and "wavelength".
    """
    validators.has_channel(od)
    validators.has_wavelengths(od)
    validators.has_wavelengths(dpf)
    validators.has_positions(geo3d, npos=3)

    E = get_extinction_coefficients(spectrum, od.wavelength)

    Einv = xrutils.pinv(E)

    dists = channel_distances(od, geo3d)
    dists = dists.pint.to("mm")

    # conc = Einv @ (optical_density / ( dists * dpf))
    conc = xr.dot(Einv, od / (dists * dpf), dims=["wavelength"])
    conc = conc.pint.to("micromolar")
    conc = conc.rename("concentration")

    return conc


def beer_lambert(
    amplitudes: xr.DataArray,
    geo3d: xr.DataArray,
    dpf: xr.DataArray,
    spectrum: str = "prahl",
):
    """Calculate concentration changes from amplitude data using the modified beer-lambert law.

    Args:
        amplitudes (xr.DataArray, (channel, wavelength, *)): The input data array containing the raw intensities.
        geo3d (xr.DataArray): The 3D coordinates of the optodes.
        dpf (xr.DataArray, (wavelength,*)): The differential pathlength factors
        spectrum (str, optional): The type of spectrum to use for calculating extinction
            coefficients. Defaults to "prahl".

    Returns:
        conc (xr.DataArray, (channel, wavelength, *)): A data array containing concentration 
            changes according to the mBLL
    """
    validators.has_channel(amplitudes)
    validators.has_wavelengths(amplitudes)
    validators.has_wavelengths(dpf)
    validators.has_positions(geo3d, npos=3)

    # calculate optical densities
    od = int2od(amplitudes)
    # calculate concentrations
    conc = od2conc(od, geo3d, dpf, spectrum)

    return conc
