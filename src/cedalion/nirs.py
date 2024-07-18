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
        spectrum: The type of spectrum to use. Currently supported options are:
            - "prahl": Extinction coefficients based on the Prahl absorption spectrum
                       (Prahl1998).
        wavelengths: An array-like object containing the wavelengths at which to
            calculate the extinction coefficients.

    Returns:
        xr.DataArray: A matrix of extinction coefficients with dimensions "chromo"
            (chromophore, e.g. HbO/HbR) and "wavelength" (e.g. 750, 850, ...) at which
            the coefficients for each chromophore are given in units of "mm^-1 / M".

    References:
        (Prahl 1998) - taken from Homer2/3, Copyright 2004 - 2006 - The General Hospital
            Corporation and President and Fellows of Harvard University.
            "These values for the molar extinction coefficient e in [cm-1/(moles/liter)]
            were compiled by Scott Prahl (prahl@ece.ogi.edu) using data from
            W. B. Gratzer, Med. Res. Council Labs, Holly Hill, London
            N. Kollias, Wellman Laboratories, Harvard Medical School, Boston
            To convert this data to absorbance A, multiply by the molar concentration
            and the pathlength.
            For example, if x is the number of grams per liter and a 1 cm cuvette is
            being used, then the absorbance is given by
            (e) [(1/cm)/(moles/liter)] (x) [g/liter] (1) [cm]
            A =  ---------------------------------------------------
                        66,500 [g/mole]
            using 66,500 as the gram molecular weight of hemoglobin.
            To convert this data to absorption coefficient in (cm-1), multiply by the
            molar concentration and 2.303,
            µa = (2.303) e (x g/liter)/(66,500 g Hb/mole)
            where x is the number of grams per liter. A typical value of x for whole
            blood is x=150 g Hb/liter."
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


def channel_distances(amplitudes: xr.DataArray, geo: xr.DataArray, dim: int = 3):
    """Calculate distances between channels.

    Args:

        amplitudes (xr.DataArray): A DataArray representing the amplitudes
            with dimensions (channel, *).
        geo (xr.DataArray): A DataArray containing the 2D or 3D coordinates of
            the channels with dimensions (channel, pos).
        dim (int, optional): Geometry dimension, must be 2 or 3. Default 3.

    Returns:
        dists (xr.DataArray): A DataArray containing the calculated distances
            between source and detector channels. The resulting DataArray
            has the dimension 'channel'.
    """
    validators.has_channel(amplitudes)
    validators.has_positions(geo, npos=dim)
    validators.is_quantified(geo)

    diff = geo.loc[amplitudes.source] - geo.loc[amplitudes.detector]
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
    od = -np.log(amplitudes / amplitudes.mean("time"))
    return od


def od2conc(
    od: xr.DataArray,
    geo: xr.DataArray,
    dpf: xr.DataArray,
    spectrum: str = "prahl",
    dim: int = 3
):
    """Calculate concentration changes from optical density data.

    Args:
        od (xr.DataArray, (channel, wavelength, *)): The optical density data array
        geo (xr.DataArray): The 2D or 3D coordinates of the optodes.
        dpf (xr.DataArray, (wavelength, *)): The differential pathlength factor data
        spectrum (str, optional): The type of spectrum to use for calculating extinction
            coefficients. Defaults to "prahl".
        dim (int, optional): Geometry dimension, must be 2 or 3. Default 3.

    Returns:
        conc (xr.DataArray, (channel, wavelength, *)): A data array containing
            concentration changes with dimensions "channel" and "wavelength".
    """
    validators.has_channel(od)
    validators.has_wavelengths(od)
    validators.has_wavelengths(dpf)
    validators.has_positions(geo, npos=dim)

    E = get_extinction_coefficients(spectrum, od.wavelength)

    Einv = xrutils.pinv(E)

    dists = channel_distances(od, geo, dim)
    dists = dists.pint.to("mm")

    # conc = Einv @ (optical_density / ( dists * dpf))
    conc = xr.dot(Einv, od / (dists * dpf), dims=["wavelength"])
    conc = conc.pint.to("micromolar")
    conc = conc.rename("concentration")

    return conc


def beer_lambert(
    amplitudes: xr.DataArray,
    geo: xr.DataArray,
    dpf: xr.DataArray,
    spectrum: str = "prahl",
    dim: int = 3,
):
    """Calculate concentration changes from amplitude using the modified BL law.

    Args:
        amplitudes (xr.DataArray, (channel, wavelength, *)): The input data array containing the raw intensities.
        geo (xr.DataArray): The 2D or 3D coordinates of the optodes.

        dpf (xr.DataArray, (wavelength,*)): The differential pathlength factors
        spectrum (str, optional): The type of spectrum to use for calculating extinction
            coefficients. Defaults to "prahl".
        dim (int, optional): Geometry dimension, must be 2 or 3. Default 3.

    Returns:
        conc (xr.DataArray, (channel, wavelength, *)): A data array containing
            concentration changes according to the mBLL.
    """

    if dim not in [2, 3]:
        raise AttributeError(f"dim must be '2' or '3' but got {dim}")
    else:
        dim = int(dim)

    validators.has_channel(amplitudes)
    validators.has_wavelengths(amplitudes)
    validators.has_wavelengths(dpf)
    validators.has_positions(geo, npos=dim)

    # calculate optical densities
    od = int2od(amplitudes)
    # calculate concentrations
    conc = od2conc(od, geo, dpf, spectrum, dim)

    return conc