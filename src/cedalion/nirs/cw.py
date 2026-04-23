"""Functionality for continous wave NIRS."""

from __future__ import annotations

import numpy as np
import xarray as xr

import cedalion
import cedalion.data
import cedalion.typing as cdt
import cedalion.validators as validators
import cedalion.xrutils as xrutils
from cedalion import units
from .common import get_extinction_coefficients, channel_distances

def int2od(amplitudes: cdt.NDTimeSeries, return_baseline: bool = False):
    """Calculate optical density from intensity amplitude  data.

    Args:
        amplitudes (xr.DataArray, (time, channel, *)): amplitude data.
        return_baseline (bool, optional): If True, also return the baseline data
            used for OD conversion (useful to get back to intensity). Defaults to False.

    Returns:
        od: (xr.DataArray, (time, channel,*): The optical density data.
        baseline: (xr.DataArray, (channel, *)): The intensity baseline data
         (average time series) used for conversion to DO.
    """
    # check negative values in amplitudes and issue an error if yes
    if np.any(amplitudes <= 0):
        raise AssertionError(
            "Error: DataArray contains negative values. Please fix, for example by "
            "setting them to NaN with "
            "'amplitudes = amplitudes.where(amplitudes >= 0, np.nan)'"
        )

    # calculate baseline
    baseline = amplitudes.mean("time")

    # conversion to optical density
    od = -np.log(amplitudes / baseline)

    if return_baseline:
        return od, baseline
    else:
        return od


def od2int(od: cdt.NDTimeSeries, baseline: cdt.NDTimeSeries):
    """Recover intensity amplitude data from optical density data.

    Args:
        od (xr.DataArray, (time, channel, *)): The optical density data.
        baseline (xr.DataArray, (channel, *)): The intensity baseline data
            (average time series) that was used for conversion to DO.

    Returns:
        amplitudes (xr.DataArray, (time, channel, *)): The amplitude data.
    """
    return baseline * np.exp(-od)


def od2conc(
    od: cdt.NDTimeSeries,
    geo3d: cdt.LabeledPoints,
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
        conc (xr.DataArray, (channel, *)): A data array containing
        concentration changes by channel.
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
    if dpf[0] != 1:
        conc = xr.dot(Einv, od / (dists * dpf), dim=["wavelength"])
    else:
        conc = xr.dot(Einv, od / (dpf * 1*units.mm), dim=["wavelength"])

    conc = conc.pint.to("micromolar")
    conc = conc.pint.quantify({"time": od.time.attrs["units"]})  # got lost in xr.dot
    conc = conc.rename("concentration")

    return conc


def conc2od(
    conc: cdt.NDTimeSeries,
    geo3d: cdt.LabeledPoints,
    dpf: xr.DataArray,
    spectrum: str = "prahl",
):
    """Calculate optical density data from concentration changes.

    Args:
        conc (xr.DataArray, (channel, *)): The concentration changes by channel.
        geo3d (xr.DataArray): The 3D coordinates of the optodes.
        dpf (xr.DataArray, (wavelength, *)): The differential pathlength factor data.
        spectrum (str, optional): The type of spectrum to use for calculating extinction
            coefficients. Defaults to "prahl".

    Returns:
        od (xr.DataArray, (channel, wavelength, *)): A data array containing
            optical density data.
    """

    conc = conc.pint.to("molar")

    # Get the extinction coefficients for the chosen spectrum
    wavelengths = dpf.wavelength.values.astype(float)
    E = cedalion.nirs.get_extinction_coefficients(spectrum, wavelengths)

    # Calculate distances between optodes for each channel
    dists = cedalion.nirs.channel_distances(conc, geo3d)
    dists = dists.pint.to("mm")
    if dpf[0] != 1:
        od = xr.dot(E, conc, dim=["chromo"]) * (dists * dpf)
    else:
        od = xr.dot(E, conc, dim=["chromo"]) * (1*units.mm * dpf)

    od = od.rename("optical_density")

    if "time" in od.dims:
        od = od.pint.quantify({"time": "s"})

    return od


def beer_lambert(
    amplitudes: cdt.NDTimeSeries,
    geo3d: cdt.LabeledPoints,
    dpf: xr.DataArray,
    spectrum: str = "prahl",
):
    """Calculate concentration changes from amplitude using the modified BL law.

    Args:
        amplitudes (xr.DataArray, (channel, wavelength, *)): The input data array
            containing the raw intensities.
        geo3d (xr.DataArray): The 3D coordinates of the optodes.
        dpf (xr.DataArray, (wavelength,*)): The differential pathlength factors
        spectrum (str, optional): The type of spectrum to use for calculating extinction
            coefficients. Defaults to "prahl".

    Returns:
        conc (xr.DataArray, (channel, *)): A data array containing
            concentration changes according to the mBLL.
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


