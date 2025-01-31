"""Functions to create the design matrix for the GLM."""

from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

import cedalion.typing as cdt
import cedalion.xrutils as xrutils

from .basis_functions import TemporalBasisFunction


def make_design_matrix(
    ts_long: cdt.NDTimeSeries,
    ts_short: cdt.NDTimeSeries | None,
    stim: pd.DataFrame,
    geo3d: cdt.LabeledPointCloud,
    basis_function: TemporalBasisFunction,
    drift_order: int | None,
    short_channel_method: str | None,
):
    """Generate the design matrix for the GLM.

    Args:
        ts_long (cdt.NDTimeSeries): Time series of long distance channels.
        ts_short (cdt.NDTimeSeries): Time series of short distance channels.
        stim (DataFrame): Stimulus DataFrame
        geo3d (cdt.LabeledPointCloud): Probe geometry
        basis_function (TemporalBasisFunction): the temporal basis function(s) to model
            the HRF.
        drift_order (int): If not None specify the highest polynomial order of the drift
            terms.
        short_channel_method (str): Specifies the method to add short channel
            information to the design matrix
            Options:
                'closest': Use the closest short channel
                'max_corr': Use the short channel with the highest correlation
                'mean': Use the average of all short channels.

    Returns:
        A tuple containing the global design_matrix and a list of channel-wise
        regressors.
    """

    dm = make_hrf_regressors(ts_long, stim, basis_function)

    if drift_order is not None:
        dm_drift = make_drift_regressors(ts_long, drift_order=drift_order)
        dm = xr.concat([dm, dm_drift], dim="regressor")


    channel_wise_regressors = None

    if (short_channel_method is not None) and (ts_short is None):
        raise ValueError("ts_short may not be None.")

    if short_channel_method == "closest":
        channel_wise_regressors = [closest_short_channel(ts_long, ts_short, geo3d)]
    elif short_channel_method == "max_corr":
        channel_wise_regressors = [max_corr_short_channel(ts_long, ts_short)]
    elif short_channel_method == "mean":
        dm_short = average_short_channel(ts_short)
        dm = xr.concat([dm, dm_short], dim="regressor")

    return dm, channel_wise_regressors


def make_drift_regressors(ts: cdt.NDTimeSeries, drift_order) -> xr.DataArray:
    """Create drift regressors.

    Args:
        ts (cdt.NDTimeSeries): Time series data.
        drift_order (int): The highest polynomial order of the drift terms.

    Returns:
        xr.DataArray: A DataArray containing the drift regressors.
    """
    dim3 = xrutils.other_dim(ts, "channel", "time")
    ndim3 = ts.sizes[dim3]

    nt = ts.sizes["time"]

    drift_regressors = np.ones((nt, drift_order + 1, ndim3))

    for i in range(1, drift_order + 1):
        tmp = np.arange(1, nt + 1, dtype=float) ** (i)
        tmp /= tmp[-1]
        drift_regressors[:, i, 0] = tmp

    for i in range(1, ndim3):
        drift_regressors[:, :, i] = drift_regressors[:, :, 0]

    regressor_names = [f"Drift {i}" for i in range(drift_order + 1)]

    drift_regressors = xr.DataArray(
        drift_regressors,
        dims=["time", "regressor", dim3],
        coords={"time": ts.time, "regressor": regressor_names, dim3: ts[dim3].values},
    )

    return drift_regressors


def pad_time_axis(time: ArrayLike, onsets: ArrayLike):
    min_onset = onsets.min()

    if min_onset < time[0]:
        dt = (time[1:] - time[:-1]).mean()
        pad_before = int(np.ceil(np.abs(min_onset - time[0]) / dt))
        padded_time = np.hstack((np.arange(-pad_before, 0) * dt, time))
    else:
        pad_before = 0
        padded_time = time

    return padded_time, pad_before


def build_stim_array(
    time: ArrayLike, onsets: ArrayLike, durations: None | ArrayLike, values: ArrayLike
) -> np.ndarray:
    """Build an array indicating active stimulus periods.

    The resuting array values are set from values between onset and onset+duration and
    zero everywhere else.

    Args:
        time: the time axis
        onsets: times of stimulus onsets
        durations: either durations of each stimulus or None, in which case the stimulus
            duration is set to one sample.
        values: Stimulus values.

    Returns:
        The array denoting

    """

    smpl_start = time.searchsorted(onsets)
    if durations is None:
        smpl_stop = smpl_start + 1
    else:
        smpl_stop = time.searchsorted(onsets + durations)

    stim = np.zeros_like(time)
    for start, stop, value in zip(smpl_start, smpl_stop, values):
        stim[start:stop] = value

    return stim



def make_hrf_regressors(
    ts: cdt.NDTimeSeries, stim: pd.DataFrame, basis_function: TemporalBasisFunction
):
    """Create regressors modelling the hemodynamic response to stimuli.

    Args:
        ts (NDTimeSeries): Time series data.
        stim (pd.DataFrame): Stimulus DataFrame.
        basis_function (TemporalBasisFunction): TemporalBasisFunction object defining
            the HRF.

    Returns:
        regressors (xr.DataArray): A DataArray containing the regressors.
    """

    # FIXME allow basis_function to be an xarray as returned by basis_function()
    # so that users can pass their own individual hrf function

    trial_types: np.ndarray = stim.trial_type.unique()

    basis = basis_function(ts)

    components = basis.component.values

    # could be "chromo" or "wavelength"
    other_dim = xrutils.other_dim(ts, "channel", "time")

    n_time = ts.sizes["time"]
    n_other = ts.sizes[other_dim]
    n_components = basis.sizes["component"]
    n_trial_types = len(trial_types)
    n_regressors = n_trial_types * n_components

    if other_dim in basis.dims:
        if not set(basis[other_dim].values) == set(ts[other_dim].values):
            raise ValueError(
                f"basis and timeseries don't match in dimension '{other_dim}'"
            )
    else:
        # if the basis function does not contain other_dim (e.g. the same HRF is applied
        # to HbO and HbR), add other_dim by copying the array.
        basis = xr.concat(n_other * [basis], dim=other_dim)
        basis = basis.transpose("time", "component", other_dim)
        basis = basis.assign_coords({other_dim: ts[other_dim]})

    # basis.time may contain time-points before the stimulus onset. To account for this
    # offset in the convolution shift the onset times.
    shifted_stim = stim.copy()
    shifted_stim["onset"] += basis.time.values.min()

    padded_time, pad_before = pad_time_axis(ts.time.values, shifted_stim["onset"])

    if n_components == 1:
        regressor_names = [f"HRF {tt}" for tt in trial_types]
    else:
        regressor_names = [f"HRF {tt} {c}" for tt in trial_types for c in components]

    regressors = np.zeros((n_time, n_regressors, n_other))

    for i_tt, trial_type in enumerate(trial_types):
        tmp = shifted_stim[shifted_stim.trial_type == trial_type]
        if basis_function.convolve_over_duration:
            stim_array = build_stim_array(
                padded_time, tmp["onset"], tmp["duration"], tmp["value"]
            )
        else:
            stim_array = build_stim_array(padded_time, tmp["onset"], None, tmp["value"])

        for i_comp in range(n_components):
            i_reg = i_tt * n_components + i_comp
            for i_other, other in enumerate(ts[other_dim].values):
                bb = basis.sel({other_dim: other})

                # Convolve the basis function with the boxcar stimuls function
                # using 'full' mode, i.e. the resulting regressor is longer than the
                # original time series and needs to be trimmed. Together with the
                # shifted onset times this moves the basis fct. to the correct position.
                regressor = np.convolve(stim_array, bb[:, i_comp])
                regressor = regressor[pad_before : pad_before + n_time]
                regressor /= regressor.max()

                regressors[:, i_reg, i_other] = regressor

    regressors = xr.DataArray(
        regressors,
        dims=["time", "regressor", other_dim],
        coords={
            "time": ts.time.values,
            "regressor": regressor_names,
            other_dim: ts[other_dim].values,
        },
    )

    # hrf_regs = hrf_regs.pint.quantify("micromolar")

    return regressors


def _regressors_from_selected_short_channels(
    ts_long: cdt.NDTimeSeries,
    ts_short: cdt.NDTimeSeries,
    selected_short_ch_indices: np.ndarray,
) -> xr.DataArray:
    # pick for each long channel from ts_short the selected closest channel
    # regressors has same dims as ts_long/ts_short and same channels as ts_long
    regressors = ts_short.isel(channel=selected_short_ch_indices)

    # coords 'channel' in regressors denotes the short channels selected from ts_short.
    # rearrange: 'channel' should name the channel for which each regressor should
    # be used. 'short_channel' names the assigned short channel.
    # Furthermore define 'computational groups', blocks of channels for which the GLM
    # can be solved together, because they have the same channel-wise regressor.
    # assign this as a additional coordinate to the channel dim.

    coords_short_channels = regressors.channel

    dim3 = xrutils.other_dim(ts_long, "channel", "time")

    keep_coords = ["time", "samples", dim3]
    drop_coords = [i for i in regressors.coords.keys() if i not in keep_coords]
    regressors = regressors.drop_vars(drop_coords)
    regressors = regressors.assign_coords(
        {
            "channel": ("channel", ts_long.channel.values),
            "short_channel": ("channel", coords_short_channels.channel.values),
            "comp_group": ("channel", selected_short_ch_indices),
        }
    )
    regressors = regressors.expand_dims({"regressor": 1})
    regressors = regressors.assign_coords({"regressor": ["short"]})

    return regressors


def closest_short_channel(
    ts_long: cdt.NDTimeSeries, ts_short: cdt.NDTimeSeries, geo3d: cdt.LabeledPointCloud
):
    """Create channel-wise regressors use closest nearby short channel.

    Args:
        ts_long (NDTimeSeries): Time series of long channels
        ts_short (NDTimeSeries): Time series of short channels
        geo3d (LabeledPointCloud): Probe geometry

    Returns:
        regressors (xr.DataArray): Channel-wise regressor
    """
    # calculate midpoints between channel optode pairs. dims: (channel, crs)
    long_channel_pos = (geo3d.loc[ts_long.source] + geo3d.loc[ts_long.detector]) / 2
    short_channel_pos = (geo3d.loc[ts_short.source] + geo3d.loc[ts_short.detector]) / 2

    # to select the smallest value units are not necassry
    long_channel_pos = long_channel_pos.pint.dequantify().values
    short_channel_pos = short_channel_pos.pint.dequantify().values

    # create a (nlong x nshort x 3) array with vectors between all l&s channels
    dist_vectors = long_channel_pos[:, None, :] - short_channel_pos[None, :, :]

    # calculate distances and find for each long channel the closest short channel
    dists = np.linalg.norm(dist_vectors, axis=-1)
    closest_short_ch_indices = np.argmin(dists, axis=1)  # size nlong

    regressors = _regressors_from_selected_short_channels(
        ts_long, ts_short, closest_short_ch_indices
    )

    return regressors


def max_corr_short_channel(ts_long: cdt.NDTimeSeries, ts_short: cdt.NDTimeSeries):
    """Create channel-wise regressors using the most correlated short channels.

    For each long channel the short channel is selected that has the highest
    correleation coefficient in any wavelength or chromophore.

    Args:
        ts_long (NDTimeSeries): time series of long channels
        ts_short (NDTimeSeries): time series of short channels

    Returns:
        xr.DataArray: channel-wise regressors
    """

    dim3 = xrutils.other_dim(ts_long, "channel", "time")

    z_long = (ts_long - ts_long.mean("time")) / ts_long.std("time")
    z_short = (ts_short - ts_short.mean("time")) / ts_short.std("time")

    # calculate correleation coefficients between all long and short channels
    # dims: (chromo|wavelength, ch_long, ch_short)
    corr_coeff = (
        xr.dot(
            z_long.rename({"channel": "ch_long"}),
            z_short.rename({"channel": "ch_short"}),
            dim="time",
        )
        / z_long.sizes["time"]
    )

    # for each long channel find the short channel with the highest correlation
    # in *any* chromophore/wavelength. Makes sure that different chromophores
    # don't get different short channels assigned.

    max_corr_short_ch_indices = corr_coeff.max(dim3).argmax("ch_short").values

    regressors = _regressors_from_selected_short_channels(
        ts_long, ts_short, max_corr_short_ch_indices
    )

    return regressors

def average_short_channel(ts_short: cdt.NDTimeSeries):
    """Create a regressor by averaging all short channels.

    Args:
        ts_short (NDTimeSeries): time series of short channels

    Returns:
        xr.DataArray: regressors
    """

    ts_short = ts_short.pint.dequantify()
    regressor = ts_short.mean("channel").expand_dims("regressor")
    regressor = regressor.assign_coords({"regressor": ["short"]})
    regressor = regressor.transpose("time", "regressor", ...)

    return regressor
