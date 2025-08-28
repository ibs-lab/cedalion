"""Functions to create the design matrix for the GLM.
Modifications for Image Space (Parcel/Vertex) Compatibility
-----------------------------------------------------------

This script extends Cedalion functions originally designed for channel space,
allowing them to also support image space data such as parcel-level or vertex-level time series.

Key changes include:
- Added flexible handling of spatial dimensions using:
    spatial_dim = xrutils.other_dim(ts, "time", "chromo")
  This enables the code to automatically detect and operate over "parcel", "vertex", or "channel" dimensions.
- Replaced hardcoded references to "channel" with dynamic spatial dimension references (e.g., [spatial_dim].values),
  ensuring compatibility with parcel-level and vertex-level data.
- Added a check for "samples" coordinate in the input time series.
  If missing, it assigns a default sample index via:
      ts = ts.assign_coords(samples=("time", np.arange(ts.sizes["time"])))
  This ensures compatibility with downstream functions that expect a "samples" coordinate,
  particularly during design matrix construction and plotting routines.
  Additionally, the "time" coordinate is explicitly labeled with units ("s") for clarity.
- Added assertions in all short-channel regressor functions 
  ("closest_short_channel_regressor", "max_corr_short_channel_regressor", 
  and "average_short_channel_regressor") to ensure they are only used in channel space.
- Added a new helper function "global_mean_regressor" that creates a global signal regressor 
  by averaging over the spatial dimension (works for channel, parcel, or vertex space).
- (Suggested improvement) In __repr__, add a check for "self.common is not None" 
  to make the representation robust when only channel-wise regressors are present.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from numpy.polynomial.legendre import legval

import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion.sigproc.frequency import sampling_rate

from .basis_functions import TemporalBasisFunction


@dataclass
class DesignMatrix:
    common : xr.DataArray | None = None
    channel_wise : list[xr.DataArray] = field(default_factory=list)

    @property
    def regressors(self):
        result = []
        if self.common is not None:
            result.extend([str(r) for r in self.common.regressor.values])
        if self.channel_wise:
            result.extend(
                str(r) for cw in self.channel_wise for r in cw.regressor.values
            )

        return result

    def __repr__(self):
        uregs = ",".join([f"'{r}'" for r in self.common.regressor.values])
        cwregs = ",".join(
            [f"'{r}'" for cw in self.channel_wise for r in cw.regressor.values]
        )
        return f"DesignMatrix(universal=[{uregs}], channel_wise=[{cwregs}])"


    def __and__(self, other: DesignMatrix):
        our_regressors = set(self.regressors)
        for reg in other.regressors:
            if reg in our_regressors:
                raise ValueError(
                    "Cannot concatenate design matrices. "
                    f"Both contain the regressor '{reg}'."
                )

        if (self.common is not None) and (other.common is not None):
            common=xr.concat([self.common, other.common], dim="regressor")
        elif (self.common is not None) and (other.common is None):
            common = self.common.copy()
        elif (self.common is None) and (other.common is not None):
            common = other.common.copy()
        elif (self.common is None) and (other.common is None):
            common = None

        return DesignMatrix(
            common=common,
            channel_wise=self.channel_wise + other.channel_wise
        )


    def iter_computational_groups(
        self,
        ts: cdt.NDTimeSeries,
        channel_groups: list[int] | None = None,
    ):
        """Combine universal and channel-wise regressors and yield a DM for each group.

        Args:
            ts: The time series to be modeled.
            channel_groups: Optional list of channel groups.

        Yields:
            A tuple containing:
                - dim3 (str): The third dimension name.
                - group_y (cdt.NDTimeSeries): The grouped time series.
                - "unit_values" (np.ndarray): Values of the spatial dimension 
                  (e.g., "channel", "parcel", or "vertex") that belong to the current computational group.
                - group_design_matrix (xr.DataArray): The grouped design matrix.
        """

        channel_wise_regressors = self.channel_wise

        dim3_name = xrutils.other_dim(self.common, "time", "regressor")

        spatial_dim = xrutils.other_dim(ts, "time", "chromo")

        for cwreg in self.channel_wise:
            assert cwreg.sizes["regressor"] == 1
            assert (ts[spatial_dim].values == cwreg[spatial_dim].values).all()

        comp_groups = []
        for reg in self.channel_wise:
            if "comp_group" in reg.coords:
                comp_groups.append(reg["comp_group"].values)
            else:
                comp_groups.append(_hash_channel_wise_regressor(reg,spatial_dim))

        if channel_groups is not None:
            assert len(channel_groups) == ts.sizes[spatial_dim]
            comp_groups.append(channel_groups)

        if len(comp_groups) == 0:
            # There are no channel-wise regressors. Just iterate over the third dim.
            # of the design matrix.
            for dim3 in self.common[dim3_name].values:
                dm = self.common.sel({dim3_name: dim3})
                # group_y = ts.sel({dim3_name: dim3})
                values = ts[spatial_dim].values
                # yield dim3, group_y, dm
                yield dim3, values, dm

            return
        else:
            # there are channel-wise regressors. For each computational group, in which
            # the channel-wise regressors are identical, we have to assemble and yield
            # the design-matrix.

            idx_with_same_comp_group = defaultdict(list)

            for i_unit, all_comp_groups in enumerate(zip(*comp_groups)):
                idx_with_same_comp_group[all_comp_groups].append(i_unit)

            for dim3 in self.common[dim3_name].values:
                dm = self.common.sel({dim3_name: dim3})

                for chan_indices in idx_with_same_comp_group.values():
                    unit_values = ts[spatial_dim][np.asarray(chan_indices)].values

                    regs = []
                    for reg in channel_wise_regressors:
                        regs.append(
                            reg.sel({spatial_dim: unit_values, dim3_name: dim3})
                            .isel(spatial_dim=0)  # regs are identical within a group
                            .pint.dequantify()
                        )

                    group_design_matrix = xr.concat([dm] + regs, dim="regressor")

                    # yield dim3, group_y, group_design_matrix
                    yield dim3, unit_values, group_design_matrix


def _hash_channel_wise_regressor(regressor: xr.DataArray) -> list[int]:
    """Hashes each unit slice of the regressor array along the spatial dimension.

    Args:
        regressor: array of regressors. Dims
            (spatial_dim, regressor, time, chromo|wavelength)

    Returns:
        A list of hash values, one for each element of the spatial dimension.
    """
    
    spatial_dim = xrutils.other_dim(regressor, "time", "chromo")

    tmp = regressor.pint.dequantify()
    n_channel = regressor.sizes[spatial_dim]
    
    return [hash(tmp.isel({spatial_dim: i}).values.data.tobytes()) for i in range(n_channel)]


def hrf_regressors(
    ts: cdt.NDTimeSeries, stim: pd.DataFrame, basis_function: TemporalBasisFunction
) -> DesignMatrix:
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
    if "samples" not in ts.coords:
        ts = ts.assign_coords(samples=("time", np.arange(ts.sizes["time"])))
        ts.time.attrs["units"] = "s"
        
    basis = basis_function(ts)

    components = basis.component.values
    spatial_dim = xrutils.other_dim(ts, "time", "chromo")

    # could be "chromo" or "wavelength"
    other_dim = xrutils.other_dim(ts, spatial_dim, "time")

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

    padded_time, pad_before = _pad_time_axis(ts.time.values, shifted_stim["onset"])

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

    return DesignMatrix(common=regressors, channel_wise=[])


def drift_regressors(ts: cdt.NDTimeSeries, drift_order) -> DesignMatrix:
    """Create drift regressors.

    Args:
        ts (cdt.NDTimeSeries): Time series data.
        drift_order (int): The highest polynomial order of the drift terms.

    Returns:
        xr.DataArray: A DataArray containing the drift regressors.
    """
    spatial_dim = xrutils.other_dim(ts, "time", "chromo") 
    dim3 = xrutils.other_dim(ts, spatial_dim, "time")
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

    return DesignMatrix(common=drift_regressors, channel_wise=[])


def drift_legendre_regressors(ts : cdt.NDTimeSeries, order : int) -> DesignMatrix:
    """Create drift regressors using Legende polynomials.

    Args:
        ts: time-series data.
        order: generate polynomials of order 0...order

    Returns:
        xr.DataArray: A DataArray containing the drift regressors.
    """

    spatial_dim = xrutils.other_dim(ts, "time", "chromo") 
    dim3 = xrutils.other_dim(ts, spatial_dim, "time")
    ndim3 = ts.sizes[dim3]

    nt = ts.sizes["time"]
    t = np.linspace(-1, 1, nt)
    drift_regressors = np.zeros((nt, order + 1, ndim3))

    # for coefficients c with length n+1 legval calculates
    # p(x) = c[0]*L0(x) + c[1]*L1(x) + ... + c[n]*Ln(x)
    # we want only Ln(x)

    for i in range(0, order + 1):
        coeffs = [0]*i + [1]
        tmp = np.sqrt((2 * i + 1) / 2) * legval(t, coeffs)
        tmp /= np.max(np.abs(tmp))
        drift_regressors[:, i, :] = tmp[:,None]

    regressor_names = [f"Drift LP {i}" for i in range(order + 1)]

    drift_regressors = xr.DataArray(
        drift_regressors,
        dims=["time", "regressor", dim3],
        coords={"time": ts.time, "regressor": regressor_names, dim3: ts[dim3].values},
    )

    return DesignMatrix(common=drift_regressors, channel_wise=[])


def drift_cosine_regressors(ts: cdt.NDTimeSeries, fmax: cdt.QFrequency) -> DesignMatrix:
    """Create drift regressors using cosine basis functions.

    Args:
        ts: time-series data.
        fmax: High-pass cutoff frequency

    Returns:
        xr.DataArray: A DataArray containing the drift regressors.
    """
    spatial_dim = xrutils.other_dim(ts, "time", "chromo") 
    dim3 = xrutils.other_dim(ts, spatial_dim, "time")
    ndim3 = ts.sizes[dim3]

    nt = ts.sizes["time"]
    fs = sampling_rate(ts)
    ncosines = int(np.floor(2 * nt * fmax / fs))

    drift_regressors = np.zeros((nt, ncosines, ndim3))

    tt = np.pi * (np.arange(nt) + 0.5) / nt

    for i in range(ncosines):
        drift_regressors[:, i, :] = np.cos(tt * i)[:,None]

    regressor_names = [f"Drift Cos {i}" for i in range(ncosines)]

    drift_regressors = xr.DataArray(
        drift_regressors,
        dims=["time", "regressor", dim3],
        coords={"time": ts.time, "regressor": regressor_names, dim3: ts[dim3].values},
    )

    return DesignMatrix(common=drift_regressors, channel_wise=[])



def _pad_time_axis(time: ArrayLike, onsets: ArrayLike):
    """Make sure that time axis includes the earliest stimulus onset."""
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
        # when durations are provided make sure that a trial is at least one sample
        # wide
        too_short_mask = smpl_stop == smpl_start
        smpl_stop[too_short_mask] = smpl_start[too_short_mask] + 1

    stim = np.zeros_like(time)
    for start, stop, value in zip(smpl_start, smpl_stop, values):
        stim[start:stop] = value

    return stim


def _regressors_from_selected_short_channels(
    ts_long: cdt.NDTimeSeries,
    ts_short: cdt.NDTimeSeries,
    selected_short_ch_indices: np.ndarray,
) -> xr.DataArray:
    """Build channel-wise short-channel regressors from a selection."""
    
    spatial_dim = xrutils.other_dim(ts_long, "time", "chromo")
    assert spatial_dim == "channel", (
        f"Short-channel regressors only make sense in channel space, "
        f"but got '{spatial_dim}'."
    )

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


def closest_short_channel_regressor(
    ts_long: cdt.NDTimeSeries, ts_short: cdt.NDTimeSeries, geo3d: cdt.LabeledPointCloud
):
    """Create channel-wise regressors using the closest nearby short channel.

    Args:
        ts_long (NDTimeSeries): Time series of long channels
        ts_short (NDTimeSeries): Time series of short channels
        geo3d (LabeledPointCloud): Probe geometry

    Returns:
        regressors (xr.DataArray): Channel-wise regressor
    """
    spatial_dim = xrutils.other_dim(ts_long, "time", "chromo")
    assert spatial_dim == "channel", (
        f"Short-channel regressors only make sense in channel space, "
        f"but got '{spatial_dim}'."
    )
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

    return DesignMatrix(common=None, channel_wise=[regressors])


def max_corr_short_channel_regressor(
    ts_long: cdt.NDTimeSeries, ts_short: cdt.NDTimeSeries
):
    """Create channel-wise regressors using the most correlated short channels.

    For each long channel the short channel is selected that has the highest
    correleation coefficient in any wavelength or chromophore.

    Args:
        ts_long (NDTimeSeries): time series of long channels
        ts_short (NDTimeSeries): time series of short channels

    Returns:
        xr.DataArray: channel-wise regressors
    """
    spatial_dim = xrutils.other_dim(ts_long, "time", "chromo")
    assert spatial_dim == "channel", (
        f"Short-channel regressors only make sense in channel space, "
        f"but got '{spatial_dim}'."
    )
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

    return DesignMatrix(common=None, channel_wise=[regressors])


def average_short_channel_regressor(ts_short: cdt.NDTimeSeries):
    """Create a regressor by averaging all short channels.

    Args:
        ts_short (NDTimeSeries): time series of short channels

    Returns:
        xr.DataArray: regressors
    """
    spatial_dim = xrutils.other_dim(ts_short, "time", "chromo")
    assert spatial_dim == "channel", (
        f"Average short-channel regressor only makes sense in channel space, "
        f"but got '{spatial_dim}'."
    )
    ts_short = ts_short.pint.dequantify()
    regressor = ts_short.mean("channel", skipna=True).expand_dims("regressor")
    regressor = regressor.assign_coords({"regressor": ["short"]})
    regressor = regressor.transpose("time", "regressor", ...)

    return DesignMatrix(common=regressor, channel_wise=[])
def global_mean_regressor(ts: cdt.NDTimeSeries) -> DesignMatrix:
    """Create a global regressor by averaging over the spatial dimension.

    Args:
        ts (NDTimeSeries): time series data (time x spatial_dim x chromo)

    Returns:
        DesignMatrix: design matrix with one global regressor
    """
    # detect the spatial dimension dynamically (channel, parcel, or vertex)
    spatial_dim = xrutils.other_dim(ts, "time", "chromo")

    # mean over spatial dimension → global signal
    regressor = ts.mean(spatial_dim, skipna=True).expand_dims("regressor")
    regressor = regressor.assign_coords({"regressor": ["global"]})
    regressor = regressor.transpose("time", "regressor", ...)

    return DesignMatrix(common=regressor, channel_wise=[])