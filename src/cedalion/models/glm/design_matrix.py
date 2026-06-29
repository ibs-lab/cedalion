"""Functions to create the design matrix for the GLM."""

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
import cedalion.dataclasses as cdc
from cedalion import cite


from cedalion.sigproc.frequency import sampling_rate
from cedalion.sigproc.physio import (compute_Hglobal_from_PCA,
                                     get_spatial_smoothing_kernel,
                                     global_component_subtract,
                                     )

from .basis_functions import TemporalBasisFunction



@dataclass
class DesignMatrix:
    common : xr.DataArray | None = None
    channel_wise : list[xr.DataArray] = field(default_factory=list)

    @property
    def regressors(self):
        """List of all regressor names in this design matrix (common + channel-wise).

        Returns:
            List of regressor name strings in order: common first, then channel-wise.
        """
        result = []
        if self.common is not None:
            result.extend([str(r) for r in self.common.regressor.values])
        if self.channel_wise:
            result.extend(
                str(r) for cw in self.channel_wise for r in cw.regressor.values
            )

        return result

    # def __repr__(self):
    #     cregs = ",".join([f"'{r}'" for r in self.common.regressor.values])
    #     cwregs = ",".join(
    #         [f"'{r}'" for cw in self.channel_wise for r in cw.regressor.values]
    #     )
    #     return f"DesignMatrix(common=[{cregs}], channel_wise=[{cwregs}])"

    def __repr__(self):
        """Return a compact string representation listing common and channel-wise regressors."""  # noqa: E501
        cregs = (
            ",".join([f"'{r}'" for r in self.common.regressor.values])
            if self.common is not None
            else ""
        )
        cwregs = ",".join(
            [f"'{r}'" for cw in self.channel_wise for r in cw.regressor.values]
        )
        return f"DesignMatrix(common=[{cregs}], channel_wise=[{cwregs}])"


    def __and__(self, other: DesignMatrix):
        """Concatenate two design matrices (``dm1 & dm2``).

        Merges common regressors by concatenating along the ``"regressor"``
        dimension and appends the channel-wise regressor lists.

        Args:
            other: Design matrix to append to this one.

        Returns:
            New :class:`DesignMatrix` combining both inputs.

        Raises:
            ValueError: If the two matrices share any regressor name.
        """
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

    # def copy(self):
    #     """Create a copy of this design matrix."""

    #     return DesignMatrix(
    #         common=self.common.copy(),
    #         channel_wise=[i.copy() for i in self.channel_wise]
    #     )

    def copy(self):
        """Create a copy of this design matrix."""

        return DesignMatrix(
            common=self.common.copy() if self.common is not None else None,
            channel_wise=[i.copy() for i in self.channel_wise]
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
                - group_design_matrix (xr.DataArray): The grouped design matrix.
        """

        channel_wise_regressors = self.channel_wise

        dim3_name = xrutils.other_dim(self.common, "time", "regressor")

        spatial_dim = cdc.get_spatial_dimension(ts)

        for cwreg in self.channel_wise:
            assert cwreg.sizes["regressor"] == 1
            assert (ts[spatial_dim].values == cwreg[spatial_dim].values).all()

        comp_groups = []
        for reg in self.channel_wise:
            if "comp_group" in reg.coords:
                comp_groups.append(reg["comp_group"].values)
            else:
                comp_groups.append(_hash_channel_wise_regressor(reg))

        if channel_groups is not None:
            assert len(channel_groups) == ts.sizes[spatial_dim]
            comp_groups.append(channel_groups)

        if len(comp_groups) == 0:
            # There are no channel-wise regressors. Just iterate over the third dim.
            # of the design matrix.
            for dim3 in self.common[dim3_name].values:
                dm = self.common.sel({dim3_name: dim3})
                # group_y = ts.sel({dim3_name: dim3})
                spatial_dim_values = ts[spatial_dim].values
                # yield dim3, group_y, dm
                yield dim3, spatial_dim_values, dm

            return
        else:
            # there are channel-wise regressors. For each computational group, in which
            # the channel-wise regressors are identical, we have to assemble and yield
            # the design-matrix.

            spatial_idx_with_same_comp_group = defaultdict(list)

            for i_spatial, all_comp_groups in enumerate(zip(*comp_groups)):
                spatial_idx_with_same_comp_group[all_comp_groups].append(i_spatial)

            for dim3 in self.common[dim3_name].values:
                dm = self.common.sel({dim3_name: dim3})

                for chan_indices in spatial_idx_with_same_comp_group.values():
                    chan_indices = np.asarray(chan_indices)
                    spatial_dim_values = ts[spatial_dim][chan_indices].values

                    regs = []
                    for reg in channel_wise_regressors:
                        regs.append(
                            reg.sel({spatial_dim: spatial_dim_values, dim3_name: dim3})
                            .isel({spatial_dim: 0})  # regs are identical within a group
                            .pint.dequantify()
                        )

                    group_design_matrix = xr.concat([dm] + regs, dim="regressor")

                    # yield dim3, group_y, group_design_matrix
                    yield dim3, spatial_dim_values, group_design_matrix


def _hash_channel_wise_regressor(regressor: xr.DataArray) -> list[int]:
    """Hashes each channel slice of the regressor array.

    Args:
        regressor: array of channel-wise regressors. Dims
            (channel, regressor, time, chromo|wavelength)

    Returns:
        A list of hash values, one hash for each channel.
    """

    spatial_dim = cdc.get_spatial_dimension(regressor)

    tmp = regressor.pint.dequantify()
    n_channel = regressor.sizes[spatial_dim]

    return [
        hash(tmp.isel({spatial_dim: i}).values.data.tobytes()) for i in range(n_channel)
    ]


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

    basis = basis_function(ts)

    components = basis.component.values

    # could be "chromo" or "wavelength"
    # Determine spatial dimension dynamically (e.g., "channel", "parcel", ...)
    # instead of assuming "channel", to support multiple representations.
    spatial_dim = cdc.get_spatial_dimension(ts)
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

                # basis functions are normalized to 1. don't normalize the convolved
                # regressor again.
                # regressor /= regressor.max()

                regressors[:, i_reg, i_other] = regressor

    return common_regressor_from_array(ts, regressors, regressor_names)

# FIXME reduce overlap with hrf_regressors

def hrf_extract_regressors(
    ts: cdt.NDTimeSeries, stim: pd.DataFrame, basis_function: TemporalBasisFunction
) -> DesignMatrix:
    """Create regressors for extracting the fitted, unconvolved HRF.

    The returned design matrix spans only the time range of the basis functions.
    Regressors are not convolved.

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
    # other_dim = xrutils.other_dim(ts, "channel", "time")
    spatial_dim = cdc.get_spatial_dimension(ts)
    other_dim = xrutils.other_dim(ts, spatial_dim, "time")

    n_time = basis.sizes["time"]
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

    #padded_time, pad_before = _pad_time_axis(ts.time.values, shifted_stim["onset"])

    if n_components == 1:
        regressor_names = [f"HRF {tt}" for tt in trial_types]
    else:
        regressor_names = [f"HRF {tt} {c}" for tt in trial_types for c in components]

    regressors = np.zeros((n_time, n_regressors, n_other))

    for i_tt, trial_type in enumerate(trial_types):
        for i_comp in range(n_components):
            i_reg = i_tt * n_components + i_comp
            for i_other, other in enumerate(ts[other_dim].values):
                bb = basis.sel({other_dim: other})
                regressor = bb[:,i_comp]
                regressor /= regressor.max()

                regressors[:, i_reg, i_other] = regressor

    return common_regressor_from_array(
        ts,
        regressors,
        regressor_names,
        time=basis.time.values,
    )



def drift_regressors(ts: cdt.NDTimeSeries, drift_order) -> DesignMatrix:
    """Create drift regressors.

    Args:
        ts (cdt.NDTimeSeries): Time series data.
        drift_order (int): The highest polynomial order of the drift terms.

    Returns:
        xr.DataArray: A DataArray containing the drift regressors.
    """
    # dim3 = xrutils.other_dim(ts, "channel", "time")
    spatial_dim = cdc.get_spatial_dimension(ts)
    dim3 = xrutils.other_dim(ts, spatial_dim, "time")
    ndim3 = ts.sizes[dim3]


    nt = ts.sizes["time"]

    drift_regressors = np.ones((nt, drift_order + 1, ndim3))

    for i in range(1, drift_order + 1):
        tmp = np.arange(1, nt + 1, dtype=float) ** (i)
        tmp /= tmp[-1]
        # Center higher drift orders to stabilize RecursiveLS.
        # Uncentered ramp + constant can cause numerical instability in
        # statsmodels RLS.
        if i > 0:
            tmp -= tmp.mean()

        drift_regressors[:, i, 0] = tmp

    for i in range(1, ndim3):
        drift_regressors[:, :, i] = drift_regressors[:, :, 0]

    regressor_names = [f"Drift {i}" for i in range(drift_order + 1)]

    return common_regressor_from_array(ts, drift_regressors, regressor_names)


def drift_legendre_regressors(ts : cdt.NDTimeSeries, order : int) -> DesignMatrix:
    """Create drift regressors using Legende polynomials.

    Args:
        ts: time-series data.
        order: generate polynomials of order 0...order

    Returns:
        xr.DataArray: A DataArray containing the drift regressors.
    """

    # dim3 = xrutils.other_dim(ts, "channel", "time")
    spatial_dim = cdc.get_spatial_dimension(ts)
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

    return common_regressor_from_array(ts, drift_regressors, regressor_names)


def drift_cosine_regressors(ts: cdt.NDTimeSeries, fmax: cdt.QFrequency) -> DesignMatrix:
    """Create drift regressors using cosine basis functions.

    Args:
        ts: time-series data.
        fmax: High-pass cutoff frequency

    Returns:
        xr.DataArray: A DataArray containing the drift regressors.
    """

    spatial_dim = cdc.get_spatial_dimension(ts)
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

    return common_regressor_from_array(ts, drift_regressors, regressor_names)



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

    spatial_dim = cdc.get_spatial_dimension(ts_long)
    dim3 = xrutils.other_dim(ts_long, spatial_dim, "time")

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
    ts_long: cdt.NDTimeSeries, ts_short: cdt.NDTimeSeries, geo3d: cdt.LabeledPoints
):
    """Create channel-wise regressors using the closest nearby short channel.

    Short-separation channels measure predominantly superficial (scalp) haemodynamics
    and are used as regressors to remove physiological noise from long channels, as
    described in :cite:t:`Huppert2009`.

    Args:
        ts_long (NDTimeSeries): Time series of long channels
        ts_short (NDTimeSeries): Time series of short channels
        geo3d (LabeledPoints): Probe geometry

    Returns:
        regressors (xr.DataArray): Channel-wise regressor
    """
    cite("Huppert2009")
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

    For each long channel the short channel with the highest correlation coefficient
    (across any wavelength or chromophore) is selected as a physiological noise
    regressor, as described in :cite:t:`Huppert2009`.

    Args:
        ts_long (NDTimeSeries): time series of long channels
        ts_short (NDTimeSeries): time series of short channels

    Returns:
        xr.DataArray: channel-wise regressors
    """

    cite("Huppert2009")
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
    """Create a physiological noise regressor by averaging all short channels.

    Computes a single global superficial noise regressor from the mean across all
    short-separation channels, as described in :cite:t:`Huppert2009`.

    Args:
        ts_short (NDTimeSeries): time series of short channels

    Returns:
        xr.DataArray: regressors
    """

    cite("Huppert2009")
    ts_short = ts_short.pint.dequantify()
    regressor = ts_short.mean("channel", skipna=True)

    return common_regressor_from_array(ts_short, regressor, "short")


def global_mean_regressor(ts: cdt.NDTimeSeries) -> DesignMatrix:
    """Create a global regressor by averaging over the spatial dimension.

    Args:
        ts (NDTimeSeries): time series data.

    Returns:
        DesignMatrix: design matrix with one global regressor.
    """
    spatial_dim = cdc.get_spatial_dimension(ts)

    ts = ts.pint.dequantify()
    regressor = ts.mean(spatial_dim, skipna=True)

    return common_regressor_from_array(ts, regressor, "global")


def common_regressor_from_array(
    ts: cdt.NDTimeSeries,
    regressors: xr.DataArray | np.ndarray,
    regressor_names: str | list[str],
    time: ArrayLike | None = None,
) -> DesignMatrix:
    """Create common GLM regressors from a precomputed array.

    Args:
        ts: Time-series data whose third-dimension coordinates are used.
        regressors: Regressor array with shape either
            ``(n_times, n_regressors, n_dim3)`` or, for a single regressor,
            ``(n_times, n_dim3)``. If an ``xr.DataArray`` is passed, it is
            dequantified before packaging.
        regressor_names: Name or names for the regressors. A single string is
            allowed only when there is one regressor.
        time: Optional time coordinate for the regressors. Defaults to ``ts.time``.

    Returns:
        Design matrix containing the regressors as common regressors.
    """

    spatial_dim = cdc.get_spatial_dimension(ts)
    dim3 = xrutils.other_dim(ts, spatial_dim, "time")

    if isinstance(regressors, xr.DataArray):
        regressors = regressors.pint.dequantify()

        if regressors.ndim == 2:
            regressors = regressors.transpose("time", dim3).values[:, None, :]
        elif regressors.ndim == 3:
            regressors = regressors.transpose("time", "regressor", dim3).values
        else:
            regressors = regressors.values
    else:
        regressors = np.asarray(regressors)

        if regressors.ndim == 2:
            regressors = regressors[:, None, :]

    if regressors.ndim != 3:
        raise ValueError(
            "regressors must have shape (n_times, n_regressors, n_dim3) "
            "or (n_times, n_dim3) for a single regressor."
        )

    if time is None:
        time = ts.time

    if regressors.shape[0] != len(time):
        raise ValueError(
            "regressors time dimension does not match time coordinate: "
            f"{regressors.shape[0]} != {len(time)}."
        )

    if regressors.shape[2] != ts.sizes[dim3]:
        raise ValueError(
            f"regressors {dim3!r} dimension does not match ts: "
            f"{regressors.shape[2]} != {ts.sizes[dim3]}."
        )

    n_regressors = regressors.shape[1]

    if isinstance(regressor_names, str):
        regressor_names = [regressor_names]

    if len(regressor_names) != n_regressors:
        raise ValueError(
            "Number of regressor names must match the regressor dimension: "
            f"{len(regressor_names)} != {n_regressors}."
        )

    regressors = xr.DataArray(
        regressors,
        dims=["time", "regressor", dim3],
        coords={
            "time": time,
            "regressor": regressor_names,
            dim3: ts[dim3].values,
        },
    )

    return DesignMatrix(common=regressors, channel_wise=[])


def global_component_regressor(
    ts: cdt.NDTimeSeries,
    ts_weights: xr.DataArray | None = None,
    k: float = 0,
    spatial_dim: str | None = None,
    spectral_dim: str | None = None,
    regressor_names: str | list[str] = "global",
) -> DesignMatrix:
    """Create common GLM regressors from the global physiological component.

    The global component is computed with
    :func:`cedalion.sigproc.physio.global_component_subtract`. The corrected
    time series returned by that function is discarded, and only the global
    component is used as a common GLM regressor.

    Args:
        ts: Time-series data.
        ts_weights: Optional per-spatial/spectral weights passed to
            ``global_component_subtract``.
        k: Component-removal mode passed to ``global_component_subtract``.
            ``k=0`` uses weighted-mean subtraction. ``k>0`` uses PCA-based
            component removal.
        spatial_dim: Spatial dimension name. If ``None``, it is inferred from
            ``ts``.
        spectral_dim: Spectral dimension name. If ``None``, it is inferred by
            ``global_component_subtract``.
        regressor_names: Name or names for the regressors.

    Returns:
        Design matrix containing the global component as common regressors.
    """

    if spatial_dim is None:
        spatial_dim = cdc.get_spatial_dimension(ts)

    _, global_component = global_component_subtract(
        ts,
        ts_weights=ts_weights,
        k=k,
        spatial_dim=spatial_dim,
        spectral_dim=spectral_dim,
    )

    return common_regressor_from_array(ts, global_component, regressor_names)


def spatially_smoothed_pca_global_component_regressor(
    ts: cdt.NDTimeSeries,
    vertices: xr.DataArray | np.ndarray,
    sigma_mm: float = 80.0,
    spatial_dim: str | None = None,
    spectral_dim: str | None = None,
    regressor_names: str | list[str] = "global_smoothed_pca",
) -> DesignMatrix:
    """Create a GLM regressor from a spatially smoothed PCA global component.

    Args:
        ts: Time-series data.
        vertices: Spatial coordinates for the spatial dimension, shape
            ``(n_spatial, 3)``.
        sigma_mm: Gaussian smoothing width in millimeters.
        spatial_dim: Spatial dimension name. If ``None``, inferred from ``ts``.
        spectral_dim: Spectral dimension name. If ``None``, inferred from
            ``ts``.
        regressor_names: Name or names for the regressors.

    Returns:
        Design matrix containing the spatially smoothed PCA global component.
    """

    if spatial_dim is None:
        spatial_dim = cdc.get_spatial_dimension(ts)

    if spectral_dim is None:
        spectral_dim = xrutils.other_dim(ts, spatial_dim, "time")

    smoothing_kernel = get_spatial_smoothing_kernel(vertices, sigma_mm)

    smoothed_global_field = compute_Hglobal_from_PCA(
        ts,
        smoothing_kernel=smoothing_kernel,
        spatial_dim=spatial_dim,
        spectral_dim=spectral_dim,
    )

    global_component = smoothed_global_field.mean(spatial_dim, skipna=True)

    return common_regressor_from_array(ts, global_component, regressor_names)