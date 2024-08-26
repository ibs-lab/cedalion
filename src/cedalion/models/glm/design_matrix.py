import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

import cedalion
import cedalion.nirs
import cedalion.typing as cdt
import cedalion.xrutils as xrutils

from .basis_functions import TemporalBasisFunction

# def make_design_matrix(
#    t: xr.DataArray,
#    s,
#    trange: list,
#    idx_basis: str,
#    params_basis,
#    drift_order: int = 0,
#    add_regressors: xr.DataArray = None,
# ):
#    """Generate the design matrix for the GLM.
#
#    Args:
#        t: xarray.DataArray time axis (time)
#        s: pandas.DataFrame or xarray.DataArray of stimulus data (time x conditions)
#        add_regressors: xarray.DataArray containing additional regressors
#            (time x regressor x chromo)
#        trange: list of the time range of the HRF regressors [pre, post]
#            (in seconds relative to stimulus onset, for example [-5, 15])
#        idx_basis: string indicating the type of basis function to use
#            "gaussians":    a consecutive sequence of gaussian functions
#            "gamma":        a modified gamma function convolved with a square-
#                            wave of duration T. Set T=0 for no convolution.
#                            The modified gamma function is
#                            (exp(1)*(t-tau).^2/sigma^2) .* exp(-(tHRF-tau).^2/sigma^2)
#            "gamma_deriv":  a modified gamma function and its derivative convolved
#                            with a square-wave of duration T. Set T=0 for no conv.
#            "afni_gamma":   GAM function from 3dDeconvolve AFNI convolved with
#                            a square-wave of duration T. Set T=0 for no convolution.
#                                            (t/(p*q))^p * exp(p-t/q)
#                            Defaults: p=8.6 q=0.547
#                            The peak is at time p*q.  The FWHM is about 2.3*sqrt(p)*q.
#            "individual":   individual selected basis function for each channel
#                            (xr.DataArray of shape (time x chromo x channels)
#        params_basis: parameters for the basis function (depending on idx_basis)
#            "gaussians":    [gms, gstd] where gms is the temporal spacing between
#                            consecutive gaussians and gstd is the width of the
#                            gaussian (standard deviation)
#            "gamma":        [tau sigma T] applied to both HbO and HbR
#                            or [tau1 sigma1 T1 tau2 sigma2 T2]
#                            where the 1 (2) indicates the parameters for HbO (HbR).
#            "gamma_deriv":  [tau sigma T] applied to both HbO and HbR
#                            or [tau1 sigma1 T1 tau2 sigma2 T2]
#                            where the 1 (2) indicates the parameters for HbO (HbR).
#            "afni_gamma":   [p q T] applied to both HbO and HbR
#                            or [p1 q1 T1 p2 q2 T2]
#                            where the 1 (2) indicates the parameters for HbO (HbR).
#            "individual":   array containing the individual basis functions
#                            (t_hrf x chromo x channels).
#        drift_order: order of the polynomial drift regressors to add to the
#            design matrix
#    Returns:
#        A: xarray.DataArray of the design matrix (time x regressor x chromo)
#    """
#
#    t = t.pint.dequantify()
#    # Convert stimulus data to xarray
#    if type(s) == pd.DataFrame:
#        s = s.cd.to_xarray(t)
#
#    # Initialize regressor list
#    regressors = []
#
#    ###########################################################################
#    # Construct the basis functions
#
#    hrf_regressors = make_hrf_regressors(t, s, trange, idx_basis, params_basis)
#    regressors.append(hrf_regressors)
#
#    ###########################################################################
#    # Add drift regressors
#
#    drift_regressors = make_drift_regressors(t, drift_order)
#    regressors.append(drift_regressors)
#
#    ###########################################################################
#    # Add additional regressors
#
#    if add_regressors is not None:
#        add_regressors = add_regressors.pint.quantify("micromolar")
#        regressors.append(add_regressors)
#
#    ###########################################################################
#    # Stack regressors into design matrix
#
#    A = xr.concat(regressors, dim="regressor")
#    A = A.pint.quantify("micromolar")
#
#    return A


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
        ts_long: time series of long distance channels
        ts_short: time series of short distance channels
        stim: stimulus DataFrame
        geo3d: probe geometry
        basis_function: the temporal basis function(s) to model the HRF
        drift_order: if not None specify the highest polynomial order of the drift terms
        short_channel_method: can be 'closest' or 'max_corr' and specifies the
            method to add short channel information ot the design matrix

    Returns:
        A tuple containing the global design_matrix and a list of channel-wise
        regressors.
    """

    dm = make_hrf_regressors(ts_long, stim, basis_function)

    if drift_order is not None:
        dm_drift = make_drift_regressors(ts_long, drift_order=drift_order)
        dm = xr.concat([dm, dm_drift], dim="regressor")

    if short_channel_method == "closest":
        if ts_short is None:
            raise ValueError("ts_short may not be None.")
        channel_wise_regressors = [closest_short_channel(ts_long, ts_short, geo3d)]
    elif short_channel_method == "max_corr":
        if ts_short is None:
            raise ValueError("ts_short may not be None.")
        channel_wise_regressors = [max_corr_short_channel(ts_long, ts_short)]
    else:
        channel_wise_regressors = None

    return dm, channel_wise_regressors


def make_drift_regressors(ts: cdt.NDTimeSeries, drift_order) -> xr.DataArray:
    dim3 = get_third_dimension(ts)
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


# def make_hrf_regressors_orig(
#    t: xr.DataArray, s: xr.DataArray, trange: list, idx_basis: str, params_basis
# ):
#    cond_names = s.condition.values
#    dt = 1 / t.cd.sampling_rate
#    n_pre = round(trange[0] / dt)
#    n_post = round(trange[1] / dt)
#    # np.arange results can be non-consistent when using non-integer steps
#    # t_hrf = np.arange(n_pre * dt, (n_post + 1) * dt, dt)
#    # using linspace instead
#    t_hrf = np.linspace(trange[0], trange[1], abs(n_post) + abs(n_pre) + 1)
#    nt = len(t)
#
#    # prune good stim
#    # handle case of conditions with 0 trials
#    lst_cond = np.where(np.sum(s > 0, axis=0) > 0)[0]
#    n_cond = len(lst_cond)
#    onset = np.zeros((nt, n_cond))
#    n_trials = np.zeros(n_cond)
#
#    # handle case of single condition
#    if n_cond == 1:
#        cond_names = np.array([cond_names])
#        s = np.expand_dims(s, axis=1)
#
#    for i_cond in range(n_cond):
#        lst_t = np.where(s[:, lst_cond[i_cond]] == 1)[0]
#        lstp = np.where((lst_t + n_pre >= 0) & (lst_t + n_post < nt))[0]
#        lst = lst_t[lstp]
#        n_trials[i_cond] = len(lst)
#        onset[lst + n_pre, i_cond] = 1
#
#    tbasis, nb = construct_basis_functions(t_hrf, idx_basis, params_basis)
#
#    # remove the third dimension if it is 1
#    if len(tbasis.shape) == 3 and tbasis.shape[2] == 1:
#        tbasis = tbasis[:, :, 0]
#
#    regressor_names = []
#
#    for i_cond in range(n_cond):
#        for b in range(nb):
#            regressor_names.append("HRF " + cond_names[i_cond] + " " + str(b + 1))
#
#    if idx_basis != "individual":
#        a_hrf = np.zeros((nt, nb * n_cond, 2))
#        for i_conc in range(2):
#            i_c = -1
#            for i_cond in range(n_cond):
#                for b in range(nb):
#                    i_c += 1
#                    if len(tbasis.shape) == 2:
#                        clmn = np.convolve(onset[:, i_cond], tbasis[:, b])
#                    else:
#                        clmn = np.convolve(onset[:, i_cond], tbasis[:, b, i_conc])
#                    clmn = clmn[:nt]
#                    a_hrf[:, i_c, i_conc] = clmn
#        hrf_regs = xr.DataArray(
#            a_hrf,
#            dims=["time", "regressor", "chromo"],
#            coords={"time": t, "regressor": regressor_names, "chromo": ["HbO", "HbR"]},
#        )
#
#    elif idx_basis == "individual":
#        a_hrf = np.zeros((nt, nb * n_cond, 2, params_basis.shape[2]))
#        for i_conc in range(2):
#            for i_ch in range(params_basis.shape[2]):
#                i_c = -1
#                for i_cond in range(n_cond):
#                    for b in range(nb):
#                        i_c += 1
#                        clmn = np.convolve(
#                            onset[:, i_cond], np.squeeze(tbasis[i_ch, :, b, i_conc])
#                        )
#                        clmn = clmn[:nt]
#                        a_hrf[:, i_c, i_conc, i_ch] = clmn
#        hrf_regs = xr.DataArray(
#            a_hrf,
#            dims=["time", "regressor", "chromo", "channel"],
#            coords={
#                "time": t,
#                "regressor": regressor_names,
#                "chromo": ["HbO", "HbR"],
#                "channel": params_basis.channel.values,
#            },
#        )
#
#    hrf_regs = hrf_regs.pint.quantify("micromolar")
#
#    return hrf_regs
#


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
    time: ArrayLike, onsets: ArrayLike, duration: None | ArrayLike
) -> np.ndarray:
    """Build an array indicating active stimulus periods.

    The array values are 1 between onset and onset+duration and zero everywhere else.

    Args:
        time: the time axis
        onsets: times of stimulus onsets
        duration: either durations of each stimulus or None, in which case the stimulus
            duration is set to one sample.

    Returns:
        The array denoting

    """

    smpl_start = time.searchsorted(onsets)
    if duration is None:
        smpl_stop = smpl_start + 1
    else:
        smpl_stop = time.searchsorted(onsets + duration)

    stim = np.zeros_like(time)
    for start, stop in zip(smpl_start, smpl_stop):
        stim[start:stop] = 1

    return stim


def get_third_dimension(ts: cdt.NDTimeSeries):
    other_dims = [d for d in ts.dims if d != "channel" and d != "time"]
    assert len(other_dims) == 1
    return other_dims[0]


def make_hrf_regressors(
    ts: cdt.NDTimeSeries, stim: pd.DataFrame, basis_function: TemporalBasisFunction
):
    """Create regressors modelling the hemodynamic response to stimuli."""

    trial_types: np.ndarray = stim.trial_type.unique()

    basis = basis_function(ts)

    components = basis.component.values

    other_dim = get_third_dimension(ts)  # could be "chromo" or "wavelength"
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
            stim_array = build_stim_array(padded_time, tmp["onset"], tmp["duration"])
        else:
            stim_array = build_stim_array(padded_time, tmp["onset"], None)

        for i_comp in range(n_components):
            i_reg = i_tt * n_components + i_comp
            for i_other, other in enumerate(ts[other_dim].values):
                bb = basis.sel({other_dim: other})

                # Convolve the basis function with the boxcar stimuls function
                # usigin 'full' mode, i.e. the resulting regressor is longer than the
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


def construct_basis_functions(t_hrf, idx_basis, params_basis):
    # Gaussians
    if idx_basis == "gaussians":
        return construct_gaussian_basis(t_hrf, params_basis)

    # Modified Gamma
    elif idx_basis == "gamma":
        return construct_modified_gamma_basis(t_hrf, params_basis)

    # Modified Gamma and Derivative
    elif idx_basis == "gamma_deriv":
        return construct_modified_gamma_deriv_basis(t_hrf, params_basis)

    # AFNI Gamma function
    elif idx_basis == "afni_gamma":
        return construct_afni_gamma_basis(t_hrf, params_basis)

    # Individualized basis function for each channel from a previously estimated HRF
    elif idx_basis == "individual":
        params_basis.pint.dequantify()
        return construct_individual_basis(t_hrf, params_basis)


def construct_gaussian_basis(t_hrf, params_basis):
    nt_hrf = len(t_hrf)
    trange = np.round([t_hrf[0], t_hrf[-1]])

    gms = params_basis[0]
    gstd = params_basis[1]

    nb = int((trange[1] - trange[0]) / gms) - 1
    tbasis = np.zeros((nt_hrf, nb))
    for b in range(nb):
        tbasis[:, b] = np.exp(
            -((t_hrf - (trange[0] + (b + 1) * gms)) ** 2) / (2 * gstd**2)
        )
        tbasis[:, b] = tbasis[:, b] / np.max(tbasis[:,])  # Normalize to 1

    return tbasis, nb


def construct_modified_gamma_basis(t_hrf, params_basis):
    nt_hrf = len(t_hrf)
    dt = t_hrf[1] - t_hrf[0]
    params_len = len(params_basis)
    if params_len == 3:
        n_conc = 1
    elif params_len == 6:
        n_conc = 2

    nb = 1
    tbasis = np.zeros((nt_hrf, nb, n_conc))
    for i_conc in range(n_conc):
        tau = params_basis[i_conc * 3]
        sigma = params_basis[i_conc * 3 + 1]
        tt = params_basis[i_conc * 3 + 2]

        tbasis[:, 0, i_conc] = (np.exp(1) * (t_hrf - tau) ** 2 / sigma**2) * np.exp(
            -((t_hrf - tau) ** 2) / sigma**2
        )
        lst_neg = np.where(t_hrf < 0)[0]
        tbasis[lst_neg, 0, i_conc] = 0

        if t_hrf[0] < tau:
            tbasis[: int((tau - t_hrf[0]) / dt), 0, i_conc] = 0

        if tt > 0:
            for ii in range(nb):
                foo = np.convolve(tbasis[:, ii, i_conc], np.ones(int(tt / dt))) / int(
                    tt / dt
                )
                tbasis[:, ii, i_conc] = foo[:nt_hrf]

    # tbasis = tbasis.squeeze()

    return tbasis, nb


def construct_modified_gamma_deriv_basis(t_hrf, params_basis):
    nt_hrf = len(t_hrf)
    dt = t_hrf[1] - t_hrf[0]
    params_len = len(params_basis)
    if params_len == 3:
        n_conc = 1
    elif params_len == 6:
        n_conc = 2

    nb = 2
    tbasis = np.zeros((nt_hrf, nb, n_conc))
    for i_conc in range(n_conc):
        tau = params_basis[i_conc * 3]
        sigma = params_basis[i_conc * 3 + 1]
        tt = params_basis[i_conc * 3 + 2]

        tbasis[:, 0, i_conc] = (np.exp(1) * (t_hrf - tau) ** 2 / sigma**2) * np.exp(
            -((t_hrf - tau) ** 2) / sigma**2
        )
        tbasis[:, 1, i_conc] = (
            2
            * np.exp(1)
            * ((t_hrf - tau) / sigma**2 - (t_hrf - tau) ** 3 / sigma**4)
            * np.exp(-((t_hrf - tau) ** 2) / sigma**2)
        )

        if t_hrf[0] < tau:
            tbasis[: int((tau - t_hrf[0]) / dt), :2, i_conc] = 0

        if tt > 0:
            for ii in range(nb):
                foo = np.convolve(tbasis[:, ii, i_conc], np.ones(int(tt / dt))) / int(
                    tt / dt
                )
                tbasis[:, ii, i_conc] = foo[:nt_hrf]

    return tbasis, nb


def construct_afni_gamma_basis(t_hrf, params_basis):
    nt_hrf = len(t_hrf)
    dt = t_hrf[1] - t_hrf[0]
    params_len = len(params_basis)
    if params_len == 3:
        n_conc = 1
    elif params_len == 6:
        n_conc = 2

    nb = 1
    tbasis = np.zeros((nt_hrf, nb, n_conc))
    for i_conc in range(n_conc):
        p = params_basis[i_conc * 3]
        q = params_basis[i_conc * 3 + 1]
        tt = params_basis[i_conc * 3 + 2]

        bas = t_hrf / (p * q)
        # Numpy does not seem to allow fractional powers of negative numbers, even if
        # the power would not result in a complex number.
        # tbasis[:, 0, i_conc] = np.array((np.array(bas, dtype=np.complex128)) ** p,
        # dtype=np.float64) * np.exp(p - t_hrf / q)
        tbasis[:, 0, i_conc] = (bas**p) * np.exp(p - t_hrf / q)

        if tt > 0:
            foo = np.convolve(tbasis[:, 0, i_conc], np.ones(int(tt / dt))) / int(
                tt / dt
            )
            tbasis[:, 0, i_conc] = foo[:nt_hrf]

    return tbasis, nb


def construct_individual_basis(t_hrf, params_basis):
    nt_hrf = len(t_hrf)
    n_conc = 2  # HbO and HbR separate basis
    nb = 1
    tbasis = np.zeros((params_basis.shape[2], nt_hrf, nb, n_conc))
    for i_conc in range(n_conc):
        for i_ch in range(params_basis.shape[2]):
            tbasis[i_ch, :, 0, i_conc] = params_basis[:, i_conc, i_ch]

    return tbasis, nb


def get_ss_regressors(
    y: xr.DataArray,
    geo3d: xr.DataArray,
    ss_method="closest",
    ss_tresh: float = 1.5,
    as_xarray: bool = False,
):
    """Get short separation channels for each long channel.

    Args:
        y: xarray.DataArray of the data (time x chromo x channels)
        geo3d: xarray.DataArray of the 3D geometry (no. of sources/detectors x dim pos)
        ss_method: method for determining short separation channels ("nearest", "corr")
        ss_tresh: threshold for short separation channels (in cm)
        get_data: whether to return the short channel data (True) or the short channel
            names (False)
        as_xarray: TBD

    Returns:
        ss: xarray.DataArray of short separation channels (channels x chromo)
    """

    # Calculate source-detector distances for each channel
    dists = (
        xrutils.norm(geo3d.loc[y.source] - geo3d.loc[y.detector], dim="pos")
        .pint.to("mm")
        .round(2)
    )

    # Identify short channels
    short_channels = dists.channel[dists < ss_tresh * cedalion.units.cm]

    if ss_method == "closest":
        middle_pos = (geo3d.loc[y.source] + geo3d.loc[y.detector]) / 2
        return closest_short_channel(y, short_channels, middle_pos, as_xarray)
    elif ss_method == "corr":
        return max_corr_short_channel(y, short_channels, as_xarray)


"""
def closest_short_channel(y, short_channels, middle_positions, as_xarray=False):
    units = y.pint.units
    y = y.pint.dequantify()

    names_init = np.array([["______", "______"] for i in range(len(y.channel))])
    if as_xarray:
        # Initialize xarray DataArray to store the closest short channel data
        closest_short = xr.DataArray(
            np.zeros((len(y.time), 2, len(y.channel))),
            dims=["time", "chromo", "channel"],
            coords={"time": y.time, "channel": y.channel, "chromo": y.chromo},
        )
        # add regressor dimension with the name of the closest short channel
        closest_short = closest_short.assign_coords(
            regressor=(["channel", "chromo"], names_init)
        )

    else:
        # Initialize for dictionary data structure
        closest_short = names_init
        closest_short = xr.DataArray(
            closest_short,
            dims=["channel", "chromo"],
            coords={"channel": y.channel, "chromo": y.chromo},
        )

    for ch in y.channel:
        distances_to_short = xrutils.norm(
            middle_positions.loc[short_channels.channel] - middle_positions.loc[ch],
            dim="pos",
        )
        closest_index = (
            distances_to_short.argmin().item()
        )  # Get index of closest short channel
        closest_name = short_channels.channel[closest_index]

        if as_xarray:
            # For each channel, fetch data from the closest short channel and assign it
            for chromo in y.chromo.values:  # Assuming 2 chromophores: HbO and HbR
                # [{"time": y.time, "chromo": chromo, "channel": ch}]
                closest_short.loc[{"chromo": chromo, "channel": ch}] = y.sel(
                    chromo=chromo, channel=closest_name
                )
                # add regressor name
                closest_short.regressor.loc[y.channel == ch, chromo] = closest_name
        else:
            # Assign the name of the closest short channel for both chromophores
            closest_short.loc[ch, "HbO"] = closest_name
            closest_short.loc[ch, "HbR"] = closest_name

    if as_xarray:
        closest_short = closest_short.pint.quantify(units)
    else:
        # Transform to regressor dictionary
        closest_short = make_reg_dict(y.pint.quantify(units), closest_short)

    return closest_short
"""


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

    dim3 = get_third_dimension(ts_long)

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
        ts_long: time series of long channels
        ts_short: time series of short channels
        geo3d: probe geometry

    Returns:
        channel-wise regressor
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
        ts_long: time series of long channels
        ts_short: time series of short channels

    Returns:
        channel-wise regressors
    """

    dim3 = get_third_dimension(ts_long)

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


# def max_corr_short_channel_orig(y, short_channels, as_xarray=False):
#    # Get indices of short channels
#    lst_ss = [np.where(y.channel == ch)[0][0] for ch in short_channels]
#
#    units = y.pint.units
#    y = y.pint.dequantify()
#
#    # HbO
#    dc = y[:, 0, :].squeeze().values
#    dc = (dc - np.mean(dc, axis=0)) / np.std(dc, axis=0)
#    cc1 = np.dot(dc.T, dc) / len(dc)
#
#    # HbR
#    dc = y[:, 1, :].squeeze().values
#    dc = (dc - np.mean(dc, axis=0)) / np.std(dc, axis=0)
#    cc2 = np.dot(dc.T, dc) / len(dc)
#
#    iNearestSS = np.zeros((cc1.shape[0], 2), dtype=int)
#    for iML in range(cc1.shape[0]):
#        iNearestSS[iML, 0] = lst_ss[np.argmax(cc1[iML, lst_ss])]
#        iNearestSS[iML, 1] = lst_ss[np.argmax(cc2[iML, lst_ss])]
#
#    channel_iSS = [y.channel[i].values.astype("U6") for i in iNearestSS]
#
#    if as_xarray:
#        # get data not names of short channels
#        ss_data = np.zeros((len(y.time), 2, len(y.channel)))
#        for i in range(len(y.channel)):
#            ss_data[:, 0, i] = y[:, 0, iNearestSS[i, 0]]
#            ss_data[:, 1, i] = y[:, 1, iNearestSS[i, 1]]
#        highest_corr_short = xr.DataArray(
#            ss_data,
#            dims=["time", "chromo", "channel"],
#            coords={"time": y.time, "channel": y.channel, "chromo": y.chromo},
#        ).pint.quantify(units)
#        # add regressor dimension
#        highest_corr_short = highest_corr_short.assign_coords(
#            regressor=(["channel", "chromo"], channel_iSS)
#        )
#
#    else:
#        # transform to regressor dictionary
#        highest_corr_short = xr.DataArray(
#            channel_iSS,
#            dims=["channel", "chromo"],
#            coords={"channel": y.channel, "chromo": y.chromo},
#        )
#        highest_corr_short = make_reg_dict(y.pint.quantify(units), highest_corr_short)
#
#    return highest_corr_short
#
#
# def make_reg_dict(y, add_regressors):
#    """Make a dictionary with regressors as keys and channels as values.
#
#    inputs:
#        y: xarray.DataArray of the data (time x chromo x channels)
#        add_regressors: xarray.DataArray containing the short channel regressor
#                        (name) for each channel (channels x chromo)
#
#    Return:
#        add_reg_dict: dictionary with regressors as keys and channels as values
#    """
#
#    unique_short = np.unique(add_regressors.values)
#
#    ss_data = y.sel(channel=unique_short)
#    ss_data = ss_data.drop_vars(["samples", "source", "detector", "sbj"])
#    ss_data = ss_data.rename({"channel": "regressor"})
#
#    reg_dict = {}
#    reg_dict["data"] = ss_data
#
#    # fill dictionary: if channels have same regressor, merge them to one key
#    for chromo in add_regressors.chromo.values:
#        reg_dict[chromo] = {}
#        add_reg_chromo = add_regressors.sel(chromo=chromo)
#        for ch in np.unique(add_reg_chromo.values):
#            reg_dict[chromo][ch] = add_reg_chromo.where(
#                add_reg_chromo == ch, drop=True
#            ).channel.values.tolist()
#    return reg_dict
