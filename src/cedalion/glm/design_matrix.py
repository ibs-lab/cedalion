import numpy as np
import pandas as pd
import xarray as xr
import cedalion
import cedalion.nirs
import cedalion.xrutils as xrutils


def make_design_matrix(
    t: xr.DataArray, s, trange: list, idx_basis: str, params_basis, drift_order: int = 0
):
    """Generate the design matrix for the GLM.

    inputs:
        y: xarray.DataArray time axis (time)
        s: pandas.DataFrame or xarray.DataArray of stimulus data (time x conditions)
        trange: list of the time range of the HRF regressors [pre, post]
                (in seconds relative to stimulus onset, for example [-5, 15])
        idx_basis: string indicating the type of basis function to use
                    "gaussians":    a consecutive sequence of gaussian functions
                    "gamma":        a modified gamma function convolved with a square-
                                    wave of duration T. Set T=0 for no convolution.
                                    The modified gamma function is
                                    (exp(1)*(t-tau).^2/sigma^2) .* exp(-(tHRF-tau).^2/sigma^2)
                    "gamma_deriv":  a modified gamma function and its derivative convolved
                                    with a square-wave of duration T. Set T=0 for no convolution.
                    "afni_gamma":   GAM function from 3dDeconvolve AFNI convolved with
                                    a square-wave of duration T. Set T=0 for no convolution.
                                                    (t/(p*q))^p * exp(p-t/q)
                                    Defaults: p=8.6 q=0.547
                                    The peak is at time p*q.  The FWHM is about 2.3*sqrt(p)*q.
                    "individual":   individual selected basis function for each channel
        params_basis: parameters for the basis function (depending on idx_basis)
                    "gaussians":    [gms, gstd] where gms is the temporal spacing between
                                    consecutive gaussians and gstd is the width of the
                                    gaussian (standard deviation)
                    "gamma":        [tau sigma T] applied to both HbO and HbR
                                    or [tau1 sigma1 T1 tau2 sigma2 T2]
                                    where the 1 (2) indicates the parameters for HbO (HbR).
                    "gamma_deriv":  [tau sigma T] applied to both HbO and HbR
                                    or [tau1 sigma1 T1 tau2 sigma2 T2]
                                    where the 1 (2) indicates the parameters for HbO (HbR).
                    "afni_gamma":   [p q T] applied to both HbO and HbR
                                    or [p1 q1 T1 p2 q2 T2]
                                    where the 1 (2) indicates the parameters for HbO (HbR).
                    "individual":   array containing the individual basis functions
                                    (t_hrf x chromo x channels).
        drift_order: order of the polynomial drift regressors to add to the design matrix
    outputs:
        A: xarray.DataArray of the design matrix (time x regressors x chromo)
    """

    # Convert stimulus data to xarray
    if type(s) == pd.DataFrame:
        s = s.cd.to_xarray(t)

    # Initialize regressor list
    regressors = []

    ###########################################################################
    # Construct the basis functions

    hrf_regressors = make_hrf_regressors(t, s, trange, idx_basis, params_basis)
    regressors.append(hrf_regressors)

    ###########################################################################
    # Add drift regressors

    drift_regressors = make_drift_regressors(t, drift_order)
    regressors.append(drift_regressors)

    ###########################################################################
    # Stack regressors into design matrix

    A = xr.concat(regressors, dim="regressor")

    return A


def make_drift_regressors(t, drift_order):

    nt = len(t)

    drift_regressors = np.ones((nt, drift_order + 1, 2))

    for ii in range(1, drift_order + 1):
        drift_regressors[:, ii, 0] = np.arange(1, nt + 1) ** (ii)
        drift_regressors[:, ii, 0] = (
            drift_regressors[:, ii, 0] / drift_regressors[-1, ii, 0]
        )

    drift_regressors[:, :, 1] = drift_regressors[:, :, 0]

    regressor_names = []
    for i in range(drift_order + 1):
        regressor_names.append("Drift " + str(i))

    drift_regressors = xr.DataArray(
        drift_regressors,
        dims=["time", "regressor", "chromo"],
        coords={"time": t, "regressor": regressor_names, "chromo": ["HbO", "HbR"]},
    )
    drift_regressors = drift_regressors.pint.quantify("micromolar")

    return drift_regressors


def make_hrf_regressors(
    t: xr.DataArray, s: xr.DataArray, trange: list, idx_basis: str, params_basis
):

    cond_names = s.condition.values
    dt = 1 / t.cd.sampling_rate
    n_pre = round(trange[0] / dt)
    n_post = round(trange[1] / dt)
    t_hrf = np.arange(n_pre * dt, (n_post + 1) * dt, dt)
    nt = len(t)

    # prune good stim
    # handle case of conditions with 0 trials
    lst_cond = np.where(np.sum(s > 0, axis=0) > 0)[0]
    n_cond = len(lst_cond)
    onset = np.zeros((nt, n_cond))
    n_trials = np.zeros(n_cond)

    for i_cond in range(n_cond):
        lst_t = np.where(s[:, lst_cond[i_cond]] == 1)[0]
        lstp = np.where((lst_t + n_pre >= 0) & (lst_t + n_post < nt))[0]
        lst = lst_t[lstp]
        n_trials[i_cond] = len(lst)
        onset[lst + n_pre, i_cond] = 1

    tbasis, nb = construct_basis_functions(t_hrf, idx_basis, params_basis)

    # remove the third dimension if it is 1
    if len(tbasis.shape) == 3 and tbasis.shape[2] == 1:
        tbasis = tbasis[:, :, 0]

    if idx_basis != "individual":
        a_hrf = np.zeros((nt, nb * n_cond, 2))
        for i_conc in range(2):
            i_c = -1
            for i_cond in range(n_cond):
                for b in range(nb):
                    i_c += 1
                    if len(tbasis.shape) == 2:
                        clmn = np.convolve(onset[:, i_cond], tbasis[:, b])
                    else:
                        clmn = np.convolve(onset[:, i_cond], tbasis[:, b, i_conc])
                    clmn = clmn[:nt]
                    a_hrf[:, i_c, i_conc] = clmn

    elif idx_basis == "individual":
        a_hrf = np.zeros((nt, nb * n_cond, 2, params_basis.shape[2]))
        for i_conc in range(2):
            for i_ch in range(params_basis.shape[2]):
                i_c = -1
                for i_cond in range(n_cond):
                    for b in range(nb):
                        i_c += 1
                        clmn = np.convolve(
                            onset[:, i_cond], np.squeeze(tbasis[i_ch, :, b, i_conc])
                        )
                        clmn = clmn[:nt]
                        a_hrf[:, i_c, i_conc, i_ch] = clmn

    regressor_names = []

    for i_cond in range(n_cond):
        for b in range(nb):
            regressor_names.append("HRF " + cond_names[i_cond] + " " + str(b + 1))

    hrf_regs = xr.DataArray(
        a_hrf,
        dims=["time", "regressor", "chromo"],
        coords={"time": t, "regressor": regressor_names, "chromo": ["HbO", "HbR"]},
    )
    hrf_regs = hrf_regs.pint.quantify("micromolar")

    return hrf_regs


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
        # tbasis[:, 0, i_conc] = np.array((np.array(bas, dtype=np.complex128)) ** p, dtype=np.float64) * np.exp(p - t_hrf / q)
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
    ssMethod="closest",
    ssTresh: float = 1.5,
    as_xarray: bool = False,
):
    """Get short separation channels for each long channel.

    inputs:
        y: xarray.DataArray of the data (time x chromo x channels)
        geo3d: xarray.DataArray of the 3D geometry (number of sources/detectors x dim pos)
        ssMethod: method for determining short separation channels ("nearest", "correlation", "average")
        ssTresh: threshold for short separation channels (in cm)
        get_data: whether to return the short channel data (True) or the short channel names (False)
    outputs:
        ss: xarray.DataArray of short separation channels (channels x chromo)
    """

    # Calculate source-detector distances for each channel
    dists = (
        xrutils.norm(geo3d.loc[y.source] - geo3d.loc[y.detector], dim="pos")
        .pint.to("mm")
        .round(2)
    )

    # Identify short channels
    short_channels = dists.channel[dists < ssTresh * cedalion.units.cm]

    if ssMethod == "closest":
        middle_pos = (geo3d.loc[y.source] + geo3d.loc[y.detector]) / 2
        return closest_short_channel(y, short_channels, middle_pos, as_xarray)
    elif ssMethod == "corr":
        return max_corr_short_channel(y, short_channels, as_xarray)


def closest_short_channel(y, short_channels, middle_positions, as_xarray=False):
    # Initialize array to store closest short channel for each channel
    if as_xarray:
        closest_short = xr.DataArray(
            np.zeros((len(y.time), 2, len(y.channel))),
            dims=["time", "chromo", "channel"],
            coords={"time": y.time, "channel": y.channel, "chromo": y.chromo},
        )
        # add regressor dimension
        # closest_short = closest_short.assign_coords(regressor = y.channel)

    else:
        closest_short = xr.DataArray(
            [["______", "______"] for i in range(len(y.channel))],
            dims=["channel", "chromo"],
            coords={"channel": y.channel, "chromo": y.chromo},
        )
    # For each channel, find the closest short channel
    for ch in y.channel:
        # Compute distances from this channel's middle position to all short channel middle positions
        distances_to_short = xrutils.norm(
            middle_positions.loc[short_channels.channel] - middle_positions.loc[ch],
            dim="pos",
        )
        # Find the closest short channel
        if as_xarray:
            closest_short_channel = short_channels[(distances_to_short).argmin()]
            closest_name = short_channels.channel[(distances_to_short).argmin()]
        else:
            closest_short_channel = short_channels.channel[
                (distances_to_short).argmin()
            ]
        # Store the result
        closest_short.loc[ch, "HbO"] = closest_short_channel
        closest_short.loc[ch, "HbR"] = closest_short_channel
        if as_xarray:
            # get data not names of short channels
            pass
            # TODO: adjust the current regressor name to closest_name
    if not as_xarray:
        # transform to regressor dictionary
        closest_short = make_reg_dict(y, closest_short)

    return closest_short


def max_corr_short_channel(y, short_channels, as_xarray=False):
    # Get indices of short channels
    lstSS = [np.where(y.channel == ch)[0][0] for ch in short_channels]

    # HbO
    dc = y[:, 0, :].squeeze()
    dc = (dc - np.mean(dc, axis=0)) / np.std(dc, axis=0)
    cc1 = np.dot(dc.T, dc) / len(dc)

    # HbR
    dc = y[:, 1, :].squeeze()
    dc = (dc - np.mean(dc, axis=0)) / np.std(dc, axis=0)
    cc2 = np.dot(dc.T, dc) / len(dc)

    iNearestSS = np.zeros((cc1.shape[0], 2), dtype=int)
    for iML in range(cc1.shape[0]):
        iNearestSS[iML, 0] = lstSS[np.argmax(cc1[iML, lstSS])]
        iNearestSS[iML, 1] = lstSS[np.argmax(cc2[iML, lstSS])]

    channel_iSS = [y.channel[i] for i in iNearestSS]

    if as_xarray:
        # get data not names of short channels
        ss_data = np.zeros((len(y.time), 2, len(y.channel)))
        for i in range(len(y.channel)):
            ss_data[:, 0, i] = y[:, 0, iNearestSS[i, 0]]
            ss_data[:, 1, i] = y[:, 1, iNearestSS[i, 1]]
        highest_corr_short = xr.DataArray(
            ss_data,
            dims=["time", "chromo", "channel"],
            coords={"time": y.time, "channel": y.channel, "chromo": y.chromo},
        )
        # add regressor dimension
        highest_corr_short = highest_corr_short.assign_coords(
            regressor=(["channel", "chromo"], channel_iSS)
        )

    else:
        # transform to regressor dictionary
        highest_corr_short = xr.DataArray(
            channel_iSS,
            dims=["channel", "chromo"],
            coords={"channel": y.channel, "chromo": y.chromo},
        )
        highest_corr_short = make_reg_dict(y, highest_corr_short)

    return highest_corr_short


def make_reg_dict(y, add_regressors):
    """Make a dictionary with regressors as keys and channels as values.

    inputs:
        y: xarray.DataArray of the data (time x chromo x channels)
        add_regressors: xarray.DataArray containing the short channel regressors (channels x chromo)
    return:
        add_reg_dict: dictionary with regressors as keys and channels as values
    """

    unique_short = np.unique(add_regressors.values)

    ss_data = y.sel(channel=unique_short)
    # remove dimensions of data_2
    ss_data = ss_data.drop_vars(["samples", "source", "detector", "sbj"])
    ss_data = ss_data.rename({"channel": "regressor"})

    reg_dict = {}
    reg_dict["data"] = ss_data

    # fill dictionary: if channels have same regressor, merge them to one key
    for chromo in add_regressors.chromo.values:
        reg_dict[chromo] = {}
        add_reg_chromo = add_regressors.sel(chromo=chromo)
        for ch in np.unique(add_reg_chromo.values):
            reg_dict[chromo][ch] = add_reg_chromo.where(
                add_reg_chromo == ch, drop=True
            ).channel.values.tolist()
    return reg_dict
