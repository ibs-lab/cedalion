from collections import defaultdict

import numpy as np
import xarray as xr
from nilearn.glm.first_level import run_glm as nilearn_run_glm

import cedalion.typing as cdt
import cedalion.xrutils as xrutils

# def solve_glm(
#    y: xr.DataArray,
#    design_matrix: xr.DataArray,
#    local_regressors={"HbO": {}, "HbR": {}},
#    noise_model="ols",
# ):
#    """Solve the GLM for a given design matrix, channel-dependent regressors and data.
#
#    Args:
#        y: xarray.DataArray containing the data (time x chromo x channels)
#        design_matrix: design matrix used for the GLM (time x regressors x chromo)
#        local_regressors: dict containing additional regressors for each chromophore
#            local_regressors['data']: xrDataArray containing the additional
#                regressors (regressor x chromo x channels)
#            local_regressors['HbO']: dict containing the additional regressors for HbO
#                local_regressors['HbO']['regressor_1']: list of HbO channels assigned
#                to regressor_1
#            local_regressors['HbR']: dict containing the additional regressors for HbR
#                local_regressors['HbR']['regressor_1']: list of HbR channels assigned
#                to regressor_1
#        noise_model: noise model used for the GLM (default = 'ols')
#
#    Returns:
#        thetas: xarray.DataArray estimated parameters of the GLM
#            (regressors x chromo x channels)
#        predicted: xarray.DataArray predicted data (time x chromo x channels)
#        predicted_hrf: xarray.DataArray predicted HRFs (time x chromo x channels)
#    """
#
#    thetas = xr.DataArray(
#        np.zeros((design_matrix.regressor.size, y.chromo.size, y.channel.size)),
#        dims=["regressor", "chromo", "channel"],
#        coords={
#            "regressor": design_matrix.regressor.values.tolist(),
#            "chromo": y.chromo,
#            "channel": y.channel,
#        },
#    )
#
#    predicted = xr.DataArray(
#        np.zeros((y.time.size, y.chromo.size, y.channel.size)),
#        dims=["time", "chromo", "channel"],
#        coords={"time": y.time, "chromo": y.chromo, "channel": y.channel},
#    )
#
#    predicted_hrf = xr.DataArray(
#        np.zeros((y.time.size, y.chromo.size, y.channel.size)),
#        dims=["time", "chromo", "channel"],
#        coords={"time": y.time, "chromo": y.chromo, "channel": y.channel},
#    )
#
#    # save units
#    units = y.pint.units
#    y = y.pint.dequantify()
#    design_matrix = design_matrix.pint.dequantify()
#    if "data" in local_regressors:
#        local_regressors["data"] = local_regressors["data"].pint.dequantify()
#
#    # process local_regressors so that
#    # each channel is assigned to not more than one regressor
#    local_regressors["HbO"] = process_regressors(
#        local_regressors["HbO"], y.channel.values
#    )
#    local_regressors["HbR"] = process_regressors(
#        local_regressors["HbR"], y.channel.values
#    )
#
#    hrf_regs = design_matrix.regressor.sel(
#        regressor=design_matrix.regressor.str.startswith("HRF")
#    )
#
#    for chromo in y.chromo.values:
#        for assigned_regs, ch in local_regressors[chromo].items():
#            # add assigned regressor to design matrix
#            if assigned_regs != ():
#                add_reg = local_regressors["data"].sel(
#                    regressor=np.array(assigned_regs), chromo=chromo
#                )
#                # add_reg = add_reg.pint.quantify("micromolar") / abs(add_reg).max()
#                dm = xr.concat(
#                    [design_matrix.sel(chromo=chromo), add_reg], dim="regressor"
#                )
#            else:
#                dm = design_matrix.sel(chromo=chromo)
#            # solve GLM for current chromophore and channels
#            thetas_temp = run_glm(
#                y.sel(channel=ch, chromo=chromo), dm, noise_model=noise_model
#            )
#            # store results
#            thetas.loc[{"chromo": chromo, "channel": ch}] = thetas_temp.sel(
#                regressor=thetas.regressor
#            )
#            predicted.loc[{"time": y.time, "chromo": chromo, "channel": ch}] = (
#                dm.values @ thetas_temp.values
#            )
#            predicted_hrf.loc[{"time": y.time, "chromo": chromo, "channel": ch}] = (
#                dm.sel(regressor=hrf_regs).values
#                @ thetas_temp.sel(regressor=hrf_regs).values
#            )
#
#    predicted = predicted.pint.quantify(units)
#    predicted_hrf = predicted_hrf.pint.quantify(units)
#
#    return thetas, predicted, predicted_hrf


def hash_channel_wise_regressor(regressor: xr.DataArray) -> list[int]:
    """Hashes each channel slice of the regressor array.

    Args:
        regressor: array of channel-wise regressors. Dims
            (channel, regressor, time, chromo|wavelength)

    Returns:
        A list of hash values, one hash for each channel.
    """

    tmp = regressor.pint.dequantify()
    n_channel = regressor.sizes["channel"]
    return [hash(tmp.isel(channel=i).values.data.tobytes()) for i in range(n_channel)]


def iter_design_matrix(
    ts: cdt.NDTimeSeries,
    design_matrix: xr.DataArray,
    channel_wise_regressors: list[xr.DataArray] | None = None,
    channel_groups: list[int] | None = None,
):
    dim3_name = xrutils.other_dim(design_matrix, "time", "regressor")

    if channel_wise_regressors is None:
        channel_wise_regressors = []

    for cwreg in channel_wise_regressors:
        assert cwreg.sizes["regressor"] == 1
        assert (ts.channel.values == cwreg.channel.values).all()

    comp_groups = []
    for reg in channel_wise_regressors:
        if "comp_group" in reg.coords:
            comp_groups.append(reg["comp_group"].values)
        else:
            comp_groups.append(hash_channel_wise_regressor(reg))

    if channel_groups is not None:
        assert len(channel_groups) == ts.sizes["channel"]
        comp_groups.append(channel_groups)

    if len(comp_groups) == 0:
        # There are no channel-wise regressors. Just iterate over the third dimension
        # of the design matrix.
        for dim3 in design_matrix[dim3_name].values:
            dm = design_matrix.sel({dim3_name: dim3})
            # group_y = ts.sel({dim3_name: dim3})
            channels = ts.channel.values
            # yield dim3, group_y, dm
            yield dim3, channels, dm

        return
    else:
        # there are channel-wise regressors. For each computational group, in which
        # the channel-wise regressors are identical, we have to assemble and yield the
        # design-matrix.

        chan_idx_with_same_comp_group = defaultdict(list)

        for i_ch, all_comp_groups in enumerate(zip(*comp_groups)):
            chan_idx_with_same_comp_group[all_comp_groups].append(i_ch)

        for dim3 in design_matrix[dim3_name].values:
            dm = design_matrix.sel({dim3_name: dim3})

            for chan_indices in chan_idx_with_same_comp_group.values():
                channels = ts.channel[np.asarray(chan_indices)].values

                regs = []
                for reg in channel_wise_regressors:
                    regs.append(
                        reg.sel({"channel": channels, dim3_name: dim3})
                        .isel(channel=0)  # regs are identical within a group
                        .pint.dequantify()
                    )

                group_design_matrix = xr.concat([dm] + regs, dim="regressor")
                # group_y = ts.sel({"channel": channels, dim3_name: dim3})

                # yield dim3, group_y, group_design_matrix
                yield dim3, channels, group_design_matrix


def fit(
    ts: cdt.NDTimeSeries,
    design_matrix: xr.DataArray,
    channel_wise_regressors: list[xr.DataArray] | None = None,
    noise_model="ols",
):
    """Fit design matrix to data.

    Args:
        ts: the time series to be modeled
        design_matrix: DataArray with dims time, regressor, chromo
        channel_wise_regressors: optional list of design matrices, with additional
            channel dimension
        noise_model: must be 'ols' for the moment

    Returns:
        thetas as a DataArray

    """
    if noise_model != "ols":
        raise NotImplementedError("support for other noise models is missing")

    # FIXME: unit handling?
    # shoud the design matrix be dimensionless? -> thetas will have units
    ts = ts.pint.dequantify()
    design_matrix = design_matrix.pint.dequantify()

    dim3_name = xrutils.other_dim(design_matrix, "time", "regressor")

    thetas = defaultdict(list)

    for dim3, group_channels, group_design_matrix in iter_design_matrix(
        ts, design_matrix, channel_wise_regressors
    ):
        group_y = ts.sel({"channel": group_channels, dim3_name: dim3}).transpose(
            "time", "channel"
        )

        _, glm_est = nilearn_run_glm(
            group_y.values, group_design_matrix.values, noise_model=noise_model
        )
        assert len(glm_est) == 1  # FIXME, holds only for OLS
        glm_est = next(iter(glm_est.values()))

        thetas[dim3].append(
            xr.DataArray(
                glm_est.theta[:, :, None],
                dims=("regressor", "channel", dim3_name),
                coords={
                    "regressor": group_design_matrix.regressor,
                    "channel": group_y.channel,
                    dim3_name: [dim3],
                },
            )
        )

    # concatenate channels
    thetas = [xr.concat(v, dim="channel") for v in thetas.values()]

    # concatenate dim3
    thetas = xr.concat(thetas, dim=dim3_name)

    return thetas


def predict(
    ts: cdt.NDTimeSeries,
    thetas: xr.DataArray,
    design_matrix: xr.DataArray,
    channel_wise_regressors: list[xr.DataArray] | None = None,
) -> cdt.NDTimeSeries:
    dim3_name = xrutils.other_dim(design_matrix, "time", "regressor")

    prediction = defaultdict(list)

    for dim3, group_channels, group_design_matrix in iter_design_matrix(
        ts, design_matrix, channel_wise_regressors
    ):
        # (dim3, channel, regressor)
        t = thetas.sel({"channel": group_channels, dim3_name: [dim3]})
        prediction[dim3].append(xr.dot(group_design_matrix, t, dim="regressor"))

    # concatenate channels
    prediction = [xr.concat(v, dim="channel") for v in prediction.values()]

    # concatenate dim3
    prediction = xr.concat(prediction, dim=dim3_name)

    return prediction


# def process_regressors(regressors, all_channels):
#    # Identify all unique channels and which regressors they belong to
#    channel_to_regressors = {}
#    for regressor, channels in regressors.items():
#        for channel in channels:
#            if channel in channel_to_regressors:
#                channel_to_regressors[channel].append(regressor)
#            else:
#                channel_to_regressors[channel] = [regressor]
#
#    # Include extra_channels in the mapping without assigning them yet
#    for channel in all_channels:
#        if channel not in channel_to_regressors:
#            channel_to_regressors[channel] = []
#
#    # Prepare final assignments structure
#    final_assignments = {regressor: [] for regressor in regressors}
#
#    # Assign channels to regressors or tuples, resolving conflicts
#    for channel, assigned_regressors in channel_to_regressors.items():
#        if len(assigned_regressors) == 1:
#            # Channel is unique to one regressor
#            final_assignments[assigned_regressors[0]].append(channel)
#        elif len(assigned_regressors) > 1:
#            # Channel is shared; create or update a tuple entry
#            assigned_regressors.sort()
#            tuple_key = tuple(assigned_regressors)
#            if tuple_key not in final_assignments:
#                final_assignments[tuple_key] = [channel]
#            else:
#                final_assignments[tuple_key].append(channel)
#
#    # Assign leftover channels to a new regressor (empty tuple) if they exist
#    assigned_channels = set(
#        [channel for channels in final_assignments.values() for channel in channels]
#    )
#    leftover_channels = [
#        channel for channel in all_channels if channel not in assigned_channels
#    ]
#    if leftover_channels:
#        final_assignments[tuple()] = leftover_channels
#
#    # Clean up the structure before returning
#    for key in list(final_assignments.keys()):
#        if not final_assignments[key]:
#            del final_assignments[key]  # Remove empty lists to clean up final output
#        else:
#            final_assignments[key].sort()  # Sort channels for readability
#
#    return final_assignments
#
#
# def run_glm(y, dm, noise_model="ols"):
#    """Run the GLM for a given design matrix and data.
#
#    Args:
#        y: xarray.DataArray containing the data (time x chromo x channels)
#        dm: design matrix used for the GLM (time x regressor x chromo)
#        noise_model: noise model used for the GLM (default = 'ols')
#
#    Returns:
#        thetas: xarray.DataArray estimated parameters of the GLM
#            (regressors x chromo x channels)
#    """
#
#    if y.pint.units:
#        y = y.pint.dequantify()
#
#    if dm.pint.units:
#        dm = dm.pint.dequantify()
#
#    # if y contains chromo dimension, run GLM for each chromophore separately
#    if "chromo" in y.dims:
#        chromo_results = []
#        for chromo in y.chromo:
#            thetas_temp = run_glm(
#                y.sel(chromo=chromo), dm.sel(chromo=chromo), noise_model=noise_model
#            )
#            thetas_temp = thetas_temp.assign_coords(chromo=chromo)
#            chromo_results.append(thetas_temp)
#        return xr.concat(chromo_results, dim="chromo")
#    # if y contains only 1 channel, expand dims
#    if "channel" not in y.dims:
#        labels, glm_est = nilearn_run_glm(
#            np.expand_dims(y, 1), dm, noise_model=noise_model
#        )
#        thetas = xr.DataArray(
#            np.squeeze(glm_est[labels[0]].theta),
#            dims=("regressor"),
#            coords={"regressor": dm.regressor},
#        )
#    else:
#        labels, glm_est = nilearn_run_glm(y, dm, noise_model=noise_model)
#        thetas = xr.DataArray(
#            glm_est[labels[0]].theta,
#            dims=("regressor", "channel"),
#            coords={"regressor": dm.regressor, "channel": y.channel},
#        )
#
#    return thetas
#
#
# def get_HRFs(
#    predicted_hrf: xr.DataArray,
#    stim: pd.DataFrame,
#    id_stim: int = 0,
#    HRFmin: int = -2,
#    HRFmax: int = 15,
# ):
#    """Get HRFs for each condition and channel estimated by the GLM.
#
#    Args:
#        predicted_hrf: xarray.DataArray containing the predicted HRFs
#            (time x chromo x channels)
#        stim: pandas.DataFrame containing the stimulus information
#        id_stim: id of the stimulus block for which the HRFs are estimated (default=0)
#        HRFmin: minimum relative time of the HRF (default = -2)
#        HRFmax: maximum relative time of the HRF (default = 15)
#
#    Return:
#        hrfs: xarray.DataArray containing HRFs for every condition and every channel
#            (time x chromo x channels x conditions)
#    """
#
#    dt = 1 / predicted_hrf.cd.sampling_rate
#
#    unit = predicted_hrf.pint.units
#    predicted_hrf = predicted_hrf.pint.dequantify()
#
#    # get id_stim-th stim onset of each condition
#    stim_onsets = stim.groupby("trial_type").onset.nth(id_stim).values
#    conds = stim.trial_type.unique()
#    stim_onsets = xr.DataArray(
#        stim_onsets, dims="condition", coords={"condition": conds}
#    )
#
#    # get time axis for HRFs:
#
#    # time_hrf = predicted_hrf.sel(
#    #    time=slice(
#    #        stim_onsets.sel(condition=conds[0]) + HRFmin,
#    #        stim_onsets.sel(condition=conds[0]) + HRFmax,
#    #    ),
#    # ).time - stim_onsets.sel(condition=conds[0])
#
#    n_pre = round(HRFmin / dt)
#    n_post = round(HRFmax / dt)
#    # np.arange results can be non-consistent when using non-integer steps
#    # t_hrf = np.arange(n_pre * dt, (n_post + 1) * dt, dt)
#    # using linspace instead
#    time_hrf = np.linspace(HRFmin, HRFmax, abs(n_post) + abs(n_pre) + 1)
#
#    hrfs = xr.DataArray(
#        np.zeros(
#            (
#                time_hrf.size,
#                predicted_hrf.chromo.size,
#                predicted_hrf.channel.size,
#                conds.size,
#            )
#        ),
#        dims=["time", "chromo", "channel", "condition"],
#        coords={
#            "time": time_hrf,
#            "chromo": predicted_hrf.chromo,
#            "channel": predicted_hrf.channel,
#            "condition": conds,
#        },
#    )
#
#    for chromo in predicted_hrf.chromo:
#        for cond in conds:
#            # select HRFs for current chromophore and condition
#            hrf = predicted_hrf.sel(
#                chromo=chromo,
#                time=slice(
#                    predicted_hrf.time.sel(
#                        time=stim_onsets.sel(condition=cond) + HRFmin, method="nearest"
#                    ),
#                    predicted_hrf.time.sel(
#                        time=stim_onsets.sel(condition=cond) + HRFmax, method="nearest"
#                    ),
#                ),
#            )
#            # change time axis to relative time
#            hrf = hrf.assign_coords(time=time_hrf)
#            # store HRFs
#            hrfs.loc[{"time": time_hrf, "chromo": chromo, "condition": cond}] = hrf
#
#    # remove baseline
#    hrfs = hrfs - hrfs.sel(time=slice(HRFmin, 0)).mean(dim="time")
#    # add units
#    hrfs = hrfs.pint.quantify(unit)
#
#    return hrfs
