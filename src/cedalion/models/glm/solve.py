"""Solve the GLM model."""

from __future__ import annotations
from collections import defaultdict

import numpy as np
import xarray as xr
from nilearn.glm.first_level import run_glm as nilearn_run_glm

import cedalion.typing as cdt
import cedalion.xrutils as xrutils

def _hash_channel_wise_regressor(regressor: xr.DataArray) -> list[int]:
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
    """Predict time series from design matrix and thetas.

    Args:
        ts: The time series to be modeled.
        thetas: The estimated parameters.
        design_matrix: DataArray with dims time, regressor, chromo.
        channel_wise_regressors: Optional list of design matrices, with additional
            channel dimension.

    Returns:
        The predicted time series.
    """
    dim3_name = xrutils.other_dim(design_matrix, "time", "regressor")

    prediction = defaultdict(list)

    for dim3, group_channels, group_design_matrix in iter_design_matrix(
        ts, design_matrix, channel_wise_regressors
    ):
        # (dim3, channel, regressor)
        t = thetas.sel({"channel": group_channels, dim3_name: [dim3]})
        prediction[dim3].append(xr.dot(group_design_matrix, t, dim="regressor"))

        """
        tmp = xr.dot(group_design_matrix, t, dim="regressor")

        # Drop coordinates that are in group_design_matrix but which we don't want
        # to be in the predicted time series. This is currently formulated as a
        # negative list. A positive list might be the better choice, though.
        tmp = tmp.drop_vars(
            [i for i in ["short_channel", "comp_group"] if i in tmp.coords]
        )

        prediction[dim3].append(tmp)
        """

    # concatenate channels
    prediction = [xr.concat(v, dim="channel") for v in prediction.values()]

    # concatenate dim3
    prediction = xr.concat(prediction, dim=dim3_name)

    return prediction


def iter_design_matrix(
    ts: cdt.NDTimeSeries,
    design_matrix: xr.DataArray,
    channel_wise_regressors: list[xr.DataArray] | None = None,
    channel_groups: list[int] | None = None,
):
    """Iterate over the design matrix and yield the design matrix for each group.

    Args:
        ts: The time series to be modeled.
        design_matrix: DataArray with dims time, regressor, chromo.
        channel_wise_regressors: Optional list of
            design matrices, with additional channel dimension.
        channel_groups: Optional list of channel groups.

    Yields:
        tuple: A tuple containing:
            - dim3 (str): The third dimension name.
            - group_y (cdt.NDTimeSeries): The grouped time series.
            - group_design_matrix (xr.DataArray): The grouped design matrix.
    """
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
            comp_groups.append(_hash_channel_wise_regressor(reg))

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
