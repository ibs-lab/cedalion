"""Temporal basis functions for the GLM."""

from __future__ import annotations
from abc import ABC, abstractmethod

import numpy as np
import pint
import xarray as xr

import cedalion.typing as cdt
from cedalion import Quantity, units
from cedalion.sigproc.frequency import sampling_rate
import cedalion.xrutils as xrutils


class TemporalBasisFunction(ABC):
    def __init__(self, convolve_over_duration: bool):
        self.convolve_over_duration = convolve_over_duration

    @abstractmethod
    def __call__(
        self,
        ts: cdt.NDTimeSeries,
    ) -> xr.DataArray:
        """Build the basis function for the given time series.

        Args:
            ts : a time series specifying the time axis and other dimensions

        Returns:
            A xr.DataArray with the basis function(s). The array has dimensions
            (time, component) or (time, component, other_dim) were other_dim is
            'chromo' or 'wavelength'.
        """
        raise NotImplementedError()


class GaussianKernelsWithTails(TemporalBasisFunction):
    r"""A consecutive sequence of gaussian functions.

    The basis functions have the form:

    .. math::
        f(t) = \exp( -(t-\mu)^2/t_{std}^2)

    The user specifies a time interval around the stimuls onset via the parameters
    t_pre and t_post. Over this time interval a series of gaussian basis functions is
    distributed:
        - between the gaussian centers there is  time gap of t_delta
        - the width of the each gaussian is specified by t_std
        - the gaussians are centered in the time interval with a margin of 3 x t_std
          left and right.
    The number of gaussians is derived automatically from these constraints.

    Args:
        t_pre (:class:`Quantity`, [time]): time before trial onset
        t_post (:class:`Quantity`, [time]): time after trial onset
        t_delta (:class:`Quantity`, [time]): the temporal spacing between consecutive
            gaussians
        t_std (:class:`Quantity`, [time]): time width of the gaussians
    """

    def __init__(
        self,
        t_pre: cdt.QTime,
        t_post: cdt.QTime,
        t_delta: cdt.QTime,
        t_std: cdt.QTime,
    ):
        super().__init__(convolve_over_duration=False)
        self.t_pre = _to_unit(t_pre, units.s)
        self.t_post = _to_unit(t_post, units.s)
        self.t_delta = _to_unit(t_delta, units.s)
        self.t_std = _to_unit(t_std, units.s)

    def __call__(
        self,
        ts: cdt.NDTimeSeries,
    ) -> xr.DataArray:
        fs = sampling_rate(ts).to(units.Hz)

        # create time-axis
        smpl_pre = int(np.ceil(self.t_pre * fs)) + 1
        smpl_post = int(np.ceil(self.t_post * fs)) + 1
        t_hrf = np.arange(-smpl_pre, smpl_post) / fs
        t_hrf = t_hrf.to("s")

        duration = t_hrf[-1] - t_hrf[0]

        # determine number of gaussians
        n_components = int(np.floor((duration - 6 * self.t_std) / self.t_delta))

        # place gaussians spaced by t_delta and centered in [-t_pre,pre].
        # centering needs an offset left and right from the interval boundaries
        t_offset = (duration - ((n_components - 1) * self.t_delta)) / 2
        mu = t_hrf[0] + t_offset + np.arange(n_components) * self.t_delta

        # build regressors. shape: (n_times, n_regressors)
        regressors = np.exp(
            -((t_hrf[:, None] - mu[None, :]) ** 2) / (2 * self.t_std) ** 2
        )
        regressors /= regressors.max(axis=0)  # normalize gaussian peaks to 1
        regressors = regressors.to_base_units().magnitude

        component_names = _generate_component_names(n_components)

        return xr.DataArray(
            regressors,
            dims=["time", "component"],
            coords={
                "time": xr.DataArray(t_hrf, dims=["time"]).pint.dequantify(),
                "component": component_names,
            },
        )


class GaussianKernels(TemporalBasisFunction):
    r"""A consecutive sequence of gaussian functions.

    The basis functions have the form:

    .. math::
        f(t) = \exp( -(t-\mu)^2/t_{std}^2)

    The user specifies a time interval around the stimuls onset via the parameters
    t_pre and t_post. Over this time interval a series of gaussian basis functions is
    distributed:
        - between the gaussian centers there is  time gap of t_delta
        - the width of the each gaussian is specified by t_std
        - the first gaussian is centered at trial onset - t_pre.
        - the model function extends strictly from -t_pre to t_post with a hard cutoff.

    The number of gaussians is derived automatically from these constraints.

    Args:
        t_pre (:class:`Quantity`, [time]): time before trial onset
        t_post (:class:`Quantity`, [time]): time after trial onset
        t_delta (:class:`Quantity`, [time]): the temporal spacing between consecutive
            gaussians
        t_std (:class:`Quantity`, [time]): time width of the gaussians
    """

    def __init__(
        self,
        t_pre: cdt.QTime,
        t_post: cdt.QTime,
        t_delta: cdt.QTime,
        t_std: cdt.QTime,
    ):
        super().__init__(convolve_over_duration=False)
        self.t_pre = _to_unit(t_pre, units.s)
        self.t_post = _to_unit(t_post, units.s)
        self.t_delta = _to_unit(t_delta, units.s)
        self.t_std = _to_unit(t_std, units.s)

    def __call__(
        self,
        ts: cdt.NDTimeSeries,
    ) -> xr.DataArray:
        fs = sampling_rate(ts).to(units.Hz)

        # create time-axis
        smpl_pre = int(np.ceil(self.t_pre * fs)) + 1
        smpl_post = int(np.ceil(self.t_post * fs)) + 1
        t_hrf = np.arange(-smpl_pre, smpl_post) / fs
        t_hrf = t_hrf.to("s")

        duration = t_hrf[-1] - t_hrf[0]

        # determine number of gaussians
        n_components = int(np.floor(duration / self.t_delta))

        # place gaussians spaced by t_delta and starting at -t_pre.
        mu = t_hrf[0] + np.arange(n_components) * self.t_delta

        # build regressors. shape: (n_times, n_regressors)
        regressors = np.exp(
            -((t_hrf[:, None] - mu[None, :]) ** 2) / (2 * self.t_std) ** 2
        )
        regressors /= regressors.max(axis=0)  # normalize gaussian peaks to 1
        regressors = regressors.to_base_units().magnitude

        component_names = _generate_component_names(n_components)

        return xr.DataArray(
            regressors,
            dims=["time", "component"],
            coords={
                "time": xr.DataArray(t_hrf, dims=["time"]).pint.dequantify(),
                "component": component_names,
            },
        )


class Gamma(TemporalBasisFunction):
    r"""Modified gamma function, optionally convolved with a square-wave.

    The basis function has the form:

    .. math::
        f(t) \sim \frac{t-\tau}{\sigma}
                  \exp \left(-\left(\frac{t - \tau}{\sigma}\right)^2\right)

    Args:
        tau: Specifies a delay of the response with respect ot stimulus onset time.
        sigma: Specifies the width of the hemodynamic reponse.
        T : If > 0, the response is additionally convoluted by a square wave of this
            width.
    """

    def __init__(
        self,
        tau: cdt.QTime | dict[str, cdt.QTime],
        sigma: cdt.QTime | dict[str, cdt.QTime],
        T: cdt.QTime | dict[str, cdt.QTime],  # noqa: N803
    ):
        super().__init__(convolve_over_duration=True)
        self.tau = _to_unit(tau, units.s)
        self.sigma = _to_unit(sigma, units.s)
        self.T = _to_unit(T, units.s)

    def __call__(
        self,
        ts: cdt.NDTimeSeries,
    ) -> xr.DataArray:
        other_dim = xrutils.other_dim(ts, "time", "channel")
        other_dim_values = ts[other_dim].values

        tau = _to_dict(self.tau, other_dim_values)
        sigma = _to_dict(self.sigma, other_dim_values)
        T = _to_dict(self.T, other_dim_values)  # noqa: N806

        # create time-axis.
        # estimate duration: x^2*exp(-x^2) drops below 1e-6 at x=4.1
        fs = sampling_rate(ts).to(units.Hz)
        duration = 4.1 * max(sigma.values()) + max(tau.values()) + max(T.values())
        n_samples = int(np.ceil(duration * fs))
        t_hrf = (np.arange(n_samples) / fs).to(units.s)

        n_components = 1
        n_other_dim = ts.sizes[other_dim]

        regressors = np.zeros((n_samples, n_components, n_other_dim))

        for i_other, other in enumerate(other_dim_values):
            x2 = np.power((t_hrf - tau[other]) / sigma[other], 2).magnitude
            r = x2 * np.exp(-x2)
            r[t_hrf < tau[other]] = 0.0

            if T[other] > 0:
                width = int(T[other] * fs)
                r = np.convolve(r, np.ones(width) / width)
                r = r[:n_samples]

            r /= r.max()
            regressors[:, 0, i_other] = r

        return xr.DataArray(
            regressors,
            dims=["time", "component", other_dim],
            coords={
                "time": xr.DataArray(t_hrf, dims=["time"]).pint.dequantify(),
                other_dim: other_dim_values,
                "component": ["gamma"],
            },
        )


class GammaDeriv(TemporalBasisFunction):
    """Modified gamma func. and its derivative, optionally convolved with a square-wave.

    Args:
        tau: onset time
        sigma: width of the HRF
        T : convolution width
    """

    def __init__(
        self,
        tau: cdt.QTime | dict[str, cdt.QTime],
        sigma: cdt.QTime | dict[str, cdt.QTime],
        T: cdt.QTime | dict[str, cdt.QTime],  # noqa: N803
    ):
        super().__init__(convolve_over_duration=True)
        self.tau = _to_unit(tau, units.s)
        self.sigma = _to_unit(sigma, units.s)
        self.T = _to_unit(T, units.s)

    def __call__(
        self,
        ts: cdt.NDTimeSeries,
    ) -> xr.DataArray:
        other_dim = xrutils.other_dim(ts, "time", "channel")
        other_dim_values = ts[other_dim].values

        tau = _to_dict(self.tau, other_dim_values)
        sigma = _to_dict(self.sigma, other_dim_values)
        T = _to_dict(self.T, other_dim_values)  # noqa: N806

        fs = sampling_rate(ts).to(units.Hz)
        duration = 4.1 * max(sigma.values()) + max(tau.values()) + max(T.values())
        n_samples = int(np.ceil(duration * fs))
        t_hrf = (np.arange(n_samples) / fs).to(units.s)

        n_components = 2
        n_other_dim = ts.sizes[other_dim]

        regressors = np.zeros((n_samples, n_components, n_other_dim))

        for i_other, other in enumerate(other_dim_values):
            x = (t_hrf - tau[other]) / sigma[other]
            x2 = x.magnitude**2
            r = x2 * np.exp(-x2)
            dr = (2 * x * (1 - x2)) * np.exp(-x2)
            dr = dr.magnitude
            r[t_hrf < tau[other]] = 0.0
            dr[t_hrf < tau[other]] = 0.0

            if T[other] > 0:
                width = int(T[other] * fs)
                r = np.convolve(r, np.ones(width) / width)[:n_samples]
                dr = np.convolve(dr, np.ones(width) / width)[:n_samples]

            r /= r.max()
            dr /= dr.max()
            regressors[:, 0, i_other] = r
            regressors[:, 1, i_other] = dr

        return xr.DataArray(
            regressors,
            dims=["time", "component", other_dim],
            coords={
                "time": xr.DataArray(t_hrf, dims=["time"]).pint.dequantify(),
                other_dim: other_dim_values,
                "component": ["gamma", "gamma_deriv"],
            },
        )


class AFNIGamma(TemporalBasisFunction):
    """AFNI gamma basis function, optionally convolved with a square-wave.

    Args:
        p: shape parameter
        q: scale parameter
        T : convolution width
    """

    def __init__(
        self,
        p: float | dict[str, float],
        q: cdt.QTime | dict[str, cdt.QTime],
        T: cdt.QTime | dict[str, cdt.QTime],  # noqa: N803
    ):
        super().__init__(convolve_over_duration=True)
        self.p = p
        self.q = _to_unit(q, units.s)
        self.T = _to_unit(T, units.s)

    def __call__(
        self,
        ts: cdt.NDTimeSeries,
    ) -> xr.DataArray:
        other_dim = xrutils.other_dim(ts, "time", "channel")
        other_dim_values = ts[other_dim].values

        p = _to_dict(self.p, other_dim_values)
        q = _to_dict(self.q, other_dim_values)
        T = _to_dict(self.T, other_dim_values)  # noqa: N806

        fs = sampling_rate(ts).to(units.Hz)
        duration = 4.1 * max(q.values()) + max(T.values())
        duration = (duration.to(units.s).magnitude + max(p.values())) * units.s
        # add p to duration. but p has not unit and duration is in seconds.
        # so we need to convert p to seconds.
        n_samples = int(np.ceil(duration * fs))
        t_hrf = (np.arange(n_samples) / fs).to(units.s)

        n_components = 1
        n_other_dim = ts.sizes[other_dim]

        regressors = np.zeros((n_samples, n_components, n_other_dim))

        for i_other, other in enumerate(other_dim_values):
            bas = t_hrf / (p[other] * q[other])
            r = np.power(bas.magnitude, p[other]) * np.exp(p[other] - t_hrf / q[other])
            r[t_hrf < 0] = 0.0
            r = r.magnitude

            if T[other] > 0:
                width = int(T[other] * fs)
                r = np.convolve(r, np.ones(width) / width)[:n_samples]

            r /= r.max()
            regressors[:, 0, i_other] = r

        return xr.DataArray(
            regressors,
            dims=["time", "component", other_dim],
            coords={
                "time": xr.DataArray(t_hrf, dims=["time"]).pint.dequantify(),
                other_dim: other_dim_values,
                "component": ["afni_gamma"],
            },
        )


# FIXME: instead of defining IndividualBasis we may want to make make_hrf_regressor
# accept xr.DataArrays directly?

# class IndividualBasis(TemporalBasisFunction):
#    """Uses individual basis functions for each channel.
#
#    Args:
#        basis_fcts: The individual basis functions with shape
#            (n_times, n_channels, n_other_dim) or (n_times, n_other_dim) if basis
#            function is the same for all channels.
#    """
#
#    # FIXME: we have a different basis function for each channel:
#    #        how to incorporate this into the channel wise regressor logic?
#
#    def __init__(self, basis_fcts: cdt.NDTimeSeries):
#        super().__init__(convolve_over_duration=False)
#        self.basis_fcts = basis_fcts
#
#    def __call__(
#        self,
#        ts: cdt.NDTimeSeries,
#    ) -> xr.DataArray:
#        if len(self.basis_fcts.shape) != 3:
#            raise ValueError("Invalid shape for basis functions. Expected"
#                             "(time, channel, other_dim).")
#        n_channels_in_basis = self.basis_fcts.sizes['channel']
#        n_channels = ts.sizes['channel']
#        other_dim, other_dim_values = get_other_dim(ts)
#
#        if n_channels != n_channels_in_basis:
#            raise ValueError("Number of channels in the time series does not match the"
#                             " number of channels in the basis functions.")
#        regressors = self.basis_fcts.transpose("time", other_dim, "channel")
#        regressors = np.expand_dims(regressors, axis=1)
#
#        return xr.DataArray(
#            regressors,
#            dims=["time", "component", other_dim, "channel"],
#            coords={
#                "time": self.basis_fcts.time,
#                "component": ["individual"],
#                other_dim: other_dim_values,
#                "channel": self.basis_fcts.channel
#            }
#        )


def _generate_component_names(n_components: int) -> list[str]:
    """Create a list of same-width, zero-padded strings from 0 to n_components-1."""

    width = int(np.ceil(np.log10(n_components)))
    fmt = "{:0%dd}" % width
    return [fmt.format(i) for i in range(n_components)]


def _to_unit(obj: Quantity | dict[str, Quantity], unit: str | pint.Unit):
    """Sets the unit of this quantity or the quantity values in the dict."""

    if isinstance(obj, Quantity):
        return obj.to(unit)
    elif isinstance(obj, dict):
        return {k: q.to(unit) for k, q in obj.items()}
    else:
        raise ValueError(f"type of obj ({type(obj)})  is not supported.")


def _to_dict(param: Quantity | dict[str, Quantity], keys):
    """Extends a parameter to a dict."""

    if isinstance(param, dict):
        # param is already a dict. check that all keys are present
        if set(param.keys()) != set(keys):
            raise ValueError("parameter is a dict but is missing keys.")
        return param
    else:
        # single parameter gets extended to a dict. same value for all keys.
        return {k: param for k in keys}
