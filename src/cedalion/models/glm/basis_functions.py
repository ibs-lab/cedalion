from abc import ABC, abstractmethod
from typing import Annotated

import numpy as np
import pint
import xarray as xr

import cedalion.typing as cdt
from cedalion import Quantity, units
from cedalion.sigproc.frequency import sampling_rate


def generate_component_names(n_components: int) -> list[str]:
    width = int(np.ceil(np.log10(n_components)))
    fmt = fmt = "{:0%dd}" % width
    return [fmt.format(i) for i in range(n_components)]


def to_unit(obj: Quantity | dict[str, Quantity], unit: str | pint.Unit):
    """Sets the unit of this quantity or the quantity values in the dict."""

    if isinstance(obj, Quantity):
        return obj.to(unit)
    elif isinstance(obj, dict):
        return {k: q.to(unit) for k, q in obj.items()}
    else:
        raise ValueError(f"type of obj ({type(obj)})  is not supported.")


def to_dict(param: Quantity | dict[str, Quantity], keys):
    """Extends a parameter to a dict."""

    if isinstance(param, dict):
        # param is already a dict. check that all keys are present
        if set(param.keys()) != set(keys):
            raise ValueError("parameter is a dict but is missing keys.")
        return param
    else:
        # single parameter gets extended to a dict. same value for all keys.
        return {k: param for k in keys}


def get_other_dim(ts: cdt.NDTimeSeries):
    """Returns the name of 3rd dimension (chromo, wavelength) and it coords."""

    other_dim = [d for d in ts.dims if d != "channel" and d != "time"]
    assert len(other_dim) == 1
    other_dim = other_dim[0]

    other_dim_values = ts[other_dim].values

    return other_dim, other_dim_values


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


class GaussianKernels(TemporalBasisFunction):
    """A consecutive sequence of gaussian functions.

    Args:
        t_pre (:class:`Quantity`, [time]): time before trial onset
        t_post (:class:`Quantity`, [time]): time after trial onset
        t_delta (:class:`Quantity`, [time]): the temporal spacing between consecutive
            gaussians
        t_std (:class:`Quantiantity`, [time]): time width of the gaussians
    """

    def __init__(
        self,
        t_pre: Annotated[Quantity, "[time]"],
        t_post: Annotated[Quantity, "[time]"],
        t_delta: Annotated[Quantity, "[time]"],
        t_std: Annotated[Quantity, "[time]"],
    ):
        super().__init__(convolve_over_duration=False)
        self.t_pre = to_unit(t_pre, units.s)
        self.t_post = to_unit(t_post, units.s)
        self.t_delta = to_unit(t_delta, units.s)
        self.t_std = to_unit(t_std, units.s)

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

        component_names = generate_component_names(n_components)

        return xr.DataArray(
            regressors,
            dims=["time", "component"],
            coords={
                "time": xr.DataArray(t_hrf, dims=["time"]).pint.dequantify(),
                "component": component_names,
            },
        )


class Gamma(TemporalBasisFunction):
    """Modified gamma function, optionally convolved with a square-wave.

    Args:
        tau: onset time
        sigma: width of the HRF
        T : convolution width
    """

    def __init__(
        self,
        tau: Annotated[Quantity, "[time]"] | dict[str, Annotated[Quantity, "[time]"]],
        sigma: Annotated[Quantity, "[time]"] | dict[str, Annotated[Quantity, "[time]"]],
        T: Annotated[Quantity, "[time]"] | dict[str, Annotated[Quantity, "[time]"]],  # noqa: N803
    ):
        super().__init__(convolve_over_duration=True)
        self.tau = to_unit(tau, units.s)
        self.sigma = to_unit(sigma, units.s)
        self.T = to_unit(T, units.s)

    def __call__(
        self,
        ts: cdt.NDTimeSeries,
    ) -> xr.DataArray:
        other_dim, other_dim_values = get_other_dim(ts)

        tau = to_dict(self.tau, other_dim_values)
        sigma = to_dict(self.sigma, other_dim_values)
        T = to_dict(self.T, other_dim_values)  # noqa: N806

        # create time-axis.
        # estimate duration: x^2*exp(-x^2) drops below 1e-6 at x=4.1
        fs = sampling_rate(ts).to(units.Hz)
        duration = 4.1 * max(sigma.values()) + max(tau.values()) + max(T.values())
        n_samples = int(np.ceil(duration * fs))
        t_hrf = (np.arange(n_samples) / fs).to(units.s)

        # determine number of gaussians
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
            },
        )
