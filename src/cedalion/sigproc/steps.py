import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.io
import cedalion.nirs
import cedalion.sigproc.motion
import cedalion.sigproc.quality
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units
from cedalion.geometry.landmarks import normalize_landmarks_labels
from cedalion.physunits import parse_quantity

# We want to provide a simpler yaml-based interface to all the different preprocessing
# methods. Therefore, we need adapaters which map yaml configuratons parameters
# to function arguments and calls.

# A registry of preprocessing adapters map method label to adapter(rec, ts, **params)
PREPROC_STEP_ADAPTERS: dict[str, Callable] = {}

_PRUNE_SDS = 1
_PRUNE_LOW_SIGNAL = 2
_PRUNE_POOR_SNR = 3
_PRUNE_SCI_PSP = 4
_PRUNE_SATURATED = 5


def preproc_step(name):
    """Decorator that adds an adapter to the registry."""

    def wrapper(fn):
        PREPROC_STEP_ADAPTERS[name] = fn
        return fn

    return wrapper


@dataclass
class Context:
    """Container for data structures that get passed into step adapters."""

    rec: cdc.Recording
    sidecar: xr.DataTree
    ts: cdt.NDTimeSeries
    step_name: str


def _prepare_sidecar_array(data: xr.DataArray) -> xr.DataArray:
    """Prepare a DataArray for storage in the preprocessing sidecar.

    Pint-backed arrays are dequantified so that they can be serialized to
    NetCDF while retaining their unit metadata.
    """
    if data.pint.units is not None:
        return data.pint.dequantify()

    return data


def _collapse_channel_mask(mask: xr.DataArray) -> xr.DataArray:
    """Collapse a clean-data mask to one boolean value per channel."""
    if "channel" not in mask.dims:
        raise ValueError(f"mask must contain a channel dimension, got {mask.dims}.")

    dims = [dim for dim in mask.dims if dim != "channel"]

    if dims:
        return mask.all(dim=dims)

    return mask


def _coerce_amplitude_threshold(
    value: float | Quantity,
    amplitude_units,
):
    """Express an amplitude threshold in the time series amplitude units."""
    if amplitude_units is None:
        if isinstance(value, Quantity):
            if not value.dimensionless:
                raise ValueError(
                    "A dimensional amplitude threshold cannot be used with "
                    "an unquantified amplitude time series."
                )
            return value.magnitude

        return value

    if isinstance(value, Quantity):
        return value.to(amplitude_units)

    return value * amplitude_units


# Define adapter functions which wrap existing cedalion functionality and adapt from
# the yaml-interface to the Python function signatures
@preproc_step("int2od")
def _int2od(ctx: Context):
    """TBD."""
    ctx.rec[ctx.step_name] = cedalion.nirs.cw.int2od(ctx.ts)


@preproc_step("od2conc")
def _od2conc(
    ctx: Context,
    dpf: dict[float, float] | tuple[float] = (1.0, 1.0),
    spectrum: str = "prahl",
):
    """TBD."""

    if isinstance(dpf, dict):
        dpf_values = list(dpf.values())
        wavelengths = list(dpf.keys())
    elif isinstance(dpf, (tuple, list)):
        # scan other time series in ctx.rec for wavelengths coordinates. First hit wins.
        for ts in ctx.rec.timeseries.values():
            if "wavelength" in ts.coords:
                dpf_values = list(dpf)
                wavelengths = ts.wavelength.values
                break
        else:
            raise ValueError(
                "DPFs were provided as a list but the recording container contains no "
                "timeseries from which wavelengths could be derived."
            )

    dpf = xr.DataArray(
        dpf_values, dims="wavelength", coords={"wavelength": wavelengths}
    )

    ctx.rec[ctx.step_name] = cedalion.nirs.cw.od2conc(
        ctx.ts, ctx.rec.geo3d, dpf, spectrum
    )


@preproc_step("tddr")
def _tddr(ctx: Context):
    """TBD."""
    ctx.rec[ctx.step_name] = cedalion.sigproc.motion.tddr(ctx.ts)


@preproc_step("freq_filter")
def _freq_filter(
    ctx: Context,
    *,
    fmin: cdt.QFrequency,
    fmax: cdt.QFrequency,
    butter_order: int = 4,
):
    """TBD."""
    ctx.rec[ctx.step_name] = cedalion.sigproc.frequency.freq_filter(
        ctx.ts, fmin=fmin, fmax=fmax, butter_order=butter_order
    )


@preproc_step("repair_amp")
def _repair_amp(
    ctx: Context, *, median_len: int = 3, interp_nan: bool = True, **kwargs
):
    """TBD."""
    ctx.rec[ctx.step_name] = cedalion.sigproc.quality.repair_amp(
        ctx.ts, median_len=median_len, interp_nan=interp_nan, **kwargs
    )


@preproc_step("wavelet")
def _wavelet(
    ctx: Context,
    *,
    iqr: float = 1.5,
    wavelet: str = "db2",
    level: int = 4,
):
    """TBD."""
    ctx.rec[ctx.step_name] = cedalion.sigproc.motion.wavelet(
        ctx.ts, iqr=iqr, wavelet=wavelet, level=level
    )


@preproc_step("sci")
def _sci(
    ctx: Context,
    window_length,
    sci_thresh: float,
    *,
    cardiac_fmin: cdt.QFrequency = 0.5 * units.Hz,
    cardiac_fmax: cdt.QFrequency = 2.5 * units.Hz,
):
    """TBD."""
    sci_values, sci_mask_values = cedalion.sigproc.quality.sci(
        ctx.ts, window_length, sci_thresh, cardiac_fmin=cardiac_fmin,
        cardiac_fmax=cardiac_fmax
    )
    ctx.sidecar[ctx.step_name] = sci_values
    ctx.sidecar[ctx.step_name + "_mask"] = sci_mask_values


@preproc_step("psp")
def _psp(
    ctx: Context,
    window_length,
    psp_thresh: float,
    *,
    cardiac_fmin: cdt.QFrequency = 0.5 * units.Hz,
    cardiac_fmax: cdt.QFrequency = 2.5 * units.Hz,
):
    """TBD."""
    psp_values, psp_mask_values = cedalion.sigproc.quality.psp(
        ctx.ts, window_length, psp_thresh, cardiac_fmin=cardiac_fmin,
        cardiac_fmax=cardiac_fmax
    )
    ctx.sidecar[ctx.step_name] = psp_values
    ctx.sidecar[ctx.step_name + "_mask"] = psp_mask_values



@preproc_step("prune")
def _prune(
    ctx: Context,
    *,
    snr_thresh: float,
    sd_thresh_min: Quantity,
    sd_thresh_max: Quantity,
    amp_thresh_min: float | Quantity,
    amp_thresh_max: float | Quantity,
    window_length: cdt.QTime,
    sci_thresh: float,
    psp_thresh: float,
    perc_time_clean_thresh: float,
    use_sci: bool = True,
    use_psp: bool = True,
):
    """Prune amplitude channels and retain diagnostic quality information.

    Initial pruning uses SNR, source-detector separation, and mean-amplitude
    criteria. SCI and PSP are evaluated per channel, and channels that failed
    the initial pruning are excluded from their clean-time masks.

    The pruned amplitude is added to the recording using ``ctx.step_name``.
    Individual metrics, masks, the final channel mask, and a categorical
    pruning reason are retained in the preprocessing sidecar.

    Args:
        ctx: Current preprocessing context.
        snr_thresh: Minimum signal-to-noise ratio.
        sd_thresh_min: Minimum source-detector separation.
        sd_thresh_max: Maximum source-detector separation.
        amp_thresh_min: Minimum acceptable mean amplitude.
        amp_thresh_max: Maximum acceptable mean amplitude.
        window_length: Window length used for SCI and PSP.
        sci_thresh: Minimum SCI value for a clean window.
        psp_thresh: Minimum PSP value for a clean window.
        perc_time_clean_thresh: Minimum fraction of clean windows required
            for a channel to remain usable.
        use_sci: Include SCI in the clean-time pruning criterion.
        use_psp: Include PSP in the clean-time pruning criterion.

    Raises:
        ValueError: If thresholds are inconsistent or outside valid ranges.
    """
    if not 0.0 <= perc_time_clean_thresh <= 1.0:
        raise ValueError("perc_time_clean_thresh must lie between 0 and 1.")

    if sd_thresh_min >= sd_thresh_max:
        raise ValueError("sd_thresh_min must be smaller than sd_thresh_max.")

    amplitude_units = ctx.ts.pint.units

    amp_thresh_min = _coerce_amplitude_threshold(
        amp_thresh_min,
        amplitude_units,
    )
    amp_thresh_max = _coerce_amplitude_threshold(
        amp_thresh_max,
        amplitude_units,
    )

    if amp_thresh_min >= amp_thresh_max:
        raise ValueError("amp_thresh_min must be smaller than amp_thresh_max.")

    # Initial quality metrics.
    snr_values, snr_mask = cedalion.sigproc.quality.snr(
        ctx.ts,
        snr_thresh=snr_thresh,
    )

    sd_dist, sd_mask = cedalion.sigproc.quality.sd_dist(
        ctx.ts,
        ctx.rec.geo3d,
        sd_range=(sd_thresh_min, sd_thresh_max),
    )

    mean_amp, amp_mask = cedalion.sigproc.quality.mean_amp(
        ctx.ts,
        amp_range=(amp_thresh_min, amp_thresh_max),
    )

    snr_channel_mask = _collapse_channel_mask(snr_mask)
    sd_channel_mask = _collapse_channel_mask(sd_mask)
    amp_channel_mask = _collapse_channel_mask(amp_mask)

    initial_channel_mask = snr_channel_mask & sd_channel_mask & amp_channel_mask

    # SCI and PSP are per-channel metrics. Compute them on the finite
    # amplitude data, then explicitly exclude channels that already failed
    # the initial pruning. This avoids feeding all-NaN channels into the
    # metric implementations.
    sci_values, sci_mask = cedalion.sigproc.quality.sci(
        ctx.ts,
        window_length,
        sci_thresh,
    )

    psp_values, psp_mask = cedalion.sigproc.quality.psp(
        ctx.ts,
        window_length,
        psp_thresh,
    )

    sci_mask = sci_mask & initial_channel_mask
    psp_mask = psp_mask & initial_channel_mask

    if use_sci and use_psp:
        coupling_mask = sci_mask & psp_mask
    elif use_sci:
        coupling_mask = sci_mask
    elif use_psp:
        coupling_mask = psp_mask
    else:
        coupling_mask = None

    if coupling_mask is None:
        time_clean_fraction = xr.ones_like(
            initial_channel_mask,
            dtype=float,
        )
        time_clean_mask = xr.ones_like(
            initial_channel_mask,
            dtype=bool,
        )
    else:
        time_clean_fraction = coupling_mask.mean(dim="time")

        time_clean_mask = time_clean_fraction > perc_time_clean_thresh

    final_channel_mask = initial_channel_mask & time_clean_mask

    if not final_channel_mask.any().item():
        raise ValueError("Pruning removed all channels with the configured thresholds.")

    # The sidecar retains masks/diagnostics for every original channel, while
    # the preprocessing time series contains only usable finite channels.
    amp_pruned, _ = cedalion.sigproc.quality.prune_ch(
        ctx.ts,
        masks=[final_channel_mask],
        operator="all",
        flag_drop=True,
    )

    # Diagnostic pruning reason.
    #
    # Precedence matches the legacy report:
    # poor SNR -> saturated -> low signal -> SDS -> SCI/PSP.
    #
    # SCI/PSP is only allowed to replace "good" for channels that passed
    # the initial stage and subsequently failed the clean-time criterion.
    low_signal = ~_collapse_channel_mask(mean_amp > amp_thresh_min)
    saturated = ~_collapse_channel_mask(mean_amp < amp_thresh_max)

    prune_reason = xr.zeros_like(
        final_channel_mask,
        dtype=np.int8,
    )

    prune_reason = xr.where(
        ~snr_channel_mask,
        _PRUNE_POOR_SNR,
        prune_reason,
    )

    prune_reason = xr.where(
        saturated,
        _PRUNE_SATURATED,
        prune_reason,
    )

    prune_reason = xr.where(
        low_signal,
        _PRUNE_LOW_SIGNAL,
        prune_reason,
    )

    prune_reason = xr.where(
        ~sd_channel_mask,
        _PRUNE_SDS,
        prune_reason,
    )

    coupling_failure = initial_channel_mask & ~time_clean_mask

    prune_reason = xr.where(
        coupling_failure,
        _PRUNE_SCI_PSP,
        prune_reason,
    )

    prune_reason.attrs["category_labels"] = (
        "0=good;"
        "1=source-detector distance;"
        "2=low signal;"
        "3=poor SNR;"
        "4=SCI/PSP;"
        "5=saturated"
    )

    ctx.rec[ctx.step_name] = amp_pruned

    prefix = ctx.step_name

    ctx.sidecar[prefix + "_snr"] = _prepare_sidecar_array(snr_values)
    ctx.sidecar[prefix + "_snr_mask"] = _prepare_sidecar_array(snr_mask)

    ctx.sidecar[prefix + "_sd_dist"] = _prepare_sidecar_array(sd_dist)
    ctx.sidecar[prefix + "_sd_mask"] = _prepare_sidecar_array(sd_mask)

    ctx.sidecar[prefix + "_mean_amp"] = _prepare_sidecar_array(mean_amp)
    ctx.sidecar[prefix + "_amp_mask"] = _prepare_sidecar_array(amp_mask)

    ctx.sidecar[prefix + "_sci"] = _prepare_sidecar_array(sci_values)
    ctx.sidecar[prefix + "_sci_mask"] = _prepare_sidecar_array(sci_mask)

    ctx.sidecar[prefix + "_psp"] = _prepare_sidecar_array(psp_values)
    ctx.sidecar[prefix + "_psp_mask"] = _prepare_sidecar_array(psp_mask)

    ctx.sidecar[prefix + "_time_clean_fraction"] = _prepare_sidecar_array(
        time_clean_fraction
    )
    ctx.sidecar[prefix + "_time_clean_mask"] = _prepare_sidecar_array(time_clean_mask)

    ctx.sidecar[prefix + "_initial_mask"] = _prepare_sidecar_array(initial_channel_mask)
    ctx.sidecar[prefix + "_mask"] = _prepare_sidecar_array(final_channel_mask)
    ctx.sidecar[prefix + "_reason"] = _prepare_sidecar_array(prune_reason)


@preproc_step("snr")
def _snr(
    ctx: Context,
    *,
    snr_thresh: float = 2.0,
):
    """Calculate SNR and store the metric and mask in the sidecar.

    This adapter does not modify the current preprocessing time series.

    Args:
        ctx: Current preprocessing context.
        snr_thresh: SNR threshold used to construct the quality mask.
    """
    snr_values, snr_mask = cedalion.sigproc.quality.snr(
        ctx.ts,
        snr_thresh=snr_thresh,
    )

    ctx.sidecar[ctx.step_name] = _prepare_sidecar_array(snr_values)
    ctx.sidecar[ctx.step_name + "_mask"] = _prepare_sidecar_array(snr_mask)


@preproc_step("gvtd")
def _gvtd(
    ctx: Context,
    *,
    stat_type: str = "histogram_mode",
    n_std: int = 10,
):
    """Calculate GVTD from amplitude and store its trace, mask, and threshold.

    This adapter expects amplitude data. For an optical-density checkpoint
    after motion correction, use the ``gvtd_from_od`` adapter.

    Args:
        ctx: Current preprocessing context.
        stat_type: Statistic used to determine the GVTD threshold.
        n_std: Number of standard deviations used for thresholding.
    """
    gvtd_values, gvtd_mask = cedalion.sigproc.quality.gvtd(
        ctx.ts,
        stat_type=stat_type,
        n_std=n_std,
    )

    # TODO: Replace this private helper call if Cedalion exposes the GVTD
    # threshold as a supported public API.
    gvtd_threshold = cedalion.sigproc.quality._get_gvtd_threshold(
        gvtd_values,
        stat_type=stat_type,
        n_std=n_std,
    )

    ctx.sidecar[ctx.step_name] = _prepare_sidecar_array(gvtd_values)
    ctx.sidecar[ctx.step_name + "_mask"] = _prepare_sidecar_array(gvtd_mask)
    ctx.sidecar[ctx.step_name + "_threshold"] = _prepare_sidecar_array(gvtd_threshold)


@preproc_step("gvtd_from_od")
def _gvtd_from_od(
    ctx: Context,
    *,
    stat_type: str = "histogram_mode",
    n_std: int = 10,
):
    """Calculate GVTD from an optical-density preprocessing checkpoint.

    The current optical-density time series is converted back to relative
    amplitude before calling the existing GVTD implementation. This allows
    GVTD to be evaluated after motion correction without applying int2od()
    directly to optical-density data.

    Args:
        ctx: Current preprocessing context.
        stat_type: Statistic used to determine the GVTD threshold.
        n_std: Number of standard deviations used for thresholding.
    """
    od = ctx.ts.pint.dequantify()

    relative_amp = np.exp(-od)
    relative_amp = relative_amp.pint.quantify("dimensionless")

    gvtd_values, gvtd_mask = cedalion.sigproc.quality.gvtd(
        relative_amp,
        stat_type=stat_type,
        n_std=n_std,
    )

    # TODO: Replace this private helper call if Cedalion exposes the GVTD
    # threshold as a supported public API.
    gvtd_threshold = cedalion.sigproc.quality._get_gvtd_threshold(
        gvtd_values,
        stat_type=stat_type,
        n_std=n_std,
    )

    ctx.sidecar[ctx.step_name] = _prepare_sidecar_array(gvtd_values)
    ctx.sidecar[ctx.step_name + "_mask"] = _prepare_sidecar_array(gvtd_mask)
    ctx.sidecar[ctx.step_name + "_threshold"] = _prepare_sidecar_array(gvtd_threshold)


@preproc_step("log_variance")
def _log_variance(
    ctx: Context,
):
    """Calculate log10 temporal variance of the current time series.

    The result retains all non-time dimensions, typically channel and
    wavelength. This adapter does not modify the preprocessing time series.

    Args:
        ctx: Current preprocessing context.

    Raises:
        ValueError: If the current time series has no time dimension.
    """
    if "time" not in ctx.ts.dims:
        raise ValueError(
            f"Log variance requires a time dimension, got dims {ctx.ts.dims}."
        )

    ts = _prepare_sidecar_array(ctx.ts)

    variance = ts.var(
        dim="time",
        skipna=False,
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        log_variance = np.log10(variance)

    log_variance = log_variance.where(np.isfinite(log_variance))

    ctx.sidecar[ctx.step_name] = log_variance


@preproc_step("spline")
def _spline(
    ctx: Context,
    *,
    p: float,
    t_motion: cdt.QTime = 0.5 * units.s,
    t_mask: cdt.QTime = 1.0 * units.s,
    stdev_thresh: float = 50.0,
    amp_thresh: float = 5.0,
):
    """TBD."""
    ma_mask = cedalion.sigproc.quality.id_motion(
            ctx.ts, t_motion=t_motion, t_mask=t_mask, stdev_thresh=stdev_thresh,
            amp_thresh=amp_thresh
        )
    ctx.rec[ctx.step_name] = cedalion.sigproc.motion.spline(
        ts=ctx.ts, t_inc_ch=ma_mask, p=p
    )


@preproc_step("pca")
def _pca(
    ctx: Context,
    *,
    n_sv: float = 0.97,
    t_motion: cdt.QTime = 0.5 * units.s,
    t_mask: cdt.QTime = 1.0 * units.s,
    stdev_thresh: float = 50.0,
    amp_thresh: float = 5.0,
):
    """Apply PCA motion correction."""

    # Detect motion separately for every channel/wavelength.
    ma_mask_ch = cedalion.sigproc.quality.id_motion(
        ctx.ts,
        t_motion=t_motion,
        t_mask=t_mask,
        stdev_thresh=stdev_thresh,
        amp_thresh=amp_thresh,
    )

    # Convert the channel-wise mask into one global time mask.
    # This is the same preparation used by pca_recurse().
    ma_mask = cedalion.sigproc.quality.id_motion_refine(
        ma_mask_ch,
        "all",
    )[0].copy()

    # Match the time alignment used by pca_recurse().
    ma_mask.values = np.hstack(
        [ma_mask.values[0], ma_mask.values[:-1]]
    )

    ts_cleaned, n_sv_used, svs = cedalion.sigproc.motion.pca(
        ctx.ts,
        ma_mask,
        n_sv=n_sv,
    )

    ctx.rec[ctx.step_name] = ts_cleaned
    ctx.sidecar[ctx.step_name + "_n_sv"] = n_sv_used

    sv_name = ctx.step_name + "_svs"
    sv_dim = ctx.step_name + "_component"
    sv_values = np.asarray(svs, dtype=float).reshape(-1)

    ctx.sidecar[sv_name] = xr.DataArray(
        sv_values,
        dims=(sv_dim,),
        coords={sv_dim: np.arange(sv_values.size)},
    )

@preproc_step("pca_recurse")
def _pca_recurse(
    ctx: Context,
    *,
    t_motion: cdt.QTime = 0.5 * units.s,
    t_mask: cdt.QTime = 1 * units.s,
    stdev_thresh: float = 20,
    amp_thresh: float = 5,
    n_sv: float = 0.97,
    max_iter: int = 5,
):
    """Apply recursive PCA motion correction."""

    ts_cleaned, svs, n_sv_ret, t_inc = (
        cedalion.sigproc.motion.pca_recurse(
            ctx.ts,
            t_motion=t_motion,
            t_mask=t_mask,
            stdev_thresh=stdev_thresh,
            amp_thresh=amp_thresh,
            n_sv=n_sv,
            max_iter=max_iter,
        )
    )

    ctx.rec[ctx.step_name] = ts_cleaned
    ctx.sidecar[ctx.step_name + "_n_sv"] = n_sv_ret

    sv_name = ctx.step_name + "_svs"
    sv_dim = ctx.step_name + "_component"

    # Converts scalar, empty, or normal arrays to a one-dimensional array.
    sv_values = np.asarray(svs, dtype=float).reshape(-1)

    ctx.sidecar[sv_name] = xr.DataArray(
        sv_values,
        dims=(sv_dim,),
        coords={sv_dim: np.arange(sv_values.size)},
    )

    if "units" in t_inc.time.attrs:
        t_inc = t_inc.copy()
        t_inc.time.attrs["units"] = str(t_inc.time.attrs["units"])

    ctx.sidecar[ctx.step_name + "_t_inc"] = t_inc


@preproc_step("spline_sg")
def _spline_sg(ctx: Context, *, p: float, frame_size: cdt.QTime = 10 * units.s):
    """TBD."""
    result = cedalion.sigproc.motion.spline_sg(ctx.ts, p=p, frame_size=frame_size)
    ctx.rec[ctx.step_name] = result


# FIXME move somewhere central
def quantify_params(params: dict):
    result = {}
    for k, v in params.items():
        if isinstance(v, str):
            try:
                v = parse_quantity(v)
            except ValueError:
                pass

        result[k] = v

    return result


def preprocess(
    input_snirf: Path | str,
    input_events: Path | str,
    input_optodes: Path | str,
    input_coordsystem: Path,
    output_snirf: Path,
    output_sidecar: Path,
    methods: list[dict],
    keep_intermediate: bool = False,
    normalize_landmarks: bool = False,
    # final_name : str # FIXME
) -> cdc.Recording:

    xrutils.unit_stripping_is_error(True)

    input_snirf = Path(input_snirf) if input_snirf else None
    input_events = Path(input_events) if input_events else None
    input_optodes = Path(input_optodes) if input_optodes else None
    input_coordsystem = Path(input_coordsystem) if input_coordsystem else None

    # FIXME: HARD CODED TIME UNITS
    records = cedalion.io.read_snirf(input_snirf, time_units="second")
    rec = records[0]

    if input_events and input_events.exists():
        # FIXME extend cedalion.io.bids.read_events_from_tsv to check schema
        stim_df = pd.read_csv(input_events, sep="\t")
        rec.stim = stim_df

    if input_optodes and input_optodes.exists():
        # FIXME check coordsystem
        # FIXME overwrite geo3d
        pass

    if normalize_landmarks:
        rec.geo3d = normalize_landmarks_labels(rec.geo3d)

    sidecar = xr.DataTree()
    sidecar.attrs["preprocess"] = json.dumps(
        {
            "steps": methods,
            "keep_intermediate": keep_intermediate,
            "normalize_landmarks": normalize_landmarks,
        },
        sort_keys=True,
        default=str,
    )

    for m in methods:
        m_name = m["name"]  # mandatory
        m_method = m["method"]  # mandatory
        m_params = quantify_params(m.get("params", {}))  # optional
        m_enabled = m.get("enable", True)  # optional

        if not m_enabled:
            continue

        last_ts = rec[next(reversed(rec.timeseries))]

        ctx = Context(rec=rec, sidecar=sidecar, ts=last_ts, step_name=m_name)

        # lookup adapter in registry
        adapter = PREPROC_STEP_ADAPTERS.get(m_method)

        if not adapter:
            raise ValueError(
                f"Unknown method '{m_method}' in step '{m_name}'. "
                "Known methods are: {sorted(PREPROC_STEP_ADAPTERS)}."
            )

        # call the adapter
        adapter(ctx, **m_params)


    if not keep_intermediate:
        for key in list(rec.timeseries.keys())[:-1]:
            del rec.timeseries[key]

    # write recording container and sidecar files
    cedalion.io.write_snirf(output_snirf, rec)
    sidecar.to_netcdf(output_sidecar)


def blockaverage(
    input_snirf: list[Path] | list[str],
    # input_preproc_sidecar : Path | str,
    output_snirf: Path,
    # output_sidecar: Path,
    # ts_name : str, # FIXME
    t_pre: cdt.QTime,
    t_post: cdt.QTime,
    trial_types: list[str] | None = None,
):
    input_snirf = [Path(i) for i in input_snirf]

    # FIXME move upstream
    t_pre = parse_quantity(t_pre)
    t_post = parse_quantity(t_post)

    print("-" * 80)
    for i in input_snirf:
        print(i)
    print("-" * 80)
    print(f"{t_pre=} {t_post=}")

    epochs = []

    for fname in input_snirf:
        rec = cedalion.io.read_snirf(fname)[0]

        # FIXME ideally users can select the time series by name
        ts = rec[next(reversed(rec.timeseries))]

        # FIXME: check trial_types, issue warnings on misconfigurations
        if trial_types is not None:
            selected_trials = sorted(
                set([i for i in rec.stim.trial_type if i in trial_types])
            )
        else:
            selected_trials = sorted(set([i for i in rec.stim.trial_type]))

        epochs.append(
            ts.cd.to_epochs(
                rec.stim,
                selected_trials,
                before=t_pre,
                after=t_post,
            )
        )

    epochs = xr.concat(epochs, dim="epoch")  # concatenate epochs from all runs

    baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")
    epochs = epochs - baseline  # baseline subtract
    blockaverage = epochs.groupby("trial_type").mean("epoch")  # mean across all epochs

    rec_out = cdc.Recording()

    # FIXME workarounds around meas_list construction in write_snirf
    rec_out["amp"] = rec["amp"]
    rec_out.stim = rec.stim

    rec_out["hrf_blockaverage"] = blockaverage
    # FIXME which metadata to carry over?

    cedalion.io.write_snirf(output_snirf, rec_out)

    # with open(output_snirf, "w") as fout:
    #    fout.write(" ")

    # with open(output_sidecar, "w") as fout:
    #    fout.write(" ")
