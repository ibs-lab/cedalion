from pathlib import Path
from typing import Callable

import pandas as pd
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.typing as cdt

import cedalion.io
import cedalion.nirs
import cedalion.sigproc.quality
import cedalion.sigproc.motion
import cedalion.xrutils as xrutils
from cedalion.physunits import parse_quantity
from cedalion import units
from dataclasses import dataclass

# We want to provide a simpler yaml-based interface to all the different preprocessing
# methods. Therefore, we need adapaters which map The preprocess snakemake rule should
# be configurable from a yaml file

# registry: method label -> adapter(rec, ts, **params)
PREPROC_STEP_ADAPTERS: dict[str, Callable] = {}


def preproc_step(name):
    """Adds an adapter to the registry."""

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
    steps: list[dict],
    keep_intermediate: bool = False,
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

    sidecar = xr.DataTree()

    for step in steps:
        step_name = step["name"]  # mandatory
        step_method = step["method"]  # mandatory
        step_params = quantify_params(step.get("params", {}))  # optional
        step_enabled = step.get("enable", True)  # optional

        if not step_enabled:
            continue

        last_ts = rec[next(reversed(rec.timeseries))]

        ctx = Context(rec=rec, sidecar=sidecar, ts=last_ts, step_name=step_name)

        # lookup adapter in registry
        adapter = PREPROC_STEP_ADAPTERS.get(step_method)

        if not adapter:
            raise ValueError(
                f"Unknown method '{step_method}' in step '{step_name}'. "
                "Known methods are: {sorted(PREPROC_STEP_ADAPTERS)}."
            )

        # call the adapter
        adapter(ctx, **step_params)

        # FIXME data quality report as additional step, configured through params

    if not keep_intermediate:
        for key in list(rec.timeseries.keys())[:-1]:
            del rec.timeseries[key]

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
