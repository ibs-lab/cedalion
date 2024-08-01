import cedalion.dataclasses as cdc
from cedalion.tasks import task
import cedalion.nirs  # FIXME cedalion.sigproc.nirs?
import cedalion.sigproc.quality
from typing import Annotated
import xarray as xr
from cedalion import Quantity


@task
def int2od(
    rec: cdc.Recording,
    ts_input: str | None = None,
    ts_output: str = "od",
):
    """Calculate optical density from intensity amplitude  data.

    Args:
        rec (Recording): container of timeseries data
        ts_input (str): name of intensity timeseries. If None, this tasks operates on
            the last timeseries in rec.timeseries.
        ts_output (str): name of optical density timeseries.
    """

    ts = rec.get_timeseries(ts_input)
    od = cedalion.nirs.int2od(ts)
    rec.set_timeseries(ts_output, od)


@task
def od2conc(
    rec: cdc.Recording,
    dpf: dict[float, float],
    spectrum: str = "prahl",
    ts_input: str | None = None,
    ts_output: str = "conc",
):
    """Calculate hemoglobin concentrations from optical density data.

    Args:
        rec: container of timeseries data
        dpf: differential path length factors
        spectrum: label of the extinction coefficients to use
        ts_input: name of intensity timeseries. If None, this tasks operates on
            the last timeseries in rec.timeseries.
        ts_output: name of optical density timeseries.
    """

    ts = rec.get_timeseries(ts_input)

    dpf = xr.DataArray(
        list(dpf.values()),
        dims="wavelength",
        coords={"wavelength": list(dpf.keys())},
    )

    conc = cedalion.nirs.od2conc(ts, rec.geo3d, dpf, spectrum)

    rec.set_timeseries(ts_output, conc)


@task
def snr(
    rec: cdc.Recording,
    snr_thresh: float = 2.0,
    ts_input: str | None = None,
    aux_obj_output: str = "snr",
    mask_output: str = "snr",
):
    ts = rec.get_timeseries(ts_input)

    snr, snr_mask = cedalion.sigproc.quality.snr(ts, snr_thresh)

    rec.aux_obj[aux_obj_output] = snr
    rec.set_mask(mask_output, snr_mask)


@task
def sd_dist(
    rec: cdc.Recording,
    sd_min: Annotated[Quantity, "[length]"],
    sd_max: Annotated[Quantity, "[length]"],
    ts_input: str | None = None,
    aux_obj_output: str = "sd_dist",
    mask_output: str = "sd_dist",
):
    ts = rec.get_timeseries(ts_input)

    sd_dist, mask = cedalion.sigproc.quality.sd_dist(ts, rec.geo3d, (sd_min, sd_max))

    rec.set_mask(mask_output, mask)
    rec.aux_obj[aux_obj_output] = sd_dist