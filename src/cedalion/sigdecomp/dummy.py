import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import units

from ..sigproc.frequency import freq_filter


@cdc.validate_schemas
def split_frequency_bands(ts: cdt.NDTimeSeries) -> cdt.NDTimeSeries:
    """Extract signal components in different frequency bands.

    This is not intended for real use but should act rather as an example how
    signal decomposition methods could be implemented.
    """

    # define frequency bands that contain different kinds of physiology
    cardiac_fmin = 0.5 * units.Hz
    cardiac_fmax = 2.5 * units.Hz

    respiratory_fmin = 0.1 * units.Hz
    respiratory_fmax = 0.5 * units.Hz

    # bandpass filter the time series
    cardiac = freq_filter(ts, cardiac_fmin, cardiac_fmax, butter_order=4)
    respiratory = freq_filter(ts, respiratory_fmin, respiratory_fmax, butter_order=4)

    # construct the resulting data array with one additional dimension
    new_dim = "band"
    result = xr.concat((cardiac, respiratory), dim=new_dim)
    result = result.assign_coords({new_dim: (new_dim, ["cardiac", "respiratory"])})

    return result
