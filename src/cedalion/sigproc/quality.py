import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units

from .frequency import freq_filter, sampling_rate


@cdc.validate_schemas
def sci(amplitudes: cdt.NDTimeSeries, window_length: Quantity):
    """Calculate the scalp-coupling index.

    [1] L. Pollonini, C. Olds, H. Abaya, H. Bortfeld, M. S. Beauchamp, and
        J. S. Oghalai, “Auditory cortex activation to natural speech and
        simulated cochlear implant speech measured with functional near-infrared
        spectroscopy,” Hearing Research, vol. 309, pp. 84–93, Mar. 2014, doi:
        10.1016/j.heares.2013.11.007.
    """

    assert "wavelength" in amplitudes.dims  # FIXME move to validate schema

    # FIXME make these configurable
    cardiac_fmin = 0.5 * units.Hz
    cardiac_fmax = 2.5 * units.Hz

    amp = freq_filter(amplitudes, cardiac_fmin, cardiac_fmax, butter_order=4)
    amp = (amp - amp.mean("time")) / amp.std("time")

    # convert window_length to samples
    nsamples = (window_length * sampling_rate(amp)).to_base_units()
    nsamples = int(np.ceil(nsamples))

    # This creates a new DataArray with a new dimension "window", that is
    # window_len_samples large. The time dimension will contain the time coordinate of
    # the first sample in the window. Setting the stride size to the same value as the
    # window length will result in non-overlapping windows.
    windows = amp.rolling(time=nsamples).construct("window", stride=nsamples)

    sci = (windows - windows.mean("window")).prod("wavelength").sum("window") / nsamples
    sci /= windows.std("window").prod("wavelength")

    return sci
