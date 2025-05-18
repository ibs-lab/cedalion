import pyxdf
import numpy as np
import xarray as xr

import cedalion
from cedalion import units


def xdf_to_aux_ts(snirf_path,
                 xdf_path,
                 mapping):
    """
    Read a SNIRF file and an XDF file, merge specified LSL streams into rec.aux_ts,
    then synchronize all data streams to the first rising edge of the digital trigger.

    Parameters
    ----------
    snirf_path : str
        Path to the .snirf file.
    xdf_path : str
        Path to the .xdf file containing LSL streams.
    mapping : dict[str, (str, int) | str]
        Mapping from desired aux_ts keys to either:
          - (LSL stream name, channel index) to include a single channel.
          - LSL stream name (str) to include all channels of that stream.

    Returns
    -------
    rec : cedalion.rec
        The SNIRF recording object with rec.aux_ts updated and all streams time-aligned
        so that the first rising edge of the trigger is at time zero.
    """
    # 1) Load SNIRF recording
    record = cedalion.io.read_snirf(snirf_path)
    rec = record[0]

    # 2) Load XDF streams
    streams, _ = pyxdf.load_xdf(xdf_path)

    # 3) Extract requested streams into rec.aux_ts
    for out_name, spec in mapping.items():
        # Determine stream name (and channel specification if provided)
        if isinstance(spec, str):
            stream_name = spec
            ch_spec = None
        else:
            try:
                stream_name, ch_spec = spec
            except Exception:
                raise ValueError(f"Invalid mapping for '{out_name}': {spec!r}")

        # Find specified LSL stream
        stream = next((s for s in streams if s['info']['name'][0] == stream_name), None)
        if stream is None:
            raise ValueError(f"Stream '{stream_name}' not found in {xdf_path!r}")

        # Pull raw time series and timestamps
        ts = np.asarray(stream.get('time_series', []))
        stamps = np.asarray(stream.get('time_stamps', []))

        # Determine channels to include
        if ch_spec is None:
            # Use all channels
            if ts.ndim == 1:
                ch_idxs = None
            elif ts.ndim == 2:
                ch_idxs = list(range(ts.shape[1]))
            else:
                raise ValueError(f"Unsupported time_series shape {ts.shape} for '{stream_name}'")
        # Use specified channel    
        elif isinstance(ch_spec, (int, np.integer)):
            ch_idxs = [int(ch_spec)]
        else:
            raise ValueError(f"Invalid channel spec for '{out_name}': {ch_spec!r}. Must be an int or stream name.")

        # Slice out requested channel(s)
        if ch_idxs is None:
            data = ts
        else:
            data = ts[:, ch_idxs] if ts.ndim == 2 else ts

        # Wrap in xarray.DataArray
        if data.ndim == 1 or (data.ndim == 2 and data.shape[1] == 1):
            da = xr.DataArray(
                data.flatten(),
                dims=('time',),
                coords={'time': stamps * units.s}
            )
        else:
            da = xr.DataArray(
                data,
                dims=('time', 'channel'),
                coords={
                    'time':    stamps * units.s,
                    'channel': ch_idxs
                }
            )
        da.coords['time'].attrs['units'] = 's'
        rec.aux_ts[out_name] = da

    # 4) Identify experiment start by first rising edge
    mask = rec.aux_ts['digital'] > 10
    i0 = mask.argmax(dim='time')
    t_start = rec.aux_ts['digital']['time'].isel(time=i0)

    # 5) Trim & zero-shift fNIRS amplitude ('amp') channel
    t0_amp = rec['amp'].sel(time=t_start.item()*units.s, method='nearest').time
    rec['amp'] = rec['amp'].sel(time=slice(t0_amp, None))
    new_time = rec['amp'].time - t0_amp
    new_time.attrs['units'] = 's'
    rec['amp'] = rec['amp'].assign_coords(time=new_time)

    # 6) Trim & zero-shift all auxiliary streams
    for name, da in rec.aux_ts.items():
        t0 = da.sel(time=t_start.item()*units.s, method='nearest').time
        da2 = da.sel(time=slice(t0, None))
        new_time = da2.time - t0
        new_time.attrs['units'] = 's'
        da2 = da2.assign_coords(time=new_time)
        rec.aux_ts[name] = da2

    return rec
