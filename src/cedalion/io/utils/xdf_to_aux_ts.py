import numpy as np
import xarray as xr
import pyxdf
import cedalion.io
from cedalion import units


# ---------------------------------------------------------------------#
# helpers                                                              #
# ---------------------------------------------------------------------#



def _first_digital_edge(rec, threshold):
    """Hardware-based sync: timestamp & value of first digital edge."""
    digital_signal = rec.aux_ts["digital"]
    if not bool((digital_signal > threshold).any()):
        raise RuntimeError("Digital trigger never exceeded threshold.")
    i0 = (digital_signal > threshold).argmax(dim="time")
    t_start = digital_signal["time"].isel(time=i0)
    start_val = digital_signal.values[i0].item()
    print(f"[SNIRF] First digital edge at t = {float(t_start):.6f}s, "
          f"value = {start_val}")
    return t_start, start_val


def _first_marker(rec, marker_stream):
    """Marker-based sync: timestamp & value of first stim marker."""
    stim_da = rec.aux_ts[marker_stream]
    if stim_da.size == 0:
        raise RuntimeError(f"aux_ts[{marker_stream!r}] contains no samples.")
    mask = (stim_da != 0) & (~np.isnan(stim_da))
    if not bool(mask.any()):
        raise RuntimeError("Stim channel contains no non-zero markers.")
    i0 = mask.argmax(dim="time")
    t_start = stim_da["time"].isel(time=i0)
    start_val = stim_da.values[i0].item() 
    print(f"[SNIRF] First stim marker at t = {float(t_start):.6f}s, "
          f"value = {start_val}")
    return t_start, start_val


def _trim_and_zero_shift(da, t_start):
    """Drop data before t_start and shift time axis to 0 s."""
    t0 = da.sel(time=t_start.item() * units.s, method="nearest").time
    da = da.sel(time=slice(t0, None))
    new_time = (da.time - t0) / units.s 
    new_time.attrs["units"] = "s"
    return da.assign_coords(time=new_time)


def _get_stream(streams, name):
    """Return the first stream whose LSL name matches *name* (or None)."""
    return next((s for s in streams if s["info"]["name"][0] == name), None)


def _resolve_stream(streams, stream_name):
    """Return the stream whose LSL name matches *stream_name*.

    Raises
    ------
    RuntimeError
        If no such stream exists in the XDF file.
    """
    s = _get_stream(streams, stream_name)
    if s is None:
        raise RuntimeError(f"Stream {stream_name!r} not found in XDF.")
    return s

def _wrap_array(ts, t0, chan_labels=None, channel_meta=None):
    """
    Convert a NumPy array + time vector into an xarray.DataArray and
    attach channel-level metadata (labels, etc.) when available.
    """
    if ts.ndim == 1:                      
        ts = ts[:, None]
        chan_labels = chan_labels or [0]
        channel_meta = channel_meta or {0: {}}

    da = xr.DataArray(
        ts,
        dims=("time", "channel"),
        coords={
            "time":    t0,
            "channel": chan_labels if chan_labels else np.arange(ts.shape[1]),
        },
    )
    if channel_meta:
        da.attrs["channel_meta"] = channel_meta
    return da




# ---------------------------------------------------------------------#
# primary function                                                     #
# ---------------------------------------------------------------------#

def xdf_to_aux_ts(
    snirf_path: str,
    xdf_path: str,
    mapping: dict[str, str],          
    *,
    marker_stream="PsychoPyMarker",
    align="digital",   
    digital_threshold: float = 10.0,               
):
    """
    Import selected XDF streams/channels into a SNIRF recording and align
    everything to a shared time-zero.

    Parameters
    ----------
    snirf_path : str
        Path to the SNIRF file with the fNIRS recording.
    xdf_path : str
        Path to the XDF file captured via Lab Streaming Layer.
    mapping : dict[str, str]
        ``{new_aux_key: stream_name}``.
    marker_stream : str, default "PsychoPyMarker"
        Stream that carries stimulus codes.
    align : {'digital', 'marker'}, default 'digital'
        How to define the common time origin.
        * ``digital`` - first digital edge in ``aux_ts['digital']``.
        * ``marker`` - first non-zero sample in ``aux_ts[marker_stream]``
          (must match the first marker in the XDF stream).
    digital_threshold : float, default 10.0
        Minimum value that defines a digital edge in the ``aux_ts['digital']``
        trigger line when ``align='digital'``.

    Returns
    -------
    Recording
        Cedalion recording with all channels time-aligned and the requested
        XDF data added to ``aux_ts``.
    """

   
    if align not in ("digital", "marker"):
        raise ValueError("align must be 'digital' or 'marker'")
    
    if not mapping:
        raise ValueError("mapping cannot be empty - no XDF data to import.")
       

    # --- 1) load SNIRF & choose alignment reference -------------------
    rec = cedalion.io.read_snirf(snirf_path)[0]

    if align == "digital":
        if "digital" not in rec.aux_ts:
            raise RuntimeError("No aux_ts['digital'] channel in SNIRF.")
        t_start, start_val = _first_digital_edge(rec, digital_threshold)
    else:
        if marker_stream not in rec.aux_ts:
            raise RuntimeError(
                f"No aux_ts[{marker_stream!r}] stim channel in SNIRF.")
        t_start, start_val = _first_marker(rec, marker_stream)

    # --- 1b) trim & zero-shift SNIRF channels -------------------------
    rec["amp"] = _trim_and_zero_shift(rec["amp"], t_start)
    for name, da in rec.aux_ts.items():
        rec.aux_ts[name] = _trim_and_zero_shift(da, t_start)

    # --- 2) load XDF, find first marker ----------------------
    streams, _ = pyxdf.load_xdf(xdf_path)
    xdf_marker_stream = _get_stream(streams, marker_stream)
    if xdf_marker_stream is None:
        raise RuntimeError(f"{marker_stream!r} stream not found in XDF.")

    marker_ts = np.asarray(xdf_marker_stream["time_series"]).squeeze()
    first_idx = np.flatnonzero(marker_ts.astype(bool))[0]
    lsl_t0 = float(xdf_marker_stream["time_stamps"][first_idx])
    marker_val = marker_ts[first_idx]

    print(f"[XDF]  First PsychoPy marker at t = {lsl_t0:.6f}s "
          f"(value = {marker_val})")

    if align == "marker" and marker_val != start_val:
        raise RuntimeError(
            "First marker mismatch between SNIRF "
            f"({start_val}) and XDF ({marker_val}).")


    # --- 3) import requested streams / channels -----------------------
    for aux_name, search in mapping.items():

        src = _resolve_stream(streams, search)

        desc       = src["info"].get("desc", [])
        chan_root  = desc[0].get("channels", []) if desc else []
        chan_list  = chan_root[0].get("channel", []) if chan_root else []

        ts   = np.asarray(src["time_series"])
        t_ls = np.asarray(src["time_stamps"])

        keep = t_ls >= lsl_t0
        ts   = ts[keep]
        t0   = t_ls[keep] - lsl_t0
   
    
        channel_meta = {i: m for i, m in enumerate(chan_list)}        
        chan_labels  = [meta.get("label", [""])[0] for meta in chan_list]
                                   


        # ── wrap in xarray ────────────────────────────────────────────
        da = _wrap_array(ts, t0, chan_labels, channel_meta)

        da.time.attrs["units"] = "s"
        rec.aux_ts[aux_name] = da

        print(f"    → aux_ts['{aux_name}']: "
              f"{len(da.time)} samples, starts at {float(da.time[0]):.6f}s")

    return rec
