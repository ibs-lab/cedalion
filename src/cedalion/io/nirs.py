from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io
import xarray as xr

import cedalion
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.utils as utils


def _read_geo3d(sd: np.ndarray):
    column_names = sd.dtype.names

    units = sd["SpatialUnit"].item()

    nsrc = int(sd["nSrcs"].item())
    src_pos = sd["SrcPos"][()]
    src_labels = utils.zero_padded_numbers(range(1, nsrc + 1), prefix="S")
    assert src_pos.shape == (nsrc, 3)

    ndet = int(sd["nDets"].item())
    det_pos = sd["DetPos"][()]
    det_labels = utils.zero_padded_numbers(range(1, ndet + 1), prefix="D")
    assert det_pos.shape == (ndet, 3)

    if "Landmarks" in column_names:
        lm_pos = sd["Landmarks"][()]
        nlm = lm_pos.shape[0]
        lm_labels = utils.zero_padded_numbers(range(1, nlm + 1), prefix="L")
    else:
        lm_pos = np.zeros((0, 3))
        nlm = 0
        lm_labels = []

    types = (
        [cdc.PointType.SOURCE] * nsrc
        + [cdc.PointType.DETECTOR] * ndet
        + [cdc.PointType.LANDMARK] * nlm
    )

    labels = src_labels + det_labels + lm_labels
    coordinates = np.vstack((src_pos, det_pos, lm_pos))

    geo3d = cdc.build_labeled_points(
        coordinates, crs="pos", units=units, labels=labels, types=types
    )

    return geo3d


def _read_timeseries(
    file: dict[str, np.ndarray], sd: str, geo3d: cdt.LabeledPointCloud
):
    t = file["t"].squeeze()
    d = file["d"]

    time_units = cedalion.units.s

    ml = file[sd]["MeasList"][()]
    n_flatchannel = len(ml)
    assert d.shape == (len(t), n_flatchannel)

    # list for unique (src, det) tuples. 1-indexed.
    unique_channels = [
        tuple(r)
        for _, r in pd.DataFrame(ml[:, :2].astype(int)).drop_duplicates().iterrows()
    ]

    src_labels = geo3d[geo3d.type == cdc.PointType.SOURCE].label.values
    det_labels = geo3d[geo3d.type == cdc.PointType.DETECTOR].label.values

    wavelengths = file[sd]["Lambda"][()]

    source = []
    detector = []
    channel = []

    data = np.zeros((len(t), len(unique_channels), len(wavelengths)))

    for i_fch in range(n_flatchannel):
        i_src = int(ml[i_fch, 0])
        i_det = int(ml[i_fch, 1])
        # ml[:,2] is unused

        i_wl = int(ml[i_fch, 3] - 1)
        i_ch = unique_channels.index((i_src, i_det))

        data[:, i_ch, i_wl] = d[:, i_fch]

    source = [src_labels[i[0] - 1] for i in unique_channels]
    detector = [det_labels[i[1] - 1] for i in unique_channels]
    channel = [src_labels[i[0] - 1] + det_labels[i[1] - 1] for i in unique_channels]

    ts = xr.DataArray(
        data,
        dims=["time", "channel", "wavelength"],
        coords={
            "time": ("time", t),
            "samples": ("time", np.arange(len(t))),
            "channel": ("channel", channel),
            "source": ("channel", source),
            "detector": ("channel", detector),
            "wavelength": ("wavelength", wavelengths),
        },
        attrs={"units": "dimensionless"},
    )
    ts = ts.pint.quantify()
    ts = ts.pint.quantify({"time": time_units})
    ts = ts.transpose("channel", "wavelength", "time")

    # build measurement list dataframe
    df_ml = pd.DataFrame(
        ml[:, [0, 1, 3]] - 1,
        columns=["sourceIndex", "detectorIndex", "wavelengthIndex"],
    )

    df_ml["wavelength"] = wavelengths[df_ml["wavelengthIndex"]]
    df_ml["channel"] = [
        f"{src_labels[r['sourceIndex']]}{det_labels[r['detectorIndex']]}"
        for _, r in df_ml.iterrows()
    ]
    df_ml["source"] = [
        src_labels[r['sourceIndex']]
        for _, r in df_ml.iterrows()
    ]
    df_ml["detector"] = [
        det_labels[r['detectorIndex']]
        for _, r in df_ml.iterrows()
    ]


    return ts, df_ml


def _read_stim(file: dict[str, np.ndarray]):
    s = file["s"][()]

    if s.ndim == 1:
        s = s[:,None]

    t = file["t"].squeeze()

    nstim = s.shape[1]

    if "CondNames" in file:
        trial_types = file["CondNames"][()]
        assert len(trial_types) == nstim
    else:
        trial_types = [str(i) for i in range(1, nstim + 1)]

    onset = []
    duration = []
    value = []
    trial_type = []

    for i_stim in range(nstim):
        current_onsets = t[np.where(s[:, i_stim])].tolist()
        onset.extend(current_onsets)
        duration.extend([0.0] * len(current_onsets))
        value.extend([1.0] * len(current_onsets))
        trial_type.extend([trial_types[i_stim]] * len(current_onsets))

    df_stim = pd.DataFrame(
        {"onset": onset, "duration": duration, "value": value, "trial_type": trial_type}
    )

    return df_stim

def _read_aux(file : dict[str, np.ndarray]):
    t = file["t"].squeeze()

    if "aux" not in file:
        return None

    dims = ["time"] + [f"dim{i}" for i in range(file["aux"][()].ndim-1)]

    return xr.DataArray(
        file["aux"][()],
        dims = dims,
        coords = {
            "time" : t
        }
    )


def read_nirs(fname: Path | str):
    if isinstance(fname, Path):
        fname = str(fname)

    file: dict[str, np.ndarray] = scipy.io.loadmat(fname, squeeze_me=True)

    if "SD3D" in file:
        geo3d = _read_geo3d(file["SD3D"])
        ts, df_ml = _read_timeseries(file, "SD3D", geo3d)
    else:
        geo3d = _read_geo3d(file["SD"])
        ts, df_ml = _read_timeseries(file, "SD", geo3d)

    df_stim = _read_stim(file)
    aux = _read_aux(file)


    rec = cdc.Recording()
    rec["amp"] = ts
    rec._measurement_lists["amp"] = df_ml
    rec.geo3d = geo3d
    rec.stim = df_stim

    if aux is not None:
        rec.aux_ts["aux"] = aux

    return rec
