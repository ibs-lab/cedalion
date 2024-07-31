import numpy as np
import xarray as xr
import pandas as pd
from snirf import Snirf
from snirf.pysnirf2 import MeasurementList, DataElement, NirsElement, Stim
from enum import Enum
import logging
from numpy.typing import ArrayLike

from typing import Any
from strenum import StrEnum

from cedalion.dataclasses import PointType

log = logging.getLogger("cedalion")


class DataType(Enum):
    # 001-100: Raw - Continuous Wave (CW)
    CW_AMPLITUDE = 1
    CW_FLUORESCENCE_AMPLITUDE = 51

    # 101-200: Raw - Frequency Domain (FD)
    FD_AC_AMPLITUDE = 101
    FD_PHASE = 102
    FD_FLUORESCENCE_AMPLITUDE = 151
    FD_FLUORESCENCE_PHASE = 152

    # 201-300: Raw - Time Domain - Gated (TD Gated)
    TDG_AMPLITUDE = 201
    TDG_FLUORESCENCE_AMPLITUDE = 251

    # 301-400: Raw - Time domain – Moments (TD Moments)
    TDM_AMPLITUDE = 301
    TDM_FLUORESCENCE_AMPLITUDE = 351

    # 401-500: Raw - Diffuse Correlation Spectroscopy (DCS)
    DCS_G2 = 401
    DCS_BFI = 410

    PROCESSED = 99999


class DataTypeLabel(StrEnum):
    # fields specified in SNIRF specs
    DOD = "dOD"  # Change in optical density
    DMEAN = "dMean"  # Change in mean time-of-flight
    DVAR = "dVar"  # Change in variance (2nd central moment)
    DSKEW = "dSkew"  # Change in skewness (3rd central moment)
    MUA = "mua"  # Absorption coefficient
    MUSP = "musp"  # Scattering coefficient
    HBO = "HbO"  # Oxygenated hemoglobin (oxyhemoglobin) concentration
    HBR = "HbR"  # Deoxygenated hemoglobin (deoxyhemoglobin) concentration
    HBT = "HbT"  # Total hemoglobin concentration
    H2O = "H2O"  # Water content
    LIPID = "Lipid"  # Lipid concentration
    BFI = "BFi"  # Blood flow index
    HRF_DOD = "HRF dOD"  # Hemodynamic response function for change in optical density
    HRF_DMEAN = "HRF dMean"  # HRF for change in mean time-of-flight
    HRF_DVAR = "HRF dVar"  # HRF for change in variance (2nd central moment)
    HRF_DSKEW = "HRF dSkew"  # HRF for change in skewness (3rd central moment)
    HRF_HBO = "HRF HbO"  # Hemodynamic response function for oxyhemoglobin conc.
    HRF_HBR = "HRF HbR"  # Hemodynamic response function for deoxyhemoglobin conc.
    HRF_HBT = "HRF HbT"  # Hemodynamic response function for total hemoglobin conc.
    HRF_BFI = "HRF BFi"  # Hemodynamic response function for blood flow index

    # fields introduced by other vendors
    RAW_SATORI = "RAW"  # Satori CW_AMPLITUDE
    RAW_NIRX = "raw-DC"  # Satori CW_AMPLITUDE


# The snirf standard allows to put different data types into the same
# data element. At least Satori does this to store processing results.
# Define groups of data types which we would like to bundle in DataArrays.

DATA_TYPE_GROUPINGS = {
    (DataType.CW_AMPLITUDE, None): "unprocessed raw",
    (DataType.CW_AMPLITUDE, DataTypeLabel.RAW_NIRX): "unprocessed raw",
    # FIXME assume that there are not processed raw channels from different
    # vendors in the same data element
    (DataType.PROCESSED, DataTypeLabel.RAW_SATORI): "processed raw",
    (DataType.PROCESSED, DataTypeLabel.RAW_NIRX): "processed raw",
    (DataType.PROCESSED, DataTypeLabel.DOD): "processed dOD",
    (DataType.PROCESSED, DataTypeLabel.HBO): "processed concentrations",
    (DataType.PROCESSED, DataTypeLabel.HBR): "processed concentrations",
    (DataType.PROCESSED, DataTypeLabel.HBT): "processed concentrations",
    (DataType.PROCESSED, DataTypeLabel.H2O): "processed concentrations",
    (DataType.PROCESSED, DataTypeLabel.LIPID): "processed concentrations",
    (DataType.PROCESSED, DataTypeLabel.DMEAN): "processed central moments",
    (DataType.PROCESSED, DataTypeLabel.DVAR): "processed central moments",
    (DataType.PROCESSED, DataTypeLabel.DSKEW): "processed central moments",
    (DataType.PROCESSED, DataTypeLabel.BFI): "processed blood flow index",
    (DataType.PROCESSED, DataTypeLabel.HRF_DOD): "processed HRF dOD",
    (DataType.PROCESSED, DataTypeLabel.HRF_DMEAN): "processed HRF central moments",
    (DataType.PROCESSED, DataTypeLabel.HRF_DVAR): "processed HRF central moments",
    (DataType.PROCESSED, DataTypeLabel.HRF_DSKEW): "processed HRF central moments",
    (DataType.PROCESSED, DataTypeLabel.HRF_HBO): "processed HRF concentrations",
    (DataType.PROCESSED, DataTypeLabel.HRF_HBR): "processed HRF concentrations",
    (DataType.PROCESSED, DataTypeLabel.HRF_HBT): "processed HRF concentrations",
    (DataType.PROCESSED, DataTypeLabel.HRF_BFI): "processed HRF blood flow index",
    (DataType.PROCESSED, DataTypeLabel.MUA): "processed absorption coefficient",
    (DataType.PROCESSED, DataTypeLabel.MUSP): "processed scattering coefficient",
}


def parse_data_type(value):
    if value is None:
        return None
    else:
        try:
            return DataType(value)
        except Exception:
            log.warning(f"unsupported DataType '{value}'")
            return None


def parse_data_type_label(value):
    if value is None:
        return None
    else:
        try:
            return DataTypeLabel(value)
        except Exception:
            log.warning(f"unsupported DataTypeLabel '{value}'")
            return None


# result container for read_snirf. Corresponds to pysnirf's NirsElement but
# the data in the attributes have been converted to our data structures
# FIXME: find a better name
class Element:
    # data: list

    def __init__(
        self,
        data=None,
        stim=None,
        geo3d=None,
        geo2d=None,
        aux=None,
        meta_data=None,
        measurement_lists=None,
    ):
        self.data = []
        self.stim = pd.DataFrame(columns=["onset", "duration", "value"])
        self.geo3d = xr.DataArray(None, coords={"label": []}, dims=["label", "pos"])
        self.geo2d = xr.DataArray(None, coords={"label": []}, dims=["label", "pos"])
        self.aux = {}
        self.meta_data = {}
        self.measurement_lists = []

        if data is not None:
            self.data = data
        if stim is not None:
            self.stim = stim
        if geo3d is not None:
            self.geo3d = geo3d
        if geo2d is not None:
            self.geo2d = geo2d
        if aux is not None:
            self.aux = aux
        if meta_data is not None:
            self.meta_data = meta_data
        if measurement_lists is not None:
            self.measurement_lists = measurement_lists


def reduce_ndim_sourceLabels(sourceLabels: np.ndarray) -> list:
    """Deal with multidimensional source labels.

    snirf supports multidimensional source labels but we don't.
    This function tries to reduce n-dimensional source labels
    to a unique common prefix to obtain only one label per source
    """
    labels = []
    for i_src in range(sourceLabels.shape[0]):
        common_prefix = []

        for characters in zip(*sourceLabels[i_src, :]):
            unique = np.unique(characters)
            if len(unique) == 1:
                common_prefix.append(unique[0])
            else:
                break
        common_prefix = "".join(common_prefix)
        log.debug(common_prefix, sourceLabels[i_src, :])
        labels.append(common_prefix)

    labels = np.asarray(labels)
    unique = np.unique(labels)
    if len(unique) < len(labels):
        raise ValueError(
            "couldn't reduce n-dimensional source labels to unique common prefixes"
        )

    return labels


def labels_and_positions(probe, dim: int = 3):
    def convert_none(probe, attrname, default):
        attr = getattr(probe, attrname)
        if attr is None:
            return default
        else:
            return attr

    if dim not in [2, 3]:
        raise AttributeError(f"dim must be '2' or '3' but got {dim}")
    else:
        dim = int(dim)

    sourceLabels = convert_none(probe, "sourceLabels", np.asarray([], dtype=str))
    detectorLabels = convert_none(probe, "detectorLabels", np.asarray([], dtype=str))
    landmarkLabels = convert_none(probe, "landmarkLabels", np.asarray([], dtype=str))

    if sourceLabels.ndim > 1:
        sourceLabels = reduce_ndim_sourceLabels(sourceLabels)

    sourcePos = convert_none(probe, f"sourcePos{dim}D", np.zeros((0, dim)))
    detectorPos = convert_none(probe, f"detectorPos{dim}D", np.zeros((0, dim)))
    landmarkPos = convert_none(probe, f"landmarkPos{dim}D", np.zeros((0, dim + 1)))[
        :, :dim
    ]  # FIXME we keep only the positional columns

    if len(sourcePos) > 0 and len(sourceLabels) == 0:
        log.warning("generating generic source labels")
        sourceLabels = np.asarray([f"S{i+1}" for i in range(len(sourcePos))])

    if len(detectorPos) > 0 and len(detectorLabels) == 0:
        log.warning("generating generic detector labels")
        detectorLabels = np.asarray([f"D{i+1}" for i in range(len(detectorPos))])

    if len(landmarkPos) != len(landmarkLabels):
        if len(landmarkPos) > 0:
            raise ValueError("landmark positions were provided but no labels")
        else:
            log.warning("landmark labels were provided but not their positions. Removing labels")
            landmarkLabels = np.asarray([], dtype=str)

    return (
        sourceLabels,
        detectorLabels,
        landmarkLabels,
        sourcePos,
        detectorPos,
        landmarkPos,
    )


def geometry_from_probe(nirs_element: NirsElement, dim: int = 3):
    probe = nirs_element.probe

    length_unit = nirs_element.metaDataTags.LengthUnit

    (
        sourceLabels,
        detectorLabels,
        landmarkLabels,
        sourcePos,
        detectorPos,
        landmarkPos,
    ) = labels_and_positions(probe, dim)

    types = (
        [PointType.SOURCE] * len(sourceLabels)
        + [PointType.DETECTOR] * len(detectorLabels)
        + [PointType.LANDMARK] * len(landmarkLabels)
    )

    labels = np.hstack([sourceLabels, detectorLabels, landmarkLabels])
    positions = np.vstack([sourcePos, detectorPos, landmarkPos])

    coords = {"label": ("label", labels), "type": ("label", types)}
    dims = ["label", "pos"]
    attrs = {"units": length_unit}

    if len(positions) == 0:
        result = xr.DataArray(None, coords=coords, dims=dims, attrs=attrs)
    elif len(labels) == len(positions):
        result = xr.DataArray(positions, coords=coords, dims=dims, attrs=attrs)
    else:
        raise ValueError("number of positions and labels does not match")

    result = result.set_xindex("type")
    result = result.pint.quantify()

    return result


def measurement_list_to_dataframe(
    measurement_list: MeasurementList, drop_none: bool = False
):
    fields = [
        "sourceIndex",
        "detectorIndex",
        "wavelengthIndex",
        "wavelengthActual",
        "wavelengthEmissionActual",
        "dataType",
        "dataUnit",
        "dataTypeLabel",
        "dataTypeIndex",
        "sourcePower",
        "detectorGain",
        "moduleIndex",
        "sourceModuleIndex",
        "detectorModuleIndex",
    ]

    df = pd.DataFrame([{f: getattr(ml, f) for f in fields} for ml in measurement_list])

    if drop_none:
        df = df[[i for i in df.columns if not df[i].isna().all()]]

    return df


def meta_data_tags_to_dict(nirs_element: NirsElement):
    mdt = nirs_element.metaDataTags

    fields = mdt._snirf_names + mdt._unspecified_names

    return {f: getattr(mdt, f) for f in fields}


def stim_to_dataframe(stim: Stim):
    dfs = []

    if len(stim) == 0:
        columns = ["onset", "duration", "value", "trial_type"]
        return pd.DataFrame(columns=columns)

    for st in stim:
        columns = ["onset", "duration", "value"]

        if st.data is None:
            tmp = pd.DataFrame(columns=columns)
        else:
            ncols = st.data.shape[1]

            if ncols > 3:
                if (st.dataLabels is not None) and (len(st.dataLabels) == ncols):
                    for i in range(3, ncols):
                        columns.append(st.dataLabels[i])
                else:
                    for i in range(3, ncols):
                        columns.append(f"col{i}")

            tmp = pd.DataFrame(st.data, columns=columns)
            tmp["trial_type"] = st.name
        dfs.append(tmp)

    return pd.concat(dfs, ignore_index=True)


def read_aux(nirs_element: NirsElement, opts: dict[str, Any]):
    result = {}

    for aux in nirs_element.aux:
        name = aux.name
        units = aux.dataUnit
        time_offset = aux.timeOffset

        # FIXME treat unspecified units as dimensionless quantities.
        if units is None:
            units = "1"

        ntimes = len(aux.time)

        aux_data = aux.dataTimeSeries

        if opts["squeeze_aux"]:
            aux_data = np.squeeze(aux_data)

        if aux_data.ndim == 1:
            dims = ["time"]
        elif aux_data.ndim == 2:
            dims = ["time", "aux_channel"]
            if aux_data.shape[1] == ntimes:
                aux_data = aux_data.transpose()
        else:
            raise ValueError("aux.dataTimeSeries must have either 1 or 2 dimensions.")

        x = xr.DataArray(
            aux_data,
            coords={"time": aux.time},
            dims=dims,
            name=name,
            attrs={"units": units, "time_offset": time_offset},
        )

        result[name] = x.pint.quantify()

    return result


def read_data_elements(
    data_element: DataElement, nirs_element: NirsElement, stim: pd.DataFrame
):
    time = data_element.time

    trial_types = stim["trial_type"].drop_duplicates().values

    samples = np.arange(len(time))

    if len(time) != data_element.dataTimeSeries.shape[0]:
        raise ValueError("length of time and dataTimeSeries arrays don't match!")

    df_ml = measurement_list_to_dataframe(data_element.measurementList)
    df_ml = denormalize_measurement_list(df_ml, nirs_element)

    # unique_data_types = df_ml[["dataType", "dataTypeLabel"]].drop_duplicates()
    data_types = df_ml[["dataType", "dataTypeLabel"]]
    data_types = data_types.transform(
        {"dataType": parse_data_type, "dataTypeLabel": parse_data_type_label}
    )
    df_ml["data_type_group"] = [
        DATA_TYPE_GROUPINGS[tuple(r)] for r in data_types.to_numpy()
    ]

    if len(df_ml["data_type_group"].drop_duplicates()) > 1:
        log.warning("found data element with multiple data types. These will be split.")

    data_arrays = []

    for data_type_group, df in df_ml.groupby("data_type_group"):
        has_wavelengths = not pd.isna(df.wavelength).all()
        has_chromo = not pd.isna(df.chromo).all()

        is_hrf = (not pd.isna(df.dataTypeIndex).all()) and ("HRF" in data_type_group)

        is_hrf = (not pd.isna(df.dataTypeIndex).all()) and ("HRF" in data_type_group)

        if has_wavelengths and not has_chromo:
            other_dim = "wavelength"
        elif has_chromo and not has_wavelengths:
            other_dim = "chromo"
        elif not has_chromo and not has_wavelengths:
            raise ValueError(
                "found channel for which neither wavelength nor "
                "chromophore is defined."
            )
        else:
            raise NotImplementedError(
                "found channel for which both wavelength "
                "and chromophore are defined."
            )

        df_coords = {}
        df_coords["channel"] = df[["channel", "source", "detector"]].drop_duplicates()
        df_coords[other_dim] = df[[other_dim]].drop_duplicates()

        unique_channel = list(df_coords["channel"]["channel"])
        unique_other_dim = list(df[other_dim].drop_duplicates())

        df["index_channel"] = [unique_channel.index(c) for c in df.channel]
        df["index_other_dim"] = [unique_other_dim.index(c) for c in df[other_dim]]

        coords = {}
        coords["time"] = ("time", time)
        coords["samples"] = ("time", samples)
        coords["channel"] = ("channel", df_coords["channel"]["channel"])
        coords["source"] = ("channel", df_coords["channel"]["source"])
        coords["detector"] = ("channel", df_coords["channel"]["detector"])
        coords[other_dim] = (other_dim, unique_other_dim)

        ts2d = data_element.dataTimeSeries

        units = df.dataUnit.unique().item()
        # FIXME treat unspecified units as dimensionless quantities.
        if units is None:
            units = "1"

        if is_hrf:
            channel_trial_types = trial_types[df["dataTypeIndex"].values - 1]
            used_trial_types = np.unique(channel_trial_types).tolist()
            df["index_trial_type"] = [
                used_trial_types.index(i) for i in channel_trial_types
            ]

            ts4d = np.zeros(
                (
                    len(unique_channel),
                    len(unique_other_dim),
                    len(used_trial_types),
                    len(time),
                ),
                dtype=data_element.dataTimeSeries.dtype,
            )

            for index, row in df.iterrows():
                ts4d[
                    row.index_channel, row.index_other_dim, row.index_trial_type, :
                ] = ts2d[:, index]

            coords["trial_type"] = used_trial_types

            da = xr.DataArray(
                ts4d,
                dims=["channel", other_dim, "trial_type", "time"],
                coords=coords,
                attrs={
                    "units": units,
                    "data_type_group": data_type_group,
                },
            )

        else:
            ts3d = np.zeros(
                (len(unique_channel), len(unique_other_dim), len(time)),
                dtype=data_element.dataTimeSeries.dtype,
            )

            for index, row in df.iterrows():
                ts3d[row.index_channel, row.index_other_dim, :] = ts2d[:, index]

            da = xr.DataArray(
                ts3d,
                dims=["channel", other_dim, "time"],
                coords=coords,
                attrs={
                    "units": units,
                    "data_type_group": data_type_group,
                },
            )
        da = da.pint.quantify()

        time_units = nirs_element.metaDataTags.TimeUnit
        try:
            da = da.pint.quantify({"time": time_units})
        except ValueError:
            pass

        data_arrays.append(da)

    return data_arrays


def _get_time_coords(
    nirs_element: NirsElement,
    data_element: DataElement,
    df_measurement_list: pd.DataFrame,
) -> dict[str, ArrayLike]:
    time = data_element.time
    time_unit = nirs_element.metaDataTags.TimeUnit

    if len(time) != data_element.dataTimeSeries.shape[0]:
        raise ValueError("length of time and dataTimeSeries arrays don't match!")

    coordinates = {
        "time": ("time", xr.DataArray(time, attrs={"units": time_unit})),
        "samples": ("time", np.arange(len(time))),
    }
    indices = None
    return indices, coordinates


def _get_channel_coords(
    nirs_element: NirsElement,
    df_measurement_list: pd.DataFrame,
) -> tuple[ArrayLike, dict[str, ArrayLike]]:
    sourceLabels, detectorLabels, landmarkLabels, _, _, _ = labels_and_positions(
        nirs_element.probe
    )

    df_measurement_list = denormalize_measurement_list(
        df_measurement_list, nirs_element
    )

    coordinates = {
        "channel": ("channel", df_measurement_list["channel"]),
        "source": ("channel", df_measurement_list["source"]),
        "detector": ("channel", df_measurement_list["detector"]),
    }
    indices = None
    return indices, coordinates


def read_nirs_element(nirs_element, opts):
    geo2d = geometry_from_probe(nirs_element, dim=2)
    geo3d = geometry_from_probe(nirs_element, dim=3)
    stim = stim_to_dataframe(nirs_element.stim)
    data = []
    for data_element in nirs_element.data:
        data.extend(read_data_elements(data_element, nirs_element, stim))

    aux = read_aux(nirs_element, opts)

    meta_data = meta_data_tags_to_dict(nirs_element)

    measurement_lists = []
    for data_element in nirs_element.data:
        df_ml = measurement_list_to_dataframe(
            data_element.measurementList, drop_none=False
        )
        df_ml = denormalize_measurement_list(df_ml, nirs_element)
        df_ml.dropna(axis=1)
        measurement_lists.append(df_ml)

    ne = Element(
        data, stim, geo3d, geo2d, aux, meta_data, measurement_lists
    )

    return ne


def read_snirf(fname, squeeze_aux=False):
    opts = {"squeeze_aux": squeeze_aux}

    with Snirf(fname, "r") as s:
        return [read_nirs_element(ne, opts) for ne in s.nirs]


def denormalize_measurement_list(df_ml: pd.DataFrame, nirs_element: NirsElement):
    sourceLabels, detectorLabels, landmarkLabels, _, _, _ = labels_and_positions(
        nirs_element.probe
    )
    wavelengths = nirs_element.probe.wavelengths

    if (df_ml.sourceIndex < 1).any() or (df_ml.detectorIndex < 1).any():
        raise ValueError(
            "measurement list has row(s) for which source or detector index < 1."
        )

    chromo_types = [
        DataTypeLabel.HBO,
        DataTypeLabel.HBR,
        DataTypeLabel.HBT,
        DataTypeLabel.LIPID,
    ]

    hrf_chromo_types = {
        DataTypeLabel.HRF_HBO: DataTypeLabel.HBO,
        DataTypeLabel.HRF_HBR: DataTypeLabel.HBR,
        DataTypeLabel.HRF_HBT: DataTypeLabel.HBT,
    }

    new_columns = []
    for _, row in df_ml.iterrows():
        sl = sourceLabels[int(row["sourceIndex"]) - 1]
        dl = detectorLabels[int(row["detectorIndex"]) - 1]
        cl = f"{sl}{dl}"
        if not ((row["wavelengthIndex"] == -1) or (row["wavelengthIndex"] is None)):
            wl = wavelengths[int(row["wavelengthIndex"]) - 1]
        else:
            wl = np.nan

        if row["dataTypeLabel"] in chromo_types:
            ch = DataTypeLabel(row["dataTypeLabel"])
        elif row["dataTypeLabel"] in hrf_chromo_types:
            ch = hrf_chromo_types[row["dataTypeLabel"]]
        else:
            ch = None

        new_columns.append((cl, sl, dl, wl, ch))

    new_columns = pd.DataFrame(
        new_columns, columns=["channel", "source", "detector", "wavelength", "chromo"]
    )

    result = pd.concat((df_ml, new_columns), axis="columns")
    return result


def measurement_list_from_stacked(
    stacked_array, data_type, trial_types, stacked_channel="snirf_channel"
):
    source_labels = list(np.unique(stacked_array.source.values))
    detector_labels = list(np.unique(stacked_array.detector.values))
    wavelengths = []

    if "wavelength" in stacked_array.coords:
        wavelengths = list(np.unique(stacked_array.wavelength.values))

    nchannel = stacked_array.sizes[stacked_channel]

    ml = dict()
    ml["sourceIndex"] = [
        source_labels.index(s) + 1 for s in stacked_array.source.values
    ]
    ml["detectorIndex"] = [
        detector_labels.index(d) + 1 for d in stacked_array.detector.values
    ]

    if data_type == "amplitude":
        ml["dataType"] = [DataType.CW_AMPLITUDE.value] * nchannel
    elif data_type == "od":
        ml["dataType"] = [DataType.PROCESSED.value] * nchannel
        ml["dataTypeLabel"] = ["dOD"] * nchannel
    elif data_type == "concentration":
        ml["dataType"] = [DataType.PROCESSED.value] * nchannel
        ml["dataTypeLabel"] = stacked_array.chromo.values
    elif data_type == "hrf":
        ml["dataType"] = [DataType.PROCESSED.value] * nchannel

        if "chromo" in stacked_array.coords:
            dtl_map = {
                "HbO": DataTypeLabel.HRF_HBO,
                "HbR": DataTypeLabel.HRF_HBR,
                "HbT": DataTypeLabel.HRF_HBT,
            }
            ml["dataTypeLabel"] = [dtl_map[c] for c in stacked_array.chromo.values]
        elif "wavelength" in stacked_array.coords:
            ml["dataTypeLabel"] = [DataTypeLabel.HRF_DOD] * nchannel

        ml["dataTypeIndex"] = [
            trial_types.index(tt) + 1 for tt in stacked_array.trial_type.values
        ]

    if "wavelength" in stacked_array.coords:
        wavelengths = list(np.unique(stacked_array.wavelength.values))
        ml["wavelengthIndex"] = [
            wavelengths.index(w) + 1 for w in stacked_array.wavelength.values
        ]

    ml["dataUnit"] = [stacked_array.attrs["units"]] * nchannel

    ml = pd.DataFrame(ml)

    return source_labels, detector_labels, wavelengths, ml


def write_snirf(
    fname: str,
    data_type: str,
    timeseries: xr.DataArray,
    geo3d: xr.DataArray,
    geo2d: xr.DataArray,
    stim: pd.DataFrame,
    aux: dict[str, xr.DataArray] = {},
    meta_data: dict = {},
):
    if data_type not in ["amplitude", "od", "concentration", "hrf"]:
        raise ValueError(
            "data_type must be either 'amplitude', 'od','concentration' or 'hrf'."
        )

    other_dim = None

    if "wavelength" in timeseries.dims:
        other_dim = "wavelength"
    elif "chromo" in timeseries.dims:
        other_dim = "chromo"
    else:
        raise ValueError(
            "expect timeseries with either 'wavelength' or 'chromo' dimensions"
        )

    if data_type == "hrf":
        if "trial_type" not in timeseries.dims:
            raise ValueError(
                "to store HRFs the timeseries needs a 'trial_type' dimension"
            )
        assert timeseries.ndim == 4
        assert "channel" in timeseries.dims
        if "reltime" in timeseries.dims:
            timeseries = timeseries.rename({"reltime": "time"})
        elif "time" in timeseries.dims:
            pass
        else:
            raise ValueError("timeseries needs 'time' or 'reltime' dimension.")

        dims_to_stack = ["trial_type", "channel", other_dim]
    else:
        assert timeseries.ndim == 3
        assert "channel" in timeseries.dims
        assert "time" in timeseries.dims

        dims_to_stack = ["channel", other_dim]

    stacked_array = timeseries.stack({"snirf_channel": dims_to_stack})
    stacked_array = stacked_array.transpose("time", "snirf_channel")
    stacked_array = stacked_array.pint.dequantify()

    trial_types = list(stim["trial_type"].drop_duplicates())

    source_labels, detector_labels, wavelengths, df_ml = measurement_list_from_stacked(
        stacked_array, data_type, trial_types
    )

    geo3d = geo3d.pint.dequantify()
    geo2d = geo2d.pint.dequantify()

    with Snirf(fname, "w") as fout:
        # create and populate nirs element
        fout.nirs.appendGroup()
        ne = fout.nirs[0]

        # set meta data
        for k, v in meta_data.items():
            setattr(ne.metaDataTags, k, v)

        ne.metaDataTags.LengthUnit = geo3d.attrs["units"]

        ne.probe.sourceLabels = source_labels
        ne.probe.detectorLabels = detector_labels
        ne.probe.wavelengths = wavelengths

        ne.probe.sourcePos3D = geo3d.loc[source_labels]
        ne.probe.detectorPos3D = geo3d.loc[detector_labels]
        
        ne.probe.sourcePos2D = geo2d.loc[source_labels]
        ne.probe.detectorPos2D = geo2d.loc[detector_labels]

        # create and populate data element
        ne.data.appendGroup()
        data = ne.data[0]

        data.dataTimeSeries = stacked_array.values
        data.time = stacked_array.time.values

        # build measurement list
        for i, row in df_ml.iterrows():
            data.measurementList.appendGroup()
            ml = data.measurementList[-1]
            for k, v in row.items():
                setattr(ml, k, v)

        # save stimulus
        for trial_type in trial_types:
            df = stim[stim.trial_type == trial_type]
            ne.stim.appendGroup()
            stim_group = ne.stim[-1]
            df = df.drop(columns="trial_type")
            assert all(df.columns[:3] == ["onset", "duration", "value"])

            stim_group.data = df.values
            stim_group.name = trial_type

            if len(df.columns) > 3:
                stim_group.dataLabels = list(df.columns[3:])

        # save aux

        for aux_name, aux_array in aux:
            ne.aux.appendGroup()
            aux_group = ne.aux[-1]

            aux_array = aux_array.pint.dequantify()

            aux_group.name = aux_name
            aux_group.dataTimeSeries = aux_array
            aux_group.dataUnit = aux_array.attrs[
                "units"
            ]  # FIXME add checks that time units match those in metaDataTags
            aux_group.time = aux_array.time
            aux_group.timeOffset = aux_array.attrs["time_offset"]

        fout.save()