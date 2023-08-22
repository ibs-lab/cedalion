import numpy as np
import xarray as xr
import pandas as pd
from snirf import Snirf
from snirf.pysnirf2 import MeasurementList, DataElement
from enum import Enum
import logging
from numpy.typing import ArrayLike
from functools import total_ordering

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


@total_ordering
class PointType(Enum):
    SOURCE = 1
    DETECTOR = 2
    LANDMARK = 3

    # provide an ordering of PointTypes so that e.g. np.unique works
    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        else:
            return NotImplemented


# result container for read_snirf
# FIXME: use a better name to avoid confusion with pysnirf's NirsElement
class NIRSElement:
    def __init__(self, data=None, stim=None, geo3d=None, geo2d=None):
        self.data = []
        self.stim = pd.DataFrame(columns=["onset", "duration", "value"])
        self.geo3d = xr.DataArray(None, coords={"label": []}, dims=["label", "pos"])
        self.geo2d = xr.DataArray(None, coords={"label": []}, dims=["label", "pos"])

        if data is not None:
            self.data = data
        if stim is not None:
            self.stim = stim
        if geo3d is not None:
            self.geo3d = geo3d
        if geo2d is not None:
            self.geo2d = geo2d


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


def labels_and_positions(probe):
    def convert_none(probe, attrname, default):
        attr = getattr(probe, attrname)
        if attr is None:
            return default
        else:
            return attr

    sourceLabels = convert_none(probe, "sourceLabels", np.asarray([], dtype=str))
    detectorLabels = convert_none(probe, "detectorLabels", np.asarray([], dtype=str))
    landmarkLabels = convert_none(probe, "landmarkLabels", np.asarray([], dtype=str))

    if sourceLabels.ndim > 1:
        sourceLabels = reduce_ndim_sourceLabels(sourceLabels)

    sourcePos3D = convert_none(probe, "sourcePos3D", np.zeros((0, 3)))
    detectorPos3D = convert_none(probe, "detectorPos3D", np.zeros((0, 3)))
    landmarkPos3D = convert_none(probe, "landmarkPos3D", np.zeros((0, 3)))[
        :, :3
    ]  # FIXME we keep only the positional columns

    if len(sourcePos3D) > 0 and len(sourceLabels) == 0:
        log.warn("generating generic source labels")
        sourceLabels = np.asarray([f"S{i}" for i in range(len(sourcePos3D))])

    if len(detectorPos3D) > 0 and len(detectorLabels) == 0:
        log.warn("generating generic detector labels")
        detectorLabels = np.asarray([f"D{i}" for i in range(len(detectorPos3D))])

    if len(landmarkLabels) != len(landmarkPos3D):
        raise ValueError("landmark positions were provided but no labels")

    return (
        sourceLabels,
        detectorLabels,
        landmarkLabels,
        sourcePos3D,
        detectorPos3D,
        landmarkPos3D,
    )


def geometry_from_probe(nirs_element: NIRSElement):
    probe = nirs_element.probe

    length_unit = nirs_element.metaDataTags.LengthUnit

    (
        sourceLabels,
        detectorLabels,
        landmarkLabels,
        sourcePos3D,
        detectorPos3D,
        landmarkPos3D,
    ) = labels_and_positions(probe)

    types = (
        [PointType.SOURCE] * len(sourceLabels)
        + [PointType.DETECTOR] * len(detectorLabels)
        + [PointType.LANDMARK] * len(landmarkLabels)
    )

    labels = np.hstack([sourceLabels, detectorLabels, landmarkLabels])
    positions = np.vstack([sourcePos3D, detectorPos3D, landmarkPos3D])

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


def stim_to_dataframe(stim):
    dfs = []

    if len(stim) == 0:
        columns = ["onset", "duration", "value"]
        return pd.DataFrame(columns=columns)

    for st in stim:
        columns = ["onset", "duration", "value"]

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


def read_data_element(data_element, nirs_element):
    time = data_element.time
    samples = np.arange(len(time))

    if len(time) != data_element.dataTimeSeries.shape[0]:
        raise ValueError("length of time and dataTimeSeries arrays don't match!")

    df_ml = measurement_list_to_dataframe(data_element.measurementList)

    unique_data_types = df_ml.dataType.unique()
    if not (
        len(unique_data_types)
        and DataType(unique_data_types.item()) == DataType.CW_AMPLITUDE
    ):
        raise ValueError("currently only CW_AMPLITUDE data is supported")

    wavelengths = nirs_element.probe.wavelengths

    sourceLabels, detectorLabels, landmarkLabels, _, _, _ = labels_and_positions(
        nirs_element.probe
    )

    df_labels = pd.DataFrame(
        [
            (f"{sl}{dl}", sl, dl)
            for sl, dl in zip(
                sourceLabels[df_ml.sourceIndex - 1],
                detectorLabels[df_ml.detectorIndex - 1],
            )
        ],
        columns=["channel", "source", "detector"],
    )

    unique_labels = df_labels.drop_duplicates(ignore_index=True)
    channel_labels = df_labels["channel"].tolist()
    unique_channel_labels = unique_labels[
        "channel"
    ].tolist()  # sorted(set(channel_labels))

    timeseries = np.zeros((len(unique_channel_labels), len(wavelengths), len(time)))

    for i, r in df_ml.iterrows():
        i_channel = unique_channel_labels.index(channel_labels[i])
        i_wavelength = r.wavelengthIndex - 1
        timeseries[i_channel, i_wavelength, :] = data_element.dataTimeSeries[:, i]

    units = df_ml.dataUnit.unique().item()
    # FIXME treat unspecified units as dimensionless quantities.
    if units is None:
        units = "1"

    da = xr.DataArray(
        timeseries,
        dims=["channel", "wavelength", "time"],
        coords=dict(
            time=("time", time),
            samples=("time", samples),
            wavelength=wavelengths,
            channel=("channel", unique_labels["channel"]),  # unique_channel_labels
            source=("channel", unique_labels["source"]),
            detector=("channel", unique_labels["detector"]),
        ),
        attrs={"units": units},
    )

    da.time.attrs["unit"] = nirs_element._metaDataTags.TimeUnit
    da = da.pint.quantify()

    # da = da.drop_indexes(["channel", "time"])
    # da = da.set_xindex(["channel", "source", "detector"])
    # da = da.set_xindex(["time", "samples"])

    return da


def _get_time_index_and_coords(
    nirs_element: NIRSElement,
    data_element: DataElement,
    df_measurement_list: pd.DataFrame,
) -> tuple[ArrayLike, dict[str, ArrayLike]]:
    time = data_element.time
    time_unit = nirs_element.metaDataTags.TimeUnit

    indices = np.arange(len(time))

    if len(time) != data_element.dataTimeSeries.shape[0]:
        raise ValueError("length of time and dataTimeSeries arrays don't match!")

    coordinates = {
        "time": ("time", xr.DataArray(time, attrs={"units": time_unit})),
        "samples": ("samples", np.arange(len(time))),
    }

    return indices, coordinates


def _get_channel_index_and_coords(
    nirs_element: NIRSElement,
    data_element: DataElement,
    df_measurement_list: pd.DataFrame,
) -> tuple[ArrayLike, dict[str, ArrayLike]]:
    sourceLabels, detectorLabels, landmarkLabels, _, _, _ = labels_and_positions(
        nirs_element.probe
    )

    # for each channel in the measurement list build a channel label
    df_channels = pd.DataFrame(
        [
            (f"{sl}{dl}", sl, dl)
            for sl, dl in zip(
                sourceLabels[df_measurement_list.sourceIndex - 1],
                detectorLabels[df_measurement_list.detectorIndex - 1],
            )
        ],
        columns=["channel_label", "source", "detector"],
    )

    # reduce channels to unique source-detector combinations
    df_unique_channels = df_channels.drop_duplicates(ignore_index=True)
    unique_channel_labels = df_unique_channels["channel_label"].tolist()

    indices = [
        unique_channel_labels.index(label) for label in df_channels["channel_label"]
    ]

    coordinates = {
        "channel": ("channel", df_unique_channels["channel_label"]),
        "source": ("channel", df_unique_channels["source"]),
        "detector": ("channel", df_unique_channels["detector"]),
    }

    return indices, coordinates


# def read_data_element_2(
#    data_element: DataElement, nirs_element: NIRSElement
# ) -> xr.DataArray:
#    """TBD."""
#    df_ml = measurement_list_to_dataframe(data_element.measurementList)
#
#    unique_data_types = df_ml["dataType"].unique()
#    if len(unique_data_types) > 1:
#        raise ValueError(
#            f"data element at {data_element.location} contains multiple data types"
#        )
#
#    data_type = DataType(unique_data_types.item())
#
#    match data_type:
#        case DataType.CW_AMPLITUDE:
#            dims = ["channel", "wavelength", "time"]
#        case DataType.TDM_AMPLITUDE:
#            dims = ["channel", "wavelength", "moment", "time"]
#            pass
#        case DataType.PROCESSED:
#            pass
#        case _:
#            raise NotImplementedError(f"data type {data_type} is not yet supported.")
#
#    time = data_element.time
#    samples = np.arange(len(time))
#
#    if len(time) != data_element.dataTimeSeries.shape[0]:
#        raise ValueError("length of time and dataTimeSeries arrays don't match!")
#
#    sourceLabels, detectorLabels, landmarkLabels, _, _, _ = labels_and_positions(
#        nirs_element.probe
#    )


def read_nirs_element(nirs_element):
    geo3d = geometry_from_probe(nirs_element)
    stim = stim_to_dataframe(nirs_element.stim)
    data = [
        read_data_element(data_element, nirs_element)
        for data_element in nirs_element.data
    ]

    ne = NIRSElement(data, stim, geo3d)  # FIXME geo2d

    return ne  # (geo, data, stim)


def read_snirf(fname):
    with Snirf(fname, "r") as s:
        return [read_nirs_element(ne) for ne in s.nirs]


def write_snirf(fname, *nirs_elements):
    pass
