import numpy as np
import xarray as xr
import pandas as pd
from snirf import Snirf
from snirf.pysnirf2 import MeasurementList, DataElement, NirsElement
from enum import Enum
import logging
from numpy.typing import ArrayLike
from functools import total_ordering
from typing import Dict


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
        sourceLabels = np.asarray([f"S{i+1}" for i in range(len(sourcePos3D))])

    if len(detectorPos3D) > 0 and len(detectorLabels) == 0:
        log.warn("generating generic detector labels")
        detectorLabels = np.asarray([f"D{i+1}" for i in range(len(detectorPos3D))])

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


def geometry_from_probe(nirs_element: NirsElement):
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


def meta_data_tags_to_dict(nirs_element: NirsElement):
    mdt = nirs_element.metaDataTags

    fields = mdt._snirf_names + mdt._unspecified_names

    return {f: getattr(mdt, f) for f in fields}


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


def read_aux(nirs_element: NirsElement):
    result = {}

    for aux in nirs_element.aux:
        name = aux.name
        units = aux.dataUnit
        time_offset = aux.timeOffset

        # FIXME treat unspecified units as dimensionless quantities.
        if units is None:
            units = "1"

        if aux.dataTimeSeries.ndim == 1:
            dims = ["time"]
        elif aux.dataTimeSeries.ndim == 2:
            dims = ["time", "aux_channel"]
        else:
            raise ValueError("aux.dataTimeSeries must have either 1 or 2 dimensions.")

        x = xr.DataArray(
            aux.dataTimeSeries,
            coords={"time": aux.time},
            dims=dims,
            name=name,
            attrs={"units": units, "time_offset": time_offset},
        )
        result[name] = x.pint.quantify()

    return result


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


def read_nirs_element(nirs_element):
    geo3d = geometry_from_probe(nirs_element)
    stim = stim_to_dataframe(nirs_element.stim)
    data = [
        read_data_element(data_element, nirs_element)
        for data_element in nirs_element.data
    ]

    aux = read_aux(nirs_element)

    meta_data = meta_data_tags_to_dict(nirs_element)

    measurement_lists = []
    for data_element in nirs_element.data:
        df_ml = measurement_list_to_dataframe(
            data_element.measurementList, drop_none=True
        )
        df_ml = denormalize_measurement_list(df_ml, nirs_element)
        measurement_lists.append(df_ml)

    ne = Element(
        data, stim, geo3d, None, aux, meta_data, measurement_lists
    )  # FIXME geo2d

    return ne


def read_snirf(fname):
    with Snirf(fname, "r") as s:
        return [read_nirs_element(ne) for ne in s.nirs]


def denormalize_measurement_list(df_ml: pd.DataFrame, nirs_element: NirsElement):
    sourceLabels, detectorLabels, landmarkLabels, _, _, _ = labels_and_positions(
        nirs_element.probe
    )
    wavelengths = nirs_element.probe.wavelengths

    if (df_ml.sourceIndex < 1).any() or (df_ml.detectorIndex < 1).any():
        raise ValueError(
            "measurement list has row(s) for which source or detector index < 1."
        )

    new_columns = []
    for _, row in df_ml.iterrows():
        sl = sourceLabels[int(row["sourceIndex"]) - 1]
        dl = detectorLabels[int(row["detectorIndex"]) - 1]
        cl = f"{sl}{dl}"
        if row["wavelengthIndex"] != -1:
            wl = wavelengths[int(row["wavelengthIndex"]) - 1]
        else:
            wl = np.nan

        new_columns.append((cl, sl, dl, wl))

    new_columns = pd.DataFrame(
        new_columns, columns=["channel", "source", "detector", "wavelength"]
    )
    result = pd.concat((df_ml, new_columns), axis="columns")
    return result


def measurement_list_from_stacked(
    stacked_array, data_type, stacked_channel="snirf_channel"
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

    if "wavelength" in stacked_array.coords:
        wavelengths = list(np.unique(stacked_array.wavelength.values))
        ml["wavelengthIndex"] = [
            wavelengths.index(w) + 1 for w in stacked_array.wavelength.values
        ]
    else:
        ml["wavelengthIndex"] = [-1] * nchannel

    ml["dataUnit"] = [stacked_array.attrs["units"]] * nchannel

    return source_labels, detector_labels, wavelengths, pd.DataFrame(ml)


def write_snirf(
    fname: str,
    data_type: str,
    timeseries: xr.DataArray,
    geo3d: xr.DataArray,
    stim: pd.DataFrame,
    measurement_list: pd.DataFrame,
    aux: Dict[str, xr.DataArray] = {},
    meta_data: Dict = {},
):
    if data_type not in ["amplitude", "od", "concentration"]:
        raise ValueError(
            "data_type must be either 'amplitude', 'od' or 'concentration'."
        )

    assert timeseries.ndim == 3

    assert "channel" in timeseries.dims
    assert "time" in timeseries.dims

    other_dim = None

    if "wavelength" in timeseries.dims:
        other_dim = "wavelength"
    elif "chromo" in timeseries.dims:
        other_dim = "chromo"
    else:
        raise ValueError(
            "expect 3-dim timeseries with either 'wavelength' or 'chromo' dimensions"
        )

    stacked_array = timeseries.stack({"snirf_channel": ["channel", other_dim]})
    stacked_array = stacked_array.transpose("time", "snirf_channel")
    stacked_array = stacked_array.pint.dequantify()

    source_labels, detector_labels, wavelengths, df_ml = measurement_list_from_stacked(
        stacked_array, data_type
    )

    geo3d = geo3d.pint.dequantify()

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
        for trial_type, df in stim.groupby("trial_type"):
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
