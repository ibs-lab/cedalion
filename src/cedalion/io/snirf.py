import logging
import re
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike
from snirf import Snirf
from snirf.pysnirf2 import DataElement, MeasurementList, NirsElement, Stim
from strenum import StrEnum

import cedalion.dataclasses as cdc
from cedalion import units
from cedalion.typing import NDTimeSeries

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

CANONICAL_NAMES = {
    "unprocessed raw": "amp",
    "processed raw": "amp",
    "processed dOD": "od",
    "processed concentrations": "conc",
    "processed central moments": "moments",
    "processed blood flow index": "bfi",
    "processed HRF dOD": "hrf_od",
    "processed HRF central moments": "hrf_moments",
    "processed HRF concentrations": "hrf_conc",
    "processed HRF blood flow index": "hrf_bfi",
    "processed absorption coefficient": "mua",
    "processed scattering coefficient": "musp",
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


def reduce_ndim_sourceLabels(sourceLabels: np.ndarray) -> list:
    """Deal with multidimensional source labels.

    snirf supports multidimensional source labels but we don't.
    This function tries to reduce n-dimensional source labels
    to a unique common prefix to obtain only one label per source.

    Args:
        sourceLabels (np.ndarray): The source labels to reduce.

    Returns:
        list: The reduced source labels.
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
    """Extract 3D coordinates of optodes and landmarks from a nirs probe variable.

    Args:
        probe: Nirs probe geometry variable, see snirf docs (:cite:t:`Tucker2022`).
        dim (int): Must be either 2 or 3.

    Returns:
        tuple: A tuple containing the source, detector, and landmark labels/positions.
    """
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

    sourcePos = sourcePos[:,0:dim]
    detectorPos = detectorPos[:,0:dim]
    landmarkPos = landmarkPos[:,0:dim]

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
            log.warning(
                "landmark labels were provided but not their positions. "
                "Removing labels"
            )
            landmarkLabels = np.asarray([], dtype=str)

    return (
        sourceLabels,
        detectorLabels,
        landmarkLabels,
        sourcePos,
        detectorPos,
        landmarkPos,
    )

def geometry_from_probe(nirs_element: NirsElement, dim: int, crs : str):
    """Extract 3D coordinates of optodes and landmarks from probe information.

    Args:
        nirs_element (NirsElement): Nirs data element as specified in the snirf
            documentation (:cite:t:`Tucker2022`).
        dim (int): Must be either 2 or 3.
        crs: the name of coordinate reference system

    Returns:
        xr.DataArray: A DataArray containing the 3D coordinates of optodes and
            landmarks, with dimensions 'label' and 'pos' and coordinates 'label' and
            'type'.
    """
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
        [cdc.PointType.SOURCE] * len(sourceLabels)
        + [cdc.PointType.DETECTOR] * len(detectorLabels)
        + [cdc.PointType.LANDMARK] * len(landmarkLabels)
    )

    labels = np.hstack([sourceLabels, detectorLabels, landmarkLabels])
    positions = np.vstack([sourcePos, detectorPos, landmarkPos])

    dims = ["label", crs]
    attrs = {"units": length_unit}

    if len(positions) == 0:
        coords = {"label": ("label", []), "type": ("label", [])}
        result = xr.DataArray(None, coords=coords, dims=dims, attrs=attrs)
    elif len(labels) == len(positions):
        coords = {"label": ("label", labels), "type": ("label", types)}
        result = xr.DataArray(positions, coords=coords, dims=dims, attrs=attrs)
    else:
        raise ValueError("number of positions and labels does not match")

    result = result.set_xindex("type")
    result = result.pint.quantify()

    return result


def measurement_list_to_dataframe(
    measurement_list: MeasurementList, drop_none: bool = False
) -> pd.DataFrame:
    """Converts a snirf MeasurementList object to a pandas DataFrame.

    Args:
        measurement_list: MeasurementList object from the snirf file.
        drop_none (bool): If True, drop columns that are None for all rows.

    Returns:
        pd.DataFrame: DataFrame containing the measurement list information.
    """
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


def meta_data_tags_to_dict(nirs_element: NirsElement) -> OrderedDict[str, Any]:
    """Converts the metaDataTags of a nirs element to a dictionary.

    Args:
        nirs_element (NirsElement): Nirs data element as specified in the snirf
            documentation (:cite:t:`Tucker2022`).

    Returns:
        OrderedDict[str, Any]: Dictionary containing the metaDataTags information.
    """
    mdt = nirs_element.metaDataTags

    fields = mdt._snirf_names + mdt._unspecified_names

    return OrderedDict({f: getattr(mdt, f) for f in fields})


def stim_to_dataframe(stim: Stim):
    """Converts a snirf Stim object to a pandas DataFrame.

    Args:
        stim (Stim): Stim object as specified in the snirf documentation
            (:cite:t:`Tucker2022`).

    Returns:
        pd.DataFrame: DataFrame containing the stimulus information.
    """
    dfs = []

    if len(stim) == 0:
        return cdc.build_stim_dataframe()

    for i_st, st in enumerate(stim):
        if st.data is None:
            tmp = cdc.build_stim_dataframe()
        elif st.data.ndim != 2:
            log.warning(f"unexpected shape of stim element {i_st}")
            tmp = cdc.build_stim_dataframe()
        else:
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


def read_aux(
    nirs_element: NirsElement, opts: dict[str, Any]
) -> OrderedDict[str, xr.DataArray]:
    """Reads the aux data from a nirs element into a dictionary of DataArrays.

    Args:
        nirs_element (NirsElement): Nirs data element as specified in the snirf
            documentation (:cite:t:`Tucker2022`).
        opts (dict[str, Any]): Options for reading the aux data. The following
            options are supported:
            - squeeze_aux (bool): If True, squeeze the aux data to remove
                dimensions of size 1.

    Returns:
        result (OrderedDict[str, xr.DataArray]): Dictionary containing the aux data
    """
    result = OrderedDict()

    for aux in nirs_element.aux:
        name = aux.name
        aux_units = aux.dataUnit
        time_offset = aux.timeOffset

        # FIXME treat unspecified units as dimensionless quantities.
        if aux_units is None:
            aux_units = "1"

        if aux_units not in units:
            raise ValueError(f"aux channel '{name}' has units '{aux_units}', which "
                             "are not defined in the unit registry. Consider adding "
                              "an alias to cedalion.units." )

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
            attrs={"units": aux_units, "time_offset": time_offset},
        )

        result[name] = x.pint.quantify()

    return result


def add_number_to_name(name, keys):
    """Changes name to name_<number>.

    Number appended to name is the smallest number that makes the new name unique with
    respect to the list of keys.

    Args:
        name (str): Name to which a number should be added.
        keys (list[str]): List of keys to which the new name should be compared.

    Returns:
        str: New name with number added.
    """

    pat = re.compile(rf"{name}(_(\d+))?")
    max_number = 1
    for k in keys:
        if match := pat.match(k):
            groups = match.groups()
            if groups[1] is not None:
                number = int(groups[1])
                max_number = max(max_number, number)
    return f"{name}_{max_number+1:02d}"


def read_data_elements(
    data_element: DataElement, nirs_element: NirsElement, stim: pd.DataFrame
) -> list[tuple[str, NDTimeSeries]]:
    """Reads the data elements from a nirs element into a list of DataArrays.

    Args:
        data_element (DataElement): DataElement obj. from the snirf file.
        nirs_element (NirsElement): Nirs data element as specified in the snirf
            documentation (:cite:t:`Tucker2022`).
        stim (pd.DataFrame): DataFrame containing the stimulus information.

    Returns:
        list[tuple[str, NDTimeSeries]]: List of tuples containing the canonical name
            of the data element and the DataArray.
    """
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
        canonical_name = CANONICAL_NAMES[data_type_group]

        has_wavelengths = not pd.isna(df.wavelength).all()
        has_chromo = not pd.isna(df.chromo).all()

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

        data_arrays.append((canonical_name, da))

    return data_arrays


def _get_time_coords(
    nirs_element: NirsElement,
    data_element: DataElement,
    df_measurement_list: pd.DataFrame,
) -> dict[str, ArrayLike]:
    """Get time coordinates for the NIRS data element.

    Args:
        nirs_element (NirsElement): NIRS data element containing metadata.
        data_element (DataElement): Data element containing time and dataTimeSeries.
        df_measurement_list (pd.DataFrame): DataFrame containing the measurement list.

    Returns:
        tuple: A tuple containing:
            - indices (None): Placeholder for indices.
            - coordinates (dict[str, ArrayLike]): Dictionary with time coordinates.
    """
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
    """Get channel coordinates for the NIRS data element.

    Args:
        nirs_element (NirsElement): NIRS data element containing probe information.
        df_measurement_list (pd.DataFrame): DataFrame containing the measurement list.

    Returns:
        tuple: A tuple containing:
            - indices (None): Placeholder for indices.
            - coordinates (dict[str, ArrayLike]): Dictionary with channel coordinates.
    """
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
    """Reads a single nirs element from a .snirf file into a Recording object.

    Args:
        nirs_element (NirsElement): Nirs data element as specified in the snirf
            documentation (:cite:t:`Tucker2022`).
        opts (dict[str, Any]): Options for reading the data element. The following
            options are supported:
            - squeeze_aux (bool): If True, squeeze the aux data to remove
                dimensions of size 1.
            - crs (str): name of the geo?d's coordinate reference system.

    Returns:
        rec (Recording): Recording object containing the data from the nirs element.
    """

    geo2d = geometry_from_probe(nirs_element, dim=2, crs=opts["crs"])
    geo3d = geometry_from_probe(nirs_element, dim=3, crs=opts["crs"])
    stim = stim_to_dataframe(nirs_element.stim)

    timeseries = OrderedDict()
    measurement_lists = OrderedDict()

    for data_element in nirs_element.data:
        df_ml = measurement_list_to_dataframe(
            data_element.measurementList, drop_none=False
        )
        df_ml = denormalize_measurement_list(df_ml, nirs_element)
        df_ml.dropna(axis=1)

        for name, ts in read_data_elements(data_element, nirs_element, stim):
            if name in timeseries:
                name = add_number_to_name(name, timeseries.keys())
            timeseries[name] = ts
            measurement_lists[name] = df_ml

    aux = read_aux(nirs_element, opts)

    meta_data = meta_data_tags_to_dict(nirs_element)

    rec = cdc.Recording(
        timeseries=timeseries,
        geo3d=geo3d,
        geo2d=geo2d,
        stim=stim,
        aux_ts=aux,
        meta_data=meta_data,
        _measurement_lists=measurement_lists,
    )

    return rec


def read_snirf(
    fname: Path | str, crs: str = "pos", squeeze_aux: bool = False
) -> list[cdc.Recording]:
    """Reads a .snirf file into a list of Recording objects.

    Args:
        fname: Path to .snirf file
        crs: the name of the geo3D's coordinate reference system
        squeeze_aux: If True, squeeze the aux data to remove dimensions of size 1.

    Returns:
        list[Recording]: List of Recording objects containing the data from the nirs
        elements in the .snirf file.
    """
    opts = {"squeeze_aux": squeeze_aux, "crs" : crs}

    if isinstance(fname, Path):
        fname = str(fname)

    with Snirf(fname, "r") as s:
        return [read_nirs_element(ne, opts) for ne in s.nirs]


def denormalize_measurement_list(df_ml: pd.DataFrame, nirs_element: NirsElement):
    """Enriches measurement list DataFrame with additional information.

    Args:
        df_ml (pd.DataFrame): DataFrame containing the measurement list information.
        nirs_element (NirsElement): Nirs data element as specified in the snirf
            documentation (:cite:t:`Tucker2022`).

    Returns:
        pd.DataFrame: DataFrame containing the measurement list information with
            additional columns for channel, source, detector, wavelength and chromo.

    """
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
    stacked_array,
    data_type,
    trial_types,
    stacked_channel="snirf_channel",
    source_labels=None,
    detector_labels=None,
    wavelengths=None,
):
    """Create a measurement list from a stacked array.

    Args:
        stacked_array (xr.DataArray): Stacked array containing the data.
        data_type (str): Data type of the data.
        trial_types (list[str]): List of trial types.
        stacked_channel (str): Name of the channel dimension in the stacked array.
        source_labels (list[str]): List of source labels.
        detector_labels (list[str]): List of detector labels.
        wavelengths (list[float]): List of wavelengths.

    Returns:
        tuple: A tuple containing the source labels, detector labels, wavelengths, and
            the measurement list.
    """
    if source_labels is None:
        source_labels = list(np.unique(stacked_array.source.values))
    if detector_labels is None:
        detector_labels = list(np.unique(stacked_array.detector.values))
    if wavelengths is None:
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


def _write_recordings(snirf_file: Snirf, rec: cdc.Recording):
    """Write a recording to a .snirf file.

    See snirf specification for details (:cite:t:`Tucker2022`)

    Args:
        snirf_file (Snirf): Snirf object to write to.
        rec (Recording): Recording object to write to the file.
    """
    # create and populate nirs element
    snirf_file.nirs.appendGroup()
    ne = snirf_file.nirs[-1]

    # meta data
    for k, v in rec.meta_data.items():
        setattr(ne.metaDataTags, k, v)

    geo3d = rec.geo3d.pint.dequantify()
    geo2d = rec.geo2d.pint.dequantify()
    ne.metaDataTags.LengthUnit = geo3d.attrs["units"]

    # probe information
    ne.probe.sourceLabels = rec.source_labels
    ne.probe.detectorLabels = rec.detector_labels
    ne.probe.wavelengths = rec.wavelengths

    if len(geo3d) > 0:
        ne.probe.sourcePos3D = geo3d.loc[rec.source_labels]
        ne.probe.detectorPos3D = geo3d.loc[rec.detector_labels]

    if len(geo2d) > 0:
        ne.probe.sourcePos2D = geo2d.loc[rec.source_labels]
        ne.probe.detectorPos2D = geo2d.loc[rec.detector_labels]

    trial_types = list(rec.stim["trial_type"].drop_duplicates())

    for key, timeseries in rec.timeseries.items():
        data_type = rec.get_timeseries_type(key)

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

        # FIXME refactor
        _, _, _, df_ml = measurement_list_from_stacked(
            stacked_array,
            data_type,
            trial_types,
            source_labels=rec.source_labels,
            detector_labels=rec.detector_labels,
            wavelengths=rec.wavelengths,
        )

        # create and populate data element
        ne.data.appendGroup()
        data = ne.data[-1]

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
        df = rec.stim[rec.stim.trial_type == trial_type]
        ne.stim.appendGroup()
        stim_group = ne.stim[-1]
        df = df.drop(columns="trial_type")
        assert all(df.columns[:3] == ["onset", "duration", "value"])

        stim_group.data = df.values
        stim_group.name = trial_type

        if len(df.columns) > 3:
            stim_group.dataLabels = list(df.columns[3:])

    # save aux

    for aux_name, aux_array in rec.aux_ts.items():
        ne.aux.appendGroup()
        aux_group = ne.aux[-1]

        aux_array = aux_array.pint.dequantify()

        aux_group.name = aux_name
        aux_group.dataTimeSeries = aux_array
        aux_group.dataUnit = aux_array.attrs["units"]
        # FIXME add checks that time units match those in metaDataTags
        aux_group.time = aux_array.time
        aux_group.timeOffset = aux_array.attrs["time_offset"]


def write_snirf(
    fname: Path | str,
    recordings: cdc.Recording | list[cdc.Recording],
):
    """Write one or more recordings to a .snirf file.

    Args:
        fname (Path | str): Path to .snirf file.
        recordings (Recording | list[Recording]): Recording object(s) to write to the
            file.
    """
    if isinstance(fname, Path):
        fname = str(fname)

    if isinstance(recordings, cdc.Recording):
        recordings = [recordings]

    with Snirf(fname, "w") as fout:
        for rec in recordings:
            _write_recordings(fout, rec)

        fout.save()
