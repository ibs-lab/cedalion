"""Utilities for converting fNIRS datasets to the BIDS standard.

Provides functions to organise raw SNIRF files into a BIDS directory tree,
generate BIDS-compliant filenames, create required sidecar files
(``dataset_description.json``, ``participants.tsv``/``.json``,
``_scans.tsv``, ``_sessions.tsv``), and read/write optode positions in the
BIDS ``_optodes.tsv`` / ``_coordsystem.json`` format.

References:
    :cite:t:`Gorgolewski2016`, :cite:t:`Luke2025`
"""
import os
import shutil
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import xarray as xr
from snirf import Snirf

import cedalion.typing as cdt
from cedalion.dataclasses import PointType, build_labeled_points

from cedalion import cite


def read_events_from_tsv(fname: str | Path):
    return pd.read_csv(fname, delimiter="\t")


def check_for_bids_field(path_parts: list, field: str):
    """@author: lauracarlton."""
    field_parts = [part for part in path_parts if field in part]
    if len(field_parts) == 0:
        value_id = None
    else:
        # assume the lowest directory level supersedes any higher directory level?
        # not sure if we should assume this
        find_value = field_parts[-1].split("_")
        value = [vals for vals in find_value if field in vals][0]
        value_id = value.split(field)[1]
        try:
            value_id = value_id.split("-")[1]
        except IndexError:
            value_id = value_id

    return value_id


def get_snirf2bids_mapping_csv(dataset_path):
    """@author: lauracarlton."""
    column_names = [
        "current_name",
        "sub",
        "ses",
        "task",
        "run",
        "acq",
        "cond",
        "cond_match",
        "duration",
    ]

    snirf2bids_mapping_df = pd.DataFrame(columns=column_names)

    # % IDENTIFY ALL SNIRF FILES IN THE DIRECTORY AND THEIR PATH

    file_list = []
    for dirpath, dirnames, filenames in os.walk(dataset_path):
        for filename in filenames:
            if filename.endswith(".snirf"):
                # Get the full path of the file
                relative_path = os.path.relpath(dirpath, dataset_path)

                # get each part of the path
                parent_folders = relative_path.split(os.sep)

                # including the filename
                filename_without_ext = os.path.splitext(filename)[0]
                parent_folders.append(filename_without_ext)

                # add to the list of file paths
                file_list.append(parent_folders)

    # % CHECK EACH FILE TO GATHER INFO TO POPULATE THE MAPPING_DF

    for path_parts in file_list:
        # need to check for sub
        subject = check_for_bids_field(path_parts, "sub")

        # check for session
        ses = check_for_bids_field(path_parts, "ses")

        # check for run
        run = check_for_bids_field(path_parts, "run")

        # check for task
        task = check_for_bids_field(path_parts, "task")

        # check for acq
        acq = check_for_bids_field(path_parts, "acq")

        bids_dict = {
            "current_name": "/".join(path_parts),
            "sub": subject,
            "ses": ses,
            "run": run,
            "task": task,
            "acq": acq,
            "cond": None,
            "cond_match": None,
            "duration": None,
        }
        snirf2bids_mapping_df = pd.concat(
            [snirf2bids_mapping_df, pd.DataFrame([bids_dict])], ignore_index=True
        )

    mapping_df_path = os.path.join(dataset_path, "snirf2BIDS_mapping.csv")
    snirf2bids_mapping_df.to_csv(mapping_df_path, index=None)
    return mapping_df_path


def find_files_with_pattern(start_dir: str | Path, pattern: str) -> List[str]:
    """Recursively finds all files matching the given pattern.

    Searches in the specified directory and subdirectories.

    Args:
        start_dir: The directory to start the search from.
        pattern: The pattern to match filenames against.

    Returns:
        A list of file paths (as strings) of all files that match the pattern.
    """
    start_path = Path(start_dir)
    return [str(file) for file in start_path.rglob(pattern)]


def create_bids_standard_filenames(row: pd.Series) -> Tuple[str, str]:
    """Generates a BIDS compliant file name and its parent directory path.

    Constructs a filename and directory path following the BIDS naming convention
    for a specific subject, session, task, acquisition, and run. The filename
    includes ``_nirs.snirf`` as the extension and the directory is nested under
    a ``nirs`` subdirectory.

    Args:
        row: A row of a Pandas DataFrame with the following columns:

            - ``"sub"``: The subject identifier (e.g., ``"01"``).
            - ``"ses"``: The session identifier (e.g., ``"01"``), can be NaN.
            - ``"task"``: The task name or identifier (e.g., ``"rest"``).
            - ``"acq"``: The acquisition identifier, can be NaN.
            - ``"run"``: The run identifier (e.g., ``"1"``), can be NaN.

    Returns:
        A tuple of ``(bids_filename, parent_directory_path)``.
    """

    cite("Gorgolewski2016")
    cite("Luke2025")

    name_str = "sub-" + str(row["sub"])
    parent_path = name_str
    if not pd.isna(row["ses"]):
        name_str += "_ses-" + str(row.ses)
        parent_path = os.path.join(parent_path, "ses-" + str(row.ses))

    name_str += "_task-" + str(row.task)

    if not pd.isna(row["acq"]):
        name_str += "_acq-" + str(row.acq)

    if not pd.isna(row["run"]):
        name_str += "_run-" + str(row.run)

    name_str += "_nirs.snirf"
    parent_path = os.path.join(parent_path, "nirs")

    return name_str, parent_path


def copy_rename_snirf(row: pd.Series, dataset_path: str, bids_dir: str):
    """Copies and renames a `.snirf` into the appropriate destination directory.

    This function takes the source file (in the `dataset_path`), renames it based on the
    information in the provided `row`, and copies it to the target `bids_dir` directory,
    following the BIDS directory structure.

    Args:
        row (pd.Series): A row from a Pandas DataFrame containing the following columns:

            - ``"current_name"``: The current name of the file (without the
              `.snirf` extension).
            - ``"parent_path"``: The relative path within the BIDS structure
              where the file should be stored.
            - ``"bids_name"``: The new BIDS-compliant name for the file.

        dataset_path (str): The path to the directory containing the original
            `.snirf` file(s) to be copied.
        bids_dir (str): The path to the root BIDS directory where the renamed
            file should be copied to.
    """

    source_file = os.path.join(dataset_path, row["current_name"] + ".snirf")
    destination_folder = os.path.join(bids_dir, row["parent_path"])
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    destination_file = os.path.join(destination_folder, row["bids_name"])
    shutil.copy(source_file, destination_file)


def search_for_acq_time_in_scan_files(dataset_path: str) -> pd.DataFrame:
    """Searches for `_scans.tsv` files in dataset_path and extracts acquisition times.

    Looks for all `_scans.tsv` files in `dataset_path`, reads them into a DataFrame, and
    processes the `filename` and `acq_time` columns. If `acq_time` does not exist in the
    merged DataFrame, it is added with ``None`` values. If no `_scans.tsv` files are
    found, an empty DataFrame with columns `filename_org` and `acq_time` is returned.

    Args:
        dataset_path (str): The path to the dataset where the `_scans.tsv` files are
            located.

    Returns:
        pd.DataFrame: A DataFrame with the following columns:

            - ``"filename_org"``: The original filename (without the `.snirf` extension)
              from the `_scans.tsv` files.
            - ``"acq_time"``: The acquisition time for each scan, or ``None`` if the
              column does not exist in the original files.
    """

    scan_paths = find_files_with_pattern(dataset_path, "*_scans.tsv")
    scan_dfs = [pd.read_csv(file, delimiter="\t") for file in scan_paths]

    if len(scan_dfs) != 0:
        scan_df = pd.concat(scan_dfs, ignore_index=True)
        scan_df.drop_duplicates(subset="filename", inplace=True)

        scan_df["filename"] = scan_df["filename"].apply(
            lambda x: str(os.path.basename(x)).replace(".snirf", "")
        )
        scan_df = scan_df.rename(columns={"filename": "filename_org"})
        if "acq_time" not in scan_df.columns:
            scan_df["acq_time"] = None
    else:
        scan_df = pd.DataFrame(columns=["filename_org", "acq_time"])
    return scan_df


def search_for_acq_time_in_snirf_files(row: pd.Series, dataset_path: str) -> datetime:
    """Extracts acquisition time from SNIRF files if missing in the `_scans.tsv` file.

    Checks if `acq_time` is ``NaN`` in the input row. If missing, loads the
    corresponding SNIRF file, extracts the measurement date and time, and returns
    it as an ISO 8601 timestamp string.

    Args:
        row: A row from ``mapping_df`` containing ``current_name`` and
            ``acq_time`` columns.
        dataset_path: Path to the dataset where the SNIRF files are located.

    Returns:
        The acquisition timestamp extracted from the SNIRF file, or the existing
        ``acq_time`` value if it is not missing.
    """
    if pd.isna(row.acq_time):
        snirf_file = os.path.join(dataset_path, row.current_name)
        with Snirf(snirf_file) as snirf_obj:
            nirs_group = next(iter(snirf_obj.nirs))
            if (
                nirs_group.metaDataTags.MeasurementDate is None
                or nirs_group.metaDataTags.MeasurementTime is None
            ):
                return "n/a"
            datetime_str = (
                f"{nirs_group.metaDataTags.MeasurementDate}"
                f"T{nirs_group.metaDataTags.MeasurementTime}"
            )
        timestamp = datetime_str.split(".")[0]
        return timestamp
    else:
        return row.acq_time


def search_for_sessions_acq_time(dataset_path: str) -> pd.DataFrame:
    """Searches `_sessions.tsv` files in the dataset path and returns session times.

    Looks for all `_sessions.tsv` files in `dataset_path`, reads them into
    DataFrames, and extracts the subject ID and session acquisition time. If
    `acq_time` does not exist in the input files, it is added with ``None``
    values. Subject IDs are extracted from filenames using a regular expression.

    Args:
        dataset_path: The path to the dataset where `_sessions.tsv` files are
            located.

    Returns:
        A DataFrame with the following columns:

            - ``"ses"``: The session identifier (extracted from the filenames).
            - ``"sub"``: The subject ID extracted from the filename.
            - ``"ses_acq_time"``: The session acquisition time, or ``None`` if
              ``acq_time`` does not exist in the original files.
    """

    session_paths = find_files_with_pattern(dataset_path, "*_sessions.tsv")
    session_dfs = []
    for f in session_paths:
        match = re.search(r"(?i)\b(?:sub|subj|subject)[-_]?(\d+)\b", f)
        ses_df = pd.read_csv(f, delimiter="\t")
        ses_df["sub"] = match.group(1)
        session_dfs.append(ses_df)
        if "acq_time" in ses_df.columns:
            ses_df = ses_df.rename(columns={"acq_time": "ses_acq_time"})
        else:
            ses_df["ses_acq_time"] = 'n/a'

    if len(session_dfs) != 0:
        session_df = pd.concat(session_dfs, ignore_index=True)
        session_df.drop_duplicates(inplace=True)
        session_df = session_df.rename(columns={"session_id": "ses"})
        session_df["ses"] = session_df["ses"].apply(lambda x: x.replace("ses-", ""))
    else:
        session_df = pd.DataFrame(columns=["ses", "sub", "ses_acq_time"])
        session_df
    return session_df


def create_scan_files(group_df: pd.DataFrame, bids_dir: str) -> None:
    """Creates and saves a `_scans.tsv` file per subject/session in the BIDS directory.

    Generates a `_scans.tsv` for each group (by subject and session) in `group_df`.
    The file contains two columns: `filename` (relative path to the NIRS file) and
    `acq_time` (acquisition time).

    Args:
        group_df: A grouped DataFrame for a particular subject and session. Must
            include at least the ``bids_name`` and ``acq_time`` columns.
        bids_dir: The path to the BIDS directory where `_scans.tsv` will be saved.
    """

    sub, ses = group_df.name
    tsv_df = group_df[["bids_name", "acq_time"]]
    tsv_df = tsv_df.rename(columns={"bids_name": "filename"})
    tsv_df["filename"] = "nirs/" + tsv_df["filename"]
    if ses != "Unknown":
        filename = "sub-" + str(sub) + "_ses-" + str(ses) + "_scans.tsv"
        path_to_save = os.path.join(
            bids_dir, "sub-" + str(sub), "ses-" + str(ses), filename
        )
    else:
        filename = "sub-" + str(sub) + "_scans.tsv"
        path_to_save = os.path.join(bids_dir, "sub-" + str(sub), filename)
    tsv_df.to_csv(path_to_save, sep="\t", index=False)


def create_session_files(group_df: pd.DataFrame, bids_dir: str) -> None:
    """Creates and saves a `_sessions.tsv` file per subject in the BIDS directory.

    Generates a `_sessions.tsv` for each subject in `group_df`. The file contains
    two columns: `ses` (session identifier) and `acq_time` (session acquisition time).

    Args:
        group_df: A grouped DataFrame for a particular subject. Must include at
            least the ``ses`` and ``ses_acq_time`` columns.
        bids_dir: The path to the BIDS directory where `_sessions.tsv` will be
            saved.
    """
    sub = group_df.name
    tsv_df = group_df[["ses", "ses_acq_time"]]
    if not tsv_df["ses"].isna().all():
        tsv_df["ses"] = "ses-" + tsv_df["ses"]
        tsv_df = tsv_df.rename(
            columns={"ses_acq_time": "acq_time", "ses": "session_id"}
        )
        tsv_df["acq_time"] = tsv_df["acq_time"].fillna("n/a")
        tsv_df.drop_duplicates(subset="session_id", inplace=True)
        if not pd.isna(tsv_df["session_id"]).any():
            filename = "sub-" + str(sub) + "_sessions.tsv"
            path_to_save = os.path.join(bids_dir, "sub-" + str(sub), filename)
            tsv_df.to_csv(path_to_save, sep="\t", index=False)


def create_data_description(
    dataset_path: str, bids_dir: str, extra_meta_data: Optional[str] = None
) -> None:
    """Creates or updates `dataset_description.json` in the BIDS directory.

    Checks for an existing `dataset_description.json` in `dataset_path` and
    updates it with relevant metadata. Additional metadata from `extra_meta_data`
    is merged in if provided. Missing required keys are filled with default values.

    Args:
        dataset_path: The path to the dataset where `dataset_description.json`
            is located.
        bids_dir: The path to the BIDS directory where the updated
            `dataset_description.json` will be saved.
        extra_meta_data: Path to a JSON file with additional metadata to include.
            Defaults to ``None``.
    """

    data_des = {
        "Name": os.path.basename(dataset_path),
        "BIDSVersion": "1.10.0",
        "License": "CC0",
        "DatasetType": "raw",
        "Authors": ["Enter author names here"],
        "Acknowledgements": (
            "Enter acknowledgements here (e.g., funding sources, institutions)."
        ),
        "HowToAcknowledge": (
            "Provide details on how to cite or acknowledge this dataset."
        ),
        "DatasetDOI": "Enter DOI here if available.",
        "Funding": ["Enter funding details here, if applicable."],
        "EthicsApprovals": ["Enter ethics approval details here, if applicable."],
        "ReferencesAndLinks": [
            "Enter references or related links here, if applicable."
        ],
    }

    result = find_files_with_pattern(dataset_path, "dataset_description.json")
    data_description_keys = data_des.keys()
    # data_des = {}
    if extra_meta_data is not None:
        with open(extra_meta_data, "r") as file:
            data = json.load(file)
            data = {key: value for key, value in data.items() if value != ""}
            data_des.update(
                {key: data[key] for key in data_description_keys if key in data}
            )
    if len(result) != 0:
        with open(result[0], "r") as file:
            data_des.update(json.load(file))
            data_des = {key: value for key, value in data_des.items() if value != ""}

    with open(os.path.join(bids_dir, "dataset_description.json"), "w") as json_file:
        json.dump(data_des, json_file, indent=4)


def check_coord_files(bids_dir: str) -> None:
    """Checks for and updates ``*_coordsystem.json`` files in a BIDS directory.

    Searches for files matching ``*_coordsystem.json`` in `bids_dir`. If
    ``NIRSCoordinateSystem`` is empty, it is set to ``"Other"``. Coordinate
    units are normalized to SI abbreviations (e.g. ``"millimeter"`` → ``"mm"``);
    unrecognized units are set to ``"n/a"``.

    Args:
        bids_dir: The path to the BIDS directory where ``*_coordsystem.json``
            files are located.
    """

    unit_mapping = {
        "millimeter": "mm",
        "centimeter": "cm",
        "meter": "m",
    }
    valid_units = {"m", "mm", "cm"}

    results = find_files_with_pattern(bids_dir, "*_coordsystem.json")
    for coord_file in results:
        with open(coord_file, "r") as file:
            data = json.load(file)

        if data["NIRSCoordinateSystem"] == "":
            data["NIRSCoordinateSystem"] = "Other"

        # Check if NIRSCoordinateUnits is valid
        units = data.get("NIRSCoordinateUnits")

        if units in unit_mapping:
            data["NIRSCoordinateUnits"] = unit_mapping[units]

        elif units not in valid_units:
            data["NIRSCoordinateUnits"] = "n/a"

        with open(coord_file, "w") as json_file:
            json.dump(data, json_file, indent=4)


def create_participants_tsv(
    bids_dir: str, mapping_df: pd.DataFrame, fields: Optional[List[str]] = None
) -> None:
    """Creates a `participants.tsv` file in a BIDS-compliant directory.

    This function generates a `participants.tsv` file based on the provided
    `mapping_df`, which must include at least a "sub" column (subject identifier). It
    ensures that the specified fields are present in the output, initializing any
    missing fields with `None`.

    Args:
        bids_dir (str): Path to the BIDS directory.
        mapping_df (pd.DataFrame): A DataFrame containing subject metadata,
                                   including a "sub" column.
        fields (List[str], optional): A list of additional participant-level fields
                                      to include in the TSV. Defaults to
                                      ["species", "age", "sex", "handedness"].

    Returns:
        None: Writes `participants.tsv` to the specified BIDS directory.
    """
    participants_df = "sub-" + mapping_df[["sub"]]
    participants_df.drop_duplicates(inplace=True)
    participants_df = participants_df.rename(columns={"sub": "participant_id"})

    if fields is None:
        fields = ["species", "age", "sex", "handedness"]

    for c in fields:
        participants_df[c] = "n/a"

    participants_df.to_csv(
        os.path.join(bids_dir, "participants.tsv"), sep="\t", index=False
    )


def create_participants_json(bids_dir: str, fields: Optional[List[str]] = None) -> None:
    """Creates or updates a `participants.json` file in a BIDS-compliant directory.

    If no custom fields are provided, this function uses a default schema based on
    BIDS recommendations. The output describes participant-level metadata for each
    field in the corresponding `participants.tsv` file.

    Args:
        bids_dir (str): Path to the BIDS directory.
        fields (List[str], optional): List of fields to include in the JSON schema.
                                      If None, a default set is used.

    Returns:
        None: Writes `participants.json` to the specified BIDS directory.
    """

    if fields is None:
        json_template = {
            "species": {
                "Description": "species of the participant",
                "Levels": {
                    "homo sapiens": "a binomial species name from the NCBI Taxonomy"
                },
            },
            "age": {"Description": "age of the participant", "Units": "year"},
            "sex": {
                "Description": "sex of the participant as reported by the participant",
                "Levels": {"M": "male", "F": "female"},
            },
            "handedness": {
                "Description": (
                    "handedness of the participant as reported by the participant"
                ),
                "Levels": {"left": "left", "right": "right"},
            },
        }

    else:
        json_template = dict.fromkeys(fields)

    with open(os.path.join(bids_dir, "participants.json"), "w") as json_file:
        json.dump(json_template, json_file, indent=4)


def create_participants_files(
    bids_dir: str,
    mapping_df: Optional[pd.DataFrame] = None,
    participants_tsv_path: Optional[str] = None,
    participants_json_path: Optional[str] = None,
    fields: Optional[List[str]] = None
):
    """Creates or updates the BIDS `participants.tsv` and `participants.json` files.

    If a `participants.tsv` file already exists and contains data, it is cleaned
    and standardized:

    - Ensures the first column is named ``participant_id``.
    - Prepends ``"sub-"`` to subject IDs if missing.
    - Sorts participants by ID.

    The corresponding `participants.json` is also updated or created based on
    the TSV's columns. If no valid `participants.tsv` is found, falls back to
    generating new files from `mapping_df`.

    Args:
        bids_dir: Path to the BIDS directory where output files will be written.
        mapping_df: Used to create `participants.tsv` if no existing file is
            found.
        participants_tsv_path: Path to an existing `participants.tsv` file.
        participants_json_path: Path to an existing `participants.json` file.
        fields: Fields to include in the schema. If ``None``, a default set is
            used.
    """

    if os.path.exists(participants_tsv_path):
        if str(participants_tsv_path).endswith(".tsv"):
            p_tsv = read_events_from_tsv(participants_tsv_path)
        elif str(participants_tsv_path).endswith(".xlsx"):
            p_tsv = pd.read_excel(participants_tsv_path, engine="openpyxl")
        else:
            p_tsv = pd.read_csv(participants_tsv_path)

        if len(p_tsv) > 0:
            sub_field = p_tsv.columns[0]

            if sub_field != "participant_id":
                p_tsv.columns = ["participant_id"] + list(p_tsv.columns[1:])
                p_tsv["participant_id"] = p_tsv["participant_id"].astype(str)
                p_tsv.drop_duplicates(subset=["participant_id"], inplace=True)

            sample_sub = p_tsv["participant_id"].iloc[0]

            if not sample_sub.startswith("sub-"):
                p_tsv["sort_key"] = p_tsv["participant_id"].map(
                    lambda val: (0, int(val)) if val.isdigit() else (1, val)
                )
                p_tsv = p_tsv.sort_values(by="sort_key").drop(columns=["sort_key"])
                p_tsv["participant_id"] = "sub-" + p_tsv["participant_id"].astype(str)
            else:
                p_tsv["participant_id"] = p_tsv["participant_id"].str.replace(
                    "sub-", ""
                )
                p_tsv["sort_key"] = p_tsv["participant_id"].map(
                    lambda val: (0, int(val)) if val.isdigit() else (1, val)
                )
                p_tsv = p_tsv.sort_values(by="sort_key").drop(columns=["sort_key"])
                p_tsv["participant_id"] = "sub-" + p_tsv["participant_id"].astype(str)

            p_tsv.to_csv(
                os.path.join(bids_dir, "participants.tsv"), sep="\t", index=False
            )

            if participants_json_path is not None:
                if os.path.exists(participants_json_path):
                    with open(participants_json_path, "r") as file:
                        participants_json = json.load(file)
                        participants_json = {
                            {sub_field: "participant_id"}.get(key, key): value
                            for key, value in participants_json.items()
                        }
                    with open(os.path.join(bids_dir, "participants.json"), "w") as file:
                        json.dump(participants_json, file)
                else:
                    create_participants_json(
                        bids_dir=bids_dir, fields=p_tsv.columns.tolist()
                    )
            else:
                create_participants_json(
                    bids_dir=bids_dir, fields=p_tsv.columns.tolist()
                )
        else:
            create_participants_tsv(
                bids_dir=bids_dir, mapping_df=mapping_df, fields=fields
            )
            create_participants_json(bids_dir=bids_dir, fields=fields)
    else:
        print("No valid participants.tsv file found. Creating default files.")
        create_participants_tsv(bids_dir=bids_dir, mapping_df=mapping_df, fields=fields)
        create_participants_json(bids_dir=bids_dir, fields=fields)


def edit_events(row: pd.Series, bids_dir: str) -> None:
    """Edits a BIDS ``_events.tsv`` file in place based on values in `row`.

    Updates the ``"duration"`` and/or ``"trial_type"`` columns of the
    corresponding ``_events.tsv`` file.

    Args:
        row: A row from the mapping DataFrame with the following keys:

            - ``"cond"``: Serialised list of keys for mapping trial types, or
              ``None``.
            - ``"cond_match"``: Serialised list of replacement values, or
              ``None``.
            - ``"duration"``: Duration to write into every event, or ``None``.
            - ``"bids_name"``: Base name of the BIDS file used to locate the
              ``_events.tsv``.
            - ``"parent_path"``: Relative path to the directory containing the
              ``_events.tsv``.

        bids_dir: The root directory of the BIDS dataset.
    """
    tsv_filename = row["bids_name"].replace("_nirs.snirf", "_events.tsv")
    event_path = os.path.join(bids_dir, row["parent_path"], tsv_filename)
    events_df = pd.read_csv(event_path, delimiter="\t")

    if row["cond"] or row["cond_match"] or row["duration"]:
        if not pd.isna(row["duration"]):
            events_df["duration"] = row["duration"]
        if not pd.isna(row["cond"]):
            keys = re.sub(r'[\[\]"]', "", row["cond"]).split(",")
            keys = [item.strip() for item in keys]
            values = re.sub(r'[\[\]"]', "", row["cond_match"]).split(",")
            values = [item.strip() for item in values]
            map_dict = dict(zip(keys, values))
            events_df["trial_type"] = (
                events_df["trial_type"].astype(str).replace(map_dict)
            )

        events_df.sort_values("onset").to_csv(event_path, sep="\t", index=False)
    else:
        events_df.sort_values("onset").to_csv(event_path, sep="\t", index=False)
    return


def sort_events(row: pd.Series, bids_dir: str) -> None:
    """Sorts the events in a BIDS ``_events.tsv`` file by onset time.

    Locates the corresponding ``_events.tsv`` file for the given row, reads it,
    sorts events by the ``"onset"`` column, and overwrites the original file.

    Args:
        row: A row from a BIDS file metadata DataFrame. Must include
            ``"bids_name"`` and ``"parent_path"`` keys.
        bids_dir: The root directory of the BIDS dataset.
    """
    tsv_filename = row["bids_name"].replace("_nirs.snirf", "_events.tsv")
    event_path = os.path.join(bids_dir, row["parent_path"], tsv_filename)
    events_df = pd.read_csv(event_path, delimiter="\t")
    events_df.sort_values("onset").to_csv(event_path, sep="\t", index=False)


def save_source(dataset_path: str, destination_path: str) -> None:
    """Copies the dataset into a ``sourcedata`` folder in `destination_path`.

    If a ``sourcedata`` subfolder already exists inside `dataset_path`, only
    that subfolder is copied. Otherwise the entire dataset is copied.

    Args:
        dataset_path: Path to the original dataset.
        destination_path: Directory where the ``sourcedata`` folder will be
            created.
    """
    source_folder = os.path.join(dataset_path, "sourcedata")

    if os.path.isdir(source_folder):
        shutil.copytree(source_folder, os.path.join(destination_path, "sourcedata"))
    else:
        shutil.copytree(dataset_path, os.path.join(destination_path, "sourcedata"))


def export_to_bids_optodes_tsv(
    tsv_filename, points: cdt.LabeledPoints, units="mm", float_format: str | None = None
):
    """Export to a bids-conform _optodes.tsv.

    Args:
        tsv_filename: Path to the output tsv file.
        points: LabeledPoints to save.
        units: coordinate units.
        float_format : Format string for floating point numbers.
    """

    cite("Gorgolewski2016")
    cite("Luke2025")


    # BIDS optodes_tsv
    points = points[
        (points.type == PointType.SOURCE) | (points.type == PointType.DETECTOR)
    ]

    points = points.pint.to(units).pint.dequantify()

    # else: types are optodes, fiducials, landmarks, electrodes
    with open(tsv_filename, 'w') as f:
        names = points.label.values
        types = [i.name.lower() for  i in points.type.values]
        header = "\t".join(["name", "type","x", "y", "z"])

        f.write(header + "\n")

        for n, t, p in zip(names, types, points.values):
            if float_format:
                ff = float_format
                f.write(
                    f"{n}\t{t}\t"
                    f"{format(p[0], ff)}\t{format(p[1], ff)}\t{format(p[2], ff)}\n"
                )
            else:
                f.write(f"{n}\t{t}\t{p[0]}\t{p[1]}\t{p[2]}\n")

    return


def load_from_bids_optodes_tsv(tsv_filename : Path | str) -> cdt.LabeledPoints:
    """Load optodes and landmarks from a BIDS *_optodes.tsv and its *_coordsystem.json.

    The coordinate system name, units, and anatomical landmarks are read from the
    accompanying ``*_coordsystem.json`` file.  The JSON is expected at the same path
    as the TSV, with ``_optodes.tsv`` replaced by ``_coordsystem.json``.

    Args:
        tsv_filename: Path to the BIDS ``*_optodes.tsv`` file.

    Returns:
        LabeledPoints with sources, detectors, and (if present) landmarks.
    """

    cite("Gorgolewski2016")
    cite("Luke2025")


    tsv_filename = Path(tsv_filename)

    if str(tsv_filename).endswith("_optodes.tsv"):
        coordsystem_filename = Path(
            str(tsv_filename).replace("_optodes.tsv", "_coordsystem.json")
        )
    else:
        coordsystem_filename = None

    crs = "unknown"
    units = "mm"
    landmark_coords = {}
    landmark_units = None

    if (coordsystem_filename is not None) and (coordsystem_filename.exists()):
        with open(coordsystem_filename) as f:
            cs = json.load(f)
        crs = cs.get("NIRSCoordinateSystem", crs)
        units = cs.get("NIRSCoordinateUnits", units)
        landmark_coords = cs.get("AnatomicalLandmarkCoordinates", {})
        landmark_units = cs.get("AnatomicalLandmarkCoordinateUnits", units)

    df = pd.read_csv(tsv_filename, sep="\t")

    type_map = {
        "source": PointType.SOURCE,
        "detector": PointType.DETECTOR,
    }

    labels = list(df["name"].values)
    types = [type_map.get(t, PointType.UNKNOWN) for t in df["type"].values]
    coordinates = df[["x", "y", "z"]].values.tolist()

    for name, xyz in landmark_coords.items():
        labels.append(name)
        types.append(PointType.LANDMARK)
        coordinates.append(xyz)

    if landmark_units is not None and landmark_units != units:
        n_optodes = len(df)
        optode_arr = build_labeled_points(
            coordinates[:n_optodes], crs=crs, units=units,
            labels=labels[:n_optodes], types=types[:n_optodes],
        )
        lm_arr = build_labeled_points(
            coordinates[n_optodes:], crs=crs, units=landmark_units,
            labels=labels[n_optodes:], types=types[n_optodes:],
        )
        lm_arr = lm_arr.pint.to(units)
        return xr.concat([optode_arr, lm_arr], dim="label")

    return build_labeled_points(
        np.array(coordinates), crs=crs, units=units, labels=labels, types=types
    )
