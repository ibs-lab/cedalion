import os
import shutil
import json
import re
from pathlib import Path
from typing import List, Tuple, Optional

import pandas as pd


def check_for_bids_field(path_parts: list,
                         field: str):
    field_parts = [part for part in path_parts if field in part]
    if len(field_parts) == 0:
        value = None
    else:
        find_value = field_parts[-1].split('_') 
        value = [vals for vals in find_value if field in vals][0]
        value = value.split('-')[1]
    return value

def read_events_from_tsv(fname: str | Path) -> pd.DataFrame:
    return pd.read_csv(fname, delimiter="\t")

def find_files_with_pattern(start_dir: str | Path, pattern: str) -> List[str]:
    """Recursively finds all files in the specified directory (and subdirectories) that match the given pattern.

    Parameters:
    -----------
    start_dir : str | Path
        The directory to start the search from.

    pattern : str
        The pattern to match filenames against.

    Returns:
    --------
    List[str]
        A list of file paths (as strings) of all files that match the pattern.
    """
    start_path = Path(start_dir)
    return [str(file) for file in start_path.rglob(pattern)]

def create_bids_standard_filenames(row: pd.Series) -> Tuple[str, str]:
    """Generates a BIDS compliant file name and its parent directory path based on the information in the given row.

    This function constructs a filename and directory path following the BIDS naming convention for a specific subject, session, task, 
    acquisition, and run based on the provided DataFrame row. The final filename will include "_nirs.snirf" as the extension, and 
    the directory path will be created under a "nirs" directory.

    Parameters:
    -----------
    row : pd.Series
        A row of a Pandas DataFrame containing the following potential columns:
        - "sub" : The subject identifier (e.g., "01")
        - "ses" : The session identifier (e.g., "01"), can be NaN if not available
        - "task" : The task name or identifier (e.g., "rest")
        - "acq" : The acquisition identifier (e.g., "01"), can be NaN if not available
        - "run" : The run identifier (e.g., "1"), can be NaN if not available

    Returns:
    --------
    Tuple[str, str]
        A tuple containing:
        1. The generated filename string based on the BIDS standard.
        2. The parent directory path where the file is expected to be located.
    """

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
    """Copies a `.snirf` file from the source directory, renaming it according to BIDS standards, and places it in the appropriate destination directory.

    This function takes the source file (in the `dataset_path`), renames it based on the information in the provided `row`, 
    and copies it to the target `bids_dir` directory, following the BIDS directory structure.

    Parameters:
    -----------
    row : pd.Series
        A row from a Pandas DataFrame containing the following columns:
        - "current_name" : The current name of the file (without the `.snirf` extension).
        - "parent_path" : The relative path within the BIDS structure where the file should be stored.
        - "bids_name" : The new BIDS-compliant name for the file.

    dataset_path : str
        The path to the directory containing the original `.snirf` file(s) to be copied.

    bids_dir : str
        The path to the root BIDS directory where the renamed file should be copied to.
    """

    source_file = os.path.join(dataset_path, row["current_name"] + ".snirf")
    destination_folder = os.path.join(bids_dir, row["parent_path"])
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    destination_file = os.path.join(destination_folder, row["bids_name"])
    shutil.copy(source_file, destination_file)


def search_for_acq_time(dataset_path: str) -> pd.DataFrame:
    """Searches for `_scans.tsv` files in the given dataset path, reads them into DataFrames, and processes them to extract the `filename` and `acq_time` columns.

    This function looks for all `_scans.tsv` files in the `dataset_path`, reads them into a DataFrame, and processes the `filename` 
    and `acq_time` columns. If the `acq_time` column does not exist in the merged DataFrame, it will be added with `None` values.
    If no `_scans.tsv` files are found, an empty DataFrame with the columns `filename_org` and `acq_time` is returned.

    Parameters:
    -----------
    dataset_path : str
        The path to the dataset where the `_scans.tsv` files are located.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the following columns:
        - `filename_org`: The original filename (without the `.snirf` extension) from the `_scans.tsv` files.
        - `acq_time`: The acquisition time for each scan. If the `acq_time` column does not exist in the original files, it will be filled with `None`.
    """

    scan_paths = find_files_with_pattern(dataset_path, "*_scans.tsv")
    scan_dfs = [pd.read_csv(file, delimiter="\t") for file in scan_paths]

    if len(scan_dfs) != 0:
        scan_df = pd.concat(scan_dfs, ignore_index=True)
        scan_df.drop_duplicates(subset="filename", inplace=True)

        scan_df["filename"] = scan_df["filename"].apply(
            lambda x: str(os.path.basename(x)).replace(".snirf", ""))
        scan_df = scan_df.rename(columns={'filename': 'filename_org'})
        if "acq_time" not in scan_df.columns:
            scan_df["acq_time"] = None
    else:
        scan_df = pd.DataFrame(columns=["filename_org", "acq_time"])
    return scan_df

def search_for_sessions_acq_time(dataset_path: str) -> pd.DataFrame:
    """Searches for `_sessions.tsv` files in the provided dataset path, reads them into DataFrames, and processes them to extract the `session_id`, `sub` (subject ID), and `ses_acq_time` (session acquisition time).

    This function looks for all `_sessions.tsv` files in the given `dataset_path`, reads them into DataFrames, and processes them 
    to extract the subject ID and session acquisition time. If the `acq_time` column does not exist in the input files, 
    it will be added with `None` values. Additionally, it extracts the subject ID from the filename using a regular expression.

    Parameters:
    -----------
    dataset_path : str
        The path to the dataset where the `_sessions.tsv` files are located.

    Returns:
    --------
    pd.DataFrame
        A DataFrame with the following columns:
        - `session_id`: The session identifier (extracted from the filenames).
        - `sub`: The subject ID extracted from the filename.
        - `ses_acq_time`: The session acquisition time for each session. If the `acq_time` column does not exist in the original files, it will be filled with `None`.
    """

    session_paths = find_files_with_pattern(dataset_path, "*_sessions.tsv")
    session_dfs = []
    for f in session_paths:
        match = re.search(r"(?i)\b(?:sub|subj|subject)[-_]?(\d+)\b", f)
        ses_df = pd.read_csv(f, delimiter="\t")
        ses_df['sub'] = match.group(1)
        session_dfs.append(ses_df)
        if "acq_time" in ses_df.columns:
            ses_df = ses_df.rename(columns={'acq_time': 'ses_acq_time'})
        else:
            ses_df["ses_acq_time"] = None

    if len(session_dfs) != 0:

        session_df = pd.concat(session_dfs, ignore_index=True)
        session_df.drop_duplicates(inplace=True)
    else:
        session_df = pd.DataFrame(columns=["session_id", "sub", "ses_acq_time"])
        session_df
    return session_df

def create_scan_files(group_df: pd.DataFrame, bids_dir: str) -> None:
    """Creates a `_scans.tsv` file for each subject (and session, if provided) from the provided DataFrame and saves it in the BIDS directory.

    This function generates a `_scans.tsv` file for each group of data (grouped by subject and session) in the `group_df` DataFrame. 
    The resulting file contains two columns: `filename` (with a relative path to the NIRS file) and `acq_time` (acquisition time). 
    The function saves this file in the appropriate directory within the BIDS dataset.

    Parameters:
    -----------
    group_df : pd.DataFrame
        A DataFrame containing a group of rows for a particular subject and session. 
        This DataFrame should include at least the `bids_name` and `acq_time` columns.

    bids_dir : str
        The path to the BIDS directory where the `_scans.tsv` file will be saved.

    Returns:
    --------
    None
        This function does not return anything. It saves the generated `_scans.tsv` file to the BIDS directory.
    """

    sub , ses = group_df.name
    tsv_df = group_df[["bids_name", "acq_time"]]
    tsv_df = tsv_df.rename(columns={'bids_name': 'filename'})
    tsv_df["filename"] = "nirs/" + tsv_df["filename"]
    if not pd.isna(ses):
        filename = "sub-"+str(sub) + "_ses-"+str(ses) + "_scans.tsv"
        path_to_save = os.path.join(bids_dir, "sub-"+str(sub), "ses-"+str(ses), filename)
    else:
        filename = "sub-"+str(sub) + "_scans.tsv"
        path_to_save = os.path.join(bids_dir, "sub-"+str(sub), filename)
    tsv_df.to_csv(path_to_save, sep='\t', index=False) 

def create_session_files(group_df: pd.DataFrame, bids_dir: str) -> None:
    """Creates a `_sessions.tsv` file for each subject from the provided DataFrame and saves it in the BIDS directory.

    This function generates a `_sessions.tsv` file for each subject in the `group_df` DataFrame. 
    The resulting file contains two columns: `ses` (session identifier) and `acq_time` (session acquisition time). 
    The function saves this file in the appropriate directory within the BIDS dataset.

    Parameters:
    -----------
    group_df : pd.DataFrame
        A DataFrame containing a group of rows for a particular subject. 
        This DataFrame should include at least the `ses` (session identifier) and `ses_acq_time` (session acquisition time) columns.

    bids_dir : str
        The path to the BIDS directory where the `_sessions.tsv` file will be saved.

    Returns:
    --------
    None
        This function does not return anything. It saves the generated `_sessions.tsv` file to the BIDS directory.
    """
    sub = group_df.name
    tsv_df = group_df[["ses", "ses_acq_time"]]
    tsv_df["ses"] = "ses-" + tsv_df["ses"]
    tsv_df = tsv_df.rename(columns={'ses_acq_time': 'acq_time'})
    if not pd.isna(tsv_df["ses"]).any():
        filename = "sub-"+str(sub) + "_sessions.tsv"
        path_to_save = os.path.join(bids_dir, "sub-"+str(sub), filename)
        tsv_df.to_csv(path_to_save, sep='\t', index=False)


def create_data_description(dataset_path: str, bids_dir: str, extra_meta_data: Optional[str] = None) -> None:
    """Creates or updates the `dataset_description.json` file in the specified BIDS directory.

    This function checks for an existing `dataset_description.json` file in the specified dataset path 
    and updates it with relevant metadata. It also adds any additional metadata from an optional external JSON file (`extra_meta_data`).
    If some required keys are missing, it will add them with default values.

    Parameters:
    -----------
    dataset_path : str
        The path to the dataset where the `dataset_description.json` file is located.

    bids_dir : str
        The path to the BIDS directory where the updated `dataset_description.json` file will be saved.

    extra_meta_data : Optional[str], default=None
        An optional path to a JSON file containing additional metadata to be included in the `dataset_description.json`. 
        If not provided, no extra metadata will be added.

    Returns:
    --------
    None
        This function does not return any value. It updates the `dataset_description.json` file in the BIDS directory.
    """

    result = find_files_with_pattern(dataset_path, "dataset_description.json")
    data_description_keys = ["Name", "DatasetType", "EthicsApprovals",
                             "ReferencesAndLinks", "Funding"]
    data_des = {}
    if extra_meta_data is not None:
        with open(extra_meta_data, 'r') as file:
            data = json.load(file)
            data = {key: value for key, value in data.items() if value != ''}
            data_des.update({key: data[key] for key in data_description_keys if key in data})
    if len(result) != 0:
        with open(result[0], 'r') as file:
            data_des.update(json.load(file))
            data_des = {key: value for key, value in data_des.items() if value != ''}

    if "Name" not in data_des:
        name = os.path.basename(dataset_path)
        data_des["Name"] = name
    if "BIDSVersion" not in data_des:
        data_des["BIDSVersion"] = '1.10.0'

    with open(os.path.join(bids_dir, "dataset_description.json"), 'w') as json_file:
        json.dump(data_des, json_file, indent=4)

def check_coord_files(bids_dir: str) -> None:
    """Checks for and updates *_coordsystem.json files in a BIDS directory.

    This function searches for files matching the pattern "*_coordsystem.json" within the specified BIDS directory. 
    If the "NIRSCoordinateSystem" field is empty, it updates the field with the value "Other" and writes the updated data back to the JSON file.

    Parameters:
    -----------
    bids_dir : str
        The path to the BIDS directory where *_coordsystem.json files are located.

    Returns:
    --------
    None
        This function does not return any value. It directly modifies the *_coordsystem.json files.
    """

    results = find_files_with_pattern(bids_dir, "*_coordsystem.json")
    for coord_file in results:
        with open(coord_file, 'r') as file:
            data = json.load(file) 
            if data["NIRSCoordinateSystem"] == "":
                data["NIRSCoordinateSystem"] = "Other"
                with open(coord_file, 'w') as json_file:
                    json.dump(data, json_file, indent=4)

