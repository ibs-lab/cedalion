from pathlib import Path
import os
import pandas as pd
import shutil

import cedalion
import cedalion.io
import json


def read_events_from_tsv(fname: str | Path):
    return pd.read_csv(fname, delimiter="\t")


def check_for_bids_field(path_parts: list, 
                         field: str):
    
    field_parts = [part for part in path_parts if field in part]
    if len(field_parts) == 0:
        value = None
    else:
        find_value = field_parts[-1].split('_') # assume the lowest directory level supersedes any higher directory level ? not sure if we should assume this 
        value = [vals for vals in find_value if field in vals][0]
        value = value.split('-')[1]
        
    return value

def find_files_with_pattern(start_dir, pattern):
    start_path = Path(start_dir)
    return [str(file) for file in start_path.rglob(pattern)]

def create_bids_standard_filenames(row):
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


def copy_rename_snirf(row, dataset_path, bids_dir):

    # Path of the source file (the file to be moved and renamed)
    source_file = os.path.join(dataset_path, row["current_name"] + ".snirf")

    recording = cedalion.io.read_snirf(source_file)[0]
    if len(recording.stim) != 0:
        destination_folder = os.path.join(bids_dir, row["parent_path"])
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)
        destination_file = os.path.join(destination_folder, row["bids_name"])
        shutil.copy(source_file, destination_file)
        return "copied"
    else:
        return "removed"


def check_coord_files(bids_dir):
    results = find_files_with_pattern(bids_dir, "*_coordsystem.json")
    for coord_file in results:
        with open(coord_file, 'r') as file:
            data = json.load(file) 
            if data["NIRSCoordinateSystem"] == "":
                data["NIRSCoordinateSystem"] = "Other"
                with open(coord_file, 'w') as json_file:
                    json.dump(data, json_file, indent=4) 

def create_scan_files(group_df, bids_dir):
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


def create_data_description(dataset_path, bids_dir, extra_meta_data = None):
    result = find_files_with_pattern(dataset_path, "dataset_description.json")
    data_description_keys = ["Name", "DatasetType", "EthicsApprovals", "ReferencesAndLinks", "Funding"]
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
