"""
This script automates the conversion of an fNIRS dataset into a BIDS-compliant format.

Main Functions:
---------------
1. Parse a CSV mapping file (`mapping_csv`) to retrieve file relationships and metadata.
2. Organize and rename SNIRF files into a BIDS-compatible structure.
3. Generate required BIDS files such as `scans.tsv`, coordinate system files, and `dataset_description.json`.
4. Handle optional extra metadata for inclusion in the BIDS dataset.

Arguments:
----------
- mapping_csv: str
    The path to a CSV file containing the mapping of SNIRF files to BIDS-compatible filenames.
- dataset_path: str
    The directory containing the input dataset.
- --extra_meta_data: str (optional)
    A JSON file containing additional metadata to be added to the BIDS dataset.

Key Steps:
----------
1. Create the `bids` directory within the specified dataset path if it does not exist.
2. Parse and process existing `*_scans.tsv` files to merge acquisition times into the mapping.
3. Generate standardized BIDS filenames for the SNIRF files using `create_bids_standard_filenames`.
4. Rename and copy SNIRF files into the appropriate BIDS directory structure.
5. Validate and recursively populate the BIDS structure using `snirf2bids`.
7. Add optional metadata from the `extra_meta_data` argument to the dataset description.

Example Usage:
--------------
python create_bids.py mapping.csv dataset_directory --extra_meta_data extra_metadata.json
"""

import os 
import pandas as pd 
from cedalion.io.bids import create_bids_standard_filenames, copy_rename_snirf, create_scan_files, find_files_with_pattern, create_data_description, check_coord_files
import snirf2bids as s2b
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Create BIDS dataset.")

# Define expected arguments
parser.add_argument("mapping_csv", type=str, help="The output of dataset parser")
parser.add_argument("dataset_path", type=str, help="your dataset directory")
parser.add_argument("--extra_meta_data", type=str, help="your extra metadata")
# Parse the arguments
args = parser.parse_args()

snirf2bids_mapping_df_path = args.mapping_csv

dataset_path = args.dataset_path


bids_dir = os.path.join(dataset_path, "bids")
if not os.path.exists(bids_dir):
    os.makedirs(bids_dir)

snirf2bids_mapping_df = pd.read_csv(snirf2bids_mapping_df_path, dtype=str)
snirf2bids_mapping_df["record_name"] = snirf2bids_mapping_df["current_name"].apply(lambda x: os.path.basename(x))

scan_paths = find_files_with_pattern(dataset_path, "*_scans.tsv")
scan_dfs = [pd.read_csv(file, sep='\t') for file in scan_paths]
if len(scan_dfs) != 0:
    scan_df = pd.concat(scan_dfs, ignore_index=True)
    scan_df.drop_duplicates(subset="filename", inplace=True)
    scan_df["filename"] = scan_df["filename"].apply(lambda x: str(os.path.basename(x)).replace(".snirf", ""))
    scan_df = scan_df.rename(columns={'filename': 'record_name'})

    snirf2bids_mapping_df = pd.merge(snirf2bids_mapping_df, scan_df, on="record_name", how="left")
else:
    snirf2bids_mapping_df["acq_time"] = None

snirf2bids_mapping_df[["bids_name", "parent_path"]] = snirf2bids_mapping_df.apply(create_bids_standard_filenames, axis=1, result_type='expand')

snirf2bids_mapping_df["status"] = snirf2bids_mapping_df.apply(copy_rename_snirf, axis=1, args=(dataset_path, bids_dir))

s2b.snirf2bids_recurse(bids_dir)

scan_df = snirf2bids_mapping_df[snirf2bids_mapping_df['status'] != "removed"]
scan_df = scan_df[["sub", "ses", "bids_name", "acq_time"]]
scan_df = scan_df.groupby(["sub", "ses"])
scan_df.apply(lambda group: create_scan_files(group, bids_dir))

extra_meta_data = args.extra_meta_data
create_data_description(dataset_path, bids_dir, extra_meta_data)

check_coord_files(bids_dir)

