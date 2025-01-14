"""This script automates the conversion of an fNIRS dataset into a BIDS-compliant format.

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
3. Parse and process existing `*_sessions.tsv` files to merge session acquisition times into the mapping.
4. Generate standardized BIDS filenames for the SNIRF files using `create_bids_standard_filenames`.
5. Rename and copy SNIRF files into the appropriate BIDS directory structure.
6. Validate and recursively populate the BIDS structure using `snirf2bids`.
7. Add optional metadata from the `extra_meta_data` argument to the dataset description.

Example Usage:
--------------
python create_bids.py mapping.csv dataset_directory --extra_meta_data extra_metadata.json
"""

import os
import pandas as pd
from cedalion.io import bids
import snirf2bids as s2b
import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description="Create BIDS dataset.")

parser.add_argument("mapping_csv", type=str, help="The output of dataset parser")
parser.add_argument("dataset_path", type=str, help="your dataset directory")
parser.add_argument("--extra_meta_data", type=str, help="your extra metadata")

# Parse the arguments
args = parser.parse_args()

mapping_df_path = args.mapping_csv
dataset_path = args.dataset_path
extra_meta_data = args.extra_meta_data

# Create a bids directory to copy and create files there
bids_dir = os.path.join(dataset_path, "bids")
if not os.path.exists(bids_dir):
    os.makedirs(bids_dir)

# Read the mapping csv file
mapping_df = pd.read_csv(mapping_df_path, dtype=str)

# extract the original snirf filenames and create a column
mapping_df["filename_org"] = mapping_df["current_name"].apply(
    lambda x: os.path.basename(x))

# Search for all acq_time info that are available in scan files in
# the original dataset directory
scan_df = bids.search_for_acq_time(dataset_path)
mapping_df = pd.merge(mapping_df, scan_df, on="filename_org", how="left")

# Search for all sessions' acq_time info that are available in session files in
# the original dataset directory
session_df = bids.search_for_sessions_acq_time(dataset_path)
mapping_df = pd.merge(mapping_df, session_df, on="sub", how="left")

# Create filenames that follow naming convention in bids structure
mapping_df[["bids_name", "parent_path"]] = mapping_df.apply(
    bids.create_bids_standard_filenames, axis=1, result_type='expand')

# Copy and rename snirf files according to bids folder structure
mapping_df.apply(bids.copy_rename_snirf, axis=1, args=(dataset_path, bids_dir))

# Create json and tsv files using snirf2bids python package
s2b.snirf2bids_recurse(bids_dir)

# Create scan files
scan_df = mapping_df[["sub", "ses", "bids_name", "acq_time"]]
scan_df = scan_df.groupby(["sub", "ses"])
scan_df.apply(lambda group: bids.create_scan_files(group, bids_dir))

# Create session files
session_df = mapping_df[["sub", "ses", "ses_acq_time"]]
session_df = session_df.groupby(["sub"])
session_df.apply(lambda group: bids.create_session_files(group, bids_dir))

# Create dataset_description.json file
bids.create_data_description(dataset_path, bids_dir, extra_meta_data)

# Correct mistakes in Coord json files
bids.check_coord_files(bids_dir)

