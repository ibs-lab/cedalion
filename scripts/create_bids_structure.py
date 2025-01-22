"""Script to convert an fNIRS dataset into a BIDS-compliant format.

This script processes an fNIRS dataset and organizes it according to the Brain Imaging Data Structure (BIDS) standard. 

Steps included:
1. Parse and validate a mapping CSV file to define dataset structure and extract metadata.
2. Rename and organize SNIRF files following BIDS naming conventions.
3. Generate essential BIDS files:
   - *_coordsystem.json
   - *_optodes.json, *_optodes.tsv
   - *_channels.tsv
   - *_events.json, *_events.tsv
   - *_nirs.json
4. Incorporate scan and session metadata from existing *_scan.tsv and *_session.tsv files.
5. Create participants.json and participants.tsv with predefined templates and metadata.
6. Generate dataset_description.json with user-provided or default metadata.
7. Ensure compliance with BIDS standards, including populating required fields like `NIRSCoordinateSystem`.
8. Allow edits to *_events.tsv files based on the mapping CSV (e.g., updating duration or trial_type).
9. Optionally preserve the original dataset in a `sourcedata` directory.

Usage:
------
Run the script with the following arguments:
- `mapping_csv` (str): Path to the mapping CSV file.
- `dataset_path` (str): Path to the original dataset.
- `destination_path` (str): Path to save the BIDS-compliant dataset.
- `--extra_meta_data` (str, optional): Path to additional metadata for dataset_description.json.

Example:
--------
$ python script_name.py mapping.csv /path/to/dataset /path/to/bids --extra_meta_data meta.json
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
parser.add_argument("destination_path", type=str, help="your destination directory to save your dataset in BIDs format")
parser.add_argument("--extra_meta_data", type=str, help="your extra metadata")

# Parse the arguments
args = parser.parse_args()

mapping_df_path = args.mapping_csv
dataset_path = args.dataset_path
destination_path = args.destination_path
extra_meta_data = args.extra_meta_data

# The `mapping_df_path` is the path to your mapping CSV file.
# If you don’t already have a mapping CSV file, you can generate one using the 
# scripts/parse_dataset.py script. After generating the CSV file, 
# you might need to manually edit it to include additional information 
# or make adjustments as required.
# A valid mapping CSV must include all SNIRF files in your dataset, 
# along with the following details for each file:
# - Subject: The identifier for the participant.
# - Session (optional): The session identifier, if applicable.
# - Task: The task name or label.
# - Run (optional): The run number, if applicable.
mapping_df = pd.read_csv(mapping_df_path, dtype=str)

# The mapping table serves as a key component for organizing and 
# processing your dataset. 
# The `ses`, `run`, and `acq` columns are optional and can be set to 
# None if not applicable. The `current_name` column contains the path 
# to the SNIRF files in your dataset. Since we will need the base filenames 
# of the SNIRF files for further processing, an additional column will 
# be created to store just the base filename.
mapping_df["filename_org"] = mapping_df["current_name"].apply(
    lambda x: os.path.basename(x))

# To ensure no important information (e.g., acquisition time) from 
# the original dataset is lost, we will:
# - Search Subdirectories: Traverse through all subdirectories within the dataset.
# - Locate Existing Scan Files: Search for all *_scan.tsv files in the dataset.
# - Integrate into Mapping Table: Extract the relevant information from these 
# files and add it to our mapping table.
# This approach ensures that any details, such as acquisition time, are retained 
# and incorporated into the BIDS-compliant structure.
scan_df = bids.search_for_acq_time(dataset_path)
mapping_df = pd.merge(mapping_df, scan_df, on="filename_org", how="left")

# Similar to *_scan.tsv files, we search for *_session.tsv files in the dataset path 
# to capture additional session-level metadata, such as acquisition times. 
# Any relevant information from these files is added to the mapping table to 
# ensure all session details are preserved.
session_df = bids.search_for_sessions_acq_time(dataset_path)
mapping_df = pd.merge(mapping_df, session_df, on=["sub", "ses"], how="left")

# The goal of this section is to rename the SNIRF files according to the BIDS naming 
# convention and place them in the appropriate directory under `destination_path`, 
# following the BIDS folder structure.
# Steps:
# 1. Generate New Filenames: Create BIDS-compliant filenames for all SNIRF records.
# 2. Determine File Locations: Identify the appropriate locations for these files within 
# the BIDS folder hierarchy.
#
# This process ensures that the dataset adheres to BIDS standards for organization and naming.
mapping_df[["bids_name", "parent_path"]] = mapping_df.apply(
    bids.create_bids_standard_filenames, axis=1, result_type='expand')

# To facilitate proper organization:
# - `parent_path`: Added to the mapping dataframe to define the location of each SNIRF 
# file within `destination_path`.
# - `bids_name`: Specifies the new BIDS-compliant name for each file.
# In the following sections, we will rename all files to their corresponding `bids_name` 
# and copy them to their designated parent_path.
_ = mapping_df.apply(bids.copy_rename_snirf, axis=1, args=(dataset_path, destination_path))

# In this step, we utilize the snirf2bids Python package to generate the necessary .tsv 
# and .json files for the BIDS structure.
# For every record, the following files will be created:
# 1. _coordsystem.json
# 2. _optodes.json
# 3. _optodes.tsv
# 4. *_channels.tsv
# 5. *_events.json
# 6. *_events.tsv
# 7. *_nirs.json
# These files are essential for ensuring the dataset adheres to BIDS standards.
s2b.snirf2bids_recurse(destination_path)

# Now, we proceed to create scan files for all subjects and sessions. 
# Previously, we searched the original dataset path for any provided scan information, 
# which will now be incorporated into the BIDS structure.
scan_df = mapping_df[["sub", "ses", "bids_name", "acq_time"]]
scan_df['ses'].fillna("Unknown", inplace=True)
scan_df = scan_df.groupby(["sub", "ses"])
scan_df.apply(lambda group: bids.create_scan_files(group, destination_path))

# The next step is to create session files for all subjects. As with the scan files, 
# we previously searched the original dataset path for any session information, 
# which will now be used to create the corresponding BIDS session files.
session_df = mapping_df[["sub", "ses", "ses_acq_time"]]
session_df = session_df.groupby(["sub"])
session_df.apply(lambda group: bids.create_session_files(group, destination_path))

# In this step, we gather all available participant information from the original dataset. 
# If any participant details are provided, they will be incorporated into the BIDS structure.
# Additionally, we create a template for the participants.json file with predefined 
# columns, including:
# - species
# - age
# - sex
# - handedness
# Each of these fields will include descriptive templates to ensure consistency in 
# the BIDS-compliant structure.
bids.create_participants_tsv(dataset_path, destination_path, mapping_df)
bids.create_participants_json(dataset_path, destination_path)

# To create the dataset_description.json file, we follow these steps:
# 1. Search for an existing dataset_description.json in the dataset path and retain 
# the provided information.
# 2. If extra_meta_data_path is specified, add the additional metadata about the dataset.
# 3. If neither dataset_description.json nor extra metadata is provided, use the basename 
# of the dataset directory as the dataset name and set the BIDS version to '1.10.0'.
bids.create_data_description(dataset_path, destination_path, extra_meta_data)

# Since an empty string is not allowed for the `NIRSCoordinateSystem` key in 
# the *_coordsystem.json file, we will populate it with "Other" to ensure BIDS compliance.
bids.check_coord_files(destination_path)

# To allow editing of the `duration` or `trial_type` columns in the *_events.tsv files, 
# the mapping CSV file must include the following extra columns:
# 1. `duration`: Specifies the new duration for each SNIRF file that needs editing.
# 2. cond and cond_match:
#     - `cond`: A list of keys.
#     - `cond_match`: A list of corresponding values.  
#     These two columns will be used to create a dictionary that maps the trial_type column.
_ = mapping_df.apply(bids.edit_events, axis=1, args=(destination_path))

# Finally there is this possiblity to keep your original data under 
# sourcedata directory at your `destination_path`.
bids.save_source(dataset_path, destination_path)
