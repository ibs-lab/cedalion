#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
command line tool 
- given a directory read in all the snirf file names 
- use the information to populate a csv file that contains as much snirf info as possible
- use then fills in any missing/incorrect information 

1. take in a directory path - use input 
2. os walks through the directory and finds all snirf files
3. checks the names for any BIDS fields that are relevant - check both parent folder and file name 
4. create a dictionary for each file with the BIDS fields filled in - if not present in the current filename use None 
5. creates a tsv file from a template that has all the desired fields that would then be pulled to generate the BIDS dataset
6. saves this tsv file in the dataset directory 

@author: lauracarlton
"""
import os 
import pandas as pd 
from cedalion.io.bids import check_for_bids_field 

column_names = ["current_name",
                "sub",
                "ses",
                "task",
                "run",
                "acq" ]


snirf2bids_mapping_df = pd.DataFrame(columns=column_names)


dataset_path = input('Please enter the path to your dataset: ')
# dataset_path = '/Users/lauracarlton/Documents/DATA/MAFC_raw'

#%% IDENTIFY ALL SNIRF FILES IN THE DIRECTORY AND THEIR PATH 

file_list = []
for dirpath, dirnames, filenames in os.walk(dataset_path):
    
    for filename in filenames:
        
       if filename.endswith('.snirf'):
            # Get the full path of the file
            relative_path = os.path.relpath(dirpath, dataset_path)
            
            # get each part of the path 
            parent_folders = relative_path.split(os.sep)
            
            # including the filename
            filename_without_ext = os.path.splitext(filename)[0]
            parent_folders.append(filename_without_ext)
            
            # add to the list of file paths
            file_list.append(parent_folders)


#%% CHECK EACH FILE TO GATHER INFO TO POPULATE THE MAPPING_DF


for path_parts in file_list:
    
    # need to check for sub
    subject = check_for_bids_field(path_parts, 'sub')

    # check for session 
    ses = check_for_bids_field(path_parts, 'ses')

    # check for run 
    run = check_for_bids_field(path_parts, 'run')
    
    # check for task
    task = check_for_bids_field(path_parts, 'task')

    # check for acq
    acq = check_for_bids_field(path_parts, 'acq')
    
    bids_dict = {"current_name": "/".join(path_parts),
                 "sub": subject,
                 "ses": ses, 
                 "run": run, 
                 "task": task, 
                 "acq": acq
                 }
    snirf2bids_mapping_df = pd.concat([snirf2bids_mapping_df, pd.DataFrame([bids_dict])], ignore_index=True)



snirf2bids_mapping_df.to_csv(os.path.join(dataset_path, 'snirf2BIDS_mapping.csv'), index=None)
    
            









