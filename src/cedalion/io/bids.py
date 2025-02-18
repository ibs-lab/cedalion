from pathlib import Path

import pandas as pd
import os


def read_events_from_tsv(fname: str | Path):
    return pd.read_csv(fname, delimiter="\t")


def check_for_bids_field(path_parts: list, 
                         field: str):
    
    field_parts = [part for part in path_parts if field in part]
    if len(field_parts) == 0:
        value_id = None
    else:
        find_value = field_parts[-1].split('_') # assume the lowest directory level supersedes any higher directory level ? not sure if we should assume this 
        value = [vals for vals in find_value if field in vals][0]
        value_id = value.split(field)[1]
        try:
            value_id = value_id.split('-')[1]
        except:
            value_id = value_id
                
            
        
    return value_id


def get_snirf2bids_mapping_csv(dataset_path):
    
    column_names = ["current_name",
                    "sub",
                    "ses",
                    "task",
                    "run",
                    "acq",
                    "cond",
                    "cond_match",
                    "duration"]
    
    
    snirf2bids_mapping_df = pd.DataFrame(columns=column_names)
    
    
    #% IDENTIFY ALL SNIRF FILES IN THE DIRECTORY AND THEIR PATH 
    
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
    
    
    #% CHECK EACH FILE TO GATHER INFO TO POPULATE THE MAPPING_DF
    
    
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
                     "acq": acq,
                     "cond": None,
                     "cond_match": None,
                     "duration": None
                     }
        snirf2bids_mapping_df = pd.concat([snirf2bids_mapping_df, pd.DataFrame([bids_dict])], ignore_index=True)
    
    
    
    snirf2bids_mapping_df.to_csv(os.path.join(dataset_path, 'snirf2BIDS_mapping.csv'), index=None)
    return snirf2bids_mapping_df