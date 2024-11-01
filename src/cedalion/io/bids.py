from pathlib import Path

import pandas as pd


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