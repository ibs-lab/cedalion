"""Functions for reading BIDS data."""

from pathlib import Path

import pandas as pd


def read_events_from_tsv(fname: str | Path):
    return pd.read_csv(fname, delimiter="\t")
