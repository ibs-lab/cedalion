
import cedalion.io
import cedalion.datasets
from cedalion.dataclasses.schemas import validate_stim_schema

def test_read_events_from_tsv():
    bids_top_level_directory = cedalion.datasets.get_multisubject_fingertapping_path()

    for fname in bids_top_level_directory.glob("**/*_events.tsv"):
        stim = cedalion.io.read_events_from_tsv(fname)

        validate_stim_schema(stim)