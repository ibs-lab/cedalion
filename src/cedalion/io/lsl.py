"""Contains functionality for handling .xdf files containing LSL streams."""

import os
import glob
import pyxdf
import numpy as np
import pandas as pd
import xarray as xr

def read_stream(stream):
    """
    Read a stream from an XDF file and return the data, timestamps, and metadata.
    """

    # Extract the data and timestamps from the stream
    data = stream['time_series']
    timestamps = stream['time_stamps']
    metadata = stream['info']['desc'][0]
    
    return data, timestamps, metadata

def extract_unique_stream(streams, stream_name):
    """ Try to extract a unique stream from the list of streams with the provided name.

    It will raise and error if there are no streams with the provided name, or 
    if there are multiple streams with the same name and non-zero samples. If there are
    multiple streams with the same name but only one of them has non-zero samples, it will
    return that stream.
    
    """
    # Read all streams with the provided name
    found_streams = [s for s in streams if s['info']['name'][0] == stream_name]
    if len(found_streams) == 0:
        raise ValueError(f"No LSL stream found with name '{stream_name}'")
    elif len(found_streams) > 1:
        print(f"Warning: {len(found_streams)} LSL streams found with name '{stream_name}'.")
        # Look for streams with non-zero samples
        stream_sample_shapes = [read_stream(s)[1].shape[0] for s in found_streams]
        # Raise errors if there are no non-zero-sample streams or more than one
        if np.count_nonzero(stream_sample_shapes) == 0:
            raise ValueError("All of them have zero samples, cannot proceed.")
        elif np.count_nonzero(stream_sample_shapes) > 1:
            raise ValueError(f"Among which {np.sum(np.count_nonzero(stream_sample_shapes))} have non-zero samples found, cannot proceed.")
        # Get the unique stream with non-zero samples
        else:
            print("But there is only one of them with non-zero samples, proceeding with it.")
            stream_ndx = np.where(stream_sample_shapes)[0][0]
    else:
        stream_ndx = 0

    # Pick the only stream with non-zero samples
    found_stream = found_streams[stream_ndx]

    return found_stream

## Markers

def get_stim_from_lsl(streams=None, xdf_file=None, stream_name='PsychoPyMarker', duration_dict=None, value_dict=None):
    """Load stimulus markers from an LSL stream and returns a DataFrame with the markers.

    Args:
        streams (list, optional): List of streams loaded from an XDF file.
        xdf_file (str, optional): Path to the XDF file containing LSL streams.
        stream_name (str): Name of the stream to extract markers from. Default is 'PsychoPyMarker'.
        duration_dict (dict, optional): Dictionary mapping trial types to their durations. 
            If None, default durations are calculated as the difference between consecutive onsets.
        value_dict (dict, optional): Dictionary mapping trial types to their values. If None, all values are set to 1.
    Returns:
        pd.DataFrame: DataFrame containing the markers with columns ['onset', 'duration', 'value', 'trial_type'].
    """

    # If xdf_file is provided, load the streams from the file
    if xdf_file is not None:
        print("Loading streams from XDF file:", xdf_file)
        streams, _ = pyxdf.load_xdf(xdf_file)
    # If streams are already provided, use them directly
    elif streams is not None:
        pass
    else:
        raise ValueError("Either streams or xdf_file must be provided.")

    # Look for the only stream with the provided name and non-zero samples
    stim_stream = extract_unique_stream(streams, stream_name)

    # Extract the data, timestamps, and metadata from the stream
    markers_data, markers_ts, markers_metadata = read_stream(stim_stream)

    # Initialize empty markerrs DataFrame
    stim_df = pd.DataFrame(columns=['onset', 'duration', 'value', 'trial_type'])

    # Set trial_type and onset columns to the markers data and timestamps from LSL
    stim_df['trial_type'] = np.array(markers_data).squeeze()
    stim_df['onset'] = markers_ts

    
    # Calculate default duration as difference between current and next onset
    durations_from_shift = stim_df['onset'].shift(-1) - stim_df['onset']
    durations_from_shift = durations_from_shift.fillna(0)  # Set duration of last row to zero
    # If durations provided for some markers, use those rather than the default
    if duration_dict is not None:
        durations_from_dict = stim_df['trial_type'].map(duration_dict)
        # Combine both durations, using the one from the dictionary if available, otherwise the default
        stim_df['duration'] = [a if ~np.isnan(a) else b for a,b in zip(durations_from_dict, durations_from_shift)]
    else:
        stim_df['duration'] = durations_from_shift

    # Fill in 'value' column based on provided dictionary or with 1 as default
    if value_dict is None:
        stim_df['value'] = 1  # Default value if no dictionary is provided
    else:
        # Default value is 1 if not specified in the dictionary
        stim_df['value'] = stim_df['trial_type'].map(value_dict).fillna(1)

    return stim_df


def lsl_stim_to_tsv(xdf_file, tsv_file=None, stream_name='PsychoPyMarker'):
    """Read LSL stimulus markers from an XDF file and store them in a TSV file.

    It removes the first marker's onset time to start at 0, and saves the markers in a TSV format.

    Args:
        xdf_file (str): Path to the XDF file containing LSL streams.
        tsv_file (str, optional): Path to save the TSV file. If None, it will be saved in the same folder as the xdf_file with the name 'events_lsl.tsv'.
        stream_name (str): Name of the stream to extract markers from. Default is 'PsychoPyMarker'.
    """
    
    # Load the markers from the XDF file
    markers_df = get_stim_from_lsl(xdf_file=xdf_file, stream_name=stream_name)

    
    if tsv_file is None:
        # Use stim.tsv as default name
        tsv_file = os.path.join(os.path.dirname(xdf_file), 'events_lsl.tsv')
    
    # If file exists, ask for confirmation to overwrite
    if os.path.exists(tsv_file):
        confirm = input(f"File {tsv_file} already exists. Overwrite? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation cancelled. File not saved.")
            return

    # Save the DataFrame to a TSV file
    markers_df.to_csv(tsv_file, sep='\t', index=False)
    
    print(f"Markers saved to {tsv_file}")
