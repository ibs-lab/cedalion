"""Helper script to download datasets.

This script retrieves datasets used in example notebooks so that notebooks can be
rendered in parallel.
"""

import cedalion.data as data

data.get_fingertapping()
data.get_fingertappingDOT()

data.get_colin27_segmentation()
data.get_icbm152_segmentation()
data.get_colin27_parcel_file()
data.get_icbm152_parcel_file()
data.get_ninja_cap_probe()
data.get_ninja_uhd_cap_probe()

data.get_precomputed_sensitivity("fingertappingDOT", "colin27")

data.get_precomputed_sensitivity("nn22_resting", "colin27")
data.get_nn22_resting_state()
data.get_photogrammetry_example_scan()
data.get_multisubject_fingertapping_snirf_paths()
