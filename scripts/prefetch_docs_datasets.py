"""Helper script to download datasets.

This script retrieves datasets used in example notebooks so that notebooks can be
rendered in parallel.
"""

import cedalion.datasets as ds

ds.get_fingertapping()
ds.get_fingertappingDOT()

ds.get_colin27_segmentation()
ds.get_icbm152_segmentation()
ds.get_colin27_parcel_file()
ds.get_icbm152_parcel_file()
ds.get_ninja_cap_probe()
ds.get_ninja_uhd_cap_probe()

ds.get_precomputed_sensitivity("fingertappingDOT", "colin27")

ds.get_precomputed_sensitivity("nn22_resting", "colin27")
ds.get_nn22_resting_state()
ds.get_photogrammetry_example_scan()
ds.get_multisubject_fingertapping_snirf_paths()