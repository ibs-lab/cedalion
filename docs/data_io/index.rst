Data structures and I/O
=======================

Cedalion builds on existing python packages for handling scientific 
data (such as numpy and xarray), but it also defines several data structures 
for working with fNIRS data in particular. It provides I/O functions for
reading and writing data in various formats.

Data structures
---------------

.. autosummary::
   :toctree: _autosummary_data_structures
   :recursive:
   :nosignatures:

   cedalion.dataclasses
   cedalion.typing
   cedalion.validators
   cedalion.physunits
   
Utilities
---------

.. autosummary::
   :toctree: _autosummary_utils
   :recursive:
   :nosignatures:

   cedalion.xrutils

I/O
---

.. autosummary::
   :toctree: _autosummary_io
   :recursive:
   :nosignatures:

   cedalion.io.snirf
   cedalion.io.anatomy
   cedalion.io.bids
   cedalion.io.forward_model
   cedalion.io.photogrammetry
   cedalion.io.probe_geometry
   cedalion.datasets


Examples
--------

.. nbgallery::
   :glob:

   ../examples/getting_started_io/10_xarray_datastructs_fnirs.ipynb
   ../examples/getting_started_io/11_recording_container.ipynb
   ../examples/getting_started_io/13_data_structures_intro.ipynb
   ../examples/getting_started_io/34_store_hrfs_in_snirf_file.ipynb