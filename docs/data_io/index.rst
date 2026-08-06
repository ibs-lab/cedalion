Data structures and I/O
=======================

Cedalion builds on the scientific Python stack — numpy, xarray, pandas — and adds
domain-specific data structures for fNIRS that carry measurement metadata (optode
labels, wavelengths, physical units) alongside the numerical values.

The three core types are:

* **NDTimeSeries** — an ``xr.DataArray`` with a ``time`` dimension and a spatial
  dimension. The spatial dimension is ``channel`` for raw and channel-space data
  (with sub-coordinates ``source`` and ``detector`` that enable joins with the optode
  geometry), ``vertex`` for image-space data on a surface mesh, or ``parcel`` for
  region-level summaries. Physical units (V, mol/L, etc.) are attached via
  pint-xarray and preserved through transformations.

* **LabeledPoints** — an ``xr.DataArray`` with dimensions ``(label, <crs>)`` mapping
  optode and landmark names to 3-D positions in a named coordinate reference system.

* **Recording** — a container whose structure closely mirrors the SNIRF file
  format and serves as the main analysis object. Reading a SNIRF file populates
  ``.timeseries`` (keyed by canonical names such as ``"amp"``, ``"od"``,
  ``"conc"``), ``.geo3d`` (probe geometry), ``.stim`` (stimulus table),
  ``.aux_ts`` (auxiliary time series), and ``.meta_data``; CW, FD, and TD data
  are all supported per the SNIRF specification. Fields such as ``.head_model``,
  ``.masks``, and image-space time series extend beyond what SNIRF currently
  specifies.

For a conceptual introduction with worked examples see the
`Concepts guide <../concepts.md>`_. For a hands-on introduction to these types start
with the example notebooks below.

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
   cedalion.data


Examples
--------

.. nbgallery::
   :glob:

   ../examples/getting_started_io/10_xarray_datastructs_fnirs.ipynb
   ../examples/getting_started_io/11_recording_container.ipynb
   ../examples/getting_started_io/12_read_snirf_files.ipynb
   ../examples/getting_started_io/13_data_structures_intro.ipynb
   ../examples/getting_started_io/14_snirf2bids.ipynb
   ../examples/getting_started_io/34_store_hrfs_in_snirf_file.ipynb