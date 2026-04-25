Signal processing
=================

Cedalion's signal processing modules cover the full preprocessing pipeline for
continuous-wave fNIRS data. A typical workflow proceeds in this order:

1. **Quality assessment** (``cedalion.sigproc.quality``) — compute channel-level
   quality metrics (scalp coupling index, SNR, peak-power ratio), build boolean masks,
   and prune channels that fail quality thresholds.

2. **CW fNIRS conversions** (``cedalion.nirs.cw``) — convert raw light
   intensities to log-ratio optical density changes — apply the modified
   Beer-Lambert law to obtain HbO and HbR concentration changes, etc. Note that 
   (``cedalion.nirs.fd``) and (``cedalion.nirs.td``) also exist but are currently 
   limited to reading/writing SNIRF files.

3. **Motion artefact correction** (``cedalion.sigproc.motion``) — detect and correct
   motion-induced spikes and baseline shifts using methods such as TDDR, CBSI, and
   wavelet-based artefact rejection.

4. **Temporal filtering** (``cedalion.sigproc.frequency``) — apply band-pass or
   low-pass filters to remove cardiac noise, high-frequency interference, and slow
   drifts.

5. **Physiological signal extraction** (``cedalion.sigproc.physio``) — functions for 
   regression or advanced analysis of systemic physiological fluctuations 
   (cardiac, respiratory, vasomotor).

6. **Epoching** (``cedalion.sigproc.epochs``) — extract stimulus-locked trial segments
   from the continuous timeseries.

The ``cedalion.sigproc.tasks`` module provides high-level ``@task``-decorated wrappers
for each step that operate directly on a ``Recording`` object.

.. autosummary::
   :toctree: _autosummary_sigproc
   :nosignatures:
   :recursive:

   cedalion.nirs
   cedalion.nirs.cw
   cedalion.nirs.fd
   cedalion.nirs.td
   cedalion.sigproc.epochs
   cedalion.sigproc.frequency
   cedalion.sigproc.motion
   cedalion.sigproc.quality
   cedalion.sigproc.physio


Examples
--------

.. nbgallery::
   :glob:

   ../examples/signal_quality/*
   ../examples/physio/*