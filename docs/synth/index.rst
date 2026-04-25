Synthetic Data
==============

Cedalion provides tools for generating synthetic fNIRS data for two purposes:

**Algorithm development and testing** — synthetic HRFs and motion artefacts can be
added to real or noise-only recordings to benchmark preprocessing and detection
algorithms under controlled conditions where the ground truth is known.

**Machine learning benchmarks** — the ``BimodalToyDataSimulation`` in
``cedalion.sim.datasets`` generates paired synthetic fNIRS+EEG datasets with
controllable signal-to-noise ratio, frequency band, inter-modality time lag, and
mixing matrix structure. It can be used to create reproducible benchmark
experiments for multimodal decomposition methods.

.. autosummary::
   :toctree: _autosummary_synth
   :recursive:

   cedalion.sim.synthetic_artifact
   cedalion.sim.synthetic_hrf


Examples
--------

.. nbgallery::
   :glob:

   ../examples/augmentation/*