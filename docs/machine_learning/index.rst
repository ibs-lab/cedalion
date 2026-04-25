Modeling and Machine Learning
=============================

Cedalion supports two complementary analysis strategies:

**Model-driven analysis** uses the General Linear Model (GLM) to estimate
haemodynamic response functions (HRFs) from known stimulus timings. A design matrix is
constructed from convolutions of stimulus boxcar functions with HRF basis functions
(Gamma, GaussianKernels, DiracDelta). The GLM is fitted per-channel, and beta
coefficients quantify the amplitude of the response to each condition. This approach is
well-suited when the experimental design is block- or event-based and stimuli are
accurately recorded.

**Data-driven analysis** uses decomposition methods (ICA, CCA, SPoC, mSPoC) to
identify shared structure across channels or modalities without relying on a stimulus
model. This is particularly useful for naturalistic recordings, EEG–fNIRS co-recordings,
or when the neural signal of interest cannot be described by a simple HRF.

Both approaches integrate with scikit-learn: Cedalion DataArrays behave as numpy
arrays to sklearn estimators, while the ``mlutils`` module provides cross-validation
utilities designed for the temporal structure of fNIRS data.

Models
------

.. autosummary::
   :toctree: _autosummary_models
   :nosignatures:
   :recursive:

   cedalion.models.glm
   cedalion.math.ar_model
   
Decomposition Methods
---------------------

.. autosummary::
    :toctree: _autosummary_decomp
    :recursive:
    :nosignatures:

    cedalion.sigdecomp.unimodal
    cedalion.sigdecomp.multimodal

Examples
--------

.. nbgallery::
   :glob:

   ../examples/machine_learning/*
   ../examples/modeling/*