Physiology
==========

fNIRS signals contain physiological fluctuations from sources unrelated to neural
activity: cardiac pulsations (~1 Hz), respiration (~0.2–0.4 Hz), and slow vasomotor
oscillations (Mayer waves, ~0.1 Hz). These fluctuations can be larger than the
haemodynamic response of interest and must be addressed in preprocessing.

A complementary perspective, treats these signals not merely as noise to be
suppressed, but as informative measurements in their own right. Because fNIRS probes
tissue optics directly, it captures systemic haemodynamic changes alongside neural
responses. In naturalistic or mobile recordings — where participants move, speak, and
experience varying levels of stress or cognitive load — the cardiac, respiratory, and
vasomotor signals recorded alongside brain activity carry information about autonomic
regulation, arousal, and bodily state.

This reframes the preprocessing goal: rather than blindly filtering or subtracting
physiological fluctuations, the preferred approach is to model them explicitly — for
example by including short-separation channels or peripheral physiological regressors
in the GLM, or by using decomposition methods that separate neural from systemic
components. When modelled correctly, the systemic signals become additional outcome
measures that enrich the analysis rather than distortions to be discarded.


.. autosummary::
   :toctree: _autosummary_models
   :recursive:

    cedalion.sigproc.physio.ampd
    cedalion.sigproc.physio.global_component_subtract


Examples
--------

.. nbgallery::
   :glob:

   ../examples/physio/*
   ../examples/signal_quality/24_downweighting_noisy_channels.ipynb