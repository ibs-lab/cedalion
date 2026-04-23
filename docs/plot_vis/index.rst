Plotting and Visualization
==========================

These modules provide tools for plotting data related to fNIRS
analysis, including functions for visualizing scalp plots, sensitivity matrices,
and optode montages.

These subpackages contain building blocks to assemble visualizations:

.. autosummary::
   :toctree: _autosummary_models
   :recursive:

    cedalion.vis.blocks
    cedalion.vis.colors

The functions in these subpackages create complete plots for different usage scenarios:

.. autosummary::
   :toctree: _autosummary_models
   :recursive:

    cedalion.vis.anatomy
    cedalion.vis.anatomy.sensitivity_matrix
    cedalion.vis.quality
    cedalion.vis.timeseries

Cedalion contains two Qt-based GUIs for inspecting time series and probes:

.. autosummary::
   :toctree: _autosummary_models
   :recursive:

    cedalion.vis.misc.plot_probe_gui
    cedalion.vis.misc.time_series_gui


Examples
--------

.. nbgallery::
   :glob:

   ../examples/plots_visualization/*