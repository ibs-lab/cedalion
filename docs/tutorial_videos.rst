Tutorial Videos
===============

.. Maintainer notes
   ----------------
   Tiles are sphinx-design cards (``grid-item-card``); see
   https://sphinx-design.readthedocs.io/en/latest/cards.html. Thumbnails live in
   ``docs/img/thumbnails`` (1280x720, no spaces in file names).

   * Publishing a content video: in its card, swap the
     ``*_greyscale-coming-soon.jpg`` thumbnail for the matching ``*_color.jpg``,
     add ``:link: <YouTube URL>`` and ``:link-type: url``, add the class
     ``video-card-available`` to ``:class-card:``, drop "(coming soon)" from
     ``:img-alt:``, and add a ``+++`` footer line "▶ Watch on YouTube".
   * Adding a tutorial video: copy a card from the "Tutorial videos" grid of
     section 1 into the section's grid (if the section has no grid yet, copy the
     whole ``.. grid::`` block) above the ``tutorial-placeholder`` card, and change
     ``More tutorial videos`` in that placeholder if it was the first one.
   * Example notebooks use nbsphinx's ``nbgallery`` so titles and thumbnails come
     from the notebooks themselves. conf.py detaches this page from the toctree
     hierarchy so the notebooks keep their usual parents in the navigation.

This page collects Cedalion's training videos in thirteen topics that follow a
typical fNIRS/DOT analysis, from installation to reproducible pipelines. Each
topic contains:

* **Content videos** (5–15 min): slide-based lectures on concepts, theory and
  intuition.
* **Tutorial videos** (5–15 min): (screen) recordings that walk through working code
  in Jupyter notebooks.
* **Example notebooks**: the executable notebooks from this documentation that
  cover the topic. Run them locally or in the cloud via the *Open in Colab* button
  (see :doc:`getting_started/colab_setup`).

Videos are being released step by step. Tiles marked *Coming soon* will link to
the recording once it is published.

.. contents:: Topics
   :local:
   :depth: 1

0. Orientation & Architecture
-----------------------------

Start here: what Cedalion is designed for, how the toolbox is organised into
subpackages, and how versioned environments and modular processing blocks keep analyses
reproducible.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/0.1.C1_Content-Video_Cedalion-Philosophy-and-Scope_greyscale-coming-soon.jpg
      :img-alt: 0.1.C1 content video: Cedalion – Philosophy and Scope (coming soon)
      :class-card: video-card

      **0.1.C1** · Cedalion – Philosophy and Scope

   .. grid-item-card::
      :img-top: img/thumbnails/0.2.C1_Content-Video_Overview-of-Toolbox-Architecture_greyscale-coming-soon.jpg
      :img-alt: 0.2.C1 content video: Overview of Toolbox Architecture (coming soon)
      :class-card: video-card

      **0.2.C1** · Overview of Toolbox Architecture

   .. grid-item-card::
      :img-top: img/thumbnails/0.3.C1_Content-Video_Reproducibility-and-Workflows-Versioning-environments-and-dependency-management_greyscale-coming-soon.jpg
      :img-alt: 0.3.C1 content video: Reproducibility & Workflows: Versioning, environments and dependency management (coming soon)
      :class-card: video-card

      **0.3.C1** · Reproducibility & Workflows: Versioning, environments and dependency management

   .. grid-item-card::
      :img-top: img/thumbnails/0.3.C2_Content-Video_Reproducibility-and-Workflows-Functional-blocks-and-Pipelines_greyscale-coming-soon.jpg
      :img-alt: 0.3.C2 content video: Reproducibility & Workflows: Functional blocks and Pipelines (coming soon)
      :class-card: video-card

      **0.3.C2** · Reproducibility & Workflows: Functional blocks and Pipelines

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

No example notebook covers this topic yet.

**Related documentation:** :doc:`Rationale <rationale>` · :doc:`Core concepts <concepts>` · :doc:`Environments <environments>` · :doc:`Changelog <CHANGELOG>` · :doc:`All examples <examples>` · :doc:`Bibliography <references>` · :doc:`Community <community/index>`


1. Installation, Execution & Data Access
----------------------------------------

How to install Cedalion locally with conda or run it on Google Colab, verify optional
backends such as MCX and NIRFASTer, and download and cache the example datasets used
throughout the documentation.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/1.1.C1_Content-Video_Basics-of-Python-plus-Installation-Overview_greyscale-coming-soon.jpg
      :img-alt: 1.1.C1 content video: Basics of Python + Installation Overview (coming soon)
      :class-card: video-card

      **1.1.C1** · Basics of Python + Installation Overview

.. rubric:: Tutorial videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/1.1.T1_Tutorial-Video_Local-installation-using-conda-WIN.png
      :img-alt: 1.1.T1 tutorial video: Local installation using conda – Windows
      :link: https://www.youtube.com/watch?v=G5zQawG6GDI
      :link-type: url
      :class-card: video-card video-card-available

      **1.1.T1** · Local installation using conda – Windows

      +++
      ▶ Watch on YouTube

   .. grid-item-card::
      :img-top: img/thumbnails/1.1.T1_Tutorial-Video_Local-installation-using-conda-MAC.png
      :img-alt: 1.1.T1 tutorial video: Local installation using conda – macOS
      :link: https://www.youtube.com/watch?v=wcS69hFFUK4
      :link-type: url
      :class-card: video-card video-card-available

      **1.1.T1** · Local installation using conda – macOS

      +++
      ▶ Watch on YouTube

.. card::
   :class-card: tutorial-placeholder

   More tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/getting_started_io/00_test_installation

**Related documentation:** :doc:`Installation <getting_started/installation>` · :doc:`Google Colab setup <getting_started/colab_setup>` · :doc:`Quick start <getting_started/quickstart>`


2. Data Structures & I/O
------------------------

Cedalion keeps data in labelled, unit-aware xarray DataArrays that are collected in a
``Recording`` container. This section introduces these data structures and shows how to
read and write them using the SNIRF and BIDS standards.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/2.1.C1_Content-Video_Introduction-to-XArrays_greyscale-coming-soon.jpg
      :img-alt: 2.1.C1 content video: Introduction to XArrays (coming soon)
      :class-card: video-card

      **2.1.C1** · Introduction to XArrays

   .. grid-item-card::
      :img-top: img/thumbnails/2.2.C1_Content-Video_Overview-of-the-recording-container_greyscale-coming-soon.jpg
      :img-alt: 2.2.C1 content video: Overview of the recording container (coming soon)
      :class-card: video-card

      **2.2.C1** · Overview of the recording container

   .. grid-item-card::
      :img-top: img/thumbnails/2.3.C1_Content-Video_File-formats-and-standards-Brief-introduction-to-SNIRF-and-BIDS_greyscale-coming-soon.jpg
      :img-alt: 2.3.C1 content video: File formats & standards: Brief introduction to SNIRF and BIDS (coming soon)
      :class-card: video-card

      **2.3.C1** · File formats & standards: Brief introduction to SNIRF and BIDS

   .. grid-item-card::
      :img-top: img/thumbnails/2.4.C1_Content-Video_Overview-of-Data-I-O-Functionality_greyscale-coming-soon.jpg
      :img-alt: 2.4.C1 content video: Overview of Data I/O Functionality (coming soon)
      :class-card: video-card

      **2.4.C1** · Overview of Data I/O Functionality

.. rubric:: Tutorial videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/2.3.T3_Tutorial-Video_Using-the-SNIRF2BIDS-conversion-tool.jpg
      :img-alt: 2.3.T3 tutorial video: Using the SNIRF2BIDS conversion tool
      :link: https://www.youtube.com/watch?v=UYL3BUg_7xE
      :link-type: url
      :class-card: video-card video-card-available

      **2.3.T3** · Using the SNIRF2BIDS conversion tool

      Hands-on workshop: BIDS-ifying fNIRS – a Python-based community tool for open data sharing

      +++
      ▶ Watch on YouTube

.. card::
   :class-card: tutorial-placeholder

   More tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/getting_started_io/10_xarray_datastructs_fnirs
   examples/getting_started_io/11_recording_container
   examples/getting_started_io/12_read_snirf_files
   examples/getting_started_io/13_data_structures_intro
   examples/getting_started_io/14_snirf2bids
   examples/getting_started_io/34_store_hrfs_in_snirf_file

**Related documentation:** :doc:`Data structures <data_structures/index>` · :doc:`Data structures and I/O <data_io/index>`


3. Modified Beer-Lambert Law
----------------------------

Converting raw light intensities to optical density and then to changes in oxy- and
deoxyhaemoglobin concentration with the modified Beer-Lambert law, including
plausibility checks and the role of source-detector distances.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/3.1.C1_Content-Video_Introduction-to-Optical-Density-and-the-mBLL_greyscale-coming-soon.jpg
      :img-alt: 3.1.C1 content video: Introduction to Optical Density and the mBLL (coming soon)
      :class-card: video-card

      **3.1.C1** · Introduction to Optical Density and the mBLL

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/getting_started_io/10_xarray_datastructs_fnirs
   examples/getting_started_io/12_read_snirf_files

**Related documentation:** :doc:`Core concepts <concepts>` · :doc:`Signal processing <sigproc/index>`


4. Signal Quality & Preprocessing
---------------------------------

How to quantify signal quality (e.g. SCI, PSP, SNR, GVTD), build and combine quality
masks, detect and correct motion artefacts, and assemble a complete preprocessing
pipeline for a raw fNIRS dataset.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/4.1.C1_Content-Video_Signal-Quality-Concepts-and-Sources-of-Noise_greyscale-coming-soon.jpg
      :img-alt: 4.1.C1 content video: Signal Quality: Concepts and Sources of Noise (coming soon)
      :class-card: video-card

      **4.1.C1** · Signal Quality: Concepts and Sources of Noise

   .. grid-item-card::
      :img-top: img/thumbnails/4.2.C1_Content-Video_Quality-Metrics-Overview-of-Frequently-Used-Methods_greyscale-coming-soon.jpg
      :img-alt: 4.2.C1 content video: Quality Metrics: Overview of Frequently Used Methods (coming soon)
      :class-card: video-card

      **4.2.C1** · Quality Metrics: Overview of Frequently Used Methods

   .. grid-item-card::
      :img-top: img/thumbnails/4.3.C1_Content-Video_Sources-of-Artifacts-in-fNIRS-Motion-and-Physiology_greyscale-coming-soon.jpg
      :img-alt: 4.3.C1 content video: Sources of Artifacts in fNIRS: Motion and Physiology (coming soon)
      :class-card: video-card

      **4.3.C1** · Sources of Artifacts in fNIRS: Motion and Physiology

   .. grid-item-card::
      :img-top: img/thumbnails/4.4.C1_Content-Video_Practical-Preprocessing-End-to-end-Example-of-A-Raw-Dataset_greyscale-coming-soon.jpg
      :img-alt: 4.4.C1 content video: Practical Preprocessing: End-to-end Example of a Raw Dataset (coming soon)
      :class-card: video-card

      **4.4.C1** · Practical Preprocessing: End-to-end Example of a Raw Dataset

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/signal_quality/20_scalp_coupling_index
   examples/signal_quality/21_data_quality_and_pruning
   examples/signal_quality/22_motion_artefacts_and_correction
   examples/signal_quality/24_downweighting_noisy_channels
   examples/signal_quality/25_intro_quality_workshop
   examples/tutorial/3_signal_processing

**Related documentation:** :doc:`Signal processing <sigproc/index>` · :doc:`Physiology <physio/index>`


5. General Linear Model (GLM)
-----------------------------

Modelling the haemodynamic response with the general linear model: design matrices with
HRF basis functions, drift and short-channel regressors, the available solvers and noise
models, and statistical inference on the estimated β values.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/5.1.C1_Content-Video_GLM-fundamentals-Short-Channels-Physiology-Regression-and-Introduction-to-the-General-Linear_greyscale-coming-soon.jpg
      :img-alt: 5.1.C1 content video: GLM fundamentals: Short Channels, Physiology Regression and Introduction to the General Linear Model (coming soon)
      :class-card: video-card

      **5.1.C1** · GLM fundamentals: Short Channels, Physiology Regression and Introduction to the General Linear Model

   .. grid-item-card::
      :img-top: img/thumbnails/5.2.C1_Content-Video_GLM-architecture-in-Cedalion_greyscale-coming-soon.jpg
      :img-alt: 5.2.C1 content video: GLM architecture in Cedalion (coming soon)
      :class-card: video-card

      **5.2.C1** · GLM architecture in Cedalion

   .. grid-item-card::
      :img-top: img/thumbnails/5.3.C1_Content-Video_GLM-Statistics-Statsmodels-in-Cedalion_greyscale-coming-soon.jpg
      :img-alt: 5.3.C1 content video: GLM Statistics: Statsmodels in Cedalion (coming soon)
      :class-card: video-card

      **5.3.C1** · GLM Statistics: Statsmodels in Cedalion

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/modeling/31_glm_basis_functions
   examples/modeling/32_glm_fingertapping_example
   examples/modeling/33_glm_illustrative_example
   examples/modeling/35_statsmodels_overview
   examples/modeling/36_glm_workshop
   examples/tutorial/4_model_driven_analysis

**Related documentation:** :doc:`Modeling and machine learning <machine_learning/index>`


6. Head Models & Forward Modeling
---------------------------------

Head models describe the anatomy that light travels through. This section covers
atlas-based and individual head models, coordinate systems, anatomical parcellations,
and forward modelling with Monte Carlo (MCX) and finite-element (NIRFASTer) photon
simulations.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/6.1.C1_Content-Video_Head-models-in-Neuroimaging-Overview_greyscale-coming-soon.jpg
      :img-alt: 6.1.C1 content video: Head models in Neuroimaging: Overview (coming soon)
      :class-card: video-card

      **6.1.C1** · Head models in Neuroimaging: Overview

   .. grid-item-card::
      :img-top: img/thumbnails/6.2.C1_Content-Video_Overview-of-standard-head-model-coordinate-systems_greyscale-coming-soon.jpg
      :img-alt: 6.2.C1 content video: Overview of standard head-model coordinate systems (coming soon)
      :class-card: video-card

      **6.2.C1** · Overview of standard head-model coordinate systems

   .. grid-item-card::
      :img-top: img/thumbnails/6.3.C1_Content-Video_Forward-modeling-with-photon-simulations_greyscale-coming-soon.jpg
      :img-alt: 6.3.C1 content video: Forward modeling with photon simulations (coming soon)
      :class-card: video-card

      **6.3.C1** · Forward modeling with photon simulations

   .. grid-item-card::
      :img-top: img/thumbnails/6.4.C1_Content-Video_Anatomical-Atlases-and-Parcellation-Schemes_greyscale-coming-soon.jpg
      :img-alt: 6.4.C1 content video: Anatomical Atlases and Parcellation Schemes (coming soon)
      :class-card: video-card

      **6.4.C1** · Anatomical Atlases and Parcellation Schemes

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/head_models/43a_head_models_overview
   examples/head_models/43b_individualized_head_models
   examples/head_models/44_head_models_crs_scaling
   examples/head_models/42_1010_system
   examples/head_models/48_headmodel_landmarks_verification
   examples/head_models/51_spring_relaxation_registration
   examples/head_models/46_precompute_fluence
   examples/head_models/45_parcel_sensitivity
   examples/head_models/52_mni_atlas_labels_aal3_brodmann
   examples/tutorial/1_heads_and_fwm

**Related documentation:** :doc:`Diffuse optical tomography <dot/index>`


7. Photogrammetric Optode Co-Registration & Probe Design
--------------------------------------------------------

Photogrammetry measures where the optodes actually sit on a participant's head. This
section covers optode detection in 3D scans, manual quality control, and labelling the
detected optodes by registering them to the probe layout. Tools for probe design are
planned.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/7.1.C1_Content-Video_Photogrammetry-and-co-registration-Motivation_greyscale-coming-soon.jpg
      :img-alt: 7.1.C1 content video: Photogrammetry & co-registration: Motivation (coming soon)
      :class-card: video-card

      **7.1.C1** · Photogrammetry & co-registration: Motivation

   .. grid-item-card::
      :img-top: img/thumbnails/7.2.C1_Content-Video_Photogrammetric-Optode-Co-Registration-in-Cedalion_greyscale-coming-soon.jpg
      :img-alt: 7.2.C1 content video: Photogrammetric Optode Co-Registration in Cedalion (coming soon)
      :class-card: video-card

      **7.2.C1** · Photogrammetric Optode Co-Registration in Cedalion

   .. grid-item-card::
      :img-top: img/thumbnails/7.3.C1_Content-Video_Probe-Design-in-Cedalion_greyscale-coming-soon.jpg
      :img-alt: 7.3.C1 content video: Probe Design in Cedalion (coming soon)
      :class-card: video-card

      **7.3.C1** · Probe Design in Cedalion

.. rubric:: Tutorial videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/7.1.T1_Tutorial-Video_Acquiring-and-inspecting-raw-photogrammetry-meshes.png
      :img-alt: 7.1.T1 tutorial video: Acquiring and inspecting raw photogrammetry meshes
      :link: https://www.youtube.com/watch?v=PMBUWHnLXUo
      :link-type: url
      :class-card: video-card video-card-available

      **7.1.T1** · Acquiring and inspecting raw photogrammetry meshes

      fNIRS/EEG photogrammetry tutorial with the Cedalion toolbox

      +++
      ▶ Watch on YouTube

.. card::
   :class-card: tutorial-placeholder

   More tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/head_models/41_photogrammetric_optode_coregistration
   examples/tutorial/2_photogrammetry
   examples/head_models/51_spring_relaxation_registration


8. DOT Image Reconstruction
---------------------------

Diffuse optical tomography reconstructs images of cortical haemodynamic activity from
channel-space measurements by inverting the sensitivity matrix. This section covers the
inverse problem, regularisation choices, and aggregating images into brain parcels.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/8.1.C1_Content-Video_Intro-to-DOT-Image-Reconstruction-and-Inverse-Problem_greyscale-coming-soon.jpg
      :img-alt: 8.1.C1 content video: Intro to DOT Image Reconstruction & Inverse Problem (coming soon)
      :class-card: video-card

      **8.1.C1** · Intro to DOT Image Reconstruction & Inverse Problem

   .. grid-item-card::
      :img-top: img/thumbnails/8.2.C1_Content-Video_Overview-Image-Reconstruction-workflow-in-Cedalion_greyscale-coming-soon.jpg
      :img-alt: 8.2.C1 content video: Overview: Image Reconstruction workflow in Cedalion (coming soon)
      :class-card: video-card

      **8.2.C1** · Overview: Image Reconstruction workflow in Cedalion

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/head_models/40_image_reconstruction
   examples/head_models/47_image_reconstruction_regularizations
   examples/head_models/45_parcel_sensitivity
   examples/tutorial/5_image_reconstruction

**Related documentation:** :doc:`Diffuse optical tomography <dot/index>`


9. Visualization & Interpretation
---------------------------------

Cedalion's plotting building blocks for time series, probe layouts, scalp and brain
surfaces and reconstructed images, together with interactive tools for inspecting data.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/9.1.C1_Content-Video_fNIRS-DOT-visualization-blocks-in-Cedalion_greyscale-coming-soon.jpg
      :img-alt: 9.1.C1 content video: fNIRS/DOT visualization blocks in Cedalion (coming soon)
      :class-card: video-card

      **9.1.C1** · fNIRS/DOT visualization blocks in Cedalion

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/plots_visualization/12_plots_example
   examples/head_models/43a_head_models_overview

**Related documentation:** :doc:`Plotting and visualization <plot_vis/index>`


10. Multimodal & Data-Driven Analysis
-------------------------------------

Data-driven methods that do not require a stimulus model: unimodal and multimodal source
decomposition (ICA, the CCA family, mSPoC) and single-trial classification with
scikit-learn.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/10.1.C1_Content-Video_Multimodal-fusion-variance-vs-decoding_greyscale-coming-soon.jpg
      :img-alt: 10.1.C1 content video: Multimodal fusion: variance vs decoding (coming soon)
      :class-card: video-card

      **10.1.C1** · Multimodal fusion: variance vs decoding

   .. grid-item-card::
      :img-top: img/thumbnails/10.2.C1_Content-Video_Source-Decomposition-Methods-Overview-CCA-ICA-SPoC-family_greyscale-coming-soon.jpg
      :img-alt: 10.2.C1 content video: Source Decomposition Methods Overview: CCA, ICA, SPoC family (coming soon)
      :class-card: video-card

      **10.2.C1** · Source Decomposition Methods Overview: CCA, ICA, SPoC family

   .. grid-item-card::
      :img-top: img/thumbnails/10.3.C1_Content-Video_ML-fNIRS-DOT-Classification-Workflows_greyscale-coming-soon.jpg
      :img-alt: 10.3.C1 content video: ML fNIRS/DOT Classification Workflows (coming soon)
      :class-card: video-card

      **10.3.C1** · ML fNIRS/DOT Classification Workflows

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/machine_learning/51_multimodal_source_decomposition_on_synthetic_fNIRS-EEG_data
   examples/machine_learning/52_ica_erbm_fingertapping_example
   examples/machine_learning/53_constrained_ICA_example
   examples/machine_learning/50_finger_tapping_lda_classification
   examples/machine_learning/50b_advanced_finger_tapping_lda_classification
   examples/tutorial/6_data_driven_analysis

**Related documentation:** :doc:`Modeling and machine learning <machine_learning/index>`


11. Simulation & Data Augmentation
----------------------------------

Synthetic haemodynamic responses, motion artefacts and paired fNIRS-EEG data with known
ground truth, for benchmarking algorithms and augmenting training data for machine
learning.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/11.1.C1_Content-Video_Synthetic-data-for-Benchmarking-and-Augmentation-Rationale_greyscale-coming-soon.jpg
      :img-alt: 11.1.C1 content video: Synthetic data for Benchmarking and Augmentation: Rationale (coming soon)
      :class-card: video-card

      **11.1.C1** · Synthetic data for Benchmarking and Augmentation: Rationale

   .. grid-item-card::
      :img-top: img/thumbnails/11.2.C1_Content-Video_Synthetic-data-generation-and-augmentation-in-Cedalion_greyscale-coming-soon.jpg
      :img-alt: 11.2.C1 content video: Synthetic data generation and augmentation in Cedalion (coming soon)
      :class-card: video-card

      **11.2.C1** · Synthetic data generation and augmentation in Cedalion

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

.. nbgallery::

   examples/augmentation/62_synthetic_hrfs_example
   examples/augmentation/61_synthetic_artifacts_example
   examples/tutorial/7_data_augmentation
   examples/machine_learning/51_multimodal_source_decomposition_on_synthetic_fNIRS-EEG_data

**Related documentation:** :doc:`Synthetic data <synth/index>`


12. Pipelines, Scaling & Reproducibility
----------------------------------------

Moving from interactive notebooks to scripted, reproducible pipelines that process many
datasets and can be shared alongside a publication.

.. rubric:: Content videos

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card::
      :img-top: img/thumbnails/12.1.C1_Content-Video_Pipeline-abstraction-From-Notebooks-to-Pipelines_greyscale-coming-soon.jpg
      :img-alt: 12.1.C1 content video: Pipeline abstraction: From Notebooks to Pipelines (coming soon)
      :class-card: video-card

      **12.1.C1** · Pipeline abstraction: From Notebooks to Pipelines

   .. grid-item-card::
      :img-top: img/thumbnails/12.2.C1_Content-Video_Snakemake-integration-in-Cedalion-Concepts-and-Introduction_greyscale-coming-soon.jpg
      :img-alt: 12.2.C1 content video: Snakemake integration in Cedalion: Concepts and Introduction (coming soon)
      :class-card: video-card

      **12.2.C1** · Snakemake integration in Cedalion: Concepts and Introduction

   .. grid-item-card::
      :img-top: img/thumbnails/12.3.C1_Content-Video_Reproducible-and-shareable-Analyses_greyscale-coming-soon.jpg
      :img-alt: 12.3.C1 content video: Reproducible and shareable Analyses (coming soon)
      :class-card: video-card

      **12.3.C1** · Reproducible and shareable Analyses

.. rubric:: Tutorial videos

.. card::
   :class-card: tutorial-placeholder

   Tutorial videos for this topic will show here as soon as they are
   recorded.

.. rubric:: Example notebooks

No example notebook covers this topic yet.

**Related documentation:** :doc:`Environments <environments>` · :doc:`Changelog <CHANGELOG>`
