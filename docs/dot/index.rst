Diffuse optical tomography
==========================

.. py:currentmodule:: cedalion.dot

Diffuse optical tomography (DOT) extends channel-space fNIRS to 3-D image
reconstruction. Rather than reporting one haemodynamic value per channel, DOT estimates
the spatial distribution of HbO and HbR changes across the cortical surface by solving
an inverse problem.

The DOT pipeline in Cedalion consists of the following steps:

1. **Head model construction** (``cedalion.dot.head_model``) — a
   ``TwoSurfaceHeadModel`` wraps segmented cortex and scalp meshes, landmark
   coordinates, and an affine transform between sensor space and MRI space.
   Pre-built models for standard atlases (``colin27``, ``icbm152``) are provided;
   custom models can be generated from individual MRI data.

2. **Optode co-registration** (``cedalion.geometry.registration``) — register
   digitized optode positions to the head model using rigid-body or affine transforms,
   or via photogrammetric scalp surface matching.

3. **Forward model** (``cedalion.dot.forward_model``) — compute the sensitivity
   matrix (also called the Jacobian or *A* matrix) that maps cortical absorption
   changes to detector-level OD changes. Two simulation backends are supported:
   GPU-accelerated Monte Carlo (MCX, via ``pmcx``/``pmcxcl``) and a finite element
   method (FEM) solver via the NIRFASTer plugin (requires a separate installation).

4. **Image reconstruction** (``cedalion.dot.image_recon``) — invert the forward model
   to recover the cortical image from the measured OD changes. The ``ImageRecon``
   class implements a regularised pseudoinverse (Wiener filter) with configurable
   regularisation parameters (``alpha_meas``, ``alpha_spatial``, ``lambda_R_conc``)
   and optional spatial basis functions to reduce the effective number of unknowns.
   Reconstruction can target absorption changes per wavelength (``mua``), haemoglobin
   concentrations directly (``conc``), or absorption first then concentration
   (``mua2conc``).

5. **Parcellation** (``cedalion.dot.head_model``) — parcel labels from the standard
   atlas are stored as vertex coordinates on the brain surface, allowing reconstructed
   vertex-space images to be aggregated to anatomical regions of interest.

.. autosummary::
   :toctree: _autosummary_dot
   :recursive:

   cedalion.geometry

   cedalion.dot


Examples
--------

.. nbgallery::
   :glob:

   ../examples/head_models/*