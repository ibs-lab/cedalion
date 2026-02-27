# Changelog

## Unreleased changes (available on the `dev` branch)
 
### Added
- Added the function `cedalion.geometry.landmarks.normalize_landmarks_labels` that case-insensitively maps common landmark labels to canonical names, by [Mohammad Orabe](https://github.com/orabe). ([#132](https://github.com/ibs-lab/cedalion/pull/132))
- Added functionality and examples for constrained ICA methods (arc-ERBM, arc-EBM),  by [Jacqueline Behrendt](https://github.com/jackybehrendt12). ([#133](https://github.com/ibs-lab/cedalion/pull/133))
- An example notebook for ICA source extraction was added, by [Jacqueline Behrendt](https://github.com/jackybehrendt12). 
([#112](https://github.com/ibs-lab/cedalion/pull/112))
- Added `TwoSurfaceHeadmodel.scale_to_headsize` and `TwoSurfaceHeadmodel.scale_to_landmarks` to adjust the head model's size to the head circumferences or digitized landmarks, respectively. By [Eike Middell](https://github.com/emiddell).
- The factory method `cedalion.dot.get_standard_headmodel` to construct the `TwoSurfaceHeadModel` of the standard Colin27 and ICBM-152 heads was added, by [Eike Middell](https://github.com/emiddell).
- Added `cedalion.xrutils.dot_dataarray_csr` for matrix products between `xr.DataArray` 
  and `scipy.sparse` arrays, by [Eike Middell](https://github.com/emiddell).
- Added `cedalion.geometry.landmarks.normalize_landmarks_labels` to map alternative landmark names (e.g., "nasion", "left ear", "nz") to their canonical 10-10 system labels (e.g. Nz, LPA). The function handles now case-insensitive matching and supports common naming conventions. Usage: `geo3d = normalize_landmarks_labels(geo3d)` before calling registration or plotting functions, by [Mohammad Orabe](https://github.com/orabe). ([#84](https://github.com/ibs-lab/cedalion/issues/84))
### Changed
- The package `cedalion.sigproc.motion_correct` was renamed to `cedalion.sigproc.motion`.
- The ICA-EBM and ICA_ERBM implementations were moved into `cedalion.sigdecomp.unimodal`.
- The class `cedalion.dot.ForwardModel` accepts also head models that are not in 
voxel space. They will be transformed to voxel space internally.
- Refactored `cedalion.plots` into `cedalion.vis` and its subpackages. This cleans up the code structure and should help with discovering existing functions. The package `cedalion.vis.blocks` emphasizes building blocks for larger visualizations. Please refer to `examples/plots_visualization/12_plots_example.ipynb` to get an overview. Importing `cedalion.plots` will throw a deprecation warning to trigger adoption. By [Eike Middell](https://github.com/emiddell).
- Renamed `LabeledPointCloud` to `LabeledPoints`.
- Split up the `.nirs` submodule into `.nirs.cw`, `.nirs.fd` and `.nirs.td`. 
- Merged the submodules `cedalion.datasets` and `cedalion.data`. All functions to
access example datasets are now available under `cedalion.data`. 
- The fiducial landmarks of the standard Colin27 and ICBM-152 heads changed. Using the landmark builder 10-10 coordinates were calculated for both heads. These are now 
  available when head models are created with `get_standard_headmodel`. The example
  notebook `48_headmodel_landmarks_verification.ipynb` documents the origin of all 
  landmarksand quantitatively compares the output of the LandmarkBuilder to published
  coordinates.  By [Eike Middell](https://github.com/emiddell).
- Added the parameter `mode` to `TwoSurfaceHeadModel.align_and_snap_to_scalp` to switch
  between constrained affine transformations (translation, rotation, isotropic scaling)
  and unconstrained affine transformations (also anisotropic scaling, shearing and reflection). The default behaviour from `align_and_snap_to_scalp` was changed from
  constrained to unconstrained affine transformations. 
- The image reconstruction methods were refactored to offer more regularization methods     
  (including spatial basis functions) as well as direct and indirect reconstructions and to provide a simpler interface to the user. All functions are now located under `cedalion.dot`. By [Laura Carlton](https://github.com/lauracarlton), [Alexander von Lühmann](https://github.com/avolu) and [Eike Middell](https://github.com/emiddell). ([#130](https://github.com/ibs-lab/cedalion/pull/86))
- For the Colin27 and ICBM-152 heads the label for the coordinate reference system was changed from `'aligned'` to `'mni'`.
- The ninja HD and UHD cap coordinates were changed from a right-handed to a left-handed
coordinate system, by [Nils Harmening](https://github.com/harmening). ([#110](https://github.com/ibs-lab/cedalion/pull/110))
- Changed the names of several motion correction algorithms from `motion_correct.motion_correct_X` 
to `motion_correct.X`. Argument names were made PEP8 compliant. The example `22_motion_artefacts_and_correction` was improved. By [Eike Middell](https://github.com/emiddell).
- The function `cedalion.vis.anatomy.plot_montage3D` now accepts a `landmarks` parameter to specify which landmarks should be highlighted. Pass `None` (default) to show all available canonical registration landmarks (e.g. Nz, Iz, LPA, RPA, Cz), a list of landmark names to show specific ones, or an empty list to show none, by [Mohammad Orabe](https://github.com/orabe). ([#84](https://github.com/ibs-lab/cedalion/issues/84))
- Included t-stat thresholding in `cedalion.vis.misc.plot_probe_gui`, by [Shannon Kelley](https://github.com/shankell212). ([#131](https://github.com/ibs-lab/cedalion/pull/131))
- Extended the GLM to work not only in channel space but also on time series with other spatial dimensions (vertices, parcels,...), by [Miray Altinkaynak](https://github.com/maltink). ([#120](https://github.com/ibs-lab/cedalion/pull/120)) 


### Deprecated
### Removed
### Fixed

- Fixed a bug in motion_correct_wavelet affecting the selection of coefficients for IQR-based thresholding. The issue caused unintended suppression of high-frequency components, particularly near the end of recordings. Changed by [Eike Middell](https://github.com/emiddell).
- Fixed a bug in the stopping criterion of motion_correct.pca_recurse caused by an inverted 
boolean mask of motion artifacts, by [Eike Middell](https://github.com/emiddell).
- Fixed an issue with constant regressors when fitting a GLM using the AR-IRLS method. The autoregressive filter used to
account for serial correlations was not properly applied to them. The fix ignores samples at the beginning of the time
series until the filter is initialized, by [Eike Middell](https://github.com/emiddell).


## Version 25.1.0 (2025-06-22)

All dependencies have been updated to recent versions. Please rebuild the environment.

### New Features:

- Added Schaefer atlas-based parcel labels for ICBM152 and Colin27 head models via FreeSurfer surface mapping, by [Shakiba Moradi](https://github.com/shakiba93).
- Spatial and measurement noise regularization options in image reconstruction,  by [David Boas](https://github.com/dboas). ([#86](https://github.com/ibs-lab/cedalion/pull/86))
- Improved import of optode and electrode coordinates, by [Nils Harmening](https://github.com/harmening). ([#95](https://github.com/ibs-lab/cedalion/pull/95))
- The interfaces for the fluence and sensitivity computations were changed to allow out-of-core computations, by [Eike Middell](https://github.com/emiddell).
- Precomputed sensitivities for all example datasets, including the ninjaCap whole head probe, are availabe in cedalion.data, by [Eike Middell](https://github.com/emiddell).

- Make all example notebooks run on Google Colab integration, by [Josef Cutler](https://github.com/jccutler). ([#96](https://github.com/ibs-lab/cedalion/pull/96))

- Added functionality to add synthetic HRFs to resting state data, 
  by [Thomas Fischer](https://github.com/thomasfischer11). ([#77](https://github.com/ibs-lab/cedalion/pull/77))
- Added functionality to add synthetic artifacts to fNIRS data , by [Josef Cutler](https://github.com/jccutler).

- Added AMPD algorithm for heart beat detection from {cite:p}`Scholkmann2012`, by [Isa Musisi](https://github.com/isamusisi).
- Functionality for global physiology removal, by [Alexander von Lühmann](https://github.com/avolu). ([#106](https://github.com/ibs-lab/cedalion/pull/106))

- Multimodal source decomposition methods, including most CCA variants, by [Tomas Codina](https://github.com/TCodina). ([#102](https://github.com/ibs-lab/cedalion/pull/102))
- The interface to fit GLMs changed. The GLM solver is now based on statsmodels and we integrated the AR-IRLS algorithm, by [Ted Huppert](https://github.com/huppertt) and [Eike Middell](https://github.com/emiddell). ([#68](https://github.com/ibs-lab/cedalion/pull/68))
- Added wavelet motion correction from {cite:p}`Molavi2012`, by [Josef Cutler](https://github.com/jccutler). ([#72](https://github.com/ibs-lab/cedalion/pull/72))

- New multi-view animated image reconstruction plots, by [David Boas](https://github.com/dboas) and [Alexander von Lühmann](https://github.com/avolu).
- Thresholding and visualizing probe sensitivity to brain parcels, by [Alexander von Lühmann](https://github.com/avolu).
- Improvements to the time-series plots, by [David Boas](https://github.com/dboas). ([#85](https://github.com/ibs-lab/cedalion/pull/85), [#108](https://github.com/ibs-lab/cedalion/pull/108))


### Bugfixes:
- Correct determination of Cz in LandmarksBuilder1010, by [Nils Harmening](https://github.com/harmening). ([#82](https://github.com/ibs-lab/cedalion/pull/82))




## Version 25.0.0 (2025-01-21)

- First named release with contributions from:
    - [Sung Ahn](https://github.com/ahns97)
    - [Jacqueline Behrendt](https://github.com/jackybehrendt12)
    - [David Boas](https://github.com/dboas)
    - [Laura Carlton](https://github.com/lauracarlton)
    - [Tomás Codina](https://github.com/TCodina)
    - [Josef Cutler](https://github.com/jccutler)
    - [Qianqian Fang](https://github.com/fangq)
    - [Thomas Fischer](https://github.com/thomasfischer11)
    - [Nils Harmening](https://github.com/harmening)
    - [Mariia Iudina](https://github.com/mashayu)
    - [Filip Jenko](https://github.com/FilipJenko)
    - [Eike Middell](https://github.com/emiddell)
    - [Shakiba Moradi](https://github.com/shakiba93)
    - [Alexander von Lühmann](https://github.com/avolu)
    
    
    
    
    
    
    
    
    
    
    
