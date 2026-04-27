# Changelog

## Unreleased changes (available on the `dev` branch)
 
## Version 26.04.0 (in preparation)

### Additions and Changes

#### Structure and Renaming:

- The package `cedalion.sigproc.motion_correct` was renamed to `cedalion.sigproc.motion`.

- The ICA-EBM and ICA_ERBM implementations were moved into `cedalion.sigdecomp.unimodal`.

- Refactored `cedalion.plots` into `cedalion.vis` and its subpackages. This cleans up the code structure and should help with discovering existing functions. The package `cedalion.vis.blocks` emphasizes building blocks for larger visualizations. Please refer to `examples/plots_visualization/12_plots_example.ipynb` to get an overview. Importing `cedalion.plots` will throw a deprecation warning to trigger adoption. By [Eike Middell](https://github.com/emiddell).

- Renamed `LabeledPointCloud` to `LabeledPoints`.

- Split up the `.nirs` submodule into `.nirs.cw`, `.nirs.fd` and `.nirs.td`. 

- Merged the submodules `cedalion.datasets` and `cedalion.data`. All functions to access example datasets are now available under `cedalion.data`. 

- Changed the names of several motion correction algorithms from `motion_correct.motion_correct_X` 
to `motion_correct.X`. Argument names were made PEP8 compliant. The example `22_motion_artefacts_and_correction` was improved. By [Eike Middell](https://github.com/emiddell).



#### Head Models & Registration

- Added `TwoSurfaceHeadmodel.scale_to_headsize` and `TwoSurfaceHeadmodel.scale_to_landmarks` to adjust the head model's size to the head circumferences or digitized landmarks, respectively. By [Eike Middell](https://github.com/emiddell).

- Higher-resolution cortex meshes for the Colin27 and ICBM152 head models, which additionally maintain a link to the freesurfer surfaces from which they were derived, by [Eike Middell](https://github.com/emiddell). ([#138](https://github.com/ibs-lab/cedalion/pull/138)). The Colin27 meshes were subsequently recomputed to remove artifacts and to fix the voxel-to-vertex mapping. Reduced meshes of the FreeSurfer inflated brains are now bundled with a 1-to-1 vertex correspondence to the pial brain meshes, and sparse voxel-to-vertex maps are stored in Matrix Market format.

- The factory method `cedalion.dot.get_standard_headmodel` to construct the `TwoSurfaceHeadModel` of the standard Colin27 and ICBM-152 heads was added, by [Eike Middell](https://github.com/emiddell).

- Added the parameter `mode` to `TwoSurfaceHeadModel.align_and_snap_to_scalp` to switch
  between constrained affine transformations (translation, rotation, isotropic scaling)
  and unconstrained affine transformations (also anisotropic scaling, shearing and reflection). The default behaviour from `align_and_snap_to_scalp` was changed from
  constrained to unconstrained affine transformations. 

- The fiducial landmarks of the standard Colin27 and ICBM-152 heads changed. Using the landmark builder 10-10 coordinates were calculated for both heads. These are now 
  available when head models are created with `get_standard_headmodel`. The example
  notebook `48_headmodel_landmarks_verification.ipynb` documents the origin of all 
  landmarksand quantitatively compares the output of the LandmarkBuilder to published
  coordinates.  By [Eike Middell](https://github.com/emiddell).

- For the Colin27 and ICBM-152 heads the label for the coordinate reference system was changed from `'aligned'` to `'mni'`.

- Added `cedalion.geometry.landmarks.normalize_landmarks_labels` to map alternative landmark names (e.g., "nasion", "left ear", "nz") to their canonical 10-10 system labels (e.g. Nz, LPA). The function handles now case-insensitive matching and supports common naming conventions. Usage: `geo3d = normalize_landmarks_labels(geo3d)` before calling registration or plotting functions, by [Mohammad Orabe](https://github.com/orabe). ([#84](https://github.com/ibs-lab/cedalion/issues/84), [#132](https://github.com/ibs-lab/cedalion/pull/132))

- The ninja HD and UHD cap coordinates were changed from a left-handed to a right-handed coordinate system, by [Nils Harmening](https://github.com/harmening). ([#110](https://github.com/ibs-lab/cedalion/pull/110))

#### Image Reconstruction

- The image reconstruction methods were refactored to offer more regularization methods     
  (including spatial basis functions) as well as direct and indirect reconstructions and to provide a simpler interface to the user. All functions are now located under `cedalion.dot`. By [Laura Carlton](https://github.com/lauracarlton), [Alexander von Lühmann](https://github.com/avolu) and [Eike Middell](https://github.com/emiddell). ([#130](https://github.com/ibs-lab/cedalion/pull/86))

- Added `cedalion.dot.ImageRecon.get_image_noise_posterior`, by [Laura Carlton](https://github.com/lauracarlton). ([#134](https://github.com/ibs-lab/cedalion/pull/134))

- The class `cedalion.dot.ForwardModel` accepts also head models that are not in voxel space. They will be transformed to voxel space internally.

- The normalization of the `dot.GaussianSpatialBasisFunctions` was changed to match the original implementation used in {cite:p}`Carlton2026`.

#### GLM

- Generalized the GLM fit routine and regressors to work not only in channel space but also on time series with other spatial dimensions (vertices, parcels,...). Center higher-order drift regressors in `design_matrix.drift_regressors`. By [Miray Altinkaynak](https://github.com/maltink) and [Reihaneh Taghizadegan](https://github.com/ReiGaan). ([#120](https://github.com/ibs-lab/cedalion/pull/120), [#142](https://github.com/ibs-lab/cedalion/issues/142)) 

- Previously, HRF regressors in the GLM where constructed from normalized basis functions and were normalized again after they were convolved over the stimulus duration. Now, only the basis functions are normalized to 1.  ([#139](https://github.com/ibs-lab/cedalion/pull/138)).


#### Signal Processing

- Added functionality and examples for constrained ICA methods (arc-ERBM, arc-EBM),  by [Jacqueline Behrendt](https://github.com/jackybehrendt12). ([#133](https://github.com/ibs-lab/cedalion/pull/133))

- An example notebook for ICA source extraction was added, by [Jacqueline Behrendt](https://github.com/jackybehrendt12). 
([#112](https://github.com/ibs-lab/cedalion/pull/112))



#### Visualization

- The function `cedalion.vis.anatomy.plot_montage3D` now accepts a `landmarks` parameter to specify which landmarks should be highlighted. Pass `None` (default) to show all available canonical registration landmarks (e.g. Nz, Iz, LPA, RPA, Cz), a list of landmark names to show specific ones, or an empty list to show none, by [Mohammad Orabe](https://github.com/orabe). ([#84](https://github.com/ibs-lab/cedalion/issues/84))

- Included t-stat thresholding in `cedalion.vis.misc.plot_probe_gui`, by [Shannon Kelley](https://github.com/shankell212). ([#131](https://github.com/ibs-lab/cedalion/pull/131))

- Added the option `draw_arcs` to `cedalion.vis.anatomy.scalp_plot`. If set to True, channels are drawn as curved lines to reduce overlap, by [Eike Middell](https://github.com/emiddell).

- Added channel plotting (lines between source-detector pairs) to `cedalion.vis.blocks.plot_labeled_points`, by [Nils Harmening](https://github.com/harmening). ([#111](https://github.com/ibs-lab/cedalion/pull/111))

- Enhanced landmark picking in `cedalion.vis.blocks.plot_surface`: callers can now choose which landmarks to pick, and the picked landmarks are returned as an `xr.DataArray`, by [Nils Harmening](https://github.com/harmening). ([#126](https://github.com/ibs-lab/cedalion/pull/126))

#### Utilities

- Added `cedalion.xrutils.dot_dataarray_csr` for matrix products between `xr.DataArray` 
  and `scipy.sparse` arrays, by [Eike Middell](https://github.com/emiddell).

- Added a `Dockerfile` to build a containerised cedalion environment, by [Nils Harmening](https://github.com/harmening). ([#6](https://github.com/ibs-lab/cedalion/pull/6))


### Fixes

- Fixed a bug in motion_correct_wavelet affecting the selection of coefficients for IQR-based thresholding. The issue caused unintended suppression of high-frequency components, particularly near the end of recordings. Changed by [Eike Middell](https://github.com/emiddell).
- Fixed a bug in the stopping criterion of motion_correct.pca_recurse caused by an inverted 
boolean mask of motion artifacts, by [Eike Middell](https://github.com/emiddell).
- Fixed an issue with constant regressors when fitting a GLM using the AR-IRLS method. The autoregressive filter used to
account for serial correlations was not properly applied to them. The fix ignores samples at the beginning of the time
series until the filter is initialized, by [Eike Middell](https://github.com/emiddell).
- Fixed the labels assigned by landmark picking in `cedalion.vis.blocks.plot_surface`, by [Nils Harmening](https://github.com/harmening). ([#126](https://github.com/ibs-lab/cedalion/pull/126))
- Fixed `cedalion.io.read_photogrammetry_einstar` to filter out unpicked positions, by [Nils Harmening](https://github.com/harmening). ([#144](https://github.com/ibs-lab/cedalion/pull/144))
- Removed a redundant `t_ras2ijk` transform when saving and loading `TwoSurfaceHeadModel`s, by [Nils Harmening](https://github.com/harmening). ([#143](https://github.com/ibs-lab/cedalion/pull/143))
- Fixed `TwoSurfaceHeadModel.__repr__` raising when `landmarks` was `None`, by [Eike Middell](https://github.com/emiddell).




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
    
    
    
    
    
    
    
    
    
    
    
