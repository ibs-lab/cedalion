# Documentation: From individual MRI scan to cedalion headmodel
**This documentation explains the workflow for MRI image segmentation for subsequent use within Cedalion.**

The Cedalion head model workflows, as exemplarily described by the jupyter notebooks [40_image_reconstruction.ipynb](https://github.com/ibs-lab/cedalion/blob/main/examples/head_models/40_image_reconstruction.ipynb), [41_photogrammetric_optode_coregistration.ipynb](https://github.com/ibs-lab/cedalion/blob/main/examples/head_models/41_photogrammetric_optode_coregistration.ipynb), [42_1010_system.ipynb](https://github.com/ibs-lab/cedalion/blob/main/examples/head_models/42_1010_system.ipynb), [43_crs_and_headmodel.ipynb](https://github.com/ibs-lab/cedalion/blob/main/examples/head_models/43_crs_and_headmodel.ipynb), require either standard or individual head models, built from raw MRI images through a series of segmentation, post-processing and brain surface extraction steps.<br>
This document details the complete processing pipeline for creating cedalion head models from MRI images. The pipeline integrates multiple tools outside Cedalion / Python, including Brainstorm, SPM, CAT12, and FreeSurfer.

## Step 1: T1 Image Segmentation
MRI segmentation is the process of partitioning an MRI scan into the different biological tissue types. The neuroscience community so far developed various powerful tools to segment a T1-weighted MRI image that all exhibit different strengths and weaknesses. We present three of them here shortly, where each of them is suffient. However, we usually do the segmentation with using both *[Nils workflow](#A:-nils-workflow-with-spm)* and *[CAT12 in Brainstorm](#B:-cat12-in-brainstorm)* and choose the best one. For a detailed brain surface model we sometimes additionally add the *[FreeSurfer workflow](#C:-detailed-analysis-with-freesurfer)*.

### A: Nils Workflow with SPM
  Nils has a [repository (harmening/MRIsegmentation)](https://github.com/harmening/MRIsegmentation) which contains a automatic MATLAB workflow for MRI segmentation (using SPM and CAT12), postprocessing and mesh generation. The [README](https://github.com/harmening/MRIsegmentation/blob/master/README.md) contains all information about how to install and start the segmentation (running [part 2 - Tissue segmentation](https://github.com/harmening/MRIsegmentation/blob/master/README.md#tissue-segmentation) is sufficient).<br>
  **Output**:
  Post-processed 6-type tissue segmentation masks (`mask_air.nii`, `mask_skin.nii`, `mask_bone.nii`, `mask_csf.nii`, `mask_gray.nii`, `mask_white.nii`), that can be directly loaded into cedalion.

### B: CAT12 in Brainstorm

1. **Brainstorm Overview**:
  [Brainstorm](https://neuroimage.usc.edu/brainstorm/Introduction) is an open-source application designed for the analysis of brain recordings. It can be used as a standalone version or installed via MATLAB, and is integral to many neuroimaging workflows. It makes it easy to perform SPM, CAT12 and Freesurfer analysis.

2. **CAT12 Toolbox**:  
  [CAT12](https://neuro-jena.github.io/cat/) is a computational toolbox that extends the SPM (Statistical Parametric Mapping) software package and is primarily used for structural MRI (sMRI) analysis. CAT12 automates many preprocessing steps required for sMRI, including:
  - **Tissue segmentation** into six types (skull, scalp, cerebrospinal fluid (CSF), gray matter, white matter, and air).
  - **Normalization and modulation** of images.
  - **Surface generation** for brain and head models.

  **Output from CAT12 used in cedalion**:<br>
  CAT12’s outputted 6-type tissue segmentations (skull, scalp, CSF, gray matter, white matter, and air) and the generated head surface.


  
### C: Detailed Analysis with FreeSurfer

1. **FreeSurfer Overview**:  
  [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/) is a widely-used software package in neuroimaging, specifically for MRI structural analysis. Known for its precise and reproducible methods, FreeSurfer is used to segment brain structures, reconstruct cortical surfaces, and quantify cortical thickness, surface area, and volume.<br>
  
2. **Run Segmentation**:  
  [Download and install FreeSurfer](https://surfer.nmr.mgh.harvard.edu/fswiki/DownloadAndInstall). To do the segmentation run
```
# Define your FREESURFER_HOME environment variable (if you havn't done already during installation)
export FREESURFER_HOME=/path/to/FreeSurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# Tell FreeSurfer where to put the segmentation results
export SUBJECTS_DIR=/path/to/your/subjects_dir

# Start segmentation
my_subject=sample
my_NIfTI=/path/to/NIfTI.nii.gz
recon-all -i $my_NIfTI -s $my_subject -all
```

3. **Processing Outputs**:
  In this workflow, FreeSurfer provides details including:
  - **Cortical and subcortical segmentation**
  - **Gray and white matter segmentation**
  - **Detailed brain surface model**

  **Key Outputs from FreeSurfer**: Cedalion’s workflow uses FreeSurfer’s gray matter and white matter segmentations and its detailed brain surface model.
  

## Step 2: Parcellation using Schaefer Atlas
  The Schaefer Atlas is a popular parcellation scheme for the human brain used in neuroimaging research. It was developed by Schaefer et al. (2018) and provides a fine-grained parcellation of the cortex based on functional connectivity data from resting-state fMRI. One of the distinctive features of the Schaefer Atlas is its organization of brain regions into well-defined networks that reflect patterns of correlated brain activity.
  The atlas offers two primary options for network parcellation:

  1. Schaefer Atlas with 7 Networks
     This version divides the brain into 7 broad functional networks. Often used in studies where the focus is on high-level brain networks, such as understanding large-scale brain organization, general connectivity patterns, or when simplifying data for group analyses.

  2. Schaefer Atlas with 17 Networks
     This version provides a more granular division of the brain into 17 functional networks. These include the original 7 networks, but with further subdivisions. Suitable for studies that require more detailed parcellation to capture subtle differences in brain function, such as exploring intra-network connectivity or specific functional regions within the brain's broader networks.

  Parcellations are computed in Freesurfer.

## Step 3: Alignment and Optimization in Brainstorm

1. **Data Alignment**:  
  To ensure consistency, outputs from CAT12 and FreeSurfer are loaded into Brainstorm, where tissue segmentation files and surfaces are aligned to MNI coordinates. This step is crucial for integrating data accurately within Cedalion’s models.

2. **Mesh Optimization**:  
  Since FreeSurfer surfaces contain ~300K vertices, we downsample the mesh in Brainstorm to a more manageable 15K vertices, which balances detail and processing efficiency in Cedalion.

## Step 4: Post-Processing of Tissue Segmentation Masks

1. **Post-Processing Workflow**:  
  To finalize tissue segmentation masks, Cedalion applies additional post-processing steps to smoothen boundaries and fill small gaps. These operations ensure cleaner and more continuous tissue delineation.

2. **Preservation of Key Segmentations**:
  - **Gray and White Matter**: FreeSurfer’s gray and white matter segmentations are retained without modification due to their high accuracy. Final checks are applied to prevent overlap between masks and to ensure clarity in tissue delineation.

## Final Outputs

For each T1 image, the workflow yields the following:

- **6 Tissue Segmentation Masks**: skull, scalp, CSF, gray matter, white matter, and air (air mask optional).
- **Head Surface and Brain Surface**
- **Parcellations json file**

Note that the brain surface, along with gray and white matter masks, originates from FreeSurfer. All other segmentation masks and the head surface are generated from CAT12.

The flowchart below provides a visual summary of this processing pipeline.
