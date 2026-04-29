# Rationale and Design Goals

## Why Cedalion?

The fNIRS community has built a rich ecosystem of analysis toolboxes, largely in MATLAB. Cedalion
complements these tools from a different starting point: what would an fNIRS toolbox
look like if it were designed from the ground up around the best practices of modern
scientific Python — labeled arrays, physical units, composable pipelines, and native
integration with machine learning libraries?

The motivation comes in part from a growing demand for analyses that span the full
pipeline in a single scriptable environment, and from the expansion of fNIRS into
naturalistic, everyday settings. Wearable and mobile systems generate large,
heterogeneous datasets that benefit from rigorous data-driven quality control and
flexible preprocessing that can adapt to variable recording conditions.

Our view of what constitutes a good answer drives three design goals.

### 1. Labels and units travel with the data

A recurring source of bugs and misinterpretations in signal processing is the loss of
metadata: a numpy array arrives at a function and nobody knows which axis is time, which
is channel, or what the values' physical units are. Cedalion uses
[xarray](https://xarray.dev) DataArrays as its primary data container. Every array
carries named dimensions (`channel`, `wavelength`, `time`, …), coordinate labels, and
— via [pint-xarray](https://pint-xarray.readthedocs.io) — physical units. Operations
that would silently drop this context raise an error instead, turning a class of latent
bugs into immediate, informative failures.

### 2. Pipelines are composable and inspectable

Cedalion separates the *data model* from the *processing steps*. The `Recording`
container holds all data objects (timeseries, masks, probe geometry, stimulus table,
head model) in named dictionaries. Processing functions are ordinary Python callables
that take and return arrays or update a `Recording` in-place. There are no hidden
pipeline objects, global state, or implicit parameter inheritance: every step is a
regular function call whose inputs and outputs can be inspected, plotted, or serialised
at any point.

### 3. Open standards, reproducibility, and the FAIR principles

Raw data is read and written in [SNIRF](https://github.com/fNIRS/snirf) format, and
datasets can be organised as [BIDS](https://bids.neuroimaging.io)-compliant archives,
making recordings Findable, Accessible, Interoperable, and Reusable (FAIR). Cedalion aims to integrate with [MNE](https://mne.tools) `Raw` to leverage  MNE's extensive EEG and MEG processing routines without leaving the Python
ecosystem. scikit-learn estimators accept Cedalion DataArrays directly, behaving like
numpy arrays while preserving coordinate metadata. This means you can enter and exit
the toolbox at any stage of an analysis, and entire pipelines — from raw data to
statistical results — can be shared as self-contained, runnable notebooks.

## Target audience

Cedalion is aimed at researchers who:

- want to write analysis code that is readable, reproducible, and shareable;
- are comfortable with Python and the scientific stack (numpy, matplotlib, pandas)
  but do not want to re-implement standard fNIRS building blocks;
- need capabilities that span the full pipeline — from raw data to image
  reconstruction and machine learning — without switching tools;
- work with naturalistic or mobile recordings where data quality is variable and
  rigorous quality control is essential.

## Relation to other toolboxes

The MATLAB-based toolboxes listed below represent the most widely used prior work.
Cedalion is not a replacement for any of them; it fills a gap for Python-first
workflows that combine multi-step preprocessing, rigorous quality control, DOT image
reconstruction, and data-driven methods in a single, scriptable environment.

| Toolbox | Primary focus | Language |
|---------|--------------|----------|
| Homer2 / Homer3 | Preprocessing, GLM | MATLAB |
| AtlasViewer | 3D probe registration, DOT | MATLAB |
| Brain AnalyzIR | Statistical inference, GLM | MATLAB |
| NeuroDOT | Preprocessing, GLM, DOT image reconstruction | MATLAB |
| NIRStorm | Multi-modal source imaging | MATLAB |
| MNE-NIRS | SNIRF I/O, MNE integration | Python |
| Cedalion | Full pipeline, DOT, data-driven ML | Python |

## Community and reproducibility

Cedalion is developed as a community resource. Each processing function is linked to
its source publication so that algorithm choices are traceable to the primary
literature. New contributions are reviewed on GitHub and follow FAIR (Findable,
Accessible, Interoperable, Reusable) data principles.

## Project status

Cedalion is developed openly at the
[IBS Lab, TU Berlin](https://www.ibs-lab.com) in collaboration with Boston
University and released under the MIT licence. The project website is
[www.cedalion.tools](http://www.cedalion.tools).

Contributions are welcome — see the [contributing guide](getting_started/contributing_code/contributing_code.md).
If you use Cedalion in published work, please cite:

> Middell, E., Carlton, L., Moradi, S., Codina, T., Fischer, T., Cutler, J., Kelley, S., Behrendt, J., Dissanayake, T., Harmening, N., Yücel, M. A., Boas, D. A., & von Lühmann, A. (2026). Cedalion Tutorial: A Python-based framework for comprehensive analysis of multimodal fNIRS &amp; DOT from the lab to the everyday world (Version 1). arXiv. https://doi.org/10.48550/ARXIV.2601.05923



