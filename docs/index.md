# Cedalion

<!--
Landing page. Layout is built with sphinx-design (grids, cards, buttons) so that all
internal links are checked by Sphinx. Styling lives in docs/_static/css/landing.css.
The toctrees at the bottom are hidden: they only feed the sidebar navigation.
-->

::::{grid} 1 1 2 2
:gutter: 3
:class-container: cdl-landing cdl-hero

:::{grid-item}
:columns: 12 12 3 3

```{image} img/landing/cedalion_logo.svg
:alt: Cedalion logo
:class: cdl-hero-logo
```
:::

:::{grid-item}
:columns: 12 12 9 9

[Cedalion]{.cdl-hero-title}

A Python framework for the data-driven analysis of functional near-infrared
spectroscopy (fNIRS) and diffuse optical tomography (DOT) in naturalistic
environments. Developed by the
[Intelligent Biomedical Sensing (IBS) Lab](https://www.ibs-lab.com) with and for the
community. With special thanks to the [BOAS Lab at Boston University's Neurophotonics Center](https://sites.bu.edu/boas/) for countless contributions.

```{button-ref} getting_started/index
:ref-type: doc
:color: primary
:class: cdl-hero-button

Get started
```
:::
::::

```{note}
:class: cdl-version-note

You are reading the documentation for the latest development version.
Use the **version switcher** (top-left of the page) to select documentation
for a specific release that matches your installed package.
```

::::{grid} 2 3 6 6
:gutter: 3
:class-container: cdl-resources

:::{grid-item-card} GitHub
:img-top: img/landing/icons/github.png
:img-alt: GitHub
:link: https://github.com/ibs-lab/cedalion
:link-type: url
:text-align: center
:class-card: cdl-resource
:::

:::{grid-item-card} Forum
:img-top: img/landing/icons/forum.png
:img-alt: Community forum
:link: https://openfnirs.org/community/cedalion/
:link-type: url
:text-align: center
:class-card: cdl-resource
:::

:::{grid-item-card} Paper
:img-top: img/landing/icons/paper.png
:img-alt: Cedalion paper
:link: https://doi.org/10.1117/1.NPh.13.S3.S32602
:link-type: url
:text-align: center
:class-card: cdl-resource
:::

:::{grid-item-card} Tutorial Notebooks
:img-top: img/landing/icons/notebooks.png
:img-alt: Tutorial notebooks
:link: tutorial
:link-type: doc
:text-align: center
:class-card: cdl-resource
:::

:::{grid-item-card} Tutorial Videos
:img-top: img/landing/icons/videos.png
:img-alt: Tutorial videos
:link: tutorial_videos
:link-type: doc
:text-align: center
:class-card: cdl-resource
:::

:::{grid-item-card} IBS Lab
:img-top: img/landing/icons/ibs_lab.png
:img-alt: IBS Lab
:link: https://ibs-lab.com/cedalion
:link-type: url
:text-align: center
:class-card: cdl-resource
:::
::::

## Features at a glance

Cedalion covers the full fNIRS and DOT analysis pipeline. Click a panel to jump to
the matching documentation page or example notebook.

<!--
Clickable feature map: each [{doc}`...`]{.fm-...} span is a Sphinx-checked link that
landing.css positions over its panel in cedalion_features.png (1974 x 994 px).
-->

:::{container} cdl-feature-map

```{image} img/landing/cedalion_features.png
:alt: Overview of Cedalion's features: preprocessing and GLM, multimodal machine learning, head models and atlases, diffuse optical tomography, and realistic data augmentation.
```

[{doc}`Preprocessing & GLM <sigproc/index>`]{.fm-hs .fm-preprocessing-glm}
[{doc}`Photogrammetric coregistration <examples/head_models/41_photogrammetric_optode_coregistration>`]{.fm-hs .fm-photogrammetric-coregistration}
[{doc}`Channel quality <examples/signal_quality/21_data_quality_and_pruning>`]{.fm-hs .fm-channel-quality}
[{doc}`Artifact rejection <examples/signal_quality/22_motion_artefacts_and_correction>`]{.fm-hs .fm-artifact-rejection}
[{doc}`Filtering & detrending <examples/tutorial/3_signal_processing>`]{.fm-hs .fm-filtering-detrending}
[{doc}`Modified Beer-Lambert law <examples/getting_started_io/10_xarray_datastructs_fnirs>`]{.fm-hs .fm-modified-beer-lambert-law}
[{doc}`General linear model <examples/modeling/32_glm_fingertapping_example>`]{.fm-hs .fm-general-linear-model}
[{doc}`Multimodal machine learning <machine_learning/index>`]{.fm-hs .fm-multimodal-machine-learning}
[{doc}`Sensor fusion <examples/machine_learning/51_multimodal_source_decomposition_on_synthetic_fNIRS-EEG_data>`]{.fm-hs .fm-sensor-fusion}
[{doc}`Latent component analysis <examples/machine_learning/52_ica_erbm_fingertapping_example>`]{.fm-hs .fm-latent-component-analysis}
[{doc}`Feature extraction <examples/machine_learning/50b_advanced_finger_tapping_lda_classification>`]{.fm-hs .fm-feature-extraction}
[{doc}`Classification <examples/machine_learning/50_finger_tapping_lda_classification>`]{.fm-hs .fm-classification}
[{doc}`Head models <examples/head_models/43a_head_models_overview>`]{.fm-hs .fm-head-models}
[{doc}`Parcellation & networks <examples/head_models/45_parcel_sensitivity>`]{.fm-hs .fm-parcellation-networks}
[{doc}`Head models & atlases <dot/index>`]{.fm-hs .fm-head-models-atlases}
[{doc}`Photon simulation <examples/tutorial/1_heads_and_fwm>`]{.fm-hs .fm-photon-simulation}
[{doc}`Image reconstruction <examples/head_models/40_image_reconstruction>`]{.fm-hs .fm-image-reconstruction}
[{doc}`Diffuse optical tomography <dot/index>`]{.fm-hs .fm-diffuse-optical-tomography}
[{doc}`Synthetic HRFs <examples/augmentation/62_synthetic_hrfs_example>`]{.fm-hs .fm-synthetic-hrfs}
[{doc}`Synthetic artifacts <examples/augmentation/61_synthetic_artifacts_example>`]{.fm-hs .fm-synthetic-artifacts}
[{doc}`Realistic data augmentation <synth/index>`]{.fm-hs .fm-realistic-data-augmentation}
:::

## Where to find what

::::{grid} 1 2 4 4
:gutter: 3
:class-container: cdl-paths

:::{grid-item-card} 1 · Get started
:class-card: cdl-path

- {doc}`Installation <getting_started/installation>`
- {doc}`Google Colab setup <getting_started/colab_setup>`
- {doc}`Quick start <getting_started/quickstart>`
- {doc}`Core concepts <concepts>`
- {doc}`Data structures <data_structures/index>`
:::

:::{grid-item-card} 2 · Learn
:class-card: cdl-path

- {doc}`Tutorial notebooks <tutorial>`
- {doc}`Tutorial videos <tutorial_videos>`
- [Tutorial paper](https://doi.org/10.1117/1.NPh.13.S3.S32602)
- {doc}`All example notebooks <examples>`
- {doc}`Rationale & design goals <rationale>`
:::

:::{grid-item-card} 3 · Analyse
:class-card: cdl-path

- {doc}`Data structures & I/O <data_io/index>`
- {doc}`Signal processing <sigproc/index>`
- {doc}`Modeling & machine learning <machine_learning/index>`
- {doc}`Diffuse optical tomography <dot/index>`
- {doc}`Physiology <physio/index>`
- {doc}`Plotting & visualization <plot_vis/index>`
- {doc}`Synthetic data <synth/index>`
:::

:::{grid-item-card} 4 · Look up & contribute
:class-card: cdl-path

- {doc}`API reference <api/modules>`
- {doc}`Bibliography <references>`
- {doc}`Changelog <CHANGELOG>`
- {doc}`Community <community/index>`
- {doc}`Contributing code <getting_started/contributing_code/contributing_code>`
- [Report an issue](https://github.com/ibs-lab/cedalion/issues)
:::
::::

## What is Cedalion?

Cedalion covers the full fNIRS and DOT analysis pipeline: from raw light intensity
through signal quality assessment, preprocessing, and hemodynamic modeling to image
reconstruction and statistical inference. It is designed for researchers who want
transparent, reproducible, and extensible analysis rather than black-box processing.

The toolbox is built on a modern Python scientific stack —
[xarray](https://xarray.dev) for labeled multi-dimensional arrays,
[pint](https://pint.readthedocs.io) for physical unit tracking, and
[MNE](https://mne.tools) for EEG/fNIRS integration — and is compatible with the
[SNIRF](https://github.com/fNIRS/snirf) and [BIDS](https://bids.neuroimaging.io)
data standards. All data transformations preserve axis labels and physical units
by design, making it straightforward to trace results back to their neuroimaging origin.

## Quick Start

The following snippet loads a bundled finger-tapping dataset and converts amplitude
measurements to haemoglobin concentration in five lines:

```python
import cedalion
import cedalion.nirs.cw as nirs
import xarray as xr

rec = cedalion.data.get_fingertapping()          # load Recording (auto-downloaded)
od  = nirs.int2od(rec["amp"])                    # amplitude → optical density
dpf = xr.DataArray([6.0, 6.0], dims="wavelength",
                   coords={"wavelength": od.wavelength})
conc = nirs.od2conc(od, rec.geo3d, dpf)          # OD → HbO / HbR (µM)
```

## How to cite

If you use Cedalion in your research, please cite the tutorial paper:

> Middell, E., Carlton, L. B., Moradi, S., Fischer, T., Cutler, J., Kelley, S. M.,
> Behrendt, J., Dissanayake, T., Yücel, M. A., Boas, D. A., & von Lühmann, A. (2026).
> Cedalion tutorial: A Python-based framework for comprehensive analysis of
> multimodal fNIRS and DOT from the lab to the everyday world.
> *Neurophotonics*, 13(S3), 1–27.
> [https://doi.org/10.1117/1.NPh.13.S3.S32602](https://doi.org/10.1117/1.NPh.13.S3.S32602)

```bibtex
@article{Middell2026,
  author  = {Middell, Eike and Carlton, Laura B. and Moradi, Shakiba and
             Fischer, Thomas and Cutler, Josef and Kelley, Shannon M. and
             Behrendt, Jacqueline and Dissanayake, Theekshana and
             Y{\"u}cel, Meryem A. and Boas, David A. and von L{\"u}hmann, Alexander},
  title   = {Cedalion tutorial: A {Python}-based framework for comprehensive
             analysis of multimodal {fNIRS} and {DOT} from the lab to the
             everyday world},
  journal = {Neurophotonics},
  volume  = {13},
  number  = {S3},
  pages   = {1--27},
  year    = {2026},
  doi     = {10.1117/1.NPh.13.S3.S32602}
}
```

Many functions implement methods from other publications. Their docstrings link to
the original papers, collected in the {doc}`bibliography <references>`; please cite
those as well when you use them.

## Partners and Funding

::::{grid} 2 3 6 6
:gutter: 4
:class-container: cdl-logos

:::{grid-item}
:child-align: center

```{image} img/landing/logos/ibs.png
:alt: Intelligent Biomedical Sensing (IBS) Lab
:target: https://www.ibs-lab.com/
```
:::

:::{grid-item}
:child-align: center

```{image} img/landing/logos/tu_berlin.png
:alt: Technische Universität Berlin
:target: https://www.tu.berlin/
```
:::

:::{grid-item}
:child-align: center

```{image} img/landing/logos/bifold.png
:alt: Berlin Institute for the Foundations of Learning and Data (BIFOLD)
:target: https://www.bifold.berlin/
```
:::

:::{grid-item}
:child-align: center

```{image} img/landing/logos/bu_neurophotonics.png
:alt: Boston University Neurophotonics Center
:target: https://www.bu.edu/neurophotonics/
```
:::

:::{grid-item}
:child-align: center

```{image} img/landing/logos/erc.png
:alt: European Research Council (ERC)
:target: https://erc.europa.eu/homepage
```
:::

:::{grid-item}
:child-align: center

```{image} img/landing/logos/bmftr.png
:alt: Federal Ministry of Research, Technology and Space (BMFTR)
:target: https://www.bmftr.bund.de/EN/Home/home_node.html
```
:::
::::

## Version
This documentation was built from commit {{commit_hash}}.

```{toctree}
:maxdepth: 1
:caption: General Info
:hidden:

rationale.md
getting_started/index.md
data_structures/index.md

community/index.md
../LICENSE.md
```

```{toctree}
:maxdepth: 1
:caption: ⭐ Tutorial
:hidden:

Paper <https://www.spiedigitallibrary.org/journals/neurophotonics/volume-13/issue-S3/S32602/Cedalion-tutorial--a-Python-based-framework-for-comprehensive-analysis/10.1117/1.NPh.13.S3.S32602.full>
Tutorial Notebooks <tutorial.rst>
Tutorial Videos <tutorial_videos.rst>
```

```{toctree}
:maxdepth: 1
:caption: Package Features
:hidden:

data_io/index
sigproc/index
machine_learning/index
dot/index
geometry/index
physio/index
plot_vis/index
synth/index
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

API reference <api/modules.rst>
Bibliography <references.rst>
All examples <examples>
```

```{toctree}
:maxdepth: 1
:caption: Project
:hidden:

Source code <https://github.com/ibs-lab/cedalion>
Issues <https://github.com/ibs-lab/cedalion/issues>
Documentation <https://doc.ibs.tu-berlin.de/cedalion/doc/dev/>
Changelog <CHANGELOG.md>
```
