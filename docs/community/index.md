# Community

Cedalion is an open-source project developed *from the community for the
community*. It was initiated at the [IBS Lab](https://www.ibs-lab.com) at
TU Berlin with the goal of building a transparent, reproducible, and extensible
fNIRS/DOT analysis platform that the whole community can rely on, contribute to,
and improve over time.

## Philosophy

Our community philosophy rests on three pillars:

**Open development.** All code, issues, and discussions are public. Every
contribution is welcome — from fixing a typo to implementing a new analysis
method.

**Attribution.** Code contributions are credited in function docstrings and in
the [Contributors](#contributors) section below. Methods implemented from
scientific publications are linked to their source paper via a
[searchable bibliography](../references.rst), giving the original authors the
visibility they deserve.

**Interoperability.** Cedalion adopts open data standards
([SNIRF](https://github.com/fNIRS/snirf), [BIDS](https://bids.neuroimaging.io))
and interfaces with widely-used toolboxes across the community rather than
reinventing the wheel. See [Toolboxes & Integrations](#toolboxes-integrations)
below for the full picture.

More detail on the design goals and community approach can be found in the
accompanying paper: :cite:t:`Middell2026`.

## Toolboxes & Integrations

[Cedalion's name is based on a Greek myth](https://en.wikipedia.org/wiki/Cedalion):
Cedalion stood on the shoulders of the giant Orion to guide him east, where the
rays of Helios restored his sight. This toolbox stands on the shoulders of many
giants, and we aim to complement existing toolboxes in the community by
interfacing with them wherever possible.

### Origins

A substantial part of Cedalion's fNIRS & DOT signal processing and head
modelling functionality traces its roots to the MATLAB toolboxes
[Homer2/3](https://github.com/BUNPC/Homer3) and
[AtlasViewer](https://github.com/BUNPC/AtlasViewer) from the Boston University
Neurophotonics Center (:cite:t:`Huppert2009`). Methods from these toolboxes
have been translated into Python and integrated into the Cedalion ecosystem with
full citation linkage.

### Active Integrations

| Toolbox | Purpose in Cedalion | Relationship |
|---|---|---|
| [Homer2/3](https://github.com/BUNPC/Homer3) | Origin of signal processing and head modelling methods | Predecessor / method source |
| [AtlasViewer](https://github.com/BUNPC/AtlasViewer) | Origin of atlas-based DOT pipeline methods | Predecessor / method source |
| [MCX / MCXCL](http://mcx.space/) | GPU-accelerated Monte Carlo photon simulation | Optional dependency (`cedalion.dot`) |
| [NIRFASTer](https://github.com/nirfaster/NIRFASTer) | FEM-based photon simulation | Plugin (`plugins/nirfaster`) |
| [MNE-Python](https://mne.tools) | SNIRF I/O compatibility; geometry utilities | Soft dependency |

### Related Toolboxes in the fNIRS/DOT Ecosystem

Cedalion is part of a broader ecosystem of fNIRS and DOT analysis tools. We aim
to complement rather than compete with these projects; cross-toolbox
compatibility via SNIRF and BIDS is a priority.

| Toolbox | Language | Focus |
|---|---|---|
| [Homer2/3](https://github.com/BUNPC/Homer3) | MATLAB | General fNIRS analysis pipeline |
| [AtlasViewer](https://github.com/BUNPC/AtlasViewer) | MATLAB | fNIRS head modeling and DOT |
| [Brain AnalyzIR](https://github.com/huppertt/nirs-toolbox) | MATLAB | Statistical fNIRS analysis |
| [NeuroDOT](https://github.com/WUSTL-ORL/NeuroDOT) | MATLAB | High-density DOT |
| [NIRStorm](https://neuroimage.usc.edu/brainstorm/Plugins/NIRStorm) | MATLAB (Brainstorm) | fNIRS in Brainstorm |
| [MNE-NIRS](https://mne.tools/mne-nirs) | Python | fNIRS extension for MNE-Python |

For a broader overview of available fNIRS software, see
[openfnirs.org/software](https://openfnirs.org/software/).

### Planned Integrations

- [NeuroKit2](https://github.com/neuropsychology/NeuroKit): physiological signal
  processing (cardiac, respiratory, EDA) to complement fNIRS in multimodal
  recordings.

## Contributors

Cedalion is driven by the [IBS Lab](https://www.ibs-lab.com) with the aim of
encouraging continuous use, contribution, and improvement from the entire
community. Every code contribution is credited here and in the relevant function
docstrings. The contributor list below is generated automatically from the
[GitHub repository](https://github.com/ibs-lab/cedalion) and can be refreshed by
running `python scripts/generate_contributors.py`.

<!-- BEGIN CONTRIBUTORS -->
% AUTO-GENERATED — do not edit by hand.
% Regenerate with: python scripts/generate_contributors.py
% Last updated: 2026-04-29

### Core Maintainers

````{raw} html
<div class="contributor-grid">
<div class="contributor-card maintainer">
  <a href="https://github.com/emiddell" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/emiddell?s=100" alt="Eike Middell" loading="lazy"/>
    <div class="contributor-name">Eike Middell</div>
    <div class="contributor-commits">304 commits</div>
  </a>
</div>
<div class="contributor-card maintainer">
  <a href="https://github.com/avolu" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/avolu?s=100" alt="Alexander von Lühmann" loading="lazy"/>
    <div class="contributor-name">Alexander von Lühmann</div>
    <div class="contributor-commits">170 commits</div>
  </a>
</div>
<div class="contributor-card maintainer">
  <a href="https://github.com/dboas" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/dboas?s=100" alt="David Boas" loading="lazy"/>
    <div class="contributor-name">David Boas</div>
    <div class="contributor-commits">15 commits</div>
  </a>
</div>
</div>
````

### Code Contributors

````{raw} html
<div class="contributor-grid">
<div class="contributor-card">
  <a href="https://github.com/lauracarlton" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/lauracarlton?s=100" alt="Laura Carlton" loading="lazy"/>
    <div class="contributor-name">Laura Carlton</div>
    <div class="contributor-commits">65 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/jccutler" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/jccutler?s=100" alt="Josef Cutler" loading="lazy"/>
    <div class="contributor-name">Josef Cutler</div>
    <div class="contributor-commits">55 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/shakiba93" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/shakiba93?s=100" alt="Shakiba Moradi" loading="lazy"/>
    <div class="contributor-name">Shakiba Moradi</div>
    <div class="contributor-commits">27 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/harmening" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/harmening?s=100" alt="nils" loading="lazy"/>
    <div class="contributor-name">nils</div>
    <div class="contributor-commits">21 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/ahns97" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/ahns97?s=100" alt="Sung Min Ahn" loading="lazy"/>
    <div class="contributor-name">Sung Min Ahn</div>
    <div class="contributor-commits">18 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/thomasfischer11" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/thomasfischer11?s=100" alt="Thomas Fischer" loading="lazy"/>
    <div class="contributor-name">Thomas Fischer</div>
    <div class="contributor-commits">18 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/mashayu" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/mashayu?s=100" alt="Mariia" loading="lazy"/>
    <div class="contributor-name">Mariia</div>
    <div class="contributor-commits">7 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/jackybehrendt12" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/jackybehrendt12?s=100" alt="Jacky Behrendt" loading="lazy"/>
    <div class="contributor-name">Jacky Behrendt</div>
    <div class="contributor-commits">3 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/TCodina" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/TCodina?s=100" alt="Tomás Codina" loading="lazy"/>
    <div class="contributor-name">Tomás Codina</div>
    <div class="contributor-commits">3 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/isamusisi" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/isamusisi?s=100" alt="Isa Musisi" loading="lazy"/>
    <div class="contributor-name">Isa Musisi</div>
    <div class="contributor-commits">2 commits</div>
  </a>
</div>
<div class="contributor-card">
  <a href="https://github.com/fangq" target="_blank" rel="noopener">
    <img src="https://avatars.githubusercontent.com/fangq?s=100" alt="Qianqian Fang" loading="lazy"/>
    <div class="contributor-name">Qianqian Fang</div>
    <div class="contributor-commits">1 commit</div>
  </a>
</div>
</div>
````
<!-- END CONTRIBUTORS -->

### Scientific Credit

This documentation contains a dedicated
[bibliography](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/references.html) where you
can search for scientific papers whose methods are implemented in Cedalion. If you
contribute code based on a published method, please add the BibTeX entry to
`cedalion/bibliography/references.bib`, cite it in the function's docstring and call
`cedalion.cite()` within the function itself — this gives the original authors the
visibility they deserve.

### Special Mentions

- Special thanks to members of the Bio Optical & Acoustic Spectroscopy (BOAS)
  Lab at Boston University's Neurophotonics Center: Laura Carlton, Sung Min Ahn,
  Meryem Yücel, and David Boas.
- Thanks to Jiaming Cao from the University of Birmingham for supporting the
  adoption of [NIRFASTer](https://github.com/nirfaster/NIRFASTer) into Cedalion.

## Get Involved

### Forum

The primary place for questions, discussion, and community support is the
[Cedalion Forum on openfnirs.org](https://openfnirs.org/community/cedalion/).
This is the best channel for usage questions, sharing results, and general
fNIRS/DOT discussion.

### GitHub Issues

For bug reports and feature requests, use
[GitHub Issues](https://github.com/ibs-lab/cedalion/issues). Please search
existing issues before opening a new one. For usage questions, the forum is
more appropriate than issues.

### Contributing Code

Contributions of all kinds are welcome — bug fixes, new analysis methods,
documentation improvements, and example notebooks. A step-by-step guide to
setting up a development environment and submitting a pull request is available
here:

[Getting Started with Contributing Code](https://doc.ibs.tu-berlin.de/cedalion/doc/dev/getting_started/contributing_code/contributing_code.html)
