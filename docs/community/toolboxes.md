# Toolboxes

[Cedalion's name is based on a Greek myth](https://en.wikipedia.org/wiki/Cedalion):
Cedalion stood on the shoulders of the giant Orion to guide him east, where the rays
of Helios restored his sight. This toolbox stands on the shoulders of many giants, and
we aim to complement existing toolboxes in the community by interfacing with them
wherever possible.

## Origins

A substantial part of Cedalion's fNIRS & DOT signal processing and head modelling
functionality traces its roots to the MATLAB toolboxes
[Homer2/3](https://github.com/BUNPC/Homer3) and
[AtlasViewer](https://github.com/BUNPC/AtlasViewer) from the Boston University
Neurophotonics Center (:cite:t:`Huppert2009`). Methods from these toolboxes have been
translated into Python and integrated into the Cedalion ecosystem with full citation
linkage.

## Active Integrations

| Toolbox | Purpose in Cedalion | Relationship |
|---|---|---|
| [Homer2/3](https://github.com/BUNPC/Homer3) | Origin of signal processing and head modelling methods | Predecessor / method source |
| [AtlasViewer](https://github.com/BUNPC/AtlasViewer) | Origin of atlas-based DOT pipeline methods | Predecessor / method source |
| [MCX / MCXCL](http://mcx.space/) | GPU-accelerated Monte Carlo photon simulation | Optional dependency (`cedalion.dot`) |
| [NIRFASTer](https://github.com/nirfaster/NIRFASTer) | FEM-based photon simulation | Plugin (`plugins/nirfaster`) |
| [MNE-Python](https://mne.tools) | SNIRF I/O compatibility; geometry utilities | Soft dependency |

## Related Toolboxes in the fNIRS/DOT Ecosystem

Cedalion is part of a broader ecosystem of fNIRS and DOT analysis tools. We aim to
complement rather than compete with these projects; cross-toolbox compatibility via
SNIRF and BIDS is a priority.

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

## Planned Integrations

- [NeuroKit2](https://github.com/neuropsychology/NeuroKit): physiological signal
  processing (cardiac, respiratory, EDA) to complement fNIRS in multimodal recordings.
