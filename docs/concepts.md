# Core Concepts

This page gives a brief introduction to fNIRS physics and to the Cedalion data model.
It is intended for researchers who are new to either fNIRS or to Cedalion's API.

## fNIRS fundamentals

### What fNIRS measures

Functional near-infrared spectroscopy (fNIRS) measures haemodynamic responses to
neural activity by shining near-infrared light (typically 650–950 nm) through the
scalp and skull and detecting the remitted light at a short distance away. Oxy- and
deoxyhaemoglobin (HbO and HbR) absorb NIR light at different rates, so changes in
their concentrations change the detected intensity. Because neural activity drives local
increases in cerebral blood flow (the *haemodynamic response*), fNIRS can indirectly
track brain function non-invasively.

### Optodes, sources, detectors, and channels

- **Source**: a light emitter; typically emits two wavelengths (e.g. 760 nm and 850 nm).
- **Detector**: a photodetector; records the light intensity arriving from all nearby sources.
- **Channel**: a source–detector pair. A single detector can form channels with multiple
  sources, and vice versa, so a montage with *S* sources and *D* detectors may have up to
  *S × D* channels.
- **Optode**: a collective term for sources and detectors.

Channel distance (source–detector separation) determines the depth of the measurement:
short-separation channels (< 1 cm) mainly sample superficial scalp haemodynamics;
long-separation channels (2.5–4.5 cm) are more sensitive to cortical responses.

### From intensity to concentration: the modified Beer-Lambert law

Raw fNIRS data are light intensities in Watts (or arbitrary ADC units). The standard
analysis pipeline converts them in two steps:

1. **Optical density (OD)**

   $$\Delta\text{OD}(\lambda) = -\log\!\left(\frac{I(t,\lambda)}{I_0(\lambda)}\right)$$

   where $I_0$ is the mean (or baseline) intensity and $I(t,\lambda)$ is the measured
   intensity at wavelength $\lambda$. OD is dimensionless.

2. **Haemoglobin concentration changes (modified Beer-Lambert law)**

   $$\begin{pmatrix}\Delta[\text{HbO}]\\ \Delta[\text{HbR}]\end{pmatrix}
   = \left(\mathbf{E} \cdot \text{DPF} \cdot d\right)^{-1}
   \begin{pmatrix}\Delta\text{OD}(\lambda_1)\\ \Delta\text{OD}(\lambda_2)\end{pmatrix}$$

   - $\mathbf{E}$ is the *extinction coefficient matrix*: tabulated absorption of HbO and
     HbR at each wavelength (unit: 1/(mm·mM)).
   - $d$ is the source–detector distance (mm).
   - DPF is the *differential path length factor*, a scalar (~6 for adult head tissue)
     that corrects for the longer effective path of diffusely scattered photons.

In Cedalion: `cedalion.nirs.cw.int2od()` performs step 1 and `cedalion.nirs.cw.od2conc()`
performs step 2.

### Diffuse optical tomography (DOT)

DOT extends channel-space fNIRS to 3-D image reconstruction. Instead of reporting one
value per channel, DOT estimates the spatial distribution of HbO/HbR changes across the
cortex by solving an inverse problem: given a forward model (how light propagates from
each source to each detector through head tissue) and the measured OD changes, what
spatial pattern of absorption changes best explains the data?

Cedalion implements DOT via the `cedalion.dot` submodule, using realistic two-surface
head models (scalp and cortex meshes derived from MRI segmentation).

---

## The Cedalion data model

### NDTimeSeries — the core array type

Cedalion represents fNIRS timeseries as `xarray.DataArray` objects. An array is called
an **NDTimeSeries** when it has at least a `channel` dimension and a `time` dimension:

```
<xarray.DataArray (channel: 28, wavelength: 2, time: 1200)>
dims:    (channel, wavelength, time)
coords:
  channel   (channel) object  'S1D1'  'S1D2'  ...
  source    (channel) object  'S1'    'S1'    ...
  detector  (channel) object  'D1'    'D2'    ...
  wavelength (wavelength) float64  760.0  850.0
  time       (time) float64  0.0  0.1  0.2  ...  (unit: s)
  samples    (time) int64    0   1   2   ...
units: V
```

Key properties:
- **Named dimensions** — index by name (`amp.sel(channel="S1D1")`), not by position.
- **Coordinate arrays** — `source` and `detector` are sub-coordinates of `channel`,
  enabling joins with the probe geometry (`rec.geo3d.loc[amp.source]`).
- **Physical units** — attached via pint-xarray (`.pint.quantify()`, `.pint.to()`).

After Beer-Lambert conversion, the `wavelength` dimension becomes a `chromo` dimension
with values `"HbO"` and `"HbR"`.

### LabeledPoints — probe geometry

Optode and landmark positions are stored as `LabeledPoints`: an `xr.DataArray` with
dimensions `(label, <crs>)` where `<crs>` is the name of the coordinate reference
system (e.g. `"digitized"`, `"ras"`):

```
<xarray.DataArray (label: 38, digitized: 3)>
dims:    (label, digitized)
coords:
  label  (label) object  'S1'  'S2'  ...  'D1'  ...  'Nz'  ...
  type   (label) PointType  SOURCE  SOURCE  ...  DETECTOR  ...  LANDMARK  ...
units: mm
```

`PointType` is an enum: `SOURCE`, `DETECTOR`, `LANDMARK`, `ELECTRODE`, `UNKNOWN`.

### Recording — the analysis container

A `Recording` object groups all data belonging to one fNIRS session:

```python
rec.timeseries   # OrderedDict of NDTimeSeries (e.g. "amp", "od", "conc")
rec["amp"]       # shortcut for rec.timeseries["amp"]
rec.geo3d        # LabeledPoints — 3-D optode and landmark positions
rec.geo2d        # LabeledPoints — 2-D projection (optional)
rec.stim         # pd.DataFrame — stimulus onset / duration / trial_type
rec.masks        # OrderedDict of boolean DataArrays (quality masks)
rec.head_model   # TwoSurfaceHeadModel (optional, for DOT)
rec.meta_data    # dict of metadata strings
```

Canonical timeseries key names used by `read_snirf` / `write_snirf`:

| Key | Data type |
|-----|-----------|
| `"amp"` | Raw amplitude (intensity) |
| `"od"` | Optical density |
| `"conc"` | HbO/HbR concentration |
| `"hrf_conc"` | Haemodynamic response function (concentration) |
| `"hrf_od"` | Haemodynamic response function (OD) |

### Quality masks

Boolean DataArrays with `True` = clean and `False` = tainted. They are stored in
`rec.masks` and applied with `cedalion.sigproc.quality.prune_ch()`:

```python
from cedalion.sigproc import quality

sci = quality.scalp_coupling_index(rec["amp"])
rec.masks["sci"] = sci > 0.7          # True = good channel

amp_pruned, dropped = quality.prune_ch(rec["amp"], rec.masks, operator="drop")
```

Multiple masks can be combined; `prune_ch` supports `operator="all"` (drop only if all
masks flag the channel bad) and `operator="any"` (drop if any mask flags it bad).

### Physical units

Cedalion uses [pint](https://pint.readthedocs.io) through
[pint-xarray](https://pint-xarray.readthedocs.io) for unit handling.
Import the unit registry and `Quantity` from cedalion directly:

```python
from cedalion import units, Quantity

distance_threshold = 1.5 * units.cm          # pint Quantity
mask = distances > distance_threshold         # comparison preserves units

conc_uM = conc.pint.to("micromolar")         # unit conversion
conc_plain = conc.pint.dequantify()           # strip units (for sklearn etc.)
```

Units on `time` coordinates can be lost after certain xarray operations (e.g.
`xr.dot`); if this happens, re-attach them with `.assign_coords(time=...)`.

---

## Typical analysis flow

```
SNIRF file
    │ cedalion.io.read_snirf()
    ▼
Recording (rec)
    │ rec["amp"] = amplitude DataArray
    │
    ├─ Quality assessment ──────────────────────► rec.masks["sci"], ["snr"], …
    │  cedalion.sigproc.quality
    │
    ├─ Amplitude → OD ─────────────────────────► rec["od"]
    │  cedalion.nirs.cw.int2od()
    │
    ├─ Motion artefact correction ──────────────► rec["od"] (corrected)
    │  cedalion.sigproc.motion
    │
    ├─ Temporal filtering ──────────────────────► rec["od"] (filtered)
    │  cedalion.sigproc.frequency
    │
    ├─ OD → concentration (mBLL) ───────────────► rec["conc"]
    │  cedalion.nirs.cw.od2conc()
    │
    ├─ GLM / epoching ──────────────────────────► beta maps / evoked responses
    │  cedalion.models.glm  /  rec["amp"].cd.to_epochs()
    │
    └─ DOT image reconstruction ────────────────► vertex-space images
       cedalion.dot
```
