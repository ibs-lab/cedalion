# Quick Start

This page shows the minimal steps to go from a raw fNIRS recording to haemoglobin
concentration changes. It assumes Cedalion is already installed — see
[Installation](installation.md) if not.

## 1. Load a recording

Cedalion reads SNIRF files with `cedalion.io.read_snirf`. The bundled example datasets
are available through `cedalion.data` and are downloaded and cached automatically on
first use:

```python
import cedalion
import cedalion.io

# Option A: use a bundled example dataset (auto-downloaded)
rec = cedalion.data.get_fingertapping()

# Option B: load your own SNIRF file
recordings = cedalion.io.read_snirf("path/to/your/data.snirf")
rec = recordings[0]   # a SNIRF file may contain multiple NirsElements
```

`rec` is a `Recording` object — a container that holds the timeseries, optode
positions, stimulus table, and other metadata of a single fNIRS session. See
[Data Structures and I/O](../data_io/index.rst) for a full description.

## 2. Inspect the data

```python
# amplitude timeseries: dims (channel, wavelength, time), unit V
amp = rec["amp"]
print(amp)

# optode positions: dims (label, pos), unit mm
print(rec.geo3d)

# stimulus onset table
print(rec.stim)
```

The amplitude array has three named dimensions — `channel`, `wavelength`, and `time` —
plus coordinate arrays for source/detector labels, wavelength values, and timestamps.
You can index by label using `.sel()`:

```python
amp.sel(channel="S1D1", wavelength=760.0)   # one channel, one wavelength
amp.sel(time=slice(0, 60))                  # first 60 seconds
```

## 3. Convert to optical density

Optical density (OD) is the log-ratio of baseline to current intensity. The modified
Beer-Lambert law relates changes in OD to haemoglobin concentration changes:

```python
import cedalion.nirs.cw as nirs

od = nirs.int2od(rec["amp"])
```

## 4. Convert to haemoglobin concentration

The conversion from OD to HbO/HbR requires differential path length factors (DPF) that
account for the longer effective path length of diffusely scattered light. A common
approximation is 6 for both wavelengths in the 700–900 nm range:

```python
import xarray as xr

dpf = xr.DataArray(
    [6.0, 6.0],
    dims="wavelength",
    coords={"wavelength": od.wavelength},
)
conc = nirs.od2conc(od, rec.geo3d, dpf)
# conc has dims (channel, chromo, time) and unit M (molar)
conc_uM = conc.pint.to("micromolar")
```

## 5. Plot a channel

```python
import matplotlib.pyplot as plt

ch = conc_uM.sel(channel="S1D1")
plt.plot(ch.time, ch.sel(chromo="HbO"), "r-", label="HbO")
plt.plot(ch.time, ch.sel(chromo="HbR"), "b-", label="HbR")
plt.xlabel("time / s")
plt.ylabel("Δc / µM")
plt.legend()
plt.tight_layout()
plt.show()
```

## What next?

| Goal | Where to look |
|------|--------------|
| Understand the data model in depth | [Data Structures and I/O](../data_io/index.rst) |
| Assess and improve data quality | [Signal processing — quality control](../sigproc/index.rst) |
| Fit a GLM to estimate haemodynamic responses | [Modeling and Machine Learning](../machine_learning/index.rst) |
| Reconstruct 3-D images | [Diffuse optical tomography](../dot/index.rst) |
| Work through a complete analysis | [Tutorial Notebooks](../tutorial.rst) |
| Learn about fNIRS fundamentals | [Concepts guide](../concepts.md) |
