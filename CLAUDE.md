# Cedalion — Claude Code Project Memory

Cedalion is a Python fNIRS/DOT analysis framework developed at the IBS Lab,
TU Berlin. It covers signal processing, head modeling, source localization,
and multimodal neuroimaging pipelines.

---

## Repository Structure

```
cedalion/
├── src/cedalion/      # Main library source code
├── tests/             # pytest unit tests (mirror src/ structure)
├── examples/          # Jupyter notebooks grouped by topic (rendered in docs)
├── docs/              # Sphinx documentation site, references.bib
├── plugins/           # Non-pip-installable third-party plugins (e.g. nirfaster)
└── scripts/           # build_docs.sh, colab_setup, etc.
```

---

## Stack & Environment

- **Python**: 3.11
- **Key libraries**: MNE 1.9, MNE-BIDS, MNE-NIRS, xarray 2025.6, pint-xarray,
  numpy 2.2, scipy 1.15, nilearn 0.11, nibabel, scikit-learn, pandas 2.3,
  matplotlib 3.10, pyvista, vtk, trimesh, pmcx/pmcxcl (Monte Carlo photon sim),
  snirf, pyxdf, pybids
- **Linter**: ruff (rules E, F, W, D; Google docstring convention)
- **Tests**: pytest + pytest-cov
- **Docs**: Sphinx + nbsphinx + sphinx-autoapi + sphinxcontrib-bibtex (RTD theme)
- **Build**: hatch + hatch-vcs (version from git tags)
- **Packaging**: conda-forge environment defined in `environment_dev.yml`

---

## Code Style

Follow **PEP 8** throughout. Key rules:
- 4-space indentation, no tabs
- Max line length 88 (ruff default)
- `snake_case` for functions and variables, `PascalCase` for classes, `UPPER_CASE`
  for module-level constants
- Prefer explicit imports over wildcard imports
- Prefer vectorized numpy/xarray operations over Python loops
- Use type annotations for all public function signatures

**Docstrings**: Google style (enforced by ruff `pydocstyle convention = "google"`).

```python
def my_function(param1: np.ndarray, param2: float) -> xr.DataArray:
    """Short one-line summary.

    Longer description if needed. Explain the algorithm or non-obvious
    behavior here.

    Args:
        param1: Description of param1. Include shape/units where relevant,
            e.g. ``(samples, channels)``, unit ``V``.
        param2: Description of param2.

    Returns:
        Description of return value, including shape/dtype/units if relevant.

    Raises:
        ValueError: If param2 is negative.

    Example:
        >>> result = my_function(data, 0.5)
    """
```

Note: Rules D100–D107 are currently **ignored with FIXME**. Substantial docstring
coverage was added in April 2026, but gaps remain — adding missing docstrings is
still a priority task.

---

## Priority Tasks (in order)

1. **Documentation**: Add/improve Google-style docstrings to public functions,
   classes and methods. Improve prose in Jupyter notebook examples. Improve
   content for the Sphinx RTD website.
2. **Tests**: Write pytest unit tests. Find and report bugs.
3. **New functionality**: Implement features in `src/cedalion/`. Create new
   example notebooks that explain and visualize core functionality for new users.

---

## Docstring & Documentation Rules

- All public functions, classes, and methods must have a Google-style docstring.
- Include `Args:`, `Returns:`, and `Raises:` sections where applicable.
- For scientific functions, document physical units (use `pint`/`pint-xarray`
  conventions), array shapes, and xarray dimension names.
- For new sphinx pages or docstring examples referencing literature, add the
  citation to `docs/references.bib` and use sphinxcontrib-bibtex cite syntax.
- Notebook prose should be written for a researcher new to fNIRS/DOT — assume
  signal processing knowledge but not familiarity with cedalion's API.

---

## Testing Rules

- Test files live in `tests/` and mirror the `src/cedalion/` structure.
  E.g. `src/cedalion/sigproc/frequency.py` → `tests/sigproc/test_frequency.py`.
- Test functions must be named `test_<what_is_being_tested>`.
- Use pytest fixtures for reusable test data; prefer small synthetic data arrays
  over loading real datasets.
- Always run `pytest tests/` before marking a task done.
- When fixing a bug, write a regression test that catches it first.

---

## Git & Safety Rules

- **Never commit directly to `main`**. Always create a feature branch.
- Branch naming: `feature/<short-description>` or `fix/<short-description>`.
- Run `ruff check src/ tests/` before committing.
- Run `pre-commit run` (configured via `.pre-commit-config.yaml`) before pushing.
- Keep commits focused — one logical change per commit.
- Do not touch unrelated files when fixing a bug or adding a feature.

---

## Domain Context

- fNIRS (functional Near-Infrared Spectroscopy) measures hemodynamic responses
  via light absorption. DOT (Diffuse Optical Tomography) extends this to 3D
  imaging. Key signals: raw optical density, HbO/HbR concentration changes
  (Beer-Lambert law), channel-space and image-space representations.
- xarray `DataArray` and `Dataset` are the primary data containers; dimensions
  typically include `channel`, `wavelength`, `time`, `vertex`.
- `pint-xarray` is used for physical units — preserve unit information through
  transformations.
- MNE objects (`Raw`, `Epochs`, `Evoked`) are used for EEG and some fNIRS
  pipelines; conversions between MNE and xarray representations exist in cedalion.

### Submodules

| Submodule | Purpose |
|---|---|
| `cedalion.nirs` | NIRS physics: `cw` (continuous wave), `fd` (frequency domain), `td` (time domain), `common` (extinction coefficients, channel distances) |
| `cedalion.sigproc` | Signal processing: `frequency` (filtering, sampling rate), `quality` (channel quality, pruning), `motion` (artifact correction), `epochs` (trial extraction), `physio` (physiological noise), `tasks` (high-level `@task` wrappers operating on `Recording`) |
| `cedalion.dataclasses` | Core containers: `Recording`, `Surface`, `PointType`, schemas, xarray accessor (`.cd`) |
| `cedalion.dot` | DOT pipeline: `head_model` (`TwoSurfaceHeadModel`), `forward_model`, `image_recon`, `tissue_properties` |
| `cedalion.geometry` | 3D geometry: `registration`, `segmentation`, `meshing`, `landmarks`, `ellipsoid` |
| `cedalion.io` | I/O: `snirf`, `bids`, `anatomy`, `forward_model`, `probe_geometry`, `photogrammetry`, `nirs` (Homer2 `.nirs` files), `utils` (HDF5/xarray serialisation helpers) |
| `cedalion.vis` | Visualization: `timeseries`, `quality`, `blocks`, `colors` |
| `cedalion.sim` | Simulation: `synthetic_hrf`, `synthetic_artifact`, `synthetic_utils`; `datasets.synthetic_fnirs_eeg` (`BimodalToyDataSimulation`) for ML benchmarks |
| `cedalion.math` | Math utilities: `ar_irls`, `ar_model`, `resample`, `stats_helpers` |
| `cedalion.models.glm` | General Linear Model: `design_matrix` (HRF, drift, short-channel regressors), `solve` (`fit`, `predict`), `basis_functions` (`Gamma`, `GaussianKernels`, `DiracDelta`) |
| `cedalion.sigdecomp.unimodal` | Unimodal decomposition: `ICA_ERBM`, `ICA_EBM`, `SPoC` |
| `cedalion.sigdecomp.multimodal` | Multimodal decomposition: `cca` (CCA, SparseCCA, RidgeCCA, ElasticNetCCA, StructuredSparseCCA, PLS, SparsePLS), `tcca` (tCCA, ElasticNetTCCA, StructuredSparseTCCA), `mspoc` (mSPoC) |
| `cedalion.mlutils` | sklearn bridge: `cv` (`create_cv_splits`, `mask_design_matrix`), `features` (`epoch_features`) |
| `cedalion.xrutils` | xarray helpers: `pinv`, `norm`, `apply_mask`, etc. |
| `cedalion.typing` | Type aliases: `NDTimeSeries`, `LabeledPoints`, `AffineTransform`, `QTime`, `QLength`, `QFrequency`, `QConcentration` |
| `cedalion.physunits` | pint unit registry; exposes `cedalion.units` and `cedalion.Quantity` |
| `cedalion.validators` | Input guard functions: `has_channel`, `has_wavelengths`, `has_positions`, `check_dimensionality` |
| `cedalion.tasks` | `@task` decorator that registers pipeline-level functions in `task_registry` |

### Key data flow

```
SNIRF / BIDS file
       │ cedalion.io
       ▼
  Recording                          ← main analysis container
  ├── timeseries: OrderedDict        ← named NDTimeSeries DataArrays
  │   ├── "amp"   (amplitude, W)
  │   ├── "od"    (optical density, dimensionless)
  │   └── "conc"  (HbO/HbR, µM)
  ├── geo3d / geo2d  (LabeledPoints) ← optode & landmark positions
  ├── stim  (DataFrame)              ← onset / duration / trial_type
  ├── masks (OrderedDict)            ← boolean quality masks
  └── head_model  (TwoSurfaceHeadModel, optional)

Intensity → OD        cedalion.nirs.cw.int2od()  /  sigproc.tasks.int2od()
OD → Concentration    cedalion.nirs.cw.od2conc() /  sigproc.tasks.od2conc()
Channel → Image       cedalion.dot.forward_model + image_recon
```

- Timeseries key names encode data type: `"amp"` or `"amp_*"` = amplitude,
  `"od"` or `"od_*"` = optical density, `"conc"` or `"conc_*"` = concentration,
  `"hrf"` or `"hrf_*"` = hemodynamic response. `rec.get_timeseries_type(key)`
  infers the type from key name or pint units.
- Two API layers exist: low-level functions (e.g., `cedalion.nirs.cw.int2od(ts)`)
  that operate on DataArrays, and high-level `@task`-decorated functions
  (e.g., `cedalion.sigproc.tasks.int2od(rec)`) that operate on a `Recording`.

### Key data types

- **`NDTimeSeries`** (`xr.DataArray`) — required dims: `time`; required coords:
  `time`, `samples`; additional dims for channel data: `channel` with sub-coords
  `source` and `detector`. The `.cd` accessor provides `.sampling_rate`,
  `.to_epochs()`, etc.
- **`LabeledPoints`** (`xr.DataArray`) — dims `("label", crs)` where `crs` is a
  string naming the coordinate reference system (e.g., `"pos"`, `"ras"`); coords
  `label` and `type` (`PointType` enum: `SOURCE`, `DETECTOR`, `LANDMARK`,
  `ELECTRODE`, `UNKNOWN`).
- **`AffineTransform`** (`xr.DataArray`) — 4×4 matrix; coordinate system names:
  `ijk` (MRI voxel), `ras` (MRI world space, mm), `pos` (probe/sensor space).
- **`Surface`** — abstract dataclass wrapping a trimesh/pyvista mesh; concrete
  subclasses hold `mesh`, `crs`, `units`, and a `vertex_coords` dict. Used inside
  `TwoSurfaceHeadModel` as `.brain` and `.scalp` surfaces.
- **Quality masks** — boolean `xr.DataArray` with `CLEAN = True`, `TAINTED = False`;
  combined with `prune_ch(..., operator="all"|"any")`.
- **Spatial dimension** — can be `channel`, `vertex`, `parcel`, or `voxel`;
  use `cdc.get_spatial_dimension(array)` to detect which is present.

### Idiomatic patterns

- **Schema validation**: decorate functions with `@cdc.validate_schemas`; type-annotate
  arguments with `cdt.NDTimeSeries` / `cdt.LabeledPoints` — validation runs
  automatically at call time.
- **Unit handling**: import `from cedalion import units, Quantity`. Attach units with
  `.pint.quantify(unit_str)`, convert with `.pint.to(target_unit)`. Units on the
  `time` coordinate are often lost by `xr.dot` and must be re-attached explicitly.
- **xr.apply_ufunc**: preferred for applying numpy operations while preserving
  xarray labels; specify `input_core_dims` / `output_core_dims` carefully.
- **Validators before logic**: call `validators.has_channel()`, `has_wavelengths()`,
  `has_positions()` at the top of public functions before any computation.
- **`xrutils`**: use helpers in `cedalion.xrutils` (e.g., `pinv`, `norm`,
  `apply_mask`) rather than reimplementing xarray/numpy operations inline.

### ML / AI integration

Cedalion is designed so that xarray DataArrays flow naturally into scikit-learn
and other ML frameworks, while preserving enough metadata to trace results back
to their neuroimaging origin.

**sklearn interface**

- `mlutils.features.epoch_features(epochs, feature_types, reltime_slices)` —
  reduces a `(epoch, channel, chromo, reltime)` DataArray to a
  `(epoch, feature)` DataArray ready for sklearn. Supported feature types:
  `"slope"`, `"mean"`, `"max"`, `"min"`, `"auc"`.
- The feature dimension is built by stacking all non-epoch dims, so coordinates
  (`channel`, `chromo`, `feature_type`) survive the stack. This lets you trace
  any sklearn feature back to its source channel and chromophore.
- Call `.pint.dequantify()` before passing to sklearn — `epoch_features` does
  this internally, but raw DataArrays need it first.
- sklearn estimators accept DataArrays directly (they behave like numpy arrays),
  and sample-dim coordinates (`is_train`, `is_test`, `y`, `trial_type`) can be
  added as extra coordinates on the epoch dim for structured train/test splits.

**Cross-validation with GLM preprocessing**

- `mlutils.cv.create_cv_splits(df_stim, n_splits)` — stratified k-fold over
  stimulus events; yields `(df_stim_train, df_stim_test)` per fold where test
  trials are consecutive (required by `mask_design_matrix`).
- `mlutils.cv.mask_design_matrix(dms, df_stim_test, before, after)` — zeros the
  design matrix in the test window so GLM parameters cannot leak test-trial
  information. Use this to include GLM-based regression (e.g., short-channel
  regression) inside the cross-validation loop.

**General Linear Model**

- `cedalion.models.glm` provides mass-univariate GLM fitting over channels:
  `glm.fit(timeseries, design_matrix, noise_model=...)` returns a result
  object; `glm.predict(timeseries, params, dms)` reconstructs the signal.
  Supported `noise_model` values: `"ols"`, `"rls"`, `"wls"`, `"ar_irls"`,
  `"gls"`, `"glsar"` (default: `"ols"`).
- Design matrices are built with the `&` operator:
  `glm.design_matrix.hrf_regressors(...) & glm.design_matrix.drift_regressors(...)`.
- HRF basis functions: `Gamma(tau, sigma)`, `GaussianKernels`, `DiracDelta`.

**Signal decomposition models (sigdecomp)**

All decomposition classes follow a consistent sklearn-like API:
`model.fit(X, Y)` → `model.transform(X, Y)` → learned filters `model.Wx`,
`model.Wy`. Inputs must be `xr.DataArray` with exactly two dims (sample + feature).

- *Unimodal*: `ICA_ERBM(X, p)` — entropy-rate-bound ICA, returns demixing matrix W.
- *Multimodal CCA family* (`sigdecomp.multimodal.cca`):
  `CCA`, `RidgeCCA` (L2), `SparseCCA` (L1), `ElasticNetCCA` (L1+L2),
  `StructuredSparseCCA` (L1 + graph Laplacian via `Lx`/`Ly`), `PLS`, `SparsePLS`.
  Multiple components extracted via deflation; `N_components=None` uses the
  minimum feature count across modalities.
- *Temporally embedded* (`sigdecomp.multimodal.tcca`):
  `tCCA`, `ElasticNetTCCA`, `StructuredSparseTCCA` — accept `time_shifts`
  (numpy array of lag values in seconds) and `shift_source` (bool); expose
  `model.optimal_shift` after fitting.
- *mSPoC* (`sigdecomp.multimodal.mspoc`): maximises covariance between EEG
  bandpower and fNIRS; `x` has higher sample rate than `y`; adds `N_restarts`
  parameter for stochastic restarts.

**Simulation for ML benchmarks**

`cedalion.sim.datasets.synthetic_fnirs_eeg.BimodalToyDataSimulation(config_dict, seed)`
generates paired synthetic fNIRS+EEG data with a controllable SNR (`gamma`),
frequency band, time lag (`dT`), and structured vs. random mixing matrices.
Use it for algorithm benchmarking and unit tests that need paired multimodal data
without a real dataset.

---

## Lessons Learned

*(Update this section when Claude makes a mistake that should not be repeated.)*

- When modifying xarray operations, check that dimension names are preserved
  and units (pint) are not silently dropped.
- pmcx/pmcxcl are GPU-based Monte Carlo photon simulation packages — do not
  mock or stub them in tests without noting clearly that the test skips GPU logic.
- `snirf2bids` is installed from a pinned git commit; do not upgrade it without
  checking compatibility.
- **Windows doc build**: `make -j 8` causes file-locking errors on Windows; use
  `make -j 1` when running `scripts/build_docs.sh` locally.
- **Docstring syntax**: Watch for `Args::` (double colon) — a silent syntax bug
  that breaks Google-style parsing. Always use a single colon: `Args:`.
- **`Attrs:` vs `Attributes:`**: Google style requires `Attributes:` (not `Attrs:`)
  for documenting dataclass/class fields.
- **Embedded string literals that look like docstrings**: A string literal
  appearing mid-function body after code has already run is NOT a docstring — it
  is dead code. Remove it; the actual docstring must be the first statement of the
  function/class body.
- **Non-Google docstring styles in the codebase**: NumPy-style (`Parameters\n---`)
  and old-style (`Inputs:`/`Outputs:`) docstrings exist in older modules. Convert
  to Google style on edit; do not leave mixed styles in the same file.
