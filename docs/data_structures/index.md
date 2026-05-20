# Data Structures

Cedalion's data model is built on xarray DataArrays with named dimensions, coordinate
labels, and physical units attached via pint-xarray. This makes it possible to write
analysis code that is self-documenting and robust to axis-ordering mistakes.

The three primary types are:

- **`NDTimeSeries`** — a multi-dimensional timeseries array with at least `channel`
  and `time` dimensions. The main data carrier through the analysis pipeline.
- **`LabeledPoints`** — a 2-D array mapping string labels to 3-D positions, used for
  optode geometry and landmarks.
- **`Recording`** — a container grouping all data for one session: timeseries, masks,
  geometry, stimulus table, and optional head model.

For a full conceptual introduction including code examples and the typical analysis
flow, see the [Concepts guide](../concepts.md).

For API reference and detailed function signatures, see
[Data structures and I/O](../data_io/index.rst).
