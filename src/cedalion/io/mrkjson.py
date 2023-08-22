import json
import xarray as xr
import numpy as np


def read_mrk_json(fname: str) -> xr.DataArray:
    with open(fname) as fin:
        x = json.load(fin)

    units = []
    labels = []
    pos = []

    for markup in x["markups"]:
        units.append(markup["coordinateUnits"])  # FIXME handling of units

        for cp in markup["controlPoints"]:
            labels.append(cp["label"])
            pos.append(cp["position"])

    unique_units = list(set(units))
    if len(unique_units) > 1:
        raise ValueError(f"more than one unit found in {fname}: {unique_units}")

    pos = np.vstack(pos)

    return xr.DataArray(
        pos,
        dims=["label", "pos"],
        coords={"label": labels},
        attrs={"unit": unique_units[0]},
    )
