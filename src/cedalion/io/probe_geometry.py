import numpy as np
import xarray as xr
import trimesh
import json
from collections import OrderedDict

import cedalion
from cedalion.dataclasses import PointType, TrimeshSurface, build_labeled_points


def load_tsv(tsv_fname: str, crs: str='ijk', units: str='mm') -> xr.DataArray:
    """Load a tsv file containing optodes or landmarks.

    Parameters
    ----------
    tsv_fname : str
        Path to the tsv file.
    crs : str
        Coordinate reference system of the points.
    units : str
        
    Returns
    -------
    xr.DataArray
        Optodes or landmarks as a Data
    """
    with open(tsv_fname, 'r') as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]

    data = OrderedDict([(line[0], np.array([float(line[1]), float(line[2]), 
                                            float(line[3])])) for line in lines])
    # Check if tsv_type is optodes
    if all([k[0] in ['S', 'D'] for k in data.keys()]):
        tsv_type = 'optodes'
    else:
        tsv_type = 'landmarks'

    if tsv_type == 'optodes':
        types = []
        for lab in data.keys():
            if lab[0] == 'S':
                types.append(PointType(1)) # sources
            elif lab[0] == 'D':
                types.append(PointType(2)) # detectors
            else:
                raise ValueError("Unknown optode type")

        geo3d = build_labeled_points(np.array(list(data.values())),
                                     labels=list(data.keys()), crs=crs,
                                     types=types, units=units)
        return geo3d
    elif tsv_type == 'landmarks':
        landmarks = xr.DataArray(np.array(list(data.values())),
                                 dims=['label', crs],
		                         coords={"label": ("label", list(data.keys())),
					    				 "type": ("label", [PointType.LANDMARK] \
                                                           * len(data))
                                        }
                                )
        return landmarks
    return data


def read_mrk_json(fname: str, crs: str) -> xr.DataArray:
    """Read a JSON file containing landmarks.

    Parameters
    ----------
    fname : str
        Path to the JSON file.
    crs : str
        Coordinate reference system of the landmarks.

    Returns
    -------
    xr.DataArray
        Landmarks as a DataArray.
    """
    with open(fname) as fin:
        x = json.load(fin)

    units = []
    labels = []
    positions = []
    types = []

    for markup in x["markups"]:
        units.append(markup["coordinateUnits"])  # FIXME handling of units

        for cp in markup["controlPoints"]:
            labels.append(cp["label"])

            # 3x3 matrix. column vectors are coordinate axes
            orientation = np.asarray(cp["orientation"]).reshape(3, 3)

            pos = cp["position"]
            positions.append(pos @ orientation)
            types.append(PointType.LANDMARK)

    unique_units = list(set(units))
    if len(unique_units) > 1:
        raise ValueError(f"more than one unit found in {fname}: {unique_units}")

    pos = np.vstack(pos)

    result = xr.DataArray(
        positions,
        dims=["label", crs],
        coords={"label": ("label", labels), "type": ("label", types)},
        attrs={"units": unique_units[0]},
    )

    result = result.pint.quantify()

    return result


def save_mrk_json(fame: str, landmarks: xr.DataArray, crs: str):
    """Save landmarks to a JSON file.

    Parameters
    ----------
    fname : str
        Path to the output file.
    landmarks : xr.DataArray
        Landmarks to save.
    crs: str
        Coordinate system of the landmarks.
    """
    control_points = [{"id": i,
                       "label": lm.label.item(),
                       "position": list(np.array(landmarks[i])),
                       "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                       }
                      for i, lm in enumerate(landmarks)]
    data_dict = {"@schema": "https://raw.githubusercontent.com/slicer/slicer/master/Modules/Loadable/Markups/Resources/Schema/markups-schema-v1.0.3.json",
                 "markups": [{
                     "type": "Fiducial",
                     "coordinateSystem": crs,
                     "coordinateUnits": "mm", #landmark.units,
                     "controlPoints": control_points,
                     }]}
    json.dump(data_dict, open(fname, "w"), indent=2)


def read_digpts(fname: str, units: str="mm") -> xr.DataArray:
    """Read a file containing digitized points.

    Parameters
    ----------
    fname : str
        Path to the file.
    units : str
        Units of the points.

    Returns
    -------
    xr.DataArray
        Digitized points as a DataArray.
    """
    with open(fname) as fin:
        lines = fin.readlines()

    labels = []
    coordinates = []

    for line in lines:
        label, coords = line.strip().split(":")
        coords = list(map(float, coords.split()))
        coordinates.append(coords)
        labels.append(label)

    result = xr.DataArray(
        coordinates,
        dims=["label", "pos"],
        coords={"label": labels},
        attrs={"units": units},
    )
    result = result.pint.quantify()

    return result


def read_einstar_obj(fname: str) -> TrimeshSurface:
    """Read a textured triangle mesh generated by Einstar devices.

    Parameters
    ----------
    fname : str
        Path to the file.

    Returns
    -------
    TrimeshSurface
        Triangle
    """
    mesh = trimesh.load(fname)
    return TrimeshSurface(mesh, crs="digitized", units=cedalion.units.mm)
