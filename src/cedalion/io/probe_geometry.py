"""Module for reading and writing probe geometry files."""

import numpy as np
import xarray as xr
import trimesh
import json
from collections import OrderedDict
import pandas as pd

import cedalion
from cedalion.dataclasses import PointType, TrimeshSurface, build_labeled_points


def load_tsv(tsv_fname: str, crs: str=None, units: str=None) -> xr.DataArray:
    """Load a tsv file containing optodes or landmarks.

    Parameters
    ----------
    tsv_fname : str
        Path to the tsv file.
    crs : str
        Coordinate reference system of the points if not in the file header.
    units : str
        Units of the points if not in the file header.

    Returns:
    -------
    xr.DataArray
        Optodes or landmarks as a Data
    """
    # load the tsv file without header
    data = pd.read_csv(tsv_fname, sep="\t", header=None)
    if data.values[0][0] == 'sourceIndex':
        # tsv file contains a meas_list (with header)
        return pd.read_csv(tsv_fname, sep="\t")
    elif data.values[0][0] == 'labels':
        # tsv file contains a header
        data = pd.read_csv(tsv_fname, sep="\t")
    else:
        datadict = {'labels': data.iloc[:, 0],
                    'X': data.iloc[:, 1],
                    'Y': data.iloc[:, 2],
                    'Z': data.iloc[:, 3]}
        if len(data.columns) > 4:
            datadict['PointType'] = data.iloc[:, 4]
        data = datadict
        
    # parse crs and units
    for k in data.keys():
        if k.startswith('crs'):
            crs = k.split('=')[1].strip()
            data = data.drop(k, axis=1)
        if k.startswith('units'):
            units = k.split('=')[1].strip()
            data = data.drop(k, axis=1)
    
    for k in ['labels', 'X', 'Y', 'Z']:
        if k not in data.keys():
            raise ValueError(f"Missing {k} in tsv file")
   
    # parse labels
    labels = data['labels'].values

    # parse types
    types = []
    if 'PointType' in data.keys():
        for t in data.get('PointType', ''):
            if t.endswith('SOURCE'):
                types.append(PointType(1)) # sources
            elif t.endswith('DETECTOR'):
                types.append(PointType(2)) # detectors
            elif t.endswith('LANDMARK'):
                types.append(PointType(3)) # landmarks
            elif t.endswith('ELECTRODE'):
                types.append(PointType(4)) # electrodes
            else:
                types.append(PointType(0)) # unknown
    else:
        # try to detect point types if not in the file
        for lab in labels:
            if lab[0] == 'S':
                types.append(PointType(1)) # sources
            elif lab[0] == 'D':
                types.append(PointType(2)) # detectors
            elif lab in ['NAS', 'Nz', 'Iz', 'LPA', 'RPA']:
                types.append(PointType(3)) # landmarks
            elif lab[0] in ['A', 'C', 'F', 'I', 'N', 'O', 'P', 'T']:	
                types.append(PointType(4)) # electrodes
            else:
                types.append(PointType(0)) # unknown
    
    # parse data
    data = np.array([data['X'].values, data['Y'].values, data['Z'].values]).T
   
    # convert to xarray DataArray
    geo3d = build_labeled_points(data, labels=labels, crs=crs,
                                 types=types, units=units)
    return geo3d


def export_to_tsv(tsv_filename, points):
    """Export optodes, fiducials, landmarks, electodes, measurement lists to a tsv file.

    Parameters
    ----------
    tsv_filename : str
        Path to the output file.

    points : xr.DataArray, pd.DataFrame
        Points to save.
                
    Returns 
    -------
    None    
    """
    # if measurement list, save it as tsv using pandas
    if isinstance(points, pd.DataFrame):
        points.to_csv(tsv_filename, sep="\t", index=False)
        return
    elif isinstance(points, xr.DataArray):
        # else: types are optodes, fiducials, landmarks, electrodes
        with open(tsv_filename, 'w') as f:
            labels = points.label.values
            types = points.type.values
            header = "labels\tX\tY\tZ\tPointType"
            if points.points.crs is not None:
                header += "\tcrs=%s" % points.points.crs
            if points.pint.units is not None:
                if points.pint.units == cedalion.units.mm:
                    header += "\tunits=mm"
                else:
                    header += "\tunits=%s" % points.pint.units
            f.write(header + "\n")

            points = np.array(points.to_numpy())
            for l, p, t in zip(labels, points, types):
                f.write("%s\t%f\t%f\t%f\t%s\n" % (l, p[0], p[1], p[2], str(t)))
    else:
        raise ValueError("Unknown points type: %s" % type(points))
    return


def read_mrk_json(fname: str, crs: str) -> xr.DataArray:
    """Read a JSON file containing landmarks.

    Parameters
    ----------
    fname : str
        Path to the JSON file.
    crs : str
        Coordinate reference system of the landmarks.

    Returns:
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


def save_mrk_json(fname: str, landmarks: xr.DataArray, crs: str):
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

    Returns:
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

    Returns:
    -------
    TrimeshSurface
        Triangle
    """
    mesh = trimesh.load(fname)
    return TrimeshSurface(mesh, crs="digitized", units=cedalion.units.mm)
