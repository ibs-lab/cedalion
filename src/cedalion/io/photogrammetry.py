"""Module for reading photogrammetry output file formats."""

from __future__ import annotations
import cedalion.dataclasses as cdc
import numpy as np
from collections import OrderedDict


def read_photogrammetry_einstar(fn: str) -> tuple:
    """Read optodes and fiducials from photogrammetry pipeline.

    This method reads the output file as returned by the
    photogrammetry pipeline using an einstar device.

    Args:
        fn: The filename of the einstar photogrammetry output file.

    Returns:
        A tuple containing:
            - fiducials: The fiducials as a cedalion LabeledPoints object.
            - optodes: The optodes as a cedalion LabeledPoints object.
    """

    fiducials, optodes = read_einstar(fn)
    fiducials, optodes = opt_fid_to_xr(fiducials, optodes)
    return fiducials, optodes


def read_einstar(fn: str) -> tuple:
    """Read optodes and fiducials from einstar devices.

    Args:
        fn: The filename of the einstar photogrammetry output file.

    Returns:
        A tuple containing:
            - fiducials: The fiducials as an OrderedDict.
            - optodes: The optodes as an OrderedDict.
    """

    with open(fn, "r") as f:
        lines = [[ll.strip() for ll in line.split(",")] for line in f.readlines()]
        lines = [[line[0], [float(ll) for ll in line[1:]]] for line in lines]
    assert lines[0][0] == "Nz"
    assert lines[1][0] == "Iz"
    assert lines[2][0] == "Rpa"
    assert lines[3][0] == "Lpa"
    assert lines[4][0] == "Cz"
    fiducials = OrderedDict(lines[:5])
    optodes = OrderedDict(lines[5:])
    return fiducials, optodes


def opt_fid_to_xr(fiducials: OrderedDict, optodes: OrderedDict) -> tuple:
    """Convert OrderedDicts fiducials and optodes to cedalion LabeledPoints objects.

    Args:
        fiducials: The fiducials as an OrderedDict.
        optodes: The optodes as an OrderedDict.

    Returns:
        A tuple containing:
            - fiducials: The fiducials as a cedalion LabeledPoints object.
            - optodes: The optodes as a cedalion LabeledPoints object.
    """

    # FIXME: this should get a different CRS
    CRS = "ijk"
    if len(fiducials) == 0:
        fidu_coords = np.zeros((0, 3))
    else:
        fidu_coords = np.array(list(fiducials.values()))

    if len(optodes) == 0:
        opt_coords = np.zeros((0, 3))
    else:
        opt_coords = np.array(list(optodes.values()))

    fiducials = cdc.build_labeled_points(
        fidu_coords, labels=list(fiducials.keys()), crs=CRS
    )  # , units="mm")

    types = []
    for lab in list(optodes.keys()):
        if lab.startswith("S"):
            types.append(cdc.PointType(1))
        elif lab.startswith("D"):
            types.append(cdc.PointType(2))
        else:
            types.append(cdc.PointType(0))
    optodes = cdc.build_labeled_points(
        opt_coords, labels=list(optodes.keys()), crs=CRS, types=types
    )
    return fiducials, optodes
