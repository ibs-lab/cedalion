import cedalion
import cedalion.dataclasses as cdc
import numpy as np
from collections import OrderedDict

def read_photogrammetry_einstar(fn):
    fiducials, optodes = read_einstar(fn)
    fiducials, optodes = opt_fid_to_xr(fiducials, optodes)
    return fiducials, optodes


def read_einstar(fn):
    with open(fn, 'r') as f:
        lines = [[l.strip() for l in line.split(',')] for line in f.readlines()]
        lines = [[line[0], [float(l) for l in line[1:]]] for line in lines]
    assert lines[0][0] == 'Nz'
    assert lines[1][0] == 'Iz'
    assert lines[2][0] == 'Rpa'
    assert lines[3][0] == 'Lpa'
    assert lines[4][0] == 'Cz'
    fiducials = OrderedDict(lines[:5])
    optodes = OrderedDict(lines[5:])
    return fiducials, optodes


def opt_fid_to_xr(fiducials, optodes):
    # FIXME: this should get a different CRS
    CRS = "ijk"
    fiducials = cdc.build_labeled_points(np.array(list(fiducials.values())),
                                     labels=list(fiducials.keys()),
                                     crs=CRS)#, units="mm")
    opt = np.array(list(optodes.values()))
    types = []
    for lab in list(optodes.keys()):
        if lab.startswith('S'):
            types.append(cdc.PointType(1))
        elif lab.startswith('D'):
            types.append(cdc.PointType(2))
        else:
            types.append(cdc.PointType(0))
    optodes = cdc.build_labeled_points(opt,
                                    labels=list(optodes.keys()),
                                    crs=CRS, types=types)
    return fiducials, optodes
