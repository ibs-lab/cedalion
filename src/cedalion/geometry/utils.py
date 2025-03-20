"""Utility functions for geometric calculations."""

import numpy as np


def m_trans(t: np.ndarray) -> np.ndarray:
    """Calculate the affine transformation matrix for a tranlation t."""
    tx, ty, tz = t
    # fmt: off
    return np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0,  1]
    ])
    # fmt: on


def m_scale3(s: np.ndarray) -> np.ndarray:
    """Calculate the affine transformation matrix for scaling s.

    Apply different scaling factors for each dimension.
    """

    sx, sy, sz = s

    # fmt: off
    return np.array([
        [sx,  0,  0, 0],
        [ 0, sy,  0, 0],
        [ 0,  0, sz, 0],
        [ 0,  0,  0, 1]
    ])
    # fmt: on


def m_scale1(s: np.ndarray) -> np.ndarray:
    """Calculate the affine transformation matrix for scaling s.

    Apply one scaling factor for all dimensions.
    """
    s = s[0]

    # fmt: off
    return np.array([
        [ s,  0,  0, 0],
        [ 0,  s,  0, 0],
        [ 0,  0,  s, 0],
        [ 0,  0,  0, 1]
    ])
    # fmt: on


def m_rot(angles: np.ndarray) -> np.ndarray:
    """Calculate the affine transformation matrix for a 3D rotation.

    R = Rz(alpha)Ry(beta)Rx(gamma)

    https://en.wikipedia.org/wiki/Rotation_matrix#General_rotations
    """
    alpha, beta, gamma = angles

    ca = np.cos(alpha)
    sa = np.sin(alpha)
    cb = np.cos(beta)
    sb = np.sin(beta)
    cg = np.cos(gamma)
    sg = np.sin(gamma)

    # fmt: off
    return np.stack( (ca*cb, ca*sb*sg - sa*cg, ca*sb*cg + sa*sg, 0.,
                      sa*cb, sa*sb*sg + ca*cg, sa*sb*cg - ca*sg, 0.,
                        -sb,            cb*sg,            cb*cg, 0.,
                          0.,              0.,               0., 1.)).reshape(4,4)
    # fmt: on


def cart2sph(
    x: np.ndarray, y: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert 3D cartesian into spherical coordinates.

    Args:
        x: cartesian x coordinates
        y: cartesian y coordinates
        z: cartesian z coordinates

    Returns:
        The spherical coordinates azimuth, elevation and radius as np.ndarrays.
    """
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return az, el, r


def pol2cart(theta : np.ndarray, rho : np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 2D polar into 2D cartesian coordinates.

    Args:
        theta: polar theta/angle coordinates
        rho: polar rho/radius coordinates

    Returns:
        The cartesian coordinates x and y as np.ndarrays.
    """

    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y
