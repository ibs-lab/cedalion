from cedalion.dataclasses.geometry import PycortexSurface, TrimeshSurface
import trimesh.creation
from cedalion import units
import numpy as np
from numpy.testing import assert_allclose


def test_pycortex_geodesic_distance_on_sphere():
    """Test geodesic distance calculation of PycortexSurface on a spherical mesh.

    Calculate distances between all vertices and the top-most point of the sphere.
    Use the analytic formula of distance on a spherical surface to assess the result.
    """
    RADIUS = 10
    CRS = "some_crs"

    # sphere centered at the origin
    s = trimesh.creation.icosphere(subdivisions=5, radius=RADIUS)
    ts = TrimeshSurface(s, CRS, units.cm)

    ps = PycortexSurface.from_trimeshsurface(ts)

    # find index of highest vertex (0,0,10)
    vertices = ps.vertices.pint.dequantify().values
    idx = np.argmax(vertices[:, 2])
    assert all(vertices[idx, :] == (0, 0, 10))

    # distance of all vertices to (0,0,10)
    distances = ps.geodesic_distance([idx], m=10)

    top_dir = np.array([0, 0, 1.])
    vertices_dirs = vertices / np.linalg.norm(vertices, axis=1)[:, None]  # (nverts,3)

    angles = np.arccos(np.dot(top_dir, vertices_dirs.T))  # shape: (nverts,), units: rad

    expected_distances = RADIUS * angles

    # FIXME 16 % relative tolerance is needed to make this test pass. That's quite high.
    # Absolute tolerance of 0.1 at a radius of 10 works, too.

    assert_allclose(distances, expected_distances, atol=0.1)
