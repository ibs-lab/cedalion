"""Mesh decimation, voxel-to-vertex mapping, and scalar upscaling utilities."""

from collections import Counter

import numpy as np
import pymeshlab
from tqdm import tqdm
import trimesh
import xarray as xr
import scipy.spatial

import cedalion.dataclasses as cdc


def upscale_scalars(
    highres_mesh: cdc.TrimeshSurface,
    lowres_mesh: cdc.TrimeshSurface,
    lowres_scalars: np.ndarray,
):
    """Upscale a scalar function from a low-resolution mesh to a higher-resolution one.

    Uses barycentric interpolation: for each high-res vertex, the closest point on the
    low-res mesh is found and the scalar value is interpolated from the enclosing face's
    three vertices.  The low-res mesh must be a spatial subset of the high-res mesh
    (e.g. produced by :func:`decimate_mesh`).

    Args:
        highres_mesh: Target surface at higher resolution.
        lowres_mesh: Source surface at lower resolution.
        lowres_scalars: Scalar values defined at each vertex of ``lowres_mesh``,
            shape ``(n_lowres_vertices,)``.

    Returns:
        NumPy array of scalar values interpolated onto ``highres_mesh`` vertices,
        shape ``(n_highres_vertices,)``.
    """

    # For each high-res vertex, find the closest point on the low-res mesh
    closest_points, distances, face_ids = trimesh.proximity.closest_point(
        lowres_mesh.mesh, highres_mesh.mesh.vertices
    )

    # Get barycentric coordinates on the low-res mesh
    bary = trimesh.triangles.points_to_barycentric(
        lowres_mesh.mesh.triangles[face_ids], closest_points
    )

    # Interpolate using barycentric weights
    face_vertices = lowres_mesh.mesh.faces[face_ids]  # (N, 3) vertex indices
    highres_scalars = np.sum(bary * lowres_scalars[face_vertices], axis=1)

    return highres_scalars


def decimate_mesh(
    surface: cdc.TrimeshSurface,
    nvertex_target: int,
    vertex_quality=None,
    selected=False,
    selection_threshold=0.5,
):
    """Decimate a triangulated surface mesh to a target vertex count.

    Uses PyMeshLab's quadric edge collapse algorithm.  When ``vertex_quality``
    is provided, it is used as a per-vertex quality scalar to guide the
    decimation.  Optionally, only vertices below a quality threshold are
    decimated when ``selected=True``.

    Args:
        surface: Input :class:`~cedalion.dataclasses.TrimeshSurface`.
        nvertex_target: Desired number of vertices in the output mesh.
        vertex_quality: Optional per-vertex quality scalar array of shape
            ``(n_vertices,)`` used to weight the decimation.
        selected: If ``True``, only decimate vertices whose quality is below
            ``selection_threshold``.
        selection_threshold: Quality threshold for vertex selection when
            ``selected=True``.

    Returns:
        New :class:`~cedalion.dataclasses.TrimeshSurface` with approximately
        ``nvertex_target`` vertices.  Per-vertex coordinates stored in
        ``surface.vertex_coords`` are transferred via nearest-neighbour lookup.
    """
    if vertex_quality is not None:
        mm_before = pymeshlab.Mesh(
            surface.mesh.vertices, surface.mesh.faces, v_scalar_array=vertex_quality
        )
        quality_weight = True
    else:
        mm_before = pymeshlab.Mesh(surface.mesh.vertices, surface.mesh.faces)
        quality_weight = False

    ms = pymeshlab.MeshSet()
    ms.add_mesh(mm_before)

    if selected:
        ms.compute_selection_by_condition_per_vertex(
            condselect=f"q <= {selection_threshold}"
        )

    print(
        f"selection: {np.sum(ms.current_mesh().vertex_selection_array())} /"
        f" {len(ms.current_mesh().vertex_selection_array())}"
    )

    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=nvertex_target / len(surface.mesh.vertices),
        qualitythr=0.3,
        preserveboundary=False,
        boundaryweight=1.0,
        preservenormal=False,
        preservetopology=True,
        optimalplacement=False,
        planarquadric=False,
        planarweight=0.001,
        qualityweight=quality_weight,
        autoclean=True,
        selected=selected,
    )

    mm_after = ms.mesh(0)

    _, vertex_indices = surface.kdtree.query(mm_after.vertex_matrix(), 1, workers=-1)

    return cdc.TrimeshSurface(
        trimesh.Trimesh(
            faces=mm_after.face_matrix(), vertices=mm_after.vertex_matrix()
        ),
        crs=surface.crs,
        units=surface.units,
        vertex_coords={k: v[vertex_indices] for k, v in surface.vertex_coords.items()},
    )


def map_voxels_to_vertices(surface: cdc.TrimeshSurface, cell_coords):
    """Map voxel centres to their nearest surface vertex.

    Projects each voxel coordinate onto the surface mesh and then finds the
    nearest vertex.  Processed in chunks to limit memory use.

    Args:
        surface: Target :class:`~cedalion.dataclasses.TrimeshSurface`.
        cell_coords: Array of voxel-centre coordinates, shape ``(N, 3)``.

    Returns:
        Tuple ``(voxel2vertex_indices, voxel_count)``:

        - **voxel2vertex_indices** (np.ndarray[int], shape ``(N,)``): index of the
          nearest vertex for each voxel.
        - **voxel_count** (np.ndarray[int], shape ``(n_vertices,)``): number of
          voxels mapped to each vertex.
    """
    chunk_size = 20000
    voxel2vertex_indices = []
    pq = trimesh.proximity.ProximityQuery(surface.mesh)

    for i_start in tqdm(np.arange(0, len(cell_coords), chunk_size)):
        # print(i_start, len(cell_coords))

        closest_points, _, _ = pq.on_surface(
            cell_coords[i_start : i_start + chunk_size]
        )
        _, vertex_indices = surface.kdtree.query(closest_points, 1, workers=-1)

        voxel2vertex_indices.extend(vertex_indices)

    voxel2vertex_indices = np.asarray(voxel2vertex_indices)

    voxel_count = np.zeros(surface.nvertices)
    for k, v in Counter(voxel2vertex_indices).items():
        voxel_count[k] = v

    return voxel2vertex_indices, voxel_count


def parcel_aware_voxels_to_vertices_map(
    surface: cdc.TrimeshSurface,
    cell_coords,
    skip_parcels=(
        "Background+FreeSurfer_Defined_Medial_Wall_LH",
        "Background+FreeSurfer_Defined_Medial_Wall_RH",
    ),
    voxel_stealing=False,
):
    """Map voxel centres to surface vertices respecting parcel boundaries.

    Each voxel is mapped only to vertices within the same cortical parcel,
    preventing leakage across parcel boundaries.  Optionally, vertices that
    would otherwise receive no voxel can "steal" the nearest voxel from a
    neighbouring vertex (``voxel_stealing``).

    Args:
        surface: Target :class:`~cedalion.dataclasses.TrimeshSurface` whose
            vertices carry a ``"parcel"`` coordinate.
        cell_coords: xr.DataArray of voxel-centre coordinates with a
            ``"parcel"`` coordinate that groups voxels by parcel.
        skip_parcels: Parcel labels to ignore (typically medial-wall parcels).
        voxel_stealing: If ``True``, reassign voxels to ensure every parcel
            vertex gets at least one voxel via linear assignment.

    Returns:
        Tuple ``(voxel2vertex_indices, voxel_count)``:

        - **voxel2vertex_indices** (xr.DataArray[int]): nearest vertex index for
          each voxel (``-1`` for skipped parcels).
        - **voxel_count** (np.ndarray[int], shape ``(n_vertices,)``): number of
          voxels mapped to each vertex.
    """
    voxel2vertex_indices = xr.DataArray(
        np.ones(len(cell_coords), dtype=int) * -1,
        dims="label",
        coords={"label": cell_coords.coords["label"]},
    )

    cell_coords = cell_coords.pint.dequantify()
    surf_vertices = surface.vertices.pint.dequantify()

    pq = trimesh.proximity.ProximityQuery(surface.mesh)

    surf_vertices_indices_all = np.arange(surface.nvertices)

    for parcel, parcel_cell_coords in tqdm(cell_coords.groupby("parcel")):
        if parcel in skip_parcels:
            continue

        # mask of a all vertices with this parcel label
        surf_vertices_mask = surf_vertices.parcel.values == parcel

        # build a tree with vertices of only this parcel
        kdtree = scipy.spatial.KDTree(surf_vertices.values[surf_vertices_mask])

        # map cell coordinates to points on the mesh
        closest_points, _, _ = pq.on_surface(parcel_cell_coords)

        # for each point on the mesh, find the closest vertex. indices returned refer
        # only to the vertices in the tree
        _, vertex_indices = kdtree.query(closest_points, 1, workers=-1)

        # translate indices to indices of surface vertices
        vertex_indices = surf_vertices_indices_all[surf_vertices_mask][vertex_indices]

        # number of vertices mapped to voxels
        nvertices_mapped = len(set(vertex_indices))

        # number of vertices with the current parcel label
        nvertices_parcel = np.sum(surf_vertices_mask)

        voxel2vertex_indices.loc[parcel_cell_coords.label] = vertex_indices

        # A small number vertices might not get a voxel assigned, creating blank spots
        # in surface plots. Assign the closest voxel to them, even if that voxel would
        # normally be mapped to another vertex.
        if voxel_stealing and (nvertices_mapped < nvertices_parcel):
            # voxel stealing
            unassigned_vertex_indices = [
                i for i in np.nonzero(surf_vertices_mask)[0] if i not in vertex_indices
            ]
            unassigned_vertex_positions = surf_vertices.values[
                unassigned_vertex_indices
            ]

            print(
                f"parcel {parcel}: {nvertices_mapped}/{nvertices_parcel} verts mapped. "
                f"reassign {len(unassigned_vertex_indices)} voxel(s) to verts "
                "without voxels."
            )

            # array of only those voxels belonging to this parcel
            current_v2v = voxel2vertex_indices[parcel_cell_coords.label.values]

            # don't reassign voxels from vertices that only have a single one
            unique_assigned_vidx, unique_vertex_voxel_counts = np.unique(
                current_v2v,
                return_counts=True,
            )
            vidx_with_only_one_voxel = unique_assigned_vidx[
                unique_vertex_voxel_counts == 1
            ]
            # flag those voxels that should not be reassigned
            keep_mask = np.isin(current_v2v, vidx_with_only_one_voxel)

            # distance between all unassigned vertices and all voxels in this parcel
            # shape (len(unassigned_vertex_positions), len(arcel_cell_coords))
            dists = scipy.spatial.distance.cdist(
                unassigned_vertex_positions,
                parcel_cell_coords.values,
                metric="euclidean",
            )
            dists[:, keep_mask] += 1000 # increase distance to make them unattractive

            # each vertex without voxel gets the closest voxel assigned
            i_vertices, i_voxels = scipy.optimize.linear_sum_assignment(dists)

            voxel2vertex_indices[parcel_cell_coords.label[i_voxels]] = (
                unassigned_vertex_indices
            )

            # afterwards all vertices of this parcel should be mapped to a voxel

            nvertices_mapped_after = len(
                set(voxel2vertex_indices[parcel_cell_coords.label.values].values)
            )
            if nvertices_mapped_after != nvertices_parcel:
                raise RuntimeError(
                    "Even after voxel stealing there are still "
                    "not all vertices assigned."
                )

    voxel_count = np.zeros(surface.nvertices)
    for k, v in Counter(voxel2vertex_indices.values).items():
        voxel_count[k] = v

    return voxel2vertex_indices, voxel_count


