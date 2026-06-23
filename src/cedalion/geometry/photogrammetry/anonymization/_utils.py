"""Shared helpers used by ``preprocessing`` and ``mask``.

Consolidates texture-image lookup, largest-connected-component selection,
face/vertex reindexing, affine application, ear midpoint, upper-head centroid,
and the ``Trimesh + _copy_visual`` rebuild pattern to avoid duplicating these
across sibling modules. Also hosts the canonical landmark label tuple.
"""

import numpy as np
import trimesh


_REQUIRED_LABELS = ("Nz", "Iz", "Cz", "LPA", "RPA")
"""Canonical landmark labels expected by the anonymization pipeline."""


def _resolve_texture_image(visual):
    """Return ``visual.image`` if set, else fall back to ``material.image``."""
    image = getattr(visual, "image", None)
    if image is not None:
        return image
    mat = getattr(visual, "material", None)
    return getattr(mat, "image", None) if mat is not None else None


def _largest_component_mask(mesh: trimesh.Trimesh) -> np.ndarray:
    """Boolean vertex mask selecting the largest connected component.

    Einstar scans contain floating fragments (loose triangles, cable shreds,
    background patches) that drag vertex extrema off into empty space; strip
    them first.

    Uses ``trimesh.graph.connected_component_labels`` on face adjacency rather
    than ``mesh.split``: split allocates a ``Trimesh`` per component and can
    OOM on scans with thousands of fragments.

    Einstar OBJs duplicate vertices along UV seams; ``Trimesh.merge_vertices``
    preserves seams (to keep textures), so face adjacency on the raw mesh
    over-fragments the head. ``trimesh.grouping.unique_rows`` does
    position-only merging for connectivity analysis.

    Args:
        mesh: A ``trimesh.Trimesh`` instance.

    Returns:
        Boolean array of shape ``(n_vertices,)``. All-True if the mesh is
        empty or already a single connected component.
    """
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    if n_faces == 0:
        return np.ones(n_verts, dtype=bool)

    _, inverse = trimesh.grouping.unique_rows(np.asarray(mesh.vertices))
    canonical_faces = inverse[mesh.faces]
    adjacency = trimesh.graph.face_adjacency(faces=canonical_faces)
    face_labels = trimesh.graph.connected_component_labels(
        adjacency, node_count=n_faces
    )
    counts = np.bincount(face_labels)
    if len(counts) <= 1:
        return np.ones(n_verts, dtype=bool)
    biggest_label = int(np.argmax(counts))
    face_mask = face_labels == biggest_label

    kept_vidx = np.unique(mesh.faces[face_mask])
    mask = np.zeros(n_verts, dtype=bool)
    mask[kept_vidx] = True
    return mask


def _reindex_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
    kept_face_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a triangle mesh by a face mask and rebuild dense vertex indices.

    Args:
        vertices: ``(N, 3)`` vertex array.
        faces: ``(F, 3)`` face index array.
        kept_face_mask: ``(F,)`` bool array.

    Returns:
        ``(new_vertices, new_faces, kept_vidx)`` where ``new_faces`` indexes
        into ``new_vertices`` and ``kept_vidx`` are the original vertex
        indices that survived (useful for slicing UVs / vertex colors).
    """
    kept_faces = faces[kept_face_mask]
    kept_vidx = np.unique(kept_faces)
    reindex = -np.ones(len(vertices), dtype=np.int64)
    reindex[kept_vidx] = np.arange(len(kept_vidx))
    new_faces = reindex[kept_faces]
    new_vertices = vertices[kept_vidx]
    return new_vertices, new_faces, kept_vidx


def _apply_affine(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply a 4x4 affine to an ``(N, 3)`` point array."""
    return points @ M[:3, :3].T + M[:3, 3]


def _ear_midpoint(Lpa: np.ndarray, Rpa: np.ndarray) -> np.ndarray:
    """Return the LPA/RPA midpoint (CTF origin in the aligned frame)."""
    return 0.5 * (Lpa + Rpa)


def _upper_head_centroid(
    vertices: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Centroid of vertices in the top 40% of the X range.

    Returns ``(centroid_xyz, x_max)``. Callers that only need the YZ
    components zero out X themselves.
    """
    x_max = float(vertices[:, 0].max())
    x_min = float(vertices[:, 0].min())
    upper = vertices[vertices[:, 0] > x_min + 0.6 * (x_max - x_min)]
    centroid = upper.mean(axis=0)
    return centroid, x_max


def _copy_visual(src_mesh, dst_mesh, vertex_index=None) -> None:
    """Copy a trimesh ``visual`` onto ``dst_mesh``, optionally reindexing.

    Passing ``visual=old.visual`` to a new ``Trimesh(...)`` silently downgrades
    a ``TextureVisuals`` to ``ColorVisuals`` because the visual still
    back-references the source mesh's vertex count. Rebuilding it from scratch
    preserves TextureVisuals.

    Args:
        src_mesh: Source trimesh.
        dst_mesh: Destination trimesh (mutated in place).
        vertex_index: Indices into the source vertex array. If given, UVs /
            vertex_colors are sliced by this index. Leave None when the
            vertex count is unchanged.
    """
    src_visual = src_mesh.visual
    uv = getattr(src_visual, "uv", None)
    n_src = len(src_mesh.vertices)

    if uv is not None and len(uv) == n_src:
        uv_arr = np.asarray(uv)
        if vertex_index is not None:
            uv_arr = uv_arr[vertex_index]
        image = _resolve_texture_image(src_visual)
        material = getattr(src_visual, "material", None)
        dst_mesh.visual = trimesh.visual.TextureVisuals(
            uv=uv_arr,
            image=image,
            material=material,
        )
        return

    vcol = getattr(src_visual, "vertex_colors", None)
    if vcol is not None and len(vcol) == n_src:
        vcol_arr = np.asarray(vcol)
        if vertex_index is not None:
            vcol_arr = vcol_arr[vertex_index]
        dst_mesh.visual.vertex_colors = vcol_arr


def _rebuild_mesh(
    src_mesh: trimesh.Trimesh,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    vertex_index: np.ndarray | None = None,
) -> trimesh.Trimesh:
    """Build a new ``Trimesh`` and re-attach the source visual.

    Used by every transform in the pipeline to keep the
    ``Trimesh(process=False)`` + ``_copy_visual`` pattern in one place.

    Note: do not replace these call sites with ``TrimeshSurface.apply_transform``
    (``src/cedalion/dataclasses/geometry.py``). That path relies on
    ``mesh.copy()`` to propagate the visual and does not explicitly re-attach
    ``TextureVisuals``; combined with the vertex-reindexing steps here (which
    need UV slicing), it would silently downgrade textured meshes.
    """
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    _copy_visual(src_mesh, new_mesh, vertex_index=vertex_index)
    return new_mesh
