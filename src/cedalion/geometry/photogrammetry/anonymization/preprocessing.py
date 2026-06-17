"""Preprocessing: axis normalization and head isolation.

Gets a raw Einstar photogrammetry scan into a standard reference frame suitable for
landmark detection and mask building:

- ``normalize_axes`` rotates around X so Y points anterior (preliminary,
  used before head isolation).
- ``isolate_head`` strips body/shoulders/chair and disconnected fragments.
- ``align_axes_from_landmarks`` maps the scan into the CTF frame
  (+X=anterior, +Y=left, +Z=up, origin at the LPA-RPA midpoint) once all 5
  landmarks are available.
"""

import logging

import numpy as np
import trimesh

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion.geometry.landmarks import normalize_landmarks_labels

from ._utils import (
    _apply_affine,
    _ear_midpoint,
    _rebuild_mesh,
    _reindex_faces,
    _transform_labeled_points,
    _upper_head_centroid,
)

logger = logging.getLogger("cedalion")


@cdc.validate_schemas
def normalize_axes(
    surface: cdc.TrimeshSurface,
    nasion: np.ndarray,
) -> tuple[cdc.TrimeshSurface, np.ndarray, np.ndarray]:
    """Rotate mesh around X-axis so Y=anterior (toward face), Z=left.

    The Einstar scanner fixes X=up (gravity-based) but leaves Y/Z arbitrary
    per scan. The forward direction is inferred from the nasion against the
    upper-head centroid.

    Args:
        surface: TrimeshSurface in raw Einstar coordinates.
        nasion: Nasion position as numpy array of shape (3,), in mm, in the
            raw Einstar frame (matching ``surface``).

    Returns:
        Tuple of (rotated_surface, rotated_nasion, rotation_matrix).
        ``rotated_nasion`` is in the same units as the input. The 3x3
        ``rotation_matrix`` can be applied to other points via ``p @ R.T``.
    """
    vertices = surface.mesh.vertices

    centroid, _ = _upper_head_centroid(np.asarray(vertices))
    angle = np.arctan2(nasion[2] - centroid[2], nasion[1] - centroid[1])
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a],
    ])

    new_mesh = _rebuild_mesh(
        surface.mesh,
        vertices=vertices @ R.T,
        faces=surface.mesh.faces,
    )
    rotated_surface = cdc.TrimeshSurface(new_mesh, crs=surface.crs, units=surface.units)

    rotated_nasion = R @ nasion

    logger.debug(
        f"Axis normalization: rotated {np.degrees(angle):.1f}deg around X. "
        f"Y now points anterior."
    )

    return rotated_surface, rotated_nasion, R


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


@cdc.validate_schemas
def isolate_head(
    surface: cdc.TrimeshSurface,
    nasion: np.ndarray,
    radius: float = 220.0,
) -> tuple[cdc.TrimeshSurface, np.ndarray]:
    """Remove shoulders, body, and chair; keep only the head.

    Uses a sphere centered on the upper-head centroid. The radius is
    chosen to capture the full head (~180mm wide x 230mm tall) while
    excluding shoulders and body. Scans that are already head-only are
    returned unchanged (the sphere just contains everything).

    The surface must be axis-normalized first (X=up, Y=anterior, Z=left).

    Args:
        surface: Axis-normalized TrimeshSurface.
        nasion: Nasion position as numpy array of shape (3,), in mm,
            in the axis-normalized frame (matching ``surface``).
        radius: Sphere radius in mm (default 220). A human head has
            ~90mm radius; 220mm adds margin for ears and jaw.

    Returns:
        Tuple of (head_surface, head_mask). head_mask is a boolean
        array of shape (n_vertices,) indicating which original
        vertices were kept.
    """
    vertices = np.asarray(surface.mesh.vertices)
    faces = np.asarray(surface.mesh.faces)

    center, x_max = _upper_head_centroid(vertices)
    midpoint_x = (x_max + nasion[0]) / 2.0
    center[0] = min(center[0], midpoint_x)

    dist = np.linalg.norm(vertices - center, axis=1)
    head_mask = dist < radius

    head_mask = head_mask & _largest_component_mask(surface.mesh)

    if head_mask.sum() < 100 or head_mask.mean() > 0.95:
        logger.debug(
            f"Head isolation: sphere contains {head_mask.mean()*100:.0f}% "
            f"of vertices -- scan is already head-only"
        )
        return surface, head_mask

    face_mask = head_mask[faces].all(axis=1)
    new_verts, new_faces, kept_vidx = _reindex_faces(vertices, faces, face_mask)

    new_mesh = _rebuild_mesh(
        surface.mesh,
        vertices=new_verts,
        faces=new_faces,
        vertex_index=kept_vidx,
    )

    head_surface = cdc.TrimeshSurface(
        new_mesh, crs=surface.crs, units=surface.units
    )

    logger.debug(
        f"Head isolation: {len(vertices):,} -> {len(new_verts):,} vertices "
        f"({len(vertices) - len(new_verts):,} removed), "
        f"center=[{center[0]:.0f},{center[1]:.0f},{center[2]:.0f}], "
        f"radius={radius:.0f}mm"
    )

    return head_surface, head_mask


@cdc.validate_schemas
def align_axes_from_landmarks(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPoints,
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPoints, np.ndarray]:
    """Map mesh + landmarks into the CTF anatomical frame.

    CTF convention:

        +X = anterior (toward Nz)
        +Y = left (toward Lpa)
        +Z = up (toward Cz)
        origin = midpoint of Lpa and Rpa (interaural midpoint)

    The returned surface and landmarks carry ``crs="ctf"``.

    Args:
        surface: Axis-normalized TrimeshSurface (post ``normalize_axes`` and
            ``isolate_head``).
        landmarks: LabeledPoints with labels Nz, Iz, Cz, and LPA/RPA (or
            aliases like Lpa/Rpa, "left ear"/"right ear"; normalized
            internally via ``normalize_landmarks_labels``)
            (matching the surface frame).

    Returns:
        Tuple of (aligned_surface, aligned_landmarks, transform).
        ``transform`` is a 4x4 homogeneous affine that maps input-frame
        points into CTF; apply as ``hom = transform @ [x, y, z, 1]``.

    Raises:
        ValueError: If any of the required landmarks (Nz, Iz, Cz, LPA, RPA)
            are missing after label normalization.
    """
    landmarks = normalize_landmarks_labels(landmarks)
    lm = landmarks.pint.dequantify().values
    labels = list(landmarks["label"].values)
    idx = {lbl: i for i, lbl in enumerate(labels)}

    required = {"Nz", "Iz", "Cz", "LPA", "RPA"}
    missing = required - set(labels)
    if missing:
        raise ValueError(f"Missing landmarks for alignment: {missing}")

    Nz = lm[idx["Nz"]]
    Cz = lm[idx["Cz"]]
    Lpa = lm[idx["LPA"]]
    Rpa = lm[idx["RPA"]]
    origin = _ear_midpoint(Lpa, Rpa)

    y_ax = Lpa - Rpa
    y_ax = y_ax / np.linalg.norm(y_ax)

    nz_dir = Nz - origin
    nz_dir = nz_dir - np.dot(nz_dir, y_ax) * y_ax
    x_ax = nz_dir / np.linalg.norm(nz_dir)

    z_ax = np.cross(x_ax, y_ax)

    if np.dot(Cz - origin, z_ax) < 0:
        z_ax = -z_ax
    y_ax = np.cross(z_ax, x_ax)

    R = np.vstack([x_ax, y_ax, z_ax])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ origin

    aligned_verts = _apply_affine(np.asarray(surface.mesh.vertices), M)
    new_mesh = _rebuild_mesh(
        surface.mesh,
        vertices=aligned_verts,
        faces=surface.mesh.faces,
    )
    aligned_surface = cdc.TrimeshSurface(
        new_mesh, crs="ctf", units=surface.units,
    )

    aligned_landmarks = _transform_labeled_points(
        landmarks, lambda p: _apply_affine(p, M), "ctf"
    )

    return aligned_surface, aligned_landmarks, M


@cdc.validate_schemas
def revert_to_einstar_frame(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPoints,
    R_normalize: np.ndarray,
    M_align: np.ndarray,
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPoints]:
    """Map an aligned surface and landmarks back into the raw Einstar frame.

    Inverse of ``normalize_axes`` composed with ``align_axes_from_landmarks``,
    so the returned mesh and landmarks carry ``crs="digitized"`` and match
    the original ``read_einstar_obj`` output.

    Note that ``isolate_head`` is not invertible: the returned mesh is still
    head-only even though its coordinates are in the digitized frame.

    Args:
        surface: TrimeshSurface in the CTF frame (post
            ``align_axes_from_landmarks``, optionally after masking).
        landmarks: LabeledPoints in the CTF frame.
        R_normalize: 3x3 rotation returned by ``normalize_axes``.
        M_align: 4x4 affine returned by ``align_axes_from_landmarks``.

    Returns:
        Tuple of (surface_digitized, landmarks_digitized). Both carry
        ``crs="digitized"`` and the mesh preserves UVs / vertex colors.
    """
    M_inv = np.linalg.inv(M_align)
    R_inv4 = np.eye(4)
    R_inv4[:3, :3] = R_normalize.T
    M_total = R_inv4 @ M_inv

    raw_verts = _apply_affine(np.asarray(surface.mesh.vertices), M_total)
    new_mesh = _rebuild_mesh(
        surface.mesh,
        vertices=raw_verts,
        faces=surface.mesh.faces,
    )
    raw_surface = cdc.TrimeshSurface(
        new_mesh, crs="digitized", units=surface.units,
    )

    raw_landmarks = _transform_labeled_points(
        landmarks, lambda p: _apply_affine(p, M_total), "digitized"
    )

    return raw_surface, raw_landmarks
