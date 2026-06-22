"""Preprocessing: Y-anterior orientation, head isolation, and CTF alignment.

Gets a raw Einstar photogrammetry scan into a standard reference frame suitable for
landmark detection and mask building:

- ``orient_y_anterior`` rotates around X so Y points anterior (preliminary,
  used before head isolation).
- ``isolate_head`` strips body/shoulders/chair and disconnected fragments.
- ``align_to_ctf`` maps the scan into the CTF frame
  (+X=anterior, +Y=left, +Z=up, origin at the LPA-RPA midpoint) once all 5
  landmarks are available.
"""

import logging

import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt

from ._utils import (
    _apply_affine,
    _ear_midpoint,
    _largest_component_mask,
    _rebuild_mesh,
    _reindex_faces,
    _upper_head_centroid,
)

logger = logging.getLogger("cedalion")


@cdc.validate_schemas
def orient_y_anterior(
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
        ``rotated_nasion`` is in the same units as the input. The 4x4
        ``rotation_matrix`` carries no translation and is suitable for
        ``LabeledPoints.points.apply_transform`` directly.
    """
    vertices = surface.mesh.vertices

    centroid, _ = _upper_head_centroid(np.asarray(vertices))
    angle = np.arctan2(nasion[2] - centroid[2], nasion[1] - centroid[1])
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.eye(4)
    R[:3, :3] = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a],
    ])

    new_mesh = _rebuild_mesh(
        surface.mesh,
        vertices=vertices @ R[:3, :3].T,
        faces=surface.mesh.faces,
    )
    rotated_surface = cdc.TrimeshSurface(new_mesh, crs=surface.crs, units=surface.units)

    rotated_nasion = R[:3, :3] @ nasion

    logger.debug(
        f"orient_y_anterior: rotated {np.degrees(angle):.1f}deg around X. "
        f"Y now points anterior."
    )

    return rotated_surface, rotated_nasion, R


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

    The surface must be Y-anterior-oriented first (X=up, Y=anterior, Z=left).

    Args:
        surface: Y-anterior-oriented TrimeshSurface (post ``orient_y_anterior``).
        nasion: Nasion position as numpy array of shape (3,), in mm,
            in the Y-anterior-oriented frame (matching ``surface``).
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

    if head_mask.mean() > 0.95:
        logger.debug(
            f"Head isolation: sphere contains {head_mask.mean()*100:.0f}% "
            f"of vertices -- scan is already head-only."
        )
        return surface, head_mask

    if head_mask.sum() < 100:
        logger.warning(
            f"Head isolation: sphere matched only {int(head_mask.sum())} "
            f"vertices out of {len(vertices)}. Likely a centroid/radius "
            f"mismatch (check nasion alignment). Returning input unchanged."
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
def align_to_ctf(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPoints,
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPoints, cdt.AffineTransform]:
    """Map mesh + landmarks into the CTF anatomical frame.

    CTF convention:

        +X = anterior (toward Nz)
        +Y = left (toward Lpa)
        +Z = up (toward Cz)
        origin = midpoint of Lpa and Rpa (interaural midpoint)

    The returned surface and landmarks carry ``crs="ctf"``.

    Args:
        surface: Y-anterior-oriented TrimeshSurface (post ``orient_y_anterior`` and
            ``isolate_head``). Labels must already be canonicalized via
            ``normalize_landmarks_labels``.
        landmarks: LabeledPoints with canonical labels Nz, Iz, Cz, LPA, RPA
            (matching the surface frame).

    Returns:
        Tuple of (aligned_surface, aligned_landmarks, T_align).
        ``T_align`` is an :class:`~cedalion.typing.AffineTransform` with
        ``dims=["ctf", surface.crs]`` carrying pint units, suitable for
        ``points.apply_transform`` and ``TrimeshSurface.apply_transform``.

    Raises:
        ValueError: If any of the required landmarks (Nz, Iz, Cz, LPA, RPA)
            are missing, or if the landmarks are degenerate (LPA==RPA, or
            Nz on the LPA-RPA axis) so an orthonormal CTF basis cannot be
            constructed.
    """
    required = {"Nz", "Iz", "Cz", "LPA", "RPA"}
    missing = required - set(landmarks["label"].values.tolist())
    if missing:
        raise ValueError(f"Missing landmarks for alignment: {missing}")

    Nz = landmarks.sel(label="Nz").pint.dequantify().values
    Cz = landmarks.sel(label="Cz").pint.dequantify().values
    Lpa = landmarks.sel(label="LPA").pint.dequantify().values
    Rpa = landmarks.sel(label="RPA").pint.dequantify().values
    origin = _ear_midpoint(Lpa, Rpa)

    ear_axis = Lpa - Rpa
    ear_norm = float(np.linalg.norm(ear_axis))
    if ear_norm < 1e-6:
        raise ValueError(
            f"Degenerate landmarks: LPA and RPA coincide "
            f"(|LPA - RPA| = {ear_norm:.3e} mm). Cannot define CTF Y-axis."
        )
    y_ax = ear_axis / ear_norm

    nz_dir = Nz - origin
    nz_dir = nz_dir - np.dot(nz_dir, y_ax) * y_ax
    nz_norm = float(np.linalg.norm(nz_dir))
    if nz_norm < 1e-6:
        raise ValueError(
            f"Degenerate landmarks: Nz lies on the LPA-RPA axis "
            f"(perpendicular component = {nz_norm:.3e} mm). "
            f"Cannot define CTF X-axis."
        )
    x_ax = nz_dir / nz_norm

    z_ax = np.cross(x_ax, y_ax)

    if np.dot(Cz - origin, z_ax) < 0:
        z_ax = -z_ax
    y_ax = np.cross(z_ax, x_ax)

    R = np.vstack([x_ax, y_ax, z_ax])
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ origin

    units_str = str(surface.units)
    T_align = cdc.affine_transform_from_numpy(
        M,
        from_crs=surface.crs,
        to_crs="ctf",
        from_units=units_str,
        to_units=units_str,
    )

    aligned_landmarks = landmarks.points.apply_transform(T_align)

    aligned_verts = _apply_affine(np.asarray(surface.mesh.vertices), M)
    new_mesh = _rebuild_mesh(
        surface.mesh,
        vertices=aligned_verts,
        faces=surface.mesh.faces,
    )
    aligned_surface = cdc.TrimeshSurface(
        new_mesh, crs="ctf", units=surface.units,
    )

    return aligned_surface, aligned_landmarks, T_align


@cdc.validate_schemas
def revert_to_einstar_frame(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPoints,
    R_normalize: np.ndarray,
    T_align: cdt.AffineTransform,
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPoints]:
    """Map an aligned surface and landmarks back into the raw Einstar frame.

    Inverse of ``orient_y_anterior`` composed with ``align_to_ctf``,
    so the returned mesh and landmarks land back in the CRS the original
    ``align_to_ctf`` was called from (i.e. ``T_align.dims[1]``),
    matching ``read_einstar_obj``'s output.

    Note that ``isolate_head`` is not invertible: the returned mesh is still
    head-only even though its coordinates are in the original frame.

    Args:
        surface: TrimeshSurface in the CTF frame (post
            ``align_to_ctf``, optionally after masking).
        landmarks: LabeledPoints in the CTF frame.
        R_normalize: 4x4 rotation returned by ``orient_y_anterior`` (a same-CRS
            pre-rotation, so it stays as raw numpy rather than an
            :class:`~cedalion.typing.AffineTransform`).
        T_align: :class:`~cedalion.typing.AffineTransform` returned by
            ``align_to_ctf``, with dims ``[ctf, source_crs]``.

    Returns:
        Tuple of (surface_revert, landmarks_revert). The output CRS is taken
        from ``T_align.dims[1]`` (the original source CRS, typically
        ``"digitized"``); the mesh preserves UVs / vertex colors.
    """
    target_crs = T_align.dims[1]
    units_str = str(surface.units)
    M_align = T_align.pint.dequantify().values
    T_align_inv = cdc.affine_transform_from_numpy(
        np.linalg.inv(M_align),
        from_crs="ctf",
        to_crs=target_crs,
        from_units=units_str,
        to_units=units_str,
    )

    # R_normalize is a pure rotation (no translation), so its inverse equals
    # its transpose.
    R_inv4 = R_normalize.T

    landmarks_unaligned = landmarks.points.apply_transform(T_align_inv)
    raw_landmarks = landmarks_unaligned.points.apply_transform(R_inv4)

    # Mesh stays on _apply_affine + _rebuild_mesh per the texture-preservation
    # contract documented in _rebuild_mesh and _copy_visual.
    M_total = R_inv4 @ np.linalg.inv(M_align)
    raw_verts = _apply_affine(np.asarray(surface.mesh.vertices), M_total)
    new_mesh = _rebuild_mesh(
        surface.mesh,
        vertices=raw_verts,
        faces=surface.mesh.faces,
    )
    raw_surface = cdc.TrimeshSurface(
        new_mesh, crs=target_crs, units=surface.units,
    )

    return raw_surface, raw_landmarks
