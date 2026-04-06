"""Facial region detection for photogrammetry scans.

Provides axis normalization, head isolation, landmark detection from a
user-provided nasion (Nz) position, and facial region mask generation
using MediaPipe face contour back-projection.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging

import numpy as np
from scipy.spatial import KDTree
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion import Quantity, units

logger = logging.getLogger("cedalion")


def normalize_axes(
    surface: cdc.TrimeshSurface,
    nasion: np.ndarray,
    forward_direction: np.ndarray = None,
) -> tuple[cdc.TrimeshSurface, np.ndarray, np.ndarray]:
    """Rotate mesh around X-axis so Y=anterior (toward face), Z=left.

    The Einstar scanner has X=up (gravity-based) but Y/Z are arbitrary per
    scan session. This function uses the nasion position to determine the
    forward direction and rotates the mesh so that Y consistently points
    toward the face (anterior) and Z points left.

    Args:
        surface: TrimeshSurface in raw Einstar coordinates.
        nasion: Nasion position as numpy array of shape (3,).
        forward_direction: Optional pre-computed forward unit vector in YZ
            plane (X=0). If None, computed from nasion vs upper-head centroid.

    Returns:
        Tuple of (rotated_surface, rotated_nasion, rotation_matrix).
        rotation_matrix is 3x3 and can be applied to other points.
    """
    vertices = surface.mesh.vertices

    if forward_direction is None:
        # Compute forward direction: nasion minus upper-head centroid in YZ
        x_max = vertices[:, 0].max()
        x_min = vertices[:, 0].min()
        upper = vertices[vertices[:, 0] > x_min + 0.6 * (x_max - x_min)]
        centroid_yz = np.array([0.0, upper[:, 1].mean(), upper[:, 2].mean()])
        nasion_yz = np.array([0.0, nasion[1], nasion[2]])
        forward_direction = nasion_yz - centroid_yz
        fwd_norm = np.linalg.norm(forward_direction)
        if fwd_norm < 1e-6:
            logger.warning("Cannot compute forward direction — returning unchanged")
            return surface, nasion.copy(), np.eye(3)
        forward_direction = forward_direction / fwd_norm

    # Angle of forward_direction relative to +Y in the YZ plane
    angle = np.arctan2(forward_direction[2], forward_direction[1])

    # Rotation around X-axis by -angle to align forward with +Y
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    R = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cos_a, -sin_a],
        [0.0, sin_a, cos_a],
    ])

    # Rotate vertices
    rotated_verts = vertices @ R.T

    # Build new mesh with rotated vertices, same faces and texture
    import trimesh
    new_mesh = trimesh.Trimesh(
        vertices=rotated_verts,
        faces=surface.mesh.faces,
        visual=surface.mesh.visual,
        process=False,
    )
    rotated_surface = cdc.TrimeshSurface(new_mesh, crs=surface.crs, units=surface.units)

    # Rotate nasion
    rotated_nasion = R @ nasion

    logger.debug(
        f"Axis normalization: rotated {np.degrees(angle):.1f}deg around X. "
        f"Y now points anterior."
    )

    return rotated_surface, rotated_nasion, R


def isolate_head(
    surface: cdc.TrimeshSurface,
    nasion: np.ndarray,
    radius: float = 220.0,
) -> tuple[cdc.TrimeshSurface, np.ndarray]:
    """Remove shoulders, body, and chair — keep only the head.

    Uses a sphere centered on the upper-head centroid. The radius is
    chosen to capture the full head (~180mm wide x 230mm tall) while
    excluding shoulders and body. Scans that are already head-only are
    returned unchanged (the sphere just contains everything).

    The surface must be axis-normalized first (X=up, Y=anterior, Z=left).

    Args:
        surface: Axis-normalized TrimeshSurface.
        nasion: Nasion position as numpy array of shape (3,).
        radius: Sphere radius in mm (default 220). A human head has
            ~90mm radius; 220mm adds margin for ears and jaw.

    Returns:
        Tuple of (head_surface, head_mask). head_mask is a boolean
        array of shape (n_vertices,) indicating which original
        vertices were kept.
    """
    import trimesh

    vertices = surface.mesh.vertices
    faces = surface.mesh.faces

    # Sphere center: Y/Z from upper-head centroid (good lateral centering).
    # X: use the lower of (upper centroid, midpoint of top+nasion).
    x_max = vertices[:, 0].max()
    x_min = vertices[:, 0].min()
    upper = vertices[:, 0] > x_min + 0.6 * (x_max - x_min)
    center = vertices[upper].mean(axis=0)
    midpoint_x = (x_max + nasion[0]) / 2.0
    center[0] = min(center[0], midpoint_x)

    # Sphere mask
    dist = np.linalg.norm(vertices - center, axis=1)
    head_mask = dist < radius

    # If the sphere captures almost everything, skip trimming
    if head_mask.sum() < 100 or head_mask.mean() > 0.95:
        logger.debug(
            f"Head isolation: sphere contains {head_mask.mean()*100:.0f}% "
            f"of vertices — scan is already head-only"
        )
        return surface, head_mask

    # Build new mesh with only head faces
    # A face is kept if ALL its vertices are in the head
    face_mask = head_mask[faces].all(axis=1)
    head_faces = faces[face_mask]

    # Reindex vertices: only keep referenced vertices
    kept_verts = np.unique(head_faces)
    reindex = np.full(len(vertices), -1, dtype=int)
    reindex[kept_verts] = np.arange(len(kept_verts))
    new_faces = reindex[head_faces]
    new_verts = vertices[kept_verts]

    # Transfer visual — subset vertex colors to match kept vertices
    new_mesh = trimesh.Trimesh(
        vertices=new_verts,
        faces=new_faces,
        process=False,
    )
    try:
        old_visual = surface.mesh.visual
        if hasattr(old_visual, 'vertex_colors') and len(old_visual.vertex_colors) == len(vertices):
            new_mesh.visual.vertex_colors = old_visual.vertex_colors[kept_verts]
    except Exception:
        pass

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


def detect_landmarks_from_nasion(
    surface: cdc.TrimeshSurface,
    nz_position: np.ndarray,
) -> cdt.LabeledPointCloud:
    """Detect anatomical landmarks from a user-provided nasion (Nz) position.

    **Important:** The surface must be axis-normalized before calling this
    function. Use ``normalize_axes()`` first so that:
    - X = up (vertical, from Einstar gravity sensor)
    - Y = anterior (toward face, from nasion direction)
    - Z = left of subject

    Given the Nz point, automatically detects Iz, LPA, RPA, Cz:
    - Cz: Highest vertex (max X) near midline
    - Iz: Most posterior vertex (min Y) at nasion height
    - LPA: Most left vertex (max Z) at ear height
    - RPA: Most right vertex (min Z) at ear height

    Args:
        surface: Axis-normalized TrimeshSurface (Y=anterior, Z=left)
        nz_position: Nasion position as numpy array of shape (3,), in mm

    Returns:
        LabeledPointCloud with landmarks labeled as Nz, Iz, Cz, LPA, RPA

    Raises:
        ValueError: If landmark configuration fails validation

    Example:
        >>> surface_norm, nz_norm, R = normalize_axes(surface, nz, fwd)
        >>> landmarks = detect_landmarks_from_nasion(surface_norm, nz_norm)
    """
    nz_position = np.asarray(nz_position, dtype=float)
    vertices = surface.mesh.vertices

    centroid = vertices.mean(axis=0)
    head_height = vertices[:, 0].max() - vertices[:, 0].min()
    head_verts = vertices

    logger.debug(
        f"Detecting landmarks from Nz={nz_position}, "
        f"head_height={head_height:.1f}mm, "
        f"head_verts={len(vertices)}"
    )

    # Cz: highest vertex (max X) near midline
    band = 0.20 * head_height
    lateral_mask = (
        (np.abs(head_verts[:, 1] - centroid[1]) < band)
        & (np.abs(head_verts[:, 2] - centroid[2]) < band)
    )
    if lateral_mask.sum() == 0:
        lateral_mask = np.ones(len(head_verts), dtype=bool)
    cz_idx = np.where(lateral_mask)[0][np.argmax(head_verts[lateral_mask, 0])]
    cz = head_verts[cz_idx]

    # Iz: most posterior (min Y) at nasion height, on the midsagittal plane.
    iz_height_mask = np.abs(head_verts[:, 0] - nz_position[0]) < 20.0
    iz_midline_mask = np.abs(head_verts[:, 2] - nz_position[2]) < 25.0
    iz_mask = iz_height_mask & iz_midline_mask
    if iz_mask.sum() == 0:
        iz_mask = np.abs(head_verts[:, 0] - nz_position[0]) < 40.0
    iz_cands = np.where(iz_mask)[0]
    iz_idx = iz_cands[np.argmin(head_verts[iz_cands, 1])]
    iz = head_verts[iz_idx]

    # LPA/RPA: geometric targets, refined by snapping to surface
    from scipy.spatial import cKDTree

    midline_z = np.mean([nz_position[2], iz[2], cz[2]])
    clean_band = (
        (head_verts[:, 0] > nz_position[0] + 10.0)
        & (head_verts[:, 0] < cz[0] - 20.0)
    )
    if clean_band.sum() > 100:
        clean_z = head_verts[clean_band, 2]
        half_width = (
            np.percentile(clean_z, 97) - np.percentile(clean_z, 3)
        ) / 2.0
    else:
        half_width = 75.0

    geo_lpa_target = np.array([nz_position[0], cz[1], midline_z + half_width])
    geo_rpa_target = np.array([nz_position[0], cz[1], midline_z - half_width])

    def _snap_ear(target):
        """Snap target to nearest surface vertex in ear region."""
        ear_region = (
            (np.abs(head_verts[:, 0] - nz_position[0]) < 30.0)
            & (np.abs(head_verts[:, 1] - cz[1]) < 40.0)
        )
        if ear_region.sum() > 10:
            idxs = np.where(ear_region)[0]
            tree = cKDTree(head_verts[idxs])
            _, local = tree.query(target)
            return head_verts[idxs[local]]
        tree = cKDTree(head_verts)
        _, idx = tree.query(target)
        return head_verts[idx]

    lpa = _snap_ear(geo_lpa_target)
    rpa = _snap_ear(geo_rpa_target)

    landmark_positions = {
        "Nz": nz_position,
        "Iz": iz,
        "Cz": cz,
        "LPA": lpa,
        "RPA": rpa,
    }

    logger.debug(
        f"Detected landmarks: Cz={cz}, Iz={iz}, LPA={lpa}, RPA={rpa}"
    )

    # Validate
    _validate_landmark_configuration(landmark_positions, centroid)

    # Create LabeledPointCloud
    labels = ["Nz", "Iz", "Cz", "LPA", "RPA"]
    coords = np.array([landmark_positions[label] for label in labels])

    landmarks = xr.DataArray(
        coords,
        dims=["label", surface.crs],
        coords={
            "label": labels,
            "type": ("label", [cdc.PointType.LANDMARK] * 5),
        },
    ).pint.quantify("mm")

    return landmarks


def get_facial_region_mask_from_nasion(
    surface: cdc.TrimeshSurface,
    nz: np.ndarray,
    forward_direction: np.ndarray,
    face_contour_3d: np.ndarray | None = None,
    protected_points: cdt.LabeledPointCloud = None,
    protection_radius: Quantity = 15.0 * units.mm,
    lateral_extension: float = 70.0,
    lower_width_scale: float = 2.5,
) -> np.ndarray:
    """Generate facial region mask using nasion and MediaPipe face contour.

    This function is axis-independent -- it uses the forward direction and
    face contour from MediaPipe rather than relying on axis alignment.
    It does NOT depend on LPA/RPA accuracy.

    The mask uses a two-part strategy derived entirely from the contour:
    - **Oval**: The MediaPipe face contour polygon defines precise
      forehead and jawline boundaries (with a depth filter to prevent
      top-of-head bleed).
    - **Side band**: 3D proximity to contour points -- any vertex within
      ``lateral_extension`` mm of a contour point is included, vertically
      constrained to chin-to-temple height to avoid top-of-head bleed.
      This naturally follows the head curvature and covers the ears.

    When no face contour is available (manual nasion mode), falls back to
    a forward-facing hemisphere around the nasion.

    Args:
        surface: The mesh surface (axis-normalized or not)
        nz: Nasion position as numpy array of shape (3,)
        forward_direction: Unit vector pointing toward the face
        face_contour_3d: 3D face oval points from MediaPipe back-projection,
            shape (N, 3) with N >= 20, or None for hemisphere fallback
        protected_points: Points to exclude (optodes + anatomical landmarks)
        protection_radius: Radius around protected points
        lateral_extension: 3D proximity radius in mm from contour points
            for ear/temple coverage (default 70.0)
        lower_width_scale: Factor to widen the lower portion of the face
            contour (below nasion). 1.0 = no change, 2.5 = 150% wider at
            chin level. Smoothly blends from 1.0 at nasion to this value
            at chin. Keeps forehead border unchanged. (default 2.5)

    Returns:
        Boolean array of shape (n_vertices,) where True = facial region
    """
    from matplotlib.path import Path

    vertices = surface.mesh.vertices
    nz = np.asarray(nz, dtype=float)
    fwd = np.asarray(forward_direction, dtype=float)
    fwd = fwd / np.linalg.norm(fwd)

    if face_contour_3d is not None and len(face_contour_3d) >= 20:
        # --- MediaPipe face oval approach (spherical projection) ---
        up_hint = np.array([1.0, 0.0, 0.0])  # X = up in Einstar coords
        if abs(np.dot(fwd, up_hint)) > 0.9:
            up_hint = np.array([0.0, 0.0, 1.0])
        u_axis = np.cross(fwd, up_hint)
        u_axis = u_axis / np.linalg.norm(u_axis)
        v_axis = np.cross(fwd, u_axis)
        v_axis = v_axis / np.linalg.norm(v_axis)

        # Head center: ~80mm behind nasion (roughly center of skull)
        head_center = nz - fwd * 80.0

        # Project contour onto unit sphere -> 2D
        c_dirs = face_contour_3d - head_center
        c_dirs = c_dirs / np.linalg.norm(c_dirs, axis=1, keepdims=True)
        contour_2d = np.column_stack([c_dirs @ u_axis, c_dirs @ v_axis])

        # Project vertices onto unit sphere -> 2D
        v_dirs = vertices - head_center
        v_fwd = v_dirs @ fwd  # positive = face side of head
        v_dirs = v_dirs / np.linalg.norm(v_dirs, axis=1, keepdims=True)
        verts_2d = np.column_stack([v_dirs @ u_axis, v_dirs @ v_axis])

        # Face mask = forward-facing vertices inside the contour polygon
        polygon = Path(contour_2d)
        facial_mask = polygon.contains_points(verts_2d) & (v_fwd > 0)

        logger.debug(
            f"Contour mask: {facial_mask.sum()} vertices "
            f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
        )
    else:
        # --- Fallback: forward-facing hemisphere ---
        verts_rel = vertices - nz
        fwd_dist = verts_rel @ fwd
        dist_to_nz = np.linalg.norm(verts_rel, axis=1)

        facial_mask = (fwd_dist > -20.0) & (dist_to_nz < 120.0)

        logger.debug(
            f"Hemisphere fallback mask: {facial_mask.sum()} vertices "
            f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
        )

    # Exclude protection zones
    protection_radius_mm = float(protection_radius.to("mm").magnitude)
    if protected_points is not None:
        protected_positions = protected_points.pint.dequantify().values
        if len(protected_positions) > 0:
            kdtree = KDTree(protected_positions)
            distances, _ = kdtree.query(vertices, k=1)
            protected_mask = distances < protection_radius_mm
            facial_mask = facial_mask & ~protected_mask

    logger.debug(
        f"Final facial mask: {facial_mask.sum()} of {len(facial_mask)} vertices "
        f"({100 * facial_mask.sum() / len(facial_mask):.1f}%)"
    )

    return facial_mask


def _validate_landmark_configuration(
    landmarks: dict[str, np.ndarray],
    centroid: np.ndarray,
) -> None:
    """Validate that detected landmarks have a plausible spatial configuration.

    Checks:
    - Cz is the highest point (max X)
    - Iz is posterior to the centroid (low Y)
    - LPA and RPA are roughly symmetric about the midline (Z)

    Args:
        landmarks: Dict mapping landmark name to position array
        centroid: Mesh centroid for reference

    Raises:
        ValueError: If configuration is implausible
    """
    cz = landmarks["Cz"]
    iz = landmarks["Iz"]
    lpa = landmarks["LPA"]
    rpa = landmarks["RPA"]

    # Cz should be highest (max X)
    all_x = [landmarks[k][0] for k in landmarks]
    if cz[0] < max(all_x) - 1.0:
        logger.warning("Cz is not the highest landmark — detection may be off")

    # Iz should be posterior to centroid (Y < centroid Y)
    if iz[1] > centroid[1]:
        logger.warning(
            f"Iz (Y={iz[1]:.1f}) is anterior to centroid (Y={centroid[1]:.1f}) "
            "— expected posterior"
        )

    # LPA should be left (Z > centroid Z) and RPA right (Z < centroid Z)
    if lpa[2] < rpa[2]:
        logger.warning(
            "LPA is to the right of RPA — landmarks may be swapped"
        )

    # LPA and RPA should be roughly symmetric
    lpa_offset = abs(lpa[2] - centroid[2])
    rpa_offset = abs(rpa[2] - centroid[2])
    if min(lpa_offset, rpa_offset) > 0 and max(lpa_offset, rpa_offset) / min(lpa_offset, rpa_offset) > 3.0:
        logger.warning(
            f"LPA/RPA asymmetry is large (offsets: {lpa_offset:.1f} vs {rpa_offset:.1f}mm)"
        )
