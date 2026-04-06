"""Automatic nasion detection from 3D mesh geometry.

Detects the nasion (Nz) from an Einstar scan without user interaction.
Only assumes X (index 0) = Up (gravity-based). The forward direction
in the Y-Z horizontal plane is determined automatically using MediaPipe
Face Landmarker: render the mesh from multiple viewpoints, detect the
face in 2D, back-project landmarks to 3D.

Returns None when automatic detection fails (e.g. EEG cap false
positives). The caller should fall back to manual nasion selection.

Initial Contributors:
    - Face Anonymization Project | 2024
"""

import logging
import pathlib

import numpy as np
from scipy.spatial import KDTree

import cedalion.dataclasses as cdc

logger = logging.getLogger("cedalion")

_MODEL_DIR = pathlib.Path(__file__).parent
_MODEL_PATH = _MODEL_DIR / "face_landmarker.task"


def _filter_contour_outliers(
    contour_3d: np.ndarray,
    max_sigma: float = 2.5,
) -> np.ndarray:
    """Remove outlier points from the face contour.

    Rejects points whose distance from the contour centroid exceeds
    max_sigma standard deviations. This catches "floating bits" from
    seam cracks where VTK picks a disconnected mesh fragment.

    Args:
        contour_3d: Face contour points, shape (N, 3).
        max_sigma: Rejection threshold in standard deviations.

    Returns:
        Filtered contour points, shape (M, 3) where M <= N.
    """
    centroid = contour_3d.mean(axis=0)
    dists = np.linalg.norm(contour_3d - centroid, axis=1)
    std_dist = dists.std()
    if std_dist < 1e-6:
        return contour_3d
    keep = dists < dists.mean() + max_sigma * std_dist
    return contour_3d[keep]


def detect_nasion_auto(
    surface: cdc.TrimeshSurface,
) -> tuple[np.ndarray, dict] | None:
    """Automatically detect the nasion using MediaPipe face detection.

    Renders the mesh from multiple viewpoints, runs MediaPipe Face
    Landmarker to find the face, then refines the nasion via
    midsagittal profile analysis.

    Returns None when detection fails -- the caller should fall back
    to manual nasion selection (see ``pick_nasion`` in ``ui.py``).

    Args:
        surface: TrimeshSurface from photogrammetry scan.

    Returns:
        Tuple of (nasion_position, metadata) or None if detection fails.
        nasion_position is a numpy array of shape (3,) in mm.
        metadata is a dict with keys: method, confidence, nose_tip,
        forward_direction, face_contour_3d.
    """
    vertices = surface.mesh.vertices

    # --- Step 1: Merge seam vertices to eliminate crack noise ---
    merged_verts = _merge_close_vertices(vertices)

    # --- Step 2: Isolate head (remove chair/body) ---
    head_verts, _ = _isolate_head_vertices(merged_verts)

    # --- Step 3: YZ centroid of upper head (needed for profile analysis) ---
    x_min = head_verts[:, 0].min()
    x_max = head_verts[:, 0].max()
    upper = head_verts[head_verts[:, 0] > x_min + 0.6 * (x_max - x_min)]
    yz_centroid = np.array([0.0, upper[:, 1].mean(), upper[:, 2].mean()])

    # --- Step 4: Try MediaPipe face detection ---
    ml_result = _find_nose_tip_mediapipe(surface.mesh, head_verts)

    if ml_result is None:
        logger.info("Automatic nasion detection failed — no face found")
        return None

    nose_tip_ml, forward_dir, nasion_proxy, frontal_reliable, face_contour = ml_result
    logger.debug(
        f"MediaPipe face found, "
        f"fwd_angle={np.degrees(np.arctan2(forward_dir[2], forward_dir[1])):.0f}deg, "
        f"frontal_reliable={frontal_reliable}, "
        f"face_contour={'yes' if face_contour is not None else 'no'}"
    )

    # --- Primary: use frontal re-render nasion directly ---
    if frontal_reliable:
        nasion = _snap_to_original(nasion_proxy, vertices)
        logger.debug(f"Auto nasion (frontal): {nasion}")
        return nasion, {
            "method": "mediapipe+frontal",
            "confidence": 0.90,
            "nose_tip": nose_tip_ml.copy(),
            "forward_direction": forward_dir.copy(),
            "face_contour_3d": face_contour,
        }

    # --- Fallback: geometric profile analysis ---
    result = _try_direction(
        head_verts, vertices, forward_dir, yz_centroid,
    )
    if result is not None:
        result["method"] = "mediapipe+profile"
        if result["dip"] > 0:
            result["confidence"] = 0.85
        else:
            result["confidence"] = 0.7
        logger.debug(
            f"Auto nasion (ML+profile): {result['nasion']}, "
            f"confidence={result['confidence']:.2f}, dip={result['dip']:.2f}"
        )
        return result["nasion"], {
            "method": result["method"],
            "confidence": result["confidence"],
            "nose_tip": result["nose_tip"],
            "forward_direction": result["forward_direction"],
            "face_contour_3d": face_contour,
        }

    logger.info("Profile analysis failed after MediaPipe detection")
    return None


# ---------------------------------------------------------------------------
# MediaPipe-based detection
# ---------------------------------------------------------------------------

def _find_nose_tip_mediapipe(
    mesh_trimesh,
    head_verts: np.ndarray,
    cam_distance: float = 400.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool] | None:
    """Detect face direction and nasion using two-pass MediaPipe rendering.

    Pass 1: Renders the mesh from 12 viewpoints to find the forward
    direction (which way the face points).

    Pass 2: Re-renders from directly in front and back-projects
    MediaPipe landmark 168 (nasion) from this frontal view. From a
    frontal camera, the nasion pixel maps correctly to the bridge of
    the nose rather than the side.

    Args:
        mesh_trimesh: trimesh.Trimesh mesh object (with texture).
        head_verts: Head-isolated vertices for centroid computation.
        cam_distance: Camera distance from head centroid in mm.

    Returns:
        Tuple of (nose_tip_3d, forward_direction, nasion_3d,
        frontal_reliable, face_contour) or None if no face detected.
        frontal_reliable is True when the frontal re-render succeeded.
    """
    try:
        import pyvista as pv
        import mediapipe as mp
        import vtk
    except ImportError as e:
        logger.info(f"MediaPipe detection unavailable: {e}")
        return None

    if not _MODEL_PATH.exists():
        logger.info(f"MediaPipe model not found at {_MODEL_PATH}")
        return None

    head_centroid = head_verts.mean(axis=0)

    # Build pyvista mesh with texture colors
    faces_pv = np.column_stack([
        np.full(len(mesh_trimesh.faces), 3), mesh_trimesh.faces
    ]).ravel()
    pv_mesh = pv.PolyData(mesh_trimesh.vertices, faces_pv)

    try:
        colors = mesh_trimesh.visual.to_color().vertex_colors[:, :3]
        pv_mesh["colors"] = colors
        has_colors = True
    except Exception:
        has_colors = False

    plotter = pv.Plotter(off_screen=True, window_size=[1024, 768])
    if has_colors:
        plotter.add_mesh(pv_mesh, scalars="colors", rgb=True)
    else:
        plotter.add_mesh(pv_mesh, color="bisque")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
        output_facial_transformation_matrixes=True,
    )

    result = None
    try:
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as lm:
            # === Pass 1: Find forward direction ===
            pass1_fwd = None
            pass1_nose_3d = None
            pass1_nz_3d = None

            for theta_deg in range(0, 360, 30):
                theta = np.radians(theta_deg)
                cam_offset = np.array([
                    0,
                    cam_distance * np.cos(theta),
                    cam_distance * np.sin(theta),
                ])
                cam_pos = head_centroid + cam_offset

                plotter.camera_position = [
                    cam_pos.tolist(),
                    head_centroid.tolist(),
                    [1, 0, 0],
                ]
                plotter.render()
                img = plotter.screenshot(return_img=True)
                H, W = img.shape[:2]

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(img),
                )
                detection = lm.detect(mp_image)

                if not detection.face_landmarks:
                    continue

                # Validate yaw
                if detection.facial_transformation_matrixes:
                    from scipy.spatial.transform import Rotation
                    mat = detection.facial_transformation_matrixes[0]
                    rot = Rotation.from_matrix(np.array(mat)[:3, :3])
                    yaw = rot.as_euler("yxz", degrees=True)[0]
                    if abs(yaw) > 75:
                        continue

                face = detection.face_landmarks[0]
                nose_lm = face[1]
                nasion_lm = face[168]

                # Back-project nose tip to validate detection
                picker = vtk.vtkCellPicker()
                picker.SetTolerance(0.01)
                picker.Pick(
                    float(int(nose_lm.x * W)),
                    float(H - 1 - int(nose_lm.y * H)),
                    0, plotter.renderer,
                )
                nose_3d = np.array(picker.GetPickPosition())
                if np.allclose(nose_3d, 0):
                    continue

                # Reject nose on top of head
                x_min_h = head_verts[:, 0].min()
                x_max_h = head_verts[:, 0].max()
                nose_rel = (nose_3d[0] - x_min_h) / (x_max_h - x_min_h)
                if nose_rel > 0.85:
                    continue

                # Forward direction: nose tip -> centroid in YZ plane
                fwd = nose_3d - head_centroid
                fwd[0] = 0.0
                fwd_norm = np.linalg.norm(fwd)
                if fwd_norm < 1e-6:
                    continue
                fwd = fwd / fwd_norm

                pass1_fwd = fwd.copy()
                pass1_nose_3d = nose_3d.copy()

                # Also grab nasion proxy from this view
                picker.Pick(
                    float(int(nasion_lm.x * W)),
                    float(H - 1 - int(nasion_lm.y * H)),
                    0, plotter.renderer,
                )
                pass1_nz_3d = np.array(picker.GetPickPosition()).copy()

                logger.debug(f"Pass 1: face found at {theta_deg}deg")
                break

            if pass1_fwd is None:
                result = None
            else:
                # === Pass 2: Frontal re-render for accurate nasion ===
                frontal_cam_pos = head_centroid + pass1_fwd * cam_distance
                plotter.camera_position = [
                    frontal_cam_pos.tolist(),
                    head_centroid.tolist(),
                    [1, 0, 0],
                ]
                plotter.render()
                frontal_img = plotter.screenshot(return_img=True)
                H_f, W_f = frontal_img.shape[:2]

                mp_frontal = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(frontal_img),
                )
                frontal_det = lm.detect(mp_frontal)

                frontal_reliable = False
                face_contour = None
                final_nz_3d = pass1_nz_3d
                final_nose_3d = pass1_nose_3d

                if frontal_det.face_landmarks:
                    f_face = frontal_det.face_landmarks[0]

                    # Back-project nasion (landmark 168) from frontal view
                    nz_lm_f = f_face[168]
                    picker_f = vtk.vtkCellPicker()
                    picker_f.SetTolerance(0.01)
                    picker_f.Pick(
                        float(int(nz_lm_f.x * W_f)),
                        float(H_f - 1 - int(nz_lm_f.y * H_f)),
                        0, plotter.renderer,
                    )
                    nz_3d_f = np.array(picker_f.GetPickPosition())

                    # Back-project nose tip (landmark 1) from frontal view
                    nose_lm_f = f_face[1]
                    picker_f.Pick(
                        float(int(nose_lm_f.x * W_f)),
                        float(H_f - 1 - int(nose_lm_f.y * H_f)),
                        0, plotter.renderer,
                    )
                    nose_3d_f = np.array(picker_f.GetPickPosition())

                    # Validate: both on mesh, nasion above nose tip
                    if (not np.allclose(nz_3d_f, 0)
                            and not np.allclose(nose_3d_f, 0)
                            and nz_3d_f[0] > nose_3d_f[0]):
                        final_nz_3d = nz_3d_f.copy()
                        final_nose_3d = nose_3d_f.copy()
                        frontal_reliable = True
                        logger.debug(
                            f"Pass 2 (frontal): nasion={final_nz_3d}, "
                            f"nose={final_nose_3d}"
                        )

                        # === Extract face oval contour ===
                        _FACE_OVAL_INDICES = [
                            10, 338, 297, 332, 284, 251, 389, 356,
                            454, 323, 361, 288, 397, 365, 379, 378,
                            400, 377, 152, 148, 176, 149, 150, 136,
                            172, 58, 132, 93, 234, 127, 162, 21,
                            54, 103, 67, 109,
                        ]
                        contour_pts = []
                        picker_c = vtk.vtkCellPicker()
                        picker_c.SetTolerance(0.01)
                        for idx in _FACE_OVAL_INDICES:
                            lm_pt = f_face[idx]
                            picker_c.Pick(
                                float(int(lm_pt.x * W_f)),
                                float(H_f - 1 - int(lm_pt.y * H_f)),
                                0, plotter.renderer,
                            )
                            pt_3d = np.array(picker_c.GetPickPosition())
                            if not np.allclose(pt_3d, 0):
                                contour_pts.append(pt_3d.copy())

                        if len(contour_pts) >= 20:
                            raw_contour = np.array(contour_pts)
                            face_contour = _filter_contour_outliers(
                                raw_contour
                            )
                            n_removed = len(raw_contour) - len(face_contour)
                            if len(face_contour) < 20:
                                face_contour = raw_contour
                                n_removed = 0
                            logger.debug(
                                f"Face oval: {len(contour_pts)}/36 "
                                f"back-projected, {n_removed} outliers removed"
                            )
                        else:
                            logger.debug(
                                f"Face oval: only {len(contour_pts)}/36 "
                                f"back-projected, skipping"
                            )
                    else:
                        logger.debug(
                            "Pass 2: frontal back-projection failed validation"
                        )
                else:
                    logger.debug("Pass 2: no face in frontal re-render")

                # Recompute forward from frontal nose tip if available
                if frontal_reliable:
                    final_fwd = final_nose_3d - head_centroid
                    final_fwd[0] = 0.0
                    fn = np.linalg.norm(final_fwd)
                    if fn > 1e-6:
                        final_fwd = final_fwd / fn
                    else:
                        final_fwd = pass1_fwd.copy()
                else:
                    final_fwd = pass1_fwd.copy()

                result = (
                    final_nose_3d.copy(),
                    final_fwd,
                    final_nz_3d.copy(),
                    frontal_reliable,
                    face_contour,
                )
    except Exception as e:
        logger.warning(f"MediaPipe detection error: {e}")
    finally:
        plotter.close()

    return result


# ---------------------------------------------------------------------------
# Geometric detection (fallback)
# ---------------------------------------------------------------------------

def _try_direction(
    head_verts: np.ndarray,
    original_verts: np.ndarray,
    forward_dir: np.ndarray,
    yz_centroid: np.ndarray,
    nose_tip_override: np.ndarray | None = None,
) -> dict | None:
    """Try a candidate forward direction and return nasion result with dip score.

    Args:
        head_verts: Head-isolated vertices.
        original_verts: Original mesh vertices for snapping.
        forward_dir: Candidate forward unit vector (X=0).
        yz_centroid: YZ centroid of upper head (X=0).
        nose_tip_override: If provided, use this as the nose tip instead of
            detecting it geometrically. Useful when ML provides a nose tip.

    Returns:
        Dict with nasion, confidence, dip score, etc., or None on failure.
    """
    lateral_dir = np.cross(np.array([1.0, 0.0, 0.0]), forward_dir)
    lateral_dir = lateral_dir / np.linalg.norm(lateral_dir)

    if nose_tip_override is not None:
        nose_tip = nose_tip_override.copy()
    else:
        nose_tip, _ = _find_nose_tip(head_verts, forward_dir)

    lat_dist = np.abs(np.dot(head_verts - nose_tip, lateral_dir))
    midline = lat_dist < 10.0
    above = head_verts[:, 0] > nose_tip[0]
    fwd_proj = np.dot(head_verts, forward_dir)
    centroid_fwd = np.dot(yz_centroid, forward_dir)
    anterior = fwd_proj > centroid_fwd

    mask = midline & above & anterior
    candidates = np.where(mask)[0]

    if len(candidates) < 5:
        mask = midline & above
        candidates = np.where(mask)[0]

    if len(candidates) < 5:
        return None

    # Bin by X height, take median forward-projection per bin
    cand_x = head_verts[candidates, 0]
    cand_fwd = fwd_proj[candidates]

    x_min, x_max = cand_x.min(), cand_x.max()
    n_bins = max(10, int((x_max - x_min) / 1.0))
    bin_edges = np.linspace(x_min, x_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_indices = np.clip(np.digitize(cand_x, bin_edges) - 1, 0, n_bins - 1)

    bin_fwd = np.full(n_bins, np.nan)
    for b in range(n_bins):
        in_bin = bin_indices == b
        if in_bin.sum() > 0:
            bin_fwd[b] = np.median(cand_fwd[in_bin])

    valid = ~np.isnan(bin_fwd)
    bin_centers_valid = bin_centers[valid]
    bin_fwd_valid = bin_fwd[valid]

    if len(bin_fwd_valid) < 3:
        return None

    # Smooth profile to suppress mesh noise
    window = min(7, len(bin_fwd_valid))
    if window >= 3:
        kernel = np.ones(window) / window
        padded = np.pad(bin_fwd_valid, (window // 2, window // 2), mode='edge')
        bin_fwd_smoothed = np.convolve(padded, kernel, mode='valid')
    else:
        bin_fwd_smoothed = bin_fwd_valid

    # Search in anatomically plausible range: nose_tip to nose_tip + 40mm
    nose_tip_x = nose_tip[0]
    search_mask = (
        (bin_centers_valid >= nose_tip_x)
        & (bin_centers_valid <= nose_tip_x + 40)
    )
    search_idx = np.where(search_mask)[0]

    if len(search_idx) >= 3:
        nasion_bin = _find_deepest_minimum(
            bin_fwd_smoothed[search_idx], min_dip=0.5
        )
        if nasion_bin is not None:
            nasion_bin = search_idx[nasion_bin]
        else:
            nasion_bin = search_idx[np.argmin(bin_fwd_smoothed[search_idx])]
    else:
        nasion_bin = _find_deepest_minimum(bin_fwd_smoothed, min_dip=0.5)
        if nasion_bin is None:
            half = max(1, len(bin_fwd_smoothed) // 2)
            nasion_bin = np.argmin(bin_fwd_smoothed[:half])

    nasion_x = bin_centers_valid[nasion_bin]
    nasion_fwd_val = bin_fwd_valid[nasion_bin]

    # Find best candidate vertex near this height
    bin_half = (bin_centers_valid[1] - bin_centers_valid[0]) if len(
        bin_centers_valid) > 1 else 2.0
    height_match = np.abs(head_verts[candidates, 0] - nasion_x) < bin_half * 2
    if height_match.sum() > 0:
        height_cands = candidates[height_match]
        fwd_dist = np.abs(fwd_proj[height_cands] - nasion_fwd_val)
        best_cand = height_cands[np.argmin(fwd_dist)]
        nasion_approx = head_verts[best_cand]
    else:
        nasion_approx = np.array([nasion_x, nose_tip[1], nose_tip[2]])
    nasion = _snap_to_original(nasion_approx, original_verts)

    # Dip depth (using smoothed profile)
    if 0 < nasion_bin < len(bin_fwd_smoothed) - 1:
        dip = (min(bin_fwd_smoothed[nasion_bin - 1], bin_fwd_smoothed[nasion_bin + 1])
               - bin_fwd_smoothed[nasion_bin])
    else:
        dip = 0.0

    confidence = float(min(1.0, dip / 5.0))

    return {
        "nasion": nasion,
        "nose_tip": nose_tip.copy(),
        "forward_direction": forward_dir.copy(),
        "confidence": confidence,
        "dip": dip,
        "method": "profile",
    }


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _merge_close_vertices(vertices: np.ndarray, tol: float = 0.01) -> np.ndarray:
    """Merge spatially coincident vertices (seam duplicates).

    Returns a deduplicated vertex array. Uses union-find to group vertices
    within tolerance, then averages their positions.

    Args:
        vertices: Mesh vertices of shape (N, 3).
        tol: Distance tolerance in mm.

    Returns:
        Unique vertices array of shape (M, 3) where M <= N.
    """
    tree = KDTree(vertices)
    pairs = tree.query_pairs(r=tol)

    if not pairs:
        return vertices.copy()

    n = len(vertices)
    parent = np.arange(n)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(n):
        parent[i] = find(i)

    unique_roots, inverse = np.unique(parent, return_inverse=True)
    n_new = len(unique_roots)

    new_verts = np.zeros((n_new, 3))
    counts = np.zeros(n_new)
    for i in range(n):
        new_verts[inverse[i]] += vertices[i]
        counts[inverse[i]] += 1
    new_verts /= counts[:, None]

    logger.debug(f"Merged seam vertices: {n} -> {n_new}")
    return new_verts


def _isolate_head_vertices(
    vertices: np.ndarray,
    max_head_height: float = 280.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Isolate head vertices by cropping to within max_head_height of the top.

    Args:
        vertices: Mesh vertices of shape (N, 3).
        max_head_height: Maximum expected head height in mm.

    Returns:
        Tuple of (head_vertices, mask) where mask is boolean array.
    """
    x_max = vertices[:, 0].max()
    mask = vertices[:, 0] > x_max - max_head_height

    if mask.sum() < 100:
        logger.warning("Head isolation: too few vertices, using all")
        mask = np.ones(len(vertices), dtype=bool)

    logger.debug(f"Head isolation: {len(vertices)} -> {mask.sum()} vertices")
    return vertices[mask], mask


def _find_nose_tip(
    vertices: np.ndarray,
    forward_direction: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Find nose tip as the most-forward vertex in the upper head.

    Args:
        vertices: Head-isolated vertices of shape (N, 3).
        forward_direction: Unit vector pointing forward in Y-Z plane.

    Returns:
        Tuple of (nose_tip_position, vertex_index).
    """
    x_min = vertices[:, 0].min()
    x_max = vertices[:, 0].max()
    x_threshold = x_min + 0.6 * (x_max - x_min)
    upper_mask = vertices[:, 0] > x_threshold

    if upper_mask.sum() == 0:
        upper_mask = np.ones(len(vertices), dtype=bool)

    fwd_proj = np.dot(vertices, forward_direction)
    fwd_proj[~upper_mask] = -np.inf
    idx = np.argmax(fwd_proj)

    return vertices[idx].copy(), idx


def _find_deepest_minimum(values: np.ndarray, min_dip: float = 0.5):
    """Find the deepest local minimum that dips at least min_dip below neighbors.

    Args:
        values: 1D array of values.
        min_dip: Minimum dip depth to be considered significant.

    Returns:
        Index of the deepest significant local minimum, or None.
    """
    best_idx = None
    best_dip = 0.0
    for i in range(1, len(values) - 1):
        if values[i] < values[i - 1] and values[i] < values[i + 1]:
            dip = min(values[i - 1], values[i + 1]) - values[i]
            if dip >= min_dip and dip > best_dip:
                best_dip = dip
                best_idx = i
    return best_idx


def _snap_to_original(point: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Snap a point to the nearest original mesh vertex.

    Args:
        point: Target position of shape (3,).
        vertices: Original mesh vertices of shape (N, 3).

    Returns:
        Nearest vertex position as numpy array of shape (3,).
    """
    tree = KDTree(vertices)
    _, idx = tree.query(point)
    return vertices[idx].copy()
