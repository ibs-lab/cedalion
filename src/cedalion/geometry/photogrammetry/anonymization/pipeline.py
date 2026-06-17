"""Top-level orchestrator: ``anonymize_scan``.

Single entry point that chains the helper functions from ``preprocessing``
and ``mask``. The individual functions remain public so downstream code can
inspect intermediate state (e.g. the cap-detection profile, the head
isolation mask), but the canonical pipeline lives here and only here.
"""

from typing import Literal

import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion.geometry.landmarks import normalize_landmarks_labels

from ._utils import _ear_midpoint, _apply_affine, _transform_labeled_points
from .mask import (
    delete_masked_vertices,
    detect_cap_boundary,
    face_mask_from_landmarks,
)
from .preprocessing import (
    align_axes_from_landmarks,
    isolate_head,
    normalize_axes,
    revert_to_einstar_frame,
)


_REQUIRED_LABELS = ("Nz", "Iz", "Cz", "LPA", "RPA")


@cdc.validate_schemas
def anonymize_scan(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPoints,
    *,
    head_isolation_radius_mm: float = 220.0,
    cap_band_width_mm: float = 15.0,
    cap_bin_size_mm: float = 1.0,
    cap_foot_grad_threshold: float = 0.2,
    cap_z_ceiling_mm: float = 40.0,
    eyebrow_offset_mm: float = 10.0,
    ear_delete_radius_mm: float = 40.0,
    landmark_keep_radius_mm: float = 8.0,
    return_frame: Literal["digitized", "ctf"] = "digitized",
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPoints]:
    """Run the full face-anonymization pipeline on a raw Einstar scan.

    Steps (each is also exposed as a standalone function for inspection):

    1. ``normalize_axes``: rotate around X so Y points anterior.
    2. ``isolate_head``: strip body, shoulders, fragments.
    3. ``align_axes_from_landmarks``: map into the CTF frame.
    4. ``detect_cap_boundary``: find the cap front edge along Z.
    5. ``face_mask_from_landmarks``: union face region + ear spheres,
       clamped below the cap.
    6. Preserve ``landmark_keep_radius_mm``-spheres around each landmark
       and a midline nasion strip from Nz to the cap.
    7. ``delete_masked_vertices``: drop triangles touching any masked
       vertex, keeping UVs in sync.
    8. (default) ``revert_to_einstar_frame``: return to ``crs="digitized"``
       so the output matches ``read_einstar_obj``'s convention and can be
       fed to ``save_anonymized_scan`` and downstream co-registration.

    Args:
        surface: Raw Einstar TrimeshSurface (``crs="digitized"``).
        landmarks: LabeledPoints with Nz, Iz, Cz, LPA, RPA (mixed-case
            aliases like Lpa/Rpa accepted; normalized via
            ``normalize_landmarks_labels``).
        head_isolation_radius_mm: Sphere radius around the upper-head
            centroid for ``isolate_head``.
        cap_band_width_mm: Y-band half-width for the cap X-profile.
        cap_bin_size_mm: Z-bin size for the cap X-profile.
        cap_foot_grad_threshold: dX/dZ below which the cap-foot is
            recognized.
        cap_z_ceiling_mm: mm above Nz at which cap detection is
            considered untrustworthy and the failsafe fires.
        eyebrow_offset_mm: failsafe cap height above Nz.
        ear_delete_radius_mm: sphere radius around LPA/RPA for the ear
            region of the deletion mask.
        landmark_keep_radius_mm: per-landmark preservation sphere radius
            and half-width of the midline nasion strip.
        return_frame: ``"digitized"`` (default) reverts back to the raw
            Einstar frame; ``"ctf"`` keeps the CTF frame.

    Returns:
        Tuple of (anonymized_surface, anonymized_landmarks). Frame is
        controlled by ``return_frame``. The surface can be written with
        ``save_anonymized_scan`` when ``return_frame="digitized"``.
    """
    landmarks = normalize_landmarks_labels(landmarks)
    labels = list(landmarks["label"].values)
    missing = set(_REQUIRED_LABELS) - set(labels)
    if missing:
        raise ValueError(f"Missing landmarks for anonymization: {missing}")
    idx = {lbl: i for i, lbl in enumerate(labels)}
    Nz_raw = landmarks.pint.dequantify().values[idx["Nz"]]

    surface_n, _, R_norm = normalize_axes(surface, Nz_raw)
    R_norm4 = np.eye(4)
    R_norm4[:3, :3] = R_norm
    crs_dim = next(d for d in landmarks.dims if d != "label")
    landmarks_n = _transform_labeled_points(
        landmarks,
        lambda p: _apply_affine(p, R_norm4),
        new_crs=crs_dim,
    )

    Nz_n = landmarks_n.pint.dequantify().values[idx["Nz"]]
    surface_n, _ = isolate_head(
        surface_n, Nz_n, radius=head_isolation_radius_mm
    )

    surface_h, landmarks_ctf, M_align = align_axes_from_landmarks(
        surface_n, landmarks_n
    )
    lm_ctf = landmarks_ctf.pint.dequantify().values
    Nz, Iz, Cz, Lpa, Rpa = (
        lm_ctf[idx[lbl]] for lbl in _REQUIRED_LABELS
    )
    ear_mid = _ear_midpoint(Lpa, Rpa)

    verts = np.asarray(surface_h.mesh.vertices)
    cap_z, *_ = detect_cap_boundary(
        verts, Nz, Cz, Lpa, Rpa,
        band_width=cap_band_width_mm,
        bin_size=cap_bin_size_mm,
        foot_grad_threshold=cap_foot_grad_threshold,
        cap_z_ceiling_mm=cap_z_ceiling_mm,
        eyebrow_offset_mm=eyebrow_offset_mm,
    )

    mask, _ = face_mask_from_landmarks(
        verts, Nz, Lpa, Rpa,
        cap_z=cap_z,
        ear_delete_radius=ear_delete_radius_mm,
    )

    for lm in (Nz, Iz, Cz, Lpa, Rpa):
        near = np.linalg.norm(verts - lm, axis=1) < landmark_keep_radius_mm
        mask[near] = False
    nasion_strip = (
        (verts[:, 2] >= Nz[2])
        & (verts[:, 2] < cap_z)
        & (np.abs(verts[:, 1] - Nz[1]) < landmark_keep_radius_mm)
        & (verts[:, 0] > ear_mid[0])
    )
    mask[nasion_strip] = False

    surface_anon = delete_masked_vertices(surface_h, mask)

    if return_frame == "ctf":
        return surface_anon, landmarks_ctf

    return revert_to_einstar_frame(
        surface_anon, landmarks_ctf, R_norm, M_align
    )
