"""Top-level orchestrator: ``anonymize_scan``.

Single entry point that chains the helper functions from ``preprocessing``
and ``mask``. The individual functions remain public so downstream code can
re-run an intermediate step or inspect the head-isolation / deletion masks,
but the canonical pipeline lives here and only here.
"""

from typing import Literal

import numpy as np

import cedalion.dataclasses as cdc
import cedalion.typing as cdt
from cedalion.errors import CRSMismatchError
from cedalion.geometry.landmarks import normalize_landmarks_labels

from ._utils import _REQUIRED_LABELS
from .mask import (
    CapDetectionParams,
    delete_masked_vertices,
    detect_cap_boundary,
    face_mask_from_landmarks,
)
from .preprocessing import (
    align_to_ctf,
    isolate_head,
    orient_y_anterior,
    revert_to_einstar_frame,
)


@cdc.validate_schemas
def anonymize_scan(
    surface: cdc.TrimeshSurface,
    landmarks: cdt.LabeledPoints,
    *,
    head_isolation_radius_mm: float = 220.0,
    ear_delete_radius_mm: float = 40.0,
    landmark_keep_radius_mm: float = 8.0,
    cap: CapDetectionParams = CapDetectionParams(),
    min_remaining_frac: float = 0.20,
    return_frame: Literal["digitized", "ctf"] = "digitized",
) -> tuple[cdc.TrimeshSurface, cdt.LabeledPoints]:
    """Run the full face-anonymization pipeline on a raw Einstar scan.

    Steps (each is also exposed as a standalone function for inspection):

    1. ``orient_y_anterior``: rotate around X so Y points anterior.
    2. ``isolate_head``: strip body, shoulders, fragments.
    3. ``align_to_ctf``: map into the CTF coordinate system.
    4. ``detect_cap_boundary``: find the cap front edge along Z.
    5. ``face_mask_from_landmarks``: union face region + ear spheres
       (clamped below the cap), then carve out preservation spheres around
       each landmark and a midline nasion strip up to the cap.
    6. ``delete_masked_vertices``: drop triangles touching any masked
       vertex, keeping UVs in sync.
    7. (default) ``revert_to_einstar_frame``: return to ``crs="digitized"``
       so the output matches ``read_einstar_obj``'s convention and can be
       fed to ``save_anonymized_scan`` and downstream co-registration.

    Args:
        surface: TrimeshSurface with a defined CRS (typically ``"digitized"``
            from :func:`cedalion.io.read_einstar_obj`).
        landmarks: LabeledPoints with Nz, Iz, Cz, LPA, RPA (mixed-case
            aliases like Lpa/Rpa accepted; normalized via
            ``normalize_landmarks_labels``). Must share ``surface.crs``;
            the CRS string itself is arbitrary as long as both agree.
        head_isolation_radius_mm: Sphere radius around the upper-head
            centroid for ``isolate_head``.
        ear_delete_radius_mm: sphere radius around LPA/RPA for the ear
            region of the deletion mask.
        landmark_keep_radius_mm: per-landmark preservation sphere radius
            and half-width of the midline nasion strip.
        cap: cap-detection parameters; see :class:`CapDetectionParams`.
        min_remaining_frac: minimum fraction of vertices that must survive
            masking, otherwise a ``RuntimeError`` is raised. Catches
            pathological inputs that would silently produce a near-empty
            mesh. Set to ``0`` to disable the post-condition.
        return_frame: ``"digitized"`` (default) reverts back to the raw
            Einstar frame; ``"ctf"`` keeps the CTF coordinate system.

    Returns:
        Tuple of (anonymized_surface, anonymized_landmarks). CRS is
        controlled by ``return_frame``. The surface can be written with
        ``save_anonymized_scan`` when ``return_frame="digitized"``.
    """
    landmarks = normalize_landmarks_labels(landmarks)
    labels = list(landmarks["label"].values)
    missing = set(_REQUIRED_LABELS) - set(labels)
    if missing:
        raise ValueError(f"Missing landmarks for anonymization: {missing}")

    # The pipeline math is intrinsic to the point positions; the CRS string
    # itself doesn't matter, but surface and landmarks must agree on it so we
    # don't silently mix coordinate systems.
    landmarks_crs = next(d for d in landmarks.dims if d != "label")
    if surface.crs != landmarks_crs:
        raise CRSMismatchError.unexpected_crs(surface.crs, landmarks_crs)
    Nz_raw = landmarks.sel(label="Nz").pint.dequantify().values

    surface_n, _, R_norm = orient_y_anterior(surface, Nz_raw)
    # orient_y_anterior is a pre-rotation within the same CRS, not a CRS change,
    # so we pass the raw 4x4 to apply_transform (the AffineTransform wrapper
    # would produce a DataArray with duplicated CRS dim names).
    landmarks_n = landmarks.points.apply_transform(R_norm)

    Nz_n = landmarks_n.sel(label="Nz").pint.dequantify().values
    surface_n, _ = isolate_head(surface_n, Nz_n, radius=head_isolation_radius_mm)

    surface_h, landmarks_ctf, T_align = align_to_ctf(surface_n, landmarks_n)

    Nz, Iz, Cz, Lpa, Rpa = (landmarks_ctf.sel(label=lbl).pint.dequantify().values
                            for lbl in _REQUIRED_LABELS)

    verts = np.asarray(surface_h.mesh.vertices)
    cap_z = detect_cap_boundary(verts, Nz, Cz, Lpa, Rpa, params=cap)

    mask, _ = face_mask_from_landmarks(
        verts, Nz, Lpa, Rpa,
        Iz=Iz, Cz=Cz,
        cap_z=cap_z,
        ear_delete_radius=ear_delete_radius_mm,
        landmark_keep_radius=landmark_keep_radius_mm,
    )

    surface_anon = delete_masked_vertices(surface_h, mask)

    if min_remaining_frac > 0:
        remaining = surface_anon.nvertices / max(surface_h.nvertices, 1)
        if remaining < min_remaining_frac:
            raise RuntimeError(
                f"Anonymization mask removed {(1 - remaining) * 100:.1f}% of "
                f"vertices ({surface_h.nvertices} -> {surface_anon.nvertices}); "
                f"expected at least {min_remaining_frac * 100:.0f}% to survive. "
                f"Check landmark CRS and cap detection."
            )

    if return_frame == "ctf":
        return surface_anon, landmarks_ctf

    return revert_to_einstar_frame(
        surface_anon, landmarks_ctf, R_norm, T_align
    )
