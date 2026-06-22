"""Face anonymization module for photogrammetry scans.

Anonymizes facial regions in 3D photogrammetry scans while preserving
optode positions and anatomical landmarks for fNIRS research.

The canonical entry point is :func:`anonymize_scan`. Pass it a raw
Einstar surface and the 5 landmarks (Nz, Iz, Cz, LPA, RPA) and it returns
the anonymized surface plus landmarks, ready for
:func:`save_anonymized_scan`.

Example:
    >>> from cedalion.geometry.photogrammetry.anonymization import (
    ...     anonymize_scan, save_anonymized_scan,
    ... )
    >>> surface_anon, landmarks_anon = anonymize_scan(surface, landmarks)
    >>> save_anonymized_scan(surface_anon, "out.obj")

Pipeline steps inside ``anonymize_scan``:

1. :func:`orient_y_anterior`: rotate around X so Y points anterior.
2. :func:`isolate_head`: strip body, shoulders, and disconnected fragments.
3. :func:`align_to_ctf`: map into the CTF frame.
4. :func:`detect_cap_boundary`: find the cap front edge along Z.
5. :func:`face_mask_from_landmarks`: face region + ear spheres clamped
   below the cap, with per-landmark preservation spheres and a midline
   nasion strip carved out.
6. :func:`delete_masked_vertices`: drop triangles touching any masked vertex.
7. :func:`revert_to_einstar_frame`: return to ``crs="digitized"`` so the
   output matches :func:`cedalion.io.read_einstar_obj`.

Each of those functions is also exported so callers that need to inspect
or override an intermediate step can do so without re-implementing the
whole pipeline.
"""

from .preprocessing import (
    orient_y_anterior,
    isolate_head,
    align_to_ctf,
    revert_to_einstar_frame,
)
from .mask import (
    CapDetectionParams,
    CapProfile,
    detect_cap_boundary,
    face_mask_from_landmarks,
    delete_masked_vertices,
    save_anonymized_scan,
)
from .pipeline import anonymize_scan


__all__ = [
    # Top-level orchestrator (canonical entry point)
    "anonymize_scan",
    # Preprocessing (Y-anterior orientation, head isolation, full alignment,
    # and the inverse mapping back to the raw Einstar frame)
    "orient_y_anterior",
    "isolate_head",
    "align_to_ctf",
    "revert_to_einstar_frame",
    # Mask construction and application
    "CapDetectionParams",
    "CapProfile",
    "detect_cap_boundary",
    "face_mask_from_landmarks",
    "delete_masked_vertices",
    "save_anonymized_scan",
]
