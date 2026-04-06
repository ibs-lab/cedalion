"""Face anonymization module for photogrammetry scans.

This module provides tools to anonymize facial regions in 3D photogrammetry
scans while preserving optode positions and anatomical landmarks for fNIRS
research. The anonymization complies with GDPR requirements while maintaining
scientific utility.

The pipeline works as follows:

1. Detect nasion automatically (or via manual click)
2. Normalize axes so Y=anterior, Z=left
3. Isolate the head (remove body/shoulders)
4. Detect remaining landmarks (Iz, Cz, LPA, RPA) from nasion
5. Generate facial region mask (MediaPipe contour or hemisphere fallback)
6. Delete facial vertices

Example:
    >>> from cedalion.geometry.photogrammetry.anonymization import (
    ...     detect_nasion_auto, pick_nasion, normalize_axes,
    ...     isolate_head, detect_landmarks_from_nasion,
    ...     get_facial_region_mask_from_nasion,
    ... )
    >>> auto = detect_nasion_auto(surface)
    >>> nasion, meta = auto if auto else (pick_nasion(surface), {})
    >>> surface, nasion, R = normalize_axes(surface, nasion)
    >>> surface, _ = isolate_head(surface, nasion)
    >>> landmarks = detect_landmarks_from_nasion(surface, nasion)

Initial Contributors:
    - Face Anonymization Project | 2024
"""

from .face_detector import (
    normalize_axes,
    isolate_head,
    detect_landmarks_from_nasion,
    get_facial_region_mask_from_nasion,
)
from .nasion_detector import detect_nasion_auto
from .ui import pick_nasion


__all__ = [
    # Nasion detection
    "detect_nasion_auto",
    "pick_nasion",
    # Axis normalization and head isolation
    "normalize_axes",
    "isolate_head",
    # Landmark and face mask detection
    "detect_landmarks_from_nasion",
    "get_facial_region_mask_from_nasion",
]
