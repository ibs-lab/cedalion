"""Tests for landmark normalization and visualization functions."""

import warnings

import numpy as np
import pytest
import trimesh
import xarray as xr

import cedalion.dataclasses as cdc
from cedalion.geometry.landmarks import (
    LandmarksBuilder1010,
    normalize_landmarks_labels,
)
from cedalion.vis.anatomy.montage import plot_montage3D


def test_normalize_duplicates_with_canonical():
    """Test duplicate handling: multiple alternatives + canonical present."""
    labels = ["NASION", "Nasion", "Nz", "INION", "Iz", "S1"]
    coords = np.random.rand(len(labels), 3) * 100
    types = [cdc.PointType.LANDMARK] * 5 + [cdc.PointType.SOURCE]

    geo3d = cdc.build_labeled_points(
        coords, crs="unknown", units="mm", labels=labels, types=types
    )

    # with pytest.raises(ValueError):
    result = normalize_landmarks_labels(geo3d)

    # canonical was present, other are not changed
    assert all(result.label.values == labels)


def test_normalize_duplicates_without_canonical():
    """Test duplicate handling: multiple alternatives + canonical _not_ present."""
    labels = ["NASION", "Nasion", "nz", "INION", "Iz", "S1"]
    coords = np.random.rand(len(labels), 3) * 100
    types = [cdc.PointType.LANDMARK] * 5 + [cdc.PointType.SOURCE]

    geo3d = cdc.build_labeled_points(
        coords, crs="unknown", units="mm", labels=labels, types=types
    )

    # canonical was not present, normalize would rename all Nz-variants to the same
    # name.
    with pytest.raises(ValueError):
        normalize_landmarks_labels(geo3d)


def test_normalize_preserve_unknown_and_canonical():
    """Test unknown labels and already-canonical labels are preserved."""
    labels = ["nz", "UnknownLandmark", "S1"]
    coords = np.random.rand(len(labels), 3) * 100
    types = [cdc.PointType.LANDMARK] * 2 + [cdc.PointType.SOURCE]

    geo3d = cdc.build_labeled_points(
        coords, crs="unknown", units="mm", labels=labels, types=types
    )

    result = normalize_landmarks_labels(geo3d)

    assert set(result.label.values) == {"Nz", "UnknownLandmark", "S1"}


def test_normalize_capitalization():
    labels = ["nZ", "iz", "lPa", "rPA", "cz"]

    coords = np.random.rand(len(labels), 3) * 100
    types = [cdc.PointType.LANDMARK] * 5

    geo3d = cdc.build_labeled_points(
        coords, crs="unknown", units="mm", labels=labels, types=types
    )

    result = normalize_landmarks_labels(geo3d)

    assert all(result.label.values == ["Nz", "Iz", "LPA", "RPA", "Cz"])


def test_normalize_empty_data():
    """Test handling of empty input."""
    geo3d = cdc.build_labeled_points(
        np.empty((0, 3)), crs="unknown", units="mm", labels=[], types=[]
    )

    result = normalize_landmarks_labels(geo3d)

    assert len(result.label) == 0


@pytest.fixture
def sample_geo3d():
    """Create sample geo3d with all canonical landmarks and optodes."""
    labels = ["Nz", "Iz", "LPA", "RPA", "Cz", "S1", "D1"]
    coords = np.array(
        [
            [0, 0, 100],  # Nz
            [0, 0, -50],  # Iz
            [-50, 0, 25],  # LPA
            [50, 0, 25],  # RPA
            [0, 50, 50],  # Cz
            [-30, 40, 80],  # S1
            [-20, 40, 80],  # D1
        ]
    )
    types = [
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.SOURCE,
        cdc.PointType.DETECTOR,
    ]

    return cdc.build_labeled_points(
        coords, crs="unknown", units="mm", labels=labels, types=types
    )


@pytest.fixture
def sample_amp():
    """Create sample amplitude data."""
    return xr.DataArray(
        np.random.rand(10, 1),
        dims=["time", "channel"],
        coords={
            "time": np.arange(10),
            "source": ("channel", ["S1"]),
            "detector": ("channel", ["D1"]),
        },
    )


def test_plot_landmarks_modes(sample_amp, sample_geo3d):
    """Test landmark modes: None (default), list, and empty list."""
    # None should show all canonical landmarks present (default)
    plot_montage3D(sample_amp, sample_geo3d, landmarks=None)
    # List should show specified landmarks
    plot_montage3D(sample_amp, sample_geo3d, landmarks=["Nz", "Iz"])
    # Empty list should show no landmarks
    plot_montage3D(sample_amp, sample_geo3d, landmarks=[])


def test_plot_default_shows_all_canonical_landmarks(sample_amp, sample_geo3d):
    """Test that default (None) shows all 5 canonical landmarks if present."""
    # sample_geo3d has all 5 canonical landmarks
    plot_montage3D(sample_amp, sample_geo3d)
    # Should not raise and would show Nz, Iz, LPA, RPA, Cz


def test_plot_default_with_partial_canonical_landmarks(sample_amp):
    """Test default shows only canonical landmarks that are present."""
    # Create geo3d with only some canonical landmarks
    labels = ["Nz", "LPA", "S1", "D1", "UnknownLandmark"]
    coords = np.array(
        [
            [0, 0, 100],  # Nz
            [-50, 0, 25],  # LPA
            [-30, 40, 80],  # S1
            [-20, 40, 80],  # D1
            [10, 10, 10],  # UnknownLandmark
        ]
    )
    types = [
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.SOURCE,
        cdc.PointType.DETECTOR,
        cdc.PointType.LANDMARK,
    ]

    geo3d = cdc.build_labeled_points(
        coords, crs="unknown", units="mm", labels=labels, types=types
    )

    # Should show only Nz and LPA (canonical landmarks present)
    # UnknownLandmark should not be shown
    plot_montage3D(sample_amp, geo3d)


def test_plot_empty_list_shows_no_landmarks(sample_amp, sample_geo3d):
    """Test that empty list explicitly shows no landmarks."""
    # Should not raise and show no landmarks
    plot_montage3D(sample_amp, sample_geo3d, landmarks=[])


def test_plot_custom_list_shows_only_specified(sample_amp, sample_geo3d):
    """Test custom list shows only specified landmarks."""
    # Should show only Nz and Cz, not the others
    plot_montage3D(sample_amp, sample_geo3d, landmarks=["Nz", "Cz"])


def test_plot_nonexistent_landmarks_filtered(sample_amp, sample_geo3d):
    """Test that non-existent landmarks are filtered out silently."""
    # Should show only Nz and RPA (NonExistent filtered out)
    plot_montage3D(sample_amp, sample_geo3d, landmarks=["Nz", "NonExistent", "RPA"])


def test_normalize_then_plot():
    """Test full workflow: normalize alternative names then plot."""
    labels = ["NASION", "INION", "lpa", "rpa", "CZ", "S1", "D1"]
    coords = np.array(
        [
            [0, 0, 100],  # NASION -> Nz
            [0, 0, -50],  # INION -> Iz
            [-50, 0, 25],  # lpa -> LPA
            [50, 0, 25],  # rpa -> RPA
            [0, 50, 50],  # CZ -> Cz
            [-30, 40, 80],  # S1
            [-20, 40, 80],  # D1
        ]
    )
    types = [
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.LANDMARK,
        cdc.PointType.SOURCE,
        cdc.PointType.DETECTOR,
    ]

    geo3d = cdc.build_labeled_points(
        coords, crs="unknown", units="mm", labels=labels, types=types
    )

    amp = xr.DataArray(
        np.random.rand(10, 1),
        dims=["time", "channel"],
        coords={
            "time": np.arange(10),
            "source": ("channel", ["S1"]),
            "detector": ("channel", ["D1"]),
        },
    )

    # Normalize labels
    normalized_geo3d = normalize_landmarks_labels(geo3d)

    # Verify normalization
    assert "Nz" in normalized_geo3d.label.values
    assert "Iz" in normalized_geo3d.label.values
    assert "LPA" in normalized_geo3d.label.values
    assert "RPA" in normalized_geo3d.label.values
    assert "Cz" in normalized_geo3d.label.values
    assert "NASION" not in normalized_geo3d.label.values

    # Plot with default (shows all canonical landmarks)
    plot_montage3D(amp, normalized_geo3d)

    # Plot with custom list
    plot_montage3D(amp, normalized_geo3d, landmarks=["Nz", "Cz"])

    # Plot with empty list
    plot_montage3D(amp, normalized_geo3d, landmarks=[])


def _icosphere_scalp():
    """Build a synthetic ellipsoid scalp + canonical Nz/Iz/LPA/RPA fiducials."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=90.0)
    # squash into a vaguely head-shaped ellipsoid (a, b, c)
    sphere.vertices = sphere.vertices * np.array([85.0, 100.0, 90.0]) / 90.0
    surf = cdc.TrimeshSurface(sphere, crs="ras", units="mm")

    landmarks = cdc.build_labeled_points(
        [
            [0.0, 100.0, 0.0],   # Nz  (anterior, +y)
            [0.0, -100.0, 0.0],  # Iz  (posterior, -y)
            [-85.0, 0.0, 0.0],   # LPA (left, -x)
            [85.0, 0.0, 0.0],    # RPA (right, +x)
        ],
        crs="ras",
        units="mm",
        labels=["Nz", "Iz", "LPA", "RPA"],
        types=[cdc.PointType.LANDMARK] * 4,
    )
    return surf, landmarks


def test_estimate_cranial_vertex_is_at_apex():
    """For a head-shaped ellipsoid Cz should sit near the +z apex."""
    surf, lms = _icosphere_scalp()
    builder = LandmarksBuilder1010(surf, lms)
    cz = builder._estimate_cranial_vertex_from_lines()
    # apex is (0, 0, 90); algorithm averages two close midpoints, should be near apex
    assert abs(cz[0]) < 5.0
    assert abs(cz[1]) < 5.0
    assert cz[2] > 85.0


def test_plot_no_canonical_landmarks_present(sample_amp):
    """Test default behavior when no canonical landmarks are present."""
    # Create geo3d with only non-canonical landmarks
    labels = ["S1", "D1", "CustomLandmark"]
    coords = np.array(
        [
            [-30, 40, 80],  # S1
            [-20, 40, 80],  # D1
            [10, 10, 10],  # CustomLandmark
        ]
    )
    types = [
        cdc.PointType.SOURCE,
        cdc.PointType.DETECTOR,
        cdc.PointType.LANDMARK,
    ]

    geo3d = cdc.build_labeled_points(
        coords, crs="unknown", units="mm", labels=labels, types=types
    )

    # Should not raise, just show no landmarks
    plot_montage3D(sample_amp, geo3d)


# Labels emitted by the seven _add_landmarks_along_line calls in
# LandmarksBuilder1010.build(), plus the five fiducials (Nz/Iz/LPA/RPA + Cz).
EXPECTED_1010_LABELS = {
    "Nz", "Iz", "LPA", "RPA", "Cz",
    # sagittal midline
    "Fpz", "AFz", "Fz", "FCz", "CPz", "Pz", "POz", "Oz",
    # coronal midline (T7/T8 added separately)
    "T7", "T8",
    # left ear arc
    "Fp1", "AF7", "F7", "FT7", "TP7", "P7", "PO7", "O1",
    # right ear arc
    "Fp2", "AF8", "F8", "FT8", "TP8", "P8", "PO8", "O2",
    # transverse rows
    "C5", "C3", "C1", "C2", "C4", "C6",
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "F5", "F3", "F1", "F2", "F4", "F6",
    "AF5", "AF3", "AF1", "AF2", "AF4", "AF6",
    "CP5", "CP3", "CP1", "CP2", "CP4", "CP6",
    "P5", "P3", "P1", "P2", "P4", "P6",
    "PO5", "PO3", "PO1", "PO2", "PO4", "PO6",
}


def test_build_constructs_full_1010_system():
    """End-to-end build() on a synthetic ellipsoid scalp.
    Exercises build() and _add_landmarks_along_line.
    """
    surf, lms = _icosphere_scalp()
    builder = LandmarksBuilder1010(surf, lms)

    with warnings.catch_warnings():
        # build() emits a "WIP: distance calculation around ears" UserWarning.
        warnings.simplefilter("ignore", UserWarning)
        result = builder.build()

    actual = set(result.label.values.tolist())
    assert EXPECTED_1010_LABELS.issubset(actual), (
        f"missing 10-10 labels: {EXPECTED_1010_LABELS - actual}"
    )
    assert len(result.label) == len(EXPECTED_1010_LABELS)
    assert str(result.pint.units) == "millimeter"

    # Geometric value-based assertions on the (85, 100, 90) ellipsoid.
    coords_mm = result.pint.to("mm").pint.dequantify()
    pts = {
        str(label): coords_mm.sel(label=label).values
        for label in coords_mm.label.values
    }

    # 1) Every landmark lies on the (85, 100, 90) ellipsoid surface.
    a, b, c = 85.0, 100.0, 90.0
    for label, p in pts.items():
        f = (p[0] / a) ** 2 + (p[1] / b) ** 2 + (p[2] / c) ** 2
        assert abs(f - 1.0) < 0.05, f"{label} off ellipsoid (f={f:.4f})"

    # 2) Sagittal-midline points have x ~ 0 (cut plane passes through y-axis).
    sagittal = [
        "Nz", "Iz", "Cz", "Fpz", "AFz", "Fz", "FCz", "CPz", "Pz", "POz", "Oz",
    ]
    for label in sagittal:
        assert abs(pts[label][0]) < 1.0, (
            f"{label} not on sagittal plane (x={pts[label][0]:.3f})"
        )

    # 3) Coronal-midline points have y ~ 0 (cut plane passes through x-axis).
    coronal = ["Cz", "T7", "T8", "LPA", "RPA"]
    for label in coronal:
        assert abs(pts[label][1]) < 1.0, (
            f"{label} not on coronal plane (y={pts[label][1]:.3f})"
        )

    # 4) Bilateral (x -> -x) symmetry: the ellipsoid and fiducials are
    # mirror-symmetric about the y-z plane. The icosphere mesh itself has
    # icosahedral (not mirror) symmetry, so allow a couple of mm.
    lr_pairs = [
        ("T7", "T8"),
        ("Fp1", "Fp2"), ("AF7", "AF8"), ("F7", "F8"), ("FT7", "FT8"),
        ("TP7", "TP8"), ("P7", "P8"), ("PO7", "PO8"), ("O1", "O2"),
        ("C1", "C2"), ("C3", "C4"), ("C5", "C6"),
        ("FC1", "FC2"), ("FC3", "FC4"), ("FC5", "FC6"),
        ("F1", "F2"), ("F3", "F4"), ("F5", "F6"),
        ("AF1", "AF2"), ("AF3", "AF4"), ("AF5", "AF6"),
        ("CP1", "CP2"), ("CP3", "CP4"), ("CP5", "CP6"),
        ("P1", "P2"), ("P3", "P4"), ("P5", "P6"),
        ("PO1", "PO2"), ("PO3", "PO4"), ("PO5", "PO6"),
    ]
    for left, right in lr_pairs:
        pl, pr = pts[left], pts[right]
        assert abs(pl[0] + pr[0]) < 2.0, f"{left}/{right} x not mirrored"
        assert abs(pl[1] - pr[1]) < 2.0, f"{left}/{right} y differs"
        assert abs(pl[2] - pr[2]) < 2.0, f"{left}/{right} z differs"

    # 5) ~10% spacing along the sagittal midline. 11 points -> 10 chord segments.
    midline_order = [
        "Nz", "Fpz", "AFz", "Fz", "FCz", "Cz", "CPz", "Pz", "POz", "Oz", "Iz",
    ]
    midline = np.array([pts[label] for label in midline_order])
    segs = np.linalg.norm(np.diff(midline, axis=0), axis=1)
    assert np.all(np.abs(segs / segs.mean() - 1.0) < 0.10), (
        f"sagittal midline segments deviate >10% from mean: {segs}"
    )
    # Ramanujan upper-half perimeter of the (y/100)^2 + (z/90)^2 = 1 ellipse.
    ae, be = 100.0, 90.0
    half_perim = (
        np.pi * (3 * (ae + be) - np.sqrt((3 * ae + be) * (ae + 3 * be))) / 2
    )
    assert abs(segs.sum() - half_perim) / half_perim < 0.05, (
        f"sagittal chord-sum {segs.sum():.2f} mm vs analytic "
        f"half-perimeter {half_perim:.2f} mm"
    )
