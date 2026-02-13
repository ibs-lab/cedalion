"""Tests for landmark normalization and visualization functions."""

import numpy as np
import pytest
import xarray as xr

import cedalion.dataclasses as cdc
from cedalion.geometry.landmarks import normalize_landmarks_labels
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
