"""Unit tests for the face anonymization pipeline."""

import os

import numpy as np
import pytest
import trimesh
from numpy.testing import assert_allclose

import cedalion
import cedalion.dataclasses as cdc

from cedalion.geometry.photogrammetry.anonymization import (
    align_axes_from_landmarks,
    anonymize_scan,
    delete_masked_vertices,
    detect_cap_boundary,
    face_mask_from_landmarks,
    isolate_head,
    normalize_axes,
    revert_to_einstar_frame,
    save_anonymized_scan,
)


@pytest.fixture
def simple_sphere_surface():
    """Unit sphere as a minimal TrimeshSurface for geometry-only tests."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=100)
    return cdc.TrimeshSurface(
        mesh=sphere, crs="scanner", units=cedalion.units.millimeter
    )


@pytest.fixture
def head_like_surface():
    """Elongated sphere; X=up, Y=anterior, Z=left (axis-normalized frame)."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=100)
    vertices = sphere.vertices.copy()
    vertices[:, 0] *= 1.2
    vertices[:, 1] *= 0.9
    head_mesh = trimesh.Trimesh(vertices=vertices, faces=sphere.faces)
    return cdc.TrimeshSurface(
        mesh=head_mesh, crs="scanner", units=cedalion.units.millimeter
    )


@pytest.fixture
def axis_normalized_landmarks():
    """5 anatomical landmarks in the post-normalize_axes frame (X=up, Y=ant, Z=left)."""
    coords = np.array([
        [0, 100, 0],     # Nz (anterior)
        [0, -100, 0],    # Iz (posterior)
        [100, 0, 0],     # Cz (top, +X)
        [0, 0, 100],     # LPA (left, +Z)
        [0, 0, -100],    # RPA (right, -Z)
    ], dtype=float)
    return cdc.build_labeled_points(
        coords,
        crs="scanner",
        units="mm",
        labels=["Nz", "Iz", "Cz", "LPA", "RPA"],
        types=[cdc.PointType.LANDMARK] * 5,
    )


def test_normalize_rotation_orthogonal(simple_sphere_surface):
    """Rotation matrix returned by normalize_axes is orthogonal."""
    nasion = np.array([0, 50, 50])
    _, _, R = normalize_axes(simple_sphere_surface, nasion)
    assert_allclose(R @ R.T, np.eye(3), atol=1e-10)


def test_normalize_nasion_to_positive_y(simple_sphere_surface):
    """After normalization the rotated nasion has a positive Y component."""
    nasion = np.array([0, 50, 50])
    _, rotated_nasion, _ = normalize_axes(simple_sphere_surface, nasion)
    assert rotated_nasion[1] > 0


def test_normalize_identity_when_aligned(simple_sphere_surface):
    """No rotation is applied when the nasion already points along +Y."""
    nasion = np.array([0, 100, 0])
    _, rotated_nasion, R = normalize_axes(simple_sphere_surface, nasion)
    assert_allclose(R, np.eye(3), atol=1e-6)
    assert_allclose(rotated_nasion, nasion, atol=1e-6)


def test_isolate_head_only_scan_unchanged(simple_sphere_surface):
    """A head-only scan (no body) is returned with almost all vertices kept."""
    nasion = np.array([0, 100, 0])
    _, mask = isolate_head(simple_sphere_surface, nasion)
    assert mask.mean() > 0.9


def test_isolate_reduces_count_with_body():
    """Body sphere 400 mm away is removed; only the head sphere survives."""
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=100)
    body = trimesh.creation.icosphere(subdivisions=3, radius=80)
    body.vertices[:, 0] -= 400
    combined = trimesh.util.concatenate([sphere, body])

    surface = cdc.TrimeshSurface(
        combined, crs="scanner", units=cedalion.units.millimeter
    )
    nasion = np.array([0, 100, 0])
    head_surface, _ = isolate_head(surface, nasion)
    assert head_surface.nvertices < surface.nvertices


def test_align_origin_at_ear_midpoint(simple_sphere_surface, axis_normalized_landmarks):
    """CTF origin is placed at the midpoint of LPA and RPA."""
    _, aligned_lm, _ = align_axes_from_landmarks(
        simple_sphere_surface, axis_normalized_landmarks
    )
    lm = aligned_lm.pint.dequantify()
    lpa = lm.sel(label="LPA").values
    rpa = lm.sel(label="RPA").values
    assert_allclose(0.5 * (lpa + rpa), [0, 0, 0], atol=1e-6)


def test_align_axes_orientation(simple_sphere_surface, axis_normalized_landmarks):
    """Nz points along +X, LPA along +Y, and Cz along +Z in the CTF frame."""
    _, aligned_lm, _ = align_axes_from_landmarks(
        simple_sphere_surface, axis_normalized_landmarks
    )
    lm = aligned_lm.pint.dequantify()
    nz = lm.sel(label="Nz").values
    lpa = lm.sel(label="LPA").values
    cz = lm.sel(label="Cz").values
    assert nz[0] > abs(nz[1]) and nz[0] > abs(nz[2])
    assert lpa[1] > abs(lpa[0]) and lpa[1] > abs(lpa[2])
    assert cz[2] > abs(cz[0]) and cz[2] > abs(cz[1])


def test_align_returns_ctf_crs(simple_sphere_surface, axis_normalized_landmarks):
    """Aligned surface carries crs='ctf'."""
    aligned_surface, _, _ = align_axes_from_landmarks(
        simple_sphere_surface, axis_normalized_landmarks
    )
    assert aligned_surface.crs == "ctf"


def test_align_raises_on_missing_landmark(
    simple_sphere_surface, axis_normalized_landmarks
):
    """ValueError is raised when a required landmark is absent."""
    partial = axis_normalized_landmarks.isel(label=slice(0, 4))
    with pytest.raises(ValueError, match="Missing landmarks"):
        align_axes_from_landmarks(simple_sphere_surface, partial)


def test_cap_boundary_in_sane_range():
    """cap_z stays inside the documented [Nz[2], Nz[2] + ceiling] window."""
    sphere = trimesh.creation.icosphere(subdivisions=4, radius=100)
    verts = np.asarray(sphere.vertices)
    Nz = np.array([100, 0, 0])
    Cz = np.array([0, 0, 100])
    Lpa = np.array([0, 1, 0])
    Rpa = np.array([0, -1, 0])
    cap_z, *_ = detect_cap_boundary(verts, Nz, Cz, Lpa, Rpa)
    assert Nz[2] <= cap_z <= Nz[2] + 40.0


def test_cap_boundary_failsafe_clamps_implausible_cap():
    """Failsafe clamps cap_z to Nz[2] + eyebrow_offset_mm when detection runs hot.

    Builds a midline X-max profile that rises steeply early then creeps gently
    up to Cz; the gradient walk-back catches the gentle phase at high z, so
    without the failsafe cap_z would land above the ceiling.
    """
    z = np.linspace(0.5, 99.5, 200)
    x = 50.0 + 50.0 * np.tanh(z / 8.0) + 0.05 * z
    y = np.zeros_like(z)
    anterior = np.column_stack([x, y, z])
    posterior = np.column_stack([-x, y, z])
    verts = np.vstack([anterior, posterior])
    Nz = np.array([100.0, 0.0, 0.0])
    Cz = np.array([0.0, 0.0, 100.0])
    Lpa = np.array([0.0, 1.0, 0.0])
    Rpa = np.array([0.0, -1.0, 0.0])
    cap_z, *_ = detect_cap_boundary(verts, Nz, Cz, Lpa, Rpa)
    assert cap_z == pytest.approx(Nz[2] + 10.0)


def test_face_mask_region_semantics():
    """Anterior-below-cap vertex is masked; posterior and above-cap are not."""
    Nz = np.array([100, 0, 10])
    Lpa = np.array([0, 100, 0])
    Rpa = np.array([0, -100, 0])
    verts = np.array([
        [50, 0, 0],     # anterior, below cap
        [-50, 0, 0],    # posterior
        [50, 0, 50],    # anterior, above cap
    ])
    mask, _ = face_mask_from_landmarks(verts, Nz, Lpa, Rpa, cap_z=20.0)
    assert mask.tolist() == [True, False, False]


def test_face_mask_ear_sphere():
    """Vertex near LPA is masked even when posterior to the ear coronal plane."""
    Nz = np.array([100, 0, 0])
    Lpa = np.array([0, 100, 0])
    Rpa = np.array([0, -100, 0])
    # 10mm posterior to ear midpoint, but 20mm from LPA (within ear_delete_radius=40)
    verts = np.array([[-10, 80, 0]])
    mask, _ = face_mask_from_landmarks(
        verts, Nz, Lpa, Rpa, cap_z=10.0, ear_delete_radius=40.0
    )
    assert mask[0]


def test_delete_no_op_on_false_mask(simple_sphere_surface):
    """All-False mask leaves the vertex count unchanged."""
    mask = np.zeros(simple_sphere_surface.nvertices, dtype=bool)
    result = delete_masked_vertices(simple_sphere_surface, mask)
    assert result.nvertices == simple_sphere_surface.nvertices


def test_delete_partial_mask_reduces_count(simple_sphere_surface):
    """Masking one hemisphere reduces the vertex count."""
    mask = np.asarray(simple_sphere_surface.mesh.vertices)[:, 0] > 0
    result = delete_masked_vertices(simple_sphere_surface, mask)
    assert result.nvertices < simple_sphere_surface.nvertices


def test_delete_preserves_crs_and_units(simple_sphere_surface):
    """CRS and units are propagated unchanged after vertex deletion."""
    mask = np.zeros(simple_sphere_surface.nvertices, dtype=bool)
    mask[0] = True
    result = delete_masked_vertices(simple_sphere_surface, mask)
    assert result.crs == simple_sphere_surface.crs
    assert result.units == simple_sphere_surface.units


def test_revert_round_trip_with_align(simple_sphere_surface, axis_normalized_landmarks):
    """align then revert recovers the original vertex positions."""
    aligned_surface, aligned_lm, M = align_axes_from_landmarks(
        simple_sphere_surface, axis_normalized_landmarks
    )
    reverted_surface, _ = revert_to_einstar_frame(
        aligned_surface, aligned_lm, R_normalize=np.eye(3), M_align=M
    )
    assert_allclose(
        np.asarray(reverted_surface.mesh.vertices),
        np.asarray(simple_sphere_surface.mesh.vertices),
        atol=1e-6,
    )


def test_revert_returns_digitized_crs(
    simple_sphere_surface, axis_normalized_landmarks
):
    """Reverted surface carries crs='digitized'."""
    aligned_surface, aligned_lm, M = align_axes_from_landmarks(
        simple_sphere_surface, axis_normalized_landmarks
    )
    reverted_surface, _ = revert_to_einstar_frame(
        aligned_surface, aligned_lm, np.eye(3), M
    )
    assert reverted_surface.crs == "digitized"


def test_save_raises_on_bad_extension(simple_sphere_surface, tmp_path):
    """ValueError is raised when out_path does not end in .obj."""
    out = str(tmp_path / "out.txt")
    with pytest.raises(ValueError, match=".obj"):
        save_anonymized_scan(simple_sphere_surface, out)


def test_save_geometry_only(simple_sphere_surface, tmp_path):
    """strip_texture=True writes an OBJ file without MTL or JPG."""
    out = str(tmp_path / "anon.obj")
    written = save_anonymized_scan(simple_sphere_surface, out, strip_texture=True)
    assert os.path.exists(out)
    assert any(p.endswith(".obj") for p in written)


def test_anonymize_scan_reduces_vertices(head_like_surface, axis_normalized_landmarks):
    """Anonymized surface has fewer vertices than the input (face region deleted)."""
    surface_anon, _ = anonymize_scan(head_like_surface, axis_normalized_landmarks)
    assert surface_anon.nvertices < head_like_surface.nvertices


def test_anonymize_scan_returns_digitized_frame(
    head_like_surface, axis_normalized_landmarks
):
    """Default return frame is 'digitized' for both surface and landmarks."""
    surface_anon, landmarks_anon = anonymize_scan(
        head_like_surface, axis_normalized_landmarks
    )
    assert surface_anon.crs == "digitized"
    assert "digitized" in landmarks_anon.dims


def test_anonymize_scan_return_frame_ctf(head_like_surface, axis_normalized_landmarks):
    """return_frame='ctf' keeps the surface in the CTF coordinate frame."""
    surface_anon, landmarks_anon = anonymize_scan(
        head_like_surface, axis_normalized_landmarks, return_frame="ctf"
    )
    assert surface_anon.crs == "ctf"


def test_anonymize_scan_raises_on_missing_landmark(
    head_like_surface, axis_normalized_landmarks
):
    """ValueError is raised when fewer than 5 required landmarks are provided."""
    partial = axis_normalized_landmarks.isel(label=slice(0, 3))
    with pytest.raises(ValueError, match="Missing landmarks"):
        anonymize_scan(head_like_surface, partial)


def test_full_anonymization_pipeline(
    head_like_surface, axis_normalized_landmarks, tmp_path
):
    """End-to-end pipeline: normalize, isolate, align, mask, revert, save."""
    nasion = axis_normalized_landmarks.pint.dequantify().sel(label="Nz").values
    surface_n, nasion_n, R = normalize_axes(head_like_surface, nasion)
    lm_arr = axis_normalized_landmarks.pint.dequantify().values
    landmarks_n = (
        axis_normalized_landmarks.pint.dequantify()
        .copy(data=lm_arr @ R.T)
        .pint.quantify()
    )
    surface_n, _ = isolate_head(surface_n, nasion_n)

    surface_h, landmarks_n, M_ctf = align_axes_from_landmarks(surface_n, landmarks_n)
    verts = np.asarray(surface_h.mesh.vertices)
    lm_n = landmarks_n.pint.dequantify()
    Nz = lm_n.sel(label="Nz").values
    Cz = lm_n.sel(label="Cz").values
    Lpa = lm_n.sel(label="LPA").values
    Rpa = lm_n.sel(label="RPA").values

    cap_z, *_ = detect_cap_boundary(verts, Nz, Cz, Lpa, Rpa)
    mask, _ = face_mask_from_landmarks(verts, Nz, Lpa, Rpa, cap_z=cap_z)
    surface_anon = delete_masked_vertices(surface_h, mask)
    surface_anon_dig, _ = revert_to_einstar_frame(
        surface_anon, landmarks_n, R, M_ctf
    )

    out = str(tmp_path / "anon.obj")
    save_anonymized_scan(surface_anon_dig, out, strip_texture=True)

    assert surface_anon.nvertices < surface_h.nvertices
    assert surface_anon_dig.crs == "digitized"
    assert os.path.exists(out)
