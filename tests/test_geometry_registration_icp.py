import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

import cedalion
import cedalion.dataclasses as cdc
import cedalion.geometry.registration as reg


def _sphere_scalp(radius=90.0, subdivisions=4):
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    return cdc.TrimeshSurface(sphere, crs="ras", units=cedalion.units.mm)


def _labeled(points, labels, types, crs):
    return cdc.build_labeled_points(
        np.asarray(points, dtype=float), labels=labels, types=types, crs=crs, units="mm"
    )


def test_register_icp_runs_and_reduces_loss():
    """register_icp runs and improves the fit.

    Regression test: the loss function called an xarray accessor on numpy arrays
    and raised an AttributeError on the first iteration.
    """
    rng = np.random.default_rng(0)
    radius = 90.0
    scalp = _sphere_scalp(radius)

    lm_labels = ["Nz", "Iz", "LPA", "RPA"]
    lm = np.array([[0, radius, 0], [0, -radius, 0], [-radius, 0, 0], [radius, 0, 0]])
    u = rng.normal(size=(20, 3))
    u[:, 2] = np.abs(u[:, 2])
    optodes = radius * u / np.linalg.norm(u, axis=1, keepdims=True)

    R = Rotation.from_euler("z", 10, degrees=True).as_matrix()
    t = np.array([3.0, -3.0, 5.0])
    probe = np.vstack([lm, optodes]) @ R.T + t

    labels = lm_labels + [f"S{i}" for i in range(20)]
    types = [cdc.PointType.LANDMARK] * 4 + [cdc.PointType.SOURCE] * 20
    geo3d = _labeled(probe, labels, types, "pos")
    landmarks = _labeled(lm, lm_labels, [cdc.PointType.LANDMARK] * 4, "ras")

    np.random.seed(0)
    losses, trafos = reg.register_icp(
        scalp, landmarks, geo3d, niterations=20, random_sample_fraction=0.5
    )

    # losses/trafos are appended whenever the fit improves
    assert len(losses) >= 2
    assert losses[-1] < losses[0]
    M = trafos[-1]
    mapped = probe @ M[:3, :3].T + M[:3, 3]
    rms_before = np.sqrt(((probe[4:] - optodes) ** 2).sum(1).mean())
    rms_after = np.sqrt(((mapped[4:] - optodes) ** 2).sum(1).mean())
    assert rms_after < rms_before


def test_spring_icp_converges_on_optode_displacement():
    """Convergence is declared only when the optodes stop moving.

    Regression test: the criterion compared the (second order) surface-projection
    distance with the tolerance, so the relaxation stopped after two iterations
    with the springs far from their rest lengths.
    """
    radius = 90.0
    scalp = _sphere_scalp(radius, subdivisions=5)

    lm = {
        "Nz": [0, radius, 0],
        "Iz": [0, -radius, 0],
        "LPA": [-radius, 0, 0],
        "RPA": [radius, 0, 0],
        "Cz": [0, 0, radius],
    }
    landmarks = _labeled(
        list(lm.values()), list(lm), [cdc.PointType.LANDMARK] * 5, "ras"
    )

    # 3x3 grid of optodes around the vertex
    opt_labels, opt_pos, opt_types = [], [], []
    k = 0
    for i, th in enumerate(np.deg2rad([-20, 0, 20])):
        for j, ph in enumerate(np.deg2rad([-20, 0, 20])):
            p = np.array([np.sin(th), np.sin(ph), 0.0])
            p[2] = np.sqrt(1 - p[0] ** 2 - p[1] ** 2)
            if (i + j) % 2 == 0:
                opt_labels.append(f"S{k}")
                opt_types.append(cdc.PointType.SOURCE)
            else:
                opt_labels.append(f"D{k}")
                opt_types.append(cdc.PointType.DETECTOR)
            opt_pos.append(radius * p)
            k += 1

    geo3d = _labeled(
        np.vstack([np.array(opt_pos), np.array(list(lm.values()), float)]),
        opt_labels + list(lm),
        opt_types + [cdc.PointType.LANDMARK] * 5,
        "pos",
    )
    positions = {lab: np.array(p) for lab, p in zip(opt_labels, opt_pos)}
    channels = [
        (s, d)
        for s in opt_labels
        if s.startswith("S")
        for d in opt_labels
        if d.startswith("D") and np.linalg.norm(positions[s] - positions[d]) < 40
    ]
    # nominal distances 20 % longer than the actual ones
    nominal = {
        (s, d): 1.2 * float(np.linalg.norm(positions[s] - positions[d]))
        for s, d in channels
    }

    _, result = reg.register_optodes_spring_icp(
        scalp,
        geo3d,
        channels,
        landmarks,
        nominal_distances=nominal,
        n_iter=400,
        k_spring=1.0,
        k_anchor=10.0,
        step_size=0.1,
        convergence_tol=0.01,
        initial_align_mode="trans_rot",
    )

    assert result.n_iterations > 5
    assert np.abs(result.spring_errors.values).mean() < 1.0
