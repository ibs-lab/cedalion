"""Registrating optodes to scalp surfaces."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import KDTree

import cedalion
import cedalion.dataclasses as cdc
import cedalion.typing as cdt
import cedalion.xrutils as xrutils

from .utils import m_rot, m_scale1, m_scale3, m_trans


def _subtract(a: cdt.LabeledPointCloud, b: cdt.LabeledPointCloud):
    """Calculate difference vectors between points and check CRS."""

    crs_a = a.points.crs
    crs_b = b.points.crs

    if not crs_a == crs_b:
        raise ValueError("point clouds are using different coordinate systems.")

    return a - b


@cdc.validate_schemas
def register_trans_rot(
    coords_target: cdt.LabeledPointCloud,
    coords_trafo: cdt.LabeledPointCloud,
):
    common_labels = coords_target.points.common_labels(coords_trafo)

    if len(common_labels) < 3:
        raise ValueError("less than 3 common coordinates found")

    from_crs = coords_trafo.points.crs
    from_units = coords_trafo.pint.units
    to_crs = coords_target.points.crs
    to_units = coords_target.pint.units

    # allow scaling only to convert between units
    unit_scale_factor = ((1 * from_units) / (1 * to_units)).to_reduced_units()
    assert unit_scale_factor.units == cedalion.units.Unit("1")
    unit_scale_factor = float(unit_scale_factor)

    # restrict to commmon labels and dequantify
    coords_trafo = coords_trafo.sel(label=common_labels).pint.dequantify()
    coords_target = coords_target.sel(label=common_labels).pint.dequantify()

    # calculate difference between centers of gravity. Use this as initial
    # parameters for the translational component.
    delta_cog = (
        coords_target.mean("label").values
        - coords_trafo.mean("label").values * unit_scale_factor
    )

    def trafo(params):
        return m_rot(params[3:6]) @ m_trans(params[0:3]) @ m_scale1([unit_scale_factor])

    def loss(params, coords_target, coords_trafo):
        M = trafo(params)
        tmp = coords_trafo.points._apply_numpy_transform(M, to_crs)
        return np.power(_subtract(coords_target, tmp), 2).sum()

    bounds = [
        (None, None),
        (None, None),
        (None, None),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
        (-2 * np.pi, 2 * np.pi),
    ]

    result = minimize(
        loss,
        [delta_cog[0], delta_cog[1], delta_cog[2], 0.0, 0.0, 0.0],
        args=(coords_target, coords_trafo),
        bounds=bounds,
    )

    trafo_opt = trafo(result.x)
    trafo_opt = cdc.affine_transform_from_numpy(
        trafo_opt,
        from_crs=from_crs,
        to_crs=to_crs,
        from_units=from_units,
        to_units=to_units,
    )

    return trafo_opt


def _std_distance_to_cog(points: cdt.LabeledPointCloud):
    dists = xrutils.norm(points - points.mean("label"), points.points.crs)
    return dists.std("label").item()


@cdc.validate_schemas
def register_trans_rot_isoscale(
    coords_target: cdt.LabeledPointCloud,
    coords_trafo: cdt.LabeledPointCloud,
):
    common_labels = coords_target.points.common_labels(coords_trafo)

    if len(common_labels) < 3:
        raise ValueError("less than 3 common coordinates found")

    from_crs = coords_trafo.points.crs
    from_units = coords_trafo.pint.units
    to_crs = coords_target.points.crs
    to_units = coords_target.pint.units

    # restrict to commmon labels and dequantify
    coords_trafo = coords_trafo.sel(label=common_labels).pint.dequantify()
    coords_target = coords_target.sel(label=common_labels).pint.dequantify()

    std_trafo = _std_distance_to_cog(coords_trafo)
    std_target = _std_distance_to_cog(coords_target)

    scale0 = std_target / std_trafo

    # calculate difference between centers of gravity. Use this as initial
    # parameters for the translational component.
    delta_cog = (
        coords_target.mean("label").values - coords_trafo.mean("label").values * scale0
    )

    def trafo(params):
        return m_rot(params[3:6]) @ m_trans(params[0:3]) @ m_scale1([params[6]])

    def loss(params, coords_target, coords_trafo):
        M = trafo(params)
        tmp = coords_trafo.points._apply_numpy_transform(M, to_crs)
        return np.power(_subtract(coords_target, tmp), 2).sum()

    result = minimize(
        loss,
        [delta_cog[0], delta_cog[1], delta_cog[2], 0.0, 0.0, 0.0, scale0],
        args=(coords_target, coords_trafo),
    )

    trafo_opt = trafo(result.x)
    trafo_opt = cdc.affine_transform_from_numpy(
        trafo_opt,
        from_crs=from_crs,
        to_crs=to_crs,
        from_units=from_units,
        to_units=to_units,
    )

    return trafo_opt


@cdc.validate_schemas
def register_icp(
    surface: cdc.Surface,
    landmarks: cdt.LabeledPointCloud,
    geo3d: cdt.LabeledPointCloud,
    niterations=1000,
    random_sample_fraction=0.5,
):
    units = "mm"
    landmarks_mm = landmarks.pint.to(units).points.to_homogeneous().pint.dequantify()
    geo3d_mm = geo3d.pint.to(units).points.to_homogeneous().pint.dequantify()
    vertices_mm = (
        surface.vertices.pint.to(units).points.to_homogeneous().pint.dequantify()
    )

    to_crs = landmarks.points.crs

    kdtree_vertices = KDTree(vertices_mm.values[:, :3])

    common_labels = landmarks_mm.points.common_labels(geo3d_mm)

    def trafo(params):
        return m_trans(params[0:3]) @ m_rot(params[6:9]) @ m_scale3(params[3:6])

    def loss(params, coords_to, coords_from):
        M = trafo(params)
        tmp = coords_from.points._apply_numpy_transform(M, to_crs)
        return np.power(_subtract(coords_to, tmp), 2).sum()

    params0 = np.zeros(9)
    current_params = params0.copy()
    best_loss = np.inf

    losses = []
    trafos = []

    geo3d_indices = np.asarray(
        [i for i in np.arange(len(geo3d_mm)) if geo3d_mm.label[i] not in common_labels]
    )
    sample_size = int(random_sample_fraction * len(geo3d_mm))

    # lm_target = landmarks_mm.loc[common_labels].values

    for i_iter in range(niterations):
        indices = np.random.choice(geo3d_indices, sample_size)
        coords_vertex_candidates = geo3d_mm[indices].values

        _, idx_closest_vertices_on_mesh = kdtree_vertices.query(
            coords_vertex_candidates[:, :3]
        )
        coords_closest_vertices_on_mesh = vertices_mm[
            idx_closest_vertices_on_mesh
        ].values

        coords_true = np.vstack(
            (landmarks_mm.loc[common_labels].values, coords_closest_vertices_on_mesh)
        )
        coords = np.vstack(
            (geo3d_mm.loc[common_labels].values, coords_vertex_candidates)
        )

        no_bounds = (None, None)
        bounds = [
            no_bounds,
            no_bounds,
            no_bounds,
            (0.5, 1.5),
            (0.5, 1.5),
            (0.5, 1.5),
            (-2 * np.pi, 2 * np.pi),
            (-2 * np.pi, 2 * np.pi),
            (-2 * np.pi, 2 * np.pi),
        ]

        result = minimize(
            loss, current_params, args=(coords_true, coords), bounds=bounds
        )
        if result.fun < best_loss:
            best_loss = result.fun
            losses.append(result.fun)
            current_params = result.x
            trafos.append(trafo(current_params))

        if i_iter % 50 == 0:
            print(i_iter, result.fun, result.success)

    # idx_best = np.argmin(losses)
    return losses, trafos
