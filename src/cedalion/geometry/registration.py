"""Registrating optodes to scalp surfaces."""
import numpy as np
from scipy.optimize import minimize,linear_sum_assignment
from scipy.spatial import KDTree

from numpy.linalg import pinv

import xarray as xr

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
        options={'disp': False},
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


def gen_xform_from_pts(p1, p2):
    """Calculate the affine transformation matrix T that transforms p1 to p2.

    Parameters:
    p1 (numpy.ndarray): Source points (p x m) where p is the number of points and m is the number of dimensions.
    p2 (numpy.ndarray): Target points (p x m) where p is the number of points and m is the number of dimensions.

    Returns:
    numpy.ndarray: Affine transformation matrix T.
    """

    T = np.eye(p1.shape[1] + 1)
    p, m = p1.shape
    q, n = p2.shape

    if p != q:
        print("Number of points for p1 and p2 must be the same")
        return None

    if m != n:
        print("Number of dimensions for p1 and p2 must be the same")
        return None

    if p < n:
        print(
            f"Cannot solve transformation with fewer anchor points ({p}) than dimensions ({n})."
        )
        return None

    A = np.hstack((p1, np.ones((p, 1))))

    for ii in range(n):
        x = np.dot(pinv(A), p2[:, ii])
        T[ii, :] = x.T

    return T


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

        print(coords)

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



def icp_with_full_transform(opt_centers, montage_points, max_iterations=50, tolerance=500):
    """Perform Iterative Closest Point (ICP) algorithm with full transformation capabilities.

    Parameters:
        opt_centers (cdt.LabeledPointCloud): Source point cloud for alignment.
        montage_points (cdt.LabeledPointCloud): Target reference point cloud.
        max_iterations (int): Maximum number of iterations for convergence.
        tolerance (float): Tolerance for convergence check.

    Returns:
        np.ndarray: Transformed source points as a numpy array with their coordinates updated to reflect the best alignment.
        np.ndarray: Transformation parameters array consisting of [tx, ty, tz, rx, ry, rz, sx, sy, sz], where 't' stands for 
                    translation components, 'r' for rotation components (in radians), and 's' for scaling components.
        np.ndarray: Indices of the target points that correspond to each source point as per the nearest neighbor search.
    """

    # Convert to homogeneous coordinates, assuming .values and .pint.dequantify() yield np.ndarray
    units = "mm"
    opt_centers_mm = opt_centers.pint.to(units).points.to_homogeneous().pint.dequantify()
    montage_points_mm = montage_points.pint.to(units).points.to_homogeneous().pint.dequantify()

    # Initialize transformation parameters: [translation, rotation (radians), scaling]
    current_params = np.zeros(9)  # tx, ty, tz, rx, ry, rz, sx, sy, sz
    best_loss = np.inf
    transformation_matrix = np.eye(4)

    def complete_transformation(params):
        """Generate a complete transformation matrix from params."""

        translation_matrix = m_trans(params[:3])
        rotation_matrix = m_rot(params[3:6])
        scaling_matrix = m_scale3(params[6:9])
        return translation_matrix @ rotation_matrix @ scaling_matrix

    def apply_numpy_transform(obj_values, transform: np.ndarray, to_crs=None, obj_units=None):
        transformed_values = np.hstack((obj_values, np.ones((obj_values.shape[0], 1)))) @ transform.T
        transformed_values = transformed_values[:, :-1]
        return transformed_values

    for iteration in range(max_iterations):
        opt_centers_mm[:, :3] = apply_numpy_transform(opt_centers_mm[:, :3],transformation_matrix)

        # Use the Hungarian algorithm to find the optimal assignment
        cost_matrix = np.linalg.norm(opt_centers_mm[:, :3].values[:, np.newaxis] - montage_points_mm[:, :3].values, axis=2)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        coords_true = montage_points_mm[col_ind, :3].values
        coords = opt_centers_mm[row_ind, :3].values

        def loss(params, coords_to, coords_from):
            transformation_matrix = complete_transformation(params)
            transformed_montage = apply_numpy_transform(coords_from[:, :3], transformation_matrix)
            return np.sum((coords_to - transformed_montage) ** 2)

        no_bounds = (None, None)
        bounds = [
            no_bounds,
            no_bounds,
            no_bounds,
            (-2 * np.pi, 2 * np.pi),
            (-2 * np.pi, 2 * np.pi),
            (-2 * np.pi, 2 * np.pi),
            (0.5, 1.5),
            (0.5, 1.5),
            (0.5, 1.5),
        ]

        # Optimization step to minimize the loss function
        result = minimize(
            loss, current_params, args=(coords_true, coords), bounds=bounds
        )
        # print(result.fun)
        # Update if improvement
        if result.fun < best_loss:
            best_loss = result.fun
            current_params = result.x
            best_idx = col_ind

        # Convergence check
        if best_loss < tolerance:
            break

        transformation_matrix = complete_transformation(current_params)

    return best_idx




def find_spread_points(points_xr):
    """Selects three points from a given set of points that are spread apart from each other in the dataset.

    Parameters:
        points_xr (xarray.DataArray): An xarray DataArray containing the points from which to select. 

    Returns:
        np.ndarray: Indices of the initial, farthest, and median-distanced points from the initial point 
                    as determined by their positions in the original dataset.
    """

    points = points_xr.values
    if len(points) < 3:
        return list(range(len(points)))  # Not enough points to select from

    # Construct KDTree from points
    tree = KDTree(points)

    # Step 1: Select the initial point (e.g., the first point)
    initial_point_index = 0
    initial_point = points[initial_point_index]

    # Step 2: Find the farthest point from the initial point
    distances, _ = tree.query(initial_point, k=len(points))
    farthest_point_index = np.argmax(distances)

    # Step 3: Find the "middle-distanced" point from the initial point
    sorted_distances_indices = np.argsort(distances)
    median_index = sorted_distances_indices[len(sorted_distances_indices) // 2]
    middle_distanced_point_index = median_index if median_index != initial_point_index else sorted_distances_indices[len(sorted_distances_indices) // 2 + 1]

    return points_xr.label.isel(label=[initial_point_index, farthest_point_index, middle_distanced_point_index]).values
