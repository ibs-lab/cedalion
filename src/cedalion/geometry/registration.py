"""Registrating optodes to scalp surfaces."""

from dataclasses import dataclass
import numpy as np
from numpy.linalg import pinv
from scipy.optimize import linear_sum_assignment, minimize
from scipy.spatial import KDTree
import xarray as xr
import warnings
import pandas as pd

import cedalion
import cedalion.dataclasses as cdc
import cedalion.geometry.utils
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion.errors import CRSMismatchError


from .utils import m_rot, m_scale1, m_scale3, m_trans


@dataclass
class SpringICPResult:
    """Quality-control details returned by :func:`register_optodes_spring_icp`.

    All distances and displacements are in the same units as the scalp surface.
    """

    initial_positions: cdt.LabeledPoints
    """Positions after the initial landmark alignment, before spring relaxation."""

    nominal_distances: dict
    """Nominal channel distances ``{(src_label, det_label): float}`` used as spring
    rest lengths.  Measured in scalp units from the phase-1 aligned positions unless
    supplied by the caller."""

    spring_errors: xr.DataArray
    """Per-channel ``actual_distance - nominal_distance`` at convergence.
    Dim ``channel``; auxiliary coords ``source`` and ``detector`` carry the
    individual optode labels."""

    landmark_errors: xr.DataArray
    """Per-anchor Euclidean distance between the final optode position and the
    target landmark position at convergence.  Dim ``label``.  Empty when no
    optode labels overlap with ``landmarks_scalp``."""

    snap_displacement_per_iter: np.ndarray
    """Maximum surface-projection displacement (over all optodes) per iteration.
    Shape ``(n_iterations,)``.  Use as a convergence diagnostic."""

    n_iterations: int
    """Number of iterations actually performed."""

    converged: bool
    """``True`` when the convergence criterion was met before *n_iter*."""


def _subtract(a: cdt.LabeledPoints, b: cdt.LabeledPoints):
    """Calculate difference vectors between points and check CRS."""

    crs_a = a.points.crs
    crs_b = b.points.crs

    if not crs_a == crs_b:
        raise ValueError("point clouds are using different coordinate systems.")

    return a - b

@cdc.validate_schemas
def register_identity(
    coords_target: cdt.LabeledPoints,
    coords_trafo: cdt.LabeledPoints,
):
    """Affine transformation that just adapts units.

    Args:
        coords_target (LabeledPoints): Target point cloud.
        coords_trafo (LabeledPoints): Source point cloud.

    Returns:
        cdt.AffineTransform: Affine transformation between the two point clouds.
    """

    from_crs = coords_trafo.points.crs
    from_units = coords_trafo.pint.units
    to_crs = coords_target.points.crs
    to_units = coords_target.pint.units

    unit_scale_factor = ((1 * from_units) / (1 * to_units)).to_reduced_units()
    assert unit_scale_factor.units == cedalion.units.Unit("1")
    unit_scale_factor = float(unit_scale_factor)

    trafo = cdc.affine_transform_from_numpy(
        m_scale1([unit_scale_factor]),
        from_crs=from_crs,
        to_crs=to_crs,
        from_units=from_units,
        to_units=to_units,
    )

    return trafo


@cdc.validate_schemas
def register_trans_rot(
    coords_target: cdt.LabeledPoints,
    coords_trafo: cdt.LabeledPoints,
):
    """Finds affine transformation between coords_target and coords_trafo.

    Uses translation and roatation rotation. Requires at least 3 common labels
    between the two point clouds.

    Args:
        coords_target (LabeledPoints): Target point cloud.
        coords_trafo (LabeledPoints): Source point cloud.

    Returns:
        cdt.AffineTransform: Affine transformation between the two point clouds.
    """
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
        options={"disp": False},
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
def register_general_affine(
    coords_target: cdt.LabeledPoints,
    coords_trafo: cdt.LabeledPoints,
):
    """Finds affine transformation between coords_target and coords_trafo.

    This method fits all 12 parameters of a 3D affine transformation without
    constraints. The transform thus contains scaling, rotation, translation,
    shear and mirroring, i.e. it can transform between left and right handed
    coordinate systems.

    Args:
        coords_target (LabeledPoints): Target point cloud.
        coords_trafo (LabeledPoints): Source point cloud.

    Returns:
        cdt.AffineTransform: Affine transformation between the two point clouds.
    """
    common_labels = coords_target.points.common_labels(coords_trafo)

    if len(common_labels) < 4:
        raise ValueError("less than 4 common coordinates found")

    from_crs = coords_trafo.points.crs
    from_units = coords_trafo.pint.units
    to_crs = coords_target.points.crs
    to_units = coords_target.pint.units

    # restrict to commmon labels and dequantify
    coords_trafo = coords_trafo.sel(label=common_labels).pint.dequantify()
    coords_target = coords_target.sel(label=common_labels).pint.dequantify()

    # Warn if the source landmark cloud (restricted to common labels) is
    # nearly coplanar — because a 12-DOF affine fit then collapses along the
    # plane normal. Threshold 0.20 separates the Nz/Iz/LPA/RPA-only case
    # (ratio ~0.12, observed 32% z-collapse on colin27) from configurations
    # that include an off-plane landmark like Cz (ratio ~0.85) or the full
    # 10-10 set (ratio ~0.7). Ratio numbers stem from experiences after
    # extensive testing of different input landmarks on 15 subjects.
    src_xyz = coords_trafo.values
    src_centered = src_xyz - src_xyz.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(src_centered, compute_uv=False)
    if sv[0] > 0 and sv[-1] / sv[0] < 0.20:
        warnings.warn(
            f"register_general_affine: source landmarks are nearly coplanar "
            f"(sigma_min/sigma_max = {sv[-1]/sv[0]:.3f} on the "
            f"{len(common_labels)} common labels). The 12-DOF affine fit "
            f"will be poorly constrained perpendicular to the plane. "
            f"Consider adding an off-plane landmark (for fNIRS/EEG fiducial "
            f"sets, Cz works well).",
            UserWarning,
            stacklevel=2,
        )

    # Create augmented matrix with homogeneous coordinates
    ones = np.ones((len(common_labels), 1))
    coords_trafo_hom = np.hstack([coords_trafo, ones])  # (N, 4)
    coords_target_hom = np.hstack([coords_target, ones])  # (N, 4)

    # Solve for transformation matrix using least squares
    # coords_trafo_hom @ A.T = coords_target_hom
    A, residuals, rank, s = np.linalg.lstsq(
        coords_trafo_hom, coords_target_hom, rcond=None
    )

    trafo_opt = cdc.affine_transform_from_numpy(
        A.T,
        from_crs=from_crs,
        to_crs=to_crs,
        from_units=from_units,
        to_units=to_units,
    )

    return trafo_opt



def _std_distance_to_cog(points: cdt.LabeledPoints):
    """Calculate the standard deviation of the distances to the center of gravity.

    Args:
        points: Point cloud for which to calculate the standard deviation of the
            distances to the center of gravity.

    Returns:
        float: Standard deviation of the distances to the center of gravity.
    """
    dists = xrutils.norm(points - points.mean("label"), points.points.crs)
    return dists.std("label").item()


@cdc.validate_schemas
def register_trans_rot_isoscale(
    coords_target: cdt.LabeledPoints,
    coords_trafo: cdt.LabeledPoints,
):
    """Finds affine transformation between coords_target and coords_trafo.

    Uses translation, rotation and isotropic scaling. Requires at least 3 common labels
    between the two point clouds.

    Args:
        coords_target (LabeledPoints): Target point cloud.
        coords_trafo (LabeledPoints): Source point cloud.

    Returns:
        cdt.AffineTransform: Affine transformation between the two point clouds.
    """
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
def register_trans_rot_scale(
    coords_target: cdt.LabeledPoints,
    coords_trafo: cdt.LabeledPoints,
):
    """Finds affine transformation between coords_target and coords_trafo.

    Uses translation, rotation and scaling. Requires at least 3 common labels
    between the two point clouds.

    Args:
        coords_target (LabeledPoints): Target point cloud.
        coords_trafo (LabeledPoints): Source point cloud.

    Returns:
        cdt.AffineTransform: Affine transformation between the two point clouds.
    """
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
        return m_rot(params[3:6]) @ m_trans(params[0:3]) @ m_scale3(params[6:9])

    def loss(params, coords_target, coords_trafo):
        M = trafo(params)
        tmp = coords_trafo.points._apply_numpy_transform(M, to_crs)
        return np.power(_subtract(coords_target, tmp), 2).sum()

    result = minimize(
        loss,
        [
            delta_cog[0],
            delta_cog[1],
            delta_cog[2],
            0.0,
            0.0,
            0.0,
            scale0,
            scale0,
            scale0,
        ],
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



def gen_xform_from_pts(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Calculate the affine transformation matrix T that transforms p1 to p2.

    Args:
        p1 (np.ndarray): Source points (p x m) where p is the number of points and m is
            the number of dimensions.
        p2 (np.ndarray): Target points (p x m) where p is the number of points and m is
            the number of dimensions.

    Returns:
        Affine transformation matrix T.
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
            f"Cannot solve transformation with fewer anchor "
            f"points ({p}) than dimensions ({n})."
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
    landmarks: cdt.LabeledPoints,
    geo3d: cdt.LabeledPoints,
    niterations=1000,
    random_sample_fraction=0.5,
):
    """Iterative Closest Point algorithm for registration.

    Args:
        surface (Surface): Surface mesh to which to register the points.
        landmarks (LabeledPoints): Landmarks to use for registration.
        geo3d (LabeledPoints): Points to register to the surface.
        niterations (int): Number of iterations for the ICP algorithm (default 1000).
        random_sample_fraction (float): Fraction of points to use in each iteration
            (default 0.5).

    Returns:
        Tuple[np.ndarray, np.ndarray]: Tuple containing the losses and transformations
    """
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


# FIXME: returns only indices?
def icp_with_full_transform(
    opt_centers: cdt.LabeledPoints,
    montage_points: cdt.LabeledPoints,
    max_iterations: int = 50,
    tolerance: float = 500.0,
):
    """Perform Iterative Closest Point algorithm with full transformation capabilities.

    Args:
        opt_centers: Source point cloud for alignment.
        montage_points: Target reference point cloud.
        max_iterations: Maximum number of iterations for convergence.
        tolerance: Tolerance for convergence check.

    Returns:
        np.ndarray: Transformed source points as a numpy array with their coordinates
            updated to reflect the best alignment.
        np.ndarray: Transformation parameters array consisting of
            [tx, ty, tz, rx, ry, rz, sx, sy, sz], where 't' stands for
            translation components, 'r' for rotation components (in radians), and 's'
            for scaling components.
        np.ndarray: Indices of the target points that correspond to each source point as
            per the nearest neighbor search.
    """

    # Convert to homogeneous coordinates, assuming .values and .pint.dequantify() yield
    # np.ndarray
    units = "mm"
    opt_centers_mm = (
        opt_centers.pint.to(units).points.to_homogeneous().pint.dequantify()
    )
    montage_points_mm = (
        montage_points.pint.to(units).points.to_homogeneous().pint.dequantify()
    )

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

    def apply_numpy_transform(
        obj_values, transform: np.ndarray, to_crs=None, obj_units=None
    ):
        transformed_values = (
            np.hstack((obj_values, np.ones((obj_values.shape[0], 1)))) @ transform.T
        )
        transformed_values = transformed_values[:, :-1]
        return transformed_values

    for iteration in range(max_iterations):
        opt_centers_mm[:, :3] = apply_numpy_transform(
            opt_centers_mm[:, :3], transformation_matrix
        )

        # Use the Hungarian algorithm to find the optimal assignment
        cost_matrix = np.linalg.norm(
            opt_centers_mm[:, :3].values[:, np.newaxis]
            - montage_points_mm[:, :3].values,
            axis=2,
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        coords_true = montage_points_mm[col_ind, :3].values
        coords = opt_centers_mm[row_ind, :3].values

        # FIXME move these out of the loop
        def loss(params, coords_to, coords_from):
            transformation_matrix = complete_transformation(params)
            transformed_montage = apply_numpy_transform(
                coords_from[:, :3], transformation_matrix
            )
            return np.sum((coords_to - transformed_montage) ** 2)

        no_bounds = (None, None)
        bounds = [
            no_bounds,
            no_bounds,
            no_bounds,
            (-2 * np.pi, 2 * np.pi),
            (-2 * np.pi, 2 * np.pi),
            (-2 * np.pi, 2 * np.pi),
            (-1.5, 1.5),
            (-1.5, 1.5),
            (-1.5, 1.5),
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


def find_spread_points(points_xr: xr.DataArray) -> np.ndarray:
    """Selects three points that are spread apart from each other in the dataset.

    Args:
        points_xr: An xarray DataArray containing the points from which to select.

    Returns:
        Indices of the initial, farthest, and median-distanced points from the initial
        point as determined by their positions in the original dataset.
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
    middle_distanced_point_index = (
        median_index
        if median_index != initial_point_index
        else sorted_distances_indices[len(sorted_distances_indices) // 2 + 1]
    )

    return points_xr.label.isel(
        label=[initial_point_index, farthest_point_index, middle_distanced_point_index]
    ).values


def simple_scalp_projection(geo3d: cdt.LabeledPoints) -> cdt.LabeledPoints:
    """Projects 3D coordinates onto a 2D plane using a simple scalp projection.

    Args:
        geo3d (LabeledPoints): 3D coordinates of points to project. Requires the
            landmarks Nz/Nasion, LPA, and RPA.

    Returns:
        A LabeledPoints containing the 2D coordinates of the projected points.
    """
    lpa = None
    rpa = None
    nz = None

    for label in geo3d.label.values:
        if label.lower() == "lpa":
            lpa = label
        if label.lower() == "rpa":
            rpa = label
        if label.lower() in {"nz", "nasion"}:
            nz = label

    if (lpa is None) or (rpa is None) or (nz is None):
        raise ValueError("this projection needs the landmarks Nz/Nasion, LPA and RPA.")

    crs = geo3d.points.crs
    # find the midpoint between LPA and RPA
    center = 0.5 * (geo3d.sel(label=lpa) + geo3d.sel(label=rpa))

    # calculate unit vectors of RAS coordindates
    ex = geo3d.sel(label=rpa) - center
    ey = geo3d.sel(label=nz) - center

    # use same norm for ex and ey so that RPA is at (1,0,0) and Nz at (0,1,0)
    normx = xrutils.norm(ex, crs).item()
    ex = (ex / normx).pint.dequantify()
    ey = (ey / normx).pint.dequantify()

    ez = xr.zeros_like(ey)
    ez[:] = np.cross(ex, ey)
    ez /= xrutils.norm(ez, crs)

    # calculate RAS coordinates
    centered = geo3d - center
    r = xr.dot(centered, ex)
    a = xr.dot(centered, ey)
    s = xr.dot(centered, ez)

    # transform into spherical coordinates
    az, el, r = cedalion.geometry.utils.cart2sph(r, a, s)

    # project into 2D by using the spherical azimuth and cos(elevation) as radius
    r2 = np.cos(el)
    x2 = r2 * np.cos(az)
    y2 = r2 * np.sin(az)

    geo2d = xr.concat((x2, y2), dim="proj2d").transpose("label", ...)
    geo2d["type"] = geo3d.type
    geo2d = geo2d.pint.dequantify()  # no units needed in this space

    return geo2d


def _mesh_closest_point(
    mesh, points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the closest point on *mesh* for each row of *points*.

    Tries ``trimesh.proximity.closest_point`` first (requires the *rtree* package
    for its spatial index).  When *rtree* is absent falls back to a pure-scipy
    implementation: scipy KDTree finds the *k* nearest vertices, then
    ``trimesh.triangles.closest_point`` checks every face adjacent to those
    vertices and returns the best hit.

    Returns:
        closest_pts : (N, 3) float – nearest surface points
        distances   : (N,)   float – Euclidean distances from *points*
        triangle_ids: (N,)   int   – index of the nearest triangle
    """
    import trimesh.proximity
    import trimesh.triangles

    try:
        return trimesh.proximity.closest_point(mesh, points)
    except ModuleNotFoundError:
        pass

    # --- KDTree fallback (no rtree) -------------------------------------------
    # Build a per-vertex KD-tree once; trimesh caches it on the mesh object.
    kdtree = KDTree(mesh.vertices)

    # k nearest vertices per query point; covers all adjacent faces for typical
    # meshes where each vertex has ≤ 8 neighbours.
    k = min(12, len(mesh.vertices))
    _, vert_idxs = kdtree.query(points, k=k)          # (N, k)

    # mesh.vertex_faces: (V, max_valence), -1 for unused slots
    vf = mesh.vertex_faces                             # cached by trimesh

    n_pts = len(points)
    closest_pts = np.empty((n_pts, 3), dtype=float)
    distances   = np.empty(n_pts, dtype=float)
    triangle_ids = np.empty(n_pts, dtype=np.intp)

    for i in range(n_pts):
        # collect unique valid face indices around the k nearest vertices
        raw = vf[vert_idxs[i]].ravel()
        face_cands = np.unique(raw[raw >= 0])

        tris = mesh.vertices[mesh.faces[face_cands]]      # (m, 3, 3)
        pts_rep = np.broadcast_to(points[i], (len(tris), 3))
        cps = trimesh.triangles.closest_point(tris, pts_rep)  # (m, 3)

        dists = np.linalg.norm(cps - points[i], axis=1)
        best  = np.argmin(dists)

        closest_pts[i]  = cps[best]
        distances[i]    = dists[best]
        triangle_ids[i] = face_cands[best]

    return closest_pts, distances, triangle_ids


def register_optodes_spring_icp(
    scalp: cdc.TrimeshSurface,
    geo3d: cdt.LabeledPoints,
    channels: list[tuple[str, str]] | cdt.NDTimeSeries | pd.DataFrame,
    landmarks_scalp: cdt.LabeledPoints,
    nominal_distances: dict[tuple[str, str], float] | None = None,
    n_iter: int = 400,
    k_spring: float = 1.0,
    k_anchor: float = 10.0,
    step_size: float = 0.1,
    convergence_tol: float = 0.01,
    initial_align_mode: str  = "general",
) -> tuple[cdt.LabeledPoints, SpringICPResult]:
    """Register optode coordinates onto a scalp mesh via spring relaxation and ICP.

    Two-phase registration:

    1. **Initial alignment** — a global transform (translation + rotation +
       optional scale) is fitted to the matched landmark pairs and applied to
       ``geo3d`` to bring them into the scalp coordinate system.

    2. **Spring-relaxation ICP** — optodes are iteratively refined:

       * Hooke's-law springs between each channel-forming source–detector pair,
         with rest length equal to the nominal inter-optode distance, resist
         distortion of channel geometry.
       * Strong anchor springs pull any coordinate in geo3d whose label appears in
         ``landmarks_scalp`` toward its known anatomical position on the scalp.
       * After each force step every optode is projected onto the nearest point
         on the scalp triangulated surface (ICP step).

    Args:
        scalp: Triangulated scalp surface in the target coordinate system.
        geo3d: Probe coordinates (sources, detectors and landmarks) in the probe
            coordinate system.
        channels: ``(source_label, detector_label)`` pairs defining which
            optode pairs form channels and therefore get a spring.
        landmarks_scalp: Anatomical landmark positions in the scalp CRS (same
            CRS as ``scalp``).  Any label that also appears in ``optodes`` is
            used as a strong anchor during the relaxation phase.
        nominal_distances: Optional pre-specified rest lengths for each channel
            spring, keyed by ``(src_label, det_label)``.  When *None*,
            distances are measured from the phase-1 aligned positions.
        n_iter: Maximum number of relaxation iterations.
        k_spring: Spring constant for channel springs.  Force magnitude scales
            linearly with extension from the nominal distance.
        k_anchor: Spring constant for landmark-anchor springs.  Should exceed
            ``k_spring`` to enforce anatomical constraints strongly.
        step_size: Fraction of the net force vector added to each position
            per iteration.  Reduce if the relaxation is unstable.
        convergence_tol: Stop early when the maximum surface-projection
            displacement across all optodes is below this value (in scalp
            units).
        initial_align_mode: Phase-1 transform type.  One of
            ``"trans_rot_isoscale"`` (7 DOF, default), ``"trans_rot"``
            (6 DOF), ``"general"`` (12 DOF full affine), or ``"identity"`` (only
            adapt units).

    Returns:
        :class:`SpringICPResult` with the final optode positions and
        per-iteration / per-channel quality-control metrics.

    Raises:
        CRSMismatchError: If ``landmarks_scalp`` is not in the same CRS as
            ``scalp``.
        ValueError: If ``initial_align_mode`` is not recognised, or if a
            channel label is absent from ``optodes``.
    """
    if landmarks_scalp.points.crs != scalp.crs:
        raise CRSMismatchError.unexpected_crs(
            expected_crs=scalp.crs, found_crs=landmarks_scalp.points.crs
        )


    if isinstance(channels, pd.DataFrame):
        # measurement list
        assert all(c in channels.columns for c in ["wavelength", "source", "detector"])
        wavelengths = channels.wavelength.unique()
        # restrict to single wavelength
        channels = channels[channels.wavelength == wavelengths[0]]
        channels = [(r.source, r.detector) for _,r in channels.iterrows()]
    elif isinstance(channels, xr.DataArray):
        # NDTime Series
        assert all(c in channels.coords for c in ["source", "detector"])
        channels = [
            (s, d) for s, d in zip(channels.source.values, channels.detector.values)
        ]
    elif isinstance(channels, list):
        # list of optode label pairs
        pass
    else:
        raise ValueError(
            "Channel definition must be either a NDTimeSeries, "
            "a measurement list DataFrame or a list of optode label pairs."
        )


    # --- Phase 1: global alignment from landmark correspondences ---------------

    landmarks_probe = geo3d[
        (geo3d.type == cdc.PointType.LANDMARK) | (geo3d.type == cdc.PointType.ELECTRODE)
    ]

    if initial_align_mode == "trans_rot_isoscale":
        t_init = register_trans_rot_isoscale(landmarks_scalp, landmarks_probe)
    elif initial_align_mode == "trans_rot":
        t_init = register_trans_rot(landmarks_scalp, landmarks_probe)
    elif initial_align_mode == "general":
        t_init = register_general_affine(landmarks_scalp, landmarks_probe)
    elif initial_align_mode == "identity":
        t_init = register_identity(landmarks_scalp, landmarks_probe)
    else:
        raise ValueError(
            f"Unknown initial_align_mode {initial_align_mode!r}. "
            "Choose 'trans_rot_isoscale', 'trans_rot', 'general', or 'identity'."
        )

    aligned = geo3d.points.apply_transform(t_init)
    aligned_dq = aligned.pint.dequantify()
    # pos_np: (N, 3) float array in scalp units – the working copy
    pos_np = aligned_dq.values.copy()

    labels = list(aligned.label.values)
    label_to_idx = {lbl: i for i, lbl in enumerate(labels)}

    missing = [lbl for ch in channels for lbl in ch if lbl not in label_to_idx]
    if missing:
        raise ValueError(
            f"Channel labels not found in optodes: {sorted(set(missing))}"
        )

    # Nominal rest lengths in scalp units, computed from aligned positions
    if nominal_distances is None:
        nominal_distances = {
            (src, det): float(
                np.linalg.norm(pos_np[label_to_idx[src]] - pos_np[label_to_idx[det]])
            )
            for src, det in channels
        }

    # Anchor targets: optodes whose label also appears in landmarks_scalp
    lm_scalp_dq = landmarks_scalp.pint.dequantify()
    scalp_lm_label_set = set(landmarks_scalp.label.values)
    anchor_labels = [lbl for lbl in labels if lbl in scalp_lm_label_set]
    anchor_targets = {
        lbl: lm_scalp_dq.sel(label=lbl).values
        for lbl in anchor_labels
    }

    # Pre-index pairs to keep the hot loop free of dict lookups
    channel_pairs = [
        (label_to_idx[src], label_to_idx[det], nominal_distances[(src, det)])
        for src, det in channels
    ]
    anchor_pairs = [
        (label_to_idx[lbl], anchor_targets[lbl])
        for lbl in anchor_labels
    ]

    # Per-optode channel count: normalise spring contributions so that an
    # optode connected to many channels receives the same total spring weight
    # as one connected to few, preventing heavily-connected optodes (typically
    # sources) from accumulating disproportionate force.
    valence = np.zeros(len(labels), dtype=float)
    for i_src, i_det, _ in channel_pairs:
        valence[i_src] += 1
        valence[i_det] += 1
    valence = np.where(valence > 0, valence, 1.0)  # avoid division by zero

    # --- Phase 2: spring-relaxation ICP ----------------------------------------

    snap_displacement_per_iter: list[float] = []
    n_iter_actual = n_iter
    converged = False

    for i_iter in range(n_iter):
        forces = np.zeros_like(pos_np)

        # Hooke's-law spring forces between channel-forming pairs,
        # normalised by each optode's channel count so that sources and
        # detectors with different numbers of connections experience the
        # same total spring weight.
        for i_src, i_det, L0 in channel_pairs:
            delta = pos_np[i_det] - pos_np[i_src]
            d = np.linalg.norm(delta)
            if d > 1e-10:
                spring_force = k_spring * (d - L0) * (delta / d)
                forces[i_src] += spring_force / valence[i_src]
                forces[i_det] -= spring_force / valence[i_det]

        # Anchor forces pulling landmarks to their scalp targets
        for i_lm, target in anchor_pairs:
            forces[i_lm] += k_anchor * (target - pos_np[i_lm])

        # Remove the surface-normal component of every force so that the ICP
        # projection step only has to correct numerical drift rather than
        # fight forces that would immediately be undone by snapping.
        _, nearest_vi = scalp.kdtree.query(pos_np)
        normals = scalp.mesh.vertex_normals[nearest_vi]  # (N, 3)
        normal_component = np.sum(forces * normals, axis=1, keepdims=True)
        forces -= normal_component * normals

        pos_np = pos_np + step_size * forces

        # ICP step: project each optode onto the nearest point on the surface
        proj_pts, snap_dists, _ = _mesh_closest_point(scalp.mesh, pos_np)
        max_snap = float(np.max(snap_dists))
        snap_displacement_per_iter.append(max_snap)
        pos_np = proj_pts

        if max_snap < convergence_tol:
            n_iter_actual = i_iter + 1
            converged = True
            break

    # --- Assemble result -------------------------------------------------------

    # Rebuild as a quantified LabeledPoints with the same coords as aligned
    pos_out = aligned_dq.copy(data=pos_np)
    aligned_units = aligned.pint.units
    if aligned_units is not None:
        pos_out = pos_out.pint.quantify(aligned_units)

    # Spring errors: signed length deviation from nominal at convergence
    spring_errors = xr.DataArray(
        np.array([
            np.linalg.norm(pos_np[label_to_idx[det]] - pos_np[label_to_idx[src]])
            - nominal_distances[(src, det)]
            for src, det in channels
        ]),
        dims=["channel"],
        coords={
            "channel": [f"{src}-{det}" for src, det in channels],
            "source":   ("channel", [src for src, _ in channels]),
            "detector": ("channel", [det for _, det in channels]),
        },
    )

    # Landmark anchor errors: distance to target position at convergence
    if anchor_labels:
        landmark_errors = xr.DataArray(
            np.array([
                np.linalg.norm(pos_np[label_to_idx[lbl]] - anchor_targets[lbl])
                for lbl in anchor_labels
            ]),
            dims=["label"],
            coords={"label": anchor_labels},
        )
    else:
        landmark_errors = xr.DataArray(
            np.empty(0, dtype=float),
            dims=["label"],
            coords={"label": np.array([], dtype=object)},
        )

    return pos_out, SpringICPResult(
        initial_positions=aligned,
        nominal_distances=nominal_distances,
        spring_errors=spring_errors,
        landmark_errors=landmark_errors,
        snap_displacement_per_iter=np.array(snap_displacement_per_iter),
        n_iterations=n_iter_actual,
        converged=converged,
    )
