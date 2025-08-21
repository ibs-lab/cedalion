"""Functions for generating synthetic hemodynamic response functions."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import pyvista as pv
import scipy.stats as stats
import xarray as xr

import cedalion.dataclasses as cdc
import cedalion.dataclasses.geometry as cdg
import cedalion.dot.forward_model as cfm
import cedalion.dot.head_model as cdhm
import cedalion.models.glm as glm
import cedalion.plots
import cedalion.typing as cdt
from cedalion import units
from cedalion.models.glm.basis_functions import TemporalBasisFunction


def build_spatial_activation(
    head_model: cdhm.TwoSurfaceHeadModel,
    seed_vertex: int,
    spatial_scale: cdt.QLength = 1 * units.cm,
    intensity_scale: cdt.QConcentration = 1 * units.micromolar,
    hbr_scale: float = None,
    m: float = 10.0,
):
    """Generates a spatial activation at a seed vertex.

    This function generates a blob of activity on the brain surface.
    The blob is centered at the seed vertex.
    Geodesic distances, and therefore also the blob, can be distorded
    due to mesh decimation or unsuitable m value.

    Args:
        head_model (cfm.TwoSurfaceHeadModel): Head model with brain and scalp surfaces.
        seed_vertex (int): Index of the seed vertex.
        spatial_scale (Quantity): Scale of the spatial size.
        intensity_scale (Quantity): Scaling factor for the intensity of the blob.
        hbr_scale (float): Scaling factor for HbR relative to HbO. If None, the blob
            will have no concentration dimension and only represent HbO.
        m (float): Geodesic distance parameter. Larger values of m will smooth &
            regularize the distance computation. Smaller values of m will roughen and
            will usually increase error in the distance computation.

    Returns:
        xr.DataArray: Spatial image with activation values for each vertex.

    Initial Contributors:
        - Thomas Fischer | t.fischer.1@campus.tu-berlin.de | 2024

    """

    spatial_scale_unit = (
        (spatial_scale / head_model.brain.units).to_base_units().magnitude
    )

    seed_pos = head_model.brain.mesh.vertices[seed_vertex]

    # if the mesh is not contiguous:
    # only calculate distances on the submesh on which the seed vertex lies

    # get a list of the distinct submeshes
    mesh_split = head_model.brain.mesh.split(only_watertight=False)

    # check in which submesh the seed vertex is
    for i, submesh in enumerate(mesh_split):
        if seed_pos in submesh.vertices:
            break

    # get index of the seed vertex in the submesh
    seed_vertex = np.where((submesh.vertices == seed_pos).all(axis=1))[0]

    # create a pycortex surface of the submesh to calculate geodesic distances
    cortex_surface = cdg.PycortexSurface(
        cdg.SimpleMesh(submesh.vertices, submesh.faces),
        crs=head_model.brain.crs,
        units=head_model.brain.units,
    )
    distances_on_submesh = cortex_surface.geodesic_distance([seed_vertex], m=m)

    # find indices of submesh in original mesh
    # convert meshes into set of tuples for fast lookup
    submesh_set = set(map(tuple, submesh.vertices))
    submesh_indices = [
        i
        for i, coord in enumerate(map(tuple, head_model.brain.mesh.vertices))
        if coord in submesh_set
    ]

    # set distances on vertices outside of the submesh to inf
    distances_from_seed = np.ones(head_model.brain.mesh.vertices.shape[0]) * np.inf
    distances_from_seed[submesh_indices] = distances_on_submesh

    # plug the distances in a normal distribution
    norm_pdf = stats.norm(scale=spatial_scale_unit).pdf

    blob_img = norm_pdf(distances_from_seed)
    blob_img = blob_img / np.max(blob_img)
    blob_img = xr.DataArray(blob_img, dims=["vertex"])

    if hbr_scale is not None:
        blob_img = np.stack([blob_img, blob_img * hbr_scale], axis=1)
        blob_img = xr.DataArray(
            blob_img,
            dims=["vertex", "chromo"],
            coords={"chromo": ["HbO", "HbR"]},
        )

    blob_img = blob_img * intensity_scale

    blob_img = blob_img.pint.to(units.molar)

    return blob_img


def build_stim_df(
    max_time: cdt.QTime,
    max_num_stims: int | None = None,
    trial_types: list[str] = ["Stim"],
    min_stim_dur: cdt.QTime = 10 * units.seconds,
    max_stim_dur: cdt.QTime = 10 * units.seconds,
    min_interval: cdt.QTime = 10 * units.seconds,
    max_interval: cdt.QTime = 30 * units.seconds,
    min_stim_value: float = 1.0,
    max_stim_value: float = 1.0,
    order: str = "alternating",
):
    """Generates a DataFrame for stimulus metadata based on provided parameters.

    Stimuli can be added in an 'alternating' or 'random' order, and the inter-stimulus
    interval (ISI) is chosen randomly between the minimum and maximum allowed intervals.

    Args:
        max_time (Quantity): Maximum total duration for the stimuli.
        max_num_stims (int): Maximum number of stimuli to be added for each trial type.
        trial_types (list): List of different trial types.
        min_stim_dur (Quantity): Minimum duration of the stimulus.
        max_stim_dur (Quantity): Maximum duration of the stimulus.
        min_interval (Quantity): Minimum inter-stimulus interval.
        max_interval (Quantity): Maximum inter-stimulus interval.
        min_stim_value (float): Minimum amplitude-value of the stimulus.
        max_stim_value (float): Maximum amplitude-value of the stimulus.
        order (str): Order of adding Stims; 'alternating' or 'random'.

    Returns:
        pd.DataFrame: DataFrame containing stimulus metadata.

    Initial Contributors:
        - Laura Carlton | lcarlton@bu.edu | 2024
        - Thomas Fischer | t.fischer.1@campus.tu-berlin.de | 2024
    """

    # Calculate a default number of stimuli if not provided, based on max_time
    if max_num_stims is None:
        max_num_stims = int(
            (max_time / ((min_stim_dur + min_interval) * len(trial_types)))
            .to_base_units()
            .magnitude
        )

    # Convert all time-related quantities to seconds
    min_stim_dur = min_stim_dur.to("s").magnitude
    max_stim_dur = max_stim_dur.to("s").magnitude
    min_interval = min_interval.to("s").magnitude
    max_interval = max_interval.to("s").magnitude
    max_time = max_time.to("s").magnitude

    current_time = round(random.uniform(min_interval, max_interval), 2)
    onset_times = []
    onset_trial_types = []
    stim_durations = []
    stim_values = []

    trial_types = [str(tt) for tt in trial_types]
    order = order.lower()

    if order not in ["alternating", "random"]:
        raise ValueError(
            f"Invalid order '{order}'. Must be either 'alternating' or 'random'."
        )

    if order == "alternating":
        stim_index = 0
        while stim_index < max_num_stims:
            for trial_type in trial_types:
                stim_dur = round(random.uniform(min_stim_dur, max_stim_dur), 2)
                interval = round(random.uniform(min_interval, max_interval), 2)
                next_time = current_time + stim_dur + interval
                if next_time > max_time:
                    # break the outer loop
                    stim_index = max_num_stims
                    break
                stim_durations.append(stim_dur)
                onset_times.append(current_time)
                onset_trial_types.append(trial_type)
                stim_val = round(random.uniform(min_stim_value, max_stim_value), 2)
                stim_values.append(stim_val)
                current_time = next_time
            stim_index += 1

    elif order == "random":
        stims_left = {trial_type: max_num_stims for trial_type in trial_types}
        while any(stims_left.values()):
            trial_type = random.choices(
                list(trial_types),
                weights=[stims_left[tt] for tt in trial_types],
            )[0]
            stim_dur = round(random.uniform(min_stim_dur, max_stim_dur), 2)
            interval = round(random.uniform(min_interval, max_interval), 2)
            next_time = current_time + stim_dur + interval
            if next_time > max_time:
                break
            stim_durations.append(stim_dur)
            onset_times.append(current_time)
            onset_trial_types.append(trial_type)
            stim_val = round(random.uniform(min_stim_value, max_stim_value), 2)
            stim_values.append(stim_val)
            current_time = next_time
            stims_left[trial_type] -= 1

    # Create the DataFrame with onset, duration, and trial type info
    stim_df = pd.DataFrame(
        {
            "onset": onset_times,
            "duration": stim_durations,
            "value": stim_values,
            "trial_type": onset_trial_types,
        }
    )

    return stim_df


@cdc.validate_schemas
def build_synthetic_hrf_timeseries(
    ts: cdt.NDTimeSeries,
    stim_df: pd.DataFrame,
    basis_fct: TemporalBasisFunction,
    spatial_pattern: xr.DataArray,
):
    """Builds a synthetic HRF timeseries based on the provided data.

    Args:
        ts (cdt.NDTimeSeries): Timeseries data.
        stim_df (pd.DataFrame): DataFrame containing stimulus metadata.
        basis_fct (TemporalBasisFunction): Temporal basis function defining the HRF.
        spatial_pattern (xr.DataArray): Spatial activation pattern (intensity scaling
                                        for each vertex/channel and trial type).

    Returns:
        cdt.NDTimeSeries: Synthetic HRF timeseries.

    Initial Contributors:
        - Thomas Fischer | t.fischer.1@campus.tu-berlin.de | 2024
    """

    dms = glm.design_matrix.hrf_regressors(ts, stim_df, basis_fct)
    hrf_regs = dms.common
    hrf_regs *= stim_df.value.max()

    # remove HRF prefix from regressor names
    hrf_regs = hrf_regs.assign_coords(
        regressor=[
            regressor.removeprefix("HRF ") for regressor in hrf_regs.regressor.values
        ]
    )
    hrf_regs = hrf_regs.rename({"regressor": "trial_type"})

    # create [time x channel/voxel x wavelength/chromo] array for each trial type
    result = spatial_pattern * hrf_regs

    return result


def get_colors(
    activations: xr.DataArray,
    vertex_colors: np.array,
    log_scale: bool = False,
    max_scale: float = None,
):
    """Maps activations to colors for visualization.

    Args:
        activations (xr.DataArray): Activation values for each vertex.
        vertex_colors (np.array): Vertex color array of the brain mesh.
        log_scale (bool): Whether to map activations on a logarithmic scale.
        max_scale (float): Maximum value to scale the activations.

    Returns:
        np.array: New vertex color array with same shape as `vertex_colors`.
    """

    if not isinstance(activations, np.ndarray):
        activations = activations.pint.dequantify()
    # linear scale:
    if max_scale is None:
        max_scale = activations.max()
    activations = (activations / max_scale) * 255
    # map on a logarithmic scale to the range [0, 255]
    if log_scale:
        activations = np.log(activations + 1)
        activations = (activations / np.log(max_scale)) * 255
    activations = activations.astype(np.uint8)
    colors = np.zeros((vertex_colors.shape), dtype=np.uint8)
    colors[:, 3] = 255
    colors[:, 0] = 255
    activations = 255 - activations
    colors[:, 1] = activations
    colors[:, 2] = activations
    colors[colors < 0] = 0
    colors[colors > 255] = 255
    return colors


def plot_spatial_activation(
    spatial_img: xr.DataArray,
    brain: cdg.TrimeshSurface,
    seed: int = None,
    title: str = "",
    log_scale: bool = False,
):
    """Plots a spatial activation pattern on the brain surface.

    Args:
        spatial_img (xr.DataArray): Activation values for each vertex.
        brain (TrimeshSurface): Brain Surface with brain mesh.
        seed (int): Seed vertex for the activation pattern.
        title (str): Title for the plot.
        log_scale (bool): Whether to map activations on a logarithmic scale.

    Returns:
        None

    Initial Contributors:
        - Thomas Fischer | t.fischer.1@campus.tu-berlin.de | 2024

    """

    if seed is None:
        seed = spatial_img.argmax()
    vertices = brain.mesh.vertices
    center_brain = np.mean(vertices, axis=0)
    colors_blob = get_colors(
        spatial_img, brain.mesh.visual.vertex_colors, log_scale=log_scale
    )
    brain.mesh.visual.vertex_colors = colors_blob
    plt_pv = pv.Plotter()
    cedalion.plots.plot_surface(plt_pv, brain)
    plt_pv.camera.position = (vertices[seed] - center_brain) * 7 + center_brain
    plt_pv.camera.focal_point = vertices[seed]
    plt_pv.camera.up = [0, 0, 1]
    plt_pv.add_text(title, position="upper_edge", font_size=20)
    plt_pv.show()
