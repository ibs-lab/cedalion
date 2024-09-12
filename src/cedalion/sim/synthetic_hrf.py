import random

import numpy as np
import pandas as pd
import pyvista as pv
import scipy.stats as stats
import xarray as xr
from scipy import signal

import cedalion.dataclasses as cdc
import cedalion.dataclasses.geometry as cdg
import cedalion.imagereco.forward_model as cfm
import cedalion.plots
import cedalion.typing as cdt
import cedalion.xrutils as xrutils
from cedalion import Quantity, units


def generate_hrf(
    time_axis: xr.DataArray,
    stim_dur: Quantity = 10 * units.seconds,
    params_basis: list = [0.1000, 3.0000, 1.8000, 3.0000],
    scale: list = [10 * units.micromolar, -4 * units.micromolar],
):
    """Generates HRF basis functions for different chromophores.

    This function calculates the HRF basis functions using gamma distributions. It
    supports adjusting the response scale for HbO and HbR using parameters provided in
    `params_basis` and `scale`.

    Args:
        time_axis: The time axis for the resulting HRF.
        stim_dur: Duration of the stimulus.
        params_basis (list of float): Parameters for tau and sigma for the modified
            gamma function for each chromophore. Expected to be a flat list where pairs
            represent [tau, sigma] for each chromophore.
        scale (list of float): Scaling factors for each chromophore, typically
            [HbO scale, HbR scale].

    Returns:
        xarray.DataArray: A DataArray object with dimensions "time" and "chromo",
            containing the HRF basis functions for each chromophore.
    """

    time_axis = time_axis - time_axis[0]
    stim_dur = (stim_dur / units.seconds).to_base_units().magnitude
    scale[0] = (scale[0] / units.molar).to_base_units().magnitude
    scale[1] = (scale[1] / units.molar).to_base_units().magnitude

    n_conc = len(params_basis) // 2
    if scale is None:
        scale = np.ones(n_conc).tolist()
    if n_conc != len(scale):
        raise ValueError(
            f"Length of `params_basis` ({len(params_basis)}) must be twice the length"
            f" of `scale` ({len(scale)})"
        )
    if stim_dur > time_axis[-1]:
        raise Warning(
            "Stimulus duration is longer than the time axis. The stimulus will be "
            "cut off."
        )

    tHRF_gamma = time_axis[time_axis <= stim_dur]
    boxcar = xr.DataArray(
        np.zeros(len(tHRF_gamma)), dims=["time"], coords={"time": tHRF_gamma}
    )
    boxcar.loc[boxcar["time"] <= stim_dur] = 1
    boxcar.loc[boxcar["time"] < 0] = 0

    stimulus = np.zeros((len(time_axis), n_conc))

    for iConc in range(n_conc):
        tau = params_basis[iConc * 2]
        sigma = params_basis[iConc * 2 + 1]

        gamma = (np.exp(1) * (tHRF_gamma - tau) ** 2 / sigma**2) * np.exp(
            -((tHRF_gamma - tau) ** 2) / sigma**2
        )
        gamma = xr.DataArray(gamma, dims=["time"], coords={"time": tHRF_gamma})
        gamma = gamma.where(gamma["time"] >= 0, 0)

        if tHRF_gamma[0] < tau:
            gamma = gamma.where(gamma["time"] >= tau, 0)

        convolved = signal.convolve(boxcar, gamma, mode="full")

        convolved = convolved[: len(time_axis)]
        normalized = convolved / np.max(np.abs(convolved)) * scale[iConc]
        stimulus[: convolved.size, iConc] = normalized[: len(time_axis)]

    tbasis = xr.DataArray(
        stimulus,
        dims=["time", "chromo"],
        coords={"time": time_axis, "chromo": ["HbO", "HbR"][:n_conc]},
    ).T
    tbasis = tbasis.assign_coords(samples=("time", np.arange(len(time_axis))))
    tbasis.pint.units = cedalion.units.molar

    return tbasis


def build_blob(
    head_model: cfm.TwoSurfaceHeadModel,
    landmark: str,
    scale: Quantity = 1 * units.cm,
    m: float = 10.0,
):
    """Generates a blob of activity at a seed landmark.

    This function generates a blob of activity on the brain surface.
    The blob is centered at the vertex closest to the seed landmark.

    Args:
        head_model (cfm.TwoSurfaceHeadModel): Head model with brain and scalp surfaces.
        landmark (str): Name of the seed landmark.
        scale (Quantity): Scale of the blob.
        m (float): Geodesic distance parameter. Larger values of m will smooth &
            regularize the distance computation. Smaller values of m will roughen and
            will usually increase error in the distance computation.

    Returns:
        xr.DataArray: Blob image with activation values for each vertex.
    """

    scale = (scale / head_model.brain.units).to_base_units().magnitude

    seed_lm = head_model.landmarks.sel(label=landmark).pint.dequantify()
    seed_vertex = head_model.brain.mesh.kdtree.query(seed_lm)[1]

    cortex_surface = cdg.PycortexSurface.from_trimeshsurface(head_model.brain)

    distances_from_seed = cortex_surface.geodesic_distance([seed_vertex], m=m)
    # distances can be distord due to mesh decimation or unsuitable m value

    norm_pdf = stats.norm(scale=scale).pdf

    blob_img = norm_pdf(distances_from_seed)
    blob_img = blob_img / np.max(blob_img)
    blob_img = xr.DataArray(blob_img, dims=["vertex"])

    return blob_img


def hrfs_from_image_reco(
    blob: xr.DataArray,
    hrf_model: xr.DataArray,
    Adot: xr.DataArray,
):
    """Maps an activation blob on the brain to HRFs in channel space.

    Args:
        blob (xr.DataArray): Activation values for each vertex.
        hrf_model (xr.DataArray): HRF model for HbO and HbR.
        Adot (xr.DataArray): Sensitivity matrix for the forward model.

    Returns:
        cdt.NDTimeseries: HRFs in channel space.
    """

    hrf_model = hrf_model.pint.to(units.molar)
    hrf_model = hrf_model.pint.dequantify()

    n_channels = Adot.channel.size
    n_v_brain = Adot.sel(vertex=Adot.is_brain).vertex.size

    Adot_stacked = cfm.ForwardModel.compute_stacked_sensitivity(Adot)
    # Adot should have units
    Adot_is_brain_stack = xr.concat([Adot.is_brain, Adot.is_brain], dim="vertex")
    Adot_is_brain_stack = Adot_is_brain_stack.rename({"vertex": "flat_vertex"})
    Adot_brain_stacked = Adot_stacked.sel(flat_vertex=Adot_is_brain_stack)

    HRF_image = add_hrf_to_vertices(hrf_model, n_v_brain, scale=blob)
    HRF_chan = Adot_brain_stacked @ HRF_image

    HRF_chan = np.stack([HRF_chan[:n_channels], HRF_chan[n_channels:]], axis=1)
    HRF_chan = xr.DataArray(
        HRF_chan,
        coords=[Adot.channel, Adot.wavelength, hrf_model.time],
        dims=["channel", "wavelength", "time"],
    ).assign_coords(samples=("time", np.arange(len(hrf_model.time))))

    return HRF_chan


def add_hrf_to_vertices(
    hrf_basis: xr.DataArray, num_vertices: int, scale: np.array = None
):
    """Adds hemodynamic response functions (HRF) for HbO and HbR to specified vertices.

    This function applies temporal HRF profiles to vertices, optionally scaling the
    response by a provided amplitude scale. It generates separate images for HbO and HbR
    and then combines them.

    Args:
        hrf_basis (xarray.DataArray): Dataset containing HRF time series for
            HbO and HbR.
        num_vertices (int): Total number of vertices in the image space.
        scale (np.array, optional): Array of scale factors of shape (num_vertices) to
            scale the amplitude of HRFs.

    Returns:
        xr.DataArray: Combined image of HbO and HbR responses across all vertices for
            all time points.
    """

    unit = hrf_basis.pint.units
    num_time_points = len(hrf_basis.time)
    hbo = hrf_basis.sel({"chromo": "HbO"})
    hbr = hrf_basis.sel({"chromo": "HbR"})
    hbo_real_image = (
        hbo.pint.dequantify().values * np.ones([num_vertices, num_time_points])
    ).T
    hbr_real_image = (
        hbr.pint.dequantify().values * np.ones([num_vertices, num_time_points])
    ).T

    if scale is not None:
        scale = scale.pint.dequantify()
        scale = scale.values
        scale = scale / np.max(scale)
        hbo_real_image *= scale
        hbr_real_image *= scale

    hrf_real_image = np.hstack([hbo_real_image, hbr_real_image])

    hrf_real_image = xr.DataArray(
        hrf_real_image,
        dims=["time", "flat_vertex"],
        coords={
            "time": hrf_basis.time,
            "chromo": ("flat_vertex", ["HbO"] * num_vertices + ["HbR"] * num_vertices),
        },
    )
    hrf_real_image.set_xindex("chromo")
    hrf_real_image.pint.units = unit

    return hrf_real_image


def build_stim_df(
    num_stims: int,
    stim_dur: Quantity = 10 * units.seconds,
    trial_types: list = ["Stim"],
    min_interval: Quantity = 5 * units.seconds,
    max_interval: Quantity = 10 * units.seconds,
    order: str = "alternating",
):
    """Generates a DataFrame for stimulus metadata based on provided parameters.

    Stimuli can be added in an 'alternating' or 'random' order, and the inter-stimulus
    interval (ISI) is chosen randomly between the minimum and maximum allowed intervals.

    Args:
        num_stims (int): Number of stimuli to be added for each trial type.
        stim_dur (int): Duration of the stimulus in seconds.
        trial_types (list): List of trial types for the stimuli.
        min_interval (int): Minimum inter-stimulus interval in seconds.
        max_interval (int): Maximum inter-stimulus interval in seconds.
        order (str): Order of adding Stims; 'alternating' or 'random'.

    Returns:
        pd.DataFrame: DataFrame containing stimulus metadata.
    """

    stim_dur = (stim_dur / units.seconds).to_base_units().magnitude
    min_interval = (min_interval / units.seconds).to_base_units().magnitude
    max_interval = (max_interval / units.seconds).to_base_units().magnitude

    current_time = -stim_dur
    onset_times = []
    onset_trial_types = []

    trial_types = [str(tt) for tt in trial_types]
    order = order.lower()

    if order not in ["alternating", "random"]:
        raise ValueError(
            f"Invalid order '{order}'. Must be either 'alternating' or 'random'."
        )

    if order == "alternating":
        for stim in range(num_stims):
            for trial_type in trial_types:
                interval = round(random.uniform(min_interval, max_interval), 2)
                current_time += stim_dur + interval
                onset_times.append(current_time)
                onset_trial_types.append(trial_type)

    elif order == "random":
        stims_left = {trial_type: num_stims for trial_type in trial_types}
        while any(stims_left.values()):
            trial_type = random.choices(
                list(trial_types),
                weights=[stims_left[tt] for tt in trial_types],
            )[0]
            interval = round(random.uniform(min_interval, max_interval), 1)
            current_time += stim_dur + interval
            onset_times.append(current_time)
            onset_trial_types.append(trial_type)
            stims_left[trial_type] -= 1

    stim_df = pd.DataFrame(
        {
            "onset": onset_times,
            "duration": [stim_dur] * len(onset_times),
            "value": [1] * len(onset_times),
            "trial_type": onset_trial_types,
        }
    )

    return stim_df


@cdc.validate_schemas
def add_hrf_to_od(od: cdt.NDTimeSeries, hrfs: cdt.NDTimeSeries, stim_df: pd.DataFrame):
    """Adds Hemodynamic Response Functions (HRFs) to optical density (OD) data.

    The timing of the HRFs is based on the provided stimulus dataframe (stim_df).

    Args:
        od (cdt.NDTimeSeries): OD timeseries data with dimensions
            ["channel", "wavelength", "time"].
        hrfs (cdt.NDTimeSeries): HRFs in channel space with dimensions
            ["channel", "wavelength", "time"] + maybe ["trial_type"].
        stim_df (pd.DataFrame): DataFrame containing stimulus metadata.

    Returns:
        cdt.NDTimeSeries: OD data with HRFs added based on the stimulus dataframe.
    """

    if "trial_type" not in hrfs.dims:
        hrfs = hrfs.expand_dims("trial_type").assign_coords(trial_type=["Stim"])

    od = od.transpose("channel", "wavelength", "time")
    hrfs = hrfs.transpose("channel", "wavelength", "time", "trial_type")

    units_od = od.pint.units

    hrfs = hrfs.pint.dequantify()
    od_w_hrf = od.pint.dequantify().copy()

    n_tpts_hrf = len(hrfs.time)
    n_tpts_data = len(od.time)
    time_axis = od.time

    for _, stim_info in stim_df.iterrows():
        current_time = stim_info["onset"]
        trial_type = stim_info["trial_type"]

        onset_idx = (np.abs(time_axis - current_time)).argmin("time")

        # Stop if the stimulus goes past the data length
        if onset_idx + n_tpts_hrf > n_tpts_data:
            print(
                f"Stimulus goes past data length. Onset time: {current_time}. "
                "Stopping loop..."
            )
            break

        # Add the HRF at this onset index
        od_w_hrf[:, :, int(onset_idx) : int(onset_idx + n_tpts_hrf)] += hrfs.sel(
            {"trial_type": trial_type}
        ).values

    return od_w_hrf.pint.quantify(units_od)


@cdc.validate_schemas
def hrf_to_long_channels(
    hrf_model: xr.DataArray,
    y: cdt.NDTimeSeries,
    geo3d: xr.DataArray,
    ss_tresh: Quantity = 1.5 * units.cm,
):
    """Add HRFs to optical density (OD) data in channel space.

    Broadcasts the HRF model to long channels based on the source-detector distances.
    Short channel hrfs are filled with zeros.

    Args:
        hrf_model (xr.DataArray): HRF model with dimensions ["time", "wavelength"].
        y (cdt.NDTimeSeries): Raw amp / OD / Chromo timeseries data with dimensions
            ["channel", "time"].
        geo3d (xr.DataArray): 3D coordinates of sources and detectors.
        ss_tresh (Quantity): Threshold for short/long channels.

    Returns:
        xr.DataArray: HRFs in channel space with dimensions
            ["channel", "time", "wavelength"].
    """

    # Calculate source-detector distances for each channel
    dists = (
        xrutils.norm(geo3d.loc[y.source] - geo3d.loc[y.detector], dim="pos")
        .pint.to("mm")
        .round(2)
    )

    # Identify long channels
    long_channels = dists.channel[dists > ss_tresh]

    hrf_long_chans = xr.DataArray(
        np.zeros((y.channel.size, hrf_model.time.size, hrf_model.wavelength.size)),
        coords=[y.channel, hrf_model.time, hrf_model.wavelength],
        dims=["channel", "time", "wavelength"],
    ).pint.quantify(hrf_model.pint.units)

    hrf_long_chans.loc[long_channels] = hrf_model

    return hrf_long_chans


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
        activations = (activations / max_scale) * 255
    colors = np.zeros((vertex_colors.shape))
    colors[:, 3] = 255
    colors[:, 0] = 255
    activations = 255 - activations
    colors[:, 1] = activations
    colors[:, 2] = activations
    colors[colors < 0] = 0
    colors[colors > 255] = 255
    return colors


def plot_blob(
    blob_img: xr.DataArray,
    brain,
    seed: int = None,
    title: str = "",
    log_scale: bool = False,
):
    """Plots a blob of activity on the brain.

    Args:
        blob_img (xr.DataArray): Activation values for each vertex.
        brain (TrimeshSurface): Brain Surface with brain mesh.
        seed (int): Seed vertex for the blob.
        title (str): Title for the plot.
        log_scale (bool): Whether to map activations on a logarithmic scale.

    Returns:
        None
    """

    if seed is None:
        seed = blob_img.argmax()
    vertices = brain.mesh.vertices
    center_brain = np.mean(vertices, axis=0)
    colors_blob = get_colors(blob_img, brain.mesh.visual.vertex_colors)
    brain.mesh.visual.vertex_colors = colors_blob
    plt_pv = pv.Plotter()
    cedalion.plots.plot_surface(plt_pv, brain)
    plt_pv.camera.position = (vertices[seed] - center_brain) * 7 + center_brain
    plt_pv.camera.focal_point = vertices[seed]
    plt_pv.camera.up = [0, 0, 1]
    plt_pv.add_text(title, position="upper_edge", font_size=20)
    plt_pv.show()
