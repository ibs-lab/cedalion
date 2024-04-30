import numpy as np
import xarray as xr
import pandas as pd
import random
from scipy import signal
import pint
from cedalion import Quantity, units
import cedalion.dataclasses as cdc
import cedalion.typing as cdt


def generate_hrf(
    trange: list,
    dt: float,
    stim_dur: float,
    params_basis: list = [0.1000, 3.0000, 1.8000, 3.0000],
    scale: list = [10, -4],
):
    """Generates Hemodynamic Response Function (HRF) basis functions for different chromophores.

    This function calculates the HRF basis functions using gamma distributions. It supports
    adjusting the response scale for HbO and HbR using parameters provided in `paramsBasis`
    and `scale`.

    Args:
        trange (list): Relative start and end time of the resulting HRF (e.g., [-2, 5]).
        dt (float): Sampling period.
        stim_dur (float): Duration of the stimulus.
        params_basis (list of float): Parameters for tau and sigma for the modified gamma function
                                     for each chromophore. Expected to be a flat list where pairs
                                     represent [tau, sigma] for each chromophore.
        scale (list of float): Scaling factors for each chromophore, typically [HbO scale, HbR scale].

    Returns:
        xarray.DataArray: A DataArray object with dimensions "time" and "chromo", containing
                          the HRF basis functions for each chromophore.
    """

    nConc = len(params_basis) // 2
    if scale is None:
        scale = np.ones(nConc).tolist()
    if nConc != len(scale):
        raise ValueError(
            f"Length of `params_basis` ({len(params_basis)}) must be twice the length of `scale` ({len(scale)})"
        )

    nPre = int(np.round(trange[0] / dt))
    nPost_gamma = int(np.round(10 / dt))
    nPost_stim = int(np.round(trange[1] / dt))

    tHRF_gamma = np.linspace(nPre * dt, (nPost_gamma - 1) * dt, nPost_gamma - nPre)
    stimulus = np.zeros([nPost_stim, nConc])

    boxcar = np.zeros(len(tHRF_gamma))
    boxcar[: int(stim_dur / dt)] = 1

    for iConc in range(nConc):
        tau = params_basis[iConc * 2]
        sigma = params_basis[iConc * 2 + 1]

        gamma = (np.exp(1) * (tHRF_gamma - tau) ** 2 / sigma**2) * np.exp(
            -((tHRF_gamma - tau) ** 2) / sigma**2
        )
        gamma[tHRF_gamma < 0] = 0  # Set gamma values before time 0 to 0

        if tHRF_gamma[0] < tau:
            # Specifically zero out gamma values before tau
            gamma[: int(np.ceil((tau - tHRF_gamma[0]) / dt))] = 0

        convolved = signal.convolve(boxcar, gamma, mode="full")[:nPost_stim]
        normalized = convolved / np.max(abs(convolved)) * scale[iConc] * 1e-6
        stimulus[:, iConc] = normalized

    tHRF_stim = np.linspace(nPre * dt, (nPost_stim - 1) * dt, nPost_stim - nPre)

    tbasis = xr.DataArray(
        stimulus,
        dims=["time", "chromo"],
        coords={"time": tHRF_stim},
    ).T
    if nConc == 2:
        tbasis = tbasis.assign_coords(chromo=["HbO", "HbR"])
    tbasis = tbasis.assign_coords(samples=("time", np.arange(len(tHRF_stim))))
    tbasis = tbasis.pint.quantify("molar")

    return tbasis


@cdc.validate_schemas
def add_hrf_to_od(
    od: cdt.NDTimeSeries,
    hrf: xr.DataArray,
    channels: dict = {},
    min_interval: float = 5,
    max_interval: float = 10,
    num_stims: int = 15,
    order: str = "alternating",
):
    """Adds Hemodynamic Response Function (HRF) to optical density (OD) data in channel space.

    Adds the HRF onto the OD timeseries data at randomly selected intervals between stimuli.
    Stimuli can be added in an 'alternating' or 'random' order, and the inter-stimulus interval
    (ISI) is chosen randomly between the minimum and maximum allowed intervals.

    Args:
        od (xr.DataArray): OD timeseries data with dimensions ["channel", "wavelength", "time"].
        hrf (xr.DataArray): HRF timeseries with dimensions ["time", "wavelength"].
        channels (dict): Mapping of trial types to the channels where HRF should be added.
        min_interval (int): Minimum inter-stimulus interval in seconds.
        max_interval (int): Maximum inter-stimulus interval in seconds.
        num_stims (int): Number of stimuli to be added.
        order (str): Order of adding HRFs; 'alternating' or 'random'.

    Returns:
        Tuple of (xr.DataArray, pd.DataFrame): Updated OD data with added HRF and a DataFrame
        containing stimulus metadata.
    """

    stim_dur = hrf.time.values[-1] - hrf.time.values[0] + 1 / hrf.time.cd.sampling_rate
    od = od.transpose("channel", "wavelength", "time")
    current_time = -stim_dur
    onset_idxs = []
    onset_times = []
    onset_trial_types = []

    try:
        n_tpts_hrf = len(hrf["time"])
        hrf = hrf.pint.dequantify().values
    except:
        n_tpts_hrf = hrf.shape[0]
        hrf = np.reshape(hrf, [hrf.shape[1] // 2, 2, hrf.shape[0]])

    n_tpts_data = len(od["time"])
    od_w_hrf = od.pint.dequantify().copy()
    time = od_w_hrf["time"]

    channel_masks = {}

    # Default case: one trial type, add HRF to all channels
    if not channels:
        channels = {"Stim": od.channel.values.tolist()}

    # Create a mask of channels for each trial type
    for trial_type, channel_list in channels.items():
        channel_masks[trial_type] = np.where(np.isin(od.channel, channel_list))[0]

    order = order.lower()

    if order not in ["alternating", "random"]:
        raise ValueError(
            f"Invalid order '{order}'. Must be either 'alternating' or 'random'."
        )

    if order == "alternating":
        for stim in range(num_stims):
            for trial_type, channel_mask in channel_masks.items():
                interval = random.uniform(min_interval, max_interval)
                current_time += stim_dur + interval
                onset_idx = (np.abs(time - current_time)).argmin()

                if onset_idx + n_tpts_hrf > n_tpts_data:
                    break  # Stop if the stimulus goes past the data length
                    print("Stimulus goes past data length. Stopping loop...")

                onset_idxs.append(int(onset_idx))
                onset_times.append(current_time)
                onset_trial_types.append(trial_type)

                # Add the HRF at this onset index
                od_w_hrf[
                    channel_mask, :, int(onset_idx) : int(onset_idx + n_tpts_hrf)
                ] += hrf

    elif order == "random":
        stims_left = {trial_type: num_stims for trial_type in channel_masks.keys()}
        while any(stims_left.values()):
            trial_type, channel_mask = random.choices(
                list(channel_masks.items()),
                weights=[stims_left[tt] for tt in channel_masks],
            )[0]
            interval = random.uniform(min_interval, max_interval)
            current_time += stim_dur + interval
            onset_idx = (np.abs(time - current_time)).argmin()

            if onset_idx + n_tpts_hrf > n_tpts_data:
                break  # Stop if the stimulus goes past the data length
                print("Stimulus goes past data length. Stopping loop...")

            onset_idxs.append(int(onset_idx))
            onset_times.append(current_time)
            onset_trial_types.append(trial_type)

            # Add the HRF at this onset index
            od_w_hrf[
                channel_mask, :, int(onset_idx) : int(onset_idx + n_tpts_hrf)
            ] += hrf

            stims_left[trial_type] -= 1

    # Create DataFrame for stimulus metadata
    stim_df = pd.DataFrame(
        {
            "onset": onset_times,
            "duration": [stim_dur] * len(onset_times),
            "value": [1] * len(onset_times),
            "trial_type": onset_trial_types,
        }
    )

    return od_w_hrf.pint.quantify(), stim_df


def get_connected_vertices(seed, vertex_list, vertices, faces, dist_thresh=10):
    """Retrieves a connected blob of vertices centered at the seed vertex.

    This function finds all vertices within a given distance threshold of the seed vertex and
    explores all faces that contain a found vertex, adding all vertices from these faces
    to the blob to ensure connectivity.

    Args:
        seed (int): Index of the seed vertex.
        vertex_list (list): List of vertex indices to consider.
        vertices (np.array): Array of vertex coordinates.
        faces (list): List of tuples/lists, each containing indices of vertices that make up a face.
        dist_thresh (float, optional): Distance threshold for considering vertices to be connected.

    Returns:
        tuple: A tuple containing:
            - list: Unique vertex indices within the blob.
            - dict: Dictionary mapping each vertex index to a list of connected vertex indices.
    """

    blob_vertices = []
    connectivity_dict = {}

    for vertex in vertex_list:
        distance = np.linalg.norm(vertices[vertex, :] - vertices[seed, :])
        if distance < dist_thresh:
            blob_vertices.append(vertex)
            connected_vertices = []

            for face in faces:
                if vertex in face:
                    connected_vertices.extend(face)

            connected_vertices = list(np.unique(connected_vertices))
            connectivity_dict[str(vertex)] = connected_vertices

    blob_vertices = list(np.unique(blob_vertices))

    return blob_vertices, connectivity_dict


def diffusion_operator(
    seed, vertex_list, vertices, faces, n_iterations=15, n_vertices=20004
):
    """Executes a diffusion process starting from a seed vertex over a number of iterations.

    The function starts by finding a blob of connected vertices around the seed. It initializes the seed's
    amplitude to 1 and then iteratively sets the amplitude of each vertex in the blob to the average amplitude
    of its connected vertices. This process simulates the diffusion of values from the seed through the blob.

    Args:
        seed (int): Index of the seed vertex.
        vertex_list (list): List of vertex indices to consider.
        vertices (np.array): Array of vertex coordinates.
        faces (list): List of tuples/lists, each containing indices of vertices that make up a face.
        n_iterations (int, optional): Number of iterations the diffusion process should run. Defaults to 15.
        n_vertices (int, optional): Total number of vertices. Defaults to 20004.

    Returns:
        tuple: A tuple containing:
            - list: Vertex indices within the blob.
            - np.array: Array representing the diffusion amplitudes across all vertices.
    """
    blob_vertices, connectivity_dict = get_connected_vertices(
        seed, vertex_list, vertices, faces, dist_thresh=10
    )

    diffusion_image = np.zeros(n_vertices)

    for _ in range(n_iterations):
        diffusion_image[seed] = 1

        for vertex in blob_vertices:
            if vertex != seed:
                connected_vertices = connectivity_dict[str(vertex)]
                diffusion_image[vertex] = np.mean(diffusion_image[connected_vertices])

    return blob_vertices, diffusion_image


def add_hrf_to_vertices(
    vertex_list, hrf_basis: xr.DataArray, scale=None, num_vertices=20004
):
    """Adds hemodynamic response functions (HRF) for HbO and HbR to specified vertices.

    This function adds a given HRF to selected vertices, optionally scaling the response by a
    provided amplitude scale.

    Args:
        vertex_list (list): List of vertex indices to which HRFs are added.
        hrf_basis (xarray.DataArray): Dataset containing HRF time series for HbO and HbR.
        scale (np.array, optional): Array of scale factors of shape (num_vertices, 1) to scale the amplitude of HRFs.
        num_vertices (int, optional): Total number of vertices in the image space. Defaults to 20004.

    Returns:
        np.array: Combined image of HbO and HbR responses across all vertices for all time points.
    """
    num_time_points = len(hrf_basis["time"])
    hbo_real_image = np.zeros([num_time_points, num_vertices])
    hbr_real_image = np.zeros([num_time_points, num_vertices])

    hbo = hrf_basis.sel({"chromo": "HbO"})
    hbr = hrf_basis.sel({"chromo": "HbR"})
    hbo_real_image[:, vertex_list] = (
        hbo.pint.dequantify().values * np.ones([len(vertex_list), num_time_points])
    ).T
    hbr_real_image[:, vertex_list] = (
        hbr.pint.dequantify().values * np.ones([len(vertex_list), num_time_points])
    ).T

    if scale is not None:
        hbo_real_image *= scale
        hbr_real_image *= scale

    hrf_real_image = np.hstack([hbo_real_image, hbr_real_image])

    return hrf_real_image
