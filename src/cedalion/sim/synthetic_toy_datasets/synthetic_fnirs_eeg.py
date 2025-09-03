"""Simulate bimodal data as a toy model for concurrent EEG–fNIRS data.

Follows an approach inspired by Dähne et al., 2014.

Reference: https://doi.org/10.1016/j.neuroimage.2014.01.014
"""

import yaml
from argparse import Namespace

import random
import math
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from datetime import datetime
from scipy.signal import hilbert, butter, lfilter  # To compute signal envelope


class BimodalToyDataSimulation:
    """Simulate bimodal data as a toy model for concurrent EEG-fNIRS data.

    This class creates coupled (bimodal) synthetic signals representing an
    "X" modality (e.g., EEG) and a "Y" modality (e.g., fNIRS). It provides
    utilities for simulating sources, generating spatial mixing patterns,
    projecting to channels via a simple forward model, computing power, and
    basic visualization.

    Toy data description:
        EEG (``x``) and fNIRS (``y``) recordings are generated from a
        pseudo-random linear mixing forward model. Sources ``s_x`` and ``s_y``
        are divided into background sources (independent between modalities)
        and target sources (co-modulated across modalities). Each EEG
        background source is constructed from a random oscillatory signal in a
        chosen frequency band multiplied by a slow-varying random
        amplitude-modulation function. This amplitude modulation acts as the
        envelope of ``s_x`` and provides an estimate of its bandpower
        timecourse. fNIRS background sources are generated directly from
        slow-varying amplitude-modulation functions in the same way.

        Target sources are built similarly, except that the same envelope
        modulating ``s_x`` is also used for the corresponding fNIRS source
        ``s_y``. This coupling may be delayed by a time-lag parameter ``dT``
        to simulate realistic physiological delays between modalities. fNIRS
        sources and recordings are then downsampled to epoch intervals of
        length ``T_epoch``.

        Background mixing matrices ``A_x`` and ``A_y`` are drawn from normal
        distributions, while target mixing matrices use Gaussian radial basis
        functions (RBFs) centered on shared positions, plus white noise, to
        create spatial patterns with local structure.

        Signal-to-noise ratio (SNR) is controlled by the parameter ``gamma``,
        which weights target versus background contributions in channel space.
        The relationship with decibel units is given by
        ``SNR [dB] = 20 * log10(gamma)``.

        EEG channel recordings and their bandpower timecourses are available
        via the attributes ``x`` (high-rate channels) and ``x_power`` (epoch
        downsampled). The downsampled ``x_power`` is aligned with fNIRS
        recordings ``y``. Target sources are accessible as ``sx_t`` and
        ``sy_t``; thei powerband timecourse of the former is available 
        as ``sx_power``.


    Args:
        config (str | dict): Path to a YAML config file or a dictionary
            containing simulation parameters. See :func:`generate_args` for
            derived fields.
        seed (int | None): Random seed for reproducibility. If ``None``,
            uses the current UNIX timestamp.
        mixing_type (str): Type of mixing to generate for target sources.
            ``'structured'`` assigns localized RBF-like patterns; any
            other value leaves patterns purely random.

    Attributes:
        args (argparse.Namespace): Simulation parameters.
        seed (int): Random seed used for reproducibility.
        sx_montage (xr.DataArray): Montage for X channels.
        sy_montage (xr.DataArray): Montage for Y channels.
        s_labels (list[str]): Labels for target sources.
        s_target_positions (xr.DataArray): 2D positions of target sources.
        sx_t (xr.DataArray): Target sources for X modality, dims ``('source', 'time')``.
        sy_t (xr.DataArray): Target sources for Y modality, dims ``('source', 'time')``, sampled over epochs.
        sx_ba (xr.DataArray): Background sources for X modality, dims ``('source', 'time')``.
        sy_ba (xr.DataArray): Background sources for Y modality, dims ``('source', 'time')``, sampled over epochs.
        sx_power (xr.DataArray): Power of target sources for X modality, dims ``('source', 'time')``, sampled over epochs.
        Ax (xr.DataArray): Mixing matrix for X modality, dims ``('channel', 'source')``.
        ax_t (xr.DataArray): Mixing patterns for target sources in X, dims ``('channel', 'source')``.
        ax_ba (xr.DataArray): Mixing patterns for background sources in X, dims ``('channel', 'source')``.
        Ay (xr.DataArray): Mixing matrix for Y modality, dims ``('channel', 'source')``.
        ay_t (xr.DataArray): Mixing patterns for target sources in Y, dims ``('channel', 'source')``.
        ay_ba (xr.DataArray): Mixing patterns for background sources in Y, dims ``('channel', 'source')``.
        x (xr.DataArray): Observed channels for X modality, dims ``('channel', 'time')``.
        x_power (xr.DataArray): Power of observed channels for X modality, dims ``('channel', 'time')``, sampled over epochs.
        y (xr.DataArray): Observed channels for Y modality, dims ``('channel', 'time')``, sampled over epochs.

    """

    def __init__(self, config, seed=None, mixing_type='structured'):

        # Parameters
        self.args = generate_args(config)
        self.seed = seed if seed is not None else int(datetime.now().timestamp())
        set_seed(self.seed)  # Set seed for reproducibility

        # Define montage as 2D coordinates of channels
        self.x_montage = self.generate_montage(self.args.Nx, channel_label='X')
        self.y_montage = self.generate_montage(self.args.Ny, channel_label='Y')

        # Simulate location of target sources
        self.s_labels = ['S' + str(i + 1) for i in range(self.args.Ns_all)]
        self.s_target_positions = np.random.rand(self.args.Ns_target, 2)
        self.s_target_positions = xr.DataArray(
            self.s_target_positions,
            dims=['source', 'dim'],
            coords={'source': self.s_labels[: self.args.Ns_target], 'dim': ['x', 'y']},
        )

        # Define coordinates
        self.time_x = np.linspace(0, self.args.T, self.args.Nt)
        self.time_y = np.linspace(0, self.args.T, self.args.Ne)

        # Simulate target and background sources
        self.sx_t, self.sy_t, self.sx_ba, self.sy_ba = self.simulate_sources()

        # Calculcate source power
        sx_power = np.abs(hilbert(self.sx_t))
        sx_power = np.concatenate(
            [(split_epochs(sxp, self.args.e_len).mean(axis=1)).reshape(1, -1) for sxp in sx_power]
        )
        self.sx_power = xr.DataArray(
            sx_power,
            dims=['source', 'time'],
            coords={'time': self.time_y, 'source': self.s_labels[: self.args.Ns_target]},
        )

        # Generate random mixing matrix and spatial patterns
        self.Ax, self.ax_t, self.ax_ba, self.Ay, self.ay_t, self.ay_ba = self.generate_patterns(
            mixing_type
        )

        # Get channels via forward model
        self.x_t, self.x_noise = self.forward_model(self.ax_t, self.ax_ba, self.sx_t, self.sx_ba)
        self.x, self.x_power = self.get_channels(
            x_t=self.x_t, x_noise=self.x_noise, gamma=self.args.gamma, calculate_power=True
        )

        self.y_t, self.y_noise = self.forward_model(self.ay_t, self.ay_ba, self.sy_t, self.sy_ba)
        self.y = self.get_channels(self.y_t, self.y_noise, self.args.gamma)

    def preprocess_data(self, train_test_split=0.8):
        """Normalize and split the simulated data into train/test subsets.

        Args:
            train_test_split (float): Fraction of the time axis to use for the
                training split (``0 < train_test_split < 1``). The remainder is
                used for testing.

        Returns:
            dict: A dictionary with keys:
                - ``'x_train'`` (xr.DataArray): X channels (train).
                - ``'x_test'`` (xr.DataArray): X channels (test).
                - ``'x_power_train'`` (xr.DataArray): X power (train).
                - ``'x_power_test'`` (xr.DataArray): X power (test).
                - ``'y_train'`` (xr.DataArray): Y channels (train).
                - ``'y_test'`` (xr.DataArray): Y channels (test).
                - ``'sx'`` (xr.DataArray): Target X sources, test portion only.
                - ``'sx_power'`` (xr.DataArray): Target X source power, test only.
                - ``'sy'`` (xr.DataArray): Target Y sources, test portion only.
        """

        # Normalize and store variables for later
        x = standardize(self.x)
        x_power = standardize(self.x_power)
        y = standardize(self.y)
        sx = standardize(self.sx_t)
        sx_power = standardize(self.sx_power)
        sy = standardize(self.sy_t)

        # Split into train and test sets
        T_train = self.args.T * train_test_split
        x_train, x_test = x.sel(time=slice(0, T_train)), x.sel(time=slice(T_train, None))
        x_power_train, x_power_test = x_power.sel(time=slice(0, T_train)), x_power.sel(
            time=slice(T_train, None)
        )
        y_train, y_test = y.sel(time=slice(0, T_train)), y.sel(time=slice(T_train, None))

        # Restrict to test set for the source signals
        sx = sx.sel(time=slice(T_train, None))
        sx_power = sx_power.sel(time=slice(T_train, None))
        sy = sy.sel(time=slice(T_train, None))

        # Wrap into a dictionary for easy access
        preprocess_data_dict = {
            'x_train': x_train,
            'x_test': x_test,
            'x_power_train': x_power_train,
            'x_power_test': x_power_test,
            'y_train': y_train,
            'y_test': y_test,
            'sx': sx,
            'sx_power': sx_power,
            'sy': sy,
        }

        return preprocess_data_dict

    @staticmethod
    def generate_montage(Nc, channel_label):
        """Create a symmetric 2D montage of channels within ``[0, 1] x [0, 1]``.

        The function builds the smallest grid with at least ``Nc`` points,
        ranks points by distance to the center, and selects the closest
        ``Nc`` points for an aesthetically symmetric montage.

        Args:
            Nc (int): Number of channels to place.
            channel_label (str): Prefix used to name channels (e.g., ``'X'`` or
                ``'Y'``). Channels are labeled as ``'<label><idx>'`` starting
                at 1.

        Returns:
            xr.DataArray: Array of shape ``(channel, dim)`` with coordinates
            ``channel=[<label>1, ..., <label>Nc]`` and ``dim=['x','y']``
            containing the 2D positions in the unit square.
        """

        # Determine grid size
        n_cols = math.ceil(math.sqrt(Nc))
        n_rows = math.ceil(Nc / n_cols)

        # Generate grid coordinates
        xs = np.linspace(0, 1, n_cols)
        ys = np.linspace(0, 1, n_rows)
        X, Y = np.meshgrid(xs, ys)
        grid = np.stack([X.ravel(), Y.ravel()], axis=1)

        # Compute distance to center
        center = np.array([0.5, 0.5])
        dists = np.linalg.norm(grid - center, axis=1)

        # Select Nc points closest to center for symmetry
        idx = np.argsort(dists)[:Nc]
        positions = grid[idx]

        # Create xarray DataArray with coordinates
        channels = [channel_label + str(i + 1) for i in range(Nc)]
        montage = xr.DataArray(
            positions, 
            dims=['channel', 'dim'], 
            coords={'channel': channels, 
                    'dim': ['x', 'y']}
        )

        return montage

    def simulate_sources(self):
        """Simulate target and background sources for both modalities.

        Target sources are constructed from band-limited oscillations with
        amplitude modulation and a modality-dependent temporal shift. Background
        sources are independent amplitude-modulated oscillations.

        Args:
            None

        Returns:
            tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
                ``(sx_t, sy_t, sx_ba, sy_ba)`` where
                - ``sx_t``: target sources for X, dims ``('source','time')`` of
                  length ``(Ns_target, Nt)``.
                - ``sy_t``: target sources for Y, dims ``('source','time')`` of
                  length ``(Ns_target, Ne)`` (epoch-averaged amplitude).
                - ``sx_ba``: background sources for X, dims
                  ``('source','time')`` of length ``(Ns_ba, Nt)``.
                - ``sy_ba``: background sources for Y, dims
                  ``('source','time')`` of length ``(Ns_ba, Ne)``.
        """

        print('Simulating sources...')

        # Target sources

        # Define extended time parameters using time-delay
        Nt_w_shift = self.args.Nt + self.args.NdT
        T_w_shift = self.args.T + self.args.dT

        # Simulate target sources with time-delay
        sx_t = np.zeros((self.args.Ns_target, self.args.Nt))
        sy_t = np.zeros((self.args.Ns_target, self.args.Ne))
        for i in range(self.args.Ns_target):
            # Simulate extended source
            amplitude_t = self.simulate_amplitude(Nt=Nt_w_shift)
            s_extended = self.simulate_random_source(T=T_w_shift, Nt=Nt_w_shift) * amplitude_t

            # Simulate time-shifted X and Y target sources
            sx_t[i] = s_extended[self.args.NdT:]
            if self.args.dT > 0:
                sy_t[i] = standardize(
                    split_epochs(amplitude_t[: -self.args.NdT], self.args.e_len).mean(axis=1).reshape(1, -1)
                )
            elif self.args.dT == 0:
                sy_t[i] = standardize(
                    split_epochs(amplitude_t, self.args.e_len).mean(axis=1).reshape(1, -1)
                )
            else:
                raise ValueError("dT must be non-negative.")

        # Invert sy source
        if self.args.invert_sy:
            sy_t = 1 - sy_t

        # Background sources
        sx_ba = np.concatenate(
            [self.simulate_random_source().reshape(1, -1) * self.simulate_amplitude() for j in range(self.args.Ns_ba)]
        )
        sy_ba = np.concatenate(
            [
                standardize(split_epochs(self.simulate_amplitude(), self.args.e_len).mean(axis=1).reshape(1, -1))
                for j in range(self.args.Ns_ba)
            ]
        )

        print('Finished')

        # Bring to xarray format
        sx_t = xr.DataArray(
            sx_t,
            dims=['source', 'time'],
            coords={'source': self.s_labels[: self.args.Ns_target], 'time': self.time_x},
        )
        sy_t = xr.DataArray(
            sy_t,
            dims=['source', 'time'],
            coords={'source': self.s_labels[: self.args.Ns_target], 'time': self.time_y},
        )
        sx_ba = xr.DataArray(
            sx_ba,
            dims=['source', 'time'],
            coords={'source': self.s_labels[self.args.Ns_target :], 'time': self.time_x},
        )
        sy_ba = xr.DataArray(
            sy_ba,
            dims=['source', 'time'],
            coords={'source': self.s_labels[self.args.Ns_target :], 'time': self.time_y},
        )

        return sx_t, sy_t, sx_ba, sy_ba

    def simulate_random_source(self, T=None, Nt=None):
        """Simulate a single random oscillatory source within ``(f_min, f_max)``.

        The source is synthesized in the frequency domain with unit amplitude
        and random phases in the desired band, then transformed back to time
        via an inverse FFT and envelope-normalized to unity.

        Args:
            T (float | None): Total duration in seconds. Defaults to
                ``self.args.T`` when ``None``.
            Nt (int | None): Number of time samples. Defaults to
                ``self.args.Nt`` when ``None``.

        Returns:
            np.ndarray: Time-domain signal of shape ``(Nt,)`` with unit
            envelope (via analytic signal magnitude normalization).
        """

        # Define optional parameters
        T = self.args.T if T is None else T
        Nt = self.args.Nt if Nt is None else Nt

        # For indexing FT
        f_min_ndx = int(self.args.f_min * T)
        f_max_ndx = int(self.args.f_max * T)

        # Define signal in frequency-domain (unit amplitude and random phases)
        Fs = np.zeros(Nt, dtype=complex)
        Fs[f_min_ndx:f_max_ndx] = 1 * np.e ** (1j * np.random.uniform(0, 2 * np.pi, (f_max_ndx - f_min_ndx)))

        # Get temporal-domain signal from inverse FT
        s = np.fft.ifft(Fs).real
        s_env = np.abs(hilbert(s))  # Signal envelope
        s /= s_env  # Normalize envelope to 1

        return s

    def simulate_amplitude(self, Nt=None):
        """Simulate an amplitude modulation signal from low-pass noise.

        Args:
            Nt (int | None): Number of samples to generate. Defaults to
                ``self.args.Nt`` when ``None``.

        Returns:
            np.ndarray: Positive amplitude modulation of shape ``(Nt,)`` scaled
            to ``[0, 1]``.
        """

        Nt = self.args.Nt if Nt is None else Nt

        noise = np.random.normal(0, 1, Nt)
        amplitude = butter_lowpass_filter(noise, cut=0.5, fs=self.args.rate, order=4)
        amplitude += np.abs(amplitude.min()) * 1.1  # Add offset to make it positive
        amplitude /= amplitude.max()

        return amplitude

    def generate_patterns(self, mixing_type='structured'):
        """Generate mixing matrices and split them into target/background.

        Args:
            mixing_type (str): If ``'structured'``, assign localized RBF-like
                patterns to target sources based on distances from true source
                positions to channel locations; otherwise leave as random.

        Returns:
            tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
                ``(Ax, ax_t, ax_ba, Ay, ay_t, ay_ba)`` where each is an
                ``xr.DataArray`` with dims ``('channel','source')``. ``*_t``
                contains only the first ``Ns_target`` sources and ``*_ba`` the
                background sources.
        """

        # Generate random mixing matrix with normal distribution
        Ax = np.random.normal(0, 1, (self.args.Nx, self.args.Ns_all))
        Ay = np.random.normal(0, 1, (self.args.Ny, self.args.Ns_all))

        # Assign localized spatial structure to target sources
        if mixing_type == 'structured':
            for k in range(self.args.Ns_target):
                # Compute squared distances between channels and this center
                dist2x = np.sum((self.x_montage.data - self.s_target_positions[k].data) ** 2, axis=1)
                dist2y = np.sum((self.y_montage.data - self.s_target_positions[k].data) ** 2, axis=1)

                # RBF pattern and amplitude
                alphax = 1
                alphay = 1
                Ax[:, k] = alphax * np.exp(-dist2x / (2 * self.args.ellx ** 2))
                Ay[:, k] = alphay * np.exp(-dist2y / (2 * self.args.elly ** 2))
                # Add noise
                Ax[:, k] += self.args.sigma_noise * np.random.randn(self.args.Nx)
                Ay[:, k] += self.args.sigma_noise * np.random.randn(self.args.Ny)

        # Bring to xarray format
        Ax = xr.DataArray(Ax, dims=['channel', 'source'], coords={'channel': self.x_montage.channel, 'source': self.s_labels})
        Ay = xr.DataArray(Ay, dims=['channel', 'source'], coords={'channel': self.y_montage.channel, 'source': self.s_labels})

        # Split into target and background sources
        ax_t = Ax[:, 0 : self.args.Ns_target]
        ax_ba = Ax[:, self.args.Ns_target :]
        ay_t = Ay[:, 0 : self.args.Ns_target]
        ay_ba = Ay[:, self.args.Ns_target :]

        return Ax, ax_t, ax_ba, Ay, ay_t, ay_ba

    def forward_model(self, a_t, a_ba, s_t, s_ba):
        """Project sources to channels and add background/noise.

        Args:
            a_t (xr.DataArray): Mixing patterns for target sources with dims
                ``('channel','source')``.
            a_ba (xr.DataArray): Mixing patterns for background sources with
                dims ``('channel','source')``.
            s_t (xr.DataArray): Target source time series with dims
                ``('source','time')``.
            s_ba (xr.DataArray): Background source time series with dims
                ``('source','time')``.

        Returns:
            tuple[xr.DataArray, xr.DataArray]: ``(x_t, x_noise)`` where
                ``x_t`` is the projected target-only signal and ``x_noise`` is
                the combination of background and Gaussian noise (both Frobenius
                normalized), each with dims ``('channel','time')``.
        """

        x_t = f_normalize(a_t.dot(s_t))
        x_ba = f_normalize(a_ba.dot(s_ba))

        noise = f_normalize(np.random.normal(0, 1, x_ba.shape))
        x_noise = f_normalize(x_ba + self.args.gamma_e * noise)

        return x_t, x_noise

    def get_channels(self, x_t, x_noise, gamma, calculate_power=False):
        """Combine target and noise components to form observed channels.

        Args:
            x_t (xr.DataArray): Target-only channels with dims
                ``('channel','time')``.
            x_noise (xr.DataArray): Background+noise channels with dims
                ``('channel','time')``.
            gamma (float): Mixture weight controlling SNR; higher values weight
                the target more heavily.
            calculate_power (bool): If ``True``, also compute envelope-based
                power per-epoch for each channel.

        Returns:
            xr.DataArray | tuple[xr.DataArray, xr.DataArray]: If
            ``calculate_power`` is ``False``, returns ``x`` (channels). If
            ``True``, returns ``(x, x_power)`` where ``x_power`` has dims
            ``('channel','time')`` over epochs.
        """

        # Combine target and noise channels
        x = gamma * x_t + x_noise

        # Calculate channel-wise power spectrum
        if calculate_power:
            x_power = np.abs(hilbert(x))
            x_power = np.concatenate(
                [(split_epochs(xp, self.args.e_len).mean(axis=1)).reshape(1, -1) for xp in x_power]
            )
            x_power = xr.DataArray(
                x_power, dims=['channel', 'time'], coords={'time': self.time_y, 'channel': x.channel}
            )

            return x, x_power
        else:
            return x

    def plot_targets(self, xlim=None, ylim=None):
        """Plot target sources and their envelopes/power with optional limits.

        Args:
            xlim (tuple[float, float] | None): ``(xmin, xmax)`` for the x-axis.
            ylim (tuple[float, float] | None): ``(ymin, ymax)`` for the y-axis.

        Returns:
            None
        """

        for i in range(self.args.Ns_target):
            plt.figure(figsize=(15, 4))
            plt.plot(self.time_x, self.sx_t[i], label='Sx')
            plt.plot(self.time_y, self.sy_t[i], '-*', label='Sy')

            # Envelope
            s_env = np.abs(hilbert(self.sx_t[i]))
            plt.plot(self.time_x, s_env, label='Sx Envelope')

            # Source power
            sx_power_norm = standardize(self.sx_power[i])
            plt.plot(self.time_y, sx_power_norm, '-*', label='Sx Power (Normalized)')

            if xlim is not None:
                plt.xlim(xlim)
            if ylim is not None:
                plt.ylim(ylim)

            # Calculate correlation between Sy and Sx_power
            corr = np.corrcoef(self.sx_power[i], self.sy_t[i])[0, 1]
            
            # Calculate correlation between Sy and envelope with time-shift
            if self.args.Nde:
                corr_shifted = np.corrcoef(
                    self.sx_power[i, :-self.args.Nde], self.sy_t[i, self.args.Nde:]
                )[0, 1]

                plt.title(
                    f'Source {i+1} | (Corr(Sy, Sx_power)= {corr:.5f}) | (Corr(Sy, Sx_power time-shifted) = {corr_shifted:.5f})'
                )
            else:
                plt.title(f'Source {i+1} | (Corr(Sy, Sx_power)= {corr:.5f})')
            
            plt.legend()
            plt.grid()
            plt.show()

    def plot_channels(self, N=2, xlim=None, ylim=None):
        """Plot ``N`` pairs of channels for X and Y modalities.

        Args:
            N (int): Number of top-index channels to plot.
            xlim (tuple[float, float] | None): ``(xmin, xmax)`` for the x-axis.
            ylim (tuple[float, float] | None): ``(ymin, ymax)`` for the y-axis.

        Returns:
            None
        """

        fig, ax = plt.subplots(N, 1, figsize=(15, 4 * N), sharex=True)
        if N == 1:
            ax = [ax]
        # Background sources + noise
        for i in range(N):
            ax[i].plot(self.time_x, self.x[i], label='X')
            ax[i].plot(self.time_y, self.y[i], label='Y')
            ax[i].set_title(f'Channel {i+1}')
            ax[i].set_xlabel('Time [s]')
            ax[i].grid()
            ax[i].legend()
            if xlim is not None:
                ax[i].set_xlim(xlim)
            if ylim is not None:
                ax[i].set_ylim(ylim)
        plt.suptitle('Observed Channels', fontsize=16, fontweight='bold')
        plt.show()

    def plot_mixing_patterns(self, Ax=None, Ay=None, cmap='viridis', activity_size=200, title=None):
        """Plot mixing patterns for target sources across both modalities.

        This method generates a scatter plot of mixing patterns ``Ax`` and
        ``Ay`` for sources, in a 2D grid following the X and Y montages,
        respectively. The scatter points represent the activity level at each
        channel. Each source is represented in a separate row, with ``Ax`` on
        the left and ``Ay`` on the right.

        Args:
            Ax (np.ndarray | xr.DataArray | None): Mixing matrix for X modality
                with dims/shape ``(n_channels, n_sources)``. If ``None``, uses
                ``self.ax_t``.
            Ay (np.ndarray | xr.DataArray | None): Mixing matrix for Y modality
                with dims/shape ``(n_channels, n_sources)``. If ``None``, uses
                ``self.ay_t``.
            cmap (str): Matplotlib colormap name used to render activity.
            activity_size (float): Marker size for scatter points.
            title (str | None): Optional figure title.

        Returns:
            None
        """

        # Inputs
        n_sources = int(self.args.Ns_target)
        if Ax is None:  # shape: n_channels x n_sources
            Ax = np.asarray(self.ax_t)
        if Ay is None:  # shape: n_channels x n_sources
            Ay = np.asarray(self.ay_t)
        x_pos = np.asarray(self.x_montage.data)  # shape: n_channels x 2
        y_pos = np.asarray(self.y_montage.data)  # shape: n_channels x 2
        s_pos = np.asarray(self.s_target_positions)  # shape: n_sources x 2

        # Robust shared color normalization across all panels (5–95th percentiles)
        vals = np.concatenate([Ax.ravel(), Ay.ravel()])
        vmin, vmax = np.nanpercentile(vals, [5, 95])
        if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
            vmin = np.nanmin(vals)
            vmax = np.nanmax(vals)
            if vmin == vmax:
                vmin -= 1e-6
                vmax += 1e-6
        norm = Normalize(vmin=vmin, vmax=vmax)

        # Consistent axis limits
        all_pts = np.vstack([x_pos, y_pos, s_pos])
        xmin, ymin = np.nanmin(all_pts, axis=0)
        xmax, ymax = np.nanmax(all_pts, axis=0)
        pad_x = 0.05 * (xmax - xmin if xmax > xmin else 1.0)
        pad_y = 0.05 * (ymax - ymin if ymax > ymin else 1.0)
        xlims = (xmin - pad_x, xmax + pad_x)
        ylims = (ymin - pad_y, ymax + pad_y)

        # Figure grid
        fig, axes = plt.subplots(
            nrows=n_sources,
            ncols=2,
            figsize=(8, max(6, 4 * n_sources)),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        if n_sources == 1:
            axes = np.array(axes).reshape(1, 2)

        # --- Scatter plots (single layer with outlines) ---
        for j in range(n_sources):
            # Ax (left)
            ax = axes[j, 0]
            sc0 = ax.scatter(
                x_pos[:, 0],
                x_pos[:, 1],
                c=Ax[:, j],
                cmap=cmap,
                norm=norm,
                s=activity_size,
                edgecolors='k',
                linewidths=0.5,
            )
            ax.scatter(s_pos[j, 0], s_pos[j, 1], marker='x', c='red', s=120, linewidths=2)
            ax.set_title(f'Ax (Source {j+1})')
            ax.set_aspect('equal', adjustable='box')
            ax.set_xlim(*xlims)
            ax.set_ylim(*ylims)

            # Ay (right)
            ay = axes[j, 1]
            sc1 = ay.scatter(
                y_pos[:, 0],
                y_pos[:, 1],
                c=Ay[:, j],
                cmap=cmap,
                norm=norm,
                s=activity_size,
                edgecolors='k',
                linewidths=0.5,
            )
            ay.scatter(s_pos[j, 0], s_pos[j, 1], marker='x', c='red', s=120, linewidths=2)
            ay.set_title(f'Ay (Source {j+1})')
            ay.set_aspect('equal', adjustable='box')
            ay.set_xlim(*xlims)
            ay.set_ylim(*ylims)

        # Shared colorbar matching the right column height
        mappable = ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        fig.canvas.draw()  # finalize positions

        right_col_axes = [axes[j, 1] for j in range(n_sources)]
        x1 = max(a.get_position().x1 for a in right_col_axes)
        y0 = min(a.get_position().y0 for a in right_col_axes)
        y1 = max(a.get_position().y1 for a in right_col_axes)

        cax = fig.add_axes([x1 + 0.01, y0, 0.02, y1 - y0])
        cbar = fig.colorbar(mappable, cax=cax, orientation='vertical', label='Activity')

        # Simple legend for source marker
        fig.legend(
            handles=[
                Line2D(
                    [0], [0], marker='x', linestyle='none', color='red', markersize=8, label='Target Source True Location'
                )
            ],
            loc='upper right',
        )

        # Title
        if title is not None:
            plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.show()



def generate_args(config):
    """Read arguments from config and extend with derived parameters.

    Args:
        config (str | dict): Path to the configuration YAML file or a dict
            containing simulation parameters. Expected base keys include
            ``T``, ``rate``, ``T_epoch``, ``dT``, ``f_min``, ``f_max``,
            ``Ns_all``, ``Ns_target``, ``Nx``, ``Ny``, ``ellx``, ``elly``,
            ``sigma_noise``, ``gamma``, and ``gamma_e``.

    Returns:
        argparse.Namespace: Namespace with original and derived fields (e.g.,
        ``Nt``, ``NdT``, ``e_len``, ``Ne``, ``Nde``, and ``Ns_ba``).
    """

    # Simulation parameters
    if type(config) is str:
        with open(config, 'r') as f:
            args = Namespace(**yaml.safe_load(f))
    elif type(config) is dict:
        args = Namespace(**config)
    else:
        raise ValueError("config must be a path to a YAML file or a dictionary.")

    # Add extra parameters
    args.Ns_ba = args.Ns_all - args.Ns_target  # Number of background sources
    args.Nt = int(args.rate * args.T)  # Number of time points
    args.NdT = int(args.rate * args.dT)  # Number of time points in time-shift
    args.e_len = int(args.rate * args.T_epoch)  # Epoch index-length
    args.Ne = args.Nt // args.e_len  # Number of epochs
    args.Nde = int(args.NdT // args.e_len)  # Number of epochs in time-shift

    return args


def set_seed(seed):
    """Set the global random seeds for reproducibility.

    Args:
        seed (int): Seed value to set for ``numpy`` and Python's ``random``.

    Returns:
        None
    """

    np.random.seed(seed)
    random.seed(seed)
    print(f"Random seed set as {seed}")


def butter_lowpass(cut, fs, order):
    """Construct a digital Butterworth low-pass filter.

    Args:
        cut (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order.

    Returns:
        tuple[np.ndarray, np.ndarray]: Numerator ``b`` and denominator ``a``
        filter coefficients suitable for :func:`scipy.signal.lfilter`.
    """

    nyq = 0.5 * fs
    cut = cut / nyq
    b, a = butter(order, cut, btype='low')
    return b, a


def butter_lowpass_filter(data, cut, fs, order=5):
    """Apply a Butterworth low-pass filter to a 1D signal.

    Args:
        data (np.ndarray): Input time series of shape ``(N,)``.
        cut (float): Cutoff frequency in Hz.
        fs (float): Sampling rate in Hz.
        order (int): Filter order.

    Returns:
        np.ndarray: Filtered signal of the same shape as ``data``.
    """

    b, a = butter_lowpass(cut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def f_normalize(x):
    """Frobenius-normalize an array.

    The array is divided by the square root of the sum of squares of all
    entries. Useful to normalize channel matrices or multichannel signals.

    Args:
        x (np.ndarray | xr.DataArray): Input array.

    Returns:
        np.ndarray | xr.DataArray: Normalized array with the same shape and
        type as ``x``.
    """

    return x / np.sqrt((x**2).sum())


def standardize(x, dim='time'):
    """Z-score standardize along a dimension (for xarray) or globally (numpy).

    Args:
        x (xr.DataArray | np.ndarray): Input array to standardize.
        dim (str): Dimension name along which to standardize when ``x`` is an
            ``xr.DataArray``. Ignored for ``np.ndarray``.

    Returns:
        xr.DataArray | np.ndarray: Standardized array with mean 0 and std 1.
    """

    if isinstance(x, xr.DataArray):
        x_standard = (x - x.mean(dim)) / x.std(dim)
    elif isinstance(x, np.ndarray):
        x_standard = (x - x.mean()) / x.std()

    return x_standard


def split_epochs(x, e_len):
    """Split a 1D signal into non-overlapping epochs of a given length.

    Args:
        x (np.ndarray): 1D input signal of length ``N``.
        e_len (int): Epoch length in samples.

    Returns:
        np.ndarray: Array of shape ``(Ne, e_len)`` where ``Ne = N // e_len``.
    """

    Ne = len(x) // e_len
    return np.stack([x[i * e_len : (i + 1) * e_len] for i in range(Ne)])
