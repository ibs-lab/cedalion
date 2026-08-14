import numpy as np
from cedalion.sigproc.bvp_wav_ana_v12 import (
                interpft,
                peakseek,
                bvp_single_ch,
                wct)


def test_interpft():
    """Tests Fourier interpolation by upsampling and downsampling.

    A periodic multiharmonic signal with a known analytical representation
    is first upsampled from 50 to 100 samples. The result is compared with
    the signal evaluated directly at the finer sampling points.

    The upsampled signal is subsequently downsampled to verify that the
    original signal can be reconstructed.
    """

    # --- Define coefficients of the multiharmonic signal ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266])

    # --- Create original and upsampled time vectors ---
    n_samples = 50
    n_upsampled = 100
    t_samples = np.arange(n_samples) / n_samples
    t_upsampled = np.arange(n_upsampled) / n_upsampled

    k = np.arange(1, 9)[:, None]

    angle_samples = 2 * np.pi * k * t_samples
    angle_upsampled = 2 * np.pi * k * t_upsampled

    # --- Generate the original multiharmonic signal ---
    x = np.sum(
        a[:, None] * np.sin(angle_samples)
        + b[:, None] * (np.cos(angle_samples) - 1),
        axis=0)

    # --- Upsample the signal using Fourier interpolation ---
    upsampled = interpft(x, n_upsampled)

    # Calculate the expected signal directly at the finer sampling points.
    expected_upsampled = np.sum(
        a[:, None] * np.sin(angle_upsampled)
        + b[:, None] * (np.cos(angle_upsampled) - 1),
        axis=0)

    # --- Check the upsampled signal ---
    assert upsampled.shape == (n_upsampled,)
    assert np.isrealobj(upsampled)
    np.testing.assert_allclose(
        upsampled,
        expected_upsampled,
        atol=1e-12)

    # --- Downsample to the original number of samples ---
    downsampled = interpft(upsampled, n_samples)

    # --- Check reconstruction of the original signal ---
    assert downsampled.shape == x.shape
    np.testing.assert_allclose(
        downsampled,
        x,
        atol=1e-12)

def test_peakseek():
    """Tests peak detection using simple and multiharmonic signals.

    The first signal contains clearly defined peaks and verifies the minimum
    peak distance and minimum peak height parameters.

    The second signal represents a realistic multiharmonic pulse waveform
    and verifies peak locations and amplitudes for a more complex signal.
    """

    # --- Test a simple signal with known peak locations ---
    simple_signal = np.array([
        10.0, 0.0, 2.0, 0.0, 3.0, 0.0,
        1.0, 0.0, 0.0, 4.0, 0.0, 10.0,
    ])

    locations, peaks = peakseek(
        simple_signal,
        minpeakdist=3,
        minpeakh=1.5,
    )

    # Peaks at the signal boundaries are not considered local maxima.
    np.testing.assert_array_equal(
        locations,
        np.array([4, 9]))
    np.testing.assert_array_equal(
        peaks,
        np.array([3.0, 4.0]))

    # --- Define coefficients of the multiharmonic signal ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266])

    # --- Generate a multiharmonic pulse waveform ---
    n_samples = 80
    t_samples = np.arange(n_samples) / n_samples
    k = np.arange(1, 9)[:, None]
    angle_samples = 2 * np.pi * k * t_samples

    multiharmonic_signal = np.sum(
        a[:, None] * np.sin(angle_samples)
        + b[:, None] * (np.cos(angle_samples) - 1),
        axis=0)

    # --- Detect peaks in the multiharmonic waveform ---
    locations, peaks = peakseek(multiharmonic_signal)

    # --- Check detected peak locations and amplitudes ---
    np.testing.assert_array_equal(
        locations,
        np.array([23, 35]))
    np.testing.assert_allclose(
        peaks,
        np.array([0.9698297535333608, 0.9508981042738599]),
        atol=1e-12)

def test_bvp_single_ch():
    """Tests BVP extraction from a single-channel concentration signal.

    Two test cases are evaluated:

    1. A linear concentration change containing only a low-frequency trend.
       The extracted BVP should therefore be zero.
    2. A physiological multiharmonic pulse superimposed on a slowly varying
       trend. The extracted trend and BVP are compared with their known
       analytical components.
    """

    # --- Test a signal containing only a linear trend ---
    concentration = np.linspace(5.0, 15.0, 100)
    original_concentration = concentration.copy()
    fs = 10.0
    fs_new = 20.0

    bvp, resampled, trend = bvp_single_ch(
        concentration,
        fs,
        fs_new)

    expected_length = 200
    expected_resampled = np.linspace(
        0.0,
        10.0,
        expected_length)

    # --- Check output dimensions and input immutability ---
    assert bvp.shape == (expected_length,)
    assert resampled.shape == (expected_length,)
    assert trend.shape == (expected_length,)
    np.testing.assert_array_equal(
        concentration,
        original_concentration)

    # --- Check separation of the linear trend ---
    np.testing.assert_allclose(
        resampled,
        expected_resampled,
        atol=1e-12)
    np.testing.assert_allclose(
        trend,
        expected_resampled,
        atol=1e-10)
    np.testing.assert_allclose(
        bvp,
        np.zeros(expected_length),
        atol=1e-10)
    np.testing.assert_allclose(
        bvp,
        resampled - trend,
        atol=1e-12)

    # --- Define coefficients of the physiological pulse waveform ---
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    # --- Define sampling and signal parameters ---
    fs = 20.0
    fs_new = 50.0
    duration = 30.0
    pulse_frequency = 1.0
    n_samples = int(fs * duration)
    n_resampled = int((fs_new / fs) * n_samples)

    time = np.linspace(0.0, duration, n_samples)
    time_resampled = np.linspace(
        0.0,
        duration,
        n_resampled)

    k = np.arange(1, 9)[:, None]

    def make_pulse(actual_time):
        """Generate the physiological multiharmonic pulse waveform."""
        angles = 2 * np.pi * k * pulse_frequency * actual_time

        pulse = np.sum(
            a[:, None] * np.sin(angles)
            + b[:, None] * (np.cos(angles) - 1),
            axis=0)

        # The mean of the multiharmonic expression is -sum(b).
        return pulse + np.sum(b)

    def make_low_frequency_trend(actual_time):
        """Generate the known low-frequency concentration trend."""
        return (
            1.2
            + 0.25 * np.sin(2 * np.pi * 0.1 * actual_time)
            - 0.01 * actual_time)

    # --- Combine the pulse and low-frequency trend ---
    pulse = make_pulse(time)
    low_frequency_trend = make_low_frequency_trend(time)
    concentration = low_frequency_trend + pulse

    # --- Extract the BVP and trend components ---
    bvp, resampled, trend = bvp_single_ch(
        concentration,
        fs,
        fs_new)

    # --- Calculate the expected resampled components ---
    expected_pulse = make_pulse(time_resampled)
    expected_trend = (
        make_low_frequency_trend(time_resampled)
        - concentration[0]
    )
    expected_resampled = expected_trend + expected_pulse

    # --- Check output dimensions and signal reconstruction ---
    assert bvp.shape == (n_resampled,)
    assert resampled.shape == (n_resampled,)
    assert trend.shape == (n_resampled,)
    np.testing.assert_allclose(
        resampled,
        expected_resampled,
        atol=0.05)
    np.testing.assert_allclose(
        bvp,
        resampled - trend,
        atol=1e-12)

    # --- Exclude the unstable LOESS boundary regions ---
    edge = int(3.5 * fs_new)
    interior = slice(edge, -edge)

    # --- Calculate trend and pulse extraction quality ---
    trend_error = (
        trend[interior]
        - expected_trend[interior]
    )
    trend_rmse = np.sqrt(
        np.mean(trend_error**2)
    )
    trend_correlation = np.corrcoef(
        trend[interior],
        expected_trend[interior],
    )[0, 1]
    bvp_correlation = np.corrcoef(
        bvp[interior],
        expected_pulse[interior],
    )[0, 1]

    # --- Check trend and BVP extraction accuracy ---
    assert trend_rmse < 0.08
    assert trend_correlation > 0.98
    assert bvp_correlation > 0.95

def test_wct():
    """Tests wavelet coherence using two sinusoidal signals with a known
    phase shift.

    The signals have identical frequencies and amplitudes. Therefore, their
    coherence should be close to one at the signal frequency. The measured
    phase should correspond to the known phase shift.
    """  # noqa: D205

    # --- Create sinusoidal test signals ---
    fs = 50.0
    duration = 20.0
    signal_frequency = 1.25
    phase_shift = np.pi / 3

    time = np.arange(int(fs * duration)) / fs

    signal_1 = np.sin(
        2 * np.pi * signal_frequency * time
    )
    signal_2 = np.sin(
        2 * np.pi * signal_frequency * time + phase_shift
    )

    # --- Calculate wavelet coherence ---
    (
        cross_wavelet,
        power_1,
        power_2,
        coherence,
        phase,
        coi,
        frequencies,
        significance,
    ) = wct(
        signal_1,
        signal_2,
        dt=1.0 / fs,
        sig=False,
    )

    # --- Check output dimensions ---
    expected_shape = (len(frequencies), len(time))

    assert cross_wavelet.shape == expected_shape
    assert power_1.shape == expected_shape
    assert power_2.shape == expected_shape
    assert coherence.shape == expected_shape
    assert phase.shape == expected_shape
    assert coi.shape == time.shape
    np.testing.assert_array_equal(significance, np.array([0]))

    # --- Select scale closest to the signal frequency ---
    frequency_idx = np.argmin(
        np.abs(frequencies - signal_frequency)
    )

    # --- Exclude values outside the cone of influence ---
    signal_period = 1.0 / frequencies[frequency_idx]
    valid_idx = signal_period <= coi

    assert np.count_nonzero(valid_idx) > len(time) // 2

    # --- Check wavelet coherence ---
    mean_coherence = np.mean(
        coherence[frequency_idx, valid_idx]
    )

    assert mean_coherence > 0.95

    # --- Calculate circular mean phase ---
    mean_phase = np.angle(
        np.mean(
            np.exp(1j * phase[frequency_idx, valid_idx])
        )
    )

    # W1 * conj(W2) produces the negative phase difference.
    expected_phase = -phase_shift
    phase_error = np.angle(
        np.exp(1j * (mean_phase - expected_phase))
    )

    # --- Check phase difference ---
    assert abs(phase_error) < 0.15
