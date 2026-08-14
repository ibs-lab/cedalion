import numpy as np

from cedalion.sigproc.bvp_wav_ana_v12 import interpft, peakseek, bvp_single_ch


def test_interpft():
    a = np.array([
        0.21362854,  0.05005765,  0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266])

    n_samples = 50
    n_upsampled = 100
    t_samples = np.arange(n_samples) / (n_samples)
    t_upsampled = np.arange(n_upsampled) / (n_upsampled)

    k = np.arange(1, 9)[:, None]

    angle_samples = 2 * np.pi * k * t_samples
    angle_upsampled = 2 * np.pi * k * t_upsampled

    x = np.sum(
        a[:, None] * np.sin(angle_samples)
        + b[:, None] * (np.cos(angle_samples) - 1),
        axis=0)

    upsampled = interpft(x, n_upsampled)
    expected_upsampled = np.sum(
        a[:, None] * np.sin(angle_upsampled)
        + b[:, None] * (np.cos(angle_upsampled) - 1),
        axis=0)

    assert upsampled.shape == (n_upsampled,)
    assert np.isrealobj(upsampled)
    np.testing.assert_allclose(upsampled, expected_upsampled, atol=1e-12)

    downsampled = interpft(upsampled, n_samples)

    assert downsampled.shape == x.shape
    np.testing.assert_allclose(downsampled, x, atol=1e-12)

def test_peakseek():
    simple_signal = np.array([
        10.0, 0.0, 2.0, 0.0, 3.0, 0.0,
        1.0, 0.0, 0.0, 4.0, 0.0, 10.0,
    ])

    locations, peaks = peakseek(
        simple_signal,
        minpeakdist=3,
        minpeakh=1.5,
    )

    np.testing.assert_array_equal(locations, np.array([4, 9]))
    np.testing.assert_array_equal(peaks, np.array([3.0, 4.0]))

    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266])

    n_samples = 80
    t_samples = np.arange(n_samples) / n_samples
    k = np.arange(1, 9)[:, None]
    angle_samples = 2 * np.pi * k * t_samples

    multiharmonic_signal = np.sum(
        a[:, None] * np.sin(angle_samples)
        + b[:, None] * (np.cos(angle_samples) - 1),
        axis=0)

    locations, peaks = peakseek(multiharmonic_signal)

    np.testing.assert_array_equal(locations, np.array([23, 35]))
    np.testing.assert_allclose(
        peaks,
        np.array([0.9698297535333608, 0.9508981042738599]),
        atol=1e-12)

def test_bvp_single_ch():
    # Simple signal: a linear concentration change should be entirely trend.
    concentration = np.linspace(5.0, 15.0, 100)
    original_concentration = concentration.copy()
    fs = 10.0
    fs_new = 20.0

    bvp, resampled, trend = bvp_single_ch(concentration, fs, fs_new)

    expected_length = 200
    expected_resampled = np.linspace(0.0, 10.0, expected_length)

    assert bvp.shape == (expected_length,)
    assert resampled.shape == (expected_length,)
    assert trend.shape == (expected_length,)
    np.testing.assert_array_equal(concentration, original_concentration)
    np.testing.assert_allclose(resampled, expected_resampled, atol=1e-12)
    np.testing.assert_allclose(trend, expected_resampled, atol=1e-10)
    np.testing.assert_allclose(bvp, np.zeros(expected_length), atol=1e-10)
    np.testing.assert_allclose(bvp, resampled - trend, atol=1e-12)

    # Physiological signal: an asymmetric multiharmonic pulse is superimposed
    # on a slowly varying concentration trend.
    a = np.array([
        0.21362854, 0.05005765, 0.00135350, -0.00906460,
        -0.00849233, -0.02022217, -0.01101648, -0.00662955,
    ])

    b = np.array([
        -0.33609957, -0.12807130, -0.08131665, -0.03903412,
        -0.02956690, -0.01606627, -0.00380470, -0.00521266,
    ])

    fs = 20.0
    fs_new = 50.0
    duration = 30.0
    pulse_frequency = 1.0
    n_samples = int(fs * duration)
    n_resampled = int((fs_new / fs) * n_samples)

    time = np.linspace(0.0, duration, n_samples)
    time_resampled = np.linspace(0.0, duration, n_resampled)
    k = np.arange(1, 9)[:, None]

    def make_pulse(actual_time):
        angles = 2 * np.pi * k * pulse_frequency * actual_time
        pulse = np.sum(
            a[:, None] * np.sin(angles)
            + b[:, None] * (np.cos(angles) - 1),
            axis=0)

        # The mean of the multiharmonic expression above is -sum(b).
        return pulse + np.sum(b)

    def make_low_frequency_trend(actual_time):
        return (
            1.2
            + 0.25 * np.sin(2 * np.pi * 0.1 * actual_time)
            - 0.01 * actual_time)

    pulse = make_pulse(time)
    low_frequency_trend = make_low_frequency_trend(time)
    concentration = low_frequency_trend + pulse

    bvp, resampled, trend = bvp_single_ch(concentration, fs, fs_new)

    expected_pulse = make_pulse(time_resampled)
    expected_trend = make_low_frequency_trend(time_resampled) - concentration[0]
    expected_resampled = expected_trend + expected_pulse

    assert bvp.shape == (n_resampled,)
    assert resampled.shape == (n_resampled,)
    assert trend.shape == (n_resampled,)
    np.testing.assert_allclose(resampled, expected_resampled, atol=0.05)
    np.testing.assert_allclose(bvp, resampled - trend, atol=1e-12)

    # Ignore the LOESS boundary region and evaluate the stable central part.
    edge = int(3.5 * fs_new)
    interior = slice(edge, -edge)

    trend_error = trend[interior] - expected_trend[interior]
    trend_rmse = np.sqrt(np.mean(trend_error**2))
    trend_correlation = np.corrcoef(
        trend[interior],
        expected_trend[interior])[0, 1]
    bvp_correlation = np.corrcoef(
        bvp[interior],
        expected_pulse[interior])[0, 1]

    assert trend_rmse < 0.08
    assert trend_correlation > 0.98
    assert bvp_correlation > 0.95
