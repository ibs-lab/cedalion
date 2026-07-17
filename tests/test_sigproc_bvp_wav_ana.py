import numpy as np

from cedalion.sigproc.bvp_wav_ana_v12 import interpft, peakseek


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
    signal = np.array([
        10.0, 0.0, 2.0, 0.0, 3.0, 0.0,
        1.0, 0.0, 0.0, 4.0, 0.0, 10.0,
    ])

    locations, peaks = peakseek(signal, minpeakdist=3, minpeakh=1.5)

    np.testing.assert_array_equal(locations, np.array([4, 9]))
    np.testing.assert_array_equal(peaks, np.array([3.0, 4.0]))

