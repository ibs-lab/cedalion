import numpy as np

from cedalion.sigproc.bvp_wav_ana_v12 import interpft


def test_interpft():
    n_samples = 80
    n_upsampled = 100
    x = np.sin(2 * np.pi * np.arange(n_samples) / n_samples)

    upsampled = interpft(x, n_upsampled)
    expected_upsampled = np.sin(
        2 * np.pi * np.arange(n_upsampled) / n_upsampled
    )

    assert upsampled.shape == (n_upsampled,)
    assert np.isrealobj(upsampled)
    np.testing.assert_allclose(upsampled, expected_upsampled, atol=1e-12)

    downsampled = interpft(upsampled, n_samples)

    assert downsampled.shape == x.shape
    np.testing.assert_allclose(downsampled, x, atol=1e-12)
