import cedalion.datasets
import cedalion.sigdecomp.dummy


def test_split_frequency_bands():
    recordings = cedalion.datasets.get_snirf_test_data()
    amp = recordings[0].timeseries["amp"]

    assert amp.dims == ("channel", "wavelength", "time")

    x = cedalion.sigdecomp.dummy.split_frequency_bands(amp)

    assert x.dims == ("band", "channel", "wavelength", "time")
    assert all(x.band == ["cardiac", "respiratory"])
    assert x.shape[1:] == amp.shape
