import os, tempfile
import numpy as np
import xarray as xr
import cedalion
import cedalion.io as cio

def create_dummy_Adot():
    """Create a dummy Adot matrix for testing."""
    channels = ['S'+str(i)+'D'+str(j) for i in range(14) for j in range(30)]
    channel = np.random.choice(channels, 100)
    num_verts = np.random.randint(500, 5000)
    wavelength = np.array([760.0, 850.0])
    Adot = xr.DataArray(np.random.rand(len(channel), num_verts, len(wavelength)),
                        dims=["channel", "vertex", "wavelength"],
                        coords={"channel": ("channel", channel),
                                "is_brain": ("vertex", np.random.randint(0, 2, num_verts)),
                                "wavelength": ("wavelength", wavelength)})
    return Adot


def test_save_Adot():
    Adot = create_dummy_Adot()
    # save to file
    dirpath = tempfile.mkdtemp()
    tmp_fn = os.path.join(dirpath, "test_Adot.nc")
    Adot.to_netcdf(tmp_fn)
    # load from file
    Adot2 = cio.load_Adot(tmp_fn)
    # compare
    assert np.all(Adot.values == Adot2.values)
    assert np.all(Adot.channel.values == Adot2.channel.values)
    assert np.all(Adot.vertex.values == Adot2.vertex.values)
    assert np.all(Adot.wavelength.values == Adot2.wavelength.values)


def test_load_Adot():
    Adot = create_dummy_Adot()
    # save to file
    dirpath = tempfile.mkdtemp()
    tmp_fn = os.path.join(dirpath, "test_Adot.nc")
    cio.save_Adot(tmp_fn, Adot)
    # load from file
    Adot2 = xr.open_dataset(tmp_fn)
    # compare
    assert np.all(Adot.values == Adot2.to_array()[0])
    assert np.all(Adot.channel.values == Adot2.channel.values)
    assert np.all(Adot.vertex.values == Adot2.vertex.values)
    assert np.all(Adot.wavelength.values == Adot2.wavelength.values)


