import pytest

import cedalion.nirs
import pint


def test_get_extinction_coefficients_notexistant():
    wavelengths = [750, 850]

    with pytest.raises(ValueError):
        cedalion.nirs.get_extinction_coefficients("nonsupported spectrum", wavelengths)


def test_get_extinction_coefficients_prahl():
    wavelengths = [750, 850]
    E = cedalion.nirs.get_extinction_coefficients("prahl", wavelengths)

    assert "chromo" in E.dims
    assert "wavelength" in E.dims
    assert E.pint.units == pint.Unit("mm^-1 / M")

    assert (E.wavelength.values == wavelengths).all()
