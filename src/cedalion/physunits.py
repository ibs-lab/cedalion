"""Builds on pint_xarray's unit registry."""

import pint_xarray
import re

units = pint_xarray.unit_registry
Quantity = units.Quantity

# Aliases that we want to provide. FIXME: maybe a definition file is more convenient?
units.define("@alias deg = o")
units.define("@alias degC = oC")
units.define("@alias ohm = Ohm")

# FIXME temporarily define ADU unit in WINGS snirf datasets to avoid an error
units.define("ADU = 1")

# leading (optional sign +) number, allowing decimals and scientific notation
_MAGNITUDE_PATTERN = re.compile(r"^\s*[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?")

def parse_quantity(s : str) -> Quantity:
    """Parse a string into a Quantity."""

    if not _MAGNITUDE_PATTERN.match(s):
        raise ValueError(f"no leading magnitude in {s}")

    return units(s)
