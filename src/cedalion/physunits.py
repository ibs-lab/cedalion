import pint_xarray

units = pint_xarray.unit_registry
Quantity = units.Quantity

# Aliases that we want to provide. FIXME: maybe a definition file is more convenient?
units.define("@alias deg = o")
units.define("@alias degC = oC")
units.define("@alias ohm = Ohm")
