"""Builds on pint_xarray's unit registry."""
import pint_xarray

units = pint_xarray.unit_registry
Quantity = units.Quantity

# Aliases that we want to provide. FIXME: maybe a definition file is more convenient?
units.define("@alias deg = o")
units.define("@alias degC = oC")
units.define("@alias ohm = Ohm")

# FIXME temporarily define ADU unit in WINGS snirf datasets to avoid an error
units.define("ADU = 1")
