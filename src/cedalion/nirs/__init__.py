"""Functionality to process NIRS data."""

from .common import (
    get_extinction_coefficients,
    channel_distances,
    split_long_short_channels
)

import cedalion.nirs.cw
import cedalion.nirs.fd
import cedalion.nirs.td
