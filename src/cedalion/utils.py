"""Utility functions."""

import numpy as np


def zero_padded_numbers(numbers: list[int], prefix: str = "") -> list[str]:
    """Translate [0,1,..,13] to ["00","01", ... "13"].

    Determine the length of the zero padding to accomodate the highest absolute
    number. Optionally, a prefix can be specified.
    """
    if len(numbers) == 0:
        return []

    length = int(np.ceil(np.log10(max(np.abs(numbers)))))
    return [prefix + str(i).zfill(length) for i in numbers]
