"""Utility functions."""

import numpy as np
import warnings
import functools


def zero_padded_numbers(numbers: list[int], prefix: str = "") -> list[str]:
    """Translate [0,1,..,13] to ["00","01", ... "13"].

    Determine the length of the zero padding to accomodate the highest absolute
    number. Optionally, a prefix can be specified.
    """
    if len(numbers) == 0:
        return []

    length = int(np.ceil(np.log10(max(np.abs(numbers)))))
    return [prefix + str(i).zfill(length) for i in numbers]


def deprecated_api(message):
    """Issue a deprecation warning."""

    # FIXME: replace with @deprecated for python 3.13
    warnings.warn(message, DeprecationWarning)


def deprecated(reason: str):
    """Marks a function as deprecated by issuing a DeprecationWarning when used."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"'{func.__name__}' is deprecated: {reason}",
                category=DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper

    return decorator
