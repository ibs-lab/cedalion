"""Utility functions."""

import numpy as np
import warnings
import functools


def zero_padded_numbers(numbers: list[int], prefix: str = "") -> list[str]:
    """Format integers as zero-padded strings sized to the largest value.

    For example, ``[0, 1, 13]`` becomes ``["00", "01", "13"]``. The padding
    width is determined by the highest absolute value in the list.

    Args:
        numbers: List of integers to format.
        prefix: Optional string prepended to each formatted number.

    Returns:
        List of zero-padded strings, one per input number.
    """
    if len(numbers) == 0:
        return []

    length = int(np.ceil(np.log10(max(np.abs(numbers)))))
    return [prefix + str(i).zfill(length) for i in numbers]


def deprecated_api(message):
    """Issue a ``DeprecationWarning`` with the given message.

    Args:
        message: Human-readable description of what is deprecated and what
            to use instead.
    """
    # FIXME: replace with @deprecated for python 3.13
    warnings.warn(message, DeprecationWarning)


def deprecated(reason: str):
    """Decorator that marks a function as deprecated.

    Wraps the decorated function so that every call emits a
    ``DeprecationWarning`` naming the function and the reason for deprecation.

    Args:
        reason: Explanation of why the function is deprecated and, where
            applicable, what should be used instead.

    Returns:
        A decorator that wraps the target function.
    """

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
