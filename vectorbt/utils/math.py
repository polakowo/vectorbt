"""Math utilities."""

import numpy as np
from numba import njit

rel_tol = 1e-10
abs_tol = 0.


def is_close(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Tell whether two values are approximately equal."""
    if np.isnan(a) or np.isnan(b):
        return False
    if np.isinf(a) or np.isinf(b):
        return False
    if a == b:
        return True
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


is_close_nb = njit(cache=True)(is_close)
"""Numba-compiled version of `is_close`."""


def is_close_or_less(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Tell whether the first value is approximately less than or equal to the second value."""
    if is_close(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return a < b


@njit(cache=True)
def is_close_or_less_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Numba-compiled version of `is_close_or_less`."""
    if is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return a < b


def is_less(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Tell whether the first value is approximately less than the second value."""
    if is_close(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return False
    return a < b


@njit(cache=True)
def is_less_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Numba-compiled version of `is_less`."""
    if is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return False
    return a < b
