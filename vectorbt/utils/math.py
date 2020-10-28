"""Math utilities."""

import numpy as np
from numba import njit

rel_tol = 1e-10
abs_tol = 0.


@njit(cache=True)
def is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Tell whether two values are approximately equal."""
    if np.isnan(a) or np.isnan(b):
        return False
    if np.isinf(a) or np.isinf(b):
        return False
    if a == b:
        return True
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


@njit(cache=True)
def is_addition_zero_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Tell whether addition of two values yields zero."""
    if np.sign(a) != np.sign(b):
        return is_close_nb(abs(a), abs(b), rel_tol=rel_tol, abs_tol=abs_tol)
    return is_close_nb(a + b, 0., rel_tol=rel_tol, abs_tol=abs_tol)


@njit(cache=True)
def is_subtraction_zero_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Tell whether subtraction of two values yields zero."""
    if np.sign(a) == np.sign(b):
        return is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    return is_close_nb(a - b, 0., rel_tol=rel_tol, abs_tol=abs_tol)


@njit(cache=True)
def is_close_or_less_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Tell whether the first value is approximately less than or equal to the second value."""
    if is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return True
    return a < b


@njit(cache=True)
def is_less_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
    """Tell whether the first value is approximately less than the second value."""
    if is_close_nb(a, b, rel_tol=rel_tol, abs_tol=abs_tol):
        return False
    return a < b
