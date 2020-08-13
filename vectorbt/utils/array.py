"""Numba-compiled utility functions for working with arrays."""

import numpy as np
from numba import njit


def is_sorted(a):
    """Checks if array is sorted."""
    return np.all(a[:-1] <= a[1:])


@njit(cache=True)
def is_sorted_nb(a):
    """Numba-compiled version of `is_sorted`."""
    for i in range(a.size-1):
        if a[i+1] < a[i]:
            return False
    return True
