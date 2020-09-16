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


@njit(cache=True)
def insert_argsort_nb(A, I):
    """Perform argsort using insertion sort.

    In-memory and without recursion -> very fast for smaller arrays."""
    for j in range(1, len(A)):
        key = A[j]
        key2 = I[j]
        i = j - 1
        while (i >= 0) & (A[i] > key):
            A[i + 1] = A[i]
            I[i + 1] = I[i]
            i = i - 1
        A[i + 1] = key
        I[i + 1] = key2


def get_ranges_arr(starts, ends):
    """Build array from start and end indices.

    Based on https://stackoverflow.com/a/37626057"""
    starts = np.asarray(starts)
    if starts.ndim == 0:
        starts = np.array([starts])
    ends = np.asarray(ends)
    if ends.ndim == 0:
        ends = np.array([ends])
    starts, end = np.broadcast_arrays(starts, ends)
    counts = ends - starts
    counts_csum = counts.cumsum()
    id_arr = np.ones(counts_csum[-1], dtype=int)
    id_arr[0] = starts[0]
    id_arr[counts_csum[:-1]] = starts[1:] - ends[:-1] + 1
    return id_arr.cumsum()
