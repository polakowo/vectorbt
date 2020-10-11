"""Numba-compiled utility functions for working with arrays."""

import numpy as np
from numba import njit


def is_sorted(a):
    """Checks if array is sorted."""
    return np.all(a[:-1] <= a[1:])


@njit(cache=True)
def is_sorted_nb(a):
    """Numba-compiled version of `is_sorted`."""
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
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


@njit(cache=True)
def uniform_summing_to_one_nb(n):
    """Generate random floats summing to one.

    See # https://stackoverflow.com/a/2640067/8141780"""
    rand_floats = np.empty(n + 1, dtype=np.float_)
    rand_floats[0] = 0.
    rand_floats[1] = 1.
    rand_floats[2:] = np.random.uniform(0, 1, n - 1)
    rand_floats = np.sort(rand_floats)
    rand_floats = rand_floats[1:] - rand_floats[:-1]
    return rand_floats


@njit(cache=True)
def renormalize_nb(n, range1, range2):
    """Convert from one range to another."""
    delta1 = range1[1] - range1[0]
    delta2 = range2[1] - range2[0]
    return (delta2 * (n - range1[0]) / delta1) + range2[0]


@njit(cache=True)
def rescale_float_to_int_nb(floats, int_range, total):
    """Rescale a float array into an int array."""
    ints = np.floor(renormalize_nb(floats, [0., 1.], int_range))
    leftover = int(total - ints.sum())
    for i in range(leftover):
        ints[np.random.choice(len(ints))] += 1
    return ints

