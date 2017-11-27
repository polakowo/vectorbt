import numpy as np


def from_idx(n, idx):
    """
    n = 6
    set_idx = 1, 3, 5
    =   0 1 0 1 0 1
    """
    v = np.zeros(n, dtype=int)
    v[idx] = 1
    return v


def first(v):
    """
        0 1 1 1 1 0
    =   0 1 0 0 0 0
    """
    rolled = np.roll(v, 1)
    rolled[0] = 0
    return (v - rolled == 1).astype(int)


def last(v):
    """
        0 1 1 1 1 0
    =   0 0 0 0 1 0
    """
    rolled = np.roll(v, -1)
    rolled[-1] = 0
    return (v - rolled == 1).astype(int)


def after_last(v):
    """
        0 1 1 1 1 0
    =   0 0 0 0 0 1
    """
    rolled = np.roll(last(v), 1)
    rolled[0] = 0
    return rolled


def ffill(arr):
    # Forward fill
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


def nst(v, n):
    """
    n = 2
        0 1 1 1 1 0
    =   0 0 1 0 0 0
    """
    cum = np.cumsum(v)
    idx = np.flatnonzero(first(v))
    z = np.zeros(len(v), dtype=int)
    z[idx] = cum[idx]
    mask = cum - ffill(z) + 1 == n
    return v * np.where(mask, 1, 0)


def from_nst(v, n):
    """
    n = 2
        0 1 1 1 1 0
    =   0 0 1 1 1 0
    """
    cum = np.cumsum(v)
    idx = np.flatnonzero(first(v))
    z = np.zeros(len(v), dtype=int)
    z[idx] = cum[idx]
    mask = cum - ffill(z) + 1 >= n
    return v * np.where(mask, 1, 0)


def AND(v1, v2):
    """
        0 0 1 1 1 0
        0 1 1 1 0 0
    =   0 0 1 1 0 0
    """
    return v1 * v2


def OR(v1, v2):
    """
        0 0 1 1 1 0
        0 1 1 1 0 0
    =   0 1 1 1 1 0
    """
    return (v1 + v2 > 0).astype(int)


def XOR(v1, v2):
    """
        0 0 1 1 1 0
        0 1 1 1 0 0
    =   0 1 0 0 1 0
    """
    return (v1 - v2 != 0).astype(int)


def NOT(v):
    """
        0 1 1 1 1 0
    =   1 0 0 0 0 1
    """
    return ((v + 1) == 1).astype(int)
