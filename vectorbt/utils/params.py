import itertools
import numpy as np


def product(a1, a2):
    """
    A B C
    D E F
    = 
    AD AE AF BD BE BF CD CE CF"""
    return np.repeat(a1, a2.shape[0]), np.tile(a2, a2.shape[0])


def range_product(a):
    """
    A B C
    =
    AA AB AC BA BB BC CA CB CC"""
    b, c = zip(*itertools.product(a, repeat=2))
    return np.asarray(b), np.asarray(c)


def range_combinations(a):
    """
    A B C
    =
    AB AC AD BC BD CD"""
    b, c = zip(*itertools.combinations(a, 2))
    return np.asarray(b), np.asarray(c)


def range_combinations_with_rep(a):
    """
    A B C
    =
    AA AB AC BB BC CC"""
    b, c = zip(*itertools.combinations_with_replacement(a, 2))
    return np.asarray(b), np.asarray(c)
