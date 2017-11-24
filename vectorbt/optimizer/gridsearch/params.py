import itertools

import numpy as np


def repeat(func, N, *args, **kwargs):
    """Execute function multiple times"""
    do = lambda: func(*args, **kwargs)
    return list(itertools.starmap(do, itertools.repeat((), N)))


def range_params(min_param, max_param, step):
    """Simple range"""
    return np.arange(min_param, max_param + step, step)


def product_params(min_param, max_param, step, dims):
    """AA AB AC BA BB BC CA CB CC"""
    arange = range_params(min_param, max_param, step)
    return list(itertools.product(arange, repeat=dims))


def combine_params(min_param, max_param, step, dims):
    """AB AC AD BC BD CD"""
    arange = range_params(min_param, max_param, step)
    return list(itertools.combinations(arange, dims))


def combine_rep_params(min_param, max_param, step, dims):
    """AA AB AC BB BC CC"""
    arange = range_params(min_param, max_param, step)
    return list(itertools.combinations_with_replacement(arange, dims))


def random_params(min_param, max_param, dims, N):
    """Generate randomized params"""
    return repeat(np.random.uniform, N, min_param, max_param, dims)


def slices(sr, n):
    """Split series into n chunks and return slices"""
    return [slice(chunk[0], chunk[-1] + 1) for chunk in np.array_split(range(len(sr.index)), n)]


def multiply(groups, params):
    """ABC * DEF = AD AE AF BD BE BF CD CE CF"""
    multi_params = zip(*(params * len(groups)))
    multi_groups = [dp for p in range(len(groups)) for dp in [p] * len(params)]
    joined = list(zip(multi_groups, *multi_params))
    return joined
