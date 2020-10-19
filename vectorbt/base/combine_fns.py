"""Functions for combining arrays."""

import numpy as np
from numba import njit
from numba.typed import List

from vectorbt.base import reshape_fns


def apply_and_concat_one(n, apply_func, *args, **kwargs):
    """For each value `i` from 0 to `n`, apply `apply_func` with arguments `*args` and `**kwargs`, 
    and concat the results along axis 1. 
    
    The result of `apply_func` must be a single 1-dim or 2-dim array.
    
    `apply_func` must accept arguments `i`, `*args` and `**kwargs`."""
    return np.hstack([reshape_fns.to_2d(apply_func(i, *args, **kwargs)) for i in range(n)])


@njit
def to_2d_one_nb(a):
    """Expand the dimensions of array `a` along axis 1.
    
    !!! note
        * `a` must be strictly homogeneous"""
    if a.ndim > 1:
        return a
    return np.expand_dims(a, axis=1)


@njit
def apply_and_concat_one_nb(n, apply_func_nb, *args):  # numba doesn't accepts **kwargs
    """A Numba-compiled version of `apply_and_concat_one`.
    
    !!! note
        * `apply_func_nb` must be Numba-compiled
        * `*args` must be Numba-compatible
        * No support for `**kwargs`
    """
    output_0 = to_2d_one_nb(apply_func_nb(0, *args))
    output = np.empty((output_0.shape[0], n * output_0.shape[1]), dtype=output_0.dtype)
    for i in range(n):
        if i == 0:
            outputs_i = output_0
        else:
            outputs_i = to_2d_one_nb(apply_func_nb(i, *args))
        output[:, i * outputs_i.shape[1]:(i + 1) * outputs_i.shape[1]] = outputs_i
    return output


def apply_and_concat_multiple(n, apply_func, *args, **kwargs):
    """Identical to `apply_and_concat_one`, except that the result of `apply_func` must be 
    multiple 1-dim or 2-dim arrays. Each of these arrays at `i` will be concatenated with the
    array at the same position at `i+1`."""
    outputs = [tuple(map(reshape_fns.to_2d, apply_func(i, *args, **kwargs))) for i in range(n)]
    return list(map(np.hstack, list(zip(*outputs))))


@njit
def to_2d_multiple_nb(a):
    """Expand the dimensions of each array in `a` along axis 1.
    
    !!! note
        * `a` must be strictly homogeneous
    """
    lst = List()
    for _a in a:
        lst.append(to_2d_one_nb(_a))
    return lst


@njit
def apply_and_concat_multiple_nb(n, apply_func_nb, *args):
    """A Numba-compiled version of `apply_and_concat_multiple`.
    
    !!! note
        * Output of `apply_func_nb` must be strictly homogeneous
        * `apply_func_nb` must be Numba-compiled
        * `*args` must be Numba-compatible
        * No support for `**kwargs`
    """
    outputs = []
    outputs_0 = to_2d_multiple_nb(apply_func_nb(0, *args))
    for j in range(len(outputs_0)):
        outputs.append(np.empty((outputs_0[j].shape[0], n * outputs_0[j].shape[1]), dtype=outputs_0[j].dtype))
    for i in range(n):
        if i == 0:
            outputs_i = outputs_0
        else:
            outputs_i = to_2d_multiple_nb(apply_func_nb(i, *args))
        for j in range(len(outputs_i)):
            outputs[j][:, i * outputs_i[j].shape[1]:(i + 1) * outputs_i[j].shape[1]] = outputs_i[j]
    return outputs


def select_and_combine(i, obj, others, combine_func, *args, **kwargs):
    """Combine `obj` and an element from `others` at `i` using `combine_func`."""
    return combine_func(obj, others[i], *args, **kwargs)


def combine_and_concat(obj, others, combine_func, *args, **kwargs):
    """Use `apply_and_concat_one` to combine `obj` with each element from `others` using `combine_func`."""
    return apply_and_concat_one(len(others), select_and_combine, obj, others, combine_func, *args, **kwargs)


@njit
def select_and_combine_nb(i, obj, others, combine_func_nb, *args):
    """A Numba-compiled version of `select_and_combine`.
    
    !!! note
        * `combine_func_nb` must be Numba-compiled
        * `obj`, `others` and `*args` must be Numba-compatible
        * `others` must be strictly homogeneous
        * No support for `**kwargs`
    """
    return combine_func_nb(obj, others[i], *args)


@njit
def combine_and_concat_nb(obj, others, combine_func_nb, *args):
    """A Numba-compiled version of `combine_and_concat`."""
    return apply_and_concat_one_nb(len(others), select_and_combine_nb, obj, others, combine_func_nb, *args)


def combine_multiple(objs, combine_func, *args, **kwargs):
    """Combine `objs` pairwise into a single object."""
    result = None
    for i in range(1, len(objs)):
        if result is None:
            result = combine_func(objs[i - 1], objs[i], *args, **kwargs)
        else:
            result = combine_func(result, objs[i], *args, **kwargs)
    return result


@njit
def combine_multiple_nb(objs, combine_func_nb, *args):
    """A Numba-compiled version of `combine_multiple`.
    
    !!! note
        * `combine_func_nb` must be Numba-compiled
        * `objs` and `*args` must be Numba-compatible
        * `objs` must be strictly homogeneous
        * No support for `**kwargs`
    """
    result = None
    for i in range(1, len(objs)):
        if result is None:
            result = combine_func_nb(objs[i - 1], objs[i], *args)
        else:
            result = combine_func_nb(result, objs[i], *args)
    return result
