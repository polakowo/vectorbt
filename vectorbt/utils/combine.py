import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List

from vectorbt.utils import reshape


def apply_and_concat_one(n, apply_func, *args, **kwargs):
    """For a range from 0 to n, apply a function and concat the results horizontally."""
    return np.hstack([reshape.to_2d(apply_func(i, *args, **kwargs)) for i in range(n)])


@njit
def to_2d_one_nb(a):
    if a.ndim > 1:
        return a
    return np.expand_dims(a, axis=1)


@njit
def apply_and_concat_one_nb(n, apply_func_nb, *args):  # numba doesn't accepts **kwargs
    output_0 = to_2d_one_nb(apply_func_nb(0, *args))
    output = np.empty((output_0.shape[0], n * output_0.shape[1]), dtype=output_0.dtype)
    for i in range(n):
        if i == 0:
            outputs_i = output_0
        else:
            outputs_i = to_2d_one_nb(apply_func_nb(i, *args))
        output[:, i*outputs_i.shape[1]:(i+1)*outputs_i.shape[1]] = outputs_i
    return output


def apply_and_concat_multiple(n, apply_func, *args, **kwargs):
    outputs = [tuple(map(reshape.to_2d, apply_func(i, *args, **kwargs))) for i in range(n)]
    return list(map(np.hstack, list(zip(*outputs))))


@njit
def to_2d_multiple_nb(a):
    lst = List()
    for _a in a:
        lst.append(to_2d_one_nb(_a))
    return lst


@njit
def apply_and_concat_multiple_nb(n, apply_func_nb, *args):  # numba doesn't accepts **kwargs
    # NOTE: apply_func_nb must return a homogeneous tuple!
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
            outputs[j][:, i*outputs_i[j].shape[1]:(i+1)*outputs_i[j].shape[1]] = outputs_i[j]
    return outputs


def apply_and_concat(obj, n, apply_func, *args, **kwargs):
    return apply_and_concat_one(n, apply_func, obj, *args, **kwargs)


@njit
def apply_and_concat_nb(obj, n, apply_func_nb, *args):
    return apply_and_concat_one_nb(n, apply_func_nb, obj, *args)


def select_and_combine(i, obj, others, combine_func, *args, **kwargs):
    return combine_func(obj, others[i], *args, **kwargs)


def combine_and_concat(obj, others, combine_func, *args, **kwargs):
    """For each element in others, combine obj and other element and concat the results horizontally."""
    return apply_and_concat(obj, len(others), select_and_combine, others, combine_func, *args, **kwargs)


@njit
def select_and_combine_nb(i, obj, others, combine_func_nb, *args):
    # NOTE: others must be homogeneuous!
    return combine_func_nb(obj, others[i], *args)


@njit
def combine_and_concat_nb(obj, others, combine_func_nb, *args):
    return apply_and_concat_nb(obj, len(others), select_and_combine_nb, others, combine_func_nb, *args)


def combine_multiple(objs, combine_func, *args, **kwargs):
    """Combine a list of objects pairwise."""
    result = None
    for i in range(1, len(objs)):
        if result is None:
            result = combine_func(objs[i-1], objs[i], *args, **kwargs)
        else:
            result = combine_func(result, objs[i], *args, **kwargs)
    return result


@njit
def combine_multiple_nb(objs, combine_func_nb, *args):
    # NOTE: each time combine_func_nb must return the array of the same type!
    # Also NOTE: objs must all have the same type and arrays in the same memory order!
    result = None
    for i in range(1, len(objs)):
        if result is None:
            result = combine_func_nb(objs[i-1], objs[i], *args)
        else:
            result = combine_func_nb(result, objs[i], *args)
    return result
