"""Functions for combining arrays.

Combine functions combine two or more NumPy arrays using a custom function. The emphasis here is
done upon stacking the results into one NumPy array - since vectorbt is all about brute-forcing
large spaces of hyperparameters, concatenating the results of each hyperparameter combination into
a single DataFrame is important. All functions are available in both Python and Numba-compiled form."""

import numpy as np
from numba import njit
from numba.typed import List
from tqdm import tqdm

from vectorbt.base import reshape_fns


def apply_and_concat_none(n, apply_func, *args, show_progress=False, **kwargs):
    """For each value `i` from 0 to `n`, apply `apply_func` with arguments `*args` and `**kwargs`,
    and output nothing. Meant for in-place outputs.

    `apply_func` must accept arguments `i`, `*args` and `**kwargs`."""
    for i in tqdm(range(n), disable=not show_progress):
        apply_func(i, *args, **kwargs)


@njit
def apply_and_concat_none_nb(n, apply_func_nb, *args):
    """A Numba-compiled version of `apply_and_concat_none`.

    !!! note
        * `apply_func_nb` must be Numba-compiled
        * `*args` must be Numba-compatible
        * No support for `**kwargs`
    """
    for i in range(n):
        apply_func_nb(i, *args)


def apply_and_concat_one(n, apply_func, *args, show_progress=False, **kwargs):
    """For each value `i` from 0 to `n`, apply `apply_func` with arguments `*args` and `**kwargs`, 
    and concat the results along axis 1. 
    
    The result of `apply_func` must be a single 1-dim or 2-dim array.
    
    `apply_func` must accept arguments `i`, `*args` and `**kwargs`."""
    outputs = []
    for i in tqdm(range(n), disable=not show_progress):
        outputs.append(reshape_fns.to_2d(apply_func(i, *args, **kwargs)))
    return np.column_stack(outputs)


@njit
def to_2d_one_nb(a):
    """Expand the dimensions of array `a` along axis 1.
    
    !!! note
        * `a` must be strictly homogeneous"""
    if a.ndim > 1:
        return a
    return np.expand_dims(a, axis=1)


@njit
def apply_and_concat_one_nb(n, apply_func_nb, *args):
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


def apply_and_concat_multiple(n, apply_func, *args, show_progress=False, **kwargs):
    """Identical to `apply_and_concat_one`, except that the result of `apply_func` must be 
    multiple 1-dim or 2-dim arrays. Each of these arrays at `i` will be concatenated with the
    array at the same position at `i+1`."""
    outputs = []
    for i in tqdm(range(n), disable=not show_progress):
        outputs.append(tuple(map(reshape_fns.to_2d, apply_func(i, *args, **kwargs))))
    return list(map(np.column_stack, list(zip(*outputs))))


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


def ray_apply(n, apply_func, *args, ray_force_init=False, ray_func_kwargs=None,
              ray_init_kwargs=None, ray_shutdown=False, **kwargs):
    """Run `apply_func` in distributed manner.

    Set `ray_reinit` to True to terminate the Ray runtime and initialize a new one.
    `ray_func_kwargs` will be passed to `ray.remote` and `ray_init_kwargs` to `ray.init`.
    Set `ray_shutdown` to True to terminate the Ray runtime upon the job end.

    """
    import ray

    if ray_init_kwargs is None:
        ray_init_kwargs = {}
    if ray_func_kwargs is None:
        ray_func_kwargs = {}
    if ray_force_init:
        if ray.is_initialized():
            ray.shutdown()
    if not ray.is_initialized():
        ray.init(**ray_init_kwargs)
    if len(ray_func_kwargs) > 0:
        apply_func = ray.remote(**ray_func_kwargs)(apply_func)
    else:
        apply_func = ray.remote(apply_func)
    # args and kwargs don't change -> put to object store
    arg_refs = ()
    for v in args:
        arg_refs += (ray.put(v),)
    kwarg_refs = {}
    for k, v in kwargs.items():
        kwarg_refs[k] = ray.put(v)
    futures = [apply_func.remote(i, *arg_refs, **kwarg_refs) for i in range(n)]
    results = ray.get(futures)
    if ray_shutdown:
        ray.shutdown()
    return results


def apply_and_concat_one_ray(*args, **kwargs):
    """Distributed version of `apply_and_concat_one`."""
    results = ray_apply(*args, **kwargs)
    return np.column_stack(list(map(reshape_fns.to_2d, results)))


def apply_and_concat_multiple_ray(*args, **kwargs):
    """Distributed version of `apply_and_concat_multiple`."""
    results = ray_apply(*args, **kwargs)
    return list(map(np.column_stack, list(zip(*results))))


def combine_and_concat_ray(obj, others, combine_func, *args, **kwargs):
    """Distributed version of `combine_and_concat`."""
    return apply_and_concat_one_ray(len(others), select_and_combine, obj, others, combine_func, *args, **kwargs)
