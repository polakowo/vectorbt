"""Utilities for validation during runtime."""

import numpy as np
import pandas as pd
from numba.core.registry import CPUDispatcher
from collections.abc import Iterable
import os

# ############# Checks ############# #


def is_series(arg):
    """Determine whether `arg` is `pd.Series`."""
    return isinstance(arg, pd.Series)


def is_frame(arg):
    """Determine whether `arg` is `pd.DataFrame`."""
    return isinstance(arg, pd.DataFrame)


def is_pandas(arg):
    """Determine whether `arg` is `pd.Series` or `pd.DataFrame`."""
    return is_series(arg) or is_frame(arg)


def is_array(arg):
    """Determine whether `arg` is any of `np.ndarray`, `pd.Series` or `pd.DataFrame`."""
    return is_pandas(arg) or isinstance(arg, np.ndarray)


def is_numba_func(arg):
    """Determine whether `arg` is a Numba-compiled function."""
    if 'NUMBA_DISABLE_JIT' in os.environ:
        if os.environ['NUMBA_DISABLE_JIT'] == '1':
            if arg.__name__.endswith('_nb'):
                return True
    return isinstance(arg, CPUDispatcher)


def is_hashable(arg):
    """Determine whether `arg` can be hashed."""
    try:
        hash(arg)
    except Exception:
        return False
    return True


def is_index_equal(arg1, arg2, strict=True):
    """Determine whether indexes are equal.

    Introduces naming tests on top of `pd.Index.equals`, but still doesn't check for types."""
    if not strict:
        return pd.Index.equals(arg1, arg2)
    if isinstance(arg1, pd.MultiIndex) and isinstance(arg2, pd.MultiIndex):
        if arg1.names != arg2.names:
            return False
    elif isinstance(arg1, pd.MultiIndex) or isinstance(arg2, pd.MultiIndex):
        return False
    else:
        if arg1.name != arg2.name:
            return False
    return pd.Index.equals(arg1, arg2)


def is_default_index(arg):
    """Determine whether index is a basic range."""
    return is_index_equal(arg, pd.RangeIndex(start=0, stop=len(arg), step=1))


def is_equal(arg1, arg2, equality_func):
    """Determine whether two objects are equal."""
    if arg1 is None or arg2 is None:
        if arg1 is not None or arg2 is not None:
            return False
    else:
        if not equality_func(arg1, arg2):
            return False
    return True


def is_namedtuple(x):
    """Determines whether object is an instance of namedtuple."""
    t = type(x)
    b = t.__bases__
    if len(b) != 1 or b[0] != tuple:
        return False
    f = getattr(t, '_fields', None)
    if not isinstance(f, tuple):
        return False
    return all(type(n) == str for n in f)


# ############# Asserts ############# #

def assert_in(arg1, arg2):
    """Raise exception if `arg1` is not in `arg2`."""
    if arg1 not in arg2:
        raise AssertionError(f"{arg1} not found in {arg2}")


def assert_numba_func(func):
    """Raise exception if `func` is not Numba-compiled."""
    if not is_numba_func(func):
        raise AssertionError(f"Function {func} must be Numba compiled")


def assert_not_none(arg):
    """Raise exception if `arg` is None."""
    if arg is None:
        raise AssertionError(f"Cannot be None")


def assert_type(arg, types):
    """Raise exception if `arg` is none of types `types`."""
    if not isinstance(arg, types):
        if isinstance(types, tuple):
            raise AssertionError(f"Type must be one of {types}, not {type(arg)}")
        else:
            raise AssertionError(f"Type must be {types}, not {type(arg)}")


def assert_subclass(arg, classes):
    """Raise exception if `arg` is not a subclass of classes `classes`."""
    if not issubclass(arg, classes):
        if isinstance(classes, tuple):
            raise AssertionError(f"Class must be a subclass of one of {classes}, not {arg}")
        else:
            raise AssertionError(f"Class must be a subclass of {classes}, not {arg}")


def assert_type_equal(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different types."""
    if type(arg1) != type(arg2):
        raise AssertionError(f"Types {type(arg1)} and {type(arg2)} do not match")


def assert_dtype(arg, dtype):
    """Raise exception if `arg` is not of data type `dtype`."""
    if not is_array(arg):
        arg = np.asarray(arg)
    if is_frame(arg):
        for i, col_dtype in enumerate(arg.dtypes):
            if col_dtype != dtype:
                raise AssertionError(f"Data type of column {i} must be {dtype}, not {col_dtype}")
    else:
        if arg.dtype != dtype:
            raise AssertionError(f"Data type must be {dtype}, not {arg.dtype}")


def assert_subdtype(arg, dtype):
    """Raise exception if `arg` is not a sub data type of `dtype`."""
    if not is_array(arg):
        arg = np.asarray(arg)
    if is_frame(arg):
        for i, col_dtype in enumerate(arg.dtypes):
            if not np.issubdtype(col_dtype, dtype):
                raise AssertionError(f"Data type of column {i} must be {dtype}, not {col_dtype}")
    else:
        if not np.issubdtype(arg.dtype, dtype):
            raise AssertionError(f"Data type must be {dtype}, not {arg.dtype}")


def assert_dtype_equal(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different data types."""
    if not is_array(arg1):
        arg1 = np.asarray(arg1)
    if not is_array(arg2):
        arg2 = np.asarray(arg2)
    if is_frame(arg1):
        dtypes1 = arg1.dtypes.to_numpy()
    else:
        dtypes1 = np.asarray([arg1.dtype])
    if is_frame(arg2):
        dtypes2 = arg2.dtypes.to_numpy()
    else:
        dtypes2 = np.asarray([arg2.dtype])
    if len(dtypes1) == len(dtypes2):
        if (dtypes1 == dtypes2).all():
            return
    elif len(np.unique(dtypes1)) == 1 and len(np.unique(dtypes2)) == 1:
        if (np.unique(dtypes1) == np.unique(dtypes2)).all():
            return
    raise AssertionError(f"Data types {dtypes1} and {dtypes2} do not match")


def assert_ndim(arg, ndims):
    """Raise exception if `arg` has a different number of dimensions than `ndims`."""
    if not is_array(arg):
        arg = np.asarray(arg)
    if isinstance(ndims, Iterable):
        if arg.ndim not in ndims:
            raise AssertionError(f"Number of dimensions must be one of {ndims}, not {arg.ndim}")
    else:
        if arg.ndim != ndims:
            raise AssertionError(f"Number of dimensions must be {ndims}, not {arg.ndim}")


def assert_len_equal(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different length.

    Does not transform arguments to NumPy arrays."""
    if len(arg1) != len(arg2):
        raise AssertionError(f"Lengths {len(arg1)} and {len(arg2)} do not match")


def assert_shape_equal(arg1, arg2, axis=None):
    """Raise exception if `arg1` and `arg2` have different shapes along `axis`."""
    if not is_array(arg1):
        arg1 = np.asarray(arg1)
    if not is_array(arg2):
        arg2 = np.asarray(arg2)
    if axis is None:
        if arg1.shape != arg2.shape:
            raise AssertionError(f"Shapes {arg1.shape} and {arg2.shape} do not match")
    else:
        if isinstance(axis, tuple):
            if arg1.shape[axis[0]] != arg2.shape[axis[1]]:
                raise AssertionError(
                    f"Axis {axis[0]} of {arg1.shape} and axis {axis[1]} of {arg2.shape} do not match")
        else:
            if arg1.shape[axis] != arg2.shape[axis]:
                raise AssertionError(f"Axis {axis} of {arg1.shape} and {arg2.shape} do not match")


def assert_index_equal(arg1, arg2, **kwargs):
    """Raise exception if `arg1` and `arg2` have different index/columns."""
    if not is_index_equal(arg1, arg2, **kwargs):
        raise AssertionError(f"Indexes {arg1} and {arg2} do not match")


def assert_meta_equal(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different metadata."""
    assert_type_equal(arg1, arg2)
    assert_shape_equal(arg1, arg2)
    if is_pandas(arg1):
        assert_index_equal(arg1.index, arg2.index)
        if is_frame(arg1):
            assert_index_equal(arg1.columns, arg2.columns)


def assert_array_equal(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different metadata or values."""
    assert_meta_equal(arg1, arg2)
    if is_pandas(arg1):
        if arg1.equals(arg2):
            return
    else:
        arg1 = np.asarray(arg1)
        arg2 = np.asarray(arg2)
        if np.array_equal(arg1, arg2):
            return
    raise AssertionError(f"Arrays do not match")


def assert_level_not_exists(arg, level_name):
    """Raise exception if index `arg` has level `level_name`."""
    if isinstance(arg, pd.MultiIndex):
        names = arg.names
    else:
        names = [arg.name]
    if level_name in names:
        raise AssertionError(f"Level {level_name} already exists in {names}")


def assert_equal(arg1, arg2):
    """Raise exception if `arg1` and `arg2` are different."""
    if arg1 != arg2:
        raise AssertionError(f"{arg1} and {arg2} do not match")


def assert_dict_valid(arg, lvl_keys):
    """Raise exception if dict `arg` has keys that are not in `lvl_keys`.

    `lvl_keys` should be a list of lists, each corresponding to a level in the dict."""
    if len(lvl_keys) == 0 or not isinstance(lvl_keys[0], (tuple, list)):
        return
    set1 = set(arg.keys())
    set2 = set(lvl_keys[0])
    if not set1.issubset(set2):
        raise AssertionError(f"Not all keys from {set1} are in {set2}")
    for k, v in arg.items():
        if isinstance(v, dict):
            assert_dict_valid(v, lvl_keys[1:])

