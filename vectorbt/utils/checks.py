"""Utilities for validation during runtime."""

import numpy as np
import pandas as pd
from numba.core.registry import CPUDispatcher
from collections.abc import Iterable

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
    return isinstance(arg, CPUDispatcher)


def is_hashable(arg):
    """Determine whether `arg` can be hashed."""
    try:
        hash(arg)
    except Exception:
        return False
    return True


# ############# Asserts ############# #

def assert_value_in(value, lst):
    """Raise exception if `value` is not in `lst`."""
    if value not in lst:
        raise Exception(f"Value {value} is outside of {lst}")


def assert_numba_func(func):
    """Raise exception if `func` is not Numba-compiled."""
    if not is_numba_func(func):
        raise Exception(f"Function {func} must be Numba compiled")


def assert_not_none(arg):
    """Raise exception if `arg` is None."""
    if arg is None:
        raise Exception(f"Cannot be None")


def assert_type(arg, types):
    """Raise exception if `arg` is none of types `types`."""
    if not isinstance(arg, types):
        if isinstance(types, tuple):
            raise Exception(f"Type must be one of {types}, not {type(arg)}")
        else:
            raise Exception(f"Type must be {types}, not {type(arg)}")


def assert_subclass(arg, classes):
    """Raise exception if `arg` is not a subclass of classes `classes`."""
    if not issubclass(arg, classes):
        if isinstance(classes, tuple):
            raise Exception(f"Class must be a subclass of one of {classes}, not {arg}")
        else:
            raise Exception(f"Class must be a subclass of {classes}, not {arg}")


def assert_same_type(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different types."""
    if type(arg1) != type(arg2):
        raise Exception(f"Types {type(arg1)} and {type(arg2)} do not match")


def assert_dtype(arg, dtype):
    """Raise exception if `arg` is not of data type `dtype`."""
    if not is_array(arg):
        arg = np.asarray(arg)
    if is_frame(arg):
        if (arg.dtypes != dtype).any():
            raise Exception(f"Data type must be {dtype}, not {arg.dtypes}")
    else:
        if arg.dtype != dtype:
            raise Exception(f"Data type must be {dtype}, not {arg.dtype}")


def assert_same_dtype(arg1, arg2):
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
    raise Exception(f"Data types {dtypes1} and {dtypes2} do not match")


def assert_ndim(arg, ndims):
    """Raise exception if `arg` has a different number of dimensions than `ndims`."""
    if not is_array(arg):
        arg = np.asarray(arg)
    if isinstance(ndims, Iterable):
        if arg.ndim not in ndims:
            raise Exception(f"Number of dimensions must be one of {ndims}, not {arg.ndim}")
    else:
        if arg.ndim != ndims:
            raise Exception(f"Number of dimensions must be {ndims}, not {arg.ndim}")


def assert_same_len(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different length.

    Does not transform arguments to NumPy arrays."""
    if len(arg1) != len(arg2):
        raise Exception(f"Lengths {len(arg1)} and {len(arg2)} do not match")


def assert_same_shape(arg1, arg2, axis=None):
    """Raise exception if `arg1` and `arg2` have different shapes along `axis`."""
    if not is_array(arg1):
        arg1 = np.asarray(arg1)
    if not is_array(arg2):
        arg2 = np.asarray(arg2)
    if axis is None:
        if arg1.shape != arg2.shape:
            raise Exception(f"Shapes {arg1.shape} and {arg2.shape} do not match")
    else:
        if isinstance(axis, tuple):
            if arg1.shape[axis[0]] != arg2.shape[axis[1]]:
                raise Exception(
                    f"Axis {axis[0]} of {arg1.shape} and axis {axis[1]} of {arg2.shape} do not match")
        else:
            if arg1.shape[axis] != arg2.shape[axis]:
                raise Exception(f"Axis {axis} of {arg1.shape} and {arg2.shape} do not match")


def assert_same_index(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different index."""
    if not pd.Index.equals(arg1.index, arg2.index):
        raise Exception(f"Indexes {arg1.index} and {arg2.index} do not match")


def assert_same_columns(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different columns."""
    if not pd.Index.equals(arg1.columns, arg2.columns):
        raise Exception(f"Columns {arg1.columns} and {arg2.columns} do not match")


def assert_same_meta(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different metadata."""
    assert_same_type(arg1, arg2)
    assert_same_shape(arg1, arg2)
    if is_pandas(arg1):
        assert_same_index(arg1, arg2)
        if is_frame(arg1):
            assert_same_columns(arg1, arg2)


def assert_same(arg1, arg2):
    """Raise exception if `arg1` and `arg2` have different metadata or values."""
    assert_same_meta(arg1, arg2)
    if is_pandas(arg1):
        if arg1.equals(arg2):
            return
    else:
        arg1 = np.asarray(arg1)
        arg2 = np.asarray(arg2)
        if np.array_equal(arg1, arg2):
            return
    raise Exception(f"Values do not match")


def assert_level_not_exists(arg, level_name):
    """Raise exception if index `arg` has level `level_name`."""
    if isinstance(arg, pd.MultiIndex):
        names = arg.names
    else:
        names = [arg.name]
    if level_name in names:
        raise Exception(f"Level {level_name} already exists in {names}")
