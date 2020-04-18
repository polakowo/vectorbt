import numpy as np
import pandas as pd
from functools import wraps, reduce, update_wrapper
from inspect import signature, Parameter
import types
import itertools
from collections.abc import Iterable
import numba
from numba import njit, literal_unroll
from numba.typed import List

# ############# Configuration ############# #


class Config(dict):
    """A simple dict with frozen keys."""

    def __init__(self, *args, **kwargs):
        self.update(*args, **kwargs)
        self.default_config = dict(self)
        for key, value in dict.items(self):
            if isinstance(value, dict):
                dict.__setitem__(self, key, Config(value))

    def __setitem__(self, key, val):
        if key not in self:
            raise KeyError(f"Key {key} is not a valid parameter")
        dict.__setitem__(self, key, val)

    def reset(self):
        self.update(self.default_config)


# You can change defaults from code
# Useful for magic methods that cannot accept keyword arguments
defaults = Config(
    broadcast=dict(
        index_from='strict',
        columns_from='stack'
    ),
    broadcast_to=dict(
        index_from=1,
        columns_from=1
    ),
    drop_duplicates=True,
    drop_redundant=True,
    keep='last'
)

# ############# Checks ############# #


def is_series(arg):
    return isinstance(arg, pd.Series)


def is_frame(arg):
    return isinstance(arg, pd.DataFrame)


def is_pandas(arg):
    return is_series(arg) or is_frame(arg)


def is_array(arg):
    return isinstance(arg, np.ndarray)


def is_array_like(arg):
    return is_pandas(arg) or is_array(arg)


def is_numba_func(arg):
    return isinstance(arg, numba.targets.registry.CPUDispatcher)


def check_not_none(arg):
    if arg is None:
        raise TypeError(f"Cannot be None")


def check_type(arg, types):
    if not isinstance(arg, types):
        if isinstance(types, tuple):
            raise TypeError(f"Type must be one of {types}, not {type(arg)}")
        else:
            raise TypeError(f"Type must be {types}, not {type(arg)}")


def check_not_type(arg, types):
    if isinstance(arg, types):
        if isinstance(types, tuple):
            raise TypeError(f"Type cannot be any of {types}")
        else:
            raise TypeError(f"Type cannot be {types}")


def check_same_type(arg1, arg2):
    if type(arg1) != type(arg2):
        raise TypeError(f"Types {type(arg1)} and {type(arg2)} do not match")


def check_dtype(arg, dtype):
    if is_frame(arg):
        if (arg.dtypes != dtype).any():
            raise ValueError(f"Data type must be {dtype}, not {arg.dtypes}")
    else:
        if arg.dtype != dtype:
            raise ValueError(f"Data type must be {dtype}, not {arg.dtype}")


def check_same_dtype(arg1, arg2):
    if not is_array_like(arg1):
        arg1 = np.asarray(arg1)
    if not is_array_like(arg2):
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
    raise ValueError(f"Data types {dtypes1} and {dtypes2} do not match")


def check_ndim(arg, ndims):
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if isinstance(ndims, tuple):
        if arg.ndim not in ndims:
            raise ValueError(f"Number of dimensions must be one of {ndims}, not {arg.ndim}")
    else:
        if arg.ndim != ndims:
            raise ValueError(f"Number of dimensions must be {ndims}, not {arg.ndim}")


def check_same_len(arg1, arg2):
    if len(arg1) != len(arg2):
        raise ValueError(f"Lengths {len(arg1)} and {len(arg2)} do not match")


def check_same_shape(arg1, arg2, along_axis=None):
    if not is_array_like(arg1):
        arg1 = np.asarray(arg1)
    if not is_array_like(arg2):
        arg2 = np.asarray(arg2)
    if along_axis is None:
        if arg1.shape != arg2.shape:
            raise ValueError(f"Shapes {arg1.shape} and {arg2.shape} do not match")
    else:
        if isinstance(along_axis, tuple):
            if arg1.shape[along_axis[0]] != arg2.shape[along_axis[1]]:
                raise ValueError(
                    f"Axis {along_axis[0]} of {arg1.shape} and axis {along_axis[1]} of {arg2.shape} do not match")
        else:
            if arg1.shape[along_axis] != arg2.shape[along_axis]:
                raise ValueError(f"Axis {along_axis} of {arg1.shape} and {arg2.shape} do not match")


def check_same_index(arg1, arg2):
    if not pd.Index.equals(arg1.index, arg2.index):
        raise ValueError(f"Indices {arg1.index} and {arg2.index} do not match")


def check_same_columns(arg1, arg2):
    if not pd.Index.equals(arg1.columns, arg2.columns):
        raise ValueError(f"Columns {arg1.columns} and {arg2.columns} do not match")


def check_same_meta(arg1, arg2, check_dtype=True):
    check_same_type(arg1, arg2)
    check_same_shape(arg1, arg2)
    if is_pandas(arg1) or is_pandas(arg2):
        check_same_index(arg1, arg2)
        check_same_columns(to_2d(arg1), to_2d(arg2))
    if is_array_like(arg1) or is_array_like(arg2):
        if check_dtype:
            check_same_dtype(arg1, arg2)


def check_same(arg1, arg2):
    check_same_meta(arg1, arg2)
    if is_pandas(arg1):
        if arg1.equals(arg2):
            return
    else:
        arg1 = np.asarray(arg1)
        arg2 = np.asarray(arg2)
        if np.array_equal(arg1, arg2):
            return
    raise ValueError(f"Values do not match")


def check_level_not_exists(arg, level_name):
    if not is_frame(arg):
        return
    if isinstance(arg.columns, pd.MultiIndex):
        names = arg.columns.names
    else:
        names = [arg.columns.name]
    if level_name in names:
        raise ValueError(f"Level {level_name} already exists in {names}")

# ############# Index and columns ############# #


def index_from_values(values, name=None, value_names=None):
    """Create index using array of values."""
    if value_names is not None:
        check_same_shape(values, value_names, along_axis=0)
        return pd.Index(value_names, name=name)  # just return the names
    value_names = []
    for i, v in enumerate(values):
        if not is_array(v):
            v = np.asarray(v)
        if np.all(v == v.item(0)):
            value_names.append(v.item(0))
        else:
            value_names.append('mix_%d' % i)
    return pd.Index(value_names, name=name)


def drop_redundant_levels(index):
    """Drop levels that have a single value."""
    if not isinstance(index, pd.Index):
        index = pd.Index(index)
    if len(index) == 1:
        return index

    if isinstance(index, pd.MultiIndex):
        levels_to_drop = []
        for i, level in enumerate(index.levels):
            if len(level) == 1:
                levels_to_drop.append(i)
            elif level.name is None and (level == np.arange(len(level))).all():  # basic range
                if len(index.get_level_values(i)) == len(level):
                    levels_to_drop.append(i)
        # Remove redundant levels only if there are some non-redundant levels left
        if len(levels_to_drop) < len(index.levels):
            return index.droplevel(levels_to_drop)
    return index


def drop_duplicate_levels(index, keep='default'):
    """Drop duplicate levels with the same name and values."""
    if isinstance(index, pd.Index) and not isinstance(index, pd.MultiIndex):
        return index
    check_type(index, pd.MultiIndex)

    levels = []
    levels_to_drop = []
    if keep == 'default':
        keep = defaults['keep']
    if keep == 'first':
        r = range(0, len(index.levels))
    elif keep == 'last':
        r = range(len(index.levels)-1, -1, -1)  # loop backwards
    for i in r:
        level = (index.levels[i].name, tuple(index.get_level_values(i).to_numpy().tolist()))
        if level not in levels:
            levels.append(level)
        else:
            levels_to_drop.append(i)
    return index.droplevel(levels_to_drop)


def clean_index(index, drop_duplicates='default', drop_redundant='default', **kwargs):
    """Clean index from redundant and/or duplicate levels."""
    if drop_duplicates == 'default':
        drop_duplicates = defaults['drop_duplicates']
    if drop_redundant == 'default':
        drop_redundant = defaults['drop_redundant']

    if drop_duplicates:
        index = drop_duplicate_levels(index, **kwargs)
    if drop_redundant:
        index = drop_redundant_levels(index)
    return index


def repeat_index(index, n):
    """Repeat each element in index n times."""
    if not isinstance(index, pd.Index):
        index = pd.Index(index)

    return np.repeat(index, n)


def tile_index(index, n):
    """Tile the whole index n times."""
    if not isinstance(index, pd.Index):
        index = pd.Index(index)

    if isinstance(index, pd.MultiIndex):
        return pd.MultiIndex.from_tuples(np.tile(index, n), names=index.names)
    return pd.Index(np.tile(index, n), name=index.name)


def stack_indices(index1, index2):
    """Stack indices."""
    check_same_shape(index1, index2)
    if not isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_arrays([index1])
    if not isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_arrays([index2])

    levels = []
    for i in range(len(index1.names)):
        levels.append(index1.get_level_values(i))
    for i in range(len(index2.names)):
        levels.append(index2.get_level_values(i))

    new_index = pd.MultiIndex.from_arrays(levels)
    return new_index


def combine_indices(index1, index2):
    """Combine indices using Cartesian product."""
    if not isinstance(index1, pd.Index):
        index1 = pd.Index(index1)
    if not isinstance(index2, pd.Index):
        index2 = pd.Index(index2)

    if len(index1) == 1:
        return index2
    elif len(index2) == 1:
        return index1

    tuples1 = np.repeat(index1.to_numpy(), len(index2))
    tuples2 = np.tile(index2.to_numpy(), len(index1))

    if isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_tuples(tuples1, names=index1.names)
    else:
        index1 = pd.Index(tuples1, name=index1.name)
    if isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_tuples(tuples2, names=index2.names)
    else:
        index2 = pd.Index(tuples2, name=index2.name)

    return stack_indices(index1, index2)


def drop_levels(index, levels):
    """Drop levels from index."""
    check_type(index, pd.MultiIndex)

    levels_to_drop = []
    if not isinstance(levels, (tuple, list)):
        levels = [levels]
    for level in levels:
        if level in index.names:
            levels_to_drop.append(level)
    if len(levels_to_drop) < len(index.names):
        # Drop only if there will be some indices left
        return index.droplevel(levels_to_drop)
    return index


def rename_levels(index, name_dict):
    """Rename index/column levels."""
    for k, v in name_dict.items():
        if isinstance(index, pd.MultiIndex):
            if k in index.names:
                index = index.rename(v, level=k)
        else:
            if index.name == k:
                index.name = v
    return index


# ############# Broadcasting ############# #

def soft_broadcast_to_ndim(arg, ndim):
    """Try to softly bring the argument to the specified number of dimensions (max 2)."""
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if ndim == 1:
        if arg.ndim == 2:
            if arg.shape[1] == 1:
                if is_pandas(arg):
                    return arg.iloc[:, 0]
                return arg[:, 0]  # downgrade
    if ndim == 2:
        if arg.ndim == 1:
            if is_pandas(arg):
                return arg.to_frame()
            return arg[:, None]  # upgrade
    return arg  # do nothing


def wrap_array(arg, index=None, columns=None, dtype=None, default_index=None, default_columns=None, to_ndim=None):
    """Wrap array into a series/dataframe."""
    if not is_array(arg):
        arg = np.asarray(arg)
    if to_ndim is not None:
        arg = soft_broadcast_to_ndim(arg, to_ndim)
    if index is None:
        index = default_index
    if columns is None:
        columns = default_columns
    if columns is not None and len(columns) == 1:
        name = columns[0]
    else:
        name = None

    # Perform checks
    if index is not None:
        check_same_shape(arg, index, along_axis=(0, 0))
    if arg.ndim == 2 and columns is not None:
        check_same_shape(arg, columns, along_axis=(1, 0))

    if arg.ndim == 1:
        return pd.Series(arg, index=index, name=name, dtype=dtype)
    return pd.DataFrame(arg, index=index, columns=columns, dtype=dtype)


def to_1d(arg, raw=False):
    """Reshape argument to one dimension."""
    if raw:
        arg = np.asarray(arg)
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        if arg.shape[1] == 1:
            if is_frame(arg):
                return arg.iloc[:, 0]
            return arg[:, 0]
    if arg.ndim == 1:
        return arg
    elif arg.ndim == 0:
        return arg.reshape((1,))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


def to_2d(arg, raw=False, expand_axis=1):
    """Reshape argument to two dimensions."""
    if raw:
        arg = np.asarray(arg)
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        if is_series(arg):
            if expand_axis == 0:
                return pd.DataFrame(arg.values[None, :], columns=arg.index)
            elif expand_axis == 1:
                return arg.to_frame()
        return np.expand_dims(arg, expand_axis)
    elif arg.ndim == 0:
        return arg.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def tile(arg, n, along_axis=1):
    """Tile array n times along specified axis."""
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if along_axis == 0:
        if arg.ndim == 1:
            if is_pandas(arg):
                return arg.vbt.wrap_array(
                    np.tile(arg.values, n),
                    index=tile_index(arg.index, n))
            return np.tile(arg, n)
        if arg.ndim == 2:
            if is_pandas(arg):
                return arg.vbt.wrap_array(
                    np.tile(arg.values, (n, 1)),
                    index=tile_index(arg.index, n))
            return np.tile(arg, (n, 1))
    elif along_axis == 1:
        arg = to_2d(arg)
        if is_pandas(arg):
            return arg.vbt.wrap_array(
                np.tile(arg.values, (1, n)),
                columns=tile_index(arg.columns, n))
        return np.tile(arg, (1, n))
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def repeat(arg, n, along_axis=1):
    """Repeat array n times along specified axis."""
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if along_axis == 0:
        if is_pandas(arg):
            return arg.vbt.wrap_array(
                np.repeat(arg.values, n, axis=0),
                index=repeat_index(arg.index, n))
        return np.repeat(arg, n, axis=0)
    elif along_axis == 1:
        arg = to_2d(arg)
        if is_pandas(arg):
            return arg.vbt.wrap_array(
                np.repeat(arg.values, n, axis=1),
                columns=repeat_index(arg.columns, n))
        return np.repeat(arg, n, axis=1)
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def align_index_to(index1, index2):
    """Align the first index to the second one. 

    Returns integer indices of occurrences and None if aligning not needed.

    The second one must contain all levels from the first (and some more)
    In all these levels, both must share the same elements.
    Only then the first index can be broadcasted to the match the shape of the second one."""
    if not isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_arrays([index1])
    if not isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_arrays([index2])
    if index1.duplicated().any():
        raise ValueError("Duplicates index values are not allowed for the first index")

    if pd.Index.equals(index1, index2):
        return pd.IndexSlice[:]
    if len(index1) <= len(index2):
        if len(index1) == 1:
            return pd.IndexSlice[np.tile([0])]
        js = []
        for i in range(len(index1.names)):
            for j in range(len(index2.names)):
                if index1.names[i] == index2.names[j]:
                    if np.array_equal(index1.levels[i], index2.levels[j]):
                        js.append(j)
                        break
        if len(index1.names) == len(js):
            new_index = pd.MultiIndex.from_arrays([index2.get_level_values(j) for j in js])
            xsorted = np.argsort(index1)
            ypos = np.searchsorted(index1[xsorted], new_index)
            return pd.IndexSlice[xsorted[ypos]]

    raise ValueError("Indices could not be aligned together")


def broadcast_index(*args, index_from=None, axis=0, is_2d=False, **kwargs):
    """Broadcast index/columns of all arguments."""
    index_str = 'columns' if axis == 1 else 'index'
    new_index = None
    if index_from is not None:
        if isinstance(index_from, int):
            # Take index/columns of the object indexed by index_from
            if axis == 1:
                new_index = to_2d(args[index_from]).columns
            else:
                new_index = args[index_from].index
        elif isinstance(index_from, str) and index_from in ('stack', 'strict'):
            # If pandas objects have different index/columns, stack them together
            # maxlen stores the length of the longest index
            max_shape = np.lib.stride_tricks._broadcast_shape(*args)
            if axis == 1 and len(max_shape) == 1:
                max_shape = (max_shape[0], 1)
            maxlen = max_shape[1] if axis == 1 else max_shape[0]
            for arg in args:
                if is_pandas(arg):
                    if is_series(arg):
                        arg = arg.to_frame()  # series name counts as a column
                    index = arg.columns if axis == 1 else arg.index
                    if new_index is None:
                        new_index = index
                    else:
                        if index_from == 'strict':
                            # If pandas objects have different index/columns, raise an exception
                            if not pd.Index.equals(index, new_index):
                                raise ValueError(f"Broadcasting {index_str} is not allowed for {index_str}_from=strict")
                        # Broadcasting index must follow the rules of a regular broadcasting operation
                        # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
                        # 1. rule: if indices are of the same length, they are simply stacked
                        # 2. rule: if index has one element, it gets repeated and then stacked

                        if pd.Index.equals(index, new_index):
                            continue
                        if len(index) != len(new_index):
                            if len(index) > 1 and len(new_index) > 1:
                                raise ValueError("Indices could not be broadcast together")
                            if len(index) > len(new_index):
                                new_index = repeat_index(new_index, len(index))
                            elif len(index) < len(new_index):
                                index = repeat_index(index, len(new_index))
                        new_index = clean_index(stack_indices(new_index, index), **kwargs)
            if maxlen > len(new_index):
                if index_from == 'strict':
                    raise ValueError(f"Broadcasting {index_str} is not allowed for {index_str}_from=strict")
                # This happens only when some numpy object is longer than the new pandas index
                # In this case, new pandas index (one element) should be repeated to match this length.
                if maxlen > 1 and len(new_index) > 1:
                    raise ValueError("Indices could not be broadcast together")
                new_index = repeat_index(new_index, maxlen)
        else:
            raise ValueError(f"Invalid value {index_from} for {'columns' if axis == 1 else 'index'}_from")
    return new_index


def wrap_broadcasted(old_arg, new_arg, is_pd=False, new_index=None, new_columns=None):
    """Transform newly broadcasted array to match the type of the original object."""
    if is_pd:
        if is_pandas(old_arg):
            if new_index is None:
                # Take index from original pandas object
                if old_arg.shape[0] == new_arg.shape[0]:
                    new_index = old_arg.index
                else:
                    new_index = repeat_index(old_arg.index, new_arg.shape[0])
            if new_columns is None:
                # Take columns from original pandas object
                if new_arg.ndim == 2:
                    if is_series(old_arg):
                        old_arg = old_arg.to_frame()
                    if old_arg.shape[1] == new_arg.shape[1]:
                        new_columns = old_arg.columns
                    else:
                        new_columns = repeat_index(old_arg.columns, new_arg.shape[1])
        else:
            if new_index is None and new_columns is None:
                # Return plain numpy array if not pandas and no rules set
                return new_arg
        return wrap_array(new_arg, index=new_index, columns=new_columns)
    return new_arg


def is_broadcasting_needed(*args):
    """Broadcasting may be expensive, do we really need it?"""
    args = list(args)
    shapes = []
    for i in range(len(args)):
        if not is_array_like(args[i]):
            args[i] = np.asarray(args[i])
        shapes.append(args[i].shape)
    return len(set(shapes)) > 1


def broadcast(*args, index_from='default', columns_from='default', writeable=False, copy_kwargs={}, **kwargs):
    """Bring multiple arguments to the same shape."""
    is_pd = False
    is_2d = False
    args = list(args)

    # Convert to np.ndarray object if not numpy or pandas
    for i in range(len(args)):
        if not is_array_like(args[i]):
            args[i] = np.asarray(args[i])
        if args[i].ndim > 1:
            is_2d = True
        if is_pandas(args[i]):
            is_pd = True

    if is_pd:
        # Convert all pd.Series objects to pd.DataFrame
        if is_2d:
            for i in range(len(args)):
                if is_series(args[i]):
                    args[i] = args[i].to_frame()

        # Decide on index and columns
        if index_from == 'default':
            index_from = defaults['broadcast']['index_from']
        if columns_from == 'default':
            columns_from = defaults['broadcast']['columns_from']
        new_index = broadcast_index(*args, index_from=index_from, axis=0, is_2d=is_2d, **kwargs)
        new_columns = broadcast_index(*args, index_from=columns_from, axis=1, is_2d=is_2d, **kwargs)
    else:
        new_index, new_columns = None, None

    # Perform broadcasting operation if needed
    if is_broadcasting_needed(*args):
        new_args = np.broadcast_arrays(*args, subok=True)
        # The problem is that broadcasting creates readonly objects and numba requires writable ones.
        # So we have to copy all of them, which is ok for small-sized arrays and not ok for large ones.

        # copy kwarg is only applied when broadcasting was done to avoid deprecation warnings
        # NOTE: If copy=False, then the resulting arrays will be readonly in the future!
        new_args = list(map(lambda x: np.array(x, copy=writeable, **copy_kwargs), new_args))
    else:
        # No copy here, just pandas -> numpy and any order to contiguous
        new_args = list(map(lambda x: np.array(x, copy=False, **copy_kwargs), args))

    # Bring arrays to their old types (e.g. array -> pandas)
    for i in range(len(new_args)):
        new_args[i] = wrap_broadcasted(args[i], new_args[i], is_pd=is_pd, new_index=new_index, new_columns=new_columns)

    return tuple(new_args)


def broadcast_to(arg1, arg2, index_from='default', columns_from='default', writeable=False, copy_kwargs={}, raw=False, **kwargs):
    """Bring first argument to the shape of second argument. 

    Closely resembles the other broadcast function."""
    if not is_array_like(arg1):
        arg1 = np.asarray(arg1)
    if not is_array_like(arg2):
        arg2 = np.asarray(arg2)

    is_2d = arg1.ndim > 1 or arg2.ndim > 1
    is_pd = is_pandas(arg1) or is_pandas(arg2)

    if is_pd:
        if is_2d:
            if is_series(arg1):
                arg1 = arg1.to_frame()
            if is_series(arg2):
                arg2 = arg2.to_frame()

        if index_from == 'default':
            index_from = defaults['broadcast_to']['index_from']
        if columns_from == 'default':
            columns_from = defaults['broadcast_to']['columns_from']
        new_index = broadcast_index(arg1, arg2, index_from=index_from, axis=0, is_2d=is_2d, **kwargs)
        new_columns = broadcast_index(arg1, arg2, index_from=columns_from, axis=1, is_2d=is_2d, **kwargs)
    else:
        new_index, new_columns = None, None

    if is_broadcasting_needed(arg1, arg2):
        arg1_new = np.broadcast_to(arg1, arg2.shape, subok=True)
        arg1_new = np.array(arg1_new, copy=writeable, **copy_kwargs)
    else:
        arg1_new = np.array(arg1, copy=False, **copy_kwargs)
    return wrap_broadcasted(arg1, arg1_new, is_pd=is_pd, new_index=new_index, new_columns=new_columns)


def broadcast_to_array_of(arg1, arg2):
    """Bring first argument to the shape of an array of second argument."""
    arg1 = np.asarray(arg1)
    arg2 = np.asarray(arg2)
    if arg1.ndim == arg2.ndim + 1:
        if arg1.shape[1:] == arg2.shape:
            return arg1
    # From here on arg1 can be only a 1-dim array
    if arg1.ndim == 0:
        arg1 = to_1d(arg1)
    check_ndim(arg1, 1)

    if arg2.ndim == 0:
        return arg1
    for i in range(arg2.ndim):
        arg1 = np.expand_dims(arg1, axis=-1)
    return np.tile(arg1, (1, *arg2.shape))

# ############# Indexing ############# #


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def add_indexing(indexing_func):
    def wrapper(cls):
        """Add iloc, loc and plain brackets indexing to a class.

        Each indexing operation is forwarded to the underlying pandas objects and
        a new instance of the class with new pandas objects is created using the indexing_func."""

        class iLoc:
            def __init__(self, obj):
                self.obj = obj

            def __getitem__(self, key):
                return indexing_func(self.obj, lambda x: x.iloc.__getitem__(key))

        class Loc:
            def __init__(self, obj):
                self.obj = obj

            def __getitem__(self, key):
                return indexing_func(self.obj, lambda x: x.loc.__getitem__(key))

        @property
        def iloc(self):
            return self._iloc

        @property
        def loc(self):
            return self._loc

        def xs(self, *args, **kwargs):
            return indexing_func(self, lambda x: x.xs(*args, **kwargs))

        def __getitem__(self, key):
            return indexing_func(self, lambda x: x.__getitem__(key))

        orig_init_method = copy_func(cls.__init__)

        def __init__(self, *args, **kwargs):
            orig_init_method(self, *args, **kwargs)
            self._iloc = iLoc(self)
            self._loc = Loc(self)

        setattr(cls, '__init__', __init__)
        setattr(cls, '__getitem__', __getitem__)
        setattr(cls, 'iloc', iloc)
        setattr(cls, 'loc', loc)
        setattr(cls, 'xs', xs)
        return cls
    return wrapper


def loc_mapper(mapper, like_df, loc_pandas_func):
    """Broadcast mapper series to a dataframe to perform advanced pandas indexing on it."""
    df_range_mapper = broadcast_to(np.arange(len(mapper.index)), like_df, index_from=1, columns_from=1)
    loced_range_mapper = loc_pandas_func(df_range_mapper)
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    if is_frame(loced_range_mapper):
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    elif is_series(loced_range_mapper):
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)


def add_param_indexing(param_name, indexing_func):
    def wrapper(cls):
        """Add loc indexing of params to a class.

        Uses a mapper, which is just a pd.Series object that maps columns to params."""

        class ParamLoc:
            def __init__(self, obj, param_mapper, level_names=None):
                check_type(param_mapper, pd.Series)

                self.obj = obj
                if param_mapper.dtype == 'O':
                    # If params are objects, we must cast them to string first
                    # The original mapper isn't touched
                    param_mapper = param_mapper.astype(str)
                self.param_mapper = param_mapper

            def get_indices(self, key):
                if self.param_mapper.dtype == 'O':
                    # We must also cast the key to string
                    if isinstance(key, slice):
                        start = str(key.start) if key.start is not None else None
                        stop = str(key.stop) if key.stop is not None else None
                        key = slice(start, stop, key.step)
                    elif isinstance(key, (list, np.ndarray)):
                        key = list(map(str, key))
                    else:
                        # Tuples, objects, etc.
                        key = str(key)
                param_mapper = self.param_mapper
                # Use pandas to perform indexing
                param_mapper = pd.Series(np.arange(len(param_mapper.index)), index=param_mapper.values)
                indices = param_mapper.loc.__getitem__(key)
                if isinstance(indices, pd.Series):
                    indices = indices.values
                return indices

            def __getitem__(self, key):
                indices = self.get_indices(key)
                is_multiple = isinstance(key, (slice, list, np.ndarray))
                level_name = self.param_mapper.name  # name of the mapper should contain level names of the params

                def loc_pandas_func(obj):
                    new_obj = obj.iloc[:, indices]
                    if not is_multiple:
                        # If we selected only one param, then remove its columns levels to keep it clean
                        if level_name is not None:
                            if is_frame(new_obj):
                                if isinstance(new_obj.columns, pd.MultiIndex):
                                    new_obj.columns = drop_levels(new_obj.columns, level_name)
                    return new_obj

                return indexing_func(self.obj, loc_pandas_func)

        @property
        def param_loc(self):
            return getattr(self, f'_{param_name}_loc')

        orig_init_method = copy_func(cls.__init__)

        def __init__(self, *args, **kwargs):
            orig_init_method(self, *args, **kwargs)
            mapper = getattr(self, f'{param_name}_mapper')
            setattr(self, f'_{param_name}_loc', ParamLoc(self, mapper))

        setattr(cls, '__init__', __init__)
        setattr(cls, f'{param_name}_loc', param_loc)
        return cls
    return wrapper

# ############# Stacking ############# #


def unstack_to_array(arg):
    """Reshape object based on multi-index into a multi-dimensional array."""
    check_type(arg, (pd.Series, pd.DataFrame))
    if is_frame(arg):
        if arg.shape[0] == 1:
            arg = arg.iloc[0, :]
        elif arg.shape[1] == 1:
            arg = arg.iloc[:, 0]
    check_type(arg.index, pd.MultiIndex)
    sr = to_1d(arg)

    vals_idx_list = []
    for i in range(len(sr.index.levels)):
        vals = sr.index.get_level_values(i).to_numpy()
        unique_vals = np.unique(vals)
        idx_map = dict(zip(unique_vals, range(len(unique_vals))))
        vals_idx = list(map(lambda x: idx_map[x], vals))
        vals_idx_list.append(vals_idx)

    a = np.full(list(map(len, sr.index.levels)), np.nan)
    a[tuple(zip(vals_idx_list))] = sr.values
    return a


def make_symmetric(arg):
    """Make object symmetric along the diagonal."""
    check_type(arg, (pd.Series, pd.DataFrame))
    arg = to_2d(arg)
    check_not_type(arg.index, pd.MultiIndex)
    check_not_type(arg.columns, pd.MultiIndex)

    names = tuple(dict.fromkeys([arg.index.name, arg.columns.name]))
    if len(names) == 1:
        names = names[0]
    unique_index = pd.Index(dict.fromkeys(arg.index.tolist() + arg.columns.tolist()), name=names)
    df_out = pd.DataFrame(index=unique_index, columns=unique_index)
    df_out.loc[:, :] = arg
    df_out[df_out.isnull()] = arg.transpose()
    return df_out


def unstack_to_df(arg, symmetric=False):
    """Reshape object based on multi-index into dataframe."""
    check_type(arg, (pd.Series, pd.DataFrame))
    if is_frame(arg):
        if arg.shape[0] == 1:
            arg = arg.iloc[0, :]
        elif arg.shape[1] == 1:
            arg = arg.iloc[:, 0]
    check_type(arg.index, pd.MultiIndex)
    sr = to_1d(arg)

    index = sr.index.levels[0]
    columns = sr.index.levels[1]
    df = pd.DataFrame(unstack_to_array(sr), index=index, columns=columns)
    if symmetric:
        return make_symmetric(df)
    return df


def apply_and_concat_one(n, apply_func, *args, **kwargs):
    """For a range from 0 to n, apply a function and concat the results horizontally."""
    return np.hstack([to_2d(apply_func(i, *args, **kwargs)) for i in range(n)])


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
    outputs = [tuple(map(to_2d, apply_func(i, *args, **kwargs))) for i in range(n)]
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


# ############# Numba decorators ############# #

def get_default_args(func):
    return {
        k: v.default
        for k, v in signature(func).parameters.items()
        if v.default is not Parameter.empty
    }


def add_safe_nb_methods(*nb_funcs):
    def wrapper(cls):
        """Wrap numba functions as methods."""
        for nb_func in nb_funcs:
            default_kwargs = get_default_args(nb_func)

            def array_operation(self, *args, nb_func=nb_func, default_kwargs=default_kwargs, **kwargs):
                if '_1d' in nb_func.__name__:
                    return self.wrap_array(nb_func(self.to_1d_array(), *args, **{**default_kwargs, **kwargs}))
                else:
                    # We work natively on 2d arrays
                    return self.wrap_array(nb_func(self.to_2d_array(), *args, **{**default_kwargs, **kwargs}))
            setattr(cls, nb_func.__name__.replace('_1d', '').replace('_nb', ''), array_operation)
        return cls
    return wrapper

# ############# Caching ############# #


def cached_property(func):
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        """Cache property to avoid recalculating it again and again."""
        obj = args[0]
        attr_name = '_' + func.__name__
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        else:
            to_be_cached = func(*args, **kwargs)
            setattr(obj, attr_name, to_be_cached)
            return to_be_cached
    return property(wrapper_decorator)


# ############# Custom accessors ############# #

class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


class Base_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object
        self._validate(self._obj)

    dtype = None

    @classmethod
    def _validate(cls, obj):
        pass

    def validate(self):
        # Don't override it, just call it for the object to be instantiated
        pass

    @classmethod
    def empty(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def empty_like(cls, *args, **kwargs):
        raise NotImplementedError

    def to_1d_array(self):
        return to_1d(self._obj, raw=True)

    def to_2d_array(self):
        return to_2d(self._obj, raw=True)

    def wrap_array(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def tile(self, n, as_columns=None):
        tiled = tile(self._obj, n, along_axis=1)
        if as_columns is not None:
            new_columns = combine_indices(as_columns, to_2d(self._obj).columns)
            return self.wrap_array(tiled.values, columns=new_columns)
        return tiled

    def repeat(self, n, as_columns=None):
        repeated = repeat(self._obj, n, along_axis=1)
        if as_columns is not None:
            new_columns = combine_indices(to_2d(self._obj).columns, as_columns)
            return self.wrap_array(repeated.values, columns=new_columns)
        return repeated

    def align_to(self, other):
        check_type(other, (pd.Series, pd.DataFrame))
        obj = to_2d(self._obj)
        other = to_2d(other)

        aligned_index = align_index_to(obj.index, other.index)
        aligned_columns = align_index_to(obj.columns, other.columns)
        obj = obj.iloc[aligned_index, aligned_columns]
        return self.wrap_array(obj.values, index=other.index, columns=other.columns)

    @class_or_instancemethod
    def broadcast(self_or_cls, *others, **kwargs):
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        if isinstance(self_or_cls, type):
            return broadcast(*others, **kwargs)
        return broadcast(self_or_cls._obj, *others, **kwargs)

    def broadcast_to(self, other, **kwargs):
        if isinstance(other, Base_Accessor):
            other = other._obj
        return broadcast_to(self._obj, other, **kwargs)

    def make_symmetric(self):
        return make_symmetric(self._obj)

    def unstack_to_array(self):
        return unstack_to_array(self._obj)

    def unstack_to_df(self, **kwargs):
        return unstack_to_df(self._obj, **kwargs)

    @class_or_instancemethod
    def concat(self_or_cls, *others, as_columns=None, broadcast_kwargs={}):
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        if isinstance(self_or_cls, type):
            objs = others
        else:
            objs = (self_or_cls._obj,) + others
        broadcasted = broadcast(*objs, **broadcast_kwargs)
        broadcasted = tuple(map(to_2d, broadcasted))
        if is_pandas(broadcasted[0]):
            concated = pd.concat(broadcasted, axis=1)
            if as_columns is not None:
                concated.columns = combine_indices(as_columns, broadcasted[0].columns)
        else:
            concated = np.hstack(broadcasted)
        return concated

    def apply_and_concat(self, ntimes, *args, apply_func=None, as_columns=None, **kwargs):
        """Apply a function n times and concatenate results into a single dataframe."""
        check_not_none(apply_func)
        if is_numba_func(apply_func):
            # NOTE: your apply_func must a numba-compiled function and arguments must be numba-compatible
            # Also NOTE: outputs of apply_func must always be 2-dimensional
            result = apply_and_concat_nb(np.asarray(self._obj), ntimes, apply_func, *args, **kwargs)
        else:
            result = apply_and_concat(np.asarray(self._obj), ntimes, apply_func, *args, **kwargs)
        # Build column hierarchy
        if as_columns is not None:
            new_columns = combine_indices(as_columns, to_2d(self._obj).columns)
        else:
            new_columns = tile_index(to_2d(self._obj).columns, ntimes)
        return self.wrap_array(result, columns=new_columns)

    def combine_with(self, other, *args, combine_func=None, broadcast_kwargs={}, **kwargs):
        """Broadcast with other and combine."""
        if isinstance(other, Base_Accessor):
            other = other._obj
        check_not_none(combine_func)
        if is_numba_func(combine_func):
            # Numba requires writable arrays
            broadcast_kwargs = {**dict(writeable=True), **broadcast_kwargs}
        new_obj, new_other = broadcast(self._obj, other, **broadcast_kwargs)
        return new_obj.vbt.wrap_array(combine_func(np.asarray(new_obj), np.asarray(new_other), *args, **kwargs))

    def combine_with_multiple(self, others, *args, combine_func=None, concat=False,
                              broadcast_kwargs={}, as_columns=None, **kwargs):
        """Broadcast with others and combine them all pairwise."""
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        check_not_none(combine_func)
        check_type(others, Iterable)
        # Broadcast arguments
        if is_numba_func(combine_func):
            # Numba requires writable arrays
            broadcast_kwargs = {**dict(writeable=True), **broadcast_kwargs}
            # Plus all of our arrays must be in the same order
            broadcast_kwargs['copy_kwargs'] = {**dict(order='C'), **broadcast_kwargs.get('copy_kwargs', {})}
        new_obj, *new_others = broadcast(self._obj, *others, **broadcast_kwargs)
        broadcasted = tuple(map(np.asarray, (new_obj, *new_others)))
        if concat:
            # Concat the results horizontally
            if is_numba_func(combine_func):
                for i in range(1, len(broadcasted)):
                    # NOTE: all inputs must have the same dtype
                    check_same_meta(broadcasted[i-1], broadcasted[i])
                result = combine_and_concat_nb(broadcasted[0], broadcasted[1:], combine_func, *args, **kwargs)
            else:
                result = combine_and_concat(broadcasted[0], broadcasted[1:], combine_func, *args, **kwargs)
            if as_columns is not None:
                new_columns = combine_indices(as_columns, to_2d(new_obj).columns)
            else:
                new_columns = tile_index(to_2d(new_obj).columns, len(others))
            return new_obj.vbt.wrap_array(result, columns=new_columns)
        else:
            # Combine arguments pairwise into one object
            if is_numba_func(combine_func):
                for i in range(1, len(broadcasted)):
                    # NOTE: all inputs must have the same dtype
                    check_same_dtype(broadcasted[i-1], broadcasted[i])
                result = combine_multiple_nb(broadcasted, combine_func, *args, **kwargs)
            else:
                result = combine_multiple(broadcasted, combine_func, *args, **kwargs)
            return new_obj.vbt.wrap_array(result)

    # Comparison operators
    def __eq__(self, other): return self.combine_with(other, combine_func=np.equal)
    def __ne__(self, other): return self.combine_with(other, combine_func=np.not_equal)
    def __lt__(self, other): return self.combine_with(other, combine_func=np.less)
    def __gt__(self, other): return self.combine_with(other, combine_func=np.greater)
    def __le__(self, other): return self.combine_with(other, combine_func=np.less_equal)
    def __ge__(self, other): return self.combine_with(other, combine_func=np.greater_equal)

    # Binary operators
    def __add__(self, other): return self.combine_with(other, combine_func=np.add)
    def __sub__(self, other): return self.combine_with(other, combine_func=np.subtract)
    def __mul__(self, other): return self.combine_with(other, combine_func=np.multiply)
    def __div__(self, other): return self.combine_with(other, combine_func=np.divide)
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rdiv__ = __div__

    # Boolean operators
    def __and__(self, other): return self.combine_with(other, combine_func=np.logical_and)
    def __or__(self, other): return self.combine_with(other, combine_func=np.logical_or)
    def __xor__(self, other): return self.combine_with(other, combine_func=np.logical_xor)
    __rand__ = __and__
    __ror__ = __or__
    __rxor__ = __xor__


class Base_DFAccessor(Base_Accessor):

    @classmethod
    def _validate(cls, obj):
        check_type(obj, pd.DataFrame)

    @classmethod
    def empty(cls, shape, fill_value=np.nan, index=None, columns=None):
        return pd.DataFrame(
            np.full(shape, fill_value),
            index=index,
            columns=columns,
            dtype=cls.dtype)

    @classmethod
    def empty_like(cls, df, fill_value=np.nan):
        cls._validate(df)

        return cls.empty(
            df.shape,
            fill_value=fill_value,
            index=df.index,
            columns=df.columns)

    def wrap_array(self, a, index=None, columns=None, dtype=None):
        return wrap_array(a,
                          index=index,
                          columns=columns,
                          dtype=dtype,
                          default_index=self._obj.index,
                          default_columns=self._obj.columns,
                          to_ndim=2)


class Base_SRAccessor(Base_Accessor):
    # series is just a dataframe with one column
    # this way we don't have to define our custom functions for working with 1d data
    @classmethod
    def _validate(cls, obj):
        check_type(obj, pd.Series)

    @classmethod
    def empty(cls, size, fill_value=np.nan, index=None, name=None):
        return pd.Series(
            np.full(size, fill_value),
            index=index,
            name=name,
            dtype=cls.dtype)

    @classmethod
    def empty_like(cls, sr, fill_value=np.nan):
        cls._validate(sr)

        return cls.empty(
            sr.shape,
            fill_value=fill_value,
            index=sr.index,
            name=sr.name)

    def wrap_array(self, a, index=None, columns=None, dtype=None):
        return wrap_array(a,
                          index=index,
                          columns=columns,
                          dtype=dtype,
                          default_index=self._obj.index,
                          default_columns=[self._obj.name],
                          to_ndim=1)
