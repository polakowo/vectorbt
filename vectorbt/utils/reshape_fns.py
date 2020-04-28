import numpy as np
import pandas as pd

from vectorbt.utils import checks, index_fns
from vectorbt.utils.common import Config

# You can change this from code
# Useful for magic methods that cannot accept keyword arguments
broadcast_defaults = Config(
    index_from='strict',
    columns_from='stack',
    ignore_single=True,
    drop_duplicates=True,
    keep='last'
)


def soft_broadcast_to_ndim(arg, ndim):
    """Try to softly bring the argument to the specified number of dimensions (max 2)."""
    if not checks.is_array_like(arg):
        arg = np.asarray(arg)
    if ndim == 1:
        if arg.ndim == 2:
            if arg.shape[1] == 1:
                if checks.is_pandas(arg):
                    return arg.iloc[:, 0]
                return arg[:, 0]  # downgrade
    if ndim == 2:
        if arg.ndim == 1:
            if checks.is_pandas(arg):
                return arg.to_frame()
            return arg[:, None]  # upgrade
    return arg  # do nothing


def wrap_array(arg, index=None, columns=None, dtype=None, default_index=None, default_columns=None, to_ndim=None):
    """Wrap array into a series/dataframe."""
    if not checks.is_array(arg):
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
        checks.assert_same_shape(arg, index, along_axis=(0, 0))
    if arg.ndim == 2 and columns is not None:
        checks.assert_same_shape(arg, columns, along_axis=(1, 0))

    if arg.ndim == 1:
        return pd.Series(arg, index=index, name=name, dtype=dtype)
    return pd.DataFrame(arg, index=index, columns=columns, dtype=dtype)


def to_1d(arg, raw=False):
    """Reshape argument to one dimension."""
    if raw:
        arg = np.asarray(arg)
    if not checks.is_array_like(arg):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        if arg.shape[1] == 1:
            if checks.is_frame(arg):
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
    if not checks.is_array_like(arg):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        if checks.is_series(arg):
            if expand_axis == 0:
                return pd.DataFrame(arg.values[None, :], columns=arg.index)
            elif expand_axis == 1:
                return arg.to_frame()
        return np.expand_dims(arg, expand_axis)
    elif arg.ndim == 0:
        return arg.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def repeat(arg, n, along_axis=1):
    """Repeat array n times along specified axis."""
    if not checks.is_array_like(arg):
        arg = np.asarray(arg)
    if along_axis == 0:
        if checks.is_pandas(arg):
            return arg.vbt.wrap_array(
                np.repeat(arg.values, n, axis=0),
                index=index_fns.repeat(arg.index, n))
        return np.repeat(arg, n, axis=0)
    elif along_axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            return arg.vbt.wrap_array(
                np.repeat(arg.values, n, axis=1),
                columns=index_fns.repeat(arg.columns, n))
        return np.repeat(arg, n, axis=1)
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def tile(arg, n, along_axis=1):
    """Tile array n times along specified axis."""
    if not checks.is_array_like(arg):
        arg = np.asarray(arg)
    if along_axis == 0:
        if arg.ndim == 1:
            if checks.is_pandas(arg):
                return arg.vbt.wrap_array(
                    np.tile(arg.values, n),
                    index=index_fns.tile(arg.index, n))
            return np.tile(arg, n)
        if arg.ndim == 2:
            if checks.is_pandas(arg):
                return arg.vbt.wrap_array(
                    np.tile(arg.values, (n, 1)),
                    index=index_fns.tile(arg.index, n))
            return np.tile(arg, (n, 1))
    elif along_axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            return arg.vbt.wrap_array(
                np.tile(arg.values, (1, n)),
                columns=index_fns.tile(arg.columns, n))
        return np.tile(arg, (1, n))
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def broadcast_index(*args, index_from=None, axis=0, is_2d=False, ignore_single='default', drop_duplicates='default', keep='default'):
    """Broadcast index/columns of all arguments."""

    if ignore_single == 'default':
        ignore_single = broadcast_defaults['ignore_single']
    if drop_duplicates == 'default':
        drop_duplicates = broadcast_defaults['drop_duplicates']
    if keep == 'default':
        keep = broadcast_defaults['keep']
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
                if checks.is_pandas(arg):
                    if checks.is_series(arg):
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
                        # 1. rule: if indexes are of the same length, they are simply stacked
                        # 2. rule: if index has one element, it gets repeated and then stacked

                        if pd.Index.equals(index, new_index):
                            continue
                        if len(index) != len(new_index):
                            if len(index) > 1 and len(new_index) > 1:
                                raise ValueError("Indexes could not be broadcast together")
                            if ignore_single:
                                # Columns of length 1 should be simply ignored
                                if len(index) > len(new_index):
                                    new_index = index
                                continue
                            else:
                                if len(index) > len(new_index):
                                    new_index = index_fns.repeat(new_index, len(index))
                                elif len(index) < len(new_index):
                                    index = index_fns.repeat(index, len(new_index))
                        new_index = index_fns.stack(new_index, index)
                        if drop_duplicates:
                            new_index = index_fns.drop_duplicate_levels(new_index, keep=keep)
            if maxlen > len(new_index):
                if index_from == 'strict':
                    raise ValueError(f"Broadcasting {index_str} is not allowed for {index_str}_from=strict")
                # This happens only when some numpy object is longer than the new pandas index
                # In this case, new pandas index (one element) should be repeated to match this length.
                if maxlen > 1 and len(new_index) > 1:
                    raise ValueError("Indexes could not be broadcast together")
                new_index = index_fns.repeat(new_index, maxlen)
        else:
            raise ValueError(f"Invalid value {index_from} for {'columns' if axis == 1 else 'index'}_from")
    return new_index


def wrap_broadcasted(old_arg, new_arg, is_pd=False, new_index=None, new_columns=None):
    """Transform newly broadcasted array to match the type of the original object."""
    if is_pd:
        if checks.is_pandas(old_arg):
            if new_index is None:
                # Take index from original pandas object
                if old_arg.shape[0] == new_arg.shape[0]:
                    new_index = old_arg.index
                else:
                    new_index = index_fns.repeat(old_arg.index, new_arg.shape[0])
            if new_columns is None:
                # Take columns from original pandas object
                if new_arg.ndim == 2:
                    if checks.is_series(old_arg):
                        old_arg = old_arg.to_frame()
                    if old_arg.shape[1] == new_arg.shape[1]:
                        new_columns = old_arg.columns
                    else:
                        new_columns = index_fns.repeat(old_arg.columns, new_arg.shape[1])
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
        if not checks.is_array_like(args[i]):
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
        if not checks.is_array_like(args[i]):
            args[i] = np.asarray(args[i])
        if args[i].ndim > 1:
            is_2d = True
        if checks.is_pandas(args[i]):
            is_pd = True

    if is_pd:
        # Convert all pd.Series objects to pd.DataFrame
        if is_2d:
            for i in range(len(args)):
                if checks.is_series(args[i]):
                    args[i] = args[i].to_frame()

        # Decide on index and columns
        if index_from == 'default':
            index_from = broadcast_defaults['index_from']
        if columns_from == 'default':
            columns_from = broadcast_defaults['columns_from']
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


def broadcast_to(arg1, arg2, index_from=1, columns_from=1, writeable=False, copy_kwargs={}, raw=False, **kwargs):
    """Bring first argument to the shape of second argument. 

    Closely resembles the other broadcast function."""
    if not checks.is_array_like(arg1):
        arg1 = np.asarray(arg1)
    if not checks.is_array_like(arg2):
        arg2 = np.asarray(arg2)

    is_2d = arg1.ndim > 1 or arg2.ndim > 1
    is_pd = checks.is_pandas(arg1) or checks.is_pandas(arg2)

    if is_pd:
        if is_2d:
            if checks.is_series(arg1):
                arg1 = arg1.to_frame()
            if checks.is_series(arg2):
                arg2 = arg2.to_frame()

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
    checks.assert_ndim(arg1, 1)

    if arg2.ndim == 0:
        return arg1
    for i in range(arg2.ndim):
        arg1 = np.expand_dims(arg1, axis=-1)
    return np.tile(arg1, (1, *arg2.shape))


def unstack_to_array(arg, levels=None):
    """Reshape object based on multi-index into a multi-dimensional array."""
    checks.assert_type(arg, pd.Series)
    checks.assert_type(arg.index, pd.MultiIndex)

    unique_idx_list = []
    vals_idx_list = []
    if levels is None:
        levels = arg.index.names
    for i in range(len(levels)):
        vals = index_fns.select_levels(arg.index, levels[i]).to_numpy()
        unique_vals = np.unique(vals)
        unique_idx_list.append(unique_vals)
        idx_map = dict(zip(unique_vals, range(len(unique_vals))))
        vals_idx = list(map(lambda x: idx_map[x], vals))
        vals_idx_list.append(vals_idx)

    a = np.full(list(map(len, unique_idx_list)), np.nan)
    a[tuple(zip(vals_idx_list))] = arg.values
    return a


def make_symmetric(arg):
    """Make object symmetric along the diagonal."""
    checks.assert_type(arg, (pd.Series, pd.DataFrame))
    arg = to_2d(arg)
    checks.assert_same_type(arg.index, arg.columns)
    if isinstance(arg.index, pd.MultiIndex):
        checks.assert_same_len(arg.index.names, arg.columns.names)
        names1, names2 = tuple(arg.index.names), tuple(arg.columns.names)
    else:
        names1, names2 = arg.index.name, arg.columns.name

    if names1 == names2:
        new_name = names1
    else:
        if isinstance(arg.index, pd.MultiIndex):
            new_name = tuple(zip(*[names1, names2]))
        else:
            new_name = (names1, names2)
    idx_vals = np.unique(np.concatenate((arg.index, arg.columns)))
    arg = arg.copy()
    if isinstance(arg.index, pd.MultiIndex):
        unique_index = pd.MultiIndex.from_tuples(idx_vals, names=new_name)
        arg.index.names = new_name
        arg.columns.names = new_name
    else:
        unique_index = pd.Index(idx_vals, name=new_name)
        arg.index.name = new_name
        arg.columns.name = new_name
    df_out = pd.DataFrame(index=unique_index, columns=unique_index)
    df_out.loc[:, :] = arg
    df_out[df_out.isnull()] = arg.transpose()
    return df_out


def unstack_to_df(arg, index_levels=None, column_levels=None, symmetric=False):
    """Reshape object based on multi-index into dataframe."""
    # Perform checks
    checks.assert_type(arg, (pd.Series, pd.DataFrame))
    if checks.is_frame(arg):
        if arg.shape[0] == 1:
            arg = arg.iloc[0, :]
        elif arg.shape[1] == 1:
            arg = arg.iloc[:, 0]
    checks.assert_type(arg.index, pd.MultiIndex)
    sr = to_1d(arg)

    if len(sr.index.levels) > 2:
        checks.assert_not_none(index_levels)
        checks.assert_not_none(column_levels)
    else:
        index_levels = 0
        column_levels = 1

    # Build new index and column hierarchies
    new_index = np.unique(index_fns.select_levels(arg.index, index_levels))
    new_columns = np.unique(index_fns.select_levels(arg.index, column_levels))
    if isinstance(index_levels, (list, tuple)):
        new_index = pd.MultiIndex.from_tuples(new_index, names=index_levels)
    else:
        new_index = pd.Index(new_index, name=index_levels)
    if isinstance(column_levels, (list, tuple)):
        new_columns = pd.MultiIndex.from_tuples(new_columns, names=column_levels)
    else:
        new_columns = pd.Index(new_columns, name=column_levels)

    # Unstack and post-process
    unstacked = unstack_to_array(sr, levels=(index_levels, column_levels))
    df = pd.DataFrame(unstacked, index=new_index, columns=new_columns)
    if symmetric:
        return make_symmetric(df)
    return df
