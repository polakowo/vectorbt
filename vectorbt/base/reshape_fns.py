"""Functions for reshaping arrays."""

import numpy as np
import pandas as pd

from vectorbt import defaults
from vectorbt.utils import checks
from vectorbt.base import index_fns, array_wrapper


def soft_broadcast_to_ndim(arg, ndim):
    """Try to softly bring `arg` to the specified number of dimensions `ndim` (max 2)."""
    if not checks.is_array(arg):
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


def to_1d(arg, raw=False):
    """Reshape argument to one dimension. 

    If `raw` is `True`, returns NumPy array.
    If 2-dim, will collapse along axis 1 (i.e., DataFrame with one column to Series)."""
    if raw or not checks.is_array(arg):
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
    raise Exception(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


def to_2d(arg, raw=False, expand_axis=1):
    """Reshape argument to two dimensions. 

    If `raw` is `True`, returns NumPy array.
    If 1-dim, will expand along axis 1 (i.e., Series to DataFrame with one column)."""
    if raw or not checks.is_array(arg):
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
    raise Exception(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def repeat(arg, n, axis=1):
    """Repeat each element in `arg` `n` times along the specified axis."""
    if not checks.is_array(arg):
        arg = np.asarray(arg)
    if axis == 0:
        if checks.is_pandas(arg):
            return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                np.repeat(arg.values, n, axis=0), index=index_fns.repeat_index(arg.index, n))
        return np.repeat(arg, n, axis=0)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                np.repeat(arg.values, n, axis=1), columns=index_fns.repeat_index(arg.columns, n))
        return np.repeat(arg, n, axis=1)
    else:
        raise Exception("Only axis 0 and 1 are supported")


def tile(arg, n, axis=1):
    """Repeat the whole `arg` `n` times along the specified axis."""
    if not checks.is_array(arg):
        arg = np.asarray(arg)
    if axis == 0:
        if arg.ndim == 2:
            if checks.is_pandas(arg):
                return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                    np.tile(arg.values, (n, 1)), index=index_fns.tile_index(arg.index, n))
            return np.tile(arg, (n, 1))
        if checks.is_pandas(arg):
            return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                np.tile(arg.values, n), index=index_fns.tile_index(arg.index, n))
        return np.tile(arg, n)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            return array_wrapper.ArrayWrapper.from_obj(arg).wrap(
                np.tile(arg.values, (1, n)), columns=index_fns.tile_index(arg.columns, n))
        return np.tile(arg, (1, n))
    else:
        raise Exception("Only axis 0 and 1 are supported")


def broadcast_index(args, to_shape, index_from=None, axis=0, ignore_single='default', drop_duplicates='default',
                    keep='default'):
    """Produce a broadcasted index/columns.

    Args:
        *args (array_like): Array-like objects.
        to_shape (tuple): Target shape.
        index_from (None, int, str or array_like): Broadcasting rule for this index/these columns.

            Accepts the following values:

            * `'default'` - take the value from `vectorbt.defaults.broadcasting`
            * `None` - use the original index/columns of the objects in `args`
            * `int` - use the index/columns of the i-nth object in `args`
            * `'strict'` - ensure that all pandas objects have the same index/columns
            * `'stack'` - stack different indexes/columns using `vectorbt.base.index_fns.stack_indexes`
            * everything else will be converted to `pd.Index`

        axis (int): Set to 0 for index and 1 for columns.
        ignore_single (bool): If `True`, won't consider index/columns with a single value.
        drop_duplicates (bool): See `vectorbt.base.index_fns.drop_duplicate_levels`.
        keep (bool): See `vectorbt.base.index_fns.drop_duplicate_levels`.

    For defaults, see `vectorbt.defaults.broadcasting`.
    """

    if ignore_single == 'default':
        ignore_single = defaults.broadcasting['ignore_single']
    if drop_duplicates == 'default':
        drop_duplicates = defaults.broadcasting['drop_duplicates']
    if keep == 'default':
        keep = defaults.broadcasting['keep']
    index_str = 'columns' if axis == 1 else 'index'
    new_index = None
    if axis == 1 and len(to_shape) == 1:
        to_shape = (to_shape[0], 1)
    maxlen = to_shape[1] if axis == 1 else to_shape[0]

    if index_from is not None:
        if isinstance(index_from, int):
            # Take index/columns of the object indexed by index_from
            if not checks.is_pandas(args[index_from]):
                raise Exception(f"Argument under index {index_from} must be a pandas object")
            new_index = index_fns.get_index(args[index_from], axis)
        elif isinstance(index_from, str):
            if index_from in ('stack', 'strict'):
                # If pandas objects have different index/columns, stack them together
                # maxlen stores the length of the longest index
                for arg in args:
                    if checks.is_pandas(arg):
                        index = index_fns.get_index(arg, axis)
                        if pd.Index.equals(index, pd.RangeIndex(start=0, stop=len(index), step=1)):
                            # ignore simple ranges without name
                            continue
                        if new_index is None:
                            new_index = index
                        else:
                            if index_from == 'strict':
                                # If pandas objects have different index/columns, raise an exception
                                if not pd.Index.equals(index, new_index):
                                    raise Exception(
                                        f"Broadcasting {index_str} is not allowed for {index_str}_from=strict")
                            # Broadcasting index must follow the rules of a regular broadcasting operation
                            # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
                            # 1. rule: if indexes are of the same length, they are simply stacked
                            # 2. rule: if index has one element, it gets repeated and then stacked

                            if pd.Index.equals(index, new_index):
                                continue
                            if len(index) != len(new_index):
                                if len(index) > 1 and len(new_index) > 1:
                                    raise Exception("Indexes could not be broadcast together")
                                if ignore_single:
                                    # Columns of length 1 should be ignored
                                    if len(index) > len(new_index):
                                        new_index = index
                                    continue
                                else:
                                    if len(index) > len(new_index):
                                        new_index = index_fns.repeat_index(new_index, len(index))
                                    elif len(index) < len(new_index):
                                        index = index_fns.repeat_index(index, len(new_index))
                            new_index = index_fns.stack_indexes(new_index, index)
            else:
                raise Exception(f"Invalid value {index_from} for {'columns' if axis == 1 else 'index'}_from")
        else:
            new_index = index_from
        if new_index is not None:
            if maxlen > len(new_index):
                if index_from == 'strict':
                    raise Exception(f"Broadcasting {index_str} is not allowed for {index_str}_from=strict")
                # This happens only when some numpy object is longer than the new pandas index
                # In this case, new pandas index (one element) should be repeated to match this length.
                if maxlen > 1 and len(new_index) > 1:
                    raise Exception("Indexes could not be broadcast together")
                new_index = index_fns.repeat_index(new_index, maxlen)
            if drop_duplicates:
                new_index = index_fns.drop_duplicate_levels(new_index, keep=keep)
    return new_index


def wrap_broadcasted(old_arg, new_arg, is_pd=False, new_index=None, new_columns=None):
    """If the newly brodcasted array was originally a pandas object, make it pandas object again 
    and assign it the newly broadcasted index/columns."""
    if is_pd:
        if checks.is_pandas(old_arg):
            if new_index is None:
                # Take index from original pandas object
                old_index = index_fns.get_index(old_arg, 0)
                if old_arg.shape[0] == new_arg.shape[0]:
                    new_index = old_index
                else:
                    new_index = index_fns.repeat_index(old_index, new_arg.shape[0])
            if new_columns is None:
                # Take columns from original pandas object
                old_columns = index_fns.get_index(old_arg, 1)
                new_ncols = new_arg.shape[1] if new_arg.ndim == 2 else 1
                if len(old_columns) == new_ncols:
                    new_columns = old_columns
                else:
                    new_columns = index_fns.repeat_index(old_columns, new_ncols)
        return array_wrapper.ArrayWrapper(index=new_index, columns=new_columns).wrap(new_arg)
    return new_arg


def broadcast(*args, to_shape=None, to_pd=None, index_from='default', columns_from='default',
              writeable=False, copy_kwargs={}, **kwargs):
    """Bring any array-like object in `args` to the same shape by using NumPy broadcasting.

    See [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

    Can broadcast pandas objects by broadcasting their index/columns with `broadcast_index`.

    Args:
        *args (array_like): Array-like objects.
        to_shape (tuple): Target shape. If set, will broadcast every element in `args` to `to_shape`.
        to_pd (bool): If `True`, converts all output arrays to pandas, otherwise returns raw NumPy
            arrays. If `None`, converts only if there is at least one pandas object among them.
        index_from (None, int, str or array_like): Broadcasting rule for index.
        columns_from (None, int, str or array_like): Broadcasting rule for columns.
        writeable (bool): If `True`, makes broadcasted arrays writable, otherwise readonly.

            !!! note
                Has effect only if broadcasting was needed for that particular array.

                Making arrays writable is possible only through copying them, which is pretty expensive.

                Numba requires arrays to be writable.

        copy_kwargs (dict): Keyword arguments passed to `np.array`. For example, to specify `order`.

            !!! note
                Has effect on every array, independent from whether broadcasting was needed or not.

        **kwargs: Keyword arguments passed to `broadcast_index`.

    For defaults, see `vectorbt.defaults.broadcasting`.

    Example:
        Without broadcasting index and columns:

        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import broadcast

        >>> v = 0
        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([1, 2, 3], index=pd.Index(['x', 'y', 'z']), name='a')
        >>> df = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
        ...     index=pd.Index(['x2', 'y2', 'z2']), 
        ...     columns=pd.Index(['a2', 'b2', 'c2']))

        >>> for i in broadcast(
        ...     v, a, sr, df,
        ...     index_from=None,
        ...     columns_from=None,
        ... ): print(i)
           0  1  2
        0  0  0  0
        1  0  0  0
        2  0  0  0
           0  1  2
        0  1  2  3
        1  1  2  3
        2  1  2  3
           a  a  a
        x  1  1  1
        y  2  2  2
        z  3  3  3
            a2  b2  c2
        x2   1   2   3
        y2   4   5   6
        z2   7   8   9
        ```

        Taking new index and columns from position:

        ```python-repl
        >>> for i in broadcast(
        ...     v, a, sr, df,
        ...     index_from=2,
        ...     columns_from=3
        ... ): print(i)
           a2  b2  c2
        x   0   0   0
        y   0   0   0
        z   0   0   0
           a2  b2  c2
        x   1   2   3
        y   1   2   3
        z   1   2   3
           a2  b2  c2
        x   1   1   1
        y   2   2   2
        z   3   3   3
           a2  b2  c2
        x   1   2   3
        y   4   5   6
        z   7   8   9
        ```

        Broadcasting index and columns through stacking:

        ```python-repl
        >>> for i in broadcast(
        ...     v, a, sr, df,
        ...     index_from='stack',
        ...     columns_from='stack'
        ... ): print(i)
              a2  b2  c2
        x x2   0   0   0
        y y2   0   0   0
        z z2   0   0   0
              a2  b2  c2
        x x2   1   2   3
        y y2   1   2   3
        z z2   1   2   3
              a2  b2  c2
        x x2   1   1   1
        y y2   2   2   2
        z z2   3   3   3
              a2  b2  c2
        x x2   1   2   3
        y y2   4   5   6
        z z2   7   8   9
        ```

        Setting index and columns manually:

        ```python-repl
        >>> for i in broadcast(
        ...     v, a, sr, df,
        ...     index_from=['a', 'b', 'c'],
        ...     columns_from=['d', 'e', 'f']
        ... ): print(i)
           d  e  f
        a  0  0  0
        b  0  0  0
        c  0  0  0
           d  e  f
        a  1  2  3
        b  1  2  3
        c  1  2  3
           d  e  f
        a  1  1  1
        b  2  2  2
        c  3  3  3
           d  e  f
        a  1  2  3
        b  4  5  6
        c  7  8  9
        ```
    """
    is_pd = False
    is_2d = False
    args = list(args)
    if isinstance(index_from, str) and index_from == 'default':
        index_from = defaults.broadcasting['index_from']
    if isinstance(columns_from, str) and columns_from == 'default':
        columns_from = defaults.broadcasting['columns_from']

    # Convert to np.ndarray object if not numpy or pandas
    # Also check whether we broadcast to pandas and whether work on 2-dim data
    for i in range(len(args)):
        if not checks.is_array(args[i]):
            args[i] = np.asarray(args[i])
        if args[i].ndim > 1:
            is_2d = True
        if checks.is_pandas(args[i]):
            is_pd = True

    if to_pd is not None:
        is_pd = to_pd  # force either raw or pandas

    # If target shape specified, check again if we work on 2-dim data
    if to_shape is not None:
        checks.assert_type(to_shape, tuple)
        if len(to_shape) > 1:
            is_2d = True

    # Convert all pd.Series objects to pd.DataFrame if we work on 2-dim data
    args_2d = [arg.to_frame() if is_2d and checks.is_series(arg) else arg for arg in args]

    # Get final shape
    if to_shape is None:
        to_shape = np.lib.stride_tricks._broadcast_shape(*args_2d)

    # Perform broadcasting
    new_args = [np.broadcast_to(arg, to_shape, subok=True) for arg in args_2d]

    # The problem is that broadcasting creates readonly objects and numba requires writable ones.
    # To make them writable we must copy, which is ok for small-sized arrays and not ok for large ones.
    # Thus check if broadcasting was needed in the first place, and if so, copy
    for i in range(len(new_args)):
        if new_args[i].shape == args_2d[i].shape:
            # Broadcasting was not needed, take old array
            new_args[i] = np.array(args_2d[i], copy=False, **copy_kwargs)
        else:
            # Broadcasting was needed, take new array
            new_args[i] = np.array(new_args[i], copy=writeable, **copy_kwargs)

    if is_pd:
        # Decide on index and columns
        # NOTE: Important to pass args, not args_2d, to preserve original shape info
        new_index = broadcast_index(args, to_shape, index_from=index_from, axis=0, **kwargs)
        new_columns = broadcast_index(args, to_shape, index_from=columns_from, axis=1, **kwargs)
    else:
        new_index, new_columns = None, None

    # Bring arrays to their old types (e.g. array -> pandas)
    for i in range(len(new_args)):
        new_args[i] = wrap_broadcasted(args[i], new_args[i], is_pd=is_pd, new_index=new_index, new_columns=new_columns)

    if len(new_args) > 1:
        return tuple(new_args)
    return new_args[0]


def broadcast_to(arg1, arg2, to_pd=None, index_from=None, columns_from=None, **kwargs):
    """Broadcast `arg1` to `arg2`.

    Keyword arguments `**kwargs` are passed to `broadcast`.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import broadcast_to

        >>> a = np.array([1, 2, 3])
        >>> sr = pd.Series([4, 5, 6], index=pd.Index(['x', 'y', 'z']), name='a')

        >>> print(broadcast_to(a, sr))
        x    1
        y    2
        z    3
        Name: a, dtype: int64

        >>> print(broadcast_to(sr, a))
        array([4, 5, 6])
        ```"""
    if not checks.is_array(arg1):
        arg1 = np.asarray(arg1)
    if not checks.is_array(arg2):
        arg2 = np.asarray(arg2)
    if to_pd is None:
        to_pd = checks.is_pandas(arg2)
    if to_pd:
        # Take index and columns from arg2
        if index_from is None:
            index_from = index_fns.get_index(arg2, 0)
        if columns_from is None:
            columns_from = index_fns.get_index(arg2, 1)
    return broadcast(arg1, to_shape=arg2.shape, to_pd=to_pd, index_from=index_from, columns_from=columns_from, **kwargs)


def broadcast_to_array_of(arg1, arg2):
    """Broadcast `arg1` to the shape `(1, *arg2.shape)`.

    `arg1` must be either a scalar, a 1-dim array, or have 1 dimension more than `arg2`.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from vectorbt.base.reshape_fns import broadcast_to_array_of

        >>> print(broadcast_to_array_of([0.1, 0.2], np.empty((2, 2))))
        [[[0.1 0.1]
          [0.1 0.1]]

         [[0.2 0.2]
          [0.2 0.2]]]
        ```"""
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


def broadcast_to_axis_of(arg1, arg2, axis, writeable=False, copy_kwargs={}):
    """Broadcast `arg1` to an axis of `arg2`.

    If `arg2` has less dimensions than requested, will broadcast `arg1` to a single number.

    For other keyword arguments, see `broadcast`."""
    if not checks.is_array(arg2):
        arg2 = np.asarray(arg2)
    if arg2.ndim < axis + 1:
        return np.broadcast_to(arg1, (1,))[0]  # to a single number
    arg1 = np.broadcast_to(arg1, (arg2.shape[axis],))
    return np.array(arg1, copy=writeable, **copy_kwargs)  # to shape of axis


def unstack_to_array(arg, levels=None):
    """Reshape `arg` based on its multi-index into a multi-dimensional array.

    Use `levels` to specify what index levels to unstack and in which order.

    Example:
        ```python-repl
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import unstack_to_array

        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)

        >>> print(unstack_to_array(sr).shape)
        (2, 2, 4)

        >>> print(unstack_to_array(sr))
        [[[ 1. nan nan nan]
         [nan  2. nan nan]]

         [[nan nan  3. nan]
        [nan nan nan  4.]]]

        >>> print(unstack_to_array(sr, levels=(2, 0)))
        [[ 1. nan]
         [ 2. nan]
         [nan  3.]
         [nan  4.]]
        ```"""
    checks.assert_type(arg, pd.Series)
    checks.assert_type(arg.index, pd.MultiIndex)

    unique_idx_list = []
    vals_idx_list = []
    if levels is None:
        levels = range(len(arg.index.levels))
    for level in levels:
        vals = index_fns.select_levels(arg.index, level).to_numpy()
        unique_vals = np.unique(vals)
        unique_idx_list.append(unique_vals)
        idx_map = dict(zip(unique_vals, range(len(unique_vals))))
        vals_idx = list(map(lambda x: idx_map[x], vals))
        vals_idx_list.append(vals_idx)

    a = np.full(list(map(len, unique_idx_list)), np.nan)
    a[tuple(zip(vals_idx_list))] = arg.values
    return a


def make_symmetric(arg):
    """Make `arg` symmetric.

    The index and columns of the resulting DataFrame will be identical.

    Requires the index and columns to have the same number of levels.

    Example:
        ```python-repl
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import make_symmetric

        >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['c', 'd'])

        >>> print(make_symmetric(df))
             a    b    c    d
        a  NaN  NaN  1.0  2.0
        b  NaN  NaN  3.0  4.0
        c  1.0  3.0  NaN  NaN
        d  2.0  4.0  NaN  NaN
        ```"""
    checks.assert_type(arg, (pd.Series, pd.DataFrame))
    arg = to_2d(arg)
    if isinstance(arg.index, pd.MultiIndex) or isinstance(arg.columns, pd.MultiIndex):
        checks.assert_type(arg.index, pd.MultiIndex)
        checks.assert_type(arg.columns, pd.MultiIndex)
        checks.assert_same(arg.index.nlevels, arg.columns.nlevels)
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
    idx_vals = list(dict.fromkeys(np.concatenate((arg.index, arg.columns))))
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
    """Reshape `arg` based on its multi-index into a DataFrame.

    Use `index_levels` to specify what index levels will form new index, and `column_levels` 
    for new columns. Set `symmetric` to `True` to make DataFrame symmetric.

    Example:
        ```python-repl
        >>> import pandas as pd
        >>> from vectorbt.base.reshape_fns import unstack_to_df

        >>> index = pd.MultiIndex.from_arrays(
        ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']], 
        ...     names=['x', 'y', 'z'])
        >>> sr = pd.Series([1, 2, 3, 4], index=index)

        >>> print(unstack_to_df(sr, index_levels=(0, 1), column_levels=2))
        z      a    b    c    d
        x y                    
        1 3  1.0  NaN  NaN  NaN
        1 4  NaN  2.0  NaN  NaN
        2 3  NaN  NaN  3.0  NaN
        2 4  NaN  NaN  NaN  4.0
        ```"""
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
        if index_levels is None:
            raise Exception("index_levels must be specified")
        if column_levels is None:
            raise Exception("column_levels must be specified")
    else:
        index_levels = 0
        column_levels = 1

    # Build new index and column hierarchies
    new_index = index_fns.select_levels(arg.index, index_levels).unique()
    new_columns = index_fns.select_levels(arg.index, column_levels).unique()

    # Unstack and post-process
    unstacked = unstack_to_array(sr, levels=(index_levels, column_levels))
    df = pd.DataFrame(unstacked, index=new_index, columns=new_columns)
    if symmetric:
        return make_symmetric(df)
    return df
