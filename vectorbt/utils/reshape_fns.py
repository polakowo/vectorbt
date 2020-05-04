"""Utility functions for reshaping arrays."""

import numpy as np
import pandas as pd

from vectorbt import defaults
from vectorbt.utils import checks, index_fns


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


def wrap_array(arg, index=None, columns=None, dtype=None, default_index=None, default_columns=None, to_ndim=None):
    """Wrap array `arg` into a Series/DataFrame with `index` (or `default_index` if None), `columns` 
    (or `default_columns` if None) and `dtype`. Also tries to bring the array to `to_ndim` 
    dimensions softly."""
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
        checks.assert_same_shape(arg, index, axis=(0, 0))
    if arg.ndim == 2 and columns is not None:
        checks.assert_same_shape(arg, columns, axis=(1, 0))

    if arg.ndim == 1:
        return pd.Series(arg, index=index, name=name, dtype=dtype)
    return pd.DataFrame(arg, index=index, columns=columns, dtype=dtype)


def wrap_array_as(arg1, arg2, **kwargs):
    """Wrap array `arg1` to be as `arg2`."""
    return wrap_array(arg1, default_index=arg2.index, default_columns=to_2d(arg2).columns, to_ndim=arg2.ndim, **kwargs)


def to_1d(arg, raw=False):
    """Reshape argument to one dimension. 

    If `raw` is `True`, returns NumPy array.
    If 2D, will collapse along axis 1 (i.e., DataFrame with one column to Series)."""
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
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


def to_2d(arg, raw=False, expand_axis=1):
    """Reshape argument to two dimensions. 

    If `raw` is `True`, returns NumPy array.
    If 1D, will expand along axis 1 (i.e., Series to DataFrame with one column)."""
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
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def repeat(arg, n, axis=1):
    """Repeat each element in `arg` `n` times along the specified axis."""
    if not checks.is_array(arg):
        arg = np.asarray(arg)
    if axis == 0:
        if checks.is_pandas(arg):
            return arg.vbt.wrap_array(
                np.repeat(arg.values, n, axis=0),
                index=index_fns.repeat(arg.index, n))
        return np.repeat(arg, n, axis=0)
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            return arg.vbt.wrap_array(
                np.repeat(arg.values, n, axis=1),
                columns=index_fns.repeat(arg.columns, n))
        return np.repeat(arg, n, axis=1)
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def tile(arg, n, axis=1):
    """Repeat the whole `arg` `n` times along the specified axis."""
    if not checks.is_array(arg):
        arg = np.asarray(arg)
    if axis == 0:
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
    elif axis == 1:
        arg = to_2d(arg)
        if checks.is_pandas(arg):
            return arg.vbt.wrap_array(
                np.tile(arg.values, (1, n)),
                columns=index_fns.tile(arg.columns, n))
        return np.tile(arg, (1, n))
    else:
        raise ValueError("Only axis 0 and 1 are supported")


def broadcast_index(*args, to_shape=None, index_from=None, axis=0, ignore_single='default', drop_duplicates='default', keep='default'):
    """Produce a broadcasted index/columns.

    Args:
        *args (array_like): Array-like objects.
        to_shape (tuple): Target shape. Optional.
        index_from (None, int, str or array_like): Broadcasting rule for this index/these columns.

            Accepts the following values:

            * `'default'` - take the value from `vectorbt.defaults.broadcast`
            * `None` - use the original index/columns of the objects in `args`
            * `int` - use the index/columns of the i-nth object in `args`
            * `'strict'` - ensure that all pandas objects have the same index/columns
            * `'stack'` - stack different indexes/columns using `vectorbt.utils.index_fns.stack`
            * everything else will be converted to `pd.Index`

        axis (int): Set to 0 for index and 1 for columns.
        ignore_single (bool): If `True`, ignores indexes/columns with one value, otherwise they will be repeated
            to match the length of the longest index/columns (can lead to pollution of levels).
        drop_duplicates (bool): See `vectorbt.utils.index_fns.drop_duplicate_levels`.
        keep (bool): See `vectorbt.utils.index_fns.drop_duplicate_levels`.
    """

    if ignore_single == 'default':
        ignore_single = defaults.broadcast['ignore_single']
    if drop_duplicates == 'default':
        drop_duplicates = defaults.broadcast['drop_duplicates']
    if keep == 'default':
        keep = defaults.broadcast['keep']
    index_str = 'columns' if axis == 1 else 'index'
    new_index = None

    if index_from is not None:
        if isinstance(index_from, int):
            # Take index/columns of the object indexed by index_from
            if axis == 1:
                new_index = to_2d(args[index_from]).columns
            else:
                new_index = args[index_from].index
        elif isinstance(index_from, str):
            if index_from in ('stack', 'strict'):
                # If pandas objects have different index/columns, stack them together
                # maxlen stores the length of the longest index
                if to_shape is None:
                    # Simulate broadcasting
                    to_shape = np.lib.stride_tricks._broadcast_shape(*args)
                if axis == 1 and len(to_shape) == 1:
                    to_shape = (to_shape[0], 1)
                maxlen = to_shape[1] if axis == 1 else to_shape[0]
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
        else:
            new_index = index_from
    return new_index


def wrap_broadcasted(old_arg, new_arg, is_pd=False, new_index=None, new_columns=None):
    """If the newly brodcasted array was originally a pandas object, make it pandas object again 
    and assign it the newly broadcasted index/columns."""
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


def broadcast(*args, to_shape=None, to_pd=None, index_from='default', columns_from='default', writeable=False, copy_kwargs={}, **kwargs):
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
        writable (bool): If `True`, makes broadcasted arrays writable, otherwise readonly.

            !!! note
                Has effect only if broadcasting was needed for that particular array.

                Making arrays writable is possible only through copying them, which is pretty expensive.

                Numba requires arrays to be writable.

        copy_kwargs (dict): Keyword arguments passed to `np.array`. For example, to specify `order`.

            !!! note
                Has effect on every array, independent from whether broadcasting was needed or not.

        **kwargs: Keyword arguments passed to `broadcast_index`.

    Example:
        Without broadcasting index and columns:

        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.utils.reshape_fns import broadcast

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
        [[0 0 0]
        [0 0 0]
        [0 0 0]]
        [[1 2 3]
        [1 2 3]
        [1 2 3]]
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
        index_from = defaults.broadcast['index_from']
    if isinstance(columns_from, str) and columns_from == 'default':
        columns_from = defaults.broadcast['columns_from']

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

    if is_pd:
        # Convert all pd.Series objects to pd.DataFrame if we work on 2-dim data
        if is_2d:
            for i in range(len(args)):
                if checks.is_series(args[i]):
                    args[i] = args[i].to_frame()

        # Decide on index and columns
        new_index = broadcast_index(*args, to_shape=to_shape, index_from=index_from, axis=0, **kwargs)
        new_columns = broadcast_index(*args, to_shape=to_shape, index_from=columns_from, axis=1, **kwargs)
    else:
        new_index, new_columns = None, None

    # Perform broadcasting
    if to_shape is None:
        new_args = np.broadcast_arrays(*args, subok=True)
    else:
        new_args = []
        for arg in args:
            new_arg = np.broadcast_to(arg, to_shape, subok=True)
            new_args.append(new_arg)

    # The problem is that broadcasting creates readonly objects and numba requires writable ones.
    # To make them writable we must copy, which is ok for small-sized arrays and not ok for large ones.
    # Thus check if broadcasting was needed in the first place, and if so, copy
    for i in range(len(new_args)):
        if new_args[i].shape == args[i].shape:
            # Broadcasting was not needed, take old array
            new_args[i] = np.array(args[i], copy=False, **copy_kwargs)
        else:
            # Broadcasting was needed, take new array
            new_args[i] = np.array(new_args[i], copy=writeable, **copy_kwargs)

    # Bring arrays to their old types (e.g. array -> pandas)
    for i in range(len(new_args)):
        new_args[i] = wrap_broadcasted(args[i], new_args[i], is_pd=is_pd, new_index=new_index, new_columns=new_columns)

    if len(new_args) > 1:
        return tuple(new_args)
    return new_args[0]


def broadcast_to(arg1, arg2, **kwargs):
    """Broadcast `arg1` to `arg2`.
    
    Keyword arguments `**kwargs` are passed to `broadcast`.
    
    Example:
        ```python-repl
        >>> import numpy as np
        >>> import pandas as pd
        >>> from vectorbt.utils.reshape_fns import broadcast_to

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
    index_from = None
    columns_from = None
    to_pd = checks.is_pandas(arg2)
    if to_pd:
        # Take index and columns from arg2
        index_from = arg2.index
        columns_from = to_2d(arg2).columns
    return broadcast(arg1, to_shape=arg2.shape, to_pd=to_pd, index_from=index_from, columns_from=columns_from, **kwargs)


def broadcast_to_array_of(arg1, arg2):
    """Broadcast `arg1` to the shape `(1, *arg2.shape)`.
    
    Example:
        ```python-repl
        >>> import numpy as np
        >>> from vectorbt.utils.reshape_fns import broadcast_to_array_of

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


def unstack_to_array(arg, levels=None):
    """Reshape `arg` based on its multi-index into a multi-dimensional array.

    Use `levels` to specify what index levels to unstack and in which order.
    
    Example:
        ```python-repl
        >>> import pandas as pd
        >>> from vectorbt.utils.reshape_fns import unstack_to_array

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
        >>> from vectorbt.utils.reshape_fns import make_symmetric

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
    """Reshape `arg` based on its multi-index into a DataFrame.
    
    Use `index_levels` to specify what index levels will form new index, and `column_levels` 
    for new columns. Set `symmetric` to `True` to make DataFrame symmetric.
    
    Example:
        ```python-repl
        >>> import pandas as pd
        >>> from vectorbt.utils.reshape_fns import unstack_to_df

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
        checks.assert_not_none(index_levels)
        checks.assert_not_none(column_levels)
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
