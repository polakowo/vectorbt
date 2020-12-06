"""Functions for reshaping arrays."""

import numpy as np
import pandas as pd
from numba import njit

from vectorbt.utils import checks
from vectorbt.base import index_fns, array_wrapper


def soft_to_ndim(arg, ndim):
    """Try to softly bring `arg` to the specified number of dimensions `ndim` (max 2)."""
    if not checks.is_array(arg):
        arg = np.asarray(arg)
    if ndim == 1:
        if arg.ndim == 2:
            if arg.shape[1] == 1:
                if checks.is_frame(arg):
                    return arg.iloc[:, 0]
                return arg[:, 0]  # downgrade
    if ndim == 2:
        if arg.ndim == 1:
            if checks.is_series(arg):
                return arg.to_frame()
            return arg[:, None]  # upgrade
    return arg  # do nothing


def to_1d(arg, raw=False):
    """Reshape argument to one dimension. 

    If `raw` is True, returns NumPy array.
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
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


def to_2d(arg, raw=False, expand_axis=1):
    """Reshape argument to two dimensions. 

    If `raw` is True, returns NumPy array.
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
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


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
        raise ValueError("Only axis 0 and 1 are supported")


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
        raise ValueError("Only axis 0 and 1 are supported")


def broadcast_index(args, to_shape, index_from=None, axis=0, ignore_sr_names=None, **kwargs):
    """Produce a broadcast index/columns.

    Args:
        *args (array_like): Array-like objects.
        to_shape (tuple): Target shape.
        index_from (None, int, str or array_like): Broadcasting rule for this index/these columns.

            Accepts the following values:

            * 'default' - take the value from `vectorbt.settings.broadcasting`
            * 'strict' - ensure that all pandas objects have the same index/columns
            * 'stack' - stack different indexes/columns using `vectorbt.base.index_fns.stack_indexes`
            * 'ignore' - ignore any index/columns
            * integer - use the index/columns of the i-nth object in `args`
            * None - use the original index/columns of the objects in `args`
            * everything else will be converted to `pd.Index`

        axis (int): Set to 0 for index and 1 for columns.
        ignore_sr_names (bool): Whether to ignore Series names if they are in conflict.

            Conflicting Series names are those that are different but not None.
        **kwargs: Keyword arguments passed to `vectorbt.base.index_fns.stack_indexes`.

    For defaults, see `vectorbt.settings.broadcasting`.

    !!! note
        Series names are treated as columns with a single element but without a name.
        If a column level without a name loses its meaning, better to convert Series to DataFrames
        with one column prior to broadcasting. If the name of a Series is not that important,
        better to drop it altogether by setting it to None.
    """
    from vectorbt import settings

    if ignore_sr_names is None:
        ignore_sr_names = settings.broadcasting['ignore_sr_names']
    index_str = 'columns' if axis == 1 else 'index'
    to_shape_2d = (to_shape[0], 1) if len(to_shape) == 1 else to_shape
    # maxlen stores the length of the longest index
    maxlen = to_shape_2d[1] if axis == 1 else to_shape_2d[0]
    new_index = None

    if index_from is not None:
        if isinstance(index_from, int):
            # Take index/columns of the object indexed by index_from
            if not checks.is_pandas(args[index_from]):
                raise TypeError(f"Argument under index {index_from} must be a pandas object")
            new_index = index_fns.get_index(args[index_from], axis)
        elif isinstance(index_from, str):
            if index_from == 'ignore':
                # Ignore index/columns
                new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
            elif index_from in ('stack', 'strict'):
                # Check whether all indexes/columns are equal
                last_index = None  # of type pd.Index
                index_conflict = False
                for arg in args:
                    if checks.is_pandas(arg):
                        index = index_fns.get_index(arg, axis)
                        if last_index is not None:
                            if not pd.Index.equals(index, last_index):
                                index_conflict = True
                        last_index = index
                        continue
                if not index_conflict:
                    new_index = last_index
                else:
                    # If pandas objects have different index/columns, stack them together
                    for arg in args:
                        if checks.is_pandas(arg):
                            index = index_fns.get_index(arg, axis)
                            if axis == 1 and checks.is_series(arg) and ignore_sr_names:
                                # ignore Series name
                                continue
                            if checks.is_default_index(index):
                                # ignore simple ranges without name
                                continue
                            if new_index is None:
                                new_index = index
                            else:
                                if index_from == 'strict':
                                    # If pandas objects have different index/columns, raise an exception
                                    if not pd.Index.equals(index, new_index):
                                        raise ValueError(
                                            f"Broadcasting {index_str} is not allowed when {index_str}_from=strict")
                                # Broadcasting index must follow the rules of a regular broadcasting operation
                                # https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html#general-broadcasting-rules
                                # 1. rule: if indexes are of the same length, they are simply stacked
                                # 2. rule: if index has one element, it gets repeated and then stacked

                                if pd.Index.equals(index, new_index):
                                    continue
                                if len(index) != len(new_index):
                                    if len(index) > 1 and len(new_index) > 1:
                                        raise ValueError("Indexes could not be broadcast together")
                                    if len(index) > len(new_index):
                                        new_index = index_fns.repeat_index(new_index, len(index))
                                    elif len(index) < len(new_index):
                                        index = index_fns.repeat_index(index, len(new_index))
                                new_index = index_fns.stack_indexes(new_index, index, **kwargs)
            else:
                raise ValueError(f"Invalid value {index_from} for {'columns' if axis == 1 else 'index'}_from")
        else:
            new_index = index_from
        if new_index is not None:
            if maxlen > len(new_index):
                if index_from == 'strict':
                    raise ValueError(f"Broadcasting {index_str} is not allowed when {index_str}_from=strict")
                # This happens only when some numpy object is longer than the new pandas index
                # In this case, new pandas index (one element) should be repeated to match this length.
                if maxlen > 1 and len(new_index) > 1:
                    raise ValueError("Indexes could not be broadcast together")
                new_index = index_fns.repeat_index(new_index, maxlen)
        elif index_from is not None:
            # new_index=None can mean two things: 1) take original metadata or 2) reset index/columns
            # In case when index_from is not None, we choose 2)
            new_index = pd.RangeIndex(start=0, stop=maxlen, step=1)
    return new_index


def wrap_broadcasted(old_arg, new_arg, is_pd=False, new_index=None, new_columns=None):
    """If the newly brodcasted array was originally a pandas object, make it pandas object again 
    and assign it the newly broadcast index/columns."""
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
        if new_arg.ndim == 2:
            return pd.DataFrame(new_arg, index=new_index, columns=new_columns)
        if new_columns is not None and len(new_columns) == 1:
            name = new_columns[0]
            if name == 0:
                name = None
        else:
            name = None
        return pd.Series(new_arg, index=new_index, name=name)
    return new_arg


def broadcast(*args, to_shape=None, to_pd=None, to_frame=None, align_index=None, align_columns=None,
              index_from='default', columns_from='default', require_kwargs=None, keep_raw=False,
              return_meta=False, **kwargs):
    """Bring any array-like object in `args` to the same shape by using NumPy broadcasting.

    See [Broadcasting](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

    Can broadcast pandas objects by broadcasting their index/columns with `broadcast_index`.

    Args:
        *args (array_like): Array-like objects.
        to_shape (tuple): Target shape. If set, will broadcast every element in `args` to `to_shape`.
        to_pd (bool, tuple or list): Whether to convert all output arrays to pandas, otherwise returns
            raw NumPy arrays. If None, converts only if there is at least one pandas object among them.
        to_frame (bool): Whether to convert all Series to DataFrames.
        align_index (bool): Whether to align index of pandas objects using multi-index.
        align_columns (bool): Whether to align columns of pandas objects using multi-index.
        index_from (any): Broadcasting rule for index.
        columns_from (any): Broadcasting rule for columns.
        require_kwargs (dict or list of dict): Keyword arguments passed to `np.require`.
        keep_raw (bool, tuple or list): Whether to keep the unbroadcasted version of the array.

            Only makes sure that the array can be broadcast to the target shape.
        return_meta (bool): If True, will also return new shape, index and columns.
        **kwargs: Keyword arguments passed to `broadcast_index`.

    For defaults, see `vectorbt.settings.broadcasting`.

    ## Example

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
    from vectorbt import settings

    is_pd = False
    is_2d = False
    args = list(args)
    if require_kwargs is None:
        require_kwargs = {}
    if align_index is None:
        align_index = settings.broadcasting['align_index']
    if align_columns is None:
        align_columns = settings.broadcasting['align_columns']
    if isinstance(index_from, str) and index_from == 'default':
        index_from = settings.broadcasting['index_from']
    if isinstance(columns_from, str) and columns_from == 'default':
        columns_from = settings.broadcasting['columns_from']

    # Convert to np.ndarray object if not numpy or pandas
    # Also check whether we broadcast to pandas and whether work on 2-dim data
    for i in range(len(args)):
        if not checks.is_array(args[i]):
            args[i] = np.asarray(args[i])
        if args[i].ndim > 1:
            is_2d = True
        if checks.is_pandas(args[i]):
            is_pd = True

    # If target shape specified, check again if we work on 2-dim data
    if to_shape is not None:
        if isinstance(to_shape, int):
            to_shape = (to_shape,)
        checks.assert_type(to_shape, tuple)
        if len(to_shape) > 1:
            is_2d = True

    if to_frame is not None:
        # force either keeping Series or converting them to DataFrames
        is_2d = to_frame

    if to_pd is not None:
        # force either raw or pandas
        if isinstance(to_pd, (tuple, list)):
            is_pd = any(to_pd)
        else:
            is_pd = to_pd

    # Align pandas objects
    if align_index:
        index_to_align = []
        for i in range(len(args)):
            if checks.is_pandas(args[i]) and len(args[i].index) > 1:
                index_to_align.append(i)
        if len(index_to_align) > 1:
            indexes = [args[i].index for i in index_to_align]
            if len(set(map(len, indexes))) > 1:
                index_indices = index_fns.align_indexes(*indexes)
                for i in range(len(args)):
                    if i in index_to_align:
                        args[i] = args[i].iloc[index_indices[index_to_align.index(i)]]
    if align_columns:
        cols_to_align = []
        for i in range(len(args)):
            if checks.is_frame(args[i]) and len(args[i].columns) > 1:
                cols_to_align.append(i)
        if len(cols_to_align) > 1:
            indexes = [args[i].columns for i in cols_to_align]
            if len(set(map(len, indexes))) > 1:
                col_indices = index_fns.align_indexes(*indexes)
                for i in range(len(args)):
                    if i in cols_to_align:
                        args[i] = args[i].iloc[:, col_indices[cols_to_align.index(i)]]

    # Convert all pd.Series objects to pd.DataFrame if we work on 2-dim data
    args_2d = [arg.to_frame() if is_2d and checks.is_series(arg) else arg for arg in args]

    # Get final shape
    if to_shape is None:
        to_shape = np.lib.stride_tricks._broadcast_shape(*args_2d)

    # Perform broadcasting
    new_args = []
    for i, arg in enumerate(args_2d):
        if isinstance(keep_raw, (tuple, list)):
            _keep_raw = keep_raw[i]
        else:
            _keep_raw = keep_raw
        bc_arg = np.broadcast_to(arg, to_shape)
        if _keep_raw:
            new_args.append(arg)
            continue
        new_args.append(bc_arg)

    # Force to match requirements
    for i in range(len(new_args)):
        if isinstance(require_kwargs, (tuple, list)):
            _require_kwargs = require_kwargs[i]
        else:
            _require_kwargs = require_kwargs
        new_args[i] = np.require(new_args[i], **_require_kwargs)

    if is_pd:
        # Decide on index and columns
        # NOTE: Important to pass args, not args_2d, to preserve original shape info
        new_index = broadcast_index(args, to_shape, index_from=index_from, axis=0, **kwargs)
        new_columns = broadcast_index(args, to_shape, index_from=columns_from, axis=1, **kwargs)
    else:
        new_index, new_columns = None, None

    # Bring arrays to their old types (e.g. array -> pandas)
    for i in range(len(new_args)):
        if isinstance(keep_raw, (tuple, list)):
            _keep_raw = keep_raw[i]
        else:
            _keep_raw = keep_raw
        if _keep_raw:
            continue
        if isinstance(to_pd, (tuple, list)):
            _is_pd = to_pd[i]
        else:
            _is_pd = is_pd
        new_args[i] = wrap_broadcasted(
            args[i],
            new_args[i],
            is_pd=_is_pd,
            new_index=new_index,
            new_columns=new_columns
        )

    if len(new_args) > 1:
        if return_meta:
            return tuple(new_args), to_shape, new_index, new_columns
        return tuple(new_args)
    if return_meta:
        return new_args[0], to_shape, new_index, new_columns
    return new_args[0]


def broadcast_to(arg1, arg2, to_pd=None, index_from=None, columns_from=None, **kwargs):
    """Broadcast `arg1` to `arg2`.

    Keyword arguments `**kwargs` are passed to `broadcast`.

    ## Example

    ```python-repl
    >>> import numpy as np
    >>> import pandas as pd
    >>> from vectorbt.base.reshape_fns import broadcast_to

    >>> a = np.array([1, 2, 3])
    >>> sr = pd.Series([4, 5, 6], index=pd.Index(['x', 'y', 'z']), name='a')

    >>> broadcast_to(a, sr)
    x    1
    y    2
    z    3
    Name: a, dtype: int64

    >>> broadcast_to(sr, a)
    array([4, 5, 6])
    ```
    """
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

    ## Example

    ```python-repl
    >>> import numpy as np
    >>> from vectorbt.base.reshape_fns import broadcast_to_array_of

    >>> broadcast_to_array_of([0.1, 0.2], np.empty((2, 2)))
    [[[0.1 0.1]
      [0.1 0.1]]

     [[0.2 0.2]
      [0.2 0.2]]]
    ```
    """
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


def broadcast_to_axis_of(arg1, arg2, axis, require_kwargs=None):
    """Broadcast `arg1` to an axis of `arg2`.

    If `arg2` has less dimensions than requested, will broadcast `arg1` to a single number.

    For other keyword arguments, see `broadcast`."""
    if require_kwargs is None:
        require_kwargs = {}
    if not checks.is_array(arg2):
        arg2 = np.asarray(arg2)
    if arg2.ndim < axis + 1:
        return np.broadcast_to(arg1, (1,))[0]  # to a single number
    arg1 = np.broadcast_to(arg1, (arg2.shape[axis],))
    arg1 = np.require(arg1, **require_kwargs)
    return arg1


def unstack_to_array(arg, levels=None):
    """Reshape `arg` based on its multi-index into a multi-dimensional array.

    Use `levels` to specify what index levels to unstack and in which order.

    ## Example

    ```python-repl
    >>> import pandas as pd
    >>> from vectorbt.base.reshape_fns import unstack_to_array

    >>> index = pd.MultiIndex.from_arrays(
    ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']])
    >>> sr = pd.Series([1, 2, 3, 4], index=index)

    >>> unstack_to_array(sr).shape
    (2, 2, 4)

    >>> unstack_to_array(sr)
    [[[ 1. nan nan nan]
     [nan  2. nan nan]]

     [[nan nan  3. nan]
    [nan nan nan  4.]]]

    >>> unstack_to_array(sr, levels=(2, 0))
    [[ 1. nan]
     [ 2. nan]
     [nan  3.]
     [nan  4.]]
    ```
    """
    checks.assert_type(arg, pd.Series)
    checks.assert_type(arg.index, pd.MultiIndex)

    unique_idx_list = []
    vals_idx_list = []
    if levels is None:
        levels = range(arg.index.nlevels)
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

    ## Example

    ```python-repl
    >>> import pandas as pd
    >>> from vectorbt.base.reshape_fns import make_symmetric

    >>> df = pd.DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['c', 'd'])

    >>> make_symmetric(df)
         a    b    c    d
    a  NaN  NaN  1.0  2.0
    b  NaN  NaN  3.0  4.0
    c  1.0  3.0  NaN  NaN
    d  2.0  4.0  NaN  NaN
    ```
    """
    checks.assert_type(arg, (pd.Series, pd.DataFrame))
    arg = to_2d(arg)
    if isinstance(arg.index, pd.MultiIndex) or isinstance(arg.columns, pd.MultiIndex):
        checks.assert_type(arg.index, pd.MultiIndex)
        checks.assert_type(arg.columns, pd.MultiIndex)
        checks.assert_array_equal(arg.index.nlevels, arg.columns.nlevels)
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
    for new columns. Set `symmetric` to True to make DataFrame symmetric.

    ## Example

    ```python-repl
    >>> import pandas as pd
    >>> from vectorbt.base.reshape_fns import unstack_to_df

    >>> index = pd.MultiIndex.from_arrays(
    ...     [[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']],
    ...     names=['x', 'y', 'z'])
    >>> sr = pd.Series([1, 2, 3, 4], index=index)

    >>> unstack_to_df(sr, index_levels=(0, 1), column_levels=2)
    z      a    b    c    d
    x y
    1 3  1.0  NaN  NaN  NaN
    1 4  NaN  2.0  NaN  NaN
    2 3  NaN  NaN  3.0  NaN
    2 4  NaN  NaN  NaN  4.0
    ```
    """
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
            raise ValueError("index_levels must be specified")
        if column_levels is None:
            raise ValueError("column_levels must be specified")
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


@njit(cache=True)
def flex_choose_i_and_col_nb(a, flex_2d):
    """Choose selection index and column based on the array's shape.

    Instead of expensive broadcasting, keep original shape and do indexing in a smart way.
    A nice feature of this is that it has almost no memory footprint and can broadcast in
    any direction indefinitely.

    Call it once before using `flex_select_nb`.

    if `flex_2d` is True, 1-dim array will correspond to columns, otherwise to rows."""
    i = -1
    col = -1
    if a.ndim == 0:
        i = 0
        col = 0
    elif a.ndim == 1:
        if flex_2d:
            i = 0
            if a.shape[0] == 1:
                col = 0
        else:
            col = 0
            if a.shape[0] == 1:
                i = 0
    else:
        if a.shape[0] == 1:
            i = 0
        if a.shape[1] == 1:
            col = 0
    return i, col


@njit(cache=True)
def flex_select_nb(i, col, a, flex_i, flex_col, flex_2d):
    """Select element of `a` as if it has been broadcast."""
    if flex_i == -1:
        flex_i = i
    if flex_col == -1:
        flex_col = col
    if a.ndim == 0:
        return a.item()
    if a.ndim == 1:
        if flex_2d:
            return a[flex_col]
        return a[flex_i]
    return a[flex_i, flex_col]


@njit(cache=True)
def flex_select_auto_nb(i, col, a, flex_2d):
    """Combines `flex_choose_i_and_col_nb` and `flex_select_nb`.

    !!! note
        Slower since it must call `flex_choose_i_and_col_nb` each time."""
    flex_i, flex_col = flex_choose_i_and_col_nb(a, flex_2d)
    return flex_select_nb(i, col, a, flex_i, flex_col, flex_2d)

