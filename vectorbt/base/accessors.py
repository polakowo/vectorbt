"""Custom pandas accessors.

!!! note
    Accessors do not utilize caching."""

import numpy as np
import pandas as pd
from collections.abc import Iterable

from vectorbt.utils import checks
from vectorbt.utils.decorators import class_or_instancemethod
from vectorbt.utils.config import merge_dicts
from vectorbt.base import combine_fns, index_fns, reshape_fns
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.base.class_helpers import (
    add_binary_magic_methods,
    add_unary_magic_methods,
    binary_magic_methods,
    unary_magic_methods
)


@add_binary_magic_methods(
    binary_magic_methods,
    lambda self, other, np_func: self.combine_with(other, combine_func=np_func)
)
@add_unary_magic_methods(
    unary_magic_methods,
    lambda self, np_func: self.apply(apply_func=np_func)
)
class Base_Accessor:
    """Accessor on top of Series and DataFrames.

    Accessible through `pd.Series.vbt` and `pd.DataFrame.vbt`, and all child accessors.

    Series is just a DataFrame with one column, hence to avoid defining methods exclusively for 1-dim data,
    we will convert any Series to a DataFrame and perform matrix computation on it. Afterwards,
    by using `Base_Accessor.wrapper`, we will convert the 2-dim output back to a Series.

    `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper`."""

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._obj = obj
        self._wrapper = ArrayWrapper.from_obj(obj, **kwargs)

    def __call__(self, *args, **kwargs):
        """Allows passing arguments to the initializer."""

        return self.__class__(self._obj, *args, **kwargs)

    @property
    def wrapper(self):
        """Array wrapper (read-only)."""
        return self._wrapper

    # ############# Creation ############# #

    @classmethod
    def empty(cls, shape, fill_value=np.nan, **kwargs):
        """Generate an empty Series/DataFrame of shape `shape` and fill with `fill_value`."""
        if not isinstance(shape, tuple) or (isinstance(shape, tuple) and len(shape) == 1):
            return pd.Series(np.full(shape, fill_value), **kwargs)
        return pd.DataFrame(np.full(shape, fill_value), **kwargs)

    @classmethod
    def empty_like(cls, other, fill_value=np.nan, **kwargs):
        """Generate an empty Series/DataFrame like `other` and fill with `fill_value`."""
        if checks.is_series(other):
            return cls.empty(other.shape, fill_value=fill_value, index=other.index, name=other.name, **kwargs)
        return cls.empty(other.shape, fill_value=fill_value, index=other.index, columns=other.columns, **kwargs)

    # ############# Index and columns ############# #

    def apply_on_index(self, apply_func, *args, axis=1, inplace=False, **kwargs):
        """Apply function `apply_func` on index of the pandas object.

        Set `axis` to 1 for columns and 0 for index.
        If `inplace` is True, modifies the pandas object. Otherwise, returns a copy."""
        checks.assert_in(axis, (0, 1))

        if axis == 1:
            obj_index = self.wrapper.columns
        else:
            obj_index = self.wrapper.index
        obj_index = apply_func(obj_index, *args, **kwargs)
        if inplace:
            if axis == 1:
                self._obj.columns = obj_index
            else:
                self._obj.index = obj_index
            return None
        else:
            obj = self._obj.copy()
            if axis == 1:
                obj.columns = obj_index
            else:
                obj.index = obj_index
            return obj

    def stack_index(self, index, on_top=True, axis=1, inplace=False):
        """See `vectorbt.base.index_fns.stack_indexes`.

        Set `on_top` to False to stack at bottom.

        See `Base_Accessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index):
            if on_top:
                return index_fns.stack_indexes(index, obj_index)
            return index_fns.stack_indexes(obj_index, index)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def drop_levels(self, levels, axis=1, inplace=False):
        """See `vectorbt.base.index_fns.drop_levels`.

        See `Base_Accessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index):
            return index_fns.drop_levels(obj_index, levels)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def rename_levels(self, name_dict, axis=1, inplace=False):
        """See `vectorbt.base.index_fns.rename_levels`.

        See `Base_Accessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index):
            return index_fns.rename_levels(obj_index, name_dict)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def select_levels(self, level_names, axis=1, inplace=False):
        """See `vectorbt.base.index_fns.select_levels`.

        See `Base_Accessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index):
            return index_fns.select_levels(obj_index, level_names)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def drop_redundant_levels(self, axis=1, inplace=False):
        """See `vectorbt.base.index_fns.drop_redundant_levels`.

        See `Base_Accessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index):
            return index_fns.drop_redundant_levels(obj_index)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    def drop_duplicate_levels(self, keep='last', axis=1, inplace=False):
        """See `vectorbt.base.index_fns.drop_duplicate_levels`.

        See `Base_Accessor.apply_on_index` for other keyword arguments."""

        def apply_func(obj_index):
            return index_fns.drop_duplicate_levels(obj_index, keep=keep)

        return self.apply_on_index(apply_func, axis=axis, inplace=inplace)

    # ############# Reshaping ############# #

    def to_1d_array(self):
        """Convert to 1-dim NumPy array

        See `vectorbt.base.reshape_fns.to_1d`."""
        return reshape_fns.to_1d(self._obj, raw=True)

    def to_2d_array(self):
        """Convert to 2-dim NumPy array.

        See `vectorbt.base.reshape_fns.to_2d`."""
        return reshape_fns.to_2d(self._obj, raw=True)

    def tile(self, n, keys=None, axis=1):
        """See `vectorbt.base.reshape_fns.tile`.

        Set `axis` to 1 for columns and 0 for index.
        Use `keys` as the outermost level."""
        tiled = reshape_fns.tile(self._obj, n, axis=axis)
        if keys is not None:
            if axis == 1:
                new_columns = index_fns.combine_indexes(keys, self.wrapper.columns)
                return tiled.vbt.wrapper.wrap(tiled.values, columns=new_columns)
            else:
                new_index = index_fns.combine_indexes(keys, self.wrapper.index)
                return tiled.vbt.wrapper.wrap(tiled.values, index=new_index)
        return tiled

    def repeat(self, n, keys=None, axis=1):
        """See `vectorbt.base.reshape_fns.repeat`.

        Set `axis` to 1 for columns and 0 for index.
        Use `keys` as the outermost level."""
        repeated = reshape_fns.repeat(self._obj, n, axis=axis)
        if keys is not None:
            if axis == 1:
                new_columns = index_fns.combine_indexes(self.wrapper.columns, keys)
                return repeated.vbt.wrapper.wrap(repeated.values, columns=new_columns)
            else:
                new_index = index_fns.combine_indexes(self.wrapper.index, keys)
                return repeated.vbt.wrapper.wrap(repeated.values, index=new_index)
        return repeated

    def align_to(self, other):
        """Align to `other` on their axes.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=['x', 'y'], columns=['a', 'b'])
        >>> df1
           a  b
        x  1  2
        y  3  4

        >>> df2 = pd.DataFrame([[5, 6, 7, 8], [9, 10, 11, 12]], index=['x', 'y'],
        ...     columns=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']]))
        >>> df2
               1       2
           a   b   a   b
        x  5   6   7   8
        y  9  10  11  12

        >>> df1.vbt.align_to(df2)
              1     2
           a  b  a  b
        x  1  2  1  2
        y  3  4  3  4
        ```
        """
        checks.assert_type(other, (pd.Series, pd.DataFrame))
        obj = reshape_fns.to_2d(self._obj)
        other = reshape_fns.to_2d(other)

        aligned_index = index_fns.align_index_to(obj.index, other.index)
        aligned_columns = index_fns.align_index_to(obj.columns, other.columns)
        obj = obj.iloc[aligned_index, aligned_columns]
        return self.wrapper.wrap(obj.values, index=other.index, columns=other.columns, group_by=False)

    @class_or_instancemethod
    def broadcast(self_or_cls, *others, **kwargs):
        """See `vectorbt.base.reshape_fns.broadcast`."""
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        if isinstance(self_or_cls, type):
            return reshape_fns.broadcast(*others, **kwargs)
        return reshape_fns.broadcast(self_or_cls._obj, *others, **kwargs)

    def broadcast_to(self, other, **kwargs):
        """See `vectorbt.base.reshape_fns.broadcast_to`."""
        if isinstance(other, Base_Accessor):
            other = other._obj
        return reshape_fns.broadcast_to(self._obj, other, **kwargs)

    def make_symmetric(self):  # pragma: no cover
        """See `vectorbt.base.reshape_fns.make_symmetric`."""
        return reshape_fns.make_symmetric(self._obj)

    def unstack_to_array(self, **kwargs):  # pragma: no cover
        """See `vectorbt.base.reshape_fns.unstack_to_array`."""
        return reshape_fns.unstack_to_array(self._obj, **kwargs)

    def unstack_to_df(self, **kwargs):  # pragma: no cover
        """See `vectorbt.base.reshape_fns.unstack_to_df`."""
        return reshape_fns.unstack_to_df(self._obj, **kwargs)

    # ############# Combining ############# #

    def apply(self, *args, apply_func=None, to_2d=False, **kwargs):
        """Apply a function `apply_func`.

        Arguments `*args` and `**kwargs` will be directly passed to `apply_func`.
        If `to_2d` is True, 2-dimensional NumPy arrays will be passed, otherwise as is.

        !!! note
            The resulted array must have the same shape as the original array.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> sr = pd.Series([1, 2], index=['x', 'y'])
        >>> sr2.vbt.apply(apply_func=lambda x: x ** 2)
        i2
        x2    1
        y2    4
        z2    9
        Name: a2, dtype: int64
        ```
        """
        checks.assert_not_none(apply_func)
        # Optionally cast to 2d array
        if to_2d:
            obj = reshape_fns.to_2d(self._obj, raw=True)
        else:
            obj = np.asarray(self._obj)
        result = apply_func(obj, *args, **kwargs)
        return self.wrapper.wrap(result, group_by=False)

    @class_or_instancemethod
    def concat(self_or_cls, *others, keys=None, broadcast_kwargs={}):
        """Concatenate with `others` along columns.

        All arguments will be broadcast using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`. Use `keys` as the outermost level.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> sr = pd.Series([1, 2], index=['x', 'y'])
        >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
        >>> sr.vbt.concat(df, keys=['c', 'd'])
              c     d
           a  b  a  b
        x  1  1  3  4
        y  2  2  5  6
        ```
        """
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        if isinstance(self_or_cls, type):
            objs = others
        else:
            objs = (self_or_cls._obj,) + others
        broadcasted = reshape_fns.broadcast(*objs, **broadcast_kwargs)
        broadcasted = tuple(map(reshape_fns.to_2d, broadcasted))
        out = pd.concat(broadcasted, axis=1, keys=keys)
        if not isinstance(out.columns, pd.MultiIndex) and np.all(out.columns == 0):
            out.columns = pd.RangeIndex(start=0, stop=len(out.columns), step=1)
        return out

    def apply_and_concat(self, ntimes, *args, apply_func=None, to_2d=False, keys=None, **kwargs):
        """Apply `apply_func` `ntimes` times and concatenate the results along columns.
        See `vectorbt.base.combine_fns.apply_and_concat_one`.

        Arguments `*args` and `**kwargs` will be directly passed to `apply_func`.
        If `to_2d` is True, 2-dimensional NumPy arrays will be passed, otherwise as is.
        Use `keys` as the outermost level.

        !!! note
            The resulted arrays to be concatenated must have the same shape as broadcast input arrays.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
        >>> df.vbt.apply_and_concat(3, [1, 2, 3],
        ...     apply_func=lambda i, a, b: a * b[i], keys=['c', 'd', 'e'])
              c       d       e
           a  b   a   b   a   b
        x  3  4   6   8   9  12
        y  5  6  10  12  15  18
        ```
        """
        checks.assert_not_none(apply_func)
        # Optionally cast to 2d array
        if to_2d:
            obj_arr = reshape_fns.to_2d(self._obj, raw=True)
        else:
            obj_arr = np.asarray(self._obj)
        if checks.is_numba_func(apply_func):
            result = combine_fns.apply_and_concat_one_nb(ntimes, apply_func, obj_arr, *args, **kwargs)
        else:
            result = combine_fns.apply_and_concat_one(ntimes, apply_func, obj_arr, *args, **kwargs)
        # Build column hierarchy
        if keys is not None:
            new_columns = index_fns.combine_indexes(keys, self.wrapper.columns)
        else:
            top_columns = pd.Index(np.arange(ntimes), name='apply_idx')
            new_columns = index_fns.combine_indexes(top_columns, self.wrapper.columns)
        return self.wrapper.wrap(result, columns=new_columns, group_by=False)

    def combine_with(self, other, *args, combine_func=None, to_2d=False, broadcast_kwargs={}, **kwargs):
        """Combine both using `combine_func` into a Series/DataFrame of the same shape.

        All arguments will be broadcast using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`.

        Arguments `*args` and `**kwargs` will be directly passed to `combine_func`.
        If `to_2d` is True, 2-dimensional NumPy arrays will be passed, otherwise as is.

        !!! note
            The resulted array must have the same shape as broadcast input arrays.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> sr = pd.Series([1, 2], index=['x', 'y'])
        >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
        >>> sr.vbt.combine_with(df, combine_func=lambda x, y: x + y)
           a  b
        x  4  5
        y  7  8
        ```
        """
        if isinstance(other, Base_Accessor):
            other = other._obj
        checks.assert_not_none(combine_func)
        if checks.is_numba_func(combine_func):
            # Numba requires writable arrays
            broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        new_obj, new_other = reshape_fns.broadcast(self._obj, other, **broadcast_kwargs)
        # Optionally cast to 2d array
        if to_2d:
            new_obj_arr = reshape_fns.to_2d(new_obj, raw=True)
            new_other_arr = reshape_fns.to_2d(new_other, raw=True)
        else:
            new_obj_arr = np.asarray(new_obj)
            new_other_arr = np.asarray(new_other)
        result = combine_func(new_obj_arr, new_other_arr, *args, **kwargs)
        return new_obj.vbt.wrapper.wrap(result)

    def combine_with_multiple(self, others, *args, combine_func=None, to_2d=False,
                              concat=False, broadcast_kwargs={}, keys=None, **kwargs):
        """Combine with `others` using `combine_func`.

        All arguments will be broadcast using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`.

        If `concat` is True, concatenate the results along columns,
        see `vectorbt.base.combine_fns.combine_and_concat`.
        Otherwise, pairwise combine into a Series/DataFrame of the same shape, 
        see `vectorbt.base.combine_fns.combine_multiple`.

        Arguments `*args` and `**kwargs` will be directly passed to `combine_func`. 
        If `to_2d` is True, 2-dimensional NumPy arrays will be passed, otherwise as is.
        Use `keys` as the outermost level.

        !!! note
            If `combine_func` is Numba-compiled, will broadcast using `WRITEABLE` and `C_CONTIGUOUS`
            flags, which can lead to an expensive computation overhead if passed objects are large and
            have different shape/memory order. You also must ensure that all objects have the same data type.

            Also remember to bring each in `*args` to a Numba-compatible format.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> sr = pd.Series([1, 2], index=['x', 'y'])
        >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])

        >>> sr.vbt.combine_with_multiple([df, df*2],
        ...     combine_func=lambda x, y: x + y)
            a   b
        x  10  13
        y  17  20

        >>> sr.vbt.combine_with_multiple([df, df*2],
        ...     combine_func=lambda x, y: x + y, concat=True, keys=['c', 'd'])
              c       d
           a  b   a   b
        x  4  5   7   9
        y  7  8  12  14
        ```
        """
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        checks.assert_not_none(combine_func)
        checks.assert_type(others, Iterable)
        # Broadcast arguments
        if checks.is_numba_func(combine_func):
            # Numba requires writeable arrays
            # Plus all of our arrays must be in the same order
            broadcast_kwargs = merge_dicts(dict(require_kwargs=dict(requirements=['W', 'C'])), broadcast_kwargs)
        new_obj, *new_others = reshape_fns.broadcast(self._obj, *others, **broadcast_kwargs)
        # Optionally cast to 2d array
        if to_2d:
            bc_arrays = tuple(map(lambda x: reshape_fns.to_2d(x, raw=True), (new_obj, *new_others)))
        else:
            bc_arrays = tuple(map(lambda x: np.asarray(x), (new_obj, *new_others)))
        if concat:
            # Concat the results horizontally
            if checks.is_numba_func(combine_func):
                for i in range(1, len(bc_arrays)):
                    checks.assert_meta_equal(bc_arrays[i - 1], bc_arrays[i])
                result = combine_fns.combine_and_concat_nb(bc_arrays[0], bc_arrays[1:], combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_and_concat(bc_arrays[0], bc_arrays[1:], combine_func, *args, **kwargs)
            columns = new_obj.vbt.wrapper.columns
            if keys is not None:
                new_columns = index_fns.combine_indexes(keys, columns)
            else:
                top_columns = pd.Index(np.arange(len(new_others)), name='combine_idx')
                new_columns = index_fns.combine_indexes(top_columns, columns)
            return new_obj.vbt.wrapper.wrap(result, columns=new_columns)
        else:
            # Combine arguments pairwise into one object
            if checks.is_numba_func(combine_func):
                for i in range(1, len(bc_arrays)):
                    checks.assert_dtype_equal(bc_arrays[i - 1], bc_arrays[i])
                result = combine_fns.combine_multiple_nb(bc_arrays, combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_multiple(bc_arrays, combine_func, *args, **kwargs)
            return new_obj.vbt.wrapper.wrap(result)


class Base_SRAccessor(Base_Accessor):
    """Accessor on top of Series.

    Accessible through `pd.Series.vbt` and all child accessors."""

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        checks.assert_type(obj, pd.Series)

        Base_Accessor.__init__(self, obj, **kwargs)

    @class_or_instancemethod
    def is_series(self_or_cls):
        return True

    @class_or_instancemethod
    def is_frame(self_or_cls):
        return False


class Base_DFAccessor(Base_Accessor):
    """Accessor on top of DataFrames.

    Accessible through `pd.DataFrame.vbt` and all child accessors."""

    def __init__(self, obj, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        checks.assert_type(obj, pd.DataFrame)

        Base_Accessor.__init__(self, obj, **kwargs)

    @class_or_instancemethod
    def is_series(self_or_cls):
        return False

    @class_or_instancemethod
    def is_frame(self_or_cls):
        return True
