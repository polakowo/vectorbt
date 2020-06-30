"""Custom pandas accessors."""

import numpy as np
import pandas as pd
from collections.abc import Iterable

from vectorbt.utils import checks
from vectorbt.utils.decorators import class_or_instancemethod
from vectorbt.base import combine_fns, index_fns, reshape_fns
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.base.common import add_binary_magic_methods, add_unary_magic_methods, AbstractOps


@add_binary_magic_methods([
    # comparison ops
    ('__eq__', np.equal),
    ('__ne__', np.not_equal),
    ('__lt__', np.less),
    ('__gt__', np.greater),
    ('__le__', np.less_equal),
    ('__ge__', np.greater_equal),
    # arithmetic ops
    ('__add__', np.add),
    ('__sub__', np.subtract),
    ('__mul__', np.multiply),
    ('__pow__', np.power),
    ('__mod__', np.mod),
    ('__floordiv__', np.floor_divide),
    ('__truediv__', np.true_divide),
    ('__radd__', lambda x, y: np.add(y, x)),
    ('__rsub__', lambda x, y: np.subtract(y, x)),
    ('__rmul__', lambda x, y: np.multiply(y, x)),
    ('__rpow__', lambda x, y: np.power(y, x)),
    ('__rmod__', lambda x, y: np.mod(y, x)),
    ('__rfloordiv__', lambda x, y: np.floor_divide(y, x)),
    ('__rtruediv__', lambda x, y: np.true_divide(y, x)),
    # mask ops
    ('__and__', np.bitwise_and),
    ('__or__', np.bitwise_or),
    ('__xor__', np.bitwise_xor),
    ('__rand__', lambda x, y: np.bitwise_and(y, x)),
    ('__ror__', lambda x, y: np.bitwise_or(y, x)),
    ('__rxor__', lambda x, y: np.bitwise_xor(y, x))
])
@add_unary_magic_methods([
    ('__neg__', np.negative),
    ('__pos__', np.positive),
    ('__abs__', np.absolute),
    ('__invert__', np.invert)
])
class Base_Accessor(ArrayWrapper, AbstractOps):
    """Accessor on top of any data series. For both, Series and DataFrames.

    Accessible through `pandas.Series.vbt` and `pandas.DataFrame.vbt`, and all child accessors.

    Series is just a DataFrame with one column, hence to avoid defining methods exclusively for 1-dim data,
    we will convert any Series to a DataFrame and perform matrix computation on it. Afterwards,
    by using `Base_Accessor.wrap`, we will convert the 2-dim output back to a Series."""

    def __init__(self, obj):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._obj = obj

        # Initialize array wrapper
        wrapper = ArrayWrapper.from_obj(obj)
        ArrayWrapper.__init__(self, index=wrapper.index, columns=wrapper.columns, ndim=wrapper.ndim)

    def __call__(self, *args, **kwargs):
        """Allows passing arguments to the initializer."""

        return self.__class__(self._obj, *args, **kwargs)

    # ############# Creation ############# #

    @classmethod
    def empty(cls, shape, fill_value=np.nan, **kwargs):
        """Generate an empty Series/DataFrame of shape `shape` and fill with `fill_value`."""
        if not isinstance(shape, tuple) or (isinstance(shape, tuple) and len(shape) == 1):
            return pd.Series(np.full(shape, fill_value), **kwargs)
        return pd.DataFrame(np.full(shape, fill_value), **kwargs)

    @classmethod
    def empty_like(cls, other, fill_value=np.nan):
        """Generate an empty Series/DataFrame like `other` and fill with `fill_value`."""
        if checks.is_series(other):
            return cls.empty(other.shape, fill_value=fill_value, index=other.index, name=other.name)
        return cls.empty(other.shape, fill_value=fill_value, index=other.index, columns=other.columns)

    # ############# Index and columns ############# #

    def apply_on_index(self, apply_func, *args, axis=1, inplace=False, **kwargs):
        """Apply function `apply_func` on index of the pandas object.

        Set `axis` to 1 for columns and 0 for index.
        If `inplace` is `True`, modifies the pandas object. Otherwise, returns a copy."""
        checks.assert_value_in(axis, (0, 1))

        if axis == 1:
            obj_index = self.columns
        else:
            obj_index = self.index
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

        Set `on_top` to `False` to stack at bottom.

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

    def tile(self, n, as_columns=None):
        """See `vectorbt.base.reshape_fns.tile`.

        Use `as_columns` as a top-level column level."""
        tiled = reshape_fns.tile(self._obj, n, axis=1)
        if as_columns is not None:
            new_columns = index_fns.combine_indexes(as_columns, self.columns)
            return self.wrap(tiled.values, columns=new_columns)
        return tiled

    def repeat(self, n, as_columns=None):
        """See `vectorbt.base.reshape_fns.repeat`.

        Use `as_columns` as a top-level column level."""
        repeated = reshape_fns.repeat(self._obj, n, axis=1)
        if as_columns is not None:
            new_columns = index_fns.combine_indexes(self.columns, as_columns)
            return self.wrap(repeated.values, columns=new_columns)
        return repeated

    def align_to(self, other):
        """Align to `other` by their indexes and columns.

        Example:
            ```python-repl
            >>> import vectorbt as vbt
            >>> import pandas as pd
            >>> df1 = pd.DataFrame([[1, 2], [3, 4]], index=['x', 'y'], columns=['a', 'b'])
            >>> df2 = pd.DataFrame([[5, 6, 7, 8], [9, 10, 11, 12]], index=['x', 'y'], 
            ...     columns=pd.MultiIndex.from_arrays([[1, 1, 2, 2], ['a', 'b', 'a', 'b']]))

            >>> print(df1.vbt.align_to(df2))
                  1     2   
               a  b  a  b
            x  1  2  1  2
            y  3  4  3  4
            ```"""
        checks.assert_type(other, (pd.Series, pd.DataFrame))
        obj = reshape_fns.to_2d(self._obj)
        other = reshape_fns.to_2d(other)

        aligned_index = index_fns.align_index_to(obj.index, other.index)
        aligned_columns = index_fns.align_index_to(obj.columns, other.columns)
        obj = obj.iloc[aligned_index, aligned_columns]
        return self.wrap(obj.values, index=other.index, columns=other.columns)

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

    def make_symmetric(self):
        """See `vectorbt.base.reshape_fns.make_symmetric`."""
        return reshape_fns.make_symmetric(self._obj)

    def unstack_to_array(self, **kwargs):
        """See `vectorbt.base.reshape_fns.unstack_to_array`."""
        return reshape_fns.unstack_to_array(self._obj, **kwargs)

    def unstack_to_df(self, **kwargs):
        """See `vectorbt.base.reshape_fns.unstack_to_df`."""
        return reshape_fns.unstack_to_df(self._obj, **kwargs)

    # ############# Combining ############# #

    def apply(self, *args, apply_func=None, pass_2d=False, **kwargs):
        """Apply a function `apply_func`.

        Arguments `*args` and `**kwargs` will be directly passed to `apply_func`.
        If `pass_2d` is `True`, 2-dimensional NumPy arrays will be passed, otherwise as is.

        !!! note
            The resulted array must have the same shape as the original array.

        Example:
            ```python-repl
            >>> import vectorbt as vbt
            >>> import pandas as pd
            >>> sr = pd.Series([1, 2], index=['x', 'y'])

            >>> print(sr2.vbt.apply(apply_func=lambda x: x ** 2))
            i2
            x2    1
            y2    4
            z2    9
            Name: a2, dtype: int64
            ```"""
        checks.assert_not_none(apply_func)
        # Optionally cast to 2d array
        if pass_2d:
            obj = reshape_fns.to_2d(self._obj, raw=True)
        else:
            obj = np.asarray(self._obj)
        result = apply_func(obj, *args, **kwargs)
        return self.wrap(result)

    @class_or_instancemethod
    def concat(self_or_cls, *others, as_columns=None, broadcast_kwargs={}):
        """Concatenate with `others` along columns.

        All arguments will be broadcasted using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`. Use `as_columns` as a top-level column level.

        Example:
            ```python-repl
            >>> import vectorbt as vbt
            >>> import pandas as pd
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])

            >>> print(sr.vbt.concat(df, as_columns=['c', 'd']))
                  c     d
               a  b  a  b
            x  1  1  3  4
            y  2  2  5  6
            ```"""
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        if isinstance(self_or_cls, type):
            objs = others
        else:
            objs = (self_or_cls._obj,) + others
        broadcasted = reshape_fns.broadcast(*objs, **broadcast_kwargs)
        broadcasted = tuple(map(reshape_fns.to_2d, broadcasted))
        concatenated = pd.concat(broadcasted, axis=1)
        if as_columns is not None:
            concatenated.columns = index_fns.combine_indexes(as_columns, broadcasted[0].columns)
        return concatenated

    def apply_and_concat(self, ntimes, *args, apply_func=None, pass_2d=False, as_columns=None, **kwargs):
        """Apply `apply_func` `ntimes` times and concatenate the results along columns.
        See `vectorbt.base.combine_fns.apply_and_concat_one`.

        Arguments `*args` and `**kwargs` will be directly passed to `apply_func`.
        If `pass_2d` is `True`, 2-dimensional NumPy arrays will be passed, otherwise as is.
        Use `as_columns` as a top-level column level.

        !!! note
            The resulted arrays to be concatenated must have the same shape as broadcasted input arrays.

        Example:
            ```python-repl
            >>> import vectorbt as vbt
            >>> import pandas as pd
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])

            >>> print(df.vbt.apply_and_concat(3, [1, 2, 3], 
            ...     apply_func=lambda i, a, b: a * b[i], as_columns=['c', 'd', 'e']))
                  c       d       e    
               a  b   a   b   a   b
            x  3  4   6   8   9  12
            y  5  6  10  12  15  18
            ```"""
        checks.assert_not_none(apply_func)
        # Optionally cast to 2d array
        if pass_2d:
            obj_arr = reshape_fns.to_2d(self._obj, raw=True)
        else:
            obj_arr = np.asarray(self._obj)
        if checks.is_numba_func(apply_func):
            result = combine_fns.apply_and_concat_one_nb(ntimes, apply_func, obj_arr, *args, **kwargs)
        else:
            result = combine_fns.apply_and_concat_one(ntimes, apply_func, obj_arr, *args, **kwargs)
        # Build column hierarchy
        if as_columns is not None:
            new_columns = index_fns.combine_indexes(as_columns, self.columns)
        else:
            new_columns = index_fns.tile_index(self.columns, ntimes)
        return self.wrap(result, columns=new_columns)

    def combine_with(self, other, *args, combine_func=None, pass_2d=False, broadcast_kwargs={}, **kwargs):
        """Combine both using `combine_func` into a Series/DataFrame of the same shape.

        All arguments will be broadcasted using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`.

        Arguments `*args` and `**kwargs` will be directly passed to `combine_func`.
        If `pass_2d` is `True`, 2-dimensional NumPy arrays will be passed, otherwise as is.

        !!! note
            The resulted array must have the same shape as broadcasted input arrays.

        Example:
            ```python-repl
            >>> import vectorbt as vbt
            >>> import pandas as pd
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])

            >>> print(sr.vbt.combine_with(df, combine_func=lambda x, y: x + y))
               a  b
            x  4  5
            y  7  8
            ```"""
        if isinstance(other, Base_Accessor):
            other = other._obj
        checks.assert_not_none(combine_func)
        if checks.is_numba_func(combine_func):
            # Numba requires writable arrays
            broadcast_kwargs = {**dict(writeable=True), **broadcast_kwargs}
        new_obj, new_other = reshape_fns.broadcast(self._obj, other, **broadcast_kwargs)
        # Optionally cast to 2d array
        if pass_2d:
            new_obj_arr = reshape_fns.to_2d(new_obj, raw=True)
            new_other_arr = reshape_fns.to_2d(new_other, raw=True)
        else:
            new_obj_arr = np.asarray(new_obj)
            new_other_arr = np.asarray(new_other)
        result = combine_func(new_obj_arr, new_other_arr, *args, **kwargs)
        return new_obj.vbt.wrap(result)

    def combine_with_multiple(self, others, *args, combine_func=None, pass_2d=False,
                              concat=False, broadcast_kwargs={}, as_columns=None, **kwargs):
        """Combine with `others` using `combine_func`.

        All arguments will be broadcasted using `vectorbt.base.reshape_fns.broadcast`
        with `broadcast_kwargs`.

        If `concat` is `True`, concatenate the results along columns, 
        see `vectorbt.base.combine_fns.combine_and_concat`.
        Otherwise, pairwise combine into a Series/DataFrame of the same shape, 
        see `vectorbt.base.combine_fns.combine_multiple`.

        Arguments `*args` and `**kwargs` will be directly passed to `combine_func`. 
        If `pass_2d` is `True`, 2-dimensional NumPy arrays will be passed, otherwise as is.
        Use `as_columns` as a top-level column level.

        !!! note
            If `combine_func` is Numba-compiled, will broadcast using `writeable=True` and
            copy using `order='C'` flags, which can lead to an expensive computation overhead if
            passed objects are large and have different shape/memory order. You also must ensure 
            that all objects have the same data type.

            Also remember to bring each in `*args` to a Numba-compatible format.

        Example:
            ```python-repl
            >>> import vectorbt as vbt
            >>> import pandas as pd
            >>> sr = pd.Series([1, 2], index=['x', 'y'])
            >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])

            >>> print(sr.vbt.combine_with_multiple([df, df*2], 
            ...     combine_func=lambda x, y: x + y))
                a   b
            x  10  13
            y  17  20

            >>> print(sr.vbt.combine_with_multiple([df, df*2], 
            ...     combine_func=lambda x, y: x + y, concat=True, as_columns=['c', 'd']))
                  c       d    
               a  b   a   b
            x  4  5   7   9
            y  7  8  12  14
            ```"""
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        checks.assert_not_none(combine_func)
        checks.assert_type(others, Iterable)
        # Broadcast arguments
        if checks.is_numba_func(combine_func):
            # Numba requires writable arrays
            broadcast_kwargs = {**dict(writeable=True), **broadcast_kwargs}
            # Plus all of our arrays must be in the same order
            broadcast_kwargs['copy_kwargs'] = {**dict(order='C'), **broadcast_kwargs.get('copy_kwargs', {})}
        new_obj, *new_others = reshape_fns.broadcast(self._obj, *others, **broadcast_kwargs)
        # Optionally cast to 2d array
        if pass_2d:
            bc_arrays = tuple(map(lambda x: reshape_fns.to_2d(x, raw=True), (new_obj, *new_others)))
        else:
            bc_arrays = tuple(map(lambda x: np.asarray(x), (new_obj, *new_others)))
        if concat:
            # Concat the results horizontally
            if checks.is_numba_func(combine_func):
                for i in range(1, len(bc_arrays)):
                    checks.assert_same_meta(bc_arrays[i - 1], bc_arrays[i])
                result = combine_fns.combine_and_concat_nb(bc_arrays[0], bc_arrays[1:], combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_and_concat(bc_arrays[0], bc_arrays[1:], combine_func, *args, **kwargs)
            columns = new_obj.vbt.columns
            if as_columns is not None:
                new_columns = index_fns.combine_indexes(as_columns, columns)
            else:
                new_columns = index_fns.tile_index(columns, len(others))
            return new_obj.vbt.wrap(result, columns=new_columns)
        else:
            # Combine arguments pairwise into one object
            if checks.is_numba_func(combine_func):
                for i in range(1, len(bc_arrays)):
                    checks.assert_same_dtype(bc_arrays[i - 1], bc_arrays[i])
                result = combine_fns.combine_multiple_nb(bc_arrays, combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_multiple(bc_arrays, combine_func, *args, **kwargs)
            return new_obj.vbt.wrap(result)


class Base_SRAccessor(Base_Accessor):
    """Accessor on top of any data series. For Series only.

    Accessible through `pandas.Series.vbt` and all child accessors."""

    def __init__(self, obj):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        checks.assert_type(obj, pd.Series)

        Base_Accessor.__init__(self, obj)

    @class_or_instancemethod
    def is_series(self_or_cls):
        return True

    @class_or_instancemethod
    def is_frame(self_or_cls):
        return False


class Base_DFAccessor(Base_Accessor):
    """Accessor on top of any data series. For DataFrames only.

    Accessible through `pandas.DataFrame.vbt` and all child accessors."""

    def __init__(self, obj):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        checks.assert_type(obj, pd.DataFrame)

        Base_Accessor.__init__(self, obj)

    @class_or_instancemethod
    def is_series(self_or_cls):
        return False

    @class_or_instancemethod
    def is_frame(self_or_cls):
        return True
