"""Custom pandas accessors with utility methods."""

import numpy as np
import pandas as pd
from collections.abc import Iterable

from vectorbt.utils import checks, combine_fns, index_fns, reshape_fns
from vectorbt.utils.common import class_or_instancemethod, fix_class_for_pdoc


class Base_Accessor():
    """Accessor with methods for both Series and DataFrames.

    Accessible through `pandas.Series.vbt` and `pandas.DataFrame.vbt`, and all child accessors.

    Series is just a DataFrame with one column, hence to avoid defining methods exclusively for 1-dim data,
    we will convert any Series to a DataFrame and perform matrix computation on it. Afterwards,
    by using `Base_Accessor.wrap_array`, we will convert the 2-dim output back to a Series."""

    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object
        self._validate(self._obj)

    @classmethod
    def _validate(cls, obj):
        """Define your validation logic here."""
        pass

    def validate(self):
        """Call this method to instantiate the accessor and invoke `Base_Accessor._validate`."""
        pass

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

    @property
    def index(self):
        """Return index of Series/DataFrame."""
        return self._obj.index

    @property
    def columns(self):
        """Return `[name]` of Series and `columns` of DataFrame."""
        if checks.is_series(self._obj):
            return pd.Index([self._obj.name])
        return self._obj.columns

    def to_array(self):
        """Convert to NumPy array."""
        return np.asarray(self._obj)

    def to_1d_array(self):
        """Convert to 1-dim NumPy array

        See `vectorbt.utils.reshape_fns.to_1d`."""
        return reshape_fns.to_1d(self._obj, raw=True)

    def to_2d_array(self):
        """Convert to 2-dim NumPy array.

        See `vectorbt.utils.reshape_fns.to_2d`."""
        return reshape_fns.to_2d(self._obj, raw=True)

    def wrap_array(self, a, **kwargs):
        """See `vectorbt.utils.reshape_fns.wrap_array_as`."""
        return reshape_fns.wrap_array_as(a, self._obj, **kwargs)

    def tile(self, n, as_columns=None):
        """See `vectorbt.utils.reshape_fns.tile`.

        Use `as_columns` as a top-level column level."""
        tiled = reshape_fns.tile(self._obj, n, axis=1)
        if as_columns is not None:
            new_columns = index_fns.combine(as_columns, self.columns)
            return self.wrap_array(tiled.values, columns=new_columns)
        return tiled

    def repeat(self, n, as_columns=None):
        """See `vectorbt.utils.reshape_fns.repeat`.

        Use `as_columns` as a top-level column level."""
        repeated = reshape_fns.repeat(self._obj, n, axis=1)
        if as_columns is not None:
            new_columns = index_fns.combine(self.columns, as_columns)
            return self.wrap_array(repeated.values, columns=new_columns)
        return repeated

    def align_to(self, other):
        """Align to `other` by their indexes and columns.

        Example:
            ```python-repl
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

        aligned_index = index_fns.align_to(obj.index, other.index)
        aligned_columns = index_fns.align_to(obj.columns, other.columns)
        obj = obj.iloc[aligned_index, aligned_columns]
        return self.wrap_array(obj.values, index=other.index, columns=other.columns)

    @class_or_instancemethod
    def broadcast(self_or_cls, *others, **kwargs):
        """See `vectorbt.utils.reshape_fns.broadcast`."""
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        if isinstance(self_or_cls, type):
            return reshape_fns.broadcast(*others, **kwargs)
        return reshape_fns.broadcast(self_or_cls._obj, *others, **kwargs)

    def broadcast_to(self, other, **kwargs):
        """See `vectorbt.utils.reshape_fns.broadcast_to`."""
        if isinstance(other, Base_Accessor):
            other = other._obj
        return reshape_fns.broadcast_to(self._obj, other, **kwargs)

    def make_symmetric(self):
        """See `vectorbt.utils.reshape_fns.make_symmetric`."""
        return reshape_fns.make_symmetric(self._obj)

    def unstack_to_array(self, **kwargs):
        """See `vectorbt.utils.reshape_fns.unstack_to_array`."""
        return reshape_fns.unstack_to_array(self._obj, **kwargs)

    def unstack_to_df(self, **kwargs):
        """See `vectorbt.utils.reshape_fns.unstack_to_df`."""
        return reshape_fns.unstack_to_df(self._obj, **kwargs)

    @class_or_instancemethod
    def concat(self_or_cls, *others, as_columns=None, broadcast_kwargs={}):
        """Concatenate with `others` along columns.

        All arguments will be broadcasted using `vectorbt.utils.reshape_fns.broadcast`
        with `broadcast_kwargs`. Use `as_columns` as a top-level column level.

        Example:
            ```python-repl
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
        if checks.is_pandas(broadcasted[0]):
            concated = pd.concat(broadcasted, axis=1)
            if as_columns is not None:
                concated.columns = index_fns.combine(as_columns, broadcasted[0].columns)
        else:
            concated = np.hstack(broadcasted)
        return concated

    def apply_and_concat(self, ntimes, *args, apply_func=None, pass_2d=False, as_columns=None, **kwargs):
        """Apply `apply_func` `ntimes` times and concatenate the results along columns.
        See `vectorbt.utils.combine_fns.apply_and_concat`.

        Arguments `*args` and `**kwargs` will be directly passed to `apply_func`.
        If `pass_2d` is `True`, 2-dimensional NumPy arrays will be passed, otherwise as is.
        Use `as_columns` as a top-level column level.

        Example:
            ```python-repl
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
            obj_arr = reshape_fns.to_2d(np.asarray(self._obj))
        else:
            obj_arr = np.asarray(self._obj)
        if checks.is_numba_func(apply_func):
            result = combine_fns.apply_and_concat_nb(obj_arr, ntimes, apply_func, *args, **kwargs)
        else:
            result = combine_fns.apply_and_concat(obj_arr, ntimes, apply_func, *args, **kwargs)
        # Build column hierarchy
        if as_columns is not None:
            new_columns = index_fns.combine(as_columns, self.columns)
        else:
            new_columns = index_fns.tile(self.columns, ntimes)
        return self.wrap_array(result, columns=new_columns)

    def combine_with(self, other, *args, combine_func=None, pass_2d=False, broadcast_kwargs={}, **kwargs):
        """Combine both using `combine_func` into a Series/DataFrame of the same shape.

        All arguments will be broadcasted using `vectorbt.utils.reshape_fns.broadcast`
        with `broadcast_kwargs`.

        Arguments `*args` and `**kwargs` will be directly passed to `combine_func`.
        If `pass_2d` is `True`, 2-dimensional NumPy arrays will be passed, otherwise as is.

        Example:
            ```python-repl
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
            new_obj_arr = reshape_fns.to_2d(np.asarray(new_obj))
            new_other_arr = reshape_fns.to_2d(np.asarray(new_other))
        else:
            new_obj_arr = np.asarray(new_obj)
            new_other_arr = np.asarray(new_other)
        result = combine_func(new_obj_arr, new_other_arr, *args, **kwargs)
        return new_obj.vbt.wrap_array(result)

    def combine_with_multiple(self, others, *args, combine_func=None, pass_2d=False,
                              concat=False, broadcast_kwargs={}, as_columns=None, **kwargs):
        """Combine with `others` using `combine_func`.

        All arguments will be broadcasted using `vectorbt.utils.reshape_fns.broadcast`
        with `broadcast_kwargs`.

        If `concat` is `True`, concatenate the results along columns, 
        see `vectorbt.utils.combine_fns.combine_and_concat`.
        Otherwise, pairwise combine into a Series/DataFrame of the same shape, 
        see `vectorbt.utils.combine_fns.combine_multiple`.

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
            bc_arrays = tuple(map(lambda x: reshape_fns.to_2d(np.asarray(x)), (new_obj, *new_others)))
        else:
            bc_arrays = tuple(map(lambda x: np.asarray(x), (new_obj, *new_others)))
        if concat:
            # Concat the results horizontally
            if checks.is_numba_func(combine_func):
                for i in range(1, len(bc_arrays)):
                    checks.assert_same_meta(bc_arrays[i-1], bc_arrays[i])
                result = combine_fns.combine_and_concat_nb(bc_arrays[0], bc_arrays[1:], combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_and_concat(bc_arrays[0], bc_arrays[1:], combine_func, *args, **kwargs)
            columns = new_obj.vbt.columns
            if as_columns is not None:
                new_columns = index_fns.combine(as_columns, columns)
            else:
                new_columns = index_fns.tile(columns, len(others))
            return new_obj.vbt.wrap_array(result, columns=new_columns)
        else:
            # Combine arguments pairwise into one object
            if checks.is_numba_func(combine_func):
                for i in range(1, len(bc_arrays)):
                    checks.assert_same_dtype(bc_arrays[i-1], bc_arrays[i])
                result = combine_fns.combine_multiple_nb(bc_arrays, combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_multiple(bc_arrays, combine_func, *args, **kwargs)
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


class Base_SRAccessor(Base_Accessor):
    """Accessor with methods for Series only.

    Accessible through `pandas.Series.vbt` and all child accessors."""

    @classmethod
    def _validate(cls, obj):
        checks.assert_type(obj, pd.Series)

    @class_or_instancemethod
    def is_series(self_or_cls):
        return True

    @class_or_instancemethod
    def is_frame(self_or_cls):
        return False


class Base_DFAccessor(Base_Accessor):
    """Accessor with methods for DataFrames only.

    Accessible through `pandas.DataFrame.vbt` and all child accessors."""

    @classmethod
    def _validate(cls, obj):
        checks.assert_type(obj, pd.DataFrame)

    @class_or_instancemethod
    def is_series(self_or_cls):
        return False

    @class_or_instancemethod
    def is_frame(self_or_cls):
        return True
