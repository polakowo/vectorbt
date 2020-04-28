import numpy as np
import pandas as pd
from collections.abc import Iterable

from vectorbt.utils import checks, combine_fns, index_fns, reshape_fns
from vectorbt.utils.common import class_or_instancemethod, fix_class_for_pdoc


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
        return reshape_fns.to_1d(self._obj, raw=True)

    def to_2d_array(self):
        return reshape_fns.to_2d(self._obj, raw=True)

    def wrap_array(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def tile(self, n, as_columns=None):
        tiled = reshape_fns.tile(self._obj, n, along_axis=1)
        if as_columns is not None:
            new_columns = index_fns.combine(as_columns, reshape_fns.to_2d(self._obj).columns)
            return self.wrap_array(tiled.values, columns=new_columns)
        return tiled

    def repeat(self, n, as_columns=None):
        repeated = reshape_fns.repeat(self._obj, n, along_axis=1)
        if as_columns is not None:
            new_columns = index_fns.combine(reshape_fns.to_2d(self._obj).columns, as_columns)
            return self.wrap_array(repeated.values, columns=new_columns)
        return repeated

    def align_to(self, other):
        checks.assert_type(other, (pd.Series, pd.DataFrame))
        obj = reshape_fns.to_2d(self._obj)
        other = reshape_fns.to_2d(other)

        aligned_index = index_fns.align_to(obj.index, other.index)
        aligned_columns = index_fns.align_to(obj.columns, other.columns)
        obj = obj.iloc[aligned_index, aligned_columns]
        return self.wrap_array(obj.values, index=other.index, columns=other.columns)

    @class_or_instancemethod
    def broadcast(self_or_cls, *others, **kwargs):
        others = tuple(map(lambda x: x._obj if isinstance(x, Base_Accessor) else x, others))
        if isinstance(self_or_cls, type):
            return reshape_fns.broadcast(*others, **kwargs)
        return reshape_fns.broadcast(self_or_cls._obj, *others, **kwargs)

    def broadcast_to(self, other, **kwargs):
        if isinstance(other, Base_Accessor):
            other = other._obj
        return reshape_fns.broadcast_to(self._obj, other, **kwargs)

    def make_symmetric(self):
        return reshape_fns.make_symmetric(self._obj)

    def unstack_to_array(self, **kwargs):
        return reshape_fns.unstack_to_array(self._obj, **kwargs)

    def unstack_to_df(self, **kwargs):
        return reshape_fns.unstack_to_df(self._obj, **kwargs)

    @class_or_instancemethod
    def concat(self_or_cls, *others, as_columns=None, broadcast_kwargs={}):
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

    def apply_and_concat(self, ntimes, *args, apply_func=None, as_columns=None, **kwargs):
        """Apply a function n times and concatenate results into a single dataframe."""
        checks.assert_not_none(apply_func)
        if checks.is_numba_func(apply_func):
            # NOTE: your apply_func must a numba-compiled function and arguments must be numba-compatible
            # Also NOTE: outputs of apply_func must always be 2-dimensional
            result = combine_fns.apply_and_concat_nb(np.asarray(self._obj), ntimes, apply_func, *args, **kwargs)
        else:
            result = combine_fns.apply_and_concat(np.asarray(self._obj), ntimes, apply_func, *args, **kwargs)
        # Build column hierarchy
        if as_columns is not None:
            new_columns = index_fns.combine(as_columns, reshape_fns.to_2d(self._obj).columns)
        else:
            new_columns = index_fns.tile(reshape_fns.to_2d(self._obj).columns, ntimes)
        return self.wrap_array(result, columns=new_columns)

    def combine_with(self, other, *args, combine_func=None, broadcast_kwargs={}, **kwargs):
        """Broadcast with other and combine.

        The returned shape is the same as broadcasted shape."""
        if isinstance(other, Base_Accessor):
            other = other._obj
        checks.assert_not_none(combine_func)
        if checks.is_numba_func(combine_func):
            # Numba requires writable arrays
            broadcast_kwargs = {**dict(writeable=True), **broadcast_kwargs}
        new_obj, new_other = reshape_fns.broadcast(self._obj, other, **broadcast_kwargs)
        return new_obj.vbt.wrap_array(combine_func(np.asarray(new_obj), np.asarray(new_other), *args, **kwargs))

    def combine_with_multiple(self, others, *args, combine_func=None, concat=False,
                              broadcast_kwargs={}, as_columns=None, **kwargs):
        """Broadcast with other objects to the same shape and combine them all pairwise.

        The returned shape is the same as broadcasted shape if concat is False.
        The returned shape is concatenation of broadcasted shapes if concat is True."""
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
        broadcasted = tuple(map(np.asarray, (new_obj, *new_others)))
        if concat:
            # Concat the results horizontally
            if checks.is_numba_func(combine_func):
                for i in range(1, len(broadcasted)):
                    # NOTE: all inputs must have the same dtype
                    checks.assert_same_meta(broadcasted[i-1], broadcasted[i])
                result = combine_fns.combine_and_concat_nb(broadcasted[0], broadcasted[1:], combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_and_concat(broadcasted[0], broadcasted[1:], combine_func, *args, **kwargs)
            if as_columns is not None:
                new_columns = index_fns.combine(as_columns, reshape_fns.to_2d(new_obj).columns)
            else:
                new_columns = index_fns.tile(reshape_fns.to_2d(new_obj).columns, len(others))
            return new_obj.vbt.wrap_array(result, columns=new_columns)
        else:
            # Combine arguments pairwise into one object
            if checks.is_numba_func(combine_func):
                for i in range(1, len(broadcasted)):
                    # NOTE: all inputs must have the same dtype
                    checks.assert_same_dtype(broadcasted[i-1], broadcasted[i])
                result = combine_fns.combine_multiple_nb(broadcasted, combine_func, *args, **kwargs)
            else:
                result = combine_fns.combine_multiple(broadcasted, combine_func, *args, **kwargs)
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
        checks.assert_type(obj, pd.DataFrame)

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
        return reshape_fns.wrap_array(
            a,
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
        checks.assert_type(obj, pd.Series)

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
        return reshape_fns.wrap_array(
            a,
            index=index,
            columns=columns,
            dtype=dtype,
            default_index=self._obj.index,
            default_columns=[self._obj.name],
            to_ndim=1)
