import numpy as np
import pandas as pd
from functools import update_wrapper
import inspect
from types import FunctionType

from vectorbt.utils import checks, indexes, reshape


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = FunctionType(f.__code__, f.__globals__, name=f.__name__,
                     argdefs=f.__defaults__,
                     closure=f.__closure__)
    g = update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def add_indexing(indexing_func):
    def wrapper(cls):
        """Add iloc, loc and __getitem__ indexing to a class.

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

        iloc.__doc__ = f"""Forwards [`pandas.Series.iloc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.iloc.html)/
        [`pandas.DataFrame.iloc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html)
        operation to each Series/DataFrame and returns a new instance of `{cls.__name__}`"""

        @property
        def loc(self):
            """Purely label-location based indexer for selection by label."""
            return self._loc

        loc.__doc__ = f"""Forwards [`pandas.Series.loc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.loc.html)/
        [`pandas.DataFrame.loc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html)
        operation to each Series/DataFrame and returns a new instance of `{cls.__name__}`"""

        def xs(self, *args, **kwargs):
            """Returns a cross-section (row(s) or column(s)) from the Series/DataFrame."""
            return indexing_func(self, lambda x: x.xs(*args, **kwargs))

        xs.__doc__ = f"""Forwards [`pandas.Series.xs`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.xs.html)/
        [`pandas.DataFrame.xs`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.xs.html)
        operation to each Series/DataFrame and returns a new instance of `{cls.__name__}`"""

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
    df_range_mapper = reshape.broadcast_to(np.arange(len(mapper.index)), like_df, index_from=1, columns_from=1)
    loced_range_mapper = loc_pandas_func(df_range_mapper)
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    if checks.is_frame(loced_range_mapper):
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    elif checks.is_series(loced_range_mapper):
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)


def add_param_indexing(param_name, indexing_func):
    def wrapper(cls):
        """Add loc indexing of params to a class.

        Uses a mapper, which is just a pd.Series object that maps columns to params."""

        class ParamLoc:
            def __init__(self, obj, param_mapper, level_names=None):
                checks.assert_type(param_mapper, pd.Series)

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
                            if checks.is_frame(new_obj):
                                if isinstance(new_obj.columns, pd.MultiIndex):
                                    new_obj.columns = indexes.drop_levels(new_obj.columns, level_name)
                    return new_obj

                return indexing_func(self.obj, loc_pandas_func)

        @property
        def param_loc(self):
            return getattr(self, f'_{param_name}_loc')

        param_loc.__doc__ = f"""Access a group of columns by parameter {param_name} using
        [`pandas.Series.loc`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.loc.html).
        
        Forwards this operation to each Series/DataFrame and returns a new instance of `{cls.__name__}`
        """

        orig_init_method = copy_func(cls.__init__)

        def __init__(self, *args, **kwargs):
            orig_init_method(self, *args, **kwargs)
            mapper = getattr(self, f'_{param_name}_mapper')
            setattr(self, f'_{param_name}_loc', ParamLoc(self, mapper))

        setattr(cls, '__init__', __init__)
        setattr(cls, f'{param_name}_loc', param_loc)
        return cls
    return wrapper
