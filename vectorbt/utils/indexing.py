"""Class decorators for indexing."""

import numpy as np
import pandas as pd
from functools import update_wrapper
import inspect
from types import FunctionType

from vectorbt.utils import checks, index_fns, reshape_fns


def copy_func(f):
    """Copy function.
    
    Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = FunctionType(f.__code__, f.__globals__, name=f.__name__,
                     argdefs=f.__defaults__,
                     closure=f.__closure__)
    g = update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def add_pd_indexing(indexing_func):
    """Add `iloc`, `loc`, `xs` and `__getitem__` indexers to a class.

    Delegates indexing operation via `indexing_func`, which must accept an instance of
    the class and a pandas indexing function.

    !!! note
        Remember than `pd_indexing_func` passed to `indexing_func` doesn't care about which
        pandas object you pass, so you must ensure that each object has same index/columns.
    
    Example:
        ```python-repl
        >>> import pandas as pd
        >>> from vectorbt.utils.indexing import add_pd_indexing

        >>> def indexing_func(c, pd_indexing_func):
        ...     return C(pd_indexing_func(c.df1), pd_indexing_func(c.df2))

        >>> @add_pd_indexing(indexing_func)
        ... class C():
        ...     def __init__(self, df1, df2):
        ...         self.df1 = df1
        ...         self.df2 = df2

        >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
        >>> c = C(df1, df2)

        >>> print(c.iloc[0])
        <__main__.C at 0x623c27cf8>

        >>> print(c.iloc[0].df1)
        a    1
        b    3
        Name: 0, dtype: int64

        >>> print(c.iloc[0].df2)
        a    5
        b    7
        Name: 0, dtype: int64
        ```
        """
    def wrapper(cls):
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


def mapper_indexing_func(mapper, like_df, pd_indexing_func):
    """Broadcast `mapper` Series to DataFrame `like_df` and perform pandas indexing using `pd_indexing_func`."""
    checks.assert_type(mapper, pd.Series)
    checks.assert_type(like_df, (pd.Series, pd.DataFrame))

    df_range_mapper = reshape_fns.broadcast_to(np.arange(len(mapper.index)), like_df)
    loced_range_mapper = pd_indexing_func(df_range_mapper)
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    if checks.is_frame(loced_range_mapper):
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    elif checks.is_series(loced_range_mapper):
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)


def add_param_indexing(param_name, indexing_func):
    """Add parameter indexer to a class.

    Parameter indexer enables accessing a group of rows and columns by a parameter array (similar to `loc`). 
    This way, one can query index/columns by another Series called a parameter mapper, which is just a
    `pandas.Series` that maps columns (its index) to params (its values).
    
    Parameter indexing is important, since querying by column/index labels alone is not always sufficient.
    For example, `pandas` doesn't let you query by list at a specific index/column level.
    
    !!! note
        This method requires from class after `__init__` having the `_{param_name}_mapper` attribute that holds 
        the parameter mapper. It will then create an indexer under the `{param_name}_loc` attribute. It uses
        the same `pd_indexing_func` as `add_pd_indexing`.
    
    Example:
        ```python-repl
        >>> import pandas as pd
        >>> from vectorbt.utils.indexing import add_param_indexing, mapper_indexing_func

        >>> def indexing_func(c, pd_indexing_func):
        ...     return C(pd_indexing_func(c.df), mapper_indexing_func(
                    c._my_param_mapper, c.df, pd_indexing_func))

        >>> @add_param_indexing('my_param', indexing_func)
        ... class C():
        ...     def __init__(self, df, param_mapper):
        ...         self.df = df
        ...         self._my_param_mapper = param_mapper # important!
                
        >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        >>> param_mapper = pd.Series(['First', 'Second'], index=['a', 'b'])
        >>> c = C(df, param_mapper)

        >>> print(c.my_param_loc['First'])
        <__main__.C object at 0x61f8b0a58>

        >>> print(c.my_param_loc['First'].df)
        0    1
        1    2
        Name: a, dtype: int64

        >>> print(c.my_param_loc['Second'].df)
        0    3
        1    4
        Name: b, dtype: int64

        >>> print(c.my_param_loc[['First', 'First', 'Second', 'Second']].df)
              a     b
        0  1  1  3  3
        1  2  2  4  4
        ```
        ```"""
    def wrapper(cls):
        class ParamLoc:
            def __init__(self, obj, mapper, level_names=None):
                checks.assert_type(mapper, pd.Series)

                self.obj = obj
                if mapper.dtype == 'O':
                    # If params are objects, we must cast them to string first
                    # The original mapper isn't touched
                    mapper = mapper.astype(str)
                self.mapper = mapper

            def get_indices(self, key):
                if self.mapper.dtype == 'O':
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
                mapper = self.mapper
                # Use pandas to perform indexing
                mapper = pd.Series(np.arange(len(mapper.index)), index=mapper.values)
                indices = mapper.loc.__getitem__(key)
                if isinstance(indices, pd.Series):
                    indices = indices.values
                return indices

            def __getitem__(self, key):
                indices = self.get_indices(key)
                is_multiple = isinstance(key, (slice, list, np.ndarray))
                level_name = self.mapper.name  # name of the mapper should contain level names of the params

                def pd_indexing_func(obj):
                    new_obj = obj.iloc[:, indices]
                    if not is_multiple:
                        # If we selected only one param, then remove its columns levels to keep it clean
                        if level_name is not None:
                            if checks.is_frame(new_obj):
                                if isinstance(new_obj.columns, pd.MultiIndex):
                                    new_obj.columns = index_fns.drop_levels(new_obj.columns, level_name)
                    return new_obj

                return indexing_func(self.obj, pd_indexing_func)

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
