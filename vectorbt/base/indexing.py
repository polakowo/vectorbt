"""Classes for indexing."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.base import index_fns, reshape_fns


class IndexingError(Exception):
    """Exception raised when an indexing error has occurred."""


class _iLoc:
    """Forwards `pd.Series.iloc`/`pd.DataFrame.iloc`
    operation to each Series/DataFrame and returns a new class instance."""

    def __init__(self, obj, indexing_func, **kwargs):
        self._obj = obj
        self._indexing_func = indexing_func
        self._indexing_kwargs = kwargs

    def __getitem__(self, key):
        return self._indexing_func(lambda x: x.iloc.__getitem__(key), **self._indexing_kwargs)


class _Loc:
    """Forwards `pd.Series.loc`/`pd.DataFrame.loc`
    operation to each Series/DataFrame and returns a new class instance."""

    def __init__(self, obj, indexing_func, **kwargs):
        self._obj = obj
        self._indexing_func = indexing_func
        self._indexing_kwargs = kwargs

    def __getitem__(self, key):
        return self._indexing_func(lambda x: x.loc.__getitem__(key), **self._indexing_kwargs)


class PandasIndexer:
    """Implements indexing using `iloc`, `loc`, `xs` and `__getitem__`.

    ## Example

    ```python-repl
    >>> import pandas as pd
    >>> from vectorbt.base.indexing import PandasIndexer

    >>> class C(PandasIndexer):
    ...     def __init__(self, df1, df2):
    ...         self.df1 = df1
    ...         self.df2 = df2
    ...         super().__init__()
    ...
    ...     def _indexing_func(self, pd_indexing_func):
    ...         return self.__class__(
    ...             pd_indexing_func(self.df1),
    ...             pd_indexing_func(self.df2)
    ...         )

    >>> df1 = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> df2 = pd.DataFrame({'a': [5, 6], 'b': [7, 8]})
    >>> c = C(df1, df2)

    >>> c.iloc[:, 0]
    <__main__.C object at 0x1a1cacbbe0>

    >>> c.iloc[:, 0].df1
    0    1
    1    2
    Name: a, dtype: int64

    >>> c.iloc[:, 0].df2
    0    5
    1    6
    Name: a, dtype: int64
    ```
    """

    def __init__(self,  **kwargs):
        self._iloc = _iLoc(self, self._indexing_func, **kwargs)
        self._loc = _Loc(self, self._indexing_func, **kwargs)
        self._indexing_kwargs = kwargs

    @property
    def iloc(self):
        return self._iloc

    iloc.__doc__ = _iLoc.__doc__

    @property
    def loc(self):
        return self._loc

    loc.__doc__ = _Loc.__doc__

    def xs(self, *args, **kwargs):
        """Forwards `pd.Series.xs`/`pd.DataFrame.xs`
        operation to each Series/DataFrame and returns a new class instance."""
        return self._indexing_func(lambda x: x.xs(*args, **kwargs), **self._indexing_kwargs)

    def __getitem__(self, key):
        return self._indexing_func(lambda x: x.__getitem__(key), **self._indexing_kwargs)

    def _indexing_func(self, pd_indexing_func, **kwargs):
        """Apply `pd_indexing_func` on all pandas objects in question
        and return a new instance of the class.

        Replace by your custom function."""
        raise NotImplementedError


class _ParamLoc:
    """Access a group of columns by parameter using `pd.Series.loc`.

    Uses `mapper` to establish link between columns and parameter values."""

    def __init__(self, obj, mapper, indexing_func, level_name=None, **kwargs):
        checks.assert_type(mapper, pd.Series)

        self._obj = obj
        if mapper.dtype == 'O':
            # If params are objects, we must cast them to string first
            # The original mapper isn't touched
            mapper = mapper.astype(str)
        self._mapper = mapper
        self._indexing_func = indexing_func
        self._level_name = level_name
        self._indexing_kwargs = kwargs

    def get_indices(self, key):
        if self._mapper.dtype == 'O':
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
        mapper = self._mapper
        # Use pandas to perform indexing
        mapper = pd.Series(np.arange(len(mapper.index)), index=mapper.values)
        indices = mapper.loc.__getitem__(key)
        if isinstance(indices, pd.Series):
            indices = indices.values
        return indices

    def __getitem__(self, key):
        indices = self.get_indices(key)
        is_multiple = isinstance(key, (slice, list, np.ndarray))

        def pd_indexing_func(obj):
            new_obj = obj.iloc[:, indices]
            if not is_multiple:
                # If we selected only one param, then remove its columns levels to keep it clean
                if self._level_name is not None:
                    if checks.is_frame(new_obj):
                        if isinstance(new_obj.columns, pd.MultiIndex):
                            new_obj.columns = index_fns.drop_levels(new_obj.columns, self._level_name)
            return new_obj

        return self._indexing_func(pd_indexing_func, **self._indexing_kwargs)


def indexing_on_mapper(mapper, ref_obj, pd_indexing_func):
    """Broadcast `mapper` Series to `ref_obj` and perform pandas indexing using `pd_indexing_func`."""
    checks.assert_type(mapper, pd.Series)
    checks.assert_type(ref_obj, (pd.Series, pd.DataFrame))

    df_range_mapper = reshape_fns.broadcast_to(np.arange(len(mapper.index)), ref_obj)
    loced_range_mapper = pd_indexing_func(df_range_mapper)
    new_mapper = mapper.iloc[loced_range_mapper.values[0]]
    if checks.is_frame(loced_range_mapper):
        return pd.Series(new_mapper.values, index=loced_range_mapper.columns, name=mapper.name)
    elif checks.is_series(loced_range_mapper):
        return pd.Series([new_mapper], index=[loced_range_mapper.name], name=mapper.name)


class ParamIndexerFactory:
    """A factory to create a class with parameter indexing.

    Parameter indexer enables accessing a group of rows and columns by a parameter array (similar to `loc`).
    This way, one can query index/columns by another Series called a parameter mapper, which is just a
    `pd.Series` that maps columns (its index) to params (its values).

    Parameter indexing is important, since querying by column/index labels alone is not always the best option.
    For example, `pandas` doesn't let you query by list at a specific index/column level.

    Args:
        param_names (list of str): Names of the parameters.

    ## Example

    ```python-repl
    >>> import pandas as pd
    >>> from vectorbt.base.indexing import ParamIndexerFactory, indexing_on_mapper

    >>> MyParamIndexer = ParamIndexerFactory(['my_param'])
    ... class C(MyParamIndexer):
    ...     def __init__(self, df, param_mapper):
    ...         self.df = df
    ...         self._my_param_mapper = param_mapper
    ...         super().__init__([param_mapper])
    ...
    ...     def _indexing_func(self, pd_indexing_func):
    ...         return self.__class__(
    ...             pd_indexing_func(self.df),
    ...             indexing_on_mapper(self._my_param_mapper, self.df, pd_indexing_func)
    ...         )

    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> param_mapper = pd.Series(['First', 'Second'], index=['a', 'b'])
    >>> c = C(df, param_mapper)

    >>> c.my_param_loc['First'].df
    0    1
    1    2
    Name: a, dtype: int64

    >>> c.my_param_loc['Second'].df
    0    3
    1    4
    Name: b, dtype: int64

    >>> c.my_param_loc[['First', 'First', 'Second', 'Second']].df
          a     b
    0  1  1  3  3
    1  2  2  4  4
    ```
    """

    def __new__(self, param_names, class_name='ParamIndexer', module_name='vectorbt.base.indexing'):

        class ParamIndexer:
            def __init__(self, param_mappers, level_names=None, **kwargs):
                checks.assert_len_equal(param_names, param_mappers)

                for i, param_name in enumerate(param_names):
                    level_name = level_names[i] if level_names is not None else None
                    _param_loc = _ParamLoc(self, param_mappers[i], self._indexing_func, level_name=level_name, **kwargs)
                    setattr(self, f'_{param_name}_loc', _param_loc)

            def _indexing_func(self, pd_indexing_func, **kwargs):
                """Apply `pd_indexing_func` on all pandas objects in question
                and return a new instance of the class.

                Replace by your custom function."""
                raise NotImplementedError

        for i, param_name in enumerate(param_names):
            @property
            def param_loc(self, param_name=param_name):
                return getattr(self, f'_{param_name}_loc')

            param_loc.__doc__ = f"""Access a group of columns by parameter `{param_name}` using `pd.Series.loc`.
            
            Forwards this operation to each Series/DataFrame and returns a new class instance.
            """

            setattr(ParamIndexer, f'{param_name}_loc', param_loc)

        ParamIndexer.__name__ = class_name
        ParamIndexer.__qualname__ = class_name
        ParamIndexer.__module__ = module_name

        return ParamIndexer
