"""An indicator factory for building new indicators with ease.

Each indicator is basically a pipeline that

* Accepts a list of time series objects (for example, OHLCV data)
* Accepts a list of parameter arrays (for example, rolling windows)
* Accepts other relevant arguments and keyword arguments
* Performs calculations to produce new time series objects (for example, rolling average)

This pipeline can be well standardized, which is done by this indicatory factory.

On top of this pipeline, it also does the following:

* Creates a new indicator class
* Creates an `__init__` method where it stores all inputs, outputs, and other artifacts
* Creates a `from_params` method that runs the main pipeline using `from_params_pipeline`
* Adds pandas indexing, i.e., you can use `iloc`, `loc`, `xs`, and `__getitem__` on the class itself
* Adds parameter indexing, i.e., use `*your_param*_loc` on the class to slice using parameters
* Adds user-defined properties
* Adds common comparison methods for all inputs, outputs and properties, e.g., crossovers

Example:
    Consider the following smaller price DataFrame `price_sm`:

    ```python-repl
    >>> import vectorbt as vbt
    >>> import numpy as np
    >>> import pandas as pd
    >>> from numba import njit
    >>> from datetime import datetime

    >>> index = pd.Index([
    ...     datetime(2020, 1, 1),
    ...     datetime(2020, 1, 2),
    ...     datetime(2020, 1, 3),
    ...     datetime(2020, 1, 4),
    ...     datetime(2020, 1, 5),
    ... ])
    >>> price_sm = pd.DataFrame({
    ...     'a': [1, 2, 3, 4, 5], 
    ...     'b': [5, 4, 3, 2, 1]}, index=index).astype(float)
    >>> print(price_sm)
                a    b
    2020-01-01  1.0  5.0
    2020-01-02  2.0  4.0
    2020-01-03  3.0  3.0
    2020-01-04  4.0  2.0
    2020-01-05  5.0  1.0
    ```

    For each column in the DataFrame, let's calculate a simple moving average and get signals 
    of price crossing it. In particular, we want to test two different window sizes: 2 and 3.

    A naive way of doing this:

    ```python-repl
    >>> ma_df = pd.DataFrame.vbt.concat(
    ...     price_sm.rolling(window=2).mean(), 
    ...     price_sm.rolling(window=3).mean(), 
    ...     as_columns=pd.Index([2, 3], name='ma_window'))
    >>> print(ma_df)
    ma_window          2         3
                a    b    a    b
    2020-01-01  NaN  NaN  NaN  NaN
    2020-01-02  1.5  4.5  NaN  NaN
    2020-01-03  2.5  3.5  2.0  4.0
    2020-01-04  3.5  2.5  3.0  3.0
    2020-01-05  4.5  1.5  4.0  2.0

    >>> above_signals = (price_sm.vbt.tile(2).vbt > ma_df)
    >>> above_signals = above_signals.vbt.signals.first(after_false=True)
    >>> print(above_signals)
    ma_window              2             3
                    a      b      a      b
    2020-01-01  False  False  False  False
    2020-01-02   True  False  False  False
    2020-01-03  False  False   True  False
    2020-01-04  False  False  False  False
    2020-01-05  False  False  False  False

    >>> below_signals = (price_sm.vbt.tile(2).vbt < ma_df)
    >>> below_signals = below_signals.vbt.signals.first(after_false=True)
    >>> print(below_signals)
    ma_window              2             3
                    a      b      a      b
    2020-01-01  False  False  False  False
    2020-01-02  False   True  False  False
    2020-01-03  False  False  False   True
    2020-01-04  False  False  False  False
    2020-01-05  False  False  False  False
    ```

    Now the same using `IndicatorFactory`:

    ```python-repl
    >>> MyMA = vbt.IndicatorFactory(
    ...     ts_names=['price_sm'],
    ...     param_names=['window'],
    ...     output_names=['ma'],
    ...     name='myma'
    ... ).from_apply_func(vbt.tseries.nb.rolling_mean_nb)

    >>> myma = MyMA.from_params(price_sm, [2, 3])
    >>> above_signals = myma.price_sm_above(myma.ma, crossed=True)
    >>> below_signals = myma.price_sm_below(myma.ma, crossed=True)
    ```

    It not only produced the handy `from_params` method, but generated a whole infrastructure to be run with
    an arbitrary number of windows. 

    For all our inputs in `ts_names` and outputs in `output_names`, it created a bunch of comparison methods 
    for generating signals, such as `above`, `below` and `equal` (use `doc()`): 

    ```python-repl
    'ma_above'
    'ma_below'
    'ma_equal'
    'price_sm_above'
    'price_sm_below'
    'price_sm_equal'
    ```

    Each of these methods uses vectorbt's own broadcasting, so you can compare time series objects with an 
    arbitrary array-like object, given their shapes can be broadcasted together. You can also compare them
    to multiple objects at once, for example:

    ```python-repl
    >>> print(myma.ma_above([1.5, 2.5], multiple=True))
    myma_ma_above                         1.5                         2.5
    myma_window               2             3             2             3
                    a      b      a      b      a      b      a      b
    2020-01-01     False  False  False  False  False  False  False  False
    2020-01-02     False   True  False  False  False   True  False  False
    2020-01-03      True   True   True   True  False   True  False   True
    2020-01-04      True   True   True   True   True  False   True   True
    2020-01-05      True  False   True   True   True  False   True  False
    ```

    `IndicatorFactory` also attached pandas indexing to the indicator class: 

    ```python-repl
    'iloc'
    'loc'
    'window_loc'
    'xs'
    ```

    This makes accessing rows and columns by labels, integer positions, and parameters much easier.

    The other advantage of using `IndicatorFactory` is broadcasting:

    * Passing multiple time series objects will broadcast them to the same shape and index/columns

    ```python-repl
    >>> price_sm2 = price_sm.copy() + 1
    >>> price_sm2.columns = ['a2', 'b2']

    >>> MyInd = vbt.IndicatorFactory(
    ...     ts_names=['price_sm', 'price_sm2'],
    ...     param_names=['p1', 'p2']
    ... ).from_apply_func(
    ...     lambda price_sm, price_sm2, p1, p2: price_sm * p1 + price_sm2 * p2
    ... )

    >>> myInd = MyInd.from_params(price_sm, price_sm2, 1, 2)
    >>> print(myInd.price_sm)
                a    b
                a2   b2
    2020-01-01  1.0  5.0
    2020-01-02  2.0  4.0
    2020-01-03  3.0  3.0
    2020-01-04  4.0  2.0
    2020-01-05  5.0  1.0
    >>> print(myInd.price_sm2)
                a    b
                a2   b2
    2020-01-01  2.0  6.0
    2020-01-02  3.0  5.0
    2020-01-03  4.0  4.0
    2020-01-04  5.0  3.0
    2020-01-05  6.0  2.0
    ```

    * Passing multiple parameters will broadcast them to arrays of the same shape

    ```python-repl
    >>> myInd = MyInd.from_params(price_sm, price_sm2, 1, 2)
    >>> print(myInd._p1_array)
    >>> print(myInd._p2_array)
    [1]
    [2]

    >>> myInd = MyInd.from_params(price_sm, price_sm2, 1, [2, 3])
    >>> print(myInd._p1_array)
    >>> print(myInd._p2_array)
    [1 1]
    [2 3]

    >>> myInd = MyInd.from_params(price_sm, price_sm2, [1, 2], [3, 4], param_product=True)
    >>> print(myInd._p1_array)
    >>> print(myInd._p2_array)
    [1 1 2 2]
    [3 4 3 4]
    ```

    This way, you can define parameter combinations of any order and shape. 
"""
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
import itertools

from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_property
from vectorbt.base import index_fns, reshape_fns, combine_fns
from vectorbt.base.indexing import PandasIndexer, ParamIndexerFactory, indexing_on_mapper
from vectorbt.tseries.common import TSArrayWrapper


def build_param_product(param_list):
    """Make Cartesian product out of all params in `param_list`."""
    param_list = list(map(reshape_fns.to_1d, param_list))
    param_list = list(zip(*list(itertools.product(*param_list))))
    param_list = list(map(np.asarray, param_list))
    return param_list


def build_column_hierarchy(param_list, level_names, ts_columns):
    """For each parameter in `param_list`, create a new column level with parameter values. 
    Combine this level with columns `ts_columns` using Cartesian product.
    
    Excludes level names that are `None`."""
    checks.assert_same_shape(param_list, level_names, axis=0)

    param_indexes = []
    for i in range(len(param_list)):
        if level_names[i] is not None:
            param_index = index_fns.index_from_values(param_list[i], name=level_names[i])
            param_indexes.append(param_index)
    if len(param_indexes) > 1:
        param_columns = index_fns.stack_indexes(*param_indexes)
    elif len(param_indexes) == 1:
        param_columns = param_indexes[0]
    else:
        param_columns = None
    if param_columns is not None:
        return index_fns.combine_indexes(param_columns, ts_columns)
    return ts_columns


def build_mapper(params, ts, new_columns, level_name):
    """Build a mapper that maps parameter values in `params` to columns in `new_columns`."""
    params_mapper = np.repeat(params, len(ts.vbt.columns))
    params_mapper = pd.Series(params_mapper, index=new_columns, name=level_name)
    return params_mapper


def build_tuple_mapper(mappers_list, new_columns, level_names):
    """Build a tuple mapper that maps tuples of parameter values to columns in `new_columns`."""
    tuple_mapper = list(zip(*list(map(lambda x: x.values, mappers_list))))
    tuple_mapper = pd.Series(tuple_mapper, index=new_columns, name=level_names)
    return tuple_mapper


def wrap_output(output, ts, new_columns):
    """Wrap a NumPy array into a pandas object with meta from `ts` and `new_columns`."""
    return ts.vbt.wrap(output, columns=new_columns)


def broadcast_ts(ts, params_len, new_columns):
    """Broadcast time series `ts` to match the length of `new_columns` through tiling."""
    if checks.is_series(ts) or len(new_columns) > ts.shape[1]:
        return ts.vbt.wrap(reshape_fns.tile(ts.values, params_len, axis=1), columns=new_columns)
    else:
        return ts.vbt.wrap(ts, columns=new_columns)


def from_params_pipeline(
        ts_list, param_list, level_names, num_outputs,
        custom_func, *args,
        pass_lists=False,
        pass_2d=True,
        param_product=False,
        broadcast_kwargs={},
        return_raw=False,
        **kwargs):
    """A pipeline for calculating an indicator, used by `IndicatorFactory`.

    Args:
        ts_list (list of array_like): A list of time series objects. At least one must be a pandas object.
        param_list (list of array_like): A list of parameters. Each element is either an array-like object
            or a single value of any type.
        level_names (list of str): A list of column level names corresponding to each parameter.
        num_outputs (int): The number of output arrays.
        custom_func (function): A custom calculation function. See `IndicatorFactory.from_custom_func`.
        *args: Arguments passed to the `custom_func`.
        pass_lists (bool): If `True`, arguments are passed to the `custom_func` as lists.
        pass_2d (bool): If `True`, time series arrays will be passed as two-dimensional, otherwise as is.
        param_product (bool): If `True`, builds a Cartesian product out of all parameters.
        broadcast_kwargs (dict): Keyword arguments passed to the `vectorbt.base.reshape_fns.broadcast`
            on time series objects.
        return_raw (bool): If `True`, returns the raw output without post-processing.
        **kwargs: Keyword arguments passed to the `custom_func`.

            Some common arguments include `return_cache` to return cache and `cache` to pass cache. 
            Those are only applicable to `custom_func` that supports it (`custom_func` created using
            `IndicatorFactory.from_apply_func` are supported by default).
            
    Returns:
        A list of transformed inputs (`pandas_like`), a list of generated outputs (`pandas_like`), 
        a list of parameter arrays (`np.ndarray`), a list of parameter mappers (`pd.Series`),
        a list of other generated outputs that are outside of  `num_outputs`.

    Explanation:

        Does the following:

        * Takes one or multiple time series objects in `ts_list` and broadcasts them. For example:

        ```python-repl
        >>> sr = pd.Series([1, 2], index=['x', 'y'])
        >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
        >>> ts_list = [sr, df]

        >>> ts_list = vbt.base.reshape_fns.broadcast(*ts_list)
        >>> print(ts_list[0])
        a  b
        x  1  1
        y  2  2
        >>> print(ts_list[1])
        a  b
        x  3  4
        y  5  6
        ```

        * Takes one or multiple parameters in `param_list`, converts them to NumPy arrays and 
            broadcasts them. For example:

        ```python-repl
        >>> p1, p2, p3 = 1, [2, 3, 4], [False]
        >>> param_list = [p1, p2, p3]

        >>> param_list = vbt.base.reshape_fns.broadcast(*param_list)
        >>> print(param_list[0])
        array([1, 1, 1])
        >>> print(param_list[1])
        array([2, 3, 4])
        >>> print(param_list[2])
        array([False, False, False])
        ```

        * Performs calculation using `custom_func` to build output arrays (`output_list`) and 
            other objects (`other_list`, optionally). For example:

        ```python-repl
        >>> def custom_func(ts1, ts2, p1, p2, p3, *args, **kwargs):
        ...     return np.hstack((
        ...         ts1 + ts2 + p1[0] * p2[0],
        ...         ts1 + ts2 + p1[1] * p2[1],
        ...         ts1 + ts2 + p1[2] * p2[2],
        ...     ))

        >>> output = custom_func(*ts_list, *param_list)
        >>> print(output)
        array([[ 6,  7,  7,  8,  8,  9],
               [ 9, 10, 10, 11, 11, 12]])
        ```

        * Creates new column hierarchy based on parameters and level names. For example:

        ```python-repl
        >>> p1_columns = pd.Index(param_list[0], name='p1')
        >>> p2_columns = pd.Index(param_list[1], name='p2')
        >>> p3_columns = pd.Index(param_list[2], name='p3')
        >>> p_columns = vbt.base.index_fns.stack_indexes(p1_columns, p2_columns, p3_columns)
        >>> new_columns = vbt.base.index_fns.combine_indexes(p_columns, ts_list[0].columns)

        >>> output_df = pd.DataFrame(output, columns=new_columns)
        >>> print(output_df)
        p1                                         1                        
        p2             2             3             4    
        p3  False  False  False  False  False  False    
                a      b      a      b      a      b
        0       6      7      7      8      8      9
        1       9     10     10     11     11     12
        ```

        * Broadcasts objects in `ts_list` to match the shape of objects in `output_list` through tiling.
            This is done to be able to compare them and generate signals, since you cannot compare NumPy 
            arrays that have totally different shapes, such as (2, 2) and (2, 6). For example:

        ```python-repl
        >>> new_ts_list = [
        ...     ts_list[0].vbt.tile(len(param_list[0]), as_columns=p_columns),
        ...     ts_list[1].vbt.tile(len(param_list[0]), as_columns=p_columns)
        ... ]
        >>> print(new_ts_list[0])
        p1                                         1                        
        p2             2             3             4    
        p3  False  False  False  False  False  False     
                a      b      a      b      a      b
        0       1      1      1      1      1      1
        1       2      2      2      2      2      2
        ```

        * Builds parameter mappers that will link parameters from `param_list` to columns in 
            `ts_list` and `output_list`. This is done to enable column indexing using parameter values.
    """
    if len(ts_list) > 1:
        # Broadcast time series
        ts_list = reshape_fns.broadcast(*ts_list, **broadcast_kwargs, writeable=True)
    # Check time series objects
    checks.assert_type(ts_list[0], (pd.Series, pd.DataFrame))
    # Convert params to 1-dim arrays
    param_list = list(map(reshape_fns.to_1d, param_list))
    if len(param_list) > 1:
        # Check level names
        checks.assert_type(level_names, (list, tuple))
        checks.assert_same_len(param_list, level_names)
        for ts in ts_list:
            # Every time series object should be free of the specified level names in its columns
            for level_name in level_names:
                if level_name is not None:
                    if checks.is_frame(ts):
                        checks.assert_level_not_exists(ts.columns, level_name)
        if param_product:
            # Make Cartesian product out of all params
            param_list = build_param_product(param_list)
        else:
            # Broadcast such that each array has the same length
            param_list = reshape_fns.broadcast(*param_list, writeable=True)
    # Perform main calculation
    if pass_2d:
        array_list = tuple(map(lambda x: reshape_fns.to_2d(np.asarray(x)), ts_list))
    else:
        array_list = tuple(map(lambda x: np.asarray(x), ts_list))
    if pass_lists:
        output_list = custom_func(array_list, param_list, *args, **kwargs)
    else:
        output_list = custom_func(*array_list, *param_list, *args, **kwargs)
    if return_raw or kwargs.get('return_cache', False):
        return output_list  # return raw cache outputs
    if not isinstance(output_list, (tuple, list, List)):
        output_list = [output_list]
    else:
        output_list = list(output_list)
    # Other outputs should be returned without post-processing (for example cache_dict)
    if len(output_list) > num_outputs:
        other_list = output_list[num_outputs:]
    else:
        other_list = []
    # Process only the num_outputs outputs
    output_list = output_list[:num_outputs]
    if len(param_list) > 0:
        # Build new column levels on top of time series levels
        new_columns = build_column_hierarchy(param_list, level_names, ts_list[0].vbt.columns)
        # Wrap into new pandas objects both time series and output objects
        new_ts_list = list(map(lambda x: broadcast_ts(x, param_list[0].shape[0], new_columns), ts_list))
        # Build mappers to easily map between parameters and columns
        mapper_list = [build_mapper(x, ts_list[0], new_columns, level_names[i]) for i, x in enumerate(param_list)]
    else:
        # Some indicators don't have any params
        new_columns = ts_list[0].vbt.columns
        new_ts_list = list(ts_list)
        mapper_list = []
    output_list = list(map(lambda x: wrap_output(x, ts_list[0], new_columns), output_list))
    if len(mapper_list) > 1:
        # Tuple object is a mapper that accepts tuples of parameters
        tuple_mapper = build_tuple_mapper(mapper_list, new_columns, tuple(level_names))
        mapper_list.append(tuple_mapper)
    return new_ts_list, output_list, param_list, mapper_list, other_list


def perform_init_checks(ts_list, output_list, param_list, mapper_list, name):
    """Perform checks on objects created by running or slicing an indicator."""
    checks.assert_type(ts_list[0], (pd.Series, pd.DataFrame))
    for ts in ts_list + output_list:
        checks.assert_same_meta(ts_list[0], ts)
    for params in param_list:
        checks.assert_same_shape(param_list[0], params)
    for mapper in mapper_list:
        checks.assert_type(mapper, pd.Series)
        checks.assert_same_index(reshape_fns.to_2d(ts_list[0]).iloc[0, :], mapper)
    checks.assert_type(name, str)


def compare(obj, other, compare_func, multiple=False, name=None, as_columns=None, **kwargs):
    """Compares `obj` to `other` to generate signals.

    Both will be broadcast together. Set `multiple` to `True` to compare with multiple arguments.
    In this case, a new column level will be created with the name `name`.

    See `vectorbt.base.accessors.Base_Accessor.combine_with`."""
    if multiple:
        if as_columns is None:
            as_columns = index_fns.index_from_values(other, name=name)
        return obj.vbt.combine_with_multiple(other, combine_func=compare_func, as_columns=as_columns, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=compare_func, **kwargs)


class IndicatorFactory():
    def __init__(self,
                 class_name='CustomIndicator',
                 module_name=__name__,
                 ts_names=['ts'],
                 param_names=['param'],
                 output_names=['output'],
                 name='custom',
                 custom_outputs={}):
        """A factory for creating new indicators.

        Args:
            ts_names (list of str): A list of names of input time series objects.
            param_names (list of str): A list of names of parameters.
            output_names (list of str): A list of names of outputs time series objects.
            name (str): A short name of the indicator.
            custom_outputs (dict): A dictionary with user-defined functions that will be
                bound to the indicator class and wrapped with `@cached_property`.

        !!! note
            The `__init__` method is never used for running the indicator, for this use `from_params`.
            The reason for this is indexing, which requires a clean `__init__` method for creating 
            a new indicator object with newly indexed attributes.
        """
        self.class_name = class_name
        self.module_name = module_name
        self.ts_names = ts_names
        self.param_names = param_names
        self.output_names = output_names
        self.name = name
        self.custom_outputs = custom_outputs

    def from_custom_func(self, custom_func, **pipeline_kwargs):
        """Build indicator class around a custom calculation function.

        !!! note
            Time series passed to `apply_func` will be 2-dimensional NumPy arrays.

            In contrast to `IndicatorFactory.from_apply_func`, it's up to you to handle caching
            and concatenate columns for each parameter (for example, by using 
            `vectorbt.base.combine_fns.apply_and_concat_one`). Also, you must ensure that each output
            array has an appropriate number of columns, which is the number of columns in input time 
            series multiplied by the number of parameter values.

        Args:
            custom_func (function): A function that takes broadcasted time series corresponding 
                to `ts_names`, broadcasted parameter arrays corresponding to `param_names`, and other 
                arguments and keyword arguments, and returns outputs corresponding to `output_names`
                and other objects that are then returned with the indicator class instance.
                Can be Numba-compiled.
            **pipeline_kwargs: Default keyword arguments passed to `from_params_pipeline`.
        Returns:
            `CustomIndicator`, and optionally other objects that are returned by `custom_func`
            and exceed `output_names`.
        Example:
            The following example does the same as the example in `IndicatorFactory.from_apply_func`.

            ```python-repl
            >>> @njit
            >>> def apply_func_nb(i, ts1, ts2, p1, p2, arg1):
            ...     return ts1 * p1[i] + arg1, ts2 * p2[i] + arg1

            >>> @njit
            ... def custom_func(ts1, ts2, p1, p2, *args):
            ...     return vbt.base.combine_fns.apply_and_concat_multiple_nb(
            ...         len(p1), apply_func_nb, ts1, ts2, p1, p2, *args)

            >>> MyInd = vbt.IndicatorFactory(
            ...     ts_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['o1', 'o2']
            ... ).from_custom_func(custom_func)

            >>> myInd = MyInd.from_params(price_sm, price_sm * 2, [1, 2], [3, 4], 100)
            >>> print(myInd.o1)
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> print(myInd.o2)
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  106.0  130.0  108.0  140.0
            2020-01-02  112.0  124.0  116.0  132.0
            2020-01-03  118.0  118.0  124.0  124.0
            2020-01-04  124.0  112.0  132.0  116.0
            2020-01-05  130.0  106.0  140.0  108.0
            ```
        """
        ts_names = self.ts_names
        param_names = self.param_names
        output_names = self.output_names
        name = self.name
        custom_outputs = self.custom_outputs

        # Add indexing methods
        def indexing_func(obj, pd_indexing_func):
            ts_list = []
            for ts_name in ts_names:
                ts_list.append(pd_indexing_func(getattr(obj, ts_name)))
            output_list = []
            for output_name in output_names:
                output_list.append(pd_indexing_func(getattr(obj, output_name)))
            param_list = []
            for param_name in param_names:
                # TODO: adapt params array according to the indexing operation
                param_list.append(getattr(obj, f'_{param_name}_array'))
            mapper_list = []
            for param_name in param_names:
                mapper_list.append(indexing_on_mapper(
                    getattr(obj, f'_{param_name}_mapper'),
                    getattr(obj, ts_names[0]), pd_indexing_func))
            if len(param_names) > 1:
                mapper_list.append(indexing_on_mapper(
                    obj._tuple_mapper, getattr(obj, ts_names[0]), pd_indexing_func))

            return obj.__class__(ts_list, output_list, param_list, mapper_list, obj.name)

        ParamIndexer = ParamIndexerFactory(param_names + (['tuple'] if len(param_names) > 1 else []))
        CustomIndicator = type(self.class_name, (PandasIndexer, ParamIndexer), {})
        CustomIndicator.__module__ = self.module_name

        # For name and each input and output, create read-only properties
        prop = property(lambda self: self._name)
        prop.__doc__ = f"""Name of the indicator (read-only)."""
        setattr(CustomIndicator, 'name', prop)

        for ts_name in ts_names:
            prop = property(lambda self, ts_name=ts_name: getattr(self, '_' + ts_name))
            prop.__doc__ = f"""Input time series (read-only)."""
            setattr(CustomIndicator, ts_name, prop)

        for output_name in output_names:
            prop = property(lambda self, output_name=output_name: getattr(self, '_' + output_name))
            prop.__doc__ = f"""Output time series (read-only)."""
            setattr(CustomIndicator, output_name, prop)

        for prop in custom_outputs.values():
            if prop.__doc__ is None:
                prop.__doc__ = f"""Custom property."""

        # Add __init__ method
        def __init__(self, ts_list, output_list, param_list, mapper_list, name):
            perform_init_checks(ts_list, output_list, param_list, mapper_list, name)

            for i, ts_name in enumerate(ts_names):
                setattr(self, f'_{ts_name}', ts_list[i])
            self.wrapper = TSArrayWrapper.from_obj(ts_list[0])
            for i, output_name in enumerate(output_names):
                setattr(self, f'_{output_name}', output_list[i])
            for i, param_name in enumerate(param_names):
                setattr(self, f'_{param_name}_array', param_list[i])
                setattr(self, f'_{param_name}_mapper', mapper_list[i])
            if len(param_names) > 1:
                setattr(self, '_tuple_mapper', mapper_list[-1])
            setattr(self, '_name', name)

            # Initialize indexers
            PandasIndexer.__init__(self, indexing_func)
            ParamIndexer.__init__(self, mapper_list, indexing_func)

        setattr(CustomIndicator, '__init__', __init__)

        # Add from_params method
        @classmethod
        def from_params(cls, *args, name=name.lower(), hide_params=[],
                        pipeline_kwargs=pipeline_kwargs, **kwargs):
            level_names = []
            for param_name in param_names:
                if param_name in hide_params:
                    level_names.append(None)
                else:
                    level_names.append(name + '_' + param_name)

            level_names = list(level_names)
            args = list(args)
            ts_list = args[:len(ts_names)]
            param_list = args[len(ts_names):len(ts_names)+len(param_names)]
            custom_func_args = args[len(ts_names)+len(param_names):]
            kwargs = {**pipeline_kwargs, **kwargs}  # overwrite default pipeline kwargs
            results = from_params_pipeline(
                ts_list, param_list, level_names, len(output_names),
                custom_func, *custom_func_args, **kwargs)
            if kwargs.get('return_raw', False) or kwargs.get('return_cache', False):
                return results
            new_ts_list, output_list, new_param_list, mapper_list, other_list = results
            obj = cls(new_ts_list, output_list, new_param_list, mapper_list, name)
            if len(other_list) > 0:
                return (obj, *tuple(other_list))
            return obj

        setattr(CustomIndicator, 'from_params', from_params)

        # Add user-defined outputs
        for prop_name, prop in custom_outputs.items():
            if not isinstance(prop, (property, cached_property)):
                prop.__name__ = prop_name
                prop = cached_property(prop)
            setattr(CustomIndicator, prop_name, prop)

        # Add comparison methods for all inputs, outputs, and user-defined properties
        comparison_attrs = set(ts_names + output_names + list(custom_outputs.keys()))
        for attr in comparison_attrs:
            def assign_comparison_method(func_name, compare_func, attr=attr):
                def comparison_method(self, other, crossed=False, wait=0, name=None, after_false=True, **kwargs):
                    if isinstance(other, self.__class__):
                        other = getattr(other, attr)
                    if name is None:
                        if attr == self.name:
                            name = f'{self.name}_{func_name}'
                        else:
                            name = f'{self.name}_{attr}_{func_name}'
                    result = compare(getattr(self, attr), other, compare_func, name=name, **kwargs)
                    if crossed:
                        return result.vbt.signals.nst(wait+1, after_false=after_false)
                    return result
                comparison_method.__qualname__ = f'{CustomIndicator.__name__}.{attr}_{func_name}'
                comparison_method.__doc__ = f"""Return `True` for each element where `{attr}` is {func_name} `other`. 

                Set `crossed` to `True` to return the first `True` after crossover. Specify `wait` to return 
                `True` only when `{attr}` is {func_name} for a number of time steps in a row after crossover.

                See `vectorbt.indicators.factory.compare`."""
                setattr(CustomIndicator, f'{attr}_{func_name}', comparison_method)

            assign_comparison_method('above', np.greater)
            assign_comparison_method('below', np.less)
            assign_comparison_method('equal', np.equal)

        return CustomIndicator

    def from_apply_func(self, apply_func, caching_func=None, **kwargs):
        """Build indicator class around a custom apply function.

        In contrast to `IndicatorFactory.from_custom_func`, this method handles a lot of things for you,
        such as caching, parameter selection, and concatenation. All you have to do is to write `apply_func`
        that accepts a selection of parameters (single values as opposed to multiple values in 
        `IndicatorFactory.from_custom_func`) and does the calculation. It then automatically concatenates
        the results into a single array per output.

        While this approach is much more simpler, it is also less flexible, since you can only work with 
        one parameter selection at a time, and can't view all parameters.

        !!! note
            Time series passed to `apply_func` will be 2-dimensional NumPy arrays.

            If `apply_func` is a Numba-compiled function: 

            * All inputs are automatically converted to NumPy arrays
            * Each argument in `*args` must be of a Numba-compatible type
            * You cannot pass keyword arguments
            * Your outputs must be arrays of the same shape, data type and data order

        Args:
            apply_func (function): A function that takes broadcasted time series arrays corresponding 
                to `ts_names`, single parameter selection corresponding to `param_names`, and other 
                arguments and keyword arguments, and returns outputs corresponding to `output_names`.
                Can be Numba-compiled.
            caching_func (function): A caching function to preprocess data beforehand.
                All returned objects will be passed as additional arguments to `apply_func`.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_custom_func`.
        Returns:
            CustomIndicator
        Example:
            ```python-repl
            >>> @njit
            ... def apply_func_nb(ts1, ts2, p1, p2, arg1):
            ...     return ts1 * p1 + arg1, ts2 * p2 + arg1

            >>> MyInd = vbt.IndicatorFactory(
            ...     ts_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['o1', 'o2']
            ... ).from_apply_func(apply_func_nb)

            >>> myInd = MyInd.from_params(price_sm, price_sm * 2, [1, 2], [3, 4], 100)
            >>> print(myInd.o1)
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> print(myInd.o2)
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  106.0  130.0  108.0  140.0
            2020-01-02  112.0  124.0  116.0  132.0
            2020-01-03  118.0  118.0  124.0  124.0
            2020-01-04  124.0  112.0  132.0  116.0
            2020-01-05  130.0  106.0  140.0  108.0
            ```
        """
        output_names = self.output_names

        num_outputs = len(output_names)

        if checks.is_numba_func(apply_func):
            if num_outputs > 1:
                apply_and_concat_func = combine_fns.apply_and_concat_multiple_nb
            else:
                apply_and_concat_func = combine_fns.apply_and_concat_one_nb

            @njit
            def select_params_func_nb(i, apply_func, ts_arr_list, param_tuples, *args):
                # Select the next tuple of parameters
                return apply_func(*ts_arr_list, *param_tuples[i], *args)

            def custom_func(ts_arr_list, param_list, *args, return_cache=False, cache=None):
                # avoid deprecation warnings
                ts_arr_list = tuple(map(np.asarray, ts_arr_list))
                typed_param_tuples = List()
                for param_tuple in list(zip(*param_list)):
                    typed_param_tuples.append(param_tuple)

                # Caching
                if cache is None and caching_func is not None:
                    cache = caching_func(*ts_arr_list, *param_list, *args)
                if return_cache:
                    return cache
                if cache is None:
                    cache = ()
                if not isinstance(cache, (tuple, list, List)):
                    cache = (cache,)

                return apply_and_concat_func(
                    param_list[0].shape[0],
                    select_params_func_nb,
                    apply_func,
                    ts_arr_list,
                    typed_param_tuples,
                    *args,
                    *cache)
        else:
            if num_outputs > 1:
                apply_and_concat_func = combine_fns.apply_and_concat_multiple
            else:
                apply_and_concat_func = combine_fns.apply_and_concat_one

            def select_params_func(i, apply_func, ts_arr_list, param_list, *args, **kwargs):
                # Select the next tuple of parameters
                param_is = list(map(lambda x: x[i], param_list))
                return apply_func(*ts_arr_list, *param_is, *args, **kwargs)

            def custom_func(ts_arr_list, param_list, *args, return_cache=False, cache=None, **kwargs):
                ts_arr_list = tuple(map(np.asarray, ts_arr_list))
                # Caching
                if cache is None and caching_func is not None:
                    cache = caching_func(*ts_arr_list, *param_list, *args, **kwargs)
                if return_cache:
                    return cache
                if cache is None:
                    cache = ()
                if not isinstance(cache, (tuple, list, List)):
                    cache = (cache,)

                return apply_and_concat_func(
                    param_list[0].shape[0],
                    select_params_func,
                    apply_func,
                    ts_arr_list,
                    param_list,
                    *args,
                    *cache,
                    **kwargs)

        return self.from_custom_func(custom_func, pass_lists=True, **kwargs)
