"""
Technical indicators are used to see past trends and anticipate future moves. This module provides a collection
of such indicators, but also a comprehensive `vectorbt.indicators.IndicatorFactory` for building new indicators
with ease.

Before running the examples, import the following libraries:
```py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
from numba import njit
import yfinance as yf

import vectorbt as vbt

ticker = yf.Ticker("BTC-USD")
price = ticker.history(start=datetime(2019, 1, 1), end=datetime(2020, 1, 1))['Close']

price.vbt.timeseries.plot()
```
![](img/Indicators_price.png)
"""
import pandas as pd
import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType, ListType, Array
from numba.typed import List, Dict
from copy import copy
import plotly.graph_objects as go
import itertools
import types

from vectorbt.utils import *
from vectorbt.accessors import *
from vectorbt.timeseries import rolling_mean_nb, rolling_std_nb, ewm_mean_nb, \
    ewm_std_nb, diff_nb, set_by_mask_nb, prepend_nb, rolling_min_nb, rolling_max_nb

# ############# Indicator factory ############# #


def build_column_hierarchy(param_list, level_names, ts_columns):
    check_same_shape(param_list, level_names, along_axis=0)
    param_indices = [index_from_values(param_list[i], name=level_names[i]) for i in range(len(param_list))]
    param_columns = None
    for param_index in param_indices:
        if param_columns is None:
            param_columns = param_index
        else:
            param_columns = stack_indices(param_columns, param_index)
    if param_columns is not None:
        return combine_indices(param_columns, ts_columns)
    return ts_columns


def build_mapper(params, ts, new_columns, level_name):
    params_mapper = np.repeat(params, len(to_2d(ts).columns))
    params_mapper = pd.Series(params_mapper, index=new_columns, name=level_name)
    return params_mapper


def build_tuple_mapper(mappers_list, new_columns, level_names):
    tuple_mapper = list(zip(*list(map(lambda x: x.values, mappers_list))))
    tuple_mapper = pd.Series(tuple_mapper, index=new_columns, name=level_names)
    return tuple_mapper


def wrap_output(output, ts, new_columns):
    return ts.vbt.wrap_array(output, columns=new_columns)


def broadcast_ts(ts, params_len, new_columns):
    if is_series(ts) or len(new_columns) > ts.shape[1]:
        return ts.vbt.wrap_array(tile(ts.values, params_len, along_axis=1), columns=new_columns)
    else:
        return ts.vbt.wrap_array(ts, columns=new_columns)


def from_params_pipeline(ts_list, param_list, level_names, num_outputs, custom_func, *args, pass_lists=False,
                         param_product=False, broadcast_kwargs={}, return_raw=False, **kwargs):
    """A pipeline for calculating an indicator, used by `vectorbt.indicators.IndicatorFactory`.

    Does the following:

    * Takes one or multiple time series objects in `ts_list` and broadcasts them

    ```python-repl
    >>> sr = pd.Series([1, 2], index=['x', 'y'])
    >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
    >>> ts_list = [sr, df]

    >>> ts_list = vbt.utils.broadcast(*ts_list)
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
        broadcasts them

    ```python-repl
    >>> p1, p2, p3 = 1, [2, 3, 4], [False]
    >>> param_list = [p1, p2, p3]

    >>> param_list = vbt.utils.broadcast(*param_list)
    >>> print(param_list[0])
    array([1, 1, 1])
    >>> print(param_list[1])
    array([2, 3, 4])
    >>> print(param_list[2])
    array([False, False, False])
    ```

    * Performs calculation using `custom_func` to build output arrays (`output_list`) and 
        other objects (`other_list`, optional)

    ```python-repl
    >>> def custom_func(ts1, ts2, p1, p2, p3, *args, **kwargs):
    ...     return pd.DataFrame.vbt.concat(
    ...         (ts1.values + ts2.values) + p1[0] * p2[0],
    ...         (ts1.values + ts2.values) + p1[1] * p2[1],
    ...         (ts1.values + ts2.values) + p1[2] * p2[2]
    ...     )

    >>> output = custom_func(*ts_list, *param_list)
    >>> print(output)
    array([[ 6,  7,  7,  8,  8,  9],
           [ 9, 10, 10, 11, 11, 12]])
    ```

    * Creates new column hierarchy based on parameters and level names

    ```python-repl
    >>> p1_columns = pd.Index(param_list[0], name='p1')
    >>> p2_columns = pd.Index(param_list[1], name='p2')
    >>> p3_columns = pd.Index(param_list[2], name='p3')
    >>> p_columns = vbt.utils.stack_indices(p1_columns, p2_columns, p3_columns)
    >>> new_columns = vbt.utils.combine_indices(p_columns, ts_list[0].columns)

    >>> output_df = pd.DataFrame(output, columns=new_columns)
    >>> print(output_df)
    p1      1      1      1      1      1      1                        
    p2      2      2      3      3      4      4    
    p3  False  False  False  False  False  False    
            a      b      a      b      a      b
    0       6      7      7      8      8      9
    1       9     10     10     11     11     12
    ```

    * Broadcasts objects in `ts_list` to match the shape of objects in `output_list` through tiling.
        This is done to be able to compare them and generate signals, since you cannot compare NumPy 
        arrays that have totally different shapes, such as (2, 2) and (2, 6).

    ```python-repl
    >>> new_ts_list = [
    ...     ts_list[0].vbt.tile(len(param_list[0]), as_columns=p_columns),
    ...     ts_list[1].vbt.tile(len(param_list[0]), as_columns=p_columns)
    ... ]
    >>> print(new_ts_list[0])
    p1      1      1      1      1      1      1                        
    p2      2      2      3      3      4      4    
    p3  False  False  False  False  False  False     
            a      b      a      b      a      b
    0       1      1      1      1      1      1
    1       2      2      2      2      2      2
    ```

    * Builds parameter mappers that will link parameters from `param_list` to columns in 
        `ts_list` and `output_list`. This is done to enable column indexing using parameter values.

    Args:
        ts_list (list of array_like): A list of time series objects. At least one must be a pandas object.
        param_list (list of array_like): A list of parameters. Each element is either an array-like object
            or a single value of any type.
        level_names (list of str): A list of column level names corresponding to each parameter.
        num_outputs (int): The number of output arrays.
        custom_func (function): A custom calculation function. See `IndicatorFactory.from_custom_func`.
        *args: Arguments passed to the `custom_func`.
        pass_lists (bool): If True, arguments are passed to the `custom_func` as lists. Defaults to False.
        param_product (bool): If True, builds a Cartesian product out of all parameters. Defaults to False.
        broadcast_kwargs (dict, optional): Keyword arguments passed to the `vectorbt.utils.broadcast` on time series objects.
        return_raw (bool): If True, returns the raw output without post-processing. Defaults to False.
        **kwargs: Keyword arguments passed to the `custom_func`.

            Some common arguments include `return_cache` to return cache and `cache` to pass cache. 
            Those are only applicable to `custom_func` that supports it (`custom_func` created using
            `IndicatorFactory.from_apply_func` are supported by default).
    Returns:
        A list of transformed inputs (`pandas_like`), a list of generated outputs (`pandas_like`), 
        a list of parameter arrays (`numpy.ndarray`), a list of parameter mappers (`pandas.Series`),
        a list of other generated outputs that are outside of  `num_outputs`.
    """
    # Check time series objects
    check_type(ts_list[0], (pd.Series, pd.DataFrame))
    for i in range(1, len(ts_list)):
        ts_list[i].vbt.timeseries.validate()
    if len(ts_list) > 1:
        # Broadcast time series
        ts_list = broadcast(*ts_list, **broadcast_kwargs, writeable=True)
    # Check level names
    check_type(level_names, (list, tuple))
    check_same_len(param_list, level_names)
    for ts in ts_list:
        # Every time series object should be free of the specified level names in its columns
        for level_name in level_names:
            check_level_not_exists(ts, level_name)
    # Convert params to 1-dim arrays
    param_list = list(map(to_1d, param_list))
    if len(param_list) > 1:
        if param_product:
            # Make Cartesian product out of all params
            param_list = list(map(to_1d, param_list))
            param_list = list(zip(*list(itertools.product(*param_list))))
            param_list = list(map(np.asarray, param_list))
        else:
            # Broadcast such that each array has the same length
            param_list = broadcast(*param_list, writeable=True)
    # Perform main calculation
    if pass_lists:
        output_list = custom_func(ts_list, param_list, *args, **kwargs)
    else:
        output_list = custom_func(*ts_list, *param_list, *args, **kwargs)
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
        new_columns = build_column_hierarchy(param_list, level_names, to_2d(ts_list[0]).columns)
        # Wrap into new pandas objects both time series and output objects
        new_ts_list = list(map(lambda x: broadcast_ts(x, param_list[0].shape[0], new_columns), ts_list))
        # Build mappers to easily map between parameters and columns
        mapper_list = [build_mapper(x, ts_list[0], new_columns, level_names[i]) for i, x in enumerate(param_list)]
    else:
        # Some indicators don't have any params
        new_columns = to_2d(ts_list[0]).columns
        new_ts_list = list(ts_list)
        mapper_list = []
    output_list = list(map(lambda x: wrap_output(x, ts_list[0], new_columns), output_list))
    if len(mapper_list) > 1:
        # Tuple object is a mapper that accepts tuples of parameters
        tuple_mapper = build_tuple_mapper(mapper_list, new_columns, tuple(level_names))
        mapper_list.append(tuple_mapper)
    return new_ts_list, output_list, param_list, mapper_list, other_list


def perform_init_checks(ts_list, output_list, param_list, mapper_list, name):
    for ts in ts_list:
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
    for i in range(1, len(ts_list) + len(output_list)):
        check_same_meta((ts_list + output_list)[i-1], (ts_list + output_list)[i])
    for i in range(1, len(param_list)):
        check_same_shape(param_list[i-1], param_list[i])
    for mapper in mapper_list:
        check_type(mapper, pd.Series)
        check_same_index(to_2d(ts_list[0]).iloc[0, :], mapper)
    check_type(name, str)


def is_equal(obj, other, multiple=False, name='is_equal', as_columns=None, **kwargs):
    if multiple:
        if as_columns is None:
            as_columns = index_from_values(other, name=name)
        return obj.vbt.combine_with_multiple(other, combine_func=np.equal, as_columns=as_columns, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=np.equal, **kwargs)


def is_above(obj, other, multiple=False, name='is_above', as_columns=None, **kwargs):
    if multiple:
        if as_columns is None:
            as_columns = index_from_values(other, name=name)
        return obj.vbt.combine_with_multiple(other, combine_func=np.greater, as_columns=as_columns, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=np.greater, **kwargs)


def is_below(obj, other, multiple=False, name='is_below', as_columns=None, **kwargs):
    if multiple:
        if as_columns is None:
            as_columns = index_from_values(other, name=name)
        return obj.vbt.combine_with_multiple(other, combine_func=np.less, as_columns=as_columns, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=np.less, **kwargs)


class IndicatorFactory():
    def __init__(self,
                 ts_names=['ts'],
                 param_names=['param'],
                 output_names=['output'],
                 name='custom',
                 custom_properties={}):
        """A factory for creating new indicators.

        Args:
            ts_names (list of str): A list of names of input time-series objects. 
                Defaults to ['ts'].
            param_names (list of str): A list of names of parameters. 
                Defaults to ['param'].
            output_names (list of str): A list of names of outputs time-series objects. 
                Defaults to ['output'].
            name (str): A short name of the indicator. 
                Defaults to 'custom'.
            custom_properties (dict, optional): A dictionary with user-defined functions that will be
                bound to the indicator class and wrapped with `@cached_property`.

        Each indicator is basically a pipeline that

        * Accepts a set of time-series objects (for example, OHLCV data)
        * Accepts a set of parameter arrays (for example, rolling windows)
        * Accepts other relevant arguments and keyword arguments
        * Performs calculations to produce new time-series objects (for example, rolling average)

        This pipeline can be well standardized, which is done by this indicatory factory.

        On top of this pipeline, it also does the following:

        * Creates a new indicator class
        * Creates an `__init__` method where it stores all inputs, outputs, and other artifacts

        !!! note
            The `__init__` method is never used for running the indicator, for this use `from_params`.
            The reason for this is indexing, which requires a clean `__init__` method for creating 
            a new indicator object with newly indexed attributes.

        * Creates a `from_params` method that runs the main pipeline using `vectorbt.indicators.from_params_pipeline`
        * Adds pandas indexing, i.e., you can use `iloc`, `loc`, `xs`, and `__getitem__` on the class itself
        * Adds parameter indexing, i.e., use `*your_param*_loc` on the class to slice using parameters
        * Adds user-defined properties
        * Adds common comparison methods for all inputs, outputs and properties, e.g., crossovers

        Consider the following smaller price dataframe `price_sm`:

        ```python-repl
        >>> index = pd.Index([
        ...     datetime(2018, 1, 1),
        ...     datetime(2018, 1, 2),
        ...     datetime(2018, 1, 3),
        ...     datetime(2018, 1, 4),
        ...     datetime(2018, 1, 5),
        ... ])
        >>> price_sm = pd.DataFrame({
        ...     'a': [1, 2, 3, 4, 5], 
        ...     'b': [5, 4, 3, 2, 1]}, index=index).astype(float)
        >>> print(price_sm)
                      a    b
        2018-01-01  1.0  5.0
        2018-01-02  2.0  4.0
        2018-01-03  3.0  3.0
        2018-01-04  4.0  2.0
        2018-01-05  5.0  1.0
        ```

        For each column in the dataframe, let's calculate a simple moving average and get signals 
        of price crossing it. In particular, we want to test two different window sizes: 2 and 3.

        A naive way of doing this:

        ```python-repl
        >>> ma_df = pd.DataFrame.vbt.concat(
        ...     price_sm.rolling(window=2).mean(), 
        ...     price_sm.rolling(window=3).mean(), 
        ...     as_columns=pd.Index([2, 3], name='ma_window'))
        >>> print(ma_df)
        ma_window     2    2    3    3
                      a    b    a    b
        2018-01-01  NaN  NaN  NaN  NaN
        2018-01-02  1.5  4.5  NaN  NaN
        2018-01-03  2.5  3.5  2.0  4.0
        2018-01-04  3.5  2.5  3.0  3.0
        2018-01-05  4.5  1.5  4.0  2.0

        >>> above_signals = (price_sm.vbt.tile(2).vbt > ma_df)
        >>> above_signals = above_signals.vbt.signals.first(after_false=True)
        >>> print(above_signals)
        ma_window       2      2      3      3
                        a      b      a      b
        2018-01-01  False  False  False  False
        2018-01-02   True  False  False  False
        2018-01-03  False  False   True  False
        2018-01-04  False  False  False  False
        2018-01-05  False  False  False  False

        >>> below_signals = (price_sm.vbt.tile(2).vbt < ma_df)
        >>> below_signals = below_signals.vbt.signals.first(after_false=True)
        >>> print(below_signals)
        ma_window       2      2      3      3
                        a      b      a      b
        2018-01-01  False  False  False  False
        2018-01-02  False   True  False  False
        2018-01-03  False  False  False   True
        2018-01-04  False  False  False  False
        2018-01-05  False  False  False  False
        ```

        Now the same using `vectorbt.indicators.IndicatorFactory`:

        ```python-repl
        >>> MyMA = vbt.IndicatorFactory(
        ...     ts_names=['price_sm'],
        ...     param_names=['window'],
        ...     output_names=['ma'],
        ...     name='myma'
        ... ).from_apply_func(vbt.timeseries.rolling_mean_nb)

        >>> myma = MyMA.from_params(price_sm, [2, 3])
        >>> above_signals = myma.price_sm_above(myma.ma, crossover=True)
        >>> below_signals = myma.price_sm_below(myma.ma, crossover=True)
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

        Each of these methods uses vectorbt's own broadcasting, so you can compare time-series objects with an 
        arbitrary array-like object, given their shapes can be broadcasted together. You can also compare them
        to multiple objects at once, for example:

        ```python-repl
        >>> myma.ma_above([1.5, 2.5], multiple=True)
        myma_ma_above    1.5    1.5    1.5    1.5    2.5    2.5    2.5    2.5
        myma_window        2      2      3      3      2      2      3      3
                           a      b      a      b      a      b      a      b
        2018-01-01     False  False  False  False  False  False  False  False
        2018-01-02     False   True  False  False  False   True  False  False
        2018-01-03      True   True   True   True  False   True  False   True
        2018-01-04      True   True   True   True   True  False   True   True
        2018-01-05      True  False   True   True   True  False   True  False
        ```

        `vectorbt.indicators.IndicatorFactory` also attached pandas indexing to the indicator class: 

        ```python-repl
        'iloc'
        'loc'
        'window_loc'
        'xs'
        ```

        This makes accessing rows and columns by labels, integer positions, and parameters much easier.

        The other advantage of using `vectorbt.indicators.IndicatorFactory` is broadcasting:

        * Passing multiple time-series objects will broadcast them to the same shape and index/columns

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
        2018-01-01  1.0  5.0
        2018-01-02  2.0  4.0
        2018-01-03  3.0  3.0
        2018-01-04  4.0  2.0
        2018-01-05  5.0  1.0
        >>> print(myInd.price_sm2)
                      a    b
                     a2   b2
        2018-01-01  2.0  6.0
        2018-01-02  3.0  5.0
        2018-01-03  4.0  4.0
        2018-01-04  5.0  3.0
        2018-01-05  6.0  2.0
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
        self.ts_names = ts_names
        self.param_names = param_names
        self.output_names = output_names
        self.name = name
        self.custom_properties = custom_properties

    def from_custom_func(self, custom_func, pass_lists=False):
        """Build indicator class around a custom calculation function.

        !!! note
            `custom_func` shouldn't be Numba-compiled, since passed time series are all pandas objects.
            Instead, define your `custom_func` as a regular Python function where you should
            convert all inputs into NumPy arrays, and then pass them to your Numba-compiled function.

            Also, in contrast to `IndicatorFactory.from_apply_func`, it's up to you to handle caching
            and concatenate columns for each parameter (for example, by using `vectorbt.utils.apply_and_concat_one`).
            Also, you must ensure that each output array has an appropriate number of columns, which
            is the number of columns in input time series multiplied by the number of parameter values.

        Args:
            custom_func (function): A function that takes broadcasted time series corresponding 
                to `ts_names`, broadcasted parameter arrays corresponding to `param_names`, and other 
                arguments and keyword arguments, and returns outputs corresponding to `output_names` 
                and other objects that are then returned with the indicator class instance.
            pass_lists (bool): If True, passes arguments as lists, otherwise passes them using 
                starred expression. Defaults to False.
        Returns:
            `CustomIndicator`, and optionally other objects that are returned by `custom_func`
            and exceed `output_names`.
        Examples:
            The following example does the same as the example in `IndicatorFactory.from_apply_func`.

            ```python-repl
            >>> @njit
            >>> def apply_func_nb(i, ts1, ts2, p1, p2, arg1):
            ...     return ts1 * p1[i] + arg1, ts2 * p2[i] + arg1

            >>> def custom_func(ts1, ts2, p1, p2, *args):
            ...     return vbt.utils.apply_and_concat_multiple_nb(len(p1), apply_func_nb, 
            ...         ts1.vbt.to_2d_array(), ts2.vbt.to_2d_array(), p1, p2, *args)

            >>> MyInd = vbt.IndicatorFactory(
            ...     ts_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['o1', 'o2']
            ... ).from_custom_func(custom_func)

            >>> myInd = MyInd.from_params(price_sm, price_sm * 2, [1, 2], [3, 4], 100)
            >>> print(myInd.o1)
            custom_p1       1      1      2      2
            custom_p2       3      3      4      4
                            a      b      a      b
            2018-01-01  101.0  105.0  102.0  110.0
            2018-01-02  102.0  104.0  104.0  108.0
            2018-01-03  103.0  103.0  106.0  106.0
            2018-01-04  104.0  102.0  108.0  104.0
            2018-01-05  105.0  101.0  110.0  102.0
            >>> print(myInd.o2)
            custom_p1       1      1      2      2
            custom_p2       3      3      4      4
                            a      b      a      b
            2018-01-01  106.0  130.0  108.0  140.0
            2018-01-02  112.0  124.0  116.0  132.0
            2018-01-03  118.0  118.0  124.0  124.0
            2018-01-04  124.0  112.0  132.0  116.0
            2018-01-05  130.0  106.0  140.0  108.0
            ```
        """

        CustomIndicator = type('CustomIndicator', (), {})
        ts_names = self.ts_names
        param_names = self.param_names
        output_names = self.output_names
        name = self.name
        custom_properties = self.custom_properties

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

        for prop in custom_properties.values():
            if prop.__doc__ is None:
                prop.__doc__ = f"""Custom property."""

        # Add __init__ method
        def __init__(self, ts_list, output_list, param_list, mapper_list, name):
            """Performs checks on pipeline artifacts and stores them as instance attributes."""
            perform_init_checks(ts_list, output_list, param_list, mapper_list, name)

            for i, ts_name in enumerate(ts_names):
                setattr(self, f'_{ts_name}', ts_list[i])
            for i, output_name in enumerate(output_names):
                setattr(self, f'_{output_name}', output_list[i])
            for i, param_name in enumerate(param_names):
                setattr(self, f'_{param_name}_array', param_list[i])
                setattr(self, f'_{param_name}_mapper', mapper_list[i])
            if len(param_names) > 1:
                setattr(self, '_tuple_mapper', mapper_list[-1])
            setattr(self, '_name', name)

        setattr(CustomIndicator, '__init__', __init__)

        # Add from_params method
        @classmethod
        def from_params(cls, *args, name=name.lower(), return_raw=False, **kwargs):
            """Runs the pipeline and initializes the class."""
            level_names = tuple([name + '_' + param_name for param_name in param_names])
            args = list(args)
            ts_list = args[:len(ts_names)]
            param_list = args[len(ts_names):len(ts_names)+len(param_names)]
            new_args = args[len(ts_names)+len(param_names):]
            results = from_params_pipeline(
                ts_list, param_list, level_names, len(output_names),
                custom_func, *new_args, pass_lists=pass_lists, return_raw=return_raw, **kwargs)
            if return_raw or kwargs.get('return_cache', False):
                return results
            new_ts_list, output_list, new_param_list, mapper_list, other_list = results
            obj = cls(new_ts_list, output_list, new_param_list, mapper_list, name)
            if len(other_list) > 0:
                return (obj,) + other_list
            return obj

        setattr(CustomIndicator, 'from_params', from_params)

        # Add indexing methods
        def indexing_func(obj, loc_pandas_func):
            ts_list = []
            for ts_name in ts_names:
                ts_list.append(loc_pandas_func(getattr(obj, ts_name)))
            output_list = []
            for output_name in output_names:
                output_list.append(loc_pandas_func(getattr(obj, output_name)))
            param_list = []
            for param_name in param_names:
                # TODO: adapt params array according to the indexing operation
                param_list.append(getattr(obj, f'_{param_name}_array'))
            mapper_list = []
            for param_name in param_names:
                mapper_list.append(loc_mapper(
                    getattr(obj, f'_{param_name}_mapper'),
                    getattr(obj, ts_names[0]), loc_pandas_func))
            if len(param_names) > 1:
                mapper_list.append(loc_mapper(obj._tuple_mapper, getattr(obj, ts_names[0]), loc_pandas_func))

            return obj.__class__(ts_list, output_list, param_list, mapper_list, obj.name)

        CustomIndicator = add_indexing(indexing_func)(CustomIndicator)
        for i, param_name in enumerate(param_names):
            CustomIndicator = add_param_indexing(param_name, indexing_func)(CustomIndicator)
        if len(param_names) > 1:
            CustomIndicator = add_param_indexing('tuple', indexing_func)(CustomIndicator)

        # Add user-defined properties
        for prop_name, prop in custom_properties.items():
            if not isinstance(prop, property):
                prop = cached_property(prop)
            setattr(CustomIndicator, prop_name, prop)

        # Add comparison methods for all inputs, outputs, and user-defined properties
        comparison_attrs = set(ts_names + output_names + list(custom_properties.keys()))
        for attr in comparison_attrs:
            def assign_comparison_method(func_name, comparison_func, attr=attr):
                def comparison_method(self, other, crossover=False, wait=0, name=None, **kwargs):
                    if isinstance(other, self.__class__):
                        other = getattr(other, attr)
                    if name is None:
                        if attr == self.name:
                            name = f'{self.name}_{func_name}'
                        else:
                            name = f'{self.name}_{attr}_{func_name}'
                    result = comparison_func(getattr(self, attr), other, name=name, **kwargs)
                    if crossover:
                        return result.vbt.signals.nst(wait+1, after_false=True)
                    return result
                comparison_method.__qualname__ = f'{CustomIndicator.__name__}.{attr}_{func_name}'
                comparison_method.__doc__ = f"""Returns True when `{attr}` is {func_name} `other`. 

                Set `crossover` to True to return the first True after crossover. Specify `wait` to return 
                True only when `{attr}` is {func_name} for a number of time steps in a row after crossover.
                
                Both will be broadcasted together. Set `multiple` to True to combine with multiple arguments. 
                For more keyword arguments, see `vectorbt.utils.Base_Accessor.combine_with`."""
                setattr(CustomIndicator, f'{attr}_{func_name}', comparison_method)

            assign_comparison_method('above', is_above)
            assign_comparison_method('below', is_below)
            assign_comparison_method('equal', is_equal)

        return CustomIndicator

    def from_apply_func(self, apply_func, caching_func=None):
        """Build indicator class around a custom apply function.

        In contrast to `IndicatorFactory.from_custom_func`, this method handles a lot of things for you,
        such as caching, parameter selection, and concatenation. All you have to do is to write `apply_func`
        that accepts a selection of parameters (single values as opposed to multiple values in 
        `IndicatorFactory.from_custom_func`) and does the calculation. It then automatically concatenates
        the results into a single array per output.

        While this approach is much more simpler, it is also less flexible, since you can only work with 
        one parameter selection at a time, and can't view all parameters.

        !!! note
            If `apply_func` is a Numba-compiled function: 

            * All inputs are automatically converted to NumPy arrays
            * Each argument in `*args` must be of a Numba-compatible type
            * You cannot pass keyword arguments
            * Your outputs must be arrays of the same shape, data type and data order

        Args:
            apply_func (function): A function (can be Numba-compiled) that takes broadcasted time 
                series arrays corresponding to `ts_names`, single parameter selection corresponding 
                to `param_names`, and other arguments and keyword arguments, and returns outputs 
                corresponding to `output_names`.
            caching_func (function): A caching function to preprocess data beforehand.
                All returned objects will be passed as additional arguments to `apply_func`.
        Returns:
            `CustomIndicator`
        Examples:
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
            custom_p1       1      1      2      2
            custom_p2       3      3      4      4
                            a      b      a      b
            2018-01-01  101.0  105.0  102.0  110.0
            2018-01-02  102.0  104.0  104.0  108.0
            2018-01-03  103.0  103.0  106.0  106.0
            2018-01-04  104.0  102.0  108.0  104.0
            2018-01-05  105.0  101.0  110.0  102.0
            >>> print(myInd.o2)
            custom_p1       1      1      2      2
            custom_p2       3      3      4      4
                            a      b      a      b
            2018-01-01  106.0  130.0  108.0  140.0
            2018-01-02  112.0  124.0  116.0  132.0
            2018-01-03  118.0  118.0  124.0  124.0
            2018-01-04  124.0  112.0  132.0  116.0
            2018-01-05  130.0  106.0  140.0  108.0
            ```
        """
        output_names = self.output_names

        num_outputs = len(output_names)

        if is_numba_func(apply_func):
            apply_and_concat_func = apply_and_concat_multiple_nb if num_outputs > 1 else apply_and_concat_one_nb

            @njit
            def select_params_func_nb(i, apply_func, ts_list, param_tuples, *args):
                # Select the next tuple of parameters
                return apply_func(*ts_list, *param_tuples[i], *args)

            def custom_func(ts_list, param_list, *args, return_cache=False, cache=None):
                # avoid deprecation warnings
                typed_ts_list = tuple(map(lambda x: x.vbt.to_2d_array(), ts_list))
                typed_param_tuples = List()
                for param_tuple in list(zip(*param_list)):
                    typed_param_tuples.append(param_tuple)

                # Caching
                if cache is None and caching_func is not None:
                    cache = caching_func(*typed_ts_list, *param_list, *args)
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
                    typed_ts_list,
                    typed_param_tuples,
                    *args,
                    *cache)
        else:
            apply_and_concat_func = apply_and_concat_multiple if num_outputs > 1 else apply_and_concat_one

            def select_params_func(i, apply_func, ts_list, param_list, *args, **kwargs):
                    # Select the next tuple of parameters
                param_is = list(map(lambda x: x[i], param_list))
                return apply_func(*ts_list, *param_is, *args, **kwargs)

            def custom_func(ts_list, param_list, *args, return_cache=False, **kwargs):
                # Caching
                if cache is None and caching_func is not None:
                    cache = caching_func(*typed_ts_list, *param_list, *args, **kwargs)
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
                    ts_list,
                    param_list,
                    *args,
                    *cache,
                    **kwargs)

        return self.from_custom_func(custom_func, pass_lists=True)

# ############# MA ############# #


@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], b1[:]), cache=True)
def ma_caching_nb(ts, windows, ewms):
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                ma = ewm_mean_nb(ts, windows[i])
            else:
                ma = rolling_mean_nb(ts, windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = ma
    return cache_dict


@njit(f8[:, :](f8[:, :], i8, b1, DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def ma_apply_func_nb(ts, window, ewm, cache_dict):
    return cache_dict[(window, int(ewm))]


MA = IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['ma'],
    name='ma'
).from_apply_func(ma_apply_func_nb, caching_func=ma_caching_nb)


class MA(MA):
    """A moving average (MA) is a widely used indicator in technical analysis that helps smooth out 
    price action by filtering out the “noise” from random short-term price fluctuations. 

    See [Moving Average (MA)](https://www.investopedia.com/terms/m/movingaverage.asp).

    Use `MA.from_params` or `MA.from_combinations` methods to run the indicator."""

    @classmethod
    def from_params(cls, ts, window, ewm=False, **kwargs):
        """Calculate moving average `MA.ma` from time series `ts` and parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window. Can be one or more values.
            ewm (bool or array_like): If True, uses exponential moving average, otherwise 
                simple moving average. Can be one or more values. Defaults to False.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.from_params_pipeline.`
        Examples:
            ```python-repl
            >>> ma = vbt.MA.from_params(price, [10, 20], ewm=[False, True])

            >>> print(ma.ma)
            ma_window         10           20
            ma_ewm         False         True
            Date                             
            2018-12-31       NaN          NaN
            2019-01-01       NaN          NaN
            2019-01-02       NaN          NaN
            ...              ...          ...
            2019-12-29  7314.459  7313.283227
            2019-12-30  7321.877  7311.351491
            2019-12-31  7322.121  7300.137063

            [366 rows x 2 columns]
            ```
        """
        return super().from_params(ts, window, ewm, **kwargs)

    @classmethod
    def from_combinations(cls, ts, windows, r, ewm=False, names=None, **kwargs):
        """Create multiple `vectorbt.indicators.MA` combinations according to `itertools.combinations`.

        Args:
            ts (pandas_like): Time series (such as price).
            windows (array_like of int): Size of the moving window. Must be multiple.
            r (int): The number of `vectorbt.indicators.MA` instances to combine.
            ewm (bool or array_like of bool): If True, uses exponential moving average, otherwise 
                uses simple moving average. Can be one or more values. Defaults to False.
            names (list of str, optional): A list of names for each `vectorbt.indicators.MA` instance.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.from_params_pipeline.`
        Examples:
            ```python-repl
            >>> fast_ma, slow_ma = vbt.MA.from_combinations(price, 
            ...     [10, 20, 30], 2, ewm=[False, False, True], names=['fast', 'slow'])

            >>> print(fast_ma.ma)
            fast_window        10        10         20
            fast_ewm        False     False      False
            Date                                      
            2018-12-31        NaN       NaN        NaN
            2019-01-01        NaN       NaN        NaN
            2019-01-02        NaN       NaN        NaN
            ...               ...       ...        ...
            2019-12-29   7314.459  7314.459  7224.1280
            2019-12-30   7321.877  7321.877  7224.8720
            2019-12-31   7322.121  7322.121  7223.6805
            
            [366 rows x 3 columns]

            >>> print(slow_ma.ma)
            slow_window         20           30           30
            slow_ewm         False         True         True
            Date                                            
            2018-12-31         NaN          NaN          NaN
            2019-01-01         NaN          NaN          NaN
            2019-01-02         NaN          NaN          NaN
            ...                ...          ...          ...
            2019-12-29   7224.1280  7393.653412  7393.653412
            2019-12-30   7224.8720  7387.159643  7387.159643
            2019-12-31   7223.6805  7374.671925  7374.671925

            [366 rows x 3 columns]
            ```

            The naive way without caching is the follows:
            ```py
            window_combs = itertools.combinations([10, 20, 30], 2)
            ewm_combs = itertools.combinations([False, False, True], 2)
            fast_windows, slow_windows = np.asarray(list(window_combs)).transpose()
            fast_ewms, slow_ewms = np.asarray(list(ewm_combs)).transpose()

            fast_ma = vbt.MA.from_params(price, fast_windows, fast_ewms, name='fast')
            slow_ma = vbt.MA.from_params(price, slow_windows, slow_ewms, name='slow')
            ```

            Having this, you can now compare these `vectorbt.indicators.MA` instances:
            ```python-repl
            >>> entry_signals = fast_ma.ma_above(slow_ma, crossover=True)
            >>> exit_signals = fast_ma.ma_below(slow_ma, crossover=True)

            >>> print(entry_signals)
            fast_window     10     10     20
            fast_ewm     False  False  False
            slow_window     20     30     30
            slow_ewm     False   True   True
            Date                            
            2018-12-31   False  False  False
            2019-01-01   False  False  False
            2019-01-02   False  False  False
            ...            ...    ...    ...
            2019-12-29   False  False  False
            2019-12-30   False  False  False
            2019-12-31   False  False  False

            [366 rows x 3 columns]
            ```

            Notice how `MA.ma_above` method created a new column hierarchy for you. You can now use
            it for indexing as follows:

            ```py
            fig = price.vbt.timeseries.plot(name='Price')
            fig = entry_signals[(10, False, 20, False)]\
                .vbt.signals.plot_markers(price, signal_type='entry', fig=fig)
            fig = exit_signals[(10, False, 20, False)]\
                .vbt.signals.plot_markers(price, signal_type='exit', fig=fig)

            fig.show()
            ```
            ![](img/MA_from_combinations.png)
        """

        if names is None:
            names = ['ma' + str(i+1) for i in range(r)]
        windows, ewm = broadcast(windows, ewm, writeable=True)
        cache_dict = cls.from_params(ts, windows, ewm=ewm, return_cache=True, **kwargs)
        param_lists = zip(*itertools.combinations(zip(windows, ewm), r))
        mas = []
        for i, param_list in enumerate(param_lists):
            i_windows, i_ewm = zip(*param_list)
            mas.append(cls.from_params(ts, i_windows, ewm=i_ewm, cache=cache_dict, name=names[i], **kwargs))
        return tuple(mas)

    def plot(self,
             ts_name=None,
             ma_name=None,
             ts_trace_kwargs={},
             ma_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `MA.ma` against `MA.ts`.

        Args:
            ts_name (str): Name of the trace for `MA.ts`.
            ma_name (str): Name of the trace for `MA.ma`.
            ts_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `MA.ts`.
            ma_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `MA.ma`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Examples:
            ```py
            ma[(10, False)].plot()
            ```

            ![](img/MA.png)"""
        check_type(self.ts, pd.Series)
        check_type(self.ma, pd.Series)

        if ts_name is None:
            ts_name = f'Price ({self.name})'
        if ma_name is None:
            ma_name = f'MA ({self.name})'

        fig = self.ts.vbt.timeseries.plot(name=ts_name, trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.timeseries.plot(name=ma_name, trace_kwargs=ma_trace_kwargs, fig=fig)

        return fig


fix_class_for_pdoc(MA)

# ############# MSTD ############# #


@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], b1[:]), cache=True)
def mstd_caching_nb(ts, windows, ewms):
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                mstd = ewm_std_nb(ts, windows[i])
            else:
                mstd = rolling_std_nb(ts, windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = mstd
    return cache_dict


@njit(f8[:, :](f8[:, :], i8, b1, DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def mstd_apply_func_nb(ts, window, ewm, cache_dict):
    return cache_dict[(window, int(ewm))]


MSTD = IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['mstd'],
    name='mstd'
).from_apply_func(mstd_apply_func_nb, caching_func=mstd_caching_nb)


class MSTD(MSTD):
    """Standard deviation is an indicator that measures the size of an assets recent price moves 
    in order to predict how volatile the price may be in the future.

    Use `MSTD.from_params` method to run the indicator."""

    @classmethod
    def from_params(cls, ts, window, ewm=False, **kwargs):
        """Calculate moving standard deviation `MSTD.mstd` from time series `ts` and 
        parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window. Can be one or more values.
            ewm (bool or array_like): If True, uses exponential moving standard deviation, 
                otherwise uses simple moving standard deviation. Can be one or more values. 
                Defaults to False.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.from_params_pipeline.`
        Examples:
            ```python-repl
            >>> mstd = vbt.MSTD.from_params(price, [10, 20], ewm=[False, True])

            >>> print(mstd.mstd)
            mstd_window         10          20
            mstd_ewm         False        True
            Date                              
            2018-12-31         NaN         NaN
            2019-01-01         NaN         NaN
            2019-01-02         NaN         NaN
            ...                ...         ...
            2019-12-29   96.910551  267.336921
            2019-12-30   91.462377  254.361497
            2019-12-31   91.077352  244.531229

            [366 rows x 2 columns]
            ```
        """
        return super().from_params(ts, window, ewm, **kwargs)

    def plot(self,
             name=None,
             trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `MSTD.mstd`.

        Args:
            name (str): Name of the trace.
            trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html).
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Examples:
            ```py
            mstd[(10, False)].plot()
            ```

            ![](img/MSTD.png)"""
        check_type(self.mstd, pd.Series)

        if name is None:
            name = f'MSTD ({self.name})'

        fig = self.mstd.vbt.timeseries.plot(name=name, trace_kwargs=trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(MSTD)

# ############# BollingerBands ############# #


@njit(UniTuple(DictType(UniTuple(i8, 2), f8[:, :]), 2)(f8[:, :], i8[:], b1[:], f8[:]), cache=True)
def bb_caching_nb(ts, windows, ewms, alphas):
    ma_cache_dict = ma_caching_nb(ts, windows, ewms)
    mstd_cache_dict = mstd_caching_nb(ts, windows, ewms)
    return ma_cache_dict, mstd_cache_dict


@njit(UniTuple(f8[:, :], 3)(f8[:, :], i8, b1, f8, DictType(UniTuple(i8, 2), f8[:, :]), DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def bb_apply_func_nb(ts, window, ewm, alpha, ma_cache_dict, mstd_cache_dict):
    # Calculate lower, middle and upper bands
    ma = np.copy(ma_cache_dict[(window, int(ewm))])
    mstd = np.copy(mstd_cache_dict[(window, int(ewm))])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


BollingerBands = IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm', 'alpha'],
    output_names=['ma', 'upper_band', 'lower_band'],
    name='bb',
    custom_properties=dict(
        percent_b=lambda self: (self.ts - self.lower_band) / (self.upper_band - self.lower_band),
        bandwidth=lambda self: (self.upper_band - self.lower_band) / self.ma
    )
).from_apply_func(bb_apply_func_nb, caching_func=bb_caching_nb)


class BollingerBands(BollingerBands):
    """A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard 
    deviations (positively and negatively) away from a simple moving average (SMA) of the security's 
    price, but can be adjusted to user preferences.

    See [Bollinger Band® Definition](https://www.investopedia.com/terms/b/bollingerbands.asp).

    Use `BollingerBands.from_params` method to run the indicator."""
    
    @classmethod
    def from_params(cls, ts, window=20, ewm=False, alpha=2, **kwargs):
        """Calculate moving average `BollingerBands.ma`, upper Bollinger Band `BollingerBands.upper_band`,
        lower Bollinger Band `BollingerBands.lower_band`, %b `BollingerBands.percent_b` and 
        bandwidth `BollingerBands.bandwidth` from time series `ts` and parameters `window`, `ewm` and `alpha`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window. Can be one or more values.
                Defaults to 20.
            ewm (bool or array_like): If True, uses exponential moving average and standard deviation, 
                otherwise uses simple moving average and standard deviation. Can be one or more values. 
                Defaults to False.
            alpha (int, float or array_like): Number of standard deviations. Can be one or more values. Defaults to 2.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.from_params_pipeline.`
        Examples:
            ```python-repl
            >>> bb = vbt.BollingerBands.from_params(price, 
            ...     window=[10, 20], alpha=[2, 3], ewm=[False, True])

            >>> print(bb.ma)
            bb_window         10           20
            bb_ewm         False         True
            bb_alpha         2.0          3.0
            Date                             
            2018-12-31       NaN          NaN
            2019-01-01       NaN          NaN
            2019-01-02       NaN          NaN
            ...              ...          ...
            2019-12-29  7314.459  7313.283227
            2019-12-30  7321.877  7311.351491
            2019-12-31  7322.121  7300.137063

            [366 rows x 2 columns]

            >>> print(bb.upper_band)
            bb_window            10           20
            bb_ewm            False         True
            bb_alpha            2.0          3.0
            Date                                
            2018-12-31          NaN          NaN
            2019-01-01          NaN          NaN
            2019-01-02          NaN          NaN
            ...                 ...          ...
            2019-12-29  7508.280102  8115.293989
            2019-12-30  7504.801754  8074.435981
            2019-12-31  7504.275704  8033.730750

            [366 rows x 2 columns]

            >>> print(bb.lower_band)
            bb_window            10           20
            bb_ewm            False         True
            bb_alpha            2.0          3.0
            Date                                
            2018-12-31          NaN          NaN
            2019-01-01          NaN          NaN
            2019-01-02          NaN          NaN
            ...                 ...          ...
            2019-12-29  7120.637898  6511.272465
            2019-12-30  7138.952246  6548.267001
            2019-12-31  7139.966296  6566.543376

            [366 rows x 2 columns]

            >>> print(bb.percent_b)
            bb_window         10        20
            bb_ewm         False      True
            bb_alpha         2.0       3.0
            Date                          
            2018-12-31       NaN       NaN
            2019-01-01       NaN       NaN
            2019-01-02       NaN       NaN
            ...              ...       ...
            2019-12-29  0.779100  0.568183
            2019-12-30  0.421069  0.487975
            2019-12-31  0.147220  0.427387

            [366 rows x 2 columns]

            >>> print(bb.bandwidth)
            bb_window         10        20
            bb_ewm         False      True
            bb_alpha         2.0       3.0
            Date                          
            2018-12-31       NaN       NaN
            2019-01-01       NaN       NaN
            2019-01-02       NaN       NaN
            ...              ...       ...
            2019-12-29  0.779100  0.568183
            2019-12-30  0.421069  0.487975
            2019-12-31  0.147220  0.427387

            [366 rows x 2 columns]
            ```
        """
        alpha = np.asarray(alpha).astype(np.float64)
        return super().from_params(ts, window, ewm, alpha, **kwargs)

    def plot(self,
             ts_name=None,
             ma_name=None,
             upper_band_name=None,
             lower_band_name=None,
             ts_trace_kwargs={},
             ma_trace_kwargs={},
             upper_band_trace_kwargs={},
             lower_band_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `BollingerBands.ma`, `BollingerBands.upper_band` and `BollingerBands.lower_band` against 
        `BollingerBands.ts`.

        Args:
            ts_name (str): Name of the trace for `BollingerBands.ts`.
            ma_name (str): Name of the trace for `BollingerBands.ma`.
            upper_band_name (str): Name of the trace for `BollingerBands.upper_band`.
            lower_band_name (str): Name of the trace for `BollingerBands.lower_band`.
            ts_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.ts`.
            ma_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.ma`.
            upper_band_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.upper_band`.
            lower_band_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.lower_band`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Examples:
            ```py
            bb[(10, False, 2)].plot()
            ```

            ![](img/BollingerBands.png)"""
        check_type(self.ts, pd.Series)
        check_type(self.ma, pd.Series)
        check_type(self.upper_band, pd.Series)
        check_type(self.lower_band, pd.Series)

        if ts_name is None:
            ts_name = f'Price ({self.name})'
        if ma_name is None:
            ma_name = f'MA ({self.name})'
        if upper_band_name is None:
            upper_band_name = f'Upper Band ({self.name})'
        if lower_band_name is None:
            lower_band_name = f'Lower Band ({self.name})'

        upper_band_trace_kwargs = {**dict(line=dict(color='grey')), **upper_band_trace_kwargs}  # default kwargs
        lower_band_trace_kwargs = {**dict(line=dict(color='grey')), **lower_band_trace_kwargs}

        fig = self.ts.vbt.timeseries.plot(
            name=ts_name, trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.timeseries.plot(
            name=ma_name, trace_kwargs=ma_trace_kwargs, fig=fig)
        fig = self.upper_band.vbt.timeseries.plot(
            name=upper_band_name, trace_kwargs=upper_band_trace_kwargs, fig=fig)
        fig = self.lower_band.vbt.timeseries.plot(
            name=lower_band_name, trace_kwargs=lower_band_trace_kwargs, fig=fig)

        return fig


fix_class_for_pdoc(BollingerBands)


# ############# RSI ############# #

@njit(DictType(UniTuple(i8, 2), UniTuple(f8[:, :], 2))(f8[:, :], i8[:], b1[:]), cache=True)
def rsi_caching_nb(ts, windows, ewms):
    delta = diff_nb(ts)[1:, :]  # otherwise ewma will be all NaN
    up, down = delta.copy(), delta.copy()
    up = set_by_mask_nb(up, up < 0, 0)
    down = np.abs(set_by_mask_nb(down, down > 0, 0))
    # Cache
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                roll_up = ewm_mean_nb(up, windows[i])
                roll_down = ewm_mean_nb(down, windows[i])
            else:
                roll_up = rolling_mean_nb(up, windows[i])
                roll_down = rolling_mean_nb(down, windows[i])
            roll_up = prepend_nb(roll_up, 1, np.nan)  # bring to old shape
            roll_down = prepend_nb(roll_down, 1, np.nan)
            cache_dict[(windows[i], int(ewms[i]))] = roll_up, roll_down
    return cache_dict


@njit(f8[:, :](f8[:, :], i8, b1, DictType(UniTuple(i8, 2), UniTuple(f8[:, :], 2))), cache=True)
def rsi_apply_func_nb(ts, window, ewm, cache_dict):
    roll_up, roll_down = cache_dict[(window, int(ewm))]
    return 100 - 100 / (1 + roll_up / roll_down)


RSI = IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['rsi'],
    name='rsi'
).from_apply_func(rsi_apply_func_nb, caching_func=rsi_caching_nb)


class RSI(RSI):
    """
    The relative strength index (RSI) is a momentum indicator that measures the magnitude of 
    recent price changes to evaluate overbought or oversold conditions in the price of a stock 
    or other asset. The RSI is displayed as an oscillator (a line graph that moves between two 
    extremes) and can have a reading from 0 to 100.

    See [Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp).

    Use `RSI.from_params` methods to run the indicator."""
    @classmethod
    def from_params(cls, ts, window=14, ewm=False, **kwargs):
        """Calculate relative strength index `RSI.rsi` from time series `ts` and parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window. Can be one or more values.
            ewm (bool or array_like): If True, uses exponential moving average, otherwise 
                simple moving average. Can be one or more values. Defaults to False.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.from_params_pipeline.`
        Examples:
            ```python-repl
            >>> rsi = vbt.RSI.from_params(price, [10, 20], ewm=[False, True])

            >>> print(rsi.rsi)
            rsi_window         10         20
            rsi_ewm         False      True 
            Date                            
            2018-12-31        NaN        NaN
            2019-01-01        NaN        NaN
            2019-01-02        NaN        NaN
            ...               ...        ...
            2019-12-29  63.397004  55.208736
            2019-12-30  53.970922  49.140169
            2019-12-31  50.121299  44.953035

            [366 rows x 2 columns]
            ```
        """
        return super().from_params(ts, window, ewm, **kwargs)

    def plot(self,
             name=None,
             trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `RSI.rsi`.

        Args:
            name (str): Name of the trace.
            trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html).
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Examples:
            ```py
            rsi[(10, False)].plot()
            ```

            ![](img/RSI.png)"""
        check_type(self.rsi, pd.Series)

        if name is None:
            name = f'RSI ({self.name})'

        fig = self.rsi.vbt.timeseries.plot(name=name, trace_kwargs=trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(RSI)


# ############# Stochastic ############# #


@njit(DictType(i8, UniTuple(f8[:, :], 2))(f8[:, :], f8[:, :], f8[:, :], i8[:], i8[:], b1[:]), cache=True)
def stoch_caching_nb(close_ts, high_ts, low_ts, k_windows, d_windows, ewms):
    cache_dict = dict()
    for i in range(k_windows.shape[0]):
        if k_windows[i] not in cache_dict:
            roll_min = rolling_min_nb(low_ts, k_windows[i])
            roll_max = rolling_max_nb(high_ts, k_windows[i])
            cache_dict[k_windows[i]] = roll_min, roll_max
    return cache_dict


@njit(UniTuple(f8[:, :], 2)(f8[:, :], f8[:, :], f8[:, :], i8, i8, b1, DictType(i8, UniTuple(f8[:, :], 2))), cache=True)
def stoch_apply_func_nb(close_ts, high_ts, low_ts, k_window, d_window, ewm, cache_dict):
    roll_min, roll_max = cache_dict[k_window]
    percent_k = 100 * (close_ts - roll_min) / (roll_max - roll_min)
    if ewm:
        percent_d = ewm_mean_nb(percent_k, d_window)
    else:
        percent_d = rolling_mean_nb(percent_k, d_window)
    percent_d[:k_window+d_window-2, :] = np.nan  # min_periods for ewm
    return percent_k, percent_d


Stochastic = IndicatorFactory(
    ts_names=['close_ts', 'high_ts', 'low_ts'],
    param_names=['k_window', 'd_window', 'ewm'],
    output_names=['percent_k', 'percent_d'],
    name='stoch'
).from_apply_func(stoch_apply_func_nb, caching_func=stoch_caching_nb)


class Stochastic(Stochastic):
    @classmethod
    def from_params(cls, close_ts, high_ts=None, low_ts=None, k_window=14, d_window=3, ewm=False, **kwargs):
        if high_ts is None:
            high_ts = close_ts
        if low_ts is None:
            low_ts = close_ts
        return super().from_params(close_ts, high_ts, low_ts, k_window, d_window, ewm, **kwargs)

    def crossover(self, **kwargs):
        above_signals = self.percent_k_above(self.percent_d, crossover=True, **kwargs)
        below_signals = self.percent_k_below(self.percent_d, crossover=True, **kwargs)
        return above_signals, below_signals

    def plot(self,
             percent_k_name=None,
             percent_d_name=None,
             percent_k_trace_kwargs={},
             percent_d_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.percent_k, pd.Series)
        check_type(self.percent_d, pd.Series)

        if percent_k_name is None:
            percent_k_name = f'%K ({self.name})'
        if percent_d_name is None:
            percent_d_name = f'%D ({self.name})'

        fig = self.percent_k.vbt.timeseries.plot(
            name=percent_k_name, trace_kwargs=percent_k_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.percent_d.vbt.timeseries.plot(
            name=percent_d_name, trace_kwargs=percent_d_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(Stochastic)


# ############# MACD ############# #

@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], i8[:], i8[:], b1[:]), cache=True)
def macd_caching_nb(ts, fast_windows, slow_windows, signal_windows, ewms):
    return ma_caching_nb(ts, np.concatenate((fast_windows, slow_windows)), np.concatenate((ewms, ewms)))


@njit(UniTuple(f8[:, :], 4)(f8[:, :], i8, i8, i8, b1, DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def macd_apply_func_nb(ts, fast_window, slow_window, signal_window, ewm, cache_dict):
    fast_ma = cache_dict[(fast_window, int(ewm))]
    slow_ma = cache_dict[(slow_window, int(ewm))]
    macd_ts = fast_ma - slow_ma
    if ewm:
        signal_ts = ewm_mean_nb(macd_ts, signal_window)
    else:
        signal_ts = rolling_mean_nb(macd_ts, signal_window)
    signal_ts[:max(fast_window, slow_window)+signal_window-2, :] = np.nan  # min_periods for ewm
    return np.copy(fast_ma), np.copy(slow_ma), macd_ts, signal_ts


MACD = IndicatorFactory(
    ts_names=['ts'],
    param_names=['fast_window', 'slow_window', 'signal_window', 'ewm'],
    output_names=['fast_ma', 'slow_ma', 'macd', 'signal'],
    name='macd',
    custom_properties=dict(
        histogram=lambda self: self.macd - self.signal,
    )
).from_apply_func(macd_apply_func_nb, caching_func=macd_caching_nb)


class MACD(MACD):
    @classmethod
    def from_params(cls, ts, fast_window=26, slow_window=12, signal_window=9, ewm=True, **kwargs):
        return super().from_params(ts, fast_window, slow_window, signal_window, ewm, **kwargs)

    def crossover(self, **kwargs):
        above_signals = self.macd_above(self.signal, crossover=True, **kwargs)
        below_signals = self.macd_below(self.signal, crossover=True, **kwargs)
        return above_signals, below_signals

    def plot(self,
             macd_name=None,
             signal_name=None,
             macd_trace_kwargs={},
             signal_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.macd, pd.Series)
        check_type(self.signal, pd.Series)

        if macd_name is None:
            macd_name = f'MACD ({self.name})'
        if signal_name is None:
            signal_name = f'Signal ({self.name})'

        fig = self.macd.vbt.timeseries.plot(
            name=macd_name, trace_kwargs=macd_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.signal.vbt.timeseries.plot(
            name=signal_name, trace_kwargs=signal_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(MACD)


# ############# OBV ############# #

@njit(f8[:, :](f8[:, :], f8[:, :]))
def obv_custom_func_nb(close_ts, volume_ts):
    obv = np.full_like(close_ts, np.nan)
    for col in range(close_ts.shape[1]):
        cumsum = 0
        for i in range(1, close_ts.shape[0]):
            if np.isnan(close_ts[i, col]) or np.isnan(close_ts[i-1, col]) or np.isnan(volume_ts[i, col]):
                continue
            if close_ts[i, col] > close_ts[i-1, col]:
                cumsum += volume_ts[i, col]
            elif close_ts[i, col] < close_ts[i-1, col]:
                cumsum += -volume_ts[i, col]
            obv[i, col] = cumsum
    return obv


def obv_custom_func(close_ts, volume_ts):
    return obv_custom_func_nb(close_ts.vbt.to_2d_array(), volume_ts.vbt.to_2d_array())


OBV = IndicatorFactory(
    ts_names=['close_ts', 'volume_ts'],
    param_names=[],
    output_names=['obv'],
    name='obv'
).from_custom_func(obv_custom_func)


class OBV(OBV):
    @classmethod
    def from_params(cls, close_ts, volume_ts):
        return super().from_params(close_ts, volume_ts)

    def plot(self,
             name=None,
             trace_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.obv, pd.Series)

        if name is None:
            name = f'OBV ({self.name})'

        fig = self.obv.vbt.timeseries.plot(name=name, trace_kwargs=trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(OBV)
