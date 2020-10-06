"""A factory for building new indicators with ease.

Each indicator is basically a pipeline that:

* Accepts a list of time series objects (for example, OHLCV data)
* Accepts a list of parameter arrays (for example, size of rolling window)
* Accepts other relevant arguments and keyword arguments
* Performs calculations to produce new time series objects (for example, rolling average)

This pipeline can be well standardized, which is done by `run_pipeline`.

`IndicatorFactory` simplifies usage of `run_pipeline` by generating and pre-configuring
a new Python class with various methods for running the indicator. It has the following features:

* Accepts time series of any shape thanks to broadcasting
* Accepts arbitrary parameter combinations
* Supports pandas indexing, i.e., you can use `iloc`, `loc`, `xs`, and `__getitem__` on the class itself
* Supports parameter indexing, i.e., use `*your_param*_loc` on the class to slice using parameters
* Exposes common signal generation methods for all inputs, outputs and properties, e.g., crossover

Consider the following price DataFrame:

```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> price = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5),
... ])).astype(float)
>>> price
            a    b
2020-01-01  1.0  5.0
2020-01-02  2.0  4.0
2020-01-03  3.0  3.0
2020-01-04  4.0  2.0
2020-01-05  5.0  1.0
```

For each column in the DataFrame, let's calculate a simple moving average and get signals
of price crossing it. In particular, we want to test two different window sizes: 2 and 3.

## Naive approach

A naive way of doing this:

```python-repl
>>> ma_df = pd.DataFrame.vbt.concat(
...     price.rolling(window=2).mean(),
...     price.rolling(window=3).mean(),
...     keys=pd.Index([2, 3], name='ma_window'))
>>> ma_df
ma_window          2         3
              a    b    a    b
2020-01-01  NaN  NaN  NaN  NaN
2020-01-02  1.5  4.5  NaN  NaN
2020-01-03  2.5  3.5  2.0  4.0
2020-01-04  3.5  2.5  3.0  3.0
2020-01-05  4.5  1.5  4.0  2.0

>>> above_signals = (price.vbt.tile(2).vbt > ma_df)
>>> above_signals = above_signals.vbt.signals.first(after_false=True)
>>> above_signals
ma_window              2             3
                a      b      a      b
2020-01-01  False  False  False  False
2020-01-02   True  False  False  False
2020-01-03  False  False   True  False
2020-01-04  False  False  False  False
2020-01-05  False  False  False  False

>>> below_signals = (price.vbt.tile(2).vbt < ma_df)
>>> below_signals = below_signals.vbt.signals.first(after_false=True)
>>> below_signals
ma_window              2             3
                a      b      a      b
2020-01-01  False  False  False  False
2020-01-02  False   True  False  False
2020-01-03  False  False  False   True
2020-01-04  False  False  False  False
2020-01-05  False  False  False  False
```

## IndicatorFactory

Now the same using `IndicatorFactory`:

```python-repl
>>> MyMA = vbt.IndicatorFactory(
...     input_names=['price'],
...     param_names=['window'],
...     output_names=['ma'],
...     short_name='myma'
... ).from_apply_func(vbt.nb.rolling_mean_nb)

>>> myma = MyMA.run(price, [2, 3])
>>> above_signals = myma.price_above(myma.ma, crossed=True)
>>> below_signals = myma.price_below(myma.ma, crossed=True)
```

The `IndicatorFactory` class is used to construct indicator classes from UDFs. First, you provide
all the necessary information to build the facade of the indicator, such as input, parameter and
output names, and the actual calculation function. The factory then generates a self-contained
indicator class capable of running arbitrary configurations of inputs and parameters. To run any
configuration, you can either use the `run` method (as we did above) or the `run_combs` method.

### run method

The main method to run an indicator is `run` that accepts 1) input time series, 2) parameters
(either positional arguments or keyword arguments if you specified `param_defaults`), and 3)
other arguments that are accepted by the calculation function.

Input time series can have any shape as long as they are Series or DataFrames. Passing multiple time
series with different shapes will broadcast them to a single shape.

```python-repl
>>> MyInd = vbt.IndicatorFactory(
...     input_names=['price1', 'price2'],
...     param_names=['p1', 'p2']
... ).from_apply_func(
...     lambda price1, price2, p1, p2: price1 * p1 + price2 * p2
... )

>>> myInd = MyInd.run(price['a'], price['b'], 1, 2)
>>> myInd.output
2020-01-01    11.0
2020-01-02    10.0
2020-01-03     9.0
2020-01-04     8.0
2020-01-05     7.0
Name: (1, 2, a, b), dtype: float64

>>> myInd = MyInd.run(price, price['b'], 1, 2)
>>> myInd.output
custom_p1            1
custom_p2            2
               a     b
2020-01-01  11.0  15.0
2020-01-02  10.0  12.0
2020-01-03   9.0   9.0
2020-01-04   8.0   6.0
2020-01-05   7.0   3.0
```

Parameters are also flexible: they can be either single values, or arrays to run multiple
configurations at once. Multiple parameters will broadcast together to have the same length.
You can even set `param_product` to True to run all possible combinations of passed parameter values.

```python-repl
>>> myInd = MyInd.run(price['a'], price['b'], 1, 2)
>>> myInd.p1_array
array([1])
>>> myInd.p2_array
array([2])

>>> myInd = MyInd.run(price['a'], price['b'], 1, [2, 3])
>>> myInd.p1_array
array([1, 1])
>>> myInd.p2_array
array([2, 3])

>>> myInd = MyInd.run(price['a'], price['b'], [1, 2], [3, 4], param_product=True)
>>> myInd.p1_array
array([1, 1, 2, 2])
>>> myInd.p2_array
array([3, 4, 3, 4])
```

The output of the `run` method will be the instance of the indicator.
All outputs can be then accessed as variables of the instance.

### run_combs method

The `run_combs` method takes the same inputs as the method above, but computes all combinations
of passed parameters and returns multiple instances that can be compared with each other.
For example, this is useful to generate crossover signals of multiple moving averages.

```python-repl
>>> myma1, myma2 = MyMA.run_combs(price, [2, 3, 4])

>>> myma1.ma
myma_1_window                   2         3
                 a    b    a    b    a    b
2020-01-01     NaN  NaN  NaN  NaN  NaN  NaN
2020-01-02     1.5  4.5  1.5  4.5  NaN  NaN
2020-01-03     2.5  3.5  2.5  3.5  2.0  4.0
2020-01-04     3.5  2.5  3.5  2.5  3.0  3.0
2020-01-05     4.5  1.5  4.5  1.5  4.0  2.0

>>> myma2.ma
myma_2_window         3                   4
                 a    b    a    b    a    b
2020-01-01     NaN  NaN  NaN  NaN  NaN  NaN
2020-01-02     NaN  NaN  NaN  NaN  NaN  NaN
2020-01-03     2.0  4.0  NaN  NaN  NaN  NaN
2020-01-04     3.0  3.0  2.5  3.5  2.5  3.5
2020-01-05     4.0  2.0  3.5  2.5  3.5  2.5

>>> myma1.ma_above(myma2.ma, crossed=True)
myma_1_window                           2             3
myma_2_window             3             4             4
                   a      b      a      b      a      b
2020-01-01     False  False  False  False  False  False
2020-01-02     False  False  False  False  False  False
2020-01-03      True  False  False  False  False  False
2020-01-04     False  False   True  False   True  False
2020-01-05     False  False  False  False  False  False
```

The main advantage is that it doesn't re-compute each combination thanks to caching.

### Comparison methods

For all our inputs in `input_names` and outputs in `output_names`, it created a bunch of comparison methods
for generating signals, such as `above`, `below` and `equal` (use `dir()`):

```python-repl
'ma_above',
'ma_below',
'ma_equal',
'price_above',
'price_below',
'price_equal',
```

Each of these methods uses vectorbt's own broadcasting, so you can compare time series objects with an
arbitrary array-like object, given their shapes can be broadcasted together. You can also compare them
to multiple objects at once, for example:

```python-repl
>>> myma.ma_above([1.5, 2.5], multiple=True)
myma_ma_above                         1.5                         2.5
myma_window               2             3             2             3
                a         b      a      b      a      b      a      b
2020-01-01     False  False  False  False  False  False  False  False
2020-01-02     False   True  False  False  False   True  False  False
2020-01-03      True   True   True   True  False   True  False   True
2020-01-04      True   True   True   True   True  False   True   True
2020-01-05      True  False   True   True   True  False   True  False
```

### Indexing

`IndicatorFactory` also attaches pandas indexing to the indicator class:

```python-repl
'iloc'
'loc'
'window_loc'
'xs'
```

This makes accessing rows and columns by labels, integer positions, and parameters much easier.

```python-repl
>>> myma[(2, 'b')]
<vectorbt.indicators.factory.CustomIndicator at 0x7fa4b3e0c4e0>

>>> myma[(2, 'b')].ma
2020-01-01    NaN
2020-01-02    4.5
2020-01-03    3.5
2020-01-04    2.5
2020-01-05    1.5
Name: (2, b), dtype: float64
```
"""
import numpy as np
import pandas as pd
from numba import njit
from numba.typed import List
import itertools
import inspect
from collections import OrderedDict

from vectorbt.utils import checks
from vectorbt.utils.decorators import classproperty, cached_property
from vectorbt.utils.config import merge_kwargs, Configured
from vectorbt.base import index_fns, reshape_fns, combine_fns
from vectorbt.base.indexing import PandasIndexer, ParamIndexerFactory
from vectorbt.base.array_wrapper import ArrayWrapper, indexing_on_wrapper_meta


def flatten_param_tuples(param_tuples):
    """Flattens a nested list of tuples using unzipping."""
    param_list = []
    unzipped_tuples = zip(*param_tuples)
    for i, unzipped in enumerate(unzipped_tuples):
        unzipped = tuple(unzipped)
        if isinstance(unzipped[0], tuple):
            param_list.extend(flatten_param_tuples(unzipped))
        else:
            param_list.append(unzipped)
    return param_list


def create_param_combs(op_tree, depth=0):
    """Create arbitrary parameter combinations from the operation tree `op_tree`.

    `op_tree` must be a tuple of tuples, each being an instruction to generate parameters.
    The first element of each tuple should a function that takes remaining elements as arguments.
    If one of the elements is a tuple, it will be unfolded in the same way.

    Example:
        ```python-repl
        >>> import numpy as np
        >>> from itertools import combinations, product

        >>> create_param_combs((product, (combinations, [0, 1, 2, 3], 2), [4, 5]))
        [(0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2),
         (1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3),
         (4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5)]
        ```
    """
    checks.assert_type(op_tree, tuple)
    new_op_tree = (op_tree[0],)
    for elem in op_tree[1:]:
        if isinstance(elem, tuple):
            new_op_tree += (create_param_combs(elem, depth=depth + 1),)
        else:
            new_op_tree += (elem,)
    out = list(new_op_tree[0](*new_op_tree[1:]))
    if depth == 0:
        # do something
        return flatten_param_tuples(out)
    return out


def broadcast_params(param_list):
    """Broadcast parameters in `param_list`."""
    max_len = max(list(map(len, param_list)))
    new_param_list = []
    for i in range(len(param_list)):
        params = param_list[i]
        if len(params) < max_len:
            if len(params) > 1:
                raise ValueError("shape mismatch: objects cannot be broadcast to a single shape")
            new_params = []
            for j in range(max_len):
                new_params.append(params[0])
            new_param_list.append(tuple(new_params))
        else:
            new_param_list.append(tuple(params))
    return new_param_list


def create_param_product(param_list):
    """Make Cartesian product out of all params in `param_list`."""
    return list(map(tuple, zip(*list(itertools.product(*param_list)))))


def prepare_params(param_list, param_settings, input_shape=None):
    """Prepare parameters."""
    new_param_list = []
    for i, params in enumerate(param_list):
        _param_settings = param_settings if isinstance(param_settings, dict) else param_settings[i]
        is_array_like = _param_settings.get('array_like', False)
        bc_to_input = _param_settings.get('bc_to_input', False)
        broadcast_kwargs = _param_settings.get('broadcast_kwargs', dict(require_kwargs=dict(requirements='W')))

        if is_array_like:
            check_against = (list, tuple, List)
        else:
            check_against = (list, tuple, List, np.ndarray)
        if isinstance(params, check_against):
            new_params = tuple(params)
        else:
            new_params = (params,)
        if bc_to_input:
            if input_shape is None:
                raise ValueError("Cannot broadcast to input if input shape is unknown")
            _new_params = reshape_fns.broadcast(
                *new_params,
                to_shape=input_shape,
                **broadcast_kwargs
            )
            if len(new_params) > 1:
                new_params = _new_params
            else:
                new_params = (_new_params,)
        new_param_list.append(new_params)
    return new_param_list


def reindex_outputs(new_params, from_params, n_ts_cols):
    """Return indices of `new_params` in `run` corrected by the number of columns `n_ts_cols`."""
    idxs = np.array([from_params.index(param_tuple) for param_tuple in new_params])
    idx_map = np.arange(len(from_params) * n_ts_cols).reshape(len(from_params), n_ts_cols)
    return idx_map[idxs].flatten()


def build_column_hierarchy(param_list, level_names, input_columns, hide_levels=[]):
    """For each parameter in `param_list`, create a new column level with parameter values. 
    Combine this level with columns `input_columns` using Cartesian product."""
    checks.assert_len_equal(param_list, level_names)

    param_indexes = []
    for i in range(len(param_list)):
        if level_names[i] not in hide_levels:
            param_index = index_fns.index_from_values(param_list[i], name=level_names[i])
            param_indexes.append(param_index)
    if len(param_indexes) > 1:
        param_columns = index_fns.stack_indexes(*param_indexes)
    elif len(param_indexes) == 1:
        param_columns = param_indexes[0]
    else:
        param_columns = None
    if param_columns is not None:
        return index_fns.combine_indexes(param_columns, input_columns)
    return input_columns


def run_pipeline(
        input_list,
        param_list,
        level_names,
        num_outputs,
        custom_func,
        *args,
        input_shape=None,
        input_index=None,
        input_columns=None,
        broadcast_kwargs=None,
        param_settings=None,
        param_product=False,
        to_2d=True,
        pass_lists=False,
        pass_input_shape=None,
        pass_is_2d=False,
        hide_levels=None,
        return_raw=False,
        use_raw=None,
        wrapper_kwargs=None,
        **kwargs):
    """A pipeline for calculating an indicator, used by `IndicatorFactory`.

    Args:
        input_list (list of array_like): A list of time series objects.
        param_list (list of array_like): A list of parameters.

            Each element is either an array-like object or a single value of any type.
        level_names (list of str): A list of column level names corresponding to each parameter.

            Should have the same length as `param_list`.
        num_outputs (int): The number of output arrays.
        custom_func (callable): A custom calculation function.

            See `IndicatorFactory.from_custom_func`.
        *args: Arguments passed to the `custom_func`.
        input_shape (tuple): Shape to broadcast each input to.

            Can be passed to `custom_func`. See `pass_input_shape`.
        input_index (any): Sets index of each input.

            Can be used to label index if no inputs passed.
        input_columns (any): Sets columns of each input.

            Can be used to label columns if no inputs passed.
        broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`
            to broadcast inputs.
        param_settings (dict or list of dict): Settings corresponding to each parameter.

            Following keys are accepted:

            * `array_like`: If array-like object was passed, it will be considered as a single value.
                To treat it as multiple values, pack it into a list.
            * `bc_to_input`: Whether to broadcast parameter to input size.
            * `broadcast_kwargs`: Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
        param_product (bool): Whether to build a Cartesian product out of all parameters.
        to_2d (bool): Whether to reshape inputs to 2-dim arrays, otherwise keep as-is.
        pass_lists (bool): Whether to pass inputs and parameters to `custom_func` as lists.
        pass_input_shape (bool): Whether to pass `input_shape` to `custom_func` as keyword argument.

            If None and no inputs passed, passes `input_shape` automatically.
        pass_is_2d (bool): Whether to pass `is_2d` to `custom_func` as keyword argument.
        hide_levels (list): A list of parameter levels to hide.
        return_raw (bool): Whether to return raw output without post-processing and hashed parameter tuples.
        use_raw (bool): Takes the raw results and uses them instead of running `custom_func`.
        wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.
        **kwargs: Keyword arguments passed to the `custom_func`.

            Some common arguments include `return_cache` to return cache and `use_cache` to use cache.
            Those are only applicable to `custom_func` that supports it (`custom_func` created using
            `IndicatorFactory.from_apply_func` are supported by default).
            
    Returns:
        Array wrapper, list of inputs (`np.ndarray`), input mapper (`np.ndarray`), list of outputs
        (`np.ndarray`), list of parameter arrays (`np.ndarray`), list of parameter mappers (`np.ndarray`),
        list of outputs that are outside of `num_outputs`.

    Explanation:
        Does the following:

        * Takes one or multiple time series objects in `input_list` and broadcasts them. For example:

        ```python-repl
        >>> sr = pd.Series([1, 2], index=['x', 'y'])
        >>> df = pd.DataFrame([[3, 4], [5, 6]], index=['x', 'y'], columns=['a', 'b'])
        >>> input_list = [sr, df]

        >>> input_list = vbt.base.reshape_fns.broadcast(*input_list)
        >>> input_list[0]
           a  b
        x  1  1
        y  2  2
        >>> input_list[1]
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
        >>> param_list[0]
        array([1, 1, 1])
        >>> param_list[1]
        array([2, 3, 4])
        >>> param_list[2]
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

        >>> output = custom_func(*input_list, *param_list)
        >>> output
        array([[ 6,  7,  7,  8,  8,  9],
               [ 9, 10, 10, 11, 11, 12]])
        ```

        * Creates new column hierarchy based on parameters and level names. For example:

        ```python-repl
        >>> p1_columns = pd.Index(param_list[0], name='p1')
        >>> p2_columns = pd.Index(param_list[1], name='p2')
        >>> p3_columns = pd.Index(param_list[2], name='p3')
        >>> p_columns = vbt.base.index_fns.stack_indexes(p1_columns, p2_columns, p3_columns)
        >>> new_columns = vbt.base.index_fns.combine_indexes(p_columns, input_list[0].columns)

        >>> output_df = pd.DataFrame(output, columns=new_columns)
        >>> output_df
        p1                                         1                        
        p2             2             3             4    
        p3  False  False  False  False  False  False    
                a      b      a      b      a      b
        0       6      7      7      8      8      9
        1       9     10     10     11     11     12
        ```

        * Broadcasts objects in `input_list` to match the shape of objects in `output_list` through tiling.
            This is done to be able to compare them and generate signals, since you cannot compare NumPy 
            arrays that have totally different shapes, such as (2, 2) and (2, 6). For example:

        ```python-repl
        >>> new_input_list = [
        ...     input_list[0].vbt.tile(len(param_list[0]), keys=p_columns),
        ...     input_list[1].vbt.tile(len(param_list[0]), keys=p_columns)
        ... ]
        >>> new_input_list[0]
        p1                                         1                        
        p2             2             3             4    
        p3  False  False  False  False  False  False     
                a      b      a      b      a      b
        0       1      1      1      1      1      1
        1       2      2      2      2      2      2
        ```

        * Builds parameter mappers that will link parameters from `param_list` to columns in 
            `input_list` and `output_list`. This is done to enable column indexing using parameter values.
    """
    if broadcast_kwargs is None:
        broadcast_kwargs = {}
    if param_settings is None:
        param_settings = {}
    if hide_levels is None:
        hide_levels = []
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    if pass_input_shape is None:
        if input_shape is not None and len(input_list) == 0:
            pass_input_shape = True
        else:
            pass_input_shape = False

    if len(input_list) > 0:
        # Broadcast time series
        if input_index is None:
            input_index = 'default'
        if input_columns is None:
            input_columns = 'default'
        broadcast_kwargs = merge_kwargs(dict(
            to_shape=input_shape,
            index_from=input_index,
            columns_from=input_columns
        ), broadcast_kwargs)
        bc_input_list, input_shape, input_index, input_columns = reshape_fns.broadcast(
            *input_list,
            return_meta=True,
            **broadcast_kwargs
        )
        if len(input_list) == 1:
            bc_input_list = (bc_input_list,)
        input_list = list(map(np.asarray, bc_input_list))

    # Reshape inputs
    passed_input_list = input_list
    passed_input_shape = input_shape
    if to_2d:
        passed_input_list = list(map(reshape_fns.to_2d, input_list))
        if input_shape is not None:
            passed_input_shape = input_shape if len(input_shape) > 1 else (input_shape[0], 1)

    # Prepare parameters
    param_list = prepare_params(param_list, param_settings, input_shape=passed_input_shape)
    if len(param_list) > 1:
        # Check level names
        checks.assert_type(level_names, (list, tuple))
        checks.assert_len_equal(param_list, level_names)
        # Columns should be free of the specified level names
        if input_columns is not None:
            for level_name in level_names:
                if level_name is not None:
                    checks.assert_level_not_exists(input_columns, level_name)
        if param_product:
            # Make Cartesian product out of all params
            param_list = create_param_product(param_list)
        else:
            # Broadcast such that each array has the same length
            param_list = broadcast_params(param_list)
    n_param_values = len(param_list[0]) if len(param_list) > 0 else 1

    # Get raw results
    if use_raw is not None:
        # Use raw results of previous run to build outputs
        output_list, param_map, n_input_cols, other_list = use_raw
        idxs = reindex_outputs(list(zip(*param_list)), param_map, n_input_cols)
        output_list = [output[:, idxs] for output in output_list]
    else:
        # Run custom function
        func_kwargs = {}
        if pass_input_shape:
            func_kwargs['input_shape'] = passed_input_shape
        if pass_is_2d:
            if input_shape is None:
                raise ValueError("Cannot determine is_2d without inputs")
            func_kwargs['is_2d'] = len(input_shape) == 2
        func_kwargs = merge_kwargs(func_kwargs, kwargs)
        if pass_lists:
            if checks.is_numba_func(custom_func):
                output = custom_func(List(passed_input_list), List(param_list), *args, **func_kwargs)
            else:
                output = custom_func(passed_input_list, param_list, *args, **func_kwargs)
        else:
            output = custom_func(*passed_input_list, *param_list, *args, **func_kwargs)

        # Return cache
        if kwargs.get('return_cache', False):
            return output

        # Post-process results
        if not isinstance(output, (tuple, list, List)):
            output_list = [output]
        else:
            output_list = list(output)
        # Other outputs should be returned without post-processing (for example cache_dict)
        if len(output_list) > num_outputs:
            other_list = output_list[num_outputs:]
        else:
            other_list = []
        # Process only the num_outputs outputs
        output_list = output_list[:num_outputs]
        output_list = list(map(reshape_fns.to_2d, output_list))

        # Return raw results if needed
        param_map = list(zip(*param_list))
        n_input_cols = output_list[0].shape[1] // n_param_values
        if return_raw:
            return output_list, param_map, n_input_cols, other_list

    # Update shape info if no inputs
    if input_shape is None:
        input_shape = (output_list[0].shape[0], n_input_cols)
    if input_index is None:
        input_index = pd.RangeIndex(start=0, step=1, stop=input_shape[0])
    if input_columns is None:
        input_columns = pd.RangeIndex(start=0, step=1, stop=input_shape[1] if len(input_shape) > 1 else 1)

    # Build column hierarchy and create mappers
    if len(param_list) > 0:
        # Build new column levels on top of time series levels
        new_columns = build_column_hierarchy(param_list, level_names, input_columns, hide_levels)
        # Build a mapper that maps old columns in inputs to new columns.
        # Instead of tiling all inputs to the shape of outputs and wasting memory,
        # we just keep a mapper and perform the tiling when needed.
        input_mapper = None
        if len(input_list) > 0:
            input_mapper = np.tile(np.arange(len(input_columns)), n_param_values)
        # Build mappers to easily map between parameters and columns
        mapper_list = [np.repeat(index_fns.index_from_values(x), len(input_columns)) for x in param_list]
    else:
        # Some indicators don't have any params
        new_columns = input_columns
        input_mapper = None
        mapper_list = []

    # Return artifacts: no pandas objects, just a wrapper and NumPy arrays
    new_ndim = len(input_shape) if output_list[0].shape[1] == 1 else output_list[0].ndim
    wrapper = ArrayWrapper(input_index, new_columns, new_ndim, **wrapper_kwargs)
    return wrapper, input_list, input_mapper, output_list, param_list, mapper_list, other_list


def perform_init_checks(wrapper, input_list, input_mapper, output_list, param_list,
                        mapper_list, short_name, level_names):
    """Perform checks on objects created by running or slicing an indicator."""
    if input_mapper is not None:
        checks.assert_equal(input_mapper.shape[0], wrapper.shape_2d[1])
    for ts in input_list:
        checks.assert_equal(ts.shape[0], wrapper.shape_2d[0])
    for ts in output_list:
        checks.assert_equal(ts.shape, wrapper.shape_2d)
    for params in param_list:
        checks.assert_len_equal(param_list[0], params)
    for mapper in mapper_list:
        checks.assert_equal(len(mapper), wrapper.shape_2d[1])
    checks.assert_type(short_name, str)
    checks.assert_len_equal(level_names, param_list)


def compare(obj, other, compare_func, multiple=False, level_name=None, keys=None, **kwargs):
    """Compares `obj` to `other` to generate signals.

    Both will be broadcast together. Set `multiple` to True to compare with multiple arguments.
    In this case, a new column level will be created with the name `level_name`.

    See `vectorbt.base.accessors.Base_Accessor.combine_with`."""
    if multiple:
        if keys is None:
            keys = index_fns.index_from_values(other, name=level_name)
        return obj.vbt.combine_with_multiple(other, combine_func=compare_func, keys=keys, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=compare_func, **kwargs)


class IndicatorFactory():
    def __init__(self,
                 class_name='CustomIndicator',
                 class_docstring='',
                 module_name=__name__,
                 short_name='custom',
                 input_names=None,
                 param_names=None,
                 param_defaults=None,
                 param_settings=None,
                 output_names=None,
                 output_flags=None,
                 custom_output_funcs=None):
        """A factory for creating new indicators.

        Args:
            class_name (str): Name for the created Python class.
            class_docstring (str): Docstring for the created Python class.
            module_name (str): Specify the module the class originates from.
            short_name (str): A short name of the indicator.
            input_names (list of str): A list of names of input time series objects.
            param_names (list of str): A list of names of parameters.
            param_defaults (dict): A dictionary of parameter defaults.

                Params with defaults should be on the right in `param_names`.
            output_names (list of str): A list of names of outputs time series objects.
            output_flags (dict): A dictionary of output flags.
            custom_output_funcs (dict): A dictionary with user-defined functions that will be
                bound to the indicator class and wrapped with `@cached_property`.

        !!! note
            The `__init__` method is not used for running the indicator, for this use `run`.
            The reason for this is indexing, which requires a clean `__init__` method for creating 
            a new indicator object with newly indexed attributes.
        """
        self.class_name = class_name
        self.class_docstring = class_docstring
        self.module_name = module_name
        self.short_name = short_name
        if input_names is None:
            input_names = []
        self.input_names = input_names
        if param_names is None:
            param_names = []
        self.param_names = param_names
        if param_defaults is None:
            param_defaults = {}
        if len(param_defaults) > 0:
            for param_name in param_defaults:
                if param_name not in param_names:
                    raise ValueError(f"Param {param_name} not in param_names")
            if sorted(param_names[-len(param_defaults):]) != sorted(param_defaults.keys()):
                raise ValueError("Params with defaults should be on the right in param_names")
        self.param_defaults = param_defaults
        if param_settings is None:
            param_settings = {}
        self.param_settings = param_settings
        if output_names is None:
            output_names = []
        if len(output_names) == 0:
            raise ValueError("Must have at least one output")
        self.output_names = output_names
        if output_flags is None:
            output_flags = {}
        if len(output_flags) > 0:
            for output_name in output_flags:
                if output_name not in output_names:
                    raise ValueError(f"Output {output_name} not in output_names")
        self.output_flags = output_flags
        if custom_output_funcs is None:
            custom_output_funcs = {}
        self.custom_output_funcs = custom_output_funcs

    def from_custom_func(self, custom_func, **pipeline_kwargs):
        """Build indicator class around a custom calculation function.

        !!! note
            In contrast to `IndicatorFactory.from_apply_func`, it's up to you to handle caching
            and concatenate columns for each parameter (for example, by using 
            `vectorbt.base.combine_fns.apply_and_concat_one`). Also, you must ensure that each output
            array has an appropriate number of columns, which is the number of columns in input time 
            series multiplied by the number of parameter combinations.

        !!! note
            Time series passed to `apply_func` will be 2-dimensional NumPy arrays.

            For each parameter value, input and output time series should have the same shape.

        Args:
            custom_func (callable): A function that takes broadcasted time series corresponding 
                to `input_names`, broadcasted parameter arrays corresponding to `param_names`, and other
                arguments and keyword arguments, and returns outputs corresponding to `output_names`
                and other objects that are then returned with the indicator class instance.
                Can be Numba-compiled.
            **pipeline_kwargs: Default keyword arguments passed to `run_pipeline`.
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
            ...     input_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['o1', 'o2']
            ... ).from_custom_func(custom_func)

            >>> myInd = MyInd.run(price, price * 2, [1, 2], [3, 4], 100)
            >>> myInd.o1
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> myInd.o2
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
        short_name = self.short_name
        input_names = self.input_names
        param_names = self.param_names
        param_defaults = self.param_defaults
        param_settings = self.param_settings
        output_names = self.output_names
        output_flags = self.output_flags
        custom_output_funcs = self.custom_output_funcs

        # Add indexing methods
        def indexing_func(obj, pd_indexing_func):
            new_wrapper, idx_idxs, _, col_idxs = indexing_on_wrapper_meta(obj.wrapper, pd_indexing_func)
            idx_idxs_arr = reshape_fns.to_1d(idx_idxs, raw=True)
            col_idxs_arr = reshape_fns.to_1d(col_idxs, raw=True)
            if np.array_equal(idx_idxs_arr, np.arange(obj.wrapper.shape_2d[0])):
                idx_idxs_arr = slice(None, None, None)
            if np.array_equal(col_idxs_arr, np.arange(obj.wrapper.shape_2d[1])):
                col_idxs_arr = slice(None, None, None)

            input_mapper = getattr(obj, '_input_mapper', None)
            if input_mapper is not None:
                input_mapper = input_mapper[col_idxs_arr]
            input_list = []
            for input_name in input_names:
                input_list.append(getattr(obj, f'_{input_name}')[idx_idxs_arr])
            output_list = []
            for output_name in output_names:
                output_list.append(getattr(obj, f'_{output_name}')[idx_idxs_arr, :][:, col_idxs_arr])
            param_list = []
            for param_name in param_names:
                param_list.append(getattr(obj, f'_{param_name}_array'))
            mapper_list = []
            for param_name in param_names:
                # Tuple mapper is a list because of its complex data type
                mapper_list.append(getattr(obj, f'_{param_name}_mapper')[col_idxs_arr])

            return obj.__class__(
                new_wrapper,
                input_list,
                input_mapper,
                output_list,
                param_list,
                mapper_list,
                obj.short_name,
                obj.level_names
            )

        ParamIndexer = ParamIndexerFactory(param_names + (['tuple'] if len(param_names) > 1 else []))
        CustomIndicator = type(self.class_name, (PandasIndexer, ParamIndexer, Configured), {})
        CustomIndicator.__module__ = self.module_name
        CustomIndicator.__doc__ = self.class_docstring

        # Create read-only properties
        prop = property(lambda _self: _self._short_name)
        prop.__doc__ = "Name of the indicator (read-only)."
        setattr(CustomIndicator, 'short_name', prop)

        prop = property(lambda _self: _self._level_names)
        prop.__doc__ = "Column level names corresponding to each parameter (read-only)."
        setattr(CustomIndicator, 'level_names', prop)

        prop = classproperty(lambda _self: input_names)
        prop.__doc__ = "Names of the input time series (read-only)."
        setattr(CustomIndicator, 'input_names', prop)

        prop = classproperty(lambda _self: param_names)
        prop.__doc__ = "Names of the parameters (read-only)."
        setattr(CustomIndicator, 'param_names', prop)

        prop = classproperty(lambda _self: output_names)
        prop.__doc__ = "Names of the output time series (read-only)."
        setattr(CustomIndicator, 'output_names', prop)

        prop = classproperty(lambda _self: output_flags)
        prop.__doc__ = "Dictionary of output flags (read-only)."
        setattr(CustomIndicator, 'output_flags', prop)

        for param_name in param_names:
            prop = property(lambda _self, param_name=param_name: getattr(_self, f'_{param_name}_array'))
            prop.__doc__ = f"Array of `{param_name}` combinations (read-only)."
            setattr(CustomIndicator, f'{param_name}_array', prop)

        for input_name in input_names:
            def input_prop(_self, input_name=input_name):
                """Input time series (read-only).

                Will broadcast to match the shape of outputs."""
                old_input = reshape_fns.to_2d(getattr(_self, '_' + input_name), raw=True)
                input_mapper = getattr(_self, '_input_mapper')
                if input_mapper is None:
                    return _self.wrapper.wrap(old_input)
                return _self.wrapper.wrap(old_input[:, input_mapper])

            input_prop.__name__ = input_name
            setattr(CustomIndicator, input_name, cached_property(input_prop))

        for output_name in output_names:
            def output_prop(_self, output_name=output_name):
                """Output time series (read-only)."""
                return _self.wrapper.wrap(getattr(_self, '_' + output_name))

            output_prop.__name__ = output_name
            if output_name in output_flags:
                _output_flags = output_flags[output_name]
                if isinstance(_output_flags, (tuple, list)):
                    _output_flags = ', '.join(_output_flags)
                output_prop.__doc__ += "\n\n" + _output_flags
            setattr(CustomIndicator, output_name, property(output_prop))

        # Add __init__ method
        def __init__(_self, wrapper, input_list, input_mapper, output_list, param_list,
                     mapper_list, short_name, level_names):
            perform_init_checks(
                wrapper,
                input_list,
                input_mapper,
                output_list,
                param_list,
                mapper_list,
                short_name,
                level_names
            )
            Configured.__init__(
                _self,
                wrapper=wrapper,
                input_list=input_list,
                input_mapper=input_mapper,
                output_list=output_list,
                param_list=param_list,
                mapper_list=mapper_list,
                short_name=short_name,
                level_names=level_names
            )

            _self.wrapper = wrapper
            for i, ts_name in enumerate(input_names):
                setattr(_self, f'_{ts_name}', input_list[i])
            setattr(_self, '_input_mapper', input_mapper)
            for i, output_name in enumerate(output_names):
                setattr(_self, f'_{output_name}', output_list[i])
            for i, param_name in enumerate(param_names):
                setattr(_self, f'_{param_name}_array', param_list[i])
                setattr(_self, f'_{param_name}_mapper', mapper_list[i])
            if len(param_names) > 1:
                tuple_mapper = list(zip(*list(mapper_list)))
                setattr(_self, '_tuple_mapper', tuple_mapper)
            else:
                tuple_mapper = None
            setattr(_self, '_short_name', short_name)
            setattr(_self, '_level_names', level_names)

            # Initialize indexers
            PandasIndexer.__init__(_self, indexing_func)
            mapper_sr_list = []
            for i, m in enumerate(mapper_list):
                mapper_sr_list.append(pd.Series(m, index=wrapper.columns))
            if tuple_mapper is not None:
                mapper_sr_list.append(pd.Series(tuple_mapper, index=wrapper.columns))
            ParamIndexer.__init__(
                _self, mapper_sr_list, indexing_func,
                level_names=[*level_names, tuple(level_names)]
            )

        setattr(CustomIndicator, '__init__', __init__)

        # Add private run method
        def_run_kwargs = dict(
            short_name=short_name,
            hide_params=[],
            hide_default=True
        )

        @classmethod
        def _run(cls,
                 *args,
                 short_name=def_run_kwargs['short_name'],
                 hide_params=def_run_kwargs['hide_params'],
                 hide_default=def_run_kwargs['hide_default'],
                 pipeline_kwargs=pipeline_kwargs,
                 **kwargs):
            if len(args) < len(input_names) + len(param_names):
                nmissed = len(input_names) + len(param_names) - len(args)
                raise ValueError(f"Missing {nmissed} required positional arguments")
            args = list(args)
            input_list = args[:len(input_names)]
            param_list = args[len(input_names):len(input_names) + len(param_names)]
            level_names = []
            hide_levels = []
            for i, pn in enumerate(param_names):
                level_names.append(short_name + '_' + pn)
                if pn in hide_params or \
                        (hide_default and pn in param_defaults and
                         np.all(np.array(param_list[i]) == param_defaults[pn])):
                    hide_levels.append(short_name + '_' + pn)
            level_names = list(level_names)
            custom_func_args = args[len(input_names) + len(param_names):]
            kwargs = {**pipeline_kwargs, **kwargs}  # overwrite default pipeline kwargs
            results = run_pipeline(
                input_list,
                param_list,
                level_names,
                len(output_names),
                custom_func,
                *custom_func_args,
                hide_levels=hide_levels,
                param_settings=[param_settings.get(pn, {}) for pn in param_names],
                **kwargs
            )
            if kwargs.get('return_raw', False) or kwargs.get('return_cache', False):
                return results
            wrapper, new_input_list, input_mapper, output_list, new_param_list, mapper_list, other_list = results
            obj = cls(
                wrapper,
                new_input_list,
                input_mapper,
                output_list,
                new_param_list,
                mapper_list,
                short_name,
                level_names
            )
            if len(other_list) > 0:
                return (obj, *tuple(other_list))
            return obj

        setattr(CustomIndicator, '_run', _run)

        # Add public run method
        # Create function dynamically to provide user with a proper signature
        def compile_run_function(func_name, docstring, default_kwargs):
            pos_params_lst = param_names[:len(param_names) - len(param_defaults)]
            kw_params_lst = [
                (param_names[i], param_defaults[param_names[i]])
                for i in range(-len(param_defaults), 0)
            ]
            kw_params_str_lst = ['{}={}'.format(k, v) for k, v in kw_params_lst]
            kw_names_str_lst = [param_names[i] for i in range(-len(param_defaults), 0)]
            def_kw_str_lst = ['{}={}'.format(k, k) for k, v in default_kwargs.items()]
            first_arg = input_names + pos_params_lst + kw_params_str_lst
            first_arg = ', '.join(first_arg) + ', ' if len(first_arg) > 0 else ''
            second_arg = def_kw_str_lst
            second_arg = ', '.join(second_arg) + ', ' if len(second_arg) > 0 else ''
            fourth_arg = input_names + pos_params_lst + kw_names_str_lst
            fourth_arg = ', '.join(fourth_arg) + ', ' if len(fourth_arg) > 0 else ''
            func_str = "@classmethod\n" \
                "def {0}(cls, {1}*args, {2}**kwargs):\n" \
                "    \"\"\"{3}\"\"\"\n" \
                "    return cls._{0}({4}*args, {2}**kwargs)".format(
                    func_name, first_arg, second_arg, docstring, fourth_arg
            )
            scope = default_kwargs
            filename = inspect.getfile(lambda: None)
            code = compile(func_str, filename, 'single')
            exec(code, scope)
            return scope[func_name]

        def create_run_docstring(docstring_func):
            as_code = lambda x: list(map(lambda y: f'`{y}`', x))
            if len(input_names) > 0:
                in_ts_str = ', '.join(as_code(input_names)[:-2] + [' and '.join(as_code(input_names[-2:]))])
                in_ts_str = 'input time series ' + in_ts_str
            else:
                in_ts_str = 'no input time series'
            if len(param_names) > 0:
                param_str = ', '.join(as_code(param_names[:-2]) + [' and '.join(as_code(param_names[-2:]))])
                param_str = 'parameters ' + param_str
            else:
                param_str = 'no parameters'
            out_ts_str = ', '.join(as_code(output_names[:-2]) + [' and '.join(as_code(output_names[-2:]))])
            return docstring_func(self.class_name, in_ts_str, param_str, out_ts_str)

        def run_docstring_func(class_name, in_ts_str, param_str, out_ts_str):
            return """Run the {0} indicator using {1}, and {2}, to
                produce output time series {3}.
    
                Pass a list of parameter names `hide_params` to hide their column levels.
                Set `hide_default` to False to show column levels of parameters with the default value passed.
                Keyword arguments are passed to `vectorbt.indicators.factory.run_pipeline`.""".format(
                class_name,
                in_ts_str,
                param_str,
                out_ts_str,
            )

        run_docstring = create_run_docstring(run_docstring_func)
        run = compile_run_function('run', run_docstring, def_run_kwargs)
        setattr(CustomIndicator, 'run', run)

        if len(param_names) > 0:
            # Add private run_combs method
            def_run_combs_kwargs = dict(
                r=2,
                param_product=False,
                comb_func=itertools.combinations,
                speed_up=True,
                short_names=None
            )

            @classmethod
            def _run_combs(cls,
                           *args,
                           r=def_run_combs_kwargs['r'],
                           param_product=def_run_combs_kwargs['param_product'],
                           comb_func=def_run_combs_kwargs['comb_func'],
                           speed_up=def_run_combs_kwargs['speed_up'],
                           short_names=def_run_combs_kwargs['short_names'],
                           **kwargs):
                if short_names is None:
                    short_names = [f'{short_name}_{str(i + 1)}' for i in range(r)]
                input_list = args[:len(input_names)]
                param_list = args[len(input_names):len(input_names) + len(param_names)]
                if param_product:
                    param_list = create_param_product(param_list)
                else:
                    param_list = broadcast_params(param_list)
                if not isinstance(param_list, (tuple, list)):
                    param_list = [param_list]
                custom_func_args = args[len(input_names) + len(param_names):]
                if speed_up:
                    raw_results = cls.run(
                        *input_list,
                        *param_list,
                        *custom_func_args,
                        return_raw=True,
                        **kwargs
                    )
                    kwargs['use_raw'] = raw_results
                instances = []
                if comb_func == itertools.product:
                    param_lists = zip(*comb_func(zip(*param_list), repeat=r))
                else:
                    param_lists = zip(*comb_func(zip(*param_list), r))
                for i, param_list in enumerate(param_lists):
                    instances.append(cls.run(
                        *input_list,
                        *zip(*param_list),
                        *custom_func_args,
                        short_name=short_names[i],
                        **kwargs
                    ))
                return tuple(instances)

            setattr(CustomIndicator, '_run_combs', _run_combs)

            # Add public run_combs method
            def run_combs_docstring_func(class_name, in_ts_str, param_str, out_ts_str):
                return """Create a combination of multiple {0} indicators using function `comb_func`. 
                    Run each indicator using input time series {1}, and {2}, to produce output time series {3}.
                    
                    `comb_func` must accept an iterable of parameter tuples and `r`. Also accepts all
                    combinatoric iterators from itertools such as `itertools.combinations`.
    
                    Pass `r` to specify how many indicators to run. Pass `short_names` to specify the
                    short name for each indicator. Set `speed_up` to True to first compute raw outputs 
                    for all parameters, and then use them to build each indicator (faster).
                    Keyword arguments are passed to `{0}.run`.""".format(
                    class_name,
                    in_ts_str,
                    param_str,
                    out_ts_str,
                )

            run_combs_docstring = create_run_docstring(run_combs_docstring_func)
            run_combs = compile_run_function('run_combs', run_combs_docstring, def_run_combs_kwargs)
            setattr(CustomIndicator, 'run_combs', run_combs)

        # Add user-defined outputs
        for prop_name, prop in custom_output_funcs.items():
            if prop.__doc__ is None:
                prop.__doc__ = f"""Custom property."""
            if not isinstance(prop, (property, cached_property)):
                prop.__name__ = prop_name
                prop = cached_property(prop)
            setattr(CustomIndicator, prop_name, prop)

        # Add comparison methods for all inputs, outputs, and user-defined properties
        comparison_attrs = set(input_names + output_names + list(custom_output_funcs.keys()))
        for attr in comparison_attrs:
            def assign_comparison_method(func_name, compare_func, attr=attr):
                def comparison_method(_self, other, crossed=False, wait=0, level_name=None, after_false=True, **kwargs):
                    if isinstance(other, _self.__class__):
                        other = getattr(other, attr)
                    if level_name is None:
                        if attr == _self.short_name:
                            level_name = f'{_self.short_name}_{func_name}'
                        else:
                            level_name = f'{_self.short_name}_{attr}_{func_name}'
                    out = compare(getattr(_self, attr), other, compare_func, level_name=level_name, **kwargs)
                    if crossed:
                        return out.vbt.signals.nst(wait + 1, after_false=after_false)
                    return out

                comparison_method.__qualname__ = f'{CustomIndicator.__name__}.{attr}_{func_name}'
                comparison_method.__doc__ = f"""Return True for each element where `{attr}` is {func_name} `other`. 

                Set `crossed` to True to return the first True after crossover. Specify `wait` to return 
                True only when `{attr}` is {func_name} for a number of time steps in a row after crossover.

                See `vectorbt.indicators.factory.compare`."""
                setattr(CustomIndicator, f'{attr}_{func_name}', comparison_method)

            assign_comparison_method('above', np.greater)
            assign_comparison_method('below', np.less)
            assign_comparison_method('equal', np.equal)

        return CustomIndicator

    def from_apply_func(self, apply_func, cache_func=None, **kwargs):
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
            apply_func (callable): A function that takes broadcasted time series arrays corresponding 
                to `input_names`, single parameter selection corresponding to `param_names`, and other
                arguments and keyword arguments, and returns outputs corresponding to `output_names`.
                Can be Numba-compiled.
            cache_func (callable): A caching function to preprocess data beforehand.
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
            ...     input_names=['ts1', 'ts2'],
            ...     param_names=['p1', 'p2'],
            ...     output_names=['o1', 'o2']
            ... ).from_apply_func(apply_func_nb)

            >>> myInd = MyInd.run(price, price * 2, [1, 2], [3, 4], 100)
            >>> myInd.o1
            custom_p1              1             2
            custom_p2              3             4
                            a      b      a      b
            2020-01-01  101.0  105.0  102.0  110.0
            2020-01-02  102.0  104.0  104.0  108.0
            2020-01-03  103.0  103.0  106.0  106.0
            2020-01-04  104.0  102.0  108.0  104.0
            2020-01-05  105.0  101.0  110.0  102.0
            >>> myInd.o2
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
            def select_params(i, args_before, input_list, param_tuples, *args):
                # Select the next tuple of parameters
                return apply_func(*args_before, *input_list, *param_tuples[i], *args)

        else:
            if num_outputs > 1:
                apply_and_concat_func = combine_fns.apply_and_concat_multiple
            else:
                apply_and_concat_func = combine_fns.apply_and_concat_one

            def select_params(i, args_before, input_list, param_tuples, *args, **_kwargs):
                # Select the next tuple of parameters
                return apply_func(*args_before, *input_list, *param_tuples[i], *args, **_kwargs)

        def custom_func(input_list, param_list, *args, return_cache=False, use_cache=None,
                        input_shape=None, is_2d=None, **_kwargs):

            n_params = len(param_list[0]) if len(param_list) > 0 else 1
            input_list = tuple(input_list)
            param_tuples = tuple(zip(*param_list))
            if len(param_list) == 0:
                param_tuples = ((),) * n_params
            args_before = ()
            if input_shape is not None:
                args_before += (input_shape,)
            args_after = ()
            if is_2d is not None:
                args_after += (is_2d,)

            # Caching
            cache = use_cache
            if cache is None and cache_func is not None:
                cache = cache_func(*args_before, *input_list, *param_list, *args, *args_after, **_kwargs)
            if return_cache:
                return cache
            if cache is None:
                cache = ()
            if not isinstance(cache, (tuple, list, List)):
                cache = (cache,)

            return apply_and_concat_func(
                n_params,
                select_params,
                args_before,
                input_list,
                param_tuples,
                *args,
                *args_after,
                *cache,
                **_kwargs)

        return self.from_custom_func(custom_func, pass_lists=True, **kwargs)

    @classmethod
    def from_talib(cls, func_name, module_name=__name__, **kwargs):
        """Build indicator class around a TA-Lib function.

        Requires [TA-Lib](https://github.com/mrjbq7/ta-lib) installed.

        For input, parameter and output names, see [docs](https://github.com/mrjbq7/ta-lib/blob/master/docs/index.md).

        Args:
            func_name (str): Function name.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_apply_func`.
        Returns:
            CustomIndicator
        Example:
            ```python-repl
            >>> SMA = vbt.IndicatorFactory.from_talib('SMA')

            >>> sma = SMA.run(price, timeperiod=[2, 3])
            >>> sma.real
            sma_timeperiod         2         3
                              a    b    a    b
            2020-01-01      NaN  NaN  NaN  NaN
            2020-01-02      1.5  4.5  NaN  NaN
            2020-01-03      2.5  3.5  2.0  4.0
            2020-01-04      3.5  2.5  3.0  3.0
            2020-01-05      4.5  1.5  4.0  2.0
            ```
        """
        import talib
        from talib import abstract

        func = getattr(talib, func_name)
        info = abstract.Function(func_name)._Function__info
        input_names = []
        for in_names in info['input_names'].values():
            if isinstance(in_names, (list, tuple)):
                input_names.extend(list(in_names))
            else:
                input_names.append(in_names)

        def custom_func(*args):
            # TA-Lib functions can only process 1-dim arrays
            # TODO: Find ways to call talib from within Numba
            ts = args[:len(input_names)]
            params = args[len(input_names):]
            if len(params) == 0:
                # No parameters
                outputs = []
                for col in range(ts[0].shape[1]):
                    outputs.append(func(*map(lambda x: x[:, col], ts)))
                idxs = slice(None, None)
            else:
                outputs = []
                param_tuples = list(zip(*params))
                unique_param_tuples = list(OrderedDict.fromkeys(param_tuples).keys())
                for param_tuple in unique_param_tuples:
                    for col in range(ts[0].shape[1]):
                        outputs.append(func(
                            *map(lambda x: x[:, col], ts),
                            *param_tuple
                        ))
                if len(param_tuples) == len(unique_param_tuples):
                    idxs = slice(None, None)
                else:
                    idxs = reindex_outputs(param_tuples, unique_param_tuples, ts[0].shape[1])
            if isinstance(outputs[0], tuple):  # multiple outputs
                outputs = list(zip(*outputs))
                return list(map(lambda x: np.column_stack(x)[:, idxs], outputs))
            return np.column_stack(outputs)[:, idxs]

        TALibIndicator = cls(
            class_name=info['name'],
            class_docstring="{}, {}".format(info['display_name'], info['group']),
            module_name=module_name,
            short_name=info['name'].lower(),
            input_names=input_names,
            param_names=list(info['parameters'].keys()),
            param_defaults=info['parameters'],
            output_names=info['output_names'],
            output_flags=info['output_flags']
        ).from_custom_func(custom_func, **kwargs)
        return TALibIndicator
