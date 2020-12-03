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

The main method to run an indicator is `run` that accepts 1) input time series, 2) parameters,
and 3) other arguments that are accepted by the calculation function.

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
arbitrary array-like object, given their shapes can be broadcast together. You can also compare them
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
from vectorbt.utils.config import merge_dicts, Configured
from vectorbt.utils.random import set_seed
from vectorbt.base import index_fns, reshape_fns, combine_fns
from vectorbt.base.indexing import ParamIndexerFactory
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping


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

    ## Example

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


def prepare_params(param_list, param_settings, input_shape=None, to_2d=False):
    """Prepare parameters."""
    new_param_list = []
    for i, params in enumerate(param_list):
        _param_settings = param_settings if isinstance(param_settings, dict) else param_settings[i]
        checks.assert_dict_valid(_param_settings, [['array_like', 'bc_to_input', 'broadcast_kwargs']])
        is_array_like = _param_settings.get('array_like', False)
        bc_to_input = _param_settings.get('bc_to_input', False)
        broadcast_kwargs = _param_settings.get('broadcast_kwargs', dict(require_kwargs=dict(requirements='W')))

        if is_array_like:
            # Array is treated as one value
            check_against = (list, tuple, List)
        else:
            # Array is treated as multiple values
            check_against = (list, tuple, List, np.ndarray)
        if isinstance(params, check_against):
            new_params = tuple(params)
        else:
            new_params = (params,)
        if bc_to_input is not False:
            # Broadcast to input or its axis
            if input_shape is None:
                raise ValueError("Cannot broadcast to input if input shape is unknown")
            if bc_to_input is True:
                to_shape = input_shape
            else:
                checks.assert_in(bc_to_input, (0, 1))
                # Note that input_shape can be 1D
                if bc_to_input == 0:
                    to_shape = input_shape[0]
                else:
                    to_shape = input_shape[1] if len(input_shape) > 1 else (1,)
            _new_params = reshape_fns.broadcast(
                *new_params,
                to_shape=to_shape,
                **broadcast_kwargs
            )
            if len(new_params) == 1:
                _new_params = (_new_params,)
            if to_2d and bc_to_input is True:
                # If inputs are meant to reshape to 2D, do the same to parameters
                # But only to those that fully resemble inputs (= not raw)
                __new_params = list(_new_params)
                for j, param in enumerate(__new_params):
                    keep_raw = broadcast_kwargs.get('keep_raw', False)
                    if keep_raw is False or (isinstance(keep_raw, (tuple, list)) and not keep_raw[j]):
                        __new_params[j] = reshape_fns.to_2d(param)
                new_params = tuple(__new_params)
            else:
                new_params = _new_params
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
        num_ret_outputs,
        custom_func,
        *args,
        input_shape=None,
        input_index=None,
        input_columns=None,
        input_list=None,
        in_output_list=None,
        in_output_settings=None,
        broadcast_kwargs=None,
        param_list=None,
        param_product=False,
        param_settings=None,
        level_names=None,
        to_2d=True,
        pass_lists=False,
        forward_input_shape=None,
        forward_flex_2d=False,
        hide_levels=None,
        return_raw=False,
        use_raw=None,
        wrapper_kwargs=None,
        seed=None,
        **kwargs):
    """A pipeline for calculating an indicator, used by `IndicatorFactory`.

    Args:
        num_ret_outputs (int): The number of output arrays returned by `custom_func`.
        custom_func (callable): A custom calculation function.

            See `IndicatorFactory.from_custom_func`.
        *args: Arguments passed to the `custom_func`.
        input_shape (tuple): Shape to broadcast each input to.

            Can be passed to `custom_func`. See `forward_input_shape`.
        input_index (any): Sets index of each input.

            Can be used to label index if no inputs passed.
        input_columns (any): Sets columns of each input.

            Can be used to label columns if no inputs passed.
        input_list (list of array_like): A list of time series objects.
        in_output_list (list of array_like): A list of in-place outputs.
            
            If an array should be generated, pass None.
        in_output_settings (dict or list of dict): Settings corresponding to each in-place output.

            Following keys are accepted:

            * `dtype`: Create this array using this data type and `np.empty`. Default is None.
        broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`
            to broadcast inputs.
        param_list (list of array_like): A list of parameters.

            Each element is either an array-like object or a single value of any type.
        param_product (bool): Whether to build a Cartesian product out of all parameters.
        param_settings (dict or list of dict): Settings corresponding to each parameter.

            Following keys are accepted:

            * `array_like`: If array-like object was passed, it will be considered as a single value.
                To treat it as multiple values, pack it into a list.
            * `bc_to_input`: Whether to broadcast parameter to input size. You can also broadcast
                parameter to an axis by passing an integer.
            * `broadcast_kwargs`: Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
        level_names (list of str): A list of column level names corresponding to each parameter.

            Should have the same length as `param_list`.
        to_2d (bool): Whether to reshape inputs to 2-dim arrays, otherwise keep as-is.
        pass_lists (bool): Whether to pass inputs and parameters to `custom_func` as lists.

            If `custom_func` is Numba-compiled, passes tuples.
        forward_input_shape (bool): Whether to pass `input_shape` to `custom_func` as keyword argument.

            If None and no inputs passed, passes `input_shape` automatically.
        forward_flex_2d (bool): Whether to pass `flex_2d` to `custom_func` as keyword argument.
        hide_levels (list): A list of parameter levels to hide.
        return_raw (bool): Whether to return raw output without post-processing and hashed parameter tuples.
        use_raw (bool): Takes the raw results and uses them instead of running `custom_func`.
        wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.
        seed (int): Set seed to make output deterministic.
        **kwargs: Keyword arguments passed to the `custom_func`.

            Some common arguments include `return_cache` to return cache and `use_cache` to use cache.
            Those are only applicable to `custom_func` that supports it (`custom_func` created using
            `IndicatorFactory.from_apply_func` are supported by default).
            
    Returns:
        Array wrapper, list of inputs (`np.ndarray`), input mapper (`np.ndarray`), list of outputs
        (`np.ndarray`), list of parameter arrays (`np.ndarray`), list of parameter mappers (`np.ndarray`),
        list of outputs that are outside of `num_ret_outputs`.

    ## Explanation

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
    if input_list is None:
        input_list = []
    if in_output_list is None:
        in_output_list = []
    if param_list is None:
        param_list = []
    if level_names is None:
        level_names = []
    if broadcast_kwargs is None:
        broadcast_kwargs = {}
    if param_settings is None:
        param_settings = {}
    if in_output_settings is None:
        in_output_settings = {}
    if hide_levels is None:
        hide_levels = []
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    if forward_input_shape is None:
        if input_shape is not None and len(input_list) == 0:
            forward_input_shape = True
        else:
            forward_input_shape = False

    in_output_idxs = [i for i, x in enumerate(in_output_list) if x is not None]
    if len(in_output_idxs) > 0:
        # In-place outputs should broadcast together with inputs
        input_list += [in_output_list[i] for i in in_output_idxs]
    if len(input_list) > 0:
        # Broadcast time series
        if input_index is None:
            input_index = 'default'
        if input_columns is None:
            input_columns = 'default'
        broadcast_kwargs = merge_dicts(dict(
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
    bc_in_output_list = []
    if len(in_output_idxs) > 0:
        bc_in_output_list = input_list[-len(in_output_idxs):]
        input_list = input_list[:-len(in_output_idxs)]

    # Prepare parameters
    # NOTE: input_shape instead of input_shape_passed since parameters should
    # broadcast by the same rules as inputs
    param_list = prepare_params(param_list, param_settings, input_shape=input_shape, to_2d=to_2d)
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

    # Reshape inputs
    input_list_passed = input_list
    input_shape_passed = input_shape
    if to_2d:
        input_list_passed = list(map(reshape_fns.to_2d, input_list))
        if input_shape is not None:
            input_shape_passed = input_shape if len(input_shape) > 1 else (input_shape[0], 1)

    # Reshape in-place outputs
    in_output_list_passed = []
    j = 0
    for i in range(len(in_output_list)):
        if i in in_output_idxs:
            in_output = bc_in_output_list[j]
            in_output = reshape_fns.tile(in_output, n_param_values, axis=1)
            j += 1
        else:
            _in_output_settings = in_output_settings if isinstance(in_output_settings, dict) else in_output_settings[i]
            checks.assert_dict_valid(_in_output_settings, [['dtype']])
            dtype = _in_output_settings.get('dtype', None)
            in_output_shape = (input_shape_passed[0], input_shape_passed[1] * n_param_values)
            in_output = np.empty(in_output_shape, dtype=dtype)
        in_output_list[i] = in_output
        in_output_tuple = ()
        for i in range(n_param_values):
            in_output_tuple += (in_output[:, i * input_shape_passed[1]: (i + 1) * input_shape_passed[1]],)
        in_output_list_passed.append(in_output_tuple)

    # Get raw results
    if use_raw is not None:
        # Use raw results of previous run to build outputs
        output_list, param_map, n_input_cols, other_list = use_raw
        idxs = reindex_outputs(list(zip(*param_list)), param_map, n_input_cols)
        output_list = [output[:, idxs] for output in output_list]
    else:
        # Prepare arguments
        func_kwargs = {}
        if forward_input_shape:
            func_kwargs['input_shape'] = input_shape_passed
        if forward_flex_2d:
            if input_shape is None:
                raise ValueError("Cannot determine flex_2d without inputs")
            func_kwargs['flex_2d'] = len(input_shape) == 2
        func_kwargs = merge_dicts(func_kwargs, kwargs)

        # Set seed
        if seed is not None:
            set_seed(seed)

        # Run the function
        if pass_lists:
            if checks.is_numba_func(custom_func):
                output = custom_func(
                    tuple(input_list_passed),
                    tuple(in_output_list_passed),
                    tuple(param_list),
                    *args, **func_kwargs
                )
            else:
                output = custom_func(
                    input_list_passed,
                    in_output_list_passed,
                    param_list,
                    *args, **func_kwargs
                )
        else:
            output = custom_func(
                *input_list_passed,
                *in_output_list_passed,
                *param_list, 
                *args, **func_kwargs
            )

        # Return cache
        if kwargs.get('return_cache', False):
            return output

        # Post-process results
        if isinstance(output, (tuple, list, List)):
            output_list = list(output)
        else:
            output_list = [output]
        # Other outputs should be returned without post-processing (for example cache_dict)
        if len(output_list) > num_ret_outputs:
            other_list = output_list[num_ret_outputs:]
        else:
            other_list = []
        # Process only the num_ret_outputs outputs
        output_list = output_list[:num_ret_outputs]
        if len(output_list) != num_ret_outputs:
            raise ValueError("Number of returned outputs other than expected")
        output_list = list(map(reshape_fns.to_2d, output_list))
        # In-place outputs are treated as outputs from here
        output_list += in_output_list

        # Return raw results if needed
        param_map = list(zip(*param_list))
        n_input_cols = output_list[0].shape[1] // n_param_values
        if return_raw:
            return output_list, param_map, n_input_cols, other_list

    # Update shape info if no inputs
    if input_shape is None:
        if n_input_cols == 1:
            input_shape = (output_list[0].shape[0],)
        else:
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


class Default:
    """Class for wrapping default values."""
    def __repr__(self):
        return self.value.__repr__()

    def __init__(self, value):
        self.value = value


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


def combine_objs(obj, other, combine_func, multiple=False, level_name=None, keys=None, **kwargs):
    """Combines/compares `obj` to `other`, for example, to generate signals.

    Both will be broadcast together. Set `multiple` to True to compare with multiple arguments.
    In this case, a new column level will be created with the name `level_name`.

    See `vectorbt.base.accessors.Base_Accessor.combine_with`."""
    if multiple:
        if keys is None:
            keys = index_fns.index_from_values(other, name=level_name)
        return obj.vbt.combine_with_multiple(other, combine_func=combine_func, keys=keys, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=combine_func, **kwargs)


class IndicatorFactory:
    def __init__(self,
                 class_name='CustomIndicator',
                 class_docstring='',
                 module_name=__name__,
                 short_name='custom',
                 prepend_name=True,
                 input_names=None,
                 param_names=None,
                 in_output_names=None,
                 output_names=None,
                 output_flags=None,
                 custom_output_funcs=None,
                 attr_settings=None):
        """A factory for creating new indicators.

        Initialize `IndicatorFactory` to create a skeleton and then use a class method
        to finish building the indicator class.

        Args:
            class_name (str): Name for the created Python class.
            class_docstring (str): Docstring for the created Python class.
            module_name (str): Specify the module the class originates from.
            short_name (str): A short name of the indicator.
            prepend_name (bool): Whether to prepend `short_name` to each parameter level.
            input_names (list of str): A list of names of input time series objects.
            param_names (list of str): A list of names of parameters.
            in_output_names (list of str): A list of names of in-place output time series objects.

                An in-place output is an output that is not returned but modified in-place.
                Some advantages of such outputs include:

                1) they don't need to be returned,
                2) they can be passed between functions as easily as inputs,
                3) they can be provided with already allocated data to safe memory,
                4) if data or default value are not provided, they are created empty to safe memory.
            output_names (list of str): A list of names of output time series objects.

                !!! note
                    Must have at least one (non-in-place) output.
            output_flags (dict): A dictionary of output flags.
            custom_output_funcs (dict): A dictionary with user-defined functions that will be
                bound to the indicator class and wrapped with `@cached_property`.
            attr_settings (dict): A dictionary of settings by attribute name.

                Attributes can be `input_names`, `in_output_names`, `output_names` and `custom_output_funcs`.

                Following keys are accepted:

                * `dtype`: Data type used to determine which methods to generate around this attribute.
                    Set to None to disable. Default is `np.float_`. Can be set to instance of
                    `collections.namedtuple` acting as enumerated type; it will then create a property
                    with suffix `readable` that contains data in a string format.

        !!! note
            The `__init__` method is not used for running the indicator, for this use `run`.
            The reason for this is indexing, which requires a clean `__init__` method for creating 
            a new indicator object with newly indexed attributes.
        """
        # Check and save parameters
        self.class_name = class_name
        checks.assert_type(class_name, str)

        self.class_docstring = class_docstring
        checks.assert_type(class_docstring, str)

        self.module_name = module_name
        checks.assert_type(module_name, str)

        self.short_name = short_name
        checks.assert_type(short_name, str)

        self.prepend_name = prepend_name
        checks.assert_type(prepend_name, bool)

        if input_names is None:
            input_names = []
        checks.assert_type(input_names, (tuple, list))
        self.input_names = input_names

        if param_names is None:
            param_names = []
        checks.assert_type(param_names, (tuple, list))
        self.param_names = param_names

        if in_output_names is None:
            in_output_names = []
        checks.assert_type(in_output_names, (tuple, list))
        self.in_output_names = in_output_names

        if output_names is None:
            output_names = []
        checks.assert_type(output_names, (tuple, list))
        if len(output_names) == 0:
            raise ValueError("Must have at least one output")
        output_names += in_output_names
        self.output_names = output_names

        if output_flags is None:
            output_flags = {}
        checks.assert_type(output_flags, dict)
        if len(output_flags) > 0:
            checks.assert_dict_valid(output_flags, [output_names])
        self.output_flags = output_flags

        if custom_output_funcs is None:
            custom_output_funcs = {}
        checks.assert_type(custom_output_funcs, dict)
        self.custom_output_funcs = custom_output_funcs

        if attr_settings is None:
            attr_settings = {}
        checks.assert_type(attr_settings, dict)
        all_attr_names = input_names + output_names + list(custom_output_funcs.keys())
        if len(attr_settings) > 0:
            checks.assert_dict_valid(attr_settings, [all_attr_names])
        self.attr_settings = attr_settings

        # Set up class
        ParamIndexer = ParamIndexerFactory(param_names + (['tuple'] if len(param_names) > 1 else []))
        CustomIndicator = type(self.class_name, (Wrapping, ParamIndexer), {})
        CustomIndicator.__module__ = self.module_name
        CustomIndicator.__doc__ = self.class_docstring

        # Add indexing methods
        def _indexing_func(obj, pd_indexing_func):
            new_wrapper, idx_idxs, _, col_idxs = obj.wrapper._indexing_func_meta(pd_indexing_func)
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

            return obj.copy(
                wrapper=new_wrapper,
                input_list=input_list,
                input_mapper=input_mapper,
                output_list=output_list,
                param_list=param_list,
                mapper_list=mapper_list
            )

        setattr(CustomIndicator, '_indexing_func', _indexing_func)

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
            Wrapping.__init__(
                _self,
                wrapper,
                input_list=input_list,
                input_mapper=input_mapper,
                output_list=output_list,
                param_list=param_list,
                mapper_list=mapper_list,
                short_name=short_name,
                level_names=level_names
            )

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
            mapper_sr_list = []
            for i, m in enumerate(mapper_list):
                mapper_sr_list.append(pd.Series(m, index=wrapper.columns))
            if tuple_mapper is not None:
                mapper_sr_list.append(pd.Series(tuple_mapper, index=wrapper.columns))
            ParamIndexer.__init__(
                _self, mapper_sr_list,
                level_names=[*level_names, tuple(level_names)]
            )

        setattr(CustomIndicator, '__init__', __init__)

        # Add user-defined outputs
        for prop_name, prop in custom_output_funcs.items():
            if prop.__doc__ is None:
                prop.__doc__ = f"""Custom property."""
            if not isinstance(prop, (property, cached_property)):
                prop.__name__ = prop_name
                prop = cached_property(prop)
            setattr(CustomIndicator, prop_name, prop)

        # Add comparison & combination methods for all inputs, outputs, and user-defined properties
        for attr_name in all_attr_names:
            _attr_settings = attr_settings.get(attr_name, {})
            checks.assert_dict_valid(_attr_settings, [['dtype']])
            dtype = _attr_settings.get('dtype', np.float_)

            def _isinstance_namedtuple(obj) -> bool:
                return (
                    isinstance(obj, tuple) and
                    hasattr(obj, '_asdict') and
                    hasattr(obj, '_fields')
                )

            if _isinstance_namedtuple(dtype):
                def attr_readable(_self, attr_name=attr_name, enum=dtype):
                    if _self.wrapper.ndim == 1:
                        return getattr(_self, attr_name).map(lambda x: '' if x == -1 else enum._fields[x])
                    return getattr(_self, attr_name).applymap(lambda x: '' if x == -1 else enum._fields[x])

                attr_readable.__qualname__ = f'{CustomIndicator.__name__}.{attr_name}_readable'
                attr_readable.__doc__ = f"""{attr_name} in readable format based on enum {dtype}."""
                setattr(CustomIndicator, f'{attr_name}_readable', property(attr_readable))

            elif np.issubdtype(dtype, np.number):
                def assign_numeric_method(func_name, combine_func, attr_name=attr_name):
                    def numeric_method(_self, other, crossed=False, wait=0, after_false=True,
                                       level_name=None, prepend_name=prepend_name, **kwargs):
                        if isinstance(other, _self.__class__):
                            other = getattr(other, attr_name)
                        if level_name is None:
                            if prepend_name:
                                if attr_name == _self.short_name:
                                    level_name = f'{_self.short_name}_{func_name}'
                                else:
                                    level_name = f'{_self.short_name}_{attr_name}_{func_name}'
                            else:
                                level_name = f'{attr_name}_{func_name}'
                        out = combine_objs(
                            getattr(_self, attr_name),
                            other,
                            combine_func,
                            level_name=level_name,
                            **kwargs
                        )
                        if crossed:
                            return out.vbt.signals.nst(wait + 1, after_false=after_false)
                        return out

                    numeric_method.__qualname__ = f'{CustomIndicator.__name__}.{attr_name}_{func_name}'
                    numeric_method.__doc__ = f"""Return True for each element where `{attr_name}` is {func_name} `other`. 
    
                    Set `crossed` to True to return the first True after crossover. Specify `wait` to return 
                    True only when `{attr_name}` is {func_name} for a number of time steps in a row after crossover.
                
                    See `vectorbt.indicators.factory.combine_objs`."""
                    setattr(CustomIndicator, f'{attr_name}_{func_name}', numeric_method)

                assign_numeric_method('above', np.greater)
                assign_numeric_method('below', np.less)
                assign_numeric_method('equal', np.equal)

            elif np.issubdtype(dtype, np.bool_):
                def assign_bool_method(func_name, combine_func, attr_name=attr_name):
                    def bool_method(_self, other, level_name=None, prepend_name=prepend_name, **kwargs):
                        if isinstance(other, _self.__class__):
                            other = getattr(other, attr_name)
                        if level_name is None:
                            if prepend_name:
                                if attr_name == _self.short_name:
                                    level_name = f'{_self.short_name}_{func_name}'
                                else:
                                    level_name = f'{_self.short_name}_{attr_name}_{func_name}'
                            else:
                                level_name = f'{attr_name}_{func_name}'
                        return combine_objs(
                            getattr(_self, attr_name),
                            other,
                            combine_func,
                            level_name=level_name,
                            **kwargs
                        )

                    bool_method.__qualname__ = f'{CustomIndicator.__name__}.{attr_name}_{func_name}'
                    bool_method.__doc__ = f"""Return `{attr_name} {func_name.upper()} other`. 

                        See `vectorbt.indicators.factory.combine_objs`."""
                    setattr(CustomIndicator, f'{attr_name}_{func_name}', bool_method)

                assign_bool_method('and', np.logical_and)
                assign_bool_method('or', np.logical_or)
                assign_bool_method('xor', np.logical_xor)

            self.CustomIndicator = CustomIndicator

    def from_custom_func(self,
                         custom_func,
                         param_settings=None,
                         in_output_settings=None,
                         hide_params=None,
                         hide_default=True,
                         variable_args=False,
                         keyword_only_args=False,
                         **pipeline_kwargs):
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
            custom_func (callable): A function that takes broadcast time series corresponding
                to `input_names`, broadcast in-place output time series corresponding to `in_output_names`,
                broadcast parameter arrays corresponding to `param_names`, and other arguments and
                keyword arguments, and returns outputs corresponding to `output_names` and `in_output_names`
                and other objects that are then returned with the indicator class instance.

                Can be Numba-compiled.
            param_settings (dict): A dictionary of settings by parameter name.

                See `run_pipeline` for keys.
            in_output_settings (dict): A dictionary of settings by in-place output name.

                See `run_pipeline` for keys.
            hide_params (list): Parameter names to hide column levels for.
            hide_default (bool): Whether to hide column levels of parameters with default value.
            variable_args (bool): Whether `run` and `run_combs` should use starred expression.
            keyword_only_args (bool): Whether `run` and `run_combs` should accept keyword-only arguments.
            **pipeline_kwargs: Keyword arguments passed to `run_pipeline`.

                Can be default values for `param_names` and `in_output_names`, but also custom keyword
                arguments passed to the `custom_func`.

                !!! note
                    Default parameters should be on the right in `param_names`.

        Returns:
            `CustomIndicator`, and optionally other objects that are returned by `custom_func`
            and exceed `output_names`.

        ## Example

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
        CustomIndicator = self.CustomIndicator

        short_name = self.short_name
        prepend_name = self.prepend_name
        input_names = self.input_names
        param_names = self.param_names
        output_names = self.output_names
        in_output_names = self.in_output_names

        if param_settings is None:
            param_settings = {}
        checks.assert_type(param_settings, dict)
        if in_output_settings is None:
            in_output_settings = {}
        checks.assert_type(in_output_settings, dict)
        if len(in_output_settings) > 0:
            checks.assert_dict_valid(in_output_settings, [in_output_names])
        if variable_args and keyword_only_args:
            raise ValueError("variable_args and keyword_only_args cannot be used together")

        for k, v in pipeline_kwargs.items():
            if k in param_names and not isinstance(v, Default):
                pipeline_kwargs[k] = Default(v)  # track default params
        pipeline_kwargs = merge_dicts({k: None for k in in_output_names}, pipeline_kwargs)

        # Add private run method
        def_run_kwargs = dict(
            short_name=short_name,
            hide_params=hide_params,
            hide_default=hide_default,
            **pipeline_kwargs
        )

        @classmethod
        def _run(cls, *args, **kwargs):
            _short_name = kwargs.pop('short_name', def_run_kwargs['short_name'])
            _hide_params = kwargs.pop('hide_params', def_run_kwargs['hide_params'])
            _hide_default = kwargs.pop('hide_default', def_run_kwargs['hide_default'])

            if _hide_params is None:
                _hide_params = []

            args = list(args)

            # Extract inputs
            input_list = args[:len(input_names)]
            checks.assert_len_equal(input_list, input_names)
            args = args[len(input_names):]

            # Extract params
            param_list = args[:len(param_names)]
            checks.assert_len_equal(param_list, param_names)
            args = args[len(param_names):]

            # Extract in-place outputs
            in_output_list = args[:len(in_output_names)]
            checks.assert_len_equal(in_output_list, in_output_names)
            args = args[len(in_output_names):]
            if not variable_args and len(args) > 0:
                raise TypeError("Variable length arguments are not supported by this function "
                                "(variable_args is set to False)")

            # Prepare column levels
            level_names = []
            hide_levels = []
            for i, pname in enumerate(param_names):
                level_name = _short_name + '_' + pname if prepend_name else pname
                level_names.append(level_name)
                if pname in _hide_params or (_hide_default and isinstance(param_list[i], Default)):
                    hide_levels.append(level_name)
            level_names = list(level_names)
            param_list = [params.value if isinstance(params, Default) else params for params in param_list]

            # Run the pipeline
            results = run_pipeline(
                len(output_names) - len(in_output_names),  # number of returned outputs
                custom_func,
                *args,
                input_list=input_list,
                in_output_list=in_output_list,
                param_list=param_list,
                level_names=level_names,
                hide_levels=hide_levels,
                param_settings=[param_settings.get(n, {}) for n in param_names],
                in_output_settings=[in_output_settings.get(n, {}) for n in in_output_names],
                **kwargs
            )

            # Return the raw result if any of the flags are set
            if kwargs.get('return_raw', False) or kwargs.get('return_cache', False):
                return results

            # Unpack the result
            wrapper, new_input_list, input_mapper, output_list, new_param_list, mapper_list, other_list = results

            # Create a new instance
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
            pos_names = []
            main_kw_names = []
            other_kw_names = []
            for k in input_names + param_names:
                if k in default_kwargs:
                    main_kw_names.append(k)
                else:
                    pos_names.append(k)
            main_kw_names.extend(in_output_names)  # in_output_names are keyword-only
            for k, v in default_kwargs.items():
                if k not in pos_names and k not in main_kw_names:
                    other_kw_names.append(k)

            _0 = func_name
            _1 = '*, ' if keyword_only_args else ''
            _2 = pos_names
            _2 = ', '.join(_2) + ', ' if len(_2) > 0 else ''
            _3 = '*args, ' if variable_args else ''
            _4 = ['{}={}'.format(k, k) for k in main_kw_names + other_kw_names]
            _4 = ', '.join(_4) + ', ' if len(_4) > 0 else ''
            _5 = docstring
            _6 = input_names + param_names + in_output_names
            _6 = ', '.join(_6) + ', ' if len(_6) > 0 else ''
            _7 = ['{}={}'.format(k, k) for k in other_kw_names]
            _7 = ', '.join(_7) + ', ' if len(_7) > 0 else ''
            func_str = "@classmethod\n" \
                "def {0}(cls, {1}{2}{3}{4}**kwargs):\n" \
                "    \"\"\"{5}\"\"\"\n" \
                "    return cls._{0}({6}{3}{7}**kwargs)".format(
                _0, _1, _2, _3, _4, _5, _6, _7
            )
            scope = {**dict(Default=Default), **default_kwargs}
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
                
                Other keyword arguments are passed to `vectorbt.indicators.factory.run_pipeline`.""".format(
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
                short_names=None,
                hide_params=hide_params,
                hide_default=hide_default,
                **pipeline_kwargs
            )

            @classmethod
            def _run_combs(cls, *args, **kwargs):
                _r = kwargs.pop('r', def_run_combs_kwargs['r'])
                _param_product = kwargs.pop('param_product', def_run_combs_kwargs['param_product'])
                _comb_func = kwargs.pop('comb_func', def_run_combs_kwargs['comb_func'])
                _speed_up = kwargs.pop('speed_up', def_run_combs_kwargs['speed_up'])
                _short_names = kwargs.pop('short_names', def_run_combs_kwargs['short_names'])
                _hide_params = kwargs.pop('hide_params', def_run_kwargs['hide_params'])
                _hide_default = kwargs.pop('hide_default', def_run_kwargs['hide_default'])

                if _hide_params is None:
                    _hide_params = []
                if _short_names is None:
                    _short_names = [f'{short_name}_{str(i + 1)}' for i in range(_r)]

                args = list(args)

                # Extract inputs
                input_list = args[:len(input_names)]
                checks.assert_len_equal(input_list, input_names)
                args = args[len(input_names):]

                # Extract params
                param_list = args[:len(param_names)]
                for i, pname in enumerate(param_names):
                    if _hide_default and isinstance(param_list[i], Default):
                        if pname not in _hide_params:
                            _hide_params.append(pname)
                        param_list[i] = param_list[i].value
                checks.assert_len_equal(param_list, param_names)
                args = args[len(param_names):]
                if not variable_args and len(args) > 0:
                    raise TypeError("Variable length arguments are not supported by this function "
                                    "(variable_args is set to False)")

                # Prepare params
                param_settings_list = [param_settings.get(n, {}) for n in param_names]
                for i in range(len(param_list)):
                    is_array_like = param_settings_list[i].get('array_like', False)
                    if is_array_like:
                        # Array is treated as one value
                        check_against = (list, tuple, List)
                    else:
                        # Array is treated as multiple values
                        check_against = (list, tuple, List, np.ndarray)
                    if isinstance(param_list[i], check_against):
                        param_list[i] = tuple(param_list[i])
                    else:
                        param_list[i] = (param_list[i],)
                if _param_product:
                    param_list = create_param_product(param_list)
                else:
                    param_list = broadcast_params(param_list)
                if not isinstance(param_list, (tuple, list)):
                    param_list = [param_list]

                # Speed up by pre-calculating raw outputs
                if _speed_up:
                    raw_results = cls._run(
                        *input_list,
                        *param_list,
                        *args,
                        return_raw=True,
                        **kwargs
                    )
                    kwargs['use_raw'] = raw_results  # use them next time

                # Generate indicator instances
                instances = []
                if _comb_func == itertools.product:
                    param_lists = zip(*_comb_func(zip(*param_list), repeat=_r))
                else:
                    param_lists = zip(*_comb_func(zip(*param_list), _r))
                for i, param_list in enumerate(param_lists):
                    instances.append(cls._run(
                        *input_list,
                        *zip(*param_list),
                        *args,
                        short_name=_short_names[i],
                        hide_params=_hide_params,
                        hide_default=_hide_default,
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
                    
                    Other keyword arguments are passed to `{0}.run`.""".format(
                    class_name,
                    in_ts_str,
                    param_str,
                    out_ts_str,
                )

            run_combs_docstring = create_run_docstring(run_combs_docstring_func)
            run_combs = compile_run_function('run_combs', run_combs_docstring, def_run_combs_kwargs)
            setattr(CustomIndicator, 'run_combs', run_combs)

        return CustomIndicator

    def from_apply_func(self, apply_func, cache_func=None, pass_kwargs=None, **kwargs):
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
            apply_func (callable): A function that takes broadcast time series arrays corresponding
                to `input_names`, single parameter selection corresponding to `param_names`, and other
                arguments and keyword arguments, and returns outputs corresponding to `output_names`.
                Can be Numba-compiled.
            cache_func (callable): A caching function to preprocess data beforehand.
                All returned objects will be passed as additional arguments to `apply_func`.
            pass_kwargs (list of str or list of tuple): Keyword arguments from `kwargs` dict to
                pass as positional arguments to the apply function.

                Defaults to []. Order matters.

                If any element is a tuple, should contain the name and the default value.
                If any element is a string, the default value is None.

                Built-in keys include:

                * `flex_2d`: See `vectorbt.base.reshape_fns.flex_choose_i_and_col_nb`.
                    Default is provided by the pipeline if `forward_flex_2d` is True.
            **kwargs: Keyword arguments passed to `IndicatorFactory.from_custom_func`.

        Returns:
            CustomIndicator

        ## Example

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
        if pass_kwargs is None:
            pass_kwargs = []
        output_names = self.output_names
        in_output_names = self.in_output_names

        num_ret_outputs = len(output_names) - len(in_output_names)

        if checks.is_numba_func(apply_func):
            if num_ret_outputs > 1:
                apply_and_concat_func = combine_fns.apply_and_concat_multiple_nb
            else:
                apply_and_concat_func = combine_fns.apply_and_concat_one_nb

            @njit
            def select_params(i, args_before, input_list, in_output_tuples, param_tuples, *args):
                # Select the next tuple of parameters
                return apply_func(*args_before, *input_list, *in_output_tuples[i], *param_tuples[i], *args)

        else:
            if num_ret_outputs > 1:
                apply_and_concat_func = combine_fns.apply_and_concat_multiple
            else:
                apply_and_concat_func = combine_fns.apply_and_concat_one

            def select_params(i, args_before, input_list, in_output_tuples, param_tuples, *args, **_kwargs):
                # Select the next tuple of parameters
                return apply_func(*args_before, *input_list, *in_output_tuples[i], *param_tuples[i], *args, **_kwargs)

        def custom_func(input_list, in_output_list, param_list, *args, input_shape=None,
                        return_cache=False, use_cache=None, **_kwargs):

            n_params = len(param_list[0]) if len(param_list) > 0 else 1
            input_list = tuple(input_list)
            in_output_tuples = tuple(zip(*in_output_list))
            if len(in_output_list) == 0:
                in_output_tuples = ((),) * n_params
            param_tuples = tuple(zip(*param_list))
            if len(param_list) == 0:
                param_tuples = ((),) * n_params
            args_before = ()
            if input_shape is not None:
                args_before += (input_shape,)

            # Pass some keyword arguments as positional
            more_args = ()
            for key in pass_kwargs:
                value = None
                if isinstance(key, tuple):
                    key, value = key
                value = _kwargs.pop(key, value)  # important: remove from kwargs
                more_args += (value,)

            # Caching
            cache = use_cache
            if cache is None and cache_func is not None:
                cache = cache_func(
                    *args_before,
                    *input_list,
                    *in_output_list,
                    *param_list,
                    *args,
                    *more_args,
                    **_kwargs
                )
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
                in_output_tuples,
                param_tuples,
                *args,
                *more_args,
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

        ## Example

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

        To get help on a function, use the `help` command:

        ```python-repl
        >>> help(SMA.run)
        Help on method run:

        run(close, timeperiod=30, short_name='sma', hide_params=None, hide_default=True, **kwargs) method of builtins.type instance
            Run the SMA indicator using input time series `close`, and parameters `timeperiod`, to
            produce output time series `real`.

            Pass a list of parameter names `hide_params` to hide their column levels.
            Set `hide_default` to False to show column levels of parameters with the default value passed.

            Other keyword arguments are passed to `vectorbt.indicators.factory.run_pipeline`.
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
            output_names=info['output_names'],
            output_flags=info['output_flags']
        ).from_custom_func(
            custom_func,
            **info['parameters'],
            **kwargs
        )
        return TALibIndicator
