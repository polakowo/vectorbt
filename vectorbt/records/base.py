"""Main class for working with records.

vectorbt works with two different representations of data: matrices and records.

A matrix, in this context, is just an array of one-dimensional arrays, each corresponding
to a separate feature. The matrix itself holds only one kind of information (one attribute).
For example, one can create a matrix for entry signals, with columns being different strategy
configurations. But what if the matrix is huge and sparse? What if there is more
information we would like to represent by each element? Creating multiple matrices would be
a waste of memory.

Records make possible representing complex, sparse information in a dense format. They are just
an array of one-dimensional arrays of fixed schema. You can imagine records being a DataFrame,
where each row represents a record and each column represents a specific attribute.

```plaintext
               a     b
         0   1.0   5.0
attr1 =  1   2.0   NaN
         2   NaN   7.0
         3   4.0   8.0
               a     b
         0   9.0  13.0
attr2 =  1  10.0   NaN
         2   NaN  15.0
         3  12.0  16.0
            |
            v
      col  idx  attr1  attr2
0       0    0      1      9
1       0    1      2     10
2       0    3      4     12
3       1    0      5     13
4       1    1      7     15
5       1    3      8     16
```

Another advantage of records is that they are not constrained by size. Multiple records can map
to a single element in a matrix. For example, one can define multiple orders at the same time step,
which is impossible to represent in a matrix form without using complex data types.

## Records class

`Records` are just [structured arrays](https://numpy.org/doc/stable/user/basics.rec.html) with a bunch
of methods and properties for processing them. Their main feature is to map the records array and
to reduce it by column (similar to the MapReduce paradigm). The main advantage is that it all happens
without conversion to the matrix form and wasting memory resources.

### Mapping

Consider the following example:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from collections import namedtuple
>>> from vectorbt.base.array_wrapper import ArrayWrapper
>>> from vectorbt.records import Records, MappedArray

>>> example_dt = np.dtype([
...     ('col', np.int64),
...     ('idx', np.int64),
...     ('some_field', np.float64)
... ])
>>> records_arr = np.array([
...     (0, 0, 10.),
...     (0, 1, 11.),
...     (0, 2, 12.),
...     (1, 0, 13.),
...     (1, 1, 14.),
...     (1, 2, 15.),
...     (2, 0, 16.),
...     (2, 1, 17.),
...     (2, 2, 18.)
... ], dtype=example_dt)
>>> wrapper = ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b', 'c'], ndim=2, freq='1 day')
>>> records = Records(wrapper, records_arr)

>>> records.records
   col  idx  some_field
0    0    0        10.0
1    0    1        11.0
2    0    2        12.0
3    1    0        13.0
4    1    1        14.0
5    1    2        15.0
6    2    0        16.0
7    2    1        17.0
8    2    2        18.0
```

`Records` can be mapped to `MappedArray` in several ways:

* Use `Records.map_field` to map a record field:

```python-repl
>>> records.map_field('some_field')
<vectorbt.records.base.MappedArray at 0x7ff49bd31a58>

>>> records.map_field('some_field').mapped_arr
array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
```

* Use `Records.map` to map records using a custom function.

```python-repl
>>> @njit
... def power_map_nb(record, pow):
...     return record.some_field ** pow

>>> records.map(power_map_nb, 2)
<vectorbt.records.base.MappedArray at 0x7ff49c990cf8>

>>> records.map(power_map_nb, 2).mapped_arr
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])
```

* Use `Records.map_array` to convert an array to `MappedArray`.

```python-repl
>>> records.map_array(records_arr['some_field'] ** 2)
<vectorbt.records.base.MappedArray object at 0x7fe9bccf2978>

>>> records.map_array(records_arr['some_field'] ** 2).mapped_arr
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])
```

## MappedArray class

When mapping records using `Records`, for example, to compute P&L of each trade record, the mapping
result is wrapped with `MappedArray` class. This class takes the mapped array and the corresponding column
and (optionally) index arrays, and offers features to directly process the mapped array without converting
it to the matrix form; for example, to compute various statistics by column, such as standard deviation.

### Reducing

Using `MappedArray`, you can then reduce by column as follows:

* Use already provided reducers such as `MappedArray.mean`:

```python-repl
>>> some_field = records.map_field('some_field')

>>> some_field.mean()
a    11.0
b    14.0
c    17.0
dtype: float64
```

* Use `MappedArray.to_matrix` to map to a matrix and then reduce manually (expensive):

```python-repl
>>> some_field.to_matrix().mean()
a    11.0
b    14.0
c    17.0
dtype: float64
```

* Use `MappedArray.reduce` to reduce using a custom function:

```python-repl
>>> @njit
... def pow_mean_reduce_nb(col, a, pow):
...     return np.mean(a ** pow)

>>> some_field.reduce(pow_mean_reduce_nb, 2)
a    121.666667
b    196.666667
c    289.666667
dtype: float64

>>> @njit
... def min_max_reduce_nb(col, a):
...     return np.array([np.min(a), np.max(a)])

>>> some_field.reduce(min_max_reduce_nb, to_array=True,
...     n_rows=2, index=['min', 'max'])
        a     b     c
min  10.0  13.0  16.0
max  12.0  15.0  18.0

>>> @njit
... def idxmin_idxmax_reduce_nb(col, a):
...     return np.array([np.argmin(a), np.argmax(a)])

>>> some_field.reduce(idxmin_idxmax_reduce_nb, to_array=True,
...     n_rows=2, to_idx=True, index=['idxmin', 'idxmax'])
        a  b  c
idxmin  x  x  x
idxmax  z  z  z
```

### Conversion

You can convert any `MappedArray` instance to the matrix form, given `idx_arr` was provided:

```python-repl
>>> some_field.to_matrix()
      a     b     c
x  10.0  13.0  16.0
y  11.0  14.0  17.0
z  12.0  15.0  18.0
```

!!! note
    Will raise an error if there are multiple records pointing to the same matrix element.

### Plotting

You can build histograms and boxplots of `MappedArray` directly:

```python-repl
>>> some_field.box()
```

![](/vectorbt/docs/img/mapped_box.png)

To use scatterplots or any other plots that require index, convert to matrix first:

```python-repl
>>> some_field.to_matrix().vbt.scatter(trace_kwargs=dict(connectgaps=True))
```

![](/vectorbt/docs/img/mapped_scatter.png)

### Grouping

One of the key features of `MappedArray` and `Records` is that you can perform reducing operations
on a group of columns as if they were a single column. Groups can be specified by `group_by`, which
can be anything from positions or names of column levels, to a NumPy array with actual groups.

There are multiple ways of define grouping:

* When creating `MappedArray` or `Records`, pass `group_by` to `vectorbt.base.array_wrapper.ArrayWrapper`:

```python-repl
>>> group_by = np.array([0, 0, 1])
>>> grouped_wrapper = ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b', 'c'], ndim=2, freq='1 day', group_by=group_by)
>>> grouped_records = Records(grouped_wrapper, records_arr)
>>> grouped_some_field = grouped_records.map_field('some_field')

>>> grouped_some_field.mean()
0    12.5
1    17.0
dtype: float64
```

* Regroup an existing `MappedArray` or `Records`:

```python-repl
>>> some_field.regroup(group_by).mean()
0    12.5
1    17.0
dtype: float64
```

* Pass `group_by` directly to the reducing method:

```python-repl
>>> some_field.mean(group_by=group_by)
0    12.5
1    17.0
dtype: float64
```

By the same way you can disable or modify existing grouping:

```python-repl
>>> grouped_some_field.mean(group_by=False)
a    11.0
b    14.0
c    17.0
dtype: float64
```

!!! note
    Grouping applies only to reducing operations, there is no change to the arrays.

### Operators

`MappedArray` implements arithmetic, comparison and logical operators. You can perform basic
operations (such as addition) on mapped arrays as if they were NumPy arrays.

```python-repl
>>> some_field ** 2
<vectorbt.records.base.MappedArray at 0x7f97bfc49358>

>>> some_field * np.array([1, 2, 3, 4, 5, 6])
<vectorbt.records.base.MappedArray at 0x7f97bfc65e80>

>>> some_field + some_field
<vectorbt.records.base.MappedArray at 0x7f97bfc492e8>
```

!!! note
    You should ensure that your `MappedArray` operand is on the left if the other operand is an array.

    Two mapped arrays must have the same metadata to be combined.

## Indexing

You can use pandas indexing on both the `Records` and `MappedArray` class, which will forward
the indexing operation to each `__init__` argument with index:

```python-repl
>>> records['a'].records
   col  idx  some_field
0    0    0        10.0
1    0    1        11.0
2    0    2        12.0

>>> some_field['a'].mapped_arr
[10. 11. 12.]
```

!!! note
    Changing index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame; for example, use `some_field.iloc[0]` instead of `some_field.iloc[:, 0]`.

    Indexing behavior depends solely upon `vectorbt.base.array_wrapper.ArrayWrapper`.
    For example, if `group_select` is enabled indexing will be performed on groups,
    otherwise on single columns.

## Caching

Both classes support caching. If a method or a property requires heavy computation, it's wrapped
with `vectorbt.utils.decorators.cached_method` and `vectorbt.utils.decorators.cached_property` respectively.
Caching can be disabled globally via `vectorbt.defaults` or locally via the method/property.
There is currently no way to disable caching for an entire class.

!!! note
    Because of caching, both classes are meant to be immutable and all properties are read-only.
    To change any attribute, use the `copy` method and pass the attribute as keyword argument.
"""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import Configured
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base import reshape_fns
from vectorbt.base.common import (
    add_binary_magic_methods,
    add_unary_magic_methods,
    binary_magic_methods,
    unary_magic_methods
)
from vectorbt.base.array_wrapper import ArrayWrapper, indexing_on_wrapper_meta
from vectorbt.generic import nb as generic_nb
from vectorbt.records import nb


def indexing_on_mapped_array_meta(obj, pd_indexing_func):
    """Perform indexing on `MappedArray` and return metadata."""
    new_wrapper, _, group_idxs, col_idxs = \
        indexing_on_wrapper_meta(obj.wrapper, pd_indexing_func, column_only_select=True)
    new_indices, new_col_arr = nb.select_mapped_cols_nb(
        obj.col_arr,
        obj.col_index,
        reshape_fns.to_1d(col_idxs)
    )
    new_mapped_arr = obj.mapped_arr[new_indices]
    if obj.idx_arr is not None:
        new_idx_arr = obj.idx_arr[new_indices]
    else:
        new_idx_arr = None
    return new_wrapper, new_mapped_arr, new_col_arr, new_idx_arr, group_idxs, col_idxs


def _mapped_array_indexing_func(obj, pd_indexing_func):
    """Perform indexing on `MappedArray`."""
    new_wrapper, new_mapped_arr, new_col_arr, new_idx_arr, _, _ = \
        indexing_on_mapped_array_meta(obj, pd_indexing_func)
    return obj.copy(
        wrapper=new_wrapper,
        mapped_arr=new_mapped_arr,
        col_arr=new_col_arr,
        idx_arr=new_idx_arr
    )


def _mapped_binary_translate_func(self, other, np_func):
    """Perform operation between two instances of `MappedArray`."""
    if isinstance(other, self.__class__):
        passed = True
        if self.wrapper != other.wrapper:
            passed = False
        if not np.array_equal(self.col_arr, other.col_arr):
            passed = False
        if self.idx_arr is not None or other.idx_arr is not None:
            if not np.array_equal(self.idx_arr, other.idx_arr):
                passed = False
        if not passed:
            raise ValueError("Both MappedArray instances must have same metadata")
        other = other.mapped_arr
    return self.__class__(
        self.wrapper,
        np_func(self.mapped_arr, other),
        self.col_arr,
        idx_arr=self.idx_arr
    )


@add_binary_magic_methods(
    binary_magic_methods,
    _mapped_binary_translate_func
)
@add_unary_magic_methods(
    unary_magic_methods,
    lambda self, np_func: self.__class__(
        self.wrapper,
        np_func(self.mapped_arr),
        self.col_arr,
        idx_arr=self.idx_arr
    )
)
class MappedArray(Configured, PandasIndexer):
    """Exposes methods and properties for working with records.

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbt.base.array_wrapper.ArrayWrapper`.
        mapped_arr (array_like): A one-dimensional array of mapped record values.
        col_arr (array_like): A one-dimensional column array.

            Must be of the same size as `mapped_arr`.
        idx_arr (array_like): A one-dimensional index array. Optional.

            Must be of the same size as `mapped_arr`.

    !!! note
        This class is meant to be immutable. To change any attribute, use `MappedArray.copy`."""

    def __init__(self, wrapper, mapped_arr, col_arr, idx_arr=None):
        Configured.__init__(
            self,
            wrapper=wrapper,
            mapped_arr=mapped_arr,
            col_arr=col_arr,
            idx_arr=idx_arr
        )
        checks.assert_type(wrapper, ArrayWrapper)
        if not isinstance(mapped_arr, np.ndarray):
            mapped_arr = np.asarray(mapped_arr)
        if not isinstance(col_arr, np.ndarray):
            col_arr = np.asarray(col_arr)
        checks.assert_shape_equal(mapped_arr, col_arr, axis=0)
        if idx_arr is not None:
            if not isinstance(idx_arr, np.ndarray):
                idx_arr = np.asarray(idx_arr)
            checks.assert_shape_equal(mapped_arr, idx_arr, axis=0)

        self._wrapper = wrapper
        self._mapped_arr = mapped_arr
        self._col_arr = col_arr
        self._idx_arr = idx_arr

        PandasIndexer.__init__(self, _mapped_array_indexing_func)

    @property
    def wrapper(self):
        """Array wrapper."""
        return self._wrapper

    def regroup(self, group_by):
        """Regroup this object."""
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            return self.copy(wrapper=self.wrapper.copy(group_by=group_by))
        return self

    @property
    def mapped_arr(self):
        """Mapped array."""
        return self._mapped_arr

    @property
    def col_arr(self):
        """Column array."""
        return self._col_arr

    @property
    def idx_arr(self):
        """Index array."""
        return self._idx_arr

    @cached_property
    def col_index(self):
        """Column index for `MappedArray.mapped_arr`."""
        return nb.mapped_col_index_nb(self.mapped_arr, self.col_arr, len(self.wrapper.columns))

    def filter_by_mask(self, mask, idx_arr=None, group_by=None, **kwargs):
        """Return a new class instance, filtered by mask."""
        if idx_arr is None:
            idx_arr = self.idx_arr
        if idx_arr is not None:
            idx_arr = self.idx_arr[mask]
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            wrapper = self.wrapper.copy(group_by=group_by)
        else:
            wrapper = self.wrapper
        return self.copy(
            wrapper=wrapper,
            mapped_arr=self.mapped_arr[mask],
            col_arr=self.col_arr[mask],
            idx_arr=idx_arr,
            **kwargs
        )

    def to_matrix(self, idx_arr=None, default_val=np.nan):
        """Convert mapped array to the matrix form.

        See `vectorbt.records.nb.mapped_to_matrix_nb`.

        !!! warning
            Mapped arrays represent information in the most memory-friendly format.
            Mapping back to the matrix form may occupy lots of memory if records are sparse."""
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        target_shape = (len(self.wrapper.index), len(self.wrapper.columns))
        result = nb.mapped_to_matrix_nb(self.mapped_arr, self.col_arr, idx_arr, target_shape, default_val)
        return self.wrapper.wrap(result, group_by=False)

    def reduce(self, reduce_func_nb, *args, idx_arr=None, to_array=False, n_rows=None, to_idx=False,
               idx_labeled=True, default_val=np.nan, cast=None, group_by=None, **kwargs):
        """Reduce mapped array by column.

        If `to_array` is False and `to_idx` is False, see `vectorbt.records.nb.reduce_mapped_nb`.
        If `to_array` is False and `to_idx` is True, see `vectorbt.records.nb.reduce_mapped_to_idx_nb`.
        If `to_array` is True and `to_idx` is False, see `vectorbt.records.nb.reduce_mapped_to_array_nb`.
        If `to_array` is True and `to_idx` is True, see `vectorbt.records.nb.reduce_mapped_to_idx_array_nb`.

        If `to_array` is True, must pass `n_rows` indicating the number of elements in the array.
        If `to_idx` is True, must pass `idx_arr`. Set `idx_labeled` to False to return raw positions
        instead of labels. Use `default_val` to set the default value and `cast` to perform casting
        on the resulting pandas object. Set `group_by` to False to disable grouping.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`."""
        # Perform checks
        checks.assert_numba_func(reduce_func_nb)
        if idx_arr is None:
            if self.idx_arr is None:
                if to_idx:
                    raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr

        # Perform main computation
        group_arr, columns = self.wrapper.grouper.get_groups_and_columns(group_by=group_by)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        if not to_array:
            if not to_idx:
                result = nb.reduce_mapped_nb(
                    self.mapped_arr,
                    col_arr,
                    len(columns),
                    default_val,
                    reduce_func_nb,
                    *args
                )
            else:
                result = nb.reduce_mapped_to_idx_nb(
                    self.mapped_arr,
                    col_arr,
                    idx_arr,
                    len(columns),
                    default_val,
                    reduce_func_nb,
                    *args
                )
        else:
            checks.assert_not_none(n_rows)
            if not to_idx:
                result = nb.reduce_mapped_to_array_nb(
                    self.mapped_arr,
                    col_arr,
                    len(columns),
                    n_rows,
                    default_val,
                    reduce_func_nb,
                    *args
                )
            else:
                result = nb.reduce_mapped_to_idx_array_nb(
                    self.mapped_arr,
                    col_arr,
                    idx_arr,
                    len(columns),
                    n_rows,
                    default_val,
                    reduce_func_nb,
                    *args
                )

        # Perform post-processing
        if to_idx:
            if idx_labeled:
                result_shape = result.shape
                result = result.flatten()
                mask = np.isnan(result)
                if mask.any():
                    # Contains NaNs
                    result[mask] = 0
                    result = result.astype(int)
                    result = self.wrapper.index[result].to_numpy()
                    result = result.astype(np.object)
                    result[mask] = np.nan
                else:
                    result = self.wrapper.index[result.astype(int)].to_numpy()
                result = np.reshape(result, result_shape)
            else:
                mask = np.isnan(result)
                result[mask] = -1
                result = result.astype(int)
        result = self.wrapper.wrap_reduced(result, group_by=group_by, **kwargs)
        if cast is not None:
            result = result.astype(cast)
        return result

    @cached_method
    def nst(self, n, group_by=None, **kwargs):
        """Return nst element of each column."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("MappedArray.nst() does not support grouping. Set group_by to False.")
        return self.reduce(generic_nb.nst_reduce_nb, n, to_array=False, to_idx=False, group_by=False, **kwargs)

    @cached_method
    def min(self, **kwargs):
        """Return min by column."""
        return self.reduce(generic_nb.min_reduce_nb, to_array=False, to_idx=False, **kwargs)

    @cached_method
    def max(self, **kwargs):
        """Return max by column."""
        return self.reduce(generic_nb.max_reduce_nb, to_array=False, to_idx=False, **kwargs)

    @cached_method
    def mean(self, **kwargs):
        """Return mean by column."""
        return self.reduce(generic_nb.mean_reduce_nb, to_array=False, to_idx=False, **kwargs)

    @cached_method
    def median(self, **kwargs):
        """Return median by column."""
        return self.reduce(generic_nb.median_reduce_nb, to_array=False, to_idx=False, **kwargs)

    @cached_method
    def std(self, ddof=1, **kwargs):
        """Return std by column."""
        return self.reduce(generic_nb.std_reduce_nb, ddof, to_array=False, to_idx=False, **kwargs)

    @cached_method
    def sum(self, default_val=0., **kwargs):
        """Return sum by column."""
        return self.reduce(
            generic_nb.sum_reduce_nb,
            to_array=False,
            to_idx=False,
            default_val=default_val,
            **kwargs
        )

    @cached_method
    def count(self, default_val=0., cast=np.int64, **kwargs):
        """Return count by column."""
        return self.reduce(
            generic_nb.count_reduce_nb,
            to_array=False,
            to_idx=False,
            default_val=default_val,
            cast=cast,
            **kwargs
        )

    @cached_method
    def describe(self, percentiles=None, ddof=1, **kwargs):
        """Return statistics by column."""
        if percentiles is not None:
            percentiles = reshape_fns.to_1d(percentiles, raw=True)
        else:
            percentiles = np.array([0.25, 0.5, 0.75])
        percentiles = percentiles.tolist()
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        percentiles = np.unique(percentiles)
        perc_formatted = pd.io.formats.format.format_percentiles(percentiles)
        index = pd.Index(['count', 'mean', 'std', 'min', *perc_formatted, 'max'])
        result = self.reduce(
            generic_nb.describe_reduce_nb,
            percentiles,
            ddof,
            to_array=True,
            n_rows=len(index),
            to_idx=False,
            index=index,
            **kwargs
        )
        if isinstance(result, pd.DataFrame):
            result.loc['count'].fillna(0., inplace=True)
        else:
            if np.isnan(result.loc['count']):
                result.loc['count'] = 0.
        return result

    @cached_method
    def idxmin(self, **kwargs):
        """Return index of min by column."""
        return self.reduce(generic_nb.argmin_reduce_nb, to_array=False, to_idx=True, **kwargs)

    @cached_method
    def idxmax(self, **kwargs):
        """Return index of max by column."""
        return self.reduce(generic_nb.argmax_reduce_nb, to_array=False, to_idx=True, **kwargs)

    def plot_by_func(self, plot_func, group_by=None):  # pragma: no cover
        """Transform data to the format suitable for plotting, and plot.

        Function `pd_plot_func` should receive Series or DataFrame and plot it.
        Should only be used by plotting methods that disregard X axis labels.

        Set `group_by` to False to disable grouping."""
        # We can't simply do to_matrix since there can be multiple records for one position in matrix
        if self.wrapper.ndim == 1:
            name = None if self.wrapper.columns[0] == 0 else self.wrapper.columns[0]
            return plot_func(pd.Series(self.mapped_arr, name=name))
        group_arr, columns = self.wrapper.grouper.get_groups_and_columns(group_by=group_by)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        a = np.full((self.mapped_arr.shape[0], len(columns)), np.nan)
        for col in range(len(columns)):
            masked_arr = self.mapped_arr[col_arr == col]
            a[:masked_arr.shape[0], col] = masked_arr
        return plot_func(pd.DataFrame(a, columns=columns))

    def hist(self, group_by=None, **kwargs):  # pragma: no cover
        """Plot histogram by column."""
        return self.plot_by_func(lambda x: x.vbt.hist(**kwargs), group_by=group_by)

    def box(self, group_by=None, **kwargs):  # pragma: no cover
        """Plot box plot by column."""
        return self.plot_by_func(lambda x: x.vbt.box(**kwargs), group_by=group_by)


def indexing_on_records_meta(obj, pd_indexing_func):
    """Perform indexing on `Records` and return metadata."""
    new_wrapper, _, group_idxs, col_idxs = \
        indexing_on_wrapper_meta(obj.wrapper, pd_indexing_func, column_only_select=True)
    new_records_arr = nb.select_record_cols_nb(
        obj.records_arr,
        obj.col_index,
        reshape_fns.to_1d(col_idxs)
    )
    return new_wrapper, new_records_arr, group_idxs, col_idxs


def _records_indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Records`."""
    new_wrapper, new_records_arr, _, _ = indexing_on_records_meta(obj, pd_indexing_func)
    return obj.copy(
        wrapper=new_wrapper,
        records_arr=new_records_arr
    )


class Records(Configured, PandasIndexer):
    """Exposes methods and properties for working with records.

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbt.base.array_wrapper.ArrayWrapper`.
        records_arr (array_like): A structured NumPy array of records.

            Must have the field `col` (column position in a matrix).
        idx_field (str): The name of the field corresponding to the index. Optional.

            Will be derived automatically if records contain field `'idx'`.

    !!! note
        This class is meant to be immutable. To change any attribute, use `Records.copy`."""

    def __init__(self, wrapper, records_arr, idx_field=None):
        Configured.__init__(
            self,
            wrapper=wrapper,
            records_arr=records_arr,
            idx_field=idx_field
        )
        checks.assert_type(wrapper, ArrayWrapper)
        if not isinstance(records_arr, np.ndarray):
            records_arr = np.asarray(records_arr)
        checks.assert_not_none(records_arr.dtype.fields)
        checks.assert_in('col', records_arr.dtype.names)
        if idx_field is not None:
            checks.assert_in(idx_field, records_arr.dtype.names)
        else:
            if 'idx' in records_arr.dtype.names:
                idx_field = 'idx'

        self._wrapper = wrapper
        self._records_arr = records_arr
        self._idx_field = idx_field

        PandasIndexer.__init__(self, _records_indexing_func)

    @property
    def wrapper(self):
        """Array wrapper."""
        return self._wrapper

    def regroup(self, group_by):
        """Regroup this object."""
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            return self.copy(wrapper=self.wrapper.copy(group_by=group_by))
        return self

    @property
    def records_arr(self):
        """Records array."""
        return self._records_arr

    @property
    def idx_field(self):
        """Index field."""
        return self._idx_field

    @cached_property
    def records(self):
        """Records."""
        return pd.DataFrame.from_records(self.records_arr)

    @cached_property
    def recarray(self):
        return self.records_arr.view(np.recarray)

    @cached_property
    def col_index(self):
        """Column index for `Records.records`."""
        return nb.record_col_index_nb(self.records_arr, len(self.wrapper.columns))

    def filter_by_mask(self, mask, group_by=None, **kwargs):
        """Return a new class instance, filtered by mask."""
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            wrapper = self.wrapper.copy(group_by=group_by)
        else:
            wrapper = self.wrapper
        return self.copy(
            wrapper=wrapper,
            records_arr=self.records_arr[mask],
            **kwargs
        )

    def map(self, map_func_nb, *args, idx_arr=None, group_by=None, **kwargs):
        """Map each record to a scalar value. Returns `MappedArray`.

        See `vectorbt.records.nb.map_records_nb`."""
        checks.assert_numba_func(map_func_nb)

        mapped_arr = nb.map_records_nb(self.records_arr, map_func_nb, *args)
        if idx_arr is None:
            if self.idx_field is not None:
                idx_arr = self.records_arr[self.idx_field]
            else:
                idx_arr = None
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            wrapper = self.wrapper.copy(group_by=group_by)
        else:
            wrapper = self.wrapper
        return MappedArray(
            wrapper,
            mapped_arr,
            self.records_arr['col'],
            idx_arr=idx_arr,
            **kwargs
        )

    def map_field(self, field, idx_arr=None, group_by=None, **kwargs):
        """Convert field to `MappedArray`."""
        if idx_arr is None:
            if self.idx_field is not None:
                idx_arr = self.records_arr[self.idx_field]
            else:
                idx_arr = None
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            wrapper = self.wrapper.copy(group_by=group_by)
        else:
            wrapper = self.wrapper
        return MappedArray(
            wrapper,
            self.records_arr[field],
            self.records_arr['col'],
            idx_arr=idx_arr,
            **kwargs
        )

    def map_array(self, a, idx_arr=None, group_by=None, **kwargs):
        """Convert array to `MappedArray`.

         The length of the array should match that of the records."""
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        checks.assert_shape_equal(a, self.records_arr)

        if idx_arr is None:
            if self.idx_field is not None:
                idx_arr = self.records_arr[self.idx_field]
            else:
                idx_arr = None
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            wrapper = self.wrapper.copy(group_by=group_by)
        else:
            wrapper = self.wrapper
        return MappedArray(
            wrapper,
            a,
            self.records_arr['col'],
            idx_arr=idx_arr,
            **kwargs
        )

    @cached_method
    def count(self, **kwargs):
        """Number of records."""
        return self.map_field('col').count(default_val=0., **kwargs)



