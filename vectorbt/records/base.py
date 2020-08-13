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
of methods and properties for processing them. Its main feature is to map the records array and
to reduce it by column (similar to the MapReduce paradigm). The main advantage is that it all happens
without conversion to the matrix form and wasting memory resources.

## MappedArray class

When mapping records using `Records`, for example, to compute P&L of each trade record, the mapping
result is wrapped with `MappedArray` class. This class takes the mapped array and the corresponding column
and (optionally) index arrays, and offers features to directly process the mapped array without converting
it to the matrix form; for example, to compute various statistics by column, such as standard deviation.

## Example

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
...     (1, 2, 15.)
... ], dtype=example_dt)
>>> wrapper = ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b'], ndim=2, freq='1 day')
>>> records = Records(records_arr, wrapper)

>>> records.records
   col  idx  some_field
0    0    0        10.0
1    0    1        11.0
2    0    2        12.0
3    1    0        13.0
4    1    1        14.0
5    1    2        15.0
```

### Mapping

There are several options for mapping:

* Use `Records.map_field` to map a record field:

```python-repl
>>> records.map_field('some_field')
<vectorbt.records.base.MappedArray at 0x7ff49bd31a58>

>>> records.map_field('some_field').mapped_arr
[10. 11. 12. 13. 14. 15.]
```

* Use `Records.map` to map records using a custom function.

```python-repl
>>> @njit
... def power_map_nb(record, pow):
...     return record.some_field ** pow

>>> records.map(power_map_nb, 2)
<vectorbt.records.base.MappedArray at 0x7ff49c990cf8>

>>> records.map(power_map_nb, 2).mapped_arr
[100. 121. 144. 169. 196. 225.]
```

* Use `Records.map_array` to convert an array to `MappedArray`.

```python-repl
>>> records.map_array(records_arr['some_field'] ** 2)
<vectorbt.records.base.MappedArray object at 0x7fe9bccf2978>

>>> records.map_array(records_arr['some_field'] ** 2).mapped_arr
[100. 121. 144. 169. 196. 225.]
```

### Reducing

Using `MappedArray`, you can then reduce by column as follows:

* Use already provided reducers such as `MappedArray.mean`:

```python-repl
>>> mapped = records.map_field('some_field')

>>> mapped.mean()
a    11.0
b    14.0
dtype: float64
```

* Use `MappedArray.to_matrix` to map to a matrix and then reduce manually (expensive):

```python-repl
>>> mapped.to_matrix().mean()
a    11.0
b    14.0
dtype: float64
```

* Use `MappedArray.reduce` to reduce using a custom function:

```python-repl
>>> @njit
... def pow_mean_reduce_nb(col, a, pow):
...     return np.mean(a ** pow)

>>> mapped.reduce(pow_mean_reduce_nb, 2)
a    121.666667
b    196.666667
dtype: float64

>>> @njit
... def min_max_reduce_nb(col, a):
...     return np.array([np.min(a), np.max(a)])

>>> mapped.reduce(min_max_reduce_nb, to_array=True,
...     n_rows=2, index=['min', 'max'])
        a     b
min  10.0  13.0
max  12.0  15.0

>>> @njit
... def idxmin_idxmax_reduce_nb(col, a):
...     return np.array([np.argmin(a), np.argmax(a)])

>>> mapped.reduce(idxmin_idxmax_reduce_nb, to_array=True,
...     n_rows=2, to_idx=True, index=['idxmin', 'idxmax'])
        a  b
idxmin  x  x
idxmax  z  z
```

### Conversion

You can convert any `MappedArray` instance to the matrix form, given `idx_arr` was provided:

```python-repl
>>> mapped.to_matrix()
      a     b
x  10.0  13.0
y  11.0  14.0
z  12.0  15.0
```

!!! note
    Will raise an error if there are multiple records pointing to the same matrix element.

### Plotting

You can build histograms and boxplots of `MappedArray` directly:

```python-repl
>>> mapped.box()
```

![](/vectorbt/docs/img/mapped_box.png)

To use scatterplots or any other plots that require index, convert to matrix first:

```python-repl
>>> mapped.to_matrix().vbt.scatter(trace_kwargs=dict(connectgaps=True))
```

![](/vectorbt/docs/img/mapped_scatter.png)

## Grouping

Additionally to reducing per column, you can also reduce per group of columns by providing
`group_by`. The `group_by` variable can be anything from positions or names of column levels,
to a NumPy array with actual groups, and can be passed to either `Records`, `MappedArray`,
or the reducing method itself.

```python-repl
>>> np.random.seed(42)
>>> index = pd.Index(['x', 'y'])
>>> columns = pd.MultiIndex.from_arrays([
...     [1, 1, 1, 2, 2, 2],
...     [1, 2, 3, 1, 2, 3]
... ], names=['a', 'b'])
>>> mapped_group = MappedArray(
...     mapped_arr=np.random.randint(1, 10, size=12),
...     col_arr=np.repeat(np.arange(6), 2),
...     wrapper=ArrayWrapper(index=index, columns=columns),
...     idx_arr=np.tile([0, 1], 6)
... )

>>> mapped_group.hist()
```

![](/vectorbt/docs/img/mapped_hist.png)

```python-repl
>>> mapped_group.hist(group_by='a')
```

![](/vectorbt/docs/img/mapped_hist_grouped.png)

!!! note
    Grouping applies only to reducing and plotting operations.

## Indexing

You can use pandas indexing on both the `Records` and `MappedArray` class, which will forward
the indexing operation to each `__init__` argument with index:

```python-repl
>>> records['a'].records
   col  idx  some_field
0    0    0        10.0
1    0    1        11.0
2    0    2        12.0

>>> mapped['a'].mapped_arr
[10. 11. 12.]
```

!!! note
    Changing index (time axis) is not supported.

## Operators

Additionally, `MappedArray` implements arithmetic, comparison and logical operators.
You can perform basic operations (such as addition) on mapped arrays as if they were NumPy arrays.

```python-repl
>>> mapped ** 2
<vectorbt.records.base.MappedArray at 0x7f97bfc49358>

>>> mapped * np.array([1, 2, 3, 4, 5, 6])
<vectorbt.records.base.MappedArray at 0x7f97bfc65e80>

>>> mapped + mapped
<vectorbt.records.base.MappedArray at 0x7f97bfc492e8>
```

!!! note
    You should ensure that your `*.vbt` operand is on the left if the other operand is an array.

    Two mapped arrays must have the same metadata to be compared/combined.
"""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base import reshape_fns
from vectorbt.base.index_grouper import IndexGrouper
from vectorbt.base.common import (
    add_binary_magic_methods,
    add_unary_magic_methods,
    binary_magic_methods,
    unary_magic_methods
)
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic import nb as generic_nb
from vectorbt.records import nb


def _mapped_indexing_func(obj, pd_indexing_func):
    """Perform indexing on `MappedArray`."""
    if obj.wrapper.ndim == 1:
        raise TypeError("Indexing on Series is not supported")

    n_rows = len(obj.wrapper.index)
    n_cols = len(obj.wrapper.columns)
    col_mapper = obj.wrapper.wrap(np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)))
    col_mapper = pd_indexing_func(col_mapper)
    if not pd.Index.equals(col_mapper.index, obj.wrapper.index):
        raise NotImplementedError("Changing index (time axis) is not supported")

    new_cols = reshape_fns.to_1d(col_mapper.values[0])  # array required
    new_indices, new_col_arr = nb.select_mapped_cols_nb(
        obj.col_arr,
        obj.col_index,
        new_cols
    )
    new_mapped_arr = obj.mapped_arr[new_indices]
    if obj.idx_arr is not None:
        new_idx_arr = obj.idx_arr[new_indices]
    else:
        new_idx_arr = None
    new_wrapper = ArrayWrapper.from_obj(col_mapper, freq=obj.wrapper.freq)
    if obj.grouper.group_by is not None:
        new_group_by = obj.grouper.group_by[new_cols]
    else:
        new_group_by = None
    return obj.__class__(
        new_mapped_arr,
        new_col_arr,
        new_wrapper,
        idx_arr=new_idx_arr,
        group_by=new_group_by
    )


def _mapped_binary_translate_func(self, other, np_func):
    """Perform operation between two instances of `MappedArray`."""
    if isinstance(other, self.__class__):
        passed = True
        if not np.array_equal(self.col_arr, other.col_arr):
            passed = False
        if self.idx_arr is not None or other.idx_arr is not None:
            if not np.array_equal(self.idx_arr, other.idx_arr):
                passed = False
        if self.wrapper != other.wrapper:
            passed = False
        if self.grouper != other.grouper:
            passed = False
        if not passed:
            raise ValueError("Both MappedArray instances must have same metadata")
        other = other.mapped_arr
    return self.__class__(
        np_func(self.mapped_arr, other),
        self.col_arr,
        self.wrapper,
        idx_arr=self.idx_arr,
        group_by=self.grouper.group_by
    )


@add_binary_magic_methods(
    binary_magic_methods,
    _mapped_binary_translate_func
)
@add_unary_magic_methods(
    unary_magic_methods,
    lambda self, np_func: self.__class__(
        np_func(self.mapped_arr),
        self.col_arr,
        self.wrapper,
        idx_arr=self.idx_arr,
        group_by=self.grouper.group_by
    )
)
class MappedArray(PandasIndexer):
    """Exposes methods and properties for working with records.

    Args:
        mapped_arr (array_like): A one-dimensional array of mapped record values.
        col_arr (array_like): A one-dimensional column array.

            Must be of the same size as `mapped_arr`.
        wrapper (ArrayWrapper): Array wrapper of type `vectorbt.base.array_wrapper.ArrayWrapper`.
        idx_arr (array_like): A one-dimensional index array. Optional.

            Must be of the same size as `mapped_arr`.
        group_by (int, str or array_like): Group columns by a mapper when reducing.

            See `vectorbt.base.index_fns.group_index`."""

    def __init__(self, mapped_arr, col_arr, wrapper, idx_arr=None, group_by=None):
        if not isinstance(mapped_arr, np.ndarray):
            mapped_arr = np.asarray(mapped_arr)
        if not isinstance(col_arr, np.ndarray):
            col_arr = np.asarray(col_arr)
        checks.assert_same_shape(mapped_arr, col_arr, axis=0)
        checks.assert_type(wrapper, ArrayWrapper)
        if idx_arr is not None:
            if not isinstance(idx_arr, np.ndarray):
                idx_arr = np.asarray(idx_arr)
            checks.assert_same_shape(mapped_arr, idx_arr, axis=0)

        self.mapped_arr = mapped_arr
        self.col_arr = col_arr
        self.wrapper = wrapper
        self.idx_arr = idx_arr
        self.grouper = IndexGrouper(self.wrapper.columns, group_by=group_by)

        PandasIndexer.__init__(self, _mapped_indexing_func)

    @cached_property
    def col_index(self):
        """Column index for `MappedArray.mapped_arr`."""
        return nb.mapped_col_index_nb(self.mapped_arr, self.col_arr, len(self.wrapper.columns))

    def filter_by_mask(self, mask, idx_arr=None, group_by=None):
        """Return a new class instance, filtered by mask."""
        if idx_arr is None:
            idx_arr = self.idx_arr
        if idx_arr is not None:
            idx_arr = self.idx_arr[mask]
        if group_by is None:
            group_by = self.grouper.group_by
        return self.__class__(
            self.mapped_arr[mask],
            self.col_arr[mask],
            self.wrapper,
            idx_arr=idx_arr,
            group_by=group_by
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
        return self.wrapper.wrap(result)

    def reduce(self, reduce_func_nb, *args, idx_arr=None, to_array=False,
               n_rows=None, to_idx=False, idx_labeled=True, default_val=np.nan,
               cast=None, group_by=None, columns=None, **kwargs):
        """Reduce mapped array by column.

        If `to_array` is `False` and `to_idx` is `False`, see `vectorbt.records.nb.reduce_mapped_nb`.
        If `to_array` is `False` and `to_idx` is `True`, see `vectorbt.records.nb.reduce_mapped_to_idx_nb`.
        If `to_array` is `True` and `to_idx` is `False`, see `vectorbt.records.nb.reduce_mapped_to_array_nb`.
        If `to_array` is `True` and `to_idx` is `True`, see `vectorbt.records.nb.reduce_mapped_to_idx_array_nb`.

        If `to_array` is `True`, must pass `n_rows` indicating the number of elements in the array.
        If `to_idx` is `True`, must pass `idx_arr`. Set `idx_labeled` to `False` to return raw positions
        instead of labels. Use `default_val` to set the default value and `cast` to perform casting
        on the resulting pandas object. Set `group_by` to `False` to disable grouping. Use `columns` to
        override the existing columns and the columns coming from `group_by`.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`."""
        # Perform checks
        checks.assert_numba_func(reduce_func_nb)
        if idx_arr is None:
            if self.idx_arr is None:
                if to_idx:
                    raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr

        # Perform main computation
        group_arr, columns = self.grouper.group_index(group_by=group_by, new_index=columns)
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
        result = self.wrapper.wrap_reduced(result, columns=columns, **kwargs)
        if cast is not None:
            result = result.astype(cast)
        return result

    @cached_method
    def nst(self, n, group_by=None, **kwargs):
        """Return nst element of each column."""
        group_by = self.grouper.resolve_group_by(group_by=group_by)
        if group_by is not None:
            raise ValueError("MappedArray.nst() does not support group_by. Set group_by to False.")
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

    def plot_by_func(self, plot_func, group_by=None, columns=None):  # pragma: no cover
        """Transform data to the format suitable for plotting, and plot.

        Function `pd_plot_func` should receive Series or DataFrame and plot it.
        Should only be used by plotting methods that disregard X axis labels.

        Set `group_by` to `False` to disable grouping."""
        # We can't simply do to_matrix since there can be multiple records for one position in matrix
        if self.wrapper.ndim == 1:
            name = None if self.wrapper.columns[0] == 0 else self.wrapper.columns[0]
            return plot_func(pd.Series(self.mapped_arr, name=name))
        group_arr, columns = self.grouper.group_index(group_by=group_by, new_index=columns)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        a = np.full((self.mapped_arr.shape[0], len(columns)), np.nan)
        for col in range(len(columns)):
            masked_arr = self.mapped_arr[col_arr == col]
            a[:masked_arr.shape[0], col] = masked_arr
        return plot_func(pd.DataFrame(a, columns=columns))

    def hist(self, group_by=None, columns=None, **kwargs):  # pragma: no cover
        """Plot histogram by column."""
        return self.plot_by_func(
            lambda x: x.vbt.hist(**kwargs),
            group_by=group_by,
            columns=columns
        )

    def box(self, group_by=None, columns=None, **kwargs):  # pragma: no cover
        """Plot box plot by column."""
        return self.plot_by_func(
            lambda x: x.vbt.box(**kwargs),
            group_by=group_by,
            columns=columns
        )


def indexing_on_records(obj, pd_indexing_func):
    """Perform indexing on `Records`."""
    if obj.wrapper.ndim == 1:
        raise TypeError("Indexing on Series is not supported")

    n_rows = len(obj.wrapper.index)
    n_cols = len(obj.wrapper.columns)
    col_mapper = obj.wrapper.wrap(np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)))
    col_mapper = pd_indexing_func(col_mapper)
    if not pd.Index.equals(col_mapper.index, obj.wrapper.index):
        raise NotImplementedError("Changing index (time axis) is not supported")

    new_cols = reshape_fns.to_1d(col_mapper.values[0])  # array required
    records = nb.select_record_cols_nb(
        obj.records_arr,
        obj.col_index,
        new_cols
    )
    wrapper = ArrayWrapper.from_obj(col_mapper, freq=obj.wrapper.freq)
    return new_cols, records, wrapper


def _records_indexing_func(obj, pd_indexing_func):
    """See `indexing_on_records`."""
    new_cols, new_records, new_wrapper = indexing_on_records(obj, pd_indexing_func)
    if obj.grouper.group_by is not None:
        new_group_by = obj.grouper.group_by[new_cols]
    else:
        new_group_by = None
    return obj.__class__(
        new_records,
        new_wrapper,
        idx_field=obj.idx_field,
        group_by=new_group_by
    )


class Records(PandasIndexer):
    """Exposes methods and properties for working with records.

    Args:
        records_arr (array_like): A structured NumPy array of records.

            Must have the field `col` (column position in a matrix).
        wrapper (ArrayWrapper): Array wrapper of type `vectorbt.base.array_wrapper.ArrayWrapper`.
        idx_field (str): The name of the field corresponding to the index. Optional.

            Will be derived automatically if records contain field `'idx'`.
        group_by (int, str or array_like): Group columns by a mapper when reducing.

            See `vectorbt.base.index_fns.group_index`."""

    def __init__(self, records_arr, wrapper, idx_field=None, group_by=None):
        if not isinstance(records_arr, np.ndarray):
            records_arr = np.asarray(records_arr)
        checks.assert_not_none(records_arr.dtype.fields)
        checks.assert_value_in('col', records_arr.dtype.names)
        checks.assert_type(wrapper, ArrayWrapper)
        if idx_field is not None:
            checks.assert_value_in(idx_field, records_arr.dtype.names)
        else:
            if 'idx' in records_arr.dtype.names:
                idx_field = 'idx'

        self.records_arr = records_arr
        self.wrapper = wrapper
        self.idx_field = idx_field
        self.grouper = IndexGrouper(self.wrapper.columns, group_by=group_by)

        PandasIndexer.__init__(self, _records_indexing_func)

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

    def filter_by_mask(self, mask, idx_field=None, group_by=None):
        """Return a new class instance, filtered by mask."""
        if idx_field is None:
            idx_field = self.idx_field
        if group_by is None:
            group_by = self.grouper.group_by
        return self.__class__(
            self.records_arr[mask],
            self.wrapper,
            idx_field=idx_field,
            group_by=group_by
        )

    def map(self, map_func_nb, *args, idx_arr=None, group_by=None):
        """Map each record to a scalar value. Returns `MappedArray`.

        See `vectorbt.records.nb.map_records_nb`."""
        checks.assert_numba_func(map_func_nb)

        mapped_arr = nb.map_records_nb(self.records_arr, map_func_nb, *args)
        if idx_arr is None:
            if self.idx_field is not None:
                idx_arr = self.records_arr[self.idx_field]
            else:
                idx_arr = None
        if group_by is None:
            group_by = self.grouper.group_by
        return MappedArray(
            mapped_arr,
            self.records_arr['col'],
            self.wrapper,
            idx_arr=idx_arr,
            group_by=group_by
        )

    def map_field(self, field, idx_arr=None, group_by=None):
        """Convert field to `MappedArray`."""
        if idx_arr is None:
            if self.idx_field is not None:
                idx_arr = self.records_arr[self.idx_field]
            else:
                idx_arr = None
        if group_by is None:
            group_by = self.grouper.group_by
        return MappedArray(
            self.records_arr[field],
            self.records_arr['col'],
            self.wrapper,
            idx_arr=idx_arr,
            group_by=group_by
        )

    def map_array(self, a, idx_arr=None, group_by=None):
        """Convert array to `MappedArray`.

         The length of the array should match that of the records."""
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        checks.assert_same_shape(a, self.records_arr)

        if idx_arr is None:
            if self.idx_field is not None:
                idx_arr = self.records_arr[self.idx_field]
            else:
                idx_arr = None
        if group_by is None:
            group_by = self.grouper.group_by
        return MappedArray(
            a,
            self.records_arr['col'],
            self.wrapper,
            idx_arr=idx_arr,
            group_by=group_by
        )

    @cached_method
    def count(self, **kwargs):
        """Number of records."""
        return MappedArray(
            self.records_arr['col'],  # doesn't matter which column to take
            self.records_arr['col'],
            self.wrapper,
            group_by=self.grouper.group_by
        ).count(default_val=0., **kwargs)

