"""Base class for working with mapped arrays.

This class takes the mapped array and the corresponding column and (optionally) index arrays,
and offers features to directly process the mapped array without converting it to the matrix form;
for example, to compute various statistics by column, such as standard deviation.

## Reducing

Using `MappedArray`, you can then reduce by column as follows:

* Use already provided reducers such as `MappedArray.mean`:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from vectorbt.base.array_wrapper import ArrayWrapper
>>> from vectorbt.records.mapped_array import MappedArray

>>> a = np.array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
>>> col_arr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
>>> idx_arr = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
>>> wrapper = ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b', 'c'], ndim=2, freq='1 day')
>>> ma = MappedArray(wrapper, a, col_arr, idx_arr=idx_arr)

>>> ma.mean()
a    11.0
b    14.0
c    17.0
dtype: float64
```

* Use `MappedArray.to_matrix` to map to a matrix and then reduce manually (expensive):

```python-repl
>>> ma.to_matrix().mean()
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

>>> ma.reduce(pow_mean_reduce_nb, 2)
a    121.666667
b    196.666667
c    289.666667
dtype: float64

>>> @njit
... def min_max_reduce_nb(col, a):
...     return np.array([np.min(a), np.max(a)])

>>> ma.reduce(min_max_reduce_nb, to_array=True,
...     n_rows=2, index=['min', 'max'])
        a     b     c
min  10.0  13.0  16.0
max  12.0  15.0  18.0

>>> @njit
... def idxmin_idxmax_reduce_nb(col, a):
...     return np.array([np.argmin(a), np.argmax(a)])

>>> ma.reduce(idxmin_idxmax_reduce_nb, to_array=True,
...     n_rows=2, to_idx=True, index=['idxmin', 'idxmax'])
        a  b  c
idxmin  x  x  x
idxmax  z  z  z
```

## Conversion

You can convert any `MappedArray` instance to the matrix form, given `idx_arr` was provided:

```python-repl
>>> ma.to_matrix()
      a     b     c
x  10.0  13.0  16.0
y  11.0  14.0  17.0
z  12.0  15.0  18.0
```

!!! note
    Will raise an error if there are multiple records pointing to the same matrix element.

## Plotting

You can build histograms and boxplots of `MappedArray` directly:

```python-repl
>>> ma.box()
```

![](/vectorbt/docs/img/mapped_box.png)

To use scatterplots or any other plots that require index, convert to matrix first:

```python-repl
>>> ma.to_matrix().vbt.scatter().show_png()
```

![](/vectorbt/docs/img/mapped_scatter.png)

## Grouping

One of the key features of `MappedArray` is that you can perform reducing operations on a group
of columns as if they were a single column. Groups can be specified by `group_by`, which
can be anything from positions or names of column levels, to a NumPy array with actual groups.

There are multiple ways of define grouping:

* When creating `MappedArray`, pass `group_by` to `vectorbt.base.array_wrapper.ArrayWrapper`:

```python-repl
>>> group_by = np.array(['first', 'first', 'second'])
>>> grouped_wrapper = wrapper.copy(group_by=group_by)
>>> grouped_ma = MappedArray(grouped_wrapper, a, col_arr, idx_arr=idx_arr)

>>> grouped_ma.mean()
first     12.5
second    17.0
dtype: float64
```

* Regroup an existing `MappedArray`:

```python-repl
>>> ma.regroup(group_by).mean()
first     12.5
second    17.0
dtype: float64
```

* Pass `group_by` directly to the reducing method:

```python-repl
>>> ma.mean(group_by=group_by)
first     12.5
second    17.0
dtype: float64
```

By the same way you can disable or modify any existing grouping:

```python-repl
>>> grouped_ma.mean(group_by=False)
a    11.0
b    14.0
c    17.0
dtype: float64
```

!!! note
    Grouping applies only to reducing operations, there is no change to the arrays.

## Operators

`MappedArray` implements arithmetic, comparison and logical operators. You can perform basic
operations (such as addition) on mapped arrays as if they were NumPy arrays.

```python-repl
>>> ma ** 2
<vectorbt.records.mapped_array.MappedArray at 0x7f97bfc49358>

>>> ma * np.array([1, 2, 3, 4, 5, 6])
<vectorbt.records.mapped_array.MappedArray at 0x7f97bfc65e80>

>>> ma + ma
<vectorbt.records.mapped_array.MappedArray at 0x7fd638004d30>
```

!!! note
    You should ensure that your `MappedArray` operand is on the left if the other operand is an array.

    If two `MappedArray` operands have different metadata, will copy metadata from the first one.

## Indexing

You can use pandas indexing on the `MappedArray` class, which will forward the indexing operation
to each `__init__` argument with index:

```python-repl
>>> ma['a'].mapped_arr
array([10., 11., 12.])

>>> grouped_ma['first'].mapped_arr
array([10., 11., 12., 13., 14., 15.])
```

!!! note
    Changing index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame; for example, use `some_field.iloc[0]` instead of `some_field.iloc[:, 0]`.

    Indexing behavior depends solely upon `vectorbt.base.array_wrapper.ArrayWrapper`.
    For example, if `group_select` is enabled indexing will be performed on groups,
    otherwise on single columns.

## Caching

`MappedArray` supports caching. If a method or a property requires heavy computation, it's wrapped
with `vectorbt.utils.decorators.cached_method` and `vectorbt.utils.decorators.cached_property`
respectively. Caching can be disabled globally via `vectorbt.defaults` or locally via the
method/property. There is currently no way to disable caching for an entire class.

!!! note
    Because of caching, class is meant to be immutable and all properties are read-only.
    To change any attribute, use the `copy` method and pass the attribute as keyword argument.

!!! note
    This class is meant to be immutable. To change any attribute, use `MappedArray.copy`."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import Configured
from vectorbt.utils.enum import to_value_map
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
        other = other.mapped_arr
    return self.copy(mapped_arr=np_func(self.mapped_arr, other))


@add_binary_magic_methods(
    binary_magic_methods,
    _mapped_binary_translate_func
)
@add_unary_magic_methods(
    unary_magic_methods,
    lambda self, np_func: self.copy(mapped_arr=np_func(self.mapped_arr))
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
        value_map (dict or namedtuple): Value map.
        filter_ids (set): IDs of applied filters.

            Prevents applying same filters again and calling contradictive attributes.
        **kwargs: Custom keyword arguments passed to the config.

            Useful if any subclass wants to extend the config.
    """

    def __init__(self, wrapper, mapped_arr, col_arr, idx_arr=None, value_map=None, filter_id=None, **kwargs):
        Configured.__init__(
            self,
            wrapper=wrapper,
            mapped_arr=mapped_arr,
            col_arr=col_arr,
            idx_arr=idx_arr,
            value_map=value_map,
            filter_id=filter_id,
            **kwargs
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
        if value_map is not None:
            if checks.is_namedtuple(value_map):
                value_map = to_value_map(value_map)
            checks.assert_type(value_map, dict)
        if filter_id is None:
            filter_id = set()

        self._wrapper = wrapper
        self._mapped_arr = mapped_arr
        self._col_arr = col_arr
        self._idx_arr = idx_arr
        self._value_map = value_map
        self._filter_id = filter_id

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

    def force_select_column(self, column=None):
        """Force selection of one column."""
        if column is not None:
            if self.wrapper.grouper.group_by is None:
                self_col = self[column]
            else:
                self_col = self.regroup(False)[column]
        else:
            self_col = self
        if self_col.wrapper.ndim > 1:
            raise TypeError("Only one column is allowed. Use indexing or column argument.")
        return self_col

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

    @property
    def value_map(self):
        """Value map."""
        return self._value_map

    @property
    def filter_ids(self):
        """IDs of applied filters."""
        return self._filter_ids

    @cached_property
    def col_index(self):
        """Column index for `MappedArray.mapped_arr`."""
        return nb.mapped_col_index_nb(self.mapped_arr, self.col_arr, len(self.wrapper.columns))

    def filter_by_mask(self, mask, idx_arr=None, value_map=None, group_by=None, filter_id=None, **kwargs):
        """Return a new class instance, filtered by mask.

        To prohibit using the same filter or filter class on the filtered instance, provide `filter_id`."""
        if idx_arr is None:
            idx_arr = self.idx_arr
        if idx_arr is not None:
            idx_arr = self.idx_arr[mask]
        if value_map is None:
            value_map = self.value_map
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            wrapper = self.wrapper.copy(group_by=group_by)
        else:
            wrapper = self.wrapper
        filter_ids = self.filter_ids.copy()
        if filter_id is not None:
            if filter_id in self.filter_ids:
                raise ValueError(f"Filter \"{filter_id}\" already applied")
            filter_ids |= {filter_id}
        return self.copy(
            wrapper=wrapper,
            mapped_arr=self.mapped_arr[mask],
            col_arr=self.col_arr[mask],
            idx_arr=idx_arr,
            value_map=value_map,
            filter_ids=filter_ids,
            **kwargs
        )

    def map_to_mask(self, inout_map_func_nb, *args):
        """Map mapped array to a mask.

        See `vectorbt.records.nb.mapped_to_mask_nb`."""
        return nb.mapped_to_mask_nb(self.mapped_arr, self.col_arr, inout_map_func_nb, *args)

    def top_n_mask(self, n):
        """Return mask of top N elements in each column."""
        return self.map_to_mask(nb.top_n_inout_map_nb, n)

    def bottom_n_mask(self, n):
        """Return mask of bottom N elements in each column."""
        return self.map_to_mask(nb.bottom_n_inout_map_nb, n)

    def top_n(self, n, **kwargs):
        """Filter top N elements from each column."""
        return self.filter_by_mask(self.top_n_mask(n), **kwargs)

    def bottom_n(self, n, **kwargs):
        """Filter bottom N elements from each column."""
        return self.filter_by_mask(self.top_n_mask(n), **kwargs)

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
        out = nb.mapped_to_matrix_nb(self.mapped_arr, self.col_arr, idx_arr, target_shape, default_val)
        return self.wrapper.wrap(out, group_by=False)

    def reduce(self, reduce_func_nb, *args, idx_arr=None, to_array=False, n_rows=None,
               to_idx=False, idx_labeled=True, default_val=np.nan, group_by=None, **kwargs):
        """Reduce mapped array by column.

        If `to_array` is False and `to_idx` is False, see `vectorbt.records.nb.reduce_mapped_nb`.
        If `to_array` is False and `to_idx` is True, see `vectorbt.records.nb.reduce_mapped_to_idx_nb`.
        If `to_array` is True and `to_idx` is False, see `vectorbt.records.nb.reduce_mapped_to_array_nb`.
        If `to_array` is True and `to_idx` is True, see `vectorbt.records.nb.reduce_mapped_to_idx_array_nb`.

        If `to_array` is True, must pass `n_rows` indicating the number of elements in the array.
        If `to_idx` is True, must pass `idx_arr`. Set `idx_labeled` to False to return raw positions instead
        of labels. Use `default_val` to set the default value. Set `group_by` to False to disable grouping.

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
                out = nb.reduce_mapped_nb(
                    self.mapped_arr,
                    col_arr,
                    len(columns),
                    default_val,
                    reduce_func_nb,
                    *args
                )
            else:
                out = nb.reduce_mapped_to_idx_nb(
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
                out = nb.reduce_mapped_to_array_nb(
                    self.mapped_arr,
                    col_arr,
                    len(columns),
                    n_rows,
                    default_val,
                    reduce_func_nb,
                    *args
                )
            else:
                out = nb.reduce_mapped_to_idx_array_nb(
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
                out_shape = out.shape
                out = out.flatten()
                mask = np.isnan(out)
                if mask.any():
                    # Contains NaNs
                    out[mask] = 0
                    out = out.astype(int)
                    out = self.wrapper.index[out].to_numpy()
                    out = out.astype(np.object)
                    out[mask] = np.nan
                else:
                    out = self.wrapper.index[out.astype(int)].to_numpy()
                out = np.reshape(out, out_shape)
            else:
                mask = np.isnan(out)
                out[mask] = -1
                out = out.astype(int)
        return self.wrapper.wrap_reduced(out, group_by=group_by, **kwargs)

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
    def count(self, default_val=0., dtype=np.int_, **kwargs):
        """Return count by column."""
        return self.reduce(
            generic_nb.count_reduce_nb,
            to_array=False,
            to_idx=False,
            default_val=default_val,
            dtype=dtype,
            **kwargs
        )

    @cached_method
    def idxmin(self, **kwargs):
        """Return index of min by column."""
        return self.reduce(generic_nb.argmin_reduce_nb, to_array=False, to_idx=True, **kwargs)

    @cached_method
    def idxmax(self, **kwargs):
        """Return index of max by column."""
        return self.reduce(generic_nb.argmax_reduce_nb, to_array=False, to_idx=True, **kwargs)

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
        out = self.reduce(
            generic_nb.describe_reduce_nb,
            percentiles,
            ddof,
            to_array=True,
            n_rows=len(index),
            to_idx=False,
            index=index,
            **kwargs
        )
        if isinstance(out, pd.DataFrame):
            out.loc['count'].fillna(0., inplace=True)
        else:
            if np.isnan(out.loc['count']):
                out.loc['count'] = 0.
        return out

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

    def value_counts(self, group_by=None, value_map=None):
        """Return a pandas object containing counts of unique values."""
        group_arr, columns = self.wrapper.grouper.get_groups_and_columns(group_by=group_by)
        unique_vals = np.unique(self.mapped_arr)
        counts_df = pd.DataFrame(np.full((len(unique_vals), len(columns)), 0), columns=columns, index=unique_vals)
        if group_arr is not None:
            col_arr = group_arr[self.col_arr]
        else:
            col_arr = self.col_arr
        for col in range(len(columns)):
            masked_arr = self.mapped_arr[col_arr == col]
            masked_unique, masked_counts = np.unique(masked_arr, return_counts=True)
            counts_df.loc[masked_unique, columns[col]] = masked_counts
        if value_map is None:
            value_map = self.value_map
        if value_map is not None:
            counts_df.index = counts_df.index.map(value_map)
        return counts_df
