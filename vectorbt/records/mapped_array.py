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
>>> import vectorbt as vbt

>>> a = np.array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
>>> col_arr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
>>> idx_arr = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
>>> wrapper = vbt.ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b', 'c'], ndim=2, freq='1 day')
>>> ma = vbt.MappedArray(wrapper, a, col_arr, idx_arr=idx_arr)

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

>>> ma.reduce(min_max_reduce_nb, to_array=True, index=['min', 'max'])
        a     b     c
min  10.0  13.0  16.0
max  12.0  15.0  18.0

>>> @njit
... def idxmin_idxmax_reduce_nb(col, a):
...     return np.array([np.argmin(a), np.argmax(a)])

>>> ma.reduce(idxmin_idxmax_reduce_nb, to_array=True,
...     to_idx=True, index=['idxmin', 'idxmax'])
        a  b  c
idxmin  x  x  x
idxmax  z  z  z
```

## Conversion

You can convert any `MappedArray` instance to the matrix form:

* Given `idx_arr` was provided:

```python-repl
>>> ma.to_matrix()
      a     b     c
x  10.0  13.0  16.0
y  11.0  14.0  17.0
z  12.0  15.0  18.0
```

!!! note
    Will raise an error if there are multiple records pointing to the same matrix element.

* Given `group_by` was provided, index can be ignored, or there are position conflicts:

```python-repl
>>> ma.stack(group_by=np.array(['first', 'first', 'second']))
   first  second
0   10.0    16.0
1   11.0    17.0
2   12.0    18.0
3   13.0     NaN
4   14.0     NaN
5   15.0     NaN
```

## Filtering

Use `MappedArray.filter_by_mask` to filter elements per column/group:

```python-repl
>>> mask = [True, False, True, False, True, False, True, False, True]
>>> filtered_ma = ma.filter_by_mask(mask)
>>> filtered_ma.count()
a    2
b    1
c    2
dtype: int64

>>> filtered_ma.id_arr
array([0, 2, 4, 6, 8])
```

## Plotting

You can build histograms and boxplots of `MappedArray` directly:

```python-repl
>>> ma.box()
```

![](/vectorbt/docs/img/mapped_box.png)

To use scatterplots or any other plots that require index, convert to matrix first:

```python-repl
>>> ma.to_matrix().vbt.scatter()
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
>>> grouped_ma = vbt.MappedArray(grouped_wrapper, a, col_arr, idx_arr=idx_arr)

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

    If two `MappedArray` operands have different metadata, will copy metadata from the first one,
    but at least their `id_arr` and `col_arr` must match.

## Indexing

You can use pandas indexing on the `MappedArray` class, which will forward the indexing operation
to each `__init__` argument with index:

```python-repl
>>> ma['a'].values
array([10., 11., 12.])

>>> grouped_ma['first'].values
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
respectively. Caching can be disabled globally via `vectorbt.settings`.

!!! note
    Because of caching, class is meant to be immutable and all properties are read-only.
    To change any attribute, use the `copy` method and pass the attribute as keyword argument.
"""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_method
from vectorbt.utils.enum import to_value_map
from vectorbt.base.reshape_fns import to_1d
from vectorbt.base.class_helpers import (
    add_binary_magic_methods,
    add_unary_magic_methods,
    binary_magic_methods,
    unary_magic_methods
)
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping
from vectorbt.generic import nb as generic_nb
from vectorbt.records import nb
from vectorbt.records.col_mapper import ColumnMapper


def combine_mapped_with_other(self, other, np_func):
    """Combine `MappedArray` with other compatible object.

    If other object is also `MappedArray`, their `id_arr` and `col_arr` must match."""
    if isinstance(other, MappedArray):
        checks.assert_array_equal(self.id_arr, other.id_arr)
        checks.assert_array_equal(self.col_arr, other.col_arr)
        other = other.values
    return self.copy(mapped_arr=np_func(self.values, other))


@add_binary_magic_methods(
    binary_magic_methods,
    combine_mapped_with_other
)
@add_unary_magic_methods(
    unary_magic_methods,
    lambda self, np_func: self.copy(mapped_arr=np_func(self.values))
)
class MappedArray(Wrapping):
    """Exposes methods and properties for working with records.

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbt.base.array_wrapper.ArrayWrapper`.
        mapped_arr (array_like): A one-dimensional array of mapped record values.
        col_arr (array_like): A one-dimensional column array.

            Must be of the same size as `mapped_arr`.
        id_arr (array_like): A one-dimensional id array. Defaults to simple range.

            Must be of the same size as `mapped_arr`.
        idx_arr (array_like): A one-dimensional index array. Optional.

            Must be of the same size as `mapped_arr`.
        value_map (namedtuple, dict or callable): Value map.
        **kwargs: Custom keyword arguments passed to the config.

            Useful if any subclass wants to extend the config.
    """

    def __init__(self, wrapper, mapped_arr, col_arr, id_arr=None, idx_arr=None, value_map=None, **kwargs):
        Wrapping.__init__(
            self,
            wrapper,
            mapped_arr=mapped_arr,
            col_arr=col_arr,
            id_arr=id_arr,
            idx_arr=idx_arr,
            value_map=value_map,
            **kwargs
        )
        mapped_arr = np.asarray(mapped_arr)
        col_arr = np.asarray(col_arr)
        checks.assert_shape_equal(mapped_arr, col_arr, axis=0)
        if id_arr is None:
            id_arr = np.arange(len(mapped_arr))
        if idx_arr is not None:
            idx_arr = np.asarray(idx_arr)
            checks.assert_shape_equal(mapped_arr, idx_arr, axis=0)
        if value_map is not None:
            if checks.is_namedtuple(value_map):
                value_map = to_value_map(value_map)

        self._mapped_arr = mapped_arr
        self._id_arr = id_arr
        self._col_arr = col_arr
        self._idx_arr = idx_arr
        self._value_map = value_map
        self._col_mapper = ColumnMapper(wrapper, col_arr)

    def _indexing_func_meta(self, pd_indexing_func):
        """Perform indexing on `MappedArray` and return metadata."""
        new_wrapper, _, group_idxs, col_idxs = \
            self.wrapper._indexing_func_meta(pd_indexing_func, column_only_select=True)
        new_indices, new_col_arr = self.col_mapper._col_idxs_meta(col_idxs)
        new_mapped_arr = self.values[new_indices]
        new_id_arr = self.id_arr[new_indices]
        if self.idx_arr is not None:
            new_idx_arr = self.idx_arr[new_indices]
        else:
            new_idx_arr = None
        return new_wrapper, new_mapped_arr, new_col_arr, new_id_arr, new_idx_arr, group_idxs, col_idxs

    def _indexing_func(self, pd_indexing_func):
        """Perform indexing on `MappedArray`."""
        new_wrapper, new_mapped_arr, new_col_arr, new_id_arr, new_idx_arr, _, _ = \
            self._indexing_func_meta(pd_indexing_func)
        return self.copy(
            wrapper=new_wrapper,
            mapped_arr=new_mapped_arr,
            col_arr=new_col_arr,
            id_arr=new_id_arr,
            idx_arr=new_idx_arr
        )

    @property
    def mapped_arr(self):
        """Mapped array."""
        return self._mapped_arr

    values = mapped_arr

    def __len__(self):
        return len(self.values)

    @property
    def col_arr(self):
        """Column array."""
        return self._col_arr

    @property
    def col_mapper(self):
        """Column mapper.

        See `vectorbt.records.col_mapper.ColumnMapper`."""
        return self._col_mapper

    @property
    def id_arr(self):
        """Id array."""
        return self._id_arr

    @property
    def idx_arr(self):
        """Index array."""
        return self._idx_arr

    @property
    def value_map(self):
        """Value map."""
        return self._value_map

    @cached_method
    def is_sorted(self, incl_id=False):
        """Check whether mapped array is sorted."""
        if incl_id:
            return nb.is_col_idx_sorted_nb(self.col_arr, self.id_arr)
        return nb.is_col_sorted_nb(self.col_arr)

    def sort(self, incl_id=False, idx_arr=None, group_by=None, **kwargs):
        """Sort mapped array by column array (primary) and id array (secondary, optional)."""
        if idx_arr is None:
            idx_arr = self.idx_arr
        if self.is_sorted(incl_id=incl_id):
            return self.copy(idx_arr=idx_arr, **kwargs).regroup(group_by)
        if incl_id:
            ind = np.lexsort((self.id_arr, self.col_arr))  # expensive!
        else:
            ind = np.argsort(self.col_arr)
        return self.copy(
            mapped_arr=self.values[ind],
            col_arr=self.col_arr[ind],
            id_arr=self.id_arr[ind],
            idx_arr=idx_arr[ind] if idx_arr is not None else None,
            **kwargs
        ).regroup(group_by)

    def filter_by_mask(self, mask, idx_arr=None, group_by=None, **kwargs):
        """Return a new class instance, filtered by mask."""
        if idx_arr is None:
            idx_arr = self.idx_arr
        return self.copy(
            mapped_arr=self.values[mask],
            col_arr=self.col_arr[mask],
            id_arr=self.id_arr[mask],
            idx_arr=idx_arr[mask] if idx_arr is not None else None,
            **kwargs
        ).regroup(group_by)

    def map_to_mask(self, inout_map_func_nb, *args, group_by=None):
        """Map mapped array to a mask.

        See `vectorbt.records.nb.mapped_to_mask_nb`."""
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        return nb.mapped_to_mask_nb(self.values, col_map, inout_map_func_nb, *args)

    @cached_method
    def top_n_mask(self, n, **kwargs):
        """Return mask of top N elements in each column."""
        return self.map_to_mask(nb.top_n_inout_map_nb, n, **kwargs)

    @cached_method
    def bottom_n_mask(self, n, **kwargs):
        """Return mask of bottom N elements in each column."""
        return self.map_to_mask(nb.bottom_n_inout_map_nb, n, **kwargs)

    @cached_method
    def top_n(self, n, **kwargs):
        """Filter top N elements from each column."""
        return self.filter_by_mask(self.top_n_mask(n), **kwargs)

    @cached_method
    def bottom_n(self, n, **kwargs):
        """Filter bottom N elements from each column."""
        return self.filter_by_mask(self.bottom_n_mask(n), **kwargs)

    @cached_method
    def is_matrix_compatible(self, idx_arr=None, group_by=None):
        """See `vectorbt.records.nb.mapped_matrix_compatible_nb`."""
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        return nb.mapped_matrix_compatible_nb(col_arr, idx_arr, target_shape)

    def to_matrix(self, idx_arr=None, default_val=np.nan, group_by=None, **kwargs):
        """Convert mapped array to the matrix form.

        See `vectorbt.records.nb.mapped_to_matrix_nb`.

        !!! note
            Will raise an error if there are multiple values pointing to the same matrix element.

        !!! warning
            Mapped arrays represent information in the most memory-friendly format.
            Mapping back to the matrix form may occupy lots of memory if records are sparse."""
        if idx_arr is None:
            if self.idx_arr is None:
                raise ValueError("Must pass idx_arr")
            idx_arr = self.idx_arr
        if not self.is_matrix_compatible(idx_arr=idx_arr, group_by=group_by):
            raise ValueError("Multiple values are pointing to the same matrix element")
        col_arr = self.col_mapper.get_col_arr(group_by=group_by)
        target_shape = self.wrapper.get_shape_2d(group_by=group_by)
        out = nb.mapped_to_matrix_nb(self.values, col_arr, idx_arr, target_shape, default_val)
        return self.wrapper.wrap(out, group_by=group_by, **kwargs)

    def reduce(self, reduce_func_nb, *args, idx_arr=None, to_array=False, to_idx=False,
               idx_labeled=True, default_val=np.nan, group_by=None, **kwargs):
        """Reduce mapped array by column.

        If `to_array` is False and `to_idx` is False, see `vectorbt.records.nb.reduce_mapped_nb`.
        If `to_array` is False and `to_idx` is True, see `vectorbt.records.nb.reduce_mapped_to_idx_nb`.
        If `to_array` is True and `to_idx` is False, see `vectorbt.records.nb.reduce_mapped_to_array_nb`.
        If `to_array` is True and `to_idx` is True, see `vectorbt.records.nb.reduce_mapped_to_idx_array_nb`.

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
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        if not to_array:
            if not to_idx:
                out = nb.reduce_mapped_nb(
                    self.values,
                    col_map,
                    default_val,
                    reduce_func_nb,
                    *args
                )
            else:
                out = nb.reduce_mapped_to_idx_nb(
                    self.values,
                    col_map,
                    idx_arr,
                    default_val,
                    reduce_func_nb,
                    *args
                )
        else:
            if not to_idx:
                out = nb.reduce_mapped_to_array_nb(
                    self.values,
                    col_map,
                    default_val,
                    reduce_func_nb,
                    *args
                )
            else:
                out = nb.reduce_mapped_to_idx_array_nb(
                    self.values,
                    col_map,
                    idx_arr,
                    default_val,
                    reduce_func_nb,
                    *args
                )

        # Perform post-processing
        if to_idx:
            nan_mask = np.isnan(out)
            if idx_labeled:
                out = out.astype(np.object)
                out[~nan_mask] = self.wrapper.index[out[~nan_mask].astype(np.int_)]
            else:
                out[nan_mask] = -1
                out = out.astype(np.int_)
        return self.wrapper.wrap_reduced(out, group_by=group_by, **kwargs)

    @cached_method
    def nst(self, n, **kwargs):
        """Return nst element of each column."""
        return self.reduce(generic_nb.nst_reduce_nb, n, to_array=False, to_idx=False, **kwargs)

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
            percentiles = to_1d(percentiles, raw=True)
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

    @cached_method
    def count(self, group_by=None, **kwargs):
        """Return count by column."""
        return self.wrapper.wrap_reduced(
            self.col_mapper.get_col_map(group_by=group_by)[1],
            group_by=group_by,
            **kwargs
        )

    @cached_method
    def value_counts(self, group_by=None, value_map=None, **kwargs):
        """Return a pandas object containing counts of unique values."""
        mapped_codes, mapped_uniques = pd.factorize(self.values)
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        value_counts = nb.mapped_value_counts_nb(mapped_codes, col_map)
        value_counts_df = self.wrapper.wrap(
            value_counts,
            index=mapped_uniques,
            group_by=group_by,
            **kwargs
        )
        if value_map is None:
            value_map = self.value_map
        if value_map is not None:
            if checks.is_namedtuple(value_map):
                value_map = to_value_map(value_map)
            value_counts_df.index = value_counts_df.index.map(value_map)
        return value_counts_df

    def stack(self, group_by=None, default_val=np.nan, **kwargs):
        """Stack into a matrix.

        Will lose index information and fill missing values with `default_val`."""
        if self.wrapper.ndim == 1:
            return self.wrapper.wrap(
                self.values,
                index=np.arange(len(self.values)),
                group_by=group_by,
                **kwargs
            )
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        out = nb.stack_mapped_nb(self.values, col_map, default_val)
        return self.wrapper.wrap(out, index=np.arange(out.shape[0]), group_by=group_by, **kwargs)

    def hist(self, group_by=None, **kwargs):  # pragma: no cover
        """Plot histogram by column."""
        return self.stack(group_by=group_by).vbt.hist(**kwargs)

    def box(self, group_by=None, **kwargs):  # pragma: no cover
        """Plot box plot by column."""
        return self.stack(group_by=group_by).vbt.box(**kwargs)



