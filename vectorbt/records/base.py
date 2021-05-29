"""Base class for working with records.

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
      id  idx  col  attr1  attr2
0      0    0    0      1      9
1      1    1    0      2     10
2      2    3    0      4     12
3      3    0    1      5     13
4      4    1    1      7     15
5      5    3    1      8     16
```

Another advantage of records is that they are not constrained by size. Multiple records can map
to a single element in a matrix. For example, one can define multiple orders at the same time step,
which is impossible to represent in a matrix form without using complex data types.

## Mapping

`Records` are just [structured arrays](https://numpy.org/doc/stable/user/basics.rec.html) with a bunch
of methods and properties for processing them. Their main feature is to map the records array and
to reduce it by column (similar to the MapReduce paradigm). The main advantage is that it all happens
without conversion to the matrix form and wasting memory resources.

Consider the following example:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from collections import namedtuple
>>> import vectorbt as vbt

>>> example_dt = np.dtype([
...     ('id', np.int64),
...     ('idx', np.int64),
...     ('col', np.int64),
...     ('some_field', np.float64)
... ])
>>> records_arr = np.array([
...     (0, 0, 0, 10.),
...     (1, 1, 0, 11.),
...     (2, 2, 0, 12.),
...     (3, 0, 1, 13.),
...     (4, 1, 1, 14.),
...     (5, 2, 1, 15.),
...     (6, 0, 2, 16.),
...     (7, 1, 2, 17.),
...     (8, 2, 2, 18.)
... ], dtype=example_dt)
>>> wrapper = vbt.ArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b', 'c'], ndim=2, freq='1 day')
>>> records = vbt.Records(wrapper, records_arr)

>>> records.records
   id  idx  col  some_field
0   0    0    0        10.0
1   1    1    0        11.0
2   2    2    0        12.0
3   3    0    1        13.0
4   4    1    1        14.0
5   5    2    1        15.0
6   6    0    2        16.0
7   7    1    2        17.0
8   8    2    2        18.0
```

`Records` can be mapped to `vectorbt.records.mapped_array.MappedArray` in several ways:

* Use `Records.map_field` to map a record field:

```python-repl
>>> records.map_field('some_field')
<vectorbt.records.mapped_array.MappedArray at 0x7ff49bd31a58>

>>> records.map_field('some_field').values
array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
```

* Use `Records.map` to map records using a custom function.

```python-repl
>>> @njit
... def power_map_nb(record, pow):
...     return record.some_field ** pow

>>> records.map(power_map_nb, 2)
<vectorbt.records.mapped_array.MappedArray at 0x7ff49c990cf8>

>>> records.map(power_map_nb, 2).values
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])
```

* Use `Records.map_array` to convert an array to `vectorbt.records.mapped_array.MappedArray`.

```python-repl
>>> records.map_array(records_arr['some_field'] ** 2)
<vectorbt.records.mapped_array.MappedArray object at 0x7fe9bccf2978>

>>> records.map_array(records_arr['some_field'] ** 2).values
array([100., 121., 144., 169., 196., 225., 256., 289., 324.])
```

## Filtering

Use `Records.filter_by_mask` to filter elements per column/group:

```python-repl
>>> mask = [True, False, True, False, True, False, True, False, True]
>>> filtered_records = records.filter_by_mask(mask)
>>> filtered_records.count()
a    2
b    1
c    2
dtype: int64

>>> filtered_records.values['id']
array([0, 2, 4, 6, 8])
```

## Grouping

One of the key features of `Records` is that you can perform reducing operations on a group
of columns as if they were a single column. Groups can be specified by `group_by`, which
can be anything from positions or names of column levels, to a NumPy array with actual groups.

There are multiple ways of define grouping:

* When creating `Records`, pass `group_by` to `vectorbt.base.array_wrapper.ArrayWrapper`:

```python-repl
>>> group_by = np.array(['first', 'first', 'second'])
>>> grouped_wrapper = wrapper.copy(group_by=group_by)
>>> grouped_records = vbt.Records(grouped_wrapper, records_arr)

>>> grouped_records.map_field('some_field').mean()
first     12.5
second    17.0
dtype: float64
```

* Regroup an existing `Records`:

```python-repl
>>> records.regroup(group_by).map_field('some_field').mean()
first     12.5
second    17.0
dtype: float64
```

* Pass `group_by` directly to the mapping method:

```python-repl
>>> records.map_field('some_field', group_by=group_by).mean()
first     12.5
second    17.0
dtype: float64
```

* Pass `group_by` directly to the reducing method:

```python-repl
>>> records.map_field('some_field').mean(group_by=group_by)
a    11.0
b    14.0
c    17.0
dtype: float64
```

!!! note
    Grouping applies only to reducing operations, there is no change to the arrays.

## Indexing

Like any other class subclassing `vectorbt.base.array_wrapper.Wrapping`, we can do pandas indexing
on a `Records` instance, which forwards indexing operation to each object with columns:

```python-repl
>>> records['a'].records
   id  idx  col  some_field
0   0    0    0        10.0
1   1    1    0        11.0
2   2    2    0        12.0

>>> grouped_records['first'].records
   id  idx  col  some_field
0   0    0    0        10.0
1   1    1    0        11.0
2   2    2    0        12.0
3   3    0    1        13.0
4   4    1    1        14.0
5   5    2    1        15.0
```

!!! note
    Changing index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame; for example, use `some_field.iloc[0]` instead of `some_field.iloc[:, 0]`.

    Indexing behavior depends solely upon `vectorbt.base.array_wrapper.ArrayWrapper`.
    For example, if `group_select` is enabled indexing will be performed on groups,
    otherwise on single columns.

## Caching

`Records` supports caching. If a method or a property requires heavy computation, it's wrapped
with `vectorbt.utils.decorators.cached_method` and `vectorbt.utils.decorators.cached_property`
respectively. Caching can be disabled globally via `caching` in `vectorbt._settings.settings`.

!!! note
    Because of caching, class is meant to be immutable and all properties are read-only.
    To change any attribute, use the `copy` method and pass the attribute as keyword argument.

## Saving and loading

Like any other class subclassing `vectorbt.utils.config.Pickleable`, we can save a `Records`
instance to the disk with `Records.save` and load it with `Records.load`.
"""

import numpy as np
import pandas as pd

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_method
from vectorbt.utils.config import merge_dicts
from vectorbt.base.reshape_fns import to_1d
from vectorbt.base.array_wrapper import ArrayWrapper, Wrapping
from vectorbt.records import nb
from vectorbt.records.mapped_array import MappedArray
from vectorbt.records.col_mapper import ColumnMapper


RecordsT = tp.TypeVar("RecordsT", bound="Records")
IndexingMetaT = tp.Tuple[ArrayWrapper, tp.RecordArray, tp.MaybeArray, tp.Array1d]


class Records(Wrapping):
    """Wraps the actual records array (such as trades) and exposes methods for mapping
    it to some array of values (such as P&L of each trade).

    Args:
        wrapper (ArrayWrapper): Array wrapper.

            See `vectorbt.base.array_wrapper.ArrayWrapper`.
        records_arr (array_like): A structured NumPy array of records.

            Must have the fields `id` (record index) and `col` (column index).
        idx_field (str): The name of the field corresponding to the index. Optional.

            Searches for a field with name 'idx' if `idx_field` is 'auto'.
            Throws an error if the name was provided explicitly and the field cannot be found.
        **kwargs: Custom keyword arguments passed to the config.

            Useful if any subclass wants to extend the config.
    """

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 idx_field: str = 'auto',
                 **kwargs) -> None:
        Wrapping.__init__(
            self,
            wrapper,
            records_arr=records_arr,
            idx_field=idx_field,
            **kwargs
        )
        records_arr = np.asarray(records_arr)
        checks.assert_not_none(records_arr.dtype.fields)
        checks.assert_in('id', records_arr.dtype.names)
        checks.assert_in('col', records_arr.dtype.names)
        if idx_field == 'auto':
            if 'idx' in records_arr.dtype.names:
                idx_field = 'idx'
        elif idx_field is not None:
            checks.assert_in(idx_field, records_arr.dtype.names)

        self._records_arr = records_arr
        self._idx_field = idx_field
        self._col_mapper = ColumnMapper(wrapper, records_arr['col'])

    def get_by_col_idxs(self, col_idxs: tp.Array1d) -> tp.RecordArray:
        """Get records corresponding to column indices.

        Returns new records array."""
        if self.col_mapper.is_sorted():
            new_records_arr = nb.record_col_range_select_nb(
                self.values, self.col_mapper.col_range, to_1d(col_idxs))  # faster
        else:
            new_records_arr = nb.record_col_map_select_nb(
                self.values, self.col_mapper.col_map, to_1d(col_idxs))
        return new_records_arr

    def indexing_func_meta(self, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> IndexingMetaT:
        """Perform indexing on `Records` and return metadata."""
        new_wrapper, _, group_idxs, col_idxs = \
            self.wrapper.indexing_func_meta(pd_indexing_func, column_only_select=True, **kwargs)
        new_records_arr = self.get_by_col_idxs(col_idxs)
        return new_wrapper, new_records_arr, group_idxs, col_idxs

    def indexing_func(self: RecordsT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> RecordsT:
        """Perform indexing on `Records`."""
        new_wrapper, new_records_arr, _, _ = self.indexing_func_meta(pd_indexing_func, **kwargs)
        return self.copy(
            wrapper=new_wrapper,
            records_arr=new_records_arr
        )

    @property
    def records_arr(self) -> tp.RecordArray:
        """Records array."""
        return self._records_arr

    @property
    def values(self) -> tp.RecordArray:
        """Records array."""
        return self.records_arr

    def __len__(self) -> int:
        return len(self.values)

    @property
    def idx_field(self) -> str:
        """Index field."""
        return self._idx_field

    @property
    def records(self) -> tp.Frame:
        """Records."""
        return pd.DataFrame.from_records(self.values)

    @property
    def recarray(self) -> tp.RecArray:
        return self.values.view(np.recarray)

    @property
    def col_mapper(self) -> ColumnMapper:
        """Column mapper.

        See `vectorbt.records.col_mapper.ColumnMapper`."""
        return self._col_mapper

    @cached_method
    def is_sorted(self, incl_id: bool = False) -> bool:
        """Check whether records are sorted."""
        if incl_id:
            return nb.is_col_idx_sorted_nb(self.values['col'], self.values['id'])
        return nb.is_col_sorted_nb(self.values['col'])

    def sort(self: RecordsT, incl_id: bool = False, group_by: tp.GroupByLike = None, **kwargs) -> RecordsT:
        """Sort records by columns (primary) and ids (secondary, optional).

        !!! note
            Sorting is expensive. A better approach is to append records already in the correct order."""
        if self.is_sorted(incl_id=incl_id):
            return self.copy(**kwargs).regroup(group_by)
        if incl_id:
            ind = np.lexsort((self.values['id'], self.values['col']))  # expensive!
        else:
            ind = np.argsort(self.values['col'])
        return self.copy(records_arr=self.values[ind], **kwargs).regroup(group_by)

    def filter_by_mask(self: RecordsT, mask: tp.Array1d, group_by: tp.GroupByLike = None, **kwargs) -> RecordsT:
        """Return a new class instance, filtered by mask."""
        return self.copy(records_arr=self.values[mask], **kwargs).regroup(group_by)

    def map(self, map_func_nb: tp.RecordMapFunc, *args, idx_field: tp.Optional[str] = None,
            value_map: tp.Optional[tp.ValueMapLike] = None, group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Map each record to a scalar value. Returns mapped array.

        See `vectorbt.records.nb.map_records_nb`."""
        checks.assert_numba_func(map_func_nb)
        mapped_arr = nb.map_records_nb(self.values, map_func_nb, *args)
        if idx_field is None:
            idx_field = self.idx_field
        if idx_field is not None:
            idx_arr = self.values[idx_field]
        else:
            idx_arr = None
        return MappedArray(
            self.wrapper,
            mapped_arr,
            self.values['col'],
            id_arr=self.values['id'],
            idx_arr=idx_arr,
            value_map=value_map,
            **kwargs
        ).regroup(group_by)

    def map_field(self, field: str, idx_field: tp.Optional[str] = None,
                  value_map: tp.Optional[tp.ValueMapLike] = None,
                  group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Convert field to mapped array."""
        if idx_field is None:
            idx_field = self.idx_field
        if idx_field is not None:
            idx_arr = self.values[idx_field]
        else:
            idx_arr = None
        return MappedArray(
            self.wrapper,
            self.values[field],
            self.values['col'],
            id_arr=self.values['id'],
            idx_arr=idx_arr,
            value_map=value_map,
            **kwargs
        ).regroup(group_by)

    def map_array(self, a: tp.ArrayLike, idx_field: tp.Optional[str] = None,
                  value_map: tp.Optional[tp.ValueMapLike] = None,
                  group_by: tp.GroupByLike = None, **kwargs) -> MappedArray:
        """Convert array to mapped array.

         The length of the array should match that of the records."""
        if not isinstance(a, np.ndarray):
            a = np.asarray(a)
        checks.assert_shape_equal(a, self.values)
        if idx_field is None:
            idx_field = self.idx_field
        if idx_field is not None:
            idx_arr = self.values[idx_field]
        else:
            idx_arr = None
        return MappedArray(
            self.wrapper,
            a,
            self.values['col'],
            id_arr=self.values['id'],
            idx_arr=idx_arr,
            value_map=value_map,
            **kwargs
        ).regroup(group_by)

    @cached_method
    def count(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Return count by column."""
        wrap_kwargs = merge_dicts(dict(name_or_index='count'), wrap_kwargs)
        return self.wrapper.wrap_reduced(
            self.col_mapper.get_col_map(group_by=group_by)[1],
            group_by=group_by, **wrap_kwargs)
