"""Main class for working with records.

`vectorbt` works with two different representations of information: matrices and records.

A matrix, in this context, is just an array of one-dimensional arrays, each corresponding
to a separate feature. The matrix itself holds only one kind of information (one attribute).
For example, one can create a matrix for entry signals, with columns being different strategy
configurations. But what if the matrix is huge and sparse? Moreover, what if there is more
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

Another advantage of records is that multiple records can map to a single element in a matrix.
For example, one can define multiple orders at the same time step.

## Schema

Records must be created using [numpy.recarray](https://numpy.org/doc/stable/user/basics.rec.html#record-arrays)
and have fields `idx` and `col` - they are needed to map record values back into a matrix form.

For example:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from collections import namedtuple
>>> from vectorbt.tseries.common import TSArrayWrapper
>>> from vectorbt.records import Records

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
>>> wrapper = TSArrayWrapper(index=['x', 'y', 'z'],
...     columns=['a', 'b'], ndim=2, freq='1 day')
>>> records = Records(records_arr, wrapper)

>>> print(records.records)
   col  idx  some_field
0    0    0        10.0
1    0    1        11.0
2    0    2        12.0
3    1    0        13.0
4    1    1        14.0
5    1    2        15.0
```

There are several options for converting record values into a matrix:

* Use `Records.map_field_to_matrix` to map a record field.

```python-repl
>>> print(records.map_field_to_matrix('some_field') ** 2)
       a      b
x  100.0  169.0
y  121.0  196.0
z  144.0  225.0
```

* Use `Records.map_records_to_matrix` to map records using a custom function.

```python-repl
>>> @njit
... def power_map_nb(record, pow):
...     return record.some_field ** pow

>>> print(records.map_records_to_matrix(power_map_nb, 2))
       a      b
x  100.0  169.0
y  121.0  196.0
z  144.0  225.0
```

* Use `Records.convert_array_to_matrix` to convert an array of already mapped values.

```python-repl
>>> print(records.convert_array_to_matrix(records_arr['some_field'] ** 2))
       a      b
x  100.0  169.0
y  121.0  196.0
z  144.0  225.0
```

Furthermore, you can reduce by column as follows:

* Use `Records.reduce_records` to reduce records:

```python-repl
>>> @njit
... def pow_mean_reduce_nb(records, pow):
...     return np.mean(records.some_field ** pow)

>>> print(records.reduce_records(pow_mean_reduce_nb, 2))
a    121.666667
b    196.666667
dtype: float64
```

* Use `Records.map_reduce_records` to map and reduce records:

```python-repl
>>> @njit
... def mean_reduce_nb(map_results, *args):
...     return np.mean(map_results)

>>> print(records.map_reduce_records(power_map_nb, mean_reduce_nb, 2))
a    121.666667
b    196.666667
dtype: float64
```

* First map to a matrix and then reduce manually:

```python-repl
>>> print(records.map_records_to_matrix(power_map_nb, 2).mean())
a    121.666667
b    196.666667
dtype: float64
```

## Indexing

You can use pandas indexing on the `Records` class itself, which will forward indexing operation
to each `__init__` argument with index:

```python-repl
>>> print(records['a'].records)
   col  idx  some_field
0    0    0        10.0
1    0    1        11.0
2    0    2        12.0

>>> print(records['a'].reduce_records(mean_reduce_nb, time_units=True))
11 days 00:00:00
```

!!! note
    Changing index (time axis) is not supported."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.base.indexing import PandasIndexer
from vectorbt.tseries.common import TSArrayWrapper
from vectorbt.records.common import indexing_on_records
from vectorbt.records import nb


def _indexing_func(obj, pd_indexing_func):
    """See `vectorbt.records.common.indexing_on_records`."""
    return obj.__class__(*indexing_on_records(obj, pd_indexing_func))


class Records(PandasIndexer):
    """Exposes methods and properties for working with records.

    Args:
        records_arr (np.ndarray): A structured NumPy array of records.

            Must have fields `idx` (index position in a matrix) and `col` (column position in a matrix).
        wrapper (TSArrayWrapper): Array wrapper of type `vectorbt.tseries.common.TSArrayWrapper`."""

    def __init__(self, records_arr, wrapper):
        checks.assert_type(records_arr, np.ndarray)
        checks.assert_not_none(records_arr.dtype.fields)
        checks.assert_value_in('idx', records_arr.dtype.names)
        checks.assert_value_in('col', records_arr.dtype.names)
        checks.assert_type(wrapper, TSArrayWrapper)

        self.records_arr = records_arr
        self.wrapper = wrapper

        PandasIndexer.__init__(self, _indexing_func)

    @cached_property
    def records_col_index(self):
        """Column index for `Records.records`."""
        return nb.index_record_cols_nb(self.records_arr, len(self.wrapper.columns))

    @cached_property
    def records(self):
        """Records."""
        return pd.DataFrame.from_records(self.records_arr)

    def map_records_to_matrix(self, map_func_nb, *args, default_val=np.nan):
        """Map each record to a value and store it in a matrix.

        See `vectorbt.records.nb.map_records_to_matrix_nb`."""
        checks.assert_numba_func(map_func_nb)

        target_shape = (len(self.wrapper.index), len(self.wrapper.columns))
        return self.wrapper.wrap(nb.map_records_to_matrix_nb(
            self.records_arr, target_shape, default_val, map_func_nb, *args))

    def convert_array_to_matrix(self, a, default_val=np.nan):
        """Convert a 1-dim array already mapped by the user.

        See `vectorbt.records.nb.convert_array_to_matrix`.

        The length of the array should match that of the records."""
        checks.assert_type(a, np.ndarray)
        checks.assert_ndim(a, 1)
        checks.assert_same_shape(a, self.records_arr, axis=0)

        target_shape = (len(self.wrapper.index), len(self.wrapper.columns))
        result = nb.convert_array_to_matrix(a, self.records_arr, target_shape, default_val)
        return self.wrapper.wrap(result)

    @cached_method
    def map_field_to_matrix(self, field, default_val=np.nan):
        """Map field to a matrix."""
        return self.convert_array_to_matrix(self.records_arr[field], default_val=default_val)

    def reduce_records(self, reduce_func_nb, *args, default_val=np.nan, **kwargs):
        """Reduce records by column.

        See `vectorbt.records.nb.reduce_records_nb`.

        `**kwargs` will be passed to `vectorbt.tseries.common.TSArrayWrapper.wrap_reduced`."""
        checks.assert_numba_func(reduce_func_nb)

        return self.wrapper.wrap_reduced(nb.reduce_records_nb(
            self.records_arr, len(self.wrapper.columns), default_val, reduce_func_nb, *args), **kwargs)

    def map_reduce_records(self, map_func_nb, reduce_func_nb, *args, default_val=np.nan, **kwargs):
        """Map each record to a value and reduce all values by column.

        See `vectorbt.records.nb.map_reduce_records_nb`.

        `**kwargs` will be passed to `vectorbt.tseries.common.TSArrayWrapper.wrap_reduced`."""
        checks.assert_numba_func(reduce_func_nb)

        return self.wrapper.wrap_reduced(nb.map_reduce_records_nb(
            self.records_arr, len(self.wrapper.columns), default_val, map_func_nb, reduce_func_nb, *args), **kwargs)

    @cached_property
    def count(self):
        """Number of records."""
        return self.reduce_records(nb.count_reduce_nb, default_val=0.)

    @cached_property
    def recarray(self):
        return self.records_arr.view(np.recarray)
