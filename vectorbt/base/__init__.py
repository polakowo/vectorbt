"""Modules with base classes and utilities for pandas objects, such as broadcasting.

## Array wrapper

vectorbt's functionality is based upon the ability to perform the most essential pandas operations
using NumPy+Numba stack. One has to convert the Series/DataFrame into the NumPy format, perform
the computation, and put the array back into the pandas format. The last step is done using
`vectorbt.base.array_wrapper.ArrayWrapper`.

It stores the metadata of the original pandas object and offers methods `wrap` and `wrap_reduced`
for wrapping NumPy arrays to match the stored metadata as closest as possible.

```python-repl
>>> from vectorbt.base.array_wrapper import ArrayWrapper

>>> aw = ArrayWrapper(index=['a', 'b', 'c'], columns=['name'], ndim=1)
>>> aw.wrap([1, 2, 3])
a    1
b    2
c    3
Name: name, dtype: int64
```

## Index functions

Index functions perform operations on index objects, such as stacking, combining,
and cleansing MultiIndex levels. "Index" in pandas context is referred to both index and columns.

## Reshape functions

Reshape functions transform a pandas object/NumPy array in some way, such as tiling, broadcasting,
and unstacking.

## Combine functions

Combine functions combine two or more NumPy arrays using a custom function. The emphasis here is
done upon stacking the results into one NumPy array - since vectorbt is all about bruteforcing
large spaces of hyperparameters, concatenating the results of each hyperparameter combination into
a single DataFrame is important. All functions are available in both Python and Numba-compiled form.

## Indexing

The main purpose of indexing classes is to provide pandas-like indexing to user-defined classes
holding objects that have rows and/or columns. This is done by forwarding indexing commands
to each structured object and constructing the new user-defined class using them. This way,
one can manupulate complex classes with dozens of pandas objects using a single command.

## Accessors

The base accessor of vectorbt is `vectorbt.base.accessors.Base_Accessor`.
You can access its methods as follows:

* `vectorbt.base.accessors.Base_SRAccessor` -> `pd.Series.vbt`
* `vectorbt.base.accessors.Base_DFAccessor` -> `pd.DataFrame.vbt`

For example:

```python-repl
>>> # vectorbt.base.accessors.Base_Accessor.make_symmetric
>>> pd.Series([1, 2, 3]).vbt.make_symmetric()
     0    1    2
0  1.0  2.0  3.0
1  2.0  NaN  NaN
2  3.0  NaN  NaN
```

It contains base methods for working with pandas objects. Most of these methods are adaptations
of combine/reshape/index functions that can work with pandas objects. For example,
`vectorbt.base.reshape_fns.broadcast` can take an arbitrary number of pandas objects, thus
you can find its variations as accessor methods.

```python-repl
>>> import pandas as pd
>>> import vectorbt as vbt

>>> sr = pd.Series([1])
>>> df = pd.DataFrame([1, 2, 3])

>>> vbt.base.reshape_fns.broadcast_to(sr, df)
   0
0  1
1  1
2  1
>>> sr.vbt.broadcast_to(df)
   0
0  1
1  1
2  1
```

Additionally, `vectorbt.base.accessors.Base_Accessor` implements arithmetic (such as `+`),
comparison (such as `>`) and logical operators (such as `&`) by doing 1) NumPy-like broadcasting
and 2) the compuation with NumPy under the hood, which is mostly much faster than with pandas.

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> df = pd.DataFrame(np.random.uniform(size=(1000, 1000)))
>>> %timeit df * 2
296 ms ± 27.4 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
>>> %timeit df.vbt * 2
5.48 ms ± 1.12 ms per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

!!! note
    You should ensure that your `*.vbt` operand is on the left if the other operand is an array.
"""

from vectorbt.base import (
    accessors,
    array_wrapper,
    combine_fns,
    common,
    index_fns,
    indexing,
    reshape_fns
)
