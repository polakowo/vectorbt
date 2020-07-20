"""Modules for working with data of any type.

## Accessors

You can access methods listed in `vectorbt.generic.accessors` as follows:

* `vectorbt.generic.accessors.Generic_SRAccessor` -> `pd.Series.vbt`
* `vectorbt.generic.accessors.Generic_DFAccessor` -> `pd.DataFrame.vbt`

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.generic.accessors.Generic_Accessor.rolling_mean
>>> pd.Series([1, 2, 3, 4]).vbt.rolling_mean(2)
0    NaN
1    1.5
2    2.5
3    3.5
dtype: float64
```

The accessors inherit `vectorbt.base.accessors` and are inherited by more
specialized accessors, such as `vectorbt.signals.accessors` and `vectorbt.returns.accessors`.

## Numba-compiled functions

`vectorbt.generic.nb` provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of the backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

```python-repl
>>> # vectorbt.generic.nb.rolling_mean_1d_nb
>>> vbt.generic.nb.rolling_mean_1d_nb(np.array([1, 2, 3, 4]), 2)
array([nan, 1.5, 2.5, 3.5])
```"""

from vectorbt.generic import accessors, nb
