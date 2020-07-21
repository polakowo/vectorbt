"""Modules for working with signals, such as entry and exit signals.

## Accessors

You can access methods listed in `vectorbt.signals.accessors` as follows:

* `vectorbt.signals.accessors.Signals_SRAccessor` -> `pd.Series.vbt.signals.*`
* `vectorbt.signals.accessors.Signals_DFAccessor` -> `pd.DataFrame.vbt.signals.*`

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.signals.accessors.Signals_Accessor.rank
>>> pd.Series([False, True, True, True, False]).vbt.signals.rank()
0    0
1    1
2    2
3    3
4    0
dtype: int64
```

The accessors extend `vectorbt.generic.accessors`.

## Numba-compiled functions

`vectorbt.signals.nb` provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of the backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

```python-repl
>>> # vectorbt.signals.nb.rank_1d_nb
>>> vbt.signals.nb.rank_1d_nb(np.array([False, True, True, True, False]))
array([0, 1, 2, 3, 0])
```
"""
