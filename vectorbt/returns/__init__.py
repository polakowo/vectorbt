"""Modules for working with returns.

Offers common financial risk and performance metrics as found in [empyrical](https://github.com/quantopian/empyrical),
but Numba-compiled and optimized for 2-dim arrays.

## Accessors

You can access methods listed in `vectorbt.returns.accessors` as follows:

* `vectorbt.returns.accessors.Returns_SRAccessor` -> `pd.Series.vbt.returns.*`
* `vectorbt.returns.accessors.Returns_DFAccessor` -> `pd.DataFrame.vbt.returns.*`

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.returns.accessors.Returns_Accessor.total
>>> price = pd.Series([1.1, 1.2, 1.3, 1.2, 1.1])
>>> returns = price.pct_change()
>>> returns.vbt.returns.total()
0.0
```

The accessors extend `vectorbt.generic.accessors`.

```python-repl
>>> # inherited from Generic_Accessor
>>> returns.vbt.returns.max()
0.09090909090909083
```

!!! note
    The underlying Series/DataFrame must already be a return series.

## Numba-compiled functions

Module `vectorbt.returns.nb` provides an arsenal of Numba-compiled functions that are used by accessors
and for measuring portfolio performance. These only accept NumPy arrays and other Numba-compatible types.

```python-repl
>>> # vectorbt.returns.nb.cum_returns_1d_nb
>>> vbt.returns.nb.cum_returns_1d_nb(returns.values)
array([0., 0.09090909, 0.18181818, 0.09090909, 0.])
```"""

