"""Modules for working with returns.

Offers common financial risk and performance metrics as found in [empyrical](https://github.com/quantopian/empyrical),
but Numba-compiled and optimized for 2-dim arrays.

## Accessors

You can access methods listed in `vectorbt.returns.accessors` as follows:

* `vectorbt.returns.accessors.Returns_SRAccessor` -> `pd.Series.vbt.returns`
* `vectorbt.returns.accessors.Returns_DFAccessor` -> `pd.DataFrame.vbt.returns`

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.returns.accessors.Returns_Accessor.total
>>> pd.Series([0.2, 0.1, 0, -0.1, -0.2]).vbt.returns.total()
-0.049599999999999866
```

The accessors extend `vectorbt.generic.accessors`.

```python-repl
>>> # inherited from Generic_Accessor
>>> pd.Series([0.2, 0.1, 0, -0.1, -0.2]).vbt.returns.max()
0.2
```

## Numba-compiled functions

`vectorbt.returns.nb` provides an arsenal of Numba-compiled functions that are used by accessors
and for measuring portfolio performance. These only accept NumPy arrays and other Numba-compatible types.

```python-repl
>>> # vectorbt.returns.nb.cum_returns_1d_nb
>>> vbt.returns.nb.cum_returns_1d_nb(np.array([0.2, 0.1, 0, -0.1, -0.2]))
array([0.2, 0.32, 0.32, 0.188, -0.0496])
```"""

from vectorbt.returns import accessors, nb
