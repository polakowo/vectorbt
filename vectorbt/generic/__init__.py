"""Modules for working with any time series.

In contrast to the `vectorbt.base` sub-package, focuses on the data itself.

## Accessors

You can access methods listed in `vectorbt.generic.accessors` as follows:

* `vectorbt.generic.accessors.Generic_SRAccessor` -> `pd.Series.vbt.*`
* `vectorbt.generic.accessors.Generic_DFAccessor` -> `pd.DataFrame.vbt.*`

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

## Plotting

Module `vectorbt.generic.plotting` provides functions for visualizing data in an efficient and convenient way.
Each creates a figure widget that is compatible with ipywidgets and enables interactive data visualization
in Jupyter Notebook and JupyterLab environments. For more details on using Plotly, see
[Getting Started with Plotly in Python](https://plotly.com/python/getting-started/).

The module can be accessed directly via `vbt.plotting`.

## Drawdowns

Class `vectorbt.generic.drawdowns.Drawdowns` accepts drawdown records and the corresponding time series
to analyze the periods of drawdown. Using `vectorbt.generic.drawdowns.Drawdowns.from_ts`, you can generate
drawdown records for any time series and analyze them right away.

Moreover, all time series accessors have a method `drawdowns`:

```python-repl
>>> price.vbt.drawdowns().current_drawdown()
-0.4473361334272673
```

## Numba-compiled functions

Module `vectorbt.generic.nb` provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of the backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

The module can be accessed directly via `vbt.nb`.

```python-repl
>>> # vectorbt.generic.nb.rolling_mean_1d_nb
>>> vbt.nb.rolling_mean_1d_nb(np.array([1, 2, 3, 4]), 2)
array([nan, 1.5, 2.5, 3.5])
```

## Enums

Module `vectorbt.generic.enums` defines enums and other schemas for `vectorbt.generic`.
"""

from vectorbt.generic.enums import *
from vectorbt.generic.drawdowns import Drawdowns

__all__ = [
    'Drawdowns'
]

__pdoc__ = {k: False for k in __all__}
