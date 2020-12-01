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

## Signal factory

The signal factory class `vectorbt.signals.factory.SignalFactory` extends
`vectorbt.indicators.factory.IndicatorFactory` to offer a convenient way to create signal generators
of any complexity. By providing it with information such as entry and exit functions and the names
of your inputs, parameters, and outputs, it will create a stand-alone class capable of generating
signals for an arbitrary combination of your inputs and parameters.

## Basic

`vectorbt.signals.basic` provides a collection of basic signal generators, such as
random signal generator, all built with `vectorbt.signals.factory.SignalFactory`.

## Numba-compiled functions

`vectorbt.signals.nb` provides an arsenal of Numba-compiled functions that are used by accessors
and in many other parts of the backtesting pipeline, such as technical indicators.
These only accept NumPy arrays and other Numba-compatible types.

```python-repl
>>> # vectorbt.signals.nb.rank_1d_nb
>>> vbt.signals.nb.rank_1d_nb(np.array([False, True, True, True, False]))
array([0, 1, 2, 3, 0])
```

## Enums

Module `vectorbt.signals.enums` defines enums and other schemas for `vectorbt.signals`.
"""

from vectorbt.signals.enums import *
from vectorbt.signals.factory import SignalFactory
from vectorbt.signals.basic import (
    RAND,
    RPROB,
    RPROBEX,
    IRPROBEX,
    STEX,
    ISTEX,
    ADVSTEX,
    IADVSTEX
)

__all__ = [
    'SignalFactory',
    'RAND',
    'RPROB',
    'RPROBEX',
    'IRPROBEX',
    'STEX',
    'ISTEX',
    'ADVSTEX',
    'IADVSTEX'
]

__pdoc__ = {k: False for k in __all__}
