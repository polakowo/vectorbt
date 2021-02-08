"""Modules for working with portfolios.

## Base class

Module `vectorbt.portfolio.base` provides the class `vectorbt.portfolio.base.Portfolio` for modeling
portfolio performance and calculating various risk and performance metrics. It uses Numba-compiled
functions from `vectorbt.portfolio.nb` for most computations and record classes based on
`vectorbt.records.base.Records` for evaluating events such as orders, logs, trades, positions and drawdowns.

An example of how to assess performance of a random strategy on Bitcoin:

```python-repl
>>> import vectorbt as vbt
>>> import pandas as pd
>>> import yfinance as yf

>>> ohlcv = vbt.utils.data.download("BTC-USD", period="max")
>>> entries, exits = pd.Series.vbt.signals.generate_random_both(
...     ohlcv.shape[0], n=10, seed=42)
>>> portfolio = vbt.Portfolio.from_signals(
...     ohlcv['Close'], entries, exits, price=ohlcv['Open'],
...     fees=0.001, init_cash=100., freq='1D')

>>> portfolio.stats()
Start                            2014-09-17 00:00:00
End                              2021-02-06 00:00:00
Duration                          2331 days 00:00:00
Init. Cash                                       100
Total Profit                                  1439.6
Total Return [%]                              1439.6
Benchmark Return [%]                          8481.8
Position Coverage [%]                        32.3466
Max. Drawdown [%]                             32.646
Avg. Drawdown [%]                            7.25155
Max. Drawdown Duration             493 days 00:00:00
Avg. Drawdown Duration    54 days 16:12:58.378378379
Num. Trades                                       10
Win Rate [%]                                      90
Best Trade [%]                               306.417
Worst Trade [%]                             -6.90366
Avg. Trade [%]                               46.7981
Max. Trade Duration                316 days 00:00:00
Avg. Trade Duration                 75 days 09:36:00
Expectancy                                    143.96
SQN                                          2.17025
Gross Exposure                              0.323466
Sharpe Ratio                                 1.50143
Sortino Ratio                                2.31876
Calmar Ratio                                 1.63687
dtype: object
```

## Orders class

Class `vectorbt.portfolio.orders.Orders` wraps order records and the corresponding time series
(such as open or close) to analyze orders. Orders are mainly populated when simulating a portfolio
and can be accessed as `vectorbt.portfolio.base.Portfolio.orders`.

## Logs class

Class `vectorbt.portfolio.logs.Logs` class wraps log records to analyze logs. Logs are mainly populated when
simulating a portfolio and can be accessed as `vectorbt.portfolio.base.Portfolio.logs`.

### Trades and Positions classes

Class `vectorbt.portfolio.trades.Trades` wraps trade records and the corresponding time series
(such as open or close) to analyze trades. Use `vectorbt.portfolio.trades.Trades.from_orders`
to generate trade records from order records. This is done automatically in the
`vectorbt.portfolio.base.Portfolio` class, available as `vectorbt.portfolio.base.Portfolio.trades`.

Class `vectorbt.portfolio.trades.Positions` has the same properties as trades and is also
provided by `vectorbt.portfolio.base.Portfolio` as `vectorbt.portfolio.base.Portfolio.positions`.

## Numba-compiled functions

Module `vectorbt.portfolio.nb` provides an arsenal of Numba-compiled functions that are used for portfolio
modeling, such as generating and filling orders. These only accept NumPy arrays and other Numba-compatible types.

## Enums

Module `vectorbt.portfolio.enums` defines enums and other schemas for `vectorbt.portfolio`.
"""

from vectorbt.portfolio.enums import *
from vectorbt.portfolio.base import Portfolio
from vectorbt.portfolio.orders import Orders
from vectorbt.portfolio.logs import Logs
from vectorbt.portfolio.trades import Trades, Positions

__all__ = [
    'Portfolio',
    'Orders',
    'Logs',
    'Trades',
    'Positions'
]

__pdoc__ = {k: False for k in __all__}
