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

>>> price = yf.Ticker("BTC-USD").history(period="max")
>>> entries, exits = pd.Series.vbt.signals.generate_random_both(
...     price.shape[0], n=10, seed=42)
>>> portfolio = vbt.Portfolio.from_signals(
...     price['Close'], entries, exits, price=price['Open'],
...     fees=0.001, init_cash=100., freq='1D')

>>> portfolio.stats()
Start                            2014-09-17 00:00:00
End                              2020-11-20 00:00:00
Duration                          2257 days 00:00:00
Init. Cash                                       100
Total Profit                                 1445.18
Total Return [%]                             1445.18
Benchmark Return [%]                          3949.4
Position Coverage [%]                        44.1737
Max. Drawdown [%]                            64.3272
Avg. Drawdown [%]                            7.64067
Max. Drawdown Duration            1070 days 00:00:00
Avg. Drawdown Duration    51 days 18:27:41.538461538
Num. Trades                                       10
Win Rate [%]                                      70
Best Trade [%]                               427.383
Worst Trade [%]                             -15.5047
Avg. Trade [%]                               56.3168
Max. Trade Duration                298 days 00:00:00
Avg. Trade Duration                 99 days 16:48:00
Expectancy                                   144.518
SQN                                          1.59598
Gross Exposure                              0.441737
Sharpe Ratio                                 1.13649
Sortino Ratio                                1.80862
Calmar Ratio                                0.865842
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
