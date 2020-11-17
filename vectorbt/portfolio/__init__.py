"""Modules for working with portfolios.

## Base class

`vectorbt.portfolio.base` provides the class `vectorbt.portfolio.base.Portfolio` for modeling portfolio
performance and calculating various risk and performance metrics. It uses Numba-compiled
functions from `vectorbt.portfolio.nb` for most computations and record classes based on
`vectorbt.records.base.Records` for evaluating events such as orders, trades and positions.

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
End                              2020-11-17 00:00:00
Duration                          2253 days 00:00:00
Total Profit                                 2035.35
Total Return [%]                             2035.35
Benchmark Return [%]                         3551.03
Position Coverage [%]                        48.7794
Max. Drawdown [%]                            50.4779
Avg. Drawdown [%]                            9.06616
Max. Drawdown Duration             509 days 00:00:00
Avg. Drawdown Duration    47 days 14:30:41.860465116
Num. Trades                                       10
Win Rate [%]                                      70
Best Trade [%]                               774.445
Worst Trade [%]                             -36.6491
Avg. Trade [%]                               95.8981
Max. Trade Duration                448 days 00:00:00
Avg. Trade Duration                109 days 21:36:00
Expectancy                                   203.535
SQN                                         0.699516
Gross Exposure                              0.487794
Sharpe Ratio                                 1.30793
Sortino Ratio                                 2.0332
Calmar Ratio                                 1.27191
Name: Close, dtype: object
```

## Orders

`vectorbt.portfolio.orders.Orders` class accepts order records and the corresponding time series
(such as open or close) to analyze orders. Orders are mainly populated when simulating a portfolio
with the `vectorbt.portfolio.base.Portfolio` class. They can be accessed by`vectorbt.portfolio.base.Portfolio.orders`.

### Trades

`vectorbt.portfolio.trades.Trades` class accepts trade records and the corresponding time series
(such as open or close) to analyze trades. Use `vectorbt.portfolio.trades.Trades.from_orders`
to generate trade records from order records. This is done automatically in the
`vectorbt.portfolio.base.Portfolio` class, available as `vectorbt.portfolio.base.Portfolio.trades`.

### Positions

`vectorbt.portfolio.positions.Positions` class has the same properties as trades and is also
natively supported by `vectorbt.portfolio.base.Portfolio`.

## Numba-compiled functions

`vectorbt.portfolio.nb` provides an arsenal of Numba-compiled functions that are used for portfolio
modeling, such as generating and filling orders. These only accept NumPy arrays and other Numba-compatible types.

## Enums

`vectorbt.portfolio.enums` defines enums and other schemas for `vectorbt.portfolio`.
"""

from vectorbt.portfolio.enums import *
from vectorbt.portfolio.base import Portfolio
from vectorbt.portfolio.orders import Orders
from vectorbt.portfolio.logs import Logs
from vectorbt.portfolio.trades import Trades, Positions
