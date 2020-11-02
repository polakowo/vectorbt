"""Modules for working with portfolios.

## Base class

`vectorbt.portfolio.base` provides the class `vectorbt.portfolio.base.Portfolio` for modeling portfolio
performance and calculating various risk and performance metrics. It uses Numba-compiled
functions from `vectorbt.portfolio.nb` for most computations and record classes from `vectorbt.records`
for evaluating events such as orders, trades and positions.

An example of how to assess performance of a random strategy on Bitcoin:

```python-repl
>>> import vectorbt as vbt
>>> import pandas as pd
>>> import yfinance as yf

>>> price = yf.Ticker("BTC-USD").history(period="max")
>>> entries, exits = pd.Series.vbt.signals.generate_random_both(
...     price.shape[0], n=10, seed=42)
>>> portfolio = vbt.Portfolio.from_signals(
...     price['Close'], entries, exits,
...     entry_price=price['Open'], exit_price=price['Open'],
...     fees=0.001, init_cash=100., freq='1D')

>>> portfolio.stats()
Start                     2014-09-17 00:00:00
End                       2020-09-13 00:00:00
Duration                   2188 days 00:00:00
Holding Duration [%]                  56.3528
Total Profit                          504.145
Total Return [%]                      504.145
Buy & Hold Return [%]                 2145.61
Max. Drawdown [%]                     70.9496
Avg. Drawdown [%]                     11.8956
Max. Drawdown Duration      569 days 00:00:00
Avg. Drawdown Duration       68 days 00:00:00
Num. Trades                                10
Win Rate [%]                               50
Best Trade [%]                        382.442
Worst Trade [%]                      -45.7088
Avg. Trade [%]                        49.4489
Max. Trade Duration         389 days 00:00:00
Avg. Trade Duration         123 days 07:12:00
Expectancy                            50.4145
SQN                                  0.603459
Sharpe Ratio                          0.86511
Sortino Ratio                         1.26662
Calmar Ratio                           0.4932
Name: (Close, Open), dtype: object
```

## Enums

`vectorbt.portfolio.enums` defines schemas for orders.

## Numba-compiled functions

`vectorbt.portfolio.nb` provides an arsenal of Numba-compiled functions that are used for portfolio
modeling, such as generating and filling orders. These only accept NumPy arrays and other Numba-compatible types.
"""

from vectorbt.portfolio.enums import (
    InitCashMode,
    CallSeqType,
    SizeType,
    ConflictMode,
    Order,
    NoOrder,
    Direction
)
from vectorbt.portfolio.base import Portfolio
