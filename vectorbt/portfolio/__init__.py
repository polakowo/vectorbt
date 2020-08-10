"""Modules for working with portfolios.

`vectorbt.portfolio` provides the class `vectorbt.portfolio.base.Portfolio` for modeling portfolio
performance and calculating various risk and performance metrics. It uses Numba-compiled
functions from `vectorbt.portfolio.nb` for most computations and record classes from `vectorbt.records`
for evaluating events such as orders, trades and positions.

## Base class

The `vectorbt.portfolio.base.Portfolio` class is the base class for backtesting. It

1. receives a set of inputs, such as entry and exit signals, orders, or a dynamic order function,
2. fills these orders,
3. generates cash, shares, and equity at each time step, and finally
4. calculates a broad range of risk & performance metrics, including max drawdown, Sharpe & Sortino ratios.

Among other features, `vectorbt.portfolio.base.Portfolio` class is very flexible:

* you can pass both Series and DataFrames as inputs - simulation will be performed per column;
* starting capital can be defined per column;
* some parameters such as fees can be even defined per element;
* all properties such as performance metrics are cached, so they can built upon each other without re-calculation;
* the whole instance can be indexed just like a pandas object.

### Example

Simulation of trading randomly on Bitcoin:

```python-repl
>>> import vectorbt as vbt
>>> import pandas as pd
>>> import yfinance as yf

>>> price = yf.Ticker("BTC-USD").history(period="max")
>>> entries, exits = pd.Series.vbt.signals.generate_random_entries_and_exits(
...     price.shape[0], n=10, seed=42
... )
>>> portfolio = vbt.Portfolio.from_signals(
...     price['Close'], entries, exits,
...     fees=0.001,
...     init_capital=100,
...     freq='1D'
... )

>>> portfolio.stats
Start                     2014-09-17 00:00:00
End                       2020-07-14 00:00:00
Duration                   2128 days 00:00:00
Holding Duration [%]                  55.5451
Total Profit                          418.811
Total Return [%]                      418.811
Buy & Hold Return [%]                 1922.34
Max. Drawdown [%]                     69.8867
Avg. Drawdown [%]                      12.449
Max. Drawdown Duration      941 days 00:00:00
Avg. Drawdown Duration       84 days 07:00:00
Num. Trades                                10
Win Rate [%]                               60
Best Trade [%]                        155.864
Worst Trade [%]                      -28.4219
Avg. Trade [%]                         28.989
Max. Trade Duration         388 days 00:00:00
Avg. Trade Duration         118 days 04:48:00
Expectancy                            41.8811
SQN                                  0.992313
Sharpe Ratio                         0.782394
Sortino Ratio                         1.18547
Calmar Ratio                         0.466892
Name: Close, dtype: object
```

## Enums

`vectorbt.portfolio.enums` defines schemas for orders.

## Numba-compiled functions

`vectorbt.portfolio.nb` provides an arsenal of Numba-compiled functions that are used for portfolio
modeling, such as generating and filling orders. These only accept NumPy arrays and other Numba-compatible types.
"""

from vectorbt.portfolio.enums import Order, SizeType
from vectorbt.portfolio.base import Portfolio
