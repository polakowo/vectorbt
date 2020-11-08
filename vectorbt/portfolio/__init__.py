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
End                              2020-11-08 00:00:00
Duration                          2245 days 00:00:00
Total Profit                                 21.0787
Total Return [%]                             21.0787
Buy & Hold Return [%]                        3243.74
Position Coverage [%]                        88.9978
Max. Drawdown [%]                            121.735
Avg. Drawdown [%]                             14.207
Max. Drawdown Duration             947 days 00:00:00
Avg. Drawdown Duration    75 days 21:13:50.769230769
Num. Trades                                       19
Win Rate [%]                                 52.6316
Best Trade [%]                               1538.25
Worst Trade [%]                             -61.4248
Avg. Trade [%]                               76.2947
Max. Trade Duration                381 days 00:00:00
Avg. Trade Duration       99 days 18:56:50.526315789
Expectancy                                   4.99948
SQN                                          0.18233
Sharpe Ratio                               -0.195858
Sortino Ratio                              -0.207835
Calmar Ratio                                0.427493
Name: (Close, Open), dtype: object
```

## Orders

`vectorbt.portfolio.orders.Orders` class accepts order records and the corresponding time series
(such as open or close) to analyze orders. Orders are mainly populated when simulating a portfolio
with the `vectorbt.portfolio.base.Portfolio` class. They can be accessed by`vectorbt.portfolio.base.Portfolio.orders`.

```python-repl
>>> orders = portfolio.orders()

>>> orders.records.head()
   col  idx      size   price      fees  side
0    0  247  0.424529  235.32  0.099900     0
1    0  303  0.849057  278.09  0.236114     1
2    0  482  0.525777  448.18  0.235643     0
3    0  700  0.202496  577.76  0.116994     1
4    0  829  0.126613  922.18  0.116760     0

>>> orders.plot()
```

![](/vectorbt/docs/img/orders_plot.png)


```python-repl
>>> orders.count()
20

>>> orders.buy.price.mean()
4685.976000000001

>>> orders.sell.price.mean()
6085.213000000001
```

### Trades

`vectorbt.portfolio.trades.Trades` class accepts trade records and the corresponding time series
(such as open or close) to analyze trades. Use `vectorbt.portfolio.trades.Trades.from_orders`
to generate trade records from order records. This is done automatically in the
`vectorbt.portfolio.base.Portfolio` class, available as `vectorbt.portfolio.base.Portfolio.trades`.

```python-repl
>>> trades = portfolio.trades()

>>> trades.records.head()

   col      size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
0    0  0.424529        247       235.32    0.099900       303      278.09
1    0  0.424529        303       278.09    0.118057       482      448.18
2    0  0.101248        482       448.18    0.045377       700      577.76
3    0  0.101248        700       577.76    0.058497       829      922.18
4    0  0.025365        829       922.18    0.023391      1210    15123.70

   exit_fees         pnl     return  direction  status  position_idx
0   0.118057   17.939136   0.179571          0       1             0
1   0.190265  -72.516414  -0.614248          1       1             1
2   0.058497   13.015846   0.286836          0       1             2
3   0.093369  -35.023715  -0.598726          1       1             3
4   0.383619  359.820194  15.382544          0       1             4

>>> trades.plot()
```

![](/vectorbt/docs/img/trades_plot.png)

```python-repl
>>> trades.expectancy()
4.999484253660235
```

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
