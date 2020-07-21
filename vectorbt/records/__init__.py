"""Modules for working with records.

Records are the second form of data representation in vectorbt. They allow storing sparse event data
such as drawdowns, orders, trades, and positions, without converting them back to the matrix form and
occupying the user's memory. For more details, see `vectorbt.records.base`.

## Base classes

`vectorbt.records.base.Records` and `vectorbt.records.base.MappedArray` are two essential classes
for working with records. The `vectorbt.records.base.Records` class wraps the actual records array
(such as trades) and offers methods for mapping it to some array of values (such as P&L of each trade).
The `vectorbt.records.base.MappedArray` class then takes the mapped array and reduces it by column.

## Drawdowns

`vectorbt.records.drawdowns.Drawdowns` class accepts drawdown records and the corresponding time series
to analyze the periods of drawdown. Using `vectorbt.records.drawdowns.Drawdowns.from_ts`, you can generate
drawdown records for any time series and analyze them right away.

```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> import yfinance as yf

>>> price = yf.Ticker("BTC-USD").history(period="max")['Close']
>>> drawdowns = vbt.Drawdowns.from_ts(price, freq='1 days')

>>> drawdowns.records.head()
   col  start_idx  valley_idx  end_idx  status
0    0          0         119      454       1
1    0        454         485      587       1
2    0        587         610      618       1
3    0        619         620      621       1
4    0        621         622      623       1

>>> drawdowns.plot()
```

![](/vectorbt/docs/img/drawdowns_plot.png)

```python-repl
>>> drawdowns.drawdown
<vectorbt.records.base.MappedArray at 0x7fafa6a11160>

>>> drawdowns.drawdown.min()
-0.8339901730487141

>>> drawdowns.drawdown.hist(trace_kwargs=dict(nbinsx=50))
```

![](/vectorbt/docs/img/drawdowns_drawdown_hist.png)

Moreover, all time series accessors have cached property `drawdowns`:

```python-repl
>>> price.vbt.drawdowns.active.current_drawdown
-0.8339901730487141
```

## Orders

`vectorbt.records.orders.Orders` class accepts order records and the corresponding time series
(such as open or close) to analyze orders. Orders are mainly populated when simulating a portfolio
with the `vectorbt.portfolio.base.Portfolio` class. They can be accessed by`vectorbt.portfolio.base.Portfolio.orders`.

```python-repl
>>> entries, exits = pd.Series.vbt.signals.generate_random_entries_and_exits(price.shape, 10, seed=42)
>>> portfolio = vbt.Portfolio.from_signals(price, entries, exits, fees=0.01, freq='1D')

>>> portfolio.orders.records.head()
   col  idx      size   price      fees  side
0    0   70  0.268778  368.37  0.990099     0
1    0  101  0.268778  315.86  0.848963     1
2    0  282  0.341620  243.59  0.832152     0
3    0  284  0.341620  249.01  0.850668     1
4    0  290  0.319607  260.89  0.833823     0

>>> portfolio.orders.plot()
```

![](/vectorbt/docs/img/orders_plot.png)


```python-repl
>>> portfolio.orders.count
20

>>> portfolio.orders.buy.price.mean()
4740.3949999999995

>>> portfolio.orders.sell.price.mean()
4528.465999999999
```

## Events

`vectorbt.records.events.Events` class is the base class for working with events, such as trades and
positions. It accepts events records with any compatible schema and the corresponding time series
(such as open or close) to analyze event data. Two main subclasses are `vectorbt.records.events.Trades`
and `vectorbt.records.events.Positions`. Both have information on opening and closing events.

In context of vectorbt, a trade is simply a sell operation. For example, if you have a single large
buy operation and 100 small sell operations, you will see 100 trades, each opening with a fraction
of the buy operation's size and fees. On the other hand, having 100 buy operations and just a single
sell operation will generate a single trade with buy price being a size-weighted average over all
purchase prices, and opening size and fees being the sum over all sizes and fees.

The same holds for positions: their size is allowed to accumulate/decrease over time. Each buy/sell
operation is tracked and then used for deriving P&L of the entire position. A position opens with
first buy operation and closes with last sell operation that results in zero security holdings.

!!! note
    Without distribution of orders, trades and positions yield the same results.

### Trades

`vectorbt.records.events.Trades` class accepts trade records and the corresponding time series
(such as open or close) to analyze trades. Use `vectorbt.records.events.Trades.from_orders`
to generate trade records from order records. This is done automatically in the
`vectorbt.portfolio.base.Portfolio` class, available as `vectorbt.portfolio.base.Portfolio.trades`.

```python-repl
>>> portfolio.trades.records.head()
   col      size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \
0    0  0.268778         70       368.37    0.990099       101      315.86
1    0  0.341620        282       243.59    0.832152       284      249.01
2    0  0.319607        290       260.89    0.833823       291      271.91
3    0  0.290620        297       293.11    0.851835       924     1039.97
4    0  0.037243       1288      7954.48    2.962508      1309     8163.42

   exit_fees         pnl    return  status  position_idx
0   0.848963  -15.952617 -0.159526       1             0
1   0.850668    0.168760  0.002008       1             1
2   0.869044    1.819204  0.021602       1             2
3   3.022357  213.177967  2.477795       1             3
4   3.040324    1.778776  0.005945       1             4

>>> portfolio.trades.plot()
```

![](/vectorbt/docs/img/trades_plot.png)

```python-repl
>>> portfolio.trades.expectancy
6.984101462164443
```

### Positions

`vectorbt.records.events.Positions` class has the same properties as trades and is also
natively supported by `vectorbt.portfolio.base.Portfolio`.

## Enums

`vectorbt.records.enums` defines schemas for all records used across vectorbt.

## Numba-compiled functions

`vectorbt.records.nb` provides an arsenal of Numba-compiled functions that are used for generating,
mapping, and reducing records. These only accept NumPy arrays and other Numba-compatible types.

"""

from vectorbt.records.enums import (
    DrawdownStatus,
    drawdown_dt,
    OrderSide,
    order_dt,
    event_dt,
    EventStatus,
    trade_dt,
    position_dt
)
from vectorbt.records.base import MappedArray, Records
from vectorbt.records.orders import Orders
from vectorbt.records.events import Events, Trades, Positions
from vectorbt.records.drawdowns import Drawdowns
