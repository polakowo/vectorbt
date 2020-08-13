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
>>> from datetime import datetime

>>> start = datetime(2019, 1, 1)
>>> end = datetime(2020, 1, 1)
>>> price = yf.Ticker("BTC-USD").history(start=start, end=end)['Close']
>>> drawdowns = vbt.Drawdowns.from_ts(price, freq='1 days')

>>> drawdowns.records.head()
   col  start_idx  valley_idx  end_idx  status
0    0          2           3        6       1
1    0          6          38       54       1
2    0         54          63       91       1
3    0         93          94       95       1
4    0         98          99      100       1

>>> drawdowns.plot()
```

![](/vectorbt/docs/img/drawdowns_plot.png)

```python-repl
>>> drawdowns.drawdown
<vectorbt.records.base.MappedArray at 0x7fafa6a11160>

>>> drawdowns.drawdown.min()
-0.48982769972565016

>>> drawdowns.drawdown.hist(trace_kwargs=dict(nbinsx=50))
```

![](/vectorbt/docs/img/drawdowns_drawdown_hist.png)

Moreover, all time series accessors have cached property `drawdowns`:

```python-repl
>>> price.vbt.drawdowns.active.current_drawdown()
-0.4473361334272673
```

## Orders

`vectorbt.records.orders.Orders` class accepts order records and the corresponding time series
(such as open or close) to analyze orders. Orders are mainly populated when simulating a portfolio
with the `vectorbt.portfolio.base.Portfolio` class. They can be accessed by`vectorbt.portfolio.base.Portfolio.orders`.

```python-repl
>>> entries, exits = pd.Series.vbt.signals.generate_random_both(price.shape, n=10, seed=42)
>>> portfolio = vbt.Portfolio.from_signals(price, entries, exits, fees=0.01, freq='1D')

>>> portfolio.orders.records.head()
   col  idx      size    price      fees  side
0    0    0  0.026454  3742.70  0.990099     0
1    0   15  0.026454  3630.68  0.960465     1
2    0   33  0.026738  3521.06  0.941446     0
3    0   39  0.026738  3666.78  0.980408     1
4    0   55  0.025220  3810.43  0.960994     0

>>> portfolio.orders.plot()
```

![](/vectorbt/docs/img/orders_plot.png)


```python-repl
>>> portfolio.orders.count()
20

>>> portfolio.orders.buy.price.mean()
6359.48

>>> portfolio.orders.sell.price.mean()
6815.132
```

## Events

`vectorbt.records.events.Events` class is the base class for working with events, such as trades and
positions. It accepts events records with any compatible schema and the corresponding time series
(such as open or close) to analyze event data. Two main subclasses are `vectorbt.records.events.Trades`
and `vectorbt.records.events.Positions`. Both have information on opening and closing events.

### Trades

`vectorbt.records.events.Trades` class accepts trade records and the corresponding time series
(such as open or close) to analyze trades. Use `vectorbt.records.events.Trades.from_orders`
to generate trade records from order records. This is done automatically in the
`vectorbt.portfolio.base.Portfolio` class, available as `vectorbt.portfolio.base.Portfolio.trades`.

```python-repl
>>> portfolio.trades.records.head()
   col      size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
0    0  0.026454          0      3742.70    0.990099        15     3630.68
1    0  0.026738         33      3521.06    0.941446        39     3666.78
2    0  0.025220         55      3810.43    0.960994        57     3854.36
3    0  0.023671         76      4025.23    0.952824       101     5064.49
4    0  0.022394        119      5247.35    1.175091       126     5746.81

   exit_fees        pnl    return  status  position_idx
0   0.960465  -4.913957 -0.049140       1             0
1   0.980408   1.974345  0.020764       1             1
2   0.972073  -0.825148 -0.008501       1             2
3   1.198830  22.448978  0.233272       1             3
4   1.286940   8.722873  0.073496       1             4

>>> portfolio.trades.plot()
```

![](/vectorbt/docs/img/trades_plot.png)

```python-repl
>>> portfolio.trades.expectancy()
5.8270083764704275
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
