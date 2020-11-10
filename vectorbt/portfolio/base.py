"""Base class for modeling portfolio and measuring its performance.

The job of the `Portfolio` class is to create a series of positions allocated 
against a cash component, produce an equity curve, incorporate basic transaction costs
and produce a set of statistics about its performance. In particular it outputs
position/profit metrics and drawdown information.

## Workflow

The workflow of `Portfolio` is simple:

1. Receives a set of inputs, such as entry and exit signals
2. Uses them to generate and fill orders in form of records (simulation part)
3. Calculates a broad range of risk & performance metrics based on these records (analysis part)

It basically builds upon the `vectorbt.portfolio.orders.Orders` class. To simplify creation of order
records and keep track of balances, it exposes several convenience methods with prefix `from_`.
For example, you can use `Portfolio.from_signals` method to generate orders from entry and exit signals.
Alternatively, you can use `Portfolio.from_order_func` to run a custom order function on each tick.
The results are then automatically passed to the constructor method of `Portfolio` and you will
receive a portfolio instance ready to be used for performance analysis.

This way, one can simulate and analyze his/her strategy in a couple of lines.

### Example

The following example does something crazy: it checks candlestick data of 6 major cryptocurrencies
in 2020 against every single pattern found in TA-Lib, and translates them into signals:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> from datetime import datetime
>>> import yfinance as yf
>>> import talib
>>> import vectorbt as vbt

>>> # Fetch price history
>>> pairs = ['BTC-USD', 'ETH-USD', 'XRP-USD', 'BNB-USD', 'BCH-USD', 'LTC-USD']
>>> start = datetime(2020, 1, 1)
>>> end = datetime(2020, 9, 1)
>>> pair_history = {p: yf.Ticker(p).history(start=start, end=end) for p in pairs}

>>> # Put assets into a single dataframe by price type
>>> price = {}
>>> for pt in ['Open', 'High', 'Low', 'Close']:
...     price[pt] = pd.DataFrame({p: df[pt] for p, df in pair_history.items()})

>>> price['Open'].head()
            BTC-USD  ETH-USD  XRP-USD  BNB-USD  BCH-USD  LTC-USD
Date
2019-12-31  7294.44   132.61   0.1945    13.95   209.30    42.77
2020-01-01  7194.89   129.63   0.1929    13.73   204.67    41.33
2020-01-02  7202.55   130.82   0.1927    13.70   204.35    42.02
2020-01-03  6984.43   127.41   0.1879    13.04   196.01    39.86
2020-01-04  7345.38   134.17   0.1935    13.67   222.54    42.38

>>> # Run every single pattern recognition indicator and combine results
>>> result = pd.DataFrame.vbt.empty_like(price['Open'], fill_value=0.)
>>> for pattern in talib.get_function_groups()['Pattern Recognition']:
...     PRecognizer = vbt.IndicatorFactory.from_talib(pattern)
...     pr = PRecognizer.run(price['Open'], price['High'], price['Low'], price['Close'])
...     result = result + pr.integer

>>> # Don't look into future
>>> result = result.vbt.fshift(1)

>>> # Treat each number as order value in USD
>>> size = result / price['Open']

>>> # Simulate portfolio
>>> portfolio = vbt.Portfolio.from_orders(
...     price['Close'], size, price=price['Open'],
...     init_cash='autoalign', fees=0.001, slippage=0.001
... )

>>> # Visualize portfolio value
>>> portfolio.value().vbt.plot()
```

![](/vectorbt/docs/img/portfolio_value.png)

## Features

### Broadcasting

`Portfolio` is very flexible towards inputs:

* Accepts both Series and DataFrames as inputs
* Broadcasts inputs to the same shape using vectorbt's own broadcasting rules
* Many inputs (such as `fees`) can be passed as a single value, value per column/row, or as a matrix
* Implements flexible indexing wherever possible to save memory

### Grouping

One of the key features of `Portfolio` is the ability to group columns. Groups can be specified by
`group_by`, which can be anything from positions or names of column levels, to a NumPy array with
actual groups. Groups can be formed to share capital between columns or to compute metrics
for a combined portfolio of multiple independent columns.

For example, let's divide our portfolio into two groups sharing the same cash:

```python-repl
>>> # Simulate combined portfolio
>>> group_by = pd.Index([
...     'first', 'first', 'first',
...     'second', 'second', 'second'
... ], name='group')
>>> comb_portfolio = vbt.Portfolio.from_orders(
...     price['Close'], size, price=price['Open'],
...     init_cash='autoalign', fees=0.001, slippage=0.001,
...     group_by=group_by, cash_sharing=True
... )

>>> # Get total profit per group
>>> comb_portfolio.total_profit()
group
first     21474.794005
second     7973.848970
dtype: float64
```

Not only can you analyze each group, but also each column in the group:

```python-repl
>>> # Get total profit per column
>>> comb_portfolio.total_profit(group_by=False)
BTC-USD     5101.957521
ETH-USD    12866.045602
XRP-USD     3506.790882
BNB-USD     5065.017577
BCH-USD     -240.275095
LTC-USD     3149.106488
dtype: float64
```

In the same way, you can introduce new grouping to the method itself:

```python-repl
>>> # Get total profit per group
>>> portfolio.total_profit(group_by=group_by)
group
first     21474.794005
second     7973.848970
dtype: float64
```

!!! note
    If cash sharing is enabled, grouping can be disabled but cannot be modified.

### Indexing

In addition, you can use pandas indexing on the `Portfolio` class itself, which forwards
indexing operation to each argument with index:

```python-repl
>>> portfolio['BTC-USD']
<vectorbt.portfolio.base.Portfolio at 0x7fac7517ac88>

>>> portfolio['BTC-USD'].total_profit()
5101.957521326392
```

Combined portfolio is indexed by group:

```python-repl
>>> comb_portfolio['first']
<vectorbt.portfolio.base.Portfolio at 0x7fac5756b828>

>>> comb_portfolio['first'].total_profit()
21474.794005172986
```

!!! note
    Changing index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame; for example, use `portfolio.iloc[0]` instead of `portfolio.iloc[:, 0]`.

    Indexing behavior depends solely upon `vectorbt.base.array_wrapper.ArrayWrapper`.
    For example, if `group_select` is enabled indexing will be performed on groups,
    otherwise on single columns. You can pass wrapper arguments with `wrapper_kwargs`.

### Logging

To collect more information on how a specific order was processed or to be able to track the whole
simulation from the beginning to the end, you can turn on logging.

```python-repl
>>> # Simulate portfolio with logging
>>> portfolio = vbt.Portfolio.from_orders(
...     price['Close'], size, price=price['Open'],
...     init_cash='autoalign', fees=0.001, slippage=0.001, log=True
... )

>>> portfolio.logs().records
      idx  col  group  cash_now  shares_now  val_price_now  value_now  \
0       0    0      0       inf    0.000000        7294.44        inf
...   ...  ...    ...       ...         ...            ...        ...
1469  244    5      5       inf  273.894381          62.84        inf

          size  size_type  direction  ...  raise_reject   log  new_cash  \
0          NaN          0          2  ...         False  True       inf
...        ...        ...        ...  ...           ...   ...       ...
1469  7.956715          0          2  ...         False  True       inf

      new_shares  res_size   res_price  res_fees  res_side  res_status  \
0       0.000000       NaN         NaN       NaN        -1           1
...          ...       ...         ...       ...       ...         ...
1469  281.851096  7.956715    62.90284    0.5005         0           0
```

Just as orders, logs are also records and thus can be easily analyzed:

```python-repl
>>> portfolio.logs().map_field('res_status', value_map=vbt.OrderStatus).value_counts()
         BTC-USD  ETH-USD  XRP-USD  BNB-USD  BCH-USD  LTC-USD
Filled       186      169      172      171      177      180
Ignored       59       76       73       74       68       65
```

Logging can also be turned on just for one order, row, or column, since as many other
variables it's specified per order and can broadcast automatically.

!!! note
    Logging can slow down simulation.

### Caching

This class supports caching. If a method or a property requires heavy computation, it's wrapped
with `vectorbt.utils.decorators.cached_method` and `vectorbt.utils.decorators.cached_property` respectively.
Caching can be disabled globally via `vectorbt.defaults` or locally via the method/property.
There is currently no way to disable caching for an entire class.

!!! note
    Because of caching, this class is meant to be immutable and all properties are read-only.
    To change any attribute, use the `copy` method and pass the attribute as keyword argument.

!!! warning
    Make sure to disable caching when working with large arrays. Note that methods in `Portfolio`
    heavily depend upon each other, and a single call may trigger a chain of caching operations.
    For example, calling `Portfolio.total_return` caches 7 different time series of the same shape
    as the reference price.

    If caching is disabled, make sure to store most important time series manually. For example,
    if you're interested in Sharpe ratio or other metrics based on returns, run and save
    `Portfolio.returns` and then use the `vectorbt.returns.accessors.Returns_Accessor` to analyze them.
    Do not use methods akin to `Portfolio.sharpe_ratio` because they will re-calculate returns each time."""

import numpy as np
import pandas as pd
from inspect import signature

from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_method
from vectorbt.utils.enum import convert_str_enum_value
from vectorbt.utils.config import Configured, merge_kwargs
from vectorbt.utils.random import set_seed
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic import nb as generic_nb
from vectorbt.generic.drawdowns import Drawdowns
from vectorbt.records.base import records_indexing_func
from vectorbt.portfolio import nb
from vectorbt.portfolio.orders import Orders, indexing_on_orders_meta
from vectorbt.portfolio.trades import Trades, Positions
from vectorbt.portfolio.logs import Logs
from vectorbt.portfolio.enums import (
    InitCashMode,
    CallSeqType,
    SizeType,
    ConflictMode,
    Direction
)


def portfolio_indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Portfolio`."""
    new_orders, group_idxs, col_idxs = indexing_on_orders_meta(obj._orders, pd_indexing_func)
    new_logs = records_indexing_func(obj._logs, pd_indexing_func)
    if isinstance(obj._init_cash, int):
        new_init_cash = obj._init_cash
    else:
        new_init_cash = to_1d(obj._init_cash, raw=True)[group_idxs if obj.cash_sharing else col_idxs]
    new_call_seq = obj.call_seq.values[:, col_idxs]

    return obj.copy(
        orders=new_orders,
        logs=new_logs,
        init_cash=new_init_cash,
        call_seq=new_call_seq
    )


def add_returns_methods(func_names):
    """Class decorator to add `vectorbt.returns.accessors.Returns_Accessor` methods to `Portfolio`."""

    def wrapper(cls):
        for func_name in func_names:
            if isinstance(func_name, tuple):
                ret_func_name = func_name[0]
            else:
                ret_func_name = func_name

            def returns_method(
                    self,
                    *args,
                    group_by=None,
                    year_freq=None,
                    ret_func_name=ret_func_name,
                    active_returns=False,
                    in_sim_order=False,
                    reuse_returns=None,
                    **kwargs):
                if reuse_returns is not None:
                    returns = reuse_returns
                else:
                    if active_returns:
                        returns = self.active_returns(group_by=group_by)
                    else:
                        returns = self.returns(group_by=group_by, in_sim_order=in_sim_order)
                returns_acc = returns.vbt.returns(freq=self.wrapper.freq, year_freq=year_freq)
                # Select only those arguments in kwargs that are also in the method's signature
                # This is done for Portfolio.stats which passes the same kwargs to multiple methods
                method = getattr(returns_acc, ret_func_name)
                sig = signature(method)
                arg_names = [p.name for p in sig.parameters.values() if p.kind == p.POSITIONAL_OR_KEYWORD]
                new_kwargs = {}
                for arg_name in arg_names:
                    if arg_name in kwargs:
                        new_kwargs[arg_name] = kwargs[arg_name]
                return method(*args, **new_kwargs)

            if isinstance(func_name, tuple):
                func_name = func_name[1]
            returns_method.__name__ = func_name
            returns_method.__qualname__ = f"Portfolio.{func_name}"
            returns_method.__doc__ = f"See `vectorbt.returns.accessors.Returns_Accessor.{ret_func_name}`."
            setattr(cls, func_name, cached_method(returns_method))
        return cls

    return wrapper


@add_returns_methods([
    ('daily', 'daily_returns'),
    ('annual', 'annual_returns'),
    ('cumulative', 'cumulative_returns'),
    ('annualized', 'annualized_return'),
    'annualized_volatility',
    'calmar_ratio',
    'omega_ratio',
    'sharpe_ratio',
    'deflated_sharpe_ratio',
    'downside_risk',
    'sortino_ratio',
    'information_ratio',
    'beta',
    'alpha',
    'tail_ratio',
    'value_at_risk',
    'conditional_value_at_risk',
    'capture',
    'up_capture',
    'down_capture',
    'drawdown',
    'max_drawdown'
])
class Portfolio(Configured, PandasIndexer):
    """Class for modeling portfolio and measuring its performance.

    Args:
        orders (Orders): Order records of type `vectorbt.portfolio.orders.Orders`.
        logs (Logs): Log records of type `vectorbt.portfolio.logs.Logs`.
        init_cash (InitCashMode, float or array_like of float): Initial capital.
        cash_sharing (bool): Whether to share cash within the same group.
        call_seq (array_like of int): Sequence of calls per row and group.
        incl_unrealized (bool): Whether to include unrealized P&L in statistics.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

    !!! note
        This class is meant to be immutable. To change any attribute, use `Portfolio.copy`."""

    def __init__(self, orders, logs, init_cash, cash_sharing, call_seq, incl_unrealized=None):
        Configured.__init__(
            self,
            orders=orders,
            logs=logs,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            incl_unrealized=incl_unrealized
        )
        # Get defaults
        from vectorbt import defaults

        if incl_unrealized is None:
            incl_unrealized = defaults.portfolio['incl_unrealized']

        # Perform checks
        checks.assert_type(orders, Orders)
        checks.assert_type(logs, Logs)

        # Store passed arguments
        self._ref_price = orders.close
        self._orders = orders
        self._logs = logs
        self._init_cash = init_cash
        self._cash_sharing = cash_sharing
        self._call_seq = call_seq
        self._incl_unrealized = incl_unrealized

        # Supercharge
        PandasIndexer.__init__(self, portfolio_indexing_func)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls,
                     close,
                     entries, exits,
                     size=None, long_size=None, short_size=None,
                     price=None, long_price=None, short_price=None,
                     fees=None, long_fees=None, short_fees=None,
                     fixed_fees=None, long_fixed_fees=None, short_fixed_fees=None,
                     slippage=None, long_slippage=None, short_slippage=None,
                     min_size=None, long_min_size=None, short_min_size=None,
                     max_size=None, long_max_size=None, short_max_size=None,
                     reject_prob=None, long_reject_prob=None, short_reject_prob=None,
                     close_first=None, long_close_first=None, short_close_first=None,
                     allow_partial=None, long_allow_partial=None, short_allow_partial=None,
                     raise_reject=None, long_raise_reject=None, short_raise_reject=None,
                     accumulate=None, long_accumulate=None, short_accumulate=None,
                     log=None, long_log=None, short_log=None,
                     conflict_mode=None, direction=None, val_price=None,
                     init_cash=None, cash_sharing=None, call_seq=None, seed=None, freq=None,
                     group_by=None, broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Simulate portfolio from entry and exit signals.

        Starting with initial cash `init_cash`, for each signal in `entries`, enters a long/short
        position by buying/selling `size` of shares. For each signal in `exits`, closes the position
        by selling/buying shares. Depending upon accumulation options, each entry signal may increase
        the position and each exit signal may decrease the position. When both entry and exit signals
        are present, ignores them by default. When grouping is enabled with `group_by`, will compute
        performance for the entire group. When, additionally, `cash_sharing` is enabled, will share
        the cash among all columns in the group.

        Args:
            close (array_like): Reference price, such as close.
                Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            entries (array_like of bool): Boolean array of entry signals.
                Will broadcast.

                Becomes a long signal if `direction` is `all` or `longonly`, otherwise short.
            exits (array_like of bool): Boolean array of exit signals.
                Will broadcast.

                Becomes a short signal if `direction` is `all` or `longonly`, otherwise long.
            size (float or array_like): Size to order.
                Will broadcast.

                * Set to any number to buy/sell some fixed amount of shares.
                    Longs are limited by cash in the account, while shorts are unlimited.
                * Set to `np.inf` to buy shares for all cash, or `-np.inf` to sell shares for
                    initial margin of 100%. If `direction` is not `all`, `-np.inf` will close the position.
                * Set to `np.nan` or 0 to skip.

                !!! note
                    Sign will be ignored.
            long_size (float or array_like): Overwrites `size` for long orders.
                Defaults to `size`. Will broadcast.
            short_size (float or array_like): Overwrites `size` for short orders.
                Defaults to `size`. Will broadcast.
            price (array_like of float): Order price.
                Defaults to `close`. Will broadcast.
            long_price (array_like of float): Overwrites `price` for long orders.
                Defaults to `price`. Will broadcast.
            short_price (array_like of float): Overwrites `price` for short orders.
                Defaults to `price`. Will broadcast.
            fees (float or array_like): Fees in percentage of the order value.
                Will broadcast.
            long_fees (float or array_like): Overwrites `fees` for long orders.
                Defaults to `fees`. Will broadcast.
            short_fees (float or array_like): Overwrites `fees` for short orders.
                Defaults to `fees`. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order.
                Will broadcast.
            long_fixed_fees (float or array_like): Overwrites `fixed_fees` for long orders.
                Defaults to `fixed_fees`. Will broadcast.
            short_fixed_fees (float or array_like): Overwrites `fixed_fees` for short orders.
                Defaults to `fixed_fees`. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price.
                Will broadcast.
            long_slippage (float or array_like): Overwrites `slippage` for long orders.
                Defaults to `slippage`. Will broadcast.
            short_slippage (float or array_like): Overwrites `slippage` for short orders.
                Defaults to `slippage`. Will broadcast.
            min_size (float or array_like): Minimum size for an order to be accepted.
                Will broadcast.
            long_min_size (float or array_like): Overwrites `min_size` for long orders.
                Defaults to `min_size`. Will broadcast.
            short_min_size (float or array_like): Overwrites `min_size` for short orders.
                Defaults to `min_size`. Will broadcast.
            max_size (float or array_like): Maximum size for an order.
                Will broadcast.

                Will be partially filled if exceeded. You might not be able to properly close
                the position if accumulation is enabled and `max_size` is too low.
            long_max_size (float or array_like): Overwrites `max_size` for long orders.
                Defaults to `max_size`. Will broadcast.
            short_max_size (float or array_like): Overwrites `max_size` for short orders.
                Defaults to `max_size`. Will broadcast.
            reject_prob (float or array_like): Order rejection probability.
                Will broadcast.
            long_reject_prob (float or array_like): Overwrites `reject_prob` for long orders.
                Defaults to `reject_prob`. Will broadcast.
            short_reject_prob (float or array_like): Overwrites `reject_prob` for short orders.
                Defaults to `reject_prob`. Will broadcast.
            close_first (bool or array_like): Whether to close the position first before reversal.
                Will broadcast.

                See `close_first` in `Portfolio.from_order_func`.
            long_close_first (bool or array_like): Overwrites `close_first` for long orders.
                Defaults to `close_first`. Will broadcast.
            short_close_first (bool or array_like): Overwrites `close_first` for short orders.
                Defaults to `close_first`. Will broadcast.
            allow_partial (bool or array_like): Whether to allow partial fills.
                Will broadcast.

                Does not apply when size is `np.inf`.
            long_allow_partial (bool or array_like): Overwrites `allow_partial` for long orders.
                Defaults to `allow_partial`. Will broadcast.
            short_allow_partial (bool or array_like): Overwrites `allow_partial` for short orders.
                Defaults to `allow_partial`. Will broadcast.
            raise_reject (bool or array_like): Whether to raise an exception if order gets rejected.
                Will broadcast.
            long_raise_reject (bool or array_like): Overwrites `raise_reject` for long orders.
                Defaults to `raise_reject`. Will broadcast.
            short_raise_reject (bool or array_like): Overwrites `raise_reject` for short orders.
                Defaults to `raise_reject`. Will broadcast.
            log (bool or array_like): Whether to log orders.
                Will broadcast.
            long_log (bool or array_like): Overwrites `log` for long orders.
                Defaults to `log`. Will broadcast.
            short_log (bool or array_like): Overwrites `log` for short orders.
                Defaults to `log`. Will broadcast.
            accumulate (bool or array_like): Whether to accumulate signals.
                Will broadcast.

                Behaves similarly to `Portfolio.from_orders`.
            long_accumulate (bool or array_like): Overwrites `accumulate` for long orders.
                Defaults to `accumulate`. Will broadcast.
            short_accumulate (bool or array_like): Overwrites `accumulate` for short orders.
                Defaults to `accumulate`. Will broadcast.
            conflict_mode (ConflictMode or array_like): See `vectorbt.portfolio.enums.ConflictMode`.
                Will broadcast.
            direction (Direction or array_like): See `vectorbt.portfolio.enums.Direction`.
                Will broadcast.
            val_price (array_like of float): Asset valuation price.
                Defaults to `price` if set, otherwise to previous `close`.
                
                See `val_price` in `Portfolio.from_orders`.
            init_cash (InitCashMode, float or array_like of float): Initial capital.

                See `init_cash` in `Portfolio.from_order_func`.
            cash_sharing (bool): Whether to share cash within the same group.

                See `cash_sharing` in `Portfolio.from_orders`.
            call_seq (CallSeqType or array_like of int): Default sequence of calls per row and group.

                See `call_seq` in `Portfolio.from_orders`.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            freq (any): Index frequency in case `close.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.
            **kwargs: Keyword arguments passed to the `__init__` method.

        All broadcastable arguments will be broadcast using `vectorbt.base.reshape_fns.broadcast`
        but keep original shape to utilize flexible indexing and to save memory.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            Arguments with prefix `short_` don't necessarily apply to short positions, but to short orders.
            For example, a single short order can close a long position and open a short one. To have
            a more fine-grained control over each order, set those arguments per row, column, or element,
            or use a more general simulation method such as `Portfolio.from_orders`.

        !!! note
            Only `SizeType.Shares` is supported. Other modes such as target percentage are not
            compatible with signals since their logic may contradict the direction the user has
            specified for the order.

        !!! hint
            If you generated signals using close price, don't forget to shift your signals by one tick
            forward, for example, with `signals.vbt.fshift(1)`. In general, make sure to use a price
            that comes after the signal.
            
        Also see notes and hints for `Portfolio.from_orders`.

        Example:
            Some of the ways of how signals are interpreted:

            ```python-repl
            >>> import pandas as pd
            >>> import vectorbt as vbt

            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> entries = pd.Series([True, True, True, False, False])
            >>> exits = pd.Series([False, False, True, True, True])

            >>> # Entry opens long, exit closes long
            >>> portfolio = vbt.Portfolio.from_signals(
            ...     close, entries, exits, size=1., direction='longonly')
            >>> portfolio.share_flow()
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4    0.0
            dtype: float64

            >>> # Entry opens short, exit closes short
            >>> portfolio = vbt.Portfolio.from_signals(
            ...     close, entries, exits, size=1., direction='shortonly')
            >>> portfolio.share_flow()
            0   -1.0
            1    0.0
            2    0.0
            3    1.0
            4    0.0
            dtype: float64

            >>> # Entry opens long and closes short, exit closes long and opens short
            >>> # Reversal within one tick
            >>> portfolio = vbt.Portfolio.from_signals(
            ...     close, entries, exits, size=1., direction='all')
            >>> portfolio.share_flow()
            0    1.0
            1    0.0
            2    0.0
            3   -2.0
            4    0.0
            dtype: float64

            >>> # Reversal within two ticks
            >>> # First signal closes position, second signal opens the opposite one
            >>> portfolio = vbt.Portfolio.from_signals(
            ...     close, entries, exits, size=1., direction='all',
            ...     close_first=True)
            >>> portfolio.share_flow()
            0    1.0
            1    0.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64

            >>> # If entry and exit, chooses exit
            >>> portfolio = vbt.Portfolio.from_signals(
            ...     close, entries, exits, size=1., direction='all',
            ...     close_first=True, conflict_mode='exit')
            >>> portfolio.share_flow()
            0    1.0
            1    0.0
            2   -1.0
            3   -1.0
            4    0.0
            dtype: float64

            >>> # Entry means long order, exit means short order
            >>> # Acts similar to `from_orders`
            >>> portfolio = vbt.Portfolio.from_signals(
            ...     close, entries, exits, size=1., accumulate=True)
            >>> portfolio.share_flow()
            0    1.0
            1    1.0
            2    0.0
            3   -1.0
            4   -1.0
            dtype: float64

            >>> # Testing multiple parameters (via broadcasting)
            >>> portfolio = vbt.Portfolio.from_signals(
            ...     close, entries, exits, direction=[list(vbt.Direction)],
            ...     broadcast_kwargs=dict(columns_from=vbt.Direction._fields))
            >>> portfolio.share_flow()
                Long  Short    All
            0  100.0 -100.0  100.0
            1    0.0    0.0    0.0
            2    0.0    0.0    0.0
            3 -100.0   50.0 -200.0
            4    0.0    0.0    0.0
            ```
        """
        # Get defaults
        from vectorbt import defaults

        if size is None:
            size = defaults.portfolio['size']
        if long_size is None:
            long_size = size
        if short_size is None:
            short_size = size
        if price is None:
            price = close
        if long_price is None:
            long_price = price
        if short_price is None:
            short_price = price
        if fees is None:
            fees = defaults.portfolio['fees']
        if long_fees is None:
            long_fees = fees
        if short_fees is None:
            short_fees = fees
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if long_fixed_fees is None:
            long_fixed_fees = fixed_fees
        if short_fixed_fees is None:
            short_fixed_fees = fixed_fees
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if long_slippage is None:
            long_slippage = slippage
        if short_slippage is None:
            short_slippage = slippage
        if min_size is None:
            min_size = defaults.portfolio['min_size']
        if long_min_size is None:
            long_min_size = min_size
        if short_min_size is None:
            short_min_size = min_size
        if max_size is None:
            max_size = defaults.portfolio['max_size']
        if long_max_size is None:
            long_max_size = max_size
        if short_max_size is None:
            short_max_size = max_size
        if reject_prob is None:
            reject_prob = defaults.portfolio['reject_prob']
        if long_reject_prob is None:
            long_reject_prob = reject_prob
        if short_reject_prob is None:
            short_reject_prob = reject_prob
        if close_first is None:
            close_first = defaults.portfolio['close_first']
        if long_close_first is None:
            long_close_first = close_first
        if short_close_first is None:
            short_close_first = close_first
        if allow_partial is None:
            allow_partial = defaults.portfolio['allow_partial']
        if long_allow_partial is None:
            long_allow_partial = allow_partial
        if short_allow_partial is None:
            short_allow_partial = allow_partial
        if raise_reject is None:
            raise_reject = defaults.portfolio['raise_reject']
        if long_raise_reject is None:
            long_raise_reject = raise_reject
        if short_raise_reject is None:
            short_raise_reject = raise_reject
        if log is None:
            log = defaults.portfolio['log']
        if long_log is None:
            long_log = log
        if short_log is None:
            short_log = log
        if accumulate is None:
            accumulate = defaults.portfolio['accumulate']
        if long_accumulate is None:
            long_accumulate = accumulate
        if short_accumulate is None:
            short_accumulate = accumulate
        if conflict_mode is None:
            conflict_mode = defaults.portfolio['conflict_mode']
        conflict_mode = convert_str_enum_value(ConflictMode, conflict_mode)
        if direction is None:
            direction = defaults.portfolio['direction']
        direction = convert_str_enum_value(Direction, direction)
        if val_price is None:
            if price is None:
                if checks.is_pandas(close):
                    val_price = close.vbt.fshift(1)
                else:
                    val_price = np.require(close, dtype=np.float_)
                    val_price = np.roll(val_price, 1, axis=0)
                    val_price[0] = np.nan
            else:
                val_price = price
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        init_cash = convert_str_enum_value(InitCashMode, init_cash)
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
        call_seq = convert_str_enum_value(CallSeqType, call_seq)
        auto_call_seq = False
        if isinstance(call_seq, int):
            if call_seq == CallSeqType.Auto:
                call_seq = CallSeqType.Default
                auto_call_seq = True
        if seed is None:
            seed = defaults.portfolio['seed']
        if seed is not None:
            set_seed(seed)
        if freq is None:
            freq = defaults.portfolio['freq']
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Broadcast inputs
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        broadcastable_args = (
            close, entries, exits,
            long_size, short_size,
            long_price, short_price,
            long_fees, short_fees,
            long_fixed_fees, short_fixed_fees,
            long_slippage, short_slippage,
            long_min_size, short_min_size,
            long_max_size, short_max_size,
            long_reject_prob, short_reject_prob,
            long_close_first, short_close_first,
            long_allow_partial, short_allow_partial,
            long_raise_reject, short_raise_reject,
            long_accumulate, short_accumulate,
            long_log, short_log,
            conflict_mode, direction, val_price
        )
        keep_raw = [False] + [True] * (len(broadcastable_args) - 1)
        broadcast_kwargs = merge_kwargs(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        broadcasted_args = broadcast(*broadcastable_args, **broadcast_kwargs, keep_raw=keep_raw)
        close = broadcasted_args[0]
        if not checks.is_pandas(close):
            close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)

        # Perform calculation
        order_records, log_records = nb.simulate_from_signals_nb(
            target_shape_2d,
            cs_group_lens,  # group only if cash sharing is enabled to speed up
            init_cash,
            call_seq,
            auto_call_seq,
            *broadcasted_args[1:],
            close.ndim == 2
        )

        # Create an instance
        orders = Orders(wrapper, order_records, close)
        logs = Logs(wrapper, log_records)
        return cls(
            orders,
            logs,
            init_cash if init_cash_mode is None else init_cash_mode,
            cash_sharing,
            call_seq,
            **kwargs
        )

    @classmethod
    def from_orders(cls, close, size, size_type=None, direction=None, price=None, fees=None,
                    fixed_fees=None, slippage=None, min_size=None, max_size=None, reject_prob=None,
                    close_first=None, allow_partial=None, raise_reject=None, log=None, val_price=None,
                    init_cash=None, cash_sharing=None, call_seq=None, freq=None, seed=None,
                    group_by=None, broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Simulate portfolio from orders.

        Starting with initial cash `init_cash`, orders the number of shares specified in `size`
        for `price`.

        Args:
            close (array_like): Reference price, such as close.
                Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            size (float or array_like): Size to order.
                Will broadcast.

                Behavior depends upon `size_type` and `direction`. For `SizeType.Shares`:

                * Set to any number to buy/sell some fixed amount of shares.
                    Longs are limited by cash in the account, while shorts are unlimited.
                * Set to `np.inf` to buy shares for all cash, or `-np.inf` to sell shares for
                    initial margin of 100%. If `direction` is not `all`, `-np.inf` will close the position.
                * Set to `np.nan` or 0 to skip.

                For any target size:

                * Set to any number to buy/sell amount of shares relative to current holdings or value.
                * Set to 0 to close the current position.
                * Set to `np.nan` to skip.
            size_type (SizeType or array_like): See `vectorbt.portfolio.enums.SizeType`.
                Will broadcast.
            direction (Direction or array_like): See `vectorbt.portfolio.enums.Direction`.
                Will broadcast.
            price (array_like of float): Order price.
                Defaults to `close`. Will broadcast.
            fees (float or array_like): Fees in percentage of the order value.
                Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order.
                Will broadcast.
            slippage (float or array_like): Slippage in percentage of price.
                Will broadcast.
            min_size (float or array_like): Minimum size for an order to be accepted.
                Will broadcast.
            max_size (float or array_like): Maximum size for an order.
                Will broadcast.

                Will be partially filled if exceeded.
            reject_prob (float or array_like): Order rejection probability.
                Will broadcast.
            close_first (bool or array_like): Whether to close the position first before reversal.
                Will broadcast.

                Otherwise reverses the position with a single order and within the same tick.
                Takes only effect under `Direction.All`. Requires a second signal to enter
                the opposite position. This allows to define parameters such as `fixed_fees` for long
                and short positions separately.
            allow_partial (bool or array_like): Whether to allow partial fills.
                Will broadcast.

                Does not apply when size is `np.inf`.
            raise_reject (bool or array_like): Whether to raise an exception if order gets rejected.
                Will broadcast.
            log (bool or array_like): Whether to log orders.
                Will broadcast.
            val_price (array_like of float): Asset valuation price.
                Defaults to `price`. Will broadcast.

                Used at the time of decision making to calculate value of each asset in the group,
                for example, to convert target value into target shares.

                !!! note
                    Make sure to use timestamp for `val_price` that comes before timestamps of
                    all orders in the group with cash sharing (previous `close` for example),
                    otherwise you're cheating yourself.
            init_cash (InitCashMode, float or array_like of float): Initial capital.

                See `init_cash` in `Portfolio.from_order_func`.
            cash_sharing (bool): Whether to share cash within the same group.

                !!! warning
                    Introduces cross-asset dependencies.

                    This method presumes that in a group of assets that share the same capital all
                    orders will be executed within the same tick and retain their price regardless
                    of their position in the queue, even though they depend upon each other and thus
                    cannot be executed in parallel.
            call_seq (CallSeqType or array_like of int): Default sequence of calls per row and group.

                Each value in this sequence should indicate the position of column in the group to
                call next. Processing of `call_seq` goes always from left to right.
                For example, `[2, 0, 1]` would first call column 'c', then 'a', and finally 'b'.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.

                If `CallSeqType.Auto` selected, rearranges calls dynamically based on order value.
                Calculates value of all orders per row and group, and sorts them by this value.
                Sell orders will be executed first to release funds for buy orders.

                !!! warning
                    `CallSeqType.Auto` should be used with caution:

                    * It not only presumes that order prices are known beforehand, but also that
                        orders can be executed in arbitrary order and still retain their price.
                        In reality, this is hardly the case: after processing one asset, some time
                        has passed and the price for other assets might have already changed.
                    * Even if you're able to specify a slippage large enough to compensate for
                        this behavior, slippage itself should depend upon execution order.
                        This method doesn't let you do that.
                    * If one order is rejected, it still may execute next orders and possibly
                        leave them without required funds.

                    For more control, use `Portfolio.from_order_func`.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            freq (any): Index frequency in case `close.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.

        All broadcastable arguments will be broadcast using `vectorbt.base.reshape_fns.broadcast`
        but keep original shape to utilize flexible indexing and to save memory.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            When `call_seq` is not `CallSeqType.Auto`, at each timestamp, processing of the assets in
            a group goes strictly in order defined in `call_seq`. This order can't be changed dynamically.

            This has one big implication for this particular method: the last asset in the call stack
            cannot be processed until other assets are processed. This is the reason why rebalancing
            cannot work properly in this setting: one has to specify percentages for all assets beforehand
            and then tweak the processing order to sell to-be-sold assets first in order to release funds
            for to-be-bought assets. This can be automatically done by using `CallSeqType.Auto`.

        !!! hint
            All broadcastable arguments can be set per frame, series, row, column, or element.

        Example:
            Buy 10 shares each tick:
            ```python-repl
            >>> import pandas as pd
            >>> import vectorbt as vbt

            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> portfolio = vbt.Portfolio.from_orders(close, 10)

            >>> portfolio.shares()
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> portfolio.cash()
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            Reverse each position by first closing it:
            ```python-repl
            >>> import numpy as np

            >>> size = [np.inf, -np.inf, -np.inf, np.inf, np.inf]
            >>> portfolio = vbt.Portfolio.from_orders(
            ...     close, size, close_first=True)

            >>> portfolio.shares()
            0    100.000000
            1      0.000000
            2    -66.666667
            3      0.000000
            4     26.666667
            dtype: float64
            >>> portfolio.cash()
            0      0.000000
            1    200.000000
            2    400.000000
            3    133.333333
            4      0.000000
            dtype: float64
            ```

            Equal-weighted portfolio as in `vectorbt.portfolio.nb.simulate_nb` example:
            It's more compact but has less control over execution:

            ```python-repl
            >>> np.random.seed(42)
            >>> close = pd.DataFrame(np.random.uniform(1, 10, size=(5, 3)))
            >>> size = pd.Series(np.full(5, 1/3))  # each column 33.3%
            >>> size[1::2] = np.nan  # skip every second tick

            >>> portfolio = vbt.Portfolio.from_orders(
            ...     close,  # acts both as reference and order price here
            ...     size,
            ...     size_type='targetpercent',
            ...     call_seq='auto',  # first sell then buy
            ...     group_by=True,  # one group
            ...     cash_sharing=True,  # assets share the same cash
            ...     fees=0.001, fixed_fees=1., slippage=0.001  # costs
            ... )

            >>> portfolio.holding_value(group_by=False).vbt.scatter()
            ```

            ![](/vectorbt/docs/img/simulate_nb.png)
        """
        # Get defaults
        from vectorbt import defaults

        if size is None:
            size = defaults.portfolio['size']
        if size_type is None:
            size_type = defaults.portfolio['size_type']
        size_type = convert_str_enum_value(SizeType, size_type)
        if direction is None:
            direction = defaults.portfolio['direction']
        direction = convert_str_enum_value(Direction, direction)
        if price is None:
            price = close
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if min_size is None:
            min_size = defaults.portfolio['min_size']
        if max_size is None:
            max_size = defaults.portfolio['max_size']
        if reject_prob is None:
            reject_prob = defaults.portfolio['reject_prob']
        if close_first is None:
            close_first = defaults.portfolio['close_first']
        if allow_partial is None:
            allow_partial = defaults.portfolio['allow_partial']
        if raise_reject is None:
            raise_reject = defaults.portfolio['raise_reject']
        if log is None:
            log = defaults.portfolio['log']
        if val_price is None:
            val_price = price
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        init_cash = convert_str_enum_value(InitCashMode, init_cash)
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
        call_seq = convert_str_enum_value(CallSeqType, call_seq)
        auto_call_seq = False
        if isinstance(call_seq, int):
            if call_seq == CallSeqType.Auto:
                call_seq = CallSeqType.Default
                auto_call_seq = True
        if seed is None:
            seed = defaults.portfolio['seed']
        if seed is not None:
            set_seed(seed)
        if freq is None:
            freq = defaults.portfolio['freq']
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Broadcast inputs
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        broadcastable_args = (
            close,
            size,
            size_type,
            direction,
            price,
            fees,
            fixed_fees,
            slippage,
            min_size,
            max_size,
            reject_prob,
            close_first,
            allow_partial,
            raise_reject,
            log,
            val_price
        )
        keep_raw = [False] + [True] * (len(broadcastable_args) - 1)
        broadcast_kwargs = merge_kwargs(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        broadcasted_args = broadcast(*broadcastable_args, **broadcast_kwargs, keep_raw=keep_raw)
        close = broadcasted_args[0]
        if not checks.is_pandas(close):
            close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)

        # Perform calculation
        order_records, log_records = nb.simulate_from_orders_nb(
            target_shape_2d,
            cs_group_lens,  # group only if cash sharing is enabled to speed up
            init_cash,
            call_seq,
            auto_call_seq,
            *broadcasted_args[1:],
            close.ndim == 2
        )

        # Create an instance
        orders = Orders(wrapper, order_records, close)
        logs = Logs(wrapper, log_records)
        return cls(
            orders,
            logs,
            init_cash if init_cash_mode is None else init_cash_mode,
            cash_sharing,
            call_seq,
            **kwargs
        )

    @classmethod
    def from_order_func(cls, close, order_func_nb, *order_args, target_shape=None, keys=None,
                        init_cash=None, cash_sharing=None, call_seq=None, active_mask=None,
                        prep_func_nb=None, prep_args=None, group_prep_func_nb=None, group_prep_args=None,
                        row_prep_func_nb=None, row_prep_args=None, segment_prep_func_nb=None,
                        segment_prep_args=None, row_wise=None, seed=None, freq=None, group_by=None,
                        broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Build portfolio from a custom order function.

        For details, see `vectorbt.portfolio.nb.simulate_nb`.

        if `row_wise` is True, also see `vectorbt.portfolio.nb.simulate_row_wise_nb`.

        Args:
            close (array_like): Reference price, such as close.
                Will broadcast to `target_shape`.

                Will be used for calculating unrealized P&L and portfolio value.

                !!! note
                    In contrast to other methods, the valuation price is previous `close`
                    instead of order price, since the price of an order is unknown before call.
                    You can still set valuation price explicitly in `segment_prep_func_nb`.
            order_func_nb (callable): Order generation function.
            *order_args: Arguments passed to `order_func_nb`.
            target_shape (tuple): Target shape to iterate over. Defaults to `close.shape`.
            keys (sequence): Outermost column level.

                Each element should correspond to one iteration over columns in `close`.
                Should be set only if `target_shape` is bigger than `close.shape`.
            init_cash (InitCashMode, float or array_like of float): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
                See `vectorbt.portfolio.enums.InitCashMode` to find optimal initial cash.

                !!! note
                    Mode `InitCashMode.AutoAlign` is applied after the portfolio is initialized
                    to set the same initial cash for all columns/groups. Changing grouping
                    will change the initial cash, so be aware when indexing.
            cash_sharing (bool): Whether to share cash within the same group.

                !!! warning
                    Introduces cross-asset dependencies.
            call_seq (CallSeqType or array_like of int): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.

                !!! note
                    CallSeqType.Auto should be implemented manually.
                    Use `auto_call_seq_ctx_nb` in `segment_prep_func_nb`.
            active_mask (int or array_like of bool): Mask of whether a particular segment should be executed.

                Supplying an integer will activate every n-th row (just for convenience).
                Supplying a boolean will broadcast to the number of rows and groups.
            prep_func_nb (callable): Simulation preparation function.
            prep_args (tuple): Packed arguments passed to `prep_func_nb`.

                Defaults to `()`.
            group_prep_func_nb (callable): Group preparation function.

                Called only if `row_wise` is False.
            group_prep_args (tuple): Packed arguments passed to `group_prep_func_nb`.

                Defaults to `()`.
            row_prep_func_nb (callable): Row preparation function.

                Called only if `row_wise` is True.
            row_prep_args (tuple): Packed arguments passed to `row_prep_func_nb`.

                Defaults to `()`.
            segment_prep_func_nb (callable): Segment preparation function.
            segment_prep_args (tuple): Packed arguments passed to `segment_prep_func_nb`.

                Defaults to `()`.
            row_wise (bool): Whether to iterate over rows rather than columns/groups.

                See `vectorbt.portfolio.nb.simulate_row_wise_nb`.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            freq (any): Index frequency in case `close.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            All passed functions should be Numba-compiled.

            Objects passed as arguments to both functions will not broadcast to `target_shape`
            as their purpose is unknown. You should broadcast manually or use flexible indexing.

            Also see notes on `Portfolio.from_orders`.

        Example:
            Buy 10 shares each tick:
            ```python-repl
            >>> import pandas as pd
            >>> from numba import njit
            >>> import vectorbt as vbt
            >>> from vectorbt.portfolio.nb import create_order_nb

            >>> @njit
            ... def order_func_nb(oc, size):
            ...     return create_order_nb(size=size, price=oc.close[oc.i, oc.col])

            >>> close = pd.Series([1, 2, 3, 4, 5])
            >>> portfolio = vbt.Portfolio.from_order_func(close, order_func_nb, 10)

            >>> portfolio.shares()
            0    10.0
            1    20.0
            2    30.0
            3    40.0
            4    40.0
            dtype: float64
            >>> portfolio.cash()
            0    90.0
            1    70.0
            2    40.0
            3     0.0
            4     0.0
            dtype: float64
            ```

            Reverse each position by first closing it. Keep state of last position to determine
            which position to open next (just for example, there are easier ways to do this):
            ```python-repl
            >>> import numpy as np

            >>> @njit
            ... def group_prep_func_nb(gc):
            ...     last_pos_state = np.array([-1])
            ...     return (last_pos_state,)

            >>> @njit
            ... def order_func_nb(oc, last_pos_state):
            ...     if oc.shares_now > 0:
            ...         size = -oc.shares_now  # close long
            ...     elif oc.shares_now < 0:
            ...         size = -oc.shares_now  # close short
            ...     else:
            ...         if last_pos_state[0] == 1:
            ...             size = -np.inf  # open short
            ...             last_pos_state[0] = -1
            ...         else:
            ...             size = np.inf  # open long
            ...             last_pos_state[0] = 1
            ...
            ...     return create_order_nb(size=size, price=oc.close[oc.i, oc.col])

            >>> portfolio = vbt.Portfolio.from_order_func(
            ...     close, order_func_nb, group_prep_func_nb=group_prep_func_nb)

            >>> portfolio.shares()
            0    100.0
            1      0.0
            2   -100.0
            3      0.0
            4     20.0
            dtype: float64
            >>> portfolio.cash()
            0      0.0
            1    200.0
            2    500.0
            3    100.0
            4      0.0
            dtype: float64
            ```

            Equal-weighted portfolio as in `vectorbt.portfolio.nb.simulate_nb` example:
            ```python-repl
            >>> from vectorbt.portfolio.nb import auto_call_seq_ctx_nb
            >>> from vectorbt.portfolio.enums import SizeType, Direction

            >>> @njit
            ... def group_prep_func_nb(gc):
            ...     '''Define empty arrays for each group.'''
            ...     size = np.empty(gc.group_len, dtype=np.float_)
            ...     size_type = np.empty(gc.group_len, dtype=np.int_)
            ...     direction = np.empty(gc.group_len, dtype=np.int_)
            ...     temp_float_arr = np.empty(gc.group_len, dtype=np.float_)
            ...     return size, size_type, direction, temp_float_arr

            >>> @njit
            ... def segment_prep_func_nb(sc, size, size_type, direction, temp_float_arr):
            ...     '''Perform rebalancing at each segment.'''
            ...     for k in range(sc.group_len):
            ...         col = sc.from_col + k
            ...         size[k] = 1 / sc.group_len
            ...         size_type[k] = SizeType.TargetPercent
            ...         direction[k] = Direction.LongOnly
            ...         sc.last_val_price[col] = sc.close[sc.i, col]
            ...     auto_call_seq_ctx_nb(sc, size, size_type, direction, temp_float_arr)
            ...     return size, size_type, direction

            >>> @njit
            ... def order_func_nb(oc, size, size_type, direction, fees, fixed_fees, slippage):
            ...     '''Place an order.'''
            ...     col_i = oc.call_seq_now[oc.call_idx]
            ...     return create_order_nb(
            ...         size=size[col_i],
            ...         size_type=size_type[col_i],
            ...         price=oc.close[oc.i, oc.col],
            ...         fees=fees, fixed_fees=fixed_fees, slippage=slippage,
            ...         direction=direction[col_i]
            ...     )

            >>> np.random.seed(42)
            >>> close = np.random.uniform(1, 10, size=(5, 3))
            >>> fees = 0.001
            >>> fixed_fees = 1.
            >>> slippage = 0.001

            >>> portfolio = vbt.Portfolio.from_order_func(
            ...     close,  # acts both as reference and order price here
            ...     order_func_nb, fees, fixed_fees, slippage,  # order_args as *args
            ...     active_mask=2,  # rebalance every second tick
            ...     group_prep_func_nb=group_prep_func_nb,
            ...     segment_prep_func_nb=segment_prep_func_nb,
            ...     cash_sharing=True, group_by=True,  # one group with cash sharing
            ... )

            >>> portfolio.holding_value(group_by=False).vbt.scatter()
            ```

            ![](/vectorbt/docs/img/simulate_nb.png)
        """
        # Get defaults
        from vectorbt import defaults

        if not checks.is_pandas(close):
            if not checks.is_array(close):
                close = np.asarray(close)
            close = pd.Series(close) if close.ndim == 1 else pd.DataFrame(close)
        if target_shape is None:
            target_shape = close.shape
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        init_cash = convert_str_enum_value(InitCashMode, init_cash)
        if isinstance(init_cash, int) and init_cash in InitCashMode:
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
        call_seq = convert_str_enum_value(CallSeqType, call_seq)
        if isinstance(call_seq, int):
            if call_seq == CallSeqType.Auto:
                raise ValueError("CallSeqType.Auto should be implemented manually. "
                                 "Use auto_call_seq_ctx_nb in segment_prep_func_nb.")
        if active_mask is None:
            active_mask = True
        if row_wise is None:
            row_wise = defaults.portfolio['row_wise']
        if seed is None:
            seed = defaults.portfolio['seed']
        if seed is not None:
            set_seed(seed)
        if freq is None:
            freq = defaults.portfolio['freq']
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        require_kwargs = dict(require_kwargs=dict(requirements='W'))
        broadcast_kwargs = merge_kwargs(require_kwargs, broadcast_kwargs)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Broadcast inputs
        target_shape_2d = (target_shape[0], target_shape[1] if len(target_shape) > 1 else 1)
        if close.shape != target_shape:
            if len(close.vbt.columns) <= target_shape_2d[1]:
                if target_shape_2d[1] % len(close.vbt.columns) != 0:
                    raise ValueError("Cannot broadcast close to target_shape")
                if keys is None:
                    keys = pd.Index(np.arange(target_shape_2d[1]), name='iteration_idx')
                tile_times = target_shape_2d[1] // len(close.vbt.columns)
                close = close.vbt.tile(tile_times, keys=keys)
        close = broadcast(close, to_shape=target_shape, **broadcast_kwargs)
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_lens = wrapper.grouper.get_group_lens(group_by=None if cash_sharing else False)
        init_cash = np.require(np.broadcast_to(init_cash, (len(cs_group_lens),)), dtype=np.float_)
        group_lens = wrapper.grouper.get_group_lens(group_by=group_by)
        if isinstance(active_mask, int):
            _active_mask = np.full((target_shape_2d[0], len(group_lens)), False)
            _active_mask[0::active_mask] = True
            active_mask = _active_mask
        else:
            active_mask = broadcast(
                active_mask,
                to_shape=(target_shape_2d[0], len(group_lens)),
                to_pd=False,
                **require_kwargs
            )
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape_2d, group_lens, call_seq_type=call_seq)

        # Prepare arguments
        if prep_func_nb is None:
            prep_func_nb = nb.empty_prep_nb
        if prep_args is None:
            prep_args = ()
        if group_prep_func_nb is None:
            group_prep_func_nb = nb.empty_prep_nb
        if group_prep_args is None:
            group_prep_args = ()
        if row_prep_func_nb is None:
            row_prep_func_nb = nb.empty_prep_nb
        if row_prep_args is None:
            row_prep_args = ()
        if segment_prep_func_nb is None:
            segment_prep_func_nb = nb.empty_prep_nb
        if segment_prep_args is None:
            segment_prep_args = ()

        # Perform calculation
        if row_wise:
            order_records, log_records = nb.simulate_row_wise_nb(
                target_shape_2d,
                to_2d(close, raw=True),
                group_lens,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
                prep_func_nb,
                prep_args,
                row_prep_func_nb,
                row_prep_args,
                segment_prep_func_nb,
                segment_prep_args,
                order_func_nb,
                order_args
            )
        else:
            order_records, log_records = nb.simulate_nb(
                target_shape_2d,
                to_2d(close, raw=True),
                group_lens,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
                prep_func_nb,
                prep_args,
                group_prep_func_nb,
                group_prep_args,
                segment_prep_func_nb,
                segment_prep_args,
                order_func_nb,
                order_args
            )

        # Create an instance
        orders = Orders(wrapper, order_records, close)
        logs = Logs(wrapper, log_records)
        return cls(
            orders,
            logs,
            init_cash if init_cash_mode is None else init_cash_mode,
            cash_sharing,
            call_seq,
            **kwargs
        )

    # ############# Properties ############# #

    @property
    def wrapper(self):
        """Array wrapper."""
        # Wrapper in orders and here can be different
        wrapper = self._orders.wrapper
        if self.cash_sharing and wrapper.grouper.allow_modify:
            # Cannot change groups if columns within them are dependent
            return wrapper.copy(allow_modify=False)
        return wrapper.copy()

    @property
    def cash_sharing(self):
        """Whether to share cash within the same group."""
        return self._cash_sharing

    @property
    def call_seq(self):
        """Sequence of calls per row and group."""
        return self.wrapper.wrap(self._call_seq, group_by=False)

    @property
    def incl_unrealized(self):
        """Whether to include unrealized trade P&L in statistics."""
        return self._incl_unrealized

    @property  # lazy property
    def _trades(self):
        _trades = Trades.from_orders(self._orders)
        self.__dict__['_trades'] = _trades
        return _trades

    @property  # lazy property
    def _positions(self):
        _positions = Positions.from_trades(self._trades)
        self.__dict__['_positions'] = _positions
        return _positions

    # ############# Regrouping ############# #

    def regroup(self, group_by):
        """Regroup this object."""
        if self.cash_sharing:
            raise ValueError("Cannot change grouping globally when cash sharing is enabled")
        if self.wrapper.grouper.is_grouping_changed(group_by=group_by):
            self.wrapper.grouper.check_group_by(group_by=group_by)
            return self.copy(orders=self._orders.regroup(group_by=group_by))
        return self

    # ############# Reference price ############# #

    @property
    def close(self):
        """Price per share series."""
        return self._ref_price

    @cached_method
    def fill_close(self, ffill=True, bfill=True):
        """Fill NaN values of `Portfolio.close`.

        Use `ffill` and `bfill` to fill forwards and backwards respectively."""
        close = to_2d(self.close, raw=True)
        if ffill and np.any(np.isnan(close[-1, :])):
            close = generic_nb.ffill_nb(close)
        if bfill and np.any(np.isnan(close[0, :])):
            close = generic_nb.ffill_nb(close[::-1, :])[::-1, :]
        return self.wrapper.wrap(close, group_by=False)

    # ############# Records ############# #

    def orders(self, group_by=None):  # doesn't require caching
        """Get order records.

        See `vectorbt.portfolio.orders.Orders`."""
        return self._orders.regroup(group_by=group_by)

    def logs(self, group_by=None):  # doesn't require caching
        """Get log records.

        See `vectorbt.portfolio.logs.Logs`."""
        return self._logs.regroup(group_by=group_by)

    def trades(self, group_by=None):  # doesn't require caching
        """Get trade records.

        See `vectorbt.portfolio.events.Trades`."""
        return self._trades.regroup(group_by=group_by)

    def positions(self, group_by=None):  # doesn't require caching
        """Get position records.

        See `vectorbt.portfolio.events.Positions`."""
        return self._positions.regroup(group_by=group_by)

    @cached_method
    def drawdowns(self, **kwargs):
        """Get drawdown records from `Portfolio.value`.

        See `vectorbt.generic.drawdowns.Drawdowns`."""
        return Drawdowns.from_ts(self.value(**kwargs), freq=self.wrapper.freq)

    # ############# Shares ############# #

    @cached_method
    def share_flow(self):
        """Get share flow series per column."""
        share_flow = nb.share_flow_nb(self.wrapper.shape_2d, self._orders.records_arr)
        return self.wrapper.wrap(share_flow, group_by=False)

    @cached_method
    def shares(self):
        """Get share series per column."""
        share_flow = to_2d(self.share_flow(), raw=True)
        shares = nb.shares_nb(share_flow)
        return self.wrapper.wrap(shares, group_by=False)

    @cached_method
    def pos_mask(self, group_by=None):
        """Get position mask per column/group."""
        shares = to_2d(self.shares(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            pos_mask = nb.pos_mask_nb(shares, group_lens)
        else:
            pos_mask = shares != 0
        return self.wrapper.wrap(pos_mask, group_by=group_by)

    @cached_method
    def long_pos_mask(self, group_by=None):
        """Get long position mask per column/group."""
        shares = to_2d(self.shares(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            long_pos_mask = nb.long_pos_mask_nb(shares, group_lens)
        else:
            long_pos_mask = shares > 0
        return self.wrapper.wrap(long_pos_mask, group_by=group_by)

    @cached_method
    def short_pos_mask(self, group_by=None):
        """Get short position mask per column/group."""
        shares = to_2d(self.shares(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            short_pos_mask = nb.short_pos_mask_nb(shares, group_lens)
        else:
            short_pos_mask = shares < 0
        return self.wrapper.wrap(short_pos_mask, group_by=group_by)

    @cached_method
    def pos_coverage(self, group_by=None):
        """Get position coverage per column/group."""
        shares = to_2d(self.shares(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            pos_coverage = nb.pos_duration_nb(shares, group_lens)
        else:
            pos_coverage = np.mean(shares != 0, axis=0)
        return self.wrapper.wrap_reduced(pos_coverage, group_by=group_by)

    @cached_method
    def long_pos_coverage(self, group_by=None):
        """Get long position coverage per column/group."""
        shares = to_2d(self.shares(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            long_pos_coverage = nb.long_pos_duration_nb(shares, group_lens)
        else:
            long_pos_coverage = np.mean(shares > 0, axis=0)
        return self.wrapper.wrap_reduced(long_pos_coverage, group_by=group_by)

    @cached_method
    def short_pos_coverage(self, group_by=None):
        """Get short position coverage per column/group."""
        shares = to_2d(self.shares(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            short_pos_coverage = nb.short_pos_duration_nb(shares, group_lens)
        else:
            short_pos_coverage = np.mean(shares < 0, axis=0)
        return self.wrapper.wrap_reduced(short_pos_coverage, group_by=group_by)

    # ############# Cash ############# #

    @cached_method
    def cash_flow(self, group_by=None):
        """Get cash flow series per column/group."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            cash_flow_ungrouped = to_2d(self.cash_flow(group_by=False), raw=True)
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            cash_flow = nb.cash_flow_grouped_nb(cash_flow_ungrouped, group_lens)
        else:
            cash_flow = nb.cash_flow_ungrouped_nb(self.wrapper.shape_2d, self._orders.records_arr)
        return self.wrapper.wrap(cash_flow, group_by=group_by)

    @cached_method
    def init_cash(self, group_by=None):
        """Get initial amount of cash per column/group.

        !!! note
            If initial cash is found automatically and no own cash is used throughout simulation
            (for example, when shorting), initial cash will be set to 1 instead of 0 to
            enable smooth calculation of returns."""
        if isinstance(self._init_cash, int):
            cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
            cash_min = np.min(np.cumsum(cash_flow, axis=0), axis=0)
            init_cash = np.where(cash_min < 0, np.abs(cash_min), 1.)
            if self._init_cash == InitCashMode.AutoAlign:
                init_cash = np.full(init_cash.shape, np.max(init_cash))
        else:
            init_cash = to_1d(self._init_cash, raw=True)
            if self.wrapper.grouper.is_grouped(group_by=group_by):
                group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
                init_cash = nb.init_cash_grouped_nb(init_cash, group_lens, self.cash_sharing)
            else:
                group_lens = self.wrapper.grouper.get_group_lens()
                init_cash = nb.init_cash_ungrouped_nb(init_cash, group_lens, self.cash_sharing)
        return self.wrapper.wrap_reduced(init_cash, group_by=group_by)

    @cached_method
    def cash(self, group_by=None, in_sim_order=False):
        """Get cash balance series per column/group."""
        if in_sim_order and not self.cash_sharing:
            raise ValueError("Cash sharing must be enabled for in_sim_order=True")

        cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            init_cash = to_1d(self.init_cash(group_by=group_by), raw=True)
            cash = nb.cash_grouped_nb(
                self.wrapper.shape_2d,
                cash_flow,
                group_lens,
                init_cash
            )
        else:
            group_lens = self.wrapper.grouper.get_group_lens()
            init_cash = to_1d(self.init_cash(group_by=in_sim_order), raw=True)
            call_seq = to_2d(self.call_seq, raw=True)
            cash = nb.cash_ungrouped_nb(
                cash_flow,
                group_lens,
                init_cash,
                call_seq,
                in_sim_order
            )
        return self.wrapper.wrap(cash, group_by=group_by)

    # ############# Performance ############# #

    @cached_method
    def holding_value(self, group_by=None):
        """Get holding value series per column/group."""
        close = to_2d(self.close, raw=True).copy()
        shares = to_2d(self.shares(), raw=True)
        close[shares == 0] = 0.  # for price being NaN
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            holding_value = nb.holding_value_grouped_nb(close, shares, group_lens)
        else:
            holding_value = nb.holding_value_ungrouped_nb(close, shares)
        return self.wrapper.wrap(holding_value, group_by=group_by)

    @cached_method
    def value(self, group_by=None, in_sim_order=False):
        """Get portfolio value series per column/group.

        By default, will generate portfolio value for each asset based on cash flows and thus
        independent from other assets, with initial cash and shares being that of the entire group.
        Useful for generating returns and comparing assets within the same group.

        When `group_by` is False and `in_sim_order` is True, returns value generated in
        simulation order (see [row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
        This value cannot be used for generating returns as-is. Useful to analyze how value
        evolved throughout simulation."""
        cash = to_2d(self.cash(group_by=group_by, in_sim_order=in_sim_order), raw=True)
        holding_value = to_2d(self.holding_value(group_by=group_by), raw=True)
        if self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and in_sim_order:
            group_lens = self.wrapper.grouper.get_group_lens()
            call_seq = to_2d(self.call_seq, raw=True)
            value = nb.value_in_sim_order_nb(cash, holding_value, group_lens, call_seq)
            # price of NaN is already addressed by ungrouped_value_nb
        else:
            value = nb.value_nb(cash, holding_value)
        return self.wrapper.wrap(value, group_by=group_by)

    @cached_method
    def total_profit(self, group_by=None):
        """Get total profit per column/group.

        Calculated directly from order records. Very fast."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            total_profit_ungrouped = to_1d(self.total_profit(group_by=False), raw=True)
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            total_profit = nb.total_profit_grouped_nb(
                total_profit_ungrouped,
                group_lens
            )
        else:
            close = to_2d(self.fill_close(), raw=True)
            init_cash_ungrouped = to_1d(self.init_cash(group_by=False), raw=True)
            total_profit = nb.total_profit_ungrouped_nb(
                self.wrapper.shape_2d,
                close,
                self._orders.records_arr,
                init_cash_ungrouped
            )
        return self.wrapper.wrap_reduced(total_profit, group_by=group_by)

    @cached_method
    def final_value(self, group_by=None):
        """Get total profit per column/group."""
        init_cash = to_1d(self.init_cash(group_by=group_by), raw=True)
        total_profit = to_1d(self.total_profit(group_by=group_by), raw=True)
        final_value = nb.final_value_nb(total_profit, init_cash)
        return self.wrapper.wrap_reduced(final_value, group_by=group_by)

    @cached_method
    def total_return(self, group_by=None):
        """Get total profit per column/group."""
        init_cash = to_1d(self.init_cash(group_by=group_by), raw=True)
        total_profit = to_1d(self.total_profit(group_by=group_by), raw=True)
        total_return = nb.total_return_nb(total_profit, init_cash)
        return self.wrapper.wrap_reduced(total_return, group_by=group_by)

    @cached_method
    def returns(self, group_by=None, in_sim_order=False):
        """Get return series per column/group based on portfolio value."""
        value = to_2d(self.value(group_by=group_by, in_sim_order=in_sim_order), raw=True)
        if self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and in_sim_order:
            group_lens = self.wrapper.grouper.get_group_lens()
            init_cash_grouped = to_1d(self.init_cash(), raw=True)
            call_seq = to_2d(self.call_seq, raw=True)
            returns = nb.returns_in_sim_order_nb(value, group_lens, init_cash_grouped, call_seq)
        else:
            init_cash = to_1d(self.init_cash(group_by=group_by), raw=True)
            returns = nb.returns_nb(value, init_cash)
        return self.wrapper.wrap(returns, group_by=group_by)

    @cached_method
    def active_returns(self, group_by=None):
        """Get active return series per column/group.

        This type of returns is based solely on cash flows and holding value rather than portfolio value.
        It ignores passive cash and thus it will return the same numbers irrespective of the amount of
        cash currently available, even `np.inf`. The scale of returns is comparable to that of going
        all in and keeping available cash at zero."""
        cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
        holding_value = to_2d(self.holding_value(group_by=group_by), raw=True)
        active_returns = nb.active_returns_nb(cash_flow, holding_value)
        return self.wrapper.wrap(active_returns, group_by=group_by)

    @cached_method
    def buy_and_hold_return(self, group_by=None):
        """Get total return of buy-and-hold.

        If grouped, invests same amount of cash into each asset and returns the total
        return of the entire group.

        !!! note
            Does not take into account fees and slippage. For this, create a separate portfolio."""
        ref_price_filled = to_2d(self.fill_close(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_lens = self.wrapper.grouper.get_group_lens(group_by=group_by)
            total_return = nb.buy_and_hold_return_grouped_nb(ref_price_filled, group_lens)
        else:
            total_return = nb.buy_and_hold_return_ungrouped_nb(ref_price_filled)
        return self.wrapper.wrap_reduced(total_return, group_by=group_by)

    @cached_method
    def stats(self, column=None, group_by=None, incl_unrealized=None, active_returns=False,
              in_sim_order=False, agg_func=lambda x: x.mean(axis=0), **kwargs):
        """Compute various statistics on this portfolio.

        `kwargs` will be passed to each `vectorbt.returns.accessors.Returns_Accessor` method.

        Can either return aggregated statistics by reducing metrics of all columns with
        `agg_func` (mean by default) or return statistics for a single column if `column`
        was specified or portfolio contains only one column of data. To display rich data types
        such as durations correctly, use an aggregation function that can be applied on `pd.Series`.

        !!! note
            Use `column` only if caching is enabled, otherwise it may re-compute the same
            objects multiple times."""
        # Pre-calculate
        trades = self.trades(group_by=group_by)
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized
        if not incl_unrealized:
            trades = trades.closed
        drawdowns = self.drawdowns(group_by=group_by)
        if active_returns:
            returns = self.active_returns(group_by=group_by)
        else:
            returns = self.returns(group_by=group_by, in_sim_order=in_sim_order)

        # Run stats
        stats_df = pd.DataFrame({
            'Start': self.wrapper.index[0],
            'End': self.wrapper.index[-1],
            'Duration': self.wrapper.shape[0] * self.wrapper.freq,
            'Total Profit': self.total_profit(group_by=group_by),
            'Total Return [%]': self.total_return(group_by=group_by) * 100,
            'Buy & Hold Return [%]': self.buy_and_hold_return(group_by=group_by) * 100,
            'Position Coverage [%]': self.pos_coverage(group_by=group_by) * 100,
            'Max. Drawdown [%]': -drawdowns.max_drawdown() * 100,
            'Avg. Drawdown [%]': -drawdowns.avg_drawdown() * 100,
            'Max. Drawdown Duration': drawdowns.max_duration(),
            'Avg. Drawdown Duration': drawdowns.avg_duration(),
            'Num. Trades': trades.count(),
            'Win Rate [%]': trades.win_rate() * 100,
            'Best Trade [%]': trades.returns.max() * 100,
            'Worst Trade [%]': trades.returns.min() * 100,
            'Avg. Trade [%]': trades.returns.mean() * 100,
            'Max. Trade Duration': trades.duration.max(time_units=True),
            'Avg. Trade Duration': trades.duration.mean(time_units=True),
            'Expectancy': trades.expectancy(),
            'SQN': trades.sqn(),
            'Sharpe Ratio': self.sharpe_ratio(reuse_returns=returns, **kwargs),
            'Sortino Ratio': self.sortino_ratio(reuse_returns=returns, **kwargs),
            'Calmar Ratio': self.calmar_ratio(reuse_returns=returns, **kwargs)
        }, index=self.wrapper.grouper.get_columns(group_by=group_by))

        # Select columns or reduce
        if stats_df.shape[0] == 1:
            return self.wrapper.wrap_reduced(stats_df.iloc[0], index=stats_df.columns)
        if column is not None:
            return stats_df.loc[column]
        if agg_func is not None:
            agg_stats_sr = pd.Series(index=stats_df.columns, name=agg_func.__name__)
            agg_stats_sr.iloc[:3] = stats_df.iloc[0, :3]
            agg_stats_sr.iloc[3:] = agg_func(stats_df.iloc[:, 3:])
            return agg_stats_sr
        return stats_df

    def plot(self,
             column=None,
             plot_orders=True,
             order_kwargs=None,
             plot_trades=False,
             trades_kwargs=None,
             trade_type='trades',
             plot_trade_pnl=True,
             trade_pnl_kwargs=None,
             plot_value=True,
             value_kwargs=None,
             plot_holding_value=True,
             holding_value_kwargs=None,
             plot_cash=True,
             cash_kwargs=None,
             plot_shares=True,
             shares_kwargs=None,
             plot_value_drawdowns=True,
             value_drawdowns_kwargs=None,
             plot_cum_returns=True,
             cum_returns_kwargs=None,
             zeroline_shape_kwargs=None,
             plot_benchmark_value=True,
             benchmark_value_kwargs=None,
             plot_benchmark_cumret=True,
             benchmark_cumret_kwargs=None,
             benchmark='max_shares_long',
             active_returns=False,
             in_sim_order=False,
             incl_unrealized=None,
             fig=None,
             vertical_spacing=0.02,
             **layout_kwargs):  # pragma: no cover
        """Plot trade PnL.

        Args:
            column (str): Name of the column to plot.
            plot_orders (bool): Whether to plot orders.
            order_kwargs (dict): Keyword arguments passed to `vectorbt.portfolio.orders.Orders.plot`.
            plot_trades (bool): Whether to plot trades.
            trades_kwargs (dict): Keyword arguments passed to `vectorbt.portfolio.trades.Trades.plot`.
            trade_type (str): Pass `'trades'` for trades and `'positions'` for positions.
            plot_trade_pnl (bool): Whether to plot PnL of trades/positions.
            trade_pnl_kwargs (dict): Keyword arguments passed to `vectorbt.portfolio.trades.Trades.plot_pnl`.
            plot_value (bool): Whether to plot portfolio value.
            value_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot`.
            plot_holding_value (bool): Whether to plot holding value.
            holding_value_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot`.
            plot_cash (bool): Whether to plot cash balance.
            cash_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot`.
            plot_shares (bool): Whether to plot share balance.
            shares_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot`.
            plot_value_drawdowns (bool): Whether to plot drawdowns of portfolio value.
            value_drawdowns_kwargs (dict): Keyword arguments passed to `vectorbt.generic.drawdowns.Drawdowns.plot`.
            plot_cum_returns (bool): Whether to plot cumulative returns of portfolio value.
            cum_returns_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot`.
            zeroline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            plot_benchmark_value (bool): Whether to plot benchmark value.
            benchmark_value_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot`.
            plot_benchmark_cumret (bool): Whether to plot cumulative returns of benchmark value.
            benchmark_cumret_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot`.
            benchmark (Portfolio, str or array_like): Benchmark to compare portfolio value against.

                The following values are allowed:

                * Instance of another `Portfolio`
                * `'inf_long'`/`'inf_short'` to simulate longing/shorting shares for all
                    initial capital of this portfolio
                * `'max_shares_long'`/`'max_shares_short'` to simulate longing/shorting
                    the maximum number of shares of this portfolio
                * Array-like value series of the same shape
            active_returns (bool): Whether to use active returns.
            in_sim_order (bool): Whether to use portfolio value generated in simulation order.
            incl_unrealized (bool): Whether to include unrealized P&L in statistics.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            vertical_spacing (float): Space between subplot rows in normalized plot coordinates.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> portfolio.plot()
            ```

            ![](/vectorbt/docs/img/portfolio.png)"""
        from vectorbt.defaults import color_schema

        if order_kwargs is None:
            order_kwargs = {}
        if trades_kwargs is None:
            trades_kwargs = {}
        if trade_pnl_kwargs is None:
            trade_pnl_kwargs = {}
        if value_kwargs is None:
            value_kwargs = {}
        if holding_value_kwargs is None:
            holding_value_kwargs = {}
        if cash_kwargs is None:
            cash_kwargs = {}
        if shares_kwargs is None:
            shares_kwargs = {}
        if value_drawdowns_kwargs is None:
            value_drawdowns_kwargs = {}
        if cum_returns_kwargs is None:
            cum_returns_kwargs = {}
        if benchmark_value_kwargs is None:
            benchmark_value_kwargs = {}
        if benchmark_cumret_kwargs is None:
            benchmark_cumret_kwargs = {}
        if zeroline_shape_kwargs is None:
            zeroline_shape_kwargs = {}
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized

        def _select_column(portfolio, column):
            """Select one column."""
            if column is not None:
                if self.wrapper.grouper.group_by is None:
                    pcol = portfolio[column]
                else:
                    pcol = portfolio.regroup(None)[column]
            else:
                pcol = portfolio
            if pcol.wrapper.ndim > 1:
                raise TypeError("Select a column first. Use indexing or column argument.")
            return pcol

        self_col = _select_column(self, column)

        if fig is None:
            # Create new figure
            fig = CustomFigureWidget()

        if plot_trades or plot_trade_pnl:
            # Extract trades/positions
            checks.assert_in(trade_type, ['trades', 'positions'])
            if trade_type == 'trades':
                trades = self_col.trades()
            else:
                trades = self_col.positions()

            if not incl_unrealized:
                # Use closed trades only
                trades = trades.closed
        else:
            trades = None

        if plot_benchmark_value or plot_benchmark_cumret:
            # Benchmark to compare value/cumulative returns against
            if isinstance(benchmark, str):
                # Simulate another portfolio
                checks.assert_in(benchmark, ['inf_long', 'inf_short', 'max_shares_long', 'max_shares_short'])
                order_size = self_col.wrapper.wrap(np.full(self_col.wrapper.shape[0], 0.))
                if benchmark == 'inf_long':
                    order_size[0] = np.inf
                elif benchmark == 'inf_short':
                    order_size[0] = -np.inf
                elif benchmark == 'max_shares_long':
                    order_size[0] = self_col.shares().abs().max()
                elif benchmark == 'max_shares_short':
                    order_size[0] = -self_col.shares().abs().max()
                benchmark = Portfolio.from_orders(self_col.close, order_size, init_cash=self_col.init_cash())
            if isinstance(benchmark, Portfolio):
                # Query value from provided portfolio
                benchmark = _select_column(benchmark, column)
                benchmark_value = benchmark.value(in_sim_order=in_sim_order)
            else:
                # Provided benchmark is an array-like object
                benchmark_value = self_col.wrapper.wrap(np.asarray(benchmark))
        else:
            benchmark_value = None

        # Count Y axes
        y1_active = plot_orders
        y2_active = plot_trades
        y3_active = plot_trade_pnl
        y4_active = plot_value or plot_holding_value or plot_cash
        y5_active = plot_shares  # not a subplot
        y6_active = plot_value_drawdowns
        y7_active = plot_cum_returns
        num_y_axes = sum([y1_active, y2_active, y3_active, y4_active, y5_active, y6_active, y7_active])
        if num_y_axes == 0:
            raise ValueError("Nothing to plot")

        # Y axes come in descending order
        y_keys = []
        yaxis_keys = []
        for i in range(num_y_axes - 1, -1, -1):
            y_keys.append('y' + str(i + 1) if i > 0 else 'y')
            yaxis_keys.append('yaxis' + str(i + 1) if i > 0 else 'yaxis')

        # Calculate domains
        domains = []
        num_subplots = num_y_axes - 1 if y5_active else num_y_axes
        if num_subplots > 0:
            for i in range(num_subplots - 1, -1, -1):
                from_y = i / num_subplots
                if i > 0:
                    from_y += vertical_spacing / 2
                to_y = (i + 1) / num_subplots
                if i < num_subplots:
                    to_y -= vertical_spacing / 2
                domains.append([from_y, to_y])

        # Plot traces
        j = 0
        yaxis_layout = {}
        if y1_active:
            # Plot orders
            order_kwargs = merge_kwargs(dict(
                close_trace_kwargs=dict(
                    yaxis=y_keys[j],
                    line_color=color_schema['blue'],
                    name='Close' if self_col.close.name is None else self_col.close.name
                ),
                buy_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                sell_trace_kwargs=dict(
                    yaxis=y_keys[j]
                )
            ), order_kwargs)
            self_col.orders().plot(**order_kwargs, fig=fig)
            yaxis_layout[yaxis_keys[j]] = dict(
                title='Price',
                domain=domains[j]
            )
            j += 1
        if y2_active:
            # Plot trades
            trades_kwargs = merge_kwargs(dict(
                close_trace_kwargs=dict(
                    yaxis=y_keys[j],
                    line_color=color_schema['blue'],
                    name='Close' if self_col.close.name is None else self_col.close.name
                ),
                entry_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                exit_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                exit_profit_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                exit_loss_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                active_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                profit_shape_kwargs=dict(
                    yref=y_keys[j]
                ),
                loss_shape_kwargs=dict(
                    yref=y_keys[j]
                )
            ), trades_kwargs)
            trades.plot(**trades_kwargs, fig=fig)
            yaxis_layout[yaxis_keys[j]] = dict(
                title='Price',
                domain=domains[j]
            )
            j += 1
        if y3_active:
            # Plot trade PnL
            trade_pnl_kwargs = merge_kwargs(dict(
                profit_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                loss_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                zeroline_shape_kwargs=dict(
                    yref=y_keys[j],
                    **zeroline_shape_kwargs
                )
            ), trade_pnl_kwargs)
            trades.plot_pnl(**trade_pnl_kwargs, fig=fig)
            yaxis_layout[yaxis_keys[j]] = dict(
                title='PnL',
                domain=domains[j]
            )
            j += 1
        if y4_active:
            if plot_value:
                # Plot portfolio and benchmark value
                value_kwargs = merge_kwargs(dict(
                    name='Value',
                    trace_kwargs=dict(
                        yaxis=y_keys[j],
                        line_color=color_schema['purple']
                    )
                ), value_kwargs)
                self_col.value(in_sim_order=in_sim_order).vbt.plot(**value_kwargs, fig=fig)
                if plot_benchmark_value:
                    benchmark_value_kwargs = merge_kwargs(dict(
                        name='Benchmark Value',
                        trace_kwargs=dict(
                            yaxis=y_keys[j],
                            line_color=color_schema['purple'],
                            line_dash='dot'
                        )
                    ), benchmark_value_kwargs)
                    benchmark_value.vbt.plot(**benchmark_value_kwargs, fig=fig)
            if plot_holding_value:
                # Plot holding value
                holding_value_kwargs = merge_kwargs(dict(
                    name='Holding Value',
                    trace_kwargs=dict(
                        yaxis=y_keys[j],
                        line_color=color_schema['cyan']
                    )
                ), holding_value_kwargs)
                self_col.holding_value().vbt.plot(**holding_value_kwargs, fig=fig)
            if plot_cash:
                # Plot cash
                cash_kwargs = merge_kwargs(dict(
                    name='Cash',
                    trace_kwargs=dict(
                        yaxis=y_keys[j],
                        line_color=color_schema['green']
                    )
                ), cash_kwargs)
                self_col.cash(in_sim_order=in_sim_order).vbt.plot(**cash_kwargs, fig=fig)
            yaxis_layout[yaxis_keys[j]] = dict(
                title='Value',
                domain=domains[j]
            )
            j += 1
        if y5_active:
            # Plot shares
            shares_kwargs = merge_kwargs(dict(
                name='Shares',
                trace_kwargs=dict(
                    yaxis=y_keys[j],
                    line_color=color_schema['brown']
                )
            ), shares_kwargs)
            self_col.shares().vbt.plot(**shares_kwargs, fig=fig)
            if y4_active:
                yaxis_layout[yaxis_keys[j]] = dict(
                    showgrid=False,
                    overlaying=y_keys[j - 1],
                    side="right",
                    title='Shares'
                )
            else:
                yaxis_layout[yaxis_keys[j]] = dict(
                    title='Shares'
                )
            j += 1
        if y6_active:
            # Plot drawdowns of portfolio value
            value_drawdowns_kwargs = merge_kwargs(dict(
                ts_trace_kwargs=dict(
                    yaxis=y_keys[j],
                    line_color=color_schema['purple'],
                    name='Close' if self_col.close.name is None else self_col.close.name
                ),
                peak_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                valley_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                recovery_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                active_trace_kwargs=dict(
                    yaxis=y_keys[j]
                ),
                ptv_shape_kwargs=dict(
                    y0=domains[j - y5_active][0],
                    y1=domains[j - y5_active][1]
                ),
                vtr_shape_kwargs=dict(
                    y0=domains[j - y5_active][0],
                    y1=domains[j - y5_active][1]
                ),
                active_shape_kwargs=dict(
                    y0=domains[j - y5_active][0],
                    y1=domains[j - y5_active][1]
                ),
            ), value_drawdowns_kwargs)
            self_col.drawdowns().plot(**value_drawdowns_kwargs, fig=fig)
            yaxis_layout[yaxis_keys[j]] = dict(
                title='Value',
                domain=domains[j - y5_active]
            )
            j += 1
        if y7_active:
            # Plot cumulative returns of portfolio and benchmark value
            cum_returns_kwargs = merge_kwargs(dict(
                name='Cum. Returns',
                trace_kwargs=dict(
                    yaxis=y_keys[j],
                    line_color=color_schema['orange']
                )
            ), cum_returns_kwargs)
            if active_returns:
                returns = self_col.active_returns()
            else:
                returns = self_col.returns(in_sim_order=in_sim_order)
            cum_returns = returns.vbt.returns.cumulative(start_value=1.)
            cum_returns.vbt.plot(**cum_returns_kwargs, fig=fig)
            if plot_benchmark_cumret:
                benchmark_cumret_kwargs = merge_kwargs(dict(
                    name='Benchmark Cum. Returns',
                    trace_kwargs=dict(
                        yaxis=y_keys[j],
                        line_color=color_schema['orange'],
                        line_dash='dot'
                    )
                ), benchmark_cumret_kwargs)
                benchmark_cumret = benchmark_value.pct_change().fillna(0.).vbt.returns.cumulative(start_value=1.)
                benchmark_cumret.vbt.plot(**benchmark_cumret_kwargs, fig=fig)
            # Plot zeroline
            fig.add_shape(**merge_kwargs(dict(
                xref="paper",
                yref=y_keys[j],
                x0=0,
                y0=1,
                x1=1,
                y1=1,
                line=dict(
                    color="gray",
                    dash="dashdot",
                )
            ), zeroline_shape_kwargs))
            yaxis_layout[yaxis_keys[j]] = dict(
                title='Cum. Returns',
                domain=domains[j - y5_active]
            )
            j += 1

        # Set up layout
        fig.update_layout(dict(
            autosize=True,
            width=900,
            height=250 * num_subplots,
            margin=dict(b=40, t=20),
            legend=dict(
                font=dict(size=10),
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            hovermode="closest",
            xaxis=dict(
                title='Date',
            )
        ))
        fig.update_layout(yaxis_layout)
        fig.update_layout(layout_kwargs)

        # Remove duplicate legend labels
        found_names = set()
        for trace in fig.data:
            if 'name' in trace:
                if trace['name'] in found_names:
                    trace['showlegend'] = False
                else:
                    found_names |= {trace['name']}

        return fig
