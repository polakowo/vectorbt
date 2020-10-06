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

It basically builds upon the `vectorbt.records.orders.Orders` class. To simplify creation of order
records and keep track of balances, it exposes several convenience methods with prefix `from_`.
For example, you can use `Portfolio.from_signals` method to generate orders from entry and exit signals.
Alternatively, you can use `Portfolio.from_order_func` to run a custom order function on each tick.
The results are then automatically passed to the constructor method of `Portfolio` and you will
receive a portfolio instance ready to be used for performance analysis.

This way, one can simulate and analyze his/her strategy in a couple of lines.

### Example

The following example does something crazy: it checks candlestick data of 6 major cryptocurrencies
in 2020 against every single pattern found in TA-Lib, and translates them into orders:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> from datetime import datetime
>>> import yfinance as yf
>>> import talib
>>> import vectorbt as vbt
>>> from vectorbt.portfolio.enums import InitCashMode

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
>>> order_size = result / price['Open']

>>> # Simulate portfolio
>>> portfolio = vbt.Portfolio.from_orders(
...     price['Close'], order_size, order_price=price['Open'],
...     init_cash=InitCashMode.AutoAlign, fees=0.001, slippage=0.001
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
* Implements flexible broadcasting wherever possible to save memory

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
...     price['Close'], order_size, order_price=price['Open'],
...     init_cash=InitCashMode.AutoAlign, fees=0.001, slippage=0.001,
...     group_by=group_by, cash_sharing=True
... )

>>> # Get total profit per group
>>> comb_portfolio.total_profit()
group
first     21793.882832
second     8333.660493
dtype: float64
```

Not only can you analyze each group, but also each column in the group:

```python-repl
>>> # Get total profit per column
>>> comb_portfolio.total_profit(group_by=False)
BTC-USD     5166.373585
ETH-USD    13098.913381
XRP-USD     3528.595867
BNB-USD     5345.521391
BCH-USD     -235.128582
LTC-USD     3223.267684
dtype: float64
```

In the same way, you can introduce new grouping to the method itself:

```python-repl
>>> # Get total profit per group
>>> portfolio.total_profit(group_by=group_by)
group
first     21793.882832
second     8333.660493
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
5166.373584618163
```

Combined portfolio is indexed by group:

```python-repl
>>> comb_portfolio['first']
<vectorbt.portfolio.base.Portfolio at 0x7fac5756b828>

>>> comb_portfolio['first'].total_profit()
21793.882832230272
```

!!! note
    Changing index (time axis) is not supported. The object should be treated as a Series
    rather than a DataFrame; for example, use `portfolio.iloc[0]` instead of `portfolio.iloc[:, 0]`.

    Indexing behavior depends solely upon `vectorbt.base.array_wrapper.ArrayWrapper`.
    For example, if `group_select` is enabled indexing will be performed on groups,
    otherwise on single columns. You can pass wrapper arguments with `wrapper_kwargs`.

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

from vectorbt import defaults
from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_method
from vectorbt.utils.config import Configured, merge_kwargs
from vectorbt.utils.random import set_seed
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic import nb as generic_nb
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import (
    SizeType,
    AccumulateExitMode,
    ConflictMode,
    CallSeqType,
    InitCashMode
)
from vectorbt.records import Orders, Trades, Positions, Drawdowns
from vectorbt.records.orders import indexing_on_orders_meta


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Portfolio`."""
    new_orders, group_idxs, col_idxs = indexing_on_orders_meta(obj._orders, pd_indexing_func)
    if isinstance(obj._init_cash, int):
        new_init_cash = obj._init_cash
    else:
        new_init_cash = to_1d(obj._init_cash, raw=True)[group_idxs if obj.cash_sharing else col_idxs]
    new_call_seq = obj.call_seq.values[:, col_idxs]

    return obj.copy(
        orders=new_orders,
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
        orders (Orders): Order records.
        init_cash (InitCashMode, float or array_like of float): Initial capital.
        cash_sharing (bool): Whether to share cash within the same group.
        call_seq (array_like of int): Sequence of calls per row and group.
        incl_unrealized (bool): Whether to include unrealized P&L in statistics.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

    !!! note
        This class is meant to be immutable. To change any attribute, use `Portfolio.copy`."""

    def __init__(self, orders, init_cash, cash_sharing, call_seq, incl_unrealized=None):
        Configured.__init__(
            self,
            orders=orders,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            incl_unrealized=incl_unrealized
        )
        # Get defaults
        if incl_unrealized is None:
            incl_unrealized = defaults.portfolio['incl_unrealized']

        # Perform checks
        checks.assert_type(orders, Orders)

        # Store passed arguments
        self._ref_price = orders.close
        self._orders = orders
        self._init_cash = init_cash
        self._cash_sharing = cash_sharing
        self._call_seq = call_seq
        self._incl_unrealized = incl_unrealized

        # Supercharge
        PandasIndexer.__init__(self, _indexing_func)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, close, entries, exits, size=None, entry_price=None, exit_price=None,
                     fees=None, fixed_fees=None, slippage=None, reject_prob=None, min_size=None,
                     init_cash=None, cash_sharing=None, call_seq=None, accumulate=None,
                     accumulate_exit_mode=None, conflict_mode=None, seed=None, freq=None, group_by=None,
                     broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Simulate portfolio from entry and exit signals.

        Starting with initial cash `init_cash`, for each signal in `entries`, enters a position
        by buying `size` of shares for `entry_price`. For each signal in `exits`, closes the position
        by selling all shares for `exit_price`. When accumulation is enabled, each entry signal will
        increase the position, and optionally each exit signal will decrease the position. When both
        entry and exit signals are present, ignores them by default. When grouping is enabled with
        `group_by`, will compute performance for the entire group. When, additionally, `cash_sharing`
        is enabled, will share the cash among all columns in the group.

        Args:
            close (pandas_like): Reference price, such as close. Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            entries (array_like of bool): Boolean array of entry signals. Will broadcast.
            exits (array_like of bool): Boolean array of exit signals. Will broadcast.
            size (float or array_like): Size to order. Will broadcast.

                * Set to positive/negative to buy/sell.
                * Set to `np.inf`/`-np.inf` to buy/sell everything.
                * Set to `np.nan` or zero to skip.
            entry_price (array_like of float): Entry price. Defaults to `close`. Will broadcast.

                !!! note
                    Setting order price to close is risky.
            exit_price (array_like of float): Exit price. Defaults to `close`. Will broadcast.

                !!! note
                    Setting order price to close is risky.
            fees (float or array_like): Fees in percentage of the order value. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price. Will broadcast.
            reject_prob (float or array_like): Order rejection probability. Will broadcast.
            min_size (float or array_like): Minimum size for an order to be accepted.

                Will broadcast to the number of columns.
            init_cash (InitCashMode, float or array_like of float): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
                See `vectorbt.portfolio.enums.InitCashMode` to find optimal initial cash.

                !!! note
                    Mode `InitCashMode.AutoAlign` is applied after the portfolio is initialized
                    to set the same initial cash for all columns/groups. Changing grouping
                    will change the initial cash, so be aware when indexing.

                    Make sure that `init_cash` is a floating number if not using `InitCashMode`.


            cash_sharing (bool): Whether to share cash within the same group.

                !!! warning
                    Order execution cannot be considered parallel anymore.

                    This method presumes that in a group of assets that share the same capital all
                    orders will be executed within the same tick and retain their price regardless
                    of their position in the queue, even though they depend upon each other and thus
                    cannot be executed in parallel. This behavior is risky.
            call_seq (CallSeqType or array_like of int): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.
            accumulate (bool): If `accumulate` is True, entering the market when already
                in the market will be allowed to increase the position.
            accumulate_exit_mode (AccumulateExitMode): See `vectorbt.portfolio.enums.AccumulateExitMode`.
            conflict_mode (ConflictMode): See `vectorbt.portfolio.enums.ConflictMode`.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            freq (any): Index frequency in case `close.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.
            **kwargs: Keyword arguments passed to the `__init__` method.

        All time series will be broadcast together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            Only `SizeType.Shares` is supported. Other modes such as target percentage are not
            compatible with signals since their logic may contradict the direction the user has
            specified for the order.

            With cash sharing enabled, at each timestamp, processing of the assets in a group
            goes strictly in order defined in `call_seq`. This order can't be changed dynamically.

        !!! hint
            If you generated signals using close price, don't forget to shift your signals by one tick
            forward, for example, with `signals.vbt.fshift(1)`. In general, make sure to use a price
            that comes after the signal.

        Example:
            Different ways of how signals are interpreted:

            ```python-repl
            >>> import numpy as np
            >>> import pandas as pd
            >>> from datetime import datetime
            >>> import vectorbt as vbt
            >>> from vectorbt.portfolio.enums import AccumulateExitMode, ConflictMode

            >>> price = pd.Series([1., 2., 3., 4., 5.], index=pd.Index([
            ...     datetime(2020, 1, 1),
            ...     datetime(2020, 1, 2),
            ...     datetime(2020, 1, 3),
            ...     datetime(2020, 1, 4),
            ...     datetime(2020, 1, 5)
            ... ]))
            >>> entries = pd.Series([True, True, True, False, False])
            >>> exits = pd.Series([False, False, True, True, True])

            >>> portfolio = vbt.Portfolio.from_signals(
            ...     price, entries, exits, size=1.)
            >>> portfolio.share_flow()
            2020-01-01    1.0
            2020-01-02    0.0
            2020-01-03    0.0
            2020-01-04   -1.0
            2020-01-05    0.0
            dtype: float64

            >>> portfolio = vbt.Portfolio.from_signals(
            ...     price, entries, exits, size=1.,
            ...     conflict_mode=ConflictMode.Exit)
            >>> portfolio.share_flow()
            2020-01-01    1.0
            2020-01-02    0.0
            2020-01-03   -1.0
            2020-01-04    0.0
            2020-01-05    0.0
            dtype: float64

            >>> portfolio = vbt.Portfolio.from_signals(
            ...     price, entries, exits, size=1.,
            ...     accumulate=True)
            >>> portfolio.share_flow()
            2020-01-01    1.0
            2020-01-02    1.0
            2020-01-03    0.0
            2020-01-04   -2.0
            2020-01-05    0.0
            dtype: float64

            >>> portfolio = vbt.Portfolio.from_signals(
            ...     price, entries, exits, size=1.,
            ...     accumulate=True,
            ...     accumulate_exit_mode=AccumulateExitMode.Reduce)
            >>> portfolio.share_flow()  # same as using from_orders
            2020-01-01    1.0
            2020-01-02    1.0
            2020-01-03    0.0
            2020-01-04   -1.0
            2020-01-05   -1.0
            dtype: float64
            ```
        """
        # Get defaults
        if size is None:
            size = defaults.portfolio['size']
        if entry_price is None:
            entry_price = close
        if exit_price is None:
            exit_price = close
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if reject_prob is None:
            reject_prob = defaults.portfolio['reject_prob']
        if min_size is None:
            min_size = defaults.portfolio['min_size']
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
            if isinstance(init_cash, str):
                init_cash = getattr(InitCashMode, init_cash)
        if isinstance(init_cash, int):
            checks.assert_in(init_cash, InitCashMode)
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
            if isinstance(call_seq, str):
                call_seq = getattr(CallSeqType, call_seq)
        if isinstance(call_seq, int):
            checks.assert_in(call_seq, CallSeqType)
            if call_seq == CallSeqType.Auto:
                raise ValueError("This method doesn't support CallSeqType.Auto")
        if accumulate is None:
            accumulate = defaults.portfolio['accumulate']
        if accumulate_exit_mode is None:
            accumulate_exit_mode = defaults.portfolio['accumulate_exit_mode']
            if isinstance(accumulate_exit_mode, str):
                accumulate_exit_mode = getattr(AccumulateExitMode, accumulate_exit_mode)
        checks.assert_in(accumulate_exit_mode, AccumulateExitMode)
        if conflict_mode is None:
            conflict_mode = defaults.portfolio['conflict_mode']
            if isinstance(conflict_mode, str):
                conflict_mode = getattr(ConflictMode, conflict_mode)
        checks.assert_in(conflict_mode, ConflictMode)
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

        # Perform checks
        checks.assert_type(close, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(close, np.floating)
        checks.assert_dtype(entries, np.bool)
        checks.assert_dtype(exits, np.bool)
        checks.assert_subdtype(size, np.floating)
        checks.assert_subdtype(entry_price, np.floating)
        checks.assert_subdtype(exit_price, np.floating)
        checks.assert_subdtype(fees, np.floating)
        checks.assert_subdtype(fixed_fees, np.floating)
        checks.assert_subdtype(slippage, np.floating)
        checks.assert_subdtype(reject_prob, np.floating)
        checks.assert_subdtype(min_size, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(call_seq, np.integer)

        # Broadcast inputs
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        keep_raw = (False, True, True, True, True, True, True, True, True, True, True)
        broadcast_kwargs = merge_kwargs(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        close, entries, exits, size, entry_price, exit_price, fees, fixed_fees, slippage, reject_prob = \
            broadcast(close, entries, exits, size, entry_price, exit_price, fees, fixed_fees,
                slippage, reject_prob, **broadcast_kwargs, keep_raw=keep_raw)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        min_size = np.require(np.broadcast_to(min_size, (target_shape_2d[1],)), requirements='W')
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape_2d, group_counts, call_seq_type=call_seq)

        # Perform calculation
        order_records = nb.simulate_from_signals_nb(
            target_shape_2d,
            cs_group_counts,  # group only if cash sharing is enabled to speed up
            init_cash,
            call_seq,
            entries,
            exits,
            size,
            entry_price,
            exit_price,
            fees,
            fixed_fees,
            slippage,
            reject_prob,
            min_size,
            accumulate,
            accumulate_exit_mode,
            conflict_mode,
            close.ndim == 2
        )

        # Create an instance
        orders = Orders(wrapper, order_records, close)
        return cls(
            orders,
            init_cash if init_cash_mode is None else init_cash_mode,
            cash_sharing,
            call_seq,
            **kwargs
        )

    @classmethod
    def from_orders(cls, close, order_size, size_type=None, order_price=None, fees=None, fixed_fees=None,
                    slippage=None, reject_prob=None, min_size=None, init_cash=None, cash_sharing=None,
                    call_seq=None, val_price=None, freq=None, seed=None, group_by=None, broadcast_kwargs=None,
                    wrapper_kwargs=None, **kwargs):
        """Simulate portfolio from orders.

        Starting with initial cash `init_cash`, orders the number of shares specified in `order_size`
        for `order_price`.

        Args:
            close (pandas_like): Reference price, such as close. Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            order_size (float or array_like): Size to order. Will broadcast.

                For any size type:

                * Set to `np.nan` to skip.
                * Set to `np.inf`/`-np.inf` to buy/sell everything.

                For `SizeType.Shares`:

                * Set to positive/negative to buy/sell.
                * Set to zero to skip.

                For target size, the final size will depend upon current holdings.
            size_type (SizeType or array_like): See `vectorbt.portfolio.enums.SizeType`.
            order_price (array_like of float): Order price. Defaults to `close`. Will broadcast.

                !!! note
                    Setting order price to close is risky.
            fees (float or array_like): Fees in percentage of the order value. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price. Will broadcast.
            reject_prob (float or array_like): Order rejection probability. Will broadcast.
            min_size (float or array_like): Minimum size for an order to be accepted.

                Will broadcast to the number of columns.
            init_cash (InitCashMode, float or array_like of float): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
                See `vectorbt.portfolio.enums.InitCashMode` to find optimal initial cash.

                !!! note
                    Mode `InitCashMode.AutoAlign` is applied after the portfolio is initialized
                    to set the same initial cash for all columns/groups. Changing grouping
                    will change the initial cash, so be aware when indexing.

                    Make sure that `init_cash` is a floating number if not using `InitCashMode`.
            cash_sharing (bool): Whether to share cash within the same group.

                !!! warning
                    Order execution cannot be considered parallel anymore.

                    This method presumes that in a group of assets that share the same capital all
                    orders will be executed within the same tick and retain their price regardless
                    of their position in the queue, even though they depend upon each other and thus
                    cannot be executed in parallel. This behavior is risky.
            call_seq (CallSeqType or array_like of int): Default sequence of calls per row and group.

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
                    * If one order is rejected, it still will execute next orders and possibly
                        leave them without funds that could have been released by the first order.

                    For more control, use `Portfolio.from_order_func`.
            val_price (array_like of float): Size valuation price. Defaults to previous `close`.
                Will broadcast.

                Used to calculate `SizeType.TargetPercent` and `SizeType.TargetValue`.

                !!! note
                    Make sure to use timestamp for `val_price` that comes before timestamps of all orders
                    in the group with cash sharing, otherwise you're cheating yourself.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            freq (any): Index frequency in case `close.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.

        All time series will be broadcast together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            When `call_seq` is not `CallSeqType.Auto`, at each timestamp, processing of the assets in
            a group goes strictly in order defined in `call_seq`. This order can't be changed dynamically.

            This has one big implication for this particular method: the last asset in the call stack
            cannot be processed until other assets are processed. This is the reason why rebalancing
            cannot work properly in this setting: one has to specify percentages for all assets beforehand
            and then tweak the processing order to sell to-be-sold assets first in order to release funds
            for to-be-bought assets. This can be automatically done by using `CallSeqType.Auto`.

        Example:
            The same equal-weighted portfolio as in `vectorbt.portfolio.nb.simulate_nb`.
            It's more compact but has no control over how order of execution impacts order price.

            ```python-repl
            >>> import numpy as np
            >>> import pandas as pd
            >>> import vectorbt as vbt
            >>> from vectorbt.portfolio.enums import SizeType, CallSeqType

            >>> np.random.seed(42)
            >>> price = pd.DataFrame(np.random.uniform(1, 10, size=(5, 3)))
            >>> orders = pd.DataFrame(np.full((5, 3), 1.) / 3)  # each column 33.3%
            >>> orders[1::2] = np.nan  # skip every second tick

            >>> portfolio = vbt.Portfolio.from_orders(
            ...     price,  # reference price for portfolio value
            ...     orders,
            ...     order_price=price,  # order price
            ...     size_type=SizeType.TargetPercent,
            ...     val_price=price,  # order price known beforehand (don't do it)
            ...     call_seq=CallSeqType.Auto,  # first sell then buy
            ...     group_by=np.array([0, 0, 0]),
            ...     cash_sharing=True,
            ...     fees=0.001, fixed_fees=1., slippage=0.001
            ... )

            >>> portfolio.holding_value(group_by=False)
                       0          1          2
            0  33.333333  33.333333  30.139624
            1  48.716002   8.385865   9.548589
            2  19.546625  22.584433  22.584433
            3  94.638155   3.043394  34.278783
            4  41.923304  38.661499  41.923304
            ```
        """
        # Get defaults
        if order_size is None:
            order_size = defaults.portfolio['order_size']
        if size_type is None:
            size_type = defaults.portfolio['size_type']
            if isinstance(size_type, str):
                size_type = getattr(SizeType, size_type)
        if order_price is None:
            order_price = close
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if reject_prob is None:
            reject_prob = defaults.portfolio['reject_prob']
        if min_size is None:
            min_size = defaults.portfolio['min_size']
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
            if isinstance(init_cash, str):
                init_cash = getattr(InitCashMode, init_cash)
        if isinstance(init_cash, int):
            checks.assert_in(init_cash, InitCashMode)
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
            if isinstance(call_seq, str):
                call_seq = getattr(CallSeqType, call_seq)
        auto_call_seq = False
        if isinstance(call_seq, int):
            checks.assert_in(call_seq, CallSeqType)
            if call_seq == CallSeqType.Auto:
                call_seq = CallSeqType.Default
                auto_call_seq = True
        if val_price is None:
            val_price = close.vbt.fshift(1)
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

        # Perform checks
        checks.assert_type(close, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(close, np.floating)
        checks.assert_subdtype(order_size, np.floating)
        checks.assert_subdtype(size_type, np.integer)
        checks.assert_subdtype(order_price, np.floating)
        checks.assert_subdtype(fees, np.floating)
        checks.assert_subdtype(fixed_fees, np.floating)
        checks.assert_subdtype(slippage, np.floating)
        checks.assert_subdtype(reject_prob, np.floating)
        checks.assert_subdtype(min_size, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(call_seq, np.integer)
        checks.assert_subdtype(val_price, np.floating)

        # Broadcast inputs
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        keep_raw = (False, True, True, True, True, True, True, True, True, True)
        broadcast_kwargs = merge_kwargs(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        close, order_size, size_type, order_price, fees, fixed_fees, slippage, reject_prob, val_price = \
            broadcast(close, order_size, size_type, order_price, fees, fixed_fees, slippage,
                      reject_prob, val_price, **broadcast_kwargs, keep_raw=keep_raw)
        target_shape_2d = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        min_size = np.require(np.broadcast_to(min_size, (target_shape_2d[1],)), requirements='W')
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape_2d, group_counts, call_seq_type=call_seq)

        # Perform calculation
        order_records = nb.simulate_from_orders_nb(
            target_shape_2d,
            cs_group_counts,  # group only if cash sharing is enabled to speed up
            init_cash,
            call_seq,
            order_size,
            size_type,
            order_price,
            fees,
            fixed_fees,
            slippage,
            reject_prob,
            min_size,
            val_price,
            auto_call_seq,
            close.ndim == 2
        )

        # Create an instance
        orders = Orders(wrapper, order_records, close)
        return cls(
            orders,
            init_cash if init_cash_mode is None else init_cash_mode,
            cash_sharing,
            call_seq,
            **kwargs
        )

    @classmethod
    def from_order_func(cls, close, order_func_nb, *order_args, target_shape=None, keys=None,
                        init_cash=None, cash_sharing=None, call_seq=None, active_mask=None, min_size=None,
                        prep_func_nb=None, prep_args=None, group_prep_func_nb=None, group_prep_args=None,
                        row_prep_func_nb=None, row_prep_args=None, segment_prep_func_nb=None,
                        segment_prep_args=None, row_wise=None, seed=None, freq=None, group_by=None,
                        broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Build portfolio from a custom order function.

        For details, see `vectorbt.portfolio.nb.simulate_nb`.

        if `row_wise` is True, also see `vectorbt.portfolio.nb.simulate_row_wise_nb`.

        Args:
            close (pandas_like): Reference price, such as close. Will broadcast to `target_shape`.

                Will be used for calculating unrealized P&L and portfolio value.

                Previous `close` will also be used for valuating assets/groups during the simulation.
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

                    Make sure that `init_cash` is a floating number if not using `InitCashMode`.
            cash_sharing (bool): Whether to share cash within the same group.

                !!! warning
                    Order execution cannot be considered parallel anymore.
            call_seq (CallSeqType or array_like of int): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.
            active_mask (bool or array_like): Mask of whether a particular segment should be executed.

                By default, will broadcast to the number of rows and groups.
            min_size (float or array_like): Minimum size for an order to be accepted.

                Will broadcast to the number of columns.
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
        """
        # Get defaults
        if target_shape is None:
            target_shape = close.shape
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
            if isinstance(init_cash, str):
                init_cash = getattr(InitCashMode, init_cash)
        if isinstance(init_cash, int):
            checks.assert_in(init_cash, InitCashMode)
            init_cash_mode = init_cash
            init_cash = np.inf
        else:
            init_cash_mode = None
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
            if isinstance(call_seq, str):
                call_seq = getattr(CallSeqType, call_seq)
        if isinstance(call_seq, int):
            checks.assert_in(call_seq, CallSeqType)
            if call_seq == CallSeqType.Auto:
                raise ValueError("CallSeqType.Auto should be implemented manually."
                                 "Use auto_call_seq_ctx_nb in segment_prep_func_nb.")
        if active_mask is None:
            active_mask = True
        if min_size is None:
            min_size = defaults.portfolio['min_size']
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

        # Perform checks
        checks.assert_type(close, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(close, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(call_seq, np.integer)

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
        min_size = np.require(np.broadcast_to(min_size, (target_shape_2d[1],)), requirements='W')
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        active_mask = broadcast(
            active_mask,
            to_shape=(target_shape_2d[0], len(group_counts)),
            to_pd=False,
            **require_kwargs
        )
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape_2d, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape_2d, group_counts, call_seq_type=call_seq)

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

        prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in prep_args])
        group_prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in group_prep_args])
        row_prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in row_prep_args])
        segment_prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in segment_prep_args])
        order_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in order_args])

        # Perform calculation
        if row_wise:
            order_records = nb.simulate_row_wise_nb(
                target_shape_2d,
                to_2d(close, raw=True),
                group_counts,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
                min_size,
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
            order_records = nb.simulate_nb(
                target_shape_2d,
                to_2d(close, raw=True),
                group_counts,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
                min_size,
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
        return cls(
            orders,
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

    # ############# Cash ############# #

    @cached_method
    def init_cash(self, group_by=None):
        """Get initial amount of cash per column/group."""
        if isinstance(self._init_cash, int):
            cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
            init_cash = -np.min(np.cumsum(cash_flow, axis=0), axis=0)
            if self._init_cash == InitCashMode.AutoAlign:
                init_cash = np.full(init_cash.shape, np.max(init_cash))
        else:
            init_cash = to_1d(self._init_cash, raw=True)
            if self.wrapper.grouper.is_grouped(group_by=group_by):
                group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
                init_cash = nb.init_cash_grouped_nb(init_cash, group_counts, self.cash_sharing)
            else:
                group_counts = self.wrapper.grouper.get_group_counts()
                init_cash = nb.init_cash_ungrouped_nb(init_cash, group_counts, self.cash_sharing)
        return self.wrapper.wrap_reduced(init_cash, group_by=group_by)

    @cached_method
    def cash_flow(self, group_by=None):
        """Get cash flow series per column/group."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            cash_flow_ungrouped = to_2d(self.cash_flow(group_by=False), raw=True)
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            cash_flow = nb.cash_flow_grouped_nb(cash_flow_ungrouped, group_counts)
        else:
            cash_flow = nb.cash_flow_ungrouped_nb(self.wrapper.shape_2d, self._orders.records_arr)
        return self.wrapper.wrap(cash_flow, group_by=group_by)

    @cached_method
    def cash(self, group_by=None, in_sim_order=False):
        """Get cash series per column/group."""
        if in_sim_order and not self.cash_sharing:
            raise ValueError("Cash sharing must be enabled for in_sim_order=True")

        cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            init_cash = to_1d(self.init_cash(group_by=group_by), raw=True)
            cash = nb.cash_grouped_nb(
                self.wrapper.shape_2d,
                cash_flow,
                group_counts,
                init_cash
            )
        else:
            group_counts = self.wrapper.grouper.get_group_counts()
            init_cash = to_1d(self.init_cash(group_by=in_sim_order), raw=True)
            call_seq = to_2d(self.call_seq, raw=True)
            cash = nb.cash_ungrouped_nb(
                cash_flow,
                group_counts,
                init_cash,
                call_seq,
                in_sim_order
            )
        return self.wrapper.wrap(cash, group_by=group_by)

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
    def holding_mask(self, group_by=None):
        """Get holding mask per column/group."""
        shares = to_2d(self.shares(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            holding_mask = nb.holding_mask_grouped_nb(shares, group_counts)
        else:
            holding_mask = shares > 0
        return self.wrapper.wrap(holding_mask, group_by=group_by)

    @cached_method
    def holding_duration(self, group_by=None):
        """Get holding duration per column/group."""
        holding_mask = to_2d(self.holding_mask(group_by=group_by))
        holding_duration = np.mean(holding_mask, axis=0)
        return self.wrapper.wrap_reduced(holding_duration, group_by=group_by)

    # ############# Records ############# #

    @cached_method
    def orders(self, group_by=None):
        """Order records.

        See `vectorbt.records.orders.Orders`."""
        return self._orders.regroup(group_by=group_by)

    @cached_method
    def trades(self, group_by=None, incl_unrealized=None):
        """Get trade records from orders.

        See `vectorbt.records.events.Trades`."""
        trades = Trades.from_orders(self.orders(group_by=group_by))
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized
        if incl_unrealized:
            return trades
        return trades.closed

    @cached_method
    def positions(self, group_by=None, incl_unrealized=None):
        """Get position records from orders.

        See `vectorbt.records.events.Positions`."""
        positions = Positions.from_orders(self.orders(group_by=group_by))
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized
        if incl_unrealized:
            return positions
        return positions.closed

    @cached_method
    def drawdowns(self, **kwargs):
        """Get drawdown records from `Portfolio.value`.

        See `vectorbt.records.drawdowns.Drawdowns`."""
        return Drawdowns.from_ts(self.value(**kwargs), freq=self.wrapper.freq)

    # ############# Performance ############# #

    @cached_method
    def holding_value(self, group_by=None):
        """Get holding value series per column/group."""
        close = to_2d(self.close, raw=True).copy()
        shares = to_2d(self.shares(), raw=True)
        close[shares == 0.] = 0.  # for price being NaN
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            holding_value = nb.holding_value_grouped_nb(close, shares, group_counts)
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
            group_counts = self.wrapper.grouper.get_group_counts()
            call_seq = to_2d(self.call_seq, raw=True)
            value = nb.value_in_sim_order_nb(cash, holding_value, group_counts, call_seq)
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
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            total_profit = nb.total_profit_grouped_nb(
                total_profit_ungrouped,
                group_counts
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
    def buy_and_hold_return(self, group_by=None):
        """Get total return of buy-and-hold.

        If grouped, invests same amount of cash into each asset and returns the total
        return of the entire group.

        !!! note
            Does not take into account fees and slippage. For this, create a separate portfolio."""
        ref_price_filled = to_2d(self.fill_close(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            total_return = nb.buy_and_hold_return_grouped_nb(ref_price_filled, group_counts)
        else:
            total_return = nb.buy_and_hold_return_ungrouped_nb(ref_price_filled)
        return self.wrapper.wrap_reduced(total_return, group_by=group_by)

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
    def returns(self, group_by=None, in_sim_order=False):
        """Get return series per column/group based on portfolio value."""
        value = to_2d(self.value(group_by=group_by, in_sim_order=in_sim_order), raw=True)
        if self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and in_sim_order:
            group_counts = self.wrapper.grouper.get_group_counts()
            init_cash_grouped = to_1d(self.init_cash(), raw=True)
            call_seq = to_2d(self.call_seq, raw=True)
            returns = nb.returns_in_sim_order_nb(value, group_counts, init_cash_grouped, call_seq)
        else:
            init_cash = to_1d(self.init_cash(group_by=group_by), raw=True)
            returns = nb.returns_nb(value, init_cash)
        return self.wrapper.wrap(returns, group_by=group_by)

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
        trades = self.trades(group_by=group_by, incl_unrealized=incl_unrealized)
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
            'Holding Duration [%]': self.holding_duration(group_by=group_by) * 100,
            'Total Profit': self.total_profit(group_by=group_by),
            'Total Return [%]': self.total_return(group_by=group_by) * 100,
            'Buy & Hold Return [%]': self.buy_and_hold_return(group_by=group_by) * 100,
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

