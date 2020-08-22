"""Base class for modeling portfolio and measuring its performance.

The job of the `Portfolio` class is to create a series of positions allocated 
against a cash component, produce an equity curve, incorporate basic transaction costs
and produce a set of statistics about its performance. In particular it outputs
position/profit metrics and drawdown information.

## Workflow

`Portfolio` class can be instantiated using main price of the asset, initial capital, 
records of filled orders, and cash and shares balances (as a result of filling orders).
It also accepts many other parameters such as annualization factor.

* Order records are used to track trades and positions, and to measure their performance.
* Main price, initial capital, and balances are used to compute risk and performance metrics.

To simplify creation of order records and keeping track of balances, it exposes several convenience methods
with prefix `from_`. For example, you can use `Portfolio.from_signals` method to create and fill orders 
based on entry and exit signals. Alternatively, you can use `Portfolio.from_order_func` to define 
a custom order function. The results are then automatically passed to the constructor method of 
`Portfolio` and you will receive a portfolio instance ready to be used for performance measurements.

## Properties

The `Portfolio` class offers numerous properties for measuring the performance of a strategy. 
They can be categorized as follows:

* Time series indexed by time, such as `Portfolio.returns`.
* Metrics indexed by columns, such as `Portfolio.total_profit`.
* Group objects with own time series and metrics, such as `Portfolio.positions`.

### Caching

Each property is cached, thus properties can effectively build upon each other, without side effects.

!!! note
    Due to caching, `Portfolio` class is meant to be atomic and immutable, thus each public attribute
    is marked as read-only. To change any parameter, you need to create a new `Portfolio` instance.

## Indexing

In addition, you can use pandas indexing on the `Portfolio` class itself, which forwards
indexing operation to each `__init__` argument with pandas type:

```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> price = pd.Series([1, 2, 3, 2, 1], index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5)
... ]), name='a')
>>> orders = pd.DataFrame({
...     'a': [np.inf, 0, 0, 0, 0],
...     'b': [1, 1, 1, 1, -np.inf],
...     'c': [np.inf, -np.inf, np.inf, -np.inf, np.inf]
... }, index=price.index)
>>> portfolio = vbt.Portfolio.from_orders(price, orders, init_cash=100)

>>> portfolio.equity
                a      b           c
2020-01-01  100.0  100.0  100.000000
2020-01-02  200.0  101.0  200.000000
2020-01-03  300.0  103.0  200.000000
2020-01-04  200.0  100.0  133.333333
2020-01-05  100.0   96.0  133.333333

>>> portfolio['a'].equity
2020-01-01    100.0
2020-01-02    200.0
2020-01-03    300.0
2020-01-04    200.0
2020-01-05    100.0
Name: a, dtype: float64
```

!!! note
    Changing index (time axis) is not supported."""

import numpy as np
import pandas as pd

from vectorbt import defaults
from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.index_grouper import IndexGrouper
from vectorbt.generic import nb as generic_nb
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import SizeType
from vectorbt.records import Orders, Trades, Positions, Drawdowns


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Portfolio`."""
    if obj.wrapper.ndim == 1:
        raise TypeError("Indexing on Series is not supported")

    n_rows = len(obj.wrapper.index)
    n_cols = len(obj.wrapper.columns)
    col_mapper = obj.wrapper.wrap(np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)))
    col_mapper = pd_indexing_func(col_mapper)
    if not pd.Index.equals(col_mapper.index, obj.wrapper.index):
        raise NotImplementedError("Changing index (time axis) is not supported")
    new_cols = col_mapper.values[0]

    # Create new Portfolio instance
    return obj.__class__(
        pd_indexing_func(obj.orders),
        obj.init_cash.iloc[new_cols],
        obj.init_shares.iloc[new_cols],
        pd_indexing_func(obj.cash),
        pd_indexing_func(obj.shares)
    )


def add_returns_methods(func_names):
    """Class decorator to add `vectorbt.returns.accessors.Returns_Accessor` methods to `Portfolio`."""

    def wrapper(cls):
        for func_name in func_names:
            @cached_method
            def returns_method(
                    self,
                    *args,
                    group_by=None,
                    year_freq=None,
                    func_name=func_name,
                    returns_kwargs=None,
                    **kwargs):
                if returns_kwargs is None:
                    returns_kwargs = {}
                returns_acc = self.returns(group_by=group_by, **returns_kwargs) \
                    .vbt.returns(freq=self.wrapper.freq, year_freq=year_freq)
                return getattr(returns_acc, func_name)(*args, **kwargs)

            returns_method.__doc__ = f"See `vectorbt.returns.accessors.Returns_Accessor.{func_name}`."
            setattr(cls, func_name, returns_method)
        return cls

    return wrapper


@add_returns_methods([
    'daily_returns',
    'annual_returns',
    'cumulative_returns',
    'total_return',
    'annualized_return',
    'annualized_volatility',
    'calmar_ratio',
    'omega_ratio',
    'sharpe_ratio',
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
    'down_capture'
    'drawdown',
    'max_drawdown'
])
class Portfolio(PandasIndexer):
    """Class for modeling portfolio and measuring its performance.

    Args:
        orders (vectorbt.records.orders.Orders): Order records.
        init_cash (float or pd.Series): Initial amount of cash.

            Each element must correspond to a column/group.
        init_shares (float or pd.Series): Initial amount of shares.

            Each element must correspond to a column/group.
        cash (pandas_like): Final cash at each time step.

            Must have the same metadata as `orders.ref_price`.
        shares (pandas_like): Final shares at each time step.

            Must have the same metadata as `orders.ref_price`.
        incl_unrealized (bool): Whether to include unrealized P&L in statistics.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

        All array objects must have the same metadata as `orders.ref_price`."""

    def __init__(self, orders, init_cash, init_shares, cash, shares, incl_unrealized=False):
        # Perform checks
        checks.assert_type(orders, Orders)
        if checks.is_series(orders.ref_price):
            checks.assert_ndim(init_cash, 0)
            checks.assert_ndim(init_shares, 0)
        else:
            checks.assert_type(init_cash, pd.Series)
            checks.assert_type(init_shares, pd.Series)
            checks.assert_same(init_cash.index, orders.grouper.get_index())
            checks.assert_same(init_shares.index, orders.wrapper.columns)
        checks.assert_same_meta(orders.ref_price, cash)
        checks.assert_same_meta(orders.ref_price, shares)

        # Store passed arguments
        self._ref_price = orders.ref_price
        self._orders = orders
        self._init_cash = init_cash
        self._init_shares = init_shares
        self._cash = cash
        self._shares = shares
        self._incl_unrealized = incl_unrealized

        # Supercharge
        PandasIndexer.__init__(self, _indexing_func)
        self.wrapper = orders.wrapper
        self.grouper = IndexGrouper(
            orders.grouper.index,
            orders.grouper.group_by,
            allow_change=False,
            allow_enable=False,
            allow_disable=True
        )

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, ref_price, entries, exits, size=np.inf, size_type=SizeType.Shares,
                     entry_price=None, exit_price=None, init_cash=None, init_shares=None, fees=None,
                     fixed_fees=None, slippage=None, accumulate=False, group_by=None, broadcast_kwargs={},
                     freq=None, **kwargs):
        """Simulate portfolio from entry and exit signals.

        For each signal in `entries`, buys `size` of shares for `entry_price` to enter
        a position. For each signal in `exits`, sells everything for `exit_price`
        to exit the position. Accumulation of orders is disabled by default.
        When both entry and exit signal are present, buys/sells the difference in size.

        !!! note
            Only `vectorbt.portfolio.enums.SizeType.Shares` and
            `vectorbt.portfolio.enums.SizeType.Cash` are supported. Other modes
            such as target percentage are not compatible with signals since
            their logic may contradict the direction the user has specified for the order.

        For more details, see `vectorbt.portfolio.nb.simulate_from_signals_nb`.

        Args:
            ref_price (pandas_like): Reference price, such as close. Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            entries (array_like): Boolean array of entry signals. Will broadcast.
            exits (array_like): Boolean array of exit signals. Will broadcast.
            size (float or array_like): The amount of shares to order. Will broadcast.

                To buy/sell everything, set the size to `np.inf`.
            size_type (int or array_like): See `vectorbt.portfolio.enums.SizeType`.

                Only `SizeType.Shares` and `SizeType.Cash` are supported.
            entry_price (array_like): Entry price. Defaults to `ref_price`. Will broadcast.
            exit_price (array_like): Exit price. Defaults to `ref_price`. Will broadcast.
            init_cash (float or array_like): Initial amount of cash per group.

                Allowed is either a single value or value per group.
            init_shares (float or array_like): Initial amount of shares per column.

                Allowed is either a single value or value per column.
            fees (float or array_like): Fees in percentage of the order value. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price. Will broadcast.
            accumulate (bool): If `accumulate` is `True`, entering the market when already
                in the market will be allowed to increase the position.
            group_by (int, str or array_like): Group columns by a mapper.
                Columns within the same group will share the same capital from `init_cash`.

                See `vectorbt.base.index_fns.IndexGrouper`.
            broadcast_kwargs: Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            freq (any): Index frequency in case `ref_price.index` is not datetime-like.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        All time series will be broadcasted together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.
        """
        # Get defaults
        if entry_price is None:
            entry_price = ref_price
        if exit_price is None:
            exit_price = ref_price
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        if init_shares is None:
            init_shares = defaults.portfolio['init_shares']
        if size is None:
            size = defaults.portfolio['size']
        if size_type is None:
            size_type = defaults.portfolio['size_type']
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(ref_price, (pd.Series, pd.DataFrame))
        checks.assert_dtype(ref_price, np.float_)
        checks.assert_dtype(entries, np.bool_)
        checks.assert_dtype(exits, np.bool_)

        # Broadcast inputs
        # Only ref_price is broadcasted, others can remain unchanged thanks to flexible indexing
        keep_raw = (False, True, True, True, True, True, True, True, True, True)
        ref_price, entries, exits, size, size_type, entry_price, \
            exit_price, fees, fixed_fees, slippage = broadcast(
                ref_price, entries, exits, size, size_type, entry_price, exit_price, fees,
                fixed_fees, slippage, **broadcast_kwargs,
                writeable=True, keep_raw=keep_raw)
        target_shape = (ref_price.shape[0], ref_price.shape[1] if ref_price.ndim > 1 else 1)
        grouper = IndexGrouper(ref_price.vbt.columns, group_by=group_by)
        group_counts = grouper.get_group_counts()
        group_init_cash = np.broadcast_to(init_cash, (len(group_counts),))  # per group
        init_shares = np.broadcast_to(init_shares, (target_shape[1],))  # per column

        # Perform calculation
        order_records, cash, shares = nb.simulate_from_signals_nb(
            target_shape,
            group_counts,
            group_init_cash,
            init_shares,
            entries,
            exits,
            size,
            size_type,
            entry_price,
            exit_price,
            fees,
            fixed_fees,
            slippage,
            accumulate,
            is_2d=ref_price.ndim == 2
        )

        # Bring to the same meta
        cash = ref_price.vbt.wrap(cash)
        shares = ref_price.vbt.wrap(shares)
        orders = Orders(order_records, ref_price, freq=freq, group_by=group_by)
        if checks.is_frame(ref_price):
            group_init_cash = pd.Series(group_init_cash, index=grouper.get_index())
            init_shares = pd.Series(init_shares, index=ref_price.vbt.columns)
        else:
            group_init_cash = group_init_cash.item(0)
            init_shares = init_shares.item(0)

        return cls(
            orders,
            group_init_cash,
            init_shares,
            cash,
            shares,
            **kwargs
        )

    # ############# Read-only properties ############# #

    @property
    def orders(self):
        """Order records.

        See `vectorbt.records.orders.Orders`."""
        return self._orders

    @property
    def init_cash(self):
        """Initial amount of cash per group."""
        return self._init_cash

    @cached_property
    def init_cash_expanded(self):
        """Initial amount of cash per column."""
        if self.grouper.group_by is None:
            return self.init_cash
        init_cash_grouped = to_1d(self.init_cash, raw=True)
        # Un-group grouped cash series using forward fill
        init_cash_expanded = np.full(len(self.wrapper.columns), np.nan, dtype=np.float_)
        group_first_idxs = self.grouper.get_group_first_idxs()
        init_cash_expanded[group_first_idxs] = init_cash_grouped
        mask = np.isnan(init_cash_expanded)
        idx = np.where(~mask, np.arange(mask.shape[0]), 0)
        np.maximum.accumulate(idx, out=idx)
        init_cash_expanded = init_cash_expanded[idx]
        return self.wrapper.wrap_reduced(init_cash_expanded)

    @property
    def init_shares(self):
        """Initial amount of shares per column."""
        return self._init_shares

    @property
    def cash(self):
        """Final cash at each step."""
        return self._cash

    @property
    def shares(self):
        """Final shares at each step."""
        return self._shares

    @property
    def ref_price(self):
        """Price per share series."""
        return self._ref_price

    @property
    def incl_unrealized(self):
        """Whether to include unrealized trade P&L in statistics."""
        return self._incl_unrealized

    # ############# Grouping ############# #

    def get_group_by(self, group_by=None):
        """Get `group_by` from either object variable or keyword argument."""
        if group_by is None:
            group_by = self.grouper.group_by
        if group_by is False and self.grouper.group_by is None:
            group_by = None
        return group_by

    # ############# Records ############# #

    @cached_method
    def trades(self, group_by=None, incl_unrealized=None):
        """Get trade records.

        See `vectorbt.records.events.Trades`."""
        trades = Trades.from_orders(self.orders, group_by=group_by)
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized
        if incl_unrealized:
            return trades
        return trades.closed

    @cached_method
    def positions(self, group_by=None, incl_unrealized=None):
        """Get position records.

        See `vectorbt.records.events.Positions`."""
        positions = Positions.from_orders(self.orders, group_by=group_by)
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized
        if incl_unrealized:
            return positions
        return positions.closed

    @cached_method
    def drawdowns(self, **kwargs):
        """Get drawdown records.

        See `vectorbt.records.drawdowns.Drawdowns`.

        Keyword arguments are passed to `Portfolio.value`."""
        return Drawdowns.from_ts(self.value(**kwargs), freq=self.wrapper.freq)

    # ############# Performance ############# #

    @cached_method
    def init_holding_value(self, group_by=None):
        """Initial holding value."""
        group_by = self.get_group_by(group_by=group_by)
        init_price = to_2d(self.ref_price, raw=True)[0, :].copy()
        init_shares = to_1d(self.init_shares, raw=True)
        init_price[init_shares == 0.] = 0.  # for price being NaN
        holding_value = init_shares * init_price

        if group_by is not None and group_by is not False:
            groups = self.grouper.get_groups(group_by=group_by)
            # Abuse histogram to get grouped sum
            holding_value = np.bincount(groups, weights=holding_value)

        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap_reduced(holding_value, columns=columns)

    @cached_method
    def init_value(self, group_by=None):
        """Initial portfolio value."""
        group_by = self.get_group_by(group_by=group_by)

        if group_by is False:
            init_cash = to_1d(self.init_cash_expanded, raw=True)
        else:
            init_cash = to_1d(self.init_cash, raw=True)

        init_holding_value = to_1d(self.init_holding_value(group_by=group_by), raw=True)
        init_value = init_cash + init_holding_value
        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap_reduced(init_value, columns=columns)

    @cached_method
    def cash_flow(self, group_by=None):
        """Get cash flows."""
        group_by = self.get_group_by(group_by=group_by)
        cash = to_2d(self.cash, raw=True)
        init_cash = to_1d(self.init_cash, raw=True)

        if group_by is False:
            group_counts = self.grouper.get_group_counts()
            cash_flow = nb.ungrouped_cash_flow_nb(cash, init_cash, group_counts)
        else:
            if group_by is not None:
                group_last_idxs = self.grouper.get_group_last_idxs(group_by=group_by)
                cash = cash[:, group_last_idxs]
            cash_flow = np.empty(cash.shape, dtype=np.float_)
            cash_flow[0, :] = cash[0, :] - init_cash
            cash_flow[1:, :] = cash[1:, :] - cash[:-1, :]

        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap(cash_flow, columns=columns)

    @cached_method
    def holding_value(self, group_by=None):
        """Get holding value."""
        group_by = self.get_group_by(group_by=group_by)
        ref_price = to_2d(self.ref_price, raw=True).copy()
        shares = to_2d(self.shares, raw=True)
        ref_price[shares == 0.] = 0.  # for price being NaN
        if group_by is None or group_by is False:
            holding_value = shares * ref_price
        else:
            group_counts = self.grouper.get_group_counts(group_by=group_by)
            holding_value = nb.grouped_holding_value_nb(ref_price, shares, group_counts)
        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap(holding_value, columns=columns)

    @cached_method
    def value(self, group_by=None, iterative=False):
        """Get portfolio value.

        By default, will generate portfolio value for each asset based on cash flows and thus
        independent from other assets, with initial cash and shares being that of the entire group.
        Useful for generating returns and comparing assets within the same group.

        When `group_by` is `False` and `iterative` is `True`, returns value generated in
        simulation order (see [row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
        This value cannot be used for generating returns as-is. Useful to analyze how value
        evolved throughout simulation."""
        group_by = self.get_group_by(group_by=group_by)
        cash = to_2d(self.cash, raw=True)
        holding_value = to_2d(self.holding_value(group_by=group_by), raw=True)

        if group_by is None:
            value = cash + holding_value
        else:
            if group_by is False:
                if iterative:
                    group_counts = self.grouper.get_group_counts()
                    value = nb.ungrouped_iter_value_nb(cash, holding_value, group_counts)
                else:
                    init_cash = to_1d(self.init_cash_expanded, raw=True)
                    cash_flow = to_2d(self.cash_flow(group_by=False), raw=True)
                    holding_value = to_2d(self.holding_value(group_by=False), raw=True)
                    value = init_cash + np.cumsum(cash_flow, axis=0) + holding_value
            else:
                group_last_idxs = self.grouper.get_group_last_idxs(group_by=group_by)
                value = cash[:, group_last_idxs] + holding_value
                # price of NaN is already addressed by ungrouped_value_nb

        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap(value, columns=columns)

    @cached_method
    def final_value(self, group_by=None, iterative=False):
        """Get final portfolio value.

        For details on `iterative`, see `Portfolio.value`."""
        value = to_2d(self.value(group_by=group_by, iterative=iterative), raw=True)
        final_value = generic_nb.ffill_nb(value)[-1, :]
        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap_reduced(final_value, columns=columns)

    @cached_method
    def total_profit(self, group_by=None):
        """Get total profit."""
        final_value = to_1d(self.final_value(group_by=group_by, iterative=False), raw=True)
        init_value = to_1d(self.init_value, raw=True)
        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap_reduced(final_value - init_value, columns=columns)

    @cached_method
    def returns(self, group_by=None, iterative=False, active_only=False):
        """Get portfolio returns.

        If `iterative` is `True`, see `Portfolio.value`.

        When `active_only` is `True`, does not take into account passive cash. This way, it will
        return the same numbers irrespective of the amount of cash currently available. Holding
        10$ or 100$ worth of security out of 1000$ available in cash will generate the same returns,
        while with `active_only=False` potential returns of 10$-worth holdings will be much smaller
        compared to 100$-worth holdings, so comparing them will be less fair."""
        if active_only:
            cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
            init_holding_value = to_1d(self.init_holding_value(group_by=group_by), raw=True)
            holding_value = to_2d(self.holding_value(group_by=group_by), raw=True)
            input_value = np.vstack((init_holding_value, holding_value[:-1, :]))
            input_value[cash_flow < 0] += -cash_flow[cash_flow < 0]
            output_value = holding_value.copy()
            output_value[cash_flow > 0] += cash_flow[cash_flow > 0]
            returns = (output_value - input_value) / input_value
        else:
            init_value = to_1d(self.init_value, raw=True)
            value = self.value(group_by=group_by, iterative=iterative)
            if group_by is False and iterative:
                group_counts = self.grouper.get_group_counts(group_by=group_by)
                returns = nb.ungrouped_iter_returns_nb(value, init_value, group_counts)
            else:
                returns = np.empty(value.shape, dtype=np.float_)
                returns[0, :] = value[0, :] - init_value
                returns[1:, :] = (value[1:, :] - value[:-1, :]) / value[:-1, :]

        returns[np.isnan(returns)] = 0.  # can happen if later price is unknown
        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap(returns, columns=columns)

    @cached_method
    def buy_and_hold_return(self, group_by=None):
        """Get total return of buy-and-hold.

        If grouped, invests same amount of cash into each asset and returns the total
        return of the entire group by summing up their holding values.

        !!! note
            Does not take into account fees and slippage. For this, create a separate portfolio."""
        if group_by is None:
            group_by = self.grouper.group_by
        ref_price = to_2d(self.ref_price, raw=True)
        if group_by is None or group_by is False:
            total_return = (ref_price[-1, :] - ref_price[0, :]) / ref_price[0, :]
        else:
            group_counts = self.grouper.get_group_counts(group_by=group_by)
            total_return = nb.grouped_buy_and_hold_return_nb(ref_price, group_counts)
        columns = self.grouper.get_index(group_by=group_by)
        return self.wrapper.wrap_reduced(total_return, columns=columns)

    @cached_method
    def stats(self, group_by=None, incl_unrealized=None, returns_kwargs=None):
        """Compute various statistics on this portfolio.

        `returns_kwargs` will be passed to `Portfolio.returns`."""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column/group first")
        positions = self.positions(group_by=group_by, incl_unrealized=incl_unrealized)
        trades = self.trades(group_by=group_by, incl_unrealized=incl_unrealized)
        drawdowns = self.drawdowns(group_by=group_by)
        if returns_kwargs is None:
            returns_kwargs = {}

        return pd.Series({
            'Start': self.wrapper.index[0],
            'End': self.wrapper.index[-1],
            'Duration': self.wrapper.shape[0] * self.wrapper.freq,
            'Holding Duration [%]': positions.coverage() * 100,
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
            'Sharpe Ratio': self.sharpe_ratio(group_by=group_by, **returns_kwargs),
            'Sortino Ratio': self.sortino_ratio(group_by=group_by, **returns_kwargs),
            'Calmar Ratio': self.calmar_ratio(group_by=group_by, **returns_kwargs)
        }, name=self.wrapper.name)


