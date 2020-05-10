"""Classes for building portfolios and measuring their performance.
    
Before running the examples:
```py
import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime

index = pd.Index([
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5)
])
price = pd.Series([1, 2, 3, 2, 1], index=index, name='a')
```"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta
from inspect import isfunction

from vectorbt import timeseries, accessors, defaults
from vectorbt.utils import indexing, checks, reshape_fns, common
from vectorbt.utils.common import cached_property
from vectorbt.portfolio import nb
from vectorbt.widgets import DefaultFigureWidget


class ArrayWrapper():
    """Provides methods for wrapping NumPy arrays."""
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def wrap_array(self, a):
        """Wrap output array to the time series format of this portfolio."""
        return self.portfolio.price.vbt.wrap_array(a)

    def wrap_reduced_array(self, a, **kwargs):
        """Wrap output array to the metric format of this portfolio."""
        return self.portfolio.price.vbt.timeseries.wrap_reduced_array(a, **kwargs)


class BasePositions(ArrayWrapper):
    """Exposes a range of attributes on top of positions in a `Portfolio` instance.
    
    This class doesn't hold any data, but creates a read-only view over position data.

    Args:
        pos_status (int): Can be any of: 
        
            * `vectorbt.portfolio.nb.OPEN` for open positions only,
            * `vectorbt.portfolio.nb.CLOSED` for closed positions only, or 
            * `None` for positions of any type.
        pos_filters (list or tuple): Can be used to further filter positions.

            Each element must be either: 
            
            * a Numba-compiled function, or 
            * a tuple of a Numba-compiled function and its (unpacked) arguments.

            !!! note
                Each `filter_func_nb` must be Numba-compiled.

    Example:
        Get the average P/L of closed positions with duration over 2 days:
        ```
        >>> from vectorbt.portfolio import CLOSED, BasePositions

        >>> orders = pd.Series([1, -1, 1, 0, -1], index=index)
        >>> portfolio = vbt.Portfolio.from_orders(price, orders, init_capital=100)
        >>> print(portfolio.positions.avg_pnl)
        -0.5

        >>> @njit
        ... def duration_filter_func_nb(col, i, map_result, duration):
        ...     return duration[i, col] >= 2

        >>> positions = BasePositions(
        ...     portfolio, 
        ...     pos_status=CLOSED, 
        ...     pos_filters=[(
        ...         duration_filter_func_nb, 
        ...         portfolio.positions.duration.vbt.to_2d_array()
        ...     )])
        >>> print(positions.avg_pnl)
        -2.0
        ```"""
    def __init__(self, portfolio, pos_status=None, pos_filters=[]):
        ArrayWrapper.__init__(self, portfolio)

        self.portfolio = portfolio
        self.pos_status = pos_status
        self.pos_filters = pos_filters

    def apply_mapper(self, map_func_nb, *args):
        """Apply `map_func_nb` on each position using `vectorbt.portfolio.nb.map_positions_nb` 
        and filter the results with `pos_filters`.
        
        This way, all time series created on top of positions will be automatically filtered."""
        checks.assert_numba_func(map_func_nb)

        # Apply map
        result = nb.map_positions_nb(
            self.portfolio.shares.vbt.to_2d_array(),
            self.pos_status,
            map_func_nb,
            *args)
        result = self.wrap_array(result)

        # Apply passed filters
        for pos_filter in self.pos_filters:
            if isfunction(pos_filter):
                filter_func_nb = pos_filter
                args = ()
            else:
                filter_func_nb = pos_filter[0]
                if len(pos_filter) > 1:
                    args = pos_filter[1:]
                else:
                    args = ()
            checks.assert_numba_func(filter_func_nb)
            result = result.vbt.timeseries.filter(filter_func_nb, *args)

        return result

    # ############# Status ############# #

    @cached_property
    def status(self):
        """Position status (open/closed) at the end of each position."""
        return self.apply_mapper(nb.status_map_func_nb)

    @cached_property
    def count(self):
        """Total position count of each column."""
        return self.status.vbt.timeseries.count()

    # ############# Duration ############# #

    @cached_property
    def duration(self):
        """Position duration at the end of each position."""
        return self.apply_mapper(nb.duration_map_func_nb, self.portfolio.price.shape)

    @cached_property
    def min_duration(self):
        """Minimum position duration of each column."""
        return self.duration.vbt.timeseries.min(time_units=True)

    @cached_property
    def max_duration(self):
        """Maximum position duration of each column."""
        return self.duration.vbt.timeseries.max(time_units=True)

    @cached_property
    def total_duration(self):
        """Total position duration of each column."""
        return self.duration.vbt.timeseries.sum(time_units=True)

    @cached_property
    def avg_duration(self):
        """Average position duration of each column."""
        return self.duration.vbt.timeseries.mean(time_units=True)

    # ############# PnL ############# #

    @cached_property
    def pnl(self):
        """Position P/L at the end of each position."""
        return self.apply_mapper(
            nb.pnl_map_func_nb,
            self.portfolio.price.vbt.to_2d_array(),
            self.portfolio.cash.vbt.to_2d_array(),
            self.portfolio.shares.vbt.to_2d_array(),
            self.portfolio.init_capital)

    def plot_pnl(self, profit_trace_kwargs={}, loss_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot position P/L as markers.

        Args:
            profit_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) for "Profit" markers.
            loss_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) for "Loss" markers.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            portfolio = vbt.Portfolio.from_orders(price, price.diff(), init_capital=100)
            portfolio.positions.plot_pnl()
            ```

            ![](img/positions_plot_pnl.png)"""
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.pnl.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)

    @cached_property
    def min_pnl(self):
        """Minimum position P/L of each column."""
        return self.pnl.vbt.timeseries.min()

    @cached_property
    def max_pnl(self):
        """Maximum position P/L of each column."""
        return self.pnl.vbt.timeseries.max()

    @cached_property
    def total_pnl(self):
        """Total position P/L of each column."""
        return self.pnl.vbt.timeseries.sum()

    @cached_property
    def avg_pnl(self):
        """Average position P/L of each column."""
        return self.pnl.vbt.timeseries.mean()

    # ############# Returns ############# #

    @cached_property
    def returns(self):
        """Position returns at the end of each position."""
        return self.apply_mapper(
            nb.returns_map_func_nb,
            self.portfolio.price.vbt.to_2d_array(),
            self.portfolio.cash.vbt.to_2d_array(),
            self.portfolio.shares.vbt.to_2d_array(),
            self.portfolio.init_capital)

    def plot_returns(self, profit_trace_kwargs={}, loss_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot position returns as markers.

        See `BasePositions.plot_pnl`."""
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.returns.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)

    @cached_property
    def min_return(self):
        """Minimum position return of each column."""
        return self.returns.vbt.timeseries.min()

    @cached_property
    def max_return(self):
        """Maximum position return of each column."""
        return self.returns.vbt.timeseries.max()

    @cached_property
    def avg_return(self):
        """Average position return of each column."""
        return self.returns.vbt.timeseries.mean()


class Positions(BasePositions):
    """Extends `BasePositions` by combining various profit/loss metrics."""

    @property
    def winning(self):
        """Winning positions of class `BasePositions`."""
        if not hasattr(self, '_winning'):
            self._winning = BasePositions(
                self.portfolio,
                pos_status=self.pos_status,
                pos_filters=[*self.pos_filters, (nb.winning_filter_func_nb, self.pnl.vbt.to_2d_array())])
        return self._winning

    @property
    def losing(self):
        """Losing positions of class `BasePositions`."""
        if not hasattr(self, '_losing'):
            self._losing = BasePositions(
                self.portfolio,
                pos_status=self.pos_status,
                pos_filters=[*self.pos_filters, (nb.losing_filter_func_nb, self.pnl.vbt.to_2d_array())])
        return self._losing

    @cached_property
    def win_rate(self):
        """How many positions won in each column."""
        winning_count = reshape_fns.to_1d(self.winning.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        win_rate = winning_count / count
        return self.wrap_reduced_array(win_rate)

    @cached_property
    def loss_rate(self):
        """How many positions lost in each column."""
        losing_count = reshape_fns.to_1d(self.losing.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        loss_rate = losing_count / count
        return self.wrap_reduced_array(loss_rate)

    @cached_property
    def profit_factor(self):
        """Profit factor of each column."""
        total_win = reshape_fns.to_1d(self.winning.total_pnl, raw=True)
        total_loss = reshape_fns.to_1d(self.losing.total_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_trades = reshape_fns.to_1d(self.portfolio.has_trades, raw=True)
        total_win[np.isnan(total_win) & has_trades] = 0.
        total_loss[np.isnan(total_loss) & has_trades] = 0.

        profit_factor = total_win / np.abs(total_loss)
        return self.wrap_reduced_array(profit_factor)

    @cached_property
    def expectancy(self):
        """Average profitability per trade (APPT) of each column."""
        win_rate = reshape_fns.to_1d(self.win_rate, raw=True)
        loss_rate = reshape_fns.to_1d(self.loss_rate, raw=True)
        avg_win = reshape_fns.to_1d(self.winning.avg_pnl, raw=True)
        avg_loss = reshape_fns.to_1d(self.losing.avg_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_trades = reshape_fns.to_1d(self.portfolio.has_trades, raw=True)
        avg_win[np.isnan(avg_win) & has_trades] = 0.
        avg_loss[np.isnan(avg_loss) & has_trades] = 0.

        expectancy = win_rate * avg_win - loss_rate * np.abs(avg_loss)
        return self.wrap_reduced_array(expectancy)


def portfolio_indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Portfolio`. 
    
    See `vectorbt.utils.indexing.add_pd_indexing`."""
    return obj.__class__(
        pd_indexing_func(obj.price),
        pd_indexing_func(obj.cash),
        pd_indexing_func(obj.shares),
        obj.init_capital,
        pd_indexing_func(obj.paid_fees),
        pd_indexing_func(obj.paid_slippage)
    )


@indexing.add_pd_indexing(portfolio_indexing_func)
class Portfolio(ArrayWrapper):
    """The job of the `Portfolio` class is to create a series of positions allocated 
    against a cash component, produce an equity curve, incorporate basic transaction costs 
    and produce a set of statistics about its performance. In particular it outputs 
    position/profit metrics and drawdown information.

    !!! note
        Portfolio is only built by using class methods with `from_` prefix.
        The `__init__` method is reserved for indexing purposes.

    It produces two types of objects:
    
    * time series such as `Portfolio.equity`, and
    * various metrics such as `Portfolio.total_profit`.
    
    The former are indexed by time, the latter are indexed by columns.

    ## Indexing

    In addition, you can use pandas indexing on the `Portfolio` class itself, which forwards
    indexing operation to each attribute with pandas type (see `portfolio_indexing_func`):
    
    ```python-repl
    >>> portfolio = vbt.Portfolio.from_orders(price, price.diff(), init_capital=100)

    >>> print(portfolio.equity)
    2018-01-01    100.0
    2018-01-02    100.0
    2018-01-03    101.0
    2018-01-04     99.0
    2018-01-05     98.0
    Name: a, dtype: float64
    >>> print(portfolio.loc['2018-01-03':].equity)
    2018-01-03    101.0
    2018-01-04     99.0
    2018-01-05     98.0
    Name: a, dtype: float64
    ```

    Note that for the new `Portfolio` instance, date `'2018-01-03'` will be the new start date.

    ## Addition

    You can also add multiple `Portfolio` instances together to combine portfolios:

    ```python-repl
    >>> portfolio1 = vbt.Portfolio.from_orders(price, price.diff(), init_capital=100)
    >>> portfolio2 = vbt.Portfolio.from_orders(price, price.diff()*2, init_capital=20)
    >>> portfolio = portfolio1 + portfolio2

    >>> print(portfolio.init_capital)
    120.0
    >>> print(portfolio.equity)
    2018-01-01    120.0
    2018-01-02    120.0
    2018-01-03    123.0
    2018-01-04    117.0
    2018-01-05    114.0
    Name: a, dtype: float64
    ```

    The only requirement is that pandas objects of both portfolios must have the same metadata."""

    def __init__(self, price, cash, shares, init_capital, paid_fees, paid_slippage):
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_same_meta(price, cash)
        checks.assert_same_meta(price, shares)

        self.price = price
        self.cash = cash
        self.shares = shares
        self.init_capital = init_capital
        self.paid_fees = paid_fees
        self.paid_slippage = paid_slippage

        ArrayWrapper.__init__(self, self)

    # ############# Magic methods ############# #

    def __add__(self, other):
        checks.assert_type(other, self.__class__)
        checks.assert_same(self.price, other.price)

        return self.__class__(
            self.price,
            self.cash + other.cash,
            self.shares + other.shares,
            self.init_capital + other.init_capital,
            self.paid_fees + other.paid_fees,
            self.paid_slippage + other.paid_slippage
        )

    def __radd__(self, other):
        return Portfolio.__add__(self, other)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, price, entries, exits, amount=np.inf, init_capital=None,
                     fees=None, slippage=None, broadcast_kwargs={}):
        """Build portfolio from entry and exit signals.

        Starting with initial capital `init_capital`, for each `True` in `entries`/`exits`, 
        orders the number of shares specified in `amount`. 

        Args:
            price (pandas_like): Price of the asset.
            entries (pandas_like): Boolean array of entry signals.
            exits (pandas_like): Boolean array of exit signals.
            amount (int, float or array_like): The amount of shares to order. 

                To buy/sell everything, set the amount to `numpy.inf`.
            init_capital (int or float): The initial capital.
            fees (float or array_like): Trading fees in percentage of the value involved.
            slippage (float or array_like): Slippage in percentage of `price`.

        All array-like arguments will be broadcasted together using `vectorbt.utils.reshape_fns.broadcast`
        with `broadcast_kwargs`. At the end, each time series object will have the same metadata.

        !!! note
            There is no mechanism implemented to prevent order accumulation, meaning multiple entry/exit 
            signals one after another may increase/decrease your position in the market. That's why we will
            later calculate P/L of positions instead of trades.

            To select at most one exit signal, use `vectorbt.signals.accessors.Signals_Accessor.first`. 

        Example:
            Portfolio value of various signal sequences:
            ```python-repl
            >>> entries = pd.DataFrame({
            ...     'a': [True, False, False, False, False],
            ...     'b': [True, True, True, True, True],
            ...     'c': [True, False, True, False, True]
            ... }, index=index)
            >>> exits = pd.DataFrame({
            ...     'a': [False, False, False, False, False],
            ...     'b': [False, False, False, False, False],
            ...     'c': [False, True, False, True, False]
            ... }, index=index)

            >>> portfolio = vbt.Portfolio.from_signals(price, entries, 
            ...     exits, amount=10, init_capital=100, fees=0.0025)

            >>> print(portfolio.cash)
                             a       b        c
            2018-01-01  89.975  89.975   89.975
            2018-01-02  89.975  69.925  109.925
            2018-01-03  89.975  39.850   79.850
            2018-01-04  89.975  19.800   99.800
            2018-01-05  89.975   9.775   89.775
            >>> print(portfolio.shares)
                           a     b     c
            2018-01-01  10.0  10.0  10.0
            2018-01-02  10.0  20.0   0.0
            2018-01-03  10.0  30.0  10.0
            2018-01-04  10.0  40.0   0.0
            2018-01-05  10.0  50.0  10.0
            >>> print(portfolio.equity)
                              a        b        c
            2018-01-01   99.975   99.975   99.975
            2018-01-02  109.975  109.925  109.925
            2018-01-03  119.975  129.850  109.850
            2018-01-04  109.975   99.800   99.800
            2018-01-05   99.975   59.775   99.775
            >>> print(portfolio.total_costs)
            a    0.025
            b    0.225
            c    0.225
            dtype: float64
            ```
        """
        # Get defaults
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        init_capital = float(init_capital)
        if fees is None:
            fees = defaults.portfolio['fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_type(entries, (pd.Series, pd.DataFrame))
        checks.assert_type(exits, (pd.Series, pd.DataFrame))
        entries.vbt.signals.validate()
        exits.vbt.signals.validate()

        # Broadcast inputs
        price, entries, exits, amount, fees, slippage = reshape_fns.broadcast(
            price, entries, exits, amount, fees, slippage, **broadcast_kwargs, writeable=True)

        # Perform calculation
        cash, shares, paid_fees, paid_slippage = nb.portfolio_nb(
            reshape_fns.to_2d(price, raw=True),
            init_capital,
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            nb.signals_order_func_nb,
            reshape_fns.to_2d(entries, raw=True),
            reshape_fns.to_2d(exits, raw=True),
            reshape_fns.to_2d(amount, raw=True))

        # Bring to the same meta
        cash = price.vbt.wrap_array(cash)
        shares = price.vbt.wrap_array(shares)
        paid_fees = price.vbt.wrap_array(paid_fees)
        paid_slippage = price.vbt.wrap_array(paid_slippage)

        return cls(price, cash, shares, init_capital, paid_fees, paid_slippage)

    @classmethod
    def from_orders(cls, price, orders, is_target=False, init_capital=None, fees=None,
                    slippage=None, broadcast_kwargs={}):
        """Build portfolio from orders.

        Starting with initial capital `init_capital`, at each time step, orders the number 
        of shares specified in `orders`. 

        Args:
            price (pandas_like): Price of the asset.
            orders (int, float or array_like): The amount of shares to order. 

                If the amount is positive, this is the number of shares to buy. 
                If the amount is negative, this is the number of shares to sell.
                To buy/sell everything, set the amount to `numpy.inf`.
            is_target (bool): If `True`, will order the difference between current and target amount.
            init_capital (int or float): The initial capital.
            fees (float or array_like): Trading fees in percentage of the value involved.
            slippage (float or array_like): Slippage in percentage of `price`.

        All array-like arguments will be broadcasted together using `vectorbt.utils.reshape_fns.broadcast`
        with `broadcast_kwargs`. At the end, each time series object will have the same metadata.

        Example:
            Portfolio value of various order sequences:
            ```python-repl
            >>> orders = pd.DataFrame({
            ...     'a': [np.inf, 0, 0, 0, 0],
            ...     'b': [1, 1, 1, 1, -np.inf],
            ...     'c': [np.inf, -np.inf, np.inf, -np.inf, np.inf]
            ... }, index=index)

            >>> portfolio = vbt.Portfolio.from_orders(price, orders, 
            ...     init_capital=100, fees=0.0025)

            >>> print(portfolio.cash)
                          a        b           c
            2018-01-01  0.0  98.9975    0.000000
            2018-01-02  0.0  96.9925  199.002494
            2018-01-03  0.0  93.9850    0.000000
            2018-01-04  0.0  91.9800  132.006642
            2018-01-05  0.0  95.9700    0.000000
            >>> print(portfolio.shares)
                                a    b           c
            2018-01-01  99.750623  1.0   99.750623
            2018-01-02  99.750623  2.0    0.000000
            2018-01-03  99.750623  3.0   66.168743
            2018-01-04  99.750623  4.0    0.000000
            2018-01-05  99.750623  0.0  131.677448
            >>> print(portfolio.equity)
                                 a         b           c
            2018-01-01   99.750623   99.9975   99.750623
            2018-01-02  199.501247  100.9925  199.002494
            2018-01-03  299.251870  102.9850  198.506228
            2018-01-04  199.501247   99.9800  132.006642
            2018-01-05   99.750623   95.9700  131.677448
            >>> print(portfolio.total_costs)
            a    0.249377
            b    0.030000
            c    1.904433
            dtype: float64
            ```
        """
        # Get defaults
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        init_capital = float(init_capital)
        if fees is None:
            fees = defaults.portfolio['fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_type(orders, (pd.Series, pd.DataFrame))

        # Broadcast inputs
        price, orders = reshape_fns.broadcast(price, orders, **broadcast_kwargs, writeable=True)
        fees = reshape_fns.broadcast_to(fees, price, to_pd=False, writeable=True)
        slippage = reshape_fns.broadcast_to(slippage, price, to_pd=False, writeable=True)

        # Perform calculation
        cash, shares, paid_fees, paid_slippage = nb.portfolio_nb(
            reshape_fns.to_2d(price, raw=True),
            init_capital,
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            nb.amount_order_func_nb,
            reshape_fns.to_2d(orders, raw=True),
            is_target)

        # Bring to the same meta
        cash = price.vbt.wrap_array(cash)
        shares = price.vbt.wrap_array(shares)
        paid_fees = price.vbt.wrap_array(paid_fees)
        paid_slippage = price.vbt.wrap_array(paid_slippage)

        return cls(price, cash, shares, init_capital, paid_fees, paid_slippage)

    @classmethod
    def from_order_func(cls, price, order_func_nb, *args, init_capital=None, fees=None, slippage=None):
        """Build portfolio from a custom order function.

        Starting with initial capital `init_capital`, at each time step, orders the number 
        of shares returned by `order_func_nb`. 

        Args:
            price (pandas_like): Price of the asset.
            order_func_nb (function): Function that returns the amount of shares to order.

                See `vectorbt.portfolio.nb.portfolio_nb`.
            *args: Arguments passed to `order_func_nb`.
            init_capital (int or float): The initial capital.
            fees (float or array_like): Trading fees in percentage of the value involved.
            slippage (float or array_like): Slippage in percentage of `price`.

        All array-like arguments will be broadcasted together using `vectorbt.utils.reshape_fns.broadcast`
        with `broadcast_kwargs`. At the end, each time series object will have the same metadata.

        !!! note
            `order_func_nb` must be Numba-compiled.

        Example:
            Portfolio value of a simple buy-and-hold strategy:
            ```python-repl
            >>> @njit
            ... def order_func_nb(col, i, run_cash, run_shares):
            ...     return 10 if i == 0 else 0

            >>> portfolio = vbt.Portfolio.from_order_func(price, 
            ...     order_func_nb, init_capital=100, fees=0.0025)

            >>> print(portfolio.cash)
            2018-01-01    89.975
            2018-01-02    89.975
            2018-01-03    89.975
            2018-01-04    89.975
            2018-01-05    89.975
            dtype: float64
            >>> print(portfolio.shares)
            2018-01-01    10.0
            2018-01-02    10.0
            2018-01-03    10.0
            2018-01-04    10.0
            2018-01-05    10.0
            dtype: float64
            >>> print(portfolio.equity)
            2018-01-01     99.975
            2018-01-02    109.975
            2018-01-03    119.975
            2018-01-04    109.975
            2018-01-05     99.975
            dtype: float64
            >>> print(portfolio.total_costs)
            0.02499999999999858
            ```
        """
        # Get defaults
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        init_capital = float(init_capital)
        if fees is None:
            fees = defaults.portfolio['fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_numba_func(order_func_nb)

        # Broadcast inputs
        fees = reshape_fns.broadcast_to(fees, price, to_pd=False, writeable=True)
        slippage = reshape_fns.broadcast_to(slippage, price, to_pd=False, writeable=True)

        # Perform calculation
        cash, shares, paid_fees, paid_slippage = nb.portfolio_nb(
            reshape_fns.to_2d(price, raw=True),
            init_capital,
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            order_func_nb,
            *args)

        # Bring to the same meta
        cash = price.vbt.wrap_array(cash)
        shares = price.vbt.wrap_array(shares)
        paid_fees = price.vbt.wrap_array(paid_fees)
        paid_slippage = price.vbt.wrap_array(paid_slippage)

        return cls(price, cash, shares, init_capital, paid_fees, paid_slippage)

    # ############# Time series ############# #

    @cached_property
    def equity(self):
        """Portfolio value at each time step."""
        equity = self.cash.vbt.to_2d_array() + self.shares.vbt.to_2d_array() * self.price.vbt.to_2d_array()
        return self.wrap_array(equity)

    @cached_property
    def returns(self):
        """Portfolio returns at each time step."""
        returns = timeseries.nb.pct_change_nb(self.equity.vbt.to_2d_array())
        return self.wrap_array(returns)

    @cached_property
    def trades(self):
        """Amount of shares ordered at each time step."""
        shares = self.shares.vbt.to_2d_array()
        trades = timeseries.nb.fillna_nb(timeseries.nb.diff_nb(shares), 0)
        trades[0, :] = shares[0, :]
        return self.wrap_array(trades)

    @cached_property
    def has_trades(self):
        """Whether any trades happened in each column."""
        has_trades = (self.trades.vbt.to_2d_array() > 0).any(axis=0)
        return self.wrap_reduced_array(has_trades)

    def plot_trades(self,
                    buy_trace_kwargs={},
                    sell_trace_kwargs={},
                    fig=None,
                    **layout_kwargs):
        """Plot trades as markers.

        Args:
            buy_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) for "Sell" markers.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            vbt.Portfolio.from_orders(price, price.diff(), init_capital=100).plot_trades()
            ```

            ![](img/portfolio_plot_trades.png)"""
        checks.assert_type(self.price, pd.Series)
        checks.assert_type(self.trades, pd.Series)
        sell_mask = self.trades < 0
        buy_mask = self.trades > 0

        # Plot time series
        fig = self.price.vbt.timeseries.plot(fig=fig, **layout_kwargs)
        # Plot markers
        buy_trace_kwargs = common.merge_kwargs(dict(
            customdata=self.trades[buy_mask],
            hovertemplate='(%{x}, %{y})<br>%{customdata:.6g}',
            marker=dict(
                symbol='triangle-up',
                color='limegreen'
            )
        ), buy_trace_kwargs)
        buy_mask.vbt.signals.plot_markers(
            self.price, name='Buy', trace_kwargs=buy_trace_kwargs, fig=fig, **layout_kwargs)
        sell_trace_kwargs = common.merge_kwargs(dict(
            customdata=self.trades[sell_mask],
            hovertemplate='(%{x}, %{y})<br>%{customdata:.6g}',
            marker=dict(
                symbol='triangle-down',
                color='orangered'
            )
        ), sell_trace_kwargs)
        sell_mask.vbt.signals.plot_markers(
            self.price, name='Sell', trace_kwargs=sell_trace_kwargs, fig=fig, **layout_kwargs)
        return fig

    @cached_property
    def drawdown(self):
        """Relative decline from a peak at each time step."""
        equity = self.equity.vbt.to_2d_array()
        drawdown = 1 - equity / timeseries.nb.expanding_max_nb(equity)
        return self.wrap_array(drawdown)

    # ############# Costs ############# #

    @cached_property
    def total_paid_fees(self):
        """Total paid fees of each column."""
        total_paid_fees = np.sum(self.paid_fees.vbt.to_2d_array(), axis=0)
        return self.wrap_reduced_array(total_paid_fees)

    @cached_property
    def total_paid_slippage(self):
        """Total paid slippage of each column."""
        total_paid_slippage = np.sum(self.paid_slippage.vbt.to_2d_array(), axis=0)
        return self.wrap_reduced_array(total_paid_slippage)

    @cached_property
    def total_costs(self):
        """Total costs of each column."""
        total_paid_fees = reshape_fns.to_1d(self.total_paid_fees, raw=True)
        total_paid_slippage = reshape_fns.to_1d(self.total_paid_slippage, raw=True)
        total_costs = total_paid_fees + total_paid_slippage
        return self.wrap_reduced_array(total_costs)

    # ############# Positions ############# #

    @property
    def positions(self):
        """Open and closed positions of class `Positions`."""
        if not hasattr(self, '_positions'):
            self._positions = Positions(self, pos_status=None)
        return self._positions

    @property
    def open_positions(self):
        """Open positions of class `Positions`."""
        if not hasattr(self, '_open_positions'):
            self._open_positions = Positions(self, pos_status=nb.OPEN)
        return self._open_positions

    @property
    def closed_positions(self):
        """Closed positions of class `Positions`."""
        if not hasattr(self, '_closed_positions'):
            self._closed_positions = Positions(self, pos_status=nb.CLOSED)
        return self._closed_positions

    # ############# Performance ############# #

    @cached_property
    def total_profit(self):
        """Total profit of each column."""
        total_profit = self.equity.vbt.to_2d_array()[-1, :] - self.init_capital
        return self.wrap_reduced_array(total_profit)

    @cached_property
    def total_return(self):
        """Total return of each column."""
        total_return = reshape_fns.to_1d(self.total_profit, raw=True) / self.init_capital
        return self.wrap_reduced_array(total_return)

    @cached_property
    def daily_return(self):
        """Total daily return of each column."""
        return self.returns.vbt.timeseries.resample_apply('D', nb.total_return_apply_func_nb)

    @cached_property
    def annual_return(self):
        """Total annual return of each column."""
        return self.returns.vbt.timeseries.resample_apply('Y', nb.total_return_apply_func_nb)

    @cached_property
    def max_drawdown(self):
        """Total maximum drawdown (MDD) of each column."""
        max_drawdown = np.max(self.drawdown.vbt.to_2d_array(), axis=0)
        return self.wrap_reduced_array(max_drawdown)
