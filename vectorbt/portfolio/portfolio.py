"""Class for building a portfolio and measuring its performance.

The job of the `Portfolio` class is to create a series of positions allocated 
    against a cash component, produce an equity curve, incorporate basic transaction costs 
    and produce a set of statistics about its performance. In particular it outputs 
    position/profit metrics and drawdown information.

Depending upon the class method, it takes some input, and for each column in this input, 
it calculates the portfolio value by tracking the price, the amount of cash held, the 
amount of shares held, but also the costs spent at each time step. It then passes these 
time series to the `__init__` method to create an instance of the `Portfolio` class.

## Properties

The `Portfolio` class offers numerous properties for measuring the performance of a strategy. 
They can be categorized as follows:

* time series indexed by time, such as `Portfolio.equity`
* metrics indexed by columns, such as `Portfolio.total_profit`
* group objects with own time series and metrics, such as `Portfolio.positions`

### Caching

Each property is cached, thus properties can effectively build upon each other. 

Take for example the `Portfolio.max_drawdown` property: it depends upon `Portfolio.drawdown`,
which in turn depends upon `Portfolio.equity`, and so on. Without caching, `Portfolio.max_drawdown` 
would have re-calculated everything starting from equity, each time.

!!! note
    `Portfolio` class is meant to be immutable due to caching, thus each public attribute is
    marked as read-only. To change an attribute, you need to create a new Portfolio instance.

### Property hierarchy

All those properties are building a hierarchy with time series and metrics as leafs, and group 
objects as nodes. By implementing custom cachable property classes `vectorbt.portfolio.properties.timeseries_property` 
and `vectorbt.portfolio.properties.metric_property`, we are also able to encode information into each property, 
such as the full name of a metric and its display format. And by defining the group properties with 
`vectorbt.portfolio.properties.group_property`, we are able to define gateaway points that can be easily traversed.

```plaintext
Portfolio
+-- @timeseries_property
+-- @metric_property
+-- @group_property
    +-- @timeseries_property
    +-- @metric_property
```

This way, the `Portfolio` class acts as an extendable tree data structure for properties with 
annotations. Instead of hard-coding the list of available time series and metrics with something 
like `_PERFORMANCE_METRICS_PROPS`, we can call `vectorbt.portfolio.properties.traverse_timeseries` 
and build the list on the fly.

!!! note
    Hierarchy and annotations are only visible when traversing the class, not the class instance.
    To add a new attribute to the hierarchy, you need to subclass `Portfolio` and define your
    properties there. Each property must be a subclass of `vectorbt.utils.decorators.custom_property`.

## Indexing

In addition, you can use pandas indexing on the `Portfolio` class itself, which forwards
indexing operation to each `__init__` argument with pandas type (see `portfolio_indexing_func`):

```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> index = pd.Index([
...     datetime(2018, 1, 1),
...     datetime(2018, 1, 2),
...     datetime(2018, 1, 3),
...     datetime(2018, 1, 4),
...     datetime(2018, 1, 5)
... ])
>>> price = pd.Series([1, 2, 3, 2, 1], index=index, name='a')

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

The only requirement is that both portfolios must have the same metadata."""

import numpy as np
import pandas as pd
from datetime import timedelta
from scipy import stats

from vectorbt import timeseries, accessors, defaults
from vectorbt.utils import indexing, checks, reshape_fns
from vectorbt.utils.config import merge_kwargs
from vectorbt.utils.decorators import class_or_instancemethod
from vectorbt.portfolio import nb
from vectorbt.portfolio.positions import Positions
from vectorbt.portfolio.common import ArrayWrapper
from vectorbt.portfolio.const import OutputFormat
from vectorbt.portfolio.props import (
    timeseries_property,
    metric_property,
    group_property,
    traverse_properties
)


def portfolio_indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Portfolio`. 

    See `vectorbt.utils.indexing.add_pd_indexing`."""
    factor_returns = obj.factor_returns
    if factor_returns is not None:
        factor_returns = pd_indexing_func(obj.factor_returns)
    return obj.__class__(
        pd_indexing_func(obj.price),
        pd_indexing_func(obj.cash),
        pd_indexing_func(obj.shares),
        obj.init_capital,
        pd_indexing_func(obj.fees_paid),
        pd_indexing_func(obj.slippage_paid),
        data_freq=obj.data_freq,
        year_freq=obj.year_freq,
        risk_free=obj.risk_free,
        required_return=obj.required_return,
        cutoff=obj.cutoff,
        factor_returns=factor_returns
    )


@indexing.add_pd_indexing(portfolio_indexing_func)
class Portfolio(ArrayWrapper):
    """Class for building a portfolio and measuring its performance.

    Args:
        price (pandas_like): Price of the asset.
        cash (pandas_like): Cash held at each time step. Must have the same metadata as `price`.
        shares (pandas_like): Shares held at each time step. Must have the same metadata as `price`.
        init_capital (int or float): The initial capital.
        fees_paid (pandas_like): Fees paid at each time step. Must have the same metadata as `price`.
        slippage_paid (pandas_like): Slippage paid at each time step. Must have the same metadata as `price`.
        data_freq (any): Data frequency in case `price.index` is not datetime-like. Will be passed to `pandas.to_timedelta`.
        year_freq (any): Year frequency. Will be passed to `pandas.to_timedelta`.
        risk_free (float): Constant risk-free return throughout the period.
        required_return (float): Minimum acceptance return of the investor.
        cutoff (float): Decimal representing the percentage cutoff for the bottom percentile of returns.
        factor_returns (pandas_like): Benchmark return to compare returns against. 
                If set, will be broadcasted to the shape of `price`.

    For defaults, see `vectorbt.defaults.portfolio`.

    !!! note
        Portfolio is only built by using class methods with `from_` prefix.
        The `__init__` method is reserved for indexing purposes."""

    def __init__(self, price, cash, shares, init_capital, fees_paid, slippage_paid, data_freq=None,
                 year_freq=None, risk_free=None, required_return=None, cutoff=None, factor_returns=None):
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_same_meta(price, cash)
        checks.assert_same_meta(price, shares)
        checks.assert_same_meta(price, fees_paid)
        checks.assert_same_meta(price, slippage_paid)

        # Time series
        self._price = price
        self._cash = cash
        self._shares = shares
        self._fees_paid = fees_paid
        self._slippage_paid = slippage_paid

        # User-defined parameters
        self._init_capital = init_capital
        if data_freq is None:
            data_freq = price.vbt.timeseries.timedelta
        else:
            data_freq = pd.to_timedelta(data_freq)
        self._data_freq = data_freq
        year_freq = defaults.portfolio['year_freq'] if year_freq is None else year_freq
        year_freq = pd.to_timedelta(year_freq)
        self._year_freq = year_freq
        self._ann_factor = year_freq / data_freq
        self._risk_free = defaults.portfolio['risk_free'] if risk_free is None else risk_free
        self._required_return = defaults.portfolio['required_return'] if required_return is None else required_return
        self._cutoff = defaults.portfolio['cutoff'] if cutoff is None else cutoff
        if factor_returns is not None:
            factor_returns = reshape_fns.broadcast_to(factor_returns, price)
        self._factor_returns = factor_returns

        ArrayWrapper.__init__(self, self.price)

    # ############# Magic methods ############# #

    def __add__(self, other):
        checks.assert_type(other, self.__class__)
        checks.assert_same(self.price, other.price)
        checks.assert_same(self.data_freq, other.data_freq)
        checks.assert_same(self.year_freq, other.year_freq)
        checks.assert_same(self.risk_free, other.risk_free)
        checks.assert_same(self.required_return, other.required_return)
        checks.assert_same(self.cutoff, other.cutoff)
        checks.assert_same(self.factor_returns, other.factor_returns)

        return self.__class__(
            self.price,
            self.cash + other.cash,
            self.shares + other.shares,
            self.init_capital + other.init_capital,
            self.fees_paid + other.fees_paid,
            self.slippage_paid + other.slippage_paid,
            data_freq=self.data_freq,
            year_freq=self.year_freq,
            risk_free=self.risk_free,
            required_return=self.required_return,
            cutoff=self.cutoff,
            factor_returns=self.factor_returns
        )

    def __radd__(self, other):
        return Portfolio.__add__(self, other)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, price, entries, exits, amount=np.inf, init_capital=None,
                     fees=None, slippage=None, broadcast_kwargs={}, **kwargs):
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
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

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
        cash, shares, fees_paid, slippage_paid = nb.portfolio_nb(
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
        fees_paid = price.vbt.wrap_array(fees_paid)
        slippage_paid = price.vbt.wrap_array(slippage_paid)

        return cls(price, cash, shares, init_capital, fees_paid, slippage_paid, **kwargs)

    @classmethod
    def from_orders(cls, price, orders, is_target=False, init_capital=None, fees=None,
                    slippage=None, broadcast_kwargs={}, **kwargs):
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
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

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
        cash, shares, fees_paid, slippage_paid = nb.portfolio_nb(
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
        fees_paid = price.vbt.wrap_array(fees_paid)
        slippage_paid = price.vbt.wrap_array(slippage_paid)

        return cls(price, cash, shares, init_capital, fees_paid, slippage_paid, **kwargs)

    @classmethod
    def from_order_func(cls, price, order_func_nb, *args, init_capital=None, fees=None, slippage=None, **kwargs):
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
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

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
        cash, shares, fees_paid, slippage_paid = nb.portfolio_nb(
            reshape_fns.to_2d(price, raw=True),
            init_capital,
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            order_func_nb,
            *args)

        # Bring to the same meta
        cash = price.vbt.wrap_array(cash)
        shares = price.vbt.wrap_array(shares)
        fees_paid = price.vbt.wrap_array(fees_paid)
        slippage_paid = price.vbt.wrap_array(slippage_paid)

        return cls(price, cash, shares, init_capital, fees_paid, slippage_paid, **kwargs)

    # ############# Built-in time series ############# #

    @timeseries_property('Price', OutputFormat.CURRENCY)
    def price(self):
        """Price per share at each time step."""
        return self._price

    @timeseries_property('Cash', OutputFormat.CURRENCY)
    def cash(self):
        """Cash held at each time step."""
        return self._cash

    @timeseries_property('Shares', OutputFormat.NONE)
    def shares(self):
        """Shares held at each time step."""
        return self._shares

    @timeseries_property('Paid fees', OutputFormat.CURRENCY)
    def fees_paid(self):
        """Paid fees at each time step."""
        return self._fees_paid

    @timeseries_property('Paid slippage', OutputFormat.CURRENCY)
    def slippage_paid(self):
        """Paid slippage at each time step."""
        return self._slippage_paid

    # ############# User-defined parameters ############# #

    @property
    def init_capital(self):
        """Initial capital."""
        return self._init_capital

    @property
    def data_freq(self):
        """Data frequency."""
        return self._data_freq

    @property
    def year_freq(self):
        """Year frequency."""
        return self._year_freq

    @property
    def ann_factor(self):
        """Annualization factor."""
        return self._ann_factor

    @property
    def risk_free(self):
        """Constant risk-free return throughout the period."""
        return self._risk_free

    @property
    def required_return(self):
        """Minimum acceptance return of the investor."""
        return self._required_return

    @property
    def cutoff(self):
        """Decimal representing the percentage cutoff for the bottom percentile of returns."""
        return self._cutoff

    @property
    def factor_returns(self):
        """Benchmark return to compare returns against."""
        return self._factor_returns

    # ############# Positions ############# #

    @group_property('Positions', Positions)
    def positions(self):
        """Positions of the portfolio."""
        return Positions(self)

    # ############# Equity ############# #

    @timeseries_property('Equity', OutputFormat.CURRENCY)
    def equity(self):
        """Portfolio value at each time step."""
        equity = self.cash.vbt.to_2d_array() + self.shares.vbt.to_2d_array() * self.price.vbt.to_2d_array()
        return self.wrap_array(equity)

    @metric_property('Total profit', OutputFormat.CURRENCY)
    def total_profit(self):
        """Total profit."""
        total_profit = self.equity.vbt.to_2d_array()[-1, :] - self.init_capital
        return self.wrap_reduced_array(total_profit)

    # ############# Returns ############# #

    @timeseries_property('Returns', OutputFormat.PERCENT)
    def returns(self):
        """Portfolio returns at each time step."""
        returns = timeseries.nb.pct_change_nb(self.equity.vbt.to_2d_array())
        return self.wrap_array(returns)

    @timeseries_property('Daily returns', OutputFormat.PERCENT)
    def daily_returns(self):
        """Daily returns."""
        if self.returns.index.inferred_freq == 'D':
            return self.returns
        return self.returns.vbt.timeseries.resample_apply('D', nb.total_return_apply_func_nb)

    @timeseries_property('Annual returns', OutputFormat.PERCENT)
    def annual_returns(self):
        """Annual returns."""
        if self.returns.index.inferred_freq == 'Y':
            return self.returns
        return self.returns.vbt.timeseries.resample_apply('Y', nb.total_return_apply_func_nb)

    @metric_property('Total return', OutputFormat.PERCENT)
    def total_return(self):
        """Total return."""
        total_return = reshape_fns.to_1d(self.total_profit, raw=True) / self.init_capital
        return self.wrap_reduced_array(total_return)

    # ############# Trades ############# #

    @timeseries_property('Trades', OutputFormat.NONE)
    def trades(self):
        """Amount of shares ordered at each time step."""
        shares = self.shares.vbt.to_2d_array()
        trades = timeseries.nb.fillna_nb(timeseries.nb.diff_nb(shares), 0)
        trades[0, :] = shares[0, :]
        return self.wrap_array(trades)

    def plot_trades(self,
                    buy_trace_kwargs={},
                    sell_trace_kwargs={},
                    fig=None,
                    **layout_kwargs):
        """Plot trades as markers.

        Args:
            buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Sell" markers.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            vbt.Portfolio.from_orders(price, price.diff(), init_capital=100).plot_trades()
            ```

            ![](/vectorbt/docs/img/portfolio_plot_trades.png)"""
        checks.assert_type(self.price, pd.Series)
        checks.assert_type(self.trades, pd.Series)
        sell_mask = self.trades < 0
        buy_mask = self.trades > 0

        # Plot time series
        fig = self.price.vbt.timeseries.plot(fig=fig, **layout_kwargs)
        # Plot markers
        buy_trace_kwargs = merge_kwargs(dict(
            customdata=self.trades[buy_mask],
            hovertemplate='(%{x}, %{y})<br>%{customdata:.6g}',
            marker=dict(
                symbol='triangle-up',
                color='limegreen'
            )
        ), buy_trace_kwargs)
        buy_mask.vbt.signals.plot_markers(
            self.price, name='Buy', trace_kwargs=buy_trace_kwargs, fig=fig, **layout_kwargs)
        sell_trace_kwargs = merge_kwargs(dict(
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

    @metric_property('Trade count', OutputFormat.NONE)
    def trade_count(self):
        """Number of trades."""
        trade_count = (self.trades.vbt.to_2d_array() > 0).sum(axis=0)
        return self.wrap_reduced_array(trade_count)

    # ############# Drawdown ############# #

    @timeseries_property('Drawdown', OutputFormat.PERCENT)
    def drawdown(self):
        """Relative decline from a peak at each time step."""
        equity = self.equity.vbt.to_2d_array()
        drawdown = 1 - equity / timeseries.nb.expanding_max_nb(equity)
        return self.wrap_array(drawdown)

    @metric_property('Max drawdown', OutputFormat.PERCENT)
    def max_drawdown(self):
        """Total maximum drawdown (MDD)."""
        max_drawdown = np.max(self.drawdown.vbt.to_2d_array(), axis=0)
        return self.wrap_reduced_array(max_drawdown)

    # ############# Costs ############# #

    @metric_property('Total paid fees', OutputFormat.CURRENCY)
    def total_fees_paid(self):
        """Total paid fees."""
        total_fees_paid = np.sum(self.fees_paid.vbt.to_2d_array(), axis=0)
        return self.wrap_reduced_array(total_fees_paid)

    @metric_property('Total paid slippage', OutputFormat.CURRENCY)
    def total_slippage_paid(self):
        """Total paid slippage."""
        total_slippage_paid = np.sum(self.slippage_paid.vbt.to_2d_array(), axis=0)
        return self.wrap_reduced_array(total_slippage_paid)

    @metric_property('Total costs', OutputFormat.CURRENCY)
    def total_costs(self):
        """Total costs."""
        total_fees_paid = reshape_fns.to_1d(self.total_fees_paid, raw=True)
        total_slippage_paid = reshape_fns.to_1d(self.total_slippage_paid, raw=True)
        total_costs = total_fees_paid + total_slippage_paid
        return self.wrap_reduced_array(total_costs)

    # ############# Risk and performance metrics ############# #

    @timeseries_property('Cumulative returns', OutputFormat.PERCENT)
    def cum_returns(self):
        """Cumulative returns at each time step."""
        return self.wrap_array(nb.cum_returns_nb(self.returns.vbt.to_2d_array()))

    @metric_property('Annualized return', OutputFormat.PERCENT)
    def annualized_return(self):
        """Mean annual growth rate of returns. 

        This is equivilent to the compound annual growth rate."""
        return self.wrap_reduced_array(nb.annualized_return_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Annualized volatility', OutputFormat.PERCENT)
    def annualized_volatility(self):
        """Annualized volatility of a strategy."""
        return self.wrap_reduced_array(nb.annualized_volatility_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Calmar ratio', OutputFormat.PERCENT)
    def calmar_ratio(self):
        """Calmar ratio, or drawdown ratio, of a strategy."""
        return self.wrap_reduced_array(nb.calmar_ratio_nb(
            self.returns.vbt.to_2d_array(),
            reshape_fns.to_1d(self.annualized_return, raw=True),
            reshape_fns.to_1d(self.max_drawdown, raw=True),
            self.ann_factor))

    @metric_property('Omega ratio', OutputFormat.PERCENT)
    def omega_ratio(self):
        """Omega ratio of a strategy."""
        return self.wrap_reduced_array(nb.omega_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            risk_free=self.risk_free,
            required_return=self.required_return))

    @metric_property('Sharpe ratio', OutputFormat.PERCENT)
    def sharpe_ratio(self):
        """Sharpe ratio of a strategy."""
        return self.wrap_reduced_array(nb.sharpe_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            risk_free=self.risk_free))

    @metric_property('Downside risk', OutputFormat.PERCENT)
    def downside_risk(self):
        """Downside deviation below a threshold."""
        return self.wrap_reduced_array(nb.downside_risk_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            required_return=self.required_return))

    @metric_property('Sortino ratio', OutputFormat.PERCENT)
    def sortino_ratio(self):
        """Sortino ratio of a strategy."""
        return self.wrap_reduced_array(nb.sortino_ratio_nb(
            self.returns.vbt.to_2d_array(),
            reshape_fns.to_1d(self.downside_risk, raw=True),
            self.ann_factor,
            required_return=self.required_return))

    @metric_property('Information ratio', OutputFormat.PERCENT)
    def information_ratio(self):
        """Information ratio of a strategy.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrap_reduced_array(nb.information_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array()))

    @metric_property('Beta', OutputFormat.PERCENT)
    def beta(self):
        """Beta.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrap_reduced_array(nb.beta_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            risk_free=self.risk_free))

    @metric_property('Annualized alpha', OutputFormat.PERCENT)
    def alpha(self):
        """Annualized alpha.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrap_reduced_array(nb.alpha_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            reshape_fns.to_1d(self.beta, raw=True),
            self.ann_factor,
            risk_free=self.risk_free))

    @metric_property('Tail ratio', OutputFormat.PERCENT)
    def tail_ratio(self):
        """Ratio between the right (95%) and left tail (5%)."""
        return self.wrap_reduced_array(nb.tail_ratio_nb(self.returns.vbt.to_2d_array()))

    @metric_property('Value at risk', OutputFormat.CURRENCY)
    def value_at_risk(self):
        """Value at risk (VaR) of a returns stream."""
        return self.wrap_reduced_array(nb.value_at_risk_nb(
            self.returns.vbt.to_2d_array(),
            cutoff=self.cutoff))

    @metric_property('Conditional value at risk', OutputFormat.CURRENCY)
    def conditional_value_at_risk(self):
        """Conditional value at risk (CVaR) of a returns stream."""
        return self.wrap_reduced_array(nb.conditional_value_at_risk_nb(
            self.returns.vbt.to_2d_array(),
            cutoff=self.cutoff))

    @metric_property('Capture ratio', OutputFormat.PERCENT)
    def capture(self):
        """Capture ratio.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrap_reduced_array(nb.capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Capture ratio (positive)', OutputFormat.PERCENT)
    def up_capture(self):
        """Capture ratio for periods when the benchmark return is positive.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrap_reduced_array(nb.up_capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Capture ratio (negative)', OutputFormat.PERCENT)
    def down_capture(self):
        """Capture ratio for periods when the benchmark return is negative.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrap_reduced_array(nb.down_capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Skewness', OutputFormat.NONE)
    def skew(self):
        """Skewness of returns."""
        return self.wrap_reduced_array(stats.skew(self.returns.vbt.to_2d_array(), axis=0, nan_policy='omit'))

    @metric_property('Kurtosis', OutputFormat.NONE)
    def kurtosis(self):
        """Kurtosis of returns."""
        return self.wrap_reduced_array(stats.kurtosis(self.returns.vbt.to_2d_array(), axis=0, nan_policy='omit'))

    # ############# Properties traversal ############# #

    @class_or_instancemethod
    def traverse_properties(self_or_cls, property_cls):
        """Traverse this class and its group properties for any properties of type `property_cls`."""
        if isinstance(self_or_cls, type):
            return traverse_properties(self_or_cls, property_cls)
        return traverse_properties(self_or_cls.__class__, property_cls)

    @class_or_instancemethod
    def traverse_timeseries(self_or_cls):
        """Traverse this class and its group properties for time series."""
        return self_or_cls.traverse_properties(timeseries_property)

    @class_or_instancemethod
    def traverse_metrics(self_or_cls):
        """Traverse this class and its group properties for metrics."""
        return self_or_cls.traverse_properties(metric_property)

