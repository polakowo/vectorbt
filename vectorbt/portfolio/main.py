"""Main class for modeling portfolio performance.

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

* Time series indexed by time, such as `Portfolio.equity`.
* Metrics indexed by columns, such as `Portfolio.total_profit`.
* Group objects with own time series and metrics, such as `Portfolio.positions`.

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
objects as nodes. By implementing custom cachable property classes `vectorbt.portfolio.common.timeseries_property` 
and `vectorbt.portfolio.common.metric_property`, we are also able to encode information into each property, 
such as the full name of a metric and its display format. And by defining the group properties with 
`vectorbt.portfolio.common.group_property`, we are able to define gateaway points that can be easily traversed.

```plaintext
Portfolio
+-- @timeseries_property
+-- @metric_property
+-- @records_property
+-- ...
+-- @group_property
    +-- @timeseries_property
    +-- @metric_property
    +-- @records_property
    +-- ...
```

This way, the `Portfolio` class acts as an extendable tree data structure for properties with 
annotations. Instead of hard-coding the list of available time series and metrics with something 
like `_PERFORMANCE_METRICS_PROPS`, we can call `Portfolio.traverse_properties` and build the list on the fly.

!!! note
    Hierarchy and annotations are only visible when traversing the class, not the class instance.
    To add a new attribute to the hierarchy, you need to subclass `Portfolio` and define your
    properties there. Each property must be a subclass of `vectorbt.utils.decorators.custom_property`.
    
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
from datetime import timedelta
from scipy import stats

from vectorbt import timeseries, accessors, defaults
from vectorbt.utils import checks, reshape_fns
from vectorbt.utils.config import merge_kwargs
from vectorbt.portfolio import nb
from vectorbt.portfolio.records import Orders, Trades, Positions
from vectorbt.portfolio.enums import OrderRecord, TradeRecord, PositionRecord
from vectorbt.portfolio.common import (
    TSRArrayWrapper,
    timeseries_property,
    metric_property,
    records_property,
    group_property,
    PropertyTraverser
)


class Portfolio(PropertyTraverser):
    """Class for building a portfolio and measuring its performance.

    Args:
        price (pandas_like): Main price of the asset.
        init_capital (int or float): The initial capital.
        order_records (array_like): Records of type `vectorbt.portfolio.enums.OrderRecord`.
        cash (pandas_like): Cash held at each time step.
        shares (pandas_like): Shares held at each time step.
        data_freq (any): Data frequency in case `price.index` is not datetime-like. 

            Will be passed to `pandas.to_timedelta`.
        year_freq (any): Year frequency. Will be passed to `pandas.to_timedelta`.
        risk_free (float): Constant risk-free return throughout the period.
        required_return (float): Minimum acceptance return of the investor.
        cutoff (float): Decimal representing the percentage cutoff for the bottom percentile of returns.
        factor_returns (pandas_like): Benchmark return to compare returns against. 

            If set, will be broadcasted to the shape of `price`.

    For defaults, see `vectorbt.defaults.portfolio`.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

        All array objects must have the same metadata as `price`."""

    def __init__(self, price, init_capital, order_records, cash, shares, data_freq=None,
                 year_freq=None, risk_free=None, required_return=None, cutoff=None, factor_returns=None):
        # Perform checks
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_type(order_records, np.ndarray)
        checks.assert_same_shape(order_records, OrderRecord, axis=(1, 0))
        checks.assert_same_meta(price, cash)
        checks.assert_same_meta(price, shares)

        # Main parameters
        self._price = price
        self._init_capital = init_capital
        self._order_records = order_records
        self._cash = cash
        self._shares = shares

        # Other parameters
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

        # Supercharge
        self.wrapper = TSRArrayWrapper.from_obj(price)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, price, entries, exits, size=np.inf, entry_price=None, exit_price=None, init_capital=None,
                     fees=None, fixed_fees=None, slippage=None, accumulate=False, broadcast_kwargs={}, **kwargs):
        """Build portfolio from entry and exit signals.

        At each entry signal in `entries`, buys `size` of shares for `entry_price` to enter
        a position. At each exit signal in `exits`, sells everything for `exit_price` 
        to exit the position. Accumulation of orders is disabled by default.

        Args:
            price (pandas_like): Main price of the asset, such as close.
            entries (array_like): Boolean array of entry signals.
            exits (array_like): Boolean array of exit signals.
            size (int, float or array_like): The amount of shares to order. 

                To buy/sell everything, set the size to `numpy.inf`.
            entry_price (array_like): Entry price. Defaults to `price`.
            exit_price (array_like): Exit price. Defaults to `price`.
            init_capital (int or float): The initial capital.
            fees (float or array_like): Fees in percentage of the order value.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order.
            slippage (float or array_like): Slippage in percentage of price.
            accumulate (bool): If `accumulate` is `True`, entering the market when already 
                in the market will be allowed to increase a position.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        All array-like arguments will be broadcasted together using `vectorbt.utils.reshape_fns.broadcast` 
        with `broadcast_kwargs`. At the end, all array objects will have the same metadata.

        Example:
            Portfolio from various signal sequences:
            ```python-repl
            >>> entries = pd.DataFrame({
            ...     'a': [True, False, False, False, False],
            ...     'b': [True, False, True, False, True],
            ...     'c': [True, True, True, True, True]
            ... }, index=index)
            >>> exits = pd.DataFrame({
            ...     'a': [False, False, False, False, False],
            ...     'b': [False, True, False, True, False],
            ...     'c': [True, True, True, True, True]
            ... }, index=index)
            >>> portfolio = vbt.Portfolio.from_signals(
            ...     price, entries, exits, size=10,
            ...     init_capital=100, fees=0.0025, fixed_fees=1., slippage=0.001)

            >>> print(portfolio.order_records)
               Column  Index  Size  Price      Fees  Side
            0     0.0    0.0  10.0  1.001  1.025025   0.0
            1     1.0    0.0  10.0  1.001  1.025025   0.0
            2     1.0    1.0  10.0  1.998  1.049950   1.0
            3     1.0    2.0  10.0  3.003  1.075075   0.0
            4     1.0    3.0  10.0  1.998  1.049950   1.0
            5     1.0    4.0  10.0  1.001  1.025025   0.0
            6     2.0    0.0  10.0  1.001  1.025025   0.0
            >>> print(portfolio.shares)
                           a     b     c
            2018-01-01  10.0  10.0  10.0
            2018-01-02  10.0   0.0  10.0
            2018-01-03  10.0  10.0  10.0
            2018-01-04  10.0   0.0  10.0
            2018-01-05  10.0  10.0  10.0
            >>> print(portfolio.cash)
                                a           b          c
            2018-01-01  88.964975   88.964975  88.964975
            2018-01-02  88.964975  107.895025  88.964975
            2018-01-03  88.964975   76.789950  88.964975
            2018-01-04  88.964975   95.720000  88.964975
            2018-01-05  88.964975   84.684975  88.964975
            ```
        """
        # Get defaults
        if entry_price is None:
            entry_price = price
        if exit_price is None:
            exit_price = price
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_dtype(entries, np.bool_)
        checks.assert_dtype(exits, np.bool_)

        # Broadcast inputs
        price, entries, exits, size, entry_price, exit_price, fees, fixed_fees, slippage = \
            reshape_fns.broadcast(price, entries, exits, size, entry_price, exit_price, fees,
                                  fixed_fees, slippage, **broadcast_kwargs, writeable=True)

        # Perform calculation
        order_records, cash, shares = nb.simulate_from_signals_nb(
            reshape_fns.to_2d(price, raw=True).shape,
            init_capital,
            reshape_fns.to_2d(entries, raw=True),
            reshape_fns.to_2d(exits, raw=True),
            reshape_fns.to_2d(size, raw=True),
            reshape_fns.to_2d(entry_price, raw=True),
            reshape_fns.to_2d(exit_price, raw=True),
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(fixed_fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            accumulate)

        # Bring to the same meta
        cash = price.vbt.wrap(cash)
        shares = price.vbt.wrap(shares)

        return cls(price, init_capital, order_records, cash, shares, **kwargs)

    @classmethod
    def from_orders(cls, price, order_size, order_price=None, init_capital=None, fees=None, fixed_fees=None,
                    slippage=None, is_target=False, broadcast_kwargs={}, **kwargs):
        """Build portfolio from orders.

        Starting with initial capital `init_capital`, at each time step, orders the number 
        of shares specified in `order_size` for `order_price`. 

        Args:
            price (pandas_like): Main price of the asset, such as close.
            order_size (int, float or array_like): The amount of shares to order. 

                If the size is positive, this is the number of shares to buy. 
                If the size is negative, this is the number of shares to sell.
                To buy/sell everything, set the size to `numpy.inf`.
            order_price (array_like): Order price. Defaults to `price`.
            init_capital (int or float): The initial capital.
            fees (float or array_like): Fees in percentage of the order value.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order.
            slippage (float or array_like): Slippage in percentage of `order_price`.
            is_target (bool): If `True`, will order the difference between current and target size.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        All array-like arguments will be broadcasted together using `vectorbt.utils.reshape_fns.broadcast` 
        with `broadcast_kwargs`. At the end, all array objects will have the same metadata.

        Example:
            Portfolio from various order sequences:
            ```python-repl
            >>> orders = pd.DataFrame({
            ...     'a': [np.inf, 0, 0, 0, 0],
            ...     'b': [1, 1, 1, 1, -np.inf],
            ...     'c': [np.inf, -np.inf, np.inf, -np.inf, np.inf]
            ... }, index=index)
            >>> portfolio = vbt.Portfolio.from_orders(price, orders, 
            ...     init_capital=100, fees=0.0025, fixed_fees=1., slippage=0.001)

            >>> print(portfolio.order_records)
                Column  Index        Size  Price      Fees  Side
            0      0.0    0.0   98.654463  1.001  1.246883   0.0
            1      1.0    0.0    1.000000  1.001  1.002502   0.0
            2      1.0    1.0    1.000000  2.002  1.005005   0.0
            3      1.0    2.0    1.000000  3.003  1.007507   0.0
            4      1.0    3.0    1.000000  2.002  1.005005   0.0
            5      1.0    4.0    4.000000  0.999  1.009990   1.0
            6      2.0    0.0   98.654463  1.001  1.246883   0.0
            7      2.0    1.0   98.654463  1.998  1.492779   1.0
            8      2.0    2.0   64.646521  3.003  1.485334   0.0
            9      2.0    3.0   64.646521  1.998  1.322909   1.0
            10     2.0    4.0  126.398131  1.001  1.316311   0.0
            >>> print(portfolio.shares)
                                a    b           c
            2018-01-01  98.654463  1.0   98.654463
            2018-01-02  98.654463  2.0    0.000000
            2018-01-03  98.654463  3.0   64.646521
            2018-01-04  98.654463  4.0    0.000000
            2018-01-05  98.654463  0.0  126.398131
            >>> print(portfolio.cash)
                          a          b             c
            2018-01-01  0.0  97.996498  0.000000e+00
            2018-01-02  0.0  94.989493  1.956188e+02
            2018-01-03  0.0  90.978985  2.842171e-14
            2018-01-04  0.0  87.971980  1.278408e+02
            2018-01-05  0.0  90.957990  0.000000e+00
            ```
        """
        # Get defaults
        if order_price is None:
            order_price = price
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        init_capital = float(init_capital)
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(price, (pd.Series, pd.DataFrame))

        # Broadcast inputs
        price, order_size, order_price, fees, fixed_fees, slippage = \
            reshape_fns.broadcast(price, order_size, order_price, fees, fixed_fees,
                                  slippage, **broadcast_kwargs, writeable=True)

        # Perform calculation
        order_records, cash, shares = nb.simulate_from_orders_nb(
            reshape_fns.to_2d(price, raw=True).shape,
            init_capital,
            reshape_fns.to_2d(order_size, raw=True),
            reshape_fns.to_2d(order_price, raw=True),
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(fixed_fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            is_target)

        # Bring to the same meta
        cash = price.vbt.wrap(cash)
        shares = price.vbt.wrap(shares)

        return cls(price, init_capital, order_records, cash, shares, **kwargs)

    @classmethod
    def from_order_func(cls, price, order_func_nb, *args, init_capital=None, **kwargs):
        """Build portfolio from a custom order function.

        Starting with initial capital `init_capital`, iterates over shape `price.shape`, and for 
        each data point, generates an order using `order_func_nb`. This way, you can specify order 
        size, price and transaction costs dynamically (for example, based on the current balance).

        To iterate over a bigger shape than `price`, you should tile/repeat `price` to the desired shape.

        Args:
            price (pandas_like): Main price of the asset, such as close.

                Must be a pandas object.
            order_func_nb (function): Function that returns an order. 

                See `vectorbt.portfolio.enums.Order`.
            *args: Arguments passed to `order_func_nb`.
            init_capital (int or float): The initial capital.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            `order_func_nb` must be Numba-compiled.

        Example:
            Portfolio from buying daily:
            ```python-repl
            >>> from vectorbt.portfolio.nb import Order

            >>> @njit
            ... def order_func_nb(col, i, run_cash, run_shares, price):
            ...     return Order(10, price[i], fees=0.01, fixed_fees=1., slippage=0.01)

            >>> portfolio = vbt.Portfolio.from_order_func(
            ...     price, order_func_nb, price.values, init_capital=100)

            >>> print(portfolio.order_records)
               Column  Index  Size  Price   Fees  Side
            0     0.0    0.0  10.0   1.01  1.101   0.0
            1     0.0    1.0  10.0   2.02  1.202   0.0
            2     0.0    2.0  10.0   3.03  1.303   0.0
            3     0.0    3.0  10.0   2.02  1.202   0.0
            4     0.0    4.0  10.0   1.01  1.101   0.0
            >>> print(portfolio.shares)
            2018-01-01    10.0
            2018-01-02    20.0
            2018-01-03    30.0
            2018-01-04    40.0
            2018-01-05    50.0
            Name: a, dtype: float64
            >>> print(portfolio.cash)
            2018-01-01    88.799
            2018-01-02    67.397
            2018-01-03    35.794
            2018-01-04    14.392
            2018-01-05     3.191
            Name: a, dtype: float64
            ```
        """
        # Get defaults
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        init_capital = float(init_capital)

        # Perform checks
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_numba_func(order_func_nb)

        # Perform calculation
        order_records, cash, shares = nb.simulate_nb(
            reshape_fns.to_2d(price, raw=True).shape,
            init_capital,
            order_func_nb,
            *args)

        # Bring to the same meta
        cash = price.vbt.wrap(cash)
        shares = price.vbt.wrap(shares)

        return cls(price, init_capital, order_records, cash, shares, **kwargs)

    # ############# Passed properties ############# #

    @property
    def init_capital(self):
        """Initial capital."""
        return self._init_capital

    @timeseries_property('Price')
    def price(self):
        """Price per share at each time step."""
        return self._price

    @timeseries_property('Cash')
    def cash(self):
        """Cash held at each time step."""
        return self._cash

    @timeseries_property('Shares')
    def shares(self):
        """Shares held at each time step."""
        return self._shares

    # ############# User-defined parameters ############# #

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

    # ############# Orders ############# #

    @records_property('Order records')
    def order_records(self):
        """Records of type `vectorbt.portfolio.enums.OrderRecord`."""
        return self.wrapper.wrap_records(self._order_records, OrderRecord)

    @group_property('Orders', Orders)
    def orders(self):
        """Time series and metrics based on order records."""
        return Orders(self.wrapper, self._order_records)

    # ############# Trades ############# #

    @records_property('Trade records')
    def trade_records(self):
        """Records of type `vectorbt.portfolio.enums.TradeRecord`."""
        trade_records = nb.trade_records_nb(self.price.vbt.to_2d_array(), self._order_records)
        return self.wrapper.wrap_records(trade_records, TradeRecord)

    @group_property('Trades', Trades)
    def trades(self):
        """Time series and metrics based on trade records."""
        return Trades(self.wrapper, self.trade_records.vbt.to_array())

    # ############# Positions ############# #

    @records_property('Position records')
    def position_records(self):
        """Records of type `vectorbt.portfolio.enums.PositionRecord`."""
        position_records = nb.position_records_nb(self.price.vbt.to_2d_array(), self._order_records)
        return self.wrapper.wrap_records(position_records, PositionRecord)

    @group_property('Positions', Positions)
    def positions(self):
        """Time series and metrics based on position records."""
        return Positions(self.wrapper, self.position_records.vbt.to_array())

    # ############# Equity ############# #

    @timeseries_property('Equity')
    def equity(self):
        """Equity."""
        equity = self.cash.vbt.to_2d_array() + self.shares.vbt.to_2d_array() * self.price.vbt.to_2d_array()
        return self.wrapper.wrap(equity)

    @metric_property('Total profit')
    def total_profit(self):
        """Total profit."""
        total_profit = self.equity.vbt.to_2d_array()[-1, :] - self.init_capital
        return self.wrapper.wrap_reduced(total_profit)

    # ############# Returns ############# #

    @timeseries_property('Returns')
    def returns(self):
        """Portfolio returns."""
        returns = timeseries.nb.pct_change_nb(self.equity.vbt.to_2d_array())
        return self.wrapper.wrap(returns)

    @timeseries_property('Daily returns')
    def daily_returns(self):
        """Daily returns."""
        if self.returns.index.inferred_freq == 'D':
            return self.returns
        return self.returns.vbt.timeseries.resample_apply('D', nb.total_return_apply_func_nb)

    @timeseries_property('Annual returns')
    def annual_returns(self):
        """Annual returns."""
        if self.returns.index.inferred_freq == 'Y':
            return self.returns
        return self.returns.vbt.timeseries.resample_apply('Y', nb.total_return_apply_func_nb)

    @metric_property('Total return')
    def total_return(self):
        """Total return."""
        total_return = reshape_fns.to_1d(self.total_profit, raw=True) / self.init_capital
        return self.wrapper.wrap_reduced(total_return)

    # ############# Drawdown ############# #

    @timeseries_property('Drawdown')
    def drawdown(self):
        """Relative decline from a peak."""
        equity = self.equity.vbt.to_2d_array()
        drawdown = 1 - equity / timeseries.nb.expanding_max_nb(equity)
        return self.wrapper.wrap(drawdown)

    @metric_property('Max drawdown')
    def max_drawdown(self):
        """Total maximum drawdown (MDD)."""
        max_drawdown = np.max(self.drawdown.vbt.to_2d_array(), axis=0)
        return self.wrapper.wrap_reduced(max_drawdown)

    # ############# Risk and performance metrics ############# #

    @timeseries_property('Cumulative returns')
    def cum_returns(self):
        """Cumulative returns."""
        return self.wrapper.wrap(nb.cum_returns_nb(self.returns.vbt.to_2d_array()))

    @metric_property('Annualized return')
    def annualized_return(self):
        """Mean annual growth rate of returns. 

        This is equivilent to the compound annual growth rate."""
        return self.wrapper.wrap_reduced(nb.annualized_return_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Annualized volatility')
    def annualized_volatility(self):
        """Annualized volatility of a strategy."""
        return self.wrapper.wrap_reduced(nb.annualized_volatility_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Calmar ratio')
    def calmar_ratio(self):
        """Calmar ratio, or drawdown ratio, of a strategy."""
        return self.wrapper.wrap_reduced(nb.calmar_ratio_nb(
            self.returns.vbt.to_2d_array(),
            reshape_fns.to_1d(self.annualized_return, raw=True),
            reshape_fns.to_1d(self.max_drawdown, raw=True),
            self.ann_factor))

    @metric_property('Omega ratio')
    def omega_ratio(self):
        """Omega ratio of a strategy."""
        return self.wrapper.wrap_reduced(nb.omega_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            risk_free=self.risk_free,
            required_return=self.required_return))

    @metric_property('Sharpe ratio')
    def sharpe_ratio(self):
        """Sharpe ratio of a strategy."""
        return self.wrapper.wrap_reduced(nb.sharpe_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            risk_free=self.risk_free))

    @metric_property('Downside risk')
    def downside_risk(self):
        """Downside deviation below a threshold."""
        return self.wrapper.wrap_reduced(nb.downside_risk_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            required_return=self.required_return))

    @metric_property('Sortino ratio')
    def sortino_ratio(self):
        """Sortino ratio of a strategy."""
        return self.wrapper.wrap_reduced(nb.sortino_ratio_nb(
            self.returns.vbt.to_2d_array(),
            reshape_fns.to_1d(self.downside_risk, raw=True),
            self.ann_factor,
            required_return=self.required_return))

    @metric_property('Information ratio')
    def information_ratio(self):
        """Information ratio of a strategy.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrapper.wrap_reduced(nb.information_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array()))

    @metric_property('Beta')
    def beta(self):
        """Beta.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrapper.wrap_reduced(nb.beta_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            risk_free=self.risk_free))

    @metric_property('Annualized alpha')
    def alpha(self):
        """Annualized alpha.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrapper.wrap_reduced(nb.alpha_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            reshape_fns.to_1d(self.beta, raw=True),
            self.ann_factor,
            risk_free=self.risk_free))

    @metric_property('Tail ratio')
    def tail_ratio(self):
        """Ratio between the right (95%) and left tail (5%)."""
        return self.wrapper.wrap_reduced(nb.tail_ratio_nb(self.returns.vbt.to_2d_array()))

    @metric_property('Value at risk')
    def value_at_risk(self):
        """Value at risk (VaR) of a returns stream."""
        return self.wrapper.wrap_reduced(nb.value_at_risk_nb(
            self.returns.vbt.to_2d_array(),
            cutoff=self.cutoff))

    @metric_property('Conditional value at risk')
    def conditional_value_at_risk(self):
        """Conditional value at risk (CVaR) of a returns stream."""
        return self.wrapper.wrap_reduced(nb.conditional_value_at_risk_nb(
            self.returns.vbt.to_2d_array(),
            cutoff=self.cutoff))

    @metric_property('Capture ratio')
    def capture(self):
        """Capture ratio.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrapper.wrap_reduced(nb.capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Capture ratio (positive)')
    def up_capture(self):
        """Capture ratio for periods when the benchmark return is positive.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrapper.wrap_reduced(nb.up_capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Capture ratio (negative)')
    def down_capture(self):
        """Capture ratio for periods when the benchmark return is negative.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.wrapper.wrap_reduced(nb.down_capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Skewness')
    def skew(self):
        """Skewness of returns."""
        return self.wrapper.wrap_reduced(stats.skew(self.returns.vbt.to_2d_array(), axis=0, nan_policy='omit'))

    @metric_property('Kurtosis')
    def kurtosis(self):
        """Kurtosis of returns."""
        return self.wrapper.wrap_reduced(stats.kurtosis(self.returns.vbt.to_2d_array(), axis=0, nan_policy='omit'))
