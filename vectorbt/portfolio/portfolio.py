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

The only requirement is that both portfolios must have the same metadata.

!!! warning
    If both portfolios have trades with opposite directions at the same time step, it may cause problems. 
    For example, if two trades cancel each other, it will produce a price of `np.inf`."""

import numpy as np
import pandas as pd
from datetime import timedelta
from scipy import stats

from vectorbt import timeseries, accessors, defaults
from vectorbt.utils import indexing, checks, reshape_fns
from vectorbt.utils.config import merge_kwargs
from vectorbt.timeseries.common import TSArrayWrapper
from vectorbt.portfolio import nb
from vectorbt.portfolio.positions import Positions
from vectorbt.portfolio.common import (
    timeseries_property,
    metric_property,
    group_property
)


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Portfolio`. 

    See `vectorbt.utils.indexing.PandasIndexing`."""
    factor_returns = obj.factor_returns
    if factor_returns is not None:
        factor_returns = pd_indexing_func(obj.factor_returns)
    return obj.__class__(
        pd_indexing_func(obj.price),
        obj.init_capital,
        pd_indexing_func(obj.trade_size),
        pd_indexing_func(obj.trade_price),
        pd_indexing_func(obj.trade_fees),
        pd_indexing_func(obj.cash),
        pd_indexing_func(obj.shares),
        data_freq=obj.data_freq,
        year_freq=obj.year_freq,
        risk_free=obj.risk_free,
        required_return=obj.required_return,
        cutoff=obj.cutoff,
        factor_returns=factor_returns
    )


_PandasIndexer = indexing.PandasIndexing(_indexing_func)


class Portfolio(_PandasIndexer):
    """Class for building a portfolio and measuring its performance.

    Args:
        price (pandas_like): Main price of the asset.
        init_capital (int or float): The initial capital.
        trade_size (pandas_like): Trade size at each time step.
        trade_price (pandas_like): Trade price at each time step.
        trade_fees (pandas_like): Trade fees at each time step.
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

    def __init__(self, price, init_capital, trade_size, trade_price, trade_fees, cash, shares, data_freq=None,
                 year_freq=None, risk_free=None, required_return=None, cutoff=None, factor_returns=None):
        checks.assert_type(price, (pd.Series, pd.DataFrame))
        checks.assert_same_meta(price, trade_size)
        checks.assert_same_meta(price, trade_price)
        checks.assert_same_meta(price, trade_fees)
        checks.assert_same_meta(price, cash)
        checks.assert_same_meta(price, shares)

        # Main parameters
        self._price = price
        self._init_capital = init_capital
        self._trade_size = trade_size
        self._trade_price = trade_price
        self._trade_fees = trade_fees
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
        self.ts_wrapper = TSArrayWrapper.from_obj(price)
        _PandasIndexer.__init__(self)

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

        # NOTE: The following approach results in information loss + sizes can cancel each other
        # If you bought 10 for 20$ and sold 10 for 40$, final size will be 0. and price np.inf
        sum_trade_size = self.trade_size + other.trade_size
        avg_trade_price = (self.trade_size * self.trade_price + other.trade_size * other.trade_price) / sum_trade_size

        return self.__class__(
            self.price,
            self.init_capital + other.init_capital,
            sum_trade_size,
            avg_trade_price,
            self.trade_fees + other.trade_fees,
            self.cash + other.cash,
            self.shares + other.shares,
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
            fees (float or array_like): Fees in percentage of the trade value.
            fixed_fees (float or array_like): Fixed amount of fees to pay per trade.
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

            >>> print(portfolio.trade_size)
                           a     b     c
            2018-01-01  10.0  10.0  10.0
            2018-01-02   NaN -10.0   NaN
            2018-01-03   NaN  10.0   NaN
            2018-01-04   NaN -10.0   NaN
            2018-01-05   NaN  10.0   NaN
            >>> print(portfolio.trade_price)
                            a      b      c
            2018-01-01  1.001  1.001  1.001
            2018-01-02    NaN  1.998    NaN
            2018-01-03    NaN  3.003    NaN
            2018-01-04    NaN  1.998    NaN
            2018-01-05    NaN  1.001    NaN
            >>> print(portfolio.trade_fees)
                               a         b         c
            2018-01-01  1.025025  1.025025  1.025025
            2018-01-02       NaN  1.049950       NaN
            2018-01-03       NaN  1.075075       NaN
            2018-01-04       NaN  1.049950       NaN
            2018-01-05       NaN  1.025025       NaN
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
        trade_size, trade_price, trade_fees, cash, shares = nb.simulate_from_signals_nb(
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
        trade_size = price.vbt.wrap(trade_size)
        trade_price = price.vbt.wrap(trade_price)
        trade_fees = price.vbt.wrap(trade_fees)
        cash = price.vbt.wrap(cash)
        shares = price.vbt.wrap(shares)

        return cls(price, init_capital, trade_size, trade_price, trade_fees, cash, shares, **kwargs)

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
            fees (float or array_like): Fees in percentage of the trade value.
            fixed_fees (float or array_like): Fixed amount of fees to pay per trade.
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

            >>> print(portfolio.trade_size)
                                a    b           c
            2018-01-01  98.654463  1.0   98.654463
            2018-01-02        NaN  1.0  -98.654463
            2018-01-03        NaN  1.0   64.646521
            2018-01-04        NaN  1.0  -64.646521
            2018-01-05        NaN -4.0  126.398131
            >>> print(portfolio.trade_price)
                            a      b      c
            2018-01-01  1.001  1.001  1.001
            2018-01-02    NaN  2.002  1.998
            2018-01-03    NaN  3.003  3.003
            2018-01-04    NaN  2.002  1.998
            2018-01-05    NaN  0.999  1.001
            >>> print(portfolio.trade_fees)
                               a         b         c
            2018-01-01  1.246883  1.002502  1.246883
            2018-01-02       NaN  1.005005  1.492779
            2018-01-03       NaN  1.007507  1.485334
            2018-01-04       NaN  1.005005  1.322909
            2018-01-05       NaN  1.009990  1.316311
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
        trade_size, trade_price, trade_fees, cash, shares = nb.simulate_from_orders_nb(
            reshape_fns.to_2d(price, raw=True).shape,
            init_capital,
            reshape_fns.to_2d(order_size, raw=True),
            reshape_fns.to_2d(order_price, raw=True),
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(fixed_fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            is_target)

        # Bring to the same meta
        trade_size = price.vbt.wrap(trade_size)
        trade_price = price.vbt.wrap(trade_price)
        trade_fees = price.vbt.wrap(trade_fees)
        cash = price.vbt.wrap(cash)
        shares = price.vbt.wrap(shares)

        return cls(price, init_capital, trade_size, trade_price, trade_fees, cash, shares, **kwargs)

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

                See `vectorbt.portfolio.nb.Order`.
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

            >>> size = 10
            >>> fees = 0.01
            >>> fixed_fees = 1
            >>> slippage = 0.01

            >>> @njit
            ... def order_func_nb(col, i, run_cash, run_shares, price):
            ...     return Order(size, price[i], fees, fixed_fees, slippage)

            >>> portfolio = vbt.Portfolio.from_order_func(
            ...     price, order_func_nb, price.values)

            >>> print(portfolio.trade_size)
            2018-01-01    10.0
            2018-01-02    10.0
            2018-01-03    10.0
            2018-01-04    10.0
            2018-01-05    10.0
            Name: a, dtype: float64
            >>> print(portfolio.trade_price)
            2018-01-01    1.01
            2018-01-02    2.02
            2018-01-03    3.03
            2018-01-04    2.02
            2018-01-05    1.01
            Name: a, dtype: float64
            >>> print(portfolio.trade_fees)
            2018-01-01    1.101
            2018-01-02    1.202
            2018-01-03    1.303
            2018-01-04    1.202
            2018-01-05    1.101
            Name: a, dtype: float64
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
        trade_size, trade_price, trade_fees, cash, shares = nb.simulate_nb(
            reshape_fns.to_2d(price, raw=True).shape,
            init_capital,
            order_func_nb,
            *args)

        # Bring to the same meta
        trade_size = price.vbt.wrap(trade_size)
        trade_price = price.vbt.wrap(trade_price)
        trade_fees = price.vbt.wrap(trade_fees)
        cash = price.vbt.wrap(cash)
        shares = price.vbt.wrap(shares)

        return cls(price, init_capital, trade_size, trade_price, trade_fees, cash, shares, **kwargs)

    # ############# Passed properties ############# #

    @property
    def init_capital(self):
        """Initial capital."""
        return self._init_capital

    @timeseries_property('Price')
    def price(self):
        """Price per share at each time step."""
        return self._price

    @timeseries_property('Trade size')
    def trade_size(self):
        """Trade size at each time step."""
        return self._trade_size

    @timeseries_property('Trade price')
    def trade_price(self):
        """Trade price at each time step."""
        return self._trade_price

    @timeseries_property('Trade fees')
    def trade_fees(self):
        """Trade fees at each time step."""
        return self._trade_fees

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

    # ############# Trades ############# #

    @timeseries_property('Trades')
    def trades(self):
        """Amount of shares ordered at each time step."""
        shares = self.shares.vbt.to_2d_array()
        trades = timeseries.nb.fillna_nb(timeseries.nb.diff_nb(shares), 0)
        trades[0, :] = shares[0, :]
        return self.ts_wrapper.wrap(trades)

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

    # ############# Positions ############# #

    @group_property('Positions', Positions)
    def positions(self):
        """Positions of the portfolio."""
        return Positions(
            self.ts_wrapper,
            nb.position_records_nb(
                self.price.vbt.to_2d_array(),
                self.trade_size.vbt.to_2d_array(),
                self.trade_price.vbt.to_2d_array(),
                self.trade_fees.vbt.to_2d_array()))

    # ############# Equity ############# #

    @timeseries_property('Equity')
    def equity(self):
        """Portfolio value at each time step."""
        equity = self.cash.vbt.to_2d_array() + self.shares.vbt.to_2d_array() * self.price.vbt.to_2d_array()
        return self.ts_wrapper.wrap(equity)

    @metric_property('Total profit')
    def total_profit(self):
        """Total profit."""
        total_profit = self.equity.vbt.to_2d_array()[-1, :] - self.init_capital
        return self.ts_wrapper.wrap_reduced(total_profit)

    # ############# Returns ############# #

    @timeseries_property('Returns')
    def returns(self):
        """Portfolio returns at each time step."""
        returns = timeseries.nb.pct_change_nb(self.equity.vbt.to_2d_array())
        return self.ts_wrapper.wrap(returns)

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
        return self.ts_wrapper.wrap_reduced(total_return)

    # ############# Drawdown ############# #

    @timeseries_property('Drawdown')
    def drawdown(self):
        """Relative decline from a peak at each time step."""
        equity = self.equity.vbt.to_2d_array()
        drawdown = 1 - equity / timeseries.nb.expanding_max_nb(equity)
        return self.ts_wrapper.wrap(drawdown)

    @metric_property('Max drawdown')
    def max_drawdown(self):
        """Total maximum drawdown (MDD)."""
        max_drawdown = np.max(self.drawdown.vbt.to_2d_array(), axis=0)
        return self.ts_wrapper.wrap_reduced(max_drawdown)

    # ############# Costs ############# #

    @metric_property('Total paid fees')
    def total_fees_paid(self):
        """Total paid fees."""
        total_fees_paid = np.sum(self.fees_paid.vbt.to_2d_array(), axis=0)
        return self.ts_wrapper.wrap_reduced(total_fees_paid)

    @metric_property('Total paid slippage')
    def total_slippage_paid(self):
        """Total paid slippage."""
        total_slippage_paid = np.sum(self.slippage_paid.vbt.to_2d_array(), axis=0)
        return self.ts_wrapper.wrap_reduced(total_slippage_paid)

    @metric_property('Total costs')
    def total_costs(self):
        """Total costs."""
        total_fees_paid = reshape_fns.to_1d(self.total_fees_paid, raw=True)
        total_slippage_paid = reshape_fns.to_1d(self.total_slippage_paid, raw=True)
        total_costs = total_fees_paid + total_slippage_paid
        return self.ts_wrapper.wrap_reduced(total_costs)

    # ############# Risk and performance metrics ############# #

    @timeseries_property('Cumulative returns')
    def cum_returns(self):
        """Cumulative returns at each time step."""
        return self.ts_wrapper.wrap(nb.cum_returns_nb(self.returns.vbt.to_2d_array()))

    @metric_property('Annualized return')
    def annualized_return(self):
        """Mean annual growth rate of returns. 

        This is equivilent to the compound annual growth rate."""
        return self.ts_wrapper.wrap_reduced(nb.annualized_return_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Annualized volatility')
    def annualized_volatility(self):
        """Annualized volatility of a strategy."""
        return self.ts_wrapper.wrap_reduced(nb.annualized_volatility_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Calmar ratio')
    def calmar_ratio(self):
        """Calmar ratio, or drawdown ratio, of a strategy."""
        return self.ts_wrapper.wrap_reduced(nb.calmar_ratio_nb(
            self.returns.vbt.to_2d_array(),
            reshape_fns.to_1d(self.annualized_return, raw=True),
            reshape_fns.to_1d(self.max_drawdown, raw=True),
            self.ann_factor))

    @metric_property('Omega ratio')
    def omega_ratio(self):
        """Omega ratio of a strategy."""
        return self.ts_wrapper.wrap_reduced(nb.omega_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            risk_free=self.risk_free,
            required_return=self.required_return))

    @metric_property('Sharpe ratio')
    def sharpe_ratio(self):
        """Sharpe ratio of a strategy."""
        return self.ts_wrapper.wrap_reduced(nb.sharpe_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            risk_free=self.risk_free))

    @metric_property('Downside risk')
    def downside_risk(self):
        """Downside deviation below a threshold."""
        return self.ts_wrapper.wrap_reduced(nb.downside_risk_nb(
            self.returns.vbt.to_2d_array(),
            self.ann_factor,
            required_return=self.required_return))

    @metric_property('Sortino ratio')
    def sortino_ratio(self):
        """Sortino ratio of a strategy."""
        return self.ts_wrapper.wrap_reduced(nb.sortino_ratio_nb(
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

        return self.ts_wrapper.wrap_reduced(nb.information_ratio_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array()))

    @metric_property('Beta')
    def beta(self):
        """Beta.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.ts_wrapper.wrap_reduced(nb.beta_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            risk_free=self.risk_free))

    @metric_property('Annualized alpha')
    def alpha(self):
        """Annualized alpha.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.ts_wrapper.wrap_reduced(nb.alpha_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            reshape_fns.to_1d(self.beta, raw=True),
            self.ann_factor,
            risk_free=self.risk_free))

    @metric_property('Tail ratio')
    def tail_ratio(self):
        """Ratio between the right (95%) and left tail (5%)."""
        return self.ts_wrapper.wrap_reduced(nb.tail_ratio_nb(self.returns.vbt.to_2d_array()))

    @metric_property('Value at risk')
    def value_at_risk(self):
        """Value at risk (VaR) of a returns stream."""
        return self.ts_wrapper.wrap_reduced(nb.value_at_risk_nb(
            self.returns.vbt.to_2d_array(),
            cutoff=self.cutoff))

    @metric_property('Conditional value at risk')
    def conditional_value_at_risk(self):
        """Conditional value at risk (CVaR) of a returns stream."""
        return self.ts_wrapper.wrap_reduced(nb.conditional_value_at_risk_nb(
            self.returns.vbt.to_2d_array(),
            cutoff=self.cutoff))

    @metric_property('Capture ratio')
    def capture(self):
        """Capture ratio.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.ts_wrapper.wrap_reduced(nb.capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Capture ratio (positive)')
    def up_capture(self):
        """Capture ratio for periods when the benchmark return is positive.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.ts_wrapper.wrap_reduced(nb.up_capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Capture ratio (negative)')
    def down_capture(self):
        """Capture ratio for periods when the benchmark return is negative.

        !!! note
            `factor_returns` must be set."""
        checks.assert_not_none(self.factor_returns)

        return self.ts_wrapper.wrap_reduced(nb.down_capture_nb(
            self.returns.vbt.to_2d_array(),
            self.factor_returns.vbt.to_2d_array(),
            self.ann_factor))

    @metric_property('Skewness')
    def skew(self):
        """Skewness of returns."""
        return self.ts_wrapper.wrap_reduced(stats.skew(self.returns.vbt.to_2d_array(), axis=0, nan_policy='omit'))

    @metric_property('Kurtosis')
    def kurtosis(self):
        """Kurtosis of returns."""
        return self.ts_wrapper.wrap_reduced(stats.kurtosis(self.returns.vbt.to_2d_array(), axis=0, nan_policy='omit'))
