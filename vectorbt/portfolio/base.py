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

>>> index = pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5)
... ])
>>> price = pd.Series([1, 2, 3, 2, 1], index=index, name='a')
>>> orders = pd.DataFrame({
...     'a': [np.inf, 0, 0, 0, 0],
...     'b': [1, 1, 1, 1, -np.inf],
...     'c': [np.inf, -np.inf, np.inf, -np.inf, np.inf]
... }, index=index)
>>> portfolio = vbt.Portfolio.from_orders(price, orders, init_capital=100)

>>> print(portfolio.equity)
                a      b           c
2020-01-01  100.0  100.0  100.000000
2020-01-02  200.0  101.0  200.000000
2020-01-03  300.0  103.0  200.000000
2020-01-04  200.0  100.0  133.333333
2020-01-05  100.0   96.0  133.333333

>>> print(portfolio['a'].equity)
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
from vectorbt.utils.decorators import cached_property
from vectorbt.base import reshape_fns
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic import nb as generic_nb
from vectorbt.portfolio import nb
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

    # Array-like params
    def index_arraylike_param(param):
        if np.asarray(param).ndim > 0:
            param = reshape_fns.broadcast_to_axis_of(param, obj.main_price, 1)
            param = param[new_cols]
        return param

    factor_returns = obj.factor_returns
    if factor_returns is not None:
        if checks.is_frame(factor_returns):
            factor_returns = reshape_fns.broadcast_to(factor_returns, obj.main_price)
            factor_returns = pd_indexing_func(factor_returns)

    # Create new Portfolio instance
    return obj.__class__(
        pd_indexing_func(obj.main_price),
        obj.init_capital.iloc[new_cols],
        pd_indexing_func(obj.orders),  # Orders class supports indexing
        pd_indexing_func(obj.cash),
        pd_indexing_func(obj.shares),
        freq=obj.freq,
        year_freq=obj.year_freq,
        levy_alpha=index_arraylike_param(obj.levy_alpha),
        risk_free=index_arraylike_param(obj.risk_free),
        required_return=index_arraylike_param(obj.required_return),
        cutoff=index_arraylike_param(obj.cutoff),
        factor_returns=factor_returns
    )


class Portfolio(PandasIndexer):
    """Class for modeling portfolio and measuring its performance.

    Args:
        main_price (pandas_like): Main price of the asset.
        init_capital (int, float or pd.Series): The initial capital.

            If `pd.Series`, must have the same index as columns in `main_price`.
        orders (vectorbt.records.orders.Orders): Order records.
        cash (pandas_like): Cash held at each time step.

            Must have the same metadata as `main_price`.
        shares (pandas_like): Shares held at each time step.

            Must have the same metadata as `main_price`.
        freq (any): Index frequency in case `main_price.index` is not datetime-like.
        year_freq (any): Year frequency for working with returns.
        levy_alpha (float or array_like): Scaling relation (Levy stability exponent).

            Single value or value per column.
        risk_free (float or array_like): Constant risk-free return throughout the period.

            Single value or value per column.
        required_return (float or array_like): Minimum acceptance return of the investor.

            Single value or value per column.
        cutoff (float or array_like): Decimal representing the percentage cutoff for the
            bottom percentile of returns.

            Single value or value per column.
        factor_returns (array_like): Benchmark return to compare returns against. Will broadcast.

            By default it's `None`, but it's required by some return-based metrics.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes.

        All array objects must have the same metadata as `main_price`."""

    def __init__(self, main_price, init_capital, orders, cash, shares, freq=None, year_freq=None,
                 levy_alpha=None, risk_free=None, required_return=None, cutoff=None,
                 factor_returns=None):
        # Perform checks
        checks.assert_type(main_price, (pd.Series, pd.DataFrame))
        if checks.is_frame(main_price):
            checks.assert_type(init_capital, pd.Series)
            checks.assert_same(main_price.columns, init_capital.index)
        else:
            checks.assert_ndim(init_capital, 0)
        checks.assert_same_meta(main_price, cash)
        checks.assert_same_meta(main_price, shares)

        # Store passed arguments
        self._main_price = main_price
        self._init_capital = init_capital
        self._orders = orders
        self._cash = cash
        self._shares = shares

        freq = main_price.vbt(freq=freq).freq
        if freq is None:
            raise ValueError("Couldn't parse the frequency of index. You must set `freq`.")
        self._freq = freq

        year_freq = main_price.vbt.returns(year_freq=year_freq).year_freq
        if freq is None:
            raise ValueError("You must set `year_freq`.")
        self._year_freq = year_freq

        # Parameters
        self._levy_alpha = defaults.portfolio['levy_alpha'] if levy_alpha is None else levy_alpha
        self._risk_free = defaults.portfolio['risk_free'] if risk_free is None else risk_free
        self._required_return = defaults.portfolio['required_return'] if required_return is None else required_return
        self._cutoff = defaults.portfolio['cutoff'] if cutoff is None else cutoff
        self._factor_returns = defaults.portfolio['factor_returns'] if factor_returns is None else factor_returns

        # Supercharge
        PandasIndexer.__init__(self, _indexing_func)
        self.wrapper = ArrayWrapper.from_obj(main_price, freq=freq)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, main_price, entries, exits, size=np.inf, entry_price=None, exit_price=None,
                     init_capital=None, fees=None, fixed_fees=None, slippage=None, accumulate=False,
                     broadcast_kwargs={}, freq=None, **kwargs):
        """Build portfolio from entry and exit signals.

        At each entry signal in `entries`, buys `size` of shares for `entry_price` to enter
        a position. At each exit signal in `exits`, sells everything for `exit_price`
        to exit the position. Accumulation of orders is disabled by default.

        Args:
            main_price (pandas_like): Main price of the asset, such as close.
            entries (array_like): Boolean array of entry signals.
            exits (array_like): Boolean array of exit signals.
            size (int, float or array_like): The amount of shares to order.

                To buy/sell everything, set the size to `np.inf`.
            entry_price (array_like): Entry price. Defaults to `main_price`.
            exit_price (array_like): Exit price. Defaults to `main_price`.
            init_capital (int, float or array_like): The initial capital.

                Single value or value per column.
            fees (float or array_like): Fees in percentage of the order value.

                Single value, value per column, or value per element.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order.

                Single value, value per column, or value per element.
            slippage (float or array_like): Slippage in percentage of price.

                Single value, value per column, or value per element.
            accumulate (bool): If `accumulate` is `True`, entering the market when already
                in the market will be allowed to increase a position.
            broadcast_kwargs: Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            freq (any): Index frequency in case `main_price.index` is not datetime-like.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        All time series will be broadcasted together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

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

            >>> print(portfolio.orders.records)
               col  idx  size  price      fees  side
            0    0    0  10.0  1.001  1.025025     0
            1    1    0  10.0  1.001  1.025025     0
            2    1    1  10.0  1.998  1.049950     1
            3    1    2  10.0  3.003  1.075075     0
            4    1    3  10.0  1.998  1.049950     1
            5    1    4  10.0  1.001  1.025025     0
            6    2    0  10.0  1.001  1.025025     0
            >>> print(portfolio.equity)
                                 a           b           c
            2020-01-01   98.964975   98.964975   98.964975
            2020-01-02  108.964975  107.895025  108.964975
            2020-01-03  118.964975  106.789950  118.964975
            2020-01-04  108.964975   95.720000  108.964975
            2020-01-05   98.964975   94.684975   98.964975
            ```
        """
        # Get defaults
        if entry_price is None:
            entry_price = main_price
        if exit_price is None:
            exit_price = main_price
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(main_price, (pd.Series, pd.DataFrame))
        checks.assert_dtype(entries, np.bool_)
        checks.assert_dtype(exits, np.bool_)

        # Broadcast inputs
        main_price, entries, exits, size, entry_price, exit_price, fees, fixed_fees, slippage = \
            reshape_fns.broadcast(
                main_price, entries, exits, size, entry_price, exit_price, fees,
                fixed_fees, slippage, **broadcast_kwargs, writeable=True)
        target_shape = (main_price.shape[0], main_price.shape[1] if main_price.ndim > 1 else 1)
        init_capital = np.broadcast_to(init_capital, (target_shape[1],))

        # Perform calculation
        order_records, cash, shares = nb.simulate_from_signals_nb(
            target_shape,
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
        wrapper = ArrayWrapper.from_obj(main_price, freq=freq)
        cash = wrapper.wrap(cash)
        shares = wrapper.wrap(shares)
        orders = Orders(order_records, main_price, freq=freq)
        if checks.is_series(main_price):
            init_capital = init_capital[0]
        else:
            init_capital = wrapper.wrap_reduced(init_capital)

        return cls(main_price, init_capital, orders, cash, shares, freq=freq, **kwargs)

    @classmethod
    def from_orders(cls, main_price, order_size, order_price=None, init_capital=None, fees=None, fixed_fees=None,
                    slippage=None, is_target=False, broadcast_kwargs={}, freq=None, **kwargs):
        """Build portfolio from orders.

        Starting with initial capital `init_capital`, at each time step, orders the number
        of shares specified in `order_size` for `order_price`.

        Args:
            main_price (pandas_like): Main price of the asset, such as close.
            order_size (int, float or array_like): The amount of shares to order.

                If the size is positive, this is the number of shares to buy.
                If the size is negative, this is the number of shares to sell.
                To buy/sell everything, set the size to `np.inf`.
            order_price (array_like): Order price. Defaults to `main_price`.
            init_capital (int, float or array_like): The initial capital.

                Single value or value per column.
            fees (float or array_like): Fees in percentage of the order value.

                Single value, value per column, or value per element.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order.

                Single value, value per column, or value per element.
            slippage (float or array_like): Slippage in percentage of price.

                Single value, value per column, or value per element.
            is_target (bool): If `True`, will order the difference between current and target size.
            broadcast_kwargs: Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            freq (any): Index frequency in case `main_price.index` is not datetime-like.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        All time series will be broadcasted together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

        Example:
            Portfolio from various order sequences:
            ```python-repl
            >>> portfolio = vbt.Portfolio.from_orders(price, orders,
            ...     init_capital=100, fees=0.0025, fixed_fees=1., slippage=0.001)

            >>> print(portfolio.orders.records)
                col  idx        size  price      fees  side
            0     0    0   98.654463  1.001  1.246883     0
            1     1    0    1.000000  1.001  1.002502     0
            2     1    1    1.000000  2.002  1.005005     0
            3     1    2    1.000000  3.003  1.007507     0
            4     1    3    1.000000  2.002  1.005005     0
            5     1    4    4.000000  0.999  1.009990     1
            6     2    0   98.654463  1.001  1.246883     0
            7     2    1   98.654463  1.998  1.492779     1
            8     2    2   64.646521  3.003  1.485334     0
            9     2    3   64.646521  1.998  1.322909     1
            10    2    4  126.398131  1.001  1.316311     0
            >>> print(portfolio.equity)
                                 a          b           c
            2020-01-01   98.654463  98.996498   98.654463
            2020-01-02  197.308925  98.989493  195.618838
            2020-01-03  295.963388  99.978985  193.939564
            2020-01-04  197.308925  95.971980  127.840840
            2020-01-05   98.654463  90.957990  126.398131
            ```
        """
        # Get defaults
        if order_price is None:
            order_price = main_price
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(main_price, (pd.Series, pd.DataFrame))

        # Broadcast inputs
        main_price, order_size, order_price, fees, fixed_fees, slippage = \
            reshape_fns.broadcast(main_price, order_size, order_price, fees, fixed_fees,
                                  slippage, **broadcast_kwargs, writeable=True)
        target_shape = (main_price.shape[0], main_price.shape[1] if main_price.ndim > 1 else 1)
        init_capital = np.broadcast_to(init_capital, (target_shape[1],))

        # Perform calculation
        order_records, cash, shares = nb.simulate_from_orders_nb(
            target_shape,
            init_capital,
            reshape_fns.to_2d(order_size, raw=True),
            reshape_fns.to_2d(order_price, raw=True),
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(fixed_fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            is_target)

        # Bring to the same meta
        wrapper = ArrayWrapper.from_obj(main_price, freq=freq)
        cash = wrapper.wrap(cash)
        shares = wrapper.wrap(shares)
        orders = Orders(order_records, main_price, freq=freq)
        if checks.is_series(main_price):
            init_capital = init_capital[0]
        else:
            init_capital = wrapper.wrap_reduced(init_capital)

        return cls(main_price, init_capital, orders, cash, shares, freq=freq, **kwargs)

    @classmethod
    def from_order_func(cls, main_price, order_func_nb, *args, init_capital=None, freq=None, **kwargs):
        """Build portfolio from a custom order function.

        Starting with initial capital `init_capital`, iterates over shape `main_price.shape`, and for
        each data point, generates an order using `order_func_nb`. This way, you can specify order
        size, price and transaction costs dynamically (for example, based on the current balance).

        To iterate over a bigger shape than `main_price`, you should tile/repeat `main_price` to the desired shape.

        Args:
            main_price (pandas_like): Main price of the asset, such as close.

                Must be a pandas object.
            order_func_nb (function): Function that returns an order.

                See `vectorbt.portfolio.enums.Order`.
            *args: Arguments passed to `order_func_nb`.
            init_capital (int, float or array_like): The initial capital.

                Single value or value per column.
            freq (any): Index frequency in case `main_price.index` is not datetime-like.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        All time series will be broadcasted together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

        !!! note
            `order_func_nb` must be Numba-compiled.

        Example:
            Portfolio from buying daily:
            ```python-repl
            >>> from vectorbt.portfolio import Order

            >>> @njit
            ... def order_func_nb(col, i, run_cash, run_shares, price):
            ...     return Order(10, price[i], fees=0.01, fixed_fees=1., slippage=0.01)

            >>> portfolio = vbt.Portfolio.from_order_func(
            ...     price, order_func_nb, price.values, init_capital=100)

            >>> print(portfolio.orders.records)
               col  idx  size  price   fees  side
            0    0    0  10.0   1.01  1.101     0
            1    0    1  10.0   2.02  1.202     0
            2    0    2  10.0   3.03  1.303     0
            3    0    3  10.0   2.02  1.202     0
            4    0    4  10.0   1.01  1.101     0
            >>> print(portfolio.equity)
            2020-01-01     98.799
            2020-01-02    107.397
            2020-01-03    125.794
            2020-01-04     94.392
            2020-01-05     53.191
            Name: a, dtype: float64
            ```
        """
        # Get defaults
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']

        # Perform checks
        checks.assert_type(main_price, (pd.Series, pd.DataFrame))
        checks.assert_numba_func(order_func_nb)

        # Broadcast inputs
        target_shape = (main_price.shape[0], main_price.shape[1] if main_price.ndim > 1 else 1)
        init_capital = np.broadcast_to(init_capital, (target_shape[1],))

        # Perform calculation
        order_records, cash, shares = nb.simulate_nb(
            target_shape,
            init_capital,
            order_func_nb,
            *args)

        # Bring to the same meta
        wrapper = ArrayWrapper.from_obj(main_price, freq=freq)
        cash = wrapper.wrap(cash)
        shares = wrapper.wrap(shares)
        orders = Orders(order_records, main_price, freq=freq)
        if checks.is_series(main_price):
            init_capital = init_capital[0]
        else:
            init_capital = wrapper.wrap_reduced(init_capital)

        return cls(main_price, init_capital, orders, cash, shares, freq=freq, **kwargs)

    # ############# Passed arguments ############# #

    @property
    def init_capital(self):
        """Initial capital."""
        return self._init_capital

    @cached_property
    def main_price(self):
        """Price per share series."""
        return self._main_price

    @cached_property
    def cash(self):
        """Cash series."""
        return self._cash

    @cached_property
    def shares(self):
        """Shares series."""
        return self._shares

    @property
    def freq(self):
        """Index frequency."""
        return self._freq

    @property
    def year_freq(self):
        """Year frequency."""
        return self._year_freq

    @property
    def levy_alpha(self):
        """Scaling relation (Levy stability exponent)."""
        return self._levy_alpha

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

    # ############# Records ############# #

    @cached_property
    def orders(self):
        """Order records.

        See `vectorbt.records.orders.Orders`."""
        return self._orders

    @cached_property
    def trades(self):
        """Trade records.

        See `vectorbt.records.events.Trades`."""
        return Trades.from_orders(self.orders)

    @cached_property
    def positions(self):
        """Position records.

        See `vectorbt.records.events.Positions`."""
        return Positions.from_orders(self.orders)

    @cached_property
    def drawdowns(self):
        """Drawdown records.

        See `vectorbt.records.drawdowns.Drawdowns`."""
        return Drawdowns.from_ts(self.equity, freq=self.freq)

    # ############# Equity ############# #

    @cached_property
    def equity(self):
        """Portfolio equity series."""
        return self.cash.vbt + self.shares.vbt * self.main_price.vbt

    @cached_property
    def final_equity(self):
        """Final equity."""
        return self.wrapper.wrap_reduced(self.equity.values[-1])

    @cached_property
    def total_profit(self):
        """Total profit."""
        equity = self.equity.vbt.to_2d_array()[-1, :]
        init_capital = reshape_fns.to_1d(self.init_capital, raw=True)
        return self.wrapper.wrap_reduced(equity - init_capital)

    # ############# Drawdown ############# #

    @cached_property
    def drawdown(self):
        """Drawdown series."""
        equity = self.equity.vbt.to_2d_array()
        return self.wrapper.wrap(equity / generic_nb.expanding_max_nb(equity) - 1)

    @cached_property
    def max_drawdown(self):
        """Max drawdown."""
        return self.drawdown.vbt.min()

    # ############# Returns ############# #

    @cached_property
    def buy_and_hold_return(self):
        """Total return of buying and holding.

        !!! note:
            Does not take into account fees and slippage. For this, create a separate portfolio."""
        returns = generic_nb.pct_change_nb(self.main_price.vbt.to_2d_array())
        return self.wrapper.wrap(returns).vbt.returns.total()

    @cached_property
    def returns(self):
        """Portfolio return series."""
        equity = self.equity.vbt.to_2d_array()
        returns = generic_nb.pct_change_nb(equity)
        init_capital = reshape_fns.to_1d(self.init_capital, raw=True)
        returns[0, :] = (equity[0, :] - init_capital) / init_capital
        return self.wrapper.wrap(returns)

    @cached_property
    def daily_returns(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.daily`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .daily()

    @cached_property
    def annual_returns(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.annual`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .annual()

    @cached_property
    def cumulative_returns(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.cumulative`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .cumulative()

    @cached_property
    def total_return(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.total`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .total()

    @cached_property
    def annualized_return(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.annualized_return`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .annualized_return()

    @cached_property
    def annualized_volatility(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.annualized_volatility`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .annualized_volatility(levy_alpha=self.levy_alpha)

    @cached_property
    def calmar_ratio(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.calmar_ratio`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .calmar_ratio()

    @cached_property
    def omega_ratio(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.omega_ratio`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .omega_ratio(risk_free=self.risk_free, required_return=self.required_return)

    @cached_property
    def sharpe_ratio(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.sharpe_ratio`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .sharpe_ratio(risk_free=self.risk_free)

    @cached_property
    def downside_risk(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.downside_risk`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .downside_risk(required_return=self.required_return)

    @cached_property
    def sortino_ratio(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.sortino_ratio`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .sortino_ratio(required_return=self.required_return)

    @cached_property
    def information_ratio(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.information_ratio`."""
        if self.factor_returns is None:
            raise ValueError("This property requires factor_returns to be set")
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .information_ratio(self.factor_returns)

    @cached_property
    def beta(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.beta`."""
        if self.factor_returns is None:
            raise ValueError("This property requires factor_returns to be set")
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .beta(self.factor_returns)

    @cached_property
    def alpha(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.alpha`."""
        if self.factor_returns is None:
            raise ValueError("This property requires factor_returns to be set")
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .alpha(self.factor_returns, risk_free=self.risk_free)

    @cached_property
    def tail_ratio(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.tail_ratio`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .tail_ratio()

    @cached_property
    def value_at_risk(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.value_at_risk`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .value_at_risk(cutoff=self.cutoff)

    @cached_property
    def conditional_value_at_risk(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.conditional_value_at_risk`."""
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .conditional_value_at_risk(cutoff=self.cutoff)

    @cached_property
    def capture(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.capture`."""
        if self.factor_returns is None:
            raise ValueError("This property requires factor_returns to be set")
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .capture(self.factor_returns)

    @cached_property
    def up_capture(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.up_capture`."""
        if self.factor_returns is None:
            raise ValueError("This property requires factor_returns to be set")
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .up_capture(self.factor_returns)

    @cached_property
    def down_capture(self):
        """See `vectorbt.returns.accessors.Returns_Accessor.down_capture`."""
        if self.factor_returns is None:
            raise ValueError("This property requires factor_returns to be set")
        return self.returns.vbt.returns(freq=self.freq, year_freq=self.year_freq) \
            .down_capture(self.factor_returns)

    # ############# Stats ############# #

    @cached_property
    def stats(self):
        """Compute various interesting statistics on this portfolio."""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        return pd.Series({
            'Start': self.wrapper.index[0],
            'End': self.wrapper.index[-1],
            'Duration': self.wrapper.shape[0] * self.freq,
            'Holding Duration [%]': self.positions.coverage * 100,
            'Total Profit': self.total_profit,
            'Total Return [%]': self.total_return * 100,
            'Buy & Hold Return [%]': self.buy_and_hold_return * 100,
            'Max. Drawdown [%]': -self.max_drawdown * 100,
            'Avg. Drawdown [%]': -self.drawdowns.avg_drawdown * 100,
            'Max. Drawdown Duration': self.drawdowns.max_duration,
            'Avg. Drawdown Duration': self.drawdowns.avg_duration,
            'Num. Trades': self.trades.count,
            'Win Rate [%]': self.trades.win_rate * 100,
            'Best Trade [%]': self.trades.returns.max() * 100,
            'Worst Trade [%]': self.trades.returns.min() * 100,
            'Avg. Trade [%]': self.trades.returns.mean() * 100,
            'Max. Trade Duration': self.trades.duration.max(time_units=True),
            'Avg. Trade Duration': self.trades.duration.mean(time_units=True),
            'Expectancy': self.trades.expectancy,
            'SQN': self.trades.sqn,
            'Sharpe Ratio': self.sharpe_ratio,
            'Sortino Ratio': self.sortino_ratio,
            'Calmar Ratio': self.calmar_ratio
        }, name=self.wrapper.name)
