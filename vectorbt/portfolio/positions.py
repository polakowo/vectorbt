"""Class for measuring performance of positions.

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
from inspect import isfunction

from vectorbt import timeseries
from vectorbt.utils import checks, reshape_fns
from vectorbt.portfolio import nb
from vectorbt.utils.common import list_module_keys
from vectorbt.portfolio.common import (
    ArrayWrapper,
    timeseries_property,
    metric_property,
    group_property,
    PositionType,
    OutputFormat
)

class BasePositions(ArrayWrapper):
    """Exposes a range of attributes on top of positions in a `Portfolio` instance.

    This class doesn't hold any data, but creates a read-only view over position data.
    Except that all time series and metric properties are cached.

    Args:
        portfolio (Portfolio): Portfolio instance.
        pos_status (int): Can be any of: 

            * `PositionType.OPEN` for open positions only,
            * `PositionType.CLOSED` for closed positions only, or 
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
        ArrayWrapper.__init__(self, portfolio.price)

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

    @timeseries_property('Status', OutputFormat.NOMINAL)
    def status(self):
        """Position status (open/closed) at the end of each position."""
        return self.apply_mapper(nb.status_map_func_nb)

    @metric_property('Total count', OutputFormat.NONE)
    def count(self):
        """Total position count."""
        return self.status.vbt.timeseries.count()

    # ############# Duration ############# #

    @timeseries_property('Duration', OutputFormat.NONE)
    def duration(self):
        """Position duration at the end of each position."""
        return self.apply_mapper(nb.duration_map_func_nb, self.portfolio.price.shape)

    @metric_property('Average duration', OutputFormat.TIME)
    def avg_duration(self):
        """Average position duration."""
        return self.duration.vbt.timeseries.mean(time_units=True)

    # ############# PnL ############# #

    @timeseries_property('P/L', OutputFormat.CURRENCY)
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
            profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Profit" markers.
            loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Loss" markers.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            portfolio = vbt.Portfolio.from_orders(price, price.diff(), init_capital=100)
            portfolio.positions.plot_pnl()
            ```

            ![](/vectorbt/docs/img/positions_plot_pnl.png)"""
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.pnl.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)

    @metric_property('Total P/L', OutputFormat.CURRENCY)
    def total_pnl(self):
        """Total position P/L."""
        return self.pnl.vbt.timeseries.sum()

    @metric_property('Average P/L', OutputFormat.CURRENCY)
    def avg_pnl(self):
        """Average position P/L."""
        return self.pnl.vbt.timeseries.mean()

    # ############# Returns ############# #

    @timeseries_property('Returns', OutputFormat.PERCENT)
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

    @metric_property('Average return', OutputFormat.PERCENT)
    def avg_return(self):
        """Average position return."""
        return self.returns.vbt.timeseries.mean()


class PnLPositions(BasePositions):
    """Extends `BasePositions` by combining various profit/loss metrics."""

    @group_property('Winning', BasePositions)
    def winning(self):
        """Winning positions of class `BasePositions`."""
        return BasePositions(
            self.portfolio,
            pos_status=self.pos_status,
            pos_filters=[*self.pos_filters, (nb.winning_filter_func_nb, self.pnl.vbt.to_2d_array())])

    @group_property('Losing', BasePositions)
    def losing(self):
        """Losing positions of class `BasePositions`."""
        return BasePositions(
            self.portfolio,
            pos_status=self.pos_status,
            pos_filters=[*self.pos_filters, (nb.losing_filter_func_nb, self.pnl.vbt.to_2d_array())])

    @metric_property('Win rate', OutputFormat.PERCENT)
    def win_rate(self):
        """How many positions are won in each column."""
        winning_count = reshape_fns.to_1d(self.winning.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        win_rate = winning_count / count
        return self.wrap_reduced_array(win_rate)

    @metric_property('Loss rate', OutputFormat.PERCENT)
    def loss_rate(self):
        """How many positions are lost in each column."""
        losing_count = reshape_fns.to_1d(self.losing.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        loss_rate = losing_count / count
        return self.wrap_reduced_array(loss_rate)

    @metric_property('Profit factor', OutputFormat.PERCENT)
    def profit_factor(self):
        """Profit factor."""
        total_win = reshape_fns.to_1d(self.winning.total_pnl, raw=True)
        total_loss = reshape_fns.to_1d(self.losing.total_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_trades = reshape_fns.to_1d(self.portfolio.trade_count, raw=True) > 0
        total_win[np.isnan(total_win) & has_trades] = 0.
        total_loss[np.isnan(total_loss) & has_trades] = 0.

        profit_factor = total_win / np.abs(total_loss)
        return self.wrap_reduced_array(profit_factor)

    @metric_property('Expectancy', OutputFormat.CURRENCY)
    def expectancy(self):
        """Average profitability per trade (APPT)."""
        win_rate = reshape_fns.to_1d(self.win_rate, raw=True)
        loss_rate = reshape_fns.to_1d(self.loss_rate, raw=True)
        avg_win = reshape_fns.to_1d(self.winning.avg_pnl, raw=True)
        avg_loss = reshape_fns.to_1d(self.losing.avg_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_trades = reshape_fns.to_1d(self.portfolio.trade_count, raw=True) > 0
        avg_win[np.isnan(avg_win) & has_trades] = 0.
        avg_loss[np.isnan(avg_loss) & has_trades] = 0.

        expectancy = win_rate * avg_win - loss_rate * np.abs(avg_loss)
        return self.wrap_reduced_array(expectancy)


class Positions(PnLPositions):
    """Extends `PnLPositions` by offering distinction between open and closed positions."""

    def __init__(self, portfolio, pos_filters=[]):
        super().__init__(portfolio, pos_filters=pos_filters)  # No pos_status here

    @group_property('Open', PnLPositions)
    def open(self):
        """Open positions of class `PnLPositions`."""
        return PnLPositions(
            self.portfolio,
            pos_status=PositionType.OPEN,
            pos_filters=self.pos_filters)

    @group_property('Closed', PnLPositions)
    def closed(self):
        """Closed positions of class `PnLPositions`."""
        return PnLPositions(
            self.portfolio,
            pos_status=PositionType.CLOSED,
            pos_filters=self.pos_filters)

    @metric_property('Closed rate', OutputFormat.PERCENT)
    def closed_rate(self):
        """How many positions are closed in each column."""
        closed_count = reshape_fns.to_1d(self.closed.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        closed_rate = closed_count / count
        return self.wrap_reduced_array(closed_rate)

__all__ = list_module_keys(__name__)