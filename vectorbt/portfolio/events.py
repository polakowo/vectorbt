"""Classes for measuring performance of any events."""

import numpy as np
import pandas as pd

from vectorbt import timeseries
from vectorbt.utils import checks, reshape_fns
from vectorbt.portfolio import nb
from vectorbt.portfolio.common import (
    ArrayWrapper,
    timeseries_property,
    metric_property,
    group_property
)


class EventMapper(ArrayWrapper):
    """Exposes a `EventMapper.map` method with a built-in filter to map events.

    Standardizes processing of the events such as trades and positions.

    Args:
        ref_obj (pandas_like): Any pandas-like object used for wrapping the result.
        event_mapper (tuple): Mapper function and unpacked arguments, to map events.

            For each event, mapper function must call a function `map_func_nb` and pass to it
            a named tuple of type `vectorbt.portfolio.enums.Event` or any of its subtypes.
            !!! note
                Function must be Numba-compiled.
        filter (tuple): Filter function and unpacked arguments, to filter map results.

            See `vectorbt.timeseries.nb.filter_nb`.

            !!! note
                Function must be Numba-compiled."""

    def __init__(self, ref_obj, event_mapper, filter=None):
        self.event_mapper = event_mapper
        self.filter = filter

        ArrayWrapper.__init__(self, ref_obj)

    def _filter(self, map_results):
        """Apply the built-in filter on map results.

        `map_results` must be a pandas object."""
        if self.filter is not None:
            checks.assert_numba_func(self.filter[0])
            return map_results.vbt.timeseries.filter(*self.filter)
        return map_results

    def map(self, map_func_nb, *args):
        """Apply `map_func_nb` on each series of events using `event_mapper` and filter the results using `filter`.

        This way, all time series will be automatically filtered."""
        checks.assert_numba_func(self.event_mapper[0])
        checks.assert_numba_func(map_func_nb)

        # Apply map
        result = self.event_mapper[0](
            *self.event_mapper[1:],
            map_func_nb,
            *args)
        return self._filter(self.wrap_timeseries(result))


class BaseEvents(EventMapper):
    """Exposes a range of attributes related to events.

    This class doesn't hold any data, but creates a read-only, filtered view over event data.
    Except that all time series and metric properties are cached.

    For arguments, see `EventMapper`. Set `use_cached` to an instance of `EventMapper` to use 
    its cache of time series properties."""

    def __init__(self, ref_obj, event_mapper, filter=None, use_cached=None):
        EventMapper.__init__(self, ref_obj, event_mapper, filter=filter)

        self.use_cached = use_cached

    # ############# Duration ############# #

    @timeseries_property('Duration')
    def duration(self):
        """Duration of each event (raw)."""
        if self.use_cached is not None:
            return self._filter(self.use_cached.duration)

        return self.map(nb.event_duration_map_func_nb)

    @metric_property('Average duration')
    def avg_duration(self):
        """Average duration of an event (in time units)."""
        return self.duration.vbt.timeseries.mean(time_units=True)

    # ############# Count ############# #

    @metric_property('Total count')
    def count(self):
        """Total count of all events."""
        return self.duration.vbt.timeseries.count()

    # ############# P&L ############# #

    @timeseries_property('P&L')
    def pnl(self):
        """Profit and loss of each event."""
        if self.use_cached is not None:
            return self._filter(self.use_cached.pnl)

        return self.map(nb.event_pnl_map_func_nb)

    @metric_property('Total P&L')
    def total_pnl(self):
        """Total profit and loss of all events."""
        return self.pnl.vbt.timeseries.sum()

    @metric_property('Average P&L')
    def avg_pnl(self):
        """Average profit and loss of an event."""
        return self.pnl.vbt.timeseries.mean()

    def plot_pnl(self, profit_trace_kwargs={}, loss_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot profit and loss of each event as a marker.

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

    # ############# Returns ############# #

    @timeseries_property('Returns')
    def returns(self):
        """Return of each event."""
        if self.use_cached is not None:
            return self._filter(self.use_cached.returns)

        return self.map(nb.event_return_map_func_nb)

    @metric_property('Average return')
    def avg_return(self):
        """Average return of an event."""
        return self.returns.vbt.timeseries.mean()

    def plot_returns(self, profit_trace_kwargs={}, loss_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot return of each event as a marker.

        See `BaseEvents.plot_pnl`."""
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.returns.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)


class Events(BaseEvents):
    """Extends `BaseEvents` by combining various profit/loss metrics."""

    @group_property('Winning', BaseEvents)
    def winning(self):
        """Winning events of type `BaseEvents`."""
        return BaseEvents(
            self.ref_obj,
            self.event_mapper,
            filter=(nb.winning_filter_func_nb, self.pnl.vbt.to_2d_array()),
            use_cached=self)

    @group_property('Losing', BaseEvents)
    def losing(self):
        """Losing events of type `BaseEvents`."""
        return BaseEvents(
            self.ref_obj,
            self.event_mapper,
            filter=(nb.losing_filter_func_nb, self.pnl.vbt.to_2d_array()),
            use_cached=self)

    @metric_property('Win rate')
    def win_rate(self):
        """Rate of profitable events."""
        winning_count = reshape_fns.to_1d(self.winning.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        win_rate = winning_count / count
        return self.wrap_metric(win_rate)

    @metric_property('Profit factor')
    def profit_factor(self):
        """Profit factor."""
        total_win = reshape_fns.to_1d(self.winning.total_pnl, raw=True)
        total_loss = reshape_fns.to_1d(self.losing.total_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_trades = ~np.all(np.isnan(self.trade_size.vbt.to_2d_array()), axis=0)
        total_win[np.isnan(total_win) & has_trades] = 0.
        total_loss[np.isnan(total_loss) & has_trades] = 0.

        profit_factor = total_win / np.abs(total_loss)
        return self.wrap_metric(profit_factor)

    @metric_property('Expectancy')
    def expectancy(self):
        """Average profitability."""
        win_rate = reshape_fns.to_1d(self.win_rate, raw=True)
        avg_win = reshape_fns.to_1d(self.winning.avg_pnl, raw=True)
        avg_loss = reshape_fns.to_1d(self.losing.avg_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_trades = ~np.all(np.isnan(self.trade_size.vbt.to_2d_array()), axis=0)
        avg_win[np.isnan(avg_win) & has_trades] = 0.
        avg_loss[np.isnan(avg_loss) & has_trades] = 0.

        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        return self.wrap_metric(expectancy)
