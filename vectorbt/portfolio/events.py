"""Classes for measuring performance of any events."""

import numpy as np
import pandas as pd

from vectorbt import timeseries
from vectorbt.utils import checks, reshape_fns
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import EventRecord
from vectorbt.portfolio.common import (
    timeseries_property,
    metric_property,
    group_property
)


class BaseEvents():
    """Exposes a range of attributes related to events.

    This class doesn't hold any data, but creates a read-only view over event records.
    Except that all time series and metric properties are cached.
    
    Args:
        ts_wrapper (TSArrayWrapper): Wrapper for wrapping time series and metrics.
        records (np.ndarray): Array of records that can be mapped to `vectorbt.portfolio.enums.EventRecord`."""

    def __init__(self, ts_wrapper, records):
        self.ts_wrapper = ts_wrapper
        self.records = records

    def map_to_matrix(self, map_func_nb, *args):
        """Convert event records to a matrix."""
        return self.ts_wrapper.wrap(
            nb.map_records_to_matrix_nb(
                self.records, 
                (len(self.ts_wrapper.index), len(self.ts_wrapper.columns)), 
                map_func_nb, 
                *args))

    # ############# Duration ############# #

    @timeseries_property('Duration')
    def duration(self):
        """Duration of each event (raw)."""
        return self.map_to_matrix(nb.duration_map_func_nb)

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
        return self.map_to_matrix(nb.field_map_func_nb, EventRecord.PnL)

    @metric_property('Total P&L')
    def total_pnl(self):
        """Total profit and loss of all events."""
        return self.pnl.vbt.timeseries.sum()

    @metric_property('Average P&L')
    def avg_pnl(self):
        """Average profit and loss of an event."""
        return self.pnl.vbt.timeseries.mean()

    def plot_pnl(self, profit_trace_kwargs={}, loss_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot profit and loss of each event as markers.

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
        return self.map_to_matrix(nb.field_map_func_nb, EventRecord.Return)

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
    """Extends `BaseEvents` by providing various profit/loss metrics."""

    @group_property('Winning', BaseEvents)
    def winning(self):
        """Winning events of type `BaseEvents`."""
        filter_mask = self.records[:, EventRecord.PnL] > 0.
        return BaseEvents(self.ts_wrapper, self.records[filter_mask, :])

    @group_property('Losing', BaseEvents)
    def losing(self):
        """Losing events of type `BaseEvents`."""
        filter_mask = self.records[:, EventRecord.PnL] < 0.
        return BaseEvents(self.ts_wrapper, self.records[filter_mask, :])

    @metric_property('Win rate')
    def win_rate(self):
        """Rate of profitable events."""
        winning_count = reshape_fns.to_1d(self.winning.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        win_rate = winning_count / count
        return self.ts_wrapper.wrap_reduced(win_rate)

    @metric_property('Profit factor')
    def profit_factor(self):
        """Profit factor."""
        total_win = reshape_fns.to_1d(self.winning.total_pnl, raw=True)
        total_loss = reshape_fns.to_1d(self.losing.total_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_trades = reshape_fns.to_1d(self.count, raw=True) > 0
        total_win[np.isnan(total_win) & has_trades] = 0.
        total_loss[np.isnan(total_loss) & has_trades] = 0.

        profit_factor = total_win / np.abs(total_loss)
        return self.ts_wrapper.wrap_reduced(profit_factor)

    @metric_property('Expectancy')
    def expectancy(self):
        """Average profitability."""
        win_rate = reshape_fns.to_1d(self.win_rate, raw=True)
        avg_win = reshape_fns.to_1d(self.winning.avg_pnl, raw=True)
        avg_loss = reshape_fns.to_1d(self.losing.avg_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_trades = reshape_fns.to_1d(self.count, raw=True) > 0
        avg_win[np.isnan(avg_win) & has_trades] = 0.
        avg_loss[np.isnan(avg_loss) & has_trades] = 0.

        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        return self.ts_wrapper.wrap_reduced(expectancy)
