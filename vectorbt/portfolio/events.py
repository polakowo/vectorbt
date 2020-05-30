"""Classes for measuring performance of any events."""

import numpy as np
import pandas as pd

from vectorbt import timeseries
from vectorbt.utils import checks, reshape_fns
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import EventRecord
from vectorbt.portfolio.records import Records
from vectorbt.portfolio.common import (
    timeseries_property, 
    metric_property, 
    group_property
)

class BaseEvents(Records):
    """Exposes methods and properties for working with any event records.

    This class doesn't hold any data, but creates a read-only view over event records.
    Except that all time series and metric properties are cached."""

    def __init__(self, wrapper, records, layout=EventRecord):
        checks.assert_same(EventRecord._fields, layout._fields[:len(EventRecord)]) # subtype of EventRecord

        super().__init__(wrapper, records, layout, EventRecord.Column, EventRecord.CloseAt)

    # ############# Duration ############# #

    @timeseries_property('Duration')
    def duration(self):
        """Duration of each event (in raw format)."""
        return self.map_records_to_matrix(nb.duration_map_func_nb)

    @metric_property('Average duration')
    def avg_duration(self):
        """Average duration of an event (in time format)."""
        return self.duration.vbt.timeseries.mean(time_units=True)

    # ############# P&L ############# #

    @timeseries_property('P&L')
    def pnl(self):
        """P&L of each event."""
        return self.map_records_to_matrix(nb.field_map_func_nb, EventRecord.PnL)

    @metric_property('Total P&L')
    def total_pnl(self):
        """Total P&L of all events."""
        return self.pnl.vbt.timeseries.sum()

    @metric_property('Average P&L')
    def avg_pnl(self):
        """Average P&L of an event."""
        return self.pnl.vbt.timeseries.mean()

    def plot_pnl(self, profit_trace_kwargs={}, loss_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot P&L of each event as markers.

        Args:
            profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Profit" markers.
            loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Loss" markers.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout."""
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.pnl.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)

    # ############# Returns ############# #

    @timeseries_property('Returns')
    def returns(self):
        """Return of each event."""
        return self.map_records_to_matrix(nb.field_map_func_nb, EventRecord.Return)

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
    """Extends `BaseEvents` by further dividing events into winning and losing."""

    @group_property('Winning', BaseEvents)
    def winning(self):
        """Winning events of type `BaseEvents`."""
        filter_mask = self._records[:, EventRecord.PnL] > 0.
        return BaseEvents(self.wrapper, self._records[filter_mask, :], layout=self.layout)

    @group_property('Losing', BaseEvents)
    def losing(self):
        """Losing events of type `BaseEvents`."""
        filter_mask = self._records[:, EventRecord.PnL] < 0.
        return BaseEvents(self.wrapper, self._records[filter_mask, :], layout=self.layout)

    @metric_property('Win rate')
    def win_rate(self):
        """Rate of profitable events."""
        winning_count = reshape_fns.to_1d(self.winning.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        win_rate = winning_count / count
        return self.wrapper.wrap_metric(win_rate)

    @metric_property('Profit factor')
    def profit_factor(self):
        """Profit factor."""
        total_win = reshape_fns.to_1d(self.winning.total_pnl, raw=True)
        total_loss = reshape_fns.to_1d(self.losing.total_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = reshape_fns.to_1d(self.count, raw=True) > 0
        total_win[np.isnan(total_win) & has_values] = 0.
        total_loss[np.isnan(total_loss) & has_values] = 0.

        profit_factor = total_win / np.abs(total_loss)
        return self.wrapper.wrap_metric(profit_factor)

    @metric_property('Expectancy')
    def expectancy(self):
        """Average profitability."""
        win_rate = reshape_fns.to_1d(self.win_rate, raw=True)
        avg_win = reshape_fns.to_1d(self.winning.avg_pnl, raw=True)
        avg_loss = reshape_fns.to_1d(self.losing.avg_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = reshape_fns.to_1d(self.count, raw=True) > 0
        avg_win[np.isnan(avg_win) & has_values] = 0.
        avg_loss[np.isnan(avg_loss) & has_values] = 0.

        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        return self.wrapper.wrap_metric(expectancy)
