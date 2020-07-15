"""Classes for working with event records."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.defaults import contrast_color_schema
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.decorators import cached_property
from vectorbt.utils.config import merge_kwargs
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.reshape_fns import to_1d
from vectorbt.tseries.common import DatetimeTypes, TSArrayWrapper
from vectorbt.records.base import Records, indexing_on_records
from vectorbt.records import nb
from vectorbt.records.enums import (
    EventStatus,
    event_dt,
    trade_dt,
    position_dt
)


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `BaseEvents`."""
    records_arr, _ = indexing_on_records(obj, pd_indexing_func)
    return obj.__class__(records_arr, pd_indexing_func(obj.main_price), freq=obj.wrapper.freq, idx_field=obj.idx_field)


class BaseEvents(Records):
    """Extends `Records` for working with event records."""

    def __init__(self, records_arr, main_price, freq=None, idx_field='exit_idx'):
        Records.__init__(self, records_arr, TSArrayWrapper.from_obj(main_price, freq=freq), idx_field=idx_field)
        PandasIndexer.__init__(self, _indexing_func)

        if not all(field in records_arr.dtype.names for field in event_dt.names):
            raise ValueError("Records array must have all fields defined in event_dt")

        self.main_price = main_price

    def filter_by_mask(self, mask):
        """Return a new class instance, filtered by mask."""
        return self.__class__(self.records_arr[mask], self.main_price, freq=self.wrapper.freq, idx_field=self.idx_field)

    def plot(self,
             main_price_trace_kwargs={},
             entry_trace_kwargs={},
             exit_trace_kwargs={},
             exit_profit_trace_kwargs={},
             exit_loss_trace_kwargs={},
             active_trace_kwargs={},
             profit_shape_kwargs={},
             loss_shape_kwargs={},
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot orders.

        Args:
            main_price_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for main price.
            entry_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Entry" markers.
            exit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit" markers.
            exit_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Profit" markers.
            exit_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Loss" markers.
            active_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Active" markers.
            profit_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for profit zones.
            loss_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for loss zones.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            import vectorbt as vbt
            import pandas as pd

            price = pd.Series([1, 2, 3, 2, 1])
            orders = pd.Series([1, -1, 1, -1, 0])
            portfolio = vbt.Portfolio.from_orders(price, orders, freq='1D')

            portfolio.trades.plot()
            ```

            ![](/vectorbt/docs/img/events.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        # Plot main price
        fig = self.main_price.vbt.tseries.plot(trace_kwargs=main_price_trace_kwargs, fig=fig, **layout_kwargs)

        # Extract information
        size = self.records_arr['size']
        entry_idx = self.records_arr['entry_idx']
        entry_price = self.records_arr['entry_price']
        entry_fees = self.records_arr['entry_fees']
        exit_idx = self.records_arr['exit_idx']
        exit_price = self.records_arr['exit_price']
        exit_fees = self.records_arr['exit_fees']
        pnl = self.records_arr['pnl']
        ret = self.records_arr['return']
        status = self.records_arr['status']

        def get_duration_str(from_idx, to_idx):
            if isinstance(self.wrapper.index, DatetimeTypes):
                duration = self.wrapper.index[to_idx] - self.wrapper.index[from_idx]
            elif self.wrapper.freq is not None:
                duration = self.wrapper.to_time_units(to_idx - from_idx)
            else:
                duration = to_idx - from_idx
            return np.vectorize(str)(duration)

        duration = get_duration_str(entry_idx, exit_idx)

        # Plot Entry markers
        entry_customdata = np.stack((size, entry_fees), axis=1)
        entry_scatter = go.Scatter(
            x=self.wrapper.index[entry_idx],
            y=entry_price,
            mode='markers',
            marker=dict(
                symbol='circle',
                color=contrast_color_schema['blue'],
                size=7,
                line=dict(
                    width=1,
                    color=adjust_lightness(contrast_color_schema['blue'])
                )
            ),
            name='Entry',
            customdata=entry_customdata,
            hovertemplate="%{x}<br>Price: %{y}<br>Size: %{customdata[0]:.4f}<br>Fees: %{customdata[1]:.2f}"
        )
        entry_scatter.update(**entry_trace_kwargs)
        fig.add_trace(entry_scatter)

        # Plot end markers
        def plot_end_markers(mask, name, color, kwargs):
            customdata = np.stack((
                size[mask],
                exit_fees[mask],
                pnl[mask],
                ret[mask],
                duration[mask]
            ), axis=1)
            scatter = go.Scatter(
                x=self.wrapper.index[exit_idx[mask]],
                y=exit_price[mask],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    color=color,
                    size=7,
                    line=dict(
                        width=1,
                        color=adjust_lightness(color)
                    )
                ),
                name=name,
                customdata=customdata,
                hovertemplate="%{x}<br>Price: %{y}" +
                              "<br>Size: %{customdata[0]:.4f}" +
                              "<br>Fees: %{customdata[1]:.2f}" +
                              "<br>PnL: %{customdata[2]:.2f}" +
                              "<br>Return: %{customdata[3]:.2%}" +
                              "<br>Duration: %{customdata[4]}"
            )
            scatter.update(**kwargs)
            fig.add_trace(scatter)

        # Plot Exit markers
        plot_end_markers(
            (status == EventStatus.Closed) & (pnl == 0.),
            'Exit',
            contrast_color_schema['gray'],
            exit_trace_kwargs
        )

        # Plot Exit - Profit markers
        plot_end_markers(
            (status == EventStatus.Closed) & (pnl > 0.),
            'Exit - Profit',
            contrast_color_schema['green'],
            exit_profit_trace_kwargs
        )

        # Plot Exit - Loss markers
        plot_end_markers(
            (status == EventStatus.Closed) & (pnl < 0.),
            'Exit - Loss',
            contrast_color_schema['red'],
            exit_loss_trace_kwargs
        )

        # Plot Active markers
        plot_end_markers(
            status == EventStatus.Open,
            'Active',
            contrast_color_schema['orange'],
            active_trace_kwargs
        )

        # Plot profit zones
        profit_mask = pnl > 0.
        for i in np.flatnonzero(profit_mask):
            fig.add_shape(**merge_kwargs(dict(
                type="rect",
                xref="x",
                yref="y",
                x0=self.wrapper.index[entry_idx[i]],
                y0=entry_price[i],
                x1=self.wrapper.index[exit_idx[i]],
                y1=exit_price[i],
                fillcolor='green',
                opacity=0.15,
                layer="below",
                line_width=0,
            ), profit_shape_kwargs))

        # Plot loss zones
        loss_mask = pnl < 0.
        for i in np.flatnonzero(loss_mask):
            fig.add_shape(**merge_kwargs(dict(
                type="rect",
                xref="x",
                yref="y",
                x0=self.wrapper.index[entry_idx[i]],
                y0=entry_price[i],
                x1=self.wrapper.index[exit_idx[i]],
                y1=exit_price[i],
                fillcolor='red',
                opacity=0.15,
                layer="below",
                line_width=0,
            ), loss_shape_kwargs))

        return fig

    # ############# Duration ############# #

    @cached_property
    def duration(self):
        """Duration of each event (in raw format)."""
        return self.map(nb.event_duration_map_nb)

    @cached_property
    def coverage(self):
        """Coverage, that is, total duration divided by the whole period."""
        coverage = to_1d(self.duration.sum(), raw=True) / self.wrapper.shape[0]
        return self.wrapper.wrap_reduced(coverage)

    # ############# PnL ############# #

    @cached_property
    def pnl(self):
        """PnL of each event."""
        return self.map_field('pnl')

    # ############# Returns ############# #

    @cached_property
    def returns(self):
        """Return of each event."""
        return self.map_field('return')


class BaseEventsByResult(BaseEvents):
    """Extends `BaseEvents` by further dividing events into winning and losing events."""

    @cached_property
    def winning(self):
        """Winning events of type `BaseEvents`."""
        filter_mask = self.records_arr['pnl'] > 0.
        return BaseEvents(
            self.records_arr[filter_mask],
            self.main_price,
            freq=self.wrapper.freq,
            idx_field=self.idx_field
        )

    @cached_property
    def losing(self):
        """Losing events of type `BaseEvents`."""
        filter_mask = self.records_arr['pnl'] < 0.
        return BaseEvents(
            self.records_arr[filter_mask],
            self.main_price,
            freq=self.wrapper.freq,
            idx_field=self.idx_field
        )

    @cached_property
    def win_rate(self):
        """Rate of profitable events."""
        winning_count = to_1d(self.winning.count, raw=True)
        count = to_1d(self.count, raw=True)

        win_rate = winning_count / count
        return self.wrapper.wrap_reduced(win_rate)

    @cached_property
    def profit_factor(self):
        """Profit factor."""
        total_win = to_1d(self.winning.pnl.sum(), raw=True)
        total_loss = to_1d(self.losing.pnl.sum(), raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = to_1d(self.count, raw=True) > 0
        total_win[np.isnan(total_win) & has_values] = 0.
        total_loss[np.isnan(total_loss) & has_values] = 0.

        profit_factor = total_win / np.abs(total_loss)
        return self.wrapper.wrap_reduced(profit_factor)

    @cached_property
    def expectancy(self):
        """Average profitability."""
        win_rate = to_1d(self.win_rate, raw=True)
        avg_win = to_1d(self.winning.pnl.mean(), raw=True)
        avg_loss = to_1d(self.losing.pnl.mean(), raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = to_1d(self.count, raw=True) > 0
        avg_win[np.isnan(avg_win) & has_values] = 0.
        avg_loss[np.isnan(avg_loss) & has_values] = 0.

        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        return self.wrapper.wrap_reduced(expectancy)

    @cached_property
    def sqn(self):
        """System Quality Number (SQN)."""
        count = to_1d(self.count, raw=True)
        pnl_mean = to_1d(self.pnl.mean(), raw=True)
        pnl_std = to_1d(self.pnl.std(), raw=True)
        sqn = np.sqrt(count) * pnl_mean / pnl_std
        return self.wrapper.wrap_reduced(sqn)


class Events(BaseEventsByResult):
    """Extends `BaseEventsByResult` by further dividing events by status."""

    @cached_property
    def status(self):
        """See `vectorbt.records.enums.EventStatus`."""
        return self.map_field('status')

    @cached_property
    def closed_rate(self):
        """Rate of closed events."""
        closed_rate = to_1d(self.closed.count, raw=True) / to_1d(self.count, raw=True)
        return self.wrapper.wrap_reduced(closed_rate)

    @cached_property
    def open(self):
        """Open events of type `BaseEventsByResult`."""
        filter_mask = self.records_arr['status'] == EventStatus.Open
        return BaseEventsByResult(
            self.records_arr[filter_mask],
            self.main_price,
            freq=self.wrapper.freq,
            idx_field=self.idx_field
        )

    @cached_property
    def closed(self):
        """Closed events of type `BaseEventsByResult`."""
        filter_mask = self.records_arr['status'] == EventStatus.Closed
        return BaseEventsByResult(
            self.records_arr[filter_mask],
            self.main_price,
            freq=self.wrapper.freq,
            idx_field=self.idx_field
        )


class Trades(Events):
    """Extends `Events` for working with trade records.

    Such records can be created by using `vectorbt.records.nb.trade_records_nb`.

    Example:
        Get count and P&L of trades:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> price = pd.Series([1, 2, 3, 4, 3, 2, 1])
        >>> orders = pd.Series([1, -0.5, -0.5, 2, -0.5, -0.5, -0.5])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders,
        ...      init_capital=100, freq='1D')

        >>> trades = vbt.Trades.from_orders(portfolio.orders)
        >>> print(trades.count)
        6
        >>> print(trades.pnl.sum())
        -3.0
        >>> print(trades.winning.count)
        2
        >>> print(trades.winning.pnl.sum())
        1.5
        ```

        Get count and P&L of trades with duration of more than 2 days:
        ```python-repl
        >>> mask = (trades.records['exit_idx'] - trades.records['entry_idx']) > 2
        >>> trades_filtered = trades.filter_by_mask(mask)
        >>> print(trades_filtered.count)
        2
        >>> print(trades_filtered.pnl.sum())
        -3.0
        ```"""

    def __init__(self, records_arr, main_price, **kwargs):
        Events.__init__(self, records_arr, main_price, **kwargs)

        if not all(field in records_arr.dtype.names for field in trade_dt.names):
            raise ValueError("Records array must have all fields defined in trade_dt")

    @classmethod
    def from_orders(cls, orders, **kwargs):
        """Build `Trades` from `Orders`."""
        trade_records = nb.trade_records_nb(orders.main_price.vbt.to_2d_array(), orders.records_arr)
        return cls(trade_records, orders.main_price, freq=orders.wrapper.freq, **kwargs)

    @cached_property
    def position_idx(self):
        """Position index of each trade."""
        return self.map_field('position_idx')



class Positions(Events):
    """Extends `Events` for working with position records.

    Such records can be created by using `vectorbt.records.nb.position_records_nb`.

    Example:
        Get count and P&L of positions:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> price = pd.Series([1, 2, 3, 4, 3, 2, 1])
        >>> orders = pd.Series([1, -0.5, -0.5, 1, -1, 2, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders,
        ...      init_capital=100, freq='1D')

        >>> positions = vbt.Positions.from_orders(portfolio.orders)
        >>> print(positions.count)
        3
        >>> print(positions.pnl.sum())
        -1.5
        >>> print(positions.open.pnl.sum())
        -2.0
        >>> print(positions.closed.pnl.sum())
        0.5
        ```

        Get count and P&L of positions with size of more than 1 share:
        ```python-repl
        >>> mask = positions.records['size'] > 1
        >>> positions_filtered = positions.filter_by_mask(mask)
        >>> print(positions_filtered.count)
        1
        >>> print(positions_filtered.pnl.sum())
        -2.0
        ```"""

    def __init__(self, records_arr, main_price, **kwargs):
        Events.__init__(self, records_arr, main_price, **kwargs)

        if not all(field in records_arr.dtype.names for field in position_dt.names):
            raise ValueError("Records array must have all fields defined in position_dt")

    @classmethod
    def from_orders(cls, orders, **kwargs):
        """Build `Positions` from `Orders`."""
        position_records = nb.position_records_nb(orders.main_price.vbt.to_2d_array(), orders.records_arr)
        return cls(position_records, orders.main_price, freq=orders.wrapper.freq, **kwargs)

