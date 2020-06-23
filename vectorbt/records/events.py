"""Classes for working with event records."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.defaults import contrast_color_schema
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.reshape_fns import to_1d
from vectorbt.utils.decorators import cached_property
from vectorbt.utils.config import merge_kwargs
from vectorbt.utils.indexing import PandasIndexer
from vectorbt.timeseries.common import DatetimeTypes, TSArrayWrapper
from vectorbt.records.main import Records
from vectorbt.records import nb
from vectorbt.records.common import indexing_on_records
from vectorbt.records.enums import (
    EventStatus,
    event_dt,
    trade_dt,
    position_dt
)


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `BaseEvents`."""
    records_arr, _ = indexing_on_records(obj, pd_indexing_func)
    return obj.__class__(records_arr, pd_indexing_func(obj.main_price), freq=obj.wrapper.freq)


class BaseEvents(Records):
    """Extends `Records` for working with event records."""

    def __init__(self, records_arr, main_price, freq=None):
        Records.__init__(self, records_arr, TSArrayWrapper.from_obj(main_price, freq=freq))
        PandasIndexer.__init__(self, _indexing_func)

        if not all(field in records_arr.dtype.names for field in event_dt.names):
            raise Exception("Records array must have all fields defined in event_dt")

        self.main_price = main_price

    def plot(self,
             main_price_trace_kwargs={},
             open_trace_kwargs={},
             close_trace_kwargs={},
             close_profit_trace_kwargs={},
             close_loss_trace_kwargs={},
             active_trace_kwargs={},
             profit_shape_kwargs={},
             loss_shape_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot orders.

        Args:
            main_price_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for main price.
            open_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Open" markers.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Close" markers.
            close_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Close - Profit" markers.
            close_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Close - Loss" markers.
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
            portfolio = vbt.Portfolio.from_orders(price, orders, init_capital=100, freq='1D')

            portfolio.trades.plot()
            ```

            ![](/vectorbt/docs/img/events.png)"""
        if self.wrapper.ndim > 1:
            raise Exception("You must select a column first")

        # Plot main price
        fig = self.main_price.vbt.timeseries.plot(trace_kwargs=main_price_trace_kwargs, fig=fig, **layout_kwargs)

        # Extract information
        size = self.records_arr['size']
        open_idx = self.records_arr['open_idx']
        open_price = self.records_arr['open_price']
        open_fees = self.records_arr['open_fees']
        close_idx = self.records_arr['close_idx']
        close_price = self.records_arr['close_price']
        close_fees = self.records_arr['close_fees']
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

        duration = get_duration_str(open_idx, close_idx)

        # Plot Open markers
        open_customdata = np.stack((size, open_fees), axis=1)
        open_scatter = go.Scatter(
            x=self.wrapper.index[open_idx],
            y=open_price,
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
            name='Open',
            customdata=open_customdata,
            hovertemplate="%{x}<br>Price: %{y}<br>Size: %{customdata[0]:.4f}<br>Fees: %{customdata[1]:.2f}"
        )
        open_scatter.update(**open_trace_kwargs)
        fig.add_trace(open_scatter)

        # Plot end markers
        def plot_end_markers(mask, name, color, kwargs):
            customdata = np.stack((
                size[mask],
                close_fees[mask],
                pnl[mask],
                ret[mask],
                duration[mask]
            ), axis=1)
            scatter = go.Scatter(
                x=self.wrapper.index[close_idx[mask]],
                y=close_price[mask],
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

        # Plot Close markers
        plot_end_markers(
            (status == EventStatus.Closed) & (pnl == 0.),
            'Close',
            contrast_color_schema['gray'],
            close_trace_kwargs
        )

        # Plot Close - Profit markers
        plot_end_markers(
            (status == EventStatus.Closed) & (pnl > 0.),
            'Close - Profit',
            contrast_color_schema['green'],
            close_profit_trace_kwargs
        )

        # Plot Close - Loss markers
        plot_end_markers(
            (status == EventStatus.Closed) & (pnl < 0.),
            'Close - Loss',
            contrast_color_schema['red'],
            close_loss_trace_kwargs
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
                x0=self.wrapper.index[open_idx[i]],
                y0=open_price[i],
                x1=self.wrapper.index[close_idx[i]],
                y1=close_price[i],
                fillcolor=contrast_color_schema['green'],
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
                x0=self.wrapper.index[open_idx[i]],
                y0=open_price[i],
                x1=self.wrapper.index[close_idx[i]],
                y1=close_price[i],
                fillcolor=contrast_color_schema['red'],
                opacity=0.15,
                layer="below",
                line_width=0,
            ), loss_shape_kwargs))

        return fig

    # ############# Duration ############# #

    @cached_property
    def duration(self):
        """Duration of each event (in raw format)."""
        return self.map_records_to_matrix(nb.event_duration_map_nb)

    @cached_property
    def avg_duration(self):
        """Average duration (in time units)."""
        return self.map_reduce_records(nb.event_duration_map_nb, nb.mean_reduce_nb, time_units=True)

    @cached_property
    def max_duration(self):
        """Maximum duration (in time units)."""
        return self.map_reduce_records(nb.event_duration_map_nb, nb.max_reduce_nb, time_units=True)

    @cached_property
    def coverage(self):
        """Coverage, that is, total duration divided by the whole period."""
        total_duration = self.map_reduce_records(nb.event_duration_map_nb, nb.sum_reduce_nb, default_val=0.)
        coverage = to_1d(total_duration, raw=True) / self.wrapper.shape[0]
        return self.wrapper.wrap_reduced(coverage)

    # ############# PnL ############# #

    @cached_property
    def pnl(self):
        """PnL of each event."""
        return self.map_field_to_matrix('pnl')

    @cached_property
    def min_pnl(self):
        """Minimum PnL."""
        return self.map_reduce_records(nb.event_pnl_map_nb, nb.min_reduce_nb)

    @cached_property
    def max_pnl(self):
        """Maximum PnL."""
        return self.map_reduce_records(nb.event_pnl_map_nb, nb.max_reduce_nb)

    @cached_property
    def avg_pnl(self):
        """Average PnL."""
        return self.map_reduce_records(nb.event_pnl_map_nb, nb.mean_reduce_nb)

    @cached_property
    def total_pnl(self):
        """Total PnL of all events."""
        return self.map_reduce_records(nb.event_pnl_map_nb, nb.sum_reduce_nb)

    # ############# Returns ############# #

    @cached_property
    def returns(self):
        """Return of each event."""
        return self.map_field_to_matrix('return')

    @cached_property
    def min_return(self):
        """Minimum return."""
        return self.map_reduce_records(nb.event_return_map_nb, nb.min_reduce_nb)

    @cached_property
    def max_return(self):
        """Maximum return."""
        return self.map_reduce_records(nb.event_return_map_nb, nb.max_reduce_nb)

    @cached_property
    def avg_return(self):
        """Average return."""
        return self.map_reduce_records(nb.event_return_map_nb, nb.mean_reduce_nb)

    @cached_property
    def sqn(self):
        """System Quality Number (SQN)."""
        return self.reduce_records(nb.event_sqn_reduce_nb, 1)  # ddof


class BaseEventsByResult(BaseEvents):
    """Extends `BaseEvents` by further dividing events into winning and losing events."""

    @cached_property
    def winning(self):
        """Winning events of type `BaseEvents`."""
        filter_mask = self.records_arr['pnl'] > 0.
        return BaseEvents(self.records_arr[filter_mask], self.main_price, freq=self.wrapper.freq)

    @cached_property
    def losing(self):
        """Losing events of type `BaseEvents`."""
        filter_mask = self.records_arr['pnl'] < 0.
        return BaseEvents(self.records_arr[filter_mask], self.main_price, freq=self.wrapper.freq)

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
        total_win = to_1d(self.winning.total_pnl, raw=True)
        total_loss = to_1d(self.losing.total_pnl, raw=True)

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
        avg_win = to_1d(self.winning.avg_pnl, raw=True)
        avg_loss = to_1d(self.losing.avg_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = to_1d(self.count, raw=True) > 0
        avg_win[np.isnan(avg_win) & has_values] = 0.
        avg_loss[np.isnan(avg_loss) & has_values] = 0.

        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        return self.wrapper.wrap_reduced(expectancy)


class Events(BaseEventsByResult):
    """Extends `BaseEventsByResult` by further dividing events by status."""

    @cached_property
    def status(self):
        """See `vectorbt.records.enums.EventStatus`."""
        return self.map_field_to_matrix('status')

    @cached_property
    def closed_rate(self):
        """Rate of closed events."""
        closed_rate = to_1d(self.closed.count, raw=True) / to_1d(self.count, raw=True)
        return self.wrapper.wrap_reduced(closed_rate)

    @cached_property
    def open(self):
        """Open events of type `BaseEventsByResult`."""
        filter_mask = self.records_arr['status'] == EventStatus.Open
        return BaseEventsByResult(self.records_arr[filter_mask], self.main_price, freq=self.wrapper.freq)

    @cached_property
    def closed(self):
        """Closed events of type `BaseEventsByResult`."""
        filter_mask = self.records_arr['status'] == EventStatus.Closed
        return BaseEventsByResult(self.records_arr[filter_mask], self.main_price, freq=self.wrapper.freq)


class Trades(Events):
    """Extends `Events` for working with trade records.

    Such records can be created by using `vectorbt.portfolio.nb.trade_records_nb`.

    Example:
        Get the average PnL of trades with duration over 2 days:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from vectorbt.records import Trades

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, -1, 1, 0, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders,
        ...      init_capital=100, freq='1D')
        >>> print(portfolio.trades.avg_pnl)
        -0.5

        >>> records_arr = portfolio.trades.records_arr
        >>> duration_mask = (records_arr['close_idx'] - records_arr['open_idx']) >= 2.
        >>> trades = Trades(portfolio.wrapper, records_arr[duration_mask])
        >>> print(trades.avg_pnl)
        -2.0
        ```

        The same can be done by using `BaseEvents.reduce_records`,
        which skips the step of transforming records into a matrix and thus saves memory.
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit

        >>> @njit
        ... def reduce_func_nb(col_rs):
        ...     duration_mask = col_rs[:, TS.CloseIdx] - col_rs[:, TS.OpenIdx] >= 2.
        ...     return np.nanmean(col_rs[duration_mask, TS.PnL])

        >>> portfolio.trades.reduce_records(reduce_func_nb)
        -2.0
        ```"""

    def __init__(self, records_arr, main_price, freq=None):
        Events.__init__(self, records_arr, main_price, freq=freq)

        if not all(field in records_arr.dtype.names for field in trade_dt.names):
            raise Exception("Records array must have all fields defined in trade_dt")

    @classmethod
    def from_orders(cls, orders):
        """Build `Trades` from `Orders`."""
        trade_records = nb.trade_records_nb(orders.main_price.vbt.to_2d_array(), orders.records_arr)
        return cls(trade_records, orders.main_price, freq=orders.wrapper.freq)

    @cached_property
    def position_idx(self):
        """Position index of each trade."""
        return self.map_field_to_matrix('position_idx')


class Positions(Events):
    """Extends `Events` for working with position records.

    Such records can be created by using `vectorbt.portfolio.nb.position_records_nb`.

    Example:
        Get the average PnL of closed positions with duration over 2 days:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from vectorbt.records import Positions

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, -1, 1, 0, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders,
        ...      init_capital=100, freq='1D')
        >>> print(portfolio.positions.avg_pnl)
        -0.5

        >>> records_arr = portfolio.positions.closed.records_arr
        >>> duration_mask = (records_arr['close_idx'] - records_arr['open_idx']) >= 2.
        >>> positions = Positions(portfolio.wrapper, records_arr[duration_mask])
        >>> print(positions.avg_pnl)
        -2.0
        ```"""

    def __init__(self, records_arr, main_price, freq=None):
        Events.__init__(self, records_arr, main_price, freq=freq)

        if not all(field in records_arr.dtype.names for field in position_dt.names):
            raise Exception("Records array must have all fields defined in position_dt")

    @classmethod
    def from_orders(cls, orders):
        """Build `Positions` from `Orders`."""
        position_records = nb.position_records_nb(orders.main_price.vbt.to_2d_array(), orders.records_arr)
        return cls(position_records, orders.main_price, freq=orders.wrapper.freq)
