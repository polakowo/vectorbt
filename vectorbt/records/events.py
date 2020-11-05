"""Classes for working with event records.

!!! note
    Without distribution of orders, trades and positions yield the same results.

!!! warning
    Both record types return both closed AND open events, which may skew your performance results
    significantly. To only consider closed events, you must query `closed` attribute explicitly."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.enums import (
    EventStatus,
    event_dt,
    trade_dt,
    position_dt
)
from vectorbt.defaults import contrast_color_schema
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import merge_kwargs, Configured
from vectorbt.utils.datetime import DatetimeTypes
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.reshape_fns import to_1d
from vectorbt.records.base import Records, indexing_on_records_meta
from vectorbt.records import nb


def indexing_on_events_meta(obj, pd_indexing_func):
    """Perform indexing on `BaseEvents` and also return metadata."""
    new_wrapper, new_records_arr, group_idxs, col_idxs = indexing_on_records_meta(obj, pd_indexing_func)
    new_ref_price = new_wrapper.wrap(obj.close.values[:, col_idxs], group_by=False)
    return obj.copy(
        wrapper=new_wrapper,
        records_arr=new_records_arr,
        close=new_ref_price
    ), group_idxs, col_idxs


def events_indexing_func(obj, pd_indexing_func):
    """Perform indexing on `BaseEvents`."""
    return indexing_on_events_meta(obj, pd_indexing_func)[0]


# ############# Events ############# #


class BaseEvents(Records):
    """Extends `Records` for working with event records."""

    def __init__(self, wrapper, records_arr, close, idx_field='exit_idx'):
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field
        )
        Configured.__init__(
            self,
            wrapper=wrapper,
            records_arr=records_arr,
            close=close,
            idx_field=idx_field
        )
        self.close = close

        if not all(field in records_arr.dtype.names for field in event_dt.names):
            raise ValueError("Records array must have all fields defined in event_dt")

        PandasIndexer.__init__(self, events_indexing_func)

    @property  # no need for cached
    def records_readable(self):
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame(columns=[
            'Column', 'Size', 'Entry Date', 'Entry Price', 'Entry Fees', 'Exit Date',
            'Exit Price', 'Exit Fees', 'P&L', 'Return', 'Status'
        ])
        out['Column'] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        out['Size'] = records_df['size']
        out['Entry Date'] = records_df['entry_idx'].map(lambda x: self.wrapper.index[x])
        out['Entry Price'] = records_df['entry_price']
        out['Entry Fees'] = records_df['entry_fees']
        out['Exit Date'] = records_df['exit_idx'].map(lambda x: self.wrapper.index[x])
        out['Exit Price'] = records_df['exit_price']
        out['Exit Fees'] = records_df['exit_fees']
        out['P&L'] = records_df['pnl']
        out['Return'] = records_df['return']
        out['Status'] = records_df['status'].map(lambda x: EventStatus._fields[x])
        return out

    def plot(self,
             column=None,
             ref_price_trace_kwargs=None,
             entry_trace_kwargs=None,
             exit_trace_kwargs=None,
             exit_profit_trace_kwargs=None,
             exit_loss_trace_kwargs=None,
             active_trace_kwargs=None,
             profit_shape_kwargs=None,
             loss_shape_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            ref_price_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for main price.
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
            ```python-repl
            >>> import vectorbt as vbt
            >>> import pandas as pd

            >>> price = pd.Series([1, 2, 3, 2, 1])
            >>> orders = pd.Series([1, -1, 1, -1, 0])
            >>> portfolio = vbt.Portfolio.from_orders(price, orders, freq='1D')

            >>> portfolio.trades.plot()
            ```

            ![](/vectorbt/docs/img/events.png)"""
        if column is not None:
            if self.wrapper.grouper.group_by is None:
                self_col = self[column]
            else:
                self_col = self.copy(wrapper=self.wrapper.copy(group_by=None))[column]
        else:
            self_col = self
        if self_col.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing or column argument.")

        if ref_price_trace_kwargs is None:
            ref_price_trace_kwargs = {}
        if entry_trace_kwargs is None:
            entry_trace_kwargs = {}
        if exit_trace_kwargs is None:
            exit_trace_kwargs = {}
        if exit_profit_trace_kwargs is None:
            exit_profit_trace_kwargs = {}
        if exit_loss_trace_kwargs is None:
            exit_loss_trace_kwargs = {}
        if active_trace_kwargs is None:
            active_trace_kwargs = {}
        if profit_shape_kwargs is None:
            profit_shape_kwargs = {}
        if loss_shape_kwargs is None:
            loss_shape_kwargs = {}

        # Plot main price
        fig = self_col.close.vbt.plot(trace_kwargs=ref_price_trace_kwargs, fig=fig, **layout_kwargs)

        # Extract information
        size = self_col.records_arr['size']
        entry_idx = self_col.records_arr['entry_idx']
        entry_price = self_col.records_arr['entry_price']
        entry_fees = self_col.records_arr['entry_fees']
        exit_idx = self_col.records_arr['exit_idx']
        exit_price = self_col.records_arr['exit_price']
        exit_fees = self_col.records_arr['exit_fees']
        pnl = self_col.records_arr['pnl']
        ret = self_col.records_arr['return']
        status = self_col.records_arr['status']

        def get_duration_str(from_idx, to_idx):
            if isinstance(self_col.wrapper.index, DatetimeTypes):
                duration = self_col.wrapper.index[to_idx] - self_col.wrapper.index[from_idx]
            elif self_col.wrapper.freq is not None:
                duration = self_col.wrapper.to_time_units(to_idx - from_idx)
            else:
                duration = to_idx - from_idx
            return np.vectorize(str)(duration)

        duration = get_duration_str(entry_idx, exit_idx)

        # Plot Entry markers
        entry_customdata = np.stack((size, entry_fees), axis=1)
        entry_scatter = go.Scatter(
            x=self_col.wrapper.index[entry_idx],
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
            hovertemplate="%{x}<br>Price: %{y}<br>Size: %{customdata[0]:.4f}<br>Fees: %{customdata[1]:.4f}"
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
                x=self_col.wrapper.index[exit_idx[mask]],
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
                              "<br>Fees: %{customdata[1]:.4f}" +
                              "<br>PnL: %{customdata[2]:.4f}" +
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
                x0=self_col.wrapper.index[entry_idx[i]],
                y0=entry_price[i],
                x1=self_col.wrapper.index[exit_idx[i]],
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
                x0=self_col.wrapper.index[entry_idx[i]],
                y0=entry_price[i],
                x1=self_col.wrapper.index[exit_idx[i]],
                y1=exit_price[i],
                fillcolor='red',
                opacity=0.15,
                layer="below",
                line_width=0,
            ), loss_shape_kwargs))

        return fig

    @cached_property
    def duration(self):
        """Duration of each event (in raw format)."""
        return self.map(nb.event_duration_map_nb)

    @cached_method
    def coverage(self, group_by=None, **kwargs):
        """Coverage, that is, total duration divided by the whole period."""
        total_duration = to_1d(self.duration.sum(group_by=group_by), raw=True)
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        return self.wrapper.wrap_reduced(total_duration / total_steps, group_by=group_by, **kwargs)

    @cached_property
    def pnl(self):
        """PnL of each event."""
        return self.map_field('pnl')

    @cached_property
    def returns(self):
        """Return of each event."""
        return self.map_field('return')


class BaseEventsByResult(BaseEvents):
    """Extends `BaseEvents` by further dividing events into winning and losing events."""
    BaseEvents = BaseEvents

    @cached_property
    def winning(self):
        """Winning events of type `BaseEvents`."""
        filter_mask = self.records_arr['pnl'] > 0.
        return self.BaseEvents(
            self.wrapper,
            self.records_arr[filter_mask],
            self.close,
            idx_field=self.idx_field
        )

    @cached_property
    def losing(self):
        """Losing events of type `BaseEvents`."""
        filter_mask = self.records_arr['pnl'] < 0.
        return self.BaseEvents(
            self.wrapper,
            self.records_arr[filter_mask],
            self.close,
            idx_field=self.idx_field
        )

    @cached_method
    def win_rate(self, group_by=None, **kwargs):
        """Rate of profitable events."""
        win_count = to_1d(self.winning.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(win_count / total_count, group_by=group_by, **kwargs)

    @cached_method
    def profit_factor(self, group_by=None, **kwargs):
        """Profit factor."""
        total_win = to_1d(self.winning.pnl.sum(group_by=group_by), raw=True)
        total_loss = to_1d(self.losing.pnl.sum(group_by=group_by), raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = to_1d(self.count(group_by=group_by), raw=True) > 0
        total_win[np.isnan(total_win) & has_values] = 0.
        total_loss[np.isnan(total_loss) & has_values] = 0.

        profit_factor = total_win / np.abs(total_loss)
        return self.wrapper.wrap_reduced(profit_factor, group_by=group_by, **kwargs)

    @cached_method
    def expectancy(self, group_by=None, **kwargs):
        """Average profitability."""
        win_rate = to_1d(self.win_rate(group_by=group_by), raw=True)
        avg_win = to_1d(self.winning.pnl.mean(group_by=group_by), raw=True)
        avg_loss = to_1d(self.losing.pnl.mean(group_by=group_by), raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = to_1d(self.count(group_by=group_by), raw=True) > 0
        avg_win[np.isnan(avg_win) & has_values] = 0.
        avg_loss[np.isnan(avg_loss) & has_values] = 0.

        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        return self.wrapper.wrap_reduced(expectancy, group_by=group_by, **kwargs)

    @cached_method
    def sqn(self, group_by=None, **kwargs):
        """System Quality Number (SQN)."""
        count = to_1d(self.count(group_by=group_by), raw=True)
        pnl_mean = to_1d(self.pnl.mean(group_by=group_by), raw=True)
        pnl_std = to_1d(self.pnl.std(group_by=group_by), raw=True)
        sqn = np.sqrt(count) * pnl_mean / pnl_std
        return self.wrapper.wrap_reduced(sqn, group_by=group_by, **kwargs)


class Events(BaseEventsByResult):
    """Extends `BaseEventsByResult` by further dividing events by status."""
    BaseEventsByResult = BaseEventsByResult

    @cached_property
    def status(self):
        """See `vectorbt.enums.EventStatus`."""
        return self.map_field('status')

    @cached_method
    def closed_rate(self, group_by=None, **kwargs):
        """Rate of closed events."""
        closed_count = to_1d(self.closed.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(closed_count / total_count, group_by=group_by, **kwargs)

    @cached_property
    def open(self):
        """Open events of type `BaseEventsByResult`."""
        filter_mask = self.records_arr['status'] == EventStatus.Open
        return self.BaseEventsByResult(
            self.wrapper,
            self.records_arr[filter_mask],
            self.close,
            idx_field=self.idx_field
        )

    @cached_property
    def closed(self):
        """Closed events of type `BaseEventsByResult`."""
        filter_mask = self.records_arr['status'] == EventStatus.Closed
        return self.BaseEventsByResult(
            self.wrapper,
            self.records_arr[filter_mask],
            self.close,
            idx_field=self.idx_field
        )


# ############# Trades ############# #


class BaseTrades(BaseEvents):
    """`BaseEvents` adapted for trades."""
    @property  # no need for cached
    def records_readable(self):
        """Records in readable format."""
        records_df = self.records
        out = super().records_readable.copy()
        out['Position'] = records_df['position_idx']
        return out


class BaseTradesByResult(BaseEventsByResult, BaseTrades):
    """`BaseEventsByResult` adapted for trades."""
    BaseEvents = BaseTrades


class Trades(Events, BaseTradesByResult):
    """Extends `Events` for working with trade records.

    Such records can be created by using `vectorbt.records.nb.trade_records_nb`.

    In context of vectorbt, a trade is simply a sell operation. For example, if you have a single large
    buy operation and 100 small sell operations, you will see 100 trades, each opening with a fraction
    of the buy operation's size and fees. On the other hand, having 100 buy operations and just a single
    sell operation will generate a single trade with buy price being a size-weighted average over all
    purchase prices, and opening size and fees being the sum over all sizes and fees.

    Example:
        Increasing position:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> vbt.Portfolio.from_orders(
        ...     pd.Series([1., 2., 3., 4., 5.]),
        ...     pd.Series([1., 1., 1., 1., -4.]),
        ...     fixed_fees=1., freq='1D'
        ... ).trades().records
           col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0    0   4.0          0          2.5         4.0         4         5.0

           exit_fees  pnl    return  status  position_idx
        0        1.0  5.0  0.357143       1             0
        ```

        Decreasing position:
        ```python-repl
        >>> vbt.Portfolio.from_orders(
        ...     pd.Series([1., 2., 3., 4., 5.]),
        ...     pd.Series([4., -1., -1., -1., -1.]),
        ...     fixed_fees=1., freq='1D'
        ... ).trades().records
           col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \
        0    0   1.0          0          1.0        0.25         1         2.0
        1    0   1.0          0          1.0        0.25         2         3.0
        2    0   1.0          0          1.0        0.25         3         4.0
        3    0   1.0          0          1.0        0.25         4         5.0

           exit_fees   pnl  return  status  position_idx
        0        1.0 -0.25    -0.2       1             0
        1        1.0  0.75     0.6       1             0
        2        1.0  1.75     1.4       1             0
        3        1.0  2.75     2.2       1             0
        ```

        Multiple positions:
        ```python-repl
        >>> vbt.Portfolio.from_orders(
        ...     pd.Series([1., 2., 3., 4., 5.]),
        ...     pd.Series([1., 1., -2., 1., -1.]),
        ...     fixed_fees=1., freq='1D'
        ... ).trades().records
           col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0    0   2.0          0          1.5         2.0         2         3.0
        1    0   1.0          3          4.0         1.0         4         5.0

           exit_fees  pnl  return  status  position_idx
        0        1.0  0.0     0.0       1             0
        1        1.0 -1.0    -0.2       1             1
        ```

        Get count and P&L of trades:
        ```python-repl
        >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.])
        >>> orders = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders,
        ...      init_cash=100., freq='1D')

        >>> trades = vbt.Trades.from_orders(portfolio.orders())
        >>> trades.count()
        6
        >>> trades.pnl.sum()
        -3.0
        >>> trades.winning.count()
        2
        >>> trades.winning.pnl.sum()
        1.5
        ```

        Get count and P&L of trades with duration of more than 2 days:
        ```python-repl
        >>> mask = (trades.records['exit_idx'] - trades.records['entry_idx']) > 2
        >>> trades_filtered = trades.filter_by_mask(mask)
        >>> trades_filtered.count()
        2
        >>> trades_filtered.pnl.sum()
        -3.0
        ```"""
    BaseEventsByResult = BaseTradesByResult

    def __init__(self, wrapper, records_arr, *args, **kwargs):
        Events.__init__(self, wrapper, records_arr, *args, **kwargs)

        if not all(field in records_arr.dtype.names for field in trade_dt.names):
            raise ValueError("Records array must have all fields defined in trade_dt")

    @classmethod
    def from_orders(cls, orders, **kwargs):
        """Build `Trades` from `Orders`."""
        trade_records_arr = nb.trade_records_nb(orders.close.vbt.to_2d_array(), orders.records_arr)
        return cls(orders.wrapper, trade_records_arr, orders.close, **kwargs)

    @cached_property
    def position_idx(self):
        """Position index of each trade."""
        return self.map_field('position_idx')


# ############# Positions ############# #


class BasePositions(BaseEvents):
    """`BaseEvents` adapted for positions."""
    pass


class BasePositionsByResult(BaseEventsByResult, BasePositions):
    """`BaseEventsByResult` adapted for positions."""
    BaseEvents = BasePositions


class Positions(Events, BasePositionsByResult):
    """Extends `Events` for working with position records.

    Such records can be created by using `vectorbt.records.nb.position_records_nb`.

    Positions can incorporate multiple trades. They are allowed to increase/decrease over time.
    Each buy/sell operation is tracked and then used for deriving P&L of the entire position.
    A position opens with first buy operation and closes with last sell operation that results
    in no security holdings.

    Example:
        Increasing position:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> vbt.Portfolio.from_orders(
        ...     pd.Series([1., 2., 3., 4., 5.]),
        ...     pd.Series([1., 1., 1., 1., -4.]),
        ...     fixed_fees=1., freq='1D'
        ... ).positions().records
           col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0    0   4.0          0          2.5         4.0         4         5.0

           exit_fees  pnl    return  status
        0        1.0  5.0  0.357143       1
        ```

        Decreasing position:
        ```python-repl
        >>> vbt.Portfolio.from_orders(
        ...     pd.Series([1., 2., 3., 4., 5.]),
        ...     pd.Series([4., -1., -1., -1., -1.]),
        ...     fixed_fees=1., freq='1D'
        ... ).positions().records
           col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0    0   4.0          0          1.0         1.0         4         3.5

           exit_fees  pnl  return  status
        0        4.0  5.0     1.0       1
        ```

        Multiple positions:
        ```python-repl
        >>> vbt.Portfolio.from_orders(
        ...     pd.Series([1., 2., 3., 4., 5.]),
        ...     pd.Series([1., 1., -2., 1., -1.]),
        ...     fixed_fees=1., freq='1D'
        ... ).positions().records
           col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
        0    0   2.0          0          1.5         2.0         2         3.0
        1    0   1.0          3          4.0         1.0         4         5.0

           exit_fees  pnl  return  status
        0        1.0  0.0     0.0       1
        1        1.0 -1.0    -0.2       1
        ```

        Get count and P&L of positions:
        ```python-repl
        >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.])
        >>> orders = pd.Series([1., -0.5, -0.5, 1., -1., 2., -1.])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders,
        ...      init_cash=100., freq='1D')

        >>> positions = vbt.Positions.from_orders(portfolio.orders())
        >>> positions.count()
        3
        >>> positions.pnl.sum()
        -1.5
        >>> positions.open.pnl.sum()
        -2.0
        >>> positions.closed.pnl.sum()
        0.5
        ```

        Get count and P&L of positions with size of more than 1 share:
        ```python-repl
        >>> mask = positions.records['size'] > 1
        >>> positions_filtered = positions.filter_by_mask(mask)
        >>> positions_filtered.count()
        1
        >>> positions_filtered.pnl.sum()
        -2.0
        ```"""
    BaseEventsByResult = BasePositionsByResult

    def __init__(self, wrapper, records_arr, *args, **kwargs):
        Events.__init__(self, wrapper, records_arr, *args, **kwargs)

        if not all(field in records_arr.dtype.names for field in position_dt.names):
            raise ValueError("Records array must have all fields defined in position_dt")

    @classmethod
    def from_orders(cls, orders, **kwargs):
        """Build `Positions` from `Orders`."""
        position_records_arr = nb.position_records_nb(orders.close.vbt.to_2d_array(), orders.records_arr)
        return cls(orders.wrapper, position_records_arr, orders.close, **kwargs)

