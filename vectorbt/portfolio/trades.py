"""Base class for working with trade records.

!!! warning
    Both record types return both closed AND open trades, which may skew your performance results.
    To only consider closed trades, you should explicitly query `closed` attribute."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.datetime import DatetimeTypes
from vectorbt.utils.enum import to_value_map
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.utils.array import min_rel_rescale, max_rel_rescale
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast_to
from vectorbt.records.base import Records
from vectorbt.portfolio.enums import TradeDirection, TradeStatus, trade_dt, position_dt, TradeType
from vectorbt.portfolio import nb


# ############# Trades ############# #


class Trades(Records):
    """Extends `Records` for working with trade records.

    In vectorbt, a trade is a partial closing operation; it's is a more fine-grained representation
    of a position. One position can incorporate multiple trades. Performance for this operation is
    calculated based on the size-weighted average of previous opening operations within the same
    position. The PnL of all trades combined always equals to the PnL of the entire position.

    For example, if you have a single large buy operation and 100 small sell operations, you will see
    100 trades, each opening with a fraction of the buy operation's size and fees. On the other hand,
    having 100 buy operations and just a single sell operation will generate a single trade with buy
    price being a size-weighted average over all purchase prices, and opening size and fees being
    the sum over all sizes and fees.

    ## Example

    Increasing position:
    ```python-repl
    >>> import vectorbt as vbt
    >>> import pandas as pd

    >>> vbt.Portfolio.from_orders(
    ...     pd.Series([1., 2., 3., 4., 5.]),
    ...     pd.Series([1., 1., 1., 1., -4.]),
    ...     fixed_fees=1.).trades().records
       id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
    0   0    0   4.0          0          2.5         4.0         4         5.0

       exit_fees  pnl  return  direction  status  position_id
    0        1.0  5.0     0.5          0       1            0
    ```

    Decreasing position:
    ```python-repl
    >>> vbt.Portfolio.from_orders(
    ...     pd.Series([1., 2., 3., 4., 5.]),
    ...     pd.Series([4., -1., -1., -1., -1.]),
    ...     fixed_fees=1.).trades().records
       id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
    0   0    0   1.0          0          1.0        0.25         1         2.0
    1   1    0   1.0          0          1.0        0.25         2         3.0
    2   2    0   1.0          0          1.0        0.25         3         4.0
    3   3    0   1.0          0          1.0        0.25         4         5.0

       exit_fees   pnl  return  direction  status  position_id
    0        1.0 -0.25   -0.25          0       1            0
    1        1.0  0.75    0.75          0       1            0
    2        1.0  1.75    1.75          0       1            0
    3        1.0  2.75    2.75          0       1            0
    ```

    Multiple reversing positions:
    ```python-repl
    >>> vbt.Portfolio.from_orders(
    ...     pd.Series([1., 2., 3., 4., 5.]),
    ...     pd.Series([1., -2., 2., -2., 1.]),
    ...     fixed_fees=1.).trades().records
       id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
    0   0    0   1.0          0          1.0         1.0         1         2.0
    1   1    0   1.0          1          2.0         0.5         2         3.0
    2   2    0   1.0          2          3.0         0.5         3         4.0
    3   3    0   1.0          3          4.0         0.5         4         5.0

       exit_fees  pnl  return  direction  status  position_id
    0        0.5 -0.5  -0.500          0       1            0
    1        0.5 -2.0  -1.000          1       1            1
    2        0.5  0.0   0.000          0       1            2
    3        1.0 -2.5  -0.625          1       1            3
    ```

    Get count and PnL of trades:
    ```python-repl
    >>> price = pd.Series([1., 2., 3., 4., 3., 2., 1.])
    >>> orders = pd.Series([1., -0.5, -0.5, 2., -0.5, -0.5, -0.5])
    >>> portfolio = vbt.Portfolio.from_orders(price, orders)

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

    Get count and PnL of trades with duration of more than 2 days:
    ```python-repl
    >>> mask = (trades.records['exit_idx'] - trades.records['entry_idx']) > 2
    >>> trades_filtered = trades.filter_by_mask(mask)
    >>> trades_filtered.count()
    2
    >>> trades_filtered.pnl.sum()
    -3.0
    ```
    """

    def __init__(self, wrapper, records_arr, close, idx_field='exit_idx',
                 trade_type=TradeType.Trade, **kwargs):
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field,
            close=close,
            trade_type=trade_type,
            **kwargs
        )
        self._close = broadcast_to(close, wrapper.dummy(group_by=False))
        self._trade_type = trade_type

        if trade_type == TradeType.Trade:
            if not all(field in records_arr.dtype.names for field in trade_dt.names):
                raise TypeError("Records array must match trade_dt")
        else:
            if not all(field in records_arr.dtype.names for field in position_dt.names):
                raise TypeError("Records array must match position_dt")

    def _indexing_func_meta(self, pd_indexing_func):
        """Perform indexing on `Trades` and also return metadata."""
        new_wrapper, new_records_arr, group_idxs, col_idxs = \
            Records._indexing_func_meta(self, pd_indexing_func)
        new_close = new_wrapper.wrap(to_2d(self.close, raw=True)[:, col_idxs], group_by=False)
        return self.copy(
            wrapper=new_wrapper,
            records_arr=new_records_arr,
            close=new_close
        ), group_idxs, col_idxs

    def _indexing_func(self, pd_indexing_func):
        """Perform indexing on `Trades`."""
        return self._indexing_func_meta(pd_indexing_func)[0]

    @property
    def close(self):
        """Reference price such as close."""
        return self._close

    @property
    def trade_type(self):
        """Trade type."""
        return self._trade_type

    @classmethod
    def from_orders(cls, orders, **kwargs):
        """Build `Trades` from `vectorbt.portfolio.orders.Orders`."""
        trade_records_arr = nb.orders_to_trades_nb(
            orders.close.vbt.to_2d_array(),
            orders.values,
            orders.col_mapper.col_map
        )
        return cls(orders.wrapper, trade_records_arr, orders.close, **kwargs)

    @property  # no need for cached
    def records_readable(self):
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame()
        _id_str = 'Trade Id' if self.trade_type == TradeType.Trade else 'Position Id'
        out[_id_str] = records_df['id']
        out['Column'] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        out['Size'] = records_df['size']
        out['Entry Date'] = records_df['entry_idx'].map(lambda x: self.wrapper.index[x])
        out['Avg. Entry Price'] = records_df['entry_price']
        out['Entry Fees'] = records_df['entry_fees']
        out['Exit Date'] = records_df['exit_idx'].map(lambda x: self.wrapper.index[x])
        out['Avg. Exit Price'] = records_df['exit_price']
        out['Exit Fees'] = records_df['exit_fees']
        out['PnL'] = records_df['pnl']
        out['Return'] = records_df['return']
        out['Direction'] = records_df['direction'].map(to_value_map(TradeDirection))
        out['Status'] = records_df['status'].map(to_value_map(TradeStatus))
        if self.trade_type == TradeType.Trade:
            out['Position Id'] = records_df['position_id']
        return out

    @cached_property
    def duration(self):
        """Duration of each trade (in raw format)."""
        return self.map(nb.trade_duration_map_nb)

    @cached_property
    def pnl(self):
        """PnL of each trade."""
        return self.map_field('pnl')

    @cached_property
    def returns(self):
        """Return of each trade."""
        return self.map_field('return')

    # ############# PnL ############# #

    @cached_property
    def winning(self):
        """Winning trades."""
        filter_mask = self.values['pnl'] > 0.
        return self.filter_by_mask(filter_mask)

    @cached_method
    def win_rate(self, group_by=None, **kwargs):
        """Rate of winning trades."""
        win_count = to_1d(self.winning.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(win_count / total_count, group_by=group_by, **kwargs)

    @cached_property
    def losing(self):
        """Losing trades."""
        filter_mask = self.values['pnl'] < 0.
        return self.filter_by_mask(filter_mask)

    @cached_method
    def loss_rate(self, group_by=None, **kwargs):
        """Rate of losing trades."""
        loss_count = to_1d(self.losing.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(loss_count / total_count, group_by=group_by, **kwargs)

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

    # ############# TradeDirection ############# #

    @cached_property
    def direction(self):
        """See `vectorbt.portfolio.enums.TradeDirection`."""
        return self.map_field('direction')

    @cached_property
    def long(self):
        """Long trades."""
        filter_mask = self.values['direction'] == TradeDirection.Long
        return self.filter_by_mask(filter_mask)

    @cached_method
    def long_rate(self, group_by=None, **kwargs):
        """Rate of long trades."""
        long_count = to_1d(self.long.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(long_count / total_count, group_by=group_by, **kwargs)

    @cached_property
    def short(self):
        """Short trades."""
        filter_mask = self.values['direction'] == TradeDirection.Short
        return self.filter_by_mask(filter_mask)

    @cached_method
    def short_rate(self, group_by=None, **kwargs):
        """Rate of short trades."""
        short_count = to_1d(self.short.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(short_count / total_count, group_by=group_by, **kwargs)

    # ############# TradeStatus ############# #

    @cached_property
    def status(self):
        """See `vectorbt.portfolio.enums.TradeStatus`."""
        return self.map_field('status')

    @cached_property
    def open(self):
        """Open trades."""
        filter_mask = self.values['status'] == TradeStatus.Open
        return self.filter_by_mask(filter_mask)

    @cached_method
    def open_rate(self, group_by=None, **kwargs):
        """Rate of open trades."""
        open_count = to_1d(self.open.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(open_count / total_count, group_by=group_by, **kwargs)

    @cached_property
    def closed(self):
        """Closed trades."""
        filter_mask = self.values['status'] == TradeStatus.Closed
        return self.filter_by_mask(filter_mask)

    @cached_method
    def closed_rate(self, group_by=None, **kwargs):
        """Rate of closed trades."""
        closed_count = to_1d(self.closed.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(closed_count / total_count, group_by=group_by, **kwargs)

    # ############# Plotting ############# #

    def plot_pnl(self,
                 column=None,
                 marker_size_range=[7, 14],
                 opacity_range=[0.75, 0.9],
                 closed_profit_trace_kwargs=None,
                 closed_loss_trace_kwargs=None,
                 open_trace_kwargs=None,
                 hline_shape_kwargs=None,
                 row=None, col=None,
                 xref='x', yref='y',
                 fig=None,
                 **layout_kwargs):  # pragma: no cover
        """Plot trade PnL.

        Args:
            column (str): Name of the column to plot.
            closed_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Profit" markers.
            closed_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Loss" markers.
            open_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Open" markers.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            row (int): Row position.
            col (int): Column position.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> trades.plot_pnl()
        ```

        ![](/vectorbt/docs/img/trades_plot_pnl.png)
        """
        from vectorbt.settings import contrast_color_schema

        self_col = self.select_series(column=column, group_by=False)


        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        marker_size_range = tuple(marker_size_range)

        if fig is None:
            fig = CustomFigureWidget()
        fig.update_layout(**layout_kwargs)
        x_domain = [0, 1]
        xaxis = 'xaxis' + xref[1:]
        if xaxis in fig.layout:
            if 'domain' in fig.layout[xaxis]:
                if fig.layout[xaxis]['domain'] is not None:
                    x_domain = fig.layout[xaxis]['domain']

        if len(self_col.values) > 0:
            # Extract information
            _id = self.values['id']
            _id_str = 'Trade Id' if self.trade_type == TradeType.Trade else 'Position Id'
            exit_idx = self.values['exit_idx']
            pnl = self.values['pnl']
            returns = self.values['return']
            status = self.values['status']

            neutral_mask = pnl == 0
            profit_mask = pnl > 0
            loss_mask = pnl < 0

            marker_size = min_rel_rescale(np.abs(returns), marker_size_range)
            opacity = max_rel_rescale(np.abs(returns), opacity_range)

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            if np.any(closed_profit_mask):
                # Plot Profit markers
                profit_scatter = go.Scatter(
                    x=self_col.wrapper.index[exit_idx[closed_profit_mask]],
                    y=pnl[closed_profit_mask],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        color=contrast_color_schema['green'],
                        size=marker_size[closed_profit_mask],
                        opacity=opacity[closed_profit_mask],
                        line=dict(
                            width=1,
                            color=adjust_lightness(contrast_color_schema['green'])
                        ),
                    ),
                    name='Closed - Profit',
                    customdata=np.stack((_id[closed_profit_mask], returns[closed_profit_mask]), axis=1),
                    hovertemplate=_id_str + ": %{customdata[0]}"
                                            "<br>Date: %{x}"
                                            "<br>PnL: %{y}"
                                            "<br>Return: %{customdata[1]:.2%}"
                )
                profit_scatter.update(**closed_profit_trace_kwargs)
                fig.add_trace(profit_scatter, row=row, col=col)

            if np.any(closed_loss_mask):
                # Plot Loss markers
                loss_scatter = go.Scatter(
                    x=self_col.wrapper.index[exit_idx[closed_loss_mask]],
                    y=pnl[closed_loss_mask],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        color=contrast_color_schema['red'],
                        size=marker_size[closed_loss_mask],
                        opacity=opacity[closed_loss_mask],
                        line=dict(
                            width=1,
                            color=adjust_lightness(contrast_color_schema['red'])
                        )
                    ),
                    name='Closed - Loss',
                    customdata=np.stack((_id[closed_loss_mask], returns[closed_loss_mask]), axis=1),
                    hovertemplate=_id_str + ": %{customdata[0]}"
                                            "<br>Date: %{x}"
                                            "<br>PnL: %{y}"
                                            "<br>Return: %{customdata[1]:.2%}"
                )
                loss_scatter.update(**closed_loss_trace_kwargs)
                fig.add_trace(loss_scatter, row=row, col=col)

            if np.any(open_mask):
                # Plot Active markers
                active_scatter = go.Scatter(
                    x=self_col.wrapper.index[exit_idx[open_mask]],
                    y=pnl[open_mask],
                    mode='markers',
                    marker=dict(
                        symbol='circle',
                        color=contrast_color_schema['orange'],
                        size=marker_size[open_mask],
                        opacity=opacity[open_mask],
                        line=dict(
                            width=1,
                            color=adjust_lightness(contrast_color_schema['orange'])
                        )
                    ),
                    name='Open',
                    customdata=np.stack((_id[open_mask], returns[open_mask]), axis=1),
                    hovertemplate=_id_str + ": %{customdata[0]}"
                                            "<br>Date: %{x}"
                                            "<br>PnL: %{y}"
                                            "<br>Return: %{customdata[1]:.2%}"
                )
                active_scatter.update(**open_trace_kwargs)
                fig.add_trace(active_scatter, row=row, col=col)

        # Plot zeroline
        fig.add_shape(**merge_dicts(dict(
            type='line',
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=0,
            x1=x_domain[1],
            y1=0,
            line=dict(
                color="gray",
                dash="dash",
            )
        ), hline_shape_kwargs))
        return fig

    def plot(self,
             column=None,
             plot_close=True,
             plot_zones=True,
             close_trace_kwargs=None,
             entry_trace_kwargs=None,
             exit_trace_kwargs=None,
             exit_profit_trace_kwargs=None,
             exit_loss_trace_kwargs=None,
             active_trace_kwargs=None,
             profit_shape_kwargs=None,
             loss_shape_kwargs=None,
             row=None, col=None,
             xref='x', yref='y',
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `Trades.close`.
            plot_zones (bool): Whether to plot zones.

                Set to False if there are many trades within one position.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Trades.close`.
            entry_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Entry" markers.
            exit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit" markers.
            exit_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Profit" markers.
            exit_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Exit - Loss" markers.
            active_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Active" markers.
            profit_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for profit zones.
            loss_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for loss zones.
            row (int): Row position.
            col (int): Column position.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> trades.plot()
        ```

        ![](/vectorbt/docs/img/trades_plot.png)"""
        from vectorbt.settings import color_schema, contrast_color_schema

        self_col = self.select_series(column=column, group_by=False)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(dict(
            line_color=color_schema['blue'],
            name='Close' if self_col.wrapper.name is None else self_col.wrapper.name
        ), close_trace_kwargs)
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

        if fig is None:
            fig = CustomFigureWidget()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if plot_close:
            fig = self_col.close.vbt.plot(trace_kwargs=close_trace_kwargs, row=row, col=col, fig=fig)

        if len(self_col.values) > 0:
            # Extract information
            _id = self_col.values['id']
            _id_str = 'Trade Id' if self.trade_type == TradeType.Trade else 'Position Id'
            size = self_col.values['size']
            entry_idx = self_col.values['entry_idx']
            entry_price = self_col.values['entry_price']
            entry_fees = self_col.values['entry_fees']
            exit_idx = self_col.values['exit_idx']
            exit_price = self_col.values['exit_price']
            exit_fees = self_col.values['exit_fees']
            pnl = self_col.values['pnl']
            ret = self_col.values['return']
            direction_value_map = to_value_map(TradeDirection)
            direction = self_col.values['direction']
            direction = np.vectorize(lambda x: str(direction_value_map[x]))(direction)
            status = self_col.values['status']

            def get_duration_str(from_idx, to_idx):
                if isinstance(self_col.wrapper.index, DatetimeTypes):
                    duration = self_col.wrapper.index[to_idx] - self_col.wrapper.index[from_idx]
                elif self_col.wrapper.freq is not None:
                    duration = self_col.wrapper.to_time_units(to_idx - from_idx)
                else:
                    duration = to_idx - from_idx
                return np.vectorize(str)(duration)

            duration = get_duration_str(entry_idx, exit_idx)

            if len(entry_idx) > 0:
                # Plot Entry markers
                entry_customdata = np.stack((
                    _id,
                    size,
                    entry_fees,
                    direction,
                    *((self_col.values['position_id'],)
                      if self.trade_type == TradeType.Trade else ())
                ), axis=1)
                entry_scatter = go.Scatter(
                    x=self_col.wrapper.index[entry_idx],
                    y=entry_price,
                    mode='markers',
                    marker=dict(
                        symbol='square',
                        color=contrast_color_schema['blue'],
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(contrast_color_schema['blue'])
                        )
                    ),
                    name='Entry',
                    customdata=entry_customdata,
                    hovertemplate=_id_str + ": %{customdata[0]}"
                                            "<br>Date: %{x}"
                                            "<br>Avg. Price: %{y}"
                                            "<br>Size: %{customdata[1]:.6f}"
                                            "<br>Fees: %{customdata[2]:.6f}"
                                            "<br>Direction: %{customdata[3]}"
                                  + ("<br>Position Id: %{customdata[4]}"
                                     if self.trade_type == TradeType.Trade else '')
                )
                entry_scatter.update(**entry_trace_kwargs)
                fig.add_trace(entry_scatter, row=row, col=col)

            # Plot end markers
            def _plot_end_markers(mask, name, color, kwargs):
                if np.any(mask):
                    customdata = np.stack((
                        _id[mask],
                        duration[mask],
                        size[mask],
                        exit_fees[mask],
                        pnl[mask],
                        ret[mask],
                        direction[mask],
                        *((self_col.values['position_id'][mask],)
                          if self.trade_type == TradeType.Trade else ())
                    ), axis=1)
                    scatter = go.Scatter(
                        x=self_col.wrapper.index[exit_idx[mask]],
                        y=exit_price[mask],
                        mode='markers',
                        marker=dict(
                            symbol='square',
                            color=color,
                            size=7,
                            line=dict(
                                width=1,
                                color=adjust_lightness(color)
                            )
                        ),
                        name=name,
                        customdata=customdata,
                        hovertemplate=_id_str + ": %{customdata[0]}"
                                                "<br>Date: %{x}"
                                                "<br>Duration: %{customdata[1]}"
                                                "<br>Avg. Price: %{y}"
                                                "<br>Size: %{customdata[2]:.6f}"
                                                "<br>Fees: %{customdata[3]:.6f}"
                                                "<br>PnL: %{customdata[4]:.6f}"
                                                "<br>Return: %{customdata[5]:.2%}"
                                                "<br>Direction: %{customdata[6]}"
                                      + ("<br>Position Id: %{customdata[7]}"
                                         if self.trade_type == TradeType.Trade else '')
                    )
                    scatter.update(**kwargs)
                    fig.add_trace(scatter, row=row, col=col)

            # Plot Exit markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl == 0.),
                'Exit',
                contrast_color_schema['gray'],
                exit_trace_kwargs
            )

            # Plot Exit - Profit markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl > 0.),
                'Exit - Profit',
                contrast_color_schema['green'],
                exit_profit_trace_kwargs
            )

            # Plot Exit - Loss markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl < 0.),
                'Exit - Loss',
                contrast_color_schema['red'],
                exit_loss_trace_kwargs
            )

            # Plot Active markers
            _plot_end_markers(
                status == TradeStatus.Open,
                'Active',
                contrast_color_schema['orange'],
                active_trace_kwargs
            )

            if plot_zones:
                profit_mask = pnl > 0.
                if np.any(profit_mask):
                    # Plot profit zones
                    for i in np.flatnonzero(profit_mask):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref=yref,
                            x0=self_col.wrapper.index[entry_idx[i]],
                            y0=entry_price[i],
                            x1=self_col.wrapper.index[exit_idx[i]],
                            y1=exit_price[i],
                            fillcolor='green',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), profit_shape_kwargs))

                loss_mask = pnl < 0.
                if np.any(loss_mask):
                    # Plot loss zones
                    for i in np.flatnonzero(loss_mask):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref=yref,
                            x0=self_col.wrapper.index[entry_idx[i]],
                            y0=entry_price[i],
                            x1=self_col.wrapper.index[exit_idx[i]],
                            y1=exit_price[i],
                            fillcolor='red',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), loss_shape_kwargs))

        return fig


# ############# Positions ############# #


class Positions(Trades):
    """Extends `Trades` for working with position records.

    In vectorbt, a position aggregates one or multiple trades sharing the same column
    and position index. It has the same layout as a trade.

    ## Example

    Increasing position:
    ```python-repl
    >>> import vectorbt as vbt
    >>> import pandas as pd

    >>> vbt.Portfolio.from_orders(
    ...     pd.Series([1., 2., 3., 4., 5.]),
    ...     pd.Series([1., 1., 1., 1., -4.]),
    ...     fixed_fees=1.).positions().records
       id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
    0   0    0   4.0          0          2.5         4.0         4         5.0

       exit_fees  pnl  return  direction  status
    0        1.0  5.0     0.5          0       1
    ```

    Decreasing position:
    ```python-repl
    >>> vbt.Portfolio.from_orders(
    ...     pd.Series([1., 2., 3., 4., 5.]),
    ...     pd.Series([4., -1., -1., -1., -1.]),
    ...     fixed_fees=1.).positions().records
       id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
    0   0    0   4.0          0          1.0         1.0         4         3.5

       exit_fees  pnl  return  direction  status
    0        4.0  5.0    1.25          0       1
    ```

    Multiple positions:
    ```python-repl
    >>> vbt.Portfolio.from_orders(
    ...     pd.Series([1., 2., 3., 4., 5.]),
    ...     pd.Series([1., -2., 2., -2., 1.]),
    ...     fixed_fees=1.).positions().records
       id  col  size  entry_idx  entry_price  entry_fees  exit_idx  exit_price  \\
    0   0    0   1.0          0          1.0         1.0         1         2.0
    1   1    0   1.0          1          2.0         0.5         2         3.0
    2   2    0   1.0          2          3.0         0.5         3         4.0
    3   3    0   1.0          3          4.0         0.5         4         5.0

       exit_fees  pnl  return  direction  status
    0        0.5 -0.5  -0.500          0       1
    1        0.5 -2.0  -1.000          1       1
    2        0.5  0.0   0.000          0       1
    3        1.0 -2.5  -0.625          1       1
    ```
    """

    def __init__(self, *args, trade_type=TradeType.Position, **kwargs):
        if trade_type != TradeType.Position:
            raise ValueError("Trade type must be TradeType.Position")
        Trades.__init__(self, *args, trade_type=trade_type, **kwargs)

    @classmethod
    def from_orders(cls, orders, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_trades(cls, trades, **kwargs):
        """Build `Positions` from `Trades`."""
        position_records_arr = nb.trades_to_positions_nb(trades.values, trades.col_mapper.col_map)
        return cls(trades.wrapper, position_records_arr, trades.close, **kwargs)

    @cached_method
    def coverage(self, group_by=None, **kwargs):
        """Coverage, that is, total duration divided by the whole period."""
        total_duration = to_1d(self.duration.sum(group_by=group_by), raw=True)
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        return self.wrapper.wrap_reduced(total_duration / total_steps, group_by=group_by, **kwargs)
