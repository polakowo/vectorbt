"""Base class for working with trade records.

Class `Trades` wraps trade records and the corresponding time series
(such as open or close) to analyze trades. Use `vectorbt.portfolio.trades.Trades.from_orders`
to generate trade records from order records. This is done automatically in the
`vectorbt.portfolio.base.Portfolio` class, available as `vectorbt.portfolio.base.Portfolio.trades`.

Class `Positions` has the same properties as trades and is also
provided by `vectorbt.portfolio.base.Portfolio` as `vectorbt.portfolio.base.Portfolio.positions`.

!!! warning
    Both record types return both closed AND open trades, which may skew your performance results.
    To only consider closed trades, you should explicitly query `closed` attribute."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt import _typing as tp
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.datetime import DatetimeIndexes
from vectorbt.utils.enum import enum_to_value_map
from vectorbt.utils.figure import make_figure, get_domain
from vectorbt.utils.array import min_rel_rescale, max_rel_rescale
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast_to
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.records.base import Records
from vectorbt.records.mapped_array import MappedArray
from vectorbt.portfolio.enums import TradeDirection, TradeStatus, trade_dt, position_dt, TradeType
from vectorbt.portfolio import nb
from vectorbt.portfolio.orders import Orders

# ############# Trades ############# #

TradesT = tp.TypeVar("TradesT", bound="Trades")


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
    ...     fixed_fees=1.).trades.records
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
    ...     fixed_fees=1.).trades.records
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
    ...     fixed_fees=1.).trades.records
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

    >>> trades = vbt.Trades.from_orders(portfolio.orders)
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

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 close: tp.ArrayLike,
                 idx_field: str = 'exit_idx',
                 trade_type: int = TradeType.Trade,
                 **kwargs) -> None:
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

    def indexing_func_meta(self: TradesT, pd_indexing_func: tp.PandasIndexingFunc,
                           **kwargs) -> tp.Tuple[TradesT, tp.MaybeArray, tp.Array1d]:
        """Perform indexing on `Trades` and also return metadata."""
        new_wrapper, new_records_arr, group_idxs, col_idxs = \
            Records.indexing_func_meta(self, pd_indexing_func, **kwargs)
        new_close = new_wrapper.wrap(to_2d(self.close, raw=True)[:, col_idxs], group_by=False)
        return self.copy(
            wrapper=new_wrapper,
            records_arr=new_records_arr,
            close=new_close
        ), group_idxs, col_idxs

    def indexing_func(self: TradesT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> TradesT:
        """Perform indexing on `Trades`."""
        return self.indexing_func_meta(pd_indexing_func, **kwargs)[0]

    @property
    def close(self) -> tp.SeriesFrame:
        """Reference price such as close."""
        return self._close

    @property
    def trade_type(self) -> int:
        """Trade type."""
        return self._trade_type

    @classmethod
    def from_orders(cls: tp.Type[TradesT], orders: Orders, **kwargs) -> TradesT:
        """Build `Trades` from `vectorbt.portfolio.orders.Orders`."""
        trade_records_arr = nb.orders_to_trades_nb(
            orders.close.vbt.to_2d_array(),
            orders.values,
            orders.col_mapper.col_map
        )
        return cls(orders.wrapper, trade_records_arr, orders.close, **kwargs)

    @property  # no need for cached
    def records_readable(self) -> tp.Frame:
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
        out['Direction'] = records_df['direction'].map(enum_to_value_map(TradeDirection))
        out['Status'] = records_df['status'].map(enum_to_value_map(TradeStatus))
        if self.trade_type == TradeType.Trade:
            out['Position Id'] = records_df['position_id']
        return out

    @cached_property
    def duration(self) -> MappedArray:
        """Duration of each trade (in raw format)."""
        return self.map(nb.trade_duration_map_nb)

    @cached_property
    def pnl(self) -> MappedArray:
        """PnL of each trade."""
        return self.map_field('pnl')

    @cached_property
    def returns(self) -> MappedArray:
        """Return of each trade."""
        return self.map_field('return')

    # ############# PnL ############# #

    @cached_property
    def winning(self: TradesT) -> TradesT:
        """Winning trades."""
        filter_mask = self.values['pnl'] > 0.
        return self.filter_by_mask(filter_mask)

    @cached_method
    def win_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of winning trades."""
        win_count = to_1d(self.winning.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='win_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(win_count / total_count, group_by=group_by, **wrap_kwargs)

    @cached_property
    def losing(self: TradesT) -> TradesT:
        """Losing trades."""
        filter_mask = self.values['pnl'] < 0.
        return self.filter_by_mask(filter_mask)

    @cached_method
    def loss_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of losing trades."""
        loss_count = to_1d(self.losing.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='loss_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(loss_count / total_count, group_by=group_by, **wrap_kwargs)

    @cached_method
    def profit_factor(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Profit factor."""
        total_win = to_1d(self.winning.pnl.sum(group_by=group_by), raw=True)
        total_loss = to_1d(self.losing.pnl.sum(group_by=group_by), raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = to_1d(self.count(group_by=group_by), raw=True) > 0
        total_win[np.isnan(total_win) & has_values] = 0.
        total_loss[np.isnan(total_loss) & has_values] = 0.

        profit_factor = total_win / np.abs(total_loss)
        wrap_kwargs = merge_dicts(dict(name_or_index='profit_factor'), wrap_kwargs)
        return self.wrapper.wrap_reduced(profit_factor, group_by=group_by, **wrap_kwargs)

    @cached_method
    def expectancy(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Average profitability."""
        win_rate = to_1d(self.win_rate(group_by=group_by), raw=True)
        avg_win = to_1d(self.winning.pnl.mean(group_by=group_by), raw=True)
        avg_loss = to_1d(self.losing.pnl.mean(group_by=group_by), raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = to_1d(self.count(group_by=group_by), raw=True) > 0
        avg_win[np.isnan(avg_win) & has_values] = 0.
        avg_loss[np.isnan(avg_loss) & has_values] = 0.

        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        wrap_kwargs = merge_dicts(dict(name_or_index='expectancy'), wrap_kwargs)
        return self.wrapper.wrap_reduced(expectancy, group_by=group_by, **wrap_kwargs)

    @cached_method
    def sqn(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """System Quality Number (SQN)."""
        count = to_1d(self.count(group_by=group_by), raw=True)
        pnl_mean = to_1d(self.pnl.mean(group_by=group_by), raw=True)
        pnl_std = to_1d(self.pnl.std(group_by=group_by), raw=True)
        sqn = np.sqrt(count) * pnl_mean / pnl_std
        wrap_kwargs = merge_dicts(dict(name_or_index='sqn'), wrap_kwargs)
        return self.wrapper.wrap_reduced(sqn, group_by=group_by, **wrap_kwargs)

    # ############# TradeDirection ############# #

    @cached_property
    def direction(self) -> MappedArray:
        """See `vectorbt.portfolio.enums.TradeDirection`."""
        return self.map_field('direction')

    @cached_property
    def long(self: TradesT) -> TradesT:
        """Long trades."""
        filter_mask = self.values['direction'] == TradeDirection.Long
        return self.filter_by_mask(filter_mask)

    @cached_method
    def long_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of long trades."""
        long_count = to_1d(self.long.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='long_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(long_count / total_count, group_by=group_by, **wrap_kwargs)

    @cached_property
    def short(self: TradesT) -> TradesT:
        """Short trades."""
        filter_mask = self.values['direction'] == TradeDirection.Short
        return self.filter_by_mask(filter_mask)

    @cached_method
    def short_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of short trades."""
        short_count = to_1d(self.short.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='short_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(short_count / total_count, group_by=group_by, **wrap_kwargs)

    # ############# TradeStatus ############# #

    @cached_property
    def status(self) -> MappedArray:
        """See `vectorbt.portfolio.enums.TradeStatus`."""
        return self.map_field('status')

    @cached_property
    def open(self: TradesT) -> TradesT:
        """Open trades."""
        filter_mask = self.values['status'] == TradeStatus.Open
        return self.filter_by_mask(filter_mask)

    @cached_method
    def open_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of open trades."""
        open_count = to_1d(self.open.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='open_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(open_count / total_count, group_by=group_by, **wrap_kwargs)

    @cached_property
    def closed(self: TradesT) -> TradesT:
        """Closed trades."""
        filter_mask = self.values['status'] == TradeStatus.Closed
        return self.filter_by_mask(filter_mask)

    @cached_method
    def closed_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of closed trades."""
        closed_count = to_1d(self.closed.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='closed_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(closed_count / total_count, group_by=group_by, **wrap_kwargs)

    # ############# Plotting ############# #

    def plot_pnl_returns(self,
                         column: tp.Optional[tp.Label] = None,
                         as_pct: bool = True,
                         marker_size_range: tp.Tuple[float, float] = (7, 14),
                         opacity_range: tp.Tuple[float, float] = (0.75, 0.9),
                         closed_profit_trace_kwargs: tp.KwargsLike = None,
                         closed_loss_trace_kwargs: tp.KwargsLike = None,
                         open_trace_kwargs: tp.KwargsLike = None,
                         hline_shape_kwargs: tp.KwargsLike = None,
                         add_trace_kwargs: tp.KwargsLike = None,
                         xref: str = 'x',
                         yref: str = 'y',
                         fig: tp.Optional[tp.BaseFigure] = None,
                         **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot trade PnL.

        Args:
            column (str): Name of the column to plot.
            as_pct (bool): Whether to set y-axis to `Trades.returns`, otherwise to `Trades.pnl`.
            marker_size_range (tuple): Range of marker size.
            opacity_range (tuple): Range of marker opacity.
            closed_profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Profit" markers.
            closed_loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Closed - Loss" markers.
            open_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Open" markers.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zeroline.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        self_col = self.select_one(column=column, group_by=False)

        if closed_profit_trace_kwargs is None:
            closed_profit_trace_kwargs = {}
        if closed_loss_trace_kwargs is None:
            closed_loss_trace_kwargs = {}
        if open_trace_kwargs is None:
            open_trace_kwargs = {}
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        marker_size_range = tuple(marker_size_range)
        xaxis = 'xaxis' + xref[1:]
        yaxis = 'yaxis' + yref[1:]

        if fig is None:
            fig = make_figure()
        if as_pct:
            _layout_kwargs = dict()
            _layout_kwargs[yaxis] = dict(tickformat='.2%')
            fig.update_layout(**_layout_kwargs)
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)

        if len(self_col.values) > 0:
            # Extract information
            _pnl_str = '%{customdata[1]:.6f}' if as_pct else '%{y}'
            _return_str = '%{y}' if as_pct else '%{customdata[1]:.2%}'
            exit_idx = self_col.values['exit_idx']
            pnl = self_col.values['pnl']
            returns = self_col.values['return']
            status = self_col.values['status']

            neutral_mask = pnl == 0
            profit_mask = pnl > 0
            loss_mask = pnl < 0

            marker_size = min_rel_rescale(np.abs(returns), marker_size_range)
            opacity = max_rel_rescale(np.abs(returns), opacity_range)

            open_mask = status == TradeStatus.Open
            closed_profit_mask = (~open_mask) & profit_mask
            closed_loss_mask = (~open_mask) & loss_mask
            open_mask &= ~neutral_mask

            def _plot_scatter(mask: tp.Array1d, name: tp.TraceName, color: tp.Any, kwargs: tp.Kwargs) -> None:
                if np.any(mask):
                    if self_col.trade_type == TradeType.Trade:
                        customdata = np.stack((
                            self_col.values['id'][mask],
                            self_col.values['position_id'][mask],
                            pnl[mask] if as_pct else returns[mask]
                        ), axis=1)
                        hovertemplate = "Trade Id: %{customdata[0]}" \
                                        "<br>Position Id: %{customdata[1]}" \
                                        "<br>Date: %{x}" \
                                        f"<br>PnL: {_pnl_str}" \
                                        f"<br>Return: {_return_str}"
                    else:
                        customdata = np.stack((
                            self_col.values['id'][mask],
                            pnl[mask] if as_pct else returns[mask]
                        ), axis=1)
                        hovertemplate = "Position Id: %{customdata[0]}" \
                                        "<br>Date: %{x}" \
                                        f"<br>PnL: {_pnl_str}" \
                                        f"<br>Return: {_return_str}"
                    scatter = go.Scatter(
                        x=self_col.wrapper.index[exit_idx[mask]],
                        y=returns[mask] if as_pct else pnl[mask],
                        mode='markers',
                        marker=dict(
                            symbol='circle',
                            color=color,
                            size=marker_size[mask],
                            opacity=opacity[mask],
                            line=dict(
                                width=1,
                                color=adjust_lightness(color)
                            ),
                        ),
                        name=name,
                        customdata=customdata,
                        hovertemplate=hovertemplate
                    )
                    scatter.update(**kwargs)
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_profit_mask,
                'Closed - Profit',
                plotting_cfg['contrast_color_schema']['green'],
                closed_profit_trace_kwargs
            )

            # Plot Closed - Profit scatter
            _plot_scatter(
                closed_loss_mask,
                'Closed - Loss',
                plotting_cfg['contrast_color_schema']['red'],
                closed_loss_trace_kwargs
            )

            # Plot Open scatter
            _plot_scatter(
                open_mask,
                'Open',
                plotting_cfg['contrast_color_schema']['orange'],
                open_trace_kwargs
            )

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

    def plot_pnl(self, **kwargs):
        """`Trades.plot_pnl_returns` with `as_pct` set to False.

        ## Example

        ```python-repl
        >>> trades.plot_pnl()
        ```

        ![](/vectorbt/docs/img/trades_plot_pnl.svg)"""
        return self.plot_pnl_returns(as_pct=False, **kwargs)

    def plot_returns(self, **kwargs):
        """`Trades.plot_pnl_returns` with `as_pct` set to True."""
        return self.plot_pnl_returns(as_pct=True, **kwargs)

    def plot(self,
             column: tp.Optional[tp.Label] = None,
             plot_close: bool = True,
             plot_zones: bool = True,
             close_trace_kwargs: tp.KwargsLike = None,
             entry_trace_kwargs: tp.KwargsLike = None,
             exit_trace_kwargs: tp.KwargsLike = None,
             exit_profit_trace_kwargs: tp.KwargsLike = None,
             exit_loss_trace_kwargs: tp.KwargsLike = None,
             active_trace_kwargs: tp.KwargsLike = None,
             profit_shape_kwargs: tp.KwargsLike = None,
             loss_shape_kwargs: tp.KwargsLike = None,
             add_trace_kwargs: tp.KwargsLike = None,
             xref: str = 'x',
             yref: str = 'y',
             fig: tp.Optional[tp.BaseFigure] = None,
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
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
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> trades.plot()
        ```

        ![](/vectorbt/docs/img/trades_plot.svg)"""
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        self_col = self.select_one(column=column, group_by=False)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(dict(
            line=dict(
                color=plotting_cfg['color_schema']['blue']
            ),
            name='Close'
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
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if plot_close:
            fig = self_col.close.vbt.plot(trace_kwargs=close_trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)

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
            direction_value_map = enum_to_value_map(TradeDirection)
            direction = self_col.values['direction']
            direction = np.vectorize(lambda x: str(direction_value_map[x]))(direction)
            status = self_col.values['status']

            def _get_duration_str(from_idx: int, to_idx: int) -> tp.Array1d:
                if isinstance(self_col.wrapper.index, DatetimeIndexes):
                    duration = self_col.wrapper.index[to_idx] - self_col.wrapper.index[from_idx]
                elif self_col.wrapper.freq is not None:
                    duration = self_col.wrapper.to_time_units(to_idx - from_idx)
                else:
                    duration = to_idx - from_idx
                return np.vectorize(str)(duration)

            duration = _get_duration_str(entry_idx, exit_idx)

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
                        color=plotting_cfg['contrast_color_schema']['blue'],
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['blue'])
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
                fig.add_trace(entry_scatter, **add_trace_kwargs)

            # Plot end markers
            def _plot_end_markers(mask: tp.Array1d, name: tp.TraceName, color: tp.Any, kwargs: tp.Kwargs) -> None:
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
                    fig.add_trace(scatter, **add_trace_kwargs)

            # Plot Exit markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl == 0.),
                'Exit',
                plotting_cfg['contrast_color_schema']['gray'],
                exit_trace_kwargs
            )

            # Plot Exit - Profit markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl > 0.),
                'Exit - Profit',
                plotting_cfg['contrast_color_schema']['green'],
                exit_profit_trace_kwargs
            )

            # Plot Exit - Loss markers
            _plot_end_markers(
                (status == TradeStatus.Closed) & (pnl < 0.),
                'Exit - Loss',
                plotting_cfg['contrast_color_schema']['red'],
                exit_loss_trace_kwargs
            )

            # Plot Active markers
            _plot_end_markers(
                status == TradeStatus.Open,
                'Active',
                plotting_cfg['contrast_color_schema']['orange'],
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


PositionsT = tp.TypeVar("PositionsT", bound="Positions")


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
    ...     fixed_fees=1.).positions.records
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
    ...     fixed_fees=1.).positions.records
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
    ...     fixed_fees=1.).positions.records
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

    def __init__(self, *args, trade_type: int = TradeType.Position, **kwargs) -> None:
        if trade_type != TradeType.Position:
            raise ValueError("Trade type must be TradeType.Position")
        Trades.__init__(self, *args, trade_type=trade_type, **kwargs)

    @classmethod
    def from_orders(cls: tp.Type[PositionsT], orders: Orders, **kwargs) -> PositionsT:
        raise NotImplementedError

    @classmethod
    def from_trades(cls: tp.Type[PositionsT], trades: Trades, **kwargs) -> PositionsT:
        """Build `Positions` from `Trades`."""
        position_records_arr = nb.trades_to_positions_nb(trades.values, trades.col_mapper.col_map)
        return cls(trades.wrapper, position_records_arr, trades.close, **kwargs)

    @cached_method
    def coverage(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Coverage, that is, total duration divided by the whole period."""
        total_duration = to_1d(self.duration.sum(group_by=group_by), raw=True)
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        wrap_kwargs = merge_dicts(dict(name_or_index='coverage'), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_duration / total_steps, group_by=group_by, **wrap_kwargs)
