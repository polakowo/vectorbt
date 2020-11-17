"""Base class for working with order records."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.enum import to_value_map
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.utils.config import merge_kwargs
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.reshape_fns import to_1d
from vectorbt.records.base import Records, indexing_on_records_meta
from vectorbt.portfolio.enums import order_dt, OrderSide


def indexing_on_orders_meta(obj, pd_indexing_func):
    """Perform indexing on `Orders`."""
    new_wrapper, new_records_arr, group_idxs, col_idxs = indexing_on_records_meta(obj, pd_indexing_func)
    new_close = new_wrapper.wrap(obj.close.values[:, col_idxs], group_by=False)
    return obj.copy(
        wrapper=new_wrapper,
        records_arr=new_records_arr,
        close=new_close
    ), group_idxs, col_idxs


def orders_indexing_func(obj, pd_indexing_func):
    """See `indexing_on_orders`."""
    return indexing_on_orders_meta(obj, pd_indexing_func)[0]


class Orders(Records):
    """Extends `Records` for working with order records.

    Example:
        Get the total number of buy and sell operations:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> price = pd.Series([1., 2., 3., 2., 1.])
        >>> size = pd.Series([1., 1., 1., 1., -1.])
        >>> orders = vbt.Portfolio.from_orders(price, size).orders()

        >>> orders.buy.count()
        4
        >>> orders.sell.count()
        1
        ```"""

    def __init__(self, wrapper, records_arr, close, idx_field='idx', **kwargs):
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field,
            close=close,
            **kwargs
        )
        self.close = close

        if not all(field in records_arr.dtype.names for field in order_dt.names):
            raise ValueError("Records array must have all fields defined in order_dt")

        PandasIndexer.__init__(self, orders_indexing_func)

    @property  # no need for cached
    def records_readable(self):
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame()
        out['Column'] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        out['Date'] = records_df['idx'].map(lambda x: self.wrapper.index[x])
        out['Size'] = records_df['size']
        out['Price'] = records_df['price']
        out['Fees'] = records_df['fees']
        out['Side'] = records_df['side'].map(to_value_map(OrderSide))
        return out

    @cached_property
    def size(self):
        """Size of each order."""
        return self.map_field('size')

    @cached_property
    def price(self):
        """Price of each order."""
        return self.map_field('price')

    @cached_property
    def fees(self):
        """Fees paid for each order."""
        return self.map_field('fees')

    # ############# OrderSide ############# #

    @cached_property
    def side(self):
        """Side of each order.

        See `vectorbt.portfolio.enums.OrderSide`."""
        return self.map_field('side')

    @cached_property
    def buy(self):
        """Buy operations."""
        filter_mask = self.records_arr['side'] == OrderSide.Buy
        return self.filter_by_mask(filter_mask)

    @cached_method
    def buy_rate(self, group_by=None, **kwargs):
        """Rate of buy operations."""
        buy_count = to_1d(self.buy.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(buy_count / total_count, group_by=group_by, **kwargs)

    @cached_property
    def sell(self):
        """Sell operations."""
        filter_mask = self.records_arr['side'] == OrderSide.Sell
        return self.filter_by_mask(filter_mask)

    @cached_method
    def sell_rate(self, group_by=None, **kwargs):
        """Rate of sell operations."""
        sell_count = to_1d(self.sell.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(sell_count / total_count, group_by=group_by, **kwargs)

    # ############# Plotting ############# #

    def plot(self,
             column=None,
             show_close=True,
             close_trace_kwargs=None,
             buy_trace_kwargs=None,
             sell_trace_kwargs=None,
             row=None, col=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            show_close (bool): Whether to show `Orders.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Orders.close`.
            buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Sell" markers.
            row (int): Row position.
            col (int): Column position.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> orders.plot()
            ```

            ![](/vectorbt/docs/img/orders_plot.png)"""
        from vectorbt.defaults import layout, contrast_color_schema

        self_col = self.force_select_column(column)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_kwargs(dict(
            line_color=layout['colorway'][0],
            name='Close' if self_col.close.name is None else self_col.close.name
        ), close_trace_kwargs)
        if buy_trace_kwargs is None:
            buy_trace_kwargs = {}
        if sell_trace_kwargs is None:
            sell_trace_kwargs = {}

        if fig is None:
            fig = CustomFigureWidget()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if show_close:
            fig = self_col.close.vbt.plot(trace_kwargs=close_trace_kwargs, row=row, col=col, fig=fig)

        if len(self_col.records_arr) > 0:
            # Extract information
            idx = self_col.records_arr['idx']
            size = self_col.records_arr['size']
            price = self_col.records_arr['price']
            fees = self_col.records_arr['fees']
            side = self_col.records_arr['side']

            # Plot Buy markers
            buy_mask = side == OrderSide.Buy
            buy_customdata = np.stack((size[buy_mask], fees[buy_mask]), axis=1)
            buy_scatter = go.Scatter(
                x=self_col.wrapper.index[idx[buy_mask]],
                y=price[buy_mask],
                mode='markers',
                marker=dict(
                    symbol='triangle-up',
                    color=contrast_color_schema['green'],
                    size=8,
                    line=dict(
                        width=1,
                        color=adjust_lightness(contrast_color_schema['green'])
                    )
                ),
                name='Buy',
                customdata=buy_customdata,
                hovertemplate="%{x}<br>Price: %{y}<br>Size: %{customdata[0]:.4f}<br>Fees: %{customdata[1]:.4f}"
            )
            buy_scatter.update(**buy_trace_kwargs)
            fig.add_trace(buy_scatter, row=row, col=col)

            # Plot Sell markers
            sell_mask = side == OrderSide.Sell
            sell_customdata = np.stack((size[sell_mask], fees[sell_mask]), axis=1)
            sell_scatter = go.Scatter(
                x=self_col.wrapper.index[idx[sell_mask]],
                y=price[sell_mask],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    color=contrast_color_schema['red'],
                    size=8,
                    line=dict(
                        width=1,
                        color=adjust_lightness(contrast_color_schema['red'])
                    )
                ),
                name='Sell',
                customdata=sell_customdata,
                hovertemplate="%{x}<br>Price: %{y}<br>Size: %{customdata[0]:.4f}<br>Fees: %{customdata[1]:.4f}"
            )
            sell_scatter.update(**sell_trace_kwargs)
            fig.add_trace(sell_scatter, row=row, col=col)

        return fig
