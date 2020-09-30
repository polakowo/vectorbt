"""Classes for working with order records."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.defaults import contrast_color_schema
from vectorbt.utils.decorators import cached_property
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.config import Configured
from vectorbt.base.indexing import PandasIndexer
from vectorbt.records.base import Records, indexing_on_records_meta
from vectorbt.records.enums import OrderSide, order_dt


def indexing_on_orders_meta(obj, pd_indexing_func):
    """Perform indexing on `BaseOrders`."""
    new_wrapper, new_records_arr, group_idxs, col_idxs = indexing_on_records_meta(obj, pd_indexing_func)
    new_ref_price = new_wrapper.wrap(obj.close.values[:, col_idxs], group_by=False)
    return obj.copy(
        wrapper=new_wrapper,
        records_arr=new_records_arr,
        close=new_ref_price
    ), group_idxs, col_idxs


def _indexing_func(obj, pd_indexing_func):
    """See `indexing_on_orders`."""
    return indexing_on_orders_meta(obj, pd_indexing_func)[0]


class BaseOrders(Records):
    """Extends `Records` for working with order records.

    Example:
        Get the total number of buy and sell operations:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> price = pd.Series([1., 2., 3., 2., 1.])
        >>> orders = pd.Series([1., 1., 1., 1., -1.])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders,
        ...      init_cash=100., freq='1D')

        >>> portfolio.orders().buy.count()
        4
        >>> portfolio.orders().sell.count()
        1
        ```"""

    def __init__(self, wrapper, records_arr, close, idx_field='idx'):
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

        if not all(field in records_arr.dtype.names for field in order_dt.names):
            raise ValueError("Records array must have all fields defined in order_dt")

        PandasIndexer.__init__(self, _indexing_func)

    def plot(self,
             column=None,
             ref_price_trace_kwargs=None,
             buy_trace_kwargs=None,
             sell_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            ref_price_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for main price.
            buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Sell" markers.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            portfolio.orders().plot()
            ```

            ![](/vectorbt/docs/img/orders.png)"""
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
        if buy_trace_kwargs is None:
            buy_trace_kwargs = {}
        if sell_trace_kwargs is None:
            sell_trace_kwargs = {}

        # Plot main price
        fig = self_col.close.vbt.plot(trace_kwargs=ref_price_trace_kwargs, fig=fig, **layout_kwargs)

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
                symbol='circle',
                color=contrast_color_schema['green'],
                size=7,
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
        fig.add_trace(buy_scatter)

        # Plot Sell markers
        sell_mask = side == OrderSide.Sell
        sell_customdata = np.stack((size[sell_mask], fees[sell_mask]), axis=1)
        sell_scatter = go.Scatter(
            x=self_col.wrapper.index[idx[sell_mask]],
            y=price[sell_mask],
            mode='markers',
            marker=dict(
                symbol='circle',
                color=contrast_color_schema['orange'],
                size=7,
                line=dict(
                    width=1,
                    color=adjust_lightness(contrast_color_schema['orange'])
                )
            ),
            name='Sell',
            customdata=sell_customdata,
            hovertemplate="%{x}<br>Price: %{y}<br>Size: %{customdata[0]:.4f}<br>Fees: %{customdata[1]:.4f}"
        )
        sell_scatter.update(**sell_trace_kwargs)
        fig.add_trace(sell_scatter)

        return fig

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


class Orders(BaseOrders):
    """Extends `BaseOrders` by further dividing orders into buy and sell orders."""

    @cached_property
    def side(self):
        """Side of each order.

        See `vectorbt.records.enums.OrderSide`."""
        return self.map_field('side')

    @cached_property
    def buy(self):
        """Buy operations of type `BaseOrders`."""
        filter_mask = self.records_arr['side'] == OrderSide.Buy
        return BaseOrders(
            self.wrapper,
            self.records_arr[filter_mask],
            self.close,
            idx_field=self.idx_field
        )

    @cached_property
    def sell(self):
        """Sell operations of type `BaseOrders`."""
        filter_mask = self.records_arr['side'] == OrderSide.Sell
        return BaseOrders(
            self.wrapper,
            self.records_arr[filter_mask],
            self.close,
            idx_field=self.idx_field
        )
