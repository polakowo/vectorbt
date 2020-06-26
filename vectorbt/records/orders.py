"""Classes for working with order records."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.defaults import contrast_color_schema
from vectorbt.utils.decorators import cached_property
from vectorbt.utils.colors import adjust_lightness
from vectorbt.base.indexing import PandasIndexer
from vectorbt.tseries.common import TSArrayWrapper
from vectorbt.records.base import Records
from vectorbt.records.enums import OrderSide, order_dt
from vectorbt.records.common import indexing_on_records
from vectorbt.records import nb


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `BaseOrders`."""
    records_arr, _ = indexing_on_records(obj, pd_indexing_func)
    return obj.__class__(records_arr, pd_indexing_func(obj.main_price), freq=obj.wrapper.freq)


class BaseOrders(Records):
    """Extends `Records` for working with order records.

    Example:
        Get the total number of buy and sell operations:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, 1, 1, 1, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders,
        ...      init_capital=100, freq='1D')

        >>> print(portfolio.orders.buy.count)
        4.0
        >>> print(portfolio.orders.sell.count)
        1.0
        ```"""

    def __init__(self, records_arr, main_price, freq=None):
        Records.__init__(self, records_arr, TSArrayWrapper.from_obj(main_price, freq=freq))
        PandasIndexer.__init__(self, _indexing_func)

        if not all(field in records_arr.dtype.names for field in order_dt.names):
            raise Exception("Records array must have all fields defined in order_dt")

        self.main_price = main_price

    def plot(self, main_price_trace_kwargs={}, buy_trace_kwargs={}, sell_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot orders.

        Args:
            main_price_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for main price.
            buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Sell" markers.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            portfolio.orders.plot()
            ```

            ![](/vectorbt/docs/img/orders.png)"""
        if self.wrapper.ndim > 1:
            raise Exception("You must select a column first")

        # Plot main price
        fig = self.main_price.vbt.tseries.plot(trace_kwargs=main_price_trace_kwargs, fig=fig, **layout_kwargs)

        # Extract information
        idx = self.records_arr['idx']
        size = self.records_arr['size']
        price = self.records_arr['price']
        fees = self.records_arr['fees']
        side = self.records_arr['side']

        # Plot Buy markers
        buy_mask = side == OrderSide.Buy
        buy_customdata = np.stack((size[buy_mask], fees[buy_mask]), axis=1)
        buy_scatter = go.Scatter(
            x=self.wrapper.index[idx[buy_mask]],
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
            hovertemplate="%{x}<br>Price: %{y}<br>Size: %{customdata[0]}<br>Fees: %{customdata[1]}"
        )
        buy_scatter.update(**buy_trace_kwargs)
        fig.add_trace(buy_scatter)

        # Plot Sell markers
        sell_mask = side == OrderSide.Sell
        sell_customdata = np.stack((size[sell_mask], fees[sell_mask]), axis=1)
        sell_scatter = go.Scatter(
            x=self.wrapper.index[idx[sell_mask]],
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
            hovertemplate="%{x}<br>Price: %{y}<br>Size: %{customdata[0]}<br>Fees: %{customdata[1]}"
        )
        sell_scatter.update(**sell_trace_kwargs)
        fig.add_trace(sell_scatter)

        return fig

    @cached_property
    def size(self):
        """Size of each order."""
        return self.map_field_to_matrix('size')

    @cached_property
    def avg_size(self):
        """Average size of an order."""
        return self.map_reduce_records(nb.order_size_map_nb, nb.mean_reduce_nb)

    @cached_property
    def total_size(self):
        """Total size of all orders."""
        return self.map_reduce_records(nb.order_size_map_nb, nb.sum_reduce_nb)

    @cached_property
    def price(self):
        """Price of each order."""
        return self.map_field_to_matrix('price')

    @cached_property
    def avg_price(self):
        """Average price of an order."""
        return self.map_reduce_records(nb.order_price_map_nb, nb.mean_reduce_nb)

    @cached_property
    def fees(self):
        """Fees paid for each order."""
        return self.map_field_to_matrix('fees')

    @cached_property
    def avg_fees(self):
        """Average fees paid for an order."""
        return self.map_reduce_records(nb.order_fees_map_nb, nb.mean_reduce_nb)

    @cached_property
    def total_fees(self):
        """Total fees paid for all orders."""
        return self.map_reduce_records(nb.order_fees_map_nb, nb.sum_reduce_nb)


class Orders(BaseOrders):
    """Extends `BaseOrders` by further dividing orders into buy and sell orders."""

    @cached_property
    def side(self):
        """Side of each order.

        See `vectorbt.records.enums.OrderSide`."""
        return self.map_field_to_matrix('side')

    @cached_property
    def buy(self):
        """Buy operations of type `BaseOrders`."""
        filter_mask = self.records_arr['side'] == OrderSide.Buy
        return BaseOrders(self.records_arr[filter_mask], self.main_price, freq=self.wrapper.freq)

    @cached_property
    def sell(self):
        """Sell operations of type `BaseOrders`."""
        filter_mask = self.records_arr['side'] == OrderSide.Sell
        return BaseOrders(self.records_arr[filter_mask], self.main_price, freq=self.wrapper.freq)
