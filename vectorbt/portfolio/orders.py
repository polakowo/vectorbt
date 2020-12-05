"""Base class for working with order records."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.enum import to_value_map
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.utils.config import merge_dicts
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast_to
from vectorbt.records.base import Records
from vectorbt.portfolio.enums import order_dt, OrderSide


class Orders(Records):
    """Extends `Records` for working with order records.

    ## Example

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
    ```
    """

    def __init__(self, wrapper, records_arr, close, idx_field='idx', **kwargs):
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field,
            close=close,
            **kwargs
        )
        self._close = broadcast_to(close, wrapper.dummy(group_by=False))

        if not all(field in records_arr.dtype.names for field in order_dt.names):
            raise TypeError("Records array must match order_dt")

    def _indexing_func_meta(self, pd_indexing_func):
        """Perform indexing on `Orders` and return metadata."""
        new_wrapper, new_records_arr, group_idxs, col_idxs = \
            Records._indexing_func_meta(self, pd_indexing_func)
        new_close = new_wrapper.wrap(to_2d(self.close, raw=True)[:, col_idxs], group_by=False)
        return self.copy(
            wrapper=new_wrapper,
            records_arr=new_records_arr,
            close=new_close
        ), group_idxs, col_idxs

    def _indexing_func(self, pd_indexing_func):
        """Perform indexing on `Orders`."""
        return self._indexing_func_meta(pd_indexing_func)[0]

    @property
    def close(self):
        """Reference price such as close."""
        return self._close

    @property  # no need for cached
    def records_readable(self):
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame()
        out['Order Id'] = records_df['id']
        out['Date'] = records_df['idx'].map(lambda x: self.wrapper.index[x])
        out['Column'] = records_df['col'].map(lambda x: self.wrapper.columns[x])
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
        filter_mask = self.values['side'] == OrderSide.Buy
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
        filter_mask = self.values['side'] == OrderSide.Sell
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
             plot_close=True,
             close_trace_kwargs=None,
             buy_trace_kwargs=None,
             sell_trace_kwargs=None,
             row=None, col=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `Orders.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Orders.close`.
            buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Sell" markers.
            row (int): Row position.
            col (int): Column position.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> orders.plot()
        ```

        ![](/vectorbt/docs/img/orders_plot.png)"""
        from vectorbt.settings import color_schema, contrast_color_schema

        self_col = self.select_series(column=column, group_by=False)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(dict(
            line_color=color_schema['blue'],
            name='Close' if self_col.wrapper.name is None else self_col.wrapper.name
        ), close_trace_kwargs)
        if buy_trace_kwargs is None:
            buy_trace_kwargs = {}
        if sell_trace_kwargs is None:
            sell_trace_kwargs = {}

        if fig is None:
            fig = CustomFigureWidget()
        fig.update_layout(**layout_kwargs)

        # Plot close
        if plot_close:
            fig = self_col.close.vbt.plot(trace_kwargs=close_trace_kwargs, row=row, col=col, fig=fig)

        if len(self_col.values) > 0:
            # Extract information
            _id = self_col.values['id']
            idx = self_col.values['idx']
            size = self_col.values['size']
            price = self_col.values['price']
            fees = self_col.values['fees']
            side = self_col.values['side']

            # Plot Buy markers
            buy_mask = side == OrderSide.Buy
            buy_customdata = np.stack((_id[buy_mask], size[buy_mask], fees[buy_mask]), axis=1)
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
                hovertemplate="Order Id: %{customdata[0]}"
                              "<br>Date: %{x}"
                              "<br>Price: %{y}"
                              "<br>Size: %{customdata[1]:.6f}"
                              "<br>Fees: %{customdata[2]:.6f}"
            )
            buy_scatter.update(**buy_trace_kwargs)
            fig.add_trace(buy_scatter, row=row, col=col)

            # Plot Sell markers
            sell_mask = side == OrderSide.Sell
            sell_customdata = np.stack((_id[sell_mask], size[sell_mask], fees[sell_mask]), axis=1)
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
                hovertemplate="Order Id: %{customdata[0]}"
                              "<br>Date: %{x}"
                              "<br>Price: %{y}"
                              "<br>Size: %{customdata[1]:.6f}"
                              "<br>Fees: %{customdata[2]:.6f}"
            )
            sell_scatter.update(**sell_trace_kwargs)
            fig.add_trace(sell_scatter, row=row, col=col)

        return fig
