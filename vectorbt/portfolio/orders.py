"""Base class for working with order records.

Class `Orders` wraps order records and the corresponding time series (such as open or close)
to analyze orders. Orders are mainly populated when simulating a portfolio and can be
accessed as `vectorbt.portfolio.base.Portfolio.orders`.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt import _typing as tp
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.enum import enum_to_value_map
from vectorbt.utils.figure import make_figure
from vectorbt.utils.config import merge_dicts
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast_to
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.records.base import Records
from vectorbt.records.mapped_array import MappedArray
from vectorbt.portfolio.enums import order_dt, OrderSide


OrdersT = tp.TypeVar("OrdersT", bound="Orders")


class Orders(Records):
    """Extends `Records` for working with order records.

    ## Example

    Get the total number of buy and sell operations:
    ```python-repl
    >>> import vectorbt as vbt
    >>> import pandas as pd

    >>> price = pd.Series([1., 2., 3., 2., 1.])
    >>> size = pd.Series([1., 1., 1., 1., -1.])
    >>> orders = vbt.Portfolio.from_orders(price, size).orders

    >>> orders.buy.count()
    4
    >>> orders.sell.count()
    1
    ```
    """

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 close: tp.ArrayLike,
                 idx_field: str = 'idx',
                 **kwargs) -> None:
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

    def indexing_func_meta(self: OrdersT, pd_indexing_func: tp.PandasIndexingFunc,
                           **kwargs) -> tp.Tuple[OrdersT, tp.MaybeArray, tp.Array1d]:
        """Perform indexing on `Orders` and return metadata."""
        new_wrapper, new_records_arr, group_idxs, col_idxs = \
            Records.indexing_func_meta(self, pd_indexing_func, **kwargs)
        new_close = new_wrapper.wrap(to_2d(self.close, raw=True)[:, col_idxs], group_by=False)
        return self.copy(
            wrapper=new_wrapper,
            records_arr=new_records_arr,
            close=new_close
        ), group_idxs, col_idxs

    def indexing_func(self: OrdersT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> OrdersT:
        """Perform indexing on `Orders`."""
        return self.indexing_func_meta(pd_indexing_func, **kwargs)[0]

    @property
    def close(self) -> tp.SeriesFrame:
        """Reference price such as close."""
        return self._close

    @property  # no need for cached
    def records_readable(self) -> tp.Frame:
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame()
        out['Order Id'] = records_df['id']
        out['Date'] = records_df['idx'].map(lambda x: self.wrapper.index[x])
        out['Column'] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        out['Size'] = records_df['size']
        out['Price'] = records_df['price']
        out['Fees'] = records_df['fees']
        out['Side'] = records_df['side'].map(enum_to_value_map(OrderSide))
        return out

    @cached_property
    def size(self) -> MappedArray:
        """Size of each order."""
        return self.map_field('size')

    @cached_property
    def price(self) -> MappedArray:
        """Price of each order."""
        return self.map_field('price')

    @cached_property
    def fees(self) -> MappedArray:
        """Fees paid for each order."""
        return self.map_field('fees')

    # ############# OrderSide ############# #

    @cached_property
    def side(self) -> MappedArray:
        """Side of each order.

        See `vectorbt.portfolio.enums.OrderSide`."""
        return self.map_field('side')

    @cached_property
    def buy(self: OrdersT) -> OrdersT:
        """Buy operations."""
        filter_mask = self.values['side'] == OrderSide.Buy
        return self.filter_by_mask(filter_mask)

    @cached_method
    def buy_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of buy operations."""
        buy_count = to_1d(self.buy.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='buy_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(buy_count / total_count, group_by=group_by, **wrap_kwargs)

    @cached_property
    def sell(self: OrdersT) -> OrdersT:
        """Sell operations."""
        filter_mask = self.values['side'] == OrderSide.Sell
        return self.filter_by_mask(filter_mask)

    @cached_method
    def sell_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of sell operations."""
        sell_count = to_1d(self.sell.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='sell_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(sell_count / total_count, group_by=group_by, **wrap_kwargs)

    # ############# Plotting ############# #

    def plot(self,
             column: tp.Optional[tp.Label] = None,
             plot_close: bool = True,
             close_trace_kwargs: tp.KwargsLike = None,
             buy_trace_kwargs: tp.KwargsLike = None,
             sell_trace_kwargs: tp.KwargsLike = None,
             add_trace_kwargs: tp.KwargsLike = None,
             fig: tp.Optional[tp.BaseFigure] = None,
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot orders.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `Orders.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Orders.close`.
            buy_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Buy" markers.
            sell_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Sell" markers.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> orders.plot()
        ```

        ![](/vectorbt/docs/img/orders_plot.svg)"""
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
        if buy_trace_kwargs is None:
            buy_trace_kwargs = {}
        if sell_trace_kwargs is None:
            sell_trace_kwargs = {}
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
                    color=plotting_cfg['contrast_color_schema']['green'],
                    size=8,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['green'])
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
            fig.add_trace(buy_scatter, **add_trace_kwargs)

            # Plot Sell markers
            sell_mask = side == OrderSide.Sell
            sell_customdata = np.stack((_id[sell_mask], size[sell_mask], fees[sell_mask]), axis=1)
            sell_scatter = go.Scatter(
                x=self_col.wrapper.index[idx[sell_mask]],
                y=price[sell_mask],
                mode='markers',
                marker=dict(
                    symbol='triangle-down',
                    color=plotting_cfg['contrast_color_schema']['red'],
                    size=8,
                    line=dict(
                        width=1,
                        color=adjust_lightness(plotting_cfg['contrast_color_schema']['red'])
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
            fig.add_trace(sell_scatter, **add_trace_kwargs)

        return fig
