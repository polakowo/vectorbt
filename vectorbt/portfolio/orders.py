"""Base class for working with order records.

Class `Orders` wraps order records and the corresponding time series (such as open or close)
to analyze orders. Orders are mainly populated when simulating a portfolio and can be
accessed as `vectorbt.portfolio.base.Portfolio.orders`.

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `Orders.metrics`.

```python-repl
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> import vectorbt as vbt

>>> np.random.seed(42)
>>> price = pd.DataFrame({
...     'a': np.random.uniform(1, 2, size=100),
...     'b': np.random.uniform(1, 2, size=100)
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> size = pd.DataFrame({
...     'a': np.random.uniform(-1, 1, size=100),
...     'b': np.random.uniform(-1, 1, size=100),
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> pf = vbt.Portfolio.from_orders(price, size, fees=0.01, freq='d')

>>> pf.orders.stats(column='a')
Start                2020-01-01 00:00:00
End                  2020-04-09 00:00:00
Period                 100 days 00:00:00
Total Records                        100
Total Buy Orders                      58
Total Sell Orders                     42
Max Size                        0.989877
Min Size                        0.003033
Avg Size                        0.508608
Avg Buy Size                    0.468802
Avg Sell Size                   0.563577
Avg Buy Price                   1.437037
Avg Sell Price                  1.515951
Total Fees                      0.740177
Min Fees                        0.000052
Max Fees                        0.016224
Avg Fees                        0.007402
Avg Buy Fees                    0.006771
Avg Sell Fees                   0.008273
Name: a, dtype: object
```

`Orders.stats` also supports (re-)grouping:

```python-repl
>>> pf.orders.stats(group_by=True)
Start                2020-01-01 00:00:00
End                  2020-04-09 00:00:00
Period                 100 days 00:00:00
Total Records                        200
Total Buy Orders                     109
Total Sell Orders                     91
Max Size                        0.989877
Min Size                        0.003033
Avg Size                        0.506279
Avg Buy Size                    0.472504
Avg Sell Size                   0.546735
Avg Buy Price                    1.47336
Avg Sell Price                  1.496759
Total Fees                      1.483343
Min Fees                        0.000052
Max Fees                        0.018319
Avg Fees                        0.007417
Avg Buy Fees                    0.006881
Avg Sell Fees                   0.008058
Name: group, dtype: object
```
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt import _typing as tp
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.enum import map_enum_values
from vectorbt.utils.figure import make_figure
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.base.reshape_fns import to_1d_array, to_2d_array, broadcast_to
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic.stats_builder import StatsBuilderMixin
from vectorbt.records.base import Records
from vectorbt.records.decorators import add_mapped_fields
from vectorbt.portfolio.enums import order_dt, OrderSide

__pdoc__ = {}

orders_mf_config = Config(
    dict(
        side=dict(defaults=dict(mapping=OrderSide))
    ),
    as_attrs=False,
    readonly=True
)
"""_"""

__pdoc__['orders_mf_config'] = f"""Config of `vectorbt.portfolio.enums.order_dt` 
mapped fields to be overridden in `Orders`.

```json
{orders_mf_config.to_doc()}
```
"""

OrdersT = tp.TypeVar("OrdersT", bound="Orders")


@add_mapped_fields(order_dt, orders_mf_config)
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
        new_close = new_wrapper.wrap(to_2d_array(self.close)[:, col_idxs], group_by=False)
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
        df = self.records.copy()
        df.columns = [
            'Order Id',
            'Date',
            'Column',
            'Size',
            'Price',
            'Fees',
            'Side'
        ]
        df['Date'] = df['Date'].map(lambda x: self.wrapper.index[x])
        df['Column'] = df['Column'].map(lambda x: self.wrapper.columns[x])
        df['Side'] = map_enum_values(df['Side'], OrderSide)
        return df

    # ############# OrderSide ############# #

    @cached_property
    def buy(self: OrdersT) -> OrdersT:
        """Buy operations."""
        filter_mask = self.values['side'] == OrderSide.Buy
        return self.filter_by_mask(filter_mask)

    @cached_method
    def buy_rate(self, group_by: tp.GroupByLike = None, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of buy operations."""
        buy_count = to_1d_array(self.buy.count(group_by=group_by))
        total_count = to_1d_array(self.count(group_by=group_by))
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
        sell_count = to_1d_array(self.sell.count(group_by=group_by))
        total_count = to_1d_array(self.count(group_by=group_by))
        wrap_kwargs = merge_dicts(dict(name_or_index='sell_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(sell_count / total_count, group_by=group_by, **wrap_kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Orders.stats`.

        Merges `vectorbt.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `orders.stats` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        orders_stats_cfg = settings['orders']['stats']

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            orders_stats_cfg
        )

    _metrics: tp.ClassVar[Config] = Config(
        dict(
            start=dict(
                title='Start',
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags='wrapper'
            ),
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags='wrapper'
            ),
            period=dict(
                title='Period',
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags='wrapper'
            ),
            total_records=dict(
                title='Total Records',
                calc_func='count',
                tags='records'
            ),
            total_buy_orders=dict(
                title='Total Buy Orders',
                calc_func='buy.count',
                tags=['orders', 'buy']
            ),
            total_sell_orders=dict(
                title='Total Sell Orders',
                calc_func='sell.count',
                tags=['orders', 'sell']
            ),
            max_size=dict(
                title='Max Size',
                calc_func='size.max',
                tags=['orders', 'size']
            ),
            min_size=dict(
                title='Min Size',
                calc_func='size.min',
                tags=['orders', 'size']
            ),
            avg_size=dict(
                title='Avg Size',
                calc_func='size.mean',
                tags=['orders', 'size']
            ),
            avg_buy_size=dict(
                title='Avg Buy Size',
                calc_func='buy.size.mean',
                tags=['orders', 'buy', 'size']
            ),
            avg_sell_size=dict(
                title='Avg Sell Size',
                calc_func='sell.size.mean',
                tags=['orders', 'sell', 'size']
            ),
            avg_buy_price=dict(
                title='Avg Buy Price',
                calc_func='buy.price.mean',
                tags=['orders', 'buy', 'price']
            ),
            avg_sell_price=dict(
                title='Avg Sell Price',
                calc_func='sell.price.mean',
                tags=['orders', 'sell', 'price']
            ),
            total_fees=dict(
                title='Total Fees',
                calc_func='fees.sum',
                tags=['orders', 'fees']
            ),
            min_fees=dict(
                title='Min Fees',
                calc_func='fees.min',
                tags=['orders', 'fees']
            ),
            max_fees=dict(
                title='Max Fees',
                calc_func='fees.max',
                tags=['orders', 'fees']
            ),
            avg_fees=dict(
                title='Avg Fees',
                calc_func='fees.mean',
                tags=['orders', 'fees']
            ),
            avg_buy_fees=dict(
                title='Avg Buy Fees',
                calc_func='buy.fees.mean',
                tags=['orders', 'buy', 'fees']
            ),
            avg_sell_fees=dict(
                title='Avg Sell Fees',
                calc_func='sell.fees.mean',
                tags=['orders', 'sell', 'fees']
            ),
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

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

        ![](/docs/img/orders_plot.svg)"""
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


Orders.override_metrics_doc(__pdoc__)
