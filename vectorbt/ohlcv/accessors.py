"""Custom pandas accessors.

!!! note
    Accessors do not utilize caching."""

import numpy as np
import plotly.graph_objects as go

from vectorbt import defaults
from vectorbt.root_accessors import register_dataframe_accessor
from vectorbt.utils import checks
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.generic.accessors import Generic_DFAccessor


@register_dataframe_accessor('ohlcv')
class OHLCV_DFAccessor(Generic_DFAccessor):  # pragma: no cover
    """Accessor on top of OHLCV data. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.ohlcv`."""

    def __init__(self, obj, column_names=None, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._column_names = column_names

        Generic_DFAccessor.__init__(self, obj, freq=freq)

    def plot(self,
             display_volume=True,
             candlestick_kwargs={},
             bar_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot OHLCV data.

        Args:
            display_volume (bool): If `True`, displays volume as bar chart.
            candlestick_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Candlestick`.
            bar_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            import vectorbt as vbt
            import yfinance as yf

            yf.Ticker("BTC-USD").history(period="max").vbt.ohlcv.plot()
            ```

            ![](/vectorbt/docs/img/ohlcv.png)"""
        column_names = defaults.ohlcv['column_names'] if self._column_names is None else self._column_names
        open = self._obj[column_names['open']]
        high = self._obj[column_names['high']]
        low = self._obj[column_names['low']]
        close = self._obj[column_names['close']]

        # Set up figure
        if fig is None:
            fig = CustomFigureWidget()
        candlestick = go.Candlestick(
            x=self.index,
            open=open,
            high=high,
            low=low,
            close=close,
            name='OHLC',
            yaxis="y2",
            xaxis="x"
        )
        candlestick.update(**candlestick_kwargs)
        fig.add_trace(candlestick)
        if display_volume:
            volume = self._obj[column_names['volume']]

            marker_colors = np.empty(volume.shape, dtype=np.object)
            marker_colors[(close.values - open.values) > 0] = 'green'
            marker_colors[(close.values - open.values) == 0] = 'lightgrey'
            marker_colors[(close.values - open.values) < 0] = 'red'
            bar = go.Bar(
                x=self.index,
                y=volume,
                marker_color=marker_colors,
                marker_line_width=0,
                name='Volume',
                yaxis="y",
                xaxis="x"
            )
            bar.update(**bar_kwargs)
            fig.add_trace(bar)
            fig.update_layout(
                yaxis2=dict(
                    domain=[0.33, 1]
                ),
                yaxis=dict(
                    domain=[0, 0.33]
                )
            )
        fig.update_layout(
            showlegend=True,
            xaxis_rangeslider_visible=False,
            xaxis_showgrid=True,
            yaxis_showgrid=True
        )
        fig.update_layout(**layout_kwargs)

        return fig