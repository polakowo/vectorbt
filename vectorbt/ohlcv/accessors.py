"""Custom pandas accessors.

!!! note
    Accessors do not utilize caching."""

import numpy as np
import plotly.graph_objects as go

from vectorbt.root_accessors import register_dataframe_accessor
from vectorbt.utils import checks
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.generic.accessors import Generic_DFAccessor


@register_dataframe_accessor('ohlcv')
class OHLCV_DFAccessor(Generic_DFAccessor):  # pragma: no cover
    """Accessor on top of OHLCV data. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.ohlcv`."""

    def __init__(self, obj, column_names=None, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._column_names = column_names

        Generic_DFAccessor.__init__(self, obj, **kwargs)

    def plot(self,
             plot_type=go.Ohlc,
             display_volume=True,
             ohlc_kwargs=None,
             bar_kwargs=None,
             fig=None,
             **layout_kwargs):
        """Plot OHLCV data.

        Args:
            plot_type: Either `plotly.graph_objects.Ohlc` or `plotly.graph_objects.Candlestick`.
            display_volume (bool): If True, displays volume as bar chart.
            ohlc_kwargs (dict): Keyword arguments passed to `plot_type`.
            bar_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import yfinance as yf

        >>> yf.Ticker("BTC-USD").history(period="max").vbt.ohlcv.plot()
        ```

        ![](/vectorbt/docs/img/ohlcv.png)
        """
        from vectorbt import settings

        column_names = settings.ohlcv['column_names'] if self._column_names is None else self._column_names
        open = self._obj[column_names['open']]
        high = self._obj[column_names['high']]
        low = self._obj[column_names['low']]
        close = self._obj[column_names['close']]

        # Set up figure
        if fig is None:
            fig = CustomFigureWidget()
            fig.update_layout(
                showlegend=True,
                xaxis_rangeslider_visible=False,
                xaxis_showgrid=True,
                yaxis_showgrid=True,
                bargap=0
            )
        fig.update_layout(**layout_kwargs)
        if ohlc_kwargs is None:
            ohlc_kwargs = {}
        if bar_kwargs is None:
            bar_kwargs = {}
        ohlc = plot_type(
            x=self.wrapper.index,
            open=open,
            high=high,
            low=low,
            close=close,
            name='OHLC',
            yaxis="y2",
            xaxis="x",
            increasing_line_color='#1b9e76',
            decreasing_line_color='#d95f02'
        )
        ohlc.update(**ohlc_kwargs)
        fig.add_trace(ohlc)
        if display_volume:
            volume = self._obj[column_names['volume']]

            marker_colors = np.empty(volume.shape, dtype=np.object)
            marker_colors[(close.values - open.values) > 0] = '#1b9e76'
            marker_colors[(close.values - open.values) == 0] = 'lightgrey'
            marker_colors[(close.values - open.values) < 0] = '#d95f02'
            bar = go.Bar(
                x=self.wrapper.index,
                y=volume,
                marker_color=marker_colors,
                marker_line_width=0,
                marker_opacity=0.7,
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
        return fig