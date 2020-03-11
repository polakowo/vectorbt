import pandas as pd
import numpy as np
from vectorbt.widgets import FigureWidget
import plotly.graph_objects as go

__all__ = []

# ############# Custom pd.DataFrame accessor ############# #


@pd.api.extensions.register_dataframe_accessor("ohlcv")
class OHLCV_DFAccessor:
    def __init__(self, obj):
        self._validate(obj)
        self._obj = obj

    @staticmethod
    def _validate(obj):
        if (obj.dtypes != np.float64).any():
            raise ValueError("All columns must be float64")
        if 'Open' not in obj.columns:
            raise ValueError("Column 'Open' missing")
        if 'High' not in obj.columns:
            raise ValueError("Column 'High' missing")
        if 'Low' not in obj.columns:
            raise ValueError("Column 'Low' missing")
        if 'Close' not in obj.columns:
            raise ValueError("Column 'Close' missing")

    def plot(self,
             display_volume=True,
             candlestick_kwargs={},
             bar_kwargs={},
             **layout_kwargs):

        open = self._obj['Open']
        high = self._obj['High']
        low = self._obj['Low']
        close = self._obj['Close']
        volume = self._obj['Volume'] if 'Volume' in self._obj.columns else None

        fig = FigureWidget()
        candlestick = go.Candlestick(
            x=self._obj.index,
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
        if display_volume and volume is not None:
            marker_colors = np.empty(volume.shape, dtype=np.object)
            marker_colors[(close - open) > 0] = 'green'
            marker_colors[(close - open) == 0] = 'lightgrey'
            marker_colors[(close - open) < 0] = 'red'
            bar = go.Bar(
                x=self._obj.index,
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
