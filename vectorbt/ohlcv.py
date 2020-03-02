from vectorbt.decorators import *
from vectorbt.widgets import FigureWidget
from vectorbt.timeseries import TimeSeries
import plotly.graph_objects as go

__all__ = ['OHLCV']

class OHLCV():
    @has_type('open', TimeSeries)
    @has_type('high', TimeSeries)
    @has_type('low', TimeSeries)
    @has_type('close', TimeSeries)
    @has_type('volume', TimeSeries)
    @have_same_shape('open', 'high')
    @have_same_shape('open', 'low')
    @have_same_shape('open', 'close')
    @have_same_shape('open', 'volume')
    def __init__(self, open, high, low, close, volume=None):
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume

    @classmethod
    def from_df(cls, df, labels=['Open', 'High', 'Low', 'Close', 'Volume']):
        open = TimeSeries(df[labels[0]].astype(float))
        high = TimeSeries(df[labels[1]].astype(float))
        low = TimeSeries(df[labels[2]].astype(float))
        close = TimeSeries(df[labels[3]].astype(float))
        if labels[4] in df.columns:
            volume = TimeSeries(df[labels[4]].astype(float))
        else:
            volume = None
        return cls(open, high, low, close, volume)

    def plot(self,
             column=None,
             index=None,
             display_volume=True,
             candlestick_kwargs={},
             bar_kwargs={}, 
             **layout_kwargs):

        if column is None:
            if self.open.shape[1] == 1:
                column = 0
            else:
                raise ValueError("For an array with multiple columns, you must pass a column index")
        open = self.open[:, column]
        high = self.high[:, column]
        low = self.low[:, column]
        close = self.close[:, column]

        fig = FigureWidget()
        candlestick = go.Candlestick(
            x=index,
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
        if display_volume and self.volume is not None:
            volume = self.volume[:, column]

            marker_colors = np.empty(volume.shape, dtype=np.object)
            marker_colors[(close - open) > 0] = 'green'
            marker_colors[(close - open) == 0] = 'lightgrey'
            marker_colors[(close - open) < 0] = 'red'
            bar = go.Bar(
                x=index,
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
