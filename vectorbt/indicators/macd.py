"""Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that 
shows the relationship between two moving averages of a security’s price.

See [Moving Average Convergence Divergence – MACD](https://www.investopedia.com/terms/m/macd.asp).

Use `MACD.from_params` methods to run the indicator."""

import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType
from vectorbt.indicators.ma import ma_caching_nb
from vectorbt.timeseries import ewm_mean_nb, rolling_mean_nb, diff_1d_nb
from vectorbt.indicators.indicator_factory import IndicatorFactory
from vectorbt.utils import *
import plotly.graph_objects as go

__all__ = ['MACD']


@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], i8[:], i8[:], b1[:], b1[:]), cache=True)
def macd_caching_nb(ts, fast_windows, slow_windows, signal_windows, macd_ewms, signal_ewms):
    return ma_caching_nb(ts, np.concatenate((fast_windows, slow_windows)), np.concatenate((macd_ewms, macd_ewms)))


@njit(UniTuple(f8[:, :], 4)(f8[:, :], i8, i8, i8, b1, b1, DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def macd_apply_func_nb(ts, fast_window, slow_window, signal_window, macd_ewm, signal_ewm, cache_dict):
    fast_ma = cache_dict[(fast_window, int(macd_ewm))]
    slow_ma = cache_dict[(slow_window, int(macd_ewm))]
    macd_ts = fast_ma - slow_ma
    if signal_ewm:
        signal_ts = ewm_mean_nb(macd_ts, signal_window)
    else:
        signal_ts = rolling_mean_nb(macd_ts, signal_window)
    signal_ts[:max(fast_window, slow_window)+signal_window-2, :] = np.nan  # min_periodd
    return np.copy(fast_ma), np.copy(slow_ma), macd_ts, signal_ts


MACD = IndicatorFactory(
    ts_names=['ts'],
    param_names=['fast_window', 'slow_window', 'signal_window', 'macd_ewm', 'signal_ewm'],
    output_names=['fast_ma', 'slow_ma', 'macd', 'signal'],
    name='macd',
    custom_properties=dict(
        histogram=lambda self: self.ts.vbt.wrap_array(self.macd.values - self.signal.values),
    )
).from_apply_func(macd_apply_func_nb, caching_func=macd_caching_nb)


class MACD(MACD):
    @classmethod
    def from_params(cls, ts, fast_window=26, slow_window=12, signal_window=9, macd_ewm=True, signal_ewm=True, **kwargs):
        """Calculate fast moving average `MACD.fast_ma`, slow moving average `MACD.slow_ma`, MACD `MACD.macd`, 
        signal `MACD.signal` and histogram `MACD.histogram` from time series `ts` and parameters `fast_window`, 
        `slow_window`, `signal_window`, `macd_ewm` and `signal_ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            fast_window (int or array_like): Size of the fast moving window for MACD. Can be one or more values.
                Defaults to 26.
            slow_window (int or array_like): Size of the slow moving window for MACD. Can be one or more values.
                Defaults to 12.
            signal_window (int or array_like): Size of the moving window for signal. Can be one or more values.
                Defaults to 9.
            macd_ewm (bool or array_like): If True, uses exponential moving average for MACD, otherwise uses 
                simple moving average. Can be one or more values. Defaults to True.
            signal_ewm (bool or array_like): If True, uses exponential moving average for signal, otherwise uses 
                simple moving average. Can be one or more values. Defaults to True.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.indicator_factory.from_params_pipeline.`
        Returns:
            MACD
        Examples:
            ```python-repl
            >>> macd = vbt.MACD.from_params(price['Close'], 
            ...     fast_window=[10, 20], slow_window=[20, 30], signal_window=[30, 40], 
            ...     macd_ewm=[False, True], signal_ewm=[True, False])

            >>> print(macd.fast_ma)
            macd_fast_window           10            20
            macd_slow_window           20            30
            macd_signal_window         30            40
            macd_macd_ewm           False          True
            macd_signal_ewm          True         False
            Date                                       
            2019-02-28                NaN           NaN
            2019-03-01                NaN           NaN
            2019-03-02                NaN           NaN
            ...                       ...           ...
            2019-08-29          10155.972  10330.457140
            2019-08-30          10039.466  10260.715507
            2019-08-31           9988.727  10200.710220

            [185 rows x 2 columns]

            >>> print(macd.slow_ma)
            macd_fast_window            10            20
            macd_slow_window            20            30
            macd_signal_window          30            40
            macd_macd_ewm            False          True
            macd_signal_ewm           True         False
            Date                                        
            2019-02-28                 NaN           NaN
            2019-03-01                 NaN           NaN
            2019-03-02                 NaN           NaN
            ...                        ...           ...
            2019-08-29          10447.3480  10423.585970
            2019-08-30          10359.5555  10370.333077
            2019-08-31          10264.9095  10322.612024

            [185 rows x 2 columns]

            >>> print(macd.macd)
            macd_fast_window          10          20
            macd_slow_window          20          30
            macd_signal_window        30          40
            macd_macd_ewm          False        True
            macd_signal_ewm         True       False
            Date                                    
            2019-02-28               NaN         NaN
            2019-03-01               NaN         NaN
            2019-03-02               NaN         NaN
            ...                      ...         ...
            2019-08-29         -291.3760  -93.128830
            2019-08-30         -320.0895 -109.617570
            2019-08-31         -276.1825 -121.901804

            [185 rows x 2 columns]

            >>> print(macd.signal)
            macd_fast_window            10         20
            macd_slow_window            20         30
            macd_signal_window          30         40
            macd_macd_ewm            False       True
            macd_signal_ewm           True      False
            Date                                     
            2019-02-28                 NaN        NaN
            2019-03-01                 NaN        NaN
            2019-03-02                 NaN        NaN
            ...                        ...        ...
            2019-08-29         -104.032603  28.622033
            2019-08-30         -117.971990  22.424149
            2019-08-31         -128.179278  16.493338

            [185 rows x 2 columns]

            >>> print(macd.histogram)
            macd_fast_window            10          20
            macd_slow_window            20          30
            macd_signal_window          30          40
            macd_macd_ewm            False        True
            macd_signal_ewm           True       False
            Date                                      
            2019-02-28                 NaN         NaN
            2019-03-01                 NaN         NaN
            2019-03-02                 NaN         NaN
            ...                        ...         ...
            2019-08-29         -187.343397 -121.750863
            2019-08-30         -202.117510 -132.041719
            2019-08-31         -148.003222 -138.395142

            [185 rows x 2 columns]
            ```
        """
        return super().from_params(ts, fast_window, slow_window, signal_window, macd_ewm, signal_ewm, **kwargs)

    def plot(self,
             macd_trace_kwargs={},
             signal_trace_kwargs={},
             histogram_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `MACD.macd`, `MACD.signal` and `MACD.histogram`.

        Args:
            macd_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `MACD.macd`.
            signal_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `MACD.signal`.
            histogram_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Bar`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Bar.html) of 
                `MACD.histogram`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.FigureWidget
        Examples:
            ```py
            macd[(10, 20, 30, False, True)].plot()
            ```

            ![](img/MACD.png)"""
        check_type(self.macd, pd.Series)
        check_type(self.signal, pd.Series)
        check_type(self.histogram, pd.Series)

        macd_trace_kwargs = {**dict(
            name=f'MACD ({self.name})'
        ), **macd_trace_kwargs}
        signal_trace_kwargs = {**dict(
            name=f'Signal ({self.name})'
        ), **signal_trace_kwargs}
        histogram_trace_kwargs = {**dict(
            name=f'Histogram ({self.name})',
            showlegend=False
        ), **histogram_trace_kwargs}

        layout_kwargs = {**dict(bargap=0), **layout_kwargs}
        fig = self.macd.vbt.timeseries.plot(trace_kwargs=macd_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.signal.vbt.timeseries.plot(trace_kwargs=signal_trace_kwargs, fig=fig, **layout_kwargs)

        # Plot histogram
        hist = self.histogram.values
        hist_diff = diff_1d_nb(hist)
        marker_colors = np.full(hist.shape, np.nan, dtype=np.object)
        marker_colors[(hist > 0) & (hist_diff > 0)] = 'green'
        marker_colors[(hist > 0) & (hist_diff <= 0)] = 'lightgreen'
        marker_colors[hist == 0] = 'lightgrey'
        marker_colors[(hist < 0) & (hist_diff < 0)] = 'red'
        marker_colors[(hist < 0) & (hist_diff >= 0)] = 'lightcoral'

        histogram_bar = go.Bar(
            x=self.histogram.index,
            y=self.histogram.values,
            marker_color=marker_colors,
            marker_line_width=0
        )
        histogram_bar.update(**histogram_trace_kwargs)
        fig.add_trace(histogram_bar)

        return fig


fix_class_for_pdoc(MACD)
