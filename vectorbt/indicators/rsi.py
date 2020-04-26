"""The relative strength index (RSI) is a momentum indicator that measures the magnitude of 
recent price changes to evaluate overbought or oversold conditions in the price of a stock 
or other asset. The RSI is displayed as an oscillator (a line graph that moves between two 
extremes) and can have a reading from 0 to 100.

See [Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp).

Use `RSI.from_params` methods to run the indicator."""

import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType
from vectorbt.timeseries import ewm_mean_nb, rolling_mean_nb, diff_nb, set_by_mask_nb, prepend_nb
from vectorbt.indicators.indicator_factory import IndicatorFactory
from vectorbt.utils import *

__all__ = ['RSI']


@njit(DictType(UniTuple(i8, 2), UniTuple(f8[:, :], 2))(f8[:, :], i8[:], b1[:]), cache=True)
def rsi_caching_nb(ts, windows, ewms):
    delta = diff_nb(ts)[1:, :]  # otherwise ewma will be all NaN
    up, down = delta.copy(), delta.copy()
    up = set_by_mask_nb(up, up < 0, 0)
    down = np.abs(set_by_mask_nb(down, down > 0, 0))
    # Cache
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                roll_up = ewm_mean_nb(up, windows[i])
                roll_down = ewm_mean_nb(down, windows[i])
            else:
                roll_up = rolling_mean_nb(up, windows[i])
                roll_down = rolling_mean_nb(down, windows[i])
            roll_up = prepend_nb(roll_up, 1, np.nan)  # bring to old shape
            roll_down = prepend_nb(roll_down, 1, np.nan)
            cache_dict[(windows[i], int(ewms[i]))] = roll_up, roll_down
    return cache_dict


@njit(f8[:, :](f8[:, :], i8, b1, DictType(UniTuple(i8, 2), UniTuple(f8[:, :], 2))), cache=True)
def rsi_apply_func_nb(ts, window, ewm, cache_dict):
    roll_up, roll_down = cache_dict[(window, int(ewm))]
    return 100 - 100 / (1 + roll_up / roll_down)


RSI = IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['rsi'],
    name='rsi'
).from_apply_func(rsi_apply_func_nb, caching_func=rsi_caching_nb)


class RSI(RSI):
    @classmethod
    def from_params(cls, ts, window=14, ewm=False, **kwargs):
        """Calculate relative strength index `RSI.rsi` from time series `ts` and parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window. Can be one or more values. Defaults to 14.
            ewm (bool or array_like): If True, uses exponential moving average, otherwise 
                simple moving average. Can be one or more values. Defaults to False.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.indicator_factory.from_params_pipeline.`
        Returns:
            RSI
        Examples:
            ```python-repl
            >>> rsi = vbt.RSI.from_params(price['Close'], [10, 20], ewm=[False, True])

            >>> print(rsi.rsi)
            rsi_window         10         20
            rsi_ewm         False       True
            Date                            
            2019-02-28        NaN        NaN
            2019-03-01        NaN        NaN
            2019-03-02        NaN        NaN
            ...               ...        ...
            2019-08-29  21.004434  34.001218
            2019-08-30  25.310248  36.190915
            2019-08-31  35.640258  37.043562

            [185 rows x 2 columns]
            ```
        """
        return super().from_params(ts, window, ewm, **kwargs)

    def plot(self,
             levels=(30, 70),
             rsi_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `RSI.rsi`.

        Args:
            trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `RSI.rsi`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.FigureWidget
        Examples:
            ```py
            rsi[(10, False)].plot()
            ```

            ![](img/RSI.png)"""
        check_type(self.rsi, pd.Series)

        rsi_trace_kwargs = {**dict(
            name=f'RSI ({self.name})'
        ), **rsi_trace_kwargs}

        layout_kwargs = {**dict(yaxis=dict(range=[-5, 105])), **layout_kwargs}
        fig = self.rsi.vbt.timeseries.plot(trace_kwargs=rsi_trace_kwargs, fig=fig, **layout_kwargs)

        # Fill void between levels
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=self.rsi.index[0],
            y0=levels[0],
            x1=self.rsi.index[-1],
            y1=levels[1],
            fillcolor="purple",
            opacity=0.1,
            layer="below",
            line_width=0,
        )

        return fig


fix_class_for_pdoc(RSI)
