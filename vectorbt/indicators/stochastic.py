"""A stochastic oscillator is a momentum indicator comparing a particular closing price of a security 
to a range of its prices over a certain period of time. It is used to generate overbought and oversold 
trading signals, utilizing a 0-100 bounded range of values.

See [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp).

Use `Stochastic.from_params` methods to run the indicator."""

import numpy as np
import pandas as pd
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType

from vectorbt import timeseries, indicators
from vectorbt.utils import checks, common


@njit(DictType(i8, UniTuple(f8[:, :], 2))(f8[:, :], f8[:, :], f8[:, :], i8[:], i8[:], b1[:]), cache=True)
def stoch_caching_nb(close_ts, high_ts, low_ts, k_windows, d_windows, d_ewms):
    """Numba-compiled caching function for `Stochastic`."""
    cache_dict = dict()
    for i in range(k_windows.shape[0]):
        if k_windows[i] not in cache_dict:
            roll_min = timeseries.nb.rolling_min_nb(low_ts, k_windows[i])
            roll_max = timeseries.nb.rolling_max_nb(high_ts, k_windows[i])
            cache_dict[k_windows[i]] = roll_min, roll_max
    return cache_dict


@njit(UniTuple(f8[:, :], 2)(f8[:, :], f8[:, :], f8[:, :], i8, i8, b1, DictType(i8, UniTuple(f8[:, :], 2))), cache=True)
def stoch_apply_func_nb(close_ts, high_ts, low_ts, k_window, d_window, d_ewm, cache_dict):
    """Numba-compiled apply function for `Stochastic`."""
    roll_min, roll_max = cache_dict[k_window]
    percent_k = 100 * (close_ts - roll_min) / (roll_max - roll_min)
    if d_ewm:
        percent_d = timeseries.nb.ewm_mean_nb(percent_k, d_window)
    else:
        percent_d = timeseries.nb.rolling_mean_nb(percent_k, d_window)
    percent_d[:k_window+d_window-2, :] = np.nan  # min_periods
    return percent_k, percent_d


Stochastic = indicators.factory.IndicatorFactory(
    ts_names=['close_ts', 'high_ts', 'low_ts'],
    param_names=['k_window', 'd_window', 'd_ewm'],
    output_names=['percent_k', 'percent_d'],
    name='stoch'
).from_apply_func(stoch_apply_func_nb, caching_func=stoch_caching_nb)


class Stochastic(Stochastic):
    @classmethod
    def from_params(cls, close_ts, high_ts=None, low_ts=None, k_window=14, d_window=3, d_ewm=False, **kwargs):
        """Calculate %K `Stochastic.percent_k` and %D `Stochastic.percent_d` from time series `close_ts`, 
        `high_ts`, and `low_ts`, and parameters `k_window`, `d_window` and `d_ewm`.

        Args:
            close_ts (pandas_like): The last closing price.
            high_ts (pandas_like, optional): The highest price. If None, uses `close_ts`.
            low_ts (pandas_like, optional): The lowest price. If None, uses `close_ts`.
            k_window (int or array_like): Size of the moving window for %K. Can be one or more values. 
                Defaults to 14.
            d_window (int or array_like): Size of the moving window for %D. Can be one or more values. 
                Defaults to 3.
            d_ewm (bool or array_like): If True, uses exponential moving average for %D, otherwise 
                simple moving average. Can be one or more values. Defaults to False.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            Stochastic
        Examples:
            ```python-repl
            >>> stoch = vbt.Stochastic.from_params(price['Close'],
            ...     high_ts=price['High'], low_ts=price['Low'],
            ...     k_window=[10, 20], d_window=[2, 3], d_ewm=[False, True])

            >>> print(stoch.percent_k)
            stoch_k_window         10         20
            stoch_d_window          2          3
            stoch_d_ewm         False       True
            Date                                
            2019-02-28            NaN        NaN
            2019-03-01            NaN        NaN
            2019-03-02            NaN        NaN
            ...                   ...        ...
            2019-08-29       5.806308   3.551280
            2019-08-30      12.819694   8.380488
            2019-08-31      19.164757   9.922813

            [185 rows x 2 columns]

            >>> print(stoch.percent_d)
            stoch_k_window         10         20
            stoch_d_window          2          3
            stoch_d_ewm         False       True
            Date                                
            2019-02-28            NaN        NaN
            2019-03-01            NaN        NaN
            2019-03-02            NaN        NaN
            ...                   ...        ...
            2019-08-29       4.437639   8.498544
            2019-08-30       9.313001   8.439516
            2019-08-31      15.992225   9.181164

            [185 rows x 2 columns]
            ```
        """
        if high_ts is None:
            high_ts = close_ts
        if low_ts is None:
            low_ts = close_ts
        return super().from_params(close_ts, high_ts, low_ts, k_window, d_window, d_ewm, **kwargs)

    def plot(self,
             levels=(30, 70),
             percent_k_trace_kwargs={},
             percent_d_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `Stochastic.percent_k` and `Stochastic.percent_d`.

        Args:
            percent_k_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `Stochastic.percent_k`.
            percent_d_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `Stochastic.percent_d`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.FigureWidget
        Examples:
            ```py
            stoch[(10, 2, False)].plot(levels=(20, 80))
            ```

            ![](img/Stochastic.png)"""
        checks.assert_type(self.percent_k, pd.Series)
        checks.assert_type(self.percent_d, pd.Series)

        percent_k_trace_kwargs = {**dict(
            name=f'%K ({self.name})'
        ), **percent_k_trace_kwargs}
        percent_d_trace_kwargs = {**dict(
            name=f'%D ({self.name})'
        ), **percent_d_trace_kwargs}

        layout_kwargs = {**dict(yaxis=dict(range=[-5, 105])), **layout_kwargs}
        fig = self.percent_k.vbt.timeseries.plot(trace_kwargs=percent_k_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.percent_d.vbt.timeseries.plot(trace_kwargs=percent_d_trace_kwargs, fig=fig, **layout_kwargs)

        # Plot levels
        # Fill void between levels
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=self.percent_k.index[0],
            y0=levels[0],
            x1=self.percent_k.index[-1],
            y1=levels[1],
            fillcolor="purple",
            opacity=0.1,
            layer="below",
            line_width=0,
        )

        return fig


common.fix_class_for_pdoc(Stochastic)
