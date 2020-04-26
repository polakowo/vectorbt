"""The average true range (ATR) is a technical analysis indicator that measures market volatility 
by decomposing the entire range of an asset price for that period.

See [Average True Range - ATR](https://www.investopedia.com/terms/a/atr.asp).

Use `ATR.from_params` method to run the indicator."""

import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType, Tuple
from vectorbt.timeseries import ewm_mean_nb, rolling_mean_nb, fshift_nb
from vectorbt.indicators.indicator_factory import IndicatorFactory
from vectorbt.utils import *

__all__ = ['ATR']


@njit(f8[:, :](f8[:, :, :]), cache=True)
def nanmax_cube_axis0_nb(a):
    b = np.empty((a.shape[1], a.shape[2]), dtype=a.dtype)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            b[i, j] = np.nanmax(a[:, i, j])
    return b


@njit(Tuple((f8[:, :], DictType(UniTuple(i8, 2), f8[:, :])))(f8[:, :], f8[:, :], f8[:, :], i8[:], b1[:]), cache=True)
def atr_caching_nb(close_ts, high_ts, low_ts, windows, ewms):
    # Calculate TR here instead of re-calculating it for each param in atr_apply_func_nb
    tr0 = high_ts - low_ts
    tr1 = np.abs(high_ts - fshift_nb(close_ts, 1))
    tr2 = np.abs(low_ts - fshift_nb(close_ts, 1))
    tr = nanmax_cube_axis0_nb(np.stack((tr0, tr1, tr2)))

    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                atr = ewm_mean_nb(tr, windows[i])
            else:
                atr = rolling_mean_nb(tr, windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = atr
    return tr, cache_dict


@njit(UniTuple(f8[:, :], 2)(f8[:, :], f8[:, :], f8[:, :], i8, b1, f8[:, :], DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def atr_apply_func_nb(close_ts, high_ts, low_ts, window, ewm, tr, cache_dict):
    return tr, cache_dict[(window, int(ewm))]


ATR = IndicatorFactory(
    ts_names=['close_ts', 'high_ts', 'low_ts'],
    param_names=['window', 'ewm'],
    output_names=['tr', 'atr'],
    name='atr'
).from_apply_func(atr_apply_func_nb, caching_func=atr_caching_nb)


class ATR(ATR):
    @classmethod
    def from_params(cls, close_ts, high_ts, low_ts, window, ewm=True, **kwargs):
        """Calculate true range `ATR.tr` and average true range `ATR.atr` from time series `close_ts`, 
        `high_ts`, and `low_ts`, and parameters `window` and `ewm`.

        Args:
            close_ts (pandas_like): The last closing price.
            high_ts (pandas_like, optional): The highest price. If None, uses `close_ts`.
            low_ts (pandas_like, optional): The lowest price. If None, uses `close_ts`.
            window (int or array_like): Size of the moving window. Can be one or more values. 
                Defaults to 14.
            ewm (bool or array_like): If True, uses exponential moving average, otherwise 
                simple moving average. Can be one or more values. Defaults to True.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.indicator_factory.from_params_pipeline.`
        Returns:
            ATR
        Examples:
            ```python-repl
            >>> atr = vbt.ATR.from_params(price['Close'], 
            ...     price['High'], price['Low'], [20, 30], [False, True])

            >>> print(atr.tr)
            atr_window      20      30
            atr_ewm      False    True
            Date                      
            2019-02-28   60.24   60.24
            2019-03-01   56.11   56.11
            2019-03-02   42.48   42.48
            ...            ...     ...
            2019-08-29  335.16  335.16
            2019-08-30  227.82  227.82
            2019-08-31  141.42  141.42

            [185 rows x 2 columns]

            >>> print(atr.atr)
            atr_window        20          30
            atr_ewm        False        True
            Date                            
            2019-02-28       NaN         NaN
            2019-03-01       NaN         NaN
            2019-03-02       NaN         NaN
            ...              ...         ...
            2019-08-29  476.9385  491.469062
            2019-08-30  458.7415  474.459365
            2019-08-31  452.0480  452.972860

            [185 rows x 2 columns]
            ```
        """
        return super().from_params(close_ts, high_ts, low_ts, window, ewm, **kwargs)

    def plot(self,
             tr_trace_kwargs={},
             atr_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `ATR.tr` and `ATR.atr`.

        Args:
            tr_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) for `ATR.tr`.
            atr_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) for `ATR.atr`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.FigureWidget
        Examples:
            ```py
            atr[(10, False)].plot()
            ```

            ![](img/ATR.png)"""
        check_type(self.tr, pd.Series)
        check_type(self.atr, pd.Series)

        tr_trace_kwargs = {**dict(
            name=f'TR ({self.name})'
        ), **tr_trace_kwargs}
        atr_trace_kwargs = {**dict(
            name=f'ATR ({self.name})'
        ), **atr_trace_kwargs}

        fig = self.tr.vbt.timeseries.plot(trace_kwargs=tr_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.atr.vbt.timeseries.plot(trace_kwargs=atr_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(ATR)
