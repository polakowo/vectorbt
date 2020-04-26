"""Standard deviation is an indicator that measures the size of an assets recent price moves 
in order to predict how volatile the price may be in the future.

Use `MSTD.from_params` method to run the indicator."""

import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType
from vectorbt.timeseries import ewm_std_nb, rolling_std_nb
from vectorbt.indicators.indicator_factory import IndicatorFactory
from vectorbt.utils import *

__all__ = ['MSTD']


@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], b1[:]), cache=True)
def mstd_caching_nb(ts, windows, ewms):
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                mstd = ewm_std_nb(ts, windows[i])
            else:
                mstd = rolling_std_nb(ts, windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = mstd
    return cache_dict


@njit(f8[:, :](f8[:, :], i8, b1, DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def mstd_apply_func_nb(ts, window, ewm, cache_dict):
    return cache_dict[(window, int(ewm))]


MSTD = IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['mstd'],
    name='mstd'
).from_apply_func(mstd_apply_func_nb, caching_func=mstd_caching_nb)


class MSTD(MSTD):
    @classmethod
    def from_params(cls, ts, window, ewm=False, **kwargs):
        """Calculate moving standard deviation `MSTD.mstd` from time series `ts` and 
        parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window. Can be one or more values.
            ewm (bool or array_like): If True, uses exponential moving standard deviation, 
                otherwise uses simple moving standard deviation. Can be one or more values. 
                Defaults to False.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.indicator_factory.from_params_pipeline.`
        Returns:
            MSTD
        Examples:
            ```python-repl
            >>> mstd = vbt.MSTD.from_params(price['Close'], [10, 20], ewm=[False, True])

            >>> print(mstd.mstd)
            mstd_window          10          20
            mstd_ewm          False        True
            Date                               
            2019-02-28          NaN         NaN
            2019-03-01          NaN         NaN
            2019-03-02          NaN         NaN
            ...                 ...         ...
            2019-08-29   342.996528  603.191266
            2019-08-30   310.101037  614.676546
            2019-08-31   332.853480  614.695088

            [185 rows x 2 columns]
            ```
        """
        return super().from_params(ts, window, ewm, **kwargs)

    def plot(self,
             mstd_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `MSTD.mstd`.

        Args:
            mstd_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `MSTD.mstd`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.FigureWidget
        Examples:
            ```py
            mstd[(10, False)].plot()
            ```

            ![](img/MSTD.png)"""
        check_type(self.mstd, pd.Series)

        mstd_trace_kwargs = {**dict(
            name=f'MSTD ({self.name})'
        ), **mstd_trace_kwargs}

        fig = self.mstd.vbt.timeseries.plot(trace_kwargs=mstd_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(MSTD)
