"""A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard 
deviations (positively and negatively) away from a simple moving average (SMA) of the security's 
price, but can be adjusted to user preferences.

See [Bollinger Band®](https://www.investopedia.com/terms/b/bollingerbands.asp).

Use `BollingerBands.from_params` method to run the indicator."""

import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType
from vectorbt.indicators.ma import ma_caching_nb
from vectorbt.indicators.mstd import mstd_caching_nb
from vectorbt.indicators.indicator_factory import IndicatorFactory
from vectorbt.widgets import layout_defaults
from vectorbt.utils import *

__all__ = ['BollingerBands']


@njit(UniTuple(DictType(UniTuple(i8, 2), f8[:, :]), 2)(f8[:, :], i8[:], b1[:], f8[:]), cache=True)
def bb_caching_nb(ts, windows, ewms, alphas):
    ma_cache_dict = ma_caching_nb(ts, windows, ewms)
    mstd_cache_dict = mstd_caching_nb(ts, windows, ewms)
    return ma_cache_dict, mstd_cache_dict


@njit(UniTuple(f8[:, :], 3)(f8[:, :], i8, b1, f8, DictType(UniTuple(i8, 2), f8[:, :]), DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def bb_apply_func_nb(ts, window, ewm, alpha, ma_cache_dict, mstd_cache_dict):
    # Calculate lower, middle and upper bands
    ma = np.copy(ma_cache_dict[(window, int(ewm))])
    mstd = np.copy(mstd_cache_dict[(window, int(ewm))])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


BollingerBands = IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm', 'alpha'],
    output_names=['ma', 'upper_band', 'lower_band'],
    name='bb',
    custom_properties=dict(
        percent_b=lambda self: self.ts.vbt.wrap_array(
            (self.ts.values - self.lower_band.values) / (self.upper_band.values - self.lower_band.values)),
        bandwidth=lambda self: self.ts.vbt.wrap_array(
            (self.upper_band.values - self.lower_band.values) / self.ma.values)
    )
).from_apply_func(bb_apply_func_nb, caching_func=bb_caching_nb)


class BollingerBands(BollingerBands):
    @classmethod
    def from_params(cls, ts, window=20, ewm=False, alpha=2, **kwargs):
        """Calculate moving average `BollingerBands.ma`, upper Bollinger Band `BollingerBands.upper_band`,
        lower Bollinger Band `BollingerBands.lower_band`, %b `BollingerBands.percent_b` and 
        bandwidth `BollingerBands.bandwidth` from time series `ts` and parameters `window`, `ewm` and `alpha`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window. Can be one or more values.
                Defaults to 20.
            ewm (bool or array_like): If True, uses exponential moving average and standard deviation, 
                otherwise uses simple moving average and standard deviation. Can be one or more values. 
                Defaults to False.
            alpha (int, float or array_like): Number of standard deviations. Can be one or more values. Defaults to 2.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.indicator_factory.from_params_pipeline.`
        Returns:
            BollingerBands
        Examples:
            ```python-repl
            >>> bb = vbt.BollingerBands.from_params(price['Close'], 
            ...     window=[10, 20], alpha=[2, 3], ewm=[False, True])

            >>> print(bb.ma)
            bb_window          10            20
            bb_ewm          False          True
            bb_alpha          2.0           3.0
            Date                               
            2019-02-28        NaN           NaN
            2019-03-01        NaN           NaN
            2019-03-02        NaN           NaN
            ...               ...           ...
            2019-08-29  10155.972  10330.457140
            2019-08-30  10039.466  10260.715507
            2019-08-31   9988.727  10200.710220

            [185 rows x 2 columns]

            >>> print(bb.upper_band)
            bb_window             10            20
            bb_ewm             False          True
            bb_alpha             2.0           3.0
            Date                                  
            2019-02-28           NaN           NaN
            2019-03-01           NaN           NaN
            2019-03-02           NaN           NaN
            ...                  ...           ...
            2019-08-29  10841.965056  12140.030938
            2019-08-30  10659.668073  12104.745144
            2019-08-31  10654.433961  12044.795485

            [185 rows x 2 columns]

            >>> print(bb.lower_band)
            bb_window            10           20
            bb_ewm            False         True
            bb_alpha            2.0          3.0
            Date                                
            2019-02-28          NaN          NaN
            2019-03-01          NaN          NaN
            2019-03-02          NaN          NaN
            ...                 ...          ...
            2019-08-29  9469.978944  8520.883342
            2019-08-30  9419.263927  8416.685869
            2019-08-31  9323.020039  8356.624955

            [185 rows x 2 columns]

            >>> print(bb.percent_b)
            bb_window         10        20
            bb_ewm         False      True
            bb_alpha         2.0       3.0
            Date                          
            2019-02-28       NaN       NaN
            2019-03-01       NaN       NaN
            2019-03-02       NaN       NaN
            ...              ...       ...
            2019-08-29  0.029316  0.273356
            2019-08-30  0.144232  0.320354
            2019-08-31  0.231063  0.345438

            [185 rows x 2 columns]

            >>> print(bb.bandwidth)
            bb_window         10        20
            bb_ewm         False      True
            bb_alpha         2.0       3.0
            Date                          
            2019-02-28       NaN       NaN
            2019-03-01       NaN       NaN
            2019-03-02       NaN       NaN
            2019-03-03       NaN       NaN
            2019-03-04       NaN       NaN
            ...              ...       ...
            2019-08-27  0.107370  0.313212
            2019-08-28  0.130902  0.325698
            2019-08-29  0.135092  0.350338
            2019-08-30  0.123553  0.359435
            2019-08-31  0.133292  0.361560

            [185 rows x 2 columns]
            ```
        """
        alpha = np.asarray(alpha).astype(np.float64)
        return super().from_params(ts, window, ewm, alpha, **kwargs)

    def plot(self,
             ts_trace_kwargs={},
             ma_trace_kwargs={},
             upper_band_trace_kwargs={},
             lower_band_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `BollingerBands.ma`, `BollingerBands.upper_band` and `BollingerBands.lower_band` against 
        `BollingerBands.ts`.

        Args:
            ts_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.ts`.
            ma_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.ma`.
            upper_band_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.upper_band`.
            lower_band_trace_kwargs (dict, optional): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.lower_band`.
            fig (plotly.graph_objects.Figure, optional): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.FigureWidget
        Examples:
            ```py
            bb[(10, False, 2)].plot()
            ```

            ![](img/BollingerBands.png)"""
        check_type(self.ts, pd.Series)
        check_type(self.ma, pd.Series)
        check_type(self.upper_band, pd.Series)
        check_type(self.lower_band, pd.Series)

        lower_band_trace_kwargs = {**dict(
            name=f'Lower Band ({self.name})',
            line=dict(color='grey', width=0),
            showlegend=False
        ), **lower_band_trace_kwargs}
        upper_band_trace_kwargs = {**dict(
            name=f'Upper Band ({self.name})',
            line=dict(color='grey', width=0),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.25)',
            showlegend=False
        ), **upper_band_trace_kwargs}  # default kwargs
        ma_trace_kwargs = {**dict(
            name=f'MA ({self.name})',
            line=dict(color=layout_defaults['colorway'][1])
        ), **ma_trace_kwargs}
        ts_trace_kwargs = {**dict(
            name=f'Price ({self.name})',
            line=dict(color=layout_defaults['colorway'][0])
        ), **ts_trace_kwargs}

        fig = self.lower_band.vbt.timeseries.plot(trace_kwargs=lower_band_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.upper_band.vbt.timeseries.plot(trace_kwargs=upper_band_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.timeseries.plot(trace_kwargs=ma_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ts.vbt.timeseries.plot(trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(BollingerBands)
