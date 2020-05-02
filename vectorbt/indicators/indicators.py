"""Custom indicators built with `vectorbt.indicators.factory.IndicatorFactory`.

Before running the examples, import the following libraries:
```py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
from numba import njit
import yfinance as yf

import vectorbt as vbt

ticker = yf.Ticker("BTC-USD")
price = ticker.history(start=datetime(2019, 3, 1), end=datetime(2019, 9, 1))

price['Close'].vbt.timeseries.plot()
```
![](img/Indicators_price.png)"""

import numpy as np
import pandas as pd
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType, Tuple
import itertools
import plotly.graph_objects as go

from vectorbt import timeseries, indicators, defaults
from vectorbt.utils import checks, reshape_fns
from vectorbt.utils.common import fix_class_for_pdoc

# ############# MA ############# #


@njit(cache=True)
def ma_caching_nb(ts, windows, ewms):
    """Numba-compiled caching function for `MA`."""
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                ma = timeseries.nb.ewm_mean_nb(ts, windows[i], minp=windows[i])
            else:
                ma = timeseries.nb.rolling_mean_nb(ts, windows[i], minp=windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = ma
    return cache_dict


@njit(cache=True)
def ma_apply_func_nb(ts, window, ewm, cache_dict):
    """Numba-compiled apply function for `MA`."""
    return cache_dict[(window, int(ewm))]


MA = indicators.factory.IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['ma'],
    name='ma'
).from_apply_func(ma_apply_func_nb, caching_func=ma_caching_nb)


class MA(MA):
    """A moving average (MA) is a widely used indicator in technical analysis that helps smooth out 
    price action by filtering out the “noise” from random short-term price fluctuations. 

    See [Moving Average (MA)](https://www.investopedia.com/terms/m/movingaverage.asp).

    Use `MA.from_params` or `MA.from_combinations` methods to run the indicator."""
    @classmethod
    def from_params(cls, ts, window, ewm=False, **kwargs):
        """Calculate moving average `MA.ma` from time series `ts` and parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window.
            ewm (bool or array_like): If `True`, uses exponential moving average, otherwise 
                simple moving average.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            MA
        Example:
            ```python-repl
            >>> ma = vbt.MA.from_params(price['Close'], [10, 20], ewm=[False, True])

            >>> print(ma.ma)
            ma_window          10            20
            ma_ewm          False          True
            Date                               
            2019-02-28        NaN           NaN
            2019-03-01        NaN           NaN
            2019-03-02        NaN           NaN
            ...               ...           ...
            2019-08-29  10155.972  10330.457140
            2019-08-30  10039.466  10260.715507
            2019-08-31   9988.727  10200.710220

            [185 rows x 2 columns]
            ```
        """
        return super().from_params(ts, window, ewm, **kwargs)

    @classmethod
    def from_combinations(cls, ts, windows, r, ewm=False, names=None, **kwargs):
        """Create multiple `MA` combinations according to `itertools.combinations`.

        Args:
            ts (pandas_like): Time series (such as price).
            windows (array_like of int): Size of the moving window. Must be multiple.
            r (int): The number of `MA` instances to combine.
            ewm (bool or array_like of bool): If `True`, uses exponential moving average, otherwise 
                uses simple moving average.
            names (list of str): A list of names for each `MA` instance.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            tuple of MA
        Example:
            ```python-repl
            >>> fast_ma, slow_ma = vbt.MA.from_combinations(price['Close'], 
            ...     [10, 20, 30], 2, ewm=[False, False, True], names=['fast', 'slow'])

            >>> print(fast_ma.ma)
            fast_window                    10          20
            fast_ewm         False      False       False
            Date                                         
            2019-02-28         NaN        NaN         NaN
            2019-03-01         NaN        NaN         NaN
            2019-03-02         NaN        NaN         NaN
            ...                ...        ...         ...
            2019-08-29   10155.972  10155.972  10447.3480
            2019-08-30   10039.466  10039.466  10359.5555
            2019-08-31    9988.727   9988.727  10264.9095

            [185 rows x 3 columns]

            >>> print(slow_ma.ma)
            slow_window          20                          30
            slow_ewm          False          True          True
            Date                                               
            2019-02-28          NaN           NaN           NaN
            2019-03-01          NaN           NaN           NaN
            2019-03-02          NaN           NaN           NaN
            ...                 ...           ...           ...
            2019-08-29   10447.3480  10423.585970  10423.585970
            2019-08-30   10359.5555  10370.333077  10370.333077
            2019-08-31   10264.9095  10322.612024  10322.612024

            [185 rows x 3 columns]

            ```

            The naive way without caching is the follows:
            ```py
            window_combs = itertools.combinations([10, 20, 30], 2)
            ewm_combs = itertools.combinations([False, False, True], 2)
            fast_windows, slow_windows = np.asarray(list(window_combs)).transpose()
            fast_ewms, slow_ewms = np.asarray(list(ewm_combs)).transpose()

            fast_ma = vbt.MA.from_params(price['Close'], 
            ...     fast_windows, fast_ewms, name='fast')
            slow_ma = vbt.MA.from_params(price['Close'], 
            ...     slow_windows, slow_ewms, name='slow')
            ```

            Having this, you can now compare these `MA` instances:
            ```python-repl
            >>> entry_signals = fast_ma.ma_above(slow_ma, crossover=True)
            >>> exit_signals = fast_ma.ma_below(slow_ma, crossover=True)

            >>> print(entry_signals)
            fast_window            10     20
            fast_ewm     False  False  False
            slow_window     20            30
            slow_ewm     False   True   True
            Date                            
            2019-02-28   False  False  False
            2019-03-01   False  False  False
            2019-03-02   False  False  False
            ...            ...    ...    ...
            2019-08-29   False  False  False
            2019-08-30   False  False  False
            2019-08-31   False  False  False

            [185 rows x 3 columns]
            ```

            Notice how `MA.ma_above` method created a new column hierarchy for you. You can now use
            it for indexing as follows:

            ```py
            fig = price['Close'].vbt.timeseries.plot(name='Price')
            fig = entry_signals[(10, False, 20, False)]\\
                .vbt.signals.plot_markers(price['Close'], signal_type='entry', fig=fig)
            fig = exit_signals[(10, False, 20, False)]\\
                .vbt.signals.plot_markers(price['Close'], signal_type='exit', fig=fig)

            fig.show()
            ```
            ![](img/MA_from_combinations.png)
        """

        if names is None:
            names = ['ma' + str(i+1) for i in range(r)]
        windows, ewm = reshape_fns.broadcast(windows, ewm, writeable=True)
        cache_dict = cls.from_params(ts, windows, ewm=ewm, return_cache=True, **kwargs)
        param_lists = zip(*itertools.combinations(zip(windows, ewm), r))
        mas = []
        for i, param_list in enumerate(param_lists):
            i_windows, i_ewm = zip(*param_list)
            mas.append(cls.from_params(ts, i_windows, ewm=i_ewm, cache=cache_dict, name=names[i], **kwargs))
        return tuple(mas)

    def plot(self,
             ts_trace_kwargs={},
             ma_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `MA.ma` against `MA.ts`.

        Args:
            ts_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `MA.ts`.
            ma_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `MA.ma`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.common.DefaultFigureWidget
        Example:
            ```py
            ma[(10, False)].plot()
            ```

            ![](img/MA.png)"""
        checks.assert_type(self.ts, pd.Series)
        checks.assert_type(self.ma, pd.Series)

        ts_trace_kwargs = {**dict(
            name=f'Price ({self.name})'
        ), **ts_trace_kwargs}
        ma_trace_kwargs = {**dict(
            name=f'MA ({self.name})'
        ), **ma_trace_kwargs}

        fig = self.ts.vbt.timeseries.plot(trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.timeseries.plot(trace_kwargs=ma_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(MA)

# ############# MSTD ############# #


@njit(cache=True)
def mstd_caching_nb(ts, windows, ewms):
    """Numba-compiled caching function for `MSTD`."""
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                mstd = timeseries.nb.ewm_std_nb(ts, windows[i], minp=windows[i])
            else:
                mstd = timeseries.nb.rolling_std_nb(ts, windows[i], minp=windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = mstd
    return cache_dict


@njit(cache=True)
def mstd_apply_func_nb(ts, window, ewm, cache_dict):
    """Numba-compiled apply function for `MSTD`."""
    return cache_dict[(window, int(ewm))]


MSTD = indicators.factory.IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['mstd'],
    name='mstd'
).from_apply_func(mstd_apply_func_nb, caching_func=mstd_caching_nb)


class MSTD(MSTD):
    """Standard deviation is an indicator that measures the size of an assets recent price moves 
    in order to predict how volatile the price may be in the future.

    Use `MSTD.from_params` method to run the indicator."""
    @classmethod
    def from_params(cls, ts, window, ewm=False, **kwargs):
        """Calculate moving standard deviation `MSTD.mstd` from time series `ts` and 
        parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window.
            ewm (bool or array_like): If `True`, uses exponential moving standard deviation, 
                otherwise uses simple moving standard deviation.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            MSTD
        Example:
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
            mstd_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `MSTD.mstd`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.common.DefaultFigureWidget
        Example:
            ```py
            mstd[(10, False)].plot()
            ```

            ![](img/MSTD.png)"""
        checks.assert_type(self.mstd, pd.Series)

        mstd_trace_kwargs = {**dict(
            name=f'MSTD ({self.name})'
        ), **mstd_trace_kwargs}

        fig = self.mstd.vbt.timeseries.plot(trace_kwargs=mstd_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(MSTD)

# ############# BollingerBands ############# #


@njit(cache=True)
def bb_caching_nb(ts, windows, ewms, alphas):
    """Numba-compiled caching function for `BollingerBands`."""
    ma_cache_dict = ma_caching_nb(ts, windows, ewms)
    mstd_cache_dict = mstd_caching_nb(ts, windows, ewms)
    return ma_cache_dict, mstd_cache_dict


@njit(cache=True)
def bb_apply_func_nb(ts, window, ewm, alpha, ma_cache_dict, mstd_cache_dict):
    """Numba-compiled apply function for `BollingerBands`."""
    # Calculate lower, middle and upper bands
    ma = np.copy(ma_cache_dict[(window, int(ewm))])
    mstd = np.copy(mstd_cache_dict[(window, int(ewm))])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


BollingerBands = indicators.factory.IndicatorFactory(
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
    """A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard 
    deviations (positively and negatively) away from a simple moving average (SMA) of the security's 
    price, but can be adjusted to user preferences.

    See [Bollinger Band®](https://www.investopedia.com/terms/b/bollingerbands.asp).

    Use `BollingerBands.from_params` method to run the indicator."""
    @classmethod
    def from_params(cls, ts, window=20, ewm=False, alpha=2, **kwargs):
        """Calculate moving average `BollingerBands.ma`, upper Bollinger Band `BollingerBands.upper_band`,
        lower Bollinger Band `BollingerBands.lower_band`, %b `BollingerBands.percent_b` and 
        bandwidth `BollingerBands.bandwidth` from time series `ts` and parameters `window`, `ewm` and `alpha`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window.
            ewm (bool or array_like): If `True`, uses exponential moving average and standard deviation, 
                otherwise uses simple moving average and standard deviation.
            alpha (int, float or array_like): Number of standard deviations.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            BollingerBands
        Example:
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
            ts_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.ts`.
            ma_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.ma`.
            upper_band_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.upper_band`.
            lower_band_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `BollingerBands.lower_band`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.common.DefaultFigureWidget
        Example:
            ```py
            bb[(10, False, 2)].plot()
            ```

            ![](img/BollingerBands.png)"""
        checks.assert_type(self.ts, pd.Series)
        checks.assert_type(self.ma, pd.Series)
        checks.assert_type(self.upper_band, pd.Series)
        checks.assert_type(self.lower_band, pd.Series)

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
            line=dict(color=defaults.layout['colorway'][1])
        ), **ma_trace_kwargs}
        ts_trace_kwargs = {**dict(
            name=f'Price ({self.name})',
            line=dict(color=defaults.layout['colorway'][0])
        ), **ts_trace_kwargs}

        fig = self.lower_band.vbt.timeseries.plot(trace_kwargs=lower_band_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.upper_band.vbt.timeseries.plot(trace_kwargs=upper_band_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.timeseries.plot(trace_kwargs=ma_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ts.vbt.timeseries.plot(trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(BollingerBands)

# ############# RSI ############# #


@njit(cache=True)
def rsi_caching_nb(ts, windows, ewms):
    """Numba-compiled caching function for `RSI`."""
    delta = timeseries.nb.diff_nb(ts)[1:, :]  # otherwise ewma will be all NaN
    up, down = delta.copy(), delta.copy()
    up = timeseries.nb.set_by_mask_nb(up, up < 0, 0)
    down = np.abs(timeseries.nb.set_by_mask_nb(down, down > 0, 0))
    # Cache
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                roll_up = timeseries.nb.ewm_mean_nb(up, windows[i], minp=windows[i])
                roll_down = timeseries.nb.ewm_mean_nb(down, windows[i], minp=windows[i])
            else:
                roll_up = timeseries.nb.rolling_mean_nb(up, windows[i], minp=windows[i])
                roll_down = timeseries.nb.rolling_mean_nb(down, windows[i], minp=windows[i])
            roll_up = timeseries.nb.prepend_nb(roll_up, 1, np.nan)  # bring to old shape
            roll_down = timeseries.nb.prepend_nb(roll_down, 1, np.nan)
            cache_dict[(windows[i], int(ewms[i]))] = roll_up, roll_down
    return cache_dict


@njit(cache=True)
def rsi_apply_func_nb(ts, window, ewm, cache_dict):
    """Numba-compiled apply function for `RSI`."""
    roll_up, roll_down = cache_dict[(window, int(ewm))]
    return 100 - 100 / (1 + roll_up / roll_down)


RSI = indicators.factory.IndicatorFactory(
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['rsi'],
    name='rsi'
).from_apply_func(rsi_apply_func_nb, caching_func=rsi_caching_nb)


class RSI(RSI):
    """The relative strength index (RSI) is a momentum indicator that measures the magnitude of 
    recent price changes to evaluate overbought or oversold conditions in the price of a stock 
    or other asset. The RSI is displayed as an oscillator (a line graph that moves between two 
    extremes) and can have a reading from 0 to 100.

    See [Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp).

    Use `RSI.from_params` methods to run the indicator."""
    @classmethod
    def from_params(cls, ts, window=14, ewm=False, **kwargs):
        """Calculate relative strength index `RSI.rsi` from time series `ts` and parameters `window` and `ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            window (int or array_like): Size of the moving window.
            ewm (bool or array_like): If `True`, uses exponential moving average, otherwise 
                simple moving average.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            RSI
        Example:
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
            trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `RSI.rsi`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.common.DefaultFigureWidget
        Example:
            ```py
            rsi[(10, False)].plot()
            ```

            ![](img/RSI.png)"""
        checks.assert_type(self.rsi, pd.Series)

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

# ############# Stochastic ############# #


@njit(cache=True)
def stoch_caching_nb(close_ts, high_ts, low_ts, k_windows, d_windows, d_ewms):
    """Numba-compiled caching function for `Stochastic`."""
    cache_dict = dict()
    for i in range(k_windows.shape[0]):
        if k_windows[i] not in cache_dict:
            roll_min = timeseries.nb.rolling_min_nb(low_ts, k_windows[i], minp=1) # min_periods=1
            roll_max = timeseries.nb.rolling_max_nb(high_ts, k_windows[i], minp=1)
            cache_dict[k_windows[i]] = roll_min, roll_max
    return cache_dict


@njit(cache=True)
def stoch_apply_func_nb(close_ts, high_ts, low_ts, k_window, d_window, d_ewm, cache_dict):
    """Numba-compiled apply function for `Stochastic`."""
    roll_min, roll_max = cache_dict[k_window]
    percent_k = 100 * (close_ts - roll_min) / (roll_max - roll_min)
    if d_ewm:
        percent_d = timeseries.nb.ewm_mean_nb(percent_k, d_window, minp=d_window)
    else:
        percent_d = timeseries.nb.rolling_mean_nb(percent_k, d_window, minp=d_window)
    return percent_k, percent_d


Stochastic = indicators.factory.IndicatorFactory(
    ts_names=['close_ts', 'high_ts', 'low_ts'],
    param_names=['k_window', 'd_window', 'd_ewm'],
    output_names=['percent_k', 'percent_d'],
    name='stoch'
).from_apply_func(stoch_apply_func_nb, caching_func=stoch_caching_nb)


class Stochastic(Stochastic):
    """A stochastic oscillator is a momentum indicator comparing a particular closing price of a security 
    to a range of its prices over a certain period of time. It is used to generate overbought and oversold 
    trading signals, utilizing a 0-100 bounded range of values.

    See [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp).

    Use `Stochastic.from_params` methods to run the indicator."""
    @classmethod
    def from_params(cls, close_ts, high_ts=None, low_ts=None, k_window=14, d_window=3, d_ewm=False, **kwargs):
        """Calculate %K `Stochastic.percent_k` and %D `Stochastic.percent_d` from time series `close_ts`, 
        `high_ts`, and `low_ts`, and parameters `k_window`, `d_window` and `d_ewm`.

        Args:
            close_ts (pandas_like): The last closing price.
            high_ts (pandas_like): The highest price. If None, uses `close_ts`.
            low_ts (pandas_like): The lowest price. If None, uses `close_ts`.
            k_window (int or array_like): Size of the moving window for %K.
            d_window (int or array_like): Size of the moving window for %D.
            d_ewm (bool or array_like): If `True`, uses exponential moving average for %D, otherwise 
                simple moving average.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            Stochastic
        Example:
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
            percent_k_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `Stochastic.percent_k`.
            percent_d_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `Stochastic.percent_d`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.common.DefaultFigureWidget
        Example:
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


fix_class_for_pdoc(Stochastic)

# ############# MACD ############# #


@njit(cache=True)
def macd_caching_nb(ts, fast_windows, slow_windows, signal_windows, macd_ewms, signal_ewms):
    """Numba-compiled caching function for `MACD`."""
    return ma_caching_nb(ts, np.concatenate((fast_windows, slow_windows)), np.concatenate((macd_ewms, macd_ewms)))


@njit(cache=True)
def macd_apply_func_nb(ts, fast_window, slow_window, signal_window, macd_ewm, signal_ewm, cache_dict):
    """Numba-compiled apply function for `MACD`."""
    fast_ma = cache_dict[(fast_window, int(macd_ewm))]
    slow_ma = cache_dict[(slow_window, int(macd_ewm))]
    macd_ts = fast_ma - slow_ma
    if signal_ewm:
        signal_ts = timeseries.nb.ewm_mean_nb(macd_ts, signal_window, minp=signal_window)
    else:
        signal_ts = timeseries.nb.rolling_mean_nb(macd_ts, signal_window, minp=signal_window)
    signal_ts[:max(fast_window, slow_window)+signal_window-2, :] = np.nan  # min_periodd
    return np.copy(fast_ma), np.copy(slow_ma), macd_ts, signal_ts


MACD = indicators.factory.IndicatorFactory(
    ts_names=['ts'],
    param_names=['fast_window', 'slow_window', 'signal_window', 'macd_ewm', 'signal_ewm'],
    output_names=['fast_ma', 'slow_ma', 'macd', 'signal'],
    name='macd',
    custom_properties=dict(
        histogram=lambda self: self.ts.vbt.wrap_array(self.macd.values - self.signal.values),
    )
).from_apply_func(macd_apply_func_nb, caching_func=macd_caching_nb)


class MACD(MACD):
    """Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that 
    shows the relationship between two moving averages of a security’s price.

    See [Moving Average Convergence Divergence – MACD](https://www.investopedia.com/terms/m/macd.asp).

    Use `MACD.from_params` methods to run the indicator."""
    @classmethod
    def from_params(cls, ts, fast_window=26, slow_window=12, signal_window=9, macd_ewm=True, signal_ewm=True, **kwargs):
        """Calculate fast moving average `MACD.fast_ma`, slow moving average `MACD.slow_ma`, MACD `MACD.macd`, 
        signal `MACD.signal` and histogram `MACD.histogram` from time series `ts` and parameters `fast_window`, 
        `slow_window`, `signal_window`, `macd_ewm` and `signal_ewm`.

        Args:
            ts (pandas_like): Time series (such as price).
            fast_window (int or array_like): Size of the fast moving window for MACD.
            slow_window (int or array_like): Size of the slow moving window for MACD.
            signal_window (int or array_like): Size of the moving window for signal.
            macd_ewm (bool or array_like): If `True`, uses exponential moving average for MACD, otherwise uses 
                simple moving average.
            signal_ewm (bool or array_like): If `True`, uses exponential moving average for signal, otherwise uses 
                simple moving average.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            MACD
        Example:
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
            macd_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `MACD.macd`.
            signal_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of 
                `MACD.signal`.
            histogram_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Bar`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Bar.html) of 
                `MACD.histogram`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.common.DefaultFigureWidget
        Example:
            ```py
            macd[(10, 20, 30, False, True)].plot()
            ```

            ![](img/MACD.png)"""
        checks.assert_type(self.macd, pd.Series)
        checks.assert_type(self.signal, pd.Series)
        checks.assert_type(self.histogram, pd.Series)

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
        hist_diff = timeseries.nb.diff_1d_nb(hist)
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

# ############# ATR ############# #


@njit(cache=True)
def atr_caching_nb(close_ts, high_ts, low_ts, windows, ewms):
    """Numba-compiled caching function for `ATR`."""
    # Calculate TR here instead of re-calculating it for each param in atr_apply_func_nb
    tr0 = high_ts - low_ts
    tr1 = np.abs(high_ts - timeseries.nb.fshift_nb(close_ts, 1))
    tr2 = np.abs(low_ts - timeseries.nb.fshift_nb(close_ts, 1))
    tr = timeseries.nb.nanmax_cube_nb(np.stack((tr0, tr1, tr2)))

    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                atr = timeseries.nb.ewm_mean_nb(tr, windows[i], minp=windows[i])
            else:
                atr = timeseries.nb.rolling_mean_nb(tr, windows[i], minp=windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = atr
    return tr, cache_dict


@njit(cache=True)
def atr_apply_func_nb(close_ts, high_ts, low_ts, window, ewm, tr, cache_dict):
    """Numba-compiled apply function for `ATR`."""
    return tr, cache_dict[(window, int(ewm))]


ATR = indicators.factory.IndicatorFactory(
    ts_names=['close_ts', 'high_ts', 'low_ts'],
    param_names=['window', 'ewm'],
    output_names=['tr', 'atr'],
    name='atr'
).from_apply_func(atr_apply_func_nb, caching_func=atr_caching_nb)


class ATR(ATR):
    """The average true range (ATR) is a technical analysis indicator that measures market volatility 
    by decomposing the entire range of an asset price for that period.

    See [Average True Range - ATR](https://www.investopedia.com/terms/a/atr.asp).

    Use `ATR.from_params` method to run the indicator."""
    @classmethod
    def from_params(cls, close_ts, high_ts, low_ts, window, ewm=True, **kwargs):
        """Calculate true range `ATR.tr` and average true range `ATR.atr` from time series `close_ts`, 
        `high_ts`, and `low_ts`, and parameters `window` and `ewm`.

        Args:
            close_ts (pandas_like): The last closing price.
            high_ts (pandas_like): The highest price.
            low_ts (pandas_like): The lowest price.
            window (int or array_like): Size of the moving window.
            ewm (bool or array_like): If `True`, uses exponential moving average, otherwise 
                simple moving average.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            ATR
        Example:
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
            tr_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) for `ATR.tr`.
            atr_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) for `ATR.atr`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.common.DefaultFigureWidget
        Example:
            ```py
            atr[(10, False)].plot()
            ```

            ![](img/ATR.png)"""
        checks.assert_type(self.tr, pd.Series)
        checks.assert_type(self.atr, pd.Series)

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

# ############# OBV ############# #


@njit(cache=True)
def obv_custom_func_nb(close_ts, volume_ts):
    """Numba-compiled custom calculation function for `OBV`."""
    obv = np.full_like(close_ts, np.nan)
    for col in range(close_ts.shape[1]):
        cumsum = 0
        for i in range(1, close_ts.shape[0]):
            if np.isnan(close_ts[i, col]) or np.isnan(close_ts[i-1, col]) or np.isnan(volume_ts[i, col]):
                continue
            if close_ts[i, col] > close_ts[i-1, col]:
                cumsum += volume_ts[i, col]
            elif close_ts[i, col] < close_ts[i-1, col]:
                cumsum += -volume_ts[i, col]
            obv[i, col] = cumsum
    return obv


OBV = indicators.factory.IndicatorFactory(
    ts_names=['close_ts', 'volume_ts'],
    param_names=[],
    output_names=['obv'],
    name='obv'
).from_custom_func(obv_custom_func_nb)


class OBV(OBV):
    """On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict 
    changes in stock price.

    See [On-Balance Volume (OBV)](https://www.investopedia.com/terms/o/onbalancevolume.asp).

    Use `OBV.from_params` methods to run the indicator."""
    @classmethod
    def from_params(cls, close_ts, volume_ts):
        """Calculate on-balance volume `OBV.obv` from time series `close_ts` and `volume_ts`, and no parameters.

        Args:
            close_ts (pandas_like): The last closing price.
            volume_ts (pandas_like): The volume.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            OBV
        Example:
            ```python-repl
            >>> obv = vbt.OBV.from_params(price['Close'], price['Volume'])

            >>> print(obv.obv)
            Date
            2019-02-28             NaN
            2019-03-01    7.661248e+09
            2019-03-02    1.524003e+10
            2019-03-03    7.986476e+09
            2019-03-04   -1.042700e+09
                                   ...     
            2019-08-27    5.613088e+11
            2019-08-28    5.437050e+11
            2019-08-29    5.266592e+11
            2019-08-30    5.402544e+11
            2019-08-31    5.517092e+11
            Name: (Close, Volume), Length: 185, dtype: float64
            ```
        """
        return super().from_params(close_ts, volume_ts)

    def plot(self,
             obv_trace_kwargs={},
             fig=None,
             **layout_kwargs):
        """Plot `OBV.obv`.

        Args:
            obv_trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html) of `OBV.obv`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Returns:
            vectorbt.widgets.common.DefaultFigureWidget
        Example:
            ```py
            obv.plot()
            ```

            ![](img/OBV.png)"""
        checks.assert_type(self.obv, pd.Series)

        obv_trace_kwargs = {**dict(
            name=f'OBV ({self.name})'
        ), **obv_trace_kwargs}

        fig = self.obv.vbt.timeseries.plot(trace_kwargs=obv_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_pdoc(OBV)
