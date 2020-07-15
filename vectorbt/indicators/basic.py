"""Indicators built with `vectorbt.indicators.factory.IndicatorFactory`.

```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from datetime import datetime, timedelta
>>> import itertools
>>> from numba import njit
>>> import yfinance as yf

>>> ticker = yf.Ticker("BTC-USD")
>>> price = ticker.history(start=datetime(2019, 3, 1), end=datetime(2019, 9, 1))
>>> price = price[['Open', 'High', 'Low', 'Close', 'Volume']]
>>> print(price)
                Open      High       Low     Close       Volume
Date
2019-02-28   3848.26   3906.06   3845.82   3854.79   8399767798
2019-03-01   3853.76   3907.80   3851.69   3859.58   7661247975
2019-03-02   3855.32   3874.61   3832.13   3864.42   7578786075
...              ...       ...       ...       ...          ...
2019-08-29   9756.79   9756.79   9421.63   9510.20  17045878500
2019-08-30   9514.84   9656.12   9428.30   9598.17  13595263986
2019-08-31   9597.54   9673.22   9531.80   9630.66  11454806419

[185 rows x 5 columns]

>>> price['Close'].vbt.tseries.plot()
```
![](/vectorbt/docs/img/Indicators_price.png)"""

import numpy as np
import itertools
import plotly.graph_objects as go

from vectorbt import tseries, defaults
from vectorbt.utils.config import merge_kwargs
from vectorbt.utils.docs import fix_class_for_docs
from vectorbt.base import reshape_fns
from vectorbt.indicators.factory import IndicatorFactory, build_param_product
from vectorbt.indicators import nb

# ############# MA ############# #


MA = IndicatorFactory(
    class_name='MA',
    module_name=__name__,
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['ma'],
    name='ma'
).from_apply_func(nb.ma_apply_nb, caching_func=nb.ma_caching_nb)


class MA(MA):
    """A moving average (MA) is a widely used indicator in technical analysis that helps smooth out 
    price action by filtering out the “noise” from random short-term price fluctuations. 

    See [Moving Average (MA)](https://www.investopedia.com/terms/m/movingaverage.asp).

    Use `MA.from_params` or `MA.from_combs` methods to run the indicator."""

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
    def from_combs(cls, ts, windows, r, ewm=False, param_product=False, names=None, **kwargs):
        """Create multiple `MA` combinations according to `itertools.combinations`.

        Args:
            ts (pandas_like): Time series (such as price).
            windows (array_like of int): Size of the moving window.
            r (int): The number of `MA` instances to combine.
            ewm (bool or array_like of bool): If `True`, uses exponential moving average, otherwise 
                uses simple moving average.
            param_product (bool): If `True`, builds a Cartesian product out of all parameters.
            names (list of str): A list of names for each `MA` instance.
            **kwargs: Keyword arguments passed to `vectorbt.indicators.factory.from_params_pipeline.`
        Returns:
            tuple of MA
        Example:
            ```python-repl
            >>> fast_ma, slow_ma = vbt.MA.from_combs(price['Close'],
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
            >>> entry_signals = fast_ma.ma_above(slow_ma, crossed=True)
            >>> exit_signals = fast_ma.ma_below(slow_ma, crossed=True)

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
            fig = price['Close'].vbt.tseries.plot(name='Price')
            fig = entry_signals[(10, False, 20, False)]\\
                .vbt.signals.plot_as_markers(price['Close'], signal_type='entry', fig=fig)
            fig = exit_signals[(10, False, 20, False)]\\
                .vbt.signals.plot_as_markers(price['Close'], signal_type='exit', fig=fig)

            fig.show()
            ```
            ![](/vectorbt/docs/img/MA_from_combs.png)
        """

        if names is None:
            names = ['ma' + str(i + 1) for i in range(r)]
        param_list = [windows, ewm]
        if param_product:
            param_list = build_param_product(param_list)
        else:
            param_list = reshape_fns.broadcast(*param_list, writeable=True)
        windows, ewm = param_list
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
             **layout_kwargs):  # pragma: no cover
        """Plot `MA.ma` against `MA.ts`.

        Args:
            ts_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.ts`.
            ma_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.ma`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            ma[(10, False)].plot()
            ```

            ![](/vectorbt/docs/img/MA.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        ts_trace_kwargs = merge_kwargs(dict(
            name=f'Price ({self.name})'
        ), ts_trace_kwargs)
        ma_trace_kwargs = merge_kwargs(dict(
            name=f'MA ({self.name})'
        ), ma_trace_kwargs)

        fig = self.ts.vbt.tseries.plot(trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.tseries.plot(trace_kwargs=ma_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(MA)

# ############# MSTD ############# #


MSTD = IndicatorFactory(
    class_name='MSTD',
    module_name=__name__,
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['mstd'],
    name='mstd'
).from_apply_func(nb.mstd_apply_nb, caching_func=nb.mstd_caching_nb)


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
             **layout_kwargs):  # pragma: no cover
        """Plot `MSTD.mstd`.

        Args:
            mstd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MSTD.mstd`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            mstd[(10, False)].plot()
            ```

            ![](/vectorbt/docs/img/MSTD.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        mstd_trace_kwargs = merge_kwargs(dict(
            name=f'MSTD ({self.name})'
        ), mstd_trace_kwargs)

        fig = self.mstd.vbt.tseries.plot(trace_kwargs=mstd_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(MSTD)

# ############# BollingerBands ############# #


BollingerBands = IndicatorFactory(
    class_name='BollingerBands',
    module_name=__name__,
    ts_names=['ts'],
    param_names=['window', 'ewm', 'alpha'],
    output_names=['middle', 'upper', 'lower'],
    name='bb',
    custom_outputs=dict(
        percent_b=lambda self: self.wrapper.wrap(
            (self.ts.values - self.lower.values) / (self.upper.values - self.lower.values)),
        bandwidth=lambda self: self.wrapper.wrap(
            (self.upper.values - self.lower.values) / self.middle.values)
    )
).from_apply_func(nb.bb_apply_nb, caching_func=nb.bb_caching_nb)


class BollingerBands(BollingerBands):
    """A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard 
    deviations (positively and negatively) away from a simple moving average (SMA) of the security's 
    price, but can be adjusted to user preferences.

    See [Bollinger Band®](https://www.investopedia.com/terms/b/bollingerbands.asp).

    Use `BollingerBands.from_params` method to run the indicator."""

    @classmethod
    def from_params(cls, ts, window=20, ewm=False, alpha=2, **kwargs):
        """Calculate middle line `BollingerBands.middle`, upper Bollinger Band `BollingerBands.upper`,
        lower Bollinger Band `BollingerBands.lower`, %b `BollingerBands.percent_b` and bandwidth
        `BollingerBands.bandwidth` from time series `ts` and parameters `window`, `ewm` and `alpha`.

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

            >>> print(bb.middle)
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

            >>> print(bb.upper)
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

            >>> print(bb.lower)
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
            ...              ...       ...
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
             middle_trace_kwargs={},
             upper_trace_kwargs={},
             lower_trace_kwargs={},
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `BollingerBands.middle`, `BollingerBands.upper` and `BollingerBands.lower` against
        `BollingerBands.ts`.

        Args:
            ts_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BollingerBands.ts`.
            middle_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BollingerBands.middle`.
            upper_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BollingerBands.upper`.
            lower_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BollingerBands.lower`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            bb[(10, False, 2)].plot()
            ```

            ![](/vectorbt/docs/img/BollingerBands.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        lower_trace_kwargs = merge_kwargs(dict(
            name=f'Lower Band ({self.name})',
            line=dict(color='silver')
        ), lower_trace_kwargs)
        upper_trace_kwargs = merge_kwargs(dict(
            name=f'Upper Band ({self.name})',
            line=dict(color='silver'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.25)'
        ), upper_trace_kwargs)  # default kwargs
        middle_trace_kwargs = merge_kwargs(dict(
            name=f'Middle Band ({self.name})',
            line=dict(color=defaults.layout['colorway'][1])
        ), middle_trace_kwargs)
        ts_trace_kwargs = merge_kwargs(dict(
            name=f'Price ({self.name})',
            line=dict(color=defaults.layout['colorway'][0])
        ), ts_trace_kwargs)

        fig = self.lower.vbt.tseries.plot(trace_kwargs=lower_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.upper.vbt.tseries.plot(trace_kwargs=upper_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.middle.vbt.tseries.plot(trace_kwargs=middle_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ts.vbt.tseries.plot(trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(BollingerBands)

# ############# RSI ############# #


RSI = IndicatorFactory(
    class_name='RSI',
    module_name=__name__,
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['rsi'],
    name='rsi'
).from_apply_func(nb.rsi_apply_nb, caching_func=nb.rsi_caching_nb)


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
             **layout_kwargs):  # pragma: no cover
        """Plot `RSI.rsi`.

        Args:
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `RSI.rsi`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            rsi[(10, False)].plot()
            ```

            ![](/vectorbt/docs/img/RSI.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        rsi_trace_kwargs = merge_kwargs(dict(
            name=f'RSI ({self.name})'
        ), rsi_trace_kwargs)

        layout_kwargs = merge_kwargs(dict(yaxis=dict(range=[-5, 105])), layout_kwargs)
        fig = self.rsi.vbt.tseries.plot(trace_kwargs=rsi_trace_kwargs, fig=fig, **layout_kwargs)

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
            opacity=0.15,
            layer="below",
            line_width=0,
        )

        return fig


fix_class_for_docs(RSI)

# ############# Stochastic ############# #


Stochastic = IndicatorFactory(
    class_name='Stochastic',
    module_name=__name__,
    ts_names=['close_ts', 'high_ts', 'low_ts'],
    param_names=['k_window', 'd_window', 'd_ewm'],
    output_names=['percent_k', 'percent_d'],
    name='stoch'
).from_apply_func(nb.stoch_apply_nb, caching_func=nb.stoch_caching_nb)


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
             shape_kwargs={},
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `Stochastic.percent_k` and `Stochastic.percent_d`.

        Args:
            percent_k_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Stochastic.percent_k`.
            percent_d_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `Stochastic.percent_d`.
            shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zone between levels.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            stoch[(10, 2, False)].plot(levels=(20, 80))
            ```

            ![](/vectorbt/docs/img/Stochastic.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        percent_k_trace_kwargs = merge_kwargs(dict(
            name=f'%K ({self.name})'
        ), percent_k_trace_kwargs)
        percent_d_trace_kwargs = merge_kwargs(dict(
            name=f'%D ({self.name})'
        ), percent_d_trace_kwargs)

        layout_kwargs = merge_kwargs(dict(yaxis=dict(range=[-5, 105])), layout_kwargs)
        fig = self.percent_k.vbt.tseries.plot(trace_kwargs=percent_k_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.percent_d.vbt.tseries.plot(trace_kwargs=percent_d_trace_kwargs, fig=fig, **layout_kwargs)

        # Plot levels
        # Fill void between levels
        shape_kwargs = merge_kwargs(dict(
            type="rect",
            xref="x",
            yref="y",
            x0=self.percent_k.index[0],
            y0=levels[0],
            x1=self.percent_k.index[-1],
            y1=levels[1],
            fillcolor="purple",
            opacity=0.15,
            layer="below",
            line_width=0,
        ), shape_kwargs)
        fig.add_shape(**shape_kwargs)

        return fig


fix_class_for_docs(Stochastic)

# ############# MACD ############# #


MACD = IndicatorFactory(
    class_name='MACD',
    module_name=__name__,
    ts_names=['ts'],
    param_names=['fast_window', 'slow_window', 'signal_window', 'macd_ewm', 'signal_ewm'],
    output_names=['macd', 'signal'],
    name='macd',
    custom_outputs=dict(
        histogram=lambda self: self.wrapper.wrap(self.macd.values - self.signal.values),
    )
).from_apply_func(nb.macd_apply_nb, caching_func=nb.macd_caching_nb)


class MACD(MACD):
    """Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that 
    shows the relationship between two moving averages of a security’s price.

    See [Moving Average Convergence Divergence – MACD](https://www.investopedia.com/terms/m/macd.asp).

    Use `MACD.from_params` methods to run the indicator."""

    @classmethod
    def from_params(cls, ts, fast_window=26, slow_window=12, signal_window=9, macd_ewm=True, signal_ewm=True, **kwargs):
        """Calculate MACD `MACD.macd`, signal `MACD.signal` and histogram `MACD.histogram` from time series `ts`
        and parameters `fast_window`, `slow_window`, `signal_window`, `macd_ewm` and `signal_ewm`.

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
             **layout_kwargs):  # pragma: no cover
        """Plot `MACD.macd`, `MACD.signal` and `MACD.histogram`.

        Args:
            macd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MACD.macd`.
            signal_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MACD.signal`.
            histogram_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar` for `MACD.histogram`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            macd[(10, 20, 30, False, True)].plot()
            ```

            ![](/vectorbt/docs/img/MACD.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        macd_trace_kwargs = merge_kwargs(dict(
            name=f'MACD ({self.name})'
        ), macd_trace_kwargs)
        signal_trace_kwargs = merge_kwargs(dict(
            name=f'Signal ({self.name})'
        ), signal_trace_kwargs)
        histogram_trace_kwargs = merge_kwargs(dict(
            name=f'Histogram ({self.name})',
            showlegend=False
        ), histogram_trace_kwargs)

        layout_kwargs = merge_kwargs(dict(bargap=0), layout_kwargs)
        fig = self.macd.vbt.tseries.plot(trace_kwargs=macd_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.signal.vbt.tseries.plot(trace_kwargs=signal_trace_kwargs, fig=fig, **layout_kwargs)

        # Plot histogram
        hist = self.histogram.values
        hist_diff = tseries.nb.diff_1d_nb(hist)
        marker_colors = np.full(hist.shape, 'silver', dtype=np.object)
        marker_colors[(hist > 0) & (hist_diff > 0)] = 'green'
        marker_colors[(hist > 0) & (hist_diff <= 0)] = 'lightgreen'
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


fix_class_for_docs(MACD)

# ############# ATR ############# #


ATR = IndicatorFactory(
    class_name='ATR',
    module_name=__name__,
    ts_names=['close_ts', 'high_ts', 'low_ts'],
    param_names=['window', 'ewm'],
    output_names=['tr', 'atr'],
    name='atr'
).from_apply_func(nb.atr_apply_nb, caching_func=nb.atr_caching_nb)


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
             **layout_kwargs):  # pragma: no cover
        """Plot `ATR.tr` and `ATR.atr`.

        Args:
            tr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.tr`.
            atr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.atr`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            atr[(10, False)].plot()
            ```

            ![](/vectorbt/docs/img/ATR.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        tr_trace_kwargs = merge_kwargs(dict(
            name=f'TR ({self.name})'
        ), tr_trace_kwargs)
        atr_trace_kwargs = merge_kwargs(dict(
            name=f'ATR ({self.name})'
        ), atr_trace_kwargs)

        fig = self.tr.vbt.tseries.plot(trace_kwargs=tr_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.atr.vbt.tseries.plot(trace_kwargs=atr_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(ATR)

# ############# OBV ############# #


OBV = IndicatorFactory(
    class_name='OBV',
    module_name=__name__,
    ts_names=['close_ts', 'volume_ts'],
    param_names=[],
    output_names=['obv'],
    name='obv'
).from_custom_func(nb.obv_custom_func_nb)


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
                                   ...
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
             **layout_kwargs):  # pragma: no cover
        """Plot `OBV.obv`.

        Args:
            obv_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OBV.obv`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            obv.plot()
            ```

            ![](/vectorbt/docs/img/OBV.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        obv_trace_kwargs = merge_kwargs(dict(
            name=f'OBV ({self.name})'
        ), obv_trace_kwargs)

        fig = self.obv.vbt.tseries.plot(trace_kwargs=obv_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(OBV)
