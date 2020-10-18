"""Indicators built with `vectorbt.indicators.factory.IndicatorFactory`.

```python-repl
>>> import vectorbt as vbt
>>> from datetime import datetime
>>> import yfinance as yf

>>> ticker = yf.Ticker("BTC-USD")
>>> price = ticker.history(start=datetime(2019, 3, 1), end=datetime(2019, 9, 1))
>>> price = price[['Open', 'High', 'Low', 'Close', 'Volume']]
>>> price
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

>>> price['Close'].vbt.plot()
```
![](/vectorbt/docs/img/Indicators_price.png)"""

import numpy as np
import plotly.graph_objects as go

from vectorbt import defaults
from vectorbt.utils.config import merge_kwargs
from vectorbt.utils.docs import fix_class_for_docs
from vectorbt.generic import nb as generic_nb
from vectorbt.indicators.factory import IndicatorFactory
from vectorbt.indicators import nb

# ############# MA ############# #


MA = IndicatorFactory(
    class_name='MA',
    module_name=__name__,
    short_name='ma',
    input_names=['close'],
    param_names=['window', 'ewm'],
    output_names=['ma']
).from_apply_func(
    nb.ma_apply_nb,
    cache_func=nb.ma_cache_nb,
    ewm=False
)


class MA(MA):
    """A moving average (MA) is a widely used indicator in technical analysis that helps smooth out 
    price action by filtering out the “noise” from random short-term price fluctuations. 

    See [Moving Average (MA)](https://www.investopedia.com/terms/m/movingaverage.asp).

    Use `MA.run` or `MA.run_combs` to run the indicator."""

    def plot(self,
             close_trace_kwargs=None,
             ma_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `MA.ma` against `MA.close`.

        Args:
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.close`.
            ma_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.ma`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> vbt.MA.run(price['Close'], 10).plot()
            ```

            ![](/vectorbt/docs/img/MA.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if ma_trace_kwargs is None:
            ma_trace_kwargs = {}
        close_trace_kwargs = merge_kwargs(dict(name='Close'), close_trace_kwargs)
        ma_trace_kwargs = merge_kwargs(dict(name='MA'), ma_trace_kwargs)

        fig = self.close.vbt.plot(trace_kwargs=close_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.plot(trace_kwargs=ma_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(MA)

# ############# MSTD ############# #


MSTD = IndicatorFactory(
    class_name='MSTD',
    module_name=__name__,
    short_name='mstd',
    input_names=['close'],
    param_names=['window', 'ewm'],
    output_names=['mstd']
).from_apply_func(
    nb.mstd_apply_nb,
    cache_func=nb.mstd_cache_nb,
    ewm=False
)


class MSTD(MSTD):
    """Standard deviation is an indicator that measures the size of an assets recent price moves
    in order to predict how volatile the price may be in the future.

    Use `MSTD.run` or `MSTD.run_combs` to run the indicator."""

    def plot(self,
             mstd_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `MSTD.mstd`.

        Args:
            mstd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MSTD.mstd`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> vbt.MSTD.run(price['Close'], 10).plot()
            ```

            ![](/vectorbt/docs/img/MSTD.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if mstd_trace_kwargs is None:
            mstd_trace_kwargs = {}
        mstd_trace_kwargs = merge_kwargs(dict(name='MSTD'), mstd_trace_kwargs)

        fig = self.mstd.vbt.plot(trace_kwargs=mstd_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(MSTD)

# ############# BBANDS ############# #


BBANDS = IndicatorFactory(
    class_name='BBANDS',
    module_name=__name__,
    short_name='bb',
    input_names=['close'],
    param_names=['window', 'ewm', 'alpha'],
    output_names=['middle', 'upper', 'lower'],
    custom_output_funcs=dict(
        percent_b=lambda self: self.wrapper.wrap(
            (self.close.values - self.lower.values) / (self.upper.values - self.lower.values)),
        bandwidth=lambda self: self.wrapper.wrap(
            (self.upper.values - self.lower.values) / self.middle.values)
    )
).from_apply_func(
    nb.bb_apply_nb,
    cache_func=nb.bb_cache_nb,
    window=20,
    ewm=False,
    alpha=2
)


class BBANDS(BBANDS):
    """A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard
    deviations (positively and negatively) away from a simple moving average (SMA) of the security's
    price, but can be adjusted to user preferences.

    See [Bollinger Band®](https://www.investopedia.com/terms/b/bollingerbands.asp).

    Use `BBANDS.run` or `BBANDS.run_combs` to run the indicator."""

    def plot(self,
             close_trace_kwargs=None,
             middle_trace_kwargs=None,
             upper_trace_kwargs=None,
             lower_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `BBANDS.middle`, `BBANDS.upper` and `BBANDS.lower` against
        `BBANDS.close`.

        Args:
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.close`.
            middle_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.middle`.
            upper_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.upper`.
            lower_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.lower`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> vbt.BBANDS.run(price['Close']).plot()
            ```

            ![](/vectorbt/docs/img/BBANDS.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if middle_trace_kwargs is None:
            middle_trace_kwargs = {}
        if upper_trace_kwargs is None:
            upper_trace_kwargs = {}
        if lower_trace_kwargs is None:
            lower_trace_kwargs = {}
        lower_trace_kwargs = merge_kwargs(dict(
            name='Lower Band',
            line=dict(color='silver')
        ), lower_trace_kwargs)
        upper_trace_kwargs = merge_kwargs(dict(
            name='Upper Band',
            line=dict(color='silver'),
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.25)'
        ), upper_trace_kwargs)  # default kwargs
        middle_trace_kwargs = merge_kwargs(dict(
            name='Middle Band',
            line=dict(color=defaults.layout['colorway'][1])
        ), middle_trace_kwargs)
        close_trace_kwargs = merge_kwargs(dict(
            name='Close',
            line=dict(color=defaults.layout['colorway'][0])
        ), close_trace_kwargs)

        fig = self.lower.vbt.plot(trace_kwargs=lower_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.upper.vbt.plot(trace_kwargs=upper_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.middle.vbt.plot(trace_kwargs=middle_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.close.vbt.plot(trace_kwargs=close_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(BBANDS)

# ############# RSI ############# #


RSI = IndicatorFactory(
    class_name='RSI',
    module_name=__name__,
    short_name='rsi',
    input_names=['close'],
    param_names=['window', 'ewm'],
    output_names=['rsi']
).from_apply_func(
    nb.rsi_apply_nb,
    cache_func=nb.rsi_cache_nb,
    window=14,
    ewm=False
)


class RSI(RSI):
    """The relative strength index (RSI) is a momentum indicator that measures the magnitude of
    recent price changes to evaluate overbought or oversold conditions in the price of a stock
    or other asset. The RSI is displayed as an oscillator (a line graph that moves between two
    extremes) and can have a reading from 0 to 100.

    See [Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp).

    Use `RSI.run` or `RSI.run_combs` to run the indicator."""

    def plot(self,
             levels=(30, 70),
             rsi_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `RSI.rsi`.

        Args:
            levels (tuple): Two extremes: bottom and top.
            rsi_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `RSI.rsi`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> vbt.RSI.run(price['Close']).plot()
            ```

            ![](/vectorbt/docs/img/RSI.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if rsi_trace_kwargs is None:
            rsi_trace_kwargs = {}
        rsi_trace_kwargs = merge_kwargs(dict(name='RSI'), rsi_trace_kwargs)

        layout_kwargs = merge_kwargs(dict(yaxis=dict(range=[-5, 105])), layout_kwargs)
        fig = self.rsi.vbt.plot(trace_kwargs=rsi_trace_kwargs, fig=fig, **layout_kwargs)

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

# ############# STOCH ############# #


STOCH = IndicatorFactory(
    class_name='STOCH',
    module_name=__name__,
    short_name='stoch',
    input_names=['high', 'low', 'close'],
    param_names=['k_window', 'd_window', 'd_ewm'],
    output_names=['percent_k', 'percent_d']
).from_apply_func(
    nb.stoch_apply_nb,
    cache_func=nb.stoch_cache_nb,
    k_window=14,
    d_window=3,
    d_ewm=False
)


class STOCH(STOCH):
    """A stochastic oscillator is a momentum indicator comparing a particular closing price of a security
    to a range of its prices over a certain period of time. It is used to generate overbought and oversold
    trading signals, utilizing a 0-100 bounded range of values.

    See [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp).

    Use `STOCH.run` or `STOCH.run_combs` to run the indicator."""

    def plot(self,
             levels=(30, 70),
             percent_k_trace_kwargs=None,
             percent_d_trace_kwargs=None,
             shape_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `STOCH.percent_k` and `STOCH.percent_d`.

        Args:
            levels (tuple): Two extremes: bottom and top.
            percent_k_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `STOCH.percent_k`.
            percent_d_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `STOCH.percent_d`.
            shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zone between levels.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> vbt.STOCH.run(price['High'], price['Low'], price['Close']).plot()
            ```

            ![](/vectorbt/docs/img/STOCH.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if percent_k_trace_kwargs is None:
            percent_k_trace_kwargs = {}
        if percent_d_trace_kwargs is None:
            percent_d_trace_kwargs = {}
        if shape_kwargs is None:
            shape_kwargs = {}
        percent_k_trace_kwargs = merge_kwargs(dict(name='%K'), percent_k_trace_kwargs)
        percent_d_trace_kwargs = merge_kwargs(dict(name='%D'), percent_d_trace_kwargs)

        layout_kwargs = merge_kwargs(dict(yaxis=dict(range=[-5, 105])), layout_kwargs)
        fig = self.percent_k.vbt.plot(trace_kwargs=percent_k_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.percent_d.vbt.plot(trace_kwargs=percent_d_trace_kwargs, fig=fig, **layout_kwargs)

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


fix_class_for_docs(STOCH)

# ############# MACD ############# #


MACD = IndicatorFactory(
    class_name='MACD',
    module_name=__name__,
    short_name='macd',
    input_names=['close'],
    param_names=['fast_window', 'slow_window', 'signal_window', 'macd_ewm', 'signal_ewm'],
    output_names=['macd', 'signal'],
    custom_output_funcs=dict(
        hist=lambda self: self.wrapper.wrap(self.macd.values - self.signal.values),
    )
).from_apply_func(
    nb.macd_apply_nb,
    cache_func=nb.macd_cache_nb,
    fast_window=26,
    slow_window=12,
    signal_window=9,
    macd_ewm=False,
    signal_ewm=False
)


class MACD(MACD):
    """Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that
    shows the relationship between two moving averages of a security’s price.

    See [Moving Average Convergence Divergence – MACD](https://www.investopedia.com/terms/m/macd.asp).

    Use `MACD.run` or `MACD.run_combs` to run the indicator."""

    def plot(self,
             macd_trace_kwargs=None,
             signal_trace_kwargs=None,
             hist_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `MACD.macd`, `MACD.signal` and `MACD.hist`.

        Args:
            macd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MACD.macd`.
            signal_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MACD.signal`.
            hist_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar` for `MACD.hist`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> vbt.MACD.run(price['Close']).plot()
            ```

            ![](/vectorbt/docs/img/MACD.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if macd_trace_kwargs is None:
            macd_trace_kwargs = {}
        if signal_trace_kwargs is None:
            signal_trace_kwargs = {}
        if hist_trace_kwargs is None:
            hist_trace_kwargs = {}
        macd_trace_kwargs = merge_kwargs(dict(name='MACD'), macd_trace_kwargs)
        signal_trace_kwargs = merge_kwargs(dict(name='Signal'), signal_trace_kwargs)
        hist_trace_kwargs = merge_kwargs(dict(name='Histogram'), hist_trace_kwargs)

        layout_kwargs = merge_kwargs(dict(bargap=0), layout_kwargs)
        fig = self.macd.vbt.plot(trace_kwargs=macd_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.signal.vbt.plot(trace_kwargs=signal_trace_kwargs, fig=fig, **layout_kwargs)

        # Plot hist
        hist = self.hist.values
        hist_diff = generic_nb.diff_1d_nb(hist)
        marker_colors = np.full(hist.shape, 'silver', dtype=np.object)
        marker_colors[(hist > 0) & (hist_diff > 0)] = 'green'
        marker_colors[(hist > 0) & (hist_diff <= 0)] = 'lightgreen'
        marker_colors[(hist < 0) & (hist_diff < 0)] = 'red'
        marker_colors[(hist < 0) & (hist_diff >= 0)] = 'lightcoral'

        hist_bar = go.Bar(
            x=self.hist.index,
            y=self.hist.values,
            marker_color=marker_colors,
            marker_line_width=0
        )
        hist_bar.update(**hist_trace_kwargs)
        fig.add_trace(hist_bar)

        return fig


fix_class_for_docs(MACD)

# ############# ATR ############# #


ATR = IndicatorFactory(
    class_name='ATR',
    module_name=__name__,
    short_name='atr',
    input_names=['high', 'low', 'close'],
    param_names=['window', 'ewm'],
    output_names=['tr', 'atr']
).from_apply_func(
    nb.atr_apply_nb,
    cache_func=nb.atr_cache_nb,
    ewm=False
)


class ATR(ATR):
    """The average true range (ATR) is a technical analysis indicator that measures market volatility
    by decomposing the entire range of an asset price for that period.

    See [Average True Range - ATR](https://www.investopedia.com/terms/a/atr.asp).

    Use `ATR.run` or `ATR.run_combs` to run the indicator."""

    def plot(self,
             tr_trace_kwargs=None,
             atr_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `ATR.tr` and `ATR.atr`.

        Args:
            tr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.tr`.
            atr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.atr`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```python-repl
            >>> vbt.ATR.run(price['High'], price['Low'], price['Close'], 10).plot()
            ```

            ![](/vectorbt/docs/img/ATR.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if tr_trace_kwargs is None:
            tr_trace_kwargs = {}
        if atr_trace_kwargs is None:
            atr_trace_kwargs = {}
        tr_trace_kwargs = merge_kwargs(dict(name='TR'), tr_trace_kwargs)
        atr_trace_kwargs = merge_kwargs(dict(name='ATR'), atr_trace_kwargs)

        fig = self.tr.vbt.plot(trace_kwargs=tr_trace_kwargs, fig=fig, **layout_kwargs)
        fig = self.atr.vbt.plot(trace_kwargs=atr_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(ATR)

# ############# OBV ############# #


OBV = IndicatorFactory(
    class_name='OBV',
    module_name=__name__,
    short_name='obv',
    input_names=['close', 'volume'],
    param_names=[],
    output_names=['obv'],
).from_custom_func(nb.obv_custom_nb)


class OBV(OBV):
    """On-balance volume (OBV) is a technical trading momentum indicator that uses volume flow to predict
    changes in stock price.

    See [On-Balance Volume (OBV)](https://www.investopedia.com/terms/o/onbalancevolume.asp).

    Use `OBV.run` to run the indicator."""

    def plot(self,
             obv_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `OBV.obv`.

        Args:
            obv_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OBV.obv`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            >>> vbt.OBV.run(price['Close'], price['Volume']).plot()
            ```

            ![](/vectorbt/docs/img/OBV.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing.")

        if obv_trace_kwargs is None:
            obv_trace_kwargs = {}
        obv_trace_kwargs = merge_kwargs(dict(name='OBV'), obv_trace_kwargs)

        fig = self.obv.vbt.plot(trace_kwargs=obv_trace_kwargs, fig=fig, **layout_kwargs)

        return fig


fix_class_for_docs(OBV)
