"""Indicators built with `vectorbt.indicators.factory.IndicatorFactory`.

```python-repl
>>> import vectorbt as vbt
>>> from datetime import datetime

>>> start = datetime(2019, 3, 1)
>>> end = datetime(2019, 9, 1)
>>> cols = ['Open', 'High', 'Low', 'Close', 'Volume']
>>> ohlcv = vbt.utils.data.download("BTC-USD", start=start, end=end, cols=cols)
>>> ohlcv
                    Open          High           Low         Close  \
Date
2019-02-28   3848.261963   3906.058350   3845.821289   3854.785400
2019-03-01   3853.757080   3907.795410   3851.692383   3859.583740
2019-03-02   3855.318115   3874.607422   3832.127930   3864.415039
2019-03-03   3862.266113   3875.483643   3836.905762   3847.175781
2019-03-04   3845.091553   3867.381836   3733.749756   3761.557129
...                  ...           ...           ...           ...
2019-08-27  10372.826172  10381.328125  10087.300781  10185.500000
2019-08-28  10203.426758  10279.366211   9716.656250   9754.422852
2019-08-29   9756.786133   9756.786133   9421.629883   9510.200195
2019-08-30   9514.844727   9656.124023   9428.302734   9598.173828
2019-08-31   9597.539062   9673.220703   9531.799805   9630.664062

                 Volume
Date
2019-02-28   8399767798
2019-03-01   7661247975
2019-03-02   7578786076
2019-03-03   7253558152
2019-03-04   9029175788
...                 ...
2019-08-27  14762609503
2019-08-28  17603790323
2019-08-29  17045878501
2019-08-30  13595263986
2019-08-31  11454806419

[185 rows x 5 columns]

>>> ohlcv.vbt.ohlcv.plot()
```
![](/vectorbt/docs/img/basic_price.png)"""

import numpy as np
import plotly.graph_objects as go

from vectorbt.utils.config import merge_dicts
from vectorbt.utils.docs import fix_class_for_docs
from vectorbt.utils.widgets import FigureWidget
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
    """Moving Average (MA).

    A moving average is a widely used indicator in technical analysis that helps smooth out
    price action by filtering out the “noise” from random short-term price fluctuations. 

    See [Moving Average (MA)](https://www.investopedia.com/terms/m/movingaverage.asp).

    Use `MA.run` or `MA.run_combs` to run the indicator."""

    def plot(self,
             column=None,
             plot_close=True,
             close_trace_kwargs=None,
             ma_trace_kwargs=None,
             add_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `MA.ma` against `MA.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `MA.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.close`.
            ma_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MA.ma`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> vbt.MA.run(ohlcv['Close'], 10).plot()
        ```

        ![](/vectorbt/docs/img/MA.png)
        """
        from vectorbt.settings import color_schema

        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if ma_trace_kwargs is None:
            ma_trace_kwargs = {}
        close_trace_kwargs = merge_dicts(dict(
            name='Close',
            line_color=color_schema['blue']
        ), close_trace_kwargs)
        ma_trace_kwargs = merge_dicts(dict(
            name='MA'
        ), ma_trace_kwargs)

        if plot_close:
            fig = self_col.close.vbt.plot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.ma.vbt.plot(
            trace_kwargs=ma_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

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
    """Moving Standard Deviation (MSTD).

    Standard deviation is an indicator that measures the size of an assets recent price moves
    in order to predict how volatile the price may be in the future.

    Use `MSTD.run` or `MSTD.run_combs` to run the indicator."""

    def plot(self,
             column=None,
             mstd_trace_kwargs=None,
             add_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `MSTD.mstd`.

        Args:
            column (str): Name of the column to plot.
            mstd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MSTD.mstd`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> vbt.MSTD.run(ohlcv['Close'], 10).plot()
        ```

        ![](/vectorbt/docs/img/MSTD.png)
        """
        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
        fig.update_layout(**layout_kwargs)

        if mstd_trace_kwargs is None:
            mstd_trace_kwargs = {}
        mstd_trace_kwargs = merge_dicts(dict(
            name='MSTD'
        ), mstd_trace_kwargs)

        fig = self_col.mstd.vbt.plot(
            trace_kwargs=mstd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

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
    """Bollinger Bands (BBANDS).

    A Bollinger Band® is a technical analysis tool defined by a set of lines plotted two standard
    deviations (positively and negatively) away from a simple moving average (SMA) of the security's
    price, but can be adjusted to user preferences.

    See [Bollinger Band®](https://www.investopedia.com/terms/b/bollingerbands.asp).

    Use `BBANDS.run` or `BBANDS.run_combs` to run the indicator."""

    def plot(self,
             column=None,
             plot_close=True,
             close_trace_kwargs=None,
             middle_trace_kwargs=None,
             upper_trace_kwargs=None,
             lower_trace_kwargs=None,
             add_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `BBANDS.middle`, `BBANDS.upper` and `BBANDS.lower` against
        `BBANDS.close`.

        Args:
            column (str): Name of the column to plot.
            plot_close (bool): Whether to plot `MA.close`.
            close_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.close`.
            middle_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.middle`.
            upper_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.upper`.
            lower_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `BBANDS.lower`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> vbt.BBANDS.run(ohlcv['Close']).plot()
        ```

        ![](/vectorbt/docs/img/BBANDS.png)
        """
        from vectorbt.settings import color_schema

        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
        fig.update_layout(**layout_kwargs)

        if close_trace_kwargs is None:
            close_trace_kwargs = {}
        if middle_trace_kwargs is None:
            middle_trace_kwargs = {}
        if upper_trace_kwargs is None:
            upper_trace_kwargs = {}
        if lower_trace_kwargs is None:
            lower_trace_kwargs = {}
        lower_trace_kwargs = merge_dicts(dict(
            name='Lower Band',
            line_color=color_schema['gray'],
        ), lower_trace_kwargs)
        upper_trace_kwargs = merge_dicts(dict(
            name='Upper Band',
            line_color=color_schema['gray'],
            fill='tonexty',
            fillcolor='rgba(128, 128, 128, 0.2)'
        ), upper_trace_kwargs)  # default kwargs
        middle_trace_kwargs = merge_dicts(dict(
            name='Middle Band'
        ), middle_trace_kwargs)
        close_trace_kwargs = merge_dicts(dict(
            name='Close',
            line=dict(color=color_schema['blue'])
        ), close_trace_kwargs)

        fig = self_col.lower.vbt.plot(
            trace_kwargs=lower_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.upper.vbt.plot(
            trace_kwargs=upper_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.middle.vbt.plot(
            trace_kwargs=middle_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        if plot_close:
            fig = self_col.close.vbt.plot(
                trace_kwargs=close_trace_kwargs,
                add_trace_kwargs=add_trace_kwargs, fig=fig)

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
    """Relative Strength Index (RSI).

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    See [Relative Strength Index (RSI)](https://www.investopedia.com/terms/r/rsi.asp).

    Use `RSI.run` or `RSI.run_combs` to run the indicator."""

    def plot(self,
             column=None,
             levels=(30, 70),
             rsi_trace_kwargs=None,
             add_trace_kwargs=None,
             xref='x', yref='y',
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `RSI.rsi`.

        Args:
            column (str): Name of the column to plot.
            levels (tuple): Two extremes: bottom and top.
            rsi_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `RSI.rsi`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> vbt.RSI.run(ohlcv['Close']).plot()
        ```

        ![](/vectorbt/docs/img/RSI.png)
        """
        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
        default_layout = dict()
        default_layout['yaxis' + yref[1:]] = dict(range=[-5, 105])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        if rsi_trace_kwargs is None:
            rsi_trace_kwargs = {}
        rsi_trace_kwargs = merge_dicts(dict(
            name='RSI'
        ), rsi_trace_kwargs)

        fig = self_col.rsi.vbt.plot(
            trace_kwargs=rsi_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        # Fill void between levels
        fig.add_shape(
            type="rect",
            xref=xref,
            yref=yref,
            x0=self_col.rsi.index[0],
            y0=levels[0],
            x1=self_col.rsi.index[-1],
            y1=levels[1],
            fillcolor="purple",
            opacity=0.2,
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
    """Stochastic Oscillator (STOCH).

    A stochastic oscillator is a momentum indicator comparing a particular closing price
    of a security to a range of its prices over a certain period of time. It is used to
    generate overbought and oversold trading signals, utilizing a 0-100 bounded range of values.

    See [Stochastic Oscillator](https://www.investopedia.com/terms/s/stochasticoscillator.asp).

    Use `STOCH.run` or `STOCH.run_combs` to run the indicator."""

    def plot(self,
             column=None,
             levels=(30, 70),
             percent_k_trace_kwargs=None,
             percent_d_trace_kwargs=None,
             shape_kwargs=None,
             add_trace_kwargs=None,
             xref='x', yref='y',
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `STOCH.percent_k` and `STOCH.percent_d`.

        Args:
            column (str): Name of the column to plot.
            levels (tuple): Two extremes: bottom and top.
            percent_k_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `STOCH.percent_k`.
            percent_d_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `STOCH.percent_d`.
            shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for zone between levels.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> vbt.STOCH.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close']).plot()
        ```

        ![](/vectorbt/docs/img/STOCH.png)
        """
        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
        default_layout = dict()
        default_layout['yaxis' + yref[1:]] = dict(range=[-5, 105])
        fig.update_layout(**default_layout)
        fig.update_layout(**layout_kwargs)

        if percent_k_trace_kwargs is None:
            percent_k_trace_kwargs = {}
        if percent_d_trace_kwargs is None:
            percent_d_trace_kwargs = {}
        if shape_kwargs is None:
            shape_kwargs = {}
        percent_k_trace_kwargs = merge_dicts(dict(
            name='%K'
        ), percent_k_trace_kwargs)
        percent_d_trace_kwargs = merge_dicts(dict(
            name='%D'
        ), percent_d_trace_kwargs)

        fig = self_col.percent_k.vbt.plot(
            trace_kwargs=percent_k_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.percent_d.vbt.plot(
            trace_kwargs=percent_d_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        # Plot levels
        # Fill void between levels
        shape_kwargs = merge_dicts(dict(
            type="rect",
            xref=xref,
            yref=yref,
            x0=self_col.percent_k.index[0],
            y0=levels[0],
            x1=self_col.percent_k.index[-1],
            y1=levels[1],
            fillcolor="purple",
            opacity=0.2,
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
    fast_window=12,
    slow_window=26,
    signal_window=9,
    macd_ewm=False,
    signal_ewm=False
)


class MACD(MACD):
    """Moving Average Convergence Divergence (MACD).

    Is a trend-following momentum indicator that shows the relationship between
    two moving averages of prices.

    See [Moving Average Convergence Divergence – MACD](https://www.investopedia.com/terms/m/macd.asp).

    Use `MACD.run` or `MACD.run_combs` to run the indicator."""

    def plot(self,
             column=None,
             macd_trace_kwargs=None,
             signal_trace_kwargs=None,
             hist_trace_kwargs=None,
             add_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `MACD.macd`, `MACD.signal` and `MACD.hist`.

        Args:
            column (str): Name of the column to plot.
            macd_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MACD.macd`.
            signal_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `MACD.signal`.
            hist_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar` for `MACD.hist`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> vbt.MACD.run(ohlcv['Close']).plot()
        ```

        ![](/vectorbt/docs/img/MACD.png)"""
        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
            fig.update_layout(bargap=0)
        fig.update_layout(**layout_kwargs)

        if macd_trace_kwargs is None:
            macd_trace_kwargs = {}
        if signal_trace_kwargs is None:
            signal_trace_kwargs = {}
        if hist_trace_kwargs is None:
            hist_trace_kwargs = {}
        macd_trace_kwargs = merge_dicts(dict(
            name='MACD'
        ), macd_trace_kwargs)
        signal_trace_kwargs = merge_dicts(dict(
            name='Signal'
        ), signal_trace_kwargs)
        hist_trace_kwargs = merge_dicts(dict(name='Histogram'), hist_trace_kwargs)

        fig = self_col.macd.vbt.plot(
            trace_kwargs=macd_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.signal.vbt.plot(
            trace_kwargs=signal_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        # Plot hist
        hist = self_col.hist.values
        hist_diff = generic_nb.diff_1d_nb(hist)
        marker_colors = np.full(hist.shape, 'silver', dtype=np.object)
        marker_colors[(hist > 0) & (hist_diff > 0)] = 'green'
        marker_colors[(hist > 0) & (hist_diff <= 0)] = 'lightgreen'
        marker_colors[(hist < 0) & (hist_diff < 0)] = 'red'
        marker_colors[(hist < 0) & (hist_diff >= 0)] = 'lightcoral'

        hist_bar = go.Bar(
            x=self_col.hist.index,
            y=self_col.hist.values,
            marker_color=marker_colors,
            marker_line_width=0
        )
        hist_bar.update(**hist_trace_kwargs)
        if add_trace_kwargs is None:
            add_trace_kwargs = {}
        fig.add_trace(hist_bar, **add_trace_kwargs)

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
    window=14,
    ewm=True
)


class ATR(ATR):
    """Average True Range (ATR).

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    See [Average True Range - ATR](https://www.investopedia.com/terms/a/atr.asp).

    Use `ATR.run` or `ATR.run_combs` to run the indicator.

    !!! note
        Uses Simple MA and Exponential MA as compared to Wilder.
    """

    def plot(self,
             column=None,
             tr_trace_kwargs=None,
             atr_trace_kwargs=None,
             add_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `ATR.tr` and `ATR.atr`.

        Args:
            column (str): Name of the column to plot.
            tr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.tr`.
            atr_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `ATR.atr`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> vbt.ATR.run(ohlcv['High'], ohlcv['Low'], ohlcv['Close'], 10).plot()
        ```

        ![](/vectorbt/docs/img/ATR.png)
        """
        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
        fig.update_layout(**layout_kwargs)

        if tr_trace_kwargs is None:
            tr_trace_kwargs = {}
        if atr_trace_kwargs is None:
            atr_trace_kwargs = {}
        tr_trace_kwargs = merge_dicts(dict(
            name='TR'
        ), tr_trace_kwargs)
        atr_trace_kwargs = merge_dicts(dict(
            name='ATR'
        ), atr_trace_kwargs)

        fig = self_col.tr.vbt.plot(
            trace_kwargs=tr_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)
        fig = self_col.atr.vbt.plot(
            trace_kwargs=atr_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

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
    """On-balance volume (OBV).

    It relates price and volume in the stock market. OBV is based on a cumulative total volume.

    See [On-Balance Volume (OBV)](https://www.investopedia.com/terms/o/onbalancevolume.asp).

    Use `OBV.run` to run the indicator."""

    def plot(self,
             column=None,
             obv_trace_kwargs=None,
             add_trace_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot `OBV.obv`.

        Args:
            column (str): Name of the column to plot.
            obv_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `OBV.obv`.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```py
        >>> vbt.OBV.run(ohlcv['Close'], ohlcv['Volume']).plot()
        ```

        ![](/vectorbt/docs/img/OBV.png)
        """
        self_col = self.select_series(column=column)

        if fig is None:
            fig = FigureWidget()
        fig.update_layout(**layout_kwargs)

        if obv_trace_kwargs is None:
            obv_trace_kwargs = {}
        obv_trace_kwargs = merge_dicts(dict(
            name='OBV'
        ), obv_trace_kwargs)

        fig = self_col.obv.vbt.plot(
            trace_kwargs=obv_trace_kwargs,
            add_trace_kwargs=add_trace_kwargs, fig=fig)

        return fig


fix_class_for_docs(OBV)
