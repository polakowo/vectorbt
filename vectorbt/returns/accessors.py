"""Custom pandas accessors.

!!! note
    The underlying Series/DataFrame must already be a return series.

    Accessors do not utilize caching.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from vectorbt.root_accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.widgets import CustomFigureWidget
from vectorbt.base import reshape_fns
from vectorbt.generic.accessors import (
    Generic_Accessor,
    Generic_SRAccessor,
    Generic_DFAccessor
)
from vectorbt.utils.datetime import freq_delta, DatetimeTypes
from vectorbt.returns import nb, metrics


class Returns_Accessor(Generic_Accessor):
    """Accessor on top of return series. For both, Series and DataFrames.

    Accessible through `pd.Series.vbt.returns` and `pd.DataFrame.vbt.returns`."""

    def __init__(self, obj, year_freq=None, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Generic_Accessor.__init__(self, obj, **kwargs)

        # Set year frequency
        self._year_freq = year_freq

    @classmethod
    def from_price(cls, price, **kwargs):
        """Returns a new `Returns_Accessor` instance with returns from `price`."""
        return cls(price.vbt.pct_change(), **kwargs)

    @property
    def year_freq(self):
        """Year frequency."""
        from vectorbt import settings

        year_freq = self._year_freq
        if year_freq is None:
            year_freq = settings.returns['year_freq']
        return freq_delta(year_freq)

    @property
    def ann_factor(self):
        """Annualization factor."""
        if self.wrapper.freq is None:
            raise ValueError("Couldn't parse the frequency of index. You must set `freq`.")
        return self.year_freq / self.wrapper.freq

    def daily(self):
        """Daily returns."""
        checks.assert_type(self.wrapper.index, DatetimeTypes)

        if self.wrapper.freq == pd.Timedelta('1D'):
            return self._obj
        return self.resample_apply('1D', nb.total_return_apply_nb)

    def annual(self):
        """Annual returns."""
        checks.assert_type(self._obj.index, DatetimeTypes)

        if self.wrapper.freq == self.year_freq:
            return self._obj
        return self.resample_apply(self.year_freq, nb.total_return_apply_nb)

    def cumulative(self, start_value=0.):
        """Cumulative returns.

        Args:
            start_value (float or array_like): The starting returns.
                Will broadcast per column."""
        start_value = np.broadcast_to(start_value, (len(self.wrapper.columns),))
        return self.wrapper.wrap(nb.cum_returns_nb(self.to_2d_array(), start_value))

    def total(self):
        """Total return."""
        return self.wrapper.wrap_reduced(nb.cum_returns_final_nb(
            self.to_2d_array(), np.full(len(self.wrapper.columns), 0.)))

    def annualized(self):
        """Mean annual growth rate of returns.

        This is equivalent to the compound annual growth rate."""
        return self.wrapper.wrap_reduced(nb.annualized_return_nb(self.to_2d_array(), self.ann_factor))

    def annualized_volatility(self, levy_alpha=2.0):
        """Annualized volatility of a strategy.

        Args:
            levy_alpha (float or array_like): Scaling relation (Levy stability exponent).
                Will broadcast per column."""
        levy_alpha = np.broadcast_to(levy_alpha, (len(self.wrapper.columns),))
        return self.wrapper.wrap_reduced(nb.annualized_volatility_nb(self.to_2d_array(), self.ann_factor, levy_alpha))

    def calmar_ratio(self):
        """Calmar ratio, or drawdown ratio, of a strategy."""
        return self.wrapper.wrap_reduced(nb.calmar_ratio_nb(self.to_2d_array(), self.ann_factor))

    def omega_ratio(self, risk_free=0., required_return=0.):
        """Omega ratio of a strategy.

        Args:
            risk_free (float or array_like): Constant risk-free return throughout the period.
                Will broadcast per column.
            required_return (float or array_like): Minimum acceptance return of the investor.
                Will broadcast per column."""
        risk_free = np.broadcast_to(risk_free, (len(self.wrapper.columns),))
        required_return = np.broadcast_to(required_return, (len(self.wrapper.columns),))
        return self.wrapper.wrap_reduced(nb.omega_ratio_nb(
            self.to_2d_array(), self.ann_factor, risk_free, required_return))

    def sharpe_ratio(self, risk_free=0.):
        """Sharpe ratio of a strategy.

        Args:
            risk_free (float or array_like): Constant risk-free return throughout the period.
                Will broadcast per column."""
        risk_free = np.broadcast_to(risk_free, (len(self.wrapper.columns),))
        return self.wrapper.wrap_reduced(nb.sharpe_ratio_nb(self.to_2d_array(), self.ann_factor, risk_free))

    def deflated_sharpe_ratio(self, risk_free=0., var_sharpe=None, nb_trials=None, ddof=0, bias=True):
        """Deflated Sharpe Ratio (DSR).

        Expresses the chance that the advertized strategy has a positive Sharpe ratio.

        If `var_sharpe` is None, is calculated based on all columns.
        If `nb_trials` is None, is set to the number of columns."""
        sharpe_ratio = reshape_fns.to_1d(self.sharpe_ratio(risk_free=risk_free), raw=True)
        if var_sharpe is None:
            var_sharpe = np.var(sharpe_ratio, ddof=ddof)
        if nb_trials is None:
            nb_trials = self.wrapper.shape_2d[1]
        returns = reshape_fns.to_2d(self._obj, raw=True)
        nanmask = np.isnan(returns)
        if nanmask.any():
            returns = returns.copy()
            returns[nanmask] = 0.
        return self.wrapper.wrap_reduced(metrics.deflated_sharpe_ratio(
            est_sharpe=sharpe_ratio / np.sqrt(self.ann_factor),
            var_sharpe=var_sharpe / self.ann_factor,
            nb_trials=nb_trials,
            backtest_horizon=self.wrapper.shape_2d[0],
            skew=skew(returns, axis=0, bias=bias),
            kurtosis=kurtosis(returns, axis=0, bias=bias)
        ))

    def downside_risk(self, required_return=0.):
        """Downside deviation below a threshold.

        Args:
            required_return (float or array_like): Minimum acceptance return of the investor.
                Will broadcast per column."""
        required_return = np.broadcast_to(required_return, (len(self.wrapper.columns),))
        return self.wrapper.wrap_reduced(nb.downside_risk_nb(self.to_2d_array(), self.ann_factor, required_return))

    def sortino_ratio(self, required_return=0.):
        """Sortino ratio of a strategy.

        Args:
            required_return (float or array_like): Minimum acceptance return of the investor.
                Will broadcast per column."""
        required_return = np.broadcast_to(required_return, (len(self.wrapper.columns),))
        return self.wrapper.wrap_reduced(nb.sortino_ratio_nb(self.to_2d_array(), self.ann_factor, required_return))

    def information_ratio(self, benchmark_rets):
        """Information ratio of a strategy.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element."""
        benchmark_rets = reshape_fns.broadcast_to(
            reshape_fns.to_2d(benchmark_rets, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))

        return self.wrapper.wrap_reduced(nb.information_ratio_nb(self.to_2d_array(), benchmark_rets))

    def beta(self, benchmark_rets):
        """Beta.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element."""
        benchmark_rets = reshape_fns.broadcast_to(
            reshape_fns.to_2d(benchmark_rets, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        return self.wrapper.wrap_reduced(nb.beta_nb(self.to_2d_array(), benchmark_rets))

    def alpha(self, benchmark_rets, risk_free=0.):
        """Annualized alpha.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element.
            risk_free (float or array_like): Constant risk-free return throughout the period.
                Will broadcast per column."""
        benchmark_rets = reshape_fns.broadcast_to(
            reshape_fns.to_2d(benchmark_rets, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        risk_free = np.broadcast_to(risk_free, (len(self.wrapper.columns),))
        return self.wrapper.wrap_reduced(nb.alpha_nb(self.to_2d_array(), benchmark_rets, self.ann_factor, risk_free))

    def tail_ratio(self):
        """Ratio between the right (95%) and left tail (5%)."""
        return self.wrapper.wrap_reduced(nb.tail_ratio_nb(self.to_2d_array()))

    def common_sense_ratio(self):
        """Common Sense Ratio."""
        return self.tail_ratio() * (1 + self.annualized())

    def value_at_risk(self, cutoff=0.05):
        """Value at risk (VaR) of a returns stream.

        Args:
            cutoff (float or array_like): Decimal representing the percentage cutoff for the
                bottom percentile of returns. Will broadcast per column."""
        cutoff = np.broadcast_to(cutoff, (len(self.wrapper.columns),))
        return self.wrapper.wrap_reduced(nb.value_at_risk_nb(self.to_2d_array(), cutoff))

    def conditional_value_at_risk(self, cutoff=0.05):
        """Conditional value at risk (CVaR) of a returns stream.

        Args:
            cutoff (float or array_like): Decimal representing the percentage cutoff for the
                bottom percentile of returns. Will broadcast per column."""
        cutoff = np.broadcast_to(cutoff, (len(self.wrapper.columns),))
        return self.wrapper.wrap_reduced(nb.conditional_value_at_risk_nb(self.to_2d_array(), cutoff))

    def capture(self, benchmark_rets):
        """Capture ratio.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element."""
        benchmark_rets = reshape_fns.broadcast_to(
            reshape_fns.to_2d(benchmark_rets, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        return self.wrapper.wrap_reduced(nb.capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor))

    def up_capture(self, benchmark_rets):
        """Capture ratio for periods when the benchmark return is positive.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element."""
        benchmark_rets = reshape_fns.broadcast_to(
            reshape_fns.to_2d(benchmark_rets, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        return self.wrapper.wrap_reduced(nb.up_capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor))

    def down_capture(self, benchmark_rets):
        """Capture ratio for periods when the benchmark return is negative.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element."""
        benchmark_rets = reshape_fns.broadcast_to(
            reshape_fns.to_2d(benchmark_rets, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        return self.wrapper.wrap_reduced(nb.down_capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor))

    def drawdown(self):
        """Relative decline from a peak."""
        return self.wrapper.wrap(nb.drawdown_nb(self.to_2d_array()))

    def max_drawdown(self):
        """Total maximum drawdown (MDD)."""
        return self.wrapper.wrap_reduced(nb.max_drawdown_nb(self.to_2d_array()))

    def drawdowns(self, group_by=None, **kwargs):
        """Generate drawdown records of cumulative returns.

        See `vectorbt.generic.drawdowns.Drawdowns`."""
        if group_by is None:
            group_by = self.wrapper.grouper.group_by
        return self.cumulative(start_value=1.).vbt(freq=self.wrapper.freq, group_by=group_by).drawdowns(**kwargs)

    def stats(self, benchmark_rets, levy_alpha=2.0, risk_free=0., required_return=0.):
        """Compute various statistics on these returns.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element.
            levy_alpha (float or array_like): Scaling relation (Levy stability exponent).
                Will broadcast per column.
            risk_free (float or array_like): Constant risk-free return throughout the period.
                Will broadcast per column.
            required_return (float or array_like): Minimum acceptance return of the investor.
                Will broadcast per column.

        ## Example

        ```python-repl
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> import yfinance as yf
        >>> import vectorbt as vbt

        >>> btc_price = yf.Ticker("BTC-USD").history()['Close']
        >>> spy_price = yf.Ticker("SPY").history()['Close']
        >>> price_df = pd.concat([btc_price, spy_price], axis=1, keys=("BTC-USD", "SPY"))
        >>> returns_df = price_df.pct_change()
        >>> returns_df["BTC-USD"].vbt.returns.stats(returns_df["SPY"])
        Start                    2020-11-01 00:00:00
        End                      2020-12-01 00:00:00
        Duration                    31 days 00:00:00
        Total Return [%]                     37.9835
        Benchmark Return [%]                 10.7935
        Annual Return [%]                    4329.46
        Annual Volatility [%]                71.5084
        Sharpe Ratio                         5.84964
        Calmar Ratio                         413.819
        Max. Drawdown [%]                   -10.4622
        Omega Ratio                          2.36607
        Sortino Ratio                        11.0962
        Skew                                0.036609
        Kurtosis                             1.04302
        Tail Ratio                           1.66878
        Common Sense Ratio                   73.9178
        Value at Risk                     -0.0412519
        Alpha                                43.0408
        Beta                                0.531022
        Name: BTC-USD, dtype: object
        ```
        """
        # Run stats
        benchmark_rets = reshape_fns.broadcast_to(benchmark_rets, self._obj)
        stats_df = pd.DataFrame({
            'Start': self.wrapper.index[0],
            'End': self.wrapper.index[-1],
            'Duration': self.wrapper.shape[0] * self.wrapper.freq,
            'Total Return [%]': self.total() * 100,
            'Benchmark Return [%]': benchmark_rets.vbt.returns.total() * 100,
            'Annual Return [%]': self.annualized() * 100,
            'Annual Volatility [%]': self.annualized_volatility(levy_alpha=levy_alpha) * 100,
            'Sharpe Ratio': self.sharpe_ratio(risk_free=risk_free),
            'Calmar Ratio': self.calmar_ratio(),
            'Max. Drawdown [%]': self.max_drawdown() * 100,
            'Omega Ratio': self.omega_ratio(required_return=required_return),
            'Sortino Ratio': self.sortino_ratio(required_return=required_return),
            'Skew': self._obj.skew(axis=0),
            'Kurtosis': self._obj.kurtosis(axis=0),
            'Tail Ratio': self.tail_ratio(),
            'Common Sense Ratio': self.common_sense_ratio(),
            'Value at Risk': self.value_at_risk(),
            'Alpha': self.alpha(benchmark_rets, risk_free=risk_free),
            'Beta': self.beta(benchmark_rets)
        }, index=self.wrapper.columns)

        # Select columns or reduce
        if self.is_series():
            return self.wrapper.wrap_reduced(stats_df.iloc[0], index=stats_df.columns)
        return stats_df


@register_series_accessor('returns')
class Returns_SRAccessor(Returns_Accessor, Generic_SRAccessor):
    """Accessor on top of return series. For Series only.

    Accessible through `pd.Series.vbt.returns`."""

    def __init__(self, obj, year_freq=None, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Generic_SRAccessor.__init__(self, obj, **kwargs)
        Returns_Accessor.__init__(self, obj, year_freq=year_freq, **kwargs)

    def plot_cum_returns(self, benchmark_rets=None, start_value=1, fill_to_benchmark=False,
                         main_kwargs=None, benchmark_kwargs=None, hline_shape_kwargs=None,
                         row=None, col=None, xref='x', yref='y',
                         fig=None, **layout_kwargs):  # pragma: no cover
        """Plot cumulative returns.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element.
            start_value (float): The starting returns.
            fill_to_benchmark (bool): Whether to fill between main and benchmark, or between main and `start_value`.
            main_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot` for main.
            benchmark_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.Generic_SRAccessor.plot` for benchmark.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for `start_value` line.
            row (int): Row position.
            col (int): Column position.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> import pandas as pd
        >>> import numpy as np

        >>> np.random.seed(0)
        >>> rets = pd.Series(np.random.uniform(-0.05, 0.05, size=100))
        >>> benchmark_rets = pd.Series(np.random.uniform(-0.05, 0.05, size=100))
        >>> rets.vbt.returns.plot_cum_returns(benchmark_rets=benchmark_rets)
        ```

        ![](/vectorbt/docs/img/plot_cum_returns.png)
        """
        from vectorbt.settings import color_schema

        if fig is None:
            fig = CustomFigureWidget()
        fig.update_layout(**layout_kwargs)
        x_domain = [0, 1]
        xaxis = 'xaxis' + xref[1:]
        if xaxis in fig.layout:
            if 'domain' in fig.layout[xaxis]:
                if fig.layout[xaxis]['domain'] is not None:
                    x_domain = fig.layout[xaxis]['domain']
        fill_to_benchmark = fill_to_benchmark and benchmark_rets is not None

        if benchmark_rets is not None:
            # Plot benchmark
            benchmark_rets = reshape_fns.broadcast_to(benchmark_rets, self._obj)
            if benchmark_kwargs is None:
                benchmark_kwargs = {}
            benchmark_kwargs = merge_dicts(dict(
                trace_kwargs=dict(
                    line_color=color_schema['gray'],
                    name='Benchmark'
                )
            ), benchmark_kwargs)
            benchmark_cumrets = benchmark_rets.vbt.returns.cumulative(start_value=start_value)
            benchmark_cumrets.vbt.plot(**benchmark_kwargs, row=row, col=col, fig=fig)
        else:
            benchmark_cumrets = None

        # Plot main
        if main_kwargs is None:
            main_kwargs = {}
        main_kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line_color=color_schema['purple'],
            ),
            other_trace_kwargs='hidden'
        ), main_kwargs)
        cumrets = self.cumulative(start_value=start_value)
        if fill_to_benchmark:
            cumrets.vbt.plot_against(benchmark_cumrets, **main_kwargs, row=row, col=col, fig=fig)
        else:
            cumrets.vbt.plot_against(start_value, **main_kwargs, row=row, col=col, fig=fig)

        # Plot hline
        if hline_shape_kwargs is None:
            hline_shape_kwargs = {}
        fig.add_shape(**merge_dicts(dict(
            type='line',
            xref="paper",
            yref=yref,
            x0=x_domain[0],
            y0=start_value,
            x1=x_domain[1],
            y1=start_value,
            line=dict(
                color="gray",
                dash="dash",
            )
        ), hline_shape_kwargs))

        return fig


@register_dataframe_accessor('returns')
class Returns_DFAccessor(Returns_Accessor, Generic_DFAccessor):
    """Accessor on top of return series. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.returns`."""

    def __init__(self, obj, year_freq=None, **kwargs):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Generic_DFAccessor.__init__(self, obj, **kwargs)
        Returns_Accessor.__init__(self, obj, year_freq=year_freq, **kwargs)
