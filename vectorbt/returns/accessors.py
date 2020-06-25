"""Custom pandas accessors.

!!! note
    The underlying Series/DataFrame must already be a return series."""

import numpy as np
import pandas as pd
from scipy import stats

from vectorbt import defaults
from vectorbt.accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks, reshape_fns
from vectorbt.tseries.accessors import (
    TimeSeries_Accessor,
    TimeSeries_SRAccessor,
    TimeSeries_DFAccessor
)
from vectorbt.tseries.nb import pct_change_nb
from vectorbt.tseries.common import freq_delta, DatetimeTypes
from vectorbt.returns import nb


class Returns_Accessor(TimeSeries_Accessor):
    """Accessor on top of return series. For both, Series and DataFrames.

    Accessible through `pandas.Series.vbt.returns` and `pandas.DataFrame.vbt.returns`."""

    def __init__(self, obj, freq=None, year_freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        # Set year frequency
        self._year_freq = year_freq

        TimeSeries_Accessor.__init__(self, obj, freq=freq)

    @classmethod
    def from_price(cls, price, **kwargs):
        """Computes the return series of `price`."""
        return cls(price.vbt.tseries.pct_change(), **kwargs)

    @property
    def year_freq(self):
        """Year frequency."""
        year_freq = self._year_freq
        if year_freq is None:
            year_freq = defaults.returns['year_freq']
        return freq_delta(year_freq)

    @property
    def ann_factor(self):
        """Annualization factor."""
        return self.year_freq / self.freq

    def daily(self):
        """Daily returns."""
        checks.assert_type(self.index, DatetimeTypes)

        if self.freq == pd.Timedelta('1 days'):
            return self._obj
        return self.resample_apply('1 days', nb.total_return_apply_nb)

    def annual(self):
        """Annual returns."""
        checks.assert_type(self._obj.index, DatetimeTypes)

        if self.freq == self.year_freq:
            return self._obj
        return self.resample_apply(self.year_freq, nb.total_return_apply_nb)

    def cumulative(self, start_value=0.):
        """Cumulative returns.

        Args:
            start_value (int, float or array_like): The starting returns."""
        start_value = np.broadcast_to(start_value, (len(self.columns),))
        return self.wrap(nb.cum_returns_nb(self.to_2d_array(), start_value))

    def total(self):
        """Total return."""
        return self.wrap_reduced(nb.cum_returns_final_nb(self.to_2d_array(), np.full(len(self.columns), 0.)))

    def drawdown(self):
        """Relative decline from a peak."""
        return self.wrap(nb.drawdown_nb(self.to_2d_array()))

    def max_drawdown(self):
        """Total maximum drawdown (MDD)."""
        return self.wrap_reduced(nb.max_drawdown_nb(self.to_2d_array()))

    def annualized_return(self):
        """Mean annual growth rate of returns.

        This is equivalent to the compound annual growth rate."""
        return self.wrap_reduced(nb.annualized_return_nb(self.to_2d_array(), self.ann_factor))

    def annualized_volatility(self, levy_alpha=2.0):
        """Annualized volatility of a strategy.

        Args:
            levy_alpha (float or array_like): Scaling relation (Levy stability exponent)."""
        levy_alpha = np.broadcast_to(levy_alpha, (len(self.columns),))
        return self.wrap_reduced(nb.annualized_volatility_nb(self.to_2d_array(), self.ann_factor, levy_alpha))

    def calmar_ratio(self):
        """Calmar ratio, or drawdown ratio, of a strategy."""
        return self.wrap_reduced(nb.calmar_ratio_nb(self.to_2d_array(), self.ann_factor))

    def omega_ratio(self, risk_free=0., required_return=0.):
        """Omega ratio of a strategy.

        Args:
            risk_free (float or array_like): Constant risk-free return throughout the period.
            required_return (float or array_like): Minimum acceptance return of the investor."""
        risk_free = np.broadcast_to(risk_free, (len(self.columns),))
        required_return = np.broadcast_to(required_return, (len(self.columns),))
        return self.wrap_reduced(nb.omega_ratio_nb(
            self.to_2d_array(), self.ann_factor, risk_free, required_return))

    def sharpe_ratio(self, risk_free=0.):
        """Sharpe ratio of a strategy.

        Args:
            risk_free (float or array_like): Constant risk-free return throughout the period."""
        risk_free = np.broadcast_to(risk_free, (len(self.columns),))
        return self.wrap_reduced(nb.sharpe_ratio_nb(self.to_2d_array(), self.ann_factor, risk_free))

    def downside_risk(self, required_return=0.):
        """Downside deviation below a threshold.

        Args:
            required_return (float or array_like): Minimum acceptance return of the investor."""
        required_return = np.broadcast_to(required_return, (len(self.columns),))
        return self.wrap_reduced(nb.downside_risk_nb(self.to_2d_array(), self.ann_factor, required_return))

    def sortino_ratio(self, required_return=0.):
        """Sortino ratio of a strategy.

        Args:
            required_return (float or array_like): Minimum acceptance return of the investor."""
        required_return = np.broadcast_to(required_return, (len(self.columns),))
        return self.wrap_reduced(nb.sortino_ratio_nb(self.to_2d_array(), self.ann_factor, required_return))

    def information_ratio(self, factor_returns):
        """Information ratio of a strategy.

        Args:
            factor_returns (array_like): Benchmark return to compare returns against. Will broadcast."""
        factor_returns = reshape_fns.broadcast_to(
            reshape_fns.to_2d(factor_returns, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))

        return self.wrap_reduced(nb.information_ratio_nb(self.to_2d_array(), factor_returns))

    def beta(self, factor_returns):
        """Beta.

        Args:
            factor_returns (array_like): Benchmark return to compare returns against. Will broadcast."""
        factor_returns = reshape_fns.broadcast_to(
            reshape_fns.to_2d(factor_returns, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        return self.wrap_reduced(nb.beta_nb(self.to_2d_array(), factor_returns))

    def alpha(self, factor_returns, risk_free=0.):
        """Annualized alpha.

        Args:
            factor_returns (array_like): Benchmark return to compare returns against. Will broadcast.
            risk_free (float or array_like): Constant risk-free return throughout the period."""
        factor_returns = reshape_fns.broadcast_to(
            reshape_fns.to_2d(factor_returns, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        risk_free = np.broadcast_to(risk_free, (len(self.columns),))
        return self.wrap_reduced(nb.alpha_nb(self.to_2d_array(), factor_returns, self.ann_factor, risk_free))

    def tail_ratio(self):
        """Ratio between the right (95%) and left tail (5%)."""
        return self.wrap_reduced(nb.tail_ratio_nb(self.to_2d_array()))

    def value_at_risk(self, cutoff=0.05):
        """Value at risk (VaR) of a returns stream.

        Args:
            cutoff (float or array_like): Decimal representing the percentage cutoff for the
                bottom percentile of returns."""
        cutoff = np.broadcast_to(cutoff, (len(self.columns),))
        return self.wrap_reduced(nb.value_at_risk_nb(self.to_2d_array(), cutoff))

    def conditional_value_at_risk(self, cutoff=0.05):
        """Conditional value at risk (CVaR) of a returns stream.

        Args:
            cutoff (float or array_like): Decimal representing the percentage cutoff for the
                bottom percentile of returns."""
        cutoff = np.broadcast_to(cutoff, (len(self.columns),))
        return self.wrap_reduced(nb.conditional_value_at_risk_nb(self.to_2d_array(), cutoff))

    def capture(self, factor_returns):
        """Capture ratio.

        Args:
            factor_returns (array_like): Benchmark return to compare returns against. Will broadcast."""
        factor_returns = reshape_fns.broadcast_to(
            reshape_fns.to_2d(factor_returns, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        return self.wrap_reduced(nb.capture_nb(self.to_2d_array(), factor_returns, self.ann_factor))

    def up_capture(self, factor_returns):
        """Capture ratio for periods when the benchmark return is positive.

        Args:
            factor_returns (array_like): Benchmark return to compare returns against. Will broadcast."""
        factor_returns = reshape_fns.broadcast_to(
            reshape_fns.to_2d(factor_returns, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        return self.wrap_reduced(nb.up_capture_nb(self.to_2d_array(), factor_returns, self.ann_factor))

    def down_capture(self, factor_returns):
        """Capture ratio for periods when the benchmark return is negative.

        Args:
            factor_returns (array_like): Benchmark return to compare returns against. Will broadcast."""
        factor_returns = reshape_fns.broadcast_to(
            reshape_fns.to_2d(factor_returns, raw=True),
            reshape_fns.to_2d(self._obj, raw=True))
        return self.wrap_reduced(nb.down_capture_nb(self.to_2d_array(), factor_returns, self.ann_factor))

    def skew(self):
        """Skewness of returns."""
        return self.wrap_reduced(stats.skew(self.to_2d_array(), axis=0, nan_policy='omit'))

    def kurtosis(self):
        """Kurtosis of returns."""
        return self.wrap_reduced(stats.kurtosis(self.to_2d_array(), axis=0, nan_policy='omit'))

    def drawdown(self, start_value=1.):
        """Drawdown of cumulative returns.

        Args:
            start_value (int, float or array_like): The starting returns.

                Default value is 1 to avoid zeros."""
        return self.cumulative(start_value=start_value).vbt.tseries(freq=self.freq).drawdown()

    def drawdowns(self, start_value=1.):
        """Drawdown records of cumulative returns.

        See `vectorbt.records.drawdowns.Drawdowns`.

        Args:
            start_value (int, float or array_like): The starting returns.

                Default value is 1 to avoid zeros."""
        return self.cumulative(start_value=start_value).vbt.tseries(freq=self.freq).drawdowns()


@register_series_accessor('returns')
class Returns_SRAccessor(Returns_Accessor, TimeSeries_SRAccessor):
    """Accessor on top of return series. For Series only.

    Accessible through `pandas.Series.vbt.returns`."""

    def __init__(self, obj, freq=None, year_freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        TimeSeries_SRAccessor.__init__(self, obj, freq=freq)
        Returns_Accessor.__init__(self, obj, freq=freq, year_freq=year_freq)


@register_dataframe_accessor('returns')
class Returns_DFAccessor(Returns_Accessor, TimeSeries_DFAccessor):
    """Accessor on top of return series. For DataFrames only.

    Accessible through `pandas.DataFrame.vbt.returns`."""

    def __init__(self, obj, freq=None, year_freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        TimeSeries_DFAccessor.__init__(self, obj, freq=freq)
        Returns_Accessor.__init__(self, obj, freq=freq, year_freq=year_freq)
