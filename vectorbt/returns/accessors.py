"""Custom pandas accessors.

Methods can be accessed as follows:

* `ReturnsSRAccessor` -> `pd.Series.vbt.returns.*`
* `ReturnsDFAccessor` -> `pd.DataFrame.vbt.returns.*`

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.returns.accessors.ReturnsAccessor.total
>>> price = pd.Series([1.1, 1.2, 1.3, 1.2, 1.1])
>>> returns = price.pct_change()
>>> returns.vbt.returns.total()
0.0
```

The accessors extend `vectorbt.generic.accessors`.

```python-repl
>>> # inherited from GenericAccessor
>>> returns.vbt.returns.max()
0.09090909090909083
```

!!! note
    The underlying Series/DataFrame must already be a return series.
"""

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis

from vectorbt import _typing as tp
from vectorbt.root_accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.figure import make_figure, get_domain
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast_to
from vectorbt.generic.drawdowns import Drawdowns
from vectorbt.generic.accessors import (
    GenericAccessor,
    GenericSRAccessor,
    GenericDFAccessor
)
from vectorbt.utils.datetime import freq_to_timedelta, DatetimeIndexes
from vectorbt.returns import nb, metrics

ReturnsAccessorT = tp.TypeVar("ReturnsAccessorT", bound="ReturnsAccessor")


class ReturnsAccessor(GenericAccessor):
    """Accessor on top of return series. For both, Series and DataFrames.

    Accessible through `pd.Series.vbt.returns` and `pd.DataFrame.vbt.returns`."""

    def __init__(self, obj: tp.SeriesFrame, year_freq: tp.Optional[tp.FrequencyLike] = None, **kwargs) -> None:
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        GenericAccessor.__init__(self, obj, **kwargs)

        # Set year frequency
        self._year_freq = year_freq

    @classmethod
    def from_price(cls: tp.Type[ReturnsAccessorT], price: tp.SeriesFrame, **kwargs) -> ReturnsAccessorT:
        """Returns a new `ReturnsAccessor` instance with returns from `price`."""
        return cls(price.vbt.pct_change(), **kwargs)

    @property
    def year_freq(self) -> pd.Timedelta:
        """Year frequency."""
        from vectorbt._settings import settings
        returns_cfg = settings['returns']

        year_freq = self._year_freq
        if year_freq is None:
            year_freq = returns_cfg['year_freq']
        return freq_to_timedelta(year_freq)

    @property
    def ann_factor(self) -> float:
        """Annualization factor."""
        if self.wrapper.freq is None:
            raise ValueError("Couldn't parse the frequency of index. You must set `freq`.")
        return self.year_freq / self.wrapper.freq

    def daily(self, **kwargs) -> tp.SeriesFrame:
        """Daily returns."""
        checks.assert_type(self.wrapper.index, DatetimeIndexes)

        if self.wrapper.freq == pd.Timedelta('1D'):
            return self._obj
        return self.resample_apply('1D', nb.total_return_apply_nb, **kwargs)

    def annual(self, **kwargs) -> tp.SeriesFrame:
        """Annual returns."""
        checks.assert_type(self._obj.index, DatetimeIndexes)

        if self.wrapper.freq == self.year_freq:
            return self._obj
        return self.resample_apply(self.year_freq, nb.total_return_apply_nb, **kwargs)

    def cumulative(self, start_value: float = 0., wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Cumulative returns.

        Args:
            start_value (float): The starting returns."""
        cumulative = nb.cum_returns_nb(self.to_2d_array(), start_value)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(cumulative, **wrap_kwargs)

    def total(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Total return."""
        result = nb.cum_returns_final_nb(self.to_2d_array(), 0.)
        wrap_kwargs = merge_dicts(dict(name_or_index='total_return'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_total(self, window: int, minp: tp.Optional[int] = None,
                      wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.total`."""
        result = nb.rolling_cum_returns_final_nb(self.to_2d_array(), window, minp, 0.)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def annualized(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Mean annual growth rate of returns.

        This is equivalent to the compound annual growth rate."""
        result = nb.annualized_return_nb(self.to_2d_array(), self.ann_factor)
        wrap_kwargs = merge_dicts(dict(name_or_index='annualized_return'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_annualized(self, window: int, minp: tp.Optional[int] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.annualized`."""
        result = nb.rolling_annualized_return_nb(self.to_2d_array(), window, minp, self.ann_factor)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def annualized_volatility(self, levy_alpha: float = 2.0, ddof: int = 1,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Annualized volatility of a strategy.

        Args:
            levy_alpha (float): Scaling relation (Levy stability exponent)."""
        result = nb.annualized_volatility_nb(self.to_2d_array(), self.ann_factor, levy_alpha, ddof)
        wrap_kwargs = merge_dicts(dict(name_or_index='annualized_volatility'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_annualized_volatility(self, window: int, minp: tp.Optional[int] = None, levy_alpha: float = 2.0,
                                      ddof: int = 1, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.annualized_volatility`."""
        result = nb.rolling_annualized_volatility_nb(
            self.to_2d_array(), window, minp, self.ann_factor, levy_alpha, ddof)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def calmar_ratio(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Calmar ratio, or drawdown ratio, of a strategy."""
        result = nb.calmar_ratio_nb(self.to_2d_array(), self.ann_factor)
        wrap_kwargs = merge_dicts(dict(name_or_index='calmar_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_calmar_ratio(self, window: int, minp: tp.Optional[int] = None,
                             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.calmar_ratio`."""
        result = nb.rolling_calmar_ratio_nb(self.to_2d_array(), window, minp, self.ann_factor)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def omega_ratio(self, risk_free: float = 0., required_return: float = 0.,
                    wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Omega ratio of a strategy.

        Args:
            risk_free (float): Constant risk-free return throughout the period.
            required_return (float): Minimum acceptance return of the investor."""
        result = nb.omega_ratio_nb(self.to_2d_array(), self.ann_factor, risk_free, required_return)
        wrap_kwargs = merge_dicts(dict(name_or_index='omega_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_omega_ratio(self, window: int, minp: tp.Optional[int] = None, risk_free: float = 0.,
                            required_return: float = 0., wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.omega_ratio`."""
        result = nb.rolling_omega_ratio_nb(
            self.to_2d_array(), window, minp, self.ann_factor, risk_free, required_return)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def sharpe_ratio(self, risk_free: float = 0., ddof: int = 1,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Sharpe ratio of a strategy.

        Args:
            risk_free (float): Constant risk-free return throughout the period."""
        result = nb.sharpe_ratio_nb(self.to_2d_array(), self.ann_factor, risk_free, ddof)
        wrap_kwargs = merge_dicts(dict(name_or_index='sharpe_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_sharpe_ratio(self, window: int, minp: tp.Optional[int] = None, risk_free: float = 0.,
                             ddof: int = 1, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.sharpe_ratio`."""
        result = nb.rolling_sharpe_ratio_nb(self.to_2d_array(), window, minp, self.ann_factor, risk_free, ddof)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def deflated_sharpe_ratio(self, risk_free: float = 0., var_sharpe: tp.Optional[float] = None,
                              nb_trials: tp.Optional[int] = None, ddof: int = 0, bias: bool = True,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Deflated Sharpe Ratio (DSR).

        Expresses the chance that the advertized strategy has a positive Sharpe ratio.

        If `var_sharpe` is None, is calculated based on all columns.
        If `nb_trials` is None, is set to the number of columns."""
        sharpe_ratio = to_1d(self.sharpe_ratio(risk_free=risk_free), raw=True)
        if var_sharpe is None:
            var_sharpe = np.var(sharpe_ratio, ddof=ddof)
        if nb_trials is None:
            nb_trials = self.wrapper.shape_2d[1]
        returns = to_2d(self._obj, raw=True)
        nanmask = np.isnan(returns)
        if nanmask.any():
            returns = returns.copy()
            returns[nanmask] = 0.
        result = metrics.deflated_sharpe_ratio(
            est_sharpe=sharpe_ratio / np.sqrt(self.ann_factor),
            var_sharpe=var_sharpe / self.ann_factor,
            nb_trials=nb_trials,
            backtest_horizon=self.wrapper.shape_2d[0],
            skew=skew(returns, axis=0, bias=bias),
            kurtosis=kurtosis(returns, axis=0, bias=bias)
        )
        wrap_kwargs = merge_dicts(dict(name_or_index='deflated_sharpe_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def downside_risk(self, required_return: float = 0., wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Downside deviation below a threshold.

        Args:
            required_return (float): Minimum acceptance return of the investor."""
        result = nb.downside_risk_nb(self.to_2d_array(), self.ann_factor, required_return)
        wrap_kwargs = merge_dicts(dict(name_or_index='downside_risk'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_downside_risk(self, window: int, minp: tp.Optional[int] = None, required_return: float = 0.,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.downside_risk`."""
        result = nb.rolling_downside_risk_nb(self.to_2d_array(), window, minp, self.ann_factor, required_return)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def sortino_ratio(self, required_return: float = 0., wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Sortino ratio of a strategy.

        Args:
            required_return (float): Minimum acceptance return of the investor.
                Will broadcast per column."""
        result = nb.sortino_ratio_nb(self.to_2d_array(), self.ann_factor, required_return)
        wrap_kwargs = merge_dicts(dict(name_or_index='sortino_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_sortino_ratio(self, window: int, minp: tp.Optional[int] = None, required_return: float = 0.,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.sortino_ratio`."""
        result = nb.rolling_sortino_ratio_nb(self.to_2d_array(), window, minp, self.ann_factor, required_return)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def information_ratio(self, benchmark_rets: tp.ArrayLike, ddof: int = 1,
                          wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Information ratio of a strategy.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against. Will broadcast."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.information_ratio_nb(self.to_2d_array(), benchmark_rets, ddof)
        wrap_kwargs = merge_dicts(dict(name_or_index='information_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_information_ratio(self, window: int, benchmark_rets: tp.ArrayLike, minp: tp.Optional[int] = None,
                                  ddof: int = 1, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.information_ratio`."""
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.rolling_information_ratio_nb(self.to_2d_array(), window, minp, benchmark_rets, ddof)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def beta(self, benchmark_rets: tp.ArrayLike, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Beta.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against. Will broadcast."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.beta_nb(self.to_2d_array(), benchmark_rets)
        wrap_kwargs = merge_dicts(dict(name_or_index='beta'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_beta(self, window: int, benchmark_rets: tp.ArrayLike, minp: tp.Optional[int] = None,
                     wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.beta`."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.rolling_beta_nb(self.to_2d_array(), window, minp, benchmark_rets)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def alpha(self, benchmark_rets: tp.ArrayLike, risk_free: float = 0.,
              wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Annualized alpha.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against. Will broadcast.
            risk_free (float): Constant risk-free return throughout the period."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.alpha_nb(self.to_2d_array(), benchmark_rets, self.ann_factor, risk_free)
        wrap_kwargs = merge_dicts(dict(name_or_index='alpha'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_alpha(self, window: int, benchmark_rets: tp.ArrayLike, minp: tp.Optional[int] = None,
                      risk_free: float = 0., wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.alpha`."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.rolling_alpha_nb(self.to_2d_array(), window, minp, benchmark_rets, self.ann_factor, risk_free)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def tail_ratio(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Ratio between the right (95%) and left tail (5%)."""
        result = nb.tail_ratio_nb(self.to_2d_array())
        wrap_kwargs = merge_dicts(dict(name_or_index='tail_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_tail_ratio(self, window: int, minp: tp.Optional[int] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.tail_ratio`."""
        result = nb.rolling_tail_ratio_nb(self.to_2d_array(), window, minp)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def common_sense_ratio(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Common Sense Ratio."""
        result = to_1d(self.tail_ratio(), raw=True) * (1 + to_1d(self.annualized(), raw=True))
        wrap_kwargs = merge_dicts(dict(name_or_index='common_sense_ratio'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_common_sense_ratio(self, window: int, minp: tp.Optional[int] = None,
                                   wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.common_sense_ratio`."""
        rolling_tail_ratio = to_2d(self.rolling_tail_ratio(window, minp=minp), raw=True)
        rolling_annualized = to_2d(self.rolling_annualized(window, minp=minp), raw=True)
        result = rolling_tail_ratio * (1 + rolling_annualized)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def value_at_risk(self, cutoff: float = 0.05, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Value at risk (VaR) of a returns stream.

        Args:
            cutoff (float): Decimal representing the percentage cutoff for the
                bottom percentile of returns."""
        result = nb.value_at_risk_nb(self.to_2d_array(), cutoff)
        wrap_kwargs = merge_dicts(dict(name_or_index='value_at_risk'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_value_at_risk(self, window: int, minp: tp.Optional[int] = None, cutoff: float = 0.05,
                              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.value_at_risk`."""
        result = nb.rolling_value_at_risk_nb(self.to_2d_array(), window, minp, cutoff)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def cond_value_at_risk(self, cutoff: float = 0.05, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Conditional value at risk (CVaR) of a returns stream.

        Args:
            cutoff (float): Decimal representing the percentage cutoff for the
                bottom percentile of returns."""
        result = nb.cond_value_at_risk_nb(self.to_2d_array(), cutoff)
        wrap_kwargs = merge_dicts(dict(name_or_index='cond_value_at_risk'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_cond_value_at_risk(self, window: int, minp: tp.Optional[int] = None, cutoff: float = 0.05,
                                   wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.cond_value_at_risk`."""
        result = nb.rolling_cond_value_at_risk_nb(self.to_2d_array(), window, minp, cutoff)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def capture(self, benchmark_rets: tp.ArrayLike, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Capture ratio.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against. Will broadcast."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor)
        wrap_kwargs = merge_dicts(dict(name_or_index='capture'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_capture(self, window: int, benchmark_rets: tp.ArrayLike, minp: tp.Optional[int] = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.capture`."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.rolling_capture_nb(self.to_2d_array(), window, minp, benchmark_rets, self.ann_factor)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def up_capture(self, benchmark_rets: tp.ArrayLike, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Capture ratio for periods when the benchmark return is positive.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against. Will broadcast."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.up_capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor)
        wrap_kwargs = merge_dicts(dict(name_or_index='up_capture'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_up_capture(self, window: int, benchmark_rets: tp.ArrayLike, minp: tp.Optional[int] = None,
                           wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.up_capture`."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.rolling_up_capture_nb(self.to_2d_array(), window, minp, benchmark_rets, self.ann_factor)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def down_capture(self, benchmark_rets: tp.ArrayLike, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Capture ratio for periods when the benchmark return is negative.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against. Will broadcast."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.down_capture_nb(self.to_2d_array(), benchmark_rets, self.ann_factor)
        wrap_kwargs = merge_dicts(dict(name_or_index='down_capture'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_down_capture(self, window: int, benchmark_rets: tp.ArrayLike, minp: tp.Optional[int] = None,
                             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.down_capture`."""
        benchmark_rets = broadcast_to(to_2d(benchmark_rets, raw=True), to_2d(self._obj, raw=True))
        result = nb.rolling_down_capture_nb(self.to_2d_array(), window, minp, benchmark_rets, self.ann_factor)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def drawdown(self, wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Relative decline from a peak."""
        result = nb.drawdown_nb(self.to_2d_array())
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    def max_drawdown(self, wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Total maximum drawdown (MDD)."""
        result = nb.max_drawdown_nb(self.to_2d_array())
        wrap_kwargs = merge_dicts(dict(name_or_index='max_drawdown'), wrap_kwargs)
        return self.wrapper.wrap_reduced(result, **wrap_kwargs)

    def rolling_max_drawdown(self, window: int, minp: tp.Optional[int] = None,
                             wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Rolling version of `ReturnsAccessor.max_drawdown`."""
        result = nb.rolling_max_drawdown_nb(self.to_2d_array(), window, minp)
        wrap_kwargs = merge_dicts({}, wrap_kwargs)
        return self.wrapper.wrap(result, **wrap_kwargs)

    @cached_property
    def drawdowns(self) -> Drawdowns:
        """`ReturnsAccessor.get_drawdowns` with default arguments."""
        return self.get_drawdowns()

    @cached_method
    def get_drawdowns(self, group_by: tp.GroupByLike = None, **kwargs) -> Drawdowns:
        """Generate drawdown records of cumulative returns.

        See `vectorbt.generic.drawdowns.Drawdowns`."""
        if group_by is None:
            group_by = self.wrapper.grouper.group_by
        return self.cumulative(start_value=1.).vbt(freq=self.wrapper.freq, group_by=group_by).get_drawdowns(**kwargs)

    def stats(self,
              benchmark_rets: tp.ArrayLike,
              levy_alpha: float = 2.0,
              risk_free: float = 0.,
              required_return: float = 0.,
              wrap_kwargs: tp.KwargsLike = None) -> tp.SeriesFrame:
        """Compute various statistics on these returns.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element.
            levy_alpha (float): Scaling relation (Levy stability exponent).
                Will broadcast per column.
            risk_free (float): Constant risk-free return throughout the period.
                Will broadcast per column.
            required_return (float): Minimum acceptance return of the investor.
                Will broadcast per column.

        ## Example

        ```python-repl
        >>> import pandas as pd
        >>> from datetime import datetime
        >>> import vectorbt as vbt

        >>> symbols = ["BTC-USD", "SPY"]
        >>> price = vbt.YFData.download(symbols, missing_index='drop').get('Close')
        >>> returns = price.pct_change()
        >>> returns["BTC-USD"].vbt.returns(freq='D').stats(returns["SPY"])
        Start                    2014-09-17 00:00:00
        End                      2021-03-12 00:00:00
        Duration                  1629 days 00:00:00
        Total Return [%]                     12296.6
        Benchmark Return [%]                 122.857
        Annual Return [%]                    194.465
        Annual Volatility [%]                88.4466
        Sharpe Ratio                         1.66841
        Calmar Ratio                         2.34193
        Max. Drawdown [%]                   -83.0363
        Omega Ratio                          1.31107
        Sortino Ratio                        2.54018
        Skew                               0.0101324
        Kurtosis                              6.6453
        Tail Ratio                           1.19828
        Common Sense Ratio                    3.5285
        Value at Risk                     -0.0664826
        Alpha                                2.90175
        Beta                                0.548808
        Name: BTC-USD, dtype: object
        ```
        """
        # Run stats
        benchmark_rets = broadcast_to(benchmark_rets, self._obj)
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
            wrap_kwargs = merge_dicts(dict(name_or_index=stats_df.columns), wrap_kwargs)
            return self.wrapper.wrap_reduced(stats_df.iloc[0], **wrap_kwargs)
        return stats_df


@register_series_accessor('returns')
class ReturnsSRAccessor(ReturnsAccessor, GenericSRAccessor):
    """Accessor on top of return series. For Series only.

    Accessible through `pd.Series.vbt.returns`."""

    def __init__(self, obj: tp.Series, year_freq: tp.Optional[tp.FrequencyLike] = None, **kwargs) -> None:
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        GenericSRAccessor.__init__(self, obj, **kwargs)
        ReturnsAccessor.__init__(self, obj, year_freq=year_freq, **kwargs)

    def plot_cum_returns(self,
                         benchmark_rets: tp.Optional[tp.ArrayLike] = None,
                         start_value: float = 1,
                         fill_to_benchmark: bool = False,
                         main_kwargs: tp.KwargsLike = None,
                         benchmark_kwargs: tp.KwargsLike = None,
                         hline_shape_kwargs: tp.KwargsLike = None,
                         add_trace_kwargs: tp.KwargsLike = None,
                         xref: str = 'x',
                         yref: str = 'y',
                         fig: tp.Optional[tp.BaseFigure] = None,
                         **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot cumulative returns.

        Args:
            benchmark_rets (array_like): Benchmark return to compare returns against.
                Will broadcast per element.
            start_value (float): The starting returns.
            fill_to_benchmark (bool): Whether to fill between main and benchmark, or between main and `start_value`.
            main_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot` for main.
            benchmark_kwargs (dict): Keyword arguments passed to `vectorbt.generic.accessors.GenericSRAccessor.plot` for benchmark.
            hline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for `start_value` line.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
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

        ![](/vectorbt/docs/img/plot_cum_returns.svg)
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        x_domain = get_domain(xref, fig)
        fill_to_benchmark = fill_to_benchmark and benchmark_rets is not None

        if benchmark_rets is not None:
            # Plot benchmark
            benchmark_rets = broadcast_to(benchmark_rets, self._obj)
            if benchmark_kwargs is None:
                benchmark_kwargs = {}
            benchmark_kwargs = merge_dicts(dict(
                trace_kwargs=dict(
                    line=dict(
                        color=plotting_cfg['color_schema']['gray']
                    ),
                    name='Benchmark'
                )
            ), benchmark_kwargs)
            benchmark_cumrets = benchmark_rets.vbt.returns.cumulative(start_value=start_value)
            benchmark_cumrets.vbt.plot(**benchmark_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        else:
            benchmark_cumrets = None

        # Plot main
        if main_kwargs is None:
            main_kwargs = {}
        main_kwargs = merge_dicts(dict(
            trace_kwargs=dict(
                line=dict(
                    color=plotting_cfg['color_schema']['purple']
                )
            ),
            other_trace_kwargs='hidden'
        ), main_kwargs)
        cumrets = self.cumulative(start_value=start_value)
        if fill_to_benchmark:
            cumrets.vbt.plot_against(benchmark_cumrets, **main_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)
        else:
            cumrets.vbt.plot_against(start_value, **main_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)

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
class ReturnsDFAccessor(ReturnsAccessor, GenericDFAccessor):
    """Accessor on top of return series. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.returns`."""

    def __init__(self, obj: tp.Frame, year_freq: tp.Optional[tp.FrequencyLike] = None, **kwargs) -> None:
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        GenericDFAccessor.__init__(self, obj, **kwargs)
        ReturnsAccessor.__init__(self, obj, year_freq=year_freq, **kwargs)
