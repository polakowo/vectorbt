import numpy as np
import pandas as pd
from datetime import datetime
import pytest
import empyrical

import vectorbt as vbt

from tests.utils import isclose

day_dt = np.timedelta64(86400000000000)

ts = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [5, 4, 3, 2, 1],
    'c': [1, 2, 3, 2, 1]
}, index=pd.DatetimeIndex([
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5)
]))
ret = ts.pct_change()

seed = 42

np.random.seed(seed)
benchmark_rets = pd.DataFrame({
    'a': ret['a'] * np.random.uniform(0.8, 1.2, ret.shape[0]),
    'b': ret['b'] * np.random.uniform(0.8, 1.2, ret.shape[0]) * 2,
    'c': ret['c'] * np.random.uniform(0.8, 1.2, ret.shape[0]) * 3
})


# ############# Global ############# #

def setup_module():
    vbt.settings.numba['check_func_suffix'] = True
    vbt.settings.caching.enabled = False
    vbt.settings.caching.whitelist = []
    vbt.settings.caching.blacklist = []
    vbt.settings.returns['year_freq'] = '252 days'  # same as empyrical


def teardown_module():
    vbt.settings.reset()


# ############# accessors.py ############# #


class TestAccessors:
    def test_freq(self):
        assert ret.vbt.returns.wrapper.freq == day_dt
        assert ret['a'].vbt.returns.wrapper.freq == day_dt
        assert ret.vbt.returns(freq='2D').wrapper.freq == day_dt * 2
        assert ret['a'].vbt.returns(freq='2D').wrapper.freq == day_dt * 2
        assert pd.Series([1, 2, 3]).vbt.returns.wrapper.freq is None
        assert pd.Series([1, 2, 3]).vbt.returns(freq='3D').wrapper.freq == day_dt * 3
        assert pd.Series([1, 2, 3]).vbt.returns(freq=np.timedelta64(4, 'D')).wrapper.freq == day_dt * 4

    def test_ann_factor(self):
        assert ret['a'].vbt.returns(year_freq='365 days').ann_factor == 365
        assert ret.vbt.returns(year_freq='365 days').ann_factor == 365
        with pytest.raises(Exception) as e_info:
            assert pd.Series([1, 2, 3]).vbt.returns(freq=None).ann_factor

    def test_from_value(self):
        pd.testing.assert_series_equal(pd.Series.vbt.returns.from_value(ts['a']).obj, ts['a'].pct_change())
        pd.testing.assert_frame_equal(pd.DataFrame.vbt.returns.from_value(ts).obj, ts.pct_change())
        assert pd.Series.vbt.returns.from_value(ts['a'], year_freq='365 days').year_freq == pd.to_timedelta('365 days')
        assert pd.DataFrame.vbt.returns.from_value(ts, year_freq='365 days').year_freq == pd.to_timedelta('365 days')

    def test_daily(self):
        ret_12h = pd.DataFrame({
            'a': [0.1, 0.1, 0.1, 0.1, 0.1],
            'b': [-0.1, -0.1, -0.1, -0.1, -0.1],
            'c': [0.1, -0.1, 0.1, -0.1, 0.1]
        }, index=pd.DatetimeIndex([
            datetime(2018, 1, 1, 0),
            datetime(2018, 1, 1, 12),
            datetime(2018, 1, 2, 0),
            datetime(2018, 1, 2, 12),
            datetime(2018, 1, 3, 0)
        ]))
        pd.testing.assert_series_equal(
            ret_12h['a'].vbt.returns.daily(),
            pd.Series(
                np.array([0.21, 0.21, 0.1]),
                index=pd.DatetimeIndex([
                    '2018-01-01',
                    '2018-01-02',
                    '2018-01-03'
                ], dtype='datetime64[ns]', freq='D'),
                name=ret_12h['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            ret_12h.vbt.returns.daily(),
            pd.DataFrame(
                np.array([
                    [0.21, -0.19, -0.01],
                    [0.21, -0.19, -0.01],
                    [0.1, -0.1, 0.1]
                ]),
                index=pd.DatetimeIndex([
                    '2018-01-01',
                    '2018-01-02',
                    '2018-01-03'
                ], dtype='datetime64[ns]', freq='D'),
                columns=ret_12h.columns
            )
        )

    def test_annual(self):
        pd.testing.assert_series_equal(
            ret['a'].vbt.returns.annual(),
            pd.Series(
                np.array([4.]),
                index=pd.DatetimeIndex(['2018-01-01'], dtype='datetime64[ns]', freq='252D'),
                name=ret['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            ret.vbt.returns.annual(),
            pd.DataFrame(
                np.array([[4., -0.8, 0.]]),
                index=pd.DatetimeIndex(['2018-01-01'], dtype='datetime64[ns]', freq='252D'),
                columns=ret.columns
            )
        )

    def test_cumulative(self):
        res_a = empyrical.cum_returns(ret['a']).rename('a')
        res_b = empyrical.cum_returns(ret['b']).rename('b')
        res_c = empyrical.cum_returns(ret['c']).rename('c')
        pd.testing.assert_series_equal(
            ret['a'].vbt.returns.cumulative(),
            res_a
        )
        pd.testing.assert_frame_equal(
            ret.vbt.returns.cumulative(),
            pd.concat([res_a, res_b, res_c], axis=1)
        )

    def test_total_return(self):
        res_a = empyrical.cum_returns_final(ret['a'])
        res_b = empyrical.cum_returns_final(ret['b'])
        res_c = empyrical.cum_returns_final(ret['c'])
        assert isclose(ret['a'].vbt.returns.total(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.total(),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('total_return')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_total(ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_annualized_return(self):
        res_a = empyrical.annual_return(ret['a'])
        res_b = empyrical.annual_return(ret['b'])
        res_c = empyrical.annual_return(ret['c'])
        assert isclose(ret['a'].vbt.returns.annualized(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.annualized(),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('annualized_return')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_annualized(ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    @pytest.mark.parametrize(
        "test_alpha",
        [1., 2., 3.],
    )
    def test_annualized_volatility(self, test_alpha):
        res_a = empyrical.annual_volatility(ret['a'], alpha=test_alpha)
        res_b = empyrical.annual_volatility(ret['b'], alpha=test_alpha)
        res_c = empyrical.annual_volatility(ret['c'], alpha=test_alpha)
        assert isclose(ret['a'].vbt.returns.annualized_volatility(levy_alpha=test_alpha), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.annualized_volatility(levy_alpha=test_alpha),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('annualized_volatility')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_annualized_volatility(ret.shape[0], minp=1, levy_alpha=test_alpha).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_calmar_ratio(self):
        res_a = empyrical.calmar_ratio(ret['a'])
        res_b = empyrical.calmar_ratio(ret['b'])
        res_c = empyrical.calmar_ratio(ret['c'])
        assert isclose(ret['a'].vbt.returns.calmar_ratio(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.calmar_ratio(),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('calmar_ratio')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_calmar_ratio(ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    @pytest.mark.parametrize(
        "test_risk_free,test_required_return",
        [(0.01, 0.1), (0.02, 0.2), (0.03, 0.3)],
    )
    def test_omega_ratio(self, test_risk_free, test_required_return):
        res_a = empyrical.omega_ratio(ret['a'], risk_free=test_risk_free, required_return=test_required_return)
        if np.isnan(res_a):
            res_a = np.inf
        res_b = empyrical.omega_ratio(ret['b'], risk_free=test_risk_free, required_return=test_required_return)
        if np.isnan(res_b):
            res_b = np.inf
        res_c = empyrical.omega_ratio(ret['c'], risk_free=test_risk_free, required_return=test_required_return)
        if np.isnan(res_c):
            res_c = np.inf
        assert isclose(ret['a'].vbt.returns.omega_ratio(
            risk_free=test_risk_free, required_return=test_required_return), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.omega_ratio(risk_free=test_risk_free, required_return=test_required_return),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('omega_ratio')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_omega_ratio(
                ret.shape[0], minp=1, risk_free=test_risk_free, required_return=test_required_return).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    @pytest.mark.parametrize(
        "test_risk_free",
        [0.01, 0.02, 0.03],
    )
    def test_sharpe_ratio(self, test_risk_free):
        res_a = empyrical.sharpe_ratio(ret['a'], risk_free=test_risk_free)
        res_b = empyrical.sharpe_ratio(ret['b'], risk_free=test_risk_free)
        res_c = empyrical.sharpe_ratio(ret['c'], risk_free=test_risk_free)
        assert isclose(ret['a'].vbt.returns.sharpe_ratio(risk_free=test_risk_free), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.sharpe_ratio(risk_free=test_risk_free),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('sharpe_ratio')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_sharpe_ratio(ret.shape[0], minp=1, risk_free=test_risk_free).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_deflated_sharpe_ratio(self):
        pd.testing.assert_series_equal(
            ret.vbt.returns.deflated_sharpe_ratio(risk_free=0.01),
            pd.Series([np.nan, np.nan, 0.0005355605507117676], index=ret.columns).rename('deflated_sharpe_ratio')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.deflated_sharpe_ratio(risk_free=0.03),
            pd.Series([np.nan, np.nan, 0.0003423112350834066], index=ret.columns).rename('deflated_sharpe_ratio')
        )

    @pytest.mark.parametrize(
        "test_required_return",
        [0.01, 0.02, 0.03],
    )
    def test_downside_risk(self, test_required_return):
        res_a = empyrical.downside_risk(ret['a'], required_return=test_required_return)
        res_b = empyrical.downside_risk(ret['b'], required_return=test_required_return)
        res_c = empyrical.downside_risk(ret['c'], required_return=test_required_return)
        assert isclose(ret['a'].vbt.returns.downside_risk(required_return=test_required_return), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.downside_risk(required_return=test_required_return),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('downside_risk')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_downside_risk(
                ret.shape[0], minp=1, required_return=test_required_return).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    @pytest.mark.parametrize(
        "test_required_return",
        [0.01, 0.02, 0.03],
    )
    def test_sortino_ratio(self, test_required_return):
        res_a = empyrical.sortino_ratio(ret['a'], required_return=test_required_return)
        res_b = empyrical.sortino_ratio(ret['b'], required_return=test_required_return)
        res_c = empyrical.sortino_ratio(ret['c'], required_return=test_required_return)
        assert isclose(ret['a'].vbt.returns.sortino_ratio(required_return=test_required_return), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.sortino_ratio(required_return=test_required_return),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('sortino_ratio')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_sortino_ratio(
                ret.shape[0], minp=1, required_return=test_required_return).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_information_ratio(self):
        res_a = empyrical.excess_sharpe(ret['a'], benchmark_rets['a'])
        res_b = empyrical.excess_sharpe(ret['b'], benchmark_rets['b'])
        res_c = empyrical.excess_sharpe(ret['c'], benchmark_rets['c'])
        assert isclose(ret['a'].vbt.returns.information_ratio(benchmark_rets['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.information_ratio(benchmark_rets),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('information_ratio')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_information_ratio(
                benchmark_rets, ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_beta(self):
        res_a = empyrical.beta(ret['a'], benchmark_rets['a'])
        res_b = empyrical.beta(ret['b'], benchmark_rets['b'])
        res_c = empyrical.beta(ret['c'], benchmark_rets['c'])
        assert isclose(ret['a'].vbt.returns.beta(benchmark_rets['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.beta(benchmark_rets),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('beta')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_beta(
                benchmark_rets, ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    @pytest.mark.parametrize(
        "test_risk_free",
        [0.01, 0.02, 0.03],
    )
    def test_alpha(self, test_risk_free):
        res_a = empyrical.alpha(ret['a'], benchmark_rets['a'], risk_free=test_risk_free)
        res_b = empyrical.alpha(ret['b'], benchmark_rets['b'], risk_free=test_risk_free)
        res_c = empyrical.alpha(ret['c'], benchmark_rets['c'], risk_free=test_risk_free)
        assert isclose(ret['a'].vbt.returns.alpha(benchmark_rets['a'], risk_free=test_risk_free), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.alpha(benchmark_rets, risk_free=test_risk_free),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('alpha')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_alpha(
                benchmark_rets, ret.shape[0], minp=1, risk_free=test_risk_free).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_tail_ratio(self):
        res_a = empyrical.tail_ratio(ret['a'])
        res_b = empyrical.tail_ratio(ret['b'])
        res_c = empyrical.tail_ratio(ret['c'])
        assert isclose(ret['a'].vbt.returns.tail_ratio(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.tail_ratio(),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('tail_ratio')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_tail_ratio(
                ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    @pytest.mark.parametrize(
        "test_cutoff",
        [0.05, 0.06, 0.07],
    )
    def test_value_at_risk(self, test_cutoff):
        # empyrical can't tolerate NaN here
        res_a = empyrical.value_at_risk(ret['a'].iloc[1:], cutoff=test_cutoff)
        res_b = empyrical.value_at_risk(ret['b'].iloc[1:], cutoff=test_cutoff)
        res_c = empyrical.value_at_risk(ret['c'].iloc[1:], cutoff=test_cutoff)
        assert isclose(ret['a'].vbt.returns.value_at_risk(cutoff=test_cutoff), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.value_at_risk(cutoff=test_cutoff),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('value_at_risk')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_value_at_risk(
                ret.shape[0], minp=1, cutoff=test_cutoff).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    @pytest.mark.parametrize(
        "test_cutoff",
        [0.05, 0.06, 0.07],
    )
    def test_cond_value_at_risk(self, test_cutoff):
        # empyrical can't tolerate NaN here
        res_a = empyrical.conditional_value_at_risk(ret['a'].iloc[1:], cutoff=test_cutoff)
        res_b = empyrical.conditional_value_at_risk(ret['b'].iloc[1:], cutoff=test_cutoff)
        res_c = empyrical.conditional_value_at_risk(ret['c'].iloc[1:], cutoff=test_cutoff)
        assert isclose(ret['a'].vbt.returns.cond_value_at_risk(cutoff=test_cutoff), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.cond_value_at_risk(cutoff=test_cutoff),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('cond_value_at_risk')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_cond_value_at_risk(
                ret.shape[0], minp=1, cutoff=test_cutoff).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_capture(self):
        res_a = empyrical.capture(ret['a'], benchmark_rets['a'])
        res_b = empyrical.capture(ret['b'], benchmark_rets['b'])
        res_c = empyrical.capture(ret['c'], benchmark_rets['c'])
        assert isclose(ret['a'].vbt.returns.capture(benchmark_rets['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.capture(benchmark_rets),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('capture')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_capture(
                benchmark_rets, ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_up_capture(self):
        res_a = empyrical.up_capture(ret['a'], benchmark_rets['a'])
        res_b = empyrical.up_capture(ret['b'], benchmark_rets['b'])
        res_c = empyrical.up_capture(ret['c'], benchmark_rets['c'])
        assert isclose(ret['a'].vbt.returns.up_capture(benchmark_rets['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.up_capture(benchmark_rets),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('up_capture')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_up_capture(
                benchmark_rets, ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_down_capture(self):
        res_a = empyrical.down_capture(ret['a'], benchmark_rets['a'])
        res_b = empyrical.down_capture(ret['b'], benchmark_rets['b'])
        res_c = empyrical.down_capture(ret['c'], benchmark_rets['c'])
        assert isclose(ret['a'].vbt.returns.down_capture(benchmark_rets['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.down_capture(benchmark_rets),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('down_capture')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_down_capture(
                benchmark_rets, ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_drawdown(self):
        pd.testing.assert_series_equal(
            ret['a'].vbt.returns.drawdown(),
            pd.Series(
                np.array([0., 0., 0., 0., 0.]),
                index=ret['a'].index,
                name=ret['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            ret.vbt.returns.drawdown(),
            pd.DataFrame(
                np.array([
                    [0., 0., 0.],
                    [0., -0.2, 0.],
                    [0., -0.4, 0.],
                    [0., -0.6, -0.33333333],
                    [0., -0.8, -0.66666667]
                ]),
                index=pd.DatetimeIndex([
                    '2018-01-01',
                    '2018-01-02',
                    '2018-01-03',
                    '2018-01-04',
                    '2018-01-05'
                ], dtype='datetime64[ns]', freq=None),
                columns=ret.columns
            )
        )

    def test_max_drawdown(self):
        res_a = empyrical.max_drawdown(ret['a'])
        res_b = empyrical.max_drawdown(ret['b'])
        res_c = empyrical.max_drawdown(ret['c'])
        assert isclose(ret['a'].vbt.returns.max_drawdown(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.max_drawdown(),
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename('max_drawdown')
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.rolling_max_drawdown(
                ret.shape[0], minp=1).iloc[-1],
            pd.Series([res_a, res_b, res_c], index=ret.columns).rename(ret.index[-1])
        )

    def test_drawdowns(self):
        assert type(ret['a'].vbt.returns.drawdowns) is vbt.Drawdowns
        assert ret['a'].vbt.returns.drawdowns.wrapper.freq == ret['a'].vbt.wrapper.freq
        assert ret['a'].vbt.returns.drawdowns.wrapper.ndim == ret['a'].ndim
        assert ret.vbt.returns.drawdowns.wrapper.ndim == ret.ndim
        assert isclose(ret['a'].vbt.returns.drawdowns.max_drawdown(), ret['a'].vbt.returns.max_drawdown())
        pd.testing.assert_series_equal(
            ret.vbt.returns.drawdowns.max_drawdown(default_val=0.),
            ret.vbt.returns.max_drawdown()
        )

    def test_stats(self):
        stat_index = pd.Index([
            'Start',
            'End',
            'Period',
            'Total Return [%]',
            'Annualized Return [%]',
            'Annualized Volatility [%]',
            'Max Drawdown [%]',
            'Max Drawdown Duration',
            'Sharpe Ratio',
            'Calmar Ratio',
            'Omega Ratio',
            'Sortino Ratio',
            'Skew',
            'Kurtosis',
            'Tail Ratio',
            'Common Sense Ratio',
            'Value at Risk'
        ], dtype='object')
        pd.testing.assert_series_equal(
            ret.vbt.returns.stats(),
            pd.Series([
                pd.Timestamp('2018-01-01 00:00:00'),
                pd.Timestamp('2018-01-05 00:00:00'),
                pd.Timedelta('5 days 00:00:00'),
                106.66666666666667,
                5.635947823148613e+36,
                621.5038995587341,
                73.33333333333333,
                pd.Timedelta('3 days 00:00:00'),
                -3.4590654330766135,
                -0.625,
                np.inf,
                np.inf,
                0.25687104876585726,
                -0.25409565813913854,
                1.9693400167084374,
                1.9860006614904637e+35,
                -0.2291666666666667
            ],
                index=stat_index,
                name='agg_func_mean'
            )
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.stats(column='a'),
            pd.Series([
                pd.Timestamp('2018-01-01 00:00:00'),
                pd.Timestamp('2018-01-05 00:00:00'),
                pd.Timedelta('5 days 00:00:00'),
                400.0,
                1.690784346944584e+37,
                533.2682251925386,
                np.nan,
                np.nan,
                24.612379624271007,
                np.nan,
                np.inf,
                np.inf,
                1.4693345482106241,
                2.030769230769236,
                3.5238095238095237,
                5.958001984471391e+35,
                0.26249999999999996
            ],
                index=stat_index,
                name='a'
            )
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.stats(column='a', settings=dict(freq='10 days', year_freq='200 days')),
            pd.Series([
                pd.Timestamp('2018-01-01 00:00:00'),
                pd.Timestamp('2018-01-05 00:00:00'),
                pd.Timedelta('50 days 00:00:00'),
                400.0,
                62400.0,
                150.23130314433288,
                np.nan,
                np.nan,
                6.933752452815364,
                np.nan,
                np.inf,
                np.inf,
                1.4693345482106241,
                2.030769230769236,
                3.5238095238095237,
                2202.3809523809523,
                0.26249999999999996
            ],
                index=stat_index,
                name='a'
            )
        )
        pd.testing.assert_series_equal(
            ret.vbt.returns.stats(column='a', settings=dict(benchmark_rets=benchmark_rets)),
            pd.Series([
                pd.Timestamp('2018-01-01 00:00:00'),
                pd.Timestamp('2018-01-05 00:00:00'),
                pd.Timedelta('5 days 00:00:00'),
                400.0,
                451.8597134178033,
                1.690784346944584e+37,
                533.2682251925386,
                np.nan,
                np.nan,
                24.612379624271007,
                np.nan,
                np.inf,
                np.inf,
                1.4693345482106241,
                2.030769230769236,
                3.5238095238095237,
                5.958001984471391e+35,
                0.26249999999999996,
                35691351.69391792,
                0.7853755858374825
            ],
                index=pd.Index([
                    'Start',
                    'End',
                    'Period',
                    'Total Return [%]',
                    'Benchmark Return [%]',
                    'Annualized Return [%]',
                    'Annualized Volatility [%]',
                    'Max Drawdown [%]',
                    'Max Drawdown Duration',
                    'Sharpe Ratio',
                    'Calmar Ratio',
                    'Omega Ratio',
                    'Sortino Ratio',
                    'Skew',
                    'Kurtosis',
                    'Tail Ratio',
                    'Common Sense Ratio',
                    'Value at Risk',
                    'Alpha',
                    'Beta'
                ], dtype='object'),
                name='a'
            )
        )
        pd.testing.assert_series_equal(
            ret['c'].vbt.returns.stats(),
            ret.vbt.returns.stats(column='c')
        )
        pd.testing.assert_series_equal(
            ret['c'].vbt.returns.stats(),
            ret.vbt.returns.stats(column='c', group_by=False)
        )
        stats_df = ret.vbt.returns.stats(agg_func=None)
        assert stats_df.shape == (3, 17)
        pd.testing.assert_index_equal(stats_df.index, ret.vbt.returns.wrapper.columns)
        pd.testing.assert_index_equal(stats_df.columns, stat_index)
