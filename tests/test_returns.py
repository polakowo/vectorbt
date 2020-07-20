import numpy as np
import pandas as pd
from datetime import datetime
import pytest
import empyrical

from vectorbt import defaults
from vectorbt.records.drawdowns import Drawdowns

from tests.utils import isclose

day_dt = np.timedelta64(86400000000000)

index = pd.DatetimeIndex([
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5)
])
ts = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [5, 4, 3, 2, 1],
    'c': [1, 2, 3, 2, 1]
}, index=index)
ret = ts.pct_change()

defaults.returns['year_freq'] = '252 days'  # same as empyrical

factor_returns = pd.DataFrame({
    'a': ret['a'] * np.random.uniform(0.8, 1.2, ret.shape[0]),
    'b': ret['b'] * np.random.uniform(0.8, 1.2, ret.shape[0]) * 2,
    'c': ret['c'] * np.random.uniform(0.8, 1.2, ret.shape[0]) * 3
})


# ############# accessors.py ############# #


class TestAccessors:
    def test_freq(self):
        assert ret.vbt.returns.freq == day_dt
        assert ret['a'].vbt.returns.freq == day_dt
        assert ret.vbt.returns(freq='2D').freq == day_dt * 2
        assert ret['a'].vbt.returns(freq='2D').freq == day_dt * 2
        assert pd.Series([1, 2, 3]).vbt.returns.freq is None
        assert pd.Series([1, 2, 3]).vbt.returns(freq='3D').freq == day_dt * 3
        assert pd.Series([1, 2, 3]).vbt.returns(freq=np.timedelta64(4, 'D')).freq == day_dt * 4

    def test_year_freq(self):
        assert ret.vbt.returns.year_freq == pd.to_timedelta(defaults.returns['year_freq'])
        assert ret['a'].vbt.returns.year_freq == pd.to_timedelta(defaults.returns['year_freq'])
        assert ret['a'].vbt.returns(year_freq='365 days').year_freq == pd.to_timedelta('365 days')
        assert ret.vbt.returns(year_freq='365 days').year_freq == pd.to_timedelta('365 days')

    def test_ann_factor(self):
        assert ret['a'].vbt.returns(year_freq='365 days').ann_factor == 365
        assert ret.vbt.returns(year_freq='365 days').ann_factor == 365

    def test_from_price(self):
        pd.testing.assert_series_equal(pd.Series.vbt.returns.from_price(ts['a'])._obj, ts['a'].pct_change())
        pd.testing.assert_frame_equal(pd.DataFrame.vbt.returns.from_price(ts)._obj, ts.pct_change())
        assert pd.Series.vbt.returns.from_price(ts['a'], year_freq='365 days').year_freq == pd.to_timedelta('365 days')
        assert pd.DataFrame.vbt.returns.from_price(ts, year_freq='365 days').year_freq == pd.to_timedelta('365 days')

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
                index=pd.DatetimeIndex(['2018-01-01'], dtype='datetime64[ns]', freq='365D'),
                name=ret['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            ret.vbt.returns.annual(),
            pd.DataFrame(
                np.array([[4., -0.8, 0.]]),
                index=pd.DatetimeIndex(['2018-01-01'], dtype='datetime64[ns]', freq='365D'),
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

    def test_total(self):
        res_a = empyrical.cum_returns_final(ret['a'])
        res_b = empyrical.cum_returns_final(ret['b'])
        res_c = empyrical.cum_returns_final(ret['c'])
        assert isclose(ret['a'].vbt.returns.total(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.total(),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_annualized_return(self):
        res_a = empyrical.annual_return(ret['a'])
        res_b = empyrical.annual_return(ret['b'])
        res_c = empyrical.annual_return(ret['c'])
        assert isclose(ret['a'].vbt.returns.annualized_return(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.annualized_return(),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
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
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_calmar_ratio(self):
        res_a = empyrical.calmar_ratio(ret['a'])
        res_b = empyrical.calmar_ratio(ret['b'])
        res_c = empyrical.calmar_ratio(ret['c'])
        assert isclose(ret['a'].vbt.returns.calmar_ratio(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.calmar_ratio(),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
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
            pd.Series([res_a, res_b, res_c], index=ret.columns)
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
            pd.Series([res_a, res_b, res_c], index=ret.columns)
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
            pd.Series([res_a, res_b, res_c], index=ret.columns)
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
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_information_ratio(self):
        res_a = empyrical.excess_sharpe(ret['a'], factor_returns['a'])
        res_b = empyrical.excess_sharpe(ret['b'], factor_returns['b'])
        res_c = empyrical.excess_sharpe(ret['c'], factor_returns['c'])
        assert isclose(ret['a'].vbt.returns.information_ratio(factor_returns['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.information_ratio(factor_returns),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_beta(self):
        res_a = empyrical.beta(ret['a'], factor_returns['a'])
        res_b = empyrical.beta(ret['b'], factor_returns['b'])
        res_c = empyrical.beta(ret['c'], factor_returns['c'])
        assert isclose(ret['a'].vbt.returns.beta(factor_returns['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.beta(factor_returns),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    @pytest.mark.parametrize(
        "test_risk_free",
        [0.01, 0.02, 0.03],
    )
    def test_alpha(self, test_risk_free):
        res_a = empyrical.alpha(ret['a'], factor_returns['a'], risk_free=test_risk_free)
        res_b = empyrical.alpha(ret['b'], factor_returns['b'], risk_free=test_risk_free)
        res_c = empyrical.alpha(ret['c'], factor_returns['c'], risk_free=test_risk_free)
        assert isclose(ret['a'].vbt.returns.alpha(factor_returns['a'], risk_free=test_risk_free), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.alpha(factor_returns, risk_free=test_risk_free),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_tail_ratio(self):
        res_a = empyrical.tail_ratio(ret['a'])
        res_b = empyrical.tail_ratio(ret['b'])
        res_c = empyrical.tail_ratio(ret['c'])
        assert isclose(ret['a'].vbt.returns.tail_ratio(), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.tail_ratio(),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    @pytest.mark.parametrize(
        "test_cutoff",
        [0.05, 0.06, 0.07],
    )
    def test_value_at_risk(self, test_cutoff):
        # empyrical can't tolerate NaNs here
        res_a = empyrical.value_at_risk(ret['a'].iloc[1:], cutoff=test_cutoff)
        res_b = empyrical.value_at_risk(ret['b'].iloc[1:], cutoff=test_cutoff)
        res_c = empyrical.value_at_risk(ret['c'].iloc[1:], cutoff=test_cutoff)
        assert isclose(ret['a'].vbt.returns.value_at_risk(cutoff=test_cutoff), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.value_at_risk(cutoff=test_cutoff),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    @pytest.mark.parametrize(
        "test_cutoff",
        [0.05, 0.06, 0.07],
    )
    def test_conditional_value_at_risk(self, test_cutoff):
        # empyrical can't tolerate NaNs here
        res_a = empyrical.conditional_value_at_risk(ret['a'].iloc[1:], cutoff=test_cutoff)
        res_b = empyrical.conditional_value_at_risk(ret['b'].iloc[1:], cutoff=test_cutoff)
        res_c = empyrical.conditional_value_at_risk(ret['c'].iloc[1:], cutoff=test_cutoff)
        assert isclose(ret['a'].vbt.returns.conditional_value_at_risk(cutoff=test_cutoff), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.conditional_value_at_risk(cutoff=test_cutoff),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_capture(self):
        res_a = empyrical.capture(ret['a'], factor_returns['a'])
        res_b = empyrical.capture(ret['b'], factor_returns['b'])
        res_c = empyrical.capture(ret['c'], factor_returns['c'])
        assert isclose(ret['a'].vbt.returns.capture(factor_returns['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.capture(factor_returns),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_up_capture(self):
        res_a = empyrical.up_capture(ret['a'], factor_returns['a'])
        res_b = empyrical.up_capture(ret['b'], factor_returns['b'])
        res_c = empyrical.up_capture(ret['c'], factor_returns['c'])
        assert isclose(ret['a'].vbt.returns.up_capture(factor_returns['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.up_capture(factor_returns),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_down_capture(self):
        res_a = empyrical.down_capture(ret['a'], factor_returns['a'])
        res_b = empyrical.down_capture(ret['b'], factor_returns['b'])
        res_c = empyrical.down_capture(ret['c'], factor_returns['c'])
        assert isclose(ret['a'].vbt.returns.down_capture(factor_returns['a']), res_a)
        pd.testing.assert_series_equal(
            ret.vbt.returns.down_capture(factor_returns),
            pd.Series([res_a, res_b, res_c], index=ret.columns)
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
            pd.Series([res_a, res_b, res_c], index=ret.columns)
        )

    def test_drawdowns(self):
        assert type(ret['a'].vbt.returns.drawdowns) is Drawdowns
        assert ret['a'].vbt.returns.drawdowns.wrapper.freq == ret['a'].vbt.returns.freq
        assert ret['a'].vbt.returns.drawdowns.wrapper.ndim == ret['a'].ndim
        assert ret.vbt.returns.drawdowns.wrapper.ndim == ret.ndim
        assert isclose(ret['a'].vbt.returns.drawdowns.max_drawdown, ret['a'].vbt.returns.max_drawdown())
        pd.testing.assert_series_equal(
            ret.vbt.returns.drawdowns.max_drawdown,
            ret.vbt.returns.max_drawdown()
        )
