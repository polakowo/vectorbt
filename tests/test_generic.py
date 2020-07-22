import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
import pytest
from itertools import product

from vectorbt.generic import nb
from vectorbt.records.drawdowns import Drawdowns

day_dt = np.timedelta64(86400000000000)

ts = pd.DataFrame({
    'a': [1, 2, 3, 4, np.nan],
    'b': [np.nan, 4, 3, 2, 1],
    'c': [1, 2, np.nan, 2, 1]
}, index=pd.DatetimeIndex([
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5)
]))

@njit
def pd_nanmean_nb(x):
    return np.nanmean(x)

@njit
def nanmean_nb(col, i, x):
    return np.nanmean(x)

@njit
def nanmean_matrix_nb(i, x):
    return np.nanmean(x)


# ############# accessors.py ############# #


class TestAccessors:
    def test_split_into_ranges(self):
        pd.testing.assert_frame_equal(
            ts['a'].vbt.split_into_ranges(n=2),
            pd.DataFrame(
                np.array([
                    [1., 4.],
                    [2., np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.DatetimeIndex([
                        '2018-01-01', '2018-01-04'
                    ], dtype='datetime64[ns]', name='range_start', freq=None),
                    pd.DatetimeIndex([
                        '2018-01-02', '2018-01-05'
                    ], dtype='datetime64[ns]', name='range_end', freq=None)
                ])
            )
        )
        pd.testing.assert_frame_equal(
            ts['a'].vbt.split_into_ranges(range_len=2),
            pd.DataFrame(
                np.array([
                    [1., 2., 3., 4.],
                    [2., 3., 4., np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.DatetimeIndex([
                        '2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'
                    ], dtype='datetime64[ns]', name='range_start', freq=None),
                    pd.DatetimeIndex([
                        '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'
                    ], dtype='datetime64[ns]', name='range_end', freq=None)
                ])
            )
        )
        pd.testing.assert_frame_equal(
            ts['a'].vbt.split_into_ranges(range_len=2, n=3),
            pd.DataFrame(
                np.array([
                    [1., 3., 4.],
                    [2., 4., np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.DatetimeIndex([
                        '2018-01-01', '2018-01-03', '2018-01-04'
                    ], dtype='datetime64[ns]', name='range_start', freq=None),
                    pd.DatetimeIndex([
                        '2018-01-02', '2018-01-04', '2018-01-05'
                    ], dtype='datetime64[ns]', name='range_end', freq=None)
                ])
            )
        )
        pd.testing.assert_frame_equal(
            ts['a'].vbt.split_into_ranges(range_len=3, n=2),
            pd.DataFrame(
                np.array([
                    [1., 3.],
                    [2., 4.],
                    [3., np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.DatetimeIndex([
                        '2018-01-01', '2018-01-03'
                    ], dtype='datetime64[ns]', name='range_start', freq=None),
                    pd.DatetimeIndex([
                        '2018-01-03', '2018-01-05'
                    ], dtype='datetime64[ns]', name='range_end', freq=None)
                ])
            )
        )
        pd.testing.assert_frame_equal(
            ts.vbt.split_into_ranges(n=2),
            pd.DataFrame(
                np.array([
                    [1., 4., np.nan, 2., 1., 2.],
                    [2., np.nan, 4., 1., 2., 1.]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.Index(['a', 'a', 'b', 'b', 'c', 'c'], dtype='object'),
                    pd.DatetimeIndex([
                        '2018-01-01', '2018-01-04', '2018-01-01', '2018-01-04', '2018-01-01', '2018-01-04'
                    ], dtype='datetime64[ns]', name='range_start', freq=None),
                    pd.DatetimeIndex([
                        '2018-01-02', '2018-01-05', '2018-01-02', '2018-01-05', '2018-01-02', '2018-01-05'
                    ], dtype='datetime64[ns]', name='range_end', freq=None)
                ])
            )
        )

    @pytest.mark.parametrize(
        "test_value",
        [-1, 0., np.nan],
    )
    def test_fillna(self, test_value):
        pd.testing.assert_series_equal(ts['a'].vbt.fillna(test_value), ts['a'].fillna(test_value))
        pd.testing.assert_frame_equal(ts.vbt.fillna(test_value), ts.fillna(test_value))

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_fshift(self, test_n):
        pd.testing.assert_series_equal(ts['a'].vbt.fshift(test_n), ts['a'].shift(test_n))
        np.testing.assert_array_equal(
            ts['a'].vbt.fshift(test_n).values,
            nb.fshift_1d_nb(ts['a'].values, test_n)
        )
        pd.testing.assert_frame_equal(ts.vbt.fshift(test_n), ts.shift(test_n))

    def test_diff(self):
        pd.testing.assert_series_equal(ts['a'].vbt.diff(), ts['a'].diff())
        np.testing.assert_array_equal(ts['a'].vbt.diff().values, nb.diff_1d_nb(ts['a'].values))
        pd.testing.assert_frame_equal(ts.vbt.diff(), ts.diff())

    def test_pct_change(self):
        pd.testing.assert_series_equal(ts['a'].vbt.pct_change(), ts['a'].pct_change(fill_method=None))
        np.testing.assert_array_equal(ts['a'].vbt.pct_change().values, nb.pct_change_1d_nb(ts['a'].values))
        pd.testing.assert_frame_equal(ts.vbt.pct_change(), ts.pct_change(fill_method=None))

    def test_ffill(self):
        pd.testing.assert_series_equal(ts['a'].vbt.ffill(), ts['a'].ffill())
        pd.testing.assert_frame_equal(ts.vbt.ffill(), ts.ffill())

    def test_product(self):
        pd.testing.assert_series_equal(ts['a'].vbt.product(), ts['a'].product())
        pd.testing.assert_frame_equal(ts.vbt.product(), ts.product())

    def test_product(self):
        assert ts['a'].vbt.product() == ts['a'].product()
        np.testing.assert_array_equal(ts.vbt.product(), ts.product())

    def test_cumsum(self):
        pd.testing.assert_series_equal(ts['a'].vbt.cumsum(), ts['a'].cumsum())
        pd.testing.assert_frame_equal(ts.vbt.cumsum(), ts.cumsum())

    def test_cumprod(self):
        pd.testing.assert_series_equal(ts['a'].vbt.cumprod(), ts['a'].cumprod())
        pd.testing.assert_frame_equal(ts.vbt.cumprod(), ts.cumprod())

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_rolling_min(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            ts['a'].vbt.rolling_min(test_window, minp=test_minp),
            ts['a'].rolling(test_window, min_periods=test_minp).min()
        )
        pd.testing.assert_frame_equal(
            ts.vbt.rolling_min(test_window, minp=test_minp),
            ts.rolling(test_window, min_periods=test_minp).min()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_rolling_max(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            ts['a'].vbt.rolling_max(test_window, minp=test_minp),
            ts['a'].rolling(test_window, min_periods=test_minp).max()
        )
        pd.testing.assert_frame_equal(
            ts.vbt.rolling_max(test_window, minp=test_minp),
            ts.rolling(test_window, min_periods=test_minp).max()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_rolling_mean(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            ts['a'].vbt.rolling_mean(test_window, minp=test_minp),
            ts['a'].rolling(test_window, min_periods=test_minp).mean()
        )
        pd.testing.assert_frame_equal(
            ts.vbt.rolling_mean(test_window, minp=test_minp),
            ts.rolling(test_window, min_periods=test_minp).mean()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_rolling_std(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            ts['a'].vbt.rolling_std(test_window, minp=test_minp),
            ts['a'].rolling(test_window, min_periods=test_minp).std()
        )
        pd.testing.assert_frame_equal(
            ts.vbt.rolling_std(test_window, minp=test_minp),
            ts.rolling(test_window, min_periods=test_minp).std()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_ewm_mean(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            ts['a'].vbt.ewm_mean(test_window, minp=test_minp),
            ts['a'].ewm(span=test_window, min_periods=test_minp).mean()
        )
        pd.testing.assert_frame_equal(
            ts.vbt.ewm_mean(test_window, minp=test_minp),
            ts.ewm(span=test_window, min_periods=test_minp).mean()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_ewm_std(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            ts['a'].vbt.ewm_std(test_window, minp=test_minp),
            ts['a'].ewm(span=test_window, min_periods=test_minp).std()
        )
        pd.testing.assert_frame_equal(
            ts.vbt.ewm_std(test_window, minp=test_minp),
            ts.ewm(span=test_window, min_periods=test_minp).std()
        )

    def test_expanding_min(self):
        pd.testing.assert_series_equal(ts['a'].vbt.expanding_min(), ts['a'].expanding().min())
        pd.testing.assert_frame_equal(ts.vbt.expanding_min(), ts.expanding().min())

    def test_expanding_max(self):
        pd.testing.assert_series_equal(ts['a'].vbt.expanding_max(), ts['a'].expanding().max())
        pd.testing.assert_frame_equal(ts.vbt.expanding_max(), ts.expanding().max())

    def test_expanding_mean(self):
        pd.testing.assert_series_equal(ts['a'].vbt.expanding_mean(), ts['a'].expanding().mean())
        pd.testing.assert_frame_equal(ts.vbt.expanding_mean(), ts.expanding().mean())

    def test_expanding_std(self):
        pd.testing.assert_series_equal(ts['a'].vbt.expanding_std(), ts['a'].expanding().std())
        pd.testing.assert_frame_equal(ts.vbt.expanding_std(), ts.expanding().std())

    @pytest.mark.parametrize(
        "test_window",
        [1, 2, 3, 4, 5],
    )
    def test_rolling_apply(self, test_window):
        pd.testing.assert_series_equal(
            ts['a'].rolling(test_window, min_periods=1).apply(pd_nanmean_nb, raw=True),
            ts['a'].vbt.rolling_apply(test_window, nanmean_nb)
        )
        pd.testing.assert_frame_equal(
            ts.rolling(test_window, min_periods=1).apply(pd_nanmean_nb, raw=True),
            ts.vbt.rolling_apply(test_window, nanmean_nb)
        )

    def test_rolling_apply_on_matrix(self):
        pd.testing.assert_frame_equal(
            ts.vbt.rolling_apply(3, nanmean_matrix_nb, on_matrix=True),
            pd.DataFrame(
                np.array([
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [2.28571429, 2.28571429, 2.28571429],
                    [2.75, 2.75, 2.75],
                    [2.28571429, 2.28571429, 2.28571429]
                ]),
                index=ts.index,
                columns=ts.columns
            )
        )

    def test_expanding_apply(self):
        pd.testing.assert_series_equal(
            ts['a'].expanding(min_periods=1).apply(pd_nanmean_nb, raw=True),
            ts['a'].vbt.expanding_apply(nanmean_nb)
        )
        pd.testing.assert_frame_equal(
            ts.expanding(min_periods=1).apply(pd_nanmean_nb, raw=True),
            ts.vbt.expanding_apply(nanmean_nb)
        )

    def test_expanding_apply_on_matrix(self):
        pd.testing.assert_frame_equal(
            ts.vbt.expanding_apply(nanmean_matrix_nb, on_matrix=True),
            pd.DataFrame(
                np.array([
                    [1., 1., 1.],
                    [2., 2., 2.],
                    [2.28571429, 2.28571429, 2.28571429],
                    [2.4, 2.4, 2.4],
                    [2.16666667, 2.16666667, 2.16666667]
                ]),
                index=ts.index,
                columns=ts.columns
            )
        )

    def test_groupby_apply(self):
        pd.testing.assert_series_equal(
            ts['a'].groupby(np.asarray([1, 1, 2, 2, 3])).apply(lambda x: pd_nanmean_nb(x.values)),
            ts['a'].vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), nanmean_nb)
        )
        pd.testing.assert_frame_equal(
            ts.groupby(np.asarray([1, 1, 2, 2, 3])).agg({
                'a': lambda x: pd_nanmean_nb(x.values),
                'b': lambda x: pd_nanmean_nb(x.values),
                'c': lambda x: pd_nanmean_nb(x.values)
            }),  # any clean way to do column-wise grouping in pandas?
            ts.vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), nanmean_nb)
        )

    def test_groupby_apply_on_matrix(self):
        pd.testing.assert_frame_equal(
            ts.vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), nanmean_matrix_nb, on_matrix=True),
            pd.DataFrame(
                np.array([
                    [2., 2., 2.],
                    [2.8, 2.8, 2.8],
                    [1., 1., 1.]
                ]),
                index=pd.Int64Index([1, 2, 3], dtype='int64'),
                columns=ts.columns
            )
        )

    @pytest.mark.parametrize(
        "test_freq",
        ['1h', '3d', '1w'],
    )
    def test_resample_apply(self, test_freq):
        pd.testing.assert_series_equal(
            ts['a'].resample(test_freq).apply(lambda x: pd_nanmean_nb(x.values)),
            ts['a'].vbt.resample_apply(test_freq, nanmean_nb)
        )
        pd.testing.assert_frame_equal(
            ts.resample(test_freq).apply(lambda x: pd_nanmean_nb(x.values)),
            ts.vbt.resample_apply(test_freq, nanmean_nb)
        )

    def test_resample_apply_on_matrix(self):
        pd.testing.assert_frame_equal(
            ts.vbt.resample_apply('3d', nanmean_matrix_nb, on_matrix=True),
            pd.DataFrame(
                np.array([
                    [2.28571429, 2.28571429, 2.28571429],
                    [2., 2., 2.]
                ]),
                index=pd.DatetimeIndex(['2018-01-01', '2018-01-04'], dtype='datetime64[ns]', freq='3D'),
                columns=ts.columns
            )
        )

    def test_applymap(self):
        @njit
        def mult_nb(col, i, x):
            return x * 2

        pd.testing.assert_series_equal(
            ts['a'].map(lambda x: x * 2),
            ts['a'].vbt.applymap(mult_nb)
        )
        pd.testing.assert_frame_equal(
            ts.applymap(lambda x: x * 2),
            ts.vbt.applymap(mult_nb)
        )

    def test_filter(self):
        @njit
        def greater_nb(col, i, x):
            return x > 2

        pd.testing.assert_series_equal(
            ts['a'].map(lambda x: x if x > 2 else np.nan),
            ts['a'].vbt.filter(greater_nb)
        )
        pd.testing.assert_frame_equal(
            ts.applymap(lambda x: x if x > 2 else np.nan),
            ts.vbt.filter(greater_nb)
        )

    def test_apply_and_reduce(self):
        @njit
        def every_nth_nb(col, a, n):
            return a[::n]

        @njit
        def sum_nb(col, a, n):
            return np.nansum(a)

        assert ts['a'].iloc[::2].sum() == ts['a'].vbt.apply_and_reduce(every_nth_nb, sum_nb, 2)
        pd.testing.assert_series_equal(
            ts.iloc[::2].sum(),
            ts.vbt.apply_and_reduce(every_nth_nb, sum_nb, 2)
        )
        pd.testing.assert_series_equal(
            ts.iloc[::2].sum() * day_dt,
            ts.vbt.apply_and_reduce(every_nth_nb, sum_nb, 2, time_units=True)
        )

    def test_reduce(self):
        @njit
        def sum_nb(col, a):
            return np.nansum(a)

        assert ts['a'].sum() == ts['a'].vbt.reduce(sum_nb)
        pd.testing.assert_series_equal(
            ts.sum(),
            ts.vbt.reduce(sum_nb)
        )
        pd.testing.assert_series_equal(
            ts.sum() * day_dt,
            ts.vbt.reduce(sum_nb, time_units=True)
        )

    def test_reduce_to_array(self):
        @njit
        def min_and_max_nb(col, a):
            result = np.empty(2)
            result[0] = np.nanmin(a)
            result[1] = np.nanmax(a)
            return result

        result = ts.apply(lambda x: np.asarray([np.min(x), np.max(x)]), axis=0)
        pd.testing.assert_series_equal(
            result['a'],
            ts['a'].vbt.reduce_to_array(min_and_max_nb)
        )
        result.index = pd.Index(['min', 'max'])
        pd.testing.assert_series_equal(
            result['a'],
            ts['a'].vbt.reduce_to_array(min_and_max_nb, index=['min', 'max'])
        )
        pd.testing.assert_frame_equal(
            result,
            ts.vbt.reduce_to_array(min_and_max_nb, index=['min', 'max'])
        )
        pd.testing.assert_frame_equal(
            result * day_dt,
            ts.vbt.reduce_to_array(min_and_max_nb, index=['min', 'max'], time_units=True)
        )

    @pytest.mark.parametrize(
        "test_func,test_func_nb",
        [
            (lambda x, **kwargs: x.min(**kwargs), nb.nanmin_nb),
            (lambda x, **kwargs: x.max(**kwargs), nb.nanmax_nb),
            (lambda x, **kwargs: x.mean(**kwargs), nb.nanmean_nb),
            (lambda x, **kwargs: x.median(**kwargs), nb.nanmedian_nb),
            (lambda x, **kwargs: x.std(**kwargs), nb.nanstd_nb),
            (lambda x, **kwargs: x.count(**kwargs), nb.nancnt_nb),
            (lambda x, **kwargs: x.sum(**kwargs), nb.nansum_nb)
        ],
    )
    def test_funcs(self, test_func, test_func_nb):
        # numeric
        assert test_func(ts['a']) == test_func(ts['a'].vbt)
        pd.testing.assert_series_equal(
            test_func(ts),
            test_func(ts.vbt)
        )
        np.testing.assert_array_equal(test_func(ts).values, test_func_nb(ts.values))
        pd.testing.assert_series_equal(
            test_func(ts) * day_dt,
            test_func(ts.vbt, time_units=True)
        )
        # boolean
        bool_ts = ts == ts
        assert test_func(bool_ts['a']) == test_func(bool_ts['a'].vbt)
        pd.testing.assert_series_equal(
            test_func(bool_ts),
            test_func(bool_ts.vbt)
        )
        pd.testing.assert_series_equal(
            test_func(bool_ts) * day_dt,
            test_func(bool_ts.vbt, time_units=True)
        )

    @pytest.mark.parametrize(
        "test_func",
        [
            lambda x, **kwargs: x.idxmin(**kwargs),
            lambda x, **kwargs: x.idxmax(**kwargs)
        ],
    )
    def test_arg_funcs(self, test_func):
        assert test_func(ts['a']) == test_func(ts['a'].vbt)
        pd.testing.assert_series_equal(
            test_func(ts),
            test_func(ts.vbt)
        )

    def test_describe(self):
        pd.testing.assert_series_equal(
            ts['a'].describe(),
            ts['a'].vbt.describe()
        )
        pd.testing.assert_frame_equal(
            ts.describe(percentiles=None),
            ts.vbt.describe(percentiles=None)
        )
        pd.testing.assert_frame_equal(
            ts.describe(percentiles=[]),
            ts.vbt.describe(percentiles=[])
        )
        pd.testing.assert_frame_equal(
            ts.describe(percentiles=np.arange(0, 1, 0.1)),
            ts.vbt.describe(percentiles=np.arange(0, 1, 0.1))
        )

    def test_drawdown(self):
        pd.testing.assert_series_equal(
            ts['a'] / ts['a'].expanding().max() - 1,
            ts['a'].vbt.drawdown()
        )
        pd.testing.assert_frame_equal(
            ts / ts.expanding().max() - 1,
            ts.vbt.drawdown()
        )

    def test_drawdowns(self):
        assert type(ts['a'].vbt.drawdowns) is Drawdowns
        assert ts['a'].vbt.drawdowns.wrapper.freq == ts['a'].vbt.freq
        assert ts['a'].vbt.drawdowns.wrapper.ndim == ts['a'].ndim
        assert ts.vbt.drawdowns.wrapper.ndim == ts.ndim
