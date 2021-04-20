import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
import pytest
from itertools import product
from sklearn.model_selection import TimeSeriesSplit

import vectorbt as vbt
from vectorbt.generic import nb

seed = 42

day_dt = np.timedelta64(86400000000000)

df = pd.DataFrame({
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
group_by = np.array(['g1', 'g1', 'g2'])


@njit
def i_or_col_pow_nb(i_or_col, x, pow):
    return np.power(x, pow)


@njit
def pow_nb(x, pow):
    return np.power(x, pow)


@njit
def nanmean_nb(x):
    return np.nanmean(x)


@njit
def i_col_nanmean_nb(i, col, x):
    return np.nanmean(x)


@njit
def i_nanmean_nb(i, x):
    return np.nanmean(x)


@njit
def col_nanmean_nb(col, x):
    return np.nanmean(x)


# ############# accessors.py ############# #


class TestAccessors:
    def test_shuffle(self):
        pd.testing.assert_series_equal(
            df['a'].vbt.shuffle(seed=seed),
            pd.Series(
                np.array([2.0, np.nan, 3.0, 1.0, 4.0]),
                index=df['a'].index,
                name=df['a'].name
            )
        )
        np.testing.assert_array_equal(
            df['a'].vbt.shuffle(seed=seed).values,
            nb.shuffle_1d_nb(df['a'].values, seed=seed)
        )
        pd.testing.assert_frame_equal(
            df.vbt.shuffle(seed=seed),
            pd.DataFrame(
                np.array([
                    [2., 2., 2.],
                    [np.nan, 4., 1.],
                    [3., 3., 2.],
                    [1., np.nan, 1.],
                    [4., 1., np.nan]
                ]),
                index=df.index,
                columns=df.columns
            )
        )

    @pytest.mark.parametrize(
        "test_value",
        [-1, 0., np.nan],
    )
    def test_fillna(self, test_value):
        pd.testing.assert_series_equal(df['a'].vbt.fillna(test_value), df['a'].fillna(test_value))
        pd.testing.assert_frame_equal(df.vbt.fillna(test_value), df.fillna(test_value))

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_bshift(self, test_n):
        pd.testing.assert_series_equal(df['a'].vbt.bshift(test_n), df['a'].shift(-test_n))
        np.testing.assert_array_equal(
            df['a'].vbt.bshift(test_n).values,
            nb.bshift_nb(df['a'].values, test_n)
        )
        pd.testing.assert_frame_equal(df.vbt.bshift(test_n), df.shift(-test_n))

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_fshift(self, test_n):
        pd.testing.assert_series_equal(df['a'].vbt.fshift(test_n), df['a'].shift(test_n))
        np.testing.assert_array_equal(
            df['a'].vbt.fshift(test_n).values,
            nb.fshift_1d_nb(df['a'].values, test_n)
        )
        pd.testing.assert_frame_equal(df.vbt.fshift(test_n), df.shift(test_n))

    def test_diff(self):
        pd.testing.assert_series_equal(df['a'].vbt.diff(), df['a'].diff())
        np.testing.assert_array_equal(df['a'].vbt.diff().values, nb.diff_1d_nb(df['a'].values))
        pd.testing.assert_frame_equal(df.vbt.diff(), df.diff())

    def test_pct_change(self):
        pd.testing.assert_series_equal(df['a'].vbt.pct_change(), df['a'].pct_change(fill_method=None))
        np.testing.assert_array_equal(df['a'].vbt.pct_change().values, nb.pct_change_1d_nb(df['a'].values))
        pd.testing.assert_frame_equal(df.vbt.pct_change(), df.pct_change(fill_method=None))

    def test_ffill(self):
        pd.testing.assert_series_equal(df['a'].vbt.ffill(), df['a'].ffill())
        pd.testing.assert_frame_equal(df.vbt.ffill(), df.ffill())

    def test_product(self):
        assert df['a'].vbt.product() == df['a'].product()
        np.testing.assert_array_equal(df.vbt.product(), df.product())

    def test_cumsum(self):
        pd.testing.assert_series_equal(df['a'].vbt.cumsum(), df['a'].cumsum())
        pd.testing.assert_frame_equal(df.vbt.cumsum(), df.cumsum())

    def test_cumprod(self):
        pd.testing.assert_series_equal(df['a'].vbt.cumprod(), df['a'].cumprod())
        pd.testing.assert_frame_equal(df.vbt.cumprod(), df.cumprod())

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_rolling_min(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df['a'].vbt.rolling_min(test_window, minp=test_minp),
            df['a'].rolling(test_window, min_periods=test_minp).min()
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_min(test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).min()
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_min(test_window),
            df.rolling(test_window).min()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_rolling_max(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df['a'].vbt.rolling_max(test_window, minp=test_minp),
            df['a'].rolling(test_window, min_periods=test_minp).max()
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_max(test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).max()
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_max(test_window),
            df.rolling(test_window).max()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_rolling_mean(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df['a'].vbt.rolling_mean(test_window, minp=test_minp),
            df['a'].rolling(test_window, min_periods=test_minp).mean()
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_mean(test_window, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).mean()
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_mean(test_window),
            df.rolling(test_window).mean()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp,test_ddof",
        list(product([1, 2, 3, 4, 5], [1, None], [0, 1]))
    )
    def test_rolling_std(self, test_window, test_minp, test_ddof):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df['a'].vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
            df['a'].rolling(test_window, min_periods=test_minp).std(ddof=test_ddof)
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
            df.rolling(test_window, min_periods=test_minp).std(ddof=test_ddof)
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_std(test_window),
            df.rolling(test_window).std()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp,test_adjust",
        list(product([1, 2, 3, 4, 5], [1, None], [False, True]))
    )
    def test_ewm_mean(self, test_window, test_minp, test_adjust):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df['a'].vbt.ewm_mean(test_window, minp=test_minp, adjust=test_adjust),
            df['a'].ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).mean()
        )
        pd.testing.assert_frame_equal(
            df.vbt.ewm_mean(test_window, minp=test_minp, adjust=test_adjust),
            df.ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).mean()
        )
        pd.testing.assert_frame_equal(
            df.vbt.ewm_mean(test_window),
            df.ewm(span=test_window).mean()
        )

    @pytest.mark.parametrize(
        "test_window,test_minp,test_adjust,test_ddof",
        list(product([1, 2, 3, 4, 5], [1, None], [False, True], [0, 1]))
    )
    def test_ewm_std(self, test_window, test_minp, test_adjust, test_ddof):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df['a'].vbt.ewm_std(test_window, minp=test_minp, adjust=test_adjust, ddof=test_ddof),
            df['a'].ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).std(ddof=test_ddof)
        )
        pd.testing.assert_frame_equal(
            df.vbt.ewm_std(test_window, minp=test_minp, adjust=test_adjust, ddof=test_ddof),
            df.ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).std(ddof=test_ddof)
        )
        pd.testing.assert_frame_equal(
            df.vbt.ewm_std(test_window),
            df.ewm(span=test_window).std()
        )

    @pytest.mark.parametrize(
        "test_minp",
        [1, 3]
    )
    def test_expanding_min(self, test_minp):
        pd.testing.assert_series_equal(
            df['a'].vbt.expanding_min(minp=test_minp),
            df['a'].expanding(min_periods=test_minp).min()
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_min(minp=test_minp),
            df.expanding(min_periods=test_minp).min()
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_min(),
            df.expanding().min()
        )

    @pytest.mark.parametrize(
        "test_minp",
        [1, 3]
    )
    def test_expanding_max(self, test_minp):
        pd.testing.assert_series_equal(
            df['a'].vbt.expanding_max(minp=test_minp),
            df['a'].expanding(min_periods=test_minp).max()
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_max(minp=test_minp),
            df.expanding(min_periods=test_minp).max()
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_max(),
            df.expanding().max()
        )

    @pytest.mark.parametrize(
        "test_minp",
        [1, 3]
    )
    def test_expanding_mean(self, test_minp):
        pd.testing.assert_series_equal(
            df['a'].vbt.expanding_mean(minp=test_minp),
            df['a'].expanding(min_periods=test_minp).mean()
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_mean(minp=test_minp),
            df.expanding(min_periods=test_minp).mean()
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_mean(),
            df.expanding().mean()
        )

    @pytest.mark.parametrize(
        "test_minp,test_ddof",
        list(product([1, 3], [0, 1]))
    )
    def test_expanding_std(self, test_minp, test_ddof):
        pd.testing.assert_series_equal(
            df['a'].vbt.expanding_std(minp=test_minp, ddof=test_ddof),
            df['a'].expanding(min_periods=test_minp).std(ddof=test_ddof)
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_std(minp=test_minp, ddof=test_ddof),
            df.expanding(min_periods=test_minp).std(ddof=test_ddof)
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_std(),
            df.expanding().std()
        )

    def test_apply_along_axis(self):
        pd.testing.assert_frame_equal(
            df.vbt.apply_along_axis(i_or_col_pow_nb, 2, axis=0),
            df.apply(pow_nb, args=(2,), axis=0, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.apply_along_axis(i_or_col_pow_nb, 2, axis=1),
            df.apply(pow_nb, args=(2,), axis=1, raw=True)
        )

    @pytest.mark.parametrize(
        "test_window,test_minp",
        list(product([1, 2, 3, 4, 5], [1, None]))
    )
    def test_rolling_apply(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df['a'].vbt.rolling_apply(test_window, i_col_nanmean_nb, minp=test_minp),
            df['a'].rolling(test_window, min_periods=test_minp).apply(nanmean_nb, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_apply(test_window, i_col_nanmean_nb, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).apply(nanmean_nb, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_apply(test_window, i_col_nanmean_nb),
            df.rolling(test_window).apply(nanmean_nb, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_apply(3, i_nanmean_nb, on_matrix=True),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan],
                    [2.75, 2.75, 2.75],
                    [np.nan, np.nan, np.nan]
                ]),
                index=df.index,
                columns=df.columns
            )
        )

    @pytest.mark.parametrize(
        "test_minp",
        [1, 3]
    )
    def test_expanding_apply(self, test_minp):
        pd.testing.assert_series_equal(
            df['a'].vbt.expanding_apply(i_col_nanmean_nb, minp=test_minp),
            df['a'].expanding(min_periods=test_minp).apply(nanmean_nb, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_apply(i_col_nanmean_nb, minp=test_minp),
            df.expanding(min_periods=test_minp).apply(nanmean_nb, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_apply(i_col_nanmean_nb),
            df.expanding().apply(nanmean_nb, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_apply(i_nanmean_nb, on_matrix=True),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan],
                    [2.0, 2.0, 2.0],
                    [2.2857142857142856, 2.2857142857142856, 2.2857142857142856],
                    [2.4, 2.4, 2.4],
                    [2.1666666666666665, 2.1666666666666665, 2.1666666666666665]
                ]),
                index=df.index,
                columns=df.columns
            )
        )

    def test_groupby_apply(self):
        pd.testing.assert_series_equal(
            df['a'].vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), i_col_nanmean_nb),
            df['a'].groupby(np.asarray([1, 1, 2, 2, 3])).apply(lambda x: nanmean_nb(x.values))
        )
        pd.testing.assert_frame_equal(
            df.vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), i_col_nanmean_nb),
            df.groupby(np.asarray([1, 1, 2, 2, 3])).agg({
                'a': lambda x: nanmean_nb(x.values),
                'b': lambda x: nanmean_nb(x.values),
                'c': lambda x: nanmean_nb(x.values)
            }),  # any clean way to do column-wise grouping in pandas?
        )

    def test_groupby_apply_on_matrix(self):
        pd.testing.assert_frame_equal(
            df.vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), i_nanmean_nb, on_matrix=True),
            pd.DataFrame(
                np.array([
                    [2., 2., 2.],
                    [2.8, 2.8, 2.8],
                    [1., 1., 1.]
                ]),
                index=pd.Int64Index([1, 2, 3], dtype='int64'),
                columns=df.columns
            )
        )

    @pytest.mark.parametrize(
        "test_freq",
        ['1h', '3d', '1w'],
    )
    def test_resample_apply(self, test_freq):
        pd.testing.assert_series_equal(
            df['a'].vbt.resample_apply(test_freq, i_col_nanmean_nb),
            df['a'].resample(test_freq).apply(lambda x: nanmean_nb(x.values))
        )
        pd.testing.assert_frame_equal(
            df.vbt.resample_apply(test_freq, i_col_nanmean_nb),
            df.resample(test_freq).apply(lambda x: nanmean_nb(x.values))
        )
        pd.testing.assert_frame_equal(
            df.vbt.resample_apply('3d', i_nanmean_nb, on_matrix=True),
            pd.DataFrame(
                np.array([
                    [2.28571429, 2.28571429, 2.28571429],
                    [2., 2., 2.]
                ]),
                index=pd.DatetimeIndex(['2018-01-01', '2018-01-04'], dtype='datetime64[ns]', freq='3D'),
                columns=df.columns
            )
        )

    def test_applymap(self):
        @njit
        def mult_nb(i, col, x):
            return x * 2

        pd.testing.assert_series_equal(
            df['a'].vbt.applymap(mult_nb),
            df['a'].map(lambda x: x * 2)
        )
        pd.testing.assert_frame_equal(
            df.vbt.applymap(mult_nb),
            df.applymap(lambda x: x * 2)
        )

    def test_filter(self):
        @njit
        def greater_nb(i, col, x):
            return x > 2

        pd.testing.assert_series_equal(
            df['a'].vbt.filter(greater_nb),
            df['a'].map(lambda x: x if x > 2 else np.nan)
        )
        pd.testing.assert_frame_equal(
            df.vbt.filter(greater_nb),
            df.applymap(lambda x: x if x > 2 else np.nan)
        )

    def test_apply_and_reduce(self):
        @njit
        def every_nth_nb(col, a, n):
            return a[::n]

        @njit
        def sum_nb(col, a, b):
            return np.nansum(a) + b

        assert df['a'].vbt.apply_and_reduce(every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,)) == \
               df['a'].iloc[::2].sum() + 3
        pd.testing.assert_series_equal(
            df.vbt.apply_and_reduce(every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,)),
            df.iloc[::2].sum().rename('apply_and_reduce') + 3
        )
        pd.testing.assert_series_equal(
            df.vbt.apply_and_reduce(
                every_nth_nb, sum_nb, apply_args=(2,),
                reduce_args=(3,), wrap_kwargs=dict(time_units=True)),
            (df.iloc[::2].sum().rename('apply_and_reduce') + 3) * day_dt
        )

    def test_reduce(self):
        @njit
        def sum_nb(col, a):
            return np.nansum(a)

        assert df['a'].vbt.reduce(sum_nb) == df['a'].sum()
        pd.testing.assert_series_equal(
            df.vbt.reduce(sum_nb),
            df.sum().rename('reduce')
        )
        pd.testing.assert_series_equal(
            df.vbt.reduce(sum_nb, wrap_kwargs=dict(time_units=True)),
            df.sum().rename('reduce') * day_dt
        )
        pd.testing.assert_series_equal(
            df.vbt.reduce(sum_nb, group_by=group_by),
            pd.Series([20.0, 6.0], index=['g1', 'g2']).rename('reduce')
        )

        @njit
        def argmax_nb(col, a):
            a = a.copy()
            a[np.isnan(a)] = -np.inf
            return np.argmax(a)

        assert df['a'].vbt.reduce(argmax_nb, to_idx=True) == df['a'].idxmax()
        pd.testing.assert_series_equal(
            df.vbt.reduce(argmax_nb, to_idx=True),
            df.idxmax().rename('reduce')
        )
        pd.testing.assert_series_equal(
            df.vbt.reduce(argmax_nb, to_idx=True, flatten=True, group_by=group_by),
            pd.Series(['2018-01-02', '2018-01-02'], dtype='datetime64[ns]', index=['g1', 'g2']).rename('reduce')
        )

        @njit
        def min_and_max_nb(col, a):
            out = np.empty(2)
            out[0] = np.nanmin(a)
            out[1] = np.nanmax(a)
            return out

        pd.testing.assert_series_equal(
            df['a'].vbt.reduce(
                min_and_max_nb, to_array=True,
                wrap_kwargs=dict(name_or_index=['min', 'max'])),
            pd.Series([np.nanmin(df['a']), np.nanmax(df['a'])], index=['min', 'max'], name='a')
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(
                min_and_max_nb, to_array=True,
                wrap_kwargs=dict(name_or_index=['min', 'max'])),
            df.apply(lambda x: pd.Series(np.asarray([np.nanmin(x), np.nanmax(x)]), index=['min', 'max']), axis=0)
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(
                min_and_max_nb, to_array=True, group_by=group_by,
                wrap_kwargs=dict(name_or_index=['min', 'max'])),
            pd.DataFrame([[1.0, 1.0], [4.0, 2.0]], index=['min', 'max'], columns=['g1', 'g2'])
        )

        @njit
        def argmin_and_argmax_nb(col, a):
            # nanargmin and nanargmax
            out = np.empty(2)
            _a = a.copy()
            _a[np.isnan(_a)] = np.inf
            out[0] = np.argmin(_a)
            _a = a.copy()
            _a[np.isnan(_a)] = -np.inf
            out[1] = np.argmax(_a)
            return out

        pd.testing.assert_series_equal(
            df['a'].vbt.reduce(
                argmin_and_argmax_nb, to_idx=True, to_array=True,
                wrap_kwargs=dict(name_or_index=['idxmin', 'idxmax'])),
            pd.Series([df['a'].idxmin(), df['a'].idxmax()], index=['idxmin', 'idxmax'], name='a')
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(
                argmin_and_argmax_nb, to_idx=True, to_array=True,
                wrap_kwargs=dict(name_or_index=['idxmin', 'idxmax'])),
            df.apply(lambda x: pd.Series(np.asarray([x.idxmin(), x.idxmax()]), index=['idxmin', 'idxmax']), axis=0)
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(argmin_and_argmax_nb, to_idx=True, to_array=True,
                          flatten=True, order='C', group_by=group_by,
                          wrap_kwargs=dict(name_or_index=['idxmin', 'idxmax'])),
            pd.DataFrame([['2018-01-01', '2018-01-01'], ['2018-01-02', '2018-01-02']],
                         dtype='datetime64[ns]', index=['idxmin', 'idxmax'], columns=['g1', 'g2'])
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(argmin_and_argmax_nb, to_idx=True, to_array=True,
                          flatten=True, order='F', group_by=group_by,
                          wrap_kwargs=dict(name_or_index=['idxmin', 'idxmax'])),
            pd.DataFrame([['2018-01-01', '2018-01-01'], ['2018-01-04', '2018-01-02']],
                         dtype='datetime64[ns]', index=['idxmin', 'idxmax'], columns=['g1', 'g2'])
        )

    def test_squeeze_grouped(self):
        pd.testing.assert_frame_equal(
            df.vbt.squeeze_grouped(i_col_nanmean_nb, group_by=group_by),
            pd.DataFrame([
                [1.0, 1.0],
                [3.0, 2.0],
                [3.0, np.nan],
                [3.0, 2.0],
                [1.0, 1.0]
            ], index=df.index, columns=['g1', 'g2'])
        )

    def test_flatten_grouped(self):
        pd.testing.assert_frame_equal(
            df.vbt.flatten_grouped(group_by=group_by, order='C'),
            pd.DataFrame([
                [1.0, 1.0],
                [np.nan, np.nan],
                [2.0, 2.0],
                [4.0, np.nan],
                [3.0, np.nan],
                [3.0, np.nan],
                [4.0, 2.0],
                [2.0, np.nan],
                [np.nan, 1.0],
                [1.0, np.nan]
            ], index=np.repeat(df.index, 2), columns=['g1', 'g2'])
        )
        pd.testing.assert_frame_equal(
            df.vbt.flatten_grouped(group_by=group_by, order='F'),
            pd.DataFrame([
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, np.nan],
                [4.0, 2.0],
                [np.nan, 1.0],
                [np.nan, np.nan],
                [4.0, np.nan],
                [3.0, np.nan],
                [2.0, np.nan],
                [1.0, np.nan]
            ], index=np.tile(df.index, 2), columns=['g1', 'g2'])
        )

    @pytest.mark.parametrize(
        "test_name,test_func,test_func_nb",
        [
            ('min', lambda x, **kwargs: x.min(**kwargs), nb.nanmin_nb),
            ('max', lambda x, **kwargs: x.max(**kwargs), nb.nanmax_nb),
            ('mean', lambda x, **kwargs: x.mean(**kwargs), nb.nanmean_nb),
            ('median', lambda x, **kwargs: x.median(**kwargs), nb.nanmedian_nb),
            ('std', lambda x, **kwargs: x.std(**kwargs, ddof=0), nb.nanstd_nb),
            ('count', lambda x, **kwargs: x.count(**kwargs), nb.nancnt_nb),
            ('sum', lambda x, **kwargs: x.sum(**kwargs), nb.nansum_nb)
        ],
    )
    def test_funcs(self, test_name, test_func, test_func_nb):
        # numeric
        assert test_func(df['a'].vbt) == test_func(df['a'])
        pd.testing.assert_series_equal(
            test_func(df.vbt),
            test_func(df).rename(test_name)
        )
        pd.testing.assert_series_equal(
            test_func(df.vbt, group_by=group_by),
            pd.Series([
                test_func(df[['a', 'b']].stack()),
                test_func(df['c'])
            ], index=['g1', 'g2']).rename(test_name)
        )
        np.testing.assert_array_equal(test_func(df).values, test_func_nb(df.values))
        pd.testing.assert_series_equal(
            test_func(df.vbt, wrap_kwargs=dict(time_units=True)),
            test_func(df).rename(test_name) * day_dt
        )
        # boolean
        bool_ts = df == df
        assert test_func(bool_ts['a'].vbt) == test_func(bool_ts['a'])
        pd.testing.assert_series_equal(
            test_func(bool_ts.vbt),
            test_func(bool_ts).rename(test_name)
        )
        pd.testing.assert_series_equal(
            test_func(bool_ts.vbt, wrap_kwargs=dict(time_units=True)),
            test_func(bool_ts).rename(test_name) * day_dt
        )

    @pytest.mark.parametrize(
        "test_name,test_func",
        [
            ('idxmin', lambda x, **kwargs: x.idxmin(**kwargs)),
            ('idxmax', lambda x, **kwargs: x.idxmax(**kwargs))
        ],
    )
    def test_arg_funcs(self, test_name, test_func):
        assert test_func(df['a'].vbt) == test_func(df['a'])
        pd.testing.assert_series_equal(
            test_func(df.vbt),
            test_func(df).rename(test_name)
        )
        pd.testing.assert_series_equal(
            test_func(df.vbt, group_by=group_by),
            pd.Series([
                test_func(df[['a', 'b']].stack())[0],
                test_func(df['c'])
            ], index=['g1', 'g2'], dtype='datetime64[ns]').rename(test_name)
        )

    def test_describe(self):
        pd.testing.assert_series_equal(
            df['a'].vbt.describe(),
            df['a'].describe()
        )
        pd.testing.assert_frame_equal(
            df.vbt.describe(percentiles=None),
            df.describe(percentiles=None)
        )
        pd.testing.assert_frame_equal(
            df.vbt.describe(percentiles=[]),
            df.describe(percentiles=[])
        )
        test_against = df.describe(percentiles=np.arange(0, 1, 0.1))
        pd.testing.assert_frame_equal(
            df.vbt.describe(percentiles=np.arange(0, 1, 0.1)),
            test_against
        )
        pd.testing.assert_frame_equal(
            df.vbt.describe(percentiles=np.arange(0, 1, 0.1), group_by=group_by),
            pd.DataFrame({
                'g1': df[['a', 'b']].stack().describe(percentiles=np.arange(0, 1, 0.1)).values,
                'g2': df['c'].describe(percentiles=np.arange(0, 1, 0.1)).values
            }, index=test_against.index)
        )

    def test_drawdown(self):
        pd.testing.assert_series_equal(
            df['a'].vbt.drawdown(),
            df['a'] / df['a'].expanding().max() - 1
        )
        pd.testing.assert_frame_equal(
            df.vbt.drawdown(),
            df / df.expanding().max() - 1
        )

    def test_drawdowns(self):
        assert type(df['a'].vbt.drawdowns) is vbt.Drawdowns
        assert df['a'].vbt.drawdowns.wrapper.freq == df['a'].vbt.wrapper.freq
        assert df['a'].vbt.drawdowns.wrapper.ndim == df['a'].ndim
        assert df.vbt.drawdowns.wrapper.ndim == df.ndim

    def test_to_mapped_array(self):
        np.testing.assert_array_equal(
            df.vbt.to_mapped_array().values,
            np.array([1., 2., 3., 4., 4., 3., 2., 1., 1., 2., 2., 1.])
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped_array().col_arr,
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped_array().idx_arr,
            np.array([0, 1, 2, 3, 1, 2, 3, 4, 0, 1, 3, 4])
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped_array(dropna=False).values,
            np.array([1., 2., 3., 4., np.nan, np.nan, 4., 3., 2., 1., 1., 2., np.nan, 2., 1.])
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped_array(dropna=False).col_arr,
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped_array(dropna=False).idx_arr,
            np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        )

    def test_zscore(self):
        pd.testing.assert_series_equal(
            df['a'].vbt.zscore(),
            (df['a'] - df['a'].mean()) / df['a'].std(ddof=0)
        )
        pd.testing.assert_frame_equal(
            df.vbt.zscore(),
            (df - df.mean()) / df.std(ddof=0)
        )

    def test_split(self):
        splitter = TimeSeriesSplit(n_splits=2)
        (train_df, train_indexes), (test_df, test_indexes) = df['a'].vbt.split(splitter)
        pd.testing.assert_frame_equal(
            train_df,
            pd.DataFrame(
                np.array([
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, 3.0],
                    [np.nan, 4.0]
                ]),
                index=pd.RangeIndex(start=0, stop=4, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                             dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'],
                             dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                train_indexes[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            test_df,
            pd.DataFrame(
                np.array([
                    [4.0, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-04'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                test_indexes[i],
                target[i]
            )
        (train_df, train_indexes), (test_df, test_indexes) = df.vbt.split(splitter)
        pd.testing.assert_frame_equal(
            train_df,
            pd.DataFrame(
                np.array([
                    [1.0, np.nan, 1.0, 1.0, np.nan, 1.0],
                    [2.0, 4.0, 2.0, 2.0, 4.0, 2.0],
                    [3.0, 3.0, np.nan, 3.0, 3.0, np.nan],
                    [np.nan, np.nan, np.nan, 4.0, 2.0, 2.0]
                ]),
                index=pd.RangeIndex(start=0, stop=4, step=1),
                columns=pd.MultiIndex.from_tuples([
                    (0, 'a'),
                    (0, 'b'),
                    (0, 'c'),
                    (1, 'a'),
                    (1, 'b'),
                    (1, 'c')
                ], names=['split_idx', None])
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                             dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'],
                             dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                train_indexes[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            test_df,
            pd.DataFrame(
                np.array([
                    [4.0, 2.0, 2.0, np.nan, 1.0, 1.0]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.MultiIndex.from_tuples([
                    (0, 'a'),
                    (0, 'b'),
                    (0, 'c'),
                    (1, 'a'),
                    (1, 'b'),
                    (1, 'c')
                ], names=['split_idx', None])
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-04'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                test_indexes[i],
                target[i]
            )

    def test_range_split(self):
        pd.testing.assert_frame_equal(
            df['a'].vbt.range_split(n=2)[0],
            pd.DataFrame(
                np.array([
                    [1., 4.],
                    [2., np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-04', '2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df['a'].vbt.range_split(n=2)[1][i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df['a'].vbt.range_split(range_len=2)[0],
            pd.DataFrame(
                np.array([
                    [1., 2., 3., 4.],
                    [2., 3., 4., np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1, 2, 3], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-02', '2018-01-03'], dtype='datetime64[ns]', name='split_1', freq=None),
            pd.DatetimeIndex(['2018-01-03', '2018-01-04'], dtype='datetime64[ns]', name='split_2', freq=None),
            pd.DatetimeIndex(['2018-01-04', '2018-01-05'], dtype='datetime64[ns]', name='split_3', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df['a'].vbt.range_split(range_len=2)[1][i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df['a'].vbt.range_split(range_len=2, n=3)[0],
            pd.DataFrame(
                np.array([
                    [1., 3., 4.],
                    [2., 4., np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1, 2], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-03', '2018-01-04'], dtype='datetime64[ns]', name='split_1', freq=None),
            pd.DatetimeIndex(['2018-01-04', '2018-01-05'], dtype='datetime64[ns]', name='split_2', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df['a'].vbt.range_split(range_len=2, n=3)[1][i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df['a'].vbt.range_split(range_len=3, n=2)[0],
            pd.DataFrame(
                np.array([
                    [1., 3.],
                    [2., 4.],
                    [3., np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                             dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-03', '2018-01-04', '2018-01-05'],
                             dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df['a'].vbt.range_split(range_len=3, n=2)[1][i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df.vbt.range_split(n=2)[0],
            pd.DataFrame(
                np.array([
                    [1.0, np.nan, 1.0, 4.0, 2.0, 2.0],
                    [2.0, 4.0, 2.0, np.nan, 1.0, 1.0]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.Int64Index([0, 0, 0, 1, 1, 1], dtype='int64', name='split_idx'),
                    pd.Index(['a', 'b', 'c', 'a', 'b', 'c'], dtype='object')
                ])
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-04', '2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df.vbt.range_split(n=2)[1][i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df.vbt.range_split(start_idxs=[0, 1], end_idxs=[2, 3])[0],
            pd.DataFrame(
                np.array([
                    [1.0, np.nan, 1.0, 2.0, 4.0, 2.0],
                    [2.0, 4.0, 2.0, 3.0, 3.0, np.nan],
                    [3.0, 3.0, np.nan, 4.0, 2.0, 2.0]
                ]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.Int64Index([0, 0, 0, 1, 1, 1], dtype='int64', name='split_idx'),
                    pd.Index(['a', 'b', 'c', 'a', 'b', 'c'], dtype='object')
                ])
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                             dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-02', '2018-01-03', '2018-01-04'],
                             dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df.vbt.range_split(start_idxs=[0, 1], end_idxs=[2, 3])[1][i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df.vbt.range_split(start_idxs=df.index[[0, 1]], end_idxs=df.index[[2, 3]])[0],
            pd.DataFrame(
                np.array([
                    [1.0, np.nan, 1.0, 2.0, 4.0, 2.0],
                    [2.0, 4.0, 2.0, 3.0, 3.0, np.nan],
                    [3.0, 3.0, np.nan, 4.0, 2.0, 2.0]
                ]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.Int64Index([0, 0, 0, 1, 1, 1], dtype='int64', name='split_idx'),
                    pd.Index(['a', 'b', 'c', 'a', 'b', 'c'], dtype='object')
                ])
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                             dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-02', '2018-01-03', '2018-01-04'],
                             dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df.vbt.range_split(start_idxs=df.index[[0, 1]], end_idxs=df.index[[2, 3]])[1][i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df.vbt.range_split(start_idxs=df.index[[0]], end_idxs=df.index[[2, 3]])[0],
            pd.DataFrame(
                np.array([
                    [1.0, np.nan, 1.0, 1.0, np.nan, 1.0],
                    [2.0, 4.0, 2.0, 2.0, 4.0, 2.0],
                    [3.0, 3.0, np.nan, 3.0, 3.0, np.nan],
                    [np.nan, np.nan, np.nan, 4.0, 2.0, 2.0]
                ]),
                index=pd.RangeIndex(start=0, stop=4, step=1),
                columns=pd.MultiIndex.from_arrays([
                    pd.Int64Index([0, 0, 0, 1, 1, 1], dtype='int64', name='split_idx'),
                    pd.Index(['a', 'b', 'c', 'a', 'b', 'c'], dtype='object')
                ])
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                             dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04'],
                             dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df.vbt.range_split(start_idxs=df.index[[0]], end_idxs=df.index[[2, 3]])[1][i],
                target[i]
            )
        with pytest.raises(Exception) as e_info:
            df.vbt.range_split()
        with pytest.raises(Exception) as e_info:
            df.vbt.range_split(start_idxs=[0, 1])
        with pytest.raises(Exception) as e_info:
            df.vbt.range_split(end_idxs=[2, 4])
        with pytest.raises(Exception) as e_info:
            df.vbt.range_split(min_len=10)
        with pytest.raises(Exception) as e_info:
            df.vbt.range_split(n=10)

    def test_rolling_split(self):
        (df1, indexes1), (df2, indexes2), (df3, indexes3) = df['a'].vbt.rolling_split(
            window_len=4, set_lens=(1, 1), left_to_right=False)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([
                    [1.0, 2.0],
                    [2.0, 3.0]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-02', '2018-01-03'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes1[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df2,
            pd.DataFrame(
                np.array([
                    [3.0, 4.0]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-03'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-04'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes2[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df3,
            pd.DataFrame(
                np.array([
                    [4.0, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-04'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes3[i],
                target[i]
            )
        (df1, indexes1), (df2, indexes2), (df3, indexes3) = df['a'].vbt.rolling_split(
            window_len=4, set_lens=(1, 1), left_to_right=True)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([
                    [1.0, 2.0]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-02'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes1[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df2,
            pd.DataFrame(
                np.array([
                    [2.0, 3.0]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-03'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes2[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df3,
            pd.DataFrame(
                np.array([
                    [3.0, 4.0],
                    [4.0, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-03', '2018-01-04'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-04', '2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes3[i],
                target[i]
            )
        (df1, indexes1), (df2, indexes2), (df3, indexes3) = df['a'].vbt.rolling_split(
            window_len=4, set_lens=(0.25, 0.25), left_to_right=[False, True])
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([
                    [1.0, 2.0],
                    [2.0, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-02'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes1[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df2,
            pd.DataFrame(
                np.array([
                    [3.0, 3.0]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-03'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-03'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes2[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df3,
            pd.DataFrame(
                np.array([
                    [4.0, 4.0],
                    [np.nan, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-04'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-04', '2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes3[i],
                target[i]
            )
        df1, indexes1 = df['a'].vbt.rolling_split(window_len=2, n=2)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([
                    [1.0, 4.0],
                    [2.0, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-04', '2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        df1, indexes1 = df['a'].vbt.rolling_split(window_len=0.4, n=2)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([
                    [1.0, 4.0],
                    [2.0, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-04', '2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes1[i],
                target[i]
            )
        with pytest.raises(Exception) as e_info:
            df.vbt.rolling_split()
        with pytest.raises(Exception) as e_info:
            df.vbt.rolling_split(window_len=3, set_lens=(3, 1))
        with pytest.raises(Exception) as e_info:
            df.vbt.rolling_split(window_len=1, set_lens=(1, 1))
        with pytest.raises(Exception) as e_info:
            df.vbt.rolling_split(n=2, min_len=10)
        with pytest.raises(Exception) as e_info:
            df.vbt.rolling_split(n=10)

    def test_expanding_split(self):
        (df1, indexes1), (df2, indexes2), (df3, indexes3) = df['a'].vbt.expanding_split(
            min_len=4, set_lens=(1, 1), left_to_right=False)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [np.nan, 3.0]
                ]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'],
                             dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03'],
                             dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes1[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df2,
            pd.DataFrame(
                np.array([
                    [3.0, 4.0]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-03'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-04'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes2[i],
                target[i]
            )
        pd.testing.assert_frame_equal(
            df3,
            pd.DataFrame(
                np.array([
                    [4.0, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-04'], dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-05'], dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes3[i],
                target[i]
            )
        df1, indexes1 = df['a'].vbt.expanding_split(n=2, min_len=2)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [np.nan, 3.0],
                    [np.nan, 4.0],
                    [np.nan, np.nan]
                ]),
                index=pd.RangeIndex(start=0, stop=5, step=1),
                columns=pd.Int64Index([0, 1], dtype='int64', name='split_idx')
            )
        )
        target = [
            pd.DatetimeIndex(['2018-01-01', '2018-01-02'],
                             dtype='datetime64[ns]', name='split_0', freq=None),
            pd.DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04', '2018-01-05'],
                             dtype='datetime64[ns]', name='split_1', freq=None)
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                indexes1[i],
                target[i]
            )
        with pytest.raises(Exception) as e_info:
            df.vbt.expanding_split(n=2, min_len=10)
        with pytest.raises(Exception) as e_info:
            df.vbt.expanding_split(n=10)
