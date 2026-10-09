from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import pytest
from numba import njit
from sklearn.model_selection import TimeSeriesSplit

import vectorbt as vbt
from vectorbt import _engine
from vectorbt.generic import dispatch, nb

seed = 42


def pandas_applymap(df: pd.DataFrame, func):
    """Pandas' native element-wise map, compatible with pandas 2.0+."""
    if hasattr(pd.DataFrame, "map"):
        return df.map(func)
    return df.applymap(func)


day_dt = np.timedelta64(86400000000000)

df = pd.DataFrame(
    {"a": [1, 2, 3, 4, np.nan], "b": [np.nan, 4, 3, 2, 1], "c": [1, 2, np.nan, 2, 1]},
    index=pd.DatetimeIndex(
        [datetime(2018, 1, 1), datetime(2018, 1, 2), datetime(2018, 1, 3), datetime(2018, 1, 4), datetime(2018, 1, 5)]
    ),
)
group_by = np.array(["g1", "g1", "g2"])


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


# ############# Global ############# #


def setup_module():
    vbt.settings.numba["check_func_suffix"] = True
    vbt.settings.caching.enabled = False
    vbt.settings.caching.whitelist = []
    vbt.settings.caching.blacklist = []


def teardown_module():
    vbt.settings.reset()


# ############# accessors.py ############# #


class TestAccessors:
    def test_indexing(self):
        assert df.vbt["a"].min() == df["a"].vbt.min()

    def test_set_by_mask(self):
        np.testing.assert_array_equal(
            nb.set_by_mask_1d_nb(np.array([1, 2, 3, 1, 2, 3]), np.array([True, False, False, True, False, False]), 0),
            np.array([0, 2, 3, 0, 2, 3]),
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_1d_nb(np.array([1, 2, 3, 1, 2, 3]), np.array([True, False, False, True, False, False]), 0.0),
            np.array([0.0, 2.0, 3.0, 0.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_nb(
                np.array([1, 2, 3, 1, 2, 3])[:, None], np.array([True, False, False, True, False, False])[:, None], 0
            ),
            np.array([0, 2, 3, 0, 2, 3])[:, None],
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_nb(
                np.array([1, 2, 3, 1, 2, 3])[:, None], np.array([True, False, False, True, False, False])[:, None], 0.0
            ),
            np.array([0.0, 2.0, 3.0, 0.0, 2.0, 3.0])[:, None],
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_mult_1d_nb(
                np.array([1, 2, 3, 1, 2, 3]),
                np.array([True, False, False, True, False, False]),
                np.array([0, -1, -1, 0, -1, -1]),
            ),
            np.array([0, 2, 3, 0, 2, 3]),
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_mult_1d_nb(
                np.array([1, 2, 3, 1, 2, 3]),
                np.array([True, False, False, True, False, False]),
                np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0]),
            ),
            np.array([0.0, 2.0, 3.0, 0.0, 2.0, 3.0]),
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_mult_nb(
                np.array([1, 2, 3, 1, 2, 3])[:, None],
                np.array([True, False, False, True, False, False])[:, None],
                np.array([0, -1, -1, 0, -1, -1])[:, None],
            ),
            np.array([0, 2, 3, 0, 2, 3])[:, None],
        )
        np.testing.assert_array_equal(
            nb.set_by_mask_mult_nb(
                np.array([1, 2, 3, 1, 2, 3])[:, None],
                np.array([True, False, False, True, False, False])[:, None],
                np.array([0.0, -1.0, -1.0, 0.0, -1.0, -1.0])[:, None],
            ),
            np.array([0.0, 2.0, 3.0, 0.0, 2.0, 3.0])[:, None],
        )

    def test_shuffle(self):
        pd.testing.assert_series_equal(
            df["a"].vbt.shuffle(seed=seed),
            pd.Series(np.array([2.0, np.nan, 3.0, 1.0, 4.0]), index=df["a"].index, name=df["a"].name),
        )
        np.testing.assert_array_equal(
            df["a"].vbt.shuffle(seed=seed).values, nb.shuffle_1d_nb(df["a"].values, seed=seed)
        )
        pd.testing.assert_frame_equal(
            df.vbt.shuffle(seed=seed),
            pd.DataFrame(
                np.array(
                    [[2.0, 2.0, 2.0], [np.nan, 4.0, 1.0], [3.0, 3.0, 2.0], [1.0, np.nan, 1.0], [4.0, 1.0, np.nan]]
                ),
                index=df.index,
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize(
        "test_value",
        [-1, 0.0, np.nan],
    )
    def test_fillna(self, test_value):
        pd.testing.assert_series_equal(df["a"].vbt.fillna(test_value), df["a"].fillna(test_value))
        pd.testing.assert_frame_equal(df.vbt.fillna(test_value), df.fillna(test_value))
        pd.testing.assert_series_equal(pd.Series([1, 2, 3]).vbt.fillna(-1), pd.Series([1, 2, 3]))
        pd.testing.assert_series_equal(
            pd.Series([False, True, False]).vbt.fillna(False), pd.Series([False, True, False])
        )

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_bshift(self, test_n):
        pd.testing.assert_series_equal(df["a"].vbt.bshift(test_n), df["a"].shift(-test_n))
        np.testing.assert_array_equal(df["a"].vbt.bshift(test_n).values, nb.bshift_1d_nb(df["a"].values, test_n))
        pd.testing.assert_frame_equal(df.vbt.bshift(test_n), df.shift(-test_n))
        pd.testing.assert_series_equal(pd.Series([1, 2, 3]).vbt.bshift(1, fill_value=-1), pd.Series([2, 3, -1]))
        pd.testing.assert_series_equal(
            pd.Series([True, True, True]).vbt.bshift(1, fill_value=False), pd.Series([True, True, False])
        )

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_fshift(self, test_n):
        pd.testing.assert_series_equal(df["a"].vbt.fshift(test_n), df["a"].shift(test_n))
        np.testing.assert_array_equal(df["a"].vbt.fshift(test_n).values, nb.fshift_1d_nb(df["a"].values, test_n))
        pd.testing.assert_frame_equal(df.vbt.fshift(test_n), df.shift(test_n))
        pd.testing.assert_series_equal(pd.Series([1, 2, 3]).vbt.fshift(1, fill_value=-1), pd.Series([-1, 1, 2]))
        pd.testing.assert_series_equal(
            pd.Series([True, True, True]).vbt.fshift(1, fill_value=False), pd.Series([False, True, True])
        )

    def test_diff(self):
        pd.testing.assert_series_equal(df["a"].vbt.diff(), df["a"].diff())
        np.testing.assert_array_equal(df["a"].vbt.diff().values, nb.diff_1d_nb(df["a"].values))
        pd.testing.assert_frame_equal(df.vbt.diff(), df.diff())

    def test_pct_change(self):
        pd.testing.assert_series_equal(df["a"].vbt.pct_change(), df["a"].pct_change(fill_method=None))
        np.testing.assert_array_equal(df["a"].vbt.pct_change().values, nb.pct_change_1d_nb(df["a"].values))
        pd.testing.assert_frame_equal(df.vbt.pct_change(), df.pct_change(fill_method=None))

    def test_bfill(self):
        pd.testing.assert_series_equal(df["b"].vbt.bfill(), df["b"].bfill())
        pd.testing.assert_frame_equal(df.vbt.bfill(), df.bfill())

    def test_ffill(self):
        pd.testing.assert_series_equal(df["a"].vbt.ffill(), df["a"].ffill())
        pd.testing.assert_frame_equal(df.vbt.ffill(), df.ffill())

    def test_product(self):
        assert df["a"].vbt.product() == df["a"].product()
        np.testing.assert_array_equal(df.vbt.product(), df.product())

    def test_cumsum(self):
        pd.testing.assert_series_equal(df["a"].vbt.cumsum(), df["a"].cumsum().ffill().fillna(0))
        pd.testing.assert_frame_equal(df.vbt.cumsum(), df.cumsum().ffill().fillna(0))

    def test_cumprod(self):
        pd.testing.assert_series_equal(df["a"].vbt.cumprod(), df["a"].cumprod().ffill().fillna(1))
        pd.testing.assert_frame_equal(df.vbt.cumprod(), df.cumprod().ffill().fillna(1))

    @pytest.mark.parametrize("test_window,test_minp", list(product([1, 2, 3, 4, 5], [1, None])))
    def test_rolling_min(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df["a"].vbt.rolling_min(test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).min(),
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_min(test_window, minp=test_minp), df.rolling(test_window, min_periods=test_minp).min()
        )
        pd.testing.assert_frame_equal(df.vbt.rolling_min(test_window), df.rolling(test_window).min())

    @pytest.mark.parametrize("test_window,test_minp", list(product([1, 2, 3, 4, 5], [1, None])))
    def test_rolling_max(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df["a"].vbt.rolling_max(test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).max(),
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_max(test_window, minp=test_minp), df.rolling(test_window, min_periods=test_minp).max()
        )
        pd.testing.assert_frame_equal(df.vbt.rolling_max(test_window), df.rolling(test_window).max())

    @pytest.mark.parametrize("test_window,test_minp", list(product([1, 2, 3, 4, 5], [1, None])))
    def test_rolling_mean(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df["a"].vbt.rolling_mean(test_window, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).mean(),
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_mean(test_window, minp=test_minp), df.rolling(test_window, min_periods=test_minp).mean()
        )
        pd.testing.assert_frame_equal(df.vbt.rolling_mean(test_window), df.rolling(test_window).mean())

    @pytest.mark.parametrize(
        "dtype,value",
        [
            (np.int8, 100),
            (np.int8, -100),
            (np.int16, 30000),
            (np.int16, -30000),
            (np.int32, 2**30),
            (np.int64, 2**62),
            (np.uint8, 200),
            (np.uint16, 60000),
            (np.uint32, 2**31),
            (np.uint64, 2**63),
        ],
    )
    def test_rolling_mean_integer_overflow(self, dtype, value):
        a = np.full(5, value, dtype=dtype)
        # A constant window has the same mean even when its prefix sum exceeds the input dtype.
        # The full-length window also catches overflow in the scalar before any prefix is evicted.
        for window in (2, len(a)):
            expected = np.full(a.shape, float(value))
            expected[: window - 1] = np.nan
            np.testing.assert_array_equal(nb.rolling_mean_1d_nb(a, window), expected)
            np.testing.assert_array_equal(nb.rolling_mean_nb(a[:, None], window), expected[:, None])

    @pytest.mark.parametrize("window", [1, 2])
    def test_rolling_mean_integer_precision_after_large_prefix(self, window):
        a = np.array([2**53, 1, 1, 1], dtype=np.int64)
        # Unbounded sums of each actual window avoid rounding a large, already retired prefix.
        # Once that prefix leaves the window, the mean of the remaining ones must still be one.
        expected = np.full(a.shape, np.nan)
        for i in range(window - 1, len(a)):
            expected[i] = sum(int(value) for value in a[i - window + 1 : i + 1]) / window
        np.testing.assert_array_equal(nb.rolling_mean_1d_nb(a, window), expected)
        np.testing.assert_array_equal(nb.rolling_mean_nb(a[:, None], window), expected[:, None])

    @pytest.mark.parametrize(
        "values", [[2**53 + 1, -(2**53)], [2**62 + 1, -(2**62)], [-(2**53) - 1, 2**53], [-(2**62) - 1, 2**62]]
    )
    def test_rolling_mean_integer_value_precision(self, values):
        a = np.array(values, dtype=np.int64)
        # Converting either large input to float before summing discards the unit difference.
        expected = np.array([np.nan, sum(values) / 2])
        np.testing.assert_array_equal(nb.rolling_mean_1d_nb(a, 2), expected)
        np.testing.assert_array_equal(nb.rolling_mean_nb(a[:, None], 2), expected[:, None])
        pd.testing.assert_series_equal(pd.Series(a).vbt.rolling_mean(2, engine="numba"), pd.Series(expected))

    @pytest.mark.parametrize(
        "dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.bool_]
    )
    def test_rolling_mean_integer_extrema(self, dtype):
        low, high = (0, 1) if dtype == np.bool_ else (int(np.iinfo(dtype).min), int(np.iinfo(dtype).max))
        a = np.array([high, high, low, low, 1, 0, high, 1, low, 0], dtype=dtype)
        base = np.column_stack((a, a[::-1]))
        strided = np.empty((2 * len(a), 4), dtype=dtype)
        strided[::2, ::2] = base
        # Extremal runs and sign changes force carries, borrows and cancellation as values leave.
        # Sum each actual slice with Python integers; pandas first casts these values to float.
        for window in (1, 2, 5, len(a) + 3):
            for minp in (1, None):
                expected = np.full(base.shape, np.nan)
                for i in range(len(a)):
                    sample = base[max(0, i - window + 1) : i + 1]
                    if len(sample) >= (window if minp is None else minp):
                        for col in range(base.shape[1]):
                            expected[i, col] = sum(int(value) for value in sample[:, col]) / len(sample)
                # Float64 conversion/division can round; zero means must still be exactly zero.
                np.testing.assert_allclose(nb.rolling_mean_1d_nb(a, window, minp), expected[:, 0], rtol=3e-16, atol=0)
                for arr in (np.ascontiguousarray(base), np.asfortranarray(base), strided[::2, ::2]):
                    np.testing.assert_allclose(nb.rolling_mean_nb(arr, window, minp), expected, rtol=3e-16, atol=0)

    @pytest.mark.parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.bool_,
            np.float32,
            np.float64,
        ],
    )
    def test_rolling_mean_boundaries(self, dtype):
        a = np.array([1, 1, 0, 1], dtype=dtype)
        # Dtype specialization must preserve validation and the existing nonpositive-window path.
        for func, arr in ((nb.rolling_mean_1d_nb, a), (nb.rolling_mean_nb, a[:, None])):
            for window, minp in ((0, 1), (-1, 1), (2, 3)):
                with pytest.raises(ValueError, match="minp must be <= window"):
                    func(arr, window, minp)
            with pytest.raises(ZeroDivisionError):
                func(arr, 0)
            np.testing.assert_array_equal(func(arr, 8), np.full(arr.shape, np.nan))
            expected = np.array([1.0, 1.0, 2 / 3, 3 / 4]).reshape(arr.shape)
            np.testing.assert_array_equal(func(arr, 2**60, 1), expected)
        # No observations or no columns must retain their shape and float64 output dtype.
        for func, arr in (
            (nb.rolling_mean_1d_nb, np.empty(0, dtype=dtype)),
            (nb.rolling_mean_nb, np.empty((0, 2), dtype=dtype)),
            (nb.rolling_mean_nb, np.empty((4, 0), dtype=dtype)),
        ):
            out = func(arr, 2)
            assert out.shape == arr.shape
            assert out.dtype == np.float64

    def test_rolling_mean_integer_eviction(self):
        a = np.array([100, 100, -100, -100, 100, 100], dtype=np.int8)
        # Adjacent pairs give 100, 0, -100, 0, 100; the two columns must remain independent.
        expected = np.array([np.nan, 100.0, 0.0, -100.0, 0.0, 100.0])
        frame = pd.DataFrame(
            np.column_stack((a, -a)),
            index=pd.date_range("2020-01-01", periods=len(a), tz="UTC", name="time"),
            columns=pd.MultiIndex.from_tuples([("price", "a"), ("price", "b")], names=["kind", "asset"]),
        )
        expected_frame = pd.DataFrame(np.column_stack((expected, -expected)), index=frame.index, columns=frame.columns)
        np.testing.assert_array_equal(nb.rolling_mean_1d_nb(a, 2), expected)
        # Both pandas containers must retain their labels when wrapping the corrected kernel result.
        pd.testing.assert_frame_equal(frame.vbt.rolling_mean(2, engine="numba"), expected_frame)
        pd.testing.assert_series_equal(frame.iloc[:, 0].vbt.rolling_mean(2, engine="numba"), expected_frame.iloc[:, 0])

    def test_rolling_mean_boolean_prefix(self):
        # Boolean prefixes must retain counts larger than one before window subtraction.
        a = np.array([True, True, True, False])
        np.testing.assert_array_equal(nb.rolling_mean_1d_nb(a, 2), np.array([np.nan, 1.0, 1.0, 0.5]))

    def test_rolling_mean_integer_downstream(self):
        close = pd.Series(np.full(5, 100, dtype=np.int8))
        # MA uses chronological windows, while FMEAN reverses them and shifts by one bar.
        np.testing.assert_array_equal(
            vbt.MA.run(close, 2, engine="numba").ma.to_numpy(), np.array([np.nan, 100.0, 100.0, 100.0, 100.0])
        )
        np.testing.assert_array_equal(
            vbt.FMEAN.run(close, 2, wait=1, engine="numba").fmean.to_numpy(),
            np.array([100.0, 100.0, 100.0, np.nan, np.nan]),
        )

    @pytest.mark.parametrize("test_window,test_minp,test_ddof", list(product([1, 2, 3, 4, 5], [1, None], [0, 1])))
    def test_rolling_std(self, test_window, test_minp, test_ddof):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df["a"].vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
            df["a"].rolling(test_window, min_periods=test_minp).std(ddof=test_ddof),
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_std(test_window, minp=test_minp, ddof=test_ddof),
            df.rolling(test_window, min_periods=test_minp).std(ddof=test_ddof),
        )
        pd.testing.assert_frame_equal(df.vbt.rolling_std(test_window), df.rolling(test_window).std())

    @pytest.mark.parametrize("test_ddof", [0, 1])
    def test_rolling_std_numerical_stability(self, test_ddof):
        rng = np.random.default_rng(42)
        window = 4000
        large_offset_series = pd.Series(1e8 + rng.normal(0, 0.01, 5000))
        got = large_offset_series.vbt.rolling_std(window, minp=window, ddof=test_ddof)
        expected = large_offset_series.rolling(window, min_periods=window).std(ddof=test_ddof)
        pd.testing.assert_series_equal(got, expected, atol=1e-6, rtol=0)

        large_offset_nan_series = large_offset_series.copy()
        large_offset_nan_series.iloc[::37] = np.nan
        got_nan = large_offset_nan_series.vbt.rolling_std(window, minp=10, ddof=test_ddof)
        expected_nan = large_offset_nan_series.rolling(window, min_periods=10).std(ddof=test_ddof)
        pd.testing.assert_series_equal(got_nan, expected_nan, atol=1e-6, rtol=0)

    def test_rolling_std_ddof_exceeds_window(self):
        a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        got = a.vbt.rolling_std(window=2, minp=1, ddof=5)
        expected = a.rolling(window=2, min_periods=1).std(ddof=5)
        pd.testing.assert_series_equal(got, expected)

        got = a.vbt.rolling_std(window=0, minp=0, ddof=0)
        expected = a.rolling(window=0, min_periods=0).std(ddof=0)
        pd.testing.assert_series_equal(got, expected)

    def test_rolling_std_recompute(self):
        rng = np.random.default_rng(7)
        window = 30
        a = 1e14 + rng.normal(0, 1.0, 50000)
        windows = np.lib.stride_tricks.sliding_window_view(a, window)
        expected = np.std(windows - windows.mean(axis=1, keepdims=True), axis=1, ddof=1)
        got = pd.Series(a).vbt.rolling_std(window, minp=window, ddof=1).values[window - 1 :]
        assert not np.any(np.isnan(got) | (got == 0))
        assert np.median(np.abs(got - expected) / expected) < 0.2

    @pytest.mark.parametrize(
        "test_window,test_minp,test_adjust", list(product([1, 2, 3, 4, 5], [1, None], [False, True]))
    )
    def test_ewm_mean(self, test_window, test_minp, test_adjust):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df["a"].vbt.ewm_mean(test_window, minp=test_minp, adjust=test_adjust),
            df["a"].ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).mean(),
        )
        pd.testing.assert_frame_equal(
            df.vbt.ewm_mean(test_window, minp=test_minp, adjust=test_adjust),
            df.ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).mean(),
        )
        pd.testing.assert_frame_equal(df.vbt.ewm_mean(test_window), df.ewm(span=test_window).mean())

    @pytest.mark.parametrize(
        "test_window,test_minp,test_adjust", list(product([1, 2, 3, 4, 5], [1, None], [False, True]))
    )
    def test_ewm_std(self, test_window, test_minp, test_adjust):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df["a"].vbt.ewm_std(test_window, minp=test_minp, adjust=test_adjust),
            df["a"].ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).std(),
        )
        pd.testing.assert_frame_equal(
            df.vbt.ewm_std(test_window, minp=test_minp, adjust=test_adjust),
            df.ewm(span=test_window, min_periods=test_minp, adjust=test_adjust).std(),
        )
        pd.testing.assert_frame_equal(df.vbt.ewm_std(test_window), df.ewm(span=test_window).std())

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_min(self, test_minp):
        pd.testing.assert_series_equal(
            df["a"].vbt.expanding_min(minp=test_minp), df["a"].expanding(min_periods=test_minp).min()
        )
        pd.testing.assert_frame_equal(df.vbt.expanding_min(minp=test_minp), df.expanding(min_periods=test_minp).min())
        pd.testing.assert_frame_equal(df.vbt.expanding_min(), df.expanding().min())

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_max(self, test_minp):
        pd.testing.assert_series_equal(
            df["a"].vbt.expanding_max(minp=test_minp), df["a"].expanding(min_periods=test_minp).max()
        )
        pd.testing.assert_frame_equal(df.vbt.expanding_max(minp=test_minp), df.expanding(min_periods=test_minp).max())
        pd.testing.assert_frame_equal(df.vbt.expanding_max(), df.expanding().max())

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_mean(self, test_minp):
        pd.testing.assert_series_equal(
            df["a"].vbt.expanding_mean(minp=test_minp), df["a"].expanding(min_periods=test_minp).mean()
        )
        pd.testing.assert_frame_equal(df.vbt.expanding_mean(minp=test_minp), df.expanding(min_periods=test_minp).mean())
        pd.testing.assert_frame_equal(df.vbt.expanding_mean(), df.expanding().mean())

    @pytest.mark.parametrize("test_minp,test_ddof", list(product([1, 3], [0, 1])))
    def test_expanding_std(self, test_minp, test_ddof):
        pd.testing.assert_series_equal(
            df["a"].vbt.expanding_std(minp=test_minp, ddof=test_ddof),
            df["a"].expanding(min_periods=test_minp).std(ddof=test_ddof),
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_std(minp=test_minp, ddof=test_ddof),
            df.expanding(min_periods=test_minp).std(ddof=test_ddof),
        )
        pd.testing.assert_frame_equal(df.vbt.expanding_std(), df.expanding().std())

    @pytest.mark.parametrize("test_engine", ["numba", "rust"])
    def test_expanding_minp_above_length(self, test_engine):
        if test_engine == "rust" and not _engine.is_rust_available():
            pytest.skip("vectorbt-rust is not installed or version-compatible")
        minp = len(df.index) + 1
        pd.testing.assert_frame_equal(
            df.vbt.expanding_mean(minp=minp, engine=test_engine), df.expanding(min_periods=minp).mean()
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_std(minp=minp, engine=test_engine), df.expanding(min_periods=minp).std()
        )

    def test_apply_along_axis(self):
        pd.testing.assert_frame_equal(
            df.vbt.apply_along_axis(i_or_col_pow_nb, 2, axis=0), df.apply(pow_nb, args=(2,), axis=0, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.apply_along_axis(i_or_col_pow_nb, 2, axis=1), df.apply(pow_nb, args=(2,), axis=1, raw=True)
        )

    @pytest.mark.parametrize("test_window,test_minp", list(product([1, 2, 3, 4, 5], [1, None])))
    def test_rolling_apply(self, test_window, test_minp):
        if test_minp is None:
            test_minp = test_window
        pd.testing.assert_series_equal(
            df["a"].vbt.rolling_apply(test_window, i_col_nanmean_nb, minp=test_minp),
            df["a"].rolling(test_window, min_periods=test_minp).apply(nanmean_nb, raw=True),
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_apply(test_window, i_col_nanmean_nb, minp=test_minp),
            df.rolling(test_window, min_periods=test_minp).apply(nanmean_nb, raw=True),
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_apply(test_window, i_col_nanmean_nb), df.rolling(test_window).apply(nanmean_nb, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.rolling_apply(3, i_nanmean_nb, on_matrix=True),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [np.nan, np.nan, np.nan],
                        [2.75, 2.75, 2.75],
                        [np.nan, np.nan, np.nan],
                    ]
                ),
                index=df.index,
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize("test_minp", [1, 3])
    def test_expanding_apply(self, test_minp):
        pd.testing.assert_series_equal(
            df["a"].vbt.expanding_apply(i_col_nanmean_nb, minp=test_minp),
            df["a"].expanding(min_periods=test_minp).apply(nanmean_nb, raw=True),
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_apply(i_col_nanmean_nb, minp=test_minp),
            df.expanding(min_periods=test_minp).apply(nanmean_nb, raw=True),
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_apply(i_col_nanmean_nb), df.expanding().apply(nanmean_nb, raw=True)
        )
        pd.testing.assert_frame_equal(
            df.vbt.expanding_apply(i_nanmean_nb, on_matrix=True),
            pd.DataFrame(
                np.array(
                    [
                        [np.nan, np.nan, np.nan],
                        [2.0, 2.0, 2.0],
                        [2.2857142857142856, 2.2857142857142856, 2.2857142857142856],
                        [2.4, 2.4, 2.4],
                        [2.1666666666666665, 2.1666666666666665, 2.1666666666666665],
                    ]
                ),
                index=df.index,
                columns=df.columns,
            ),
        )

    def test_groupby_apply(self):
        pd.testing.assert_series_equal(
            df["a"].vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), i_col_nanmean_nb),
            df["a"].groupby(np.asarray([1, 1, 2, 2, 3])).apply(lambda x: nanmean_nb(x.values)),
        )
        pd.testing.assert_frame_equal(
            df.vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), i_col_nanmean_nb),
            df.groupby(np.asarray([1, 1, 2, 2, 3])).agg(
                {
                    "a": lambda x: nanmean_nb(x.values),
                    "b": lambda x: nanmean_nb(x.values),
                    "c": lambda x: nanmean_nb(x.values),
                }
            ),  # any clean way to do column-wise grouping in pandas?
        )

    def test_groupby_apply_on_matrix(self):
        pd.testing.assert_frame_equal(
            df.vbt.groupby_apply(np.asarray([1, 1, 2, 2, 3]), i_nanmean_nb, on_matrix=True),
            pd.DataFrame(
                np.array([[2.0, 2.0, 2.0], [2.8, 2.8, 2.8], [1.0, 1.0, 1.0]]),
                index=pd.Index([1, 2, 3], dtype="int64"),
                columns=df.columns,
            ),
        )

    @pytest.mark.parametrize(
        "test_freq",
        ["1h", "3d", "1w"],
    )
    def test_resample_apply(self, test_freq):
        pd.testing.assert_series_equal(
            df["a"].vbt.resample_apply(test_freq, i_col_nanmean_nb),
            df["a"].resample(test_freq).apply(lambda x: nanmean_nb(x.values)),
        )
        pd.testing.assert_frame_equal(
            df.vbt.resample_apply(test_freq, i_col_nanmean_nb),
            df.resample(test_freq).apply(lambda x: nanmean_nb(x.values)),
        )
        pd.testing.assert_frame_equal(
            df.vbt.resample_apply("3d", i_nanmean_nb, on_matrix=True),
            pd.DataFrame(
                np.array([[2.28571429, 2.28571429, 2.28571429], [2.0, 2.0, 2.0]]),
                index=pd.DatetimeIndex(["2018-01-01", "2018-01-04"], dtype="datetime64[ns]", freq="3D"),
                columns=df.columns,
            ),
        )

    def test_applymap(self):
        @njit
        def mult_nb(i, col, x):
            return x * 2

        pd.testing.assert_series_equal(df["a"].vbt.applymap(mult_nb), df["a"].map(lambda x: x * 2))
        pd.testing.assert_frame_equal(df.vbt.applymap(mult_nb), pandas_applymap(df, lambda x: x * 2))

    def test_filter(self):
        @njit
        def greater_nb(i, col, x):
            return x > 2

        pd.testing.assert_series_equal(df["a"].vbt.filter(greater_nb), df["a"].map(lambda x: x if x > 2 else np.nan))
        pd.testing.assert_frame_equal(df.vbt.filter(greater_nb), pandas_applymap(df, lambda x: x if x > 2 else np.nan))

    def test_apply_and_reduce(self):
        @njit
        def every_nth_nb(col, a, n):
            return a[::n]

        @njit
        def sum_nb(col, a, b):
            return np.nansum(a) + b

        assert (
            df["a"].vbt.apply_and_reduce(every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,))
            == df["a"].iloc[::2].sum() + 3
        )
        pd.testing.assert_series_equal(
            df.vbt.apply_and_reduce(every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,)),
            df.iloc[::2].sum().rename("apply_and_reduce") + 3,
        )
        pd.testing.assert_series_equal(
            df.vbt.apply_and_reduce(apply_func=every_nth_nb, reduce_func=sum_nb, apply_args=(2,), reduce_args=(3,)),
            df.iloc[::2].sum().rename("apply_and_reduce") + 3,
        )
        pd.testing.assert_series_equal(
            df.vbt.apply_and_reduce(every_nth_nb, dispatch.sum_reduce, apply_args=(2,), engine="numba"),
            df.iloc[::2].sum().rename("apply_and_reduce"),
        )
        pd.testing.assert_series_equal(
            df.vbt.apply_and_reduce(
                every_nth_nb, sum_nb, apply_args=(2,), reduce_args=(3,), wrap_kwargs=dict(to_timedelta=True)
            ),
            (df.iloc[::2].sum().rename("apply_and_reduce") + 3) * day_dt,
        )

    def test_reduce(self):
        @njit
        def sum_nb(col, a):
            return np.nansum(a)

        assert df["a"].vbt.reduce(sum_nb) == df["a"].sum()
        assert df["a"].vbt.reduce(reduce_func=sum_nb) == df["a"].sum()
        pd.testing.assert_series_equal(df.vbt.reduce(sum_nb), df.sum().rename("reduce"))
        pd.testing.assert_series_equal(df.vbt.reduce(dispatch.sum_reduce, engine="numba"), df.sum().rename("reduce"))
        pd.testing.assert_series_equal(
            df.vbt.reduce(sum_nb, wrap_kwargs=dict(to_timedelta=True)), df.sum().rename("reduce") * day_dt
        )
        pd.testing.assert_series_equal(
            df.vbt.reduce(sum_nb, group_by=group_by), pd.Series([20.0, 6.0], index=["g1", "g2"]).rename("reduce")
        )

        @njit
        def argmax_nb(col, a):
            a = a.copy()
            a[np.isnan(a)] = -np.inf
            return np.argmax(a)

        assert df["a"].vbt.reduce(argmax_nb, returns_idx=True) == df["a"].idxmax()
        pd.testing.assert_series_equal(df.vbt.reduce(argmax_nb, returns_idx=True), df.idxmax().rename("reduce"))
        pd.testing.assert_series_equal(
            df.vbt.reduce(argmax_nb, returns_idx=True, flatten=True, group_by=group_by),
            pd.Series(["2018-01-02", "2018-01-02"], dtype="datetime64[ns]", index=["g1", "g2"]).rename("reduce"),
        )

        @njit
        def min_and_max_nb(col, a):
            out = np.empty(2)
            out[0] = np.nanmin(a)
            out[1] = np.nanmax(a)
            return out

        pd.testing.assert_series_equal(
            df["a"].vbt.reduce(min_and_max_nb, returns_array=True, wrap_kwargs=dict(name_or_index=["min", "max"])),
            pd.Series([np.nanmin(df["a"]), np.nanmax(df["a"])], index=["min", "max"], name="a"),
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(min_and_max_nb, returns_array=True, wrap_kwargs=dict(name_or_index=["min", "max"])),
            df.apply(lambda x: pd.Series(np.asarray([np.nanmin(x), np.nanmax(x)]), index=["min", "max"]), axis=0),
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(
                min_and_max_nb, returns_array=True, group_by=group_by, wrap_kwargs=dict(name_or_index=["min", "max"])
            ),
            pd.DataFrame([[1.0, 1.0], [4.0, 2.0]], index=["min", "max"], columns=["g1", "g2"]),
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
            df["a"].vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                wrap_kwargs=dict(name_or_index=["idxmin", "idxmax"]),
            ),
            pd.Series([df["a"].idxmin(), df["a"].idxmax()], index=["idxmin", "idxmax"], name="a"),
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                wrap_kwargs=dict(name_or_index=["idxmin", "idxmax"]),
            ),
            df.apply(lambda x: pd.Series(np.asarray([x.idxmin(), x.idxmax()]), index=["idxmin", "idxmax"]), axis=0),
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                flatten=True,
                order="C",
                group_by=group_by,
                wrap_kwargs=dict(name_or_index=["idxmin", "idxmax"]),
            ),
            pd.DataFrame(
                [["2018-01-01", "2018-01-01"], ["2018-01-02", "2018-01-02"]],
                dtype="datetime64[ns]",
                index=["idxmin", "idxmax"],
                columns=["g1", "g2"],
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.reduce(
                argmin_and_argmax_nb,
                returns_idx=True,
                returns_array=True,
                flatten=True,
                order="F",
                group_by=group_by,
                wrap_kwargs=dict(name_or_index=["idxmin", "idxmax"]),
            ),
            pd.DataFrame(
                [["2018-01-01", "2018-01-01"], ["2018-01-04", "2018-01-02"]],
                dtype="datetime64[ns]",
                index=["idxmin", "idxmax"],
                columns=["g1", "g2"],
            ),
        )

    def test_squeeze_grouped(self):
        pd.testing.assert_frame_equal(
            df.vbt.squeeze_grouped(i_col_nanmean_nb, group_by=group_by),
            pd.DataFrame(
                [[1.0, 1.0], [3.0, 2.0], [3.0, np.nan], [3.0, 2.0], [1.0, 1.0]], index=df.index, columns=["g1", "g2"]
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.squeeze_grouped(squeeze_func=dispatch.sum_squeeze, group_by=group_by, engine="numba"),
            pd.DataFrame(
                [[1.0, 1.0], [6.0, 2.0], [6.0, 0.0], [6.0, 2.0], [1.0, 1.0]], index=df.index, columns=["g1", "g2"]
            ),
        )
        assert df["a"].vbt.squeeze_grouped(i_col_nanmean_nb, group_by=True) == 2.5

    def test_flatten_grouped(self):
        pd.testing.assert_frame_equal(
            df.vbt.flatten_grouped(group_by=group_by, order="C"),
            pd.DataFrame(
                [
                    [1.0, 1.0],
                    [np.nan, np.nan],
                    [2.0, 2.0],
                    [4.0, np.nan],
                    [3.0, np.nan],
                    [3.0, np.nan],
                    [4.0, 2.0],
                    [2.0, np.nan],
                    [np.nan, 1.0],
                    [1.0, np.nan],
                ],
                index=np.repeat(df.index, 2),
                columns=["g1", "g2"],
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.flatten_grouped(group_by=group_by, order="F"),
            pd.DataFrame(
                [
                    [1.0, 1.0],
                    [2.0, 2.0],
                    [3.0, np.nan],
                    [4.0, 2.0],
                    [np.nan, 1.0],
                    [np.nan, np.nan],
                    [4.0, np.nan],
                    [3.0, np.nan],
                    [2.0, np.nan],
                    [1.0, np.nan],
                ],
                index=np.tile(df.index, 2),
                columns=["g1", "g2"],
            ),
        )
        pd.testing.assert_series_equal(
            pd.DataFrame([[False, True], [False, True]]).vbt.flatten_grouped(group_by=True, order="C"),
            pd.Series([False, True, False, True], name="group"),
        )
        pd.testing.assert_series_equal(
            pd.DataFrame([[False, True], [False, True]]).vbt.flatten_grouped(group_by=True, order="F"),
            pd.Series([False, False, True, True], name="group"),
        )
        pd.testing.assert_frame_equal(
            pd.Series([False, True, True, False]).vbt.flatten_grouped(group_by=[0, 0, 0, 1]),
            pd.DataFrame([[0.0, 0.0], [1.0, np.nan], [1.0, np.nan]], columns=pd.Index([0, 1], dtype="int64")),
        )

    @pytest.mark.parametrize(
        "test_name,test_func,test_func_nb",
        [
            ("min", lambda x, **kwargs: x.min(**kwargs), nb.nanmin_nb),
            ("max", lambda x, **kwargs: x.max(**kwargs), nb.nanmax_nb),
            ("mean", lambda x, **kwargs: x.mean(**kwargs), nb.nanmean_nb),
            ("median", lambda x, **kwargs: x.median(**kwargs), nb.nanmedian_nb),
            ("std", lambda x, **kwargs: x.std(**kwargs, ddof=0), nb.nanstd_nb),
            ("count", lambda x, **kwargs: x.count(**kwargs), nb.nancnt_nb),
            ("sum", lambda x, **kwargs: x.sum(**kwargs), nb.nansum_nb),
        ],
    )
    def test_funcs(self, test_name, test_func, test_func_nb):
        # numeric
        assert test_func(df["a"].vbt) == test_func(df["a"])
        pd.testing.assert_series_equal(test_func(df.vbt), test_func(df).rename(test_name))
        pd.testing.assert_series_equal(
            test_func(df.vbt, group_by=group_by),
            pd.Series([test_func(df[["a", "b"]].stack()), test_func(df["c"])], index=["g1", "g2"]).rename(test_name),
        )
        np.testing.assert_array_equal(test_func(df).values, test_func_nb(df.values))
        pd.testing.assert_series_equal(
            test_func(df.vbt, wrap_kwargs=dict(to_timedelta=True)), test_func(df).rename(test_name) * day_dt
        )
        # boolean
        bool_ts = df == df
        assert test_func(bool_ts["a"].vbt) == test_func(bool_ts["a"])
        pd.testing.assert_series_equal(test_func(bool_ts.vbt), test_func(bool_ts).rename(test_name))
        pd.testing.assert_series_equal(
            test_func(bool_ts.vbt, wrap_kwargs=dict(to_timedelta=True)), test_func(bool_ts).rename(test_name) * day_dt
        )

    @pytest.mark.parametrize(
        "test_name,test_func",
        [("idxmin", lambda x, **kwargs: x.idxmin(**kwargs)), ("idxmax", lambda x, **kwargs: x.idxmax(**kwargs))],
    )
    def test_arg_funcs(self, test_name, test_func):
        assert test_func(df["a"].vbt) == test_func(df["a"])
        pd.testing.assert_series_equal(test_func(df.vbt), test_func(df).rename(test_name))
        pd.testing.assert_series_equal(
            test_func(df.vbt, group_by=group_by),
            pd.Series(
                [test_func(df[["a", "b"]].stack())[0], test_func(df["c"])], index=["g1", "g2"], dtype="datetime64[ns]"
            ).rename(test_name),
        )

    def test_describe(self):
        pd.testing.assert_series_equal(df["a"].vbt.describe(), df["a"].describe())
        pd.testing.assert_frame_equal(df.vbt.describe(percentiles=None), df.describe(percentiles=None))
        pd.testing.assert_frame_equal(df.vbt.describe(percentiles=[]), df.describe(percentiles=[]))
        test_against = df.describe(percentiles=np.arange(0, 1, 0.1))
        pd.testing.assert_frame_equal(df.vbt.describe(percentiles=np.arange(0, 1, 0.1)), test_against)
        pd.testing.assert_frame_equal(
            df.vbt.describe(percentiles=np.arange(0, 1, 0.1), group_by=group_by),
            pd.DataFrame(
                {
                    "g1": df[["a", "b"]].stack().describe(percentiles=np.arange(0, 1, 0.1)).values,
                    "g2": df["c"].describe(percentiles=np.arange(0, 1, 0.1)).values,
                },
                index=test_against.index,
            ),
        )

    def test_value_counts(self):
        pd.testing.assert_series_equal(
            df["a"].vbt.value_counts(),
            pd.Series(
                np.array([1, 1, 1, 1, 1]), index=pd.Index([1.0, 2.0, 3.0, 4.0, np.nan], dtype="float64"), name="a"
            ),
        )
        mapping = {1.0: "one", 2.0: "two", 3.0: "three", 4.0: "four"}
        pd.testing.assert_series_equal(
            df["a"].vbt.value_counts(mapping=mapping),
            pd.Series(
                np.array([1, 1, 1, 1, 1]),
                index=pd.Index(["one", "two", "three", "four", None], dtype="object"),
                name="a",
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.value_counts(),
            pd.DataFrame(
                np.array([[1, 1, 2], [1, 1, 2], [1, 1, 0], [1, 1, 0], [1, 1, 1]]),
                index=pd.Index([1.0, 2.0, 3.0, 4.0, np.nan], dtype="float64"),
                columns=df.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.value_counts(group_by=group_by),
            pd.DataFrame(
                np.array([[2, 2], [2, 2], [2, 0], [2, 0], [2, 1]]),
                index=pd.Index([1.0, 2.0, 3.0, 4.0, np.nan], dtype="float64"),
                columns=pd.Index(["g1", "g2"], dtype="object"),
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.value_counts(sort=True),
            pd.DataFrame(
                np.array([[1, 1, 2], [1, 1, 2], [1, 1, 1], [1, 1, 0], [1, 1, 0]]),
                index=pd.Index([1.0, 2.0, np.nan, 3.0, 4.0], dtype="float64"),
                columns=df.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.value_counts(sort=True, ascending=True),
            pd.DataFrame(
                np.array([[1, 1, 0], [1, 1, 0], [1, 1, 1], [1, 1, 2], [1, 1, 2]]),
                index=pd.Index([3.0, 4.0, np.nan, 1.0, 2.0], dtype="float64"),
                columns=df.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.value_counts(sort=True, normalize=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.06666666666666667, 0.06666666666666667, 0.13333333333333333],
                        [0.06666666666666667, 0.06666666666666667, 0.13333333333333333],
                        [0.06666666666666667, 0.06666666666666667, 0.06666666666666667],
                        [0.06666666666666667, 0.06666666666666667, 0.0],
                        [0.06666666666666667, 0.06666666666666667, 0.0],
                    ]
                ),
                index=pd.Index([1.0, 2.0, np.nan, 3.0, 4.0], dtype="float64"),
                columns=df.columns,
            ),
        )
        pd.testing.assert_frame_equal(
            df.vbt.value_counts(sort=True, normalize=True, dropna=True),
            pd.DataFrame(
                np.array(
                    [
                        [0.08333333333333333, 0.08333333333333333, 0.16666666666666666],
                        [0.08333333333333333, 0.08333333333333333, 0.16666666666666666],
                        [0.08333333333333333, 0.08333333333333333, 0.0],
                        [0.08333333333333333, 0.08333333333333333, 0.0],
                    ]
                ),
                index=pd.Index([1.0, 2.0, 3.0, 4.0], dtype="float64"),
                columns=df.columns,
            ),
        )

    def test_drawdown(self):
        pd.testing.assert_series_equal(df["a"].vbt.drawdown(), df["a"] / df["a"].expanding().max() - 1)
        pd.testing.assert_frame_equal(df.vbt.drawdown(), df / df.expanding().max() - 1)

    def test_drawdowns(self):
        assert type(df["a"].vbt.drawdowns) is vbt.Drawdowns
        assert df["a"].vbt.drawdowns.wrapper.freq == df["a"].vbt.wrapper.freq
        assert df["a"].vbt.drawdowns.wrapper.ndim == df["a"].ndim
        assert df.vbt.drawdowns.wrapper.ndim == df.ndim

    def test_to_mapped(self):
        np.testing.assert_array_equal(
            df.vbt.to_mapped().values, np.array([1.0, 2.0, 3.0, 4.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0])
        )
        np.testing.assert_array_equal(df.vbt.to_mapped().col_arr, np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]))
        np.testing.assert_array_equal(df.vbt.to_mapped().idx_arr, np.array([0, 1, 2, 3, 1, 2, 3, 4, 0, 1, 3, 4]))
        np.testing.assert_array_equal(
            df.vbt.to_mapped(dropna=False).values,
            np.array([1.0, 2.0, 3.0, 4.0, np.nan, np.nan, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, np.nan, 2.0, 1.0]),
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped(dropna=False).col_arr, np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])
        )
        np.testing.assert_array_equal(
            df.vbt.to_mapped(dropna=False).idx_arr, np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        )

    def test_zscore(self):
        pd.testing.assert_series_equal(df["a"].vbt.zscore(), (df["a"] - df["a"].mean()) / df["a"].std(ddof=0))
        pd.testing.assert_frame_equal(df.vbt.zscore(), (df - df.mean()) / df.std(ddof=0))

    def test_split(self):
        splitter = TimeSeriesSplit(n_splits=2)
        (train_df, train_indexes), (test_df, test_indexes) = df["a"].vbt.split(splitter)
        pd.testing.assert_frame_equal(
            train_df,
            pd.DataFrame(
                np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [np.nan, 4.0]]),
                index=pd.RangeIndex(start=0, stop=4, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None
            ),
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"],
                dtype="datetime64[ns]",
                name="split_1",
                freq=None,
            ),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(train_indexes[i], target[i])
        pd.testing.assert_frame_equal(
            test_df,
            pd.DataFrame(
                np.array([[4.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-04"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(test_indexes[i], target[i])
        (train_df, train_indexes), (test_df, test_indexes) = df.vbt.split(splitter)
        pd.testing.assert_frame_equal(
            train_df,
            pd.DataFrame(
                np.array(
                    [
                        [1.0, np.nan, 1.0, 1.0, np.nan, 1.0],
                        [2.0, 4.0, 2.0, 2.0, 4.0, 2.0],
                        [3.0, 3.0, np.nan, 3.0, 3.0, np.nan],
                        [np.nan, np.nan, np.nan, 4.0, 2.0, 2.0],
                    ]
                ),
                index=pd.RangeIndex(start=0, stop=4, step=1),
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")], names=["split_idx", None]
                ),
            ),
        )
        target = [
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None
            ),
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"],
                dtype="datetime64[ns]",
                name="split_1",
                freq=None,
            ),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(train_indexes[i], target[i])
        pd.testing.assert_frame_equal(
            test_df,
            pd.DataFrame(
                np.array([[4.0, 2.0, 2.0, np.nan, 1.0, 1.0]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.MultiIndex.from_tuples(
                    [(0, "a"), (0, "b"), (0, "c"), (1, "a"), (1, "b"), (1, "c")], names=["split_idx", None]
                ),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-04"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(test_indexes[i], target[i])

    def test_range_split(self):
        pd.testing.assert_frame_equal(
            df["a"].vbt.range_split(n=2)[0],
            pd.DataFrame(
                np.array([[1.0, 4.0], [2.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(df["a"].vbt.range_split(n=2)[1][i], target[i])
        pd.testing.assert_frame_equal(
            df["a"].vbt.range_split(range_len=2)[0],
            pd.DataFrame(
                np.array([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1, 2, 3], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_1", freq=None),
            pd.DatetimeIndex(["2018-01-03", "2018-01-04"], dtype="datetime64[ns]", name="split_2", freq=None),
            pd.DatetimeIndex(["2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_3", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(df["a"].vbt.range_split(range_len=2)[1][i], target[i])
        pd.testing.assert_frame_equal(
            df["a"].vbt.range_split(range_len=2, n=3)[0],
            pd.DataFrame(
                np.array([[1.0, 3.0, 4.0], [2.0, 4.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1, 2], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-03", "2018-01-04"], dtype="datetime64[ns]", name="split_1", freq=None),
            pd.DatetimeIndex(["2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_2", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(df["a"].vbt.range_split(range_len=2, n=3)[1][i], target[i])
        pd.testing.assert_frame_equal(
            df["a"].vbt.range_split(range_len=3, n=2)[0],
            pd.DataFrame(
                np.array([[1.0, 3.0], [2.0, 4.0], [3.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None
            ),
            pd.DatetimeIndex(
                ["2018-01-03", "2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None
            ),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(df["a"].vbt.range_split(range_len=3, n=2)[1][i], target[i])
        pd.testing.assert_frame_equal(
            df.vbt.range_split(n=2)[0],
            pd.DataFrame(
                np.array([[1.0, np.nan, 1.0, 4.0, 2.0, 2.0], [2.0, 4.0, 2.0, np.nan, 1.0, 1.0]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.MultiIndex.from_arrays(
                    [
                        pd.Index([0, 0, 0, 1, 1, 1], dtype="int64", name="split_idx"),
                        pd.Index(["a", "b", "c", "a", "b", "c"], dtype="object"),
                    ]
                ),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(df.vbt.range_split(n=2)[1][i], target[i])
        pd.testing.assert_frame_equal(
            df.vbt.range_split(start_idxs=[0, 1], end_idxs=[2, 3])[0],
            pd.DataFrame(
                np.array(
                    [
                        [1.0, np.nan, 1.0, 2.0, 4.0, 2.0],
                        [2.0, 4.0, 2.0, 3.0, 3.0, np.nan],
                        [3.0, 3.0, np.nan, 4.0, 2.0, 2.0],
                    ]
                ),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.MultiIndex.from_arrays(
                    [
                        pd.Index([0, 0, 0, 1, 1, 1], dtype="int64", name="split_idx"),
                        pd.Index(["a", "b", "c", "a", "b", "c"], dtype="object"),
                    ]
                ),
            ),
        )
        target = [
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None
            ),
            pd.DatetimeIndex(
                ["2018-01-02", "2018-01-03", "2018-01-04"], dtype="datetime64[ns]", name="split_1", freq=None
            ),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(df.vbt.range_split(start_idxs=[0, 1], end_idxs=[2, 3])[1][i], target[i])
        pd.testing.assert_frame_equal(
            df.vbt.range_split(start_idxs=df.index[[0, 1]], end_idxs=df.index[[2, 3]])[0],
            pd.DataFrame(
                np.array(
                    [
                        [1.0, np.nan, 1.0, 2.0, 4.0, 2.0],
                        [2.0, 4.0, 2.0, 3.0, 3.0, np.nan],
                        [3.0, 3.0, np.nan, 4.0, 2.0, 2.0],
                    ]
                ),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.MultiIndex.from_arrays(
                    [
                        pd.Index([0, 0, 0, 1, 1, 1], dtype="int64", name="split_idx"),
                        pd.Index(["a", "b", "c", "a", "b", "c"], dtype="object"),
                    ]
                ),
            ),
        )
        target = [
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None
            ),
            pd.DatetimeIndex(
                ["2018-01-02", "2018-01-03", "2018-01-04"], dtype="datetime64[ns]", name="split_1", freq=None
            ),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df.vbt.range_split(start_idxs=df.index[[0, 1]], end_idxs=df.index[[2, 3]])[1][i], target[i]
            )
        pd.testing.assert_frame_equal(
            df.vbt.range_split(start_idxs=df.index[[0]], end_idxs=df.index[[2, 3]])[0],
            pd.DataFrame(
                np.array(
                    [
                        [1.0, np.nan, 1.0, 1.0, np.nan, 1.0],
                        [2.0, 4.0, 2.0, 2.0, 4.0, 2.0],
                        [3.0, 3.0, np.nan, 3.0, 3.0, np.nan],
                        [np.nan, np.nan, np.nan, 4.0, 2.0, 2.0],
                    ]
                ),
                index=pd.RangeIndex(start=0, stop=4, step=1),
                columns=pd.MultiIndex.from_arrays(
                    [
                        pd.Index([0, 0, 0, 1, 1, 1], dtype="int64", name="split_idx"),
                        pd.Index(["a", "b", "c", "a", "b", "c"], dtype="object"),
                    ]
                ),
            ),
        )
        target = [
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None
            ),
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04"],
                dtype="datetime64[ns]",
                name="split_1",
                freq=None,
            ),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(
                df.vbt.range_split(start_idxs=df.index[[0]], end_idxs=df.index[[2, 3]])[1][i], target[i]
            )
        with pytest.raises(Exception):
            df.vbt.range_split()
        with pytest.raises(Exception):
            df.vbt.range_split(start_idxs=[0, 1])
        with pytest.raises(Exception):
            df.vbt.range_split(end_idxs=[2, 4])
        with pytest.raises(Exception):
            df.vbt.range_split(min_len=10)
        with pytest.raises(Exception):
            df.vbt.range_split(n=10)

    def test_rolling_split(self):
        (df1, indexes1), (df2, indexes2), (df3, indexes3) = df["a"].vbt.rolling_split(
            window_len=4, set_lens=(1, 1), left_to_right=False
        )
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([[1.0, 2.0], [2.0, 3.0]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes1[i], target[i])
        pd.testing.assert_frame_equal(
            df2,
            pd.DataFrame(
                np.array([[3.0, 4.0]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-04"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes2[i], target[i])
        pd.testing.assert_frame_equal(
            df3,
            pd.DataFrame(
                np.array([[4.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-04"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes3[i], target[i])
        (df1, indexes1), (df2, indexes2), (df3, indexes3) = df["a"].vbt.rolling_split(
            window_len=4, set_lens=(1, 1), left_to_right=True
        )
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([[1.0, 2.0]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-02"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes1[i], target[i])
        pd.testing.assert_frame_equal(
            df2,
            pd.DataFrame(
                np.array([[2.0, 3.0]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-03"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes2[i], target[i])
        pd.testing.assert_frame_equal(
            df3,
            pd.DataFrame(
                np.array([[3.0, 4.0], [4.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-03", "2018-01-04"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes3[i], target[i])
        (df1, indexes1), (df2, indexes2), (df3, indexes3) = df["a"].vbt.rolling_split(
            window_len=4, set_lens=(0.25, 0.25), left_to_right=[False, True]
        )
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([[1.0, 2.0], [2.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-02"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes1[i], target[i])
        pd.testing.assert_frame_equal(
            df2,
            pd.DataFrame(
                np.array([[3.0, 3.0]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-03"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes2[i], target[i])
        pd.testing.assert_frame_equal(
            df3,
            pd.DataFrame(
                np.array([[4.0, 4.0], [np.nan, np.nan]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-04"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes3[i], target[i])
        df1, indexes1 = df["a"].vbt.rolling_split(window_len=2, n=2)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([[1.0, 4.0], [2.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        df1, indexes1 = df["a"].vbt.rolling_split(window_len=0.4, n=2)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([[1.0, 4.0], [2.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=2, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-04", "2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes1[i], target[i])
        with pytest.raises(Exception):
            df.vbt.rolling_split()
        with pytest.raises(Exception):
            df.vbt.rolling_split(window_len=3, set_lens=(3, 1))
        with pytest.raises(Exception):
            df.vbt.rolling_split(window_len=1, set_lens=(1, 1))
        with pytest.raises(Exception):
            df.vbt.rolling_split(n=2, min_len=10)
        with pytest.raises(Exception):
            df.vbt.rolling_split(n=10)

    def test_expanding_split(self):
        (df1, indexes1), (df2, indexes2), (df3, indexes3) = df["a"].vbt.expanding_split(
            min_len=4, set_lens=(1, 1), left_to_right=False
        )
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([[1.0, 1.0], [2.0, 2.0], [np.nan, 3.0]]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03"], dtype="datetime64[ns]", name="split_1", freq=None
            ),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes1[i], target[i])
        pd.testing.assert_frame_equal(
            df2,
            pd.DataFrame(
                np.array([[3.0, 4.0]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-03"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-04"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes2[i], target[i])
        pd.testing.assert_frame_equal(
            df3,
            pd.DataFrame(
                np.array([[4.0, np.nan]]),
                index=pd.RangeIndex(start=0, stop=1, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-04"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(["2018-01-05"], dtype="datetime64[ns]", name="split_1", freq=None),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes3[i], target[i])
        df1, indexes1 = df["a"].vbt.expanding_split(n=2, min_len=2)
        pd.testing.assert_frame_equal(
            df1,
            pd.DataFrame(
                np.array([[1.0, 1.0], [2.0, 2.0], [np.nan, 3.0], [np.nan, 4.0], [np.nan, np.nan]]),
                index=pd.RangeIndex(start=0, stop=5, step=1),
                columns=pd.Index([0, 1], dtype="int64", name="split_idx"),
            ),
        )
        target = [
            pd.DatetimeIndex(["2018-01-01", "2018-01-02"], dtype="datetime64[ns]", name="split_0", freq=None),
            pd.DatetimeIndex(
                ["2018-01-01", "2018-01-02", "2018-01-03", "2018-01-04", "2018-01-05"],
                dtype="datetime64[ns]",
                name="split_1",
                freq=None,
            ),
        ]
        for i in range(len(target)):
            pd.testing.assert_index_equal(indexes1[i], target[i])
        with pytest.raises(Exception):
            df.vbt.expanding_split(n=2, min_len=10)
        with pytest.raises(Exception):
            df.vbt.expanding_split(n=10)

    def test_crossed_above(self):
        sr1 = pd.Series([np.nan, 3, 2, 1, 2, 3, 4])
        sr2 = pd.Series([1, 2, 3, 4, 3, 2, 1])
        pd.testing.assert_series_equal(
            sr1.vbt.crossed_above(sr2), pd.Series([False, False, False, False, False, True, False])
        )
        pd.testing.assert_series_equal(
            sr1.vbt.crossed_above(sr2, wait=1), pd.Series([False, False, False, False, False, False, True])
        )
        sr3 = pd.Series([1, 2, 3, np.nan, 5, 1, 5])
        sr4 = pd.Series([3, 2, 1, 1, 1, 5, 1])
        pd.testing.assert_series_equal(
            sr3.vbt.crossed_above(sr4), pd.Series([False, False, True, False, False, False, True])
        )
        pd.testing.assert_series_equal(
            sr3.vbt.crossed_above(sr4, wait=1), pd.Series([False, False, False, False, False, False, False])
        )

    def test_crossed_below(self):
        sr1 = pd.Series([np.nan, 3, 2, 1, 2, 3, 4])
        sr2 = pd.Series([1, 2, 3, 4, 3, 2, 1])
        pd.testing.assert_series_equal(
            sr1.vbt.crossed_below(sr2), pd.Series([False, False, True, False, False, False, False])
        )
        pd.testing.assert_series_equal(
            sr1.vbt.crossed_below(sr2, wait=1), pd.Series([False, False, False, True, False, False, False])
        )
        sr3 = pd.Series([1, 2, 3, np.nan, 5, 1, 5])
        sr4 = pd.Series([3, 2, 1, 1, 1, 5, 1])
        pd.testing.assert_series_equal(
            sr3.vbt.crossed_above(sr4), pd.Series([False, False, True, False, False, False, True])
        )
        pd.testing.assert_series_equal(
            sr3.vbt.crossed_above(sr4, wait=1), pd.Series([False, False, False, False, False, False, False])
        )

    def test_stats(self):
        stats_index = pd.Index(
            ["Start", "End", "Period", "Count", "Mean", "Std", "Min", "Median", "Max", "Min Index", "Max Index"],
            dtype="object",
        )
        pd.testing.assert_series_equal(
            df.vbt.stats(),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    4.0,
                    2.1666666666666665,
                    1.0531130555537456,
                    1.0,
                    2.1666666666666665,
                    3.3333333333333335,
                ],
                index=stats_index[:-2],
                name="agg_func_mean",
            ),
        )
        pd.testing.assert_series_equal(
            df.vbt.stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    4,
                    2.5,
                    1.2909944487358056,
                    1.0,
                    2.5,
                    4.0,
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-04 00:00:00"),
                ],
                index=stats_index,
                name="a",
            ),
        )
        pd.testing.assert_series_equal(
            df.vbt.stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    8,
                    2.5,
                    1.1952286093343936,
                    1.0,
                    2.5,
                    4.0,
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-02 00:00:00"),
                ],
                index=stats_index,
                name="g1",
            ),
        )
        pd.testing.assert_series_equal(df["c"].vbt.stats(), df.vbt.stats(column="c"))
        pd.testing.assert_series_equal(df["c"].vbt.stats(), df.vbt.stats(column="c", group_by=False))
        pd.testing.assert_series_equal(
            df.vbt(group_by=group_by)["g2"].stats(), df.vbt(group_by=group_by).stats(column="g2")
        )
        pd.testing.assert_series_equal(
            df.vbt(group_by=group_by)["g2"].stats(), df.vbt.stats(column="g2", group_by=group_by)
        )
        stats_df = df.vbt.stats(agg_func=None)
        assert stats_df.shape == (3, 11)
        pd.testing.assert_index_equal(stats_df.index, df.vbt.wrapper.columns)
        pd.testing.assert_index_equal(stats_df.columns, stats_index)

    def test_stats_mapping(self):
        mapping = {x: "test_" + str(x) for x in pd.unique(df.values.flatten())}
        stats_index = pd.Index(
            [
                "Start",
                "End",
                "Period",
                "Value Counts: test_1.0",
                "Value Counts: test_2.0",
                "Value Counts: test_3.0",
                "Value Counts: test_4.0",
                "Value Counts: test_nan",
            ],
            dtype="object",
        )
        pd.testing.assert_series_equal(
            df.vbt(mapping=mapping).stats(),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    1.3333333333333333,
                    1.3333333333333333,
                    0.6666666666666666,
                    0.6666666666666666,
                    1.0,
                ],
                index=stats_index,
                name="agg_func_mean",
            ),
        )
        pd.testing.assert_series_equal(
            df.vbt(mapping=mapping).stats(column="a"),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    1,
                    1,
                    1,
                    1,
                    1,
                ],
                index=stats_index,
                name="a",
            ),
        )
        pd.testing.assert_series_equal(
            df.vbt(mapping=mapping).stats(column="g1", group_by=group_by),
            pd.Series(
                [
                    pd.Timestamp("2018-01-01 00:00:00"),
                    pd.Timestamp("2018-01-05 00:00:00"),
                    pd.Timedelta("5 days 00:00:00"),
                    2,
                    2,
                    2,
                    2,
                    2,
                ],
                index=stats_index,
                name="g1",
            ),
        )
        pd.testing.assert_series_equal(df.vbt(mapping=mapping).stats(), df.vbt.stats(settings=dict(mapping=mapping)))
        pd.testing.assert_series_equal(
            df["c"].vbt(mapping=mapping).stats(settings=dict(incl_all_keys=True)),
            df.vbt(mapping=mapping).stats(column="c"),
        )
        pd.testing.assert_series_equal(
            df["c"].vbt(mapping=mapping).stats(settings=dict(incl_all_keys=True)),
            df.vbt(mapping=mapping).stats(column="c", group_by=False),
        )
        pd.testing.assert_series_equal(
            df.vbt(mapping=mapping, group_by=group_by)["g2"].stats(settings=dict(incl_all_keys=True)),
            df.vbt(mapping=mapping, group_by=group_by).stats(column="g2"),
        )
        pd.testing.assert_series_equal(
            df.vbt(mapping=mapping, group_by=group_by)["g2"].stats(settings=dict(incl_all_keys=True)),
            df.vbt(mapping=mapping).stats(column="g2", group_by=group_by),
        )
        stats_df = df.vbt(mapping=mapping).stats(agg_func=None)
        assert stats_df.shape == (3, 8)
        pd.testing.assert_index_equal(stats_df.index, df.vbt.wrapper.columns)
        pd.testing.assert_index_equal(stats_df.columns, stats_index)
