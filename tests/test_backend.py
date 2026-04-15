import numpy as np
import pandas as pd
import pytest

import vectorbt as vbt
from vectorbt import _backend
from vectorbt.generic.enums import drawdown_dt, range_dt
from vectorbt.generic import dispatch, nb
from vectorbt.indicators import dispatch as indicator_dispatch
from vectorbt.indicators import nb as indicator_nb


def teardown_module():
    vbt.settings.reset()
    _backend.clear_backend_cache()


class TestBackendResolution:
    def test_array_compatible_with_rust(self):
        assert _backend.array_compatible_with_rust(np.array([1.0, 2.0], dtype=np.float64)).supported
        assert _backend.array_compatible_with_rust(np.asfortranarray(np.ones((2, 3), dtype=np.float64))).supported

        int_support = _backend.array_compatible_with_rust(np.array([1, 2], dtype=np.int64))
        assert not int_support.supported
        assert "float64" in int_support.reason

        strided_support = _backend.array_compatible_with_rust(np.array([1.0, 2.0, 3.0])[::2])
        assert not strided_support.supported
        assert "contiguous 1D" in strided_support.reason

    def test_global_rust_support_helpers(self):
        assert _backend.combine_rust_support(_backend.RustSupport(True), _backend.RustSupport(True)).supported

        unsupported = _backend.combine_rust_support(
            _backend.RustSupport(True),
            _backend.RustSupport(False, "unsupported"),
        )
        assert not unsupported.supported
        assert unsupported.reason == "unsupported"

        assert _backend.non_neg_int_compatible_with_rust("n", 0).supported
        assert not _backend.non_neg_int_compatible_with_rust("n", -1).supported
        assert not _backend.callback_unsupported_with_rust().supported

        rolling_support = _backend.rolling_compatible_with_rust(
            np.ones((2, 2), dtype=np.float64),
            2,
            None,
        )
        assert rolling_support.supported

    def test_resolve_backend(self, monkeypatch):
        monkeypatch.setattr(_backend, "is_rust_available", lambda: True)

        assert _backend.resolve_backend("auto", _backend.RustSupport(True)) == "rust"
        assert _backend.resolve_backend("auto", _backend.RustSupport(False, "unsupported")) == "numba"
        assert _backend.resolve_backend("numba", _backend.RustSupport(True)) == "numba"
        assert _backend.resolve_backend("rust", _backend.RustSupport(True)) == "rust"

        with pytest.raises(ValueError, match="Invalid backend"):
            _backend.resolve_backend("bad", _backend.RustSupport(True))

        with pytest.raises(TypeError, match="RustSupport"):
            _backend.resolve_backend("auto", True)

    def test_resolve_backend_unavailable_rust(self, monkeypatch):
        monkeypatch.setattr(_backend, "is_rust_available", lambda: False)

        assert _backend.resolve_backend("auto", _backend.RustSupport(True)) == "numba"
        with pytest.raises(ImportError, match="vectorbt-rust is not installed"):
            _backend.resolve_backend("rust", _backend.RustSupport(True))

    def test_resolve_backend_unsupported_rust_reason(self, monkeypatch):
        monkeypatch.setattr(_backend, "is_rust_available", lambda: True)

        with pytest.raises(ValueError, match="requires float64"):
            _backend.resolve_backend("rust", _backend.RustSupport(False, "Rust backend requires float64 arrays."))

    def test_callback_function_rejects_explicit_rust(self, monkeypatch):
        monkeypatch.setattr(_backend, "is_rust_available", lambda: True)

        with pytest.raises(ValueError, match="callback-accepting"):
            dispatch.apply(np.ones((2, 2)), lambda col, a: a, backend="rust")


@pytest.mark.skipif(not _backend.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
class TestGenericRustParity:
    def test_dispatch_matches_numba(self):
        a_1d = np.array([1.0, np.nan, 3.0, 4.0, np.nan], dtype=np.float64)
        a = np.array(
            [
                [1.0, np.nan, 1.0],
                [2.0, 4.0, 2.0],
                [np.nan, 3.0, np.nan],
                [4.0, 2.0, 2.0],
                [np.nan, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        mask_1d = np.array([True, False, True, False, True])
        mask = np.array(
            [
                [True, False, True],
                [False, True, False],
                [True, False, True],
                [False, True, False],
                [True, False, True],
            ]
        )
        values_1d = np.arange(a_1d.shape[0], dtype=np.float64)
        values = np.arange(a.size, dtype=np.float64).reshape(a.shape)

        cases = [
            (dispatch.set_by_mask_1d, nb.set_by_mask_1d_nb, (a_1d, mask_1d, -1.0)),
            (dispatch.set_by_mask, nb.set_by_mask_nb, (a, mask, -1.0)),
            (dispatch.set_by_mask_mult_1d, nb.set_by_mask_mult_1d_nb, (a_1d, mask_1d, values_1d)),
            (dispatch.set_by_mask_mult, nb.set_by_mask_mult_nb, (a, mask, values)),
            (dispatch.fillna_1d, nb.fillna_1d_nb, (a_1d, -1.0)),
            (dispatch.fillna, nb.fillna_nb, (a, -1.0)),
            (dispatch.bshift_1d, nb.bshift_1d_nb, (a_1d, 2, -1.0)),
            (dispatch.bshift, nb.bshift_nb, (a, 2, -1.0)),
            (dispatch.fshift_1d, nb.fshift_1d_nb, (a_1d, 2, -1.0)),
            (dispatch.fshift, nb.fshift_nb, (a, 2, -1.0)),
            (dispatch.diff_1d, nb.diff_1d_nb, (a_1d, 1)),
            (dispatch.diff, nb.diff_nb, (a, 1)),
            (dispatch.pct_change_1d, nb.pct_change_1d_nb, (a_1d, 1)),
            (dispatch.pct_change, nb.pct_change_nb, (a, 1)),
            (dispatch.bfill_1d, nb.bfill_1d_nb, (a_1d,)),
            (dispatch.bfill, nb.bfill_nb, (a,)),
            (dispatch.ffill_1d, nb.ffill_1d_nb, (a_1d,)),
            (dispatch.ffill, nb.ffill_nb, (a,)),
            (dispatch.nanprod, nb.nanprod_nb, (a,)),
            (dispatch.nancumsum, nb.nancumsum_nb, (a,)),
            (dispatch.nancumprod, nb.nancumprod_nb, (a,)),
            (dispatch.nansum, nb.nansum_nb, (a,)),
            (dispatch.nancnt, nb.nancnt_nb, (a,)),
            (dispatch.nanmin, nb.nanmin_nb, (a,)),
            (dispatch.nanmax, nb.nanmax_nb, (a,)),
            (dispatch.nanmean, nb.nanmean_nb, (a,)),
            (dispatch.nanmedian, nb.nanmedian_nb, (a,)),
            (dispatch.nanstd_1d, nb.nanstd_1d_nb, (a_1d, 0)),
            (dispatch.nanstd, nb.nanstd_nb, (a, 0)),
            (dispatch.rolling_min_1d, nb.rolling_min_1d_nb, (a_1d, 2, None)),
            (dispatch.rolling_min, nb.rolling_min_nb, (a, 2, None)),
            (dispatch.rolling_max_1d, nb.rolling_max_1d_nb, (a_1d, 2, None)),
            (dispatch.rolling_max, nb.rolling_max_nb, (a, 2, None)),
            (dispatch.rolling_mean_1d, nb.rolling_mean_1d_nb, (a_1d, 2, None)),
            (dispatch.rolling_mean, nb.rolling_mean_nb, (a, 2, None)),
            (dispatch.rolling_std_1d, nb.rolling_std_1d_nb, (a_1d, 2, None, 0)),
            (dispatch.rolling_std, nb.rolling_std_nb, (a, 2, None, 0)),
            (dispatch.ewm_mean_1d, nb.ewm_mean_1d_nb, (a_1d, 3, 0, False)),
            (dispatch.ewm_mean, nb.ewm_mean_nb, (a, 3, 0, False)),
            (dispatch.ewm_mean_1d, nb.ewm_mean_1d_nb, (a_1d, 3, None, False)),
            (dispatch.ewm_mean, nb.ewm_mean_nb, (a, 3, None, False)),
            (dispatch.ewm_std_1d, nb.ewm_std_1d_nb, (a_1d, 3, 0, False, 0)),
            (dispatch.ewm_std, nb.ewm_std_nb, (a, 3, 0, False, 0)),
            (dispatch.ewm_std_1d, nb.ewm_std_1d_nb, (a_1d, 3, None, False, 0)),
            (dispatch.ewm_std, nb.ewm_std_nb, (a, 3, None, False, 0)),
            (dispatch.expanding_min_1d, nb.expanding_min_1d_nb, (a_1d, 1)),
            (dispatch.expanding_min, nb.expanding_min_nb, (a, 1)),
            (dispatch.expanding_max_1d, nb.expanding_max_1d_nb, (a_1d, 1)),
            (dispatch.expanding_max, nb.expanding_max_nb, (a, 1)),
            (dispatch.expanding_mean_1d, nb.expanding_mean_1d_nb, (a_1d, 1)),
            (dispatch.expanding_mean, nb.expanding_mean_nb, (a, 1)),
            (dispatch.expanding_std_1d, nb.expanding_std_1d_nb, (a_1d, 1, 0)),
            (dispatch.expanding_std, nb.expanding_std_nb, (a, 1, 0)),
            (dispatch.flatten_forder, nb.flatten_forder_nb, (a,)),
            (dispatch.flatten_grouped, nb.flatten_grouped_nb, (a, np.array([1, 2], dtype=np.int64), False)),
            (dispatch.flatten_grouped, nb.flatten_grouped_nb, (a, np.array([1, 2], dtype=np.int64), True)),
            (
                dispatch.flatten_uniform_grouped,
                nb.flatten_uniform_grouped_nb,
                (a, np.array([1, 1, 1], dtype=np.int64), False),
            ),
            (
                dispatch.flatten_uniform_grouped,
                nb.flatten_uniform_grouped_nb,
                (a, np.array([1, 1, 1], dtype=np.int64), True),
            ),
            (dispatch.describe_reduce, nb.describe_reduce_nb, (0, a_1d, np.array([0.25, 0.5, 0.75]), 0)),
            (
                dispatch.value_counts,
                nb.value_counts_nb,
                (
                    np.array([[0, 1, 2], [1, 1, 0]], dtype=np.int64),
                    3,
                    np.array([1, 2], dtype=np.int64),
                ),
            ),
            (
                dispatch.range_duration,
                nb.range_duration_nb,
                (
                    np.array([0, 1, 3], dtype=np.int64),
                    np.array([2, 4, 5], dtype=np.int64),
                    np.array([0, 1, 0], dtype=np.int64),
                ),
            ),
            (
                dispatch.range_coverage,
                nb.range_coverage_nb,
                (
                    np.array([0, 1, 3], dtype=np.int64),
                    np.array([2, 4, 5], dtype=np.int64),
                    np.array([0, 1, 0], dtype=np.int64),
                    (np.array([0, 1, 2], dtype=np.int64), np.array([2, 1], dtype=np.int64)),
                    np.array([5, 6], dtype=np.int64),
                    False,
                    False,
                ),
            ),
            (
                dispatch.ranges_to_mask,
                nb.ranges_to_mask_nb,
                (
                    np.array([0, 1, 3], dtype=np.int64),
                    np.array([2, 4, 5], dtype=np.int64),
                    np.array([0, 1, 0], dtype=np.int64),
                    (np.array([0, 1, 2], dtype=np.int64), np.array([2, 1], dtype=np.int64)),
                    6,
                ),
            ),
            (
                dispatch.dd_drawdown,
                nb.dd_drawdown_nb,
                (
                    np.array([10.0, 8.0]),
                    np.array([5.0, 4.0]),
                ),
            ),
            (
                dispatch.dd_decline_duration,
                nb.dd_decline_duration_nb,
                (
                    np.array([1, 2], dtype=np.int64),
                    np.array([3, 4], dtype=np.int64),
                ),
            ),
            (
                dispatch.dd_recovery_duration,
                nb.dd_recovery_duration_nb,
                (
                    np.array([3, 4], dtype=np.int64),
                    np.array([5, 8], dtype=np.int64),
                ),
            ),
            (
                dispatch.dd_recovery_duration_ratio,
                nb.dd_recovery_duration_ratio_nb,
                (
                    np.array([1, 2], dtype=np.int64),
                    np.array([3, 4], dtype=np.int64),
                    np.array([5, 8], dtype=np.int64),
                ),
            ),
            (
                dispatch.dd_recovery_return,
                nb.dd_recovery_return_nb,
                (
                    np.array([5.0, 4.0]),
                    np.array([8.0, 6.0]),
                ),
            ),
            (dispatch.crossed_above_1d, nb.crossed_above_1d_nb, (a_1d, np.array([0.0, 2.0, 2.0, 5.0, 1.0]), 0)),
            (dispatch.crossed_above, nb.crossed_above_nb, (a, np.full(a.shape, 2.0), 0)),
        ]
        for dispatch_func, nb_func, args in cases:
            np.testing.assert_allclose(dispatch_func(*args, backend="rust"), nb_func(*args), equal_nan=True)

        reducers = [
            (dispatch.nth_reduce, nb.nth_reduce_nb, (0, a_1d, -1)),
            (dispatch.nth_index_reduce, nb.nth_index_reduce_nb, (0, a_1d, -1)),
            (dispatch.min_reduce, nb.min_reduce_nb, (0, a_1d)),
            (dispatch.max_reduce, nb.max_reduce_nb, (0, a_1d)),
            (dispatch.mean_reduce, nb.mean_reduce_nb, (0, a_1d)),
            (dispatch.median_reduce, nb.median_reduce_nb, (0, a_1d)),
            (dispatch.std_reduce, nb.std_reduce_nb, (0, a_1d, 0)),
            (dispatch.sum_reduce, nb.sum_reduce_nb, (0, a_1d)),
            (dispatch.count_reduce, nb.count_reduce_nb, (0, a_1d)),
            (dispatch.argmin_reduce, nb.argmin_reduce_nb, (0, a_1d)),
            (dispatch.argmax_reduce, nb.argmax_reduce_nb, (0, a_1d)),
            (dispatch.min_squeeze, nb.min_squeeze_nb, (0, 0, a_1d)),
            (dispatch.max_squeeze, nb.max_squeeze_nb, (0, 0, a_1d)),
            (dispatch.sum_squeeze, nb.sum_squeeze_nb, (0, 0, a_1d)),
            (dispatch.any_squeeze, nb.any_squeeze_nb, (0, 0, a_1d)),
        ]
        for dispatch_func, nb_func, args in reducers:
            np.testing.assert_allclose(dispatch_func(*args, backend="rust"), nb_func(*args), equal_nan=True)

    def test_dispatch_optimized_kernels_match_numba_by_layout(self):
        base = np.array(
            [
                [1.0, np.nan, 1.0],
                [2.0, 4.0, 2.0],
                [np.nan, 3.0, np.nan],
                [4.0, 2.0, 2.0],
                [np.nan, 1.0, 1.0],
            ],
            dtype=np.float64,
        )

        for a in (np.ascontiguousarray(base), np.asfortranarray(base)):
            cases = [
                (dispatch.fillna, nb.fillna_nb, (a, -1.0)),
                (dispatch.bshift, nb.bshift_nb, (a, 2, -1.0)),
                (dispatch.fshift, nb.fshift_nb, (a, 2, -1.0)),
                (dispatch.diff, nb.diff_nb, (a, 1)),
                (dispatch.pct_change, nb.pct_change_nb, (a, 1)),
                (dispatch.bfill, nb.bfill_nb, (a,)),
                (dispatch.ffill, nb.ffill_nb, (a,)),
                (dispatch.nanprod, nb.nanprod_nb, (a,)),
                (dispatch.nancumsum, nb.nancumsum_nb, (a,)),
                (dispatch.nancumprod, nb.nancumprod_nb, (a,)),
                (dispatch.nansum, nb.nansum_nb, (a,)),
                (dispatch.nancnt, nb.nancnt_nb, (a,)),
                (dispatch.nanmean, nb.nanmean_nb, (a,)),
                (dispatch.nanstd, nb.nanstd_nb, (a, 0)),
                (dispatch.rolling_mean, nb.rolling_mean_nb, (a, 2, None)),
                (dispatch.rolling_std, nb.rolling_std_nb, (a, 2, None, 0)),
                (dispatch.expanding_mean, nb.expanding_mean_nb, (a, 1)),
                (dispatch.expanding_std, nb.expanding_std_nb, (a, 1, 0)),
            ]
            for dispatch_func, nb_func, args in cases:
                np.testing.assert_allclose(dispatch_func(*args, backend="rust"), nb_func(*args), equal_nan=True)

        empty_cols = np.empty((3, 0), dtype=np.float64)
        empty_col_cases = [
            (dispatch.nansum, nb.nansum_nb, (empty_cols,)),
            (dispatch.nanprod, nb.nanprod_nb, (empty_cols,)),
            (dispatch.nancnt, nb.nancnt_nb, (empty_cols,)),
            (dispatch.nanmean, nb.nanmean_nb, (empty_cols,)),
            (dispatch.nanstd, nb.nanstd_nb, (empty_cols, 0)),
            (dispatch.rolling_mean, nb.rolling_mean_nb, (empty_cols, 2, None)),
            (dispatch.expanding_mean, nb.expanding_mean_nb, (empty_cols, 1)),
        ]
        for dispatch_func, nb_func, args in empty_col_cases:
            np.testing.assert_allclose(dispatch_func(*args, backend="rust"), nb_func(*args), equal_nan=True)

    def test_dispatch_matches_numba_for_record_outputs(self):
        ts = np.array(
            [
                [np.nan, np.nan, np.nan],
                [1.0, np.nan, 1.0],
                [2.0, 2.0, np.nan],
                [np.nan, 3.0, 3.0],
                [4.0, np.nan, 2.0],
            ],
            dtype=np.float64,
        )

        ranges = dispatch.find_ranges(ts, np.nan, backend="rust")
        expected_ranges = nb.find_ranges_nb(ts, np.nan)
        assert ranges.dtype == range_dt
        assert ranges.dtype.itemsize == range_dt.itemsize
        np.testing.assert_array_equal(ranges, expected_ranges)

        drawdowns = dispatch.get_drawdowns(ts, backend="rust")
        expected_drawdowns = nb.get_drawdowns_nb(ts)
        assert drawdowns.dtype == drawdown_dt
        assert drawdowns.dtype.itemsize == drawdown_dt.itemsize
        np.testing.assert_array_equal(drawdowns, expected_drawdowns)

    def test_dispatch_get_drawdowns_empty_rows(self):
        ts = np.empty((0, 3), dtype=np.float64)

        drawdowns = dispatch.get_drawdowns(ts, backend="rust")
        expected_drawdowns = nb.get_drawdowns_nb(ts)
        assert drawdowns.dtype == drawdown_dt
        assert drawdowns.dtype.itemsize == drawdown_dt.itemsize
        np.testing.assert_array_equal(drawdowns, expected_drawdowns)

    def test_dispatch_std_ddof_larger_than_window_len(self):
        a_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = np.column_stack((a_1d, a_1d + 1.0))

        np.testing.assert_allclose(
            dispatch.rolling_std_1d(a_1d, 2, 1, 3, backend="rust"),
            np.full(a_1d.shape, np.nan),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dispatch.rolling_std(a, 2, 1, 3, backend="rust"),
            np.full(a.shape, np.nan),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dispatch.expanding_std_1d(a_1d, 1, 5, backend="rust"),
            np.full(a_1d.shape, np.nan),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dispatch.expanding_std(a, 1, 5, backend="rust"),
            np.full(a.shape, np.nan),
            equal_nan=True,
        )

    def test_dispatch_drawdown_helpers_broadcast(self):
        peak = np.array([10.0, 8.0, 12.0])
        valley = np.array([5.0])
        np.testing.assert_allclose(
            dispatch.dd_drawdown(peak, valley, backend="rust"),
            nb.dd_drawdown_nb(peak, valley),
        )

        start = np.array([1, 2, 3], dtype=np.int64)
        end = np.array([5], dtype=np.int64)
        valley_idx = np.array([3, 4, 5], dtype=np.int64)
        np.testing.assert_allclose(
            dispatch.dd_recovery_duration(valley_idx, end, backend="rust"),
            nb.dd_recovery_duration_nb(valley_idx, end),
        )
        np.testing.assert_allclose(
            dispatch.dd_recovery_duration_ratio(start, valley_idx, end, backend="rust"),
            nb.dd_recovery_duration_ratio_nb(start, valley_idx, end),
        )

        with pytest.raises(ValueError, match="broadcast"):
            dispatch.dd_drawdown(peak, np.array([5.0, 4.0]), backend="rust")

    def test_dispatch_rust_shuffle_is_seeded(self):
        a = np.arange(12, dtype=np.float64).reshape(4, 3)

        out1 = dispatch.shuffle(a, seed=42, backend="rust")
        out2 = dispatch.shuffle(a, seed=42, backend="rust")
        np.testing.assert_array_equal(out1, out2)
        for col in range(a.shape[1]):
            np.testing.assert_array_equal(np.sort(out1[:, col]), np.sort(a[:, col]))

        identical = np.tile(np.arange(20, dtype=np.float64).reshape(-1, 1), (1, 4))
        shuffled = dispatch.shuffle(identical, seed=42, backend="rust")
        assert any(not np.array_equal(shuffled[:, 0], shuffled[:, col]) for col in range(1, identical.shape[1]))

    def test_dispatch_auto_falls_back_for_unsupported_array(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)

        np.testing.assert_allclose(dispatch.diff(a, backend="auto"), nb.diff_nb(a), equal_nan=True)
        with pytest.raises(ValueError, match="float64"):
            dispatch.diff(a, backend="rust")

    def test_generic_accessors_accept_backend(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, np.nan],
                "b": [np.nan, 4, 3, 2, 1],
                "c": [1, 2, np.nan, 2, 1],
            }
        )

        pd.testing.assert_frame_equal(df.vbt.diff(1, backend="rust"), df.vbt.diff(1, backend="numba"))
        pd.testing.assert_frame_equal(df.vbt.rolling_mean(2, backend="rust"), df.vbt.rolling_mean(2, backend="numba"))
        pd.testing.assert_frame_equal(
            df.vbt.rolling_std(2, ddof=0, backend="rust"),
            df.vbt.rolling_std(2, ddof=0, backend="numba"),
        )

    def test_generic_accessors_use_global_backend(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

        try:
            vbt.settings["backend"] = "rust"
            pd.testing.assert_frame_equal(df.vbt.rolling_mean(2), df.vbt.rolling_mean(2, backend="rust"))
        finally:
            vbt.settings.reset()


@pytest.mark.skipif(not _backend.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
class TestIndicatorRustParity:
    def test_indicator_exports_match_nb_inventory(self):
        import vectorbt_rust.indicators as rust_indicators

        expected = [
            "ma_rs",
            "mstd_rs",
            "ma_cache_rs",
            "ma_apply_rs",
            "mstd_cache_rs",
            "mstd_apply_rs",
            "bb_cache_rs",
            "bb_apply_rs",
            "rsi_cache_rs",
            "rsi_apply_rs",
            "stoch_cache_rs",
            "stoch_apply_rs",
            "macd_cache_rs",
            "macd_apply_rs",
            "true_range_rs",
            "atr_cache_rs",
            "atr_apply_rs",
            "obv_custom_rs",
        ]
        assert [name for name in expected if hasattr(rust_indicators, name)] == expected

    def test_dispatch_matches_numba(self):
        close = np.array(
            [
                [1.0, np.nan, 1.0],
                [2.0, 4.0, 2.0],
                [np.nan, 3.0, np.nan],
                [4.0, 2.0, 2.0],
                [np.nan, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        high = close * 1.1
        low = close * 0.9
        volume = np.array(
            [
                [4.0, 1.0, np.nan],
                [3.0, np.nan, 2.0],
                [2.0, 3.0, 3.0],
                [1.0, 4.0, 4.0],
                [2.0, 5.0, 5.0],
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            indicator_dispatch.ma(close, 2, False, backend="rust"), indicator_nb.ma_nb(close, 2, False), equal_nan=True
        )
        np.testing.assert_allclose(
            indicator_dispatch.ma(close, 2, True, backend="rust"), indicator_nb.ma_nb(close, 2, True), equal_nan=True
        )
        np.testing.assert_allclose(
            indicator_dispatch.mstd(close, 2, False, backend="rust"),
            indicator_nb.mstd_nb(close, 2, False),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            indicator_dispatch.mstd(close, 2, True, backend="rust"),
            indicator_nb.mstd_nb(close, 2, True),
            equal_nan=True,
        )

        ma_cache = indicator_dispatch.ma_cache(close, [2, 2, 3], [False, False, True], False, backend="rust")
        ma_cache_nb = indicator_nb.ma_cache_nb(close, [2, 2, 3], [False, False, True], False)
        np.testing.assert_allclose(
            indicator_dispatch.ma_apply(close, 3, True, False, ma_cache, backend="rust"),
            indicator_nb.ma_apply_nb(close, 3, True, False, ma_cache_nb),
            equal_nan=True,
        )

        mstd_cache = indicator_dispatch.mstd_cache(close, [2, 3], [False, True], False, 0, backend="rust")
        mstd_cache_nb = indicator_nb.mstd_cache_nb(close, [2, 3], [False, True], False, 0)
        np.testing.assert_allclose(
            indicator_dispatch.mstd_apply(close, 3, True, False, 0, mstd_cache, backend="rust"),
            indicator_nb.mstd_apply_nb(close, 3, True, False, 0, mstd_cache_nb),
            equal_nan=True,
        )

        bb_cache = indicator_dispatch.bb_cache(close, [2, 3], [False, True], [2.0, 3.0], False, 0, backend="rust")
        bb_cache_nb = indicator_nb.bb_cache_nb(close, [2, 3], [False, True], [2.0, 3.0], False, 0)
        for actual, expected in zip(
            indicator_dispatch.bb_apply(close, 3, True, 2.0, False, 0, *bb_cache, backend="rust"),
            indicator_nb.bb_apply_nb(close, 3, True, 2.0, False, 0, *bb_cache_nb),
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)

        rsi_cache = indicator_dispatch.rsi_cache(close, [2, 3], [False, True], False, backend="rust")
        rsi_cache_nb = indicator_nb.rsi_cache_nb(close, [2, 3], [False, True], False)
        np.testing.assert_allclose(
            indicator_dispatch.rsi_apply(close, 3, True, False, rsi_cache, backend="rust"),
            indicator_nb.rsi_apply_nb(close, 3, True, False, rsi_cache_nb),
            equal_nan=True,
        )

        stoch_cache = indicator_dispatch.stoch_cache(
            high, low, close, [2, 3], [2, 3], [False, True], False, backend="rust"
        )
        stoch_cache_nb = indicator_nb.stoch_cache_nb(high, low, close, [2, 3], [2, 3], [False, True], False)
        for actual, expected in zip(
            indicator_dispatch.stoch_apply(high, low, close, 3, 2, True, False, stoch_cache, backend="rust"),
            indicator_nb.stoch_apply_nb(high, low, close, 3, 2, True, False, stoch_cache_nb),
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)

        macd_cache = indicator_dispatch.macd_cache(
            close, [2, 3], [3, 4], [2, 3], [False, True], [False, True], False, backend="rust"
        )
        macd_cache_nb = indicator_nb.macd_cache_nb(close, [2, 3], [3, 4], [2, 3], [False, True], [False, True], False)
        for actual, expected in zip(
            indicator_dispatch.macd_apply(close, 2, 3, 2, False, True, False, macd_cache, backend="rust"),
            indicator_nb.macd_apply_nb(close, 2, 3, 2, False, True, False, macd_cache_nb),
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)

        np.testing.assert_allclose(
            indicator_dispatch.true_range(high, low, close, backend="rust"),
            indicator_nb.true_range_nb(high, low, close),
            equal_nan=True,
        )
        atr_cache = indicator_dispatch.atr_cache(high, low, close, [2, 3], [False, True], False, backend="rust")
        atr_cache_nb = indicator_nb.atr_cache_nb(high, low, close, [2, 3], [False, True], False)
        for actual, expected in zip(
            indicator_dispatch.atr_apply(high, low, close, 3, True, False, *atr_cache, backend="rust"),
            indicator_nb.atr_apply_nb(high, low, close, 3, True, False, *atr_cache_nb),
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)
        np.testing.assert_allclose(
            indicator_dispatch.obv_custom(close, volume, backend="rust"),
            indicator_nb.obv_custom_nb(close, volume),
            equal_nan=True,
        )

    def test_basic_indicators_match_numba(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
        high = close * 1.1
        low = close * 0.9
        volume = pd.Series([4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])

        pd.testing.assert_frame_equal(
            vbt.MA.run(close, window=(2, 3), ewm=(False, True), param_product=True, backend="rust").ma,
            vbt.MA.run(close, window=(2, 3), ewm=(False, True), param_product=True, backend="numba").ma,
        )
        pd.testing.assert_frame_equal(
            vbt.MSTD.run(close, window=(2, 3), ewm=(False, True), param_product=True, backend="rust").mstd,
            vbt.MSTD.run(close, window=(2, 3), ewm=(False, True), param_product=True, backend="numba").mstd,
        )
        for attr in ("middle", "upper", "lower"):
            pd.testing.assert_frame_equal(
                getattr(
                    vbt.BBANDS.run(close, window=(2, 3), ewm=(False, True), param_product=True, backend="rust"), attr
                ),
                getattr(
                    vbt.BBANDS.run(close, window=(2, 3), ewm=(False, True), param_product=True, backend="numba"), attr
                ),
            )
        pd.testing.assert_frame_equal(
            vbt.RSI.run(close, window=(2, 3), ewm=(False, True), param_product=True, backend="rust").rsi,
            vbt.RSI.run(close, window=(2, 3), ewm=(False, True), param_product=True, backend="numba").rsi,
        )
        for attr in ("percent_k", "percent_d"):
            pd.testing.assert_frame_equal(
                getattr(
                    vbt.STOCH.run(
                        high,
                        low,
                        close,
                        k_window=(2, 3),
                        d_window=2,
                        d_ewm=(False, True),
                        param_product=True,
                        backend="rust",
                    ),
                    attr,
                ),
                getattr(
                    vbt.STOCH.run(
                        high,
                        low,
                        close,
                        k_window=(2, 3),
                        d_window=2,
                        d_ewm=(False, True),
                        param_product=True,
                        backend="numba",
                    ),
                    attr,
                ),
            )
        for attr in ("macd", "signal"):
            pd.testing.assert_frame_equal(
                getattr(
                    vbt.MACD.run(
                        close,
                        fast_window=(2, 3),
                        slow_window=4,
                        signal_window=2,
                        macd_ewm=(False, True),
                        signal_ewm=True,
                        param_product=True,
                        backend="rust",
                    ),
                    attr,
                ),
                getattr(
                    vbt.MACD.run(
                        close,
                        fast_window=(2, 3),
                        slow_window=4,
                        signal_window=2,
                        macd_ewm=(False, True),
                        signal_ewm=True,
                        param_product=True,
                        backend="numba",
                    ),
                    attr,
                ),
            )
        for attr in ("tr", "atr"):
            pd.testing.assert_frame_equal(
                getattr(
                    vbt.ATR.run(high, low, close, window=(2, 3), ewm=(False, True), param_product=True, backend="rust"),
                    attr,
                ),
                getattr(
                    vbt.ATR.run(
                        high, low, close, window=(2, 3), ewm=(False, True), param_product=True, backend="numba"
                    ),
                    attr,
                ),
            )
        pd.testing.assert_series_equal(
            vbt.OBV.run(close, volume, backend="rust").obv,
            vbt.OBV.run(close, volume, backend="numba").obv,
        )
