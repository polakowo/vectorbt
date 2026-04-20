import numpy as np
import pandas as pd
import pytest

import vectorbt as vbt
from tests.utils import record_arrays_close
from vectorbt import _engine
from vectorbt.generic.enums import drawdown_dt, range_dt
from vectorbt.generic import dispatch, nb
from vectorbt.indicators import dispatch as indicator_dispatch
from vectorbt.indicators import nb as indicator_nb
from vectorbt.labels import dispatch as labels_dispatch
from vectorbt.labels import nb as labels_nb
from vectorbt.labels.enums import TrendMode
from vectorbt.returns import dispatch as returns_dispatch
from vectorbt.returns import nb as returns_nb
from vectorbt.records import dispatch as records_dispatch
from vectorbt.records import nb as records_nb
from vectorbt.signals import dispatch as signal_dispatch
from vectorbt.signals import nb as signal_nb
from vectorbt.portfolio import dispatch as portfolio_dispatch
from vectorbt.portfolio import nb as portfolio_nb
from vectorbt.portfolio.enums import CallSeqType, Direction, RejectedOrderError, SizeType


def teardown_module():
    vbt.settings.reset()
    _engine.clear_engine_cache()


class TestEngineResolution:
    def test_array_compatible_with_rust(self):
        assert _engine.array_compatible_with_rust(np.array([1.0, 2.0], dtype=np.float64)).supported
        assert _engine.array_compatible_with_rust(np.asfortranarray(np.ones((2, 3), dtype=np.float64))).supported

        f32_support = _engine.array_compatible_with_rust(np.array([1.0, 2.0], dtype=np.float32))
        assert f32_support.supported
        assert f32_support.requires_conversion
        assert f32_support.conversions[0].dtype == np.dtype(np.float64)

        int_support = _engine.array_compatible_with_rust(np.array([1, 2], dtype=np.int64))
        assert not int_support.supported
        assert "cannot be safely cast" in int_support.reason

        int32_support = _engine.array_compatible_with_rust(
            np.array([1, 2], dtype=np.int32),
            dtype=np.int64,
        )
        assert int32_support.supported
        assert int32_support.requires_conversion

        unsafe_support = _engine.array_compatible_with_rust(
            np.array([1.0, 2.0], dtype=np.float64),
            dtype=np.int64,
        )
        assert not unsafe_support.supported
        assert "cannot be safely cast" in unsafe_support.reason

        strided_support = _engine.array_compatible_with_rust(np.array([1.0, 2.0, 3.0])[::2])
        assert strided_support.supported

        exact_support = _engine.exact_array_compatible_with_rust(np.array([1.0, 2.0], dtype=np.float32))
        assert not exact_support.supported
        assert "exact float64" in exact_support.reason

        assert _engine.array_shape_compatible_with_rust("a", np.ones((2, 3)), (2, 3)).supported
        shape_support = _engine.array_shape_compatible_with_rust("a", np.ones((3,)), (2, 3))
        assert not shape_support.supported
        assert "shape (2, 3)" in shape_support.reason

    def test_prepare_array_for_rust(self):
        f64_arr = np.array([1.0, 2.0], dtype=np.float64)
        assert _engine.prepare_array_for_rust(f64_arr, dtype=np.float64) is f64_arr

        f32_arr = np.array([1.0, 2.0], dtype=np.float32)
        prepared = _engine.prepare_array_for_rust(f32_arr, dtype=np.float64)
        assert prepared.dtype == np.dtype(np.float64)
        assert prepared.tolist() == [1.0, 2.0]

        with pytest.raises(ValueError, match="cannot be safely cast"):
            _engine.prepare_array_for_rust(np.array([1, 2], dtype=np.int64), dtype=np.float64)

    def test_global_rust_support_helpers(self):
        assert _engine.combine_rust_support(_engine.RustSupport(True), _engine.RustSupport(True)).supported

        unsupported = _engine.combine_rust_support(
            _engine.RustSupport(True),
            _engine.RustSupport(False, "unsupported"),
        )
        assert not unsupported.supported
        assert unsupported.reason == "unsupported"

        assert _engine.non_neg_int_compatible_with_rust("n", 0).supported
        assert not _engine.non_neg_int_compatible_with_rust("n", -1).supported
        assert not _engine.callback_unsupported_with_rust().supported
        assert _engine.unit_interval_compatible_with_rust("cutoff", 0.5).supported
        assert not _engine.unit_interval_compatible_with_rust("cutoff", 1.5).supported

        rolling_support = _engine.rolling_compatible_with_rust(
            np.ones((2, 2), dtype=np.float64),
            2,
            None,
        )
        assert rolling_support.supported

    def test_resolve_engine(self, monkeypatch):
        monkeypatch.setattr(_engine, "is_rust_available", lambda: True)

        assert _engine.resolve_engine("auto", _engine.RustSupport(True)) == "rust"
        assert _engine.resolve_engine("auto", _engine.RustSupport(False, "unsupported")) == "numba"
        assert _engine.resolve_engine("numba", _engine.RustSupport(True)) == "numba"
        assert _engine.resolve_engine("rust", _engine.RustSupport(True)) == "rust"

        with pytest.raises(ValueError, match="Invalid engine"):
            _engine.resolve_engine("bad", _engine.RustSupport(True))

        with pytest.raises(TypeError, match="RustSupport"):
            _engine.resolve_engine("auto", True)

    def test_resolve_engine_unavailable_rust(self, monkeypatch):
        monkeypatch.setattr(_engine, "is_rust_available", lambda: False)

        assert _engine.resolve_engine("auto", _engine.RustSupport(True)) == "numba"
        with pytest.raises(ImportError, match="vectorbt-rust is not installed"):
            _engine.resolve_engine("rust", _engine.RustSupport(True))

    def test_resolve_engine_unsupported_rust_reason(self, monkeypatch):
        monkeypatch.setattr(_engine, "is_rust_available", lambda: True)

        with pytest.raises(ValueError, match="requires float64"):
            _engine.resolve_engine("rust", _engine.RustSupport(False, "Rust engine requires float64 arrays."))

    def test_ohlcv_stats_use_generic_dispatch_engine(self, monkeypatch):
        engines = []
        orig_bfill_1d = dispatch.bfill_1d
        orig_ffill_1d = dispatch.ffill_1d

        def bfill_1d_spy(a, engine=None):
            engines.append(engine)
            return orig_bfill_1d(a, engine="numba")

        def ffill_1d_spy(a, engine=None):
            engines.append(engine)
            return orig_ffill_1d(a, engine="numba")

        monkeypatch.setattr(dispatch, "bfill_1d", bfill_1d_spy)
        monkeypatch.setattr(dispatch, "ffill_1d", ffill_1d_spy)

        ohlcv = pd.DataFrame(
            {
                "Open": [np.nan, 10.0, 11.0],
                "High": [np.nan, 12.0, 13.0],
                "Low": [np.nan, 9.0, 10.0],
                "Close": [np.nan, 11.0, 12.0],
                "Volume": [np.nan, 100.0, 200.0],
            },
            index=pd.date_range("2020-01-01", periods=3),
        )
        stats = ohlcv.vbt.ohlcv.stats(
            metrics=["first_price", "last_price", "first_volume", "last_volume"],
            settings=dict(engine="rust"),
        )

        assert stats["First Price"] == 10.0
        assert stats["Last Price"] == 12.0
        assert stats["First Volume"] == 100.0
        assert stats["Last Volume"] == 200.0
        assert engines == ["rust", "rust", "rust", "rust"]

    def test_callback_function_rejects_explicit_rust(self, monkeypatch):
        monkeypatch.setattr(_engine, "is_rust_available", lambda: True)

        with pytest.raises(ValueError, match="callback-accepting"):
            dispatch.apply(np.ones((2, 2)), lambda col, a: a, engine="rust")

    @pytest.mark.skipif(not _engine.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
    def test_explicit_rust_accepts_safe_cast_arrays(self):
        f32_arr = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)

        np.testing.assert_array_equal(
            dispatch.fillna(f32_arr, 0, engine="rust"),
            dispatch.fillna(f32_arr, 0, engine="numba"),
        )

        np.testing.assert_allclose(
            returns_dispatch.cum_returns(f32_arr, 1, engine="rust"),
            returns_dispatch.cum_returns(f32_arr, 1, engine="numba"),
        )

        np.testing.assert_allclose(dispatch.nansum(f32_arr, engine="rust"), dispatch.nansum(f32_arr, engine="numba"))
        np.testing.assert_allclose(
            dispatch.dd_drawdown(
                np.array([10.0, 8.0], dtype=np.float32),
                np.array([5.0, 4.0], dtype=np.float32),
                engine="rust",
            ),
            dispatch.dd_drawdown(
                np.array([10.0, 8.0], dtype=np.float32),
                np.array([5.0, 4.0], dtype=np.float32),
                engine="numba",
            ),
        )


@pytest.mark.skipif(not _engine.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
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
            np.testing.assert_allclose(dispatch_func(*args, engine="rust"), nb_func(*args), equal_nan=True)

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
            np.testing.assert_allclose(dispatch_func(*args, engine="rust"), nb_func(*args), equal_nan=True)

    def test_optimized_kernels_by_layout(self):
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
                np.testing.assert_allclose(dispatch_func(*args, engine="rust"), nb_func(*args), equal_nan=True)

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
            np.testing.assert_allclose(dispatch_func(*args, engine="rust"), nb_func(*args), equal_nan=True)

    def test_record_outputs_match_numba(self):
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

        ranges = dispatch.find_ranges(ts, np.nan, engine="rust")
        expected_ranges = nb.find_ranges_nb(ts, np.nan)
        assert ranges.dtype == range_dt
        assert ranges.dtype.itemsize == range_dt.itemsize
        np.testing.assert_array_equal(ranges, expected_ranges)

        drawdowns = dispatch.get_drawdowns(ts, engine="rust")
        expected_drawdowns = nb.get_drawdowns_nb(ts)
        assert drawdowns.dtype == drawdown_dt
        assert drawdowns.dtype.itemsize == drawdown_dt.itemsize
        np.testing.assert_array_equal(drawdowns, expected_drawdowns)

    def test_dispatch_get_drawdowns_empty_rows(self):
        ts = np.empty((0, 3), dtype=np.float64)

        drawdowns = dispatch.get_drawdowns(ts, engine="rust")
        expected_drawdowns = nb.get_drawdowns_nb(ts)
        assert drawdowns.dtype == drawdown_dt
        assert drawdowns.dtype.itemsize == drawdown_dt.itemsize
        np.testing.assert_array_equal(drawdowns, expected_drawdowns)

    def test_dispatch_std_ddof_larger_than_window_len(self):
        a_1d = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        a = np.column_stack((a_1d, a_1d + 1.0))

        np.testing.assert_allclose(
            dispatch.rolling_std_1d(a_1d, 2, 1, 3, engine="rust"),
            np.full(a_1d.shape, np.nan),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dispatch.rolling_std(a, 2, 1, 3, engine="rust"),
            np.full(a.shape, np.nan),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dispatch.expanding_std_1d(a_1d, 1, 5, engine="rust"),
            np.full(a_1d.shape, np.nan),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dispatch.expanding_std(a, 1, 5, engine="rust"),
            np.full(a.shape, np.nan),
            equal_nan=True,
        )

    def test_dispatch_drawdown_helpers_broadcast(self):
        peak = np.array([10.0, 8.0, 12.0])
        valley = np.array([5.0])
        np.testing.assert_allclose(
            dispatch.dd_drawdown(peak, valley, engine="rust"),
            nb.dd_drawdown_nb(peak, valley),
        )

        start = np.array([1, 2, 3], dtype=np.int64)
        end = np.array([5], dtype=np.int64)
        valley_idx = np.array([3, 4, 5], dtype=np.int64)
        np.testing.assert_allclose(
            dispatch.dd_recovery_duration(valley_idx, end, engine="rust"),
            nb.dd_recovery_duration_nb(valley_idx, end),
        )
        np.testing.assert_allclose(
            dispatch.dd_recovery_duration_ratio(start, valley_idx, end, engine="rust"),
            nb.dd_recovery_duration_ratio_nb(start, valley_idx, end),
        )

        with pytest.raises(ValueError, match="broadcast"):
            dispatch.dd_drawdown(peak, np.array([5.0, 4.0]), engine="rust")

    def test_dispatch_rust_shuffle_is_seeded(self):
        a = np.arange(12, dtype=np.float64).reshape(4, 3)

        out1 = dispatch.shuffle(a, seed=42, engine="rust")
        out2 = dispatch.shuffle(a, seed=42, engine="rust")
        np.testing.assert_array_equal(out1, out2)
        for col in range(a.shape[1]):
            np.testing.assert_array_equal(np.sort(out1[:, col]), np.sort(a[:, col]))

        identical = np.tile(np.arange(20, dtype=np.float64).reshape(-1, 1), (1, 4))
        shuffled = dispatch.shuffle(identical, seed=42, engine="rust")
        assert any(not np.array_equal(shuffled[:, 0], shuffled[:, col]) for col in range(1, identical.shape[1]))

    def test_dispatch_random_auto_uses_numba(self):
        a = np.arange(12, dtype=np.float64).reshape(4, 3)
        np.testing.assert_array_equal(dispatch.shuffle(a, seed=42, engine="auto"), nb.shuffle_nb(a, seed=42))

    def test_auto_falls_back_for_bad_array(self):
        a = np.array([[1, 2], [3, 4]], dtype=np.int64)

        np.testing.assert_allclose(dispatch.diff(a, engine="auto"), nb.diff_nb(a), equal_nan=True)
        with pytest.raises(ValueError, match="float64"):
            dispatch.diff(a, engine="rust")

    def test_dispatch_supports_strided_1d_arrays(self):
        a = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0], dtype=np.float64)[::2]
        mask = np.array([True, False, True, False, True, False], dtype=np.bool_)[::2]
        values = np.arange(6, dtype=np.float64)[::2]

        np.testing.assert_allclose(dispatch.fillna_1d(a, -1.0, engine="rust"), nb.fillna_1d_nb(a, -1.0), equal_nan=True)
        np.testing.assert_allclose(dispatch.diff_1d(a, 1, engine="rust"), nb.diff_1d_nb(a, 1), equal_nan=True)
        np.testing.assert_allclose(
            dispatch.set_by_mask_1d(a, mask, -1.0, engine="rust"),
            nb.set_by_mask_1d_nb(a, mask, -1.0),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            dispatch.set_by_mask_mult_1d(a, mask, values, engine="rust"),
            nb.set_by_mask_mult_1d_nb(a, mask, values),
            equal_nan=True,
        )

    def test_generic_accessors_accept_engine(self):
        df = pd.DataFrame(
            {
                "a": [1, 2, 3, 4, np.nan],
                "b": [np.nan, 4, 3, 2, 1],
                "c": [1, 2, np.nan, 2, 1],
            }
        )

        pd.testing.assert_frame_equal(df.vbt.diff(1, engine="rust"), df.vbt.diff(1, engine="numba"))
        pd.testing.assert_frame_equal(df.vbt.rolling_mean(2, engine="rust"), df.vbt.rolling_mean(2, engine="numba"))
        pd.testing.assert_frame_equal(
            df.vbt.rolling_std(2, ddof=0, engine="rust"),
            df.vbt.rolling_std(2, ddof=0, engine="numba"),
        )

    def test_generic_accessors_use_global_engine(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [3.0, 2.0, 1.0]})

        try:
            vbt.settings["engine"] = "rust"
            pd.testing.assert_frame_equal(df.vbt.rolling_mean(2), df.vbt.rolling_mean(2, engine="rust"))
        finally:
            vbt.settings.reset()


@pytest.mark.skipif(not _engine.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
class TestReturnsRustParity:
    def test_dispatch_matches_numba(self):
        values = np.array(
            [
                [1.0, 5.0, 1.0],
                [2.0, 4.0, 2.0],
                [3.0, 3.0, 3.0],
                [4.0, 2.0, 2.0],
                [5.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        init_value = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
        returns = returns_nb.returns_nb(values, init_value)
        benchmark = np.array(
            [
                [np.nan, np.nan, np.nan],
                [0.8, -0.15, 0.9],
                [0.4, -0.2, 0.6],
                [0.25, -0.3, -0.25],
                [0.2, -0.5, -0.4],
            ],
            dtype=np.float64,
        )

        np.testing.assert_allclose(
            returns_dispatch.returns_1d(values[:, 0], np.nan, engine="rust"),
            returns_nb.returns_1d_nb(values[:, 0], np.nan),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            returns_dispatch.cum_returns_1d(returns[:, 0], 0.0, engine="rust"),
            returns_nb.cum_returns_1d_nb(returns[:, 0], 0.0),
            equal_nan=True,
        )
        assert returns_dispatch.get_return(1.0, 2.0, engine="rust") == returns_nb.get_return_nb(1.0, 2.0)
        np.testing.assert_allclose(
            returns_dispatch.get_return(0.0, np.nan, engine="rust"),
            returns_nb.get_return_nb(0.0, np.nan),
            equal_nan=True,
        )
        assert returns_dispatch.cum_returns_final_1d(
            returns[:, 0], 0.0, engine="rust"
        ) == returns_nb.cum_returns_final_1d_nb(returns[:, 0], 0.0)

        cases = [
            (returns_dispatch.returns, returns_nb.returns_nb, (values, init_value)),
            (returns_dispatch.cum_returns, returns_nb.cum_returns_nb, (returns, 0.0)),
            (returns_dispatch.cum_returns_final, returns_nb.cum_returns_final_nb, (returns, 0.0)),
            (returns_dispatch.annualized_return, returns_nb.annualized_return_nb, (returns, 365.0)),
            (returns_dispatch.annualized_volatility, returns_nb.annualized_volatility_nb, (returns, 365.0, 2.0, 1)),
            (returns_dispatch.drawdown, returns_nb.drawdown_nb, (returns,)),
            (returns_dispatch.max_drawdown, returns_nb.max_drawdown_nb, (returns,)),
            (returns_dispatch.calmar_ratio, returns_nb.calmar_ratio_nb, (returns, 365.0)),
            (returns_dispatch.omega_ratio, returns_nb.omega_ratio_nb, (returns, 365.0, 0.01, 0.1)),
            (returns_dispatch.sharpe_ratio, returns_nb.sharpe_ratio_nb, (returns, 365.0, 0.01, 1)),
            (returns_dispatch.downside_risk, returns_nb.downside_risk_nb, (returns, 365.0, 0.1)),
            (returns_dispatch.sortino_ratio, returns_nb.sortino_ratio_nb, (returns, 365.0, 0.1)),
            (returns_dispatch.information_ratio, returns_nb.information_ratio_nb, (returns, benchmark, 1)),
            (returns_dispatch.beta, returns_nb.beta_nb, (returns, benchmark)),
            (returns_dispatch.alpha, returns_nb.alpha_nb, (returns, benchmark, 365.0, 0.01)),
            (returns_dispatch.tail_ratio, returns_nb.tail_ratio_nb, (returns,)),
            (returns_dispatch.value_at_risk, returns_nb.value_at_risk_nb, (returns, 0.05)),
            (returns_dispatch.cond_value_at_risk, returns_nb.cond_value_at_risk_nb, (returns, 0.05)),
            (returns_dispatch.capture, returns_nb.capture_nb, (returns, benchmark, 365.0)),
            (returns_dispatch.up_capture, returns_nb.up_capture_nb, (returns, benchmark, 365.0)),
            (returns_dispatch.down_capture, returns_nb.down_capture_nb, (returns, benchmark, 365.0)),
        ]
        for dispatch_func, nb_func, args in cases:
            np.testing.assert_allclose(dispatch_func(*args, engine="rust"), nb_func(*args), equal_nan=True)

    def test_dispatch_explicit_rust_validation(self):
        returns = np.array([[np.nan, 0.1], [0.2, 0.3]], dtype=np.float64)
        benchmark = np.array([[np.nan], [0.2]], dtype=np.float64)

        with pytest.raises(ValueError, match="same shape"):
            returns_dispatch.beta(returns, benchmark, engine="rust")

        with pytest.raises(ValueError, match="between 0 and 1"):
            returns_dispatch.value_at_risk(returns, 1.5, engine="rust")

    def test_dispatch_rolling_matches_numba(self):
        returns = np.array(
            [
                [np.nan, np.nan, np.nan],
                [0.10, -0.20, 0.30],
                [0.05, np.nan, -0.10],
                [-0.02, 0.15, 0.00],
                [0.04, -0.05, 0.20],
                [np.nan, 0.03, -0.04],
            ],
            dtype=np.float64,
        )
        benchmark = np.array(
            [
                [np.nan, np.nan, np.nan],
                [0.08, -0.10, 0.20],
                [0.02, -0.05, -0.08],
                [-0.01, 0.10, 0.02],
                [0.03, -0.03, 0.12],
                [0.01, 0.02, -0.02],
            ],
            dtype=np.float64,
        )
        cases = [
            (
                returns_dispatch.rolling_cum_returns_final,
                returns_nb.rolling_cum_returns_final_nb,
                (returns, 3, None, 0.0),
            ),
            (
                returns_dispatch.rolling_annualized_return,
                returns_nb.rolling_annualized_return_nb,
                (returns, 3, None, 365.0),
            ),
            (
                returns_dispatch.rolling_annualized_volatility,
                returns_nb.rolling_annualized_volatility_nb,
                (returns, 3, None, 365.0, 2.0, 1),
            ),
            (returns_dispatch.rolling_max_drawdown, returns_nb.rolling_max_drawdown_nb, (returns, 3, None)),
            (returns_dispatch.rolling_calmar_ratio, returns_nb.rolling_calmar_ratio_nb, (returns, 3, None, 365.0)),
            (
                returns_dispatch.rolling_omega_ratio,
                returns_nb.rolling_omega_ratio_nb,
                (returns, 3, None, 365.0, 0.01, 0.1),
            ),
            (
                returns_dispatch.rolling_sharpe_ratio,
                returns_nb.rolling_sharpe_ratio_nb,
                (returns, 3, None, 365.0, 0.01, 1),
            ),
            (
                returns_dispatch.rolling_downside_risk,
                returns_nb.rolling_downside_risk_nb,
                (returns, 3, None, 365.0, 0.1),
            ),
            (
                returns_dispatch.rolling_sortino_ratio,
                returns_nb.rolling_sortino_ratio_nb,
                (returns, 3, None, 365.0, 0.1),
            ),
            (
                returns_dispatch.rolling_information_ratio,
                returns_nb.rolling_information_ratio_nb,
                (returns, 3, None, benchmark, 1),
            ),
            (returns_dispatch.rolling_beta, returns_nb.rolling_beta_nb, (returns, 3, None, benchmark)),
            (returns_dispatch.rolling_alpha, returns_nb.rolling_alpha_nb, (returns, 3, None, benchmark, 365.0, 0.01)),
            (returns_dispatch.rolling_tail_ratio, returns_nb.rolling_tail_ratio_nb, (returns, 3, None)),
            (returns_dispatch.rolling_value_at_risk, returns_nb.rolling_value_at_risk_nb, (returns, 3, None, 0.05)),
            (
                returns_dispatch.rolling_cond_value_at_risk,
                returns_nb.rolling_cond_value_at_risk_nb,
                (returns, 3, None, 0.05),
            ),
            (returns_dispatch.rolling_capture, returns_nb.rolling_capture_nb, (returns, 3, None, benchmark, 365.0)),
            (
                returns_dispatch.rolling_up_capture,
                returns_nb.rolling_up_capture_nb,
                (returns, 3, None, benchmark, 365.0),
            ),
            (
                returns_dispatch.rolling_down_capture,
                returns_nb.rolling_down_capture_nb,
                (returns, 3, None, benchmark, 365.0),
            ),
        ]
        for minp in (None, 1, 2):
            for dispatch_func, nb_func, args in cases:
                args = args[:2] + (minp,) + args[3:]
                np.testing.assert_allclose(dispatch_func(*args, engine="rust"), nb_func(*args), equal_nan=True)

        with pytest.raises(ValueError, match="minp must be <= window"):
            returns_dispatch.rolling_sharpe_ratio(returns, 2, 3, 365.0, engine="rust")
        with pytest.raises(ValueError, match="same shape"):
            returns_dispatch.rolling_beta(returns, 3, None, benchmark[:, :1], engine="rust")

    def test_dispatch_drawdown_edge_cases_match_numba(self):
        returns = np.array([[np.inf, -np.inf], [0.1, -0.1]], dtype=np.float64)

        np.testing.assert_allclose(
            returns_dispatch.drawdown(returns, engine="rust"),
            returns_nb.drawdown_nb(returns),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            returns_dispatch.max_drawdown(returns, engine="rust"),
            returns_nb.max_drawdown_nb(returns),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            returns_dispatch.calmar_ratio(returns, 365.0, engine="rust"),
            returns_nb.calmar_ratio_nb(returns, 365.0),
            equal_nan=True,
        )

        empty_returns = np.empty((0, 2), dtype=np.float64)
        with pytest.raises(ValueError, match="zero-size array"):
            returns_dispatch.max_drawdown(empty_returns, engine="rust")
        with pytest.raises(ValueError, match="zero-size array"):
            returns_nb.max_drawdown_nb(empty_returns)
        with pytest.raises(ValueError, match="zero-size array"):
            returns_dispatch.calmar_ratio(empty_returns, 365.0, engine="rust")
        with pytest.raises(ValueError, match="zero-size array"):
            returns_nb.calmar_ratio_nb(empty_returns, 365.0)
        with pytest.raises(ZeroDivisionError):
            returns_dispatch.annualized_return(empty_returns, 365.0, engine="rust")
        with pytest.raises(ZeroDivisionError):
            returns_nb.annualized_return_nb(empty_returns, 365.0)
        with pytest.raises(ZeroDivisionError):
            returns_dispatch.cond_value_at_risk(empty_returns, engine="rust")
        with pytest.raises(ZeroDivisionError):
            returns_nb.cond_value_at_risk_nb(empty_returns)

    def test_accessor_methods_match_numba(self):
        price = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 4.0, 3.0, 2.0, 1.0],
                "c": [1.0, 2.0, 3.0, 2.0, 1.0],
            },
            index=pd.date_range("2020-01-01", periods=5, freq="D"),
        )
        rets = price.pct_change()
        benchmark_rets = pd.DataFrame(
            {
                "a": [np.nan, 0.8, 0.4, 0.25, 0.2],
                "b": [np.nan, -0.15, -0.2, -0.3, -0.5],
                "c": [np.nan, 0.9, 0.6, -0.25, -0.4],
            },
            index=rets.index,
        )

        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.returns.from_value(price, engine="rust").obj,
            pd.DataFrame.vbt.returns.from_value(price, engine="numba").obj,
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns.daily(engine="rust"),
            rets.vbt.returns.daily(engine="numba"),
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns.annual(engine="rust"),
            rets.vbt.returns.annual(engine="numba"),
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns(engine="rust").daily(),
            rets.vbt.returns(engine="numba").daily(),
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns(engine="rust").annual(),
            rets.vbt.returns(engine="numba").annual(),
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns.cumulative(engine="rust"), rets.vbt.returns.cumulative(engine="numba")
        )
        pd.testing.assert_series_equal(rets.vbt.returns.total(engine="rust"), rets.vbt.returns.total(engine="numba"))
        pd.testing.assert_series_equal(
            rets.vbt.returns.annualized_volatility(engine="rust"),
            rets.vbt.returns.annualized_volatility(engine="numba"),
        )
        pd.testing.assert_series_equal(
            rets.vbt.returns.information_ratio(benchmark_rets=benchmark_rets, engine="rust"),
            rets.vbt.returns.information_ratio(benchmark_rets=benchmark_rets, engine="numba"),
        )
        pd.testing.assert_series_equal(
            rets.vbt.returns.common_sense_ratio(engine="rust"),
            rets.vbt.returns.common_sense_ratio(engine="numba"),
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns.drawdown(engine="rust"), rets.vbt.returns.drawdown(engine="numba")
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns.rolling_total(window=3, minp=1, engine="rust"),
            rets.vbt.returns.rolling_total(window=3, minp=1, engine="numba"),
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns.rolling_annualized_volatility(window=3, minp=1, engine="rust"),
            rets.vbt.returns.rolling_annualized_volatility(window=3, minp=1, engine="numba"),
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns.rolling_information_ratio(
                benchmark_rets=benchmark_rets,
                window=3,
                minp=1,
                engine="rust",
            ),
            rets.vbt.returns.rolling_information_ratio(
                benchmark_rets=benchmark_rets,
                window=3,
                minp=1,
                engine="numba",
            ),
        )
        pd.testing.assert_frame_equal(
            rets.vbt.returns.rolling_common_sense_ratio(window=3, minp=1, engine="rust"),
            rets.vbt.returns.rolling_common_sense_ratio(window=3, minp=1, engine="numba"),
        )


@pytest.mark.skipif(not _engine.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
class TestSignalsRustParity:
    def test_random_auto_numba_explicit_rust(self):
        shape = (8, 2)
        n = np.array([2, 3], dtype=np.int64)

        auto = signal_dispatch.generate_rand(shape, n, seed=42, engine="auto")
        numba = signal_nb.generate_rand_nb(shape, n, seed=42)
        rust = signal_dispatch.generate_rand(shape, n, seed=42, engine="rust")

        np.testing.assert_array_equal(auto, numba)
        assert rust.shape == auto.shape
        np.testing.assert_array_equal(rust.sum(axis=0), n)

    def test_random_dispatch_validates_shapes(self):
        shape = (8, 2)
        with pytest.raises(ValueError, match="shape \\(2,\\)"):
            signal_dispatch.generate_rand(shape, np.array([2, 3, 4], dtype=np.int64), seed=42, engine="rust")
        with pytest.raises(ValueError, match="shape \\(8, 2\\)"):
            signal_dispatch.generate_rand_by_prob(
                shape,
                np.array([0.2, 0.3], dtype=np.float64),
                True,
                True,
                seed=42,
                engine="rust",
            )

    def test_ohlc_stop_requires_exact_outputs(self):
        entries = np.array([[True, False], [False, False]], dtype=np.bool_)
        price = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float64)
        stop_price_out = np.full(entries.shape, np.nan, dtype=np.float32)
        stop_type_out = np.full(entries.shape, -1, dtype=np.int64)

        with pytest.raises(ValueError, match="exact float64"):
            signal_dispatch.generate_ohlc_stop_ex(
                entries,
                price,
                price,
                price,
                price,
                stop_price_out,
                stop_type_out,
                np.full(entries.shape, np.nan, dtype=np.float64),
                np.full(entries.shape, False, dtype=np.bool_),
                np.full(entries.shape, np.nan, dtype=np.float64),
                np.full(entries.shape, False, dtype=np.bool_),
                True,
                1,
                True,
                False,
                True,
                True,
                engine="rust",
            )

    def test_dispatch_matches_numba_for_masks(self):
        entries = np.array(
            [
                [True, False, True],
                [True, False, False],
                [True, True, True],
                [False, True, False],
                [False, True, True],
            ],
            dtype=np.bool_,
        )
        exits = np.array(
            [
                [True, True, False],
                [False, False, False],
                [True, False, True],
                [False, False, False],
                [True, True, True],
            ],
            dtype=np.bool_,
        )

        for entry_first in (True, False):
            result = signal_dispatch.clean_enex(entries, exits, entry_first, engine="rust")
            expected = signal_nb.clean_enex_nb(entries, exits, entry_first)
            np.testing.assert_array_equal(result[0], expected[0])
            np.testing.assert_array_equal(result[1], expected[1])

            entries_1d = entries[:, 0]
            exits_1d = exits[:, 0]
            result_1d = signal_dispatch.clean_enex_1d(entries_1d, exits_1d, entry_first, engine="rust")
            expected_1d = signal_nb.clean_enex_1d_nb(entries_1d, exits_1d, entry_first)
            np.testing.assert_array_equal(result_1d[0], expected_1d[0])
            np.testing.assert_array_equal(result_1d[1], expected_1d[1])

    def test_dispatch_matches_numba_for_mask_layouts(self):
        a_c = np.array(
            [
                [True, False, False],
                [False, True, False],
                [True, True, True],
                [False, False, True],
                [True, False, True],
            ],
            dtype=np.bool_,
        )
        a_f = np.asfortranarray(a_c)
        reset_by = np.array(
            [
                [False, False, False],
                [True, False, False],
                [False, False, True],
                [False, True, False],
                [False, False, False],
            ],
            dtype=np.bool_,
        )

        for a in (a_c, a_f):
            np.testing.assert_array_equal(
                signal_dispatch.sig_pos_rank(a, None, False, False, engine="rust"),
                signal_nb.rank_nb(a, None, False, signal_nb.sig_pos_rank_nb, np.full(a.shape[1], -1), False),
            )
            np.testing.assert_array_equal(
                signal_dispatch.sig_pos_rank(a, reset_by, False, True, engine="rust"),
                signal_nb.rank_nb(
                    a,
                    reset_by,
                    False,
                    signal_nb.sig_pos_rank_nb,
                    np.full(a.shape[1], -1),
                    True,
                ),
            )
            np.testing.assert_array_equal(
                signal_dispatch.part_pos_rank(a, reset_by, True, engine="rust"),
                signal_nb.rank_nb(
                    a,
                    reset_by,
                    True,
                    signal_nb.part_pos_rank_nb,
                    np.full(a.shape[1], -1),
                ),
            )
            np.testing.assert_array_equal(
                signal_dispatch.nth_index(a, -1, engine="rust"),
                signal_nb.nth_index_nb(a, -1),
            )
            np.testing.assert_allclose(
                signal_dispatch.norm_avg_index(a, engine="rust"),
                signal_nb.norm_avg_index_nb(a),
                equal_nan=True,
            )

    def test_record_outputs_match_numba(self):
        a = np.array(
            [
                [True, False, False],
                [False, True, False],
                [True, True, True],
                [False, False, False],
                [True, False, True],
            ],
            dtype=np.bool_,
        )
        b = np.array(
            [
                [False, True, False],
                [True, False, False],
                [False, False, True],
                [True, True, False],
                [False, False, True],
            ],
            dtype=np.bool_,
        )

        cases = [
            (signal_dispatch.between_ranges, signal_nb.between_ranges_nb, (a,)),
            (signal_dispatch.between_two_ranges, signal_nb.between_two_ranges_nb, (a, b, False)),
            (signal_dispatch.between_two_ranges, signal_nb.between_two_ranges_nb, (a, b, True)),
            (signal_dispatch.partition_ranges, signal_nb.partition_ranges_nb, (a,)),
            (signal_dispatch.between_partition_ranges, signal_nb.between_partition_ranges_nb, (a,)),
        ]
        for dispatch_func, nb_func, args in cases:
            result = dispatch_func(*args, engine="rust")
            expected = nb_func(*args)
            assert result.dtype == range_dt
            assert result.dtype.itemsize == range_dt.itemsize
            np.testing.assert_array_equal(result, expected)

    def test_dispatch_index_edge_cases(self):
        a = np.array(
            [
                [False, False, True],
                [False, False, False],
                [False, True, False],
            ],
            dtype=np.bool_,
        )
        no_signal_cols = np.zeros((3, 2), dtype=np.bool_)
        empty_rows = np.empty((0, 2), dtype=np.bool_)
        single_row = np.array([[True, False]], dtype=np.bool_)
        single_signal = np.array([True], dtype=np.bool_)

        a_1d = a[:, 1]
        assert signal_dispatch.nth_index_1d(a_1d, 0, engine="rust") == signal_nb.nth_index_1d_nb(a_1d, 0)
        assert signal_dispatch.nth_index_1d(a_1d, -1, engine="rust") == signal_nb.nth_index_1d_nb(a_1d, -1)
        np.testing.assert_array_equal(signal_dispatch.nth_index(a, 1, engine="rust"), signal_nb.nth_index_nb(a, 1))
        with pytest.raises(ZeroDivisionError):
            signal_dispatch.norm_avg_index_1d(single_signal, engine="rust")
        with pytest.raises(ZeroDivisionError):
            signal_nb.norm_avg_index_1d_nb(single_signal)
        with pytest.raises(ZeroDivisionError):
            signal_dispatch.norm_avg_index(no_signal_cols, engine="rust")
        with pytest.raises(ZeroDivisionError):
            signal_nb.norm_avg_index_nb(no_signal_cols)
        with pytest.raises(ZeroDivisionError):
            signal_dispatch.norm_avg_index(empty_rows, engine="rust")
        with pytest.raises(ZeroDivisionError):
            signal_nb.norm_avg_index_nb(empty_rows)
        with pytest.raises(ZeroDivisionError):
            signal_dispatch.norm_avg_index(single_row, engine="rust")
        with pytest.raises(ZeroDivisionError):
            signal_nb.norm_avg_index_nb(single_row)

    def test_auto_falls_back_for_bad_array(self):
        a = np.array([[1, 0], [0, 1]], dtype=np.int64)

        np.testing.assert_array_equal(signal_dispatch.nth_index(a, 0, engine="auto"), signal_nb.nth_index_nb(a, 0))
        with pytest.raises(ValueError, match="bool"):
            signal_dispatch.nth_index(a, 0, engine="rust")
        with pytest.raises(ValueError, match="same shape"):
            signal_dispatch.clean_enex(
                np.ones((2, 2), dtype=np.bool_),
                np.ones((2, 1), dtype=np.bool_),
                True,
                engine="rust",
            )

    def test_signal_accessors_accept_engine(self):
        mask = pd.DataFrame(
            {
                "a": [True, False, True, False, True],
                "b": [False, True, True, False, False],
                "c": [True, False, False, True, True],
            }
        )
        exits = pd.Series([False, True, False, True, False])

        pd.testing.assert_frame_equal(
            mask.vbt.signals.clean(exits, engine="rust")[0],
            mask.vbt.signals.clean(exits, engine="numba")[0],
        )
        pd.testing.assert_frame_equal(
            mask.vbt.signals.pos_rank(engine="rust"), mask.vbt.signals.pos_rank(engine="numba")
        )
        pd.testing.assert_frame_equal(
            mask.vbt.signals.pos_rank(reset_by=~mask, allow_gaps=True, engine="rust"),
            mask.vbt.signals.pos_rank(reset_by=~mask, allow_gaps=True, engine="numba"),
        )
        pd.testing.assert_frame_equal(
            mask.vbt.signals.partition_pos_rank(after_false=True, engine="rust"),
            mask.vbt.signals.partition_pos_rank(after_false=True, engine="numba"),
        )
        pd.testing.assert_frame_equal(mask.vbt.signals.first(engine="rust"), mask.vbt.signals.first(engine="numba"))
        pd.testing.assert_frame_equal(mask.vbt.signals.nth(1, engine="rust"), mask.vbt.signals.nth(1, engine="numba"))
        pd.testing.assert_frame_equal(
            mask.vbt.signals.from_nth(1, engine="rust"),
            mask.vbt.signals.from_nth(1, engine="numba"),
        )
        pd.testing.assert_series_equal(
            mask.vbt.signals.nth_index(-1, engine="rust"),
            mask.vbt.signals.nth_index(-1, engine="numba"),
        )
        pd.testing.assert_series_equal(
            mask.vbt.signals.norm_avg_index(engine="rust"),
            mask.vbt.signals.norm_avg_index(engine="numba"),
        )
        pd.testing.assert_frame_equal(
            mask.vbt.signals.between_ranges(engine="rust").records_readable,
            mask.vbt.signals.between_ranges(engine="numba").records_readable,
        )
        pd.testing.assert_frame_equal(
            mask.vbt.signals.between_ranges(other=~mask, engine="rust").records_readable,
            mask.vbt.signals.between_ranges(other=~mask, engine="numba").records_readable,
        )
        pd.testing.assert_frame_equal(
            mask.vbt.signals.partition_ranges(engine="rust").records_readable,
            mask.vbt.signals.partition_ranges(engine="numba").records_readable,
        )
        pd.testing.assert_frame_equal(
            mask.vbt.signals.between_partition_ranges(engine="rust").records_readable,
            mask.vbt.signals.between_partition_ranges(engine="numba").records_readable,
        )


@pytest.mark.skipif(not _engine.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
class TestIndicatorRustParity:
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
            indicator_dispatch.ma(close, 2, False, engine="rust"), indicator_nb.ma_nb(close, 2, False), equal_nan=True
        )
        np.testing.assert_allclose(
            indicator_dispatch.ma(close, 2, True, engine="rust"), indicator_nb.ma_nb(close, 2, True), equal_nan=True
        )
        np.testing.assert_allclose(
            indicator_dispatch.mstd(close, 2, False, engine="rust"),
            indicator_nb.mstd_nb(close, 2, False),
            equal_nan=True,
        )
        np.testing.assert_allclose(
            indicator_dispatch.mstd(close, 2, True, engine="rust"),
            indicator_nb.mstd_nb(close, 2, True),
            equal_nan=True,
        )

        ma_cache = indicator_dispatch.ma_cache(close, [2, 2, 3], [False, False, True], False, engine="rust")
        ma_cache_nb = indicator_nb.ma_cache_nb(close, [2, 2, 3], [False, False, True], False)
        np.testing.assert_allclose(
            indicator_dispatch.ma_apply(close, 3, True, False, ma_cache, engine="rust"),
            indicator_nb.ma_apply_nb(close, 3, True, False, ma_cache_nb),
            equal_nan=True,
        )

        mstd_cache = indicator_dispatch.mstd_cache(close, [2, 3], [False, True], False, 0, engine="rust")
        mstd_cache_nb = indicator_nb.mstd_cache_nb(close, [2, 3], [False, True], False, 0)
        np.testing.assert_allclose(
            indicator_dispatch.mstd_apply(close, 3, True, False, 0, mstd_cache, engine="rust"),
            indicator_nb.mstd_apply_nb(close, 3, True, False, 0, mstd_cache_nb),
            equal_nan=True,
        )

        bb_cache = indicator_dispatch.bb_cache(close, [2, 3], [False, True], [2.0, 3.0], False, 0, engine="rust")
        bb_cache_nb = indicator_nb.bb_cache_nb(close, [2, 3], [False, True], [2.0, 3.0], False, 0)
        for actual, expected in zip(
            indicator_dispatch.bb_apply(close, 3, True, 2.0, False, 0, *bb_cache, engine="rust"),
            indicator_nb.bb_apply_nb(close, 3, True, 2.0, False, 0, *bb_cache_nb),
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)

        rsi_cache = indicator_dispatch.rsi_cache(close, [2, 3], [False, True], False, engine="rust")
        rsi_cache_nb = indicator_nb.rsi_cache_nb(close, [2, 3], [False, True], False)
        np.testing.assert_allclose(
            indicator_dispatch.rsi_apply(close, 3, True, False, rsi_cache, engine="rust"),
            indicator_nb.rsi_apply_nb(close, 3, True, False, rsi_cache_nb),
            equal_nan=True,
        )

        stoch_cache = indicator_dispatch.stoch_cache(
            high, low, close, [2, 3], [2, 3], [False, True], False, engine="rust"
        )
        stoch_cache_nb = indicator_nb.stoch_cache_nb(high, low, close, [2, 3], [2, 3], [False, True], False)
        for actual, expected in zip(
            indicator_dispatch.stoch_apply(high, low, close, 3, 2, True, False, stoch_cache, engine="rust"),
            indicator_nb.stoch_apply_nb(high, low, close, 3, 2, True, False, stoch_cache_nb),
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)

        macd_cache = indicator_dispatch.macd_cache(
            close, [2, 3], [3, 4], [2, 3], [False, True], [False, True], False, engine="rust"
        )
        macd_cache_nb = indicator_nb.macd_cache_nb(close, [2, 3], [3, 4], [2, 3], [False, True], [False, True], False)
        for actual, expected in zip(
            indicator_dispatch.macd_apply(close, 2, 3, 2, False, True, False, macd_cache, engine="rust"),
            indicator_nb.macd_apply_nb(close, 2, 3, 2, False, True, False, macd_cache_nb),
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)

        np.testing.assert_allclose(
            indicator_dispatch.true_range(high, low, close, engine="rust"),
            indicator_nb.true_range_nb(high, low, close),
            equal_nan=True,
        )
        atr_cache = indicator_dispatch.atr_cache(high, low, close, [2, 3], [False, True], False, engine="rust")
        atr_cache_nb = indicator_nb.atr_cache_nb(high, low, close, [2, 3], [False, True], False)
        for actual, expected in zip(
            indicator_dispatch.atr_apply(high, low, close, 3, True, False, *atr_cache, engine="rust"),
            indicator_nb.atr_apply_nb(high, low, close, 3, True, False, *atr_cache_nb),
        ):
            np.testing.assert_allclose(actual, expected, equal_nan=True)
        np.testing.assert_allclose(
            indicator_dispatch.obv_custom(close, volume, engine="rust"),
            indicator_nb.obv_custom_nb(close, volume),
            equal_nan=True,
        )

    def test_basic_indicators_match_numba(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0])
        high = close * 1.1
        low = close * 0.9
        volume = pd.Series([4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0])

        pd.testing.assert_frame_equal(
            vbt.MA.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="rust").ma,
            vbt.MA.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="numba").ma,
        )
        pd.testing.assert_frame_equal(
            vbt.MSTD.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="rust").mstd,
            vbt.MSTD.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="numba").mstd,
        )
        for attr in ("middle", "upper", "lower"):
            pd.testing.assert_frame_equal(
                getattr(
                    vbt.BBANDS.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="rust"), attr
                ),
                getattr(
                    vbt.BBANDS.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="numba"), attr
                ),
            )
        pd.testing.assert_frame_equal(
            vbt.RSI.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="rust").rsi,
            vbt.RSI.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="numba").rsi,
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
                        engine="rust",
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
                        engine="numba",
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
                        engine="rust",
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
                        engine="numba",
                    ),
                    attr,
                ),
            )
        for attr in ("tr", "atr"):
            pd.testing.assert_frame_equal(
                getattr(
                    vbt.ATR.run(high, low, close, window=(2, 3), ewm=(False, True), param_product=True, engine="rust"),
                    attr,
                ),
                getattr(
                    vbt.ATR.run(high, low, close, window=(2, 3), ewm=(False, True), param_product=True, engine="numba"),
                    attr,
                ),
            )
        pd.testing.assert_series_equal(
            vbt.OBV.run(close, volume, engine="rust").obv,
            vbt.OBV.run(close, volume, engine="numba").obv,
        )


@pytest.mark.skipif(not _engine.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
class TestLabelsRustParity:
    def test_dispatch_matches_numba(self):
        close_c = np.array(
            [
                [1.0, 5.0, 1.0],
                [2.0, 4.0, 2.0],
                [3.0, 3.0, np.nan],
                [4.0, 2.0, 2.0],
                [3.0, 1.0, 1.0],
                [2.0, 2.0, 3.0],
                [1.0, 3.0, 4.0],
                [2.0, 4.0, 5.0],
            ],
            dtype=np.float64,
        )
        close_f = np.asfortranarray(close_c)

        for close in (close_c, close_f):
            for ewm in (False, True):
                for wait in (0, 1, 2):
                    np.testing.assert_allclose(
                        labels_dispatch.future_mean_apply(close, 3, ewm, wait, False, engine="rust"),
                        labels_nb.future_mean_apply_nb(close, 3, ewm, wait, False),
                        equal_nan=True,
                    )
                    np.testing.assert_allclose(
                        labels_dispatch.future_std_apply(close, 3, ewm, wait, False, 0, engine="rust"),
                        labels_nb.future_std_apply_nb(close, 3, ewm, wait, False, 0),
                        equal_nan=True,
                    )
                np.testing.assert_allclose(
                    labels_dispatch.future_min_apply(close, 3, 1, engine="rust"),
                    labels_nb.future_min_apply_nb(close, 3, 1),
                    equal_nan=True,
                )
                np.testing.assert_allclose(
                    labels_dispatch.future_max_apply(close, 3, 1, engine="rust"),
                    labels_nb.future_max_apply_nb(close, 3, 1),
                    equal_nan=True,
                )

            for n in (1, 2, 3):
                np.testing.assert_allclose(
                    labels_dispatch.fixed_labels_apply(close, n, engine="rust"),
                    labels_nb.fixed_labels_apply_nb(close, n),
                    equal_nan=True,
                )

            np.testing.assert_allclose(
                labels_dispatch.mean_labels_apply(close, 3, False, 1, False, engine="rust"),
                labels_nb.mean_labels_apply_nb(close, 3, False, 1, False),
                equal_nan=True,
            )

            # Scalar thresholds
            pos_th = 0.1
            neg_th = 0.1
            np.testing.assert_array_equal(
                labels_dispatch.local_extrema_apply(close, pos_th, neg_th, True, engine="rust"),
                labels_nb.local_extrema_apply_nb(close, pos_th, neg_th, True),
            )

            # 1D per-column thresholds
            pos_th_col = np.array([0.05, 0.1, 0.2], dtype=np.float64)
            neg_th_col = np.array([0.1, 0.1, 0.05], dtype=np.float64)
            np.testing.assert_array_equal(
                labels_dispatch.local_extrema_apply(close, pos_th_col, neg_th_col, True, engine="rust"),
                labels_nb.local_extrema_apply_nb(close, pos_th_col, neg_th_col, True),
            )

            square_close = np.array(
                [
                    [1.02514604, 0.97357903, 1.12808453],
                    [1.04665363, 0.86927573, 1.20966649],
                    [1.31962091, 1.03393063, 1.03940950],
                ],
                dtype=np.float64,
            )
            pos_th_col = np.array([0.01, 0.08, 0.25], dtype=np.float64)
            neg_th_col = np.array([0.4, 0.25, 0.08], dtype=np.float64)
            np.testing.assert_array_equal(
                labels_dispatch.local_extrema_apply(square_close, pos_th_col, neg_th_col, True, engine="rust"),
                labels_nb.local_extrema_apply_nb(square_close, pos_th_col, neg_th_col, True),
            )
            for mode in (
                TrendMode.Binary,
                TrendMode.BinaryCont,
                TrendMode.BinaryContSat,
                TrendMode.PctChange,
                TrendMode.PctChangeNorm,
            ):
                np.testing.assert_allclose(
                    labels_dispatch.trend_labels_apply(square_close, pos_th_col, neg_th_col, mode, True, engine="rust"),
                    labels_nb.trend_labels_apply_nb(square_close, pos_th_col, neg_th_col, mode, True),
                    equal_nan=True,
                )
            for wait in (0, 1, 2):
                np.testing.assert_allclose(
                    labels_dispatch.breakout_labels(square_close, 2, pos_th_col, neg_th_col, wait, True, engine="rust"),
                    labels_nb.breakout_labels_nb(square_close, 2, pos_th_col, neg_th_col, wait, True),
                    equal_nan=True,
                )

            for mode in (
                TrendMode.Binary,
                TrendMode.BinaryCont,
                TrendMode.BinaryContSat,
                TrendMode.PctChange,
                TrendMode.PctChangeNorm,
            ):
                np.testing.assert_allclose(
                    labels_dispatch.trend_labels_apply(close, pos_th, neg_th, mode, True, engine="rust"),
                    labels_nb.trend_labels_apply_nb(close, pos_th, neg_th, mode, True),
                    equal_nan=True,
                )

            le = labels_dispatch.local_extrema_apply(close, pos_th, neg_th, True, engine="numba")
            np.testing.assert_allclose(
                labels_dispatch.bn_trend_labels(close, le, engine="rust"),
                labels_nb.bn_trend_labels_nb(close, le),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                labels_dispatch.bn_cont_trend_labels(close, le, engine="rust"),
                labels_nb.bn_cont_trend_labels_nb(close, le),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                labels_dispatch.bn_cont_sat_trend_labels(close, le, pos_th, neg_th, True, engine="rust"),
                labels_nb.bn_cont_sat_trend_labels_nb(close, le, pos_th, neg_th, True),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                labels_dispatch.pct_trend_labels(close, le, False, engine="rust"),
                labels_nb.pct_trend_labels_nb(close, le, False),
                equal_nan=True,
            )
            np.testing.assert_allclose(
                labels_dispatch.pct_trend_labels(close, le, True, engine="rust"),
                labels_nb.pct_trend_labels_nb(close, le, True),
                equal_nan=True,
            )

            for wait in (0, 1, 2):
                np.testing.assert_allclose(
                    labels_dispatch.breakout_labels(close, 3, pos_th, neg_th, wait, True, engine="rust"),
                    labels_nb.breakout_labels_nb(close, 3, pos_th, neg_th, wait, True),
                    equal_nan=True,
                )

    def test_auto_falls_back_for_bad_array(self):
        close_int = np.arange(15, dtype=np.int32).reshape(5, 3)
        np.testing.assert_array_equal(
            labels_dispatch.fixed_labels_apply(close_int, 1, engine="auto"),
            labels_nb.fixed_labels_apply_nb(close_int, 1),
        )
        with pytest.raises(ValueError, match="float64"):
            labels_dispatch.fixed_labels_apply(close_int, 1, engine="rust")

    def test_dispatch_raises_on_shape_mismatch(self):
        close = np.ones((4, 2), dtype=np.float64)
        local_extrema = np.zeros((4, 3), dtype=np.int64)
        with pytest.raises(ValueError, match="same shape"):
            labels_dispatch.bn_trend_labels(close, local_extrema, engine="rust")

    def test_basic_labels_match_numba(self):
        close = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0],
                "b": [4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0],
            }
        )

        pd.testing.assert_frame_equal(
            vbt.FMEAN.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="rust").fmean,
            vbt.FMEAN.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="numba").fmean,
        )
        pd.testing.assert_frame_equal(
            vbt.FSTD.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="rust").fstd,
            vbt.FSTD.run(close, window=(2, 3), ewm=(False, True), param_product=True, engine="numba").fstd,
        )
        pd.testing.assert_frame_equal(
            vbt.FMIN.run(close, window=(2, 3), engine="rust").fmin,
            vbt.FMIN.run(close, window=(2, 3), engine="numba").fmin,
        )
        pd.testing.assert_frame_equal(
            vbt.FMAX.run(close, window=(2, 3), engine="rust").fmax,
            vbt.FMAX.run(close, window=(2, 3), engine="numba").fmax,
        )
        pd.testing.assert_frame_equal(
            vbt.FIXLB.run(close, n=(1, 2), engine="rust").labels,
            vbt.FIXLB.run(close, n=(1, 2), engine="numba").labels,
        )
        pd.testing.assert_frame_equal(
            vbt.MEANLB.run(close, window=2, ewm=False, engine="rust").labels,
            vbt.MEANLB.run(close, window=2, ewm=False, engine="numba").labels,
        )
        pd.testing.assert_frame_equal(
            vbt.LEXLB.run(close, pos_th=0.1, neg_th=0.1, engine="rust").labels,
            vbt.LEXLB.run(close, pos_th=0.1, neg_th=0.1, engine="numba").labels,
        )
        for mode in (
            TrendMode.Binary,
            TrendMode.BinaryCont,
            TrendMode.BinaryContSat,
            TrendMode.PctChange,
            TrendMode.PctChangeNorm,
        ):
            pd.testing.assert_frame_equal(
                vbt.TRENDLB.run(close, pos_th=0.1, neg_th=0.1, mode=mode, engine="rust").labels,
                vbt.TRENDLB.run(close, pos_th=0.1, neg_th=0.1, mode=mode, engine="numba").labels,
            )
        pd.testing.assert_frame_equal(
            vbt.BOLB.run(close, window=3, pos_th=0.05, neg_th=0.05, engine="rust").labels,
            vbt.BOLB.run(close, window=3, pos_th=0.05, neg_th=0.05, engine="numba").labels,
        )


@pytest.mark.skipif(not _engine.is_rust_available(), reason="vectorbt-rust is not installed")
class TestRecordsRustParity:
    def test_col_range(self):
        col_arr = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        np.testing.assert_array_equal(
            records_dispatch.col_range(col_arr, 3, engine="rust"),
            records_nb.col_range_nb(col_arr, 3),
        )

    def test_col_range_empty_cols(self):
        col_arr = np.array([0, 0, 2, 2], dtype=np.int64)
        np.testing.assert_array_equal(
            records_dispatch.col_range(col_arr, 3, engine="rust"),
            records_nb.col_range_nb(col_arr, 3),
        )

    def test_col_range_select(self):
        col_arr = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2], dtype=np.int64)
        cr = records_nb.col_range_nb(col_arr, 3)
        new_cols = np.array([0, 2], dtype=np.int64)
        rust_idxs, rust_cols = records_dispatch.col_range_select(cr, new_cols, engine="rust")
        nb_idxs, nb_cols = records_nb.col_range_select_nb(cr, new_cols)
        np.testing.assert_array_equal(rust_idxs, nb_idxs)
        np.testing.assert_array_equal(rust_cols, nb_cols)

    def test_col_map(self):
        col_arr = np.array([0, 1, 2, 0, 1, 2, 0, 2, 2], dtype=np.int64)
        rust_idxs, rust_lens = records_dispatch.col_map(col_arr, 3, engine="rust")
        nb_idxs, nb_lens = records_nb.col_map_nb(col_arr, 3)
        np.testing.assert_array_equal(rust_idxs, nb_idxs)
        np.testing.assert_array_equal(rust_lens, nb_lens)

    def test_col_map_select(self):
        col_arr = np.array([0, 1, 2, 0, 1, 2, 0, 2, 2], dtype=np.int64)
        cm = records_nb.col_map_nb(col_arr, 3)
        new_cols = np.array([0, 2], dtype=np.int64)
        rust_idxs, rust_cols = records_dispatch.col_map_select(cm, new_cols, engine="rust")
        nb_idxs, nb_cols = records_nb.col_map_select_nb(cm, new_cols)
        np.testing.assert_array_equal(rust_idxs, nb_idxs)
        np.testing.assert_array_equal(rust_cols, nb_cols)

    def test_record_col_range_select(self):
        rec_dt = np.dtype(
            [("id", np.int64), ("col", np.int64), ("start_idx", np.int64), ("end_idx", np.int64), ("status", np.int64)],
            align=True,
        )
        records = np.array(
            [(0, 0, 10, 20, 0), (1, 0, 30, 40, 1), (2, 1, 50, 60, 0), (3, 1, 70, 80, 1), (4, 2, 90, 100, 0)],
            dtype=rec_dt,
        )
        cr = records_nb.col_range_nb(records["col"], 3)
        new_cols = np.array([0, 2], dtype=np.int64)
        np.testing.assert_array_equal(
            records_dispatch.record_col_range_select(records, cr, new_cols, engine="rust"),
            records_nb.record_col_range_select_nb(records, cr, new_cols),
        )

    def test_record_col_map_select(self):
        rec_dt = np.dtype(
            [("id", np.int64), ("col", np.int64), ("val", np.float64)],
            align=True,
        )
        records = np.array(
            [(0, 0, 1.1), (1, 2, 2.2), (2, 0, 3.3), (3, 1, 4.4), (4, 2, 5.5)],
            dtype=rec_dt,
        )
        cm = records_nb.col_map_nb(records["col"], 3)
        new_cols = np.array([0, 2], dtype=np.int64)
        np.testing.assert_array_equal(
            records_dispatch.record_col_map_select(records, cm, new_cols, engine="rust"),
            records_nb.record_col_map_select_nb(records, cm, new_cols),
        )

    def test_is_col_sorted(self):
        sorted_arr = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        unsorted_arr = np.array([0, 2, 1, 1, 0], dtype=np.int64)
        assert records_dispatch.is_col_sorted(sorted_arr, engine="rust") is True
        assert records_dispatch.is_col_sorted(unsorted_arr, engine="rust") is False

    def test_is_col_idx_sorted(self):
        col_arr = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        id_sorted = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        id_unsorted = np.array([1, 0, 0, 1, 0], dtype=np.int64)
        assert records_dispatch.is_col_idx_sorted(col_arr, id_sorted, engine="rust") is True
        assert records_dispatch.is_col_idx_sorted(col_arr, id_unsorted, engine="rust") is False

    def test_is_mapped_expandable(self):
        col_arr = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        idx_arr = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        assert records_dispatch.is_mapped_expandable(col_arr, idx_arr, (2, 3), engine="rust") is True
        # Conflicting positions
        idx_dup = np.array([0, 0, 0, 1, 0], dtype=np.int64)
        assert records_dispatch.is_mapped_expandable(col_arr, idx_dup, (2, 3), engine="rust") is False

    def test_expand_mapped(self):
        mapped_arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64)
        col_arr = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        idx_arr = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        target_shape = (2, 3)
        np.testing.assert_allclose(
            records_dispatch.expand_mapped(mapped_arr, col_arr, idx_arr, target_shape, np.nan, engine="rust"),
            records_nb.expand_mapped_nb(mapped_arr, col_arr, idx_arr, target_shape, np.nan),
            equal_nan=True,
        )

    def test_stack_expand_mapped(self):
        mapped_arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64)
        col_arr = np.array([0, 0, 0, 1, 1, 2, 2], dtype=np.int64)
        cm = records_nb.col_map_nb(col_arr, 3)
        np.testing.assert_allclose(
            records_dispatch.stack_expand_mapped(mapped_arr, cm, np.nan, engine="rust"),
            records_nb.stack_expand_mapped_nb(mapped_arr, cm, np.nan),
            equal_nan=True,
        )

    def test_mapped_value_counts(self):
        codes = np.array([0, 1, 0, 2, 1, 0, 2, 1, 0], dtype=np.int64)
        col_arr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        cm = records_nb.col_map_nb(col_arr, 3)
        np.testing.assert_array_equal(
            records_dispatch.mapped_value_counts(codes, 3, cm, engine="rust"),
            records_nb.mapped_value_counts_nb(codes, 3, cm),
        )

    def test_top_n_mapped_mask(self):
        mapped_arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=np.float64)
        col_arr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        cm = records_nb.col_map_nb(col_arr, 3)
        np.testing.assert_array_equal(
            records_dispatch.top_n_mapped_mask(mapped_arr, cm, 2, engine="rust"),
            records_nb.mapped_to_mask_nb(mapped_arr, cm, records_nb.top_n_inout_map_nb, 2),
        )

    def test_bottom_n_mapped_mask(self):
        mapped_arr = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0], dtype=np.float64)
        col_arr = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=np.int64)
        cm = records_nb.col_map_nb(col_arr, 3)
        np.testing.assert_array_equal(
            records_dispatch.bottom_n_mapped_mask(mapped_arr, cm, 2, engine="rust"),
            records_nb.mapped_to_mask_nb(mapped_arr, cm, records_nb.bottom_n_inout_map_nb, 2),
        )

    def test_empty_arrays(self):
        col_arr = np.array([], dtype=np.int64)
        cr = records_dispatch.col_range(col_arr, 0, engine="rust")
        assert cr.shape == (0, 2)
        cm_idxs, cm_lens = records_dispatch.col_map(col_arr, 0, engine="rust")
        assert cm_idxs.shape == (0,)
        assert cm_lens.shape == (0,)

    def test_single_column(self):
        col_arr = np.array([0, 0, 0], dtype=np.int64)
        np.testing.assert_array_equal(
            records_dispatch.col_range(col_arr, 1, engine="rust"),
            records_nb.col_range_nb(col_arr, 1),
        )
        rust_idxs, rust_lens = records_dispatch.col_map(col_arr, 1, engine="rust")
        nb_idxs, nb_lens = records_nb.col_map_nb(col_arr, 1)
        np.testing.assert_array_equal(rust_idxs, nb_idxs)
        np.testing.assert_array_equal(rust_lens, nb_lens)


@pytest.mark.skipif(not _engine.is_rust_available(), reason="vectorbt-rust is not installed or version-compatible")
class TestPortfolioRustParity:
    @pytest.fixture(autouse=True)
    def setup(self):
        np.random.seed(42)
        self.close = np.array(
            [[10.0, 50.0, 100.0], [11.0, 48.0, 105.0], [9.0, 52.0, 95.0], [12.0, 47.0, 110.0], [10.5, 51.0, 108.0]],
            dtype=np.float64,
        )
        self.size = np.array(
            [[1.0, -1.0, 0.5], [-0.5, 0.5, -0.5], [1.0, -1.0, 1.0], [-1.0, 1.0, -1.0], [0.0, 0.0, 0.0]],
            dtype=np.float64,
        )
        self.target_shape = (5, 3)
        self.group_lens = np.array([2, 1], dtype=np.int64)
        self.group_lens_ungrouped = np.array([1, 1, 1], dtype=np.int64)
        self.init_cash = np.array([100.0, 100.0, 100.0], dtype=np.float64)
        self.init_cash_grouped = np.array([200.0, 100.0], dtype=np.float64)
        self.call_seq = np.zeros(self.target_shape, dtype=np.int64)
        self.call_seq[:, 1] = 1
        self.call_seq[:, 2] = 0

    def test_from_orders_parity(self):
        price = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 4.0, 3.0, 2.0, 1.0],
                "c": [2.0, 3.0, 2.5, 3.5, 4.0],
            }
        )
        size = pd.Series([1.0, -1.0, np.nan, 2.0, -2.0])
        kwargs = dict(
            close=price,
            size=size,
            direction="both",
            init_cash="auto",
            fees=0.01,
            fixed_fees=0.1,
            slippage=0.01,
            min_size=0.0,
            max_size=10.0,
            size_granularity=0.5,
            log=True,
        )
        pf_numba = vbt.Portfolio.from_orders(engine="numba", **kwargs)
        pf_rust = vbt.Portfolio.from_orders(engine="rust", **kwargs)
        record_arrays_close(pf_rust.order_records, pf_numba.order_records)
        record_arrays_close(pf_rust.log_records, pf_numba.log_records)
        pd.testing.assert_series_equal(pf_rust.init_cash, pf_numba.init_cash)

    def test_from_orders_error_parity(self):
        price = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        size = pd.Series([1.0, -1.0, np.nan, 1.0, -1.0])
        for engine in ("numba", "rust"):
            with pytest.raises(ValueError, match="order.fees must be finite"):
                _ = vbt.Portfolio.from_orders(
                    price,
                    size,
                    direction="both",
                    fees=np.inf,
                    engine=engine,
                ).order_records

    def test_from_signals_parity(self):
        price = pd.DataFrame(
            {
                "a": [1.0, 2.0, 3.0, 4.0, 5.0],
                "b": [5.0, 4.0, 3.0, 2.0, 1.0],
                "c": [2.0, 3.0, 2.5, 3.5, 4.0],
            }
        )
        entries = pd.Series([True, False, False, True, False])
        exits = pd.Series([False, True, False, False, True])
        kwargs = dict(
            close=price,
            entries=entries,
            exits=exits,
            direction="both",
            init_cash="auto",
            size=1.0,
            fees=0.01,
            fixed_fees=0.1,
            slippage=0.01,
            max_size=10.0,
            size_granularity=0.5,
            log=True,
        )
        pf_numba = vbt.Portfolio.from_signals(engine="numba", **kwargs)
        pf_rust = vbt.Portfolio.from_signals(engine="rust", **kwargs)
        record_arrays_close(pf_rust.order_records, pf_numba.order_records)
        record_arrays_close(pf_rust.log_records, pf_numba.log_records)
        pd.testing.assert_series_equal(pf_rust.init_cash, pf_numba.init_cash)

    def test_from_signals_auto_cash_inf_size(self):
        price = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        entries = pd.Series([True, False, False, False, False])
        exits = pd.Series([False, True, False, False, False])

        pf_numba = vbt.Portfolio.from_signals(
            price,
            entries,
            exits,
            direction="both",
            size=1.0,
            init_cash="auto",
            engine="numba",
        )
        pf_rust = vbt.Portfolio.from_signals(
            price,
            entries,
            exits,
            direction="both",
            size=1.0,
            init_cash="auto",
            engine="rust",
        )
        record_arrays_close(pf_rust.order_records, pf_numba.order_records)
        assert pf_rust.init_cash == pf_numba.init_cash

        with pytest.raises(ValueError, match="Attempt to go in long direction infinitely"):
            _ = vbt.Portfolio.from_signals(
                price,
                entries,
                exits,
                direction="both",
                init_cash=np.inf,
                engine="rust",
            ).order_records

    def _run_sim(self, engine, **kwargs):
        call_seq = self.call_seq.copy()
        return portfolio_dispatch.simulate_from_orders(
            self.target_shape,
            self.group_lens_ungrouped,
            self.init_cash.copy(),
            call_seq,
            size=self.size.copy(),
            price=self.close.copy(),
            close=self.close.copy(),
            fees=np.full(self.target_shape, 0.001, dtype=np.float64),
            max_orders=self.target_shape[0] * self.target_shape[1],
            max_logs=0,
            flex_2d=True,
            engine=engine,
            **kwargs,
        )

    def test_simulate_from_orders_parity(self):
        or_rust, _ = self._run_sim("rust")
        or_nb, _ = self._run_sim("numba")
        assert len(or_rust) == len(or_nb)
        for field in ["id", "col", "idx", "size", "price", "fees", "side"]:
            np.testing.assert_allclose(or_rust[field], or_nb[field], err_msg=f"Mismatch in {field}")

    def test_orders_prepares_base_arrays(self):
        call_seq_rust = self.call_seq.copy()
        rust_orders, rust_logs = portfolio_dispatch.simulate_from_orders(
            self.target_shape,
            self.group_lens_ungrouped.astype(np.int32),
            self.init_cash.astype(np.float32),
            call_seq_rust,
            size=self.size.astype(np.float32),
            price=self.close.astype(np.float32),
            close=self.close.astype(np.float32),
            fees=np.full(self.target_shape, 0.001, dtype=np.float32),
            max_orders=self.target_shape[0] * self.target_shape[1],
            max_logs=0,
            flex_2d=True,
            engine="rust",
        )
        numba_orders, numba_logs = portfolio_dispatch.simulate_from_orders(
            self.target_shape,
            self.group_lens_ungrouped.astype(np.int32),
            self.init_cash.astype(np.float32),
            self.call_seq.copy(),
            size=self.size.astype(np.float32),
            price=self.close.astype(np.float32),
            close=self.close.astype(np.float32),
            fees=np.full(self.target_shape, 0.001, dtype=np.float32),
            max_orders=self.target_shape[0] * self.target_shape[1],
            max_logs=0,
            flex_2d=True,
            engine="numba",
        )
        record_arrays_close(rust_orders, numba_orders)
        record_arrays_close(rust_logs, numba_logs)

    def test_orders_require_exact_call_seq(self):
        with pytest.raises(ValueError, match="exact int64"):
            portfolio_dispatch.simulate_from_orders(
                self.target_shape,
                self.group_lens_ungrouped,
                self.init_cash.copy(),
                self.call_seq.astype(np.int32),
                size=self.size.copy(),
                price=self.close.copy(),
                close=self.close.copy(),
                max_orders=self.target_shape[0] * self.target_shape[1],
                max_logs=0,
                flex_2d=True,
                engine="rust",
            )

    def test_orders_validate_flex_inputs(self):
        with pytest.raises(ValueError, match="`size` to broadcast"):
            portfolio_dispatch.simulate_from_orders(
                self.target_shape,
                self.group_lens_ungrouped,
                self.init_cash.copy(),
                self.call_seq.copy(),
                size=np.ones((2, 2), dtype=np.float64),
                price=self.close.copy(),
                close=self.close.copy(),
                max_orders=self.target_shape[0] * self.target_shape[1],
                max_logs=0,
                flex_2d=True,
                engine="rust",
            )

        with pytest.raises(ValueError, match="`size` to be float64-compatible"):
            portfolio_dispatch.simulate_from_orders(
                self.target_shape,
                self.group_lens_ungrouped,
                self.init_cash.copy(),
                self.call_seq.copy(),
                size=np.full(self.target_shape, "x"),
                price=self.close.copy(),
                close=self.close.copy(),
                max_orders=self.target_shape[0] * self.target_shape[1],
                max_logs=0,
                flex_2d=True,
                engine="rust",
            )

    def test_signals_validate_flex_inputs(self):
        with pytest.raises(ValueError, match="`entries` to broadcast"):
            portfolio_dispatch.simulate_from_signals(
                self.target_shape,
                self.group_lens_ungrouped,
                self.init_cash.copy(),
                self.call_seq.copy(),
                entries=np.ones((2, 2), dtype=np.bool_),
                exits=np.zeros(self.target_shape, dtype=np.bool_),
                price=self.close.copy(),
                close=self.close.copy(),
                max_orders=self.target_shape[0] * self.target_shape[1],
                max_logs=0,
                flex_2d=True,
                engine="rust",
            )

    def test_simulate_from_orders_auto_call_seq(self):
        or_rust, _ = self._run_sim("rust", auto_call_seq=True)
        or_nb, _ = self._run_sim("numba", auto_call_seq=True)
        assert len(or_rust) == len(or_nb)
        for field in ["id", "col", "idx", "size", "price", "fees", "side"]:
            np.testing.assert_allclose(or_rust[field], or_nb[field], err_msg=f"Mismatch in {field}")

    def test_simulate_from_orders_cash_sharing(self):
        call_seq_cs = portfolio_nb.build_call_seq(self.target_shape, self.group_lens, call_seq_type=0)
        for engine in ("rust", "numba"):
            or_res, _ = portfolio_dispatch.simulate_from_orders(
                self.target_shape,
                self.group_lens,
                self.init_cash_grouped.copy(),
                call_seq_cs.copy(),
                size=self.size.copy(),
                price=self.close.copy(),
                close=self.close.copy(),
                fees=np.full(self.target_shape, 0.001, dtype=np.float64),
                max_orders=self.target_shape[0] * self.target_shape[1],
                max_logs=0,
                flex_2d=True,
                engine=engine,
            )
            if engine == "rust":
                or_rust = or_res
            else:
                or_nb_res = or_res
        assert len(or_rust) == len(or_nb_res)
        for field in ["id", "col", "idx", "size", "price", "fees", "side"]:
            np.testing.assert_allclose(or_rust[field], or_nb_res[field], err_msg=f"Mismatch in {field}")

    def test_approx_order_value_percent_parity(self):
        cases = [
            (0.5, SizeType.Percent, Direction.Both, 100.0, 10.0, 80.0, 2.0, 120.0),
            (-0.5, SizeType.Percent, Direction.Both, 100.0, 10.0, 80.0, 2.0, 120.0),
            (0.5, SizeType.Percent, Direction.ShortOnly, 100.0, 10.0, 80.0, 2.0, 120.0),
            (-0.5, SizeType.Percent, Direction.LongOnly, 100.0, 10.0, 80.0, 2.0, 120.0),
        ]
        for case in cases:
            np.testing.assert_allclose(
                portfolio_dispatch.approx_order_value(*case, engine="rust"),
                portfolio_nb.approx_order_value_nb(*case),
                err_msg=f"approx_order_value mismatch for {case}",
            )

    def test_orders_validate_group_lens(self):
        with pytest.raises(ValueError, match="group_lens has incorrect total number of columns"):
            portfolio_dispatch.simulate_from_orders(
                (2, 2),
                np.array([1], dtype=np.int64),
                np.array([100.0], dtype=np.float64),
                np.zeros((2, 2), dtype=np.int64),
                close=np.ones((2, 2), dtype=np.float64),
                max_orders=4,
                engine="rust",
            )

    def test_orders_zero_max_logs(self):
        target_shape = (1, 1)
        group_lens = np.array([1], dtype=np.int64)
        init_cash = np.array([100.0], dtype=np.float64)
        call_seq = np.zeros(target_shape, dtype=np.int64)
        close = np.ones(target_shape, dtype=np.float64)
        log = np.ones(target_shape, dtype=np.bool_)

        rust_orders, rust_logs = portfolio_dispatch.simulate_from_orders(
            target_shape,
            group_lens,
            init_cash,
            call_seq.copy(),
            close=close,
            log=log,
            max_orders=1,
            max_logs=0,
            engine="rust",
        )
        numba_orders, numba_logs = portfolio_nb.simulate_from_orders_nb(
            target_shape,
            group_lens,
            init_cash,
            call_seq.copy(),
            close=close,
            log=log,
            max_orders=1,
            max_logs=0,
        )
        assert len(rust_orders) == len(numba_orders)
        assert len(rust_logs) == len(numba_logs)
        for field in numba_logs.dtype.names:
            np.testing.assert_allclose(rust_logs[field], numba_logs[field], equal_nan=True, err_msg=field)

    def test_orders_raise_reject_exception(self):
        with pytest.raises(RejectedOrderError, match="Final size is less than requested"):
            portfolio_dispatch.simulate_from_orders(
                (1, 1),
                np.array([1], dtype=np.int64),
                np.array([100.0], dtype=np.float64),
                np.zeros((1, 1), dtype=np.int64),
                size=np.array([[1000.0]], dtype=np.float64),
                close=np.array([[1.0]], dtype=np.float64),
                allow_partial=np.array([[False]], dtype=np.bool_),
                raise_reject=np.array([[True]], dtype=np.bool_),
                max_orders=1,
                max_logs=1,
                engine="rust",
            )

    def test_orders_max_orders_exception(self):
        with pytest.raises(IndexError, match="order_records index out of range"):
            portfolio_dispatch.simulate_from_orders(
                (2, 1),
                np.array([1], dtype=np.int64),
                np.array([100.0], dtype=np.float64),
                np.zeros((2, 1), dtype=np.int64),
                size=np.array([[1.0], [-1.0]], dtype=np.float64),
                close=np.array([[1.0], [2.0]], dtype=np.float64),
                max_orders=1,
                max_logs=1,
                engine="rust",
            )

    def test_orders_max_logs_exception(self):
        with pytest.raises(IndexError, match="log_records index out of range"):
            portfolio_dispatch.simulate_from_orders(
                (2, 1),
                np.array([1], dtype=np.int64),
                np.array([100.0], dtype=np.float64),
                np.zeros((2, 1), dtype=np.int64),
                size=np.array([[1.0], [-1.0]], dtype=np.float64),
                close=np.array([[1.0], [2.0]], dtype=np.float64),
                log=np.array([[True], [True]], dtype=np.bool_),
                max_orders=2,
                max_logs=1,
                engine="rust",
            )

    def test_asset_flow_parity(self):
        or_nb, _ = self._run_sim("numba")
        col_map = records_nb.col_map_nb(or_nb["col"], self.target_shape[1])
        for direction in (0, 1, 2):
            rust = portfolio_dispatch.asset_flow(self.target_shape, or_nb, col_map, direction, engine="rust")
            numba = portfolio_nb.asset_flow_nb(self.target_shape, or_nb, col_map, direction)
            np.testing.assert_allclose(rust, numba, err_msg=f"asset_flow mismatch dir={direction}")

    def test_cash_flow_parity(self):
        or_nb, _ = self._run_sim("numba")
        col_map = records_nb.col_map_nb(or_nb["col"], self.target_shape[1])
        for free in (True, False):
            rust = portfolio_dispatch.cash_flow(self.target_shape, or_nb, col_map, free, engine="rust")
            numba = portfolio_nb.cash_flow_nb(self.target_shape, or_nb, col_map, free)
            np.testing.assert_allclose(rust, numba, err_msg=f"cash_flow mismatch free={free}")

    def test_derived_metric_helpers_parity(self):
        or_nb, _ = self._run_sim("numba")
        col_map = records_nb.col_map_nb(or_nb["col"], self.target_shape[1])
        cf = portfolio_nb.cash_flow_nb(self.target_shape, or_nb, col_map, False)
        cash = portfolio_nb.cash_nb(cf, self.init_cash)
        af = portfolio_nb.asset_flow_nb(self.target_shape, or_nb, col_map, 2)
        assets = portfolio_nb.assets_nb(af)
        av = portfolio_nb.asset_value_nb(self.close, assets)
        tp_ = portfolio_nb.total_profit_nb(self.target_shape, self.close, or_nb, col_map)
        random_av = np.random.rand(5, 3).astype(np.float64) * 10
        random_cash = np.random.rand(5, 3).astype(np.float64) * 100
        random_cf = np.random.randn(5, 3).astype(np.float64)

        cases = [
            (portfolio_dispatch.assets(af, engine="rust"), portfolio_nb.assets_nb(af), "assets"),
            (portfolio_dispatch.cash(cf, self.init_cash, engine="rust"), cash, "cash"),
            (portfolio_dispatch.asset_value(self.close, assets, engine="rust"), av, "asset_value"),
            (portfolio_dispatch.value(cash, av, engine="rust"), portfolio_nb.value_nb(cash, av), "value"),
            (
                portfolio_dispatch.total_profit(self.target_shape, self.close, or_nb, col_map, engine="rust"),
                tp_,
                "total_profit",
            ),
            (
                portfolio_dispatch.final_value(tp_, self.init_cash, engine="rust"),
                portfolio_nb.final_value_nb(tp_, self.init_cash),
                "final_value",
            ),
            (
                portfolio_dispatch.total_return(tp_, self.init_cash, engine="rust"),
                portfolio_nb.total_return_nb(tp_, self.init_cash),
                "total_return",
            ),
            (
                portfolio_dispatch.benchmark_value(self.close, self.init_cash, engine="rust"),
                portfolio_nb.benchmark_value_nb(self.close, self.init_cash),
                "benchmark_value",
            ),
            (
                portfolio_dispatch.gross_exposure(random_av, random_cash, engine="rust"),
                portfolio_nb.gross_exposure_nb(random_av, random_cash),
                "gross_exposure",
            ),
            (
                portfolio_dispatch.asset_returns(random_cf, random_av, engine="rust"),
                portfolio_nb.asset_returns_nb(random_cf, random_av),
                "asset_returns",
            ),
        ]
        for rust, numba, label in cases:
            np.testing.assert_allclose(rust, numba, err_msg=label)

    def test_get_entry_trades_parity(self):
        or_nb, _ = self._run_sim("numba")
        col_map = records_nb.col_map_nb(or_nb["col"], self.target_shape[1])
        rust = portfolio_dispatch.get_entry_trades(or_nb, self.close, col_map, engine="rust")
        numba = portfolio_nb.get_entry_trades_nb(or_nb, self.close, col_map)
        assert len(rust) == len(numba)
        for field in ["id", "col", "size", "entry_idx", "entry_price", "exit_idx", "exit_price", "pnl", "direction"]:
            np.testing.assert_allclose(rust[field], numba[field], err_msg=f"entry_trades mismatch: {field}")

    def test_get_exit_trades_parity(self):
        or_nb, _ = self._run_sim("numba")
        col_map = records_nb.col_map_nb(or_nb["col"], self.target_shape[1])
        rust = portfolio_dispatch.get_exit_trades(or_nb, self.close, col_map, engine="rust")
        numba = portfolio_nb.get_exit_trades_nb(or_nb, self.close, col_map)
        assert len(rust) == len(numba)
        for field in ["id", "col", "size", "entry_idx", "entry_price", "exit_idx", "exit_price", "pnl", "direction"]:
            np.testing.assert_allclose(rust[field], numba[field], err_msg=f"exit_trades mismatch: {field}")

    def test_get_positions_parity(self):
        or_nb, _ = self._run_sim("numba")
        col_map = records_nb.col_map_nb(or_nb["col"], self.target_shape[1])
        trades = portfolio_nb.get_exit_trades_nb(or_nb, self.close, col_map)
        trade_col_map = records_nb.col_map_nb(trades["col"], self.target_shape[1])
        rust = portfolio_dispatch.get_positions(trades, trade_col_map, engine="rust")
        numba = portfolio_nb.get_positions_nb(trades, trade_col_map)
        assert len(rust) == len(numba)
        for field in ["id", "col", "size", "entry_idx", "entry_price", "exit_idx", "exit_price", "pnl", "direction"]:
            np.testing.assert_allclose(rust[field], numba[field], err_msg=f"positions mismatch: {field}")

    def test_grouped_functions_parity(self):
        a = np.random.rand(5, 3).astype(np.float64)
        gl = np.array([2, 1], dtype=np.int64)
        np.testing.assert_allclose(
            portfolio_dispatch.sum_grouped(a, gl, engine="rust"),
            portfolio_nb.sum_grouped_nb(a, gl),
        )
        np.testing.assert_allclose(
            portfolio_dispatch.cash_flow_grouped(a, gl, engine="rust"),
            portfolio_nb.cash_flow_grouped_nb(a, gl),
        )
        np.testing.assert_allclose(
            portfolio_dispatch.asset_value_grouped(a, gl, engine="rust"),
            portfolio_nb.asset_value_grouped_nb(a, gl),
        )
        np.testing.assert_allclose(
            portfolio_dispatch.init_cash_grouped(self.init_cash, gl, True, engine="rust"),
            portfolio_nb.init_cash_grouped_nb(self.init_cash, gl, True),
        )
        np.testing.assert_allclose(
            portfolio_dispatch.init_cash(self.init_cash, gl, False, engine="rust"),
            portfolio_nb.init_cash_nb(self.init_cash, gl, False),
        )
        np.testing.assert_allclose(
            portfolio_dispatch.total_profit_grouped(self.init_cash, gl, engine="rust"),
            portfolio_nb.total_profit_grouped_nb(self.init_cash, gl),
        )
        np.testing.assert_allclose(
            portfolio_dispatch.benchmark_value_grouped(self.close, gl, self.init_cash_grouped, engine="rust"),
            portfolio_nb.benchmark_value_grouped_nb(self.close, gl, self.init_cash_grouped),
        )

    def test_build_call_seq_parity(self):
        for cst in (0, 1):
            np.testing.assert_array_equal(
                portfolio_dispatch.build_call_seq(self.target_shape, self.group_lens, cst, engine="rust"),
                portfolio_nb.build_call_seq_nb(self.target_shape, self.group_lens, cst),
            )

    def test_build_call_seq_random_shuffles(self):
        target_shape = (100, 4)
        group_lens = np.array([4], dtype=np.int64)
        default = portfolio_dispatch.build_call_seq(target_shape, group_lens, CallSeqType.Default, engine="rust")
        random_seq = portfolio_dispatch.build_call_seq(target_shape, group_lens, CallSeqType.Random, engine="rust")
        assert any(not np.array_equal(row, default[0]) for row in random_seq)
        for row in random_seq:
            np.testing.assert_array_equal(np.sort(row), np.array([0, 1, 2, 3]))

    def test_random_auto_numba_explicit_rust(self):
        shape = (5, 1)
        group_lens = np.array([1], dtype=np.int64)
        init_cash = np.array([100.0], dtype=np.float64)
        call_seq = np.zeros(shape, dtype=np.int64)
        close = np.arange(1.0, 6.0).reshape(shape)
        reject_prob = np.ones(shape, dtype=np.float64)

        np.random.seed(42)
        auto_orders, _ = portfolio_dispatch.simulate_from_orders(
            shape,
            group_lens,
            init_cash,
            call_seq.copy(),
            close=close,
            reject_prob=reject_prob,
            engine="auto",
        )
        np.random.seed(42)
        numba_orders, _ = portfolio_dispatch.simulate_from_orders(
            shape,
            group_lens,
            init_cash,
            call_seq.copy(),
            close=close,
            reject_prob=reject_prob,
            engine="numba",
        )
        rust_orders, _ = portfolio_dispatch.simulate_from_orders(
            shape,
            group_lens,
            init_cash,
            call_seq.copy(),
            close=close,
            reject_prob=reject_prob,
            seed=42,
            engine="rust",
        )
        assert len(rust_orders) >= 0
        np.testing.assert_array_equal(auto_orders, numba_orders)

    def test_fallback_auto(self):
        int32_close = self.close.astype(np.int32)
        result = portfolio_dispatch.assets(int32_close, engine="auto")
        assert result is not None

    def test_f_order_arrays(self):
        close_f = np.asfortranarray(self.close)
        assets = np.asfortranarray(np.ones_like(self.close))
        rust = portfolio_dispatch.asset_value(close_f, assets, engine="rust")
        numba = portfolio_nb.asset_value_nb(close_f, assets)
        np.testing.assert_allclose(rust, numba)

    def test_portfolio_end_to_end(self):
        price = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
        size = pd.Series([1.0, -1.0, 1.0, -1.0, 0.0])

        pf = vbt.Portfolio.from_orders(price, size, fees=0.01)
        assert len(pf.orders.values) == 8
        assert pf.total_profit().shape == (2,)
        assert pf.final_value().shape == (2,)
        assert pf.value().shape == (5, 2)
        assert pf.returns().shape == (5, 2)
        assert pf.asset_returns().shape == (5, 2)
        assert pf.benchmark_value().shape == (5, 2)
        assert pf.gross_exposure().shape == (5, 2)
        assert len(pf.entry_trades.values) > 0
        assert len(pf.exit_trades.values) > 0
        assert len(pf.positions.values) > 0
