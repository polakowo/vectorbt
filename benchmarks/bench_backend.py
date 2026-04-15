# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Benchmark Numba kernels against available Rust counterparts."""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Callable, Optional

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "vectorbt-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "vectorbt-cache"))

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vectorbt import _backend
from vectorbt.generic import nb
from vectorbt.indicators import nb as indicator_nb

try:
    from vectorbt_rust import generic as rust_generic
except ImportError:
    rust_generic = None

try:
    from vectorbt_rust import indicators as rust_indicators
except ImportError:
    rust_indicators = None


@dataclass(frozen=True)
class BenchmarkCase:
    """One Numba/Rust benchmark pair."""

    name: str
    nb_func: Callable
    nb_args: tuple
    rs_func: Callable
    rs_args: tuple
    check: bool = True


def make_array(rows: int, cols: int, nan_ratio: float, seed: int) -> np.ndarray:
    """Create deterministic float64 benchmark input."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(rows, cols)).astype(np.float64)
    if nan_ratio > 0:
        a[rng.random(size=a.shape) < nan_ratio] = np.nan
    return a


def make_cases(a: np.ndarray, window: int, seed: int) -> list[BenchmarkCase]:
    """Create all benchmark cases with available Rust counterparts."""
    if rust_generic is None:
        return []

    a_1d = np.ascontiguousarray(a[:, 0])
    other = np.full(a.shape, 0.15, dtype=np.float64)
    other_1d = np.ascontiguousarray(other[:, 0])
    mask = np.isfinite(a)
    mask_1d = np.ascontiguousarray(mask[:, 0])
    values = np.arange(a.size, dtype=np.float64).reshape(a.shape)
    values_1d = np.ascontiguousarray(values[:, 0])
    group_lens = np.array([max(1, a.shape[1] // 2), a.shape[1] - max(1, a.shape[1] // 2)], dtype=np.int64)
    if group_lens[-1] == 0:
        group_lens = np.array([a.shape[1]], dtype=np.int64)
    uniform_group_lens = np.ones(a.shape[1], dtype=np.int64)
    codes = np.mod(np.arange(a.size, dtype=np.int64).reshape(a.shape), 5)
    perc = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    range_start = np.array([0, 1, max(2, a.shape[0] // 3)], dtype=np.int64)
    range_end = np.array([max(1, a.shape[0] // 4), max(2, a.shape[0] // 2), a.shape[0] - 1], dtype=np.int64)
    range_status = np.array([0, 1, 0], dtype=np.int64)
    col_map = (np.array([0, 1, 2], dtype=np.int64), np.array([2, 1], dtype=np.int64))
    index_lens = np.array([a.shape[0], a.shape[0]], dtype=np.int64)
    peak_val = np.array([10.0, 8.0, 12.0], dtype=np.float64)
    valley_val = np.array([5.0, 4.0, 9.0], dtype=np.float64)
    end_val = np.array([8.0, 6.0, 11.0], dtype=np.float64)
    start_idx = np.array([1, 2, 3], dtype=np.int64)
    valley_idx = np.array([3, 4, 5], dtype=np.int64)
    end_idx = np.array([5, 8, 9], dtype=np.int64)
    high = a + np.abs(a) * 0.1 + 0.1
    low = a - np.abs(a) * 0.1 - 0.1
    close = a
    volume = np.nan_to_num(np.abs(a) * 100.0 + 1.0, nan=1.0).astype(np.float64)

    indicator_cases = []
    if rust_indicators is not None:
        windows = [max(2, window // 2), window]
        ewms = [False, True]
        alphas = [2.0, 3.0]
        fast_windows = [max(2, window // 3), max(3, window // 2)]
        slow_windows = [max(3, window), max(4, window + 1)]
        signal_windows = [max(2, window // 3), max(3, window // 2)]
        d_windows = [max(2, window // 3), max(3, window // 2)]
        d_ewms = [False, True]

        nb_ma_cache = indicator_nb.ma_cache_nb(close, windows, ewms, False)
        rs_ma_cache = rust_indicators.ma_cache_rs(close, windows, ewms, False)
        nb_mstd_cache = indicator_nb.mstd_cache_nb(close, windows, ewms, False, 0)
        rs_mstd_cache = rust_indicators.mstd_cache_rs(close, windows, ewms, False, 0)
        nb_bb_cache = indicator_nb.bb_cache_nb(close, windows, ewms, alphas, False, 0)
        rs_bb_cache = rust_indicators.bb_cache_rs(close, windows, ewms, alphas, False, 0)
        nb_rsi_cache = indicator_nb.rsi_cache_nb(close, windows, ewms, False)
        rs_rsi_cache = rust_indicators.rsi_cache_rs(close, windows, ewms, False)
        nb_stoch_cache = indicator_nb.stoch_cache_nb(high, low, close, windows, d_windows, d_ewms, False)
        rs_stoch_cache = rust_indicators.stoch_cache_rs(high, low, close, windows, d_windows, d_ewms, False)
        nb_macd_cache = indicator_nb.macd_cache_nb(
            close,
            fast_windows,
            slow_windows,
            signal_windows,
            ewms,
            d_ewms,
            False,
        )
        rs_macd_cache = rust_indicators.macd_cache_rs(
            close,
            fast_windows,
            slow_windows,
            signal_windows,
            ewms,
            d_ewms,
            False,
        )
        nb_atr_cache = indicator_nb.atr_cache_nb(high, low, close, windows, ewms, False)
        rs_atr_cache = rust_indicators.atr_cache_rs(high, low, close, windows, ewms, False)

        indicator_cases = [
            BenchmarkCase(
                "indicator.ma",
                indicator_nb.ma_nb,
                (close, window, False, False),
                rust_indicators.ma_rs,
                (close, window, False, False),
            ),
            BenchmarkCase(
                "indicator.mstd",
                indicator_nb.mstd_nb,
                (close, window, False, False, 0),
                rust_indicators.mstd_rs,
                (close, window, False, False, 0),
            ),
            BenchmarkCase(
                "indicator.ma_cache",
                indicator_nb.ma_cache_nb,
                (close, windows, ewms, False),
                rust_indicators.ma_cache_rs,
                (close, windows, ewms, False),
            ),
            BenchmarkCase(
                "indicator.ma_apply",
                indicator_nb.ma_apply_nb,
                (close, window, True, False, nb_ma_cache),
                rust_indicators.ma_apply_rs,
                (close, window, True, False, rs_ma_cache),
            ),
            BenchmarkCase(
                "indicator.mstd_cache",
                indicator_nb.mstd_cache_nb,
                (close, windows, ewms, False, 0),
                rust_indicators.mstd_cache_rs,
                (close, windows, ewms, False, 0),
            ),
            BenchmarkCase(
                "indicator.mstd_apply",
                indicator_nb.mstd_apply_nb,
                (close, window, True, False, 0, nb_mstd_cache),
                rust_indicators.mstd_apply_rs,
                (close, window, True, False, 0, rs_mstd_cache),
            ),
            BenchmarkCase(
                "indicator.bb_cache",
                indicator_nb.bb_cache_nb,
                (close, windows, ewms, alphas, False, 0),
                rust_indicators.bb_cache_rs,
                (close, windows, ewms, alphas, False, 0),
            ),
            BenchmarkCase(
                "indicator.bb_apply",
                indicator_nb.bb_apply_nb,
                (close, window, True, 2.0, False, 0, *nb_bb_cache),
                rust_indicators.bb_apply_rs,
                (close, window, True, 2.0, False, 0, *rs_bb_cache),
            ),
            BenchmarkCase(
                "indicator.rsi_cache",
                indicator_nb.rsi_cache_nb,
                (close, windows, ewms, False),
                rust_indicators.rsi_cache_rs,
                (close, windows, ewms, False),
            ),
            BenchmarkCase(
                "indicator.rsi_apply",
                indicator_nb.rsi_apply_nb,
                (close, window, True, False, nb_rsi_cache),
                rust_indicators.rsi_apply_rs,
                (close, window, True, False, rs_rsi_cache),
            ),
            BenchmarkCase(
                "indicator.stoch_cache",
                indicator_nb.stoch_cache_nb,
                (high, low, close, windows, d_windows, d_ewms, False),
                rust_indicators.stoch_cache_rs,
                (high, low, close, windows, d_windows, d_ewms, False),
            ),
            BenchmarkCase(
                "indicator.stoch_apply",
                indicator_nb.stoch_apply_nb,
                (high, low, close, window, d_windows[-1], True, False, nb_stoch_cache),
                rust_indicators.stoch_apply_rs,
                (high, low, close, window, d_windows[-1], True, False, rs_stoch_cache),
            ),
            BenchmarkCase(
                "indicator.macd_cache",
                indicator_nb.macd_cache_nb,
                (close, fast_windows, slow_windows, signal_windows, ewms, d_ewms, False),
                rust_indicators.macd_cache_rs,
                (close, fast_windows, slow_windows, signal_windows, ewms, d_ewms, False),
            ),
            BenchmarkCase(
                "indicator.macd_apply",
                indicator_nb.macd_apply_nb,
                (close, fast_windows[0], slow_windows[0], signal_windows[0], ewms[0], d_ewms[1], False, nb_macd_cache),
                rust_indicators.macd_apply_rs,
                (close, fast_windows[0], slow_windows[0], signal_windows[0], ewms[0], d_ewms[1], False, rs_macd_cache),
            ),
            BenchmarkCase(
                "indicator.true_range",
                indicator_nb.true_range_nb,
                (high, low, close),
                rust_indicators.true_range_rs,
                (high, low, close),
            ),
            BenchmarkCase(
                "indicator.atr_cache",
                indicator_nb.atr_cache_nb,
                (high, low, close, windows, ewms, False),
                rust_indicators.atr_cache_rs,
                (high, low, close, windows, ewms, False),
            ),
            BenchmarkCase(
                "indicator.atr_apply",
                indicator_nb.atr_apply_nb,
                (high, low, close, window, True, False, *nb_atr_cache),
                rust_indicators.atr_apply_rs,
                (high, low, close, window, True, False, *rs_atr_cache),
            ),
            BenchmarkCase(
                "indicator.obv_custom",
                indicator_nb.obv_custom_nb,
                (close, volume),
                rust_indicators.obv_custom_rs,
                (close, volume),
            ),
        ]

    cases = [
        BenchmarkCase("shuffle_1d", nb.shuffle_1d_nb, (a_1d, seed), rust_generic.shuffle_1d_rs, (a_1d, seed), False),
        BenchmarkCase("shuffle", nb.shuffle_nb, (a, seed), rust_generic.shuffle_rs, (a, seed), False),
        BenchmarkCase(
            "set_by_mask_1d",
            nb.set_by_mask_1d_nb,
            (a_1d, mask_1d, -1.0),
            rust_generic.set_by_mask_1d_rs,
            (a_1d, mask_1d, -1.0),
        ),
        BenchmarkCase("set_by_mask", nb.set_by_mask_nb, (a, mask, -1.0), rust_generic.set_by_mask_rs, (a, mask, -1.0)),
        BenchmarkCase(
            "set_by_mask_mult_1d",
            nb.set_by_mask_mult_1d_nb,
            (a_1d, mask_1d, values_1d),
            rust_generic.set_by_mask_mult_1d_rs,
            (a_1d, mask_1d, values_1d),
        ),
        BenchmarkCase(
            "set_by_mask_mult",
            nb.set_by_mask_mult_nb,
            (a, mask, values),
            rust_generic.set_by_mask_mult_rs,
            (a, mask, values),
        ),
        BenchmarkCase("fillna_1d", nb.fillna_1d_nb, (a_1d, -1.0), rust_generic.fillna_1d_rs, (a_1d, -1.0)),
        BenchmarkCase("fillna", nb.fillna_nb, (a, -1.0), rust_generic.fillna_rs, (a, -1.0)),
        BenchmarkCase("bshift_1d", nb.bshift_1d_nb, (a_1d, 2, -1.0), rust_generic.bshift_1d_rs, (a_1d, 2, -1.0)),
        BenchmarkCase("bshift", nb.bshift_nb, (a, 2, -1.0), rust_generic.bshift_rs, (a, 2, -1.0)),
        BenchmarkCase("fshift_1d", nb.fshift_1d_nb, (a_1d, 2, -1.0), rust_generic.fshift_1d_rs, (a_1d, 2, -1.0)),
        BenchmarkCase("fshift", nb.fshift_nb, (a, 2, -1.0), rust_generic.fshift_rs, (a, 2, -1.0)),
        BenchmarkCase("diff_1d", nb.diff_1d_nb, (a_1d, 1), rust_generic.diff_1d_rs, (a_1d, 1)),
        BenchmarkCase("diff", nb.diff_nb, (a, 1), rust_generic.diff_rs, (a, 1)),
        BenchmarkCase("pct_change_1d", nb.pct_change_1d_nb, (a_1d, 1), rust_generic.pct_change_1d_rs, (a_1d, 1)),
        BenchmarkCase("pct_change", nb.pct_change_nb, (a, 1), rust_generic.pct_change_rs, (a, 1)),
        BenchmarkCase("bfill_1d", nb.bfill_1d_nb, (a_1d,), rust_generic.bfill_1d_rs, (a_1d,)),
        BenchmarkCase("bfill", nb.bfill_nb, (a,), rust_generic.bfill_rs, (a,)),
        BenchmarkCase("ffill_1d", nb.ffill_1d_nb, (a_1d,), rust_generic.ffill_1d_rs, (a_1d,)),
        BenchmarkCase("ffill", nb.ffill_nb, (a,), rust_generic.ffill_rs, (a,)),
        BenchmarkCase("nanprod", nb.nanprod_nb, (a,), rust_generic.nanprod_rs, (a,)),
        BenchmarkCase("nancumsum", nb.nancumsum_nb, (a,), rust_generic.nancumsum_rs, (a,)),
        BenchmarkCase("nancumprod", nb.nancumprod_nb, (a,), rust_generic.nancumprod_rs, (a,)),
        BenchmarkCase("nansum", nb.nansum_nb, (a,), rust_generic.nansum_rs, (a,)),
        BenchmarkCase("nancnt", nb.nancnt_nb, (a,), rust_generic.nancnt_rs, (a,)),
        BenchmarkCase("nanmin", nb.nanmin_nb, (a,), rust_generic.nanmin_rs, (a,)),
        BenchmarkCase("nanmax", nb.nanmax_nb, (a,), rust_generic.nanmax_rs, (a,)),
        BenchmarkCase("nanmean", nb.nanmean_nb, (a,), rust_generic.nanmean_rs, (a,)),
        BenchmarkCase("nanmedian", nb.nanmedian_nb, (a,), rust_generic.nanmedian_rs, (a,)),
        BenchmarkCase("nanstd_1d", nb.nanstd_1d_nb, (a_1d, 0), rust_generic.nanstd_1d_rs, (a_1d, 0)),
        BenchmarkCase("nanstd", nb.nanstd_nb, (a, 0), rust_generic.nanstd_rs, (a, 0)),
        BenchmarkCase(
            "rolling_min_1d",
            nb.rolling_min_1d_nb,
            (a_1d, window, None),
            rust_generic.rolling_min_1d_rs,
            (a_1d, window, None),
        ),
        BenchmarkCase(
            "rolling_min", nb.rolling_min_nb, (a, window, None), rust_generic.rolling_min_rs, (a, window, None)
        ),
        BenchmarkCase(
            "rolling_max_1d",
            nb.rolling_max_1d_nb,
            (a_1d, window, None),
            rust_generic.rolling_max_1d_rs,
            (a_1d, window, None),
        ),
        BenchmarkCase(
            "rolling_max", nb.rolling_max_nb, (a, window, None), rust_generic.rolling_max_rs, (a, window, None)
        ),
        BenchmarkCase(
            "rolling_mean_1d",
            nb.rolling_mean_1d_nb,
            (a_1d, window, None),
            rust_generic.rolling_mean_1d_rs,
            (a_1d, window, None),
        ),
        BenchmarkCase(
            "rolling_mean", nb.rolling_mean_nb, (a, window, None), rust_generic.rolling_mean_rs, (a, window, None)
        ),
        BenchmarkCase(
            "rolling_std_1d",
            nb.rolling_std_1d_nb,
            (a_1d, window, None, 0),
            rust_generic.rolling_std_1d_rs,
            (a_1d, window, None, 0),
        ),
        BenchmarkCase(
            "rolling_std", nb.rolling_std_nb, (a, window, None, 0), rust_generic.rolling_std_rs, (a, window, None, 0)
        ),
        BenchmarkCase(
            "ewm_mean_1d",
            nb.ewm_mean_1d_nb,
            (a_1d, window, 0, False),
            rust_generic.ewm_mean_1d_rs,
            (a_1d, window, 0, False),
        ),
        BenchmarkCase(
            "ewm_mean", nb.ewm_mean_nb, (a, window, 0, False), rust_generic.ewm_mean_rs, (a, window, 0, False)
        ),
        BenchmarkCase(
            "ewm_std_1d",
            nb.ewm_std_1d_nb,
            (a_1d, window, 0, False, 0),
            rust_generic.ewm_std_1d_rs,
            (a_1d, window, 0, False, 0),
        ),
        BenchmarkCase(
            "ewm_std", nb.ewm_std_nb, (a, window, 0, False, 0), rust_generic.ewm_std_rs, (a, window, 0, False, 0)
        ),
        BenchmarkCase(
            "expanding_min_1d", nb.expanding_min_1d_nb, (a_1d, 1), rust_generic.expanding_min_1d_rs, (a_1d, 1)
        ),
        BenchmarkCase("expanding_min", nb.expanding_min_nb, (a, 1), rust_generic.expanding_min_rs, (a, 1)),
        BenchmarkCase(
            "expanding_max_1d", nb.expanding_max_1d_nb, (a_1d, 1), rust_generic.expanding_max_1d_rs, (a_1d, 1)
        ),
        BenchmarkCase("expanding_max", nb.expanding_max_nb, (a, 1), rust_generic.expanding_max_rs, (a, 1)),
        BenchmarkCase(
            "expanding_mean_1d", nb.expanding_mean_1d_nb, (a_1d, 1), rust_generic.expanding_mean_1d_rs, (a_1d, 1)
        ),
        BenchmarkCase("expanding_mean", nb.expanding_mean_nb, (a, 1), rust_generic.expanding_mean_rs, (a, 1)),
        BenchmarkCase(
            "expanding_std_1d", nb.expanding_std_1d_nb, (a_1d, 1, 0), rust_generic.expanding_std_1d_rs, (a_1d, 1, 0)
        ),
        BenchmarkCase("expanding_std", nb.expanding_std_nb, (a, 1, 0), rust_generic.expanding_std_rs, (a, 1, 0)),
        BenchmarkCase("flatten_forder", nb.flatten_forder_nb, (a,), rust_generic.flatten_forder_rs, (a,)),
        BenchmarkCase(
            "flatten_grouped",
            nb.flatten_grouped_nb,
            (a, group_lens, False),
            rust_generic.flatten_grouped_rs,
            (a, group_lens, False),
        ),
        BenchmarkCase(
            "flatten_uniform_grouped",
            nb.flatten_uniform_grouped_nb,
            (a, uniform_group_lens, True),
            rust_generic.flatten_uniform_grouped_rs,
            (a, uniform_group_lens, True),
        ),
        BenchmarkCase("nth_reduce", nb.nth_reduce_nb, (0, a_1d, -1), rust_generic.nth_reduce_rs, (a_1d, -1)),
        BenchmarkCase(
            "nth_index_reduce", nb.nth_index_reduce_nb, (0, a_1d, -1), rust_generic.nth_index_reduce_rs, (a_1d, -1)
        ),
        BenchmarkCase("min_reduce", nb.min_reduce_nb, (0, a_1d), rust_generic.min_reduce_rs, (a_1d,)),
        BenchmarkCase("max_reduce", nb.max_reduce_nb, (0, a_1d), rust_generic.max_reduce_rs, (a_1d,)),
        BenchmarkCase("mean_reduce", nb.mean_reduce_nb, (0, a_1d), rust_generic.mean_reduce_rs, (a_1d,)),
        BenchmarkCase("median_reduce", nb.median_reduce_nb, (0, a_1d), rust_generic.median_reduce_rs, (a_1d,)),
        BenchmarkCase("std_reduce", nb.std_reduce_nb, (0, a_1d, 0), rust_generic.std_reduce_rs, (a_1d, 0)),
        BenchmarkCase("sum_reduce", nb.sum_reduce_nb, (0, a_1d), rust_generic.sum_reduce_rs, (a_1d,)),
        BenchmarkCase("count_reduce", nb.count_reduce_nb, (0, a_1d), rust_generic.count_reduce_rs, (a_1d,)),
        BenchmarkCase("argmin_reduce", nb.argmin_reduce_nb, (0, a_1d), rust_generic.argmin_reduce_rs, (a_1d,)),
        BenchmarkCase("argmax_reduce", nb.argmax_reduce_nb, (0, a_1d), rust_generic.argmax_reduce_rs, (a_1d,)),
        BenchmarkCase(
            "describe_reduce",
            nb.describe_reduce_nb,
            (0, a_1d, perc, 0),
            rust_generic.describe_reduce_rs,
            (a_1d, perc, 0),
        ),
        BenchmarkCase(
            "value_counts",
            nb.value_counts_nb,
            (codes, 5, group_lens),
            rust_generic.value_counts_rs,
            (codes, 5, group_lens),
        ),
        BenchmarkCase("min_squeeze", nb.min_squeeze_nb, (0, 0, a_1d), rust_generic.min_squeeze_rs, (a_1d,)),
        BenchmarkCase("max_squeeze", nb.max_squeeze_nb, (0, 0, a_1d), rust_generic.max_squeeze_rs, (a_1d,)),
        BenchmarkCase("sum_squeeze", nb.sum_squeeze_nb, (0, 0, a_1d), rust_generic.sum_squeeze_rs, (a_1d,)),
        BenchmarkCase("any_squeeze", nb.any_squeeze_nb, (0, 0, a_1d), rust_generic.any_squeeze_rs, (a_1d,)),
        BenchmarkCase("find_ranges", nb.find_ranges_nb, (a, np.nan), rust_generic.find_ranges_rs, (a, np.nan)),
        BenchmarkCase(
            "range_duration",
            nb.range_duration_nb,
            (range_start, range_end, range_status),
            rust_generic.range_duration_rs,
            (range_start, range_end, range_status),
        ),
        BenchmarkCase(
            "range_coverage",
            nb.range_coverage_nb,
            (range_start, range_end, range_status, col_map, index_lens, False, False),
            rust_generic.range_coverage_rs,
            (range_start, range_end, range_status, col_map, index_lens, False, False),
        ),
        BenchmarkCase(
            "ranges_to_mask",
            nb.ranges_to_mask_nb,
            (range_start, range_end, range_status, col_map, a.shape[0]),
            rust_generic.ranges_to_mask_rs,
            (range_start, range_end, range_status, col_map, a.shape[0]),
        ),
        BenchmarkCase("get_drawdowns", nb.get_drawdowns_nb, (a,), rust_generic.get_drawdowns_rs, (a,)),
        BenchmarkCase(
            "dd_drawdown",
            nb.dd_drawdown_nb,
            (peak_val, valley_val),
            rust_generic.dd_drawdown_rs,
            (peak_val, valley_val),
        ),
        BenchmarkCase(
            "dd_decline_duration",
            nb.dd_decline_duration_nb,
            (start_idx, valley_idx),
            rust_generic.dd_decline_duration_rs,
            (start_idx, valley_idx),
        ),
        BenchmarkCase(
            "dd_recovery_duration",
            nb.dd_recovery_duration_nb,
            (valley_idx, end_idx),
            rust_generic.dd_recovery_duration_rs,
            (valley_idx, end_idx),
        ),
        BenchmarkCase(
            "dd_recovery_duration_ratio",
            nb.dd_recovery_duration_ratio_nb,
            (start_idx, valley_idx, end_idx),
            rust_generic.dd_recovery_duration_ratio_rs,
            (start_idx, valley_idx, end_idx),
        ),
        BenchmarkCase(
            "dd_recovery_return",
            nb.dd_recovery_return_nb,
            (valley_val, end_val),
            rust_generic.dd_recovery_return_rs,
            (valley_val, end_val),
        ),
        BenchmarkCase(
            "crossed_above_1d",
            nb.crossed_above_1d_nb,
            (a_1d, other_1d, 0),
            rust_generic.crossed_above_1d_rs,
            (a_1d, other_1d, 0),
        ),
        BenchmarkCase(
            "crossed_above", nb.crossed_above_nb, (a, other, 0), rust_generic.crossed_above_rs, (a, other, 0)
        ),
    ]
    cases.extend(indicator_cases)
    return cases


def time_call(func: Callable, args: tuple, repeat: int, warmup: int) -> float:
    """Return best runtime in seconds."""
    for _ in range(warmup):
        func(*args)
    best = np.inf
    for _ in range(repeat):
        start = time.perf_counter()
        func(*args)
        best = min(best, time.perf_counter() - start)
    return best


def assert_same(case: BenchmarkCase) -> None:
    """Verify that Rust and Numba return equivalent results."""
    if not case.check:
        return
    nb_out = case.nb_func(*case.nb_args)
    rs_out = case.rs_func(*case.rs_args)
    assert_nested_same(rs_out, nb_out)


def assert_nested_same(actual, expected) -> None:
    """Compare arrays, scalars, tuples, and cache dictionaries recursively."""
    if isinstance(actual, dict) or hasattr(actual, "keys"):
        actual_keys = set(actual.keys())
        expected_keys = set(expected.keys())
        assert actual_keys == expected_keys
        for key in actual_keys:
            assert_nested_same(actual[key], expected[key])
        return
    if isinstance(actual, tuple) or isinstance(expected, tuple):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            assert_nested_same(actual_item, expected_item)
        return
    if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
        if getattr(actual.dtype, "fields", None) is not None or getattr(expected.dtype, "fields", None) is not None:
            np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-12, equal_nan=True)
        return
    np.testing.assert_allclose(actual, expected, equal_nan=True)


def main() -> None:
    """Run benchmarks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--cols", type=int, default=50)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--nan-ratio", type=float, default=0.05)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pattern", type=str, default=None, help="Only run cases whose name contains this string.")
    parser.add_argument("--check", action="store_true", help="Check Rust/Numba output parity before timing.")
    args = parser.parse_args()

    if not _backend.is_rust_available() or rust_generic is None:
        raise SystemExit("vectorbt-rust is not installed or version-compatible")

    a = make_array(args.rows, args.cols, args.nan_ratio, args.seed)
    cases = make_cases(a, args.window, args.seed)
    if args.pattern is not None:
        cases = [case for case in cases if args.pattern in case.name]

    print("function,numba_s,rust_s,speedup")
    for case in cases:
        if args.check:
            assert_same(case)
        numba_s = time_call(case.nb_func, case.nb_args, args.repeat, args.warmup)
        rust_s = time_call(case.rs_func, case.rs_args, args.repeat, args.warmup)
        print(f"{case.name},{numba_s:.6f},{rust_s:.6f},{numba_s / rust_s:.3f}")


if __name__ == "__main__":
    main()
