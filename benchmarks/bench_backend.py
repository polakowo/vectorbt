# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Benchmark Numba kernels against available Rust counterparts."""

import argparse
from dataclasses import dataclass, replace
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
from vectorbt.labels import nb as labels_nb
from vectorbt.records import nb as records_nb
from vectorbt.returns import nb as returns_nb
from vectorbt.signals import nb as signal_nb

try:
    from vectorbt_rust import generic as rust_generic
except ImportError:
    rust_generic = None

try:
    from vectorbt_rust import indicators as rust_indicators
except ImportError:
    rust_indicators = None

try:
    from vectorbt_rust import labels as rust_labels
except ImportError:
    rust_labels = None

try:
    from vectorbt_rust import returns as rust_returns
except ImportError:
    rust_returns = None

try:
    from vectorbt_rust import records as rust_records
except ImportError:
    rust_records = None

try:
    from vectorbt_rust import signals as rust_signals
except ImportError:
    rust_signals = None

try:
    from vectorbt_rust import portfolio as rust_portfolio
except ImportError:
    rust_portfolio = None

from vectorbt.portfolio import nb as portfolio_nb


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
    signal_grid = np.arange(a.size, dtype=np.int64).reshape(a.shape)
    signal_mask = np.ascontiguousarray(signal_grid % 17 == 0)
    signal_mask_1d = np.ascontiguousarray(signal_mask[:, 0])
    other_signal_mask = np.ascontiguousarray(signal_grid % 19 == 0)
    other_signal_mask_1d = np.ascontiguousarray(other_signal_mask[:, 0])
    signal_n = np.full(a.shape[1], max(1, min(a.shape[0] // 20, a.shape[0] // 2)), dtype=np.int64)
    signal_prob = np.full(a.shape, 0.05, dtype=np.float64)
    exit_prob = np.full(a.shape, 0.07, dtype=np.float64)
    signal_ts = np.nan_to_num(np.cumsum(np.nan_to_num(a, nan=0.0), axis=0) + 100.0, nan=100.0).astype(np.float64)
    signal_stop = np.full(a.shape, -0.02, dtype=np.float64)
    signal_trailing = np.ones(a.shape, dtype=np.bool_)
    signal_open = signal_ts.copy()
    signal_high = signal_ts + np.abs(np.nan_to_num(a, nan=0.0)) * 0.1 + 0.1
    signal_low = signal_ts - np.abs(np.nan_to_num(a, nan=0.0)) * 0.1 - 0.1
    signal_close = signal_ts + np.nan_to_num(a, nan=0.0) * 0.01
    returns_init_value = np.full(a.shape[1], np.nan, dtype=np.float64)
    returns_data = returns_nb.returns_nb(signal_close, returns_init_value)
    returns_value_1d = np.ascontiguousarray(signal_close[:, 0])
    returns_data_1d = np.ascontiguousarray(returns_data[:, 0])
    benchmark_rets = (np.nan_to_num(a, nan=0.0) * 0.01).astype(np.float64)
    sl_stop = np.full(a.shape, 0.02, dtype=np.float64)
    sl_trail = np.ones(a.shape, dtype=np.bool_)
    tp_stop = np.full(a.shape, 0.03, dtype=np.float64)
    reverse = np.zeros(a.shape, dtype=np.bool_)

    def signal_sig_pos_rank_nb() -> np.ndarray:
        sig_pos_temp = np.full(a.shape[1], -1, dtype=np.int64)
        return signal_nb.rank_nb(signal_mask, None, False, signal_nb.sig_pos_rank_nb, sig_pos_temp, False)

    def signal_part_pos_rank_nb() -> np.ndarray:
        part_pos_temp = np.full(a.shape[1], -1, dtype=np.int64)
        return signal_nb.rank_nb(signal_mask, None, False, signal_nb.part_pos_rank_nb, part_pos_temp)

    def signal_ohlc_stop_ex_nb() -> np.ndarray:
        stop_price_out = np.empty(a.shape, dtype=np.float64)
        stop_type_out = np.empty(a.shape, dtype=np.int64)
        return signal_nb.generate_ohlc_stop_ex_nb(
            signal_mask,
            signal_open,
            signal_high,
            signal_low,
            signal_close,
            stop_price_out,
            stop_type_out,
            sl_stop,
            sl_trail,
            tp_stop,
            reverse,
            True,
            1,
            True,
            False,
            True,
            True,
        )

    def signal_ohlc_stop_ex_rs() -> np.ndarray:
        stop_price_out = np.empty(a.shape, dtype=np.float64)
        stop_type_out = np.empty(a.shape, dtype=np.int64)
        return rust_signals.generate_ohlc_stop_ex_rs(
            signal_mask,
            signal_open,
            signal_high,
            signal_low,
            signal_close,
            stop_price_out,
            stop_type_out,
            sl_stop,
            sl_trail,
            tp_stop,
            reverse,
            True,
            1,
            True,
            False,
            True,
        )

    def signal_ohlc_stop_enex_nb() -> tuple[np.ndarray, np.ndarray]:
        stop_price_out = np.empty(a.shape, dtype=np.float64)
        stop_type_out = np.empty(a.shape, dtype=np.int64)
        return signal_nb.generate_ohlc_stop_enex_nb(
            signal_mask,
            signal_open,
            signal_high,
            signal_low,
            signal_close,
            stop_price_out,
            stop_type_out,
            sl_stop,
            sl_trail,
            tp_stop,
            reverse,
            True,
            1,
            1,
            True,
            True,
        )

    def signal_ohlc_stop_enex_rs() -> tuple[np.ndarray, np.ndarray]:
        stop_price_out = np.empty(a.shape, dtype=np.float64)
        stop_type_out = np.empty(a.shape, dtype=np.int64)
        return rust_signals.generate_ohlc_stop_enex_rs(
            signal_mask,
            signal_open,
            signal_high,
            signal_low,
            signal_close,
            stop_price_out,
            stop_type_out,
            sl_stop,
            sl_trail,
            tp_stop,
            reverse,
            True,
            1,
            1,
            True,
        )

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
                "indicators.ma",
                indicator_nb.ma_nb,
                (close, window, False, False),
                rust_indicators.ma_rs,
                (close, window, False, False),
            ),
            BenchmarkCase(
                "indicators.mstd",
                indicator_nb.mstd_nb,
                (close, window, False, False, 0),
                rust_indicators.mstd_rs,
                (close, window, False, False, 0),
            ),
            BenchmarkCase(
                "indicators.ma_cache",
                indicator_nb.ma_cache_nb,
                (close, windows, ewms, False),
                rust_indicators.ma_cache_rs,
                (close, windows, ewms, False),
            ),
            BenchmarkCase(
                "indicators.ma_apply",
                indicator_nb.ma_apply_nb,
                (close, window, True, False, nb_ma_cache),
                rust_indicators.ma_apply_rs,
                (close, window, True, False, rs_ma_cache),
            ),
            BenchmarkCase(
                "indicators.mstd_cache",
                indicator_nb.mstd_cache_nb,
                (close, windows, ewms, False, 0),
                rust_indicators.mstd_cache_rs,
                (close, windows, ewms, False, 0),
            ),
            BenchmarkCase(
                "indicators.mstd_apply",
                indicator_nb.mstd_apply_nb,
                (close, window, True, False, 0, nb_mstd_cache),
                rust_indicators.mstd_apply_rs,
                (close, window, True, False, 0, rs_mstd_cache),
            ),
            BenchmarkCase(
                "indicators.bb_cache",
                indicator_nb.bb_cache_nb,
                (close, windows, ewms, alphas, False, 0),
                rust_indicators.bb_cache_rs,
                (close, windows, ewms, alphas, False, 0),
            ),
            BenchmarkCase(
                "indicators.bb_apply",
                indicator_nb.bb_apply_nb,
                (close, window, True, 2.0, False, 0, *nb_bb_cache),
                rust_indicators.bb_apply_rs,
                (close, window, True, 2.0, False, 0, *rs_bb_cache),
            ),
            BenchmarkCase(
                "indicators.rsi_cache",
                indicator_nb.rsi_cache_nb,
                (close, windows, ewms, False),
                rust_indicators.rsi_cache_rs,
                (close, windows, ewms, False),
            ),
            BenchmarkCase(
                "indicators.rsi_apply",
                indicator_nb.rsi_apply_nb,
                (close, window, True, False, nb_rsi_cache),
                rust_indicators.rsi_apply_rs,
                (close, window, True, False, rs_rsi_cache),
            ),
            BenchmarkCase(
                "indicators.stoch_cache",
                indicator_nb.stoch_cache_nb,
                (high, low, close, windows, d_windows, d_ewms, False),
                rust_indicators.stoch_cache_rs,
                (high, low, close, windows, d_windows, d_ewms, False),
            ),
            BenchmarkCase(
                "indicators.stoch_apply",
                indicator_nb.stoch_apply_nb,
                (high, low, close, window, d_windows[-1], True, False, nb_stoch_cache),
                rust_indicators.stoch_apply_rs,
                (high, low, close, window, d_windows[-1], True, False, rs_stoch_cache),
            ),
            BenchmarkCase(
                "indicators.macd_cache",
                indicator_nb.macd_cache_nb,
                (close, fast_windows, slow_windows, signal_windows, ewms, d_ewms, False),
                rust_indicators.macd_cache_rs,
                (close, fast_windows, slow_windows, signal_windows, ewms, d_ewms, False),
            ),
            BenchmarkCase(
                "indicators.macd_apply",
                indicator_nb.macd_apply_nb,
                (close, fast_windows[0], slow_windows[0], signal_windows[0], ewms[0], d_ewms[1], False, nb_macd_cache),
                rust_indicators.macd_apply_rs,
                (close, fast_windows[0], slow_windows[0], signal_windows[0], ewms[0], d_ewms[1], False, rs_macd_cache),
            ),
            BenchmarkCase(
                "indicators.true_range",
                indicator_nb.true_range_nb,
                (high, low, close),
                rust_indicators.true_range_rs,
                (high, low, close),
            ),
            BenchmarkCase(
                "indicators.atr_cache",
                indicator_nb.atr_cache_nb,
                (high, low, close, windows, ewms, False),
                rust_indicators.atr_cache_rs,
                (high, low, close, windows, ewms, False),
            ),
            BenchmarkCase(
                "indicators.atr_apply",
                indicator_nb.atr_apply_nb,
                (high, low, close, window, True, False, *nb_atr_cache),
                rust_indicators.atr_apply_rs,
                (high, low, close, window, True, False, *rs_atr_cache),
            ),
            BenchmarkCase(
                "indicators.obv_custom",
                indicator_nb.obv_custom_nb,
                (close, volume),
                rust_indicators.obv_custom_rs,
                (close, volume),
            ),
        ]

    signal_cases = []
    if rust_signals is not None:
        signal_cases = [
            BenchmarkCase(
                "signals.clean_enex_1d",
                signal_nb.clean_enex_1d_nb,
                (signal_mask_1d, other_signal_mask_1d, True),
                rust_signals.clean_enex_1d_rs,
                (signal_mask_1d, other_signal_mask_1d, True),
            ),
            BenchmarkCase(
                "signals.clean_enex",
                signal_nb.clean_enex_nb,
                (signal_mask, other_signal_mask, True),
                rust_signals.clean_enex_rs,
                (signal_mask, other_signal_mask, True),
            ),
            BenchmarkCase(
                "signals.between_ranges",
                signal_nb.between_ranges_nb,
                (signal_mask,),
                rust_signals.between_ranges_rs,
                (signal_mask,),
            ),
            BenchmarkCase(
                "signals.between_two_ranges",
                signal_nb.between_two_ranges_nb,
                (signal_mask, other_signal_mask, False),
                rust_signals.between_two_ranges_rs,
                (signal_mask, other_signal_mask, False),
            ),
            BenchmarkCase(
                "signals.partition_ranges",
                signal_nb.partition_ranges_nb,
                (signal_mask,),
                rust_signals.partition_ranges_rs,
                (signal_mask,),
            ),
            BenchmarkCase(
                "signals.between_partition_ranges",
                signal_nb.between_partition_ranges_nb,
                (signal_mask,),
                rust_signals.between_partition_ranges_rs,
                (signal_mask,),
            ),
            BenchmarkCase(
                "signals.sig_pos_rank",
                signal_sig_pos_rank_nb,
                (),
                rust_signals.sig_pos_rank_rs,
                (signal_mask, None, False, False),
            ),
            BenchmarkCase(
                "signals.part_pos_rank",
                signal_part_pos_rank_nb,
                (),
                rust_signals.part_pos_rank_rs,
                (signal_mask, None, False),
            ),
            BenchmarkCase(
                "signals.nth_index_1d",
                signal_nb.nth_index_1d_nb,
                (signal_mask_1d, -1),
                rust_signals.nth_index_1d_rs,
                (signal_mask_1d, -1),
            ),
            BenchmarkCase(
                "signals.nth_index",
                signal_nb.nth_index_nb,
                (signal_mask, -1),
                rust_signals.nth_index_rs,
                (signal_mask, -1),
            ),
            BenchmarkCase(
                "signals.norm_avg_index_1d",
                signal_nb.norm_avg_index_1d_nb,
                (signal_mask_1d,),
                rust_signals.norm_avg_index_1d_rs,
                (signal_mask_1d,),
            ),
            BenchmarkCase(
                "signals.norm_avg_index",
                signal_nb.norm_avg_index_nb,
                (signal_mask,),
                rust_signals.norm_avg_index_rs,
                (signal_mask,),
            ),
            BenchmarkCase(
                "signals.generate_rand",
                signal_nb.generate_rand_nb,
                (a.shape, signal_n, seed),
                rust_signals.generate_rand_rs,
                (a.shape[0], a.shape[1], signal_n, seed),
                False,
            ),
            BenchmarkCase(
                "signals.generate_rand_by_prob",
                signal_nb.generate_rand_by_prob_nb,
                (a.shape, signal_prob, False, True, seed),
                rust_signals.generate_rand_by_prob_rs,
                (a.shape[0], a.shape[1], signal_prob, False, seed),
                False,
            ),
            BenchmarkCase(
                "signals.generate_rand_ex",
                signal_nb.generate_rand_ex_nb,
                (signal_mask, 1, True, False, seed),
                rust_signals.generate_rand_ex_rs,
                (signal_mask, 1, True, False, seed),
                False,
            ),
            BenchmarkCase(
                "signals.generate_rand_ex_by_prob",
                signal_nb.generate_rand_ex_by_prob_nb,
                (signal_mask, exit_prob, 1, True, False, True, seed),
                rust_signals.generate_rand_ex_by_prob_rs,
                (signal_mask, exit_prob, 1, True, False, seed),
                False,
            ),
            BenchmarkCase(
                "signals.generate_rand_enex",
                signal_nb.generate_rand_enex_nb,
                (a.shape, signal_n, 1, 1, seed),
                rust_signals.generate_rand_enex_rs,
                (a.shape[0], a.shape[1], signal_n, 1, 1, seed),
                False,
            ),
            BenchmarkCase(
                "signals.generate_rand_enex_by_prob",
                signal_nb.generate_rand_enex_by_prob_nb,
                (a.shape, signal_prob, exit_prob, 1, 1, True, True, True, seed),
                rust_signals.generate_rand_enex_by_prob_rs,
                (a.shape[0], a.shape[1], signal_prob, exit_prob, 1, 1, True, True, seed),
                False,
            ),
            BenchmarkCase(
                "signals.generate_stop_ex",
                signal_nb.generate_stop_ex_nb,
                (signal_mask, signal_ts, signal_stop, signal_trailing, 1, True, False, True, True),
                rust_signals.generate_stop_ex_rs,
                (signal_mask, signal_ts, signal_stop, signal_trailing, 1, True, False, True),
            ),
            BenchmarkCase(
                "signals.generate_stop_enex",
                signal_nb.generate_stop_enex_nb,
                (signal_mask, signal_ts, signal_stop, signal_trailing, 1, 1, True, True),
                rust_signals.generate_stop_enex_rs,
                (signal_mask, signal_ts, signal_stop, signal_trailing, 1, 1, True),
            ),
            BenchmarkCase("signals.generate_ohlc_stop_ex", signal_ohlc_stop_ex_nb, (), signal_ohlc_stop_ex_rs, ()),
            BenchmarkCase(
                "signals.generate_ohlc_stop_enex", signal_ohlc_stop_enex_nb, (), signal_ohlc_stop_enex_rs, ()
            ),
        ]

    labels_cases = []
    if rust_labels is not None:
        labels_close = signal_ts
        labels_pos_th = np.full(labels_close.shape, 0.05, dtype=np.float64)
        labels_neg_th = np.full(labels_close.shape, 0.05, dtype=np.float64)
        nb_local_extrema = labels_nb.local_extrema_apply_nb(labels_close, labels_pos_th, labels_neg_th, True)
        rs_local_extrema = rust_labels.local_extrema_apply_rs(labels_close, labels_pos_th, labels_neg_th)

        labels_cases = [
            BenchmarkCase(
                "labels.future_mean_apply",
                labels_nb.future_mean_apply_nb,
                (labels_close, window, False, 1, False),
                rust_labels.future_mean_apply_rs,
                (labels_close, window, False, 1, False),
            ),
            BenchmarkCase(
                "labels.future_std_apply",
                labels_nb.future_std_apply_nb,
                (labels_close, window, False, 1, False, 0),
                rust_labels.future_std_apply_rs,
                (labels_close, window, False, 1, False, 0),
                False,
            ),
            BenchmarkCase(
                "labels.future_min_apply",
                labels_nb.future_min_apply_nb,
                (labels_close, window, 1),
                rust_labels.future_min_apply_rs,
                (labels_close, window, 1),
            ),
            BenchmarkCase(
                "labels.future_max_apply",
                labels_nb.future_max_apply_nb,
                (labels_close, window, 1),
                rust_labels.future_max_apply_rs,
                (labels_close, window, 1),
            ),
            BenchmarkCase(
                "labels.fixed_labels_apply",
                labels_nb.fixed_labels_apply_nb,
                (labels_close, 1),
                rust_labels.fixed_labels_apply_rs,
                (labels_close, 1),
            ),
            BenchmarkCase(
                "labels.mean_labels_apply",
                labels_nb.mean_labels_apply_nb,
                (labels_close, window, False, 1, False),
                rust_labels.mean_labels_apply_rs,
                (labels_close, window, False, 1, False),
                False,
            ),
            BenchmarkCase(
                "labels.local_extrema_apply",
                labels_nb.local_extrema_apply_nb,
                (labels_close, labels_pos_th, labels_neg_th, True),
                rust_labels.local_extrema_apply_rs,
                (labels_close, labels_pos_th, labels_neg_th),
            ),
            BenchmarkCase(
                "labels.bn_trend_labels",
                labels_nb.bn_trend_labels_nb,
                (labels_close, nb_local_extrema),
                rust_labels.bn_trend_labels_rs,
                (labels_close, rs_local_extrema),
            ),
            BenchmarkCase(
                "labels.bn_cont_trend_labels",
                labels_nb.bn_cont_trend_labels_nb,
                (labels_close, nb_local_extrema),
                rust_labels.bn_cont_trend_labels_rs,
                (labels_close, rs_local_extrema),
            ),
            BenchmarkCase(
                "labels.bn_cont_sat_trend_labels",
                labels_nb.bn_cont_sat_trend_labels_nb,
                (labels_close, nb_local_extrema, labels_pos_th, labels_neg_th, True),
                rust_labels.bn_cont_sat_trend_labels_rs,
                (labels_close, rs_local_extrema, labels_pos_th, labels_neg_th),
            ),
            BenchmarkCase(
                "labels.pct_trend_labels",
                labels_nb.pct_trend_labels_nb,
                (labels_close, nb_local_extrema, False),
                rust_labels.pct_trend_labels_rs,
                (labels_close, rs_local_extrema, False),
            ),
            BenchmarkCase(
                "labels.trend_labels_apply",
                labels_nb.trend_labels_apply_nb,
                (labels_close, labels_pos_th, labels_neg_th, 0, True),
                rust_labels.trend_labels_apply_rs,
                (labels_close, labels_pos_th, labels_neg_th, 0),
            ),
            BenchmarkCase(
                "labels.breakout_labels",
                labels_nb.breakout_labels_nb,
                (labels_close, window, labels_pos_th, labels_neg_th, 1, True),
                rust_labels.breakout_labels_rs,
                (labels_close, window, labels_pos_th, labels_neg_th, 1),
            ),
        ]

    returns_cases = []
    if rust_returns is not None:
        returns_cases = [
            BenchmarkCase(
                "returns.get_return",
                returns_nb.get_return_nb,
                (1.0, 2.0),
                rust_returns.get_return_rs,
                (1.0, 2.0),
            ),
            BenchmarkCase(
                "returns.returns_1d",
                returns_nb.returns_1d_nb,
                (returns_value_1d, np.nan),
                rust_returns.returns_1d_rs,
                (returns_value_1d, np.nan),
            ),
            BenchmarkCase(
                "returns.returns",
                returns_nb.returns_nb,
                (signal_close, returns_init_value),
                rust_returns.returns_rs,
                (signal_close, returns_init_value),
            ),
            BenchmarkCase(
                "returns.cum_returns_1d",
                returns_nb.cum_returns_1d_nb,
                (returns_data_1d, 0.0),
                rust_returns.cum_returns_1d_rs,
                (returns_data_1d, 0.0),
            ),
            BenchmarkCase(
                "returns.cum_returns",
                returns_nb.cum_returns_nb,
                (returns_data, 0.0),
                rust_returns.cum_returns_rs,
                (returns_data, 0.0),
            ),
            BenchmarkCase(
                "returns.cum_returns_final_1d",
                returns_nb.cum_returns_final_1d_nb,
                (returns_data_1d, 0.0),
                rust_returns.cum_returns_final_1d_rs,
                (returns_data_1d, 0.0),
            ),
            BenchmarkCase(
                "returns.cum_returns_final",
                returns_nb.cum_returns_final_nb,
                (returns_data, 0.0),
                rust_returns.cum_returns_final_rs,
                (returns_data, 0.0),
            ),
            BenchmarkCase(
                "returns.annualized_return",
                returns_nb.annualized_return_nb,
                (returns_data, 365.0),
                rust_returns.annualized_return_rs,
                (returns_data, 365.0),
            ),
            BenchmarkCase(
                "returns.annualized_volatility",
                returns_nb.annualized_volatility_nb,
                (returns_data, 365.0, 2.0, 1),
                rust_returns.annualized_volatility_rs,
                (returns_data, 365.0, 2.0, 1),
            ),
            BenchmarkCase(
                "returns.drawdown",
                returns_nb.drawdown_nb,
                (returns_data,),
                rust_returns.drawdown_rs,
                (returns_data,),
            ),
            BenchmarkCase(
                "returns.max_drawdown",
                returns_nb.max_drawdown_nb,
                (returns_data,),
                rust_returns.max_drawdown_rs,
                (returns_data,),
            ),
            BenchmarkCase(
                "returns.calmar_ratio",
                returns_nb.calmar_ratio_nb,
                (returns_data, 365.0),
                rust_returns.calmar_ratio_rs,
                (returns_data, 365.0),
            ),
            BenchmarkCase(
                "returns.omega_ratio",
                returns_nb.omega_ratio_nb,
                (returns_data, 365.0, 0.01, 0.1),
                rust_returns.omega_ratio_rs,
                (returns_data, 365.0, 0.01, 0.1),
            ),
            BenchmarkCase(
                "returns.sharpe_ratio",
                returns_nb.sharpe_ratio_nb,
                (returns_data, 365.0, 0.01, 1),
                rust_returns.sharpe_ratio_rs,
                (returns_data, 365.0, 0.01, 1),
            ),
            BenchmarkCase(
                "returns.downside_risk",
                returns_nb.downside_risk_nb,
                (returns_data, 365.0, 0.1),
                rust_returns.downside_risk_rs,
                (returns_data, 365.0, 0.1),
            ),
            BenchmarkCase(
                "returns.sortino_ratio",
                returns_nb.sortino_ratio_nb,
                (returns_data, 365.0, 0.1),
                rust_returns.sortino_ratio_rs,
                (returns_data, 365.0, 0.1),
            ),
            BenchmarkCase(
                "returns.information_ratio",
                returns_nb.information_ratio_nb,
                (returns_data, benchmark_rets, 1),
                rust_returns.information_ratio_rs,
                (returns_data, benchmark_rets, 1),
            ),
            BenchmarkCase(
                "returns.beta",
                returns_nb.beta_nb,
                (returns_data, benchmark_rets),
                rust_returns.beta_rs,
                (returns_data, benchmark_rets),
            ),
            BenchmarkCase(
                "returns.alpha",
                returns_nb.alpha_nb,
                (returns_data, benchmark_rets, 365.0, 0.01),
                rust_returns.alpha_rs,
                (returns_data, benchmark_rets, 365.0, 0.01),
            ),
            BenchmarkCase(
                "returns.tail_ratio",
                returns_nb.tail_ratio_nb,
                (returns_data,),
                rust_returns.tail_ratio_rs,
                (returns_data,),
            ),
            BenchmarkCase(
                "returns.value_at_risk",
                returns_nb.value_at_risk_nb,
                (returns_data, 0.05),
                rust_returns.value_at_risk_rs,
                (returns_data, 0.05),
            ),
            BenchmarkCase(
                "returns.cond_value_at_risk",
                returns_nb.cond_value_at_risk_nb,
                (returns_data, 0.05),
                rust_returns.cond_value_at_risk_rs,
                (returns_data, 0.05),
            ),
            BenchmarkCase(
                "returns.capture",
                returns_nb.capture_nb,
                (returns_data, benchmark_rets, 365.0),
                rust_returns.capture_rs,
                (returns_data, benchmark_rets, 365.0),
            ),
            BenchmarkCase(
                "returns.up_capture",
                returns_nb.up_capture_nb,
                (returns_data, benchmark_rets, 365.0),
                rust_returns.up_capture_rs,
                (returns_data, benchmark_rets, 365.0),
            ),
            BenchmarkCase(
                "returns.down_capture",
                returns_nb.down_capture_nb,
                (returns_data, benchmark_rets, 365.0),
                rust_returns.down_capture_rs,
                (returns_data, benchmark_rets, 365.0),
            ),
            BenchmarkCase(
                "returns.rolling_total",
                returns_nb.rolling_cum_returns_final_nb,
                (returns_data, window, None, 0.0),
                rust_returns.rolling_cum_returns_final_rs,
                (returns_data, window, None, 0.0),
            ),
            BenchmarkCase(
                "returns.rolling_annualized",
                returns_nb.rolling_annualized_return_nb,
                (returns_data, window, None, 365.0),
                rust_returns.rolling_annualized_return_rs,
                (returns_data, window, None, 365.0),
            ),
            BenchmarkCase(
                "returns.rolling_annualized_volatility",
                returns_nb.rolling_annualized_volatility_nb,
                (returns_data, window, None, 365.0, 2.0, 1),
                rust_returns.rolling_annualized_volatility_rs,
                (returns_data, window, None, 365.0, 2.0, 1),
            ),
            BenchmarkCase(
                "returns.rolling_max_drawdown",
                returns_nb.rolling_max_drawdown_nb,
                (returns_data, window, None),
                rust_returns.rolling_max_drawdown_rs,
                (returns_data, window, None),
            ),
            BenchmarkCase(
                "returns.rolling_calmar_ratio",
                returns_nb.rolling_calmar_ratio_nb,
                (returns_data, window, None, 365.0),
                rust_returns.rolling_calmar_ratio_rs,
                (returns_data, window, None, 365.0),
            ),
            BenchmarkCase(
                "returns.rolling_omega_ratio",
                returns_nb.rolling_omega_ratio_nb,
                (returns_data, window, None, 365.0, 0.01, 0.1),
                rust_returns.rolling_omega_ratio_rs,
                (returns_data, window, None, 365.0, 0.01, 0.1),
            ),
            BenchmarkCase(
                "returns.rolling_sharpe_ratio",
                returns_nb.rolling_sharpe_ratio_nb,
                (returns_data, window, None, 365.0, 0.01, 1),
                rust_returns.rolling_sharpe_ratio_rs,
                (returns_data, window, None, 365.0, 0.01, 1),
            ),
            BenchmarkCase(
                "returns.rolling_downside_risk",
                returns_nb.rolling_downside_risk_nb,
                (returns_data, window, None, 365.0, 0.1),
                rust_returns.rolling_downside_risk_rs,
                (returns_data, window, None, 365.0, 0.1),
            ),
            BenchmarkCase(
                "returns.rolling_sortino_ratio",
                returns_nb.rolling_sortino_ratio_nb,
                (returns_data, window, None, 365.0, 0.1),
                rust_returns.rolling_sortino_ratio_rs,
                (returns_data, window, None, 365.0, 0.1),
            ),
            BenchmarkCase(
                "returns.rolling_information_ratio",
                returns_nb.rolling_information_ratio_nb,
                (returns_data, window, None, benchmark_rets, 1),
                rust_returns.rolling_information_ratio_rs,
                (returns_data, window, None, benchmark_rets, 1),
            ),
            BenchmarkCase(
                "returns.rolling_beta",
                returns_nb.rolling_beta_nb,
                (returns_data, window, None, benchmark_rets),
                rust_returns.rolling_beta_rs,
                (returns_data, window, None, benchmark_rets),
            ),
            BenchmarkCase(
                "returns.rolling_alpha",
                returns_nb.rolling_alpha_nb,
                (returns_data, window, None, benchmark_rets, 365.0, 0.01),
                rust_returns.rolling_alpha_rs,
                (returns_data, window, None, benchmark_rets, 365.0, 0.01),
            ),
            BenchmarkCase(
                "returns.rolling_tail_ratio",
                returns_nb.rolling_tail_ratio_nb,
                (returns_data, window, None),
                rust_returns.rolling_tail_ratio_rs,
                (returns_data, window, None),
            ),
            BenchmarkCase(
                "returns.rolling_value_at_risk",
                returns_nb.rolling_value_at_risk_nb,
                (returns_data, window, None, 0.05),
                rust_returns.rolling_value_at_risk_rs,
                (returns_data, window, None, 0.05),
            ),
            BenchmarkCase(
                "returns.rolling_cond_value_at_risk",
                returns_nb.rolling_cond_value_at_risk_nb,
                (returns_data, window, None, 0.05),
                rust_returns.rolling_cond_value_at_risk_rs,
                (returns_data, window, None, 0.05),
            ),
            BenchmarkCase(
                "returns.rolling_capture",
                returns_nb.rolling_capture_nb,
                (returns_data, window, None, benchmark_rets, 365.0),
                rust_returns.rolling_capture_rs,
                (returns_data, window, None, benchmark_rets, 365.0),
            ),
            BenchmarkCase(
                "returns.rolling_up_capture",
                returns_nb.rolling_up_capture_nb,
                (returns_data, window, None, benchmark_rets, 365.0),
                rust_returns.rolling_up_capture_rs,
                (returns_data, window, None, benchmark_rets, 365.0),
            ),
            BenchmarkCase(
                "returns.rolling_down_capture",
                returns_nb.rolling_down_capture_nb,
                (returns_data, window, None, benchmark_rets, 365.0),
                rust_returns.rolling_down_capture_rs,
                (returns_data, window, None, benchmark_rets, 365.0),
            ),
        ]

    # Records benchmark data
    n_records = a.shape[0] * a.shape[1]
    rec_col_arr_sorted = np.repeat(np.arange(a.shape[1], dtype=np.int64), a.shape[0])
    rec_col_arr_unsorted = np.tile(np.arange(a.shape[1], dtype=np.int64), a.shape[0])
    rec_rng = np.random.default_rng(seed)
    rec_mapped_arr = rec_rng.normal(size=n_records).astype(np.float64)
    rec_idx_arr = np.tile(np.arange(a.shape[0], dtype=np.int64), a.shape[1])
    rec_col_map = records_nb.col_map_nb(rec_col_arr_sorted, a.shape[1])
    rec_col_range = records_nb.col_range_nb(rec_col_arr_sorted, a.shape[1])
    rec_codes = np.mod(np.arange(n_records, dtype=np.int64), 5)
    rec_new_cols = np.arange(0, a.shape[1], max(1, a.shape[1] // 3), dtype=np.int64)
    rec_col_idxs, rec_col_lens = rec_col_map

    rec_struct_dt = np.dtype(
        [("id", np.int64), ("col", np.int64), ("start_idx", np.int64), ("end_idx", np.int64), ("status", np.int64)],
        align=True,
    )
    rec_records = np.empty(n_records, dtype=rec_struct_dt)
    rec_records["id"] = np.arange(n_records, dtype=np.int64)
    rec_records["col"] = rec_col_arr_sorted
    rec_records["start_idx"] = rec_rng.integers(0, 100, size=n_records, dtype=np.int64)
    rec_records["end_idx"] = rec_records["start_idx"] + rec_rng.integers(1, 50, size=n_records, dtype=np.int64)
    rec_records["status"] = rec_rng.integers(0, 2, size=n_records, dtype=np.int64)

    records_cases = []
    if rust_records is not None:
        records_cases = [
            BenchmarkCase(
                "records.col_range",
                records_nb.col_range_nb,
                (rec_col_arr_sorted, a.shape[1]),
                rust_records.col_range_rs,
                (rec_col_arr_sorted, a.shape[1]),
            ),
            BenchmarkCase(
                "records.col_range_select",
                records_nb.col_range_select_nb,
                (rec_col_range, rec_new_cols),
                rust_records.col_range_select_rs,
                (rec_col_range, rec_new_cols),
            ),
            BenchmarkCase(
                "records.col_map",
                records_nb.col_map_nb,
                (rec_col_arr_unsorted, a.shape[1]),
                rust_records.col_map_rs,
                (rec_col_arr_unsorted, a.shape[1]),
            ),
            BenchmarkCase(
                "records.col_map_select",
                records_nb.col_map_select_nb,
                (rec_col_map, rec_new_cols),
                rust_records.col_map_select_rs,
                (rec_col_idxs, rec_col_lens, rec_new_cols),
            ),
            BenchmarkCase(
                "records.is_col_sorted",
                records_nb.is_col_sorted_nb,
                (rec_col_arr_sorted,),
                rust_records.is_col_sorted_rs,
                (rec_col_arr_sorted,),
            ),
            BenchmarkCase(
                "records.is_col_idx_sorted",
                records_nb.is_col_idx_sorted_nb,
                (rec_col_arr_sorted, rec_idx_arr),
                rust_records.is_col_idx_sorted_rs,
                (rec_col_arr_sorted, rec_idx_arr),
            ),
            BenchmarkCase(
                "records.is_mapped_expandable",
                records_nb.is_mapped_expandable_nb,
                (
                    rec_col_arr_sorted[: a.shape[0] * min(a.shape[1], 3)],
                    rec_idx_arr[: a.shape[0] * min(a.shape[1], 3)],
                    (a.shape[0], min(a.shape[1], 3)),
                ),
                rust_records.is_mapped_expandable_rs,
                (
                    rec_col_arr_sorted[: a.shape[0] * min(a.shape[1], 3)],
                    rec_idx_arr[: a.shape[0] * min(a.shape[1], 3)],
                    (a.shape[0], min(a.shape[1], 3)),
                ),
            ),
            BenchmarkCase(
                "records.expand_mapped",
                records_nb.expand_mapped_nb,
                (
                    rec_mapped_arr[: a.shape[0] * min(a.shape[1], 3)],
                    rec_col_arr_sorted[: a.shape[0] * min(a.shape[1], 3)],
                    rec_idx_arr[: a.shape[0] * min(a.shape[1], 3)],
                    (a.shape[0], min(a.shape[1], 3)),
                    np.nan,
                ),
                rust_records.expand_mapped_rs,
                (
                    rec_mapped_arr[: a.shape[0] * min(a.shape[1], 3)],
                    rec_col_arr_sorted[: a.shape[0] * min(a.shape[1], 3)],
                    rec_idx_arr[: a.shape[0] * min(a.shape[1], 3)],
                    (a.shape[0], min(a.shape[1], 3)),
                    np.nan,
                ),
            ),
            BenchmarkCase(
                "records.stack_expand_mapped",
                records_nb.stack_expand_mapped_nb,
                (rec_mapped_arr, rec_col_map, np.nan),
                rust_records.stack_expand_mapped_rs,
                (rec_mapped_arr, rec_col_idxs, rec_col_lens, np.nan),
            ),
            BenchmarkCase(
                "records.mapped_value_counts",
                records_nb.mapped_value_counts_nb,
                (rec_codes, 5, rec_col_map),
                rust_records.mapped_value_counts_rs,
                (rec_codes, 5, rec_col_idxs, rec_col_lens),
            ),
            BenchmarkCase(
                "records.top_n_mapped_mask",
                lambda arr, cm, n: records_nb.mapped_to_mask_nb(arr, cm, records_nb.top_n_inout_map_nb, n),
                (rec_mapped_arr, rec_col_map, 3),
                rust_records.top_n_mapped_mask_rs,
                (rec_mapped_arr, rec_col_idxs, rec_col_lens, 3),
            ),
            BenchmarkCase(
                "records.bottom_n_mapped_mask",
                lambda arr, cm, n: records_nb.mapped_to_mask_nb(arr, cm, records_nb.bottom_n_inout_map_nb, n),
                (rec_mapped_arr, rec_col_map, 3),
                rust_records.bottom_n_mapped_mask_rs,
                (rec_mapped_arr, rec_col_idxs, rec_col_lens, 3),
            ),
            BenchmarkCase(
                "records.record_col_range_select",
                records_nb.record_col_range_select_nb,
                (rec_records, rec_col_range, rec_new_cols),
                rust_records.record_col_range_select_rs,
                (rec_records, rec_col_range, rec_new_cols),
            ),
            BenchmarkCase(
                "records.record_col_map_select",
                records_nb.record_col_map_select_nb,
                (rec_records, rec_col_map, rec_new_cols),
                rust_records.record_col_map_select_rs,
                (rec_records, rec_col_idxs, rec_col_lens, rec_new_cols),
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
    cases = [replace(case, name=f"generic.{case.name}") for case in cases]
    cases.extend(indicator_cases)
    cases.extend(signal_cases)
    cases.extend(labels_cases)
    cases.extend(records_cases)
    cases.extend(returns_cases)

    # --- Portfolio ---
    portfolio_cases = []
    if rust_portfolio is not None:
        rows, cols = a.shape
        pf_close = np.abs(a) + 1.0  # ensure positive prices
        pf_size = np.tile(np.array([1.0, -0.5, 0.5, -1.0, 0.0], dtype=np.float64), (cols, 1)).T
        if pf_size.shape[0] < rows:
            pf_size = np.resize(pf_size, (rows, cols)).astype(np.float64)
        else:
            pf_size = pf_size[:rows, :cols].copy()
        pf_target_shape = (rows, cols)
        pf_group_lens = np.ones(cols, dtype=np.int64)
        pf_init_cash = np.full(cols, 10000.0, dtype=np.float64)
        pf_call_seq = np.zeros(pf_target_shape, dtype=np.int64)
        pf_fees = np.full(pf_target_shape, 0.001, dtype=np.float64)
        pf_size_type = np.zeros(pf_target_shape, dtype=np.int64)
        pf_direction = np.full(pf_target_shape, 2, dtype=np.int64)
        pf_fixed_fees = np.zeros(pf_target_shape, dtype=np.float64)
        pf_slippage = np.zeros(pf_target_shape, dtype=np.float64)
        pf_min_size = np.zeros(pf_target_shape, dtype=np.float64)
        pf_max_size = np.full(pf_target_shape, np.inf, dtype=np.float64)
        pf_size_gran = np.full(pf_target_shape, np.nan, dtype=np.float64)
        pf_reject_prob = np.zeros(pf_target_shape, dtype=np.float64)
        pf_lock_cash = np.zeros(pf_target_shape, dtype=np.bool_)
        pf_allow_partial = np.ones(pf_target_shape, dtype=np.bool_)
        pf_raise_reject = np.zeros(pf_target_shape, dtype=np.bool_)
        pf_log = np.zeros(pf_target_shape, dtype=np.bool_)
        pf_val_price = np.full(pf_target_shape, np.inf, dtype=np.float64)
        max_orders = rows * cols

        # Pre-run Numba to get order records for post-sim benchmarks
        nb_or, _ = portfolio_nb.simulate_from_orders_nb(
            pf_target_shape,
            pf_group_lens,
            pf_init_cash.copy(),
            pf_call_seq.copy(),
            pf_size,
            pf_close,
            pf_size_type,
            pf_direction,
            pf_fees,
            pf_fixed_fees,
            pf_slippage,
            pf_min_size,
            pf_max_size,
            pf_size_gran,
            pf_reject_prob,
            pf_lock_cash,
            pf_allow_partial,
            pf_raise_reject,
            pf_log,
            pf_val_price,
            pf_close,
            False,
            True,
            False,
            max_orders,
            0,
            True,
        )
        pf_col_map = records_nb.col_map_nb(nb_or["col"], cols)
        pf_col_idxs, pf_col_lens = pf_col_map
        pf_af = portfolio_nb.asset_flow_nb(pf_target_shape, nb_or, pf_col_map, 2)
        pf_assets = portfolio_nb.assets_nb(pf_af)
        pf_cf = portfolio_nb.cash_flow_nb(pf_target_shape, nb_or, pf_col_map, False)
        pf_cash = portfolio_nb.cash_nb(pf_cf, pf_init_cash)
        pf_av = portfolio_nb.asset_value_nb(pf_close, pf_assets)

        portfolio_cases = [
            BenchmarkCase(
                "portfolio.simulate_from_orders",
                lambda: portfolio_nb.simulate_from_orders_nb(
                    pf_target_shape,
                    pf_group_lens,
                    pf_init_cash.copy(),
                    pf_call_seq.copy(),
                    pf_size,
                    pf_close,
                    pf_size_type,
                    pf_direction,
                    pf_fees,
                    pf_fixed_fees,
                    pf_slippage,
                    pf_min_size,
                    pf_max_size,
                    pf_size_gran,
                    pf_reject_prob,
                    pf_lock_cash,
                    pf_allow_partial,
                    pf_raise_reject,
                    pf_log,
                    pf_val_price,
                    pf_close,
                    False,
                    True,
                    False,
                    max_orders,
                    0,
                    True,
                ),
                (),
                lambda: rust_portfolio.simulate_from_orders_rs(
                    pf_target_shape,
                    pf_group_lens,
                    pf_init_cash.copy(),
                    pf_call_seq.copy(),
                    pf_size,
                    pf_close,
                    pf_size_type,
                    pf_direction,
                    pf_fees,
                    pf_fixed_fees,
                    pf_slippage,
                    pf_min_size,
                    pf_max_size,
                    pf_size_gran,
                    pf_reject_prob,
                    pf_lock_cash,
                    pf_allow_partial,
                    pf_raise_reject,
                    pf_log,
                    pf_val_price,
                    pf_close,
                    auto_call_seq=False,
                    ffill_val_price=True,
                    update_value=False,
                    max_orders=max_orders,
                    max_logs=0,
                ),
                (),
                check=False,
            ),
            BenchmarkCase(
                "portfolio.asset_flow",
                portfolio_nb.asset_flow_nb,
                (pf_target_shape, nb_or, pf_col_map, 2),
                rust_portfolio.asset_flow_rs,
                (nb_or, pf_col_idxs, pf_col_lens, pf_target_shape, 2),
            ),
            BenchmarkCase(
                "portfolio.cash_flow",
                portfolio_nb.cash_flow_nb,
                (pf_target_shape, nb_or, pf_col_map, False),
                rust_portfolio.cash_flow_rs,
                (nb_or, pf_col_idxs, pf_col_lens, pf_target_shape, False),
            ),
            BenchmarkCase(
                "portfolio.total_profit",
                portfolio_nb.total_profit_nb,
                (pf_target_shape, pf_close, nb_or, pf_col_map),
                rust_portfolio.total_profit_rs,
                (pf_target_shape, pf_close, nb_or, pf_col_idxs, pf_col_lens),
            ),
            BenchmarkCase(
                "portfolio.asset_value",
                portfolio_nb.asset_value_nb,
                (pf_close, pf_assets),
                rust_portfolio.asset_value_rs,
                (pf_close, pf_assets),
            ),
            BenchmarkCase(
                "portfolio.value",
                portfolio_nb.value_nb,
                (pf_cash, pf_av),
                rust_portfolio.value_rs,
                (pf_cash, pf_av),
            ),
            BenchmarkCase(
                "portfolio.gross_exposure",
                portfolio_nb.gross_exposure_nb,
                (pf_av, pf_cash),
                rust_portfolio.gross_exposure_rs,
                (pf_av, pf_cash),
            ),
            BenchmarkCase(
                "portfolio.get_entry_trades",
                portfolio_nb.get_entry_trades_nb,
                (nb_or, pf_close, pf_col_map),
                rust_portfolio.get_entry_trades_rs,
                (nb_or, pf_close, pf_col_idxs, pf_col_lens),
                check=False,
            ),
            BenchmarkCase(
                "portfolio.get_exit_trades",
                portfolio_nb.get_exit_trades_nb,
                (nb_or, pf_close, pf_col_map),
                rust_portfolio.get_exit_trades_rs,
                (nb_or, pf_close, pf_col_idxs, pf_col_lens),
                check=False,
            ),
        ]

    cases.extend(portfolio_cases)
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
