# Copyright (c) 2017-2026 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Numba-compiled functions.

Provides an arsenal of Numba-compiled functions that are used by indicator
classes. These only accept NumPy arrays and other Numba-compatible types.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument should be Numba-compiled."""

import numpy as np
from numba import njit
from numba.core.types import Type
from numba.extending import overload
from numba.np.numpy_support import as_dtype

from vectorbt import _typing as tp
from vectorbt.generic import nb as generic_nb


@njit(cache=True)
def ma_nb(a: tp.Array2d, window: int, ewm: bool, adjust: bool = False) -> tp.Array2d:
    """Compute simple or exponential moving average (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_mean_nb(a, window, minp=window, adjust=adjust)
    return generic_nb.rolling_mean_nb(a, window, minp=window)


@njit(cache=True)
def mstd_nb(a: tp.Array2d, window: int, ewm: bool, adjust: bool = False, ddof: int = 0) -> tp.Array2d:
    """Compute simple or exponential moving STD (`ewm=True`).

    With `ewm=True`, `ddof=0` uses population variance, while `ddof=1`
    applies the unbiased weighted correction."""
    if ewm:
        return generic_nb.ewm_std_nb(a, window, minp=window, adjust=adjust, ddof=ddof)
    return generic_nb.rolling_std_nb(a, window, minp=window, ddof=ddof)


@njit(cache=True)
def ma_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool,
) -> tp.Dict[int, tp.Array2d]:
    """Caching function for `vectorbt.indicators.basic.MA`."""
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(close, windows[i], ewms[i], adjust=adjust)
    return cache_dict


@njit(cache=True)
def ma_apply_nb(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool,
    cache_dict: tp.Dict[int, tp.Array2d],
) -> tp.Array2d:
    """Apply function for `vectorbt.indicators.basic.MA`."""
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def mstd_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool,
    ddof: int,
) -> tp.Dict[int, tp.Array2d]:
    """Caching function for `vectorbt.indicators.basic.MSTD`."""
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = mstd_nb(close, windows[i], ewms[i], adjust=adjust, ddof=ddof)
    return cache_dict


@njit(cache=True)
def mstd_apply_nb(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool,
    ddof: int,
    cache_dict: tp.Dict[int, tp.Array2d],
) -> tp.Array2d:
    """Apply function for `vectorbt.indicators.basic.MSTD`."""
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def bb_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    alphas: tp.List[float],
    adjust: bool,
    ddof: int,
) -> tp.Tuple[tp.Dict[int, tp.Array2d], tp.Dict[int, tp.Array2d]]:
    """Caching function for `vectorbt.indicators.basic.BBANDS`."""
    ma_cache_dict = ma_cache_nb(close, windows, ewms, adjust)
    mstd_cache_dict = mstd_cache_nb(close, windows, ewms, adjust, ddof)
    return ma_cache_dict, mstd_cache_dict


@njit(cache=True)
def bb_apply_nb(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    alpha: float,
    adjust: bool,
    ddof: int,
    ma_cache_dict: tp.Dict[int, tp.Array2d],
    mstd_cache_dict: tp.Dict[int, tp.Array2d],
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbt.indicators.basic.BBANDS`."""
    # Calculate lower, middle and upper bands
    h = hash((window, ewm))
    ma = np.copy(ma_cache_dict[h])
    mstd = np.copy(mstd_cache_dict[h])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


@njit(cache=True)
def rsi_cache_nb(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool,
) -> tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]:
    """Caching function for `vectorbt.indicators.basic.RSI`."""
    delta = generic_nb.diff_nb(close)
    up, down = delta.copy(), delta.copy()
    up = generic_nb.set_by_mask_nb(up, up < 0, 0)
    down = np.abs(generic_nb.set_by_mask_nb(down, down > 0, 0))

    # Cache
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            roll_up = ma_nb(up, windows[i], ewms[i], adjust=adjust)
            roll_down = ma_nb(down, windows[i], ewms[i], adjust=adjust)
            cache_dict[h] = roll_up, roll_down
    return cache_dict


@njit(cache=True)
def rsi_apply_nb(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool,
    cache_dict: tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]],
) -> tp.Array2d:
    """Apply function for `vectorbt.indicators.basic.RSI`."""
    h = hash((window, ewm))
    roll_up, roll_down = cache_dict[h]
    rs = roll_up / roll_down
    return 100 - 100 / (1 + rs)


@njit(cache=True)
def stoch_cache_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    k_windows: tp.List[int],
    d_windows: tp.List[int],
    d_ewms: tp.List[bool],
    adjust: bool,
) -> tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]]:
    """Caching function for `vectorbt.indicators.basic.STOCH`."""
    cache_dict = dict()
    for i in range(len(k_windows)):
        h = hash(k_windows[i])
        if h not in cache_dict:
            roll_min = generic_nb.rolling_min_nb(low, k_windows[i])
            roll_max = generic_nb.rolling_max_nb(high, k_windows[i])
            cache_dict[h] = roll_min, roll_max
    return cache_dict


@njit(cache=True)
def stoch_apply_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    k_window: int,
    d_window: int,
    d_ewm: bool,
    adjust: bool,
    cache_dict: tp.Dict[int, tp.Tuple[tp.Array2d, tp.Array2d]],
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbt.indicators.basic.STOCH`."""
    h = hash(k_window)
    roll_min, roll_max = cache_dict[h]
    percent_k = 100 * (close - roll_min) / (roll_max - roll_min)
    percent_d = ma_nb(percent_k, d_window, d_ewm, adjust=adjust)
    return percent_k, percent_d


@njit(cache=True)
def macd_cache_nb(
    close: tp.Array2d,
    fast_windows: tp.List[int],
    slow_windows: tp.List[int],
    signal_windows: tp.List[int],
    macd_ewms: tp.List[bool],
    signal_ewms: tp.List[bool],
    adjust: bool,
) -> tp.Dict[int, tp.Array2d]:
    """Caching function for `vectorbt.indicators.basic.MACD`."""
    windows = fast_windows.copy()
    windows.extend(slow_windows)
    ewms = macd_ewms.copy()
    ewms.extend(macd_ewms)
    return ma_cache_nb(close, windows, ewms, adjust)


@njit(cache=True)
def macd_apply_nb(
    close: tp.Array2d,
    fast_window: int,
    slow_window: int,
    signal_window: int,
    macd_ewm: bool,
    signal_ewm: bool,
    adjust: bool,
    cache_dict: tp.Dict[int, tp.Array2d],
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbt.indicators.basic.MACD`."""
    fast_h = hash((fast_window, macd_ewm))
    slow_h = hash((slow_window, macd_ewm))
    fast_ma = cache_dict[fast_h]
    slow_ma = cache_dict[slow_h]
    macd_ts = fast_ma - slow_ma
    signal_ts = ma_nb(macd_ts, signal_window, signal_ewm, adjust=adjust)
    return macd_ts, signal_ts


@njit(cache=True)
def _true_range_distance_nb(a: tp.Scalar, b: tp.Scalar) -> tp.Scalar:
    """Calculate a price distance without overflowing integer subtraction."""
    integer_types = (np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32)
    if isinstance(a, integer_types) and isinstance(b, integer_types):
        # Flipping the sign bit maps signed order to uint64 order without losing unit differences.
        unsigned_a = np.uint64(a) ^ np.uint64(1 << 63)
        unsigned_b = np.uint64(b) ^ np.uint64(1 << 63)
        return float(unsigned_a - unsigned_b) if unsigned_a >= unsigned_b else float(unsigned_b - unsigned_a)
    if isinstance(a, np.uint64) or isinstance(b, np.uint64):
        return abs(a - b) if a >= b else abs(b - a)
    return abs(a - b)


@njit(cache=True, inline="always")
def _true_range_from_close_nb(high: tp.Scalar, low: tp.Scalar, prev_close: tp.Scalar) -> tp.Scalar:
    """Calculate one true range while retaining integer distance precision."""
    return max(high - low, _true_range_distance_nb(high, prev_close), _true_range_distance_nb(low, prev_close))


@njit(cache=True)
def true_range_nb(high: tp.Array2d, low: tp.Array2d, close: tp.Array2d) -> tp.Array2d:
    """Calculate true range."""
    tr = np.empty(high.shape, dtype=np.float64)
    if tr.shape[0] == 0:
        return tr
    for col in range(tr.shape[1]):
        tr[0, col] = high[0, col] - low[0, col]
    if tr.shape[1] == 1:
        for i in range(1, tr.shape[0]):
            tr[i, 0] = _true_range_from_close_nb(high[i, 0], low[i, 0], close[i - 1, 0])
        return tr
    if high.flags.c_contiguous and low.flags.c_contiguous and close.flags.c_contiguous:
        # Traverse row-major prices across columns rather than rereading strided columns.
        for i in range(1, tr.shape[0]):
            for col in range(tr.shape[1]):
                tr[i, col] = _true_range_from_close_nb(high[i, col], low[i, col], close[i - 1, col])
        return tr
    if tr.shape[1] < 8:
        # Narrow output rows share cache lines, so a simple column traversal suffices.
        for col in range(tr.shape[1]):
            for i in range(1, tr.shape[0]):
                tr[i, col] = _true_range_from_close_nb(high[i, col], low[i, col], close[i - 1, col])
        return tr
    # Keep column-major reads contiguous while reusing nearby output cache lines.
    for row_start in range(1, tr.shape[0], 256):
        row_end = min(row_start + 256, tr.shape[0])
        for col in range(tr.shape[1]):
            for i in range(row_start, row_end):
                tr[i, col] = _true_range_from_close_nb(high[i, col], low[i, col], close[i - 1, col])
    return tr


@njit(cache=True)
def atr_cache_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool,
) -> tp.Tuple[tp.Array2d, tp.Dict[int, tp.Array2d]]:
    """Caching function for `vectorbt.indicators.basic.ATR`."""
    # Calculate TR here instead of re-calculating it for each param in atr_apply_nb
    tr = true_range_nb(high, low, close)
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(tr, windows[i], ewms[i], adjust=adjust)
    return tr, cache_dict


@njit(cache=True)
def atr_apply_nb(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool,
    tr: tp.Array2d,
    cache_dict: tp.Dict[int, tp.Array2d],
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbt.indicators.basic.ATR`."""
    h = hash((window, ewm))
    return tr, cache_dict[h]


def _prepare_obv_volume_nb(volume_ts):
    """Copy volumes into a signed contribution dtype, retaining floating accumulation precision."""
    nb_enabled = isinstance(volume_ts, Type)
    dtype = as_dtype(volume_ts.dtype) if nb_enabled else volume_ts.dtype
    if dtype.kind in "biu":
        # Match nancumsum's output promotion before negation, including unsigned and Boolean volumes.
        dtype = np.promote_types(dtype, int)

    def impl(volume_ts):
        return volume_ts.astype(dtype)

    if not nb_enabled:
        return impl(volume_ts)
    return impl


ol_prepare_obv_volume_nb = overload(_prepare_obv_volume_nb)(_prepare_obv_volume_nb)


@njit(cache=True)
def obv_custom_nb(close: tp.Array2d, volume_ts: tp.Array2d) -> tp.Array2d:
    """Custom calculation function for `vectorbt.indicators.basic.OBV`.

    Start with the first bar's volume. Add volume when the close rises, subtract
    it when the close falls, and leave OBV unchanged for equal finite closes.

    Compare each close with the immediately previous bar. A NaN close breaks
    the comparison chain, so the next finite close adds its full volume."""
    obv = _prepare_obv_volume_nb(volume_ts)
    # Compare original prices before a floating shift can erase integer direction.
    # Keep the first bar and non-finite comparison-chain behavior unchanged.
    for col in range(close.shape[1]):
        for i in range(1, close.shape[0]):
            if close[i, col] < close[i - 1, col]:
                obv[i, col] = -obv[i, col]
            elif np.isfinite(close[i, col]) and close[i, col] == close[i - 1, col]:
                obv[i, col] = 0
    obv = generic_nb.nancumsum_nb(obv)
    return obv
