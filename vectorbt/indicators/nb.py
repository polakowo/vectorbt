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

from vectorbt.generic import nb as generic_nb


@njit(cache=True)
def ma_nb(a, window, ewm, adjust=False):
    """Compute simple or exponential moving average (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_mean_nb(a, window, minp=window, adjust=adjust)
    return generic_nb.rolling_mean_nb(a, window, minp=window)


@njit(cache=True)
def mstd_nb(a, window, ewm, adjust=False, ddof=0):
    """Compute simple or exponential moving STD (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_std_nb(a, window, minp=window, adjust=adjust, ddof=ddof)
    return generic_nb.rolling_std_nb(a, window, minp=window, ddof=ddof)


@njit(cache=True)
def ma_cache_nb(close, windows, ewms, adjust):
    """Caching function for `vectorbt.indicators.basic.MA`."""
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(close, windows[i], ewms[i], adjust=adjust)
    return cache_dict


@njit(cache=True)
def ma_apply_nb(close, window, ewm, adjust, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MA`."""
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def mstd_cache_nb(close, windows, ewms, adjust, ddof):
    """Caching function for `vectorbt.indicators.basic.MSTD`."""
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = mstd_nb(close, windows[i], ewms[i], adjust=adjust, ddof=ddof)
    return cache_dict


@njit(cache=True)
def mstd_apply_nb(close, window, ewm, adjust, ddof, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MSTD`."""
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def bb_cache_nb(close, windows, ewms, alphas, adjust, ddof):
    """Caching function for `vectorbt.indicators.basic.BBANDS`."""
    ma_cache_dict = ma_cache_nb(close, windows, ewms, adjust)
    mstd_cache_dict = mstd_cache_nb(close, windows, ewms, adjust, ddof)
    return ma_cache_dict, mstd_cache_dict


@njit(cache=True)
def bb_apply_nb(close, window, ewm, alpha, adjust, ddof, ma_cache_dict, mstd_cache_dict):
    """Apply function for `vectorbt.indicators.basic.BBANDS`."""
    # Calculate lower, middle and upper bands
    h = hash((window, ewm))
    ma = np.copy(ma_cache_dict[h])
    mstd = np.copy(mstd_cache_dict[h])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


@njit(cache=True)
def rsi_cache_nb(close, windows, ewms, adjust):
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
def rsi_apply_nb(close, window, ewm, adjust, cache_dict):
    """Apply function for `vectorbt.indicators.basic.RSI`."""
    h = hash((window, ewm))
    roll_up, roll_down = cache_dict[h]
    rs = roll_up / roll_down
    return 100 - 100 / (1 + rs)


@njit(cache=True)
def stoch_cache_nb(high, low, close, k_windows, d_windows, d_ewms, adjust):
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
def stoch_apply_nb(high, low, close, k_window, d_window, d_ewm, adjust, cache_dict):
    """Apply function for `vectorbt.indicators.basic.STOCH`."""
    h = hash(k_window)
    roll_min, roll_max = cache_dict[h]
    percent_k = 100 * (close - roll_min) / (roll_max - roll_min)
    percent_d = ma_nb(percent_k, d_window, d_ewm, adjust=adjust)
    return percent_k, percent_d


@njit(cache=True)
def macd_cache_nb(close, fast_windows, slow_windows, signal_windows, macd_ewms, signal_ewms, adjust):
    """Caching function for `vectorbt.indicators.basic.MACD`."""
    windows = fast_windows.copy()
    windows.extend(slow_windows)
    ewms = macd_ewms.copy()
    ewms.extend(macd_ewms)
    return ma_cache_nb(close, windows, ewms, adjust)


@njit(cache=True)
def macd_apply_nb(close, fast_window, slow_window, signal_window, macd_ewm, signal_ewm, adjust, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MACD`."""
    fast_h = hash((fast_window, macd_ewm))
    slow_h = hash((slow_window, macd_ewm))
    fast_ma = cache_dict[fast_h]
    slow_ma = cache_dict[slow_h]
    macd_ts = fast_ma - slow_ma
    signal_ts = ma_nb(macd_ts, signal_window, signal_ewm, adjust=adjust)
    return macd_ts, signal_ts


@njit(cache=True)
def true_range_nb(high, low, close):
    """Calculate true range."""
    prev_close = generic_nb.fshift_nb(close, 1)
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.empty(prev_close.shape, dtype=np.float_)
    for col in range(tr.shape[1]):
        for i in range(tr.shape[0]):
            tr[i, col] = max(tr1[i, col], tr2[i, col], tr3[i, col])
    return tr


@njit(cache=True)
def atr_cache_nb(high, low, close, windows, ewms, adjust):
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
def atr_apply_nb(high, low, close, window, ewm, adjust, tr, cache_dict):
    """Apply function for `vectorbt.indicators.basic.ATR`."""
    h = hash((window, ewm))
    return tr, cache_dict[h]


@njit(cache=True)
def obv_custom_nb(close, volume_ts):
    """Custom calculation function for `vectorbt.indicators.basic.OBV`."""
    obv = generic_nb.set_by_mask_mult_nb(volume_ts, close < generic_nb.fshift_nb(close, 1), -volume_ts)
    obv = generic_nb.cumsum_nb(obv)
    return obv
