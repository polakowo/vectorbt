"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument should be Numba-compiled."""

import numpy as np
from numba import njit

from vectorbt.generic import nb as generic_nb


@njit(cache=True)
def ma_nb(a, window, ewm):
    """Compute simple or exponential moving average (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_mean_nb(a, window, minp=window, adjust=False)
    return generic_nb.rolling_mean_nb(a, window, minp=window)


@njit(cache=True)
def mstd_nb(a, window, ewm):
    """Compute simple or exponential moving STD (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_std_nb(a, window, minp=window, adjust=False, ddof=0)
    return generic_nb.rolling_std_nb(a, window, minp=window, ddof=0)


@njit(cache=True)
def ma_cache_nb(ts, windows, ewms):
    """Caching function for `vectorbt.indicators.basic.MA`."""
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(ts, windows[i], ewms[i])
    return cache_dict


@njit(cache=True)
def ma_apply_nb(ts, window, ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MA`."""
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def mstd_cache_nb(ts, windows, ewms):
    """Caching function for `vectorbt.indicators.basic.MSTD`."""
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = mstd_nb(ts, windows[i], ewms[i])
    return cache_dict


@njit(cache=True)
def mstd_apply_nb(ts, window, ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MSTD`."""
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def bb_cache_nb(ts, windows, ewms, alphas):
    """Caching function for `vectorbt.indicators.basic.BBANDS`."""
    ma_cache_dict = ma_cache_nb(ts, windows, ewms)
    mstd_cache_dict = mstd_cache_nb(ts, windows, ewms)
    return ma_cache_dict, mstd_cache_dict


@njit(cache=True)
def bb_apply_nb(ts, window, ewm, alpha, ma_cache_dict, mstd_cache_dict):
    """Apply function for `vectorbt.indicators.basic.BBANDS`."""
    # Calculate lower, middle and upper bands
    h = hash((window, ewm))
    ma = np.copy(ma_cache_dict[h])
    mstd = np.copy(mstd_cache_dict[h])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


@njit(cache=True)
def rsi_cache_nb(ts, windows, ewms):
    """Caching function for `vectorbt.indicators.basic.RSI`."""
    delta = generic_nb.diff_nb(ts)  # otherwise ewma will be all NaN
    up, down = delta.copy(), delta.copy()
    up = generic_nb.set_by_mask_nb(up, up < 0, 0)
    down = np.abs(generic_nb.set_by_mask_nb(down, down > 0, 0))
    # Cache
    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            roll_up = ma_nb(up, windows[i], ewms[i])
            roll_down = ma_nb(down, windows[i], ewms[i])
            cache_dict[h] = roll_up, roll_down
    return cache_dict


@njit(cache=True)
def rsi_apply_nb(ts, window, ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.RSI`."""
    h = hash((window, ewm))
    roll_up, roll_down = cache_dict[h]
    rs = roll_up / roll_down
    return 100 - 100 / (1 + rs)


@njit(cache=True)
def stoch_cache_nb(high_ts, low_ts, close_ts, k_windows, d_windows, d_ewms):
    """Caching function for `vectorbt.indicators.basic.STOCH`."""
    cache_dict = dict()
    for i in range(len(k_windows)):
        h = hash(k_windows[i])
        if h not in cache_dict:
            roll_min = generic_nb.rolling_min_nb(low_ts, k_windows[i])
            roll_max = generic_nb.rolling_max_nb(high_ts, k_windows[i])
            cache_dict[h] = roll_min, roll_max
    return cache_dict


@njit(cache=True)
def stoch_apply_nb(high_ts, low_ts, close_ts, k_window, d_window, d_ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.STOCH`."""
    h = hash(k_window)
    roll_min, roll_max = cache_dict[h]
    percent_k = 100 * (close_ts - roll_min) / (roll_max - roll_min)
    percent_d = ma_nb(percent_k, d_window, d_ewm)
    return percent_k, percent_d


@njit(cache=True)
def macd_cache_nb(ts, fast_windows, slow_windows, signal_windows, macd_ewms, signal_ewms):
    """Caching function for `vectorbt.indicators.basic.MACD`."""
    return ma_cache_nb(ts, fast_windows + slow_windows, macd_ewms + macd_ewms)


@njit(cache=True)
def macd_apply_nb(ts, fast_window, slow_window, signal_window, macd_ewm, signal_ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MACD`."""
    fast_h = hash((fast_window, macd_ewm))
    slow_h = hash((slow_window, macd_ewm))
    fast_ma = cache_dict[fast_h]
    slow_ma = cache_dict[slow_h]
    macd_ts = fast_ma - slow_ma
    signal_ts = ma_nb(macd_ts, signal_window, signal_ewm)
    return macd_ts, signal_ts


@njit(cache=True)
def nanmax_cube_nb(a):
    """Return max of a cube by reducing the axis 0."""
    out = np.empty((a.shape[1], a.shape[2]), dtype=np.float_)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            out[i, j] = np.nanmax(a[:, i, j])
    return out


@njit(cache=True)
def true_range(high_ts, low_ts, close_ts):
    """Calculate true range."""
    prev_close = generic_nb.fshift_nb(close_ts, 1)
    tr1 = high_ts - low_ts
    tr2 = np.abs(high_ts - prev_close)
    tr3 = np.abs(low_ts - prev_close)
    tr = nanmax_cube_nb(np.stack((tr1, tr2, tr3)))
    return tr


@njit(cache=True)
def atr_cache_nb(high_ts, low_ts, close_ts, windows, ewms):
    """Caching function for `vectorbt.indicators.basic.ATR`."""
    # Calculate TR here instead of re-calculating it for each param in atr_apply_nb
    tr = true_range(high_ts, low_ts, close_ts)

    cache_dict = dict()
    for i in range(len(windows)):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(tr, windows[i], ewms[i])
    return tr, cache_dict


@njit(cache=True)
def atr_apply_nb(high_ts, low_ts, close_ts, window, ewm, tr, cache_dict):
    """Apply function for `vectorbt.indicators.basic.ATR`."""
    h = hash((window, ewm))
    return tr, cache_dict[h]


@njit(cache=True)
def obv_custom_nb(close_ts, volume_ts):
    """Custom calculation function for `vectorbt.indicators.basic.OBV`."""
    obv = generic_nb.set_by_mask_mult_nb(volume_ts, close_ts < generic_nb.fshift_nb(close_ts, 1), -volume_ts)
    obv = generic_nb.cumsum_nb(obv)
    return obv
