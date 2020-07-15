"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import njit

from vectorbt import tseries


@njit(cache=True)
def ma_caching_nb(ts, windows, ewms):
    """Caching function for `vectorbt.indicators.basic.MA`."""
    cache_dict = dict()
    for i in range(windows.shape[0]):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            if ewms[i]:
                ma = tseries.nb.ewm_mean_nb(ts, windows[i])
            else:
                ma = tseries.nb.rolling_mean_nb(ts, windows[i])
            cache_dict[h] = ma
    return cache_dict


@njit(cache=True)
def ma_apply_nb(ts, window, ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MA`."""
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def mstd_caching_nb(ts, windows, ewms):
    """Caching function for `vectorbt.indicators.basic.MSTD`."""
    cache_dict = dict()
    for i in range(windows.shape[0]):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            if ewms[i]:
                mstd = tseries.nb.ewm_std_nb(ts, windows[i])
            else:
                mstd = tseries.nb.rolling_std_nb(ts, windows[i])
            cache_dict[h] = mstd
    return cache_dict


@njit(cache=True)
def mstd_apply_nb(ts, window, ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MSTD`."""
    h = hash((window, ewm))
    return cache_dict[h]


@njit(cache=True)
def bb_caching_nb(ts, windows, ewms, alphas):
    """Caching function for `vectorbt.indicators.basic.BollingerBands`."""
    ma_cache_dict = ma_caching_nb(ts, windows, ewms)
    mstd_cache_dict = mstd_caching_nb(ts, windows, ewms)
    return ma_cache_dict, mstd_cache_dict


@njit(cache=True)
def bb_apply_nb(ts, window, ewm, alpha, ma_cache_dict, mstd_cache_dict):
    """Apply function for `vectorbt.indicators.basic.BollingerBands`."""
    # Calculate lower, middle and upper bands
    h = hash((window, ewm))
    ma = np.copy(ma_cache_dict[h])
    mstd = np.copy(mstd_cache_dict[h])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


@njit(cache=True)
def rsi_caching_nb(ts, windows, ewms):
    """Caching function for `vectorbt.indicators.basic.RSI`."""
    delta = tseries.nb.diff_nb(ts)[1:, :]  # otherwise ewma will be all NaN
    up, down = delta.copy(), delta.copy()
    up = tseries.nb.set_by_mask_nb(up, up < 0, 0)
    down = np.abs(tseries.nb.set_by_mask_nb(down, down > 0, 0))
    # Cache
    cache_dict = dict()
    for i in range(windows.shape[0]):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            if ewms[i]:
                roll_up = tseries.nb.ewm_mean_nb(up, windows[i])
                roll_down = tseries.nb.ewm_mean_nb(down, windows[i])
            else:
                roll_up = tseries.nb.rolling_mean_nb(up, windows[i])
                roll_down = tseries.nb.rolling_mean_nb(down, windows[i])
            roll_up = tseries.nb.prepend_nb(roll_up, 1, np.nan)  # bring to old shape
            roll_down = tseries.nb.prepend_nb(roll_down, 1, np.nan)
            cache_dict[h] = roll_up, roll_down
    return cache_dict


@njit(cache=True)
def rsi_apply_nb(ts, window, ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.RSI`."""
    h = hash((window, ewm))
    roll_up, roll_down = cache_dict[h]
    return 100 - 100 / (1 + roll_up / roll_down)


@njit(cache=True)
def stoch_caching_nb(close_ts, high_ts, low_ts, k_windows, d_windows, d_ewms):
    """Caching function for `vectorbt.indicators.basic.Stochastic`."""
    cache_dict = dict()
    for i in range(k_windows.shape[0]):
        h = hash(k_windows[i])
        if h not in cache_dict:
            roll_min = tseries.nb.rolling_min_nb(low_ts, k_windows[i])
            roll_max = tseries.nb.rolling_max_nb(high_ts, k_windows[i])
            cache_dict[h] = roll_min, roll_max
    return cache_dict


@njit(cache=True)
def stoch_apply_nb(close_ts, high_ts, low_ts, k_window, d_window, d_ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.Stochastic`."""
    h = hash(k_window)
    roll_min, roll_max = cache_dict[h]
    percent_k = 100 * (close_ts - roll_min) / (roll_max - roll_min)
    if d_ewm:
        percent_d = tseries.nb.ewm_mean_nb(percent_k, d_window)
    else:
        percent_d = tseries.nb.rolling_mean_nb(percent_k, d_window)
    return percent_k, percent_d


@njit(cache=True)
def macd_caching_nb(ts, fast_windows, slow_windows, signal_windows, macd_ewms, signal_ewms):
    """Caching function for `vectorbt.indicators.basic.MACD`."""
    return ma_caching_nb(ts, np.concatenate((fast_windows, slow_windows)), np.concatenate((macd_ewms, macd_ewms)))


@njit(cache=True)
def macd_apply_nb(ts, fast_window, slow_window, signal_window, macd_ewm, signal_ewm, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MACD`."""
    fast_h = hash((fast_window, macd_ewm))
    slow_h = hash((slow_window, macd_ewm))
    fast_ma = cache_dict[fast_h]
    slow_ma = cache_dict[slow_h]
    macd_ts = fast_ma - slow_ma
    if signal_ewm:
        signal_ts = tseries.nb.ewm_mean_nb(macd_ts, signal_window)
    else:
        signal_ts = tseries.nb.rolling_mean_nb(macd_ts, signal_window)
    return macd_ts, signal_ts


@njit(cache=True)
def nanmax_cube_nb(a):
    """Return max of a cube by reducing the axis 0."""
    result = np.empty((a.shape[1], a.shape[2]), dtype=np.float_)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            result[i, j] = np.nanmax(a[:, i, j])
    return result


@njit(cache=True)
def atr_caching_nb(close_ts, high_ts, low_ts, windows, ewms):
    """Caching function for `vectorbt.indicators.basic.ATR`."""
    # Calculate TR here instead of re-calculating it for each param in atr_apply_nb
    tr0 = high_ts - low_ts
    tr1 = np.abs(high_ts - tseries.nb.fshift_nb(close_ts, 1))
    tr2 = np.abs(low_ts - tseries.nb.fshift_nb(close_ts, 1))
    tr = nanmax_cube_nb(np.stack((tr0, tr1, tr2)))

    cache_dict = dict()
    for i in range(windows.shape[0]):
        h = hash((windows[i], ewms[i]))
        if h not in cache_dict:
            if ewms[i]:
                atr = tseries.nb.ewm_mean_nb(tr, windows[i])
            else:
                atr = tseries.nb.rolling_mean_nb(tr, windows[i])
            cache_dict[h] = atr
    return tr, cache_dict


@njit(cache=True)
def atr_apply_nb(close_ts, high_ts, low_ts, window, ewm, tr, cache_dict):
    """Apply function for `vectorbt.indicators.basic.ATR`."""
    h = hash((window, ewm))
    return tr, cache_dict[h]


@njit(cache=True)
def obv_custom_func_nb(close_ts, volume_ts):
    """Custom calculation function for `vectorbt.indicators.basic.OBV`."""
    obv = np.full(close_ts.shape, np.nan, dtype=np.float_)
    for col in range(close_ts.shape[1]):
        cumsum = 0
        for i in range(1, close_ts.shape[0]):
            if np.isnan(close_ts[i, col]) or np.isnan(close_ts[i-1, col]) or np.isnan(volume_ts[i, col]):
                continue
            if close_ts[i, col] > close_ts[i-1, col]:
                cumsum += volume_ts[i, col]
            elif close_ts[i, col] < close_ts[i-1, col]:
                cumsum += -volume_ts[i, col]
            obv[i, col] = cumsum
    return obv