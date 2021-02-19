"""Numba-compiled functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument should be Numba-compiled."""

import numpy as np
from numba import njit

from vectorbt.generic import nb as generic_nb


@njit(cache=True)
def ma_nb(a, period, ewm, adjust=False):
    """Compute simple or exponential moving average (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_mean_nb(a, period, minp=period, adjust=adjust)
    return generic_nb.rolling_mean_nb(a, period, minp=period)


@njit(cache=True)
def mstd_nb(a, period, ewm, adjust=False, ddof=0):
    """Compute simple or exponential moving STD (`ewm=True`)."""
    if ewm:
        return generic_nb.ewm_std_nb(a, period, minp=period, adjust=adjust, ddof=ddof)
    return generic_nb.rolling_std_nb(a, period, minp=period, ddof=ddof)


@njit(cache=True)
def ma_cache_nb(ts, periods, ewms, adjust):
    """Caching function for `vectorbt.indicators.basic.MA`."""
    cache_dict = dict()
    for i in range(len(periods)):
        h = hash((periods[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(ts, periods[i], ewms[i], adjust=adjust)
    return cache_dict


@njit(cache=True)
def ma_apply_nb(ts, period, ewm, adjust, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MA`."""
    h = hash((period, ewm))
    return cache_dict[h]


@njit(cache=True)
def mstd_cache_nb(ts, periods, ewms, adjust, ddof):
    """Caching function for `vectorbt.indicators.basic.MSTD`."""
    cache_dict = dict()
    for i in range(len(periods)):
        h = hash((periods[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = mstd_nb(ts, periods[i], ewms[i], adjust=adjust, ddof=ddof)
    return cache_dict


@njit(cache=True)
def mstd_apply_nb(ts, period, ewm, adjust, ddof, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MSTD`."""
    h = hash((period, ewm))
    return cache_dict[h]


@njit(cache=True)
def bb_cache_nb(ts, periods, ewms, alphas, adjust, ddof):
    """Caching function for `vectorbt.indicators.basic.BBANDS`."""
    ma_cache_dict = ma_cache_nb(ts, periods, ewms, adjust)
    mstd_cache_dict = mstd_cache_nb(ts, periods, ewms, adjust, ddof)
    return ma_cache_dict, mstd_cache_dict


@njit(cache=True)
def bb_apply_nb(ts, period, ewm, alpha, adjust, ddof, ma_cache_dict, mstd_cache_dict):
    """Apply function for `vectorbt.indicators.basic.BBANDS`."""
    # Calculate lower, middle and upper bands
    h = hash((period, ewm))
    ma = np.copy(ma_cache_dict[h])
    mstd = np.copy(mstd_cache_dict[h])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma, ma + alpha * mstd, ma - alpha * mstd


@njit(cache=True)
def rsi_cache_nb(ts, periods, ewms, adjust):
    """Caching function for `vectorbt.indicators.basic.RSI`."""
    delta = generic_nb.diff_nb(ts)
    up, down = delta.copy(), delta.copy()
    up = generic_nb.set_by_mask_nb(up, up < 0, 0)
    down = np.abs(generic_nb.set_by_mask_nb(down, down > 0, 0))

    # Cache
    cache_dict = dict()
    for i in range(len(periods)):
        h = hash((periods[i], ewms[i]))
        if h not in cache_dict:
            roll_up = ma_nb(up, periods[i], ewms[i], adjust=adjust)
            roll_down = ma_nb(down, periods[i], ewms[i], adjust=adjust)
            cache_dict[h] = roll_up, roll_down
    return cache_dict


@njit(cache=True)
def rsi_apply_nb(ts, period, ewm, adjust, cache_dict):
    """Apply function for `vectorbt.indicators.basic.RSI`."""
    h = hash((period, ewm))
    roll_up, roll_down = cache_dict[h]
    rs = roll_up / roll_down
    return 100 - 100 / (1 + rs)


@njit(cache=True)
def stoch_cache_nb(high_ts, low_ts, close_ts, k_periods, d_periods, d_ewms, adjust):
    """Caching function for `vectorbt.indicators.basic.STOCH`."""
    cache_dict = dict()
    for i in range(len(k_periods)):
        h = hash(k_periods[i])
        if h not in cache_dict:
            roll_min = generic_nb.rolling_min_nb(low_ts, k_periods[i])
            roll_max = generic_nb.rolling_max_nb(high_ts, k_periods[i])
            cache_dict[h] = roll_min, roll_max
    return cache_dict


@njit(cache=True)
def stoch_apply_nb(high_ts, low_ts, close_ts, k_period, d_period, d_ewm, adjust, cache_dict):
    """Apply function for `vectorbt.indicators.basic.STOCH`."""
    h = hash(k_period)
    roll_min, roll_max = cache_dict[h]
    percent_k = 100 * (close_ts - roll_min) / (roll_max - roll_min)
    percent_d = ma_nb(percent_k, d_period, d_ewm, adjust=adjust)
    return percent_k, percent_d


@njit(cache=True)
def macd_cache_nb(ts, fast_periods, slow_periods, signal_periods, macd_ewms, signal_ewms, adjust):
    """Caching function for `vectorbt.indicators.basic.MACD`."""
    periods = fast_periods.copy()
    periods.extend(slow_periods)
    ewms = macd_ewms.copy()
    ewms.extend(macd_ewms)
    return ma_cache_nb(ts, periods, ewms, adjust)


@njit(cache=True)
def macd_apply_nb(ts, fast_period, slow_period, signal_period, macd_ewm, signal_ewm, adjust, cache_dict):
    """Apply function for `vectorbt.indicators.basic.MACD`."""
    fast_h = hash((fast_period, macd_ewm))
    slow_h = hash((slow_period, macd_ewm))
    fast_ma = cache_dict[fast_h]
    slow_ma = cache_dict[slow_h]
    macd_ts = fast_ma - slow_ma
    signal_ts = ma_nb(macd_ts, signal_period, signal_ewm, adjust=adjust)
    return macd_ts, signal_ts


@njit(cache=True)
def true_range_nb(high_ts, low_ts, close_ts):
    """Calculate true range."""
    prev_close = generic_nb.fshift_nb(close_ts, 1)
    tr1 = high_ts - low_ts
    tr2 = np.abs(high_ts - prev_close)
    tr3 = np.abs(low_ts - prev_close)
    tr = np.empty(prev_close.shape, dtype=np.float_)
    for col in range(tr.shape[1]):
        for i in range(tr.shape[0]):
            tr[i, col] = max(tr1[i, col], tr2[i, col], tr3[i, col])
    return tr


@njit(cache=True)
def atr_cache_nb(high_ts, low_ts, close_ts, periods, ewms, adjust):
    """Caching function for `vectorbt.indicators.basic.ATR`."""
    # Calculate TR here instead of re-calculating it for each param in atr_apply_nb
    tr = true_range_nb(high_ts, low_ts, close_ts)
    cache_dict = dict()
    for i in range(len(periods)):
        h = hash((periods[i], ewms[i]))
        if h not in cache_dict:
            cache_dict[h] = ma_nb(tr, periods[i], ewms[i], adjust=adjust)
    return tr, cache_dict


@njit(cache=True)
def atr_apply_nb(high_ts, low_ts, close_ts, period, ewm, adjust, tr, cache_dict):
    """Apply function for `vectorbt.indicators.basic.ATR`."""
    h = hash((period, ewm))
    return tr, cache_dict[h]


@njit(cache=True)
def obv_custom_nb(close_ts, volume_ts):
    """Custom calculation function for `vectorbt.indicators.basic.OBV`."""
    obv = generic_nb.set_by_mask_mult_nb(volume_ts, close_ts < generic_nb.fshift_nb(close_ts, 1), -volume_ts)
    obv = generic_nb.cumsum_nb(obv)
    return obv
