import pandas as pd
import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1
from vectorbt.timeseries import rolling_mean_nb, rolling_std_nb, ewm_mean_nb, ewm_std_nb, diff_nb, set_by_mask_nb, prepend_nb

from vectorbt.decorators import *
from vectorbt.signals import Signals, generate_exits_nb
from vectorbt.timeseries import TimeSeries

__all__ = ['DMAC', 'BollingerBands', 'RSI']

# ############# Numba functions ############# #


@njit
def stack_outputs_nb(a, b, output_func):
    """Stack outputs along axis 1.
    
    We always work with 2D data, so stack all new combinations horizontally."""
    c = np.empty((a.shape[0], a.shape[1] * b.shape[0]), dtype=b1)
    for i in range(b.shape[0]):
        c[:, i*a.shape[1]:(i+1)*a.shape[1]] = output_func(a, b[i, :, :])
    return c


greater_than = njit(lambda a, b: a > b)
less_than = njit(lambda a, b: a < b)


@njit(b1[:, :](f8[:, :], f8[:, :, :]), cache=True)
def above_thresholds_nb(a, thresholds):
    return stack_outputs_nb(a, thresholds, greater_than)


@njit(b1[:, :](f8[:, :], f8[:, :, :]), cache=True)
def below_thresholds_nb(a, thresholds):
    return stack_outputs_nb(a, thresholds, less_than)

# ############# MovingAverage ############# #


@njit(UniTuple(f8[:, :], 2)(f8[:, :], i8[:], i8[:], b1, b1), cache=True)
def dmac_nb(ts, fast_windows, slow_windows, is_ewm, is_min_periods):
    """For each fast and slow window, calculate the corresponding SMA/EMA."""
    # Cache moving averages to effectively reduce the number of operations
    unique_windows = np.unique(np.concatenate((fast_windows, slow_windows)))
    cache_d = dict()
    for i in range(unique_windows.shape[0]):
        if is_ewm:
            ma = ewm_mean_nb(ts, unique_windows[i])
        else:
            ma = rolling_mean_nb(ts, unique_windows[i])
        if is_min_periods:
            ma[:unique_windows[i], :] = np.nan
        cache_d[unique_windows[i]] = ma
    # Concatenate moving averages out of cache and return
    fast_mas = np.empty((ts.shape[0], ts.shape[1] * fast_windows.shape[0]), dtype=f8)
    slow_mas = np.empty((ts.shape[0], ts.shape[1] * fast_windows.shape[0]), dtype=f8)
    for i in range(fast_windows.shape[0]):
        fast_mas[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = cache_d[fast_windows[i]]
        slow_mas[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = cache_d[slow_windows[i]]
    return fast_mas, slow_mas


class DMAC():
    """The Dual Moving Average Crossover trading system uses two moving averages, 
    one short and one long. The system trades when the short moving average 
    crosses the long moving average."""

    @to_2d('ts')
    @to_1d('fast_windows')
    @to_1d('slow_windows')
    @broadcast('fast_windows', 'slow_windows')
    @has_dtype('fast_windows', np.int64)
    @has_dtype('slow_windows', np.int64)
    @has_type('ts', TimeSeries)
    def __init__(self, ts, fast_windows, slow_windows, is_ewm=False, is_min_periods=True):
        # fast_windows and slow_windows can be either np.ndarray or single number
        fast, slow = dmac_nb(ts, fast_windows, slow_windows, is_ewm, is_min_periods)
        self.fast = TimeSeries(fast)
        self.slow = TimeSeries(slow)

    def is_fast_above_slow(self):
        return Signals(self.fast > self.slow)

    def is_fast_below_slow(self):
        return Signals(self.fast < self.slow)

    def crossover_signals(self):
        entries = self.is_fast_above_slow().first_true(after_false=True)  # crossover is first true after false
        exits = self.is_fast_below_slow().first_true(after_false=True)
        return entries, exits

# ############# BollingerBands ############# #


@njit(UniTuple(f8[:, :], 3)(f8[:, :], i8[:], i8[:], b1, b1), cache=True)
def bb_nb(ts, ns, ks, is_ewm, is_min_periods):
    """For each N and K, calculate the corresponding upper, middle and lower BB bands."""
    # Cache moving averages to effectively reduce the number of operations
    unique_windows = np.unique(ns)
    cache_d = dict()
    for i in range(unique_windows.shape[0]):
        if is_ewm:
            ma = ewm_mean_nb(ts, unique_windows[i])
            mstd = ewm_std_nb(ts, unique_windows[i])
        else:
            ma = rolling_mean_nb(ts, unique_windows[i])
            mstd = rolling_std_nb(ts, unique_windows[i])
        if is_min_periods:
            ma[:unique_windows[i], :] = np.nan
            mstd[:unique_windows[i], :] = np.nan
        cache_d[unique_windows[i]] = ma, mstd
    # Calculate lower, middle and upper bands
    upper = np.empty((ts.shape[0], ts.shape[1] * ns.shape[0]), dtype=f8)
    middle = np.empty((ts.shape[0], ts.shape[1] * ns.shape[0]), dtype=f8)
    lower = np.empty((ts.shape[0], ts.shape[1] * ns.shape[0]), dtype=f8)
    for i in range(ns.shape[0]):
        ma, mstd = cache_d[ns[i]]
        upper[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = ma + ks[i] * mstd  # (MA + Kσ)
        middle[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = ma  # MA
        lower[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = ma - ks[i] * mstd  # (MA - Kσ)
    return upper, middle, lower


class BollingerBands():
    """Bollinger Bands® are volatility bands placed above and below a moving average."""

    @to_2d('ts')
    @to_1d('windows')
    @to_1d('std_ns')
    @broadcast('windows', 'std_ns')
    @has_dtype('windows', np.int64)
    @has_dtype('std_ns', np.int64)
    @has_type('ts', TimeSeries)
    def __init__(self, ts, windows, std_ns, is_ewm=False, is_min_periods=True):
        # windows and std_ns can be either np.ndarray or single number
        self.ts = np.tile(ts, (1, windows.shape[0]))
        upper, middle, lower = bb_nb(ts, windows, std_ns, is_ewm, is_min_periods)
        self.upper = TimeSeries(upper)
        self.middle = TimeSeries(middle)
        self.lower = TimeSeries(lower)

    @cached_property
    def percent_b(self):
        """Shows where price is in relation to the bands.
        %b equals 1 at the upper band and 0 at the lower band."""
        return TimeSeries((self.ts - self.lower) / (self.upper - self.lower))

    @cached_property
    def bandwidth(self):
        """Bandwidth tells how wide the Bollinger Bands are on a normalized basis."""
        return TimeSeries((self.upper - self.lower) / self.middle)

    @broadcast_to_combs_of('thresholds', 'self.ts')
    @to_dtype('thresholds', np.float64)
    def is_percent_b_above(self, thresholds):
        return Signals(above_thresholds_nb(self.percent_b, thresholds))

    @broadcast_to_combs_of('thresholds', 'self.ts')
    @to_dtype('thresholds', np.float64)
    def is_percent_b_below(self, thresholds):
        return Signals(below_thresholds_nb(self.percent_b, thresholds))

    @broadcast_to_combs_of('thresholds', 'self.ts')
    @to_dtype('thresholds', np.float64)
    def is_bandwidth_above(self, thresholds):
        return Signals(above_thresholds_nb(self.bandwidth, thresholds))

    @broadcast_to_combs_of('thresholds', 'self.ts')
    @to_dtype('thresholds', np.float64)
    def is_bandwidth_below(self, thresholds):
        return Signals(below_thresholds_nb(self.bandwidth, thresholds))


# ############# RSI ############# #

@njit(f8[:, :](f8[:, :], i8[:], b1, b1), cache=True)
def rsi_nb(ts, windows, is_ewm, is_min_periods):
    """For each window, calculate the RSI."""
    delta = diff_nb(ts)[1:, :]  # otherwise ewma will be all NaN
    up, down = delta.copy(), delta.copy()
    up = set_by_mask_nb(up, up < 0, 0)
    down = np.abs(set_by_mask_nb(down, down > 0, 0))
    # Cache moving averages to effectively reduce the number of operations
    unique_windows = np.unique(windows)
    cache_d = dict()
    for i in range(unique_windows.shape[0]):
        if is_ewm:
            roll_up = ewm_mean_nb(up, unique_windows[i])
            roll_down = ewm_mean_nb(down, unique_windows[i])
        else:
            roll_up = rolling_mean_nb(up, unique_windows[i])
            roll_down = rolling_mean_nb(down, unique_windows[i])
        roll_up = prepend_nb(roll_up, 1, np.nan)  # bring to old shape
        roll_down = prepend_nb(roll_down, 1, np.nan)
        if is_min_periods:
            roll_up[:unique_windows[i], :] = np.nan
            roll_down[:unique_windows[i], :] = np.nan
        cache_d[unique_windows[i]] = roll_up, roll_down
    # Calculate RSI
    rsi = np.empty((ts.shape[0], ts.shape[1] * windows.shape[0]), dtype=f8)
    for i in range(windows.shape[0]):
        roll_up, roll_down = cache_d[windows[i]]
        rsi[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = 100 - 100 / (1 + roll_up / roll_down)
    return rsi


class RSI():
    """The relative strength index (RSI) is a momentum indicator that 
    measures the magnitude of recent price changes to evaluate overbought 
    or oversold conditions in the price of a stock or other asset."""

    @to_2d('ts')
    @to_1d('windows')
    @has_dtype('windows', np.int64)
    @has_type('ts', TimeSeries)
    def __init__(self, ts, windows, is_ewm=False, is_min_periods=True):
        self.rsi = TimeSeries(rsi_nb(ts, windows, is_ewm, is_min_periods))

    @broadcast_to_combs_of('thresholds', 'self.rsi')
    @to_dtype('thresholds', np.float64)
    def is_rsi_above(self, thresholds):
        return Signals(above_thresholds_nb(self.rsi, thresholds))

    @broadcast_to_combs_of('thresholds', 'self.rsi')
    @to_dtype('thresholds', np.float64)
    def is_rsi_below(self, thresholds):
        return Signals(below_thresholds_nb(self.rsi, thresholds))
