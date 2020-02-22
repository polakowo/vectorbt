import pandas as pd
import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1
from vectorbt.timeseries import rolling_mean_nb, rolling_std_nb, ewm_mean_nb, ewm_std_nb, diff_nb, \
    set_by_mask_nb, prepend_nb, _expanding_max_1d_nb, _pct_change_1d_nb, _ffill_1d_nb

from vectorbt.decorators import *
from vectorbt.signals import Signals, generate_exits_nb
from vectorbt.timeseries import TimeSeries

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

    @property
    def percent_b(self):
        """Shows where price is in relation to the bands.
        %b equals 1 at the upper band and 0 at the lower band."""
        return TimeSeries((self.ts - self.lower) / (self.upper - self.lower))

    @property
    def bandwidth(self):
        """Bandwidth tells how wide the Bollinger Bands are on a normalized basis."""
        return TimeSeries((self.upper - self.lower) / self.middle)

    @to_2d('threshold')
    @broadcast_to('threshold', 'self.ts')
    def is_percent_b_above_threshold(self, threshold):
        return Signals(self.percent_b > threshold)

    @to_2d('threshold')
    @broadcast_to('threshold', 'self.ts')
    def is_percent_b_below_threshold(self, threshold):
        return Signals(self.percent_b < threshold)

    @to_2d('threshold')
    @broadcast_to('threshold', 'self.ts')
    def is_bandwidth_above_threshold(self, threshold):
        return Signals(self.bandwidth > threshold)

    @to_2d('threshold')
    @broadcast_to('threshold', 'self.ts')
    def is_bandwidth_below_threshold(self, threshold):
        return Signals(self.bandwidth < threshold)


# ############# RSI ############# #

@njit(f8[:, :](f8[:, :], i8[:], b1, b1), cache=True)
def rsi_nb(ts, windows, is_ewm, is_min_periods):
    """For each window, calculate the RSI."""
    delta = diff_nb(ts, 1)[1:, :]  # otherwise ewma will be all NaN
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

    @to_2d('threshold')
    @broadcast_to('threshold', 'self.rsi')
    def is_rsi_above_threshold(self, threshold):
        return Signals(self.rsi > threshold)

    @to_2d('threshold')
    @broadcast_to('threshold', 'self.rsi')
    def is_rsi_below_threshold(self, threshold):
        return Signals(self.rsi < threshold)


# ############# Risk minimization ############# #

@njit(b1[:](b1[:, :], i8, i8, i8, f8[:, :], f8[:, :], b1), cache=True)
def stoploss_exit_mask_nb(entries, col_idx, prev_idx, next_idx, ts, stop, is_relative):
    """Index of the first event below the stop."""
    ts = ts[:, col_idx]
    # Stop is defined at the entry point
    stop = stop[prev_idx, col_idx]
    if is_relative:
        stop = (1 - stop) * ts[prev_idx]
    return ts < stop


@njit(b1[:, :](f8[:, :], b1[:, :], f8[:, :, :], b1, b1), cache=True)
def stoploss_exits_nb(ts, entries, stops, is_relative, only_first):
    """Calculate exit signals based on stop loss strategy.

    An approach here significantly differs from the approach with rolling windows.
    If user wants to try out different rolling windows, he can pass them as a 1d array.
    Here, user must be able to try different stops not only for the `ts` itself,
    but also for each element in `ts`, since stops may vary with time.
    This requires the variable `stops` to be a 3d array (cube) out of 2d matrices of form of `ts`.
    For example, if you want to try stops 0.1 and 0.2, both must have the shape of `ts`,
    wrapped into an array, thus forming a cube (2, ts.shape[0], ts.shape[1])"""

    exits = np.empty((ts.shape[0], ts.shape[1] * stops.shape[0]), dtype=b1)
    for i in range(stops.shape[0]):
        i_exits = generate_exits_nb(entries, stoploss_exit_mask_nb, only_first, ts, stops[i, :, :], is_relative)
        exits[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = i_exits
    return exits


class StopLoss():
    @to_2d('ts')
    @to_2d('entries')
    @broadcast('ts', 'entries')
    @broadcast_to_cube_of('stops', 'ts')
    @has_type('ts', TimeSeries)
    @has_type('entries', Signals)
    # stops can be either a number, an array of numbers, or an array of matrices each of ts shape
    def __init__(self, ts, entries, stops, is_relative=True, only_first=True):
        """A stop-loss is designed to limit an investor's loss on a security position. 
        Setting a stop-loss order for 10% below the price at which you bought the stock 
        will limit your loss to 10%."""

        self.exits = Signals(stoploss_exits_nb(ts, entries, stops, is_relative, only_first))


@njit(b1[:](b1[:, :], i8, i8, i8, f8[:, :], f8[:, :], b1), cache=True)
def tstop_exit_mask_nb(entries, col_idx, prev_idx, next_idx, ts, stop, is_relative):
    """Index of the first event below the trailing stop."""
    exit_mask = np.empty(ts.shape[0], dtype=b1)
    ts = ts[prev_idx:next_idx, col_idx]
    stop = stop[prev_idx:next_idx, col_idx]
    # Propagate the maximum value from the entry using expanding max
    peak = _expanding_max_1d_nb(ts)
    if np.min(stop) != np.max(stop):
        # Propagate the stop value of the last max
        raising_idxs = np.flatnonzero(_pct_change_1d_nb(peak))
        stop_temp = np.full(ts.shape, np.nan)
        stop_temp[raising_idxs] = stop[raising_idxs]
        stop_temp = _ffill_1d_nb(stop_temp)
        stop_temp[np.isnan(stop_temp)] = -np.inf
        stop = stop_temp
    if is_relative:
        stop = (1 - stop) * peak
    exit_mask[prev_idx:next_idx] = ts < stop
    return exit_mask


@njit(b1[:, :](f8[:, :], b1[:, :], f8[:, :, :], b1, b1), cache=True)
def tstop_exits_nb(ts, entries, stops, is_relative, only_first):
    """Calculate exit signals based on trailing stop strategy."""
    exits = np.empty((ts.shape[0], ts.shape[1] * stops.shape[0]), dtype=b1)
    for i in range(stops.shape[0]):
        i_exits = generate_exits_nb(entries, tstop_exit_mask_nb, only_first, ts, stops[i, :, :], is_relative)
        exits[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = i_exits
    return exits


class TrailingStop():
    @to_2d('ts')
    @to_2d('entries')
    @broadcast('ts', 'entries')
    @broadcast_to_cube_of('stops', 'ts')
    @has_type('ts', TimeSeries)
    @has_type('entries', Signals)
    # stops can be either a number, an array of numbers, or an array of matrices each of ts shape
    def __init__(self, ts, entries, stops, is_relative=True, only_first=True):
        """A Trailing Stop order is a stop order that can be set at a defined percentage 
        or amount away from the current market price. The main difference between a regular 
        stop loss and a trailing stop is that the trailing stop moves as the price moves."""

        self.exits = Signals(tstop_exits_nb(ts, entries, stops, is_relative, only_first))
