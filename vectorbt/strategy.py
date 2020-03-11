import pandas as pd
import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1
from vectorbt.timeseries import rolling_mean_2d_nb, rolling_std_2d_nb, ewm_mean_2d_nb, \
    ewm_std_2d_nb, diff_2d_nb, set_by_mask_2d_nb, prepend_2d_nb
from vectorbt.utils import *
from copy import deepcopy

__all__ = ['DMAC', 'BollingerBands', 'RSI']

# ############# Numba functions ############# #


@njit
def stack_outputs_nb(a, b, output_func):
    """Stack outputs along axis 1.

    We always work with 2D data, so stack all new combinations horizontally."""
    c = np.empty((a.shape[0], a.shape[1] * b.shape[0]), dtype=b1)
    for i in range(b.shape[0]):
        c[:, i*a.shape[1]:(i+1)*a.shape[1]] = output_func(a, b[i])
    return c


greater_than = njit(lambda a, b: a > b)
less_than = njit(lambda a, b: a < b)


@njit(b1[:, :](f8[:, :], f8[:]), cache=True)
def above_thresholds_nb(a, thresholds):
    return stack_outputs_nb(a, thresholds, greater_than)


@njit(b1[:, :](f8[:, :], f8[:]), cache=True)
def below_thresholds_nb(a, thresholds):
    return stack_outputs_nb(a, thresholds, less_than)

# ############# DMAC ############# #


@njit(UniTuple(f8[:, :], 2)(f8[:, :], i8[:], i8[:], b1, b1), cache=True)
def dmac_nb(ts, fast_windows, slow_windows, is_ewm, is_min_periods):
    """For each fast and slow window, calculate the corresponding SMA/EMA."""
    # Cache moving averages to effectively reduce the number of operations
    unique_windows = np.unique(np.concatenate((fast_windows, slow_windows)))
    cache_d = dict()
    for i in range(unique_windows.shape[0]):
        if is_ewm:
            ma = ewm_mean_2d_nb(ts, unique_windows[i])
        else:
            ma = rolling_mean_2d_nb(ts, unique_windows[i])
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


def dmac_indexing_func(obj, getitem_func):
    obj_copy = deepcopy(obj)
    obj_copy.ts = getitem_func(obj.ts)
    obj_copy.fast = getitem_func(obj.fast)
    obj_copy.slow = getitem_func(obj.slow)
    return obj_copy


@add_indexing_methods(dmac_indexing_func)
class DMAC():
    """The Dual Moving Average Crossover trading system uses two moving averages, 
    one short and one long. The system trades when the short moving average 
    crosses the long moving average."""

    def __init__(self, ts, fast_windows, slow_windows, is_ewm=False, is_min_periods=True):
        # Checks and preprocessing
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.timeseries.validate()
        fast_windows = to_1d(fast_windows)
        slow_windows = to_1d(slow_windows)
        fast_windows, slow_windows = broadcast(fast_windows, slow_windows)
        check_dtype(fast_windows, np.int64)
        check_dtype(slow_windows, np.int64)

        # fast_windows and slow_windows can be either np.ndarray or single number
        fast, slow = dmac_nb(ts.arr.to_2d_array(), fast_windows, slow_windows, is_ewm, is_min_periods)

        # Build column hierarchy
        fast_index = pd.DataFrame.cols.index_from_params(fast_windows, name='fast_window')
        slow_index = pd.DataFrame.cols.index_from_params(slow_windows, name='slow_window')
        param_columns = pd.DataFrame.cols.stack_indexes(fast_index, slow_index)
        columns = ts.cols.combine_columns(param_columns)

        if fast_windows.shape[0] > 1:
            self.ts = ts.arr.wrap_array(np.tile(to_2d(ts), (1, fast_windows.shape[0])), columns=columns)
        else:
            self.ts = ts.arr.wrap_array(ts, columns=columns)
        self.fast = ts.arr.wrap_array(fast, columns=columns)
        self.slow = ts.arr.wrap_array(slow, columns=columns)

    def is_fast_above_slow(self):
        return self.fast > self.slow

    def is_fast_below_slow(self):
        return self.fast < self.slow

    def crossover_signals(self):
        # crossover is first true after false
        entries = self.is_fast_above_slow().signals.first_true(after_false=True)
        exits = self.is_fast_below_slow().signals.first_true(after_false=True)
        return entries, exits

    def plot(self,
             plot_ts=True,
             ts_scatter_kwargs={},
             fast_scatter_kwargs={},
             slow_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        # Checks and preprocessing
        check_type(self.ts, pd.Series)
        check_type(self.fast, pd.Series)
        check_type(self.slow, pd.Series)

        ts_scatter_kwargs = {**dict(name='Price'), **ts_scatter_kwargs}
        fast_scatter_kwargs = {**dict(name='Fast MA'), **fast_scatter_kwargs}
        slow_scatter_kwargs = {**dict(name='Slow MA'), **slow_scatter_kwargs}

        if plot_ts:
            fig = self.ts.timeseries.plot(scatter_kwargs=ts_scatter_kwargs, fig=fig, **layout_kwargs)
        fig = self.fast.timeseries.plot(scatter_kwargs=fast_scatter_kwargs, fig=fig)
        fig = self.slow.timeseries.plot(scatter_kwargs=slow_scatter_kwargs, fig=fig)

        return fig


# ############# BollingerBands ############# #


@njit(UniTuple(f8[:, :], 3)(f8[:, :], i8[:], i8[:], b1, b1), cache=True)
def bb_nb(ts, ns, ks, is_ewm, is_min_periods):
    """For each N and K, calculate the corresponding upper, middle and lower BB bands."""
    # Cache moving averages to effectively reduce the number of operations
    unique_windows = np.unique(ns)
    cache_d = dict()
    for i in range(unique_windows.shape[0]):
        if is_ewm:
            ma = ewm_mean_2d_nb(ts, unique_windows[i])
            mstd = ewm_std_2d_nb(ts, unique_windows[i])
        else:
            ma = rolling_mean_2d_nb(ts, unique_windows[i])
            mstd = rolling_std_2d_nb(ts, unique_windows[i])
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


def compare_with_thresholds(compare_func, pd_obj, thresholds):
    # Checks and preprocessing
    thresholds = to_1d(thresholds)
    thresholds = thresholds.astype(np.float64)

    arr = compare_func(pd_obj.arr.to_2d_array(), thresholds)

    # Build column hierarchy
    param_columns = pd.DataFrame.cols.index_from_params(thresholds, name='threshold')
    columns = pd_obj.cols.combine_columns(param_columns)

    return pd_obj.arr.wrap_array(arr, columns=columns)


def bb_indexing_func(obj, getitem_func):
    obj_copy = deepcopy(obj)
    obj_copy.ts = getitem_func(obj.ts)
    obj_copy.upper = getitem_func(obj.upper)
    obj_copy.middle = getitem_func(obj.middle)
    obj_copy.lower = getitem_func(obj.lower)
    # Don't forget to remove cached attributes
    if hasattr(obj_copy, '_percent_b'):
        delattr(obj_copy, '_percent_b')
    if hasattr(obj_copy, '_bandwidth'):
        delattr(obj_copy, '_bandwidth')
    return obj_copy


@add_indexing_methods(bb_indexing_func)
class BollingerBands():
    """Bollinger Bands® are volatility bands placed above and below a moving average."""

    def __init__(self, ts, windows, std_ns, is_ewm=False, is_min_periods=True):
        # Checks and preprocessing
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.timeseries.validate()
        windows = to_1d(windows)
        std_ns = to_1d(std_ns)
        windows, std_ns = broadcast(windows, std_ns)
        check_dtype(windows, np.int64)
        check_dtype(std_ns, np.int64)

        # windows and std_ns can be either np.ndarray or single number
        upper, middle, lower = bb_nb(ts.arr.to_2d_array(), windows, std_ns, is_ewm, is_min_periods)

        # Build column hierarchy
        window_index = pd.DataFrame.cols.index_from_params(windows, name='window')
        std_n_index = pd.DataFrame.cols.index_from_params(std_ns, name='std_n')
        param_columns = pd.DataFrame.cols.stack_indexes(window_index, std_n_index)
        columns = ts.cols.combine_columns(param_columns)

        if windows.shape[0] > 1:
            self.ts = ts.arr.wrap_array(np.tile(to_2d(ts), (1, windows.shape[0])), columns=columns)
        else:
            self.ts = ts.arr.wrap_array(ts, columns=columns)
        self.upper = ts.arr.wrap_array(upper, columns=columns)
        self.middle = ts.arr.wrap_array(middle, columns=columns)
        self.lower = ts.arr.wrap_array(lower, columns=columns)

    @cached_property
    def percent_b(self):
        """Shows where price is in relation to the bands.
        %b equals 1 at the upper band and 0 at the lower band."""
        return (self.ts - self.lower) / (self.upper - self.lower)

    @cached_property
    def bandwidth(self):
        """Bandwidth tells how wide the Bollinger Bands are on a normalized basis."""
        return (self.upper - self.lower) / self.middle

    def is_percent_b_above(self, thresholds):
        return compare_with_thresholds(above_thresholds_nb, self.percent_b, thresholds)

    def is_percent_b_below(self, thresholds):
        return compare_with_thresholds(below_thresholds_nb, self.percent_b, thresholds)

    def is_bandwidth_above(self, thresholds):
        return compare_with_thresholds(above_thresholds_nb, self.bandwidth, thresholds)

    def is_bandwidth_below(self, thresholds):
        return compare_with_thresholds(below_thresholds_nb, self.bandwidth, thresholds)

    def plot(self,
             plot_ts=True,
             ts_scatter_kwargs={},
             upper_scatter_kwargs={},
             middle_scatter_kwargs={},
             lower_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        # Checks and preprocessing
        check_type(self.ts, pd.Series)
        check_type(self.upper, pd.Series)
        check_type(self.middle, pd.Series)
        check_type(self.lower, pd.Series)

        ts_scatter_kwargs = {**dict(name='Price'), **ts_scatter_kwargs}  # user kwargs override default kwargs
        upper_scatter_kwargs = {**dict(name='Upper Band', line=dict(color='grey')), **upper_scatter_kwargs}
        middle_scatter_kwargs = {**dict(name='Middle Band'), **middle_scatter_kwargs}
        lower_scatter_kwargs = {**dict(name='Lower Band', line=dict(color='grey')), **lower_scatter_kwargs}

        if plot_ts:
            fig = self.ts.timeseries.plot(scatter_kwargs=ts_scatter_kwargs, fig=fig, **layout_kwargs)
        fig = self.upper.timeseries.plot(scatter_kwargs=upper_scatter_kwargs, fig=fig)
        fig = self.middle.timeseries.plot(scatter_kwargs=middle_scatter_kwargs, fig=fig)
        fig = self.lower.timeseries.plot(scatter_kwargs=lower_scatter_kwargs, fig=fig)

        return fig


# ############# RSI ############# #

@njit(f8[:, :](f8[:, :], i8[:], b1, b1), cache=True)
def rsi_nb(ts, windows, is_ewm, is_min_periods):
    """For each window, calculate the RSI."""
    delta = diff_2d_nb(ts)[1:, :]  # otherwise ewma will be all NaN
    up, down = delta.copy(), delta.copy()
    up = set_by_mask_2d_nb(up, up < 0, 0)
    down = np.abs(set_by_mask_2d_nb(down, down > 0, 0))
    # Cache moving averages to effectively reduce the number of operations
    unique_windows = np.unique(windows)
    cache_d = dict()
    for i in range(unique_windows.shape[0]):
        if is_ewm:
            roll_up = ewm_mean_2d_nb(up, unique_windows[i])
            roll_down = ewm_mean_2d_nb(down, unique_windows[i])
        else:
            roll_up = rolling_mean_2d_nb(up, unique_windows[i])
            roll_down = rolling_mean_2d_nb(down, unique_windows[i])
        roll_up = prepend_2d_nb(roll_up, 1, np.nan)  # bring to old shape
        roll_down = prepend_2d_nb(roll_down, 1, np.nan)
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


def rsi_indexing_func(obj, getitem_func):
    obj_copy = deepcopy(obj)
    obj_copy.ts = getitem_func(obj.ts)
    obj_copy.rsi = getitem_func(obj.rsi)
    return obj_copy


@add_indexing_methods(rsi_indexing_func)
class RSI():
    """The relative strength index (RSI) is a momentum indicator that 
    measures the magnitude of recent price changes to evaluate overbought 
    or oversold conditions in the price of a stock or other asset."""

    def __init__(self, ts, windows, is_ewm=False, is_min_periods=True):
        # Checks and preprocessing
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.timeseries.validate()
        windows = to_1d(windows)
        check_dtype(windows, np.int64)

        # windows can be either np.ndarray or single number
        rsi = rsi_nb(ts.arr.to_2d_array(), windows, is_ewm, is_min_periods)

        # Build column hierarchy
        param_columns = pd.DataFrame.cols.index_from_params(windows, name='window')
        columns = ts.cols.combine_columns(param_columns)

        if windows.shape[0] > 1:
            self.ts = ts.arr.wrap_array(np.tile(to_2d(ts), (1, windows.shape[0])), columns=columns)
        else:
            self.ts = ts.arr.wrap_array(ts, columns=columns)
        self.rsi = ts.arr.wrap_array(rsi, columns=columns)

    def is_rsi_above(self, thresholds):
        return compare_with_thresholds(above_thresholds_nb, self.rsi, thresholds)

    def is_rsi_below(self, thresholds):
        return compare_with_thresholds(below_thresholds_nb, self.rsi, thresholds)

    def plot(self,
             rsi_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        # Checks and preprocessing
        check_type(self.rsi, pd.Series)

        rsi_scatter_kwargs = {**dict(name='RSI'), **rsi_scatter_kwargs}  # user kwargs override default kwargs

        fig = self.rsi.timeseries.plot(scatter_kwargs=rsi_scatter_kwargs, fig=fig, **layout_kwargs)

        return fig
