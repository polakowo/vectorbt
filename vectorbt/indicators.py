import pandas as pd
import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1
from copy import copy
import plotly.graph_objects as go
import itertools

from vectorbt.utils import *
from vectorbt.accessors import *
from vectorbt.timeseries import rolling_mean_nb, rolling_std_nb, ewm_mean_nb, \
    ewm_std_nb, diff_nb, set_by_mask_nb, prepend_nb

__all__ = ['MA', 'BollingerBands', 'RSI']

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


def compare_with_thresholds(compare_func, pd_obj, thresholds, level_name='threshold'):
    thresholds = broadcast_to_array_of(thresholds, pd_obj.vbt.to_2d_array())
    thresholds = thresholds.astype(np.float64)

    # Compare each threshold matrix with pandas object and stack the results horizontally
    a = compare_func(pd_obj.vbt.to_2d_array(), thresholds)

    # Build column hierarchy
    param_columns = index_from_params(thresholds, name=level_name)
    columns = combine_indexes(param_columns, to_2d(pd_obj).columns)

    return pd_obj.vbt.wrap_array(a, columns=columns)

# ############# MA ############# #


@njit(f8[:, :](f8[:, :], i8[:], b1, b1), cache=True)
def ma_nb(ts, windows, is_ewm, is_min_periods):
    """For each window, calculate the corresponding SMA/EMA."""
    # Cache moving averages to effectively reduce the number of operations
    unique_windows = np.unique(windows)
    cache_d = dict()
    for i in range(unique_windows.shape[0]):
        if is_ewm:
            ma = ewm_mean_nb(ts, unique_windows[i])
        else:
            ma = rolling_mean_nb(ts, unique_windows[i])
        if is_min_periods:
            ma[:unique_windows[i], :] = np.nan
        cache_d[unique_windows[i]] = ma
    # Stack moving averages out of cache and return
    mas = np.empty((ts.shape[0], ts.shape[1] * windows.shape[0]), dtype=f8)
    for i in range(windows.shape[0]):
        mas[:, i*ts.shape[1]:(i+1)*ts.shape[1]] = cache_d[windows[i]]
    return mas


def ma_indexing_func(obj, loc_pandas_func):
    window_mapper = loc_mapper(obj.window_mapper, obj.ts, loc_pandas_func)
    return MA(loc_pandas_func(obj.ts), loc_pandas_func(obj.ma), window_mapper)


@add_indexing(ma_indexing_func)
@add_param_indexing('window', ma_indexing_func)
class MA():
    """A moving average (MA) is a widely used indicator in technical analysis that helps smooth out 
    price action by filtering out the “noise” from random short-term price fluctuations. 
    It is a trend-following, or lagging, indicator because it is based on past prices."""

    def __init__(self, ts, ma, window_mapper):
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
        check_same_meta(ts, ma)
        check_type(window_mapper, pd.Series)
        check_same_columns(to_2d(ts), to_2d(window_mapper).transpose())

        self.ts = ts
        self.ma = ma
        self.window_mapper = window_mapper

    @classmethod
    def from_params(cls, ts, windows, is_ewm=False, is_min_periods=True, level_name='window', from_ma=None):
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
        check_level_not_exists(ts, level_name)
        # windows can be either np.ndarray or single number
        windows = to_1d(windows)
        check_dtype(windows, np.int64)

        if from_ma is not None:
            # Use another MA to take windows from there, good for caching
            ma = from_ma.window_loc(clean_columns=False)[windows].ma.to_numpy()
        else:
            # Calculate MA from scratch
            ma = ma_nb(ts.vbt.to_2d_array(), windows, is_ewm, is_min_periods)

        # Build column hierarchy
        param_columns = index_from_params(windows, name=level_name)
        columns = combine_indexes(param_columns, to_2d(ts).columns)

        # Build window column mapper
        window_mapper = np.repeat(windows, len(to_2d(ts).columns))
        window_mapper = pd.Series(window_mapper, index=columns)

        ma = ts.vbt.wrap_array(ma, columns=columns)
        if windows.shape[0] > 1:
            ts = ts.vbt.wrap_array(np.tile(to_2d(ts), (1, windows.shape[0])), columns=columns)
        else:
            ts = ts.vbt.wrap_array(ts, columns=columns)

        return cls(ts, ma, window_mapper)

    @classmethod
    def from_combinations(cls, ts, windows, r, level_names=None, **kwargs):
        windows = to_1d(windows)
        check_dtype(windows, np.int64)
        window_lists = list(zip(*list(itertools.combinations(windows, r))))

        ma = cls.from_params(ts, windows, **kwargs)

        for i, window_list in enumerate(window_lists):
            if level_names is not None:
                yield cls.from_params(ts, window_list, level_name=level_names[i], from_ma=ma)
            else:
                yield cls.from_params(ts, window_list, from_ma=ma)

    @property
    def window_loc(self):
        return self._window_loc

    def is_above(self, other, **kwargs):
        if isinstance(other, MA):
            other = other.ma
        return self.ma.vbt.combine_with(other, np_combine_func=np.greater, **kwargs)

    def is_below(self, other, **kwargs):
        if isinstance(other, MA):
            other = other.ma
        return self.ma.vbt.combine_with(other, np_combine_func=np.less, **kwargs)

    def __gt__(self, other):
        return self.is_above(other)

    def __lt__(self, other):
        return self.is_below(other)

    def crossover_signals(self, other, **kwargs):
        # entry signal is first time this is about other
        entries = self.is_above(other, **kwargs).vbt.signals.first_true(after_false=True)
        # exit signal is first time this is below other
        exits = self.is_below(other, **kwargs).vbt.signals.first_true(after_false=True)
        return entries, exits

    def plot(self,
             plot_ts=True,
             ts_name='Price',
             ma_name='MA',
             ts_scatter_kwargs={},
             ma_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.ts, pd.Series)
        check_type(self.ma, pd.Series)

        if plot_ts:
            fig = self.ts.vbt.timeseries.plot(name=ts_name, scatter_kwargs=ts_scatter_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.timeseries.plot(name=ma_name, scatter_kwargs=ma_scatter_kwargs, fig=fig)

        return fig

    def plot_crossover_signals(self, other,
                               other_name=None,
                               other_scatter_kwargs={},
                               entry_scatter_kwargs={},
                               exit_scatter_kwargs={},
                               fig=None,
                               **plot_kwargs):
        check_type(self.ts, pd.Series)
        check_type(self.ma, pd.Series)
        if isinstance(other, MA):
            other = other.ma
            if other_name is None:
                other_name = 'Other MA'
        else:
            if other_name is None:
                other_name = 'Other'
        check_type(other, pd.Series)

        fig = self.plot(**plot_kwargs)
        fig = other.vbt.timeseries.plot(name=other_name, scatter_kwargs=other_scatter_kwargs, fig=fig)

        # Plot markets
        entries, exits = self.crossover_signals(other)
        entry_scatter = go.Scatter(
            x=self.ts.index[entries],
            y=self.ts[entries],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                color='limegreen',
                size=10
            ),
            name='Entry'
        )
        entry_scatter.update(**entry_scatter_kwargs)
        fig.add_trace(entry_scatter)
        exit_scatter = go.Scatter(
            x=self.ts.index[exits],
            y=self.ts[exits],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                color='orangered',
                size=10
            ),
            name='Exit'
        )
        exit_scatter.update(**exit_scatter_kwargs)
        fig.add_trace(exit_scatter)

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


def bb_indexing_func(obj, loc_pandas_func):
    window_mapper = loc_mapper(obj.window_mapper, obj.ts, loc_pandas_func)
    std_n_mapper = loc_mapper(obj.std_n_mapper, obj.ts, loc_pandas_func)
    param_mapper = loc_mapper(obj.param_mapper, obj.ts, loc_pandas_func)
    return BollingerBands(loc_pandas_func(obj.ts),
                          loc_pandas_func(obj.upper),
                          loc_pandas_func(obj.middle),
                          loc_pandas_func(obj.lower),
                          window_mapper,
                          std_n_mapper,
                          param_mapper)


@add_indexing(bb_indexing_func)
@add_param_indexing('window', bb_indexing_func)
@add_param_indexing('std_n', bb_indexing_func)
@add_param_indexing('param', bb_indexing_func)
class BollingerBands():
    """Bollinger Bands® are volatility bands placed above and below a moving average."""

    def __init__(self, ts, upper, middle, lower, window_mapper, std_n_mapper, param_mapper):
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
        check_same_meta(ts, upper)
        check_same_meta(ts, middle)
        check_same_meta(ts, lower)
        check_type(window_mapper, pd.Series)
        check_type(std_n_mapper, pd.Series)
        check_type(param_mapper, pd.Series)
        check_same_columns(to_2d(ts), to_2d(window_mapper).transpose())
        check_same_columns(to_2d(ts), to_2d(std_n_mapper).transpose())
        check_same_columns(to_2d(ts), to_2d(param_mapper).transpose())

        self.ts = ts
        self.upper = upper
        self.middle = middle
        self.lower = lower
        self.window_mapper = window_mapper
        self.std_n_mapper = std_n_mapper
        self.param_mapper = param_mapper

    @classmethod
    def from_params(cls, ts, windows, std_ns, is_ewm=False, is_min_periods=True, level_names=('window', 'std_n')):
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
        check_type(level_names, (list, tuple))
        check_level_not_exists(ts, level_names[0])
        check_level_not_exists(ts, level_names[1])
        windows = to_1d(windows)
        std_ns = to_1d(std_ns)
        windows, std_ns = broadcast(windows, std_ns)
        check_dtype(windows, np.int64)
        check_dtype(std_ns, np.int64)

        # windows and std_ns can be either np.ndarray or single number
        upper, middle, lower = bb_nb(ts.vbt.to_2d_array(), windows, std_ns, is_ewm, is_min_periods)

        # Build column hierarchy
        window_index = index_from_params(windows, name='window')
        std_n_index = index_from_params(std_ns, name='std_n')
        param_columns = stack_indexes(window_index, std_n_index)
        columns = combine_indexes(param_columns, to_2d(ts).columns)

        # Build window column mapper
        window_mapper = np.repeat(windows, len(to_2d(ts).columns))
        window_mapper = pd.Series(window_mapper, index=columns)
        std_n_mapper = np.repeat(std_ns, len(to_2d(ts).columns))
        std_n_mapper = pd.Series(std_n_mapper, index=columns)
        param_mapper = list(zip(window_mapper.values, std_n_mapper.values))
        param_mapper = pd.Series(param_mapper, index=columns)

        upper = ts.vbt.wrap_array(upper, columns=columns)
        middle = ts.vbt.wrap_array(middle, columns=columns)
        lower = ts.vbt.wrap_array(lower, columns=columns)
        if windows.shape[0] > 1:
            ts = ts.vbt.wrap_array(np.tile(to_2d(ts), (1, windows.shape[0])), columns=columns)
        else:
            ts = ts.vbt.wrap_array(ts, columns=columns)

        return cls(ts, upper, middle, lower, window_mapper, std_n_mapper, param_mapper)

    @cached_property
    def percent_b(self):
        """Shows where price is in relation to the bands.
        %b equals 1 at the upper band and 0 at the lower band."""
        return (self.ts - self.lower) / (self.upper - self.lower)

    @cached_property
    def bandwidth(self):
        """Bandwidth tells how wide the Bollinger Bands are on a normalized basis."""
        return (self.upper - self.lower) / self.middle

    def is_percent_b_above(self, thresholds, **kwargs):
        # thresholds can be a number, an array, a matrix of shape self.percent_b (element-wise), or an array of it (3-dim)
        return compare_with_thresholds(above_thresholds_nb, self.percent_b, thresholds, **kwargs)

    def is_percent_b_below(self, thresholds, **kwargs):
        return compare_with_thresholds(below_thresholds_nb, self.percent_b, thresholds, **kwargs)

    def is_bandwidth_above(self, thresholds, **kwargs):
        return compare_with_thresholds(above_thresholds_nb, self.bandwidth, thresholds, **kwargs)

    def is_bandwidth_below(self, thresholds, **kwargs):
        return compare_with_thresholds(below_thresholds_nb, self.bandwidth, thresholds, **kwargs)

    def plot(self,
             plot_ts=True,
             ts_name='Price',
             upper_name='Upper Band',
             middle_name='Middle Band',
             lower_name='Lower Band',
             ts_scatter_kwargs={},
             upper_scatter_kwargs={},
             middle_scatter_kwargs={},
             lower_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.ts, pd.Series)
        check_type(self.upper, pd.Series)
        check_type(self.middle, pd.Series)
        check_type(self.lower, pd.Series)

        upper_scatter_kwargs = {**dict(line=dict(color='grey')), **upper_scatter_kwargs}  # default kwargs
        lower_scatter_kwargs = {**dict(line=dict(color='grey')), **lower_scatter_kwargs}

        if plot_ts:
            fig = self.ts.vbt.timeseries.plot(name=ts_name, scatter_kwargs=ts_scatter_kwargs, fig=fig, **layout_kwargs)
        fig = self.upper.vbt.timeseries.plot(name=upper_name, scatter_kwargs=upper_scatter_kwargs, fig=fig)
        fig = self.middle.vbt.timeseries.plot(name=middle_name, scatter_kwargs=middle_scatter_kwargs, fig=fig)
        fig = self.lower.vbt.timeseries.plot(name=lower_name, scatter_kwargs=lower_scatter_kwargs, fig=fig)

        return fig


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


def rsi_indexing_func(obj, loc_pandas_func):
    window_mapper = loc_mapper(obj.window_mapper, obj.ts, loc_pandas_func)
    return RSI(loc_pandas_func(obj.ts), loc_pandas_func(obj.rsi), window_mapper)


@add_indexing(rsi_indexing_func)
@add_param_indexing('window', rsi_indexing_func)
class RSI():
    """The relative strength index (RSI) is a momentum indicator that 
    measures the magnitude of recent price changes to evaluate overbought 
    or oversold conditions in the price of a stock or other asset."""

    def __init__(self, ts, rsi, window_mapper):
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
        check_same_meta(ts, rsi)
        check_type(window_mapper, pd.Series)
        check_same_columns(to_2d(ts), to_2d(window_mapper).transpose())

        self.ts = ts
        self.rsi = rsi
        self.window_mapper = window_mapper

    @classmethod
    def from_params(cls, ts, windows, is_ewm=False, is_min_periods=True, level_name='window'):
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
        check_level_not_exists(ts, level_name)
        # windows can be either np.ndarray or single number
        windows = to_1d(windows)
        check_dtype(windows, np.int64)

        rsi = rsi_nb(ts.vbt.to_2d_array(), windows, is_ewm, is_min_periods)

        # Build column hierarchy
        param_columns = index_from_params(windows, name=level_name)
        columns = combine_indexes(param_columns, to_2d(ts).columns)

        # Build window column mapper
        window_mapper = np.repeat(windows, len(to_2d(ts).columns))
        window_mapper = pd.Series(window_mapper, index=columns)

        rsi = ts.vbt.wrap_array(rsi, columns=columns)
        if windows.shape[0] > 1:
            ts = ts.vbt.wrap_array(np.tile(to_2d(ts), (1, windows.shape[0])), columns=columns)
        else:
            ts = ts.vbt.wrap_array(ts, columns=columns)

        return cls(ts, rsi, window_mapper)

    def is_rsi_above(self, thresholds):
        return compare_with_thresholds(above_thresholds_nb, self.rsi, thresholds)

    def is_rsi_below(self, thresholds):
        return compare_with_thresholds(below_thresholds_nb, self.rsi, thresholds)

    def plot(self,
             rsi_name='RSI',
             rsi_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.rsi, pd.Series)

        fig = self.rsi.vbt.timeseries.plot(name=rsi_name, scatter_kwargs=rsi_scatter_kwargs, fig=fig)

        return fig
