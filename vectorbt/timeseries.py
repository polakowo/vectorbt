from vectorbt.decorators import *
from vectorbt.widgets import FigureWidget
import numpy as np
import pandas as pd
import inspect
import sys
from numba import njit, f8, i8, b1
import plotly.graph_objects as go

# ############# Numba functions ############# #

# All functions below require input to be a two-dimensional NumPy array,
# where first axis is index and second axis are columns.
# They all move along the first axis (axis=0)


@njit(f8[:, :](f8[:, :], b1[:, :], f8), cache=True)
def set_by_mask_nb(a, mask, value):
    """Set value by 2D boolean mask."""
    b = a.copy()
    for j in range(b.shape[1]):
        b[mask[:, j], j] = value
    return b


@njit(f8[:, :](f8[:, :], f8), cache=True)
def fillna_nb(a, fill_value):
    """Fill NaNs with fill_value."""
    return set_by_mask_nb(a, np.isnan(a), fill_value)


@njit(f8[:, :](f8[:, :], i8, f8), cache=True)
def prepend_nb(a, n, fill_value):
    """Prepend n values to the array."""
    b = np.full((a.shape[0]+n, a.shape[1]), fill_value, dtype=a.dtype)
    b[-a.shape[0]:] = a
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def fshift_nb(a, n):
    """Shift forward by n."""
    a = prepend_nb(a, n, np.nan)
    return a[:-n, :]


@njit(f8[:, :](f8[:, :], i8), cache=True)
def diff_nb(a, n):
    """Calculate the n-th discrete difference."""
    b = np.full_like(a, np.nan)
    for i in range(a.shape[1]):
        b[n:, i] = np.diff(a[:, i].copy(), n=n)
    return b


@njit(f8[:](f8[:]), cache=True)
def _pct_change_1d_nb(a):
    """Compute the percentage change (1D)."""
    return np.concatenate((np.full(1, np.nan), np.diff(a.copy()) / a[:-1]))


@njit(f8[:, :](f8[:, :]), cache=True)
def pct_change_nb(a):
    """Compute the percentage change."""
    b = np.empty_like(a)
    for i in range(a.shape[1]):
        b[:, i] = _pct_change_1d_nb(a[:, i])
    return b


@njit(f8[:](f8[:]), cache=True)
def _ffill_1d_nb(a):
    """Fill NaNs with the last value (1D)."""
    b = np.empty_like(a)
    maxval = a[0]
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            b[i] = maxval
        else:
            b[i] = a[i]
            maxval = b[i]
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def ffill_nb(a):
    """Fill NaNs with the last value."""
    b = np.empty_like(a)
    for j in range(a.shape[1]):
        b[:, j] = _ffill_1d_nb(a[:, j])
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def cumsum_nb(a):
    """Cumulative sum."""
    b = np.empty_like(a, dtype=a.dtype)
    for j in range(a.shape[1]):
        b[:, j] = np.nancumsum(a[:, j])
        b[np.isnan(a[:, j]), j] = np.nan
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def cumprod_nb(a):
    """Cumulative product."""
    b = np.empty_like(a, dtype=a.dtype)
    for j in range(a.shape[1]):
        b[:, j] = np.nancumprod(a[:, j])
        b[np.isnan(a[:, j]), j] = np.nan
    return b


@njit(f8[:, :](f8[:], i8), cache=True)
def _rolling_window_1d_nb(a, window):
    """Rolling window over the array."""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


@njit(f8[:](f8[:], i8), cache=True)
def _rolling_mean_1d_nb(a, window):
    """Rolling mean (1D)."""
    b = np.empty_like(a)
    cumsum_arr = np.zeros_like(a)
    cumsum = 0
    nancnt_arr = np.zeros_like(a)
    nancnt = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            nancnt = nancnt + 1
        nancnt_arr[i] = nancnt
        cumsum = cumsum + (0 if np.isnan(a[i]) else a[i])
        cumsum_arr[i] = cumsum
        if i < window:
            window_len = i + 1 - nancnt  # think of nanmean
            window_cumsum = cumsum
        else:
            window_len = window - (nancnt - nancnt_arr[i-window])
            window_cumsum = cumsum - cumsum_arr[i-window]
        if window_len == 0:
            b[i] = np.nan
            continue
        b[i] = window_cumsum / window_len
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def rolling_mean_nb(a, window):
    """Rolling mean."""
    b = np.empty_like(a)
    for j in range(a.shape[1]):
        b[:, j] = _rolling_mean_1d_nb(a[:, j], window)
    return b


@njit(f8[:](f8[:], i8), cache=True)
def _rolling_std_1d_nb(a, window):
    """Rolling std (1D) for ddof = 0."""
    b = np.empty_like(a)
    cumsum_arr = np.zeros_like(a)
    cumsum = 0
    cumsum_sq_arr = np.zeros_like(a)
    cumsum_sq = 0
    nancnt_arr = np.zeros_like(a)
    nancnt = 0
    mean = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            nancnt = nancnt + 1
        nancnt_arr[i] = nancnt
        cumsum = cumsum + (0 if np.isnan(a[i]) else a[i])
        cumsum_arr[i] = cumsum
        cumsum_sq = cumsum_sq + (0 if np.isnan(a[i]) else a[i] ** 2)
        cumsum_sq_arr[i] = cumsum_sq
        if i < window:
            window_len = i + 1 - nancnt  # think of nanmean
            window_cumsum = cumsum
            window_cumsum_sq = cumsum_sq
        else:
            window_len = window - (nancnt - nancnt_arr[i-window])
            window_cumsum = cumsum - cumsum_arr[i-window]
            window_cumsum_sq = cumsum_sq - cumsum_sq_arr[i-window]
        if window_len == 0:
            b[i] = np.nan
            continue
        mean = window_cumsum / window_len
        b[i] = np.sqrt(np.abs(window_cumsum_sq - 2 * window_cumsum * mean + window_len * mean ** 2) / window_len)
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def rolling_std_nb(a, window):
    """Rolling std."""
    b = np.empty_like(a)
    for j in range(a.shape[1]):
        b[:, j] = _rolling_std_1d_nb(a[:, j], window)
    return b


@njit(f8[:](f8[:]), cache=True)
def _expanding_max_1d_nb(a):
    """Expanding max (1D)."""
    b = np.empty_like(a)
    maxv = np.nan
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            if np.isnan(maxv):
                b[i] = np.nan
                continue
        else:
            if np.isnan(maxv) or a[i] > maxv:
                maxv = a[i]
        b[i] = maxv
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def expanding_max_nb(a):
    """Expanding max."""
    b = np.empty_like(a)
    for j in range(a.shape[1]):
        b[:, j] = _expanding_max_1d_nb(a[:, j])
    return b


@njit(f8[:](f8[:], i8), cache=True)
def _ewm_mean_1d_nb(vals, span):
    """Adaptation of pandas._libs.window.aggregations.window_aggregations.ewma with default params."""
    N = len(vals)
    output = np.empty(N, dtype=f8)
    if N == 0:
        return output
    com = (span - 1) / 2.0
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1.
    weighted_avg = vals[0]
    is_observation = (weighted_avg == weighted_avg)
    output[0] = weighted_avg
    old_wt = 1.

    for i in range(1, N):
        cur = vals[i]
        is_observation = (cur == cur)
        if weighted_avg == weighted_avg:
            old_wt *= old_wt_factor
            if is_observation:
                # avoid numerical errors on constant series
                if weighted_avg != cur:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (old_wt + new_wt)
                old_wt += new_wt
        elif is_observation:
            weighted_avg = cur
        output[i] = weighted_avg
    return output


@njit(f8[:, :](f8[:, :], i8), cache=True)
def ewm_mean_nb(a, span):
    """Exponential weighted moving average."""
    b = np.empty_like(a)
    for i in range(a.shape[1]):
        b[:, i] = _ewm_mean_1d_nb(a[:, i], span)
    return b


@njit(f8[:](f8[:], i8), cache=True)
def _ewm_std_1d_nb(vals, span):
    """Adaptation of pandas._libs.window.aggregations.window_aggregations.ewmcov with default params."""
    N = len(vals)
    output = np.empty(N, dtype=f8)
    if N == 0:
        return output
    com = (span - 1) / 2.0
    minp = 1.
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1.
    mean_x = vals[0]
    mean_y = vals[0]
    is_observation = ((mean_x == mean_x) and (mean_y == mean_y))
    nobs = int(is_observation)
    if not is_observation:
        mean_x = np.nan
        mean_y = np.nan
    output[0] = np.nan
    cov = 0.
    sum_wt = 1.
    sum_wt2 = 1.
    old_wt = 1.

    for i in range(1, N):
        cur_x = vals[i]
        cur_y = vals[i]
        is_observation = ((cur_x == cur_x) and (cur_y == cur_y))
        nobs += is_observation
        if mean_x == mean_x:
            sum_wt *= old_wt_factor
            sum_wt2 *= (old_wt_factor * old_wt_factor)
            old_wt *= old_wt_factor
            if is_observation:
                old_mean_x = mean_x
                old_mean_y = mean_y

                # avoid numerical errors on constant series
                if mean_x != cur_x:
                    mean_x = ((old_wt * old_mean_x) +
                              (new_wt * cur_x)) / (old_wt + new_wt)

                # avoid numerical errors on constant series
                if mean_y != cur_y:
                    mean_y = ((old_wt * old_mean_y) +
                              (new_wt * cur_y)) / (old_wt + new_wt)
                cov = ((old_wt * (cov + ((old_mean_x - mean_x) *
                                         (old_mean_y - mean_y)))) +
                       (new_wt * ((cur_x - mean_x) *
                                  (cur_y - mean_y)))) / (old_wt + new_wt)
                sum_wt += new_wt
                sum_wt2 += (new_wt * new_wt)
                old_wt += new_wt
        elif is_observation:
            mean_x = cur_x
            mean_y = cur_y

        if nobs >= minp:
            numerator = sum_wt * sum_wt
            denominator = numerator - sum_wt2
            if (denominator > 0.):
                output[i] = ((numerator / denominator) * cov)
            else:
                output[i] = np.nan
        else:
            output[i] = np.nan

    return np.sqrt(output)


@njit(f8[:, :](f8[:, :], i8), cache=True)
def ewm_std_nb(a, span):
    """Exponential weighted moving STD."""
    b = np.empty_like(a)
    for i in range(a.shape[1]):
        b[:, i] = _ewm_std_1d_nb(a[:, i], span)
    return b

# ############# Main class ############# #


# List numba functions in current module
nb_funcs = [
    obj
    for name, obj in inspect.getmembers(sys.modules[__name__])
    if '_nb' in name and not name.startswith('_')
]

# Add numba functions as methods to the TimeSeries class
@add_nb_methods(*nb_funcs)
class TimeSeries(np.ndarray):
    """Similar to pd.DataFrame, but optimized for complex matrix operations.
    NOTE: There is no index nor columns vars - you will have to handle them separately."""

    @to_2d('input_array')
    @has_dtype('input_array', np.float64)
    def __new__(cls, input_array):
        return np.asarray(input_array).view(cls)

    @classmethod
    def full(cls, *args, **kwargs):
        return cls(np.full(*args, **kwargs))

    @classmethod
    def full_like(cls, *args, **kwargs):
        return cls(np.full_like(*args, **kwargs))

    @classmethod
    @to_1d('input_array')
    @has_dtype('input_array', np.float64)
    def from_rolling_window(cls, input_array, window, step=1):
        strided = _rolling_window_1d_nb(input_array, window)
        return cls(strided.transpose()[:, ::step])

    @to_2d('self')
    @to_2d('benchmark')
    @broadcast_to('benchmark', 'self')
    def plot(self,
             column=None,
             label='TimeSeries',
             benchmark=None,
             benchmark_label='Benchmark',
             index=None,
             benchmark_scatter_kwargs={},
             ts_scatter_kwargs={},
             fig=None, 
             **layout_kwargs):

        if column is None:
            if self.shape[1] == 1:
                column = 0
            else:
                raise ValueError("For an array with multiple columns, you must pass a column index")
        ts = self[:, column]
        if benchmark is not None:
            benchmark = benchmark[:, column]

        if index is None:
            index = np.arange(ts.shape[0])
        if fig is None:
            fig = FigureWidget()
            fig.update_layout(showlegend=True)
            fig.update_layout(**layout_kwargs)

        if benchmark is not None:
            # Plot benchmark
            benchmark_scatter = go.Scatter(
                x=index,
                y=benchmark,
                mode='lines',
                name=benchmark_label,
                line_color='#ff7f0e'
            )
            benchmark_scatter.update(**benchmark_scatter_kwargs)
            fig.add_trace(benchmark_scatter)
            _min, _max = np.min(np.concatenate((ts, benchmark))), np.max(np.concatenate((ts, benchmark)))
        else:
            _min, _max = np.min(ts), np.max(ts)
        # Plot TimeSeries
        ts_scatter = go.Scatter(
            x=index,
            y=ts,
            mode='lines',
            name=label,
            line_color='#1f77b4'
        )
        ts_scatter.update(**ts_scatter_kwargs)
        fig.add_trace(ts_scatter)

        # Adjust y-axis
        space = 0.05 * (_max - _min)
        fig.update_yaxes(range=[_min - space, _max + space])

        return fig
