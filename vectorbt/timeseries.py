from vectorbt.utils import *
from vectorbt.widgets import FigureWidget
import numpy as np
import pandas as pd
import inspect
import sys
from numba import njit, guvectorize, f8, i8, b1
import plotly.graph_objects as go

__all__ = []

# ############# Numba functions ############# #

# Although we don't need most of the following 1d functions here, they are needed by other modules


@njit(f8[:](f8[:], i8, f8), cache=True)
def prepend_1d_nb(a, n, value):
    """Prepend n values to the array."""
    b = np.full(a.shape[0]+n, value)
    b[n:] = a
    return b


@njit(f8[:, :](f8[:, :], i8, f8), cache=True)
def prepend_2d_nb(a, n, value):
    b = np.full((a.shape[0]+n, a.shape[1]), value)
    b[n:, :] = a
    return b


@njit(f8[:, :](f8[:], i8), cache=True)
def rolling_window_1d_nb(a, window):
    """Rolling window over the array."""
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

# Functions below have shape in = shape out


@njit(f8[:](f8[:], b1[:], f8), cache=True)
def set_by_mask_1d_nb(a, mask, value):
    """Set value by boolean mask."""
    b = a.copy()
    b[mask] = value
    return b


@njit(f8[:, :](f8[:, :], b1[:, :], f8), cache=True)
def set_by_mask_2d_nb(a, mask, value):
    b = a.copy()
    for col in range(b.shape[1]):
        b[mask[:, col], col] = value
    return b


@njit(f8[:](f8[:], f8), cache=True)
def fillna_1d_nb(a, value):
    """Fill NaNs with value."""
    return set_by_mask_1d_nb(a, np.isnan(a), value)


@njit(f8[:, :](f8[:, :], f8), cache=True)
def fillna_2d_nb(a, value):
    return set_by_mask_2d_nb(a, np.isnan(a), value)


@njit(f8[:](f8[:], i8), cache=True)
def fshift_1d_nb(a, n):
    """Shift forward by n."""
    b = np.full_like(a, np.nan)
    b[n:] = a[:-n]
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def fshift_2d_nb(a, n):
    b = np.full_like(a, np.nan)
    b[n:, :] = a[:-n, :]
    return b


@njit(f8[:](f8[:]), cache=True)
def diff_1d_nb(a):
    """Calculate the n-th discrete difference."""
    b = np.full_like(a, np.nan)
    b[1:] = np.diff(a.copy())
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def diff_2d_nb(a):
    b = np.full_like(a, np.nan)
    for col in range(a.shape[1]):
        b[1:, col] = np.diff(a[:, col].copy())
    return b


@njit(f8[:](f8[:]), cache=True)
def pct_change_1d_nb(a):
    """Compute the percentage change."""
    b = np.full_like(a, np.nan)
    b[1:] = np.diff(a.copy()) / a[:-1]
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def pct_change_2d_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = pct_change_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:]), cache=True)
def ffill_1d_nb(a):
    """Fill NaNs with the last value."""
    b = np.full_like(a, np.nan)
    maxval = a[0]
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            b[i] = maxval
        else:
            b[i] = a[i]
            maxval = b[i]
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def ffill_2d_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = ffill_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:]), cache=True)
def cumsum_1d_nb(a):
    """Cumulative sum."""
    b = np.full_like(a, np.nan)
    cumsum = 0
    for i in range(a.shape[0]):
        if ~np.isnan(a[i]):
            cumsum += a[i]
            b[i] = cumsum
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def cumsum_2d_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = cumsum_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:]), cache=True)
def cumprod_1d_nb(a):
    """Cumulative product."""
    b = np.full_like(a, np.nan)
    cumprod = 1
    for i in range(a.shape[0]):
        if ~np.isnan(a[i]):
            cumprod *= a[i]
            b[i] = cumprod
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def cumprod_2d_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = cumprod_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:], i8), cache=True)
def rolling_mean_1d_nb(a, window):
    """Rolling mean."""
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
def rolling_mean_2d_nb(a, window):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = rolling_mean_1d_nb(a[:, col], window)
    return b


@njit(f8[:](f8[:], i8), cache=True)
def rolling_std_1d_nb(a, window):
    """Rolling std for ddof = 0."""
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
def rolling_std_2d_nb(a, window):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = rolling_std_1d_nb(a[:, col], window)
    return b


@njit(f8[:](f8[:]), cache=True)
def expanding_max_1d_nb(a):
    """Expanding max."""
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
def expanding_max_2d_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = expanding_max_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:], i8), cache=True)
def ewm_mean_1d_nb(vals, span):
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
def ewm_mean_2d_nb(a, span):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = ewm_mean_1d_nb(a[:, col], span)
    return b


@njit(f8[:](f8[:], i8), cache=True)
def ewm_std_1d_nb(vals, span):
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
def ewm_std_2d_nb(a, span):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = ewm_std_1d_nb(a[:, col], span)
    return b

# ############# Custom accessors ############# #


@add_safe_nb_methods(
    fillna_2d_nb,
    fshift_2d_nb,
    diff_2d_nb,
    pct_change_2d_nb,
    ffill_2d_nb,
    cumsum_2d_nb,
    cumprod_2d_nb,
    rolling_mean_2d_nb,
    rolling_std_2d_nb,
    expanding_max_2d_nb,
    ewm_mean_2d_nb,
    ewm_std_2d_nb)
class TimeSeries_Accessor():
    dtype = np.float64

    @classmethod
    def _validate(cls, obj):
        if cls.dtype is not None:
            check_dtype(obj, cls.dtype)

    def rolling_window(self, window, step=1):
        """Generate a new DataFrame from a rolling window."""
        strided = rolling_window_1d_nb(self.to_1d_array(), window)
        columns = np.arange(strided.shape[0])[::step]
        rolled = strided.transpose()[:, ::step]
        index = np.arange(rolled.shape[0])
        return self.wrap_array(rolled, index=index, columns=columns)


@pd.api.extensions.register_dataframe_accessor("timeseries")
class TimeSeries_DFAccessor(TimeSeries_Accessor, Base_DFAccessor):

    def plot(self, scatter_kwargs={}, fig=None, **layout_kwargs):
        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].timeseries.plot(
                scatter_kwargs=scatter_kwargs[col] if isinstance(scatter_kwargs, list) else scatter_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig


@pd.api.extensions.register_series_accessor("timeseries")
class TimeSeries_SRAccessor(TimeSeries_Accessor, Base_SRAccessor):

    def plot(self, scatter_kwargs={}, fig=None, **layout_kwargs):
        if fig is None:
            fig = FigureWidget()
            fig.update_layout(**layout_kwargs)
        if self._obj.name is not None:
            fig.update_layout(showlegend=True)

        scatter = go.Scatter(
            x=self._obj.index,
            y=self._obj.values,
            mode='lines',
            name=str(self._obj.name) if self._obj.name is not None else None
        )
        scatter.update(**scatter_kwargs)
        fig.add_trace(scatter)

        return fig
