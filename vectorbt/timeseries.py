from vectorbt.utils.decorators import *
import numpy as np
import pandas as pd
import inspect
import sys
from numba import njit, f8, i8, b1, optional
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

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


@njit(f8[:, :](f8[:, :]))
def diff_nb(a):
    b = np.full_like(a, np.nan)
    for i in range(a.shape[1]):
        b[1:, i] = np.diff(a[:, i].copy())
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def pct_change_nb(a):
    """Compute the percentage change from the immediately previous row."""
    b = np.full_like(a, np.nan)
    for i in range(a.shape[1]):
        b[1:, i] = np.diff(a[:, i].copy()) / a[:-1, i]
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def ffill_nb(a):
    """Fill NaNs with the last value."""
    b = np.empty_like(a)
    for j in range(a.shape[1]):
        maxval = a[0, j]
        for i in range(a.shape[0]):
            if np.isnan(a[i, j]):
                b[i, j] = maxval
            else:
                b[i, j] = a[i, j]
                maxval = b[i, j]
    return b


@njit(f8[:, :, :](f8[:, :], i8), cache=True)
def _rolling_window_nb(a, window):
    """Rolling window over the array.

    Produce the pandas result when min_periods=1."""
    a = prepend_nb(a, window-1, np.nan)
    shape = (a.shape[0] - int(window) + 1, int(window)) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    strided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return strided  # raw output


@njit(f8[:, :](f8[:, :, :]), cache=True)
def _nanmean_nb(a):
    # numba doesn't support axis kwarg in np.nan_ functions
    b = np.empty((a.shape[0], a.shape[2]), dtype=a.dtype)
    for j in range(a.shape[2]):
        for i in range(a.shape[0]):
            b[i, j] = np.nanmean(a[i, :, j])
    return b


@njit(f8[:, :](f8[:, :, :]), cache=True)
def _nanstd_nb(a):
    # numba doesn't support axis kwarg in np.nan_ functions
    b = np.empty((a.shape[0], a.shape[2]), dtype=a.dtype)
    for j in range(a.shape[2]):
        for i in range(a.shape[0]):
            b[i, j] = np.nanstd(a[i, :, j])
    return b


@njit(f8[:, :](f8[:, :, :]), cache=True)
def _nanmax_nb(a):
    # numba doesn't support axis kwarg in np.nan_ functions
    b = np.empty((a.shape[0], a.shape[2]), dtype=a.dtype)
    for j in range(a.shape[2]):
        for i in range(a.shape[0]):
            b[i, j] = np.nanmax(a[i, :, j])
    return b


@njit(f8[:, :](f8[:, :], optional(i8)), cache=True)
def rolling_mean_nb(a, window):
    if window is None:  # expanding
        window = a.shape[0]
    return _nanmean_nb(_rolling_window_nb(a, window))


@njit(f8[:, :](f8[:, :], optional(i8)), cache=True)
def rolling_std_nb(a, window):
    if window is None:  # expanding
        window = a.shape[0]
    return _nanstd_nb(_rolling_window_nb(a, window))


@njit(f8[:, :](f8[:, :], optional(i8)), cache=True)
def rolling_max_nb(a, window):
    if window is None:  # expanding
        window = a.shape[0]
    return _nanmax_nb(_rolling_window_nb(a, window))


@njit(f8[:](f8[:], i8), cache=True)
def _ewma_nb(arr_in, window):
    # https://stackoverflow.com/a/51392341/8141780
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=arr_in.dtype)
    alpha = 2 / float(window + 1)
    w = 1
    ewma_old = arr_in[0]
    ewma[0] = ewma_old
    for i in range(1, n):
        w += (1-alpha)**i
        ewma_old = ewma_old*(1-alpha) + arr_in[i]
        ewma[i] = ewma_old / w
    return ewma


@njit(f8[:](f8[:], i8), cache=True)
def _ewma_infinite_hist_nb(arr_in, window):
    # https://stackoverflow.com/a/51392341/8141780
    n = arr_in.shape[0]
    ewma = np.empty(n, dtype=arr_in.dtype)
    alpha = 2 / float(window + 1)
    ewma[0] = arr_in[0]
    for i in range(1, n):
        ewma[i] = arr_in[i] * alpha + ewma[i-1] * (1 - alpha)
    return ewma


@njit(f8[:, :](f8[:, :], optional(i8), b1), cache=True)
def ewma_nb(a, window, adjust):
    """Exponential weighted moving average."""
    b = np.empty_like(a)
    if window is None:
        window = a.shape[0]  # expanding
    for i in range(a.shape[1]):
        if adjust:
            b[:, i] = _ewma_nb(a[:, i], window)
        else:
            b[:, i] = _ewma_infinite_hist_nb(a[:, i], window)
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def cumsum_nb(a):
    """Cumulative sum (axis=0)."""
    b = np.empty_like(a, dtype=a.dtype)
    for j in range(a.shape[1]):
        b[:, j] = np.cumsum(a[:, j])
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def cumprod_nb(a):
    """Cumulative product (axis=0)."""
    b = np.empty_like(a, dtype=a.dtype)
    for j in range(a.shape[1]):
        b[:, j] = np.cumprod(a[:, j])
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
    """Base class inherited by TimeSeries, Signals and Positions.
    Similar to pd.DataFrame, but optimized for complex matrix operations.
    NOTE: There is no index nor columns vars - you will have to handle them separately."""

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        if obj.ndim == 1:
            obj = obj[:, None]  # expand
        if obj.ndim != 2:
            raise ValueError("Argument input_array must be a two-dimensional array")
        if obj.dtype != np.float64:
            raise TypeError("dtype must be np.float64")

        # Be aware: index and columns do not change when slicing, subscribing or modifying the array!
        # Since arrays are meant to be the same shape throught processing.
        # If you want to change these variables as well, do it explicitly in your code.

        # Finally, we must return the newly created object:
        return obj

    @classmethod
    def full(cls, *args, **kwargs):
        return cls(np.full(*args, **kwargs))

    @classmethod
    def full_like(cls, *args, **kwargs):
        return cls(np.full_like(*args, **kwargs))

    @to_1d('self')
    @to_1d('benchmark')
    @broadcast_to('benchmark', 'self')
    def plot(self, index=None, label=None, benchmark=None, benchmark_label=None, ax=None, **kwargs):
        """Plot TimeSeries as a line."""
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots()

        # Plot TimeSeries
        ts_df = pd.DataFrame(self)
        if index is not None:
            ts_df.index = pd.Index(index)
        if label is not None:
            ts_df.columns = [label]
        else:
            ts_df.columns = ['TimeSeries']
        ts_df.plot(ax=ax, **kwargs)

        # Plot benchmark
        if benchmark is not None:
            if isinstance(benchmark, (int, float, complex)):
                benchmark_df = pd.DataFrame(np.full(len(ts_df.index), benchmark))
                benchmark_df.columns = [str(benchmark)]
                benchmark_df.index = ts_df.index
            else:
                benchmark_df = pd.DataFrame(np.asarray(benchmark))
                benchmark_df.columns = ['Benchmark']
                benchmark_df.index = ts_df.index
            if benchmark_label is not None:
                benchmark_df.columns = [benchmark_label]
            benchmark_df.plot(ax=ax)
            ax.fill_between(
                ts_df.index,
                ts_df.iloc[:, 0],
                benchmark_df.iloc[:, 0],
                where=ts_df.iloc[:, 0] > benchmark_df.iloc[:, 0],
                facecolor='#add8e6',
                interpolate=True)
            ax.fill_between(
                ts_df.index,
                ts_df.iloc[:, 0],
                benchmark_df.iloc[:, 0],
                where=ts_df.iloc[:, 0] < benchmark_df.iloc[:, 0],
                facecolor='#ffcccb',
                interpolate=True)

        if no_ax:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax
