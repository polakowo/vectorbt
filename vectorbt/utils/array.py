import numpy as np
import pandas as pd

from vectorbt.utils.decorators import ndim2

# All functions below require input to be a two-dimensional NumPy array,
# where first axis is index and second axis are columns
# They all move along the first axis (axis=0)


@ndim2
def fshift(a, n=1, fill=0):
    """Shift to the right."""
    rolled = np.roll(a, n, axis=0)
    rolled[:n, :] = fill
    return rolled

@ndim2
def bshift(a, n=1, fill=0):
    """Shift to the left."""
    rolled = np.roll(a, -n, axis=0)
    rolled[-n:, :] = fill
    return rolled

@ndim2
def ffill(a, fill=0):
    """Fill zeros with the last non-zero value."""
    prev_idxs = np.tile(np.arange(a.shape[0])[:, None], (1, a.shape[1]))
    prev_idxs[a == 0] = fill
    prev_idxs = np.maximum.accumulate(prev_idxs, axis=0)
    return np.take_along_axis(a, prev_idxs, 0)

@ndim2
def pct_change(a):
    """pd.pct_change in NumPy."""
    return np.insert(np.diff(a, axis=0) / a[:-1, :], 0, np.zeros((1, a.shape[1])), axis=0)

@ndim2
def shuffle_along_axis(a):
    """Shuffle multidimensional array."""
    idx = np.random.rand(*a.shape).argsort(axis=0)
    return np.take_along_axis(a, idx, axis=0)

@ndim2
def rolling_window(a, window=None):
    """Rolling window over the array.

    Produce the pandas result when min_periods=1."""
    if window is None:  # expanding
        window = a.shape[0]
    a = np.insert(a, 0, np.full((window-1, a.shape[1]), np.nan), axis=0)
    shape = (a.shape[0] - window + 1, window) + a.shape[1:]
    strides = (a.strides[0],) + a.strides
    strided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return strided  # raw output

@ndim2
def rolling_mean(a, **kwargs):
    return np.nanmean(rolling_window(a, **kwargs), axis=0)

@ndim2
def rolling_std(a, **kwargs):
    return np.nanstd(rolling_window(a, **kwargs), axis=0)

@ndim2
def rolling_max(a, **kwargs):
    return np.nanmax(rolling_window(a, **kwargs), axis=0)

@ndim2
def ewma(a, window=None):
    """Exponential weighted moving average.
    Produce the pandas result when min_periods=1 and adjust=False.

    https://stackoverflow.com/a/42926270
    """
    if window is None:  # expanding
        window = a.shape[0]
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = a.shape[0]
    pows = alpha_rev ** (np.arange(n + 1))
    pows = pows[:, None]
    scale_arr = 1 / pows[:-1, :]
    offset = a[0, :] * pows[1:, :]
    pw0 = alpha * alpha_rev ** (n-1)
    mult = a * pw0 * scale_arr
    cumsums = mult.cumsum(axis=0)
    out = offset + cumsums * scale_arr[::-1]
    return out


class Array2D(np.ndarray):
    """Base class inherited by TimeSeries, Signals and Positions.
    Array class is similar to pd.DataFrame class, but:
        - Its implemented in pure NumPy for fast processing (orders of magnitude over pandas),
        - Its class methods are optimized for efficient matrix operations.
    """

    def __new__(cls, input_array):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        if obj.ndim == 1:
            obj = obj[:, None] # expand
        if obj.ndim != 2:
            raise ValueError("This method requires a two-dimensional array")

        # Be aware: index and columns do not change when slicing, subscribing or modifying the array!
        # Since arrays are meant to be the same shape throught processing.
        # If you want to change these variables as well, do it explicitly in your code.

        # Finally, we must return the newly created object:
        return obj

    @classmethod
    def empty(cls, shape, fill):
        """Create and fill an empty array."""
        return cls(np.full(shape, fill))

    @ndim2
    def broadcast_columns(self, other):
        """If the array has less columns than the other one, replicate them."""
        if not isinstance(other, Array2D):
            raise TypeError("Argument other must be a subclass of Array2D")
        if self.shape[0] != other.shape[0]:
            raise ValueError("Argument other must have the same index")
        if self.shape == other.shape:
            return self

        # Expand columns horizontally
        if self.shape[1] == 1:
            return self.__class__(np.tile(self, (1, other.shape[1])))
        elif self.shape[1] > 1 and other.shape[1] > 1:
            raise ValueError("Multiple columns on both sides cannot be broadcasted")
