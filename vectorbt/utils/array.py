import numpy as np
import pandas as pd

# All functions below require input to be a two-dimensional NumPy array,
# where first axis is index and second axis are columns
# They all move along the first axis (axis=0)


def fshift(a, n=1, fill=0):
    """Shift to the right."""
    rolled = np.roll(a, n, axis=0)
    rolled[:n, :] = fill
    return rolled


def bshift(a, n=1, fill=0):
    """Shift to the left."""
    rolled = np.roll(a, -n, axis=0)
    rolled[-n:, :] = fill
    return rolled


def ffill(a, fill=0):
    """Fill zeros with the last non-zero value."""
    prev_idxs = np.tile(np.arange(a.shape[0])[:, None], (1, a.shape[1]))
    prev_idxs[a == 0] = fill
    prev_idxs = np.maximum.accumulate(prev_idxs, axis=0)
    return np.take_along_axis(a, prev_idxs, 0)


def pct_change(a):
    """pd.pct_change in NumPy."""
    return np.insert(np.diff(a, axis=0) / a[:-1, :], 0, np.zeros((1, a.shape[1])), axis=0)


def shuffle_along_axis(a):
    """Shuffle multidimensional array against an axis."""
    idx = np.random.rand(*a.shape).argsort(axis=0)
    return np.take_along_axis(a, idx, axis=0)


def rolling(a, window=None):
    """Rolling window over the array.
    Produce the pandas result when min_periods=1."""
    if window is None:  # expanding
        window = a.shape[0]
    a = np.insert(a.astype(float), 0, np.full(window-1, np.nan))
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    strided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return strided  # raw output


def rolling_mean(a, window=None):
    return np.nanmean(rolling(a, window=window), axis=1)


def rolling_std(a, window=None):
    return np.nanstd(rolling(a, window=window), axis=1)


def rolling_max(a, window=None):
    return np.nanmax(rolling(a, window=window), axis=1)


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
    scale_arr = 1 / pows[:-1]
    offset = a[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n-1)
    mult = a * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out


class Array(np.ndarray):
    """Base class inherited by TimeSeries, Signals and Positions.
    Array class is similar to pd.DataFrame class, but:
        - Its implemented in pure NumPy for fast processing (orders of magnitude over pandas),
        - Its class methods are optimized for efficient matrix operations.

    Array object can be either a 1D or 2D array and has the same indexing logic 
    as pd.Series and pd.DataFrame respectively.
    """

    def __new__(cls, input_array, index=None, columns=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)

        # Check shape
        if obj.ndim > 2:
            raise ValueError("Number of dimensions must be 1 or 2")

        # Set index
        if index is not None:
            if obj.shape[0] != len(index):
                raise ValueError("Array and index do not match")
            obj.index = np.asarray(index)
        elif isinstance(input_array, pd.Series) or isinstance(input_array, pd.DataFrame):
            obj.index = input_array.index.to_numpy()
        else:
            obj.index = np.arange(obj.shape[0])

        # Set columns
        if obj.ndim == 1:
            obj.columns = None
        else:
            if columns is not None:
                if obj.shape[1] != len(columns):
                    raise ValueError("Array and columns do not match")
                obj.columns = np.asarray(columns)
            elif isinstance(input_array, pd.DataFrame):
                obj.columns = input_array.columns.to_numpy()
            else:
                obj.columns = np.arange(obj.shape[1])

        # Be aware: index and columns do not change when slicing, subscribing or modifying the array!
        # Since arrays are meant to be the same shape throught processing.
        # If you want to change these variables as well, do it explicitly in your code.

        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return
        self.index = getattr(obj, 'index', None)
        self.columns = getattr(obj, 'columns', None)

    @classmethod
    def empty(cls, shape, fill, index=None, columns=None):
        """Create and fill an empty array."""
        return cls(np.full(shape, fill), index=index, columns=columns)

    def align_columns(self, other, expand_dims=False):
        """If the array has less columns, replicate them horizontally."""
        if not np.array_equal(self.index, other.index):
            raise ValueError("Arguments self and other must share the same index")

        if not expand_dims and self.ndim == 1 and other.ndim == 1:
            return self
        index = self.index
        columns = self.columns

        # Expand columns horizontally for both shapes to match
        if self.ndim == 1:
            self = self[:, None]
        if other.ndim == 1:
            other = other[:, None]
        if self.shape != other.shape:
            if self.shape[1] > 1 and other.shape[1] > 1:
                raise ValueError("Multiple columns on both sides cannot be aligned")
            if self.shape[1] == 1:
                # One time series but multiple signals -> broadcast time series
                self = np.tile(self, (1, other.shape[1]))
                columns = other.columns

        return self.__class__(self, index=index, columns=columns)

    def select_column(self, column_name):
        """Return the 1D array corresponding to a column."""
        if self.ndim == 1:
            raise ValueError("1D arrays do not have columns")
        if column_name not in self.columns:
            raise ValueError(f"Column '{column_name}' not found")
        column_idx = np.argwhere(self.columns == column_name)[0][0]
        return self.__class__(self[:, column_idx], index=self.index)

    def to_pandas(self):
        """Transform to pd.DataFrame."""
        if self.ndim == 1:
            return pd.Series(self, index=self.index)
        else:
            return pd.DataFrame(self, index=self.index, columns=self.columns)

    def __repr__(self):
        return str(self.to_pandas())

    __str__ = __repr__
