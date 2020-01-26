import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from numba import njit

from vectorbt.utils.array import Array, fshift, bshift, ffill, shuffle_along_axis
from vectorbt.utils.decorators import expand_dims, requires_1dim
from vectorbt.timeseries import TimeSeries

@njit
def generate_exits(ts, entries, exit_func):
    # NOTE: Cannot be vectorized since each exit signal depends on the previous entry signal.
    # You will have to write your exit_func to be compatible with numba.

    exits = np.full(ts.shape, False)
    for i in range(ts.shape[1]):
        exit_idxs = []
        entry_idxs = np.flatnonzero(entries[:, i])
        # Every two adjacent signals in the vector become a bin
        bins = np.column_stack((entry_idxs, np.append(entry_idxs[1:], -1)))
        for j in range(bins.shape[0]):
            prev_idx = bins[j][0]
            next_idx = bins[j][1]
            if next_idx == -1:
                next_idx = None
            # If entry is the last element, ignore
            if prev_idx < ts.shape[0] - 1:
                # Apply exit function on the bin space only (between two entries)
                exit_idx = exit_func(ts[:, i], prev_idx, next_idx)
                if exit_idx is not None:
                    exit_idxs.append(int(exit_idx))
        # Take note that x, y, and z are all relative indices
        exits[np.asarray(exit_idxs), i] = True
    return exits

@njit # 38.4 µs vs 214 µs
def random_exit_func(ts, prev_idx, next_idx):
    if next_idx is None:
        return np.random.choice(np.arange(ts.shape[0])[prev_idx+1:])
    if next_idx - prev_idx > 1:
        return np.random.choice(np.arange(ts.shape[0])[prev_idx+1:next_idx])
    return None

@njit # 41.8 µs vs 83.9 µs
def generate_entries_and_exits(ts, entry_func, exit_func):
    # NOTE: Cannot be vectorized since both signals depend on each other.
    # You will have to write your entry_func and exit_func to be compatible with numba.

    entries = np.full(ts.shape, False)
    exits = np.full(ts.shape, False)

    for i in range(ts.shape[1]):
        idxs = [entry_func(ts[:, i], None, None)]  # entry comes first
        while True:
            prev_idx = idxs[-1]
            if prev_idx < ts.shape[0] - 1:
                if len(idxs) % 2 == 0:  # exit or entry?
                    idx = entry_func(ts[:, i], prev_idx, None)
                else:
                    idx = exit_func(ts[:, i], prev_idx, None)
                if idx is None:
                    break
                idxs.append(idx)
            else:
                break
        entries[np.asarray(idxs[0::2]), i] = True
        exits[np.asarray(idxs[1::2]), i] = True
    return entries, exits


class Signals(Array):
    """Signals class extends the Array class by implementing boolean operations."""

    def __new__(cls, input_array, index=None, columns=None):
        obj = Array(input_array, index=index, columns=columns).view(cls)
        if obj.dtype != np.bool:
            raise TypeError("dtype must be bool")
        return obj

    @classmethod
    def empty(cls, shape, index=None, columns=None):
        """Create and fill an empty array with False."""
        return super().empty(shape, False, index=index, columns=columns)

    @classmethod
    def generate_random(cls, shape, n, index=None, columns=None, seed=None):
        """Generate entry signals randomly."""
        if seed is not None:
            np.random.seed(seed)
        if not isinstance(shape, tuple):
            # Expand (x,) to (x,1)
            new_shape = (shape, 1)
        elif len(shape) == 1:
            new_shape = (shape[0], 1)
        else:
            new_shape = shape

        idxs = np.tile(np.arange(new_shape[0])[:, None], (1, new_shape[1]))
        idxs = shuffle_along_axis(idxs)[:n]
        entries = np.full(new_shape, False)
        entries[idxs, np.arange(new_shape[1])[None, :]] = True
        if not isinstance(shape, tuple) or len(shape) == 1:
            # Collapse (x,1) back to (x,)
            entries = entries[:, 0]
        return cls(entries, index=index, columns=columns)

    @classmethod
    def generate_random_like(cls, ts, *args, **kwargs):
        """Generate entry signals randomly in form of ts."""
        return cls.generate_random(ts.shape, *args, index=ts.index, columns=ts.columns, **kwargs)

    @classmethod
    def generate_random_exits(cls, entries, seed=None):
        """Generate an exit signal after every entry signal randomly."""
        if not isinstance(entries, Signals):
            raise TypeError("Argument entries must be Signals")

        return Signals.generate_exits(entries, entries, random_exit_func)

    @classmethod
    def generate_exits(cls, ts, entries, exit_func):
        """Generate an exit signal after every entry signal using exit_func."""
        if not isinstance(ts, TimeSeries) and not isinstance(ts, Signals): # pass entries as ts if you don't need ts
            raise TypeError("Argument ts must be TimeSeries or Signals")
        if not isinstance(entries, Signals):
            raise TypeError("Argument entries must be Signals")
        if not np.array_equal(ts.index, entries.index):
            raise ValueError("Arguments ts and entries must share the same index")

        ts_ndim, entries_ndim = ts.ndim, entries.ndim
        ts = ts.align_columns(entries, expand_dims=True)
        entries = entries.align_columns(ts, expand_dims=True)
        # Do not forget to wrap exit_func with @njit in your code!
        exits = generate_exits(ts, entries, exit_func)
        # Collapse dims if two 1d arrays provided
        if ts_ndim == 1 and entries_ndim == 1:
            exits = exits[:, 0]
        return cls(exits, index=entries.index, columns=entries.columns)

    @classmethod
    def generate_entries_and_exits(cls, ts, entry_func, exit_func):
        """Generate entry and exit signals one after another iteratively.
        
        Use this if your entries depend on previous exit signals, otherwise generate entries first."""
        if not isinstance(ts, TimeSeries):
            raise TypeError("Argument ts must be TimeSeries")

        # Expand dims if two 1d arrays provided
        ts_ndim = ts.ndim
        if ts_ndim == 1:
            ts = ts[:, None]
        entries, exits = generate_entries_and_exits(ts, entry_func, exit_func)
        # Collapse dims if two 1d arrays provided
        if ts_ndim == 1:
            entries = entries[:, 0]
            exits = exits[:, 0]
        return cls(entries, index=ts.index, columns=ts.columns), cls(exits, index=ts.index, columns=ts.columns)

    @expand_dims
    def fshift(self, n):
        """Shift the elements to the right."""
        return fshift(self, n)

    @expand_dims
    def bshift(self, n):
        """Shift the elements to the left."""
        return bshift(self, n)

    @expand_dims
    def first(self):
        """Set True at the first occurrence in each series of consecutive occurrences."""
        return self & ~self.fshift(1)

    @expand_dims
    def last(self):
        """Set True at the last occurrence in each series of consecutive occurrences."""
        return self & ~self.bshift(1)

    @expand_dims
    def rank(self):
        """Assign position to each occurrence in each series of consecutive occurrences."""
        idxs = np.nonzero(self.first().astype(int))
        ranks = np.zeros(self.shape)
        cum = np.array(np.cumsum(self.astype(int), axis=0))
        ranks[idxs] = cum[idxs]
        ranks = cum - ffill(ranks) + 1
        ranks[~self] = 0
        return ranks  # produces np.ndarray, not Signals!

    @expand_dims
    def first_nst(self, n):
        """Set True at the occurrence that is n data points after the first occurrence."""
        return Signals(self.rank() == n, index=self.index, columns=self.columns)

    @expand_dims
    def from_first_nst(self, n):
        """Set True at the occurrence that at least n data points after the first occurrence."""
        return Signals(self.rank() >= n, index=self.index, columns=self.columns)

    @requires_1dim
    def non_empty(self):
        """Return True's using pandas."""
        non_empty_idxs = np.argwhere(self).transpose()[0]
        return pd.Series(self[non_empty_idxs], index=self.index[non_empty_idxs])

    @requires_1dim
    def plot(self, label='Signals', ax=None):
        """Plot signals as a line."""
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots()
        pd.DataFrame(self.astype(int).to_pandas(), columns=[label]).plot(ax=ax, color='#1f77b4')
        if no_ax:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax
