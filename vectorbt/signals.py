from vectorbt.timeseries import TimeSeries
from vectorbt.utils.decorators import has_type, to_dim1, to_dim2, broadcast
from vectorbt.utils.array import Array2D, fshift, bshift, ffill, shuffle_along_axis
from numba import njit
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


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


@njit  # 41.8 µs vs 83.9 µs
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


class Signals(Array2D):
    """Signals class extends the Array class by implementing boolean operations."""

    def __new__(cls, input_array):
        obj = Array2D(input_array).view(cls)
        if obj.dtype != np.bool:
            raise TypeError("dtype must be bool")
        return obj

    #################
    # Class methods #
    #################

    @classmethod
    def empty(cls, shape):
        """Create and fill an empty array with False."""
        return super().empty(shape, False)

    @classmethod
    def generate_random(cls, shape, n, seed=None):
        """Generate entry signals randomly."""
        if seed is not None:
            np.random.seed(seed)
        if not isinstance(shape, tuple):
            # Expand x to (x,1)
            new_shape = (shape, 1)
        elif len(shape) == 1:
            # Expand (x,) to (x,1)
            new_shape = (shape[0], 1)
        else:
            new_shape = shape

        idxs = np.tile(np.arange(new_shape[0])[:, None], (1, new_shape[1]))
        idxs = shuffle_along_axis(idxs)[:n]
        entries = np.full(new_shape, False)
        entries[idxs, np.arange(new_shape[1])[None, :]] = True
        return cls(entries)

    @classmethod
    @has_type(1, Array2D)
    @to_dim2(1)
    def generate_random_like(cls, ts, *args, **kwargs):
        """Generate entry signals randomly in form of ts."""
        return cls.generate_random(ts.shape, *args, **kwargs)

    @classmethod
    @has_type(1, Array2D)
    @has_type(2, 0)
    @broadcast(1, 2)
    def generate_exits(cls, ts, entries, exit_func):
        """Generate an exit signal after every entry signal using exit_func."""

        # Do not forget to wrap exit_func with @njit in your code!
        exits = generate_exits(ts, entries, exit_func)
        return cls(exits)

    @classmethod
    @has_type(1, 0)
    @to_dim2(1)
    def generate_random_exits(cls, entries, seed=None):
        """Generate an exit signal after every entry signal randomly."""
        if seed is not None:
            np.random.seed(seed)

        # For each column, we need to randomly pick an index between two entry signals
        cumsum = entries.flatten(order='F').cumsum().reshape(entries.shape, order='F')
        cumsum[entries.cumsum(axis=0) == 0] = 0
        flattened = cumsum.flatten(order='F')

        unique, counts = np.unique(flattened, return_counts=True)
        unique, counts = unique[unique != 0], counts[unique != 0]
        rel_rand_idxs = np.floor(np.random.uniform(size=len(unique)) * (counts - 1)) + 1
        entry_idxs = np.argwhere(entries.flatten(order='F') == 1).transpose()[0]
        valid_idxs = np.argwhere(counts > 1).transpose()[0]
        rand_idxs = rel_rand_idxs[valid_idxs] + entry_idxs[valid_idxs]

        exits = np.full(len(flattened), False)
        exits[rand_idxs.astype(int)] = True
        exits = exits.reshape(entries.shape, order='F')
        return cls(exits)

    @classmethod
    @has_type(1, Array2D)
    @to_dim2(1)
    def generate_entries_and_exits(cls, ts, entry_func, exit_func):
        """Generate entry and exit signals one after another iteratively.

        Use this if your entries depend on previous exit signals, otherwise generate entries first."""

        # Do not forget to wrap entry_func and exit_func with @njit in your code!
        entries, exits = generate_entries_and_exits(ts, entry_func, exit_func)
        return cls(entries), cls(exits)

    ######################
    # Boolean operations #
    ######################

    @to_dim2(0)
    def fshift(self, n):
        """Shift the elements to the right."""
        return fshift(self, n)

    @to_dim2(0)
    def bshift(self, n):
        """Shift the elements to the left."""
        return bshift(self, n)

    @to_dim2(0)
    def first(self):
        """Set True at the first event in each series of consecutive events."""
        return self & ~self.fshift(1)

    @to_dim2(0)
    def last(self):
        """Set True at the last event in each series of consecutive events."""
        return self & ~self.bshift(1)

    @to_dim2(0)
    def rank(self):
        """Assign position to each event in each series of consecutive events."""
        idxs = np.nonzero(self.first().astype(int))
        ranks = np.zeros(self.shape)
        cum = np.array(np.cumsum(self.astype(int), axis=0))
        ranks[idxs] = cum[idxs]
        ranks = cum - ffill(ranks) + 1
        ranks[~self] = 0
        return ranks  # produces np.ndarray, not Signals!

    @to_dim2(0)
    def first_nst(self, n):
        """Set True at the event that is n data points after the first event."""
        return Signals(self.rank() == n)

    @to_dim2(0)
    def from_first_nst(self, n):
        """Set True at the event that at least n data points after the first event."""
        return Signals(self.rank() >= n)

    @to_dim1(0)
    def plot(self, index=None, label=None, ax=None, **kwargs):
        """Plot signals as a line."""
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots()

        # Plot Signals
        df = pd.DataFrame(self.astype(int))
        if index is not None:
            df.index = pd.Index(index)
        if label is not None:
            df.columns = [label]
        else:
            df.columns = ['Signals']
        df.plot(ax=ax, **kwargs)

        if no_ax:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax
