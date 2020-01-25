import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vectorbt.utils.array import *
from vectorbt.utils.plot import *
from vectorbt.timeseries import TimeSeries

class Signals(np.ndarray):

    def __new__(cls, input_array, index=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        if obj.dtype != np.bool:
            raise TypeError("dtype is not bool")
        # add the new attribute to the created instance
        if index is not None:
            if obj.shape[0] != len(index):
                raise TypeError("Index has different shape")
            obj.index = index
        elif isinstance(input_array, pd.Series):
            obj.index = input_array.index.to_numpy()
        else:
            raise ValueError("Index is not set")
        # Finally, we must return the newly created object:
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self.index = getattr(obj, 'index', None)

    @classmethod
    def empty(cls, shape, index=None):
        """Create an empty array and fill with False."""
        return Signals(np.full(shape, False), index=index)

    @classmethod
    def generate_exits(cls, ts, entries, exit_func):
        """Generate exit signals based on entry signals.

        For each range between two entries in entries, search for an exit signal."""
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")
        if not isinstance(entries, Signals): 
            raise TypeError("Argument entries is not Signals")
        
        # TODO: Vectorize
        exit_idxs = []
        entry_idxs = np.flatnonzero(entries)
        # Every two adjacent signals in the vector become a bin
        bins = list(zip(entry_idxs, np.append(entry_idxs[1:], None)))
        for prev_idx, next_idx in bins:
            # If entry is the last element, ignore
            if prev_idx < ts.shape[0] - 1:
                # Apply exit function on the bin space only (between two entries)
                exit_idx = exit_func(ts, prev_idx=prev_idx, next_idx=next_idx)
                if exit_idx is not None:
                    exit_idxs.append(exit_idx)
        exits = Signals.empty(ts.shape, index=ts.index)
        # Take note that x, y, and z are all relative indices
        exits[exit_idxs] = True
        return exits

    @classmethod
    def generate_entries_and_exits(cls, ts, entry_func, exit_func):
        """Generate entry and exit signals one after another."""
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")

        # TODO: Vectorize
        idxs = [entry_func(ts)]  # entry comes first
        while True:
            prev_idx = idxs[-1]
            if prev_idx < ts.shape[0] - 1:
                if len(idxs) % 2 == 0:  # exit or entry?
                    idx = entry_func(ts, prev_idx=prev_idx)
                else:
                    idx = exit_func(ts, prev_idx=prev_idx)
                if idx is None:
                    break
                idxs.append(idx)
            else:
                break
        entries = Signals.empty(ts.shape, index=ts.index)
        entries[idxs[0::2]] = True
        exits = Signals.empty(ts.shape, index=ts.index)
        exits[idxs[1::2]] = True
        return entries, exits

    @classmethod
    def generate_random_entries(cls, ts, n, seed=None):
        """Generate entries randomly."""
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")

        if seed is not None:
            np.random.seed(seed)
        # Entries cannot be one after another
        idxs = np.random.choice(np.arange(ts.shape[0])[::2], size=n, replace=False)
        entries = Signals.empty(ts.shape, index=ts.index)
        entries[idxs] = True
        return entries

    @classmethod
    def generate_random_exits(cls, ts, entries, seed=None):
        """Generate exits between entries randomly."""
        if not isinstance(ts, TimeSeries): 
            raise TypeError("Argument ts is not TimeSeries")
        if not isinstance(entries, Signals): 
            raise TypeError("Argument entries is not Signals")

        def random_exit_func(ts, prev_idx=None, next_idx=None):
            return np.random.choice(np.arange(ts.shape[0])[prev_idx+1:next_idx])

        return Signals.generate_exits(ts, entries, random_exit_func)

    def rshift(self, n):
        """Shift the elements to the right."""
        return rshift(self, n)

    def lshift(self, n):
        """Shift the elements to the left."""
        return lshift(self, n)

    def first(self):
        """Set True at the first occurrence in each series of consecutive occurrences."""
        return self & ~self.rshift(1)

    def last(self):
        """Set True at the last occurrence in each series of consecutive occurrences."""
        return self & ~self.lshift(1)

    def rank(self):
        """Assign position to each occurrence in each series of consecutive occurrences."""
        idxs = np.flatnonzero(self.first().astype(int))
        ranks = np.zeros(self.shape[0])
        cum = np.array(np.cumsum(self.astype(int)))
        ranks[idxs] = cum[idxs]
        ranks = cum - ffill(ranks) + 1
        ranks[~self] = 0
        return ranks # produces np.ndarray, not Signals!

    def first_nst(self, n):
        """Set True at the occurrence that is n data points after the first occurrence."""
        return Signals(self.rank() == n, index=self.index)

    def from_first_nst(self, n):
        """Set True at the occurrence that at least n data points after the first occurrence."""
        return Signals(self.rank() >= n, index=self.index)

    def non_empty(self):
        """Return True values as pd.Series."""
        non_empty_idxs = np.argwhere(self).transpose()[0]
        return pd.Series(self[non_empty_idxs], index=self.index[non_empty_idxs])

    def describe(self):
        """Describe using pd.Series."""
        return pd.Series(self).value_counts()

    def plot(self):
        """Plot signals as a line."""
        fig, ax = plt.subplots()
        plot_line(ax, self, "Signals")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
