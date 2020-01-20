import numpy as np
import pandas as pd
import random

from vectorbt.array import ffill

##############
# Main class #
##############


class Signals(pd.Series):
    @property
    def _constructor(self):
        return Signals

    @classmethod
    def from_idx(cls, n, idx):
        """Create signals of size n where idx are indices of occurrences (True)."""
        sr = pd.Series(False, index=range(n), dtype=bool)
        sr.iloc[idx] = True
        return Signals(sr)

    @property
    def ints(self):
        """Transform bools into integers."""
        return self.astype(int)

    def rshift(self, n):
        """Shift the series to the right."""
        return self.shift(periods=n).fillna(False)

    def lshift(self, n):
        """Shift the series to the left."""
        return self.shift(periods=-n).fillna(False)

    def first(self):
        """Set True at the first occurrence in each series of consecutive occurrences."""
        return self & ~self.rshift(1)

    def last(self):
        """Set True at the last occurrence in each series of consecutive occurrences."""
        return self & ~self.lshift(1)

    def reverse(self):
        """Reverse the series."""
        sig = self.copy()[::-1]
        sig.index = self.index
        return sig

    def rank(self):
        """Assign position to each occurrence in each series of consecutive occurrences."""
        idx = np.flatnonzero(self.first().ints)
        z = np.zeros(len(self.ints))
        cum = np.cumsum(self.ints.to_numpy())
        z[idx] = cum[idx]
        z = cum - ffill(z) + 1
        z[~self.to_numpy()] = np.nan
        return pd.Series(z, index=self.index)

    def first_nst(self, n):
        """Set True at the occurrence that is n data points after the first occurrence."""
        return Signals(self.rank() == n, dtype=bool)

    def from_first_nst(self, n):
        """Set True at each occurrence after the nst occurrence."""
        return Signals(self.rank() >= n, dtype=bool)


############################
# Ways to generate signals #
############################


def compare(sr1, sr2, compare_func=lambda x, y: x > y):
    """Compare both array using compare_func."""
    return Signals(compare_func(sr1, sr2), dtype=bool)


def nst_change_in_row(sr, n, change_func=lambda x: x.diff(), compare_func=lambda x: x > 0):
    """Set True at each consecutive change according to compare_func."""
    return Signals(compare_func(change_func(sr)).reindex(sr.index).fillna(False)).from_first_nst(n)


def rolling_window(sr, window, window_func=lambda x: np.mean(x), compare_func=lambda x, y: x > y, **kwargs):
    """Compare each element in the series to the rolling window aggregation."""
    return Signals(compare_func(sr, sr.rolling(window, **kwargs).apply(window_func, raw=False)))


def generate_exits(sr, entry_sr, exit_func):
    """Generate exit signals based on entry signals.
    
    For each range between two entries in entry_sr, search for an exit signal."""
    assert(len(sr.index) == len(entry_sr.index))
    exit_idx = []
    entry_idx = np.flatnonzero(entry_sr)
    # Every two adjacent signals in the vector become a bin
    bins = list(zip(entry_idx, np.append(entry_idx[1:], None)))
    for x, z in bins:
        if x < len(entry_sr.index) - 1:
            # If entry is the last element, ignore
            x = x + 1
            # Apply exit function on the bin space only (between two entries)
            y = exit_func(sr.iloc[x:z])
            exit_idx.append(x + y)
    exits = pd.Series(False, index=sr.index)
    # Take note that x, y, and z are all relative indices
    exits.iloc[exit_idx] = True
    return Signals(exits)


def generate_entries_and_exits(sr, entry_func, exit_func):
    """Generate entry and exit signals one after another."""
    idx = [entry_func(sr)] # entry comes first
    while True:
        x = idx[-1]
        if x < len(sr.index) - 1:
            x = x + 1
        else:
            break
        if len(idx) % 2 == 0:  # exit or entry?
            y = entry_func(sr.iloc[x:])
        else:
            y = exit_func(sr.iloc[x:])
        if y is None:
            break
        idx.append(x + y)
    entries = pd.Series(False, index=sr.index)
    entries.iloc[idx[0::2]] = True
    exits = pd.Series(False, index=sr.index)
    exits.iloc[idx[1::2]] = True
    return entries, exits


def generate_random_entries(sr, n, seed=None):
    """Generate entries randomly."""
    # Entries cannot be one after another
    idx = sr[::2].sample(n, random_state=seed).index
    entries = pd.Series(False, index=sr.index)
    entries.loc[idx] = True
    return entries


def generate_random_exits(sr, entry_sr, seed=None):
    """Generate exits between entries randomly."""
    random.seed(seed)
    return generate_exits(sr, entry_sr, lambda x: random.choice(range(len(x.index))))
