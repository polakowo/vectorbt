import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vectorbt.utils.array import *
from vectorbt.utils.plot import *
from vectorbt.timeseries import TimeSeries
from vectorbt.signals import Signals

class Positions(np.ndarray):
    def __new__(cls, input_array, index=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
        if obj.dtype != np.integer:
            raise TypeError("Data type is not integer")
        if not ((obj >= -1) & (obj <= 1)).all():
            raise TypeError("Values are outside of {-1, 0, 1}")
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
    def from_signals(cls, entries, exits):
        """Generate positions from entry and exit signals."""
        if not isinstance(entries, Signals): 
            raise TypeError("Argument entries is not Signals")
        if not isinstance(exits, Signals): 
            raise TypeError("Argument exits is not Signals")
        # Safety check whether index of entries and exits is the same
        if not np.array_equal(entries.index, exits.index):
            raise ValueError("Arguments entries and exits do not share the same index")

        temp = np.zeros(entries.shape, dtype=int)
        temp[entries] = 1
        temp[exits] = -1
        temp[entries & exits] = 0
        # remove all exit signals before first entry signals
        temp[:np.argwhere(entries)[0][0]] = 0
        # take first signal from each consecutive series of signals of same type
        temp = ffill(temp)
        positions = Positions(np.zeros_like(temp), index=entries.index)
        positions[Signals(temp == 1, index=entries.index).first()] = 1
        positions[Signals(temp == -1, index=entries.index).first()] = -1
        return positions

    def non_empty(self):
        """Return nonzero values as pd.Series."""
        non_empty_idxs = np.argwhere(self).transpose()[0]
        return pd.Series(self[non_empty_idxs], index=self.index[non_empty_idxs])

    def describe(self):
        """Describe using pd.Series."""
        return pd.Series(self).value_counts()

    def plot(self, ts, label='TimeSeries'):
        """Plot positions over the ts."""
        ts.plot(label=label, positions=self)

