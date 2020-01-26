import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from vectorbt.utils.array import Array, ffill
from vectorbt.signals import Signals
from vectorbt.utils.decorators import requires_1dim

class Positions(Array):
    def __new__(cls, input_array, index=None, columns=None):
        obj = Array(input_array, index=index, columns=columns).view(cls)
        if obj.dtype != np.integer:
            raise TypeError("dtype must be integer")
        if not ((obj >= -1) & (obj <= 1)).all():
            raise TypeError("Values must be one of -1, 0, or 1")
        return obj

    @classmethod
    def empty(cls, shape, index=None, columns=None):
        """Create and fill an empty array with 0."""
        return super().empty(shape, 0, index=index, columns=columns)

    @classmethod
    def from_signals(cls, entries, exits):
        """Generate positions from entry and exit signals."""
        if not isinstance(entries, Signals): 
            raise TypeError("Argument entries must be Signals")
        if not isinstance(exits, Signals): 
            raise TypeError("Argument exits must be Signals")
        # Safety check whether index of entries and exits is the same
        if not np.array_equal(entries.index, exits.index):
            raise ValueError("Arguments entries and exits must share the same index")

        entries_ndim = entries.ndim
        exits_ndim = exits.ndim
        # Bring to the same shape by replicating the columns
        entries = entries.align_columns(exits, expand_dims=True)
        exits = exits.align_columns(entries, expand_dims=True)

        both = np.zeros(entries.shape, dtype=int)
        both[entries] = 1
        both[exits] = -1
        both[entries & exits] = 0
        # remove all exit signals before first entry signals
        both[both.cumsum(axis=0) == -1] = 0
        # take first signal from each consecutive series of signals of same type
        both = ffill(both)
        positions = np.zeros_like(both)
        positions[Signals(both == 1).first()] = 1
        positions[Signals(both == -1).first()] = -1

        # Collapse dims back
        if entries_ndim == 1 and exits_ndim == 1:
            positions = positions[:, 0]
        return Positions(positions, index=entries.index, columns=entries.columns)

    @requires_1dim
    def plot(self, ts, label='TimeSeries', ax=None):
        """Plot positions markers on top of ts."""
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots()
            pd.DataFrame(ts.to_pandas(), columns=[label]).plot(ax=ax, color='#1f77b4')
        pos_idxs = np.argwhere(self == 1).transpose()[0]
        neg_idxs = np.argwhere(self == -1).transpose()[0]
        pd.DataFrame(ts[pos_idxs], index=ts.index[pos_idxs], columns=['Buy']).plot(
            marker='^', color='lime', markersize=10, linestyle='None', ax=ax)
        pd.DataFrame(ts[neg_idxs], index=ts.index[neg_idxs], columns=['Sell']).plot(
            marker='v', color='orangered', markersize=10, linestyle='None', ax=ax)
        if no_ax:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax

