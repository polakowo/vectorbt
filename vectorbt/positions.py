from vectorbt.utils.decorators import has_type, to_dim1, broadcast
from vectorbt.timeseries import TimeSeries
from vectorbt.signals import Signals
from vectorbt.utils.array import Array2D, ffill
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class Positions(Array2D):
    def __new__(cls, input_array):
        obj = Array2D(input_array).view(cls)
        if obj.dtype != np.integer:
            raise TypeError("dtype must be integer")
        if not ((obj >= -1) & (obj <= 1)).all():
            raise TypeError("Values must be one of -1, 0, or 1")
        return obj

    @classmethod
    def empty(cls, shape):
        """Create and fill an empty array with 0."""
        return super().empty(shape, 0)

    @classmethod
    @has_type(1, Signals)
    @has_type(2, Signals)
    @broadcast(1, 2)
    def from_signals(cls, entries, exits):
        """Generate positions from entry and exit signals."""
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
        return Positions(positions)

    @to_dim1(0)
    @has_type(1, TimeSeries)
    def plot(self, ts, index=None, ax=None, **kwargs):
        """Plot position markers on top of ts."""
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots()
            # Plot TimeSeries
            ax = ts.plot(index=index, ax=ax, **kwargs)
        # Plot Positions
        pos_idxs = np.argwhere(self == 1).transpose()[0]
        neg_idxs = np.argwhere(self == -1).transpose()[0]
        buy_df = pd.DataFrame(ts[pos_idxs], columns=['Buy'])
        sell_df = pd.DataFrame(ts[neg_idxs], columns=['Sell'])
        if index is not None:
            buy_df.index = pd.Index(index)[pos_idxs]
            sell_df.index = pd.Index(index)[neg_idxs]
        else:
            buy_df.index = pos_idxs
            sell_df.index = neg_idxs
        buy_df.plot(marker='^', color='lime', markersize=10, linestyle='None', ax=ax)
        sell_df.plot(marker='v', color='orangered', markersize=10, linestyle='None', ax=ax)

        if no_ax:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax
