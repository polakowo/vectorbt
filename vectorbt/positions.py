from vectorbt.signals import Signals
from vectorbt.timeseries import TimeSeries
from vectorbt.utils.decorators import *
from numba import njit, i1, b1
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# ############# Numba functions ############# #


@njit(i1[:, :](b1[:, :], b1[:, :]))
def from_signals_nb(entries, exits):
    positions = np.zeros(entries.shape, dtype=i1)
    for j in range(entries.shape[1]):
        prev_val = 0
        for i in range(entries.shape[0]):
            # Place buy and sell orders one after another
            if entries[i, j] and not exits[i, j]:
                if prev_val == 0 or prev_val == -1:
                    positions[i, j] = 1
                prev_val = 1
            elif exits[i, j] and not entries[i, j]:
                # Sell if previous signal was entry
                if prev_val == 1:
                    positions[i, j] = -1
                prev_val = -1
    return positions

# ############# Main class ############# #


class Positions(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if obj.ndim == 1:
            obj = obj[:, None]  # expand
        if obj.ndim != 2:
            raise ValueError("Argument input_array must be a two-dimensional array")
        if obj.dtype != np.int8:
            raise TypeError("dtype must be np.int8")
        if not ((obj >= -1) & (obj <= 1)).all():
            raise TypeError("Values must be one of -1, 0, or 1")
        return obj

    @classmethod
    @has_type('entries', Signals)
    @has_type('exits', Signals)
    @broadcast_both('entries', 'exits')
    def from_signals(cls, entries, exits):
        """Generate positions from entry and exit signals."""
        return cls(from_signals_nb(entries, exits))

    @to_dim1('self')
    @to_dim1('ts')
    @has_type('ts', TimeSeries)
    def plot(self, ts=None, index=None, ax=None, **kwargs):
        """Plot position markers on top of ts."""
        pos_idxs = np.argwhere(self == 1).transpose()[0]
        neg_idxs = np.argwhere(self == -1).transpose()[0]

        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots()

        if ts is not None:
            # Plot TimeSeries
            ax = ts.plot(index=index, ax=ax, **kwargs)
            buy_vals = ts[pos_idxs]
            sell_vals = ts[neg_idxs]
        else:
            buy_vals = np.full(len(pos_idxs), 1)
            sell_vals = np.full(len(neg_idxs), -1)

        # Plot Positions
        buy_df = pd.DataFrame(buy_vals, columns=['Buy'])
        sell_df = pd.DataFrame(sell_vals, columns=['Sell'])
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
