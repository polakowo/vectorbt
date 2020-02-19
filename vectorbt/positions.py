from vectorbt.signals import Signals
from vectorbt.timeseries import TimeSeries
from vectorbt.decorators import *
from numba import njit, i1, b1
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ############# Numba functions ############# #


@njit(i1[:, :](b1[:, :], b1[:, :]), cache=True)
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
    @to_2d('input_array')
    @has_dtype('input_array', np.int8)
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        if not ((obj >= -1) & (obj <= 1)).all():
            raise TypeError("Values must be one of -1, 0, or 1")
        return obj

    @classmethod
    @broadcast('entries', 'exits')
    @has_type('entries', Signals)
    @has_type('exits', Signals)
    def from_signals(cls, entries, exits):
        """Generate positions from entry and exit signals."""
        return cls(from_signals_nb(entries, exits))

    @to_2d('self')
    @to_2d('ts')
    @broadcast('self', 'ts')
    @has_type('ts', TimeSeries)
    def plot(self,
             ts,
             column=None,
             plot_ts=True,
             index=None,
             buy_scatter_kwargs={},
             sell_scatter_kwargs={},
             figsize=(800, 300),
             return_fig=False,
             static=True,
             fig=None,
             **kwargs):

        if column is None:
            if self.shape[1] == 1:
                column = 0
            else:
                raise ValueError("For an array with multiple columns, you must pass a column index")
        # Plot TimeSeries
        if plot_ts:
            fig = ts.plot(column, index=index, figsize=figsize, return_fig=True, fig=fig, **kwargs)
        elif fig is None:
            raise ValueError("Plot TimeSeries or specify a FigureWidget object")
        # Plot Positions
        positions = self[:, column]
        ts = ts[:, column]
        buy_idxs = np.where(positions == 1)[0]
        sell_idxs = np.where(positions == -1)[0]
        if index is None:
            index = np.arange(positions.shape[0])
        buy_scatter = go.Scatter(
            x=index[buy_idxs],
            y=ts[buy_idxs],
            mode='markers',
            marker=go.scatter.Marker(
                symbol='triangle-up',
                color='lime',
                size=10
            ),
            name='Buy'
        )
        buy_scatter.update(**buy_scatter_kwargs)
        fig.add_trace(buy_scatter)
        sell_scatter = go.Scatter(
            x=index[sell_idxs],
            y=ts[sell_idxs],
            mode='markers',
            marker=go.scatter.Marker(
                symbol='triangle-down',
                color='orangered',
                size=10
            ),
            name='Sell'
        )
        sell_scatter.update(**sell_scatter_kwargs)
        fig.add_trace(sell_scatter)

        if return_fig:
            return fig
        else:
            if static:
                fig.show(renderer="png", width=figsize[0], height=figsize[1])
            else:
                fig.show()
