import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vectorbt.utils.array import Array
from vectorbt.utils.decorators import requires_1dim
from vectorbt.utils.plot import plot_line

class TimeSeries(Array):
    def __new__(cls, input_array, index=None, columns=None):
        obj = Array(input_array, index=index, columns=columns).view(cls)
        if obj.dtype != np.float:
            raise TypeError("dtype must be float")
        return obj

    @classmethod
    def empty(cls, shape, index=None, columns=None):
        """Create and fill an empty array with 0."""
        return super().empty(shape, 0, index=index, columns=columns)

    @requires_1dim
    def plot(self, label='TimeSeries', positions=None, **kwargs):
        """Plot time series as a line."""
        fig, ax = plt.subplots()
        plot_line(ax, self, label, **kwargs)
        if positions is not None:
            plot_markers(ax, self, positions == 1, positions == -1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    