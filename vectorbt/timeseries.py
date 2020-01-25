import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from vectorbt.utils.plot import *

class TimeSeries(np.ndarray):
    def __new__(cls, input_array, index=None):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type
        obj = np.asarray(input_array).view(cls)
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

    def describe(self):
        """Describe using pd.Series."""
        return pd.Series(self).describe()

    def plot(self, label='TimeSeries', benchmark=None, benchmark_label='Benchmark', positions=None):
        """Plot time series as a line."""
        fig, ax = plt.subplots()
        plot_line(ax, self, label, b=benchmark, b_label=benchmark_label)
        if positions is not None:
            plot_markers(ax, self, positions == 1, positions == -1)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()
    