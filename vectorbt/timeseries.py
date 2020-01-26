import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from vectorbt.utils.array import Array
from vectorbt.utils.decorators import requires_1dim

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
    def plot(self, label='TimeSeries', benchmark=None, benchmark_label='Benchmark', positions=None, ax=None):
        """Plot time series as a line."""
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots()
        # Plot a
        ts = self.to_pandas()
        pd.DataFrame(ts, columns=[label]).plot(ax=ax, color='#1f77b4')
        # Plot b
        if benchmark is not None:
            if isinstance(benchmark, (int, float, complex)):
                benchmark = pd.Series(benchmark, index=ts.index)
            pd.DataFrame(benchmark, columns=[benchmark_label]).plot(ax=ax, color='#1f77b4', linestyle='--')
            ax.fill_between(ts.index, ts, benchmark, where=ts>benchmark, facecolor='#add8e6', interpolate=True)
            ax.fill_between(ts.index, ts, benchmark, where=ts<benchmark, facecolor='#ffcccb', interpolate=True)
        if positions is not None:
            ax = positions.plot(self, ax=ax)
        if no_ax:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax
    