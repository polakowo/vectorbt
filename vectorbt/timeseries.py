from vectorbt.utils.decorators import to_dim1, has_type
from vectorbt.utils.array import Array2D
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


class TimeSeries(Array2D):
    def __new__(cls, input_array):
        obj = Array2D(input_array).view(cls)
        if obj.dtype != np.float:
            raise TypeError("dtype must be float")
        return obj

    @classmethod
    def empty(cls, shape):
        """Create and fill an empty array with 0."""
        return super().empty(shape, 0)

    @to_dim1(0)
    def plot(self, index=None, label=None, benchmark=None, benchmark_label=None, ax=None, **kwargs):
        """Plot time series as a line."""
        no_ax = ax is None
        if no_ax:
            fig, ax = plt.subplots()

        # Plot TimeSeries
        ts_df = pd.DataFrame(self)
        if index is not None:
            ts_df.index = pd.Index(index)
        if label is not None:
            ts_df.columns = [label]
        else:
            ts_df.columns = ['TimeSeries']
        ts_df.plot(ax=ax, **kwargs)

        # Plot benchmark
        if benchmark is not None:
            if isinstance(benchmark, pd.Series):
                benchmark_df = benchmark
                benchmark_df.columns = ['Benchmark']
            elif isinstance(benchmark, (int, float, complex)):
                benchmark_df = pd.DataFrame(np.full(len(ts_df.index), benchmark))
                benchmark_df.columns = [str(benchmark)]
                benchmark_df.index = ts_df.index
            else:
                benchmark_df = pd.DataFrame(benchmark)
                benchmark_df.columns = ['Benchmark']
                benchmark_df.index = ts_df.index
            if benchmark_label is not None:
                benchmark_df.columns = [benchmark_label]
            benchmark_df.plot(ax=ax)
            ax.fill_between(
                ts_df.index,
                ts_df.iloc[:, 0],
                benchmark_df.iloc[:, 0],
                where=ts_df.iloc[:, 0] > benchmark_df.iloc[:, 0],
                facecolor='#add8e6',
                interpolate=True)
            ax.fill_between(
                ts_df.index,
                ts_df.iloc[:, 0],
                benchmark_df.iloc[:, 0],
                where=ts_df.iloc[:, 0] < benchmark_df.iloc[:, 0],
                facecolor='#ffcccb',
                interpolate=True)

        if no_ax:
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        return ax
