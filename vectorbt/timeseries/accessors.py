"""Custom pandas accessors.

!!! note
    Input arrays can be of any data type, but most output arrays are `numpy.float64`.
    
```py
import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime

index = pd.Index([
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5)
])
columns = ['a', 'b', 'c']
df = pd.DataFrame([
    [1, 5, 1],
    [2, 4, 2],
    [3, 3, 3],
    [4, 2, 2],
    [5, 1, 1]
], index=index, columns=columns)
```"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools
from numba.typed import Dict
from datetime import timedelta

from vectorbt.accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks, index_fns, reshape_fns
from vectorbt.utils.decorators import add_nb_methods
from vectorbt.utils.accessors import Base_DFAccessor, Base_SRAccessor
from vectorbt.timeseries import nb
from vectorbt.widgets.common import DefaultFigureWidget

try:
    # Adapted from https://github.com/quantopian/empyrical/blob/master/empyrical/utils.py
    import bottleneck as bn

    nanmean = bn.nanmean
    nanstd = bn.nanstd
    nansum = bn.nansum
    nanmax = bn.nanmax
    nanmin = bn.nanmin
    nanargmax = bn.nanargmax
    nanargmin = bn.nanargmin
except ImportError:
    # slower numpy
    nanmean = np.nanmean
    nanstd = np.nanstd
    nansum = np.nansum
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin


def to_time_units(obj, time_delta):
    """Multiply each element with `time_delta` to get result in time units."""
    total_seconds = pd.Timedelta(time_delta).total_seconds()
    def to_td(x): return timedelta(seconds=x * total_seconds) if ~np.isnan(x) else np.nan
    to_td = np.vectorize(to_td, otypes=[np.object])
    return obj.vbt.wrap_array(to_td(obj.vbt.to_array()))


@add_nb_methods(
    nb.fillna_nb,
    nb.fshift_nb,
    nb.diff_nb,
    nb.pct_change_nb,
    nb.ffill_nb,
    nb.cumsum_nb,
    nb.cumprod_nb,
    nb.rolling_min_nb,
    nb.rolling_max_nb,
    nb.rolling_mean_nb,
    nb.rolling_std_nb,
    nb.ewm_mean_nb,
    nb.ewm_std_nb,
    nb.expanding_min_nb,
    nb.expanding_max_nb,
    nb.expanding_mean_nb,
    nb.expanding_std_nb,
    module_name='vectorbt.timeseries.nb')
class TimeSeries_Accessor():
    """Accessor with methods for both Series and DataFrames.

    Accessible through `pandas.Series.vbt.timeseries` and `pandas.DataFrame.vbt.timeseries`."""

    def rolling_apply(self, window, apply_func_nb, *args, on_matrix=False):
        """See `vectorbt.timeseries.nb.rolling_apply_nb` and 
        `vectorbt.timeseries.nb.rolling_apply_matrix_nb` for `on_matrix=True`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.rolling_apply(3, mean_nb))
                          a    b         c
            2018-01-01  1.0  5.0  1.000000
            2018-01-02  1.5  4.5  1.500000
            2018-01-03  2.0  4.0  2.000000
            2018-01-04  3.0  3.0  2.333333
            2018-01-05  4.0  2.0  2.000000

            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.rolling_apply(3, 
            ...     mean_matrix_nb, on_matrix=True))
                               a         b         c
            2018-01-01  2.333333  2.333333  2.333333
            2018-01-02  2.500000  2.500000  2.500000
            2018-01-03  2.666667  2.666667  2.666667
            2018-01-04  2.777778  2.777778  2.777778
            2018-01-05  2.666667  2.666667  2.666667
            ```"""
        checks.assert_numba_func(apply_func_nb)

        if on_matrix:
            result = nb.rolling_apply_matrix_nb(self.to_2d_array(), window, apply_func_nb, *args)
        else:
            result = nb.rolling_apply_nb(self.to_2d_array(), window, apply_func_nb, *args)
        return self.wrap_array(result)

    def expanding_apply(self, apply_func_nb, *args, on_matrix=False):
        """See `vectorbt.timeseries.nb.expanding_apply_nb` and 
        `vectorbt.timeseries.nb.expanding_apply_matrix_nb` for `on_matrix=True`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.expanding_apply(mean_nb))
                          a    b    c
            2018-01-01  1.0  5.0  1.0
            2018-01-02  1.5  4.5  1.5
            2018-01-03  2.0  4.0  2.0
            2018-01-04  2.5  3.5  2.0
            2018-01-05  3.0  3.0  1.8

            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.expanding_apply( 
            ...     mean_matrix_nb, on_matrix=True))
                               a         b         c
            2018-01-01  2.333333  2.333333  2.333333
            2018-01-02  2.500000  2.500000  2.500000
            2018-01-03  2.666667  2.666667  2.666667
            2018-01-04  2.666667  2.666667  2.666667
            2018-01-05  2.600000  2.600000  2.600000
            ```"""
        checks.assert_numba_func(apply_func_nb)

        if on_matrix:
            result = nb.expanding_apply_matrix_nb(self.to_2d_array(), apply_func_nb, *args)
        else:
            result = nb.expanding_apply_nb(self.to_2d_array(), apply_func_nb, *args)
        return self.wrap_array(result)

    def groupby_apply(self, by, apply_func_nb, *args, on_matrix=False, **kwargs):
        """See `vectorbt.timeseries.nb.groupby_apply_nb` and 
        `vectorbt.timeseries.nb.groupby_apply_matrix_nb` for `on_matrix=True`.

        For `by`, see `pandas.DataFrame.groupby`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.groupby_apply([1, 1, 2, 2, 3], 
            ...     mean_nb))
                 a    b    c
            1  1.5  4.5  1.5
            2  3.5  2.5  2.5
            3  5.0  1.0  1.0

            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.groupby_apply([1, 1, 2, 2, 3], 
            ...     mean_matrix_nb, on_matrix=True))
                      a         b         c
            1  2.500000  2.500000  2.500000
            2  2.833333  2.833333  2.833333
            3  2.333333  2.333333  2.333333
            ```"""
        checks.assert_numba_func(apply_func_nb)

        regrouped = self._obj.groupby(by, axis=0, **kwargs)
        groups = Dict()
        for i, (k, v) in enumerate(regrouped.indices.items()):
            groups[i] = np.asarray(v)
        if on_matrix:
            result = nb.groupby_apply_matrix_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        else:
            result = nb.groupby_apply_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        return self.wrap_array(result, index=list(regrouped.indices.keys()))

    def resample_apply(self, freq, apply_func_nb, *args, on_matrix=False, **kwargs):
        """See `vectorbt.timeseries.nb.groupby_apply_nb` and 
        `vectorbt.timeseries.nb.groupby_apply_matrix_nb` for `on_matrix=True`.

        For `freq`, see `pandas.DataFrame.resample`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.resample_apply('2d', mean_nb))
                          a    b    c
            2018-01-01  1.5  4.5  1.5
            2018-01-03  3.5  2.5  2.5
            2018-01-05  5.0  1.0  1.0

            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.resample_apply('2d', 
            ...     mean_matrix_nb, on_matrix=True))
                               a         b         c
            2018-01-01  2.500000  2.500000  2.500000
            2018-01-03  2.833333  2.833333  2.833333
            2018-01-05  2.333333  2.333333  2.333333
            ```"""
        checks.assert_numba_func(apply_func_nb)

        resampled = self._obj.resample(freq, axis=0, **kwargs)
        groups = Dict()
        for i, (k, v) in enumerate(resampled.indices.items()):
            groups[i] = np.asarray(v)
        if on_matrix:
            result = nb.groupby_apply_matrix_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        else:
            result = nb.groupby_apply_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        result_obj = self.wrap_array(result, index=list(resampled.indices.keys()))
        resampled_arr = np.full((resampled.ngroups, self.to_2d_array().shape[1]), np.nan)
        resampled_obj = self.wrap_array(resampled_arr, index=pd.Index(list(resampled.groups.keys()), freq=freq))
        resampled_obj.loc[result_obj.index] = result_obj.values
        return resampled_obj

    def rolling_window(self, window, n=None):
        """Split time series into `n` time ranges each `window` long.

        The result will be a new DataFrame with index of length `window` and columns of length
        `len(columns) * n`. If `n` is `None`, will return the maximum number of time ranges.

        Example:
            ```python-repl
            >>> print(df.vbt.timeseries.rolling_window(2, n=2))
                                a                     b                     c           
            start_date 2018-01-01 2018-01-04 2018-01-01 2018-01-04 2018-01-01 2018-01-04
            0                 1.0        4.0        5.0        2.0        1.0        2.0
            1                 2.0        5.0        4.0        1.0        2.0        1.0 
            ```"""
        cube = nb.rolling_window_nb(self.to_2d_array(), window)
        if n is not None:
            idxs = np.round(np.linspace(0, cube.shape[2]-1, n)).astype(int)
            cube = cube[:, :, idxs]
        else:
            idxs = np.arange(cube.shape[2])
        matrix = np.hstack(cube)
        range_columns = pd.Index(self.index[idxs], name='start_date')
        new_columns = index_fns.combine_indexes(self.columns, range_columns)
        return pd.DataFrame(matrix, columns=new_columns)

    def applymap(self, apply_func_nb, *args):
        """See `vectorbt.timeseries.nb.applymap_nb`.

        Example:
            ```python-repl
            >>> multiply_nb = njit(lambda col, i, a: a ** 2)
            >>> print(df.vbt.timeseries.applymap(multiply_nb))
                           a     b    c
            2018-01-01   1.0  25.0  1.0
            2018-01-02   4.0  16.0  4.0
            2018-01-03   9.0   9.0  9.0
            2018-01-04  16.0   4.0  4.0
            2018-01-05  25.0   1.0  1.0
            ```"""
        checks.assert_numba_func(apply_func_nb)

        result = nb.applymap_nb(self.to_2d_array(), apply_func_nb, *args)
        return self.wrap_array(result)

    def filter(self, filter_func_nb, *args):
        """See `vectorbt.timeseries.nb.filter_nb`.

        Example:
            ```python-repl
            >>> greater_nb = njit(lambda col, i, a: a > 2)
            >>> print(df.vbt.timeseries.filter(greater_nb))
                          a    b    c
            2018-01-01  NaN  5.0  NaN
            2018-01-02  NaN  4.0  NaN
            2018-01-03  3.0  3.0  3.0
            2018-01-04  4.0  NaN  NaN
            2018-01-05  5.0  NaN  NaN
            ```"""
        checks.assert_numba_func(filter_func_nb)

        result = nb.filter_nb(self.to_2d_array(), filter_func_nb, *args)
        return self.wrap_array(result)

    @property
    def timedelta(self):
        """Return time delta of the index frequency."""
        checks.assert_type(self.index, (pd.DatetimeIndex, pd.PeriodIndex))

        if self.index.freq is not None:
            return pd.to_timedelta(pd.tseries.frequencies.to_offset(self.index.freq))
        elif self.index.inferred_freq is not None:
            return pd.to_timedelta(pd.tseries.frequencies.to_offset(self.index.inferred_freq))
        return (self.index[1:] - self.index[:-1]).min()

    def wrap_reduced_array(self, a, index=None, time_units=False):
        """Wrap result of reduction.

        If `time_units` is set, calls `vectorbt.timeseries.common.to_time_units`."""
        if a.ndim == 1:
            # Each column reduced to a single value
            a_obj = pd.Series(a, index=self.columns)
            if time_units:
                if isinstance(time_units, bool):
                    time_units = self.timedelta
                a_obj = to_time_units(a_obj, time_units)
            if self.is_frame():
                return a_obj
            return a_obj.iloc[0]
        else:
            # Each column reduced to an array
            if index is None:
                index = pd.Index(range(a.shape[0]))
            a_obj = self.wrap_array(a, index=index)
            if time_units:
                if isinstance(time_units, bool):
                    time_units = self.timedelta
                a_obj = to_time_units(a_obj, time_units)
            return a_obj

    def apply_and_reduce(self, apply_func_nb, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.timeseries.nb.apply_and_reduce_nb`.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`.

        Example:
            ```python-repl
            >>> greater_nb = njit(lambda col, a: a[a > 2])
            >>> mean_nb = njit(lambda col, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.apply_and_reduce(greater_nb, mean_nb))
            a    4.0
            b    4.0
            c    3.0
            dtype: float64
            ```"""
        checks.assert_numba_func(apply_func_nb)
        checks.assert_numba_func(reduce_func_nb)

        result = nb.apply_and_reduce_nb(self.to_2d_array(), apply_func_nb, reduce_func_nb, *args)
        return self.wrap_reduced_array(result, **kwargs)

    def reduce(self, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.timeseries.nb.reduce_nb`.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, a: np.nanmean(a))
            >>> print(df.vbt.timeseries.reduce(mean_nb))
            a    3.0
            b    3.0
            c    1.8
            dtype: float64
            ```"""
        checks.assert_numba_func(reduce_func_nb)

        result = nb.reduce_nb(self.to_2d_array(), reduce_func_nb, *args)
        return self.wrap_reduced_array(result, **kwargs)

    def reduce_to_array(self, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.timeseries.nb.reduce_to_array_nb`.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`.

        Example:
            ```python-repl
            >>> min_max_nb = njit(lambda col, a: np.array([np.nanmin(a), np.nanmax(a)]))
            >>> print(df.vbt.timeseries.reduce_to_array(min_max_nb, index=['min', 'max']))
                   a    b    c
            min  1.0  1.0  1.0
            max  5.0  5.0  3.0
            ```"""
        checks.assert_numba_func(reduce_func_nb)

        result = nb.reduce_to_array_nb(self.to_2d_array(), reduce_func_nb, *args)
        return self.wrap_reduced_array(result, **kwargs)

    def min(self, **kwargs):
        """Return min of non-NaN elements.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`."""
        result = nanmin(self.to_2d_array(), axis=0)
        return self.wrap_reduced_array(result, **kwargs)

    def max(self, **kwargs):
        """Return max of non-NaN elements.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`."""
        result = nanmax(self.to_2d_array(), axis=0)
        return self.wrap_reduced_array(result, **kwargs)

    def mean(self, **kwargs):
        """Return mean of non-NaN elements.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`."""
        result = nanmean(self.to_2d_array(), axis=0)
        return self.wrap_reduced_array(result, **kwargs)

    def std(self, ddof=1, **kwargs):
        """Return standard deviation of non-NaN elements.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`."""
        result = nanstd(self.to_2d_array(), ddof=ddof, axis=0)
        return self.wrap_reduced_array(result, **kwargs)

    def count(self, **kwargs):
        """Return count of non-NaN elements.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`."""
        result = np.sum(~np.isnan(self.to_2d_array()), axis=0)
        return self.wrap_reduced_array(result, **kwargs)

    def sum(self, **kwargs):
        """Return sum of non-NaN elements.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`."""
        result = nansum(self.to_2d_array(), axis=0)
        return self.wrap_reduced_array(result, **kwargs)

    def argmin(self, **kwargs):
        """Return index of min of non-NaN elements.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`."""
        result = self.index[nanargmin(self.to_2d_array(), axis=0)]
        return self.wrap_reduced_array(result, **kwargs)

    def argmax(self, **kwargs):
        """Return index of max of non-NaN elements.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`."""
        result = self.index[nanargmax(self.to_2d_array(), axis=0)]
        return self.wrap_reduced_array(result, **kwargs)

    def describe(self, percentiles=[0.25, 0.5, 0.75], **kwargs):
        """See `vectorbt.timeseries.nb.describe_reduce_func_nb`.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced_array`.

        For `percentiles`, see `pandas.DataFrame.describe`.

        Example:
            ```python-repl
            >>> print(df.vbt.timeseries.describe())
                           a         b        c
            count   5.000000  5.000000  5.00000
            mean    3.000000  3.000000  1.80000
            std     1.581139  1.581139  0.83666
            min     1.000000  1.000000  1.00000
            25.00%  2.000000  2.000000  1.00000
            50.00%  3.000000  3.000000  2.00000
            75.00%  4.000000  4.000000  2.00000
            max     5.000000  5.000000  3.00000
            ```"""
        if percentiles is not None:
            percentiles = reshape_fns.to_1d(percentiles)
        else:
            percentiles = np.empty(0)
        index = pd.Index(['count', 'mean', 'std', 'min', *map(lambda x: '%.2f%%' % (x * 100), percentiles), 'max'])
        return self.reduce_to_array(nb.describe_reduce_func_nb, percentiles, index=index, **kwargs)


@register_series_accessor('timeseries')
class TimeSeries_SRAccessor(TimeSeries_Accessor, Base_SRAccessor):
    """Accessor with methods for Series only.

    Accessible through `pandas.Series.vbt.timeseries`."""

    def plot(self, name=None, trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot Series as a line.

        Args:
            name (str): Name of the time series.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            df['a'].vbt.timeseries.plot()
            ```

            ![](/vectorbt/docs/img/timeseries_sr_plot.png)"""
        if fig is None:
            fig = DefaultFigureWidget()
            fig.update_layout(**layout_kwargs)
        if name is None:
            name = self._obj.name

        scatter = go.Scatter(
            x=self.index,
            y=self.to_array(),
            mode='lines',
            name=str(name),
            showlegend=name is not None
        )
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter)

        return fig

    def plot_against(self, other, name=None, other_name=None, above_trace_kwargs={}, below_trace_kwargs={},
                     other_trace_kwargs={}, equal_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot Series against `other` as markers.

        Args:
            other (float, int, or array_like): The other time series/value.
            name (str): Name of the time series.
            other_name (str): Name of the other time series/value.
            other_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for `other`.
            above_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for values above `other`.
            below_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for values below `other`.
            equal_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for values equal `other`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            Can plot against single values such as benchmarks.
            ```py
            df['a'].vbt.timeseries.plot_against(3)
            ```

            ![](/vectorbt/docs/img/timeseries_plot_against_line.png)

            But also against other time series.

            ```py
            df['a'].vbt.timeseries.plot_against(df['b'])
            ```

            ![](/vectorbt/docs/img/timeseries_plot_against_series.png)"""
        if name is None:
            name = self._obj.name
        if other_name is None:
            other_name = getattr(other, 'name', None)

        # Prepare data
        other = reshape_fns.to_1d(other)
        other = reshape_fns.broadcast_to(other, self._obj)
        above_obj = self._obj[self._obj > other]
        below_obj = self._obj[self._obj < other]
        equal_obj = self._obj[self._obj == other]

        # Set up figure
        if fig is None:
            fig = DefaultFigureWidget()
            fig.update_layout(**layout_kwargs)

        # Plot other
        other_scatter = go.Scatter(
            x=other.index,
            y=other,
            line=dict(
                color="grey",
                width=2,
                dash="dot",
            ),
            name=other_name,
            showlegend=other_name is not None
        )
        other_scatter.update(**other_trace_kwargs)
        fig.add_trace(other_scatter)

        # Plot markets
        above_scatter = go.Scatter(
            x=above_obj.index,
            y=above_obj,
            mode='markers',
            marker=dict(
                symbol='circle',
                color='green',
                size=10
            ),
            name=f'{name} (above)',
            showlegend=name is not None
        )
        above_scatter.update(**above_trace_kwargs)
        fig.add_trace(above_scatter)

        below_scatter = go.Scatter(
            x=below_obj.index,
            y=below_obj,
            mode='markers',
            marker=dict(
                symbol='circle',
                color='red',
                size=10
            ),
            name=f'{name} (below)',
            showlegend=name is not None
        )
        below_scatter.update(**below_trace_kwargs)
        fig.add_trace(below_scatter)

        equal_scatter = go.Scatter(
            x=equal_obj.index,
            y=equal_obj,
            mode='markers',
            marker=dict(
                symbol='circle',
                color='grey',
                size=10
            ),
            name=f'{name} (equal)',
            showlegend=name is not None
        )
        equal_scatter.update(**equal_trace_kwargs)
        fig.add_trace(equal_scatter)

        # If other is a straight line, make y-axis symmetric
        if np.all(other.values == other.values.item(0)):
            maxval = np.nanmax(np.abs(self.to_array()))
            space = 0.1 * 2 * maxval
            y = other.values.item(0)
            fig.update_layout(
                yaxis=dict(
                    range=[y-(maxval+space), y+maxval+space]
                ),
                shapes=[dict(
                    type="line",
                    xref="paper",
                    yref='y',
                    x0=0, x1=1, y0=y, y1=y,
                    line=dict(
                        color="grey",
                        width=2,
                        dash="dot",
                    ))]
            )

        return fig


@register_dataframe_accessor('timeseries')
class TimeSeries_DFAccessor(TimeSeries_Accessor, Base_DFAccessor):
    """Accessor with methods for DataFrames only.

    Accessible through `pandas.DataFrame.vbt.timeseries`."""

    def plot(self, trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot each column in DataFrame as a line.

        Args:
            trace_kwargs (dict or list of dict): Keyword arguments passed to each `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            df[['a', 'b']].vbt.timeseries.plot()
            ```

            ![](/vectorbt/docs/img/timeseries_df_plot.png)"""

        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.timeseries.plot(
                trace_kwargs=trace_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig


@register_dataframe_accessor('ohlcv')
class OHLCV_DFAccessor(TimeSeries_DFAccessor):
    """Accessor with methods for DataFrames only.

    Accessible through `pandas.DataFrame.vbt.ohlcv`."""

    def __init__(self, obj):
        super().__init__(obj)
        self()  # set column map

    def __call__(self, open='Open', high='High', low='Low', close='Close', volume='Volume'):
        """Accessor is callable to be able to provide column names."""
        self._column_map = dict(
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume
        )
        return self

    def plot(self,
             display_volume=True,
             candlestick_kwargs={},
             bar_kwargs={},
             **layout_kwargs):
        """Plot OHLCV data.

        Args:
            display_volume (bool): If `True`, displays volume as bar chart.
            candlestick_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Candlestick`.
            bar_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Bar`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            import vectorbt as vbt
            import yfinance as yf

            yf.Ticker("BTC-USD").history(period="max").vbt.ohlcv.plot()
            ```

            ![](/vectorbt/docs/img/ohlcv.png)"""
        open = self._obj[self._column_map['open']]
        high = self._obj[self._column_map['high']]
        low = self._obj[self._column_map['low']]
        close = self._obj[self._column_map['close']]

        fig = DefaultFigureWidget()
        candlestick = go.Candlestick(
            x=self.index,
            open=open,
            high=high,
            low=low,
            close=close,
            name='OHLC',
            yaxis="y2",
            xaxis="x"
        )
        candlestick.update(**candlestick_kwargs)
        fig.add_trace(candlestick)
        if display_volume:
            volume = self._obj[self._column_map['volume']]

            marker_colors = np.empty(volume.shape, dtype=np.object)
            marker_colors[(close.values - open.values) > 0] = 'green'
            marker_colors[(close.values - open.values) == 0] = 'lightgrey'
            marker_colors[(close.values - open.values) < 0] = 'red'
            bar = go.Bar(
                x=self.index,
                y=volume,
                marker_color=marker_colors,
                marker_line_width=0,
                name='Volume',
                yaxis="y",
                xaxis="x"
            )
            bar.update(**bar_kwargs)
            fig.add_trace(bar)
            fig.update_layout(
                yaxis2=dict(
                    domain=[0.33, 1]
                ),
                yaxis=dict(
                    domain=[0, 0.33]
                )
            )
        fig.update_layout(
            showlegend=True,
            xaxis_rangeslider_visible=False,
            xaxis_showgrid=True,
            yaxis_showgrid=True
        )
        fig.update_layout(**layout_kwargs)

        return fig
