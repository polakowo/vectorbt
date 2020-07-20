"""Custom pandas accessors.

!!! note
    Input arrays can be of any column-oriented data type, but most output arrays are `np.float64`.
    
```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> ts = pd.DataFrame({
...     'a': [1, 2, 3, 4, 5],
...     'b': [5, 4, 3, 2, 1],
...     'c': [1, 2, 3, 2, 1]
... }, index=pd.Index([
...     datetime(2020, 1, 1),
...     datetime(2020, 1, 2),
...     datetime(2020, 1, 3),
...     datetime(2020, 1, 4),
...     datetime(2020, 1, 5)
... ]))
>>> print(ts)
            a  b  c
2020-01-01  1  5  1
2020-01-02  2  4  2
2020-01-03  3  3  3
2020-01-04  4  2  2
2020-01-05  5  1  1
```"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numba.typed import Dict

from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_property
from vectorbt.base import index_fns, reshape_fns
from vectorbt.base.accessors import Base_Accessor, Base_DFAccessor, Base_SRAccessor
from vectorbt.base.common import add_nb_methods
from vectorbt.generic import nb
from vectorbt.records.drawdowns import Drawdowns
from vectorbt.utils.widgets import CustomFigureWidget

try:  # pragma: no cover
    # Adapted from https://github.com/quantopian/empyrical/blob/master/empyrical/utils.py
    import bottleneck as bn

    nanmean = bn.nanmean
    nanstd = bn.nanstd
    nansum = bn.nansum
    nanmax = bn.nanmax
    nanmin = bn.nanmin
    nanmedian = bn.nanmedian
    nanargmax = bn.nanargmax
    nanargmin = bn.nanargmin
except ImportError:
    # slower numpy
    nanmean = np.nanmean
    nanstd = np.nanstd
    nansum = np.nansum
    nanmax = np.nanmax
    nanmin = np.nanmin
    nanmedian = np.nanmedian
    nanargmax = np.nanargmax
    nanargmin = np.nanargmin


@add_nb_methods([
    nb.fillna_nb,
    nb.fshift_nb,
    nb.diff_nb,
    nb.pct_change_nb,
    nb.ffill_nb,
    nb.product_nb,
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
    nb.expanding_std_nb
], module_name='vectorbt.generic.nb')
class Generic_Accessor(Base_Accessor):
    """Accessor on top of data of any type. For both, Series and DataFrames.

    Accessible through `pd.Series.vbt` and `pd.DataFrame.vbt`.

    You can call the accessor and specify index frequency if your index isn't datetime-like."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Base_Accessor.__init__(self, obj, freq=freq)

    def split_into_ranges(self, n=None, range_len=None):
        """Split into `n` ranges each `range_len` long.

        At least one of `range_len` and `n` must be set.
        If `range_len` is `None`, will split evenly into `n` ranges.
        If `n` is `None`, will return the maximum number of ranges of length `range_len`.

        !!! note
            The datetime-like format of the index will be lost as result of this operation.
            Make sure to store the index metadata such as frequency information beforehand.

        Example:
            ```python-repl
            >>> print(ts.vbt.split_into_ranges(n=2))
                                            a                     b                     c
            range_start 2020-01-01 2020-01-04 2020-01-01 2020-01-04 2020-01-01 2020-01-04
            range_end   2020-01-02 2020-01-05 2020-01-02 2020-01-05 2020-01-02 2020-01-05
            0                  1.0        4.0        5.0        2.0        1.0        2.0
            1                  2.0        5.0        4.0        1.0        2.0        1.0
            >>> print(ts.vbt.split_into_ranges(range_len=4))
                                            a                     b                     c
            range_start 2020-01-01 2020-01-02 2020-01-01 2020-01-02 2020-01-01 2020-01-02
            range_end   2020-01-04 2020-01-05 2020-01-04 2020-01-05 2020-01-04 2020-01-05
            0                  1.0        2.0        5.0        4.0        1.0        2.0
            1                  2.0        3.0        4.0        3.0        2.0        3.0
            2                  3.0        4.0        3.0        2.0        3.0        2.0
            3                  4.0        5.0        2.0        1.0        2.0        1.0
            ```"""
        if range_len is None and n is None:
            raise ValueError("At least range_len or n must be set")

        if range_len is None:
            range_len = len(self.index) // n
        cube = nb.rolling_window_nb(self.to_2d_array(), range_len)
        if n is not None:
            if n > cube.shape[2]:
                raise ValueError(f"n cannot be bigger than the maximum number of ranges {cube.shape[2]}")
            idxs = np.round(np.linspace(0, cube.shape[2] - 1, n)).astype(int)
            cube = cube[:, :, idxs]
        else:
            idxs = np.arange(cube.shape[2])
        matrix = np.hstack(cube)
        range_starts = pd.Index(self.index[idxs], name='range_start')
        range_ends = pd.Index(self.index[idxs + range_len - 1], name='range_end')
        range_columns = index_fns.stack_indexes(range_starts, range_ends)
        new_columns = index_fns.combine_indexes(self.columns, range_columns)
        return pd.DataFrame(matrix, columns=new_columns)

    def rolling_apply(self, window, apply_func_nb, *args, on_matrix=False):
        """See `vectorbt.generic.nb.rolling_apply_nb` and
        `vectorbt.generic.nb.rolling_apply_matrix_nb` for `on_matrix=True`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(ts.vbt.rolling_apply(3, mean_nb))
                          a    b         c
            2020-01-01  1.0  5.0  1.000000
            2020-01-02  1.5  4.5  1.500000
            2020-01-03  2.0  4.0  2.000000
            2020-01-04  3.0  3.0  2.333333
            2020-01-05  4.0  2.0  2.000000
            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(ts.vbt.rolling_apply(3, mean_matrix_nb, on_matrix=True))
                               a         b         c
            2020-01-01  2.333333  2.333333  2.333333
            2020-01-02  2.500000  2.500000  2.500000
            2020-01-03  2.666667  2.666667  2.666667
            2020-01-04  2.777778  2.777778  2.777778
            2020-01-05  2.666667  2.666667  2.666667
            ```"""
        checks.assert_numba_func(apply_func_nb)

        if on_matrix:
            result = nb.rolling_apply_matrix_nb(self.to_2d_array(), window, apply_func_nb, *args)
        else:
            result = nb.rolling_apply_nb(self.to_2d_array(), window, apply_func_nb, *args)
        return self.wrap(result)

    def expanding_apply(self, apply_func_nb, *args, on_matrix=False):
        """See `vectorbt.generic.nb.expanding_apply_nb` and
        `vectorbt.generic.nb.expanding_apply_matrix_nb` for `on_matrix=True`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(ts.vbt.expanding_apply(mean_nb))
                          a    b    c
            2020-01-01  1.0  5.0  1.0
            2020-01-02  1.5  4.5  1.5
            2020-01-03  2.0  4.0  2.0
            2020-01-04  2.5  3.5  2.0
            2020-01-05  3.0  3.0  1.8
            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(ts.vbt.expanding_apply(mean_matrix_nb, on_matrix=True))
                               a         b         c
            2020-01-01  2.333333  2.333333  2.333333
            2020-01-02  2.500000  2.500000  2.500000
            2020-01-03  2.666667  2.666667  2.666667
            2020-01-04  2.666667  2.666667  2.666667
            2020-01-05  2.600000  2.600000  2.600000
            ```"""
        checks.assert_numba_func(apply_func_nb)

        if on_matrix:
            result = nb.expanding_apply_matrix_nb(self.to_2d_array(), apply_func_nb, *args)
        else:
            result = nb.expanding_apply_nb(self.to_2d_array(), apply_func_nb, *args)
        return self.wrap(result)

    def groupby_apply(self, by, apply_func_nb, *args, on_matrix=False, **kwargs):
        """See `vectorbt.generic.nb.groupby_apply_nb` and
        `vectorbt.generic.nb.groupby_apply_matrix_nb` for `on_matrix=True`.

        For `by`, see `pd.DataFrame.groupby`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(ts.vbt.groupby_apply([1, 1, 2, 2, 3], mean_nb))
                 a    b    c
            1  1.5  4.5  1.5
            2  3.5  2.5  2.5
            3  5.0  1.0  1.0
            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(ts.vbt.groupby_apply([1, 1, 2, 2, 3],
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
        return self.wrap_reduced(result, index=list(regrouped.indices.keys()))

    def resample_apply(self, freq, apply_func_nb, *args, on_matrix=False, **kwargs):
        """See `vectorbt.generic.nb.groupby_apply_nb` and
        `vectorbt.generic.nb.groupby_apply_matrix_nb` for `on_matrix=True`.

        For `freq`, see `pd.DataFrame.resample`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(ts.vbt.resample_apply('2d', mean_nb))
                          a    b    c
            2020-01-01  1.5  4.5  1.5
            2020-01-03  3.5  2.5  2.5
            2020-01-05  5.0  1.0  1.0
            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(ts.vbt.resample_apply('2d', mean_matrix_nb, on_matrix=True))
                               a         b         c
            2020-01-01  2.500000  2.500000  2.500000
            2020-01-03  2.833333  2.833333  2.833333
            2020-01-05  2.333333  2.333333  2.333333
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
        result_obj = self.wrap(result, index=list(resampled.indices.keys()))
        resampled_arr = np.full((resampled.ngroups, self.to_2d_array().shape[1]), np.nan)
        resampled_obj = self.wrap(resampled_arr, index=pd.Index(list(resampled.groups.keys()), freq=freq))
        resampled_obj.loc[result_obj.index] = result_obj.values
        return resampled_obj

    def applymap(self, apply_func_nb, *args):
        """See `vectorbt.generic.nb.applymap_nb`.

        Example:
            ```python-repl
            >>> multiply_nb = njit(lambda col, i, a: a ** 2)
            >>> print(ts.vbt.applymap(multiply_nb))
                           a     b    c
            2020-01-01   1.0  25.0  1.0
            2020-01-02   4.0  16.0  4.0
            2020-01-03   9.0   9.0  9.0
            2020-01-04  16.0   4.0  4.0
            2020-01-05  25.0   1.0  1.0
            ```"""
        checks.assert_numba_func(apply_func_nb)

        result = nb.applymap_nb(self.to_2d_array(), apply_func_nb, *args)
        return self.wrap(result)

    def filter(self, filter_func_nb, *args):
        """See `vectorbt.generic.nb.filter_nb`.

        Example:
            ```python-repl
            >>> greater_nb = njit(lambda col, i, a: a > 2)
            >>> print(ts.vbt.filter(greater_nb))
                          a    b    c
            2020-01-01  NaN  5.0  NaN
            2020-01-02  NaN  4.0  NaN
            2020-01-03  3.0  3.0  3.0
            2020-01-04  4.0  NaN  NaN
            2020-01-05  5.0  NaN  NaN
            ```"""
        checks.assert_numba_func(filter_func_nb)

        result = nb.filter_nb(self.to_2d_array(), filter_func_nb, *args)
        return self.wrap(result)

    def apply_and_reduce(self, apply_func_nb, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.generic.nb.apply_and_reduce_nb`.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> greater_nb = njit(lambda col, a: a[a > 2])
            >>> mean_nb = njit(lambda col, a: np.nanmean(a))
            >>> print(ts.vbt.apply_and_reduce(greater_nb, mean_nb))
            a    4.0
            b    4.0
            c    3.0
            dtype: float64
            ```"""
        checks.assert_numba_func(apply_func_nb)
        checks.assert_numba_func(reduce_func_nb)

        result = nb.apply_and_reduce_nb(self.to_2d_array(), apply_func_nb, reduce_func_nb, *args)
        return self.wrap_reduced(result, **kwargs)

    def reduce(self, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.generic.nb.reduce_nb`.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, a: np.nanmean(a))
            >>> print(ts.vbt.reduce(mean_nb))
            a    3.0
            b    3.0
            c    1.8
            dtype: float64
            ```"""
        checks.assert_numba_func(reduce_func_nb)

        result = nb.reduce_nb(self.to_2d_array(), reduce_func_nb, *args)
        return self.wrap_reduced(result, **kwargs)

    def reduce_to_array(self, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.generic.nb.reduce_to_array_nb`.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> min_max_nb = njit(lambda col, a: np.array([np.nanmin(a), np.nanmax(a)]))
            >>> print(ts.vbt.reduce_to_array(min_max_nb, index=['min', 'max']))
                   a    b    c
            min  1.0  1.0  1.0
            max  5.0  5.0  3.0
            ```"""
        checks.assert_numba_func(reduce_func_nb)

        result = nb.reduce_to_array_nb(self.to_2d_array(), reduce_func_nb, *args)
        return self.wrap_reduced(result, **kwargs)

    def min(self, **kwargs):
        """Return min of non-NaN elements."""
        arr = self.to_2d_array()
        if arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            _nanmin = np.nanmin
        else:
            _nanmin = nanmin
        return self.wrap_reduced(_nanmin(arr, axis=0), **kwargs)

    def max(self, **kwargs):
        """Return max of non-NaN elements."""
        arr = self.to_2d_array()
        if arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            _nanmax = np.nanmax
        else:
            _nanmax = nanmax
        return self.wrap_reduced(_nanmax(arr, axis=0), **kwargs)

    def mean(self, **kwargs):
        """Return mean of non-NaN elements."""
        arr = self.to_2d_array()
        if arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            _nanmean = np.nanmean
        else:
            _nanmean = nanmean
        return self.wrap_reduced(_nanmean(arr, axis=0), **kwargs)

    def median(self, **kwargs):
        """Return median of non-NaN elements."""
        arr = self.to_2d_array()
        if arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            _nanmedian = np.nanmedian
        else:
            _nanmedian = nanmedian
        return self.wrap_reduced(_nanmedian(arr, axis=0), **kwargs)

    def std(self, ddof=1, **kwargs):
        """Return standard deviation of non-NaN elements."""
        arr = self.to_2d_array()
        if arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            _nanstd = np.nanstd
        else:
            _nanstd = nanstd
        return self.wrap_reduced(_nanstd(arr, ddof=ddof, axis=0), **kwargs)

    def sum(self, **kwargs):
        """Return sum of non-NaN elements."""
        arr = self.to_2d_array()
        if arr.dtype != int and arr.dtype != float:
            # bottleneck can't consume other than that
            _nansum = np.nansum
        else:
            _nansum = nansum
        return self.wrap_reduced(_nansum(arr, axis=0), **kwargs)

    def count(self, **kwargs):
        """Return count of non-NaN elements."""
        return self.wrap_reduced(np.sum(~np.isnan(self.to_2d_array()), axis=0), **kwargs)

    def idxmin(self, **kwargs):
        """Return index of min of non-NaN elements."""
        return self.wrap_reduced(self.index[nanargmin(self.to_2d_array(), axis=0)], **kwargs)

    def idxmax(self, **kwargs):
        """Return index of max of non-NaN elements."""
        return self.wrap_reduced(self.index[nanargmax(self.to_2d_array(), axis=0)], **kwargs)

    def describe(self, percentiles=None, ddof=1, **kwargs):
        """See `vectorbt.generic.nb.describe_reduce_nb`.

        `**kwargs` will be passed to `Generic_Accessor.wrap_reduced`.

        For `percentiles`, see `pd.DataFrame.describe`.

        Example:
            ```python-repl
            >>> print(ts.vbt.describe())
                          a         b        c
            count  5.000000  5.000000  5.00000
            mean   3.000000  3.000000  1.80000
            std    1.581139  1.581139  0.83666
            min    1.000000  1.000000  1.00000
            25%    2.000000  2.000000  1.00000
            50%    3.000000  3.000000  2.00000
            75%    4.000000  4.000000  2.00000
            max    5.000000  5.000000  3.00000
            ```"""
        if percentiles is not None:
            percentiles = reshape_fns.to_1d(percentiles, raw=True)
        else:
            percentiles = np.array([0.25, 0.5, 0.75])
        percentiles = percentiles.tolist()
        if 0.5 not in percentiles:
            percentiles.append(0.5)
        percentiles = np.unique(percentiles)
        perc_formatted = pd.io.formats.format.format_percentiles(percentiles)
        index = pd.Index(['count', 'mean', 'std', 'min', *perc_formatted, 'max'])
        return self.reduce_to_array(nb.describe_reduce_nb, percentiles, ddof, index=index, **kwargs)

    def drawdown(self):
        """Drawdown series."""
        return self.wrap(self.to_2d_array() / nb.expanding_max_nb(self.to_2d_array()) - 1)

    @cached_property
    def drawdowns(self):
        """Drawdown records.

        See `vectorbt.records.drawdowns.Drawdowns`."""
        return Drawdowns.from_ts(self._obj, freq=self.freq)


class Generic_SRAccessor(Generic_Accessor, Base_SRAccessor):
    """Accessor on top of data of any type. For Series only.

    Accessible through `pd.Series.vbt`."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Base_SRAccessor.__init__(self, obj, freq=freq)
        Generic_Accessor.__init__(self, obj, freq=freq)

    def plot(self, name=None, trace_kwargs={}, fig=None, **layout_kwargs):  # pragma: no cover
        """Plot Series as a line.

        Args:
            name (str): Name of the trace.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            ts['a'].vbt.plot()
            ```

            ![](/vectorbt/docs/img/tseries_sr_plot.png)"""
        if fig is None:
            fig = CustomFigureWidget()
        fig.update_layout(**layout_kwargs)
        if name is None:
            name = self._obj.name

        scatter = go.Scatter(
            x=self.index,
            y=self._obj.values,
            mode='lines',
            name=str(name),
            showlegend=name is not None
        )
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter)

        return fig


class Generic_DFAccessor(Generic_Accessor, Base_DFAccessor):
    """Accessor on top of data of any type. For DataFrames only.

    Accessible through `pd.DataFrame.vbt`."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Base_DFAccessor.__init__(self, obj, freq=freq)
        Generic_Accessor.__init__(self, obj, freq=freq)

    def plot(self, trace_kwargs={}, fig=None, **layout_kwargs):  # pragma: no cover
        """Plot each column in DataFrame as a line.

        Args:
            trace_kwargs (dict or list of dict): Keyword arguments passed to each `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            ts[['a', 'b']].vbt.plot()
            ```

            ![](/vectorbt/docs/img/tseries_df_plot.png)"""

        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.plot(
                trace_kwargs=trace_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig
