"""Custom pandas accessors.

!!! note
    Input arrays can be of any type, but most output arrays are `np.float64`.

    Accessors do not utilize caching.
    
```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd
>>> from numba import njit
>>> from datetime import datetime

>>> df = pd.DataFrame({
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
>>> df
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
import warnings

from vectorbt.utils import checks
from vectorbt.utils.config import merge_kwargs
from vectorbt.base import index_fns, reshape_fns
from vectorbt.base.accessors import Base_Accessor, Base_DFAccessor, Base_SRAccessor
from vectorbt.base.common import add_nb_methods
from vectorbt.generic import plotting, nb
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
    nb.expanding_min_nb,
    nb.expanding_max_nb,
    nb.expanding_mean_nb
], module_name='vectorbt.generic.nb')
class Generic_Accessor(Base_Accessor):
    """Accessor on top of data of any type. For both, Series and DataFrames.

    Accessible through `pd.Series.vbt` and `pd.DataFrame.vbt`."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Base_Accessor.__init__(self, obj, freq=freq)

    def rolling_std(self, window, minp=1, ddof=1):  # pragma: no cover
        """See `vectorbt.generic.nb.rolling_std_nb`."""
        return self.wrap(nb.rolling_std_nb(self.to_2d_array(), window, minp=minp, ddof=ddof))

    def expanding_std(self, minp=1, ddof=1):  # pragma: no cover
        """See `vectorbt.generic.nb.expanding_std_nb`."""
        return self.wrap(nb.expanding_std_nb(self.to_2d_array(), minp=minp, ddof=ddof))

    def ewm_mean(self, span, minp=0, adjust=True):  # pragma: no cover
        """See `vectorbt.generic.nb.ewm_mean_nb`."""
        return self.wrap(nb.ewm_mean_nb(self.to_2d_array(), span, minp=minp, adjust=adjust))

    def ewm_std(self, span, minp=0, adjust=True, ddof=1):  # pragma: no cover
        """See `vectorbt.generic.nb.ewm_std_nb`."""
        return self.wrap(nb.ewm_std_nb(self.to_2d_array(), span, minp=minp, adjust=adjust, ddof=ddof))

    def split_into_ranges(self, n=None, range_len=None, start_idxs=None, end_idxs=None):
        """Either split into `n` ranges each `range_len` long, or split into ranges between
        `start_idxs` and `end_idxs`.

        At least one of `range_len`, `n`, or `start_idxs` and `end_idxs` must be set.
        If `range_len` is `None`, will split evenly into `n` ranges.
        If `n` is `None`, will return the maximum number of ranges of length `range_len`.
        If `start_idxs` and `end_idxs`, will split into ranges between both arrays.
        Both index arrays must be either NumPy arrays with positions (last exclusive)
        or pandas indexes with labels (last inclusive).

        Created levels `range_start` and `range_end` will contain labels (last inclusive).

        !!! note
            Ranges must have the same length.

            The datetime-like format of the index will be lost as result of this operation.
            Make sure to store the index metadata such as frequency information beforehand.

        Example:
            ```python-repl
            >>> df.vbt.split_into_ranges(n=2)
                                            a                     b                     c
            range_start 2020-01-01 2020-01-04 2020-01-01 2020-01-04 2020-01-01 2020-01-04
            range_end   2020-01-02 2020-01-05 2020-01-02 2020-01-05 2020-01-02 2020-01-05
            0                  1.0        4.0        5.0        2.0        1.0        2.0
            1                  2.0        5.0        4.0        1.0        2.0        1.0
            >>> df.vbt.split_into_ranges(range_len=4)
                                            a                     b                     c
            range_start 2020-01-01 2020-01-02 2020-01-01 2020-01-02 2020-01-01 2020-01-02
            range_end   2020-01-04 2020-01-05 2020-01-04 2020-01-05 2020-01-04 2020-01-05
            0                  1.0        2.0        5.0        4.0        1.0        2.0
            1                  2.0        3.0        4.0        3.0        2.0        3.0
            2                  3.0        4.0        3.0        2.0        3.0        2.0
            3                  4.0        5.0        2.0        1.0        2.0        1.0
            >>> df.vbt.split_into_ranges(start_idxs=[0, 1], end_idxs=[4, 5])
                                            a                     b                     c
            range_start 2020-01-01 2020-01-02 2020-01-01 2020-01-02 2020-01-01 2020-01-02
            range_end   2020-01-04 2020-01-05 2020-01-04 2020-01-05 2020-01-04 2020-01-05
            0                    1          2          5          4          1          2
            1                    2          3          4          3          2          3
            2                    3          4          3          2          3          2
            3                    4          5          2          1          2          1
            >>> df.vbt.split_into_ranges(
            ...     start_idxs=pd.Index(['2020-01-01', '2020-01-03']),
            ...     end_idxs=pd.Index(['2020-01-02', '2020-01-04'])
            ... )
                                            a                     b                     c
            range_start 2020-01-01 2020-01-03 2020-01-01 2020-01-03 2020-01-01 2020-01-03
            range_end   2020-01-02 2020-01-04 2020-01-02 2020-01-04 2020-01-02 2020-01-04
            0                    1          3          5          3          1          3
            1                    2          4          4          2          2          2
            ```"""
        if start_idxs is None and end_idxs is None:
            if range_len is None and n is None:
                raise ValueError("At least range_len, n, or start_idxs and end_idxs must be set")
            if range_len is None:
                range_len = len(self.index) // n
            start_idxs = np.arange(len(self.index) - range_len + 1)
            end_idxs = np.arange(range_len, len(self.index) + 1)
        elif start_idxs is None or end_idxs is None:
            raise ValueError("Both start_idxs and end_idxs must be set")
        else:
            if isinstance(start_idxs, pd.Index):
                start_idxs = np.where(self.index.isin(start_idxs))[0]
            else:
                start_idxs = np.asarray(start_idxs)
            if isinstance(end_idxs, pd.Index):
                end_idxs = np.where(self.index.isin(end_idxs))[0] + 1
            else:
                end_idxs = np.asarray(end_idxs)

        if np.any((end_idxs - start_idxs) != (end_idxs - start_idxs).item(0)):
            raise ValueError("Ranges must have the same length")

        if n is not None:
            if n > len(start_idxs):
                raise ValueError(f"n cannot be bigger than the maximum number of ranges {len(start_idxs)}")
            idxs = np.round(np.linspace(0, len(start_idxs) - 1, n)).astype(int)
            start_idxs = start_idxs[idxs]
            end_idxs = end_idxs[idxs]
        matrix = nb.concat_ranges_nb(self.to_2d_array(), start_idxs, end_idxs)
        range_starts = pd.Index(self.index[start_idxs], name='range_start')
        range_ends = pd.Index(self.index[end_idxs - 1], name='range_end')
        range_columns = index_fns.stack_indexes(range_starts, range_ends)
        new_columns = index_fns.combine_indexes(self.columns, range_columns)
        return pd.DataFrame(matrix, columns=new_columns)

    def rolling_apply(self, window, apply_func_nb, *args, on_matrix=False):
        """See `vectorbt.generic.nb.rolling_apply_nb` and
        `vectorbt.generic.nb.rolling_apply_matrix_nb` for `on_matrix=True`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> df.vbt.rolling_apply(3, mean_nb)
                          a    b         c
            2020-01-01  1.0  5.0  1.000000
            2020-01-02  1.5  4.5  1.500000
            2020-01-03  2.0  4.0  2.000000
            2020-01-04  3.0  3.0  2.333333
            2020-01-05  4.0  2.0  2.000000
            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> df.vbt.rolling_apply(3, mean_matrix_nb, on_matrix=True)
                               a         b         c
            2020-01-01  2.333333  2.333333  2.333333
            2020-01-02  2.500000  2.500000  2.500000
            2020-01-03  2.666667  2.666667  2.666667
            2020-01-04  2.777778  2.777778  2.777778
            2020-01-05  2.666667  2.666667  2.666667
            ```"""
        checks.assert_numba_func(apply_func_nb)

        if on_matrix:
            out = nb.rolling_apply_matrix_nb(self.to_2d_array(), window, apply_func_nb, *args)
        else:
            out = nb.rolling_apply_nb(self.to_2d_array(), window, apply_func_nb, *args)
        return self.wrap(out)

    def expanding_apply(self, apply_func_nb, *args, on_matrix=False):
        """See `vectorbt.generic.nb.expanding_apply_nb` and
        `vectorbt.generic.nb.expanding_apply_matrix_nb` for `on_matrix=True`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> df.vbt.expanding_apply(mean_nb)
                          a    b    c
            2020-01-01  1.0  5.0  1.0
            2020-01-02  1.5  4.5  1.5
            2020-01-03  2.0  4.0  2.0
            2020-01-04  2.5  3.5  2.0
            2020-01-05  3.0  3.0  1.8
            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> df.vbt.expanding_apply(mean_matrix_nb, on_matrix=True)
                               a         b         c
            2020-01-01  2.333333  2.333333  2.333333
            2020-01-02  2.500000  2.500000  2.500000
            2020-01-03  2.666667  2.666667  2.666667
            2020-01-04  2.666667  2.666667  2.666667
            2020-01-05  2.600000  2.600000  2.600000
            ```"""
        checks.assert_numba_func(apply_func_nb)

        if on_matrix:
            out = nb.expanding_apply_matrix_nb(self.to_2d_array(), apply_func_nb, *args)
        else:
            out = nb.expanding_apply_nb(self.to_2d_array(), apply_func_nb, *args)
        return self.wrap(out)

    def groupby_apply(self, by, apply_func_nb, *args, on_matrix=False, **kwargs):
        """See `vectorbt.generic.nb.groupby_apply_nb` and
        `vectorbt.generic.nb.groupby_apply_matrix_nb` for `on_matrix=True`.

        For `by`, see `pd.DataFrame.groupby`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> df.vbt.groupby_apply([1, 1, 2, 2, 3], mean_nb)
                 a    b    c
            1  1.5  4.5  1.5
            2  3.5  2.5  2.5
            3  5.0  1.0  1.0
            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> df.vbt.groupby_apply([1, 1, 2, 2, 3], mean_matrix_nb, on_matrix=True)
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
            out = nb.groupby_apply_matrix_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        else:
            out = nb.groupby_apply_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        return self.wrap_reduced(out, index=list(regrouped.indices.keys()))

    def resample_apply(self, freq, apply_func_nb, *args, on_matrix=False, **kwargs):
        """See `vectorbt.generic.nb.groupby_apply_nb` and
        `vectorbt.generic.nb.groupby_apply_matrix_nb` for `on_matrix=True`.

        For `freq`, see `pd.DataFrame.resample`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> df.vbt.resample_apply('2d', mean_nb)
                          a    b    c
            2020-01-01  1.5  4.5  1.5
            2020-01-03  3.5  2.5  2.5
            2020-01-05  5.0  1.0  1.0
            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> df.vbt.resample_apply('2d', mean_matrix_nb, on_matrix=True)
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
            out = nb.groupby_apply_matrix_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        else:
            out = nb.groupby_apply_nb(self.to_2d_array(), groups, apply_func_nb, *args)
        out_obj = self.wrap(out, index=list(resampled.indices.keys()))
        resampled_arr = np.full((resampled.ngroups, self.to_2d_array().shape[1]), np.nan)
        resampled_obj = self.wrap(resampled_arr, index=pd.Index(list(resampled.groups.keys()), freq=freq))
        resampled_obj.loc[out_obj.index] = out_obj.values
        return resampled_obj

    def applymap(self, apply_func_nb, *args):
        """See `vectorbt.generic.nb.applymap_nb`.

        Example:
            ```python-repl
            >>> multiply_nb = njit(lambda col, i, a: a ** 2)
            >>> df.vbt.applymap(multiply_nb)
                           a     b    c
            2020-01-01   1.0  25.0  1.0
            2020-01-02   4.0  16.0  4.0
            2020-01-03   9.0   9.0  9.0
            2020-01-04  16.0   4.0  4.0
            2020-01-05  25.0   1.0  1.0
            ```"""
        checks.assert_numba_func(apply_func_nb)

        out = nb.applymap_nb(self.to_2d_array(), apply_func_nb, *args)
        return self.wrap(out)

    def filter(self, filter_func_nb, *args):
        """See `vectorbt.generic.nb.filter_nb`.

        Example:
            ```python-repl
            >>> greater_nb = njit(lambda col, i, a: a > 2)
            >>> df.vbt.filter(greater_nb)
                          a    b    c
            2020-01-01  NaN  5.0  NaN
            2020-01-02  NaN  4.0  NaN
            2020-01-03  3.0  3.0  3.0
            2020-01-04  4.0  NaN  NaN
            2020-01-05  5.0  NaN  NaN
            ```"""
        checks.assert_numba_func(filter_func_nb)

        out = nb.filter_nb(self.to_2d_array(), filter_func_nb, *args)
        return self.wrap(out)

    def apply_and_reduce(self, apply_func_nb, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.generic.nb.apply_and_reduce_nb`.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> greater_nb = njit(lambda col, a: a[a > 2])
            >>> mean_nb = njit(lambda col, a: np.nanmean(a))
            >>> df.vbt.apply_and_reduce(greater_nb, mean_nb)
            a    4.0
            b    4.0
            c    3.0
            dtype: float64
            ```"""
        checks.assert_numba_func(apply_func_nb)
        checks.assert_numba_func(reduce_func_nb)

        out = nb.apply_and_reduce_nb(self.to_2d_array(), apply_func_nb, reduce_func_nb, *args)
        return self.wrap_reduced(out, **kwargs)

    def reduce(self, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.generic.nb.reduce_nb`.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, a: np.nanmean(a))
            >>> df.vbt.reduce(mean_nb)
            a    3.0
            b    3.0
            c    1.8
            dtype: float64
            ```"""
        checks.assert_numba_func(reduce_func_nb)

        out = nb.reduce_nb(self.to_2d_array(), reduce_func_nb, *args)
        return self.wrap_reduced(out, **kwargs)

    def reduce_to_array(self, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.generic.nb.reduce_to_array_nb`.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> min_max_nb = njit(lambda col, a: np.array([np.nanmin(a), np.nanmax(a)]))
            >>> df.vbt.reduce_to_array(min_max_nb, index=['min', 'max'])
                   a    b    c
            min  1.0  1.0  1.0
            max  5.0  5.0  3.0
            ```"""
        checks.assert_numba_func(reduce_func_nb)

        out = nb.reduce_to_array_nb(self.to_2d_array(), reduce_func_nb, *args)
        return self.wrap_reduced(out, **kwargs)

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
        """Return labeled index of min of non-NaN elements."""
        obj = self.to_2d_array()
        out = np.full(obj.shape[1], np.nan, dtype=np.object)
        nan_mask = np.all(np.isnan(obj), axis=0)
        out[~nan_mask] = self.index[nanargmin(obj[:, ~nan_mask], axis=0)]
        return self.wrap_reduced(out, **kwargs)

    def idxmax(self, **kwargs):
        """Return labeled index of max of non-NaN elements."""
        obj = self.to_2d_array()
        out = np.full(obj.shape[1], np.nan, dtype=np.object)
        nan_mask = np.all(np.isnan(obj), axis=0)
        out[~nan_mask] = self.index[nanargmax(obj[:, ~nan_mask], axis=0)]
        return self.wrap_reduced(out, **kwargs)

    def describe(self, percentiles=None, ddof=1, **kwargs):
        """See `vectorbt.generic.nb.describe_reduce_nb`.

        `**kwargs` will be passed to `vectorbt.base.array_wrapper.ArrayWrapper.wrap_reduced`.

        For `percentiles`, see `pd.DataFrame.describe`.

        Example:
            ```python-repl
            >>> df.vbt.describe()
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

    def drawdowns(self, **kwargs):
        """Generate drawdown records.

        See `vectorbt.records.drawdowns.Drawdowns`."""
        return Drawdowns.from_ts(self._obj, freq=self.freq, **kwargs)

    # ############# Plotting ############# #

    def bar(self, trace_names=None, x_labels=None, **kwargs):  # pragma: no cover
        """See `vectorbt.generic.plotting.create_bar`."""
        if x_labels is None:
            x_labels = self.index
        if trace_names is None:
            if self.is_frame() or (self.is_series() and self.name is not None):
                trace_names = self.columns
        return plotting.create_bar(
            data=self.to_2d_array(),
            trace_names=trace_names,
            x_labels=x_labels,
            **kwargs
        )

    def scatter(self, trace_names=None, x_labels=None, **kwargs):  # pragma: no cover
        """See `vectorbt.generic.plotting.create_scatter`."""
        if x_labels is None:
            x_labels = self.index
        if trace_names is None:
            if self.is_frame() or (self.is_series() and self.name is not None):
                trace_names = self.columns
        return plotting.create_scatter(
            data=self.to_2d_array(),
            trace_names=trace_names,
            x_labels=x_labels,
            **kwargs
        )

    def hist(self, trace_names=None, **kwargs):  # pragma: no cover
        """See `vectorbt.generic.plotting.create_hist`."""
        if trace_names is None:
            if self.is_frame() or (self.is_series() and self.name is not None):
                trace_names = self.columns
        return plotting.create_hist(
            data=self.to_2d_array(),
            trace_names=trace_names,
            **kwargs
        )

    def box(self, trace_names=None, **kwargs):  # pragma: no cover
        """See `vectorbt.generic.plotting.create_box`."""
        if trace_names is None:
            if self.is_frame() or (self.is_series() and self.name is not None):
                trace_names = self.columns
        return plotting.create_box(
            data=self.to_2d_array(),
            trace_names=trace_names,
            **kwargs
        )


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
            ```python-repl
            >>> df['a'].vbt.plot()
            ```

            ![](/vectorbt/docs/img/sr_plot.png)"""
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

    def heatmap(self, x_level=None, y_level=None, symmetric=False, x_labels=None, y_labels=None,
                slider_level=None, slider_labels=None, **kwargs):  # pragma: no cover
        """Create a heatmap figure based on object's multi-index and values.

        If multi-index contains more than two levels or you want them in specific order,
        pass `x_level` and `y_level`, each (`int` if index or `str` if name) corresponding
        to an axis of the heatmap. Optionally, pass `slider_level` to use a level as a slider.

        See `vectorbt.generic.plotting.create_heatmap` for other keyword arguments."""
        (x_level, y_level), (slider_level,) = index_fns.pick_levels(
            self.index,
            required_levels=(x_level, y_level),
            optional_levels=(slider_level,)
        )

        x_level_vals = self.index.get_level_values(x_level)
        y_level_vals = self.index.get_level_values(y_level)
        x_name = x_level_vals.name if x_level_vals.name is not None else 'x'
        y_name = y_level_vals.name if y_level_vals.name is not None else 'y'
        kwargs = merge_kwargs(dict(
            trace_kwargs=dict(
                hovertemplate=f"{x_name}: %{{x}}<br>" +
                              f"{y_name}: %{{y}}<br>" +
                              "value: %{z}<extra></extra>"
            ),
            xaxis_title=x_level_vals.name,
            yaxis_title=y_level_vals.name
        ), kwargs)

        if slider_level is None:
            # No grouping
            df = self.unstack_to_df(index_levels=x_level, column_levels=y_level, symmetric=symmetric)
            fig = df.vbt.heatmap(x_labels=x_labels, y_labels=y_labels, **kwargs)
        else:
            # Requires grouping
            # See https://plotly.com/python/sliders/
            fig = None
            _slider_labels = []
            for i, (name, group) in enumerate(self._obj.groupby(level=slider_level)):
                if slider_labels is not None:
                    name = slider_labels[i]
                _slider_labels.append(name)
                df = group.vbt.unstack_to_df(index_levels=x_level, column_levels=y_level, symmetric=symmetric)
                if x_labels is None:
                    x_labels = df.columns
                if y_labels is None:
                    y_labels = df.index
                _kwargs = merge_kwargs(dict(
                    trace_kwargs=dict(
                        name=str(name) if name is not None else None,
                        visible=False
                    ),
                    width=600,
                    height=520,
                ), kwargs)
                fig = plotting.create_heatmap(
                    data=df.vbt.to_2d_array(),
                    x_labels=x_labels,
                    y_labels=y_labels,
                    fig=fig,
                    **_kwargs
                )
            fig.data[0].visible = True
            steps = []
            for i in range(len(fig.data)):
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)}, {}],
                    label=str(_slider_labels[i]) if _slider_labels[i] is not None else None
                )
                step["args"][0]["visible"][i] = True
                steps.append(step)
            prefix = f'{self.index.names[slider_level]}: ' if self.index.names[slider_level] is not None else None
            sliders = [dict(
                active=0,
                currentvalue={"prefix": prefix},
                pad={"t": 50},
                steps=steps
            )]
            fig.update_layout(
                sliders=sliders
            )

        return fig

    def volume(self, x_level=None, y_level=None, z_level=None, x_labels=None, y_labels=None,
               z_labels=None, slider_level=None, slider_labels=None, **kwargs):  # pragma: no cover
        """Create a 3D volume figure based on object's multi-index and values.

        If multi-index contains more than three levels or you want them in specific order, pass
        `x_level`, `y_level`, and `z_level`, each (`int` if index or `str` if name) corresponding
        to an axis of the volume. Optionally, pass `slider_level` to use a level as a slider.

        See `vectorbt.generic.plotting.create_volume` for other keyword arguments."""
        (x_level, y_level, z_level), (slider_level,) = index_fns.pick_levels(
            self.index,
            required_levels=(x_level, y_level, z_level),
            optional_levels=(slider_level,)
        )

        x_level_vals = self.index.get_level_values(x_level)
        y_level_vals = self.index.get_level_values(y_level)
        z_level_vals = self.index.get_level_values(z_level)
        # Labels are just unique level values
        if x_labels is None:
            x_labels = np.unique(x_level_vals)
        if y_labels is None:
            y_labels = np.unique(y_level_vals)
        if z_labels is None:
            z_labels = np.unique(z_level_vals)

        x_name = x_level_vals.name if x_level_vals.name is not None else 'x'
        y_name = y_level_vals.name if y_level_vals.name is not None else 'y'
        z_name = z_level_vals.name if z_level_vals.name is not None else 'z'
        kwargs = merge_kwargs(dict(
            trace_kwargs=dict(
                hovertemplate=f"{x_name}: %{{x}}<br>" +
                              f"{y_name}: %{{y}}<br>" +
                              f"{z_name}: %{{z}}<br>" +
                              "value: %{value}<extra></extra>"
            ),
            scene=dict(
                xaxis_title=x_level_vals.name,
                yaxis_title=y_level_vals.name,
                zaxis_title=z_level_vals.name
            )
        ), kwargs)

        contains_nans = False
        if slider_level is None:
            # No grouping
            v = self.unstack_to_array(levels=(x_level, y_level, z_level))
            if np.isnan(v).any():
                contains_nans = True
            fig = plotting.create_volume(
                data=v,
                x_labels=x_labels,
                y_labels=y_labels,
                z_labels=z_labels,
                **kwargs
            )
        else:
            # Requires grouping
            # See https://plotly.com/python/sliders/
            fig = None
            _slider_labels = []
            for i, (name, group) in enumerate(self._obj.groupby(level=slider_level)):
                if slider_labels is not None:
                    name = slider_labels[i]
                _slider_labels.append(name)
                v = group.vbt.unstack_to_array(levels=(x_level, y_level, z_level))
                if np.isnan(v).any():
                    contains_nans = True
                _kwargs = merge_kwargs(dict(
                    trace_kwargs=dict(
                        name=str(name) if name is not None else None,
                        visible=False
                    ),
                    width=700,
                    height=520,
                ), kwargs)
                fig = plotting.create_volume(
                    data=v,
                    x_labels=x_labels,
                    y_labels=y_labels,
                    z_labels=z_labels,
                    fig=fig,
                    **_kwargs
                )
            fig.data[0].visible = True
            steps = []
            for i in range(len(fig.data)):
                step = dict(
                    method="update",
                    args=[{"visible": [False] * len(fig.data)}, {}],
                    label=str(_slider_labels[i]) if _slider_labels[i] is not None else None
                )
                step["args"][0]["visible"][i] = True
                steps.append(step)
            prefix = f'{self.index.names[slider_level]}: ' if self.index.names[slider_level] is not None else None
            sliders = [dict(
                active=0,
                currentvalue={"prefix": prefix},
                pad={"t": 50},
                steps=steps
            )]
            fig.update_layout(
                sliders=sliders
            )

        if contains_nans:
            warnings.warn("Data contains NaNs. In case of visualization issues, use .show() method on the widget.")
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
            ```python-repl
            >>> df[['a', 'b']].vbt.plot()
            ```

            ![](/vectorbt/docs/img/df_plot.png)"""

        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.plot(
                trace_kwargs=trace_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig

    def heatmap(self, x_labels=None, y_labels=None, **kwargs):  # pragma: no cover
        """See `vectorbt.generic.plotting.create_heatmap`."""
        if x_labels is None:
            x_labels = self.columns
        if y_labels is None:
            y_labels = self.index
        return plotting.create_heatmap(
            data=self.to_2d_array(),
            x_labels=x_labels,
            y_labels=y_labels,
            **kwargs
        )
