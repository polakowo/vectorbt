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
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    datetime(2020, 1, 3),
    datetime(2020, 1, 4),
    datetime(2020, 1, 5)
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
from numba.typed import Dict

from vectorbt import defaults
from vectorbt.accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks
from vectorbt.base import index_fns, reshape_fns
from vectorbt.base.accessors import Base_Accessor, Base_DFAccessor, Base_SRAccessor
from vectorbt.base.common import add_nb_methods
from vectorbt.tseries import nb
from vectorbt.tseries.common import TSArrayWrapper
from vectorbt.records.drawdowns import Drawdowns
from vectorbt.widgets.common import DefaultFigureWidget

try:
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
], module_name='vectorbt.tseries.nb')
class TimeSeries_Accessor(TSArrayWrapper, Base_Accessor):
    """Accessor on top of time series. For both, Series and DataFrames.

    Accessible through `pandas.Series.vbt.tseries` and `pandas.DataFrame.vbt.tseries`.

    You can call the accessor and specify index frequency if your index isn't datetime-like."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Base_Accessor.__init__(self, obj)

        # Initialize array wrapper
        wrapper = TSArrayWrapper.from_obj(obj)
        TSArrayWrapper.__init__(
            self,
            index=wrapper.index,
            columns=wrapper.columns,
            ndim=wrapper.ndim,
            freq=freq)

    def split_into_ranges(self, range_len=None, n=None):
        """Split time series into `n` ranges each `range_len` long.

        At least one of `range_len` and `n` must be set.
        If `range_len` is `None`, will split evenly into `n` ranges.
        If `n` is `None`, will return the maximum number of ranges of length `range_len`.

        !!! note
            The datetime-like format of the index will be lost as result of this operation.
            Make sure to store the index metadata such as frequency information beforehand.

        Example:
            ```python-repl
            >>> print(df.vbt.tseries.split_into_ranges(n=2))
                                           a                     b                     c
            start_date 2020-01-01 2020-01-04 2020-01-01 2020-01-04 2020-01-01 2020-01-04
            end_date   2020-01-02 2020-01-05 2020-01-02 2020-01-05 2020-01-02 2020-01-05
            0                 1.0        4.0        5.0        2.0        1.0        2.0
            1                 2.0        5.0        4.0        1.0        2.0        1.0

            >>> print(df.vbt.tseries.split_into_ranges(range_len=4))
                                           a                     b                     c
            start_date 2020-01-01 2020-01-02 2020-01-01 2020-01-02 2020-01-01 2020-01-02
            end_date   2020-01-04 2020-01-05 2020-01-04 2020-01-05 2020-01-04 2020-01-05
            0                 1.0        2.0        5.0        4.0        1.0        2.0
            1                 2.0        3.0        4.0        3.0        2.0        3.0
            2                 3.0        4.0        3.0        2.0        3.0        2.0
            3                 4.0        5.0        2.0        1.0        2.0        1.0
            ```"""
        if range_len is None:
            checks.assert_not_none(n)
        elif n is None:
            checks.assert_not_none(range_len)

        if range_len is None:
            range_len = len(self.index) // n
        cube = nb.rolling_window_nb(self.to_2d_array(), range_len)
        if n is not None:
            if n > cube.shape[2]:
                raise Exception(f"n cannot be bigger than the maximum number of ranges {cube.shape[2]}")
            idxs = np.round(np.linspace(0, cube.shape[2] - 1, n)).astype(int)
            cube = cube[:, :, idxs]
        else:
            idxs = np.arange(cube.shape[2])
        matrix = np.hstack(cube)
        start_dates = pd.Index(self.index[idxs], name='start_date')
        end_dates = pd.Index(self.index[idxs + range_len - 1], name='end_date')
        range_columns = index_fns.stack_indexes(start_dates, end_dates)
        new_columns = index_fns.combine_indexes(self.columns, range_columns)
        return pd.DataFrame(matrix, columns=new_columns)

    def rolling_apply(self, window, apply_func_nb, *args, on_matrix=False):
        """See `vectorbt.tseries.nb.rolling_apply_nb` and
        `vectorbt.tseries.nb.rolling_apply_matrix_nb` for `on_matrix=True`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(df.vbt.tseries.rolling_apply(3, mean_nb))
                          a    b         c
            2020-01-01  1.0  5.0  1.000000
            2020-01-02  1.5  4.5  1.500000
            2020-01-03  2.0  4.0  2.000000
            2020-01-04  3.0  3.0  2.333333
            2020-01-05  4.0  2.0  2.000000

            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(df.vbt.tseries.rolling_apply(3,
            ...     mean_matrix_nb, on_matrix=True))
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
        """See `vectorbt.tseries.nb.expanding_apply_nb` and
        `vectorbt.tseries.nb.expanding_apply_matrix_nb` for `on_matrix=True`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(df.vbt.tseries.expanding_apply(mean_nb))
                          a    b    c
            2020-01-01  1.0  5.0  1.0
            2020-01-02  1.5  4.5  1.5
            2020-01-03  2.0  4.0  2.0
            2020-01-04  2.5  3.5  2.0
            2020-01-05  3.0  3.0  1.8

            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(df.vbt.tseries.expanding_apply(
            ...     mean_matrix_nb, on_matrix=True))
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
        """See `vectorbt.tseries.nb.groupby_apply_nb` and
        `vectorbt.tseries.nb.groupby_apply_matrix_nb` for `on_matrix=True`.

        For `by`, see `pandas.DataFrame.groupby`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(df.vbt.tseries.groupby_apply([1, 1, 2, 2, 3], mean_nb))
                 a    b    c
            1  1.5  4.5  1.5
            2  3.5  2.5  2.5
            3  5.0  1.0  1.0

            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(df.vbt.tseries.groupby_apply([1, 1, 2, 2, 3],
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
        """See `vectorbt.tseries.nb.groupby_apply_nb` and
        `vectorbt.tseries.nb.groupby_apply_matrix_nb` for `on_matrix=True`.

        For `freq`, see `pandas.DataFrame.resample`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, i, a: np.nanmean(a))
            >>> print(df.vbt.tseries.resample_apply('2d', mean_nb))
                          a    b    c
            2020-01-01  1.5  4.5  1.5
            2020-01-03  3.5  2.5  2.5
            2020-01-05  5.0  1.0  1.0

            >>> mean_matrix_nb = njit(lambda i, a: np.nanmean(a))
            >>> print(df.vbt.tseries.resample_apply('2d',
            ...     mean_matrix_nb, on_matrix=True))
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
        """See `vectorbt.tseries.nb.applymap_nb`.

        Example:
            ```python-repl
            >>> multiply_nb = njit(lambda col, i, a: a ** 2)
            >>> print(df.vbt.tseries.applymap(multiply_nb))
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
        """See `vectorbt.tseries.nb.filter_nb`.

        Example:
            ```python-repl
            >>> greater_nb = njit(lambda col, i, a: a > 2)
            >>> print(df.vbt.tseries.filter(greater_nb))
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
        """See `vectorbt.tseries.nb.apply_and_reduce_nb`.

        `**kwargs` will be passed to `vectorbt.tseries.common.TSArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> greater_nb = njit(lambda col, a: a[a > 2])
            >>> mean_nb = njit(lambda col, a: np.nanmean(a))
            >>> print(df.vbt.tseries.apply_and_reduce(greater_nb, mean_nb))
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
        """See `vectorbt.tseries.nb.reduce_nb`.

        `**kwargs` will be passed to `vectorbt.tseries.common.TSArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> mean_nb = njit(lambda col, a: np.nanmean(a))
            >>> print(df.vbt.tseries.reduce(mean_nb))
            a    3.0
            b    3.0
            c    1.8
            dtype: float64
            ```"""
        checks.assert_numba_func(reduce_func_nb)

        result = nb.reduce_nb(self.to_2d_array(), reduce_func_nb, *args)
        return self.wrap_reduced(result, **kwargs)

    def reduce_to_array(self, reduce_func_nb, *args, **kwargs):
        """See `vectorbt.tseries.nb.reduce_to_array_nb`.

        `**kwargs` will be passed to `vectorbt.tseries.common.TSArrayWrapper.wrap_reduced`.

        Example:
            ```python-repl
            >>> min_max_nb = njit(lambda col, a: np.array([np.nanmin(a), np.nanmax(a)]))
            >>> print(df.vbt.tseries.reduce_to_array(min_max_nb, index=['min', 'max']))
                   a    b    c
            min  1.0  1.0  1.0
            max  5.0  5.0  3.0
            ```"""
        checks.assert_numba_func(reduce_func_nb)

        result = nb.reduce_to_array_nb(self.to_2d_array(), reduce_func_nb, *args)
        return self.wrap_reduced(result, **kwargs)

    def min(self, **kwargs):
        """Return min of non-NaN elements."""
        return self.wrap_reduced(nanmin(self.to_2d_array(), axis=0), **kwargs)

    def max(self, **kwargs):
        """Return max of non-NaN elements."""
        return self.wrap_reduced(nanmax(self.to_2d_array(), axis=0), **kwargs)

    def mean(self, **kwargs):
        """Return mean of non-NaN elements."""
        return self.wrap_reduced(nanmean(self.to_2d_array(), axis=0), **kwargs)

    def median(self, **kwargs):
        """Return median of non-NaN elements."""
        return self.wrap_reduced(nanmedian(self.to_2d_array(), axis=0), **kwargs)

    def std(self, ddof=1, **kwargs):
        """Return standard deviation of non-NaN elements."""
        return self.wrap_reduced(nanstd(self.to_2d_array(), ddof=ddof, axis=0), **kwargs)

    def count(self, **kwargs):
        """Return count of non-NaN elements."""
        return self.wrap_reduced(np.sum(~np.isnan(self.to_2d_array()), axis=0), **kwargs)

    def sum(self, **kwargs):
        """Return sum of non-NaN elements."""
        return self.wrap_reduced(nansum(self.to_2d_array(), axis=0), **kwargs)

    def argmin(self, **kwargs):
        """Return index of min of non-NaN elements."""
        return self.wrap_reduced(self.index[nanargmin(self.to_2d_array(), axis=0)], **kwargs)

    def argmax(self, **kwargs):
        """Return index of max of non-NaN elements."""
        return self.wrap_reduced(self.index[nanargmax(self.to_2d_array(), axis=0)], **kwargs)

    def describe(self, percentiles=[0.25, 0.5, 0.75], ddof=1, **kwargs):
        """See `vectorbt.tseries.nb.describe_reduce_nb`.

        `**kwargs` will be passed to `TimeSeries_Accessor.wrap_reduced`.

        For `percentiles`, see `pandas.DataFrame.describe`.

        Example:
            ```python-repl
            >>> print(df.vbt.tseries.describe())
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
        return self.reduce_to_array(nb.describe_reduce_nb, percentiles, ddof, index=index, **kwargs)

    def drawdown(self):
        """Drawdown series."""
        return self.wrap(self.to_2d_array() / nb.expanding_max_nb(self.to_2d_array()) - 1)

    def drawdowns(self):
        """Drawdown records.

        See `vectorbt.records.drawdowns.Drawdowns`."""
        return Drawdowns.from_ts(self._obj, freq=self.freq)


@register_series_accessor('tseries')
class TimeSeries_SRAccessor(TimeSeries_Accessor, Base_SRAccessor):
    """Accessor on top of time series. For Series only.

    Accessible through `pandas.Series.vbt.tseries`."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Base_SRAccessor.__init__(self, obj)
        TimeSeries_Accessor.__init__(self, obj, freq=freq)

    def plot(self, name=None, trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot Series as a line.

        Args:
            name (str): Name of the time series.
            trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            df['a'].vbt.tseries.plot()
            ```

            ![](/vectorbt/docs/img/tseries_sr_plot.png)"""
        if fig is None:
            fig = DefaultFigureWidget()
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


@register_dataframe_accessor('tseries')
class TimeSeries_DFAccessor(TimeSeries_Accessor, Base_DFAccessor):
    """Accessor on top of time series. For DataFrames only.

    Accessible through `pandas.DataFrame.vbt.tseries`."""

    def __init__(self, obj, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        Base_DFAccessor.__init__(self, obj)
        TimeSeries_Accessor.__init__(self, obj, freq=freq)

    def plot(self, trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot each column in DataFrame as a line.

        Args:
            trace_kwargs (dict or list of dict): Keyword arguments passed to each `plotly.graph_objects.Scatter`.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            df[['a', 'b']].vbt.tseries.plot()
            ```

            ![](/vectorbt/docs/img/tseries_df_plot.png)"""

        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.tseries.plot(
                trace_kwargs=trace_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig


@register_dataframe_accessor('ohlcv')
class OHLCV_DFAccessor(TimeSeries_DFAccessor):
    """Accessor on top of OHLCV data. For DataFrames only.

    Accessible through `pandas.DataFrame.vbt.ohlcv`."""

    def __init__(self, obj, column_names=None, freq=None):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._column_names = column_names

        TimeSeries_DFAccessor.__init__(self, obj, freq=freq)

    def plot(self,
             display_volume=True,
             candlestick_kwargs={},
             bar_kwargs={},
             fig=None,
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
        column_names = defaults.ohlcv['column_names'] if self._column_names is None else self._column_names
        open = self._obj[column_names['open']]
        high = self._obj[column_names['high']]
        low = self._obj[column_names['low']]
        close = self._obj[column_names['close']]

        # Set up figure
        if fig is None:
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
            volume = self._obj[column_names['volume']]

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
