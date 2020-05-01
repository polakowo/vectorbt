"""Accessors with functions for working with time series data.

Accessible through `pandas.vbt.timeseries`.

!!! note
    All Series/DataFrames must be `numpy.float64`."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import itertools

from vectorbt.accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks, reshape_fns, index_fns
from vectorbt.utils.common import add_safe_nb_methods
from vectorbt.utils.accessors import Base_DFAccessor, Base_SRAccessor
from vectorbt.timeseries import nb
from vectorbt.widgets.common import DefaultFigureWidget


@add_safe_nb_methods(
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
    nb.rolling_apply_nb,
    nb.expanding_apply_nb,
    module_name='vectorbt.timeseries.nb')
class TimeSeries_Accessor():
    """Accessor with methods for both Series and DataFrames."""
    dtype = np.float64

    @classmethod
    def _validate(cls, obj):
        if cls.dtype is not None:
            checks.assert_dtype(obj, cls.dtype)

    def groupby_apply(self, by, apply_func_nb, on_2d=False):
        """See `vectorbt.timeseries.nb.groupby_apply_nb`."""
        groups, applied = nb.groupby_apply_nb(self.to_2d_array(), by, apply_func_nb, on_2d=on_2d)
        return self.wrap_array(applied, index=groups)

    def resample_apply(self, freq, apply_func_nb, on_2d=False, **kwargs):
        """Resample time-series data and apply function `apply_func_nb` to each group of resampled values.

        Numba equivalent to `pd.Series(a).resample(freq).apply(apply_func_nb, raw=True)`.

        If `on_2d` is `True`, will apply to all columns as matrix, otherwise to each column individually."""
        resampled = self._obj.resample(freq, **kwargs)
        # Build a mask that acts as a map between new and old index
        # It works on resampled.indices instead of resampled.groups, so there is no redundancy
        maxlen = self._obj.shape[0]
        ll = list(resampled.indices.values())
        mask = np.full((len(ll), maxlen), False, bool)
        mask_idxs = np.array(list(itertools.zip_longest(*ll, fillvalue=np.nan))).T
        mask_idxs = (np.arange(mask_idxs.shape[0])[:, None] * maxlen + mask_idxs).flatten()
        mask_idxs = mask_idxs[~np.isnan(mask_idxs)]
        mask_idxs = mask_idxs.astype(int)
        mask_idxs = np.unravel_index(mask_idxs.astype(int), mask.shape)
        mask[mask_idxs] = True
        # Apply a function to each group of values from the old DataFrame index by new mask
        applied = nb.apply_by_mask_nb(self.to_2d_array(), mask, apply_func_nb, on_2d=on_2d)
        # Finally, map output to the new DataFrame using resampled.groups
        applied_obj = self.wrap_array(applied, index=list(resampled.indices.keys()))
        resampled_arr = np.full((resampled.ngroups, self.to_2d_array().shape[1]), np.nan)
        resampled_obj = self.wrap_array(resampled_arr, index=pd.Index(list(resampled.groups.keys()), freq=freq))
        resampled_obj.loc[applied_obj.index] = applied_obj.values
        return resampled_obj

    def rolling_window(self, window, n=None):
        """Split time series into `n` time ranges each `window` long.

        The result will be a new DataFrame with index of length `window` and columns of length
        `len(columns) * n`. If `n` is `None`, will return the maximum number of time ranges.

        Example:
            ```python-repl
            >>> import pandas as pd
            >>> df = pd.DataFrame({
            ...     'a': [1., 2., 3., 4.], 
            ...     'b': [5., 6., 7., 8.]
            ... }, index=['w', 'x', 'y', 'z'])

            >>> print(df.vbt.timeseries.rolling_window(2))
                                    a              b          
            start_date    w    x    y    w    x    y
            0           1.0  2.0  3.0  5.0  6.0  7.0
            1           2.0  3.0  4.0  6.0  7.0  8.0
            ```"""
        cube = nb.rolling_window_nb(self.to_2d_array(), window)
        if n is not None:
            idxs = np.round(np.linspace(0, cube.shape[2]-1, n)).astype(int)
            cube = cube[:, :, idxs]
        else:
            idxs = np.arange(cube.shape[2])
        matrix = np.hstack(cube)
        range_columns = pd.Index(self._obj.index[idxs], name='start_date')
        new_columns = index_fns.combine(reshape_fns.to_2d(self._obj).columns, range_columns)
        return pd.DataFrame(matrix, columns=new_columns)


@register_series_accessor('timeseries')
class TimeSeries_SRAccessor(TimeSeries_Accessor, Base_SRAccessor):
    """Accessor with methods for Series only."""

    def plot(self, name=None, trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot time series as a line.

        Args:
            name (str): Name of the trace.
            trace_kwargs (dict): Keyword arguments passed to [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html).
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            sr = pd.Series([1., 2., 3.], index=['x', 'y', 'z'])

            sr.vbt.timeseries.plot()
            ```

            ![](img/timeseries_sr_plot)"""
        if fig is None:
            fig = DefaultFigureWidget()
            fig.update_layout(**layout_kwargs)
        if name is None:
            name = self._obj.name
        if name is not None:
            fig.update_layout(showlegend=True)

        scatter = go.Scatter(
            x=self._obj.index,
            y=self._obj.values,
            mode='lines',
            name=str(name) if name is not None else None
        )
        scatter.update(**trace_kwargs)
        fig.add_trace(scatter)

        return fig


@register_dataframe_accessor('timeseries')
class TimeSeries_DFAccessor(TimeSeries_Accessor, Base_DFAccessor):
    """Accessor with methods for DataFrames only."""

    def plot(self, trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot each column in time series as a line.

        Args:
            trace_kwargs (dict or list of dict): Keyword arguments passed to each [`plotly.graph_objects.Scatter`](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html).
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            df = pd.DataFrame({'a': [1., 2.], 'b': [3., 4.]}, index=['x', 'y'])

            df.vbt.timeseries.plot()
            ```

            ![](img/timeseries_df_plot.png)"""
        
        for col in range(self._obj.shape[1]):
            fig = self._obj.iloc[:, col].vbt.timeseries.plot(
                trace_kwargs=trace_kwargs,
                fig=fig,
                **layout_kwargs
            )

        return fig


@register_dataframe_accessor('ohlcv')
class OHLCV_DFAccessor(TimeSeries_DFAccessor):
    def __init__(self, obj):
        super().__init__(obj)
        self()  # set column map

    def __call__(self, open='Open', high='High', low='Low', close='Close', volume='Volume'):
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
             trace_kwargs={},
             **layout_kwargs):
        open = self._obj[self._column_map['open']]
        high = self._obj[self._column_map['high']]
        low = self._obj[self._column_map['low']]
        close = self._obj[self._column_map['close']]

        fig = DefaultFigureWidget()
        candlestick = go.Candlestick(
            x=self._obj.index,
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
                x=self._obj.index,
                y=volume,
                marker_color=marker_colors,
                marker_line_width=0,
                name='Volume',
                yaxis="y",
                xaxis="x"
            )
            bar.update(**trace_kwargs)
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
