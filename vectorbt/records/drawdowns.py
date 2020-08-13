"""Classes for working with drawdown records."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.defaults import contrast_color_schema
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.base.reshape_fns import to_1d
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.utils.config import merge_kwargs
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.datetime import DatetimeTypes
from vectorbt.records.base import Records, indexing_on_records
from vectorbt.records.enums import DrawdownStatus, drawdown_dt
from vectorbt.records import nb


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `BaseDrawdowns`."""
    new_cols, new_records, _ = indexing_on_records(obj, pd_indexing_func)
    if obj.grouper.group_by is not None:
        new_group_by = obj.grouper.group_by[new_cols]
    else:
        new_group_by = None
    return obj.__class__(
        new_records,
        pd_indexing_func(obj.ts),
        freq=obj.wrapper.freq,
        idx_field=obj.idx_field,
        group_by=new_group_by
    )


class BaseDrawdowns(Records):
    """Extends `Records` for working with drawdown records.

    Requires `records_arr` to have all fields defined in `vectorbt.records.enums.drawdown_dt`."""

    def __init__(self, records_arr, ts, freq=None, idx_field='end_idx', group_by=None):
        Records.__init__(
            self,
            records_arr,
            ArrayWrapper.from_obj(ts, freq=freq),
            idx_field=idx_field,
            group_by=group_by
        )
        PandasIndexer.__init__(self, _indexing_func)

        if not all(field in records_arr.dtype.names for field in drawdown_dt.names):
            raise ValueError("Records array must have all fields defined in drawdown_dt")

        self.ts = ts

    @classmethod
    def from_ts(cls, ts, **kwargs):
        """Build `BaseDrawdowns` from time series `ts`.

        `**kwargs` such as `freq` will be passed to `BaseDrawdowns.__init__`."""
        records_arr = nb.drawdown_records_nb(ts.vbt.to_2d_array())
        return cls(records_arr, ts, **kwargs)

    def filter_by_mask(self, mask, idx_field=None, group_by=None):
        """Return a new class instance, filtered by mask."""
        if idx_field is None:
            idx_field = self.idx_field
        if group_by is None:
            group_by = self.grouper.group_by
        return self.__class__(
            self.records_arr[mask],
            self.ts,
            freq=self.wrapper.freq,
            idx_field=idx_field,
            group_by=group_by
        )

    def plot(self,
             ts_trace_kwargs={},
             peak_trace_kwargs={},
             valley_trace_kwargs={},
             recovery_trace_kwargs={},
             active_trace_kwargs={},
             ptv_shape_kwargs={},
             vtr_shape_kwargs={},
             active_shape_kwargs={},
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot drawdowns over `Drawdowns.ts`.

        Args:
            ts_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for time series.
            peak_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for peak values.
            valley_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for valley values.
            recovery_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for recovery values.
            active_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for active recovery values.
            ptv_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for PtV zones.
            vtr_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for VtR zones.
            active_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for active VtR zones.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.
        Example:
            ```py
            import vectorbt as vbt
            import pandas as pd

            ts = pd.Series([1, 2, 1, 2, 3, 2, 1, 2])
            vbt.records.Drawdowns.from_ts(ts, freq='1 days').plot()
            ```

            ![](/vectorbt/docs/img/drawdowns.png)"""
        if self.wrapper.ndim > 1:
            raise TypeError("You must select a column first")

        fig = self.ts.vbt.plot(trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)

        if self.records_arr.shape[0] == 0:
            return fig

        # Extract information
        start_idx = self.records_arr['start_idx']
        valley_idx = self.records_arr['valley_idx']
        end_idx = self.records_arr['end_idx']
        status = self.records_arr['status']

        start_val = self.ts.values[start_idx]
        valley_val = self.ts.values[valley_idx]
        end_val = self.ts.values[end_idx]

        def get_duration_str(from_idx, to_idx):
            if isinstance(self.wrapper.index, DatetimeTypes):
                duration = self.wrapper.index[to_idx] - self.wrapper.index[from_idx]
            elif self.wrapper.freq is not None:
                duration = self.wrapper.to_time_units(to_idx - from_idx)
            else:
                duration = to_idx - from_idx
            return np.vectorize(str)(duration)

        # Plot peak markers and zones
        peak_mask = start_idx != np.roll(end_idx, 1)  # peak and recovery at same time -> recovery wins
        peak_scatter = go.Scatter(
            x=self.ts.index[start_idx[peak_mask]],
            y=start_val[peak_mask],
            mode='markers',
            marker=dict(
                symbol='circle',
                color=contrast_color_schema['blue'],
                size=7,
                line=dict(
                    width=1,
                    color=adjust_lightness(contrast_color_schema['blue'])
                )
            ),
            name='Peak'
        )
        peak_scatter.update(**peak_trace_kwargs)
        fig.add_trace(peak_scatter)

        recovery_mask = status == DrawdownStatus.Recovered
        if np.any(recovery_mask):
            # Plot valley markers and zones
            valley_drawdown = (valley_val[recovery_mask] - start_val[recovery_mask]) / start_val[recovery_mask]
            valley_duration = get_duration_str(start_idx[recovery_mask], valley_idx[recovery_mask])
            valley_customdata = np.stack((valley_drawdown, valley_duration), axis=1)
            valley_scatter = go.Scatter(
                x=self.ts.index[valley_idx[recovery_mask]],
                y=valley_val[recovery_mask],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    color=contrast_color_schema['red'],
                    size=7,
                    line=dict(
                        width=1,
                        color=adjust_lightness(contrast_color_schema['red'])
                    )
                ),
                name='Valley',
                customdata=valley_customdata,
                hovertemplate="(%{x}, %{y})<br>Drawdown: %{customdata[0]:.2%}<br>Duration: %{customdata[1]}"
            )
            valley_scatter.update(**valley_trace_kwargs)
            fig.add_trace(valley_scatter)

            for i in np.flatnonzero(recovery_mask):
                fig.add_shape(**merge_kwargs(dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=self.ts.index[start_idx[i]],
                    y0=0,
                    x1=self.ts.index[valley_idx[i]],
                    y1=1,
                    fillcolor='red',
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                ), ptv_shape_kwargs))

            # Plot recovery markers and zones
            recovery_return = (end_val[recovery_mask] - valley_val[recovery_mask]) / valley_val[recovery_mask]
            recovery_duration = get_duration_str(valley_idx[recovery_mask], end_idx[recovery_mask])
            recovery_customdata = np.stack((recovery_return, recovery_duration), axis=1)
            recovery_scatter = go.Scatter(
                x=self.ts.index[end_idx[recovery_mask]],
                y=end_val[recovery_mask],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    color=contrast_color_schema['green'],
                    size=7,
                    line=dict(
                        width=1,
                        color=adjust_lightness(contrast_color_schema['green'])
                    )
                ),
                name='Recovery/Peak',
                customdata=recovery_customdata,
                hovertemplate="(%{x}, %{y})<br>Return: %{customdata[0]:.2%}<br>Duration: %{customdata[1]}"
            )
            recovery_scatter.update(**recovery_trace_kwargs)
            fig.add_trace(recovery_scatter)

            for i in np.flatnonzero(recovery_mask):
                fig.add_shape(**merge_kwargs(dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=self.ts.index[valley_idx[i]],
                    y0=0,
                    x1=self.ts.index[end_idx[i]],
                    y1=1,
                    fillcolor='green',
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                ), vtr_shape_kwargs))

        # Plot active markers and zones
        active_mask = ~recovery_mask
        if np.any(active_mask):
            active_drawdown = (valley_val[active_mask] - start_val[active_mask]) / start_val[active_mask]
            active_duration = get_duration_str(valley_idx[active_mask], end_idx[active_mask])
            active_customdata = np.stack((active_drawdown, active_duration), axis=1)
            active_scatter = go.Scatter(
                x=self.ts.index[end_idx[active_mask]],
                y=end_val[active_mask],
                mode='markers',
                marker=dict(
                    symbol='circle',
                    color=contrast_color_schema['orange'],
                    size=7,
                    line=dict(
                        width=1,
                        color=adjust_lightness(contrast_color_schema['orange'])
                    )
                ),
                name='Active',
                customdata=active_customdata,
                hovertemplate="(%{x}, %{y})<br>Drawdown: %{customdata[0]:.2%}<br>Duration: %{customdata[1]}"
            )
            active_scatter.update(**active_trace_kwargs)
            fig.add_trace(active_scatter)

            for i in np.flatnonzero(active_mask):
                fig.add_shape(**merge_kwargs(dict(
                    type="rect",
                    xref="x",
                    yref="paper",
                    x0=self.ts.index[start_idx[i]],
                    y0=0,
                    x1=self.ts.index[end_idx[i]],
                    y1=1,
                    fillcolor='orange',
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                ), active_shape_kwargs))

        return fig

    @cached_property
    def start_value(self):
        """Start value of each drawdown."""
        return self.map(nb.dd_start_value_map_nb, self.ts.vbt.to_2d_array())

    @cached_property
    def valley_value(self):
        """Valley value of each drawdown."""
        return self.map(nb.dd_valley_value_map_nb, self.ts.vbt.to_2d_array())

    @cached_property
    def end_value(self):
        """End value of each drawdown."""
        return self.map(nb.dd_end_value_map_nb, self.ts.vbt.to_2d_array())

    @cached_property
    def drawdown(self):
        """Drawdown value (in percentage)."""
        return self.map(nb.dd_drawdown_map_nb, self.ts.vbt.to_2d_array())

    @cached_method
    def avg_drawdown(self, default_val=0., **kwargs):
        """Average drawdown (ADD)."""
        return self.drawdown.mean(default_val=default_val, **kwargs)

    @cached_method
    def max_drawdown(self, default_val=0., **kwargs):
        """Maximum drawdown (MDD)."""
        return self.drawdown.min(default_val=default_val, **kwargs)

    @cached_property
    def duration(self):
        """Duration of each drawdown (in raw format)."""
        return self.map(nb.dd_duration_map_nb)

    @cached_method
    def avg_duration(self, time_units=True, **kwargs):
        """Average drawdown duration (in time units)."""
        return self.duration.mean(time_units=time_units, **kwargs)

    @cached_method
    def max_duration(self, time_units=True, **kwargs):
        """Maximum drawdown duration (in time units)."""
        return self.duration.max(time_units=time_units, **kwargs)

    @cached_method
    def coverage(self, group_by=None, columns=None, **kwargs):
        """Coverage, that is, total duration divided by the whole period."""
        total_duration = to_1d(self.duration.sum(group_by=group_by), raw=True)
        total_steps = self.grouper.get_group_counts(group_by=group_by) * self.wrapper.shape[0]
        columns = self.grouper.get_new_index(group_by=group_by, new_index=columns)
        return self.wrapper.wrap_reduced(total_duration / total_steps, columns=columns, **kwargs)

    @cached_property
    def ptv_duration(self):
        """Peak-to-valley (PtV) duration of each drawdown."""
        return self.map(nb.dd_ptv_duration_map_nb)


class ActiveDrawdowns(BaseDrawdowns):
    """Extends `BaseDrawdowns` by properties for active drawdowns."""

    @cached_method
    def current_drawdown(self, group_by=None, **kwargs):
        """Current drawdown from peak.

        Does not support `group_by`."""
        curr_end_val = self.end_value.nst(-1, group_by=group_by)
        curr_start_val = self.start_value.nst(-1, group_by=group_by)
        curr_drawdown = (curr_end_val - curr_start_val) / curr_start_val
        return self.wrapper.wrap_reduced(curr_drawdown, **kwargs)

    @cached_method
    def current_duration(self, time_units=True, **kwargs):
        """Current duration from peak.

        Does not support `group_by`."""
        return self.duration.nst(-1, time_units=time_units, **kwargs)

    @cached_method
    def current_return(self, **kwargs):
        """Current return from valley.

        Does not support `group_by`."""
        recovery_return = self.map(nb.dd_recovery_return_map_nb, self.ts.vbt.to_2d_array())
        return recovery_return.nst(-1, **kwargs)


class RecoveredDrawdowns(BaseDrawdowns):
    """Extends `BaseDrawdowns` by properties for recovered drawdowns."""

    @cached_property
    def recovery_return(self):
        """Recovery return of each drawdown."""
        return self.map(nb.dd_recovery_return_map_nb, self.ts.vbt.to_2d_array())

    @cached_property
    def vtr_duration(self):
        """Valley-to-recovery (VtR) duration of each drawdown."""
        return self.map(nb.dd_vtr_duration_map_nb)

    @cached_property
    def vtr_duration_ratio(self):
        """Ratio of VtR duration to total duration of each drawdown.

        The time from valley to recovery divided by the time from peak to valley."""
        return self.map(nb.dd_vtr_duration_ratio_map_nb)


class Drawdowns(BaseDrawdowns):
    """Extends `BaseDrawdowns` by further dividing drawdowns into active and recovered.

    Example:
        Compare the average duration of active and recovered drawdowns:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> ts = pd.DataFrame([
        ...     [1, 1, 1],
        ...     [2, 2, 2],
        ...     [3, 3, 1],
        ...     [2, 2, 3],
        ...     [1, 3, 1]
        ... ], columns=['a', 'b', 'c'])
        >>> drawdowns = vbt.Drawdowns.from_ts(ts, freq='1 days')

        >>> drawdowns.records
           col  start_idx  valley_idx  end_idx  status
        0    0          2           4        4       0
        1    1          2           3        4       1
        2    2          1           2        3       1
        3    2          3           4        4       0
        >>> drawdowns.active.avg_duration()
        a   2 days
        b      NaT
        c   1 days
        dtype: timedelta64[ns]
        >>> drawdowns.recovered.avg_duration()
        a      NaT
        b   2 days
        c   2 days
        dtype: timedelta64[ns]
        ```"""

    @cached_property
    def status(self):
        """See `vectorbt.records.enums.DrawdownStatus`."""
        return self.map_field('status')

    @cached_method
    def recovered_rate(self, group_by=None, columns=None, **kwargs):
        """Rate of recovered drawdowns."""
        recovered_count = to_1d(self.recovered.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        columns = self.grouper.get_new_index(group_by=group_by, new_index=columns)
        return self.wrapper.wrap_reduced(recovered_count / total_count, columns=columns, **kwargs)

    @cached_property
    def active(self):
        """Active drawdowns of type `BaseDrawdowns`."""
        filter_mask = self.records_arr['status'] == DrawdownStatus.Active
        return ActiveDrawdowns(
            self.records_arr[filter_mask],
            self.ts,
            freq=self.wrapper.freq,
            idx_field=self.idx_field,
            group_by=self.grouper.group_by
        )

    @cached_property
    def recovered(self):
        """Recovered drawdowns of type `RecoveredDrawdowns`."""
        filter_mask = self.records_arr['status'] == DrawdownStatus.Recovered
        return RecoveredDrawdowns(
            self.records_arr[filter_mask],
            self.ts,
            freq=self.wrapper.freq,
            idx_field=self.idx_field,
            group_by=self.grouper.group_by
        )
