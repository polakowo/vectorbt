"""Base class for working with drawdown records."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import merge_kwargs
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.datetime import DatetimeTypes
from vectorbt.utils.enum import to_value_map
from vectorbt.base.reshape_fns import to_1d
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic import nb
from vectorbt.generic.enums import DrawdownStatus, drawdown_dt
from vectorbt.records.base import Records, indexing_on_records_meta


def drawdowns_indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Drawdowns`."""
    new_wrapper, new_records_arr, _, col_idxs = indexing_on_records_meta(obj, pd_indexing_func)
    new_ts = new_wrapper.wrap(obj.ts.values[:, col_idxs], group_by=False)
    return obj.copy(
        wrapper=new_wrapper,
        records_arr=new_records_arr,
        ts=new_ts
    )


class Drawdowns(Records):
    """Extends `Records` for working with drawdown records.

    Requires `records_arr` to have all fields defined in `vectorbt.generic.enums.drawdown_dt`.

    Example:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import numpy as np
        >>> import pandas as pd
        >>> import yfinance as yf
        >>> from datetime import datetime

        >>> start = datetime(2019, 1, 1)
        >>> end = datetime(2020, 1, 1)
        >>> price = yf.Ticker("BTC-USD").history(start=start, end=end)['Close']
        >>> drawdowns = vbt.Drawdowns.from_ts(price, freq='1 days')

        >>> drawdowns.records.head()
           col  start_idx  valley_idx  end_idx  status
        0    0          2           3        6       1
        1    0          6          38       54       1
        2    0         54          63       91       1
        3    0         93          94       95       1
        4    0         98          99      100       1

        >>> drawdowns.plot()
        ```

        ![](/vectorbt/docs/img/drawdowns_plot.png)

        ```python-repl
        >>> drawdowns.drawdown
        <vectorbt.records.base.MappedArray at 0x7fafa6a11160>

        >>> drawdowns.drawdown.min()
        -0.48982769972565016

        >>> drawdowns.drawdown.hist(trace_kwargs=dict(nbinsx=50))
        ```

        ![](/vectorbt/docs/img/drawdowns_drawdown_hist.png)"""

    def __init__(self, wrapper, records_arr, ts, idx_field='end_idx', **kwargs):
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field,
            ts=ts,
            **kwargs
        )
        self._ts = ts

        if not all(field in records_arr.dtype.names for field in drawdown_dt.names):
            raise ValueError("Records array must have all fields defined in drawdown_dt")

        PandasIndexer.__init__(self, drawdowns_indexing_func)

    @classmethod
    def from_ts(cls, ts, idx_field='end_idx', **kwargs):
        """Build `Drawdowns` from time series `ts`.

        `**kwargs` such as `freq` will be passed to `Drawdowns.__init__`."""
        records_arr = nb.find_drawdowns_nb(ts.vbt.to_2d_array())
        wrapper = ArrayWrapper.from_obj(ts, **kwargs)
        return cls(wrapper, records_arr, ts, idx_field=idx_field)

    @property
    def ts(self):
        """Original time series that records are built from."""
        return self._ts

    @property  # no need for cached
    def records_readable(self):
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame()
        out['Column'] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        out['Start Date'] = records_df['start_idx'].map(lambda x: self.wrapper.index[x])
        out['Valley Date'] = records_df['valley_idx'].map(lambda x: self.wrapper.index[x])
        out['End Date'] = records_df['end_idx'].map(lambda x: self.wrapper.index[x])
        out['Status'] = records_df['status'].map(to_value_map(DrawdownStatus))
        return out

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
    def coverage(self, group_by=None, **kwargs):
        """Coverage, that is, total duration divided by the whole period."""
        total_duration = to_1d(self.duration.sum(group_by=group_by), raw=True)
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        return self.wrapper.wrap_reduced(total_duration / total_steps, group_by=group_by, **kwargs)

    @cached_property
    def ptv_duration(self):
        """Peak-to-valley (PtV) duration of each drawdown."""
        return self.map(nb.dd_ptv_duration_map_nb)

    # ############# DrawdownStatus.Active ############# #

    @cached_method
    def current_drawdown(self, group_by=None, **kwargs):
        """Current drawdown from peak.

        Does not support `group_by`."""
        curr_end_val = self.end_value.nst(-1, group_by=group_by)
        curr_start_val = self.start_value.nst(-1, group_by=group_by)
        curr_drawdown = (curr_end_val - curr_start_val) / curr_start_val
        return self.wrapper.wrap_reduced(curr_drawdown, group_by=group_by, **kwargs)

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

    # ############# DrawdownStatus.Recovered ############# #

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

    # ############# DrawdownStatus ############# #

    @cached_property
    def status(self):
        """See `vectorbt.generic.enums.DrawdownStatus`."""
        return self.map_field('status')

    @cached_property
    def active(self):
        """Active drawdowns."""
        filter_mask = self.records_arr['status'] == DrawdownStatus.Active
        return self.filter_by_mask(filter_mask)

    @cached_method
    def active_rate(self, group_by=None, **kwargs):
        """Rate of recovered drawdowns."""
        active_count = to_1d(self.active.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(active_count / total_count, group_by=group_by, **kwargs)

    @cached_property
    def recovered(self):
        """Recovered drawdowns."""
        filter_mask = self.records_arr['status'] == DrawdownStatus.Recovered
        return self.filter_by_mask(filter_mask)

    @cached_method
    def recovered_rate(self, group_by=None, **kwargs):
        """Rate of recovered drawdowns."""
        recovered_count = to_1d(self.recovered.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        return self.wrapper.wrap_reduced(recovered_count / total_count, group_by=group_by, **kwargs)

    # ############# Plotting ############# #

    def plot(self,
             column=None,
             ts_trace_kwargs=None,
             peak_trace_kwargs=None,
             valley_trace_kwargs=None,
             recovery_trace_kwargs=None,
             active_trace_kwargs=None,
             ptv_shape_kwargs=None,
             vtr_shape_kwargs=None,
             active_shape_kwargs=None,
             fig=None,
             **layout_kwargs):  # pragma: no cover
        """Plot drawdowns over `Drawdowns.ts`.

        Args:
            column (str): Name of the column to plot.
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
            ```python-repl
            >>> import vectorbt as vbt
            >>> import pandas as pd

            >>> ts = pd.Series([1, 2, 1, 2, 3, 2, 1, 2])
            >>> vbt.Drawdowns.from_ts(ts, freq='1 days').plot()
            ```

            ![](/vectorbt/docs/img/drawdowns.png)"""
        from vectorbt.defaults import contrast_color_schema

        if column is not None:
            if self.wrapper.grouper.group_by is None:
                self_col = self[column]
            else:
                self_col = self.copy(wrapper=self.wrapper.copy(group_by=None))[column]
        else:
            self_col = self
        if self_col.wrapper.ndim > 1:
            raise TypeError("Select a column first. Use indexing or column argument.")

        if ts_trace_kwargs is None:
            ts_trace_kwargs = {}
        if peak_trace_kwargs is None:
            peak_trace_kwargs = {}
        if valley_trace_kwargs is None:
            valley_trace_kwargs = {}
        if recovery_trace_kwargs is None:
            recovery_trace_kwargs = {}
        if active_trace_kwargs is None:
            active_trace_kwargs = {}
        if ptv_shape_kwargs is None:
            ptv_shape_kwargs = {}
        if vtr_shape_kwargs is None:
            vtr_shape_kwargs = {}
        if active_shape_kwargs is None:
            active_shape_kwargs = {}

        fig = self_col.ts.vbt.plot(trace_kwargs=ts_trace_kwargs, fig=fig, **layout_kwargs)
        if len(self_col.records_arr) == 0:
            return fig

        # Extract information
        start_idx = self_col.records_arr['start_idx']
        valley_idx = self_col.records_arr['valley_idx']
        end_idx = self_col.records_arr['end_idx']
        status = self_col.records_arr['status']

        start_val = self_col.ts.values[start_idx]
        valley_val = self_col.ts.values[valley_idx]
        end_val = self_col.ts.values[end_idx]

        def get_duration_str(from_idx, to_idx):
            if isinstance(self_col.wrapper.index, DatetimeTypes):
                duration = self_col.wrapper.index[to_idx] - self_col.wrapper.index[from_idx]
            elif self_col.wrapper.freq is not None:
                duration = self_col.wrapper.to_time_units(to_idx - from_idx)
            else:
                duration = to_idx - from_idx
            return np.vectorize(str)(duration)

        # Plot peak markers and zones
        peak_mask = start_idx != np.roll(end_idx, 1)  # peak and recovery at same time -> recovery wins
        if np.any(peak_mask):
            peak_scatter = go.Scatter(
                x=self_col.ts.index[start_idx[peak_mask]],
                y=start_val[peak_mask],
                mode='markers',
                marker=dict(
                    symbol='diamond',
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
                x=self_col.ts.index[valley_idx[recovery_mask]],
                y=valley_val[recovery_mask],
                mode='markers',
                marker=dict(
                    symbol='diamond',
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
                    x0=self_col.ts.index[start_idx[i]],
                    y0=0,
                    x1=self_col.ts.index[valley_idx[i]],
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
                x=self_col.ts.index[end_idx[recovery_mask]],
                y=end_val[recovery_mask],
                mode='markers',
                marker=dict(
                    symbol='diamond',
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
                    x0=self_col.ts.index[valley_idx[i]],
                    y0=0,
                    x1=self_col.ts.index[end_idx[i]],
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
                x=self_col.ts.index[end_idx[active_mask]],
                y=end_val[active_mask],
                mode='markers',
                marker=dict(
                    symbol='diamond',
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
                    x0=self_col.ts.index[start_idx[i]],
                    y0=0,
                    x1=self_col.ts.index[end_idx[i]],
                    y1=1,
                    fillcolor='orange',
                    opacity=0.15,
                    layer="below",
                    line_width=0,
                ), active_shape_kwargs))

        return fig
