"""Base class for working with drawdown records.

Class `Drawdowns` accepts drawdown records and the corresponding time series
to analyze the periods of drawdown. Using `Drawdowns.from_ts`, you can generate
drawdown records for any time series and analyze them right away.

Moreover, all time series accessors have a method `drawdowns`:

```python-repl
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.generic.accessors.GenericAccessor.drawdowns.current_drawdown
>>> pd.Series([5, 4, 3, 4]).vbt.drawdowns.current_drawdown()
-0.2
```
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt import _typing as tp
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import merge_dicts
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.datetime import DatetimeIndexes
from vectorbt.utils.enum import enum_to_value_map
from vectorbt.utils.figure import make_figure, get_domain
from vectorbt.base.reshape_fns import to_1d, broadcast_to, to_pd_array
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic import nb
from vectorbt.generic.enums import DrawdownStatus, drawdown_dt
from vectorbt.records.base import Records
from vectorbt.records.mapped_array import MappedArray


DrawdownsT = tp.TypeVar("DrawdownsT", bound="Drawdowns")


class Drawdowns(Records):
    """Extends `Records` for working with drawdown records.

    Requires `records_arr` to have all fields defined in `vectorbt.generic.enums.drawdown_dt`.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt
    >>> import numpy as np
    >>> import pandas as pd
    >>> from datetime import datetime

    >>> start = '2019-01-01 UTC'  # crypto is in UTC
    >>> end = '2020-01-01 UTC'
    >>> price = vbt.YFData.download('BTC-USD', start=start, end=end).get('Close')
    >>> drawdowns = vbt.Drawdowns.from_ts(price, freq='1 days')

    >>> drawdowns.records.head()
       id  col  start_idx  valley_idx  end_idx  status
    0   0    0          1           2        5       1
    1   1    0          5          37       53       1
    2   2    0         53          62       90       1
    3   3    0         92          93       94       1
    4   4    0         97          98       99       1

    >>> drawdowns.drawdown
    <vectorbt.records.base.MappedArray at 0x7fafa6a11160>

    >>> drawdowns.drawdown.min()
    -0.48982813000684766

    >>> drawdowns.drawdown.histplot(trace_kwargs=dict(nbinsx=50))
    ```

    ![](/vectorbt/docs/img/drawdowns_drawdown_histplot.svg)
    """

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 ts: tp.ArrayLike,
                 idx_field: str = 'end_idx',
                 **kwargs) -> None:
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field,
            ts=ts,
            **kwargs
        )
        self._ts = broadcast_to(ts, wrapper.dummy(group_by=False))

        if not all(field in records_arr.dtype.names for field in drawdown_dt.names):
            raise TypeError("Records array must match drawdown_dt")

    def indexing_func(self: DrawdownsT, pd_indexing_func: tp.PandasIndexingFunc, **kwargs) -> DrawdownsT:
        """Perform indexing on `Drawdowns`."""
        new_wrapper, new_records_arr, _, col_idxs = \
            Records.indexing_func_meta(self, pd_indexing_func, **kwargs)
        new_ts = new_wrapper.wrap(self.ts.values[:, col_idxs], group_by=False)
        return self.copy(
            wrapper=new_wrapper,
            records_arr=new_records_arr,
            ts=new_ts
        )

    @classmethod
    def from_ts(cls: tp.Type[DrawdownsT], ts: tp.ArrayLike, idx_field: str = 'end_idx', **kwargs) -> DrawdownsT:
        """Build `Drawdowns` from time series `ts`.

        `**kwargs` such as `freq` will be passed to `Drawdowns.__init__`."""
        pd_ts = to_pd_array(ts)
        records_arr = nb.find_drawdowns_nb(pd_ts.vbt.to_2d_array())
        wrapper = ArrayWrapper.from_obj(pd_ts, **kwargs)
        return cls(wrapper, records_arr, pd_ts, idx_field=idx_field)

    @property
    def ts(self) -> tp.SeriesFrame:
        """Original time series that records are built from."""
        return self._ts

    @property  # no need for cached
    def records_readable(self) -> tp.Frame:
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame()
        out['Drawdown Id'] = records_df['id']
        out['Column'] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        out['Start Date'] = records_df['start_idx'].map(lambda x: self.wrapper.index[x])
        out['Valley Date'] = records_df['valley_idx'].map(lambda x: self.wrapper.index[x])
        out['End Date'] = records_df['end_idx'].map(lambda x: self.wrapper.index[x])
        out['Status'] = records_df['status'].map(enum_to_value_map(DrawdownStatus))
        return out

    @cached_property
    def start_value(self) -> MappedArray:
        """Start value of each drawdown."""
        return self.map(nb.dd_start_value_map_nb, self.ts.vbt.to_2d_array())

    @cached_property
    def valley_value(self) -> MappedArray:
        """Valley value of each drawdown."""
        return self.map(nb.dd_valley_value_map_nb, self.ts.vbt.to_2d_array())

    @cached_property
    def end_value(self) -> MappedArray:
        """End value of each drawdown."""
        return self.map(nb.dd_end_value_map_nb, self.ts.vbt.to_2d_array())

    @cached_property
    def drawdown(self) -> MappedArray:
        """Drawdown value (in percentage)."""
        return self.map(nb.dd_drawdown_map_nb, self.ts.vbt.to_2d_array())

    @cached_method
    def avg_drawdown(self, default_val: float = 0., **kwargs) -> tp.MaybeSeries:
        """Average drawdown (ADD)."""
        kwargs = merge_dicts(dict(wrap_kwargs=dict(name_or_index='avg_drawdown')), kwargs)
        return self.drawdown.mean(default_val=default_val, **kwargs)

    @cached_method
    def max_drawdown(self, default_val: float = 0., **kwargs) -> tp.MaybeSeries:
        """Maximum drawdown (MDD)."""
        kwargs = merge_dicts(dict(wrap_kwargs=dict(name_or_index='max_drawdown')), kwargs)
        return self.drawdown.min(default_val=default_val, **kwargs)

    @cached_property
    def duration(self) -> MappedArray:
        """Duration of each drawdown (in raw format)."""
        return self.map(nb.dd_duration_map_nb)

    @cached_method
    def avg_duration(self, **kwargs) -> tp.MaybeSeries:
        """Average drawdown duration (in time units)."""
        kwargs = merge_dicts(dict(wrap_kwargs=dict(time_units=True, name_or_index='avg_duration')), kwargs)
        return self.duration.mean(**kwargs)

    @cached_method
    def max_duration(self, **kwargs) -> tp.MaybeSeries:
        """Maximum drawdown duration (in time units)."""
        kwargs = merge_dicts(dict(wrap_kwargs=dict(time_units=True, name_or_index='max_duration')), kwargs)
        return self.duration.max(**kwargs)

    @cached_method
    def coverage(self, group_by: tp.GroupByLike = None,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Coverage, that is, total duration divided by the whole period."""
        total_duration = to_1d(self.duration.sum(group_by=group_by), raw=True)
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        wrap_kwargs = merge_dicts(dict(name_or_index='coverage'), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_duration / total_steps, group_by=group_by, **wrap_kwargs)

    @cached_property
    def ptv_duration(self) -> MappedArray:
        """Peak-to-valley (PtV) duration of each drawdown."""
        return self.map(nb.dd_ptv_duration_map_nb)

    # ############# DrawdownStatus ############# #

    @cached_property
    def status(self) -> MappedArray:
        """See `vectorbt.generic.enums.DrawdownStatus`."""
        return self.map_field('status')

    @cached_property
    def active(self: DrawdownsT) -> DrawdownsT:
        """Active drawdowns."""
        filter_mask = self.values['status'] == DrawdownStatus.Active
        return self.filter_by_mask(filter_mask)

    @cached_method
    def active_rate(self, group_by: tp.GroupByLike = None,
                    wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of recovered drawdowns."""
        active_count = to_1d(self.active.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='active_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(active_count / total_count, group_by=group_by, **wrap_kwargs)

    @cached_property
    def recovered(self: DrawdownsT) -> DrawdownsT:
        """Recovered drawdowns."""
        filter_mask = self.values['status'] == DrawdownStatus.Recovered
        return self.filter_by_mask(filter_mask)

    @cached_method
    def recovered_rate(self, group_by: tp.GroupByLike = None,
                       wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of recovered drawdowns."""
        recovered_count = to_1d(self.recovered.count(group_by=group_by), raw=True)
        total_count = to_1d(self.count(group_by=group_by), raw=True)
        wrap_kwargs = merge_dicts(dict(name_or_index='recovered_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(recovered_count / total_count, group_by=group_by, **wrap_kwargs)

    # ############# DrawdownStatus.Active ############# #

    @cached_method
    def current_drawdown(self, group_by: tp.GroupByLike = None,
                         wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Current drawdown from peak.

        Does not support grouping."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        curr_end_val = self.active.end_value.nst(-1, group_by=group_by)
        curr_start_val = self.active.start_value.nst(-1, group_by=group_by)
        curr_drawdown = (curr_end_val - curr_start_val) / curr_start_val
        wrap_kwargs = merge_dicts(dict(name_or_index='current_drawdown'), wrap_kwargs)
        return self.wrapper.wrap_reduced(curr_drawdown, group_by=group_by, **wrap_kwargs)

    @cached_method
    def current_duration(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """Current duration from peak.

        Does not support grouping."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        kwargs = merge_dicts(dict(wrap_kwargs=dict(time_units=True, name_or_index='current_duration')), kwargs)
        return self.active.duration.nst(-1, group_by=group_by, **kwargs)

    @cached_method
    def current_return(self, group_by: tp.GroupByLike = None, **kwargs) -> tp.MaybeSeries:
        """Current return from valley.

        Does not support grouping."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        recovery_return = self.active.map(nb.dd_recovery_return_map_nb, self.ts.vbt.to_2d_array())
        kwargs = merge_dicts(dict(wrap_kwargs=dict(name_or_index='current_return')), kwargs)
        return recovery_return.nst(-1, group_by=group_by, **kwargs)

    # ############# DrawdownStatus.Recovered ############# #

    @cached_property
    def recovery_return(self) -> MappedArray:
        """Recovery return of each drawdown."""
        return self.recovered.map(nb.dd_recovery_return_map_nb, self.ts.vbt.to_2d_array())

    @cached_property
    def vtr_duration(self) -> MappedArray:
        """Valley-to-recovery (VtR) duration of each drawdown."""
        return self.recovered.map(nb.dd_vtr_duration_map_nb)

    @cached_property
    def vtr_duration_ratio(self) -> MappedArray:
        """Ratio of VtR duration to total duration of each drawdown.

        The time from valley to recovery divided by the time from peak to valley."""
        return self.recovered.map(nb.dd_vtr_duration_ratio_map_nb)

    # ############# Plotting ############# #

    def plot(self,
             column: tp.Optional[tp.Label] = None,
             top_n: int = 5,
             plot_ts: bool = True,
             plot_zones: bool = True,
             ts_trace_kwargs: tp.KwargsLike = None,
             peak_trace_kwargs: tp.KwargsLike = None,
             valley_trace_kwargs: tp.KwargsLike = None,
             recovery_trace_kwargs: tp.KwargsLike = None,
             active_trace_kwargs: tp.KwargsLike = None,
             ptv_shape_kwargs: tp.KwargsLike = None,
             vtr_shape_kwargs: tp.KwargsLike = None,
             active_shape_kwargs: tp.KwargsLike = None,
             add_trace_kwargs: tp.KwargsLike = None,
             xref: str = 'x',
             yref: str = 'y',
             fig: tp.Optional[tp.BaseFigure] = None,
             **layout_kwargs) -> tp.BaseFigure:  # pragma: no cover
        """Plot drawdowns over `Drawdowns.ts`.

        Args:
            column (str): Name of the column to plot.
            top_n (int): Filter top N drawdown records by maximum drawdown.
            plot_ts (bool): Whether to plot time series.
            plot_zones (bool): Whether to plot zones.
            ts_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for time series.
            peak_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for peak values.
            valley_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for valley values.
            recovery_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for recovery values.
            active_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for active recovery values.
            ptv_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for PtV zones.
            vtr_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for VtR zones.
            active_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for active VtR zones.
            add_trace_kwargs (dict): Keyword arguments passed to `add_trace`.
            xref (str): X coordinate axis.
            yref (str): Y coordinate axis.
            fig (Figure or FigureWidget): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout.

        ## Example

        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd

        >>> ts = pd.Series([1, 2, 1, 2, 3, 2, 1, 2])
        >>> vbt.Drawdowns.from_ts(ts, freq='1 days').plot()
        ```

        ![](/vectorbt/docs/img/drawdowns_plot.svg)
        """
        from vectorbt._settings import settings
        plotting_cfg = settings['plotting']

        self_col = self.select_one(column=column, group_by=False)
        if top_n is not None:
            # Drawdowns is negative, thus top_n becomes bottom_n
            self_col = self_col.filter_by_mask(self_col.drawdown.bottom_n_mask(top_n))

        if ts_trace_kwargs is None:
            ts_trace_kwargs = {}
        ts_trace_kwargs = merge_dicts(dict(
            line=dict(
                color=plotting_cfg['color_schema']['blue']
            )
        ), ts_trace_kwargs)
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
        if add_trace_kwargs is None:
            add_trace_kwargs = {}

        if fig is None:
            fig = make_figure()
        fig.update_layout(**layout_kwargs)
        y_domain = get_domain(yref, fig)

        if plot_ts:
            fig = self_col.ts.vbt.plot(trace_kwargs=ts_trace_kwargs, add_trace_kwargs=add_trace_kwargs, fig=fig)

        if len(self_col.values) > 0:
            # Extract information
            _id = self_col.values['id']
            start_idx = self_col.values['start_idx']
            valley_idx = self_col.values['valley_idx']
            end_idx = self_col.values['end_idx']
            status = self_col.values['status']

            start_val = self_col.ts.values[start_idx]
            valley_val = self_col.ts.values[valley_idx]
            end_val = self_col.ts.values[end_idx]

            def get_duration_str(from_idx, to_idx):
                if isinstance(self_col.wrapper.index, DatetimeIndexes):
                    duration = self_col.wrapper.index[to_idx] - self_col.wrapper.index[from_idx]
                elif self_col.wrapper.freq is not None:
                    duration = self_col.wrapper.to_time_units(to_idx - from_idx)
                else:
                    duration = to_idx - from_idx
                return np.vectorize(str)(duration)

            # Plot peak markers
            peak_mask = start_idx != np.roll(end_idx, 1)  # peak and recovery at same time -> recovery wins
            if np.any(peak_mask):
                peak_customdata = _id[peak_mask][:, None]
                peak_scatter = go.Scatter(
                    x=self_col.ts.index[start_idx[peak_mask]],
                    y=start_val[peak_mask],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color=plotting_cfg['contrast_color_schema']['blue'],
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['blue'])
                        )
                    ),
                    name='Peak',
                    customdata=peak_customdata,
                    hovertemplate="Drawdown Id: %{customdata[0]}"
                                  "<br>Date: %{x}"
                                  "<br>Price: %{y}"
                )
                peak_scatter.update(**peak_trace_kwargs)
                fig.add_trace(peak_scatter, **add_trace_kwargs)

            recovery_mask = status == DrawdownStatus.Recovered
            if np.any(recovery_mask):
                # Plot valley markers
                valley_drawdown = (valley_val[recovery_mask] - start_val[recovery_mask]) / start_val[recovery_mask]
                valley_duration = get_duration_str(start_idx[recovery_mask], valley_idx[recovery_mask])
                valley_customdata = np.stack((_id[recovery_mask], valley_drawdown, valley_duration), axis=1)
                valley_scatter = go.Scatter(
                    x=self_col.ts.index[valley_idx[recovery_mask]],
                    y=valley_val[recovery_mask],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color=plotting_cfg['contrast_color_schema']['red'],
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['red'])
                        )
                    ),
                    name='Valley',
                    customdata=valley_customdata,
                    hovertemplate="Drawdown Id: %{customdata[0]}"
                                  "<br>Date: %{x}"
                                  "<br>Price: %{y}"
                                  "<br>Drawdown: %{customdata[1]:.2%}"
                                  "<br>Duration: %{customdata[2]}"
                )
                valley_scatter.update(**valley_trace_kwargs)
                fig.add_trace(valley_scatter, **add_trace_kwargs)

                if plot_zones:
                    # Plot drawdown zones
                    for i in np.flatnonzero(recovery_mask):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref="paper",
                            x0=self_col.ts.index[start_idx[i]],
                            y0=y_domain[0],
                            x1=self_col.ts.index[valley_idx[i]],
                            y1=y_domain[1],
                            fillcolor='red',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), ptv_shape_kwargs))

                # Plot recovery markers
                recovery_return = (end_val[recovery_mask] - valley_val[recovery_mask]) / valley_val[recovery_mask]
                recovery_duration = get_duration_str(valley_idx[recovery_mask], end_idx[recovery_mask])
                recovery_customdata = np.stack((_id[recovery_mask], recovery_return, recovery_duration), axis=1)
                recovery_scatter = go.Scatter(
                    x=self_col.ts.index[end_idx[recovery_mask]],
                    y=end_val[recovery_mask],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color=plotting_cfg['contrast_color_schema']['green'],
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['green'])
                        )
                    ),
                    name='Recovery/Peak',
                    customdata=recovery_customdata,
                    hovertemplate="Drawdown Id: %{customdata[0]}"
                                  "<br>Date: %{x}"
                                  "<br>Price: %{y}"
                                  "<br>Return: %{customdata[1]:.2%}"
                                  "<br>Duration: %{customdata[2]}"
                )
                recovery_scatter.update(**recovery_trace_kwargs)
                fig.add_trace(recovery_scatter, **add_trace_kwargs)

                if plot_zones:
                    # Plot recovery zones
                    for i in np.flatnonzero(recovery_mask):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref="paper",
                            x0=self_col.ts.index[valley_idx[i]],
                            y0=y_domain[0],
                            x1=self_col.ts.index[end_idx[i]],
                            y1=y_domain[1],
                            fillcolor='green',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), vtr_shape_kwargs))

            # Plot active markers
            active_mask = ~recovery_mask
            if np.any(active_mask):
                active_drawdown = (valley_val[active_mask] - start_val[active_mask]) / start_val[active_mask]
                active_duration = get_duration_str(valley_idx[active_mask], end_idx[active_mask])
                active_customdata = np.stack((_id[active_mask], active_drawdown, active_duration), axis=1)
                active_scatter = go.Scatter(
                    x=self_col.ts.index[end_idx[active_mask]],
                    y=end_val[active_mask],
                    mode='markers',
                    marker=dict(
                        symbol='diamond',
                        color=plotting_cfg['contrast_color_schema']['orange'],
                        size=7,
                        line=dict(
                            width=1,
                            color=adjust_lightness(plotting_cfg['contrast_color_schema']['orange'])
                        )
                    ),
                    name='Active',
                    customdata=active_customdata,
                    hovertemplate="Drawdown Id: %{customdata[0]}"
                                  "<br>Date: %{x}"
                                  "<br>Price: %{y}"
                                  "<br>Drawdown: %{customdata[1]:.2%}"
                                  "<br>Duration: %{customdata[2]}"
                )
                active_scatter.update(**active_trace_kwargs)
                fig.add_trace(active_scatter, **add_trace_kwargs)

                if plot_zones:
                    # Plot active drawdown zones
                    for i in np.flatnonzero(active_mask):
                        fig.add_shape(**merge_dicts(dict(
                            type="rect",
                            xref=xref,
                            yref="paper",
                            x0=self_col.ts.index[start_idx[i]],
                            y0=y_domain[0],
                            x1=self_col.ts.index[end_idx[i]],
                            y1=y_domain[1],
                            fillcolor='orange',
                            opacity=0.2,
                            layer="below",
                            line_width=0,
                        ), active_shape_kwargs))

        return fig
