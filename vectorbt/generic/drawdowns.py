"""Base class for working with drawdown records.

Class `Drawdowns` accepts drawdown records and the corresponding time series
to analyze the periods of drawdown. Using `Drawdowns.from_ts`, you can generate
drawdown records for any time series and analyze them right away.

Moreover, all time series accessors have a method `drawdowns`:

```python-repl
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.generic.accessors.GenericAccessor.drawdowns.active_drawdown
>>> pd.Series([5, 4, 3, 4]).vbt.drawdowns.active_drawdown()
-0.2
```

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `Drawdowns.metrics`.

```python-repl
>>> df = pd.DataFrame({
...     'a': [1, 2, 1, 3, 2],
...     'b': [2, 3, 1, 2, 1]
... })

>>> df['a'].vbt(freq='d').drawdowns.stats()
Start                                        0
End                                          4
Period                         5 days 00:00:00
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Active Drawdown [%]                  33.333333
Active Duration                1 days 00:00:00
Active Recovery [%]                        0.0
Active Recovery Return [%]                 0.0
Active Recovery Duration       0 days 00:00:00
Max Drawdown [%]                          50.0
Avg Drawdown [%]                          50.0
Max Drawdown Duration          2 days 00:00:00
Avg Drawdown Duration          2 days 00:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                0.5
Name: a, dtype: object
```

By default, the metrics `max_dd`, `avg_dd`, `max_dd_duration`, and `avg_dd_duration` do
not include active drawdowns. To change that, pass `incl_active=True`:

```python-repl
>>> df['a'].vbt(freq='d').drawdowns.stats(settings=dict(incl_active=True))
Start                                        0
End                                          4
Period                         5 days 00:00:00
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Active Drawdown [%]                  33.333333
Active Duration                1 days 00:00:00
Active Recovery [%]                        0.0
Active Recovery Return [%]                 0.0
Active Recovery Duration       0 days 00:00:00
Max Drawdown [%]                          50.0
Avg Drawdown [%]                     41.666667
Max Drawdown Duration          2 days 00:00:00
Avg Drawdown Duration          1 days 12:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                0.5
Name: a, dtype: object
```

`Drawdowns.stats` also supports (re-)grouping:

```python-repl
>>> df.vbt(freq='d', group_by=True).drawdowns.stats()
Start                                        0
End                                          4
Period                         5 days 00:00:00
Total Records                                3
Total Recovered Drawdowns                    1
Total Active Drawdowns                       2
Max Drawdown [%]                          50.0
Avg Drawdown [%]                          50.0
Max Drawdown Duration          2 days 00:00:00
Avg Drawdown Duration          2 days 00:00:00
Max Recovery Return [%]                  200.0
Avg Recovery Return [%]                  200.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                0.5
Name: group, dtype: object
```
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt import _typing as tp
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.utils.colors import adjust_lightness
from vectorbt.utils.datetime import DatetimeIndexes
from vectorbt.utils.enum import map_enum_values
from vectorbt.utils.figure import make_figure, get_domain
from vectorbt.utils.template import RepEval
from vectorbt.base.reshape_fns import to_1d_array, to_2d_array, broadcast_to, to_pd_array
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic import nb
from vectorbt.generic.enums import DrawdownStatus, drawdown_dt
from vectorbt.generic.stats_builder import StatsBuilderMixin
from vectorbt.records.base import Records
from vectorbt.records.mapped_array import MappedArray
from vectorbt.records.decorators import add_mapped_fields

__pdoc__ = {}

drawdowns_mf_config = Config(
    dict(
        status=dict(defaults=dict(mapping=DrawdownStatus))
    ),
    as_attrs=False,
    readonly=True
)
"""_"""

__pdoc__['drawdowns_mf_config'] = f"""Config of `vectorbt.generic.enums.drawdown_dt` 
mapped fields to be overridden in `Drawdowns`.

```json
{drawdowns_mf_config.to_doc()}
```
"""

DrawdownsT = tp.TypeVar("DrawdownsT", bound="Drawdowns")


@add_mapped_fields(drawdown_dt, drawdowns_mf_config)
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

    ![](/docs/img/drawdowns_drawdown_histplot.svg)
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
        records_arr = nb.find_drawdowns_nb(to_2d_array(pd_ts))
        wrapper = ArrayWrapper.from_obj(pd_ts, **kwargs)
        return cls(wrapper, records_arr, pd_ts, idx_field=idx_field)

    @property
    def ts(self) -> tp.SeriesFrame:
        """Original time series that records are built from."""
        return self._ts

    @property  # no need for cached
    def records_readable(self) -> tp.Frame:
        """Records in readable format."""
        df = self.records.copy()
        df.columns = [
            'Drawdown Id',
            'Column',
            'Start Date',
            'Valley Date',
            'End Date',
            'Status'
        ]
        df['Column'] = df['Column'].map(lambda x: self.wrapper.columns[x])
        df['Start Date'] = df['Start Date'].map(lambda x: self.wrapper.index[x])
        df['Valley Date'] = df['Valley Date'].map(lambda x: self.wrapper.index[x])
        df['End Date'] = df['End Date'].map(lambda x: self.wrapper.index[x])
        df['Status'] = map_enum_values(df['Status'], DrawdownStatus)
        return df

    @cached_property
    def start_value(self) -> MappedArray:
        """Start value of each drawdown."""
        return self.map(nb.dd_start_value_map_nb, to_2d_array(self.ts))

    @cached_property
    def valley_value(self) -> MappedArray:
        """Valley value of each drawdown."""
        return self.map(nb.dd_valley_value_map_nb, to_2d_array(self.ts))

    @cached_property
    def end_value(self) -> MappedArray:
        """End value of each drawdown."""
        return self.map(nb.dd_end_value_map_nb, to_2d_array(self.ts))

    @cached_property
    def drawdown(self) -> MappedArray:
        """Drawdown value (in percentage)."""
        return self.map(nb.dd_drawdown_map_nb, to_2d_array(self.ts))

    @cached_method
    def avg_drawdown(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Average drawdown (ADD)."""
        wrap_kwargs = merge_dicts(dict(name_or_index='avg_drawdown'), wrap_kwargs)
        return self.drawdown.mean(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def max_drawdown(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Maximum drawdown (MDD)."""
        wrap_kwargs = merge_dicts(dict(name_or_index='max_drawdown'), wrap_kwargs)
        return self.drawdown.min(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_property
    def duration(self) -> MappedArray:
        """Duration of each drawdown (in raw format)."""
        return self.map(nb.dd_duration_map_nb)

    @cached_method
    def avg_duration(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Average drawdown duration (in time units)."""
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='avg_duration'), wrap_kwargs)
        return self.duration.mean(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def max_duration(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Maximum drawdown duration (in time units)."""
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='max_duration'), wrap_kwargs)
        return self.duration.max(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def coverage(self, group_by: tp.GroupByLike = None,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Coverage, that is, total duration divided by the whole period."""
        total_duration = to_1d_array(self.duration.sum(group_by=group_by))
        total_steps = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        wrap_kwargs = merge_dicts(dict(name_or_index='coverage'), wrap_kwargs)
        return self.wrapper.wrap_reduced(total_duration / total_steps, group_by=group_by, **wrap_kwargs)

    @cached_property
    def decline_duration(self) -> MappedArray:
        """Decline duration of each drawdown."""
        return self.map(nb.dd_decline_duration_map_nb)

    @cached_property
    def recovery_return(self) -> MappedArray:
        """Recovery return of each drawdown."""
        return self.map(nb.dd_recovery_return_map_nb, to_2d_array(self.ts))

    @cached_property
    def recovery_duration(self) -> MappedArray:
        """Recovery duration of each drawdown."""
        return self.map(nb.dd_recovery_duration_map_nb)

    @cached_property
    def recovery_duration_ratio(self) -> MappedArray:
        """Ratio of recovery duration to total duration of each drawdown.

        The time from valley to recovery divided by the time from peak to valley."""
        return self.map(nb.dd_recovery_duration_ratio_map_nb)

    # ############# DrawdownStatus ############# #

    @cached_property
    def active(self: DrawdownsT) -> DrawdownsT:
        """Active drawdowns."""
        filter_mask = self.values['status'] == DrawdownStatus.Active
        return self.filter_by_mask(filter_mask)

    @cached_method
    def active_rate(self, group_by: tp.GroupByLike = None,
                    wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Rate of recovered drawdowns."""
        active_count = to_1d_array(self.active.count(group_by=group_by))
        total_count = to_1d_array(self.count(group_by=group_by))
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
        recovered_count = to_1d_array(self.recovered.count(group_by=group_by))
        total_count = to_1d_array(self.count(group_by=group_by))
        wrap_kwargs = merge_dicts(dict(name_or_index='recovered_rate'), wrap_kwargs)
        return self.wrapper.wrap_reduced(recovered_count / total_count, group_by=group_by, **wrap_kwargs)

    # ############# Active drawdown ############# #

    @cached_method
    def active_drawdown(self, group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Active drawdown from peak.

        Does not support grouping."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        curr_end_val = self.active.end_value.nth(-1, group_by=group_by)
        curr_start_val = self.active.start_value.nth(-1, group_by=group_by)
        curr_drawdown = (curr_end_val - curr_start_val) / curr_start_val
        wrap_kwargs = merge_dicts(dict(name_or_index='active_drawdown'), wrap_kwargs)
        return self.wrapper.wrap_reduced(curr_drawdown, group_by=group_by, **wrap_kwargs)

    @cached_method
    def active_duration(self, group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Active duration from peak.

        Does not support grouping."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='active_duration'), wrap_kwargs)
        return self.active.duration.nth(-1, group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def active_recovery(self, group_by: tp.GroupByLike = None,
                        wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Active recovery.

        Does not support grouping."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        curr_start_val = self.active.start_value.nth(-1, group_by=group_by)
        curr_end_val = self.active.end_value.nth(-1, group_by=group_by)
        curr_valley_val = self.active.valley_value.nth(-1, group_by=group_by)
        curr_recovery = (curr_end_val - curr_valley_val) / (curr_start_val - curr_valley_val)
        wrap_kwargs = merge_dicts(dict(name_or_index='active_recovery'), wrap_kwargs)
        return self.wrapper.wrap_reduced(curr_recovery, group_by=group_by, **wrap_kwargs)

    @cached_method
    def active_recovery_return(self, group_by: tp.GroupByLike = None,
                               wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Active recovery return from valley.

        Does not support grouping."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        wrap_kwargs = merge_dicts(dict(name_or_index='active_recovery_return'), wrap_kwargs)
        return self.active.recovery_return.nth(-1, group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def active_recovery_duration(self, group_by: tp.GroupByLike = None,
                                 wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Active recovery duration from valley.

        Does not support grouping."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            raise ValueError("Grouping is not supported by this method")
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='active_recovery_duration'), wrap_kwargs)
        return self.active.recovery_duration.nth(-1, group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Drawdowns.stats`.

        Merges `vectorbt.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `drawdowns.stats` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        drawdowns_stats_cfg = settings['drawdowns']['stats']

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            drawdowns_stats_cfg
        )

    _metrics: tp.ClassVar[Config] = Config(
        dict(
            start=dict(
                title='Start',
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None,
                tags='wrapper'
            ),
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None,
                tags='wrapper'
            ),
            period=dict(
                title='Period',
                calc_func=lambda self: len(self.wrapper.index),
                apply_to_timedelta=True,
                agg_func=None,
                tags='wrapper'
            ),
            total_records=dict(
                title='Total Records',
                calc_func='count',
                tags='records'
            ),
            total_recovered=dict(
                title='Total Recovered Drawdowns',
                calc_func='recovered.count',
                tags='drawdowns'
            ),
            total_active=dict(
                title='Total Active Drawdowns',
                calc_func='active.count',
                tags='drawdowns'
            ),
            active_dd=dict(
                title='Active Drawdown [%]',
                calc_func='active_drawdown',
                post_calc_func=lambda self, out, settings: -out * 100,
                check_is_not_grouped=True,
                tags=['drawdowns', 'active']
            ),
            active_duration=dict(
                title='Active Duration',
                calc_func='active_duration',
                fill_wrap_kwargs=True,
                check_is_not_grouped=True,
                tags=['drawdowns', 'active', 'duration']
            ),
            active_recovery=dict(
                title='Active Recovery [%]',
                calc_func='active_recovery',
                post_calc_func=lambda self, out, settings: out * 100,
                check_is_not_grouped=True,
                tags=['drawdowns', 'active']
            ),
            active_recovery_return=dict(
                title='Active Recovery Return [%]',
                calc_func='active_recovery_return',
                post_calc_func=lambda self, out, settings: out * 100,
                check_is_not_grouped=True,
                tags=['drawdowns', 'active']
            ),
            active_recovery_duration=dict(
                title='Active Recovery Duration',
                calc_func='active_recovery_duration',
                fill_wrap_kwargs=True,
                check_is_not_grouped=True,
                tags=['drawdowns', 'active', 'duration']
            ),
            max_dd=dict(
                title='Max Drawdown [%]',
                calc_func=RepEval("'max_drawdown' if incl_active else 'recovered.max_drawdown'"),
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=RepEval("['drawdowns'] if incl_active else ['drawdowns', 'recovered']")
            ),
            avg_dd=dict(
                title='Avg Drawdown [%]',
                calc_func=RepEval("'avg_drawdown' if incl_active else 'recovered.avg_drawdown'"),
                post_calc_func=lambda self, out, settings: -out * 100,
                tags=RepEval("['drawdowns'] if incl_active else ['drawdowns', 'recovered']")
            ),
            max_dd_duration=dict(
                title='Max Drawdown Duration',
                calc_func=RepEval("'max_duration' if incl_active else 'recovered.max_duration'"),
                fill_wrap_kwargs=True,
                tags=RepEval("['drawdowns', 'duration'] if incl_active else ['drawdowns', 'recovered', 'duration']")
            ),
            avg_dd_duration=dict(
                title='Avg Drawdown Duration',
                calc_func=RepEval("'avg_duration' if incl_active else 'recovered.avg_duration'"),
                fill_wrap_kwargs=True,
                tags=RepEval("['drawdowns', 'duration'] if incl_active else ['drawdowns', 'recovered', 'duration']")
            ),
            max_return=dict(
                title='Max Recovery Return [%]',
                calc_func='recovered.recovery_return.max',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=['drawdowns', 'recovered']
            ),
            avg_return=dict(
                title='Avg Recovery Return [%]',
                calc_func='recovered.recovery_return.mean',
                post_calc_func=lambda self, out, settings: out * 100,
                tags=['drawdowns', 'recovered']
            ),
            max_recovery_duration=dict(
                title='Max Recovery Duration',
                calc_func='recovered.recovery_duration.max',
                apply_to_timedelta=True,
                tags=['drawdowns', 'recovered', 'duration']
            ),
            avg_recovery_duration=dict(
                title='Avg Recovery Duration',
                calc_func='recovered.recovery_duration.mean',
                apply_to_timedelta=True,
                tags=['drawdowns', 'recovered', 'duration']
            ),
            recovery_duration_ratio=dict(
                title='Avg Recovery Duration Ratio',
                calc_func='recovered.recovery_duration_ratio.mean',
                tags=['drawdowns', 'recovered']
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        return self._metrics

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
             decline_shape_kwargs: tp.KwargsLike = None,
             recovery_shape_kwargs: tp.KwargsLike = None,
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
            decline_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for decline zones.
            recovery_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for recovery zones.
            active_shape_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Figure.add_shape` for active recovery zones.
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

        ![](/docs/img/drawdowns_plot.svg)
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
        if decline_shape_kwargs is None:
            decline_shape_kwargs = {}
        if recovery_shape_kwargs is None:
            recovery_shape_kwargs = {}
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
                    duration = self_col.wrapper.to_timedelta(to_idx - from_idx)
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
                        ), decline_shape_kwargs))

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
                        ), recovery_shape_kwargs))

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


Drawdowns.override_metrics_doc(__pdoc__)
