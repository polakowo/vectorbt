"""Base class for working with range records.

Range records capture information on ranges. They are useful for analyzing duration of processes,
such as drawdowns, trades, and positions. They also come in handy when analyzing distance between events,
such as entry and exit signals.

Each range has a starting point and an ending point. For example, the points for `range(20)`
are 0 and 20 (not 19!) respectively.

!!! note
    Be aware that if a range hasn't ended in a column, its `end_idx` will point at the latest index.
    Make sure to account for this when computing custom metrics involving duration.

```python-repl
>>> import vectorbt as vbt
>>> import numpy as np
>>> import pandas as pd

>>> start = '2019-01-01 UTC'  # crypto is in UTC
>>> end = '2020-01-01 UTC'
>>> price = vbt.YFData.download('BTC-USD', start=start, end=end).get('Close')
>>> fast_ma = vbt.MA.run(price, 10)
>>> slow_ma = vbt.MA.run(price, 50)
>>> fast_above_slow = fast_ma.ma_above(slow_ma)

>>> ranges = vbt.Ranges.from_ts(fast_above_slow, wrapper_kwargs=dict(freq='d'))

>>> ranges.records_readable
   Range Id  Column                Start Date                  End Date
0         0       0 2019-02-19 00:00:00+00:00 2019-07-25 00:00:00+00:00
1         1       0 2019-08-08 00:00:00+00:00 2019-08-19 00:00:00+00:00
2         2       0 2019-11-01 00:00:00+00:00 2019-11-20 00:00:00+00:00

>>> ranges.duration.max(wrap_kwargs=dict(to_timedelta=True))
Timedelta('156 days 00:00:00')
```

## From accessors

Moreover, all time series accessors have a property `ranges` and a method `get_ranges`:

```python-repl
>>> # vectorbt.generic.accessors.GenericAccessor.ranges.coverage
>>> fast_above_slow.vbt.ranges.coverage()
0.5081967213114754
```

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `Ranges.metrics`.

```python-repl
>>> df = pd.DataFrame({
...     'a': [1, 2, np.nan, np.nan, 5, 6],
...     'b': [np.nan, 2, np.nan, 4, np.nan, 6]
... })
>>> ranges = df.vbt(freq='d').ranges

>>> ranges['a'].stats()
Start                             0
End                               5
Period              6 days 00:00:00
Total Records                     2
Coverage            4 days 00:00:00
Overlap Coverage    0 days 00:00:00
Duration: Min       2 days 00:00:00
Duration: Median    2 days 00:00:00
Duration: Max       2 days 00:00:00
Duration: Mean      2 days 00:00:00
Duration: Std       0 days 00:00:00
Name: a, dtype: object
```

`Ranges.stats` also supports (re-)grouping:

```python-repl
>>> ranges.stats(group_by=True)
Start                                       0
End                                         5
Period                        6 days 00:00:00
Total Records                               5
Coverage                      5 days 00:00:00
Overlap Coverage              2 days 00:00:00
Duration: Min                 1 days 00:00:00
Duration: Median              1 days 00:00:00
Duration: Max                 2 days 00:00:00
Duration: Mean                1 days 09:36:00
Duration: Std       0 days 13:08:43.228968446
Name: group, dtype: object

```
"""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils.decorators import cached_property, cached_method
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.base.reshape_fns import to_pd_array, to_2d_array
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic.enums import RangeStatus, range_dt
from vectorbt.generic.stats_builder import StatsBuilderMixin
from vectorbt.generic import nb
from vectorbt.records.base import Records
from vectorbt.records.mapped_array import MappedArray
from vectorbt.records.decorators import attach_fields

__pdoc__ = {}

ranges_fields_config = Config(
    dict(
        dtype=range_dt,
        field_settings=dict(
            start_idx=dict(
                mapping='index'
            ),
            end_idx=dict(
                mapping='index'
            ),
            status=dict(
                attach_filters=True
            )
        )
    ),
    as_attrs=False,
    readonly=True
)
"""_"""

__pdoc__['ranges_fields_config'] = f"""Config of `Ranges` fields.

```json
{ranges_fields_config.to_doc()}
```
"""

RangesT = tp.TypeVar("RangesT", bound="Ranges")


class Ranges(Records):
    """Extends `Records` for working with range records.

    Requires `records_arr` to have all fields defined in `vectorbt.generic.enums.range_dt`."""

    dtype: tp.ClassVar[np.dtype] = range_dt
    """Data type corresponding to range records."""

    idx_field: tp.ClassVar[str] = 'end_idx'
    """Name of the field that holds indices."""

    start_idx_field: tp.ClassVar[str] = 'start_idx'
    """Name of the field that holds start indices."""

    end_idx_field: tp.ClassVar[str] = 'end_idx'
    """Name of the field that holds end indices."""

    status_field: tp.ClassVar[str] = 'status'
    """Name of the field that holds statuses."""

    status_mapping: tp.ClassVar[tp.MappingLike] = RangeStatus
    """Mapping of `Ranges.status_field`."""

    @classmethod
    def from_ts(cls: tp.Type[RangesT],
                ts: tp.ArrayLike,
                gap_value: tp.Optional[tp.Scalar] = None,
                wrapper_kwargs: tp.KwargsLike = None,
                **kwargs) -> RangesT:
        """Build `Ranges` from time series `ts`.

        Searches for sequences of

        * True values in boolean data (False acts as a gap),
        * positive values in integer data (-1 acts as a gap), and
        * non-NaN values in any other data (NaN acts as a gap).

        `**kwargs` will be passed to `Ranges.__init__`."""
        if wrapper_kwargs is None:
            wrapper_kwargs = {}

        ts_arr = to_2d_array(ts)
        if gap_value is None:
            if np.issubdtype(ts_arr.dtype, np.bool_):
                gap_value = False
            elif np.issubdtype(ts_arr.dtype, np.integer):
                gap_value = -1
            else:
                gap_value = np.nan
        records_arr = nb.find_ranges_nb(ts_arr, gap_value)
        wrapper = ArrayWrapper.from_obj(to_pd_array(ts), **wrapper_kwargs)
        return cls(wrapper, records_arr, **kwargs)

    @property  # no need for cached
    def records_readable(self) -> tp.Frame:
        """Records in readable format."""
        df = self.records.copy()
        df.columns = [
            'Range Id',
            'Column',
            'Start Date',
            'End Date',
            'Status'
        ]
        df['Column'] = self.map_field(self.col_field).apply_mapping('columns')
        df['Start Date'] = self.map_field(self.start_idx_field).apply_mapping('index')
        df['End Date'] = self.map_field(self.end_idx_field).apply_mapping('index')
        df['Status'] = self.map_field(self.status_field).apply_mapping(self.status_mapping)
        return df

    @cached_property
    def duration(self) -> MappedArray:
        """Duration of each range (in raw format)."""
        duration = nb.duration_nb(
            self.values[self.start_idx_field],
            self.values[self.end_idx_field],
            self.values[self.status_field]
        )
        return self.map_array(duration)

    @cached_method
    def avg_duration(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Average range duration (as timedelta)."""
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='avg_duration'), wrap_kwargs)
        return self.duration.mean(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def max_duration(self, group_by: tp.GroupByLike = None,
                     wrap_kwargs: tp.KwargsLike = None, **kwargs) -> tp.MaybeSeries:
        """Maximum range duration (as timedelta)."""
        wrap_kwargs = merge_dicts(dict(to_timedelta=True, name_or_index='max_duration'), wrap_kwargs)
        return self.duration.max(group_by=group_by, wrap_kwargs=wrap_kwargs, **kwargs)

    @cached_method
    def coverage(self,
                 overlapping: bool = False,
                 normalize: bool = True,
                 group_by: tp.GroupByLike = None,
                 wrap_kwargs: tp.KwargsLike = None) -> tp.MaybeSeries:
        """Coverage, that is, the number of steps that are covered by all ranges.

        See `vectorbt.generic.nb.range_coverage_nb`."""
        col_map = self.col_mapper.get_col_map(group_by=group_by)
        index_lens = self.wrapper.grouper.get_group_lens(group_by=group_by) * self.wrapper.shape[0]
        coverage = nb.range_coverage_nb(
            self.values[self.start_idx_field],
            self.values[self.end_idx_field],
            self.values[self.status_field],
            col_map,
            index_lens,
            overlapping=overlapping,
            normalize=normalize
        )
        wrap_kwargs = merge_dicts(dict(name_or_index='coverage'), wrap_kwargs)
        return self.wrapper.wrap_reduced(coverage, group_by=group_by, **wrap_kwargs)

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Drawdowns.stats`.

        Merges `vectorbt.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `ranges.stats` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        ranges_stats_cfg = settings['ranges']['stats']

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            ranges_stats_cfg
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
            coverage=dict(
                title='Coverage',
                calc_func='coverage',
                normalize=False,
                apply_to_timedelta=True,
                tags=['ranges', 'coverage']
            ),
            overlap_coverage=dict(
                title='Overlap Coverage',
                calc_func='coverage',
                overlapping=True,
                normalize=False,
                apply_to_timedelta=True,
                tags=['ranges', 'coverage']
            ),
            duration=dict(
                title='Duration',
                calc_func='duration.describe',
                post_calc_func=lambda self, out, settings: {
                    'Min': out['min'],
                    'Median': out['50%'],
                    'Max': out['max'],
                    'Mean': out['mean'],
                    'Std': out['std']
                },
                apply_to_timedelta=True,
                tags=['ranges', 'duration']
            ),
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        return self._metrics


Ranges.override_metrics_doc(__pdoc__)
