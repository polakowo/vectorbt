"""Base class for working with log records.

Class `Logs` wraps log records to analyze logs. Logs are mainly populated when
simulating a portfolio and can be accessed as `vectorbt.portfolio.base.Portfolio.logs`.

## Stats

!!! hint
    See `vectorbt.generic.stats_builder.StatsBuilderMixin.stats` and `Logs.metrics`.

```python-repl
>>> import pandas as pd
>>> import numpy as np
>>> from datetime import datetime, timedelta
>>> import vectorbt as vbt

>>> np.random.seed(42)
>>> price = pd.DataFrame({
...     'a': np.random.uniform(1, 2, size=100),
...     'b': np.random.uniform(1, 2, size=100)
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> size = pd.DataFrame({
...     'a': np.random.uniform(-100, 100, size=100),
...     'b': np.random.uniform(-100, 100, size=100),
... }, index=[datetime(2020, 1, 1) + timedelta(days=i) for i in range(100)])
>>> pf = vbt.Portfolio.from_orders(price, size, fees=0.01, freq='d', log=True)

>>> pf.logs.stats(column='a')
Start                             2020-01-01 00:00:00
End                               2020-04-09 00:00:00
Period                              100 days 00:00:00
Total Records                                     100
Status Counts: None                                 0
Status Counts: Filled                              88
Status Counts: Ignored                              0
Status Counts: Rejected                            12
Status Info Counts: None                           88
Status Info Counts: NoCashLong                     12
Name: a, dtype: object
```

`Logs.stats` also supports (re-)grouping:

```python-repl
>>> pf.logs.stats(group_by=True)
Start                             2020-01-01 00:00:00
End                               2020-04-09 00:00:00
Period                              100 days 00:00:00
Total Records                                     200
Status Counts: None                                 0
Status Counts: Filled                             187
Status Counts: Ignored                              0
Status Counts: Rejected                            13
Status Info Counts: None                          187
Status Info Counts: NoCashLong                     13
Name: group, dtype: object
```"""

import pandas as pd

from vectorbt import _typing as tp
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.utils.enum import map_enum_values
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.base.reshape_fns import to_dict
from vectorbt.generic.stats_builder import StatsBuilderMixin
from vectorbt.records.base import Records
from vectorbt.records.decorators import add_mapped_fields
from vectorbt.portfolio.enums import (
    log_dt,
    SizeType,
    Direction,
    OrderSide,
    OrderStatus,
    StatusInfo
)

__pdoc__ = {}

logs_mf_config = Config(
    dict(
        size_type=dict(defaults=dict(mapping=SizeType)),
        direction=dict(defaults=dict(mapping=Direction)),
        res_side=dict(defaults=dict(mapping=OrderSide)),
        res_status=dict(defaults=dict(mapping=OrderStatus)),
        res_status_info=dict(defaults=dict(mapping=StatusInfo))
    ),
    as_attrs=False,
    readonly=True
)
"""_"""

__pdoc__['logs_mf_config'] = f"""Config of `vectorbt.portfolio.enums.log_dt` 
mapped fields to be overridden in `Logs`.

```json
{logs_mf_config.to_doc()}
```
"""


@add_mapped_fields(log_dt, logs_mf_config)
class Logs(Records):
    """Extends `Records` for working with log records.

    !!! note
        Some features require the log records to be sorted prior to the processing.
        Use the `vectorbt.records.base.Records.sort` method."""

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 idx_field: str = 'idx',
                 **kwargs) -> None:
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field,
            **kwargs
        )

        if not all(field in records_arr.dtype.names for field in log_dt.names):
            raise TypeError("Records array must match debug_info_dt")

    @property  # no need for cached
    def records_readable(self) -> tp.Frame:
        """Records in readable format."""
        df = self.records.copy()
        df.columns = pd.MultiIndex.from_tuples([
            ('Context', 'Log Id'),
            ('Context', 'Date'),
            ('Context', 'Column'),
            ('Context', 'Group'),
            ('Context', 'Cash'),
            ('Context', 'Position'),
            ('Context', 'Debt'),
            ('Context', 'Free Cash'),
            ('Context', 'Val Price'),
            ('Context', 'Value'),
            ('Order', 'Size'),
            ('Order', 'Price'),
            ('Order', 'Size Type'),
            ('Order', 'Direction'),
            ('Order', 'Fees'),
            ('Order', 'Fixed Fees'),
            ('Order', 'Slippage'),
            ('Order', 'Min Size'),
            ('Order', 'Max Size'),
            ('Order', 'Rejection Prob'),
            ('Order', 'Lock Cash'),
            ('Order', 'Allow Partial'),
            ('Order', 'Raise Rejection'),
            ('Order', 'Log'),
            ('New Context', 'Cash'),
            ('New Context', 'Position'),
            ('New Context', 'Debt'),
            ('New Context', 'Free Cash'),
            ('New Context', 'Val Price'),
            ('New Context', 'Value'),
            ('Order Result', 'Size'),
            ('Order Result', 'Price'),
            ('Order Result', 'Fees'),
            ('Order Result', 'Side'),
            ('Order Result', 'Status'),
            ('Order Result', 'Status Info'),
            ('Order Result', 'Order Id')
        ])

        df[('Context', 'Date')] = df[('Context', 'Date')].map(lambda x: self.wrapper.index[x])
        df[('Context', 'Column')] = df[('Context', 'Column')].map(lambda x: self.wrapper.columns[x])
        df[('Order', 'Size Type')] = map_enum_values(df[('Order', 'Size Type')], SizeType)
        df[('Order', 'Direction')] = map_enum_values(df[('Order', 'Direction')], Direction)
        df[('Order Result', 'Side')] = map_enum_values(df[('Order Result', 'Side')], OrderSide)
        df[('Order Result', 'Status')] = map_enum_values(df[('Order Result', 'Status')], OrderStatus)
        df[('Order Result', 'Status Info')] = map_enum_values(df[('Order Result', 'Status Info')], StatusInfo)
        return df

    # ############# Stats ############# #

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `Orders.stats`.

        Merges `vectorbt.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `logs.stats` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        logs_stats_cfg = settings['logs']['stats']

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            logs_stats_cfg
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
            res_status_counts=dict(
                title='Status Counts',
                calc_func='res_status.value_counts',
                incl_all_keys=True,
                post_calc_func=lambda self, out, settings: to_dict(out, orient='index_series'),
                tags=['logs', 'res_status', 'value_counts']
            ),
            res_status_info_counts=dict(
                title='Status Info Counts',
                calc_func='res_status_info.value_counts',
                post_calc_func=lambda self, out, settings: to_dict(out, orient='index_series'),
                tags=['logs', 'res_status_info', 'value_counts']
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        return self._metrics


Logs.override_metrics_doc(__pdoc__)
