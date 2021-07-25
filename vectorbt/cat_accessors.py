"""Custom pandas accessors for categorical data.

Categorical data in vectorbt is data of integer type, optionally with a mapping attached.

Methods can be accessed as follows:

* `CatAccessor` -> `pd.DataFrame.vbt.cat.*`

The accessors inherit `vectorbt.generic.accessors`.

!!! note
    Accessors do not utilize caching."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.root_accessors import register_series_accessor, register_dataframe_accessor
from vectorbt.utils import checks
from vectorbt.utils.config import Config, merge_dicts
from vectorbt.utils.mapping import to_mapping, apply_mapping
from vectorbt.generic.stats_builder import StatsBuilderMixin
from vectorbt.generic.accessors import GenericAccessor, GenericSRAccessor, GenericDFAccessor


class CatAccessor(GenericAccessor):
    """Accessor on top of categorical data.

    Accessible through `pd.Series.vbt.cat` and `pd.DataFrame.vbt.cat`."""

    def __init__(self, obj: tp.Frame, mapping: tp.Optional[tp.MappingLike] = None, **kwargs) -> None:
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj

        checks.assert_dtype(obj, np.int_)
        if mapping is not None:
            mapping = to_mapping(mapping)
        self._mapping = mapping

        GenericAccessor.__init__(self, obj, mapping=mapping, **kwargs)

    @property
    def sr_accessor_cls(self):
        """Accessor class for `pd.Series`."""
        return CatSRAccessor

    @property
    def df_accessor_cls(self):
        """Accessor class for `pd.DataFrame`."""
        return CatDFAccessor

    @property
    def mapping(self) -> tp.Optional[tp.Mapping]:
        """Mapping."""
        return self._mapping

    def map(self, **kwargs) -> tp.SeriesFrame:
        """See `vectorbt.utils.mapping.apply_mapping`."""
        return apply_mapping(self.obj, self.mapping, **kwargs)

    @property
    def stats_defaults(self) -> tp.Kwargs:
        """Defaults for `CatAccessor.stats`.

        Merges `vectorbt.generic.stats_builder.StatsBuilderMixin.stats_defaults` and
        `cat.stats` in `vectorbt._settings.settings`."""
        from vectorbt._settings import settings
        cat_stats_cfg = settings['cat']['stats']

        return merge_dicts(
            StatsBuilderMixin.stats_defaults.__get__(self),
            dict(settings=dict(mapping=self.mapping)),
            cat_stats_cfg
        )

    _metrics: tp.ClassVar[Config] = Config(
        dict(
            start=dict(
                title='Start',
                calc_func=lambda self: self.wrapper.index[0],
                agg_func=None
            ),
            end=dict(
                title='End',
                calc_func=lambda self: self.wrapper.index[-1],
                agg_func=None
            ),
            period=dict(
                title='Period',
                calc_func=lambda self: len(self.wrapper.index),
                auto_to_duration=True,
                agg_func=None
            ),
            value_counts=dict(
                title='Value Counts',
                calc_func=lambda value_counts: value_counts.vbt.to_dict(orient='index_series')
            )
        ),
        copy_kwargs=dict(copy_mode='deep')
    )

    @property
    def metrics(self) -> Config:
        return self._metrics


@register_series_accessor('cat')
class CatSRAccessor(CatAccessor, GenericSRAccessor):
    """Accessor on top of categorical series. For Series only.

    Accessible through `pd.Series.vbt.cat`."""


@register_dataframe_accessor('cat')
class CatDFAccessor(CatAccessor, GenericDFAccessor):
    """Accessor on top of categorical series. For DataFrames only.

    Accessible through `pd.DataFrame.vbt.cat`."""


__pdoc__ = dict()
CatAccessor.override_metrics_doc(__pdoc__)
