"""Custom pandas accessors."""

from vectorbt.accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.widgets import basic
from vectorbt.utils import checks


@register_dataframe_accessor('bar')
@register_series_accessor('bar')
class Bar_Accessor():
    """Allows calling `vectorbt.widgets.basic.Bar`
    using `pd.Series.vbt.bar` and `pd.DataFrame.vbt.bar`."""

    def __init__(self, obj):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._obj = obj

    def __call__(self, x_labels=None, trace_names=None, **kwargs):
        obj = self._obj
        if x_labels is None:
            x_labels = obj.index
        if trace_names is None:
            if obj.vbt.is_frame() or (obj.vbt.is_series() and obj.name is not None):
                trace_names = obj.vbt.columns
        return basic.Bar(x_labels, trace_names=trace_names, data=obj.vbt.to_2d_array(), **kwargs)


@register_dataframe_accessor('scatter')
@register_series_accessor('scatter')
class Scatter_Accessor():
    """Allows calling `vectorbt.widgets.basic.Scatter`
    using `pd.Series.vbt.scatter` and `pd.DataFrame.vbt.scatter`."""

    def __init__(self, obj):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._obj = obj

    def __call__(self, x_labels=None, trace_names=None, **kwargs):
        obj = self._obj
        if x_labels is None:
            x_labels = obj.index
        if trace_names is None:
            if obj.vbt.is_frame() or (obj.vbt.is_series() and obj.name is not None):
                trace_names = obj.vbt.columns
        return basic.Scatter(x_labels, trace_names=trace_names, data=obj.vbt.to_2d_array(), **kwargs)


@register_dataframe_accessor('hist')
@register_series_accessor('hist')
class Histogram_Accessor():
    """Allows calling `vectorbt.widgets.basic.Histogram`
    using `pd.Series.vbt.hist` and `pd.DataFrame.vbt.hist`."""

    def __init__(self, obj):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._obj = obj

    def __call__(self, trace_names=None, **kwargs):
        obj = self._obj
        if trace_names is None:
            if obj.vbt.is_frame() or (obj.vbt.is_series() and obj.name is not None):
                trace_names = obj.vbt.columns
        return basic.Histogram(trace_names=trace_names, data=obj.vbt.to_2d_array(), **kwargs)


@register_dataframe_accessor('box')
@register_series_accessor('box')
class Box_Accessor():
    """Allows calling `vectorbt.widgets.basic.Box`
    using `pd.Series.vbt.boxplot` and `pd.DataFrame.vbt.boxplot`."""

    def __init__(self, obj):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._obj = obj

    def __call__(self, trace_names=None, **kwargs):
        obj = self._obj
        if trace_names is None:
            if obj.vbt.is_frame() or (obj.vbt.is_series() and obj.name is not None):
                trace_names = obj.vbt.columns
        return basic.Box(trace_names=trace_names, data=obj.vbt.to_2d_array(), **kwargs)


@register_dataframe_accessor('heatmap')
@register_series_accessor('heatmap')
class Heatmap_Accessor():
    """Allows calling `vectorbt.widgets.basic.Heatmap`
    using `pd.Series.vbt.heatmap` and `pd.DataFrame.vbt.heatmap`."""

    def __init__(self, obj):
        if not checks.is_pandas(obj):  # parent accessor
            obj = obj._obj
        self._obj = obj

    def __call__(self, x_labels=None, y_labels=None, **kwargs):
        obj = self._obj
        if x_labels is None:
            x_labels = obj.vbt.columns
        if y_labels is None:
            y_labels = obj.index
        return basic.Heatmap(x_labels, y_labels, data=obj.vbt.to_2d_array(), **kwargs)
