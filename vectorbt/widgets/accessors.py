"""Custom pandas accessors."""

from vectorbt.accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.widgets import basic
from vectorbt.utils import checks


@register_dataframe_accessor('Bar')
@register_series_accessor('Bar')
class Bar_Accessor():
    """Allows calling `vectorbt.widgets.basic.Bar` using `pandas.Series.vbt.Bar` and `pandas.DataFrame.vbt.Bar`."""

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


@register_dataframe_accessor('Scatter')
@register_series_accessor('Scatter')
class Scatter_Accessor():
    """Allows calling `vectorbt.widgets.basic.Scatter` using `pandas.Series.vbt.Scatter` and `pandas.DataFrame.vbt.Scatter`."""

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


@register_dataframe_accessor('Histogram')
@register_series_accessor('Histogram')
class Histogram_Accessor():
    """Allows calling `vectorbt.widgets.basic.Histogram` using `pandas.Series.vbt.Histogram` and `pandas.DataFrame.vbt.Histogram`."""

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


@register_dataframe_accessor('Heatmap')
@register_series_accessor('Heatmap')
class Heatmap_Accessor():
    """Allows calling `vectorbt.widgets.basic.Heatmap` using `pandas.Series.vbt.Heatmap` and `pandas.DataFrame.vbt.Heatmap`."""

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
