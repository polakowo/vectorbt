"""Custom pandas accessors for widgets."""

from vectorbt.accessors import register_dataframe_accessor, register_series_accessor
from vectorbt.utils import checks, reshape_fns
from vectorbt.widgets import widgets


@register_dataframe_accessor('Bar')
@register_series_accessor('Bar')
class Bar_Accessor():
    """Allows calling `Bar` using a pandas accessor function `pandas.vbt.Bar`."""

    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, x_labels=None, trace_names=None, **kwargs):
        if x_labels is None:
            x_labels = self._obj.index
        if trace_names is None:
            if checks.is_frame(self._obj) or (checks.is_series(self._obj) and self._obj.name is not None):
                trace_names = reshape_fns.to_2d(self._obj).columns
        return widgets.Bar(x_labels, trace_names=trace_names, data=self._obj.values, **kwargs)


@register_dataframe_accessor('Scatter')
@register_series_accessor('Scatter')
class Scatter_Accessor():
    """Allows calling `Scatter` using a pandas accessor function `pandas.vbt.Scatter`."""

    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, x_labels=None, trace_names=None, **kwargs):
        if x_labels is None:
            x_labels = self._obj.index
        if trace_names is None:
            if checks.is_frame(self._obj) or (checks.is_series(self._obj) and self._obj.name is not None):
                trace_names = reshape_fns.to_2d(self._obj).columns
        return widgets.Scatter(x_labels, trace_names=trace_names, data=self._obj.values, **kwargs)


@register_dataframe_accessor('Histogram')
@register_series_accessor('Histogram')
class Histogram_Accessor():
    """Allows calling `Histogram` using a pandas accessor function `pandas.vbt.Histogram`."""

    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, trace_names=None, **kwargs):
        if trace_names is None:
            if checks.is_frame(self._obj) or (checks.is_series(self._obj) and self._obj.name is not None):
                trace_names = reshape_fns.to_2d(self._obj).columns
        return widgets.Histogram(trace_names=trace_names, data=self._obj.values, **kwargs)


@register_dataframe_accessor('Heatmap')
@register_series_accessor('Heatmap')
class Heatmap_Accessor():
    """Allows calling `Heatmap` using a pandas accessor function `pandas.vbt.Heatmap`."""

    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object

    def __call__(self, x_labels=None, y_labels=None, **kwargs):
        if x_labels is None:
            x_labels = reshape_fns.to_2d(self._obj).columns
        if y_labels is None:
            y_labels = self._obj.index
        return widgets.Heatmap(x_labels, y_labels, data=self._obj.values, **kwargs)
