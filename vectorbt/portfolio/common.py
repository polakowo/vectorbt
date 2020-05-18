"""Common functions and classes."""

from vectorbt.utils import decorators
from vectorbt.utils.common import list_module_keys

class PositionType:
    OPEN = 0
    """Open position."""
    CLOSED = 1
    """Closed position."""


class OutputFormat:
    PERCENT = '%'
    """Output is a ratio that can be converted to percentage."""
    CURRENCY = '$'
    """Output is in currency units such as USD."""
    TIME = 'time'
    """Output is in time units such as days."""
    NOMINAL = 'nominal'
    """Output consists of nominal data."""
    NONE = ''
    """Output doesn't need any formatting."""


class ArrayWrapper():
    """Provides methods for wrapping NumPy arrays."""

    def __init__(self, ts):
        self.ts = ts

    def wrap_array(self, a, **kwargs):
        """Wrap output array to the time series format of this portfolio."""
        return self.ts.vbt.wrap_array(a, **kwargs)

    def wrap_reduced_array(self, a, **kwargs):
        """Wrap output array to the metric format of this portfolio."""
        return self.ts.vbt.timeseries.wrap_reduced_array(a, **kwargs)


class timeseries_property():
    """Cached property holding a time series."""

    def __init__(self, name, format):
        self.name = name
        self.format = format

    def __call__(self, func):
        return decorators.cached_property(func, parent_cls=self.__class__, name=self.name, format=self.format)


class metric_property():
    """Cached property holding a metric."""

    def __init__(self, name, format):
        self.name = name
        self.format = format

    def __call__(self, func):
        return decorators.cached_property(func, parent_cls=self.__class__, name=self.name, format=self.format)


class group_property():
    """Cached property holding a group.

    This will act as gateaway to a class with other custom properties and methods."""

    def __init__(self, name, cls):
        self.name = name
        self.cls = cls

    def __call__(self, func):
        return decorators.cached_property(func, parent_cls=self.__class__, name=self.name, cls=self.cls)


def traverse_properties(cls, parent_cls):
    """Traverse `cls` and its group properties for properties of type `parent_cls`.

    Returns a nested dict of property attributes."""
    attrs = {}
    for attr in dir(cls):
        prop = getattr(cls, attr)
        if isinstance(prop, decorators.custom_property):
            if prop.parent_cls == parent_cls or prop.parent_cls == group_property:
                attrs[attr] = {k: v for k, v in prop.__dict__.items() if k in prop._custom_attrs}
                if prop.parent_cls == group_property:
                    attrs[attr]['children'] = traverse_properties(prop.cls, parent_cls)
    return attrs


def traverse_timeseries(cls):
    """Traverse `cls` and its group properties for time series."""
    return traverse_properties(cls, timeseries_property)


def traverse_metrics(cls):
    """Traverse `cls` and its group properties for metrics."""
    return traverse_properties(cls, metric_property)


__all__ = list_module_keys(__name__)