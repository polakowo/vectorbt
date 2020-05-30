"""Common functions and classes."""

import pandas as pd

from vectorbt.utils import reshape_fns
from vectorbt.timeseries.common import TSArrayWrapper
from vectorbt.utils.decorators import class_or_instancemethod, custom_property, cached_property


class PArrayWrapper(TSArrayWrapper):
    """Introduces convenient portfolio methods on top of `vectorbt.timeseries.common.TSArrayWrapper`."""
    wrap_timeseries = TSArrayWrapper.wrap
    wrap_metric = TSArrayWrapper.wrap_reduced
    
    def wrap_records(self, records, layout):
        return pd.DataFrame(records, columns=layout._fields)


class timeseries_property():
    """Cached property holding a time series."""

    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        return cached_property(func, property_cls=self.__class__, name=self.name)


class metric_property():
    """Cached property holding a metric."""

    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        return cached_property(func, property_cls=self.__class__, name=self.name)


class records_property():
    """Cached property holding records."""

    def __init__(self, name):
        self.name = name

    def __call__(self, func):
        return cached_property(func, property_cls=self.__class__, name=self.name)


class group_property():
    """Cached property holding a group.

    This will act as gateaway to a class with other custom properties and methods."""

    def __init__(self, name, cls):
        self.name = name
        self.cls = cls

    def __call__(self, func):
        return cached_property(func, property_cls=self.__class__, name=self.name, cls=self.cls)


def traverse_properties(cls, property_cls):
    """Traverse `cls` and its group properties for properties of type `property_cls`.

    Returns a nested dict of property attributes."""
    attrs = {}
    for attr in dir(cls):
        prop = getattr(cls, attr)
        if isinstance(prop, custom_property):
            if prop.property_cls == property_cls or prop.property_cls == group_property:
                attrs[attr] = {k: v for k, v in prop.__dict__.items() if k in prop._custom_attrs}
                if prop.property_cls == group_property:
                    attrs[attr]['children'] = traverse_properties(prop.cls, property_cls)
    return attrs


class PropertyTraverser():
    """Attaches class and instance methods for recursive property traversal."""
    
    @class_or_instancemethod
    def traverse_properties(self_or_cls, property_cls):
        """Recursive traversal for any properties of type `property_cls`."""
        if isinstance(self_or_cls, type):
            return traverse_properties(self_or_cls, property_cls)
        return traverse_properties(self_or_cls.__class__, property_cls)

    @class_or_instancemethod
    def traverse_timeseries(self_or_cls):
        """Recursive traversal for time series."""
        return self_or_cls.traverse_properties(timeseries_property)

    @class_or_instancemethod
    def traverse_metrics(self_or_cls):
        """Recursive traversal for metrics."""
        return self_or_cls.traverse_properties(metric_property)

    @class_or_instancemethod
    def traverse_records(self_or_cls):
        """Recursive traversal for records."""
        return self_or_cls.traverse_properties(records_property)
