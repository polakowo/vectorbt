"""Common functions and classes."""

from vectorbt.utils.decorators import class_or_instancemethod, custom_property, cached_property


class ArrayWrapper():
    """Provides methods for wrapping NumPy arrays."""

    def __init__(self, ref_obj):
        self.ref_obj = ref_obj

    def wrap_timeseries(self, a, **kwargs):
        """Wrap output array to the time series format of this portfolio."""
        return self.ref_obj.vbt.wrap_array(a, **kwargs)

    def wrap_metric(self, a, **kwargs):
        """Wrap output array to the metric format of this portfolio."""
        return self.ref_obj.vbt.timeseries.wrap_reduced_array(a, **kwargs)


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
    @class_or_instancemethod
    def traverse_properties(self_or_cls, property_cls):
        """Traverse this class and its group properties for any properties of type `property_cls`."""
        if isinstance(self_or_cls, type):
            return traverse_properties(self_or_cls, property_cls)
        return traverse_properties(self_or_cls.__class__, property_cls)

    @class_or_instancemethod
    def traverse_timeseries(self_or_cls):
        """Traverse this class and its group properties for time series."""
        return self_or_cls.traverse_properties(timeseries_property)

    @class_or_instancemethod
    def traverse_metrics(self_or_cls):
        """Traverse this class and its group properties for metrics."""
        return self_or_cls.traverse_properties(metric_property)
