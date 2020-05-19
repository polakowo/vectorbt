"""Property definitions for time series and metrics."""

from vectorbt.utils.decorators import custom_property, cached_property


class timeseries_property():
    """Cached property holding a time series."""

    def __init__(self, name, format):
        self.name = name
        self.format = format

    def __call__(self, func):
        return cached_property(func, property_cls=self.__class__, name=self.name, format=self.format)


class metric_property():
    """Cached property holding a metric."""

    def __init__(self, name, format):
        self.name = name
        self.format = format

    def __call__(self, func):
        return cached_property(func, property_cls=self.__class__, name=self.name, format=self.format)


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
