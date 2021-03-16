"""Utilities for working with class/instance attributes."""

import inspect

from vectorbt.utils import checks


def deep_getattr(obj, attr_chain):
    """Retrieve attribute consecutively.

    `attr_chain` can be:

    * string -> get variable/property or method without arguments
    * tuple of string -> call method without arguments
    * tuple of string and tuple -> call method and pass positional arguments (unpacked)
    * tuple of string, tuple, and dict -> call method and pass positional and keyword arguments (unpacked)
    * tuple or list of any of the above
    """
    checks.assert_type(attr_chain, (str, tuple, list))

    if isinstance(attr_chain, str):
        if '.' in attr_chain:
            return deep_getattr(obj, attr_chain.split('.'))
        result = getattr(obj, attr_chain)
        if inspect.ismethod(result):
            return result()
        return result
    if isinstance(attr_chain, tuple):
        if len(attr_chain) == 1 \
                and isinstance(attr_chain[0], str):
            return getattr(obj, attr_chain[0])()
        if len(attr_chain) == 2 \
                and isinstance(attr_chain[0], str) \
                and isinstance(attr_chain[1], tuple):
            return getattr(obj, attr_chain[0])(*attr_chain[1])
        if len(attr_chain) == 3 \
                and isinstance(attr_chain[0], str) \
                and isinstance(attr_chain[1], tuple) \
                and isinstance(attr_chain[2], dict):
            return getattr(obj, attr_chain[0])(*attr_chain[1], **attr_chain[2])
    result = obj
    for attr in attr_chain:
        result = deep_getattr(result, attr)
    return result


def traverse_attr_kwargs(cls, key=None, value=None):
    """Traverse the class `cls` and its attributes for properties/methods with attribute `kwargs`,
    and optionally a specific `key` and `value`.

    Class attributes acting as children should have a key `child_cls`.

    Returns a nested dict of attributes."""
    checks.assert_type(cls, type)

    if value is not None and not isinstance(value, tuple):
        value = (value,)
    attrs = {}
    for attr in dir(cls):
        prop = getattr(cls, attr)
        if hasattr(prop, 'kwargs'):
            kwargs = getattr(prop, 'kwargs')
            if key is None:
                attrs[attr] = kwargs
            else:
                if key in kwargs:
                    if value is None:
                        attrs[attr] = kwargs
                    else:
                        _value = kwargs[key]
                        if _value in value:
                            attrs[attr] = kwargs
            if 'child_cls' in kwargs:
                child_cls = kwargs['child_cls']
                checks.assert_type(child_cls, type)
                attrs[attr] = kwargs
                attrs[attr]['child_attrs'] = traverse_attr_kwargs(child_cls, key, value)
    return attrs

