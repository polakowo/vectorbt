"""Utilities for working with class/instance attributes."""

import inspect

from vectorbt import _typing as tp
from vectorbt.utils import checks


def deep_getattr(obj: tp.Any, attr_chain: tp.Union[str, tuple, list]) -> tp.Any:
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
