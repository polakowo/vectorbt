"""Utilities for working with class/instance attributes."""

import inspect
from collections.abc import Iterable

from vectorbt import _typing as tp
from vectorbt.utils import checks


def deep_getattr(obj: tp.Any, attr_chain: tp.Union[str, tuple, Iterable], call_last_method: bool = True) -> tp.Any:
    """Retrieve attribute consecutively.

    The attribute chain `attr_chain` can be:

    * string -> get variable/property or method without arguments
    * tuple of string -> call method without arguments
    * tuple of string and tuple -> call method and pass positional arguments (unpacked)
    * tuple of string, tuple, and dict -> call method and pass positional and keyword arguments (unpacked)
    * iterable of any of the above

    !!! hint
        If your chain includes only attributes and functions without arguments,
        you can represent this chain as a single (but probably long) string.
    """
    checks.assert_type(attr_chain, (str, tuple, Iterable))

    if isinstance(attr_chain, str):
        if '.' in attr_chain:
            return deep_getattr(obj, attr_chain.split('.'), call_last_method=call_last_method)
        result = getattr(obj, attr_chain)
        if inspect.ismethod(result) and call_last_method:
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
    for i, attr in enumerate(attr_chain):
        if i < len(attr_chain) - 1:
            result = deep_getattr(result, attr, call_last_method=True)
        else:
            result = deep_getattr(result, attr, call_last_method=call_last_method)
    return result
