"""Utilities for working with class/instance attributes."""

from collections.abc import Iterable

from vectorbt import _typing as tp
from vectorbt.utils import checks


def default_getattr_func(obj: tp.Any,
                         attr: str,
                         args: tp.Optional[tp.Args] = None,
                         kwargs: tp.Optional[tp.Kwargs] = None,
                         call_attr: bool = True) -> tp.Any:
    """Default `getattr_func`."""
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    out = getattr(obj, attr)
    if callable(out) and call_attr:
        return out(*args, **kwargs)
    return out


def deep_getattr(obj: tp.Any,
                 attr_chain: tp.Union[str, tuple, Iterable],
                 getattr_func: tp.Callable = default_getattr_func,
                 call_last_attr: bool = True) -> tp.Any:
    """Retrieve attribute consecutively.

    The attribute chain `attr_chain` can be:

    * string -> get variable/property or method without arguments
    * tuple of string -> call method without arguments
    * tuple of string and tuple -> call method and pass positional arguments (unpacked)
    * tuple of string, tuple, and dict -> call method and pass positional and keyword arguments (unpacked)
    * iterable of any of the above

    Use `getattr_func` to overwrite the default behavior of accessing an attribute (see `default_getattr_func`).

    !!! hint
        If your chain includes only attributes and functions without arguments,
        you can represent this chain as a single (but probably long) string.
    """
    checks.assert_type(attr_chain, (str, tuple, Iterable))

    if isinstance(attr_chain, str):
        if '.' in attr_chain:
            return deep_getattr(
                obj,
                attr_chain.split('.'),
                getattr_func=getattr_func,
                call_last_attr=call_last_attr
            )
        return getattr_func(obj, attr_chain, call_attr=call_last_attr)
    if isinstance(attr_chain, tuple):
        if len(attr_chain) == 1 \
                and isinstance(attr_chain[0], str):
            return getattr_func(obj, attr_chain[0])
        if len(attr_chain) == 2 \
                and isinstance(attr_chain[0], str) \
                and isinstance(attr_chain[1], tuple):
            return getattr_func(obj, attr_chain[0], args=attr_chain[1])
        if len(attr_chain) == 3 \
                and isinstance(attr_chain[0], str) \
                and isinstance(attr_chain[1], tuple) \
                and isinstance(attr_chain[2], dict):
            return getattr_func(obj, attr_chain[0], args=attr_chain[1], kwargs=attr_chain[2])
    result = obj
    for i, attr in enumerate(attr_chain):
        if i < len(attr_chain) - 1:
            result = deep_getattr(
                result,
                attr,
                getattr_func=getattr_func,
                call_last_attr=True
            )
        else:
            result = deep_getattr(
                result,
                attr,
                getattr_func=getattr_func,
                call_last_attr=call_last_attr
            )
    return result
