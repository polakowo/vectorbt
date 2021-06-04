"""Utilities for working with templates."""

from copy import copy
from string import Template

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import set_dict_item


class Rep(Template):
    """Template to replace the whole string with a mapped value.

    Returns the mapped value (which is not necessarily a string)."""
    pass


class Sub(Template):
    """Template to substitute parts of the string with the respective mapped values.

    Returns a string."""
    pass


def deep_substitute(obj: tp.Any, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
    """Traverses an object recursively and substitutes all templates using a mapping.

    Traverses tuples, lists, dicts and (frozen-)sets. Does not look for templates in keys.

    ## Example

    ```python-repl
    >>> from vectorbt.utils.template import Rep, Sub, deep_substitute

    >>> deep_substitute(Rep('key'), {'key': 100})
    100
    >>> deep_substitute(Sub('$key$key'), {'key': 100})
    '100100'
    >>> deep_substitute([Rep('key'), Sub('$key$key')], {'key': 100})
    [100, '100100']
    ```"""
    if mapping is None:
        mapping = {}
    if isinstance(obj, Rep):
        return mapping[obj.template]
    if isinstance(obj, Template):
        return obj.substitute(mapping)
    if isinstance(obj, dict):
        obj = copy(obj)
        for k, v in obj.items():
            set_dict_item(obj, k, deep_substitute(v, mapping=mapping), force=True)
        return obj
    if isinstance(obj, (tuple, list, set, frozenset)):
        result = []
        for o in obj:
            result.append(deep_substitute(o, mapping=mapping))
        if checks.is_namedtuple(obj):
            return type(obj)(*result)
        return type(obj)(result)
    return obj
