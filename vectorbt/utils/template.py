"""Utilities for working with templates."""

from copy import copy
from string import Template

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import set_dict_item, get_func_arg_names


class Sub(Template):
    """Template to substitute parts of the string with the respective values from `mapping`.

    Returns a string."""
    pass


class Rep:
    """Key to be replaced with the respective value from `mapping`."""

    def __init__(self, key: tp.Hashable) -> None:
        self._key = key

    @property
    def key(self) -> tp.Hashable:
        """Key to be replaced."""
        return self._key


class RepEval:
    """Expression to be evaluated with `mapping` used as locals."""

    def __init__(self, expression: str) -> None:
        self._expression = expression

    @property
    def expression(self) -> str:
        """Expression to be evaluated."""
        return self._expression


class RepFunc:
    """Function to be run with argument names from `mapping`."""

    def __init__(self, func: tp.Callable) -> None:
        self._func = func

    @property
    def func(self) -> tp.Callable:
        """Replacement function to be run."""
        return self._func


def deep_substitute(obj: tp.Any, mapping: tp.Optional[tp.Mapping] = None) -> tp.Any:
    """Traverses an object recursively and substitutes all templates using a mapping.

    Traverses tuples, lists, dicts and (frozen-)sets. Does not look for templates in keys.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> vbt.deep_substitute(vbt.Sub('$key'), {'key': 100})
    '100'
    >>> vbt.deep_substitute(vbt.Sub('$key$key'), {'key': 100})
    '100100'
    >>> vbt.deep_substitute(vbt.Rep('key'), {'key': 100})
    100
    >>> vbt.deep_substitute([vbt.Rep('key'), vbt.Sub('$key$key')], {'key': 100})
    [100, '100100']
    >>> vbt.deep_substitute(vbt.RepFunc(lambda key: key == 100), {'key': 100})
    True
    >>> vbt.deep_substitute(vbt.RepEval('key == 100'), {'key': 100})
    True
    ```"""
    if mapping is None:
        mapping = {}
    if isinstance(obj, RepFunc):
        func_arg_names = get_func_arg_names(obj.func)
        func_kwargs = dict()
        for k, v in mapping.items():
            if k in func_arg_names:
                func_kwargs[k] = v
        return obj.func(**func_kwargs)
    if isinstance(obj, RepEval):
        return eval(obj.expression, {}, mapping)
    if isinstance(obj, Rep):
        return mapping[obj.key]
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
