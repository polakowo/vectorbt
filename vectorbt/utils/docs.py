"""Utilities for documentation."""

import inspect
import sys
from types import FunctionType, MethodType

from vectorbt.utils.decorators import custom_property


def is_from_module(obj, module):
    """Return whether `obj` is from module `module`."""
    mod = inspect.getmodule(inspect.unwrap(obj))
    return mod is None or mod.__name__ == module.__name__


def list_module_keys(module_name, whitelist=[], blacklist=[]):
    """List the names of all public functions and classes defined in the module `module_name`.

    Includes the names listed in `whitelist` and excludes the names listed in `blacklist`."""
    module = sys.modules[module_name]
    return [name for name, obj in inspect.getmembers(module)
            if (not name.startswith("_") and is_from_module(obj, module)
                and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))
                and name not in blacklist) or name in whitelist]


def fix_class_for_docs(cls):
    """Make functions and properties that were defined in any superclass of `cls` visible 
    in the documentation of `cls`."""
    for func_name in dir(cls):
        if not func_name.startswith("_"):
            func = getattr(cls, func_name)
            if isinstance(func, (FunctionType, MethodType, property, custom_property)):
                setattr(cls, func_name, func)
