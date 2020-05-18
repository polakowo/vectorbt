"""Common utilities."""

import pandas as pd
import inspect
import sys
from types import FunctionType

from vectorbt.utils.decorators import custom_property, custom_method

# ############# Configuration ############# #


class Config(dict):
    """A simple dict with (optionally) frozen keys."""

    def __init__(self, *args, frozen=True, **kwargs):
        self.frozen = frozen
        self.update(*args, **kwargs)
        self.default_config = dict(self)
        for key, value in dict.items(self):
            if isinstance(value, dict):
                dict.__setitem__(self, key, Config(value, frozen=frozen))

    def __setitem__(self, key, val):
        if self.frozen and key not in self:
            raise KeyError(f"Key {key} is not a valid parameter")
        dict.__setitem__(self, key, val)

    def reset(self):
        """Reset dictionary to the one passed at instantiation."""
        self.update(self.default_config)


def merge_kwargs(x, y):
    """Merge dictionaries `x` and `y`.

    By conflicts, `y` wins."""
    z = {}
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        if isinstance(x[key], dict) and isinstance(y[key], dict):
            z[key] = merge_kwargs(x[key], y[key])
        else:
            z[key] = y[key]
    for key in x.keys() - overlapping_keys:
        z[key] = x[key]
    for key in y.keys() - overlapping_keys:
        z[key] = y[key]
    return z

# ############# Documentation ############# #


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


def fix_class_for_pdoc(cls):
    """Make functions and properties that were defined in any superclass of `cls` visible 
    in the documentation of `cls`."""
    for func_name in dir(cls):
        if not func_name.startswith("_"):
            func = getattr(cls, func_name)
            if isinstance(func, (FunctionType, property, custom_property, custom_method)):
                setattr(cls, func_name, func)
