"""Common utility functions and classes."""

import numpy as np
import pandas as pd
from functools import wraps
import inspect
from types import FunctionType

from vectorbt import defaults

# ############# Configuration ############# #


def merge_kwargs(x, y):
    """Replace conflicting entries in `x` by entries from `y`."""
    z = {}
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        if isinstance(x[key], dict) and isinstance(y[key], dict):
            z[key] = dict_of_dicts_merge(x[key], y[key])
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


def list_valid_module_keys(module):
    """List all public functions and classes defined in the `module`."""
    return [name for name, obj in inspect.getmembers(module)
            if not name.startswith("_")
            and is_from_module(obj, module)
            and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))]


def add__all__to_module(module, blacklist=[]):
    """Add keys that are in `module` and not in `blacklist` to `__all__`."""
    __all__ = getattr(module, '__all__', [])
    all_keys = list_valid_module_keys(module)
    for k in all_keys:
        if k not in blacklist and k not in __all__:
            __all__.append(k)
    module.__all__ = __all__


def fix_class_for_pdoc(cls):
    """Make functions and properties that were defined in any superclass of `cls` visible 
    in the documentation of `cls`."""
    for func_name in dir(cls):
        if not func_name.startswith("_"):
            func = getattr(cls, func_name)
            if isinstance(func, FunctionType):
                setattr(cls, func_name, func)
            if isinstance(func, property):
                setattr(cls, func_name, func)

# ############# Decorators ############# #


def get_kwargs(func):
    """Get names and default values of keyword arguments from the signature of `func`."""
    return {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def add_nb_methods(*nb_funcs, module_name=None):
    """Class decorator to wrap each Numba function in `nb_funcs` as a method of this class."""
    def wrapper(cls):
        for nb_func in nb_funcs:
            default_kwargs = get_kwargs(nb_func)

            def array_operation(self, *args, nb_func=nb_func, default_kwargs=default_kwargs, **kwargs):
                if '_1d' in nb_func.__name__:
                    return self.wrap_array(nb_func(self.to_1d_array(), *args, **{**default_kwargs, **kwargs}))
                else:
                    # We work natively on 2d arrays
                    return self.wrap_array(nb_func(self.to_2d_array(), *args, **{**default_kwargs, **kwargs}))
            # Replace the function's signature with the original one
            sig = inspect.signature(nb_func)
            self_arg = tuple(inspect.signature(array_operation).parameters.values())[0]
            sig = sig.replace(parameters=(self_arg,)+tuple(sig.parameters.values())[1:])
            array_operation.__signature__ = sig
            if module_name is not None:
                array_operation.__doc__ = f"See `{module_name}.{nb_func.__name__}`"
            else:
                array_operation.__doc__ = f"See `{nb_func.__name__}`"
            setattr(cls, nb_func.__name__.replace('_1d', '').replace('_nb', ''), array_operation)
        return cls
    return wrapper


def cached_property(func):
    """Function decorator similar to `property` but with caching."""
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        obj = args[0]
        attr_name = '_' + func.__name__
        if defaults.cached_property and hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        else:
            to_be_cached = func(*args, **kwargs)
            setattr(obj, attr_name, to_be_cached)
            return to_be_cached
    return property(wrapper_decorator)


class class_or_instancemethod(classmethod):
    """Function decorator that binds `self` to a class if the function is called as class method, 
    otherwise to an instance."""

    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)
