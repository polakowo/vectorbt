import numpy as np
import pandas as pd
from functools import wraps
import inspect
from types import FunctionType

# ############# Configuration ############# #


class Config(dict):
    """A simple dict with (optionally) frozen keys."""

    def __init__(self, *args, frozen=True, **kwargs):
        self.frozen = frozen
        self.update(*args, **kwargs)
        self.default_config = dict(self)
        for key, value in dict.items(self):
            if isinstance(value, dict):
                dict.__setitem__(self, key, Config(value))

    def __setitem__(self, key, val):
        if self.frozen and key not in self:
            raise KeyError(f"Key {key} is not a valid parameter")
        dict.__setitem__(self, key, val)

    def reset(self):
        self.update(self.default_config)

# ############# Documentation ############# #


def is_from_module(obj, module):
    """Check if `obj` is from the module named by `module_name`."""
    mod = inspect.getmodule(inspect.unwrap(obj))
    return mod is None or mod.__name__ == module.__name__


def list_valid_module_keys(module):
    """List all functions and classes defined in the `module`."""
    return [name for name, obj in inspect.getmembers(module)
            if not name.startswith("_")
            and is_from_module(obj, module)
            and ((inspect.isroutine(obj) and callable(obj)) or inspect.isclass(obj))]


def add__all__to_module(module, blacklist=[]):
    """Add to `__all__` list keys that are in `module` and not in `blacklist`."""
    __all__ = getattr(module, '__all__', [])
    all_keys = list_valid_module_keys(module)
    for k in all_keys:
        if k not in blacklist and k not in __all__:
            __all__.append(k)
    module.__all__ = __all__


def fix_class_for_pdoc(cls):
    """Make class attributes that were defined in the superclass appear in the documentation of this class."""
    for func_name in dir(cls):
        if not func_name.startswith("_"):
            func = getattr(cls, func_name)
            if isinstance(func, FunctionType):
                setattr(cls, func_name, func)
            if isinstance(func, property):
                setattr(cls, func_name, func)

# ############# Decorators ############# #

def get_default_args(func):
    return {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def add_safe_nb_methods(*nb_funcs):
    def wrapper(cls):
        """Wrap numba functions as methods."""
        for nb_func in nb_funcs:
            default_kwargs = get_default_args(nb_func)

            def array_operation(self, *args, nb_func=nb_func, default_kwargs=default_kwargs, **kwargs):
                if '_1d' in nb_func.__name__:
                    return self.wrap_array(nb_func(self.to_1d_array(), *args, **{**default_kwargs, **kwargs}))
                else:
                    # We work natively on 2d arrays
                    return self.wrap_array(nb_func(self.to_2d_array(), *args, **{**default_kwargs, **kwargs}))
            setattr(cls, nb_func.__name__.replace('_1d', '').replace('_nb', ''), array_operation)
        return cls
    return wrapper


def cached_property(func):
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        """Cache property to avoid recalculating it again and again."""
        obj = args[0]
        attr_name = '_' + func.__name__
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        else:
            to_be_cached = func(*args, **kwargs)
            setattr(obj, attr_name, to_be_cached)
            return to_be_cached
    return property(wrapper_decorator)


class class_or_instancemethod(classmethod):
    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)