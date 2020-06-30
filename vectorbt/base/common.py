"""Common functions and classes."""

import numpy as np
import inspect
from abc import ABC, abstractmethod

from vectorbt.utils import checks
from vectorbt.base.array_wrapper import ArrayWrapper


def get_kwargs(func):
    """Get names and default values of keyword arguments from the signature of `func`."""
    return {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def add_nb_methods(nb_funcs, module_name=None):
    """Class decorator to wrap each Numba function in `nb_funcs` as a method of an accessor class.

    Requires the class to be a subclass of `vectorbt.base.array_wrapper.ArrayWrapper`."""

    def wrapper(cls):
        checks.assert_subclass(cls, ArrayWrapper)

        for nb_func in nb_funcs:
            default_kwargs = get_kwargs(nb_func)

            def nb_method(self, *args, nb_func=nb_func, default_kwargs=default_kwargs, **kwargs):
                if '_1d' in nb_func.__name__:
                    # One-dimensional array as input
                    a = nb_func(self.to_1d_array(), *args, **{**default_kwargs, **kwargs})
                    if np.asarray(a).ndim == 0 or len(self.index) != a.shape[0]:
                        return self.wrap_reduced(a)
                    return self.wrap(a)
                else:
                    # Two-dimensional array as input
                    a = nb_func(self.to_2d_array(), *args, **{**default_kwargs, **kwargs})
                    if np.asarray(a).ndim == 0 or a.ndim == 1 or len(self.index) != a.shape[0]:
                        return self.wrap_reduced(a)
                    return self.wrap(a)

            # Replace the function's signature with the original one
            sig = inspect.signature(nb_func)
            self_arg = tuple(inspect.signature(nb_method).parameters.values())[0]
            sig = sig.replace(parameters=(self_arg,) + tuple(sig.parameters.values())[1:])
            nb_method.__signature__ = sig
            if module_name is not None:
                nb_method.__doc__ = f"See `{module_name}.{nb_func.__name__}`"
            else:
                nb_method.__doc__ = f"See `{nb_func.__name__}`"
            setattr(cls, nb_func.__name__.replace('_1d', '').replace('_nb', ''), nb_method)
        return cls

    return wrapper


class AbstractOps(ABC):
    """Abstract class defining methods needed for operations."""
    @abstractmethod
    def apply(self, apply_func=None):
        pass

    @abstractmethod
    def combine_with(self, other, combine_func=None):
        pass


def add_binary_magic_methods(combine_funcs):
    """Class decorator to add binary magic methods using NumPy to the class.

    Requires the class to be a subclass of `AbstractOps`."""

    def wrapper(cls):
        checks.assert_subclass(cls, AbstractOps)

        for fname, combine_func in combine_funcs:

            def magic_func(self, other, combine_func=combine_func):
                return self.combine_with(other, combine_func=combine_func)

            setattr(cls, fname, magic_func)
        return cls

    return wrapper


def add_unary_magic_methods(apply_funcs):
    """Class decorator to add unary magic methods using NumPy to the class.

    Requires the class to be a subclass of `AbstractOps`."""

    def wrapper(cls):
        checks.assert_subclass(cls, AbstractOps)

        for fname, apply_func in apply_funcs:

            def magic_func(self, apply_func=apply_func):
                return self.apply(apply_func=apply_func)

            setattr(cls, fname, magic_func)
        return cls

    return wrapper