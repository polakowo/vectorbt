"""Class decorators and other helpers."""

import numpy as np
import inspect


def get_kwargs(func):
    """Get names and default values of keyword arguments from the signature of `func`."""
    return {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def add_nb_methods(nb_funcs, module_name=None):
    """Class decorator to wrap each Numba function in `nb_funcs` as a method of an accessor class.

    Requires the instance to have attribute `wrapper` of type `vectorbt.base.array_wrapper.ArrayWrapper`."""

    def wrapper(cls):
        for nb_func in nb_funcs:
            default_kwargs = get_kwargs(nb_func)

            def nb_method(self, *args, nb_func=nb_func, default_kwargs=default_kwargs, **kwargs):
                if '_1d' in nb_func.__name__:
                    # One-dimensional array as input
                    a = nb_func(self.to_1d_array(), *args, **{**default_kwargs, **kwargs})
                    if np.asarray(a).ndim == 0 or len(self.wrapper.index) != a.shape[0]:
                        return self.wrapper.wrap_reduced(a)
                    return self.wrapper.wrap(a)
                else:
                    # Two-dimensional array as input
                    a = nb_func(self.to_2d_array(), *args, **{**default_kwargs, **kwargs})
                    if np.asarray(a).ndim == 0 or a.ndim == 1 or len(self.wrapper.index) != a.shape[0]:
                        return self.wrapper.wrap_reduced(a)
                    return self.wrapper.wrap(a)

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


def add_binary_magic_methods(np_funcs, translate_func):
    """Class decorator to add binary magic methods using NumPy to the class."""

    def wrapper(cls):
        for fname, np_func in np_funcs:
            def magic_func(self, other, np_func=np_func):
                return translate_func(self, other, np_func)

            setattr(cls, fname, magic_func)
        return cls

    return wrapper


def add_unary_magic_methods(np_funcs, translate_func):
    """Class decorator to add unary magic methods using NumPy to the class."""

    def wrapper(cls):
        for fname, np_func in np_funcs:
            def magic_func(self, np_func=np_func):
                return translate_func(self, np_func)

            setattr(cls, fname, magic_func)
        return cls

    return wrapper


binary_magic_methods = [
    # comparison ops
    ('__eq__', np.equal),
    ('__ne__', np.not_equal),
    ('__lt__', np.less),
    ('__gt__', np.greater),
    ('__le__', np.less_equal),
    ('__ge__', np.greater_equal),
    # arithmetic ops
    ('__add__', np.add),
    ('__sub__', np.subtract),
    ('__mul__', np.multiply),
    ('__pow__', np.power),
    ('__mod__', np.mod),
    ('__floordiv__', np.floor_divide),
    ('__truediv__', np.true_divide),
    ('__radd__', lambda x, y: np.add(y, x)),
    ('__rsub__', lambda x, y: np.subtract(y, x)),
    ('__rmul__', lambda x, y: np.multiply(y, x)),
    ('__rpow__', lambda x, y: np.power(y, x)),
    ('__rmod__', lambda x, y: np.mod(y, x)),
    ('__rfloordiv__', lambda x, y: np.floor_divide(y, x)),
    ('__rtruediv__', lambda x, y: np.true_divide(y, x)),
    # mask ops
    ('__and__', np.bitwise_and),
    ('__or__', np.bitwise_or),
    ('__xor__', np.bitwise_xor),
    ('__rand__', lambda x, y: np.bitwise_and(y, x)),
    ('__ror__', lambda x, y: np.bitwise_or(y, x)),
    ('__rxor__', lambda x, y: np.bitwise_xor(y, x))
]

unary_magic_methods = [
    ('__neg__', np.negative),
    ('__pos__', np.positive),
    ('__abs__', np.absolute),
    ('__invert__', np.invert)
]
