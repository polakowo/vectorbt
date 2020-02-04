from functools import wraps
from inspect import signature, Parameter
import numpy as np


def _get_arg(arg_name, func, *args, **kwargs):
    """Search for arg in arguments and keyword arguments."""
    if arg_name in kwargs and kwargs[arg_name] is not None:
        # in kwargs
        return kwargs[arg_name]
    else:
        func_params = signature(func).parameters
        if arg_name in func_params:
            if func_params[arg_name].default is Parameter.empty:
                # in args
                return args[list(func_params.keys()).index(arg_name)]
            else:
                # in function's kwargs with default value
                default_value = func_params[arg_name].default
                if default_value is not None:
                    return default_value
        else:
            raise ValueError(f"Argument {arg_name} not found")
    return None


def _set_arg(arg, arg_name, func, *args, **kwargs):
    """Motify arguments or keyword arguments to include new arg."""
    func_params = signature(func).parameters
    if func_params[arg_name].default is Parameter.empty:
        # modify args
        arg_idx = list(func_params.keys()).index(arg_name)
        args = list(args)
        args[arg_idx] = arg
    else:
        # modify kwargs
        kwargs[arg_name] = arg
    return args, kwargs


def has_type(arg_name, types):
    def has_type_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """If array not of type, raise an exception."""
            nonlocal types
            arg = _get_arg(arg_name, func, *args, **kwargs)
            if arg is None:
                return func(*args, **kwargs)
            if isinstance(types, str):
                # We cannot pass class name as an argument of a class method's decorator,
                # hence specify index of an argument which should act as this class or its instance
                arg2 = _get_arg(types, func, *args, **kwargs)
                if arg2.__class__ == type:  # is class
                    types = arg2
                else:  # is instance
                    types = arg2.__class__
            if not isinstance(arg, types):
                if isinstance(types, tuple):
                    raise ValueError(f"Argument {arg_name} must be one of types {types}")
                else:
                    raise ValueError(f"Argument {arg_name} must be of type {types}")
            return func(*args, **kwargs)
        return wrapper_decorator
    return has_type_decorator


def has_dtype(arg_name, dtype):
    def has_dtype_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """If array not of dtype, raise an exception."""
            arg = _get_arg(arg_name, func, *args, **kwargs)
            if arg is None:
                return func(*args, **kwargs)
            if not np.issubdtype(arg.dtype, dtype):
                raise ValueError(f"Argument {arg_name} must be of dtype {dtype}")
            return func(*args, **kwargs)
        return wrapper_decorator
    return has_dtype_decorator


def to_dtype(arg_name, dtype):
    def to_dtype_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Convert to another dtype."""
            arg = _get_arg(arg_name, func, *args, **kwargs)
            if arg is None:
                return func(*args, **kwargs)
            arg = arg.astype(dtype)
            args, kwargs = _set_arg(arg, arg_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return to_dtype_decorator


def _to_dim1(arg, allow_number):
    if allow_number and isinstance(arg, (int, float, complex)):
        return arg
    if arg.ndim == 1:
        return arg
    if arg.ndim == 2 and arg.shape[1] == 1:
        return arg[:, 0]
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


def to_dim1(arg_name, allow_number=False):
    def to_dim1_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Try to reshape to 2 dimensions."""
            arg = _get_arg(arg_name, func, *args, **kwargs)
            if arg is None:
                return func(*args, **kwargs)
            arg = _to_dim1(arg, allow_number=allow_number)
            args, kwargs = _set_arg(arg, arg_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return to_dim1_decorator


def _to_dim2(arg, allow_number):
    if allow_number and isinstance(arg, (int, float, complex)):
        return arg
    if arg.ndim == 2:
        return arg
    if arg.ndim == 1:
        return arg[:, None]
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def to_dim2(arg_name, allow_number=False):
    def to_dim2_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Try to reshape to 2 dimensions."""
            arg = _get_arg(arg_name, func, *args, **kwargs)
            if arg is None:
                return func(*args, **kwargs)
            arg = _to_dim2(arg, allow_number=allow_number)
            args, kwargs = _set_arg(arg, arg_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return to_dim2_decorator


def broadcast_both(arg1_name, arg2_name):
    def broadcast_both_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """If an array has less columns than the other one, replicate columns."""
            a = _get_arg(arg1_name, func, *args, **kwargs)
            b = _get_arg(arg2_name, func, *args, **kwargs)
            if a is None or b is None:
                return func(*args, **kwargs)

            # If one of the arguments is a single value, replicate
            if isinstance(a, (int, float, complex)):
                a = np.full(b.shape, a)
            if isinstance(b, (int, float, complex)):
                b = np.full(a.shape, b)

            # Expand columns horizontally
            a = _to_dim2(a, False)
            b = _to_dim2(b, False)
            if a.shape[0] != b.shape[0]:
                raise ValueError(f"Arguments {arg1_name} and {arg2_name} must have the same index")
            if a.shape != b.shape:
                if a.shape[1] == 1:
                    a = np.tile(a, (1, b.shape[1]))
                elif b.shape[1] == 1:
                    b = np.tile(b, (1, a.shape[1]))
                else:
                    raise ValueError("Cannot broadcast if both arrays have multiple columns")

            args, kwargs = _set_arg(a, arg1_name, func, *args, **kwargs)
            args, kwargs = _set_arg(b, arg2_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return broadcast_both_decorator


def broadcast_to(from_name, to_name):
    def broadcast_to_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """If the array a has less columns than the array b, replicate columns."""
            # Broadcast a to b
            a = _get_arg(from_name, func, *args, **kwargs)
            b = _get_arg(to_name, func, *args, **kwargs)
            if a is None or b is None:
                return func(*args, **kwargs)

            # If one of the arguments is a single value, replicate
            if isinstance(a, (int, float, complex)):
                a = np.full(b.shape, a)

            # Expand columns horizontally
            a = _to_dim2(a, False)
            if a.shape[0] != b.shape[0]:
                raise ValueError(f"Arguments {from_name} and {to_name} must have the same index")
            if a.shape != b.shape:
                if a.shape[1] == 1 and b.shape[1] > 1:
                    a = np.tile(a, (1, b.shape[1]))
                elif a.shape[1] > 1 and b.shape[1] == 1:
                    raise ValueError(f"Argument {to_name} has less columns than argument {from_name}")
                else:
                    raise ValueError("Cannot broadcast if both arrays have multiple columns")

            args, kwargs = _set_arg(a, from_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return broadcast_to_decorator


def add_nb_methods(*nb_funcs):
    """Wrap numba functions as methods."""
    def wrapper(cls):
        for nb_func in nb_funcs:
            @to_dim2('self')
            def array_operation(self, *args, nb_func=nb_func):
                return cls(nb_func(self, *args))
            setattr(cls, nb_func.__name__.replace('_nb', ''), array_operation)
        return cls
    return wrapper


def cache_property(func):
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
    return wrapper_decorator
