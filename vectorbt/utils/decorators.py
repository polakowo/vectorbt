from functools import wraps
from inspect import signature, Parameter
import numpy as np


def _get_arg(arg_name, func, *args, **kwargs):
    """Search for arg in arguments and keyword arguments."""
    arg_attr = None
    if '.' in arg_name:
        arg_name, arg_attr = arg_name.split('.')
    if arg_name in kwargs and kwargs[arg_name] is not None:
        # in kwargs
        arg = kwargs[arg_name]
        if arg_attr is not None:
            return getattr(arg, arg_attr)
        return arg
    else:
        func_params = signature(func).parameters
        if arg_name in func_params:
            if func_params[arg_name].default is Parameter.empty:
                # in args
                arg = args[list(func_params.keys()).index(arg_name)]
                if arg_attr is not None:
                    return getattr(arg, arg_attr)
                return arg
            else:
                # in function's kwargs with default value
                default_value = func_params[arg_name].default
                if default_value is not None:
                    arg = default_value
                    if arg_attr is not None:
                        return getattr(arg, arg_attr)
                    return arg
        else:
            raise ValueError(f"Argument {arg_name} not found")
    return None


def _set_arg(arg, arg_name, func, *args, **kwargs):
    """Modify arguments or keyword arguments to include new arg."""
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


def have_same_shape(arg1_name, arg2_name, along_axis=None):
    def have_same_shape_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """If both arrays do not have the same shape, raise an exception."""
            arg1 = _get_arg(arg1_name, func, *args, **kwargs)
            arg2 = _get_arg(arg2_name, func, *args, **kwargs)
            if arg1 is None or arg2 is None:
                return func(*args, **kwargs)
            if not isinstance(arg1, np.ndarray):
                arg1 = np.asarray(arg1)
            if not isinstance(arg2, np.ndarray):
                arg2 = np.asarray(arg2)
            if along_axis is None:
                if arg1.shape != arg2.shape:
                    raise ValueError(f"Arguments {arg1_name} and {arg2_name} must have the same shape")
            else:
                if isinstance(along_axis, tuple):
                    if arg1.shape[along_axis[0]] != arg2.shape[along_axis[1]]:
                        raise ValueError(f"Axis {along_axis[0]} of argument {arg1_name} must be the same as axis {along_axis[1]} of argument {arg2_name}")
                else:
                    if arg1.shape[along_axis] != arg2.shape[along_axis]:
                        raise ValueError(f"Arguments {arg1_name} and {arg2_name} must have the same axis {along_axis}")
            return func(*args, **kwargs)
        return wrapper_decorator
    return have_same_shape_decorator


def _to_1d(arg, arg_name):
    if not isinstance(arg, np.ndarray):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        if arg.shape[1] == 1:
            return arg[:, 0]
    if arg.ndim == 1:
        return arg
    elif arg.ndim == 0:
        return arg.reshape((1,))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array {arg_name} to 1 dimension")


def to_1d(arg_name):
    def to_1d_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Try to reshape to 2 dimensions."""
            arg = _get_arg(arg_name, func, *args, **kwargs)
            if arg is None:
                return func(*args, **kwargs)
            arg = _to_1d(arg, arg_name)
            args, kwargs = _set_arg(arg, arg_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return to_1d_decorator


def _to_2d(arg, expand_axis):
    if not isinstance(arg, np.ndarray):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        return np.expand_dims(arg, expand_axis)
    elif arg.ndim == 0:
        return arg.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def to_2d(arg_name, expand_axis=1):
    def to_2d_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Try to reshape to 2 dimensions."""
            arg = _get_arg(arg_name, func, *args, **kwargs)
            if arg is None:
                return func(*args, **kwargs)
            arg = _to_2d(arg, expand_axis)
            args, kwargs = _set_arg(arg, arg_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return to_2d_decorator


def broadcast(arg1_name, arg2_name):
    def broadcast_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Bring both arguments to the same shape."""
            a = _get_arg(arg1_name, func, *args, **kwargs)
            b = _get_arg(arg2_name, func, *args, **kwargs)
            if a is None or b is None:
                return func(*args, **kwargs)
            a, b = np.broadcast_arrays(a, b, subok=True)
            a = a.copy()  # deprecation warning
            b = b.copy()
            args, kwargs = _set_arg(a, arg1_name, func, *args, **kwargs)
            args, kwargs = _set_arg(b, arg2_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return broadcast_decorator


def broadcast_to(arg1_name, arg2_name):
    def broadcast_to_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Bring the first argument to the same shape of the second argument."""
            a = _get_arg(arg1_name, func, *args, **kwargs)
            b = _get_arg(arg2_name, func, *args, **kwargs)
            if a is None or b is None:
                return func(*args, **kwargs)
            a = np.broadcast_to(a, b.shape, subok=True)
            a = a.copy()  # deprecation warning
            args, kwargs = _set_arg(a, arg1_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return broadcast_to_decorator


def broadcast_to_cube_of(arg1_name, arg2_name):
    def broadcast_to_cube_of_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Reshape the first argument to be a cube out of second argument."""
            a = _get_arg(arg1_name, func, *args, **kwargs)
            b = _get_arg(arg2_name, func, *args, **kwargs)
            if a is None or b is None:
                return func(*args, **kwargs)
            if not isinstance(a, np.ndarray):
                a = np.asarray(a)
            if not isinstance(b, np.ndarray):
                b = np.asarray(b)
            if a.ndim == 0:
                a = np.broadcast_to(a, b.shape, subok=True)[None, :]
            elif a.ndim == 1:
                a = np.tile(a[:, None][:, None], (1, b.shape[0], b.shape[1]))
            elif a.ndim == 2:
                if a.shape[1] != b.shape[0] or a.shape[2] != b.shape[1]:
                    a = np.broadcast_to(a, b.shape, subok=True)[None, :]
            a = a.copy()  # deprecation warning
            args, kwargs = _set_arg(a, arg1_name, func, *args, **kwargs)
            return func(*args, **kwargs)
        return wrapper_decorator
    return broadcast_to_cube_of_decorator


def add_nb_methods(*nb_funcs):
    """Wrap numba functions as methods."""
    def wrapper(cls):
        for nb_func in nb_funcs:
            def get_default_args(func):
                return {
                    k: v.default
                    for k, v in signature(func).parameters.items()
                    if v.default is not Parameter.empty
                }
            default_kwargs = get_default_args(nb_func)
            @to_2d('self') # default shape for all our classes
            def array_operation(self, *args, nb_func=nb_func, default_kwargs=default_kwargs, **kwargs):
                return cls(nb_func(self, *args, **{**default_kwargs, **kwargs})) # kwargs must be specified for numba
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
