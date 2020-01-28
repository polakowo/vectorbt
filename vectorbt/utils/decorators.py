from functools import wraps
import numpy as np


def has_type(arg_idx, types):
    def array_type_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """If not of type, raise an exception."""
            nonlocal types
            if isinstance(arg_idx, str):
                if arg_idx not in kwargs or kwargs[arg_idx] is None:
                    return func(*args, **kwargs)
                arg = kwargs[arg_idx]
            else:
                arg = args[arg_idx]
            if isinstance(types, int):
                # We cannot pass class name as an argument of a class method's decorator,
                # hence specify index of an argument which should act as this class or its instance
                if args[types].__class__ == type:  # is class
                    types = args[types]
                else:  # is instance
                    types = args[types].__class__
            if not isinstance(arg, types):
                if isinstance(types, tuple):
                    raise ValueError(f"Argument {arg_idx} must be one of types {types}")
                else:
                    raise ValueError(f"Argument {arg_idx} must be of type {types}")
            return func(*args, **kwargs)
        return wrapper_decorator
    return array_type_decorator


def _to_dim2(arg):
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        return arg[:, None]
    else:
        raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def to_dim2(arg_idx):
    def to_dim2_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Try to reshape to 2 dimensions."""
            if isinstance(arg_idx, str):
                if arg_idx not in kwargs or kwargs[arg_idx] is None:
                    return func(*args, **kwargs)
                kwargs[arg_idx] = _to_dim2(kwargs[arg_idx])
            else:
                args = list(args)
                args[arg_idx] = _to_dim2(args[arg_idx])
            return func(*args, **kwargs)
        return wrapper_decorator
    return to_dim2_decorator


def _to_dim1(arg):
    if arg.ndim == 1:
        return arg
    elif arg.ndim == 2 and arg.shape[1] == 1:
        return arg[:, 0]
    else:
        raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


def to_dim1(arg_idx):
    def to_dim1_decorator(func):
        @wraps(func)
        def wrapper_decorator(*args, **kwargs):
            """Try to reshape to 1 dimension."""
            if isinstance(arg_idx, str):
                if arg_idx not in kwargs or kwargs[arg_idx] is None:
                    return func(*args, **kwargs)
                kwargs[arg_idx] = _to_dim1(kwargs[arg_idx])
            else:
                args = list(args)
                args[arg_idx] = _to_dim1(args[arg_idx])
            return func(*args, **kwargs)
        return wrapper_decorator
    return to_dim1_decorator


def broadcast(arg1_idx, arg2_idx):
    def broadcast_decorator(func):
        @wraps(func)
        @to_dim2(arg1_idx)
        @to_dim2(arg2_idx)
        def wrapper_decorator(*args, **kwargs):
            """If the array has less columns than the other one, replicate them."""
            args = list(args)
            a = args[arg1_idx]
            b = args[arg2_idx]

            if a.shape[0] != b.shape[0]:
                raise ValueError(f"Arguments at index {arg1_idx} and {arg2_idx} must have the same index")
            if a.shape == b.shape:
                return func(*args, **kwargs)

            # Expand columns horizontally
            if a.shape[1] == 1:
                a = a.__class__(np.tile(a, (1, b.shape[1])))
            elif b.shape[1] == 1:
                b = b.__class__(np.tile(b, (1, a.shape[1])))
            else:
                raise ValueError("Cannot broadcast if both arrays have multiple columns")
            args[arg1_idx] = a
            args[arg2_idx] = b
            return func(*args, **kwargs)
        return wrapper_decorator
    return broadcast_decorator
