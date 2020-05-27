"""Class and function decorators."""

from functools import wraps, lru_cache, RLock
import inspect

from vectorbt import defaults
from vectorbt.utils import checks, reshape_fns


def get_kwargs(func):
    """Get names and default values of keyword arguments from the signature of `func`."""
    return {
        k: v.default
        for k, v in inspect.signature(func).parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def add_nb_methods(*nb_funcs, module_name=None):
    """Class decorator to wrap each Numba function in `nb_funcs` as a method of an accessor class."""
    def wrapper(cls):
        for nb_func in nb_funcs:
            default_kwargs = get_kwargs(nb_func)

            def array_operation(self, *args, nb_func=nb_func, default_kwargs=default_kwargs, **kwargs):
                if '_1d' in nb_func.__name__:
                    return self.wrap(nb_func(self.to_1d_array(), *args, **{**default_kwargs, **kwargs}))
                else:
                    # We work natively on 2d arrays
                    return self.wrap(nb_func(self.to_2d_array(), *args, **{**default_kwargs, **kwargs}))
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

class custom_property():
    """Custom extensible, read-only property."""

    def __init__(self, func, **kwargs):
        self.func = func
        self.__doc__ = getattr(func, '__doc__')
        self._custom_attrs = list(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self.func(instance)

    def __set__(self, obj, value):
        raise AttributeError("can't set attribute")

_NOT_FOUND = object()

class cached_property(custom_property):
    """Custom cacheable property.

    Similar to `functools.cached_property`, but without changing the original attribute.
    
    Disables caching if 
    
    * `vectorbt.defaults.caching` is `False`, or
    * `disabled` attribute is to `True`."""

    def __init__(self, func, disabled=False, **kwargs):
        super().__init__(func, **kwargs)
        self.attrname = None
        self.lock = RLock()
        self.disabled = disabled

    def clear_cache(self, instance):
        """Clear the cache for this property belonging to `instance`."""
        if hasattr(instance, self.attrname):
            delattr(instance, self.attrname)

    def __set_name__(self, owner, name):
        self.attrname = '_' + name # here is the difference

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if not defaults.caching or self.disabled: # you can manually disable cache here
            return super().__get__(instance, owner=owner)
        cache = instance.__dict__
        val = cache.get(self.attrname, _NOT_FOUND)
        if val is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                val = cache.get(self.attrname, _NOT_FOUND)
                if val is _NOT_FOUND:
                    val = self.func(instance)
                    cache[self.attrname] = val
        return val


class custom_method():
    """Custom extensible, read-only method."""

    def __init__(self, func, **kwargs):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func
        self._custom_attrs = list(kwargs.keys())
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        func = self.func
        @wraps(func)
        def decorated(*args, **kwargs):
            return func(instance, *args, **kwargs)
        return decorated

    def __set__(self, obj, value):
        raise AttributeError("can't set attribute")


class cached_method(custom_method):
    """Custom cacheable method.
    
    Disables caching if 
    
    * `vectorbt.defaults.caching` is `False`,
    * `disabled` attribute is to `True`, or
    * a non-hashable object was passed as positional or keyword argument."""
    def __init__(self, func, maxsize=128, typed=False, disabled=False, **kwargs):
        super().__init__(func, **kwargs)
        self.maxsize = maxsize
        self.typed = typed
        self.attrname = None
        self.lock = RLock()
        self.disabled = disabled

    def clear_cache(self, instance):
        """Clear the cache for this method belonging to `instance`."""
        if hasattr(instance, self.attrname):
            delattr(instance, self.attrname)

    def __set_name__(self, owner, name):
        self.attrname = '_' + name # here is the difference

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if not defaults.caching or self.disabled: # you can manually disable cache here
            return super().__get__(instance, owner=owner)
        cache = instance.__dict__
        func = cache.get(self.attrname, _NOT_FOUND)
        if func is _NOT_FOUND:
            with self.lock:
                # check if another thread filled cache while we awaited lock
                func = cache.get(self.attrname, _NOT_FOUND)
                if func is _NOT_FOUND:
                    func = lru_cache(maxsize=self.maxsize, typed=self.typed)(self.func)
                    cache[self.attrname] = func # store function instead of output
        @wraps(func)
        def decorated(*args, **kwargs):
            if defaults.caching:
                # Check if object can be hashed
                hashable = True
                for arg in args:
                    if not checks.is_hashable(arg):
                        hashable = False
                        break
                for k, v in kwargs.items():
                    if not checks.is_hashable(v):
                        hashable = False
                        break
                if not hashable:
                    # If not, do not invoke lru_cache
                    return self.func(instance, *args, **kwargs)
            return func(instance, *args, **kwargs)
        return decorated


class class_or_instancemethod(classmethod):
    """Function decorator that binds `self` to a class if the function is called as class method, 
    otherwise to an instance."""

    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)
