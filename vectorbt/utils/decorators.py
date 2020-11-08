"""Class and function decorators."""

from functools import wraps, lru_cache, RLock

from vectorbt.utils import checks


class class_or_instancemethod(classmethod):
    """Function decorator that binds `self` to a class if the function is called as class method,
    otherwise to an instance."""

    def __get__(self, instance, type_):
        descr_get = super().__get__ if instance is None else self.__func__.__get__
        return descr_get(instance, type_)


class classproperty(object):
    """Property that can be called on a class."""

    def __init__(self, f):
        self.f = f

    def __get__(self, obj, owner):
        return self.f(owner)


class custom_property():
    """Custom extensible, read-only property.

    Can be called both as
    ```plaintext
    @custom_property
    def user_function...
    ```
    and
    ```plaintext
    @custom_property(**kwargs)
    def user_function...
    ```

    !!! note
        `custom_property` instances belong to classes, not class instances. Thus changing the property,
        for example, by disabling caching, will do the same for each instance of the class where
        the property has been defined."""

    def __new__(cls, *args, **kwargs):
        if len(args) == 0:
            return lambda func: cls(func, **kwargs)
        elif len(args) == 1:
            return super().__new__(cls)
        else:
            raise ValueError("Either function or keyword arguments must be passed")

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs
        self.__doc__ = getattr(func, '__doc__')

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        return self.func(instance)

    def __set__(self, obj, value):
        raise AttributeError("can't set attribute")


_NOT_FOUND = object()


class cached_property(custom_property):
    """Extends `custom_property` with caching.

    Similar to `functools.cached_property`, but without changing the original attribute.
    
    Disables caching if 
    
    * `vectorbt.defaults.caching` is False, or
    * `disabled` attribute is to True."""

    def __init__(self, func, disabled=False, **kwargs):
        super().__init__(func, **kwargs)
        self.attrname = '__cached_' + func.__name__
        self.lock = RLock()
        self.disabled = disabled

    def clear_cache(self, instance):
        """Clear the cache for this property belonging to `instance`."""
        if hasattr(instance, self.attrname):
            delattr(instance, self.attrname)

    def __set_name__(self, owner, name):
        self.attrname = '__cached_' + name  # here is the difference

    def __get__(self, instance, owner=None):
        from vectorbt import defaults

        if instance is None:
            return self
        if not defaults.caching or self.disabled:  # you can manually disable cache here
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


def custom_method(*args, **kwargs):
    """Custom extensible method.

    Stores `**kwargs` as attributes of the wrapper function.

    Can be called both as
    ```plaintext
    @cached_method
    def user_function...
    ```
    and
    ```plaintext
    @cached_method(maxsize=128, typed=False, disabled=False, **kwargs)
    def user_function...
    ```

    !!! note:
        We cannot use a class here since pdoc will treat the method as an instance variable."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.func = func
        wrapper.kwargs = kwargs

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    else:
        raise ValueError("Either function or keyword arguments must be passed")


def cached_method(*args, maxsize=128, typed=False, disabled=False, **kwargs):
    """Extends `custom_method` with caching.

    Internally uses `functools.lru_cache`.

    Disables caching if

    * `vectorbt.defaults.caching` is False,
    * `disabled` attribute is to True, or
    * a non-hashable object was passed as positional or keyword argument.

    Cache can be cleared by calling `clear_cache` with instance as argument.

    !!! note:
        Assumes that the instance (provided as `self`) won't change. If calculation depends
        upon object attributes that can be changed, it won't notice the change."""

    def decorator(func):
        @wraps(func)
        def wrapper(instance, *args, **kwargs):
            from vectorbt import defaults

            def partial_func(*args, **kwargs):
                # Ignores non-hashable instances
                return func(instance, *args, **kwargs)

            if not defaults.caching or wrapper.disabled:  # you can manually disable cache here
                return func(instance, *args, **kwargs)
            cache = instance.__dict__
            cached_func = cache.get(wrapper.attrname, _NOT_FOUND)
            if cached_func is _NOT_FOUND:
                with wrapper.lock:
                    # check if another thread filled cache while we awaited lock
                    cached_func = cache.get(wrapper.attrname, _NOT_FOUND)
                    if cached_func is _NOT_FOUND:
                        cached_func = lru_cache(maxsize=wrapper.maxsize, typed=wrapper.typed)(partial_func)
                        cache[wrapper.attrname] = cached_func  # store function instead of output

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
                return func(instance, *args, **kwargs)
            return cached_func(*args, **kwargs)

        wrapper.func = func
        wrapper.maxsize = maxsize
        wrapper.typed = typed
        wrapper.attrname = '__cached_' + func.__name__
        wrapper.lock = RLock()
        wrapper.disabled = disabled
        wrapper.kwargs = kwargs

        def clear_cache(instance):
            """Clear the cache for this method belonging to `instance`."""
            if hasattr(instance, wrapper.attrname):
                delattr(instance, wrapper.attrname)

        setattr(wrapper, 'clear_cache', clear_cache)

        return wrapper

    if len(args) == 0:
        return decorator
    elif len(args) == 1:
        return decorator(args[0])
    else:
        raise ValueError("Either function or keyword arguments must be passed")


def traverse_attr_kwargs(cls, key=None, value=None):
    """Traverse `cls` and its children for properties/methods with `kwargs`,
    and optionally a specific `key` and `value`.

    Class attributes acting as children should have a key `child_cls`.

    Returns a nested dict of attributes."""
    checks.assert_type(cls, type)

    if value is not None and not isinstance(value, tuple):
        value = (value,)
    attrs = {}
    for attr in dir(cls):
        prop = getattr(cls, attr)
        if hasattr(prop, 'kwargs'):
            kwargs = getattr(prop, 'kwargs')
            if key is None:
                attrs[attr] = kwargs
            else:
                if key in kwargs:
                    if value is None:
                        attrs[attr] = kwargs
                    else:
                        _value = kwargs[key]
                        if _value in value:
                            attrs[attr] = kwargs
            if 'child_cls' in kwargs:
                child_cls = kwargs['child_cls']
                checks.assert_type(child_cls, type)
                attrs[attr] = kwargs
                attrs[attr]['child_attrs'] = traverse_attr_kwargs(child_cls, key, value)
    return attrs
