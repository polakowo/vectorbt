"""Utilities for configuration."""

from copy import copy
from collections import namedtuple
import dill
import inspect

from vectorbt.utils import checks
from vectorbt.utils.attr import deep_getattr


def get_func_kwargs(func):
    """Get keyword arguments of the function."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


class atomic_dict(dict):
    """Dict that behaves like a single value when merging."""
    pass


def merge_dicts(*dicts):
    """Merge dicts."""
    x, y = dicts[0], dicts[1]
    if x is None:
        x = {}
    if y is None:
        y = {}
    checks.assert_type(x, dict)
    checks.assert_type(y, dict)

    if len(x) == 0:
        z = y.copy()
    elif len(y) == 0:
        z = x.copy()
    else:
        z = {}
        overlapping_keys = [k for k in x if k in y]  # order matters
        for k in overlapping_keys:
            if isinstance(x[k], dict) and isinstance(y[k], dict) and not isinstance(y[k], atomic_dict):
                z[k] = merge_dicts(x[k], y[k])
            else:
                z[k] = y[k]
        for k in [k for k in x if k not in y]:
            z[k] = x[k]
        for k in [k for k in y if k not in x]:
            z[k] = y[k]

    if len(dicts) > 2:
        return merge_dicts(z, *dicts[2:])
    return z


def copy_dict(dct):
    """Copy dict using shallow-deep copy hybrid.
    
    Traverses all nested dicts and copies each value using shallow copy."""
    dct_copy = dict()
    for k, v in dct.items():
        if isinstance(v, dict):
            dct_copy[k] = copy_dict(v)
        else:
            dct_copy[k] = copy(v)
    return dct_copy


_RaiseKeyError = object()

DumpTuple = namedtuple('DumpTuple', ('cls', 'dumps'))


class Pickleable:
    """Superclass that defines abstract properties and methods for pickle-able classes."""

    def dumps(self, **kwargs):
        """Pickle to a string."""
        raise NotImplementedError

    @classmethod
    def loads(cls, dumps, **kwargs):
        """Unpickle from a string."""
        raise NotImplementedError

    def save(self, fname, **kwargs):
        """Save dumps to a file."""
        dumps = self.dumps(**kwargs)
        with open(fname, "wb") as f:
            f.write(dumps)

    @classmethod
    def load(cls, fname, **kwargs):
        """Load dumps from a file and create new instance."""
        with open(fname, "rb") as f:
            dumps = f.read()
        return cls.loads(dumps, **kwargs)


class Config(dict, Pickleable):
    """Extends dict with config features."""

    def __init__(self, *args, frozen=False, read_only=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._frozen = frozen
        self._read_only = read_only
        self._init_config = copy_dict(self) if not read_only else None

    @property
    def frozen(self):
        """Whether this dict's keys are frozen."""
        return self._frozen

    @property
    def read_only(self):
        """Whether this dict is read-only."""
        return self._read_only

    @property
    def init_config(self):
        """Initial config."""
        return self._init_config

    def __setitem__(self, k, v):
        if self.read_only:
            raise TypeError("Config is read-only")
        if self.frozen:
            if k not in self:
                raise KeyError(f"Key '{k}' is not valid")
        super().__setitem__(k, v)

    def __delitem__(self, k):
        if self.read_only:
            raise TypeError("Config is read-only")
        super().__delitem__(k)

    def pop(self, k, v=_RaiseKeyError):
        if self.read_only:
            raise TypeError("Config is read-only")
        if v is _RaiseKeyError:
            return super().pop(k)
        return super().pop(k, v)

    def popitem(self):
        if self.read_only:
            raise TypeError("Config is read-only")
        return super().popitem()

    def clear(self):
        if self.read_only:
            raise TypeError("Config is read-only")
        return super().clear()

    def update(self, *args, force_update=False, **kwargs):
        other = dict(*args, **kwargs)
        if force_update:
            super().update(other)
            return
        if self.read_only:
            raise TypeError("Config is read-only")
        if self.frozen:
            for k in other:
                if k not in self:
                    raise KeyError(f"Key '{k}' is not valid")
        super().update(other)

    def copy(self):
        return type(self)(self)

    def merge_with(self, other, **kwargs):
        """Merge this and other dict into a new config."""
        return self.__class__(merge_dicts(self, other), **kwargs)

    def reset(self):
        """Reset to the initial config."""
        if self.read_only:
            raise TypeError("Config is read-only")
        self.update(copy_dict(self.init_config), force_update=True)

    def dumps(self, **kwargs):
        """Pickle to a string."""
        config = dict(frozen=self.frozen, read_only=self.read_only)
        for k, v in self.items():
            if k in ('frozen', 'readonly'):
                raise ValueError(f"Keyword argument repeated: {k}")
            if isinstance(v, Pickleable):
                config[k] = DumpTuple(cls=v.__class__, dumps=v.dumps(**kwargs))
            else:
                config[k] = v
        return dill.dumps(config, **kwargs)

    @classmethod
    def loads(cls, dumps, **kwargs):
        """Unpickle from a string."""
        config = dill.loads(dumps, **kwargs)
        for k, v in config.items():
            if isinstance(v, DumpTuple):
                config[k] = v.cls.loads(v.dumps, **kwargs)
        return cls(**config)

    def __eq__(self, other):
        return checks.is_deep_equal(dict(self), dict(other))


class AtomicConfig(Config, atomic_dict):
    """Config that behaves like a single value when merging."""
    pass


class Configured(Pickleable):
    """Class with an initialization config.

    All operations are done using config rather than the instance, which makes it easier to pickle.

    !!! warning
        If the instance has writable attributes or depends upon global defaults,
        their values won't be copied over. Make sure to pass them explicitly to
        make the saved & loaded / copied instance resilient to changes in globals."""

    def __init__(self, **config):
        self._config = Config(config, read_only=True)

    @property
    def config(self):
        """Initialization config."""
        return self._config

    def copy(self, **new_config):
        """Create a new instance based on the config.

        !!! warning
            This "copy" operation won't return a copy of the instance but a new instance
            initialized with the same config."""
        return self.__class__(**self.config.merge_with(new_config))

    def dumps(self, **kwargs):
        """Pickle to a string."""
        return self.config.dumps(**kwargs)

    @classmethod
    def loads(cls, dumps, **kwargs):
        """Unpickle from a string."""
        return cls(**Config.loads(dumps, **kwargs))

    def __eq__(self, other):
        """Objects are equal if their configs are equal."""
        if type(self) != type(other):
            return False
        return self.config == other.config

    def getattr(self, attr_chain):
        """See `vectorbt.utils.attr.deep_getattr`."""
        return deep_getattr(self, attr_chain)

    def update_config(self, *args, **kwargs):
        """Force-update the config."""
        self.config.update(*args, **kwargs, force_update=True)
