"""Utilities for configuration."""

from copy import copy
from collections import namedtuple
import dill
import inspect
from pathlib import Path

from vectorbt import typing as tp
from vectorbt.utils import checks
from vectorbt.utils.attr import deep_getattr


def resolve_dict(dct: tp.DictLikeSequence, i: tp.Optional[int] = None) -> dict:
    """Select keyword arguments."""
    if dct is None:
        dct = {}
    if isinstance(dct, dict):
        return dict(dct)
    if i is not None:
        _dct = dct[i]
        if _dct is None:
            _dct = {}
        return dict(_dct)
    raise ValueError("Cannot resolve dict")


def get_func_kwargs(func: tp.Callable) -> dict:
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


def merge_dicts(*dicts: tp.DictLike) -> dict:
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


def copy_dict(dct: tp.DictLike) -> dict:
    """Copy dict using shallow-deep copy hybrid.
    
    Traverses all nested dicts and copies each value using shallow copy."""
    if dct is None:
        return {}
    dct_copy = type(dct)()
    for k, v in dct.items():
        if isinstance(v, dict):
            dct_copy[k] = copy_dict(v)
        else:
            dct_copy[k] = copy(v)
    return dct_copy


_RaiseKeyError = object()

DumpTuple = namedtuple('DumpTuple', ('cls', 'dumps'))


PickleableT = tp.TypeVar("PickleableT", bound="Pickleable")


class Pickleable:
    """Superclass that defines abstract properties and methods for pickle-able classes."""

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
        raise NotImplementedError

    @classmethod
    def loads(cls: tp.Type[PickleableT], dumps: bytes, **kwargs) -> PickleableT:
        """Unpickle from bytes."""
        raise NotImplementedError

    def save(self, fname: tp.Union[str, Path], **kwargs) -> None:
        """Save dumps to a file."""
        dumps = self.dumps(**kwargs)
        with open(fname, "wb") as f:
            f.write(dumps)

    @classmethod
    def load(cls: tp.Type[PickleableT], fname: tp.Union[str, Path], **kwargs) -> PickleableT:
        """Load dumps from a file and create new instance."""
        with open(fname, "rb") as f:
            dumps = f.read()
        return cls.loads(dumps, **kwargs)


ConfigT = tp.TypeVar("ConfigT", bound="Config")


class Config(dict, Pickleable):
    """Extends dict with config features."""

    def __init__(self,
                 *args,
                 frozen: bool = False,
                 read_only: bool = False,
                 init_config: tp.DictLike = None,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._frozen = frozen
        self._read_only = read_only
        if init_config is None and not read_only:
            init_config = copy_dict(dict(self))
        self._init_config = init_config

    @property
    def frozen(self) -> bool:
        """Whether this config's keys are frozen."""
        return self._frozen

    @property
    def read_only(self) -> bool:
        """Whether this config is read-only."""
        return self._read_only

    @property
    def init_config(self) -> tp.DictLike:
        """Initial config."""
        return self._init_config

    def __setitem__(self, k: tp.Any, v: tp.Any) -> None:
        if self.read_only:
            raise TypeError("Config is read-only")
        if self.frozen:
            if k not in self:
                raise KeyError(f"Key '{k}' is not valid")
        super().__setitem__(k, v)

    def __delitem__(self, k: tp.Any) -> None:
        if self.read_only:
            raise TypeError("Config is read-only")
        super().__delitem__(k)

    def pop(self, k: tp.Any, v: tp.Any = _RaiseKeyError) -> tp.Any:
        """Remove and return the pair by the key."""
        if self.read_only:
            raise TypeError("Config is read-only")
        if v is _RaiseKeyError:
            return super().pop(k)
        return super().pop(k, v)

    def popitem(self) -> tp.Tuple[tp.Any, tp.Any]:
        """Remove and return some pair."""
        if self.read_only:
            raise TypeError("Config is read-only")
        return super().popitem()

    def clear(self) -> None:
        """Remove all items."""
        if self.read_only:
            raise TypeError("Config is read-only")
        super().clear()

    def update(self, *args, force_update: bool = False, **kwargs) -> None:
        """Update config."""
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

    def copy(self: ConfigT) -> ConfigT:
        """Copy config."""
        return self.__class__(
            self,
            frozen=self.frozen,
            read_only=self.read_only,
            init_config=copy_dict(self.init_config)
        )

    def merge_with(self: ConfigT, other: dict, **kwargs) -> ConfigT:
        """Merge this and other dict into a new config."""
        return self.__class__(merge_dicts(self, other), **kwargs)

    def reset(self) -> None:
        """Reset to the initial config."""
        if self.read_only:
            raise TypeError("Config is read-only")
        self.update(copy_dict(self.init_config), force_update=True)

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
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
    def loads(cls: tp.Type[ConfigT], dumps: bytes, **kwargs) -> ConfigT:
        """Unpickle from bytes."""
        config = dill.loads(dumps, **kwargs)
        for k, v in config.items():
            if isinstance(v, DumpTuple):
                config[k] = v.cls.loads(v.dumps, **kwargs)
        return cls(**config)

    def __eq__(self, other: tp.Any) -> bool:
        return checks.is_deep_equal(dict(self), dict(other))


class AtomicConfig(Config, atomic_dict):
    """Config that behaves like a single value when merging."""
    pass


ConfiguredT = tp.TypeVar("ConfiguredT", bound="Configured")


class Configured(Pickleable):
    """Class with an initialization config.

    All operations are done using config rather than the instance, which makes it easier to pickle.

    !!! warning
        If the instance has writable attributes or depends upon global defaults,
        their values won't be copied over. Make sure to pass them explicitly to
        make the saved & loaded / copied instance resilient to changes in globals."""

    def __init__(self, **config) -> None:
        self._config = Config(config, read_only=True)

    @property
    def config(self) -> Config:
        """Initialization config."""
        return self._config

    def copy(self: ConfiguredT, **new_config) -> ConfiguredT:
        """Create a new instance based on the config.

        !!! warning
            This "copy" operation won't return a copy of the instance but a new instance
            initialized with the same config."""
        return self.__class__(**self.config.merge_with(new_config))

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
        return self.config.dumps(**kwargs)

    @classmethod
    def loads(cls: tp.Type[ConfiguredT], dumps: bytes, **kwargs) -> ConfiguredT:
        """Unpickle from bytes."""
        return cls(**Config.loads(dumps, **kwargs))

    def __eq__(self, other: tp.Any) -> bool:
        """Objects are equal if their configs are equal."""
        if type(self) != type(other):
            return False
        return self.config == other.config

    def getattr(self, attr_chain: tp.Union[str, tuple, list]) -> tp.Any:
        """See `vectorbt.utils.attr.deep_getattr`."""
        return deep_getattr(self, attr_chain)

    def update_config(self, *args, **kwargs) -> None:
        """Force-update the config."""
        self.config.update(*args, **kwargs, force_update=True)
