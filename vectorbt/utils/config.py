"""Utilities for configuration."""

from copy import copy, deepcopy
from collections import namedtuple
import dill
import inspect

from vectorbt import _typing as tp
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
    """Get keyword arguments with defaults of a function."""
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }


def get_func_arg_names(func: tp.Callable) -> tp.List[str]:
    """Get argument names of a function."""
    signature = inspect.signature(func)
    return [
        p.name for p in signature.parameters.values()
        if p.kind != p.VAR_POSITIONAL and p.kind != p.VAR_KEYWORD
    ]


class atomic_dict(dict):
    """Dict that behaves like a single value when merging."""
    pass


InConfigLikeT = tp.Union[None, dict, "ConfigT"]
OutConfigLikeT = tp.Union[dict, "ConfigT"]


def convert_to_dict(dct: InConfigLikeT, nested: bool = True) -> dict:
    """Convert any dict (apart from `atomic_dict`) to `dict`.

    Set `nested` to True to convert all child dicts in recursive manner."""
    if dct is None:
        dct = {}
    if isinstance(dct, atomic_dict):
        dct = atomic_dict(dct)
    else:
        dct = dict(dct)
    if not nested:
        return dct
    for k, v in dct.items():
        if isinstance(v, dict):
            dct[k] = convert_to_dict(v, nested=nested)
        else:
            dct[k] = v
    return dct


def set_dict_item(dct: dict, k: tp.Any, v: tp.Any, force: bool = False) -> None:
    """Set dict item.

    If the dict is of the type `Config`, also passes `force` keyword to override blocking flags."""
    if isinstance(dct, Config):
        dct.__setitem__(k, v, force=force)
    else:
        dct[k] = v


def copy_dict(dct: InConfigLikeT, copy_mode: str = 'shallow', nested: bool = True) -> OutConfigLikeT:
    """Copy dict based on a copy mode.

    The following modes are supported:

    * 'shallow': Copies keys only.
    * 'hybrid': Copies keys and values using `copy.copy`.
    * 'deep': Copies the whole thing using `copy.deepcopy`.

    Set `nested` to True to copy all child dicts in recursive manner."""
    if dct is None:
        dct = {}
    checks.assert_type(copy_mode, str)
    copy_mode = copy_mode.lower()
    if copy_mode not in ['shallow', 'hybrid', 'deep']:
        raise ValueError(f"Copy mode '{copy_mode}' not supported")

    if copy_mode == 'deep':
        return deepcopy(dct)
    if isinstance(dct, Config):
        return dct.copy(
            copy_mode=copy_mode,
            nested=nested
        )
    dct_copy = copy(dct)  # copy structure using shallow copy
    for k, v in dct_copy.items():
        if nested and isinstance(v, dict):
            _v = copy_dict(v, copy_mode=copy_mode, nested=nested)
        else:
            if copy_mode == 'hybrid':
                _v = copy(v)  # copy values using shallow copy
            else:
                _v = v
        set_dict_item(dct_copy, k, _v, force=True)
    return dct_copy


def update_dict(x: InConfigLikeT, y: InConfigLikeT, nested: bool = True, force: bool = False) -> None:
    """Update dict with keys and values from other dict.

    Set `nested` to True to update all child dicts in recursive manner.
    For `force`, see `set_dict_item`.

    If you want to treat any dict as a single value, wrap it with `atomic_dict`.

    !!! note
        If the child dict is not atomic, it will copy only its values, not its meta."""
    if x is None:
        return
    if y is None:
        return
    checks.assert_type(x, dict)
    checks.assert_type(y, dict)

    for k, v in y.items():
        if nested \
                and k in x \
                and isinstance(x[k], dict) \
                and isinstance(v, dict) \
                and not isinstance(v, atomic_dict):
            update_dict(x[k], v, force=force)
        else:
            set_dict_item(x, k, v, force=force)


def merge_dicts(*dicts: InConfigLikeT,
                to_dict: bool = True,
                copy_mode: tp.Optional[str] = 'shallow',
                nested: bool = True) -> OutConfigLikeT:
    """Merge dicts.

    Args:
        *dicts (dict): Dicts.
        to_dict: Whether to call `convert_to_dict` on each dict prior to copying.
        copy_mode: Mode for `copy_dict` to copy each dict prior to merging.

            Pass None to not copy.
        nested: Whether to merge all child dicts in recursive manner."""
    # copy only once
    if to_dict:
        dicts = tuple([convert_to_dict(dct, nested=nested) for dct in dicts])
    if copy_mode is not None:
        dicts = tuple([copy_dict(dct, copy_mode=copy_mode, nested=nested) for dct in dicts])
    x, y = dicts[0], dicts[1]
    if isinstance(x, atomic_dict) or isinstance(y, atomic_dict):
        x = y
    else:
        update_dict(x, y, nested=nested, force=True)
    if len(dicts) > 2:
        return merge_dicts(
            x, *dicts[2:],
            to_dict=False,  # executed only once
            copy_mode=None,  # executed only once
            nested=nested
        )
    return x


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

    def save(self, fname: tp.FileName, **kwargs) -> None:
        """Save dumps to a file."""
        dumps = self.dumps(**kwargs)
        with open(fname, "wb") as f:
            f.write(dumps)

    @classmethod
    def load(cls: tp.Type[PickleableT], fname: tp.FileName, **kwargs) -> PickleableT:
        """Load dumps from a file and create new instance."""
        with open(fname, "rb") as f:
            dumps = f.read()
        return cls.loads(dumps, **kwargs)


PickleableDictT = tp.TypeVar("PickleableDictT", bound="PickleableDict")


class PickleableDict(Pickleable, dict):
    """Dict that may contain values of type `Pickleable`."""

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
        dct = dict()
        for k, v in self.items():
            if isinstance(v, Pickleable):
                dct[k] = DumpTuple(cls=v.__class__, dumps=v.dumps(**kwargs))
            else:
                dct[k] = v
        return dill.dumps(dct, **kwargs)

    @classmethod
    def loads(cls: tp.Type[PickleableDictT], dumps: bytes, **kwargs) -> PickleableDictT:
        """Unpickle from bytes."""
        config = dill.loads(dumps, **kwargs)
        for k, v in config.items():
            if isinstance(v, DumpTuple):
                config[k] = v.cls.loads(v.dumps, **kwargs)
        return cls(**config)

    def load_update(self, fname: tp.FileName, **kwargs) -> None:
        """Load dumps from a file and update this instance."""
        self.clear()
        self.update(self.load(fname, **kwargs))


ConfigT = tp.TypeVar("ConfigT", bound="Config")


class Config(PickleableDict):
    """Extends dict with config features such as nested updates, frozen keys/values, and pickling.

    Args:
        dct (dict): Dict to construct this config from.
        copy_kwargs (dict): Keyword arguments passed to `copy_dict` for copying `dct` and `reset_dct`.

            Copy mode defaults to 'shallow' if `readonly`, otherwise to 'hybrid'.
        reset_dct (dict): Dict to fall back to in case of resetting.

            If None, copies `dct` using `reset_dct_copy_kwargs`.
        reset_dct_copy_kwargs (dict): Keyword arguments that override `copy_kwargs` for `reset_dct`.
        frozen_keys (bool): Whether to deny updates to the keys of the config.

            Defaults to False.
        readonly (bool): Whether to deny updates to the keys and values of the config.

            Defaults to False.
        nested (bool): Whether to do operations recursively on each child dict.

            Such operations include copy, update, and merge.
            Disable to treat each child dict as a single value. Defaults to True.
        convert_dicts (bool or type): Whether to convert child dicts to configs with the same configuration.

            This will trigger a waterfall reaction across all child dicts.
            Won't convert dicts that are already configs.
            Apart from boolean, you can set it to any subclass of `Config` to use it for construction.
            Requires `nested` to be True. Defaults to False.
        as_attrs (bool): Whether to enable accessing dict keys via the dot notation.

            Enables autocompletion (but only during runtime!).
            Raises error in case of naming conflicts.
            Defaults to True if `frozen` or `readonly`, otherwise False.

    Defaults can be overridden with settings under `config` in `vectorbt._settings.settings`.

    If another config is passed, its properties are copied over, but they can still be overridden
    with the arguments passed to the initializer.

    !!! note
        All arguments are applied only once during initialization.
    """

    _copy_kwargs_: tp.Kwargs
    _reset_dct_: dict
    _reset_dct_copy_kwargs_: tp.Kwargs
    _frozen_keys_: bool
    _readonly_: bool
    _nested_: bool
    _convert_dicts_: tp.Union[bool, tp.Type["Config"]]
    _as_attrs_: bool

    def __init__(self,
                 dct: tp.DictLike = None,
                 copy_kwargs: tp.KwargsLike = None,
                 reset_dct: tp.DictLike = None,
                 reset_dct_copy_kwargs: tp.KwargsLike = None,
                 frozen_keys: tp.Optional[bool] = None,
                 readonly: tp.Optional[bool] = None,
                 nested: tp.Optional[bool] = None,
                 convert_dicts: tp.Optional[tp.Union[bool, tp.Type["Config"]]] = None,
                 as_attrs: tp.Optional[bool] = None) -> None:
        try:
            from vectorbt._settings import settings
            configured_cfg = settings['config']
        except ImportError:
            configured_cfg = {}

        if dct is None:
            dct = dict()

        # Resolve params
        def _resolve_param(pname: str, p: tp.Any, default: tp.Any, merge: bool = False) -> tp.Any:
            cfg_default = configured_cfg.get(pname, None)
            dct_p = getattr(dct, pname + '_') if isinstance(dct, Config) else None

            if merge and isinstance(default, dict):
                return merge_dicts(default, cfg_default, dct_p, p)
            if p is not None:
                return p
            if dct_p is not None:
                return dct_p
            if cfg_default is not None:
                return cfg_default
            return default

        reset_dct = _resolve_param('reset_dct', reset_dct, None)
        frozen_keys = _resolve_param('frozen_keys', frozen_keys, False)
        readonly = _resolve_param('readonly', readonly, False)
        nested = _resolve_param('nested', nested, False)
        convert_dicts = _resolve_param('convert_dicts', convert_dicts, False)
        as_attrs = _resolve_param('as_attrs', as_attrs, frozen_keys or readonly)
        reset_dct_copy_kwargs = merge_dicts(copy_kwargs, reset_dct_copy_kwargs)
        copy_kwargs = _resolve_param(
            'copy_kwargs',
            copy_kwargs,
            dict(
                copy_mode='shallow' if readonly else 'hybrid',
                nested=nested
            ),
            merge=True
        )
        reset_dct_copy_kwargs = _resolve_param(
            'reset_dct_copy_kwargs',
            reset_dct_copy_kwargs,
            dict(
                copy_mode='shallow' if readonly else 'hybrid',
                nested=nested
            ),
            merge=True
        )

        # Copy dict
        dct = copy_dict(dict(dct), **copy_kwargs)

        # Convert child dicts
        if convert_dicts:
            if not nested:
                raise ValueError("convert_dicts requires nested to be True")
            for k, v in dct.items():
                if isinstance(v, dict) and not isinstance(v, Config):
                    if isinstance(convert_dicts, bool):
                        config_cls = self.__class__
                    elif issubclass(convert_dicts, Config):
                        config_cls = convert_dicts
                    else:
                        raise TypeError("convert_dicts must be either boolean or a subclass of Config")
                    dct[k] = config_cls(
                        v,
                        copy_kwargs=copy_kwargs,
                        reset_dct_copy_kwargs=reset_dct_copy_kwargs,
                        frozen_keys=frozen_keys,
                        readonly=readonly,
                        nested=nested,
                        convert_dicts=convert_dicts,
                        as_attrs=as_attrs
                    )

        # Copy initial config
        if reset_dct is None:
            reset_dct = dct
        reset_dct = copy_dict(dict(reset_dct), **reset_dct_copy_kwargs)

        dict.__init__(self, dct)

        # Store params in an instance variable
        checks.assert_type(copy_kwargs, dict)
        checks.assert_type(reset_dct, dict)
        checks.assert_type(reset_dct_copy_kwargs, dict)
        checks.assert_type(frozen_keys, bool)
        checks.assert_type(readonly, bool)
        checks.assert_type(nested, bool)
        checks.assert_type(convert_dicts, (bool, type))
        checks.assert_type(as_attrs, bool)

        self.__dict__['_copy_kwargs_'] = copy_kwargs
        self.__dict__['_reset_dct_'] = reset_dct
        self.__dict__['_reset_dct_copy_kwargs_'] = reset_dct_copy_kwargs
        self.__dict__['_frozen_keys_'] = frozen_keys
        self.__dict__['_readonly_'] = readonly
        self.__dict__['_nested_'] = nested
        self.__dict__['_convert_dicts_'] = convert_dicts
        self.__dict__['_as_attrs_'] = as_attrs

        # Set keys as attributes for autocomplete
        if as_attrs:
            for k, v in self.items():
                if k in self.__dir__():
                    raise ValueError(f"Cannot set key '{k}' as attribute of the config. Disable set_attrs.")
                self.__dict__[k] = v

    @property
    def copy_kwargs_(self) -> tp.Kwargs:
        """Parameters for copying `dct`."""
        return self._copy_kwargs_

    @property
    def reset_dct_(self) -> dict:
        """Dict to fall back to in case of resetting."""
        return self._reset_dct_

    @property
    def reset_dct_copy_kwargs_(self) -> tp.Kwargs:
        """Parameters for copying `reset_dct`."""
        return self._reset_dct_copy_kwargs_

    @property
    def frozen_keys_(self) -> bool:
        """Whether to deny updates to the keys and values of the config."""
        return self._frozen_keys_

    @property
    def readonly_(self) -> bool:
        """Whether to deny any updates to the config."""
        return self._readonly_

    @property
    def nested_(self) -> bool:
        """Whether to do operations recursively on each child dict."""
        return self._nested_

    @property
    def convert_dicts_(self) -> tp.Union[bool, tp.Type["Config"]]:
        """Whether to convert child dicts to configs with the same configuration."""
        return self._convert_dicts_

    @property
    def as_attrs_(self) -> bool:
        """Whether to enable accessing dict keys via dot notation."""
        return self._as_attrs_

    def __setattr__(self, k: str, v: tp.Any) -> None:
        if self.as_attrs_:
            self.__setitem__(k, v)

    def __setitem__(self, k: str, v: tp.Any, force: bool = False) -> None:
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            if k not in self:
                raise KeyError(f"Config keys are frozen: key '{k}' not found")
        dict.__setitem__(self, k, v)
        if self.as_attrs_:
            self.__dict__[k] = v

    def __delattr__(self, k: str) -> None:
        if self.as_attrs_:
            self.__delitem__(k)

    def __delitem__(self, k: str, force: bool = False) -> None:
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            raise KeyError(f"Config keys are frozen")
        dict.__delitem__(self, k)
        if self.as_attrs_:
            del self.__dict__[k]

    def _clear_attrs(self, prior_keys: tp.Iterable[str]) -> None:
        """Remove attributes of the removed keys given keys prior to the removal."""
        if self.as_attrs_:
            for k in set(prior_keys).difference(self.keys()):
                del self.__dict__[k]

    def pop(self, k: str, v: tp.Any = _RaiseKeyError, force: bool = False) -> tp.Any:
        """Remove and return the pair by the key."""
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            raise KeyError(f"Config keys are frozen")
        prior_keys = list(self.keys())
        if v is _RaiseKeyError:
            result = dict.pop(self, k)
        else:
            result = dict.pop(self, k, v)
        self._clear_attrs(prior_keys)
        return result

    def popitem(self, force: bool = False) -> tp.Tuple[tp.Any, tp.Any]:
        """Remove and return some pair."""
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            raise KeyError(f"Config keys are frozen")
        prior_keys = list(self.keys())
        result = dict.popitem(self)
        self._clear_attrs(prior_keys)
        return result

    def clear(self, force: bool = False) -> None:
        """Remove all items."""
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        if not force and self.frozen_keys_:
            raise KeyError(f"Config keys are frozen")
        prior_keys = list(self.keys())
        dict.clear(self)
        self._clear_attrs(prior_keys)

    def update(self, *args, nested: tp.Optional[bool] = None, force: bool = False, **kwargs) -> None:
        """Update the config.

        See `update_dict`."""
        other = dict(*args, **kwargs)
        if nested is None:
            nested = self.nested_
        update_dict(self, other, nested=nested, force=force)

    def __copy__(self: ConfigT) -> ConfigT:
        """Shallow operation, primarily used by `copy.copy`.

        Does not take into account copy parameters."""
        cls = self.__class__
        self_copy = cls.__new__(cls)
        for k, v in self.__dict__.items():
            if k not in self_copy:  # otherwise copies dict keys twice
                self_copy.__dict__[k] = v
        self_copy.clear(force=True)
        self_copy.update(copy(dict(self)), nested=False, force=True)
        return self_copy

    def __deepcopy__(self: ConfigT, memo: tp.DictLike = None) -> ConfigT:
        """Deep operation, primarily used by `copy.deepcopy`.

        Does not take into account copy parameters."""
        if memo is None:
            memo = {}
        cls = self.__class__
        self_copy = cls.__new__(cls)
        memo[id(self)] = self_copy
        for k, v in self.__dict__.items():
            if k not in self_copy:  # otherwise copies dict keys twice
                self_copy.__dict__[k] = deepcopy(v, memo)
        self_copy.clear(force=True)
        self_copy.update(deepcopy(dict(self), memo), nested=False, force=True)
        return self_copy

    def copy(self: ConfigT, reset_dct_copy_kwargs: tp.KwargsLike = None, **copy_kwargs) -> ConfigT:
        """Copy the instance in the same way it's done during initialization.

        `copy_kwargs` override `Config.copy_kwargs_` and `Config.reset_dct_copy_kwargs_` via merging.
        `reset_dct_copy_kwargs` override merged `Config.reset_dct_copy_kwargs_`."""
        self_copy = self.__copy__()

        reset_dct_copy_kwargs = merge_dicts(self.reset_dct_copy_kwargs_, copy_kwargs, reset_dct_copy_kwargs)
        reset_dct = copy_dict(dict(self.reset_dct_), **reset_dct_copy_kwargs)
        self.__dict__['_reset_dct_'] = reset_dct

        copy_kwargs = merge_dicts(self.copy_kwargs_, copy_kwargs)
        dct = copy_dict(dict(self), **copy_kwargs)
        self_copy.update(dct, nested=False, force=True)

        return self_copy

    def merge_with(self: ConfigT, other: InConfigLikeT,
                   nested: tp.Optional[bool] = None, **kwargs) -> OutConfigLikeT:
        """Merge with another dict into one single dict.

        See `merge_dicts`."""
        if nested is None:
            nested = self.nested_
        return merge_dicts(self, other, nested=nested, **kwargs)

    def reset(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """Clears the config and updates it with the initial config.

        `reset_dct_copy_kwargs` override `Config.reset_dct_copy_kwargs_`."""
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(self.reset_dct_copy_kwargs_, reset_dct_copy_kwargs)
        reset_dct = copy_dict(dict(self.reset_dct_), **reset_dct_copy_kwargs)
        self.clear(force=True)
        self.update(self.reset_dct_, nested=False, force=True)
        self.__dict__['_reset_dct_'] = reset_dct

    def make_checkpoint(self, force: bool = False, **reset_dct_copy_kwargs) -> None:
        """Replace `reset_dct` by the current state.

        `reset_dct_copy_kwargs` override `Config.reset_dct_copy_kwargs_`."""
        if not force and self.readonly_:
            raise TypeError("Config is read-only")
        reset_dct_copy_kwargs = merge_dicts(self.reset_dct_copy_kwargs_, reset_dct_copy_kwargs)
        reset_dct = copy_dict(dict(self), **reset_dct_copy_kwargs)
        self.__dict__['_reset_dct_'] = reset_dct

    def dumps(self, **kwargs) -> bytes:
        """Pickle to bytes."""
        return dill.dumps(dict(
            dct=PickleableDict(self).dumps(**kwargs),
            copy_kwargs=self.copy_kwargs_,
            reset_dct=PickleableDict(self.reset_dct_).dumps(**kwargs),
            reset_dct_copy_kwargs=self.reset_dct_copy_kwargs_,
            frozen_keys=self.frozen_keys_,
            readonly=self.readonly_,
            nested=self.nested_,
            convert_dicts=self.convert_dicts_,
            as_attrs=self.as_attrs_
        ), **kwargs)

    @classmethod
    def loads(cls: tp.Type[ConfigT], dumps: bytes, **kwargs) -> ConfigT:
        """Unpickle from bytes."""
        obj = dill.loads(dumps, **kwargs)
        return cls(
            dct=PickleableDict.loads(obj['dct'], **kwargs),
            copy_kwargs=obj['copy_kwargs'],
            reset_dct=PickleableDict.loads(obj['reset_dct'], **kwargs),
            reset_dct_copy_kwargs=obj['reset_dct_copy_kwargs'],
            frozen_keys=obj['frozen_keys'],
            readonly=obj['readonly'],
            nested=obj['nested'],
            convert_dicts=obj['convert_dicts'],
            as_attrs=obj['as_attrs']
        )

    def load_update(self, fname: tp.FileName, **kwargs) -> None:
        """Load dumps from a file and update this instance.

        !!! note
            Updates both the config properties and dictionary."""
        loaded = self.load(fname, **kwargs)
        self.clear(force=True)
        self.__dict__.clear()
        self.__dict__.update(loaded.__dict__)
        self.update(loaded, nested=False, force=True)

    def __eq__(self, other: tp.Any) -> bool:
        return checks.is_deep_equal(dict(self), dict(other))


class AtomicConfig(Config, atomic_dict):
    """Config that behaves like a single value when merging."""
    pass


ConfiguredT = tp.TypeVar("ConfiguredT", bound="Configured")


class Configured(Pickleable):
    """Class with an initialization config.

    All operations are done using config rather than the instance, which makes it easier to pickle.

    Config settings are defined under `config.configured` in `vectorbt._settings.settings`.

    !!! warning
        If the instance has writable attributes or depends upon global defaults,
        their values won't be copied over. Make sure to pass them explicitly to
        make the saved & loaded / copied instance resilient to changes in globals."""

    def __init__(self, **config) -> None:
        from vectorbt._settings import settings
        configured_cfg = settings['config']['configured']

        self._config = Config(config, **configured_cfg)

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

    def getattr(self, *args, **kwargs) -> tp.Any:
        """See `vectorbt.utils.attr.deep_getattr`."""
        return deep_getattr(self, *args, **kwargs)

    def update_config(self, *args, **kwargs) -> None:
        """Force-update the config."""
        self.config.update(*args, **kwargs, force=True)
