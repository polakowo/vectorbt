"""Class and function decorators."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt.utils.config import merge_dicts, Config
from vectorbt.utils.decorators import cached_property
from vectorbt.utils.mapping import to_mapping
from vectorbt.records.mapped_array import MappedArray

WrapperFuncT = tp.Callable[[tp.Type[tp.T]], tp.Type[tp.T]]


def add_mapped_fields(dtype: np.dtype, config: tp.Optional[Config] = None, on_conflict: str = 'raise') -> WrapperFuncT:
    """Class decorator to override mapped field properties in a `vectorbt.records.base.Records` class.

    `config` should contain fields (keys) and dictionaries (values) with the following keys:

    * `target_name`: Name of the target property. Defaults to the field name. Set to None to disable.
    * `defaults`: Dictionary with defaults for `vectorbt.records.base.Records.map_field`.

    If a field name is not in the config, it will be added automatically.

    If an attribute with the same name already exists in the class:

    * it will be overridden if `on_conflict` is 'override'
    * it will be ignored if `on_conflict` is 'ignore'
    * an error will be raised if `on_conflict` is 'raise'
    """
    if config is None:
        config = {}

    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        from vectorbt.records.base import Records

        checks.assert_subclass(cls, Records)
        checks.assert_not_none(dtype.fields)

        for field_name in dtype.names:
            if field_name in config:
                settings = config[field_name]
                target_name = settings.get('target_name', field_name)
                defaults = settings.get('defaults', None)
            else:
                target_name = field_name
                defaults = None
            if hasattr(cls, target_name):
                if on_conflict.lower() == 'raise':
                    raise ValueError(f"An attribute with the name '{target_name}' already exists in {cls}")
                elif on_conflict.lower() == 'ignore':
                    continue
                elif on_conflict.lower() == 'override':
                    pass
                else:
                    raise ValueError(f"Value '{on_conflict}' is invalid for on_conflict")
            if not target_name.isidentifier():
                raise ValueError(f"Name '{target_name}' is not a valid attribute name")

            def new_prop(self, _field_name: str = field_name, _defaults: tp.KwargsLike = defaults) -> MappedArray:
                return self.map_field(_field_name, **merge_dicts({}, _defaults))

            new_prop.__doc__ = f"Mapped array of the field `{field_name}`."
            new_prop.__name__ = target_name
            setattr(cls, target_name, cached_property(new_prop))
        return cls

    return wrapper


def attach_fields(config: tp.Optional[Config] = None, on_conflict: str = 'raise') -> WrapperFuncT:
    def wrapper(cls: tp.Type[tp.T]) -> tp.Type[tp.T]:
        from vectorbt.records.base import Records, RecordsT

        checks.assert_subclass(cls, Records)

        dtype = cls.field_config.get('dtype', None)
        field_settings = cls.field_config.get('field_settings', {})

        if dtype is not None:
            for field_name in dtype.names:
                # Check target name
                target_name = field_settings.get(field_name, {}).get('rename_to', field_name)
                if not target_name.isidentifier():
                    raise ValueError(f"Name '{target_name}' is not a valid attribute name")
                if hasattr(cls, target_name):
                    if on_conflict.lower() == 'raise':
                        raise ValueError(f"An attribute with the name '{target_name}' already exists in {cls}")
                    elif on_conflict.lower() == 'ignore':
                        continue
                    elif on_conflict.lower() == 'override':
                        pass
                    else:
                        raise ValueError(f"Value '{on_conflict}' is invalid for on_conflict")

                # Add mapped field property
                mapping = field_settings.get(field_name, {}).get('mapping', None)
                attach_mapped_field = field_settings.get(field_name, {}).get('attach_mapped_field', True)

                if attach_mapped_field:
                    def mapped_field_prop(self,
                                          _field_name: str = field_name,
                                          _mapping: tp.MappingLike = mapping,
                                          **kwargs) -> MappedArray:
                        return self.map_field(
                            _field_name,
                            mapping=_mapping,
                            **kwargs
                        )

                    mapped_field_prop.__doc__ = f"Mapped array of the field `{field_name}`."
                    mapped_field_prop.__name__ = target_name
                    setattr(cls, target_name, cached_property(mapped_field_prop))

                # Add filter properties
                attach_filters = field_settings.get(field_name, {}).get('attach_filters', False)

                if (isinstance(attach_filters, bool) and attach_filters) or \
                        (not isinstance(attach_filters, bool) and attach_filters is not None):
                    if isinstance(attach_filters, bool):
                        if mapping is None:
                            raise ValueError(f"Field '{field_name}': Mapping is required to add filters")
                        filter_mapping = to_mapping(mapping, reverse=True)
                    else:
                        filter_mapping = to_mapping(attach_filters, reverse=True)

                    for filter_branch, filter_value in filter_mapping.items():
                        if not isinstance(filter_branch, str):
                            raise TypeError(f"Field '{field_name}': Each filter name must be a string. "
                                            f"Received '{filter_branch}'.")
                        filter_branch = filter_branch.lower()
                        if not checks.is_valid_variable_name(filter_branch):
                            raise ValueError(f"Field '{field_name}': Each filter name must be a valid variable name. "
                                             f"Received '{filter_branch}'.")

                        def filter_prop(self: RecordsT,
                                        _field_name: str = field_name,
                                        _filter_value: tp.Any = filter_value,
                                        **kwargs) -> RecordsT:
                            filter_mask = self.values[_field_name] == _filter_value
                            return self.filter_by_mask(filter_mask, **kwargs)

                        filter_prop.__doc__ = f"Filtered records using `{field_name} == {filter_value}`."
                        filter_prop.__name__ = filter_branch
                        setattr(cls, filter_branch, cached_property(filter_prop))
            return cls

    return wrapper
