"""Enum utilities.

In vectorbt, enums are represented by instances of named tuples to be easily used in Numba.
Their values start with 0, while -1 means there is no value."""

from vectorbt import _typing as tp
from vectorbt.utils.mapping import to_mapping, apply_mapping


def map_enum_fields(field: tp.Any, enum: tp.Enum, **kwargs) -> tp.Any:
    """Map fields to values.

    See `vectorbt.utils.mapping.apply_mapping`."""
    mapping = to_mapping(enum, reverse=True)

    return apply_mapping(field, mapping, **kwargs)


def map_enum_values(value: tp.Any, enum: tp.Enum, **kwargs) -> tp.Any:
    """Map values to fields.

    See `vectorbt.utils.mapping.apply_mapping`."""
    mapping = to_mapping(enum, reverse=False)

    return apply_mapping(value, mapping, **kwargs)
