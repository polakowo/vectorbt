"""Enum utilities.

In vectorbt, enums are represented by instances of named tuples to be easily used in Numba."""

from vectorbt import typing as tp


def get_caseins_enum_attr(enum: tp.NamedTuple, attr: str) -> tp.Any:
    """Case-insensitive `getattr` for enums."""
    lower_attr_keys = list(map(lambda x: x.lower(), enum._fields))
    attr_idx = lower_attr_keys.index(attr.lower())
    orig_attr = enum._fields[attr_idx]
    return getattr(enum, orig_attr)


def prepare_enum_value(enum: tp.NamedTuple, value: tp.Any) -> tp.Any:
    """Prepare value of an enum.

    `enum` is expected to be an instance of `collections.namedtuple`.
    `value` can a string of any case or a tuple/list of such, otherwise returns unmodified value."""

    def _converter(x):
        if isinstance(x, str):
            return get_caseins_enum_attr(enum, str(x))
        return x

    if isinstance(value, str):
        # Convert str to int
        value = _converter(value)
    elif isinstance(value, (tuple, list)):
        # Convert each in the list
        value_type = type(value)
        value = list(value)
        for i in range(len(value)):
            if isinstance(value[i], (tuple, list)):
                value[i] = prepare_enum_value(enum, value[i])
            else:
                value[i] = _converter(value[i])
        return value_type(value)
    return value


def to_value_map(enum: tp.NamedTuple) -> dict:
    """Create value map from enum."""
    value_map = dict(zip(tuple(enum), enum._fields))
    if -1 not in value_map:
        value_map[-1] = None
    return value_map
