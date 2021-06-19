"""Enum utilities.

In vectorbt, enums are represented by instances of named tuples to be easily used in Numba."""
import numpy as np
import pandas as pd

from vectorbt import _typing as tp


def enum_to_field_map(enum: tp.NamedTuple) -> tp.Dict[tp.Optional[str], int]:
    """Convert an enum to a field map."""
    field_map = {k.lower(): v for k, v in enum._asdict().items()}
    if -1 not in enum:
        field_map[None] = -1
    return field_map


def enum_to_value_map(enum: tp.NamedTuple) -> tp.Dict[int, tp.Optional[str]]:
    """Convert an enum to a value map."""
    value_map = {v: k for k, v in enum._asdict().items()}
    if -1 not in enum:
        value_map[-1] = None
    return value_map


def cast_enum_value(value: tp.Any, enum: tp.NamedTuple) -> tp.Any:
    """Cast string to an enum value.

    `enum` is expected to be an instance of `collections.namedtuple`.
    `value` can a string of any case and with any number of underscores, or a tuple/list/array of such,
    otherwise returns the unmodified value.

    !!! note
        Will only cast array/Series/DataFrame if the first element is a string.
    """
    field_map = enum_to_field_map(enum)

    def _converter(x):
        if isinstance(x, str):
            return field_map[x.lower().replace('_', '')]
        return x

    if isinstance(value, str):
        value = _converter(value)
    if isinstance(value, (tuple, list)):
        result = [cast_enum_value(v, enum) for v in value]
        if isinstance(value, tuple):
            result = tuple(result)
        return result
    if isinstance(value, np.ndarray):
        if value.size > 0 and isinstance(value.item(0), str):
            return np.vectorize(_converter)(value)
    if isinstance(value, pd.Series):
        if value.size > 0 and isinstance(value.iloc[0], str):
            return value.map(_converter)
    if isinstance(value, pd.DataFrame):
        if value.size > 0 and isinstance(value.iloc[0, 0], str):
            return value.applymap(_converter)
    return value
