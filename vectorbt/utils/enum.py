"""Enum utilities."""

import numpy as np
import pandas as pd

from vectorbt.utils.checks import assert_in, assert_type
import warnings


def caseins_getattr(enum, attr):
    """Case-insensitive `getattr` for enumerated types."""
    lower_attr_keys = list(map(lambda x: x.lower(), enum._fields))
    attr_idx = lower_attr_keys.index(attr.lower())
    orig_attr = enum._fields[attr_idx]
    return getattr(enum, orig_attr)


def convert_str_enum_value(enum, value):
    """Converts any enumerated value of type string into integer.

    `enum` is expected to be an instance of `collections.namedtuple`.
    `value` can a string of any case, or a tuple/list of such."""

    def _converter(x):
        if isinstance(x, str):
            return caseins_getattr(enum, str(x))
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
                value[i] = convert_str_enum_value(enum, value[i])
            else:
                value[i] = _converter(value[i])
        return value_type(value)
    return value
