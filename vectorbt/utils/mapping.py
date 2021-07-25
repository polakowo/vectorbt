"""Mapping utilities."""

import numpy as np
import pandas as pd

from vectorbt import _typing as tp
from vectorbt.utils import checks


def reverse_mapping(mapping: tp.Mapping) -> dict:
    """Reverse a mapping.

    Returns a dict."""
    return {v: k for k, v in mapping.items()}


def to_mapping(mapping_like: tp.MappingLike, reverse: bool = False) -> dict:
    """Convert mapping-like object to a mapping.

    Enable `reverse` to apply `reverse_mapping` on the result dict."""
    if checks.is_namedtuple(mapping_like):
        mapping = {v: k for k, v in mapping_like._asdict().items()}
        if -1 not in mapping_like:
            mapping[-1] = None
    elif not checks.is_mapping(mapping_like):
        if checks.is_index(mapping_like):
            mapping_like = pd.Series(mapping_like)
        if checks.is_series(mapping_like):
            mapping = mapping_like.to_dict()
        else:
            mapping = dict(enumerate(mapping_like))
    else:
        mapping = dict(mapping_like)
    if reverse:
        mapping = reverse_mapping(mapping)
    return mapping


def apply_mapping(obj: tp.Any,
                  mapping_like: tp.Optional[tp.MappingLike] = None,
                  reverse: bool = False,
                  ignore_case: bool = True,
                  ignore_underscores: bool = True,
                  ignore_other_types: bool = True,
                  na_sentinel: tp.Any = None) -> tp.Any:
    """Apply mapping on object using a mapping-like object.

    Args:
        obj (any): Any object.

            Can take a scalar, tuple, list, set, frozenset, NumPy array, Index, Series, and DataFrame.
        mapping_like (mapping_like): Any mapping-like object.

            See `to_mapping`.
        reverse (bool): See `reverse` in `to_mapping`.
        ignore_case (bool): Whether to ignore the case if the key is a string.
        ignore_underscores (bool): Whether to ignore underscores if the key is a string.
        ignore_other_types (bool): Whether to prepare the object and the keys, for example, by lowering their case.
        na_sentinel (any): Value to mark “not found”.

    !!! note
        Casts array/Series/DataFrame if the first element is of the same type as the first mapping's key.
    """
    if mapping_like is None:
        return na_sentinel

    if ignore_case and ignore_underscores:
        key_func = lambda x: x.lower().replace('_', '')
    elif ignore_case:
        key_func = lambda x: x.lower()
    elif ignore_underscores:
        key_func = lambda x: x.replace('_', '')
    else:
        key_func = lambda x: x

    mapping = to_mapping(mapping_like, reverse=reverse)

    new_mapping = dict()
    for k, v in mapping.items():
        if pd.isnull(k):
            na_sentinel = v
        else:
            if isinstance(k, str):
                k = key_func(k)
            new_mapping[k] = v
    keys = list(new_mapping.keys())

    key = keys[0]
    if key is None:
        if len(new_mapping) > 1:
            key = keys[1]
        else:
            raise ValueError("Mapping keys contain only one value: null")

    def _same_type(x: tp.Any) -> bool:
        if type(x) == type(key):
            return True
        if np.issubdtype(type(x), type(key)):
            return True
        return False

    def _converter(x: tp.Any) -> tp.Any:
        if pd.isnull(x):
            return na_sentinel
        if isinstance(x, str):
            x = key_func(x)
        return new_mapping[x]

    if _same_type(obj):
        return _converter(obj)
    if isinstance(obj, (tuple, list, set, frozenset)):
        result = [apply_mapping(
            v,
            mapping_like=mapping_like,
            reverse=reverse,
            ignore_case=ignore_case,
            ignore_underscores=ignore_underscores,
            ignore_other_types=ignore_other_types,
            na_sentinel=na_sentinel
        ) for v in obj]
        return type(obj)(result)
    if isinstance(obj, np.ndarray):
        if obj.size == 0:
            return obj
        if _same_type(obj.item(0)):
            return np.vectorize(_converter)(obj)
    if isinstance(obj, pd.Series):
        if obj.size == 0:
            return obj
        if _same_type(obj.iloc[0]):
            return obj.map(_converter)
    if isinstance(obj, pd.Index):
        if obj.size == 0:
            return obj
        if _same_type(obj[0]):
            return obj.map(_converter)
    if isinstance(obj, pd.DataFrame):
        if obj.size == 0:
            return obj
        series = []
        for sr_name, sr in obj.iteritems():
            if _same_type(sr.iloc[0]):
                series.append(sr.map(_converter))
            else:
                if not ignore_other_types:
                    raise ValueError(f"Type of column '{sr_name}' is {type(obj)}, must be {type(key)}")
                series.append(sr)
        return pd.concat(series, axis=1, keys=obj.columns)
    if not ignore_other_types:
        raise ValueError(f"Type of object is {type(obj)}, must be {type(key)}")
    return obj
