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
        ignore_other_types (bool): Whether to ignore other data types. Otherwise, throws an error.
        na_sentinel (any): Value to mark “not found”.
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

    key_types = set()
    new_mapping = dict()
    for k, v in mapping.items():
        if pd.isnull(k):
            na_sentinel = v
        else:
            if isinstance(k, str):
                k = key_func(k)
            new_mapping[k] = v
            key_types.add(type(k))

    def _type_in_key_types(x_type: type) -> bool:
        for key_type in key_types:
            if x_type is key_type:
                return True
            x_dtype = np.dtype(x_type)
            key_dtype = np.dtype(key_type)
            if x_dtype is key_dtype:
                return True
            if np.issubdtype(x_dtype, np.number) and np.issubdtype(key_dtype, np.number):
                return True
            if np.issubdtype(x_dtype, np.bool_) and np.issubdtype(key_dtype, np.bool_):
                return True
            if np.issubdtype(x_dtype, np.flexible) and np.issubdtype(key_dtype, np.flexible):
                return True
        return False

    def _converter(x: tp.Any) -> tp.Any:
        if pd.isnull(x):
            return na_sentinel
        if isinstance(x, str):
            x = key_func(x)
        return new_mapping[x]

    if _type_in_key_types(type(obj)):
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
        if _type_in_key_types(type(obj.item(0))):
            return np.vectorize(_converter)(obj)
        if not ignore_other_types:
            raise ValueError(f"Type is {type(obj.item(0))}, must be one of types {key_types}")
        return obj
    if isinstance(obj, pd.Series):
        if obj.size == 0:
            return obj
        if _type_in_key_types(type(obj.iloc[0])):
            return obj.map(_converter)
        if not ignore_other_types:
            raise ValueError(f"Type is {type(obj.iloc[0])}, must be one of types {key_types}")
        return obj
    if isinstance(obj, pd.Index):
        if obj.size == 0:
            return obj
        if _type_in_key_types(type(obj[0])):
            return obj.map(_converter)
        if not ignore_other_types:
            raise ValueError(f"Type is {type(obj[0])}, must be one of types {key_types}")
        return obj
    if isinstance(obj, pd.DataFrame):
        if obj.size == 0:
            return obj
        series = []
        for sr_name, sr in obj.iteritems():
            if _type_in_key_types(type(sr.iloc[0])):
                series.append(sr.map(_converter))
            else:
                if not ignore_other_types:
                    raise ValueError(f"Type is {type(sr.iloc[0])}, must be one of types {key_types}")
                series.append(sr)
        return pd.concat(series, axis=1, keys=obj.columns)
    if not ignore_other_types:
        raise ValueError(f"Type is {type(obj)}, must be one of types {key_types}")
    return obj
