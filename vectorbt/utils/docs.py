"""Utilities for documentation."""
import json
import numpy as np

from vectorbt import _typing as tp


def prepare_for_docs(obj: tp.Any, replace: tp.DictLike = None, path: str = None) -> tp.Any:
    """Prepare object for use in documentation."""
    if isinstance(obj, np.dtype) and hasattr(obj, "fields"):
        return dict(zip(
            dict(obj.fields).keys(),
            list(map(lambda x: str(x[0]), dict(obj.fields).values()))
        ))
    if isinstance(obj, tuple) and hasattr(obj, "_asdict"):
        return prepare_for_docs(obj._asdict(), replace, path)
    if isinstance(obj, (tuple, list)):
        return [prepare_for_docs(v, replace, path) for v in obj]
    if isinstance(obj, dict):
        if replace is None:
            replace = {}
        new_obj = dict()
        for k, v in obj.items():
            if path is None:
                new_path = k
            else:
                new_path = path + '.' + k
            if new_path in replace:
                new_obj[k] = replace[new_path]
            else:
                new_obj[k] = prepare_for_docs(v, replace, new_path)
        return new_obj
    return obj


def to_doc(obj: tp.Any, replace: tp.DictLike = None, path: str = None, **kwargs) -> str:
    """Convert object to a JSON string."""
    kwargs = {**dict(indent=4, default=str), **kwargs}
    return json.dumps(prepare_for_docs(obj, replace, path), **kwargs)
