import inspect
import re

import numpy as np
import pandas as pd
import pytest
from pandas.api.types import is_datetime64_any_dtype, is_object_dtype, is_string_dtype, is_timedelta64_dtype

from vectorbt import _engine


RUST_ENGINE_EXPR_RE = re.compile(r"engine\s*=\s*['\"]rust['\"]")
RUST_UNAVAILABLE_SKIP = pytest.mark.skip(reason="vectorbt-rust is not installed or version-compatible")

ORIGINAL_ASSERT_INDEX_EQUAL = pd.testing.assert_index_equal
ORIGINAL_ASSERT_SERIES_EQUAL = pd.testing.assert_series_equal
ORIGINAL_ASSERT_FRAME_EQUAL = pd.testing.assert_frame_equal


def normalize_test_index(index):
    if isinstance(index, pd.MultiIndex):
        levels = [normalize_test_index(level) for level in index.levels]
        return index.set_levels(levels)
    if is_datetime64_any_dtype(index.dtype):
        return pd.DatetimeIndex(index.to_numpy(dtype="datetime64[ns]"), name=index.name, freq=None)
    if is_timedelta64_dtype(index.dtype):
        return pd.TimedeltaIndex(index.to_numpy(dtype="timedelta64[ns]"), name=index.name, freq=None)
    if is_string_dtype(index.dtype) or (
        is_object_dtype(index.dtype)
        and all(isinstance(value, str) for value in index.dropna().to_numpy(dtype=object))
    ):
        values = [None if pd.isna(value) else value for value in index.astype(object)]
        return pd.Index(values, dtype=object, name=index.name)
    return index


def normalize_test_pandas(obj):
    if isinstance(obj, pd.Series):
        if is_datetime64_any_dtype(obj.dtype):
            obj = pd.Series(pd.DatetimeIndex(obj).as_unit("ns"), index=obj.index, name=obj.name, dtype="datetime64[ns]")
        elif is_timedelta64_dtype(obj.dtype):
            obj = pd.Series(pd.TimedeltaIndex(obj).as_unit("ns"), index=obj.index, name=obj.name, dtype="timedelta64[ns]")
        elif is_string_dtype(obj.dtype) or is_object_dtype(obj.dtype):
            obj = pd.Series(
                [np.nan if pd.isna(value) else value for value in obj.to_numpy(dtype=object)],
                index=obj.index,
                name=obj.name,
                dtype=object,
            )
        obj = obj.copy(deep=False)
        obj.index = normalize_test_index(obj.index)
        return obj
    elif isinstance(obj, pd.DataFrame) and any(
        is_datetime64_any_dtype(dtype) or is_timedelta64_dtype(dtype) or is_string_dtype(dtype) or is_object_dtype(dtype)
        for dtype in obj.dtypes
    ):
        columns = []
        for i, dtype in enumerate(obj.dtypes):
            column = obj.iloc[:, i]
            if is_datetime64_any_dtype(dtype):
                column = pd.Series(pd.DatetimeIndex(column).as_unit("ns").astype(object), dtype=object)
            elif is_timedelta64_dtype(dtype):
                column = pd.Series(pd.TimedeltaIndex(column).as_unit("ns").astype(object), dtype=object)
            elif is_string_dtype(dtype) or is_object_dtype(dtype):
                column = pd.Series(
                    [np.nan if pd.isna(value) else value for value in column.to_numpy(dtype=object)],
                    dtype=object,
                )
            columns.append(column.reset_index(drop=True))
        new_obj = pd.DataFrame(np.column_stack([column.to_numpy(dtype=object) for column in columns]), dtype=object)
        new_obj.index = obj.index
        new_obj.columns = obj.columns
        obj = new_obj
    if isinstance(obj, pd.DataFrame):
        obj = obj.copy(deep=False)
        obj.index = normalize_test_index(obj.index)
        obj.columns = normalize_test_index(obj.columns)
    return obj


def assert_index_equal_compat(left, right, *args, **kwargs):
    try:
        ORIGINAL_ASSERT_INDEX_EQUAL(left, right, *args, **kwargs)
    except AssertionError:
        ORIGINAL_ASSERT_INDEX_EQUAL(normalize_test_index(left), normalize_test_index(right), *args, **kwargs)


def assert_series_equal_compat(left, right, *args, **kwargs):
    try:
        ORIGINAL_ASSERT_SERIES_EQUAL(left, right, *args, **kwargs)
    except AssertionError:
        if "rtol" in kwargs or "atol" in kwargs:
            raise
        ORIGINAL_ASSERT_SERIES_EQUAL(normalize_test_pandas(left), normalize_test_pandas(right), *args, **kwargs)


def assert_frame_equal_compat(left, right, *args, **kwargs):
    try:
        ORIGINAL_ASSERT_FRAME_EQUAL(left, right, *args, **kwargs)
    except AssertionError:
        if "rtol" in kwargs or "atol" in kwargs:
            raise
        ORIGINAL_ASSERT_FRAME_EQUAL(normalize_test_pandas(left), normalize_test_pandas(right), *args, **kwargs)


assert_index_equal_compat.__signature__ = inspect.signature(ORIGINAL_ASSERT_INDEX_EQUAL)
assert_series_equal_compat.__signature__ = inspect.signature(ORIGINAL_ASSERT_SERIES_EQUAL)
assert_frame_equal_compat.__signature__ = inspect.signature(ORIGINAL_ASSERT_FRAME_EQUAL)

pd.testing.assert_index_equal = assert_index_equal_compat
pd.testing.assert_series_equal = assert_series_equal_compat
pd.testing.assert_frame_equal = assert_frame_equal_compat


def pytest_collection_modifyitems(items):
    if _engine.is_rust_available():
        return

    for item in items:
        try:
            source = inspect.getsource(item.obj)
        except (OSError, TypeError):
            continue
        if RUST_ENGINE_EXPR_RE.search(source):
            item.add_marker(RUST_UNAVAILABLE_SKIP)
