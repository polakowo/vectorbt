"""Utilities for type hints."""

import sys
import numpy as np
import pandas as pd
from pandas import Series, DataFrame as Frame, Index
from typing import *
from numpy.typing import ArrayLike as _ArrayLike, DTypeLike
from pandas._typing import Dtype as PandasDTypeLike
from pandas.core.arrays.base import ExtensionArray
from datetime import datetime, timedelta, tzinfo
from mypy_extensions import VarArg, KwArg
from pandas.tseries.offsets import DateOffset

if sys.version_info < (3, 8, 0):
    from typing_extensions import *

T = TypeVar("T")
"""Generic type."""

F = TypeVar("F", bound=Callable[..., Any])
"""Generic function."""

Func = Callable[..., Any]
"""Any callable."""

NumbaFunc = Callable[..., Any]
"""Any Numba-compiled callable."""

MaybeTuple = Union[T, Tuple[T, ...]]
"""Value or tuple of values of the same type."""

MaybeList = Union[T, List[T]]
"""Value or list of values of the same type."""

TupleList = Union[List[T], Tuple[T, ...]]
"""Tuple/list of values of the same type."""

MaybeTupleList = Union[T, List[T], Tuple[T, ...]]
"""Value or tuple/list of values of the same type."""

Array = np.ndarray
"""NumPy array."""

Array1d = np.ndarray
"""NumPy array (1 dimension)."""

Array2d = np.ndarray
"""NumPy array (2 dimensions)."""

Array3d = np.ndarray
"""NumPy array (3 dimensions)."""

Scalar = Union[str, float, int, complex, bool, object, np.generic]
"""Any scalar."""

Number = Union[int, float, complex, np.number, np.bool_]
"""Any number scalar."""

SNumber = Union[int, float, np.number]
"""Any simple number scalar."""

MaybeArray = Union[Scalar, Array]
"""Scalar or NumPy array."""

MaybeArray1d = Union[Scalar, Array1d]
"""Scalar or NumPy array (1 dimension)."""

MaybeNumberArray = Union[Number, Array]
"""Number or NumPy array."""

MaybeNumberArray1d = Union[Number, Array1d]
"""Number or NumPy array (1 dimension)."""

MaybeSNumberArray = Union[SNumber, Array]
"""Simple number or NumPy array."""

MaybeSNumberArray1d = Union[SNumber, Array1d]
"""Simple number or NumPy array (1 dimension)."""

SeriesFrame = Union[Series, Frame]
"""Series or DataFrame."""

MaybeSeriesFrame = Union[Scalar, SeriesFrame]
"""Scalar, Series or DataFrame."""

AnyArray = Union[Array, SeriesFrame]
"""NumPy array (1 or 2 dimensions), Series or DataFrame."""

MaybeAnyArray = Union[Scalar, AnyArray]
"""Scalar, NumPy array (1 or 2 dimensions), Series or DataFrame."""

AnyArray1d = Union[Array1d, Series]
"""NumPy array (1 dimension) or Series."""

AnyArray2d = Union[Array2d, Frame]
"""NumPy array (2 dimensions) or DataFrame."""

ArrayLike = Union[_ArrayLike, ExtensionArray, Index, SeriesFrame]
"""Object that can be converted to array."""

IndexLike = Union[_ArrayLike, ExtensionArray, Index, Series]
"""Object that can be converted to index."""

Level = Union[str, int]
"""Index level"""

LevelSequence = Sequence[Level]
"""Multiple index levels."""

MaybeLevelSequence = Union[Level, LevelSequence]
"""Single or multiple index levels."""

FrequencyLike = Union[str, float, pd.Timedelta, timedelta, np.timedelta64, DateOffset]
"""Any object that can be coerced into a timedelta."""

TimezoneLike = Union[None, str, float, timedelta, tzinfo]
"""Any object that can be coerced into a timezone."""

DatetimeLike = Union[str, float, pd.Timestamp, np.datetime64, datetime]
"""Any object that can be coerced into a datetime."""

IndexingFunc = Callable[[SeriesFrame], MaybeSeriesFrame]
"""Indexing function on Series and DataFrames."""

GroupByLike = Union[None, bool, MaybeLevelSequence, IndexLike]
"""Any object that can be coerced into a group-by index."""

Shape = Tuple[int, ...]
"""Shape."""

NameIndex = Union[None, Any, Index]
"""Name or index."""
