"""Utilities for type hints."""

import numpy as np
import pandas as pd
from typing import *
from numpy.typing import ArrayLike, DTypeLike

__pdoc__ = {}

T = TypeVar("T")
"""Generic type."""

F = TypeVar("F", bound=Callable[..., Any])
"""Generic function."""

MaybeTuple = Union[T, Tuple[T, ...]]
"""Value or tuple of values of the same type."""

MaybeList = Union[T, List[T]]
"""Value or list of values of the same type."""

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

AnyArray = Union[pd.Series, pd.DataFrame, np.ndarray]
"""NumPy array, Series or DataFrame."""
