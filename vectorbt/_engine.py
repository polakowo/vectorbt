# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Centralized engine resolution for dispatching between Numba and Rust engines."""

from dataclasses import dataclass, field

import numpy as np

from vectorbt import _typing as tp
from vectorbt._settings import settings
from vectorbt.base import reshape_fns

_rust_status: tp.Optional[bool] = None
"""Status of Rust availability."""


@dataclass(frozen=True)
class RustConversion:
    """Array conversion required before calling the Rust engine."""

    dtype: np.dtype = field()
    """Target NumPy dtype expected by Rust."""

    n_elements: int = field()
    """Number of array elements to be converted."""


@dataclass(frozen=True)
class RustSupport:
    """Rust support result for an engine-neutral function call."""

    supported: bool = field()
    """Whether the Rust engine supports this function call."""

    reason: str = field(default="")
    """Reason shown when this function call cannot use the Rust engine."""

    conversions: tp.Tuple[RustConversion, ...] = field(default_factory=tuple)
    """Soft array conversions required before calling Rust."""

    @property
    def requires_conversion(self) -> bool:
        """Whether this call needs any soft conversion before Rust dispatch."""
        return len(self.conversions) > 0

    def __bool__(self) -> bool:
        return self.supported


def is_rust_available() -> bool:
    """Return whether `vectorbt-rust` is installed and version-compatible."""
    global _rust_status
    if _rust_status is None:
        try:
            import vectorbt_rust
            from vectorbt._version import __version__

            vbt_ver = __version__.rsplit(".", 1)[0]
            rust_ver = vectorbt_rust.__version__.rsplit(".", 1)[0]
            if vbt_ver != rust_ver:
                import warnings

                warnings.warn(
                    f"vectorbt {__version__} / vectorbt-rust "
                    f"{vectorbt_rust.__version__} version mismatch; "
                    f"falling back to numba"
                )
                _rust_status = False
            else:
                _rust_status = True
        except ImportError:
            _rust_status = False
    return _rust_status


def clear_engine_cache() -> None:
    """Clear cached engine availability checks."""
    global _rust_status
    _rust_status = None


def callback_unsupported_with_rust() -> RustSupport:
    """Return Rust support for callback-accepting functions."""
    return RustSupport(False, "Rust engine is not implemented for callback-accepting functions.")


def combine_rust_support(*support_results: RustSupport) -> RustSupport:
    """Return the first unsupported Rust support result."""
    conversions = []
    for support in support_results:
        if not isinstance(support, RustSupport):
            raise TypeError("Each support result must be a RustSupport instance.")
        if not support.supported:
            return support
        conversions.extend(support.conversions)
    return RustSupport(True, conversions=tuple(conversions))


def non_neg_int_compatible_with_rust(name: str, value: tp.Optional[int]) -> RustSupport:
    """Return whether a Python integer parameter can be passed to Rust as `usize`."""
    if value is not None and value < 0:
        return RustSupport(False, f"Rust engine requires `{name}` to be non-negative.")
    return RustSupport(True)


def array_compatible_with_rust(a: tp.Any, dtype: tp.Any = np.float64) -> RustSupport:
    """Return whether the array is compatible with the Rust engine."""
    if not isinstance(a, np.ndarray):
        return RustSupport(False, "Rust engine requires a NumPy array.")
    dtype = np.dtype(dtype)
    if a.dtype != dtype:
        same_kind = a.dtype.kind == dtype.kind or (a.dtype.kind in "iu" and dtype.kind in "iu")
        if not same_kind or not np.can_cast(a.dtype, dtype, casting="safe"):
            return RustSupport(
                False,
                f"Rust engine requires {dtype.name} arrays and `{a.dtype.name}` cannot be safely cast.",
            )
        if a.ndim not in (1, 2):
            return RustSupport(False, "Rust engine requires 1D or 2D arrays.")
        return RustSupport(True, conversions=(RustConversion(dtype, a.size),))
    if a.ndim not in (1, 2):
        return RustSupport(False, "Rust engine requires 1D or 2D arrays.")
    return RustSupport(True)


def exact_array_compatible_with_rust(a: tp.Any, dtype: tp.Any = np.float64) -> RustSupport:
    """Return whether the array already has the exact dtype expected by Rust."""
    support = array_compatible_with_rust(a, dtype=dtype)
    if not support.supported:
        return support
    if support.requires_conversion:
        dtype = np.dtype(dtype)
        return RustSupport(False, f"Rust engine requires exact {dtype.name} arrays for mutable arguments.")
    return support


def prepare_array_for_rust(a: tp.Any, dtype: tp.Any = np.float64) -> tp.Array:
    """Return `a` as the exact dtype expected by Rust.

    Exact dtype arrays are returned unchanged. Other arrays are accepted only
    when NumPy considers the cast safe.
    """
    support = array_compatible_with_rust(a, dtype=dtype)
    if not support.supported:
        raise ValueError(support.reason)
    dtype = np.dtype(dtype)
    if a.dtype == dtype:
        return a
    return np.asarray(a, dtype=dtype)


def prepare_flex_array_for_rust(
    a: tp.Any,
    shape: tp.Shape,
    dtype: tp.Any = np.float64,
    flex_2d: bool = True,
    name: str = "array",
) -> tp.Array:
    """Return a compact flexible array for Rust without broadcasting to ``shape``."""
    support = flex_array_compatible_with_rust(name, a, shape, dtype=dtype, flex_2d=flex_2d)
    if not support.supported:
        raise ValueError(support.reason)
    dtype = np.dtype(dtype)
    arr = np.asarray(a)
    if arr.dtype == dtype:
        return arr
    return np.asarray(a, dtype=dtype)


def matching_shape_compatible_with_rust(name: str, a: tp.Any, other: tp.Any) -> RustSupport:
    """Return whether two arrays have the same shape."""
    if not isinstance(a, np.ndarray) or not isinstance(other, np.ndarray):
        return RustSupport(False, f"Rust engine requires `{name}` to be a NumPy array.")
    if a.shape != other.shape:
        return RustSupport(False, f"Rust engine requires `{name}` to have the same shape as input.")
    return RustSupport(True)


def array_shape_compatible_with_rust(name: str, a: tp.Any, shape: tp.Shape) -> RustSupport:
    """Return whether an array has the exact shape required by Rust."""
    if not isinstance(a, np.ndarray):
        return RustSupport(False, f"Rust engine requires `{name}` to be a NumPy array.")
    if a.shape != tuple(shape):
        return RustSupport(False, f"Rust engine requires `{name}` to have shape {tuple(shape)}.")
    return RustSupport(True)


def flex_array_compatible_with_rust(
    name: str,
    a: tp.Any,
    shape: tp.Shape,
    dtype: tp.Any = np.float64,
    flex_2d: bool = True,
) -> RustSupport:
    """Return whether an array-like can be broadcast and cast before Rust dispatch."""
    arr = np.asarray(a)
    dtype = np.dtype(dtype)
    if arr.ndim == 0:
        try:
            np.asarray(a, dtype=dtype)
        except (TypeError, ValueError):
            return RustSupport(False, f"Rust engine requires `{name}` to be convertible to {dtype.name}.")
        support = RustSupport(True)
        if arr.dtype != dtype:
            support = RustSupport(True, conversions=(RustConversion(dtype, 1),))
    else:
        support = array_compatible_with_rust(arr, dtype=dtype)
        if not support.supported:
            return RustSupport(False, f"Rust engine requires `{name}` to be {dtype.name}-compatible.")
    try:
        if arr.ndim == 1 and len(shape) == 2:
            if flex_2d:
                arr = arr.reshape(1, -1)
            else:
                arr = arr.reshape(-1, 1)
        else:
            arr = reshape_fns.to_2d_array(arr)
        np.broadcast_to(arr, shape)
    except ValueError:
        return RustSupport(False, f"Rust engine requires `{name}` to broadcast to shape {tuple(shape)}.")
    return support


def scalar_compatible_with_rust(name: str, value: tp.Any) -> RustSupport:
    """Return whether a scalar can be passed to Rust as f64."""
    if isinstance(value, np.ndarray):
        return RustSupport(False, f"Rust engine requires `{name}` to be a scalar.")
    try:
        float(value)
    except (TypeError, ValueError):
        return RustSupport(False, f"Rust engine requires `{name}` to be convertible to float64.")
    return RustSupport(True)


def unit_interval_compatible_with_rust(name: str, value: tp.Any) -> RustSupport:
    """Return whether a scalar lies within the closed unit interval."""
    scalar_support = scalar_compatible_with_rust(name, value)
    if not scalar_support.supported:
        return scalar_support
    value = float(value)
    if value < 0.0 or value > 1.0:
        return RustSupport(False, f"Rust engine requires `{name}` to be between 0 and 1.")
    return RustSupport(True)


def non_neg_array_compatible_with_rust(name: str, a: tp.Any) -> RustSupport:
    """Return whether an array contains only non-negative int64 values for Rust."""
    support = array_compatible_with_rust(a, dtype=np.int64)
    if not support.supported:
        return support
    if np.any(a < 0):
        return RustSupport(False, f"Rust engine requires `{name}` to contain non-negative values.")
    return RustSupport(True)


def array_and_non_neg_int_compatible_with_rust(a: tp.Any, name: str, value: tp.Optional[int]) -> RustSupport:
    """Return whether an array and a non-negative integer parameter are compatible with the Rust engine."""
    return combine_rust_support(array_compatible_with_rust(a), non_neg_int_compatible_with_rust(name, value))


def mask_and_array_compatible_with_rust(a: tp.Any, mask: tp.Any) -> RustSupport:
    """Return whether an array and its boolean mask are compatible with the Rust engine."""
    return combine_rust_support(
        array_compatible_with_rust(a),
        array_compatible_with_rust(mask, dtype=np.bool_),
        matching_shape_compatible_with_rust("mask", a, mask),
    )


def mask_and_values_compatible_with_rust(a: tp.Any, mask: tp.Any, values: tp.Any) -> RustSupport:
    """Return whether an array, its mask, and replacement values are compatible with the Rust engine."""
    return combine_rust_support(
        mask_and_array_compatible_with_rust(a, mask),
        array_compatible_with_rust(values),
        matching_shape_compatible_with_rust("values", a, values),
    )


def col_range_compatible_with_rust(col_range: tp.Any) -> RustSupport:
    """Return whether a ColRange (2D int64 array) is compatible with the Rust engine."""
    if not isinstance(col_range, np.ndarray):
        return RustSupport(False, "Rust engine requires `col_range` to be a NumPy array.")
    if col_range.ndim != 2 or col_range.shape[1] != 2:
        return RustSupport(False, "Rust engine requires `col_range` to have shape (n_cols, 2).")
    return array_compatible_with_rust(col_range, dtype=np.int64)


def col_map_compatible_with_rust(col_map: tp.Any) -> RustSupport:
    """Return whether a column map (pair of arrays) is compatible with the Rust engine."""
    try:
        col_idxs, col_lens = col_map
    except (TypeError, ValueError):
        return RustSupport(False, "Rust engine requires `col_map` to be a pair of NumPy arrays.")
    return combine_rust_support(
        array_compatible_with_rust(col_idxs, dtype=np.int64),
        non_neg_array_compatible_with_rust("col_lens", col_lens),
    )


def rolling_compatible_with_rust(a: tp.Any, window: int, minp: tp.Optional[int]) -> RustSupport:
    """Return whether rolling arguments are compatible with the Rust engine."""
    return combine_rust_support(
        array_compatible_with_rust(a),
        non_neg_int_compatible_with_rust("window", window),
        non_neg_int_compatible_with_rust("minp", minp),
    )


def broadcast_to_shape(a: tp.ArrayLike, shape: tp.Shape, dtype: tp.Optional[tp.DTypeLike] = None) -> tp.Array:
    """Cast array to dtype if provided and broadcast to shape."""
    arr = np.asarray(a, dtype=dtype)
    return np.broadcast_to(arr, shape)


def broadcast_2d_to_shape(a: tp.ArrayLike, shape: tp.Shape, dtype: tp.Optional[tp.DTypeLike] = None) -> tp.Array:
    """Cast array to dtype if provided, reshape to 2D, and broadcast to shape."""
    arr = np.asarray(a, dtype=dtype)
    arr = reshape_fns.to_2d_array(arr)
    return np.broadcast_to(arr, shape)


def flex_broadcast_to_shape(
    a: tp.ArrayLike,
    shape: tp.Shape,
    dtype: tp.Optional[tp.DTypeLike] = None,
    flex_2d: bool = True,
) -> tp.Array:
    """Cast array to dtype if provided and broadcast to shape using flexible 2D semantics.

    If `shape` is 2D, a 1D array is treated as a single row when `flex_2d`
    is True and as a single column otherwise.
    """
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim == 1 and len(shape) == 2:
        if flex_2d:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, 1)
    else:
        arr = reshape_fns.to_2d_array(arr)
    return np.broadcast_to(arr, shape)


def resolve_random_engine(engine: tp.Optional[str] = None) -> str:
    """Resolve engine for randomized functions.

    Randomized functions default to Numba, including `engine='auto'`, to preserve
    legacy NumPy/Numba random streams unless an engine is requested explicitly.
    """
    if engine is None or engine == "auto":
        return "numba"
    return engine


def seed_for_rust(seed: tp.Optional[int], engine: tp.Optional[str], supports_rust: RustSupport) -> tp.Optional[int]:
    """Return seed only when the resolved engine is Rust."""
    if resolve_engine(engine, supports_rust=supports_rust) == "rust":
        return seed
    return None


def resolve_engine(engine: tp.Optional[str] = None, supports_rust: RustSupport = RustSupport(True)) -> str:
    """Resolve which engine to use for a given function call.

    Set `engine` to override the global `settings['engine']`.
    Set `supports_rust` to a `RustSupport` instance for callback-accepting
    functions, unsupported dtypes, or any other condition that prevents Rust
    dispatch.

    Returns `'numba'` or `'rust'`."""
    if engine is None:
        engine = settings["engine"]
    if not isinstance(supports_rust, RustSupport):
        raise TypeError("supports_rust must be a RustSupport instance.")
    if engine == "numba":
        return "numba"
    if engine == "rust":
        if not is_rust_available():
            raise ImportError(
                "vectorbt-rust is not installed. "
                "Install with: pip install vectorbt-rust "
                "(or pip install vectorbt[rust])"
            )
        if not supports_rust.supported:
            raise ValueError(f"{supports_rust.reason} Use engine='numba'.")
        return "rust"
    if engine != "auto":
        raise ValueError("Invalid engine. Expected 'auto', 'numba', or 'rust'.")
    if supports_rust.supported and is_rust_available():
        return "rust"
    return "numba"
