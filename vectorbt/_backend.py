# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Centralized backend resolution for dispatching between numba and Rust backends."""

from dataclasses import dataclass, field

import numpy as np

from vectorbt import _typing as tp
from vectorbt._settings import settings

_rust_status: tp.Optional[bool] = None
"""Status of Rust availability."""


@dataclass(frozen=True)
class RustSupport:
    """Rust support result for a backend-neutral function call."""

    supported: bool = field()
    """Whether the Rust backend supports this function call."""

    reason: str = field(default="")
    """Reason shown when this function call cannot use the Rust backend."""

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


def clear_backend_cache() -> None:
    """Clear cached backend availability checks."""
    global _rust_status
    _rust_status = None


def callback_unsupported_with_rust() -> RustSupport:
    """Return Rust support for callback-accepting functions."""
    return RustSupport(False, "Rust backend is not implemented for callback-accepting functions.")


def combine_rust_support(*support_results: RustSupport) -> RustSupport:
    """Return the first unsupported Rust support result."""
    for support in support_results:
        if not isinstance(support, RustSupport):
            raise TypeError("Each support result must be a RustSupport instance.")
        if not support.supported:
            return support
    return RustSupport(True)


def non_neg_int_compatible_with_rust(name: str, value: tp.Optional[int]) -> RustSupport:
    """Return whether a Python integer parameter can be passed to Rust as `usize`."""
    if value is not None and value < 0:
        return RustSupport(False, f"Rust backend requires `{name}` to be non-negative.")
    return RustSupport(True)


def array_compatible_with_rust(a: tp.Any, dtype: tp.Any = np.float64) -> RustSupport:
    """Return whether the array is compatible with the Rust backend."""
    if not isinstance(a, np.ndarray):
        return RustSupport(False, "Rust backend requires a NumPy array.")
    if a.dtype != dtype:
        return RustSupport(False, f"Rust backend requires {np.dtype(dtype).name} arrays.")
    if a.ndim not in (1, 2):
        return RustSupport(False, "Rust backend requires 1D or 2D arrays.")
    if a.ndim == 1 and not (a.flags["C_CONTIGUOUS"] or a.flags["F_CONTIGUOUS"]):
        return RustSupport(False, "Rust backend requires contiguous 1D arrays.")
    return RustSupport(True)


def matching_shape_compatible_with_rust(name: str, a: tp.Any, other: tp.Any) -> RustSupport:
    """Return whether two arrays have the same shape."""
    if not isinstance(a, np.ndarray) or not isinstance(other, np.ndarray):
        return RustSupport(False, f"Rust backend requires `{name}` to be a NumPy array.")
    if a.shape != other.shape:
        return RustSupport(False, f"Rust backend requires `{name}` to have the same shape as input.")
    return RustSupport(True)


def scalar_compatible_with_rust(name: str, value: tp.Any) -> RustSupport:
    """Return whether a scalar can be passed to Rust as f64."""
    if isinstance(value, np.ndarray):
        return RustSupport(False, f"Rust backend requires `{name}` to be a scalar.")
    try:
        float(value)
    except (TypeError, ValueError):
        return RustSupport(False, f"Rust backend requires `{name}` to be convertible to float64.")
    return RustSupport(True)


def non_neg_array_compatible_with_rust(name: str, a: tp.Any) -> RustSupport:
    """Return whether an array contains only non-negative int64 values for Rust."""
    support = array_compatible_with_rust(a, dtype=np.int64)
    if not support.supported:
        return support
    if np.any(a < 0):
        return RustSupport(False, f"Rust backend requires `{name}` to contain non-negative values.")
    return RustSupport(True)


def array_and_non_neg_int_compatible_with_rust(a: tp.Any, name: str, value: tp.Optional[int]) -> RustSupport:
    """Return whether an array and a non-negative integer parameter are compatible with the Rust backend."""
    return combine_rust_support(array_compatible_with_rust(a), non_neg_int_compatible_with_rust(name, value))


def mask_and_array_compatible_with_rust(a: tp.Any, mask: tp.Any) -> RustSupport:
    """Return whether an array and its boolean mask are compatible with the Rust backend."""
    return combine_rust_support(
        array_compatible_with_rust(a),
        array_compatible_with_rust(mask, dtype=np.bool_),
        matching_shape_compatible_with_rust("mask", a, mask),
    )


def mask_and_values_compatible_with_rust(a: tp.Any, mask: tp.Any, values: tp.Any) -> RustSupport:
    """Return whether an array, its mask, and replacement values are compatible with the Rust backend."""
    return combine_rust_support(
        mask_and_array_compatible_with_rust(a, mask),
        array_compatible_with_rust(values),
        matching_shape_compatible_with_rust("values", a, values),
    )


def col_map_compatible_with_rust(col_map: tp.Any) -> RustSupport:
    """Return whether a column map (pair of arrays) is compatible with the Rust backend."""
    try:
        col_idxs, col_lens = col_map
    except (TypeError, ValueError):
        return RustSupport(False, "Rust backend requires `col_map` to be a pair of NumPy arrays.")
    return combine_rust_support(
        array_compatible_with_rust(col_idxs, dtype=np.int64), non_neg_array_compatible_with_rust("col_lens", col_lens)
    )


def rolling_compatible_with_rust(a: tp.Any, window: int, minp: tp.Optional[int]) -> RustSupport:
    """Return whether rolling arguments are compatible with the Rust backend."""
    return combine_rust_support(
        array_compatible_with_rust(a),
        non_neg_int_compatible_with_rust("window", window),
        non_neg_int_compatible_with_rust("minp", minp),
    )


def resolve_backend(backend: tp.Optional[str] = None, supports_rust: RustSupport = RustSupport(True)) -> str:
    """Resolve which backend to use for a given function call.

    Set `backend` to override the global `settings['backend']`.
    Set `supports_rust` to a `RustSupport` instance for callback-accepting
    functions, unsupported dtypes, or any other condition that prevents Rust
    dispatch.

    Returns `'numba'` or `'rust'`."""
    if backend is None:
        backend = settings["backend"]
    if not isinstance(supports_rust, RustSupport):
        raise TypeError("supports_rust must be a RustSupport instance.")
    if backend == "numba":
        return "numba"
    if backend == "rust":
        if not is_rust_available():
            raise ImportError(
                "vectorbt-rust is not installed. "
                "Install with: pip install vectorbt-rust "
                "(or pip install vectorbt[rust])"
            )
        if not supports_rust.supported:
            raise ValueError(f"{supports_rust.reason} Use backend='numba'.")
        return "rust"
    if backend != "auto":
        raise ValueError("Invalid backend. Expected 'auto', 'numba', or 'rust'.")
    if supports_rust.supported and is_rust_available():
        return "rust"
    return "numba"
