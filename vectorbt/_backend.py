# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Centralized backend resolution for dispatching between numba and Rust backends."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt._settings import settings

_rust_status: tp.Optional[bool] = None  # None = unchecked, True/False = cached


def is_rust_available() -> bool:
    """Whether vectorbt-rust is installed and version-compatible."""
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


def supports_rust(a: tp.Any) -> bool:
    """Whether array `a` is compatible with the Rust backend.

    The Rust bindings require contiguous float64 arrays. Integer arrays,
    non-contiguous slices (e.g. `a[::2]`), and other dtypes must fall
    back to numba so that auto mode never breaks inputs that numba handles."""
    return isinstance(a, np.ndarray) and a.dtype == np.float64 and a.flags["C_CONTIGUOUS"]


def resolve_backend(backend: tp.Optional[str] = None, supports_rust: bool = True) -> str:
    """Resolve which backend to use for a given function call.

    Set `backend` to override the global `settings['backend']`.
    Set `supports_rust` to False for callback-accepting functions, unsupported
    dtypes, or any other condition that prevents Rust dispatch.

    Returns `'numba'` or `'rust'`.

    Raises `ImportError` if `backend='rust'` and vectorbt-rust is not installed.
    Raises `ValueError` if `backend='rust'` and the function doesn't support Rust."""
    if backend is None:
        backend = settings["backend"]

    if backend == "numba":
        return "numba"

    if backend == "rust":
        if not is_rust_available():
            raise ImportError(
                "vectorbt-rust is not installed. "
                "Install with: pip install vectorbt-rust "
                "(or pip install vectorbt[rust])"
            )
        if not supports_rust:
            raise ValueError(
                "This function does not support backend='rust' " "(e.g. it accepts callbacks). Use backend='numba'."
            )
        return "rust"

    # backend == 'auto'
    if supports_rust and is_rust_available():
        return "rust"
    return "numba"
