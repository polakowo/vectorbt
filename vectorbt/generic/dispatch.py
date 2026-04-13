# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Backend-neutral dispatch wrappers for generic functions."""

from vectorbt import _typing as tp
from vectorbt._backend import resolve_backend, array_compatible_with_rust


# ======================== Rolling functions ========================


def rolling_mean_1d(
    a: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.rolling_mean_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_mean_1d_rs

        return rolling_mean_1d_rs(a, window, minp)
    from vectorbt.generic.nb import rolling_mean_1d_nb

    return rolling_mean_1d_nb(a, window, minp)


def rolling_mean(
    a: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.rolling_mean_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_mean_rs

        return rolling_mean_rs(a, window, minp)
    from vectorbt.generic.nb import rolling_mean_nb

    return rolling_mean_nb(a, window, minp)


def rolling_std_1d(
    a: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.rolling_std_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_std_1d_rs

        return rolling_std_1d_rs(a, window, minp, ddof)
    from vectorbt.generic.nb import rolling_std_1d_nb

    return rolling_std_1d_nb(a, window, minp, ddof)


def rolling_std(
    a: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    ddof: int = 0,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.rolling_std_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_std_rs

        return rolling_std_rs(a, window, minp, ddof)
    from vectorbt.generic.nb import rolling_std_nb

    return rolling_std_nb(a, window, minp, ddof)


def rolling_min_1d(
    a: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.rolling_min_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_min_1d_rs

        return rolling_min_1d_rs(a, window, minp)
    from vectorbt.generic.nb import rolling_min_1d_nb

    return rolling_min_1d_nb(a, window, minp)


def rolling_min(
    a: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.rolling_min_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_min_rs

        return rolling_min_rs(a, window, minp)
    from vectorbt.generic.nb import rolling_min_nb

    return rolling_min_nb(a, window, minp)


def rolling_max_1d(
    a: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.rolling_max_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_max_1d_rs

        return rolling_max_1d_rs(a, window, minp)
    from vectorbt.generic.nb import rolling_max_1d_nb

    return rolling_max_1d_nb(a, window, minp)


def rolling_max(
    a: tp.Array2d,
    window: int,
    minp: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.rolling_max_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_max_rs

        return rolling_max_rs(a, window, minp)
    from vectorbt.generic.nb import rolling_max_nb

    return rolling_max_nb(a, window, minp)


# ======================== Diff ========================


def diff_1d(a: tp.Array1d, n: int = 1, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.diff_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import diff_1d_rs

        return diff_1d_rs(a, n)
    from vectorbt.generic.nb import diff_1d_nb

    return diff_1d_nb(a, n)


def diff(a: tp.Array2d, n: int = 1, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.diff_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import diff_rs

        return diff_rs(a, n)
    from vectorbt.generic.nb import diff_nb

    return diff_nb(a, n)


# ======================== Callback functions (numba only) ========================


def apply(a: tp.Array2d, apply_func_nb: tp.ApplyFunc, *args: tp.Any, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.apply_nb`."""
    resolve_backend(backend, supports_rust=False)
    from vectorbt.generic.nb import apply_nb

    return apply_nb(a, apply_func_nb, *args)


def row_apply(
    a: tp.Array2d,
    apply_func_nb: tp.RowApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.row_apply_nb`."""
    resolve_backend(backend, supports_rust=False)
    from vectorbt.generic.nb import row_apply_nb

    return row_apply_nb(a, apply_func_nb, *args)


def rolling_apply(
    a: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    apply_func_nb: tp.RollApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.rolling_apply_nb`."""
    resolve_backend(backend, supports_rust=False)
    from vectorbt.generic.nb import rolling_apply_nb

    return rolling_apply_nb(a, window, minp, apply_func_nb, *args)


def rolling_matrix_apply(
    a: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    apply_func_nb: tp.RollMatrixApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.rolling_matrix_apply_nb`."""
    resolve_backend(backend, supports_rust=False)
    from vectorbt.generic.nb import rolling_matrix_apply_nb

    return rolling_matrix_apply_nb(a, window, minp, apply_func_nb, *args)


def reduce(
    a: tp.Array2d,
    reduce_func_nb: tp.ReduceFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.reduce_nb`."""
    resolve_backend(backend, supports_rust=False)
    from vectorbt.generic.nb import reduce_nb

    return reduce_nb(a, reduce_func_nb, *args)


def applymap(
    a: tp.Array2d,
    map_func_nb: tp.ApplyMapFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.applymap_nb`."""
    resolve_backend(backend, supports_rust=False)
    from vectorbt.generic.nb import applymap_nb

    return applymap_nb(a, map_func_nb, *args)


def filter_(
    a: tp.Array2d,
    filter_func_nb: tp.FilterFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.filter_nb`."""
    resolve_backend(backend, supports_rust=False)
    from vectorbt.generic.nb import filter_nb

    return filter_nb(a, filter_func_nb, *args)
