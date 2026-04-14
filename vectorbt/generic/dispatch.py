# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Backend-neutral dispatch wrappers for generic functions."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt.utils import checks
from vectorbt._backend import (
    array_and_non_neg_int_compatible_with_rust,
    array_compatible_with_rust,
    callback_unsupported_with_rust,
    col_map_compatible_with_rust,
    combine_rust_support,
    mask_and_array_compatible_with_rust,
    mask_and_values_compatible_with_rust,
    matching_shape_compatible_with_rust,
    non_neg_array_compatible_with_rust,
    non_neg_int_compatible_with_rust,
    resolve_backend,
    rolling_compatible_with_rust,
    scalar_compatible_with_rust,
)


def shuffle_1d(a: tp.Array1d, seed: tp.Optional[int] = None, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.shuffle_1d_nb`."""
    if backend is None or backend == "auto":
        from vectorbt.generic.nb import shuffle_1d_nb

        return shuffle_1d_nb(a, seed)
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("seed", seed),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import shuffle_1d_rs

        return shuffle_1d_rs(a, seed)
    from vectorbt.generic.nb import shuffle_1d_nb

    return shuffle_1d_nb(a, seed)


def shuffle(a: tp.Array2d, seed: tp.Optional[int] = None, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.shuffle_nb`."""
    if backend is None or backend == "auto":
        from vectorbt.generic.nb import shuffle_nb

        return shuffle_nb(a, seed)
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("seed", seed),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import shuffle_rs

        return shuffle_rs(a, seed)
    from vectorbt.generic.nb import shuffle_nb

    return shuffle_nb(a, seed)


def set_by_mask_1d(
    arr: tp.Array1d,
    mask: tp.Array1d,
    value: tp.Scalar,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.set_by_mask_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            mask_and_array_compatible_with_rust(arr, mask),
            scalar_compatible_with_rust("value", value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import set_by_mask_1d_rs

        return set_by_mask_1d_rs(arr, mask, value)
    from vectorbt.generic.nb import set_by_mask_1d_nb

    return set_by_mask_1d_nb(arr, mask, value)


def set_by_mask(
    arr: tp.Array2d,
    mask: tp.Array2d,
    value: tp.Scalar,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.set_by_mask_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            mask_and_array_compatible_with_rust(arr, mask),
            scalar_compatible_with_rust("value", value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import set_by_mask_rs

        return set_by_mask_rs(arr, mask, value)
    from vectorbt.generic.nb import set_by_mask_nb

    return set_by_mask_nb(arr, mask, value)


def set_by_mask_mult_1d(
    arr: tp.Array1d,
    mask: tp.Array1d,
    values: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.set_by_mask_mult_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=mask_and_values_compatible_with_rust(arr, mask, values))
    if eng == "rust":
        from vectorbt_rust.generic import set_by_mask_mult_1d_rs

        return set_by_mask_mult_1d_rs(arr, mask, values)
    from vectorbt.generic.nb import set_by_mask_mult_1d_nb

    return set_by_mask_mult_1d_nb(arr, mask, values)


def set_by_mask_mult(
    arr: tp.Array2d,
    mask: tp.Array2d,
    values: tp.Array2d,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.set_by_mask_mult_nb`."""
    eng = resolve_backend(backend, supports_rust=mask_and_values_compatible_with_rust(arr, mask, values))
    if eng == "rust":
        from vectorbt_rust.generic import set_by_mask_mult_rs

        return set_by_mask_mult_rs(arr, mask, values)
    from vectorbt.generic.nb import set_by_mask_mult_nb

    return set_by_mask_mult_nb(arr, mask, values)


def fillna_1d(a: tp.Array1d, value: tp.Scalar, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.fillna_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            scalar_compatible_with_rust("value", value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import fillna_1d_rs

        return fillna_1d_rs(a, value)
    from vectorbt.generic.nb import fillna_1d_nb

    return fillna_1d_nb(a, value)


def fillna(a: tp.Array2d, value: tp.Scalar, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.fillna_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            scalar_compatible_with_rust("value", value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import fillna_rs

        return fillna_rs(a, value)
    from vectorbt.generic.nb import fillna_nb

    return fillna_nb(a, value)


def bshift_1d(
    arr: tp.Array1d,
    n: int = 1,
    fill_value: tp.Scalar = float("nan"),
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.bshift_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_and_non_neg_int_compatible_with_rust(arr, "n", n),
            scalar_compatible_with_rust("fill_value", fill_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import bshift_1d_rs

        return bshift_1d_rs(arr, n, fill_value)
    from vectorbt.generic.nb import bshift_1d_nb

    return bshift_1d_nb(arr, n, fill_value)


def bshift(
    arr: tp.Array2d,
    n: int = 1,
    fill_value: tp.Scalar = float("nan"),
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.bshift_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_and_non_neg_int_compatible_with_rust(arr, "n", n),
            scalar_compatible_with_rust("fill_value", fill_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import bshift_rs

        return bshift_rs(arr, n, fill_value)
    from vectorbt.generic.nb import bshift_nb

    return bshift_nb(arr, n, fill_value)


def fshift_1d(
    arr: tp.Array1d,
    n: int = 1,
    fill_value: tp.Scalar = float("nan"),
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.fshift_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_and_non_neg_int_compatible_with_rust(arr, "n", n),
            scalar_compatible_with_rust("fill_value", fill_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import fshift_1d_rs

        return fshift_1d_rs(arr, n, fill_value)
    from vectorbt.generic.nb import fshift_1d_nb

    return fshift_1d_nb(arr, n, fill_value)


def fshift(
    arr: tp.Array2d,
    n: int = 1,
    fill_value: tp.Scalar = float("nan"),
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.fshift_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_and_non_neg_int_compatible_with_rust(arr, "n", n),
            scalar_compatible_with_rust("fill_value", fill_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import fshift_rs

        return fshift_rs(arr, n, fill_value)
    from vectorbt.generic.nb import fshift_nb

    return fshift_nb(arr, n, fill_value)


def diff_1d(a: tp.Array1d, n: int = 1, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.diff_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_and_non_neg_int_compatible_with_rust(a, "n", n))
    if eng == "rust":
        from vectorbt_rust.generic import diff_1d_rs

        return diff_1d_rs(a, n)
    from vectorbt.generic.nb import diff_1d_nb

    return diff_1d_nb(a, n)


def diff(a: tp.Array2d, n: int = 1, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.diff_nb`."""
    eng = resolve_backend(backend, supports_rust=array_and_non_neg_int_compatible_with_rust(a, "n", n))
    if eng == "rust":
        from vectorbt_rust.generic import diff_rs

        return diff_rs(a, n)
    from vectorbt.generic.nb import diff_nb

    return diff_nb(a, n)


def pct_change_1d(a: tp.Array1d, n: int = 1, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.pct_change_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_and_non_neg_int_compatible_with_rust(a, "n", n))
    if eng == "rust":
        from vectorbt_rust.generic import pct_change_1d_rs

        return pct_change_1d_rs(a, n)
    from vectorbt.generic.nb import pct_change_1d_nb

    return pct_change_1d_nb(a, n)


def pct_change(a: tp.Array2d, n: int = 1, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.pct_change_nb`."""
    eng = resolve_backend(backend, supports_rust=array_and_non_neg_int_compatible_with_rust(a, "n", n))
    if eng == "rust":
        from vectorbt_rust.generic import pct_change_rs

        return pct_change_rs(a, n)
    from vectorbt.generic.nb import pct_change_nb

    return pct_change_nb(a, n)


def bfill_1d(a: tp.Array1d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.bfill_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import bfill_1d_rs

        return bfill_1d_rs(a)
    from vectorbt.generic.nb import bfill_1d_nb

    return bfill_1d_nb(a)


def bfill(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.bfill_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import bfill_rs

        return bfill_rs(a)
    from vectorbt.generic.nb import bfill_nb

    return bfill_nb(a)


def ffill_1d(a: tp.Array1d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.ffill_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import ffill_1d_rs

        return ffill_1d_rs(a)
    from vectorbt.generic.nb import ffill_1d_nb

    return ffill_1d_nb(a)


def ffill(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.ffill_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import ffill_rs

        return ffill_rs(a)
    from vectorbt.generic.nb import ffill_nb

    return ffill_nb(a)


def nanprod(arr: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.nanprod_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(arr))
    if eng == "rust":
        from vectorbt_rust.generic import nanprod_rs

        return nanprod_rs(arr)
    from vectorbt.generic.nb import nanprod_nb

    return nanprod_nb(arr)


def nancumsum(arr: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.nancumsum_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(arr))
    if eng == "rust":
        from vectorbt_rust.generic import nancumsum_rs

        return nancumsum_rs(arr)
    from vectorbt.generic.nb import nancumsum_nb

    return nancumsum_nb(arr)


def nancumprod(arr: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.nancumprod_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(arr))
    if eng == "rust":
        from vectorbt_rust.generic import nancumprod_rs

        return nancumprod_rs(arr)
    from vectorbt.generic.nb import nancumprod_nb

    return nancumprod_nb(arr)


def nansum(arr: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.nansum_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(arr))
    if eng == "rust":
        from vectorbt_rust.generic import nansum_rs

        return nansum_rs(arr)
    from vectorbt.generic.nb import nansum_nb

    return nansum_nb(arr)


def nancnt(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.nancnt_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import nancnt_rs

        return nancnt_rs(a)
    from vectorbt.generic.nb import nancnt_nb

    return nancnt_nb(a)


def nanmin(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.nanmin_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import nanmin_rs

        return nanmin_rs(a)
    from vectorbt.generic.nb import nanmin_nb

    return nanmin_nb(a)


def nanmax(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.nanmax_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import nanmax_rs

        return nanmax_rs(a)
    from vectorbt.generic.nb import nanmax_nb

    return nanmax_nb(a)


def nanmean(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.nanmean_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import nanmean_rs

        return nanmean_rs(a)
    from vectorbt.generic.nb import nanmean_nb

    return nanmean_nb(a)


def nanmedian(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.nanmedian_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import nanmedian_rs

        return nanmedian_rs(a)
    from vectorbt.generic.nb import nanmedian_nb

    return nanmedian_nb(a)


def nanstd_1d(a: tp.Array1d, ddof: int = 0, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.nanstd_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import nanstd_1d_rs

        return nanstd_1d_rs(a, ddof)
    from vectorbt.generic.nb import nanstd_1d_nb

    return nanstd_1d_nb(a, ddof)


def nanstd(a: tp.Array2d, ddof: int = 0, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.nanstd_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import nanstd_rs

        return nanstd_rs(a, ddof)
    from vectorbt.generic.nb import nanstd_nb

    return nanstd_nb(a, ddof)


# ############# Rolling functions ############# #


def rolling_min_1d(
    a: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.rolling_min_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=rolling_compatible_with_rust(a, window, minp))
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
    eng = resolve_backend(backend, supports_rust=rolling_compatible_with_rust(a, window, minp))
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
    eng = resolve_backend(backend, supports_rust=rolling_compatible_with_rust(a, window, minp))
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
    eng = resolve_backend(backend, supports_rust=rolling_compatible_with_rust(a, window, minp))
    if eng == "rust":
        from vectorbt_rust.generic import rolling_max_rs

        return rolling_max_rs(a, window, minp)
    from vectorbt.generic.nb import rolling_max_nb

    return rolling_max_nb(a, window, minp)


def rolling_mean_1d(
    a: tp.Array1d,
    window: int,
    minp: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.rolling_mean_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=rolling_compatible_with_rust(a, window, minp))
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
    eng = resolve_backend(backend, supports_rust=rolling_compatible_with_rust(a, window, minp))
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
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(a, window, minp),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
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
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(a, window, minp),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import rolling_std_rs

        return rolling_std_rs(a, window, minp, ddof)
    from vectorbt.generic.nb import rolling_std_nb

    return rolling_std_nb(a, window, minp, ddof)


def ewm_mean_1d(
    a: tp.Array1d,
    span: int,
    minp: tp.Optional[int] = 0,
    adjust: bool = False,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.ewm_mean_1d_nb`."""
    if minp is None:
        minp = span
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("span", span),
            non_neg_int_compatible_with_rust("minp", minp),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import ewm_mean_1d_rs

        return ewm_mean_1d_rs(a, span, minp, adjust)
    from vectorbt.generic.nb import ewm_mean_1d_nb

    return ewm_mean_1d_nb(a, span, minp, adjust)


def ewm_mean(
    a: tp.Array2d,
    span: int,
    minp: tp.Optional[int] = 0,
    adjust: bool = False,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.ewm_mean_nb`."""
    if minp is None:
        minp = span
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("span", span),
            non_neg_int_compatible_with_rust("minp", minp),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import ewm_mean_rs

        return ewm_mean_rs(a, span, minp, adjust)
    from vectorbt.generic.nb import ewm_mean_nb

    return ewm_mean_nb(a, span, minp, adjust)


def ewm_std_1d(
    a: tp.Array1d,
    span: int,
    minp: tp.Optional[int] = 0,
    adjust: bool = False,
    ddof: int = 0,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.ewm_std_1d_nb`."""
    if minp is None:
        minp = span
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("span", span),
            non_neg_int_compatible_with_rust("minp", minp),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import ewm_std_1d_rs

        return ewm_std_1d_rs(a, span, minp, adjust, ddof)
    from vectorbt.generic.nb import ewm_std_1d_nb

    return ewm_std_1d_nb(a, span, minp, adjust, ddof)


def ewm_std(
    a: tp.Array2d,
    span: int,
    minp: tp.Optional[int] = 0,
    adjust: bool = False,
    ddof: int = 0,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.ewm_std_nb`."""
    if minp is None:
        minp = span
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("span", span),
            non_neg_int_compatible_with_rust("minp", minp),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import ewm_std_rs

        return ewm_std_rs(a, span, minp, adjust, ddof)
    from vectorbt.generic.nb import ewm_std_nb

    return ewm_std_nb(a, span, minp, adjust, ddof)


# ############# Expanding functions ############# #


def expanding_min_1d(a: tp.Array1d, minp: int = 1, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.expanding_min_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("minp", minp),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import expanding_min_1d_rs

        return expanding_min_1d_rs(a, minp)
    from vectorbt.generic.nb import expanding_min_1d_nb

    return expanding_min_1d_nb(a, minp)


def expanding_min(a: tp.Array2d, minp: int = 1, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.expanding_min_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("minp", minp),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import expanding_min_rs

        return expanding_min_rs(a, minp)
    from vectorbt.generic.nb import expanding_min_nb

    return expanding_min_nb(a, minp)


def expanding_max_1d(a: tp.Array1d, minp: int = 1, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.expanding_max_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("minp", minp),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import expanding_max_1d_rs

        return expanding_max_1d_rs(a, minp)
    from vectorbt.generic.nb import expanding_max_1d_nb

    return expanding_max_1d_nb(a, minp)


def expanding_max(a: tp.Array2d, minp: int = 1, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.expanding_max_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("minp", minp),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import expanding_max_rs

        return expanding_max_rs(a, minp)
    from vectorbt.generic.nb import expanding_max_nb

    return expanding_max_nb(a, minp)


def expanding_mean_1d(a: tp.Array1d, minp: int = 1, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.expanding_mean_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("minp", minp),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import expanding_mean_1d_rs

        return expanding_mean_1d_rs(a, minp)
    from vectorbt.generic.nb import expanding_mean_1d_nb

    return expanding_mean_1d_nb(a, minp)


def expanding_mean(a: tp.Array2d, minp: int = 1, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.expanding_mean_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("minp", minp),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import expanding_mean_rs

        return expanding_mean_rs(a, minp)
    from vectorbt.generic.nb import expanding_mean_nb

    return expanding_mean_nb(a, minp)


def expanding_std_1d(
    a: tp.Array1d,
    minp: int = 1,
    ddof: int = 0,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.expanding_std_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("minp", minp),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import expanding_std_1d_rs

        return expanding_std_1d_rs(a, minp, ddof)
    from vectorbt.generic.nb import expanding_std_1d_nb

    return expanding_std_1d_nb(a, minp, ddof)


def expanding_std(
    a: tp.Array2d,
    minp: int = 1,
    ddof: int = 0,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.expanding_std_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("minp", minp),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import expanding_std_rs

        return expanding_std_rs(a, minp, ddof)
    from vectorbt.generic.nb import expanding_std_nb

    return expanding_std_nb(a, minp, ddof)


# ############# Apply functions ############# #


def apply(a: tp.Array2d, apply_func: tp.ApplyFunc, *args: tp.Any, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.apply_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import apply_nb

    return apply_nb(a, apply_func, *args)


def row_apply(
    a: tp.Array2d,
    apply_func: tp.RowApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.row_apply_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import row_apply_nb

    return row_apply_nb(a, apply_func, *args)


def rolling_apply(
    a: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    apply_func: tp.RollApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.rolling_apply_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import rolling_apply_nb

    return rolling_apply_nb(a, window, minp, apply_func, *args)


def rolling_matrix_apply(
    a: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    apply_func: tp.RollMatrixApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.rolling_matrix_apply_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import rolling_matrix_apply_nb

    return rolling_matrix_apply_nb(a, window, minp, apply_func, *args)


def expanding_apply(
    a: tp.Array2d,
    minp: tp.Optional[int],
    apply_func: tp.RollApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.expanding_apply_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import expanding_apply_nb

    return expanding_apply_nb(a, minp, apply_func, *args)


def expanding_matrix_apply(
    a: tp.Array2d,
    minp: tp.Optional[int],
    apply_func: tp.RollMatrixApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.expanding_matrix_apply_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import expanding_matrix_apply_nb

    return expanding_matrix_apply_nb(a, minp, apply_func, *args)


def groupby_apply(
    a: tp.Array2d,
    groups: tp.Any,
    apply_func: tp.GroupByApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.groupby_apply_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import groupby_apply_nb

    return groupby_apply_nb(a, groups, apply_func, *args)


def groupby_matrix_apply(
    a: tp.Array2d,
    groups: tp.Any,
    apply_func: tp.GroupByMatrixApplyFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.groupby_matrix_apply_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import groupby_matrix_apply_nb

    return groupby_matrix_apply_nb(a, groups, apply_func, *args)


# ############# Map, filter and reduce ############# #


def applymap(
    a: tp.Array2d,
    map_func: tp.ApplyMapFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.applymap_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import applymap_nb

    return applymap_nb(a, map_func, *args)


def filter(
    a: tp.Array2d,
    filter_func: tp.FilterFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.filter_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import filter_nb

    return filter_nb(a, filter_func, *args)


filter_ = filter


def apply_and_reduce(
    a: tp.Array2d,
    apply_func: tp.ApplyFunc,
    apply_args: tuple,
    reduce_func: tp.ReduceFunc,
    reduce_args: tuple,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.apply_and_reduce_nb`."""
    if checks.is_backend_dispatch_func(reduce_func, func_suffix="_reduce"):
        out = None
        apply_is_dispatch = checks.is_backend_dispatch_func(apply_func)
        for col in range(a.shape[1]):
            if apply_is_dispatch:
                temp = apply_func(col, a[:, col], *apply_args, backend=backend)
            else:
                temp = apply_func(col, a[:, col], *apply_args)
            _out = reduce_func(col, temp, *reduce_args, backend=backend)
            if out is None:
                out = np.empty(a.shape[1], dtype=np.asarray(_out).dtype)
            out[col] = _out
        return out
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import apply_and_reduce_nb

    return apply_and_reduce_nb(a, apply_func, apply_args, reduce_func, reduce_args)


def reduce(
    a: tp.Array2d,
    reduce_func: tp.ReduceFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.reduce_nb`."""
    if checks.is_backend_dispatch_func(reduce_func, func_suffix="_reduce"):
        first_out = reduce_func(0, a[:, 0], *args, backend=backend)
        out = np.empty(a.shape[1], dtype=np.asarray(first_out).dtype)
        out[0] = first_out
        for col in range(1, a.shape[1]):
            out[col] = reduce_func(col, a[:, col], *args, backend=backend)
        return out
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import reduce_nb

    return reduce_nb(a, reduce_func, *args)


def reduce_to_array(
    a: tp.Array2d,
    reduce_func: tp.ReduceArrayFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.reduce_to_array_nb`."""
    if checks.is_backend_dispatch_func(reduce_func, func_suffix="_reduce"):
        out = None
        for col in range(a.shape[1]):
            _out = np.asarray(reduce_func(col, a[:, col], *args, backend=backend))
            if out is None:
                out = np.empty((_out.shape[0], a.shape[1]), dtype=_out.dtype)
            out[:, col] = _out
        return out
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import reduce_to_array_nb

    return reduce_to_array_nb(a, reduce_func, *args)


def reduce_grouped(
    a: tp.Array2d,
    group_lens: tp.Array1d,
    reduce_func: tp.GroupReduceFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.reduce_grouped_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import reduce_grouped_nb

    return reduce_grouped_nb(a, group_lens, reduce_func, *args)


def flatten_forder(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.flatten_forder_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import flatten_forder_rs

        return flatten_forder_rs(a)
    from vectorbt.generic.nb import flatten_forder_nb

    return flatten_forder_nb(a)


def flat_reduce_grouped(
    a: tp.Array2d,
    group_lens: tp.Array1d,
    in_c_order: bool,
    reduce_func: tp.FlatGroupReduceFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.flat_reduce_grouped_nb`."""
    if checks.is_backend_dispatch_func(reduce_func, func_suffix="_reduce"):
        from_col = 0
        out = None
        for group, group_len in enumerate(group_lens):
            to_col = from_col + group_len
            group_arr = a[:, from_col:to_col]
            flat_arr = group_arr.ravel(order="C" if in_c_order else "F")
            _out = reduce_func(group, flat_arr, *args, backend=backend)
            if out is None:
                out = np.empty(len(group_lens), dtype=np.asarray(_out).dtype)
            out[group] = _out
            from_col = to_col
        return out
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import flat_reduce_grouped_nb

    return flat_reduce_grouped_nb(a, group_lens, in_c_order, reduce_func, *args)


def reduce_grouped_to_array(
    a: tp.Array2d,
    group_lens: tp.Array1d,
    reduce_func: tp.GroupReduceArrayFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.reduce_grouped_to_array_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import reduce_grouped_to_array_nb

    return reduce_grouped_to_array_nb(a, group_lens, reduce_func, *args)


def flat_reduce_grouped_to_array(
    a: tp.Array2d,
    group_lens: tp.Array1d,
    in_c_order: bool,
    reduce_func: tp.FlatGroupReduceArrayFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.flat_reduce_grouped_to_array_nb`."""
    if checks.is_backend_dispatch_func(reduce_func, func_suffix="_reduce"):
        from_col = 0
        out = None
        for group, group_len in enumerate(group_lens):
            to_col = from_col + group_len
            group_arr = a[:, from_col:to_col]
            flat_arr = group_arr.ravel(order="C" if in_c_order else "F")
            _out = np.asarray(reduce_func(group, flat_arr, *args, backend=backend))
            if out is None:
                out = np.full((_out.shape[0], len(group_lens)), np.nan, dtype=_out.dtype)
            out[:, group] = _out
            from_col = to_col
        return out
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import flat_reduce_grouped_to_array_nb

    return flat_reduce_grouped_to_array_nb(a, group_lens, in_c_order, reduce_func, *args)


def squeeze_grouped(
    a: tp.Array2d,
    group_lens: tp.Array1d,
    squeeze_func: tp.GroupSqueezeFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.squeeze_grouped_nb`."""
    if checks.is_backend_dispatch_func(squeeze_func, func_suffix="_squeeze"):
        from_col = 0
        out = None
        for group, group_len in enumerate(group_lens):
            to_col = from_col + group_len
            for i in range(a.shape[0]):
                _out = squeeze_func(i, group, a[i, from_col:to_col], *args, backend=backend)
                if out is None:
                    out = np.empty((a.shape[0], len(group_lens)), dtype=np.asarray(_out).dtype)
                out[i, group] = _out
            from_col = to_col
        return out
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.generic.nb import squeeze_grouped_nb

    return squeeze_grouped_nb(a, group_lens, squeeze_func, *args)


# ############# Reshaping ############# #


def flatten_grouped(
    a: tp.Array2d,
    group_lens: tp.Array1d,
    in_c_order: bool,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.flatten_grouped_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_array_compatible_with_rust("group_lens", group_lens),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import flatten_grouped_rs

        return flatten_grouped_rs(a, group_lens, in_c_order)
    from vectorbt.generic.nb import flatten_grouped_nb

    return flatten_grouped_nb(a, group_lens, in_c_order)


def flatten_uniform_grouped(
    a: tp.Array2d,
    group_lens: tp.Array1d,
    in_c_order: bool,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.flatten_uniform_grouped_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_array_compatible_with_rust("group_lens", group_lens),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import flatten_uniform_grouped_rs

        return flatten_uniform_grouped_rs(a, group_lens, in_c_order)
    from vectorbt.generic.nb import flatten_uniform_grouped_nb

    return flatten_uniform_grouped_nb(a, group_lens, in_c_order)


# ############# Reducers ############# #


def nth_reduce(col: int, a: tp.Array1d, n: int, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.nth_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import nth_reduce_rs

        return nth_reduce_rs(a, n)
    from vectorbt.generic.nb import nth_reduce_nb

    return nth_reduce_nb(col, a, n)


def nth_index_reduce(col: int, a: tp.Array1d, n: int, backend: tp.Optional[str] = None) -> int:
    """Backend-neutral `vectorbt.generic.nb.nth_index_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import nth_index_reduce_rs

        return nth_index_reduce_rs(a, n)
    from vectorbt.generic.nb import nth_index_reduce_nb

    return nth_index_reduce_nb(col, a, n)


def min_reduce(col: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.min_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import min_reduce_rs

        return min_reduce_rs(a)
    from vectorbt.generic.nb import min_reduce_nb

    return min_reduce_nb(col, a)


def max_reduce(col: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.max_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import max_reduce_rs

        return max_reduce_rs(a)
    from vectorbt.generic.nb import max_reduce_nb

    return max_reduce_nb(col, a)


def mean_reduce(col: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.mean_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import mean_reduce_rs

        return mean_reduce_rs(a)
    from vectorbt.generic.nb import mean_reduce_nb

    return mean_reduce_nb(col, a)


def median_reduce(col: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.median_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import median_reduce_rs

        return median_reduce_rs(a)
    from vectorbt.generic.nb import median_reduce_nb

    return median_reduce_nb(col, a)


def std_reduce(col: int, a: tp.Array1d, ddof: int, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.std_reduce_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import std_reduce_rs

        return std_reduce_rs(a, ddof)
    from vectorbt.generic.nb import std_reduce_nb

    return std_reduce_nb(col, a, ddof)


def sum_reduce(col: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.sum_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import sum_reduce_rs

        return sum_reduce_rs(a)
    from vectorbt.generic.nb import sum_reduce_nb

    return sum_reduce_nb(col, a)


def count_reduce(col: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> int:
    """Backend-neutral `vectorbt.generic.nb.count_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import count_reduce_rs

        return count_reduce_rs(a)
    from vectorbt.generic.nb import count_reduce_nb

    return count_reduce_nb(col, a)


def argmin_reduce(col: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> int:
    """Backend-neutral `vectorbt.generic.nb.argmin_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import argmin_reduce_rs

        return argmin_reduce_rs(a)
    from vectorbt.generic.nb import argmin_reduce_nb

    return argmin_reduce_nb(col, a)


def argmax_reduce(col: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> int:
    """Backend-neutral `vectorbt.generic.nb.argmax_reduce_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import argmax_reduce_rs

        return argmax_reduce_rs(a)
    from vectorbt.generic.nb import argmax_reduce_nb

    return argmax_reduce_nb(col, a)


def describe_reduce(
    col: int,
    a: tp.Array1d,
    perc: tp.Array1d,
    ddof: int,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.describe_reduce_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            array_compatible_with_rust(perc),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import describe_reduce_rs

        return describe_reduce_rs(a, perc, ddof)
    from vectorbt.generic.nb import describe_reduce_nb

    return describe_reduce_nb(col, a, perc, ddof)


# ############# Value counts ############# #


def value_counts(
    codes: tp.Array2d,
    n_uniques: int,
    group_lens: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.value_counts_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            non_neg_array_compatible_with_rust("codes", codes),
            non_neg_int_compatible_with_rust("n_uniques", n_uniques),
            non_neg_array_compatible_with_rust("group_lens", group_lens),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import value_counts_rs

        return value_counts_rs(codes, n_uniques, group_lens)
    from vectorbt.generic.nb import value_counts_nb

    return value_counts_nb(codes, n_uniques, group_lens)


# ############# Group squeezers ############# #


def min_squeeze(col: int, group: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.min_squeeze_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import min_squeeze_rs

        return min_squeeze_rs(a)
    from vectorbt.generic.nb import min_squeeze_nb

    return min_squeeze_nb(col, group, a)


def max_squeeze(col: int, group: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.max_squeeze_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import max_squeeze_rs

        return max_squeeze_rs(a)
    from vectorbt.generic.nb import max_squeeze_nb

    return max_squeeze_nb(col, group, a)


def sum_squeeze(col: int, group: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.generic.nb.sum_squeeze_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import sum_squeeze_rs

        return sum_squeeze_rs(a)
    from vectorbt.generic.nb import sum_squeeze_nb

    return sum_squeeze_nb(col, group, a)


def any_squeeze(col: int, group: int, a: tp.Array1d, backend: tp.Optional[str] = None) -> bool:
    """Backend-neutral `vectorbt.generic.nb.any_squeeze_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.generic import any_squeeze_rs

        return any_squeeze_rs(a)
    from vectorbt.generic.nb import any_squeeze_nb

    return any_squeeze_nb(col, group, a)


# ############# Ranges ############# #


def find_ranges(ts: tp.Array2d, gap_value: tp.Scalar, backend: tp.Optional[str] = None) -> tp.RecordArray:
    """Backend-neutral `vectorbt.generic.nb.find_ranges_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(ts),
            scalar_compatible_with_rust("gap_value", gap_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import find_ranges_rs

        return find_ranges_rs(ts, gap_value)
    from vectorbt.generic.nb import find_ranges_nb

    return find_ranges_nb(ts, gap_value)


def range_duration(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.range_duration_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(start_idx_arr, np.int64),
            array_compatible_with_rust(end_idx_arr, np.int64),
            array_compatible_with_rust(status_arr, np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import range_duration_rs

        return range_duration_rs(start_idx_arr, end_idx_arr, status_arr)
    from vectorbt.generic.nb import range_duration_nb

    return range_duration_nb(start_idx_arr, end_idx_arr, status_arr)


def range_coverage(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    col_map: tp.ColMap,
    index_lens: tp.Array1d,
    overlapping: bool = False,
    normalize: bool = False,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.range_coverage_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(start_idx_arr, np.int64),
            array_compatible_with_rust(end_idx_arr, np.int64),
            array_compatible_with_rust(status_arr, np.int64),
            col_map_compatible_with_rust(col_map),
            non_neg_array_compatible_with_rust("index_lens", index_lens),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import range_coverage_rs

        return range_coverage_rs(start_idx_arr, end_idx_arr, status_arr, col_map, index_lens, overlapping, normalize)
    from vectorbt.generic.nb import range_coverage_nb

    return range_coverage_nb(start_idx_arr, end_idx_arr, status_arr, col_map, index_lens, overlapping, normalize)


def ranges_to_mask(
    start_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    status_arr: tp.Array2d,
    col_map: tp.ColMap,
    index_len: int,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.ranges_to_mask_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(start_idx_arr, np.int64),
            array_compatible_with_rust(end_idx_arr, np.int64),
            array_compatible_with_rust(status_arr, np.int64),
            col_map_compatible_with_rust(col_map),
            non_neg_int_compatible_with_rust("index_len", index_len),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import ranges_to_mask_rs

        return ranges_to_mask_rs(start_idx_arr, end_idx_arr, status_arr, col_map, index_len)
    from vectorbt.generic.nb import ranges_to_mask_nb

    return ranges_to_mask_nb(start_idx_arr, end_idx_arr, status_arr, col_map, index_len)


# ############# Drawdowns ############# #


def get_drawdowns(ts: tp.Array2d, backend: tp.Optional[str] = None) -> tp.RecordArray:
    """Backend-neutral `vectorbt.generic.nb.get_drawdowns_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(ts))
    if eng == "rust":
        from vectorbt_rust.generic import get_drawdowns_rs

        return get_drawdowns_rs(ts)
    from vectorbt.generic.nb import get_drawdowns_nb

    return get_drawdowns_nb(ts)


def dd_drawdown(peak_val_arr: tp.Array1d, valley_val_arr: tp.Array1d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.dd_drawdown_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(peak_val_arr),
            array_compatible_with_rust(valley_val_arr),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import dd_drawdown_rs

        return dd_drawdown_rs(peak_val_arr, valley_val_arr)
    from vectorbt.generic.nb import dd_drawdown_nb

    return dd_drawdown_nb(peak_val_arr, valley_val_arr)


def dd_decline_duration(
    start_idx_arr: tp.Array1d,
    valley_idx_arr: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.dd_decline_duration_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(start_idx_arr, np.int64),
            array_compatible_with_rust(valley_idx_arr, np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import dd_decline_duration_rs

        return dd_decline_duration_rs(start_idx_arr, valley_idx_arr)
    from vectorbt.generic.nb import dd_decline_duration_nb

    return dd_decline_duration_nb(start_idx_arr, valley_idx_arr)


def dd_recovery_duration(
    valley_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.dd_recovery_duration_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(valley_idx_arr, np.int64),
            array_compatible_with_rust(end_idx_arr, np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import dd_recovery_duration_rs

        return dd_recovery_duration_rs(valley_idx_arr, end_idx_arr)
    from vectorbt.generic.nb import dd_recovery_duration_nb

    return dd_recovery_duration_nb(valley_idx_arr, end_idx_arr)


def dd_recovery_duration_ratio(
    start_idx_arr: tp.Array1d,
    valley_idx_arr: tp.Array1d,
    end_idx_arr: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.dd_recovery_duration_ratio_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(start_idx_arr, np.int64),
            array_compatible_with_rust(valley_idx_arr, np.int64),
            array_compatible_with_rust(end_idx_arr, np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import dd_recovery_duration_ratio_rs

        return dd_recovery_duration_ratio_rs(start_idx_arr, valley_idx_arr, end_idx_arr)
    from vectorbt.generic.nb import dd_recovery_duration_ratio_nb

    return dd_recovery_duration_ratio_nb(start_idx_arr, valley_idx_arr, end_idx_arr)


def dd_recovery_return(
    valley_val_arr: tp.Array1d,
    end_val_arr: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.dd_recovery_return_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(valley_val_arr),
            array_compatible_with_rust(end_val_arr),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import dd_recovery_return_rs

        return dd_recovery_return_rs(valley_val_arr, end_val_arr)
    from vectorbt.generic.nb import dd_recovery_return_nb

    return dd_recovery_return_nb(valley_val_arr, end_val_arr)


# ############# Crossover ############# #


def crossed_above_1d(arr1: tp.Array1d, arr2: tp.Array1d, wait: int = 0, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.generic.nb.crossed_above_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(arr1),
            array_compatible_with_rust(arr2),
            matching_shape_compatible_with_rust("arr2", arr1, arr2),
            non_neg_int_compatible_with_rust("wait", wait),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import crossed_above_1d_rs

        return crossed_above_1d_rs(arr1, arr2, wait)
    from vectorbt.generic.nb import crossed_above_1d_nb

    return crossed_above_1d_nb(arr1, arr2, wait)


def crossed_above(arr1: tp.Array2d, arr2: tp.Array2d, wait: int = 0, backend: tp.Optional[str] = None) -> tp.Array2d:
    """Backend-neutral `vectorbt.generic.nb.crossed_above_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(arr1),
            array_compatible_with_rust(arr2),
            matching_shape_compatible_with_rust("arr2", arr1, arr2),
            non_neg_int_compatible_with_rust("wait", wait),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.generic import crossed_above_rs

        return crossed_above_rs(arr1, arr2, wait)
    from vectorbt.generic.nb import crossed_above_nb

    return crossed_above_nb(arr1, arr2, wait)
