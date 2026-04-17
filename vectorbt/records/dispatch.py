# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Backend-neutral dispatch wrappers for records functions."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt._backend import (
    array_compatible_with_rust,
    col_map_compatible_with_rust,
    col_range_compatible_with_rust,
    combine_rust_support,
    resolve_backend,
    scalar_compatible_with_rust,
)


# ############# Indexing #############


def col_range(col_arr: tp.Array1d, n_cols: int, backend: tp.Optional[str] = None) -> tp.ColRange:
    """Backend-neutral `vectorbt.records.nb.col_range_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=array_compatible_with_rust(col_arr, dtype=np.int64),
    )
    if eng == "rust":
        from vectorbt_rust.records import col_range_rs

        return col_range_rs(col_arr, n_cols)
    from vectorbt.records.nb import col_range_nb

    return col_range_nb(col_arr, n_cols)


def col_range_select(
    col_range: tp.ColRange,
    new_cols: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Backend-neutral `vectorbt.records.nb.col_range_select_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            col_range_compatible_with_rust(col_range),
            array_compatible_with_rust(new_cols, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import col_range_select_rs

        return col_range_select_rs(col_range, new_cols)
    from vectorbt.records.nb import col_range_select_nb

    return col_range_select_nb(col_range, new_cols)


def col_map(col_arr: tp.Array1d, n_cols: int, backend: tp.Optional[str] = None) -> tp.ColMap:
    """Backend-neutral `vectorbt.records.nb.col_map_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=array_compatible_with_rust(col_arr, dtype=np.int64),
    )
    if eng == "rust":
        from vectorbt_rust.records import col_map_rs

        return col_map_rs(col_arr, n_cols)
    from vectorbt.records.nb import col_map_nb

    return col_map_nb(col_arr, n_cols)


def col_map_select(
    col_map: tp.ColMap,
    new_cols: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Backend-neutral `vectorbt.records.nb.col_map_select_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            col_map_compatible_with_rust(col_map),
            array_compatible_with_rust(new_cols, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import col_map_select_rs

        col_idxs, col_lens = col_map
        return col_map_select_rs(col_idxs, col_lens, new_cols)
    from vectorbt.records.nb import col_map_select_nb

    return col_map_select_nb(col_map, new_cols)


def record_array_compatible_with_rust(records: tp.Any) -> "RustSupport":
    """Return whether a structured record array is compatible with the Rust backend.

    Requires a contiguous NumPy structured array whose first two fields are int64
    (matching the vectorbt convention of id, col as the first two fields)."""
    from vectorbt._backend import RustSupport

    if not isinstance(records, np.ndarray):
        return RustSupport(False, "Rust backend requires `records` to be a NumPy array.")
    if records.dtype.names is None:
        return RustSupport(False, "Rust backend requires `records` to be a structured array.")
    names = records.dtype.names
    if len(names) < 2:
        return RustSupport(False, "Rust backend requires `records` to have at least 2 fields.")
    id_dtype = records.dtype.fields[names[0]][0]
    col_dtype = records.dtype.fields[names[1]][0]
    if id_dtype != np.dtype(np.int64) or col_dtype != np.dtype(np.int64):
        return RustSupport(False, "Rust backend requires the first two record fields to be int64.")
    col_offset = records.dtype.fields[names[1]][1]
    if col_offset != 8:
        return RustSupport(False, "Rust backend requires the `col` field at byte offset 8.")
    return RustSupport(True)


def record_col_range_select(
    records: tp.RecordArray,
    col_range: tp.ColRange,
    new_cols: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.RecordArray:
    """Backend-neutral `vectorbt.records.nb.record_col_range_select_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            record_array_compatible_with_rust(records),
            col_range_compatible_with_rust(col_range),
            array_compatible_with_rust(new_cols, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import record_col_range_select_rs

        return record_col_range_select_rs(records, col_range, new_cols)
    from vectorbt.records.nb import record_col_range_select_nb

    return record_col_range_select_nb(records, col_range, new_cols)


def record_col_map_select(
    records: tp.RecordArray,
    col_map: tp.ColMap,
    new_cols: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> tp.RecordArray:
    """Backend-neutral `vectorbt.records.nb.record_col_map_select_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            record_array_compatible_with_rust(records),
            col_map_compatible_with_rust(col_map),
            array_compatible_with_rust(new_cols, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import record_col_map_select_rs

        col_idxs, col_lens = col_map
        return record_col_map_select_rs(records, col_idxs, col_lens, new_cols)
    from vectorbt.records.nb import record_col_map_select_nb

    return record_col_map_select_nb(records, col_map, new_cols)


# ############# Sorting #############


def is_col_sorted(col_arr: tp.Array1d, backend: tp.Optional[str] = None) -> bool:
    """Backend-neutral `vectorbt.records.nb.is_col_sorted_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=array_compatible_with_rust(col_arr, dtype=np.int64),
    )
    if eng == "rust":
        from vectorbt_rust.records import is_col_sorted_rs

        return is_col_sorted_rs(col_arr)
    from vectorbt.records.nb import is_col_sorted_nb

    return is_col_sorted_nb(col_arr)


def is_col_idx_sorted(
    col_arr: tp.Array1d,
    id_arr: tp.Array1d,
    backend: tp.Optional[str] = None,
) -> bool:
    """Backend-neutral `vectorbt.records.nb.is_col_idx_sorted_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(col_arr, dtype=np.int64),
            array_compatible_with_rust(id_arr, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import is_col_idx_sorted_rs

        return is_col_idx_sorted_rs(col_arr, id_arr)
    from vectorbt.records.nb import is_col_idx_sorted_nb

    return is_col_idx_sorted_nb(col_arr, id_arr)


# ############# Expansion #############


def is_mapped_expandable(
    col_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    target_shape: tp.Shape,
    backend: tp.Optional[str] = None,
) -> bool:
    """Backend-neutral `vectorbt.records.nb.is_mapped_expandable_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(col_arr, dtype=np.int64),
            array_compatible_with_rust(idx_arr, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import is_mapped_expandable_rs

        return is_mapped_expandable_rs(col_arr, idx_arr, target_shape)
    from vectorbt.records.nb import is_mapped_expandable_nb

    return is_mapped_expandable_nb(col_arr, idx_arr, target_shape)


def expand_mapped(
    mapped_arr: tp.Array1d,
    col_arr: tp.Array1d,
    idx_arr: tp.Array1d,
    target_shape: tp.Shape,
    fill_value: float,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.records.nb.expand_mapped_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(mapped_arr),
            array_compatible_with_rust(col_arr, dtype=np.int64),
            array_compatible_with_rust(idx_arr, dtype=np.int64),
            scalar_compatible_with_rust("fill_value", fill_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import expand_mapped_rs

        return expand_mapped_rs(mapped_arr, col_arr, idx_arr, target_shape, fill_value)
    from vectorbt.records.nb import expand_mapped_nb

    return expand_mapped_nb(mapped_arr, col_arr, idx_arr, target_shape, fill_value)


def stack_expand_mapped(
    mapped_arr: tp.Array1d,
    col_map: tp.ColMap,
    fill_value: float,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.records.nb.stack_expand_mapped_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(mapped_arr),
            col_map_compatible_with_rust(col_map),
            scalar_compatible_with_rust("fill_value", fill_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import stack_expand_mapped_rs

        col_idxs, col_lens = col_map
        return stack_expand_mapped_rs(mapped_arr, col_idxs, col_lens, fill_value)
    from vectorbt.records.nb import stack_expand_mapped_nb

    return stack_expand_mapped_nb(mapped_arr, col_map, fill_value)


# ############# Reducing #############


def mapped_value_counts(
    codes: tp.Array1d,
    n_uniques: int,
    col_map: tp.ColMap,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.records.nb.mapped_value_counts_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(codes, dtype=np.int64),
            col_map_compatible_with_rust(col_map),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import mapped_value_counts_rs

        col_idxs, col_lens = col_map
        return mapped_value_counts_rs(codes, n_uniques, col_idxs, col_lens)
    from vectorbt.records.nb import mapped_value_counts_nb

    return mapped_value_counts_nb(codes, n_uniques, col_map)


# ############# Mapping #############


def top_n_mapped_mask(
    mapped_arr: tp.Array1d,
    col_map: tp.ColMap,
    n: int,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral top-N mask computation.

    Rust path: standalone `top_n_mapped_mask_rs`.
    Numba path: `mapped_to_mask_nb` with `top_n_inout_map_nb` callback."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(mapped_arr),
            col_map_compatible_with_rust(col_map),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import top_n_mapped_mask_rs

        col_idxs, col_lens = col_map
        return top_n_mapped_mask_rs(mapped_arr, col_idxs, col_lens, n)
    from vectorbt.records.nb import mapped_to_mask_nb, top_n_inout_map_nb

    return mapped_to_mask_nb(mapped_arr, col_map, top_n_inout_map_nb, n)


def bottom_n_mapped_mask(
    mapped_arr: tp.Array1d,
    col_map: tp.ColMap,
    n: int,
    backend: tp.Optional[str] = None,
) -> tp.Array1d:
    """Backend-neutral bottom-N mask computation.

    Rust path: standalone `bottom_n_mapped_mask_rs`.
    Numba path: `mapped_to_mask_nb` with `bottom_n_inout_map_nb` callback."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(mapped_arr),
            col_map_compatible_with_rust(col_map),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.records import bottom_n_mapped_mask_rs

        col_idxs, col_lens = col_map
        return bottom_n_mapped_mask_rs(mapped_arr, col_idxs, col_lens, n)
    from vectorbt.records.nb import bottom_n_inout_map_nb, mapped_to_mask_nb

    return mapped_to_mask_nb(mapped_arr, col_map, bottom_n_inout_map_nb, n)
