# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Backend-neutral dispatch wrappers for signal functions."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt._backend import (
    array_compatible_with_rust,
    combine_rust_support,
    matching_shape_compatible_with_rust,
    resolve_backend,
)


def clean_enex_1d(
    entries: tp.Array1d,
    exits: tp.Array1d,
    entry_first: bool,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array1d, tp.Array1d]:
    """Backend-neutral `vectorbt.signals.nb.clean_enex_1d_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entries, dtype=np.bool_),
            array_compatible_with_rust(exits, dtype=np.bool_),
            matching_shape_compatible_with_rust("exits", entries, exits),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import clean_enex_1d_rs

        return clean_enex_1d_rs(entries, exits, entry_first)
    from vectorbt.signals.nb import clean_enex_1d_nb

    return clean_enex_1d_nb(entries, exits, entry_first)


def clean_enex(
    entries: tp.Array2d,
    exits: tp.Array2d,
    entry_first: bool,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Backend-neutral `vectorbt.signals.nb.clean_enex_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entries, dtype=np.bool_),
            array_compatible_with_rust(exits, dtype=np.bool_),
            matching_shape_compatible_with_rust("exits", entries, exits),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import clean_enex_rs

        return clean_enex_rs(entries, exits, entry_first)
    from vectorbt.signals.nb import clean_enex_nb

    return clean_enex_nb(entries, exits, entry_first)


def between_ranges(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.RecordArray:
    """Backend-neutral `vectorbt.signals.nb.between_ranges_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a, dtype=np.bool_))
    if eng == "rust":
        from vectorbt_rust.signals import between_ranges_rs

        return between_ranges_rs(a)
    from vectorbt.signals.nb import between_ranges_nb

    return between_ranges_nb(a)


def between_two_ranges(
    a: tp.Array2d,
    b: tp.Array2d,
    from_other: bool = False,
    backend: tp.Optional[str] = None,
) -> tp.RecordArray:
    """Backend-neutral `vectorbt.signals.nb.between_two_ranges_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a, dtype=np.bool_),
            array_compatible_with_rust(b, dtype=np.bool_),
            matching_shape_compatible_with_rust("b", a, b),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import between_two_ranges_rs

        return between_two_ranges_rs(a, b, from_other=from_other)
    from vectorbt.signals.nb import between_two_ranges_nb

    return between_two_ranges_nb(a, b, from_other=from_other)


def partition_ranges(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.RecordArray:
    """Backend-neutral `vectorbt.signals.nb.partition_ranges_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a, dtype=np.bool_))
    if eng == "rust":
        from vectorbt_rust.signals import partition_ranges_rs

        return partition_ranges_rs(a)
    from vectorbt.signals.nb import partition_ranges_nb

    return partition_ranges_nb(a)


def between_partition_ranges(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.RecordArray:
    """Backend-neutral `vectorbt.signals.nb.between_partition_ranges_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a, dtype=np.bool_))
    if eng == "rust":
        from vectorbt_rust.signals import between_partition_ranges_rs

        return between_partition_ranges_rs(a)
    from vectorbt.signals.nb import between_partition_ranges_nb

    return between_partition_ranges_nb(a)


def _rank_support(a: tp.Array2d, reset_by: tp.Optional[tp.Array2d]):
    if reset_by is None:
        return array_compatible_with_rust(a, dtype=np.bool_)
    return combine_rust_support(
        array_compatible_with_rust(a, dtype=np.bool_),
        array_compatible_with_rust(reset_by, dtype=np.bool_),
        matching_shape_compatible_with_rust("reset_by", a, reset_by),
    )


def sig_pos_rank(
    a: tp.Array2d,
    reset_by: tp.Optional[tp.Array2d],
    after_false: bool,
    allow_gaps: bool,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral signal position rank specialization."""
    eng = resolve_backend(backend, supports_rust=_rank_support(a, reset_by))
    if eng == "rust":
        from vectorbt_rust.signals import sig_pos_rank_rs

        return sig_pos_rank_rs(a, reset_by, after_false, allow_gaps)
    from vectorbt.signals.nb import rank_nb, sig_pos_rank_nb

    sig_pos_temp = np.full(a.shape[1], -1, dtype=np.int64)
    return rank_nb(a, reset_by, after_false, sig_pos_rank_nb, sig_pos_temp, allow_gaps)


def part_pos_rank(
    a: tp.Array2d,
    reset_by: tp.Optional[tp.Array2d],
    after_false: bool,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral partition position rank specialization."""
    eng = resolve_backend(backend, supports_rust=_rank_support(a, reset_by))
    if eng == "rust":
        from vectorbt_rust.signals import part_pos_rank_rs

        return part_pos_rank_rs(a, reset_by, after_false)
    from vectorbt.signals.nb import rank_nb, part_pos_rank_nb

    part_pos_temp = np.full(a.shape[1], -1, dtype=np.int64)
    return rank_nb(a, reset_by, after_false, part_pos_rank_nb, part_pos_temp)


def nth_index_1d(a: tp.Array1d, n: int, backend: tp.Optional[str] = None) -> int:
    """Backend-neutral `vectorbt.signals.nb.nth_index_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a, dtype=np.bool_))
    if eng == "rust":
        from vectorbt_rust.signals import nth_index_1d_rs

        return nth_index_1d_rs(a, n)
    from vectorbt.signals.nb import nth_index_1d_nb

    return nth_index_1d_nb(a, n)


def nth_index(a: tp.Array2d, n: int, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.signals.nb.nth_index_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a, dtype=np.bool_))
    if eng == "rust":
        from vectorbt_rust.signals import nth_index_rs

        return nth_index_rs(a, n)
    from vectorbt.signals.nb import nth_index_nb

    return nth_index_nb(a, n)


def norm_avg_index_1d(a: tp.Array1d, backend: tp.Optional[str] = None) -> float:
    """Backend-neutral `vectorbt.signals.nb.norm_avg_index_1d_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a, dtype=np.bool_))
    if eng == "rust":
        from vectorbt_rust.signals import norm_avg_index_1d_rs

        return norm_avg_index_1d_rs(a)
    from vectorbt.signals.nb import norm_avg_index_1d_nb

    return norm_avg_index_1d_nb(a)


def norm_avg_index(a: tp.Array2d, backend: tp.Optional[str] = None) -> tp.Array1d:
    """Backend-neutral `vectorbt.signals.nb.norm_avg_index_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a, dtype=np.bool_))
    if eng == "rust":
        from vectorbt_rust.signals import norm_avg_index_rs

        return norm_avg_index_rs(a)
    from vectorbt.signals.nb import norm_avg_index_nb

    return norm_avg_index_nb(a)
