# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Engine-neutral dispatch wrappers for label functions."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt._engine import (
    array_compatible_with_rust,
    prepare_array_for_rust,
    combine_rust_support,
    flex_broadcast_to_shape,
    matching_shape_compatible_with_rust,
    non_neg_int_compatible_with_rust,
    resolve_engine,
)


def future_mean_apply(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    wait: int = 1,
    adjust: bool = False,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.future_mean_apply_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            non_neg_int_compatible_with_rust("window", window),
            non_neg_int_compatible_with_rust("wait", wait),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import future_mean_apply_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        return future_mean_apply_rs(close, window, ewm, wait, adjust)
    from vectorbt.labels.nb import future_mean_apply_nb

    return future_mean_apply_nb(close, window, ewm, wait, adjust)


def future_std_apply(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    wait: int = 1,
    adjust: bool = False,
    ddof: int = 0,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.future_std_apply_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            non_neg_int_compatible_with_rust("window", window),
            non_neg_int_compatible_with_rust("wait", wait),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import future_std_apply_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        return future_std_apply_rs(close, window, ewm, wait, adjust, ddof)
    from vectorbt.labels.nb import future_std_apply_nb

    return future_std_apply_nb(close, window, ewm, wait, adjust, ddof)


def future_min_apply(
    close: tp.Array2d,
    window: int,
    wait: int = 1,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.future_min_apply_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            non_neg_int_compatible_with_rust("window", window),
            non_neg_int_compatible_with_rust("wait", wait),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import future_min_apply_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        return future_min_apply_rs(close, window, wait)
    from vectorbt.labels.nb import future_min_apply_nb

    return future_min_apply_nb(close, window, wait)


def future_max_apply(
    close: tp.Array2d,
    window: int,
    wait: int = 1,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.future_max_apply_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            non_neg_int_compatible_with_rust("window", window),
            non_neg_int_compatible_with_rust("wait", wait),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import future_max_apply_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        return future_max_apply_rs(close, window, wait)
    from vectorbt.labels.nb import future_max_apply_nb

    return future_max_apply_nb(close, window, wait)


def fixed_labels_apply(
    close: tp.Array2d,
    n: int,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.fixed_labels_apply_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            non_neg_int_compatible_with_rust("n", n),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import fixed_labels_apply_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        return fixed_labels_apply_rs(close, n)
    from vectorbt.labels.nb import fixed_labels_apply_nb

    return fixed_labels_apply_nb(close, n)


def mean_labels_apply(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    wait: int = 1,
    adjust: bool = False,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.mean_labels_apply_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            non_neg_int_compatible_with_rust("window", window),
            non_neg_int_compatible_with_rust("wait", wait),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import mean_labels_apply_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        return mean_labels_apply_rs(close, window, ewm, wait, adjust)
    from vectorbt.labels.nb import mean_labels_apply_nb

    return mean_labels_apply_nb(close, window, ewm, wait, adjust)


def local_extrema_apply(
    close: tp.Array2d,
    pos_th: tp.MaybeArray[float],
    neg_th: tp.MaybeArray[float],
    flex_2d: bool = True,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.local_extrema_apply_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=array_compatible_with_rust(close, dtype=np.float64),
    )
    if eng == "rust":
        from vectorbt_rust.labels import local_extrema_apply_rs

        pos_th = flex_broadcast_to_shape(pos_th, close.shape, np.float64, flex_2d)
        neg_th = flex_broadcast_to_shape(neg_th, close.shape, np.float64, flex_2d)
        close = prepare_array_for_rust(close, dtype=np.float64)
        return local_extrema_apply_rs(close, pos_th, neg_th)
    from vectorbt.labels.nb import local_extrema_apply_nb

    return local_extrema_apply_nb(close, pos_th, neg_th, flex_2d)


def bn_trend_labels(
    close: tp.Array2d,
    local_extrema: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.bn_trend_labels_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            array_compatible_with_rust(local_extrema, dtype=np.int64),
            matching_shape_compatible_with_rust("local_extrema", close, local_extrema),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import bn_trend_labels_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        local_extrema = prepare_array_for_rust(local_extrema, dtype=np.int64)
        return bn_trend_labels_rs(close, local_extrema)
    from vectorbt.labels.nb import bn_trend_labels_nb

    return bn_trend_labels_nb(close, local_extrema)


def bn_cont_trend_labels(
    close: tp.Array2d,
    local_extrema: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.bn_cont_trend_labels_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            array_compatible_with_rust(local_extrema, dtype=np.int64),
            matching_shape_compatible_with_rust("local_extrema", close, local_extrema),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import bn_cont_trend_labels_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        local_extrema = prepare_array_for_rust(local_extrema, dtype=np.int64)
        return bn_cont_trend_labels_rs(close, local_extrema)
    from vectorbt.labels.nb import bn_cont_trend_labels_nb

    return bn_cont_trend_labels_nb(close, local_extrema)


def bn_cont_sat_trend_labels(
    close: tp.Array2d,
    local_extrema: tp.Array2d,
    pos_th: tp.MaybeArray[float],
    neg_th: tp.MaybeArray[float],
    flex_2d: bool = True,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.bn_cont_sat_trend_labels_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            array_compatible_with_rust(local_extrema, dtype=np.int64),
            matching_shape_compatible_with_rust("local_extrema", close, local_extrema),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import bn_cont_sat_trend_labels_rs

        pos_th = flex_broadcast_to_shape(pos_th, close.shape, np.float64, flex_2d)
        neg_th = flex_broadcast_to_shape(neg_th, close.shape, np.float64, flex_2d)
        close = prepare_array_for_rust(close, dtype=np.float64)
        local_extrema = prepare_array_for_rust(local_extrema, dtype=np.int64)
        return bn_cont_sat_trend_labels_rs(close, local_extrema, pos_th, neg_th)
    from vectorbt.labels.nb import bn_cont_sat_trend_labels_nb

    return bn_cont_sat_trend_labels_nb(close, local_extrema, pos_th, neg_th, flex_2d)


def pct_trend_labels(
    close: tp.Array2d,
    local_extrema: tp.Array2d,
    normalize: bool,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.pct_trend_labels_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            array_compatible_with_rust(local_extrema, dtype=np.int64),
            matching_shape_compatible_with_rust("local_extrema", close, local_extrema),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import pct_trend_labels_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        local_extrema = prepare_array_for_rust(local_extrema, dtype=np.int64)
        return pct_trend_labels_rs(close, local_extrema, normalize)
    from vectorbt.labels.nb import pct_trend_labels_nb

    return pct_trend_labels_nb(close, local_extrema, normalize)


def trend_labels_apply(
    close: tp.Array2d,
    pos_th: tp.MaybeArray[float],
    neg_th: tp.MaybeArray[float],
    mode: int,
    flex_2d: bool = True,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.trend_labels_apply_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=array_compatible_with_rust(close, dtype=np.float64),
    )
    if eng == "rust":
        from vectorbt_rust.labels import trend_labels_apply_rs

        pos_th = flex_broadcast_to_shape(pos_th, close.shape, np.float64, flex_2d)
        neg_th = flex_broadcast_to_shape(neg_th, close.shape, np.float64, flex_2d)
        close = prepare_array_for_rust(close, dtype=np.float64)
        return trend_labels_apply_rs(close, pos_th, neg_th, int(mode))
    from vectorbt.labels.nb import trend_labels_apply_nb

    return trend_labels_apply_nb(close, pos_th, neg_th, mode, flex_2d)


def breakout_labels(
    close: tp.Array2d,
    window: int,
    pos_th: tp.MaybeArray[float],
    neg_th: tp.MaybeArray[float],
    wait: int = 1,
    flex_2d: bool = True,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.labels.nb.breakout_labels_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close, dtype=np.float64),
            non_neg_int_compatible_with_rust("window", window),
            non_neg_int_compatible_with_rust("wait", wait),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.labels import breakout_labels_rs

        pos_th = flex_broadcast_to_shape(pos_th, close.shape, np.float64, flex_2d)
        neg_th = flex_broadcast_to_shape(neg_th, close.shape, np.float64, flex_2d)
        close = prepare_array_for_rust(close, dtype=np.float64)
        return breakout_labels_rs(close, window, pos_th, neg_th, wait)
    from vectorbt.labels.nb import breakout_labels_nb

    return breakout_labels_nb(close, window, pos_th, neg_th, wait, flex_2d)
