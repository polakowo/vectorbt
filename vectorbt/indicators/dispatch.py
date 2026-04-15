# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Backend-neutral dispatch wrappers for indicator functions."""

from vectorbt import _typing as tp
from vectorbt._backend import (
    array_compatible_with_rust,
    combine_rust_support,
    matching_shape_compatible_with_rust,
    resolve_backend,
)


def ma(
    a: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool = False,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.indicators.nb.ma_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.indicators import ma_rs

        return ma_rs(a, window, ewm, adjust)
    from vectorbt.indicators.nb import ma_nb

    return ma_nb(a, window, ewm, adjust=adjust)


def mstd(
    a: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool = False,
    ddof: int = 0,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.indicators.nb.mstd_nb`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(a))
    if eng == "rust":
        from vectorbt_rust.indicators import mstd_rs

        return mstd_rs(a, window, ewm, adjust, ddof)
    from vectorbt.indicators.nb import mstd_nb

    return mstd_nb(a, window, ewm, adjust=adjust, ddof=ddof)


def ma_cache(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool,
    backend: tp.Optional[str] = None,
) -> dict:
    """Caching function for `vectorbt.indicators.basic.MA`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import ma_cache_rs

        return ma_cache_rs(close, windows, ewms, adjust)
    from vectorbt.indicators.nb import ma_cache_nb

    return ma_cache_nb(close, windows, ewms, adjust)


def ma_apply(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool,
    cache_dict: dict,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function for `vectorbt.indicators.basic.MA`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import ma_apply_rs

        return ma_apply_rs(close, window, ewm, adjust, cache_dict)
    from vectorbt.indicators.nb import ma_apply_nb

    return ma_apply_nb(close, window, ewm, adjust, cache_dict)


def mstd_cache(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool,
    ddof: int,
    backend: tp.Optional[str] = None,
) -> dict:
    """Caching function for `vectorbt.indicators.basic.MSTD`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import mstd_cache_rs

        return mstd_cache_rs(close, windows, ewms, adjust, ddof)
    from vectorbt.indicators.nb import mstd_cache_nb

    return mstd_cache_nb(close, windows, ewms, adjust, ddof)


def mstd_apply(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool,
    ddof: int,
    cache_dict: dict,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function for `vectorbt.indicators.basic.MSTD`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import mstd_apply_rs

        return mstd_apply_rs(close, window, ewm, adjust, ddof, cache_dict)
    from vectorbt.indicators.nb import mstd_apply_nb

    return mstd_apply_nb(close, window, ewm, adjust, ddof, cache_dict)


def bb_cache(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    alphas: tp.List[float],
    adjust: bool,
    ddof: int,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[dict, dict]:
    """Caching function for `vectorbt.indicators.basic.BBANDS`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import bb_cache_rs

        return bb_cache_rs(close, windows, ewms, alphas, adjust, ddof)
    from vectorbt.indicators.nb import bb_cache_nb

    return bb_cache_nb(close, windows, ewms, alphas, adjust, ddof)


def bb_apply(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    alpha: float,
    adjust: bool,
    ddof: int,
    ma_cache_dict: dict,
    mstd_cache_dict: dict,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbt.indicators.basic.BBANDS`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import bb_apply_rs

        return bb_apply_rs(close, window, ewm, alpha, adjust, ddof, ma_cache_dict, mstd_cache_dict)
    from vectorbt.indicators.nb import bb_apply_nb

    return bb_apply_nb(close, window, ewm, alpha, adjust, ddof, ma_cache_dict, mstd_cache_dict)


def rsi_cache(
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool,
    backend: tp.Optional[str] = None,
) -> dict:
    """Caching function for `vectorbt.indicators.basic.RSI`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import rsi_cache_rs

        return rsi_cache_rs(close, windows, ewms, adjust)
    from vectorbt.indicators.nb import rsi_cache_nb

    return rsi_cache_nb(close, windows, ewms, adjust)


def rsi_apply(
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool,
    cache_dict: dict,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function for `vectorbt.indicators.basic.RSI`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import rsi_apply_rs

        return rsi_apply_rs(close, window, ewm, adjust, cache_dict)
    from vectorbt.indicators.nb import rsi_apply_nb

    return rsi_apply_nb(close, window, ewm, adjust, cache_dict)


def stoch_cache(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    k_windows: tp.List[int],
    d_windows: tp.List[int],
    d_ewms: tp.List[bool],
    adjust: bool,
    backend: tp.Optional[str] = None,
) -> dict:
    """Caching function for `vectorbt.indicators.basic.STOCH`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(high),
            array_compatible_with_rust(low),
            array_compatible_with_rust(close),
            matching_shape_compatible_with_rust("low", high, low),
            matching_shape_compatible_with_rust("close", high, close),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.indicators import stoch_cache_rs

        return stoch_cache_rs(high, low, close, k_windows, d_windows, d_ewms, adjust)
    from vectorbt.indicators.nb import stoch_cache_nb

    return stoch_cache_nb(high, low, close, k_windows, d_windows, d_ewms, adjust)


def stoch_apply(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    k_window: int,
    d_window: int,
    d_ewm: bool,
    adjust: bool,
    cache_dict: dict,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbt.indicators.basic.STOCH`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(high),
            array_compatible_with_rust(low),
            array_compatible_with_rust(close),
            matching_shape_compatible_with_rust("low", high, low),
            matching_shape_compatible_with_rust("close", high, close),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.indicators import stoch_apply_rs

        return stoch_apply_rs(high, low, close, k_window, d_window, d_ewm, adjust, cache_dict)
    from vectorbt.indicators.nb import stoch_apply_nb

    return stoch_apply_nb(high, low, close, k_window, d_window, d_ewm, adjust, cache_dict)


def macd_cache(
    close: tp.Array2d,
    fast_windows: tp.List[int],
    slow_windows: tp.List[int],
    signal_windows: tp.List[int],
    macd_ewms: tp.List[bool],
    signal_ewms: tp.List[bool],
    adjust: bool,
    backend: tp.Optional[str] = None,
) -> dict:
    """Caching function for `vectorbt.indicators.basic.MACD`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import macd_cache_rs

        return macd_cache_rs(close, fast_windows, slow_windows, signal_windows, macd_ewms, signal_ewms, adjust)
    from vectorbt.indicators.nb import macd_cache_nb

    return macd_cache_nb(close, fast_windows, slow_windows, signal_windows, macd_ewms, signal_ewms, adjust)


def macd_apply(
    close: tp.Array2d,
    fast_window: int,
    slow_window: int,
    signal_window: int,
    macd_ewm: bool,
    signal_ewm: bool,
    adjust: bool,
    cache_dict: dict,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbt.indicators.basic.MACD`."""
    eng = resolve_backend(backend, supports_rust=array_compatible_with_rust(close))
    if eng == "rust":
        from vectorbt_rust.indicators import macd_apply_rs

        return macd_apply_rs(
            close,
            fast_window,
            slow_window,
            signal_window,
            macd_ewm,
            signal_ewm,
            adjust,
            cache_dict,
        )
    from vectorbt.indicators.nb import macd_apply_nb

    return macd_apply_nb(
        close,
        fast_window,
        slow_window,
        signal_window,
        macd_ewm,
        signal_ewm,
        adjust,
        cache_dict,
    )


def true_range(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.indicators.nb.true_range_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(high),
            array_compatible_with_rust(low),
            array_compatible_with_rust(close),
            matching_shape_compatible_with_rust("low", high, low),
            matching_shape_compatible_with_rust("close", high, close),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.indicators import true_range_rs

        return true_range_rs(high, low, close)
    from vectorbt.indicators.nb import true_range_nb

    return true_range_nb(high, low, close)


def atr_cache(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    windows: tp.List[int],
    ewms: tp.List[bool],
    adjust: bool,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, dict]:
    """Caching function for `vectorbt.indicators.basic.ATR`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(high),
            array_compatible_with_rust(low),
            array_compatible_with_rust(close),
            matching_shape_compatible_with_rust("low", high, low),
            matching_shape_compatible_with_rust("close", high, close),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.indicators import atr_cache_rs

        return atr_cache_rs(high, low, close, windows, ewms, adjust)
    from vectorbt.indicators.nb import atr_cache_nb

    return atr_cache_nb(high, low, close, windows, ewms, adjust)


def atr_apply(
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    window: int,
    ewm: bool,
    adjust: bool,
    tr: tp.Array2d,
    cache_dict: dict,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function for `vectorbt.indicators.basic.ATR`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(high),
            array_compatible_with_rust(low),
            array_compatible_with_rust(close),
            array_compatible_with_rust(tr),
            matching_shape_compatible_with_rust("low", high, low),
            matching_shape_compatible_with_rust("close", high, close),
            matching_shape_compatible_with_rust("tr", high, tr),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.indicators import atr_apply_rs

        return atr_apply_rs(high, low, close, window, ewm, adjust, tr, cache_dict)
    from vectorbt.indicators.nb import atr_apply_nb

    return atr_apply_nb(high, low, close, window, ewm, adjust, tr, cache_dict)


def obv_custom(
    close: tp.Array2d,
    volume_ts: tp.Array2d,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.indicators.nb.obv_custom_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close),
            array_compatible_with_rust(volume_ts),
            matching_shape_compatible_with_rust("volume_ts", close, volume_ts),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.indicators import obv_custom_rs

        return obv_custom_rs(close, volume_ts)
    from vectorbt.indicators.nb import obv_custom_nb

    return obv_custom_nb(close, volume_ts)
