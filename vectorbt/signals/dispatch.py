# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Backend-neutral dispatch wrappers for signal functions."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt._backend import (
    array_compatible_with_rust,
    flex_broadcast_to_shape,
    callback_unsupported_with_rust,
    combine_rust_support,
    matching_shape_compatible_with_rust,
    non_neg_array_compatible_with_rust,
    resolve_backend,
    resolve_random_backend,
    seed_for_rust,
)


def generate(
    shape: tp.RelaxedShape,
    pick_first: bool,
    choice_func: tp.ChoiceFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.generate_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.signals.nb import generate_nb

    return generate_nb(shape, pick_first, choice_func, *args)


def generate_ex(
    entries: tp.Array2d,
    wait: int,
    until_next: bool,
    skip_until_exit: bool,
    pick_first: bool,
    exit_choice_func: tp.ChoiceFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.generate_ex_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.signals.nb import generate_ex_nb

    return generate_ex_nb(entries, wait, until_next, skip_until_exit, pick_first, exit_choice_func, *args)


def generate_enex(
    shape: tp.RelaxedShape,
    entry_wait: int,
    exit_wait: int,
    entry_pick_first: bool,
    exit_pick_first: bool,
    entry_choice_func: tp.ChoiceFunc,
    entry_args: tp.Args,
    exit_choice_func: tp.ChoiceFunc,
    exit_args: tp.Args,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Backend-neutral `vectorbt.signals.nb.generate_enex_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.signals.nb import generate_enex_nb

    return generate_enex_nb(
        shape,
        entry_wait,
        exit_wait,
        entry_pick_first,
        exit_pick_first,
        entry_choice_func,
        entry_args,
        exit_choice_func,
        exit_args,
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


def generate_rand(
    shape: tp.Shape,
    n: tp.Array1d,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.generate_rand_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=non_neg_array_compatible_with_rust("n", n),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_rand_rs

        return generate_rand_rs(shape[0], shape[1], n, seed=seed)
    from vectorbt.signals.nb import generate_rand_nb

    return generate_rand_nb(shape, n, seed=seed)


def generate_rand_by_prob(
    shape: tp.Shape,
    prob: tp.Array2d,
    pick_first: bool,
    flex_2d: bool,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.generate_rand_by_prob_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=array_compatible_with_rust(prob, dtype=np.float64),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_rand_by_prob_rs

        return generate_rand_by_prob_rs(shape[0], shape[1], prob, pick_first, seed=seed)
    from vectorbt.signals.nb import generate_rand_by_prob_nb

    return generate_rand_by_prob_nb(shape, prob, pick_first, flex_2d, seed=seed)


def generate_rand_ex(
    entries: tp.Array2d,
    wait: int,
    until_next: bool,
    skip_until_exit: bool,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.generate_rand_ex_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=array_compatible_with_rust(entries, dtype=np.bool_),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_rand_ex_rs

        return generate_rand_ex_rs(entries, wait, until_next, skip_until_exit, seed=seed)
    from vectorbt.signals.nb import generate_rand_ex_nb

    return generate_rand_ex_nb(entries, wait, until_next, skip_until_exit, seed=seed)


def generate_rand_ex_by_prob(
    entries: tp.Array2d,
    prob: tp.Array2d,
    wait: int,
    until_next: bool,
    skip_until_exit: bool,
    flex_2d: bool,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.generate_rand_ex_by_prob_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entries, dtype=np.bool_),
            array_compatible_with_rust(prob, dtype=np.float64),
            matching_shape_compatible_with_rust("prob", entries, prob),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_rand_ex_by_prob_rs

        return generate_rand_ex_by_prob_rs(
            entries,
            prob,
            wait,
            until_next,
            skip_until_exit,
            seed=seed,
        )
    from vectorbt.signals.nb import generate_rand_ex_by_prob_nb

    return generate_rand_ex_by_prob_nb(
        entries,
        prob,
        wait,
        until_next,
        skip_until_exit,
        flex_2d,
        seed=seed,
    )


def generate_rand_enex(
    shape: tp.Shape,
    n: tp.Array1d,
    entry_wait: int,
    exit_wait: int,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Backend-neutral `vectorbt.signals.nb.generate_rand_enex_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=non_neg_array_compatible_with_rust("n", n),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_rand_enex_rs

        return generate_rand_enex_rs(shape[0], shape[1], n, entry_wait, exit_wait, seed=seed)
    from vectorbt.signals.nb import generate_rand_enex_nb

    return generate_rand_enex_nb(shape, n, entry_wait, exit_wait, seed=seed)


def generate_rand_enex_by_prob(
    shape: tp.Shape,
    entry_prob: tp.Array2d,
    exit_prob: tp.Array2d,
    entry_wait: int,
    exit_wait: int,
    entry_pick_first: bool,
    exit_pick_first: bool,
    flex_2d: bool,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Backend-neutral `vectorbt.signals.nb.generate_rand_enex_by_prob_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entry_prob, dtype=np.float64),
            array_compatible_with_rust(exit_prob, dtype=np.float64),
            matching_shape_compatible_with_rust("exit_prob", entry_prob, exit_prob),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_rand_enex_by_prob_rs

        return generate_rand_enex_by_prob_rs(
            shape[0],
            shape[1],
            entry_prob,
            exit_prob,
            entry_wait,
            exit_wait,
            entry_pick_first,
            exit_pick_first,
            seed=seed,
        )
    from vectorbt.signals.nb import generate_rand_enex_by_prob_nb

    return generate_rand_enex_by_prob_nb(
        shape,
        entry_prob,
        exit_prob,
        entry_wait,
        exit_wait,
        entry_pick_first,
        exit_pick_first,
        flex_2d,
        seed=seed,
    )


def generate_stop_ex(
    entries: tp.Array2d,
    ts: tp.Array2d,
    stop: tp.Array2d,
    trailing: tp.Array2d,
    wait: int,
    until_next: bool,
    skip_until_exit: bool,
    pick_first: bool,
    flex_2d: bool,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.generate_stop_ex_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entries, dtype=np.bool_),
            array_compatible_with_rust(ts, dtype=np.float64),
            array_compatible_with_rust(stop, dtype=np.float64),
            array_compatible_with_rust(trailing, dtype=np.bool_),
            matching_shape_compatible_with_rust("ts", entries, ts),
            matching_shape_compatible_with_rust("stop", entries, stop),
            matching_shape_compatible_with_rust("trailing", entries, trailing),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_stop_ex_rs

        return generate_stop_ex_rs(
            entries,
            ts,
            stop,
            trailing,
            wait,
            until_next,
            skip_until_exit,
            pick_first,
        )
    from vectorbt.signals.nb import generate_stop_ex_nb

    return generate_stop_ex_nb(
        entries,
        ts,
        stop,
        trailing,
        wait,
        until_next,
        skip_until_exit,
        pick_first,
        flex_2d,
    )


def generate_stop_enex(
    entries: tp.Array2d,
    ts: tp.Array2d,
    stop: tp.Array2d,
    trailing: tp.Array2d,
    entry_wait: int,
    exit_wait: int,
    pick_first: bool,
    flex_2d: bool,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Backend-neutral `vectorbt.signals.nb.generate_stop_enex_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entries, dtype=np.bool_),
            array_compatible_with_rust(ts, dtype=np.float64),
            array_compatible_with_rust(stop, dtype=np.float64),
            array_compatible_with_rust(trailing, dtype=np.bool_),
            matching_shape_compatible_with_rust("ts", entries, ts),
            matching_shape_compatible_with_rust("stop", entries, stop),
            matching_shape_compatible_with_rust("trailing", entries, trailing),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_stop_enex_rs

        return generate_stop_enex_rs(
            entries,
            ts,
            stop,
            trailing,
            entry_wait,
            exit_wait,
            pick_first,
        )
    from vectorbt.signals.nb import generate_stop_enex_nb

    return generate_stop_enex_nb(
        entries,
        ts,
        stop,
        trailing,
        entry_wait,
        exit_wait,
        pick_first,
        flex_2d,
    )


def generate_ohlc_stop_ex(
    entries: tp.Array2d,
    open: tp.Array2d,
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    stop_price_out: tp.Array2d,
    stop_type_out: tp.Array2d,
    sl_stop: tp.Array2d,
    sl_trail: tp.Array2d,
    tp_stop: tp.Array2d,
    reverse: tp.Array2d,
    is_open_safe: bool,
    wait: int,
    until_next: bool,
    skip_until_exit: bool,
    pick_first: bool,
    flex_2d: bool,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.generate_ohlc_stop_ex_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entries, dtype=np.bool_),
            array_compatible_with_rust(open, dtype=np.float64),
            array_compatible_with_rust(high, dtype=np.float64),
            array_compatible_with_rust(low, dtype=np.float64),
            array_compatible_with_rust(close, dtype=np.float64),
            array_compatible_with_rust(stop_price_out, dtype=np.float64),
            array_compatible_with_rust(stop_type_out, dtype=np.int64),
            array_compatible_with_rust(sl_stop, dtype=np.float64),
            array_compatible_with_rust(sl_trail, dtype=np.bool_),
            array_compatible_with_rust(tp_stop, dtype=np.float64),
            array_compatible_with_rust(reverse, dtype=np.bool_),
            matching_shape_compatible_with_rust("open", entries, open),
            matching_shape_compatible_with_rust("high", entries, high),
            matching_shape_compatible_with_rust("low", entries, low),
            matching_shape_compatible_with_rust("close", entries, close),
            matching_shape_compatible_with_rust("stop_price_out", entries, stop_price_out),
            matching_shape_compatible_with_rust("stop_type_out", entries, stop_type_out),
            matching_shape_compatible_with_rust("sl_stop", entries, sl_stop),
            matching_shape_compatible_with_rust("sl_trail", entries, sl_trail),
            matching_shape_compatible_with_rust("tp_stop", entries, tp_stop),
            matching_shape_compatible_with_rust("reverse", entries, reverse),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_ohlc_stop_ex_rs

        return generate_ohlc_stop_ex_rs(
            entries,
            open,
            high,
            low,
            close,
            stop_price_out,
            stop_type_out,
            sl_stop,
            sl_trail,
            tp_stop,
            reverse,
            is_open_safe,
            wait,
            until_next,
            skip_until_exit,
            pick_first,
        )
    from vectorbt.signals.nb import generate_ohlc_stop_ex_nb

    return generate_ohlc_stop_ex_nb(
        entries,
        open,
        high,
        low,
        close,
        stop_price_out,
        stop_type_out,
        sl_stop,
        sl_trail,
        tp_stop,
        reverse,
        is_open_safe,
        wait,
        until_next,
        skip_until_exit,
        pick_first,
        flex_2d,
    )


def generate_ohlc_stop_enex(
    entries: tp.Array2d,
    open: tp.Array2d,
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    stop_price_out: tp.Array2d,
    stop_type_out: tp.Array2d,
    sl_stop: tp.Array2d,
    sl_trail: tp.Array2d,
    tp_stop: tp.Array2d,
    reverse: tp.Array2d,
    is_open_safe: bool,
    entry_wait: int,
    exit_wait: int,
    pick_first: bool,
    flex_2d: bool,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Backend-neutral `vectorbt.signals.nb.generate_ohlc_stop_enex_nb`."""
    eng = resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entries, dtype=np.bool_),
            array_compatible_with_rust(open, dtype=np.float64),
            array_compatible_with_rust(high, dtype=np.float64),
            array_compatible_with_rust(low, dtype=np.float64),
            array_compatible_with_rust(close, dtype=np.float64),
            array_compatible_with_rust(stop_price_out, dtype=np.float64),
            array_compatible_with_rust(stop_type_out, dtype=np.int64),
            array_compatible_with_rust(sl_stop, dtype=np.float64),
            array_compatible_with_rust(sl_trail, dtype=np.bool_),
            array_compatible_with_rust(tp_stop, dtype=np.float64),
            array_compatible_with_rust(reverse, dtype=np.bool_),
            matching_shape_compatible_with_rust("open", entries, open),
            matching_shape_compatible_with_rust("high", entries, high),
            matching_shape_compatible_with_rust("low", entries, low),
            matching_shape_compatible_with_rust("close", entries, close),
            matching_shape_compatible_with_rust("stop_price_out", entries, stop_price_out),
            matching_shape_compatible_with_rust("stop_type_out", entries, stop_type_out),
            matching_shape_compatible_with_rust("sl_stop", entries, sl_stop),
            matching_shape_compatible_with_rust("sl_trail", entries, sl_trail),
            matching_shape_compatible_with_rust("tp_stop", entries, tp_stop),
            matching_shape_compatible_with_rust("reverse", entries, reverse),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.signals import generate_ohlc_stop_enex_rs

        return generate_ohlc_stop_enex_rs(
            entries,
            open,
            high,
            low,
            close,
            stop_price_out,
            stop_type_out,
            sl_stop,
            sl_trail,
            tp_stop,
            reverse,
            is_open_safe,
            entry_wait,
            exit_wait,
            pick_first,
        )
    from vectorbt.signals.nb import generate_ohlc_stop_enex_nb

    return generate_ohlc_stop_enex_nb(
        entries,
        open,
        high,
        low,
        close,
        stop_price_out,
        stop_type_out,
        sl_stop,
        sl_trail,
        tp_stop,
        reverse,
        is_open_safe,
        entry_wait,
        exit_wait,
        pick_first,
        flex_2d,
    )


def rand_apply(
    input_shape: tp.Shape,
    n: tp.Array1d,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function used by `vectorbt.signals.generators.RAND`."""
    backend = resolve_random_backend(backend)
    n = np.broadcast_to(np.asarray(n, dtype=np.int64), input_shape[1])
    return generate_rand(
        input_shape,
        n,
        seed=seed_for_rust(seed, backend, non_neg_array_compatible_with_rust("n", n)),
        backend=backend,
    )


def rand_ex_apply(
    entries: tp.Array2d,
    wait: int = 1,
    until_next: bool = True,
    skip_until_exit: bool = False,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function used by `vectorbt.signals.generators.RANDX`."""
    backend = resolve_random_backend(backend)
    return generate_rand_ex(
        entries,
        wait,
        until_next,
        skip_until_exit,
        seed=seed_for_rust(seed, backend, array_compatible_with_rust(entries, dtype=np.bool_)),
        backend=backend,
    )


def rand_enex_apply(
    input_shape: tp.Shape,
    n: tp.Array1d,
    entry_wait: int = 1,
    exit_wait: int = 1,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function used by `vectorbt.signals.generators.RANDNX`."""
    backend = resolve_random_backend(backend)
    n = np.broadcast_to(np.asarray(n, dtype=np.int64), input_shape[1])
    return generate_rand_enex(
        input_shape,
        n,
        entry_wait,
        exit_wait,
        seed=seed_for_rust(seed, backend, non_neg_array_compatible_with_rust("n", n)),
        backend=backend,
    )


def rand_by_prob_apply(
    input_shape: tp.Shape,
    prob: tp.Array2d,
    flex_2d: bool,
    pick_first: bool = False,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function used by `vectorbt.signals.generators.RPROB`."""
    backend = resolve_random_backend(backend)
    prob = flex_broadcast_to_shape(prob, input_shape, np.float64)
    return generate_rand_by_prob(
        input_shape,
        prob,
        pick_first,
        flex_2d,
        seed=seed_for_rust(seed, backend, array_compatible_with_rust(prob, dtype=np.float64)),
        backend=backend,
    )


def rand_ex_by_prob_apply(
    entries: tp.Array2d,
    prob: tp.Array2d,
    flex_2d: bool,
    wait: int = 1,
    until_next: bool = True,
    skip_until_exit: bool = False,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function used by `vectorbt.signals.generators.RPROBX`."""
    backend = resolve_random_backend(backend)
    prob = flex_broadcast_to_shape(prob, entries.shape, np.float64)
    return generate_rand_ex_by_prob(
        entries,
        prob,
        wait,
        until_next,
        skip_until_exit,
        flex_2d,
        seed=seed_for_rust(
            seed,
            backend,
            combine_rust_support(
                array_compatible_with_rust(entries, dtype=np.bool_),
                array_compatible_with_rust(prob, dtype=np.float64),
                matching_shape_compatible_with_rust("prob", entries, prob),
            ),
        ),
        backend=backend,
    )


def rand_enex_by_prob_apply(
    input_shape: tp.Shape,
    entry_prob: tp.Array2d,
    exit_prob: tp.Array2d,
    flex_2d: bool,
    entry_wait: int = 1,
    exit_wait: int = 1,
    entry_pick_first: bool = True,
    exit_pick_first: bool = True,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function used by `vectorbt.signals.generators.RPROBNX`."""
    backend = resolve_random_backend(backend)
    entry_prob = flex_broadcast_to_shape(entry_prob, input_shape, np.float64)
    exit_prob = flex_broadcast_to_shape(exit_prob, input_shape, np.float64)
    return generate_rand_enex_by_prob(
        input_shape,
        entry_prob,
        exit_prob,
        entry_wait,
        exit_wait,
        entry_pick_first,
        exit_pick_first,
        flex_2d,
        seed=seed_for_rust(
            seed,
            backend,
            combine_rust_support(
                array_compatible_with_rust(entry_prob, dtype=np.float64),
                array_compatible_with_rust(exit_prob, dtype=np.float64),
                matching_shape_compatible_with_rust("exit_prob", entry_prob, exit_prob),
            ),
        ),
        backend=backend,
    )


def rand_chain_by_prob_apply(
    entries: tp.Array2d,
    prob: tp.Array2d,
    flex_2d: bool,
    wait: int = 1,
    entry_wait: int = 1,
    exit_wait: tp.Optional[int] = None,
    entry_pick_first: bool = True,
    exit_pick_first: bool = True,
    until_next: bool = True,
    skip_until_exit: bool = False,
    seed: tp.Optional[int] = None,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function used by `vectorbt.signals.generators.RPROBCX`."""
    backend = resolve_random_backend(backend)
    if exit_wait is None:
        exit_wait = wait
    prob = flex_broadcast_to_shape(prob, entries.shape, np.float64)
    if resolve_backend(
        backend,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(entries.astype(np.float64), dtype=np.float64),
            array_compatible_with_rust(prob, dtype=np.float64),
            matching_shape_compatible_with_rust("exit_prob", entries, prob),
        ),
    ) == "numba":
        from vectorbt.signals.nb import first_choice_nb, rand_by_prob_choice_nb

        temp_idx_arr = np.empty((entries.shape[0],), dtype=np.int64)
        return generate_enex(
            entries.shape,
            entry_wait,
            exit_wait,
            entry_pick_first,
            exit_pick_first,
            first_choice_nb,
            (entries,),
            rand_by_prob_choice_nb,
            (prob, exit_pick_first, temp_idx_arr, flex_2d),
            backend=backend,
        )
    entry_prob = entries.astype(np.float64)
    return generate_rand_enex_by_prob(
        entries.shape,
        entry_prob,
        prob,
        entry_wait,
        exit_wait,
        entry_pick_first,
        exit_pick_first,
        flex_2d,
        seed=seed_for_rust(
            seed,
            backend,
            combine_rust_support(
                array_compatible_with_rust(entry_prob, dtype=np.float64),
                array_compatible_with_rust(prob, dtype=np.float64),
                matching_shape_compatible_with_rust("exit_prob", entry_prob, prob),
            ),
        ),
        backend=backend,
    )


def stop_ex_apply(
    entries: tp.Array2d,
    ts: tp.Array2d,
    stop: tp.Array2d,
    trailing: tp.Array2d,
    flex_2d: bool,
    wait: int = 1,
    until_next: bool = True,
    skip_until_exit: bool = False,
    pick_first: bool = True,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function used by `vectorbt.signals.generators.STX`."""
    ts = np.asarray(ts, dtype=np.float64)
    stop = flex_broadcast_to_shape(stop, entries.shape, np.float64)
    trailing = flex_broadcast_to_shape(trailing, entries.shape, np.bool_)
    return generate_stop_ex(
        entries,
        ts,
        stop,
        trailing,
        wait,
        until_next,
        skip_until_exit,
        pick_first,
        flex_2d,
        backend=backend,
    )


def stop_enex_apply(
    entries: tp.Array2d,
    ts: tp.Array2d,
    stop: tp.Array2d,
    trailing: tp.Array2d,
    flex_2d: bool,
    wait: int = 1,
    entry_wait: int = 1,
    exit_wait: tp.Optional[int] = None,
    pick_first: bool = True,
    until_next: bool = True,
    skip_until_exit: bool = False,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function used by `vectorbt.signals.generators.STCX`."""
    if exit_wait is None:
        exit_wait = wait
    ts = np.asarray(ts, dtype=np.float64)
    stop = flex_broadcast_to_shape(stop, entries.shape, np.float64)
    trailing = flex_broadcast_to_shape(trailing, entries.shape, np.bool_)
    return generate_stop_enex(
        entries,
        ts,
        stop,
        trailing,
        entry_wait,
        exit_wait,
        pick_first,
        flex_2d,
        backend=backend,
    )


def ohlc_stop_ex_apply(
    entries: tp.Array2d,
    open: tp.Array2d,
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    stop_price_out: tp.Array2d,
    stop_type_out: tp.Array2d,
    sl_stop: tp.Array2d,
    sl_trail: tp.Array2d,
    tp_stop: tp.Array2d,
    reverse: tp.Array2d,
    flex_2d: bool,
    is_open_safe: bool = True,
    wait: int = 1,
    until_next: bool = True,
    skip_until_exit: bool = False,
    pick_first: bool = True,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Apply function used by `vectorbt.signals.generators.OHLCSTX`."""
    open = np.asarray(open, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    sl_stop = flex_broadcast_to_shape(sl_stop, entries.shape, np.float64)
    sl_trail = flex_broadcast_to_shape(sl_trail, entries.shape, np.bool_)
    tp_stop = flex_broadcast_to_shape(tp_stop, entries.shape, np.float64)
    reverse = flex_broadcast_to_shape(reverse, entries.shape, np.bool_)
    return generate_ohlc_stop_ex(
        entries,
        open,
        high,
        low,
        close,
        stop_price_out,
        stop_type_out,
        sl_stop,
        sl_trail,
        tp_stop,
        reverse,
        is_open_safe,
        wait,
        until_next,
        skip_until_exit,
        pick_first,
        flex_2d,
        backend=backend,
    )


def ohlc_stop_enex_apply(
    entries: tp.Array2d,
    open: tp.Array2d,
    high: tp.Array2d,
    low: tp.Array2d,
    close: tp.Array2d,
    stop_price_out: tp.Array2d,
    stop_type_out: tp.Array2d,
    sl_stop: tp.Array2d,
    sl_trail: tp.Array2d,
    tp_stop: tp.Array2d,
    reverse: tp.Array2d,
    flex_2d: bool,
    is_open_safe: bool = True,
    wait: int = 1,
    entry_wait: int = 1,
    exit_wait: tp.Optional[int] = None,
    pick_first: bool = True,
    until_next: bool = True,
    skip_until_exit: bool = False,
    backend: tp.Optional[str] = None,
) -> tp.Tuple[tp.Array2d, tp.Array2d]:
    """Apply function used by `vectorbt.signals.generators.OHLCSTCX`."""
    if exit_wait is None:
        exit_wait = wait
    open = np.asarray(open, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    sl_stop = flex_broadcast_to_shape(sl_stop, entries.shape, np.float64)
    sl_trail = flex_broadcast_to_shape(sl_trail, entries.shape, np.bool_)
    tp_stop = flex_broadcast_to_shape(tp_stop, entries.shape, np.float64)
    reverse = flex_broadcast_to_shape(reverse, entries.shape, np.bool_)
    return generate_ohlc_stop_enex(
        entries,
        open,
        high,
        low,
        close,
        stop_price_out,
        stop_type_out,
        sl_stop,
        sl_trail,
        tp_stop,
        reverse,
        is_open_safe,
        entry_wait,
        exit_wait,
        pick_first,
        flex_2d,
        backend=backend,
    )


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


def rank(
    a: tp.Array2d,
    reset_by: tp.Optional[tp.Array2d],
    after_false: bool,
    rank_func: tp.RankFunc,
    *args: tp.Any,
    backend: tp.Optional[str] = None,
) -> tp.Array2d:
    """Backend-neutral `vectorbt.signals.nb.rank_nb`."""
    resolve_backend(backend, supports_rust=callback_unsupported_with_rust())
    from vectorbt.signals.nb import rank_nb

    return rank_nb(a, reset_by, after_false, rank_func, *args)


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
