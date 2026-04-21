# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Engine-neutral dispatch wrappers for portfolio functions."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt._engine import (
    RustSupport,
    array_compatible_with_rust,
    exact_array_compatible_with_rust,
    prepare_array_for_rust,
    prepare_flex_array_for_rust,
    col_map_compatible_with_rust,
    combine_rust_support,
    flex_array_compatible_with_rust,
    resolve_engine,
    resolve_random_engine,
)
from vectorbt.portfolio.enums import order_dt, trade_dt
from vectorbt.records.dispatch import record_array_compatible_with_rust


def order_record_array_compatible_with_rust(records: tp.Any) -> RustSupport:
    """Return whether order records have the exact Rust-compatible dtype."""
    support = record_array_compatible_with_rust(records)
    if not support.supported:
        return support
    if records.dtype != order_dt:
        return RustSupport(False, "Rust engine requires order records to have `order_dt` dtype.")
    return RustSupport(True)


def trade_record_array_compatible_with_rust(records: tp.Any) -> RustSupport:
    """Return whether trade records have the exact Rust-compatible dtype."""
    support = record_array_compatible_with_rust(records)
    if not support.supported:
        return support
    if records.dtype != trade_dt:
        return RustSupport(False, "Rust engine requires trade records to have `trade_dt` dtype.")
    return RustSupport(True)


# ############# Core order functions ############# #


def order_not_filled(status: int, status_info: int, engine: tp.Optional[str] = None):
    """Engine-neutral `vectorbt.portfolio.nb.order_not_filled_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import order_not_filled_rs

        return order_not_filled_rs(status, status_info)
    from vectorbt.portfolio.nb import order_not_filled_nb

    return order_not_filled_nb(status, status_info)


def buy(
    exec_state,
    size,
    price,
    direction=2,
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=0.0,
    max_size=np.inf,
    size_granularity=np.nan,
    lock_cash=False,
    allow_partial=True,
    percent=np.nan,
    engine: tp.Optional[str] = None,
):
    """Engine-neutral `vectorbt.portfolio.nb.buy_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import buy_rs

        return buy_rs(
            exec_state,
            size,
            price,
            direction,
            fees,
            fixed_fees,
            slippage,
            min_size,
            max_size,
            size_granularity,
            lock_cash,
            allow_partial,
            percent,
        )
    from vectorbt.portfolio.nb import buy_nb

    return buy_nb(
        exec_state,
        size,
        price,
        direction,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        lock_cash,
        allow_partial,
        percent,
    )


def sell(
    exec_state,
    size,
    price,
    direction=2,
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=0.0,
    max_size=np.inf,
    size_granularity=np.nan,
    lock_cash=False,
    allow_partial=True,
    percent=np.nan,
    engine: tp.Optional[str] = None,
):
    """Engine-neutral `vectorbt.portfolio.nb.sell_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import sell_rs

        return sell_rs(
            exec_state,
            size,
            price,
            direction,
            fees,
            fixed_fees,
            slippage,
            min_size,
            max_size,
            size_granularity,
            lock_cash,
            allow_partial,
            percent,
        )
    from vectorbt.portfolio.nb import sell_nb

    return sell_nb(
        exec_state,
        size,
        price,
        direction,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        lock_cash,
        allow_partial,
        percent,
    )


def execute_order(state, order, engine: tp.Optional[str] = None):
    """Engine-neutral `vectorbt.portfolio.nb.execute_order_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import execute_order_rs

        return execute_order_rs(state, order)
    from vectorbt.portfolio.nb import execute_order_nb

    return execute_order_nb(state, order)


def raise_rejected_order(order_result, engine: tp.Optional[str] = None):
    """Engine-neutral `vectorbt.portfolio.nb.raise_rejected_order_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import (
            OrderResult as RustOrderResult,
            raise_rejected_order_rs,
        )

        if not isinstance(order_result, RustOrderResult):
            order_result = RustOrderResult(
                order_result.size,
                order_result.price,
                order_result.fees,
                order_result.side,
                order_result.status,
                order_result.status_info,
            )
        return raise_rejected_order_rs(order_result)
    from vectorbt.portfolio.nb import raise_rejected_order_nb

    return raise_rejected_order_nb(order_result)


def update_value(
    cash_before,
    cash_now,
    position_before,
    position_now,
    val_price_before,
    order_price,
    value_before,
    engine: tp.Optional[str] = None,
):
    """Engine-neutral `vectorbt.portfolio.nb.update_value_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import update_value_rs

        return update_value_rs(
            cash_before,
            cash_now,
            position_before,
            position_now,
            val_price_before,
            order_price,
            value_before,
        )
    from vectorbt.portfolio.nb import update_value_nb

    return update_value_nb(
        cash_before,
        cash_now,
        position_before,
        position_now,
        val_price_before,
        order_price,
        value_before,
    )


def order(
    size=np.nan,
    price=np.inf,
    size_type=0,
    direction=2,
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=0.0,
    max_size=np.inf,
    size_granularity=np.nan,
    reject_prob=0.0,
    lock_cash=False,
    allow_partial=True,
    raise_reject=False,
    log=False,
    engine: tp.Optional[str] = None,
):
    """Engine-neutral `vectorbt.portfolio.nb.order_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import order_rs

        return order_rs(
            size,
            price,
            size_type,
            direction,
            fees,
            fixed_fees,
            slippage,
            min_size,
            max_size,
            size_granularity,
            reject_prob,
            lock_cash,
            allow_partial,
            raise_reject,
            log,
        )
    from vectorbt.portfolio.nb import order_nb as order_nb_func

    return order_nb_func(
        size,
        price,
        size_type,
        direction,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        reject_prob,
        lock_cash,
        allow_partial,
        raise_reject,
        log,
    )


def close_position(
    price=np.inf,
    fees=0.0,
    fixed_fees=0.0,
    slippage=0.0,
    min_size=0.0,
    max_size=np.inf,
    size_granularity=np.nan,
    reject_prob=0.0,
    lock_cash=False,
    allow_partial=True,
    raise_reject=False,
    log=False,
    engine: tp.Optional[str] = None,
):
    """Engine-neutral `vectorbt.portfolio.nb.close_position_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import close_position_rs

        return close_position_rs(
            price,
            fees,
            fixed_fees,
            slippage,
            min_size,
            max_size,
            size_granularity,
            reject_prob,
            lock_cash,
            allow_partial,
            raise_reject,
            log,
        )
    from vectorbt.portfolio.nb import close_position_nb

    return close_position_nb(
        price,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        reject_prob,
        lock_cash,
        allow_partial,
        raise_reject,
        log,
    )


def order_nothing(engine: tp.Optional[str] = None):
    """Engine-neutral `vectorbt.portfolio.nb.order_nothing_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import order_nothing_rs

        return order_nothing_rs()
    from vectorbt.portfolio.nb import order_nothing_nb

    return order_nothing_nb()


# ############# Call sequence & validation ############# #


def check_group_lens(group_lens: tp.Array1d, n_cols: int, engine: tp.Optional[str] = None) -> None:
    """Engine-neutral `vectorbt.portfolio.nb.check_group_lens_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=array_compatible_with_rust(group_lens, dtype=np.int64),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import check_group_lens_rs

        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return check_group_lens_rs(group_lens, n_cols)
    from vectorbt.portfolio.nb import check_group_lens_nb

    return check_group_lens_nb(group_lens, n_cols)


def check_group_init_cash(
    group_lens: tp.Array1d,
    n_cols: int,
    init_cash: tp.Array1d,
    cash_sharing: bool,
    engine: tp.Optional[str] = None,
) -> None:
    """Engine-neutral `vectorbt.portfolio.nb.check_group_init_cash_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(group_lens, dtype=np.int64),
            array_compatible_with_rust(init_cash),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import check_group_init_cash_rs

        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        return check_group_init_cash_rs(group_lens, n_cols, init_cash, cash_sharing)
    from vectorbt.portfolio.nb import check_group_init_cash_nb

    return check_group_init_cash_nb(group_lens, n_cols, init_cash, cash_sharing)


def is_grouped(group_lens: tp.Array1d, engine: tp.Optional[str] = None) -> bool:
    """Engine-neutral `vectorbt.portfolio.nb.is_grouped_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=array_compatible_with_rust(group_lens, dtype=np.int64),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import is_grouped_rs

        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return is_grouped_rs(group_lens)
    from vectorbt.portfolio.nb import is_grouped_nb

    return is_grouped_nb(group_lens)


def shuffle_call_seq(
    call_seq: tp.Array2d,
    group_lens: tp.Array1d,
    seed: tp.Optional[int] = None,
    engine: tp.Optional[str] = None,
) -> None:
    """Engine-neutral `vectorbt.portfolio.nb.shuffle_call_seq_nb`."""
    eng = resolve_random_engine(engine)
    if eng == "numba":
        from vectorbt.portfolio.nb import shuffle_call_seq_nb

        return shuffle_call_seq_nb(call_seq, group_lens)
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            exact_array_compatible_with_rust(call_seq, dtype=np.int64),
            array_compatible_with_rust(group_lens, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import shuffle_call_seq_rs

        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return shuffle_call_seq_rs(call_seq, group_lens, seed)
    from vectorbt.portfolio.nb import shuffle_call_seq_nb

    return shuffle_call_seq_nb(call_seq, group_lens)


def build_call_seq(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    call_seq_type: int = 0,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.build_call_seq_nb`."""
    if call_seq_type == 2:
        eng = resolve_random_engine(engine)
        if eng == "numba":
            from vectorbt.portfolio.nb import build_call_seq_nb

            return build_call_seq_nb(target_shape, group_lens, call_seq_type)
    eng = resolve_engine(
        engine,
        supports_rust=array_compatible_with_rust(group_lens, dtype=np.int64),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import build_call_seq_rs

        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return build_call_seq_rs(target_shape, group_lens, call_seq_type)
    from vectorbt.portfolio.nb import build_call_seq_nb

    return build_call_seq_nb(target_shape, group_lens, call_seq_type)


def get_group_value(
    from_col: int,
    to_col: int,
    cash_now: float,
    last_position: tp.Array1d,
    last_val_price: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> float:
    """Engine-neutral `vectorbt.portfolio.nb.get_group_value_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(last_position),
            array_compatible_with_rust(last_val_price),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import get_group_value_rs

        last_position = prepare_array_for_rust(last_position, dtype=np.float64)
        last_val_price = prepare_array_for_rust(last_val_price, dtype=np.float64)
        return get_group_value_rs(from_col, to_col, cash_now, last_position, last_val_price)
    from vectorbt.portfolio.nb import get_group_value_nb

    return get_group_value_nb(from_col, to_col, cash_now, last_position, last_val_price)


def approx_order_value(
    size: float,
    size_type: int,
    direction: int,
    cash_now: float,
    position: float,
    free_cash: float,
    val_price: float,
    value: float,
    engine: tp.Optional[str] = None,
) -> float:
    """Engine-neutral `vectorbt.portfolio.nb.approx_order_value_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import approx_order_value_rs

        return approx_order_value_rs(size, size_type, direction, cash_now, position, free_cash, val_price, value)
    from vectorbt.portfolio.nb import approx_order_value_nb

    return approx_order_value_nb(size, size_type, direction, cash_now, position, free_cash, val_price, value)


# ############# Simulation ############# #


def simulate_from_orders(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    init_cash: tp.Array1d,
    call_seq: tp.Array2d,
    size: tp.ArrayLike = np.asarray(np.inf),
    price: tp.ArrayLike = np.asarray(np.inf),
    size_type: tp.ArrayLike = np.asarray(0),
    direction: tp.ArrayLike = np.asarray(2),
    fees: tp.ArrayLike = np.asarray(0.0),
    fixed_fees: tp.ArrayLike = np.asarray(0.0),
    slippage: tp.ArrayLike = np.asarray(0.0),
    min_size: tp.ArrayLike = np.asarray(0.0),
    max_size: tp.ArrayLike = np.asarray(np.inf),
    size_granularity: tp.ArrayLike = np.asarray(np.nan),
    reject_prob: tp.ArrayLike = np.asarray(0.0),
    lock_cash: tp.ArrayLike = np.asarray(False),
    allow_partial: tp.ArrayLike = np.asarray(True),
    raise_reject: tp.ArrayLike = np.asarray(False),
    log: tp.ArrayLike = np.asarray(False),
    val_price: tp.ArrayLike = np.asarray(np.inf),
    close: tp.ArrayLike = np.asarray(np.nan),
    auto_call_seq: bool = False,
    ffill_val_price: bool = True,
    update_value: bool = False,
    max_orders: tp.Optional[int] = None,
    max_logs: int = 0,
    flex_2d: bool = True,
    seed: tp.Optional[int] = None,
    engine: tp.Optional[str] = None,
) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Engine-neutral `vectorbt.portfolio.nb.simulate_from_orders_nb`."""
    if np.any(np.asarray(reject_prob) > 0):
        eng = resolve_random_engine(engine)
        if eng == "numba":
            from vectorbt.portfolio.nb import simulate_from_orders_nb

            return simulate_from_orders_nb(
                target_shape,
                group_lens,
                init_cash,
                call_seq,
                size,
                price,
                size_type,
                direction,
                fees,
                fixed_fees,
                slippage,
                min_size,
                max_size,
                size_granularity,
                reject_prob,
                lock_cash,
                allow_partial,
                raise_reject,
                log,
                val_price,
                close,
                auto_call_seq=auto_call_seq,
                ffill_val_price=ffill_val_price,
                update_value=update_value,
                max_orders=max_orders,
                max_logs=max_logs,
                flex_2d=flex_2d,
            )
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(group_lens, dtype=np.int64),
            array_compatible_with_rust(init_cash),
            exact_array_compatible_with_rust(call_seq, dtype=np.int64),
            flex_array_compatible_with_rust("size", size, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("price", price, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("size_type", size_type, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust("direction", direction, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust("fees", fees, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("fixed_fees", fixed_fees, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("slippage", slippage, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("min_size", min_size, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("max_size", max_size, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("size_granularity", size_granularity, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("reject_prob", reject_prob, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("lock_cash", lock_cash, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("allow_partial", allow_partial, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("raise_reject", raise_reject, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("log", log, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("val_price", val_price, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("close", close, target_shape, np.float64, flex_2d),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import simulate_from_orders_rs

        size_arr = prepare_flex_array_for_rust(size, target_shape, np.float64, flex_2d, name="size")
        price_arr = prepare_flex_array_for_rust(price, target_shape, np.float64, flex_2d, name="price")
        size_type_arr = prepare_flex_array_for_rust(size_type, target_shape, np.int64, flex_2d, name="size_type")
        direction_arr = prepare_flex_array_for_rust(direction, target_shape, np.int64, flex_2d, name="direction")
        fees_arr = prepare_flex_array_for_rust(fees, target_shape, np.float64, flex_2d, name="fees")
        fixed_fees_arr = prepare_flex_array_for_rust(fixed_fees, target_shape, np.float64, flex_2d, name="fixed_fees")
        slippage_arr = prepare_flex_array_for_rust(slippage, target_shape, np.float64, flex_2d, name="slippage")
        min_size_arr = prepare_flex_array_for_rust(min_size, target_shape, np.float64, flex_2d, name="min_size")
        max_size_arr = prepare_flex_array_for_rust(max_size, target_shape, np.float64, flex_2d, name="max_size")
        size_granularity_arr = prepare_flex_array_for_rust(
            size_granularity,
            target_shape,
            np.float64,
            flex_2d,
            name="size_granularity",
        )
        reject_prob_arr = prepare_flex_array_for_rust(reject_prob, target_shape, np.float64, flex_2d, name="reject_prob")
        lock_cash_arr = prepare_flex_array_for_rust(lock_cash, target_shape, np.bool_, flex_2d, name="lock_cash")
        allow_partial_arr = prepare_flex_array_for_rust(
            allow_partial,
            target_shape,
            np.bool_,
            flex_2d,
            name="allow_partial",
        )
        raise_reject_arr = prepare_flex_array_for_rust(
            raise_reject,
            target_shape,
            np.bool_,
            flex_2d,
            name="raise_reject",
        )
        log_arr = prepare_flex_array_for_rust(log, target_shape, np.bool_, flex_2d, name="log")
        val_price_arr = prepare_flex_array_for_rust(val_price, target_shape, np.float64, flex_2d, name="val_price")
        close_arr = prepare_flex_array_for_rust(close, target_shape, np.float64, flex_2d, name="close")

        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        return simulate_from_orders_rs(
            target_shape,
            group_lens,
            init_cash,
            call_seq,
            size_arr,
            price_arr,
            size_type_arr,
            direction_arr,
            fees_arr,
            fixed_fees_arr,
            slippage_arr,
            min_size_arr,
            max_size_arr,
            size_granularity_arr,
            reject_prob_arr,
            lock_cash_arr,
            allow_partial_arr,
            raise_reject_arr,
            log_arr,
            val_price_arr,
            close_arr,
            auto_call_seq=auto_call_seq,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            max_orders=max_orders,
            max_logs=max_logs,
            seed=seed,
            flex_2d=flex_2d,
        )
    from vectorbt.portfolio.nb import simulate_from_orders_nb

    return simulate_from_orders_nb(
        target_shape,
        group_lens,
        init_cash,
        call_seq,
        size,
        price,
        size_type,
        direction,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        reject_prob,
        lock_cash,
        allow_partial,
        raise_reject,
        log,
        val_price,
        close,
        auto_call_seq=auto_call_seq,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        max_orders=max_orders,
        max_logs=max_logs,
        flex_2d=flex_2d,
    )


def simulate_from_signals(
    target_shape: tp.Shape,
    group_lens: tp.Array1d,
    init_cash: tp.Array1d,
    call_seq: tp.Array2d,
    entries: tp.ArrayLike = np.asarray(False),
    exits: tp.ArrayLike = np.asarray(False),
    direction: tp.ArrayLike = np.asarray(0),
    long_entries: tp.ArrayLike = np.asarray(False),
    long_exits: tp.ArrayLike = np.asarray(False),
    short_entries: tp.ArrayLike = np.asarray(False),
    short_exits: tp.ArrayLike = np.asarray(False),
    size: tp.ArrayLike = np.asarray(np.inf),
    price: tp.ArrayLike = np.asarray(np.inf),
    size_type: tp.ArrayLike = np.asarray(0),
    fees: tp.ArrayLike = np.asarray(0.0),
    fixed_fees: tp.ArrayLike = np.asarray(0.0),
    slippage: tp.ArrayLike = np.asarray(0.0),
    min_size: tp.ArrayLike = np.asarray(0.0),
    max_size: tp.ArrayLike = np.asarray(np.inf),
    size_granularity: tp.ArrayLike = np.asarray(np.nan),
    reject_prob: tp.ArrayLike = np.asarray(0.0),
    lock_cash: tp.ArrayLike = np.asarray(False),
    allow_partial: tp.ArrayLike = np.asarray(True),
    raise_reject: tp.ArrayLike = np.asarray(False),
    log: tp.ArrayLike = np.asarray(False),
    accumulate: tp.ArrayLike = np.asarray(0),
    upon_long_conflict: tp.ArrayLike = np.asarray(0),
    upon_short_conflict: tp.ArrayLike = np.asarray(0),
    upon_dir_conflict: tp.ArrayLike = np.asarray(0),
    upon_opposite_entry: tp.ArrayLike = np.asarray(4),
    val_price: tp.ArrayLike = np.asarray(np.inf),
    open: tp.ArrayLike = np.asarray(np.nan),
    high: tp.ArrayLike = np.asarray(np.nan),
    low: tp.ArrayLike = np.asarray(np.nan),
    close: tp.ArrayLike = np.asarray(np.nan),
    sl_stop: tp.ArrayLike = np.asarray(np.nan),
    sl_trail: tp.ArrayLike = np.asarray(False),
    tp_stop: tp.ArrayLike = np.asarray(np.nan),
    stop_entry_price: tp.ArrayLike = np.asarray(3),
    stop_exit_price: tp.ArrayLike = np.asarray(0),
    upon_stop_exit: tp.ArrayLike = np.asarray(0),
    upon_stop_update: tp.ArrayLike = np.asarray(1),
    use_stops: bool = True,
    auto_call_seq: bool = False,
    ffill_val_price: bool = True,
    update_value: bool = False,
    max_orders: tp.Optional[int] = None,
    max_logs: int = 0,
    flex_2d: bool = True,
    seed: tp.Optional[int] = None,
    engine: tp.Optional[str] = None,
) -> tp.Tuple[tp.RecordArray, tp.RecordArray]:
    """Engine-neutral `vectorbt.portfolio.nb.simulate_from_signals_nb`."""
    if np.any(np.asarray(reject_prob) > 0):
        eng = resolve_random_engine(engine)
        if eng == "numba":
            from vectorbt.portfolio.nb import simulate_from_signals_nb

            return simulate_from_signals_nb(
                target_shape,
                group_lens,
                init_cash,
                call_seq,
                entries,
                exits,
                direction,
                long_entries,
                long_exits,
                short_entries,
                short_exits,
                size,
                price,
                size_type,
                fees,
                fixed_fees,
                slippage,
                min_size,
                max_size,
                size_granularity,
                reject_prob,
                lock_cash,
                allow_partial,
                raise_reject,
                log,
                accumulate,
                upon_long_conflict,
                upon_short_conflict,
                upon_dir_conflict,
                upon_opposite_entry,
                val_price,
                open,
                high,
                low,
                close,
                sl_stop,
                sl_trail,
                tp_stop,
                stop_entry_price,
                stop_exit_price,
                upon_stop_exit,
                upon_stop_update,
                use_stops=use_stops,
                auto_call_seq=auto_call_seq,
                ffill_val_price=ffill_val_price,
                update_value=update_value,
                max_orders=max_orders,
                max_logs=max_logs,
                flex_2d=flex_2d,
            )
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(group_lens, dtype=np.int64),
            array_compatible_with_rust(init_cash),
            exact_array_compatible_with_rust(call_seq, dtype=np.int64),
            flex_array_compatible_with_rust("entries", entries, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("exits", exits, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("direction", direction, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust("long_entries", long_entries, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("long_exits", long_exits, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("short_entries", short_entries, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("short_exits", short_exits, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("size", size, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("price", price, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("size_type", size_type, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust("fees", fees, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("fixed_fees", fixed_fees, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("slippage", slippage, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("min_size", min_size, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("max_size", max_size, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("size_granularity", size_granularity, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("reject_prob", reject_prob, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("lock_cash", lock_cash, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("allow_partial", allow_partial, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("raise_reject", raise_reject, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("log", log, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("accumulate", accumulate, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust(
                "upon_long_conflict",
                upon_long_conflict,
                target_shape,
                np.int64,
                flex_2d,
            ),
            flex_array_compatible_with_rust(
                "upon_short_conflict",
                upon_short_conflict,
                target_shape,
                np.int64,
                flex_2d,
            ),
            flex_array_compatible_with_rust("upon_dir_conflict", upon_dir_conflict, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust(
                "upon_opposite_entry",
                upon_opposite_entry,
                target_shape,
                np.int64,
                flex_2d,
            ),
            flex_array_compatible_with_rust("val_price", val_price, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("open", open, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("high", high, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("low", low, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("close", close, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("sl_stop", sl_stop, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("sl_trail", sl_trail, target_shape, np.bool_, flex_2d),
            flex_array_compatible_with_rust("tp_stop", tp_stop, target_shape, np.float64, flex_2d),
            flex_array_compatible_with_rust("stop_entry_price", stop_entry_price, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust("stop_exit_price", stop_exit_price, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust("upon_stop_exit", upon_stop_exit, target_shape, np.int64, flex_2d),
            flex_array_compatible_with_rust("upon_stop_update", upon_stop_update, target_shape, np.int64, flex_2d),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import simulate_from_signals_rs

        entries_arr = prepare_flex_array_for_rust(entries, target_shape, np.bool_, flex_2d, name="entries")
        exits_arr = prepare_flex_array_for_rust(exits, target_shape, np.bool_, flex_2d, name="exits")
        direction_arr = prepare_flex_array_for_rust(direction, target_shape, np.int64, flex_2d, name="direction")
        long_entries_arr = prepare_flex_array_for_rust(
            long_entries,
            target_shape,
            np.bool_,
            flex_2d,
            name="long_entries",
        )
        long_exits_arr = prepare_flex_array_for_rust(long_exits, target_shape, np.bool_, flex_2d, name="long_exits")
        short_entries_arr = prepare_flex_array_for_rust(
            short_entries,
            target_shape,
            np.bool_,
            flex_2d,
            name="short_entries",
        )
        short_exits_arr = prepare_flex_array_for_rust(
            short_exits,
            target_shape,
            np.bool_,
            flex_2d,
            name="short_exits",
        )
        size_arr = prepare_flex_array_for_rust(size, target_shape, np.float64, flex_2d, name="size")
        price_arr = prepare_flex_array_for_rust(price, target_shape, np.float64, flex_2d, name="price")
        size_type_arr = prepare_flex_array_for_rust(size_type, target_shape, np.int64, flex_2d, name="size_type")
        fees_arr = prepare_flex_array_for_rust(fees, target_shape, np.float64, flex_2d, name="fees")
        fixed_fees_arr = prepare_flex_array_for_rust(fixed_fees, target_shape, np.float64, flex_2d, name="fixed_fees")
        slippage_arr = prepare_flex_array_for_rust(slippage, target_shape, np.float64, flex_2d, name="slippage")
        min_size_arr = prepare_flex_array_for_rust(min_size, target_shape, np.float64, flex_2d, name="min_size")
        max_size_arr = prepare_flex_array_for_rust(max_size, target_shape, np.float64, flex_2d, name="max_size")
        size_granularity_arr = prepare_flex_array_for_rust(
            size_granularity,
            target_shape,
            np.float64,
            flex_2d,
            name="size_granularity",
        )
        reject_prob_arr = prepare_flex_array_for_rust(reject_prob, target_shape, np.float64, flex_2d, name="reject_prob")
        lock_cash_arr = prepare_flex_array_for_rust(lock_cash, target_shape, np.bool_, flex_2d, name="lock_cash")
        allow_partial_arr = prepare_flex_array_for_rust(
            allow_partial,
            target_shape,
            np.bool_,
            flex_2d,
            name="allow_partial",
        )
        raise_reject_arr = prepare_flex_array_for_rust(
            raise_reject,
            target_shape,
            np.bool_,
            flex_2d,
            name="raise_reject",
        )
        log_arr = prepare_flex_array_for_rust(log, target_shape, np.bool_, flex_2d, name="log")
        accumulate_arr = prepare_flex_array_for_rust(accumulate, target_shape, np.int64, flex_2d, name="accumulate")
        upon_long_conflict_arr = prepare_flex_array_for_rust(
            upon_long_conflict,
            target_shape,
            np.int64,
            flex_2d,
            name="upon_long_conflict",
        )
        upon_short_conflict_arr = prepare_flex_array_for_rust(
            upon_short_conflict,
            target_shape,
            np.int64,
            flex_2d,
            name="upon_short_conflict",
        )
        upon_dir_conflict_arr = prepare_flex_array_for_rust(
            upon_dir_conflict,
            target_shape,
            np.int64,
            flex_2d,
            name="upon_dir_conflict",
        )
        upon_opposite_entry_arr = prepare_flex_array_for_rust(
            upon_opposite_entry,
            target_shape,
            np.int64,
            flex_2d,
            name="upon_opposite_entry",
        )
        val_price_arr = prepare_flex_array_for_rust(val_price, target_shape, np.float64, flex_2d, name="val_price")
        open_arr = prepare_flex_array_for_rust(open, target_shape, np.float64, flex_2d, name="open")
        high_arr = prepare_flex_array_for_rust(high, target_shape, np.float64, flex_2d, name="high")
        low_arr = prepare_flex_array_for_rust(low, target_shape, np.float64, flex_2d, name="low")
        close_arr = prepare_flex_array_for_rust(close, target_shape, np.float64, flex_2d, name="close")
        sl_stop_arr = prepare_flex_array_for_rust(sl_stop, target_shape, np.float64, flex_2d, name="sl_stop")
        sl_trail_arr = prepare_flex_array_for_rust(sl_trail, target_shape, np.bool_, flex_2d, name="sl_trail")
        tp_stop_arr = prepare_flex_array_for_rust(tp_stop, target_shape, np.float64, flex_2d, name="tp_stop")
        stop_entry_price_arr = prepare_flex_array_for_rust(
            stop_entry_price,
            target_shape,
            np.int64,
            flex_2d,
            name="stop_entry_price",
        )
        stop_exit_price_arr = prepare_flex_array_for_rust(
            stop_exit_price,
            target_shape,
            np.int64,
            flex_2d,
            name="stop_exit_price",
        )
        upon_stop_exit_arr = prepare_flex_array_for_rust(
            upon_stop_exit,
            target_shape,
            np.int64,
            flex_2d,
            name="upon_stop_exit",
        )
        upon_stop_update_arr = prepare_flex_array_for_rust(
            upon_stop_update,
            target_shape,
            np.int64,
            flex_2d,
            name="upon_stop_update",
        )

        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        return simulate_from_signals_rs(
            target_shape,
            group_lens,
            init_cash,
            call_seq,
            entries_arr,
            exits_arr,
            direction_arr,
            long_entries_arr,
            long_exits_arr,
            short_entries_arr,
            short_exits_arr,
            size_arr,
            price_arr,
            size_type_arr,
            fees_arr,
            fixed_fees_arr,
            slippage_arr,
            min_size_arr,
            max_size_arr,
            size_granularity_arr,
            reject_prob_arr,
            lock_cash_arr,
            allow_partial_arr,
            raise_reject_arr,
            log_arr,
            accumulate_arr,
            upon_long_conflict_arr,
            upon_short_conflict_arr,
            upon_dir_conflict_arr,
            upon_opposite_entry_arr,
            val_price_arr,
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            sl_stop_arr,
            sl_trail_arr,
            tp_stop_arr,
            stop_entry_price_arr,
            stop_exit_price_arr,
            upon_stop_exit_arr,
            upon_stop_update_arr,
            use_stops=use_stops,
            auto_call_seq=auto_call_seq,
            ffill_val_price=ffill_val_price,
            update_value=update_value,
            max_orders=max_orders,
            max_logs=max_logs,
            seed=seed,
            flex_2d=flex_2d,
        )
    from vectorbt.portfolio.nb import simulate_from_signals_nb

    return simulate_from_signals_nb(
        target_shape,
        group_lens,
        init_cash,
        call_seq,
        entries,
        exits,
        direction,
        long_entries,
        long_exits,
        short_entries,
        short_exits,
        size,
        price,
        size_type,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        reject_prob,
        lock_cash,
        allow_partial,
        raise_reject,
        log,
        accumulate,
        upon_long_conflict,
        upon_short_conflict,
        upon_dir_conflict,
        upon_opposite_entry,
        val_price,
        open,
        high,
        low,
        close,
        sl_stop,
        sl_trail,
        tp_stop,
        stop_entry_price,
        stop_exit_price,
        upon_stop_exit,
        upon_stop_update,
        use_stops=use_stops,
        auto_call_seq=auto_call_seq,
        ffill_val_price=ffill_val_price,
        update_value=update_value,
        max_orders=max_orders,
        max_logs=max_logs,
        flex_2d=flex_2d,
    )


# ############# Scalar helpers ############# #


def get_trade_stats(
    size: float,
    entry_price: float,
    entry_fees: float,
    exit_price: float,
    exit_fees: float,
    direction: int,
    engine: tp.Optional[str] = None,
) -> tp.Tuple[float, float]:
    """Engine-neutral `vectorbt.portfolio.nb.get_trade_stats_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import get_trade_stats_rs

        return get_trade_stats_rs(size, entry_price, entry_fees, exit_price, exit_fees, direction)
    from vectorbt.portfolio.nb import get_trade_stats_nb

    return get_trade_stats_nb(size, entry_price, entry_fees, exit_price, exit_fees, direction)


def get_entry_trades(
    order_records: tp.RecordArray,
    close: tp.Array2d,
    col_map: tp.ColMap,
    engine: tp.Optional[str] = None,
) -> tp.RecordArray:
    """Engine-neutral `vectorbt.portfolio.nb.get_entry_trades_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            order_record_array_compatible_with_rust(order_records),
            array_compatible_with_rust(close),
            col_map_compatible_with_rust(col_map),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import get_entry_trades_rs

        col_idxs, col_lens = col_map
        close = prepare_array_for_rust(close, dtype=np.float64)
        col_idxs = prepare_array_for_rust(col_idxs, dtype=np.int64)
        col_lens = prepare_array_for_rust(col_lens, dtype=np.int64)
        return get_entry_trades_rs(order_records, close, col_idxs, col_lens)
    from vectorbt.portfolio.nb import get_entry_trades_nb

    return get_entry_trades_nb(order_records, close, col_map)


def get_exit_trades(
    order_records: tp.RecordArray,
    close: tp.Array2d,
    col_map: tp.ColMap,
    engine: tp.Optional[str] = None,
) -> tp.RecordArray:
    """Engine-neutral `vectorbt.portfolio.nb.get_exit_trades_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            order_record_array_compatible_with_rust(order_records),
            array_compatible_with_rust(close),
            col_map_compatible_with_rust(col_map),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import get_exit_trades_rs

        col_idxs, col_lens = col_map
        close = prepare_array_for_rust(close, dtype=np.float64)
        col_idxs = prepare_array_for_rust(col_idxs, dtype=np.int64)
        col_lens = prepare_array_for_rust(col_lens, dtype=np.int64)
        return get_exit_trades_rs(order_records, close, col_idxs, col_lens)
    from vectorbt.portfolio.nb import get_exit_trades_nb

    return get_exit_trades_nb(order_records, close, col_map)


def trade_winning_streak(
    records: tp.RecordArray,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.trade_winning_streak_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=trade_record_array_compatible_with_rust(records),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import trade_winning_streak_rs

        return trade_winning_streak_rs(records)
    from vectorbt.portfolio.nb import trade_winning_streak_nb

    return trade_winning_streak_nb(records)


# ############# Asset flow & position ############# #


def trade_losing_streak(
    records: tp.RecordArray,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.trade_losing_streak_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=trade_record_array_compatible_with_rust(records),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import trade_losing_streak_rs

        return trade_losing_streak_rs(records)
    from vectorbt.portfolio.nb import trade_losing_streak_nb

    return trade_losing_streak_nb(records)


def get_positions(
    trade_records: tp.RecordArray,
    col_map: tp.ColMap,
    engine: tp.Optional[str] = None,
) -> tp.RecordArray:
    """Engine-neutral `vectorbt.portfolio.nb.get_positions_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            trade_record_array_compatible_with_rust(trade_records),
            col_map_compatible_with_rust(col_map),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import get_positions_rs

        col_idxs, col_lens = col_map
        col_idxs = prepare_array_for_rust(col_idxs, dtype=np.int64)
        col_lens = prepare_array_for_rust(col_lens, dtype=np.int64)
        return get_positions_rs(trade_records, col_idxs, col_lens)
    from vectorbt.portfolio.nb import get_positions_nb

    return get_positions_nb(trade_records, col_map)


# ############# Cash flow & liquidity ############# #


def get_long_size(position_before: float, position_now: float, engine: tp.Optional[str] = None) -> float:
    """Engine-neutral `vectorbt.portfolio.nb.get_long_size_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import get_long_size_rs

        return get_long_size_rs(position_before, position_now)
    from vectorbt.portfolio.nb import get_long_size_nb

    return get_long_size_nb(position_before, position_now)


def get_short_size(position_before: float, position_now: float, engine: tp.Optional[str] = None) -> float:
    """Engine-neutral `vectorbt.portfolio.nb.get_short_size_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import get_short_size_rs

        return get_short_size_rs(position_before, position_now)
    from vectorbt.portfolio.nb import get_short_size_nb

    return get_short_size_nb(position_before, position_now)


def asset_flow(
    target_shape: tp.Shape,
    order_records: tp.RecordArray,
    col_map: tp.ColMap,
    direction: int,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.asset_flow_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            order_record_array_compatible_with_rust(order_records),
            col_map_compatible_with_rust(col_map),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import asset_flow_rs

        col_idxs, col_lens = col_map
        col_idxs = prepare_array_for_rust(col_idxs, dtype=np.int64)
        col_lens = prepare_array_for_rust(col_lens, dtype=np.int64)
        return asset_flow_rs(order_records, col_idxs, col_lens, target_shape, direction)
    from vectorbt.portfolio.nb import asset_flow_nb

    return asset_flow_nb(target_shape, order_records, col_map, direction)


def assets(asset_flow: tp.Array2d, engine: tp.Optional[str] = None) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.assets_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=array_compatible_with_rust(asset_flow),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import assets_rs

        asset_flow = prepare_array_for_rust(asset_flow, dtype=np.float64)
        return assets_rs(asset_flow)
    from vectorbt.portfolio.nb import assets_nb

    return assets_nb(asset_flow)


def get_free_cash_diff(
    position_before: float,
    position_now: float,
    debt_now: float,
    price: float,
    fees: float,
    engine: tp.Optional[str] = None,
) -> tp.Tuple[float, float]:
    """Engine-neutral `vectorbt.portfolio.nb.get_free_cash_diff_nb`."""
    eng = resolve_engine(engine, supports_rust=RustSupport(True))
    if eng == "rust":
        from vectorbt_rust.portfolio import get_free_cash_diff_rs

        return get_free_cash_diff_rs(position_before, position_now, debt_now, price, fees)
    from vectorbt.portfolio.nb import get_free_cash_diff_nb

    return get_free_cash_diff_nb(position_before, position_now, debt_now, price, fees)


def cash_flow(
    target_shape: tp.Shape,
    order_records: tp.RecordArray,
    col_map: tp.ColMap,
    free: bool,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.cash_flow_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            order_record_array_compatible_with_rust(order_records),
            col_map_compatible_with_rust(col_map),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import cash_flow_rs

        col_idxs, col_lens = col_map
        col_idxs = prepare_array_for_rust(col_idxs, dtype=np.int64)
        col_lens = prepare_array_for_rust(col_lens, dtype=np.int64)
        return cash_flow_rs(order_records, col_idxs, col_lens, target_shape, free)
    from vectorbt.portfolio.nb import cash_flow_nb

    return cash_flow_nb(target_shape, order_records, col_map, free)


def sum_grouped(a: tp.Array2d, group_lens: tp.Array1d, engine: tp.Optional[str] = None) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.sum_grouped_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(a),
            array_compatible_with_rust(group_lens, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import sum_grouped_rs

        a = prepare_array_for_rust(a, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return sum_grouped_rs(a, group_lens)
    from vectorbt.portfolio.nb import sum_grouped_nb

    return sum_grouped_nb(a, group_lens)


def cash_flow_grouped(
    cash_flow: tp.Array2d,
    group_lens: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.cash_flow_grouped_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(cash_flow),
            array_compatible_with_rust(group_lens, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import cash_flow_grouped_rs

        cash_flow = prepare_array_for_rust(cash_flow, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return cash_flow_grouped_rs(cash_flow, group_lens)
    from vectorbt.portfolio.nb import cash_flow_grouped_nb

    return cash_flow_grouped_nb(cash_flow, group_lens)


# ############# Performance metrics ############# #


def init_cash_grouped(
    init_cash: tp.Array1d,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.init_cash_grouped_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(init_cash),
            array_compatible_with_rust(group_lens, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import init_cash_grouped_rs

        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return init_cash_grouped_rs(init_cash, group_lens, cash_sharing)
    from vectorbt.portfolio.nb import init_cash_grouped_nb

    return init_cash_grouped_nb(init_cash, group_lens, cash_sharing)


def init_cash(
    init_cash: tp.Array1d,
    group_lens: tp.Array1d,
    cash_sharing: bool,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.init_cash_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(init_cash),
            array_compatible_with_rust(group_lens, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import init_cash_rs

        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return init_cash_rs(init_cash, group_lens, cash_sharing)
    from vectorbt.portfolio.nb import init_cash_nb

    return init_cash_nb(init_cash, group_lens, cash_sharing)


def cash(cash_flow: tp.Array2d, init_cash: tp.Array1d, engine: tp.Optional[str] = None) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.cash_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(cash_flow),
            array_compatible_with_rust(init_cash),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import cash_rs

        cash_flow = prepare_array_for_rust(cash_flow, dtype=np.float64)
        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        return cash_rs(cash_flow, init_cash)
    from vectorbt.portfolio.nb import cash_nb

    return cash_nb(cash_flow, init_cash)


def cash_in_sim_order(
    cash_flow: tp.Array2d,
    group_lens: tp.Array1d,
    init_cash_grouped: tp.Array1d,
    call_seq: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.cash_in_sim_order_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(cash_flow),
            array_compatible_with_rust(group_lens, dtype=np.int64),
            array_compatible_with_rust(init_cash_grouped),
            array_compatible_with_rust(call_seq, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import cash_in_sim_order_rs

        cash_flow = prepare_array_for_rust(cash_flow, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        init_cash_grouped = prepare_array_for_rust(init_cash_grouped, dtype=np.float64)
        call_seq = prepare_array_for_rust(call_seq, dtype=np.int64)
        return cash_in_sim_order_rs(cash_flow, group_lens, init_cash_grouped, call_seq)
    from vectorbt.portfolio.nb import cash_in_sim_order_nb

    return cash_in_sim_order_nb(cash_flow, group_lens, init_cash_grouped, call_seq)


def cash_grouped(
    target_shape: tp.Shape,
    cash_flow_grouped: tp.Array2d,
    group_lens: tp.Array1d,
    init_cash_grouped: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.cash_grouped_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(cash_flow_grouped),
            array_compatible_with_rust(group_lens, dtype=np.int64),
            array_compatible_with_rust(init_cash_grouped),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import cash_grouped_rs

        cash_flow_grouped = prepare_array_for_rust(cash_flow_grouped, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        init_cash_grouped = prepare_array_for_rust(init_cash_grouped, dtype=np.float64)
        return cash_grouped_rs(target_shape, cash_flow_grouped, group_lens, init_cash_grouped)
    from vectorbt.portfolio.nb import cash_grouped_nb

    return cash_grouped_nb(target_shape, cash_flow_grouped, group_lens, init_cash_grouped)


def asset_value(close: tp.Array2d, assets: tp.Array2d, engine: tp.Optional[str] = None) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.asset_value_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close),
            array_compatible_with_rust(assets),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import asset_value_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        assets = prepare_array_for_rust(assets, dtype=np.float64)
        return asset_value_rs(close, assets)
    from vectorbt.portfolio.nb import asset_value_nb

    return asset_value_nb(close, assets)


def asset_value_grouped(
    asset_value: tp.Array2d,
    group_lens: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.asset_value_grouped_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(asset_value),
            array_compatible_with_rust(group_lens, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import asset_value_grouped_rs

        asset_value = prepare_array_for_rust(asset_value, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return asset_value_grouped_rs(asset_value, group_lens)
    from vectorbt.portfolio.nb import asset_value_grouped_nb

    return asset_value_grouped_nb(asset_value, group_lens)


def value_in_sim_order(
    cash: tp.Array2d,
    asset_value: tp.Array2d,
    group_lens: tp.Array1d,
    call_seq: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.value_in_sim_order_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(cash),
            array_compatible_with_rust(asset_value),
            array_compatible_with_rust(group_lens, dtype=np.int64),
            array_compatible_with_rust(call_seq, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import value_in_sim_order_rs

        cash = prepare_array_for_rust(cash, dtype=np.float64)
        asset_value = prepare_array_for_rust(asset_value, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        call_seq = prepare_array_for_rust(call_seq, dtype=np.int64)
        return value_in_sim_order_rs(cash, asset_value, group_lens, call_seq)
    from vectorbt.portfolio.nb import value_in_sim_order_nb

    return value_in_sim_order_nb(cash, asset_value, group_lens, call_seq)


def value(cash: tp.Array2d, asset_value: tp.Array2d, engine: tp.Optional[str] = None) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.value_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(cash),
            array_compatible_with_rust(asset_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import value_rs

        cash = prepare_array_for_rust(cash, dtype=np.float64)
        asset_value = prepare_array_for_rust(asset_value, dtype=np.float64)
        return value_rs(cash, asset_value)
    from vectorbt.portfolio.nb import value_nb

    return value_nb(cash, asset_value)


def total_profit(
    target_shape: tp.Shape,
    close: tp.Array2d,
    order_records: tp.RecordArray,
    col_map: tp.ColMap,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.total_profit_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            order_record_array_compatible_with_rust(order_records),
            col_map_compatible_with_rust(col_map),
            array_compatible_with_rust(close),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import total_profit_rs

        col_idxs, col_lens = col_map
        close = prepare_array_for_rust(close, dtype=np.float64)
        col_idxs = prepare_array_for_rust(col_idxs, dtype=np.int64)
        col_lens = prepare_array_for_rust(col_lens, dtype=np.int64)
        return total_profit_rs(target_shape, close, order_records, col_idxs, col_lens)
    from vectorbt.portfolio.nb import total_profit_nb

    return total_profit_nb(target_shape, close, order_records, col_map)


def total_profit_grouped(
    total_profit: tp.Array1d,
    group_lens: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.total_profit_grouped_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(total_profit),
            array_compatible_with_rust(group_lens, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import total_profit_grouped_rs

        total_profit = prepare_array_for_rust(total_profit, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        return total_profit_grouped_rs(total_profit, group_lens)
    from vectorbt.portfolio.nb import total_profit_grouped_nb

    return total_profit_grouped_nb(total_profit, group_lens)


def final_value(
    total_profit: tp.Array1d,
    init_cash: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.final_value_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(total_profit),
            array_compatible_with_rust(init_cash),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import final_value_rs

        total_profit = prepare_array_for_rust(total_profit, dtype=np.float64)
        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        return final_value_rs(total_profit, init_cash)
    from vectorbt.portfolio.nb import final_value_nb

    return final_value_nb(total_profit, init_cash)


def total_return(
    total_profit: tp.Array1d,
    init_cash: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.total_return_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(total_profit),
            array_compatible_with_rust(init_cash),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import total_return_rs

        total_profit = prepare_array_for_rust(total_profit, dtype=np.float64)
        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        return total_return_rs(total_profit, init_cash)
    from vectorbt.portfolio.nb import total_return_nb

    return total_return_nb(total_profit, init_cash)


def returns_in_sim_order(
    value_iso: tp.Array2d,
    group_lens: tp.Array1d,
    init_cash_grouped: tp.Array1d,
    call_seq: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.returns_in_sim_order_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(value_iso),
            array_compatible_with_rust(group_lens, dtype=np.int64),
            array_compatible_with_rust(init_cash_grouped),
            array_compatible_with_rust(call_seq, dtype=np.int64),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import returns_in_sim_order_rs

        value_iso = prepare_array_for_rust(value_iso, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        init_cash_grouped = prepare_array_for_rust(init_cash_grouped, dtype=np.float64)
        call_seq = prepare_array_for_rust(call_seq, dtype=np.int64)
        return returns_in_sim_order_rs(value_iso, group_lens, init_cash_grouped, call_seq)
    from vectorbt.portfolio.nb import returns_in_sim_order_nb

    return returns_in_sim_order_nb(value_iso, group_lens, init_cash_grouped, call_seq)


# ############# Trade/position records ############# #


def asset_returns(
    cash_flow: tp.Array2d,
    asset_value: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.asset_returns_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(cash_flow),
            array_compatible_with_rust(asset_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import asset_returns_rs

        cash_flow = prepare_array_for_rust(cash_flow, dtype=np.float64)
        asset_value = prepare_array_for_rust(asset_value, dtype=np.float64)
        return asset_returns_rs(cash_flow, asset_value)
    from vectorbt.portfolio.nb import asset_returns_nb

    return asset_returns_nb(cash_flow, asset_value)


def benchmark_value(
    close: tp.Array2d,
    init_cash: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.benchmark_value_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close),
            array_compatible_with_rust(init_cash),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import benchmark_value_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        init_cash = prepare_array_for_rust(init_cash, dtype=np.float64)
        return benchmark_value_rs(close, init_cash)
    from vectorbt.portfolio.nb import benchmark_value_nb

    return benchmark_value_nb(close, init_cash)


def benchmark_value_grouped(
    close: tp.Array2d,
    group_lens: tp.Array1d,
    init_cash_grouped: tp.Array1d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.benchmark_value_grouped_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(close),
            array_compatible_with_rust(group_lens, dtype=np.int64),
            array_compatible_with_rust(init_cash_grouped),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import benchmark_value_grouped_rs

        close = prepare_array_for_rust(close, dtype=np.float64)
        group_lens = prepare_array_for_rust(group_lens, dtype=np.int64)
        init_cash_grouped = prepare_array_for_rust(init_cash_grouped, dtype=np.float64)
        return benchmark_value_grouped_rs(close, group_lens, init_cash_grouped)
    from vectorbt.portfolio.nb import benchmark_value_grouped_nb

    return benchmark_value_grouped_nb(close, group_lens, init_cash_grouped)


def total_benchmark_return(
    benchmark_value: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.portfolio.nb.total_benchmark_return_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=array_compatible_with_rust(benchmark_value),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import total_benchmark_return_rs

        benchmark_value = prepare_array_for_rust(benchmark_value, dtype=np.float64)
        return total_benchmark_return_rs(benchmark_value)
    from vectorbt.portfolio.nb import total_benchmark_return_nb

    return total_benchmark_return_nb(benchmark_value)


def gross_exposure(
    asset_value: tp.Array2d,
    cash: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.portfolio.nb.gross_exposure_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(asset_value),
            array_compatible_with_rust(cash),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.portfolio import gross_exposure_rs

        asset_value = prepare_array_for_rust(asset_value, dtype=np.float64)
        cash = prepare_array_for_rust(cash, dtype=np.float64)
        return gross_exposure_rs(asset_value, cash)
    from vectorbt.portfolio.nb import gross_exposure_nb

    return gross_exposure_nb(asset_value, cash)
