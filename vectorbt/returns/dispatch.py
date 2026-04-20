# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Engine-neutral dispatch wrappers for returns functions."""

import numpy as np

from vectorbt import _typing as tp
from vectorbt._engine import (
    RustSupport,
    array_compatible_with_rust,
    prepare_array_for_rust,
    combine_rust_support,
    matching_shape_compatible_with_rust,
    non_neg_int_compatible_with_rust,
    resolve_engine,
    rolling_compatible_with_rust,
    scalar_compatible_with_rust,
    unit_interval_compatible_with_rust,
)


def returns_init_value_compatible_with_rust(value: tp.Any, init_value: tp.Any) -> RustSupport:
    """Return whether `init_value` matches the number of columns in `value`."""
    value_support = array_compatible_with_rust(value)
    if not value_support.supported:
        return value_support
    init_support = array_compatible_with_rust(init_value)
    if not init_support.supported:
        return init_support
    if value.ndim != 2:
        return RustSupport(False, "Rust engine requires `value` to be a 2D NumPy array.")
    if init_value.ndim != 1:
        return RustSupport(False, "Rust engine requires `init_value` to be a 1D NumPy array.")
    if init_value.shape[0] != value.shape[1]:
        return RustSupport(False, "Rust engine requires `init_value` to match the number of columns in `value`.")
    return RustSupport(True)


def get_return(input_value: float, output_value: float, engine: tp.Optional[str] = None) -> float:
    """Engine-neutral `vectorbt.returns.nb.get_return_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            scalar_compatible_with_rust("input_value", input_value),
            scalar_compatible_with_rust("output_value", output_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import get_return_rs

        return get_return_rs(input_value, output_value)
    from vectorbt.returns.nb import get_return_nb

    return get_return_nb(input_value, output_value)


def returns_1d(value: tp.Array1d, init_value: float, engine: tp.Optional[str] = None) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.returns_1d_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(value),
            scalar_compatible_with_rust("init_value", init_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import returns_1d_rs

        value = prepare_array_for_rust(value, dtype=np.float64)
        return returns_1d_rs(value, init_value)
    from vectorbt.returns.nb import returns_1d_nb

    return returns_1d_nb(value, init_value)


def returns(value: tp.Array2d, init_value: tp.Array1d, engine: tp.Optional[str] = None) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.returns_nb`."""
    eng = resolve_engine(engine, supports_rust=returns_init_value_compatible_with_rust(value, init_value))
    if eng == "rust":
        from vectorbt_rust.returns import returns_rs

        value = prepare_array_for_rust(value, dtype=np.float64)
        init_value = prepare_array_for_rust(init_value, dtype=np.float64)
        return returns_rs(value, init_value)
    from vectorbt.returns.nb import returns_nb

    return returns_nb(value, init_value)


def cum_returns_1d(returns: tp.Array1d, start_value: float, engine: tp.Optional[str] = None) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.cum_returns_1d_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("start_value", start_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import cum_returns_1d_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return cum_returns_1d_rs(returns, start_value)
    from vectorbt.returns.nb import cum_returns_1d_nb

    return cum_returns_1d_nb(returns, start_value)


def cum_returns(returns: tp.Array2d, start_value: float, engine: tp.Optional[str] = None) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.cum_returns_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("start_value", start_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import cum_returns_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return cum_returns_rs(returns, start_value)
    from vectorbt.returns.nb import cum_returns_nb

    return cum_returns_nb(returns, start_value)


def cum_returns_final_1d(
    returns: tp.Array1d,
    start_value: float = 0.0,
    engine: tp.Optional[str] = None,
) -> float:
    """Engine-neutral `vectorbt.returns.nb.cum_returns_final_1d_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("start_value", start_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import cum_returns_final_1d_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return cum_returns_final_1d_rs(returns, start_value)
    from vectorbt.returns.nb import cum_returns_final_1d_nb

    return cum_returns_final_1d_nb(returns, start_value)


def cum_returns_final(
    returns: tp.Array2d,
    start_value: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.cum_returns_final_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("start_value", start_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import cum_returns_final_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return cum_returns_final_rs(returns, start_value)
    from vectorbt.returns.nb import cum_returns_final_nb

    return cum_returns_final_nb(returns, start_value)


def rolling_cum_returns_final(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    start_value: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_cum_returns_final_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            scalar_compatible_with_rust("start_value", start_value),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_cum_returns_final_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_cum_returns_final_rs(returns, window, minp, start_value)
    from vectorbt.returns.nb import rolling_cum_returns_final_nb

    return rolling_cum_returns_final_nb(returns, window, minp, start_value)


def annualized_return(returns: tp.Array2d, ann_factor: float, engine: tp.Optional[str] = None) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.annualized_return_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import annualized_return_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return annualized_return_rs(returns, ann_factor)
    from vectorbt.returns.nb import annualized_return_nb

    return annualized_return_nb(returns, ann_factor)


def rolling_annualized_return(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    ann_factor: float,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_annualized_return_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_annualized_return_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_annualized_return_rs(returns, window, minp, ann_factor)
    from vectorbt.returns.nb import rolling_annualized_return_nb

    return rolling_annualized_return_nb(returns, window, minp, ann_factor)


def annualized_volatility(
    returns: tp.Array2d,
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 1,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.annualized_volatility_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("levy_alpha", levy_alpha),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import annualized_volatility_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return annualized_volatility_rs(returns, ann_factor, levy_alpha, ddof)
    from vectorbt.returns.nb import annualized_volatility_nb

    return annualized_volatility_nb(returns, ann_factor, levy_alpha, ddof)


def rolling_annualized_volatility(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    ann_factor: float,
    levy_alpha: float = 2.0,
    ddof: int = 1,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_annualized_volatility_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("levy_alpha", levy_alpha),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_annualized_volatility_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_annualized_volatility_rs(returns, window, minp, ann_factor, levy_alpha, ddof)
    from vectorbt.returns.nb import rolling_annualized_volatility_nb

    return rolling_annualized_volatility_nb(returns, window, minp, ann_factor, levy_alpha, ddof)


def drawdown(returns: tp.Array2d, engine: tp.Optional[str] = None) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.drawdown_nb`."""
    eng = resolve_engine(engine, supports_rust=array_compatible_with_rust(returns))
    if eng == "rust":
        from vectorbt_rust.returns import drawdown_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return drawdown_rs(returns)
    from vectorbt.returns.nb import drawdown_nb

    return drawdown_nb(returns)


def max_drawdown(returns: tp.Array2d, engine: tp.Optional[str] = None) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.max_drawdown_nb`."""
    eng = resolve_engine(engine, supports_rust=array_compatible_with_rust(returns))
    if eng == "rust":
        from vectorbt_rust.returns import max_drawdown_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return max_drawdown_rs(returns)
    from vectorbt.returns.nb import max_drawdown_nb

    return max_drawdown_nb(returns)


def rolling_max_drawdown(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_max_drawdown_nb`."""
    eng = resolve_engine(engine, supports_rust=rolling_compatible_with_rust(returns, window, minp))
    if eng == "rust":
        from vectorbt_rust.returns import rolling_max_drawdown_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_max_drawdown_rs(returns, window, minp)
    from vectorbt.returns.nb import rolling_max_drawdown_nb

    return rolling_max_drawdown_nb(returns, window, minp)


def calmar_ratio(returns: tp.Array2d, ann_factor: float, engine: tp.Optional[str] = None) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.calmar_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import calmar_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return calmar_ratio_rs(returns, ann_factor)
    from vectorbt.returns.nb import calmar_ratio_nb

    return calmar_ratio_nb(returns, ann_factor)


def rolling_calmar_ratio(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    ann_factor: float,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_calmar_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_calmar_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_calmar_ratio_rs(returns, window, minp, ann_factor)
    from vectorbt.returns.nb import rolling_calmar_ratio_nb

    return rolling_calmar_ratio_nb(returns, window, minp, ann_factor)


def omega_ratio(
    returns: tp.Array2d,
    ann_factor: float,
    risk_free: float = 0.0,
    required_return: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.omega_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("risk_free", risk_free),
            scalar_compatible_with_rust("required_return", required_return),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import omega_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return omega_ratio_rs(returns, ann_factor, risk_free, required_return)
    from vectorbt.returns.nb import omega_ratio_nb

    return omega_ratio_nb(returns, ann_factor, risk_free, required_return)


def rolling_omega_ratio(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    ann_factor: float,
    risk_free: float = 0.0,
    required_return: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_omega_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("risk_free", risk_free),
            scalar_compatible_with_rust("required_return", required_return),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_omega_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_omega_ratio_rs(returns, window, minp, ann_factor, risk_free, required_return)
    from vectorbt.returns.nb import rolling_omega_ratio_nb

    return rolling_omega_ratio_nb(returns, window, minp, ann_factor, risk_free, required_return)


def sharpe_ratio(
    returns: tp.Array2d,
    ann_factor: float,
    risk_free: float = 0.0,
    ddof: int = 1,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.sharpe_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("risk_free", risk_free),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import sharpe_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return sharpe_ratio_rs(returns, ann_factor, risk_free, ddof)
    from vectorbt.returns.nb import sharpe_ratio_nb

    return sharpe_ratio_nb(returns, ann_factor, risk_free, ddof)


def rolling_sharpe_ratio(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    ann_factor: float,
    risk_free: float = 0.0,
    ddof: int = 1,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_sharpe_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("risk_free", risk_free),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_sharpe_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_sharpe_ratio_rs(returns, window, minp, ann_factor, risk_free, ddof)
    from vectorbt.returns.nb import rolling_sharpe_ratio_nb

    return rolling_sharpe_ratio_nb(returns, window, minp, ann_factor, risk_free, ddof)


def downside_risk(
    returns: tp.Array2d,
    ann_factor: float,
    required_return: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.downside_risk_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("required_return", required_return),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import downside_risk_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return downside_risk_rs(returns, ann_factor, required_return)
    from vectorbt.returns.nb import downside_risk_nb

    return downside_risk_nb(returns, ann_factor, required_return)


def rolling_downside_risk(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    ann_factor: float,
    required_return: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_downside_risk_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("required_return", required_return),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_downside_risk_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_downside_risk_rs(returns, window, minp, ann_factor, required_return)
    from vectorbt.returns.nb import rolling_downside_risk_nb

    return rolling_downside_risk_nb(returns, window, minp, ann_factor, required_return)


def sortino_ratio(
    returns: tp.Array2d,
    ann_factor: float,
    required_return: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.sortino_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("required_return", required_return),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import sortino_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return sortino_ratio_rs(returns, ann_factor, required_return)
    from vectorbt.returns.nb import sortino_ratio_nb

    return sortino_ratio_nb(returns, ann_factor, required_return)


def rolling_sortino_ratio(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    ann_factor: float,
    required_return: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_sortino_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("required_return", required_return),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_sortino_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_sortino_ratio_rs(returns, window, minp, ann_factor, required_return)
    from vectorbt.returns.nb import rolling_sortino_ratio_nb

    return rolling_sortino_ratio_nb(returns, window, minp, ann_factor, required_return)


def information_ratio(
    returns: tp.Array2d,
    benchmark_rets: tp.Array2d,
    ddof: int = 1,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.information_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import information_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return information_ratio_rs(returns, benchmark_rets, ddof)
    from vectorbt.returns.nb import information_ratio_nb

    return information_ratio_nb(returns, benchmark_rets, ddof)


def rolling_information_ratio(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    benchmark_rets: tp.Array2d,
    ddof: int = 1,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_information_ratio_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            non_neg_int_compatible_with_rust("ddof", ddof),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_information_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return rolling_information_ratio_rs(returns, window, minp, benchmark_rets, ddof)
    from vectorbt.returns.nb import rolling_information_ratio_nb

    return rolling_information_ratio_nb(returns, window, minp, benchmark_rets, ddof)


def beta(returns: tp.Array2d, benchmark_rets: tp.Array2d, engine: tp.Optional[str] = None) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.beta_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import beta_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return beta_rs(returns, benchmark_rets)
    from vectorbt.returns.nb import beta_nb

    return beta_nb(returns, benchmark_rets)


def rolling_beta(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    benchmark_rets: tp.Array2d,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_beta_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_beta_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return rolling_beta_rs(returns, window, minp, benchmark_rets)
    from vectorbt.returns.nb import rolling_beta_nb

    return rolling_beta_nb(returns, window, minp, benchmark_rets)


def alpha(
    returns: tp.Array2d,
    benchmark_rets: tp.Array2d,
    ann_factor: float,
    risk_free: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.alpha_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("risk_free", risk_free),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import alpha_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return alpha_rs(returns, benchmark_rets, ann_factor, risk_free)
    from vectorbt.returns.nb import alpha_nb

    return alpha_nb(returns, benchmark_rets, ann_factor, risk_free)


def rolling_alpha(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    benchmark_rets: tp.Array2d,
    ann_factor: float,
    risk_free: float = 0.0,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_alpha_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            scalar_compatible_with_rust("ann_factor", ann_factor),
            scalar_compatible_with_rust("risk_free", risk_free),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_alpha_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return rolling_alpha_rs(returns, window, minp, benchmark_rets, ann_factor, risk_free)
    from vectorbt.returns.nb import rolling_alpha_nb

    return rolling_alpha_nb(returns, window, minp, benchmark_rets, ann_factor, risk_free)


def tail_ratio(returns: tp.Array2d, engine: tp.Optional[str] = None) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.tail_ratio_nb`."""
    eng = resolve_engine(engine, supports_rust=array_compatible_with_rust(returns))
    if eng == "rust":
        from vectorbt_rust.returns import tail_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return tail_ratio_rs(returns)
    from vectorbt.returns.nb import tail_ratio_nb

    return tail_ratio_nb(returns)


def rolling_tail_ratio(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_tail_ratio_nb`."""
    eng = resolve_engine(engine, supports_rust=rolling_compatible_with_rust(returns, window, minp))
    if eng == "rust":
        from vectorbt_rust.returns import rolling_tail_ratio_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_tail_ratio_rs(returns, window, minp)
    from vectorbt.returns.nb import rolling_tail_ratio_nb

    return rolling_tail_ratio_nb(returns, window, minp)


def value_at_risk(
    returns: tp.Array2d,
    cutoff: float = 0.05,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.value_at_risk_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            unit_interval_compatible_with_rust("cutoff", cutoff),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import value_at_risk_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return value_at_risk_rs(returns, cutoff)
    from vectorbt.returns.nb import value_at_risk_nb

    return value_at_risk_nb(returns, cutoff)


def rolling_value_at_risk(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    cutoff: float = 0.05,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_value_at_risk_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            unit_interval_compatible_with_rust("cutoff", cutoff),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_value_at_risk_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_value_at_risk_rs(returns, window, minp, cutoff)
    from vectorbt.returns.nb import rolling_value_at_risk_nb

    return rolling_value_at_risk_nb(returns, window, minp, cutoff)


def cond_value_at_risk(
    returns: tp.Array2d,
    cutoff: float = 0.05,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.cond_value_at_risk_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            unit_interval_compatible_with_rust("cutoff", cutoff),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import cond_value_at_risk_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return cond_value_at_risk_rs(returns, cutoff)
    from vectorbt.returns.nb import cond_value_at_risk_nb

    return cond_value_at_risk_nb(returns, cutoff)


def rolling_cond_value_at_risk(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    cutoff: float = 0.05,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_cond_value_at_risk_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            unit_interval_compatible_with_rust("cutoff", cutoff),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_cond_value_at_risk_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        return rolling_cond_value_at_risk_rs(returns, window, minp, cutoff)
    from vectorbt.returns.nb import rolling_cond_value_at_risk_nb

    return rolling_cond_value_at_risk_nb(returns, window, minp, cutoff)


def capture(
    returns: tp.Array2d,
    benchmark_rets: tp.Array2d,
    ann_factor: float,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.capture_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import capture_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return capture_rs(returns, benchmark_rets, ann_factor)
    from vectorbt.returns.nb import capture_nb

    return capture_nb(returns, benchmark_rets, ann_factor)


def rolling_capture(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    benchmark_rets: tp.Array2d,
    ann_factor: float,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_capture_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_capture_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return rolling_capture_rs(returns, window, minp, benchmark_rets, ann_factor)
    from vectorbt.returns.nb import rolling_capture_nb

    return rolling_capture_nb(returns, window, minp, benchmark_rets, ann_factor)


def up_capture(
    returns: tp.Array2d,
    benchmark_rets: tp.Array2d,
    ann_factor: float,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.up_capture_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import up_capture_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return up_capture_rs(returns, benchmark_rets, ann_factor)
    from vectorbt.returns.nb import up_capture_nb

    return up_capture_nb(returns, benchmark_rets, ann_factor)


def rolling_up_capture(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    benchmark_rets: tp.Array2d,
    ann_factor: float,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_up_capture_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_up_capture_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return rolling_up_capture_rs(returns, window, minp, benchmark_rets, ann_factor)
    from vectorbt.returns.nb import rolling_up_capture_nb

    return rolling_up_capture_nb(returns, window, minp, benchmark_rets, ann_factor)


def down_capture(
    returns: tp.Array2d,
    benchmark_rets: tp.Array2d,
    ann_factor: float,
    engine: tp.Optional[str] = None,
) -> tp.Array1d:
    """Engine-neutral `vectorbt.returns.nb.down_capture_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            array_compatible_with_rust(returns),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import down_capture_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return down_capture_rs(returns, benchmark_rets, ann_factor)
    from vectorbt.returns.nb import down_capture_nb

    return down_capture_nb(returns, benchmark_rets, ann_factor)


def rolling_down_capture(
    returns: tp.Array2d,
    window: int,
    minp: tp.Optional[int],
    benchmark_rets: tp.Array2d,
    ann_factor: float,
    engine: tp.Optional[str] = None,
) -> tp.Array2d:
    """Engine-neutral `vectorbt.returns.nb.rolling_down_capture_nb`."""
    eng = resolve_engine(
        engine,
        supports_rust=combine_rust_support(
            rolling_compatible_with_rust(returns, window, minp),
            array_compatible_with_rust(benchmark_rets),
            matching_shape_compatible_with_rust("benchmark_rets", returns, benchmark_rets),
            scalar_compatible_with_rust("ann_factor", ann_factor),
        ),
    )
    if eng == "rust":
        from vectorbt_rust.returns import rolling_down_capture_rs

        returns = prepare_array_for_rust(returns, dtype=np.float64)
        benchmark_rets = prepare_array_for_rust(benchmark_rets, dtype=np.float64)
        return rolling_down_capture_rs(returns, window, minp, benchmark_rets, ann_factor)
    from vectorbt.returns.nb import rolling_down_capture_nb

    return rolling_down_capture_nb(returns, window, minp, benchmark_rets, ann_factor)
