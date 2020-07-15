"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function.
    Data is processed along index (axis 0).

    All functions passed as argument must be Numba-compiled."""

import numpy as np
from numba import njit

from vectorbt import tseries

# ############# Financial risk and performance metrics ############# #


@njit(cache=True)
def total_return_apply_nb(col, idxs, returns):
    """Calculate total return from returns."""
    return tseries.nb.product_1d_nb(returns + 1) - 1

# Functions from empyrical but Numba-compiled


@njit(cache=True)
def cum_returns_1d_nb(returns, start_value=0.):
    """See `empyrical.cum_returns`."""
    nanmask = np.isnan(returns)
    if nanmask.any():
        returns = returns.copy()
        returns[nanmask] = 0
    result = tseries.nb.cumprod_1d_nb(returns + 1.)
    if start_value == 0.:
        return result - 1.
    return result * start_value


@njit(cache=True)
def cum_returns_nb(returns, start_value_arr):
    """2-dim version of `cum_returns_1d_nb`.

    `start_value_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty_like(returns, dtype=np.float_)
    for col in range(returns.shape[1]):
        result[:, col] = cum_returns_1d_nb(returns[:, col], start_value=start_value_arr[col])
    return result


@njit(cache=True)
def cum_returns_final_1d_nb(returns, start_value=0.):
    """See `empyrical.cum_returns_final`."""
    result = tseries.nb.product_1d_nb(returns + 1.)
    if start_value == 0.:
        return result - 1.
    return result * start_value


@njit(cache=True)
def cum_returns_final_nb(returns, start_value_arr):
    """2-dim version of `cum_returns_final_1d_nb`.

    `start_value_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = cum_returns_final_1d_nb(returns[:, col], start_value=start_value_arr[col])
    return result


@njit(cache=True)
def annualized_return_1d_nb(returns, ann_factor):
    """See `empyrical.annual_return`."""
    end_value = cum_returns_final_1d_nb(returns, start_value=1.)
    return end_value ** (ann_factor / returns.shape[0]) - 1


@njit(cache=True)
def annualized_return_nb(returns, ann_factor):
    """2-dim version of `annualized_return_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = annualized_return_1d_nb(returns[:, col], ann_factor)
    return result


@njit(cache=True)
def annualized_volatility_1d_nb(returns, ann_factor, levy_alpha=2.0):
    """See `empyrical.annual_volatility`."""
    if returns.shape[0] < 2:
        return np.nan

    return tseries.nb.nanstd_1d_nb(returns, ddof=1) * ann_factor ** (1.0 / levy_alpha)


@njit(cache=True)
def annualized_volatility_nb(returns, ann_factor, levy_alpha_arr):
    """2-dim version of `annualized_volatility_1d_nb`.

    `levy_alpha_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = annualized_volatility_1d_nb(returns[:, col], ann_factor, levy_alpha=levy_alpha_arr[col])
    return result


@njit(cache=True)
def drawdown_1d_nb(returns):
    """Drawdown of cumulative returns."""
    cum_returns = cum_returns_1d_nb(returns, start_value=100.)
    max_returns = tseries.nb.expanding_max_1d_nb(cum_returns, minp=1)
    return cum_returns / max_returns - 1


@njit(cache=True)
def drawdown_nb(returns):
    """2-dim version of `drawdown_1d_nb`."""
    result = np.empty_like(returns, dtype=np.float_)
    for col in range(returns.shape[1]):
        result[:, col] = drawdown_1d_nb(returns[:, col])
    return result


@njit(cache=True)
def max_drawdown_1d_nb(returns):
    """See `empyrical.max_drawdown`."""
    return np.min(drawdown_1d_nb(returns))


@njit(cache=True)
def max_drawdown_nb(returns):
    """2-dim version of `max_drawdown_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = max_drawdown_1d_nb(returns[:, col])
    return result


@njit(cache=True)
def calmar_ratio_1d_nb(returns, ann_factor):
    """See `empyrical.calmar_ratio`."""
    max_drawdown = max_drawdown_1d_nb(returns)
    if max_drawdown == 0.:
        return np.nan
    annualized_return = annualized_return_1d_nb(returns, ann_factor)
    return annualized_return / np.abs(max_drawdown)


@njit(cache=True)
def calmar_ratio_nb(returns, ann_factor):
    """2-dim version of `calmar_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = calmar_ratio_1d_nb(returns[:, col], ann_factor)
    return result


@njit(cache=True)
def omega_ratio_1d_nb(returns, ann_factor, risk_free=0., required_return=0.):
    """See `empyrical.omega_ratio`."""
    if ann_factor == 1:
        return_threshold = required_return
    elif ann_factor <= -1:
        return np.nan
    else:
        return_threshold = (1 + required_return) ** (1. / ann_factor) - 1
    returns_less_thresh = returns - risk_free - return_threshold
    numer = np.sum(returns_less_thresh[returns_less_thresh > 0.0])
    denom = -1.0 * np.sum(returns_less_thresh[returns_less_thresh < 0.0])
    if denom == 0.:
        return np.inf
    return numer / denom


@njit(cache=True)
def omega_ratio_nb(returns, ann_factor, risk_free_arr, required_return_arr):
    """2-dim version of `omega_ratio_1d_nb`.

    `risk_free_arr` and `required_return_arr` should be arrays of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = omega_ratio_1d_nb(
            returns[:, col], ann_factor, risk_free=risk_free_arr[col], required_return=required_return_arr[col])
    return result


@njit(cache=True)
def sharpe_ratio_1d_nb(returns, ann_factor, risk_free=0.):
    """See `empyrical.sharpe_ratio`."""
    if returns.shape[0] < 2:
        return np.nan

    returns_risk_adj = returns - risk_free
    mean = np.nanmean(returns_risk_adj)
    std = tseries.nb.nanstd_1d_nb(returns_risk_adj, ddof=1)
    if std == 0.:
        return np.inf
    return mean / std * np.sqrt(ann_factor)


@njit(cache=True)
def sharpe_ratio_nb(returns, ann_factor, risk_free_arr):
    """2-dim version of `sharpe_ratio_1d_nb`.

    `risk_free_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = sharpe_ratio_1d_nb(returns[:, col], ann_factor, risk_free=risk_free_arr[col])
    return result


@njit(cache=True)
def downside_risk_1d_nb(returns, ann_factor, required_return=0.):
    """See `empyrical.downside_risk`."""
    adj_returns = returns - required_return
    adj_returns[adj_returns > 0] = 0
    return np.sqrt(np.nanmean(adj_returns ** 2)) * np.sqrt(ann_factor)


@njit(cache=True)
def downside_risk_nb(returns, ann_factor, required_return_arr):
    """2-dim version of `downside_risk_1d_nb`.

    `required_return_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = downside_risk_1d_nb(returns[:, col], ann_factor, required_return=required_return_arr[col])
    return result


@njit(cache=True)
def sortino_ratio_1d_nb(returns, ann_factor, required_return=0.):
    """See `empyrical.sortino_ratio`."""
    if returns.shape[0] < 2:
        return np.nan

    adj_returns = returns - required_return
    average_annualized_return = np.nanmean(adj_returns) * ann_factor
    downside_risk = downside_risk_1d_nb(returns, ann_factor, required_return=required_return)
    if downside_risk == 0.:
        return np.inf
    return average_annualized_return / downside_risk


@njit(cache=True)
def sortino_ratio_nb(returns, ann_factor, required_return_arr):
    """2-dim version of `sortino_ratio_1d_nb`.

    `required_return_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = sortino_ratio_1d_nb(returns[:, col], ann_factor, required_return=required_return_arr[col])
    return result


@njit(cache=True)
def information_ratio_1d_nb(returns, factor_returns):
    """See `empyrical.excess_sharpe`."""
    if returns.shape[0] < 2:
        return np.nan

    active_return = returns - factor_returns
    return np.nanmean(active_return) / tseries.nb.nanstd_1d_nb(active_return, ddof=1)


@njit(cache=True)
def information_ratio_nb(returns, factor_returns):
    """2-dim version of `information_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = information_ratio_1d_nb(returns[:, col], factor_returns[:, col])
    return result


@njit(cache=True)
def beta_1d_nb(returns, factor_returns):
    """See `empyrical.beta`."""
    if factor_returns.shape[0] < 2:
        return np.nan

    independent = np.where(
        np.isnan(returns),
        np.nan,
        factor_returns,
    )
    ind_residual = independent - np.nanmean(independent)
    covariances = np.nanmean(ind_residual * returns)
    ind_residual = ind_residual ** 2
    ind_variances = np.nanmean(ind_residual)
    if ind_variances < 1.0e-30:
        ind_variances = np.nan
    if ind_variances == 0.:
        return np.inf
    return covariances / ind_variances


@njit(cache=True)
def beta_nb(returns, factor_returns):
    """2-dim version of `beta_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = beta_1d_nb(returns[:, col], factor_returns[:, col])
    return result


@njit(cache=True)
def alpha_1d_nb(returns, factor_returns, ann_factor, risk_free=0.):
    """See `empyrical.alpha`."""
    if returns.shape[0] < 2:
        return np.nan

    adj_returns = returns - risk_free
    adj_factor_returns = factor_returns - risk_free
    beta = beta_1d_nb(returns, factor_returns)
    alpha_series = adj_returns - (beta * adj_factor_returns)
    return (np.nanmean(alpha_series) + 1) ** ann_factor - 1


@njit(cache=True)
def alpha_nb(returns, factor_returns, ann_factor, risk_free_arr):
    """2-dim version of `alpha_1d_nb`.

    `risk_free_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = alpha_1d_nb(returns[:, col], factor_returns[:, col], ann_factor, risk_free=risk_free_arr[col])
    return result


@njit(cache=True)
def tail_ratio_1d_nb(returns):
    """See `empyrical.tail_ratio`."""
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan
    perc_95 = np.abs(np.percentile(returns, 95))
    perc_5 = np.abs(np.percentile(returns, 5))
    if perc_5 == 0.:
        return np.inf
    return perc_95 / perc_5


@njit(cache=True)
def tail_ratio_nb(returns):
    """2-dim version of `tail_ratio_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = tail_ratio_1d_nb(returns[:, col])
    return result


@njit(cache=True)
def value_at_risk_1d_nb(returns, cutoff=0.05):
    """See `empyrical.value_at_risk`."""
    returns = returns[~np.isnan(returns)]
    if len(returns) < 1:
        return np.nan
    return np.percentile(returns, 100 * cutoff)


@njit(cache=True)
def value_at_risk_nb(returns, cutoff_arr):
    """2-dim version of `value_at_risk_1d_nb`.

    `cutoff_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = value_at_risk_1d_nb(returns[:, col], cutoff=cutoff_arr[col])
    return result


@njit(cache=True)
def conditional_value_at_risk_1d_nb(returns, cutoff=0.05):
    """See `empyrical.conditional_value_at_risk`."""
    cutoff_index = int((len(returns) - 1) * cutoff)
    return np.mean(np.partition(returns, cutoff_index)[:cutoff_index + 1])


@njit(cache=True)
def conditional_value_at_risk_nb(returns, cutoff_arr):
    """2-dim version of `conditional_value_at_risk_1d_nb`.

    `cutoff_arr` should be an array of shape `returns.shape[1]`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = conditional_value_at_risk_1d_nb(returns[:, col], cutoff=cutoff_arr[col])
    return result


@njit(cache=True)
def capture_1d_nb(returns, factor_returns, ann_factor):
    """See `empyrical.capture`."""
    annualized_return1 = annualized_return_1d_nb(returns, ann_factor)
    annualized_return2 = annualized_return_1d_nb(factor_returns, ann_factor)
    if annualized_return2 == 0.:
        return np.inf
    return annualized_return1 / annualized_return2


@njit(cache=True)
def capture_nb(returns, factor_returns, ann_factor):
    """2-dim version of `capture_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = capture_1d_nb(returns[:, col], factor_returns[:, col], ann_factor)
    return result


@njit(cache=True)
def up_capture_1d_nb(returns, factor_returns, ann_factor):
    """See `empyrical.up_capture`."""
    returns = returns[factor_returns > 0]
    factor_returns = factor_returns[factor_returns > 0]
    if returns.shape[0] < 1:
        return np.nan
    annualized_return1 = annualized_return_1d_nb(returns, ann_factor)
    annualized_return2 = annualized_return_1d_nb(factor_returns, ann_factor)
    if annualized_return2 == 0.:
        return np.inf
    return annualized_return1 / annualized_return2


@njit(cache=True)
def up_capture_nb(returns, factor_returns, ann_factor):
    """2-dim version of `up_capture_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = up_capture_1d_nb(returns[:, col], factor_returns[:, col], ann_factor)
    return result


@njit(cache=True)
def down_capture_1d_nb(returns, factor_returns, ann_factor):
    """See `empyrical.down_capture`."""
    returns = returns[factor_returns < 0]
    factor_returns = factor_returns[factor_returns < 0]
    if returns.shape[0] < 1:
        return np.nan
    annualized_return1 = annualized_return_1d_nb(returns, ann_factor)
    annualized_return2 = annualized_return_1d_nb(factor_returns, ann_factor)
    if annualized_return2 == 0.:
        return np.inf
    return annualized_return1 / annualized_return2


@njit(cache=True)
def down_capture_nb(returns, factor_returns, ann_factor):
    """2-dim version of `down_capture_1d_nb`."""
    result = np.empty(returns.shape[1], dtype=np.float_)
    for col in range(returns.shape[1]):
        result[col] = down_capture_1d_nb(returns[:, col], factor_returns[:, col], ann_factor)
    return result
