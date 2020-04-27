import numpy as np
from numba import njit, b1, i1, i8, f8
from numba.types import UniTuple


@njit(f8(i8, i8, f8, f8, b1[:, :], b1[:, :], f8[:, :], b1), cache=True)
def signals_order_func_np(i, col, run_cash, run_shares, entries, exits, volume, accumulate):
    """Order function to buy/sell based on signals."""
    if run_shares > 0:
        if entries[i, col] and not exits[i, col]:
            if accumulate:
                return volume[i, col]
        elif not entries[i, col] and exits[i, col]:
            return -volume[i, col]
    else:
        if entries[i, col] and not exits[i, col]:
            return volume[i, col]
        elif not entries[i, col] and exits[i, col]:
            if accumulate:
                return -volume[i, col]
    return 0.


@njit(f8(i8, i8, f8, f8, f8[:, :], b1), cache=True)
def orders_order_func_np(i, col, run_cash, run_shares, orders, is_target):
    """Buy/sell the amount of shares specified by orders."""
    if is_target:
        return orders[i, col] - run_shares
    else:
        return orders[i, col]


@njit
def portfolio_np(ts, investment, slippage, commission, order_func_np, *args):
    """Calculate portfolio value in cash and shares."""
    cash = np.empty_like(ts)
    shares = np.empty_like(ts)

    for col in range(ts.shape[1]):
        run_cash = investment
        run_shares = 0
        for i in range(ts.shape[0]):
            volume = order_func_np(i, col, run_cash, run_shares, *args)  # the amount of shares to buy/sell
            if volume > 0:
                # Buy volume
                adj_price = ts[i, col] * (1 + slippage)  # slippage applies on price
                req_cash = volume * adj_price
                req_cash /= (1 - commission)  # total cash required for this volume
                if req_cash <= run_cash:  # sufficient cash
                    run_shares += volume
                    run_cash -= req_cash
                else:  # not sufficient cash, volume will be less than requested
                    adj_cash = run_cash
                    adj_cash *= (1 - commission)  # commission in % applies on transaction volume
                    run_shares += adj_cash / adj_price
                    run_cash = 0
            elif volume < 0:
                # Sell volume
                adj_price = ts[i, col] * (1 - slippage)
                adj_shares = min(run_shares, abs(volume))
                adj_cash = adj_shares * adj_price
                adj_cash *= (1 - commission)
                run_shares -= adj_shares
                run_cash += adj_cash
            cash[i, col] = run_cash
            shares[i, col] = run_shares

    return cash, shares


@njit(UniTuple(f8[:, :], 2)(f8[:, :], f8, f8, f8, b1[:, :], b1[:, :], f8[:, :], b1), cache=True)
def portfolio_from_signals_np(ts, investment, slippage, commission, entries, exits, volume, accumulate):
    """Calculate portfolio value using signals."""
    return portfolio_np(ts, investment, slippage, commission, signals_order_func_np, entries, exits, volume, accumulate)


@njit(UniTuple(f8[:, :], 2)(f8[:, :], f8, f8, f8, f8[:, :], b1), cache=True)
def portfolio_from_orders_np(ts, investment, slippage, commission, orders, is_target):
    """Calculate portfolio value using orders."""
    return portfolio_np(ts, investment, slippage, commission, orders_order_func_np, orders, is_target)


@njit(b1(f8[:]), cache=True)
def detect_order_accumulation_1d_nb(trades):
    """Detect accumulation of orders, that is, position is being increased/decreased gradually.

    When it happens, it's not easy to calculate P/L of a position anymore."""
    entry_i = -1
    position = False
    for i in range(trades.shape[0]):
        if trades[i] > 0:
            if position:
                return True
            entry_i = i
            position = True
        elif trades[i] < 0:
            if not position:
                return True
            if trades[entry_i] != abs(trades[i]):
                return True
            position = False
    return False


@njit(b1[:](f8[:, :]), cache=True)
def detect_order_accumulation_nb(trades):
    """Detect accumulation of orders, that is, position is being increased/decreased gradually.

    When it happens, it's not easy to calculate P/L of a position anymore."""
    a = np.full(trades.shape[1], False, dtype=b1)
    for col in range(trades.shape[1]):
        a[col] = detect_order_accumulation_1d_nb(trades[:, col])
    return a


@njit
def apply_on_positions(trades, apply_func, *args):
    """Apply a function on each position."""
    if detect_order_accumulation_nb(trades).any():
        raise ValueError("Order accumulation detected. Cannot calculate performance per position.")
    out = np.full_like(trades, np.nan)

    for col in range(trades.shape[1]):
        entry_i = -1
        position = False
        for i in range(trades.shape[0]):
            if position and trades[i, col] < 0:
                out[i, col] = apply_func(entry_i, i, col, trades, *args)
                position = False
            elif not position and trades[i, col] > 0:
                entry_i = i
                position = True
            if position and i == trades.shape[0] - 1:  # unrealized
                out[i, col] = apply_func(entry_i, i, col, trades, *args)
    return out


_profits_nb = njit(lambda entry_i, exit_i, col, trades, equity: equity[exit_i, col] - equity[entry_i, col])
_returns_nb = njit(lambda entry_i, exit_i, col, trades, equity: equity[exit_i, col] / equity[entry_i, col] - 1)


@njit(f8[:, :](f8[:, :], f8[:, :]), cache=True)
def position_profits_nb(trades, equity):
    """Calculate P/L per position."""
    return apply_on_positions(trades, _profits_nb, equity)


@njit(f8[:, :](f8[:, :], f8[:, :]), cache=True)
def position_returns_nb(trades, equity):
    """Calculate returns per trade."""
    return apply_on_positions(trades, _returns_nb, equity)


@njit
def apply_on_position_profits_nb(position_profits, apply_func, mask_func):
    applied = np.zeros(position_profits.shape[1])

    for col in range(position_profits.shape[1]):
        mask = mask_func(position_profits[:, col])
        if mask.any():
            masked = position_profits[:, col][mask]
            applied[col] = apply_func(masked)
    return applied


_nanmean_nb = njit(lambda x: np.nanmean(x))
_nansum_nb = njit(lambda x: np.nansum(x))
_win_mask_nb = njit(lambda x: x > 0)
_loss_mask_nb = njit(lambda x: x < 0)


@njit(f8[:](f8[:, :]))
def sum_win_nb(position_profits):
    return apply_on_position_profits_nb(position_profits, _nansum_nb, _win_mask_nb)


@njit(f8[:](f8[:, :]))
def sum_loss_nb(position_profits):
    return np.abs(apply_on_position_profits_nb(position_profits, _nansum_nb, _loss_mask_nb))


@njit(f8[:](f8[:, :]))
def avg_win_nb(position_profits):
    return apply_on_position_profits_nb(position_profits, _nanmean_nb, _win_mask_nb)


@njit(f8[:](f8[:, :]))
def avg_loss_nb(position_profits):
    return np.abs(apply_on_position_profits_nb(position_profits, _nanmean_nb, _loss_mask_nb))
