import numpy as np
import pandas as pd
from numba import njit, b1, i1, i8, f8
from numba.types import UniTuple
import plotly.graph_objects as go

from vectorbt.utils import *
from vectorbt.accessors import *
from vectorbt.widgets import FigureWidget
from vectorbt.timeseries import pct_change_nb, fillna_nb, expanding_max_nb, diff_nb

__all__ = ['Portfolio']

# ############# Numba functions ############# #


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
                adj_price = ts[i, col] * (1 + slippage[i, col])  # slippage applies on price
                req_cash = volume * adj_price
                req_cash /= (1 - commission[i, col])  # total cash required for this volume
                if req_cash <= run_cash:  # sufficient cash
                    run_shares += volume
                    run_cash -= req_cash
                else:  # not sufficient cash, volume will be less than requested
                    adj_cash = run_cash
                    adj_cash *= (1 - commission[i, col])  # commission in % applies on transaction volume
                    run_shares += adj_cash / adj_price
                    run_cash = 0
            elif volume < 0:
                # Sell volume
                adj_price = ts[i, col] * (1 - slippage[i, col])
                adj_shares = min(run_shares, abs(volume))
                adj_cash = adj_shares * adj_price
                adj_cash *= (1 - commission[i, col])
                run_shares -= adj_shares
                run_cash += adj_cash
            cash[i, col] = run_cash
            shares[i, col] = run_shares

    return cash, shares


@njit(UniTuple(f8[:, :], 2)(f8[:, :], f8, f8[:, :], f8[:, :], b1[:, :], b1[:, :], f8[:, :], b1), cache=True)
def portfolio_from_signals_np(ts, investment, slippage, commission, entries, exits, volume, accumulate):
    """Calculate portfolio value using signals."""
    return portfolio_np(ts, investment, slippage, commission, signals_order_func_np, entries, exits, volume, accumulate)


@njit(UniTuple(f8[:, :], 2)(f8[:, :], f8, f8[:, :], f8[:, :], f8[:, :], b1), cache=True)
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


nanmean_nb = njit(lambda x: np.nanmean(x))
nansum_nb = njit(lambda x: np.nansum(x))
win_mask_nb = njit(lambda x: x > 0)
loss_mask_nb = njit(lambda x: x < 0)


@njit(f8[:](f8[:, :]))
def sum_win_nb(position_profits):
    return apply_on_position_profits_nb(position_profits, nansum_nb, win_mask_nb)


@njit(f8[:](f8[:, :]))
def sum_loss_nb(position_profits):
    return np.abs(apply_on_position_profits_nb(position_profits, nansum_nb, loss_mask_nb))


@njit(f8[:](f8[:, :]))
def avg_win_nb(position_profits):
    return apply_on_position_profits_nb(position_profits, nanmean_nb, win_mask_nb)


@njit(f8[:](f8[:, :]))
def avg_loss_nb(position_profits):
    return np.abs(apply_on_position_profits_nb(position_profits, nanmean_nb, loss_mask_nb))

# ############# Custom accessors ############# #


def indexing_func(obj, loc_pandas_func):
    return obj.__class__(
        loc_pandas_func(obj.ts),
        loc_pandas_func(obj.cash),
        loc_pandas_func(obj.shares),
        obj.investment
    )


@add_indexing(indexing_func)
class Portfolio():

    def __init__(self, ts, cash, shares, investment):
        # Checks and preprocessing
        check_type(ts, (pd.Series, pd.DataFrame))
        check_type(cash, (pd.Series, pd.DataFrame))
        check_type(shares, (pd.Series, pd.DataFrame))

        ts.vbt.timeseries.validate()
        cash.vbt.timeseries.validate()
        shares.vbt.timeseries.validate()

        check_same_meta(ts, cash)
        check_same_meta(ts, shares)

        self.ts = ts
        self.cash = cash
        self.shares = shares
        self.investment = investment

    # ############# Indexing and magic methods ############# #

    def __add__(self, other):
        check_type(other, self.__class__)
        check_same(self.ts, other.ts)

        return self.__class__(
            self.ts,
            self.cash + other.cash,
            self.shares + other.shares,
            self.investment + other.investment
        )

    def __radd__(self, other):
        return Portfolio.__add__(self, other)

    def __sub__(self, other):
        check_type(other, self.__class__)
        check_same(self.ts, other.ts)

        return self.__class__(
            self.ts,
            self.cash - other.cash,
            self.shares - other.shares,
            self.investment - other.investment
        )

    def __rsub__(self, other):
        return Portfolio.__sub__(self, other)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, ts, entries, exits, volume=np.inf, accumulate=False, investment=1., slippage=0., commission=0., **kwargs):
        """Build portfolio based on entry and exit signals and the corresponding volume.

        Set volume to the number of shares to buy/sell.
        Set volume to np.inf to buy/sell everything.
        Set accumulate to False to avoid producing new orders if already in the market."""
        # Checks and preprocessing
        check_type(ts, (pd.Series, pd.DataFrame))
        check_type(entries, (pd.Series, pd.DataFrame))
        check_type(exits, (pd.Series, pd.DataFrame))

        ts.vbt.timeseries.validate()
        entries.vbt.signals.validate()
        exits.vbt.signals.validate()

        check_same_index(ts, entries)
        check_same_index(ts, exits)

        ts, entries, exits = broadcast(ts, entries, exits, **kwargs)

        volume = broadcast_to(volume, ts)
        slippage = broadcast_to(slippage, ts)
        commission = broadcast_to(commission, ts)

        volume = to_2d(np.asarray(volume).astype(np.float64))
        slippage = to_2d(np.asarray(slippage).astype(np.float64))
        commission = to_2d(np.asarray(commission).astype(np.float64))
        investment = float(investment)

        cash, shares = portfolio_from_signals_np(
            ts.vbt.to_2d_array(),
            investment,
            slippage,
            commission,
            entries.vbt.to_2d_array(),
            exits.vbt.to_2d_array(),
            volume,
            accumulate)

        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)

        return cls(ts, cash, shares, investment)

    @classmethod
    def from_orders(cls, ts, orders, is_target=False, investment=1., slippage=0., commission=0., **kwargs):
        """Build portfolio based on orders.

        Set an orders element to positive/negative number - a number of shares to buy/sell.
        Set is_target to True to specify the target amount of shares to hold."""
        check_type(ts, (pd.Series, pd.DataFrame))
        check_type(orders, (pd.Series, pd.DataFrame))

        ts.vbt.timeseries.validate()
        orders.vbt.timeseries.validate()

        check_same_index(ts, orders)

        ts, orders = broadcast(ts, orders, **kwargs)

        slippage = broadcast_to(slippage, ts)
        commission = broadcast_to(commission, ts)

        slippage = to_2d(np.asarray(slippage).astype(np.float64))
        commission = to_2d(np.asarray(commission).astype(np.float64))
        investment = float(investment)

        cash, shares = portfolio_from_orders_np(
            ts.vbt.to_2d_array(),
            investment,
            slippage,
            commission,
            orders.vbt.to_2d_array(),
            is_target)

        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)

        return cls(ts, cash, shares, investment)

    @classmethod
    def from_order_func(cls, ts, order_func_np, *args, investment=1., slippage=0., commission=0.):
        """Build portfolio based on order function."""
        check_type(ts, (pd.Series, pd.DataFrame))

        ts.vbt.timeseries.validate()

        slippage = broadcast_to(slippage, ts)
        commission = broadcast_to(commission, ts)

        slippage = to_2d(np.asarray(slippage).astype(np.float64))
        commission = to_2d(np.asarray(commission).astype(np.float64))
        investment = float(investment)

        cash, shares = portfolio_np(
            ts.vbt.to_2d_array(),
            investment,
            slippage,
            commission,
            order_func_np,
            *args)

        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)

        return cls(ts, cash, shares, investment)

    # ############# General properties ############# #

    @cached_property
    def equity(self):
        return self.ts.vbt.wrap_array(self.cash.values + self.shares.values * self.ts.values)

    @cached_property
    def equity_in_shares(self):
        return self.ts.vbt.wrap_array(self.equity.values / self.ts.values)

    @cached_property
    def returns(self):
        return self.ts.vbt.wrap_array(pct_change_nb(self.equity.vbt.to_2d_array()))

    @cached_property
    def drawdown(self):
        drawdown = 1 - self.equity.vbt.to_2d_array() / expanding_max_nb(self.equity.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(drawdown)

    @cached_property
    def trades(self):
        shares = self.shares.vbt.to_2d_array()
        trades = fillna_nb(diff_nb(shares), 0)
        trades[0, :] = shares[0, :]
        return self.ts.vbt.wrap_array(trades)

    @cached_property
    def position_profits(self):
        position_profits = position_profits_nb(self.trades.vbt.to_2d_array(), self.equity.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(position_profits)

    @cached_property
    def position_returns(self):
        position_returns = position_returns_nb(self.trades.vbt.to_2d_array(), self.equity.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(position_returns)

    @cached_property
    def win_mask(self):
        position_profits = self.position_profits.values.copy()
        position_profits[np.isnan(position_profits)] = 0 # avoid warnings
        win_mask = position_profits > 0
        return self.ts.vbt.wrap_array(win_mask)

    @cached_property
    def loss_mask(self):
        position_profits = self.position_profits.values.copy()
        position_profits[np.isnan(position_profits)] = 0
        loss_mask = position_profits < 0
        return self.ts.vbt.wrap_array(loss_mask)

    @cached_property
    def position_mask(self):
        position_mask = ~np.isnan(self.position_profits.values)
        return self.ts.vbt.wrap_array(position_mask)

    # ############# Trade P/L properties ############# #

    @cached_property
    def sum_win(self):
        """Sum of wins."""
        sum_win = sum_win_nb(self.position_profits.vbt.to_2d_array())
        if isinstance(self.position_profits, pd.DataFrame):
            return pd.Series(sum_win, index=self.position_profits.columns)
        else:
            return sum_win[0]

    @cached_property
    def sum_loss(self):
        """Sum of losses (always positive)."""
        sum_loss = sum_loss_nb(self.position_profits.vbt.to_2d_array())
        if isinstance(self.position_profits, pd.DataFrame):
            return pd.Series(sum_loss, index=self.position_profits.columns)
        else:
            return sum_loss[0]

    @cached_property
    def avg_win(self):
        """Average win."""
        avg_win = avg_win_nb(self.position_profits.vbt.to_2d_array())
        if isinstance(self.position_profits, pd.DataFrame):
            return pd.Series(avg_win, index=self.position_profits.columns)
        else:
            return avg_win[0]

    @cached_property
    def avg_loss(self):
        """Average loss (always positive)."""
        avg_loss = avg_loss_nb(self.position_profits.vbt.to_2d_array())
        if isinstance(self.position_profits, pd.DataFrame):
            return pd.Series(avg_loss, index=self.position_profits.columns)
        else:
            return avg_loss[0]

    @cached_property
    def win_prob(self):
        """Fraction of wins."""
        return self.win_mask.sum(axis=0) / self.position_mask.sum(axis=0)

    @cached_property
    def loss_prob(self):
        """Fraction of losses."""
        return self.loss_mask.sum(axis=0) / self.position_mask.sum(axis=0)

    # ############# Performance properties ############# #

    @cached_property
    def profit_factor(self):
        return self.sum_win / self.sum_loss

    @cached_property
    def appt(self):
        """Average profitability per trade (APPT)

        For every trade you place, you are likely to win/lose this amount.
        What matters is that your APPT comes up positive."""
        return self.win_prob * self.avg_win - self.loss_prob * self.avg_loss

    @cached_property
    def total_net_profit(self):
        return self.equity.iloc[-1] - self.investment

    @cached_property
    def total_return(self):
        return self.total_net_profit / self.investment

    @cached_property
    def mdd(self):
        """A maximum drawdown (MDD) is the maximum observed loss from a peak 
        to a trough of a portfolio, before a new peak is attained."""
        return self.drawdown.max(axis=0)

    # ############# Plotting ############# #

    def plot_trades(self,
                    buy_scatter_kwargs={},
                    sell_scatter_kwargs={},
                    fig=None,
                    **ts_kwargs):
        # Checks and preprocessing
        check_type(self.ts, pd.Series)
        check_type(self.trades, pd.Series)
        sell_mask = self.trades < 0
        buy_mask = self.trades > 0

        # Plot TimeSeries
        fig = self.ts.vbt.timeseries.plot(fig=fig, **ts_kwargs)

        # Plot markets
        buy_scatter = go.Scatter(
            x=self.trades.index[buy_mask],
            y=self.ts[buy_mask],
            customdata=self.trades[buy_mask],
            hovertemplate='(%{x}, %{y})<br>%{customdata:.6g}',
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                color='limegreen',
                size=10
            ),
            name='Buy'
        )
        buy_scatter.update(**buy_scatter_kwargs)
        fig.add_trace(buy_scatter)
        sell_scatter = go.Scatter(
            x=self.trades.index[sell_mask],
            y=self.ts[sell_mask],
            customdata=self.trades[sell_mask],
            hovertemplate='(%{x}, %{y})<br>%{customdata:.6g}',
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                color='orangered',
                size=10
            ),
            name='Sell'
        )
        sell_scatter.update(**sell_scatter_kwargs)
        fig.add_trace(sell_scatter)

        return fig

    def plot_position_profits(self,
                              profit_scatter_kwargs={},
                              loss_scatter_kwargs={},
                              fig=None,
                              **layout_kwargs):
        # Checks and preprocessing
        check_type(self.position_profits, pd.Series)
        profits = self.position_profits.copy()
        profits[self.position_profits <= 0] = np.nan
        losses = self.position_profits.copy()
        losses[self.position_profits >= 0] = np.nan

        # Set up figure
        if fig is None:
            fig = FigureWidget()
            fig.update_layout(showlegend=True)
            fig.update_layout(**layout_kwargs)

        # Plot markets
        profit_scatter = go.Scatter(
            x=self.position_profits.index,
            y=profits,
            mode='markers',
            marker=dict(
                symbol='circle',
                color='green',
                size=10
            ),
            name='Profit'
        )
        profit_scatter.update(**profit_scatter_kwargs)
        fig.add_trace(profit_scatter)
        loss_scatter = go.Scatter(
            x=self.position_profits.index,
            y=losses,
            mode='markers',
            marker=dict(
                symbol='circle',
                color='red',
                size=10
            ),
            name='Loss'
        )
        loss_scatter.update(**loss_scatter_kwargs)
        fig.add_trace(loss_scatter)

        # Set up axes
        maxval = np.nanmax(np.abs(self.position_profits.values))
        space = 0.1 * 2 * maxval
        fig.update_layout(
            yaxis=dict(
                range=[-(maxval+space), maxval+space]
            ),
            shapes=[dict(
                type="line",
                xref="paper",
                yref='y',
                x0=0, x1=1, y0=0, y1=0,
                line=dict(
                    color="grey",
                    width=2,
                    dash="dot",
                ))]
        )

        return fig
