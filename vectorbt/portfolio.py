import numpy as np
import pandas as pd
from numba import njit, i1, f8
from vectorbt.timeseries import pct_change_nb, fillna_nb, rolling_max_nb
from vectorbt.utils.decorators import *
from vectorbt.timeseries import TimeSeries
from vectorbt.positions import Positions

# ############# Numba functions ############# #


@njit(f8[:, :](f8[:, :], i1[:, :], f8, f8[:, :], f8[:, :]))
def equity_nb(ts, positions, investment, fees, slippage):
    equity = np.zeros_like(ts)
    for j in range(ts.shape[1]):
        j_equity = investment
        bought = False
        for i in range(ts.shape[0]):
            if bought and i > 0:
                i_returns = ts[i, j] / ts[i-1, j] - 1
                j_equity *= 1 + i_returns
            if positions[i, j] != 0:
                bought = ~bought
                j_equity *= 1 - fees[i, j]
                j_equity *= 1 - slippage[i, j]
            equity[i, j] = j_equity
    return equity


@njit(f8[:, :](f8[:, :], i1[:, :]))
def trade_profits_nb(equity, positions):
    trade_profits = np.zeros_like(equity)
    for j in range(equity.shape[1]):
        bought_i = -1
        bought = False
        for i in range(equity.shape[0]):
            if bought and positions[i, j] == -1:
                trade_profits[i, j] = equity[i, j] - equity[bought_i, j]
                bought = False
            elif ~bought and positions[i, j] == 1:
                bought_i = i
                bought = True
    return trade_profits


@njit(f8[:, :](f8[:, :], i1[:, :]))
def trade_returns_nb(equity, positions):
    trade_returns = np.zeros_like(equity)
    for j in range(equity.shape[1]):
        bought_i = -1
        bought = False
        for i in range(equity.shape[0]):
            if bought and positions[i, j] == -1:
                trade_returns[i, j] = equity[i, j] / equity[bought_i, j] - 1
                bought = False
            elif ~bought and positions[i, j] == 1:
                bought_i = i
                bought = True
    return trade_returns

# ############# Main class ############# #


class Portfolio():
    """Propagate investment through positions to generate equity and returns."""

    @has_type('ts', TimeSeries)
    @has_type('positions', Positions)
    @broadcast_both('ts', 'positions')
    @broadcast_to('fees', 'ts')
    @broadcast_to('slippage', 'ts')
    @to_dtype('fees', np.float64)
    @to_dtype('slippage', np.float64)
    def __init__(self, ts, positions, investment=1., fees=0., slippage=0.):
        self.ts = ts
        self.positions = positions
        self.investment = investment
        self.fees = fees
        self.slippage = slippage

    # ############# TimeSeries objects ############# #

    @property
    @cache_property
    def equity(self):
        return TimeSeries(equity_nb(self.ts, self.positions, self.investment, self.fees, self.slippage))

    @property
    @cache_property
    def equity_in_shares(self):
        return TimeSeries(self.equity / self.ts)

    @property
    @cache_property
    def returns(self):
        return TimeSeries(fillna_nb(pct_change_nb(self.equity), 0))

    @property
    @cache_property
    def trade_profits(self):
        return TimeSeries(trade_profits_nb(self.equity, self.positions))

    @property
    @cache_property
    def trade_returns(self):
        return TimeSeries(trade_returns_nb(self.equity, self.positions))

    # ############# Performance metrics ############# #

    def reduce_win(self, func):
        """Perform reducing operation on wins."""
        trade_profits = self.trade_profits.copy()
        trade_profits[trade_profits <= 0] = np.nan
        return np.asarray(func(trade_profits))

    def reduce_loss(self, func):
        """Perform reducing operation on losses."""
        trade_profits = self.trade_profits.copy()
        trade_profits[trade_profits >= 0] = np.nan
        return np.asarray(np.abs(func(trade_profits)))

    @property
    def sum_win(self):
        return self.reduce_win(lambda x: np.nansum(x, axis=0))

    @property
    def sum_loss(self):
        return self.reduce_loss(lambda x: np.nansum(x, axis=0))

    @property
    def avg_win(self):
        return self.reduce_win(lambda x: np.nanmean(x, axis=0))

    @property
    def avg_loss(self):
        return self.reduce_loss(lambda x: np.nanmean(x, axis=0))

    @property
    def win_prob(self):
        """Profitability = % of total trades that resulted in profits."""
        trade_profits = self.trade_profits.copy()
        num_pos = np.sum(trade_profits > 0, axis=0)
        num_all = np.sum(trade_profits != 0, axis=0)
        return np.asarray(num_pos / num_all)

    @property
    def loss_prob(self):
        return 1 - self.win_prob

    @property
    def appt(self):
        """Average profitability per trade (APPT)

        For every trade you place, you are likely to win/lose this amount.
        What matters is that your APPT comes up positive."""
        return (self.win_prob * self.avg_win) - (self.loss_prob * self.avg_loss)

    @property
    def total_net_profit(self):
        return self.sum_win - self.sum_loss

    @property
    def profit_factor(self):
        return self.sum_win / self.sum_loss

    @property
    def mdd(self):
        """A maximum drawdown (MDD) is the maximum observed loss from a peak 
        to a trough of a portfolio, before a new peak is attained."""
        return np.max(1 - self.equity / rolling_max_nb(self.equity, None), axis=0)
