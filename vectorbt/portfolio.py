import numpy as np
import pandas as pd
from numba import njit, i1, f8
from vectorbt.timeseries import pct_change_nb, fillna_nb, expanding_max_nb
from vectorbt.decorators import *
from vectorbt.timeseries import TimeSeries
from vectorbt.positions import Positions

# ############# Numba functions ############# #


@njit(f8[:, :](f8[:, :], i1[:, :], f8, f8[:, :], f8[:, :]), cache=True)
def equity_nb(ts, positions, investment, fees, slippage):
    equity = np.empty_like(ts)
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


@njit(f8[:, :](f8[:, :], i1[:, :]), cache=True)
def trade_profits_nb(equity, positions):
    trade_profits = np.full_like(equity, np.nan)
    for j in range(equity.shape[1]):
        bought_i = -1
        bought = False
        for i in range(equity.shape[0]):
            if bought and (positions[i, j] == -1 or i == equity.shape[0] - 1):
                trade_profits[i, j] = equity[i, j] - equity[bought_i, j]
                bought = False
            elif ~bought and positions[i, j] == 1:
                bought_i = i
                bought = True
    return trade_profits


@njit(f8[:, :](f8[:, :], i1[:, :]), cache=True)
def trade_returns_nb(equity, positions):
    trade_returns = np.full_like(equity, np.nan)
    for j in range(equity.shape[1]):
        bought_i = -1
        bought = False
        for i in range(equity.shape[0]):
            if bought and (positions[i, j] == -1 or i == equity.shape[0] - 1):
                trade_returns[i, j] = equity[i, j] / equity[bought_i, j] - 1
                bought = False
            elif ~bought and positions[i, j] == 1:
                bought_i = i
                bought = True
    return trade_returns

# ############# Main class ############# #


class Portfolio():
    """Propagate investment through positions to generate equity and returns."""

    @broadcast('ts', 'positions')
    @broadcast_to('fees', 'ts')
    @broadcast_to('slippage', 'ts')
    @to_dtype('fees', np.float64)
    @to_dtype('slippage', np.float64)
    @has_type('ts', TimeSeries)
    @has_type('positions', Positions)
    def __init__(self, ts, positions, investment=1., fees=0., slippage=0.):
        self.ts = ts
        self.positions = positions
        self.investment = investment
        self.fees = fees
        self.slippage = slippage

    # ############# TimeSeries objects ############# #

    @property
    @cache_property
    @return_type(TimeSeries)
    def equity(self):
        return equity_nb(self.ts, self.positions, self.investment, self.fees, self.slippage)

    @property
    @cache_property
    @return_type(TimeSeries)
    def equity_in_shares(self):
        return self.equity / self.ts

    @property
    @cache_property
    @return_type(TimeSeries)
    def returns(self):
        return fillna_nb(pct_change_nb(self.equity), 0)

    @property
    @cache_property
    @return_type(TimeSeries)
    def trade_profits(self):
        return trade_profits_nb(self.equity, self.positions)

    @property
    @cache_property
    @return_type(TimeSeries)
    def trade_returns(self):
        return trade_returns_nb(self.equity, self.positions)

    # ############# Performance metrics ############# #

    @property
    def key_metrics_df(self):
        """Build dataframe with key metrics."""
        df = pd.DataFrame(columns=range(self.ts.shape[1]))
        props = ['num_trades', 'total_net_profit', 'avg_win', 'avg_loss', 'win_prob', 'loss_prob', 'appt', 'mdd']
        for prop in props:
            df.loc[prop, :] = getattr(self, prop)
        return df

    def __repr__(self):
        return self.key_metrics_df.__repr__()

    __str__ = __repr__

    @property
    @return_type(np.array)
    def num_trades(self):
        return np.sum(self.positions == 1, axis=0)

    @return_type(np.array)
    def reduce_win(self, func):
        """Perform reducing operation on wins."""
        trade_profits = self.trade_profits.copy()
        trade_profits[trade_profits <= 0] = np.nan
        return func(trade_profits)

    @return_type(np.array)
    def reduce_loss(self, func):
        """Perform reducing operation on losses."""
        trade_profits = self.trade_profits.copy()
        trade_profits[trade_profits >= 0] = np.nan
        return np.abs(func(trade_profits))

    @property
    @return_type(np.array)
    def sum_win(self):
        sum_win = self.reduce_win(lambda x: np.nansum(x, axis=0))
        sum_win[np.isnan(sum_win)] = 0
        return sum_win

    @property
    @return_type(np.array)
    def sum_loss(self):
        sum_loss = self.reduce_loss(lambda x: np.nansum(x, axis=0))
        sum_loss[np.isnan(sum_loss)] = 0
        return sum_loss

    @property
    @return_type(np.array)
    def avg_win(self):
        avg_win = self.reduce_win(lambda x: np.nanmean(x, axis=0))
        avg_win[np.isnan(avg_win)] = 0
        return avg_win

    @property
    @return_type(np.array)
    def avg_loss(self):
        avg_loss = self.reduce_loss(lambda x: np.nanmean(x, axis=0))
        avg_loss[np.isnan(avg_loss)] = 0
        return avg_loss

    @property
    @return_type(np.array)
    def win_prob(self):
        """Profitability = % of total trades that resulted in profits."""
        return np.sum(self.trade_profits > 0, axis=0) / self.num_trades

    @property
    @return_type(np.array)
    def loss_prob(self):
        return np.sum(self.trade_profits < 0, axis=0) / self.num_trades

    @property
    @return_type(np.array)
    def profit_factor(self):
        return self.sum_win / self.sum_loss

    @property
    @return_type(np.array)
    def appt(self):
        """Average profitability per trade (APPT)

        For every trade you place, you are likely to win/lose this amount.
        What matters is that your APPT comes up positive."""
        return self.win_prob * self.avg_win - self.loss_prob * self.avg_loss

    @property
    @return_type(np.array)
    def total_net_profit(self):
        return self.equity[-1, :] - self.investment

    @property
    @return_type(np.array)
    def roi(self):
        return self.total_net_profit / self.investment

    @property
    @return_type(np.array)
    def mdd(self):
        """A maximum drawdown (MDD) is the maximum observed loss from a peak 
        to a trough of a portfolio, before a new peak is attained."""
        return np.max(1 - self.equity / expanding_max_nb(self.equity), axis=0)
