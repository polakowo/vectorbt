import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt import timeseries, accessors, defaults
from vectorbt.utils import indexing, checks, reshape_fns
from vectorbt.utils.common import cached_property
from vectorbt.portfolio import nb
from vectorbt.widgets import DefaultFigureWidget


def indexing_func(obj, pd_indexing_func):
    return obj.__class__(
        pd_indexing_func(obj.ts),
        pd_indexing_func(obj.cash),
        pd_indexing_func(obj.shares),
        obj.investment,
        obj.slippage,
        obj.commission
    )


@indexing.add_pd_indexing(indexing_func)
class Portfolio():

    def __init__(self, ts, cash, shares, investment, slippage, commission):
        checks.assert_type(ts, (pd.Series, pd.DataFrame))

        checks.assert_same_meta(ts, cash)
        checks.assert_same_meta(ts, shares)

        self.ts = ts
        self.cash = cash
        self.shares = shares
        self.investment = investment
        self.slippage = slippage
        self.commission = commission

    # ############# Magic methods ############# #

    def __add__(self, other):
        checks.assert_type(other, self.__class__)
        checks.assert_same(self.ts, other.ts)
        checks.assert_same(self.slippage, other.slippage)
        checks.assert_same(self.commission, other.commission)

        return self.__class__(
            self.ts,
            self.cash + other.cash,
            self.shares + other.shares,
            self.investment + other.investment,
            self.slippage,
            self.commission
        )

    def __radd__(self, other):
        return Portfolio.__add__(self, other)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, ts, entries, exits, volume=np.inf, accumulate=False, investment=None, slippage=None, commission=None, broadcast_kwargs={}):
        """Build portfolio based on entry and exit signals and the corresponding volume.

        Set volume to the number of shares to buy/sell.
        Set volume to np.inf to buy/sell everything.
        Set accumulate to `False` to avoid producing new orders if already in the market."""
        if investment is None:
            investment = defaults.portfolio['investment']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if commission is None:
            commission = defaults.portfolio['commission']

        checks.assert_type(ts, (pd.Series, pd.DataFrame))
        checks.assert_type(entries, (pd.Series, pd.DataFrame))
        checks.assert_type(exits, (pd.Series, pd.DataFrame))

        entries.vbt.signals.validate()
        exits.vbt.signals.validate()

        ts, entries, exits = reshape_fns.broadcast(ts, entries, exits, **broadcast_kwargs, writeable=True)

        volume = reshape_fns.broadcast_to(volume, ts, writeable=True, copy_kwargs={'dtype': np.float64})

        investment = float(investment)
        slippage = float(slippage)
        commission = float(commission)

        cash, shares = nb.portfolio_np(
            ts.vbt.to_2d_array(),
            investment,
            slippage,
            commission,
            nb.signals_order_func_np,
            entries.vbt.to_2d_array(),
            exits.vbt.to_2d_array(),
            volume.vbt.to_2d_array())

        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)

        return cls(ts, cash, shares, investment, slippage, commission)

    @classmethod
    def from_volume(cls, ts, volume, is_target=False, investment=None, slippage=None, commission=None, broadcast_kwargs={}):
        """Build portfolio based on volume.

        Set an volume element to positive/negative number - a number of shares to buy/sell.
        Set is_target to `True` to specify the target amount of shares to hold."""
        if investment is None:
            investment = defaults.portfolio['investment']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if commission is None:
            commission = defaults.portfolio['commission']

        checks.assert_type(ts, (pd.Series, pd.DataFrame))
        checks.assert_type(volume, (pd.Series, pd.DataFrame))

        ts, volume = reshape_fns.broadcast(ts, volume, **broadcast_kwargs, writeable=True)

        investment = float(investment)
        slippage = float(slippage)
        commission = float(commission)

        cash, shares = nb.portfolio_np(
            ts.vbt.to_2d_array(),
            investment,
            slippage,
            commission,
            nb.volume_order_func_np,
            volume.vbt.to_2d_array(),
            is_target)

        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)

        return cls(ts, cash, shares, investment, slippage, commission)

    @classmethod
    def from_order_func(cls, ts, order_func_np, *args, investment=None, slippage=None, commission=None):
        """Build portfolio based on order function."""
        if investment is None:
            investment = defaults.portfolio['investment']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if commission is None:
            commission = defaults.portfolio['commission']

        checks.assert_type(ts, (pd.Series, pd.DataFrame))

        investment = float(investment)
        slippage = float(slippage)
        commission = float(commission)

        cash, shares = nb.portfolio_np(
            ts.vbt.to_2d_array(),
            investment,
            slippage,
            commission,
            order_func_np,
            *args)

        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)

        return cls(ts, cash, shares, investment, slippage, commission)

    # ############# General properties ############# #

    @cached_property
    def equity(self):
        equity = self.cash.vbt.to_2d_array() + self.shares.vbt.to_2d_array() * self.ts.vbt.to_2d_array()
        return self.ts.vbt.wrap_array(equity)

    @cached_property
    def equity_in_shares(self):
        equity_in_shares = self.equity.vbt.to_2d_array() / self.ts.vbt.to_2d_array()
        return self.ts.vbt.wrap_array(equity_in_shares)

    @cached_property
    def returns(self):
        returns = timeseries.nb.pct_change_nb(self.equity.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(returns)

    @cached_property
    def returns_in_shares(self):
        returns_in_shares = timeseries.nb.pct_change_nb(self.equity_in_shares.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(returns_in_shares)

    @cached_property
    def drawdown(self):
        drawdown = 1 - self.equity.vbt.to_2d_array() / timeseries.nb.expanding_max_nb(self.equity.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(drawdown)

    @cached_property
    def trades(self):
        shares = self.shares.vbt.to_2d_array()
        trades = timeseries.nb.fillna_nb(timeseries.nb.diff_nb(shares), 0)
        trades[0, :] = shares[0, :]
        return self.ts.vbt.wrap_array(trades)

    @cached_property
    def position_pnl(self):
        position_pnl = nb.map_positions_nb(
            self.shares.vbt.to_2d_array(),
            nb.pnl_map_func_nb,
            self.ts.vbt.to_2d_array(),
            self.cash.vbt.to_2d_array(),
            self.shares.vbt.to_2d_array(),
            self.investment)
        return self.ts.vbt.wrap_array(position_pnl)

    @cached_property
    def position_returns(self):
        position_returns = nb.map_positions_nb(
            self.shares.vbt.to_2d_array(),
            nb.returns_map_func_nb,
            self.ts.vbt.to_2d_array(),
            self.cash.vbt.to_2d_array(),
            self.shares.vbt.to_2d_array(),
            self.investment)
        return self.ts.vbt.wrap_array(position_returns)

    # ############# Performance metrics ############# #

    def wrap_metric(self, a):
        if checks.is_frame(self.ts):
            return pd.Series(a, index=self.ts.columns)
        # Single value
        if not checks.is_series(a):
            return a[0]
        return a

    @cached_property
    def win_sum(self):
        win_sum = nb.reduce_map_results(self.position_pnl.vbt.to_2d_array(), nb.win_sum_reduce_func_nb)
        return self.wrap_metric(win_sum)

    @cached_property
    def loss_sum(self):
        loss_sum = nb.reduce_map_results(self.position_pnl.vbt.to_2d_array(), nb.loss_sum_reduce_func_nb)
        return self.wrap_metric(loss_sum)

    @cached_property
    def win_mean(self):
        win_mean = nb.reduce_map_results(self.position_pnl.vbt.to_2d_array(), nb.win_mean_reduce_func_nb)
        return self.wrap_metric(win_mean)

    @cached_property
    def loss_mean(self):
        loss_mean = nb.reduce_map_results(self.position_pnl.vbt.to_2d_array(), nb.loss_mean_reduce_func_nb)
        return self.wrap_metric(loss_mean)

    @cached_property
    def win_rate(self):
        win_rate = nb.reduce_map_results(self.position_pnl.vbt.to_2d_array(), nb.win_rate_reduce_func_nb)
        return self.wrap_metric(win_rate)

    @cached_property
    def loss_rate(self):
        loss_rate = nb.reduce_map_results(self.position_pnl.vbt.to_2d_array(), nb.loss_rate_reduce_func_nb)
        return self.wrap_metric(loss_rate)

    @cached_property
    def profit_factor(self):
        profit_factor = reshape_fns.to_1d(self.win_sum, raw=True) / reshape_fns.to_1d(self.loss_sum, raw=True)
        return self.wrap_metric(profit_factor)

    @cached_property
    def appt(self):
        """Average profitability per trade (APPT)

        For every trade you place, you are likely to win/lose this amount.
        What matters is that your APPT comes up positive."""
        appt = reshape_fns.to_1d(self.win_rate, raw=True) * reshape_fns.to_1d(self.win_mean, raw=True) - \
            reshape_fns.to_1d(self.loss_rate, raw=True) * reshape_fns.to_1d(self.loss_mean, raw=True)
        return self.wrap_metric(appt)

    @cached_property
    def total_profit(self):
        total_profit = self.equity.vbt.to_2d_array()[-1, :] - self.investment
        return self.wrap_metric(total_profit)

    @cached_property
    def total_return(self):
        total_return = reshape_fns.to_1d(self.total_profit, raw=True) / self.investment
        return self.wrap_metric(total_return)

    @cached_property
    def mdd(self):
        """A maximum drawdown (MDD) is the maximum observed loss from a peak 
        to a trough of a portfolio, before a new peak is attained."""
        mdd = np.max(self.drawdown.vbt.to_2d_array(), axis=0)
        return self.wrap_metric(mdd)

    # ############# Plotting ############# #

    def plot_trades(self,
                    buy_trace_kwargs={},
                    sell_trace_kwargs={},
                    fig=None,
                    **ts_kwargs):
        checks.assert_type(self.ts, pd.Series)
        checks.assert_type(self.trades, pd.Series)
        sell_mask = self.trades < 0
        buy_mask = self.trades > 0

        # Plot TimeSeries
        fig = self.ts.vbt.timeseries.plot(fig=fig, **ts_kwargs)

        # Plot markers
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
        buy_scatter.update(**buy_trace_kwargs)
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
        sell_scatter.update(**sell_trace_kwargs)
        fig.add_trace(sell_scatter)

        return fig

    def plot_position_pnl(self,
                          profit_trace_kwargs={},
                          loss_trace_kwargs={},
                          fig=None,
                          **layout_kwargs):
        checks.assert_type(self.position_pnl, pd.Series)
        profits = self.position_pnl.copy()
        profits[self.position_pnl <= 0] = np.nan
        losses = self.position_pnl.copy()
        losses[self.position_pnl >= 0] = np.nan

        # Set up figure
        if fig is None:
            fig = DefaultFigureWidget()
            fig.update_layout(showlegend=True)
            fig.update_layout(**layout_kwargs)

        # Plot markets
        profit_scatter = go.Scatter(
            x=self.position_pnl.index,
            y=profits,
            mode='markers',
            marker=dict(
                symbol='circle',
                color='green',
                size=10
            ),
            name='Profit'
        )
        profit_scatter.update(**profit_trace_kwargs)
        fig.add_trace(profit_scatter)
        loss_scatter = go.Scatter(
            x=self.position_pnl.index,
            y=losses,
            mode='markers',
            marker=dict(
                symbol='circle',
                color='red',
                size=10
            ),
            name='Loss'
        )
        loss_scatter.update(**loss_trace_kwargs)
        fig.add_trace(loss_scatter)

        # Set up axes
        maxval = np.nanmax(np.abs(self.position_pnl.vbt.to_2d_array()))
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
