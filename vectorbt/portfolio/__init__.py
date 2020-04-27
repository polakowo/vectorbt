import numpy as np
import pandas as pd
import plotly.graph_objects as go

from vectorbt import utils, timeseries, accessors
from vectorbt.portfolio import nb
from vectorbt.widgets import FigureWidget

# You can change default portfolio values from code
portfolio_defaults = utils.Config(
    investment=1.,
    slippage=0.,
    commission=0.
)


def indexing_func(obj, loc_pandas_func):
    return obj.__class__(
        loc_pandas_func(obj.ts),
        loc_pandas_func(obj.cash),
        loc_pandas_func(obj.shares),
        obj.investment,
        obj.slippage,
        obj.commission
    )


@utils.add_indexing(indexing_func)
class Portfolio():

    def __init__(self, ts, cash, shares, investment, slippage, commission):
        utils.assert_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()

        utils.assert_same_meta(ts, cash)
        utils.assert_same_meta(ts, shares)

        self.ts = ts
        self.cash = cash
        self.shares = shares
        self.investment = investment
        self.slippage = slippage
        self.commission = commission

    # ############# Magic methods ############# #

    def __add__(self, other):
        utils.assert_type(other, self.__class__)
        utils.assert_same(self.ts, other.ts)
        utils.assert_same(self.slippage, other.slippage)
        utils.assert_same(self.commission, other.commission)

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
        Set accumulate to False to avoid producing new orders if already in the market."""
        if investment is None:
            investment = portfolio_defaults['investment']
        if slippage is None:
            slippage = portfolio_defaults['slippage']
        if commission is None:
            commission = portfolio_defaults['commission']

        utils.assert_type(ts, (pd.Series, pd.DataFrame))
        utils.assert_type(entries, (pd.Series, pd.DataFrame))
        utils.assert_type(exits, (pd.Series, pd.DataFrame))

        ts.vbt.timeseries.validate()
        entries.vbt.signals.validate()
        exits.vbt.signals.validate()

        ts, entries, exits = utils.broadcast(ts, entries, exits, **broadcast_kwargs, writeable=True)

        volume = utils.broadcast_to(volume, ts, writeable=True, copy_kwargs={'dtype': np.float64})

        investment = float(investment)
        slippage = float(slippage)
        commission = float(commission)

        cash, shares = nb.portfolio_from_signals_np(
            ts.vbt.to_2d_array(),
            investment,
            slippage,
            commission,
            entries.vbt.to_2d_array(),
            exits.vbt.to_2d_array(),
            volume.vbt.to_2d_array(),
            accumulate)

        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)

        return cls(ts, cash, shares, investment, slippage, commission)

    @classmethod
    def from_orders(cls, ts, orders, is_target=False, investment=None, slippage=None, commission=None, broadcast_kwargs={}):
        """Build portfolio based on orders.

        Set an orders element to positive/negative number - a number of shares to buy/sell.
        Set is_target to True to specify the target amount of shares to hold."""
        if investment is None:
            investment = portfolio_defaults['investment']
        if slippage is None:
            slippage = portfolio_defaults['slippage']
        if commission is None:
            commission = portfolio_defaults['commission']

        utils.assert_type(ts, (pd.Series, pd.DataFrame))
        utils.assert_type(orders, (pd.Series, pd.DataFrame))

        ts.vbt.timeseries.validate()
        orders.vbt.timeseries.validate()

        ts, orders = utils.broadcast(ts, orders, **broadcast_kwargs, writeable=True)

        investment = float(investment)
        slippage = float(slippage)
        commission = float(commission)

        cash, shares = nb.portfolio_from_orders_np(
            ts.vbt.to_2d_array(),
            investment,
            slippage,
            commission,
            orders.vbt.to_2d_array(),
            is_target)

        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)

        return cls(ts, cash, shares, investment, slippage, commission)

    @classmethod
    def from_order_func(cls, ts, order_func_np, *args, investment=None, slippage=None, commission=None):
        """Build portfolio based on order function."""
        if investment is None:
            investment = portfolio_defaults['investment']
        if slippage is None:
            slippage = portfolio_defaults['slippage']
        if commission is None:
            commission = portfolio_defaults['commission']

        utils.assert_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()

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

    @utils.common.cached_property
    def equity(self):
        return self.ts.vbt.wrap_array(self.cash.vbt.to_2d_array() + self.shares.vbt.to_2d_array() * self.ts.vbt.to_2d_array())

    @utils.common.cached_property
    def equity_in_shares(self):
        return self.ts.vbt.wrap_array(self.equity.vbt.to_2d_array() / self.ts.vbt.to_2d_array())

    @utils.common.cached_property
    def returns(self):
        return self.ts.vbt.wrap_array(timeseries.nb.pct_change_nb(self.equity.vbt.to_2d_array()))

    @utils.common.cached_property
    def drawdown(self):
        drawdown = 1 - self.equity.vbt.to_2d_array() / timeseries.nb.expanding_max_nb(self.equity.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(drawdown)

    @utils.common.cached_property
    def trades(self):
        shares = self.shares.vbt.to_2d_array()
        trades = timeseries.nb.fillna_nb(timeseries.nb.diff_nb(shares), 0)
        trades[0, :] = shares[0, :]
        return self.ts.vbt.wrap_array(trades)

    @utils.common.cached_property
    def position_profits(self):
        position_profits = nb.position_profits_nb(self.trades.vbt.to_2d_array(), self.equity.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(position_profits)

    @utils.common.cached_property
    def position_returns(self):
        position_returns = nb.position_returns_nb(self.trades.vbt.to_2d_array(), self.equity.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(position_returns)

    @utils.common.cached_property
    def win_mask(self):
        position_profits = self.position_profits.vbt.to_2d_array().copy()
        position_profits[np.isnan(position_profits)] = 0  # avoid warnings
        win_mask = position_profits > 0
        return self.ts.vbt.wrap_array(win_mask)

    @utils.common.cached_property
    def loss_mask(self):
        position_profits = self.position_profits.vbt.to_2d_array().copy()
        position_profits[np.isnan(position_profits)] = 0
        loss_mask = position_profits < 0
        return self.ts.vbt.wrap_array(loss_mask)

    @utils.common.cached_property
    def position_mask(self):
        position_mask = ~np.isnan(self.position_profits.vbt.to_2d_array())
        return self.ts.vbt.wrap_array(position_mask)

    # ############# Performance metrics ############# #

    def wrap_metric(self, a):
        if utils.is_frame(self.ts):
            return pd.Series(a, index=self.ts.columns)
        # Single value
        if utils.is_array(a):
            return a[0]
        return a

    @utils.common.cached_property
    def sum_win(self):
        """Sum of wins."""
        sum_win = nb.sum_win_nb(self.position_profits.vbt.to_2d_array())
        return self.wrap_metric(sum_win)

    @utils.common.cached_property
    def sum_loss(self):
        """Sum of losses (always positive)."""
        sum_loss = nb.sum_loss_nb(self.position_profits.vbt.to_2d_array())
        return self.wrap_metric(sum_loss)

    @utils.common.cached_property
    def avg_win(self):
        """Average win."""
        avg_win = nb.avg_win_nb(self.position_profits.vbt.to_2d_array())
        return self.wrap_metric(avg_win)

    @utils.common.cached_property
    def avg_loss(self):
        """Average loss (always positive)."""
        avg_loss = nb.avg_loss_nb(self.position_profits.vbt.to_2d_array())
        return self.wrap_metric(avg_loss)

    @utils.common.cached_property
    def win_rate(self):
        """Fraction of wins."""
        win_rate = np.sum(self.win_mask.vbt.to_2d_array(), axis=0) / \
            np.sum(self.position_mask.vbt.to_2d_array(), axis=0)
        return self.wrap_metric(win_rate)

    @utils.common.cached_property
    def loss_rate(self):
        """Fraction of losses."""
        loss_rate = np.sum(self.loss_mask.vbt.to_2d_array(), axis=0) / \
            np.sum(self.position_mask.vbt.to_2d_array(), axis=0)
        return self.wrap_metric(loss_rate)

    @utils.common.cached_property
    def profit_factor(self):
        profit_factor = utils.to_1d(self.sum_win, raw=True) / utils.to_1d(self.sum_loss, raw=True)
        return self.wrap_metric(profit_factor)

    @utils.common.cached_property
    def appt(self):
        """Average profitability per trade (APPT)

        For every trade you place, you are likely to win/lose this amount.
        What matters is that your APPT comes up positive."""
        appt = utils.to_1d(self.win_rate, raw=True) * utils.to_1d(self.avg_win, raw=True) - \
            utils.to_1d(self.loss_rate, raw=True) * utils.to_1d(self.avg_loss, raw=True)
        return self.wrap_metric(appt)

    @utils.common.cached_property
    def total_profit(self):
        total_profit = self.equity.vbt.to_2d_array()[-1, :] - self.investment
        return self.wrap_metric(total_profit)

    @utils.common.cached_property
    def total_return(self):
        total_return = utils.to_1d(self.total_profit, raw=True) / self.investment
        return self.wrap_metric(total_return)

    @utils.common.cached_property
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
        utils.assert_type(self.ts, pd.Series)
        utils.assert_type(self.trades, pd.Series)
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

    def plot_position_profits(self,
                              profit_trace_kwargs={},
                              loss_trace_kwargs={},
                              fig=None,
                              **layout_kwargs):
        utils.assert_type(self.position_profits, pd.Series)
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
        profit_scatter.update(**profit_trace_kwargs)
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
        loss_scatter.update(**loss_trace_kwargs)
        fig.add_trace(loss_scatter)

        # Set up axes
        maxval = np.nanmax(np.abs(self.position_profits.vbt.to_2d_array()))
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
