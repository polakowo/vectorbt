import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import timedelta

from vectorbt import timeseries, accessors, defaults
from vectorbt.utils import indexing, checks, reshape_fns, common
from vectorbt.utils.common import cached_property
from vectorbt.portfolio import nb
from vectorbt.widgets import DefaultFigureWidget


class ResultWrapper():
    def __init__(self, base_ts):
        self.base_ts = base_ts

    def wrap_array(self, a):
        return self.base_ts.vbt.wrap_array(a)

    def wrap_metric(self, a):
        if checks.is_frame(self.base_ts):
            return pd.Series(a, index=self.base_ts.columns)
        # Single value
        if not checks.is_series(a):
            return a[0]
        return a

    def to_time_units(self, result):
        total_seconds = (self.base_ts.index[1] - self.base_ts.index[0]).total_seconds()
        to_timedelta = np.vectorize(lambda x: pd.Timedelta(
            timedelta(seconds=x * total_seconds)) if ~np.isnan(x) else np.nan, otypes=[np.object])
        return to_timedelta(result)

    def apply_mapper(self, map_func_nb, *args):
        raise NotImplementedError

    def apply_filter(self, obj, filter_func_nb, *args):
        checks.assert_numba_func(filter_func_nb)
        checks.assert_same_meta(self.base_ts, obj)

        result = nb.filter_map_results(
            obj.vbt.to_2d_array(),
            filter_func_nb,
            *args)
        return self.wrap_array(result)

    def apply_reducer(self, obj, reduce_func_nb, *args, time_units=False):
        checks.assert_numba_func(reduce_func_nb)
        checks.assert_same_meta(self.base_ts, obj)

        result = nb.reduce_map_results(
            obj.vbt.to_2d_array(),
            reduce_func_nb,
            *args)
        if time_units:
            result = self.to_time_units(result)
        return self.wrap_metric(result)


class BasePositions(ResultWrapper):
    def __init__(self, portfolio, pos_status=None, pos_filters=[]):
        ResultWrapper.__init__(self, portfolio.ts)

        self.portfolio = portfolio
        self.pos_status = pos_status
        self.pos_filters = pos_filters

    def apply_mapper(self, map_func_nb, *args):
        checks.assert_numba_func(map_func_nb)

        # Apply map
        result = nb.map_positions_nb(
            self.portfolio.shares.vbt.to_2d_array(),
            self.pos_status,
            map_func_nb,
            *args)
        result = self.wrap_array(result)

        # Apply passed filters
        for pos_filter in self.pos_filters:
            filter_func_nb = pos_filter[0]
            if len(pos_filter) > 1:
                args = pos_filter[1:]
            else:
                args = ()
            result = self.apply_filter(result, filter_func_nb, *args)
            result = self.wrap_array(result)

        return result

    # ############# Status ############# #

    @cached_property
    def status(self):
        return self.apply_mapper(nb.status_map_func_nb)

    @cached_property
    def count(self):
        return self.apply_reducer(self.status, nb.cnt_reduce_func_nb)

    # ############# Duration ############# #

    @cached_property
    def duration(self):
        return self.apply_mapper(nb.duration_map_func_nb, self.portfolio.ts.shape)

    @cached_property
    def avg_duration(self):
        return self.apply_reducer(self.duration, nb.mean_reduce_func_nb, time_units=True)

    # ############# PnL ############# #

    @cached_property
    def pnl(self):
        return self.apply_mapper(
            nb.pnl_map_func_nb,
            self.portfolio.ts.vbt.to_2d_array(),
            self.portfolio.cash.vbt.to_2d_array(),
            self.portfolio.shares.vbt.to_2d_array(),
            self.portfolio.init_capital)

    def plot_pnl(self, fig=None, profit_trace_kwargs={}, loss_trace_kwargs={}, **layout_kwargs):
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.pnl.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)

    @cached_property
    def total(self):
        return self.apply_reducer(self.pnl, nb.sum_reduce_func_nb)

    @cached_property
    def avg(self):
        return self.apply_reducer(self.pnl, nb.mean_reduce_func_nb)

    @cached_property
    def total_win(self):
        return self.apply_reducer(self.pnl, nb.pos_sum_reduce_func_nb)

    @cached_property
    def total_loss(self):
        return self.apply_reducer(self.pnl, nb.neg_sum_reduce_func_nb)

    @cached_property
    def avg_win(self):
        return self.apply_reducer(self.pnl, nb.pos_mean_reduce_func_nb)

    @cached_property
    def avg_loss(self):
        return self.apply_reducer(self.pnl, nb.neg_mean_reduce_func_nb)

    @cached_property
    def win_rate(self):
        return self.apply_reducer(self.pnl, nb.pos_rate_reduce_func_nb)

    @cached_property
    def loss_rate(self):
        return self.apply_reducer(self.pnl, nb.neg_rate_reduce_func_nb)

    @cached_property
    def profit_factor(self):
        total_win = reshape_fns.to_1d(self.total_win, raw=True)
        total_loss = reshape_fns.to_1d(self.total_loss, raw=True)
        profit_factor = total_win / total_loss
        return self.wrap_metric(profit_factor)

    @cached_property
    def expectancy(self):
        win_rate = reshape_fns.to_1d(self.win_rate, raw=True)
        avg_win = reshape_fns.to_1d(self.avg_win, raw=True)
        loss_rate = reshape_fns.to_1d(self.loss_rate, raw=True)
        avg_loss = reshape_fns.to_1d(self.avg_loss, raw=True)
        expectancy = win_rate * avg_win - loss_rate * avg_loss
        return self.wrap_metric(expectancy)

    # ############# Returns ############# #

    @cached_property
    def returns(self):
        return self.apply_mapper(
            nb.returns_map_func_nb,
            self.portfolio.ts.vbt.to_2d_array(),
            self.portfolio.cash.vbt.to_2d_array(),
            self.portfolio.shares.vbt.to_2d_array(),
            self.portfolio.init_capital)

    def plot_returns(self, fig=None, profit_trace_kwargs={}, loss_trace_kwargs={}, **layout_kwargs):
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.returns.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)


class Positions(BasePositions):
    def __init__(self, portfolio, pos_status=None, pos_filters=[]):
        super().__init__(portfolio, pos_status=pos_status, pos_filters=pos_filters)

        self.winning = BasePositions(
            portfolio,
            pos_status=pos_status,
            pos_filters=[*pos_filters, (nb.winning_filter_func_nb, self.pnl.vbt.to_2d_array())])
        self.losing = BasePositions(
            portfolio,
            pos_status=pos_status,
            pos_filters=[*pos_filters, (nb.losing_filter_func_nb, self.pnl.vbt.to_2d_array())])


def portfolio_indexing_func(obj, pd_indexing_func):
    return obj.__class__(
        pd_indexing_func(obj.ts),
        pd_indexing_func(obj.cash),
        pd_indexing_func(obj.shares),
        obj.init_capital,
        pd_indexing_func(obj.paid_fees),
        pd_indexing_func(obj.paid_slippage)
    )


@indexing.add_pd_indexing(portfolio_indexing_func)
class Portfolio(ResultWrapper):

    def __init__(self, ts, cash, shares, init_capital, paid_fees, paid_slippage):
        checks.assert_type(ts, (pd.Series, pd.DataFrame))
        checks.assert_same_meta(ts, cash)
        checks.assert_same_meta(ts, shares)
        ResultWrapper.__init__(self, ts)

        self.ts = ts
        self.cash = cash
        self.shares = shares
        self.init_capital = init_capital
        self.paid_fees = paid_fees
        self.paid_slippage = paid_slippage

        self.positions = Positions(self, pos_status=None)
        self.open_positions = Positions(self, pos_status=nb.OPEN)
        self.closed_positions = Positions(self, pos_status=nb.CLOSED)

    # ############# Magic methods ############# #

    def __add__(self, other):
        checks.assert_type(other, self.__class__)
        checks.assert_same(self.ts, other.ts)

        return self.__class__(
            self.ts,
            self.cash + other.cash,
            self.shares + other.shares,
            self.init_capital + other.init_capital,
            self.paid_fees + other.paid_fees,
            self.paid_slippage + other.paid_slippage
        )

    def __radd__(self, other):
        return Portfolio.__add__(self, other)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, ts, entries, exits, volume=np.inf, accumulate=False, init_capital=None,
                     fees=None, slippage=None, broadcast_kwargs={}):
        """Build portfolio based on entry and exit signals and the corresponding volume.

        Set volume to the number of shares to buy/sell.
        Set volume to np.inf to buy/sell everything.
        Set accumulate to `False` to avoid producing new orders if already in the market."""
        # Get defaults
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        init_capital = float(init_capital)
        if fees is None:
            fees = defaults.portfolio['fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(ts, (pd.Series, pd.DataFrame))
        checks.assert_type(entries, (pd.Series, pd.DataFrame))
        checks.assert_type(exits, (pd.Series, pd.DataFrame))
        entries.vbt.signals.validate()
        exits.vbt.signals.validate()

        # Broadcast inputs
        ts, entries, exits, volume = reshape_fns.broadcast(
            ts, entries, exits, volume, **broadcast_kwargs, writeable=True)
        fees = reshape_fns.broadcast_to(fees, ts, to_pd=False, writeable=True)
        slippage = reshape_fns.broadcast_to(slippage, ts, to_pd=False, writeable=True)

        # Perform calculation
        cash, shares, paid_fees, paid_slippage = nb.portfolio_np(
            reshape_fns.to_2d(ts, raw=True),
            init_capital,
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            nb.signals_order_func_np,
            reshape_fns.to_2d(entries, raw=True),
            reshape_fns.to_2d(exits, raw=True),
            reshape_fns.to_2d(volume, raw=True))

        # Bring to the same meta
        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)
        paid_fees = ts.vbt.wrap_array(paid_fees)
        paid_slippage = ts.vbt.wrap_array(paid_slippage)

        return cls(ts, cash, shares, init_capital, paid_fees, paid_slippage)

    @classmethod
    def from_volume(cls, ts, volume, is_target=False, init_capital=None, fees=None,
                    slippage=None, broadcast_kwargs={}):
        """Build portfolio based on volume.

        Set an volume element to positive/negative number - a number of shares to buy/sell.
        Set is_target to `True` to specify the target amount of shares to hold."""
        # Get defaults
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        init_capital = float(init_capital)
        if fees is None:
            fees = defaults.portfolio['fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(ts, (pd.Series, pd.DataFrame))
        checks.assert_type(volume, (pd.Series, pd.DataFrame))

        # Broadcast inputs
        ts, volume = reshape_fns.broadcast(ts, volume, **broadcast_kwargs, writeable=True)
        fees = reshape_fns.broadcast_to(fees, ts, to_pd=False, writeable=True)
        slippage = reshape_fns.broadcast_to(slippage, ts, to_pd=False, writeable=True)

        # Perform calculation
        cash, shares, paid_fees, paid_slippage = nb.portfolio_np(
            reshape_fns.to_2d(ts, raw=True),
            init_capital,
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            nb.volume_order_func_np,
            reshape_fns.to_2d(volume, raw=True),
            is_target)

        # Bring to the same meta
        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)
        paid_fees = ts.vbt.wrap_array(paid_fees)
        paid_slippage = ts.vbt.wrap_array(paid_slippage)

        return cls(ts, cash, shares, init_capital, paid_fees, paid_slippage)

    @classmethod
    def from_order_func(cls, ts, order_func_np, *args, init_capital=None, fees=None, slippage=None):
        """Build portfolio based on order function."""
        # Get defaults
        if init_capital is None:
            init_capital = defaults.portfolio['init_capital']
        init_capital = float(init_capital)
        if fees is None:
            fees = defaults.portfolio['fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']

        # Perform checks
        checks.assert_type(ts, (pd.Series, pd.DataFrame))
        checks.assert_numba_func(order_func_np)

        # Broadcast inputs
        fees = reshape_fns.broadcast_to(fees, ts, to_pd=False, writeable=True)
        slippage = reshape_fns.broadcast_to(slippage, ts, to_pd=False, writeable=True)

        # Perform calculation
        cash, shares, paid_fees, paid_slippage = nb.portfolio_np(
            reshape_fns.to_2d(ts, raw=True),
            init_capital,
            reshape_fns.to_2d(fees, raw=True),
            reshape_fns.to_2d(slippage, raw=True),
            order_func_np,
            *args)

        # Bring to the same meta
        cash = ts.vbt.wrap_array(cash)
        shares = ts.vbt.wrap_array(shares)
        paid_fees = ts.vbt.wrap_array(paid_fees)
        paid_slippage = ts.vbt.wrap_array(paid_slippage)

        return cls(ts, cash, shares, init_capital, paid_fees, paid_slippage)

    # ############# Time series ############# #

    @cached_property
    def equity(self):
        equity = self.cash.vbt.to_2d_array() + self.shares.vbt.to_2d_array() * self.ts.vbt.to_2d_array()
        return self.wrap_array(equity)

    @cached_property
    def equity_in_shares(self):
        equity_in_shares = self.equity.vbt.to_2d_array() / self.ts.vbt.to_2d_array()
        return self.wrap_array(equity_in_shares)

    @cached_property
    def returns(self):
        returns = timeseries.nb.pct_change_nb(self.equity.vbt.to_2d_array())
        return self.wrap_array(returns)

    @cached_property
    def trades(self):
        shares = self.shares.vbt.to_2d_array()
        trades = timeseries.nb.fillna_nb(timeseries.nb.diff_nb(shares), 0)
        trades[0, :] = shares[0, :]
        return self.wrap_array(trades)

    def plot_trades(self,
                    buy_trace_kwargs={},
                    sell_trace_kwargs={},
                    fig=None,
                    **layout_kwargs):
        checks.assert_type(self.ts, pd.Series)
        checks.assert_type(self.trades, pd.Series)
        sell_mask = self.trades < 0
        buy_mask = self.trades > 0

        # Plot time series
        fig = self.ts.vbt.timeseries.plot(fig=fig, **layout_kwargs)
        # Plot markers
        buy_trace_kwargs = common.merge_kwargs(dict(
            customdata=self.trades[buy_mask],
            hovertemplate='(%{x}, %{y})<br>%{customdata:.6g}',
            marker=dict(
                symbol='triangle-up',
                color='limegreen'
            )
        ), buy_trace_kwargs)
        buy_mask.vbt.signals.plot_markers(
            self.ts, name='Buy', trace_kwargs=buy_trace_kwargs, fig=fig, **layout_kwargs)
        sell_trace_kwargs = common.merge_kwargs(dict(
            customdata=self.trades[sell_mask],
            hovertemplate='(%{x}, %{y})<br>%{customdata:.6g}',
            marker=dict(
                symbol='triangle-down',
                color='orangered'
            )
        ), sell_trace_kwargs)
        sell_mask.vbt.signals.plot_markers(
            self.ts, name='Sell', trace_kwargs=sell_trace_kwargs, fig=fig, **layout_kwargs)
        return fig

    @cached_property
    def drawdown(self):
        drawdown = 1 - self.equity.vbt.to_2d_array() / timeseries.nb.expanding_max_nb(self.equity.vbt.to_2d_array())
        return self.wrap_array(drawdown)

    # ############# Costs ############# #

    @cached_property
    def total_paid_fees(self):
        total_paid_fees = np.sum(self.paid_fees.vbt.to_2d_array(), axis=0)
        return self.wrap_metric(total_paid_fees)

    @cached_property
    def total_paid_slippage(self):
        total_paid_slippage = np.sum(self.paid_slippage.vbt.to_2d_array(), axis=0)
        return self.wrap_metric(total_paid_slippage)

    @cached_property
    def total_costs(self):
        return self.total_paid_fees + self.total_paid_slippage

    # ############# General ############# #

    @cached_property
    def total_profit(self):
        total_profit = self.equity.vbt.to_2d_array()[-1, :] - self.init_capital
        return self.wrap_metric(total_profit)

    @cached_property
    def total_return(self):
        total_return = reshape_fns.to_1d(self.total_profit, raw=True) / self.init_capital
        return self.wrap_metric(total_return)

    @cached_property
    def annual_return(self):
        return self.returns.vbt.timeseries.groupby_apply(self.ts.index.year, nb.total_return_apply_func_nb)

    @cached_property
    def max_drawdown(self):
        max_drawdown = np.max(self.drawdown.vbt.to_2d_array(), axis=0)
        return self.wrap_metric(max_drawdown)
