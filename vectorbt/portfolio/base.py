"""Base class for modeling portfolio and measuring its performance.

The job of the `Portfolio` class is to create a series of positions allocated 
against a cash component, produce an equity curve, incorporate basic transaction costs
and produce a set of statistics about its performance. In particular it outputs
position/profit metrics and drawdown information."""

import numpy as np
import pandas as pd

from vectorbt import defaults
from vectorbt.utils import checks
from vectorbt.utils.decorators import cached_method
from vectorbt.utils.config import Configured, merge_kwargs
from vectorbt.utils.random import set_seed
from vectorbt.base.reshape_fns import to_1d, to_2d, broadcast
from vectorbt.base.indexing import PandasIndexer
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.generic import nb as generic_nb
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import SizeType, AccumulateExitMode, ConflictMode, CallSeqType
from vectorbt.records import Orders, Trades, Positions, Drawdowns
from vectorbt.records.orders import indexing_on_orders_meta


def _indexing_func(obj, pd_indexing_func):
    """Perform indexing on `Portfolio`."""
    new_orders, group_idxs, col_idxs = indexing_on_orders_meta(obj.orders, pd_indexing_func)
    if obj.wrapper.grouper.group_by is None:
        # Grouping disabled
        new_init_cash = obj.init_cash.values[col_idxs]
    else:
        # Grouping enabled
        new_init_cash = obj.init_cash.values[group_idxs if obj.cash_sharing else col_idxs]
    new_call_seq = obj.call_seq.values[:, col_idxs]

    return obj.copy(
        orders=new_orders,
        init_cash=new_init_cash,
        call_seq=new_call_seq
    )


def add_returns_methods(func_names):
    """Class decorator to add `vectorbt.returns.accessors.Returns_Accessor` methods to `Portfolio`."""

    def wrapper(cls):
        for func_name in func_names:
            if isinstance(func_name, tuple):
                ret_func_name = func_name[0]
            else:
                ret_func_name = func_name

            def returns_method(
                    self,
                    *args,
                    group_by=None,
                    year_freq=None,
                    ret_func_name=ret_func_name,
                    active_returns=False,
                    **kwargs):
                if active_returns:
                    returns = self.active_returns(group_by=group_by)
                else:
                    returns = self.returns(group_by=group_by)
                returns_acc = returns.vbt.returns(freq=self.wrapper.freq, year_freq=year_freq)
                return getattr(returns_acc, ret_func_name)(*args, **kwargs)

            if isinstance(func_name, tuple):
                func_name = func_name[1]
            returns_method.__qualname__ = f"Portfolio.{func_name}"
            returns_method.__doc__ = f"See `vectorbt.returns.accessors.Returns_Accessor.{func_name}`."
            setattr(cls, func_name, cached_method(returns_method))
        return cls

    return wrapper


@add_returns_methods([
    ('daily', 'daily_returns'),
    ('annual', 'annual_returns'),
    ('cumulative', 'cumulative_returns'),
    ('total', 'total_return'),
    ('annualized', 'annualized_return'),
    'annualized_volatility',
    'calmar_ratio',
    'omega_ratio',
    'sharpe_ratio',
    'downside_risk',
    'sortino_ratio',
    'information_ratio',
    'beta',
    'alpha',
    'tail_ratio',
    'value_at_risk',
    'conditional_value_at_risk',
    'capture',
    'up_capture',
    'down_capture',
    'drawdown',
    'max_drawdown'
])
class Portfolio(Configured, PandasIndexer):
    """Class for modeling portfolio and measuring its performance.

    Args:
        orders (Orders): Order records.
        init_cash (float or array_like): Initial capital.
        cash_sharing (bool): Whether to share cash within the same group.
        call_seq (array_like): Sequence of calls per row and group.
        incl_unrealized (bool): Whether to include unrealized P&L in statistics.

    !!! note
        Use class methods with `from_` prefix to build a portfolio.
        The `__init__` method is reserved for indexing purposes."""

    def __init__(self, orders, init_cash, cash_sharing, call_seq, incl_unrealized=None):
        Configured.__init__(
            self,
            orders=orders,
            init_cash=init_cash,
            cash_sharing=cash_sharing,
            call_seq=call_seq,
            incl_unrealized=incl_unrealized
        )
        # Get defaults
        if incl_unrealized is None:
            incl_unrealized = defaults.portfolio['incl_unrealized']

        # Perform checks
        checks.assert_type(orders, Orders)

        # Store passed arguments
        self._ref_price = orders.close
        self._orders = orders
        self._init_cash = init_cash
        self._cash_sharing = cash_sharing
        self._call_seq = call_seq
        self._incl_unrealized = incl_unrealized

        # Supercharge
        PandasIndexer.__init__(self, _indexing_func)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, close, entries, exits, size=np.inf, entry_price=None, exit_price=None,
                     fees=None, fixed_fees=None, slippage=None, min_size=None, reject_prob=None,
                     init_cash=None, cash_sharing=None, call_seq=None, accumulate=None,
                     accumulate_exit_mode=None, conflict_mode=None, seed=None, freq=None, group_by=None,
                     broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Simulate portfolio from entry and exit signals.

        Starting with initial cash `init_cash`, for each signal in `entries`, enters a position
        by buying `size` of shares for `entry_price`. For each signal in `exits`, closes the position
        by selling all shares for `exit_price`. When accumulation is enabled, each entry signal will
        increase the position, and optionally each exit signal will decrease the position. When both
        entry and exit signals are present, ignores them by default. When grouping is enabled with
        `group_by`, will compute performance for the entire group. When, additionally, `cash_sharing`
        is enabled, will share the cash among all columns in the group.

        Args:
            close (pandas_like): Reference price, such as close. Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            entries (array_like): Boolean array of entry signals. Will broadcast.
            exits (array_like): Boolean array of exit signals. Will broadcast.
            size (float or array_like): Size to order. Will broadcast.

                * Set to positive/negative to buy/sell.
                * Set to `np.inf`/`-np.inf` to buy/sell everything.
                * Set to `np.nan` or zero to skip.
            entry_price (array_like): Entry price. Defaults to `close`. Will broadcast.

                !!! note
                    Setting order price to close is risky.
            exit_price (array_like): Exit price. Defaults to `close`. Will broadcast.

                !!! note
                    Setting order price to close is risky.
            fees (float or array_like): Fees in percentage of the order value. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price. Will broadcast.
            min_size (float or array_like): Minimum size for an order to be accepted. Will broadcast.
            reject_prob (float or array_like): Order rejection probability. Will broadcast.
            init_cash (float or array_like): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
            cash_sharing (bool): Whether to share cash within the same group.

                !!! warning
                    Order execution cannot be considered parallel anymore.

                    This method presumes that in a group of assets that share the same capital all
                    orders will be executed within the same tick and retain their price regardless
                    of their position in the queue, even though they depend upon each other and thus
                    cannot be executed in parallel. This behavior is risky.
            call_seq (int or array_like): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.
            accumulate (bool): If `accumulate` is `True`, entering the market when already
                in the market will be allowed to increase the position.
            accumulate_exit_mode (int): See `vectorbt.portfolio.enums.AccumulateExitMode`.
            conflict_mode (int): See `vectorbt.portfolio.enums.ConflictMode`.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            freq (any): Index frequency in case `close.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.
            **kwargs: Keyword arguments passed to the `__init__` method.

        All time series will be broadcast together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            Only `SizeType.Shares` is supported. Other modes such as target percentage are not
            compatible with signals since their logic may contradict the direction the user has
            specified for the order.

            With cash sharing enabled, at each timestamp, processing of the assets in a group
            goes strictly in order defined in `call_seq`. This order can't be changed dynamically.

        !!! hint
            If you generated signals using close price, don't forget to shift your signals by one tick
            forward, for example, with `signals.vbt.fshift(1)`. In general, make sure to use price
            that comes after you generated your signals.
        """
        # Get defaults
        if size is None:
            size = defaults.portfolio['size']
        if entry_price is None:
            entry_price = close
        if exit_price is None:
            exit_price = close
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if min_size is None:
            min_size = defaults.portfolio['min_size']
        if reject_prob is None:
            reject_prob = defaults.portfolio['reject_prob']
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
            if isinstance(call_seq, str):
                call_seq = getattr(CallSeqType, call_seq)
        if accumulate is None:
            accumulate = defaults.portfolio['accumulate']
        if accumulate_exit_mode is None:
            accumulate_exit_mode = defaults.portfolio['accumulate_exit_mode']
            if isinstance(accumulate_exit_mode, str):
                accumulate_exit_mode = getattr(AccumulateExitMode, accumulate_exit_mode)
        if conflict_mode is None:
            conflict_mode = defaults.portfolio['conflict_mode']
            if isinstance(conflict_mode, str):
                conflict_mode = getattr(ConflictMode, conflict_mode)
        if seed is None:
            seed = defaults.portfolio['seed']
        if seed is not None:
            set_seed(seed)
        if freq is None:
            freq = defaults.portfolio['freq']
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Perform checks
        checks.assert_type(close, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(close, np.floating)
        checks.assert_dtype(entries, np.bool)
        checks.assert_dtype(exits, np.bool)
        checks.assert_subdtype(size, np.floating)
        checks.assert_subdtype(entry_price, np.floating)
        checks.assert_subdtype(exit_price, np.floating)
        checks.assert_subdtype(fees, np.floating)
        checks.assert_subdtype(fixed_fees, np.floating)
        checks.assert_subdtype(slippage, np.floating)
        checks.assert_subdtype(min_size, np.floating)
        checks.assert_subdtype(reject_prob, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(call_seq, np.integer)

        # Broadcast inputs
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        keep_raw = (False, True, True, True, True, True, True, True, True, True, True)
        broadcast_kwargs = merge_kwargs(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        close, entries, exits, size, entry_price, \
            exit_price, fees, fixed_fees, slippage, min_size, reject_prob = broadcast(
                close, entries, exits, size, entry_price, exit_price, fees, fixed_fees,
                slippage, min_size, reject_prob, **broadcast_kwargs, keep_raw=keep_raw)
        target_shape = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape, group_counts, call_seq_type=call_seq)

        # Perform calculation
        order_records = nb.simulate_from_signals_nb(
            target_shape,
            cs_group_counts,  # group only if cash sharing is enabled to speed up
            init_cash,
            call_seq,
            entries,
            exits,
            size,
            entry_price,
            exit_price,
            fees,
            fixed_fees,
            slippage,
            min_size,
            reject_prob,
            accumulate,
            accumulate_exit_mode,
            conflict_mode,
            close.ndim == 2
        )

        # Create an instance
        orders = Orders(wrapper, order_records, close)
        return cls(
            orders,
            init_cash,
            cash_sharing,
            call_seq,
            **kwargs
        )

    @classmethod
    def from_orders(cls, close, order_size, size_type=None, order_price=None, fees=None, fixed_fees=None,
                    slippage=None, min_size=None, reject_prob=None, init_cash=None, cash_sharing=None,
                    call_seq=None, val_price=None, dynamic_call_seq=None, freq=None, seed=None, group_by=None,
                    broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Simulate portfolio from orders.

        Starting with initial cash `init_cash`, orders the number of shares specified in `order_size`
        for `order_price`.

        Args:
            close (pandas_like): Reference price, such as close. Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            order_size (float or array_like): Size to order. Will broadcast.

                For any size type:

                * Set to `np.nan` to skip.
                * Set to `np.inf`/`-np.inf` to buy/sell everything.

                For `SizeType.Shares`:

                * Set to positive/negative to buy/sell.
                * Set to zero to skip.

                For target size, the final size will depend upon current holdings.
            size_type (int or array_like): See `vectorbt.portfolio.enums.SizeType`.
            order_price (array_like): Order price. Defaults to `close`. Will broadcast.

                !!! note
                    Setting order price to close is risky.
            fees (float or array_like): Fees in percentage of the order value. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price. Will broadcast.
            min_size (float or array_like): Minimum size for an order to be accepted. Will broadcast.
            reject_prob (float or array_like): Order rejection probability. Will broadcast.
            init_cash (float or array_like): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
            cash_sharing (bool): Whether to share cash within the same group.

                !!! warning
                    Order execution cannot be considered parallel anymore.

                    This method presumes that in a group of assets that share the same capital all
                    orders will be executed within the same tick and retain their price regardless
                    of their position in the queue, even though they depend upon each other and thus
                    cannot be executed in parallel. This behavior is risky.
            call_seq (int or array_like): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.
            val_price (array_like): Group valuation price. Defaults to previous `close`. Will broadcast.

                Used for `SizeType.TargetPercent`.

                !!! note
                    Make sure to use timestamp for `val_price` that comes before timestamps of all orders
                    in the group with cash sharing, otherwise you're cheating yourself.
            dynamic_call_seq (bool): Whether to rearrange calls dynamically based on order value.
                Overrides `call_seq`.

                Calculates value of all orders per row and group, and sorts them by this value.
                Sell orders will be executed first to release funds for buy orders.

                !!! warning
                    This mode should be used with caution:

                    * It not only presumes that order prices are known beforehand, but also that orders
                    can be executed in arbitrary order and still retain their price. In reality, this is
                    hardly the case: after processing one asset, some time has passed and the price for
                    other assets might have already changed.
                    * Even if you're able to specify a slippage large enough to compensate for this behavior,
                    slippage itself should depend upon execution order. This method doesn't let you do that.
                    * If one order is rejected, it still will execute next orders and possibly leave
                    them without funds that could have been released by the first order.

                    For more control, use `Portfolio.from_order_func`.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            freq (any): Index frequency in case `close.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.

        All time series will be broadcast together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            When `dynamic_call_seq` is `False`, at each timestamp, processing of the assets in a group
            goes strictly in order defined in `call_seq`. This order can't be changed dynamically.

            This has one big implication for this particular method: the last asset in the call stack
            cannot be processed until other assets are processed. This is the reason why rebalancing
            cannot work properly in this setting: one has to specify percentages for all assets beforehand and
            then tweak the processing order to sell to-be-sold assets first in order to release funds for
            to-be-bought assets. This can be automatically done by enabling `dynamic_call_seq`.
        """
        # Get defaults
        if order_size is None:
            order_size = defaults.portfolio['order_size']
        if size_type is None:
            size_type = defaults.portfolio['size_type']
            if isinstance(size_type, str):
                size_type = getattr(SizeType, size_type)
        if order_price is None:
            order_price = close
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if min_size is None:
            min_size = defaults.portfolio['min_size']
        if reject_prob is None:
            reject_prob = defaults.portfolio['reject_prob']
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
            if isinstance(call_seq, str):
                call_seq = getattr(CallSeqType, call_seq)
        if val_price is None:
            val_price = close.vbt.fshift(1)
        if dynamic_call_seq is None:
            dynamic_call_seq = defaults.portfolio['dynamic_call_seq']
        if dynamic_call_seq:
            call_seq = CallSeqType.Default  # overrides
        if seed is None:
            seed = defaults.portfolio['seed']
        if seed is not None:
            set_seed(seed)
        if freq is None:
            freq = defaults.portfolio['freq']
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Perform checks
        checks.assert_type(close, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(close, np.floating)
        checks.assert_subdtype(order_size, np.floating)
        checks.assert_subdtype(size_type, np.integer)
        checks.assert_subdtype(order_price, np.floating)
        checks.assert_subdtype(fees, np.floating)
        checks.assert_subdtype(fixed_fees, np.floating)
        checks.assert_subdtype(slippage, np.floating)
        checks.assert_subdtype(min_size, np.floating)
        checks.assert_subdtype(reject_prob, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(call_seq, np.integer)
        checks.assert_subdtype(val_price, np.floating)

        # Broadcast inputs
        # Only close is broadcast, others can remain unchanged thanks to flexible indexing
        keep_raw = (False, True, True, True, True, True, True, True, True, True)
        broadcast_kwargs = merge_kwargs(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        close, order_size, size_type, order_price, fees, fixed_fees, slippage, \
            min_size, reject_prob, val_price = broadcast(
                close, order_size, size_type, order_price, fees, fixed_fees, slippage,
                min_size, reject_prob, val_price, **broadcast_kwargs, keep_raw=keep_raw)
        target_shape = (close.shape[0], close.shape[1] if close.ndim > 1 else 1)
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape, group_counts, call_seq_type=call_seq)

        # Perform calculation
        order_records = nb.simulate_from_orders_nb(
            target_shape,
            cs_group_counts,  # group only if cash sharing is enabled to speed up
            init_cash,
            call_seq,
            order_size,
            size_type,
            order_price,
            fees,
            fixed_fees,
            slippage,
            min_size,
            reject_prob,
            val_price,
            dynamic_call_seq,
            close.ndim == 2
        )

        # Create an instance
        orders = Orders(wrapper, order_records, close)
        return cls(
            orders,
            init_cash,
            cash_sharing,
            call_seq,
            **kwargs
        )

    @classmethod
    def from_order_func(cls, close, order_func_nb, *order_args, target_shape=None, keys=None,
                        init_cash=None, cash_sharing=None, call_seq=None, active_mask=None, prep_func_nb=None,
                        prep_args=None, group_prep_func_nb=None, group_prep_args=None, row_prep_func_nb=None,
                        row_prep_args=None, segment_prep_func_nb=None, segment_prep_args=None, row_wise=None,
                        seed=None, freq=None, group_by=None, broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Build portfolio from a custom order function.

        For details, see `vectorbt.portfolio.nb.simulate_nb`.

        if `row_wise` is `True`, also see `vectorbt.portfolio.nb.simulate_row_wise_nb`.

        Args:
            close (pandas_like): Reference price, such as close. Will broadcast to `target_shape`.

                Will be used for calculating unrealized P&L and portfolio value.

                Previous `close` will also be used for valuating assets/groups during the simulation.
            order_func_nb (function): Order generation function.
            *order_args: Arguments passed to `order_func_nb`.
            target_shape (tuple): Target shape to iterate over. Defaults to `close.shape`.
            keys (sequence): Outermost column level.

                Each element should correspond to one iteration over columns in `close`.
                Should be set only if `target_shape` is bigger than `close.shape`.
            init_cash (float or array_like): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
            cash_sharing (bool): Whether to share cash within the same group.

                !!! warning
                    Order execution cannot be considered parallel anymore.
            call_seq (int or array_like): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                * Set to array to specify custom sequence. Will not broadcast.
            active_mask (bool or array_like): Mask of whether a particular segment should be executed.

                By default, will broadcast to the number of rows and groups.
            prep_func_nb (function): Simulation preparation function.
            prep_args (tuple): Packed arguments passed to `prep_func_nb`.

                Defaults to `()`.
            group_prep_func_nb (function): Group preparation function.

                Called only if `row_wise` is `False`.
            group_prep_args (tuple): Packed arguments passed to `group_prep_func_nb`.

                Defaults to `()`.
            row_prep_func_nb (function): Row preparation function.

                Called only if `row_wise` is `True`.
            row_prep_args (tuple): Packed arguments passed to `row_prep_func_nb`.

                Defaults to `()`.
            segment_prep_func_nb (function): Segment preparation function.
            segment_prep_args (tuple): Packed arguments passed to `segment_prep_func_nb`.

                Defaults to `()`.
            row_wise (bool): Whether to iterate over rows rather than columns/groups.

                See `vectorbt.portfolio.nb.simulate_row_wise_nb`.
            seed (int): Seed to be set for both `call_seq` and at the beginning of the simulation.
            freq (any): Index frequency in case `close.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.
            **kwargs: Keyword arguments passed to the `__init__` method.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            All passed functions should be Numba-compiled.

            Objects passed as arguments to both functions will not broadcast to `target_shape`
            as their purpose is unknown. You should broadcast manually or use flexible indexing.

            Also see notes on `Portfolio.from_orders`.
        """
        # Get defaults
        if target_shape is None:
            target_shape = close.shape
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
            if isinstance(call_seq, str):
                call_seq = getattr(CallSeqType, call_seq)
        if active_mask is None:
            active_mask = True
        if row_wise is None:
            row_wise = defaults.portfolio['row_wise']
        if seed is None:
            seed = defaults.portfolio['seed']
        if seed is not None:
            set_seed(seed)
        if freq is None:
            freq = defaults.portfolio['freq']
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        require_kwargs = dict(require_kwargs=dict(requirements='W'))
        broadcast_kwargs = merge_kwargs(require_kwargs, broadcast_kwargs)
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Perform checks
        checks.assert_type(close, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(close, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(call_seq, np.integer)

        # Broadcast inputs
        target_shape = (target_shape[0], target_shape[1] if len(target_shape) > 1 else 1)
        if close.shape != target_shape:
            if len(close.vbt.columns) < target_shape[1]:
                if target_shape[1] % len(close.vbt.columns) != 0:
                    raise ValueError("Cannot broadcast close to target_shape")
                if keys is None:
                    keys = pd.Index(np.arange(target_shape[1]), name='iteration_idx')
                tile_times = target_shape[1] // len(close.vbt.columns)
                close = close.vbt.tile(tile_times, keys=keys)
        close = broadcast(close, to_shape=target_shape, **broadcast_kwargs)
        wrapper = ArrayWrapper.from_obj(close, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        active_mask = broadcast(
            active_mask,
            to_shape=(target_shape[0], len(group_counts)),
            to_pd=False,
            **require_kwargs
        )
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(call_seq, to_shape=target_shape, to_pd=False))
        else:
            call_seq = nb.build_call_seq(target_shape, group_counts, call_seq_type=call_seq)

        # Prepare arguments
        if prep_func_nb is None:
            prep_func_nb = nb.empty_prep_nb
        if prep_args is None:
            prep_args = ()
        if group_prep_func_nb is None:
            group_prep_func_nb = nb.empty_prep_nb
        if group_prep_args is None:
            group_prep_args = ()
        if row_prep_func_nb is None:
            row_prep_func_nb = nb.empty_prep_nb
        if row_prep_args is None:
            row_prep_args = ()
        if segment_prep_func_nb is None:
            segment_prep_func_nb = nb.empty_prep_nb
        if segment_prep_args is None:
            segment_prep_args = ()

        prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in prep_args])
        group_prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in group_prep_args])
        row_prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in row_prep_args])
        segment_prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in segment_prep_args])
        order_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in order_args])

        # Perform calculation
        if row_wise:
            order_records = nb.simulate_row_wise_nb(
                target_shape,
                close.values,
                group_counts,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
                prep_func_nb,
                prep_args,
                row_prep_func_nb,
                row_prep_args,
                segment_prep_func_nb,
                segment_prep_args,
                order_func_nb,
                *order_args
            )
        else:
            order_records = nb.simulate_nb(
                target_shape,
                close.values,
                group_counts,
                init_cash,
                cash_sharing,
                call_seq,
                active_mask,
                prep_func_nb,
                prep_args,
                group_prep_func_nb,
                group_prep_args,
                segment_prep_func_nb,
                segment_prep_args,
                order_func_nb,
                *order_args
            )

        # Create an instance
        orders = Orders(wrapper, order_records, close)
        return cls(
            orders,
            init_cash,
            cash_sharing,
            call_seq,
            **kwargs
        )

    # ############# Properties ############# #

    @property
    def wrapper(self):
        """Array wrapper."""
        # Wrapper in orders and here can be different
        wrapper = self.orders.wrapper
        if self.cash_sharing and wrapper.grouper.allow_change:
            # Cannot change groups if columns within them are dependent
            return wrapper.copy(allow_change=False)
        return wrapper.copy()

    @property
    def cash_sharing(self):
        """Whether to share cash within the same group."""
        return self._cash_sharing

    @property
    def call_seq(self):
        """Sequence of calls per row and group."""
        return self.wrapper.wrap(self._call_seq, group_by=False)

    @property
    def incl_unrealized(self):
        """Whether to include unrealized trade P&L in statistics."""
        return self._incl_unrealized

    # ############# Regrouping ############# #

    def regroup(self, group_by=None):
        """Regroup this object."""
        if group_by is None or group_by is True:
            return self
        self.wrapper.grouper.check_group_by(group_by=group_by)
        return self.copy(orders=self.orders.regroup(group_by=group_by))

    # ############# Reference price ############# #

    @property
    def close(self):
        """Price per share series."""
        return self._ref_price

    @cached_method
    def fill_ref_price(self, ffill=True, bfill=True):
        """Fill NaN values of `Portfolio.close`.

        Use `ffill` and `bfill` to fill forwards and backwards respectively."""
        close = to_2d(self.close, raw=True)
        if ffill and np.any(np.isnan(close[-1, :])):
            close = generic_nb.ffill_nb(close)
        if bfill and np.any(np.isnan(close[0, :])):
            close = generic_nb.ffill_nb(close[::-1, :])[::-1, :]
        return self.wrapper.wrap(close, group_by=False)

    # ############# Cash ############# #

    @property
    def init_cash(self):
        """Initial amount of cash per column/group.

        Returns value per group if `cash_sharing` is `True`."""
        return self.wrapper.wrap_reduced(self._init_cash, group_by=self.cash_sharing)

    @cached_method
    def init_cash_regrouped(self, group_by=None):
        """Get cash flow series per column/group."""
        init_cash = to_1d(self.init_cash, raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            init_cash_regrouped = nb.init_cash_grouped_nb(init_cash, group_counts, self.cash_sharing)
        else:
            group_counts = self.wrapper.grouper.get_group_counts()
            init_cash_regrouped = nb.init_cash_ungrouped_nb(init_cash, group_counts, self.cash_sharing)
        return self.wrapper.wrap_reduced(init_cash_regrouped, group_by=group_by)

    @cached_method
    def cash_flow(self, group_by=None):
        """Get cash flow series per column/group."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            cash_flow_ungrouped = to_2d(self.cash_flow(group_by=False), raw=True)
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            cash_flow = nb.cash_flow_grouped_nb(cash_flow_ungrouped, group_counts)
        else:
            cash_flow = nb.cash_flow_ungrouped_nb(self.wrapper.shape_2d, self.orders.records_arr)
        return self.wrapper.wrap(cash_flow, group_by=group_by)

    @cached_method
    def cash(self, group_by=None, in_sim_order=False):
        """Get cash series per column/group."""
        if in_sim_order and not self.cash_sharing:
            raise ValueError("in_sim_order requires enabled cash sharing")

        cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            init_cash_grouped = to_1d(self.init_cash_regrouped(group_by=group_by), raw=True)
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            cash = nb.cash_grouped_nb(
                self.wrapper.shape_2d,
                cash_flow,
                group_counts,
                init_cash_grouped
            )
        else:
            init_cash = to_1d(self.init_cash, raw=True)
            group_counts = self.wrapper.grouper.get_group_counts()
            call_seq = to_2d(self.call_seq, raw=True)
            cash = nb.cash_ungrouped_nb(
                cash_flow,
                group_counts,
                init_cash,
                self.cash_sharing,
                call_seq,
                in_sim_order
            )
        return self.wrapper.wrap(cash, group_by=group_by)

    # ############# Shares ############# #

    @cached_method
    def share_flow(self):
        """Get share flow series per column."""
        share_flow = nb.share_flow_nb(self.wrapper.shape_2d, self.orders.records_arr)
        return self.wrapper.wrap(share_flow, group_by=False)

    @cached_method
    def shares(self):
        """Get share series per column."""
        share_flow = to_2d(self.share_flow(), raw=True)
        shares = nb.shares_nb(share_flow)
        return self.wrapper.wrap(shares, group_by=False)

    # ############# Records ############# #

    @property
    def orders(self):
        """Order records.

        See `vectorbt.records.orders.Orders`."""
        return self._orders

    @cached_method
    def orders_regrouped(self, group_by=None):
        """Regroup order records.

        See `vectorbt.records.orders.Orders`."""
        return self._orders.regroup(group_by=group_by)

    @cached_method
    def trades(self, group_by=None, incl_unrealized=None):
        """Get trade records.

        See `vectorbt.records.events.Trades`."""
        trades = Trades.from_orders(self.orders_regrouped(group_by=group_by))
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized
        if incl_unrealized:
            return trades
        return trades.closed

    @cached_method
    def positions(self, group_by=None, incl_unrealized=None):
        """Get position records.

        See `vectorbt.records.events.Positions`."""
        positions = Positions.from_orders(self.orders_regrouped(group_by=group_by))
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized
        if incl_unrealized:
            return positions
        return positions.closed

    @cached_method
    def drawdowns(self, **kwargs):
        """Get drawdown records.

        See `vectorbt.records.drawdowns.Drawdowns`.

        Keyword arguments are passed to `Portfolio.value`."""
        return Drawdowns.from_ts(self.value(**kwargs), freq=self.wrapper.freq)

    # ############# Performance ############# #

    @cached_method
    def holding_value(self, group_by=None):
        """Get holding value series per column/group."""
        close = to_2d(self.close, raw=True).copy()
        shares = to_2d(self.shares(), raw=True)
        close[shares == 0.] = 0.  # for price being NaN
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            holding_value = nb.holding_value_grouped_nb(close, shares, group_counts)
        else:
            holding_value = nb.holding_value_ungrouped_nb(close, shares)
        return self.wrapper.wrap(holding_value, group_by=group_by)

    @cached_method
    def value(self, group_by=None, in_sim_order=False):
        """Get portfolio value series per column/group.

        By default, will generate portfolio value for each asset based on cash flows and thus
        independent from other assets, with initial cash and shares being that of the entire group.
        Useful for generating returns and comparing assets within the same group.

        When `group_by` is `False` and `in_sim_order` is `True`, returns value generated in
        simulation order (see [row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
        This value cannot be used for generating returns as-is. Useful to analyze how value
        evolved throughout simulation."""
        cash = to_2d(self.cash(group_by=group_by, in_sim_order=in_sim_order), raw=True)
        holding_value = to_2d(self.holding_value(group_by=group_by), raw=True)
        if self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and in_sim_order:
            group_counts = self.wrapper.grouper.get_group_counts()
            call_seq = to_2d(self.call_seq, raw=True)
            value = nb.value_in_sim_order_nb(cash, holding_value, group_counts, call_seq)
            # price of NaN is already addressed by ungrouped_value_nb
        else:
            value = nb.value_nb(cash, holding_value)
        return self.wrapper.wrap(value, group_by=group_by)

    @cached_method
    def final_value(self, group_by=None, **kwargs):
        """Get final portfolio value per column/group.

        For keyword arguments, see `Portfolio.value`."""
        value = to_2d(self.value(group_by=group_by, **kwargs), raw=True)
        final_value = nb.final_value_nb(value)
        return self.wrapper.wrap_reduced(final_value, group_by=group_by)

    @cached_method
    def total_profit(self, group_by=None):
        """Get total profit per column/group."""
        init_cash_regrouped = to_1d(self.init_cash_regrouped(group_by=group_by), raw=True)
        final_value = to_1d(self.final_value(group_by=group_by, in_sim_order=False), raw=True)
        total_profit = nb.total_profit_nb(init_cash_regrouped, final_value)
        return self.wrapper.wrap_reduced(total_profit, group_by=group_by)

    @cached_method
    def active_returns(self, group_by=None):
        """Get active return series per column/group.

        This type of returns is based solely on cash flows and holding value rather than portfolio value.
        It ignores passive cash and thus it will return the same numbers irrespective of the amount of
        cash currently available, even `np.inf`. The scale of returns is comparable to that of going
        all in and keeping available cash at zero."""
        cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
        holding_value = to_2d(self.holding_value(group_by=group_by), raw=True)
        active_returns = nb.active_returns_nb(cash_flow, holding_value)
        return self.wrapper.wrap(active_returns, group_by=group_by)

    @cached_method
    def returns(self, group_by=None, in_sim_order=False):
        """Get portfolio return series per column/group."""
        value = to_2d(self.value(group_by=group_by, in_sim_order=in_sim_order), raw=True)
        if self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and in_sim_order:
            group_counts = self.wrapper.grouper.get_group_counts()
            init_cash = to_1d(self.init_cash, raw=True)
            call_seq = to_2d(self.call_seq, raw=True)
            returns = nb.returns_in_sim_order_nb(value, group_counts, init_cash, call_seq)
        else:
            init_cash_regrouped = to_1d(self.init_cash_regrouped(group_by=group_by), raw=True)
            returns = nb.returns_nb(value, init_cash_regrouped)
        return self.wrapper.wrap(returns, group_by=group_by)

    @cached_method
    def buy_and_hold_return(self, group_by=None):
        """Get total return of buy-and-hold.

        If grouped, invests same amount of cash into each asset and returns the total
        return of the entire group.

        !!! note
            Does not take into account fees and slippage. For this, create a separate portfolio."""
        ref_price_filled = to_2d(self.fill_ref_price(), raw=True)
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            total_return = nb.buy_and_hold_return_grouped_nb(ref_price_filled, group_counts)
        else:
            total_return = nb.buy_and_hold_return_ungrouped_nb(ref_price_filled)
        return self.wrapper.wrap_reduced(total_return, group_by=group_by)

    @cached_method
    def stats(self, column=None, group_by=None, incl_unrealized=None, **returns_kwargs):
        """Compute various statistics on this portfolio.

        `returns_kwargs` will be passed to `Portfolio.returns`."""
        if self.wrapper.grouper.is_grouped(group_by=group_by):
            groups = self.wrapper.grouper.get_columns(group_by=group_by)
            if len(groups) > 1 and column is None:
                raise TypeError("Select a group first. Use indexing or column argument.")
            if column is None:
                column = groups[0]
        elif self.wrapper.grouper.is_grouping_disabled(group_by=group_by):
            if self.wrapper.ndim > 1 and column is None:
                raise TypeError("Select a column in the group first. Use column argument.")
            if column is None:
                column = self.wrapper.columns[0]
        else:
            if self.wrapper.ndim > 1 and column is None:
                raise TypeError("Select a column first. Use indexing or column argument.")
            if column is None:
                column = self.wrapper.columns[0]

        def _select_col(obj):
            if checks.is_series(obj):
                return obj[column]
            return obj

        positions = self.positions(group_by=group_by, incl_unrealized=incl_unrealized)
        trades = self.trades(group_by=group_by, incl_unrealized=incl_unrealized)
        drawdowns = self.drawdowns(group_by=group_by)

        return pd.Series({
            'Start': self.wrapper.index[0],
            'End': self.wrapper.index[-1],
            'Duration': self.wrapper.shape[0] * self.wrapper.freq,
            'Holding Duration [%]': _select_col(positions.coverage() * 100),
            'Total Profit': _select_col(self.total_profit(group_by=group_by)),
            'Total Return [%]': _select_col(self.total_return(group_by=group_by) * 100),
            'Buy & Hold Return [%]': _select_col(self.buy_and_hold_return(group_by=group_by) * 100),
            'Max. Drawdown [%]': _select_col(-drawdowns.max_drawdown() * 100),
            'Avg. Drawdown [%]': _select_col(-drawdowns.avg_drawdown() * 100),
            'Max. Drawdown Duration': _select_col(drawdowns.max_duration()),
            'Avg. Drawdown Duration': _select_col(drawdowns.avg_duration()),
            'Num. Trades': _select_col(trades.count()),
            'Win Rate [%]': _select_col(trades.win_rate() * 100),
            'Best Trade [%]': _select_col(trades.returns.max() * 100),
            'Worst Trade [%]': _select_col(trades.returns.min() * 100),
            'Avg. Trade [%]': _select_col(trades.returns.mean() * 100),
            'Max. Trade Duration': _select_col(trades.duration.max(time_units=True)),
            'Avg. Trade Duration': _select_col(trades.duration.mean(time_units=True)),
            'Expectancy': _select_col(trades.expectancy()),
            'SQN': _select_col(trades.sqn()),
            'Sharpe Ratio': _select_col(self.sharpe_ratio(group_by=group_by, **returns_kwargs)),
            'Sortino Ratio': _select_col(self.sortino_ratio(group_by=group_by, **returns_kwargs)),
            'Calmar Ratio': _select_col(self.calmar_ratio(group_by=group_by, **returns_kwargs))
        }, name=column)


