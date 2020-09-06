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
        new_call_seq = obj.call_seq.values[:, col_idxs]
    else:
        # Grouping enabled
        new_init_cash = obj.init_cash.values[group_idxs if obj.cash_sharing else col_idxs]
        new_call_seq = obj.call_seq.values[:, group_idxs]

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
                    returns_kwargs=None,
                    **kwargs):

                if returns_kwargs is None:
                    returns_kwargs = {}
                returns_acc = self.returns(group_by=group_by, **returns_kwargs) \
                    .vbt.returns(freq=self.wrapper.freq, year_freq=year_freq)
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
        self._ref_price = orders.ref_price
        self._orders = orders
        self._init_cash = init_cash
        self._cash_sharing = cash_sharing
        self._call_seq = call_seq
        self._incl_unrealized = incl_unrealized

        # Supercharge
        PandasIndexer.__init__(self, _indexing_func)

    # ############# Class methods ############# #

    @classmethod
    def from_signals(cls, ref_price, entries, exits, size=np.inf, size_type=None, entry_price=None,
                     exit_price=None, init_cash=None, cash_sharing=None, call_seq=None, fees=None,
                     fixed_fees=None, slippage=None, accumulate=None, accumulate_exit_mode=None,
                     conflict_mode=None, freq=None, group_by=None, broadcast_kwargs=None,
                     wrapper_kwargs=None, **kwargs):
        """Simulate portfolio from entry and exit signals.

        Starting with initial cash `init_cash`, for each signal in `entries`, enters a position
        by buying `size` of shares for `entry_price`. For each signal in `exits`, closes the position
        by selling all shares for `exit_price`. When accumulation is enabled, each entry signal will
        increase the position, and optionally each exit signal will decrease the position. When both
        entry and exit signals are present, ignores them by default. When grouping is enabled with
        `group_by`, will compute performance for the entire group. When, additionally, `cash_sharing`
        is enabled, will share the cash among all columns in the group.

        Args:
            ref_price (pandas_like): Reference price, such as close. Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            entries (array_like): Boolean array of entry signals. Will broadcast.
            exits (array_like): Boolean array of exit signals. Will broadcast.
            size (float or array_like): Size to order. Will broadcast.

                * Set to positive/negative to buy/sell.
                * Set to `np.inf`/`-np.inf` to buy/sell everything.
                * Set to `np.nan` or zero to skip.
            size_type (int or array_like): See `vectorbt.portfolio.enums.SizeType`.

                Only `SizeType.Shares` and `SizeType.Cash` are supported.
            entry_price (array_like): Entry price. Defaults to `ref_price`. Will broadcast.
            exit_price (array_like): Exit price. Defaults to `ref_price`. Will broadcast.
            init_cash (float or array_like): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
            cash_sharing (bool): Whether to share cash within the same group.
            call_seq (int or array_like): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                    Any other integer will become a seed for a random sequence.
                * Set to array to specify custom sequence. Will not broadcast.
            fees (float or array_like): Fees in percentage of the order value. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price. Will broadcast.
            accumulate (bool): If `accumulate` is `True`, entering the market when already
                in the market will be allowed to increase the position.
            accumulate_exit_mode: See `vectorbt.portfolio.enums.AccumulateExitMode`.
            conflict_mode: See `vectorbt.portfolio.enums.ConflictMode`.
            freq (any): Index frequency in case `ref_price.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.
            **kwargs: Keyword arguments passed to the `__init__` method.

        All time series will be broadcast together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            Only `SizeType.Shares` and `SizeType.Cash` are supported. Other modes such as target
            percentage are not compatible with signals since their logic may contradict the direction
            the user has specified for the order.

            With cash sharing enabled, at each timestamp, processing of the assets in a group
            goes strictly in order defined in `call_seq`. This order can't be changed dynamically.
        """
        # Get defaults
        if entry_price is None:
            entry_price = ref_price
        if exit_price is None:
            exit_price = ref_price
        if size is None:
            size = defaults.portfolio['size']
        if size_type is None:
            size_type = defaults.portfolio['size_type']
            if isinstance(size_type, str):
                size_type = getattr(SizeType, size_type)
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
            if isinstance(call_seq, str):
                call_seq = getattr(CallSeqType, call_seq)
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
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
        if freq is None:
            freq = defaults.portfolio['freq']
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Perform checks
        checks.assert_type(ref_price, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(ref_price, np.floating)
        checks.assert_subdtype(size, np.floating)
        checks.assert_subdtype(entry_price, np.floating)
        checks.assert_subdtype(exit_price, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(fees, np.floating)
        checks.assert_subdtype(fixed_fees, np.floating)
        checks.assert_subdtype(slippage, np.floating)
        checks.assert_dtype(entries, np.bool)
        checks.assert_dtype(exits, np.bool)
        checks.assert_subdtype(call_seq, np.integer)

        # Broadcast inputs
        # Only ref_price is broadcast, others can remain unchanged thanks to flexible indexing
        keep_raw = (False, True, True, True, True, True, True, True, True, True)
        broadcast_kwargs = merge_kwargs(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        ref_price, entries, exits, size, size_type, entry_price, \
            exit_price, fees, fixed_fees, slippage = broadcast(
                ref_price, entries, exits, size, size_type, entry_price, exit_price, fees,
                fixed_fees, slippage, **broadcast_kwargs, keep_raw=keep_raw)
        target_shape = (ref_price.shape[0], ref_price.shape[1] if ref_price.ndim > 1 else 1)
        wrapper = ArrayWrapper.from_obj(ref_price, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(
                call_seq,
                to_shape=target_shape,
                to_pd=False
            ))
        else:
            if call_seq < len(CallSeqType):
                call_seq = nb.build_call_seq(
                    target_shape,
                    group_counts,
                    call_seq_type=call_seq
                )
            else:
                call_seq = nb.build_call_seq(
                    target_shape,
                    group_counts,
                    call_seq_type=CallSeqType.Random,
                    seed=call_seq
                )

        # Perform calculation
        order_records = nb.simulate_from_signals_nb(
            target_shape,
            cs_group_counts,  # grouping without cash sharing has no effect
            init_cash,
            cash_sharing,
            call_seq,
            entries,
            exits,
            size,
            size_type,
            entry_price,
            exit_price,
            fees,
            fixed_fees,
            slippage,
            accumulate,
            accumulate_exit_mode,
            conflict_mode,
            ref_price.ndim == 2
        )

        # Create an instance
        orders = Orders(wrapper, order_records, ref_price)
        return cls(
            orders,
            init_cash,
            cash_sharing,
            call_seq,
            **kwargs
        )

    @classmethod
    def from_orders(cls, ref_price, order_size, size_type=None, order_price=None, init_cash=None,
                    cash_sharing=None, call_seq=None, fees=None, fixed_fees=None, slippage=None,
                    freq=None, group_by=None, broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Simulate portfolio from orders.

        Starting with initial cash `init_cash`, orders the number of shares specified in `order_size` 
        for `order_price`.

        Args:
            ref_price (pandas_like): Reference price, such as close. Will broadcast.

                Will be used for calculating unrealized P&L and portfolio value.
            order_size (float or array_like): Size to order. Will broadcast.

                For any size type:

                * Set to `np.nan` to skip.
                * Set to `np.inf`/`-np.inf` to buy/sell everything.

                For `SizeType.Shares` and `SizeType.Cash`:

                * Set to positive/negative to buy/sell.
                * Set to zero to skip.

                For target size, the final size will depend upon current holdings.
            size_type (int or array_like): See `vectorbt.portfolio.enums.SizeType`.
            order_price (array_like): Order price. Defaults to `ref_price`. Will broadcast.
            init_cash (float or array_like): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
            cash_sharing (bool): Whether to share cash within the same group.
            call_seq (int or array_like): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                    Any other integer will become a seed for a random sequence.
                * Set to array to specify custom sequence. Will not broadcast.
            fees (float or array_like): Fees in percentage of the order value. Will broadcast.
            fixed_fees (float or array_like): Fixed amount of fees to pay per order. Will broadcast.
            slippage (float or array_like): Slippage in percentage of price. Will broadcast.
            freq (any): Index frequency in case `ref_price.index` is not datetime-like.
            group_by (any): Group columns. See `vectorbt.base.column_grouper.ColumnGrouper`.
            broadcast_kwargs (dict): Keyword arguments passed to `vectorbt.base.reshape_fns.broadcast`.
            wrapper_kwargs (dict): Keyword arguments passed to `vectorbt.base.array_wrapper.ArrayWrapper`.

        All time series will be broadcast together using `vectorbt.base.reshape_fns.broadcast`.
        At the end, they will have the same metadata.

        For defaults, see `vectorbt.defaults.portfolio`.

        !!! note
            With cash sharing enabled, at each timestamp, processing of the assets in a group
            goes strictly in order defined in `call_seq`. This order can't be changed dynamically.

            This has one big implication for this particular method: the last asset in the call stack
            cannot be processed until other assets are processed. This is the reason why `SizeType.TargetPercent`
            cannot work properly in this setting: one has to specify percentages for all assets
            beforehand and then tweak the processing order to sell low-performing assets first in order
            to release funds for high-performing assets. That's why `SizeType.TargetPercent` doesn't mean
            here the percentage of the current group value, but the percentage of the value the
            current asset has access to, which is the available cash at this point + the holding
            value of this particular asset. The holding values of other assets are not taken into account
            as they are not processed yet and thus their holdings are frozen until their turn comes.

            Hence, `SizeType.TargetPercent` should be used with caution.
        """
        # Get defaults
        if order_price is None:
            order_price = ref_price
        if order_size is None:
            order_size = defaults.portfolio['order_size']
        if size_type is None:
            size_type = defaults.portfolio['size_type']
            if isinstance(size_type, str):
                size_type = getattr(SizeType, size_type)
        if init_cash is None:
            init_cash = defaults.portfolio['init_cash']
        if cash_sharing is None:
            cash_sharing = defaults.portfolio['cash_sharing']
        if call_seq is None:
            call_seq = defaults.portfolio['call_seq']
            if isinstance(call_seq, str):
                call_seq = getattr(CallSeqType, call_seq)
        if fees is None:
            fees = defaults.portfolio['fees']
        if fixed_fees is None:
            fixed_fees = defaults.portfolio['fixed_fees']
        if slippage is None:
            slippage = defaults.portfolio['slippage']
        if freq is None:
            freq = defaults.portfolio['freq']
        if broadcast_kwargs is None:
            broadcast_kwargs = {}
        if wrapper_kwargs is None:
            wrapper_kwargs = {}
        if not wrapper_kwargs.get('group_select', True) and cash_sharing:
            raise ValueError("group_select cannot be disabled if cash_sharing=True")

        # Perform checks
        checks.assert_type(ref_price, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(ref_price, np.floating)
        checks.assert_subdtype(order_size, np.floating)
        checks.assert_subdtype(order_price, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(fees, np.floating)
        checks.assert_subdtype(fixed_fees, np.floating)
        checks.assert_subdtype(slippage, np.floating)
        checks.assert_subdtype(call_seq, np.integer)

        # Broadcast inputs
        # Only ref_price is broadcast, others can remain unchanged thanks to flexible indexing
        keep_raw = (False, True, True, True, True, True, True)
        broadcast_kwargs = merge_kwargs(dict(require_kwargs=dict(requirements='W')), broadcast_kwargs)
        ref_price, order_size, size_type, order_price, fees, fixed_fees, slippage = broadcast(
            ref_price, order_size, size_type, order_price, fees,
            fixed_fees, slippage, **broadcast_kwargs, keep_raw=keep_raw)
        target_shape = (ref_price.shape[0], ref_price.shape[1] if ref_price.ndim > 1 else 1)
        wrapper = ArrayWrapper.from_obj(ref_price, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(
                call_seq,
                to_shape=target_shape,
                to_pd=False
            ))
        else:
            if call_seq < len(CallSeqType):
                call_seq = nb.build_call_seq(
                    target_shape,
                    group_counts,
                    call_seq_type=call_seq
                )
            else:
                call_seq = nb.build_call_seq(
                    target_shape,
                    group_counts,
                    call_seq_type=CallSeqType.Random,
                    seed=call_seq
                )

        # Perform calculation
        order_records = nb.simulate_from_orders_nb(
            target_shape,
            cs_group_counts,  # grouping without cash sharing has no effect
            init_cash,
            cash_sharing,
            call_seq,
            order_size,
            size_type,
            order_price,
            fees,
            fixed_fees,
            slippage,
            ref_price.ndim == 2
        )

        # Create an instance
        orders = Orders(wrapper, order_records, ref_price)
        return cls(
            orders,
            init_cash,
            cash_sharing,
            call_seq,
            **kwargs
        )

    @classmethod
    def from_order_func(cls, ref_price, order_func_nb, *order_args, target_shape=None, keys=None,
                        init_cash=None, cash_sharing=None, call_seq=None, active_mask=None, row_prep_func_nb=None,
                        row_prep_args=None, call_seq_func_nb=None, call_seq_args=None, row_wise=None, freq=None,
                        group_by=None, broadcast_kwargs=None, wrapper_kwargs=None, **kwargs):
        """Build portfolio from a custom order function.

        For details, see `vectorbt.portfolio.nb.simulate_nb`.

        if `row_wise` is `True`, also see `vectorbt.portfolio.nb.simulate_row_wise_nb`.

        Args:
            ref_price (pandas_like): Reference price, such as close. Will broadcast to `target_shape`.

                Will be used for calculating unrealized P&L and portfolio value.
            order_func_nb (function): Order generation function.
            *order_args: Arguments passed to `order_func_nb`.
            target_shape (tuple): Target shape to iterate over. Defaults to `ref_price.shape`.
            keys (sequence): Outermost column level.

                Each element should correspond to one iteration over columns in `ref_price`.
                Should be set only if `target_shape` is bigger than `ref_price.shape`.
            init_cash (float or array_like): Initial capital.

                By default, will broadcast to the number of columns.
                If cash sharing is enabled, will broadcast to the number of groups.
            cash_sharing (bool): Whether to share cash within the same group.
            call_seq (int or array_like): Default sequence of calls per row and group.

                * Use `vectorbt.portfolio.enums.CallSeqType` to select a sequence type.
                    Any other integer will become a seed for a random sequence.
                * Set to array to specify custom sequence. Will not broadcast.
            active_mask (bool or array_like): Mask of whether a particular row should be executed.

                By default, will broadcast to the number of rows and groups (2-dim array).
                If row-wise, will broadcast to the number of rows (1-dim array).
            row_prep_func_nb (function): Row preparation function.
            row_prep_args (tuple): Tuple of arguments passed to `row_prep_func_nb`.

                Defaults to `order_args`. To pass nothing, set to `()`.
            call_seq_func_nb (function): Call sequence generation function.
            call_seq_args (tuple): Tuple of arguments passed to `call_seq_func_nb`.

                Defaults to `row_prep_args`. To pass nothing, set to `()`.
            row_wise (bool): Whether to iterate over rows rather than columns/groups.

                See `vectorbt.portfolio.nb.simulate_row_wise_nb`.
            freq (any): Index frequency in case `ref_price.index` is not datetime-like.
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
            target_shape = ref_price.shape
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
        checks.assert_type(ref_price, (pd.Series, pd.DataFrame))
        checks.assert_subdtype(ref_price, np.floating)
        checks.assert_subdtype(init_cash, np.floating)
        checks.assert_subdtype(call_seq, np.integer)

        # Broadcast inputs
        target_shape = (target_shape[0], target_shape[1] if len(target_shape) > 1 else 1)
        if ref_price.shape != target_shape:
            if len(ref_price.vbt.columns) < target_shape[1]:
                if target_shape[1] % len(ref_price.vbt.columns) != 0:
                    raise ValueError("Cannot broadcast ref_price to target_shape")
                if keys is None:
                    keys = pd.Index(np.arange(target_shape[1]), name='iteration_idx')
                tile_times = target_shape[1] // len(ref_price.vbt.columns)
                ref_price = ref_price.vbt.tile(tile_times, keys=keys)
            ref_price = broadcast(ref_price, to_shape=target_shape, **broadcast_kwargs)
        wrapper = ArrayWrapper.from_obj(ref_price, freq=freq, group_by=group_by, **wrapper_kwargs)
        cs_group_counts = wrapper.grouper.get_group_counts(group_by=cash_sharing)
        init_cash = np.broadcast_to(init_cash, (len(cs_group_counts),))
        group_counts = wrapper.grouper.get_group_counts(group_by=group_by)
        if row_wise:
            active_mask = broadcast(
                active_mask,
                to_shape=(target_shape[0],),
                to_pd=False,
                **require_kwargs
            )
        else:
            active_mask = broadcast(
                active_mask,
                to_shape=(target_shape[0], len(group_counts)),
                to_pd=False,
                **require_kwargs
            )
        if checks.is_array(call_seq):
            call_seq = nb.require_call_seq(broadcast(
                call_seq,
                to_shape=target_shape,
                to_pd=False
            ))
        else:
            if call_seq < len(CallSeqType):
                call_seq = nb.build_call_seq(
                    target_shape,
                    group_counts,
                    call_seq_type=call_seq
                )
            else:
                call_seq = nb.build_call_seq(
                    target_shape,
                    group_counts,
                    call_seq_type=CallSeqType.Random,
                    seed=call_seq
                )

        # Resolve functions and arguments
        order_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in order_args])
        if row_prep_func_nb is None:
            row_prep_func_nb = nb.empty_row_prep_nb
        if row_prep_args is None:
            row_prep_args = order_args
        else:
            row_prep_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in row_prep_args])
        if call_seq_func_nb is None:
            call_seq_func_nb = nb.default_call_seq_nb
        if call_seq_args is None:
            call_seq_args = row_prep_args
        else:
            call_seq_args = tuple([arg.values if checks.is_pandas(arg) else arg for arg in call_seq_args])

        # Perform calculation
        if row_wise:
            func = nb.simulate_row_wise_nb
        else:
            func = nb.simulate_nb
        order_records = func(
            target_shape,
            group_counts,
            init_cash,
            cash_sharing,
            call_seq,
            active_mask,
            row_prep_func_nb,
            row_prep_args,
            call_seq_func_nb,
            call_seq_args,
            order_func_nb,
            *order_args
        )

        # Create an instance
        orders = Orders(wrapper, order_records, ref_price)
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

    def regroup(self, group_by=None):
        """Regroup this object."""
        if group_by is None or group_by is True:
            return self
        self.wrapper.grouper.check_group_by(group_by=group_by)
        return self.copy(wrapper=self.wrapper.copy(group_by=group_by))

    @property
    def cash_sharing(self):
        """Whether to share cash within the same group."""
        return self._cash_sharing

    @property
    def incl_unrealized(self):
        """Whether to include unrealized trade P&L in statistics."""
        return self._incl_unrealized

    # ############# Call sequence ############# #

    @property
    def call_seq(self):
        """Sequence of calls per row and group."""
        return self.wrapper.wrap(self._call_seq, group_by=False)

    # ############# Cash ############# #

    @property
    def init_cash(self):
        """Initial amount of cash per column/group.

        Returns value per group if `cash_sharing` is `True`."""
        return self.wrapper.wrap_reduced(self._init_cash, group_by=self.cash_sharing)

    @cached_method
    def regroup_init_cash(self, group_by=None):
        """Get initial amount of cash based on column grouping."""
        init_cash = to_1d(self.init_cash, raw=True)
        if self.cash_sharing and self.wrapper.grouper.is_grouping_disabled(group_by=group_by):
            # Un-group grouped cash series using forward fill
            init_cash_ungrouped = np.full(len(self.wrapper.columns), np.nan, dtype=np.float_)
            group_start_idxs = self.wrapper.grouper.get_group_start_idxs()
            init_cash_ungrouped[group_start_idxs] = init_cash
            mask = np.isnan(init_cash_ungrouped)
            idx = np.where(~mask, np.arange(mask.shape[0]), 0)
            np.maximum.accumulate(idx, out=idx)
            init_cash = init_cash_ungrouped[idx]
        elif not self.cash_sharing and self.wrapper.grouper.is_grouped(group_by=group_by):
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            init_cash = nb.cash_grouped_nb(init_cash[None, :], group_counts, False)[0, :]
        return self.wrapper.wrap_reduced(init_cash, group_by=group_by)

    @property
    def cash(self):
        """Final cash series per column."""
        return self.wrapper.wrap(self._cash, group_by=False)

    @cached_method
    def regroup_cash(self, group_by=None):
        """Get final cash series based on column grouping."""
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.cash
        cash = to_2d(self.cash, raw=True)
        group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
        cash = nb.cash_grouped_nb(cash, group_counts, self.cash_sharing)
        return self.wrapper.wrap(cash, group_by=group_by)

    # ############# Shares ############# #

    @property
    def shares(self):
        """Final shares series per column."""
        return self.wrapper.wrap(self._shares, group_by=False)

    # ############# Reference price ############# #

    @property
    def ref_price(self):
        """Price per share series."""
        return self._ref_price

    @cached_method
    def fill_ref_price(self, ffill=True, bfill=True):
        """Fill NaN values of `Portfolio.ref_price`.

        Use `ffill` and `bfill` to fill forwards and backwards respectively."""
        ref_price = to_2d(self.ref_price, raw=True)
        if ffill and np.any(np.isnan(ref_price[-1, :])):
            ref_price = generic_nb.ffill_nb(ref_price)
        if bfill and np.any(np.isnan(ref_price[0, :])):
            ref_price = generic_nb.ffill_nb(ref_price[::-1, :])[::-1, :]
        return self.wrapper.wrap(ref_price, group_by=False)

    # ############# Records ############# #

    @property
    def orders(self):
        """Order records.

        See `vectorbt.records.orders.Orders`."""
        return self._orders

    @cached_method
    def regroup_orders(self, group_by=None):
        """Regroup order records.

        See `vectorbt.records.orders.Orders`."""
        return self._orders.regroup(group_by=group_by)

    @cached_method
    def trades(self, group_by=None, incl_unrealized=None):
        """Get trade records.

        See `vectorbt.records.events.Trades`."""
        trades = Trades.from_orders(self.regroup_orders(group_by=group_by))
        if incl_unrealized is None:
            incl_unrealized = self.incl_unrealized
        if incl_unrealized:
            return trades
        return trades.closed

    @cached_method
    def positions(self, group_by=None, incl_unrealized=None):
        """Get position records.

        See `vectorbt.records.events.Positions`."""
        positions = Positions.from_orders(self.regroup_orders(group_by=group_by))
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
    def cash_flow(self, group_by=None):
        """Get cash flow."""
        cash_flow = nb.cash_flow_nb(self.ref_price.shape, self.orders.records_arr)
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            return self.wrapper.wrap(cash_flow, group_by=group_by)
        group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
        cash_flow_grouped = nb.cash_flow_grouped_nb(cash_flow, group_counts)
        return self.wrapper.wrap(cash_flow_grouped, group_by=group_by)

    @cached_method
    def holding_value(self, group_by=None):
        """Get holding value."""
        ref_price = to_2d(self.ref_price, raw=True).copy()
        shares = to_2d(self.shares, raw=True)
        ref_price[shares == 0.] = 0.  # for price being NaN
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            holding_value = shares * ref_price
        else:
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            holding_value = nb.grouped_holding_value_nb(ref_price, shares, group_counts)
        return self.wrapper.wrap(holding_value, group_by=group_by)

    @cached_method
    def value(self, group_by=None, iterative=False):
        """Get portfolio value.

        By default, will generate portfolio value for each asset based on cash flows and thus
        independent from other assets, with initial cash and shares being that of the entire group.
        Useful for generating returns and comparing assets within the same group.

        When `group_by` is `False` and `iterative` is `True`, returns value generated in
        simulation order (see [row-major order](https://en.wikipedia.org/wiki/Row-_and_column-major_order).
        This value cannot be used for generating returns as-is. Useful to analyze how value
        evolved throughout simulation."""
        if iterative and not self.cash_sharing:
            raise ValueError("Shared cash must be enabled for iterative mode")
        cash = to_2d(self.regroup_cash(group_by=group_by), raw=True)
        holding_value = to_2d(self.holding_value(group_by=group_by), raw=True)
        if self.wrapper.grouper.is_grouping_disabled(group_by=group_by):
            if iterative:
                group_counts = self.wrapper.grouper.get_group_counts()
                value = nb.ungrouped_iter_value_nb(cash, holding_value, group_counts)
                # price of NaN is already addressed by ungrouped_value_nb
            else:
                init_cash = to_1d(self.regroup_init_cash(group_by=False), raw=True)
                cash_flow = to_2d(self.cash_flow(group_by=False), raw=True)
                holding_value = to_2d(self.holding_value(group_by=False), raw=True)
                value = init_cash + np.cumsum(cash_flow, axis=0) + holding_value
        else:
            value = cash + holding_value
        return self.wrapper.wrap(value, group_by=group_by)

    @cached_method
    def final_value(self, group_by=None, iterative=False):
        """Get final portfolio value.

        For details on `iterative`, see `Portfolio.value`."""
        value = to_2d(self.value(group_by=group_by, iterative=iterative), raw=True)
        final_value = generic_nb.ffill_nb(value)[-1, :]
        return self.wrapper.wrap_reduced(final_value, group_by=group_by)

    @cached_method
    def total_profit(self, group_by=None):
        """Get total profit."""
        init_cash = to_1d(self.regroup_init_cash(group_by=group_by), raw=True)
        final_value = to_1d(self.final_value(group_by=group_by, iterative=False), raw=True)
        total_profit = final_value - init_cash
        return self.wrapper.wrap_reduced(total_profit, group_by=group_by)

    @cached_method
    def active_returns(self, group_by=None):
        """Get active portfolio returns.

        This type of returns is based solely on holding value and cash flows rather than portfolio value.
        It ignores passive cash and thus it will return the same numbers irrespective of the amount of
        cash currently available, even `np.inf`. The scale of returns is comparable to that of going
        all in and keeping available cash at zero."""
        holding_value = to_2d(self.holding_value(group_by=group_by), raw=True)
        cash_flow = to_2d(self.cash_flow(group_by=group_by), raw=True)
        input_value = np.empty(holding_value.shape, dtype=np.float_)
        input_value[0, :] = 0.
        input_value[1:, :] = holding_value[:-1, :]
        input_value[cash_flow < 0] += -cash_flow[cash_flow < 0]
        output_value = holding_value.copy()
        output_value[cash_flow > 0] += cash_flow[cash_flow > 0]
        np.divide(output_value - input_value, input_value, where=input_value != 0., out=input_value)
        return self.wrapper.wrap(input_value, group_by=group_by)

    @cached_method
    def returns(self, group_by=None, iterative=False, active=False):
        """Get portfolio returns.

        For more details on `iterative`, see `Portfolio.value`.
        For more details on `active`, see `Portfolio.active_returns`."""
        if active:
            returns = self.active_returns(group_by=group_by)
        else:
            init_cash = to_1d(self.regroup_init_cash(group_by=group_by), raw=True)
            value = to_2d(self.value(group_by=group_by, iterative=iterative), raw=True)
            if self.cash_sharing and self.wrapper.grouper.is_grouping_disabled(group_by=group_by) and iterative:
                group_counts = self.wrapper.grouper.get_group_counts()
                returns = nb.ungrouped_iter_returns_nb(value, init_cash, group_counts)
            else:
                returns = np.empty(value.shape, dtype=np.float_)
                returns[0, :] = value[0, :] - init_cash
                returns[1:, :] = (value[1:, :] - value[:-1, :]) / value[:-1, :]
        return self.wrapper.wrap(returns, group_by=group_by)

    @cached_method
    def buy_and_hold_return(self, group_by=None):
        """Get total return of buy-and-hold.

        If grouped, invests same amount of cash into each asset and returns the total
        return of the entire group.

        !!! note
            Does not take into account fees and slippage. For this, create a separate portfolio."""
        ref_price_filled = to_2d(self.fill_ref_price(), raw=True)
        if not self.wrapper.grouper.is_grouped(group_by=group_by):
            total_return = (ref_price_filled[-1, :] - ref_price_filled[0, :]) / ref_price_filled[0, :]
        else:
            group_counts = self.wrapper.grouper.get_group_counts(group_by=group_by)
            total_return = nb.grouped_buy_and_hold_return_nb(ref_price_filled, group_counts)
        return self.wrapper.wrap_reduced(total_return, group_by=group_by)

    @cached_method
    def stats(self, column=None, group_by=None, incl_unrealized=None, returns_kwargs=None):
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
        if returns_kwargs is None:
            returns_kwargs = {}

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


