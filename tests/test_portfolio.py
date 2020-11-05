import numpy as np
import pandas as pd
from numba import njit, typeof
from numba.typed import List
from datetime import datetime, timedelta
import pytest

import vectorbt as vbt
from vectorbt.enums import (
    SizeType,
    Direction,
    ConflictMode,
    CallSeqType,
    Order,
    NoOrder,
    OrderResult,
    InitCashMode,
    order_dt,
    trade_dt,
    position_dt,
    debug_info_dt
)
from vectorbt import defaults
from vectorbt.utils.random import set_seed
from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.portfolio import nb

from tests.utils import record_arrays_close

seed = 42

day_dt = np.timedelta64(86400000000000)

defaults.returns['year_freq'] = '252 days'  # same as empyrical

price = pd.Series([1., 2., 3., 4., 5.], index=pd.Index([
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    datetime(2020, 1, 3),
    datetime(2020, 1, 4),
    datetime(2020, 1, 5)
]))
price_wide = price.vbt.tile(3, keys=['a', 'b', 'c'])
big_price = pd.DataFrame(np.random.uniform(size=(1000,)))
big_price.index = [datetime(2018, 1, 1) + timedelta(days=i) for i in range(1000)]
big_price_wide = big_price.vbt.tile(1000)


# ############# nb ############# #

def assert_same_tuple(tup1, tup2):
    for i in range(len(tup1)):
        assert tup1[i] == tup2[i] or np.isnan(tup1[i]) and np.isnan(tup2[i])


def test_process_order_nb():
    # Errors, ignored and rejected orders
    debug_info_record = np.empty(1, dtype=debug_info_dt)[0]
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=0))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=10), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=1))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, -100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, np.nan, 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., np.inf, 10., 1100.,
            nb.create_order_nb(size=10, price=10), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., np.nan, 10., 1100.,
            nb.create_order_nb(size=10, price=10), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., np.inf, 1100.,
            nb.create_order_nb(size=10, price=10), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., -10., 1100.,
            nb.create_order_nb(size=10, price=10), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., -100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, direction=Direction.Long), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, direction=Direction.Short), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=np.inf), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=-10), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fees=np.inf), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fees=-1), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fixed_fees=np.inf), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fixed_fees=-1), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, slippage=np.inf), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, slippage=-1), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, min_size=np.inf), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, min_size=-1), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, max_size=0), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, max_size=-10), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, reject_prob=np.nan), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, reject_prob=-1), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, reject_prob=2), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., np.nan,
        nb.create_order_nb(size=1, price=10, size_type=SizeType.TargetPercent), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=3))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., np.nan, 1100.,
        nb.create_order_nb(size=10, price=10, size_type=SizeType.TargetValue), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=2))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=15, price=10, max_size=10, allow_partial=False, raise_reject=True),
            debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=15, price=10, max_size=10, allow_partial=False), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=7))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, reject_prob=1., raise_reject=True), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=10, price=10, reject_prob=1.), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=8))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, np.inf, 100., 10., 1100.,
            nb.create_order_nb(size=np.inf, price=10, direction=Direction.Long), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, np.inf, 100., 10., 1100.,
            nb.create_order_nb(size=np.inf, price=10, direction=Direction.All), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 1100.,
        nb.create_order_nb(size=-10, price=10, direction=Direction.Short), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 0.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=6))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., np.inf,
            nb.create_order_nb(size=np.inf, price=10, direction=Direction.Short), debug_info_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., np.inf,
            nb.create_order_nb(size=-np.inf, price=10, direction=Direction.All), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 1100.,
        nb.create_order_nb(size=-10, price=10, direction=Direction.Long), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 0.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=5))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fixed_fees=100, raise_reject=True), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=10, price=10, fixed_fees=100), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=9))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, min_size=100, raise_reject=True), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=10, price=10, min_size=100), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=10))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=100, price=10, allow_partial=False, raise_reject=True), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=100, price=10, allow_partial=False), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=11))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=-10, price=10, min_size=100, raise_reject=True), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=-10, price=10, min_size=100), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=10))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=-200, price=10, direction=Direction.Long, allow_partial=False,
                               raise_reject=True),
            debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=-200, price=10, direction=Direction.Long, allow_partial=False), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=11))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            0, 0, 0, 100., 100., 10., 1100.,
            nb.create_order_nb(size=-10, price=10, fixed_fees=1000, raise_reject=True), debug_info_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 100., 10., 1100.,
        nb.create_order_nb(size=-10, price=10, fixed_fees=1000), debug_info_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=9))

    # Calculations
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=10, price=10, fees=0.1, fixed_fees=1, slippage=0.1), debug_info_record)
    assert cash_now == 0.
    assert shares_now == 8.18181818181818
    assert_same_tuple(order_result, OrderResult(
        size=8.18181818181818, price=11.0, fees=10.000000000000014, side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=100, price=10, fees=0.1, fixed_fees=1, slippage=0.1), debug_info_record)
    assert cash_now == 0.
    assert shares_now == 8.18181818181818
    assert_same_tuple(order_result, OrderResult(
        size=8.18181818181818, price=11.0, fees=10.000000000000014, side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=-10, price=10, fees=0.1, fixed_fees=1, slippage=0.1), debug_info_record)
    assert cash_now == 180.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10.0, price=9.0, fees=10.0, side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=-100, price=10, fees=0.1, fixed_fees=1, slippage=0.1), debug_info_record)
    assert cash_now == 909.
    assert shares_now == -100.
    assert_same_tuple(order_result, OrderResult(
        size=100.0, price=9.0, fees=91.0, side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=10, price=10, size_type=SizeType.TargetShares), debug_info_record)
    assert cash_now == 0.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=-10, price=10, size_type=SizeType.TargetShares), debug_info_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=100, price=10, size_type=SizeType.TargetValue), debug_info_record)
    assert cash_now == 0.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=-100, price=10, size_type=SizeType.TargetValue), debug_info_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=1, price=10, size_type=SizeType.TargetPercent), debug_info_record)
    assert cash_now == 0.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=-1, price=10, size_type=SizeType.TargetPercent), debug_info_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=np.inf, price=10), debug_info_record)
    assert cash_now == 0.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=-np.inf, price=10), debug_info_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=1, status=0, status_info=-1))

    # Debugging
    _ = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(debug=True), debug_info_record)
    assert_same_tuple(debug_info_record, (
        0, 0, 0, 100., 0., 10., 100., np.nan, 0, np.nan, 0., 0., 0., 0., np.inf, 0.,
        True, False, 2, 100., 0., np.nan, np.nan, np.nan, -1, 1, 0))
    _ = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=np.inf, price=10, debug=True), debug_info_record)
    assert_same_tuple(debug_info_record, (
        0, 0, 0, 100., 0., 10., 100., np.inf, 0, 10., 0., 0., 0., 0., np.inf, 0.,
        True, False, 2, 0., 10., 10., 10., 0., 0, 0, -1))
    _ = nb.process_order_nb(
        0, 0, 0, 100., 0., 10., 100.,
        nb.create_order_nb(size=-np.inf, price=10, debug=True), debug_info_record)
    assert_same_tuple(debug_info_record, (
        0, 0, 0, 100., 0., 10., 100., -np.inf, 0, 10., 0., 0., 0., 0., np.inf, 0.,
        True, False, 2, 200., -10., 10., 10., 0., 1, 0, -1))


def test_build_call_seq_nb():
    group_lens = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(
        nb.build_call_seq_nb((10, 10), group_lens, CallSeqType.Default),
        nb.build_call_seq((10, 10), group_lens, CallSeqType.Default)
    )
    np.testing.assert_array_equal(
        nb.build_call_seq_nb((10, 10), group_lens, CallSeqType.Reversed),
        nb.build_call_seq((10, 10), group_lens, CallSeqType.Reversed)
    )
    set_seed(seed)
    out1 = nb.build_call_seq_nb((10, 10), group_lens, CallSeqType.Random)
    set_seed(seed)
    out2 = nb.build_call_seq((10, 10), group_lens, CallSeqType.Random)
    np.testing.assert_array_equal(out1, out2)


# ############# from_signals ############# #

entries = pd.Series([True, True, True, False, False], index=price.index)
entries_wide = entries.vbt.tile(3, keys=['a', 'b', 'c'])

exits = pd.Series([False, False, True, True, True], index=price.index)
exits_wide = exits.vbt.tile(3, keys=['a', 'b', 'c'])


class TestFromSignals:
    def test_one_column(self):
        portfolio = vbt.Portfolio.from_signals(price.tolist(), entries.tolist(), exits.tolist())
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_signals(price, entries, exits)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.index,
            pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Int64Index([0], dtype='int64')
        )
        assert portfolio.wrapper.ndim == 1
        assert portfolio.wrapper.freq == day_dt
        assert portfolio.wrapper.grouper.group_by is None

    def test_multiple_columns(self):
        portfolio = vbt.Portfolio.from_signals(price_wide, entries, exits)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 3, 100., 4., 0., 1),
                (1, 0, 100., 1., 0., 0), (1, 3, 100., 4., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.index,
            pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Index(['a', 'b', 'c'], dtype='object')
        )
        assert portfolio.wrapper.ndim == 2
        assert portfolio.wrapper.freq == day_dt
        assert portfolio.wrapper.grouper.group_by is None

    def test_group_by(self):
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, group_by=np.array([0, 0, 1]))
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 3, 100., 4., 0., 1),
                (1, 0, 100., 1., 0., 0), (1, 3, 100., 4., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.grouper.group_by,
            pd.Int64Index([0, 0, 1], dtype='int64')
        )
        pd.testing.assert_series_equal(
            portfolio.init_cash(),
            pd.Series([200., 100.], index=pd.Int64Index([0, 1], dtype='int64'))
        )
        assert not portfolio.cash_sharing

    def test_cash_sharing(self):
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, group_by=np.array([0, 0, 1]), cash_sharing=True)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 3, 100., 4., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.grouper.group_by,
            pd.Int64Index([0, 0, 1], dtype='int64')
        )
        pd.testing.assert_series_equal(
            portfolio.init_cash(),
            pd.Series([100., 100.], index=pd.Int64Index([0, 1], dtype='int64'))
        )
        assert portfolio.cash_sharing

    def test_call_seq(self):
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Default)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 3, 100., 4., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ])
        )
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Reversed)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (1, 0, 100., 1., 0., 0), (1, 3, 100., 4., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]
            ])
        )
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Random, seed=seed)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (1, 0, 100., 1., 0., 0), (1, 3, 100., 4., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]
            ])
        )
        with pytest.raises(Exception) as e_info:
            _ = vbt.Portfolio.from_signals(
                price_wide, entries, exits, group_by=np.array([0, 0, 1]),
                cash_sharing=True, call_seq=CallSeqType.Auto)

    def test_size(self):
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=[1., 2., np.inf])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 4., 0., 1),
                (1, 0, 2., 1., 0., 0), (1, 3, 2., 4., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 3, 100., 4., 0., 1)
            ], dtype=order_dt)
        )

    def test_init_cash(self):
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=10., init_cash=[1., 10., np.inf])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 4., 0., 1),
                (1, 0, 10., 1., 0., 0), (1, 3, 10., 4., 0., 1),
                (2, 0, 10., 1., 0., 0), (2, 3, 10., 4., 0., 1)
            ], dtype=order_dt)
        )
        assert type(portfolio._init_cash) == np.ndarray
        base_portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=10., init_cash=np.inf)
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=10., init_cash=InitCashMode.Auto)
        record_arrays_close(
            portfolio.orders().records_arr,
            base_portfolio.orders().records_arr
        )
        assert portfolio._init_cash == InitCashMode.Auto
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=10., init_cash=InitCashMode.AutoAlign)
        record_arrays_close(
            portfolio.orders().records_arr,
            base_portfolio.orders().records_arr
        )
        assert portfolio._init_cash == InitCashMode.AutoAlign

    def test_costs(self):
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=1., fees=[0., 0.01, 1.])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 4., 0., 1),
                (1, 0, 1., 1., 0.01, 0), (1, 3, 1., 4., 0.04, 1),
                (2, 0, 1., 1., 1., 0), (2, 3, 1., 4., 4., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=1., fixed_fees=[0., 0.01, 1.])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 4., 0., 1),
                (1, 0, 1., 1., 0.01, 0), (1, 3, 1., 4., 0.01, 1),
                (2, 0, 1., 1., 1., 0), (2, 3, 1., 4., 1., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=1., slippage=[0., 0.01, 1.])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 4., 0., 1),
                (1, 0, 1., 1.01, 0., 0), (1, 3, 1., 3.96, 0., 1),
                (2, 0, 1., 2., 0., 0), (2, 3, 1., 0., 0., 1)
            ], dtype=order_dt)
        )

    def test_entry_and_exit_price(self):
        portfolio = vbt.Portfolio.from_signals(
            price, entries, exits, size=1.,
            entry_price=price * 1.01, exit_price=price * 0.99)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1.01, 0., 0), (0, 3, 1., 3.96, 0., 1)
            ], dtype=order_dt)
        )

    def test_accumulate_exit_mode(self):
        portfolio = vbt.Portfolio.from_signals(
            price, entries, exits, size=1.,
            accumulate=True, accumulate_exit_mode=AccumulateExitMode.Close)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 3, 2., 4., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_signals(
            price, entries, exits, size=1.,
            accumulate=True, accumulate_exit_mode=AccumulateExitMode.Reduce)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 3, 1., 4., 0., 1), (0, 4, 1., 5., 0., 1)
            ], dtype=order_dt)
        )

    def test_conflict_mode(self):
        portfolio = vbt.Portfolio.from_signals(
            price, entries, exits, size=1.,
            accumulate=True, conflict_mode=ConflictMode.Ignore)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 3, 2., 4., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_signals(
            price, entries, exits, size=1.,
            accumulate=True, conflict_mode=ConflictMode.Exit)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 2, 2., 3., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_signals(
            price, entries, exits, size=1.,
            accumulate=True, conflict_mode=ConflictMode.Target)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 2, 1., 3., 0., 1), (0, 3, 1., 4., 0., 1)
            ], dtype=order_dt)
        )

    def test_min_size(self):
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=1., min_size=[0., 1., 10.])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 4., 0., 1),
                (1, 0, 1., 1., 0., 0), (1, 3, 1., 4., 0., 1)
            ], dtype=order_dt)
        )

    def test_reject_prob(self):
        portfolio = vbt.Portfolio.from_signals(
            price_wide, entries, exits, size=1., reject_prob=[0., 0.5, 1.], seed=seed)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 4., 0., 1),
                (1, 1, 1., 2., 0., 0), (1, 3, 1., 4., 0., 1)
            ], dtype=order_dt)
        )


# ############# from_orders ############# #

order_size = pd.Series([np.inf, -np.inf, np.nan, np.inf, -np.inf], index=price.index)
order_size_wide = order_size.vbt.tile(3, keys=['a', 'b', 'c'])
order_size_one = pd.Series([1, -1, np.nan, 1, -1], index=price.index)


class TestFromOrders:
    def test_one_column(self):
        portfolio = vbt.Portfolio.from_orders(price.tolist(), order_size.tolist())
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 3, 50., 4., 0., 0), (0, 4, 50., 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(price, order_size)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 3, 50., 4., 0., 0), (0, 4, 50., 5., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.index,
            pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Int64Index([0], dtype='int64')
        )
        assert portfolio.wrapper.ndim == 1
        assert portfolio.wrapper.freq == day_dt
        assert portfolio.wrapper.grouper.group_by is None

    def test_multiple_columns(self):
        portfolio = vbt.Portfolio.from_orders(price_wide, order_size)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 3, 50., 4., 0., 0), (0, 4, 50., 5., 0., 1),
                (1, 0, 100., 1., 0., 0), (1, 1, 100., 2., 0., 1),
                (1, 3, 50., 4., 0., 0), (1, 4, 50., 5., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 1, 100., 2., 0., 1),
                (2, 3, 50., 4., 0., 0), (2, 4, 50., 5., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.index,
            pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Index(['a', 'b', 'c'], dtype='object')
        )
        assert portfolio.wrapper.ndim == 2
        assert portfolio.wrapper.freq == day_dt
        assert portfolio.wrapper.grouper.group_by is None

    def test_group_by(self):
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size, group_by=np.array([0, 0, 1]))
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 3, 50., 4., 0., 0), (0, 4, 50., 5., 0., 1),
                (1, 0, 100., 1., 0., 0), (1, 1, 100., 2., 0., 1),
                (1, 3, 50., 4., 0., 0), (1, 4, 50., 5., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 1, 100., 2., 0., 1),
                (2, 3, 50., 4., 0., 0), (2, 4, 50., 5., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.grouper.group_by,
            pd.Int64Index([0, 0, 1], dtype='int64')
        )
        pd.testing.assert_series_equal(
            portfolio.init_cash(),
            pd.Series([200., 100.], index=pd.Int64Index([0, 1], dtype='int64'))
        )
        assert not portfolio.cash_sharing

    def test_cash_sharing(self):
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size, group_by=np.array([0, 0, 1]), cash_sharing=True)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 3, 50., 4., 0., 0), (0, 4, 50., 5., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 1, 100., 2., 0., 1),
                (2, 3, 50., 4., 0., 0), (2, 4, 50., 5., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.grouper.group_by,
            pd.Int64Index([0, 0, 1], dtype='int64')
        )
        pd.testing.assert_series_equal(
            portfolio.init_cash(),
            pd.Series([100., 100.], index=pd.Int64Index([0, 1], dtype='int64'))
        )
        assert portfolio.cash_sharing

    def test_call_seq(self):
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Default)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 3, 50., 4., 0., 0), (0, 4, 50., 5., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 1, 100., 2., 0., 1),
                (2, 3, 50., 4., 0., 0), (2, 4, 50., 5., 0., 1)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ])
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Reversed)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (1, 0, 100., 1., 0., 0), (1, 1, 100., 2., 0., 1),
                (1, 3, 50., 4., 0., 0), (1, 4, 50., 5., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 1, 100., 2., 0., 1),
                (2, 3, 50., 4., 0., 0), (2, 4, 50., 5., 0., 1)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]
            ])
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Random, seed=seed)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (1, 0, 100., 1., 0., 0), (1, 1, 100., 2., 0., 1),
                (1, 3, 50., 4., 0., 0), (1, 4, 50., 5., 0., 1),
                (2, 0, 100., 1., 0., 0), (2, 1, 100., 2., 0., 1),
                (2, 3, 50., 4., 0., 0), (2, 4, 50., 5., 0., 1)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]
            ])
        )
        price_one = pd.Series([1., 1., 1., 1., 1.], index=price.index)
        target_hold_value = pd.DataFrame({
            'a': [0., 70., 30., 0., 70.],
            'b': [30., 0., 70., 30., 30.],
            'c': [70., 30., 0., 70., 0.]
        }, index=price.index)
        portfolio = vbt.Portfolio.from_orders(
            price_one, target_hold_value, size_type=SizeType.TargetValue,
            group_by=np.array([0, 0, 0]), cash_sharing=True, val_price=price_one,
            call_seq=CallSeqType.Random, seed=seed)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 1, 70., 1., 0., 0), (0, 2, 40., 1., 0., 1),
                (0, 3, 30., 1., 0., 1), (0, 4, 30., 1., 0., 0),
                (1, 0, 30., 1., 0., 0), (1, 1, 30., 1., 0., 1),
                (1, 2, 40., 1., 0., 0), (1, 3, 10., 1., 0., 1),
                (2, 0, 70., 1., 0., 0), (2, 1, 40., 1., 0., 1),
                (2, 2, 30., 1., 0., 1), (2, 3, 40., 1., 0., 0),
                (2, 4, 40., 1., 0., 1)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 2],
                [1, 2, 0],
                [0, 1, 2],
                [1, 2, 0],
                [0, 1, 2]
            ])
        )
        records_result = np.array([
            (0, 1, 70., 1., 0., 0), (0, 2, 40., 1., 0., 1),
            (0, 3, 30., 1., 0., 1), (0, 4, 70., 1., 0., 0),
            (1, 0, 30., 1., 0., 0), (1, 1, 30., 1., 0., 1),
            (1, 2, 70., 1., 0., 0), (1, 3, 40., 1., 0., 1),
            (2, 0, 70., 1., 0., 0), (2, 1, 40., 1., 0., 1),
            (2, 2, 30., 1., 0., 1), (2, 3, 70., 1., 0., 0),
            (2, 4, 70., 1., 0., 1)
        ], dtype=order_dt)
        call_seq_result = np.array([
            [0, 1, 2],
            [2, 1, 0],
            [0, 2, 1],
            [1, 0, 2],
            [2, 1, 0]
        ])
        portfolio = vbt.Portfolio.from_orders(
            price_one, target_hold_value, size_type=SizeType.TargetValue,
            group_by=np.array([0, 0, 0]), cash_sharing=True, val_price=price_one,
            call_seq=CallSeqType.Auto)
        record_arrays_close(
            portfolio.orders().records_arr,
            records_result
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            call_seq_result
        )
        portfolio = vbt.Portfolio.from_orders(
            price_one, target_hold_value / 100., size_type=SizeType.TargetPercent,
            group_by=np.array([0, 0, 0]), cash_sharing=True, val_price=price_one,
            call_seq=CallSeqType.Auto)
        record_arrays_close(
            portfolio.orders().records_arr,
            records_result
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            call_seq_result
        )

    def test_size_type(self):
        portfolio = vbt.Portfolio.from_orders(
            price, order_size_one, size_type=SizeType.Shares)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 3, 1., 4., 0., 0), (0, 4, 1., 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price, 50., size_type=SizeType.TargetShares)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 50., 1., 0., 0)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide, 50., size_type=SizeType.TargetShares,
            group_by=np.array([0, 0, 1]), cash_sharing=True)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 50., 1., 0., 0), (1, 0, 50., 1., 0., 0),
                (2, 0, 50., 1., 0., 0)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price.iloc[1:], 50., size_type=SizeType.TargetValue,
            val_price=price.iloc[:-1].values)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 50., 2., 0., 0), (0, 1, 25., 3., 0., 1),
                (0, 2, 8.33333333, 4., 0., 1), (0, 3, 4.16666667, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price.iloc[1:], 50., size_type=SizeType.TargetValue,
            val_price=price.iloc[1:].values)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 25., 2., 0., 0), (0, 1, 8.33333333, 3., 0., 1),
                (0, 2, 4.16666667, 4., 0., 1), (0, 3, 2.5, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide.iloc[1:], 50., size_type=SizeType.TargetValue,
            group_by=np.array([0, 0, 1]), cash_sharing=True,
            val_price=price_wide.iloc[:-1].values)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 50., 2., 0., 0), (0, 1, 25., 3., 0., 1),
                (0, 2, 8.33333333, 4., 0., 1), (0, 3, 4.16666667, 5., 0., 1),
                (1, 1, 25., 3., 0., 0), (1, 2, 8.33333333, 4., 0., 1),
                (1, 3, 4.16666667, 5., 0., 1), (2, 0, 50., 2., 0., 0),
                (2, 1, 25., 3., 0., 1), (2, 2, 8.33333333, 4., 0., 1),
                (2, 3, 4.16666667, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide.iloc[1:], 50., size_type=SizeType.TargetValue,
            group_by=np.array([0, 0, 1]), cash_sharing=True,
            val_price=price_wide.iloc[1:].values)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 25., 2., 0., 0), (0, 1, 8.33333333, 3., 0., 1),
                (0, 2, 4.16666667, 4., 0., 1), (0, 3, 2.5, 5., 0., 1),
                (1, 0, 25., 2., 0., 0), (1, 1, 8.33333333, 3., 0., 1),
                (1, 2, 4.16666667, 4., 0., 1), (1, 3, 2.5, 5., 0., 1),
                (2, 0, 25., 2., 0., 0), (2, 1, 8.33333333, 3., 0., 1),
                (2, 2, 4.16666667, 4., 0., 1), (2, 3, 2.5, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price.iloc[1:], 0.5, size_type=SizeType.TargetPercent,
            val_price=price.iloc[:-1].values)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 50., 2., 0., 0), (0, 1, 25., 3., 0., 1),
                (0, 3, 3.125, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price.iloc[1:], 0.5, size_type=SizeType.TargetPercent,
            val_price=price.iloc[1:].values)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 25., 2., 0., 0), (0, 1, 4.16666667, 3., 0., 1),
                (0, 2, 2.60416667, 4., 0., 1), (0, 3, 1.82291667, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide.iloc[1:], 0.5, size_type=SizeType.TargetPercent,
            group_by=np.array([0, 0, 1]), cash_sharing=True,
            val_price=price_wide.iloc[:-1].values)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 50., 2., 0., 0), (0, 1, 25., 3., 0., 1),
                (1, 1, 25., 3., 0., 0), (2, 0, 50., 2., 0., 0),
                (2, 1, 25., 3., 0., 1), (2, 3, 3.125, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide.iloc[1:], 0.5, size_type=SizeType.TargetPercent,
            group_by=np.array([0, 0, 1]), cash_sharing=True,
            val_price=price_wide.iloc[1:].values)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 25., 2., 0., 0), (1, 0, 25., 2., 0., 0),
                (2, 0, 25., 2., 0., 0), (2, 1, 4.16666667, 3., 0., 1),
                (2, 2, 2.60416667, 4., 0., 1), (2, 3, 1.82291667, 5., 0., 1)
            ], dtype=order_dt)
        )

    def test_order_price(self):
        portfolio = vbt.Portfolio.from_orders(
            price, order_size, order_price=price * 1.1)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 90.90909091, 1.1, 0., 0), (0, 1, 90.90909091, 2.2, 0., 1),
                (0, 3, 45.45454545, 4.4, 0., 0), (0, 4, 45.45454545, 5.5, 0., 1)
            ], dtype=order_dt)
        )

    def test_init_cash(self):
        portfolio = vbt.Portfolio.from_orders(
            price_wide, 10., init_cash=[1., 10., np.inf])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (1, 0, 10., 1., 0., 0),
                (2, 0, 10., 1., 0., 0), (2, 1, 10., 2., 0., 0),
                (2, 2, 10., 3., 0., 0), (2, 3, 10., 4., 0., 0),
                (2, 4, 10., 5., 0., 0)
            ], dtype=order_dt)
        )
        assert type(portfolio._init_cash) == np.ndarray
        base_portfolio = vbt.Portfolio.from_orders(
            price_wide, 10., init_cash=np.inf)
        portfolio = vbt.Portfolio.from_orders(
            price_wide, 10., init_cash=InitCashMode.Auto)
        record_arrays_close(
            portfolio.orders().records_arr,
            base_portfolio.orders().records_arr
        )
        assert portfolio._init_cash == InitCashMode.Auto
        portfolio = vbt.Portfolio.from_orders(
            price_wide, 10., init_cash=InitCashMode.AutoAlign)
        record_arrays_close(
            portfolio.orders().records_arr,
            base_portfolio.orders().records_arr
        )
        assert portfolio._init_cash == InitCashMode.AutoAlign

    def test_costs(self):
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size_one, fees=[0., 0.01, 1.])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 3, 1., 4., 0., 0), (0, 4, 1., 5., 0., 1),
                (1, 0, 1., 1., 0.01, 0), (1, 1, 1., 2., 0.02, 1),
                (1, 3, 1., 4., 0.04, 0), (1, 4, 1., 5., 0.05, 1),
                (2, 0, 1., 1., 1., 0), (2, 1, 1., 2., 2., 1),
                (2, 3, 1., 4., 4., 0), (2, 4, 1., 5., 5., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size_one, fixed_fees=[0., 0.01, 1.])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 3, 1., 4., 0., 0), (0, 4, 1., 5., 0., 1),
                (1, 0, 1., 1., 0.01, 0), (1, 1, 1., 2., 0.01, 1),
                (1, 3, 1., 4., 0.01, 0), (1, 4, 1., 5., 0.01, 1),
                (2, 0, 1., 1., 1., 0), (2, 1, 1., 2., 1., 1),
                (2, 3, 1., 4., 1., 0), (2, 4, 1., 5., 1., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size_one, slippage=[0., 0.01, 1.])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 3, 1., 4., 0., 0), (0, 4, 1., 5., 0., 1),
                (1, 0, 1., 1.01, 0., 0), (1, 1, 1., 1.98, 0., 1),
                (1, 3, 1., 4.04, 0., 0), (1, 4, 1., 4.95, 0., 1),
                (2, 0, 1., 2., 0., 0), (2, 1, 1., 0., 0., 1),
                (2, 3, 1., 8., 0., 0), (2, 4, 1., 0., 0., 1)
            ], dtype=order_dt)
        )

    def test_min_size(self):
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size_one, min_size=[0., 1., 10.])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 3, 1., 4., 0., 0), (0, 4, 1., 5., 0., 1),
                (1, 0, 1., 1., 0., 0), (1, 1, 1., 2., 0., 1),
                (1, 3, 1., 4., 0., 0), (1, 4, 1., 5., 0., 1)
            ], dtype=order_dt)
        )

    def test_reject_prob(self):
        portfolio = vbt.Portfolio.from_orders(
            price_wide, order_size_one, reject_prob=[0., 0.5, 1.], seed=seed)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 3, 1., 4., 0., 0), (0, 4, 1., 5., 0., 1),
                (1, 3, 1., 4., 0., 0), (1, 4, 1., 5., 0., 1)
            ], dtype=order_dt)
        )


# ############# from_order_func ############# #

@njit
def order_func_nb(oc, size):
    return Order(size if oc.i % 2 == 0 else -size, SizeType.Shares, oc.close[oc.i, oc.col], 0., 0., 0., 0.)


class TestFromOrderFunc:
    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_one_column(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(price.tolist(), order_func_nb, np.inf, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 2, 66.66666667, 3., 0., 0), (0, 3, 66.66666667, 4., 0., 1),
                (0, 4, 53.33333333, 5., 0., 0)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_order_func(price, order_func_nb, np.inf, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 2, 66.66666667, 3., 0., 0), (0, 3, 66.66666667, 4., 0., 1),
                (0, 4, 53.33333333, 5., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.index,
            pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Int64Index([0], dtype='int64')
        )
        assert portfolio.wrapper.ndim == 1
        assert portfolio.wrapper.freq == day_dt
        assert portfolio.wrapper.grouper.group_by is None

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_multiple_columns(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(price_wide, order_func_nb, np.inf, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 2, 66.66666667, 3., 0., 0), (0, 3, 66.66666667, 4., 0., 1),
                (0, 4, 53.33333333, 5., 0., 0), (1, 0, 100., 1., 0., 0),
                (1, 1, 100., 2., 0., 1), (1, 2, 66.66666667, 3., 0., 0),
                (1, 3, 66.66666667, 4., 0., 1), (1, 4, 53.33333333, 5., 0., 0),
                (2, 0, 100., 1., 0., 0), (2, 1, 100., 2., 0., 1),
                (2, 2, 66.66666667, 3., 0., 0), (2, 3, 66.66666667, 4., 0., 1),
                (2, 4, 53.33333333, 5., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.index,
            pd.DatetimeIndex(['2020-01-01', '2020-01-02', '2020-01-03', '2020-01-04', '2020-01-05'])
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Index(['a', 'b', 'c'], dtype='object')
        )
        assert portfolio.wrapper.ndim == 2
        assert portfolio.wrapper.freq == day_dt
        assert portfolio.wrapper.grouper.group_by is None

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_target_shape(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(
            price, order_func_nb, np.inf,
            target_shape=(5,), row_wise=test_row_wise)
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Int64Index([0], dtype='int64')
        )
        assert portfolio.wrapper.ndim == 1
        portfolio = vbt.Portfolio.from_order_func(
            price, order_func_nb, np.inf,
            target_shape=(5, 1), row_wise=test_row_wise)
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Int64Index([0], dtype='int64', name='iteration_idx')
        )
        assert portfolio.wrapper.ndim == 2
        portfolio = vbt.Portfolio.from_order_func(
            price, order_func_nb, np.inf,
            target_shape=(5, 1), row_wise=test_row_wise,
            keys=pd.Index(['first'], name='custom'))
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Index(['first'], dtype='object', name='custom')
        )
        assert portfolio.wrapper.ndim == 2
        portfolio = vbt.Portfolio.from_order_func(
            price, order_func_nb, np.inf,
            target_shape=(5, 3), row_wise=test_row_wise)
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Int64Index([0, 1, 2], dtype='int64', name='iteration_idx')
        )
        assert portfolio.wrapper.ndim == 2
        portfolio = vbt.Portfolio.from_order_func(
            price, order_func_nb, np.inf,
            target_shape=(5, 3), row_wise=test_row_wise,
            keys=pd.Index(['first', 'second', 'third'], name='custom'))
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            pd.Index(['first', 'second', 'third'], dtype='object', name='custom')
        )
        assert portfolio.wrapper.ndim == 2

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_group_by(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, np.inf,
            group_by=np.array([0, 0, 1]), row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 2, 66.66666667, 3., 0., 0), (0, 3, 66.66666667, 4., 0., 1),
                (0, 4, 53.33333333, 5., 0., 0), (1, 0, 100., 1., 0., 0),
                (1, 1, 100., 2., 0., 1), (1, 2, 66.66666667, 3., 0., 0),
                (1, 3, 66.66666667, 4., 0., 1), (1, 4, 53.33333333, 5., 0., 0),
                (2, 0, 100., 1., 0., 0), (2, 1, 100., 2., 0., 1),
                (2, 2, 66.66666667, 3., 0., 0), (2, 3, 66.66666667, 4., 0., 1),
                (2, 4, 53.33333333, 5., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.grouper.group_by,
            pd.Int64Index([0, 0, 1], dtype='int64')
        )
        pd.testing.assert_series_equal(
            portfolio.init_cash(),
            pd.Series([200., 100.], index=pd.Int64Index([0, 1], dtype='int64'))
        )
        assert not portfolio.cash_sharing

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_cash_sharing(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, np.inf,
            group_by=np.array([0, 0, 1]), cash_sharing=True, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 2, 66.66666667, 3., 0., 0), (0, 3, 66.66666667, 4., 0., 1),
                (0, 4, 53.33333333, 5., 0., 0), (2, 0, 100., 1., 0., 0),
                (2, 1, 100., 2., 0., 1), (2, 2, 66.66666667, 3., 0., 0),
                (2, 3, 66.66666667, 4., 0., 1), (2, 4, 53.33333333, 5., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.grouper.group_by,
            pd.Int64Index([0, 0, 1], dtype='int64')
        )
        pd.testing.assert_series_equal(
            portfolio.init_cash(),
            pd.Series([100., 100.], index=pd.Int64Index([0, 1], dtype='int64'))
        )
        assert portfolio.cash_sharing

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_call_seq(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, np.inf, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Default, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 100., 1., 0., 0), (0, 1, 100., 2., 0., 1),
                (0, 2, 66.66666667, 3., 0., 0), (0, 3, 66.66666667, 4., 0., 1),
                (0, 4, 53.33333333, 5., 0., 0), (2, 0, 100., 1., 0., 0),
                (2, 1, 100., 2., 0., 1), (2, 2, 66.66666667, 3., 0., 0),
                (2, 3, 66.66666667, 4., 0., 1), (2, 4, 53.33333333, 5., 0., 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 1, 0]
            ])
        )
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, np.inf, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Reversed, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (1, 0, 100., 1., 0., 0), (1, 1, 100., 2., 0., 1),
                (1, 2, 66.66666667, 3., 0., 0), (1, 3, 66.66666667, 4., 0., 1),
                (1, 4, 53.33333333, 5., 0., 0), (2, 0, 100., 1., 0., 0),
                (2, 1, 100., 2., 0., 1), (2, 2, 66.66666667, 3., 0., 0),
                (2, 3, 66.66666667, 4., 0., 1), (2, 4, 53.33333333, 5., 0., 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]
            ])
        )
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, np.inf, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq=CallSeqType.Random, seed=seed, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (1, 0, 100., 1., 0., 0), (1, 1, 100., 2., 0., 1),
                (1, 2, 66.66666667, 3., 0., 0), (1, 3, 66.66666667, 4., 0., 1),
                (1, 4, 53.33333333, 5., 0., 0), (2, 0, 100., 1., 0., 0),
                (2, 1, 100., 2., 0., 1), (2, 2, 66.66666667, 3., 0., 0),
                (2, 3, 66.66666667, 4., 0., 1), (2, 4, 53.33333333, 5., 0., 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [1, 0, 0],
                [0, 1, 0],
                [1, 0, 0],
                [1, 0, 0],
                [1, 0, 0]
            ])
        )
        with pytest.raises(Exception) as e_info:
            _ = vbt.Portfolio.from_order_func(
                price_wide, order_func_nb, np.inf, group_by=np.array([0, 0, 1]),
                cash_sharing=True, call_seq=CallSeqType.Auto, row_wise=test_row_wise
            )

        price_one = pd.Series([1., 1., 1., 1., 1.], index=price.index)
        target_hold_value = pd.DataFrame({
            'a': [0., 70., 30., 0., 70.],
            'b': [30., 0., 70., 30., 30.],
            'c': [70., 30., 0., 70., 0.]
        }, index=price.index)

        @njit
        def segment_prep_func_nb(sc, target_hold_value, price_one):
            order_size = np.copy(target_hold_value[sc.i, sc.from_col:sc.to_col])
            order_size_type = np.full(sc.group_len, SizeType.TargetValue)
            temp_float_arr = np.empty(sc.group_len, dtype=np.float_)
            auto_call_seq_ctx_nb(sc, order_size, order_size_type, temp_float_arr)
            sc.last_val_price[sc.from_col:sc.to_col] = price_one[sc.i]
            return order_size, order_size_type

        @njit
        def pct_order_func_nb(oc, order_size, order_size_type, price_one):
            col_i = oc.call_seq_now[oc.call_idx]
            return Order(
                order_size[col_i],
                order_size_type[col_i],
                price_one[oc.i],
                0., 0., 0., 0.
            )

        portfolio = vbt.Portfolio.from_order_func(
            price_wide, pct_order_func_nb, price_one, group_by=np.array([0, 0, 0]),
            cash_sharing=True, call_seq=CallSeqType.Default,
            segment_prep_func_nb=segment_prep_func_nb,
            segment_prep_args=(target_hold_value, price_one), row_wise=test_row_wise)
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 2],
                [2, 1, 0],
                [0, 2, 1],
                [1, 0, 2],
                [2, 1, 0]
            ])
        )

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_target_value(self, test_row_wise):
        @njit
        def target_val_segment_prep_func_nb(sc, val_price):
            sc.last_val_price[sc.from_col:sc.to_col] = val_price[sc.i]
            return ()

        @njit
        def target_val_order_func_nb(oc):
            return Order(50., SizeType.TargetValue, oc.close[oc.i, oc.col], 0., 0., 0., 0.)

        portfolio = vbt.Portfolio.from_order_func(
            price.iloc[1:], target_val_order_func_nb,
            segment_prep_func_nb=target_val_segment_prep_func_nb,
            segment_prep_args=(price.iloc[:-1],), row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 50., 2., 0., 0), (0, 1, 25., 3., 0., 1),
                (0, 2, 8.33333333, 4., 0., 1), (0, 3, 4.16666667, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_order_func(
            price.iloc[1:], target_val_order_func_nb,
            segment_prep_func_nb=target_val_segment_prep_func_nb,
            segment_prep_args=(price.iloc[1:],), row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 25., 2., 0., 0), (0, 1, 8.33333333, 3., 0., 1),
                (0, 2, 4.16666667, 4., 0., 1), (0, 3, 2.5, 5., 0., 1)
            ], dtype=order_dt)
        )

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_target_percent(self, test_row_wise):
        @njit
        def target_pct_segment_prep_func_nb(sc, val_price):
            sc.last_val_price[sc.from_col:sc.to_col] = val_price[sc.i]
            return ()

        @njit
        def target_pct_order_func_nb(oc):
            return Order(0.5, SizeType.TargetPercent, oc.close[oc.i, oc.col], 0., 0., 0., 0.)

        portfolio = vbt.Portfolio.from_order_func(
            price.iloc[1:], target_pct_order_func_nb,
            segment_prep_func_nb=target_pct_segment_prep_func_nb,
            segment_prep_args=(price.iloc[:-1],), row_wise=test_row_wise)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 50., 2., 0., 0), (0, 1, 25., 3., 0., 1),
                (0, 3, 3.125, 5., 0., 1)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_order_func(
            price.iloc[1:], target_pct_order_func_nb,
            segment_prep_func_nb=target_pct_segment_prep_func_nb,
            segment_prep_args=(price.iloc[1:],), row_wise=True)
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 25., 2., 0., 0), (0, 1, 4.16666667, 3., 0., 1),
                (0, 2, 2.60416667, 4., 0., 1), (0, 3, 1.82291667, 5., 0., 1)
            ], dtype=order_dt)
        )

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_init_cash(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, 10., row_wise=test_row_wise, init_cash=[1., 10., np.inf])
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 2, 0.66666667, 3., 0., 0), (0, 3, 0.66666667, 4., 0., 1),
                (0, 4, 0.53333333, 5., 0., 0), (1, 0, 10., 1., 0., 0),
                (1, 1, 10., 2., 0., 1), (1, 2, 6.66666667, 3., 0., 0),
                (1, 3, 6.66666667, 4., 0., 1), (1, 4, 5.33333333, 5., 0., 0),
                (2, 0, 10., 1., 0., 0), (2, 1, 10., 2., 0., 1),
                (2, 2, 10., 3., 0., 0), (2, 3, 10., 4., 0., 1),
                (2, 4, 10., 5., 0., 0)
            ], dtype=order_dt)
        )
        assert type(portfolio._init_cash) == np.ndarray
        base_portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, 10., row_wise=test_row_wise, init_cash=np.inf)
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, 10., row_wise=test_row_wise, init_cash=InitCashMode.Auto)
        record_arrays_close(
            portfolio.orders().records_arr,
            base_portfolio.orders().records_arr
        )
        assert portfolio._init_cash == InitCashMode.Auto
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, 10., row_wise=test_row_wise, init_cash=InitCashMode.AutoAlign)
        record_arrays_close(
            portfolio.orders().records_arr,
            base_portfolio.orders().records_arr
        )
        assert portfolio._init_cash == InitCashMode.AutoAlign

    def test_func_calls(self):
        @njit
        def prep_func_nb(simc, call_i, sim_lst):
            call_i[0] += 1
            sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def group_prep_func_nb(gc, call_i, group_lst):
            call_i[0] += 1
            group_lst.append(call_i[0])
            return (call_i,)

        @njit
        def segment_prep_func_nb(sc, call_i, segment_lst):
            call_i[0] += 1
            segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def order_func_nb(oc, call_i, order_lst):
            call_i[0] += 1
            order_lst.append(call_i[0])
            return NoOrder

        call_i = np.array([0])
        sim_lst = List.empty_list(typeof(0))
        group_lst = List.empty_list(typeof(0))
        segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        _ = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, order_lst,
            group_by=np.array([0, 0, 1]),
            prep_func_nb=prep_func_nb, prep_args=(call_i, sim_lst),
            group_prep_func_nb=group_prep_func_nb, group_prep_args=(group_lst,),
            segment_prep_func_nb=segment_prep_func_nb, segment_prep_args=(segment_lst,)
        )
        assert call_i[0] == 28
        assert list(sim_lst) == [1]
        assert list(group_lst) == [2, 18]
        assert list(segment_lst) == [3, 6, 9, 12, 15, 19, 21, 23, 25, 27]
        assert list(order_lst) == [4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 20, 22, 24, 26, 28]

        call_i = np.array([0])
        sim_lst = List.empty_list(typeof(0))
        group_lst = List.empty_list(typeof(0))
        segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        active_mask = np.array([
            [False, True],
            [False, False],
            [False, True],
            [False, False],
            [False, True],
        ])
        _ = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, order_lst,
            group_by=np.array([0, 0, 1]),
            prep_func_nb=prep_func_nb, prep_args=(call_i, sim_lst),
            group_prep_func_nb=group_prep_func_nb, group_prep_args=(group_lst,),
            segment_prep_func_nb=segment_prep_func_nb, segment_prep_args=(segment_lst,),
            active_mask=active_mask
        )
        assert call_i[0] == 8
        assert list(sim_lst) == [1]
        assert list(group_lst) == [2]
        assert list(segment_lst) == [3, 5, 7]
        assert list(order_lst) == [4, 6, 8]

    def test_func_calls_row_wise(self):
        @njit
        def prep_func_nb(simc, call_i, sim_lst):
            call_i[0] += 1
            sim_lst.append(call_i[0])
            return (call_i,)

        @njit
        def row_prep_func_nb(gc, call_i, row_lst):
            call_i[0] += 1
            row_lst.append(call_i[0])
            return (call_i,)

        @njit
        def segment_prep_func_nb(sc, call_i, segment_lst):
            call_i[0] += 1
            segment_lst.append(call_i[0])
            return (call_i,)

        @njit
        def order_func_nb(oc, call_i, order_lst):
            call_i[0] += 1
            order_lst.append(call_i[0])
            return NoOrder

        call_i = np.array([0])
        sim_lst = List.empty_list(typeof(0))
        row_lst = List.empty_list(typeof(0))
        segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        _ = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, order_lst,
            group_by=np.array([0, 0, 1]),
            prep_func_nb=prep_func_nb, prep_args=(call_i, sim_lst),
            row_prep_func_nb=row_prep_func_nb, row_prep_args=(row_lst,),
            segment_prep_func_nb=segment_prep_func_nb, segment_prep_args=(segment_lst,),
            row_wise=True
        )
        assert call_i[0] == 31
        assert list(sim_lst) == [1]
        assert list(row_lst) == [2, 8, 14, 20, 26]
        assert list(segment_lst) == [3, 6, 9, 12, 15, 18, 21, 24, 27, 30]
        assert list(order_lst) == [4, 5, 7, 10, 11, 13, 16, 17, 19, 22, 23, 25, 28, 29, 31]

        call_i = np.array([0])
        sim_lst = List.empty_list(typeof(0))
        row_lst = List.empty_list(typeof(0))
        segment_lst = List.empty_list(typeof(0))
        order_lst = List.empty_list(typeof(0))
        active_mask = np.array([
            [False, False],
            [False, True],
            [True, False],
            [True, True],
            [False, False],
        ])
        _ = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, order_lst,
            group_by=np.array([0, 0, 1]),
            prep_func_nb=prep_func_nb, prep_args=(call_i, sim_lst),
            row_prep_func_nb=row_prep_func_nb, row_prep_args=(row_lst,),
            segment_prep_func_nb=segment_prep_func_nb, segment_prep_args=(segment_lst,),
            active_mask=active_mask,
            row_wise=True
        )
        assert call_i[0] == 14
        assert list(sim_lst) == [1]
        assert list(row_lst) == [2, 5, 9]
        assert list(segment_lst) == [3, 6, 10, 13]
        assert list(order_lst) == [4, 7, 8, 11, 12, 14]


# ############# Portfolio ############# #

price_na = pd.DataFrame({
    'a': [np.nan, 2., 3., 4., 5.],
    'b': [1., 2., np.nan, 4., 5.],
    'c': [1., 2., 3., 4., np.nan]
}, index=price.index)
order_size_new = pd.Series([1., 0.1, -1., -0.1, 1.])
group_by = pd.Index(['first', 'first', 'second'])

portfolio = vbt.Portfolio.from_orders(
    price_na, order_size_new, size_type=SizeType.Shares,
    fees=0.01, fixed_fees=0.1, slippage=0.01,
    init_cash=[100., 100., 100.],
    call_seq=CallSeqType.Reversed,
    group_by=None
)  # independent

portfolio_grouped = vbt.Portfolio.from_orders(
    price_na, order_size_new, size_type=SizeType.Shares,
    fees=0.01, fixed_fees=0.1, slippage=0.01,
    init_cash=[100., 100., 100.],
    call_seq=CallSeqType.Reversed,
    group_by=group_by,
    cash_sharing=False
)  # grouped

portfolio_shared = vbt.Portfolio.from_orders(
    price_na, order_size_new, size_type=SizeType.Shares,
    fees=0.01, fixed_fees=0.1, slippage=0.01,
    init_cash=[200., 100.],
    call_seq=CallSeqType.Reversed,
    group_by=group_by,
    cash_sharing=True
)  # shared


class TestPortfolio:
    def test_indexing(self):
        assert portfolio['a'].orders() == portfolio.orders()['a']
        assert portfolio['a'].init_cash() == portfolio.init_cash()['a']
        pd.testing.assert_series_equal(portfolio['a'].call_seq, portfolio.call_seq['a'])
        assert portfolio['c'].orders() == portfolio.orders()['c']
        assert portfolio['c'].init_cash() == portfolio.init_cash()['c']
        pd.testing.assert_series_equal(portfolio['c'].call_seq, portfolio.call_seq['c'])
        assert portfolio[['c']].orders() == portfolio.orders()[['c']]
        pd.testing.assert_series_equal(portfolio[['c']].init_cash(), portfolio.init_cash()[['c']])
        pd.testing.assert_frame_equal(portfolio[['c']].call_seq, portfolio.call_seq[['c']])

        assert portfolio_grouped['first'].orders() == portfolio_grouped.orders()['first']
        assert portfolio_grouped['first'].init_cash() == portfolio_grouped.init_cash()['first']
        pd.testing.assert_frame_equal(portfolio_grouped['first'].call_seq, portfolio_grouped.call_seq[['a', 'b']])
        assert portfolio_grouped[['first']].orders() == portfolio_grouped.orders()[['first']]
        pd.testing.assert_series_equal(
            portfolio_grouped[['first']].init_cash(),
            portfolio_grouped.init_cash()[['first']])
        pd.testing.assert_frame_equal(portfolio_grouped[['first']].call_seq, portfolio_grouped.call_seq[['a', 'b']])
        assert portfolio_grouped['second'].orders() == portfolio_grouped.orders()['second']
        assert portfolio_grouped['second'].init_cash() == portfolio_grouped.init_cash()['second']
        pd.testing.assert_series_equal(portfolio_grouped['second'].call_seq, portfolio_grouped.call_seq['c'])
        assert portfolio_grouped[['second']].orders() == portfolio_grouped.orders()[['second']]
        pd.testing.assert_series_equal(
            portfolio_grouped[['second']].init_cash(),
            portfolio_grouped.init_cash()[['second']])
        pd.testing.assert_frame_equal(portfolio_grouped[['second']].call_seq, portfolio_grouped.call_seq[['c']])

        assert portfolio_shared['first'].orders() == portfolio_shared.orders()['first']
        assert portfolio_shared['first'].init_cash() == portfolio_shared.init_cash()['first']
        pd.testing.assert_frame_equal(portfolio_shared['first'].call_seq, portfolio_shared.call_seq[['a', 'b']])
        assert portfolio_shared[['first']].orders() == portfolio_shared.orders()[['first']]
        pd.testing.assert_series_equal(portfolio_shared[['first']].init_cash(), portfolio_shared.init_cash()[['first']])
        pd.testing.assert_frame_equal(portfolio_shared[['first']].call_seq, portfolio_shared.call_seq[['a', 'b']])
        assert portfolio_shared['second'].orders() == portfolio_shared.orders()['second']
        assert portfolio_shared['second'].init_cash() == portfolio_shared.init_cash()['second']
        pd.testing.assert_series_equal(portfolio_shared['second'].call_seq, portfolio_shared.call_seq['c'])
        assert portfolio_shared[['second']].orders() == portfolio_shared.orders()[['second']]
        pd.testing.assert_series_equal(
            portfolio_shared[['second']].init_cash(),
            portfolio_shared.init_cash()[['second']])
        pd.testing.assert_frame_equal(portfolio_shared[['second']].call_seq, portfolio_shared.call_seq[['c']])

    def test_wrapper(self):
        assert portfolio.wrapper == ArrayWrapper.from_obj(price_na)
        assert portfolio_grouped.wrapper == ArrayWrapper.from_obj(price_na, group_by=group_by)
        assert portfolio_shared.wrapper == ArrayWrapper.from_obj(price_na, group_by=group_by, allow_modify=False)

    def test_cash_sharing(self):
        assert not portfolio.cash_sharing
        assert not portfolio_grouped.cash_sharing
        assert portfolio_shared.cash_sharing

    def test_call_seq(self):
        pd.testing.assert_frame_equal(
            portfolio.call_seq,
            pd.DataFrame(
                np.array([
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.call_seq,
            pd.DataFrame(
                np.array([
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.call_seq,
            pd.DataFrame(
                np.array([
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0],
                    [1, 0, 0]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )

    def test_incl_unrealized(self):
        assert not vbt.Portfolio.from_orders(price_na, 1000., incl_unrealized=False).incl_unrealized
        assert vbt.Portfolio.from_orders(price_na, 1000., incl_unrealized=True).incl_unrealized

    def test_regroup(self):
        pd.testing.assert_index_equal(portfolio.regroup(group_by).wrapper.grouper.group_by, group_by)
        assert portfolio_grouped.regroup(False).wrapper.grouper.group_by is None
        with pytest.raises(Exception) as e_info:
            _ = portfolio_shared.regroup(False)

    def test_close(self):
        pd.testing.assert_frame_equal(portfolio.close, price_na)
        pd.testing.assert_frame_equal(portfolio_grouped.close, price_na)
        pd.testing.assert_frame_equal(portfolio_shared.close, price_na)

    def test_fill_close(self):
        pd.testing.assert_frame_equal(
            portfolio.fill_close(ffill=False, bfill=False),
            price_na
        )
        pd.testing.assert_frame_equal(
            portfolio.fill_close(ffill=True, bfill=False),
            price_na.ffill()
        )
        pd.testing.assert_frame_equal(
            portfolio.fill_close(ffill=False, bfill=True),
            price_na.bfill()
        )
        pd.testing.assert_frame_equal(
            portfolio.fill_close(ffill=True, bfill=True),
            price_na.ffill().bfill()
        )

    def test_init_cash(self):
        pd.testing.assert_series_equal(
            portfolio.init_cash(),
            pd.Series(
                np.array([100.0, 100.0, 100.0]),
                index=price_na.columns
            )
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.init_cash(),
            pd.Series(
                np.array([200.0, 100.0]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.init_cash(),
            pd.Series(
                np.array([200.0, 100.0]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na, 1000., init_cash=InitCashMode.Auto, group_by=None).init_cash(),
            pd.Series(
                np.array([14000., 12000., 10000.]),
                index=price_na.columns
            )
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na, 1000., init_cash=InitCashMode.Auto, group_by=group_by).init_cash(),
            pd.Series(
                np.array([26000.0, 10000.0]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na, 1000., init_cash=InitCashMode.Auto, group_by=group_by, cash_sharing=True).init_cash(),
            pd.Series(
                np.array([26000.0, 10000.0]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na, 1000., init_cash=InitCashMode.AutoAlign, group_by=None).init_cash(),
            pd.Series(
                np.array([14000., 14000., 14000.]),
                index=price_na.columns
            )
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na, 1000., init_cash=InitCashMode.AutoAlign, group_by=group_by).init_cash(),
            pd.Series(
                np.array([26000.0, 26000.0]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na, 1000., init_cash=InitCashMode.AutoAlign, group_by=group_by, cash_sharing=True).init_cash(),
            pd.Series(
                np.array([26000.0, 26000.0]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.init_cash(group_by=False),
            pd.Series(
                np.array([100., 100., 100.]),
                index=price_na.columns
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.init_cash(group_by=False),
            pd.Series(
                np.array([200., 200., 100.]),
                index=price_na.columns
            )
        )
        pd.testing.assert_series_equal(
            portfolio.init_cash(group_by=group_by),
            pd.Series(
                np.array([200., 100.]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.init_cash(),
            pd.Series(
                np.array([200., 100.]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.init_cash(),
            pd.Series(
                np.array([200., 100.]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )

    def test_cash_flow(self):
        result = pd.DataFrame(
            np.array([
                [0., -1.1201, -1.1201],
                [-0.30402, -0.30402, -0.30402],
                [0.19403, 0., 2.8403],
                [0., 0.29204, 0.29204],
                [-5.2005, -5.2005, 0.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.cash_flow(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.cash_flow(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.cash_flow(group_by=False),
            result
        )
        result = pd.DataFrame(
            np.array([
                [-1.1201, -1.1201],
                [-0.60804, -0.30402],
                [0.19403, 2.8403],
                [0.29204, 0.29204],
                [-10.401, 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_frame_equal(
            portfolio.cash_flow(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.cash_flow(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.cash_flow(),
            result
        )

    def test_cash(self):
        result = pd.DataFrame(
            np.array([
                [100., 98.8799, 98.8799],
                [99.69598, 98.57588, 98.57588],
                [99.89001, 98.57588, 101.41618],
                [99.89001, 98.86792, 101.70822],
                [94.68951, 93.66742, 101.70822]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.cash(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.cash(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.cash(group_by=False),
            pd.DataFrame(
                np.array([
                    [200., 198.8799, 98.8799],
                    [199.69598, 198.57588, 98.57588],
                    [199.89001, 198.57588, 101.41618],
                    [199.89001, 198.86792, 101.70822],
                    [194.68951, 193.66742, 101.70822]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.cash(group_by=False, in_sim_order=True),
            pd.DataFrame(
                np.array([
                    [198.8799, 198.8799, 98.8799],
                    [198.27186, 198.57588, 98.57588],
                    [198.46589, 198.27186, 101.41618],
                    [198.75793, 198.75793, 101.70822],
                    [188.35693, 193.55743, 101.70822]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [198.8799, 98.8799],
                [198.27186, 98.57588],
                [198.46589, 101.41618],
                [198.75793, 101.70822],
                [188.35693, 101.70822]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_frame_equal(
            portfolio.cash(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.cash(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.cash(),
            result
        )

    def test_share_flow(self):
        result = pd.DataFrame(
            np.array([
                [0., 1., 1.],
                [0.1, 0.1, 0.1],
                [-0.1, 0., -1.],
                [0., -0.1, -0.1],
                [1., 1., 0.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.share_flow(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.share_flow(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.share_flow(),
            result
        )

    def test_shares(self):
        result = pd.DataFrame(
            np.array([
                [0., 1., 1.],
                [0.1, 1.1, 1.1],
                [0., 1.1, 0.1],
                [0., 1., 0.],
                [1., 2., 0.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.shares(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.shares(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.shares(),
            result
        )

    def test_orders(self):
        record_arrays_close(
            portfolio.orders().records_arr,
            np.array([
                (0, 1, 0.1, 2.02, 0.10202, 0), (0, 2, 0.1, 2.97, 0.10297, 1),
                (0, 4, 1., 5.05, 0.1505, 0), (1, 0, 1., 1.01, 0.1101, 0),
                (1, 1, 0.1, 2.02, 0.10202, 0), (1, 3, 0.1, 3.96, 0.10396, 1),
                (1, 4, 1., 5.05, 0.1505, 0), (2, 0, 1., 1.01, 0.1101, 0),
                (2, 1, 0.1, 2.02, 0.10202, 0), (2, 2, 1., 2.97, 0.1297, 1),
                (2, 3, 0.1, 3.96, 0.10396, 1)
            ], dtype=order_dt)
        )
        result = pd.Series(
            np.array([3, 4, 4]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.orders().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.orders(group_by=False).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.orders(group_by=False).count(),
            result
        )
        result = pd.Series(
            np.array([7, 4]),
            index=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_series_equal(
            portfolio.orders(group_by=group_by).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.orders().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.orders().count(),
            result
        )

    def test_trades(self):
        record_arrays_close(
            portfolio.trades(incl_unrealized=True).records_arr,
            np.array([
                (0, 0.1, 1, 2.02, 0.10202, 2, 2.97, 0.10297, -0.10999, -0.36178541, 1, 0),
                (0, 1., 4, 5.05, 0.1505, 4, 5., 0., -0.2005, -0.03855399, 0, 1),
                (1, 0.1, 0, 1.10181818, 0.01928364, 3, 3.96, 0.10396, 0.16257455, 1.25573688, 1, 2),
                (1, 2., 0, 3.07590909, 0.34333636, 4, 5., 0., 3.50484545, 0.53960925, 0, 2),
                (2, 1., 0, 1.10181818, 0.19283636, 2, 2.97, 0.1297, 1.54564545, 1.19386709, 1, 3),
                (2, 0.1, 0, 1.10181818, 0.01928364, 3, 3.96, 0.10396, 0.16257455, 1.25573688, 1, 3)
            ], dtype=trade_dt)
        )
        pd.testing.assert_series_equal(
            portfolio.trades(incl_unrealized=True).count(),
            pd.Series(
                np.array([2, 2, 2]),
                index=price_na.columns
            )
        )
        result = pd.Series(
            np.array([1, 1, 2]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.trades().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.trades(group_by=False).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.trades(group_by=False).count(),
            result
        )
        result = pd.Series(
            np.array([2, 2]),
            index=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_series_equal(
            portfolio.trades(group_by=group_by).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.trades().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.trades().count(),
            result
        )

    def test_positions(self):
        record_arrays_close(
            portfolio.positions(incl_unrealized=True).records_arr,
            np.array([
                (0, 0.1, 1, 2.02, 0.10202, 2, 2.97, 0.10297, -0.10999, -0.36178541, 1),
                (0, 1., 4, 5.05, 0.1505, 4, 5., 0., -0.2005, -0.03855399, 0),
                (1, 2.1, 0, 2.98190476, 0.36262, 4, np.nan, 0.10396, np.nan, np.nan, 0),
                (2, 1.1, 0, 1.10181818, 0.21212, 3, 3.06, 0.23366, 1.70822, 1.19949162, 1)
            ], dtype=position_dt)
        )
        pd.testing.assert_series_equal(
            portfolio.positions(incl_unrealized=True).count(),
            pd.Series(
                np.array([2, 1, 1]),
                index=price_na.columns
            )
        )
        result = pd.Series(
            np.array([1, 0, 1]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.positions().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.positions(group_by=False).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.positions(group_by=False).count(),
            result
        )
        result = pd.Series(
            np.array([1, 1]),
            index=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_series_equal(
            portfolio.positions(group_by=group_by).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.positions().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.positions().count(),
            result
        )

    def test_drawdowns(self):
        result = pd.Series(
            np.array([1, 0, 1]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.drawdowns().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.drawdowns(group_by=False).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.drawdowns(group_by=False).count(),
            result
        )
        result = pd.Series(
            np.array([0, 1]),
            index=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_series_equal(
            portfolio.drawdowns(group_by=group_by).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.drawdowns().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.drawdowns().count(),
            result
        )

    def test_holding_value(self):
        result = pd.DataFrame(
            np.array([
                [0., 1., 1.],
                [0.2, 2.2, 2.2],
                [0., np.nan, 0.3],
                [0., 4., 0.],
                [5., 10., 0.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.holding_value(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.holding_value(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.holding_value(group_by=False),
            result
        )
        result = pd.DataFrame(
            np.array([
                [1., 1.],
                [2.4, 2.2],
                [np.nan, 0.3],
                [4., 0.],
                [15., 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_frame_equal(
            portfolio.holding_value(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.holding_value(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.holding_value(),
            result
        )

    def test_value(self):
        result = pd.DataFrame(
            np.array([
                [100., 99.8799, 99.8799],
                [99.89598, 100.77588, 100.77588],
                [99.89001, np.nan, 101.71618],
                [99.89001, 102.86792, 101.70822],
                [99.68951, 103.66742, 101.70822]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.value(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.value(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.value(group_by=False),
            pd.DataFrame(
                np.array([
                    [200., 199.8799, 99.8799],
                    [199.89598, 200.77588, 100.77588],
                    [199.89001, np.nan, 101.71618],
                    [199.89001, 202.86792, 101.70822],
                    [199.68951, 203.66742, 101.70822]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.value(group_by=False, in_sim_order=True),
            pd.DataFrame(
                np.array([
                    [199.8799, 199.8799, 99.8799],
                    [200.67186, 200.77588, 100.77588],
                    [np.nan, np.nan, 101.71618],
                    [202.75793, 202.75793, 101.70822],
                    [203.35693, 203.55743, 101.70822]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [199.8799, 99.8799],
                [200.67186, 100.77588],
                [np.nan, 101.71618],
                [202.75793, 101.70822],
                [203.35693, 101.70822]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_frame_equal(
            portfolio.value(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.value(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.value(),
            result
        )

    def test_final_value(self):
        result = pd.Series(
            np.array([99.68951, 103.66742, 101.70822]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.final_value(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.final_value(group_by=False),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.final_value(group_by=False),
            pd.Series(
                np.array([199.68951, 203.66742, 101.70822]),
                index=price_na.columns
            )
        )
        result = pd.Series(
            np.array([203.35693, 101.70822]),
            index=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_series_equal(
            portfolio.final_value(group_by=group_by),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.final_value(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.final_value(),
            result
        )

    def test_total_profit(self):
        result = pd.Series(
            np.array([-0.31049, 3.66742, 1.70822]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.total_profit(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.total_profit(group_by=False),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.total_profit(group_by=False),
            result
        )
        result = pd.Series(
            np.array([3.35693, 1.70822]),
            index=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_series_equal(
            portfolio.total_profit(group_by=group_by),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.total_profit(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.total_profit(),
            result
        )

    def test_total_return(self):
        result = pd.Series(
            np.array([-0.0031049, 0.0366742, 0.0170822]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.total_return(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.total_return(group_by=False),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.total_return(group_by=False),
            pd.Series(
                np.array([-0.00155245, 0.0183371, 0.0170822]),
                index=price_na.columns
            )
        )
        result = pd.Series(
            np.array([0.01678465, 0.0170822]),
            index=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_series_equal(
            portfolio.total_return(group_by=group_by),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.total_return(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.total_return(),
            result
        )

    def test_buy_and_hold_return(self):
        result = pd.Series(
            np.array([1.5, 4.0, 3.0]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.buy_and_hold_return(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.buy_and_hold_return(group_by=False),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.buy_and_hold_return(group_by=False),
            result
        )
        result = pd.Series(
            np.array([2.75, 3.00]),
            index=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_series_equal(
            portfolio.buy_and_hold_return(group_by=group_by),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.buy_and_hold_return(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.buy_and_hold_return(),
            result
        )

    def test_active_returns(self):
        result = pd.DataFrame(
            np.array([
                [0., -0.10722257, -0.10722257],
                [-0.34214854, 0.68709069, 0.68709069],
                [-0.02985, np.nan, 0.42740909],
                [0., np.nan, -0.02653333],
                [-0.03855399, 0.08689745, 0.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.active_returns(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.active_returns(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.active_returns(group_by=False),
            result
        )
        result = pd.DataFrame(
            np.array([
                [-0.10722257, -0.10722257],
                [0.49250019, 0.68709069],
                [np.nan, 0.42740909],
                [np.nan, -0.02653333],
                [0.04159433, 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_frame_equal(
            portfolio.active_returns(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.active_returns(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.active_returns(),
            result
        )

    def test_returns(self):
        result = pd.DataFrame(
            np.array([
                [0.00000000e+00, -1.20100000e-03, -1.20100000e-03],
                [-1.04020000e-03, 8.97057366e-03, 8.97057366e-03],
                [-5.97621646e-05, np.nan, 9.33060570e-03],
                [0.00000000e+00, np.nan, -7.82569695e-05],
                [-2.00720773e-03, 7.77210232e-03, 0.00000000e+00]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.returns(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.returns(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.returns(group_by=False),
            pd.DataFrame(
                np.array([
                    [0.00000000e+00, -6.00500000e-04, -1.20100000e-03],
                    [-5.20100000e-04, 4.48259180e-03, 8.97057366e-03],
                    [-2.98655331e-05, np.nan, 9.33060570e-03],
                    [0.00000000e+00, np.nan, -7.82569695e-05],
                    [-1.00305163e-03, 3.94098781e-03, 0.00000000e+00]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.returns(group_by=False, in_sim_order=True),
            pd.DataFrame(
                np.array([
                    [0.00000000e+00, -6.00500000e-04, -1.20100000e-03],
                    [-5.18090121e-04, 4.48259180e-03, 8.97057366e-03],
                    [np.nan, np.nan, 9.33060570e-03],
                    [0.00000000e+00, np.nan, -7.82569695e-05],
                    [-9.84980013e-04, 3.94312568e-03, 0.00000000e+00]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [-6.00500000e-04, -1.20100000e-03],
                [3.96217929e-03, 8.97057366e-03],
                [np.nan, 9.33060570e-03],
                [np.nan, -7.82569695e-05],
                [2.95426176e-03, 0.00000000e+00]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object')
        )
        pd.testing.assert_frame_equal(
            portfolio.returns(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.returns(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.returns(),
            result
        )

    def test_return_method(self):
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(),
            pd.Series(
                np.array([13.94345369, 10.2576347]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(risk_free=0.01),
            pd.Series(
                np.array([-52.28636182, -19.87302406]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(year_freq='365D'),
            pd.Series(
                np.array([16.78094911, 12.34506527]),
                index=pd.Index(['first', 'second'], dtype='object')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(group_by=False),
            pd.Series(
                np.array([-11.05899826, 14.82902058, 10.2576347]),
                index=price_na.columns
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(group_by=False, active_returns=True),
            pd.Series(
                np.array([-8.90341357, 8.52024702, 9.03964722]),
                index=price_na.columns
            )
        )

    def test_stats(self):
        pd.testing.assert_series_equal(
            portfolio.stats(),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 66.66666666666667,
                    1.688383333333339, 1.688383333333339, 283.3333333333333,
                    0.1061052323180044, 0.1061052323180044,
                    pd.Timedelta('3 days 00:00:00'), pd.Timedelta('3 days 00:00:00'),
                    1.3333333333333333, 66.66666666666667, 71.65627811190701,
                    69.59395190344144, 70.62511500767424, pd.Timedelta('2 days 08:00:00'),
                    pd.Timedelta('2 days 04:00:00'), 0.30223151515151503,
                    1.2350921335789002, 4.664569696591953, 69.7516711230091,
                    8590.91490584092
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Holding Duration [%]', 'Total Profit',
                    'Total Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]',
                    'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
                    'Num. Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                    'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
                    'Expectancy', 'SQN', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='<lambda>')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].stats(),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 40.0, -0.3104900000000015,
                    -0.3104900000000015, 150.0, 0.3104900000000015, 0.3104900000000015,
                    pd.Timedelta('4 days 00:00:00'), pd.Timedelta('4 days 00:00:00'), 1, 0.0,
                    -36.17854088546808, -36.17854088546808, -36.17854088546808,
                    pd.Timedelta('1 days 00:00:00'), pd.Timedelta('1 days 00:00:00'),
                    -0.10999000000000006, np.nan, -11.057783842772304, -9.75393669809172,
                    -46.721467294341814
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Holding Duration [%]', 'Total Profit',
                    'Total Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]',
                    'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
                    'Num. Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                    'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
                    'Expectancy', 'SQN', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].stats(required_return=0.1, risk_free=0.01),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 40.0, -0.3104900000000015,
                    -0.3104900000000015, 150.0, 0.3104900000000015, 0.3104900000000015,
                    pd.Timedelta('4 days 00:00:00'), pd.Timedelta('4 days 00:00:00'), 1, 0.0,
                    -36.17854088546808, -36.17854088546808, -36.17854088546808,
                    pd.Timedelta('1 days 00:00:00'), pd.Timedelta('1 days 00:00:00'),
                    -0.10999000000000006, np.nan, -188.9975847831419, -15.874008737030774,
                    -46.721467294341814
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Holding Duration [%]', 'Total Profit',
                    'Total Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]',
                    'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
                    'Num. Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                    'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
                    'Expectancy', 'SQN', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].stats(active_returns=True),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 40.0, -0.3104900000000015,
                    -0.3104900000000015, 150.0, 0.3104900000000015, 0.3104900000000015,
                    pd.Timedelta('4 days 00:00:00'), pd.Timedelta('4 days 00:00:00'), 1, 0.0,
                    -36.17854088546808, -36.17854088546808, -36.17854088546808,
                    pd.Timedelta('1 days 00:00:00'), pd.Timedelta('1 days 00:00:00'),
                    -0.10999000000000006, np.nan, -8.903413572386716, -8.433416371939986,
                    -2.5880511634968824
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Holding Duration [%]', 'Total Profit',
                    'Total Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]',
                    'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
                    'Num. Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                    'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
                    'Expectancy', 'SQN', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].stats(incl_unrealized=True),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 40.0, -0.3104900000000015,
                    -0.3104900000000015, 150.0, 0.3104900000000015, 0.3104900000000015,
                    pd.Timedelta('4 days 00:00:00'), pd.Timedelta('4 days 00:00:00'), 2, 0.0,
                    -3.8553985193731357, -36.17854088546808, -20.01696970242061,
                    pd.Timedelta('1 days 00:00:00'), pd.Timedelta('0 days 12:00:00'),
                    -0.15524499999999997, -3.4304496740691692, -11.057783842772304,
                    -9.75393669809172, -46.721467294341814
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Holding Duration [%]', 'Total Profit',
                    'Total Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]',
                    'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
                    'Num. Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                    'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
                    'Expectancy', 'SQN', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(portfolio['c'].stats(), portfolio.stats(column='c'))
        pd.testing.assert_series_equal(portfolio['c'].stats(), portfolio_grouped.stats(column='c', group_by=False))
        pd.testing.assert_series_equal(portfolio_grouped['second'].stats(), portfolio_grouped.stats(column='second'))
