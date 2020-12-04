import numpy as np
import pandas as pd
from numba import njit, typeof
from numba.typed import List
from datetime import datetime, timedelta
import pytest

import vectorbt as vbt
from vectorbt.portfolio.enums import *
from vectorbt.generic.enums import drawdown_dt
from vectorbt import settings
from vectorbt.utils.random import set_seed
from vectorbt.portfolio import nb

from tests.utils import record_arrays_close

seed = 42

day_dt = np.timedelta64(86400000000000)

settings.returns['year_freq'] = '252 days'  # same as empyrical

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
    log_record = np.empty(1, dtype=log_dt)[0]
    log_record[0] = 0
    log_record[1] = 0
    log_record[2] = 0
    log_record[3] = 0
    log_record[-1] = 0
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=0))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=10), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=1))

    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            -100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            np.nan, 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., np.inf, 10., 1100.,
            nb.create_order_nb(size=10, price=10), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., np.nan, 10., 1100.,
            nb.create_order_nb(size=10, price=10), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., -100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, direction=Direction.LongOnly), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, direction=Direction.ShortOnly), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=np.inf), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=-10), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fees=np.inf), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fees=-1), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fixed_fees=np.inf), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fixed_fees=-1), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, slippage=np.inf), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, slippage=-1), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, min_size=np.inf), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, min_size=-1), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, max_size=0), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, max_size=-10), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, reject_prob=np.nan), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, reject_prob=-1), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, reject_prob=2), log_record)

    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., np.nan,
        nb.create_order_nb(size=1, price=10, size_type=SizeType.TargetPercent), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=3))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., -10.,
        nb.create_order_nb(size=1, price=10, size_type=SizeType.TargetPercent), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=4))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., np.inf, 1100.,
            nb.create_order_nb(size=10, price=10, size_type=SizeType.TargetValue), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., -10., 1100.,
            nb.create_order_nb(size=10, price=10, size_type=SizeType.TargetValue), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., np.nan, 1100.,
        nb.create_order_nb(size=10, price=10, size_type=SizeType.TargetValue), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=2))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., -10., 10., 1100.,
        nb.create_order_nb(size=np.inf, price=10, direction=Direction.ShortOnly), log_record)
    assert cash_now == 100.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=6))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., -10., 10., 1100.,
        nb.create_order_nb(size=-np.inf, price=10, direction=Direction.All), log_record)
    assert cash_now == 100.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=6))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 10., 10., 1100.,
        nb.create_order_nb(size=0, price=10), log_record)
    assert cash_now == 100.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=1, status_info=5))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=15, price=10, max_size=10, allow_partial=False, raise_reject=True), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=15, price=10, max_size=10, allow_partial=False), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=9))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, reject_prob=1., raise_reject=True), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=10, price=10, reject_prob=1.), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=10))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0., 100., 10., 1100.,
        nb.create_order_nb(size=10, price=10, direction=Direction.LongOnly), log_record)
    assert cash_now == 0.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=7))
    cash_now, shares_now, order_result = nb.process_order_nb(
        0., 100., 10., 1100.,
        nb.create_order_nb(size=10, price=10, direction=Direction.All), log_record)
    assert cash_now == 0.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=7))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            np.inf, 100., 10., 1100.,
            nb.create_order_nb(size=np.inf, price=10, direction=Direction.LongOnly), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            np.inf, 100., 10., 1100.,
            nb.create_order_nb(size=np.inf, price=10, direction=Direction.All), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 1100.,
        nb.create_order_nb(size=-10, price=10, direction=Direction.ShortOnly), log_record)
    assert cash_now == 100.
    assert shares_now == 0.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=8))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            np.inf, 100., 10., 1100.,
            nb.create_order_nb(size=np.inf, price=10, direction=Direction.ShortOnly), log_record)
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            np.inf, 100., 10., 1100.,
            nb.create_order_nb(size=-np.inf, price=10, direction=Direction.All), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 1100.,
        nb.create_order_nb(size=-10, price=10, direction=Direction.LongOnly), log_record)
    assert cash_now == 100.
    assert shares_now == 0.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=8))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, fixed_fees=100, raise_reject=True), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=10, price=10, fixed_fees=100), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=11))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=10, price=10, min_size=100, raise_reject=True), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=10, price=10, min_size=100), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=12))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=100, price=10, allow_partial=False, raise_reject=True), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=100, price=10, allow_partial=False), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=13))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=-10, price=10, min_size=100, raise_reject=True), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=-10, price=10, min_size=100), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=12))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=-200, price=10, direction=Direction.LongOnly, allow_partial=False,
                               raise_reject=True),
            log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=-200, price=10, direction=Direction.LongOnly, allow_partial=False), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=13))
    with pytest.raises(Exception) as e_info:
        cash_now, shares_now, order_result = nb.process_order_nb(
            100., 100., 10., 1100.,
            nb.create_order_nb(size=-10, price=10, fixed_fees=1000, raise_reject=True), log_record)
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 100., 10., 1100.,
        nb.create_order_nb(size=-10, price=10, fixed_fees=1000), log_record)
    assert cash_now == 100.
    assert shares_now == 100.
    assert_same_tuple(order_result, OrderResult(
        size=np.nan, price=np.nan, fees=np.nan, side=-1, status=2, status_info=11))

    # Calculations
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=10, price=10, fees=0.1, fixed_fees=1, slippage=0.1), log_record)
    assert cash_now == 0.
    assert shares_now == 8.18181818181818
    assert_same_tuple(order_result, OrderResult(
        size=8.18181818181818, price=11.0, fees=10.000000000000014, side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=100, price=10, fees=0.1, fixed_fees=1, slippage=0.1), log_record)
    assert cash_now == 0.
    assert shares_now == 8.18181818181818
    assert_same_tuple(order_result, OrderResult(
        size=8.18181818181818, price=11.0, fees=10.000000000000014, side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=-10, price=10, fees=0.1, fixed_fees=1, slippage=0.1), log_record)
    assert cash_now == 180.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10.0, price=9.0, fees=10.0, side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=-100, price=10, fees=0.1, fixed_fees=1, slippage=0.1), log_record)
    assert cash_now == 909.
    assert shares_now == -100.
    assert_same_tuple(order_result, OrderResult(
        size=100.0, price=9.0, fees=91.0, side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=10, price=10, size_type=SizeType.TargetShares), log_record)
    assert cash_now == 0.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=-10, price=10, size_type=SizeType.TargetShares), log_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=100, price=10, size_type=SizeType.TargetValue), log_record)
    assert cash_now == 0.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=-100, price=10, size_type=SizeType.TargetValue), log_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=1, price=10, size_type=SizeType.TargetPercent), log_record)
    assert cash_now == 0.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=-1, price=10, size_type=SizeType.TargetPercent), log_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=np.inf, price=10), log_record)
    assert cash_now == 0.
    assert shares_now == 10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=0, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=-np.inf, price=10), log_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=10., price=10.0, fees=0., side=1, status=0, status_info=-1))
    cash_now, shares_now, order_result = nb.process_order_nb(
        150., -5., 10., 100.,
        nb.create_order_nb(size=-np.inf, price=10), log_record)
    assert cash_now == 200.
    assert shares_now == -10.
    assert_same_tuple(order_result, OrderResult(
        size=5., price=10.0, fees=0., side=1, status=0, status_info=-1))

    # Logging
    _ = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(log=True), log_record)
    assert_same_tuple(log_record, (
        0, 0, 0, 0, 100., 0., 10., 100., np.nan, 0, 2, np.nan, 0., 0., 0., 0., np.inf, 0.,
        False, True, False, True, 100., 0., np.nan, np.nan, np.nan, -1, 1, 0, 0
    ))
    _ = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=np.inf, price=10, log=True), log_record)
    assert_same_tuple(log_record, (
        0, 0, 0, 0, 100., 0., 10., 100., np.inf, 0, 2, 10., 0., 0., 0., 0., np.inf, 0.,
        False, True, False, True, 0., 10., 10., 10., 0., 0, 0, -1, 0
    ))
    _ = nb.process_order_nb(
        100., 0., 10., 100.,
        nb.create_order_nb(size=-np.inf, price=10, log=True), log_record)
    assert_same_tuple(log_record, (
        0, 0, 0, 0, 100., 0., 10., 100., -np.inf, 0, 2, 10., 0., 0., 0., 0., np.inf, 0.,
        False, True, False, True, 200., -10., 10., 10., 0., 1, 0, -1, 0
    ))


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


def from_signals_all(price=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(price, entries, exits, direction='all', **kwargs)


def from_signals_longonly(price=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(price, entries, exits, direction='longonly', **kwargs)


def from_signals_shortonly(price=price, entries=entries, exits=exits, **kwargs):
    return vbt.Portfolio.from_signals(price, entries, exits, direction='shortonly', **kwargs)


class TestFromSignals:
    def test_one_column(self):
        record_arrays_close(
            from_signals_all().order_records,
            np.array([
                (0, 0, 0, 100., 1., 0., 0), (1, 3, 0, 200., 4., 0., 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly().order_records,
            np.array([
                (0, 0, 0, 100., 1., 0., 0), (1, 3, 0, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly().order_records,
            np.array([
                (0, 0, 0, 100., 1., 0., 1), (1, 3, 0, 50., 4., 0., 0)
            ], dtype=order_dt)
        )
        portfolio = from_signals_all()
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
        record_arrays_close(
            from_signals_all(price=price_wide).order_records,
            np.array([
                (0, 0, 0, 100., 1., 0., 0), (1, 3, 0, 200., 4., 0., 1),
                (2, 0, 1, 100., 1., 0., 0), (3, 3, 1, 200., 4., 0., 1),
                (4, 0, 2, 100., 1., 0., 0), (5, 3, 2, 200., 4., 0., 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(price=price_wide).order_records,
            np.array([
                (0, 0, 0, 100., 1., 0., 0), (1, 3, 0, 100., 4., 0., 1),
                (2, 0, 1, 100., 1., 0., 0), (3, 3, 1, 100., 4., 0., 1),
                (4, 0, 2, 100., 1., 0., 0), (5, 3, 2, 100., 4., 0., 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(price=price_wide).order_records,
            np.array([
                (0, 0, 0, 100., 1., 0., 1), (1, 3, 0, 50., 4., 0., 0),
                (2, 0, 1, 100., 1., 0., 1), (3, 3, 1, 50., 4., 0., 0),
                (4, 0, 2, 100., 1., 0., 1), (5, 3, 2, 50., 4., 0., 0)
            ], dtype=order_dt)
        )
        portfolio = from_signals_all(price=price_wide)
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

    def test_size(self):
        record_arrays_close(
            from_signals_all(size=[[-1, 0, 1, np.inf]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 2.0, 4.0, 0.0, 1), (2, 0, 2, 1.0, 1.0, 0.0, 0),
                (3, 3, 2, 2.0, 4.0, 0.0, 1), (4, 0, 3, 100.0, 1.0, 0.0, 0), (5, 3, 3, 200.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=[[-1, 0, 1, np.inf]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 1.0, 4.0, 0.0, 1), (2, 0, 2, 1.0, 1.0, 0.0, 0),
                (3, 3, 2, 1.0, 4.0, 0.0, 1), (4, 0, 3, 100.0, 1.0, 0.0, 0), (5, 3, 3, 100.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=[[-1, 0, 1, np.inf]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 3, 0, 1.0, 4.0, 0.0, 0), (2, 0, 2, 1.0, 1.0, 0.0, 1),
                (3, 3, 2, 1.0, 4.0, 0.0, 0), (4, 0, 3, 100.0, 1.0, 0.0, 1), (5, 3, 3, 50.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_price(self):
        record_arrays_close(
            from_signals_all(price=price * 1.01).order_records,
            np.array([
                (0, 0, 0, 99.00990099009901, 1.01, 0.0, 0), (1, 3, 0, 198.01980198019803, 4.04, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(price=price * 1.01).order_records,
            np.array([
                (0, 0, 0, 99.00990099, 1.01, 0., 0), (1, 3, 0, 99.00990099, 4.04, 0., 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(price=price * 1.01).order_records,
            np.array([
                (0, 0, 0, 99.00990099009901, 1.01, 0.0, 1), (1, 3, 0, 49.504950495049506, 4.04, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_fees(self):
        record_arrays_close(
            from_signals_all(size=1, fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 2.0, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.0, 0.1, 0),
                (3, 3, 1, 2.0, 4.0, 0.8, 1), (4, 0, 2, 1.0, 1.0, 1.0, 0), (5, 3, 2, 2.0, 4.0, 8.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1, fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 1.0, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.0, 0.1, 0),
                (3, 3, 1, 1.0, 4.0, 0.4, 1), (4, 0, 2, 1.0, 1.0, 1.0, 0), (5, 3, 2, 1.0, 4.0, 4.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=1, fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 3, 0, 1.0, 4.0, 0.0, 0), (2, 0, 1, 1.0, 1.0, 0.1, 1),
                (3, 3, 1, 1.0, 4.0, 0.4, 0), (4, 0, 2, 1.0, 1.0, 1.0, 1), (5, 3, 2, 1.0, 4.0, 4.0, 0)
            ], dtype=order_dt)
        )

    def test_fixed_fees(self):
        record_arrays_close(
            from_signals_all(size=1, fixed_fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 2.0, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.0, 0.1, 0),
                (3, 3, 1, 2.0, 4.0, 0.1, 1), (4, 0, 2, 1.0, 1.0, 1.0, 0), (5, 3, 2, 2.0, 4.0, 1.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1, fixed_fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 1.0, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.0, 0.1, 0),
                (3, 3, 1, 1.0, 4.0, 0.1, 1), (4, 0, 2, 1.0, 1.0, 1.0, 0), (5, 3, 2, 1.0, 4.0, 1.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=1, fixed_fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 3, 0, 1.0, 4.0, 0.0, 0), (2, 0, 1, 1.0, 1.0, 0.1, 1),
                (3, 3, 1, 1.0, 4.0, 0.1, 0), (4, 0, 2, 1.0, 1.0, 1.0, 1), (5, 3, 2, 1.0, 4.0, 1.0, 0)
            ], dtype=order_dt)
        )

    def test_slippage(self):
        record_arrays_close(
            from_signals_all(size=1, slippage=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 2.0, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.1, 0.0, 0),
                (3, 3, 1, 2.0, 3.6, 0.0, 1), (4, 0, 2, 1.0, 2.0, 0.0, 0), (5, 3, 2, 2.0, 0.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1, slippage=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 1.0, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.1, 0.0, 0),
                (3, 3, 1, 1.0, 3.6, 0.0, 1), (4, 0, 2, 1.0, 2.0, 0.0, 0), (5, 3, 2, 1.0, 0.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=1, slippage=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 3, 0, 1.0, 4.0, 0.0, 0), (2, 0, 1, 1.0, 0.9, 0.0, 1),
                (3, 3, 1, 1.0, 4.4, 0.0, 0), (4, 0, 2, 1.0, 0.0, 0.0, 1), (5, 3, 2, 1.0, 8.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_min_size(self):
        record_arrays_close(
            from_signals_all(size=1, min_size=[[0., 1., 2.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 2.0, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.0, 0.0, 0),
                (3, 3, 1, 2.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1, min_size=[[0., 1., 2.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 1.0, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.0, 0.0, 0),
                (3, 3, 1, 1.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=1, min_size=[[0., 1., 2.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 3, 0, 1.0, 4.0, 0.0, 0), (2, 0, 1, 1.0, 1.0, 0.0, 1),
                (3, 3, 1, 1.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_max_size(self):
        record_arrays_close(
            from_signals_all(size=1, max_size=[[0.5, 1., np.inf]]).order_records,
            np.array([
                (0, 0, 0, 0.5, 1.0, 0.0, 0), (1, 3, 0, 0.5, 4.0, 0.0, 1), (2, 4, 0, 0.5, 5.0, 0.0, 1),
                (3, 0, 1, 1.0, 1.0, 0.0, 0), (4, 3, 1, 1.0, 4.0, 0.0, 1), (5, 4, 1, 1.0, 5.0, 0.0, 1),
                (6, 0, 2, 1.0, 1.0, 0.0, 0), (7, 3, 2, 2.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1, max_size=[[0.5, 1., np.inf]]).order_records,
            np.array([
                (0, 0, 0, 0.5, 1.0, 0.0, 0), (1, 3, 0, 0.5, 4.0, 0.0, 1), (2, 0, 1, 1.0, 1.0, 0.0, 0),
                (3, 3, 1, 1.0, 4.0, 0.0, 1), (4, 0, 2, 1.0, 1.0, 0.0, 0), (5, 3, 2, 1.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=1, max_size=[[0.5, 1., np.inf]]).order_records,
            np.array([
                (0, 0, 0, 0.5, 1.0, 0.0, 1), (1, 3, 0, 0.5, 4.0, 0.0, 0), (2, 0, 1, 1.0, 1.0, 0.0, 1),
                (3, 3, 1, 1.0, 4.0, 0.0, 0), (4, 0, 2, 1.0, 1.0, 0.0, 1), (5, 3, 2, 1.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_reject_prob(self):
        record_arrays_close(
            from_signals_all(size=1., reject_prob=[[0., 0.5, 1.]], seed=42).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 2.0, 4.0, 0.0, 1), (2, 1, 1, 1.0, 2.0, 0.0, 0),
                (3, 3, 1, 2.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1., reject_prob=[[0., 0.5, 1.]], seed=42).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 3, 0, 1.0, 4.0, 0.0, 1), (2, 1, 1, 1.0, 2.0, 0.0, 0),
                (3, 3, 1, 1.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=1., reject_prob=[[0., 0.5, 1.]], seed=42).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 3, 0, 1.0, 4.0, 0.0, 0), (2, 1, 1, 1.0, 2.0, 0.0, 1),
                (3, 3, 1, 1.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_close_first(self):
        record_arrays_close(
            from_signals_all(close_first=[[False, True]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 200.0, 4.0, 0.0, 1), (2, 0, 1, 100.0, 1.0, 0.0, 0),
                (3, 3, 1, 100.0, 4.0, 0.0, 1), (4, 4, 1, 80.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_all(
                price=pd.Series(price.values[::-1], index=price.index),
                entries=pd.Series(entries.values[::-1], index=price.index),
                exits=pd.Series(exits.values[::-1], index=price.index),
                close_first=[[False, True]]
            ).order_records,
            np.array([
                (0, 0, 0, 20.0, 5.0, 0.0, 1), (1, 3, 0, 100.0, 2.0, 0.0, 0), (2, 0, 1, 20.0, 5.0, 0.0, 1),
                (3, 3, 1, 20.0, 2.0, 0.0, 0), (4, 4, 1, 160.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_allow_partial(self):
        record_arrays_close(
            from_signals_all(size=1000, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 1100.0, 4.0, 0.0, 1), (2, 3, 1, 1000.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1000, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 100.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=1000, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 1000.0, 1.0, 0.0, 1), (1, 3, 0, 275.0, 4.0, 0.0, 0), (2, 0, 1, 1000.0, 1.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_all(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 200.0, 4.0, 0.0, 1), (2, 0, 1, 100.0, 1.0, 0.0, 0),
                (3, 3, 1, 200.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 100.0, 4.0, 0.0, 1), (2, 0, 1, 100.0, 1.0, 0.0, 0),
                (3, 3, 1, 100.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=np.inf, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 3, 0, 50.0, 4.0, 0.0, 0), (2, 0, 1, 100.0, 1.0, 0.0, 1)
            ], dtype=order_dt)
        )

    def test_raise_reject(self):
        record_arrays_close(
            from_signals_all(size=1000, allow_partial=True, raise_reject=True).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 1100.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1000, allow_partial=True, raise_reject=True).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 100.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        with pytest.raises(Exception) as e_info:
            _ = from_signals_shortonly(size=1000, allow_partial=True, raise_reject=True).order_records
        with pytest.raises(Exception) as e_info:
            _ = from_signals_all(size=1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception) as e_info:
            _ = from_signals_longonly(size=1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception) as e_info:
            _ = from_signals_shortonly(size=1000, allow_partial=False, raise_reject=True).order_records

    def test_accumulate(self):
        record_arrays_close(
            from_signals_all(size=1, accumulate=True).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 0), (2, 3, 0, 1.0, 4.0, 0.0, 1),
                (3, 4, 0, 1.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(size=1, accumulate=True).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 0), (2, 3, 0, 1.0, 4.0, 0.0, 1),
                (3, 4, 0, 1.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(size=1, accumulate=True).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_log(self):
        record_arrays_close(
            from_signals_all(log=True).log_records,
            np.array([
                (0, 0, 0, 0, 100.0, 0.0, 1.0, 100.0, np.inf, 0, 2, 1.0, 0.0, 0.0, 0.0, 1e-08, np.inf, 0.0,
                 False, True, False, True, 0.0, 100.0, 100.0, 1.0, 0.0, 0, 0, -1, 0),
                (1, 3, 0, 0, 0.0, 100.0, 4.0, 400.0, -np.inf, 0, 2, 4.0, 0.0, 0.0, 0.0, 1e-08, np.inf, 0.0,
                 False, True, False, True, 800.0, -100.0, 200.0, 4.0, 0.0, 1, 0, -1, 1)
            ], dtype=log_dt)
        )

    def test_conflict_mode(self):
        kwargs = dict(
            price=price.iloc[:3],
            entries=pd.DataFrame([
                [True, True, True, True, True],
                [True, True, True, True, False],
                [True, True, True, True, True]
            ]),
            exits=pd.DataFrame([
                [True, True, True, True, True],
                [False, False, False, False, True],
                [True, True, True, True, True]
            ]),
            size=1.,
            conflict_mode=[[
                'ignore',
                'entry',
                'exit',
                'opposite',
                'opposite'
            ]]
        )
        record_arrays_close(
            from_signals_all(**kwargs).order_records,
            np.array([
                (0, 1, 0, 1.0, 2.0, 0.0, 0), (1, 0, 1, 1.0, 1.0, 0.0, 0), (2, 0, 2, 1.0, 1.0, 0.0, 1),
                (3, 1, 2, 2.0, 2.0, 0.0, 0), (4, 2, 2, 2.0, 3.0, 0.0, 1), (5, 1, 3, 1.0, 2.0, 0.0, 0),
                (6, 2, 3, 2.0, 3.0, 0.0, 1), (7, 1, 4, 1.0, 2.0, 0.0, 1), (8, 2, 4, 2.0, 3.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(**kwargs).order_records,
            np.array([
                (0, 1, 0, 1.0, 2.0, 0.0, 0), (1, 0, 1, 1.0, 1.0, 0.0, 0), (2, 1, 2, 1.0, 2.0, 0.0, 0),
                (3, 2, 2, 1.0, 3.0, 0.0, 1), (4, 1, 3, 1.0, 2.0, 0.0, 0), (5, 2, 3, 1.0, 3.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(**kwargs).order_records,
            np.array([
                (0, 1, 0, 1.0, 2.0, 0.0, 1), (1, 0, 1, 1.0, 1.0, 0.0, 1), (2, 1, 2, 1.0, 2.0, 0.0, 1),
                (3, 2, 2, 1.0, 3.0, 0.0, 0), (4, 1, 3, 1.0, 2.0, 0.0, 1), (5, 2, 3, 1.0, 3.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_init_cash(self):
        record_arrays_close(
            from_signals_all(price=price_wide, size=1., init_cash=[0., 1., 100.]).order_records,
            np.array([
                (0, 3, 0, 1.0, 4.0, 0.0, 1), (1, 0, 1, 1.0, 1.0, 0.0, 0), (2, 3, 1, 2.0, 4.0, 0.0, 1),
                (3, 0, 2, 1.0, 1.0, 0.0, 0), (4, 3, 2, 2.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_longonly(price=price_wide, size=1., init_cash=[0., 1., 100.]).order_records,
            np.array([
                (0, 0, 1, 1.0, 1.0, 0.0, 0), (1, 3, 1, 1.0, 4.0, 0.0, 1), (2, 0, 2, 1.0, 1.0, 0.0, 0),
                (3, 3, 2, 1.0, 4.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_signals_shortonly(price=price_wide, size=1., init_cash=[0., 1., 100.]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 3, 0, 0.25, 4.0, 0.0, 0), (2, 0, 1, 1.0, 1.0, 0.0, 1),
                (3, 3, 1, 0.5, 4.0, 0.0, 0), (4, 0, 2, 1.0, 1.0, 0.0, 1), (5, 3, 2, 1.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )
        with pytest.raises(Exception) as e_info:
            _ = from_signals_all(init_cash=np.inf).order_records
        with pytest.raises(Exception) as e_info:
            _ = from_signals_longonly(init_cash=np.inf).order_records
        with pytest.raises(Exception) as e_info:
            _ = from_signals_shortonly(init_cash=np.inf).order_records

    def test_group_by(self):
        portfolio = from_signals_all(price=price_wide, group_by=np.array([0, 0, 1]))
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 200.0, 4.0, 0.0, 1), (2, 0, 1, 100.0, 1.0, 0.0, 0),
                (3, 3, 1, 200.0, 4.0, 0.0, 1), (4, 0, 2, 100.0, 1.0, 0.0, 0), (5, 3, 2, 200.0, 4.0, 0.0, 1)
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
        portfolio = from_signals_all(price=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 200.0, 4.0, 0.0, 1), (2, 3, 1, 200.0, 4.0, 0.0, 1),
                (3, 0, 2, 100.0, 1.0, 0.0, 0), (4, 3, 2, 200.0, 4.0, 0.0, 1)
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
        with pytest.raises(Exception) as e_info:
            _ = portfolio.regroup(group_by=False)

    def test_call_seq(self):
        portfolio = from_signals_all(price=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 3, 0, 200.0, 4.0, 0.0, 1), (2, 3, 1, 200.0, 4.0, 0.0, 1),
                (3, 0, 2, 100.0, 1.0, 0.0, 0), (4, 3, 2, 200.0, 4.0, 0.0, 1)
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
        portfolio = from_signals_all(
            price=price_wide, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq='reversed')
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 3, 1, 200.0, 4.0, 0.0, 1), (2, 3, 0, 200.0, 4.0, 0.0, 1),
                (3, 0, 2, 100.0, 1.0, 0.0, 0), (4, 3, 2, 200.0, 4.0, 0.0, 1)
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
        portfolio = from_signals_all(
            price=price_wide, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq='random', seed=seed)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 3, 1, 200.0, 4.0, 0.0, 1), (2, 3, 0, 200.0, 4.0, 0.0, 1),
                (3, 0, 2, 100.0, 1.0, 0.0, 0), (4, 3, 2, 200.0, 4.0, 0.0, 1)
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
        kwargs = dict(
            price=1.,
            entries=pd.DataFrame([
                [False, False, True],
                [False, True, False],
                [True, False, False],
                [False, False, True],
                [False, True, False],
            ]),
            exits=pd.DataFrame([
                [False, False, False],
                [False, False, True],
                [False, True, False],
                [True, False, False],
                [False, False, True],
            ]),
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq='auto'
        )
        portfolio = from_signals_all(**kwargs)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 2, 100.0, 1.0, 0.0, 0), (1, 1, 2, 200.0, 1.0, 0.0, 1), (2, 1, 1, 200.0, 1.0, 0.0, 0),
                (3, 2, 1, 400.0, 1.0, 0.0, 1), (4, 2, 0, 400.0, 1.0, 0.0, 0), (5, 3, 0, 800.0, 1.0, 0.0, 1),
                (6, 3, 2, 800.0, 1.0, 0.0, 0), (7, 4, 2, 1400.0, 1.0, 0.0, 1), (8, 4, 1, 1400.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 2],
                [2, 0, 1],
                [1, 2, 0],
                [0, 1, 2],
                [2, 0, 1]
            ])
        )
        portfolio = from_signals_longonly(**kwargs)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 2, 100.0, 1.0, 0.0, 0), (1, 1, 2, 100.0, 1.0, 0.0, 1), (2, 1, 1, 100.0, 1.0, 0.0, 0),
                (3, 2, 1, 100.0, 1.0, 0.0, 1), (4, 2, 0, 100.0, 1.0, 0.0, 0), (5, 3, 0, 100.0, 1.0, 0.0, 1),
                (6, 3, 2, 100.0, 1.0, 0.0, 0), (7, 4, 2, 100.0, 1.0, 0.0, 1), (8, 4, 1, 100.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 2],
                [2, 0, 1],
                [1, 2, 0],
                [0, 1, 2],
                [2, 0, 1]
            ])
        )
        portfolio = from_signals_shortonly(**kwargs)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 2, 100.0, 1.0, 0.0, 1), (1, 1, 1, 200.0, 1.0, 0.0, 1), (2, 1, 2, 100.0, 1.0, 0.0, 0),
                (3, 2, 0, 300.0, 1.0, 0.0, 1), (4, 2, 1, 200.0, 1.0, 0.0, 0), (5, 3, 2, 400.0, 1.0, 0.0, 1),
                (6, 3, 0, 300.0, 1.0, 0.0, 0), (7, 4, 1, 500.0, 1.0, 0.0, 1), (8, 4, 2, 400.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [2, 0, 1],
                [1, 0, 2],
                [0, 2, 1],
                [2, 1, 0],
                [1, 0, 2]
            ])
        )


# ############# from_orders ############# #

order_size = pd.Series([np.inf, -np.inf, np.nan, np.inf, -np.inf], index=price.index)
order_size_wide = order_size.vbt.tile(3, keys=['a', 'b', 'c'])
order_size_one = pd.Series([1, -1, np.nan, 1, -1], index=price.index)


def from_orders_all(price=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(price, size, direction='all', **kwargs)


def from_orders_longonly(price=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(price, size, direction='longonly', **kwargs)


def from_orders_shortonly(price=price, size=order_size, **kwargs):
    return vbt.Portfolio.from_orders(price, size, direction='shortonly', **kwargs)


class TestFromOrders:
    def test_one_column(self):
        record_arrays_close(
            from_orders_all().order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1), (2, 3, 0, 100.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly().order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 100.0, 2.0, 0.0, 1), (2, 3, 0, 50.0, 4.0, 0.0, 0),
                (3, 4, 0, 50.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly().order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 1, 0, 100.0, 2.0, 0.0, 0)
            ], dtype=order_dt)
        )
        portfolio = from_orders_all()
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
        record_arrays_close(
            from_orders_all(price=price_wide).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1), (2, 3, 0, 100.0, 4.0, 0.0, 0),
                (3, 0, 1, 100.0, 1.0, 0.0, 0), (4, 1, 1, 200.0, 2.0, 0.0, 1), (5, 3, 1, 100.0, 4.0, 0.0, 0),
                (6, 0, 2, 100.0, 1.0, 0.0, 0), (7, 1, 2, 200.0, 2.0, 0.0, 1), (8, 3, 2, 100.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(price=price_wide).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 100.0, 2.0, 0.0, 1), (2, 3, 0, 50.0, 4.0, 0.0, 0),
                (3, 4, 0, 50.0, 5.0, 0.0, 1), (4, 0, 1, 100.0, 1.0, 0.0, 0), (5, 1, 1, 100.0, 2.0, 0.0, 1),
                (6, 3, 1, 50.0, 4.0, 0.0, 0), (7, 4, 1, 50.0, 5.0, 0.0, 1), (8, 0, 2, 100.0, 1.0, 0.0, 0),
                (9, 1, 2, 100.0, 2.0, 0.0, 1), (10, 3, 2, 50.0, 4.0, 0.0, 0), (11, 4, 2, 50.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(price=price_wide).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 1, 0, 100.0, 2.0, 0.0, 0), (2, 0, 1, 100.0, 1.0, 0.0, 1),
                (3, 1, 1, 100.0, 2.0, 0.0, 0), (4, 0, 2, 100.0, 1.0, 0.0, 1), (5, 1, 2, 100.0, 2.0, 0.0, 0)
            ], dtype=order_dt)
        )
        portfolio = from_orders_all(price=price_wide)
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

    def test_size_inf(self):
        record_arrays_close(
            from_orders_all(size=[[np.inf, -np.inf]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 1, 100.0, 1.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=[[np.inf, -np.inf]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=[[np.inf, -np.inf]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 1)
            ], dtype=order_dt)
        )

    def test_price(self):
        record_arrays_close(
            from_orders_all(price=price * 1.01).order_records,
            np.array([
                (0, 0, 0, 99.00990099009901, 1.01, 0.0, 0), (1, 1, 0, 198.01980198019803, 2.02, 0.0, 1),
                (2, 3, 0, 99.00990099009901, 4.04, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(price=price * 1.01).order_records,
            np.array([
                (0, 0, 0, 99.00990099009901, 1.01, 0.0, 0), (1, 1, 0, 99.00990099009901, 2.02, 0.0, 1),
                (2, 3, 0, 49.504950495049506, 4.04, 0.0, 0), (3, 4, 0, 49.504950495049506, 5.05, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(price=price * 1.01).order_records,
            np.array([
                (0, 0, 0, 99.00990099009901, 1.01, 0.0, 1), (1, 1, 0, 99.00990099009901, 2.02, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_fees(self):
        record_arrays_close(
            from_orders_all(size=order_size_one, fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.0, 0.1, 0), (5, 1, 1, 1.0, 2.0, 0.2, 1),
                (6, 3, 1, 1.0, 4.0, 0.4, 0), (7, 4, 1, 1.0, 5.0, 0.5, 1), (8, 0, 2, 1.0, 1.0, 1.0, 0),
                (9, 1, 2, 1.0, 2.0, 2.0, 1), (10, 3, 2, 1.0, 4.0, 4.0, 0), (11, 4, 2, 1.0, 5.0, 5.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size_one, fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.0, 0.1, 0), (5, 1, 1, 1.0, 2.0, 0.2, 1),
                (6, 3, 1, 1.0, 4.0, 0.4, 0), (7, 4, 1, 1.0, 5.0, 0.5, 1), (8, 0, 2, 1.0, 1.0, 1.0, 0),
                (9, 1, 2, 1.0, 2.0, 2.0, 1), (10, 3, 2, 1.0, 4.0, 4.0, 0), (11, 4, 2, 1.0, 5.0, 5.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size_one, fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 1, 0, 1.0, 2.0, 0.0, 0), (2, 3, 0, 1.0, 4.0, 0.0, 1),
                (3, 4, 0, 1.0, 5.0, 0.0, 0), (4, 0, 1, 1.0, 1.0, 0.1, 1), (5, 1, 1, 1.0, 2.0, 0.2, 0),
                (6, 3, 1, 1.0, 4.0, 0.4, 1), (7, 4, 1, 1.0, 5.0, 0.5, 0), (8, 0, 2, 1.0, 1.0, 1.0, 1),
                (9, 1, 2, 1.0, 2.0, 2.0, 0), (10, 3, 2, 1.0, 4.0, 4.0, 1), (11, 4, 2, 1.0, 5.0, 5.0, 0)
            ], dtype=order_dt)
        )

    def test_fixed_fees(self):
        record_arrays_close(
            from_orders_all(size=order_size_one, fixed_fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.0, 0.1, 0), (5, 1, 1, 1.0, 2.0, 0.1, 1),
                (6, 3, 1, 1.0, 4.0, 0.1, 0), (7, 4, 1, 1.0, 5.0, 0.1, 1), (8, 0, 2, 1.0, 1.0, 1.0, 0),
                (9, 1, 2, 1.0, 2.0, 1.0, 1), (10, 3, 2, 1.0, 4.0, 1.0, 0), (11, 4, 2, 1.0, 5.0, 1.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size_one, fixed_fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.0, 0.1, 0), (5, 1, 1, 1.0, 2.0, 0.1, 1),
                (6, 3, 1, 1.0, 4.0, 0.1, 0), (7, 4, 1, 1.0, 5.0, 0.1, 1), (8, 0, 2, 1.0, 1.0, 1.0, 0),
                (9, 1, 2, 1.0, 2.0, 1.0, 1), (10, 3, 2, 1.0, 4.0, 1.0, 0), (11, 4, 2, 1.0, 5.0, 1.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size_one, fixed_fees=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 1, 0, 1.0, 2.0, 0.0, 0), (2, 3, 0, 1.0, 4.0, 0.0, 1),
                (3, 4, 0, 1.0, 5.0, 0.0, 0), (4, 0, 1, 1.0, 1.0, 0.1, 1), (5, 1, 1, 1.0, 2.0, 0.1, 0),
                (6, 3, 1, 1.0, 4.0, 0.1, 1), (7, 4, 1, 1.0, 5.0, 0.1, 0), (8, 0, 2, 1.0, 1.0, 1.0, 1),
                (9, 1, 2, 1.0, 2.0, 1.0, 0), (10, 3, 2, 1.0, 4.0, 1.0, 1), (11, 4, 2, 1.0, 5.0, 1.0, 0)
            ], dtype=order_dt)
        )

    def test_slippage(self):
        record_arrays_close(
            from_orders_all(size=order_size_one, slippage=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.1, 0.0, 0), (5, 1, 1, 1.0, 1.8, 0.0, 1),
                (6, 3, 1, 1.0, 4.4, 0.0, 0), (7, 4, 1, 1.0, 4.5, 0.0, 1), (8, 0, 2, 1.0, 2.0, 0.0, 0),
                (9, 1, 2, 1.0, 0.0, 0.0, 1), (10, 3, 2, 1.0, 8.0, 0.0, 0), (11, 4, 2, 1.0, 0.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size_one, slippage=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.1, 0.0, 0), (5, 1, 1, 1.0, 1.8, 0.0, 1),
                (6, 3, 1, 1.0, 4.4, 0.0, 0), (7, 4, 1, 1.0, 4.5, 0.0, 1), (8, 0, 2, 1.0, 2.0, 0.0, 0),
                (9, 1, 2, 1.0, 0.0, 0.0, 1), (10, 3, 2, 1.0, 8.0, 0.0, 0), (11, 4, 2, 1.0, 0.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size_one, slippage=[[0., 0.1, 1.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 1, 0, 1.0, 2.0, 0.0, 0), (2, 3, 0, 1.0, 4.0, 0.0, 1),
                (3, 4, 0, 1.0, 5.0, 0.0, 0), (4, 0, 1, 1.0, 0.9, 0.0, 1), (5, 1, 1, 1.0, 2.2, 0.0, 0),
                (6, 3, 1, 1.0, 3.6, 0.0, 1), (7, 4, 1, 1.0, 5.5, 0.0, 0), (8, 0, 2, 1.0, 0.0, 0.0, 1),
                (9, 1, 2, 1.0, 4.0, 0.0, 0), (10, 3, 2, 1.0, 0.0, 0.0, 1), (11, 4, 2, 1.0, 10.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_min_size(self):
        record_arrays_close(
            from_orders_all(size=order_size_one, min_size=[[0., 1., 2.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.0, 0.0, 0), (5, 1, 1, 1.0, 2.0, 0.0, 1),
                (6, 3, 1, 1.0, 4.0, 0.0, 0), (7, 4, 1, 1.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size_one, min_size=[[0., 1., 2.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.0, 0.0, 0), (5, 1, 1, 1.0, 2.0, 0.0, 1),
                (6, 3, 1, 1.0, 4.0, 0.0, 0), (7, 4, 1, 1.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size_one, min_size=[[0., 1., 2.]]).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 1, 0, 1.0, 2.0, 0.0, 0), (2, 3, 0, 1.0, 4.0, 0.0, 1),
                (3, 4, 0, 1.0, 5.0, 0.0, 0), (4, 0, 1, 1.0, 1.0, 0.0, 1), (5, 1, 1, 1.0, 2.0, 0.0, 0),
                (6, 3, 1, 1.0, 4.0, 0.0, 1), (7, 4, 1, 1.0, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_max_size(self):
        record_arrays_close(
            from_orders_all(size=order_size_one, max_size=[[0.5, 1., np.inf]]).order_records,
            np.array([
                (0, 0, 0, 0.5, 1.0, 0.0, 0), (1, 1, 0, 0.5, 2.0, 0.0, 1), (2, 3, 0, 0.5, 4.0, 0.0, 0),
                (3, 4, 0, 0.5, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.0, 0.0, 0), (5, 1, 1, 1.0, 2.0, 0.0, 1),
                (6, 3, 1, 1.0, 4.0, 0.0, 0), (7, 4, 1, 1.0, 5.0, 0.0, 1), (8, 0, 2, 1.0, 1.0, 0.0, 0),
                (9, 1, 2, 1.0, 2.0, 0.0, 1), (10, 3, 2, 1.0, 4.0, 0.0, 0), (11, 4, 2, 1.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size_one, max_size=[[0.5, 1., np.inf]]).order_records,
            np.array([
                (0, 0, 0, 0.5, 1.0, 0.0, 0), (1, 1, 0, 0.5, 2.0, 0.0, 1), (2, 3, 0, 0.5, 4.0, 0.0, 0),
                (3, 4, 0, 0.5, 5.0, 0.0, 1), (4, 0, 1, 1.0, 1.0, 0.0, 0), (5, 1, 1, 1.0, 2.0, 0.0, 1),
                (6, 3, 1, 1.0, 4.0, 0.0, 0), (7, 4, 1, 1.0, 5.0, 0.0, 1), (8, 0, 2, 1.0, 1.0, 0.0, 0),
                (9, 1, 2, 1.0, 2.0, 0.0, 1), (10, 3, 2, 1.0, 4.0, 0.0, 0), (11, 4, 2, 1.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size_one, max_size=[[0.5, 1., np.inf]]).order_records,
            np.array([
                (0, 0, 0, 0.5, 1.0, 0.0, 1), (1, 1, 0, 0.5, 2.0, 0.0, 0), (2, 3, 0, 0.5, 4.0, 0.0, 1),
                (3, 4, 0, 0.5, 5.0, 0.0, 0), (4, 0, 1, 1.0, 1.0, 0.0, 1), (5, 1, 1, 1.0, 2.0, 0.0, 0),
                (6, 3, 1, 1.0, 4.0, 0.0, 1), (7, 4, 1, 1.0, 5.0, 0.0, 0), (8, 0, 2, 1.0, 1.0, 0.0, 1),
                (9, 1, 2, 1.0, 2.0, 0.0, 0), (10, 3, 2, 1.0, 4.0, 0.0, 1), (11, 4, 2, 1.0, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_reject_prob(self):
        record_arrays_close(
            from_orders_all(size=order_size_one, reject_prob=[[0., 0.5, 1.]], seed=42).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 1, 1, 1.0, 2.0, 0.0, 1), (5, 3, 1, 1.0, 4.0, 0.0, 0),
                (6, 4, 1, 1.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size_one, reject_prob=[[0., 0.5, 1.]], seed=42).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 1.0, 2.0, 0.0, 1), (2, 3, 0, 1.0, 4.0, 0.0, 0),
                (3, 4, 0, 1.0, 5.0, 0.0, 1), (4, 3, 1, 1.0, 4.0, 0.0, 0), (5, 4, 1, 1.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size_one, reject_prob=[[0., 0.5, 1.]], seed=42).order_records,
            np.array([
                (0, 0, 0, 1.0, 1.0, 0.0, 1), (1, 1, 0, 1.0, 2.0, 0.0, 0), (2, 3, 0, 1.0, 4.0, 0.0, 1),
                (3, 4, 0, 1.0, 5.0, 0.0, 0), (4, 3, 1, 1.0, 4.0, 0.0, 1), (5, 4, 1, 1.0, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_close_first(self):
        close_first_size = pd.Series([np.inf, -np.inf, -np.inf, np.inf, np.inf])
        record_arrays_close(
            from_orders_all(size=close_first_size, close_first=[[False, True]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1),
                (2, 3, 0, 100.0, 4.0, 0.0, 0), (3, 0, 1, 100.0, 1.0, 0.0, 0),
                (4, 1, 1, 100.0, 2.0, 0.0, 1), (5, 2, 1, 66.66666666666667, 3.0, 0.0, 1),
                (6, 3, 1, 66.66666666666667, 4.0, 0.0, 0), (7, 4, 1, 26.666666666666664, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=close_first_size, close_first=[[False, True]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 100.0, 2.0, 0.0, 1),
                (2, 3, 0, 50.0, 4.0, 0.0, 0), (3, 0, 1, 100.0, 1.0, 0.0, 0),
                (4, 1, 1, 100.0, 2.0, 0.0, 1), (5, 3, 1, 50.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=close_first_size, close_first=[[False, True]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 1, 0, 100.0, 2.0, 0.0, 0),
                (2, 0, 1, 100.0, 1.0, 0.0, 1), (3, 1, 1, 100.0, 2.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_allow_partial(self):
        record_arrays_close(
            from_orders_all(size=order_size_one * 1000, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 1000.0, 2.0, 0.0, 1), (2, 3, 0, 500.0, 4.0, 0.0, 0),
                (3, 4, 0, 1000.0, 5.0, 0.0, 1), (4, 1, 1, 1000.0, 2.0, 0.0, 1), (5, 4, 1, 1000.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size_one * 1000, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 100.0, 2.0, 0.0, 1), (2, 3, 0, 50.0, 4.0, 0.0, 0),
                (3, 4, 0, 50.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size_one * 1000, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 1000.0, 1.0, 0.0, 1), (1, 1, 0, 550.0, 2.0, 0.0, 0), (2, 3, 0, 1000.0, 4.0, 0.0, 1),
                (3, 4, 0, 800.0, 5.0, 0.0, 0), (4, 0, 1, 1000.0, 1.0, 0.0, 1), (5, 3, 1, 1000.0, 4.0, 0.0, 1),
                (6, 4, 1, 1000.0, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_all(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1), (2, 3, 0, 100.0, 4.0, 0.0, 0),
                (3, 0, 1, 100.0, 1.0, 0.0, 0), (4, 1, 1, 200.0, 2.0, 0.0, 1), (5, 3, 1, 100.0, 4.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 100.0, 2.0, 0.0, 1), (2, 3, 0, 50.0, 4.0, 0.0, 0),
                (3, 4, 0, 50.0, 5.0, 0.0, 1), (4, 0, 1, 100.0, 1.0, 0.0, 0), (5, 1, 1, 100.0, 2.0, 0.0, 1),
                (6, 3, 1, 50.0, 4.0, 0.0, 0), (7, 4, 1, 50.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size, allow_partial=[[True, False]]).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 1), (1, 1, 0, 100.0, 2.0, 0.0, 0), (2, 0, 1, 100.0, 1.0, 0.0, 1),
                (3, 1, 1, 100.0, 2.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_raise_reject(self):
        record_arrays_close(
            from_orders_all(size=order_size_one * 1000, allow_partial=True, raise_reject=True).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 1000.0, 2.0, 0.0, 1), (2, 3, 0, 500.0, 4.0, 0.0, 0),
                (3, 4, 0, 1000.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=order_size_one * 1000, allow_partial=True, raise_reject=True).order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 100.0, 2.0, 0.0, 1), (2, 3, 0, 50.0, 4.0, 0.0, 0),
                (3, 4, 0, 50.0, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=order_size_one * 1000, allow_partial=True, raise_reject=True).order_records,
            np.array([
                (0, 0, 0, 1000.0, 1.0, 0.0, 1), (1, 1, 0, 550.0, 2.0, 0.0, 0), (2, 3, 0, 1000.0, 4.0, 0.0, 1),
                (3, 4, 0, 800.0, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )
        with pytest.raises(Exception) as e_info:
            _ = from_orders_all(size=order_size_one * 1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception) as e_info:
            _ = from_orders_longonly(size=order_size_one * 1000, allow_partial=False, raise_reject=True).order_records
        with pytest.raises(Exception) as e_info:
            _ = from_orders_shortonly(size=order_size_one * 1000, allow_partial=False, raise_reject=True).order_records

    def test_log(self):
        record_arrays_close(
            from_orders_all(log=True).log_records,
            np.array([
                (0, 0, 0, 0, 100.0, 0.0, 1.0, 100.0, np.inf, 0, 2, 1.0, 0.0, 0.0, 0.0, 1e-08, np.inf, 0.0,
                 False, True, False, True, 0.0, 100.0, 100.0, 1.0, 0.0, 0, 0, -1, 0),
                (1, 1, 0, 0, 0.0, 100.0, 2.0, 200.0, -np.inf, 0, 2, 2.0, 0.0, 0.0, 0.0, 1e-08, np.inf, 0.0,
                 False, True, False, True, 400.0, -100.0, 200.0, 2.0, 0.0, 1, 0, -1, 1),
                (2, 2, 0, 0, 400.0, -100.0, 3.0, 100.0, np.nan, 0, 2, 3.0, 0.0, 0.0, 0.0, 1e-08, np.inf, 0.0,
                 False, True, False, True, 400.0, -100.0, np.nan, np.nan, np.nan, -1, 1, 0, -1),
                (3, 3, 0, 0, 400.0, -100.0, 4.0, 0.0, np.inf, 0, 2, 4.0, 0.0, 0.0, 0.0, 1e-08, np.inf, 0.0,
                 False, True, False, True, 0.0, 0.0, 100.0, 4.0, 0.0, 0, 0, -1, 2),
                (4, 4, 0, 0, 0.0, 0.0, 5.0, 0.0, -np.inf, 0, 2, 5.0, 0.0, 0.0, 0.0, 1e-08, np.inf, 0.0,
                 False, True, False, True, 0.0, 0.0, np.nan, np.nan, np.nan, -1, 2, 6, -1)
            ], dtype=log_dt)
        )

    def test_group_by(self):
        portfolio = from_orders_all(price=price_wide, group_by=np.array([0, 0, 1]))
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1), (2, 3, 0, 100.0, 4.0, 0.0, 0),
                (3, 0, 1, 100.0, 1.0, 0.0, 0), (4, 1, 1, 200.0, 2.0, 0.0, 1), (5, 3, 1, 100.0, 4.0, 0.0, 0),
                (6, 0, 2, 100.0, 1.0, 0.0, 0), (7, 1, 2, 200.0, 2.0, 0.0, 1), (8, 3, 2, 100.0, 4.0, 0.0, 0)
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
        portfolio = from_orders_all(price=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1), (2, 1, 1, 200.0, 2.0, 0.0, 1),
                (3, 3, 0, 200.0, 4.0, 0.0, 0), (4, 4, 0, 200.0, 5.0, 0.0, 1), (5, 0, 2, 100.0, 1.0, 0.0, 0),
                (6, 1, 2, 200.0, 2.0, 0.0, 1), (7, 3, 2, 100.0, 4.0, 0.0, 0)
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
        with pytest.raises(Exception) as e_info:
            _ = portfolio.regroup(group_by=False)

    def test_call_seq(self):
        portfolio = from_orders_all(price=price_wide, group_by=np.array([0, 0, 1]), cash_sharing=True)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1), (2, 1, 1, 200.0, 2.0, 0.0, 1),
                (3, 3, 0, 200.0, 4.0, 0.0, 0), (4, 4, 0, 200.0, 5.0, 0.0, 1), (5, 0, 2, 100.0, 1.0, 0.0, 0),
                (6, 1, 2, 200.0, 2.0, 0.0, 1), (7, 3, 2, 100.0, 4.0, 0.0, 0)
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
        portfolio = from_orders_all(
            price=price_wide, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq='reversed')
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 1, 1, 200.0, 2.0, 0.0, 1), (2, 1, 0, 200.0, 2.0, 0.0, 1),
                (3, 3, 1, 200.0, 4.0, 0.0, 0), (4, 4, 1, 200.0, 5.0, 0.0, 1), (5, 0, 2, 100.0, 1.0, 0.0, 0),
                (6, 1, 2, 200.0, 2.0, 0.0, 1), (7, 3, 2, 100.0, 4.0, 0.0, 0)
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
        portfolio = from_orders_all(
            price=price_wide, group_by=np.array([0, 0, 1]),
            cash_sharing=True, call_seq='random', seed=seed)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 1, 1, 200.0, 2.0, 0.0, 1), (2, 3, 1, 100.0, 4.0, 0.0, 0),
                (3, 0, 2, 100.0, 1.0, 0.0, 0), (4, 1, 2, 200.0, 2.0, 0.0, 1), (5, 3, 2, 100.0, 4.0, 0.0, 0)
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
        kwargs = dict(
            price=1.,
            size=pd.DataFrame([
                [0., 0., np.inf],
                [0., np.inf, -np.inf],
                [np.inf, -np.inf, 0.],
                [-np.inf, 0., np.inf],
                [0., np.inf, -np.inf],
            ]),
            group_by=np.array([0, 0, 0]),
            cash_sharing=True,
            call_seq='auto'
        )
        portfolio = from_orders_all(**kwargs)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 2, 100.0, 1.0, 0.0, 0), (1, 1, 2, 200.0, 1.0, 0.0, 1), (2, 1, 1, 200.0, 1.0, 0.0, 0),
                (3, 2, 1, 400.0, 1.0, 0.0, 1), (4, 2, 0, 400.0, 1.0, 0.0, 0), (5, 3, 0, 800.0, 1.0, 0.0, 1),
                (6, 3, 2, 800.0, 1.0, 0.0, 0), (7, 4, 2, 1400.0, 1.0, 0.0, 1), (8, 4, 1, 1400.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 2],
                [2, 0, 1],
                [1, 2, 0],
                [0, 1, 2],
                [2, 0, 1]
            ])
        )
        portfolio = from_orders_longonly(**kwargs)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 2, 100.0, 1.0, 0.0, 0), (1, 1, 2, 100.0, 1.0, 0.0, 1), (2, 1, 1, 100.0, 1.0, 0.0, 0),
                (3, 2, 1, 100.0, 1.0, 0.0, 1), (4, 2, 0, 100.0, 1.0, 0.0, 0), (5, 3, 0, 100.0, 1.0, 0.0, 1),
                (6, 3, 2, 100.0, 1.0, 0.0, 0), (7, 4, 2, 100.0, 1.0, 0.0, 1), (8, 4, 1, 100.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [0, 1, 2],
                [2, 0, 1],
                [1, 2, 0],
                [0, 1, 2],
                [2, 0, 1]
            ])
        )
        portfolio = from_orders_shortonly(**kwargs)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 2, 100.0, 1.0, 0.0, 1), (1, 1, 1, 200.0, 1.0, 0.0, 1), (2, 1, 2, 100.0, 1.0, 0.0, 0),
                (3, 2, 0, 300.0, 1.0, 0.0, 1), (4, 2, 1, 200.0, 1.0, 0.0, 0), (5, 3, 2, 400.0, 1.0, 0.0, 1),
                (6, 3, 0, 300.0, 1.0, 0.0, 0), (7, 4, 1, 500.0, 1.0, 0.0, 1), (8, 4, 2, 400.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )
        np.testing.assert_array_equal(
            portfolio.call_seq.values,
            np.array([
                [2, 0, 1],
                [1, 0, 2],
                [0, 2, 1],
                [2, 1, 0],
                [1, 0, 2]
            ])
        )

    def test_target_shares(self):
        record_arrays_close(
            from_orders_all(size=[[75., -75.]], size_type='targetshares').order_records,
            np.array([
                (0, 0, 0, 75.0, 1.0, 0.0, 0), (1, 0, 1, 75.0, 1.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=[[75., -75.]], size_type='targetshares').order_records,
            np.array([
                (0, 0, 0, 75.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=[[75., -75.]], size_type='targetshares').order_records,
            np.array([
                (0, 0, 0, 75.0, 1.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_all(
                price=price_wide, size=75., size_type='targetshares',
                group_by=np.array([0, 0, 0]), cash_sharing=True).order_records,
            np.array([
                (0, 0, 0, 75.0, 1.0, 0.0, 0), (1, 0, 1, 25.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_target_value(self):
        record_arrays_close(
            from_orders_all(size=[[50., -50.]], size_type='targetvalue').order_records,
            np.array([
                (0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 1, 0, 25.0, 2.0, 0.0, 1),
                (2, 2, 0, 8.333333333333332, 3.0, 0.0, 1), (3, 3, 0, 4.166666666666668, 4.0, 0.0, 1),
                (4, 4, 0, 2.5, 5.0, 0.0, 1), (5, 0, 1, 50.0, 1.0, 0.0, 1),
                (6, 1, 1, 25.0, 2.0, 0.0, 0), (7, 2, 1, 8.333333333333332, 3.0, 0.0, 0),
                (8, 3, 1, 4.166666666666668, 4.0, 0.0, 0), (9, 4, 1, 2.5, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=[[50., -50.]], size_type='targetvalue').order_records,
            np.array([
                (0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 1, 0, 25.0, 2.0, 0.0, 1),
                (2, 2, 0, 8.333333333333332, 3.0, 0.0, 1), (3, 3, 0, 4.166666666666668, 4.0, 0.0, 1),
                (4, 4, 0, 2.5, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=[[50., -50.]], size_type='targetvalue').order_records,
            np.array([
                (0, 0, 0, 50.0, 1.0, 0.0, 1), (1, 1, 0, 25.0, 2.0, 0.0, 0),
                (2, 2, 0, 8.333333333333332, 3.0, 0.0, 0), (3, 3, 0, 4.166666666666668, 4.0, 0.0, 0),
                (4, 4, 0, 2.5, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_all(
                price=price_wide, size=50., size_type='targetvalue',
                group_by=np.array([0, 0, 0]), cash_sharing=True).order_records,
            np.array([
                (0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 0, 1, 50.0, 1.0, 0.0, 0),
                (2, 1, 0, 25.0, 2.0, 0.0, 1), (3, 1, 1, 25.0, 2.0, 0.0, 1),
                (4, 1, 2, 25.0, 2.0, 0.0, 0), (5, 2, 0, 8.333333333333332, 3.0, 0.0, 1),
                (6, 2, 1, 8.333333333333332, 3.0, 0.0, 1), (7, 2, 2, 8.333333333333332, 3.0, 0.0, 1),
                (8, 3, 0, 4.166666666666668, 4.0, 0.0, 1), (9, 3, 1, 4.166666666666668, 4.0, 0.0, 1),
                (10, 3, 2, 4.166666666666668, 4.0, 0.0, 1), (11, 4, 0, 2.5, 5.0, 0.0, 1),
                (12, 4, 1, 2.5, 5.0, 0.0, 1), (13, 4, 2, 2.5, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )

    def test_target_percent(self):
        record_arrays_close(
            from_orders_all(size=[[0.5, -0.5]], size_type='targetpercent').order_records,
            np.array([
                (0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 1, 0, 12.5, 2.0, 0.0, 1), (2, 2, 0, 6.25, 3.0, 0.0, 1),
                (3, 3, 0, 3.90625, 4.0, 0.0, 1), (4, 4, 0, 2.734375, 5.0, 0.0, 1), (5, 0, 1, 50.0, 1.0, 0.0, 1),
                (6, 1, 1, 37.5, 2.0, 0.0, 0), (7, 2, 1, 6.25, 3.0, 0.0, 0), (8, 3, 1, 2.34375, 4.0, 0.0, 0),
                (9, 4, 1, 1.171875, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_longonly(size=[[0.5, -0.5]], size_type='targetpercent').order_records,
            np.array([
                (0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 1, 0, 12.5, 2.0, 0.0, 1), (2, 2, 0, 6.25, 3.0, 0.0, 1),
                (3, 3, 0, 3.90625, 4.0, 0.0, 1), (4, 4, 0, 2.734375, 5.0, 0.0, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_shortonly(size=[[0.5, -0.5]], size_type='targetpercent').order_records,
            np.array([
                (0, 0, 0, 50.0, 1.0, 0.0, 1), (1, 1, 0, 37.5, 2.0, 0.0, 0), (2, 2, 0, 6.25, 3.0, 0.0, 0),
                (3, 3, 0, 2.34375, 4.0, 0.0, 0), (4, 4, 0, 1.171875, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            from_orders_all(
                price=price_wide, size=0.5, size_type='targetpercent',
                group_by=np.array([0, 0, 0]), cash_sharing=True).order_records,
            np.array([
                (0, 0, 0, 50.0, 1.0, 0.0, 0), (1, 0, 1, 50.0, 1.0, 0.0, 0)
            ], dtype=order_dt)
        )

    def test_auto_seq(self):
        target_hold_value = pd.DataFrame({
            'a': [0., 70., 30., 0., 70.],
            'b': [30., 0., 70., 30., 30.],
            'c': [70., 30., 0., 70., 0.]
        }, index=price.index)
        pd.testing.assert_frame_equal(
            from_orders_all(
                price=1., size=target_hold_value, size_type='targetvalue',
                group_by=np.array([0, 0, 0]), cash_sharing=True,
                call_seq='auto').holding_value(group_by=False),
            target_hold_value
        )
        pd.testing.assert_frame_equal(
            from_orders_all(
                price=1., size=target_hold_value / 100, size_type='targetpercent',
                group_by=np.array([0, 0, 0]), cash_sharing=True,
                call_seq='auto').holding_value(group_by=False),
            target_hold_value
        )


# ############# from_order_func ############# #

@njit
def order_func_nb(oc, size):
    return nb.create_order_nb(size=size if oc.i % 2 == 0 else -size, price=oc.close[oc.i, oc.col])


class TestFromOrderFunc:
    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_one_column(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(price.tolist(), order_func_nb, np.inf, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1),
                (2, 2, 0, 133.33333333333334, 3.0, 0.0, 0), (3, 3, 0, 66.66666666666669, 4.0, 0.0, 1),
                (4, 4, 0, 53.33333333333335, 5.0, 0.0, 0)
            ], dtype=order_dt)
        )
        portfolio = vbt.Portfolio.from_order_func(price, order_func_nb, np.inf, row_wise=test_row_wise)
        record_arrays_close(
            portfolio.order_records,
            np.array([
                (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1),
                (2, 2, 0, 133.33333333333334, 3.0, 0.0, 0), (3, 3, 0, 66.66666666666669, 4.0, 0.0, 1),
                (4, 4, 0, 53.33333333333335, 5.0, 0.0, 0)
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
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 1, 100.0, 1.0, 0.0, 0),
                    (2, 0, 2, 100.0, 1.0, 0.0, 0), (3, 1, 0, 200.0, 2.0, 0.0, 1),
                    (4, 1, 1, 200.0, 2.0, 0.0, 1), (5, 1, 2, 200.0, 2.0, 0.0, 1),
                    (6, 2, 0, 133.33333333333334, 3.0, 0.0, 0), (7, 2, 1, 133.33333333333334, 3.0, 0.0, 0),
                    (8, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (9, 3, 0, 66.66666666666669, 4.0, 0.0, 1),
                    (10, 3, 1, 66.66666666666669, 4.0, 0.0, 1), (11, 3, 2, 66.66666666666669, 4.0, 0.0, 1),
                    (12, 4, 0, 53.33333333333335, 5.0, 0.0, 0), (13, 4, 1, 53.33333333333335, 5.0, 0.0, 0),
                    (14, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1),
                    (2, 2, 0, 133.33333333333334, 3.0, 0.0, 0), (3, 3, 0, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 4, 0, 53.33333333333335, 5.0, 0.0, 0), (5, 0, 1, 100.0, 1.0, 0.0, 0),
                    (6, 1, 1, 200.0, 2.0, 0.0, 1), (7, 2, 1, 133.33333333333334, 3.0, 0.0, 0),
                    (8, 3, 1, 66.66666666666669, 4.0, 0.0, 1), (9, 4, 1, 53.33333333333335, 5.0, 0.0, 0),
                    (10, 0, 2, 100.0, 1.0, 0.0, 0), (11, 1, 2, 200.0, 2.0, 0.0, 1),
                    (12, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (13, 3, 2, 66.66666666666669, 4.0, 0.0, 1),
                    (14, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
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
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 1, 100.0, 1.0, 0.0, 0),
                    (2, 0, 2, 100.0, 1.0, 0.0, 0), (3, 1, 0, 200.0, 2.0, 0.0, 1),
                    (4, 1, 1, 200.0, 2.0, 0.0, 1), (5, 1, 2, 200.0, 2.0, 0.0, 1),
                    (6, 2, 0, 133.33333333333334, 3.0, 0.0, 0), (7, 2, 1, 133.33333333333334, 3.0, 0.0, 0),
                    (8, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (9, 3, 0, 66.66666666666669, 4.0, 0.0, 1),
                    (10, 3, 1, 66.66666666666669, 4.0, 0.0, 1), (11, 3, 2, 66.66666666666669, 4.0, 0.0, 1),
                    (12, 4, 0, 53.33333333333335, 5.0, 0.0, 0), (13, 4, 1, 53.33333333333335, 5.0, 0.0, 0),
                    (14, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 1, 100.0, 1.0, 0.0, 0),
                    (2, 1, 0, 200.0, 2.0, 0.0, 1), (3, 1, 1, 200.0, 2.0, 0.0, 1),
                    (4, 2, 0, 133.33333333333334, 3.0, 0.0, 0), (5, 2, 1, 133.33333333333334, 3.0, 0.0, 0),
                    (6, 3, 0, 66.66666666666669, 4.0, 0.0, 1), (7, 3, 1, 66.66666666666669, 4.0, 0.0, 1),
                    (8, 4, 0, 53.33333333333335, 5.0, 0.0, 0), (9, 4, 1, 53.33333333333335, 5.0, 0.0, 0),
                    (10, 0, 2, 100.0, 1.0, 0.0, 0), (11, 1, 2, 200.0, 2.0, 0.0, 1),
                    (12, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (13, 3, 2, 66.66666666666669, 4.0, 0.0, 1),
                    (14, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
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
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 2, 100.0, 1.0, 0.0, 0),
                    (2, 1, 0, 200.0, 2.0, 0.0, 1), (3, 1, 1, 200.0, 2.0, 0.0, 1),
                    (4, 1, 2, 200.0, 2.0, 0.0, 1), (5, 2, 0, 266.6666666666667, 3.0, 0.0, 0),
                    (6, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (7, 3, 0, 333.33333333333337, 4.0, 0.0, 1),
                    (8, 3, 2, 66.66666666666669, 4.0, 0.0, 1), (9, 4, 0, 266.6666666666667, 5.0, 0.0, 0),
                    (10, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1),
                    (2, 1, 1, 200.0, 2.0, 0.0, 1), (3, 2, 0, 266.6666666666667, 3.0, 0.0, 0),
                    (4, 3, 0, 333.33333333333337, 4.0, 0.0, 1), (5, 4, 0, 266.6666666666667, 5.0, 0.0, 0),
                    (6, 0, 2, 100.0, 1.0, 0.0, 0), (7, 1, 2, 200.0, 2.0, 0.0, 1),
                    (8, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (9, 3, 2, 66.66666666666669, 4.0, 0.0, 1),
                    (10, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
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
            cash_sharing=True, row_wise=test_row_wise)
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 0, 2, 100.0, 1.0, 0.0, 0),
                    (2, 1, 0, 200.0, 2.0, 0.0, 1), (3, 1, 1, 200.0, 2.0, 0.0, 1),
                    (4, 1, 2, 200.0, 2.0, 0.0, 1), (5, 2, 0, 266.6666666666667, 3.0, 0.0, 0),
                    (6, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (7, 3, 0, 333.33333333333337, 4.0, 0.0, 1),
                    (8, 3, 2, 66.66666666666669, 4.0, 0.0, 1), (9, 4, 0, 266.6666666666667, 5.0, 0.0, 0),
                    (10, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 100.0, 1.0, 0.0, 0), (1, 1, 0, 200.0, 2.0, 0.0, 1),
                    (2, 1, 1, 200.0, 2.0, 0.0, 1), (3, 2, 0, 266.6666666666667, 3.0, 0.0, 0),
                    (4, 3, 0, 333.33333333333337, 4.0, 0.0, 1), (5, 4, 0, 266.6666666666667, 5.0, 0.0, 0),
                    (6, 0, 2, 100.0, 1.0, 0.0, 0), (7, 1, 2, 200.0, 2.0, 0.0, 1),
                    (8, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (9, 3, 2, 66.66666666666669, 4.0, 0.0, 1),
                    (10, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
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
            cash_sharing=True, call_seq='reversed', row_wise=test_row_wise)
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 0, 2, 100.0, 1.0, 0.0, 0),
                    (2, 1, 1, 200.0, 2.0, 0.0, 1), (3, 1, 0, 200.0, 2.0, 0.0, 1),
                    (4, 1, 2, 200.0, 2.0, 0.0, 1), (5, 2, 1, 266.6666666666667, 3.0, 0.0, 0),
                    (6, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (7, 3, 1, 333.33333333333337, 4.0, 0.0, 1),
                    (8, 3, 2, 66.66666666666669, 4.0, 0.0, 1), (9, 4, 1, 266.6666666666667, 5.0, 0.0, 0),
                    (10, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 1, 0, 200.0, 2.0, 0.0, 1), (3, 2, 1, 266.6666666666667, 3.0, 0.0, 0),
                    (4, 3, 1, 333.33333333333337, 4.0, 0.0, 1), (5, 4, 1, 266.6666666666667, 5.0, 0.0, 0),
                    (6, 0, 2, 100.0, 1.0, 0.0, 0), (7, 1, 2, 200.0, 2.0, 0.0, 1),
                    (8, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (9, 3, 2, 66.66666666666669, 4.0, 0.0, 1),
                    (10, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
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
            cash_sharing=True, call_seq='random', seed=seed, row_wise=test_row_wise)
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 0, 2, 100.0, 1.0, 0.0, 0),
                    (2, 1, 1, 200.0, 2.0, 0.0, 1), (3, 1, 2, 200.0, 2.0, 0.0, 1),
                    (4, 2, 1, 133.33333333333334, 3.0, 0.0, 0), (5, 2, 2, 133.33333333333334, 3.0, 0.0, 0),
                    (6, 3, 1, 66.66666666666669, 4.0, 0.0, 1), (7, 3, 0, 66.66666666666669, 4.0, 0.0, 1),
                    (8, 3, 2, 66.66666666666669, 4.0, 0.0, 1), (9, 4, 1, 106.6666666666667, 5.0, 0.0, 0),
                    (10, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 1, 100.0, 1.0, 0.0, 0), (1, 1, 1, 200.0, 2.0, 0.0, 1),
                    (2, 2, 1, 133.33333333333334, 3.0, 0.0, 0), (3, 3, 1, 66.66666666666669, 4.0, 0.0, 1),
                    (4, 3, 0, 66.66666666666669, 4.0, 0.0, 1), (5, 4, 1, 106.6666666666667, 5.0, 0.0, 0),
                    (6, 0, 2, 100.0, 1.0, 0.0, 0), (7, 1, 2, 200.0, 2.0, 0.0, 1),
                    (8, 2, 2, 133.33333333333334, 3.0, 0.0, 0), (9, 3, 2, 66.66666666666669, 4.0, 0.0, 1),
                    (10, 4, 2, 53.33333333333335, 5.0, 0.0, 0)
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
                cash_sharing=True, call_seq='auto', row_wise=test_row_wise
            )

        target_hold_value = pd.DataFrame({
            'a': [0., 70., 30., 0., 70.],
            'b': [30., 0., 70., 30., 30.],
            'c': [70., 30., 0., 70., 0.]
        }, index=price.index)

        @njit
        def segment_prep_func_nb(sc, target_hold_value):
            order_size = np.copy(target_hold_value[sc.i, sc.from_col:sc.to_col])
            order_size_type = np.full(sc.group_len, SizeType.TargetValue)
            direction = np.full(sc.group_len, Direction.All)
            temp_float_arr = np.empty(sc.group_len, dtype=np.float_)
            nb.auto_call_seq_ctx_nb(sc, order_size, order_size_type, direction, temp_float_arr)
            sc.last_val_price[sc.from_col:sc.to_col] = sc.close[sc.i, sc.from_col:sc.to_col]
            return order_size, order_size_type, direction

        @njit
        def pct_order_func_nb(oc, order_size, order_size_type, direction):
            col_i = oc.call_seq_now[oc.call_idx]
            return nb.create_order_nb(
                size=order_size[col_i],
                size_type=order_size_type[col_i],
                price=oc.close[oc.i, col_i],
                direction=direction[col_i]
            )

        portfolio = vbt.Portfolio.from_order_func(
            price_wide * 0 + 1, pct_order_func_nb, group_by=np.array([0, 0, 0]),
            cash_sharing=True, segment_prep_func_nb=segment_prep_func_nb,
            segment_prep_args=(target_hold_value.values,), row_wise=test_row_wise)
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
        pd.testing.assert_frame_equal(
            portfolio.holding_value(group_by=False),
            target_hold_value
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
            return nb.create_order_nb(size=50., size_type=SizeType.TargetValue, price=oc.close[oc.i, oc.col])

        portfolio = vbt.Portfolio.from_order_func(
            price.iloc[1:], target_val_order_func_nb, row_wise=test_row_wise)
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 1, 0, 25.0, 3.0, 0.0, 0), (1, 2, 0, 8.333333333333332, 4.0, 0.0, 1),
                    (2, 3, 0, 4.166666666666668, 5.0, 0.0, 1)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 1, 0, 25.0, 3.0, 0.0, 0), (1, 2, 0, 8.333333333333332, 4.0, 0.0, 1),
                    (2, 3, 0, 4.166666666666668, 5.0, 0.0, 1)
                ], dtype=order_dt)
            )
        portfolio = vbt.Portfolio.from_order_func(
            price.iloc[1:], target_val_order_func_nb,
            segment_prep_func_nb=target_val_segment_prep_func_nb,
            segment_prep_args=(price.iloc[:-1].values,), row_wise=test_row_wise)
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 50.0, 2.0, 0.0, 0), (1, 1, 0, 25.0, 3.0, 0.0, 1),
                    (2, 2, 0, 8.333333333333332, 4.0, 0.0, 1), (3, 3, 0, 4.166666666666668, 5.0, 0.0, 1)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 50.0, 2.0, 0.0, 0), (1, 1, 0, 25.0, 3.0, 0.0, 1),
                    (2, 2, 0, 8.333333333333332, 4.0, 0.0, 1), (3, 3, 0, 4.166666666666668, 5.0, 0.0, 1)
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
            return nb.create_order_nb(size=0.5, size_type=SizeType.TargetPercent, price=oc.close[oc.i, oc.col])

        portfolio = vbt.Portfolio.from_order_func(
            price.iloc[1:], target_pct_order_func_nb, row_wise=test_row_wise)
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 1, 0, 25.0, 3.0, 0.0, 0), (1, 2, 0, 8.333333333333332, 4.0, 0.0, 1),
                    (2, 3, 0, 1.0416666666666679, 5.0, 0.0, 1)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 1, 0, 25.0, 3.0, 0.0, 0), (1, 2, 0, 8.333333333333332, 4.0, 0.0, 1),
                    (2, 3, 0, 1.0416666666666679, 5.0, 0.0, 1)
                ], dtype=order_dt)
            )
        portfolio = vbt.Portfolio.from_order_func(
            price.iloc[1:], target_pct_order_func_nb,
            segment_prep_func_nb=target_pct_segment_prep_func_nb,
            segment_prep_args=(price.iloc[:-1].values,), row_wise=test_row_wise)
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 50.0, 2.0, 0.0, 0), (1, 1, 0, 25.0, 3.0, 0.0, 1),
                    (2, 3, 0, 3.125, 5.0, 0.0, 1)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 50.0, 2.0, 0.0, 0), (1, 1, 0, 25.0, 3.0, 0.0, 1),
                    (2, 3, 0, 3.125, 5.0, 0.0, 1)
                ], dtype=order_dt)
            )

    @pytest.mark.parametrize(
        "test_row_wise",
        [False, True],
    )
    def test_init_cash(self, test_row_wise):
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, 10., row_wise=test_row_wise, init_cash=[1., 10., np.inf])
        if test_row_wise:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 0, 1, 10.0, 1.0, 0.0, 0),
                    (2, 0, 2, 10.0, 1.0, 0.0, 0), (3, 1, 0, 10.0, 2.0, 0.0, 1),
                    (4, 1, 1, 10.0, 2.0, 0.0, 1), (5, 1, 2, 10.0, 2.0, 0.0, 1),
                    (6, 2, 0, 6.666666666666667, 3.0, 0.0, 0), (7, 2, 1, 6.666666666666667, 3.0, 0.0, 0),
                    (8, 2, 2, 10.0, 3.0, 0.0, 0), (9, 3, 0, 10.0, 4.0, 0.0, 1),
                    (10, 3, 1, 10.0, 4.0, 0.0, 1), (11, 3, 2, 10.0, 4.0, 0.0, 1),
                    (12, 4, 0, 8.0, 5.0, 0.0, 0), (13, 4, 1, 8.0, 5.0, 0.0, 0),
                    (14, 4, 2, 10.0, 5.0, 0.0, 0)
                ], dtype=order_dt)
            )
        else:
            record_arrays_close(
                portfolio.order_records,
                np.array([
                    (0, 0, 0, 1.0, 1.0, 0.0, 0), (1, 1, 0, 10.0, 2.0, 0.0, 1),
                    (2, 2, 0, 6.666666666666667, 3.0, 0.0, 0), (3, 3, 0, 10.0, 4.0, 0.0, 1),
                    (4, 4, 0, 8.0, 5.0, 0.0, 0), (5, 0, 1, 10.0, 1.0, 0.0, 0),
                    (6, 1, 1, 10.0, 2.0, 0.0, 1), (7, 2, 1, 6.666666666666667, 3.0, 0.0, 0),
                    (8, 3, 1, 10.0, 4.0, 0.0, 1), (9, 4, 1, 8.0, 5.0, 0.0, 0),
                    (10, 0, 2, 10.0, 1.0, 0.0, 0), (11, 1, 2, 10.0, 2.0, 0.0, 1),
                    (12, 2, 2, 10.0, 3.0, 0.0, 0), (13, 3, 2, 10.0, 4.0, 0.0, 1),
                    (14, 4, 2, 10.0, 5.0, 0.0, 0)
                ], dtype=order_dt)
            )
        assert type(portfolio._init_cash) == np.ndarray
        base_portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, 10., row_wise=test_row_wise, init_cash=np.inf)
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, 10., row_wise=test_row_wise, init_cash=InitCashMode.Auto)
        record_arrays_close(
            portfolio.order_records,
            base_portfolio.orders().values
        )
        assert portfolio._init_cash == InitCashMode.Auto
        portfolio = vbt.Portfolio.from_order_func(
            price_wide, order_func_nb, 10., row_wise=test_row_wise, init_cash=InitCashMode.AutoAlign)
        record_arrays_close(
            portfolio.order_records,
            base_portfolio.orders().values
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
directions = ['longonly', 'shortonly', 'all']
group_by = pd.Index(['first', 'first', 'second'], name='group')

portfolio = vbt.Portfolio.from_orders(
    price_na, order_size_new, size_type='shares', direction=directions,
    fees=0.01, fixed_fees=0.1, slippage=0.01, log=True,
    call_seq='reversed', group_by=None,
    init_cash=[100., 100., 100.], freq='1D'
)  # independent

portfolio_grouped = vbt.Portfolio.from_orders(
    price_na, order_size_new, size_type='shares', direction=directions,
    fees=0.01, fixed_fees=0.1, slippage=0.01, log=True,
    call_seq='reversed', group_by=group_by, cash_sharing=False,
    init_cash=[100., 100., 100.], freq='1D'
)  # grouped

portfolio_shared = vbt.Portfolio.from_orders(
    price_na, order_size_new, size_type='shares', direction=directions,
    fees=0.01, fixed_fees=0.1, slippage=0.01, log=True,
    call_seq='reversed', group_by=group_by, cash_sharing=True,
    init_cash=[200., 100.], freq='1D'
)  # shared


class TestPortfolio:
    def test_wrapper(self):
        pd.testing.assert_index_equal(
            portfolio.wrapper.index,
            price_na.index
        )
        pd.testing.assert_index_equal(
            portfolio.wrapper.columns,
            price_na.columns
        )
        assert portfolio.wrapper.ndim == 2
        assert portfolio.wrapper.grouper.group_by is None
        assert portfolio.wrapper.grouper.allow_enable
        assert portfolio.wrapper.grouper.allow_disable
        assert portfolio.wrapper.grouper.allow_modify
        pd.testing.assert_index_equal(
            portfolio_grouped.wrapper.index,
            price_na.index
        )
        pd.testing.assert_index_equal(
            portfolio_grouped.wrapper.columns,
            price_na.columns
        )
        assert portfolio_grouped.wrapper.ndim == 2
        pd.testing.assert_index_equal(
            portfolio_grouped.wrapper.grouper.group_by,
            group_by
        )
        assert portfolio_grouped.wrapper.grouper.allow_enable
        assert portfolio_grouped.wrapper.grouper.allow_disable
        assert portfolio_grouped.wrapper.grouper.allow_modify
        pd.testing.assert_index_equal(
            portfolio_shared.wrapper.index,
            price_na.index
        )
        pd.testing.assert_index_equal(
            portfolio_shared.wrapper.columns,
            price_na.columns
        )
        assert portfolio_shared.wrapper.ndim == 2
        pd.testing.assert_index_equal(
            portfolio_shared.wrapper.grouper.group_by,
            group_by
        )
        assert not portfolio_shared.wrapper.grouper.allow_enable
        assert portfolio_shared.wrapper.grouper.allow_disable
        assert not portfolio_shared.wrapper.grouper.allow_modify

    def test_indexing(self):
        assert portfolio['a'].wrapper == portfolio.wrapper['a']
        assert portfolio['a'].orders() == portfolio.orders()['a']
        assert portfolio['a'].logs() == portfolio.logs()['a']
        assert portfolio['a'].init_cash() == portfolio.init_cash()['a']
        pd.testing.assert_series_equal(portfolio['a'].call_seq, portfolio.call_seq['a'])

        assert portfolio['c'].wrapper == portfolio.wrapper['c']
        assert portfolio['c'].orders() == portfolio.orders()['c']
        assert portfolio['c'].logs() == portfolio.logs()['c']
        assert portfolio['c'].init_cash() == portfolio.init_cash()['c']
        pd.testing.assert_series_equal(portfolio['c'].call_seq, portfolio.call_seq['c'])

        assert portfolio[['c']].wrapper == portfolio.wrapper[['c']]
        assert portfolio[['c']].orders() == portfolio.orders()[['c']]
        assert portfolio[['c']].logs() == portfolio.logs()[['c']]
        pd.testing.assert_series_equal(portfolio[['c']].init_cash(), portfolio.init_cash()[['c']])
        pd.testing.assert_frame_equal(portfolio[['c']].call_seq, portfolio.call_seq[['c']])

        assert portfolio_grouped['first'].wrapper == portfolio_grouped.wrapper['first']
        assert portfolio_grouped['first'].orders() == portfolio_grouped.orders()['first']
        assert portfolio_grouped['first'].logs() == portfolio_grouped.logs()['first']
        assert portfolio_grouped['first'].init_cash() == portfolio_grouped.init_cash()['first']
        pd.testing.assert_frame_equal(portfolio_grouped['first'].call_seq, portfolio_grouped.call_seq[['a', 'b']])

        assert portfolio_grouped[['first']].wrapper == portfolio_grouped.wrapper[['first']]
        assert portfolio_grouped[['first']].orders() == portfolio_grouped.orders()[['first']]
        assert portfolio_grouped[['first']].logs() == portfolio_grouped.logs()[['first']]
        pd.testing.assert_series_equal(
            portfolio_grouped[['first']].init_cash(),
            portfolio_grouped.init_cash()[['first']])
        pd.testing.assert_frame_equal(portfolio_grouped[['first']].call_seq, portfolio_grouped.call_seq[['a', 'b']])

        assert portfolio_grouped['second'].wrapper == portfolio_grouped.wrapper['second']
        assert portfolio_grouped['second'].orders() == portfolio_grouped.orders()['second']
        assert portfolio_grouped['second'].logs() == portfolio_grouped.logs()['second']
        assert portfolio_grouped['second'].init_cash() == portfolio_grouped.init_cash()['second']
        pd.testing.assert_series_equal(portfolio_grouped['second'].call_seq, portfolio_grouped.call_seq['c'])

        assert portfolio_grouped[['second']].orders() == portfolio_grouped.orders()[['second']]
        assert portfolio_grouped[['second']].wrapper == portfolio_grouped.wrapper[['second']]
        assert portfolio_grouped[['second']].orders() == portfolio_grouped.orders()[['second']]
        assert portfolio_grouped[['second']].logs() == portfolio_grouped.logs()[['second']]
        pd.testing.assert_series_equal(
            portfolio_grouped[['second']].init_cash(),
            portfolio_grouped.init_cash()[['second']])
        pd.testing.assert_frame_equal(portfolio_grouped[['second']].call_seq, portfolio_grouped.call_seq[['c']])

        assert portfolio_shared['first'].wrapper == portfolio_shared.wrapper['first']
        assert portfolio_shared['first'].orders() == portfolio_shared.orders()['first']
        assert portfolio_shared['first'].logs() == portfolio_shared.logs()['first']
        assert portfolio_shared['first'].init_cash() == portfolio_shared.init_cash()['first']
        pd.testing.assert_frame_equal(portfolio_shared['first'].call_seq, portfolio_shared.call_seq[['a', 'b']])

        assert portfolio_shared[['first']].orders() == portfolio_shared.orders()[['first']]
        assert portfolio_shared[['first']].wrapper == portfolio_shared.wrapper[['first']]
        assert portfolio_shared[['first']].orders() == portfolio_shared.orders()[['first']]
        assert portfolio_shared[['first']].logs() == portfolio_shared.logs()[['first']]
        pd.testing.assert_series_equal(
            portfolio_shared[['first']].init_cash(),
            portfolio_shared.init_cash()[['first']])
        pd.testing.assert_frame_equal(portfolio_shared[['first']].call_seq, portfolio_shared.call_seq[['a', 'b']])

        assert portfolio_shared['second'].wrapper == portfolio_shared.wrapper['second']
        assert portfolio_shared['second'].orders() == portfolio_shared.orders()['second']
        assert portfolio_shared['second'].logs() == portfolio_shared.logs()['second']
        assert portfolio_shared['second'].init_cash() == portfolio_shared.init_cash()['second']
        pd.testing.assert_series_equal(portfolio_shared['second'].call_seq, portfolio_shared.call_seq['c'])

        assert portfolio_shared[['second']].wrapper == portfolio_shared.wrapper[['second']]
        assert portfolio_shared[['second']].orders() == portfolio_shared.orders()[['second']]
        assert portfolio_shared[['second']].logs() == portfolio_shared.logs()[['second']]
        pd.testing.assert_series_equal(
            portfolio_shared[['second']].init_cash(),
            portfolio_shared.init_cash()[['second']])
        pd.testing.assert_frame_equal(portfolio_shared[['second']].call_seq, portfolio_shared.call_seq[['c']])

    def test_regroup(self):
        assert portfolio.regroup(None) == portfolio
        assert portfolio.regroup(False) == portfolio
        assert portfolio.regroup(group_by) != portfolio
        pd.testing.assert_index_equal(portfolio.regroup(group_by).wrapper.grouper.group_by, group_by)
        assert portfolio_grouped.regroup(None) == portfolio_grouped
        assert portfolio_grouped.regroup(False) != portfolio_grouped
        assert portfolio_grouped.regroup(False).wrapper.grouper.group_by is None
        assert portfolio_grouped.regroup(group_by) == portfolio_grouped
        assert portfolio_shared.regroup(None) == portfolio_shared
        with pytest.raises(Exception) as e_info:
            _ = portfolio_shared.regroup(False)
        assert portfolio_shared.regroup(group_by) == portfolio_shared

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

    def test_orders(self):
        record_arrays_close(
            portfolio.orders().values,
            np.array([
                (0, 1, 0, 0.1, 2.02, 0.10202, 0), (1, 2, 0, 0.1, 2.9699999999999998, 0.10297, 1),
                (2, 4, 0, 1.0, 5.05, 0.1505, 0), (3, 0, 1, 1.0, 0.99, 0.10990000000000001, 1),
                (4, 1, 1, 0.1, 1.98, 0.10198, 1), (5, 3, 1, 0.1, 4.04, 0.10404000000000001, 0),
                (6, 4, 1, 1.0, 4.95, 0.14950000000000002, 1), (7, 0, 2, 1.0, 1.01, 0.1101, 0),
                (8, 1, 2, 0.1, 2.02, 0.10202, 0), (9, 2, 2, 1.0, 2.9699999999999998, 0.1297, 1),
                (10, 3, 2, 0.1, 3.96, 0.10396000000000001, 1)
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
            index=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_logs(self):
        record_arrays_close(
            portfolio.logs().values,
            np.array([
                (0, 0, 0, 0, 100.0, 0.0, np.nan, 100.0, 1.0, 0, 0, np.nan, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 100.0, 0.0, np.nan, np.nan, np.nan, -1, 1, 1, -1),
                (1, 1, 0, 0, 100.0, 0.0, 2.0, 100.0, 0.1, 0, 0, 2.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 99.69598, 0.1, 0.1, 2.02, 0.10202, 0, 0, -1, 0),
                (2, 2, 0, 0, 99.69598, 0.1, 3.0, 99.99598, -1.0, 0, 0, 3.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 99.89001, 0.0, 0.1, 2.9699999999999998, 0.10297, 1, 0, -1, 1),
                (3, 3, 0, 0, 99.89001, 0.0, 4.0, 99.89001, -0.1, 0, 0, 4.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 99.89001, 0.0, np.nan, np.nan, np.nan, -1, 2, 8, -1),
                (4, 4, 0, 0, 99.89001, 0.0, 5.0, 99.89001, 1.0, 0, 0, 5.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 94.68951, 1.0, 1.0, 5.05, 0.1505, 0, 0, -1, 2),
                (5, 0, 1, 1, 100.0, 0.0, 1.0, 100.0, 1.0, 0, 1, 1.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 100.8801, -1.0, 1.0, 0.99, 0.10990000000000001, 1, 0, -1, 3),
                (6, 1, 1, 1, 100.8801, -1.0, 2.0, 98.8801, 0.1, 0, 1, 2.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 100.97612, -1.1, 0.1, 1.98, 0.10198, 1, 0, -1, 4),
                (7, 2, 1, 1, 100.97612, -1.1, np.nan, np.nan, -1.0, 0, 1, np.nan, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 100.97612, -1.1, np.nan, np.nan, np.nan, -1, 1, 1, -1),
                (8, 3, 1, 1, 100.97612, -1.1, 4.0, 96.57611999999999, -0.1, 0, 1, 4.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 100.46808, -1.0, 0.1, 4.04, 0.10404000000000001, 0, 0, -1, 5),
                (9, 4, 1, 1, 100.46808, -1.0, 5.0, 95.46808, 1.0, 0, 1, 5.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 105.26858, -2.0, 1.0, 4.95, 0.14950000000000002, 1, 0, -1, 6),
                (10, 0, 2, 2, 100.0, 0.0, 1.0, 100.0, 1.0, 0, 2, 1.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 98.8799, 1.0, 1.0, 1.01, 0.1101, 0, 0, -1, 7),
                (11, 1, 2, 2, 98.8799, 1.0, 2.0, 100.8799, 0.1, 0, 2, 2.0, 0.01, 0.1, 0.01, 1e-08, np.inf,
                 0.0, False, True, False, True, 98.57588000000001, 1.1, 0.1, 2.02, 0.10202, 0, 0, -1, 8),
                (12, 2, 2, 2, 98.57588000000001, 1.1, 3.0, 101.87588000000001, -1.0, 0, 2, 3.0,
                 0.01, 0.1, 0.01, 1e-08, np.inf, 0.0, False, True, False, True, 101.41618000000001,
                 0.10000000000000009, 1.0, 2.9699999999999998, 0.1297, 1, 0, -1, 9),
                (13, 3, 2, 2, 101.41618000000001, 0.10000000000000009, 4.0, 101.81618000000002,
                 -0.1, 0, 2, 4.0, 0.01, 0.1, 0.01, 1e-08, np.inf, 0.0, False, True, False, True,
                 101.70822000000001, 0.0, 0.1, 3.96, 0.10396000000000001, 1, 0, -1, 10),
                (14, 4, 2, 2, 101.70822000000001, 0.0, np.nan, 101.70822000000001, 1.0, 0, 2, np.nan, 0.01, 0.1, 0.01,
                 1e-08, np.inf, 0.0, False, True, False, True, 101.70822000000001, 0.0, np.nan, np.nan, np.nan, -1, 1, 1, -1)
            ], dtype=log_dt)
        )
        result = pd.Series(
            np.array([5, 5, 5]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.logs().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.logs(group_by=False).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.logs(group_by=False).count(),
            result
        )
        result = pd.Series(
            np.array([10, 5]),
            index=pd.Index(['first', 'second'], dtype='object', name='group')
        )
        pd.testing.assert_series_equal(
            portfolio.logs(group_by=group_by).count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.logs().count(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.logs().count(),
            result
        )

    def test_trades(self):
        record_arrays_close(
            portfolio.trades().values,
            np.array([
                (0, 0, 0.1, 1, 2.02, 0.10202, 2, 2.9699999999999998, 0.10297,
                 -0.10999000000000003, -0.5445049504950497, 0, 1, 0),
                (1, 0, 1.0, 4, 5.05, 0.1505, 4, 5.0, 0.0,
                 -0.20049999999999982, -0.03970297029702967, 0, 0, 1),
                (2, 1, 0.1, 0, 1.0799999999999998, 0.019261818181818182,
                 3, 4.04, 0.10404000000000001, -0.4193018181818182, -3.882424242424243, 1, 1, 2),
                (3, 1, 2.0, 0, 3.015, 0.3421181818181819, 4, 5.0, 0.0,
                 -4.312118181818182, -0.7151108095884214, 1, 0, 2),
                (4, 2, 1.0, 0, 1.1018181818181818, 0.19283636363636364, 2,
                 2.9699999999999998, 0.1297, 1.5456454545454543, 1.4028135313531351, 0, 1, 3),
                (5, 2, 0.10000000000000009, 0, 1.1018181818181818, 0.019283636363636378,
                 3, 3.96, 0.10396000000000001, 0.1625745454545457, 1.4755115511551162, 0, 1, 3)
            ], dtype=trade_dt)
        )
        result = pd.Series(
            np.array([2, 2, 2]),
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
            np.array([4, 2]),
            index=pd.Index(['first', 'second'], dtype='object', name='group')
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
            portfolio.positions().values,
            np.array([
                (0, 0, 0.1, 1, 2.02, 0.10202, 2, 2.9699999999999998,
                 0.10297, -0.10999000000000003, -0.5445049504950497, 0, 1),
                (1, 0, 1.0, 4, 5.05, 0.1505, 4, 5.0, 0.0,
                 -0.20049999999999982, -0.03970297029702967, 0, 0),
                (2, 1, 2.1, 0, 2.9228571428571426, 0.36138000000000003, 4, 4.954285714285714,
                 0.10404000000000001, -4.731420000000001, -0.7708406647116326, 1, 0),
                (3, 2, 1.1, 0, 1.1018181818181818, 0.21212000000000003, 3,
                 3.06, 0.23366000000000003, 1.7082200000000003, 1.4094224422442245, 0, 1)
            ], dtype=position_dt)
        )
        result = pd.Series(
            np.array([2, 1, 1]),
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
            np.array([3, 1]),
            index=pd.Index(['first', 'second'], dtype='object', name='group')
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
        record_arrays_close(
            portfolio.drawdowns().values,
            np.array([
                (0, 0, 0, 4, 4, 0), (1, 1, 0, 4, 4, 0), (2, 2, 2, 3, 4, 0)
            ], dtype=drawdown_dt)
        )
        result = pd.Series(
            np.array([1, 1, 1]),
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
            np.array([1, 1]),
            index=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_share_flow(self):
        pd.testing.assert_frame_equal(
            portfolio.share_flow(direction='longonly'),
            pd.DataFrame(
                np.array([
                    [0., 0., 1.],
                    [0.1, 0., 0.1],
                    [-0.1, 0., -1.],
                    [0., 0., -0.1],
                    [1., 0., 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio.share_flow(direction='shortonly'),
            pd.DataFrame(
                np.array([
                    [0., 1., 0.],
                    [0., 0.1, 0.],
                    [0., 0., 0.],
                    [0., -0.1, 0.],
                    [0., 1., 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [0., -1., 1.],
                [0.1, -0.1, 0.1],
                [-0.1, 0., -1.],
                [0., 0.1, -0.1],
                [1., -1., 0.]
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
        pd.testing.assert_frame_equal(
            portfolio.shares(direction='longonly'),
            pd.DataFrame(
                np.array([
                    [0., 0., 1.],
                    [0.1, 0., 1.1],
                    [0., 0., 0.1],
                    [0., 0., 0.],
                    [1., 0., 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio.shares(direction='shortonly'),
            pd.DataFrame(
                np.array([
                    [0., 1., 0.],
                    [0., 1.1, 0.],
                    [0., 1.1, 0.],
                    [0., 1., 0.],
                    [0., 2., 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [0., -1., 1.],
                [0.1, -1.1, 1.1],
                [0., -1.1, 0.1],
                [0., -1., 0.],
                [1., -2., 0.]
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

    def test_pos_mask(self):
        pd.testing.assert_frame_equal(
            portfolio.pos_mask(direction='longonly'),
            pd.DataFrame(
                np.array([
                    [False, False, True],
                    [True, False, True],
                    [False, False, True],
                    [False, False, False],
                    [True, False, False]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio.pos_mask(direction='shortonly'),
            pd.DataFrame(
                np.array([
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False],
                    [False, True, False]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [False, True, True],
                [True, True, True],
                [False, True, True],
                [False, True, False],
                [True, True, False]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.pos_mask(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.pos_mask(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.pos_mask(group_by=False),
            result
        )
        result = pd.DataFrame(
            np.array([
                [True, True],
                [True, True],
                [True, True],
                [True, False],
                [True, False]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
        )
        pd.testing.assert_frame_equal(
            portfolio.pos_mask(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.pos_mask(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.pos_mask(),
            result
        )

    def test_pos_coverage(self):
        pd.testing.assert_series_equal(
            portfolio.pos_coverage(direction='longonly'),
            pd.Series(np.array([0.4, 0., 0.6]), index=price_na.columns)
        )
        pd.testing.assert_series_equal(
            portfolio.pos_coverage(direction='shortonly'),
            pd.Series(np.array([0., 1., 0.]), index=price_na.columns)
        )
        result = pd.Series(np.array([0.4, 1., 0.6]), index=price_na.columns)
        pd.testing.assert_series_equal(
            portfolio.pos_coverage(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.pos_coverage(group_by=False),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.pos_coverage(group_by=False),
            result
        )
        result = pd.Series(np.array([0.7, 0.6]), pd.Index(['first', 'second'], dtype='object', name='group'))
        pd.testing.assert_series_equal(
            portfolio.pos_coverage(group_by=group_by),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.pos_coverage(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.pos_coverage(),
            result
        )

    def test_cash_flow(self):
        pd.testing.assert_frame_equal(
            portfolio.cash_flow(short_cash=False),
            pd.DataFrame(
                np.array([
                    [0., -1.0999, -1.1201],
                    [-0.30402, -0.29998, -0.30402],
                    [0.19403, 0., 2.8403],
                    [0., 0.29996, 0.29204],
                    [-5.2005, -5.0995, 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [0., 0.8801, -1.1201],
                [-0.30402, 0.09602, -0.30402],
                [0.19403, 0., 2.8403],
                [0., -0.50804, 0.29204],
                [-5.2005, 4.8005, 0.]
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
                [0.8801, -1.1201],
                [-0.208, -0.30402],
                [0.19403, 2.8403],
                [-0.50804, 0.29204],
                [-0.4, 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_init_cash(self):
        pd.testing.assert_series_equal(
            portfolio.init_cash(),
            pd.Series(np.array([100., 100., 100.]), index=price_na.columns)
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.init_cash(group_by=False),
            pd.Series(np.array([100., 100., 100.]), index=price_na.columns)
        )
        pd.testing.assert_series_equal(
            portfolio_shared.init_cash(group_by=False),
            pd.Series(np.array([200., 200., 100.]), index=price_na.columns)
        )
        result = pd.Series(np.array([200., 100.]), pd.Index(['first', 'second'], dtype='object', name='group'))
        pd.testing.assert_series_equal(
            portfolio.init_cash(group_by=group_by),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.init_cash(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.init_cash(),
            result
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
                index=pd.Index(['first', 'second'], dtype='object', name='group')
            )
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na, 1000., init_cash=InitCashMode.Auto, group_by=group_by, cash_sharing=True).init_cash(),
            pd.Series(
                np.array([26000.0, 10000.0]),
                index=pd.Index(['first', 'second'], dtype='object', name='group')
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
                index=pd.Index(['first', 'second'], dtype='object', name='group')
            )
        )
        pd.testing.assert_series_equal(
            vbt.Portfolio.from_orders(
                price_na, 1000., init_cash=InitCashMode.AutoAlign, group_by=group_by, cash_sharing=True).init_cash(),
            pd.Series(
                np.array([26000.0, 26000.0]),
                index=pd.Index(['first', 'second'], dtype='object', name='group')
            )
        )

    def test_cash(self):
        pd.testing.assert_frame_equal(
            portfolio.cash(short_cash=False),
            pd.DataFrame(
                np.array([
                    [100., 98.9001, 98.8799],
                    [99.69598, 98.60012, 98.57588],
                    [99.89001, 98.60012, 101.41618],
                    [99.89001, 98.90008, 101.70822],
                    [94.68951, 93.80058, 101.70822]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [100., 100.8801, 98.8799],
                [99.69598, 100.97612, 98.57588],
                [99.89001, 100.97612, 101.41618],
                [99.89001, 100.46808, 101.70822],
                [94.68951, 105.26858, 101.70822]
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
                    [200., 200.8801, 98.8799],
                    [199.69598, 200.97612, 98.57588],
                    [199.89001, 200.97612, 101.41618],
                    [199.89001, 200.46808, 101.70822],
                    [194.68951, 205.26858, 101.70822]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.cash(group_by=False, in_sim_order=True),
            pd.DataFrame(
                np.array([
                    [200., 200.8801, 98.8799],
                    [199.69598, 200.09602, 99.69598],
                    [200.19403, 200., 102.8403],
                    [200., 199.49196, 100.29204],
                    [194.7995, 204.8005, 100.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [200.8801, 98.8799],
                [200.6721, 98.57588],
                [200.86613, 101.41618],
                [200.35809, 101.70822],
                [199.95809, 101.70822]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_holding_value(self):
        pd.testing.assert_frame_equal(
            portfolio.holding_value(direction='longonly'),
            pd.DataFrame(
                np.array([
                    [0., 0., 1.],
                    [0.2, 0., 2.2],
                    [0., 0., 0.3],
                    [0., 0., 0.],
                    [5., 0., 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio.holding_value(direction='shortonly'),
            pd.DataFrame(
                np.array([
                    [0., 1., 0.],
                    [0., 2.2, 0.],
                    [0., np.nan, 0.],
                    [0., 4., 0.],
                    [0., 10., 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [0., -1., 1.],
                [0.2, -2.2, 2.2],
                [0., np.nan, 0.3],
                [0., -4., 0.],
                [5., -10., 0.]
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
                [-1., 1.],
                [-2., 2.2],
                [np.nan, 0.3],
                [-4., 0.],
                [-5., 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_gross_exposure(self):
        pd.testing.assert_frame_equal(
            portfolio.gross_exposure(direction='longonly'),
            pd.DataFrame(
                np.array([
                    [0., 0., 0.01001202],
                    [0.00200208, 0., 0.02183062],
                    [0., 0., 0.00294938],
                    [0., 0., 0.],
                    [0.05015573, 0., 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio.gross_exposure(direction='shortonly'),
            pd.DataFrame(
                np.array([
                    [0., 0.01001, 0.],
                    [0., 0.02182537, 0.],
                    [0., np.nan, 0.],
                    [0., 0.03887266, 0.],
                    [0., 0.09633858, 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [0., -0.01021449, 0.01001202],
                [0.00200208, -0.02282155, 0.02183062],
                [0., np.nan, 0.00294938],
                [0., -0.0421496, 0.],
                [0.05015573, -0.11933092, 0.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.gross_exposure(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.gross_exposure(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.gross_exposure(group_by=False),
            pd.DataFrame(
                np.array([
                    [0., -0.00505305, 0.01001202],
                    [0.00100052, -0.01120162, 0.02183062],
                    [0., np.nan, 0.00294938],
                    [0., -0.02052334, 0.],
                    [0.02503887, -0.05440679, 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [-0.005003, 0.01001202],
                [-0.01006684, 0.02183062],
                [np.nan, 0.00294938],
                [-0.02037095, 0.],
                [-0.02564654, 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
        )
        pd.testing.assert_frame_equal(
            portfolio.gross_exposure(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.gross_exposure(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.gross_exposure(),
            result
        )

    def test_net_exposure(self):
        result = pd.DataFrame(
            np.array([
                [0., -0.01001, 0.01001202],
                [0.00200208, -0.02182537, 0.02183062],
                [0., np.nan, 0.00294938],
                [0., -0.03887266, 0.],
                [0.05015573, -0.09633858, 0.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.net_exposure(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.net_exposure(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.net_exposure(group_by=False),
            pd.DataFrame(
                np.array([
                    [0., -0.0050025, 0.01001202],
                    [0.00100052, -0.01095617, 0.02183062],
                    [0., np.nan, 0.00294938],
                    [0., -0.01971414, 0.],
                    [0.02503887, -0.04906757, 0.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [-0.00495344, 0.01001202],
                [-0.00984861, 0.02183062],
                [np.nan, 0.00294938],
                [-0.01957348, 0.],
                [-0.02323332, 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
        )
        pd.testing.assert_frame_equal(
            portfolio.net_exposure(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.net_exposure(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.net_exposure(),
            result
        )

    def test_value(self):
        result = pd.DataFrame(
            np.array([
                [100., 99.8801, 99.8799],
                [99.89598, 98.77612, 100.77588],
                [99.89001, np.nan, 101.71618],
                [99.89001, 96.46808, 101.70822],
                [99.68951, 95.26858, 101.70822]
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
                    [200., 199.8801, 99.8799],
                    [199.89598, 198.77612, 100.77588],
                    [199.89001, np.nan, 101.71618],
                    [199.89001, 196.46808, 101.70822],
                    [199.68951, 195.26858, 101.70822]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.value(group_by=False, in_sim_order=True),
            pd.DataFrame(
                np.array([
                    [199., 199.8801, 99.8799],
                    [197.69598, 197.89602, 101.89598],
                    [np.nan, np.nan, 103.1403],
                    [196., 195.49196, 100.29204],
                    [189.7995, 194.8005, 100.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [199.8801, 99.8799],
                [198.6721, 100.77588],
                [np.nan, 101.71618],
                [196.35809, 101.70822],
                [194.95809, 101.70822]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_total_profit(self):
        result = pd.Series(
            np.array([-0.31049, -4.73142, 1.70822]),
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
            np.array([-5.04191, 1.70822]),
            index=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_final_value(self):
        result = pd.Series(
            np.array([99.68951, 95.26858, 101.70822]),
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
                np.array([199.68951, 195.26858, 101.70822]),
                index=price_na.columns
            )
        )
        result = pd.Series(
            np.array([194.95809, 101.70822]),
            index=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_total_return(self):
        result = pd.Series(
            np.array([-0.0031049, -0.0473142, 0.0170822]),
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
                np.array([-0.00155245, -0.0236571, 0.0170822]),
                index=price_na.columns
            )
        )
        result = pd.Series(
            np.array([-0.02520955, 0.0170822]),
            index=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_returns(self):
        result = pd.DataFrame(
            np.array([
                [0.00000000e+00, -1.19900000e-03, -1.20100000e-03],
                [-1.04020000e-03, -1.10530526e-02, 8.97057366e-03],
                [-5.97621646e-05, np.nan, 9.33060570e-03],
                [0.00000000e+00, np.nan, -7.82569695e-05],
                [-2.00720773e-03, -1.24341648e-02, 0.00000000e+00]
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
                    [0.00000000e+00, -5.99500000e-04, -1.20100000e-03],
                    [-5.20100000e-04, -5.52321117e-03, 8.97057366e-03],
                    [-2.98655331e-05, np.nan, 9.33060570e-03],
                    [0.00000000e+00, np.nan, -7.82569695e-05],
                    [-1.00305163e-03, -6.10531746e-03, 0.00000000e+00]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.returns(group_by=False, in_sim_order=True),
            pd.DataFrame(
                np.array([
                    [-0.00440314, -0.0005995, -0.001201],
                    [-0.00101083, -0.00554764, 0.02018504],
                    [np.nan, np.nan, 0.01221167],
                    [0.00259878, np.nan, -0.02761539],
                    [-0.02567242, -0.0061199, -0.0029119]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [-5.99500000e-04, -1.20100000e-03],
                [-6.04362315e-03, 8.97057366e-03],
                [np.nan, 9.33060570e-03],
                [np.nan, -7.82569695e-05],
                [-7.12983101e-03, 0.00000000e+00]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_active_returns(self):
        result = pd.DataFrame(
            np.array([
                [0., -np.inf, -np.inf],
                [-np.inf, -1.10398, 0.89598],
                [-0.02985, np.nan, 0.42740909],
                [0., np.nan, -0.02653333],
                [-np.inf, -0.299875, 0.]
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
                [-np.inf, -np.inf],
                [-1.208, 0.89598],
                [np.nan, 0.42740909],
                [np.nan, -0.02653333],
                [-0.35, 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
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

    def test_market_value(self):
        result = pd.DataFrame(
            np.array([
                [100., 100., 100.],
                [100., 200., 200.],
                [150., 200., 300.],
                [200., 400., 400.],
                [250., 500., 400.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.market_value(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.market_value(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.market_value(group_by=False),
            pd.DataFrame(
                np.array([
                    [200., 200., 100.],
                    [200., 400., 200.],
                    [300., 400., 300.],
                    [400., 800., 400.],
                    [500., 1000., 400.]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        result = pd.DataFrame(
            np.array([
                [200., 100.],
                [300., 200.],
                [350., 300.],
                [600., 400.],
                [750., 400.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
        )
        pd.testing.assert_frame_equal(
            portfolio.market_value(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.market_value(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.market_value(),
            result
        )

    def test_market_returns(self):
        result = pd.DataFrame(
            np.array([
                [0., 0., 0.],
                [0., 1., 1.],
                [0.5, 0., 0.5],
                [0.33333333, 1., 0.33333333],
                [0.25, 0.25, 0.]
            ]),
            index=price_na.index,
            columns=price_na.columns
        )
        pd.testing.assert_frame_equal(
            portfolio.market_returns(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.market_returns(group_by=False),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.market_returns(group_by=False),
            result
        )
        result = pd.DataFrame(
            np.array([
                [0., 0.],
                [0.5, 1.],
                [0.16666667, 0.5],
                [0.71428571, 0.33333333],
                [0.25, 0.]
            ]),
            index=price_na.index,
            columns=pd.Index(['first', 'second'], dtype='object', name='group')
        )
        pd.testing.assert_frame_equal(
            portfolio.market_returns(group_by=group_by),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_grouped.market_returns(),
            result
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.market_returns(),
            result
        )

    def test_total_market_return(self):
        result = pd.Series(
            np.array([1.5, 4., 3.]),
            index=price_na.columns
        )
        pd.testing.assert_series_equal(
            portfolio.total_market_return(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.total_market_return(group_by=False),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.total_market_return(group_by=False),
            result
        )
        result = pd.Series(
            np.array([2.75, 3.]),
            index=pd.Index(['first', 'second'], dtype='object', name='group')
        )
        pd.testing.assert_series_equal(
            portfolio.total_market_return(group_by=group_by),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_grouped.total_market_return(),
            result
        )
        pd.testing.assert_series_equal(
            portfolio_shared.total_market_return(),
            result
        )

    def test_return_method(self):
        pd.testing.assert_frame_equal(
            portfolio_shared.cumulative_returns(),
            pd.DataFrame(
                np.array([
                    [-0.0005995, -0.001201],
                    [-0.0066395, 0.0077588],
                    [-0.0066395, 0.0171618],
                    [-0.0066395, 0.0170822],
                    [-0.01372199, 0.0170822]
                ]),
                index=price_na.index,
                columns=pd.Index(['first', 'second'], dtype='object', name='group')
            )
        )
        pd.testing.assert_frame_equal(
            portfolio_shared.cumulative_returns(group_by=False),
            pd.DataFrame(
                np.array([
                    [0., -0.0005995, -0.001201],
                    [-0.0005201, -0.0061194, 0.0077588],
                    [-0.00054995, -0.0061194, 0.0171618],
                    [-0.00054995, -0.0061194, 0.0170822],
                    [-0.00155245, -0.01218736, 0.0170822]
                ]),
                index=price_na.index,
                columns=price_na.columns
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(),
            pd.Series(
                np.array([-20.82791491, 10.2576347]),
                index=pd.Index(['first', 'second'], dtype='object', name='group')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(risk_free=0.01),
            pd.Series(
                np.array([-66.19490297745766, -19.873024060759022]),
                index=pd.Index(['first', 'second'], dtype='object', name='group')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(year_freq='365D'),
            pd.Series(
                np.array([-25.06639947, 12.34506527]),
                index=pd.Index(['first', 'second'], dtype='object', name='group')
            )
        )
        pd.testing.assert_series_equal(
            portfolio_shared.sharpe_ratio(group_by=False),
            pd.Series(
                np.array([-11.058998255347488, -21.39151322377427, 10.257634695847853]),
                index=price_na.columns
            )
        )

    def test_stats(self):
        pd.testing.assert_series_equal(
            portfolio.stats(),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 100.0, -1.1112299999999966,
                    -1.1112299999999966, 283.3333333333333, 66.66666666666667,
                    1.6451238489727062, 1.6451238489727062,
                    pd.Timedelta('3 days 08:00:00'), pd.Timedelta('3 days 08:00:00'),
                    1.3333333333333333, 33.333333333333336, -98.38058805880588,
                    -100.8038553855386, -99.59222172217225,
                    pd.Timedelta('2 days 08:00:00'), pd.Timedelta('2 days 04:00:00'),
                    0.10827272727272726, 1.2350921335789007, -0.01041305691622876,
                    -7.373390156195147, 25.695952942372134, 5717.085878360386
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Init. Cash', 'Total Profit',
                    'Total Return [%]', 'Benchmark Return [%]', 'Position Coverage [%]',
                    'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
                    'Avg. Drawdown Duration', 'Num. Trades', 'Win Rate [%]',
                    'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
                    'Max. Trade Duration', 'Avg. Trade Duration', 'Expectancy', 'SQN',
                    'Gross Exposure', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='<lambda>')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].stats(),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 100.0, -0.3104900000000015,
                    -0.3104900000000015, 150.0, 40.0, 0.3104900000000015,
                    0.3104900000000015, pd.Timedelta('4 days 00:00:00'),
                    pd.Timedelta('4 days 00:00:00'), 1, 0.0, -54.450495049504966,
                    -54.450495049504966, -54.450495049504966,
                    pd.Timedelta('1 days 00:00:00'), pd.Timedelta('1 days 00:00:00'),
                    -0.10999000000000003, np.nan, 0.010431562217554364,
                    -11.057783842772304, -9.75393669809172, -46.721467294341814
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Init. Cash', 'Total Profit',
                    'Total Return [%]', 'Benchmark Return [%]', 'Position Coverage [%]',
                    'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
                    'Avg. Drawdown Duration', 'Num. Trades', 'Win Rate [%]',
                    'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
                    'Max. Trade Duration', 'Avg. Trade Duration', 'Expectancy', 'SQN',
                    'Gross Exposure', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].stats(required_return=0.1, risk_free=0.01),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 100.0, -0.3104900000000015,
                    -0.3104900000000015, 150.0, 40.0, 0.3104900000000015,
                    0.3104900000000015, pd.Timedelta('4 days 00:00:00'),
                    pd.Timedelta('4 days 00:00:00'), 1, 0.0, -54.450495049504966,
                    -54.450495049504966, -54.450495049504966,
                    pd.Timedelta('1 days 00:00:00'), pd.Timedelta('1 days 00:00:00'),
                    -0.10999000000000003, np.nan, 0.010431562217554364,
                    -188.9975847831419, -15.874008737030774, -46.721467294341814
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Init. Cash', 'Total Profit',
                    'Total Return [%]', 'Benchmark Return [%]', 'Position Coverage [%]',
                    'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
                    'Avg. Drawdown Duration', 'Num. Trades', 'Win Rate [%]',
                    'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
                    'Max. Trade Duration', 'Avg. Trade Duration', 'Expectancy', 'SQN',
                    'Gross Exposure', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].stats(active_returns=True),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 100.0, -0.3104900000000015,
                    -0.3104900000000015, 150.0, 40.0, 0.3104900000000015,
                    0.3104900000000015, pd.Timedelta('4 days 00:00:00'),
                    pd.Timedelta('4 days 00:00:00'), 1, 0.0, -54.450495049504966,
                    -54.450495049504966, -54.450495049504966,
                    pd.Timedelta('1 days 00:00:00'), pd.Timedelta('1 days 00:00:00'),
                    -0.10999000000000003, np.nan, 0.010431562217554364, np.nan, np.nan, np.nan
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Init. Cash', 'Total Profit',
                    'Total Return [%]', 'Benchmark Return [%]', 'Position Coverage [%]',
                    'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
                    'Avg. Drawdown Duration', 'Num. Trades', 'Win Rate [%]',
                    'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
                    'Max. Trade Duration', 'Avg. Trade Duration', 'Expectancy', 'SQN',
                    'Gross Exposure', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].stats(incl_unrealized=True),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 100.0, -0.3104900000000015,
                    -0.3104900000000015, 150.0, 40.0, 0.3104900000000015,
                    0.3104900000000015, pd.Timedelta('4 days 00:00:00'),
                    pd.Timedelta('4 days 00:00:00'), 2, 0.0, -3.9702970297029667,
                    -54.450495049504966, -29.210396039603964,
                    pd.Timedelta('1 days 00:00:00'), pd.Timedelta('0 days 12:00:00'),
                    -0.1552449999999999, -3.43044967406917, 0.010431562217554364,
                    -11.057783842772304, -9.75393669809172, -46.721467294341814
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Init. Cash', 'Total Profit',
                    'Total Return [%]', 'Benchmark Return [%]', 'Position Coverage [%]',
                    'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
                    'Avg. Drawdown Duration', 'Num. Trades', 'Win Rate [%]',
                    'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
                    'Max. Trade Duration', 'Avg. Trade Duration', 'Expectancy', 'SQN',
                    'Gross Exposure', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(
            portfolio_grouped['first'].stats(),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), 200.0, -5.0419100000000014,
                    -2.5209550000000007, 275.0, 70.0, 2.46248125751388,
                    2.46248125751388, pd.Timedelta('4 days 00:00:00'),
                    pd.Timedelta('4 days 00:00:00'), 2, 0.0, -54.450495049504966,
                    -388.2424242424243, -221.34645964596461,
                    pd.Timedelta('3 days 00:00:00'), pd.Timedelta('2 days 00:00:00'),
                    -0.2646459090909091, -1.711191707103453, -0.015271830375806438,
                    -20.827914910501114, -13.477807138901431, -38.202477209943744
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Init. Cash', 'Total Profit',
                    'Total Return [%]', 'Benchmark Return [%]', 'Position Coverage [%]',
                    'Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
                    'Avg. Drawdown Duration', 'Num. Trades', 'Win Rate [%]',
                    'Best Trade [%]', 'Worst Trade [%]', 'Avg. Trade [%]',
                    'Max. Trade Duration', 'Avg. Trade Duration', 'Expectancy', 'SQN',
                    'Gross Exposure', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='first')
        )
        pd.testing.assert_series_equal(
            portfolio['c'].stats(),
            portfolio.stats(column='c')
        )
        pd.testing.assert_series_equal(
            portfolio['c'].stats(),
            portfolio_grouped.stats(column='c', group_by=False)
        )
        pd.testing.assert_series_equal(
            portfolio_grouped['second'].stats(),
            portfolio_grouped.stats(column='second')
        )

    def test_returns_stats(self):
        pd.testing.assert_series_equal(
            portfolio.returns_stats(),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), -0.3514495165378495,
                    283.3333333333333, 16.311169116434524, 6.502016923413218,
                    -7.373390156195147, 5717.085878360386, -0.8844312951385028,
                    4.768700318817701, 25.695952942372134, 0.3292440354159287,
                    -1.5661463405418332, 3.21984512467388, 7.438903089386976,
                    -0.005028770371222715, -0.3453376142959778, 0.0014301079570843596
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Total Return [%]', 'Benchmark Return [%]',
                    'Annual Return [%]', 'Annual Volatility [%]', 'Sharpe Ratio',
                    'Calmar Ratio', 'Max. Drawdown [%]', 'Omega Ratio', 'Sortino Ratio',
                    'Skew', 'Kurtosis', 'Tail Ratio', 'Common Sense Ratio', 'Value at Risk',
                    'Alpha', 'Beta'
                ], dtype='object'),
                name='<lambda>')
        )
        pd.testing.assert_series_equal(
            portfolio['a'].returns_stats(),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), -0.3104900000000077,
                    150.0, -14.50654838022003, 1.4162092947628355,
                    -11.057783842772304, -46.721467294341814, -0.3104899999999966,
                    0.0, -9.75393669809172, -1.2191070234483876,
                    0.12297560887596681, 0.0, 0.0,
                    -0.0018138061822238526, -0.1792948451549693, 0.0007493142128979539
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Total Return [%]', 'Benchmark Return [%]',
                    'Annual Return [%]', 'Annual Volatility [%]', 'Sharpe Ratio',
                    'Calmar Ratio', 'Max. Drawdown [%]', 'Omega Ratio', 'Sortino Ratio',
                    'Skew', 'Kurtosis', 'Tail Ratio', 'Common Sense Ratio', 'Value at Risk',
                    'Alpha', 'Beta'
                ], dtype='object'),
                name='a')
        )
        pd.testing.assert_series_equal(
            portfolio_grouped['first'].returns_stats(),
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'), pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'), -1.3721992501327551,
                    275.0, -50.161243737582794, 5.554699810092672,
                    -20.827914910501114, -38.202477209943744, -1.3130364154638197,
                    0.0, -13.477807138901431, 1.5461796513775738,
                    np.nan, 0.1629223848940603, 0.08119849030426794,
                    -0.00702121022812057, -0.3758779170662886, -0.010888246304159315
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Total Return [%]', 'Benchmark Return [%]',
                    'Annual Return [%]', 'Annual Volatility [%]', 'Sharpe Ratio',
                    'Calmar Ratio', 'Max. Drawdown [%]', 'Omega Ratio', 'Sortino Ratio',
                    'Skew', 'Kurtosis', 'Tail Ratio', 'Common Sense Ratio', 'Value at Risk',
                    'Alpha', 'Beta'
                ], dtype='object'),
                name='first')
        )
        pd.testing.assert_series_equal(
            portfolio['c'].returns_stats(),
            portfolio.returns_stats(column='c')
        )
        pd.testing.assert_series_equal(
            portfolio['c'].returns_stats(),
            portfolio_grouped.returns_stats(column='c', group_by=False)
        )
        pd.testing.assert_series_equal(
            portfolio_grouped['second'].returns_stats(),
            portfolio_grouped.returns_stats(column='second')
        )

    def test_plot_methods(self):
        _ = portfolio.plot(column='a', subplots='all')
        _ = portfolio_grouped.plot(column='first', subplots='all')
        _ = portfolio_grouped.plot(column='a', subplots='all', group_by=False)
        with pytest.raises(Exception) as e_info:
            _ = portfolio.plot(subplots='all')
        with pytest.raises(Exception) as e_info:
            _ = portfolio_grouped.plot(subplots='all')
        with pytest.raises(Exception) as e_info:
            _ = portfolio_shared.plot(column='a', subplots='all', group_by=False)
