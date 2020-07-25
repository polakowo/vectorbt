import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime

from vectorbt.records import order_dt, trade_dt, position_dt, drawdown_dt
from vectorbt.portfolio.enums import FilledOrder, SizeType

from tests.utils import record_arrays_close

seed = 42

day_dt = np.timedelta64(86400000000000)

price = pd.Series([1, 2, 3, 2, 1], index=pd.Index([
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    datetime(2020, 1, 3),
    datetime(2020, 1, 4),
    datetime(2020, 1, 5)
]), name='Price')

entries = pd.DataFrame({
    'a': [True, True, True, False, False],
    'b': [True, False, True, False, True],
    'c': [False, True, False, True, False]
}, index=price.index)

exits = pd.DataFrame({
    'a': [False, False, True, True, True],
    'b': [False, True, False, True, False],
    'c': [True, False, True, False, True]
}, index=price.index)

order_size = pd.DataFrame({
    'a': [1, 0.1, -1, -0.1, -1],
    'b': [1, 1, 1, 1, -np.inf],
    'c': [np.inf, -np.inf, np.inf, -np.inf, np.inf]
}, index=price.index)


@njit
def order_func_nb(order_context, price, fees, fixed_fees, slippage):
    col = order_context.col
    i = order_context.i
    size = col + 1
    if i % 2 == 1:
        size *= -1
    return vbt.portfolio.nb.Order(
        size, SizeType.Shares, price[i, col], fees[i, col], fixed_fees[i, col], slippage[i, col])


# test_portfolio
init_capital = [100, 200, 300]
levy_alpha = [1., 2., 3.]
risk_free = [0.01, 0.02, 0.03]
required_return = [0.1, 0.2, 0.3]
cutoff = [0.01, 0.02, 0.03]
factor_returns = price.vbt.combine_with_multiple(
    [0.9, 1., 1.1],
    combine_func=np.multiply,
    concat=True,
    keys=entries.columns
)
test_portfolio = vbt.Portfolio.from_signals(
    price, entries, exits,
    fees=0.01,
    init_capital=init_capital,
    freq='1 days',
    year_freq='252 days',
    levy_alpha=levy_alpha,
    risk_free=risk_free,
    required_return=required_return,
    cutoff=cutoff,
    factor_returns=factor_returns
)


# ############# nb.py ############# #

class TestNumba:
    def test_buy_in_cash_nb(self):
        from vectorbt.portfolio.nb import buy_in_cash_nb

        assert buy_in_cash_nb(5, 0, 10, 4, 0, 0, 0) == \
               (1.0, 0.4, FilledOrder(size=0.4, price=10, fees=0, side=0))
        assert buy_in_cash_nb(5, 0, 10, 5, 0, 0, 0) == \
               (0.0, 0.5, FilledOrder(size=0.5, price=10, fees=0, side=0))
        assert buy_in_cash_nb(5, 0, 10, 6, 0, 0, 0) == \
               (0.0, 0.5, FilledOrder(size=0.5, price=10, fees=0, side=0))
        assert buy_in_cash_nb(100, 0, 10, 5, 0, 4, 0) == \
               (95.0, 0.1, FilledOrder(size=0.1, price=10, fees=4, side=0))
        assert buy_in_cash_nb(100, 0, 10, 5, 0, 5, 0) == \
               (100.0, 0.0, None)
        assert buy_in_cash_nb(100, 0, 10, 5, 0, 6, 0) == \
               (100.0, 0.0, None)
        assert buy_in_cash_nb(100, 0, 10, 5, 0, 0, 0.1) == \
               (95.0, 0.45454545454545453, FilledOrder(size=0.45454545454545453, price=11.0, fees=0, side=0))
        assert buy_in_cash_nb(100, 0, 10, 5, 0.1, 0, 0) == \
               (95.0, 0.45, FilledOrder(size=0.45, price=10, fees=0.5, side=0))
        assert buy_in_cash_nb(100, 0, 10, 5, 1, 0, 0) == \
               (95.0, 0.0, FilledOrder(size=0.0, price=10, fees=5, side=0))

    def test_sell_in_cash_nb(self):
        from vectorbt.portfolio.nb import sell_in_cash_nb

        assert sell_in_cash_nb(0, 100, 10, 50, 0, 0, 0) == \
               (50, 95.0, FilledOrder(size=5.0, price=10, fees=0.0, side=1))
        assert sell_in_cash_nb(0, 100, 10, 50, 0, 0, 0.1) == \
               (50.0, 94.44444444444444, FilledOrder(size=5.555555555555555, price=9.0, fees=0.0, side=1))
        assert sell_in_cash_nb(0, 100, 10, 50, 0, 40, 0) == \
               (50, 91.0, FilledOrder(size=9.0, price=10, fees=40.0, side=1))
        assert sell_in_cash_nb(0, 100, 10, 50, 0.1, 0, 0) == \
               (50.0, 94.44444444444444, FilledOrder(size=5.555555555555555, price=10, fees=5.555555555555557, side=1))
        assert sell_in_cash_nb(0, 5, 10, 100, 0, 0, 0) == \
               (50, 0.0, FilledOrder(size=5.0, price=10, fees=0.0, side=1))
        assert sell_in_cash_nb(0, 5, 10, 100, 0, 0, 0.1) == \
               (45.0, 0.0, FilledOrder(size=5.0, price=9.0, fees=0.0, side=1))
        assert sell_in_cash_nb(0, 5, 10, 100, 0, 40, 0) == \
               (10, 0.0, FilledOrder(size=5.0, price=10, fees=40.0, side=1))
        assert sell_in_cash_nb(0, 5, 10, 100, 0.1, 0, 0) == \
               (45.0, 0.0, FilledOrder(size=5.0, price=10, fees=5.0, side=1))
        assert sell_in_cash_nb(100, 5, 10, 100, 0, 100, 0) == \
               (50, 0.0, FilledOrder(size=5.0, price=10, fees=100.0, side=1))
        assert sell_in_cash_nb(0, 5, 10, 100, 0, 100, 0) == \
               (0, 5.0, None)

    def test_buy_in_shares_nb(self):
        from vectorbt.portfolio.nb import buy_in_shares_nb

        assert buy_in_shares_nb(100, 0, 10, 5, 0, 0, 0) == \
               (50.0, 5.0, FilledOrder(size=5.0, price=10, fees=0.0, side=0))
        assert buy_in_shares_nb(100, 0, 10, 5, 0, 0, 0.1) == \
               (45.0, 5.0, FilledOrder(size=5.0, price=11.0, fees=0.0, side=0))
        assert buy_in_shares_nb(100, 0, 10, 5, 0, 40, 0) == \
               (10.0, 5.0, FilledOrder(size=5.0, price=10, fees=40.0, side=0))
        assert buy_in_shares_nb(100, 0, 10, 5, 0.1, 0, 0) == \
               (44.99999999999999, 5.0, FilledOrder(size=5.0, price=10, fees=5.000000000000007, side=0))
        assert buy_in_shares_nb(40, 0, 10, 5, 0, 0, 0) == \
               (0.0, 4.0, FilledOrder(size=4.0, price=10, fees=0.0, side=0))
        assert buy_in_shares_nb(40, 0, 10, 5, 0, 0, 0.1) == \
               (0.0, 3.6363636363636362, FilledOrder(size=3.6363636363636362, price=11.0, fees=0.0, side=0))
        assert buy_in_shares_nb(40, 0, 10, 5, 0, 40, 0) == \
               (40.0, 0.0, None)
        assert buy_in_shares_nb(40, 0, 10, 5, 0.1, 0, 0) == \
               (0.0, 3.636363636363636, FilledOrder(size=3.636363636363636, price=10, fees=3.6363636363636402, side=0))

    def test_sell_in_shares_nb(self):
        from vectorbt.portfolio.nb import sell_in_shares_nb

        assert sell_in_shares_nb(0, 5, 10, 4, 0, 0, 0) == \
               (40, 1.0, FilledOrder(size=4, price=10, fees=0, side=1))
        assert sell_in_shares_nb(0, 5, 10, 5, 0, 0, 0) == \
               (50, 0.0, FilledOrder(size=5, price=10, fees=0, side=1))
        assert sell_in_shares_nb(0, 5, 10, 6, 0, 0, 0) == \
               (50, 0.0, FilledOrder(size=5, price=10, fees=0, side=1))
        assert sell_in_shares_nb(0, 5, 10, 5, 0, 40, 0) == \
               (10, 0.0, FilledOrder(size=5, price=10, fees=40, side=1))
        assert sell_in_shares_nb(0, 5, 10, 5, 0, 50, 0) == \
               (0, 5.0, None)
        assert sell_in_shares_nb(0, 5, 10, 5, 0, 60, 0) == \
               (0, 5.0, None)
        assert sell_in_shares_nb(100, 5, 10, 5, 0, 60, 0) == \
               (90, 0.0, FilledOrder(size=5, price=10, fees=60, side=1))
        assert sell_in_shares_nb(0, 5, 10, 5, 0, 0, 0.1) == \
               (45.0, 0.0, FilledOrder(size=5, price=9.0, fees=0.0, side=1))
        assert sell_in_shares_nb(0, 5, 10, 5, 0.1, 0, 0) == \
               (45.0, 0.0, FilledOrder(size=5, price=10, fees=5.0, side=1))


# ############# base.py ############# #

class TestPortfolio:
    def test_from_signals(self):
        portfolio = vbt.Portfolio.from_signals(price, entries['a'], exits['a'], size=1)
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 2., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio.shares,
            pd.Series(np.array([1., 1., 1., 0., 0.]), index=price.index, name=('a', 'Price'))
        )
        pd.testing.assert_series_equal(
            portfolio.cash,
            pd.Series(np.array([99., 99., 99., 101., 101.]), index=price.index, name=('a', 'Price'))
        )
        portfolio2 = vbt.Portfolio.from_signals(price, entries, exits, size=1)
        record_arrays_close(
            portfolio2.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 2., 0., 1),
                (1, 0, 1., 1., 0., 0), (1, 1, 1., 2., 0., 1),
                (1, 2, 1., 3., 0., 0), (1, 3, 1., 2., 0., 1),
                (1, 4, 1., 1., 0., 0), (2, 1, 1., 2., 0., 0),
                (2, 2, 1., 3., 0., 1), (2, 3, 1., 2., 0., 0),
                (2, 4, 1., 1., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio2.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [1., 0., 1.],
                [1., 1., 0.],
                [0., 0., 1.],
                [0., 1., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio2.cash,
            pd.DataFrame(np.array([
                [99., 99., 100.],
                [99., 101., 98.],
                [99., 98., 101.],
                [101., 100., 99.],
                [101., 99., 100.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_signals_size(self):
        portfolio = vbt.Portfolio.from_signals(price, entries, exits, size=[1, 2, np.inf])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 2., 0., 1),
                (1, 0, 2., 1., 0., 0), (1, 1, 2., 2., 0., 1),
                (1, 2, 2., 3., 0., 0), (1, 3, 2., 2., 0., 1),
                (1, 4, 2., 1., 0., 0), (2, 1, 50., 2., 0., 0),
                (2, 2, 50., 3., 0., 1), (2, 3, 75., 2., 0., 0),
                (2, 4, 75., 1., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 2., 0.],
                [1., 0., 50.],
                [1., 2., 0.],
                [0., 0., 75.],
                [0., 2., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99., 98., 100.],
                [99., 102., 0.],
                [99., 96., 150.],
                [101., 100., 0.],
                [101., 98., 75.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_signals_init_capital(self):
        portfolio = vbt.Portfolio.from_signals(price, entries, exits, init_capital=[1, 10, 100])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 2., 0., 1),
                (1, 0, 10., 1., 0., 0), (1, 1, 10., 2., 0., 1),
                (1, 2, 6.66666667, 3., 0., 0), (1, 3, 6.66666667, 2., 0., 1),
                (1, 4, 13.33333333, 1., 0., 0), (2, 1, 50., 2., 0., 0),
                (2, 2, 50., 3., 0., 1), (2, 3, 75., 2., 0., 0),
                (2, 4, 75., 1., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 10., 0.],
                [1., 0., 50.],
                [1., 6.66666667, 0.],
                [0., 0., 75.],
                [0., 13.33333333, 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [0., 0., 100.],
                [0., 20., 0.],
                [0., 0., 150.],
                [2., 13.33333333, 0.],
                [2., 0., 75.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_signals_fees(self):
        portfolio = vbt.Portfolio.from_signals(price, entries, exits, size=1, fees=[0., 0.01, np.inf])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0.e+00, 0), (0, 3, 1., 2., 0.e+00, 1),
                (1, 0, 1., 1., 1.e-02, 0), (1, 1, 1., 2., 2.e-02, 1),
                (1, 2, 1., 3., 3.e-02, 0), (1, 3, 1., 2., 2.e-02, 1),
                (1, 4, 1., 1., 1.e-02, 0), (2, 1, 0., 2., 1.e+02, 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [1., 0., 0.],
                [1., 1., 0.],
                [0., 0., 0.],
                [0., 1., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99., 98.99, 100.],
                [99., 100.97, 0.],
                [99., 97.94, 0.],
                [101., 99.92, 0.],
                [101., 98.91, 0.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_signals_fixed_fees(self):
        portfolio = vbt.Portfolio.from_signals(price, entries, exits, size=1, fixed_fees=[0., 1., np.inf])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 2., 0., 1),
                (1, 0, 1., 1., 1., 0), (1, 1, 1., 2., 1., 1),
                (1, 2, 1., 3., 1., 0), (1, 3, 1., 2., 1., 1),
                (1, 4, 1., 1., 1., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [1., 0., 0.],
                [1., 1., 0.],
                [0., 0., 0.],
                [0., 1., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99., 98., 100.],
                [99., 99., 100.],
                [99., 95., 100.],
                [101., 96., 100.],
                [101., 94., 100.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_signals_slippage(self):
        portfolio = vbt.Portfolio.from_signals(price, entries, exits, size=1, slippage=[0., 0.01, np.inf])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 2., 0., 1),
                (1, 0, 1., 1.01, 0., 0), (1, 1, 1., 1.98, 0., 1),
                (1, 2, 1., 3.03, 0., 0), (1, 3, 1., 1.98, 0., 1),
                (1, 4, 1., 1.01, 0., 0), (2, 1, 0., np.inf, 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [1., 0., 0.],
                [1., 1., 0.],
                [0., 0., 0.],
                [0., 1., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99., 98.99, 100.],
                [99., 100.97, 0.],
                [99., 97.94, 0.],
                [101., 99.92, 0.],
                [101., 98.91, 0.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_signals_price(self):
        portfolio = vbt.Portfolio.from_signals(
            price, entries, exits, size=1, entry_price=price * 0.9, exit_price=price * 1.1)
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 0.9, 0., 0), (0, 3, 1., 2.2, 0., 1),
                (1, 0, 1., 0.9, 0., 0), (1, 1, 1., 2.2, 0., 1),
                (1, 2, 1., 2.7, 0., 0), (1, 3, 1., 2.2, 0., 1),
                (1, 4, 1., 0.9, 0., 0), (2, 1, 1., 1.8, 0., 0),
                (2, 2, 1., 3.3, 0., 1), (2, 3, 1., 1.8, 0., 0),
                (2, 4, 1., 1.1, 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [1., 0., 1.],
                [1., 1., 0.],
                [0., 0., 1.],
                [0., 1., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99.1, 99.1, 100.],
                [99.1, 101.3, 98.2],
                [99.1, 98.6, 101.5],
                [101.3, 100.8, 99.7],
                [101.3, 99.9, 100.8]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_signals_accumulate(self):
        portfolio = vbt.Portfolio.from_signals(price, entries, exits, size=1, accumulate=True)
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 2, 1., 3., 0., 1), (0, 3, 1., 2., 0., 1),
                (1, 0, 1., 1., 0., 0), (1, 1, 1., 2., 0., 1),
                (1, 2, 1., 3., 0., 0), (1, 3, 1., 2., 0., 1),
                (1, 4, 1., 1., 0., 0), (2, 1, 1., 2., 0., 0),
                (2, 2, 1., 3., 0., 1), (2, 3, 1., 2., 0., 0),
                (2, 4, 1., 1., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [2., 0., 1.],
                [1., 1., 0.],
                [0., 0., 1.],
                [0., 1., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99., 99., 100.],
                [97., 101., 98.],
                [100., 98., 101.],
                [102., 100., 99.],
                [102., 99., 100.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_signals_size_type(self):
        entries = [True, False, True, True, True]
        exits = [False, True, False, False, True]
        portfolio = vbt.Portfolio.from_signals(
            price, entries=entries, exits=exits, size=[1, 2, 3, 4, 5],
            size_type=SizeType.Shares, accumulate=False
        )
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 2, 3., 3., 0., 0), (0, 4, 2., 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio.shares,
            pd.Series(np.array([1., 0., 3., 3., 5.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio.cash,
            pd.Series(np.array([ 99., 101.,  92.,  92.,  90.]), index=price.index, name=price.name)
        )
        portfolio2 = vbt.Portfolio.from_signals(
            price, entries=entries, exits=exits, size=[1, 2, 3, 4, 5],
            size_type=SizeType.Cash, accumulate=False
        )
        record_arrays_close(
            portfolio2.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 2, 1., 3., 0., 0), (0, 4, 93., 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio2.shares,
            pd.Series(np.array([ 1.,  0.,  1.,  1., 94.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio2.cash,
            pd.Series(np.array([ 99., 101.,  98.,  98.,   5.]), index=price.index, name=price.name)
        )
        portfolio3 = vbt.Portfolio.from_signals(
            price, entries=entries, exits=exits, size=[1, 2, 3, 4, 5],
            size_type=SizeType.Shares, accumulate=True
        )
        record_arrays_close(
            portfolio3.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 2, 3., 3., 0., 0), (0, 3, 4., 2., 0., 0),
                (0, 4, 2., 1., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio3.shares,
            pd.Series(np.array([1., 0., 3., 7., 5.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio3.cash,
            pd.Series(np.array([ 99., 101.,  92.,  84.,  86.]), index=price.index, name=price.name)
        )
        portfolio4 = vbt.Portfolio.from_signals(
            price, entries=entries, exits=exits, size=[1, 2, 3, 4, 5],
            size_type=SizeType.Cash, accumulate=True
        )
        record_arrays_close(
            portfolio4.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 1),
                (0, 2, 1., 3., 0., 0), (0, 3, 2., 2., 0., 0),
                (0, 4, 89., 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio4.shares,
            pd.Series(np.array([ 1.,  0.,  1.,  3., 92.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio4.cash,
            pd.Series(np.array([ 99., 101.,  98.,  94.,   5.]), index=price.index, name=price.name)
        )
        try:
            _ = vbt.Portfolio.from_signals(
                price, entries=entries, exits=exits, size=[1, 2, 3, 4, 5],
                size_type=SizeType.TargetShares
            )
            raise Exception
        except:
            pass
        try:
            _ = vbt.Portfolio.from_signals(
                price, entries=entries, exits=exits, size=[1, 2, 3, 4, 5],
                size_type=SizeType.TargetCash
            )
            raise Exception
        except:
            pass
        try:
            _ = vbt.Portfolio.from_signals(
                price, entries=entries, exits=exits, size=[1, 2, 3, 4, 5],
                size_type=SizeType.TargetValue
            )
            raise Exception
        except:
            pass
        try:
            _ = vbt.Portfolio.from_signals(
                price, entries=entries, exits=exits, size=[1, 2, 3, 4, 5],
                size_type=SizeType.TargetPercent
            )
            raise Exception
        except:
            pass

    def test_from_orders(self):
        portfolio = vbt.Portfolio.from_orders(price, order_size['a'])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 0.1, 2., 0., 0),
                (0, 2, 1., 3., 0., 1), (0, 3, 0.1, 2., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio.shares,
            pd.Series(np.array([1., 1.1, 0.1, 0., 0.]), index=price.index, name=('a', 'Price'))
        )
        pd.testing.assert_series_equal(
            portfolio.cash,
            pd.Series(np.array([99., 98.8, 101.8, 102., 102.]), index=price.index, name=('a', 'Price'))
        )
        portfolio2 = vbt.Portfolio.from_orders(price, order_size)
        record_arrays_close(
            portfolio2.orders.records_arr,
            np.array([
                (0, 0, 1.00000000e+00, 1., 0., 0),
                (0, 1, 1.00000000e-01, 2., 0., 0),
                (0, 2, 1.00000000e+00, 3., 0., 1),
                (0, 3, 1.00000000e-01, 2., 0., 1),
                (1, 0, 1.00000000e+00, 1., 0., 0),
                (1, 1, 1.00000000e+00, 2., 0., 0),
                (1, 2, 1.00000000e+00, 3., 0., 0),
                (1, 3, 1.00000000e+00, 2., 0., 0),
                (1, 4, 4.00000000e+00, 1., 0., 1),
                (2, 0, 1.00000000e+02, 1., 0., 0),
                (2, 1, 1.00000000e+02, 2., 0., 1),
                (2, 2, 6.66666667e+01, 3., 0., 0),
                (2, 3, 6.66666667e+01, 2., 0., 1),
                (2, 4, 1.33333333e+02, 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio2.shares,
            pd.DataFrame(np.array([
                [1.00000000e+00, 1.00000000e+00, 1.00000000e+02],
                [1.10000000e+00, 2.00000000e+00, 0.00000000e+00],
                [1.00000000e-01, 3.00000000e+00, 6.66666667e+01],
                [0.00000000e+00, 4.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 1.33333333e+02]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio2.cash,
            pd.DataFrame(np.array([
                [99., 99., 0.],
                [98.8, 97., 200.],
                [101.8, 94., 0.],
                [102., 92., 133.33333333],
                [102., 96., 0.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_orders_init_capital(self):
        portfolio = vbt.Portfolio.from_orders(price, order_size, init_capital=[1, 10, 100])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 2, 1., 3., 0., 1),
                (1, 0, 1., 1., 0., 0), (1, 1, 1., 2., 0., 0),
                (1, 2, 1., 3., 0., 0), (1, 3, 1., 2., 0., 0),
                (1, 4, 4., 1., 0., 1), (2, 0, 100., 1., 0., 0),
                (2, 1, 100., 2., 0., 1), (2, 2, 66.66666667, 3., 0., 0),
                (2, 3, 66.66666667, 2., 0., 1), (2, 4, 133.33333333, 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 100.],
                [1., 2., 0.],
                [0., 3., 66.66666667],
                [0., 4., 0.],
                [0., 0., 133.33333333]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [0., 9., 0.],
                [0., 7., 200.],
                [3., 4., 0.],
                [3., 2., 133.33333333],
                [3., 6., 0.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_orders_fees(self):
        portfolio = vbt.Portfolio.from_orders(price, order_size, fees=[0., 0.01, np.inf])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0.e+00, 0), (0, 1, 0.1, 2., 0.e+00, 0),
                (0, 2, 1., 3., 0.e+00, 1), (0, 3, 0.1, 2., 0.e+00, 1),
                (1, 0, 1., 1., 1.e-02, 0), (1, 1, 1., 2., 2.e-02, 0),
                (1, 2, 1., 3., 3.e-02, 0), (1, 3, 1., 2., 2.e-02, 0),
                (1, 4, 4., 1., 4.e-02, 1), (2, 0, 0., 1., 1.e+02, 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [1.1, 2., 0.],
                [0.1, 3., 0.],
                [0., 4., 0.],
                [0., 0., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99., 98.99, 0.],
                [98.8, 96.97, 0.],
                [101.8, 93.94, 0.],
                [102., 91.92, 0.],
                [102., 95.88, 0.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_orders_fixed_fees(self):
        portfolio = vbt.Portfolio.from_orders(price, order_size, fixed_fees=[0., 1., np.inf])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 0.1, 2., 0., 0),
                (0, 2, 1., 3., 0., 1), (0, 3, 0.1, 2., 0., 1),
                (1, 0, 1., 1., 1., 0), (1, 1, 1., 2., 1., 0),
                (1, 2, 1., 3., 1., 0), (1, 3, 1., 2., 1., 0),
                (1, 4, 4., 1., 1., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [1.1, 2., 0.],
                [0.1, 3., 0.],
                [0., 4., 0.],
                [0., 0., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99., 98., 100.],
                [98.8, 95., 100.],
                [101.8, 91., 100.],
                [102., 88., 100.],
                [102., 91., 100.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_orders_slippage(self):
        portfolio = vbt.Portfolio.from_orders(price, order_size, slippage=[0., 0.01, np.inf])
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 0.1, 2., 0., 0),
                (0, 2, 1., 3., 0., 1), (0, 3, 0.1, 2., 0., 1),
                (1, 0, 1., 1.01, 0., 0), (1, 1, 1., 2.02, 0., 0),
                (1, 2, 1., 3.03, 0., 0), (1, 3, 1., 2.02, 0., 0),
                (1, 4, 4., 0.99, 0., 1), (2, 0, 0., np.inf, 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1., 1., 0.],
                [1.1, 2., 0.],
                [0.1, 3., 0.],
                [0., 4., 0.],
                [0., 0., 0.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99., 98.99, 0.],
                [98.8, 96.97, 0.],
                [101.8, 93.94, 0.],
                [102., 91.92, 0.],
                [102., 95.88, 0.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_orders_price(self):
        portfolio = vbt.Portfolio.from_orders(price, order_size, order_price=0.9 * price)
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1.00000000e+00, 0.9, 0., 0),
                (0, 1, 1.00000000e-01, 1.8, 0., 0),
                (0, 2, 1.00000000e+00, 2.7, 0., 1),
                (0, 3, 1.00000000e-01, 1.8, 0., 1),
                (1, 0, 1.00000000e+00, 0.9, 0., 0),
                (1, 1, 1.00000000e+00, 1.8, 0., 0),
                (1, 2, 1.00000000e+00, 2.7, 0., 0),
                (1, 3, 1.00000000e+00, 1.8, 0., 0),
                (1, 4, 4.00000000e+00, 0.9, 0., 1),
                (2, 0, 1.11111111e+02, 0.9, 0., 0),
                (2, 1, 1.11111111e+02, 1.8, 0., 1),
                (2, 2, 7.40740741e+01, 2.7, 0., 0),
                (2, 3, 7.40740741e+01, 1.8, 0., 1),
                (2, 4, 1.48148148e+02, 0.9, 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [1.00000000e+00, 1.00000000e+00, 1.11111111e+02],
                [1.10000000e+00, 2.00000000e+00, 0.00000000e+00],
                [1.00000000e-01, 3.00000000e+00, 7.40740741e+01],
                [0.00000000e+00, 4.00000000e+00, 0.00000000e+00],
                [0.00000000e+00, 0.00000000e+00, 1.48148148e+02]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [99.1, 99.1, 0.],
                [98.92, 97.3, 200.],
                [101.62, 94.6, 0.],
                [101.8, 92.8, 133.33333333],
                [101.8, 96.4, 0.]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_orders_size_type(self):
        portfolio = vbt.Portfolio.from_orders(price, [1, 2, 3, 4, 5], size_type=SizeType.Shares)
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 2., 2., 0., 0), (0, 2, 3., 3., 0., 0),
                (0, 3, 4., 2., 0., 0), (0, 4, 5., 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio.shares,
            pd.Series(np.array([ 1.,  3.,  6., 10., 15.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio.cash,
            pd.Series(np.array([99., 95., 86., 78., 73.]), index=price.index, name=price.name)
        )
        portfolio2 = vbt.Portfolio.from_orders(price, [1, 2, 3, 4, 5], size_type=SizeType.TargetShares)
        record_arrays_close(
            portfolio2.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 2, 1., 3., 0., 0), (0, 3, 1., 2., 0., 0),
                (0, 4, 1., 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio2.shares,
            pd.Series(np.array([1., 2., 3., 4., 5.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio2.cash,
            pd.Series(np.array([99., 97., 94., 92., 91.]), index=price.index, name=price.name)
        )
        portfolio3 = vbt.Portfolio.from_orders(price, [1, 2, 3, 4, 5], size_type=SizeType.Cash)
        record_arrays_close(
            portfolio3.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 2, 1., 3., 0., 0), (0, 3, 2., 2., 0., 0),
                (0, 4, 5., 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio3.shares,
            pd.Series(np.array([ 1.,  2.,  3.,  5., 10.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio3.cash,
            pd.Series(np.array([99., 97., 94., 90., 85.]), index=price.index, name=price.name)
        )
        portfolio4 = vbt.Portfolio.from_orders(price, [1, 2, 3, 4, 5], size_type=SizeType.TargetCash)
        record_arrays_close(
            portfolio4.orders.records_arr,
            np.array([
                (0, 0, 99., 1., 0., 0), (0, 1, 0.5, 2., 0., 1),
                (0, 2, 0.33333333, 3., 0., 1), (0, 3, 0.5, 2., 0., 1),
                (0, 4, 1., 1., 0., 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio4.shares,
            pd.Series(np.array([99., 98.5, 98.16666667, 97.66666667, 96.66666667]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio4.cash,
            pd.Series(np.array([1., 2., 3., 4., 5.]), index=price.index, name=price.name)
        )
        portfolio5 = vbt.Portfolio.from_orders(price, [1, 2, 3, 4, 5], size_type=SizeType.TargetValue)
        record_arrays_close(
            portfolio5.orders.records_arr,
            np.array([
                (0, 0, 1., 1., 0., 0), (0, 3, 1., 2., 0., 0),
                (0, 4, 3., 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio5.shares,
            pd.Series(np.array([1., 1., 1., 2., 5.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio5.cash,
            pd.Series(np.array([99., 99., 99., 97., 94.]), index=price.index, name=price.name)
        )
        portfolio6 = vbt.Portfolio.from_orders(price, [0.1, 0.2, 0.3, 0.4, 0.5], size_type=SizeType.TargetPercent)
        record_arrays_close(
            portfolio6.orders.records_arr,
            np.array([
                (0, 0, 10., 1., 0., 0), (0, 1, 1., 2., 0., 0),
                (0, 2, 1.1, 3., 0., 0), (0, 3, 9.68, 2., 0., 0),
                (0, 4, 21.78, 1., 0., 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio6.shares,
            pd.Series(np.array([10., 11., 12.1, 21.78, 43.56]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio6.cash,
            pd.Series(np.array([90., 88., 84.7, 65.34, 43.56]), index=price.index, name=price.name)
        )

    def test_from_order_func(self):
        portfolio = vbt.Portfolio.from_order_func(
            price,
            order_func_nb,
            price.values[:, None],
            np.full(price.shape, 0.01)[:, None],
            np.full(price.shape, 1)[:, None],
            np.full(price.shape, 0.01)[:, None]
        )
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (0, 0, 1., 1.01, 1.0101, 0), (0, 1, 1., 1.98, 1.0198, 1),
                (0, 2, 1., 3.03, 1.0303, 0), (0, 3, 1., 1.98, 1.0198, 1),
                (0, 4, 1., 1.01, 1.0101, 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_series_equal(
            portfolio.shares,
            pd.Series(np.array([1., 0., 1., 0., 1.]), index=price.index, name=price.name)
        )
        pd.testing.assert_series_equal(
            portfolio.cash,
            pd.Series(np.array([97.9799, 98.9401, 94.8798, 95.84, 93.8199]), index=price.index, name=price.name)
        )
        portfolio2 = vbt.Portfolio.from_order_func(
            price.vbt.tile(3, keys=entries.columns),
            order_func_nb,
            price.vbt.tile(3).values,
            np.full((price.shape[0], 3), 0.01),
            np.full((price.shape[0], 3), 1),
            np.full((price.shape[0], 3), 0.01)
        )
        record_arrays_close(
            portfolio2.orders.records_arr,
            np.array([
                (0, 0, 1., 1.01, 1.0101, 0), (0, 1, 1., 1.98, 1.0198, 1),
                (0, 2, 1., 3.03, 1.0303, 0), (0, 3, 1., 1.98, 1.0198, 1),
                (0, 4, 1., 1.01, 1.0101, 0), (1, 0, 2., 1.01, 1.0202, 0),
                (1, 1, 2., 1.98, 1.0396, 1), (1, 2, 2., 3.03, 1.0606, 0),
                (1, 3, 2., 1.98, 1.0396, 1), (1, 4, 2., 1.01, 1.0202, 0),
                (2, 0, 3., 1.01, 1.0303, 0), (2, 1, 3., 1.98, 1.0594, 1),
                (2, 2, 3., 3.03, 1.0909, 0), (2, 3, 3., 1.98, 1.0594, 1),
                (2, 4, 3., 1.01, 1.0303, 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio2.shares,
            pd.DataFrame(np.array([
                [1., 2., 3.],
                [0., 0., 0.],
                [1., 2., 3.],
                [0., 0., 0.],
                [1., 2., 3.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio2.cash,
            pd.DataFrame(np.array([
                [97.9799, 96.9598, 95.9397],
                [98.9401, 99.8802, 100.8203],
                [94.8798, 92.7596, 90.6394],
                [95.84, 95.68, 95.52],
                [93.8199, 92.6398, 91.4597]
            ]), index=price.index, columns=entries.columns)
        )

    def test_from_order_func_init_capital(self):
        portfolio = vbt.Portfolio.from_order_func(
            price.vbt.tile(3, keys=entries.columns),
            order_func_nb,
            price.vbt.tile(3).values,
            np.full((price.shape[0], 3), 0.01),
            np.full((price.shape[0], 3), 1),
            np.full((price.shape[0], 3), 0.01),
            init_capital=[1, 10, 100]
        )
        record_arrays_close(
            portfolio.orders.records_arr,
            np.array([
                (1, 0, 2., 1.01, 1.0202, 0), (1, 1, 2., 1.98, 1.0396, 1),
                (1, 2, 2., 3.03, 1.0606, 0), (1, 3, 2., 1.98, 1.0396, 1),
                (1, 4, 2., 1.01, 1.0202, 0), (2, 0, 3., 1.01, 1.0303, 0),
                (2, 1, 3., 1.98, 1.0594, 1), (2, 2, 3., 3.03, 1.0909, 0),
                (2, 3, 3., 1.98, 1.0594, 1), (2, 4, 3., 1.01, 1.0303, 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(
            portfolio.shares,
            pd.DataFrame(np.array([
                [0., 2., 3.],
                [0., 0., 0.],
                [0., 2., 3.],
                [0., 0., 0.],
                [0., 2., 3.]
            ]), index=price.index, columns=entries.columns)
        )
        pd.testing.assert_frame_equal(
            portfolio.cash,
            pd.DataFrame(np.array([
                [1., 6.9598, 95.9397],
                [1., 9.8802, 100.8203],
                [1., 2.7596, 90.6394],
                [1., 5.68, 95.52],
                [1., 2.6398, 91.4597]
            ]), index=price.index, columns=entries.columns)
        )

    def test_single_params(self):
        portfolio = vbt.Portfolio.from_signals(
            price, entries, exits,
            init_capital=init_capital[0],
            freq='1 days',
            year_freq='252 days',
            levy_alpha=levy_alpha[0],
            risk_free=risk_free[0],
            required_return=required_return[0],
            cutoff=cutoff[0],
            factor_returns=factor_returns['a']
        )
        pd.testing.assert_series_equal(
            portfolio.init_capital,
            pd.Series(np.full(3, init_capital[0]), index=entries.columns)
        )
        assert portfolio.freq == day_dt
        assert portfolio.year_freq == 252 * day_dt
        assert portfolio.levy_alpha == levy_alpha[0]
        assert portfolio.risk_free == risk_free[0]
        assert portfolio.required_return == required_return[0]
        assert portfolio.cutoff == cutoff[0]
        pd.testing.assert_series_equal(portfolio.factor_returns, factor_returns['a'])

        # indexing
        assert portfolio['a'].init_capital == init_capital[0]
        assert portfolio.freq == day_dt
        assert portfolio.year_freq == 252 * day_dt
        assert portfolio['a'].levy_alpha == levy_alpha[0]
        assert portfolio['a'].risk_free == risk_free[0]
        assert portfolio['a'].required_return == required_return[0]
        assert portfolio['a'].cutoff == cutoff[0]
        pd.testing.assert_series_equal(portfolio['a'].factor_returns, factor_returns['a'])

    def test_multiple_params(self):
        pd.testing.assert_series_equal(
            test_portfolio.init_capital,
            pd.Series(init_capital, index=entries.columns)
        )
        assert test_portfolio.freq == day_dt
        assert test_portfolio.year_freq == 252 * day_dt
        np.testing.assert_array_equal(test_portfolio.levy_alpha, np.array(levy_alpha))
        np.testing.assert_array_equal(test_portfolio.risk_free, np.array(risk_free))
        np.testing.assert_array_equal(test_portfolio.required_return, np.array(required_return))
        np.testing.assert_array_equal(test_portfolio.cutoff, np.array(cutoff))
        pd.testing.assert_frame_equal(test_portfolio.factor_returns, factor_returns)

        # indexing
        assert test_portfolio['a'].init_capital == init_capital[0]
        assert test_portfolio['a'].freq == day_dt
        assert test_portfolio['a'].year_freq == 252 * day_dt
        assert test_portfolio['a'].levy_alpha == levy_alpha[0]
        assert test_portfolio['a'].risk_free == risk_free[0]
        assert test_portfolio['a'].required_return == required_return[0]
        assert test_portfolio['a'].cutoff == cutoff[0]
        pd.testing.assert_series_equal(test_portfolio['a'].factor_returns, factor_returns['a'])

    def test_indexing(self):
        pd.testing.assert_series_equal(
            test_portfolio.iloc[:, 0].main_price,
            test_portfolio.main_price.iloc[:, 0]
        )
        pd.testing.assert_series_equal(
            test_portfolio.loc[:, 'a'].main_price,
            test_portfolio.main_price.loc[:, 'a']
        )
        pd.testing.assert_series_equal(
            test_portfolio['a'].main_price,
            test_portfolio.main_price['a']
        )
        pd.testing.assert_frame_equal(
            test_portfolio.iloc[:, [0, 1]].main_price,
            test_portfolio.main_price.iloc[:, [0, 1]]
        )
        pd.testing.assert_frame_equal(
            test_portfolio.loc[:, ['a', 'b']].main_price,
            test_portfolio.main_price.loc[:, ['a', 'b']]
        )
        pd.testing.assert_frame_equal(
            test_portfolio[['a', 'b']].main_price,
            test_portfolio.main_price[['a', 'b']]
        )
        try:
            _ = test_portfolio.iloc[::2, :]  # changing time not supported
            raise Exception
        except:
            pass
        _ = test_portfolio.iloc[np.arange(test_portfolio.wrapper.shape[0]), :]  # won't change time

    def test_records(self):
        # orders
        record_arrays_close(
            test_portfolio.orders.records_arr,
            np.array([
                (0, 0, 99.00990099, 1., 0.99009901, 0),
                (0, 3, 99.00990099, 2., 1.98019802, 1),
                (1, 0, 198.01980198, 1., 1.98019802, 0),
                (1, 1, 198.01980198, 2., 3.96039604, 1),
                (1, 2, 129.39907852, 3., 3.88197236, 0),
                (1, 3, 129.39907852, 2., 2.58798157, 1),
                (1, 4, 253.67344106, 1., 2.53673441, 0),
                (2, 1, 148.51485149, 2., 2.97029703, 0),
                (2, 2, 148.51485149, 3., 4.45544554, 1),
                (2, 3, 218.36094501, 2., 4.3672189, 0),
                (2, 4, 218.36094501, 1., 2.18360945, 1)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(test_portfolio.orders.main_price, test_portfolio.main_price)
        assert test_portfolio.orders.wrapper.freq == day_dt
        # trades
        record_arrays_close(
            test_portfolio.trades.records_arr,
            np.array([
                (0, 99.00990099, 0, 1., 0.99009901, 3, 2., 1.98019802, 96.03960396, 0.96039604, 1, 0),
                (1, 198.01980198, 0, 1., 1.98019802, 1, 2., 3.96039604, 192.07920792, 0.96039604, 1, 1),
                (1, 129.39907852, 2, 3., 3.88197236, 3, 2., 2.58798157, -135.86903245, -0.34653465, 1, 2),
                (1, 253.67344106, 4, 1., 2.53673441, 4, 1., 0., -2.53673441, -0.00990099, 0, 3),
                (2, 148.51485149, 1, 2., 2.97029703, 2, 3., 4.45544554, 141.08910891, 0.47029703, 1, 4),
                (2, 218.36094501, 3, 2., 4.3672189, 4, 1., 2.18360945, -224.91177336, -0.50990099, 1, 5)
            ], dtype=trade_dt)
        )
        pd.testing.assert_frame_equal(test_portfolio.trades.main_price, test_portfolio.main_price)
        assert test_portfolio.trades.wrapper.freq == day_dt
        # positions
        record_arrays_close(
            test_portfolio.positions.records_arr,
            np.array([
                (0, 99.00990099, 0, 1., 0.99009901, 3, 2., 1.98019802, 96.03960396, 0.96039604, 1),
                (1, 198.01980198, 0, 1., 1.98019802, 1, 2., 3.96039604, 192.07920792, 0.96039604, 1),
                (1, 129.39907852, 2, 3., 3.88197236, 3, 2., 2.58798157, -135.86903245, -0.34653465, 1),
                (1, 253.67344106, 4, 1., 2.53673441, 4, 1., 0., -2.53673441, -0.00990099, 0),
                (2, 148.51485149, 1, 2., 2.97029703, 2, 3., 4.45544554, 141.08910891, 0.47029703, 1),
                (2, 218.36094501, 3, 2., 4.3672189, 4, 1., 2.18360945, -224.91177336, -0.50990099, 1)
            ], dtype=position_dt)
        )
        pd.testing.assert_frame_equal(test_portfolio.positions.main_price, test_portfolio.main_price)
        assert test_portfolio.positions.wrapper.freq == day_dt
        # drawdowns
        record_arrays_close(
            test_portfolio.drawdowns.records_arr,
            np.array([
                (0, 2, 3, 4, 0), (1, 1, 4, 4, 0), (2, 0, 1, 2, 1), (2, 2, 4, 4, 0)
            ], dtype=drawdown_dt)
        )
        pd.testing.assert_frame_equal(test_portfolio.drawdowns.ts, test_portfolio.equity)
        assert test_portfolio.drawdowns.wrapper.freq == day_dt

    def test_equity(self):
        pd.testing.assert_series_equal(
            test_portfolio['a'].equity,
            pd.Series(
                np.array([99.00990099, 198.01980198, 297.02970297, 196.03960396, 196.03960396]),
                index=price.index,
                name='a'
            )
        )
        pd.testing.assert_frame_equal(
            test_portfolio.equity,
            pd.DataFrame(
                np.array([
                    [99.00990099, 198.01980198, 300.],
                    [198.01980198, 392.07920792, 297.02970297],
                    [297.02970297, 388.19723557, 441.08910891],
                    [196.03960396, 256.21017547, 436.72189001],
                    [196.03960396, 253.67344106, 216.17733556]
                ]),
                index=price.index,
                columns=entries.columns
            )
        )

    def test_final_equity(self):
        assert test_portfolio['a'].final_equity == 196.03960396039605
        pd.testing.assert_series_equal(
            test_portfolio.final_equity,
            pd.Series(np.array([196.03960396, 253.67344106, 216.17733556]), index=entries.columns)
        )

    def test_total_profit(self):
        assert test_portfolio['a'].total_profit == 96.03960396039605
        pd.testing.assert_series_equal(
            test_portfolio.total_profit,
            pd.Series(np.array([96.03960396, 53.67344106, -83.82266444]), index=entries.columns)
        )

    def test_drawdown(self):
        pd.testing.assert_series_equal(
            test_portfolio['a'].drawdown,
            pd.Series(np.array([0., 0., 0., -0.34, -0.34]), index=price.index, name='a')
        )
        pd.testing.assert_frame_equal(
            test_portfolio.drawdown,
            pd.DataFrame(
                np.array([
                    [0., 0., 0.],
                    [0., 0., -0.00990099],
                    [0., -0.00990099, 0.],
                    [-0.34, -0.34653465, -0.00990099],
                    [-0.34, -0.35300461, -0.50990099]
                ]),
                index=price.index,
                columns=entries.columns
            )
        )

    def test_max_drawdown(self):
        assert test_portfolio['a'].max_drawdown == -0.33999999999999997
        pd.testing.assert_series_equal(
            test_portfolio.max_drawdown,
            pd.Series(np.array([-0.34, -0.35300461, -0.50990099]), index=entries.columns)
        )

    def test_buy_and_hold_return(self):
        assert test_portfolio['a'].buy_and_hold_return == 0.0
        pd.testing.assert_series_equal(
            test_portfolio.buy_and_hold_return,
            pd.Series(np.array([0.0, 0.0, 0.0]), index=entries.columns)
        )

    def test_returns(self):
        pd.testing.assert_series_equal(
            test_portfolio['a'].returns,
            pd.Series(np.array([-0.00990099, 1., 0.5, -0.34, 0.]), index=price.index, name='a')
        )
        pd.testing.assert_frame_equal(
            test_portfolio.returns,
            pd.DataFrame(
                np.array([
                    [-0.00990099, -0.00990099, 0.],
                    [1., 0.98, -0.00990099],
                    [0.5, -0.00990099, 0.485],
                    [-0.34, -0.34, -0.00990099],
                    [0., -0.00990099, -0.505]
                ]),
                index=price.index,
                columns=entries.columns
            )
        )

    def test_daily_returns(self):
        pd.testing.assert_series_equal(
            test_portfolio['a'].daily_returns,
            test_portfolio['a'].returns
        )
        pd.testing.assert_frame_equal(
            test_portfolio.returns,
            test_portfolio.daily_returns
        )

    def test_annual_return(self):
        pd.testing.assert_series_equal(
            test_portfolio['a'].annual_returns,
            pd.Series(
                np.array([0.96039604]),
                index=pd.DatetimeIndex(['2020-01-01'], dtype='datetime64[ns]', freq='252D'),
                name='a'
            )
        )
        pd.testing.assert_frame_equal(
            test_portfolio.annual_returns,
            pd.DataFrame(
                np.array([[0.96039604, 0.26836721, -0.27940888]]),
                index=pd.DatetimeIndex(['2020-01-01'], dtype='datetime64[ns]', freq='252D'),
                columns=entries.columns
            )
        )

    def test_cumulative_returns(self):
        pd.testing.assert_series_equal(
            test_portfolio['a'].cumulative_returns,
            pd.Series(
                np.array([-0.00990099, 0.98019802, 1.97029703, 0.96039604, 0.96039604]),
                index=price.index,
                name='a'
            )
        )
        pd.testing.assert_frame_equal(
            test_portfolio.cumulative_returns,
            pd.DataFrame(
                np.array([
                    [-0.00990099, -0.00990099, 0.],
                    [0.98019802, 0.96039604, -0.00990099],
                    [1.97029703, 0.94098618, 0.47029703],
                    [0.96039604, 0.28105088, 0.45573963],
                    [0.96039604, 0.26836721, -0.27940888]
                ]),
                index=price.index,
                columns=entries.columns
            )
        )

    def test_total_return(self):
        assert test_portfolio['a'].total_return == 0.9603960396039604
        pd.testing.assert_series_equal(
            test_portfolio.total_return,
            pd.Series(np.array([0.96039604, 0.26836721, -0.27940888]), index=entries.columns)
        )

    def test_annualized_return(self):
        assert test_portfolio['a'].annualized_return == 542161095949729.56
        pd.testing.assert_series_equal(
            test_portfolio.annualized_return,
            pd.Series(np.array([5.42161096e+14, 1.59788495e+05, -9.99999933e-01]), index=entries.columns)
        )

    def test_annualized_volatility(self):
        assert test_portfolio['a'].annualized_volatility == 132.2191242654978
        pd.testing.assert_series_equal(
            test_portfolio.annualized_volatility,
            pd.Series(np.array([132.21912427, 7.94440058, 2.21101734]), index=entries.columns)
        )

    def test_calmar_ratio(self):
        assert test_portfolio['a'].calmar_ratio == 1594591458675675.2
        pd.testing.assert_series_equal(
            test_portfolio.calmar_ratio,
            pd.Series(np.array([1.59459146e+15, 4.52652718e+05, -1.96116492e+00]), index=entries.columns)
        )

    def test_omega_ratio(self):
        assert test_portfolio['a'].omega_ratio == 3.882163392568907
        pd.testing.assert_series_equal(
            test_portfolio.omega_ratio,
            pd.Series(np.array([3.88216339, 2.11948842, 0.69950732]), index=entries.columns)
        )

    def test_sharpe_ratio(self):
        assert test_portfolio['a'].sharpe_ratio == 6.656842846055576
        pd.testing.assert_series_equal(
            test_portfolio.sharpe_ratio,
            pd.Series(np.array([6.65684285, 3.23737078, -1.72149475]), index=entries.columns)
        )

    def test_downside_risk(self):
        assert test_portfolio['a'].downside_risk == 3.2969960073204563
        pd.testing.assert_series_equal(
            test_portfolio.downside_risk,
            pd.Series(np.array([3.29699601, 4.62150127, 6.84668922]), index=entries.columns)
        )

    def test_sortino_ratio(self):
        assert test_portfolio['a'].sortino_ratio == 9.93783129438448
        pd.testing.assert_series_equal(
            test_portfolio.sortino_ratio,
            pd.Series(np.array([9.93783129, -4.24992412, -11.33482437]), index=entries.columns)
        )

    def test_information_ratio(self):
        assert test_portfolio['a'].information_ratio == -1.9476532644416638
        pd.testing.assert_series_equal(
            test_portfolio.information_ratio,
            pd.Series(np.array([-1.94765326, -1.78416485, -3.03369131]), index=entries.columns)
        )

    def test_beta(self):
        assert test_portfolio['a'].beta == 0.2936193619361937
        pd.testing.assert_series_equal(
            test_portfolio.beta,
            pd.Series(np.array([0.29361936, 0.04712871, 0.31884403]), index=entries.columns)
        )

    def test_alpha(self):
        assert test_portfolio['a'].alpha == -1.
        pd.testing.assert_series_equal(
            test_portfolio.alpha,
            pd.Series(np.array([-1., 92.48601804, -1.]), index=entries.columns)
        )

    def test_tail_ratio(self):
        assert test_portfolio['a'].tail_ratio == 3.284908933217693
        pd.testing.assert_series_equal(
            test_portfolio.tail_ratio,
            pd.Series(np.array([3.28490893, 2.85429315, 0.95571164]), index=entries.columns)
        )

    def test_value_at_risk(self):
        assert test_portfolio['a'].value_at_risk == -0.32679603960396036
        pd.testing.assert_series_equal(
            test_portfolio.value_at_risk,
            pd.Series(np.array([-0.32679604, -0.31359208, -0.44558812]), index=entries.columns)
        )

    def test_conditional_value_at_risk(self):
        assert test_portfolio['a'].conditional_value_at_risk == -0.33999999999999997
        pd.testing.assert_series_equal(
            test_portfolio.conditional_value_at_risk,
            pd.Series(np.array([-0.34, -0.34, -0.505]), index=entries.columns)
        )

    def test_capture(self):
        assert test_portfolio['a'].capture == 8.411412319796107e-88
        pd.testing.assert_series_equal(
            test_portfolio.capture,
            pd.Series(np.array([8.41141232e-088, 2.64287063e-104, -4.72462219e-116]), index=entries.columns)
        )

    def test_up_capture(self):
        assert test_portfolio['a'].up_capture == 8.411412319796107e-88
        pd.testing.assert_series_equal(
            test_portfolio.up_capture,
            pd.Series(np.array([8.41141232e-088, 2.64287063e-104, -4.72462219e-116]), index=entries.columns)
        )

    def test_down_capture(self):
        assert np.isnan(test_portfolio['a'].down_capture)
        pd.testing.assert_series_equal(
            test_portfolio.down_capture,
            pd.Series(np.array([np.nan, np.nan, np.nan]), index=entries.columns)
        )

    def test_stats(self):
        pd.testing.assert_series_equal(
            test_portfolio['c'].stats,
            pd.Series(
                np.array([
                    pd.Timestamp('2020-01-01 00:00:00'),
                    pd.Timestamp('2020-01-05 00:00:00'),
                    pd.Timedelta('5 days 00:00:00'),
                    40.0,
                    -83.82266444466228,
                    -27.940888148220754,
                    0.0,
                    50.99009900990099,
                    25.99009900990099,
                    pd.Timedelta('2 days 00:00:00'),
                    pd.Timedelta('2 days 00:00:00'), 2,
                    50.0,
                    47.029702970297045,
                    -50.99009900990099,
                    -1.9801980198019735,
                    pd.Timedelta('1 days 00:00:00'),
                    pd.Timedelta('1 days 00:00:00'),
                    -41.91133222233114,
                    -0.22902312127116764,
                    -1.7214947542575758,
                    -11.33482436788235,
                    -1.9611649167074043
                ]),
                index=pd.Index([
                    'Start', 'End', 'Duration', 'Holding Duration [%]', 'Total Profit',
                    'Total Return [%]', 'Buy & Hold Return [%]', 'Max. Drawdown [%]',
                    'Avg. Drawdown [%]', 'Max. Drawdown Duration', 'Avg. Drawdown Duration',
                    'Num. Trades', 'Win Rate [%]', 'Best Trade [%]', 'Worst Trade [%]',
                    'Avg. Trade [%]', 'Max. Trade Duration', 'Avg. Trade Duration',
                    'Expectancy', 'SQN', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'
                ], dtype='object'),
                name='c'
            )
        )
