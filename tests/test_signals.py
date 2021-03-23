import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
import pytest

from vectorbt.signals import nb

seed = 42

day_dt = np.timedelta64(86400000000000)

sig = pd.DataFrame([
    [True, False, False],
    [False, True, False],
    [False, False, True],
    [True, False, False],
    [False, True, False]
], index=pd.Index([
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    datetime(2020, 1, 3),
    datetime(2020, 1, 4),
    datetime(2020, 1, 5)
]), columns=['a', 'b', 'c'])

ts = pd.Series([1., 2., 3., 2., 1.], index=sig.index)

price = pd.DataFrame({
    'open': [10, 11, 12, 11, 10],
    'high': [11, 12, 13, 12, 11],
    'low': [9, 10, 11, 10, 9],
    'close': [10, 11, 12, 11, 10]
})


# ############# accessors.py ############# #


class TestAccessors:
    def test_freq(self):
        assert sig.vbt.signals.wrapper.freq == day_dt
        assert sig['a'].vbt.signals.wrapper.freq == day_dt
        assert sig.vbt.signals(freq='2D').wrapper.freq == day_dt * 2
        assert sig['a'].vbt.signals(freq='2D').wrapper.freq == day_dt * 2
        assert pd.Series([False, True]).vbt.signals.wrapper.freq is None
        assert pd.Series([False, True]).vbt.signals(freq='3D').wrapper.freq == day_dt * 3
        assert pd.Series([False, True]).vbt.signals(freq=np.timedelta64(4, 'D')).wrapper.freq == day_dt * 4

    @pytest.mark.parametrize(
        "test_n",
        [1, 2, 3, 4, 5],
    )
    def test_fshift(self, test_n):
        pd.testing.assert_series_equal(sig['a'].vbt.signals.fshift(test_n), sig['a'].shift(test_n, fill_value=False))
        np.testing.assert_array_equal(
            sig['a'].vbt.signals.fshift(test_n).values,
            nb.fshift_1d_nb(sig['a'].values, test_n)
        )
        pd.testing.assert_frame_equal(sig.vbt.signals.fshift(test_n), sig.shift(test_n, fill_value=False))

    def test_empty(self):
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.empty(5, index=np.arange(10, 15), name='a'),
            pd.Series(np.full(5, False), index=np.arange(10, 15), name='a')
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.empty((5, 3), index=np.arange(10, 15), columns=['a', 'b', 'c']),
            pd.DataFrame(np.full((5, 3), False), index=np.arange(10, 15), columns=['a', 'b', 'c'])
        )
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.empty_like(sig['a']),
            pd.Series(np.full(sig['a'].shape, False), index=sig['a'].index, name=sig['a'].name)
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.empty_like(sig),
            pd.DataFrame(np.full(sig.shape, False), index=sig.index, columns=sig.columns)
        )

    def test_generate(self):
        @njit
        def choice_func_nb(from_i, to_i, col, n):
            if col == 0:
                return np.arange(from_i, to_i)
            elif col == 1:
                return np.full(1, from_i)
            else:
                return np.full(1, to_i - n)

        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate(5, choice_func_nb, 1, index=sig['a'].index, name=sig['a'].name),
            pd.Series(
                np.array([True, True, True, True, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        with pytest.raises(Exception) as e_info:
            _ = pd.Series.vbt.signals.generate((5, 2), choice_func_nb, 1)
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.generate((5, 3), choice_func_nb, 1, index=sig.index, columns=sig.columns),
            pd.DataFrame(
                np.array([
                    [True, True, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, False],
                    [True, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

    def test_generate_both(self):
        @njit
        def entry_func_nb(from_i, to_i, col, temp_int):
            temp_int[0] = from_i
            return temp_int[:1]

        @njit
        def exit_func_nb(from_i, to_i, col, temp_int):
            temp_int[0] = from_i
            return temp_int[:1]

        temp_int = np.empty((sig.shape[0],), dtype=np.int_)

        en, ex = pd.Series.vbt.signals.generate_both(
            5, entry_func_nb, exit_func_nb, (temp_int,), (temp_int,),
            index=sig['a'].index, name=sig['a'].name)
        pd.testing.assert_series_equal(
            en,
            pd.Series(
                np.array([True, False, True, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            ex,
            pd.Series(
                np.array([False, True, False, True, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_both(
            (5, 3), entry_func_nb, exit_func_nb, (temp_int,), (temp_int,),
            index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [False, False, False],
                    [True, True, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, True, True],
                    [False, False, False],
                    [True, True, True],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        en, ex = pd.Series.vbt.signals.generate_both(
            (5,), entry_func_nb, exit_func_nb, (temp_int,), (temp_int,),
            index=sig['a'].index, name=sig['a'].name, entry_wait=1, exit_wait=0)
        pd.testing.assert_series_equal(
            en,
            pd.Series(
                np.array([True, True, True, True, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            ex,
            pd.Series(
                np.array([True, True, True, True, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        en, ex = pd.Series.vbt.signals.generate_both(
            (5,), entry_func_nb, exit_func_nb, (temp_int,), (temp_int,),
            index=sig['a'].index, name=sig['a'].name, entry_wait=0, exit_wait=1)
        pd.testing.assert_series_equal(
            en,
            pd.Series(
                np.array([True, True, True, True, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            ex,
            pd.Series(
                np.array([False, True, True, True, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )

    def test_generate_exits(self):
        @njit
        def choice_func_nb(from_i, to_i, col, temp_int):
            temp_int[0] = from_i
            return temp_int[:1]

        temp_int = np.empty((sig.shape[0],), dtype=np.int_)

        pd.testing.assert_series_equal(
            sig['a'].vbt.signals.generate_exits(choice_func_nb, temp_int),
            pd.Series(
                np.array([False, True, False, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_exits(choice_func_nb, temp_int),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_exits(choice_func_nb, temp_int, wait=0),
            pd.DataFrame(
                np.array([
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, False, False],
                    [False, True, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

    def test_clean(self):
        entries = pd.DataFrame([
            [True, False, True],
            [True, False, False],
            [True, True, True],
            [False, True, False],
            [False, True, True]
        ], index=sig.index, columns=sig.columns)
        exits = pd.Series([True, False, True, False, True], index=sig.index)
        pd.testing.assert_frame_equal(
            entries.vbt.signals.clean(),
            pd.DataFrame(
                np.array([
                    [True, False, True],
                    [False, False, False],
                    [False, True, True],
                    [False, False, False],
                    [False, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.clean(entries),
            pd.DataFrame(
                np.array([
                    [True, False, True],
                    [False, False, False],
                    [False, True, True],
                    [False, False, False],
                    [False, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            entries.vbt.signals.clean(exits)[0],
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, True, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            entries.vbt.signals.clean(exits)[1],
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            entries.vbt.signals.clean(exits, entry_first=False)[0],
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, True, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            entries.vbt.signals.clean(exits, entry_first=False)[1],
            pd.DataFrame(
                np.array([
                    [False, True, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.clean(entries, exits)[0],
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, True, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.clean(entries, exits)[1],
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        with pytest.raises(Exception) as e_info:
            _ = pd.Series.vbt.signals.clean(entries, entries, entries)

    def test_generate_random(self):
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate_random(
                5, n=3, seed=seed, index=sig['a'].index, name=sig['a'].name),
            pd.Series(
                np.array([False, True, True, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        with pytest.raises(Exception) as e_info:
            _ = pd.Series.vbt.signals.generate_random((5, 2), n=3)
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3), n=3, seed=seed, index=sig.index, columns=sig.columns),
            pd.DataFrame(
                np.array([
                    [False, False, True],
                    [True, True, True],
                    [True, True, False],
                    [False, True, True],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3), n=[0, 1, 2], seed=seed, index=sig.index, columns=sig.columns),
            pd.DataFrame(
                np.array([
                    [False, False, True],
                    [False, False, True],
                    [False, False, False],
                    [False, True, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate_random(
                5, prob=0.5, seed=seed, index=sig['a'].index, name=sig['a'].name),
            pd.Series(
                np.array([True, False, False, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        with pytest.raises(Exception) as e_info:
            _ = pd.Series.vbt.signals.generate_random((5, 2), prob=3)
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3), prob=0.5, seed=seed, index=sig.index, columns=sig.columns),
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [False, True, False],
                    [False, False, False],
                    [False, False, True],
                    [True, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random(
                (5, 3), prob=[0., 0.5, 1.], seed=seed, index=sig.index, columns=sig.columns),
            pd.DataFrame(
                np.array([
                    [False, True, True],
                    [False, True, True],
                    [False, False, True],
                    [False, False, True],
                    [False, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        with pytest.raises(Exception) as e_info:
            pd.DataFrame.vbt.signals.generate_random((5, 3))

    def test_generate_random_exits(self):
        pd.testing.assert_series_equal(
            sig['a'].vbt.signals.generate_random_exits(seed=seed),
            pd.Series(
                np.array([False, False, True, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_random_exits(seed=seed),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [True, True, False],
                    [False, False, False],
                    [True, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_random_exits(seed=seed, wait=0),
            pd.DataFrame(
                np.array([
                    [True, False, False],
                    [False, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, True, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_series_equal(
            sig['a'].vbt.signals.generate_random_exits(prob=1., seed=seed),
            pd.Series(
                np.array([False, True, False, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_random_exits(prob=1., seed=seed),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_random_exits(prob=[0., 0.5, 1.], seed=seed),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, True, True],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_random_exits(prob=1., wait=0, seed=seed),
            pd.DataFrame(
                np.array([
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, False, False],
                    [False, True, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

    def test_generate_random_both(self):
        # n
        en, ex = pd.Series.vbt.signals.generate_random_both(
            5, n=2, seed=seed, index=sig['a'].index, name=sig['a'].name)
        pd.testing.assert_series_equal(
            en,
            pd.Series(
                np.array([True, False, True, False, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            ex,
            pd.Series(
                np.array([False, True, False, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3), n=2, seed=seed, index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [False, False, False],
                    [True, True, False],
                    [False, False, True],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, True, True],
                    [False, False, False],
                    [False, True, False],
                    [True, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3), n=[0, 1, 2], seed=seed, index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [False, False, True],
                    [False, True, False],
                    [False, False, False],
                    [False, False, True],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, True],
                    [False, False, False],
                    [False, True, False],
                    [False, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both((2, 3), n=2, seed=seed, entry_wait=1, exit_wait=0)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [True, True, True],
                ])
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [True, True, True]
                ])
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both((3, 3), n=2, seed=seed, entry_wait=0, exit_wait=1)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [True, True, True],
                    [False, False, False]
                ])
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ])
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both((7, 3), n=2, seed=seed, entry_wait=2, exit_wait=2)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, True, True],
                    [False, False, False],
                    [False, False, False]
                ])
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [True, True, True]
                ])
            )
        )
        n = 10
        a = np.full(n * 2, 0.)
        for i in range(10000):
            en, ex = pd.Series.vbt.signals.generate_random_both(1000, n, entry_wait=2, exit_wait=2)
            _a = np.empty((n * 2,), dtype=np.int_)
            _a[0::2] = np.flatnonzero(en)
            _a[1::2] = np.flatnonzero(ex)
            a += _a
        greater = a > 10000000 / (2 * n + 1) * np.arange(0, 2 * n)
        less = a < 10000000 / (2 * n + 1) * np.arange(2, 2 * n + 2)
        assert np.all(greater & less)

        # probs
        en, ex = pd.Series.vbt.signals.generate_random_both(
            5, entry_prob=0.5, exit_prob=1., seed=seed, index=sig['a'].index, name=sig['a'].name)
        pd.testing.assert_series_equal(
            en,
            pd.Series(
                np.array([True, False, False, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            ex,
            pd.Series(
                np.array([False, True, False, False, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3), entry_prob=0.5, exit_prob=1., seed=seed, index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
                    [False, False, True],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, True, True],
                    [False, False, False],
                    [False, False, False],
                    [False, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3), entry_prob=[0., 0.5, 1.], exit_prob=[0., 0.5, 1.],
            seed=seed, index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [False, True, True],
                    [False, False, False],
                    [False, False, True],
                    [False, False, False],
                    [False, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, True, True],
                    [False, False, False],
                    [False, False, True],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        en, ex = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3), entry_prob=1., exit_prob=1., exit_wait=0,
            seed=seed, index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(
                np.array([
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True],
                    [True, True, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        # none
        with pytest.raises(Exception) as e_info:
            pd.DataFrame.vbt.signals.generate_random((5, 3))

    def test_generate_stop_exits(self):
        e = pd.Series([True, False, False, False, False, False])
        t = pd.Series([2, 3, 4, 3, 2, 1]).astype(np.float64)

        # stop loss
        pd.testing.assert_series_equal(
            e.vbt.signals.generate_stop_exits(t, -0.1),
            pd.Series(np.array([False, False, False, False, False, True]))
        )
        pd.testing.assert_series_equal(
            e.vbt.signals.generate_stop_exits(t, -0.1, trailing=True),
            pd.Series(np.array([False, False, False, True, False, False]))
        )
        pd.testing.assert_series_equal(
            e.vbt.signals.generate_stop_exits(t, -0.1, trailing=True, first=False),
            pd.Series(np.array([False, False, False, True, True, True]))
        )
        pd.testing.assert_frame_equal(
            e.vbt.signals.generate_stop_exits(t.vbt.tile(3), [-0., -0.5, -1.], trailing=True, first=False),
            pd.DataFrame(np.array([
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, True, False],
                [False, True, False]
            ]))
        )
        pd.testing.assert_series_equal(
            e.vbt.signals.generate_stop_exits(t, -0.1, trailing=True, exit_wait=3),
            pd.Series(np.array([False, False, False, False, True, False]))
        )
        # take profit
        pd.testing.assert_series_equal(
            e.vbt.signals.generate_stop_exits(4 - t, 0.1),
            pd.Series(np.array([False, False, False, False, False, True]))
        )
        pd.testing.assert_series_equal(
            e.vbt.signals.generate_stop_exits(4 - t, 0.1, trailing=True),
            pd.Series(np.array([False, False, False, True, False, False]))
        )
        pd.testing.assert_series_equal(
            e.vbt.signals.generate_stop_exits(4 - t, 0.1, trailing=True, first=False),
            pd.Series(np.array([False, False, False, True, True, True]))
        )
        pd.testing.assert_frame_equal(
            e.vbt.signals.generate_stop_exits((4 - t).vbt.tile(3), [0., 0.5, 1.], trailing=True, first=False),
            pd.DataFrame(np.array([
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, True, True],
                [False, True, True],
                [False, True, True]
            ]))
        )
        pd.testing.assert_series_equal(
            e.vbt.signals.generate_stop_exits(4 - t, 0.1, trailing=True, exit_wait=3),
            pd.Series(np.array([False, False, False, False, True, False]))
        )
        # iteratively
        e = pd.Series([True, True, True, True, True, True])
        en, ex = e.vbt.signals.generate_stop_exits(t, -0.1, trailing=True, iteratively=True)
        pd.testing.assert_series_equal(
            en,
            pd.Series(np.array([True, False, False, False, True, False]))
        )
        pd.testing.assert_series_equal(
            ex,
            pd.Series(np.array([False, False, False, True, False, True]))
        )
        en, ex = e.vbt.signals.generate_stop_exits(t, -0.1, trailing=True, entry_wait=2, iteratively=True)
        pd.testing.assert_series_equal(
            en,
            pd.Series(np.array([True, False, False, False, False, True]))
        )
        pd.testing.assert_series_equal(
            ex,
            pd.Series(np.array([False, False, False, True, False, False]))
        )
        en, ex = e.vbt.signals.generate_stop_exits(t, -0.1, trailing=True, exit_wait=2, iteratively=True)
        pd.testing.assert_series_equal(
            en,
            pd.Series(np.array([True, False, False, False, True, False]))
        )
        pd.testing.assert_series_equal(
            ex,
            pd.Series(np.array([False, False, False, True, False, False]))
        )

    def test_generate_ohlc_stop_exits(self):
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_exits(ts, -0.1),
            sig.vbt.signals.generate_ohlc_stop_exits(ts, sl_stop=0.1)
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_exits(ts, -0.1, trailing=True),
            sig.vbt.signals.generate_ohlc_stop_exits(ts, ts_stop=0.1)
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_exits(ts, 0.1),
            sig.vbt.signals.generate_ohlc_stop_exits(ts, tp_stop=0.1)
        )

        def _test_ohlc_stop_exits(**kwargs):
            out_dict = {'hit_price': np.nan, 'stop_type': -1}
            result = sig.vbt.signals.generate_ohlc_stop_exits(
                price['open'], price['high'], price['low'], price['close'],
                out_dict=out_dict, **kwargs
            )
            if isinstance(result, tuple):
                _, ex = result
            else:
                ex = result
            return result, out_dict['hit_price'], out_dict['stop_type']

        ex, hit_price, stop_type = _test_ohlc_stop_exits()
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(np.array([
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1]
            ]), index=sig.index, columns=sig.columns)
        )
        ex, hit_price, stop_type = _test_ohlc_stop_exits(sl_stop=0.1)
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(np.array([
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [True, False, False]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 10.8],
                [9.9, np.nan, np.nan]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, 0],
                [0, -1, -1]
            ]), index=sig.index, columns=sig.columns)
        )
        ex, hit_price, stop_type = _test_ohlc_stop_exits(ts_stop=0.1)
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(np.array([
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, True, True],
                [True, False, False]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, 11.7, 10.8],
                [9.9, np.nan, np.nan]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, 1, 1],
                [1, -1, -1]
            ]), index=sig.index, columns=sig.columns)
        )
        ex, hit_price, stop_type = _test_ohlc_stop_exits(tp_stop=0.1)
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(np.array([
                [False, False, False],
                [True, False, False],
                [False, True, False],
                [False, False, False],
                [False, False, False]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan],
                [11.0, np.nan, np.nan],
                [np.nan, 12.1, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1],
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, -1],
                [-1, -1, -1]
            ]), index=sig.index, columns=sig.columns)
        )
        ex, hit_price, stop_type = _test_ohlc_stop_exits(sl_stop=0.1, ts_stop=0.1, tp_stop=0.1)
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(np.array([
                [False, False, False],
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, False, False]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan],
                [11.0, np.nan, np.nan],
                [np.nan, 12.1, np.nan],
                [np.nan, np.nan, 10.8],
                [9.9, np.nan, np.nan]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1],
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 0],
                [0, -1, -1]
            ]), index=sig.index, columns=sig.columns)
        )
        ex, hit_price, stop_type = _test_ohlc_stop_exits(
            sl_stop=[0., 0.1, 0.2], ts_stop=[0., 0.1, 0.2], tp_stop=[0., 0.1, 0.2])
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(np.array([
                [False, False, False],
                [False, False, False],
                [False, True, False],
                [False, False, False],
                [False, False, True]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, 12.1, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 9.6]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, 2, -1],
                [-1, -1, -1],
                [-1, -1, 0]
            ]), index=sig.index, columns=sig.columns)
        )
        ex, hit_price, stop_type = _test_ohlc_stop_exits(sl_stop=0.1, ts_stop=0.1, tp_stop=0.1, exit_wait=0)
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(np.array([
                [True, False, False],
                [False, False, False],
                [False, True, False],
                [False, False, True],
                [True, True, False]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            hit_price,
            pd.DataFrame(np.array([
                [9.0, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, 12.1, np.nan],
                [np.nan, np.nan, 10.8],
                [9.9, 9.0, np.nan]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            stop_type,
            pd.DataFrame(np.array([
                [0, -1, -1],
                [-1, -1, -1],
                [-1, 2, -1],
                [-1, -1, 0],
                [0, 0, -1]
            ]), index=sig.index, columns=sig.columns)
        )
        (en, ex), hit_price, stop_type = _test_ohlc_stop_exits(sl_stop=0.1, ts_stop=0.1, tp_stop=0.1, iteratively=True)
        pd.testing.assert_frame_equal(
            en,
            pd.DataFrame(np.array([
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, False, False],
                [False, True, False]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            ex,
            pd.DataFrame(np.array([
                [False, False, False],
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, False, False]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan],
                [11.0, np.nan, np.nan],
                [np.nan, 12.1, np.nan],
                [np.nan, np.nan, 10.8],
                [9.9, np.nan, np.nan]
            ]), index=sig.index, columns=sig.columns)
        )
        pd.testing.assert_frame_equal(
            stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1],
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 0],
                [0, -1, -1]
            ]), index=sig.index, columns=sig.columns)
        )

    def test_map_reduce_between(self):
        @njit
        def distance_map_nb(from_i, to_i, col):
            return to_i - from_i

        @njit
        def mean_reduce_nb(col, a):
            return np.nanmean(a)

        other_sig = pd.DataFrame([
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [True, False, True]
        ], index=sig.index, columns=sig.columns)

        assert sig['a'].vbt.signals.map_reduce_between(
            map_func_nb=distance_map_nb,
            reduce_func_nb=mean_reduce_nb
        ) == 3.0
        pd.testing.assert_series_equal(
            sig.vbt.signals.map_reduce_between(
                map_func_nb=distance_map_nb,
                reduce_func_nb=mean_reduce_nb
            ),
            pd.Series([3., 3., np.nan], index=sig.columns).rename('map_reduce_between')
        )
        assert sig['a'].vbt.signals.map_reduce_between(
            other=other_sig['b'],
            map_func_nb=distance_map_nb,
            reduce_func_nb=mean_reduce_nb
        ) == 3.0
        pd.testing.assert_series_equal(
            sig.vbt.signals.map_reduce_between(
                other=other_sig,
                map_func_nb=distance_map_nb,
                reduce_func_nb=mean_reduce_nb
            ),
            pd.Series([1.5, 2., 2.], index=sig.columns).rename('map_reduce_between')
        )

    def test_map_reduce_partitions(self):
        @njit
        def distance_map_nb(from_i, to_i, col):
            return to_i - from_i

        @njit
        def mean_reduce_nb(col, a):
            return np.nanmean(a)

        assert (~sig['a']).vbt.signals.map_reduce_partitions(
            map_func_nb=distance_map_nb,
            reduce_func_nb=mean_reduce_nb
        ) == 1.5
        pd.testing.assert_series_equal(
            (~sig).vbt.signals.map_reduce_partitions(
                map_func_nb=distance_map_nb,
                reduce_func_nb=mean_reduce_nb
            ),
            pd.Series([1.5, 1.5, 2.], index=sig.columns).rename('map_reduce_partitions')
        )

    def test_num_signals(self):
        assert sig['a'].vbt.signals.num_signals() == 2
        pd.testing.assert_series_equal(
            sig.vbt.signals.num_signals(),
            pd.Series([2, 2, 1], index=sig.columns).rename('num_signals')
        )

    def test_avg_distance(self):
        assert sig['a'].vbt.signals.avg_distance() == 3.
        pd.testing.assert_series_equal(
            sig.vbt.signals.avg_distance(),
            pd.Series([3., 3., np.nan], index=sig.columns).rename('avg_distance')
        )
        other_sig = pd.DataFrame([
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [True, False, True]
        ], index=sig.index, columns=sig.columns)
        assert sig['a'].vbt.signals.avg_distance(to=other_sig['a']) == 1.5
        pd.testing.assert_series_equal(
            sig.vbt.signals.avg_distance(to=other_sig),
            pd.Series([1.5, 2., 2.], index=sig.columns).rename('avg_distance')
        )

    def test_rank(self):
        pd.testing.assert_series_equal(
            (~sig['a']).vbt.signals.rank(),
            pd.Series([0, 1, 2, 0, 1], index=sig['a'].index, name=sig['a'].name)
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank(),
            pd.DataFrame(
                np.array([
                    [0, 1, 1],
                    [1, 0, 2],
                    [2, 1, 0],
                    [0, 2, 1],
                    [1, 0, 2]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank(after_false=True),
            pd.DataFrame(
                np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [2, 1, 0],
                    [0, 2, 1],
                    [1, 0, 2]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank(allow_gaps=True),
            pd.DataFrame(
                np.array([
                    [0, 1, 1],
                    [1, 0, 2],
                    [2, 2, 0],
                    [0, 3, 3],
                    [3, 0, 4]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank(reset_by=sig['a'], allow_gaps=True),
            pd.DataFrame(
                np.array([
                    [0, 1, 1],
                    [1, 0, 2],
                    [2, 2, 0],
                    [0, 1, 1],
                    [1, 0, 2]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank(reset_by=sig, allow_gaps=True),
            pd.DataFrame(
                np.array([
                    [0, 1, 1],
                    [1, 0, 2],
                    [2, 1, 0],
                    [0, 2, 1],
                    [1, 0, 2]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

    def test_rank_partitions(self):
        pd.testing.assert_series_equal(
            (~sig['a']).vbt.signals.rank_partitions(),
            pd.Series([0, 1, 1, 0, 2], index=sig['a'].index, name=sig['a'].name)
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank_partitions(),
            pd.DataFrame(
                np.array([
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 2, 0],
                    [0, 2, 2],
                    [2, 0, 2]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank_partitions(after_false=True),
            pd.DataFrame(
                np.array([
                    [0, 0, 0],
                    [1, 0, 0],
                    [1, 1, 0],
                    [0, 1, 1],
                    [2, 0, 1]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank_partitions(reset_by=sig['a']),
            pd.DataFrame(
                np.array([
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 2, 0],
                    [0, 1, 1],
                    [1, 0, 1]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.rank_partitions(reset_by=sig),
            pd.DataFrame(
                np.array([
                    [0, 1, 1],
                    [1, 0, 1],
                    [1, 1, 0],
                    [0, 1, 1],
                    [1, 0, 1]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

    def test_rank_funs(self):
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.first(),
            pd.DataFrame(
                np.array([
                    [False, True, True],
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.nst(2),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, True],
                    [True, False, False],
                    [False, True, False],
                    [False, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            (~sig).vbt.signals.from_nst(1),
            pd.DataFrame(
                np.array([
                    [False, True, True],
                    [True, False, True],
                    [True, True, False],
                    [False, True, True],
                    [True, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

    @pytest.mark.parametrize(
        "test_func,test_func_pd",
        [
            (lambda x, *args, **kwargs: x.AND(args, **kwargs), lambda x, y: x & y),
            (lambda x, *args, **kwargs: x.OR(args, **kwargs), lambda x, y: x | y),
            (lambda x, *args, **kwargs: x.XOR(args, **kwargs), lambda x, y: x ^ y)
        ],
    )
    def test_logical_funcs(self, test_func, test_func_pd):
        pd.testing.assert_series_equal(
            test_func(sig['a'].vbt.signals, True, [True, False, False, False, False]),
            test_func_pd(test_func_pd(sig['a'], True), [True, False, False, False, False])
        )
        pd.testing.assert_frame_equal(
            test_func(sig['a'].vbt.signals, True, [True, False, False, False, False], concat=True),
            pd.concat((
                test_func_pd(sig['a'], True),
                test_func_pd(sig['a'], [True, False, False, False, False])
            ), axis=1, keys=[0, 1], names=['combine_idx'])
        )
        pd.testing.assert_frame_equal(
            test_func(sig.vbt.signals, True, [[True], [False], [False], [False], [False]]),
            test_func_pd(test_func_pd(sig, True), np.broadcast_to([[True], [False], [False], [False], [False]], (5, 3)))
        )
        pd.testing.assert_frame_equal(
            test_func(sig.vbt.signals, True, [[True], [False], [False], [False], [False]], concat=True),
            pd.concat((
                test_func_pd(sig, True),
                test_func_pd(sig, np.broadcast_to([[True], [False], [False], [False], [False]], (5, 3)))
            ), axis=1, keys=[0, 1], names=['combine_idx'])
        )


# ############# factory.py ############# #


class TestFactory:
    def test_both(self):
        @njit
        def cache_nb(ts1, ts2, in_out1, in_out2, n1, n2, arg0, temp_idx_arr0, kw0):
            return arg0

        @njit
        def choice_nb(from_i, to_i, col, ts, in_out, n, arg, temp_idx_arr, kw, cache):
            in_out[from_i, col] = ts[from_i, col] * n + arg + kw + cache
            temp_idx_arr[0] = from_i
            return temp_idx_arr[:1]

        MySignals = vbt.SignalFactory(
            input_names=['ts1', 'ts2'],
            in_output_names=['in_out1', 'in_out2'],
            param_names=['n1', 'n2']
        ).from_choice_func(
            cache_func=cache_nb,
            cache_settings=dict(
                pass_inputs=['ts1', 'ts2'],
                pass_in_outputs=['in_out1', 'in_out2'],
                pass_params=['n1', 'n2'],
                pass_kwargs=['temp_idx_arr0', ('kw0', 1000)]
            ),
            entry_choice_func=choice_nb,
            entry_settings=dict(
                pass_inputs=['ts1'],
                pass_in_outputs=['in_out1'],
                pass_params=['n1'],
                pass_kwargs=['temp_idx_arr1', ('kw1', 1000)],
                pass_cache=True
            ),
            exit_choice_func=choice_nb,
            exit_settings=dict(
                pass_inputs=['ts2'],
                pass_in_outputs=['in_out2'],
                pass_params=['n2'],
                pass_kwargs=['temp_idx_arr2', ('kw2', 1000)],
                pass_cache=True
            ),
            in_output_settings=dict(
                in_out1=dict(
                    dtype=np.float_
                ),
                in_out2=dict(
                    dtype=np.float_
                )
            ),
            in_out1=np.nan,
            in_out2=np.nan,
            var_args=True
        )
        my_sig = MySignals.run(
            np.arange(5), np.arange(5), [0, 1], [1, 0],
            cache_args=(0,), entry_args=(100,), exit_args=(100,)
        )
        pd.testing.assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(np.array([
                [True, True],
                [False, False],
                [True, True],
                [False, False],
                [True, True]
            ]), columns=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)],
                names=['custom_n1', 'custom_n2'])
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(np.array([
                [False, False],
                [True, True],
                [False, False],
                [True, True],
                [False, False],
            ]), columns=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)],
                names=['custom_n1', 'custom_n2'])
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.in_out1,
            pd.DataFrame(np.array([
                [1100.0, 1100.0],
                [np.nan, np.nan],
                [1100.0, 1102.0],
                [np.nan, np.nan],
                [1100.0, 1104.0]
            ]), columns=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)],
                names=['custom_n1', 'custom_n2'])
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(np.array([
                [np.nan, np.nan],
                [1101.0, 1100.0],
                [np.nan, np.nan],
                [1103.0, 1100.0],
                [np.nan, np.nan],
            ]), columns=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)],
                names=['custom_n1', 'custom_n2'])
            )
        )
        my_sig = MySignals.run(
            np.arange(7), np.arange(7), [0, 1], [1, 0],
            cache_args=(0,), entry_args=(100,), exit_args=(100,),
            entry_kwargs=dict(wait=2), exit_kwargs=dict(wait=2)
        )
        pd.testing.assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(np.array([
                [True, True],
                [False, False],
                [False, False],
                [False, False],
                [True, True],
                [False, False],
                [False, False]
            ]), columns=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)],
                names=['custom_n1', 'custom_n2'])
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(np.array([
                [False, False],
                [False, False],
                [True, True],
                [False, False],
                [False, False],
                [False, False],
                [True, True]
            ]), columns=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)],
                names=['custom_n1', 'custom_n2'])
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.in_out1,
            pd.DataFrame(np.array([
                [1100.0, 1100.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1100.0, 1104.0],
                [np.nan, np.nan],
                [np.nan, np.nan]
            ]), columns=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)],
                names=['custom_n1', 'custom_n2'])
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(np.array([
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1102.0, 1100.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1106.0, 1100.0]
            ]), columns=pd.MultiIndex.from_tuples(
                [(0, 1), (1, 0)],
                names=['custom_n1', 'custom_n2'])
            )
        )

    def test_exit_only(self):
        @njit
        def choice_nb(from_i, to_i, col, ts, in_out, n, arg, temp_idx_arr, kw):
            in_out[from_i, col] = ts[from_i, col] * n + arg + kw
            temp_idx_arr[0] = from_i
            return temp_idx_arr[:1]

        MySignals = vbt.SignalFactory(
            input_names=['ts2'],
            in_output_names=['in_out2'],
            param_names=['n2'],
            exit_only=True
        ).from_choice_func(
            exit_choice_func=choice_nb,
            exit_settings=dict(
                pass_inputs=['ts2'],
                pass_in_outputs=['in_out2'],
                pass_params=['n2'],
                pass_kwargs=['temp_idx_arr2', ('kw2', 1000)],
                pass_cache=True
            ),
            in_output_settings=dict(
                in_out2=dict(
                    dtype=np.float_
                )
            ),
            in_out2=np.nan,
            var_args=True
        )
        e = np.array([True, False, True, False, True])
        my_sig = MySignals.run(e, np.arange(5), [1, 0], 100)
        pd.testing.assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(np.array([
                [True, True],
                [False, False],
                [True, True],
                [False, False],
                [True, True]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(np.array([
                [False, False],
                [True, True],
                [False, False],
                [True, True],
                [False, False]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(np.array([
                [np.nan, np.nan],
                [1101.0, 1100.0],
                [np.nan, np.nan],
                [1103.0, 1100.0],
                [np.nan, np.nan],
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        e = np.array([True, False, False, True, False, False])
        my_sig = MySignals.run(e, np.arange(6), [1, 0], 100, wait=2)
        pd.testing.assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(np.array([
                [True, True],
                [False, False],
                [False, False],
                [True, True],
                [False, False],
                [False, False]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(np.array([
                [False, False],
                [False, False],
                [True, True],
                [False, False],
                [False, False],
                [True, True]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(np.array([
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1102.0, 1100.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1105.0, 1100.0]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )

    def test_iteratively(self):
        @njit
        def choice_nb(from_i, to_i, col, ts, in_out, n, arg, temp_idx_arr, kw):
            in_out[from_i, col] = ts[from_i, col] * n + arg + kw
            temp_idx_arr[0] = from_i
            return temp_idx_arr[:1]

        MySignals = vbt.SignalFactory(
            input_names=['ts2'],
            in_output_names=['in_out2'],
            param_names=['n2'],
            iteratively=True
        ).from_choice_func(
            exit_choice_func=choice_nb,
            exit_settings=dict(
                pass_inputs=['ts2'],
                pass_in_outputs=['in_out2'],
                pass_params=['n2'],
                pass_kwargs=['temp_idx_arr2', ('kw2', 1000)],
                pass_cache=True
            ),
            in_output_settings=dict(
                in_out2=dict(
                    dtype=np.float_
                )
            ),
            in_out2=np.nan,
            var_args=True
        )
        e = np.array([True, True, True, True, True])
        my_sig = MySignals.run(e, np.arange(5), [1, 0], 100)
        pd.testing.assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(np.array([
                [True, True],
                [True, True],
                [True, True],
                [True, True],
                [True, True]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.new_entries,
            pd.DataFrame(np.array([
                [True, True],
                [False, False],
                [True, True],
                [False, False],
                [True, True]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(np.array([
                [False, False],
                [True, True],
                [False, False],
                [True, True],
                [False, False]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(np.array([
                [np.nan, np.nan],
                [1101.0, 1100.0],
                [np.nan, np.nan],
                [1103.0, 1100.0],
                [np.nan, np.nan],
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        e = np.array([True, True, True, True, True, True])
        my_sig = MySignals.run(e, np.arange(6), [1, 0], 100, wait=2)
        pd.testing.assert_frame_equal(
            my_sig.entries,
            pd.DataFrame(np.array([
                [True, True],
                [True, True],
                [True, True],
                [True, True],
                [True, True],
                [True, True]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.new_entries,
            pd.DataFrame(np.array([
                [True, True],
                [False, False],
                [False, False],
                [True, True],
                [False, False],
                [False, False]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.exits,
            pd.DataFrame(np.array([
                [False, False],
                [False, False],
                [True, True],
                [False, False],
                [False, False],
                [True, True]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )
        pd.testing.assert_frame_equal(
            my_sig.in_out2,
            pd.DataFrame(np.array([
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1102.0, 1100.0],
                [np.nan, np.nan],
                [np.nan, np.nan],
                [1105.0, 1100.0]
            ]), columns=pd.Int64Index([1, 0], dtype='int64', name='custom_n2')
            )
        )


# ############# generators.py ############# #

class TestGenerators:
    def test_RAND(self):
        rand = vbt.RAND.run(n=1, input_shape=(6,), seed=seed)
        pd.testing.assert_series_equal(
            rand.entries,
            pd.Series(np.array([True, False, False, False, False, False]), name=1)
        )
        pd.testing.assert_series_equal(
            rand.exits,
            pd.Series(np.array([False, True, False, False, False, False]), name=1)
        )
        rand = vbt.RAND.run(n=[1, 2, 3], input_shape=(6,), seed=seed)
        pd.testing.assert_frame_equal(
            rand.entries,
            pd.DataFrame(np.array([
                [True, True, True],
                [False, False, False],
                [False, True, True],
                [False, False, False],
                [False, False, True],
                [False, False, False]
            ]), columns=pd.Int64Index([1, 2, 3], dtype='int64', name='rand_n')
            )
        )
        pd.testing.assert_frame_equal(
            rand.exits,
            pd.DataFrame(np.array([
                [False, False, False],
                [True, True, True],
                [False, False, False],
                [False, True, True],
                [False, False, False],
                [False, False, True]
            ]), columns=pd.Int64Index([1, 2, 3], dtype='int64', name='rand_n')
            )
        )
        rand = vbt.RAND.run(n=[np.array([1, 2]), np.array([3, 4])], input_shape=(8, 2), seed=seed)
        pd.testing.assert_frame_equal(
            rand.entries,
            pd.DataFrame(np.array([
                [False, True, True, True],
                [True, False, False, False],
                [False, False, False, True],
                [False, False, True, False],
                [False, True, False, True],
                [False, False, True, False],
                [False, False, False, True],
                [False, False, False, False]
            ]), columns=pd.MultiIndex.from_tuples([
                (1, 0),
                (2, 1),
                (3, 0),
                (4, 1)
            ], names=['rand_n', None])
            )
        )
        pd.testing.assert_frame_equal(
            rand.exits,
            pd.DataFrame(np.array([
                [False, False, False, False],
                [False, False, True, True],
                [False, False, False, False],
                [False, True, False, True],
                [False, False, True, False],
                [True, False, False, True],
                [False, False, True, False],
                [False, True, False, True]
            ]), columns=pd.MultiIndex.from_tuples([
                (1, 0),
                (2, 1),
                (3, 0),
                (4, 1)
            ], names=['rand_n', None])
            )
        )

    def test_RPROB(self):
        rprob = vbt.RPROB.run(entry_prob=1., exit_prob=1., input_shape=(5,), seed=seed)
        pd.testing.assert_series_equal(
            rprob.entries,
            pd.Series(np.array([True, False, True, False, True]), name=(1.0, 1.0))
        )
        pd.testing.assert_series_equal(
            rprob.exits,
            pd.Series(np.array([False, True, False, True, False]), name=(1.0, 1.0))
        )
        rprob = vbt.RPROB.run(
            entry_prob=np.asarray([1., 0., 1., 0., 1.]),
            exit_prob=np.asarray([0., 1., 0., 1., 0.]),
            input_shape=(5,), seed=seed)
        pd.testing.assert_series_equal(
            rprob.entries,
            pd.Series(np.array([True, False, True, False, True]), name=('array_0', 'array_0'))
        )
        pd.testing.assert_series_equal(
            rprob.exits,
            pd.Series(np.array([False, True, False, True, False]), name=('array_0', 'array_0'))
        )
        rprob = vbt.RPROB.run(entry_prob=[0.5, 1.], exit_prob=[1., 0.5], input_shape=(5,), seed=seed)
        pd.testing.assert_frame_equal(
            rprob.entries,
            pd.DataFrame(np.array([
                [True, True],
                [False, False],
                [False, True],
                [False, False],
                [True, False]
            ]), columns=pd.MultiIndex.from_tuples(
                [(0.5, 1.0), (1.0, 0.5)],
                names=['rprob_entry_prob', 'rprob_exit_prob'])
            )
        )
        pd.testing.assert_frame_equal(
            rprob.exits,
            pd.DataFrame(np.array([
                [False, False],
                [True, True],
                [False, False],
                [False, False],
                [False, False]
            ]), columns=pd.MultiIndex.from_tuples(
                [(0.5, 1.0), (1.0, 0.5)],
                names=['rprob_entry_prob', 'rprob_exit_prob'])
            )
        )

    def test_RPROBEX(self):
        rprobex = vbt.RPROBEX.run(sig, prob=[0., 0.5, 1.], seed=seed)
        pd.testing.assert_frame_equal(
            rprobex.exits,
            pd.DataFrame(np.array([
                [False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, True, False, False],
                [False, False, False, False, True, False, False, True, False],
                [False, False, False, False, False, False, False, False, True],
                [False, False, False, False, False, False, True, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.0, 'a'),
                (0.0, 'b'),
                (0.0, 'c'),
                (0.5, 'a'),
                (0.5, 'b'),
                (0.5, 'c'),
                (1.0, 'a'),
                (1.0, 'b'),
                (1.0, 'c')
            ], names=['rprobex_prob', None])
            )
        )

    def test_IRPROBEX(self):
        irprobex = vbt.IRPROBEX.run(sig, prob=[0., 0.5, 1.], seed=seed)
        pd.testing.assert_frame_equal(
            irprobex.new_entries,
            pd.DataFrame(np.array([
                [True, False, False, True, False, False, True, False, False],
                [False, True, False, False, True, False, False, True, False],
                [False, False, True, False, False, True, False, False, True],
                [False, False, False, True, False, False, True, False, False],
                [False, False, False, False, True, False, False, True, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.0, 'a'),
                (0.0, 'b'),
                (0.0, 'c'),
                (0.5, 'a'),
                (0.5, 'b'),
                (0.5, 'c'),
                (1.0, 'a'),
                (1.0, 'b'),
                (1.0, 'c')
            ], names=['irprobex_prob', None])
            )
        )
        pd.testing.assert_frame_equal(
            irprobex.exits,
            pd.DataFrame(np.array([
                [False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, True, False, False],
                [False, False, False, True, False, False, False, True, False],
                [False, False, False, False, True, True, False, False, True],
                [False, False, False, False, False, False, True, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.0, 'a'),
                (0.0, 'b'),
                (0.0, 'c'),
                (0.5, 'a'),
                (0.5, 'b'),
                (0.5, 'c'),
                (1.0, 'a'),
                (1.0, 'b'),
                (1.0, 'c')
            ], names=['irprobex_prob', None])
            )
        )

    def test_STEX(self):
        stex = vbt.STEX.run(sig, ts, 0.1)
        pd.testing.assert_frame_equal(
            stex.exits,
            pd.DataFrame(np.array([
                [False, False, False],
                [True, False, False],
                [False, True, False],
                [False, False, False],
                [False, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 'a'),
                (0.1, 'b'),
                (0.1, 'c')
            ], names=['stex_stop', None])
            )
        )
        stex = vbt.STEX.run(sig, ts, np.asarray([0.1, 0.1, -0.1, -0.1, -0.1])[:, None])
        pd.testing.assert_frame_equal(
            stex.exits,
            pd.DataFrame(np.array([
                [False, False, False],
                [True, False, False],
                [False, True, False],
                [False, False, True],
                [True, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                ('array_0', 'a'),
                ('array_0', 'b'),
                ('array_0', 'c')
            ], names=['stex_stop', None])
            )
        )
        stex = vbt.STEX.run(sig, ts, [0.1, 0.1, -0.1, -0.1], trailing=[False, True, False, True])
        pd.testing.assert_frame_equal(
            stex.exits,
            pd.DataFrame(np.array([
                [False, False, False, False, False, False, False, False, False, False, False, False],
                [True, False, False, True, False, False, False, False, False, False, False, False],
                [False, True, False, False, True, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, True, False, True, True],
                [False, False, False, False, False, False, True, False, False, True, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, False, 'a'),
                (0.1, False, 'b'),
                (0.1, False, 'c'),
                (0.1, True, 'a'),
                (0.1, True, 'b'),
                (0.1, True, 'c'),
                (-0.1, False, 'a'),
                (-0.1, False, 'b'),
                (-0.1, False, 'c'),
                (-0.1, True, 'a'),
                (-0.1, True, 'b'),
                (-0.1, True, 'c')
            ], names=['stex_stop', 'stex_trailing', None])
            )
        )

    def test_ISTEX(self):
        istex = vbt.ISTEX.run(sig, ts, [0.1, 0.1, -0.1, -0.1], trailing=[False, True, False, True])
        pd.testing.assert_frame_equal(
            istex.new_entries,
            pd.DataFrame(np.array([
                [True, False, False, True, False, False, True, False, False, True, False, False],
                [False, True, False, False, True, False, False, True, False, False, True, False],
                [False, False, True, False, False, True, False, False, True, False, False, True],
                [True, False, False, True, False, False, False, False, False, False, False, False],
                [False, True, False, False, True, False, False, False, False, False, True, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, False, 'a'),
                (0.1, False, 'b'),
                (0.1, False, 'c'),
                (0.1, True, 'a'),
                (0.1, True, 'b'),
                (0.1, True, 'c'),
                (-0.1, False, 'a'),
                (-0.1, False, 'b'),
                (-0.1, False, 'c'),
                (-0.1, True, 'a'),
                (-0.1, True, 'b'),
                (-0.1, True, 'c')
            ], names=['istex_stop', 'istex_trailing', None])
            )
        )
        pd.testing.assert_frame_equal(
            istex.exits,
            pd.DataFrame(np.array([
                [False, False, False, False, False, False, False, False, False, False, False, False],
                [True, False, False, True, False, False, False, False, False, False, False, False],
                [False, True, False, False, True, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False, False, True, True, True, True],
                [False, False, False, False, False, False, False, True, False, False, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, False, 'a'),
                (0.1, False, 'b'),
                (0.1, False, 'c'),
                (0.1, True, 'a'),
                (0.1, True, 'b'),
                (0.1, True, 'c'),
                (-0.1, False, 'a'),
                (-0.1, False, 'b'),
                (-0.1, False, 'c'),
                (-0.1, True, 'a'),
                (-0.1, True, 'b'),
                (-0.1, True, 'c')
            ], names=['istex_stop', 'istex_trailing', None])
            )
        )

    def test_OHLCSTEX(self):
        ohlcstex = vbt.OHLCSTEX.run(
            sig, price['open'], price['high'], price['low'], price['close'],
            sl_stop=0.1
        )
        pd.testing.assert_frame_equal(
            ohlcstex.exits,
            pd.DataFrame(np.array([
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, True],
                [True, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 'a'),
                (0.1, 'b'),
                (0.1, 'c')
            ], names=['ohlcstex_sl_stop', None])
            )
        )
        pd.testing.assert_frame_equal(
            ohlcstex.hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan],
                [np.nan, np.nan, 10.8],
                [9.9, np.nan, np.nan]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 'a'),
                (0.1, 'b'),
                (0.1, 'c')
            ], names=['ohlcstex_sl_stop', None])
            )
        )
        pd.testing.assert_frame_equal(
            ohlcstex.stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, -1],
                [-1, -1, 0],
                [0, -1, -1]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 'a'),
                (0.1, 'b'),
                (0.1, 'c')
            ], names=['ohlcstex_sl_stop', None])
            )
        )
        ohlcstex = vbt.OHLCSTEX.run(
            sig, price['open'], price['high'], price['low'], price['close'],
            sl_stop=[0.1, 0., 0.], ts_stop=[0., 0.1, 0.], tp_stop=[0., 0., 0.1]
        )
        pd.testing.assert_frame_equal(
            ohlcstex.exits,
            pd.DataFrame(np.array([
                [False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, True, False, False],
                [False, False, False, False, False, False, False, True, False],
                [False, False, True, False, True, True, False, False, False],
                [True, False, False, True, False, False, False, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 0., 0., 'a'),
                (0.1, 0., 0., 'b'),
                (0.1, 0., 0., 'c'),
                (0., 0.1, 0., 'a'),
                (0., 0.1, 0., 'b'),
                (0., 0.1, 0., 'c'),
                (0., 0., 0.1, 'a'),
                (0., 0., 0.1, 'b'),
                (0., 0., 0.1, 'c')
            ], names=['ohlcstex_sl_stop', 'ohlcstex_ts_stop', 'ohlcstex_tp_stop', None])
            )
        )
        pd.testing.assert_frame_equal(
            ohlcstex.hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 11., np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 12.1, np.nan],
                [np.nan, np.nan, 10.8, np.nan, 11.7, 10.8, np.nan, np.nan, np.nan],
                [9.9, np.nan, np.nan, 9.9, np.nan, np.nan, np.nan, np.nan, np.nan]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 0., 0., 'a'),
                (0.1, 0., 0., 'b'),
                (0.1, 0., 0., 'c'),
                (0., 0.1, 0., 'a'),
                (0., 0.1, 0., 'b'),
                (0., 0.1, 0., 'c'),
                (0., 0., 0.1, 'a'),
                (0., 0., 0.1, 'b'),
                (0., 0., 0.1, 'c')
            ], names=['ohlcstex_sl_stop', 'ohlcstex_ts_stop', 'ohlcstex_tp_stop', None])
            )
        )
        pd.testing.assert_frame_equal(
            ohlcstex.stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 2, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, 2, -1],
                [-1, -1, 0, -1, 1, 1, -1, -1, -1],
                [0, -1, -1, 1, -1, -1, -1, -1, -1]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 0., 0., 'a'),
                (0.1, 0., 0., 'b'),
                (0.1, 0., 0., 'c'),
                (0., 0.1, 0., 'a'),
                (0., 0.1, 0., 'b'),
                (0., 0.1, 0., 'c'),
                (0., 0., 0.1, 'a'),
                (0., 0., 0.1, 'b'),
                (0., 0., 0.1, 'c')
            ], names=['ohlcstex_sl_stop', 'ohlcstex_ts_stop', 'ohlcstex_tp_stop', None])
            )
        )

    def test_IOHLCSTEX(self):
        iohlcstex = vbt.IOHLCSTEX.run(
            sig, price['open'], price['high'], price['low'], price['close'],
            sl_stop=[0.1, 0., 0.], ts_stop=[0., 0.1, 0.], tp_stop=[0., 0., 0.1]
        )
        pd.testing.assert_frame_equal(
            iohlcstex.exits,
            pd.DataFrame(np.array([
                [False, False, False, False, False, False, False, False, False],
                [False, False, False, False, False, False, True, False, False],
                [False, False, False, False, False, False, False, True, False],
                [False, False, True, True, True, True, False, False, False],
                [True, True, False, False, False, False, False, False, False]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 0., 0., 'a'),
                (0.1, 0., 0., 'b'),
                (0.1, 0., 0., 'c'),
                (0., 0.1, 0., 'a'),
                (0., 0.1, 0., 'b'),
                (0., 0.1, 0., 'c'),
                (0., 0., 0.1, 'a'),
                (0., 0., 0.1, 'b'),
                (0., 0., 0.1, 'c')
            ], names=['iohlcstex_sl_stop', 'iohlcstex_ts_stop', 'iohlcstex_tp_stop', None])
            )
        )
        pd.testing.assert_frame_equal(
            iohlcstex.hit_price,
            pd.DataFrame(np.array([
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 11., np.nan, np.nan],
                [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 12.1, np.nan],
                [np.nan, np.nan, 10.8, 11.7, 11.7, 10.8, np.nan, np.nan, np.nan],
                [9., 9.9, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 0., 0., 'a'),
                (0.1, 0., 0., 'b'),
                (0.1, 0., 0., 'c'),
                (0., 0.1, 0., 'a'),
                (0., 0.1, 0., 'b'),
                (0., 0.1, 0., 'c'),
                (0., 0., 0.1, 'a'),
                (0., 0., 0.1, 'b'),
                (0., 0., 0.1, 'c')
            ], names=['iohlcstex_sl_stop', 'iohlcstex_ts_stop', 'iohlcstex_tp_stop', None])
            )
        )
        pd.testing.assert_frame_equal(
            iohlcstex.stop_type,
            pd.DataFrame(np.array([
                [-1, -1, -1, -1, -1, -1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1, 2, -1, -1],
                [-1, -1, -1, -1, -1, -1, -1, 2, -1],
                [-1, -1, 0, 1, 1, 1, -1, -1, -1],
                [0, 0, -1, -1, -1, -1, -1, -1, -1]
            ]), index=sig.index, columns=pd.MultiIndex.from_tuples([
                (0.1, 0., 0., 'a'),
                (0.1, 0., 0., 'b'),
                (0.1, 0., 0., 'c'),
                (0., 0.1, 0., 'a'),
                (0., 0.1, 0., 'b'),
                (0., 0.1, 0., 'c'),
                (0., 0., 0.1, 'a'),
                (0., 0., 0.1, 'b'),
                (0., 0., 0.1, 'c')
            ], names=['iohlcstex_sl_stop', 'iohlcstex_ts_stop', 'iohlcstex_tp_stop', None])
            )
        )

