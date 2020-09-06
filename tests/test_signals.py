import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
import pytest

from vectorbt.signals import nb

seed = 42

day_dt = np.timedelta64(86400000000000)

index = pd.Index([
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    datetime(2020, 1, 3),
    datetime(2020, 1, 4),
    datetime(2020, 1, 5)
])
columns = ['a', 'b', 'c']
sig = pd.DataFrame([
    [True, False, False],
    [False, True, False],
    [False, False, True],
    [True, False, False],
    [False, True, False]
], index=index, columns=columns)

# ############# accessors.py ############# #


class TestAccessors:
    def test_freq(self):
        assert sig.vbt.signals.freq == day_dt
        assert sig['a'].vbt.signals.freq == day_dt
        assert sig.vbt.signals(freq='2D').freq == day_dt * 2
        assert sig['a'].vbt.signals(freq='2D').freq == day_dt * 2
        assert pd.Series([False, True]).vbt.signals.freq is None
        assert pd.Series([False, True]).vbt.signals(freq='3D').freq == day_dt * 3
        assert pd.Series([False, True]).vbt.signals(freq=np.timedelta64(4, 'D')).freq == day_dt * 4

    def test_shuffle(self):
        pd.testing.assert_series_equal(
            sig['a'].vbt.signals.shuffle(seed=seed),
            pd.Series(
                np.array([False, False, False, True, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        np.testing.assert_array_equal(
            sig['a'].vbt.signals.shuffle(seed=seed).values,
            nb.shuffle_1d_nb(sig['a'].values, seed=seed)
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.shuffle(seed=seed),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, True, False],
                    [False, False, False],
                    [True, False, False],
                    [True, True, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

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
        def choice_func_nb(col, from_i, to_i, n):
            if col == 0:
                return np.arange(from_i, to_i)
            elif col == 1:
                return np.full(1, from_i)
            else:
                return np.full(1, to_i-n)

        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate(5, choice_func_nb, 1, index=sig['a'].index, name=sig['a'].name),
            pd.Series(
                np.array([True, True, True, True, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate((5,), choice_func_nb, 1, index=sig['a'].index, name=sig['a'].name),
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

    def test_generate_exits(self):
        @njit
        def choice_func_nb(col, from_i, to_i, n):
            if col == 0:
                return np.arange(from_i, to_i)
            elif col == 1:
                return np.full(1, from_i)
            else:
                return np.full(1, to_i - n)

        pd.testing.assert_series_equal(
            sig['a'].vbt.signals.generate_exits(choice_func_nb, 1),
            pd.Series(
                np.array([False, True, True, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_exits(choice_func_nb, 1),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [True, True, False],
                    [False, False, False],
                    [True, False, True]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

    def test_generate_both(self):
        @njit
        def entry_choice_func_nb(col, from_i, to_i, wait1):
            next_pos = col + from_i + wait1
            if next_pos < to_i:
                return np.array([next_pos])
            return np.empty(0, dtype=np.int_)

        @njit
        def exit_choice_func_nb(col, from_i, to_i, wait2):
            next_pos = col + from_i + wait2
            if next_pos < to_i:
                return np.array([next_pos])
            return np.empty(0, dtype=np.int_)

        a, b = pd.Series.vbt.signals.generate_both(
            5, entry_choice_func_nb, exit_choice_func_nb, (0,), (1,), index=sig['a'].index, name=sig['a'].name)
        pd.testing.assert_series_equal(
            a,
            pd.Series(
                np.array([True, False, False, True, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            b,
            pd.Series(
                np.array([False, False, True, False, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        a, b = pd.Series.vbt.signals.generate_both(
            (5,), entry_choice_func_nb, exit_choice_func_nb, (0,), (1,), index=sig['a'].index, name=sig['a'].name)
        pd.testing.assert_series_equal(
            a,
            pd.Series(
                np.array([True, False, False, True, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            b,
            pd.Series(
                np.array([False, False, True, False, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        a, b = pd.DataFrame.vbt.signals.generate_both(
            (5, 3), entry_choice_func_nb, exit_choice_func_nb, (0,), (1,), index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            a,
            pd.DataFrame(
                np.array([
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, False, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            b,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, True, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )

    def test_generate_random(self):
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate_random(5, n=3, seed=seed, index=sig['a'].index, name=sig['a'].name),
            pd.Series(
                np.array([False, True, True, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate_random((5,), n=3, seed=seed, index=sig['a'].index, name=sig['a'].name),
            pd.Series(
                np.array([False, True, True, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        with pytest.raises(Exception) as e_info:
            _ = pd.Series.vbt.signals.generate_random((5, 2), n=3)
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.signals.generate_random((5, 3), n=3, seed=seed, index=sig.index, columns=sig.columns),
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
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate_random(
                5, prob=0.5, seed=seed, index=sig['a'].index, name=sig['a'].name),
            pd.Series(
                np.array([True, False, False, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            pd.Series.vbt.signals.generate_random(
                (5,), prob=0.5, seed=seed, index=sig['a'].index, name=sig['a'].name),
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
                np.array([False, False,  True, False,  True]),
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
        pd.testing.assert_series_equal(
            sig['a'].vbt.signals.generate_random_exits(prob=0.5, seed=seed),
            pd.Series(
                np.array([False, True, False, False, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_random_exits(prob=0.5, seed=seed),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [False, False, False],
                    [False, False, True],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        
    def test_generate_random_both(self):
        a, b = pd.Series.vbt.signals.generate_random_both(
            5, n=1, seed=seed, index=sig['a'].index, name=sig['a'].name)
        pd.testing.assert_series_equal(
            a,
            pd.Series(
                np.array([False, True, False, False, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            b,
            pd.Series(
                np.array([False, False, False, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        a, b = pd.Series.vbt.signals.generate_random_both(
            (5,), n=1, seed=seed, index=sig['a'].index, name=sig['a'].name)
        pd.testing.assert_series_equal(
            a,
            pd.Series(
                np.array([False, True, False, False, False]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        pd.testing.assert_series_equal(
            b,
            pd.Series(
                np.array([False, False, False, False, True]),
                index=sig['a'].index,
                name=sig['a'].name
            )
        )
        a, b = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3), n=1, seed=seed, index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            a,
            pd.DataFrame(
                np.array([
                    [False, False, True],
                    [True, True, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        pd.testing.assert_frame_equal(
            b,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, True],
                    [False, False, False],
                    [False, True, False],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=sig.columns
            )
        )
        a, b = pd.DataFrame.vbt.signals.generate_random_both(
            (5, 3), entry_prob=0.5, exit_prob=1., seed=seed, index=sig.index, columns=sig.columns)
        pd.testing.assert_frame_equal(
            a,
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
            b,
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
        with pytest.raises(Exception) as e_info:
            pd.DataFrame.vbt.signals.generate_random((5, 3))

    def test_generate_stop_loss_exits(self):
        ts = pd.Series([1, 2, 3, 2, 1], index=index, name=sig['a'].name)

        pd.testing.assert_series_equal(
            sig['a'].vbt.signals.generate_stop_loss_exits(ts, 0.1),
            pd.Series(
                np.array([False, False, False, False, True]),
                index=sig['a'].index,
                name=(0.1, sig['a'].name)
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_loss_exits(ts, 0.1),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, True],
                    [True, False, False]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c')
                ], names=['stop_loss', None])
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_loss_exits(ts, 0.1, first=False),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, True],
                    [True, False, True]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c')
                ], names=['stop_loss', None])
            )
        )
        target = pd.DataFrame(
            np.array([
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False],
                [False, False, True, False, False, True],
                [True, False, False, True, False, False]
            ]),
            index=sig.index,
            columns=pd.MultiIndex.from_tuples([
                (0.1, 'a'),
                (0.1, 'b'),
                (0.1, 'c'),
                (0.2, 'a'),
                (0.2, 'b'),
                (0.2, 'c')
            ], names=['stop_loss', None])
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_loss_exits(ts, [0.1, 0.2]),
            target
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_loss_exits(
                ts, np.concatenate((np.full((1, 5, 3), 0.1), np.full((1, 5, 3), 0.2)))),
            target
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_loss_exits(ts, [0.1, 0.2], keys=['sl1', 'sl2']),
            pd.DataFrame(
                target.values,
                index=target.index,
                columns=pd.MultiIndex.from_tuples([
                    ('sl1', 'a'),
                    ('sl1', 'b'),
                    ('sl1', 'c'),
                    ('sl2', 'a'),
                    ('sl2', 'b'),
                    ('sl2', 'c')
                ])
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_stop_loss_exits(ts, [0.1, 0.2], trailing=True),
            pd.DataFrame(
                np.array([
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, False, False, False, False, False],
                    [False, True, True, False, True, True],
                    [True, False, False, True, False, False]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c'),
                    (0.2, 'a'),
                    (0.2, 'b'),
                    (0.2, 'c')
                ], names=['trail_stop', None])
            )
        )
        new_entries, new_exits = sig.vbt.signals.generate_stop_loss_exits(ts, 0.1, iteratively=True)
        pd.testing.assert_frame_equal(
            new_entries,
            pd.DataFrame(
                np.array([
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [False, False, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c')
                ], names=['stop_loss', None])
            )
        )
        pd.testing.assert_frame_equal(
            new_exits,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [False, False, False],
                    [False, False, False],
                    [False, False, True],
                    [False, True, False]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c')
                ], names=['stop_loss', None])
            )
        )

    def test_generate_take_profit_exits(self):
        ts = pd.Series([1, 2, 3, 2, 1], index=index, name=sig['a'].name)

        pd.testing.assert_series_equal(
            sig['a'].vbt.signals.generate_take_profit_exits(ts, 0.1),
            pd.Series(
                np.array([False, True, False, False, False]),
                index=sig['a'].index,
                name=(0.1, sig['a'].name)
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_take_profit_exits(ts, 0.1),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [False, True, False],
                    [False, False, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c')
                ], names=['take_profit', None])
            )
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_take_profit_exits(ts, 0.1, first=False),
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [True, True, False],
                    [False, False, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c')
                ], names=['take_profit', None])
            )
        )
        target = pd.DataFrame(
            np.array([
                [False, False, False, False, False, False],
                [True, False, False, True, False, False],
                [False, True, False, False, True, False],
                [False, False, False, False, False, False],
                [False, False, False, False, False, False]
            ]),
            index=sig.index,
            columns=pd.MultiIndex.from_tuples([
                (0.1, 'a'),
                (0.1, 'b'),
                (0.1, 'c'),
                (0.2, 'a'),
                (0.2, 'b'),
                (0.2, 'c')
            ], names=['take_profit', None])
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_take_profit_exits(ts, [0.1, 0.2]),
            target
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_take_profit_exits(
                ts, np.concatenate((np.full((1, 5, 3), 0.1), np.full((1, 5, 3), 0.2)))),
            target
        )
        pd.testing.assert_frame_equal(
            sig.vbt.signals.generate_take_profit_exits(ts, [0.1, 0.2], keys=['tp1', 'tp2']),
            pd.DataFrame(
                target.values,
                index=target.index,
                columns=pd.MultiIndex.from_tuples([
                    ('tp1', 'a'),
                    ('tp1', 'b'),
                    ('tp1', 'c'),
                    ('tp2', 'a'),
                    ('tp2', 'b'),
                    ('tp2', 'c')
                ])
            )
        )
        new_entries, new_exits = sig.vbt.signals.generate_take_profit_exits(ts, 0.1, iteratively=True)
        pd.testing.assert_frame_equal(
            new_entries,
            pd.DataFrame(
                np.array([
                    [True, False, False],
                    [False, True, False],
                    [False, False, True],
                    [True, False, False],
                    [False, True, False]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c')
                ], names=['take_profit', None])
            )
        )
        pd.testing.assert_frame_equal(
            new_exits,
            pd.DataFrame(
                np.array([
                    [False, False, False],
                    [True, False, False],
                    [False, True, False],
                    [False, False, False],
                    [False, False, False]
                ]),
                index=sig.index,
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a'),
                    (0.1, 'b'),
                    (0.1, 'c')
                ], names=['take_profit', None])
            )
        )

    def test_map_reduce_between(self):
        @njit
        def distance_map_nb(col, from_i, to_i):
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
        ], index=index, columns=columns)

        assert sig['a'].vbt.signals.map_reduce_between(
            map_func_nb=distance_map_nb,
            reduce_func_nb=mean_reduce_nb
        ) == 3.0
        pd.testing.assert_series_equal(
            sig.vbt.signals.map_reduce_between(
                map_func_nb=distance_map_nb,
                reduce_func_nb=mean_reduce_nb
            ),
            pd.Series([3., 3., np.nan], index=sig.columns)
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
            pd.Series([1.5, 2., 2.], index=sig.columns)
        )

    def test_map_reduce_partitions(self):
        @njit
        def distance_map_nb(col, from_i, to_i):
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
            pd.Series([1.5, 1.5, 2.], index=sig.columns)
        )

    def test_num_signals(self):
        assert sig['a'].vbt.signals.num_signals() == 2
        pd.testing.assert_series_equal(
            sig.vbt.signals.num_signals(),
            pd.Series([2, 2, 1], index=sig.columns)
        )

    def test_avg_distance(self):
        assert sig['a'].vbt.signals.avg_distance() == 3.
        pd.testing.assert_series_equal(
            sig.vbt.signals.avg_distance(),
            pd.Series([3., 3., np.nan], index=sig.columns)
        )
        other_sig = pd.DataFrame([
            [False, False, False],
            [False, False, False],
            [True, False, False],
            [False, True, False],
            [True, False, True]
        ], index=index, columns=columns)
        assert sig['a'].vbt.signals.avg_distance(other=other_sig['a']) == 1.5
        pd.testing.assert_series_equal(
            sig.vbt.signals.avg_distance(other=other_sig),
            pd.Series([1.5, 2., 2.], index=sig.columns)
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
            (lambda x, *args, **kwargs: x.AND(*args, **kwargs), lambda x, y: x & y),
            (lambda x, *args, **kwargs: x.OR(*args, **kwargs), lambda x, y: x | y),
            (lambda x, *args, **kwargs: x.XOR(*args, **kwargs), lambda x, y: x ^ y)
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

