import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
import pytest

from tests.utils import seed

ts = pd.DataFrame({
    'a': [1, 2, 3, 4, np.nan],
    'b': [np.nan, 4, 3, 2, 1],
    'c': [1, 2, np.nan, 2, 1]
}, index=pd.DatetimeIndex([
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5)
]))

CustomInd = vbt.IndicatorFactory(
    ts_names=['ts1', 'ts2'],
    param_names=['p1', 'p2'],
    output_names=['o1', 'o2'],
    custom_outputs={
        'co1': lambda self: self.ts1 + self.ts2,
        'co2': property(lambda self: self.o1 + self.o2)
    }
).from_apply_func(lambda ts1, ts2, p1, p2: (ts1 * p1, ts2 * p2))

custom_ind = CustomInd.from_params(ts, ts * 2, [1, 2], [3, 4])

# ############# factory.py ############# #


class TestFactory:
    def test_from_custom_func(self):
        def apply_func(i, ts, p, a, b=10):
            return ts * p[i] + a + b

        @njit
        def apply_func_nb(i, ts, p, a, b):
            return ts * p[i] + a + b  # numba doesn't support **kwargs

        # Custom function can be anything that takes time series, params and other arguments, and returns outputs
        def custom_func(ts, p, *args, **kwargs):
            return vbt.base.combine_fns.apply_and_concat_one(len(p), apply_func, ts, p, *args, **kwargs)

        @njit
        def custom_func_nb(ts, p, *args):
            return vbt.base.combine_fns.apply_and_concat_one_nb(len(p), apply_func_nb, ts, p, *args)

        target1 = pd.DataFrame(
            np.array([
                [110., np.nan, 110., 111., np.nan, 111.],
                [110., 110., 110., 112., 114., 112.],
                [110., 110., np.nan, 113., 113., np.nan],
                [110., 110., 110., 114., 112., 112.],
                [np.nan, 110., 110., np.nan, 111., 111.]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 'a'),
                (0, 'b'),
                (0, 'c'),
                (1, 'a'),
                (1, 'b'),
                (1, 'c')
            ], names=['custom_param', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func).from_params(ts, [0, 1], 10, b=100).output,
            target1
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func_nb).from_params(ts, [0, 1], 10, 100).output,
            target1
        )
        target2 = pd.DataFrame(
            np.array([
                [110., 111.],
                [110., 112.],
                [110., 113.],
                [110., 114.],
                [np.nan, np.nan]
            ]),
            index=ts.index,
            columns=pd.Int64Index([0, 1], dtype='int64', name='custom_param')
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func).from_params(ts['a'], [0, 1], 10, b=100).output,
            target2
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func_nb).from_params(ts['a'], [0, 1], 10, 100).output,
            target2
        )
        target3 = pd.Series(
            np.array([110., 110., 110., 110., np.nan]),
            index=ts.index,
            name=(0, 'a')
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func).from_params(ts['a'], 0, 10, b=100).output,
            target3
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func_nb).from_params(ts['a'], 0, 10, 100).output,
            target3
        )

    def test_from_apply_func(self):
        # Apply function is performed on each parameter `p` individually, and each output is then stacked for you
        # Apply functions are less customizable than custom functions, but are simpler to write
        # Notice there is no longer an `i` argument
        def apply_func(ts, p, a, b=10):
            return ts * p + a + b

        @njit
        def apply_func_nb(ts, p, a, b):
            return ts * p + a + b  # numba doesn't support **kwargs

        target1 = pd.DataFrame(
            np.array([
                [110., np.nan, 110., 111., np.nan, 111.],
                [110., 110., 110., 112., 114., 112.],
                [110., 110., np.nan, 113., 113., np.nan],
                [110., 110., 110., 114., 112., 112.],
                [np.nan, 110., 110., np.nan, 111., 111.]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 'a'),
                (0, 'b'),
                (0, 'c'),
                (1, 'a'),
                (1, 'b'),
                (1, 'c')
            ], names=['custom_param', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func).from_params(ts, [0, 1], 10, b=100).output,
            target1
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func_nb).from_params(ts, [0, 1], 10, 100).output,
            target1
        )
        target2 = pd.DataFrame(
            np.array([
                [110., 111.],
                [110., 112.],
                [110., 113.],
                [110., 114.],
                [np.nan, np.nan]
            ]),
            index=ts.index,
            columns=pd.Int64Index([0, 1], dtype='int64', name='custom_param')
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func).from_params(ts['a'], [0, 1], 10, b=100).output,
            target2
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func_nb).from_params(ts['a'], [0, 1], 10, 100).output,
            target2
        )
        target3 = pd.Series(
            np.array([110., 110., 110., 110., np.nan]),
            index=ts.index,
            name=(0, 'a')
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func).from_params(ts['a'], 0, 10, b=100).output,
            target3
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func_nb).from_params(ts['a'], 0, 10, 100).output,
            target3
        )

    def test_multiple_ts(self):
        target = pd.DataFrame(
            np.array([
                [0., np.nan, 0., 2., np.nan, 2.],
                [0., 0., 0., 6., 20., 6.],
                [0., 0., np.nan, 12., 12., np.nan],
                [0., 0., 0., 20., 6., 6.],
                [np.nan, 0., 0., np.nan, 2., 2.]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 'a'),
                (0, 'b'),
                (0, 'c'),
                (1, 'a'),
                (1, 'b'),
                (1, 'c')
            ], names=['custom_param', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(ts_names=['ts1', 'ts2']) \
                .from_apply_func(lambda ts1, ts2, p: ts1 * ts2 * p) \
                .from_params(ts, ts + 1, [0, 1]).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(ts_names=['ts1', 'ts2']) \
                .from_apply_func(njit(lambda ts1, ts2, p: ts1 * ts2 * p)) \
                .from_params(ts, ts + 1, [0, 1]).output,
            target
        )

    def test_multiple_params(self):
        target = pd.DataFrame(
            np.array([
                [2., np.nan, 2., 4., np.nan, 4.],
                [4., 8., 4., 8., 16., 8.],
                [6., 6., np.nan, 12., 12., np.nan],
                [8., 4., 4., 16., 8., 8.],
                [np.nan, 2., 2., np.nan, 4., 4.]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 2, 'a'),
                (0, 2, 'b'),
                (0, 2, 'c'),
                (1, 3, 'a'),
                (1, 3, 'b'),
                (1, 3, 'c')
            ], names=['custom_p1', 'custom_p2', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(lambda ts, p1, p2: ts * (p1 + p2)) \
                .from_params(ts, [0, 1], [2, 3]).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(njit(lambda ts, p1, p2: ts * (p1 + p2))) \
                .from_params(ts, [0, 1], [2, 3]).output,
            target
        )

    def test_param_product(self):
        target = pd.DataFrame(
            np.array([
                [2., np.nan, 2., 3., np.nan, 3., 3., np.nan, 3., 4., np.nan, 4.],
                [4., 8., 4., 6., 12., 6., 6., 12., 6., 8., 16., 8.],
                [6., 6., np.nan, 9., 9., np.nan, 9., 9., np.nan, 12., 12., np.nan],
                [8., 4., 4., 12., 6., 6., 12., 6., 6., 16., 8., 8.],
                [np.nan, 2., 2., np.nan, 3., 3., np.nan, 3., 3., np.nan, 4., 4.]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 2, 'a'),
                (0, 2, 'b'),
                (0, 2, 'c'),
                (0, 3, 'a'),
                (0, 3, 'b'),
                (0, 3, 'c'),
                (1, 2, 'a'),
                (1, 2, 'b'),
                (1, 2, 'c'),
                (1, 3, 'a'),
                (1, 3, 'b'),
                (1, 3, 'c')
            ], names=['custom_p1', 'custom_p2', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(lambda ts, p1, p2: ts * (p1 + p2)) \
                .from_params(ts, [0, 1], [2, 3], param_product=True).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(njit(lambda ts, p1, p2: ts * (p1 + p2))) \
                .from_params(ts, [0, 1], [2, 3], param_product=True).output,
            target
        )

    def test_multiple_outputs(self):
        target1 = pd.DataFrame(
            np.array([
                [0., np.nan, 0., 1., np.nan, 1.],
                [0., 0., 0., 2., 4., 2.],
                [0., 0., np.nan, 3., 3., np.nan],
                [0., 0., 0., 4., 2., 2.],
                [np.nan, 0., 0., np.nan, 1., 1.]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 'a'),
                (0, 'b'),
                (0, 'c'),
                (1, 'a'),
                (1, 'b'),
                (1, 'c')
            ], names=['custom_param', None])
        )
        target2 = pd.DataFrame(
            np.array([
                [0., np.nan, 0., 1., np.nan, 1.],
                [0., 0., 0., 2., 4., 2.],
                [0., 0., np.nan, 3., 3., np.nan],
                [0., 0., 0., 4., 2., 2.],
                [np.nan, 0., 0., np.nan, 1., 1.]
            ]),
            index=target1.index,
            columns=target1.columns
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(output_names=['o1', 'o2']) \
                .from_apply_func(lambda ts, p: (ts * p, ts * p ** 2)) \
                .from_params(ts, [0, 1]).o1,
            target1
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(output_names=['o1', 'o2']) \
                .from_apply_func(njit(lambda ts, p: (ts * p, ts * p ** 2))) \
                .from_params(ts, [0, 1]).o1,
            target1
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(output_names=['o1', 'o2']) \
                .from_apply_func(lambda ts, p: (ts * p, ts * p ** 2)) \
                .from_params(ts, [0, 1]).o2,
            target2
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(output_names=['o1', 'o2']) \
                .from_apply_func(njit(lambda ts, p: (ts * p, ts * p ** 2))) \
                .from_params(ts, [0, 1]).o2,
            target2
        )

    def test_cache(self):
        def caching_func(ts, params):
            np.random.seed(seed)
            return np.random.uniform(0, 1)

        target = pd.DataFrame(
            np.array([
                [0.37454012, np.nan, 0.37454012, 1.37454012, np.nan, 1.37454012],
                [0.37454012, 0.37454012, 0.37454012, 2.37454012, 4.37454012, 2.37454012],
                [0.37454012, 0.37454012, np.nan, 3.37454012, 3.37454012, np.nan],
                [0.37454012, 0.37454012, 0.37454012, 4.37454012, 2.37454012, 2.37454012],
                [np.nan, 0.37454012, 0.37454012, np.nan, 1.37454012, 1.37454012]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 'a'),
                (0, 'b'),
                (0, 'c'),
                (1, 'a'),
                (1, 'b'),
                (1, 'c')
            ], names=['custom_param', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(
                lambda ts, param, c: ts * param + c,
                caching_func=caching_func
            ).from_params(ts, [0, 1]).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(
                njit(lambda ts, param, c: ts * param + c),
                caching_func=njit(caching_func)
            ).from_params(ts, [0, 1]).output,
            target
        )
        # return_cache
        cache = vbt.IndicatorFactory().from_apply_func(
            lambda ts, param, c: ts * param + c,
            caching_func=caching_func,
            return_cache=True
        ).from_params(ts, [0, 1])
        assert cache == 0.3745401188473625
        cache = vbt.IndicatorFactory().from_apply_func(
            njit(lambda ts, param, c: ts * param + c),
            caching_func=njit(caching_func),
            return_cache=True
        ).from_params(ts, [0, 1])
        assert cache == 0.3745401188473625
        # pass cache
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(
                lambda ts, param, c: ts * param + c,
                cache=cache
            ).from_params(ts, [0, 1]).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(
                njit(lambda ts, param, c: ts * param + c),
                cache=cache
            ).from_params(ts, [0, 1]).output,
            target
        )

    def test_return_raw(self):
        target = np.array([
            [110., np.nan, 110., 111., np.nan, 111.],
            [110., 110., 110., 112., 114., 112.],
            [110., 110., np.nan, 113., 113., np.nan],
            [110., 110., 110., 114., 112., 112.],
            [np.nan, 110., 110., np.nan, 111., 111.]
        ])
        np.testing.assert_array_equal(
            vbt.IndicatorFactory().from_apply_func(
                lambda ts, p, a, b=10: ts * p + a + b,
                return_raw=True
            ).from_params(ts, [0, 1], 10, b=100),
            target
        )
        np.testing.assert_array_equal(
            vbt.IndicatorFactory().from_apply_func(
                njit(lambda ts, p, a, b: ts * p + a + b),
                return_raw=True
            ).from_params(ts, [0, 1], 10, 100),
            target
        )

    def test_no_params(self):
        target = pd.DataFrame(
            np.array([
                [111., np.nan, 111.],
                [112., 114., 112.],
                [113., 113., np.nan],
                [114., 112., 112.],
                [np.nan, 111., 111.]
            ]),
            index=ts.index,
            columns=ts.columns
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(
                lambda ts, a, b=10: ts + a + b
            ).from_params(ts, 10, b=100).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(
                njit(lambda ts, a, b: ts + a + b)
            ).from_params(ts, 10, 100).output,
            target
        )

    def test_pass_1d(self):
        target = pd.Series(
            np.array([12., 14., 16., 18., np.nan]),
            index=ts['a'].index,
            name=ts['a'].name
        )
        try:
            pd.testing.assert_series_equal(
                vbt.IndicatorFactory(param_names=[]).from_custom_func(
                    lambda ts, a, b=10: ts + a + b
                ).from_params(ts['a'], 10, b=ts['a'].values).output,
                target
            )
            raise ValueError
        except:
            pass
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory(param_names=[]).from_custom_func(
                lambda ts, a, b=10: ts + a + b,
                pass_2d=False
            ).from_params(ts['a'], 10, b=ts['a'].values).output,
            target
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory(param_names=[]).from_custom_func(
                njit(lambda ts, a, b: ts + a + b),
                pass_2d=False
            ).from_params(ts['a'], 10, ts['a'].values).output,
            target
        )

    def test_pass_lists(self):
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(
                lambda ts_list, param_list: ts_list[0] * param_list[0],
                pass_lists=True
            ).from_params(ts, 2).output,
            ts * 2
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(
                lambda ts_list, param_list: njit(lambda x, y: x * y)(ts_list[0], param_list[0]),
                pass_lists=True
            ).from_params(ts, 2).output,
            ts * 2
        )

    def test_other(self):
        obj, other = vbt.IndicatorFactory(
            ts_names=['ts1', 'ts2'],
            param_names=['p1', 'p2'],
            output_names=['o1', 'o2']
        ).from_apply_func(
            lambda ts1, ts2, p1, p2: (ts1 * p1, ts2 * p2, ts1 * p1 + ts2 * p2)
        ).from_params(ts, ts + 1, [0, 1], [1, 2])
        np.testing.assert_array_equal(
            other,
            np.array([
                [2., np.nan, 2., 5., np.nan, 5.],
                [3., 5., 3., 8., 14., 8.],
                [4., 4., np.nan, 11., 11., np.nan],
                [5., 3., 3., 14., 8., 8.],
                [np.nan, 2., 2., np.nan, 5., 5.]
            ])
        )
        obj2, other2 = vbt.IndicatorFactory(
            ts_names=['ts1', 'ts2'],
            param_names=['p1', 'p2'],
            output_names=['o1', 'o2']
        ).from_apply_func(
            njit(lambda ts1, ts2, p1, p2: (ts1 * p1, ts2 * p2, ts1 * p1 + ts2 * p2))
        ).from_params(ts, ts + 1, [0, 1], [1, 2])
        np.testing.assert_array_equal(other, other2)

    def test_mappers(self):
        obj = vbt.IndicatorFactory(param_names=['p1', 'p2']) \
            .from_apply_func(lambda ts, p1, p2: ts * (p1 + p2)) \
            .from_params(ts, 0, 2)
        pd.testing.assert_series_equal(
            obj._p1_mapper,
            pd.Series([0, 0, 0], index=ts.columns, name='custom_p1')
        )
        pd.testing.assert_series_equal(
            obj._p2_mapper,
            pd.Series([2, 2, 2], index=ts.columns, name='custom_p2')
        )
        pd.testing.assert_series_equal(
            obj._tuple_mapper,
            pd.Series([(0, 2), (0, 2), (0, 2)], index=ts.columns, name=('custom_p1', 'custom_p2'))
        )
        obj = vbt.IndicatorFactory(param_names=['p1', 'p2']) \
            .from_apply_func(lambda ts, p1, p2: ts * (p1 + p2)) \
            .from_params(ts, [0, 1], [1, 2])
        pd.testing.assert_series_equal(
            obj._p1_mapper,
            pd.Series(
                np.array([0, 0, 0, 1, 1, 1]),
                index=pd.MultiIndex.from_tuples([
                    (0, 1, 'a'),
                    (0, 1, 'b'),
                    (0, 1, 'c'),
                    (1, 2, 'a'),
                    (1, 2, 'b'),
                    (1, 2, 'c')
                ], names=['custom_p1', 'custom_p2', None]),
                name='custom_p1'
            )
        )
        pd.testing.assert_series_equal(
            obj._p2_mapper,
            pd.Series(
                np.array([1, 1, 1, 2, 2, 2]),
                index=pd.MultiIndex.from_tuples([
                    (0, 1, 'a'),
                    (0, 1, 'b'),
                    (0, 1, 'c'),
                    (1, 2, 'a'),
                    (1, 2, 'b'),
                    (1, 2, 'c')
                ], names=['custom_p1', 'custom_p2', None]),
                name='custom_p2'
            )
        )
        pd.testing.assert_series_equal(
            obj._tuple_mapper,
            pd.Series(
                [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (1, 2)],
                index=pd.MultiIndex.from_tuples([
                    (0, 1, 'a'),
                    (0, 1, 'b'),
                    (0, 1, 'c'),
                    (1, 2, 'a'),
                    (1, 2, 'b'),
                    (1, 2, 'c')
                ], names=['custom_p1', 'custom_p2', None]),
                name=('custom_p1', 'custom_p2')
            )
        )

    def test_name(self):
        assert vbt.IndicatorFactory(name='my_ind') \
            .from_apply_func(lambda ts, p: ts * p) \
            .from_params(ts, [0, 1]).name == 'my_ind'
        assert vbt.IndicatorFactory() \
           .from_apply_func(lambda ts, p: ts * p) \
           .from_params(ts, [0, 1], name='my_ind').name == 'my_ind'

    def test_hide_params(self):
        assert vbt.IndicatorFactory(param_names=['p1', 'p2']) \
           .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
           .from_params(ts, [0, 1], [1, 2], hide_params=['p2']).output.columns.names == ['custom_p1', None]
        assert vbt.IndicatorFactory(param_names=['p1', 'p2']) \
           .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
           .from_params(ts, [0, 1], [1, 2], hide_params=['p1']).output.columns.names == ['custom_p2', None]

    def test_wrapper(self):
        pd.testing.assert_index_equal(
            custom_ind.wrapper.index,
            pd.DatetimeIndex([
                '2018-01-01',
                '2018-01-02',
                '2018-01-03',
                '2018-01-04',
                '2018-01-05'
            ], dtype='datetime64[ns]', freq=None)
        )
        pd.testing.assert_index_equal(
            custom_ind.wrapper.columns,
            pd.MultiIndex.from_tuples([
                (1, 3, 'a'),
                (1, 3, 'b'),
                (1, 3, 'c'),
                (2, 4, 'a'),
                (2, 4, 'b'),
                (2, 4, 'c')
            ], names=['custom_p1', 'custom_p2', None])
        )
        assert custom_ind.wrapper.ndim == 2
        assert custom_ind.wrapper.shape == (5, 6)
        assert custom_ind.wrapper.freq == pd.Timedelta('1 days 00:00:00')

    @pytest.mark.parametrize(
        "test_attr",
        ['ts1', 'ts2', 'o1', 'o2', 'co1', 'co2']
    )
    def test_pandas_indexing(self, test_attr):
        pd.testing.assert_frame_equal(
            getattr(custom_ind.iloc[np.arange(3), np.arange(3)], test_attr),
            getattr(custom_ind, test_attr).iloc[np.arange(3), np.arange(3)]
        )
        pd.testing.assert_series_equal(
            getattr(custom_ind.loc[:, (1, 3, 'a')], test_attr),
            getattr(custom_ind, test_attr).loc[:, (1, 3, 'a')]
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind.loc[:, (1, 3)], test_attr),
            getattr(custom_ind, test_attr).loc[:, (1, 3)]
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind[(1, 3)], test_attr),
            getattr(custom_ind, test_attr)[(1, 3)]
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind.xs(1, axis=1, level=0), test_attr),
            getattr(custom_ind, test_attr).xs(1, axis=1, level=0)
        )

    @pytest.mark.parametrize(
        "test_attr",
        ['ts1', 'ts2', 'o1', 'o2', 'co1', 'co2']
    )
    def test_param_indexing(self, test_attr):
        pd.testing.assert_frame_equal(
            getattr(custom_ind.p1_loc[2], test_attr),
            getattr(custom_ind, test_attr).xs(2, level='custom_p1', axis=1)
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind.p1_loc[1:2], test_attr),
            pd.concat((
                getattr(custom_ind, test_attr).xs(1, level='custom_p1', drop_level=False, axis=1),
                getattr(custom_ind, test_attr).xs(2, level='custom_p1', drop_level=False, axis=1)
            ), axis=1)
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind.p1_loc[[1, 1, 1]], test_attr),
            pd.concat((
                getattr(custom_ind, test_attr).xs(1, level='custom_p1', drop_level=False, axis=1),
                getattr(custom_ind, test_attr).xs(1, level='custom_p1', drop_level=False, axis=1),
                getattr(custom_ind, test_attr).xs(1, level='custom_p1', drop_level=False, axis=1)
            ), axis=1)
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind.tuple_loc[(1, 3)], test_attr),
            getattr(custom_ind, test_attr).xs((1, 3), level=('custom_p1', 'custom_p2'), axis=1)
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind.tuple_loc[(1, 3):(2, 4)], test_attr),
            pd.concat((
                getattr(custom_ind, test_attr).xs((1, 3), level=('custom_p1', 'custom_p2'), drop_level=False, axis=1),
                getattr(custom_ind, test_attr).xs((2, 4), level=('custom_p1', 'custom_p2'), drop_level=False, axis=1)
            ), axis=1)
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind.tuple_loc[[(1, 3), (1, 3), (1, 3)]], test_attr),
            pd.concat((
                getattr(custom_ind, test_attr).xs((1, 3), level=('custom_p1', 'custom_p2'), drop_level=False, axis=1),
                getattr(custom_ind, test_attr).xs((1, 3), level=('custom_p1', 'custom_p2'), drop_level=False, axis=1),
                getattr(custom_ind, test_attr).xs((1, 3), level=('custom_p1', 'custom_p2'), drop_level=False, axis=1)
            ), axis=1)
        )

    @pytest.mark.parametrize(
        "test_attr",
        ['ts1', 'ts2', 'o1', 'o2', 'co1', 'co2']
    )
    def test_comparison_methods(self, test_attr):
        pd.testing.assert_frame_equal(
            getattr(custom_ind, test_attr+'_above')(2),
            getattr(custom_ind, test_attr) > 2
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind, test_attr + '_below')(2),
            getattr(custom_ind, test_attr) < 2
        )
        pd.testing.assert_frame_equal(
            getattr(custom_ind, test_attr + '_equal')(2),
            getattr(custom_ind, test_attr) == 2
        )

    def test_above(self):
        pd.testing.assert_frame_equal(custom_ind.o1_above(2), custom_ind.o1 > 2)
        target = pd.DataFrame(
            np.array([
                [False, False, False, False, False, False, False, False, False, False, False, False],
                [False, True, False, True, True, True, False, True, False, True, True, True],
                [True, True, False, True, True, False, False, False, False, True, True, False],
                [True, False, False, True, True, True, True, False, False, True, True, True],
                [False, False, False, False, False, False, False, False, False, False, False, False]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (2, 1, 3, 'a'),
                (2, 1, 3, 'b'),
                (2, 1, 3, 'c'),
                (2, 2, 4, 'a'),
                (2, 2, 4, 'b'),
                (2, 2, 4, 'c'),
                (3, 1, 3, 'a'),
                (3, 1, 3, 'b'),
                (3, 1, 3, 'c'),
                (3, 2, 4, 'a'),
                (3, 2, 4, 'b'),
                (3, 2, 4, 'c')
            ], names=['custom_o1_above', 'custom_p1', 'custom_p2', None])
        )
        pd.testing.assert_frame_equal(
            custom_ind.o1_above([2, 3], multiple=True),
            target
        )
        columns2 = target.columns.rename('my_above', 0)
        pd.testing.assert_frame_equal(
            custom_ind.o1_above([2, 3], name='my_above', multiple=True),
            pd.DataFrame(
                target.values,
                index=target.index,
                columns=columns2
            )
        )
        pd.testing.assert_frame_equal(
            custom_ind.o1_above(2, crossed=True),
            pd.DataFrame(
                np.array([
                    [False, False, False, False, False, False],
                    [False, True, False, True, True, True],
                    [True, False, False, False, False, False],
                    [False, False, False, False, False, True],
                    [False, False, False, False, False, False]
                ]),
                index=ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (1, 3, 'a'),
                    (1, 3, 'b'),
                    (1, 3, 'c'),
                    (2, 4, 'a'),
                    (2, 4, 'b'),
                    (2, 4, 'c')
                ], names=['custom_p1', 'custom_p2', None])
            )
        )

    def test_attr_list(self):
        attr_list = [
            '_iloc',
            '_indexing_func',
            '_loc',
            '_name',
            '_o1',
            '_o2',
            '_p1_array',
            '_p1_loc',
            '_p1_mapper',
            '_p2_array',
            '_p2_loc',
            '_p2_mapper',
            '_ts1',
            '_ts2',
            '_tuple_loc',
            '_tuple_mapper',
            'from_params',
            'iloc',
            'loc',
            'name',
            'o1',
            'o1_above',
            'o1_below',
            'o1_equal',
            'o2',
            'o2_above',
            'o2_below',
            'o2_equal',
            'p1_loc',
            'p2_loc',
            'ts1',
            'ts1_above',
            'ts1_below',
            'ts1_equal',
            'ts2',
            'ts2_above',
            'ts2_below',
            'ts2_equal',
            'tuple_loc',
            'wrapper',
            'xs'
        ]
        for attr in attr_list:
            assert attr in dir(custom_ind)


# ############# basic.py ############# #

close_ts = pd.Series([1, 2, 3, 4, 3, 2, 1], index=pd.DatetimeIndex([
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5),
    datetime(2018, 1, 6),
    datetime(2018, 1, 7)
]))
high_ts = close_ts * 1.1
low_ts = close_ts * 0.9
volume_ts = pd.Series([4, 3, 2, 1, 2, 3, 4], index=close_ts.index)

class TestBasic:
    def test_MA(self):
        pd.testing.assert_frame_equal(
            vbt.MA.from_params(close_ts, window=(2, 3), ewm=(False, True), param_product=True).ma,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [1.5, 1.75, np.nan, np.nan],
                    [2.5, 2.61538462, 2., 2.42857143],
                    [3.5, 3.55, 3., 3.26666667],
                    [3.5, 3.18181818, 3.33333333, 3.12903226],
                    [2.5, 2.39285714, 3., 2.55555556],
                    [1.5, 1.46386093, 2., 1.77165354]
                ]),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (2, False),
                    (2, True),
                    (3, False),
                    (3, True)
                ], names=['ma_window', 'ma_ewm'])
            )
        )
        ma1, ma2 = vbt.MA.from_combs(
            close_ts,
            (2, 3, 4),
            2,
            ewm=[False, True],
            param_product=True,
            names=['test1', 'test2']
        )
        pd.testing.assert_frame_equal(
            ma1.ma,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan],
                    [1.5, 1.5, 1.5, 1.5, 1.5,
                     1.75, 1.75, 1.75, 1.75, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan],
                    [2.5, 2.5, 2.5, 2.5, 2.5,
                     2.61538462, 2.61538462, 2.61538462, 2.61538462, 2.,
                     2., 2., 2.42857143, 2.42857143, np.nan],
                    [3.5, 3.5, 3.5, 3.5, 3.5,
                     3.55, 3.55, 3.55, 3.55, 3.,
                     3., 3., 3.26666667, 3.26666667, 2.5],
                    [3.5, 3.5, 3.5, 3.5, 3.5,
                     3.18181818, 3.18181818, 3.18181818, 3.18181818, 3.33333333,
                     3.33333333, 3.33333333, 3.12903226, 3.12903226, 3.],
                    [2.5, 2.5, 2.5, 2.5, 2.5,
                     2.39285714, 2.39285714, 2.39285714, 2.39285714, 3.,
                     3., 3., 2.55555556, 2.55555556, 3.],
                    [1.5, 1.5, 1.5, 1.5, 1.5,
                     1.46386093, 1.46386093, 1.46386093, 1.46386093, 2.,
                     2., 2., 1.77165354, 1.77165354, 2.5]
                ]),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (2, False),
                    (2, False),
                    (2, False),
                    (2, False),
                    (2, False),
                    (2,  True),
                    (2,  True),
                    (2,  True),
                    (2,  True),
                    (3, False),
                    (3, False),
                    (3, False),
                    (3,  True),
                    (3,  True),
                    (4, False)
                ], names=['test1_window', 'test1_ewm'])
            )
        )
        pd.testing.assert_frame_equal(
            ma2.ma,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan],
                    [1.75, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan],
                    [2.61538462, 2., 2.42857143, np.nan, np.nan,
                     2., 2.42857143, np.nan, np.nan, 2.42857143,
                     np.nan, np.nan, np.nan, np.nan, np.nan],
                    [3.55, 3., 3.26666667, 2.5, 3.09558824,
                     3., 3.26666667, 2.5, 3.09558824, 3.26666667,
                     2.5, 3.09558824, 2.5, 3.09558824, 3.09558824],
                    [3.18181818, 3.33333333, 3.12903226, 3., 3.05412908,
                     3.33333333, 3.12903226, 3., 3.05412908, 3.12903226,
                     3., 3.05412908, 3., 3.05412908, 3.05412908],
                    [2.39285714, 3., 2.55555556, 3., 2.61184211,
                     3., 2.55555556, 3., 2.61184211, 2.55555556,
                     3., 2.61184211, 3., 2.61184211, 2.61184211],
                    [1.46386093, 2., 1.77165354, 2.5, 1.94853696,
                     2., 1.77165354, 2.5, 1.94853696, 1.77165354,
                     2.5, 1.94853696, 2.5, 1.94853696, 1.94853696]
                ]),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (2,  True),
                    (3, False),
                    (3,  True),
                    (4, False),
                    (4,  True),
                    (3, False),
                    (3,  True),
                    (4, False),
                    (4,  True),
                    (3,  True),
                    (4, False),
                    (4,  True),
                    (4, False),
                    (4,  True),
                    (4,  True)
                ], names=['test2_window', 'test2_ewm'])
            )
        )

    def test_MSTD(self):
        pd.testing.assert_frame_equal(
            vbt.MSTD.from_params(close_ts, window=(2, 3), ewm=(False, True), param_product=True).mstd,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [0.70710678, 0.70710678, np.nan, np.nan],
                    [0.70710678, 0.91986621, 1., 0.96362411],
                    [0.70710678, 1.05975324, 1., 1.17716366],
                    [0.70710678, 0.70710678, 0.57735027, 0.8210929],
                    [0.70710678, 0.88707098, 1., 0.90101496],
                    [0.70710678, 1.06029102, 1., 1.14630853]
                ]),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (2, False),
                    (2, True),
                    (3, False),
                    (3, True)
                ], names=['mstd_window', 'mstd_ewm'])
            )
        )

    def test_BollingerBands(self):
        columns = pd.MultiIndex.from_tuples([
            (2, False, 2.0),
            (2, False, 3.0),
            (2, True, 2.0),
            (2, True, 3.0),
            (3, False, 2.0),
            (3, False, 3.0),
            (3, True, 2.0),
            (3, True, 3.0)
        ], names=['bb_window', 'bb_ewm', 'bb_alpha'])
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.from_params(
                close_ts,
                window=(2, 3),
                alpha=(2, 3),
                ewm=(False, True),
                param_product=True
            ).middle,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [1.5, 1.5, 1.75, 1.75, np.nan,
                     np.nan, np.nan, np.nan],
                    [2.5, 2.5, 2.61538462, 2.61538462, 2.,
                     2., 2.42857143, 2.42857143],
                    [3.5, 3.5, 3.55, 3.55, 3.,
                     3., 3.26666667, 3.26666667],
                    [3.5, 3.5, 3.18181818, 3.18181818, 3.33333333,
                     3.33333333, 3.12903226, 3.12903226],
                    [2.5, 2.5, 2.39285714, 2.39285714, 3.,
                     3., 2.55555556, 2.55555556],
                    [1.5, 1.5, 1.46386093, 1.46386093, 2.,
                     2., 1.77165354, 1.77165354]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.from_params(
                close_ts,
                window=(2, 3),
                alpha=(2, 3),
                ewm=(False, True),
                param_product=True
            ).upper,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [2.91421356, 3.62132034, 3.16421356, 3.87132034, np.nan,
                     np.nan, np.nan, np.nan],
                    [3.91421356, 4.62132034, 4.45511704, 5.37498325, 4.,
                     5., 4.35581965, 5.31944376],
                    [4.91421356, 5.62132034, 5.66950647, 6.72925971, 5.,
                     6., 5.62099399, 6.79815765],
                    [4.91421356, 5.62132034, 4.59603174, 5.30313853, 4.48803387,
                     5.06538414, 4.77121806, 5.59231095],
                    [3.91421356, 4.62132034, 4.1669991, 5.05407008, 5.,
                     6., 4.35758547, 5.25860043],
                    [2.91421356, 3.62132034, 3.58444297, 4.64473399, 4.,
                     5., 4.0642706, 5.21057913]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.from_params(
                close_ts,
                window=(2, 3),
                alpha=(2, 3),
                ewm=(False, True),
                param_product=True
            ).lower,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [0.08578644, -0.62132034, 0.33578644, -0.37132034, np.nan,
                     np.nan, np.nan, np.nan],
                    [1.08578644, 0.37867966, 0.77565219, -0.14421402, 0.,
                     -1., 0.50132321, -0.46230091],
                    [2.08578644, 1.37867966, 1.43049353, 0.37074029, 1.,
                     0., 0.91233934, -0.26482432],
                    [2.08578644, 1.37867966, 1.76760462, 1.06049784, 2.17863279,
                     1.60128253, 1.48684646, 0.66575356],
                    [1.08578644, 0.37867966, 0.61871518, -0.2683558, 1.,
                     0., 0.75352564, -0.14748932],
                    [0.08578644, -0.62132034, -0.65672111, -1.71701212, 0.,
                     -1., -0.52096352, -1.66727205]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.from_params(
                close_ts,
                window=(2, 3),
                alpha=(2, 3),
                ewm=(False, True),
                param_product=True
            ).percent_b,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [0.6767767, 0.61785113, 0.58838835, 0.55892557, np.nan,
                     np.nan, np.nan, np.nan],
                    [0.6767767, 0.61785113, 0.60453025, 0.56968683, 0.75,
                     0.66666667, 0.64824986, 0.59883324],
                    [0.6767767, 0.61785113, 0.60615679, 0.57077119, 0.75,
                     0.66666667, 0.65574158, 0.60382772],
                    [0.3232233, 0.38214887, 0.43571757, 0.45714504, 0.35566243,
                     0.40377496, 0.46071326, 0.47380884],
                    [0.3232233, 0.38214887, 0.38928249, 0.42618833, 0.25,
                     0.33333333, 0.34585285, 0.39723523],
                    [0.3232233, 0.38214887, 0.39062886, 0.42708591, 0.25,
                     0.33333333, 0.33170902, 0.38780601]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.from_params(
                close_ts,
                window=(2, 3),
                alpha=(2, 3),
                ewm=(False, True),
                param_product=True
            ).bandwidth,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [1.88561808, 2.82842712, 1.61624407, 2.42436611, np.nan,
                     np.nan, np.nan, np.nan],
                    [1.13137085, 1.69705627, 1.40685421, 2.11028131, 2.,
                     3., 1.5871456, 2.38071839],
                    [0.80812204, 1.21218305, 1.19408815, 1.79113223, 1.33333333,
                     2., 1.44142489, 2.16213734],
                    [0.80812204, 1.21218305, 0.88893424, 1.33340136, 0.69282032,
                     1.03923048, 1.04964453, 1.5744668],
                    [1.13137085, 1.69705627, 1.48286492, 2.22429738, 1.33333333,
                     2., 1.41028428, 2.11542643],
                    [1.88561808, 2.82842712, 2.89724521, 4.34586782, 2.,
                     3., 2.58810993, 3.88216489]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )

    def test_RSI(self):
        pd.testing.assert_frame_equal(
            vbt.RSI.from_params(close_ts, window=(2, 3), ewm=(False, True), param_product=True).rsi,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [100., 100., np.nan, np.nan],
                    [100., 100., 100., 100.],
                    [50., 32.5, 66.66666667, 46.66666667],
                    [0., 10.74380165, 33.33333333, 22.58064516],
                    [0., 3.57142857, 0., 11.11111111]
                ]),
                index=close_ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (2, False),
                    (2, True),
                    (3, False),
                    (3, True)
                ], names=['rsi_window', 'rsi_ewm'])
            )
        )

    def test_Stochastic(self):
        columns = pd.MultiIndex.from_tuples([
            (2, 2, False),
            (2, 2, True),
            (2, 3, False),
            (2, 3, True),
            (3, 2, False),
            (3, 2, True),
            (3, 3, False),
            (3, 3, True)
        ], names=['stoch_k_window', 'stoch_d_window', 'stoch_d_ewm'])
        pd.testing.assert_frame_equal(
            vbt.Stochastic.from_params(
                close_ts,
                high_ts=high_ts,
                low_ts=low_ts,
                k_window=(2, 3),
                d_window=(2, 3),
                d_ewm=(False, True),
                param_product=True
            ).percent_k,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [84.61538462, 84.61538462, 84.61538462, 84.61538462, np.nan,
                     np.nan, np.nan, np.nan],
                    [80., 80., 80., 80., 87.5,
                     87.5, 87.5, 87.5],
                    [76.47058824, 76.47058824, 76.47058824, 76.47058824, 84.61538462,
                     84.61538462, 84.61538462, 84.61538462],
                    [17.64705882, 17.64705882, 17.64705882, 17.64705882, 17.64705882,
                     17.64705882, 17.64705882, 17.64705882],
                    [13.33333333, 13.33333333, 13.33333333, 13.33333333, 7.69230769,
                     7.69230769, 7.69230769, 7.69230769],
                    [7.69230769, 7.69230769, 7.69230769, 7.69230769, 4.16666667,
                     4.16666667, 4.16666667, 4.16666667]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.Stochastic.from_params(
                close_ts,
                high_ts=high_ts,
                low_ts=low_ts,
                k_window=(2, 3),
                d_window=(2, 3),
                d_ewm=(False, True),
                param_product=True
            ).percent_d,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [82.30769231, 81.15384615, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [78.23529412, 77.91159067, 80.36199095, 78.64253394, 86.05769231,
                     85.33653846, np.nan, np.nan],
                    [47.05882353, 37.23303167, 58.03921569, 46.11161388, 51.13122172,
                     38.47459102, 63.25414781, 46.75985779],
                    [15.49019608, 21.23406006, 35.81699346, 29.19379166, 12.66968326,
                     17.69654977, 36.65158371, 25.92383107],
                    [10.51282051, 12.19382428, 12.89089995, 18.27240298, 5.92948718,
                     8.6393553, 9.83534439, 14.69432686]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )

    def test_MACD(self):
        columns = pd.MultiIndex.from_tuples([
            (2, 3, 2, False, False),
            (2, 3, 2, False, True),
            (2, 3, 2, True, False),
            (2, 3, 2, True, True),
            (2, 3, 3, False, False),
            (2, 3, 3, False, True),
            (2, 3, 3, True, False),
            (2, 3, 3, True, True),
            (2, 4, 2, False, False),
            (2, 4, 2, False, True),
            (2, 4, 2, True, False),
            (2, 4, 2, True, True),
            (2, 4, 3, False, False),
            (2, 4, 3, False, True),
            (2, 4, 3, True, False),
            (2, 4, 3, True, True),
            (3, 3, 2, False, False),
            (3, 3, 2, False, True),
            (3, 3, 2, True, False),
            (3, 3, 2, True, True),
            (3, 3, 3, False, False),
            (3, 3, 3, False, True),
            (3, 3, 3, True, False),
            (3, 3, 3, True, True),
            (3, 4, 2, False, False),
            (3, 4, 2, False, True),
            (3, 4, 2, True, False),
            (3, 4, 2, True, True),
            (3, 4, 3, False, False),
            (3, 4, 3, False, True),
            (3, 4, 3, True, False),
            (3, 4, 3, True, True)
        ], names=['macd_fast_window', 'macd_slow_window', 'macd_signal_window', 'macd_macd_ewm', 'macd_signal_ewm'])

        pd.testing.assert_frame_equal(
            vbt.MACD.from_params(
                close_ts,
                fast_window=(2, 3),
                slow_window=(3, 4),
                signal_window=(2, 3),
                macd_ewm=(False, True),
                signal_ewm=(False, True),
                param_product=True
            ).macd,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [0.5, 0.5, 0.18681319, 0.18681319, 0.5,
                     0.5, 0.18681319, 0.18681319, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     0., 0., 0., 0., np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [0.5, 0.5, 0.28333333, 0.28333333, 0.5,
                     0.5, 0.28333333, 0.28333333, 1., 1.,
                     0.45441176, 0.45441176, 1., 1., 0.45441176,
                     0.45441176, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.5,
                     0.5, 0.17107843, 0.17107843, 0.5, 0.5,
                     0.17107843, 0.17107843],
                    [0.16666667, 0.16666667, 0.05278592, 0.05278592, 0.16666667,
                     0.16666667, 0.05278592, 0.05278592, 0.5, 0.5,
                     0.1276891, 0.1276891, 0.5, 0.5, 0.1276891,
                     0.1276891, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.33333333,
                     0.33333333, 0.07490318, 0.07490318, 0.33333333, 0.33333333,
                     0.07490318, 0.07490318],
                    [-0.5, -0.5, -0.16269841, -0.16269841, -0.5,
                     -0.5, -0.16269841, -0.16269841, -0.5, -0.5,
                     -0.21898496, -0.21898496, -0.5, -0.5, -0.21898496,
                     -0.21898496, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.,
                     0., -0.05628655, -0.05628655, 0., 0.,
                     -0.05628655, -0.05628655],
                    [-0.5, -0.5, -0.30779261, -0.30779261, -0.5,
                     -0.5, -0.30779261, -0.30779261, -1., -1.,
                     -0.48467603, -0.48467603, -1., -1., -0.48467603,
                     -0.48467603, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.5,
                     -0.5, -0.17688342, -0.17688342, -0.5, -0.5,
                     -0.17688342, -0.17688342]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.MACD.from_params(
                close_ts,
                fast_window=(2, 3),
                slow_window=(3, 4),
                signal_window=(2, 3),
                macd_ewm=(False, True),
                signal_ewm=(False, True),
                param_product=True
            ).signal,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [0.5, 0.5, 0.23507326, 0.2592033, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [0.33333333, 0.26923077, 0.16805963, 0.11629896, 0.38888889,
                     0.30952381, 0.17431081, 0.13780336, 0.75, 0.625,
                     0.29105043, 0.20936977, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.41666667,
                     0.375, 0.12299081, 0.09894699, np.nan, np.nan,
                     np.nan, np.nan],
                    [-0.16666667, -0.25, -0.05495624, -0.07202427, 0.05555556,
                     -0.12222222, 0.05780695, -0.02246425, 0., -0.15384615,
                     -0.04564793, -0.08718351, 0.33333333, 0., 0.12103864,
                     -0.0237357, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.16666667,
                     0.11538462, 0.00930832, -0.00852238, 0.27777778, 0.16666667,
                     0.06323169, 0.01367694],
                    [-0.5, -0.41735537, -0.23524551, -0.22985266, -0.27777778,
                     -0.3172043, -0.13923503, -0.1697305, -0.75, -0.725,
                     -0.3518305, -0.35549096, -0.33333333, -0.53333333, -0.19199063,
                     -0.26957054, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.25,
                     -0.3, -0.11658499, -0.12216608, -0.05555556, -0.18888889,
                     -0.0527556, -0.08795525]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.MACD.from_params(
                close_ts,
                fast_window=(2, 3),
                slow_window=(3, 4),
                signal_window=(2, 3),
                macd_ewm=(False, True),
                signal_ewm=(False, True),
                param_product=True
            ).histogram,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [0., 0., 0.04826007, 0.02413004, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [-0.16666667, -0.1025641, -0.1152737, -0.06351304, -0.22222222,
                     -0.14285714, -0.12152489, -0.08501744, -0.25, -0.125,
                     -0.16336133, -0.08168066, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.08333333,
                     -0.04166667, -0.04808763, -0.02404381, np.nan, np.nan,
                     np.nan, np.nan],
                    [-0.33333333, -0.25, -0.10774217, -0.09067415, -0.55555556,
                     -0.37777778, -0.22050536, -0.14023416, -0.5, -0.34615385,
                     -0.17333703, -0.13180146, -0.83333333, -0.5, -0.3400236,
                     -0.19524927, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.16666667,
                     -0.11538462, -0.06559487, -0.04776417, -0.27777778, -0.16666667,
                     -0.11951824, -0.06996349],
                    [0., -0.08264463, -0.0725471, -0.07793995, -0.22222222,
                     -0.1827957, -0.16855758, -0.13806211, -0.25, -0.275,
                     -0.13284553, -0.12918507, -0.66666667, -0.46666667, -0.2926854,
                     -0.21510549, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.25,
                     -0.2, -0.06029844, -0.05471734, -0.44444444, -0.31111111,
                     -0.12412782, -0.08892817]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )

    def test_ATR(self):
        columns = pd.MultiIndex.from_tuples([
            (2, False),
            (2, True),
            (3, False),
            (3, True)
        ], names=['atr_window', 'atr_ewm'])
        pd.testing.assert_frame_equal(
            vbt.ATR.from_params(close_ts, high_ts, low_ts, window=(2, 3), ewm=(False, True), param_product=True).tr,
            pd.DataFrame(
                np.array([
                    [0.2, 0.2, 0.2, 0.2],
                    [1.2, 1.2, 1.2, 1.2],
                    [1.3, 1.3, 1.3, 1.3],
                    [1.4, 1.4, 1.4, 1.4],
                    [1.3, 1.3, 1.3, 1.3],
                    [1.2, 1.2, 1.2, 1.2],
                    [1.1, 1.1, 1.1, 1.1]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.ATR.from_params(close_ts, high_ts, low_ts, window=(2, 3), ewm=(False, True), param_product=True).atr,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [0.7, 0.95, np.nan, np.nan],
                    [1.25, 1.19230769, 0.9, 1.11428571],
                    [1.35, 1.3325, 1.3, 1.26666667],
                    [1.35, 1.3107438, 1.33333333, 1.28387097],
                    [1.25, 1.23681319, 1.3, 1.24126984],
                    [1.15, 1.14556267, 1.2, 1.17007874]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )

    def test_OBV(self):
        pd.testing.assert_series_equal(
            vbt.OBV.from_params(close_ts, volume_ts).obv,
            pd.Series(
                np.array([np.nan,  3.,  5.,  6.,  4.,  1., -3.]),
                index=close_ts.index,
                name=close_ts.name
            )
        )
