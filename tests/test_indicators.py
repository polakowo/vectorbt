import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
import pytest
from itertools import product, combinations

seed = 42

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
    input_names=['ts1', 'ts2'],
    param_names=['p1', 'p2'],
    param_defaults={'p2': 2},
    output_names=['o1', 'o2'],
    custom_output_funcs={
        'co1': lambda self: self.ts1 + self.ts2,
        'co2': property(lambda self: self.o1 + self.o2)
    }
).from_apply_func(lambda ts1, ts2, p1, p2: (ts1 * p1, ts2 * p2))

custom_ind = CustomInd.run(ts, ts['a'], [1, 2], 3)


# ############# factory.py ############# #


class TestFactory:
    def test_create_param_combs(self):
        assert vbt.indicators.create_param_combs(
            (combinations, [0, 1, 2, 3], 2)) == [
                   [0, 0, 0, 1, 1, 2],
                   [1, 2, 3, 2, 3, 3]
               ]
        assert vbt.indicators.create_param_combs(
            (product, (combinations, [0, 1, 2, 3], 2), [4, 5])) == [
                   [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2],
                   [1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 3, 3],
                   [4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5]
               ]
        assert vbt.indicators.create_param_combs(
            (product, (combinations, [0, 1, 2], 2), (combinations, [3, 4, 5], 2))) == [
                   [0, 0, 0, 0, 0, 0, 1, 1, 1],
                   [1, 1, 1, 2, 2, 2, 2, 2, 2],
                   [3, 3, 4, 3, 3, 4, 3, 3, 4],
                   [4, 5, 5, 4, 5, 5, 4, 5, 5]
               ]

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
            vbt.IndicatorFactory().from_custom_func(custom_func).run(ts, [0, 1], 10, b=100).output,
            target1
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func_nb).run(ts, [0, 1], 10, 100).output,
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
            vbt.IndicatorFactory().from_custom_func(custom_func).run(ts['a'], [0, 1], 10, b=100).output,
            target2
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func_nb).run(ts['a'], [0, 1], 10, 100).output,
            target2
        )
        target3 = pd.Series(
            np.array([110., 110., 110., 110., np.nan]),
            index=ts.index,
            name=(0, 'a')
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func).run(ts['a'], 0, 10, b=100).output,
            target3
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory().from_custom_func(custom_func_nb).run(ts['a'], 0, 10, 100).output,
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
            vbt.IndicatorFactory().from_apply_func(apply_func).run(ts, [0, 1], 10, b=100).output,
            target1
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func_nb).run(ts, [0, 1], 10, 100).output,
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
            vbt.IndicatorFactory().from_apply_func(apply_func).run(ts['a'], [0, 1], 10, b=100).output,
            target2
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func_nb).run(ts['a'], [0, 1], 10, 100).output,
            target2
        )
        target3 = pd.Series(
            np.array([110., 110., 110., 110., np.nan]),
            index=ts.index,
            name=(0, 'a')
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func).run(ts['a'], 0, 10, b=100).output,
            target3
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory().from_apply_func(apply_func_nb).run(ts['a'], 0, 10, 100).output,
            target3
        )

    def test_inputs(self):
        # inputs
        ts1 = ts
        ts2 = ts['a'].vbt.broadcast_to(ts)
        np.testing.assert_array_equal(
            custom_ind._ts1,
            ts1.values
        )
        np.testing.assert_array_equal(
            custom_ind._ts2,
            ts2.values
        )
        np.testing.assert_array_equal(
            custom_ind._input_mapper,
            np.array([0, 1, 2, 0, 1, 2])
        )
        pd.testing.assert_frame_equal(
            custom_ind.ts1,
            custom_ind.wrapper.wrap(ts1.vbt.tile(2).values)
        )
        pd.testing.assert_frame_equal(
            custom_ind.ts2,
            custom_ind.wrapper.wrap(ts2.vbt.tile(2))
        )

    def test_params(self):
        def apply_func(ts, p1, p2):
            return ts * (p1 + p2)

        @njit
        def apply_func_nb(ts, p1, p2):
            return ts * (p1 + p2)

        target = pd.DataFrame(
            np.array([
                [2., np.nan, 2.],
                [4., 8., 4.],
                [6., 6., np.nan],
                [8., 4., 4.],
                [np.nan, 2., 2.]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 2, 'a'),
                (0, 2, 'b'),
                (0, 2, 'c')],
                names=['custom_p1', 'custom_p2', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(apply_func) \
                .run(ts, 0, 2).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(apply_func_nb) \
                .run(ts, 0, 2).output,
            target
        )
        target2 = pd.DataFrame(
            np.array([
                [2., np.nan, 2., 3., np.nan, 3.],
                [4., 8., 4., 6., 12., 6.],
                [6., 6., np.nan, 9., 9., np.nan],
                [8., 4., 4., 12., 6., 6.],
                [np.nan, 2., 2., np.nan, 3., 3.]
            ]),
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (0, 2, 'a'),
                (0, 2, 'b'),
                (0, 2, 'c'),
                (0, 3, 'a'),
                (0, 3, 'b'),
                (0, 3, 'c')
            ], names=['custom_p1', 'custom_p2', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(apply_func) \
                .run(ts, 0, [2, 3]).output,
            target2
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(apply_func_nb) \
                .run(ts, 0, [2, 3]).output,
            target2
        )
        target3 = pd.DataFrame(
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
                .from_apply_func(apply_func) \
                .run(ts, [0, 1], [2, 3]).output,
            target3
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(apply_func_nb) \
                .run(ts, [0, 1], [2, 3]).output,
            target3
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

        def apply_func(ts, p1, p2):
            return ts * (p1 + p2)

        @njit
        def apply_func_nb(ts, p1, p2):
            return ts * (p1 + p2)

        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(apply_func) \
                .run(ts, [0, 1], [2, 3], param_product=True).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                .from_apply_func(apply_func_nb) \
                .run(ts, [0, 1], [2, 3], param_product=True).output,
            target
        )

    def test_outputs(self):

        def apply_func(ts, p):
            return (ts * p, ts * p ** 2)

        @njit
        def apply_func_nb(ts, p):
            return (ts * p, ts * p ** 2)

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
                .from_apply_func(apply_func) \
                .run(ts, [0, 1]).o1,
            target1
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(output_names=['o1', 'o2']) \
                .from_apply_func(apply_func_nb) \
                .run(ts, [0, 1]).o1,
            target1
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(output_names=['o1', 'o2']) \
                .from_apply_func(apply_func) \
                .run(ts, [0, 1]).o2,
            target2
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(output_names=['o1', 'o2']) \
                .from_apply_func(apply_func_nb) \
                .run(ts, [0, 1]).o2,
            target2
        )

    def test_cache(self):

        def caching_func(ts, params):
            np.random.seed(seed)
            return np.random.uniform(0, 1)

        @njit
        def caching_func_nb(ts, params):
            np.random.seed(seed)
            return np.random.uniform(0, 1)

        def apply_func(ts, param, c):
            return ts * param + c

        @njit
        def apply_func_nb(ts, param, c):
            return ts * param + c

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
                apply_func,
                caching_func=caching_func
            ).run(ts, [0, 1]).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(
                apply_func_nb,
                caching_func=caching_func_nb
            ).run(ts, [0, 1]).output,
            target
        )
        # return_cache
        cache = vbt.IndicatorFactory().from_apply_func(
            apply_func,
            caching_func=caching_func,
            return_cache=True
        ).run(ts, [0, 1])
        assert cache == 0.3745401188473625
        cache = vbt.IndicatorFactory().from_apply_func(
            apply_func_nb,
            caching_func=caching_func_nb,
            return_cache=True
        ).run(ts, [0, 1])
        assert cache == 0.3745401188473625
        # pass cache
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(
                apply_func,
                use_cache=cache
            ).run(ts, [0, 1]).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(
                apply_func_nb,
                use_cache=cache
            ).run(ts, [0, 1]).output,
            target
        )

    def test_return_raw(self):

        def apply_func(ts, p, a, b=10):
            return ts * p + a + b

        @njit
        def apply_func_nb(ts, p, a, b):
            return ts * p + a + b

        target = np.array([
            [110., np.nan, 110., 111., np.nan, 111.],
            [110., 110., 110., 112., 114., 112.],
            [110., 110., np.nan, 113., 113., np.nan],
            [110., 110., 110., 114., 112., 112.],
            [np.nan, 110., 110., np.nan, 111., 111.]
        ])
        target_map = [(0,), (1,)]
        np.testing.assert_array_equal(
            vbt.IndicatorFactory().from_apply_func(
                apply_func,
                return_raw=True
            ).run(ts, [0, 1], 10, b=100)[0],
            target
        )
        np.testing.assert_array_equal(
            vbt.IndicatorFactory().from_apply_func(
                apply_func,
                return_raw=True
            ).run(ts, [0, 1], 10, b=100)[1],
            target_map
        )
        np.testing.assert_array_equal(
            vbt.IndicatorFactory().from_apply_func(
                apply_func_nb,
                return_raw=True
            ).run(ts, [0, 1], 10, 100)[1],
            target_map
        )

    def test_use_raw(self):

        def apply_func(ts, p, a, b=10):
            return ts * p + a + b

        @njit
        def apply_func_nb(ts, p, a, b):
            return ts * p + a + b

        raw_results = vbt.IndicatorFactory().from_apply_func(
            apply_func,
            return_raw=True
        ).run(ts, [0, 1], 10, b=100)
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_apply_func(
                apply_func,
                use_raw=raw_results
            ).run(ts, [0, 1], 10, b=100).output,
            vbt.IndicatorFactory().from_apply_func(
                apply_func
            ).run(ts, [0, 1], 10, b=100).output
        )

    def test_no_params(self):

        def custom_func(ts, a, b=10):
            return ts + a + b

        @njit
        def custom_func_nb(ts, a, b):
            return ts + a + b

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
            vbt.IndicatorFactory(param_names=[]).from_custom_func(custom_func).run(ts, 10, b=100).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory(param_names=[]).from_custom_func(custom_func_nb).run(ts, 10, 100).output,
            target
        )

    def test_pass_1d(self):

        def custom_func(ts, a, b=10):
            return ts + a + b

        @njit
        def custom_func_nb(ts, a, b):
            return ts + a + b

        target = pd.Series(
            np.array([12., 14., 16., 18., np.nan]),
            index=ts['a'].index,
            name=ts['a'].name
        )
        with pytest.raises(Exception) as e_info:
            pd.testing.assert_series_equal(
                vbt.IndicatorFactory(param_names=[]).from_custom_func(
                    custom_func
                ).run(ts['a'], 10, b=ts['a'].values).output,
                target
            )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory(param_names=[]).from_custom_func(
                custom_func,
                pass_2d=False
            ).run(ts['a'], 10, b=ts['a'].values).output,
            target
        )
        pd.testing.assert_series_equal(
            vbt.IndicatorFactory(param_names=[]).from_custom_func(
                custom_func_nb,
                pass_2d=False
            ).run(ts['a'], 10, ts['a'].values).output,
            target
        )

    def test_pass_lists(self):

        def custom_func(ts_list, param_list):
            return ts_list[0] * param_list[0]

        def custom_func2(ts_list, param_list):
            return njit(lambda x, y: x * y)(ts_list[0], param_list[0])

        target = pd.DataFrame(
            ts.values * 2,
            index=ts.index,
            columns=pd.MultiIndex.from_tuples([
                (2, 'a'),
                (2, 'b'),
                (2, 'c')
            ], names=['custom_param', None])
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(
                custom_func,
                pass_lists=True
            ).run(ts, 2).output,
            target
        )
        pd.testing.assert_frame_equal(
            vbt.IndicatorFactory().from_custom_func(
                custom_func2,
                pass_lists=True
            ).run(ts, 2).output,
            target
        )

    def test_other(self):

        def apply_func(ts1, ts2, p1, p2):
            return (ts1 * p1, ts2 * p2, ts1 * p1 + ts2 * p2)

        @njit
        def apply_func_nb(ts1, ts2, p1, p2):
            return (ts1 * p1, ts2 * p2, ts1 * p1 + ts2 * p2)

        obj, other = vbt.IndicatorFactory(
            input_names=['ts1', 'ts2'],
            param_names=['p1', 'p2'],
            output_names=['o1', 'o2']
        ).from_apply_func(
            apply_func
        ).run(ts, ts + 1, [0, 1], [1, 2])
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
            input_names=['ts1', 'ts2'],
            param_names=['p1', 'p2'],
            output_names=['o1', 'o2']
        ).from_apply_func(
            apply_func_nb
        ).run(ts, ts + 1, [0, 1], [1, 2])
        np.testing.assert_array_equal(other, other2)

    def test_mappers(self):
        obj = vbt.IndicatorFactory(param_names=['p1', 'p2']) \
            .from_apply_func(lambda ts, p1, p2: ts * (p1 + p2)) \
            .run(ts, 0, 2)
        np.testing.assert_array_equal(
            obj._p1_mapper,
            np.array([0, 0, 0])
        )
        np.testing.assert_array_equal(
            obj._p2_mapper,
            np.array([2, 2, 2])
        )
        assert obj._tuple_mapper == [(0, 2), (0, 2), (0, 2)]
        obj = vbt.IndicatorFactory(param_names=['p1', 'p2']) \
            .from_apply_func(lambda ts, p1, p2: ts * (p1 + p2)) \
            .run(ts, [0, 1], [1, 2])
        np.testing.assert_array_equal(
            obj._p1_mapper,
            np.array([0, 0, 0, 1, 1, 1])
        )
        np.testing.assert_array_equal(
            obj._p2_mapper,
            np.array([1, 1, 1, 2, 2, 2])
        )
        assert obj._tuple_mapper == [(0, 1), (0, 1), (0, 1), (1, 2), (1, 2), (1, 2)]

    def test_short_name(self):
        assert vbt.IndicatorFactory(short_name='my_ind') \
                   .from_apply_func(lambda ts, p: ts * p) \
                   .run(ts, [0, 1]).short_name == 'my_ind'
        assert vbt.IndicatorFactory() \
                   .from_apply_func(lambda ts, p: ts * p) \
                   .run(ts, [0, 1], short_name='my_ind').short_name == 'my_ind'

    def test_hide_params(self):
        assert vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                   .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
                   .run(ts, [0, 1], 2, hide_params=[]) \
                   .output.columns.names == ['custom_p1', 'custom_p2', None]
        assert vbt.IndicatorFactory(param_names=['p1', 'p2']) \
                   .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
                   .run(ts, [0, 1], 2, hide_params=['p2']) \
                   .output.columns.names == ['custom_p1', None]

    def test_hide_default(self):
        assert vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
                   .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
                   .run(ts, [0, 1], 2, hide_default=False) \
                   .output.columns.names == ['custom_p1', 'custom_p2', None]
        assert vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
                   .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
                   .run(ts, [0, 1], 2, hide_default=True) \
                   .output.columns.names == ['custom_p1', None]

    def test_run_combs(self):
        # itertools.combinations
        ind1 = vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
            .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
            .run(ts, [2, 2, 3], [10, 10, 11], short_name='custom_1')
        ind2 = vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
            .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
            .run(ts, [3, 4, 4], [11, 12, 12], short_name='custom_2')
        ind1_1, ind2_1 = vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
            .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
            .run_combs(ts, [2, 3, 4], [10, 11, 12], r=2, speed_up=False)
        ind1_2, ind2_2 = vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
            .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
            .run_combs(ts, [2, 3, 4], [10, 11, 12], r=2, speed_up=True)
        pd.testing.assert_frame_equal(
            ind1.output,
            ind1_1.output
        )
        pd.testing.assert_frame_equal(
            ind2.output,
            ind2_1.output
        )
        pd.testing.assert_frame_equal(
            ind1.output,
            ind1_2.output
        )
        pd.testing.assert_frame_equal(
            ind2.output,
            ind2_2.output
        )
        # itertools.product
        ind3 = vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
            .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
            .run(ts, [2, 2, 2, 3, 3, 3, 4, 4, 4], [10, 10, 10, 11, 11, 11, 12, 12, 12], short_name='custom_1')
        ind4 = vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
            .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
            .run(ts, [2, 3, 4, 2, 3, 4, 2, 3, 4], [10, 11, 12, 10, 11, 12, 10, 11, 12], short_name='custom_2')
        ind3_1, ind4_1 = vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
            .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
            .run_combs(ts, [2, 3, 4], [10, 11, 12], r=2, comb_func=product, speed_up=False)
        ind3_2, ind4_2 = vbt.IndicatorFactory(param_names=['p1', 'p2'], param_defaults={'p2': 2}) \
            .from_apply_func(lambda ts, p1, p2: ts * p1 * p2) \
            .run_combs(ts, [2, 3, 4], [10, 11, 12], r=2, comb_func=product, speed_up=True)
        pd.testing.assert_frame_equal(
            ind3.output,
            ind3_1.output
        )
        pd.testing.assert_frame_equal(
            ind4.output,
            ind4_1.output
        )
        pd.testing.assert_frame_equal(
            ind3.output,
            ind3_2.output
        )
        pd.testing.assert_frame_equal(
            ind4.output,
            ind4_2.output
        )

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
                (2, 3, 'a'),
                (2, 3, 'b'),
                (2, 3, 'c')
            ], names=['custom_p1', 'custom_p2', None])
        )
        assert custom_ind.wrapper.ndim == 2
        assert custom_ind.wrapper.shape == (5, 6)
        assert custom_ind.wrapper.freq == pd.Timedelta('1 days 00:00:00')

    def test_properties(self):
        # Class properties
        assert CustomInd.input_names == ['ts1', 'ts2']
        assert CustomInd.param_names == ['p1', 'p2']
        assert CustomInd.output_names == ['o1', 'o2']
        assert CustomInd.output_flags == {}

        # Instance properties
        assert custom_ind.input_names == ['ts1', 'ts2']
        assert custom_ind.param_names == ['p1', 'p2']
        assert custom_ind.output_names == ['o1', 'o2']
        assert custom_ind.output_flags == {}
        assert custom_ind.short_name == 'custom'
        assert custom_ind.level_names == ['custom_p1', 'custom_p2']
        np.testing.assert_array_equal(custom_ind.p1_array, np.array([1, 2]))
        np.testing.assert_array_equal(custom_ind.p2_array, np.array([3, 3]))

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
            getattr(custom_ind.tuple_loc[(1, 3):(2, 3)], test_attr),
            pd.concat((
                getattr(custom_ind, test_attr).xs((1, 3), level=('custom_p1', 'custom_p2'), drop_level=False, axis=1),
                getattr(custom_ind, test_attr).xs((2, 3), level=('custom_p1', 'custom_p2'), drop_level=False, axis=1)
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
            getattr(custom_ind, test_attr + '_above')(2),
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
                (2, 2, 3, 'a'),
                (2, 2, 3, 'b'),
                (2, 2, 3, 'c'),
                (3, 1, 3, 'a'),
                (3, 1, 3, 'b'),
                (3, 1, 3, 'c'),
                (3, 2, 3, 'a'),
                (3, 2, 3, 'b'),
                (3, 2, 3, 'c')
            ], names=['custom_o1_above', 'custom_p1', 'custom_p2', None])
        )
        pd.testing.assert_frame_equal(
            custom_ind.o1_above([2, 3], multiple=True),
            target
        )
        columns2 = target.columns.rename('my_above', 0)
        pd.testing.assert_frame_equal(
            custom_ind.o1_above([2, 3], level_name='my_above', multiple=True),
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
                    (2, 3, 'a'),
                    (2, 3, 'b'),
                    (2, 3, 'c')
                ], names=['custom_p1', 'custom_p2', None])
            )
        )

    def test_attr_list(self):
        attr_list = [
            '_iloc',
            '_indexing_func',
            '_input_mapper',
            '_loc',
            '_short_name',
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
            'run',
            'iloc',
            'loc',
            'short_name',
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
        assert len(set(attr_list).intersection(dir(custom_ind))) == len(attr_list)

    def test_from_talib(self):
        # with params
        BBANDS = vbt.IndicatorFactory.from_talib('BBANDS')
        pd.testing.assert_frame_equal(
            BBANDS.run(ts, timeperiod=2, nbdevup=2, nbdevdn=2).upperband,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan],
                    [2.5, np.nan, 2.5],
                    [3.5, 4.5, np.nan],
                    [4.5, 3.5, np.nan],
                    [np.nan, 2.5, np.nan]
                ]),
                index=ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (2, 'a'),
                    (2, 'b'),
                    (2, 'c')
                ], names=['bbands_timeperiod', None])
            )
        )
        pd.testing.assert_frame_equal(
            BBANDS.run(ts, timeperiod=2, nbdevup=2, nbdevdn=2).middleband,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan],
                    [1.5, np.nan, 1.5],
                    [2.5, 3.5, np.nan],
                    [3.5, 2.5, np.nan],
                    [np.nan, 1.5, np.nan]
                ]),
                index=ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (2, 'a'),
                    (2, 'b'),
                    (2, 'c')
                ], names=['bbands_timeperiod', None])
            )
        )
        pd.testing.assert_frame_equal(
            BBANDS.run(ts, timeperiod=2, nbdevup=2, nbdevdn=2).lowerband,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan],
                    [0.5, np.nan, 0.5],
                    [1.5, 2.5, np.nan],
                    [2.5, 1.5, np.nan],
                    [np.nan, 0.5, np.nan]
                ]),
                index=ts.index,
                columns=pd.MultiIndex.from_tuples([
                    (2, 'a'),
                    (2, 'b'),
                    (2, 'c')
                ], names=['bbands_timeperiod', None])
            )
        )
        attr_list = [
            '__class__',
            '__delattr__',
            '__dict__',
            '__dir__',
            '__doc__',
            '__eq__',
            '__format__',
            '__ge__',
            '__getattribute__',
            '__getitem__',
            '__gt__',
            '__hash__',
            '__init__',
            '__init_subclass__',
            '__le__',
            '__lt__',
            '__module__',
            '__ne__',
            '__new__',
            '__reduce__',
            '__reduce_ex__',
            '__repr__',
            '__setattr__',
            '__sizeof__',
            '__str__',
            '__subclasshook__',
            '__weakref__',
            '_run_combs',
            '_run',
            'close',
            'close_above',
            'close_below',
            'close_equal',
            'run_combs',
            'run',
            'iloc',
            'loc',
            'lowerband',
            'lowerband_above',
            'lowerband_below',
            'lowerband_equal',
            'matype_loc',
            'middleband',
            'middleband_above',
            'middleband_below',
            'middleband_equal',
            'nbdevdn_loc',
            'nbdevup_loc',
            'short_name',
            'timeperiod_loc',
            'tuple_loc',
            'upperband',
            'upperband_above',
            'upperband_below',
            'upperband_equal',
            'xs'
        ]
        assert len(set(attr_list).intersection(dir(BBANDS))) == len(attr_list)
        # without params
        OBV = vbt.IndicatorFactory.from_talib('OBV')
        pd.testing.assert_frame_equal(
            OBV.run(ts, ts * 2).real,
            pd.DataFrame(
                np.array([
                    [2., np.nan, 2.],
                    [6., 8., 6.],
                    [12., 2., 6.],
                    [20., -2., 6.],
                    [20., -4., 4.]
                ]),
                index=ts.index,
                columns=ts.columns
            )
        )
        attr_list = [
            '__class__',
            '__delattr__',
            '__dict__',
            '__dir__',
            '__doc__',
            '__eq__',
            '__format__',
            '__ge__',
            '__getattribute__',
            '__getitem__',
            '__gt__',
            '__hash__',
            '__init__',
            '__init_subclass__',
            '__le__',
            '__lt__',
            '__module__',
            '__ne__',
            '__new__',
            '__reduce__',
            '__reduce_ex__',
            '__repr__',
            '__setattr__',
            '__sizeof__',
            '__str__',
            '__subclasshook__',
            '__weakref__',
            '_run',
            'close',
            'close_above',
            'close_below',
            'close_equal',
            'run',
            'iloc',
            'loc',
            'real',
            'real_above',
            'real_below',
            'real_equal',
            'short_name',
            'volume',
            'volume_above',
            'volume_below',
            'volume_equal',
            'xs'
        ]
        assert len(set(attr_list).intersection(dir(OBV))) == len(attr_list)


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
            vbt.MA.run(close_ts, window=(2, 3), ewm=(False, True), param_product=True).ma,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [1.5, 1.66666667, np.nan, np.nan],
                    [2.5, 2.55555556, 2., 2.25],
                    [3.5, 3.51851852, 3., 3.125],
                    [3.5, 3.17283951, 3.33333333, 3.0625],
                    [2.5, 2.3909465, 3., 2.53125],
                    [1.5, 1.46364883, 2., 1.765625]
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

    def test_MSTD(self):
        pd.testing.assert_frame_equal(
            vbt.MSTD.run(close_ts, window=(2, 3), ewm=(False, True), param_product=True).mstd,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [0.5, 0.70710678, np.nan, np.nan],
                    [0.5, 0.97467943, 0.81649658, 1.04880885],
                    [0.5, 1.11434207, 0.81649658, 1.30018314],
                    [0.5, 0.73001838, 0.47140452, 0.91715673],
                    [0.5, 0.88824841, 0.81649658, 0.9182094],
                    [0.5, 1.05965735, 0.81649658, 1.14049665]
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
            vbt.BollingerBands.run(
                close_ts,
                window=(2, 3),
                alpha=(2., 3.),
                ewm=(False, True),
                param_product=True
            ).middle,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [1.5, 1.5, 1.66666667, 1.66666667, np.nan,
                     np.nan, np.nan, np.nan],
                    [2.5, 2.5, 2.55555556, 2.55555556, 2.,
                     2., 2.25, 2.25],
                    [3.5, 3.5, 3.51851852, 3.51851852, 3.,
                     3., 3.125, 3.125],
                    [3.5, 3.5, 3.17283951, 3.17283951, 3.33333333,
                     3.33333333, 3.0625, 3.0625],
                    [2.5, 2.5, 2.3909465, 2.3909465, 3.,
                     3., 2.53125, 2.53125],
                    [1.5, 1.5, 1.46364883, 1.46364883, 2.,
                     2., 1.765625, 1.765625]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.run(
                close_ts,
                window=(2, 3),
                alpha=(2., 3.),
                ewm=(False, True),
                param_product=True
            ).upper,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [2.5, 3., 3.08088023, 3.78798701, np.nan,
                     np.nan, np.nan, np.nan],
                    [3.5, 4., 4.50491442, 5.47959386, 3.63299316,
                     4.44948974, 4.3476177, 5.39642654],
                    [4.5, 5., 5.74720265, 6.86154472, 4.63299316,
                     5.44948974, 5.72536627, 7.02554941],
                    [4.5, 5., 4.63287626, 5.36289463, 4.27614237,
                     4.7475469, 4.89681346, 5.8139702],
                    [3.5, 4., 4.16744332, 5.05569172, 4.63299316,
                     5.44948974, 4.3676688, 5.2858782],
                    [2.5, 3., 3.58296354, 4.64262089, 3.63299316,
                     4.44948974, 4.04661829, 5.18711494]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.run(
                close_ts,
                window=(2, 3),
                alpha=(2., 3.),
                ewm=(False, True),
                param_product=True
            ).lower,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [0.5, 0., 0.2524531, -0.45465368, np.nan,
                     np.nan, np.nan, np.nan],
                    [1.5, 1., 0.60619669, -0.36848275, 0.36700684,
                     -0.44948974, 0.1523823, -0.89642654],
                    [2.5, 2., 1.28983438, 0.17549232, 1.36700684,
                     0.55051026, 0.52463373, -0.77554941],
                    [2.5, 2., 1.71280275, 0.98278438, 2.39052429,
                     1.91911977, 1.22818654, 0.3110298],
                    [1.5, 1., 0.61444969, -0.27379872, 1.36700684,
                     0.55051026, 0.6948312, -0.2233782],
                    [0.5, 0., -0.65566587, -1.71532322, 0.36700684,
                     -0.44948974, -0.51536829, -1.65586494]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.run(
                close_ts,
                window=(2, 3),
                alpha=(2., 3.),
                ewm=(False, True),
                param_product=True
            ).percent_b,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [0.75, 0.66666667, 0.61785113, 0.57856742, np.nan,
                     np.nan, np.nan, np.nan],
                    [0.75, 0.66666667, 0.61399759, 0.5759984, 0.80618622,
                     0.70412415, 0.67877424, 0.61918282],
                    [0.75, 0.66666667, 0.60801923, 0.57201282, 0.80618622,
                     0.70412415, 0.66824553, 0.61216369],
                    [0.25, 0.33333333, 0.44080988, 0.46053992, 0.3232233,
                     0.38214887, 0.48296365, 0.48864244],
                    [0.25, 0.33333333, 0.38996701, 0.42664468, 0.19381378,
                     0.29587585, 0.35535707, 0.40357138],
                    [0.25, 0.33333333, 0.3906135, 0.42707567, 0.19381378,
                     0.29587585, 0.3321729, 0.38811526]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.BollingerBands.run(
                close_ts,
                window=(2, 3),
                alpha=(2., 3.),
                ewm=(False, True),
                param_product=True
            ).bandwidth,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [1.33333333, 2., 1.69705627, 2.54558441, np.nan,
                     np.nan, np.nan, np.nan],
                    [0.8, 1.2, 1.5255852, 2.2883778, 1.63299316,
                     2.44948974, 1.86454906, 2.7968236],
                    [0.57142857, 0.85714286, 1.26683098, 1.90024647, 1.08866211,
                     1.63299316, 1.66423442, 2.49635162],
                    [0.57142857, 0.85714286, 0.92033445, 1.38050168, 0.56568542,
                     0.84852814, 1.197919, 1.79687849],
                    [0.8, 1.2, 1.48601971, 2.22902956, 1.08866211,
                     1.63299316, 1.45099757, 2.17649636],
                    [1.33333333, 2., 2.8959333, 4.34389996, 1.63299316,
                     2.44948974, 2.58378001, 3.87567002]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )

    def test_RSI(self):
        pd.testing.assert_frame_equal(
            vbt.RSI.run(close_ts, window=(2, 3), ewm=(False, True), param_product=True).rsi,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [100., 100., np.nan, np.nan],
                    [100., 100., 100., 100.],
                    [50., 33.33333333, 66.66666667, 50.],
                    [0., 11.11111111, 33.33333333, 25.],
                    [0., 3.7037037, 0., 12.5]
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
            vbt.Stochastic.run(
                high_ts,
                low_ts,
                close_ts,
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
            vbt.Stochastic.run(
                high_ts,
                low_ts,
                close_ts,
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
                    [82.30769231, 81.53846154, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan],
                    [78.23529412, 78.15987934, 80.36199095, 79.38914027, 86.05769231,
                     85.57692308, np.nan, np.nan],
                    [47.05882353, 37.81799899, 58.03921569, 48.51809955, 51.13122172,
                     40.29034691, 63.25414781, 51.85237557],
                    [15.49019608, 21.49488855, 35.81699346, 30.92571644, 12.66968326,
                     18.55832076, 36.65158371, 29.77234163],
                    [10.51282051, 12.29316798, 12.89089995, 19.30901207, 5.92948718,
                     8.9638847, 9.83534439, 16.96950415]
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
            vbt.MACD.run(
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
                    [0.5, 0.5, 0.30555556, 0.30555556, 0.5,
                     0.5, 0.30555556, 0.30555556, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     0., 0., 0., 0., np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [0.5, 0.5, 0.39351852, 0.39351852, 0.5,
                     0.5, 0.39351852, 0.39351852, 1., 1.,
                     0.69451852, 0.69451852, 1., 1., 0.69451852,
                     0.69451852, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.5,
                     0.5, 0.301, 0.301, 0.5, 0.5,
                     0.301, 0.301],
                    [0.16666667, 0.16666667, 0.11033951, 0.11033951, 0.16666667,
                     0.16666667, 0.11033951, 0.11033951, 0.5, 0.5,
                     0.27843951, 0.27843951, 0.5, 0.5, 0.27843951,
                     0.27843951, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.33333333,
                     0.33333333, 0.1681, 0.1681, 0.33333333, 0.33333333,
                     0.1681, 0.1681],
                    [-0.5, -0.5, -0.1403035, -0.1403035, -0.5,
                     -0.5, -0.1403035, -0.1403035, -0.5, -0.5,
                     -0.1456935, -0.1456935, -0.5, -0.5, -0.1456935,
                     -0.1456935, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.,
                     0., -0.00539, -0.00539, 0., 0.,
                     -0.00539, -0.00539],
                    [-0.5, -0.5, -0.30197617, -0.30197617, -0.5,
                     -0.5, -0.30197617, -0.30197617, -1., -1.,
                     -0.45833517, -0.45833517, -1., -1., -0.45833517,
                     -0.45833517, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.5,
                     -0.5, -0.156359, -0.156359, -0.5, -0.5,
                     -0.156359, -0.156359]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.MACD.run(
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
                    [0.5, 0.5, 0.34953704, 0.36419753, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [0.33333333, 0.27777778, 0.25192901, 0.19495885, 0.38888889,
                     0.33333333, 0.26980453, 0.22993827, 0.75, 0.66666667,
                     0.48647901, 0.41713251, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.41666667,
                     0.38888889, 0.23455, 0.2124, np.nan, np.nan,
                     np.nan, np.nan],
                    [-0.16666667, -0.24074074, -0.014982, -0.02854938, 0.05555556,
                     -0.08333333, 0.12118484, 0.04481739, 0., -0.11111111,
                     0.066373, 0.04191517, 0.33333333, 0.125, 0.27575484,
                     0.17039276, 0., 0., 0., 0.,
                     0., 0., 0., 0., 0.16666667,
                     0.12962963, 0.081355, 0.06720667, 0.27777778, 0.20833333,
                     0.15457, 0.11458],
                    [-0.5, -0.41358025, -0.22113983, -0.2108339, -0.27777778,
                     -0.29166667, -0.11064672, -0.12857939, -0.75, -0.7037037,
                     -0.30201433, -0.29158505, -0.33333333, -0.4375, -0.10852972,
                     -0.1439712, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.25,
                     -0.29012346, -0.0808745, -0.08183711, -0.05555556, -0.14583333,
                     0.002117, -0.0208895]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )
        pd.testing.assert_frame_equal(
            vbt.MACD.run(
                close_ts,
                fast_window=(2, 3),
                slow_window=(3, 4),
                signal_window=(2, 3),
                macd_ewm=(False, True),
                signal_ewm=(False, True),
                param_product=True
            ).hist,
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
                    [0., 0., 0.04398148, 0.02932099, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan, np.nan, np.nan, np.nan,
                     np.nan, np.nan],
                    [-0.16666667, -0.11111111, -0.14158951, -0.08461934, -0.22222222,
                     -0.16666667, -0.15946502, -0.11959877, -0.25, -0.16666667,
                     -0.20803951, -0.138693, np.nan, np.nan, np.nan,
                     np.nan, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.08333333,
                     -0.05555556, -0.06645, -0.0443, np.nan, np.nan,
                     np.nan, np.nan],
                    [-0.33333333, -0.25925926, -0.1253215, -0.11175412, -0.55555556,
                     -0.41666667, -0.26148834, -0.18512088, -0.5, -0.38888889,
                     -0.2120665, -0.18760867, -0.83333333, -0.625, -0.42144834,
                     -0.31608626, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.16666667,
                     -0.12962963, -0.086745, -0.07259667, -0.27777778, -0.20833333,
                     -0.15996, -0.11997],
                    [0., -0.08641975, -0.08083633, -0.09114226, -0.22222222,
                     -0.20833333, -0.19132945, -0.17339678, -0.25, -0.2962963,
                     -0.15632083, -0.16675011, -0.66666667, -0.5625, -0.34980545,
                     -0.31436396, 0., 0., 0., 0.,
                     0., 0., 0., 0., -0.25,
                     -0.20987654, -0.0754845, -0.07452189, -0.44444444, -0.35416667,
                     -0.158476, -0.1354695]
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
            vbt.ATR.run(high_ts, low_ts, close_ts, window=(2, 3), ewm=(False, True), param_product=True).tr,
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
            vbt.ATR.run(high_ts, low_ts, close_ts, window=(2, 3), ewm=(False, True), param_product=True).atr,
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [0.7, 0.86666667, np.nan, np.nan],
                    [1.25, 1.15555556, 0.9, 1.],
                    [1.35, 1.31851852, 1.3, 1.2],
                    [1.35, 1.30617284, 1.33333333, 1.25],
                    [1.25, 1.23539095, 1.3, 1.225],
                    [1.15, 1.14513032, 1.2, 1.1625]
                ]),
                index=close_ts.index,
                columns=columns
            )
        )

    def test_OBV(self):
        pd.testing.assert_series_equal(
            vbt.OBV.run(close_ts, volume_ts).obv,
            pd.Series(
                np.array([4., 7., 9., 10., 8., 5., 1.]),
                index=close_ts.index,
                name=close_ts.name
            )
        )
