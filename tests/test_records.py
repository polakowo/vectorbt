import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit

from vectorbt.base.array_wrapper import ArrayWrapper
from vectorbt.records import (
    drawdown_dt,
    order_dt,
    event_dt,
    trade_dt,
    position_dt
)
from vectorbt.records.drawdowns import ActiveDrawdowns, RecoveredDrawdowns
from vectorbt.records.orders import BaseOrders
from vectorbt.records.events import BaseEvents, BaseEventsByResult

from tests.utils import record_arrays_close

day_dt = np.timedelta64(86400000000000)

# ############# base.py ############# #

example_dt = np.dtype([
    ('col', np.int64),
    ('idx', np.int64),
    ('some_field1', np.float64),
    ('some_field2', np.float64)
], align=True)

records_arr = np.array([
    (0, 0, 10, 21),
    (0, 1, 11, 22),
    (0, 2, 12, 23),
    (1, 0, 13, 24),
    (1, 1, 14, 25),
    (1, 2, 15, 26),
    (2, 0, 16, 27),
    (2, 1, 17, 28),
    (2, 2, 18, 29)
], dtype=example_dt)

wrapper = ArrayWrapper(
    index=['x', 'y', 'z'],
    columns=['a', 'b', 'c', 'd'],
    ndim=2,
    freq='1 days'
)
records = vbt.Records(records_arr, wrapper)

mapped_array = vbt.MappedArray(
    records_arr['some_field1'],
    records_arr['col'],
    wrapper,
    idx_arr=records_arr['idx']
)


class TestMappedArray:
    def test_mapped_arr(self):
        np.testing.assert_array_equal(
            mapped_array.mapped_arr,
            np.array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
        )
        np.testing.assert_array_equal(
            mapped_array['a'].mapped_arr,
            np.array([10., 11., 12.])
        )

    def test_col_arr(self):
        np.testing.assert_array_equal(
            mapped_array.col_arr,
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        )
        np.testing.assert_array_equal(
            mapped_array['a'].col_arr,
            np.array([0, 0, 0])
        )

    def test_idx_arr(self):
        np.testing.assert_array_equal(
            mapped_array.idx_arr,
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        )
        np.testing.assert_array_equal(
            mapped_array['a'].idx_arr,
            np.array([0, 1, 2])
        )

    def test_col_index(self):
        np.testing.assert_array_equal(
            mapped_array.col_index,
            np.array([
                [0, 3],
                [3, 6],
                [6, 9],
                [-1, -1]
            ])
        )
        np.testing.assert_array_equal(
            mapped_array['a'].col_index,
            np.array([
                [0, 3]
            ])
        )

    def test_filter_by_mask(self):
        mask = mapped_array.mapped_arr >= mapped_array.mapped_arr.mean()
        filtered = mapped_array.filter_by_mask(mask)
        np.testing.assert_array_equal(
            filtered.mapped_arr,
            np.array([14., 15., 16., 17., 18.])
        )
        np.testing.assert_array_equal(filtered.col_arr, mapped_array.col_arr[mask])
        np.testing.assert_array_equal(filtered.idx_arr, mapped_array.idx_arr[mask])
        mask_a = mapped_array['a'].mapped_arr >= mapped_array['a'].mapped_arr.mean()
        np.testing.assert_array_equal(
            mapped_array['a'].filter_by_mask(mask_a).mapped_arr,
            np.array([11., 12.])
        )

    def test_to_matrix(self):
        target = pd.DataFrame(
            np.array([
                [10., 13., 16., np.nan],
                [11., 14., 17., np.nan],
                [12., 15., 18., np.nan]
            ]),
            index=wrapper.index,
            columns=wrapper.columns
        )
        pd.testing.assert_frame_equal(
            mapped_array.to_matrix(),
            target
        )
        pd.testing.assert_series_equal(
            mapped_array['a'].to_matrix(),
            target['a']
        )
        pd.testing.assert_frame_equal(
            mapped_array.to_matrix(default_val=0.),
            target.fillna(0.)
        )
        mapped_array2 = vbt.MappedArray(
            records_arr['some_field1'].tolist() + [1],
            records_arr['col'].tolist() + [2],
            wrapper,
            idx_arr=records_arr['idx'].tolist() + [2]
        )
        try:
            _ = mapped_array2.to_matrix()
            raise Exception
        except:
            pass

    def test_reduce(self):
        @njit
        def mean_reduce_nb(col, a):
            return np.mean(a)

        target = pd.Series(
            np.array([11., 14., 17., np.nan]),
            index=wrapper.columns
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb),
            target
        )
        assert mapped_array['a'].reduce(mean_reduce_nb) == target['a']
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, default_val=0.),
            target.fillna(0.)
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, time_units=True),
            target * day_dt
        )

    def test_reduce_to_array(self):
        @njit
        def min_max_reduce_nb(col, a):
            return np.array([np.min(a), np.max(a)])

        target = pd.DataFrame(
            np.array([
                [10., 13., 16., np.nan],
                [12., 15., 18., np.nan]
            ]),
            index=pd.Index(['min', 'max'], dtype='object'),
            columns=wrapper.columns
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce_to_array(min_max_reduce_nb, index=['min', 'max']),
            target
        )
        pd.testing.assert_series_equal(
            mapped_array['a'].reduce_to_array(min_max_reduce_nb, index=['min', 'max']),
            target['a']
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce_to_array(min_max_reduce_nb, default_val=0., index=['min', 'max']),
            target.fillna(0.)
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce_to_array(min_max_reduce_nb, time_units=True, index=['min', 'max']),
            target * day_dt
        )

    def test_nst(self):
        pd.testing.assert_series_equal(
            mapped_array.nst(0),
            pd.Series(np.array([10., 13., 16., np.nan]), index=wrapper.columns)
        )
        assert mapped_array['a'].nst(0) == 10.
        pd.testing.assert_series_equal(
            mapped_array.nst(-1),
            pd.Series(np.array([12., 15., 18., np.nan]), index=wrapper.columns)
        )
        assert mapped_array['a'].nst(-1) == 12.
        try:
            _ = mapped_array.nst(10)
            raise Exception
        except:
            pass

    def test_min(self):
        pd.testing.assert_series_equal(
            mapped_array.min(),
            mapped_array.to_matrix().min()
        )
        assert mapped_array['a'].min() == mapped_array['a'].to_matrix().min()

    def test_max(self):
        pd.testing.assert_series_equal(
            mapped_array.max(),
            mapped_array.to_matrix().max()
        )
        assert mapped_array['a'].max() == mapped_array['a'].to_matrix().max()

    def test_mean(self):
        pd.testing.assert_series_equal(
            mapped_array.mean(),
            mapped_array.to_matrix().mean()
        )
        assert mapped_array['a'].mean() == mapped_array['a'].to_matrix().mean()

    def test_median(self):
        pd.testing.assert_series_equal(
            mapped_array.median(),
            mapped_array.to_matrix().median()
        )
        assert mapped_array['a'].median() == mapped_array['a'].to_matrix().median()

    def test_std(self):
        pd.testing.assert_series_equal(
            mapped_array.std(),
            mapped_array.to_matrix().std()
        )
        assert mapped_array['a'].std() == mapped_array['a'].to_matrix().std()
        pd.testing.assert_series_equal(
            mapped_array.std(ddof=0.),
            mapped_array.to_matrix().std(ddof=0.)
        )

    def test_sum(self):
        pd.testing.assert_series_equal(
            mapped_array.sum(),
            mapped_array.to_matrix().sum()
        )
        assert mapped_array['a'].sum() == mapped_array['a'].to_matrix().sum()

    def test_count(self):
        pd.testing.assert_series_equal(
            mapped_array.count(),
            mapped_array.to_matrix().count()
        )
        assert mapped_array['a'].count() == mapped_array['a'].to_matrix().count()

    def test_describe(self):
        pd.testing.assert_frame_equal(
            mapped_array.describe(percentiles=None),
            mapped_array.to_matrix().describe(percentiles=None)
        )
        pd.testing.assert_series_equal(
            mapped_array['a'].describe(),
            mapped_array['a'].to_matrix().describe()
        )
        pd.testing.assert_frame_equal(
            mapped_array.describe(percentiles=[]),
            mapped_array.to_matrix().describe(percentiles=[])
        )
        pd.testing.assert_frame_equal(
            mapped_array.describe(percentiles=np.arange(0, 1, 0.1)),
            mapped_array.to_matrix().describe(percentiles=np.arange(0, 1, 0.1))
        )

    def test_idxmin(self):
        pd.testing.assert_series_equal(
            mapped_array.idxmin(),
            mapped_array.to_matrix().idxmin()
        )
        assert mapped_array['a'].idxmin() == mapped_array['a'].to_matrix().idxmin()

    def test_idxmax(self):
        pd.testing.assert_series_equal(
            mapped_array.idxmax(),
            mapped_array.to_matrix().idxmax()
        )
        assert mapped_array['a'].idxmax() == mapped_array['a'].to_matrix().idxmax()

    def test_indexing(self):
        np.testing.assert_array_equal(
            mapped_array['a'].mapped_arr,
            np.array([10., 11., 12.])
        )
        np.testing.assert_array_equal(
            mapped_array['a'].col_arr,
            np.array([0, 0, 0])
        )
        pd.testing.assert_index_equal(
            mapped_array['a'].wrapper.columns,
            pd.Index(['a'], dtype='object')
        )
        np.testing.assert_array_equal(
            mapped_array[['a', 'a']].mapped_arr,
            np.array([10., 11., 12., 10., 11., 12.])
        )
        np.testing.assert_array_equal(
            mapped_array[['a', 'a']].col_arr,
            np.array([0, 0, 0, 1, 1, 1])
        )
        pd.testing.assert_index_equal(
            mapped_array[['a', 'a']].wrapper.columns,
            pd.Index(['a', 'a'], dtype='object')
        )
        np.testing.assert_array_equal(
            mapped_array[['a', 'b']].mapped_arr,
            np.array([10., 11., 12., 13., 14., 15.])
        )
        np.testing.assert_array_equal(
            mapped_array[['a', 'b']].col_arr,
            np.array([0, 0, 0, 1, 1, 1])
        )
        pd.testing.assert_index_equal(
            mapped_array[['a', 'b']].wrapper.columns,
            pd.Index(['a', 'b'], dtype='object')
        )
        try:
            _ = mapped_array.iloc[::2, :]  # changing time not supported
            raise Exception
        except:
            pass
        _ = mapped_array.iloc[np.arange(mapped_array.wrapper.shape[0]), :]  # won't change time

    def test_magic(self):
        a = vbt.MappedArray(
            records_arr['some_field1'],
            records_arr['col'],
            wrapper,
            idx_arr=records_arr['idx']
        )
        b = records_arr['some_field2']
        a_bool = vbt.MappedArray(
            records_arr['some_field1'] > np.mean(records_arr['some_field1']),
            records_arr['col'],
            wrapper,
            idx_arr=records_arr['idx']
        )
        b_bool = records_arr['some_field2'] > np.mean(records_arr['some_field2'])

        # test what's allowed
        np.testing.assert_array_equal((a * b).mapped_arr, a.mapped_arr * b)
        np.testing.assert_array_equal((a * vbt.MappedArray(
            records_arr['some_field2'],
            records_arr['col'],
            wrapper,
            idx_arr=records_arr['idx']
        )).mapped_arr, a.mapped_arr * b)
        try:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                records_arr['some_field2'],
                records_arr['col'] * 2,
                wrapper,
                idx_arr=records_arr['idx']
            )).mapped_arr, a.mapped_arr * b)
            raise Exception
        except:
            pass
        try:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                records_arr['some_field2'],
                records_arr['col'],
                wrapper,
                idx_arr=None
            )).mapped_arr, a.mapped_arr * b)
            raise Exception
        except:
            pass
        try:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                records_arr['some_field2'],
                records_arr['col'],
                wrapper,
                idx_arr=records_arr['idx'] * 2
            )).mapped_arr, a.mapped_arr * b)
            raise Exception
        except:
            pass
        try:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                records_arr['some_field2'],
                records_arr['col'],
                ArrayWrapper(
                    index=['x', 'y', 'z'],
                    columns=['a', 'b', 'c', 'd'],
                    ndim=2,
                    freq=None
                ),
                idx_arr=records_arr['idx']
            )).mapped_arr, a.mapped_arr * b)
            raise Exception
        except:
            pass

        # binary ops
        # comparison ops
        np.testing.assert_array_equal((a == b).mapped_arr, a.mapped_arr == b)
        np.testing.assert_array_equal((a != b).mapped_arr, a.mapped_arr != b)
        np.testing.assert_array_equal((a < b).mapped_arr, a.mapped_arr < b)
        np.testing.assert_array_equal((a > b).mapped_arr, a.mapped_arr > b)
        np.testing.assert_array_equal((a <= b).mapped_arr, a.mapped_arr <= b)
        np.testing.assert_array_equal((a >= b).mapped_arr, a.mapped_arr >= b)
        # arithmetic ops
        np.testing.assert_array_equal((a + b).mapped_arr, a.mapped_arr + b)
        np.testing.assert_array_equal((a - b).mapped_arr, a.mapped_arr - b)
        np.testing.assert_array_equal((a * b).mapped_arr, a.mapped_arr * b)
        np.testing.assert_array_equal((a ** b).mapped_arr, a.mapped_arr ** b)
        np.testing.assert_array_equal((a % b).mapped_arr, a.mapped_arr % b)
        np.testing.assert_array_equal((a // b).mapped_arr, a.mapped_arr // b)
        np.testing.assert_array_equal((a / b).mapped_arr, a.mapped_arr / b)
        # __r*__ is only called if the left object does not have an __*__ method
        np.testing.assert_array_equal((10 + a).mapped_arr, 10 + a.mapped_arr)
        np.testing.assert_array_equal((10 - a).mapped_arr, 10 - a.mapped_arr)
        np.testing.assert_array_equal((10 * a).mapped_arr, 10 * a.mapped_arr)
        np.testing.assert_array_equal((10 ** a).mapped_arr, 10 ** a.mapped_arr)
        np.testing.assert_array_equal((10 % a).mapped_arr, 10 % a.mapped_arr)
        np.testing.assert_array_equal((10 // a).mapped_arr, 10 // a.mapped_arr)
        np.testing.assert_array_equal((10 / a).mapped_arr, 10 / a.mapped_arr)
        # mask ops
        np.testing.assert_array_equal((a_bool & b_bool).mapped_arr, a_bool.mapped_arr & b_bool)
        np.testing.assert_array_equal((a_bool | b_bool).mapped_arr, a_bool.mapped_arr | b_bool)
        np.testing.assert_array_equal((a_bool ^ b_bool).mapped_arr, a_bool.mapped_arr ^ b_bool)
        np.testing.assert_array_equal((True & a_bool).mapped_arr, True & a_bool.mapped_arr)
        np.testing.assert_array_equal((True | a_bool).mapped_arr, True | a_bool.mapped_arr)
        np.testing.assert_array_equal((True ^ a_bool).mapped_arr, True ^ a_bool.mapped_arr)
        # unary ops
        np.testing.assert_array_equal((-a).mapped_arr, -a.mapped_arr)
        np.testing.assert_array_equal((+a).mapped_arr, +a.mapped_arr)
        np.testing.assert_array_equal((abs(-a)).mapped_arr, abs((-a.mapped_arr)))


class TestRecords:
    def test_records(self):
        pd.testing.assert_frame_equal(
            records.records,
            pd.DataFrame.from_records(records_arr)
        )

    def test_recarray(self):
        np.testing.assert_array_equal(records.recarray.some_field1, records.records_arr['some_field1'])
        np.testing.assert_array_equal(records['a'].recarray.some_field1, records['a'].records_arr['some_field1'])

    def test_col_index(self):
        target = np.array([
            [0, 3],
            [3, 6],
            [6, 9],
            [-1, -1]
        ])
        np.testing.assert_array_equal(
            records.col_index,
            target
        )
        np.testing.assert_array_equal(
            records['a'].col_index,
            target[0:1]
        )

    def test_filter_by_mask(self):
        mask = records.records_arr['some_field1'] >= records.records_arr['some_field1'].mean()
        filtered = records.filter_by_mask(mask)
        record_arrays_close(
            records.records_arr,
            np.array([
                (0, 0, 10., 21.),
                (0, 1, 11., 22.),
                (0, 2, 12., 23.),
                (1, 0, 13., 24.),
                (1, 1, 14., 25.),
                (1, 2, 15., 26.),
                (2, 0, 16., 27.),
                (2, 1, 17., 28.),
                (2, 2, 18., 29.)
            ], dtype=example_dt)
        )
        mask_a = records['a'].records_arr['some_field1'] >= records['a'].records_arr['some_field1'].mean()
        record_arrays_close(
            records['a'].filter_by_mask(mask_a).records_arr,
            np.array([
                (0, 1, 11., 22.),
                (0, 2, 12., 23.)
            ], dtype=example_dt)
        )

    def test_map_field(self):
        np.testing.assert_array_equal(
            records.map_field('some_field1').mapped_arr,
            np.array([10., 11., 12., 13., 14., 15., 16., 17., 18.])
        )
        np.testing.assert_array_equal(
            records['a'].map_field('some_field1').mapped_arr,
            np.array([10., 11., 12.])
        )
        np.testing.assert_array_equal(
            records.map_field('some_field1', idx_arr=records_arr['idx'] * 2).idx_arr,
            records_arr['idx'] * 2
        )

    def test_map(self):
        @njit
        def map_func_nb(record):
            return record['some_field1'] + record['some_field2']

        np.testing.assert_array_equal(
            records.map(map_func_nb).mapped_arr,
            np.array([31., 33., 35., 37., 39., 41., 43., 45., 47.])
        )
        np.testing.assert_array_equal(
            records['a'].map(map_func_nb).mapped_arr,
            np.array([31., 33., 35.])
        )
        np.testing.assert_array_equal(
            records.map(map_func_nb, idx_arr=records_arr['idx'] * 2).idx_arr,
            records_arr['idx'] * 2
        )

    def test_map_array(self):
        arr = records_arr['some_field1'] + records_arr['some_field2']
        np.testing.assert_array_equal(
            records.map_array(arr).mapped_arr,
            np.array([31., 33., 35., 37., 39., 41., 43., 45., 47.])
        )
        np.testing.assert_array_equal(
            records['a'].map_array(arr[:3]).mapped_arr,
            np.array([31., 33., 35.])
        )
        np.testing.assert_array_equal(
            records.map_array(arr, idx_arr=records_arr['idx'] * 2).idx_arr,
            records_arr['idx'] * 2
        )

    def test_count(self):
        target = pd.Series(
            np.array([3, 3, 3, 0]),
            index=wrapper.columns
        )
        pd.testing.assert_series_equal(
            records.count,
            target
        )
        assert records['a'].count == target['a']

    def test_indexing(self):
        record_arrays_close(
            records['a'].records_arr,
            np.array([
                (0, 0, 10., 21.),
                (0, 1, 11., 22.),
                (0, 2, 12., 23.)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records['a'].wrapper.columns,
            pd.Index(['a'], dtype='object')
        )
        record_arrays_close(
            records[['a', 'b']].records_arr,
            np.array([
                (0, 0, 10., 21.),
                (0, 1, 11., 22.),
                (0, 2, 12., 23.),
                (1, 0, 13., 24.),
                (1, 1, 14., 25.),
                (1, 2, 15., 26.)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records[['a', 'a']].wrapper.columns,
            pd.Index(['a', 'a'], dtype='object')
        )
        record_arrays_close(
            records[['a', 'a']].records_arr,
            np.array([
                (0, 0, 10., 21.),
                (0, 1, 11., 22.),
                (0, 2, 12., 23.),
                (1, 0, 10., 21.),
                (1, 1, 11., 22.),
                (1, 2, 12., 23.)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records[['a', 'a']].wrapper.columns,
            pd.Index(['a', 'a'], dtype='object')
        )
        try:
            _ = records.iloc[::2, :]  # changing time not supported
            raise Exception
        except:
            pass
        _ = records.iloc[np.arange(records.wrapper.shape[0]), :]  # won't change time

    def test_filtering(self):
        filtered_records = vbt.Records(records_arr[[0, -1]], wrapper)
        record_arrays_close(
            filtered_records.records_arr,
            np.array([(0, 0, 10., 21.), (2, 2, 18., 29.)], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records.col_index,
            np.array([
                [0, 1],
                [-1, -1],
                [1, 2],
                [-1, -1]
            ])
        )
        # a
        record_arrays_close(
            filtered_records['a'].records_arr,
            np.array([(0, 0, 10., 21.)], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['a'].col_index,
            np.array([[0, 1]])
        )
        np.testing.assert_array_equal(
            filtered_records['a'].map_field('some_field1').mapped_arr,
            np.array([10.])
        )
        assert filtered_records['a'].map_field('some_field1').min() == 10.
        assert filtered_records['a'].count == 1.
        # b
        record_arrays_close(
            filtered_records['b'].records_arr,
            np.array([], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['b'].col_index,
            np.array([[-1, -1]])
        )
        np.testing.assert_array_equal(
            filtered_records['b'].map_field('some_field1').mapped_arr,
            np.array([])
        )
        assert filtered_records['b'].count == 0.
        assert np.isnan(filtered_records['b'].map_field('some_field1').min())
        # c
        record_arrays_close(
            filtered_records['c'].records_arr,
            np.array([(0, 2, 18.0, 29.0)], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['c'].col_index,
            np.array([[0, 1]])
        )
        np.testing.assert_array_equal(
            filtered_records['c'].map_field('some_field1').mapped_arr,
            np.array([18.])
        )
        assert filtered_records['c'].count == 1.
        assert filtered_records['c'].map_field('some_field1').min() == 18.
        # d
        record_arrays_close(
            filtered_records['d'].records_arr,
            np.array([], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['d'].col_index,
            np.array([[-1, -1]])
        )
        np.testing.assert_array_equal(
            filtered_records['d'].map_field('some_field1').mapped_arr,
            np.array([])
        )
        assert filtered_records['d'].count == 0.
        assert np.isnan(filtered_records['d'].map_field('some_field1').min())


# ############# drawdowns.py ############# #


ts = pd.DataFrame({
    'a': [2, 1, 3, 1, 4, 1],
    'b': [1, 2, 1, 3, 1, 4],
    'c': [1, 2, 3, 2, 1, 2],
    'd': [1, 2, 3, 4, 5, 6]
})

drawdowns = vbt.Drawdowns.from_ts(ts, freq='1 days')


class TestDrawdowns:
    def test_from_ts(self):
        drawdowns = vbt.Drawdowns.from_ts(ts, freq='1 days', idx_field='start_idx')
        record_arrays_close(
            drawdowns.records_arr,
            np.array([
                (0, 0, 1, 2, 1),
                (0, 2, 3, 4, 1),
                (0, 4, 5, 5, 0),
                (1, 1, 2, 3, 1),
                (1, 3, 4, 5, 1),
                (2, 2, 4, 5, 0)
            ], dtype=drawdown_dt)
        )
        pd.testing.assert_frame_equal(drawdowns.ts, ts)
        assert drawdowns.wrapper.freq == day_dt
        assert drawdowns.idx_field == 'start_idx'

    def test_filter_by_mask(self):
        mask = drawdowns.records_arr['col'] > 0
        filtered = drawdowns.filter_by_mask(mask)
        record_arrays_close(
            filtered.records_arr,
            np.array([
                (1, 1, 2, 3, 1),
                (1, 3, 4, 5, 1),
                (2, 2, 4, 5, 0)
            ], dtype=drawdown_dt)
        )
        mask_a = drawdowns['a'].records_arr['col'] > 0
        record_arrays_close(
            drawdowns['a'].filter_by_mask(mask_a).records_arr,
            np.array([], dtype=drawdown_dt)
        )

    def test_start_value(self):
        np.testing.assert_array_equal(
            drawdowns.start_value.mapped_arr,
            np.array([2., 3., 4., 2., 3., 3.])
        )
        np.testing.assert_array_equal(
            drawdowns['a'].start_value.mapped_arr,
            np.array([2., 3., 4.])
        )

    def test_valley_value(self):
        np.testing.assert_array_equal(
            drawdowns.valley_value.mapped_arr,
            np.array([1., 1., 1., 1., 1., 1.])
        )
        np.testing.assert_array_equal(
            drawdowns['a'].valley_value.mapped_arr,
            np.array([1., 1., 1.])
        )

    def test_end_value(self):
        np.testing.assert_array_equal(
            drawdowns.end_value.mapped_arr,
            np.array([3., 4., 1., 3., 4., 2.])
        )
        np.testing.assert_array_equal(
            drawdowns['a'].end_value.mapped_arr,
            np.array([3., 4., 1.])
        )

    def test_drawdown(self):
        np.testing.assert_array_almost_equal(
            drawdowns.drawdown.mapped_arr,
            np.array([-0.5, -0.66666667, -0.75, -0.5, -0.66666667, -0.66666667])
        )
        np.testing.assert_array_almost_equal(
            drawdowns['a'].drawdown.mapped_arr,
            np.array([-0.5, -0.66666667, -0.75])
        )
        pd.testing.assert_frame_equal(
            drawdowns.drawdown.to_matrix(),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, np.nan],
                    [np.nan, np.nan, np.nan, np.nan],
                    [-0.5, np.nan, np.nan, np.nan],
                    [np.nan, -0.5, np.nan, np.nan],
                    [-0.66666669, np.nan, np.nan, np.nan],
                    [-0.75, -0.66666669, -0.66666669, np.nan]
                ]),
                index=ts.index,
                columns=ts.columns
            )
        )

    def test_avg_drawdown(self):
        pd.testing.assert_series_equal(
            drawdowns.avg_drawdown,
            pd.Series(np.array([-0.63888889, -0.58333333, -0.66666667, 0.]), index=wrapper.columns)
        )
        assert drawdowns['a'].avg_drawdown == -0.6388888888888888

    def test_max_drawdown(self):
        pd.testing.assert_series_equal(
            drawdowns.max_drawdown,
            pd.Series(np.array([-0.75, -0.66666667, -0.66666667, 0.]), index=wrapper.columns)
        )
        assert drawdowns['a'].max_drawdown == -0.75

    def test_duration(self):
        np.testing.assert_array_almost_equal(
            drawdowns.duration.mapped_arr,
            np.array([2., 2., 1., 2., 2., 3.])
        )
        np.testing.assert_array_almost_equal(
            drawdowns['a'].duration.mapped_arr,
            np.array([2., 2., 1.])
        )

    def test_avg_duration(self):
        pd.testing.assert_series_equal(
            drawdowns.avg_duration,
            pd.Series(
                np.array([144000000000000, 172800000000000, 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            )
        )
        assert drawdowns['a'].avg_duration == np.timedelta64(144000000000000)

    def test_max_duration(self):
        pd.testing.assert_series_equal(
            drawdowns.max_duration,
            pd.Series(
                np.array([172800000000000, 172800000000000, 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            )
        )
        assert drawdowns['a'].max_duration == np.timedelta64(172800000000000)

    def test_coverage(self):
        pd.testing.assert_series_equal(
            drawdowns.coverage,
            pd.Series(np.array([0.83333333, 0.66666667, 0.5, 0.]), index=ts.columns)
        )
        assert drawdowns['a'].coverage == 0.8333333333333334

    def test_ptv_duration(self):
        np.testing.assert_array_almost_equal(
            drawdowns.ptv_duration.mapped_arr,
            np.array([1., 1., 1., 1., 1., 2.])
        )
        np.testing.assert_array_almost_equal(
            drawdowns['a'].ptv_duration.mapped_arr,
            np.array([1., 1., 1.])
        )

    def test_status(self):
        np.testing.assert_array_almost_equal(
            drawdowns.status.mapped_arr,
            np.array([1, 1, 0, 1, 1, 0])
        )
        np.testing.assert_array_almost_equal(
            drawdowns['a'].status.mapped_arr,
            np.array([1, 1, 0])
        )

    def test_recovered_rate(self):
        pd.testing.assert_series_equal(
            drawdowns.recovered_rate,
            pd.Series(np.array([0.66666667, 1., 0., np.nan]), index=ts.columns)
        )
        assert drawdowns['a'].recovered_rate == 0.6666666666666666

    def test_active_records(self):
        assert isinstance(drawdowns.active, ActiveDrawdowns)
        assert drawdowns.active.idx_field == drawdowns.idx_field
        assert drawdowns.active.wrapper.freq == drawdowns.wrapper.freq
        record_arrays_close(
            drawdowns.active.records_arr,
            np.array([
                (0, 4, 5, 5, 0),
                (2, 2, 4, 5, 0)
            ], dtype=drawdown_dt)
        )
        record_arrays_close(
            drawdowns['a'].active.records_arr,
            np.array([
                (0, 4, 5, 5, 0)
            ], dtype=drawdown_dt)
        )
        record_arrays_close(
            drawdowns.active['a'].records_arr,
            np.array([
                (0, 4, 5, 5, 0)
            ], dtype=drawdown_dt)
        )

    def test_current_drawdown(self):
        pd.testing.assert_series_equal(
            drawdowns.active.current_drawdown,
            pd.Series(np.array([-0.75, np.nan, -0.66666667, np.nan]), index=wrapper.columns)
        )
        assert drawdowns['a'].active.current_drawdown == -0.75

    def test_current_duration(self):
        pd.testing.assert_series_equal(
            drawdowns.active.current_duration,
            pd.Series(
                np.array([86400000000000, 'NaT', 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            )
        )
        assert drawdowns['a'].active.current_duration == np.timedelta64(86400000000000)

    def test_current_return(self):
        pd.testing.assert_series_equal(
            drawdowns.active.current_return,
            pd.Series(np.array([0., np.nan, 1., np.nan]), index=wrapper.columns)
        )
        assert drawdowns['a'].active.current_return == 0.

    def test_recovered_records(self):
        assert isinstance(drawdowns.recovered, RecoveredDrawdowns)
        assert drawdowns.recovered.idx_field == drawdowns.idx_field
        assert drawdowns.recovered.wrapper.freq == drawdowns.wrapper.freq
        record_arrays_close(
            drawdowns.recovered.records_arr,
            np.array([
                (0, 0, 1, 2, 1),
                (0, 2, 3, 4, 1),
                (1, 1, 2, 3, 1),
                (1, 3, 4, 5, 1)
            ], dtype=drawdown_dt)
        )
        record_arrays_close(
            drawdowns['a'].recovered.records_arr,
            np.array([
                (0, 0, 1, 2, 1),
                (0, 2, 3, 4, 1)
            ], dtype=drawdown_dt)
        )
        record_arrays_close(
            drawdowns.recovered['a'].records_arr,
            np.array([
                (0, 0, 1, 2, 1),
                (0, 2, 3, 4, 1)
            ], dtype=drawdown_dt)
        )

    def test_recovery_return(self):
        np.testing.assert_array_almost_equal(
            drawdowns.recovered.recovery_return.mapped_arr,
            np.array([2., 3., 2., 3.])
        )
        np.testing.assert_array_almost_equal(
            drawdowns['a'].recovered.recovery_return.mapped_arr,
            np.array([2., 3.])
        )

    def test_vtr_duration(self):
        np.testing.assert_array_almost_equal(
            drawdowns.recovered.vtr_duration.mapped_arr,
            np.array([1., 1., 1., 1.])
        )
        np.testing.assert_array_almost_equal(
            drawdowns['a'].recovered.vtr_duration.mapped_arr,
            np.array([1., 1.])
        )

    def test_vtr_duration_ratio(self):
        np.testing.assert_array_almost_equal(
            drawdowns.recovered.vtr_duration_ratio.mapped_arr,
            np.array([0.5, 0.5, 0.5, 0.5])
        )
        np.testing.assert_array_almost_equal(
            drawdowns['a'].recovered.vtr_duration_ratio.mapped_arr,
            np.array([0.5, 0.5])
        )


# ############# orders.py ############# #

order_records_arr = np.array([
    (0, 2, 33.00330033, 3., 0.99009901, 0),
    (0, 3, 33.00330033, 4., 1.32013201, 1),
    (0, 4, 25.8798157, 5., 1.29399079, 0),
    (0, 6, 25.8798157, 7., 1.8115871, 1),
    (1, 2, 14.14427157, 7., 0.99009901, 0),
    (1, 3, 14.14427157, 6., 0.84865629, 1),
    (1, 4, 16.63702438, 5., 0.83185122, 0),
    (1, 5, 16.63702438, 4., 0.66548098, 1),
    (2, 0, 99.00990099, 1., 0.99009901, 0),
    (2, 1, 99.00990099, 2., 1.98019802, 1),
    (2, 6, 194.09861778, 1., 1.94098618, 0),
    (3, 2, 49.5049505, 2., 0.99009901, 0),
    (3, 4, 49.5049505, 2., 0.99009901, 1),
    (3, 6, 24.26232722, 4., 0.97049309, 0)
], dtype=order_dt)

price = pd.DataFrame({
    'a': [1, 2, 3, 4, 5, 6, 7],
    'b': [9, 8, 7, 6, 5, 4, 3],
    'c': [1, 2, 3, 4, 3, 2, 1],
    'd': [4, 3, 2, 1, 2, 3, 4]
})

orders = vbt.Orders(order_records_arr, price, freq='1 days')


class TestOrders:
    def test_filter_by_mask(self):
        mask = orders.records_arr['col'] > 1
        filtered = orders.filter_by_mask(mask)
        record_arrays_close(
            filtered.records_arr,
            np.array([
                (2, 0, 99.00990099, 1., 0.99009901, 0),
                (2, 1, 99.00990099, 2., 1.98019802, 1),
                (2, 6, 194.09861778, 1., 1.94098618, 0),
                (3, 2, 49.5049505, 2., 0.99009901, 0),
                (3, 4, 49.5049505, 2., 0.99009901, 1),
                (3, 6, 24.26232722, 4., 0.97049309, 0)
            ], dtype=order_dt)
        )
        pd.testing.assert_frame_equal(filtered.main_price, orders.main_price)
        assert filtered.wrapper == orders.wrapper
        mask_a = orders['a'].records_arr['col'] > 1
        record_arrays_close(
            orders['a'].filter_by_mask(mask_a).records_arr,
            np.array([], dtype=order_dt)
        )

    def test_size(self):
        np.testing.assert_array_equal(
            orders.size.mapped_arr,
            np.array([
                33.00330033, 33.00330033, 25.8798157, 25.8798157, 14.14427157,
                14.14427157, 16.63702438, 16.63702438, 99.00990099, 99.00990099,
                194.09861778, 49.5049505, 49.5049505, 24.26232722
            ])
        )
        np.testing.assert_array_equal(
            orders['a'].size.mapped_arr,
            np.array([33.00330033, 33.00330033, 25.8798157, 25.8798157])
        )
        pd.testing.assert_frame_equal(
            orders.size.to_matrix(),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, 99.00990099, np.nan],
                    [np.nan, np.nan, 99.00990099, np.nan],
                    [33.00330033, 14.14427157, np.nan, 49.5049505],
                    [33.00330033, 14.14427157, np.nan, np.nan],
                    [25.8798157, 16.63702438, np.nan, 49.5049505],
                    [np.nan, 16.63702438, np.nan, np.nan],
                    [25.8798157, np.nan, 194.09861778, 24.26232722]
                ]),
                index=price.index,
                columns=price.columns
            )
        )

    def test_price(self):
        np.testing.assert_array_equal(
            orders.price.mapped_arr,
            np.array([3., 4., 5., 7., 7., 6., 5., 4., 1., 2., 1., 2., 2., 4.])
        )
        np.testing.assert_array_equal(
            orders['a'].price.mapped_arr,
            np.array([3., 4., 5., 7.])
        )

    def test_fees(self):
        np.testing.assert_array_equal(
            orders.fees.mapped_arr,
            np.array([
                0.99009901, 1.32013201, 1.29399079, 1.8115871, 0.99009901,
                0.84865629, 0.83185122, 0.66548098, 0.99009901, 1.98019802,
                1.94098618, 0.99009901, 0.99009901, 0.97049309
            ])
        )
        np.testing.assert_array_equal(
            orders['a'].fees.mapped_arr,
            np.array([0.99009901, 1.32013201, 1.29399079, 1.8115871])
        )

    def test_side(self):
        np.testing.assert_array_equal(
            orders.side.mapped_arr,
            np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
        )
        np.testing.assert_array_equal(
            orders['a'].side.mapped_arr,
            np.array([0, 1, 0, 1])
        )

    def test_buy_records(self):
        assert isinstance(orders.buy, BaseOrders)
        assert orders.buy.idx_field == orders.idx_field
        assert orders.buy.wrapper.freq == orders.wrapper.freq
        record_arrays_close(
            orders.buy.records_arr,
            np.array([
                (0, 2, 33.00330033, 3., 0.99009901, 0),
                (0, 4, 25.8798157, 5., 1.29399079, 0),
                (1, 2, 14.14427157, 7., 0.99009901, 0),
                (1, 4, 16.63702438, 5., 0.83185122, 0),
                (2, 0, 99.00990099, 1., 0.99009901, 0),
                (2, 6, 194.09861778, 1., 1.94098618, 0),
                (3, 2, 49.5049505, 2., 0.99009901, 0),
                (3, 6, 24.26232722, 4., 0.97049309, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            orders['a'].buy.records_arr,
            np.array([
                (0, 2, 33.00330033, 3., 0.99009901, 0),
                (0, 4, 25.8798157, 5., 1.29399079, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            orders.buy['a'].records_arr,
            np.array([
                (0, 2, 33.00330033, 3., 0.99009901, 0),
                (0, 4, 25.8798157, 5., 1.29399079, 0)
            ], dtype=order_dt)
        )

    def test_sell_records(self):
        assert isinstance(orders.sell, BaseOrders)
        assert orders.sell.idx_field == orders.idx_field
        assert orders.sell.wrapper.freq == orders.wrapper.freq
        record_arrays_close(
            orders.sell.records_arr,
            np.array([
                (0, 3, 33.00330033, 4., 1.32013201, 1),
                (0, 6, 25.8798157, 7., 1.8115871, 1),
                (1, 3, 14.14427157, 6., 0.84865629, 1),
                (1, 5, 16.63702438, 4., 0.66548098, 1),
                (2, 1, 99.00990099, 2., 1.98019802, 1),
                (3, 4, 49.5049505, 2., 0.99009901, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            orders['a'].sell.records_arr,
            np.array([
                (0, 3, 33.00330033, 4., 1.32013201, 1),
                (0, 6, 25.8798157, 7., 1.8115871, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            orders.sell['a'].records_arr,
            np.array([
                (0, 3, 33.00330033, 4., 1.32013201, 1),
                (0, 6, 25.8798157, 7., 1.8115871, 1)
            ], dtype=order_dt)
        )


# ############# events.py ############# #

event_records_arr = np.array([
    (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
    (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1),
    (1, 14.14427157, 2, 7., 0.99009901, 3, 6., 0.84865629, -15.98302687, -0.15983027, 1),
    (1, 16.63702438, 4, 5., 0.83185122, 5, 4., 0.66548098, -18.13435658, -0.21584158, 1),
    (2, 99.00990099, 0, 1., 0.99009901, 1, 2., 1.98019802, 96.03960396, 0.96039604, 1),
    (2, 194.09861778, 6, 1., 1.94098618, 6, 1., 0., -1.94098618, -0.00990099, 0),
    (3, 49.5049505, 2, 2., 0.99009901, 4, 2., 0.99009901, -1.98019802, -0.01980198, 1),
    (3, 24.26232722, 6, 4., 0.97049309, 6, 4., 0., -0.97049309, -0.00990099, 0)
], dtype=event_dt)

events = vbt.Events(event_records_arr, price, freq='1 days')


class TestEvents:
    def test_filter_by_mask(self):
        mask = events.records_arr['col'] > 1
        filtered = events.filter_by_mask(mask)
        record_arrays_close(
            filtered.records_arr,
            np.array([
                (2, 99.00990099, 0, 1., 0.99009901, 1, 2., 1.98019802, 96.03960396, 0.96039604, 1),
                (2, 194.09861778, 6, 1., 1.94098618, 6, 1., 0., -1.94098618, -0.00990099, 0),
                (3, 49.5049505, 2, 2., 0.99009901, 4, 2., 0.99009901, -1.98019802, -0.01980198, 1),
                (3, 24.26232722, 6, 4., 0.97049309, 6, 4., 0., -0.97049309, -0.00990099, 0)
            ], dtype=event_dt)
        )
        pd.testing.assert_frame_equal(filtered.main_price, events.main_price)
        assert filtered.wrapper == events.wrapper
        mask_a = events['a'].records_arr['col'] > 1
        record_arrays_close(
            events['a'].filter_by_mask(mask_a).records_arr,
            np.array([], dtype=event_dt)
        )

    def test_duration(self):
        np.testing.assert_array_almost_equal(
            events.duration.mapped_arr,
            np.array([1., 2., 1., 1., 1., 0., 2., 0.])
        )
        np.testing.assert_array_almost_equal(
            events['a'].duration.mapped_arr,
            np.array([1., 2.])
        )

    def test_coverage(self):
        pd.testing.assert_series_equal(
            events.coverage,
            pd.Series(np.array([0.42857143, 0.28571429, 0.14285714, 0.28571429]), index=ts.columns)
        )
        assert events['a'].coverage == 0.42857142857142855

    def test_pnl(self):
        np.testing.assert_array_almost_equal(
            events.pnl.mapped_arr,
            np.array([
                30.69306931, 48.65405351, -15.98302687, -18.13435658,
                96.03960396, -1.94098618, -1.98019802, -0.97049309
            ])
        )
        np.testing.assert_array_almost_equal(
            events['a'].pnl.mapped_arr,
            np.array([30.69306931, 48.65405351])
        )

    def test_returns(self):
        np.testing.assert_array_almost_equal(
            events.returns.mapped_arr,
            np.array([
                0.30693069, 0.37227723, -0.15983027, -0.21584158, 0.96039604,
                -0.00990099, -0.01980198, -0.00990099
            ])
        )
        np.testing.assert_array_almost_equal(
            events['a'].returns.mapped_arr,
            np.array([0.30693069, 0.37227723])
        )

    def test_winning_records(self):
        assert isinstance(events.winning, BaseEvents)
        assert events.winning.idx_field == events.idx_field
        assert events.winning.wrapper.freq == events.wrapper.freq
        record_arrays_close(
            events.winning.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1),
                (2, 99.00990099, 0, 1., 0.99009901, 1, 2., 1.98019802, 96.03960396, 0.96039604, 1)
            ], dtype=event_dt)
        )
        record_arrays_close(
            events['a'].winning.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=event_dt)
        )
        record_arrays_close(
            events.winning['a'].records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=event_dt)
        )

    def test_losing_records(self):
        assert isinstance(events.losing, BaseEvents)
        assert events.losing.idx_field == events.idx_field
        assert events.losing.wrapper.freq == events.wrapper.freq
        record_arrays_close(
            events.losing.records_arr,
            np.array([
                (1, 14.14427157, 2, 7., 0.99009901, 3, 6., 0.84865629, -15.98302687, -0.15983027, 1),
                (1, 16.63702438, 4, 5., 0.83185122, 5, 4., 0.66548098, -18.13435658, -0.21584158, 1),
                (2, 194.09861778, 6, 1., 1.94098618, 6, 1., 0., -1.94098618, -0.00990099, 0),
                (3, 49.5049505, 2, 2., 0.99009901, 4, 2., 0.99009901, -1.98019802, -0.01980198, 1),
                (3, 24.26232722, 6, 4., 0.97049309, 6, 4., 0., -0.97049309, -0.00990099, 0)
            ], dtype=event_dt)
        )
        record_arrays_close(
            events['a'].losing.records_arr,
            np.array([], dtype=event_dt)
        )
        record_arrays_close(
            events.losing['a'].records_arr,
            np.array([], dtype=event_dt)
        )

    def test_win_rate(self):
        pd.testing.assert_series_equal(
            events.win_rate,
            pd.Series(np.array([1., 0., 0.5, 0.]), index=ts.columns)
        )
        assert events['a'].win_rate == 1.

    def test_profit_factor(self):
        pd.testing.assert_series_equal(
            events.profit_factor,
            pd.Series(np.array([np.inf, 0., 49.47979792, 0.]), index=ts.columns)
        )
        assert np.isinf(events['a'].profit_factor)

    def test_expectancy(self):
        pd.testing.assert_series_equal(
            events.expectancy,
            pd.Series(np.array([39.67356141, -17.05869172, 47.04930889, -1.47534555]), index=ts.columns)
        )
        assert events['a'].expectancy == 39.67356141

    def test_sqn(self):
        pd.testing.assert_series_equal(
            events.sqn,
            pd.Series(np.array([4.41774916, -15.85874229, 0.96038019, -2.9223301]), index=ts.columns)
        )
        assert events['a'].sqn == 4.4177491576436

    def test_status(self):
        np.testing.assert_array_almost_equal(
            events.status.mapped_arr,
            np.array([1, 1, 1, 1, 1, 0, 1, 0])
        )
        np.testing.assert_array_almost_equal(
            events['a'].status.mapped_arr,
            np.array([1, 1])
        )

    def test_closed_rate(self):
        pd.testing.assert_series_equal(
            events.closed_rate,
            pd.Series(np.array([1., 1., 0.5, 0.5]), index=ts.columns)
        )
        assert events['a'].closed_rate == 1.0

    def test_open_records(self):
        assert isinstance(events.open, BaseEventsByResult)
        assert events.open.idx_field == events.idx_field
        assert events.open.wrapper.freq == events.wrapper.freq
        record_arrays_close(
            events.open.records_arr,
            np.array([
                (2, 194.09861778, 6, 1., 1.94098618, 6, 1., 0., -1.94098618, -0.00990099, 0),
                (3, 24.26232722, 6, 4., 0.97049309, 6, 4., 0., -0.97049309, -0.00990099, 0)
            ], dtype=event_dt)
        )
        record_arrays_close(
            events['a'].open.records_arr,
            np.array([], dtype=event_dt)
        )
        record_arrays_close(
            events.open['a'].records_arr,
            np.array([], dtype=event_dt)
        )

    def test_closed_records(self):
        assert isinstance(events.closed, BaseEventsByResult)
        assert events.closed.idx_field == events.idx_field
        assert events.closed.wrapper.freq == events.wrapper.freq
        record_arrays_close(
            events.closed.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1),
                (1, 14.14427157, 2, 7., 0.99009901, 3, 6., 0.84865629, -15.98302687, -0.15983027, 1),
                (1, 16.63702438, 4, 5., 0.83185122, 5, 4., 0.66548098, -18.13435658, -0.21584158, 1),
                (2, 99.00990099, 0, 1., 0.99009901, 1, 2., 1.98019802, 96.03960396, 0.96039604, 1),
                (3, 49.5049505, 2, 2., 0.99009901, 4, 2., 0.99009901, -1.98019802, -0.01980198, 1)
            ], dtype=event_dt)
        )
        record_arrays_close(
            events['a'].closed.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=event_dt)
        )
        record_arrays_close(
            events.closed['a'].records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=event_dt)
        )


trades = vbt.Trades.from_orders(orders)


class TestTrades:
    def test_from_orders(self):
        trades = vbt.Trades.from_orders(orders, idx_field='entry_idx')
        record_arrays_close(
            trades.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1, 0),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1, 1),
                (1, 14.14427157, 2, 7., 0.99009901, 3, 6., 0.84865629, -15.98302687, -0.15983027, 1, 2),
                (1, 16.63702438, 4, 5., 0.83185122, 5, 4., 0.66548098, -18.13435658, -0.21584158, 1, 3),
                (2, 99.00990099, 0, 1., 0.99009901, 1, 2., 1.98019802, 96.03960396, 0.96039604, 1, 4),
                (2, 194.09861778, 6, 1., 1.94098618, 6, 1., 0., -1.94098618, -0.00990099, 0, 5),
                (3, 49.5049505, 2, 2., 0.99009901, 4, 2., 0.99009901, -1.98019802, -0.01980198, 1, 6),
                (3, 24.26232722, 6, 4., 0.97049309, 6, 4., 0., -0.97049309, -0.00990099, 0, 7)
            ], dtype=trade_dt)
        )
        pd.testing.assert_frame_equal(trades.main_price, price)
        assert trades.wrapper.freq == day_dt
        assert trades.idx_field == 'entry_idx'

    def test_position_idx(self):
        np.testing.assert_array_almost_equal(
            trades.position_idx.mapped_arr,
            np.array([0, 1, 2, 3, 4, 5, 6, 7])
        )
        np.testing.assert_array_almost_equal(
            trades['a'].position_idx.mapped_arr,
            np.array([0, 1])
        )


positions = vbt.Positions.from_orders(orders)


class TestPositions:
    def test_from_orders(self):
        positions = vbt.Positions.from_orders(orders, idx_field='entry_idx')
        record_arrays_close(
            positions.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1),
                (1, 14.14427157, 2, 7., 0.99009901, 3, 6., 0.84865629, -15.98302687, -0.15983027, 1),
                (1, 16.63702438, 4, 5., 0.83185122, 5, 4., 0.66548098, -18.13435658, -0.21584158, 1),
                (2, 99.00990099, 0, 1., 0.99009901, 1, 2., 1.98019802, 96.03960396, 0.96039604, 1),
                (2, 194.09861778, 6, 1., 1.94098618, 6, 4., 0., 580.35486716, 2.96039604, 0),
                (3, 49.5049505, 2, 2., 0.99009901, 4, 2., 0.99009901, -1.98019802, -0.01980198, 1),
                (3, 24.26232722, 6, 4., 0.97049309, 6, 4., 0., -0.97049309, -0.00990099, 0)
            ], dtype=position_dt)
        )
        pd.testing.assert_frame_equal(positions.main_price, price)
        assert positions.wrapper.freq == day_dt
        assert positions.idx_field == 'entry_idx'