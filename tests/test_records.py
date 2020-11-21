import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
import pytest

from vectorbt.base.array_wrapper import ArrayWrapper

from tests.utils import record_arrays_close

day_dt = np.timedelta64(86400000000000)

# ############# base.py ############# #

example_dt = np.dtype([
    ('col', np.int64),
    ('idx', np.int64),
    ('some_field1', np.float64),
    ('some_field2', np.float64)
], align=True)

records_arr = np.asarray([
    (0, 0, 10, 21),
    (0, 1, 11, 20),
    (0, 2, 12, 19),
    (1, 0, 13, 18),
    (1, 1, 14, 17),
    (1, 2, 13, 18),
    (2, 0, 12, 19),
    (2, 1, 11, 20),
    (2, 2, 10, 21)
], dtype=example_dt)

group_by = pd.Index([0, 1, 1, 1])

wrapper = ArrayWrapper(
    index=['x', 'y', 'z'],
    columns=['a', 'b', 'c', 'd'],
    ndim=2,
    freq='1 days'
)
wrapper_grouped = wrapper.copy(group_by=group_by)

records = vbt.records.Records(wrapper, records_arr)
records_grouped = vbt.records.Records(wrapper_grouped, records_arr)

mapped_array = records.map_field('some_field1')
mapped_array_grouped = records_grouped.map_field('some_field1')


class TestMappedArray:
    def test_regroup(self):
        pd.testing.assert_index_equal(
            mapped_array.regroup(group_by=group_by).wrapper.grouper.group_by,
            mapped_array_grouped.wrapper.grouper.group_by
        )

    def test_mapped_arr(self):
        np.testing.assert_array_equal(
            mapped_array.mapped_arr,
            np.array([10., 11., 12., 13., 14., 13., 12., 11., 10.])
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
            np.array([12., 13., 14., 13., 12.])
        )
        np.testing.assert_array_equal(filtered.col_arr, mapped_array.col_arr[mask])
        np.testing.assert_array_equal(filtered.idx_arr, mapped_array.idx_arr[mask])
        mask_a = mapped_array['a'].mapped_arr >= mapped_array['a'].mapped_arr.mean()
        np.testing.assert_array_equal(
            mapped_array['a'].filter_by_mask(mask_a).mapped_arr,
            np.array([11., 12.])
        )
        assert mapped_array_grouped.filter_by_mask(mask).wrapper == mapped_array_grouped.wrapper
        assert mapped_array_grouped.filter_by_mask(mask, group_by=False).wrapper.grouper.group_by is None

    def test_to_matrix(self):
        target = pd.DataFrame(
            np.array([
                [10., 13., 12., np.nan],
                [11., 14., 11., np.nan],
                [12., 13., 10., np.nan]
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
            wrapper,
            records_arr['some_field1'].tolist() + [1],
            records_arr['col'].tolist() + [2],
            idx_arr=records_arr['idx'].tolist() + [2]
        )
        with pytest.raises(Exception) as e_info:
            _ = mapped_array2.to_matrix()

    def test_reduce(self):
        @njit
        def mean_reduce_nb(col, a):
            return np.mean(a)

        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb),
            pd.Series(np.array([11., 13.333333333333334, 11., np.nan]), index=wrapper.columns)
        )
        assert mapped_array['a'].reduce(mean_reduce_nb) == 11.
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, default_val=0.),
            pd.Series(np.array([11., 13.333333333333334, 11., 0.]), index=wrapper.columns)
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, default_val=0., dtype=np.int_),
            pd.Series(np.array([11, 13, 11, 0]), index=wrapper.columns)
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, time_units=True),
            pd.Series(np.array([11., 13.333333333333334, 11., np.nan]), index=wrapper.columns) * day_dt
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.reduce(mean_reduce_nb),
            pd.Series([11., 12.166666666666666], index=pd.Int64Index([0, 1], dtype='int64'))
        )
        assert mapped_array_grouped[0].reduce(mean_reduce_nb) == 11.
        pd.testing.assert_series_equal(
            mapped_array_grouped[[0]].reduce(mean_reduce_nb),
            pd.Series([11.], index=pd.Int64Index([0], dtype='int64'))
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb),
            mapped_array_grouped.reduce(mean_reduce_nb, group_by=False)
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, group_by=group_by),
            mapped_array_grouped.reduce(mean_reduce_nb)
        )

    def test_reduce_to_idx(self):
        @njit
        def argmin_reduce_nb(col, a):
            return np.argmin(a)

        pd.testing.assert_series_equal(
            mapped_array.reduce(argmin_reduce_nb, to_idx=True),
            pd.Series(np.array(['x', 'x', 'z', np.nan], dtype=np.object), index=wrapper.columns)
        )
        assert mapped_array['a'].reduce(argmin_reduce_nb, to_idx=True) == 'x'
        pd.testing.assert_series_equal(
            mapped_array.reduce(argmin_reduce_nb, to_idx=True, idx_labeled=False),
            pd.Series(np.array([0, 0, 2, -1], dtype=int), index=wrapper.columns)
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.reduce(argmin_reduce_nb, to_idx=True, idx_labeled=False),
            pd.Series(np.array([0, 2], dtype=int), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_reduce_to_array(self):
        @njit
        def min_max_reduce_nb(col, a):
            return np.array([np.min(a), np.max(a)])

        with pytest.raises(Exception) as e_info:
            _ = mapped_array.reduce(min_max_reduce_nb, to_array=True)
        with pytest.raises(Exception) as e_info:
            _ = mapped_array.reduce(min_max_reduce_nb, to_array=True, n_rows=3)
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, n_rows=2),
            pd.DataFrame(
                np.array([
                    [10., 13., 10., np.nan],
                    [12., 14., 12., np.nan]
                ]),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, n_rows=2, index=['min', 'max']),
            pd.DataFrame(
                np.array([
                    [10., 13., 10., np.nan],
                    [12., 14., 12., np.nan]
                ]),
                index=pd.Index(['min', 'max'], dtype='object'),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_series_equal(
            mapped_array['a'].reduce(min_max_reduce_nb, to_array=True, n_rows=2, index=['min', 'max']),
            pd.Series([10., 12.], index=pd.Index(['min', 'max'], dtype='object'), name='a')
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, n_rows=2, default_val=0.),
            pd.DataFrame(
                np.array([
                    [10., 13., 10., 0.],
                    [12., 14., 12., 0.]
                ]),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, n_rows=2, time_units=True),
            pd.DataFrame(
                np.array([
                    [10., 13., 10., np.nan],
                    [12., 14., 12., np.nan]
                ]),
                columns=wrapper.columns
            ) * day_dt
        )
        pd.testing.assert_frame_equal(
            mapped_array_grouped.reduce(min_max_reduce_nb, to_array=True, n_rows=2),
            pd.DataFrame(
                np.array([
                    [10., 10.],
                    [12., 14.]
                ]),
                columns=pd.Int64Index([0, 1], dtype='int64')
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, n_rows=2),
            mapped_array_grouped.reduce(min_max_reduce_nb, to_array=True, n_rows=2, group_by=False)
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, n_rows=2, group_by=group_by),
            mapped_array_grouped.reduce(min_max_reduce_nb, to_array=True, n_rows=2)
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped[0].reduce(min_max_reduce_nb, to_array=True, n_rows=2),
            pd.Series([10., 12.])
        )
        pd.testing.assert_frame_equal(
            mapped_array_grouped[[0]].reduce(min_max_reduce_nb, to_array=True, n_rows=2),
            pd.DataFrame([[10.], [12.]], columns=pd.Int64Index([0], dtype='int64'))
        )

    def test_reduce_to_idx_array(self):
        @njit
        def idxmin_idxmax_reduce_nb(col, a):
            return np.array([np.argmin(a), np.argmax(a)])

        with pytest.raises(Exception) as e_info:
            _ = mapped_array.reduce(idxmin_idxmax_reduce_nb, to_array=True, to_idx=True)
        with pytest.raises(Exception) as e_info:
            _ = mapped_array.reduce(idxmin_idxmax_reduce_nb, to_array=True, n_rows=3, to_idx=True)
        pd.testing.assert_frame_equal(
            mapped_array.reduce(idxmin_idxmax_reduce_nb, to_array=True, n_rows=2, to_idx=True),
            pd.DataFrame(
                np.array([
                    ['x', 'x', 'z', np.nan],
                    ['z', 'y', 'x', np.nan]
                ], dtype=np.object),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb,
                to_array=True,
                n_rows=2,
                to_idx=True,
                index=['min', 'max']
            ),
            pd.DataFrame(
                np.array([
                    ['x', 'x', 'z', np.nan],
                    ['z', 'y', 'x', np.nan]
                ], dtype=np.object),
                index=pd.Index(['min', 'max'], dtype='object'),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_series_equal(
            mapped_array['a'].reduce(
                idxmin_idxmax_reduce_nb,
                to_array=True,
                n_rows=2,
                to_idx=True,
                index=['min', 'max']
            ),
            pd.Series(
                np.array(['x', 'z'], dtype=np.object),
                index=pd.Index(['min', 'max'], dtype='object'),
                name='a'
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb,
                to_array=True,
                n_rows=2,
                to_idx=True,
                idx_labeled=False
            ),
            pd.DataFrame(
                np.array([
                    [0, 0, 2, -1],
                    [2, 1, 0, -1]
                ]),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array_grouped.reduce(
                idxmin_idxmax_reduce_nb,
                to_array=True,
                n_rows=2,
                to_idx=True,
                idx_labeled=False
            ),
            pd.DataFrame(
                np.array([
                    [0, 2],
                    [2, 1]
                ]),
                columns=pd.Int64Index([0, 1], dtype='int64')
            )
        )

    def test_nst(self):
        pd.testing.assert_series_equal(
            mapped_array.nst(0),
            pd.Series(np.array([10., 13., 12., np.nan]), index=wrapper.columns)
        )
        assert mapped_array['a'].nst(0) == 10.
        pd.testing.assert_series_equal(
            mapped_array.nst(-1),
            pd.Series(np.array([12., 13., 10., np.nan]), index=wrapper.columns)
        )
        assert mapped_array['a'].nst(-1) == 12.
        with pytest.raises(Exception) as e_info:
            _ = mapped_array.nst(10)
        with pytest.raises(Exception) as e_info:
            _ = mapped_array_grouped.nst(0)
        pd.testing.assert_series_equal(
            mapped_array_grouped.nst(0, group_by=False),
            pd.Series(np.array([10., 13., 12., np.nan]), index=wrapper.columns)
        )

    def test_min(self):
        pd.testing.assert_series_equal(
            mapped_array.min(),
            mapped_array.to_matrix().min()
        )
        assert mapped_array['a'].min() == mapped_array['a'].to_matrix().min()
        pd.testing.assert_series_equal(
            mapped_array_grouped.min(),
            pd.Series([10., 10.], index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_max(self):
        pd.testing.assert_series_equal(
            mapped_array.max(),
            mapped_array.to_matrix().max()
        )
        assert mapped_array['a'].max() == mapped_array['a'].to_matrix().max()
        pd.testing.assert_series_equal(
            mapped_array_grouped.max(),
            pd.Series([12., 14.], index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_mean(self):
        pd.testing.assert_series_equal(
            mapped_array.mean(),
            mapped_array.to_matrix().mean()
        )
        assert mapped_array['a'].mean() == mapped_array['a'].to_matrix().mean()
        pd.testing.assert_series_equal(
            mapped_array_grouped.mean(),
            pd.Series([11., 12.166667], index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_median(self):
        pd.testing.assert_series_equal(
            mapped_array.median(),
            mapped_array.to_matrix().median()
        )
        assert mapped_array['a'].median() == mapped_array['a'].to_matrix().median()
        pd.testing.assert_series_equal(
            mapped_array_grouped.median(),
            pd.Series([11., 12.5], index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_std(self):
        pd.testing.assert_series_equal(
            mapped_array.std(),
            mapped_array.to_matrix().std()
        )
        assert mapped_array['a'].std() == mapped_array['a'].to_matrix().std()
        pd.testing.assert_series_equal(
            mapped_array.std(ddof=0),
            mapped_array.to_matrix().std(ddof=0)
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.std(),
            pd.Series([1.0, 1.4719601443879746], index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_sum(self):
        pd.testing.assert_series_equal(
            mapped_array.sum(),
            mapped_array.to_matrix().sum()
        )
        assert mapped_array['a'].sum() == mapped_array['a'].to_matrix().sum()
        pd.testing.assert_series_equal(
            mapped_array_grouped.sum(),
            pd.Series([33.0, 73.0], index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_count(self):
        pd.testing.assert_series_equal(
            mapped_array.count(),
            mapped_array.to_matrix().count()
        )
        assert mapped_array['a'].count() == mapped_array['a'].to_matrix().count()
        pd.testing.assert_series_equal(
            mapped_array_grouped.count(),
            pd.Series([3, 6], index=pd.Int64Index([0, 1], dtype='int64'))
        )

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
        pd.testing.assert_frame_equal(
            mapped_array_grouped.describe(),
            pd.DataFrame(
                np.array([
                    [3., 6.],
                    [11., 12.16666667],
                    [1., 1.47196014],
                    [10., 10.],
                    [10.5, 11.25],
                    [11., 12.5],
                    [11.5, 13.],
                    [12., 14.]
                ]),
                columns=pd.Int64Index([0, 1], dtype='int64'),
                index=mapped_array.describe().index
            )
        )

    def test_idxmin(self):
        pd.testing.assert_series_equal(
            mapped_array.idxmin(),
            mapped_array.to_matrix().idxmin()
        )
        assert mapped_array['a'].idxmin() == mapped_array['a'].to_matrix().idxmin()
        pd.testing.assert_series_equal(
            mapped_array_grouped.idxmin(),
            pd.Series(np.array(['x', 'z'], dtype=np.object), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_idxmax(self):
        pd.testing.assert_series_equal(
            mapped_array.idxmax(),
            mapped_array.to_matrix().idxmax()
        )
        assert mapped_array['a'].idxmax() == mapped_array['a'].to_matrix().idxmax()
        pd.testing.assert_series_equal(
            mapped_array_grouped.idxmax(),
            pd.Series(np.array(['z', 'y'], dtype=np.object), index=pd.Int64Index([0, 1], dtype='int64'))
        )

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
            mapped_array['b'].mapped_arr,
            np.array([13., 14., 13.])
        )
        np.testing.assert_array_equal(
            mapped_array['b'].col_arr,
            np.array([0, 0, 0])
        )
        pd.testing.assert_index_equal(
            mapped_array['b'].wrapper.columns,
            pd.Index(['b'], dtype='object')
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
            np.array([10., 11., 12., 13., 14., 13.])
        )
        np.testing.assert_array_equal(
            mapped_array[['a', 'b']].col_arr,
            np.array([0, 0, 0, 1, 1, 1])
        )
        pd.testing.assert_index_equal(
            mapped_array[['a', 'b']].wrapper.columns,
            pd.Index(['a', 'b'], dtype='object')
        )
        with pytest.raises(Exception) as e_info:
            _ = mapped_array.iloc[::2, :]  # changing time not supported
        pd.testing.assert_index_equal(
            mapped_array_grouped[0].wrapper.columns,
            pd.Index(['a'], dtype='object')
        )
        assert mapped_array_grouped[0].wrapper.ndim == 1
        assert mapped_array_grouped[0].wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            mapped_array_grouped[0].wrapper.grouper.group_by,
            pd.Int64Index([0], dtype='int64')
        )
        pd.testing.assert_index_equal(
            mapped_array_grouped[1].wrapper.columns,
            pd.Index(['b', 'c', 'd'], dtype='object')
        )
        assert mapped_array_grouped[1].wrapper.ndim == 2
        assert mapped_array_grouped[1].wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            mapped_array_grouped[1].wrapper.grouper.group_by,
            pd.Int64Index([1, 1, 1], dtype='int64')
        )
        pd.testing.assert_index_equal(
            mapped_array_grouped[[0]].wrapper.columns,
            pd.Index(['a'], dtype='object')
        )
        assert mapped_array_grouped[[0]].wrapper.ndim == 2
        assert mapped_array_grouped[[0]].wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            mapped_array_grouped[[0]].wrapper.grouper.group_by,
            pd.Int64Index([0], dtype='int64')
        )
        pd.testing.assert_index_equal(
            mapped_array_grouped[[0, 1]].wrapper.columns,
            pd.Index(['a', 'b', 'c', 'd'], dtype='object')
        )
        assert mapped_array_grouped[[0, 1]].wrapper.ndim == 2
        assert mapped_array_grouped[[0, 1]].wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            mapped_array_grouped[[0, 1]].wrapper.grouper.group_by,
            pd.Int64Index([0, 1, 1, 1], dtype='int64')
        )

    def test_magic(self):
        a = vbt.MappedArray(
            wrapper,
            records_arr['some_field1'],
            records_arr['col'],
            idx_arr=records_arr['idx']
        )
        b = records_arr['some_field2']
        a_bool = vbt.MappedArray(
            wrapper,
            records_arr['some_field1'] > np.mean(records_arr['some_field1']),
            records_arr['col'],
            idx_arr=records_arr['idx']
        )
        b_bool = records_arr['some_field2'] > np.mean(records_arr['some_field2'])

        # test what's allowed
        np.testing.assert_array_equal((a * b).mapped_arr, a.mapped_arr * b)
        np.testing.assert_array_equal((a * vbt.MappedArray(
            wrapper,
            records_arr['some_field2'],
            records_arr['col'],
            idx_arr=records_arr['idx']
        )).mapped_arr, a.mapped_arr * b)
        with pytest.raises(Exception) as e_info:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                wrapper,
                records_arr['some_field2'],
                records_arr['col'] * 2,
                idx_arr=records_arr['idx']
            )).mapped_arr, a.mapped_arr * b)
        with pytest.raises(Exception) as e_info:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                wrapper,
                records_arr['some_field2'],
                records_arr['col'],
                idx_arr=None
            )).mapped_arr, a.mapped_arr * b)
        with pytest.raises(Exception) as e_info:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                wrapper,
                records_arr['some_field2'],
                records_arr['col'],
                idx_arr=records_arr['idx'] * 2
            )).mapped_arr, a.mapped_arr * b)
        with pytest.raises(Exception) as e_info:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                wrapper.copy(group_by=group_by),
                records_arr['some_field2'],
                records_arr['col'],
                idx_arr=records_arr['idx'],
            )).mapped_arr, a.mapped_arr * b)
        with pytest.raises(Exception) as e_info:
            np.testing.assert_array_equal((a * vbt.MappedArray(
                ArrayWrapper(
                    index=['x', 'y', 'z'],
                    columns=['a', 'b', 'c', 'd'],
                    ndim=2,
                    freq=None
                ),
                records_arr['some_field2'],
                records_arr['col'],
                idx_arr=records_arr['idx']
            )).mapped_arr, a.mapped_arr * b)

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
    def test_regroup(self):
        pd.testing.assert_index_equal(
            records.regroup(group_by=group_by).wrapper.grouper.group_by,
            records_grouped.wrapper.grouper.group_by
        )

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
        record_arrays_close(
            records.filter_by_mask(mask).records_arr,
            np.array([
                (0, 2, 12., 19.),
                (1, 0, 13., 18.),
                (1, 1, 14., 17.),
                (1, 2, 13., 18.),
                (2, 0, 12., 19.)
            ], dtype=example_dt)
        )
        mask_a = records['a'].records_arr['some_field1'] >= records['a'].records_arr['some_field1'].mean()
        record_arrays_close(
            records['a'].filter_by_mask(mask_a).records_arr,
            np.array([
                (0, 1, 11., 20.),
                (0, 2, 12., 19.)
            ], dtype=example_dt)
        )
        assert records_grouped.filter_by_mask(mask).wrapper == records_grouped.wrapper

    def test_map_field(self):
        np.testing.assert_array_equal(
            records.map_field('some_field1').mapped_arr,
            np.array([10., 11., 12., 13., 14., 13., 12., 11., 10.])
        )
        np.testing.assert_array_equal(
            records['a'].map_field('some_field1').mapped_arr,
            np.array([10., 11., 12.])
        )
        np.testing.assert_array_equal(
            records.map_field('some_field1', idx_arr=records_arr['idx'] * 2).idx_arr,
            records_arr['idx'] * 2
        )
        assert records_grouped.map_field('some_field1').wrapper == records_grouped.wrapper
        assert records_grouped.map_field('some_field1', group_by=False).wrapper.grouper.group_by is None

    def test_map(self):
        @njit
        def map_func_nb(record):
            return record['some_field1'] + record['some_field2']

        np.testing.assert_array_equal(
            records.map(map_func_nb).mapped_arr,
            np.array([31., 31., 31., 31., 31., 31., 31., 31., 31.])
        )
        np.testing.assert_array_equal(
            records['a'].map(map_func_nb).mapped_arr,
            np.array([31., 31., 31.])
        )
        np.testing.assert_array_equal(
            records.map(map_func_nb, idx_arr=records_arr['idx'] * 2).idx_arr,
            records_arr['idx'] * 2
        )
        assert records_grouped.map(map_func_nb).wrapper == records_grouped.wrapper
        assert records_grouped.map(map_func_nb, group_by=False).wrapper.grouper.group_by is None

    def test_map_array(self):
        arr = records_arr['some_field1'] + records_arr['some_field2']
        np.testing.assert_array_equal(
            records.map_array(arr).mapped_arr,
            np.array([31., 31., 31., 31., 31., 31., 31., 31., 31.])
        )
        np.testing.assert_array_equal(
            records['a'].map_array(arr[:3]).mapped_arr,
            np.array([31., 31., 31.])
        )
        np.testing.assert_array_equal(
            records.map_array(arr, idx_arr=records_arr['idx'] * 2).idx_arr,
            records_arr['idx'] * 2
        )
        assert records_grouped.map_array(arr).wrapper == records_grouped.wrapper
        assert records_grouped.map_array(arr, group_by=False).wrapper.grouper.group_by is None

    def test_count(self):
        pd.testing.assert_series_equal(
            records.count(),
            pd.Series(
                np.array([3, 3, 3, 0]),
                index=wrapper.columns
            )
        )
        assert records['a'].count() == 3
        pd.testing.assert_series_equal(
            records_grouped.count(),
            pd.Series(
                np.array([3, 6]),
                index=pd.Int64Index([0, 1], dtype='int64')
            )
        )
        assert records_grouped[0].count() == 3

    def test_indexing(self):
        record_arrays_close(
            records['a'].records_arr,
            np.array([
                (0, 0, 10, 21),
                (0, 1, 11, 20),
                (0, 2, 12, 19)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records['a'].wrapper.columns,
            pd.Index(['a'], dtype='object')
        )
        record_arrays_close(
            records['b'].records_arr,
            np.array([
                (0, 0, 13, 18),
                (0, 1, 14, 17),
                (0, 2, 13, 18)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records['b'].wrapper.columns,
            pd.Index(['b'], dtype='object')
        )
        record_arrays_close(
            records[['a', 'a']].records_arr,
            np.array([
                (0, 0, 10, 21),
                (0, 1, 11, 20),
                (0, 2, 12, 19),
                (1, 0, 10, 21),
                (1, 1, 11, 20),
                (1, 2, 12, 19)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records[['a', 'a']].wrapper.columns,
            pd.Index(['a', 'a'], dtype='object')
        )
        record_arrays_close(
            records[['a', 'b']].records_arr,
            np.array([
                (0, 0, 10, 21),
                (0, 1, 11, 20),
                (0, 2, 12, 19),
                (1, 0, 13, 18),
                (1, 1, 14, 17),
                (1, 2, 13, 18)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records[['a', 'b']].wrapper.columns,
            pd.Index(['a', 'b'], dtype='object')
        )
        with pytest.raises(Exception) as e_info:
            _ = records.iloc[::2, :]  # changing time not supported
        pd.testing.assert_index_equal(
            records_grouped[0].wrapper.columns,
            pd.Index(['a'], dtype='object')
        )
        assert records_grouped[0].wrapper.ndim == 1
        assert records_grouped[0].wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            records_grouped[0].wrapper.grouper.group_by,
            pd.Int64Index([0], dtype='int64')
        )
        pd.testing.assert_index_equal(
            records_grouped[1].wrapper.columns,
            pd.Index(['b', 'c', 'd'], dtype='object')
        )
        assert records_grouped[1].wrapper.ndim == 2
        assert records_grouped[1].wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            records_grouped[1].wrapper.grouper.group_by,
            pd.Int64Index([1, 1, 1], dtype='int64')
        )
        pd.testing.assert_index_equal(
            records_grouped[[0]].wrapper.columns,
            pd.Index(['a'], dtype='object')
        )
        assert records_grouped[[0]].wrapper.ndim == 2
        assert records_grouped[[0]].wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            records_grouped[[0]].wrapper.grouper.group_by,
            pd.Int64Index([0], dtype='int64')
        )
        pd.testing.assert_index_equal(
            records_grouped[[0, 1]].wrapper.columns,
            pd.Index(['a', 'b', 'c', 'd'], dtype='object')
        )
        assert records_grouped[[0, 1]].wrapper.ndim == 2
        assert records_grouped[[0, 1]].wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            records_grouped[[0, 1]].wrapper.grouper.group_by,
            pd.Int64Index([0, 1, 1, 1], dtype='int64')
        )

    def test_filtering(self):
        filtered_records = vbt.Records(wrapper, records_arr[[0, -1]])
        record_arrays_close(
            filtered_records.records_arr,
            np.array([(0, 0, 10., 21.), (2, 2, 10., 21.)], dtype=example_dt)
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
        assert filtered_records['a'].count() == 1.
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
        assert np.isnan(filtered_records['b'].map_field('some_field1').min())
        assert filtered_records['b'].count() == 0.
        # c
        record_arrays_close(
            filtered_records['c'].records_arr,
            np.array([(0, 2, 10., 21.)], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['c'].col_index,
            np.array([[0, 1]])
        )
        np.testing.assert_array_equal(
            filtered_records['c'].map_field('some_field1').mapped_arr,
            np.array([10.])
        )
        assert filtered_records['c'].map_field('some_field1').min() == 10.
        assert filtered_records['c'].count() == 1.
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
        assert np.isnan(filtered_records['d'].map_field('some_field1').min())
        assert filtered_records['d'].count() == 0.


# ############# drawdowns.py ############# #


ts = pd.DataFrame({
    'a': [2, 1, 3, 1, 4, 1],
    'b': [1, 2, 1, 3, 1, 4],
    'c': [1, 2, 3, 2, 1, 2],
    'd': [1, 2, 3, 4, 5, 6]
})

drawdowns = vbt.Drawdowns.from_ts(ts, freq='1 days')
drawdowns_grouped = vbt.Drawdowns.from_ts(ts, freq='1 days', group_by=group_by)


class TestDrawdowns:
    def test_records_readable(self):
        pd.testing.assert_frame_equal(
            drawdowns.records_readable,
            pd.DataFrame({
                'Column': ['a', 'a', 'a', 'b', 'b', 'c'],
                'Start Date': [0, 2, 4, 1, 3, 2],
                'Valley Date': [1, 3, 5, 2, 4, 4],
                'End Date': [2, 4, 5, 3, 5, 5],
                'Status': ['Recovered', 'Recovered', 'Active', 'Recovered', 'Recovered', 'Active']
            })
        )

    def test_from_ts(self):
        record_arrays_close(
            drawdowns_grouped.records_arr,
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
        assert drawdowns.idx_field == 'end_idx'
        pd.testing.assert_index_equal(
            drawdowns_grouped.wrapper.grouper.group_by,
            group_by
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
            drawdowns.avg_drawdown(),
            pd.Series(np.array([-0.63888889, -0.58333333, -0.66666667, 0.]), index=wrapper.columns)
        )
        assert drawdowns['a'].avg_drawdown() == -0.6388888888888888
        pd.testing.assert_series_equal(
            drawdowns_grouped.avg_drawdown(),
            pd.Series(np.array([-0.6388888888888888, -0.611111111111111]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_max_drawdown(self):
        pd.testing.assert_series_equal(
            drawdowns.max_drawdown(),
            pd.Series(np.array([-0.75, -0.66666667, -0.66666667, 0.]), index=wrapper.columns)
        )
        assert drawdowns['a'].max_drawdown() == -0.75
        pd.testing.assert_series_equal(
            drawdowns_grouped.max_drawdown(),
            pd.Series(np.array([-0.75, -0.6666666666666666]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

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
            drawdowns.avg_duration(),
            pd.Series(
                np.array([144000000000000, 172800000000000, 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            )
        )
        assert drawdowns['a'].avg_duration() == np.timedelta64(144000000000000)
        pd.testing.assert_series_equal(
            drawdowns_grouped.avg_duration(),
            pd.Series(np.array([144000000000000, 201600000000000], dtype='timedelta64[ns]'), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_max_duration(self):
        pd.testing.assert_series_equal(
            drawdowns.max_duration(),
            pd.Series(
                np.array([172800000000000, 172800000000000, 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            )
        )
        assert drawdowns['a'].max_duration() == np.timedelta64(172800000000000)
        pd.testing.assert_series_equal(
            drawdowns_grouped.max_duration(),
            pd.Series(np.array([172800000000000, 259200000000000], dtype='timedelta64[ns]'), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_coverage(self):
        pd.testing.assert_series_equal(
            drawdowns.coverage(),
            pd.Series(np.array([0.83333333, 0.66666667, 0.5, 0.]), index=ts.columns)
        )
        assert drawdowns['a'].coverage() == 0.8333333333333334
        pd.testing.assert_series_equal(
            drawdowns_grouped.coverage(),
            pd.Series(np.array([0.8333333333333334, 0.3888888888888889]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

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
            drawdowns.recovered_rate(),
            pd.Series(np.array([0.66666667, 1., 0., np.nan]), index=ts.columns)
        )
        assert drawdowns['a'].recovered_rate() == 0.6666666666666666
        pd.testing.assert_series_equal(
            drawdowns_grouped.recovered_rate(),
            pd.Series(np.array([0.6666666666666666, 0.6666666666666666]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_active_records(self):
        assert isinstance(drawdowns.active, ActiveDrawdowns)
        assert drawdowns.active.wrapper == drawdowns.wrapper
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
            drawdowns.active.current_drawdown(),
            pd.Series(np.array([-0.75, np.nan, -0.3333333333333333, np.nan]), index=wrapper.columns)
        )
        assert drawdowns['a'].active.current_drawdown() == -0.75
        with pytest.raises(Exception) as e_info:
            drawdowns_grouped.active.current_drawdown()

    def test_current_duration(self):
        pd.testing.assert_series_equal(
            drawdowns.active.current_duration(),
            pd.Series(
                np.array([86400000000000, 'NaT', 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            )
        )
        assert drawdowns['a'].active.current_duration() == np.timedelta64(86400000000000)
        with pytest.raises(Exception) as e_info:
            drawdowns_grouped.active.current_duration()

    def test_current_return(self):
        pd.testing.assert_series_equal(
            drawdowns.active.current_return(),
            pd.Series(np.array([0., np.nan, 1., np.nan]), index=wrapper.columns)
        )
        assert drawdowns['a'].active.current_return() == 0.
        with pytest.raises(Exception) as e_info:
            drawdowns_grouped.active.current_return()

    def test_recovered_records(self):
        assert isinstance(drawdowns.recovered, RecoveredDrawdowns)
        assert drawdowns.recovered.wrapper == drawdowns.wrapper
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

wrapper2 = vbt.base.array_wrapper.ArrayWrapper.from_obj(price, freq='1 days')
wrapper2_grouped = wrapper2.copy(group_by=group_by)
orders = vbt.Orders(wrapper2, order_records_arr, price)
orders_grouped = vbt.Orders(wrapper2_grouped, order_records_arr, price)


class TestOrders:
    def test_records_readable(self):
        pd.testing.assert_frame_equal(
            orders.records_readable,
            pd.DataFrame({
                'Column': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'd', 'd', 'd'],
                'Date': [2, 3, 4, 6, 2, 3, 4, 5, 0, 1, 6, 2, 4, 6],
                'Size': [33.00330033, 33.00330033, 25.8798157, 25.8798157, 14.14427157,
                         14.14427157, 16.63702438, 16.63702438, 99.00990099, 99.00990099,
                         194.09861778, 49.5049505, 49.5049505, 24.26232722],
                'Price': [3., 4., 5., 7., 7., 6., 5., 4., 1., 2., 1., 2., 2., 4.],
                'Fees': [0.99009901, 1.32013201, 1.29399079, 1.8115871, 0.99009901, 0.84865629,
                         0.83185122, 0.66548098, 0.99009901, 1.98019802, 1.94098618, 0.99009901,
                         0.99009901, 0.97049309],
                'Side': ['Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell', 'Buy',
                         'Buy', 'Sell', 'Buy']
            })
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
        assert orders.buy.wrapper == orders.wrapper
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
        assert orders.sell.wrapper == orders.wrapper
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
], dtype=trade_dt)

events = vbt.Events(wrapper2, event_records_arr, price)
events_grouped = vbt.Events(wrapper2_grouped, event_records_arr, price)


class TestEvents:
    def test_records_readable(self):
        pd.testing.assert_frame_equal(
            events.records_readable,
            pd.DataFrame({
                'Column': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                'Size': [33.00330033, 25.8798157, 14.14427157, 16.63702438, 99.00990099,
                         194.09861778, 49.5049505, 24.26232722],
                'Entry Date': [2, 4, 2, 4, 0, 6, 2, 6],
                'Entry Price': [3., 5., 7., 5., 1., 1., 2., 4.],
                'Entry Fees': [0.99009901, 1.29399079, 0.99009901, 0.83185122, 0.99009901, 1.94098618,
                               0.99009901, 0.97049309],
                'Exit Date': [3, 6, 3, 5, 1, 6, 4, 6],
                'Exit Price': [4., 7., 6., 4., 2., 1., 2., 4.],
                'Exit Fees': [1.32013201, 1.8115871, 0.84865629, 0.66548098, 1.98019802, 0.,
                              0.99009901, 0.],
                'P&L': [30.69306931, 48.65405351, -15.98302687, -18.13435658, 96.03960396,
                        -1.94098618, -1.98019802, -0.97049309],
                'Return': [0.30693069, 0.37227723, -0.15983027, -0.21584158, 0.96039604, -0.00990099,
                           -0.01980198, -0.00990099],
                'Status': ['Closed', 'Closed', 'Closed', 'Closed', 'Closed', 'Open', 'Closed', 'Open']
            })
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
            events.coverage(),
            pd.Series(np.array([0.42857143, 0.28571429, 0.14285714, 0.28571429]), index=ts.columns)
        )
        assert events['a'].coverage() == 0.42857142857142855
        pd.testing.assert_series_equal(
            events_grouped.coverage(),
            pd.Series(np.array([0.42857142857142855, 0.23809523809523808]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

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
        assert events.winning.wrapper == events.wrapper
        record_arrays_close(
            events.winning.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1),
                (2, 99.00990099, 0, 1., 0.99009901, 1, 2., 1.98019802, 96.03960396, 0.96039604, 1)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            events['a'].winning.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            events.winning['a'].records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=trade_dt)
        )

    def test_losing_records(self):
        assert isinstance(events.losing, BaseEvents)
        assert events.losing.wrapper == events.wrapper
        record_arrays_close(
            events.losing.records_arr,
            np.array([
                (1, 14.14427157, 2, 7., 0.99009901, 3, 6., 0.84865629, -15.98302687, -0.15983027, 1),
                (1, 16.63702438, 4, 5., 0.83185122, 5, 4., 0.66548098, -18.13435658, -0.21584158, 1),
                (2, 194.09861778, 6, 1., 1.94098618, 6, 1., 0., -1.94098618, -0.00990099, 0),
                (3, 49.5049505, 2, 2., 0.99009901, 4, 2., 0.99009901, -1.98019802, -0.01980198, 1),
                (3, 24.26232722, 6, 4., 0.97049309, 6, 4., 0., -0.97049309, -0.00990099, 0)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            events['a'].losing.records_arr,
            np.array([], dtype=trade_dt)
        )
        record_arrays_close(
            events.losing['a'].records_arr,
            np.array([], dtype=trade_dt)
        )

    def test_win_rate(self):
        pd.testing.assert_series_equal(
            events.win_rate(),
            pd.Series(np.array([1., 0., 0.5, 0.]), index=ts.columns)
        )
        assert events['a'].win_rate() == 1.
        pd.testing.assert_series_equal(
            events_grouped.win_rate(),
            pd.Series(np.array([1.0, 0.16666666666666666]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_profit_factor(self):
        pd.testing.assert_series_equal(
            events.profit_factor(),
            pd.Series(np.array([np.inf, 0., 49.47979792, 0.]), index=ts.columns)
        )
        assert np.isinf(events['a'].profit_factor())
        pd.testing.assert_series_equal(
            events_grouped.profit_factor(),
            pd.Series(np.array([np.inf, 2.461981963629304]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_expectancy(self):
        pd.testing.assert_series_equal(
            events.expectancy(),
            pd.Series(np.array([39.67356141, -17.05869172, 47.04930889, -1.47534555]), index=ts.columns)
        )
        assert events['a'].expectancy() == 39.67356141
        pd.testing.assert_series_equal(
            events_grouped.expectancy(),
            pd.Series(np.array([39.67356141, 9.505090536666662]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_sqn(self):
        pd.testing.assert_series_equal(
            events.sqn(),
            pd.Series(np.array([4.41774916, -15.85874229, 0.96038019, -2.9223301]), index=ts.columns)
        )
        assert events['a'].sqn() == 4.4177491576436
        pd.testing.assert_series_equal(
            events_grouped.sqn(),
            pd.Series(np.array([4.4177491576436, 0.5405954574869949]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

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
            events.closed_rate(),
            pd.Series(np.array([1., 1., 0.5, 0.5]), index=ts.columns)
        )
        assert events['a'].closed_rate() == 1.0
        pd.testing.assert_series_equal(
            events_grouped.closed_rate(),
            pd.Series(np.array([1.0, 0.6666666666666666]), index=pd.Int64Index([0, 1], dtype='int64'))
        )

    def test_open_records(self):
        assert isinstance(events.open, BaseEventsByResult)
        assert events.open.wrapper == events.wrapper
        record_arrays_close(
            events.open.records_arr,
            np.array([
                (2, 194.09861778, 6, 1., 1.94098618, 6, 1., 0., -1.94098618, -0.00990099, 0),
                (3, 24.26232722, 6, 4., 0.97049309, 6, 4., 0., -0.97049309, -0.00990099, 0)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            events['a'].open.records_arr,
            np.array([], dtype=trade_dt)
        )
        record_arrays_close(
            events.open['a'].records_arr,
            np.array([], dtype=trade_dt)
        )

    def test_closed_records(self):
        assert isinstance(events.closed, BaseEventsByResult)
        assert events.closed.wrapper == events.wrapper
        record_arrays_close(
            events.closed.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1),
                (1, 14.14427157, 2, 7., 0.99009901, 3, 6., 0.84865629, -15.98302687, -0.15983027, 1),
                (1, 16.63702438, 4, 5., 0.83185122, 5, 4., 0.66548098, -18.13435658, -0.21584158, 1),
                (2, 99.00990099, 0, 1., 0.99009901, 1, 2., 1.98019802, 96.03960396, 0.96039604, 1),
                (3, 49.5049505, 2, 2., 0.99009901, 4, 2., 0.99009901, -1.98019802, -0.01980198, 1)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            events['a'].closed.records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            events.closed['a'].records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=trade_dt)
        )


trades = vbt.Trades.from_orders(orders)
trades_grouped = vbt.Trades.from_orders(orders_grouped)


class TestTrades:
    def test_records_readable(self):
        pd.testing.assert_frame_equal(
            trades.records_readable,
            pd.DataFrame({
                'Column': ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd'],
                'Size': [33.00330033, 25.8798157, 14.14427157, 16.63702438, 99.00990099,
                         194.09861778, 49.5049505, 24.26232722],
                'Entry Date': [2, 4, 2, 4, 0, 6, 2, 6],
                'Entry Price': [3., 5., 7., 5., 1., 1., 2., 4.],
                'Entry Fees': [0.99009901, 1.29399079, 0.99009901, 0.83185122, 0.99009901, 1.94098618,
                               0.99009901, 0.97049309],
                'Exit Date': [3, 6, 3, 5, 1, 6, 4, 6],
                'Exit Price': [4., 7., 6., 4., 2., 1., 2., 4.],
                'Exit Fees': [1.32013201, 1.8115871, 0.84865629, 0.66548098, 1.98019802, 0.,
                              0.99009901, 0.],
                'P&L': [30.69306931, 48.65405351, -15.98302687, -18.13435658, 96.03960396,
                        -1.94098618, -1.98019802, -0.97049309],
                'Return': [0.30693069, 0.37227723, -0.15983027, -0.21584158, 0.96039604, -0.00990099,
                           -0.01980198, -0.00990099],
                'Status': ['Closed', 'Closed', 'Closed', 'Closed', 'Closed', 'Open', 'Closed', 'Open'],
                'Position': [0, 1, 2, 3, 4, 5, 6, 7]
            })
        )

    def test_from_orders(self):
        trades = vbt.Trades.from_orders(orders)
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
        record_arrays_close(
            trades['a'].records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1, 0),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1, 1)
            ], dtype=trade_dt)
        )
        pd.testing.assert_frame_equal(trades.close, price)
        assert trades.wrapper.freq == day_dt
        assert trades.idx_field == 'exit_idx'
        pd.testing.assert_index_equal(
            trades_grouped.wrapper.grouper.group_by,
            group_by
        )

    def test_position_idx(self):
        np.testing.assert_array_almost_equal(
            trades.position_idx.mapped_arr,
            np.array([0, 1, 2, 3, 4, 5, 6, 7])
        )
        np.testing.assert_array_almost_equal(
            trades['a'].position_idx.mapped_arr,
            np.array([0, 1])
        )

    def test_hiearchy(self):
        assert isinstance(trades.closed, BaseTradesByResult)
        assert isinstance(trades.closed.winning, BaseTrades)


positions = vbt.Positions.from_orders(orders)
positions_grouped = vbt.Positions.from_orders(orders_grouped)


class TestPositions:
    def test_from_orders(self):
        positions = vbt.Positions.from_orders(orders)
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
        record_arrays_close(
            positions['a'].records_arr,
            np.array([
                (0, 33.00330033, 2, 3., 0.99009901, 3, 4., 1.32013201, 30.69306931, 0.30693069, 1),
                (0, 25.8798157, 4, 5., 1.29399079, 6, 7., 1.8115871, 48.65405351, 0.37227723, 1)
            ], dtype=position_dt)
        )
        pd.testing.assert_frame_equal(positions.close, price)
        assert positions.wrapper.freq == day_dt
        assert positions.idx_field == 'exit_idx'
        pd.testing.assert_index_equal(
            positions_grouped.wrapper.grouper.group_by,
            group_by
        )

    def test_hiearchy(self):
        assert isinstance(positions.closed, BasePositionsByResult)
        assert isinstance(positions.closed.winning, BasePositions)

