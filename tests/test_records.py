import vectorbt as vbt
import numpy as np
import pandas as pd
from numba import njit
from datetime import datetime
import pytest

from vectorbt.generic.enums import drawdown_dt
from vectorbt.portfolio.enums import order_dt, trade_dt, position_dt, log_dt

from tests.utils import record_arrays_close

day_dt = np.timedelta64(86400000000000)

example_dt = np.dtype([
    ('id', np.int64),
    ('idx', np.int64),
    ('col', np.int64),
    ('some_field1', np.float64),
    ('some_field2', np.float64)
], align=True)

records_arr = np.asarray([
    (0, 0, 0, 10, 21),
    (1, 1, 0, 11, 20),
    (2, 2, 0, 12, 19),
    (3, 0, 1, 13, 18),
    (4, 1, 1, 14, 17),
    (5, 2, 1, 13, 18),
    (6, 0, 2, 12, 19),
    (7, 1, 2, 11, 20),
    (8, 2, 2, 10, 21)
], dtype=example_dt)

group_by = pd.Index(['g1', 'g1', 'g2', 'g2'])

wrapper = vbt.ArrayWrapper(
    index=['x', 'y', 'z'],
    columns=['a', 'b', 'c', 'd'],
    ndim=2,
    freq='1 days'
)
wrapper_grouped = wrapper.copy(group_by=group_by)

records = vbt.records.Records(wrapper, records_arr)
records_grouped = vbt.records.Records(wrapper_grouped, records_arr)
records_nosort = records.copy(records_arr=records.records_arr[::-1])


# ############# col_mapper.py ############# #


class TestColumnMapper:
    def test_col_arr(self):
        np.testing.assert_array_equal(
            records['a'].col_mapper.col_arr,
            np.array([0, 0, 0])
        )
        np.testing.assert_array_equal(
            records.col_mapper.col_arr,
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        )

    def test_get_col_arr(self):
        np.testing.assert_array_equal(
            records.col_mapper.get_col_arr(),
            records.col_mapper.col_arr
        )
        np.testing.assert_array_equal(
            records_grouped['g1'].col_mapper.get_col_arr(),
            np.array([0, 0, 0, 0, 0, 0])
        )
        np.testing.assert_array_equal(
            records_grouped.col_mapper.get_col_arr(),
            np.array([0, 0, 0, 0, 0, 0, 1, 1, 1])
        )

    def test_col_range(self):
        np.testing.assert_array_equal(
            records['a'].col_mapper.col_range,
            np.array([
                [0, 3]
            ])
        )
        np.testing.assert_array_equal(
            records.col_mapper.col_range,
            np.array([
                [0, 3],
                [3, 6],
                [6, 9],
                [-1, -1]
            ])
        )

    def test_get_col_range(self):
        np.testing.assert_array_equal(
            records.col_mapper.get_col_range(),
            np.array([
                [0, 3],
                [3, 6],
                [6, 9],
                [-1, -1]
            ])
        )
        np.testing.assert_array_equal(
            records_grouped['g1'].col_mapper.get_col_range(),
            np.array([[0, 6]])
        )
        np.testing.assert_array_equal(
            records_grouped.col_mapper.get_col_range(),
            np.array([[0, 6], [6, 9]])
        )

    def test_col_map(self):
        np.testing.assert_array_equal(
            records['a'].col_mapper.col_map[0],
            np.array([0, 1, 2])
        )
        np.testing.assert_array_equal(
            records['a'].col_mapper.col_map[1],
            np.array([3])
        )
        np.testing.assert_array_equal(
            records.col_mapper.col_map[0],
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        )
        np.testing.assert_array_equal(
            records.col_mapper.col_map[1],
            np.array([3, 3, 3, 0])
        )

    def test_get_col_map(self):
        np.testing.assert_array_equal(
            records.col_mapper.get_col_map()[0],
            records.col_mapper.col_map[0]
        )
        np.testing.assert_array_equal(
            records.col_mapper.get_col_map()[1],
            records.col_mapper.col_map[1]
        )
        np.testing.assert_array_equal(
            records_grouped['g1'].col_mapper.get_col_map()[0],
            np.array([0, 1, 2, 3, 4, 5])
        )
        np.testing.assert_array_equal(
            records_grouped['g1'].col_mapper.get_col_map()[1],
            np.array([6])
        )
        np.testing.assert_array_equal(
            records_grouped.col_mapper.get_col_map()[0],
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        )
        np.testing.assert_array_equal(
            records_grouped.col_mapper.get_col_map()[1],
            np.array([6, 3])
        )

    def test_is_sorted(self):
        assert records.col_mapper.is_sorted()
        assert not records_nosort.col_mapper.is_sorted()


# ############# mapped_array.py ############# #

mapped_array = records.map_field('some_field1')
mapped_array_grouped = records_grouped.map_field('some_field1')
mapped_array_nosort = mapped_array.copy(
    col_arr=mapped_array.col_arr[::-1],
    id_arr=mapped_array.id_arr[::-1],
    idx_arr=mapped_array.idx_arr[::-1]
)


class TestMappedArray:
    def test_config(self, tmp_path):
        assert vbt.MappedArray.loads(mapped_array.dumps()) == mapped_array
        mapped_array.save(tmp_path / 'mapped_array')
        assert vbt.MappedArray.load(tmp_path / 'mapped_array') == mapped_array

    def test_mapped_arr(self):
        np.testing.assert_array_equal(
            mapped_array['a'].values,
            np.array([10., 11., 12.])
        )
        np.testing.assert_array_equal(
            mapped_array.values,
            np.array([10., 11., 12., 13., 14., 13., 12., 11., 10.])
        )

    def test_id_arr(self):
        np.testing.assert_array_equal(
            mapped_array['a'].id_arr,
            np.array([0, 1, 2])
        )
        np.testing.assert_array_equal(
            mapped_array.id_arr,
            np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        )

    def test_col_arr(self):
        np.testing.assert_array_equal(
            mapped_array['a'].col_arr,
            np.array([0, 0, 0])
        )
        np.testing.assert_array_equal(
            mapped_array.col_arr,
            np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        )

    def test_idx_arr(self):
        np.testing.assert_array_equal(
            mapped_array['a'].idx_arr,
            np.array([0, 1, 2])
        )
        np.testing.assert_array_equal(
            mapped_array.idx_arr,
            np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
        )

    def test_is_sorted(self):
        assert mapped_array.is_sorted()
        assert mapped_array.is_sorted(incl_id=True)
        assert not mapped_array_nosort.is_sorted()
        assert not mapped_array_nosort.is_sorted(incl_id=True)

    def test_sort(self):
        assert mapped_array.sort().is_sorted()
        assert mapped_array.sort().is_sorted(incl_id=True)
        assert mapped_array.sort(incl_id=True).is_sorted(incl_id=True)
        assert mapped_array_nosort.sort().is_sorted()
        assert not mapped_array_nosort.sort().is_sorted(incl_id=True)
        assert mapped_array_nosort.sort(incl_id=True).is_sorted(incl_id=True)

    def test_filter_by_mask(self):
        mask_a = mapped_array['a'].values >= mapped_array['a'].values.mean()
        np.testing.assert_array_equal(
            mapped_array['a'].filter_by_mask(mask_a).id_arr,
            np.array([1, 2])
        )
        mask = mapped_array.values >= mapped_array.values.mean()
        filtered = mapped_array.filter_by_mask(mask)
        np.testing.assert_array_equal(
            filtered.id_arr,
            np.array([2, 3, 4, 5, 6])
        )
        np.testing.assert_array_equal(filtered.col_arr, mapped_array.col_arr[mask])
        np.testing.assert_array_equal(filtered.idx_arr, mapped_array.idx_arr[mask])
        assert mapped_array_grouped.filter_by_mask(mask).wrapper == mapped_array_grouped.wrapper
        assert mapped_array_grouped.filter_by_mask(mask, group_by=False).wrapper.grouper.group_by is None

    def test_map_to_mask(self):
        @njit
        def every_2_nb(inout, idxs, col, mapped_arr):
            inout[idxs[::2]] = True

        np.testing.assert_array_equal(
            mapped_array.map_to_mask(every_2_nb),
            np.array([True, False, True, True, False, True, True, False, True])
        )

    def test_top_n_mask(self):
        np.testing.assert_array_equal(
            mapped_array.top_n_mask(1),
            np.array([False, False, True, False, True, False, True, False, False])
        )

    def test_bottom_n_mask(self):
        np.testing.assert_array_equal(
            mapped_array.bottom_n_mask(1),
            np.array([True, False, False, True, False, False, False, False, True])
        )

    def test_top_n(self):
        np.testing.assert_array_equal(
            mapped_array.top_n(1).id_arr,
            np.array([2, 4, 6])
        )

    def test_bottom_n(self):
        np.testing.assert_array_equal(
            mapped_array.bottom_n(1).id_arr,
            np.array([0, 3, 8])
        )

    def test_to_pd(self):
        target = pd.DataFrame(
            np.array([
                [10., 13., 12., np.nan],
                [11., 14., 11., np.nan],
                [12., 13., 10., np.nan]
            ]),
            index=wrapper.index,
            columns=wrapper.columns
        )
        pd.testing.assert_series_equal(
            mapped_array['a'].to_pd(),
            target['a']
        )
        pd.testing.assert_frame_equal(
            mapped_array.to_pd(),
            target
        )
        pd.testing.assert_frame_equal(
            mapped_array.to_pd(default_val=0.),
            target.fillna(0.)
        )
        mapped_array2 = vbt.MappedArray(
            wrapper,
            records_arr['some_field1'].tolist() + [1],
            records_arr['col'].tolist() + [2],
            idx_arr=records_arr['idx'].tolist() + [2]
        )
        with pytest.raises(Exception) as e_info:
            _ = mapped_array2.to_pd()
        pd.testing.assert_series_equal(
            mapped_array['a'].to_pd(ignore_index=True),
            pd.Series(np.array([10., 11., 12.]), name='a')
        )
        pd.testing.assert_frame_equal(
            mapped_array.to_pd(ignore_index=True),
            pd.DataFrame(
                np.array([
                    [10., 13., 12., np.nan],
                    [11., 14., 11., np.nan],
                    [12., 13., 10., np.nan]
                ]),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.to_pd(default_val=0, ignore_index=True),
            pd.DataFrame(
                np.array([
                    [10., 13., 12., 0.],
                    [11., 14., 11., 0.],
                    [12., 13., 10., 0.]
                ]),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array_grouped.to_pd(ignore_index=True),
            pd.DataFrame(
                np.array([
                    [10., 12.],
                    [11., 11.],
                    [12., 10.],
                    [13., np.nan],
                    [14., np.nan],
                    [13., np.nan],
                ]),
                columns=pd.Index(['g1', 'g2'], dtype='object')
            )
        )

    def test_reduce(self):
        @njit
        def mean_reduce_nb(col, a):
            return np.mean(a)

        assert mapped_array['a'].reduce(mean_reduce_nb) == 11.
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb),
            pd.Series(np.array([11., 13.333333333333334, 11., np.nan]), index=wrapper.columns).rename('reduce')
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, default_val=0.),
            pd.Series(np.array([11., 13.333333333333334, 11., 0.]), index=wrapper.columns).rename('reduce')
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, default_val=0., wrap_kwargs=dict(dtype=np.int_)),
            pd.Series(np.array([11, 13, 11, 0]), index=wrapper.columns).rename('reduce')
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(mean_reduce_nb, wrap_kwargs=dict(time_units=True)),
            pd.Series(np.array([11., 13.333333333333334, 11., np.nan]), index=wrapper.columns).rename('reduce') * day_dt
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.reduce(mean_reduce_nb),
            pd.Series([12.166666666666666, 11.0], index=pd.Index(['g1', 'g2'], dtype='object')).rename('reduce')
        )
        assert mapped_array_grouped['g1'].reduce(mean_reduce_nb) == 12.166666666666666
        pd.testing.assert_series_equal(
            mapped_array_grouped[['g1']].reduce(mean_reduce_nb),
            pd.Series([12.166666666666666], index=pd.Index(['g1'], dtype='object')).rename('reduce')
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

        assert mapped_array['a'].reduce(argmin_reduce_nb, to_idx=True) == 'x'
        pd.testing.assert_series_equal(
            mapped_array.reduce(argmin_reduce_nb, to_idx=True),
            pd.Series(np.array(['x', 'x', 'z', np.nan], dtype=object), index=wrapper.columns).rename('reduce')
        )
        pd.testing.assert_series_equal(
            mapped_array.reduce(argmin_reduce_nb, to_idx=True, idx_labeled=False),
            pd.Series(np.array([0, 0, 2, -1], dtype=int), index=wrapper.columns).rename('reduce')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.reduce(argmin_reduce_nb, to_idx=True, idx_labeled=False),
            pd.Series(np.array([0, 2], dtype=int), index=pd.Index(['g1', 'g2'], dtype='object')).rename('reduce')
        )

    def test_reduce_to_array(self):
        @njit
        def min_max_reduce_nb(col, a):
            return np.array([np.min(a), np.max(a)])

        pd.testing.assert_series_equal(
            mapped_array['a'].reduce(min_max_reduce_nb, to_array=True, wrap_kwargs=dict(name_or_index=['min', 'max'])),
            pd.Series([10., 12.], index=pd.Index(['min', 'max'], dtype='object'), name='a')
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, wrap_kwargs=dict(name_or_index=['min', 'max'])),
            pd.DataFrame(
                np.array([
                    [10., 13., 10., np.nan],
                    [12., 14., 12., np.nan]
                ]),
                index=pd.Index(['min', 'max'], dtype='object'),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, default_val=0.),
            pd.DataFrame(
                np.array([
                    [10., 13., 10., 0.],
                    [12., 14., 12., 0.]
                ]),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, wrap_kwargs=dict(time_units=True)),
            pd.DataFrame(
                np.array([
                    [10., 13., 10., np.nan],
                    [12., 14., 12., np.nan]
                ]),
                columns=wrapper.columns
            ) * day_dt
        )
        pd.testing.assert_frame_equal(
            mapped_array_grouped.reduce(min_max_reduce_nb, to_array=True),
            pd.DataFrame(
                np.array([
                    [10., 10.],
                    [14., 12.]
                ]),
                columns=pd.Index(['g1', 'g2'], dtype='object')
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True),
            mapped_array_grouped.reduce(min_max_reduce_nb, to_array=True, group_by=False)
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(min_max_reduce_nb, to_array=True, group_by=group_by),
            mapped_array_grouped.reduce(min_max_reduce_nb, to_array=True)
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped['g1'].reduce(min_max_reduce_nb, to_array=True),
            pd.Series([10., 14.], name='g1')
        )
        pd.testing.assert_frame_equal(
            mapped_array_grouped[['g1']].reduce(min_max_reduce_nb, to_array=True),
            pd.DataFrame([[10.], [14.]], columns=pd.Index(['g1'], dtype='object'))
        )

    def test_reduce_to_idx_array(self):
        @njit
        def idxmin_idxmax_reduce_nb(col, a):
            return np.array([np.argmin(a), np.argmax(a)])

        pd.testing.assert_series_equal(
            mapped_array['a'].reduce(
                idxmin_idxmax_reduce_nb,
                to_array=True,
                to_idx=True,
                wrap_kwargs=dict(name_or_index=['min', 'max'])
            ),
            pd.Series(
                np.array(['x', 'z'], dtype=object),
                index=pd.Index(['min', 'max'], dtype='object'),
                name='a'
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb,
                to_array=True,
                to_idx=True,
                wrap_kwargs=dict(name_or_index=['min', 'max'])
            ),
            pd.DataFrame(
                np.array([
                    ['x', 'x', 'z', np.nan],
                    ['z', 'y', 'x', np.nan]
                ], dtype=object),
                index=pd.Index(['min', 'max'], dtype='object'),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.reduce(
                idxmin_idxmax_reduce_nb,
                to_array=True,
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
                to_idx=True,
                idx_labeled=False
            ),
            pd.DataFrame(
                np.array([
                    [0, 2],
                    [1, 0]
                ]),
                columns=pd.Index(['g1', 'g2'], dtype='object')
            )
        )

    def test_nst(self):
        assert mapped_array['a'].nst(0) == 10.
        pd.testing.assert_series_equal(
            mapped_array.nst(0),
            pd.Series(np.array([10., 13., 12., np.nan]), index=wrapper.columns).rename('nst')
        )
        assert mapped_array['a'].nst(-1) == 12.
        pd.testing.assert_series_equal(
            mapped_array.nst(-1),
            pd.Series(np.array([12., 13., 10., np.nan]), index=wrapper.columns).rename('nst')
        )
        with pytest.raises(Exception) as e_info:
            _ = mapped_array.nst(10)
        pd.testing.assert_series_equal(
            mapped_array_grouped.nst(0),
            pd.Series(np.array([10., 12.]), index=pd.Index(['g1', 'g2'], dtype='object')).rename('nst')
        )

    def test_min(self):
        assert mapped_array['a'].min() == mapped_array['a'].to_pd().min()
        pd.testing.assert_series_equal(
            mapped_array.min(),
            mapped_array.to_pd().min().rename('min')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.min(),
            pd.Series([10., 10.], index=pd.Index(['g1', 'g2'], dtype='object')).rename('min')
        )

    def test_max(self):
        assert mapped_array['a'].max() == mapped_array['a'].to_pd().max()
        pd.testing.assert_series_equal(
            mapped_array.max(),
            mapped_array.to_pd().max().rename('max')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.max(),
            pd.Series([14., 12.], index=pd.Index(['g1', 'g2'], dtype='object')).rename('max')
        )

    def test_mean(self):
        assert mapped_array['a'].mean() == mapped_array['a'].to_pd().mean()
        pd.testing.assert_series_equal(
            mapped_array.mean(),
            mapped_array.to_pd().mean().rename('mean')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.mean(),
            pd.Series([12.166667, 11.], index=pd.Index(['g1', 'g2'], dtype='object')).rename('mean')
        )

    def test_median(self):
        assert mapped_array['a'].median() == mapped_array['a'].to_pd().median()
        pd.testing.assert_series_equal(
            mapped_array.median(),
            mapped_array.to_pd().median().rename('median')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.median(),
            pd.Series([12.5, 11.], index=pd.Index(['g1', 'g2'], dtype='object')).rename('median')
        )

    def test_std(self):
        assert mapped_array['a'].std() == mapped_array['a'].to_pd().std()
        pd.testing.assert_series_equal(
            mapped_array.std(),
            mapped_array.to_pd().std().rename('std')
        )
        pd.testing.assert_series_equal(
            mapped_array.std(ddof=0),
            mapped_array.to_pd().std(ddof=0).rename('std')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.std(),
            pd.Series([1.4719601443879746, 1.0], index=pd.Index(['g1', 'g2'], dtype='object')).rename('std')
        )

    def test_sum(self):
        assert mapped_array['a'].sum() == mapped_array['a'].to_pd().sum()
        pd.testing.assert_series_equal(
            mapped_array.sum(),
            mapped_array.to_pd().sum().rename('sum')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.sum(),
            pd.Series([73.0, 33.0], index=pd.Index(['g1', 'g2'], dtype='object')).rename('sum')
        )

    def test_count(self):
        assert mapped_array['a'].count() == mapped_array['a'].to_pd().count()
        pd.testing.assert_series_equal(
            mapped_array.count(),
            mapped_array.to_pd().count().rename('count')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.count(),
            pd.Series([6, 3], index=pd.Index(['g1', 'g2'], dtype='object')).rename('count')
        )

    def test_idxmin(self):
        assert mapped_array['a'].idxmin() == mapped_array['a'].to_pd().idxmin()
        pd.testing.assert_series_equal(
            mapped_array.idxmin(),
            mapped_array.to_pd().idxmin().rename('idxmin')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.idxmin(),
            pd.Series(
                np.array(['x', 'z'], dtype=object),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('idxmin')
        )

    def test_idxmax(self):
        assert mapped_array['a'].idxmax() == mapped_array['a'].to_pd().idxmax()
        pd.testing.assert_series_equal(
            mapped_array.idxmax(),
            mapped_array.to_pd().idxmax().rename('idxmax')
        )
        pd.testing.assert_series_equal(
            mapped_array_grouped.idxmax(),
            pd.Series(
                np.array(['y', 'x'], dtype=object),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('idxmax')
        )

    def test_describe(self):
        pd.testing.assert_series_equal(
            mapped_array['a'].describe(),
            mapped_array['a'].to_pd().describe()
        )
        pd.testing.assert_frame_equal(
            mapped_array.describe(percentiles=None),
            mapped_array.to_pd().describe(percentiles=None)
        )
        pd.testing.assert_frame_equal(
            mapped_array.describe(percentiles=[]),
            mapped_array.to_pd().describe(percentiles=[])
        )
        pd.testing.assert_frame_equal(
            mapped_array.describe(percentiles=np.arange(0, 1, 0.1)),
            mapped_array.to_pd().describe(percentiles=np.arange(0, 1, 0.1))
        )
        pd.testing.assert_frame_equal(
            mapped_array_grouped.describe(),
            pd.DataFrame(
                np.array([
                    [6., 3.],
                    [12.16666667, 11.],
                    [1.47196014, 1.],
                    [10., 10.],
                    [11.25, 10.5],
                    [12.5, 11.],
                    [13., 11.5],
                    [14., 12.]
                ]),
                columns=pd.Index(['g1', 'g2'], dtype='object'),
                index=mapped_array.describe().index
            )
        )

    def test_value_counts(self):
        pd.testing.assert_series_equal(
            mapped_array['a'].value_counts(),
            pd.Series(
                np.array([1, 1, 1]),
                index=pd.Float64Index([10.0, 11.0, 12.0], dtype='float64'),
                name='a'
            )
        )
        value_map = {10: 'ten', 11: 'eleven', 12: 'twelve'}
        pd.testing.assert_series_equal(
            mapped_array['a'].value_counts(value_map=value_map),
            pd.Series(
                np.array([1, 1, 1]),
                index=pd.Index(['ten', 'eleven', 'twelve'], dtype='object'),
                name='a'
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array.value_counts(),
            pd.DataFrame(
                np.array([
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [1, 0, 1, 0],
                    [0, 2, 0, 0],
                    [0, 1, 0, 0]
                ]),
                index=pd.Float64Index([10.0, 11.0, 12.0, 13.0, 14.0], dtype='float64'),
                columns=wrapper.columns
            )
        )
        pd.testing.assert_frame_equal(
            mapped_array_grouped.value_counts(),
            pd.DataFrame(
                np.array([
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [2, 0],
                    [1, 0]
                ]),
                index=pd.Float64Index([10.0, 11.0, 12.0, 13.0, 14.0], dtype='float64'),
                columns=pd.Index(['g1', 'g2'], dtype='object')
            )
        )

    def test_indexing(self):
        np.testing.assert_array_equal(
            mapped_array['a'].id_arr,
            np.array([0, 1, 2])
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
            mapped_array['b'].id_arr,
            np.array([3, 4, 5])
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
            mapped_array[['a', 'a']].id_arr,
            np.array([0, 1, 2, 0, 1, 2])
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
            mapped_array[['a', 'b']].id_arr,
            np.array([0, 1, 2, 3, 4, 5])
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
            mapped_array_grouped['g1'].wrapper.columns,
            pd.Index(['a', 'b'], dtype='object')
        )
        assert mapped_array_grouped['g1'].wrapper.ndim == 2
        assert mapped_array_grouped['g1'].wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            mapped_array_grouped['g1'].wrapper.grouper.group_by,
            pd.Index(['g1', 'g1'], dtype='object')
        )
        pd.testing.assert_index_equal(
            mapped_array_grouped['g2'].wrapper.columns,
            pd.Index(['c', 'd'], dtype='object')
        )
        assert mapped_array_grouped['g2'].wrapper.ndim == 2
        assert mapped_array_grouped['g2'].wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            mapped_array_grouped['g2'].wrapper.grouper.group_by,
            pd.Index(['g2', 'g2'], dtype='object')
        )
        pd.testing.assert_index_equal(
            mapped_array_grouped[['g1']].wrapper.columns,
            pd.Index(['a', 'b'], dtype='object')
        )
        assert mapped_array_grouped[['g1']].wrapper.ndim == 2
        assert mapped_array_grouped[['g1']].wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            mapped_array_grouped[['g1']].wrapper.grouper.group_by,
            pd.Index(['g1', 'g1'], dtype='object')
        )
        pd.testing.assert_index_equal(
            mapped_array_grouped[['g1', 'g2']].wrapper.columns,
            pd.Index(['a', 'b', 'c', 'd'], dtype='object')
        )
        assert mapped_array_grouped[['g1', 'g2']].wrapper.ndim == 2
        assert mapped_array_grouped[['g1', 'g2']].wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            mapped_array_grouped[['g1', 'g2']].wrapper.grouper.group_by,
            pd.Index(['g1', 'g1', 'g2', 'g2'], dtype='object')
        )

    def test_magic(self):
        a = vbt.MappedArray(
            wrapper,
            records_arr['some_field1'],
            records_arr['col'],
            id_arr=records_arr['id'],
            idx_arr=records_arr['idx']
        )
        a_inv = vbt.MappedArray(
            wrapper,
            records_arr['some_field1'][::-1],
            records_arr['col'][::-1],
            id_arr=records_arr['id'][::-1],
            idx_arr=records_arr['idx'][::-1]
        )
        b = records_arr['some_field2']
        a_bool = vbt.MappedArray(
            wrapper,
            records_arr['some_field1'] > np.mean(records_arr['some_field1']),
            records_arr['col'],
            id_arr=records_arr['id'],
            idx_arr=records_arr['idx']
        )
        b_bool = records_arr['some_field2'] > np.mean(records_arr['some_field2'])
        assert a ** a == a ** 2
        with pytest.raises(Exception) as e_info:
            _ = a * a_inv

        # binary ops
        # comparison ops
        np.testing.assert_array_equal((a == b).values, a.values == b)
        np.testing.assert_array_equal((a != b).values, a.values != b)
        np.testing.assert_array_equal((a < b).values, a.values < b)
        np.testing.assert_array_equal((a > b).values, a.values > b)
        np.testing.assert_array_equal((a <= b).values, a.values <= b)
        np.testing.assert_array_equal((a >= b).values, a.values >= b)
        # arithmetic ops
        np.testing.assert_array_equal((a + b).values, a.values + b)
        np.testing.assert_array_equal((a - b).values, a.values - b)
        np.testing.assert_array_equal((a * b).values, a.values * b)
        np.testing.assert_array_equal((a ** b).values, a.values ** b)
        np.testing.assert_array_equal((a % b).values, a.values % b)
        np.testing.assert_array_equal((a // b).values, a.values // b)
        np.testing.assert_array_equal((a / b).values, a.values / b)
        # __r*__ is only called if the left object does not have an __*__ method
        np.testing.assert_array_equal((10 + a).values, 10 + a.values)
        np.testing.assert_array_equal((10 - a).values, 10 - a.values)
        np.testing.assert_array_equal((10 * a).values, 10 * a.values)
        np.testing.assert_array_equal((10 ** a).values, 10 ** a.values)
        np.testing.assert_array_equal((10 % a).values, 10 % a.values)
        np.testing.assert_array_equal((10 // a).values, 10 // a.values)
        np.testing.assert_array_equal((10 / a).values, 10 / a.values)
        # mask ops
        np.testing.assert_array_equal((a_bool & b_bool).values, a_bool.values & b_bool)
        np.testing.assert_array_equal((a_bool | b_bool).values, a_bool.values | b_bool)
        np.testing.assert_array_equal((a_bool ^ b_bool).values, a_bool.values ^ b_bool)
        np.testing.assert_array_equal((True & a_bool).values, True & a_bool.values)
        np.testing.assert_array_equal((True | a_bool).values, True | a_bool.values)
        np.testing.assert_array_equal((True ^ a_bool).values, True ^ a_bool.values)
        # unary ops
        np.testing.assert_array_equal((-a).values, -a.values)
        np.testing.assert_array_equal((+a).values, +a.values)
        np.testing.assert_array_equal((abs(-a)).values, abs((-a.values)))


# ############# base.py ############# #

class TestRecords:
    def test_config(self, tmp_path):
        assert vbt.Records.loads(records['a'].dumps()) == records['a']
        assert vbt.Records.loads(records.dumps()) == records
        records.save(tmp_path / 'records')
        assert vbt.Records.load(tmp_path / 'records') == records

    def test_records(self):
        pd.testing.assert_frame_equal(
            records.records,
            pd.DataFrame.from_records(records_arr)
        )

    def test_recarray(self):
        np.testing.assert_array_equal(records['a'].recarray.some_field1, records['a'].values['some_field1'])
        np.testing.assert_array_equal(records.recarray.some_field1, records.values['some_field1'])

    def test_is_sorted(self):
        assert records.is_sorted()
        assert records.is_sorted(incl_id=True)
        assert not records_nosort.is_sorted()
        assert not records_nosort.is_sorted(incl_id=True)

    def test_sort(self):
        assert records.sort().is_sorted()
        assert records.sort().is_sorted(incl_id=True)
        assert records.sort(incl_id=True).is_sorted(incl_id=True)
        assert records_nosort.sort().is_sorted()
        assert not records_nosort.sort().is_sorted(incl_id=True)
        assert records_nosort.sort(incl_id=True).is_sorted(incl_id=True)

    def test_filter_by_mask(self):
        mask_a = records['a'].values['some_field1'] >= records['a'].values['some_field1'].mean()
        record_arrays_close(
            records['a'].filter_by_mask(mask_a).values,
            np.array([
                (1, 1, 0, 11., 20.), (2, 2, 0, 12., 19.)
            ], dtype=example_dt)
        )
        mask = records.values['some_field1'] >= records.values['some_field1'].mean()
        filtered = records.filter_by_mask(mask)
        record_arrays_close(
            filtered.values,
            np.array([
                (2, 2, 0, 12., 19.), (3, 0, 1, 13., 18.), (4, 1, 1, 14., 17.),
                (5, 2, 1, 13., 18.), (6, 0, 2, 12., 19.)
            ], dtype=example_dt)
        )
        assert records_grouped.filter_by_mask(mask).wrapper == records_grouped.wrapper

    def test_map_field(self):
        np.testing.assert_array_equal(
            records['a'].map_field('some_field1').values,
            np.array([10., 11., 12.])
        )
        np.testing.assert_array_equal(
            records.map_field('some_field1').values,
            np.array([10., 11., 12., 13., 14., 13., 12., 11., 10.])
        )
        np.testing.assert_array_equal(
            records.map_field('some_field1', idx_field='col').idx_arr,
            records_arr['col']
        )
        assert records_grouped.map_field('some_field1').wrapper == records_grouped.wrapper
        assert records_grouped.map_field('some_field1', group_by=False).wrapper.grouper.group_by is None

    def test_map(self):
        @njit
        def map_func_nb(record):
            return record['some_field1'] + record['some_field2']

        np.testing.assert_array_equal(
            records['a'].map(map_func_nb).values,
            np.array([31., 31., 31.])
        )
        np.testing.assert_array_equal(
            records.map(map_func_nb).values,
            np.array([31., 31., 31., 31., 31., 31., 31., 31., 31.])
        )
        np.testing.assert_array_equal(
            records.map(map_func_nb, idx_field='col').idx_arr,
            records_arr['col']
        )
        assert records_grouped.map(map_func_nb).wrapper == records_grouped.wrapper
        assert records_grouped.map(map_func_nb, group_by=False).wrapper.grouper.group_by is None

    def test_map_array(self):
        arr = records_arr['some_field1'] + records_arr['some_field2']
        np.testing.assert_array_equal(
            records['a'].map_array(arr[:3]).values,
            np.array([31., 31., 31.])
        )
        np.testing.assert_array_equal(
            records.map_array(arr).values,
            np.array([31., 31., 31., 31., 31., 31., 31., 31., 31.])
        )
        np.testing.assert_array_equal(
            records.map_array(arr, idx_field='col').idx_arr,
            records_arr['col']
        )
        assert records_grouped.map_array(arr).wrapper == records_grouped.wrapper
        assert records_grouped.map_array(arr, group_by=False).wrapper.grouper.group_by is None

    def test_count(self):
        assert records['a'].count() == 3
        pd.testing.assert_series_equal(
            records.count(),
            pd.Series(
                np.array([3, 3, 3, 0]),
                index=wrapper.columns
            ).rename('count')
        )
        assert records_grouped['g1'].count() == 6
        pd.testing.assert_series_equal(
            records_grouped.count(),
            pd.Series(
                np.array([6, 3]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('count')
        )

    def test_indexing(self):
        record_arrays_close(
            records['a'].values,
            np.array([
                (0, 0, 0, 10., 21.), (1, 1, 0, 11., 20.), (2, 2, 0, 12., 19.)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records['a'].wrapper.columns,
            pd.Index(['a'], dtype='object')
        )
        record_arrays_close(
            records['b'].values,
            np.array([
                (3, 0, 0, 13., 18.), (4, 1, 0, 14., 17.), (5, 2, 0, 13., 18.)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records['b'].wrapper.columns,
            pd.Index(['b'], dtype='object')
        )
        record_arrays_close(
            records[['a', 'a']].values,
            np.array([
                (0, 0, 0, 10., 21.), (1, 1, 0, 11., 20.), (2, 2, 0, 12., 19.),
                (0, 0, 1, 10., 21.), (1, 1, 1, 11., 20.), (2, 2, 1, 12., 19.)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records[['a', 'a']].wrapper.columns,
            pd.Index(['a', 'a'], dtype='object')
        )
        record_arrays_close(
            records[['a', 'b']].values,
            np.array([
                (0, 0, 0, 10., 21.), (1, 1, 0, 11., 20.), (2, 2, 0, 12., 19.),
                (3, 0, 1, 13., 18.), (4, 1, 1, 14., 17.), (5, 2, 1, 13., 18.)
            ], dtype=example_dt)
        )
        pd.testing.assert_index_equal(
            records[['a', 'b']].wrapper.columns,
            pd.Index(['a', 'b'], dtype='object')
        )
        with pytest.raises(Exception) as e_info:
            _ = records.iloc[::2, :]  # changing time not supported
        pd.testing.assert_index_equal(
            records_grouped['g1'].wrapper.columns,
            pd.Index(['a', 'b'], dtype='object')
        )
        assert records_grouped['g1'].wrapper.ndim == 2
        assert records_grouped['g1'].wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            records_grouped['g1'].wrapper.grouper.group_by,
            pd.Index(['g1', 'g1'], dtype='object')
        )
        pd.testing.assert_index_equal(
            records_grouped['g2'].wrapper.columns,
            pd.Index(['c', 'd'], dtype='object')
        )
        assert records_grouped['g2'].wrapper.ndim == 2
        assert records_grouped['g2'].wrapper.grouped_ndim == 1
        pd.testing.assert_index_equal(
            records_grouped['g2'].wrapper.grouper.group_by,
            pd.Index(['g2', 'g2'], dtype='object')
        )
        pd.testing.assert_index_equal(
            records_grouped[['g1']].wrapper.columns,
            pd.Index(['a', 'b'], dtype='object')
        )
        assert records_grouped[['g1']].wrapper.ndim == 2
        assert records_grouped[['g1']].wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            records_grouped[['g1']].wrapper.grouper.group_by,
            pd.Index(['g1', 'g1'], dtype='object')
        )
        pd.testing.assert_index_equal(
            records_grouped[['g1', 'g2']].wrapper.columns,
            pd.Index(['a', 'b', 'c', 'd'], dtype='object')
        )
        assert records_grouped[['g1', 'g2']].wrapper.ndim == 2
        assert records_grouped[['g1', 'g2']].wrapper.grouped_ndim == 2
        pd.testing.assert_index_equal(
            records_grouped[['g1', 'g2']].wrapper.grouper.group_by,
            pd.Index(['g1', 'g1', 'g2', 'g2'], dtype='object')
        )

    def test_filtering(self):
        filtered_records = vbt.Records(wrapper, records_arr[[0, -1]])
        record_arrays_close(
            filtered_records.values,
            np.array([(0, 0, 0, 10., 21.), (8, 2, 2, 10., 21.)], dtype=example_dt)
        )
        # a
        record_arrays_close(
            filtered_records['a'].values,
            np.array([(0, 0, 0, 10., 21.)], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['a'].map_field('some_field1').id_arr,
            np.array([0])
        )
        assert filtered_records['a'].map_field('some_field1').min() == 10.
        assert filtered_records['a'].count() == 1.
        # b
        record_arrays_close(
            filtered_records['b'].values,
            np.array([], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['b'].map_field('some_field1').id_arr,
            np.array([])
        )
        assert np.isnan(filtered_records['b'].map_field('some_field1').min())
        assert filtered_records['b'].count() == 0.
        # c
        record_arrays_close(
            filtered_records['c'].values,
            np.array([(8, 2, 0, 10., 21.)], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['c'].map_field('some_field1').id_arr,
            np.array([8])
        )
        assert filtered_records['c'].map_field('some_field1').min() == 10.
        assert filtered_records['c'].count() == 1.
        # d
        record_arrays_close(
            filtered_records['d'].values,
            np.array([], dtype=example_dt)
        )
        np.testing.assert_array_equal(
            filtered_records['d'].map_field('some_field1').id_arr,
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
}, index=[
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    datetime(2020, 1, 3),
    datetime(2020, 1, 4),
    datetime(2020, 1, 5),
    datetime(2020, 1, 6)
])

drawdowns = vbt.Drawdowns.from_ts(ts, freq='1 days')
drawdowns_grouped = vbt.Drawdowns.from_ts(ts, freq='1 days', group_by=group_by)


class TestDrawdowns:
    def test_from_ts(self):
        record_arrays_close(
            drawdowns.values,
            np.array([
                (0, 0, 0, 1, 2, 1), (1, 0, 2, 3, 4, 1), (2, 0, 4, 5, 5, 0), (3, 1, 1, 2, 3, 1),
                (4, 1, 3, 4, 5, 1), (5, 2, 2, 4, 5, 0)
            ], dtype=drawdown_dt)
        )
        pd.testing.assert_frame_equal(drawdowns.ts, ts)
        assert drawdowns.wrapper.freq == day_dt
        assert drawdowns.idx_field == 'end_idx'
        pd.testing.assert_index_equal(
            drawdowns_grouped.wrapper.grouper.group_by,
            group_by
        )

    def test_records_readable(self):
        records_readable = drawdowns.records_readable

        np.testing.assert_array_equal(
            records_readable['Drawdown Id'].values,
            np.array([
                0, 1, 2, 3, 4, 5
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Column'].values,
            np.array([
                'a', 'a', 'a', 'b', 'b', 'c'
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Start Date'].values,
            np.array([
                '2020-01-01T00:00:00.000000000', '2020-01-03T00:00:00.000000000',
                '2020-01-05T00:00:00.000000000', '2020-01-02T00:00:00.000000000',
                '2020-01-04T00:00:00.000000000', '2020-01-03T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable['Valley Date'].values,
            np.array([
                '2020-01-02T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-03T00:00:00.000000000',
                '2020-01-05T00:00:00.000000000', '2020-01-05T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable['End Date'].values,
            np.array([
                '2020-01-03T00:00:00.000000000', '2020-01-05T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-06T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable['Status'].values,
            np.array([
                'Recovered', 'Recovered', 'Active', 'Recovered', 'Recovered', 'Active'
            ])
        )

    def test_start_value(self):
        np.testing.assert_array_equal(
            drawdowns['a'].start_value.values,
            np.array([2., 3., 4.])
        )
        np.testing.assert_array_equal(
            drawdowns.start_value.values,
            np.array([2., 3., 4., 2., 3., 3.])
        )

    def test_valley_value(self):
        np.testing.assert_array_equal(
            drawdowns['a'].valley_value.values,
            np.array([1., 1., 1.])
        )
        np.testing.assert_array_equal(
            drawdowns.valley_value.values,
            np.array([1., 1., 1., 1., 1., 1.])
        )

    def test_end_value(self):
        np.testing.assert_array_equal(
            drawdowns['a'].end_value.values,
            np.array([3., 4., 1.])
        )
        np.testing.assert_array_equal(
            drawdowns.end_value.values,
            np.array([3., 4., 1., 3., 4., 2.])
        )

    def test_drawdown(self):
        np.testing.assert_array_almost_equal(
            drawdowns['a'].drawdown.values,
            np.array([-0.5, -0.66666667, -0.75])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.drawdown.values,
            np.array([-0.5, -0.66666667, -0.75, -0.5, -0.66666667, -0.66666667])
        )
        pd.testing.assert_frame_equal(
            drawdowns.drawdown.to_pd(),
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
        assert drawdowns['a'].avg_drawdown() == -0.6388888888888888
        pd.testing.assert_series_equal(
            drawdowns.avg_drawdown(),
            pd.Series(
                np.array([-0.63888889, -0.58333333, -0.66666667, 0.]),
                index=wrapper.columns
            ).rename('avg_drawdown')
        )
        pd.testing.assert_series_equal(
            drawdowns_grouped.avg_drawdown(),
            pd.Series(
                np.array([-0.6166666666666666, -0.6666666666666666]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('avg_drawdown')
        )

    def test_max_drawdown(self):
        assert drawdowns['a'].max_drawdown() == -0.75
        pd.testing.assert_series_equal(
            drawdowns.max_drawdown(),
            pd.Series(
                np.array([-0.75, -0.66666667, -0.66666667, 0.]),
                index=wrapper.columns
            ).rename('max_drawdown')
        )
        pd.testing.assert_series_equal(
            drawdowns_grouped.max_drawdown(),
            pd.Series(
                np.array([-0.75, -0.6666666666666666]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('max_drawdown')
        )

    def test_duration(self):
        np.testing.assert_array_almost_equal(
            drawdowns['a'].duration.values,
            np.array([2., 2., 1.])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.duration.values,
            np.array([2., 2., 1., 2., 2., 3.])
        )

    def test_avg_duration(self):
        assert drawdowns['a'].avg_duration() == np.timedelta64(144000000000000)
        pd.testing.assert_series_equal(
            drawdowns.avg_duration(),
            pd.Series(
                np.array([144000000000000, 172800000000000, 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            ).rename('avg_duration')
        )
        pd.testing.assert_series_equal(
            drawdowns_grouped.avg_duration(),
            pd.Series(
                np.array([155520000000000, 259200000000000], dtype='timedelta64[ns]'),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('avg_duration')
        )

    def test_max_duration(self):
        assert drawdowns['a'].max_duration() == np.timedelta64(172800000000000)
        pd.testing.assert_series_equal(
            drawdowns.max_duration(),
            pd.Series(
                np.array([172800000000000, 172800000000000, 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            ).rename('max_duration')
        )
        pd.testing.assert_series_equal(
            drawdowns_grouped.max_duration(),
            pd.Series(
                np.array([172800000000000, 259200000000000], dtype='timedelta64[ns]'),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('max_duration')
        )

    def test_coverage(self):
        assert drawdowns['a'].coverage() == 0.8333333333333334
        pd.testing.assert_series_equal(
            drawdowns.coverage(),
            pd.Series(
                np.array([0.83333333, 0.66666667, 0.5, 0.]),
                index=ts.columns
            ).rename('coverage')
        )
        pd.testing.assert_series_equal(
            drawdowns_grouped.coverage(),
            pd.Series(
                np.array([0.75, 0.25]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('coverage')
        )

    def test_ptv_duration(self):
        np.testing.assert_array_almost_equal(
            drawdowns['a'].ptv_duration.values,
            np.array([1., 1., 1.])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.ptv_duration.values,
            np.array([1., 1., 1., 1., 1., 2.])
        )

    def test_status(self):
        np.testing.assert_array_almost_equal(
            drawdowns['a'].status.values,
            np.array([1, 1, 0])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.status.values,
            np.array([1, 1, 0, 1, 1, 0])
        )

    def test_active_records(self):
        assert isinstance(drawdowns.active, vbt.Drawdowns)
        assert drawdowns.active.wrapper == drawdowns.wrapper
        record_arrays_close(
            drawdowns['a'].active.values,
            np.array([
                (2, 0, 4, 5, 5, 0)
            ], dtype=drawdown_dt)
        )
        record_arrays_close(
            drawdowns['a'].active.values,
            drawdowns.active['a'].values
        )
        record_arrays_close(
            drawdowns.active.values,
            np.array([
                (2, 0, 4, 5, 5, 0), (5, 2, 2, 4, 5, 0)
            ], dtype=drawdown_dt)
        )

    def test_active_rate(self):
        assert drawdowns['a'].active_rate() == 0.3333333333333333
        pd.testing.assert_series_equal(
            drawdowns.active_rate(),
            pd.Series(
                np.array([0.3333333333333333, 0., 1., np.nan]),
                index=ts.columns
            ).rename('active_rate')
        )
        pd.testing.assert_series_equal(
            drawdowns_grouped.active_rate(),
            pd.Series(
                np.array([0.2, 1.0]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('active_rate')
        )

    def test_recovered_records(self):
        assert isinstance(drawdowns.recovered, vbt.Drawdowns)
        assert drawdowns.recovered.wrapper == drawdowns.wrapper
        record_arrays_close(
            drawdowns['a'].recovered.values,
            np.array([
                (0, 0, 0, 1, 2, 1), (1, 0, 2, 3, 4, 1)
            ], dtype=drawdown_dt)
        )
        record_arrays_close(
            drawdowns['a'].recovered.values,
            drawdowns.recovered['a'].values
        )
        record_arrays_close(
            drawdowns.recovered.values,
            np.array([
                (0, 0, 0, 1, 2, 1), (1, 0, 2, 3, 4, 1), (3, 1, 1, 2, 3, 1),
                (4, 1, 3, 4, 5, 1)
            ], dtype=drawdown_dt)
        )

    def test_recovered_rate(self):
        assert drawdowns['a'].recovered_rate() == 0.6666666666666666
        pd.testing.assert_series_equal(
            drawdowns.recovered_rate(),
            pd.Series(
                np.array([0.66666667, 1., 0., np.nan]),
                index=ts.columns
            ).rename('recovered_rate')
        )
        pd.testing.assert_series_equal(
            drawdowns_grouped.recovered_rate(),
            pd.Series(
                np.array([0.8, 0.0]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('recovered_rate')
        )

    def test_current_drawdown(self):
        assert drawdowns['a'].current_drawdown() == -0.75
        pd.testing.assert_series_equal(
            drawdowns.current_drawdown(),
            pd.Series(
                np.array([-0.75, np.nan, -0.3333333333333333, np.nan]),
                index=wrapper.columns
            ).rename('current_drawdown')
        )
        with pytest.raises(Exception) as e_info:
            drawdowns_grouped.current_drawdown()

    def test_current_duration(self):
        assert drawdowns['a'].current_duration() == np.timedelta64(86400000000000)
        pd.testing.assert_series_equal(
            drawdowns.current_duration(),
            pd.Series(
                np.array([86400000000000, 'NaT', 259200000000000, 'NaT'], dtype='timedelta64[ns]'),
                index=wrapper.columns
            ).rename('current_duration')
        )
        with pytest.raises(Exception) as e_info:
            drawdowns_grouped.current_duration()

    def test_current_return(self):
        assert drawdowns['a'].current_return() == 0.
        pd.testing.assert_series_equal(
            drawdowns.current_return(),
            pd.Series(
                np.array([0., np.nan, 1., np.nan]),
                index=wrapper.columns
            ).rename('current_return')
        )
        with pytest.raises(Exception) as e_info:
            drawdowns_grouped.current_return()

    def test_recovery_return(self):
        np.testing.assert_array_almost_equal(
            drawdowns['a'].recovered.recovery_return.values,
            np.array([2., 3.])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.recovered.recovery_return.values,
            np.array([2., 3., 2., 3.])
        )

    def test_vtr_duration(self):
        np.testing.assert_array_almost_equal(
            drawdowns['a'].recovered.vtr_duration.values,
            np.array([1., 1.])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.recovered.vtr_duration.values,
            np.array([1., 1., 1., 1.])
        )

    def test_vtr_duration_ratio(self):
        np.testing.assert_array_almost_equal(
            drawdowns['a'].recovered.vtr_duration_ratio.values,
            np.array([0.5, 0.5])
        )
        np.testing.assert_array_almost_equal(
            drawdowns.recovered.vtr_duration_ratio.values,
            np.array([0.5, 0.5, 0.5, 0.5])
        )


# ############# orders.py ############# #

close = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], index=[
    datetime(2020, 1, 1),
    datetime(2020, 1, 2),
    datetime(2020, 1, 3),
    datetime(2020, 1, 4),
    datetime(2020, 1, 5),
    datetime(2020, 1, 6),
    datetime(2020, 1, 7),
    datetime(2020, 1, 8)
]).vbt.tile(4, keys=['a', 'b', 'c', 'd'])

size = np.full(close.shape, np.nan, dtype=np.float_)
size[:, 0] = [1, 0.1, -1, -0.1, np.nan, 1, -1, 2]
size[:, 1] = [-1, -0.1, 1, 0.1, np.nan, -1, 1, -2]
size[:, 2] = [1, 0.1, -1, -0.1, np.nan, 1, -2, 2]
orders = vbt.Portfolio.from_orders(close, size, fees=0.01, freq='1 days').orders
orders_grouped = orders.regroup(group_by)


class TestOrders:
    def test_records_readable(self):
        records_readable = orders.records_readable

        np.testing.assert_array_equal(
            records_readable['Order Id'].values,
            np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Date'].values,
            np.array([
                '2020-01-01T00:00:00.000000000', '2020-01-02T00:00:00.000000000',
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-07T00:00:00.000000000',
                '2020-01-08T00:00:00.000000000', '2020-01-01T00:00:00.000000000',
                '2020-01-02T00:00:00.000000000', '2020-01-03T00:00:00.000000000',
                '2020-01-04T00:00:00.000000000', '2020-01-06T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-01T00:00:00.000000000', '2020-01-02T00:00:00.000000000',
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-07T00:00:00.000000000',
                '2020-01-08T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable['Column'].values,
            np.array([
                'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b',
                'b', 'c', 'c', 'c', 'c', 'c', 'c', 'c'
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Size'].values,
            np.array([
                1.0, 0.1, 1.0, 0.1, 1.0, 1.0, 2.0, 1.0, 0.1, 1.0, 0.1, 1.0, 1.0,
                2.0, 1.0, 0.1, 1.0, 0.1, 1.0, 2.0, 2.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Price'].values,
            np.array([
                1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0,
                8.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 8.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Fees'].values,
            np.array([
                0.01, 0.002, 0.03, 0.004, 0.06, 0.07, 0.16, 0.01, 0.002, 0.03,
                0.004, 0.06, 0.07, 0.16, 0.01, 0.002, 0.03, 0.004, 0.06, 0.14,
                0.16
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Side'].values,
            np.array([
                'Buy', 'Buy', 'Sell', 'Sell', 'Buy', 'Sell', 'Buy', 'Sell', 'Sell',
                'Buy', 'Buy', 'Sell', 'Buy', 'Sell', 'Buy', 'Buy', 'Sell', 'Sell',
                'Buy', 'Sell', 'Buy'
            ])
        )

    def test_size(self):
        np.testing.assert_array_equal(
            orders['a'].size.values,
            np.array([1., 0.1, 1., 0.1, 1., 1., 2.])
        )
        np.testing.assert_array_equal(
            orders.size.values,
            np.array([
                1., 0.1, 1., 0.1, 1., 1., 2., 1., 0.1, 1., 0.1, 1., 1.,
                2., 1., 0.1, 1., 0.1, 1., 2., 2.
            ])
        )

    def test_price(self):
        np.testing.assert_array_equal(
            orders['a'].price.values,
            np.array([1., 2., 3., 4., 6., 7., 8.])
        )
        np.testing.assert_array_equal(
            orders.price.values,
            np.array([
                1., 2., 3., 4., 6., 7., 8., 1., 2., 3., 4., 6., 7., 8., 1., 2., 3.,
                4., 6., 7., 8.
            ])
        )

    def test_fees(self):
        np.testing.assert_array_equal(
            orders['a'].fees.values,
            np.array([0.01, 0.002, 0.03, 0.004, 0.06, 0.07, 0.16])
        )
        np.testing.assert_array_equal(
            orders.fees.values,
            np.array([
                0.01, 0.002, 0.03, 0.004, 0.06, 0.07, 0.16, 0.01, 0.002,
                0.03, 0.004, 0.06, 0.07, 0.16, 0.01, 0.002, 0.03, 0.004,
                0.06, 0.14, 0.16
            ])
        )

    def test_side(self):
        np.testing.assert_array_equal(
            orders['a'].side.values,
            np.array([0, 0, 1, 1, 0, 1, 0])
        )
        np.testing.assert_array_equal(
            orders.side.values,
            np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0])
        )

    def test_buy_records(self):
        assert isinstance(orders.buy, vbt.Orders)
        assert orders.buy.wrapper == orders.wrapper
        record_arrays_close(
            orders['a'].buy.values,
            np.array([
                (0, 0, 0, 1., 1., 0.01, 0), (1, 1, 0, 0.1, 2., 0.002, 0),
                (4, 5, 0, 1., 6., 0.06, 0), (6, 7, 0, 2., 8., 0.16, 0)
            ], dtype=order_dt)
        )
        record_arrays_close(
            orders['a'].buy.values,
            orders.buy['a'].values
        )
        record_arrays_close(
            orders.buy.values,
            np.array([
                (0, 0, 0, 1., 1., 0.01, 0), (1, 1, 0, 0.1, 2., 0.002, 0),
                (4, 5, 0, 1., 6., 0.06, 0), (6, 7, 0, 2., 8., 0.16, 0),
                (9, 2, 1, 1., 3., 0.03, 0), (10, 3, 1, 0.1, 4., 0.004, 0),
                (12, 6, 1, 1., 7., 0.07, 0), (14, 0, 2, 1., 1., 0.01, 0),
                (15, 1, 2, 0.1, 2., 0.002, 0), (18, 5, 2, 1., 6., 0.06, 0),
                (20, 7, 2, 2., 8., 0.16, 0)
            ], dtype=order_dt)
        )

    def test_buy_rate(self):
        assert orders['a'].buy_rate() == 0.5714285714285714
        pd.testing.assert_series_equal(
            orders.buy_rate(),
            pd.Series(
                np.array([0.57142857, 0.42857143, 0.57142857, np.nan]),
                index=close.columns
            ).rename('buy_rate')
        )
        pd.testing.assert_series_equal(
            orders_grouped.buy_rate(),
            pd.Series(
                np.array([0.5, 0.57142857]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('buy_rate')
        )

    def test_sell_records(self):
        assert isinstance(orders.sell, vbt.Orders)
        assert orders.sell.wrapper == orders.wrapper
        record_arrays_close(
            orders['a'].sell.values,
            np.array([
                (2, 2, 0, 1., 3., 0.03, 1), (3, 3, 0, 0.1, 4., 0.004, 1),
                (5, 6, 0, 1., 7., 0.07, 1)
            ], dtype=order_dt)
        )
        record_arrays_close(
            orders['a'].sell.values,
            orders.sell['a'].values
        )
        record_arrays_close(
            orders.sell.values,
            np.array([
                (2, 2, 0, 1., 3., 0.03, 1), (3, 3, 0, 0.1, 4., 0.004, 1),
                (5, 6, 0, 1., 7., 0.07, 1), (7, 0, 1, 1., 1., 0.01, 1),
                (8, 1, 1, 0.1, 2., 0.002, 1), (11, 5, 1, 1., 6., 0.06, 1),
                (13, 7, 1, 2., 8., 0.16, 1), (16, 2, 2, 1., 3., 0.03, 1),
                (17, 3, 2, 0.1, 4., 0.004, 1), (19, 6, 2, 2., 7., 0.14, 1)
            ], dtype=order_dt)
        )

    def test_sell_rate(self):
        assert orders['a'].sell_rate() == 0.42857142857142855
        pd.testing.assert_series_equal(
            orders.sell_rate(),
            pd.Series(
                np.array([0.42857143, 0.57142857, 0.42857143, np.nan]),
                index=close.columns
            ).rename('sell_rate')
        )
        pd.testing.assert_series_equal(
            orders_grouped.sell_rate(),
            pd.Series(
                np.array([0.5, 0.42857143]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('sell_rate')
        )


# ############# trades.py ############# #

trades = vbt.Trades.from_orders(orders)
trades_grouped = vbt.Trades.from_orders(orders_grouped)


class TestTrades:
    def test_records_arr(self):
        record_arrays_close(
            trades.values,
            np.array([
                (0, 0, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 0),
                (1, 0, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 0),
                (2, 0, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 1),
                (3, 0, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 0, 0, 2),
                (4, 1, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, -1.95, -1.7875, 1, 1, 3),
                (5, 1, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, -0.296, -2.71333333, 1, 1, 3),
                (6, 1, 1., 5, 6., 0.06, 6, 7., 0.07, -1.13, -0.18833333, 1, 1, 4),
                (7, 1, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 1, 0, 5),
                (8, 2, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 6),
                (9, 2, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 6),
                (10, 2, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 7),
                (11, 2, 1., 6, 7., 0.07, 7, 8., 0.08, -1.15, -0.16428571, 1, 1, 8),
                (12, 2, 1., 7, 8., 0.08, 7, 8., 0., -0.08, -0.01, 0, 0, 9)
            ], dtype=trade_dt)
        )
        reversed_col_orders = orders.copy(records_arr=np.concatenate((
            orders.values[orders.values['col'] == 2],
            orders.values[orders.values['col'] == 1],
            orders.values[orders.values['col'] == 0]
        )))
        record_arrays_close(
            vbt.Trades.from_orders(reversed_col_orders).values,
            trades.values
        )

    def test_records_readable(self):
        records_readable = trades.records_readable

        np.testing.assert_array_equal(
            records_readable['Trade Id'].values,
            np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Column'].values,
            np.array([
                'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c'
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Size'].values,
            np.array([
                1.0, 0.10000000000000009, 1.0, 2.0, 1.0, 0.10000000000000009, 1.0,
                2.0, 1.0, 0.10000000000000009, 1.0, 1.0, 1.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Entry Date'].values,
            np.array([
                '2020-01-01T00:00:00.000000000', '2020-01-01T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-01T00:00:00.000000000', '2020-01-01T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-01T00:00:00.000000000', '2020-01-01T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-07T00:00:00.000000000',
                '2020-01-08T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable['Avg. Entry Price'].values,
            np.array([
                1.0909090909090908, 1.0909090909090908, 6.0, 8.0,
                1.0909090909090908, 1.0909090909090908, 6.0, 8.0,
                1.0909090909090908, 1.0909090909090908, 6.0, 7.0, 8.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Entry Fees'].values,
            np.array([
                0.010909090909090908, 0.0010909090909090918, 0.06, 0.16,
                0.010909090909090908, 0.0010909090909090918, 0.06, 0.16,
                0.010909090909090908, 0.0010909090909090918, 0.06, 0.07, 0.08
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Exit Date'].values,
            np.array([
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-08T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable['Avg. Exit Price'].values,
            np.array([
                3.0, 4.0, 7.0, 8.0, 3.0, 4.0, 7.0, 8.0, 3.0, 4.0, 7.0, 8.0, 8.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Exit Fees'].values,
            np.array([
                0.03, 0.004, 0.07, 0.0, 0.03, 0.004, 0.07, 0.0, 0.03, 0.004, 0.07, 0.08, 0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['PnL'].values,
            np.array([
                1.8681818181818182, 0.2858181818181821, 0.8699999999999999, -0.16,
                -1.9500000000000002, -0.29600000000000026, -1.1300000000000001,
                -0.16, 1.8681818181818182, 0.2858181818181821, 0.8699999999999999,
                -1.1500000000000001, -0.08
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Return'].values,
            np.array([
                1.7125000000000001, 2.62, 0.145, -0.01, -1.7875000000000003,
                -2.7133333333333334, -0.18833333333333335, -0.01,
                1.7125000000000001, 2.62, 0.145, -0.1642857142857143, -0.01
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Direction'].values,
            np.array([
                'Long', 'Long', 'Long', 'Long', 'Short', 'Short', 'Short',
                'Short', 'Long', 'Long', 'Long', 'Short', 'Long'
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Status'].values,
            np.array([
                'Closed', 'Closed', 'Closed', 'Open', 'Closed', 'Closed', 'Closed',
                'Open', 'Closed', 'Closed', 'Closed', 'Closed', 'Open'
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Position Id'].values,
            np.array([
                0, 0, 1, 2, 3, 3, 4, 5, 6, 6, 7, 8, 9
            ])
        )

    def test_duration(self):
        np.testing.assert_array_almost_equal(
            trades['a'].duration.values,
            np.array([2., 3., 1., 0.])
        )
        np.testing.assert_array_almost_equal(
            trades.duration.values,
            np.array([2., 3., 1., 0., 2., 3., 1., 0., 2., 3., 1., 1., 0.])
        )

    def test_pnl(self):
        np.testing.assert_array_almost_equal(
            trades['a'].pnl.values,
            np.array([1.86818182, 0.28581818, 0.87, -0.16])
        )
        np.testing.assert_array_almost_equal(
            trades.pnl.values,
            np.array([
                1.86818182, 0.28581818, 0.87, -0.16, -1.95,
                -0.296, -1.13, -0.16, 1.86818182, 0.28581818,
                0.87, -1.15, -0.08
            ])
        )

    def test_returns(self):
        np.testing.assert_array_almost_equal(
            trades['a'].returns.values,
            np.array([1.7125, 2.62, 0.145, -0.01])
        )
        np.testing.assert_array_almost_equal(
            trades.returns.values,
            np.array([
                1.7125, 2.62, 0.145, -0.01, -1.7875,
                -2.71333333, -0.18833333, -0.01, 1.7125, 2.62,
                0.145, -0.16428571, -0.01
            ])
        )

    def test_winning_records(self):
        assert isinstance(trades.winning, vbt.Trades)
        assert trades.winning.wrapper == trades.wrapper
        record_arrays_close(
            trades['a'].winning.values,
            np.array([
                (0, 0, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 0),
                (1, 0, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 0),
                (2, 0, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 1)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            trades['a'].winning.values,
            trades.winning['a'].values
        )
        record_arrays_close(
            trades.winning.values,
            np.array([
                (0, 0, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 0),
                (1, 0, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 0),
                (2, 0, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 1),
                (8, 2, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 6),
                (9, 2, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 6),
                (10, 2, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 7)
            ], dtype=trade_dt)
        )

    def test_win_rate(self):
        assert trades['a'].win_rate() == 0.75
        pd.testing.assert_series_equal(
            trades.win_rate(),
            pd.Series(
                np.array([0.75, 0., 0.6, np.nan]),
                index=close.columns
            ).rename('win_rate')
        )
        pd.testing.assert_series_equal(
            trades_grouped.win_rate(),
            pd.Series(
                np.array([0.375, 0.6]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('win_rate')
        )

    def test_losing_records(self):
        assert isinstance(trades.losing, vbt.Trades)
        assert trades.losing.wrapper == trades.wrapper
        record_arrays_close(
            trades['a'].losing.values,
            np.array([
                (3, 0, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 0, 0, 2)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            trades['a'].losing.values,
            trades.losing['a'].values
        )
        record_arrays_close(
            trades.losing.values,
            np.array([
                (3, 0, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 0, 0, 2),
                (4, 1, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, -1.95, -1.7875, 1, 1, 3),
                (5, 1, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, -0.296, -2.71333333, 1, 1, 3),
                (6, 1, 1., 5, 6., 0.06, 6, 7., 0.07, -1.13, -0.18833333, 1, 1, 4),
                (7, 1, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 1, 0, 5),
                (11, 2, 1., 6, 7., 0.07, 7, 8., 0.08, -1.15, -0.16428571, 1, 1, 8),
                (12, 2, 1., 7, 8., 0.08, 7, 8., 0., -0.08, -0.01, 0, 0, 9)
            ], dtype=trade_dt)
        )

    def test_loss_rate(self):
        assert trades['a'].loss_rate() == 0.25
        pd.testing.assert_series_equal(
            trades.loss_rate(),
            pd.Series(
                np.array([0.25, 1., 0.4, np.nan]),
                index=close.columns
            ).rename('loss_rate')
        )
        pd.testing.assert_series_equal(
            trades_grouped.loss_rate(),
            pd.Series(
                np.array([0.625, 0.4]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('loss_rate')
        )

    def test_profit_factor(self):
        assert trades['a'].profit_factor() == 18.9
        pd.testing.assert_series_equal(
            trades.profit_factor(),
            pd.Series(
                np.array([18.9, 0., 2.45853659, np.nan]),
                index=ts.columns
            ).rename('profit_factor')
        )
        pd.testing.assert_series_equal(
            trades_grouped.profit_factor(),
            pd.Series(
                np.array([0.81818182, 2.45853659]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('profit_factor')
        )

    def test_expectancy(self):
        assert trades['a'].expectancy() == 0.716
        pd.testing.assert_series_equal(
            trades.expectancy(),
            pd.Series(
                np.array([0.716, -0.884, 0.3588, np.nan]),
                index=ts.columns
            ).rename('expectancy')
        )
        pd.testing.assert_series_equal(
            trades_grouped.expectancy(),
            pd.Series(
                np.array([-0.084, 0.3588]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('expectancy')
        )

    def test_sqn(self):
        assert trades['a'].sqn() == 1.634155521947584
        pd.testing.assert_series_equal(
            trades.sqn(),
            pd.Series(
                np.array([1.63415552, -2.13007307, 0.71660403, np.nan]),
                index=ts.columns
            ).rename('sqn')
        )
        pd.testing.assert_series_equal(
            trades_grouped.sqn(),
            pd.Series(
                np.array([-0.20404671, 0.71660403]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('sqn')
        )

    def test_direction(self):
        np.testing.assert_array_almost_equal(
            trades['a'].direction.values,
            np.array([0, 0, 0, 0])
        )
        np.testing.assert_array_almost_equal(
            trades.direction.values,
            np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0])
        )

    def test_long_records(self):
        assert isinstance(trades.long, vbt.Trades)
        assert trades.long.wrapper == trades.wrapper
        record_arrays_close(
            trades['a'].long.values,
            np.array([
                (0, 0, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 0),
                (1, 0, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 0),
                (2, 0, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 1),
                (3, 0, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 0, 0, 2)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            trades['a'].long.values,
            trades.long['a'].values
        )
        record_arrays_close(
            trades.long.values,
            np.array([
                (0, 0, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 0),
                (1, 0, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 0),
                (2, 0, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 1),
                (3, 0, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 0, 0, 2),
                (8, 2, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 6),
                (9, 2, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 6),
                (10, 2, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 7),
                (12, 2, 1., 7, 8., 0.08, 7, 8., 0., -0.08, -0.01, 0, 0, 9)
            ], dtype=trade_dt)
        )

    def test_long_rate(self):
        assert trades['a'].long_rate() == 1.0
        pd.testing.assert_series_equal(
            trades.long_rate(),
            pd.Series(
                np.array([1., 0., 0.8, np.nan]),
                index=close.columns
            ).rename('long_rate')
        )
        pd.testing.assert_series_equal(
            trades_grouped.long_rate(),
            pd.Series(
                np.array([0.5, 0.8]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('long_rate')
        )

    def test_short_records(self):
        assert isinstance(trades.short, vbt.Trades)
        assert trades.short.wrapper == trades.wrapper
        record_arrays_close(
            trades['a'].short.values,
            np.array([], dtype=trade_dt)
        )
        record_arrays_close(
            trades['a'].short.values,
            trades.short['a'].values
        )
        record_arrays_close(
            trades.short.values,
            np.array([
                (4, 1, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, -1.95, -1.7875, 1, 1, 3),
                (5, 1, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, -0.296, -2.71333333, 1, 1, 3),
                (6, 1, 1., 5, 6., 0.06, 6, 7., 0.07, -1.13, -0.18833333, 1, 1, 4),
                (7, 1, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 1, 0, 5),
                (11, 2, 1., 6, 7., 0.07, 7, 8., 0.08, -1.15, -0.16428571, 1, 1, 8)
            ], dtype=trade_dt)
        )

    def test_short_rate(self):
        assert trades['a'].short_rate() == 0.
        pd.testing.assert_series_equal(
            trades.short_rate(),
            pd.Series(
                np.array([0., 1., 0.2, np.nan]),
                index=close.columns
            ).rename('short_rate')
        )
        pd.testing.assert_series_equal(
            trades_grouped.short_rate(),
            pd.Series(
                np.array([0.5, 0.2]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('short_rate')
        )

    def test_status(self):
        np.testing.assert_array_almost_equal(
            trades['a'].status.values,
            np.array([1, 1, 1, 0])
        )
        np.testing.assert_array_almost_equal(
            trades.status.values,
            np.array([1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0])
        )

    def test_open_records(self):
        assert isinstance(trades.open, vbt.Trades)
        assert trades.open.wrapper == trades.wrapper
        record_arrays_close(
            trades['a'].open.values,
            np.array([
                (3, 0, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 0, 0, 2)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            trades['a'].open.values,
            trades.open['a'].values
        )
        record_arrays_close(
            trades.open.values,
            np.array([
                (3, 0, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 0, 0, 2),
                (7, 1, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 1, 0, 5),
                (12, 2, 1., 7, 8., 0.08, 7, 8., 0., -0.08, -0.01, 0, 0, 9)
            ], dtype=trade_dt)
        )

    def test_open_rate(self):
        assert trades['a'].open_rate() == 0.25
        pd.testing.assert_series_equal(
            trades.open_rate(),
            pd.Series(
                np.array([0.25, 0.25, 0.2, np.nan]),
                index=close.columns
            ).rename('open_rate')
        )
        pd.testing.assert_series_equal(
            trades_grouped.open_rate(),
            pd.Series(
                np.array([0.25, 0.2]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('open_rate')
        )

    def test_closed_records(self):
        assert isinstance(trades.closed, vbt.Trades)
        assert trades.closed.wrapper == trades.wrapper
        record_arrays_close(
            trades['a'].closed.values,
            np.array([
                (0, 0, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 0),
                (1, 0, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 0),
                (2, 0, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 1)
            ], dtype=trade_dt)
        )
        record_arrays_close(
            trades['a'].closed.values,
            trades.closed['a'].values
        )
        record_arrays_close(
            trades.closed.values,
            np.array([
                (0, 0, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 0),
                (1, 0, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 0),
                (2, 0, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 1),
                (4, 1, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, -1.95, -1.7875, 1, 1, 3),
                (5, 1, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, -0.296, -2.71333333, 1, 1, 3),
                (6, 1, 1., 5, 6., 0.06, 6, 7., 0.07, -1.13, -0.18833333, 1, 1, 4),
                (8, 2, 1., 0, 1.09090909, 0.01090909, 2, 3., 0.03, 1.86818182, 1.7125, 0, 1, 6),
                (9, 2, 0.1, 0, 1.09090909, 0.00109091, 3, 4., 0.004, 0.28581818, 2.62, 0, 1, 6),
                (10, 2, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1, 7),
                (11, 2, 1., 6, 7., 0.07, 7, 8., 0.08, -1.15, -0.16428571, 1, 1, 8)
            ], dtype=trade_dt)
        )

    def test_closed_rate(self):
        assert trades['a'].closed_rate() == 0.75
        pd.testing.assert_series_equal(
            trades.closed_rate(),
            pd.Series(
                np.array([0.75, 0.75, 0.8, np.nan]),
                index=close.columns
            ).rename('closed_rate')
        )
        pd.testing.assert_series_equal(
            trades_grouped.closed_rate(),
            pd.Series(
                np.array([0.75, 0.8]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('closed_rate')
        )


positions = vbt.Positions.from_trades(trades)
positions_grouped = vbt.Positions.from_trades(trades_grouped)


class TestPositions:
    def test_records_arr(self):
        record_arrays_close(
            positions.values,
            np.array([
                (0, 0, 1.1, 0, 1.09090909, 0.012, 3, 3.09090909, 0.034, 2.154, 1.795, 0, 1),
                (1, 0, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1),
                (2, 0, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 0, 0),
                (3, 1, 1.1, 0, 1.09090909, 0.012, 3, 3.09090909, 0.034, -2.246, -1.87166667, 1, 1),
                (4, 1, 1., 5, 6., 0.06, 6, 7., 0.07, -1.13, -0.18833333, 1, 1),
                (5, 1, 2., 7, 8., 0.16, 7, 8., 0., -0.16, -0.01, 1, 0),
                (6, 2, 1.1, 0, 1.09090909, 0.012, 3, 3.09090909, 0.034, 2.154, 1.795, 0, 1),
                (7, 2, 1., 5, 6., 0.06, 6, 7., 0.07, 0.87, 0.145, 0, 1),
                (8, 2, 1., 6, 7., 0.07, 7, 8., 0.08, -1.15, -0.16428571, 1, 1),
                (9, 2, 1., 7, 8., 0.08, 7, 8., 0., -0.08, -0.01, 0, 0)
            ], dtype=position_dt)
        )
        reversed_col_trades = trades.copy(records_arr=np.concatenate((
            trades.values[trades.values['col'] == 2],
            trades.values[trades.values['col'] == 1],
            trades.values[trades.values['col'] == 0]
        )))
        record_arrays_close(
            vbt.Positions.from_trades(reversed_col_trades).values,
            positions.values
        )

    def test_records_readable(self):
        records_readable = positions.records_readable

        np.testing.assert_array_equal(
            records_readable['Position Id'].values,
            np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Column'].values,
            np.array([
                'a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c', 'c'
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Size'].values,
            np.array([
                1.1, 1.0, 2.0, 1.1, 1.0, 2.0, 1.1, 1.0, 1.0, 1.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Entry Date'].values,
            np.array([
                '2020-01-01T00:00:00.000000000', '2020-01-06T00:00:00.000000000',
                '2020-01-08T00:00:00.000000000', '2020-01-01T00:00:00.000000000',
                '2020-01-06T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-01T00:00:00.000000000', '2020-01-06T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable['Avg. Entry Price'].values,
            np.array([
                1.0909090909090908, 6.0, 8.0, 1.0909090909090908, 6.0, 8.0, 1.0909090909090908, 6.0, 7.0, 8.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Entry Fees'].values,
            np.array([
                0.012, 0.06, 0.16, 0.012, 0.06, 0.16, 0.012, 0.06, 0.07, 0.08
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Exit Date'].values,
            np.array([
                '2020-01-04T00:00:00.000000000', '2020-01-07T00:00:00.000000000',
                '2020-01-08T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-04T00:00:00.000000000', '2020-01-07T00:00:00.000000000',
                '2020-01-08T00:00:00.000000000', '2020-01-08T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable['Avg. Exit Price'].values,
            np.array([
                3.090909090909091, 7.0, 8.0, 3.090909090909091, 7.0, 8.0, 3.090909090909091, 7.0, 8.0, 8.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Exit Fees'].values,
            np.array([
                0.034, 0.07, 0.0, 0.034, 0.07, 0.0, 0.034, 0.07, 0.08, 0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable['PnL'].values,
            np.array([
                2.1540000000000004, 0.8699999999999999, -0.16, -2.246, -1.1300000000000001,
                -0.16, 2.1540000000000004, 0.8699999999999999, -1.1500000000000001, -0.08
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Return'].values,
            np.array([
                1.7950000000000004, 0.145, -0.01, -1.8716666666666668, -0.18833333333333335,
                -0.01, 1.7950000000000004, 0.145, -0.1642857142857143, -0.01
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Direction'].values,
            np.array([
                'Long', 'Long', 'Long', 'Short', 'Short', 'Short', 'Long', 'Long', 'Short', 'Long'
            ])
        )
        np.testing.assert_array_equal(
            records_readable['Status'].values,
            np.array([
                'Closed', 'Closed', 'Open', 'Closed', 'Closed', 'Open', 'Closed', 'Closed', 'Closed', 'Open'
            ])
        )

    def test_coverage(self):
        assert positions['a'].coverage() == 0.5
        pd.testing.assert_series_equal(
            positions.coverage(),
            pd.Series(
                np.array([0.5, 0.5, 0.625, 0.]),
                index=close.columns
            ).rename('coverage')
        )
        pd.testing.assert_series_equal(
            positions_grouped.coverage(),
            pd.Series(
                np.array([0.5, 0.3125]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('coverage')
        )


# ############# logs.py ############# #
logs = vbt.Portfolio.from_orders(close, size, fees=0.01, log=True, freq='1 days').logs
logs_grouped = logs.regroup(group_by)


class TestLogs:
    def test_records_readable(self):
        records_readable = logs.records_readable

        np.testing.assert_array_equal(
            records_readable[('Context', 'Log Id')].values,
            np.array([
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27,
                28, 29, 30, 31
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Date')].values,
            np.array([
                '2020-01-01T00:00:00.000000000', '2020-01-02T00:00:00.000000000',
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-05T00:00:00.000000000', '2020-01-06T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-01T00:00:00.000000000', '2020-01-02T00:00:00.000000000',
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-05T00:00:00.000000000', '2020-01-06T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-01T00:00:00.000000000', '2020-01-02T00:00:00.000000000',
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-05T00:00:00.000000000', '2020-01-06T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000',
                '2020-01-01T00:00:00.000000000', '2020-01-02T00:00:00.000000000',
                '2020-01-03T00:00:00.000000000', '2020-01-04T00:00:00.000000000',
                '2020-01-05T00:00:00.000000000', '2020-01-06T00:00:00.000000000',
                '2020-01-07T00:00:00.000000000', '2020-01-08T00:00:00.000000000'
            ], dtype='datetime64[ns]')
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Column')].values,
            np.array([
                'a', 'a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c', 'c',
                'c', 'c', 'c', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd'
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Group')].values,
            np.array([
                0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Cash')].values,
            np.array([
                100.0, 98.99, 98.788, 101.758, 102.154, 102.154, 96.094, 103.024, 100.0, 100.99, 101.18799999999999,
                98.15799999999999, 97.75399999999999, 97.75399999999999, 103.69399999999999, 96.624, 100.0, 98.99,
                98.788, 101.758, 102.154, 102.154, 96.094, 109.954, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                100.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Position')].values,
            np.array([
                0.0, 1.0, 1.1, 0.10000000000000009, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, -1.1, -0.10000000000000009, 0.0, 0.0,
                -1.0, 0.0, 0.0, 1.0, 1.1, 0.10000000000000009, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Debt')].values,
            np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.2, 0.10909090909090913, 0.0, 0.0, 6.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Free Cash')].values,
            np.array([
                100.0, 98.99, 98.788, 101.758, 102.154, 102.154, 96.094, 103.024, 100.0, 98.99, 98.788,
                97.93981818181818, 97.754, 97.754, 91.694, 96.624, 100.0, 98.99, 98.788, 101.758, 102.154, 102.154,
                96.094, 95.954, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Val Price')].values,
            np.array([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Context', 'Value')].values,
            np.array([
                100.0, 100.99, 102.088, 102.158, 102.154, 102.154, 103.094, 103.024, 100.0, 98.99, 97.88799999999999,
                97.75799999999998, 97.75399999999999, 97.75399999999999, 96.69399999999999, 96.624, 100.0, 100.99,
                102.088, 102.158, 102.154, 102.154, 103.094, 101.954, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                100.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Size')].values,
            np.array([
                1.0, 0.1, -1.0, -0.1, np.nan, 1.0, -1.0, 2.0, -1.0, -0.1, 1.0, 0.1, np.nan, -1.0, 1.0, -2.0, 1.0, 0.1,
                -1.0, -0.1, np.nan, 1.0, -2.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Price')].values,
            np.array([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Size Type')].values,
            np.array([
                'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount',
                'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount',
                'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount', 'Amount',
                'Amount', 'Amount'
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Direction')].values,
            np.array([
                'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All',
                'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All', 'All',
                'All', 'All'
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Fees')].values,
            np.array([
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01,
                0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Fixed Fees')].values,
            np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Slippage')].values,
            np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Min Size')].values,
            np.array([
                1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08,
                1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08, 1e-08,
                1e-08, 1e-08
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Max Size')].values,
            np.array([
                np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                np.inf, np.inf, np.inf, np.inf, np.inf, np.inf
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Rejection Prob')].values,
            np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Allow Partial')].values,
            np.array([
                True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True, True, True, True
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Raise Rejection')].values,
            np.array([
                False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                False, False, False, False, False, False, False, False, False, False, False, False, False, False, False,
                False, False
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order', 'Log')].values,
            np.array([
                True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True,
                True, True, True, True, True, True, True, True, True, True, True, True, True, True, True
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('New Context', 'Cash')].values,
            np.array([
                98.99, 98.788, 101.758, 102.154, 102.154, 96.094, 103.024, 86.864, 100.99, 101.18799999999999,
                98.15799999999999, 97.75399999999999, 97.75399999999999, 103.69399999999999, 96.624, 112.464, 98.99,
                98.788, 101.758, 102.154, 102.154, 96.094, 109.954, 93.794, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                100.0, 100.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('New Context', 'Position')].values,
            np.array([
                1.0, 1.1, 0.10000000000000009, 0.0, 0.0, 1.0, 0.0, 2.0, -1.0, -1.1, -0.10000000000000009, 0.0, 0.0,
                -1.0, 0.0, -2.0, 1.0, 1.1, 0.10000000000000009, 0.0, 0.0, 1.0, -1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('New Context', 'Debt')].values,
            np.array([
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.2, 0.10909090909090913, 0.0, 0.0, 6.0, 0.0, 16.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('New Context', 'Free Cash')].values,
            np.array([
                98.99, 98.788, 101.758, 102.154, 102.154, 96.094, 103.024, 86.864, 98.99, 98.788, 97.93981818181818,
                97.754, 97.754, 91.694, 96.624, 80.464, 98.99, 98.788, 101.758, 102.154, 102.154, 96.094, 95.954,
                93.794, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('New Context', 'Val Price')].values,
            np.array([
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('New Context', 'Value')].values,
            np.array([
                100.0, 100.99, 102.088, 102.158, 102.154, 102.154, 103.094, 103.024, 100.0, 98.99, 97.88799999999999,
                97.75799999999998, 97.75399999999999, 97.75399999999999, 96.69399999999999, 96.624, 100.0, 100.99,
                102.088, 102.158, 102.154, 102.154, 103.094, 101.954, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0,
                100.0
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order Result', 'Size')].values,
            np.array([
                1.0, 0.1, 1.0, 0.1, np.nan, 1.0, 1.0, 2.0, 1.0, 0.1, 1.0, 0.1, np.nan, 1.0, 1.0, 2.0, 1.0, 0.1, 1.0,
                0.1, np.nan, 1.0, 2.0, 2.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order Result', 'Price')].values,
            np.array([
                1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0,
                4.0, np.nan, 6.0, 7.0, 8.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order Result', 'Fees')].values,
            np.array([
                0.01, 0.002, 0.03, 0.004, np.nan, 0.06, 0.07, 0.16, 0.01, 0.002, 0.03, 0.004, np.nan, 0.06, 0.07, 0.16,
                0.01, 0.002, 0.03, 0.004, np.nan, 0.06, 0.14, 0.16, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
                np.nan, np.nan
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order Result', 'Side')].values,
            np.array([
                'Buy', 'Buy', 'Sell', 'Sell', None, 'Buy', 'Sell', 'Buy', 'Sell', 'Sell', 'Buy', 'Buy', None, 'Sell',
                'Buy', 'Sell', 'Buy', 'Buy', 'Sell', 'Sell', None, 'Buy', 'Sell', 'Buy', None, None, None, None, None,
                None, None, None
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order Result', 'Status')].values,
            np.array([
                'Filled', 'Filled', 'Filled', 'Filled', 'Ignored', 'Filled', 'Filled', 'Filled', 'Filled', 'Filled',
                'Filled', 'Filled', 'Ignored', 'Filled', 'Filled', 'Filled', 'Filled', 'Filled', 'Filled', 'Filled',
                'Ignored', 'Filled', 'Filled', 'Filled', 'Ignored', 'Ignored', 'Ignored', 'Ignored', 'Ignored',
                'Ignored', 'Ignored', 'Ignored'
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order Result', 'Status Info')].values,
            np.array([
                None, None, None, None, 'SizeNaN', None, None, None, None, None, None, None, 'SizeNaN', None, None,
                None, None, None, None, None, 'SizeNaN', None, None, None, 'SizeNaN', 'SizeNaN', 'SizeNaN', 'SizeNaN',
                'SizeNaN', 'SizeNaN', 'SizeNaN', 'SizeNaN'
            ])
        )
        np.testing.assert_array_equal(
            records_readable[('Order Result', 'Order Id')].values,
            np.array([
                0, 1, 2, 3, -1, 4, 5, 6, 7, 8, 9, 10, -1, 11, 12, 13, 14, 15, 16, 17, -1, 18, 19, 20, -1, -1, -1, -1,
                -1, -1, -1, -1
            ])
        )

    def test_count(self):
        assert logs['a'].count() == 8
        pd.testing.assert_series_equal(
            logs.count(),
            pd.Series(
                np.array([8, 8, 8, 8]),
                index=pd.Index(['a', 'b', 'c', 'd'], dtype='object')
            ).rename('count')
        )
        pd.testing.assert_series_equal(
            logs_grouped.count(),
            pd.Series(
                np.array([16, 16]),
                index=pd.Index(['g1', 'g2'], dtype='object')
            ).rename('count')
        )
