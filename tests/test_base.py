import numpy as np
import pandas as pd
from numba import njit
import pytest

from vectorbt import defaults
from vectorbt.base import (
    accessors,
    array_wrapper,
    combine_fns,
    common,
    index_fns,
    indexing,
    reshape_fns
)

defaults.broadcasting['index_from'] = 'stack'
defaults.broadcasting['columns_from'] = 'stack'

# Initialize global variables
a1 = np.array([1])
a2 = np.array([1, 2, 3])
a3 = np.array([[1, 2, 3]])
a4 = np.array([[1], [2], [3]])
a5 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
sr_none = pd.Series([1])
sr1 = pd.Series([1], index=pd.Index(['x1'], name='i1'), name='a1')
sr2 = pd.Series([1, 2, 3], index=pd.Index(['x2', 'y2', 'z2'], name='i2'), name='a2')
df_none = pd.DataFrame([[1]])
df1 = pd.DataFrame(
    [[1]],
    index=pd.Index(['x3'], name='i3'),
    columns=pd.Index(['a3'], name='c3'))
df2 = pd.DataFrame(
    [[1], [2], [3]],
    index=pd.Index(['x4', 'y4', 'z4'], name='i4'),
    columns=pd.Index(['a4'], name='c4'))
df3 = pd.DataFrame(
    [[1, 2, 3]],
    index=pd.Index(['x5'], name='i5'),
    columns=pd.Index(['a5', 'b5', 'c5'], name='c5'))
df4 = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
    index=pd.Index(['x6', 'y6', 'z6'], name='i6'),
    columns=pd.Index(['a6', 'b6', 'c6'], name='c6'))
multi_i = pd.MultiIndex.from_arrays([['x7', 'y7', 'z7'], ['x8', 'y8', 'z8']], names=['i7', 'i8'])
multi_c = pd.MultiIndex.from_arrays([['a7', 'b7', 'c7'], ['a8', 'b8', 'c8']], names=['c7', 'c8'])
df5 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], index=multi_i, columns=multi_c)


# ############# array_wrapper.py ############# #


class TestArrayWrapper:

    def test_wrap(self):
        pd.testing.assert_series_equal(
            array_wrapper.ArrayWrapper(index=sr1.index, columns=[0], ndim=1).wrap(a1),  # empty
            pd.Series(a1, index=sr1.index, name=None)
        )
        pd.testing.assert_series_equal(
            array_wrapper.ArrayWrapper(index=sr1.index, columns=[sr1.name], ndim=1).wrap(a1),
            pd.Series(a1, index=sr1.index, name=sr1.name)
        )
        pd.testing.assert_frame_equal(
            array_wrapper.ArrayWrapper(index=sr1.index, columns=[sr1.name], ndim=2).wrap(a1),
            pd.DataFrame(a1, index=sr1.index, columns=[sr1.name])
        )
        pd.testing.assert_series_equal(
            array_wrapper.ArrayWrapper(index=sr2.index, columns=[sr2.name], ndim=1).wrap(a2),
            pd.Series(a2, index=sr2.index, name=sr2.name)
        )
        pd.testing.assert_frame_equal(
            array_wrapper.ArrayWrapper(index=sr2.index, columns=[sr2.name], ndim=2).wrap(a2),
            pd.DataFrame(a2, index=sr2.index, columns=[sr2.name])
        )
        pd.testing.assert_series_equal(
            array_wrapper.ArrayWrapper(index=df2.index, columns=df2.columns, ndim=1).wrap(a2),
            pd.Series(a2, index=df2.index, name=df2.columns[0])
        )
        pd.testing.assert_frame_equal(
            array_wrapper.ArrayWrapper(index=df2.index, columns=df2.columns, ndim=2).wrap(a2),
            pd.DataFrame(a2, index=df2.index, columns=df2.columns)
        )
        pd.testing.assert_frame_equal(
            array_wrapper.ArrayWrapper.from_obj(df2).wrap(a2, index=df4.index),
            pd.DataFrame(a2, index=df4.index, columns=df2.columns)
        )

    def test_wrap_reduced(self):
        sr_wrapper = array_wrapper.ArrayWrapper.from_obj(sr2)
        df_wrapper = array_wrapper.ArrayWrapper.from_obj(df4)
        # sr to value
        assert df_wrapper.wrap_reduced(0) == 0
        assert sr_wrapper.wrap_reduced(np.array([0])) == 0  # result of computation on 2d
        # sr to array
        pd.testing.assert_series_equal(
            sr_wrapper.wrap_reduced(np.array([0, 1]), index=['x', 'y']),
            pd.Series(np.array([0, 1]), index=['x', 'y'], name=sr2.name)
        )
        # df to value
        assert sr_wrapper.wrap_reduced(0) == 0
        # df to value per column
        pd.testing.assert_series_equal(
            df_wrapper.wrap_reduced(np.array([0, 1, 2])),
            pd.Series(np.array([0, 1, 2]), index=df4.columns)
        )
        # df to array per column
        pd.testing.assert_frame_equal(
            df_wrapper.wrap_reduced(np.array([[0, 1, 2], [3, 4, 5]]), index=['x', 'y']),
            pd.DataFrame(np.array([[0, 1, 2], [3, 4, 5]]), index=['x', 'y'], columns=df4.columns)
        )

    def test_eq(self):
        assert array_wrapper.ArrayWrapper.from_obj(sr2) == array_wrapper.ArrayWrapper.from_obj(sr2)
        assert array_wrapper.ArrayWrapper.from_obj(df2) == array_wrapper.ArrayWrapper.from_obj(df2)
        assert array_wrapper.ArrayWrapper.from_obj(sr2) != array_wrapper.ArrayWrapper.from_obj(df2)
        assert array_wrapper.ArrayWrapper.from_obj(sr2) != [0, 1, 2]


# ############# index_fns.py ############# #

class TestIndexFns:
    def test_get_index(self):
        pd.testing.assert_index_equal(index_fns.get_index(sr1, 0), sr1.index)
        pd.testing.assert_index_equal(index_fns.get_index(sr1, 1), pd.Index([sr1.name]))
        pd.testing.assert_index_equal(index_fns.get_index(pd.Series([1, 2, 3]), 1), pd.Index([0]))  # empty
        pd.testing.assert_index_equal(index_fns.get_index(df1, 0), df1.index)
        pd.testing.assert_index_equal(index_fns.get_index(df1, 1), df1.columns)

    def test_index_from_values(self):
        pd.testing.assert_index_equal(
            index_fns.index_from_values([0.1, 0.2], name='a'),
            pd.Float64Index([0.1, 0.2], dtype='float64', name='a')
        )
        pd.testing.assert_index_equal(
            index_fns.index_from_values(np.tile(np.arange(1, 4)[:, None][:, None], (1, 3, 3)), name='b'),
            pd.Int64Index([1, 2, 3], dtype='int64', name='b')
        )
        pd.testing.assert_index_equal(
            index_fns.index_from_values(np.random.uniform(size=(3, 3, 3)), name='c'),
            pd.Index(['mix_0', 'mix_1', 'mix_2'], dtype='object', name='c')
        )

    def test_repeat_index(self):
        i = pd.Int64Index([1, 2, 3], name='i')
        pd.testing.assert_index_equal(
            index_fns.repeat_index(i, 3),
            pd.Int64Index([1, 1, 1, 2, 2, 2, 3, 3, 3], dtype='int64', name='i')
        )
        pd.testing.assert_index_equal(
            index_fns.repeat_index(multi_i, 3),
            pd.MultiIndex.from_tuples([
                ('x7', 'x8'),
                ('x7', 'x8'),
                ('x7', 'x8'),
                ('y7', 'y8'),
                ('y7', 'y8'),
                ('y7', 'y8'),
                ('z7', 'z8'),
                ('z7', 'z8'),
                ('z7', 'z8')
            ], names=['i7', 'i8'])
        )
        pd.testing.assert_index_equal(
            index_fns.repeat_index([0], 3),  # empty
            pd.Int64Index([0, 1, 2], dtype='int64')
        )
        pd.testing.assert_index_equal(
            index_fns.repeat_index(sr_none.index, 3),  # simple range
            pd.RangeIndex(start=0, stop=3, step=1)
        )

    def test_tile_index(self):
        i = pd.Int64Index([1, 2, 3], name='i')
        pd.testing.assert_index_equal(
            index_fns.tile_index(i, 3),
            pd.Int64Index([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype='int64', name='i')
        )
        pd.testing.assert_index_equal(
            index_fns.tile_index(multi_i, 3),
            pd.MultiIndex.from_tuples([
                ('x7', 'x8'),
                ('y7', 'y8'),
                ('z7', 'z8'),
                ('x7', 'x8'),
                ('y7', 'y8'),
                ('z7', 'z8'),
                ('x7', 'x8'),
                ('y7', 'y8'),
                ('z7', 'z8')
            ], names=['i7', 'i8'])
        )
        pd.testing.assert_index_equal(
            index_fns.tile_index([0], 3),  # empty
            pd.Int64Index([0, 1, 2], dtype='int64')
        )
        pd.testing.assert_index_equal(
            index_fns.tile_index(sr_none.index, 3),  # simple range
            pd.RangeIndex(start=0, stop=3, step=1)
        )

    def test_stack_indexes(self):
        pd.testing.assert_index_equal(
            index_fns.stack_indexes(sr2.index, df2.index, df5.index),
            pd.MultiIndex.from_tuples([
                ('x2', 'x4', 'x7', 'x8'),
                ('y2', 'y4', 'y7', 'y8'),
                ('z2', 'z4', 'z7', 'z8')
            ], names=['i2', 'i4', 'i7', 'i8'])
        )

    def test_combine_indexes(self):
        pd.testing.assert_index_equal(
            index_fns.combine_indexes(pd.Index([1]), pd.Index([2, 3]), ignore_single=False),
            pd.MultiIndex.from_tuples([
                (1, 2),
                (1, 3)
            ])
        )
        pd.testing.assert_index_equal(
            index_fns.combine_indexes(pd.Index([1]), pd.Index([2, 3]), ignore_single=True),
            pd.Int64Index([2, 3], dtype='int64')
        )
        pd.testing.assert_index_equal(
            index_fns.combine_indexes(pd.Index([1, 2]), pd.Index([3]), ignore_single=False),
            pd.MultiIndex.from_tuples([
                (1, 3),
                (2, 3)
            ])
        )
        pd.testing.assert_index_equal(
            index_fns.combine_indexes(pd.Index([1, 2]), pd.Index([3]), ignore_single=True),
            pd.Int64Index([1, 2], dtype='int64')
        )
        pd.testing.assert_index_equal(
            index_fns.combine_indexes(pd.Index([1]), pd.Index([2, 3]), ignore_single=(False, True)),
            pd.Int64Index([2, 3], dtype='int64')
        )
        pd.testing.assert_index_equal(
            index_fns.combine_indexes(df2.index, df5.index),
            pd.MultiIndex.from_tuples([
                ('x4', 'x7', 'x8'),
                ('x4', 'y7', 'y8'),
                ('x4', 'z7', 'z8'),
                ('y4', 'x7', 'x8'),
                ('y4', 'y7', 'y8'),
                ('y4', 'z7', 'z8'),
                ('z4', 'x7', 'x8'),
                ('z4', 'y7', 'y8'),
                ('z4', 'z7', 'z8')
            ], names=['i4', 'i7', 'i8'])
        )

    def test_drop_levels(self):
        pd.testing.assert_index_equal(
            index_fns.drop_levels(multi_i, 'i7'),
            pd.Index(['x8', 'y8', 'z8'], dtype='object', name='i8')
        )
        pd.testing.assert_index_equal(
            index_fns.drop_levels(multi_i, 'i8'),
            pd.Index(['x7', 'y7', 'z7'], dtype='object', name='i7')
        )
        pd.testing.assert_index_equal(
            index_fns.drop_levels(multi_i, ['i7', 'i8']),  # won't do anything
            pd.MultiIndex.from_tuples([
                ('x7', 'x8'),
                ('y7', 'y8'),
                ('z7', 'z8')
            ], names=['i7', 'i8'])
        )

    def test_rename_levels(self):
        i = pd.Int64Index([1, 2, 3], name='i')
        pd.testing.assert_index_equal(
            index_fns.rename_levels(i, {'i': 'f'}),
            pd.Int64Index([1, 2, 3], dtype='int64', name='f')
        )
        pd.testing.assert_index_equal(
            index_fns.rename_levels(multi_i, {'i7': 'f7', 'i8': 'f8'}),
            pd.MultiIndex.from_tuples([
                ('x7', 'x8'),
                ('y7', 'y8'),
                ('z7', 'z8')
            ], names=['f7', 'f8'])
        )

    def test_select_levels(self):
        pd.testing.assert_index_equal(
            index_fns.select_levels(multi_i, 'i7'),
            pd.Index(['x7', 'y7', 'z7'], dtype='object', name='i7')
        )
        pd.testing.assert_index_equal(
            index_fns.select_levels(multi_i, ['i7']),
            pd.MultiIndex.from_tuples([
                ('x7',),
                ('y7',),
                ('z7',)
            ], names=['i7'])
        )
        pd.testing.assert_index_equal(
            index_fns.select_levels(multi_i, ['i7', 'i8']),
            pd.MultiIndex.from_tuples([
                ('x7', 'x8'),
                ('y7', 'y8'),
                ('z7', 'z8')
            ], names=['i7', 'i8'])
        )

    def test_drop_redundant_levels(self):
        pd.testing.assert_index_equal(
            index_fns.drop_redundant_levels(pd.Index(['a', 'a'])),  # ignores levels with single element
            pd.Index(['a', 'a'], dtype='object')
        )
        pd.testing.assert_index_equal(
            index_fns.drop_redundant_levels(pd.Index(['a', 'a'], name='hi')),
            pd.Index(['a', 'a'], dtype='object', name='hi')
        )
        pd.testing.assert_index_equal(
            index_fns.drop_redundant_levels(pd.MultiIndex.from_arrays([['a', 'a'], ['b', 'b']], names=['hi', 'hi2'])),
            pd.MultiIndex.from_tuples([
                ('a', 'b'),
                ('a', 'b')
            ], names=['hi', 'hi2'])
        )
        pd.testing.assert_index_equal(
            index_fns.drop_redundant_levels(pd.MultiIndex.from_arrays([['a', 'b'], ['a', 'b']], names=['hi', 'hi2'])),
            pd.MultiIndex.from_tuples([
                ('a', 'a'),
                ('b', 'b')
            ], names=['hi', 'hi2'])
        )
        pd.testing.assert_index_equal(  # ignores 0-to-n
            index_fns.drop_redundant_levels(pd.MultiIndex.from_arrays([[0, 1], ['a', 'b']], names=[None, 'hi2'])),
            pd.Index(['a', 'b'], dtype='object', name='hi2')
        )
        pd.testing.assert_index_equal(  # legit
            index_fns.drop_redundant_levels(pd.MultiIndex.from_arrays([[0, 2], ['a', 'b']], names=[None, 'hi2'])),
            pd.MultiIndex.from_tuples([
                (0, 'a'),
                (2, 'b')
            ], names=[None, 'hi2'])
        )
        pd.testing.assert_index_equal(  # legit (w/ name)
            index_fns.drop_redundant_levels(pd.MultiIndex.from_arrays([[0, 1], ['a', 'b']], names=['hi', 'hi2'])),
            pd.MultiIndex.from_tuples([
                (0, 'a'),
                (1, 'b')
            ], names=['hi', 'hi2'])
        )

    def test_drop_duplicate_levels(self):
        pd.testing.assert_index_equal(
            index_fns.drop_duplicate_levels(pd.MultiIndex.from_arrays(
                [[1, 2, 3], [1, 2, 3]], names=['a', 'a'])),
            pd.Int64Index([1, 2, 3], dtype='int64', name='a')
        )
        pd.testing.assert_index_equal(
            index_fns.drop_duplicate_levels(pd.MultiIndex.from_tuples(
                [(0, 1, 2, 1), ('a', 'b', 'c', 'b')], names=['x', 'y', 'z', 'y']), keep='last'),
            pd.MultiIndex.from_tuples([
                (0, 2, 1),
                ('a', 'c', 'b')
            ], names=['x', 'z', 'y'])
        )
        pd.testing.assert_index_equal(
            index_fns.drop_duplicate_levels(pd.MultiIndex.from_tuples(
                [(0, 1, 2, 1), ('a', 'b', 'c', 'b')], names=['x', 'y', 'z', 'y']), keep='first'),
            pd.MultiIndex.from_tuples([
                (0, 1, 2),
                ('a', 'b', 'c')
            ], names=['x', 'y', 'z'])
        )

    def test_align_index_to(self):
        multi_c1 = pd.MultiIndex.from_arrays([['a8', 'b8']], names=['c8'])
        multi_c2 = pd.MultiIndex.from_arrays([['a7', 'a7', 'c7', 'c7'], ['a8', 'b8', 'a8', 'b8']], names=['c7', 'c8'])
        np.testing.assert_array_equal(index_fns.align_index_to(multi_c1, multi_c2), np.array([0, 1, 0, 1]))


# ############# reshape_fns.py ############# #


class TestReshapeFns:
    def test_soft_broadcast_to_ndim(self):
        np.testing.assert_array_equal(reshape_fns.soft_broadcast_to_ndim(a2, 1), a2)
        pd.testing.assert_series_equal(reshape_fns.soft_broadcast_to_ndim(sr2, 1), sr2)
        pd.testing.assert_series_equal(reshape_fns.soft_broadcast_to_ndim(df2, 1), df2.iloc[:, 0])
        pd.testing.assert_frame_equal(reshape_fns.soft_broadcast_to_ndim(df4, 1), df4)  # cannot -> do nothing
        np.testing.assert_array_equal(reshape_fns.soft_broadcast_to_ndim(a2, 2), a2[:, None])
        pd.testing.assert_frame_equal(reshape_fns.soft_broadcast_to_ndim(sr2, 2), sr2.to_frame())
        pd.testing.assert_frame_equal(reshape_fns.soft_broadcast_to_ndim(df2, 2), df2)

    def test_to_1d(self):
        np.testing.assert_array_equal(reshape_fns.to_1d(None), np.asarray([None]))
        np.testing.assert_array_equal(reshape_fns.to_1d(0), np.asarray([0]))
        np.testing.assert_array_equal(reshape_fns.to_1d(a2), a2)
        pd.testing.assert_series_equal(reshape_fns.to_1d(sr2), sr2)
        pd.testing.assert_series_equal(reshape_fns.to_1d(df2), df2.iloc[:, 0])
        np.testing.assert_array_equal(reshape_fns.to_1d(df2, raw=True), df2.iloc[:, 0].values)

    def test_to_2d(self):
        np.testing.assert_array_equal(reshape_fns.to_2d(None), np.asarray([[None]]))
        np.testing.assert_array_equal(reshape_fns.to_2d(0), np.asarray([[0]]))
        np.testing.assert_array_equal(reshape_fns.to_2d(a2), a2[:, None])
        pd.testing.assert_frame_equal(reshape_fns.to_2d(sr2), sr2.to_frame())
        pd.testing.assert_frame_equal(reshape_fns.to_2d(df2), df2)
        np.testing.assert_array_equal(reshape_fns.to_2d(df2, raw=True), df2.values)

    def test_repeat_axis0(self):
        target = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        np.testing.assert_array_equal(reshape_fns.repeat(0, 3, axis=0), np.full(3, 0))
        np.testing.assert_array_equal(
            reshape_fns.repeat(a2, 3, axis=0),
            target)
        pd.testing.assert_series_equal(
            reshape_fns.repeat(sr2, 3, axis=0),
            pd.Series(target, index=index_fns.repeat_index(sr2.index, 3), name=sr2.name))
        pd.testing.assert_frame_equal(
            reshape_fns.repeat(df2, 3, axis=0),
            pd.DataFrame(target, index=index_fns.repeat_index(df2.index, 3), columns=df2.columns))

    def test_repeat_axis1(self):
        target = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(reshape_fns.repeat(0, 3, axis=1), np.full((1, 3), 0))
        np.testing.assert_array_equal(
            reshape_fns.repeat(a2, 3, axis=1),
            target)
        pd.testing.assert_frame_equal(
            reshape_fns.repeat(sr2, 3, axis=1),
            pd.DataFrame(target, index=sr2.index, columns=index_fns.repeat_index([sr2.name], 3)))
        pd.testing.assert_frame_equal(
            reshape_fns.repeat(df2, 3, axis=1),
            pd.DataFrame(target, index=df2.index, columns=index_fns.repeat_index(df2.columns, 3)))

    def test_tile_axis0(self):
        target = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        np.testing.assert_array_equal(reshape_fns.tile(0, 3, axis=0), np.full(3, 0))
        np.testing.assert_array_equal(
            reshape_fns.tile(a2, 3, axis=0),
            target)
        pd.testing.assert_series_equal(
            reshape_fns.tile(sr2, 3, axis=0),
            pd.Series(target, index=index_fns.tile_index(sr2.index, 3), name=sr2.name))
        pd.testing.assert_frame_equal(
            reshape_fns.tile(df2, 3, axis=0),
            pd.DataFrame(target, index=index_fns.tile_index(df2.index, 3), columns=df2.columns))

    def test_tile_axis1(self):
        target = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
        np.testing.assert_array_equal(reshape_fns.tile(0, 3, axis=1), np.full((1, 3), 0))
        np.testing.assert_array_equal(
            reshape_fns.tile(a2, 3, axis=1),
            target)
        pd.testing.assert_frame_equal(
            reshape_fns.tile(sr2, 3, axis=1),
            pd.DataFrame(target, index=sr2.index, columns=index_fns.tile_index([sr2.name], 3)))
        pd.testing.assert_frame_equal(
            reshape_fns.tile(df2, 3, axis=1),
            pd.DataFrame(target, index=df2.index, columns=index_fns.tile_index(df2.columns, 3)))

    def test_broadcast_numpy(self):
        # 1d
        to_broadcast = 0, a1, a2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshape_fns.broadcast(*to_broadcast)
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(
                broadcasted[i],
                broadcasted_arrs[i]
            )
        # 2d
        to_broadcast = 0, a1, a2, a3, a4, a5
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshape_fns.broadcast(*to_broadcast)
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(
                broadcasted[i],
                broadcasted_arrs[i]
            )

    def test_broadcast_stack(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast,
            index_from='stack',
            columns_from='stack',
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=sr2.index,
                    name=(sr1.name, sr2.name)
                )
            )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(np.broadcast_arrays(
            *to_broadcast_a,
            *[x.to_frame() for x in to_broadcast_sr],  # here is the difference
            *to_broadcast_df
        ))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast_a, *to_broadcast_sr, *to_broadcast_df,
            index_from='stack',
            columns_from='stack',
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples([
                        ('x2', 'x4', 'x6'),
                        ('y2', 'y4', 'y6'),
                        ('z2', 'z4', 'z6')
                    ], names=['i2', 'i4', 'i6']),
                    columns=pd.MultiIndex.from_tuples([
                        ('a5', 'a6'),
                        ('b5', 'b6'),
                        ('c5', 'c6')
                    ], names=['c5', 'c6'])
                )
            )

    def test_broadcast_none(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast,
            index_from=None,
            columns_from=None,
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(4):
            pd.testing.assert_series_equal(
                broadcasted[i],
                pd.Series(broadcasted_arrs[i], index=pd.RangeIndex(start=0, stop=3, step=1))
            )
        pd.testing.assert_series_equal(
            broadcasted[4],
            pd.Series(broadcasted_arrs[4], index=pd.Index(['x1', 'x1', 'x1'], name='i1'), name=sr1.name)
        )
        pd.testing.assert_series_equal(
            broadcasted[5],
            pd.Series(broadcasted_arrs[5], index=sr2.index, name=sr2.name)
        )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(np.broadcast_arrays(
            *to_broadcast_a,
            *[x.to_frame() for x in to_broadcast_sr],  # here is the difference
            *to_broadcast_df
        ))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast_a, *to_broadcast_sr, *to_broadcast_df,
            index_from=None,
            columns_from=None,
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(7):
            pd.testing.assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=pd.RangeIndex(start=0, stop=3, step=1),
                    columns=pd.RangeIndex(start=0, stop=3, step=1)
                )
            )
        pd.testing.assert_frame_equal(
            broadcasted[7],
            pd.DataFrame(
                broadcasted_arrs[7],
                index=pd.Index(['x1', 'x1', 'x1'], dtype='object', name='i1'),
                columns=pd.Index(['a1', 'a1', 'a1'], dtype='object')
            )
        )
        pd.testing.assert_frame_equal(
            broadcasted[8],
            pd.DataFrame(
                broadcasted_arrs[8],
                index=sr2.index,
                columns=pd.Index(['a2', 'a2', 'a2'], dtype='object')
            )
        )
        pd.testing.assert_frame_equal(
            broadcasted[9],
            pd.DataFrame(
                broadcasted_arrs[9],
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=pd.RangeIndex(start=0, stop=3, step=1)
            )
        )
        pd.testing.assert_frame_equal(
            broadcasted[10],
            pd.DataFrame(
                broadcasted_arrs[10],
                index=pd.Index(['x3', 'x3', 'x3'], dtype='object', name='i3'),
                columns=pd.Index(['a3', 'a3', 'a3'], dtype='object', name='c3')
            )
        )
        pd.testing.assert_frame_equal(
            broadcasted[11],
            pd.DataFrame(
                broadcasted_arrs[11],
                index=df2.index,
                columns=pd.Index(['a4', 'a4', 'a4'], dtype='object', name='c4')
            )
        )
        pd.testing.assert_frame_equal(
            broadcasted[12],
            pd.DataFrame(
                broadcasted_arrs[12],
                index=pd.Index(['x5', 'x5', 'x5'], dtype='object', name='i5'),
                columns=df3.columns
            )
        )
        pd.testing.assert_frame_equal(
            broadcasted[13],
            pd.DataFrame(
                broadcasted_arrs[13],
                index=df4.index,
                columns=df4.columns
            )
        )

    def test_broadcast_specify(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast,
            index_from=multi_i,
            columns_from=['name'],  # should translate to Series name
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=multi_i,
                    name='name'
                )
            )
        broadcasted = reshape_fns.broadcast(
            *to_broadcast,
            index_from=multi_i,
            columns_from=[0],  # should translate to None
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=multi_i,
                    name=None
                )
            )
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(np.broadcast_arrays(
            *to_broadcast_a,
            *[x.to_frame() for x in to_broadcast_sr],  # here is the difference
            *to_broadcast_df
        ))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast_a, *to_broadcast_sr, *to_broadcast_df,
            index_from=multi_i,
            columns_from=multi_c,
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=multi_i,
                    columns=multi_c
                )
            )

    def test_broadcast_idx(self):
        # 1d
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast,
            index_from=-1,
            columns_from=-1,  # should translate to Series name
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=sr2.index,
                    name=sr2.name
                )
            )
        try:
            _ = reshape_fns.broadcast(
                *broadcasted,
                index_from=0,
                columns_from=0,
                drop_duplicates=True,
                ignore_single=True
            )
            raise ValueError
        except:
            pass
        # 2d
        to_broadcast_a = 0, a1, a2, a3, a4, a5
        to_broadcast_sr = sr_none, sr1, sr2
        to_broadcast_df = df_none, df1, df2, df3, df4
        broadcasted_arrs = list(np.broadcast_arrays(
            *to_broadcast_a,
            *[x.to_frame() for x in to_broadcast_sr],  # here is the difference
            *to_broadcast_df
        ))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast_a, *to_broadcast_sr, *to_broadcast_df,
            index_from=-1,
            columns_from=-1,
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=df4.index,
                    columns=df4.columns
                )
            )

    def test_broadcast_strict(self):
        # 1d
        to_broadcast = sr1, sr2
        try:
            _ = reshape_fns.broadcast(
                *to_broadcast,
                index_from='strict',  # changing index not allowed
                columns_from='stack',
                drop_duplicates=True,
                ignore_single=True
            )
            raise ValueError
        except:
            pass
        # 2d
        to_broadcast = df1, df2
        try:
            _ = reshape_fns.broadcast(
                *to_broadcast,
                index_from='stack',
                columns_from='strict',  # changing columns not allowed
                drop_duplicates=True,
                ignore_single=True
            )
            raise ValueError
        except:
            pass

    def test_broadcast_dirty(self):
        # 1d
        to_broadcast = sr2, 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast,
            index_from='stack',
            columns_from='stack',
            drop_duplicates=False,
            ignore_single=False
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_series_equal(
                broadcasted[i],
                pd.Series(
                    broadcasted_arrs[i],
                    index=pd.MultiIndex.from_tuples([
                        ('x2', 'x1', 'x2'),
                        ('y2', 'x1', 'y2'),
                        ('z2', 'x1', 'z2')
                    ], names=['i2', 'i1', 'i2']),
                    name=('a2', 'a1', 'a2')
                )
            )

    def test_broadcast_to_shape(self):
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = [np.broadcast_to(x.to_frame() if isinstance(x, pd.Series) else x, (3, 3)) for x in
                            to_broadcast]
        broadcasted = reshape_fns.broadcast(
            *to_broadcast,
            to_shape=(3, 3),
            index_from='stack',
            columns_from='stack',
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            pd.testing.assert_frame_equal(
                broadcasted[i],
                pd.DataFrame(
                    broadcasted_arrs[i],
                    index=sr2.index,
                    columns=pd.MultiIndex.from_tuples([
                        ('a1', 'a2'),
                        ('a1', 'a2'),
                        ('a1', 'a2')
                    ])
                )
            )

    def test_broadcast_to_pd(self):
        to_broadcast = 0, a1, a2, sr_none, sr1, sr2
        broadcasted_arrs = list(np.broadcast_arrays(*to_broadcast))
        broadcasted = reshape_fns.broadcast(
            *to_broadcast,
            to_pd=False,  # to NumPy
            index_from='stack',
            columns_from='stack',
            drop_duplicates=True,
            ignore_single=True
        )
        for i in range(len(broadcasted)):
            np.testing.assert_array_equal(
                broadcasted[i],
                broadcasted_arrs[i]
            )

    def test_broadcast_copy(self):
        a, _ = reshape_fns.broadcast(np.empty((1,)), a5, writeable=False)  # readonly
        assert not a.flags.writeable
        a, _ = reshape_fns.broadcast(np.empty((1,)), a5, writeable=True)  # writeable
        assert a.flags.writeable
        a, _ = reshape_fns.broadcast(np.empty((1,)), a5, writeable=False,
                                     copy_kwargs={'order': 'C'})  # writeable, C order
        assert a.flags.writeable  # writeable since it was copied to make C order
        assert not np.isfortran(a)
        a, _ = reshape_fns.broadcast(a5, a5)  # no broadcasting, writeable
        assert a.flags.writeable
        a, _ = reshape_fns.broadcast(a5, a5, copy_kwargs={'order': 'C'})  # no copy
        assert not np.isfortran(a)
        a, _ = reshape_fns.broadcast(a5, a5, copy_kwargs={'order': 'F'})  # copy
        assert np.isfortran(a)

    def test_broadcast_to(self):
        np.testing.assert_array_equal(reshape_fns.broadcast_to(0, a5), np.broadcast_to(0, a5.shape))
        pd.testing.assert_series_equal(
            reshape_fns.broadcast_to(0, sr2),
            pd.Series(np.broadcast_to(0, sr2.shape), index=sr2.index, name=sr2.name)
        )
        pd.testing.assert_frame_equal(
            reshape_fns.broadcast_to(0, df5),
            pd.DataFrame(np.broadcast_to(0, df5.shape), index=df5.index, columns=df5.columns)
        )
        pd.testing.assert_frame_equal(
            reshape_fns.broadcast_to(sr2, df5),
            pd.DataFrame(np.broadcast_to(sr2.to_frame(), df5.shape), index=df5.index, columns=df5.columns)
        )
        pd.testing.assert_frame_equal(
            reshape_fns.broadcast_to(sr2, df5, index_from=0, columns_from=0),
            pd.DataFrame(
                np.broadcast_to(sr2.to_frame(), df5.shape),
                index=sr2.index,
                columns=pd.Index(['a2', 'a2', 'a2'], dtype='object'))
        )

    @pytest.mark.parametrize(
        "test_input",
        [0, a2, a5, sr2, df5, np.zeros((2, 2, 2))],
    )
    def test_broadcast_to_array_of(self, test_input):
        # broadcasting first element to be an array out of the second argument
        np.testing.assert_array_equal(
            reshape_fns.broadcast_to_array_of(0.1, test_input),
            np.full((1, *np.asarray(test_input).shape), 0.1)
        )
        np.testing.assert_array_equal(
            reshape_fns.broadcast_to_array_of([0.1], test_input),
            np.full((1, *np.asarray(test_input).shape), 0.1)
        )
        np.testing.assert_array_equal(
            reshape_fns.broadcast_to_array_of([0.1, 0.2], test_input),
            np.concatenate((
                np.full((1, *np.asarray(test_input).shape), 0.1),
                np.full((1, *np.asarray(test_input).shape), 0.2)
            ))
        )
        np.testing.assert_array_equal(
            reshape_fns.broadcast_to_array_of(np.expand_dims(np.asarray(test_input), 0), test_input),  # do nothing
            np.expand_dims(np.asarray(test_input), 0)
        )

    def test_broadcast_to_axis_of(self):
        np.testing.assert_array_equal(
            reshape_fns.broadcast_to_axis_of(10, np.empty((2,)), 0),
            np.full(2, 10)
        )
        assert reshape_fns.broadcast_to_axis_of(10, np.empty((2,)), 1) == 10
        np.testing.assert_array_equal(
            reshape_fns.broadcast_to_axis_of(10, np.empty((2, 3)), 0),
            np.full(2, 10)
        )
        np.testing.assert_array_equal(
            reshape_fns.broadcast_to_axis_of(10, np.empty((2, 3)), 1),
            np.full(3, 10)
        )
        assert reshape_fns.broadcast_to_axis_of(10, np.empty((2, 3)), 2) == 10

    def test_unstack_to_array(self):
        i = pd.MultiIndex.from_arrays([[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']])
        sr = pd.Series([1, 2, 3, 4], index=i)
        np.testing.assert_array_equal(
            reshape_fns.unstack_to_array(sr),
            np.asarray([[
                [1., np.nan, np.nan, np.nan],
                [np.nan, 2., np.nan, np.nan]
            ], [
                [np.nan, np.nan, 3., np.nan],
                [np.nan, np.nan, np.nan, 4.]
            ]])
        )
        np.testing.assert_array_equal(
            reshape_fns.unstack_to_array(sr, levels=(0,)),
            np.asarray([2., 4.])
        )
        np.testing.assert_array_equal(
            reshape_fns.unstack_to_array(sr, levels=(2, 0)),
            np.asarray([
                [1., np.nan],
                [2., np.nan],
                [np.nan, 3.],
                [np.nan, 4.],
            ])
        )

    def test_make_symmetric(self):
        pd.testing.assert_frame_equal(
            reshape_fns.make_symmetric(sr2),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, 1.0],
                    [np.nan, np.nan, np.nan, 2.0],
                    [np.nan, np.nan, np.nan, 3.0],
                    [1.0, 2.0, 3.0, np.nan]
                ]),
                index=pd.Index(['x2', 'y2', 'z2', 'a2'], dtype='object', name=('i2', None)),
                columns=pd.Index(['x2', 'y2', 'z2', 'a2'], dtype='object', name=('i2', None))
            )
        )
        pd.testing.assert_frame_equal(
            reshape_fns.make_symmetric(df2),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, 1.0],
                    [np.nan, np.nan, np.nan, 2.0],
                    [np.nan, np.nan, np.nan, 3.0],
                    [1.0, 2.0, 3.0, np.nan]
                ]),
                index=pd.Index(['x4', 'y4', 'z4', 'a4'], dtype='object', name=('i4', 'c4')),
                columns=pd.Index(['x4', 'y4', 'z4', 'a4'], dtype='object', name=('i4', 'c4'))
            )
        )
        pd.testing.assert_frame_equal(
            reshape_fns.make_symmetric(df5),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, 1.0, 2.0, 3.0],
                    [np.nan, np.nan, np.nan, 4.0, 5.0, 6.0],
                    [np.nan, np.nan, np.nan, 7.0, 8.0, 9.0],
                    [1.0, 4.0, 7.0, np.nan, np.nan, np.nan],
                    [2.0, 5.0, 8.0, np.nan, np.nan, np.nan],
                    [3.0, 6.0, 9.0, np.nan, np.nan, np.nan]
                ]),
                index=pd.MultiIndex.from_tuples([
                    ('x7', 'x8'),
                    ('y7', 'y8'),
                    ('z7', 'z8'),
                    ('a7', 'a8'),
                    ('b7', 'b8'),
                    ('c7', 'c8')
                ], names=[('i7', 'c7'), ('i8', 'c8')]),
                columns=pd.MultiIndex.from_tuples([
                    ('x7', 'x8'),
                    ('y7', 'y8'),
                    ('z7', 'z8'),
                    ('a7', 'a8'),
                    ('b7', 'b8'),
                    ('c7', 'c8')
                ], names=[('i7', 'c7'), ('i8', 'c8')])
            )
        )
        pd.testing.assert_frame_equal(
            reshape_fns.make_symmetric(pd.Series([1, 2, 3], name='yo')),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, np.nan, 1.0],
                    [np.nan, np.nan, np.nan, 2.0],
                    [np.nan, np.nan, np.nan, 3.0],
                    [1.0, 2.0, 3.0, np.nan]
                ]),
                index=pd.Index([0, 1, 2, 'yo'], dtype='object'),
                columns=pd.Index([0, 1, 2, 'yo'], dtype='object')
            )
        )

    def test_unstack_to_df(self):
        pd.testing.assert_frame_equal(
            reshape_fns.unstack_to_df(df5.iloc[0]),
            pd.DataFrame(
                np.array([
                    [1.0, np.nan, np.nan],
                    [np.nan, 2.0, np.nan],
                    [np.nan, np.nan, 3.0]
                ]),
                index=pd.Index(['a7', 'b7', 'c7'], dtype='object', name='c7'),
                columns=pd.Index(['a8', 'b8', 'c8'], dtype='object', name='c8')
            )
        )
        i = pd.MultiIndex.from_arrays([[1, 1, 2, 2], [3, 4, 3, 4], ['a', 'b', 'c', 'd']])
        sr = pd.Series([1, 2, 3, 4], index=i)
        pd.testing.assert_frame_equal(
            reshape_fns.unstack_to_df(sr, index_levels=0, column_levels=1),
            pd.DataFrame(
                np.array([
                    [1.0, 2.0],
                    [3.0, 4.0]
                ]),
                index=pd.Int64Index([1, 2], dtype='int64'),
                columns=pd.Int64Index([3, 4], dtype='int64')
            )
        )
        pd.testing.assert_frame_equal(
            reshape_fns.unstack_to_df(sr, index_levels=(0, 1), column_levels=2),
            pd.DataFrame(
                np.array([
                    [1.0, np.nan, np.nan, np.nan],
                    [np.nan, 2.0, np.nan, np.nan],
                    [np.nan, np.nan, 3.0, np.nan],
                    [np.nan, np.nan, np.nan, 4.0]
                ]),
                index=pd.MultiIndex.from_tuples([
                    (1, 3),
                    (1, 4),
                    (2, 3),
                    (2, 4)
                ]),
                columns=pd.Index(['a', 'b', 'c', 'd'], dtype='object')
            )
        )
        pd.testing.assert_frame_equal(
            reshape_fns.unstack_to_df(sr, index_levels=0, column_levels=1, symmetric=True),
            pd.DataFrame(
                np.array([
                    [np.nan, np.nan, 1.0, 2.0],
                    [np.nan, np.nan, 3.0, 4.0],
                    [1.0, 3.0, np.nan, np.nan],
                    [2.0, 4.0, np.nan, np.nan]
                ]),
                index=pd.Int64Index([1, 2, 3, 4], dtype='int64'),
                columns=pd.Int64Index([1, 2, 3, 4], dtype='int64')
            )
        )


# ############# indexing.py ############# #

def indexing_func(obj, pd_indexing_func):
    # As soon as you call iloc etc., performs it on each dataframe and mapper and returns a new class instance
    param1_mapper = indexing.indexing_on_mapper(obj._param1_mapper, obj.a, pd_indexing_func)
    param2_mapper = indexing.indexing_on_mapper(obj._param2_mapper, obj.a, pd_indexing_func)
    tuple_mapper = indexing.indexing_on_mapper(obj._tuple_mapper, obj.a, pd_indexing_func)
    return H(pd_indexing_func(obj.a), param1_mapper, param2_mapper, tuple_mapper)


PandasIndexer = indexing.PandasIndexer
ParamIndexer = indexing.ParamIndexerFactory(['param1', 'param2', 'tuple'])


class H(PandasIndexer, ParamIndexer):
    def __init__(self, a, param1_mapper, param2_mapper, tuple_mapper):
        self.a = a

        self._param1_mapper = param1_mapper
        self._param2_mapper = param2_mapper
        self._tuple_mapper = tuple_mapper

        PandasIndexer.__init__(self, indexing_func)
        ParamIndexer.__init__(self, [param1_mapper, param2_mapper, tuple_mapper], indexing_func)

    @classmethod
    def from_params(cls, a, params1, params2, level_names=('p1', 'p2')):
        a = reshape_fns.to_2d(a)
        # Build column hierarchy
        params1_idx = pd.Index(params1, name=level_names[0])
        params2_idx = pd.Index(params2, name=level_names[1])
        params_idx = index_fns.stack_indexes(params1_idx, params2_idx)
        new_columns = index_fns.combine_indexes(params_idx, a.columns)

        # Build mappers
        param1_mapper = np.repeat(params1, len(a.columns))
        param1_mapper = pd.Series(param1_mapper, index=new_columns, name=params1_idx.name)

        param2_mapper = np.repeat(params2, len(a.columns))
        param2_mapper = pd.Series(param2_mapper, index=new_columns, name=params2_idx.name)

        tuple_mapper = list(zip(*list(map(lambda x: x.values, [param1_mapper, param2_mapper]))))
        tuple_mapper = pd.Series(tuple_mapper, index=new_columns, name=(params1_idx.name, params2_idx.name))

        # Tile a to match the length of new_columns
        a = array_wrapper.ArrayWrapper(index=a.index, columns=new_columns).wrap(reshape_fns.tile(a.values, 4, axis=1))
        return cls(a, param1_mapper, param2_mapper, tuple_mapper)


# Similate an indicator with two params
h = H.from_params(df4, [0.1, 0.1, 0.2, 0.2], [0.3, 0.4, 0.5, 0.6])


class TestIndexing:

    def test_pandas_indexing(self):
        # __getitem__
        pd.testing.assert_series_equal(
            h[(0.1, 0.3, 'a6')].a,
            pd.Series(
                np.array([1, 4, 7]),
                index=pd.Index(['x6', 'y6', 'z6'], dtype='object', name='i6'),
                name=(0.1, 0.3, 'a6')
            )
        )
        # loc
        pd.testing.assert_frame_equal(
            h.loc[:, (0.1, 0.3, 'a6'):(0.1, 0.3, 'c6')].a,
            pd.DataFrame(
                np.array([
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ]),
                index=pd.Index(['x6', 'y6', 'z6'], dtype='object', name='i6'),
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 0.3, 'a6'),
                    (0.1, 0.3, 'b6'),
                    (0.1, 0.3, 'c6')
                ], names=['p1', 'p2', 'c6'])
            )
        )
        # iloc
        pd.testing.assert_frame_equal(
            h.iloc[-2:, -2:].a,
            pd.DataFrame(
                np.array([
                    [5, 6],
                    [8, 9]
                ]),
                index=pd.Index(['y6', 'z6'], dtype='object', name='i6'),
                columns=pd.MultiIndex.from_tuples([
                    (0.2, 0.6, 'b6'),
                    (0.2, 0.6, 'c6')
                ], names=['p1', 'p2', 'c6'])
            )
        )
        # xs
        pd.testing.assert_frame_equal(
            h.xs((0.1, 0.3), level=('p1', 'p2'), axis=1).a,
            pd.DataFrame(
                np.array([
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ]),
                index=pd.Index(['x6', 'y6', 'z6'], dtype='object', name='i6'),
                columns=pd.Index(['a6', 'b6', 'c6'], dtype='object', name='c6')
            )
        )

    def test_param_indexing(self):
        # param1
        pd.testing.assert_frame_equal(
            h.param1_loc[0.1].a,
            pd.DataFrame(
                np.array([
                    [1, 2, 3, 1, 2, 3],
                    [4, 5, 6, 4, 5, 6],
                    [7, 8, 9, 7, 8, 9]
                ]),
                index=pd.Index(['x6', 'y6', 'z6'], dtype='object', name='i6'),
                columns=pd.MultiIndex.from_tuples([
                    (0.3, 'a6'),
                    (0.3, 'b6'),
                    (0.3, 'c6'),
                    (0.4, 'a6'),
                    (0.4, 'b6'),
                    (0.4, 'c6')
                ], names=['p2', 'c6'])
            )
        )
        # param2
        pd.testing.assert_frame_equal(
            h.param2_loc[0.3].a,
            pd.DataFrame(
                np.array([
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ]),
                index=pd.Index(['x6', 'y6', 'z6'], dtype='object', name='i6'),
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 'a6'),
                    (0.1, 'b6'),
                    (0.1, 'c6')
                ], names=['p1', 'c6'])
            )
        )
        # tuple
        pd.testing.assert_frame_equal(
            h.tuple_loc[(0.1, 0.3)].a,
            pd.DataFrame(
                np.array([
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ]),
                index=pd.Index(['x6', 'y6', 'z6'], dtype='object', name='i6'),
                columns=pd.Index(['a6', 'b6', 'c6'], dtype='object', name='c6')
            )
        )
        pd.testing.assert_frame_equal(
            h.tuple_loc[(0.1, 0.3):(0.1, 0.3)].a,
            pd.DataFrame(
                np.array([
                    [1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]
                ]),
                index=pd.Index(['x6', 'y6', 'z6'], dtype='object', name='i6'),
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 0.3, 'a6'),
                    (0.1, 0.3, 'b6'),
                    (0.1, 0.3, 'c6')
                ], names=['p1', 'p2', 'c6'])
            )
        )
        pd.testing.assert_frame_equal(
            h.tuple_loc[[(0.1, 0.3), (0.1, 0.3)]].a,
            pd.DataFrame(
                np.array([
                    [1, 2, 3, 1, 2, 3],
                    [4, 5, 6, 4, 5, 6],
                    [7, 8, 9, 7, 8, 9]
                ]),
                index=pd.Index(['x6', 'y6', 'z6'], dtype='object', name='i6'),
                columns=pd.MultiIndex.from_tuples([
                    (0.1, 0.3, 'a6'),
                    (0.1, 0.3, 'b6'),
                    (0.1, 0.3, 'c6'),
                    (0.1, 0.3, 'a6'),
                    (0.1, 0.3, 'b6'),
                    (0.1, 0.3, 'c6')
                ], names=['p1', 'p2', 'c6'])
            )
        )


# ############# combine_fns.py ############# #

class TestCombineFns:
    def test_apply_and_concat_one(self):
        # 1d
        target = np.array([
            [11, 21, 31],
            [12, 22, 32],
            [13, 23, 33]
        ])
        np.testing.assert_array_equal(
            combine_fns.apply_and_concat_one(3, lambda i, x, a: x + a[i], sr2.values, [10, 20, 30]),
            target
        )
        np.testing.assert_array_equal(
            combine_fns.apply_and_concat_one_nb(3, njit(lambda i, x, a: x + a[i]), sr2.values, (10, 20, 30)),
            target
        )
        # 2d
        target2 = np.array([
            [11, 12, 13, 21, 22, 23, 31, 32, 33],
            [14, 15, 16, 24, 25, 26, 34, 35, 36],
            [17, 18, 19, 27, 28, 29, 37, 38, 39]
        ])
        np.testing.assert_array_equal(
            combine_fns.apply_and_concat_one(3, lambda i, x, a: x + a[i], df4.values, [10, 20, 30]),
            target2
        )
        np.testing.assert_array_equal(
            combine_fns.apply_and_concat_one_nb(3, njit(lambda i, x, a: x + a[i]), df4.values, (10, 20, 30)),
            target2
        )

    def test_apply_and_concat_multiple(self):
        # 1d
        target_a = np.array([
            [1, 1, 1],
            [2, 2, 2],
            [3, 3, 3]
        ])
        target_b = np.array([
            [11, 21, 31],
            [12, 22, 32],
            [13, 23, 33]
        ])
        a, b = combine_fns.apply_and_concat_multiple(3, lambda i, x, a: (x, x + a[i]), sr2.values, [10, 20, 30])
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combine_fns.apply_and_concat_multiple_nb(3, njit(lambda i, x, a: (x, x + a[i])), sr2.values,
                                                        (10, 20, 30))
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        # 2d
        target_a = np.array([
            [1, 2, 3, 1, 2, 3, 1, 2, 3],
            [4, 5, 6, 4, 5, 6, 4, 5, 6],
            [7, 8, 9, 7, 8, 9, 7, 8, 9]
        ])
        target_b = np.array([
            [11, 12, 13, 21, 22, 23, 31, 32, 33],
            [14, 15, 16, 24, 25, 26, 34, 35, 36],
            [17, 18, 19, 27, 28, 29, 37, 38, 39]
        ])
        a, b = combine_fns.apply_and_concat_multiple(3, lambda i, x, a: (x, x + a[i]), df4.values, [10, 20, 30])
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)
        a, b = combine_fns.apply_and_concat_multiple_nb(3, njit(lambda i, x, a: (x, x + a[i])), df4.values,
                                                        (10, 20, 30))
        np.testing.assert_array_equal(a, target_a)
        np.testing.assert_array_equal(b, target_b)

    def test_combine_and_concat(self):
        # 1d
        target = np.array([
            [103, 104],
            [106, 108],
            [109, 112]
        ])
        np.testing.assert_array_equal(
            combine_fns.combine_and_concat(
                sr2.values, (sr2.values * 2, sr2.values * 3), lambda x, y, a: x + y + a, 100),
            target
        )
        np.testing.assert_array_equal(
            combine_fns.combine_and_concat_nb(
                sr2.values, (sr2.values * 2, sr2.values * 3), njit(lambda x, y, a: x + y + a), 100),
            target
        )
        # 2d
        target2 = np.array([
            [103, 106, 109, 104, 108, 112],
            [112, 115, 118, 116, 120, 124],
            [121, 124, 127, 128, 132, 136]
        ])
        np.testing.assert_array_equal(
            combine_fns.combine_and_concat(
                df4.values, (df4.values * 2, df4.values * 3), lambda x, y, a: x + y + a, 100),
            target2
        )
        np.testing.assert_array_equal(
            combine_fns.combine_and_concat_nb(
                df4.values, (df4.values * 2, df4.values * 3), njit(lambda x, y, a: x + y + a), 100),
            target2
        )

    def test_combine_multiple(self):
        # 1d
        target = np.array([206, 212, 218])
        np.testing.assert_array_equal(
            combine_fns.combine_multiple(
                (sr2.values, sr2.values * 2, sr2.values * 3), lambda x, y, a: x + y + a, 100),
            target
        )
        np.testing.assert_array_equal(
            combine_fns.combine_multiple_nb(
                (sr2.values, sr2.values * 2, sr2.values * 3), njit(lambda x, y, a: x + y + a), 100),
            target
        )
        # 2d
        target2 = np.array([
            [206, 212, 218],
            [224, 230, 236],
            [242, 248, 254]
        ])
        np.testing.assert_array_equal(
            combine_fns.combine_multiple(
                (df4.values, df4.values * 2, df4.values * 3), lambda x, y, a: x + y + a, 100),
            target2
        )
        np.testing.assert_array_equal(
            combine_fns.combine_multiple_nb(
                (df4.values, df4.values * 2, df4.values * 3), njit(lambda x, y, a: x + y + a), 100),
            target2
        )


# ############# common.py ############# #

class TestCommon:
    def test_add_nb_methods_1d(self):
        @njit
        def same_shape_1d_nb(a): return a ** 2

        @njit
        def wkw_1d_nb(a, b=10): return a ** 3 + b

        @njit
        def reduced_dim0_1d_nb(a): return 0

        @njit
        def reduced_dim1_one_1d_nb(a): return np.zeros(1)

        @njit
        def reduced_dim1_1d_nb(a): return np.zeros(a.shape[0] - 1)

        @common.add_nb_methods([
            same_shape_1d_nb,
            wkw_1d_nb,
            reduced_dim0_1d_nb,
            reduced_dim1_one_1d_nb,
            reduced_dim1_1d_nb
        ])
        class H_1d(accessors.Base_Accessor):
            def __init__(self, sr):
                super().__init__(sr)

        pd.testing.assert_series_equal(H_1d(sr2).same_shape(), sr2 ** 2)
        pd.testing.assert_series_equal(H_1d(sr2).wkw(), sr2 ** 3 + 10)
        pd.testing.assert_series_equal(H_1d(sr2).wkw(b=20), sr2 ** 3 + 20)
        assert H_1d(sr2).reduced_dim0() == 0
        assert H_1d(sr2).reduced_dim1_one() == 0
        pd.testing.assert_series_equal(H_1d(sr2).reduced_dim1(), pd.Series(np.zeros(sr2.shape[0] - 1), name=sr2.name))

    def test_add_nb_methods_2d(self):
        @njit
        def same_shape_nb(a): return a ** 2

        @njit
        def wkw_nb(a, b=10): return a ** 3 + b

        @njit
        def reduced_dim0_nb(a): return 0

        @njit
        def reduced_dim1_nb(a): return np.zeros(a.shape[1])

        @njit
        def reduced_dim2_nb(a): return np.zeros((a.shape[0] - 1, a.shape[1]))

        @common.add_nb_methods([
            same_shape_nb,
            wkw_nb,
            reduced_dim0_nb,
            reduced_dim1_nb,
            reduced_dim2_nb
        ])
        class H(accessors.Base_Accessor):
            pass

        pd.testing.assert_frame_equal(H(df3).same_shape(), df3 ** 2)
        pd.testing.assert_frame_equal(H(df3).wkw(), df3 ** 3 + 10)
        pd.testing.assert_frame_equal(H(df3).wkw(b=20), df3 ** 3 + 20)
        assert H(df3).reduced_dim0() == 0
        pd.testing.assert_series_equal(H(df3).reduced_dim1(), pd.Series(np.zeros(df3.shape[1]), index=df3.columns))
        pd.testing.assert_frame_equal(
            H(df3).reduced_dim2(),
            pd.DataFrame(np.zeros((df3.shape[0] - 1, df3.shape[1])), columns=df3.columns)
        )


# ############# accessors.py ############# #

class TestAccessors:
    def test_props(self):
        assert sr1.vbt.is_series()
        assert not sr1.vbt.is_frame()
        assert not df1.vbt.is_series()
        assert df2.vbt.is_frame()

    def test_wrapper(self):
        pd.testing.assert_index_equal(sr2.vbt.index, sr2.index)
        pd.testing.assert_index_equal(sr2.vbt.columns, sr2.to_frame().columns)
        assert sr2.vbt.ndim == sr2.ndim
        assert sr2.vbt.name == sr2.name
        assert pd.Series([1, 2, 3]).vbt.name is None
        assert sr2.vbt.shape == sr2.shape
        pd.testing.assert_index_equal(df4.vbt.index, df4.index)
        pd.testing.assert_index_equal(df4.vbt.columns, df4.columns)
        assert df4.vbt.ndim == df4.ndim
        assert df4.vbt.name is None
        assert df4.vbt.shape == df4.shape
        pd.testing.assert_series_equal(sr2.vbt.wrap(a2), sr2)
        pd.testing.assert_series_equal(sr2.vbt.wrap(df2), sr2)
        pd.testing.assert_series_equal(
            sr2.vbt.wrap(df2.values, index=df2.index, columns=df2.columns),
            pd.Series(df2.values[:, 0], index=df2.index, name=df2.columns[0])
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.wrap(df4.values, columns=df4.columns),
            pd.DataFrame(df4.values, index=sr2.index, columns=df4.columns)
        )
        pd.testing.assert_frame_equal(df2.vbt.wrap(a2), df2)
        pd.testing.assert_frame_equal(df2.vbt.wrap(sr2), df2)
        pd.testing.assert_frame_equal(
            df2.vbt.wrap(df4.values, columns=df4.columns),
            pd.DataFrame(df4.values, index=df2.index, columns=df4.columns)
        )

    def test_empty(self):
        pd.testing.assert_series_equal(
            pd.Series.vbt.empty(5, index=np.arange(10, 15), name='a', fill_value=5),
            pd.Series(np.full(5, 5), index=np.arange(10, 15), name='a')
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.empty((5, 3), index=np.arange(10, 15), columns=['a', 'b', 'c'], fill_value=5),
            pd.DataFrame(np.full((5, 3), 5), index=np.arange(10, 15), columns=['a', 'b', 'c'])
        )
        pd.testing.assert_series_equal(
            pd.Series.vbt.empty_like(sr2, fill_value=5),
            pd.Series(np.full(sr2.shape, 5), index=sr2.index, name=sr2.name)
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.empty_like(df4, fill_value=5),
            pd.DataFrame(np.full(df4.shape, 5), index=df4.index, columns=df4.columns)
        )

    def test_apply_func_on_index(self):
        pd.testing.assert_frame_equal(
            df1.vbt.apply_on_index(lambda idx: idx + '_yo', axis=0),
            pd.DataFrame(
                np.asarray([1]),
                index=pd.Index(['x3_yo'], dtype='object', name='i3'),
                columns=pd.Index(['a3'], dtype='object', name='c3')
            )
        )
        pd.testing.assert_frame_equal(
            df1.vbt.apply_on_index(lambda idx: idx + '_yo', axis=1),
            pd.DataFrame(
                np.asarray([1]),
                index=pd.Index(['x3'], dtype='object', name='i3'),
                columns=pd.Index(['a3_yo'], dtype='object', name='c3')
            )
        )
        df1_copy = df1.copy()
        df1_copy.vbt.apply_on_index(lambda idx: idx + '_yo', axis=0, inplace=True)
        pd.testing.assert_frame_equal(
            df1_copy,
            pd.DataFrame(
                np.asarray([1]),
                index=pd.Index(['x3_yo'], dtype='object', name='i3'),
                columns=pd.Index(['a3'], dtype='object', name='c3')
            )
        )

    def test_to_array(self):
        np.testing.assert_array_equal(sr2.vbt.to_1d_array(), sr2.values)
        np.testing.assert_array_equal(sr2.vbt.to_2d_array(), sr2.to_frame().values)
        np.testing.assert_array_equal(df2.vbt.to_1d_array(), df2.iloc[:, 0].values)
        np.testing.assert_array_equal(df2.vbt.to_2d_array(), df2.values)

    def test_tile(self):
        pd.testing.assert_frame_equal(
            df4.vbt.tile(2, as_columns=['a', 'b']),
            pd.DataFrame(
                np.asarray([
                    [1, 2, 3, 1, 2, 3],
                    [4, 5, 6, 4, 5, 6],
                    [7, 8, 9, 7, 8, 9]
                ]),
                index=df4.index,
                columns=pd.MultiIndex.from_tuples([
                    ('a', 'a6'),
                    ('a', 'b6'),
                    ('a', 'c6'),
                    ('b', 'a6'),
                    ('b', 'b6'),
                    ('b', 'c6')
                ], names=[None, 'c6'])
            )
        )

    def test_repeat(self):
        pd.testing.assert_frame_equal(
            df4.vbt.repeat(2, as_columns=['a', 'b']),
            pd.DataFrame(
                np.asarray([
                    [1, 1, 2, 2, 3, 3],
                    [4, 4, 5, 5, 6, 6],
                    [7, 7, 8, 8, 9, 9]
                ]),
                index=df4.index,
                columns=pd.MultiIndex.from_tuples([
                    ('a6', 'a'),
                    ('a6', 'b'),
                    ('b6', 'a'),
                    ('b6', 'b'),
                    ('c6', 'a'),
                    ('c6', 'b')
                ], names=['c6', None])
            )
        )

    def test_align_to(self):
        multi_c1 = pd.MultiIndex.from_arrays([['a8', 'b8']], names=['c8'])
        multi_c2 = pd.MultiIndex.from_arrays([['a7', 'a7', 'c7', 'c7'], ['a8', 'b8', 'a8', 'b8']], names=['c7', 'c8'])
        df10 = pd.DataFrame([[1, 2], [4, 5], [7, 8]], columns=multi_c1)
        df20 = pd.DataFrame([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]], columns=multi_c2)
        pd.testing.assert_frame_equal(
            df10.vbt.align_to(df20),
            pd.DataFrame(
                np.asarray([
                    [1, 2, 1, 2],
                    [4, 5, 4, 5],
                    [7, 8, 7, 8]
                ]),
                index=pd.RangeIndex(start=0, stop=3, step=1),
                columns=multi_c2
            )
        )

    def test_broadcast(self):
        a, b = pd.Series.vbt.broadcast(sr2, 10)
        b_target = pd.Series(np.full(sr2.shape, 10), index=sr2.index, name=sr2.name)
        pd.testing.assert_series_equal(a, sr2)
        pd.testing.assert_series_equal(b, b_target)
        a, b = sr2.vbt.broadcast(10)
        pd.testing.assert_series_equal(a, sr2)
        pd.testing.assert_series_equal(b, b_target)
        pd.testing.assert_frame_equal(sr2.vbt.broadcast_to(df2), df2)

    def test_apply(self):
        pd.testing.assert_series_equal(sr2.vbt.apply(apply_func=lambda x: x ** 2), sr2 ** 2)
        pd.testing.assert_series_equal(sr2.vbt.apply(apply_func=lambda x: x ** 2, pass_2d=True), sr2 ** 2)
        pd.testing.assert_frame_equal(df4.vbt.apply(apply_func=lambda x: x ** 2), df4 ** 2)

    def test_concat(self):
        target = pd.DataFrame(
            np.array([
                [1, 1, 1, 10, 10, 10, 1, 2, 3],
                [2, 2, 2, 10, 10, 10, 4, 5, 6],
                [3, 3, 3, 10, 10, 10, 7, 8, 9]
            ]),
            index=pd.MultiIndex.from_tuples([
                ('x2', 'x6'),
                ('y2', 'y6'),
                ('z2', 'z6')
            ], names=['i2', 'i6']),
            columns=pd.MultiIndex.from_tuples([
                ('a', 'a6'),
                ('a', 'b6'),
                ('a', 'c6'),
                ('b', 'a6'),
                ('b', 'b6'),
                ('b', 'c6'),
                ('c', 'a6'),
                ('c', 'b6'),
                ('c', 'c6')
            ], names=[None, 'c6'])
        )
        pd.testing.assert_frame_equal(
            pd.DataFrame.vbt.concat(sr2, 10, df4, as_columns=['a', 'b', 'c']),
            target
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.concat(10, df4, as_columns=['a', 'b', 'c']),
            target
        )

    def test_apply_and_concat(self):
        target = pd.DataFrame(
            np.array([
                [112, 113, 114],
                [113, 114, 115],
                [114, 115, 116]
            ]),
            index=pd.Index(['x2', 'y2', 'z2'], dtype='object', name='i2'),
            columns=pd.Index(['a', 'b', 'c'], dtype='object')
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3, np.array([1, 2, 3]), 10, apply_func=lambda i, x, y, c, d=1: x + y[i] + c + d, d=100,
                as_columns=['a', 'b', 'c']
            ),
            target
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3, np.array([1, 2, 3]), 10, apply_func=njit(lambda i, x, y, c, d=1: x + y[i] + c + 100),
                as_columns=['a', 'b', 'c']
            ),
            target
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3, np.array([1, 2, 3]), 10, apply_func=lambda i, x, y, c, d=1: x + y[i] + c + d, d=100
            ),
            pd.DataFrame(
                target.values,
                index=target.index,
                columns=pd.Int64Index([0, 1, 2], dtype='int64', name='apply_idx')
            )
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.apply_and_concat(
                3, np.array([[1], [2], [3]]), 10, apply_func=lambda i, x, y, c, d=1: x + y + c + d, d=100,
                as_columns=['a', 'b', 'c'],
                pass_2d=True  # otherwise (3, 1) + (1, 3) = (3, 3) != (3, 1) -> error
            ),
            pd.DataFrame(
                np.array([
                    [112, 112, 112],
                    [114, 114, 114],
                    [116, 116, 116]
                ]),
                index=target.index,
                columns=target.columns
            )
        )
        target2 = pd.DataFrame(
            np.array([
                [112, 113, 114],
                [113, 114, 115],
                [114, 115, 116]
            ]),
            index=pd.Index(['x4', 'y4', 'z4'], dtype='object', name='i4'),
            columns=pd.Index(['a', 'b', 'c'], dtype='object')
        )
        pd.testing.assert_frame_equal(
            df2.vbt.apply_and_concat(
                3, np.array([1, 2, 3]), 10, apply_func=lambda i, x, y, c, d=1: x + y[i] + c + d, d=100,
                as_columns=['a', 'b', 'c']
            ),
            target2
        )
        pd.testing.assert_frame_equal(
            df2.vbt.apply_and_concat(
                3, np.array([1, 2, 3]), 10, apply_func=njit(lambda i, x, y, c, d=1: x + y[i] + c + 100),
                as_columns=['a', 'b', 'c']
            ),
            target2
        )

    def test_combine_with(self):
        pd.testing.assert_series_equal(
            sr2.vbt.combine_with(10, 100, d=1000, combine_func=lambda x, y, c, d=1: x + y + c + d),
            pd.Series(
                np.array([1111, 1112, 1113]),
                index=pd.Index(['x2', 'y2', 'z2'], dtype='object', name='i2'),
                name='a2'
            )
        )
        pd.testing.assert_series_equal(
            sr2.vbt.combine_with(10, 100, 1000, combine_func=njit(lambda x, y, c, d: x + y + c + d)),
            pd.Series(
                np.array([1111, 1112, 1113]),
                index=pd.Index(['x2', 'y2', 'z2'], dtype='object', name='i2'),
                name='a2'
            )
        )
        pd.testing.assert_series_equal(
            sr2.vbt.combine_with(10, combine_func=njit(lambda x, y: x + y + np.array([[1], [2], [3]])), pass_2d=True),
            pd.Series(
                np.array([12, 14, 16]),
                index=pd.Index(['x2', 'y2', 'z2'], dtype='object', name='i2'),
                name='a2'
            )
        )
        pd.testing.assert_frame_equal(
            df4.vbt.combine_with(sr2, combine_func=lambda x, y: x + y),
            pd.DataFrame(
                np.array([
                    [2, 3, 4],
                    [6, 7, 8],
                    [10, 11, 12]
                ]),
                index=pd.MultiIndex.from_tuples([
                    ('x6', 'x2'),
                    ('y6', 'y2'),
                    ('z6', 'z2')
                ], names=['i6', 'i2']),
                columns=pd.Index(['a6', 'b6', 'c6'], dtype='object', name='c6')
            )
        )

    def test_combine_with_multiple(self):
        target = pd.DataFrame(
            np.array([
                [232, 233, 234],
                [236, 237, 238],
                [240, 241, 242]
            ]),
            index=pd.MultiIndex.from_tuples([
                ('x2', 'x6'),
                ('y2', 'y6'),
                ('z2', 'z6')
            ], names=['i2', 'i6']),
            columns=pd.Index(['a6', 'b6', 'c6'], dtype='object', name='c6')
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.combine_with_multiple(
                [10, df4], 10, b=100,
                combine_func=lambda x, y, a, b=1: x + y + a + b
            ),
            target
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.combine_with_multiple(
                [10, df4], 10, 100,
                combine_func=njit(lambda x, y, a, b: x + y + a + b)
            ),
            target
        )
        pd.testing.assert_frame_equal(
            df4.vbt.combine_with_multiple(
                [10, sr2], 10, b=100,
                combine_func=lambda x, y, a, b=1: x + y + a + b
            ),
            pd.DataFrame(
                target.values,
                index=pd.MultiIndex.from_tuples([
                    ('x6', 'x2'),
                    ('y6', 'y2'),
                    ('z6', 'z2')
                ], names=['i6', 'i2']),
                columns=target.columns
            )
        )
        target2 = pd.DataFrame(
            np.array([
                [121, 121, 121, 112, 113, 114],
                [122, 122, 122, 116, 117, 118],
                [123, 123, 123, 120, 121, 122]
            ]),
            index=pd.MultiIndex.from_tuples([
                ('x2', 'x6'),
                ('y2', 'y6'),
                ('z2', 'z6')
            ], names=['i2', 'i6']),
            columns=pd.MultiIndex.from_tuples([
                (0, 'a6'),
                (0, 'b6'),
                (0, 'c6'),
                (1, 'a6'),
                (1, 'b6'),
                (1, 'c6')
            ], names=['combine_idx', 'c6'])
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.combine_with_multiple(
                [10, df4], 10, b=100,
                combine_func=lambda x, y, a, b=1: x + y + a + b,
                concat=True
            ),
            target2
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.combine_with_multiple(
                [10, df4], 10, 100,
                combine_func=njit(lambda x, y, a, b: x + y + a + b),
                concat=True
            ),
            target2
        )
        pd.testing.assert_frame_equal(
            sr2.vbt.combine_with_multiple(
                [10, df4], 10, b=100,
                combine_func=lambda x, y, a, b=1: x + y + a + b,
                concat=True,
                as_columns=['a', 'b']
            ),
            pd.DataFrame(
                target2.values,
                index=target2.index,
                columns=pd.MultiIndex.from_tuples([
                    ('a', 'a6'),
                    ('a', 'b6'),
                    ('a', 'c6'),
                    ('b', 'a6'),
                    ('b', 'b6'),
                    ('b', 'c6')
                ], names=[None, 'c6'])
            )
        )

    @pytest.mark.parametrize(
        "test_input",
        [sr2, df5],
    )
    def test_magic(self, test_input):
        a = test_input
        b = test_input.copy()
        np.random.shuffle(b.values)
        assert_func = pd.testing.assert_series_equal if isinstance(a, pd.Series) else pd.testing.assert_frame_equal

        # binary ops
        # comparison ops
        assert_func(a.vbt == b, a == b)
        assert_func(a.vbt != b, a != b)
        assert_func(a.vbt < b, a < b)
        assert_func(a.vbt > b, a > b)
        assert_func(a.vbt <= b, a <= b)
        assert_func(a.vbt >= b, a >= b)
        # arithmetic ops
        assert_func(a.vbt + b, a + b)
        assert_func(a.vbt - b, a - b)
        assert_func(a.vbt * b, a * b)
        assert_func(a.vbt ** b, a ** b)
        assert_func(a.vbt % b, a % b)
        assert_func(a.vbt // b, a // b)
        assert_func(a.vbt / b, a / b)
        # __r*__ is only called if the left object does not have an __*__ method
        assert_func(10 + a.vbt, 10 + a)
        assert_func(10 - a.vbt, 10 - a)
        assert_func(10 * a.vbt, 10 * a)
        assert_func(10 ** a.vbt, 10 ** a)
        assert_func(10 % a.vbt, 10 % a)
        assert_func(10 // a.vbt, 10 // a)
        assert_func(10 / a.vbt, 10 / a)
        # mask ops
        assert_func(a.vbt & b, a & b)
        assert_func(a.vbt | b, a | b)
        assert_func(a.vbt ^ b, a ^ b)
        assert_func(10 & a.vbt, 10 & a)
        assert_func(10 | a.vbt, 10 | a)
        assert_func(10 ^ a.vbt, 10 ^ a)
        # unary ops
        assert_func(-a.vbt, -a.vbt)
        assert_func(+a.vbt, +a.vbt)
        assert_func(abs((-a).vbt), abs((-a).vbt))
