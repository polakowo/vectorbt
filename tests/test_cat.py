import pandas as pd
import numpy as np
from datetime import datetime

import vectorbt as vbt

df = pd.DataFrame({
    'a': [1, 2, 3, 4, 5],
    'b': [5, 4, 3, 2, 1],
    'c': [1, 2, 3, 2, 1]
}, index=pd.DatetimeIndex([
    datetime(2018, 1, 1),
    datetime(2018, 1, 2),
    datetime(2018, 1, 3),
    datetime(2018, 1, 4),
    datetime(2018, 1, 5)
]))
group_by = np.array(['g1', 'g1', 'g2'])


# ############# Global ############# #

def setup_module():
    vbt.settings.numba['check_func_suffix'] = True
    vbt.settings.caching.enabled = False
    vbt.settings.caching.whitelist = []
    vbt.settings.caching.blacklist = []


def teardown_module():
    vbt.settings.reset()


# ############# accessors.py ############# #


class TestAccessors:
    def test_stats(self):
        stat_index = pd.Index([
            'Start', 'End', 'Period',
            'Value Counts: 1',
            'Value Counts: 2',
            'Value Counts: 3',
            'Value Counts: 4',
            'Value Counts: 5'
        ], dtype='object')
        pd.testing.assert_series_equal(
            df.vbt.cat.stats(column='a'),
            pd.Series([
                pd.Timestamp('2018-01-01 00:00:00'),
                pd.Timestamp('2018-01-05 00:00:00'),
                pd.Timedelta('5 days 00:00:00'),
                1, 1, 1, 1, 1
            ],
                index=stat_index,
                name='a'
            )
        )
        pd.testing.assert_series_equal(
            df.vbt.cat.stats(column='a', settings=dict(mapping={
                1: 'test1',
                2: 'test2',
                3: 'test3',
                4: 'test4',
                5: 'test5'
            })),
            pd.Series([
                pd.Timestamp('2018-01-01 00:00:00'),
                pd.Timestamp('2018-01-05 00:00:00'),
                pd.Timedelta('5 days 00:00:00'),
                1, 1, 1, 1, 1
            ],
                index=pd.Index([
                    'Start', 'End', 'Period',
                    'Value Counts: test1',
                    'Value Counts: test2',
                    'Value Counts: test3',
                    'Value Counts: test4',
                    'Value Counts: test5'
                ], dtype='object'),
                name='a'
            )
        )
        pd.testing.assert_series_equal(
            df.vbt.cat['c'].stats(),
            pd.Series([
                pd.Timestamp('2018-01-01 00:00:00'),
                pd.Timestamp('2018-01-05 00:00:00'),
                pd.Timedelta('5 days 00:00:00'),
                2, 2, 1
            ],
                index=stat_index[:-2],
                name='c'
            )
        )
        pd.testing.assert_series_equal(
            df.vbt.cat.stats(column='c'),
            df.vbt.cat(group_by=group_by).stats(column='c', group_by=False)
        )
        pd.testing.assert_series_equal(
            df.vbt.cat(group_by=group_by)['g2'].stats(),
            pd.Series([
                pd.Timestamp('2018-01-01 00:00:00'),
                pd.Timestamp('2018-01-05 00:00:00'),
                pd.Timedelta('5 days 00:00:00'),
                2, 2, 1
            ],
                index=stat_index[:-2],
                name='g2'
            )
        )
        pd.testing.assert_series_equal(
            df.vbt.cat(group_by=group_by).stats(column='g2'),
            df.vbt.cat.stats(column='g2', group_by=group_by)
        )
        stats_df = df.vbt.cat.stats(agg_func=None)
        assert stats_df.shape == (3, 8)
        pd.testing.assert_index_equal(stats_df.index, df.vbt.cat.wrapper.columns)
        pd.testing.assert_index_equal(stats_df.columns, stat_index)
