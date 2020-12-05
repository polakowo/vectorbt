"""Base class for working with log records."""

import pandas as pd

from vectorbt.records.base import Records
from vectorbt.portfolio.enums import (
    log_dt,
    SizeType,
    Direction,
    OrderSide,
    OrderStatus,
    StatusInfo
)


class Logs(Records):
    """Extends `Records` for working with log records.

    !!! note
        Some features require the log records to be sorted prior to the processing.
        Use the `vectorbt.records.base.Records.sort` method."""

    def __init__(self, wrapper, records_arr, idx_field='idx', **kwargs):
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field,
            **kwargs
        )

        if not all(field in records_arr.dtype.names for field in log_dt.names):
            raise TypeError("Records array must match debug_info_dt")

    @property  # no need for cached
    def records_readable(self):
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame(columns=pd.MultiIndex.from_tuples([
            ('Context', 'Log Id'),
            ('Context', 'Date'),
            ('Context', 'Column'),
            ('Context', 'Group'),
            ('Context', 'Cash'),
            ('Context', 'Shares'),
            ('Context', 'Val. Price'),
            ('Context', 'Value'),
            ('Order', 'Size'),
            ('Order', 'Size Type'),
            ('Order', 'Direction'),
            ('Order', 'Price'),
            ('Order', 'Fees'),
            ('Order', 'Fixed Fees'),
            ('Order', 'Slippage'),
            ('Order', 'Min. Size'),
            ('Order', 'Max. Size'),
            ('Order', 'Rejection Prob.'),
            ('Order', 'Close First?'),
            ('Order', 'Allow Partial?'),
            ('Order', 'Raise Rejection?'),
            ('Order', 'Log?'),
            ('Result', 'New Cash'),
            ('Result', 'New Shares'),
            ('Result', 'Size'),
            ('Result', 'Price'),
            ('Result', 'Fees'),
            ('Result', 'Side'),
            ('Result', 'Status'),
            ('Result', 'Status Info'),
            ('Result', 'Order Id')
        ]))

        def map_enum(sr, enum):
            return sr.map(lambda x: enum._fields[x] if x != -1 else None)

        out.iloc[:, 0] = records_df['id']
        out.iloc[:, 1] = records_df['idx'].map(lambda x: self.wrapper.index[x])
        out.iloc[:, 2] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        out.iloc[:, 3] = records_df['group']
        out.iloc[:, 4] = records_df['cash_now']
        out.iloc[:, 5] = records_df['shares_now']
        out.iloc[:, 6] = records_df['val_price_now']
        out.iloc[:, 7] = records_df['value_now']
        out.iloc[:, 8] = records_df['size']
        out.iloc[:, 9] = map_enum(records_df['size_type'], SizeType)
        out.iloc[:, 10] = map_enum(records_df['direction'], Direction)
        out.iloc[:, 11] = records_df['price']
        out.iloc[:, 12] = records_df['fees']
        out.iloc[:, 13] = records_df['fixed_fees']
        out.iloc[:, 14] = records_df['slippage']
        out.iloc[:, 15] = records_df['min_size']
        out.iloc[:, 16] = records_df['max_size']
        out.iloc[:, 17] = records_df['reject_prob']
        out.iloc[:, 18] = records_df['close_first']
        out.iloc[:, 19] = records_df['allow_partial']
        out.iloc[:, 20] = records_df['raise_reject']
        out.iloc[:, 21] = records_df['log']
        out.iloc[:, 22] = records_df['new_cash']
        out.iloc[:, 23] = records_df['new_shares']
        out.iloc[:, 24] = records_df['res_size']
        out.iloc[:, 25] = records_df['res_price']
        out.iloc[:, 26] = records_df['res_fees']
        out.iloc[:, 27] = map_enum(records_df['res_side'], OrderSide)
        out.iloc[:, 28] = map_enum(records_df['res_status'], OrderStatus)
        out.iloc[:, 29] = map_enum(records_df['res_status_info'], StatusInfo)
        out.iloc[:, 30] = records_df['order_id']
        return out
