"""Classes for working with log records."""

import pandas as pd

from vectorbt.enums import log_dt, SizeType, Direction, OrderSide, OrderStatus, StatusInfo
from vectorbt.records.base import Records


class Logs(Records):
    """Extends `Records` for working with log records."""
    
    def __init__(self, wrapper, records_arr, idx_field='idx'):
        Records.__init__(
            self,
            wrapper,
            records_arr,
            idx_field=idx_field
        )

        if not all(field in records_arr.dtype.names for field in log_dt.names):
            raise ValueError("Records array must have all fields defined in debug_info_dt")

    @property  # no need for cached
    def records_readable(self):
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame(columns=pd.MultiIndex.from_tuples([
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
            ('Order', 'Minimum Size'),
            ('Order', 'Maximum Size'),
            ('Order', 'Rejection Prob.'),
            ('Order', 'Close First?'),
            ('Order', 'Allow Partial?'),
            ('Order', 'Raise Rejection?'),
            ('Result', 'New Cash'),
            ('Result', 'New Shares'),
            ('Result', 'Size'),
            ('Result', 'Price'),
            ('Result', 'Fees'),
            ('Result', 'Side'),
            ('Result', 'Status'),
            ('Result', 'Status Info')
        ]))

        def map_enum(sr, enum):
            return sr.map(lambda x: enum._fields[x] if x != -1 else None)

        out[('Context', 'Date')] = records_df['idx'].map(lambda x: self.wrapper.index[x])
        out[('Context', 'Column')] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        groups = self.wrapper.grouper.get_columns()
        out[('Context', 'Group')] = records_df['group'].map(lambda x: groups[x])
        out[('Context', 'Cash')] = records_df['cash_now']
        out[('Context', 'Shares')] = records_df['shares_now']
        out[('Context', 'Val. Price')] = records_df['val_price_now']
        out[('Context', 'Value')] = records_df['value_now']
        out[('Order', 'Size')] = records_df['size']
        out[('Order', 'Size Type')] = map_enum(records_df['size_type'], SizeType)
        out[('Order', 'Direction')] = map_enum(records_df['direction'], Direction)
        out[('Order', 'Price')] = records_df['price']
        out[('Order', 'Fees')] = records_df['fees']
        out[('Order', 'Fixed Fees')] = records_df['fixed_fees']
        out[('Order', 'Slippage')] = records_df['slippage']
        out[('Order', 'Minimum Size')] = records_df['min_size']
        out[('Order', 'Maximum Size')] = records_df['max_size']
        out[('Order', 'Rejection Prob.')] = records_df['reject_prob']
        out[('Order', 'Close First?')] = records_df['close_first']
        out[('Order', 'Allow Partial?')] = records_df['allow_partial']
        out[('Order', 'Raise Rejection?')] = records_df['raise_reject']
        out[('Result', 'New Cash')] = records_df['new_cash']
        out[('Result', 'New Shares')] = records_df['new_shares']
        out[('Result', 'Size')] = records_df['res_size']
        out[('Result', 'Price')] = records_df['res_price']
        out[('Result', 'Fees')] = records_df['res_fees']
        out[('Result', 'Side')] = map_enum(records_df['res_side'], OrderSide)
        out[('Result', 'Status')] = map_enum(records_df['res_status'], OrderStatus)
        out[('Result', 'Status Info')] = map_enum(records_df['res_status_info'], StatusInfo)
        return out
