"""Base class for working with log records.

Class `Logs` wraps log records to analyze logs. Logs are mainly populated when
simulating a portfolio and can be accessed as `vectorbt.portfolio.base.Portfolio.logs`."""

import pandas as pd

from vectorbt import _typing as tp
from vectorbt.base.array_wrapper import ArrayWrapper
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

    def __init__(self,
                 wrapper: ArrayWrapper,
                 records_arr: tp.RecordArray,
                 idx_field: str = 'idx',
                 **kwargs) -> None:
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
    def records_readable(self) -> tp.Frame:
        """Records in readable format."""
        records_df = self.records
        out = pd.DataFrame(columns=pd.MultiIndex.from_tuples([
            ('Context', 'Log Id'),
            ('Context', 'Date'),
            ('Context', 'Column'),
            ('Context', 'Group'),
            ('Context', 'Cash'),
            ('Context', 'Position'),
            ('Context', 'Debt'),
            ('Context', 'Free Cash'),
            ('Context', 'Val Price'),
            ('Context', 'Value'),
            ('Order', 'Size'),
            ('Order', 'Price'),
            ('Order', 'Size Type'),
            ('Order', 'Direction'),
            ('Order', 'Fees'),
            ('Order', 'Fixed Fees'),
            ('Order', 'Slippage'),
            ('Order', 'Min Size'),
            ('Order', 'Max Size'),
            ('Order', 'Rejection Prob'),
            ('Order', 'Lock Cash'),
            ('Order', 'Allow Partial'),
            ('Order', 'Raise Rejection'),
            ('Order', 'Log'),
            ('New Context', 'Cash'),
            ('New Context', 'Position'),
            ('New Context', 'Debt'),
            ('New Context', 'Free Cash'),
            ('New Context', 'Val Price'),
            ('New Context', 'Value'),
            ('Order Result', 'Size'),
            ('Order Result', 'Price'),
            ('Order Result', 'Fees'),
            ('Order Result', 'Side'),
            ('Order Result', 'Status'),
            ('Order Result', 'Status Info'),
            ('Order Result', 'Order Id')
        ]))

        def map_enum(sr, enum):
            return sr.map(lambda x: enum._fields[x] if x != -1 else None)

        out.iloc[:, 0] = records_df['id']
        out.iloc[:, 1] = records_df['idx'].map(lambda x: self.wrapper.index[x])
        out.iloc[:, 2] = records_df['col'].map(lambda x: self.wrapper.columns[x])
        out.iloc[:, 3] = records_df['group']
        out.iloc[:, 4] = records_df['cash']
        out.iloc[:, 5] = records_df['position']
        out.iloc[:, 6] = records_df['debt']
        out.iloc[:, 7] = records_df['free_cash']
        out.iloc[:, 8] = records_df['val_price']
        out.iloc[:, 9] = records_df['value']
        out.iloc[:, 10] = records_df['size']
        out.iloc[:, 11] = records_df['price']
        out.iloc[:, 12] = map_enum(records_df['size_type'], SizeType)
        out.iloc[:, 13] = map_enum(records_df['direction'], Direction)
        out.iloc[:, 14] = records_df['fees']
        out.iloc[:, 15] = records_df['fixed_fees']
        out.iloc[:, 16] = records_df['slippage']
        out.iloc[:, 17] = records_df['min_size']
        out.iloc[:, 18] = records_df['max_size']
        out.iloc[:, 19] = records_df['reject_prob']
        out.iloc[:, 20] = records_df['lock_cash']
        out.iloc[:, 21] = records_df['allow_partial']
        out.iloc[:, 22] = records_df['raise_reject']
        out.iloc[:, 23] = records_df['log']
        out.iloc[:, 24] = records_df['new_cash']
        out.iloc[:, 25] = records_df['new_position']
        out.iloc[:, 26] = records_df['new_debt']
        out.iloc[:, 27] = records_df['new_free_cash']
        out.iloc[:, 28] = records_df['new_val_price']
        out.iloc[:, 29] = records_df['new_value']
        out.iloc[:, 30] = records_df['res_size']
        out.iloc[:, 31] = records_df['res_price']
        out.iloc[:, 32] = records_df['res_fees']
        out.iloc[:, 33] = map_enum(records_df['res_side'], OrderSide)
        out.iloc[:, 34] = map_enum(records_df['res_status'], OrderStatus)
        out.iloc[:, 35] = map_enum(records_df['res_status_info'], StatusInfo)
        out.iloc[:, 36] = records_df['order_id']
        return out
