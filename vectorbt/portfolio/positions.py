"""Class for measuring performance of positions."""

import numpy as np
import pandas as pd
from inspect import isfunction

from vectorbt import timeseries
from vectorbt.utils import checks, reshape_fns
from vectorbt.portfolio import nb
from vectorbt.portfolio.common import timeseries_property, metric_property, group_property
from vectorbt.portfolio.enums import PositionStatus, PositionRecord
from vectorbt.portfolio.events import Events


class Positions(Events):
    """Extends `vectorbt.portfolio.events.Events` with position-related properties.

    Requires records of type `vectorbt.portfolio.enums.PositionRecord`.

    Example:
        Get the average PnL of closed positions with duration over 2 days:
        ```
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from numba import njit
        >>> from datetime import datetime
        >>> from vectorbt.portfolio.positions import Positions
        >>> from vectorbt.portfolio.enums import PositionStatus

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, -1, 1, 0, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders, 
        ...      init_capital=100, data_freq='1D')
        >>> print(portfolio.positions.avg_pnl)
        -0.5

        >>> @njit
        ... def filter_func_nb(col, i, value, status, duration):
        ...     return status[i, col] == PositionStatus.Closed and duration[i, col] >= 2
        >>> filter = (
        ...     filter_func_nb, 
        ...     portfolio.positions.status.vbt.to_2d_array(), 
        ...     portfolio.positions.duration.vbt.to_2d_array()
        ... )

        >>> positions = Positions(
        ...     portfolio.price, 
        ...     portfolio.trade_size, 
        ...     portfolio.trade_price, 
        ...     portfolio.trade_fees, 
        ...     filter=filter,
        ...     use_cached=portfolio.positions)
        >>> print(positions.avg_pnl)
        -2.0
        ```"""

    def __init__(self, ts_wrapper, records):
        # Check that records have position layout
        checks.assert_same_shape(records, PositionRecord, axis=(1, 0))

        super().__init__(ts_wrapper, records)

    @timeseries_property('Status')
    def status(self):
        """Status.

        See `vectorbt.portfolio.enums.PositionStatus`."""
        return self.map_to_matrix(nb.field_map_func_nb, PositionRecord.Status)

    @group_property('Open', Events)
    def open(self):
        """Open positions of type `Events`."""
        filter_mask = self.records[:, PositionRecord.Status] == PositionStatus.Open
        return Events(self.ts_wrapper, self.records[filter_mask, :])

    @group_property('Closed', Events)
    def closed(self):
        """Closed positions of type `Events`."""
        filter_mask = self.records[:, PositionRecord.Status] == PositionStatus.Closed
        return Events(self.ts_wrapper, self.records[filter_mask, :])

    @metric_property('Closed rate')
    def closed_rate(self):
        """Rate of closed positions."""
        closed_count = reshape_fns.to_1d(self.closed.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        closed_rate = closed_count / count
        return self.ts_wrapper.wrap_reduced(closed_rate)
