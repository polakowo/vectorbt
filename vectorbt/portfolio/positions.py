"""Class for measuring performance of positions."""

import numpy as np
import pandas as pd
from inspect import isfunction

from vectorbt import timeseries
from vectorbt.utils import reshape_fns
from vectorbt.portfolio import nb
from vectorbt.portfolio.enums import PositionStatus, PositionRecord
from vectorbt.portfolio.events import BaseEvents, Events
from vectorbt.portfolio.common import (
    timeseries_property, 
    metric_property, 
    group_property
)


class Positions(Events):
    """Extends `vectorbt.portfolio.events.Events` for working with position records.

    For details on creation, see `vectorbt.portfolio.nb.position_records_nb`.

    Requires records of type `vectorbt.portfolio.enums.PositionRecord`.

    Example:
        Get the average P&L of closed positions with duration over 2 days:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from vectorbt.portfolio.positions import Positions
        >>> from vectorbt.portfolio.enums import PositionStatus, PositionRecord as PR

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, -1, 1, 0, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders, 
        ...      init_capital=100, data_freq='1D')
        >>> print(portfolio.positions.avg_pnl)
        -0.5

        >>> def filter_records(r):
        ...      closed_mask = r[:, PR.Status] == PositionStatus.Closed
        ...      duration_mask = (r[:, PR.CloseAt] - r[:, PR.OpenAt]) >= 2.
        ...      return r[closed_mask & duration_mask, :]

        >>> filtered_records = filter_records(portfolio.position_records.values)
        >>> positions = Positions(portfolio.wrapper, filtered_records)
        >>> print(positions.avg_pnl)
        -2.0
        ```"""

    def __init__(self, wrapper, records):
        super().__init__(wrapper, records, layout=PositionRecord)

    @timeseries_property('Status')
    def status(self):
        """See `vectorbt.portfolio.enums.PositionStatus`."""
        return self.map_records_to_matrix(nb.field_map_func_nb, PositionRecord.Status)

    @group_property('Open', Events)
    def open(self):
        """Open positions of type `Events`."""
        filter_mask = self._records[:, PositionRecord.Status] == PositionStatus.Open
        return Events(self.wrapper, self._records[filter_mask, :], layout=self.layout)

    @group_property('Closed', Events)
    def closed(self):
        """Closed positions of type `Events`."""
        filter_mask = self._records[:, PositionRecord.Status] == PositionStatus.Closed
        return Events(self.wrapper, self._records[filter_mask, :], layout=self.layout)

    @metric_property('Closed rate')
    def closed_rate(self):
        """Rate of closed positions."""
        closed_count = reshape_fns.to_1d(self.closed.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        closed_rate = closed_count / count
        return self.wrapper.wrap_metric(closed_rate)
