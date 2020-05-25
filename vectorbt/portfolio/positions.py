"""Class for measuring performance of positions."""

import numpy as np
import pandas as pd
from inspect import isfunction

from vectorbt import timeseries
from vectorbt.utils import checks, reshape_fns
from vectorbt.portfolio import nb
from vectorbt.portfolio.common import ArrayWrapper, timeseries_property, metric_property, group_property
from vectorbt.portfolio.enums import PositionStatus
from vectorbt.portfolio.events import Events


class Positions(Events):
    """Extends `vectorbt.portfolio.events.Events` with position-related properties.

    Args:
        price (pandas_like): Main price of the asset.
        trade_size (pandas_like): Trade size at each time step.
        trade_price (pandas_like): Trade price at each time step.
        trade_fees (pandas_like): Trade fees at each time step.
        filter: See `vectorbt.portfolio.events.EventMapper`.
        use_cached: See `vectorbt.portfolio.events.BaseEvents`.

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

    def __init__(self, price, trade_size, trade_price, trade_fees, filter=None, use_cached=None):
        self.price = price
        self.trade_size = trade_size
        self.trade_price = trade_price
        self.trade_fees = trade_fees

        mapper = (
            nb.map_positions_nb,
            price.vbt.to_2d_array(),
            trade_size.vbt.to_2d_array(),
            trade_price.vbt.to_2d_array(),
            trade_fees.vbt.to_2d_array()
        )
        super().__init__(price, mapper, filter=filter, use_cached=use_cached)

    @timeseries_property('Status')
    def status(self):
        """Status.

        See `vectorbt.portfolio.enums.PositionStatus`."""
        if self.use_cached is not None:
            return self._filter(self.use_cached.status)

        return self.map(nb.pos_status_map_func_nb)

    @group_property('Open', Events)
    def open(self):
        """Open positions of type `Events`."""
        filter = (
            nb.open_filter_func_nb,
            self.status.vbt.to_2d_array()
        )
        return Events(self.price, self.event_mapper, filter=filter, use_cached=self)

    @group_property('Closed', Events)
    def closed(self):
        """Closed positions of type `Events`."""
        filter = (
            nb.closed_filter_func_nb,
            self.status.vbt.to_2d_array()
        )
        return Events(self.price, self.event_mapper, filter=filter, use_cached=self)

    @metric_property('Closed rate')
    def closed_rate(self):
        """Rate of closed positions."""
        closed_count = reshape_fns.to_1d(self.closed.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        closed_rate = closed_count / count
        return self.wrap_metric(closed_rate)
