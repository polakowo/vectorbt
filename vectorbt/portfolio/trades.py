"""Class for measuring performance of trades."""

from vectorbt.portfolio.enums import TradeRecord
from vectorbt.portfolio.events import Events
from vectorbt.portfolio.common import (
    timeseries_property, 
    metric_property, 
    group_property
)



class Trades(Events):
    """Extends `vectorbt.portfolio.events.Events` for working with trade records.

    For details on creation, see `vectorbt.portfolio.nb.trade_records_nb`.

    Requires records of type `vectorbt.portfolio.enums.TradeRecord`.
    
    Example:
        Get the average P&L of trades with duration over 2 days:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from vectorbt.portfolio.trades import Trades
        >>> from vectorbt.portfolio.enums import TradeRecord as TR

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, -1, 1, 0, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders, 
        ...      init_capital=100, data_freq='1D')
        >>> print(portfolio.trades.avg_pnl)
        -0.5

        >>> def filter_records(r):
        ...      duration_mask = (r[:, TR.CloseAt] - r[:, TR.OpenAt]) >= 2.
        ...      return r[duration_mask, :]

        >>> filtered_records = filter_records(portfolio.trade_records.values)
        >>> trades = Trades(portfolio.wrapper, filtered_records)
        >>> print(trades.avg_pnl)
        -2.0
        ```
        
        The same can be done by using `vectorbt.portfolio.events.BaseEvents.reduce_records`, 
        which skips the step of transforming records into a matrix and thus saves memory.
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit

        >>> @njit
        ... def reduce_func_nb(r):
        ...     duration_mask = r[:, TR.CloseAt] - r[:, TR.OpenAt] >= 2.
        ...     return np.nanmean(r[duration_mask, TR.PnL])

        >>> portfolio.trades.reduce_records(reduce_func_nb)
        -2.0
        ```"""

    def __init__(self, wrapper, records):
        super().__init__(wrapper, records, layout=TradeRecord)
