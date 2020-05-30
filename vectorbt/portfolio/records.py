"""Classes for working with records.

Information on orders, trades and positions can be quite rich. For example, each position has
size, opening and closing price, PnL, return, and more. Instead of creating a bunch of
large matrices to hold this information (with most elements being NaN anyway), we will store 
this information as records.

Records is just an array of arrays of fixed schema (= 2-dim NumPy array) holding some information.
You can imagine records being a DataFrame, where each row represents a record and each 
column represents a specific type of information. For example, `vectorbt.portfolio.enums.OrderRecord` 
defines the layout for order records, that is, shows at which column index to find what kind of
information:

```plaintext
OrderRecord(Column=0, Index=1, Size=2, Price=3, Fees=4, Side=5)
          +
[[0. 0. 1. 1. 0. 0.]
 [0. 1. 1. 2. 0. 0.]
 [0. 2. 1. 3. 0. 0.]
 [0. 3. 1. 2. 0. 0.]
 [0. 4. 1. 1. 0. 1.]]
          |
          v
   Column  Index  Size  Price  Fees  Side
0     0.0    0.0   1.0    1.0   0.0   0.0
1     0.0    1.0   1.0    2.0   0.0   0.0
2     0.0    2.0   1.0    3.0   0.0   0.0
3     0.0    3.0   1.0    2.0   0.0   0.0
4     0.0    4.0   1.0    1.0   0.0   1.0
```

!!! note
    Since records are stored as a single NumPy array, all columns are casted to a single 
    data type - mostly `numpy.float64`."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks, reshape_fns
from vectorbt import timeseries
from vectorbt.portfolio import nb
from vectorbt.portfolio.common import (
    TSRArrayWrapper,
    timeseries_property, 
    metric_property, 
    records_property,
    group_property
)
from vectorbt.portfolio.enums import (
    OrderRecord, 
    OrderSide, 
    EventRecord, 
    TradeRecord, 
    PositionRecord, 
    PositionStatus
)

class Records():
    """Exposes methods and properties for working with any records.

    This class doesn't hold any data, but creates a read-only view over records.
    Except that all time series and metric properties are cached.

    Args:
        wrapper (TSRArrayWrapper): Array wrapper of type `vectorbt.portfolio.common.TSRArrayWrapper`.
        records (np.ndarray): An array of records.
        layout: An instance of a `namedtuple` class that acts as a layout for the records.
        col_field (int): Field index representing a column index.
        row_field (int): Field index representing a row index."""

    def __init__(self, wrapper, records, layout, col_field, row_field):
        checks.assert_type(records, np.ndarray)
        checks.assert_same_shape(records, layout, axis=(1, 0))

        self.wrapper = wrapper
        self._records = records
        self.layout = layout
        self.col_field = col_field
        self.row_field = row_field

    @records_property('Records')
    def records(self):
        """Records."""
        return self.wrapper.wrap_records(self._records, self.layout)

    def map_records_to_matrix(self, map_func_nb, *args):
        """Map each record to a value that is then stored in a matrix.
        
        See `vectorbt.portfolio.nb.map_records_to_matrix_nb`."""
        checks.assert_numba_func(map_func_nb)

        return self.wrapper.wrap(
            nb.map_records_to_matrix_nb(
                self._records,
                (len(self.wrapper.index), len(self.wrapper.columns)),
                self.col_field,
                self.row_field,
                map_func_nb,
                *args))

    def reduce_records(self, reduce_func_nb, *args):
        """Perform a reducing operation over the records of each column.
        
        See `vectorbt.portfolio.nb.reduce_records_nb`."""
        checks.assert_numba_func(reduce_func_nb)

        return self.wrapper.wrap_reduced(
            nb.reduce_records_nb(
                self._records,
                len(self.wrapper.columns),
                self.col_field,
                reduce_func_nb,
                *args))

    @metric_property('Total count')
    def count(self):
        """Total count of all events."""
        return self.reduce_records(nb.count_reduce_func_nb)


class BaseEvents(Records):
    """Extends `Records` for working with event records."""

    def __init__(self, wrapper, records, layout=EventRecord):
        checks.assert_same(EventRecord._fields, layout._fields[:len(EventRecord)])  # subtype of EventRecord

        super().__init__(wrapper, records, layout, EventRecord.Column, EventRecord.CloseAt)

    # ############# Duration ############# #

    @timeseries_property('Duration')
    def duration(self):
        """Duration of each event (in raw format)."""
        return self.map_records_to_matrix(nb.duration_map_func_nb)

    @metric_property('Average duration')
    def avg_duration(self):
        """Average duration of an event (in time units)."""
        return self.duration.vbt.timeseries.mean(time_units=True)

    # ############# PnL ############# #

    @timeseries_property('PnL')
    def pnl(self):
        """PnL of each event."""
        return self.map_records_to_matrix(nb.field_map_func_nb, EventRecord.PnL)

    @metric_property('Total PnL')
    def total_pnl(self):
        """Total PnL of all events."""
        return self.pnl.vbt.timeseries.sum()

    @metric_property('Average PnL')
    def avg_pnl(self):
        """Average PnL of an event."""
        return self.pnl.vbt.timeseries.mean()

    def plot_pnl(self, profit_trace_kwargs={}, loss_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot PnL of each event as markers.

        Args:
            profit_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Profit" markers.
            loss_trace_kwargs (dict): Keyword arguments passed to `plotly.graph_objects.Scatter` for "Loss" markers.
            fig (plotly.graph_objects.Figure): Figure to add traces to.
            **layout_kwargs: Keyword arguments for layout."""
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.pnl.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)

    # ############# Returns ############# #

    @timeseries_property('Returns')
    def returns(self):
        """Return of each event."""
        return self.map_records_to_matrix(nb.field_map_func_nb, EventRecord.Return)

    @metric_property('Average return')
    def avg_return(self):
        """Average return of an event."""
        return self.returns.vbt.timeseries.mean()

    def plot_returns(self, profit_trace_kwargs={}, loss_trace_kwargs={}, fig=None, **layout_kwargs):
        """Plot return of each event as a marker.

        See `BaseEvents.plot_pnl`."""
        checks.assert_type(self.pnl, pd.Series)

        above_trace_kwargs = {**dict(name='Profit'), **profit_trace_kwargs}
        below_trace_kwargs = {**dict(name='Loss'), **loss_trace_kwargs}
        return self.returns.vbt.timeseries.plot_against(0, above_trace_kwargs=above_trace_kwargs, below_trace_kwargs=below_trace_kwargs)


class Events(BaseEvents):
    """Extends `BaseEvents` by further dividing events into winning and losing."""

    @group_property('Winning', BaseEvents)
    def winning(self):
        """Winning events of type `BaseEvents`."""
        filter_mask = self._records[:, EventRecord.PnL] > 0.
        return BaseEvents(self.wrapper, self._records[filter_mask, :], layout=self.layout)

    @group_property('Losing', BaseEvents)
    def losing(self):
        """Losing events of type `BaseEvents`."""
        filter_mask = self._records[:, EventRecord.PnL] < 0.
        return BaseEvents(self.wrapper, self._records[filter_mask, :], layout=self.layout)

    @metric_property('Win rate')
    def win_rate(self):
        """Rate of profitable events."""
        winning_count = reshape_fns.to_1d(self.winning.count, raw=True)
        count = reshape_fns.to_1d(self.count, raw=True)

        win_rate = winning_count / count
        return self.wrapper.wrap_reduced(win_rate)

    @metric_property('Profit factor')
    def profit_factor(self):
        """Profit factor."""
        total_win = reshape_fns.to_1d(self.winning.total_pnl, raw=True)
        total_loss = reshape_fns.to_1d(self.losing.total_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = reshape_fns.to_1d(self.count, raw=True) > 0
        total_win[np.isnan(total_win) & has_values] = 0.
        total_loss[np.isnan(total_loss) & has_values] = 0.

        profit_factor = total_win / np.abs(total_loss)
        return self.wrapper.wrap_reduced(profit_factor)

    @metric_property('Expectancy')
    def expectancy(self):
        """Average profitability."""
        win_rate = reshape_fns.to_1d(self.win_rate, raw=True)
        avg_win = reshape_fns.to_1d(self.winning.avg_pnl, raw=True)
        avg_loss = reshape_fns.to_1d(self.losing.avg_pnl, raw=True)

        # Otherwise columns with only wins or losses will become NaNs
        has_values = reshape_fns.to_1d(self.count, raw=True) > 0
        avg_win[np.isnan(avg_win) & has_values] = 0.
        avg_loss[np.isnan(avg_loss) & has_values] = 0.

        expectancy = win_rate * avg_win - (1 - win_rate) * np.abs(avg_loss)
        return self.wrapper.wrap_reduced(expectancy)


class BaseOrders(Records):
    """Extends `Records` for working with order records.

    Requires records of type `vectorbt.portfolio.enums.OrderRecord`.
    
    Example:
        Get the total number of buy and sell operations:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from vectorbt.portfolio.records import Orders
        >>> from vectorbt.portfolio.enums import OrderSide, OrderRecord as OR

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, 1, 1, 1, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders, 
        ...      init_capital=100, data_freq='1D')

        >>> print(portfolio.orders.buy.count)
        4.0
        >>> print(portfolio.orders.sell.count)
        1.0
        ```"""
    def __init__(self, wrapper, records):
        super().__init__(wrapper, records, OrderRecord, OrderRecord.Column, OrderRecord.Index)

    @timeseries_property('Size')
    def size(self):
        """Size of each order."""
        return self.map_records_to_matrix(nb.field_map_func_nb, OrderRecord.Size)

    @timeseries_property('Price')
    def price(self):
        """Price of each order."""
        return self.map_records_to_matrix(nb.field_map_func_nb, OrderRecord.Price)

    @timeseries_property('Fees')
    def fees(self):
        """Fees of each order."""
        return self.map_records_to_matrix(nb.field_map_func_nb, OrderRecord.Fees)

    @metric_property('Total fees')
    def total_fees(self):
        """Total fees of all orders."""
        return self.fees.vbt.timeseries.sum()


class Orders(BaseOrders):
    """Extends `BaseOrders` by further dividing orders into buy and sell orders."""

    @timeseries_property('Side')
    def side(self):
        """Side of each order.
        
        See `vectorbt.portfolio.enums.OrderSide`."""
        return self.map_records_to_matrix(nb.field_map_func_nb, OrderRecord.Side)

    @group_property('Buy', BaseOrders)
    def buy(self):
        """Buy operations of type `BaseOrders`."""
        filter_mask = self._records[:, OrderRecord.Side] == OrderSide.Buy
        return BaseOrders(self.wrapper, self._records[filter_mask, :])

    @group_property('Sell', BaseOrders)
    def sell(self):
        """Sell operations of type `BaseOrders`."""
        filter_mask = self._records[:, OrderRecord.Side] == OrderSide.Sell
        return BaseOrders(self.wrapper, self._records[filter_mask, :])



class Trades(Events):
    """Extends `Events` for working with trade records.

    Requires records of type `vectorbt.portfolio.enums.TradeRecord`.
    Such records can be created by using `vectorbt.portfolio.nb.trade_records_nb`.
    
    Example:
        Get the average PnL of trades with duration over 2 days:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from vectorbt.portfolio.records import Trades
        >>> from vectorbt.portfolio.enums import TradeRecord as TR

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, -1, 1, 0, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders, 
        ...      init_capital=100, data_freq='1D')
        >>> print(portfolio.trades.avg_pnl)
        -0.5

        >>> records = portfolio.trade_records.values
        >>> duration_mask = (records[:, TR.CloseAt] - records[:, TR.OpenAt]) >= 2.
        >>> filtered_records = records[duration_mask, :]
        >>> trades = Trades(portfolio.wrapper, filtered_records)
        >>> print(trades.avg_pnl)
        -2.0
        ```
        
        The same can be done by using `BaseEvents.reduce_records`, 
        which skips the step of transforming records into a matrix and thus saves memory.
        ```python-repl
        >>> import numpy as np
        >>> from numba import njit

        >>> @njit
        ... def reduce_func_nb(col_rs):
        ...     duration_mask = col_rs[:, TR.CloseAt] - col_rs[:, TR.OpenAt] >= 2.
        ...     return np.nanmean(col_rs[duration_mask, TR.PnL])

        >>> portfolio.trades.reduce_records(reduce_func_nb)
        -2.0
        ```"""

    def __init__(self, wrapper, records):
        super().__init__(wrapper, records, layout=TradeRecord)


class Positions(Events):
    """Extends `Events` for working with position records.

    Requires records of type `vectorbt.portfolio.enums.PositionRecord`.
    Such records can be created by using `vectorbt.portfolio.nb.position_records_nb`.

    Example:
        Get the average PnL of closed positions with duration over 2 days:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from vectorbt.portfolio.records import Positions
        >>> from vectorbt.portfolio.enums import PositionStatus, PositionRecord as PR

        >>> price = pd.Series([1, 2, 3, 2, 1])
        >>> orders = pd.Series([1, -1, 1, 0, -1])
        >>> portfolio = vbt.Portfolio.from_orders(price, orders, 
        ...      init_capital=100, data_freq='1D')
        >>> print(portfolio.positions.avg_pnl)
        -0.5
        
        >>> records = portfolio.position_records.values
        >>> closed_mask = records[:, PR.Status] == PositionStatus.Closed
        >>> duration_mask = (records[:, PR.CloseAt] - records[:, PR.OpenAt]) >= 2.
        >>> filtered_records = records[closed_mask & duration_mask, :]
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
        return self.wrapper.wrap_reduced(closed_rate)