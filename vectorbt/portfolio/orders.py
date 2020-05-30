"""Class for measuring performance of trades."""

from vectorbt.portfolio import nb
from vectorbt.portfolio.records import Records
from vectorbt.portfolio.enums import OrderRecord, OrderSide
from vectorbt.portfolio.common import (
    timeseries_property, 
    metric_property, 
    group_property
)


class BaseOrders(Records):
    """Extends `vectorbt.portfolio.records.Records` for working with order records.

    For details on creation, see `vectorbt.portfolio.nb.simulate_nb`.

    Requires records of type `vectorbt.portfolio.enums.OrderRecord`.
    
    Example:
        Get the total number of buy and sell operations:
        ```python-repl
        >>> import vectorbt as vbt
        >>> import pandas as pd
        >>> from vectorbt.portfolio.orders import Orders
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
        """See `vectorbt.portfolio.enums.OrderSide`."""
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
