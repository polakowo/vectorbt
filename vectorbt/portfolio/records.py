"""Class for working with records."""

import numpy as np
import pandas as pd

from vectorbt.utils import checks
from vectorbt.portfolio import nb
from vectorbt.portfolio.common import (
    PArrayWrapper,
    timeseries_property, 
    metric_property, 
    records_property,
    group_property,
    PropertyTraverser
)

class Records(PropertyTraverser):
    """Exposes methods and properties for working with any records.

    Args:
        wrapper (PArrayWrapper): Array wrapper of type `vectorbt.portfolio.common.PArrayWrapper`.
        records (np.ndarray): An array of records.
        layout: An instance of a `namedtuple` class that fits the records.
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
        """Records of layout `Records.layout`."""
        return self.wrapper.wrap_records(self._records, self.layout)

    def map_records_to_matrix(self, map_func_nb, *args):
        """Reshape records into a matrix.
        
        See `vectorbt.portfolio.nb.map_records_to_matrix_nb`."""
        checks.assert_numba_func(map_func_nb)

        return self.wrapper.wrap_timeseries(
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

        return self.wrapper.wrap_metric(
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