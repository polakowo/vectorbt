"""Common functions and classes."""

import numpy as np
import pandas as pd

from vectorbt.utils import reshape_fns
from vectorbt.tseries.common import TSArrayWrapper
from vectorbt.records import nb


def indexing_on_records(obj, pd_indexing_func):
    """Perform indexing on `Records`."""
    if obj.wrapper.ndim == 1:
        raise Exception("Indexing on Series is not supported")

    n_rows = len(obj.wrapper.index)
    n_cols = len(obj.wrapper.columns)
    col_mapper = obj.wrapper.wrap(np.broadcast_to(np.arange(n_cols), (n_rows, n_cols)))
    col_mapper = pd_indexing_func(col_mapper)
    if not pd.Index.equals(col_mapper.index, obj.wrapper.index):
        raise Exception("Changing index (time axis) is not supported")

    new_cols = reshape_fns.to_1d(col_mapper.values[0])  # array required
    records = nb.select_record_cols_nb(
        obj.records_arr,
        obj.records_col_index,
        new_cols
    )
    wrapper = TSArrayWrapper.from_obj(col_mapper, freq=obj.wrapper.freq)
    return records, wrapper
