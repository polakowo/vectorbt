import numpy as np
import pandas as pd

from vectorbt.utils.decorators import has_type, to_dim2
from vectorbt.utils.array import fshift, pct_change, ffill
from vectorbt.timeseries import TimeSeries
from vectorbt.positions import Positions


class Portfolio():
    """Propagate investment through positions to generate equity and returns."""

    @has_type(1, TimeSeries)
    @to_dim2(1)
    @has_type(2, Positions)
    @to_dim2(2)
    def __init__(self, ts, positions, investment=1, fees=0, slippage=0):
        self.ts = ts
        self.positions = positions
        self.investment = investment
        self.fees = fees
        self.slippage = slippage

    @property
    def equity(self):
        """Calculate running equity."""
        # Calculate equity
        pos_idx = np.nonzero(self.positions)
        equity = np.ones(self.positions.shape)
        returns_mask = fshift(ffill(self.positions), 1) == 1
        equity[returns_mask] += pct_change(self.ts)[returns_mask]
        # Apply fees and slippage
        if isinstance(self.slippage, np.ndarray):
            equity[pos_idx] *= 1 - self.slippage[pos_idx]
        else:
            equity[pos_idx] *= 1 - self.slippage
        if isinstance(self.fees, np.ndarray):
            equity[pos_idx] *= 1 - self.fees[pos_idx]
        else:
            equity[pos_idx] *= 1 - self.fees
        equity = np.cumprod(equity, axis=0)
        return TimeSeries(equity)

    @property
    def cash_equity(self):
        """Calculate running equity in cash."""
        return self.equity * self.investment

    @property
    def share_equity(self):
        """Calculate running equity in shares."""
        return self.equity * self.investment / self.ts

    @property
    def returns(self):
        """Calculate returns at each time step."""
        return pct_change(self.equity)
