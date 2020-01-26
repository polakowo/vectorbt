import numpy as np
import pandas as pd

from vectorbt.utils.array import *
from vectorbt.timeseries import TimeSeries
from vectorbt.positions import Positions


class Portfolio():
    """Propagate investment through positions to generate equity and returns."""

    def __init__(self, ts, positions, investment=1, fees=0, slippage=0):
        if not isinstance(ts, TimeSeries):
            raise TypeError("Argument ts is not TimeSeries")
        if not isinstance(positions, Positions):
            raise TypeError("Argument positions is not Positions")
        if isinstance(slippage, np.ndarray) and not isinstance(slippage, TimeSeries):
            raise TypeError("Argument slippage is not TimeSeries")
        # Safety check whether index of ts and positions is the same
        if not np.array_equal(ts.index, positions.index):
            raise ValueError("Arguments ts and positions do not share the same index")
        if isinstance(slippage, np.ndarray) and not np.array_equal(ts.index, slippage.index):
            raise ValueError("Arguments ts and slippage do not share the same index")

        self.ts = ts
        self.positions = positions
        self.investment = investment
        self.fees = fees
        self.slippage = slippage

    @property
    def equity(self):
        """Calculate running equity."""
        # Calculate equity
        pos_idx = np.flatnonzero(self.positions)
        equity = np.ones(len(self.positions))
        returns_mask = fshift(ffill(self.positions), 1) == 1
        equity[returns_mask] += pct_change(self.ts)[returns_mask]
        # Apply fees and slippage
        if isinstance(self.slippage, np.ndarray):
            equity[pos_idx] *= 1 - self.slippage[pos_idx]
        else:
            equity[pos_idx] *= 1 - self.slippage
        equity[pos_idx] *= 1 - self.fees
        equity = np.cumprod(equity)
        # NaN before first position
        equity[:pos_idx[0]] = 1
        return TimeSeries(equity, index=self.ts.index)

    @property
    def cash_equity(self):
        """Calculate running equity in cash."""
        return TimeSeries(self.equity * self.investment, index=self.ts.index)

    @property
    def share_equity(self):
        """Calculate running equity in shares."""
        return TimeSeries(self.equity * self.investment / self.ts, index=self.ts.index)

    @property
    def returns(self):
        """Calculate returns at each time step."""
        return TimeSeries(pct_change(self.equity), index=self.ts.index)

    @property
    def position_returns(self):
        """Calculate returns on positions."""
        nonzero_idxs = np.flatnonzero(self.positions)
        position_equity = (np.abs(self.positions) * self.equity)[nonzero_idxs]
        position_returns = pct_change(np.insert(position_equity, 0, 1))[1:]
        return TimeSeries(position_returns, index=self.ts.index[nonzero_idxs])
