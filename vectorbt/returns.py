import numpy as np
import pandas as pd

def from_equity(equity_sr):
    """Generate returns from equity"""
    return equity_sr.pct_change().fillna(0)


def resample(returns_sr, period):
    """Resample returns"""
    return (returns_sr + 1).cumprod().resample(period).last().pct_change().fillna(0)


def plot(returns_sr):
    from vectorbt import graphics

    graphics.plot_line(returns_sr, benchmark=0)
