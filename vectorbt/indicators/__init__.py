"""Modules for building and running indicators.

Technical indicators are used to see past trends and anticipate future moves.
See [Using Technical Indicators to Develop Trading Strategies](https://www.investopedia.com/articles/trading/11/indicators-and-strategies-explained.asp)."""

from vectorbt.indicators.factory import IndicatorFactory
from vectorbt.indicators.basic import (
    MA,
    MSTD,
    BBANDS,
    RSI,
    STOCH,
    MACD,
    ATR,
    OBV
)

__all__ = [
    'IndicatorFactory',
    'MA',
    'MSTD',
    'BBANDS',
    'RSI',
    'STOCH',
    'MACD',
    'ATR',
    'OBV'
]

__pdoc__ = {k: False for k in __all__}
