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


def talib(*args, **kwargs):
    """Shortcut for `vectorbt.indicators.factory.IndicatorFactory.from_talib`."""
    return IndicatorFactory.from_talib(*args, **kwargs)


def pandas_ta(*args, **kwargs):
    """Shortcut for `vectorbt.indicators.factory.IndicatorFactory.from_pandas_ta`."""
    return IndicatorFactory.from_pandas_ta(*args, **kwargs)


def ta(*args, **kwargs):
    """Shortcut for `vectorbt.indicators.factory.IndicatorFactory.from_ta`."""
    return IndicatorFactory.from_ta(*args, **kwargs)


__all__ = [
    'IndicatorFactory',
    'talib',
    'pandas_ta',
    'ta',
    'MA',
    'MSTD',
    'BBANDS',
    'RSI',
    'STOCH',
    'MACD',
    'ATR',
    'OBV'
]
__whitelist__ = [
    'talib',
    'pandas_ta',
    'ta'
]

__pdoc__ = {k: k in __whitelist__ for k in __all__}
