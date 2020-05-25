"""Package for building and running indicators.

Technical indicators are used to see past trends and anticipate future moves.

See [Using Technical Indicators to Develop Trading Strategies](https://www.investopedia.com/articles/trading/11/indicators-and-strategies-explained.asp)."""

from vectorbt.indicators import factory, indicators

from vectorbt.indicators.factory import IndicatorFactory
from vectorbt.indicators.indicators import MA, MSTD, BollingerBands, RSI, Stochastic, MACD, ATR, OBV