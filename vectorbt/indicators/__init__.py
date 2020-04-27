"""
Technical indicators are used to see past trends and anticipate future moves. This module provides a collection
of such indicators, but also a comprehensive `vectorbt.indicators.factory.IndicatorFactory` for 
building new indicators with ease.

Before running the examples, import the following libraries:
```py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import itertools
from numba import njit
import yfinance as yf

import vectorbt as vbt

ticker = yf.Ticker("BTC-USD")
price = ticker.history(start=datetime(2019, 3, 1), end=datetime(2019, 9, 1))

price['Close'].vbt.timeseries.plot()
```
![](img/Indicators_price.png)
"""
from vectorbt.indicators import factory, ma, mstd, bollinger_bands, rsi, stochastic, macd, obv, atr

from vectorbt.indicators.factory import IndicatorFactory
from vectorbt.indicators.ma import MA
from vectorbt.indicators.mstd import MSTD
from vectorbt.indicators.bollinger_bands import BollingerBands
from vectorbt.indicators.rsi import RSI
from vectorbt.indicators.stochastic import Stochastic
from vectorbt.indicators.macd import MACD
from vectorbt.indicators.obv import OBV
from vectorbt.indicators.atr import ATR