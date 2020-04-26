"""
Technical indicators are used to see past trends and anticipate future moves. This module provides a collection
of such indicators, but also a comprehensive `vectorbt.indicators.IndicatorFactory` for building new indicators
with ease.

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
from vectorbt.indicators import indicator_factory, ma, mstd, bollinger_bands, rsi, stochastic, macd, obv, atr

from vectorbt.indicators.indicator_factory import *
from vectorbt.indicators.ma import *
from vectorbt.indicators.mstd import *
from vectorbt.indicators.bollinger_bands import *
from vectorbt.indicators.rsi import *
from vectorbt.indicators.stochastic import *
from vectorbt.indicators.macd import *
from vectorbt.indicators.obv import *
from vectorbt.indicators.atr import *

