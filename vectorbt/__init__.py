from vectorbt import timeseries, signals, indicators, portfolio, widgets, utils, accessors

from vectorbt.indicators import IndicatorFactory, MA, MSTD, BollingerBands, RSI, Stochastic, MACD, OBV, ATR
from vectorbt.portfolio import Portfolio
from vectorbt.widgets import Indicator, Bar, Scatter, Histogram, Heatmap

# Add documentation whitelist, must be at the end!

widgets.__pdoc__ = {}
utils.add_all_from_module(widgets.__pdoc__, widgets, whitelist=[
    'FigureWidget',
    'UpdatableFigureWidget',
    'Indicator',
    'Bar',
    'Scatter',
    'Histogram',
    'Heatmap'
])
indicators.__pdoc__ = {}
utils.add_all_from_module(indicators.__pdoc__, indicators, whitelist=[
    'from_params_pipeline',
    'IndicatorFactory',
    'MA',
    'MSTD',
    'BollingerBands',
    'RSI',
    'Stochastic',
    'MACD',
    'OBV',
    'ATR'
])
