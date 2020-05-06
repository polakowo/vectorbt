"""
The majority of the backtesting is done here. It can receive a set of signals, orders or 
a custom order function, and create a series of positions, allocated against a cash component. 
The job of this module is to produce an equity curve.
"""
from vectorbt.portfolio import nb, portfolio

from vectorbt.portfolio.portfolio import Portfolio