#!/usr/bin/env python3
"""
VectorBT example: Live-ready Dual Moving Average Crossover (DMAC) on BTC price data.

Usage:
  python examples/dmac_live.py
"""

import vectorbt as vbt
import pandas as pd


def run_dmac_live(symbol='BTC-USD', fast_window=50, slow_window=200, init_cash=10000, fees=0.001, freq='1h', n_rows=500):
    price = vbt.YFData.download(symbol).get('Close')
    if len(price) > n_rows:
        price = price.iloc[-n_rows:]

    fast_ma = vbt.MA.run(price, window=fast_window).ma
    slow_ma = vbt.MA.run(price, window=slow_window).ma

    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)

    portfolio = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=init_cash,
        fees=fees,
        freq=freq,
        slippage=None,
        init_positions=False
    )

    return {
        'price': price,
        'fast_ma': fast_ma,
        'slow_ma': slow_ma,
        'entries': entries,
        'exits': exits,
        'portfolio': portfolio
    }


if __name__ == '__main__':
    result = run_dmac_live(n_rows=720)
    portfolio = result['portfolio']

    print('DMAC live-like run for BTC-USD')
    print('Data points:', len(result['price']))
    print('Fast MA (last):', round(result['fast_ma'].iloc[-1], 2))
    print('Slow MA (last):', round(result['slow_ma'].iloc[-1], 2))
    print('Entries:', int(result['entries'].sum()))
    print('Exits:', int(result['exits'].sum()))
    print('Final value:', round(portfolio.final_value(), 2))
    print('Total return:', round(portfolio.total_return() * 100, 2), '%')
    print('Total trades:', int(portfolio.trades.count()))
    print('Sharpe ratio:', round(portfolio.sharpe_ratio(), 3))
    print('Max drawdown:', round(portfolio.max_drawdown() * 100, 3), '%')
