# vectorbt

![Made by Vectors Market](logo.png)

This package shares a similar idea as behind [most other Python backtesting packages](https://github.com/mementum/backtrader#alternatives), but designed especially for fast strategy backtesting, tuning and comparison at scale. 

It builds upon numpy and Numba to obtain orders-of-magnitude speedup over pandas. Furthermore, it integrates [plotly.py](https://github.com/plotly/plotly.py) and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) to build interactive charts and complex dashboards. Due to its high processing performance, vectorbt is able to re-calculate data on the fly, thus enabling the user to interact with data-hungry widgets without significant delays.

Here a snippet for testing 4851 window combinations of a dual SMA crossover strategy on the whole Microsoft stock history in about 5 seconds:

```python
import vectorbt as vbt
import numpy as np
import itertools
import yfinance as yf

# Prepare data
msft = yf.Ticker("MSFT")
history = msft.history(period="max")
ohlcv = vbt.OHLCV.from_df(history)
investment = 100 # $

# Create window combinations
windows = np.arange(2, 101)
comb = itertools.combinations(np.arange(len(windows)), 2) # twice less params
fast_idxs, slow_idxs = np.asarray(list(comb)).transpose()
fast_windows, slow_windows = windows[fast_idxs], windows[slow_idxs]

# Calculate the performance of the strategy
dmac = vbt.DMAC(ohlcv.open, fast_windows, slow_windows)
entries, exits = dmac.crossover_signals()
positions = vbt.Positions.from_signals(entries, exits)
portfolio = vbt.Portfolio(ohlcv.open, positions, investment=investment)
tnp = portfolio.total_net_profit

# Plot heatmap
tnp_matrix = np.empty((len(windows), len(windows)))
tnp_matrix[fast_idxs, slow_idxs] = tnp
tnp_matrix[slow_idxs, fast_idxs] = tnp # symmetry

vbt.Heatmap(data=tnp_matrix, x_labels=windows, y_labels=windows, width=600, height=450).show_png()
```

![msft_heatmap.png](msft_heatmap.png)

## Motivation

As data scientist and recent trader, I've been curious of how effective is technical analysis. I wanted answers to general questions like "How this strategy compares to the other ones? What technical indicators are best and what are worst for this market? Is technical analysis of any use at all, or is it just a buzzword and everything in the market is fully governed by random choice?"

To answer these and more, you need to set up experiments where you traverse thousands or even millions of parameter combinations, time ranges and markets, to see what performs best where. While there are many great backtesting libraries for Python, I found none that could handle these amounts of tests in a timely manner. 

The idea behind vectorbt is to create a backtesting library that operates entirely on NumPy arrays and is powered by Numba to deliver backtesting at scale. It also integrates Plotly to display charts and dashbaords akin to Tableau right in the Jupyter notebook.

## How it works?

Each vectorbt class is a subclass of `np.ndarray` with a custom set of methods optimized for working with time series data. For example, the `Signals` class is a binary NumPy array supporting advanced binary operations. Each method is either vectorized or Numba compiled for best peformance; most of the times even a badly "looped" Numba is faster than vectorized NumPy though. Moreover, each class stricly accepts a 2-dimensional array, where first axis is index (time) and second axis are columns (features), and provides standardized methods for processing 2-dimensional data along first axis. Thus, similar to a `pd.DataFrame`, one can do a single operation to transform tons of columns simultaneously. This, for example, is the magic behind backtesting thousands of window combinations at once.

For more details, check [tests](tests/Modules.ipynb).

## Installation

```
pip install git+https://github.com/polakowo/vectorbt.git
```

Note: importing vectorbt for the first time may take a while due to compilation.

## Examples

- [Testing Dual Moving Average Crossover (DMAC) strategy on Bitcoin](examples/Bitcoin_DMAC.ipynb)
- [Testing stop-loss and trailing stop orders](examples/StopLoss.ipynb)

Note: you will need to run the notebook to play with widgets.
