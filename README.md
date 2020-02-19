# vectorbt

This package shares a similar idea as behind [most other Python backtesting packages](https://github.com/mementum/backtrader#alternatives), but designed especially for fast strategy backtesting, tuning and comparison at scale. It combines numpy and Numba magic to obtain orders-of-magnitude speedup over pandas; for example, it takes less time to calculate a rolling window over a `(100, 10)` array (207 µs) than initialize a single `pd.DataFrame` object (220 µs). Furthermore, it integrates [plotly.py](https://github.com/plotly/plotly.py) and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) to build interactive charts and complex dashboards. Due to its high processing performance, vectorbt is able to re-calculate data on the fly, thus enabling the user to interact with data-hungry widgets without significant delays.

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

vbt.Heatmap(windows, windows, data=tnp_matrix, figsize=(600, 450)).show_png()
```

![msft_heatmap.png](msft_heatmap.png)

A more advanced example can be found [here](hello).

#### How it works?

Each vectorbt class is a subclass of `np.ndarray` with a custom set of methods optimized for working with time series data. For example, the `Signals` class is a binary NumPy array supporting advanced binary operations. Each method is either vectorized or Numba compiled for best peformance; most of the times even a badly "looped" Numba is faster than vectorized NumPy though. Moreover, each object is stricly a 2-dimensional array, where first axis is index (time) and second axis are columns (features). Thus, similar to a `pd.DataFrame`, one can do a single operation to transform tons of columns simultaneously. This, for example, is the magic behind backtesting thousands of window combinations at once.