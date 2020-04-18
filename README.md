![](https://img.shields.io/badge/version-0.5-blue)

# vectorbt

![Made by Vectors Market](logo.png)

vectorbt is a backtesting library on steroids - it operates entirely on pandas and NumPy, and is accelerated by [Numba](https://github.com/numba/numba) to backtest and analyze trading strategies at scale. It also integrates [plotly.py](https://github.com/plotly/plotly.py) and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) to display complex charts and dashboards akin to Tableau right in the Jupyter notebook. Due to its high processing performance, vectorbt is able to re-calculate data on the fly, thus enabling the user to interact with data-hungry widgets without significant delays.

## Motivation

While there are [many other great backtesting packages for Python](https://github.com/mementum/backtrader#alternatives), vectorbt is more of a data mining tool: it excels at processing performance and offers interactive tools to explore complex phenomena in trading. With it you can traverse a huge number of parameter combinations, time periods and instruments in no time, to explore where your strategy performs best and to uncover hidden patterns in data.

Take a simple [Dual Moving Average Crossover](https://en.wikipedia.org/wiki/Moving_average_crossover) strategy for example. By calculating the performance of each reasonable window combination and plotting the whole thing as a heatmap (as we do below), you can easily identify how performance depends on window size. If you additionally calculate the same heatmap over multiple time periods, you will spot how performance varies with downtrends and uptrends. By doing the same for other strategies such as holding and trading randomly, you can compare them using significance tests. With vectorbt, this analysis can be done in minutes, and will effectively save you hours of getting the same insights using other libraries.

Here a snippet for testing 4851 window combinations of a dual SMA crossover strategy on the whole Microsoft stock history in about 10 seconds:

```python
import vectorbt as vbt
import numpy as np
import yfinance as yf

# Define params
windows = np.arange(2, 101)
investment = 100 # in $
commission = 0.001 # in %

# Prepare data
msft = yf.Ticker("MSFT")
df = msft.history(period="max")
price = df['Open']

# Calculate the performance of the strategy
dmac = vbt.DMAC.from_combinations(price, windows)
entries, exits = dmac.crossover()
portfolio = vbt.Portfolio.from_signals(price, entries, exits, investment=investment, commission=commission)
performance = portfolio.total_net_profit

# Plot heatmap
tnp_df = performance.vbt.unstack_to_df(symmetric=True)
tnp_df.vbt.Heatmap(width=600, height=450).show_png()
```

![msft_heatmap.png](msft_heatmap.png)

## How it works?

vectorbt combines pandas, NumPy and Numba sauce to obtain orders-of-magnitude speedup over other libraries. It takes advantage of the vectorized nature of time series data such as price and signals, and implements Numba-compiled functions for traversing matrices along their index and column axes. 

In contrast to most other vectorized backtesting libraries, where backtesting is limited to simple arrays, vectorbt is optimized for working with 2-dimensional data. It treats each index of a dataframe as time and each column as a distinct feature that should be backtested, and performs calculations on the entire matrix at once. This way, user can construct huge matrices with millions of columns (such as parameter combinations, strategy instances, etc.) and calculate their performance with a single operation, without any loops. This, for example, is the magic behind backtesting thousands of window combinations at once.

### Efficiency

Using pandas and NumPy alone is not enough. Often, vectorized implementation is hard to read or cannot be properly defined at all, and one must rely on an iterative approach instead, where processing of a matrix in a element-by-element fashion is a must. That's where Numba comes into play: it compiles slow Python loops to be run at native machine code speed. While there is a subset of pandas functionality such as window functions that is compiled with Cython and/or Numba, it cannot be accessed within a user-defined code, since Numba cannot do any compilation on pandas objects.

But what about pandas vs NumPy? Some operations may be extremely slow compared to their NumPy counterparts; for example, the `pct_change` operation in NumPy is nearly 70 times faster than its pandas equivalent:

```
a = np.random.randint(10, size=(1000, 1000)).astype(float)
a_df = pd.DataFrame(a)

%timeit np.diff(a, axis=0) / a[:-1, :]
3.69 ms ± 110 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

%timeit a_df.pct_change()
266 ms ± 7.26 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Hence, vectorbt uses NumPy + Numba wherever possible in the backtesting pipeline and offers an arsenal of advanced Numba-compiled functions ready to be used in a user-defined code. Pandas is mainly used for wrapping the resulted NumPy arrays and for high-level operations that are outside of the pipeline, require advanced indexing, or just for convenience.

### Usability

Working with NumPy alone, from the user's point of view, is problematic, since important information in form of index and columns and all indexing checks must be explicitly handled by the user, making analysis prone to errors. In order to use pandas objects but still be able to profit from optimized code, vectorbt introduces a namespace (accessor) to pandas objects (see [extending pandas](https://pandas.pydata.org/pandas-docs/stable/development/extending.html)). This way, user can easily switch betweeen native pandas functionality such as advanced indexing, and highly-performant vectorbt methods. Moreover, each vectorbt method is flexible and can work on both series and dataframes.

For more details, check [tests](tests/Modules.ipynb).

## Installation

```
pip install git+https://github.com/polakowo/vectorbt.git
```

See [Jupyter Notebook and JupyterLab Support](https://plotly.com/python/getting-started/#jupyter-notebook-support) for Plotly figures.

Note: importing vectorbt for the first time may take a while due to compilation.

## Examples

- [Testing Dual Moving Average Crossover (DMAC) strategy on Bitcoin](examples/Bitcoin-DMAC.ipynb)
- [Comparing stop-loss and trailing stop orders](examples/StopLoss-vs-TrailingStop.ipynb)

Note: you will need to run the notebook to play with widgets.
