![](https://img.shields.io/travis/polakowo/vectorbt/master.svg?branch=master&style=for-the-badge)
![](https://img.shields.io/badge/version-0.9-blue?style=for-the-badge)

# vectorbt

![Made by Vectors Market](logo.png)

vectorbt is a backtesting library on steroids - it operates entirely on pandas and NumPy, and is accelerated by [Numba](https://github.com/numba/numba) to backtest and analyze trading strategies at speed and scale :fire:

It follows a unique approach to backtesting that builds upon vectorized matrix calculations and fast iterative processing with Numba for event-driven backtesting. It also integrates [plotly.py](https://github.com/plotly/plotly.py) and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) to display complex charts and dashboards akin to Tableau right in the Jupyter notebook. Due to its high processing performance, vectorbt is able to process data on the fly and thus enable the user to interact with data-hungry widgets without significant delays.

## Motivation

While there are [many other great backtesting packages for Python](https://github.com/mementum/backtrader#alternatives), vectorbt is more of a data mining tool: it excels at processing performance and offers interactive tools to explore complex phenomena in trading. With it you can traverse a huge number of strategy configurations, time periods and instruments in little time, to explore where your strategy performs best and to uncover hidden patterns in data.

Take a simple [Dual Moving Average Crossover](https://en.wikipedia.org/wiki/Moving_average_crossover) strategy for example. By calculating the performance of each reasonable window combination and plotting the whole thing as a heatmap (as we do below), you can easily identify how performance depends on window size. If you additionally calculate the same heatmap over multiple time periods, you will spot how performance varies with downtrends and uptrends. By doing the same for other strategies such as holding and trading randomly, you can compare them using significance tests. With vectorbt, this analysis can be done in minutes, and will effectively save you hours of getting the same insights using other libraries.

### Example

Here a snippet for testing 4851 window combinations of a dual SMA crossover strategy on the whole Bitcoin history in under 5 seconds (Note: compiling with Numba may take a while):

```python
import vectorbt as vbt
import numpy as np
import yfinance as yf

# Define params
windows = np.arange(2, 101)
init_capital = 100 # in $
fees = 0.001 # in %

# Prepare data
ticker = yf.Ticker("BTC-USD")
price = ticker.history(period="max")['Close']

# Generate signals
fast_ma, slow_ma = vbt.MA.from_combinations(price, windows, 2)
entries = fast_ma.ma_above(slow_ma, crossed=True)
exits = fast_ma.ma_below(slow_ma, crossed=True)

# Calculate performance
portfolio = vbt.Portfolio.from_signals(price, entries, exits, init_capital=init_capital, fees=fees)
performance = portfolio.total_return * 100

# Plot heatmap
perf_matrix = performance.vbt.unstack_to_df(
    index_levels='ma1_window', 
    column_levels='ma2_window', 
    symmetric=True)
perf_matrix.vbt.Heatmap(
    xaxis_title='Slow window', 
    yaxis_title='Fast window', 
    trace_kwargs=dict(colorbar=dict(title='Total return in %')),
    width=600, height=450)
```

![dmac_heatmap.png](dmac_heatmap.png)

## How it works?

vectorbt combines pandas, NumPy and Numba sauce to obtain orders-of-magnitude speedup over other libraries.

It natively works on pandas objects, while performing all calculations using NumPy and Numba under the hood. It introduces a namespace (accessor) to pandas objects (see [extending pandas](https://pandas.pydata.org/pandas-docs/stable/development/extending.html)). This way, user can easily switch betweeen native pandas functionality such as indexing, and highly-performant vectorbt methods. Moreover, each vectorbt method is flexible and can work on both Series and DataFrames.

In contrast to most other vectorized backtesting libraries where backtesting is limited to simple arrays (think of an array for price, an array for signals, an array for equity, etc.), vectorbt is optimized for working with 2-dimensional data: it treats each index of a DataFrame as time axis and each column as a distinct feature that should be backtested, and performs calculations on the entire matrix at once. This way, user can construct huge matrices with millions of columns and calculate the performance for each one with a single matrix operation, without any Pythonic loops. This is the magic behind backtesting thousands of window combinations at once, as we did above.

### Why not only pandas?

While there is a subset of pandas functionality that is already compiled with Cython or Numba, such as window functions, it cannot be accessed within user-defined Numba code, since Numba cannot do any compilation on pandas objects. Take for example generating trailing stop orders: to calculate expanding maximum for each order, you cannot do `df.expanding().max()` from within Numba, but write and compile your own expanding max function wrapped with `@njit`. That's why vectorbt also provides an arsenal of Numba-compiled functions that are ready to be used everywhere.

Moreover, compared to NumPy, some pandas operations may be extremely slow compared to their NumPy counterparts; for example, the `pct_change` operation in NumPy is nearly 70 times faster than its pandas equivalent:

```
a = np.random.uniform(size=(1000, 1000))
a_df = pd.DataFrame(a)

>>> %timeit np.diff(a, axis=0) / a[:-1, :]
3.69 ms ± 110 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

>>> %timeit a_df.pct_change()
266 ms ± 7.26 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Hence, vectorbt uses NumPy + Numba wherever possible in the backtesting pipeline.

#### Broadcasting and indexing

The other problem relies in broadcasting rules implemented in pandas: they are less flexible than in NumPy. Also, pandas follows strict rules regarding indexing; for example, you will have issues using multiple DataFrames with different index/columns in the same operation, but such operations are quite common in backtesting (think of combining signals from different indicators, each having columns of the same cardinality but different labels).

To solve this, vectobt borrows broadcasting rules from NumPy and implements itws own indexing rules that allow operations between pandas objects of the same shape, regardless of their index/columns - those are simply stacked upon each other in the resulting object.

For example, consider the following DataFrames with different index/columns:

```python
df = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
    index=pd.Index(['x', 'y', 'z'], name='idx'), 
    columns=pd.Index(['a', 'b', 'c'], name='cols'))
    
df2 = pd.DataFrame(
    [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
    index=pd.Index(['x2', 'y2', 'z2'], name='idx2'), 
    columns=pd.Index(['a2', 'b2', 'c2'], name='cols2'))
```

Addition operation using pandas yields NaNs:

```
>>> print(df + df2)

     a  a2   b  b2   c  c2
x  NaN NaN NaN NaN NaN NaN
x2 NaN NaN NaN NaN NaN NaN
y  NaN NaN NaN NaN NaN NaN
y2 NaN NaN NaN NaN NaN NaN
z  NaN NaN NaN NaN NaN NaN
z2 NaN NaN NaN NaN NaN NaN
```

Addition operation using vectorbt yields correct results:

```
>>> print(df.vbt + df2.vbt)

cols       a   b   c
cols2     a2  b2  c2
idx idx2            
x   x2     2   4   6
y   y2     8  10  12
z   z2    14  16  18
```

### Why not only NumPy?

Working with NumPy alone, from the user's point of view, is problematic, since important information in form of index and columns and all indexing checks must be explicitly handled by the user, making analysis prone to errors.

But also, vectorized implementation is hard to read or cannot be properly defined at all, and one must rely on an iterative approach instead, which is processing data in element-by-element fashion. That's where Numba comes into play: it allows both writing iterative code and compiling slow Python loops to be run at native machine code speed.

The [previous versions](https://github.com/polakowo/vectorbt/tree/9f270820dd3e5dc4ff5468dbcc14a29c4f45f557) of vectorbt were written in pure NumPy which led to more performance but less usability.

## Features

- Extends pandas using a custom `vbt` accessor
- For high performance, most operations are done stricly using NumPy and Numba 
- Provides a collection of utility functions for working with data
    - Implements NumPy-like broadcasting for pandas, among other features.
- `vbt.timeseries` accessor for working with time series data
    - Compiled versions of common pandas functions, such as rolling, groupby, and resample
- `vbt.signals` accessor for working with signals data
    - Entry, exit and random signal generation, ranking and distance functions
    - Generation of stop loss, trailing stop and take profit signals
- `vbt.portfolio` accessor for modeling portfolio performance
    - Accepts signals, orders, or custom order function
    - Provides common financial risk and performance metrics for returns, orders, trades and positions
- Provides a range of technical indicators with full Numba support
    - Moving average and STD, Bollinger Bands, RSI, Stochastic Oscillator, MACD, and more.
    - Each indicator offers methods for generating signals and plotting
    - Each indicator accepts arbitrary parameter combinations, such as single values, arrays, or Cartesian product
    - Indicator factory for construction of complex technical indicators in a simplified way
- Interactive Plotly-based widgets to visualize backtest results
    - Indicator, Bar, Scatter, Histogram and Heatmap
    - Each provides a method for efficiently updating data
    - Full integration with ipywidgets for displaying interactive dashboards in Jupyter

## Installation

```
pip install git+https://github.com/polakowo/vectorbt.git
```

See [Jupyter Notebook and JupyterLab Support](https://plotly.com/python/getting-started/#jupyter-notebook-support) for Plotly figures.

## [Documentation](https://polakowo.io/vectorbt/)

## Example notebooks

- [Who beats Bitcoin: Dual moving average crossover, trading randomly or holding?](examples/Bitcoin-DMAC.ipynb)
- [How stop-loss and trailing stop orders perform on crypto?](examples/StopLoss-vs-TrailingStop.ipynb)

Note: you will need to run the notebook to play with widgets.

## Credits

- Logo made by [Freepik](https://www.flaticon.com/authors/freepik) from [Flaticon](www.flaticon.com)
