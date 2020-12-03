![](https://img.shields.io/travis/polakowo/vectorbt/master.svg?branch=master&style=for-the-badge)
![](https://img.shields.io/codecov/c/github/polakowo/vectorbt?style=for-the-badge)
![](https://img.shields.io/pypi/v/vectorbt?color=blueviolet&style=for-the-badge)

# vectorbt

![Logo](https://raw.githubusercontent.com/polakowo/vectorbt/master/logo.png)

vectorbt is a backtesting library on steroids - it operates entirely on pandas and NumPy objects, and is 
accelerated by [Numba](https://github.com/numba/numba) to analyze trading strategies at speed and scale :fire:

In contrast to conventional libraries, vectorbt represents trading data as nd-arrays.
This enables superfast computation using vectorized operations with NumPy and non-vectorized but compiled 
operations with Numba, for example, for hyperparameter optimization. It also integrates 
[plotly.py](https://github.com/plotly/plotly.py) and [ipywidgets](https://github.com/jupyter-widgets/ipywidgets) 
to display complex charts and dashboards akin to Tableau right in the Jupyter notebook. Due to high 
performance, vectorbt is able to process large amounts of data even without GPU and parallelization (both are work 
in progress), and enable the user to interact with data-hungry widgets without significant delays.

With vectorbt you can
* Analyze and engineer features for any time series data
* Supercharge pandas and your favorite tools to run much faster
* Test thousands of strategies, configurations, assets, and time ranges in one go
* Test machine learning models
* Build interactive charts/dashboards without leaving Jupyter

## Basic example

We hate boilerplate code, that's why in vectorbt you can start with as little as a couple of lines.
Here is how much profit we would have made if we bought 10 Netflix shares back in 2002:

```python
import yfinance as yf
import pandas as pd
import vectorbt as vbt

price = yf.Ticker('NFLX').history(period='max')['Close']
size = pd.Series.vbt.empty_like(price, 0.)
size.iloc[0] = 1.
portfolio = vbt.Portfolio.from_orders(price, size, init_cash='auto')
portfolio.total_profit()
```

```plaintext
5021.8
```

## Advanced example

Here a snippet for testing 10,000 window combinations of a dual SMA crossover strategy on BTC, USD and XRP
from 2017 onwards, in under 5 seconds (Note: first time compiling with Numba may take a while):

```python
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import vectorbt as vbt

# Define your params
assets = ["BTC-USD", "ETH-USD", "LTC-USD"]
yf_kwargs = dict(start=datetime(2017, 1, 1))
windows = np.arange(2, 101)
portfolio_kwargs = dict(fees=0.001)

# Fetch daily price
price = {}
for asset in assets:
    price[asset] = yf.Ticker(asset).history(**yf_kwargs)['Close']
price = pd.DataFrame(price)
price.columns.name = 'asset'

# Compute moving averages for all combinations of fast and slow windows
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])

# Generate crossover signals for each combination
entries = fast_ma.ma_above(slow_ma, crossed=True)
exits = fast_ma.ma_below(slow_ma, crossed=True)

# Run simulation
portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq='1D', **portfolio_kwargs)

# Get total return, reshape to symmetric matrix, and plot the whole thing
fig = portfolio.total_return().vbt.heatmap(
    x_level='fast_window', y_level='slow_window', slider_level='asset', symmetric=True,
    trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%')))
fig.show()
```

![dmac_heatmap.gif](https://raw.githubusercontent.com/polakowo/vectorbt/master/img/dmac_heatmap.gif)

Digging into each strategy configuration is as simple as indexing with pandas:

```python
portfolio[(10, 20, 'ETH-USD')].stats()
```

```plaintext
Start                            2016-12-31 00:00:00
End                              2020-11-20 00:00:00
Duration                          1421 days 00:00:00
Init. Cash                                       100
Total Profit                                 42677.6
Total Return [%]                             42677.6
Benchmark Return [%]                         6289.46
Position Coverage [%]                         55.665
Max. Drawdown [%]                            70.7334
Avg. Drawdown [%]                            9.70645
Max. Drawdown Duration             760 days 00:00:00
Avg. Drawdown Duration    30 days 14:43:38.181818182
Num. Trades                                       33
Win Rate [%]                                 57.5758
Best Trade [%]                               477.295
Worst Trade [%]                             -27.7724
Avg. Trade [%]                               36.1783
Max. Trade Duration                 79 days 00:00:00
Avg. Trade Duration                 22 days 16:00:00
Expectancy                                   929.696
SQN                                           1.7616
Gross Exposure                               0.55665
Sharpe Ratio                                 2.27012
Sortino Ratio                                4.10306
Calmar Ratio                                 5.28868
Name: (10, 20, ETH-USD), dtype: object
```

```python
fig = portfolio[(10, 20, 'ETH-USD')].plot()
fig.update_traces(xaxis="x3")
fig.update_xaxes(spikemode='across+marker')
fig.show()
```

![dmac_portfolio.png](https://raw.githubusercontent.com/polakowo/vectorbt/master/img/dmac_portfolio.png)

## Motivation

While there are [many other great backtesting packages for Python](https://github.com/mementum/backtrader#alternatives), 
vectorbt is more of a data science tool: it excels at processing performance and offers interactive tools to explore 
complex phenomena in trading. With it you can traverse a huge number of strategy configurations, time periods and 
instruments in little time, to explore where your strategy performs best and to uncover hidden patterns in data.

Take a simple [Dual Moving Average Crossover](https://en.wikipedia.org/wiki/Moving_average_crossover) strategy 
as example. By calculating the performance of each reasonable window combination and plotting the whole thing 
as a heatmap (as we do above), we can analyze how performance depends upon window size. If we additionally 
compute the same heatmap over multiple time periods, we may observe how performance varies with downtrends 
and uptrends. Finally, by running the same pipeline over other strategies such as holding and trading randomly, 
we can compare them and decide whether our strategy is worth executing. With vectorbt, this analysis can 
be done in minutes and save time and cost of getting the same insights elsewhere.

## How it works?

vectorbt combines pandas, NumPy and Numba sauce to obtain orders-of-magnitude speedup over other libraries. 
It natively works on pandas objects, while performing all computations using NumPy and Numba under the hood. 
This way, it is often much faster than pandas alone:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> big_ts = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

# pandas
>>> %timeit big_ts.expanding().max()
48.4 ms ± 557 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# vectorbt
>>> %timeit big_ts.vbt.expanding_max()
8.82 ms ± 121 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

In contrast to most other similar backtesting libraries where backtesting is limited to simple arrays 
(think of an array for price, an array for signals, etc.), vectorbt is optimized for working with 
2-dimensional data: it treats index of a DataFrame as time axis and columns as distinct features
that should be backtest, and performs computations on the entire matrix at once, without "Pythonic" loops.

To make the library easier to use, vectorbt introduces a namespace (accessor) to pandas objects 
(see [extending pandas](https://pandas.pydata.org/pandas-docs/stable/development/extending.html)). 
This way, user can easily switch between pandas and vectorbt functionality. Moreover, each vectorbt 
method is flexible towards inputs and can work on both Series and DataFrames.

## Features

- Extends pandas using a custom `vbt` accessor
    -> Compatible with any library
- For high performance, most operations are done strictly using NumPy and Numba 
    -> Much faster than comparable operations in pandas
    
```python-repl
# pandas
>>> %timeit big_ts + 1
242 ms ± 3.58 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# vectorbt
>>> %timeit big_ts.vbt + 1
3.32 ms ± 19.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```
    
- Helper functions for combining, transforming, and indexing NumPy and pandas objects
    - NumPy-like broadcasting for pandas, among other features
    
```python-repl
# pandas
>>> pd.Series([1, 2, 3]) + pd.DataFrame([[1, 2, 3]])
   0  1  2
0  2  4  6

# vectorbt
>>> pd.Series([1, 2, 3]).vbt + pd.DataFrame([[1, 2, 3]])
   0  1  2
0  2  3  4
1  3  4  5
2  4  5  6
```
   
- Compiled versions of common pandas functions, such as rolling, groupby, and resample

```python-repl
# pandas
>>> %timeit big_ts.rolling(2).apply(np.mean, raw=True)
7.32 s ± 431 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# vectorbt
>>> mean_nb = njit(lambda col, i, x: np.mean(x))
>>> %timeit big_ts.vbt.rolling_apply(2, mean_nb)
86.2 ms ± 7.97 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

- Drawdown analysis

```python-repl
>>> pd.Series([2, 1, 3, 2]).vbt.drawdowns().plot().show()
```

![drawdowns.png](https://raw.githubusercontent.com/polakowo/vectorbt/master/img/drawdowns.png)

- Functions for working with signals
    - Entry, exit and random signal generation
    - Ranking and distance functions
    
```python-repl
>>> pd.Series([False, True, True, True]).vbt.signals.first()
0    False
1     True
2    False
3    False
dtype: bool
```

- Signal factory for building iterative signal generators
    - Also includes a range of basic generators such for random signals

```python-repl
>>> rand = vbt.RAND.run(n=[0, 1, 2], input_shape=(6,), seed=42)
>>> rand.entries
rand_n      0      1      2
0       False   True   True
1       False  False  False
2       False  False  False
3       False  False   True
4       False  False  False
5       False  False  False
>>> rand.exits
rand_n      0      1      2
0       False  False  False
1       False  False   True
2       False  False  False
3       False   True  False
4       False  False   True
5       False  False  False
```
    
- Functions for working with returns
    - Compiled versions of metrics found in [empyrical](https://github.com/quantopian/empyrical)

```python-repl
>>> pd.Series([0.01, -0.01, 0.01]).vbt.returns(freq='1D').sharpe_ratio()
5.515130702591433
```
    
- Class for modeling portfolios
    - Accepts signals, orders, and custom order function
    - Supports long and short positions
    - Supports individual and multi-asset mixed portfolios
    - Provides metrics and tools for analyzing returns, orders, trades and positions
    
```python-repl
>>> price = [1., 2., 3., 2., 1.]
>>> entries = [True, False, True, False, False]
>>> exits = [False, True, False, True, False]
>>> portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq='1D')
>>> portfolio.trades().plot().show()
```

![trades.png](https://raw.githubusercontent.com/polakowo/vectorbt/master/img/trades.png)
    
- A range of basic technical indicators with full Numba support
    - Moving average, Bollinger Bands, RSI, Stochastic, MACD, and more
    - Each offers methods for generating signals and plotting
    - Each allows arbitrary parameter combinations, from arrays to Cartesian products
    
```python-repl
>>> vbt.MA.run([1, 2, 3], window=[2, 3], ewm=[False, True]).ma
ma_window     2         3
ma_ewm    False      True 
0           NaN       NaN
1           1.5       NaN
2           2.5  2.428571
``` 

- Indicator factory for building complex technical indicators with ease
    - Supports [TA-Lib](https://github.com/mrjbq7/ta-lib) indicators out of the box
    
```python-repl
>>> SMA = vbt.IndicatorFactory.from_talib('SMA')
>>> SMA.run([1., 2., 3.], timeperiod=[2, 3]).real
sma_timeperiod    2    3
0               NaN  NaN
1               1.5  NaN
2               2.5  2.0
``` 
    
- Interactive Plotly-based widgets to visualize backtest results
    - Full integration with ipywidgets for displaying interactive dashboards in Jupyter

```python-repl
>>> a = np.random.normal(0, 4, size=10000)
>>> pd.Series(a).vbt.box(horizontal=True, trace_kwargs=dict(boxmean='sd')).show()
``` 

![Box.png](https://raw.githubusercontent.com/polakowo/vectorbt/master/img/Box.png)

## Installation

```
pip install vectorbt
```

See [Jupyter Notebook and JupyterLab Support](https://plotly.com/python/getting-started/#jupyter-notebook-support) 
for Plotly figures.

## [Documentation](https://polakowo.io/vectorbt/)

## Example notebooks

- [Assessing performance of DMAC on Bitcoin](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/BitcoinDMAC.ipynb)
- [Comparing effectiveness of stop signals](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/StopSignals.ipynb)
- [Backtesting per trading session](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/TradingSessions.ipynb)

Note: you will need to run the notebook to play with widgets.

## Dashboards

- [Detecting and backtesting common candlestick patterns](https://github.com/polakowo/vectorbt/tree/master/apps/candlestick-patterns)
