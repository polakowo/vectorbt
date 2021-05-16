[![Build Status](https://img.shields.io/travis/polakowo/vectorbt/master.svg?branch=master&style=for-the-badge)](https://travis-ci.org/github/polakowo/vectorbt)
[![Code Coverage](https://img.shields.io/codecov/c/github/polakowo/vectorbt?style=for-the-badge)](https://codecov.io/gh/polakowo/vectorbt)
[![Website](https://img.shields.io/website?style=for-the-badge&url=https%3A%2F%2Fpolakowo.io%2Fvectorbt%2Fdocs%2Findex.html)](https://polakowo.io/vectorbt)
[![PyPi](https://img.shields.io/pypi/v/vectorbt?color=blueviolet&style=for-the-badge)](https://pypi.org/project/vectorbt)
[![Downloads](https://img.shields.io/pypi/dm/vectorbt?color=orange&style=for-the-badge)](https://pepy.tech/project/vectorbt)
[![License](https://img.shields.io/pypi/l/vectorbt?color=yellow&style=for-the-badge)](https://github.com/polakowo/vectorbt/blob/master/LICENSE)
[![Gitter](https://img.shields.io/gitter/room/polakowo/vectorbt?color=9cf&style=for-the-badge)](https://gitter.im/vectorbt/community)
[![Binder](https://img.shields.io/badge/launch-binder-d6604a?style=for-the-badge)](https://mybinder.org/v2/gh/polakowo/vectorbt/HEAD?urlpath=lab)
[![Patreon](https://img.shields.io/badge/support-sponsor-ff69b4?style=for-the-badge)](https://www.patreon.com/vectorbt)

# vectorbt

![Logo](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/favicon/favicon-100x100-dark.png)

vectorbt is a backtesting library on steroids - it operates entirely on pandas and NumPy objects, and is 
accelerated by [Numba](https://github.com/numba/numba) to analyze time series at speed and scale :fire:

In contrast to conventional libraries, vectorbt represents any data as nd-arrays.
This enables superfast computation using vectorized operations with NumPy and non-vectorized but dynamically 
compiled operations with Numba. It also integrates [plotly.py](https://github.com/plotly/plotly.py) and 
[ipywidgets](https://github.com/jupyter-widgets/ipywidgets) to display complex charts and dashboards akin 
to Tableau right in the Jupyter notebook. Due to high performance, vectorbt is able to process large amounts of 
data even without GPU and parallelization, and enable the user to interact with data-hungry widgets 
without significant delays.

With vectorbt you can
* Analyze time series and engineer new features
* Supercharge pandas and your favorite tools to run much faster
* Test many trading strategies, configurations, assets, and periods in one go
* Test machine learning models
* Build interactive charts/dashboards without leaving Jupyter

## Installation

```bash
pip install vectorbt
```

To also install optional dependencies:

```bash
pip install vectorbt[full]
```

See [License](https://github.com/polakowo/vectorbt#license) notes on optional dependencies.

Troubleshooting:

* [TA-Lib support](https://github.com/mrjbq7/ta-lib#dependencies)
* [Jupyter Notebook and JupyterLab support](https://plotly.com/python/getting-started/#jupyter-notebook-support)

### Docker

You can pull the most recent Docker image if you [have Docker installed](https://docs.docker.com/install/).

```bash
docker run --rm -p 8888:8888 -v "$PWD":/home/jovyan/work polakowo/vectorbt
```

This command pulls the latest `polakowo/vectorbt` image from Docker Hub. It then starts a container running 
a Jupyter Notebook server and exposes the server on host port 8888. Visiting `http://127.0.0.1:8888/?token=<token>` 
in a browser loads JupyterLab, where token is the secret token printed in the console. Docker destroys 
the container after notebook server exit, but any files written to the working directory in the container 
remain intact in the working directory on the host. See [Jupyter Docker Stacks - Quick Start](https://github.com/jupyter/docker-stacks#quick-start).

There are two types of images: 

* [polakowo/vectorbt](https://hub.docker.com/r/polakowo/vectorbt): vanilla version (default)
* [polakowo/vectorbt-full](https://hub.docker.com/r/polakowo/vectorbt-full): full version (with optional dependencies)

Each Docker image is based on [jupyter/scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook) 
and comes with Jupyter environment, vectorbt, and other scientific packages installed.

## Examples

You can start backtesting with just a couple of lines.

Here is how much profit we would have made if we invested $100 into Bitcoin in 2014 and held 
(Note: first time compiling with Numba may take a while):

```python
import vectorbt as vbt

price = vbt.YFData.download('BTC-USD').get('Close')

portfolio = vbt.Portfolio.from_holding(price, init_cash=100)
portfolio.total_profit()
```

```plaintext
8412.436065824717
```

The crossover of 10-day SMA and 50-day SMA under the same conditions:

```python
fast_ma = vbt.MA.run(price, 10)
slow_ma = vbt.MA.run(price, 50)
entries = fast_ma.ma_above(slow_ma, crossover=True)
exits = fast_ma.ma_below(slow_ma, crossover=True)

portfolio = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)
portfolio.total_profit()
```

```plaintext
12642.617149066731
```

Quickly assessing the performance of 1000 random signal strategies on BTC and ETH:

```python
import numpy as np

symbols = ["BTC-USD", "ETH-USD"]
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

n = np.random.randint(10, 101, size=1000).tolist()
portfolio = vbt.Portfolio.from_random_signals(price, n=n, init_cash=100, seed=42)

mean_expectancy = portfolio.trades.expectancy().groupby(['rand_n', 'symbol']).mean()
fig = mean_expectancy.unstack().vbt.scatterplot(xaxis_title='rand_n', yaxis_title='mean_expectancy')
fig.show()
```

![rand_scatter.svg](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/rand_scatter.svg)

For fans of hyperparameter optimization, here is a snippet for testing 10000 window combinations of a 
dual SMA crossover strategy on BTC, USD and LTC:

```python
symbols = ["BTC-USD", "ETH-USD", "LTC-USD"]
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])
entries = fast_ma.ma_above(slow_ma, crossover=True)
exits = fast_ma.ma_below(slow_ma, crossover=True)

portfolio_kwargs = dict(size=np.inf, fees=0.001, freq='1D')
portfolio = vbt.Portfolio.from_signals(price, entries, exits, **portfolio_kwargs)

fig = portfolio.total_return().vbt.heatmap(
    x_level='fast_window', y_level='slow_window', slider_level='symbol', symmetric=True,
    trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%')))
fig.show()
```

![dmac_heatmap.gif](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/dmac_heatmap.gif)

Digging into each strategy configuration is as simple as indexing with pandas:

```python
portfolio[(10, 20, 'ETH-USD')].stats()
```

```plaintext
Start                      2015-08-07 00:00:00+00:00
End                        2021-05-02 00:00:00+00:00
Duration                          2092 days 00:00:00
Init. Cash                                       100
Total Profit                                  846151
Total Return [%]                              846151
Benchmark Return [%]                          106176
Position Coverage [%]                        56.1185
Max. Drawdown [%]                             70.735
Avg. Drawdown [%]                            12.2078
Max. Drawdown Duration             760 days 00:00:00
Avg. Drawdown Duration    29 days 14:19:42.089552239
Num. Trades                                       50
Win Rate [%]                                      54
Best Trade [%]                                1075.8
Worst Trade [%]                             -29.5934
Avg. Trade [%]                               47.8175
Max. Trade Duration                 80 days 00:00:00
Avg. Trade Duration                 22 days 20:38:24
Expectancy                                   12322.9
SQN                                          1.43759
Gross Exposure                              0.561185
Sharpe Ratio                                 2.17109
Sortino Ratio                                3.81812
Calmar Ratio                                 5.43505
Name: (10, 20, ETH-USD), dtype: object
```

```python
portfolio[(10, 20, 'ETH-USD')].plot().show()
```

![dmac_portfolio.svg](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/dmac_portfolio.svg)

It's not all about backtesting - vectorbt can be used to facilitate financial data analysis and visualization.
Let's generate a GIF that animates the %B and bandwidth of Bollinger Bands for different symbols:

```python
symbols = ["BTC-USD", "ETH-USD", "ADA-USD"]
price = vbt.YFData.download(symbols, period='6mo', missing_index='drop').get('Close')
bbands = vbt.BBANDS.run(price)

def plot(index, bbands):
    bbands = bbands.loc[index]
    fig = vbt.make_subplots(
        rows=5, cols=1, shared_xaxes=True, 
        row_heights=[*[0.5 / 3] * len(symbols), 0.25, 0.25], vertical_spacing=0.05,
        subplot_titles=(*symbols, '%B', 'Bandwidth'))
    fig.update_layout(template='vbt_dark', showlegend=False, width=750, height=650)
    for i, symbol in enumerate(symbols):
        bbands.close[symbol].vbt.lineplot(add_trace_kwargs=dict(row=i + 1, col=1), fig=fig)
    bbands.percent_b.vbt.ts_heatmap(
        trace_kwargs=dict(zmin=0, zmid=0.5, zmax=1, colorscale='Spectral', colorbar=dict(
            y=(fig.layout.yaxis4.domain[0] + fig.layout.yaxis4.domain[1]) / 2, len=0.2
        )), add_trace_kwargs=dict(row=4, col=1), fig=fig)
    bbands.bandwidth.vbt.ts_heatmap(
        trace_kwargs=dict(colorbar=dict(
            y=(fig.layout.yaxis5.domain[0] + fig.layout.yaxis5.domain[1]) / 2, len=0.2
        )), add_trace_kwargs=dict(row=5, col=1), fig=fig)
    return fig

vbt.save_animation('bbands.gif', bbands.wrapper.index, plot, bbands, delta=90, step=3, fps=3)
```

```plaintext
100%|██████████| 31/31 [00:21<00:00,  1.21it/s]
```

![bbands.gif](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/bbands.gif)

## Motivation

While there are [many great backtesting packages for Python](https://github.com/mementum/backtrader#alternatives), 
vectorbt combines an extremely fast backtester and a data science tool: it excels at processing performance and offers 
interactive tools to explore complex phenomena in trading. With it, you can traverse a huge number of strategy configurations, 
time periods, and instruments in little time, to explore where your strategy performs best and to uncover hidden patterns in data.

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
(price, signals, etc.), vectorbt is optimized for working with multi-dimensional data: it treats index 
of a DataFrame as time axis and columns as distinct features that should be backtest, and performs 
computations on the entire matrix at once, without slow Python loops.

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
    
- Functions for combining, transforming, and indexing NumPy and pandas objects
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

- Splitting functions for time series cross-validation
    - Supports [scikit-learn](https://github.com/scikit-learn/scikit-learn) splitters

```python-repl
>>> pd.Series([1, 2, 3, 4, 5]).vbt.expanding_split()[0]
split_idx    0    1    2    3  4
0          1.0  1.0  1.0  1.0  1
1          NaN  2.0  2.0  2.0  2
2          NaN  NaN  3.0  3.0  3
3          NaN  NaN  NaN  4.0  4
4          NaN  NaN  NaN  NaN  5
```

- Drawdown analysis

```python-repl
>>> pd.Series([2, 1, 3, 2]).vbt.drawdowns.plot().show()
```

![drawdowns.svg](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/drawdowns.svg)

- Functions for working with signals
    - Entry, exit, and random signal generators
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
    - Includes basic generators such for random signal generation

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
    - Numba-compiled versions of metrics found in [empyrical](https://github.com/quantopian/empyrical)
    - Rolling versions of most metrics

```python-repl
>>> pd.Series([0.01, -0.01, 0.01]).vbt.returns(freq='1D').sharpe_ratio()
5.515130702591433
```
    
- Class for modeling portfolios
    - Accepts signals, orders, and custom order function
    - Supports long and short positions
    - Supports individual and multi-asset mixed portfolios
    - Offers metrics and tools for analyzing returns, orders, trades and positions
    - Allows saving and loading from disk using [dill](https://github.com/uqfoundation/dill)
    
```python-repl
>>> price = [1., 2., 3., 2., 1.]
>>> entries = [True, False, True, False, False]
>>> exits = [False, True, False, True, False]
>>> portfolio = vbt.Portfolio.from_signals(price, entries, exits, freq='1D')
>>> portfolio.trades.plot().show()
```

![trades.svg](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/trades.svg)

- Indicator factory for building complex technical indicators with ease
    - Includes technical indicators with full Numba support
        - Moving average, Bollinger Bands, RSI, Stochastic, MACD, and more
    - Each indicator has methods for generating signals and plotting
    - Each indicator takes arbitrary parameter combinations, from arrays to Cartesian products
    - Supports [ta](https://github.com/bukosabino/ta), [pandas-ta](https://github.com/twopirllc/pandas-ta), and [TA-Lib](https://github.com/mrjbq7/ta-lib) indicators out of the box
    - Supports parallelization with [Ray](https://github.com/ray-project/ray)

```python-repl
>>> price = pd.Series([1, 2, 3, 4, 5], dtype=float)
>>> vbt.MA.run(price, [2, 3]).ma  # vectorbt
ma_window    2    3
0          NaN  NaN
1          1.5  NaN
2          2.5  2.0
3          3.5  3.0
4          4.5  4.0

>>> vbt.ta('SMAIndicator').run(price, [2, 3]).sma_indicator  # ta
smaindicator_window    2    3
0                    NaN  NaN
1                    1.5  NaN
2                    2.5  2.0
3                    3.5  3.0
4                    4.5  4.0

>>> vbt.pandas_ta('SMA').run(price, [2, 3]).sma  # pandas-ta
sma_length    2    3
0           NaN  NaN
1           1.5  NaN
2           2.5  2.0
3           3.5  3.0
4           4.5  4.0

>>> vbt.talib('SMA').run(price, [2, 3]).real  # TA-Lib
sma_timeperiod    2    3
0               NaN  NaN
1               1.5  NaN
2               2.5  2.0
3               3.5  3.0
4               4.5  4.0
``` 

- Label generation for machine learning
    - Labeling based on local extrema, breakouts, and more

```python-repl
>>> price = np.cumprod(np.random.uniform(-0.1, 0.1, size=100) + 1)
>>> vbt.LEXLB.run(price, 0.2, 0.2).plot().show()
``` 

![local_extrema.svg](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/local_extrema.svg)

- Classes for downloading and (periodically) updating data
    - Includes APIs such as [ccxt](https://github.com/ccxt/ccxt), [yfinance](https://github.com/ranaroussi/yfinance) and [python-binance](https://github.com/sammchardy/python-binance)
- Telegram bot based on [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Interactive Plotly-based widgets for visual data analysis

## Resources

Head over to the [documentation](https://polakowo.io/vectorbt/docs/index.html) to get started.

### Notebooks

- [Performance analysis of Moving Average Crossover](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/BitcoinDMAC.ipynb)
- [Performance analysis of stop signals](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/StopSignals.ipynb)
- [Backtesting per trading session](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/TradingSessions.ipynb)
- [Portfolio optimization](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/PortfolioOptimization.ipynb)
- [Plotting MACD parameters as 3D volume](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/MACDVolume.ipynb)
- [Walk-forward optimization](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/WalkForwardOptimization.ipynb)
- [Running Telegram signal bot](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/TelegramSignals.ipynb)
- [Porting RSI strategy from backtrader](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/PortingBTStrategy.ipynb)
- [Pairs trading (vs backtrader)](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/PairsTrading.ipynb)

Note: you must run the notebook to play with the widgets.

### Dashboards

- [Detecting and backtesting common candlestick patterns](https://github.com/polakowo/vectorbt/tree/master/apps/candlestick-patterns)

### Articles

- [Stop Loss, Trailing Stop, or Take Profit? 2 Million Backtests Shed Light](https://polakowo.medium.com/stop-loss-trailing-stop-or-take-profit-2-million-backtests-shed-light-dde23bda40be)

## How to contribute

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

First, you need to install vectorbt from the repository:

```bash
pip uninstall vectorbt
git clone https://github.com/polakowo/vectorbt.git
cd vectorbt
pip install -e .
```

After making changes, make sure you did not break any functionality:

```bash
pytest
```

Please make sure to update tests as appropriate.

## License

This work is licensed under Apache 2.0, but installing optional dependencies may be subject to a stronger license.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. 
USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.

