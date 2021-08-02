[![Python Versions](https://img.shields.io/pypi/pyversions/vectorbt.svg?logo=python&logoColor=white)](https://pypi.org/project/vectorbt)
[![License](https://img.shields.io/pypi/l/vectorbt?color=yellow)](https://github.com/polakowo/vectorbt/blob/master/LICENSE)
[![PyPi](https://img.shields.io/pypi/v/vectorbt?color=blueviolet)](https://pypi.org/project/vectorbt)
[![Build Status](https://travis-ci.com/polakowo/vectorbt.svg?branch=master)](https://travis-ci.com/polakowo/vectorbt)
[![codecov](https://codecov.io/gh/polakowo/vectorbt/branch/master/graph/badge.svg?token=YTLNAI7PS3)](https://codecov.io/gh/polakowo/vectorbt)
[![Website](https://img.shields.io/website?url=https://vectorbt.dev/)](https://vectorbt.dev/)
[![Downloads](https://pepy.tech/badge/vectorbt)](https://pepy.tech/project/vectorbt)
[![Binder](https://img.shields.io/badge/launch-binder-d6604a)](https://mybinder.org/v2/gh/polakowo/vectorbt/HEAD?urlpath=lab)
[![Join the chat at https://gitter.im/vectorbt/community](https://badges.gitter.im/vectorbt.svg)](https://gitter.im/vectorbt/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Patreon](https://img.shields.io/badge/support-sponsor-ff69b4?logo=patreon)](https://www.patreon.com/vectorbt)

# vectorbt :boom:

vectorbt is a backtesting library on steroids — it operates entirely on pandas and NumPy objects, and is 
accelerated by [Numba](https://github.com/numba/numba) to analyze time series at speed and scale.

In contrast to other backtesters, vectorbt represents data as nd-arrays.
This enables superfast computation using vectorized operations with NumPy and non-vectorized but dynamically 
compiled operations with Numba. It also integrates [plotly.py](https://github.com/plotly/plotly.py) and 
[ipywidgets](https://github.com/jupyter-widgets/ipywidgets) to display complex charts and dashboards akin 
to Tableau right in the Jupyter notebook. Due to high performance, vectorbt can process large amounts of 
data even without GPU and parallelization and enables the user to interact with data-hungry widgets 
without significant delays.

With vectorbt, you can
* Build your pipelines in a few lines of code
* Retain full control over execution (as opposed to web-based services such as TradingView)
* Optimize your trading strategy against many parameters, assets, and periods in one go
* Uncover hidden patterns in financial markets
* Analyze time series and engineer new features for ML models
* Supercharge pandas and your favorite tools to run much faster
* Visualize strategy performance using interactive charts and dashboards (both in Jupyter and browser)
* Fetch and process data periodically, send Telegram notifications, and more

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

Start backtesting with just a couple of lines:

Here is how much profit we would have made if we invested $100 into Bitcoin in 2014 and held 
(Note: first time compiling with Numba may take a while):

```python
import vectorbt as vbt

price = vbt.YFData.download('BTC-USD').get('Close')

pf = vbt.Portfolio.from_holding(price, init_cash=100)
pf.total_profit()
```

```plaintext
8961.008555963961
```

The crossover of 10-day SMA and 50-day SMA:

```python
fast_ma = vbt.MA.run(price, 10)
slow_ma = vbt.MA.run(price, 50)
entries = fast_ma.ma_above(slow_ma, crossover=True)
exits = fast_ma.ma_below(slow_ma, crossover=True)

pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)
pf.total_profit()
```

```plaintext
16423.251963801864
```

Generate 1,000 random strategies and test them on BTC and ETH:

```python
import numpy as np

symbols = ["BTC-USD", "ETH-USD"]
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

n = np.random.randint(10, 101, size=1000).tolist()
pf = vbt.Portfolio.from_random_signals(price, n=n, init_cash=100, seed=42)

mean_expectancy = pf.trades.expectancy().groupby(['randnx_n', 'symbol']).mean()
fig = mean_expectancy.unstack().vbt.scatterplot(xaxis_title='randnx_n', yaxis_title='mean_expectancy')
fig.show()
```

![rand_scatter.svg](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/rand_scatter.svg)

For fans of hyperparameter optimization: here is a snippet for testing 10,000 window combinations of a 
dual SMA crossover strategy on BTC, USD, and LTC:

```python
symbols = ["BTC-USD", "ETH-USD", "LTC-USD"]
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])
entries = fast_ma.ma_above(slow_ma, crossover=True)
exits = fast_ma.ma_below(slow_ma, crossover=True)

pf_kwargs = dict(size=np.inf, fees=0.001, freq='1D')
pf = vbt.Portfolio.from_signals(price, entries, exits, **pf_kwargs)

fig = pf.total_return().vbt.heatmap(
    x_level='fast_window', y_level='slow_window', slider_level='symbol', symmetric=True,
    trace_kwargs=dict(colorbar=dict(title='Total return', tickformat='%')))
fig.show()
```

<img width="650" src="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/dmac_heatmap.gif">

Digging into each strategy configuration is as simple as indexing with pandas:

```python
pf[(10, 20, 'ETH-USD')].stats()
```

```plaintext
Start                          2015-08-07 00:00:00+00:00
End                            2021-08-01 00:00:00+00:00
Period                                2183 days 00:00:00
Start Value                                        100.0
End Value                                  620402.791485
Total Return [%]                           620302.791485
Benchmark Return [%]                        92987.961948
Max Gross Exposure [%]                             100.0
Total Fees Paid                             10991.676981
Max Drawdown [%]                               70.734951
Max Drawdown Duration                  760 days 00:00:00
Total Trades                                          54
Total Closed Trades                                   53
Total Open Trades                                      1
Open Trade P&L                              67287.940601
Win Rate [%]                                   52.830189
Best Trade [%]                               1075.803607
Worst Trade [%]                               -29.593414
Avg Winning Trade [%]                          95.695343
Avg Losing Trade [%]                          -11.890246
Avg Winning Trade Duration    35 days 23:08:34.285714286
Avg Losing Trade Duration                8 days 00:00:00
Profit Factor                                   2.651143
Expectancy                                   10434.24247
Sharpe Ratio                                    2.041211
Calmar Ratio                                      4.6747
Omega Ratio                                     1.547013
Sortino Ratio                                   3.519894
Name: (10, 20, ETH-USD), dtype: object
```

The same for plotting:

```python
pf[(10, 20, 'ETH-USD')].plot().show()
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
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.15,
        subplot_titles=('%B', 'Bandwidth'))
    fig.update_layout(template='vbt_dark', showlegend=False, width=750, height=400)
    bbands.percent_b.vbt.ts_heatmap(
        trace_kwargs=dict(zmin=0, zmid=0.5, zmax=1, colorscale='Spectral', colorbar=dict(
            y=(fig.layout.yaxis.domain[0] + fig.layout.yaxis.domain[1]) / 2, len=0.5
        )), add_trace_kwargs=dict(row=1, col=1), fig=fig)
    bbands.bandwidth.vbt.ts_heatmap(
        trace_kwargs=dict(colorbar=dict(
            y=(fig.layout.yaxis2.domain[0] + fig.layout.yaxis2.domain[1]) / 2, len=0.5
        )), add_trace_kwargs=dict(row=2, col=1), fig=fig)
    return fig

vbt.save_animation('bbands.gif', bbands.wrapper.index, plot, bbands, delta=90, step=3, fps=3)
```

```plaintext
100%|██████████| 31/31 [00:21<00:00,  1.21it/s]
```

<img width="750" src="https://raw.githubusercontent.com/polakowo/vectorbt/master/static/bbands.gif">

## How it works?

vectorbt combines pandas, NumPy, and Numba sauce to obtain orders-of-magnitude speedup over other libraries. 
It natively works on pandas objects while performing all computations using NumPy and Numba under the hood. 
This way, it is often much faster than pandas alone:

```python-repl
>>> big_ts = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

# pandas
>>> %timeit big_ts.expanding().max()
48.4 ms ± 557 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# vectorbt
>>> %timeit big_ts.vbt.expanding_max()
8.82 ms ± 121 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

In contrast to other backtesters, vectorbt is optimized for working with multi-dimensional data: 
it treats the index of a Series/DataFrame as a time axis and columns as distinct configurations that 
should be backtested, and performs computations on the entire array at once, without slow Python loops.

To make the library easier to use, vectorbt introduces a namespace (accessor) to pandas objects 
(see [extending pandas](https://pandas.pydata.org/pandas-docs/stable/development/extending.html)). 
This way, users can easily switch between pandas and vectorbt functionality. Moreover, each vectorbt 
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
    - Smart broadcasting for pandas
    
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

- Transformation functions for rescaling and normalization
- Drawdown analysis

```python-repl
>>> pd.Series([2, 1, 3, 2]).vbt.drawdowns.plot().show()
```

![drawdowns.svg](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/drawdowns.svg)

- Functions for working with signals
    - Entry, exit, and random signal generators
    - Stop signal, ranking, and map-reduce functions
    
```python-repl
>>> pd.Series([False, True, True, True]).vbt.signals.first()
0    False
1     True
2    False
3    False
dtype: bool
```

- Signal factory for building signal generators
    - Includes basic generators such for random signal generation

```python-repl
>>> randnx = vbt.RANDNX.run(n=[0, 1, 2], input_shape=(6,), seed=42)
>>> randnx.entries
randnx_n      0      1      2
0         False   True   True
1         False  False  False
2         False  False  False
3         False  False   True
4         False  False  False
5         False  False  False
>>> randnx.exits
randnx_n      0      1      2
0         False  False  False
1         False  False   True
2         False  False  False
3         False   True  False
4         False  False   True
5         False  False  False
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
>>> pf = vbt.Portfolio.from_signals(price, entries, exits, freq='1D')
>>> pf.trades.plot().show()
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

- Tailored statistics for many backtesting components
  
```python-repl
>>> pd.Series([1, 2, 3, 2, 3, 2, 1, 2]).vbt(freq='d').drawdowns.stats()
Start                                        0
End                                          7
Period                         8 days 00:00:00
Total Records                                2
Total Recovered Drawdowns                    1
Total Active Drawdowns                       1
Active Drawdown [%]                  33.333333
Active Duration                3 days 00:00:00
Active Recovery [%]                       50.0
Active Recovery Return [%]               100.0
Active Recovery Duration       1 days 00:00:00
Max Drawdown [%]                     33.333333
Avg Drawdown [%]                     33.333333
Max Drawdown Duration          2 days 00:00:00
Avg Drawdown Duration          2 days 00:00:00
Max Recovery Return [%]                   50.0
Avg Recovery Return [%]                   50.0
Max Recovery Duration          1 days 00:00:00
Avg Recovery Duration          1 days 00:00:00
Avg Recovery Duration Ratio                0.5
dtype: object
```

- Label generation for ML models

```python-repl
>>> price = np.cumprod(np.random.uniform(-0.1, 0.1, size=100) + 1)
>>> vbt.LEXLB.run(price, 0.2, 0.2).plot().show()
``` 

![local_extrema.svg](https://raw.githubusercontent.com/polakowo/vectorbt/master/static/local_extrema.svg)

- Classes for downloading and (periodically) updating data
    - Includes APIs such as [ccxt](https://github.com/ccxt/ccxt), [yfinance](https://github.com/ranaroussi/yfinance) and [python-binance](https://github.com/sammchardy/python-binance)
    - Allows creation of new data classes with ease
- Telegram bot based on [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- Interactive Plotly-based widgets for visual data analysis

## Resources

### Documentation

Head over to the [documentation](https://vectorbt.dev/docs/index.html) to get started.

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

## Getting Help

- If you need supervision or any help with your implementation, [join a private chat](https://www.patreon.com/vectorbt)
- For questions on Numba and other parts, the best place to go to is [StackOverflow](https://stackoverflow.com/)
- If you have general questions, start a new [GitHub Discussion](https://github.com/polakowo/vectorbt/discussions)
  - Alternatively, you can ask on [Gitter](https://gitter.im/vectorbt/community)
- If you found what appears to be a bug, please [create a new issue](https://github.com/polakowo/vectorbt/issues)
- For other inquiries, please [contact the author](mailto:olegpolakow@gmail.com)

## Contributing

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
