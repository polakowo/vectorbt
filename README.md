# vectorbt: Powerful toolkit for quants.

[![Python Versions](https://img.shields.io/pypi/pyversions/vectorbt.svg?logo=python&logoColor=white)](https://pypi.org/project/vectorbt)
[![License](https://img.shields.io/badge/license-Fair%20Code-yellow)](https://github.com/polakowo/vectorbt/blob/master/LICENSE.md)
[![PyPi](https://img.shields.io/pypi/v/vectorbt?color=blueviolet)](https://pypi.org/project/vectorbt)
[![Build Status](https://app.travis-ci.com/polakowo/vectorbt.svg?branch=master)](https://app.travis-ci.com/github/polakowo/vectorbt)
[![codecov](https://codecov.io/gh/polakowo/vectorbt/branch/master/graph/badge.svg?token=YTLNAI7PS3)](https://codecov.io/gh/polakowo/vectorbt)
[![Website](https://img.shields.io/website?url=https://vectorbt.dev/)](https://vectorbt.dev/)
[![Downloads](https://pepy.tech/badge/vectorbt)](https://pepy.tech/project/vectorbt)
[![Binder](https://img.shields.io/badge/launch-binder-d6604a)](https://mybinder.org/v2/gh/polakowo/vectorbt/HEAD?urlpath=lab)
[![Join the chat at https://gitter.im/vectorbt/community](https://badges.gitter.im/vectorbt.svg)](https://gitter.im/vectorbt/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Patreon](https://img.shields.io/badge/support-sponsor-ff69b4?logo=patreon)](https://www.patreon.com/vectorbt)

## [Installation](#installation) · [Features](#zap-features) · [Usage](#sparkles-usage) · [Resources](#resources) · [License](#license)

vectorbt takes a novel approach to backtesting: it operates entirely on pandas and NumPy objects, and is accelerated by 
[Numba](https://github.com/numba/numba) to analyze any data at speed and scale. This allows for testing of many thousands 
of strategies in **seconds**.

In contrast to other backtesters, vectorbt represents complex data as (structured) NumPy arrays.
This enables superfast computation using vectorized operations with NumPy and non-vectorized but dynamically 
compiled operations with Numba. It also integrates [Plotly](https://github.com/plotly/plotly.py) and 
[Jupyter Widgets](https://github.com/jupyter-widgets/ipywidgets) to display complex charts and dashboards akin 
to Tableau right in the Jupyter notebook. Due to high performance, vectorbt can process large amounts of 
data even without GPU and parallelization and enables the user to interact with data-hungry widgets 
without significant delays.

With vectorbt, you can
* Backtest strategies in **a couple of lines** of Python code
* Enjoy the best of both worlds: the ecosystem of Python and **the speed of C**
* Retain **full control** over execution (as opposed to web-based services such as TradingView)
* Optimize your trading strategy against many parameters, assets, and periods **in one go**
* Uncover **hidden patterns** in financial markets
* Analyze time series and engineer new features for **ML models**
* Supercharge pandas and your favorite tools to run much faster
* Visualize strategy performance using **interactive charts and dashboards** (both in Jupyter and browser)
* Fetch and process data periodically, send Telegram notifications, and more

## Installation

```bash
pip install -U vectorbt
```

To also install optional dependencies:

```bash
pip install -U "vectorbt[full]"
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

* **[polakowo/vectorbt](https://hub.docker.com/r/polakowo/vectorbt)**: vanilla version (default)
* **[polakowo/vectorbt-full](https://hub.docker.com/r/polakowo/vectorbt-full)**: full version (with optional dependencies)

Each Docker image is based on [jupyter/scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook) 
and comes with Jupyter environment, vectorbt, and other scientific packages installed.

## :zap: Features

### Pandas

- [x] **Pandas acceleration**: Compiled versions of most popular pandas functions, such as mapping, reducing, rolling, grouping, and resamping. For best performance, most operations are done strictly using NumPy and Numba. Attaches a custom accessor on top of pandas to easily switch between pandas and vectorbt functionality. **[>>](https://vectorbt.dev/docs/generic/accessors.html)**
- [x] **Flexible broadcasting**: Mechanism for broadcasting array-like objects of arbitrary shapes, including pandas objects with MultiIndex.
- [x] **Pandas utilities**: Grouping columns, wrapping NumPy arrays, transforming pandas objects and their indexes, and more. **[>>](https://vectorbt.dev/docs/base/index.html)**

### Data

- [x] **Data acquisition**: Supports various data providers, such as **[Yahoo Finance](https://github.com/ranaroussi/yfinance)**, **[Binance](https://github.com/sammchardy/python-binance)**, and **[CCXT](https://github.com/ccxt/ccxt)**. Can merge multiple symbols with different index, as well as update them. **[>>](https://vectorbt.dev/docs/data/custom.html)**
- [x] **Data generation**: Supports various (random) data generators, such as **[GBM](https://en.wikipedia.org/wiki/Geometric_Brownian_motion)**. **[>>](https://vectorbt.dev/docs/data/custom.html)**
- [x] **Scheduled data updates**: Can periodically update any previously downloaded data. **[>>](https://vectorbt.dev/docs/data/updater.html)**
- [x] **Data preparation**: Transformation, rescaling, and normalization of data. Custom splitters for cross-validation. Supports **[Scikit-Learn](https://github.com/scikit-learn/scikit-learn)** splitters, such as for K-Folds cross-validation. **[>>](https://vectorbt.dev/docs/generic/accessors.html)**
- [x] **Labeling for ML**: Discrete and continuous label generation for effective training of ML models. **[>>](https://vectorbt.dev/docs/labels/generators.html)**

### Indicators

- [x] **Technical indicators**: Most popular technical indicators with full Numba support, including Moving Average, Bollinger Bands, RSI, Stochastic, MACD, and more. Out-of-the-box support for 99% indicators in **[Technical Analysis Library](https://github.com/bukosabino/ta)**, **[Pandas TA](https://github.com/twopirllc/pandas-ta)**, and **[TA-Lib](https://github.com/mrjbq7/ta-lib)** thanks to built-in parsers. Each indicator is wrapped with the vectorbt's indicator engine and thus accepts arbitrary hyperparameter combinations - from arrays to Cartesian products. **[>>](https://vectorbt.dev/docs/indicators/basic.html)**
- [x] **Indicator factory**: Sophisticated factory for building custom technical indicators of any complexity. Takes a function and does all the magic for you: generates an indicator skeleton that takes inputs and parameters of any shape and type, and runs the vectorbt's indicator engine. The easiest and most flexible way to create indicators you will find in open source. **[>>](https://vectorbt.dev/docs/indicators/factory.html)**

### Signals

- [x] **Signal analysis**: Generation, mapping and reducing, ranking, and distribution analysis of entry and exit signals. **[>>](https://vectorbt.dev/docs/signals/accessors.html)**
- [x] **Signal generators**: Random and stop loss (SL, TSL, TP, etc.) signal generators with full Numba support. **[>>](https://vectorbt.dev/docs/signals/generators.html)**
- [x] **Signal factory**: Signal factory based on indicator factory specialized for iterative signal generation. **[>>](https://vectorbt.dev/docs/signals/factory.html)**

### Modeling

- [x] **Portfolio modeling**: The fastest backtesting engine in open source: fills 1,000,000 orders in 70-100ms on Apple M1. Flexible and powerful simulation functions for portfolio modeling, highly optimized for best performance and memory consumption. Supports two major simulation modes: 1) vectorized backtesting using user-provided arrays, such as orders, signals, and records, and 2) event-driven backtesting using user-defined callbacks. Supports shorting and individual as well as multi-asset mixed portfolios. Combines many features across vectorbt into a single behemoth class. **[>>](https://vectorbt.dev/docs/portfolio/base.html)**

### Performance

- [x] **Performance metrics**: Numba-compiled versions of metrics from **[empyrical](https://github.com/quantopian/empyrical)** and their highly-optimized rolling versions. Adapter for **[QuantStats](https://github.com/ranaroussi/quantstats)**. **[>>](https://vectorbt.dev/docs/returns/accessors.html)**
- [x] **Stats builder**: Class for building statistics out of custom metrics. Implements a preset of tailored statistics for many backtesting components, such as signals, returns, and portfolio. **[>>](https://vectorbt.dev/docs/generic/stats_builder.html)**
- [x] **Records and mapped arrays**: In-house data structures for analyzing complex data, such as simulation logs. Fully compiled with Numba. **[>>](https://vectorbt.dev/docs/records/index.html)**
- [x] **Trade analysis**: Retrospective analysis of trades from various view points. Supports entry trades, exit trades, and positions. **[>>](https://vectorbt.dev/docs/portfolio/trades.html)**
- [x] **Drawdown analysis**: Drawdown statistics of any numeric time series. **[>>](https://vectorbt.dev/docs/generic/drawdowns.html)**

### Plotting

- [x] **Data visualization**: Numerous flexible data plotting functions distributed across vectorbt.
- [x] **Figures and widgets**: Custom interactive figures and widgets using **[Plotly](https://github.com/plotly/plotly.py)**, such as Heatmap and Volume. All custom widgets have dedicated methods for efficiently updating their state. **[>>](https://vectorbt.dev/docs/generic/plotting.html)**
- [x] **Plots builder**: Class for building plots out of custom subplots. Implements a preset of tailored subplots for many backtesting components, such as signals, returns, and portfolio. **[>>](https://vectorbt.dev/docs/generic/plots_builder.html)**

### Extra

- [x] **Notifications**: Telegram bot based on **[Python Telegram Bot](https://github.com/python-telegram-bot/python-telegram-bot)**. **[>>](https://vectorbt.dev/docs/messaging/telegram.html)**
- [x] **General utilities**: Scheduling using **[schedule](https://github.com/dbader/schedule)**, templates, decorators, configs, and more. **[>>](https://vectorbt.dev/docs/utils/index.html)**
- [x] **Caching**: Property and method decorators for caching most frequently used objects.
- [x] **Persistance**: Most Python objects including data and portfolio can be saved to a file and retrieved back using **[Dill](https://github.com/uqfoundation/dill)**.

## :sparkles: Usage

vectorbt allows you to easily backtest strategies with a couple of lines of Python code.

* Here is how much profit we would have made if we invested $100 into Bitcoin in 2014:

```python
import vectorbt as vbt

price = vbt.YFData.download('BTC-USD').get('Close')

pf = vbt.Portfolio.from_holding(price, init_cash=100)
pf.total_profit()
```

```plaintext
8961.008555963961
```

* Buy whenever 10-day SMA crosses above 50-day SMA and sell when opposite:

```python
fast_ma = vbt.MA.run(price, 10)
slow_ma = vbt.MA.run(price, 50)
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

pf = vbt.Portfolio.from_signals(price, entries, exits, init_cash=100)
pf.total_profit()
```

```plaintext
16423.251963801864
```

* Generate 1,000 strategies with random signals and test them on BTC and ETH:

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

* For fans of hyperparameter optimization: here is a snippet for testing 10,000 window combinations of a 
dual SMA crossover strategy on BTC, USD, and LTC:

```python
symbols = ["BTC-USD", "ETH-USD", "LTC-USD"]
price = vbt.YFData.download(symbols, missing_index='drop').get('Close')

windows = np.arange(2, 101)
fast_ma, slow_ma = vbt.MA.run_combs(price, window=windows, r=2, short_names=['fast', 'slow'])
entries = fast_ma.ma_crossed_above(slow_ma)
exits = fast_ma.ma_crossed_below(slow_ma)

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
Open Trade PnL                              67287.940601
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

* Let's generate a GIF that animates the %B and bandwidth of Bollinger Bands for different symbols:

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

And this is just the tip of the iceberg of what's possible. Check out [Resources](#resources) to learn more.

## Resources

### Documentation

Head over to the [documentation](https://vectorbt.dev/) to get started.

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

### Getting Help

- If you need supervision or any help with your implementation, [join a private chat](https://www.patreon.com/vectorbt)
- For questions on Numba and other parts, the best place to go to is [StackOverflow](https://stackoverflow.com/)
- If you have general questions, start a new [GitHub Discussion](https://github.com/polakowo/vectorbt/discussions)
  - Alternatively, you can ask on [Gitter](https://gitter.im/vectorbt/community)
- If you found what appears to be a bug, please [create a new issue](https://github.com/polakowo/vectorbt/issues)
- For other inquiries, please [contact the author](mailto:olegpolakow@gmail.com)

## License

This work is [fair-code](http://faircode.io/) distributed under [Apache 2.0 with Commons Clause](https://github.com/polakowo/vectorbt/blob/master/LICENSE.md) license. 
The source code is open and everyone (individuals and organizations) can use it for free. 
However, it is not allowed to sell products and services that are mostly just this software.

If you have any questions about this or want to apply for a license exception, please [contact the author](mailto:olegpolakow@gmail.com).

Installing optional dependencies may be subject to a more restrictive license.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose. 
USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
