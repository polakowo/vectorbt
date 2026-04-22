---
title: Getting started
---

# Getting started

<div align="center">
	<a href="https://vectorbt.pro/" alt="https://vectorbt.pro/">
        <img src="/assets/logo/header-pro.svg" />
    </a>
</div>
<div align="center">
	<a href="https://vectorbt.dev/" alt="https://vectorbt.dev/">
        <img src="/assets/logo/header.svg" />
    </a>
</div>

## What is VectorBT?

VectorBT is a Python package for quantitative analysis that takes a novel approach to backtesting: 
it operates entirely on pandas and NumPy objects, accelerated by [Numba](https://github.com/numba/numba) 
and [Rust](https://www.rust-lang.org/) to analyze any data at speed and scale. This makes it possible 
to test many thousands of strategies in *seconds*.

Unlike other backtesters, VectorBT represents complex data as structured NumPy arrays.
This enables ultra-fast computation through vectorized operations with NumPy, dynamically 
compiled operations with Numba, and optional precompiled Rust kernels with Python bindings via
[PyO3](https://pyo3.rs/) for the most performance-critical paths.

It also integrates [Plotly](https://github.com/plotly/plotly.py) and 
[Jupyter Widgets](https://github.com/jupyter-widgets/ipywidgets) to display rich charts and dashboards 
— akin to Tableau — right in the Jupyter notebook. Thanks to its high performance, VectorBT can process 
large amounts of data even without a GPU or parallelization, enabling users to interact with 
data-hungry widgets without noticeable delays.

With VectorBT, you can

<div class="grid cards" markdown>

- :material-weather-windy:{ .lg .middle } **Simple**

    ---

    Backtest strategies in a couple of lines of Python code

- :material-speedometer:{ .lg .middle } **Blazing fast**

    ---

    Enjoy the best of both worlds: the ecosystem of Python and the speed of C

- :material-eye-outline:{ .lg .middle } **Full control**

    ---

    Retain full control over execution and your data (as opposed to web-based services such as TradingView)

- :material-data-matrix-scan:{ .lg .middle } **Scalable**

    ---

    Optimize your trading strategy against many parameters, assets, and periods in one go

- :material-select-search:{ .lg .middle } **Insightful**

    ---

    Uncover hidden patterns in financial markets

- :material-robot-outline:{ .lg .middle } **ML-ready**

    ---

    Analyze time series and engineer new features for ML models

- :material-meteor:{ .lg .middle } **Supercharged**

    ---

    Supercharge pandas and your favorite tools to run much faster

- :material-chart-bar:{ .lg .middle } **Interactive**

    ---

    Visualize strategy performance using interactive charts and dashboards (both in Jupyter and browser)

- :material-car-battery:{ .lg .middle } **Batteries included**

    ---

    Fetch and process data periodically, send Telegram notifications, and more

- :gem:{ .lg .middle } **VectorBT PRO**

    ---

    [__Support us__](https://vectorbt.pro/) to get access to parallelization, portfolio optimization, 
    pattern recognition, event projections, limit orders, leverage, and 100+ other hot features!

</div>

## Quick start

=== "Core features"

    ```sh
    pip install -U vectorbt
    ```

=== "Core features + Rust"

    ```sh
    pip install -U "vectorbt[rust]"
    ```

=== "All features"

    ```sh
    pip install -U "vectorbt[full]"
    ```

=== "All features + Rust"

    ```sh
    pip install -U "vectorbt[full,rust]"
    ```

```pycon
>>> import vectorbt as vbt

>>> data = vbt.YFData.download("BTC-USD")
>>> price = data.get("Close")
>>> pf = vbt.Portfolio.from_holding(price, init_cash=100)
>>> print(pf.total_profit())

17089.25554191831
```

See [Installation](getting-started/installation.md) for Docker and other options, 
or [Usage](getting-started/usage.md) for more examples.

## Why VectorBT?

While there are many great backtesting packages for Python, VectorBT uniquely combines an extremely 
fast backtester with a data science toolkit: it excels at raw processing performance while offering 
interactive tools to explore complex phenomena in trading. With it, you can sweep a huge number of 
strategy configurations, time periods, and instruments in seconds, discover where your strategy 
performs best, and uncover hidden patterns in data. Having this kind of analytical power at your 
fingertips can give you a real information advantage in your own trading.

## How it works

VectorBT was designed to address common performance shortcomings of backtesting libraries. 
It builds on the idea that each strategy instance can be represented in a vectorized form, 
so multiple instances can be packed into a single multi-dimensional array, processed highly 
efficiently, and compared with ease. This overhauls the traditional OOP approach, where 
strategies are represented as classes and other data structures that are easier to write and extend, 
but harder to analyze at scale and much slower without additional optimization effort.

Thanks to the time-series nature of trading data, most aspects of backtesting can be 
translated into vectors. Instead of processing one element at a time, vectorization lets you 
avoid naive looping and apply the same operation to all elements at once. The path-dependency 
problem inherent to vectorization is solved by using compiled backends:

- **Numba** compiles Python code on the fly, allowing you to write complex logic
in pure Python while still achieving C-like performance.
- **Rust** provides precompiled kernels for the most performance-critical paths, which can be 
enabled with a single argument.

## Example

Imagine a complex strategy with many hyperparameters that need to be tuned. Brute-forcing every 
combination might seem infeasible, but we can still interpolate — and VectorBT makes exactly this 
possible. It doesn't care whether you have one strategy instance or millions. As long as their 
vectors can be concatenated into a matrix and you have enough memory, you can analyze them all in 
one go.

Let's start with fetching the daily price of Bitcoin:

```pycon
>>> import numpy as np
>>> import pandas as pd
>>> from datetime import datetime

>>> import vectorbt as vbt

>>> # Prepare data
>>> start = '2019-01-01 UTC'  # crypto is in UTC
>>> end = '2020-01-01 UTC'
>>> btc_price = vbt.YFData.download('BTC-USD', start=start, end=end).get('Close')

>>> btc_price
Date
2019-01-01 00:00:00+00:00    3843.520020
2019-01-02 00:00:00+00:00    3943.409424
2019-01-03 00:00:00+00:00    3836.741211
...                                  ...
2019-12-30 00:00:00+00:00    7292.995117
2019-12-31 00:00:00+00:00    7193.599121
2020-01-01 00:00:00+00:00    7200.174316
Freq: D, Name: Close, Length: 366, dtype: float64
```

We are going to test a simple Dual Moving Average Crossover (DMAC) strategy using the 
`MA` class to calculate moving averages and generate signals.

Our first test is straightforward: buy when the 10-day moving average crosses above the 20-day moving
average, and sell when the opposite occurs.

```pycon
>>> fast_ma = vbt.MA.run(btc_price, 10, short_name='fast')
>>> slow_ma = vbt.MA.run(btc_price, 20, short_name='slow')

>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> entries
Date
2019-01-01 00:00:00+00:00    False
2019-01-02 00:00:00+00:00    False
2019-01-03 00:00:00+00:00    False
...                            ...
2019-12-30 00:00:00+00:00    False
2019-12-31 00:00:00+00:00    False
2020-01-01 00:00:00+00:00    False
Freq: D, Length: 366, dtype: bool

>>> exits = fast_ma.ma_crossed_below(slow_ma)
>>> exits
Date
2019-01-01 00:00:00+00:00    False
2019-01-02 00:00:00+00:00    False
2019-01-03 00:00:00+00:00    False
...                            ...
2019-12-30 00:00:00+00:00    False
2019-12-31 00:00:00+00:00    False
2020-01-01 00:00:00+00:00    False
Freq: D, Length: 366, dtype: bool

>>> pf = vbt.Portfolio.from_signals(btc_price, entries, exits)
>>> pf.total_return()
0.636680693047752
```

A single DMAC instance produces one column of signals and one performance value.

Adding another strategy instance is as simple as adding another column. Here we pass an array of
window sizes instead of a single value. For each window size, VectorBT computes a moving
average over the entire price series and stores it in its own column.

```pycon
>>> # Multiple strategy instances: (10, 30) and (20, 30)
>>> fast_ma = vbt.MA.run(btc_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(btc_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> entries
fast_window                   10     20
slow_window                   30     30
Date
2019-01-01 00:00:00+00:00  False  False
2019-01-02 00:00:00+00:00  False  False
2019-01-03 00:00:00+00:00  False  False
...                          ...    ...
2019-12-30 00:00:00+00:00  False  False
2019-12-31 00:00:00+00:00  False  False
2020-01-01 00:00:00+00:00  False  False

[366 rows x 2 columns]

>>> exits = fast_ma.ma_crossed_below(slow_ma)
>>> exits
fast_window                   10     20
slow_window                   30     30
Date
2019-01-01 00:00:00+00:00  False  False
2019-01-02 00:00:00+00:00  False  False
2019-01-03 00:00:00+00:00  False  False
...                          ...    ...
2019-12-30 00:00:00+00:00  False  False
2019-12-31 00:00:00+00:00  False  False
2020-01-01 00:00:00+00:00  False  False

[366 rows x 2 columns]

>>> pf = vbt.Portfolio.from_signals(btc_price, entries, exits)
>>> pf.total_return()
fast_window  slow_window
10           30             0.848840
20           30             0.543411
Name: total_return, dtype: float64
```

For convenience, VectorBT automatically creates the column levels `fast_window` and `slow_window`, 
making it easy to tell which window size corresponds to which column.

Notice how the signal generation code stays the same across examples — most functions in VectorBT work on
time series of any shape. This makes it possible to build analysis pipelines that are universal to the input data.

Representing different features as columns opens up endless possibilities for backtesting.
For example, we can go a step further and test the same strategy on Ethereum. To compare both instruments,
simply combine the Bitcoin and Ethereum price series into one DataFrame and run the same backtesting pipeline.

```pycon
>>> # Multiple strategy instances and instruments
>>> eth_price = vbt.YFData.download('ETH-USD', start=start, end=end).get('Close')
>>> comb_price = btc_price.vbt.concat(eth_price,
...     keys=pd.Index(['BTC', 'ETH'], name='symbol'))
>>> comb_price.vbt.drop_levels(-1, inplace=True)
>>> comb_price
symbol                             BTC         ETH
Date
2019-01-01 00:00:00+00:00  3843.520020  140.819412
2019-01-02 00:00:00+00:00  3943.409424  155.047684
2019-01-03 00:00:00+00:00  3836.741211  149.135010
...                                ...         ...
2019-12-30 00:00:00+00:00  7292.995117  132.633484
2019-12-31 00:00:00+00:00  7193.599121  129.610855
2020-01-01 00:00:00+00:00  7200.174316  130.802002

[366 rows x 2 columns]

>>> fast_ma = vbt.MA.run(comb_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(comb_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> entries
fast_window                          10            20
slow_window                          30            30
symbol                       BTC    ETH    BTC    ETH
Date
2019-01-01 00:00:00+00:00  False  False  False  False
2019-01-02 00:00:00+00:00  False  False  False  False
2019-01-03 00:00:00+00:00  False  False  False  False
...                          ...    ...    ...    ...
2019-12-30 00:00:00+00:00  False  False  False  False
2019-12-31 00:00:00+00:00  False  False  False  False
2020-01-01 00:00:00+00:00  False  False  False  False

[366 rows x 4 columns]

>>> exits = fast_ma.ma_crossed_below(slow_ma)
>>> exits
fast_window                          10            20
slow_window                          30            30
symbol                       BTC    ETH    BTC    ETH
Date
2019-01-01 00:00:00+00:00  False  False  False  False
2019-01-02 00:00:00+00:00  False  False  False  False
2019-01-03 00:00:00+00:00  False  False  False  False
...                          ...    ...    ...    ...
2019-12-30 00:00:00+00:00  False  False  False  False
2019-12-31 00:00:00+00:00  False  False  False  False
2020-01-01 00:00:00+00:00  False  False  False  False

[366 rows x 4 columns]

>>> pf = vbt.Portfolio.from_signals(comb_price, entries, exits)
>>> pf.total_return()
fast_window  slow_window  symbol
10           30           BTC       0.848840
                          ETH       0.244204
20           30           BTC       0.543411
                          ETH      -0.319102
Name: total_return, dtype: float64

>>> mean_return = pf.total_return().groupby('symbol').mean()
>>> mean_return.vbt.barplot(xaxis_title='Symbol', yaxis_title='Mean total return')
```

![](/assets/images/index_by_symbol.svg)

Strategies and instruments aren't the only things that can act as separate features — time can too. 
If we want to find out when our strategy performs best, we can backtest over multiple time periods. 
VectorBT can split one time period into many segments of the same length and frequency and represent 
them as distinct columns. For example, let's split the entire time period into two equal halves 
and backtest them at once.

```pycon
>>> # Multiple strategy instances, instruments, and time periods
>>> mult_comb_price, _ = comb_price.vbt.range_split(n=2)
>>> mult_comb_price
split_idx                         0                         1
symbol              BTC         ETH           BTC         ETH
0           3843.520020  140.819412  11961.269531  303.099976
1           3943.409424  155.047684  11215.437500  284.523224
2           3836.741211  149.135010  10978.459961  287.997528
...                 ...         ...           ...         ...
180        10817.155273  290.695984   7292.995117  132.633484
181        10583.134766  293.641113   7193.599121  129.610855
182        10801.677734  291.596436   7200.174316  130.802002

[183 rows x 4 columns]

>>> fast_ma = vbt.MA.run(mult_comb_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(mult_comb_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_crossed_above(slow_ma)
>>> exits = fast_ma.ma_crossed_below(slow_ma)

>>> pf = vbt.Portfolio.from_signals(mult_comb_price, entries, exits, freq='1D')
>>> pf.total_return()
fast_window  slow_window  split_idx  symbol
10           30           0          BTC       1.632259
                                     ETH       0.946786
                          1          BTC      -0.288720
                                     ETH      -0.308387
20           30           0          BTC       1.721449
                                     ETH       0.343274
                          1          BTC      -0.418280
                                     ETH      -0.257947
Name: total_return, dtype: float64
```

Notice how the index is no longer datetime-like, since it now captures multiple time periods.
That's why we need to pass the frequency `freq` to the `Portfolio`
class so it can compute performance metrics such as the Sharpe ratio.

The index hierarchy of the resulting performance series can then be used to group results
by any feature, such as window pair, symbol, or time period.

```pycon
>>> mean_return = pf.total_return().groupby(['split_idx', 'symbol']).mean()
>>> mean_return.unstack(level=-1).vbt.barplot(
...     xaxis_title='Split index',
...     yaxis_title='Mean total return',
...     legend_title_text='Symbol')
```

![](/assets/images/index_by_any.svg)

There is much more to backtesting than stacking columns: VectorBT provides tools for
every part of the backtesting pipeline — from building indicators and generating signals, to
modeling portfolio performance and visualizing results.

## Disclaimer

This software is for educational purposes only. Do not risk money you cannot afford to lose.

Use the software at your own risk. The authors and affiliates assume no responsibility for your trading results.
