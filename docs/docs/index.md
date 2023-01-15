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

## What is vectorbt?

vectorbt is a Python package for quantitative analysis that takes a novel approach to backtesting: 
it operates entirely on pandas and NumPy objects, and is accelerated by [Numba](https://github.com/numba/numba) 
to analyze any data at speed and scale. This allows for testing of many thousands of strategies in **seconds**.

In contrast to other backtesters, vectorbt represents complex data as (structured) NumPy arrays.
This enables superfast computation using vectorized operations with NumPy and non-vectorized but dynamically 
compiled operations with Numba. It also integrates [Plotly](https://github.com/plotly/plotly.py) and 
[Jupyter Widgets](https://github.com/jupyter-widgets/ipywidgets) to display complex charts and dashboards akin 
to Tableau right in the Jupyter notebook. Due to high performance, vectorbt can process large amounts of 
data even without GPU and parallelization and enables the user to interact with data-hungry widgets 
without significant delays.

With vectorbt, you can

<div class="grid cards" markdown>

- :fontawesome-solid-wind:{ .lg .middle } 

    ---

    Backtest strategies in a couple of lines of Python code

- :fontawesome-solid-gauge-simple-high:{ .lg .middle } 

    ---

    Enjoy the best of both worlds: the ecosystem of Python and the speed of C

- :fontawesome-regular-eye:{ .lg .middle } 

    ---

    Retain full control over execution and your data (as opposed to web-based services such as TradingView)

- :fontawesome-solid-flask:{ .lg .middle } 

    ---

    Optimize your trading strategy against many parameters, assets, and periods in one go

- :fontawesome-solid-magnifying-glass-dollar:{ .lg .middle } 

    ---

    Uncover hidden patterns in financial markets

- :fontawesome-solid-robot:{ .lg .middle } 

    ---

    Analyze time series and engineer new features for ML models

- :fontawesome-solid-meteor:{ .lg .middle } 

    ---

    Supercharge pandas and your favorite tools to run much faster

- :fontawesome-solid-chart-pie:{ .lg .middle } 

    ---

    Visualize strategy performance using interactive charts and dashboards (both in Jupyter and browser)

- :fontawesome-regular-gem:{ .lg .middle } 

    ---

    Fetch and process data periodically, send Telegram notifications, and more

- :gem:{ .lg .middle }

    ---

    [__Support us__](https://vectorbt.pro/) to get access to parallelization, portfolio optimization, 
    pattern recognition, event projections, limit orders, leverage, and 100+ other hot features!

</div>

## Why vectorbt?

While there are many great backtesting packages for Python, vectorbt combines an extremely fast 
backtester and a data science tool: it excels at processing performance and offers interactive tools 
to explore complex phenomena in trading. With it, you can traverse a huge number of strategy 
configurations, time periods, and instruments in little time, to explore where your strategy 
performs best and to uncover hidden patterns in data. Accessing and analyzing this information 
for yourself could give you an information advantage in your own trading.

## How it works

vectorbt was implemented to address common performance shortcomings of backtesting libraries. 
It builds upon the idea that each instance of a trading strategy can be represented in a vectorized form, 
so multiple strategy instances can be packed into a single multi-dimensional array, processed in a 
highly efficient manner, and compared easily. It overhauls the traditional OOP approach that represents 
strategies as classes and other data structures, which are easier to write and extend compared to vectors, 
but harder to analyze and also require additional effort to do it quickly.

Thanks to the time-series nature of trading data, most of the aspects related to backtesting can be 
translated into vectors. Instead of processing one element at a time, vectorization allows us to avoid 
naive looping and perform the same operation on all elements at the same time. The path-dependency 
problem related to vectorization is solved by using Numba - it allows both writing iterative code 
and compiling slow Python loops to be run at the native machine code speed.

## Example

Let's say we have a complex strategy that has lots of (hyper-)parameters that have to be tuned. While
brute-forcing all combinations seems to be a rather unrealistic attempt, we can still interpolate, and
vectorbt makes exactly this possible. It doesn't care whether we have one strategy instance or millions.
As soon as their vectors can be concatenated into a matrix and we have enough memory, we can analyze
them in one go.

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

We are going to test a simple Dual Moving Average Crossover (DMAC) strategy. For this, we are going to
use `MA` class for calculating moving averages and generating signals.

Our first test is rather simple: buy when the 10-day moving average crosses above the 20-day moving
average, and sell when opposite.

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

One strategy instance of DMAC produced one column in signals and one performance value.

Adding one more strategy instance is as simple as adding one more column. Here we are passing an array of
window sizes instead of a single value. For each window size in this array, it computes a moving
average over the entire price series and stores it in a distinct column.

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

For the sake of convenience, vectorbt has created the column levels `fast_window` and `slow_window` for us
to easily distinguish which window size corresponds to which column.

Notice how signal generation part remains the same for each example - most functions in vectorbt work on
time series of any shape. This allows creation of analysis pipelines that are universal to input data.

The representation of different features as columns offers endless possibilities for backtesting.
We could, for example, go a step further and conduct the same tests for Ethereum. To compare both instruments,
combine price series for Bitcoin and Ethereum into one DataFrame and run the same backtesting pipeline.

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

Not only strategies and instruments can act as separate features, but also time. If we want to find out
when our strategy performs best, it's reasonable to backtest over multiple time periods. vectorbt allows
us to split one time period into many, given they have the same length and frequency, and represent
them as distinct columns. For example, let's split the whole time period into two equal time periods
and backest them at once.

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

Notice how index is no more datetime-like, since it captures multiple time periods.
That's why it's required here to pass the frequency `freq` to the `Portfolio`
class in order to be able to compute performance metrics such as the Sharpe ratio.

The index hierarchy of the final performance series can be then used to group the performance
by any feature, such as window pair, symbol, and time period.

```pycon
>>> mean_return = pf.total_return().groupby(['split_idx', 'symbol']).mean()
>>> mean_return.unstack(level=-1).vbt.barplot(
...     xaxis_title='Split index',
...     yaxis_title='Mean total return',
...     legend_title_text='Symbol')
```

![](/assets/images/index_by_any.svg)

There is much more to backtesting than simply stacking columns: vectorbt offers functions for
most parts of a backtesting pipeline - from building indicators and generating signals, to
modeling portfolio performance and visualizing results.

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose.
USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.