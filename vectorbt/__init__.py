"""Ultimate Python library for time series analysis and backtesting at scale.

While there are many great backtesting packages for Python, vectorbt combines an extremely fast backtester 
and a data science tool: it excels at processing performance and offers interactive tools to explore complex 
phenomena in trading. With it, you can traverse a huge number of strategy configurations, time periods, and 
instruments in little time, to explore where your strategy performs best and to uncover hidden patterns in data. 
Accessing and analyzing this information for yourself could give you an information advantage in your own trading.

## How it works?

vectorbt was implemented to address common performance shortcomings of backtesting libraries.
It builds upon the idea that each instance of a trading strategy can be represented in a vectorized form,
so multiple strategy instances can be packed into a single multi-dimensional array, processed in a highly
efficient manner, and compared easily. It overhauls the traditional OOP approach that represents strategies
as classes or other data structures, which are easier to write and extend compared to vectors, but harder to 
analyze and also require additional effort to do it quickly.

Thanks to the time-series nature of trading data, most of the aspects related to backtesting can be translated
into vectors. Instead of processing one element at a time, vectorization allows us to avoid naive
looping and perform the same operation on all elements at the same time. The path-dependency problem
related to vectorization is solved by using Numba - it allows both writing iterative code and compiling slow
Python loops to be run at the native machine code speed.

## Performance

While it might seem tempting to perform all sorts of computations with pandas alone, the NumPy+Numba combo
outperforms pandas significantly:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import vectorbt as vbt

>>> big_ts = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

>>> %timeit big_ts.pct_change()
280 ms ± 12.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit big_ts.vbt.pct_change()
5.95 ms ± 380 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

But also pandas functions already compiled with Cython/Numba are often slower:

```python-repl
>>> %timeit big_ts.expanding().max()
48.4 ms ± 557 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit big_ts.vbt.expanding_max()
8.82 ms ± 121 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Moreover, pandas functions cannot be accessed within user-defined Numba code, since Numba cannot do any
compilation on pandas objects. Take for example generating trailing stop orders: to calculate expanding
maximum for each order, we cannot simply do `df.expanding().max()` from within Numba, but we must write
and compile our own expanding max function wrapped with `@njit`. That's why vectorbt provides an arsenal
of Numba-compiled functions for any sort of task.

## Usability

From the user's perspective, working with NumPy and Numba alone is not easy, since important information
in form of index and columns and all typing checks must be explicitly handled by the user,
making analysis prone to errors. That's why vectorbt introduces a namespace (accessor) to pandas objects
(see [extending pandas](https://pandas.pydata.org/pandas-docs/stable/development/extending.html)).
This way, user can easily switch between pandas and vectorbt functionality. Moreover, each vectorbt
method is flexible towards input and can work on both Series and DataFrames.

Another argument against using exclusively NumPy is iterative code: sometimes vectorized implementation is hard
to read or cannot be properly defined at all, and one must rely on an iterative approach instead -
processing data in an element-by-element fashion. That's where Numba comes into play.

The [previous versions](https://github.com/polakowo/vectorbt/tree/9f270820dd3e5dc4ff5468dbcc14a29c4f45f557)
of vectorbt were written in pure NumPy, which resulted in more performance but less usability.

### Indexing

vectorbt makes use of [hierarchical indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
to store valuable information on each backtest. Take a simple crossover strategy as example:
it depends on the size of the fast and slow windows, and other hyperparameters such as whether
it is SMA or EMA. Each of these hyperparameters becomes an additional dimension for manipulating data
and gets stored as a separate column level. Below is an example of a column hierarchy for MACD:

```python-repl
>>> macd = vbt.MACD.run(
...     pd.Series([1, 2, 3, 4, 3, 2, 1]),
...     fast_window=(2, 3),
...     slow_window=(3, 4),
...     signal_window=(2, 3),
...     macd_ewm=(True, False),
...     signal_ewm=(False, True)
... )

>>> macd.signal
macd_fast_window           2         3
macd_slow_window           3         4
macd_signal_window         2         3
macd_macd_ewm           True     False
macd_signal_ewm        False      True
0                        NaN       NaN
1                        NaN       NaN
2                        NaN       NaN
3                   0.349537       NaN
4                   0.251929       NaN
5                  -0.014982  0.208333
6                  -0.221140 -0.145833
```

Columns here capture different strategy configurations that can now be easily analyzed and compared.
We might, for example, consider grouping our performance by `macd_fast_window` to see how the size of
the fast window impacts profitability of the strategy.

The other advantage of vectorbt is that it ensures that the column hierarchy is preserved across
the whole backtesting pipeline - from signal generation to performance modeling.

### Broadcasting

vectobt borrows broadcasting rules from NumPy. For example, consider the following objects:

```python-repl
>>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
>>> sr
x    1
y    2
z    3
dtype: int64

>>> df = pd.DataFrame([[4, 5, 6]], index=['x', 'y', 'z'], columns=['a', 'b', 'c'])
>>> df
   a  b  c
x  4  5  6
y  4  5  6
z  4  5  6
```

Despite both having the same index, pandas won't add them correctly:

```python-repl
>>> sr + df  # pandas
    a   b   c   x   y   z
x NaN NaN NaN NaN NaN NaN
y NaN NaN NaN NaN NaN NaN
z NaN NaN NaN NaN NaN NaN
```

And here is the expected result using vectorbt:

```python-repl
>>> sr.vbt + df  # vectorbt
   a  b  c
x  5  6  7
y  6  7  8
z  7  8  9
```

In case where index or columns in both objects are different, they are stacked upon each other:

```python-repl
>>> df2 = pd.DataFrame([[4, 5, 6]], index=['x', 'y', 'z'], columns=['a2', 'b2', 'c2'])
>>> df2
   a2  b2  c2
x   4   5   6
y   4   5   6
z   4   5   6

>>> df + df2  # pandas
    a  a2   b  b2   c  c2
x NaN NaN NaN NaN NaN NaN
y NaN NaN NaN NaN NaN NaN
z NaN NaN NaN NaN NaN NaN

>>> df.vbt + df2  # vectorbt
   a   b   c
  a2  b2  c2
x  8  10  12
y  8  10  12
z  8  10  12
```

This way, we can perform operations on objects of arbitrary broadcastable shapes and still
preserve their individual information. This is handy for combining DataFrames with lots of metadata,
such as indicators or signals with many hyperparameters.

Another feature of vectorbt is that it can broadcast objects with incompatible shapes but common
multi-index levels - those having the same name, or being without name but having overlapping values.

For example:

```python-repl
>>> df3 = pd.DataFrame(
...     [[7, 8, 9, 10, 11, 12]],
...     index=['x', 'y', 'z'],
...     columns=pd.MultiIndex.from_tuples([
...         (1, 'a'),
...         (1, 'b'),
...         (1, 'c'),
...         (2, 'a'),
...         (2, 'b'),
...         (2, 'c'),
...     ]))
>>> df3
   1         2
   a  b  c   a   b   c
x  7  8  9  10  11  12
y  7  8  9  10  11  12
z  7  8  9  10  11  12

>>> df + df3  # pandas
ValueError: cannot join with no overlapping index names

>>> df.vbt + df3  # vectorbt
    1           2
    a   b   c   a   b   c
x  11  13  15  14  16  18
y  11  13  15  14  16  18
z  11  13  15  14  16  18
```

## Example

To better understand how these concepts fit together, consider the following example.

We have a complex strategy that has lots of (hyper-)parameters that have to be tuned. While
brute-forcing all combinations seems to be a rather unrealistic attempt, we can still interpolate, and
vectorbt makes exactly this possible. It doesn't care whether we have one strategy instance or millions.
As soon as their vectors can be concatenated into a matrix and we have enough memory, we can analyze
them in one go.

Let's start with fetching the daily price of Bitcoin:

```python-repl
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
use `vectorbt.indicators.basic.MA` class for calculating moving averages and generating signals.

Our first test is rather simple: buy when the 10-day moving average crosses above the 20-day moving
average, and sell when opposite.

```python-repl
>>> fast_ma = vbt.MA.run(btc_price, 10, short_name='fast')
>>> slow_ma = vbt.MA.run(btc_price, 20, short_name='slow')

>>> entries = fast_ma.ma_above(slow_ma, crossover=True)
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

>>> exits = fast_ma.ma_below(slow_ma, crossover=True)
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

>>> portfolio = vbt.Portfolio.from_signals(btc_price, entries, exits)
>>> portfolio.total_return()
0.636680693047752
```

One strategy instance of DMAC produced one column in signals and one performance value.

Adding one more strategy instance is as simple as adding a new column. Here we are passing an array of
window sizes instead of a single value. For each window size in this array, it computes a moving
average over the entire price series and stores it as a distinct column.

```python-repl
>>> # Multiple strategy instances: (10, 30) and (20, 30)
>>> fast_ma = vbt.MA.run(btc_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(btc_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_above(slow_ma, crossover=True)
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

>>> exits = fast_ma.ma_below(slow_ma, crossover=True)
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

>>> portfolio = vbt.Portfolio.from_signals(btc_price, entries, exits)
>>> portfolio.total_return()
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

```python-repl
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

>>> entries = fast_ma.ma_above(slow_ma, crossover=True)
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

>>> exits = fast_ma.ma_below(slow_ma, crossover=True)
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

>>> portfolio = vbt.Portfolio.from_signals(comb_price, entries, exits)
>>> portfolio.total_return()
fast_window  slow_window  symbol
10           30           BTC       0.848840
                          ETH       0.244204
20           30           BTC       0.543411
                          ETH      -0.319102
Name: total_return, dtype: float64

>>> mean_return = portfolio.total_return().groupby('symbol').mean()
>>> mean_return.vbt.barplot(xaxis_title='Symbol', yaxis_title='Mean total return')
```

![](/vectorbt/docs/img/index_by_symbol.svg)

Not only strategies and instruments can act as separate features, but also time. If we want to find out
when our strategy performs best, it's reasonable to test it over multiple time periods. vectorbt allows
us to split one time period into many, given they have the same length and frequency, and represent
them as distinct columns. For example, let's split the whole time period into two equal time periods
and backest them at once.

```python-repl
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

>>> entries = fast_ma.ma_above(slow_ma, crossover=True)
>>> exits = fast_ma.ma_below(slow_ma, crossover=True)

>>> portfolio = vbt.Portfolio.from_signals(mult_comb_price, entries, exits, freq='1D')
>>> portfolio.total_return()
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
That's why it's required here to pass the frequency `freq` to the `vectorbt.portfolio.base.Portfolio`
class method in order to be able to compute performance metrics such as the Sharpe ratio.

The index hierarchy of the final performance series can be then used to group the performance
by any feature, such as window pair, symbol, and time period.

```python-repl
>>> mean_return = portfolio.total_return().groupby(['split_idx', 'symbol']).mean()
>>> mean_return.unstack(level=-1).vbt.barplot(
...     xaxis_title='Split index',
...     yaxis_title='Mean total return',
...     legend_title_text='Symbol')
```

![](/vectorbt/docs/img/index_by_any.svg)

There is much more to backtesting than simply stacking columns: vectorbt offers functions for
most parts of a backtesting pipeline, from building indicators and generating signals, to
modeling portfolio performance and visualizing results.

## Resources

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

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose.
USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
"""

__pdoc__ = {}

# Import version
from vectorbt._version import __version__ as _version
__version__ = _version

# Most important modules
from vectorbt.generic import nb, plotting
from vectorbt._settings import settings

# Most important classes
from vectorbt.utils import *
from vectorbt.base import *
from vectorbt.data import *
from vectorbt.generic import *
from vectorbt.indicators import *
from vectorbt.signals import *
from vectorbt.records import *
from vectorbt.portfolio import *
from vectorbt.labels import *
from vectorbt.messaging import *

# Import all submodules
from vectorbt.utils.module import import_submodules

# silence NumbaExperimentalFeatureWarning
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)

import_submodules(__name__)

__pdoc__['_settings'] = True
