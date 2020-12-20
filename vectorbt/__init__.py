"""Python library for backtesting and analyzing trading strategies at scale.

While there are many great backtesting packages for Python, vectorbt was designed specifically for data science:
it excels at processing performance and offers interactive tools to explore complex phenomena in trading. 
With it you can traverse a huge number of strategy configurations, time periods and instruments in seconds, 
to explore where your strategy performs best and to uncover hidden patterns in data. Accessing and analyzing
this information for yourself could give you an information advantage in your own trading.

## How it works

vectorbt was implemented to address common performance shortcomings of backtesting libraries.
It builds upon the idea that each instance of a trading strategy can be represented in a vectorized form,
so multiple strategy instances can be packed into a single multi-dimensional array, processed in a highly
efficient manner, and compared easily. It overhauls the traditional OOP approach that represents strategies
as classes or other data structures, which are far more easier to write and extend, but harder to analyze
compared to vectors, and also require additional effort to do it fast.

Thanks to the time series nature of trading data, most of the aspects related to backtesting can be translated
into vectors. Instead of processing one element at a time, vectorization allows us to avoid naive
looping and perform the same operation on all elements at the same time. The path-dependency problem
related to vectorization is solved by using Numba - it allows both writing iterative code and compiling slow
Python loops to be run at native machine code speed.

## Performance

While it might seem tempting to perform all sorts of computations with pandas alone, the NumPy+Numba combo
outperforms pandas significantly, especially for basic operations:

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

But also pandas functions already compiled with Cython/Numba are slower:

```python-repl
>>> %timeit big_ts.expanding().max()
48.4 ms ± 557 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit big_ts.vbt.expanding_max()
8.82 ms ± 121 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Moreover, pandas functions cannot be accessed within user-defined Numba code, since Numba cannot do any
compilation on pandas objects. Take for example generating trailing stop orders: to calculate expanding
maximum for each order, you cannot simply do `df.expanding().max()` from within Numba, but you must write
and compile your own expanding max function wrapped with `@njit`. That's why vectorbt provides an arsenal
of Numba-compiled functions for any sort of task.

## Usability

From the user's point of view, working with NumPy and Numba alone is difficult, since important information
in form of index and columns and all indexing checks must be explicitly handled by the user,
making analysis prone to errors. That's why vectorbt introduces a namespace (accessor) to pandas objects
(see [extending pandas](https://pandas.pydata.org/pandas-docs/stable/development/extending.html)).
This way, user can easily switch between pandas and vectorbt functionality. Moreover, each vectorbt
method is flexible towards inputs and can work on both Series and DataFrames.

Another argument against using exclusively NumPy is iterative code: sometimes vectorized implementation is hard
to read or cannot be properly defined at all, and one must rely on an iterative approach instead -
processing data in element-by-element fashion. That's where Numba comes into play.

The [previous versions](https://github.com/polakowo/vectorbt/tree/9f270820dd3e5dc4ff5468dbcc14a29c4f45f557)
of vectorbt were written in pure NumPy which led to greater performance but lesser usability.

### Indexing

vectorbt makes use of [hierarchical indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
to store valuable information on each backtest. Take for example a simple crossover strategy:
it depends on the size of the fast and slow windows, and other hyper-parameters such as whether
it is SMA or EMA. Each of these hyper-parameters becomes an additional dimension for manipulating data
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
You might, for example, consider grouping your performance by `macd_fast_window` to see how the size of
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

Despite both having the same index, pandas can't figure out how to add them correctly:

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

This way, you can perform operations on objects of arbitrary broadcastable shapes, and still
preserve their individual information. This is handy for combining DataFrames with lots of metadata,
such as indicators or signals with many hyperparameters.

Another feature of vectorbt is that it can broadcast objects with incompatible shapes but common
multi-index levels - those that have the same name, or are without name but have overlapping values.

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

You have a complex strategy that has lots of (hyper-)parameters that have to be tuned. While
brute-forcing all combinations seems to be a rather unrealistic attempt, you can still interpolate, and
vectorbt makes exactly this possible. It doesn't care whether you have one strategy instance or millions.
As soon as their vectors can be concatenated into a matrix and you have enough memory, you can analyze
them in one go.

Let's start with fetching the daily price of Bitcoin:

```python-repl
>>> import numpy as np
>>> import pandas as pd
>>> import yfinance as yf
>>> from datetime import datetime

>>> import vectorbt as vbt

>>> # Prepare data
>>> start = datetime(2019, 1, 1)
>>> end = datetime(2020, 1, 1)
>>> btc_price = yf.Ticker("BTC-USD").history(start=start, end=end)['Open']

>>> btc_price
Date
2018-12-31    3866.84
2019-01-01    3746.71
2019-01-02    3849.22
               ...
2019-12-29    7317.65
2019-12-30    7420.27
2019-12-31    7294.44
Name: Open, Length: 366, dtype: float64
```

We are going to test a simple Dual Moving Average Crossover (DMAC) strategy. For this, we are going to
use `vectorbt.indicators.basic.MA` class for calculating moving averages and generating signals.

Our first test is rather simple: buy when the 10-day moving average crosses above the 20-day moving
average, and sell when the opposite happens.

```python-repl
>>> fast_ma = vbt.MA.run(btc_price, 10, short_name='fast')
>>> slow_ma = vbt.MA.run(btc_price, 20, short_name='slow')

>>> entries = fast_ma.ma_above(slow_ma, crossed=True)
>>> entries
Date
2018-12-31    False
2019-01-01    False
2019-01-02    False
              ...
2019-12-29    False
2019-12-30    False
2019-12-31    False
Name: (10, 20, Open), Length: 366, dtype: bool

>>> exits = fast_ma.ma_below(slow_ma, crossed=True)
>>> exits
Date
2018-12-31    False
2019-01-01    False
2019-01-02    False
              ...
2019-12-29    False
2019-12-30    False
2019-12-31    False
Name: (10, 20, Open), Length: 366, dtype: bool

>>> portfolio = vbt.Portfolio.from_signals(btc_price, entries, exits)
>>> portfolio.total_return()
0.663318597097753
```

One strategy instance of DMAC produced one column in signals and one performance value.

Adding one more strategy instance is as simple as adding a new column. Here we are passing an array of
window sizes instead of a single value. For each window size in this array, it computes a moving
average over the entire price series and stores it as a distinct column.

```python-repl
>>> # Multiple strategy instances: (10, 30) and (20, 30)
>>> fast_ma = vbt.MA.run(btc_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(btc_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_above(slow_ma, crossed=True)
>>> entries
fast_window     10     20
slow_window     30     30
Date
2018-12-31   False  False
2019-01-01   False  False
2019-01-02   False  False
...            ...    ...
2019-12-29   False  False
2019-12-30    True  False
2019-12-31   False  False

[366 rows x 2 columns]

>>> exits = fast_ma.ma_below(slow_ma, crossed=True)
>>> exits
fast_window     10     20
slow_window     30     30
Date
2018-12-31   False  False
2019-01-01   False  False
2019-01-02   False  False
...            ...    ...
2019-12-29   False  False
2019-12-30   False  False
2019-12-31   False  False

[366 rows x 2 columns]

>>> portfolio = vbt.Portfolio.from_signals(btc_price, entries, exits)
>>> portfolio.total_return()
fast_window  slow_window
10           30             0.865956
20           30             0.547047
dtype: float64
```

For the sake of convenience, vectorbt has created column levels `fast_window` and `slow_window` for you
to easily identify which window size corresponds to which column.

Notice how signal generation part remains the same for each example - most functions in vectorbt work on
time series of any shape. This allows creation of analysis pipelines that are universal to input data.

The representation of different features as columns offers endless possibilities for backtesting.
You could, for example, go a step further and conduct the same tests for Ethereum. To compare both instruments,
combine price series for Bitcoin and Ethereum into one DataFrame and run the same backtesting pipeline on it.

```python-repl
>>> # Multiple strategy instances and instruments
>>> eth_price = yf.Ticker("ETH-USD").history(start=start, end=end)['Open']
>>> comb_price = btc_price.vbt.concat(eth_price,
...     keys=pd.Index(['BTC', 'ETH'], name='asset'))
>>> comb_price
asset           BTC     ETH
Date
2018-12-31  3866.84  140.03
2019-01-01  3746.71  133.42
2019-01-02  3849.22  141.52
...             ...     ...
2019-12-29  7317.65  128.27
2019-12-30  7420.27  134.80
2019-12-31  7294.44  132.61

[366 rows x 2 columns]

>>> fast_ma = vbt.MA.run(comb_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(comb_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_above(slow_ma, crossed=True)
>>> entries
fast_window     10            20
slow_window     30            30
asset          BTC    ETH    BTC    ETH
Date
2018-12-31   False  False  False  False
2019-01-01   False  False  False  False
2019-01-02   False  False  False  False
...            ...    ...    ...    ...
2019-12-29   False  False  False  False
2019-12-30    True  False  False  False
2019-12-31   False  False  False  False

[366 rows x 4 columns]

>>> exits = fast_ma.ma_below(slow_ma, crossed=True)
>>> exits
fast_window     10            20
slow_window     30            30
asset          BTC    ETH    BTC    ETH
Date
2018-12-31   False  False  False  False
2019-01-01   False  False  False  False
2019-01-02   False  False  False  False
...            ...    ...    ...    ...
2019-12-29   False  False  False  False
2019-12-30   False  False  False  False
2019-12-31   False  False  False  False

[366 rows x 4 columns]

>>> portfolio = vbt.Portfolio.from_signals(comb_price, entries, exits)
>>> portfolio.total_return()
fast_window  slow_window  asset
10           30           BTC      0.865956
                          ETH      0.249013
20           30           BTC      0.547047
                          ETH     -0.319945
dtype: float64

>>> mean_return = portfolio.total_return().groupby('asset').mean()
>>> mean_return.vbt.bar(
...     xaxis_title='Asset',
...     yaxis_title='Mean total return')
```

![](/vectorbt/docs/img/index_by_asset.png)

Not only strategies and instruments can act as separate features, but also time. If you want to find out
when your strategy performs best, it's reasonable to test it over multiple time periods. vectorbt allows
you to split one time period into many (given they have the same length and frequency) and represent
them as distinct columns. For example, let's split `[2019-1-1, 2020-1-1]` into two equal time periods -
`[2018-12-31, 2019-07-01]` and `[2019-07-02, 2019-12-31]`, and backtest them all at once.

```python-repl
>>> # Multiple strategy instances, instruments and time periods
>>> mult_comb_price = comb_price.vbt.split_into_ranges(n=2)
>>> mult_comb_price
asset             BTC                   ETH
range_start 2018-12-31 2019-07-02 2018-12-31 2019-07-02
range_end   2019-07-01 2019-12-31 2019-07-01 2019-12-31
0              3866.84   10588.68     140.03     293.54
1              3746.71   10818.16     133.42     291.76
2              3849.22   11972.72     141.52     303.03
3              3931.05   11203.10     155.20     284.38
4              3832.04   10982.54     148.91     287.89
..                 ...        ...        ...        ...
178           13017.12    7238.14     336.96     126.37
179           11162.17    7289.03     294.14     127.21
180           12400.76    7317.65     311.28     128.27
181           11931.99    7420.27     319.58     134.80
182           10796.93    7294.44     290.27     132.61

[183 rows x 4 columns]

>>> fast_ma = vbt.MA.run(mult_comb_price, [10, 20], short_name='fast')
>>> slow_ma = vbt.MA.run(mult_comb_price, [30, 30], short_name='slow')

>>> entries = fast_ma.ma_above(slow_ma, crossed=True)
>>> exits = fast_ma.ma_below(slow_ma, crossed=True)

>>> portfolio = vbt.Portfolio.from_signals(mult_comb_price, entries, exits, freq='1D')
>>> portfolio.total_return()
fast_window  slow_window  asset  range_start  range_end
10           30           BTC    2018-12-31   2019-07-01    1.631617
                                 2019-07-02   2019-12-31   -0.281432
                          ETH    2018-12-31   2019-07-01    0.941945
                                 2019-07-02   2019-12-31   -0.306689
20           30           BTC    2018-12-31   2019-07-01    1.725547
                                 2019-07-02   2019-12-31   -0.417770
                          ETH    2018-12-31   2019-07-01    0.336136
                                 2019-07-02   2019-12-31   -0.257854
dtype: float64
```

Notice how index is no more datetime-like, since it captures multiple time periods.
That's why it's required here to pass the frequency `freq` to the `vectorbt.portfolio.base.Portfolio`
class method in order to be able to compute performance metrics such as Sharpe ratio.

The index hierarchy of the final performance series can be then used to group performance
by any feature, such as window pair, asset, and time period.

```python-repl
>>> mean_return = portfolio.total_return().groupby(['range_end', 'asset']).mean()
>>> mean_return = mean_return.unstack(level=-1).vbt.bar(
...     xaxis_title='End date',
...     yaxis_title='Mean total return',
...     legend_title_text='Asset')
```

![](/vectorbt/docs/img/index_by_any.png)

There is much more to backtesting than simply stacking columns: vectorbt offers functions for
most parts of a common backtesting pipeline, from building indicators and generating signals, to
modeling portfolio performance and visualizing results.

## Resources

### Notebooks

- [Assessing performance of DMAC on Bitcoin](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/BitcoinDMAC.ipynb)
- [Comparing effectiveness of stop signals](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/StopSignals.ipynb)
- [Backtesting per trading session](https://nbviewer.jupyter.org/github/polakowo/vectorbt/blob/master/examples/TradingSessions.ipynb)

Note: you need to run the notebook to play with widgets.

### Dashboards

- [Detecting and backtesting common candlestick patterns](https://github.com/polakowo/vectorbt/tree/master/apps/candlestick-patterns)

### Articles

- [Stop Loss, Trailing Stop, or Take Profit? 2 Million Backtests Shed Light](https://polakowo.medium.com/stop-loss-trailing-stop-or-take-profit-2-million-backtests-shed-light-dde23bda40be)

## Disclaimer

This software is for educational purposes only. Do not risk money which you are afraid to lose.
USE THE SOFTWARE AT YOUR OWN RISK. THE AUTHORS AND ALL AFFILIATES ASSUME NO RESPONSIBILITY FOR YOUR TRADING RESULTS.
"""

# Load all accessors
import vectorbt.root_accessors
import vectorbt.base.accessors
import vectorbt.generic.accessors
import vectorbt.signals.accessors
import vectorbt.returns.accessors
import vectorbt.ohlcv.accessors

# Most important modules
from vectorbt.generic import nb, plotting
from vectorbt import settings

# Most important classes
from vectorbt.utils import *
from vectorbt.base import *
from vectorbt.generic import *
from vectorbt.indicators import *
from vectorbt.signals import *
from vectorbt.records import *
from vectorbt.portfolio import *

# silence NumbaExperimentalFeatureWarning
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)
