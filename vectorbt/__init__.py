"""Python library for backtesting and analyzing trading strategies at scale.

While there are many great backtesting packages for Python, vectorbt was designed specifically for data mining: 
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
compared to vectors and require additional effort to do it fast.

Thanks to the time series nature of trading data, most of the aspects related to backtesting can be translated
to vectors. Instead of performing operations on one element at a time, vectorization allows us to avoid naive
looping and perform the same operation on all the elements at the same time. The path-dependency problem
related to vectorization is solved by using Numba - it allows both writing iterative code and compiling slow
Python loops to be run at native machine code speed.

## Performance

While it might seem tempting to perform all sorts of computations with pandas alone, the NumPy+Numba combo
outperforms pandas significantly, especially for basic operations:

```python-repl
>>> import pandas as pd
>>> import vectorbt as vbt

>>> big_ts = pd.DataFrame(np.random.uniform(size=(1000, 1000)))

>>> %timeit big_ts.pct_change()
280 ms ± 12.6 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

>>> %timeit big_ts.vbt.tseries.pct_change()
5.95 ms ± 380 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

But also pandas functions that are already compiled with Cython/Numba are slower:

```python-repl
>>> %timeit big_ts.expanding().max()
48.4 ms ± 557 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

>>> %timeit big_ts.vbt.tseries.expanding_max()
8.82 ms ± 121 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

Moreover, pandas functions cannot be accessed within user-defined Numba code, since Numba cannot do any
compilation on pandas objects. Take for example generating trailing stop orders: to calculate expanding
maximum for each order, you cannot simply do `df.expanding().max()` from within Numba, but you must write
and compile your own expanding max function wrapped with `@njit`. That's why vectorbt provides an arsenal
of Numba-compiled functions that are ready to be used everywhere.

## Usability

Working with NumPy and Numba alone, from the user's point of view, is problematic, since important information
in form of index and columns and all indexing checks must be explicitly handled by the user,
making analysis prone to errors. That's why vectorbt introduces a namespace (accessor) to pandas objects
(see [extending pandas](https://pandas.pydata.org/pandas-docs/stable/development/extending.html)).
This way, user can easily switch between native pandas functionality such as indexing, and highly-efficient
vectorbt methods. Moreover, each vectorbt method is flexible and can work on both Series and DataFrames.

Another argument against using exclusively NumPy is iterative code: sometimes vectorized implementation is hard
to read or cannot be properly defined at all, and one must rely on an iterative approach instead,
which is processing data in element-by-element fashion. That's where Numba comes into play.

The [previous versions](https://github.com/polakowo/vectorbt/tree/9f270820dd3e5dc4ff5468dbcc14a29c4f45f557)
of vectorbt were written in pure NumPy which led to more performance but less usability.

### Indexing

vectorbt makes use of [hierarchical indexing](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)
to store valuable information on each backtest. Take for example a simple crossover strategy:
it depends on the size of the fast and slow windows, and other hyper-parameters such as whether
it is SMA or EMA. Each of these hyper-parameters becomes an additional dimension for manipulating data
and gets stored as a separate column level. Below is an example of a column hierarchy for MACD:

```python-repl
>>> import pandas as pd
>>> import vectorbt as vbt

>>> macd = vbt.MACD.from_params(
...     pd.Series([1, 2, 3, 4, 3, 2, 1]),
...     fast_window=(2, 3),
...     slow_window=(3, 4),
...     signal_window=(2, 3),
...     macd_ewm=(True, False),
...     signal_ewm=(False, True)
... )

>>> print(macd.signal)
macd_fast_window           2         3
macd_slow_window           3         4
macd_signal_window         2         3
macd_macd_ewm           True     False
macd_signal_ewm        False      True
0                        NaN       NaN
1                        NaN       NaN
2                        NaN       NaN
3                   0.235073       NaN
4                   0.168060       NaN
5                  -0.054956  0.166667
6                  -0.235246 -0.188889
```

Columns here capture different strategy configurations that can now be easily analyzed and compared.
You might, for example, consider grouping your performance by `macd_fast_window` to see how the size of
the fast window impacts profitability of the strategy.

The other advantage of vectorbt is that it ensures that the column hierarchy is preserved across
the whole backtesting pipeline, from signal generation, to performance modeling.

### Broadcasting

Moreover, vectobt borrows broadcasting rules from NumPy and implements its own indexing rules that
allow operations between pandas objects of compatible shapes, regardless of their index/columns -
those are simply stacked upon each other in the resulting object.

For example, consider the following objects:

```python-repl
>>> sr = pd.Series([1, 2, 3], index=['x', 'y', 'z'])
>>> print(sr)
x    1
y    2
z    3
dtype: int64

>>> df = pd.DataFrame([[4, 5, 6]], index=['x', 'y', 'z'], columns=['a', 'b', 'c'])
>>> print(df)
   a  b  c
x  4  5  6
y  4  5  6
z  4  5  6
```

Despite both having the same index, pandas can't figure out how to add them correctly:

```python-repl
>>> print(sr + df)
    a   b   c   x   y   z
x NaN NaN NaN NaN NaN NaN
y NaN NaN NaN NaN NaN NaN
z NaN NaN NaN NaN NaN NaN
```

And here is the expected result using vectorbt:

```python-repl
>>> print(sr.vbt + df)
   a  b  c
x  5  6  7
y  6  7  8
z  7  8  9
```

This way, you can perform operations on objects of arbitrary (but compatible) shapes, and
still preserve their index information. This is handy for combining complex DataFrames, such as
signals from different indicators.

## Example

To better understand how these concepts fit together in vectorbt, consider the following example.
You have a complex strategy that has lots of parameters. While brute-forcing all parameter combinations
seems to be a rather unrealistic attempt, vectorbt makes exactly this possible. It doesn't care whether
you have one strategy instance or millions. As soon as their vectors can be concatenated into a matrix,
you can analyze them in one go.

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

>>> print(btc_price)
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

We will test a simple Dual Moving Average Crossover (DMAC) strategy. For this, we will be using the
`vectorbt.indicators.basic.MA` class for calculating moving averages and generating signals.

Our first test will be rather simple: buy when the 10-day moving average crosses above the 20-day moving
average, and sell when the opposite happens.

```python-repl
>>> # (10, 20) - 10 day moving average crosses 20 day moving average
>>> fast_ma = vbt.MA.from_params(btc_price, 10, name='fast', hide_params=['ewm'])
>>> slow_ma = vbt.MA.from_params(btc_price, 20, name='slow', hide_params=['ewm'])

>>> entries = fast_ma.ma_above(slow_ma, crossed=True)
>>> print(entries)
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
>>> print(exits)
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
>>> print(portfolio.total_return)
0.6633185970977524
```

One strategy instance of DMAC produced one column in signals and one performance value.

Adding one more strategy instance is as simple as adding a new column. Here we are passing an array of
window sizes instead of a single value. For each window size in this array, it will compute a moving
average over the entire price series and store it as a distinct column.

```python-repl
>>> # Multiple strategy instances: (10, 30) and (20, 30)
>>> fast_ma = vbt.MA.from_params(btc_price,
...     [10, 20], name='fast', hide_params=['ewm'])
>>> slow_ma = vbt.MA.from_params(btc_price,
...     [30, 30], name='slow', hide_params=['ewm'])

>>> entries = fast_ma.ma_above(slow_ma, crossed=True)
>>> print(entries)
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
>>> print(exits)
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
>>> print(portfolio.total_return)
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
...     as_columns=pd.Index(['BTC', 'ETH'], name='asset'))
>>> print(comb_price)
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

>>> fast_ma = vbt.MA.from_params(comb_price,
...     [10, 20], name='fast', hide_params=['ewm'])
>>> slow_ma = vbt.MA.from_params(comb_price,
...     [30, 30], name='slow', hide_params=['ewm'])

>>> entries = fast_ma.ma_above(slow_ma, crossed=True)
>>> print(entries)
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
>>> print(exits)
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

>>> # Notice that we need to align the price to the shape of signals
>>> portfolio = vbt.Portfolio.from_signals(
...     comb_price.vbt.tile(2), entries, exits)
>>> print(portfolio.total_return)
fast_window  slow_window  asset
10           30           BTC      0.865956
                          ETH      0.249013
20           30           BTC      0.547047
                          ETH     -0.319945
dtype: float64

>>> mean_return = portfolio.total_return.groupby('asset').mean()
>>> mean_return.vbt.Bar(
...     xaxis_title='Asset',
...     yaxis_title='Mean total return').show_png()
```

![](/vectorbt/docs/img/index_by_asset.png)

Not only strategies and instruments can act as separate features, but also time! If you want to find out
when your strategy performs best, it's reasonable to test it over multiple time periods. vectorbt allows
you to split one time period into many (given they have the same length and frequency) and represent
them as distinct columns. For example, let's split `[2019-1-1, 2020-1-1]` into two equal time periods -
`[2018-12-31, 2019-07-01]` and `[2019-07-02, 2019-12-31]`, and backtest them all at once.

```python-repl
>>> # Multiple strategy instances, instruments and time periods
>>> mult_comb_price = comb_price.vbt.tseries.split_into_ranges(n=2)
>>> print(mult_comb_price)
asset             BTC                   ETH
start_date 2018-12-31 2019-07-02 2018-12-31 2019-07-02
end_date   2019-07-01 2019-12-31 2019-07-01 2019-12-31
0             3866.84   10588.68     140.03     293.54
1             3746.71   10818.16     133.42     291.76
2             3849.22   11972.72     141.52     303.03
3             3931.05   11203.10     155.20     284.38
4             3832.04   10982.54     148.91     287.89
..                ...        ...        ...        ...
178          13017.12    7238.14     336.96     126.37
179          11162.17    7289.03     294.14     127.21
180          12400.76    7317.65     311.28     128.27
181          11931.99    7420.27     319.58     134.80
182          10796.93    7294.44     290.27     132.61

[183 rows x 4 columns]

>>> fast_ma = vbt.MA.from_params(mult_comb_price,
...     [10, 20], name='fast', hide_params=['ewm'])
>>> slow_ma = vbt.MA.from_params(mult_comb_price,
...     [30, 30], name='slow', hide_params=['ewm'])

>>> entries = fast_ma.ma_above(slow_ma, crossed=True)
>>> exits = fast_ma.ma_below(slow_ma, crossed=True)

>>> portfolio = vbt.Portfolio.from_signals(
...     mult_comb_price.vbt.tile(2), entries, exits, freq='1D')
>>> print(portfolio.total_return)
fast_window  slow_window  asset  start_date  end_date
10           30           BTC    2018-12-31  2019-07-01    1.631617
                                 2019-07-02  2019-12-31   -0.281432
                          ETH    2018-12-31  2019-07-01    0.941945
                                 2019-07-02  2019-12-31   -0.306689
20           30           BTC    2018-12-31  2019-07-01    1.725547
                                 2019-07-02  2019-12-31   -0.417770
                          ETH    2018-12-31  2019-07-01    0.336136
                                 2019-07-02  2019-12-31   -0.257854
dtype: float64
```

Notice how index is no more datetime-like, since it captures multiple time periods.
That's why it's required here to pass the frequency `freq` to the `vectorbt.portfolio.main.Portfolio`
class methods in order to be able to compute performance metrics such as Sharpe ratio.

The index hierarchy of the final performance series can be then used to group performance
by any feature, such as window pair, asset, and time period.

```python-repl
>>> mean_return = portfolio.total_return.groupby(['end_date', 'asset']).mean()
>>> mean_return = mean_return.unstack(level=-1).vbt.Bar(
...     xaxis_title='End date',
...     yaxis_title='Mean total return',
...     legend_title_text='Asset').show_png()
```

![](/vectorbt/docs/img/index_by_any.png)

There is much more to backtesting than simply stacking columns. vectorbt offers functions for
most parts of a common backtesting pipeline, from building indicators and generating signals, to
modeling portfolio performance and visualizing results.

## Package structure

The package consists of a series of packages and modules each playing its role in the backtesting pipeline.

### accessors

An accessor adds additional “namespace” to pandas objects.

The `vectorbt.accessors` registers a custom `vbt` accessor on top of each `pandas.Series` and
`pandas.DataFrame` object. It is the main entry point for all other accessors:

```plaintext
vbt.tseries.accessors.TimeSeries_SR/DFAccessor  -> pd.Series/DataFrame.vbt.tseries
vbt.tseries.accessors.OHLCV_DFAccessor          -> pd.DataFrame.vbt.ohlcv
vbt.signals.accessors.Signals_SR/DFAccessor        -> pd.Series/DataFrame.vbt.signals
vbt.returns.accessors.Returns_SR/DFAccessor        -> pd.Series/DataFrame.vbt.returns
vbt.widgets.accessors.Bar_Accessor                 -> pd.Series/DataFrame.vbt.Bar
vbt.widgets.accessors.Scatter_Accessor             -> pd.Series/DataFrame.vbt.Scatter
vbt.widgets.accessors.Histogram_Accessor           -> pd.Series/DataFrame.vbt.Histogram
vbt.widgets.accessors.Heatmap_Accessor             -> pd.Series/DataFrame.vbt.Heatmap
vbt.utils.accessors.Base_SR/DFAccessor             -> pd.Series/DataFrame.vbt.*
```

Additionally, some accessors subclass other accessors building the following inheritance hiearchy:

```plaintext
vbt.utils.accessors.Base_SR/DFAccessor
    -> vbt.tseries.accessors.TimeSeries_SR/DFAccessor
        -> vbt.tseries.accessors.OHLCV_DFAccessor
        -> vbt.signals.accessors.Signals_SR/DFAccessor
        -> vbt.returns.accessors.Returns_SR/DFAccessor
vbt.widgets.accessors.Bar_Accessor
vbt.widgets.accessors.Scatter_Accessor
vbt.widgets.accessors.Histogram_Accessor
vbt.widgets.accessors.Heatmap_Accessor
```

So, for example, the method `pd.Series.vbt.to_2d_array` is also available as `pd.Series.vbt.returns.to_2d_array`.

### utils

`vectorbt.utils` provides an extensive collection of utilities, such as pandas broadcasting.
If any of the functions can take a pandas object as input, it will be offered as an accessor method.
For example, `vectorbt.utils.reshape_fns.broadcast` can take an arbitrary number of pandas
objects, thus you can find its variations in `vectorbt.utils.accessors.Base_Accessor` class.
`vectorbt.utils.accessors` has utility classes that are subclassed by `vbt`, `vbt.tseries` and
`vbt.signals` accessors.

```python-repl
>>> sr = pd.Series([1])
>>> df = pd.DataFrame([1, 2, 3])

>>> vbt.utils.reshape_fns.broadcast_to(sr, df)
   0
0  1
1  1
2  1
>>> sr.vbt.broadcast_to(df)
   0
0  1
1  1
2  1
```

### indicators

`vectorbt.indicators` provides a collection of common technical trading indicators such as Bollinger Bands,
but also a factory class `vectorbt.indicators.factory.IndicatorFactory` to create new indicators of any
complexity and with ease. It doesn't provide any pandas accessors. You can access all the indicators either
by `vbt.*` or `vbt.indicators.*`.

```python-repl
>>> import pandas as pd
>>> import vectorbt as vbt

>>> # vectorbt.indicators.basic.MA
>>> vbt.MA.from_params(pd.Series([1, 2, 3]), [2, 3]).ma
ma_window     2     3
ma_ewm    False False
0           NaN   NaN
1           1.5   NaN
2           2.5   2.0
```

### tseries

`vectorbt.tseries` provides accessors and Numba-compiled functions for working with any
time series data, such as price series.

You can access methods listed in `vectorbt.tseries.accessors` as follows:

* `vectorbt.tseries.accessors.TimeSeries_SRAccessor` -> `pandas.Series.vbt.tseries.*`
* `vectorbt.tseries.accessors.TimeSeries_DFAccessor` -> `pandas.DataFrame.vbt.tseries.*`

```python-repl
>>> # vectorbt.tseries.accessors.TimeSeries_Accessor.rolling_mean
>>> pd.Series([1, 2, 3, 4]).vbt.tseries.rolling_mean(2)
0    NaN
1    1.5
2    2.5
3    3.5
dtype: float64
```

`vectorbt.tseries.nb` provides a range of Numba-compiled functions that are used by accessors.
These only accept NumPy arrays and other Numba-compatible types.

```python-repl
>>> # vectorbt.tseries.nb.rolling_mean_1d_nb
>>> vbt.tseries.nb.rolling_mean_1d_nb(np.asarray([1, 2, 3, 4]), 2)
array([nan, 1.5, 2.5, 3.5])
```

### signals

`vectorbt.signals` provides accessors and Numba-compiled functions for working with signals,
such as entry and exit signals. Since signals are a special case of time series, their accessors
extend the time series accessors.

```python-repl
>>> # vectorbt.signals.accessors.Signals_Accessor.rank
>>> pd.Series([False, True, True, True, False]).vbt.signals.rank()
0    0
1    1
2    2
3    3
4    0
dtype: int64
```

### returns

`vectorbt.returns` provides accessors and Numba-compiled functions for working with returns.
It provides common financial risk and performance metrics. Since returns are a special case of time series,
their accessors extend the time series accessors.

```python-repl
>>> # vectorbt.returns.accessors.Returns_Accessor.total
>>> pd.Series([0.2, 0.1, 0, -0.1, -0.2]).vbt.returns.total()
-0.049599999999999866

>>> # inherited from TimeSeries_Accessor
>>> pd.Series([0.2, 0.1, 0, -0.1, -0.2]).vbt.returns.max()
0.2
```

### records

`vectorbt.records` provides a collection of classes for working with event data, such as trades and
positions. They wrap [NumPy's structured arrays](https://numpy.org/doc/stable/user/basics.rec.html),
and offer properties and methods for analyzing them. They are instantiated as properties within the
`vectorbt.portfolio.main.Portfolio` class, such as `vectorbt.portfolio.main.Portfolio.trades`, or can
be used directly for custom analysis. They don't provide any pandas accessors.

```python-repl
>>> drawdowns = vbt.records.Drawdowns.from_ts(pd.Series([1, 2, 1, 2, 3, 2, 1, 2]))
>>> drawdowns.records
   col  idx  start_idx  valley_idx  end_idx  status
0    0    3          1           2        3       1
1    0    7          4           6        7       0

>>> drawdowns.max_drawdown
-0.6666666666666666
```

### portfolio

`vectorbt.portfolio` provides the class `vectorbt.portfolio.main.Portfolio` for modeling portfolio
performance and calculating various risk and performance metrics. It uses Numba-compiled
functions from `vectorbt.portfolio.nb` for most computations and `vectorbt.portfolio.records`
for tracking events such as orders, trades and positions. It doesn't provide any pandas accessors.
You can access the class directly by `vbt.Portfolio`.

```python-repl
>>> price = pd.Series([1, 2, 3, 4])
>>> entries = pd.Series([True, False, True, False])
>>> exits = pd.Series([False, True, False, True])

>>> # vectorbt.portfolio.main.Portfolio
>>> vbt.Portfolio.from_signals(price, entries, exits, freq='1D').equity
0    100.000000
1    200.000000
2    200.000000
3    266.666667
dtype: float64
```

### widgets

`vectorbt.widgets` provides widgets for visualizing data in an efficient and convenient way.
You can access basic widgets listed in `vectorbt.widgets.basic` as `pandas.Series.vbt.*` and `pandas.DataFrame.vbt.*`.

```python-repl
>>> # vectorbt.widgets.accessors.Histogram_Accessor
>>> pd.Series(np.random.normal(size=100000)).vbt.Histogram()
```

![](/vectorbt/docs/img/hist_normal.png)

### defaults

`vectorbt.defaults` contains default parameters for vectorbt.

For example, you can change default width and height of each plot:
```python-repl
>>> vbt.defaults.layout['width'] = 800
>>> vbt.defaults.layout['height'] = 400
```

Changes take effect immediately.
"""

from vectorbt import (
    indicators,
    portfolio,
    returns,
    signals,
    tseries,
    utils,
    widgets,
    accessors,
    defaults
)

# Most important classes
from vectorbt.widgets import (
    DefaultFigureWidget,
    Indicator,
    Bar,
    Scatter,
    Histogram,
    Heatmap
)
from vectorbt.portfolio import Portfolio
from vectorbt.indicators import (
    IndicatorFactory,
    MA,
    MSTD,
    BollingerBands,
    RSI,
    Stochastic,
    MACD,
    OBV,
    ATR
)

# silence NumbaExperimentalFeatureWarning
import warnings
from numba.core.errors import NumbaExperimentalFeatureWarning

warnings.filterwarnings("ignore", category=NumbaExperimentalFeatureWarning)