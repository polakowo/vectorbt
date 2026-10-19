---
icon: lucide/gem
title: Features
---

# Features

Use these examples to build a research workflow: prepare labeled data, calculate
indicators, generate signals, simulate portfolios, and inspect the results.
Most core examples use small, fixed inputs so you can examine the intermediate results
before applying the same operations to larger datasets.

Install the [core package](installation.md#with-pip) for the local-data examples.
Provider downloads, external indicator libraries, and notifications have optional
dependencies; their sections describe the additional setup. The
[Usage](usage.md) page shows longer market-data examples and interactive demos.

## Start with a strategy comparison

Which moving-average pair works on which asset? Compute six window pairs for two
assets, then backtest all twelve combinations in one call. The synthetic prices
below make this example reproducible without downloads.

```python title="Compare twelve strategy and asset combinations"
import numpy as np
import pandas as pd
import vectorbt as vbt

rng = np.random.default_rng(42)
close = pd.DataFrame(
    100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, size=(180, 2)), axis=0)),
    index=pd.date_range("2024-01-01", periods=180, freq="D"),
    columns=pd.Index(["Asset A", "Asset B"], name="asset"),
)
fast, slow = vbt.MA.run_combs(
    close, window=[3, 5, 10, 20], r=2, short_names=["fast", "slow"]
)
portfolio = vbt.Portfolio.from_signals(
    close,
    fast.ma_crossed_above(slow).shift(1, fill_value=False),
    fast.ma_crossed_below(slow).shift(1, fill_value=False),
    init_cash=1000, fees=0.001, freq="1D",
)
returns = (portfolio.total_return().unstack("asset") * 100).round(2)
print(returns)
```

Output:

```text
asset                    Asset A  Asset B
fast_window slow_window
3           5               9.87   -11.49
            10              9.29    -8.22
            20              2.84    -6.29
5           10              6.51    -2.39
            20              3.89    -3.24
10          20              3.97    -8.08
```

The same 3/5-day pair gains 9.87% on one path and loses 11.49% on the other.
The parameter labels make that comparison explicit. Each strategy-asset pair has
its own 1,000 cash units; returns include 0.1% order fees. Signals observed at one
close execute at the following close. These are in-sample results on synthetic
data, intended to explain the workflow.

Changing the window list expands the strategy grid without adding simulation
loops. Continue with [vectorized backtesting](#fast-vectorized-backtesting) for
order sizing, next-open execution, and shared cash, or
[robustness testing](#robustness-testing) to evaluate selected parameters on later data.

## Pandas

### Pandas-native API

Importing VectorBT registers the `.vbt` accessor on pandas Series and DataFrames.
You keep the dates, asset names, and labeled results while using compiled operations
for rolling calculations, reductions, grouping, and transformations. Use ordinary
pandas methods to clean and align data first; an accessor does not infer missing
trading sessions or decide how missing prices should be filled.

```python title="Align two assets and calculate rolling means"
import pandas as pd
import vectorbt as vbt

index = pd.date_range("2024-01-01", periods=5, freq="D")
asset_a = pd.Series([100., 102., 101., 104., 106.], index=index)
asset_b = pd.Series([50., 51., 53., 54.], index=index.delete(2))
close = pd.concat({"A": asset_a, "B": asset_b}, axis=1)

mean = close.vbt.rolling_mean(window=3, minp=3, engine="numba")
print(mean.round(2))
```

Output:

```text
                 A   B
2024-01-01     NaN NaN
2024-01-02     NaN NaN
2024-01-03  101.00 NaN
2024-01-04  102.33 NaN
2024-01-05  103.67 NaN
```

`concat` aligns observations by date, leaving a missing value for B on the third
day. Requiring three valid observations keeps B's affected windows empty. Fill
missing values only when that matches your data policy; backward filling prices
would introduce information from later observations.

Custom rolling functions receive the row number, column number, and window.
Numba callbacks can express calculations that are not built in. Here the result
compares the latest value with the window mean and population standard deviation;
a constant window is assigned zero rather than dividing by zero.

```python title="Apply a custom rolling z-score"
import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit

@njit
def zscore_nb(i, col, window):
    std = np.std(window)
    if std == 0.:
        return 0.
    return (window[-1] - np.mean(window)) / std

price = pd.Series([10., 10., 10., 12., 14.], name="price")
zscore = price.vbt.rolling_apply(3, zscore_nb, engine="numba")
print(zscore.round(4))
```

Output:

```text
0       NaN
1       NaN
2    0.0000
3    1.4142
4    1.2247
Name: price, dtype: float64
```

To compare implementations on your own machine, use the same input and warm both
paths before timing them. This bounded benchmark uses ten columns; increase the
column count to 1,000 to benchmark a million-value workload. The
reported times depend on the hardware and package versions, and exclude the first
compilation. The equality check compares the actual calculated values.

```python title="Compare pandas and VectorBT rolling z-scores"
from timeit import repeat
import numpy as np
import pandas as pd
from numba import njit
import vectorbt as vbt

@njit
def window_zscore_nb(window):
    std = np.std(window)
    if std == 0.:
        return 0.
    return (window[-1] - np.mean(window)) / std

@njit
def rolling_zscore_nb(i, col, window):
    return window_zscore_nb(window)

rng = np.random.default_rng(42)
values = pd.DataFrame(rng.uniform(size=(1000, 10)))

def pandas_zscore():
    return values.rolling(2).apply(window_zscore_nb, raw=True)

def vectorbt_zscore():
    return values.vbt.rolling_apply(2, rolling_zscore_nb, engine="numba")

pd.testing.assert_frame_equal(pandas_zscore(), vectorbt_zscore())
for name, calculate in [("pandas", pandas_zscore), ("vectorbt", vectorbt_zscore)]:
    seconds = min(repeat(calculate, repeat=3, number=1))
    print(f"{name}: {seconds:.6f} seconds (best of 3 warmed runs)")
```

Output:

```text
pandas: 0.001950 seconds (best of 3 warmed runs)
vectorbt: 0.000636 seconds (best of 3 warmed runs)
```

Sample output; timings vary by machine.

Other accessor utilities work on labels and shapes rather than prices. For example,
`pd.Series([1, 2, 3]).vbt.make_symmetric()` expands a packed triangular Series
into a symmetric matrix. See the [generic accessor API](../api/generic/accessors.md)
and [base accessor API](../api/base/accessors.md) for method signatures and wrapping options.

The Series index supplies the row and column labels; mirrored off-diagonal values
are filled, while combinations absent from the input remain missing.

```python title="Build a symmetric matrix"
import pandas as pd
import vectorbt as vbt

matrix = pd.Series([1, 2, 3]).vbt.make_symmetric()
print(matrix)
```

Output:

```text
     0    1    2
0  1.0  2.0  3.0
1  2.0  NaN  NaN
2  3.0  NaN  NaN
```

### Flexible broadcasting

Broadcasting lets one date-indexed Series apply to every asset column, or a scalar
setting apply to every simulation. This differs from pandas' default DataFrame +
Series arithmetic, which normally matches the Series index against column labels.
Make the intended orientation explicit and inspect labels before launching a large sweep.

```python title="Broadcast a daily adjustment across assets"
import pandas as pd
import vectorbt as vbt

index = pd.date_range("2024-01-01", periods=3, freq="D")
close = pd.DataFrame({"A": [100., 101., 102.], "B": [50., 51., 52.]}, index=index)
adjustment = pd.Series([1., 2., 3.], index=index)
adjusted = adjustment.vbt + close
print(adjusted)

# Two windows applied to two assets produce four labeled result columns.
ma = vbt.MA.run(close, window=[2, 3]).ma
print(ma)
```

Output:

```text
                A     B
2024-01-01  101.0  51.0
2024-01-02  103.0  53.0
2024-01-03  105.0  55.0
ma_window       2            3
                A     B      A     B
2024-01-01    NaN   NaN    NaN   NaN
2024-01-02  100.5  50.5    NaN   NaN
2024-01-03  101.5  51.5  101.0  51.0
```

The window level distinguishes parameter values from asset names. With multiple
parameter lists, indicator factories normally pair values by position; use
`param_product=True` when you actually want their Cartesian product. Broadcasting
compatible inputs is different from searching parameter combinations. Large grids
still produce large outputs, so start with a small selection and check its shape.
See the [broadcasting API](../api/base/reshape_fns.md#vectorbt.base.reshape_fns.broadcast)
and [indicator factory](../api/indicators/factory.md).

### Optional Rust engine

The base installation includes Numba. Install the compatible optional extension
with `pip install "vectorbt[rust]"` to enable precompiled Rust kernels for supported
operations. Engine selection happens per operation, not for the entire workflow:
pandas alignment and output wrapping still happen in Python.

- `engine="numba"` explicitly uses Numba, including supported custom Numba callbacks.
- `engine="auto"` uses Rust when the extension is compatible and the particular call
  is supported; otherwise it uses Numba. Randomized operations default to Numba
  under `"auto"` to preserve their established random streams.
- `engine="rust"` requests Rust explicitly. A missing or incompatible extension raises
  `ImportError`; an unsupported call raises `ValueError` rather than silently switching engines.

```python title="Choose an engine without changing the calculation"
import pandas as pd
import vectorbt as vbt

price = pd.Series([100., 101., 103., 102.])
numba_mean = price.vbt.rolling_mean(2, engine="numba")
auto_mean = price.vbt.rolling_mean(2, engine="auto")
pd.testing.assert_series_equal(auto_mean, numba_mean)
print(pd.DataFrame({"numba": numba_mean, "auto": auto_mean}))

# Optional: run only after installing the Rust extension.
# rust_mean = price.vbt.rolling_mean(2, engine="rust")
# pd.testing.assert_series_equal(rust_mean, numba_mean)
```

Output:

```text
   numba   auto
0    NaN    NaN
1  100.5  100.5
2  102.0  102.0
3  102.5  102.5
```

Set `vbt.settings["engine"] = "numba"` or `"rust"` to change the process-wide
default, and restore the previous value after an experiment. Callback-based
operations, including the custom z-score above, require Numba. Consult the
[installation instructions](installation.md) and the relevant method's API;
Rust support does not imply support for every option or input dtype. Compare
engines with representative data and include conversion and compilation costs
when evaluating your own workload.

The following optional benchmark simulates 1,000 bars for 1,000 independent
columns. Alternating entries and exits produce one million filled orders. It
allocates large arrays and records; reduce `n_columns` for a smaller smoke test.
Each backend is warmed before measuring three single simulations. These timings
exclude initial compilation and do not establish a general speed advantage.

```python title="Optional: process one million orders"
from timeit import repeat
import numpy as np
import pandas as pd
import vectorbt as vbt

n_bars, n_columns = 1000, 1000
rng = np.random.default_rng(42)
close = pd.DataFrame(100. + rng.uniform(size=(n_bars, n_columns)))
entries = pd.DataFrame(False, index=close.index, columns=close.columns)
exits = entries.copy()
entries.iloc[::2] = True
exits.iloc[1::2] = True

engines = ["numba"]
# After installing a matching vectorbt[rust] extension, add "rust" to engines.
for engine in engines:
    def simulate():
        return vbt.Portfolio.from_signals(
            close, entries, exits, size=1., init_cash=1000., fees=0.,
            engine=engine,
        )

    warmup = simulate()
    print(engine, "filled orders:", int(warmup.orders.count().sum()))
    del warmup
    seconds = min(repeat(simulate, repeat=3, number=1))
    print(f"{engine}: {seconds:.6f} seconds (best of 3 warmed runs)")
```

Output:

```text
numba filled orders: 1000000
numba: 0.058820 seconds (best of 3 warmed runs)
```

Sample output; timings vary by machine.

!!! tip "VectorBT PRO"

    [Array expressions](https://vectorbt.pro/features/tooling/time-series-operations/#array-expressions) align and
    broadcast arrays before evaluating a formula. [Index dictionaries](https://vectorbt.pro/features/tooling/multidimensional-research/#index-dictionaries)
    let you assign values by date and asset without constructing full arrays yourself.
    Supported calculations can also [process columns in parallel](https://vectorbt.pro/features/performance/compute-backends/#parallel-numba).

## Data

### Built-in data access

Data objects combine one or more symbols, align their indexes and columns, and
expose fields through `.get()`. You can start with local pandas objects without
installing a provider or making a network request. This is also useful for testing
strategies against a fixed input snapshot.

```python title="Wrap a local OHLCV snapshot"
from io import StringIO
import pandas as pd
import vectorbt as vbt

csv = StringIO("""Date,Open,High,Low,Close,Volume
2024-01-01,100,103,99,102,1000
2024-01-02,102,104,101,103,1200
2024-01-03,103,105,100,101,1100
""")
prices = pd.read_csv(csv, index_col="Date", parse_dates=True)
data = vbt.Data.from_data({"DEMO": prices}, tz_localize="UTC", download_kwargs={})
close = data.get("Close")
print(close)
```

Output:

```text
Date
2024-01-01 00:00:00+00:00    102
2024-01-02 00:00:00+00:00    103
2024-01-03 00:00:00+00:00    101
Freq: D, Name: Close, dtype: int64
```

Replace `StringIO` with a CSV path for real local files. The empty
`download_kwargs` dictionary records that this snapshot has no provider request settings. With multiple symbols,
`missing_index="nan"` retains the union of timestamps and introduces missing
values, while `missing_index="drop"` keeps shared timestamps. Neither policy
checks market calendars or corporate actions. Validate sorting, duplicates,
time zones, adjusted prices, and missing observations before calculating signals.
See [Data.from_data](../api/data/base.md#vectorbt.data.base.Data.from_data)
and [Data.get](../api/data/base.md#vectorbt.data.base.Data.get).

Synthetic generators provide reproducible fixtures for exercising the same data
pipeline. A fixed seed and fixed date range avoid changes between runs.

```python title="Generate reproducible GBM paths"
import pandas as pd
import vectorbt as vbt

kwargs = dict(
    start="2024-01-01 UTC", end="2024-01-10 UTC", freq="D",
    seed=vbt.symbol_dict({"A": 42, "B": 43}),
)
gbm_data = vbt.GBMData.download(["A", "B"], **kwargs)
paths = gbm_data.get()
print(paths.head(3).round(3))
fig = gbm_data.plot(showlegend=False)
```

Output:

```text
symbol                           A        B
2024-01-01 00:00:00+00:00  102.387  101.169
2024-01-02 00:00:00+00:00  101.554   96.555
2024-01-03 00:00:00+00:00  104.766   94.627
```

The symbol-specific seeds give A and B distinct reproducible paths. GBM is a
model for testing code, not evidence of how a real strategy performs.
Call `fig.show()` in a notebook or browser-enabled session to explore the result.

<div class="feature-embed">
  <iframe
    class="feature-embed__iframe"
    src="/assets/interactive/features_gbm_data.html"
    title="GBM data chart"
    loading="lazy"
    scrolling="no">
  </iframe>
</div>

For remote data, install the provider dependency you need. For example, install
`yfinance`, then run the following optional network example. Provider responses
can change, and access is subject to the provider's availability and terms.

```python title="Optional: download Yahoo Finance daily bars"
import vectorbt as vbt

# Requires yfinance and network access.
market = vbt.YFData.download(
    ["AAPL", "MSFT"], start="2024-01-01", end="2024-02-01",
    interval="1d", auto_adjust=True, missing_index="nan",
)
close = market.get("Close")
print(close.head())
```

Output:

```text
Illustrative schema (requires a configured provider; values omitted):

Date          AAPL             MSFT
<timestamp>   <adjusted close> <adjusted close>
```

Alpaca requires the optional `alpaca-py` package, API credentials, and access to
the selected data feed. Keep secrets in environment variables. This optional
example requests a fixed historical minute-bar range; returned rows and available
history depend on your account and the provider. The feed and time range are
explicit rather than relying on the reader's current time.

```python title="Optional: download Alpaca minute bars"
import os
import vectorbt as vbt

alpaca_data = vbt.AlpacaData.download(
    "AAPL",
    start="2024-01-02 14:30:00 UTC",
    end="2024-01-02 15:00:00 UTC",
    timeframe="1m",
    feed="iex",
    api_key=os.environ["APCA_API_KEY_ID"],
    secret_key=os.environ["APCA_API_SECRET_KEY"],
)
print(alpaca_data.get().head())
```

Output:

```text
Illustrative schema (requires a configured provider; values omitted):

timestamp      Open    High    Low    Close    Volume
<timestamp>    <price> <price> <price> <price>  <volume>
```

[BinanceData](../api/data/custom.md#vectorbt.data.custom.BinanceData),
[CCXTData](../api/data/custom.md#vectorbt.data.custom.CCXTData), and
[AlpacaData](../api/data/custom.md#vectorbt.data.custom.AlpacaData) have their own
client dependencies and credential requirements. Keep credentials outside notebooks.
For downloaded data, `data.update()` returns an updated data object; assign its
result to retain changes. A plain local `Data` snapshot has no provider implementation
to call when updating.
[DataUpdater](../api/data/updater.md) can schedule repeated updates, but does not
turn a provider into a local streaming feed.

Subclass `DataUpdater` to run validation or reporting after each update. Here a
small GBM subclass advances its end timestamp by exactly five seconds on every
call, so the demonstration does not depend on waiting for the wall clock. The
updater counts completed updates and raises `vbt.CancelledError` at its limit.

```python title="Append synthetic observations with a DataUpdater subclass"
import pandas as pd
import vectorbt as vbt

class FiveSecondGBM(vbt.GBMData):
    def update_symbol(self, symbol, **kwargs):
        end = self.data[symbol].index[-1] + pd.Timedelta(seconds=5)
        return super().update_symbol(symbol, end=end, **kwargs)

class MyDataUpdater(vbt.DataUpdater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_count = 0

    def update(self, count_limit=None, **kwargs):
        previous_rows = len(self.data.wrapper.index)
        super().update(**kwargs)
        added = len(self.data.wrapper.index) - previous_rows
        self.update_count += 1
        print(f"Data updated with {added} data points")
        if count_limit is not None and self.update_count >= count_limit:
            raise vbt.CancelledError

data = FiveSecondGBM.download(
    "SYMBOL", start="2024-01-01 00:00:00 UTC",
    end="2024-01-01 00:00:04 UTC", freq="1s", seed=42,
)
my_updater = MyDataUpdater(data)
try:
    my_updater.update(count_limit=2)
    my_updater.update(count_limit=2)
except vbt.CancelledError:
    pass
print(my_updater.data.get().tail(3).round(3))
```

Output:

```text
Data updated with 5 data points
Data updated with 5 data points
2024-01-01 00:00:12+00:00    102.674
2024-01-01 00:00:13+00:00     99.703
2024-01-01 00:00:14+00:00     94.661
Freq: s, dtype: float64
```

To run the same updates on a real schedule, continue the preceding setup with the
following optional block. It starts a foreground loop, executes twice at five-second
intervals, then exits through `CancelledError`. Run it deliberately in a script or
notebook cell that may remain busy for about ten seconds. It does not fetch live
market data; replace the synthetic source with a configured provider for that.

```python title="Optional: append random data every 5 seconds"
# Continue the DataUpdater subclass example above; start a fresh updater.
my_updater = MyDataUpdater(data)
my_updater.update_every(5, "seconds", count_limit=2)
```

Output:

```text
Expected output (two scheduled updates):
Data updated with 5 data points
Data updated with 5 data points
```

### Robustness testing

Time-ordered validation asks whether a parameter selected on past observations
continues to work on later observations. VectorBT's range, rolling, and expanding
splitters define contiguous subsets and return their original indexes. They do
not fit a model or prevent leakage automatically.

```python title="Inspect rolling train, validation, and test ranges"
import numpy as np
import pandas as pd
import vectorbt as vbt

price = pd.Series(np.arange(10.), index=pd.date_range("2024-01-01", periods=10))
train, valid, test = price.vbt.rolling_split(
    window_len=5, set_lens=(1, 1), left_to_right=False,
)
print(train[0].iloc[:, :2])
first_split = pd.DataFrame({
    "set": ["train", "valid", "test"],
    "start": [train[1][0][0], valid[1][0][0], test[1][0][0]],
    "end": [train[1][0][-1], valid[1][0][-1], test[1][0][-1]],
})
print(first_split.to_string(index=False))
assert train[1][0][-1] < valid[1][0][0] < test[1][0][0]
fig = price.vbt.rolling_split(
    window_len=5, set_lens=(1, 1), left_to_right=False,
    plot=True, trace_names=["train", "valid", "test"],
)
```

Output:

```text
split_idx    0    1
0          0.0  1.0
1          1.0  2.0
2          2.0  3.0
  set      start        end
train 2024-01-01 2024-01-03
valid 2024-01-04 2024-01-04
 test 2024-01-05 2024-01-05
```

<div class="feature-embed">
  <iframe
    class="feature-embed__iframe feature-embed__iframe--rolling-split"
    src="/assets/interactive/features_rolling_split.html"
    title="Rolling split chart"
    loading="lazy"
    scrolling="no">
  </iframe>
</div>

The next example selects a moving-average window on 80 training bars, freezes it,
and evaluates the following 20 bars. Each test portfolio starts with cash; this
is a sequence of separate experiments, not a continuously rebalanced portfolio.
Indicators retain historical warm-up data, but their values at a given date use
only observations available by that date. Signals are shifted one bar so a close
observed today cannot trigger a fill at that same close.

```python title="Select a window on training data and evaluate the next block"
import numpy as np
import pandas as pd
import vectorbt as vbt

rng = np.random.default_rng(42)
close = pd.Series(
    100. * np.exp(np.cumsum(rng.normal(0.001, 0.01, 160))),
    index=pd.date_range("2024-01-01", periods=160, freq="D"),
)
windows = [5, 10, 20]
rows = []
for test_start in range(80, len(close), 20):
    train_start = test_start - 80
    test_end = min(test_start + 20, len(close))
    history = close.iloc[:test_end]
    positions = {}
    for window in windows:
        mean = history.vbt.rolling_mean(window, engine="numba")
        positions[window] = (history > mean).shift(1, fill_value=False)

    def simulate(window, start, end):
        held = positions[window].iloc[start:end]
        return vbt.Portfolio.from_signals(
            close.iloc[start:end], entries=held, exits=~held,
            init_cash=1000., fees=0.001, freq="D", engine="numba",
        )

    train_returns = {
        window: simulate(window, train_start, test_start).total_return()
        for window in windows
    }
    best_window = max(train_returns, key=train_returns.get)
    test_return = simulate(best_window, test_start, test_end).total_return()
    rows.append((close.index[test_start], best_window, test_return))

results = pd.DataFrame(rows, columns=["test_start", "window", "test_return"])
print(results.round({"test_return": 4}).to_string(index=False))
```

Output:

```text
test_start  window  test_return
2024-03-21      10      -0.0567
2024-04-10       5      -0.0257
2024-04-30       5       0.0335
2024-05-20       5      -0.0585
```

Report every test block, including losses, rather than choosing the best test
period. This small synthetic exercise illustrates the procedure; it does not
establish robustness. In real research, preserve a final untouched holdout,
include execution costs and realistic timing, and fit scalers or other learned
transformations using training data only. For forward-looking targets, leave a
gap large enough to prevent training labels from reaching into the test period.
See the [splitter API](../api/generic/splitters.md) and
[rolling_split](../api/generic/accessors.md#vectorbt.generic.accessors.GenericAccessor.rolling_split).

Label generators can provide retrospective training targets. They are not all
usable as live signals: local extrema require future observations to confirm a
turning point. For example, the following plot is an analysis of an already
observed path, not a strategy that knew each extremum at the time.

```python title="Inspect retrospective local-extrema labels"
import numpy as np
import pandas as pd
import vectorbt as vbt

rng = np.random.default_rng(42)
price = np.cumprod(1. + rng.uniform(-0.1, 0.1, size=100))
labels = vbt.LEXLB.run(price, 0.2, 0.2)
fig = labels.plot()
summary = pd.Series({
    "Local maxima": int((labels.labels == 1).sum()),
    "Local minima": int((labels.labels == -1).sum()),
})
print(summary)
```

Output:

```text
Local maxima    3
Local minima    4
dtype: int64
```

<div class="feature-embed">
  <iframe
    class="feature-embed__iframe"
    src="/assets/interactive/features_local_extrema.html"
    title="Local extrema chart"
    loading="lazy"
    scrolling="no">
  </iframe>
</div>

!!! tip "VectorBT PRO"

    Read market data from [Databento](https://vectorbt.pro/features/data/market-data-sources/#databento),
    load local [CSV and HDF5 files](https://vectorbt.pro/features/data/data-storage-and-databases/#local-data), or run
    [SQL queries](https://vectorbt.pro/features/data/data-storage-and-databases/#sql-queries) on data objects.
    [Data caching](https://vectorbt.pro/features/data/data-storage-and-databases/#data-caching) saves fetched data to disk for reuse.

## Indicators

### Rich indicator ecosystem

Built-in indicators include moving averages, Bollinger Bands, RSI, Stochastic, and MACD.
Their outputs retain the input index and attach parameter values to column labels, so results
can be selected without remembering column positions. Built-in calculations use compiled
kernels; supported calls can also use the optional Rust engine.

```python title="Compare moving-average parameters"
import pandas as pd
import vectorbt as vbt

close = pd.Series(
    [100., 102., 101., 104., 103., 106., 105., 108.],
    index=pd.date_range("2024-01-01", periods=8, freq="D"),
    name="DEMO",
)
ma = vbt.MA.run(
    close, window=[2, 3], ewm=[False, True], param_product=True
)
print(ma.ma.round(2).to_string())
print("Three-bar SMA on the final day:", round(ma.ma[(3, False)].iloc[-1], 2))
```

Output:

```text
ma_window       2               3
ma_ewm      False   True    False   True
2024-01-01    NaN     NaN     NaN     NaN
2024-01-02  101.0  101.33     NaN     NaN
2024-01-03  101.5  101.11  101.00  101.00
2024-01-04  102.5  103.04  102.33  102.50
2024-01-05  103.5  103.01  102.67  102.75
2024-01-06  104.5  105.00  104.33  104.38
2024-01-07  105.5  105.00  104.67  104.69
2024-01-08  106.5  107.00  106.33  106.34
Three-bar SMA on the final day: 106.33
```

`param_product=True` evaluates the Cartesian product: two windows times two averaging modes
produce four columns. Without it, parameter lists are paired element by element. Simple
moving averages need a full window: the first `window - 1` observations are `NaN`, and must
not be treated as valid signals. Inspect each indicator's warmup requirements rather than
filling missing values with information from later bars.

Optional integrations wrap installed [TA-Lib](https://github.com/mrjbq7/ta-lib),
[Pandas TA](https://github.com/twopirllc/pandas-ta), and
[Technical Analysis Library](https://github.com/bukosabino/ta) indicators with the same
parameter and indexing machinery. For example, `vbt.talib("SMA")`,
`vbt.pandas_ta("SMA")`, and `vbt.ta("SMAIndicator")` build wrappers for the corresponding
installed libraries. Available names, outputs, and dependencies vary by library and version.
See the [indicator factory API](/api/indicators/factory/) for integration details.

The three wrappers below expose the underlying libraries' moving averages through VectorBT.
Each integration requires its corresponding package. The example runs installed libraries
and identifies missing optional packages. Output attribute names belong to the wrapped
implementation; the captured output below includes the installed TA-Lib integration.

```python title="Optional: compute 2 moving averages at once with integrations"
from importlib.util import find_spec
import pandas as pd
import vectorbt as vbt

price = pd.Series([1., 2., 3., 4., 5.])
print("Built-in MA:")
print(vbt.MA.run(price, [2, 3]).ma.to_string())

if find_spec("ta") is not None:
    print("ta:")
    print(vbt.ta("SMAIndicator").run(price, [2, 3]).sma_indicator.to_string())
else:
    print("ta: optional package not installed")

if find_spec("pandas_ta") is not None:
    print("pandas_ta:")
    print(vbt.pandas_ta("SMA").run(price, [2, 3]).sma.to_string())
else:
    print("pandas_ta: optional package not installed")

if find_spec("talib") is not None:
    print("TA-Lib:")
    print(vbt.talib("SMA").run(price, [2, 3]).real.to_string())
else:
    print("TA-Lib: optional package not installed")
```

Output:

```text
Built-in MA:
ma_window    2    3
0          NaN  NaN
1          1.5  NaN
2          2.5  2.0
3          3.5  3.0
4          4.5  4.0
ta: optional package not installed
pandas_ta: optional package not installed
TA-Lib:
sma_timeperiod    2    3
0               NaN  NaN
1               1.5  NaN
2               2.5  2.0
3               3.5  3.0
4               4.5  4.0
```

Use `IndicatorFactory` when a reusable research feature needs named inputs, parameters,
and outputs. Here the output measures how far the price is above or below its trailing mean,
in percentage points. The function accepts a two-dimensional array, matching the factory's
input convention; using pandas inside it keeps this example readable rather than compiled.

```python title="Build a custom price-deviation indicator"
import pandas as pd
import vectorbt as vbt


def deviation(close, window):
    mean = pd.DataFrame(close).rolling(window, min_periods=window).mean().to_numpy()
    return 100 * (close / mean - 1)


Deviation = vbt.IndicatorFactory(
    class_name="Deviation",
    short_name="dev",
    input_names=["close"],
    param_names=["window"],
    output_names=["pct"],
).from_apply_func(deviation)

close = pd.Series([100., 102., 101., 104., 103., 106.], name="DEMO")
result = Deviation.run(close, window=[2, 3])
print(result.pct.round(3).to_string())
```

Output:

```text
dev_window      2      3
0             NaN    NaN
1           0.990    NaN
2          -0.493  0.000
3           1.463  1.629
4          -0.483  0.325
5           1.435  1.597
```

A value of `2` means the price is 2% above its trailing mean, not that the strategy earned
2%. The first observations remain missing, and the calculation uses no future values.
The [factory API](/api/indicators/factory/#vectorbt.indicators.factory.IndicatorFactory)
also covers multiple outputs, caching, and Numba-compatible calculation functions.

Factories can also create outputs without an input price series. With
`require_input_shape=True`, the calculation receives the requested output shape. This seeded
random-path example keeps that workflow separate from the price-deviation indicator above.
The `start` and `mu` lists are paired here rather than expanded as a Cartesian product.

```python title="Construct a random indicator"
import numpy as np
from numba import njit
import vectorbt as vbt


@njit
def apply_func_nb(input_shape, start, mu, sigma):
    rand_returns = np.random.normal(mu, sigma, input_shape)
    return start * vbt.nb.nancumprod_nb(rand_returns + 1)


RandomInd = vbt.IndicatorFactory(
    param_names=["start", "mu", "sigma"],
    output_names=["output"],
).from_apply_func(
    apply_func_nb,
    require_input_shape=True,
    seed=42,
)
print(RandomInd.run(5, [100, 200], [-0.01, 0.01], 0.01).output.round(3).to_string())
```

Output:

```text
custom_start     100      200
custom_mu      -0.01     0.01
custom_sigma    0.01     0.01
0             99.497  201.532
1             98.364  206.730
2             98.018  210.383
3             98.530  211.500
4             97.314  214.762
```

!!! tip "VectorBT PRO"

    Build indicators from [expressions](https://vectorbt.pro/features/indicators/indicator-development/#indicator-expressions),
    calculate [rolling regressions](https://vectorbt.pro/features/indicators/technical-indicators/#rolling-ols), or use
    [WorldQuant's 101 alphas](https://vectorbt.pro/features/indicators/technical-indicators/#worldquant-alphas).
    The [multiple-timeframe tutorial](https://vectorbt.pro/tutorials/mtf-analysis/)
    shows how to combine indicators calculated at different frequencies.

## Signals

### Signal tooling

Signals are Boolean arrays indexed by time, optionally with one column per asset or strategy.
Crossover helpers turn indicator comparisons into discrete events: being above a moving
average for five bars is different from crossing above it once. Signal counts describe the
requested events, not filled orders; position state, cash, and execution rules determine
which events actually trade.

```python title="Generate crossovers and inspect their spacing"
import numpy as np
import pandas as pd
import vectorbt as vbt

index = pd.date_range("2024-01-01", periods=120, freq="D")
t = np.arange(len(index))
close = pd.Series(100 + 0.05 * t + 5 * np.sin(t / 4), index=index, name="DEMO")
fast = vbt.MA.run(close, 5)
slow = vbt.MA.run(close, 20)
entries = fast.ma_crossed_above(slow)
exits = fast.ma_crossed_below(slow)
stats = entries.vbt.signals(freq="1D").stats()
print(stats.loc[["Total", "First Index", "Last Index"]].to_string())
print("Bars between entries:", entries.vbt.signals.between_ranges().duration.values)

mask = pd.Series([True, True, False, True, True, True])
print("Partition lengths [bars]:", mask.vbt.signals.partition_ranges().duration.values)
```

Output:

```text
Total                            4
First Index    2024-01-27 00:00:00
Last Index     2024-04-12 00:00:00
Bars between entries: [26 25 25]
Partition lengths [bars]: [2 3]
```

`between_ranges()` measures intervals between successive signals; `partition_ranges()`
measures consecutive runs of `True`. Raw durations count bars. Passing `freq="1D"` gives
signal statistics a one-day bar duration. These crossovers become known at the close;
shift them forward before simulating execution on the following bar, as shown below.
The [signals accessor API](/api/signals/accessors/) also covers cleaning, ranking,
and mapping signals.

For controlled experiments, random generators can produce repeatable entry/exit sequences.
A seed fixes this fixture; it does not turn random signals into a useful trading strategy.

```python title="Generate a repeatable grid of random signals"
import pandas as pd
import vectorbt as vbt

signals = vbt.RPROBNX.run(
    input_shape=(10,),
    entry_prob=[0.25, 0.5],
    exit_prob=[0.25, 0.5],
    param_product=True,
    seed=42,
)
print(pd.DataFrame({
    "Entries": signals.entries.sum(),
    "Exits": signals.exits.sum(),
}).to_string())
```

Output:

```text
                                      Entries  Exits
rprobnx_entry_prob rprobnx_exit_prob
0.25               0.25                     2      1
                   0.50                     2      2
0.50               0.25                     2      1
                   0.50                     3      2
```

For custom iterative placement, a choice function returns integer row positions inside
`[from_i, to_i)`. The following example enters each column once and exits on the last row.
Factories can also support stop-loss, trailing-stop, and take-profit signal workflows;
see [signal generators](/api/signals/generators/) and the
[signal factory API](/api/signals/factory/).

```python title="Place entries and exits with a signal factory"
import numpy as np
from numba import njit
import vectorbt as vbt


@njit
def entry_choice_func(from_i, to_i, col):
    return np.array([col])


@njit
def exit_choice_func(from_i, to_i, col):
    return np.array([to_i - 1])


MySignals = vbt.SignalFactory().from_choice_func(
    entry_choice_func=entry_choice_func,
    exit_choice_func=exit_choice_func,
    entry_kwargs=dict(wait=1),
    exit_kwargs=dict(wait=0),
)
custom = MySignals.run(input_shape=(3, 3))
print("Entries:")
print(custom.entries.to_string())
print("Exits:")
print(custom.exits.to_string())
```

Output:

```text
Entries:
       0      1      2
0   True  False  False
1  False   True  False
2  False  False   True
Exits:
       0      1      2
0  False  False  False
1  False  False  False
2   True   True   True
```

!!! tip "VectorBT PRO"

    Detect [confirmed price pivots](https://vectorbt.pro/features/indicators/trading-signals/#pivot-detection)
    and [outliers](https://vectorbt.pro/features/indicators/trading-signals/#signal-detection), or
    [backtest each signal individually](https://vectorbt.pro/features/indicators/trading-signals/#signal-unraveling).
    See the [signal development tutorial](https://vectorbt.pro/tutorials/signal-development/)
    for examples of building entry and exit signals.

## Modeling

### Fast, vectorized backtesting

Represent candidate strategies as labeled columns, then simulate them together with
`Portfolio.from_signals`. Use `Portfolio.from_orders` for explicit order arrays, or
`Portfolio.from_order_func` for callbacks whose decisions depend on evolving simulation
state. Supported simulation paths offer optional Rust acceleration; throughput depends
on the data, order count, configuration, and backend.

The following offline example evaluates all three distinct pairs of moving-average windows.
Synthetic daily prices demonstrate the API without downloads or claims of trading performance.
Signals calculated from one close execute at the **next day's synthetic open**: shifting
entries and exits by one row prevents them from using a close that was unknown at execution.
The `close` series still values holdings at each day's end.

```python title="Backtest three labeled strategies with trading costs"
import numpy as np
import pandas as pd
import vectorbt as vbt

index = pd.date_range("2024-01-01", periods=120, freq="D")
t = np.arange(len(index))
close = pd.Series(100 + 0.05 * t + 5 * np.sin(t / 4), index=index, name="DEMO")
open_price = close.shift(1).fillna(close.iloc[0]) * (1 + 0.001 * np.cos(t / 3))

fast, slow = vbt.MA.run_combs(
    close, window=[5, 10, 20], r=2, short_names=["fast", "slow"]
)
entries = fast.ma_crossed_above(slow).shift(1, fill_value=False)
exits = fast.ma_crossed_below(slow).shift(1, fill_value=False)

pf = vbt.Portfolio.from_signals(
    close,
    entries,
    exits,
    price=open_price,
    size=1,
    init_cash=1_000,
    fees=0.001,
    slippage=0.0005,
    cash_sharing=False,
    freq="1D",
    log=True,
)
summary = pd.DataFrame({
    "Total return [%]": pf.total_return() * 100,
    "Profit [cash]": pf.total_profit(),
    "Filled orders": pf.orders.count(),
})
print(summary.round(3).to_string())
selected = pf[(5, 20)]  # Select by (fast_window, slow_window), not position
print("First three fills for (5, 20):")
print(selected.orders.records_readable[
    ["Timestamp", "Size", "Price", "Fees", "Side"]
].head(3).round(3).to_string(index=False))
```

Output:

```text
                         Total return [%]  Profit [cash]  Filled orders
fast_window slow_window
5           10                      1.248         12.483              8
            20                     -1.539        -15.393              8
10          20                     -3.151        -31.515              8
First three fills for (5, 20):
 Timestamp  Size   Price  Fees Side
2024-01-28   1.0 102.333 0.102  Buy
2024-02-11   1.0  99.275 0.099 Sell
2024-02-23   1.0 104.793 0.105  Buy
```

`size=1` requests one asset unit per entry, with long-only signals and no accumulation by
default. Each column starts with its **own** 1,000 cash units. `fees=0.001` charges 0.1% of
order value, while `slippage=0.0005` moves execution prices adversely by 0.05%. These are
separate costs. `total_return()` is a decimal fraction; the table multiplies it by 100.
Warmup bars remain in the evaluation period, with cash uninvested until a valid entry.

To model three strategy sleeves competing for one pool, continue the example with:

```python title="Use a shared cash pool instead of independent accounts"
shared = vbt.Portfolio.from_signals(
    close,
    entries,
    exits,
    price=open_price,
    size=1,
    init_cash=1_000,
    fees=0.001,
    slippage=0.0005,
    group_by=True,
    cash_sharing=True,
    call_seq="auto",
    freq="1D",
)
print(shared.value().iloc[[0, -1]].round(3).rename("Equity [cash]").to_frame().to_string())
print("Group total return [%]:", round(shared.total_return() * 100, 3))
```

Output:

```text
            Equity [cash]
2024-01-01       1000.000
2024-04-29        965.575
Group total return [%]: -3.442
```

Here the group has 1,000 cash units **in total**, versus 3,000 across the independent runs.
These are different allocation experiments, not directly comparable returns. Grouping alone
aggregates results; `cash_sharing=True` makes columns use the same cash. Processing order
matters when funds are constrained; `call_seq="auto"` attempts to execute sells before buys.
Select parameters on separate validation data rather than treating the best in-sample
column as evidence of a profitable strategy. See the
[portfolio API](/api/portfolio/base/#vectorbt.portfolio.base.Portfolio.from_signals)
for sizing, grouping, execution prices, and order-resolution rules.

For a market-data version, this optional example requires Yahoo Finance support and an
internet connection. It downloads a fixed historical interval, calculates the classic
50/200-day crossover, and overlays positions on the moving averages. Signals from a close
execute at the following day's open; fees are 0.5% per order. Provider adjustments and
historical-data revisions can change results. The output shown below was verified with an
offline synthetic fixture, not a fresh Yahoo download; live results will differ.

```python title="Optional: backtest the Golden Cross with market data"
import vectorbt as vbt

btc = vbt.YFData.download(
    "BTC-USD", start="2018-01-01", end="2024-01-01"
)
price = btc.get("Close")
open_price = btc.get("Open")
fast_ma = vbt.MA.run(price, 50, short_name="fast_ma")
slow_ma = vbt.MA.run(price, 200, short_name="slow_ma")
entries = fast_ma.ma_crossed_above(slow_ma).shift(1, fill_value=False)
exits = fast_ma.ma_crossed_below(slow_ma).shift(1, fill_value=False)
golden_pf = vbt.Portfolio.from_signals(
    price, entries, exits, price=open_price, fees=0.005, freq="1D"
)
print(golden_pf.orders.records_readable[
    ["Timestamp", "Size", "Price", "Fees", "Side"]
].head(3).round(3).to_string(index=False))
fig = price.vbt.plot(trace_kwargs=dict(name="Close"))
fast_ma.ma.vbt.plot(trace_kwargs=dict(name="Fast MA"), fig=fig)
slow_ma.ma.vbt.plot(trace_kwargs=dict(name="Slow MA"), fig=fig)
golden_pf.positions.plot(close_trace_kwargs=dict(visible=False), fig=fig)
```

Output:

```text
Offline validation fixture; not Yahoo Finance data:
 Timestamp  Size   Price  Fees Side
2020-09-25 0.934 106.529 0.498  Buy
2021-01-30 0.934  94.994 0.444 Sell
2021-06-03 0.820 107.168 0.439  Buy
```

Call `fig.show()` to explore the plot. The embedded historical chart below is an existing
illustration, not a freshly reproduced result of this download.

<div class="feature-embed">
  <iframe
    class="feature-embed__iframe"
    src="/assets/interactive/features_golden_crossover.html"
    title="Golden crossover chart"
    loading="lazy"
    scrolling="no">
  </iframe>
</div>

!!! tip "VectorBT PRO"

    Simulate [limit orders](https://vectorbt.pro/features/backtesting/orders-and-execution/#limit-orders),
    [leverage](https://vectorbt.pro/features/backtesting/fees-slippage-and-leverage/#leverage),
    [stop ladders](https://vectorbt.pro/features/backtesting/stop-loss-and-take-profit/#stop-laddering), and
    [cash deposits and withdrawals](https://vectorbt.pro/features/backtesting/portfolio-accounting/#cash-deposits).
    The [portfolio optimization tutorial](https://vectorbt.pro/tutorials/portfolio-optimization/)
    covers asset allocation and rebalancing.

## Analysis

### Portfolio backtesting

Inspect the selected strategy from the modeling example before reducing it to a single
score. Orders report fills; trades match entries and exits; positions combine connected
trades. Open trades use the final available valuation, so their unrealized results must
not be confused with completed round trips.

```python title="Inspect metrics, completed trades, and drawdowns"
# Continue the modeling example: selected is pf[(5, 20)].
stats = selected.stats()
print(stats.loc[[
    "Total Return [%]", "Max Drawdown [%]", "Total Closed Trades", "Total Open Trades"
]].to_string())
print("First two completed trades:")
print(selected.trades.closed.records_readable[
    ["Entry Timestamp", "Exit Timestamp", "Return", "Status"]
].head(2).round(4).to_string(index=False))
print("Open trades:", selected.trades.open.count())
print("First two positions:")
print(selected.positions.records_readable[
    ["Size", "PnL", "Return", "Status"]
].head(2).round(4).to_string(index=False))
print("Drawdown records:")
print(selected.drawdowns.records_readable[
    ["Peak Timestamp", "Valley Timestamp", "Status"]
].head(2).to_string(index=False))
print("Annualized Sharpe:", round(selected.get_returns_acc(year_freq="365D").sharpe_ratio(), 4))
figure = selected.plot(subplots=["value", "trades", "drawdowns"])
```

Output:

```text
Total Return [%]      -1.539336
Max Drawdown [%]        1.94245
Total Closed Trades           4
Total Open Trades             0
First two completed trades:
Entry Timestamp Exit Timestamp  Return Status
     2024-01-28     2024-02-11 -0.0319 Closed
     2024-02-23     2024-03-07 -0.0427 Closed
Open trades: 0
First two positions:
 Size     PnL  Return Status
  1.0 -3.2598 -0.0319 Closed
  1.0 -4.4785 -0.0427 Closed
Drawdown records:
Peak Timestamp Valley Timestamp Status
    2024-02-02       2024-04-26 Active
Annualized Sharpe: -4.3411
```

Call `figure.show()` to explore the equity curve, trades, and drawdowns interactively.
The output above selects a few metrics and records; `selected.stats()` returns the full report.
`stats()` uses readable labels such as `Total Return [%]` and `Max Drawdown [%]`, while
methods such as `total_return()` return decimal fractions. Undefined metrics can be `NaN`,
for example a win/loss statistic with no completed trades. The fixture uses calendar days:
`freq="1D"` and a `365D` year imply 365 observations per year. Configure an appropriate
annualization convention for your market; an index of business dates does not by itself
specify a trading-year convention. Short synthetic histories are unsuitable for estimating
stable risk-adjusted performance.

The stats builder also summarizes objects other than portfolios. This small mask makes its
signal counts, run lengths, and spacing easy to inspect without a strategy simulation.
Dates are attached to the series so reported first and last indices are actual timestamps.

```python title="Analyze the distribution of signals in a mask"
import pandas as pd
import vectorbt as vbt

mask = pd.Series(
    [False, True, True, True, False, True, False],
    index=pd.date_range("2020-01-01", periods=7, freq="D"),
)
stats = mask.vbt.signals(freq="1D").stats()
print(stats.loc[[
    "Total", "Rate [%]", "Total Partitions",
    "Partition Length: Min", "Partition Length: Max"
]].to_string())
```

Output:

```text
Total                                  4
Rate [%]                       57.142857
Total Partitions                       2
Partition Length: Min    1 days 00:00:00
Partition Length: Max    3 days 00:00:00
```

Drawdown records distinguish recovered declines from active declines that have not yet
regained a peak. The plot below shows a separate historical price-drawdown example. For
strategy risk, analyze the portfolio's equity curve (`selected.drawdowns`), rather than
substituting drawdowns of the underlying asset.

Drawdowns can also be calculated directly from a price series, independent of any portfolio.
This deterministic fixture contains repeated peaks and dips; `top_n=3` limits the plotted
records to the three deepest drawdowns. Price drawdowns and strategy-equity drawdowns
answer different questions.

```python title="Plot 3 deepest price dips"
import numpy as np
import pandas as pd
import vectorbt as vbt

t = np.arange(120)
price = pd.Series(
    100 + 0.05 * t + 5 * np.sin(t / 4),
    index=pd.date_range("2024-01-01", periods=len(t), freq="D"),
)
fig = price.vbt.drawdowns.plot(top_n=3)
print("Three deepest drawdowns [fraction]:")
print(price.vbt.drawdowns.drawdown.bottom_n(3).to_pd().dropna().round(4).to_string())
```

Output:

```text
Three deepest drawdowns [fraction]:
2024-01-30   -0.0887
2024-02-24   -0.0877
2024-03-20   -0.0869
```

Printed drawdowns are decimal fractions: `-0.08` is an 8% decline from the relevant peak.
Call `fig.show()` to explore this fixture. The existing embedded historical price-drawdown
chart illustrates the same operation with a separate data set.

<div class="feature-embed">
  <iframe
    class="feature-embed__iframe"
    src="/assets/interactive/features_top_drawdowns.html"
    title="Top drawdowns chart"
    loading="lazy"
    scrolling="no">
  </iframe>
</div>

Records and mapped arrays make event-level checks possible without converting every record
to a dense table. Because the modeling example enabled `log=True`, inspect its order logs
and find the largest charged fees with:

```python title="Inspect logged order fees"
# Continue the modeling example; logging was explicitly enabled there.
print(selected.logs.records_readable[
    ["Timestamp", "Request Price", "Result Price", "Result Fees", "Result Status"]
].head(3).round(4).to_string(index=False))
fees = selected.logs.map_field("res_fees")
print("Largest fees [cash]:")
print(fees.top_n(3).to_pd().dropna().round(4).to_string())
```

Output:

```text
 Timestamp  Request Price  Result Price  Result Fees Result Status
2024-01-28       102.2823      102.3335       0.1023        Filled
2024-02-11        99.3249       99.2752       0.0993        Filled
2024-02-23       104.7404      104.7928       0.1048        Filled
Largest fees [cash]:
2024-02-23    0.1048
2024-03-19    0.1059
2024-04-13    0.1068
```

Logs can include ignored or rejected requests as well as fills. Filter by status when
analyzing execution outcomes, and enable logging selectively for large simulations because
it adds memory cost. See [logs](/api/portfolio/logs/), [trade records](/api/portfolio/trades/),
[drawdowns](/api/generic/drawdowns/), and the
[stats builder](/api/generic/stats_builder/) for filters and custom metrics.

Mapped-array arithmetic can compare requested and executed prices, preserving order record
locations. Here random signal placement is seeded, and each order's slippage is drawn from
a seeded fixture. This is a test of execution accounting, not a signal-performance example.
Absolute ratios treat buy and sell price changes consistently; multiply by 100 to express
slippage in percent. The largest values are selected from records before conversion to pandas.

```python title="Parse 5 highest slippage values from logs"
import numpy as np
import pandas as pd
import vectorbt as vbt

t = np.arange(120)
price = pd.Series(
    100 + 0.05 * t + 5 * np.sin(t / 4),
    index=pd.date_range("2024-01-01", periods=len(t), freq="D"),
)
slippage = np.random.default_rng(42).uniform(0, 0.005, size=len(price))
logs = vbt.Portfolio.from_random_signals(
    price, n=5, seed=42, size=1, init_cash=1_000,
    slippage=slippage, log=True, freq="1D",
).logs
requested = logs.map_field("req_price")
executed = logs.map_field("res_price")
slippage_ma = (executed - requested) / requested
slippage_ma = slippage_ma.replace(mapped_arr=np.abs(slippage_ma.values))
print(slippage_ma.top_n(5).to_pd().dropna().round(6).to_string())
```

Output:

```text
2024-01-27    0.002334
2024-02-10    0.002186
2024-02-14    0.004161
2024-02-25    0.003904
2024-03-05    0.003174
```

This fixture has enough cash for its one-unit entries. For general logs, filter successful
fills before comparing execution prices; rejected requests have no executed price.

Entry-trade records attribute subsequent trade outcomes to individual entries. Their returns
are known retrospectively, not at the entry time. This example shows realized or end-of-series marked outcomes rather than a forecast.

```python title="Inspect the return attributed to each entry trade"
import numpy as np
import pandas as pd
import vectorbt as vbt

t = np.arange(120)
price = pd.Series(
    100 + 0.05 * t + 5 * np.sin(t / 4),
    index=pd.date_range("2024-01-01", periods=len(t), freq="D"),
)
entry_trades = vbt.Portfolio.from_random_signals(
    price, n=5, seed=42, size=1, init_cash=1_000, freq="1D"
).entry_trades
returns_by_entry = entry_trades.returns.to_pd()
print("Return attributed to each entry [fraction]:")
print(returns_by_entry.dropna().round(4).to_string())
print("First two entry trades:")
print(entry_trades.records_readable[
    ["Entry Timestamp", "Exit Timestamp", "Return", "Status"]
].head(2).round(4).to_string(index=False))
```

Output:

```text
Return attributed to each entry [fraction]:
2024-01-05   -0.0088
2024-01-27   -0.0302
2024-02-14    0.0155
2024-02-25   -0.0523
2024-03-14    0.0944
First two entry trades:
Entry Timestamp Exit Timestamp  Return Status
     2024-01-05     2024-01-11 -0.0088 Closed
     2024-01-27     2024-02-10 -0.0302 Closed
```

Returns above are decimal fractions, so `0.02` means 2%. Random placement is used only to
populate records. In a real strategy, entry-time features must be kept separate from these
future-dependent outcomes.

The optional [QuantStats adapter](/api/returns/qs_adapter/) exposes reports and plots for
portfolio returns. With QuantStats installed, `selected.qs.plot_snapshot()` plots the
selected column; use `selected.get_qs(year_freq="365D")` to make the annualization convention
explicit. Install QuantStats separately and select a single strategy before creating a
report. The historical snapshot below illustrates the integration rather than results of
this synthetic example.

The returns accessor supports QuantStats independently of a portfolio. This optional example
requires QuantStats to be installed. Its synthetic input keeps the reporting API reproducible
without a data-provider download. The code prints the prepared returns, then opens
a performance snapshot chart. The existing historical image is illustrative.

```python title="Optional: visualize performance using QuantStats"
import numpy as np
import pandas as pd
import vectorbt as vbt

t = np.arange(365)
price = pd.Series(
    100 + 0.05 * t + 5 * np.sin(t / 4),
    index=pd.date_range("2024-01-01", periods=len(t), freq="D"),
)
returns = price.vbt.to_returns()
print(returns.head().round(6).to_string())
returns.vbt.returns(freq="1D", year_freq="365D").qs.plot_snapshot()
```

Output:

```text
2024-01-01         NaN
2024-01-02    0.012870
2024-01-03    0.011947
2024-01-04    0.010352
2024-01-05    0.008200
Freq: D

Graphical output with QuantStats installed: a performance snapshot chart.
```

![](/assets/images/features_qs_plot_snapshot.png)

!!! tip "VectorBT PRO"

    Find [price patterns](https://vectorbt.pro/features/analytics/patterns-and-event-studies/#patterns) and analyze subsequent
    price movements with [event projections](https://vectorbt.pro/features/analytics/patterns-and-event-studies/#projections).
    Measure [maximum adverse and favorable excursion (MAE and MFE)](https://vectorbt.pro/features/analytics/trade-analytics/#mae-and-mfe)
    and [edge ratio](https://vectorbt.pro/features/analytics/trade-analytics/#edge-ratio) to examine trade entries and exits.

## Plotting

### Interactive visualization

VectorBT returns Plotly figures, so charts support hover labels, zooming, legend toggles, and the standard Plotly toolbar. You can combine its plotting helpers with ordinary Plotly traces instead of moving data into a separate visualization API. A figure is a view of a calculation: changing the visible range does not rerun a portfolio simulation.

The embedded charts illustrate the plotting capabilities; the standalone examples below use fixed local data and do not necessarily reproduce those snapshots exactly.

- [x] **Data visualization**: Flexible plotting helpers distributed across VectorBT.

```python title="Plot time series against each other"
import numpy as np
import pandas as pd
import vectorbt as vbt

rng = np.random.default_rng(42)
index = pd.date_range("2024-01-01", periods=100, freq="D")
series = pd.DataFrame(
    np.cumprod(1 + rng.normal(0, 0.01, size=(100, 2)), axis=0),
    index=index,
    columns=["Strategy", "Benchmark"],
)
fig = series["Strategy"].vbt.plot_against(series["Benchmark"])
# fig.show()
print(series.tail(3).round(4).to_string())
```

Output:

```text
            Strategy  Benchmark
2024-04-07    0.9751     0.9568
2024-04-08    0.9814     0.9530
2024-04-09    0.9813     0.9514
```

<div class="feature-embed">
  <iframe
    class="feature-embed__iframe"
    src="/assets/interactive/features_plot_against.html"
    title="Plot against chart"
    loading="lazy"
    scrolling="no">
  </iframe>
</div>

- [x] **Figures and widgets**: Interactive figures and widgets using [Plotly](https://github.com/plotly/plotly.py), including heatmaps and volume plots. Plotting objects provide dedicated `update` methods to replace data without reconstructing the whole figure.

```python title="Create and update a volume"
import numpy as np
import vectorbt as vbt

volume = vbt.plotting.Volume(
    data=np.arange(1, 28).reshape(3, 3, 3),
    x_labels=["a", "b", "c"],
    y_labels=["d", "e", "f"],
    z_labels=["g", "h", "i"],
)
updated = np.arange(27, 0, -1).reshape(3, 3, 3)
volume.update(updated)
# volume.fig.show()
print("First slice after update:")
print(updated[0])
```

Output:

```text
First slice after update:
[[27 26 25]
 [24 23 22]
 [21 20 19]]
```

<div class="feature-embed">
  <iframe
    class="feature-embed__iframe feature-embed__iframe--volume"
    src="/assets/interactive/features_volume.html"
    title="Volume widget"
    loading="lazy"
    scrolling="no">
  </iframe>
</div>

- [x] **Plots builder**: Assemble tailored subplots for signals, returns, portfolios, and other analysis objects. Select one strategy column before plotting a parameter sweep, then inspect its statistics alongside the chart.

```python title="Inspect one strategy from a parameter sweep"
import numpy as np
import pandas as pd
import vectorbt as vbt

close = pd.Series(
    100 + np.arange(80) * 0.15 + 4 * np.sin(np.arange(80) / 4),
    index=pd.date_range("2024-01-01", periods=80, freq="D"),
    name="Close",
)
ma = vbt.MA.run(close, window=[3, 5, 10])
portfolio = vbt.Portfolio.from_signals(
    close,
    ma.close_crossed_above(ma.ma).shift(1, fill_value=False),
    ma.close_crossed_below(ma.ma).shift(1, fill_value=False),
    init_cash=1000, fees=0.001, freq="1D",
)
selected = portfolio[5]
summary = selected.stats()
fig = selected.plot(subplots=["cash", "assets", "value"])
print(summary[["Total Return [%]", "Max Drawdown [%]", "Total Trades"]].to_string())
# fig.show()
```

Output:

```text
Total Return [%]    24.608134
Max Drawdown [%]     1.125275
Total Trades                3
```

Signals use a completed close and are shifted one observation before execution. The simulation therefore fills at the next observed close, not at the next open. Supply explicit execution prices when modeling a different fill convention.

<div class="feature-embed">
  <iframe
    class="feature-embed__iframe feature-embed__iframe--portfolio-plot"
    src="/assets/interactive/features_portfolio_plot.html"
    title="Portfolio balances chart"
    loading="lazy"
    scrolling="no">
  </iframe>
</div>

For a custom report, use ordinary Plotly subplots. Shared x-axes let a reader compare price movements and account value on the same dates without combining incompatible scales. The following example includes a buy-and-hold reference in the account's currency; both curves start with the same cash.

```python title="Combine portfolio results with custom Plotly traces"
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import vectorbt as vbt

close = pd.Series(
    [100, 102, 101, 104, 106, 103, 107, 109],
    index=pd.date_range("2024-01-01", periods=8, freq="D"),
)
portfolio = vbt.Portfolio.from_holding(close, init_cash=1000, fees=0.001, freq="1D")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
fig.add_trace(go.Scatter(x=close.index, y=close, name="Close"), row=1, col=1)
fig.add_trace(
    go.Scatter(x=close.index, y=portfolio.value(), name="Portfolio value"),
    row=2, col=1,
)
fig.add_trace(
    go.Scatter(x=close.index, y=1000 * close / close.iloc[0], name="Buy and hold (no fees)"),
    row=2, col=1,
)
fig.update_yaxes(title_text="Price", row=1, col=1)
fig.update_yaxes(title_text="Account value", row=2, col=1)
fig.update_layout(template="plotly", height=600)
# fig.show()
comparison = pd.DataFrame({
    "Portfolio value": portfolio.value(),
    "Buy and hold (no fees)": 1000 * close / close.iloc[0],
})
print(comparison.iloc[[0, -1]].round(2).to_string())
```

Output:

```text
            Portfolio value  Buy and hold (no fees)
2024-01-01           999.00                  1000.0
2024-01-08          1088.91                  1090.0
```

**Browser reports versus notebook callbacks.** Save any of these figures with `fig.write_html("report.html", include_plotlyjs=True, full_html=True)`. This embeds Plotly's JavaScript in a standalone file, so a recipient can open it without a Python server or an internet connection. Hover, zoom, legend toggles, and controls already represented in the figure remain interactive. The file is larger than a CDN-linked export because it includes the plotting library.

Python callbacks attached to a `FigureWidget` need a running notebook kernel; exporting HTML does not package Python code or turn it into a web application. VectorBT uses widgets by default through `vbt.settings.plotting.use_widgets`. The notebook environment must support the installed Plotly widget backend (the base installation includes `ipywidgets` and `anywidget`). Use `plotly.graph_objects.Figure` for a regular figure when no Python callbacks are needed. Static PNG/SVG export is a separate operation and requires the optional Kaleido setup; it removes browser interactivity.

See the [plotting accessors](../api/generic/accessors.md),
[plotting objects](../api/generic/plotting.md), and
[figure utilities](../api/utils/figure.md) for trace options, updates, and subplot composition.

!!! tip "VectorBT PRO"

    Plot [TA-Lib indicator outputs](https://vectorbt.pro/features/indicators/technical-indicators/#ta-lib-plotting)
    or [long and short trade signals](https://vectorbt.pro/features/analytics/trade-analytics/#trade-signals).
    [Portfolio slicing](https://vectorbt.pro/features/tooling/multidimensional-research/#slicing) lets you select
    and plot a date range without running the simulation again.

## Extra

### Automation tools

Schedule repeated calculations with `ScheduleManager` and connect completed results to notification handlers. Scheduling decides when Python calls a function; it does not fetch data, persist state, retry failures, or place orders by itself. Keep those responsibilities explicit in the job so the same function can also be run manually.

```python title="Refresh a local price series without starting a service"
import pandas as pd
import vectorbt as vbt

close = pd.Series([100.0, 101.0], index=pd.date_range("2024-01-01", periods=2, freq="D"))
incoming_prices = iter([102.0, 103.0])
snapshots = []

def refresh():
    global close
    next_date = close.index[-1] + pd.Timedelta(days=1)
    close = pd.concat([close, pd.Series([next(incoming_prices)], index=[next_date])])
    portfolio = vbt.Portfolio.from_holding(close, init_cash=1000, freq="1D")
    snapshots.append({"as_of": next_date, "total_return": float(portfolio.total_return())})

manager = vbt.ScheduleManager()
manager.every(10, "seconds").do(refresh)
manager.scheduler.run_all()  # Run each job once now, without waiting.
manager.scheduler.run_all()
manager.scheduler.clear()
print(pd.DataFrame(snapshots).set_index("as_of").to_string())
```

Output:

```text
            total_return
as_of
2024-01-03          0.02
2024-01-04          0.03
```

This deterministic example appends two local observations and finishes. In a live job, replace the local iterator with a provider's incremental download and validate timestamps, missing data, and duplicate bars before recalculating. `manager.scheduler.run_pending()` checks due jobs once and returns; a host application can call it from its own loop. `manager.start()` instead runs a blocking loop until interrupted. Do not call it merely to test that a job was registered. Synchronous jobs run in the calling thread, so a slow data request delays other jobs.

In a notebook, fetch recent exchange trades, render them as SVG, and replace the previous output. The following optional notebook example needs `ccxt`, network access to Binance, and a working [Plotly static-image export setup](https://plotly.com/python/static-image-export/) (`kaleido` and compatible Chrome/Chromium for Kaleido 1+). Exchange availability and returned history vary by region and endpoint. Empty responses leave the last chart in place.

```python title="Optional: every 10 seconds display the latest Bitcoin trades on Binance"
import ccxt
import pandas as pd
import vectorbt as vbt
from IPython.display import SVG, clear_output, display
from vectorbt.utils.datetime_ import datetime_to_ms

exchange = ccxt.binance({"enableRateLimit": True})

def display_recent_trades():
    since = datetime_to_ms((pd.Timestamp.now(tz="UTC") - pd.Timedelta(seconds=10)).to_pydatetime())
    trades = exchange.fetch_trades("BTC/USDT", since=since)
    if not trades:
        return
    trades = sorted(trades, key=lambda trade: trade["timestamp"])
    price = pd.Series(
        [trade["price"] for trade in trades],
        index=pd.to_datetime([trade["timestamp"] for trade in trades], unit="ms", utc=True),
        name="BTC/USDT",
    )
    svg = price.vbt.plot().to_image(format="svg")
    clear_output(wait=True)
    display(SVG(svg))

scheduler = vbt.ScheduleManager()
scheduler.every(10, "seconds").do(display_recent_trades)
try:
    scheduler.start()  # Intentional blocking service; interrupt the notebook to stop.
finally:
    scheduler.scheduler.clear()
```

Output:

```text
Illustrative notebook output (no live request executed):
[SVG chart of BTC/USDT trade prices against UTC timestamps]
[Chart replaced after each successful refresh]
```

SVG is deliberately static here: Python refreshes the notebook output every ten seconds. Use a Plotly figure or a Python-backed widget instead when readers should hover or zoom. A request or export error stops this simple service; handle retries and provider outages in the job when adapting it for ongoing use.

![](/assets/images/features_scheduler.svg)

**Telegram integration (optional).** Install `python-telegram-bot` separately or through `vectorbt[full]`. VectorBT's `TelegramBot` has separate implementations for the legacy API and the 20+ `ApplicationBuilder` API. Its methods such as `send_message` remain blocking wrappers; the upstream library's [20+ callbacks are asynchronous](https://docs.python-telegram-bot.org/en/v20.6/telegram.ext.applicationbuilder.html). Do not copy legacy `Updater`/`Filters` recipes directly into modern upstream code.

The optional script below uses the 20+ implementation and sends one completed report. It requires an actual bot token, an authorized chat, and network access; it is not part of the offline examples. The recipient must first start the bot in Telegram. Store configuration in environment variables rather than a notebook, repository, or shared HTML report.

```python title="Optional: send a report through Telegram (requires a live service)"
import os
import vectorbt as vbt

bot = vbt.TelegramBot(token=os.environ["TELEGRAM_BOT_TOKEN"])
try:
    bot.start(in_background=True)
    bot.send_message(int(os.environ["TELEGRAM_CHAT_ID"]), "Research run complete; report saved.")
finally:
    bot.stop()
```

Output:

```text
Expected Telegram message:
Research run complete; report saved.
```

Keep polling alive only as long as the application needs it and stop the bot during shutdown. This is a notification transport, not an order execution or monitoring service.

For an interactive bot, subclass `TelegramBot` and register a custom command. This optional standalone script handles `/get BTC/USDT` ticker requests using `python-telegram-bot>=20` and asynchronous CCXT. It requires both optional packages, network access, and `TELEGRAM_BOT_TOKEN`. Use Telegram's `/help` command to see the accepted syntax.

```python title="Optional: launch a bot that returns the latest ticker on Binance"
import os
import ccxt.async_support as ccxt
from telegram.ext import CommandHandler
import vectorbt as vbt

class BinanceTickerBot(vbt.TelegramBot):
    @property
    def custom_handlers(self):
        return (CommandHandler("get", self.get),)

    @property
    def help_message(self):
        return "Type /get BTC/USDT to get the latest ticker on Binance."

    async def get(self, update, context):
        if update.effective_chat is None:
            return
        chat_id = update.effective_chat.id
        if len(context.args) != 1:
            await context.bot.send_message(chat_id=chat_id, text=self.help_message)
            return
        exchange = ccxt.binance({"enableRateLimit": True})
        try:
            ticker = await exchange.fetch_ticker(context.args[0])
            last = ticker.get("last")
            text = f"{context.args[0]}: {last}" if last is not None else "No last price available."
        except ccxt.BaseError as error:
            text = f"Ticker unavailable: {type(error).__name__}"
        finally:
            await exchange.close()
        await context.bot.send_message(chat_id=chat_id, text=text)

bot = BinanceTickerBot(token=os.environ["TELEGRAM_BOT_TOKEN"])
try:
    bot.start()  # Poll until interrupted.
finally:
    bot.stop()
```

Output:

```text
Illustrative Telegram conversation (no live quote fetched):
You: /get BTC/USDT
Bot: BTC/USDT: <latest price returned by Binance>
```

Inside an asynchronous handler, await the upstream `context.bot.send_message` method rather than calling VectorBT's blocking `self.send_message` wrapper from the bot event loop. CCXT's [asynchronous Python API](https://github.com/ccxt/ccxt/wiki/Manual#synchronous-vs-asynchronous-calls) likewise requires awaiting requests and closing the exchange when finished. This example fetches public ticker data and needs no exchange API credentials.

The [scheduling API](../api/utils/schedule_.md) describes job registration and
lifecycle methods. The [Telegram API](../api/messaging/telegram.md) covers
custom handlers, authorization, and message delivery.

### Caching and persistence

- [x] **Caching**: Cache frequently reused properties and methods. A cached result can become stale when external state changes; prefer immutable inputs or explicitly clear the relevant cache.

```python title="Cache a deterministic calculation"
import vectorbt as vbt

class Research:
    def __init__(self):
        self.calls = 0

    @vbt.cached_method
    def square(self, value):
        self.calls += 1
        return value * value

research = Research()
print(research.square(3), research.square(3))  # 9 9
print(research.calls)  # 1: the second call used the cache.
Research.square.clear_cache(research)
print(research.square(3), research.calls)  # 9 2: recalculated after clearing.
```

Output:

```text
9 9
1
9 2
```

Clearing a cache only discards stored results; the next call can cache again. To disable caching for a specific method and instance through the global policy, add a `CacheCondition` to the blacklist. The continuation below removes its condition afterward so subsequent examples retain their original settings.

```python title="Temporarily disable caching through the global blacklist"
import vectorbt as vbt

condition = vbt.CacheCondition(instance=research, func="square")
blacklist = vbt.settings.caching["blacklist"]
blacklist.append(condition)
try:
    calls_before = research.calls
    print(research.square(3), research.square(3))
    print(research.calls - calls_before)  # 2: neither call used the cache.
finally:
    blacklist.remove(condition)
```

Output:

```text
9 9
2
```

- [x] **Persistence**: Data, portfolios, and many other objects can be saved and restored through Dill. A saved portfolio retains the simulated records; it does not download newer prices when loaded. Only load serialized objects from trusted sources, and keep the package versions used to create important research artifacts.

```python title="Save and restore a portfolio in a temporary directory"
from pathlib import Path
from tempfile import TemporaryDirectory
import pandas as pd
import vectorbt as vbt

close = pd.Series([100.0, 102.0, 101.0, 105.0])
portfolio = vbt.Portfolio.from_holding(close, init_cash=1000, freq="1D")
with TemporaryDirectory() as directory:
    path = Path(directory) / "portfolio.pkl"
    portfolio.save(path)
    restored = vbt.Portfolio.load(path)
    print(restored.total_return())
```

Output:

```text
0.05
```

See [caching decorators](../api/utils/decorators.md) for cache controls and
[configuration and persistence](../api/utils/config.md) for serialization methods.

!!! tip "VectorBT PRO"

    Run independent functions in parallel with [tasks](https://vectorbt.pro/features/tooling/workflow-automation/#tasks),
    inspect memory use with the [caching registry](https://vectorbt.pro/features/performance/parallel-execution-and-caching/#caching),
    and save objects in readable [configuration files](https://vectorbt.pro/features/tooling/configuration-and-persistence/#configuration-files).

## Python workflows

### Composable Python API

Put a research step in a function with explicit data, parameters, and costs. Return labeled values instead of relying on notebook state or reading numbers from a chart. This makes the same calculation usable from a notebook, a batch script, a test, or a Python tool called by an AI agent. VectorBT does not require an agent framework: callers use the same pandas objects and Python functions.

```python title="Return comparable, labeled research results"
import numpy as np
import pandas as pd
import vectorbt as vbt

def research_windows(close, windows=(3, 5, 10), fees=0.001):
    if not isinstance(close, pd.Series) or close.empty or close.isna().any():
        raise ValueError("Pass a nonempty price Series without missing values")
    if not close.index.is_monotonic_increasing or close.index.has_duplicates:
        raise ValueError("Prices must have a sorted, unique index")
    if fees < 0 or not windows or any(window < 1 for window in windows):
        raise ValueError("Use nonnegative fees and positive windows")
    curves, rows = {}, {}
    for window in windows:
        ma = vbt.MA.run(close, window=window)
        portfolio = vbt.Portfolio.from_signals(
            close,
            ma.close_crossed_above(ma.ma).shift(1, fill_value=False),
            ma.close_crossed_below(ma.ma).shift(1, fill_value=False),
            init_cash=1000, fees=fees, freq="1D",
        )
        curves[window] = portfolio.value()
        rows[window] = {
            "total_return": float(portfolio.total_return()),
            "max_drawdown": float(portfolio.max_drawdown()),
            "trade_count": int(portfolio.trades.count()),
        }
    metrics = pd.DataFrame.from_dict(rows, orient="index").rename_axis("window")
    equity = pd.DataFrame(curves).rename_axis(columns="window")
    return {"metrics": metrics, "equity": equity}

close = pd.Series(
    100 + np.arange(100) * 0.1 + 3 * np.sin(np.arange(100) / 3),
    index=pd.date_range("2024-01-01", periods=100, freq="D"),
)
results = research_windows(close, windows=(3, 7), fees=0.001)
print(results["metrics"].round(6).to_string())
payload = results["metrics"].reset_index().to_json(orient="records", double_precision=4)
print("\nJSON:")
print(payload)
```

Output:

```text
        total_return  max_drawdown  trade_count
window
3           0.273346     -0.010855            5
7           0.176759     -0.023727            5

JSON:
[{"window":3,"total_return":0.2733,"max_drawdown":-0.0109,"trade_count":5},{"window":7,"total_return":0.1768,"max_drawdown":-0.0237,"trade_count":5}]
```

Returns and drawdowns here are fractions, not percentages; column labels carry the window used for each run. This example assumes daily bars and supplies `freq="1D"` explicitly. Signals are shifted to fill at the following observed close. Adapt the frequency and execution convention to your dataset rather than inheriting these assumptions for intraday data. The JSON summary is convenient for a tool response, while the equity DataFrame retains the time index for further analysis and plotting. Preserve the input data and configuration alongside important results so another caller can reproduce the run. Selecting a window on these same prices is in-sample research, not evidence of out-of-sample performance; use a separate evaluation period before drawing conclusions.
