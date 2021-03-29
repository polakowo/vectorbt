"""Custom data classes that subclass `vectorbt.data.base.Data`."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import warnings
from functools import wraps

from vectorbt.utils.datetime import (
    get_utc_tz,
    get_local_tz,
    to_tzaware_datetime,
    datetime_to_ms
)
from vectorbt.utils.config import merge_dicts, get_func_kwargs
from vectorbt.data.base import Data


class SyntheticData(Data):
    """`Data` for synthetically generated data."""

    @classmethod
    def generate_symbol(cls, symbol, index, **kwargs):
        """Abstract method to generate a symbol."""
        raise NotImplementedError

    @classmethod
    def download_symbol(cls, symbol, start=0, end='now', freq=None, date_range_kwargs=None, **kwargs):
        """Download the symbol.

        Generates datetime index and passes it to `SyntheticData.generate_symbol` to fill
        the Series/DataFrame with generated data."""
        if date_range_kwargs is None:
            date_range_kwargs = {}
        index = pd.date_range(
            start=to_tzaware_datetime(start),
            end=to_tzaware_datetime(end),
            freq=freq,
            **date_range_kwargs
        )
        if len(index) == 0:
            raise ValueError("Date range is empty")
        return cls.generate_symbol(symbol, index, **kwargs)

    def update_symbol(self, symbol, **kwargs):
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `SyntheticData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, **kwargs)


def generate_gbm_paths(S0, mu, sigma, T, M, I, seed=None):
    """Generate using Geometric Brownian Motion (GBM).

    See https://stackoverflow.com/a/45036114/8141780."""
    if seed is not None:
        np.random.seed(seed)

    dt = float(T) / M
    paths = np.zeros((M + 1, I), np.float64)
    paths[0] = S0
    for t in range(1, M + 1):
        rand = np.random.standard_normal(I)
        paths[t] = paths[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * rand)
    return paths


class GBMData(SyntheticData):
    """`SyntheticData` for data generated using Geometric Brownian Motion (GBM).

    ## Example

    See the example under `BinanceData`.

    ```python-repl
    >>> import vectorbt as vbt

    >>> gbm_data = vbt.GBMData.download('GBM', start='2 hours ago', end='now', freq='1min', seed=42)
    >>> gbm_data.get()
    2021-03-15 17:56:59.174370+01:00    102.386605
    2021-03-15 17:57:59.174370+01:00    101.554203
    2021-03-15 17:58:59.174370+01:00    104.765771
    ...                                        ...
    2021-03-15 19:54:59.174370+01:00     51.614839
    2021-03-15 19:55:59.174370+01:00     53.525376
    2021-03-15 19:56:59.174370+01:00     55.615250
    Freq: T, Length: 121, dtype: float64

    >>> import time
    >>> time.sleep(60)

    >>> gbm_data = gbm_data.update()
    >>> gbm_data.get()
    2021-03-15 17:56:59.174370+01:00    102.386605
    2021-03-15 17:57:59.174370+01:00    101.554203
    2021-03-15 17:58:59.174370+01:00    104.765771
    ...                                        ...
    2021-03-15 19:55:59.174370+01:00     53.525376
    2021-03-15 19:56:59.174370+01:00     51.082220
    2021-03-15 19:57:59.174370+01:00     54.725304
    Freq: T, Length: 122, dtype: float64
    ```"""

    @classmethod
    def generate_symbol(cls, symbol, index, S0=100., mu=0., sigma=0.05, T=None, I=1, seed=None):
        """Generate the symbol using `generate_gbm_paths`.

        Args:
            symbol (str): Symbol.
            index (pd.Index): Pandas index.
            S0 (float): Value at time 0.

                Does not appear as the first value in the output data.
            mu (float): Drift, or mean of the percentage change.
            sigma (float): Standard deviation of the percentage change.
            T (int): Number of time steps.

                Defaults to the length of `index`.
            I (int): Number of generated paths (columns in our case).
            seed (int): Set seed to make the results deterministic.
        """
        if T is None:
            T = len(index)
        out = generate_gbm_paths(S0, mu, sigma, T, len(index), I, seed=seed)[1:]
        if out.shape[1] == 1:
            return pd.Series(out[:, 0], index=index)
        columns = pd.RangeIndex(stop=out.shape[1], name='path')
        return pd.DataFrame(out, index=index, columns=columns)

    def update_symbol(self, symbol, **kwargs):
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `GBMData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        _ = download_kwargs.pop('S0', None)
        S0 = self.data[symbol].iloc[-2]
        _ = download_kwargs.pop('T', None)
        download_kwargs['seed'] = None
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, S0=S0, **kwargs)


class YFData(Data):
    """`Data` for data coming from `yfinance`.

    !!! note
        Sometimes `yfinance` returns a tz-naive datetime index. To produce a
        tz-aware datetime index, pass `tz_localize` to `YFData.download`. In this case,
        you would need to find a timezone that fits the data (+0500, +0000, etc.)

    ## Example

    Fetch the business day except the last 5 minutes of trading data, and then update with the missing 5 minutes:

    ```python-repl
    >>> import vectorbt as vbt

    >>> yf_data = vbt.YFData.download(
    ...     "TSLA",
    ...     start='2021-03-12 09:30:00 -0500',
    ...     end='2021-03-12 15:55:00 -0500',
    ...     interval='1m'
    ... )
    >>> yf_data.get()
                                     Open        High         Low       Close  \\
    Datetime
    2021-03-12 09:30:00-05:00  670.000000  674.455811  667.510010  673.445007
    2021-03-12 09:31:00-05:00  673.424988  674.455811  668.099976  669.599976
    2021-03-12 09:32:00-05:00  669.270020  673.000000  668.239990  670.929993
    ...                               ...         ...         ...         ...
    2021-03-12 15:52:00-05:00  690.630005  691.030029  690.380005  690.929993
    2021-03-12 15:53:00-05:00  690.989990  692.200012  690.960022  692.039978
    2021-03-12 15:54:00-05:00  692.000000  692.783875  691.929993  692.479980

                               Volume  Dividends  Stock Splits
    Datetime
    2021-03-12 09:30:00-05:00       0          0             0
    2021-03-12 09:31:00-05:00  198906          0             0
    2021-03-12 09:32:00-05:00  202002          0             0
    ...                           ...        ...           ...
    2021-03-12 15:52:00-05:00   97941          0             0
    2021-03-12 15:53:00-05:00  155552          0             0
    2021-03-12 15:54:00-05:00  110320          0             0

    [382 rows x 7 columns]

    >>> yf_data = yf_data.update(end='2021-03-12 16:00:00 -0500')
    >>> yf_data.get()
                                     Open        High         Low       Close  \\
    Datetime
    2021-03-12 09:30:00-05:00  670.000000  674.455811  667.510010  673.445007
    2021-03-12 09:31:00-05:00  673.424988  674.455811  668.099976  669.599976
    2021-03-12 09:32:00-05:00  669.270020  673.000000  668.239990  670.929993
    ...                               ...         ...         ...         ...
    2021-03-12 15:57:00-05:00  693.239990  693.599976  693.039978  693.250000
    2021-03-12 15:58:00-05:00  693.255005  693.419983  692.640015  692.950012
    2021-03-12 15:59:00-05:00  692.909973  694.099976  692.570007  693.840027

                               Volume  Dividends  Stock Splits
    Datetime
    2021-03-12 09:30:00-05:00       0          0             0
    2021-03-12 09:31:00-05:00  198906          0             0
    2021-03-12 09:32:00-05:00  202002          0             0
    ...                           ...        ...           ...
    2021-03-12 15:57:00-05:00  136808          0             0
    2021-03-12 15:58:00-05:00  104432          0             0
    2021-03-12 15:59:00-05:00  192474          0             0

    [387 rows x 7 columns]
    ```
    """

    @classmethod
    def download_symbol(cls, symbol, period='max', start=None, end=None, **kwargs):
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            start (any): Start datetime.

                See `vectorbt.utils.datetime.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime.to_tzaware_datetime`.
            **kwargs: Keyword arguments passed to `yfinance.base.TickerBase.history`.
        """
        import yfinance as yf

        # yfinance still uses mktime, which assumes that the passed date is in local time
        if start is not None:
            start = to_tzaware_datetime(start, tz=get_local_tz())
        if end is not None:
            end = to_tzaware_datetime(end, tz=get_local_tz())

        return yf.Ticker(symbol).history(period=period, start=start, end=end, **kwargs)

    def update_symbol(self, symbol, **kwargs):
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `YFData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, **kwargs)


class BinanceData(Data):
    """`Data` for data coming from `python-binance`.

    ## Example

    Fetch the 1-minute data of the last 2 hours, wait 1 minute, and update:

    ```python-repl
    >>> import vectorbt as vbt

    >>> binance_data = vbt.BinanceData.download(
    ...     "BTCUSDT",
    ...     start='2 hours ago UTC',
    ...     end='now UTC',
    ...     interval='1m'
    ... )
    >>> binance_data.get()
    2021-03-22 21:39:22.067000+00:00 - 2021-03-22 23:39:00+00:00: : 1it [00:00,  3.80it/s]

                                   Open      High       Low     Close     Volume  \\
    Open time
    2021-03-22 21:40:00+00:00  54676.30  54691.41  54532.77  54549.49  35.566780
    2021-03-22 21:41:00+00:00  54547.36  54593.96  54543.23  54589.81  32.813574
    2021-03-22 21:42:00+00:00  54589.81  54655.85  54585.57  54650.04  19.616003
    ...                             ...       ...       ...       ...        ...
    2021-03-22 23:37:00+00:00  54443.83  54568.80  54403.49  54511.87  26.505448
    2021-03-22 23:38:00+00:00  54511.86  54511.86  54388.00  54406.59  24.879893
    2021-03-22 23:39:00+00:00  54411.55  54430.74  54386.70  54430.73   4.010454

                                                    Close time  Quote volume  \\
    Open time
    2021-03-22 21:40:00+00:00 2021-03-22 21:40:59.999000+00:00  1.943891e+06
    2021-03-22 21:41:00+00:00 2021-03-22 21:41:59.999000+00:00  1.790163e+06
    2021-03-22 21:42:00+00:00 2021-03-22 21:42:59.999000+00:00  1.071824e+06
    ...                                                    ...           ...
    2021-03-22 23:37:00+00:00 2021-03-22 23:37:59.999000+00:00  1.444602e+06
    2021-03-22 23:38:00+00:00 2021-03-22 23:38:59.999000+00:00  1.354316e+06
    2021-03-22 23:39:00+00:00 2021-03-22 23:39:59.999000+00:00  2.182080e+05

                               Number of trades  Taker base volume  \\
    Open time
    2021-03-22 21:40:00+00:00              1544          15.160921
    2021-03-22 21:41:00+00:00              1638          15.132376
    2021-03-22 21:42:00+00:00              1022          10.442487
    ...                                     ...                ...
    2021-03-22 23:37:00+00:00               795          13.412746
    2021-03-22 23:38:00+00:00               815          10.808586
    2021-03-22 23:39:00+00:00               209           2.312393

                               Taker quote volume
    Open time
    2021-03-22 21:40:00+00:00        8.288645e+05
    2021-03-22 21:41:00+00:00        8.256025e+05
    2021-03-22 21:42:00+00:00        5.706188e+05
    ...                                       ...
    2021-03-22 23:37:00+00:00        7.310994e+05
    2021-03-22 23:38:00+00:00        5.883322e+05
    2021-03-22 23:39:00+00:00        1.258189e+05

    [120 rows x 10 columns]

    >>> import time
    >>> time.sleep(60)

    >>> binance_data = binance_data.update()
    >>> binance_data.get()
                                   Open      High       Low     Close     Volume  \\
    Open time
    2021-03-22 21:40:00+00:00  54676.30  54691.41  54532.77  54549.49  35.566780
    2021-03-22 21:41:00+00:00  54547.36  54593.96  54543.23  54589.81  32.813574
    2021-03-22 21:42:00+00:00  54589.81  54655.85  54585.57  54650.04  19.616003
    ...                             ...       ...       ...       ...        ...
    2021-03-22 23:38:00+00:00  54511.86  54511.86  54388.00  54406.59  24.879893
    2021-03-22 23:39:00+00:00  54411.55  54499.49  54386.70  54460.91  21.214675
    2021-03-22 23:40:00+00:00  54460.90  54571.00  54458.12  54570.99  11.484854

                                                    Close time  Quote volume  \\
    Open time
    2021-03-22 21:40:00+00:00 2021-03-22 21:40:59.999000+00:00  1.943891e+06
    2021-03-22 21:41:00+00:00 2021-03-22 21:41:59.999000+00:00  1.790163e+06
    2021-03-22 21:42:00+00:00 2021-03-22 21:42:59.999000+00:00  1.071824e+06
    ...                                                    ...           ...
    2021-03-22 23:38:00+00:00 2021-03-22 23:38:59.999000+00:00  1.354316e+06
    2021-03-22 23:39:00+00:00 2021-03-22 23:39:59.999000+00:00  1.155268e+06
    2021-03-22 23:40:00+00:00 2021-03-22 23:40:59.999000+00:00  6.262655e+05

                               Number of trades  Taker base volume  \\
    Open time
    2021-03-22 21:40:00+00:00              1544          15.160921
    2021-03-22 21:41:00+00:00              1638          15.132376
    2021-03-22 21:42:00+00:00              1022          10.442487
    ...                                     ...                ...
    2021-03-22 23:38:00+00:00               815          10.808586
    2021-03-22 23:39:00+00:00               693          10.565621
    2021-03-22 23:40:00+00:00               448           6.107031

                               Taker quote volume
    Open time
    2021-03-22 21:40:00+00:00        8.288645e+05
    2021-03-22 21:41:00+00:00        8.256025e+05
    2021-03-22 21:42:00+00:00        5.706188e+05
    ...                                       ...
    2021-03-22 23:38:00+00:00        5.883322e+05
    2021-03-22 23:39:00+00:00        5.753598e+05
    2021-03-22 23:40:00+00:00        3.330209e+05

    [121 rows x 10 columns]
    ```"""

    @classmethod
    def download(cls, symbols, client=None, **kwargs):
        """Override `vectorbt.data.base.Data.download` to instantiate a Binance client."""
        from binance.client import Client
        from vectorbt import settings

        client_kwargs = dict()
        for k in get_func_kwargs(Client):
            if k in kwargs:
                client_kwargs[k] = kwargs.pop(k)
        client_kwargs = merge_dicts(settings.data['binance'], client_kwargs)
        if client is None:
            client = Client(**client_kwargs)
        return super(BinanceData, cls).download(symbols, client=client, **kwargs)

    @classmethod
    def download_symbol(cls, symbol, client=None, interval=None, start=0, end='now UTC',
                        delay=500, limit=500, show_progress=True):
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            client (binance.client.Client): Binance client of type `binance.client.Client`.

                Overrides `binance` settings defined in `vectorbt.settings.data`.
            interval (str): Kline interval.

                See `binance.enums`.
            start (any): Start datetime.

                See `vectorbt.utils.datetime.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime.to_tzaware_datetime`.
            delay (int or float): Time to sleep after each request (in milliseconds).
            limit (int): The maximum number of returned items.
            show_progress (bool): Whether to show the progress bar.
        """
        if client is None:
            raise ValueError("client must be provided")

        # Establish the timestamps
        start_ts = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        try:
            first_data = client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=1,
                startTime=0,
                endTime=None
            )
            first_valid_ts = first_data[0][0]
            next_start_ts = start_ts = max(start_ts, first_valid_ts)
        except:
            next_start_ts = start_ts
        end_ts = datetime_to_ms(to_tzaware_datetime(end, tz=get_utc_tz()))

        def _ts_to_str(ts):
            return str(pd.Timestamp(to_tzaware_datetime(ts, tz=get_utc_tz())))

        # Iteratively collect the data
        data = []
        with tqdm(disable=not show_progress) as pbar:
            pbar.set_description(_ts_to_str(start_ts))
            while True:
                # Fetch the klines for the next interval
                next_data = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=next_start_ts,
                    endTime=end_ts
                )
                if len(data) > 0:
                    next_data = list(filter(lambda d: next_start_ts < d[0] < end_ts, next_data))
                else:
                    next_data = list(filter(lambda d: d[0] < end_ts, next_data))

                # Update the timestamps and the progress bar
                if not len(next_data):
                    break
                data += next_data
                pbar.set_description("{} - {}".format(
                    _ts_to_str(start_ts),
                    _ts_to_str(next_data[-1][0])
                ))
                pbar.update(1)
                next_start_ts = next_data[-1][0]
                if delay is not None:
                    time.sleep(delay / 1000)  # be kind to api

        # Convert data to a DataFrame
        df = pd.DataFrame(data, columns=[
            'Open time',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume',
            'Close time',
            'Quote volume',
            'Number of trades',
            'Taker base volume',
            'Taker quote volume',
            'Ignore'
        ])
        df.index = pd.to_datetime(df['Open time'], unit='ms', utc=True)
        del df['Open time']
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)
        df['Close time'] = pd.to_datetime(df['Close time'], unit='ms', utc=True)
        df['Quote volume'] = df['Quote volume'].astype(float)
        df['Number of trades'] = df['Number of trades'].astype(int)
        df['Taker base volume'] = df['Taker base volume'].astype(float)
        df['Taker quote volume'] = df['Taker quote volume'].astype(float)
        del df['Ignore']

        return df

    def update_symbol(self, symbol, **kwargs):
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `BinanceData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        download_kwargs['show_progress'] = False
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, **kwargs)


class CCXTData(Data):
    """`Data` for data coming from `ccxt`.

    ## Example

    Fetch the 1-minute data of the last 2 hours, wait 1 minute, and update:

    ```python-repl
    >>> import vectorbt as vbt

    >>> ccxt_data = vbt.CCXTData.download(
    ...     "BTC/USDT",
    ...     start='2 hours ago UTC',
    ...     end='now UTC',
    ...     timeframe='1m'
    ... )
    >>> ccxt_data.get()
    2021-03-22 21:08:45.826000+00:00 - 2021-03-22 23:08:00+00:00: : 1it [00:00,  1.96it/s]

                                   Open      High       Low     Close      Volume
    Open time
    2021-03-22 21:09:00+00:00  54840.01  54843.01  54752.86  54819.23   34.311598
    2021-03-22 21:10:00+00:00  54819.23  54868.27  54818.05  54857.81   33.024841
    2021-03-22 21:11:00+00:00  54853.90  54917.13  54848.26  54915.80   30.612839
    ...                             ...       ...       ...       ...         ...
    2021-03-22 23:06:00+00:00  54802.16  54871.14  54802.16  54850.00   67.792298
    2021-03-22 23:07:00+00:00  54850.00  54856.23  54771.89  54816.47   63.103731
    2021-03-22 23:08:00+00:00  54816.45  54826.41  54733.86  54765.74   48.825476

    [120 rows x 5 columns]

    >>> import time
    >>> time.sleep(60)

    >>> ccxt_data = ccxt_data.update()
    >>> ccxt_data.get()
                                   Open      High       Low     Close      Volume
    Open time
    2021-03-22 21:09:00+00:00  54840.01  54843.01  54752.86  54819.23   34.311598
    2021-03-22 21:10:00+00:00  54819.23  54868.27  54818.05  54857.81   33.024841
    2021-03-22 21:11:00+00:00  54853.90  54917.13  54848.26  54915.80   30.612839
    ...                             ...       ...       ...       ...         ...
    2021-03-22 23:07:00+00:00  54850.00  54856.23  54771.89  54816.47   63.103731
    2021-03-22 23:08:00+00:00  54816.45  54826.41  54733.86  54777.12   74.730137
    2021-03-22 23:09:00+00:00  54777.12  54869.48  54770.55  54827.52   45.687450

    [121 rows x 5 columns]
    ```"""

    @classmethod
    def download_symbol(cls, symbol, exchange='binance', config=None, timeframe='1d', start=0,
                        end='now UTC', delay=None, limit=500, retries=3, show_progress=True, params=None):
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            exchange (str or object): Exchange identifier or an exchange object of type
                `ccxt.base.exchange.Exchange`.
            config (dict): Config passed to the exchange upon instantiation.

                Overrides settings under `ccxt` in `vectorbt.settings.data`.

                Will raise an exception if exchange has been already instantiated.
            timeframe (str): Timeframe supported by the exchange.
            start (any): Start datetime.

                See `vectorbt.utils.datetime.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime.to_tzaware_datetime`.
            delay (int or float): Time to sleep after each request (in milliseconds).

                !!! note
                    Use only if `enableRateLimit` is not set.
            limit (int): The maximum number of returned items.
            retries (int): The number of retries on failure to fetch data.
            show_progress (bool): Whether to show the progress bar.
            params (dict): Exchange-specific key-value parameters.
        """
        import ccxt
        from vectorbt import settings

        if config is None:
            config = {}
        if params is None:
            params = {}
        if isinstance(exchange, str):
            if not hasattr(ccxt, exchange):
                raise ValueError(f"Exchange {exchange} not found")
            # Resolve config
            default_config = {}
            for k, v in settings.data['ccxt'].items():
                # Get general (not per exchange) settings
                if k in ccxt.exchanges:
                    continue
                default_config[k] = v
            if exchange in settings.data['ccxt']:
                default_config = merge_dicts(default_config, settings.data['ccxt'][exchange])
            config = merge_dicts(default_config, config)
            exchange = getattr(ccxt, exchange)(config)
        else:
            if len(config) > 0:
                raise ValueError("Cannot apply config after instantiation of the exchange")
        if not exchange.has['fetchOHLCV']:
            raise ValueError(f"Exchange {exchange} does not support OHLCV")
        if timeframe not in exchange.timeframes:
            raise ValueError(f"Exchange {exchange} does not support {timeframe} timeframe")
        if exchange.has['fetchOHLCV'] == 'emulated':
            warnings.warn("Using emulated OHLCV candles", stacklevel=2)

        def _retry(method):
            @wraps(method)
            def retry_method(*args, **kwargs):
                for i in range(retries):
                    try:
                        return method(*args, **kwargs)
                    except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                        if i == retries - 1:
                            raise e
                    if delay is not None:
                        time.sleep(delay / 1000)

            return retry_method

        @_retry
        def _fetch(_since, _limit):
            return exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                since=_since,
                limit=_limit,
                params=params
            )

        # Establish the timestamps
        start_ts = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        try:
            first_data = _fetch(0, 1)
            first_valid_ts = first_data[0][0]
            next_start_ts = start_ts = max(start_ts, first_valid_ts)
        except:
            next_start_ts = start_ts
        end_ts = datetime_to_ms(to_tzaware_datetime(end, tz=get_utc_tz()))

        def _ts_to_str(ts):
            return str(pd.Timestamp(to_tzaware_datetime(ts, tz=get_utc_tz())))

        # Iteratively collect the data
        data = []
        with tqdm(disable=not show_progress) as pbar:
            pbar.set_description(_ts_to_str(start_ts))
            while True:
                # Fetch the klines for the next interval
                next_data = _fetch(next_start_ts, limit)
                if len(data) > 0:
                    next_data = list(filter(lambda d: next_start_ts < d[0] < end_ts, next_data))
                else:
                    next_data = list(filter(lambda d: d[0] < end_ts, next_data))

                # Update the timestamps and the progress bar
                if not len(next_data):
                    break
                data += next_data
                pbar.set_description("{} - {}".format(
                    _ts_to_str(start_ts),
                    _ts_to_str(next_data[-1][0])
                ))
                pbar.update(1)
                next_start_ts = next_data[-1][0]
                if delay is not None:
                    time.sleep(delay / 1000)  # be kind to api

        # Convert data to a DataFrame
        df = pd.DataFrame(data, columns=[
            'Open time',
            'Open',
            'High',
            'Low',
            'Close',
            'Volume'
        ])
        df.index = pd.to_datetime(df['Open time'], unit='ms', utc=True)
        del df['Open time']
        df['Open'] = df['Open'].astype(float)
        df['High'] = df['High'].astype(float)
        df['Low'] = df['Low'].astype(float)
        df['Close'] = df['Close'].astype(float)
        df['Volume'] = df['Volume'].astype(float)

        return df

    def update_symbol(self, symbol, **kwargs):
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `CCXTData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        download_kwargs['show_progress'] = False
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, **kwargs)