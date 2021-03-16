"""Custom data classes that subclass `vectorbt.data.base.Data`."""

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import math

from vectorbt.utils.datetime import (
    get_utc_tz,
    get_local_tz,
    to_tzaware_datetime,
    datetime_to_ms,
    interval_to_ms
)
from vectorbt.utils.config import merge_dicts
from vectorbt.data.base import Data


class SyntheticData(Data):
    """`Data` for synthetically generated data."""

    @classmethod
    def generator(cls, symbol, index, **kwargs):
        """Generates data based on symbol and datetime-like index."""
        raise NotImplementedError

    @classmethod
    def downloader(cls, symbol, start=0, end='now', freq=None, date_range_kwargs=None, **kwargs):
        """Downloader for `SyntheticData`."""
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
        return cls.generator(symbol, index, **kwargs)

    def updater(self, symbol, **kwargs):
        """Updater for `SyntheticData`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.downloader(symbol, **kwargs)


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
    >>> import time
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
    def generator(cls, symbol, index, S0=100., mu=0., sigma=0.05, T=None, I=1, seed=None):
        """Generator that uses `generate_gbm_paths`."""
        if T is None:
            T = len(index)
        out = generate_gbm_paths(S0, mu, sigma, T, len(index), I, seed=seed)[1:]
        if out.shape[1] == 1:
            return pd.Series(out[:, 0], index=index)
        columns = pd.RangeIndex(stop=out.shape[1], name='path')
        return pd.DataFrame(out, index=index, columns=columns)

    def updater(self, symbol, **kwargs):
        """Updater for `GBMData`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        _ = download_kwargs.pop('S0', None)
        S0 = self.data[symbol].iloc[-2]
        _ = download_kwargs.pop('T', None)
        download_kwargs['seed'] = None
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.downloader(symbol, S0=S0, **kwargs)


class YFData(Data):
    """`Data` for data coming from `yfinance`.

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
                                     Open        High         Low       Close  \
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
                                     Open        High         Low       Close  \
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
    def downloader(cls, symbol, period='max', start=None, end=None, **kwargs):
        """Downloader for `YFData`."""
        import yfinance as yf

        # yfinance still uses mktime, which assumes that the passed date is in local time
        if start is not None:
            start = to_tzaware_datetime(start, tz=get_local_tz())
        if end is not None:
            end = to_tzaware_datetime(end, tz=get_local_tz())

        return yf.Ticker(symbol).history(period=period, start=start, end=end, **kwargs)

    def updater(self, symbol, **kwargs):
        """Updater for `YFData`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.downloader(symbol, **kwargs)


class BinanceData(Data):
    """`Data` for data coming from `python-binance`.

    ## Example

    Fetch the 1-minute data of the last 2 hours, wait 1 minute, and update:

    ```python-repl
    >>> import time
    >>> import vectorbt as vbt
    >>> from binance.client import Client

    >>> client = Client("YOUR_API_KEY", "YOUR_API_SECRET")

    >>> binance_data = vbt.BinanceData.download(
    ...     "BTCUSDT",
    ...     client=client,
    ...     start='2 hours ago UTC',
    ...     end='now UTC',
    ...     interval=Client.KLINE_INTERVAL_1MINUTE
    ... )
    >>> binance_data.get()
                                   Open      High       Low     Close      Volume  \
    Open time
    2021-03-15 16:15:00+00:00  56605.70  56605.70  56435.99  56518.06  106.451053
    2021-03-15 16:16:00+00:00  56518.06  56530.56  56418.91  56426.89   56.294601
    2021-03-15 16:17:00+00:00  56426.90  56514.62  56413.29  56493.42   41.278765
    ...                             ...       ...       ...       ...         ...
    2021-03-15 18:12:00+00:00  56007.56  56094.42  56001.46  56078.43   19.070416
    2021-03-15 18:13:00+00:00  56088.13  56100.00  56027.87  56050.15   42.997610
    2021-03-15 18:14:00+00:00  56050.15  56100.00  56040.69  56093.41    9.698548

                               Quote volume  Number of trades  Taker base volume  \
    Open time
    2021-03-15 16:15:00+00:00  6.017505e+06              3125          43.344499
    2021-03-15 16:16:00+00:00  3.179439e+06              1939          23.623797
    2021-03-15 16:17:00+00:00  2.330885e+06              2413          19.640592
    ...                                 ...               ...                ...
    2021-03-15 18:12:00+00:00  1.068856e+06               884          11.105932
    2021-03-15 18:13:00+00:00  2.410950e+06               915          17.350580
    2021-03-15 18:14:00+00:00  5.437963e+05               283           4.227839

                               Taker quote volume
    Open time
    2021-03-15 16:15:00+00:00        2.450125e+06
    2021-03-15 16:16:00+00:00        1.334227e+06
    2021-03-15 16:17:00+00:00        1.108966e+06
    ...                                       ...
    2021-03-15 18:12:00+00:00        6.224484e+05
    2021-03-15 18:13:00+00:00        9.728524e+05
    2021-03-15 18:14:00+00:00        2.370699e+05

    [120 rows x 9 columns]

    >>> time.sleep(60)

    >>> binance_data = binance_data.update()
    >>> binance_data.get()
                                   Open      High       Low     Close      Volume  \
    Open time
    2021-03-15 16:15:00+00:00  56605.70  56605.70  56435.99  56518.06  106.451053
    2021-03-15 16:16:00+00:00  56518.06  56530.56  56418.91  56426.89   56.294601
    2021-03-15 16:17:00+00:00  56426.90  56514.62  56413.29  56493.42   41.278765
    ...                             ...       ...       ...       ...         ...
    2021-03-15 18:13:00+00:00  56088.13  56100.00  56027.87  56050.15   42.997610
    2021-03-15 18:14:00+00:00  56050.15  56100.00  56003.94  56024.22   34.441996
    2021-03-15 18:15:00+00:00  56028.62  56028.64  55934.48  55977.10   18.174774

                               Quote volume  Number of trades  Taker base volume  \
    Open time
    2021-03-15 16:15:00+00:00  6.017505e+06              3125          43.344499
    2021-03-15 16:16:00+00:00  3.179439e+06              1939          23.623797
    2021-03-15 16:17:00+00:00  2.330885e+06              2413          19.640592
    ...                                 ...               ...                ...
    2021-03-15 18:13:00+00:00  2.410950e+06               915          17.350580
    2021-03-15 18:14:00+00:00  1.930522e+06               938          12.175991
    2021-03-15 18:15:00+00:00  1.017411e+06               659           7.057494

                               Taker quote volume
    Open time
    2021-03-15 16:15:00+00:00        2.450125e+06
    2021-03-15 16:16:00+00:00        1.334227e+06
    2021-03-15 16:17:00+00:00        1.108966e+06
    ...                                       ...
    2021-03-15 18:13:00+00:00        9.728524e+05
    2021-03-15 18:14:00+00:00        6.824422e+05
    2021-03-15 18:15:00+00:00        3.950199e+05

    [121 rows x 9 columns]
    ```"""

    @classmethod
    def downloader(cls, symbol, client=None, interval=None, start=0, end='now UTC',
                   limit=500, sleep_each=3, show_progress=False, **kwargs):
        """Downloader for `BinanceData`."""
        if client is None:
            raise ValueError("client is required")

        data = []

        # Establish the timestamps
        if interval is None:
            interval = client.KLINE_INTERVAL_1DAY
        timeframe = interval_to_ms(interval)
        start = datetime_to_ms(to_tzaware_datetime(start, tz=get_utc_tz()))
        kline = client.get_klines(
            symbol=symbol,
            interval=interval,
            limit=1,
            startTime=0,
            endTime=None
        )
        first_valid_ts = kline[0][0]
        start = max(start, first_valid_ts)
        end = datetime_to_ms(to_tzaware_datetime(end, tz=get_utc_tz()))

        # Iteratively collect the data
        idx = 1
        with tqdm(total=math.ceil((end - start) / timeframe), disable=not show_progress) as pbar:
            while True:
                # Fetch the klines for the next interval
                temp_data = client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit,
                    startTime=start,
                    endTime=end
                )

                # Update the timestamps and the progress bar
                if not len(temp_data):
                    break
                data += temp_data
                if len(temp_data) < limit:
                    pbar.update(math.ceil((temp_data[-1][6] - start) / timeframe))
                    break
                next_start = temp_data[-1][0] + timeframe
                pbar.update(math.ceil((next_start - start) / timeframe))
                start = next_start
                if idx % sleep_each == 0:
                    time.sleep(1)  # be kind to api
                idx += 1

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
        del df['Close time']
        df['Quote volume'] = df['Quote volume'].astype(float)
        df['Number of trades'] = df['Number of trades'].astype(int)
        df['Taker base volume'] = df['Taker base volume'].astype(float)
        df['Taker quote volume'] = df['Taker quote volume'].astype(float)
        del df['Ignore']

        return df

    def updater(self, symbol, **kwargs):
        """Updater for `BinanceData`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.downloader(symbol, **kwargs)
