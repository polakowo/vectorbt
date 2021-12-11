# Copyright (c) 2021 Oleg Polakow. All rights reserved.
# This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

"""Custom data classes that subclass `vectorbt.data.base.Data`."""

import time
import warnings
from functools import wraps

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from vectorbt import _typing as tp
from vectorbt.data.base import Data
from vectorbt.utils.config import merge_dicts, get_func_kwargs
from vectorbt.utils.datetime_ import (
    get_utc_tz,
    get_local_tz,
    to_tzaware_datetime,
    datetime_to_ms
)

try:
    from binance.client import Client as ClientT
except ImportError:
    ClientT = tp.Any
try:
    from ccxt.base.exchange import Exchange as ExchangeT
except ImportError:
    ExchangeT = tp.Any


class SyntheticData(Data):
    """`Data` for synthetically generated data."""

    @classmethod
    def generate_symbol(cls, symbol: tp.Label, index: tp.Index, **kwargs) -> tp.SeriesFrame:
        """Abstract method to generate a symbol."""
        raise NotImplementedError

    @classmethod
    def download_symbol(cls,
                        symbol: tp.Label,
                        start: tp.DatetimeLike = 0,
                        end: tp.DatetimeLike = 'now',
                        freq: tp.Union[None, str, pd.DateOffset] = None,
                        date_range_kwargs: tp.KwargsLike = None,
                        **kwargs) -> tp.SeriesFrame:
        """Download the symbol.

        Generates datetime index and passes it to `SyntheticData.generate_symbol` to fill
        the Series/DataFrame with generated data."""
        if date_range_kwargs is None:
            date_range_kwargs = {}
        index = pd.date_range(
            start=to_tzaware_datetime(start, tz=get_utc_tz()),
            end=to_tzaware_datetime(end, tz=get_utc_tz()),
            freq=freq,
            **date_range_kwargs
        )
        if len(index) == 0:
            raise ValueError("Date range is empty")
        return cls.generate_symbol(symbol, index, **kwargs)

    def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.SeriesFrame:
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `SyntheticData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, **kwargs)


def generate_gbm_paths(S0: float, mu: float, sigma: float, T: int, M: int, I: int,
                       seed: tp.Optional[int] = None) -> tp.Array2d:
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
    2021-05-02 14:14:15.182089+00:00    102.386605
    2021-05-02 14:15:15.182089+00:00    101.554203
    2021-05-02 14:16:15.182089+00:00    104.765771
    ...                                        ...
    2021-05-02 16:12:15.182089+00:00     51.614839
    2021-05-02 16:13:15.182089+00:00     53.525376
    2021-05-02 16:14:15.182089+00:00     55.615250
    Freq: T, Length: 121, dtype: float64

    >>> import time
    >>> time.sleep(60)

    >>> gbm_data = gbm_data.update()
    >>> gbm_data.get()
    2021-05-02 14:14:15.182089+00:00    102.386605
    2021-05-02 14:15:15.182089+00:00    101.554203
    2021-05-02 14:16:15.182089+00:00    104.765771
    ...                                        ...
    2021-05-02 16:13:15.182089+00:00     53.525376
    2021-05-02 16:14:15.182089+00:00     51.082220
    2021-05-02 16:15:15.182089+00:00     54.725304
    Freq: T, Length: 122, dtype: float64
    ```"""

    @classmethod
    def generate_symbol(cls,
                        symbol: tp.Label,
                        index: tp.Index,
                        S0: float = 100.,
                        mu: float = 0.,
                        sigma: float = 0.05,
                        T: tp.Optional[int] = None,
                        I: int = 1,
                        seed: tp.Optional[int] = None) -> tp.SeriesFrame:
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

    def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.SeriesFrame:
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

    Stocks are usually in the timezone "+0500" and cryptocurrencies in UTC.

    !!! warning
        Data coming from Yahoo is not the most stable data out there. Yahoo may manipulate data
        how they want, add noise, return missing data points (see volume in the example below), etc.
        It's only used in vectorbt for demonstration purposes.

    ## Example

    Fetch the business day except the last 5 minutes of trading data, and then update with the missing 5 minutes:

    ```python-repl
    >>> import vectorbt as vbt

    >>> yf_data = vbt.YFData.download(
    ...     "TSLA",
    ...     start='2021-04-12 09:30:00 -0400',
    ...     end='2021-04-12 09:35:00 -0400',
    ...     interval='1m'
    ... )
    >>> yf_data.get())
                                     Open        High         Low       Close  \\
    Datetime
    2021-04-12 13:30:00+00:00  685.080017  685.679993  684.765015  685.679993
    2021-04-12 13:31:00+00:00  684.625000  686.500000  684.010010  685.500000
    2021-04-12 13:32:00+00:00  685.646790  686.820007  683.190002  686.455017
    2021-04-12 13:33:00+00:00  686.455017  687.000000  685.000000  685.565002
    2021-04-12 13:34:00+00:00  685.690002  686.400024  683.200012  683.715027

                               Volume  Dividends  Stock Splits
    Datetime
    2021-04-12 13:30:00+00:00       0          0             0
    2021-04-12 13:31:00+00:00  152276          0             0
    2021-04-12 13:32:00+00:00  168363          0             0
    2021-04-12 13:33:00+00:00  129607          0             0
    2021-04-12 13:34:00+00:00  134620          0             0

    >>> yf_data = yf_data.update(end='2021-04-12 09:40:00 -0400')
    >>> yf_data.get()
                                     Open        High         Low       Close  \\
    Datetime
    2021-04-12 13:30:00+00:00  685.080017  685.679993  684.765015  685.679993
    2021-04-12 13:31:00+00:00  684.625000  686.500000  684.010010  685.500000
    2021-04-12 13:32:00+00:00  685.646790  686.820007  683.190002  686.455017
    2021-04-12 13:33:00+00:00  686.455017  687.000000  685.000000  685.565002
    2021-04-12 13:34:00+00:00  685.690002  686.400024  683.200012  683.715027
    2021-04-12 13:35:00+00:00  683.604980  684.340027  682.760071  684.135010
    2021-04-12 13:36:00+00:00  684.130005  686.640015  683.333984  686.563904
    2021-04-12 13:37:00+00:00  686.530029  688.549988  686.000000  686.635010
    2021-04-12 13:38:00+00:00  686.593201  689.500000  686.409973  688.179993
    2021-04-12 13:39:00+00:00  688.500000  689.347595  687.710022  688.070007

                               Volume  Dividends  Stock Splits
    Datetime
    2021-04-12 13:30:00+00:00       0          0             0
    2021-04-12 13:31:00+00:00  152276          0             0
    2021-04-12 13:32:00+00:00  168363          0             0
    2021-04-12 13:33:00+00:00  129607          0             0
    2021-04-12 13:34:00+00:00       0          0             0
    2021-04-12 13:35:00+00:00  110500          0             0
    2021-04-12 13:36:00+00:00  148384          0             0
    2021-04-12 13:37:00+00:00  243851          0             0
    2021-04-12 13:38:00+00:00  203569          0             0
    2021-04-12 13:39:00+00:00   93308          0             0
    ```
    """

    @classmethod
    def download_symbol(cls,
                        symbol: tp.Label,
                        period: str = 'max',
                        start: tp.Optional[tp.DatetimeLike] = None,
                        end: tp.Optional[tp.DatetimeLike] = None,
                        **kwargs) -> tp.Frame:
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            period (str): Period.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            **kwargs: Keyword arguments passed to `yfinance.base.TickerBase.history`.
        """
        import yfinance as yf

        # yfinance still uses mktime, which assumes that the passed date is in local time
        if start is not None:
            start = to_tzaware_datetime(start, tz=get_local_tz())
        if end is not None:
            end = to_tzaware_datetime(end, tz=get_local_tz())

        return yf.Ticker(symbol).history(period=period, start=start, end=end, **kwargs)

    def update_symbol(self, symbol: tp.Label, **kwargs) -> tp.Frame:
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `YFData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, **kwargs)


BinanceDataT = tp.TypeVar("BinanceDataT", bound="BinanceData")


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
    2021-05-02 14:47:20.478000+00:00 - 2021-05-02 16:47:00+00:00: : 1it [00:00,  3.42it/s]
                                   Open      High       Low     Close     Volume  \\
    Open time
    2021-05-02 14:48:00+00:00  56867.44  56913.57  56857.40  56913.56  28.709976
    2021-05-02 14:49:00+00:00  56913.56  56913.57  56845.94  56888.00  19.734841
    2021-05-02 14:50:00+00:00  56888.00  56947.32  56879.78  56934.71  23.150163
    ...                             ...       ...       ...       ...        ...
    2021-05-02 16:45:00+00:00  56664.13  56666.77  56641.11  56644.03  40.852719
    2021-05-02 16:46:00+00:00  56644.02  56663.43  56605.17  56605.18  27.573654
    2021-05-02 16:47:00+00:00  56605.18  56657.55  56605.17  56627.12   7.719933

                                                    Close time  Quote volume  \\
    Open time
    2021-05-02 14:48:00+00:00 2021-05-02 14:48:59.999000+00:00  1.633534e+06
    2021-05-02 14:49:00+00:00 2021-05-02 14:49:59.999000+00:00  1.122519e+06
    2021-05-02 14:50:00+00:00 2021-05-02 14:50:59.999000+00:00  1.317969e+06
    ...                                                    ...           ...
    2021-05-02 16:45:00+00:00 2021-05-02 16:45:59.999000+00:00  2.314579e+06
    2021-05-02 16:46:00+00:00 2021-05-02 16:46:59.999000+00:00  1.561548e+06
    2021-05-02 16:47:00+00:00 2021-05-02 16:47:59.999000+00:00  4.371848e+05

                               Number of trades  Taker base volume  \\
    Open time
    2021-05-02 14:48:00+00:00               991          13.771152
    2021-05-02 14:49:00+00:00               816           5.981942
    2021-05-02 14:50:00+00:00              1086          10.813757
    ...                                     ...                ...
    2021-05-02 16:45:00+00:00              1006          18.106933
    2021-05-02 16:46:00+00:00               916          14.869411
    2021-05-02 16:47:00+00:00               353           3.903321

                               Taker quote volume
    Open time
    2021-05-02 14:48:00+00:00        7.835391e+05
    2021-05-02 14:49:00+00:00        3.402170e+05
    2021-05-02 14:50:00+00:00        6.156418e+05
    ...                                       ...
    2021-05-02 16:45:00+00:00        1.025892e+06
    2021-05-02 16:46:00+00:00        8.421173e+05
    2021-05-02 16:47:00+00:00        2.210323e+05

    [120 rows x 10 columns]

    >>> import time
    >>> time.sleep(60)

    >>> binance_data = binance_data.update()
    >>> binance_data.get()
                                   Open      High       Low     Close     Volume  \\
    Open time
    2021-05-02 14:48:00+00:00  56867.44  56913.57  56857.40  56913.56  28.709976
    2021-05-02 14:49:00+00:00  56913.56  56913.57  56845.94  56888.00  19.734841
    2021-05-02 14:50:00+00:00  56888.00  56947.32  56879.78  56934.71  23.150163
    ...                             ...       ...       ...       ...        ...
    2021-05-02 16:46:00+00:00  56644.02  56663.43  56605.17  56605.18  27.573654
    2021-05-02 16:47:00+00:00  56605.18  56657.55  56605.17  56625.76  14.615437
    2021-05-02 16:48:00+00:00  56625.75  56643.60  56614.32  56623.01   5.895843

                                                    Close time  Quote volume  \\
    Open time
    2021-05-02 14:48:00+00:00 2021-05-02 14:48:59.999000+00:00  1.633534e+06
    2021-05-02 14:49:00+00:00 2021-05-02 14:49:59.999000+00:00  1.122519e+06
    2021-05-02 14:50:00+00:00 2021-05-02 14:50:59.999000+00:00  1.317969e+06
    ...                                                    ...           ...
    2021-05-02 16:46:00+00:00 2021-05-02 16:46:59.999000+00:00  1.561548e+06
    2021-05-02 16:47:00+00:00 2021-05-02 16:47:59.999000+00:00  8.276017e+05
    2021-05-02 16:48:00+00:00 2021-05-02 16:48:59.999000+00:00  3.338702e+05

                               Number of trades  Taker base volume  \\
    Open time
    2021-05-02 14:48:00+00:00               991          13.771152
    2021-05-02 14:49:00+00:00               816           5.981942
    2021-05-02 14:50:00+00:00              1086          10.813757
    ...                                     ...                ...
    2021-05-02 16:46:00+00:00               916          14.869411
    2021-05-02 16:47:00+00:00               912           7.778489
    2021-05-02 16:48:00+00:00               308           2.358130

                               Taker quote volume
    Open time
    2021-05-02 14:48:00+00:00        7.835391e+05
    2021-05-02 14:49:00+00:00        3.402170e+05
    2021-05-02 14:50:00+00:00        6.156418e+05
    ...                                       ...
    2021-05-02 16:46:00+00:00        8.421173e+05
    2021-05-02 16:47:00+00:00        4.404362e+05
    2021-05-02 16:48:00+00:00        1.335474e+05

    [121 rows x 10 columns]
    ```"""

    @classmethod
    def download(cls: tp.Type[BinanceDataT],
                 symbols: tp.Labels,
                 client: tp.Optional["ClientT"] = None,
                 **kwargs) -> BinanceDataT:
        """Override `vectorbt.data.base.Data.download` to instantiate a Binance client."""
        from binance.client import Client
        from vectorbt._settings import settings
        binance_cfg = settings['data']['binance']

        client_kwargs = dict()
        for k in get_func_kwargs(Client):
            if k in kwargs:
                client_kwargs[k] = kwargs.pop(k)
        client_kwargs = merge_dicts(binance_cfg, client_kwargs)
        if client is None:
            client = Client(**client_kwargs)
        return super(BinanceData, cls).download(symbols, client=client, **kwargs)

    @classmethod
    def download_symbol(cls,
                        symbol: str,
                        client: tp.Optional["ClientT"] = None,
                        interval: str = '1d',
                        start: tp.DatetimeLike = 0,
                        end: tp.DatetimeLike = 'now UTC',
                        delay: tp.Optional[float] = 500,
                        limit: int = 500,
                        show_progress: bool = True,
                        tqdm_kwargs: tp.KwargsLike = None) -> tp.Frame:
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            client (binance.client.Client): Binance client of type `binance.client.Client`.
            interval (str): Kline interval.

                See `binance.enums`.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            delay (float): Time to sleep after each request (in milliseconds).
            limit (int): The maximum number of returned items.
            show_progress (bool): Whether to show the progress bar.
            tqdm_kwargs (dict): Keyword arguments passed to `tqdm`.

        For defaults, see `data.binance` in `vectorbt._settings.settings`.
        """
        if client is None:
            raise ValueError("client must be provided")

        if tqdm_kwargs is None:
            tqdm_kwargs = {}
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

        def _ts_to_str(ts: tp.DatetimeLike) -> str:
            return str(pd.Timestamp(to_tzaware_datetime(ts, tz=get_utc_tz())))

        # Iteratively collect the data
        data: tp.List[list] = []
        with tqdm(disable=not show_progress, **tqdm_kwargs) as pbar:
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

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
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
    2021-05-02 14:50:26.305000+00:00 - 2021-05-02 16:50:00+00:00: : 1it [00:00,  1.96it/s]
                                   Open      High       Low     Close     Volume
    Open time
    2021-05-02 14:51:00+00:00  56934.70  56964.59  56910.00  56948.99  22.158319
    2021-05-02 14:52:00+00:00  56948.99  56999.00  56940.04  56977.62  46.958464
    2021-05-02 14:53:00+00:00  56977.61  56987.09  56882.98  56885.42  27.752200
    ...                             ...       ...       ...       ...        ...
    2021-05-02 16:48:00+00:00  56625.75  56643.60  56595.47  56596.01  15.452510
    2021-05-02 16:49:00+00:00  56596.00  56664.14  56596.00  56640.35  12.777475
    2021-05-02 16:50:00+00:00  56640.35  56675.82  56640.35  56670.65   6.882321

    [120 rows x 5 columns]

    >>> import time
    >>> time.sleep(60)

    >>> ccxt_data = ccxt_data.update()
    >>> ccxt_data.get()
                                   Open      High       Low     Close     Volume
    Open time
    2021-05-02 14:51:00+00:00  56934.70  56964.59  56910.00  56948.99  22.158319
    2021-05-02 14:52:00+00:00  56948.99  56999.00  56940.04  56977.62  46.958464
    2021-05-02 14:53:00+00:00  56977.61  56987.09  56882.98  56885.42  27.752200
    ...                             ...       ...       ...       ...        ...
    2021-05-02 16:49:00+00:00  56596.00  56664.14  56596.00  56640.35  12.777475
    2021-05-02 16:50:00+00:00  56640.35  56689.99  56640.35  56678.33  14.610231
    2021-05-02 16:51:00+00:00  56678.33  56688.99  56636.89  56653.42  11.647158

    [121 rows x 5 columns]
    ```"""

    @classmethod
    def download_symbol(cls,
                        symbol: str,
                        exchange: tp.Union[str, "ExchangeT"] = 'binance',
                        config: tp.Optional[dict] = None,
                        timeframe: str = '1d',
                        start: tp.DatetimeLike = 0,
                        end: tp.DatetimeLike = 'now UTC',
                        delay: tp.Optional[float] = None,
                        limit: tp.Optional[int] = 500,
                        retries: int = 3,
                        show_progress: bool = True,
                        params: tp.Optional[dict] = None,
                        tqdm_kwargs: tp.KwargsLike = None) -> tp.Frame:
        """Download the symbol.

        Args:
            symbol (str): Symbol.
            exchange (str or object): Exchange identifier or an exchange object of type
                `ccxt.base.exchange.Exchange`.
            config (dict): Config passed to the exchange upon instantiation.

                Will raise an exception if exchange has been already instantiated.
            timeframe (str): Timeframe supported by the exchange.
            start (any): Start datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            end (any): End datetime.

                See `vectorbt.utils.datetime_.to_tzaware_datetime`.
            delay (float): Time to sleep after each request (in milliseconds).

                !!! note
                    Use only if `enableRateLimit` is not set.
            limit (int): The maximum number of returned items.
            retries (int): The number of retries on failure to fetch data.
            show_progress (bool): Whether to show the progress bar.
            tqdm_kwargs (dict): Keyword arguments passed to `tqdm`.
            params (dict): Exchange-specific key-value parameters.

        For defaults, see `data.ccxt` in `vectorbt._settings.settings`.
        """
        import ccxt
        from vectorbt._settings import settings
        ccxt_cfg = settings['data']['ccxt']

        if config is None:
            config = {}
        if tqdm_kwargs is None:
            tqdm_kwargs = {}
        if params is None:
            params = {}
        if isinstance(exchange, str):
            if not hasattr(ccxt, exchange):
                raise ValueError(f"Exchange {exchange} not found")
            # Resolve config
            default_config = {}
            for k, v in ccxt_cfg.items():
                # Get general (not per exchange) settings
                if k in ccxt.exchanges:
                    continue
                default_config[k] = v
            if exchange in ccxt_cfg:
                default_config = merge_dicts(default_config, ccxt_cfg[exchange])
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
        data: tp.List[list] = []
        with tqdm(disable=not show_progress, **tqdm_kwargs) as pbar:
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

    def update_symbol(self, symbol: str, **kwargs) -> tp.Frame:
        """Update the symbol.

        `**kwargs` will override keyword arguments passed to `CCXTData.download_symbol`."""
        download_kwargs = self.select_symbol_kwargs(symbol, self.download_kwargs)
        download_kwargs['start'] = self.data[symbol].index[-1]
        download_kwargs['show_progress'] = False
        kwargs = merge_dicts(download_kwargs, kwargs)
        return self.download_symbol(symbol, **kwargs)
