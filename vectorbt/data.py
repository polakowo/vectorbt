from datetime import datetime, timedelta
from timeit import default_timer as timer

import pandas as pd
import pytz
from poloniex import Poloniex


# Load pair
###########


def now_dt():
    """Current datetime"""
    return pytz.utc.localize(datetime.utcnow())


def ago_dt(**kwargs):
    """Time ago"""
    return now_dt() - timedelta(**kwargs)


def dt_to_ts(date):
    """Date to timestamp"""
    return int(date.timestamp())


def load_cryptopair(pair, from_dt, to_dt, period=300):
    """Load OHLC data on a cryptocurrency pair from Poloniex exchange"""
    polo = Poloniex()
    t = timer()
    chart_data = polo.returnChartData(pair, period=period, start=dt_to_ts(from_dt), end=dt_to_ts(to_dt))
    print("done. %.2fs" % (timer() - t))
    chart_df = pd.DataFrame(chart_data)
    chart_df.set_index('date', drop=True, inplace=True)
    chart_df.index = pd.to_datetime(chart_df.index, unit='s')
    chart_df.fillna(method='ffill', inplace=True) # fill gaps forwards
    chart_df.fillna(method='bfill', inplace=True) # fill gaps backwards
    chart_df = chart_df.astype(float)
    chart_df.rename(columns={'open': 'O', 'high': 'H', 'low': 'L', 'close': 'C', 'volume': 'V'}, inplace=True)
    return chart_df[['O', 'H', 'L', 'C', 'V']]
