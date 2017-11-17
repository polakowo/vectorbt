from datetime import datetime, timedelta
from timeit import default_timer as timer

import pandas as pd
import pytz
from poloniex import Poloniex


# Load pair
###########


def now_dt():
    # Current datetime
    return pytz.utc.localize(datetime.utcnow())


def ago_dt(**kwargs):
    return now_dt() - timedelta(**kwargs)


def dt_to_ts(date):
    # Date to timestamp
    return int(date.timestamp())


def ts_to_dt(ts):
    # Timestamp to date
    return datetime.fromtimestamp(ts, )


def load_cryptopair(pair, from_dt, to_dt, period=300):
    # Load OHLC data on a cryptocurrency pair from Poloniex exchange
    polo = Poloniex()
    t = timer()
    chart_data = polo.returnChartData(pair, period=period, start=dt_to_ts(from_dt), end=dt_to_ts(to_dt))
    print('passed. %.2fs' % (timer() - t))
    chart_df = pd.DataFrame(chart_data)
    chart_df.set_index('date', drop=True, inplace=True)
    chart_df.index = pd.to_datetime(chart_df.index, unit='s')
    chart_df.dropna(inplace=True)
    chart_df = chart_df.astype(float)
    return chart_df
