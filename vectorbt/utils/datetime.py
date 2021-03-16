"""Utilities for working with dates and time."""

import numpy as np
import pandas as pd
import dateparser
from datetime import datetime, timezone

from vectorbt.utils import checks

DatetimeTypes = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)


def freq_delta(freq):
    """Return delta of frequency."""
    if isinstance(freq, str) and not freq[0].isdigit():
        # Otherwise "ValueError: unit abbreviation w/o a number"
        freq = '1' + freq
    return pd.to_timedelta(freq)


def to_time_units(obj, freq):
    """Multiply each element with `freq_delta` to get result in time units."""
    if not checks.is_any_array(obj):
        obj = np.asarray(obj)
    return obj * freq_delta(freq)


def get_utc_tz():
    """Get UTC timezone."""
    return timezone.utc


def get_local_tz():
    """Get local timezone."""
    return timezone(datetime.now(timezone.utc).astimezone().utcoffset())


def to_tzaware_datetime(dt, tz=None):
    """Convert value to a timezone-aware `datetime.datetime`.

    Timestamps are localized to UTC, while naive datetime is localized to the local time.
    To explicitly convert the datetime to a timezone, use `tz`."""
    if isinstance(dt, float):
        dt = datetime.fromtimestamp(dt, timezone.utc)
    elif isinstance(dt, int):
        if len(str(dt)) > 10:
            dt = datetime.fromtimestamp(dt / 10 ** (len(str(dt)) - 10), timezone.utc)
        else:
            dt = datetime.fromtimestamp(dt, timezone.utc)
    elif isinstance(dt, str):
        dt = dateparser.parse(dt)
    elif isinstance(dt, pd.Timestamp):
        dt = dt.to_pydatetime()
    elif isinstance(dt, np.datetime64):
        dt = dt.astype(datetime)

    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        dt = dt.astimezone()
    else:
        dt = dt.replace(tzinfo=timezone(dt.tzinfo.utcoffset(dt)))
    if tz is not None:
        dt = dt.astimezone(tz)
    return dt


def datetime_to_ms(dt):
    """Convert a datetime to milliseconds."""
    epoch = datetime.fromtimestamp(0, dt.tzinfo)
    return int((dt - epoch).total_seconds() * 1000.0)


def interval_to_ms(interval):
    """Convert an interval string to milliseconds."""
    seconds_per_unit = {
        "m": 60,
        "h": 60 * 60,
        "d": 24 * 60 * 60,
        "w": 7 * 24 * 60 * 60,
    }
    try:
        return int(interval[:-1]) * seconds_per_unit[interval[-1]] * 1000
    except (ValueError, KeyError):
        return None
