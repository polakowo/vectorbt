"""Utilities for working with dates and time."""

import numpy as np
import pandas as pd
import dateparser
from datetime import datetime, timezone, timedelta, time as dt_time
import time
from schedule import Scheduler

DatetimeTypes = (pd.DatetimeIndex, pd.TimedeltaIndex, pd.PeriodIndex)


def to_timedelta(arg, **kwargs):
    """`pd.to_timedelta` that uses unit abbreviation with number."""
    if isinstance(arg, str) and not arg[0].isdigit():
        # Otherwise "ValueError: unit abbreviation w/o a number"
        freq = '1' + arg
    return pd.to_timedelta(arg, **kwargs)


def to_time_units(obj, time_delta):
    """Multiply each element with timedelta to get result in time units."""
    return obj * to_timedelta(time_delta)


def get_utc_tz():
    """Get UTC timezone."""
    return timezone.utc


def get_local_tz():
    """Get local timezone."""
    return timezone(datetime.now(timezone.utc).astimezone().utcoffset())


def convert_tzaware_time(t, tz_out):
    """Return as non-naive time.

    `datetime.time` should have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).timetz()


def tzaware_to_naive_time(t, tz_out):
    """Return as naive time.

    `datetime.time` should have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


def naive_to_tzaware_time(t, tz_out):
    """Return as non-naive time.

    `datetime.time` should not have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time().replace(tzinfo=tz_out)


def convert_naive_time(t, tz_out):
    """Return as naive time.

    `datetime.time` should not have `tzinfo` set."""
    return datetime.combine(datetime.today(), t).astimezone(tz_out).time()


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


class ScheduleManager:
    """Class that manages `schedule.Scheduler`."""

    units = (
        "second",
        "seconds",
        "minute",
        "minutes",
        "hour",
        "hours",
        "day",
        "days",
        "week",
        "weeks"
    )

    weekdays = (
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
    )

    def __init__(self, scheduler=None):
        if scheduler is None:
            scheduler = Scheduler()
        self._scheduler = scheduler

    @property
    def scheduler(self):
        """Scheduler."""
        return self._scheduler

    def job_func(self, *args, **kwargs):
        """Job abstract method."""
        raise NotImplementedError

    def every(self, *args, to=None, until=None, tags=None, **kwargs):
        """Create a new job that runs every `interval` units of time.

        `*args` can include at most four different arguments: `interval`, `unit`, `start_day`, and `at`,
        in the strict order:

        * `interval`: integer or `datetime.timedelta`
        * `unit`: `ScheduleManager.units`
        * `start_day`: `ScheduleManager.weekdays`
        * `at`: string or `datetime.time`.

        See the package `schedule` for more details.

        ## Example

        ```python-repl
        >>> import datetime
        >>> import pytz
        >>> import vectorbt as vbt

        >>> class MyManager(vbt.ScheduleManager):
        ...     def job_func(self, message="I'm working..."):
        ...         print(message)

        >>> my_manager = MyManager()

        >>> # add jobs
        >>> my_manager.every()
        Every 1 second do job_func() (last run: [never], next run: 2021-03-18 19:06:47)

        >>> my_manager.every(10, 'minutes')
        Every 10 minutes do job_func() (last run: [never], next run: 2021-03-18 19:16:46)

        >>> my_manager.every('hour')
        Every 1 hour do job_func() (last run: [never], next run: 2021-03-18 20:06:46)

        >>> my_manager.every('10:30')
        Every 1 day at 10:30:00 do job_func() (last run: [never], next run: 2021-03-19 10:30:00)

        >>> my_manager.every('day', '10:30')
        Every 1 day at 10:30:00 do job_func() (last run: [never], next run: 2021-03-19 10:30:00)

        >>> my_manager.every('day', datetime.time(9, 30, tzinfo=pytz.utc))
        Every 1 day at 10:30:00 do job_func() (last run: [never], next run: 2021-03-19 10:30:00)

        >>> my_manager.every('monday')
        Every 1 week do job_func() (last run: [never], next run: 2021-03-22 19:06:46)

        >>> my_manager.every('wednesday', '13:15')
        Every 1 week at 13:15:00 do job_func() (last run: [never], next run: 2021-03-24 13:15:00)

        >>> my_manager.every('minute', ':17')
        Every 1 minute at 00:00:17 do job_func() (last run: [never], next run: 2021-03-18 19:07:17)

        >>> my_manager.every(10, message="Hello")
        Every 10 seconds do job_func(message='Hello') (last run: [never], next run: 2021-03-18 19:06:56)

        >>> my_manager.start()
        ```
        """
        # Parse arguments
        interval = 1
        unit = None
        start_day = None
        at = None

        def _is_arg_interval(arg):
            return isinstance(arg, (int, timedelta))

        def _is_arg_unit(arg):
            return isinstance(arg, str) and arg in self.units

        def _is_arg_start_day(arg):
            return isinstance(arg, str) and arg in self.weekdays

        def _is_arg_at(arg):
            return (isinstance(arg, str) and ':' in arg) or isinstance(arg, dt_time)

        expected_args = ['interval', 'unit', 'start_day', 'at']
        for i, arg in enumerate(args):
            if 'interval' in expected_args and _is_arg_interval(arg):
                interval = arg
                expected_args = expected_args[expected_args.index('interval') + 1:]
                continue
            if 'unit' in expected_args and _is_arg_unit(arg):
                unit = arg
                expected_args = expected_args[expected_args.index('unit') + 1:]
                continue
            if 'start_day' in expected_args and _is_arg_start_day(arg):
                start_day = arg
                expected_args = expected_args[expected_args.index('start_day') + 1:]
                continue
            if 'at' in expected_args and _is_arg_at(arg):
                at = arg
                expected_args = expected_args[expected_args.index('at') + 1:]
                continue
            raise ValueError(f"Arg at index {i} is unexpected")

        if at is not None:
            if unit is None and start_day is None:
                unit = 'days'
        if unit is None and start_day is None:
            unit = 'seconds'

        job = self.scheduler.every(interval)
        if unit is not None:
            job = getattr(job, unit)
        if start_day is not None:
            job = getattr(job, start_day)
        if at is not None:
            if isinstance(at, dt_time):
                if job.unit == "days" or job.start_day:
                    if at.tzinfo is not None:
                        at = tzaware_to_naive_time(at, None)
                at = at.isoformat()
                if job.unit == "hours":
                    at = ':'.join(at.split(':')[1:])
                if job.unit == "minutes":
                    at = ':' + at.split(':')[2]
            job = job.at(at)
        if to is not None:
            job = job.to(to)
        if until is not None:
            job = job.until(until)
        if tags is not None:
            if not isinstance(tags, tuple):
                tags = (tags,)
            job = job.tag(*tags)
        return job.do(self.job_func, **kwargs)

    def start(self, sleep=1, clear_with_tag=None):
        """Run pending jobs in a loop."""
        try:
            while True:
                if clear_with_tag is not None:
                    self.scheduler.clear(clear_with_tag)
                self.scheduler.run_pending()
                time.sleep(sleep)
        except KeyboardInterrupt:
            pass
