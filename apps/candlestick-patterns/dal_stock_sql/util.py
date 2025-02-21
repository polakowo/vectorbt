import os
import time
import logging
from math import ceil
import datetime as dt
import pytz


# Convert timedelta to PostgreSQL-compatible interval string
def timedelta_to_postgres_interval(td):
    days = td.days
    seconds = td.seconds
    microseconds = td.microseconds

    # Extract hours, minutes, and seconds from the total seconds
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60

    # Format the interval string
    interval_str = f"{days} days {hours} hours {minutes} minutes {seconds} seconds"
    return interval_str


def nbar_by_day(bar_size):
    val = int(bar_size.split()[0])
    unit = bar_size.split()[1]
    span_paris = dt.timedelta(minutes=510)
    if unit == 'day':
        d = 1
    elif unit == 'hour':
        d = 9
    elif unit == 'mins':
        d = ceil(span_paris / dt.timedelta(minutes=val))
    else:
        raise NotImplementedError
    return d


def barsize_to_timedelta(bar_size, val=None):
    if val is None:
        val = int(bar_size.split()[0])
    unit = bar_size.split()[1]
    if unit == 'day':
        d = dt.timedelta(days=val)
    elif unit == 'hour':
        d = dt.timedelta(hours=val)
    elif unit == 'mins':
        d = dt.timedelta(minutes=val)
    else:
        raise NotImplementedError
    return d


def barsize_to_end(bar_size, timezone=None):
    if not timezone:
        timezone = pytz.timezone("Europe/Paris")
    end = dt.datetime.now()
    end = timezone.localize(end)
    if bar_size == '1 day':
        end -= dt.timedelta(days=1)
    elif bar_size == '1 week':
        end -= dt.timedelta(weeks=1)
    elif bar_size == '1 hour':
        end -= dt.timedelta(minutes=65)
    elif bar_size == '15 mins':
        end -= dt.timedelta(minutes=20)
    elif bar_size == '5 mins':
        end -= dt.timedelta(minutes=10)
    else:
        raise NotImplementedError
    return end


def barsize_to_table(bar_size):
    table = None
    if bar_size == '1 day':
        table = 'day1'
    elif bar_size == '1 week':
        table = 'week1'
    elif bar_size == '1 hour':
        table = 'hour1'
    elif bar_size == '15 mins':
        table = 'minute15'
    elif bar_size == '5 mins':
        table = 'minute5'
    else:
        raise NotImplementedError
    return table


class Singleton:
    """
    A non-thread-safe helper class to ease implementing singletons.
    This should be used as a decorator -- not a metaclass -- to the
    class that should be a singleton.

    The decorated class can define one `__init__` function that
    takes only the `self` argument. Also, the decorated class cannot be
    inherited from. Other than that, there are no restrictions that apply
    to the decorated class.

    To get the singleton instance, use the `Instance` method. Trying
    to use `__call__` will result in a `TypeError` being raised.

    """

    def __init__(self, decorated):
        self._decorated = decorated

    def Instance(self):
        """
        Returns the singleton instance. Upon its first call, it creates a
        new instance of the decorated class and calls its `__init__` method.
        On all subsequent calls, the already created instance is returned.

        """
        try:
            return self._instance
        except AttributeError:
            self._instance = self._decorated()
            return self._instance

    def __call__(self):
        raise TypeError('Singletons must be accessed through `Instance()`.')

    def __instancecheck__(self, inst):
        return isinstance(inst, self._decorated)


def printWhenExecuting(fn):
    def fn2(self):
        print("   doing", fn.__name__)
        fn(self)
        print("   done w/", fn.__name__)

    return fn2

def SetupLogger():
    if not os.path.exists("../log"):
        os.makedirs("../log")

    time.strftime("pyibapi.%Y%m%d_%H%M%S.log")
    recfmt = '%(asctime)s.%(msecs)03d %(levelname)s (%(threadName)s) %(filename)s:%(lineno)d %(message)s'

    timefmt = '%Y-%m-%dT%H:%M:%S'

    # logging.basicConfig( level=logging.DEBUG,
                       # format=recfmt, datefmt=timefmt)
    logging.basicConfig(filename=time.strftime("../log/past_ibkr.%y%m%d_%H%M%S.log"),
                        filemode="w",
                        level=logging.INFO,
                        format=recfmt, datefmt=timefmt)
    logger = logging.getLogger()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logger.addHandler(console)

