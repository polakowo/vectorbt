"""Class for scheduling data updates."""

import logging

from vectorbt.utils.schedule import ScheduleManager
from vectorbt.utils.config import Configured

logger = logging.getLogger(__name__)


class DataUpdater(Configured):
    """Class for scheduling data updates.

    ## Example

    ```python-repl
    >>> import vectorbt as vbt

    >>> class MyDataUpdater(vbt.DataUpdater):
    ...     def __init__(self, *args, **kwargs):
    ...         super().__init__(*args, **kwargs)
    ...         self.update_count = 0
    ...
    ...     def update(self, count_limit=None):
    ...         prev_index_len = len(self.data.wrapper.index)
    ...         super().update()
    ...         new_index_len = len(self.data.wrapper.index)
    ...         print(f"Data updated with {new_index_len - prev_index_len} data points")
    ...         self.update_count += 1
    ...         if count_limit is not None and self.update_count >= count_limit:
    ...             raise vbt.CancelledError

    >>> data = vbt.GBMData.download('SYMBOL', start='1 minute ago', freq='1s')
    >>> my_updater = MyDataUpdater(data)
    >>> my_updater.update_every(count_limit=10)
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    >>> my_updater.data.get()
    2021-03-19 21:14:00.359754+01:00     97.934842
    2021-03-19 21:14:01.359754+01:00     89.750977
    2021-03-19 21:14:02.359754+01:00     89.361692
    2021-03-19 21:14:03.359754+01:00     91.725854
    2021-03-19 21:14:04.359754+01:00     97.759234
                                           ...
    2021-03-19 21:15:06.359754+01:00    140.974050
    2021-03-19 21:15:07.359754+01:00    143.796632
    2021-03-19 21:15:08.359754+01:00    143.065523
    2021-03-19 21:15:09.359754+01:00    131.059728
    2021-03-19 21:15:10.359754+01:00    126.803748
    Freq: S, Length: 71, dtype: float64
    ```

    Update in the background:

    ```python-repl
    >>> my_updater = MyDataUpdater(data)
    >>> my_updater.update_every(in_background=True, count_limit=10)
    Data updated with 13 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    Data updated with 1 data points
    >>> my_updater.data.get()
    2021-03-19 21:14:00.359754+01:00     97.934842
    2021-03-19 21:14:01.359754+01:00     89.750977
    2021-03-19 21:14:02.359754+01:00     89.361692
    2021-03-19 21:14:03.359754+01:00     91.725854
    2021-03-19 21:14:04.359754+01:00     97.759234
                                           ...
    2021-03-19 21:15:18.359754+01:00    161.816107
    2021-03-19 21:15:19.359754+01:00    165.721875
    2021-03-19 21:15:20.359754+01:00    174.419841
    2021-03-19 21:15:21.359754+01:00    189.774741
    2021-03-19 21:15:22.359754+01:00    187.604753
    Freq: S, Length: 83, dtype: float64
    ```
    """
    def __init__(self, data, schedule_manager=None, **kwargs):
        Configured.__init__(
            self,
            data=data,
            schedule_manager=schedule_manager,
            **kwargs
        )
        self._data = data
        if schedule_manager is None:
            schedule_manager = ScheduleManager()
        self._schedule_manager = schedule_manager

    @property
    def data(self):
        """Data instance.

        See `vectorbt.data.base.Data`."""
        return self._data

    @property
    def schedule_manager(self):
        """Schedule manager instance.

        See `vectorbt.utils.schedule.ScheduleManager`."""
        return self._schedule_manager

    def update(self, **kwargs):
        """Method that updates data.

        Override to do pre- and postprocessing.

        To stop this method from running again, raise `vectorbt.utils.schedule.CancelledError`."""
        self._data = self.data.update(**kwargs)
        self.update_config(data=self.data)
        new_index = self.data.wrapper.index
        logger.info(f"Updated data has {len(new_index)} rows from {new_index[0]} to {new_index[-1]}")

    def update_every(self, *args, to=None, until=None, tags=None, in_background=False, start_kwargs=None, **kwargs):
        """Schedule `DataUpdater.update`.

        For `*args`, `to`, `until` and `tags`, see `vectorbt.utils.schedule.ScheduleManager.every`.

        If `in_background` is set to True, starts in the background as an `asyncio` task.
        The task can be stopped with `vectorbt.utils.schedule.ScheduleManager.stop`.

        `**kwargs` are passed to `DataUpdater.update`."""
        if start_kwargs is None:
            start_kwargs = {}
        self.schedule_manager.every(*args, to=to, until=until, tags=tags).do(self.update, **kwargs)
        if in_background:
            self.schedule_manager.start_in_background(**start_kwargs)
        else:
            self.schedule_manager.start(**start_kwargs)
