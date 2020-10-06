"""Signal generators built with `vectorbt.signals.factory.SignalFactory`."""

from vectorbt.signals.factory import SignalFactory
from vectorbt.signals.nb import stop_choice_nb

stop_param_setup = dict(
    array_like=True,  # passing an array means passing one value
    bc_to_input=True,  # check whether it can be broadcast to input
    broadcast_kwargs=dict(
        keep_raw=True  # don't materialize, keep shape for flexible indexing
    )
)
"""Setup used for stop values used in `vectorbt.signals.nb.stop_choice_nb`."""

StopExitSignals = SignalFactory(
    class_name='StopExitSignals',
    module_name=__name__,
    short_name='stex',
    input_names=['ts'],
    param_names=['stop'],
    param_settings=dict(stop=stop_param_setup),
    exit_only=True,
    iteratively=False
).from_choice_func(
    exit_choice_func=stop_choice_nb,
    exit_settings=dict(
        pass_inputs=['ts'],
        pass_params=['stop'],
        pass_first=True,
        pass_temp_int=True,
        pass_is_2d=True,
    )
)
"""Exit signal generator based on stop values. 

Generates `exits` based on `entries` and `vectorbt.signals.nb.stop_choice_nb`.

Args:
    entries (array_like): Entry array. Will broadcast.
    ts (array_like): Time series array such as price. Will broadcast.
    stops (float or array_like): One or more stop values (per row/column/element).
    stop_pos (StopPosition): See `vectorbt.signals.enums.StopPosition`."""

IStopExitSignals = SignalFactory(
    class_name='IStopExitSignals',
    module_name=__name__,
    short_name='istex',
    input_names=['ts'],
    param_names=['stop'],
    param_settings=dict(stop=stop_param_setup),
    exit_only=True,
    iteratively=True
).from_choice_func(
    exit_choice_func=stop_choice_nb,
    exit_settings=dict(
        pass_inputs=['ts'],
        pass_params=['stop'],
        pass_first=True,
        pass_temp_int=True,
        pass_is_2d=True,
    )
)
"""Exit signal generator based on stop values. 

Iteratively generates `new_entries` and `exits` based on `entries` and `vectorbt.signals.nb.stop_choice_nb`.

For arguments, see `StopExitSignals`."""

