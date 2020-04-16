import pandas as pd
import numpy as np
from numba import njit
from numba.types import UniTuple, f8, i8, b1, DictType, ListType, Array
from numba.typed import List, Dict
from copy import copy
import plotly.graph_objects as go
import itertools

from vectorbt.utils import *
from vectorbt.accessors import *
from vectorbt.timeseries import rolling_mean_nb, rolling_std_nb, ewm_mean_nb, \
    ewm_std_nb, diff_nb, set_by_mask_nb, prepend_nb, rolling_min_nb, rolling_max_nb

__all__ = ['IndicatorFactory', 'MA', 'MSTD', 'BollingerBands', 'RSI', 'Stochastic', 'MACD', 'OBV']

# ############# Indicator factory ############# #


def build_column_hierarchy(param_list, level_names, ts_columns):
    check_same_shape(param_list, level_names, along_axis=0)
    param_indices = [index_from_values(param_list[i], name=level_names[i]) for i in range(len(param_list))]
    param_columns = None
    for param_index in param_indices:
        if param_columns is None:
            param_columns = param_index
        else:
            param_columns = stack_indices(param_columns, param_index)
    if param_columns is not None:
        return combine_indices(param_columns, ts_columns)
    return ts_columns


def build_mapper(params, ts, new_columns, level_name):
    params_mapper = np.repeat(params, len(to_2d(ts).columns))
    params_mapper = pd.Series(params_mapper, index=new_columns, name=level_name)
    return params_mapper


def build_tuple_mapper(mappers_list, new_columns, level_names):
    tuple_mapper = list(zip(*list(map(lambda x: x.values, mappers_list))))
    tuple_mapper = pd.Series(tuple_mapper, index=new_columns, name=level_names)
    return tuple_mapper


def wrap_output(output, ts, new_columns):
    return ts.vbt.wrap_array(output, columns=new_columns)


def broadcast_ts(ts, params_len, new_columns):
    if is_series(ts) or len(new_columns) > ts.shape[1]:
        return ts.vbt.wrap_array(tile(ts.values, params_len, along_axis=1), columns=new_columns)
    else:
        return ts.vbt.wrap_array(ts, columns=new_columns)


def from_params_pipeline(ts_list, param_list, level_names, custom_func, *args, pass_lists=False,
                         param_product=False, broadcast_kwargs={}, **kwargs):
    """A pipeline to calculate an indicator based on its parameters.

    Does the following:
        - Takes one or multiple time series objects (ts_list) and broadcasts them,
        - Takes one or multiple parameter arrays (param_list) and broadcasts them,
        - Performs calculation (custom_func) to build indicator arrays (output_list),
        - Creates new column hierarchy based on parameters and level names,
        - Broadcasts time series objects to match the shape of the output objects,
        - Builds mappers that will link parameters to columns."""
    # Check time series objects
    check_type(ts_list[0], (pd.Series, pd.DataFrame))
    for i in range(1, len(ts_list)):
        ts_list[i].vbt.timeseries.validate()
    if len(ts_list) > 1:
        # Broadcast time series
        ts_list = broadcast(*ts_list, **broadcast_kwargs, writeable=True)
    # Check level names
    check_type(level_names, (list, tuple))
    check_same_len(param_list, level_names)
    for ts in ts_list:
        # Every time series object should be free of the specified level names in its columns
        for level_name in level_names:
            check_level_not_exists(ts, level_name)
    # Convert params to 1-dim arrays
    param_list = list(map(to_1d, param_list))
    if len(param_list) > 1:
        if param_product:
            # Make Cartesian product out of all params
            param_list = list(map(to_1d, param_list))
            param_list = list(zip(*list(itertools.product(*param_list))))
            param_list = list(map(np.asarray, param_list))
        else:
            # Broadcast such that each array has the same length
            param_list = broadcast(*param_list, writeable=True)
    # Perform main calculation
    if pass_lists:
        output_list = custom_func(ts_list, param_list, *args, **kwargs)
    else:
        output_list = custom_func(*ts_list, *param_list, *args, **kwargs)
    if not isinstance(output_list, (tuple, list, List)):
        output_list = (output_list,)
    if len(param_list) > 0:
        # Build new column levels on top of time series levels
        new_columns = build_column_hierarchy(param_list, level_names, to_2d(ts_list[0]).columns)
        # Wrap into new pandas objects both time series and output objects
        new_ts_list = list(map(lambda x: broadcast_ts(x, param_list[0].shape[0], new_columns), ts_list))
        # Build mappers to easily map between parameters and columns
        mapper_list = [build_mapper(x, ts_list[0], new_columns, level_names[i]) for i, x in enumerate(param_list)]
    else:
        # Some indicators don't have any params
        new_columns = to_2d(ts_list[0]).columns
        new_ts_list = ts_list
        mapper_list = []
    output_list = list(map(lambda x: wrap_output(x, ts_list[0], new_columns), output_list))
    if len(mapper_list) > 1:
        # Tuple object is a mapper that accepts tuples of parameters
        tuple_mapper = build_tuple_mapper(mapper_list, new_columns, tuple(level_names))
        mapper_list.append(tuple_mapper)
    return new_ts_list, output_list, mapper_list


def perform_init_checks(ts_list, output_list, mapper_list):
    for ts in ts_list:
        check_type(ts, (pd.Series, pd.DataFrame))
        ts.vbt.timeseries.validate()
    for i in range(1, len(ts_list) + len(output_list)):
        check_same_meta((ts_list + output_list)[i-1], (ts_list + output_list)[i])
    for mapper in mapper_list:
        check_type(mapper, pd.Series)
        check_same_index(to_2d(ts_list[0]).iloc[0, :], mapper)


def is_equal(obj, other, multiple=False, name='is_equal', **kwargs):
    if multiple:
        as_columns = index_from_values(other, name=name)
        return obj.vbt.combine_with_multiple(other, combine_func=np.equal, as_columns=as_columns, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=np.equal, **kwargs)


def is_above(obj, other, multiple=False, name='is_above', **kwargs):
    if multiple:
        as_columns = index_from_values(other, name=name)
        return obj.vbt.combine_with_multiple(other, combine_func=np.greater, as_columns=as_columns, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=np.greater, **kwargs)


def is_below(obj, other, multiple=False, name='is_below', **kwargs):
    if multiple:
        as_columns = index_from_values(other, name=name)
        return obj.vbt.combine_with_multiple(other, combine_func=np.less, as_columns=as_columns, concat=True, **kwargs)
    return obj.vbt.combine_with(other, combine_func=np.less, **kwargs)


class BaseIndicator():
    def __init__(self, name):
        self.name = name

    @classmethod
    def from_params(cls, *args, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError


class IndicatorFactory():
    """Build a stand-alone indicator class based on parameters.

    Does the following:
        - Creates an __init__ method where it stores all artifacts,
        - Creates a from_params method that runs the main indicator pipeline and is called by the user,
        - Adds pandas indexing, i.e., you can use iloc, loc, [] and other on the class itself,
        - Adds parameter indexing, i.e., use *your_param*_loc on the class to slice using parameters,
        - Adds user-defined properties,
        - Adds common comparison methods for all inputs, outputs and properties, e.g., crossovers."""

    @classmethod
    def from_custom_func(self,
                         custom_func,
                         ts_names=['ts'],
                         param_names=['param'],
                         output_names=['output'],
                         short_name='custom',
                         custom_properties={},
                         comparison_params={},
                         pass_lists=False):
        """Custom function can be anything that takes time series, params and other arguments, and returns outputs."""

        # Build class
        class CustomIndicator(BaseIndicator):
            def __init__(self, *args):
                ts_list = args[:len(ts_names)]
                output_list = args[len(ts_names):len(ts_names)+len(output_names)]
                mapper_list = args[len(ts_names)+len(output_names):-1]
                perform_init_checks(ts_list, output_list, mapper_list)

                for i, ts_name in enumerate(ts_names):
                    setattr(self, ts_name, ts_list[i])
                for i, output_name in enumerate(output_names):
                    setattr(self, output_name, output_list[i])
                for i, param_name in enumerate(param_names):
                    setattr(self, param_name + '_mapper', mapper_list[i])
                if len(param_names) > 1:
                    setattr(self, 'tuple_mapper', mapper_list[-1])
                super().__init__(args[-1])

            @classmethod
            def from_params(cls, *args, name=short_name.lower(), **kwargs):
                level_names = tuple([name + '_' + param_name for param_name in param_names])
                ts_list = args[:len(ts_names)]
                param_list = args[len(ts_names):len(ts_names)+len(param_names)]
                new_args = args[len(ts_names)+len(param_names):]
                new_ts_list, output_list, mapper_list = from_params_pipeline(
                    ts_list, param_list, level_names, custom_func, *new_args, pass_lists=pass_lists, **kwargs)
                return cls(*new_ts_list, *output_list, *mapper_list, name)

        # Add indexing methods
        def indexing_func(obj, loc_pandas_func):
            ts = []
            for ts_name in ts_names:
                ts.append(loc_pandas_func(getattr(obj, ts_name)))
            outputs = []
            for output_name in output_names:
                ts.append(loc_pandas_func(getattr(obj, output_name)))
            mappers = []
            for param_name in param_names:
                mappers.append(loc_mapper(getattr(obj, param_name + '_mapper'),
                                          getattr(obj, ts_names[0]), loc_pandas_func))
            if len(param_names) > 1:
                mappers.append(loc_mapper(obj.tuple_mapper, getattr(obj, ts_names[0]), loc_pandas_func))

            return obj.__class__(*ts, *outputs, *mappers, obj.name)

        CustomIndicator = add_indexing(indexing_func)(CustomIndicator)
        for i, param_name in enumerate(param_names):
            CustomIndicator = add_param_indexing(param_name, indexing_func)(CustomIndicator)
        if len(param_names) > 1:
            CustomIndicator = add_param_indexing('tuple', indexing_func)(CustomIndicator)

        # Add user-defined properties
        for property_name, property_func in custom_properties.items():
            @cached_property
            def custom_property(self, property_func=property_func):
                return property_func(self)
            setattr(CustomIndicator, property_name, custom_property)

        # Add comparison methods for all inputs, outputs, and user-defined properties
        comparison_attrs = set(ts_names + output_names + list(custom_properties.keys()))
        for attr in comparison_attrs:
            allow_with_class = False
            include_attr_name = True
            if attr in comparison_params:
                allow_with_class = comparison_params[attr].get('allow_with_class', allow_with_class)
                include_attr_name = comparison_params[attr].get('include_attr_name', include_attr_name)

            def create_comparison_method(func_name,
                                         comparison_func,
                                         attr=attr,
                                         allow_with_class=allow_with_class,
                                         include_attr_name=include_attr_name):
                def comparison_method(self, other, name=None, **kwargs):
                    if allow_with_class:
                        if isinstance(other, self.__class__):
                            other = getattr(other, attr)
                    if name is None:
                        if include_attr_name:
                            name = self.name + f'_{attr}_' + func_name
                        else:
                            name = self.name + '_' + func_name
                    return comparison_func(getattr(self, attr), other, name=name, **kwargs)
                return comparison_method

            def create_crossover_method(attr=attr,
                                        allow_with_class=allow_with_class,
                                        include_attr_name=include_attr_name):
                def crossover_method(self, other, wait=0, name=None, **kwargs):
                    above_method = getattr(self, f'{attr}_above')
                    below_method = getattr(self, f'{attr}_below')
                    if name is None:
                        if include_attr_name:
                            name = self.name + f'_{attr}_crossover'
                        else:
                            name = self.name + '_crossover'
                    # entry signal is first time this is about other
                    above_signals = above_method(other, name=name, **kwargs)\
                        .vbt.signals.nst(wait+1, after_false=True)
                    # exit signal is first time this is below other
                    below_signals = below_method(other, name=name, **kwargs)\
                        .vbt.signals.nst(wait+1, after_false=True)
                    return above_signals, below_signals
                return crossover_method

            setattr(CustomIndicator, f'{attr}_above', create_comparison_method('above', is_above))
            setattr(CustomIndicator, f'{attr}_below', create_comparison_method('below', is_below))
            setattr(CustomIndicator, f'{attr}_equal', create_comparison_method('equal', is_equal))
            setattr(CustomIndicator, f'{attr}_crossover', create_crossover_method())

        return CustomIndicator

    @classmethod
    def from_apply_func(cls, apply_func, caching_func=None, output_names=['output'], **kwargs):
        """Apply function is performed on each parameter individually.
        
        Apply functions are simpler to write since parameter selection and concating is done for you.
        
        But it has some limitations:
            - If your apply function isn't numba compiled, concating is also not numba compiled.
            - You can work with one parameter selection at a time, and can't view all parameters.

        You can also use a caching function to preprocess data beforehand.
        The outputs of the caching function will flow as additional arguments to the apply function."""
        num_outputs = len(output_names)

        if is_numba_func(apply_func):
            apply_and_concat_func = apply_and_concat_multiple_nb if num_outputs > 1 else apply_and_concat_one_nb

            @njit
            def select_params_func_nb(i, apply_func, ts_list, param_tuples, *args):
                # Select the next tuple of parameters
                return apply_func(*ts_list, *param_tuples[i], *args)

            def custom_func(ts_list, param_list, *args):
                # avoid deprecation warnings
                ts_list = list(map(lambda x: x.vbt.to_2d_array(), ts_list))
                typed_ts_list = tuple(ts_list)
                typed_param_tuples = tuple(list(zip(*param_list)))

                # User-defined preprocessing function (useful for caching)
                if caching_func is not None:
                    more_args = caching_func(*ts_list, *param_list, *args)
                    if not isinstance(more_args, (tuple, list, List)):
                        more_args = (more_args,)
                else:
                    more_args = ()

                return apply_and_concat_func(
                    param_list[0].shape[0],
                    select_params_func_nb,
                    apply_func,
                    typed_ts_list,
                    typed_param_tuples,
                    *args,
                    *more_args)
        else:
            apply_and_concat_func = apply_and_concat_multiple if num_outputs > 1 else apply_and_concat_one

            def select_params_func(i, apply_func, ts_list, param_list, *args, **kwargs):
                    # Select the next tuple of parameters
                param_is = list(map(lambda x: x[i], param_list))
                return apply_func(*ts_list, *param_is, *args, **kwargs)

            def custom_func(ts_list, param_list, *args, **kwargs):
                # User-defined preprocessing function (useful for caching)
                if caching_func is not None:
                    more_args = caching_func(*ts_list, *param_list, *args)
                    if not isinstance(more_args, (tuple, list)):
                        more_args = (more_args,)
                else:
                    more_args = ()

                return apply_and_concat_func(
                    param_list[0].shape[0],
                    select_params_func,
                    apply_func,
                    ts_list,
                    param_list,
                    *args,
                    *more_args,
                    **kwargs)

        return cls.from_custom_func(custom_func, output_names=output_names, pass_lists=True, **kwargs)


# ############# MA ############# #

@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], b1[:]), cache=True)
def ma_caching_nb(ts, windows, ewms):
    # Cache moving averages to effectively reduce the number of operations.
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                ma = ewm_mean_nb(ts, windows[i])
            else:
                ma = rolling_mean_nb(ts, windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = ma
    return cache_dict


@njit(f8[:, :](i8, i8[:], b1[:], DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def ma_apply_func_nb(i, windows, ewms, cache_dict):
    # For i-th window, take moving average out of cache and return.
    return cache_dict[(windows[i], int(ewms[i]))]


@njit(f8[:, :](f8[:, :], i8[:], b1[:]), cache=True)
def ma_custom_func_nb(ts, windows, ewms):
    # Run the apply function on each window and concat results horizontally.
    cache_dict = ma_caching_nb(ts, windows, ewms)
    return apply_and_concat_one_nb(len(windows), ma_apply_func_nb, windows, ewms, cache_dict)


def ma_custom_func(ts, windows, ewms, from_ma=None):
    if from_ma is not None:
        # Use another MA to take windows from there to avoid re-calculation
        indices = from_ma.tuple_loc.get_indices(list(zip(windows, ewms)))
        return from_ma.ma.values[:, indices]
    # Calculate MA from scratch
    return ma_custom_func_nb(ts.vbt.to_2d_array(), windows, ewms)


FactoryMA = IndicatorFactory.from_custom_func(
    ma_custom_func,
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['ma'],
    short_name='ma',
    comparison_params=dict(
        ma=dict(
            allow_with_class=True,
            include_attr_name=False
        )
    )
)


class MA(FactoryMA):
    @classmethod
    def from_params(cls, ts, window, ewm=False, **kwargs):
        return super().from_params(ts, window, ewm, **kwargs)

    @classmethod
    def from_combinations(cls, ts, windows, r, ewm=False, names=None, **kwargs):
        if names is None:
            names = ['ma_' + str(i+1) for i in range(r)]
        windows, ewm = broadcast(windows, ewm, writeable=True)
        ma = cls.from_params(ts, windows, ewm=ewm, **kwargs)
        param_lists = zip(*itertools.combinations(zip(windows, ewm), r))
        mas = []
        for i, param_list in enumerate(param_lists):
            i_windows, i_ewm = zip(*param_list)
            mas.append(cls.from_params(ts, i_windows, ewm=i_ewm, from_ma=ma, name=names[i], **kwargs))
        return tuple(mas)

    def plot(self,
             plot_ts=True,
             ts_name=None,
             ma_name=None,
             ts_scatter_kwargs={},
             ma_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.ts, pd.Series)
        check_type(self.ma, pd.Series)

        if ts_name is None:
            ts_name = f'Price ({self.name})'
        if ma_name is None:
            ma_name = f'MA ({self.name})'

        if plot_ts:
            fig = self.ts.vbt.timeseries.plot(name=ts_name, scatter_kwargs=ts_scatter_kwargs, fig=fig, **layout_kwargs)
        fig = self.ma.vbt.timeseries.plot(name=ma_name, scatter_kwargs=ma_scatter_kwargs, fig=fig)

        return fig

    def plot_ma_crossover(self, other,
                          crossover_kwargs={},
                          other_name=None,
                          other_scatter_kwargs={},
                          entry_scatter_kwargs={},
                          exit_scatter_kwargs={},
                          fig=None,
                          **plot_kwargs):
        check_type(self.ts, pd.Series)
        check_type(self.ma, pd.Series)

        if isinstance(other, MA):
            if other_name is None:
                other_name = f'MA ({other.name})'
            other = other.ma
        else:
            if other_name is None:
                other_name = 'Other'
        check_type(other, pd.Series)

        fig = self.plot(**plot_kwargs)
        fig = other.vbt.timeseries.plot(name=other_name, scatter_kwargs=other_scatter_kwargs, fig=fig)

        # Plot markets
        entries, exits = self.ma_crossover(other, **crossover_kwargs)
        entry_scatter = go.Scatter(
            x=self.ts.index[entries],
            y=self.ts[entries],
            mode='markers',
            marker=dict(
                symbol='triangle-up',
                color='limegreen',
                size=10
            ),
            name='Entry'
        )
        entry_scatter.update(**entry_scatter_kwargs)
        fig.add_trace(entry_scatter)
        exit_scatter = go.Scatter(
            x=self.ts.index[exits],
            y=self.ts[exits],
            mode='markers',
            marker=dict(
                symbol='triangle-down',
                color='orangered',
                size=10
            ),
            name='Exit'
        )
        exit_scatter.update(**exit_scatter_kwargs)
        fig.add_trace(exit_scatter)

        return fig


# ############# MSTD ############# #

@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], b1[:]), cache=True)
def mstd_caching_nb(ts, windows, ewms):
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                ma = ewm_std_nb(ts, windows[i])
            else:
                ma = rolling_std_nb(ts, windows[i])
            cache_dict[(windows[i], int(ewms[i]))] = ma
    return cache_dict


@njit(f8[:, :](f8[:, :], i8, b1, DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def mstd_apply_func_nb(ts, window, ewm, cache_dict):
    return cache_dict[(window, int(ewm))]


FactoryMSTD = IndicatorFactory.from_apply_func(
    mstd_apply_func_nb,
    caching_func=mstd_caching_nb,
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['mstd'],
    short_name='mstd',
    comparison_params=dict(
        mstd=dict(
            allow_with_class=True,
            include_attr_name=False
        )
    )
)


class MSTD(FactoryMSTD):
    @classmethod
    def from_params(cls, ts, window, ewm=False, **kwargs):
        return super().from_params(ts, window, ewm, **kwargs)

    def plot(self,
             name=None,
             scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.mstd, pd.Series)

        if name is None:
            name = f'MSTD ({self.name})'

        fig = self.mstd.vbt.timeseries.plot(name=name, scatter_kwargs=scatter_kwargs, fig=fig, **layout_kwargs)

        return fig

# ############# BollingerBands ############# #


@njit(UniTuple(DictType(UniTuple(i8, 2), f8[:, :]), 2)(f8[:, :], i8[:], b1[:], f8[:]), cache=True)
def bb_caching_nb(ts, windows, ewms, alphas):
    ma_cache_dict = ma_caching_nb(ts, windows, ewms)
    mstd_cache_dict = mstd_caching_nb(ts, windows, ewms)
    return ma_cache_dict, mstd_cache_dict


@njit(UniTuple(f8[:, :], 3)(f8[:, :], i8, b1, f8, DictType(UniTuple(i8, 2), f8[:, :]), DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def bb_apply_func_nb(ts, window, ewm, alpha, ma_cache_dict, mstd_cache_dict):
    # Calculate lower, middle and upper bands
    ma = np.copy(ma_cache_dict[(window, int(ewm))])
    mstd = np.copy(mstd_cache_dict[(window, int(ewm))])
    # # (MA + Kσ), MA, (MA - Kσ)
    return ma + alpha * mstd, ma, ma - alpha * mstd


FactoryBollingerBands = IndicatorFactory.from_apply_func(
    bb_apply_func_nb,
    caching_func=bb_caching_nb,
    ts_names=['ts'],
    param_names=['window', 'ewm', 'alpha'],
    output_names=['upper_band', 'middle_band', 'lower_band'],
    short_name='bb',
    custom_properties=dict(
        percent_b=lambda self: (self.ts - self.lower_band) / (self.upper_band - self.lower_band),
        bandwidth=lambda self: (self.upper_band - self.lower_band) / self.middle_band
    )
)


class BollingerBands(FactoryBollingerBands):
    @classmethod
    def from_params(cls, ts, window=20, ewm=False, alpha=2, **kwargs):
        alpha = np.asarray(alpha).astype(np.float64)
        return super().from_params(ts, window, ewm, alpha, **kwargs)

    def plot(self,
             plot_ts=True,
             ts_name=None,
             upper_band_name=None,
             middle_band_name=None,
             lower_band_name=None,
             ts_scatter_kwargs={},
             upper_band_scatter_kwargs={},
             middle_band_scatter_kwargs={},
             lower_band_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.ts, pd.Series)
        check_type(self.upper_band, pd.Series)
        check_type(self.middle_band, pd.Series)
        check_type(self.lower_band, pd.Series)

        if ts_name is None:
            ts_name = f'Price ({self.name})'
        if upper_band_name is None:
            upper_band_name = f'Upper Band ({self.name})'
        if middle_band_name is None:
            middle_band_name = f'Middle Band ({self.name})'
        if lower_band_name is None:
            lower_band_name = f'Lower Band ({self.name})'

        upper_band_scatter_kwargs = {**dict(line=dict(color='grey')), **upper_band_scatter_kwargs}  # default kwargs
        lower_band_scatter_kwargs = {**dict(line=dict(color='grey')), **lower_band_scatter_kwargs}

        if plot_ts:
            fig = self.ts.vbt.timeseries.plot(name=ts_name, scatter_kwargs=ts_scatter_kwargs, fig=fig, **layout_kwargs)
        fig = self.upper_band.vbt.timeseries.plot(
            name=upper_band_name, scatter_kwargs=upper_band_scatter_kwargs, fig=fig)
        fig = self.middle_band.vbt.timeseries.plot(
            name=middle_band_name, scatter_kwargs=middle_band_scatter_kwargs, fig=fig)
        fig = self.lower_band.vbt.timeseries.plot(
            name=lower_band_name, scatter_kwargs=lower_band_scatter_kwargs, fig=fig)

        return fig


# ############# RSI ############# #

@njit(DictType(UniTuple(i8, 2), UniTuple(f8[:, :], 2))(f8[:, :], i8[:], b1[:]), cache=True)
def rsi_caching_nb(ts, windows, ewms):
    delta = diff_nb(ts)[1:, :]  # otherwise ewma will be all NaN
    up, down = delta.copy(), delta.copy()
    up = set_by_mask_nb(up, up < 0, 0)
    down = np.abs(set_by_mask_nb(down, down > 0, 0))
    # Cache
    cache_dict = dict()
    for i in range(windows.shape[0]):
        if (windows[i], int(ewms[i])) not in cache_dict:
            if ewms[i]:
                roll_up = ewm_mean_nb(up, windows[i])
                roll_down = ewm_mean_nb(down, windows[i])
            else:
                roll_up = rolling_mean_nb(up, windows[i])
                roll_down = rolling_mean_nb(down, windows[i])
            roll_up = prepend_nb(roll_up, 1, np.nan)  # bring to old shape
            roll_down = prepend_nb(roll_down, 1, np.nan)
            cache_dict[(windows[i], int(ewms[i]))] = roll_up, roll_down
    return cache_dict


@njit(f8[:, :](f8[:, :], i8, b1, DictType(UniTuple(i8, 2), UniTuple(f8[:, :], 2))), cache=True)
def rsi_apply_func_nb(ts, window, ewm, cache_dict):
    roll_up, roll_down = cache_dict[(window, int(ewm))]
    return 100 - 100 / (1 + roll_up / roll_down)


FactoryRSI = IndicatorFactory.from_apply_func(
    rsi_apply_func_nb,
    caching_func=rsi_caching_nb,
    ts_names=['ts'],
    param_names=['window', 'ewm'],
    output_names=['rsi'],
    short_name='rsi',
    comparison_params=dict(
        rsi=dict(
            allow_with_class=True,
            include_attr_name=False
        )
    )
)


class RSI(FactoryRSI):
    @classmethod
    def from_params(cls, ts, window=14, ewm=False, **kwargs):
        return super().from_params(ts, window, ewm, **kwargs)

    def plot(self,
             name=None,
             scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.rsi, pd.Series)

        if name is None:
            name = f'RSI ({self.name})'

        fig = self.rsi.vbt.timeseries.plot(name=name, scatter_kwargs=scatter_kwargs, fig=fig, **layout_kwargs)

        return fig


# ############# Stochastic ############# #


@njit(DictType(i8, UniTuple(f8[:, :], 2))(f8[:, :], f8[:, :], f8[:, :], i8[:], i8[:], b1[:]), cache=True)
def stoch_caching_nb(close_ts, high_ts, low_ts, k_windows, d_windows, ewms):
    cache_dict = dict()
    for i in range(k_windows.shape[0]):
        if k_windows[i] not in cache_dict:
            roll_min = rolling_min_nb(low_ts, k_windows[i])
            roll_max = rolling_max_nb(high_ts, k_windows[i])
            cache_dict[k_windows[i]] = roll_min, roll_max
    return cache_dict


@njit(UniTuple(f8[:, :], 2)(f8[:, :], f8[:, :], f8[:, :], i8, i8, b1, DictType(i8, UniTuple(f8[:, :], 2))), cache=True)
def stoch_apply_func_nb(close_ts, high_ts, low_ts, k_window, d_window, ewm, cache_dict):
    roll_min, roll_max = cache_dict[k_window]
    percent_k = 100 * (close_ts - roll_min) / (roll_max - roll_min)
    if ewm:
        percent_d = ewm_mean_nb(percent_k, d_window)
    else:
        percent_d = rolling_mean_nb(percent_k, d_window)
    percent_d[:k_window+d_window-2, :] = np.nan  # min_periods for ewm
    return percent_k, percent_d


FactoryStochastic = IndicatorFactory.from_apply_func(
    stoch_apply_func_nb,
    caching_func=stoch_caching_nb,
    ts_names=['close_ts', 'high_ts', 'low_ts'],
    param_names=['k_window', 'd_window', 'ewm'],
    output_names=['percent_k', 'percent_d'],
    short_name='stoch'
)


class Stochastic(FactoryStochastic):
    @classmethod
    def from_params(cls, close_ts, high_ts=None, low_ts=None, k_window=14, d_window=3, ewm=False, **kwargs):
        if high_ts is None:
            high_ts = close_ts
        if low_ts is None:
            low_ts = close_ts
        return super().from_params(close_ts, high_ts, low_ts, k_window, d_window, ewm, **kwargs)

    def crossover_signals(self, wait=0, **kwargs):
        pk_above_signals = self.is_percent_k_above(
            self.percent_d, **kwargs).vbt.signals.nst(wait+1, after_false=True)
        pk_below_signals = self.is_percent_k_below(
            self.percent_d, **kwargs).vbt.signals.nst(wait+1, after_false=True)
        return pk_above_signals, pk_below_signals

    def plot(self,
             percent_k_name=None,
             percent_d_name=None,
             percent_k_scatter_kwargs={},
             percent_d_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.percent_k, pd.Series)
        check_type(self.percent_d, pd.Series)

        if percent_k_name is None:
            percent_k_name = f'%K ({self.name})'
        if percent_d_name is None:
            percent_d_name = f'%D ({self.name})'

        fig = self.percent_k.vbt.timeseries.plot(
            name=percent_k_name, scatter_kwargs=percent_k_scatter_kwargs, fig=fig, **layout_kwargs)
        fig = self.percent_d.vbt.timeseries.plot(name=percent_d_name, scatter_kwargs=percent_d_scatter_kwargs, fig=fig)

        return fig


# ############# MACD ############# #

@njit(DictType(UniTuple(i8, 2), f8[:, :])(f8[:, :], i8[:], i8[:], i8[:], b1[:]), cache=True)
def macd_caching_nb(ts, fast_windows, slow_windows, signal_windows, ewms):
    return ma_caching_nb(ts, np.concatenate((fast_windows, slow_windows)), np.concatenate((ewms, ewms)))


@njit(UniTuple(f8[:, :], 4)(f8[:, :], i8, i8, i8, b1, DictType(UniTuple(i8, 2), f8[:, :])), cache=True)
def macd_apply_func_nb(ts, fast_window, slow_window, signal_window, ewm, cache_dict):
    fast_ma = cache_dict[(fast_window, int(ewm))]
    slow_ma = cache_dict[(slow_window, int(ewm))]
    macd_ts = fast_ma - slow_ma
    if ewm:
        signal_ts = ewm_mean_nb(macd_ts, signal_window)
    else:
        signal_ts = rolling_mean_nb(macd_ts, signal_window)
    signal_ts[:max(fast_window, slow_window)+signal_window-2, :] = np.nan  # min_periods for ewm
    return np.copy(fast_ma), np.copy(slow_ma), macd_ts, signal_ts


FactoryMACD = IndicatorFactory.from_apply_func(
    macd_apply_func_nb,
    caching_func=macd_caching_nb,
    ts_names=['ts'],
    param_names=['fast_window', 'slow_window', 'signal_window', 'ewm'],
    output_names=['fast_ma', 'slow_ma', 'macd', 'signal'],
    short_name='macd',
    custom_properties=dict(
        histogram=lambda self: self.macd - self.signal,
    )
)


class MACD(FactoryMACD):
    @classmethod
    def from_params(cls, ts, fast_window=26, slow_window=12, signal_window=9, ewm=True, **kwargs):
        return super().from_params(ts, fast_window, slow_window, signal_window, ewm, **kwargs)

    def plot(self,
             macd_name=None,
             signal_name=None,
             macd_scatter_kwargs={},
             signal_scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.macd, pd.Series)
        check_type(self.signal, pd.Series)

        if macd_name is None:
            macd_name = f'MACD ({self.name})'
        if signal_name is None:
            signal_name = f'Signal ({self.name})'

        fig = self.macd.vbt.timeseries.plot(
            name=macd_name, scatter_kwargs=macd_scatter_kwargs, fig=fig, **layout_kwargs)
        fig = self.signal.vbt.timeseries.plot(name=signal_name, scatter_kwargs=signal_scatter_kwargs, fig=fig)

        return fig


# ############# OBV ############# #

@njit(f8[:, :](f8[:, :], f8[:, :]))
def obv_custom_func_nb(close_ts, volume_ts):
    obv = np.full_like(close_ts, np.nan)
    for col in range(close_ts.shape[1]):
        cumsum = 0
        for i in range(1, close_ts.shape[0]):
            if np.isnan(close_ts[i, col]) or np.isnan(close_ts[i-1, col]) or np.isnan(volume_ts[i, col]):
                continue
            if close_ts[i, col] > close_ts[i-1, col]:
                cumsum += volume_ts[i, col]
            elif close_ts[i, col] < close_ts[i-1, col]:
                cumsum += -volume_ts[i, col]
            obv[i, col] = cumsum
    return obv


def obv_custom_func(close_ts, volume_ts):
    return obv_custom_func_nb(close_ts.vbt.to_2d_array(), volume_ts.vbt.to_2d_array())


FactoryOBV = IndicatorFactory.from_custom_func(
    obv_custom_func,
    ts_names=['close_ts', 'volume_ts'],
    param_names=[],
    output_names=['obv'],
    short_name='obv'
)


class OBV(FactoryOBV):
    @classmethod
    def from_params(cls, close_ts, volume_ts):
        return super().from_params(close_ts, volume_ts)

    def plot(self,
             name=None,
             scatter_kwargs={},
             fig=None,
             **layout_kwargs):
        check_type(self.obv, pd.Series)

        if name is None:
            name = f'OBV ({self.name})'

        fig = self.obv.vbt.timeseries.plot(name=name, scatter_kwargs=scatter_kwargs, fig=fig, **layout_kwargs)

        return fig
