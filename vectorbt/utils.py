import numpy as np
import pandas as pd
from functools import wraps, reduce, update_wrapper
from inspect import signature, Parameter
import types
import itertools
from collections.abc import Iterable


def is_series(arg):
    return isinstance(arg, pd.Series)


def is_frame(arg):
    return isinstance(arg, pd.DataFrame)


def is_pandas(arg):
    return is_series(arg) or is_frame(arg)


def is_array(arg):
    return isinstance(arg, np.ndarray)


def is_array_like(arg):
    return is_pandas(arg) or is_array(arg)

# ############# Tests ############# #


def check_not_none(arg):
    if arg is None:
        raise TypeError(f"Cannot be None")


def check_type(arg, types):
    if not isinstance(arg, types):
        if isinstance(types, tuple):
            raise TypeError(f"Type must be one of {types}")
        else:
            raise TypeError(f"Type must be {types}")


def check_dtype(arg, dtype):
    if is_frame(arg):
        if (arg.dtypes != dtype).any():
            raise ValueError(f"Data type must be {dtype} for all columns")
    else:
        if arg.dtype != dtype:
            raise ValueError(f"Data type must be {dtype}")


def check_same_type(arg1, arg2):
    if type(arg1) != type(arg2):
        raise ValueError(f"Must have the same type")


def check_same_dtype(arg1, arg2):
    if not is_array_like(arg1):
        arg1 = np.asarray(arg1)
    if not is_array_like(arg2):
        arg2 = np.asarray(arg2)
    if is_frame(arg1):
        dtypes1 = arg1.dtypes.to_numpy()
    else:
        dtypes1 = np.asarray([arg1.dtype])
    if is_frame(arg2):
        dtypes2 = arg2.dtypes.to_numpy()
    else:
        dtypes2 = np.asarray([arg2.dtype])
    if len(dtypes1) == len(dtypes2):
        if (dtypes1 == dtypes2).all():
            return
    elif len(np.unique(dtypes1)) == 1 and len(np.unique(dtypes2)) == 1:
        if (np.unique(dtypes1) == np.unique(dtypes2)).all():
            return
    raise ValueError(f"Data types do not match")


def check_ndim(arg, ndims):
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if isinstance(ndims, tuple):
        if arg.ndim not in ndims:
            raise ValueError(f"Number of dimensions must be one of {ndims}")
    else:
        if arg.ndim != ndims:
            raise ValueError(f"Number of dimensions must be {ndims}")


def check_same_shape(arg1, arg2, along_axis=None):
    if not is_array_like(arg1):
        arg1 = np.asarray(arg1)
    if not is_array_like(arg2):
        arg2 = np.asarray(arg2)
    if along_axis is None:
        if arg1.shape != arg2.shape:
            raise ValueError(f"Must have the same shape")
    else:
        if isinstance(along_axis, tuple):
            if arg1.shape[along_axis[0]] != arg2.shape[along_axis[1]]:
                raise ValueError(
                    f"Axis {along_axis[0]} of first and axis {along_axis[1]} of second must equal")
        else:
            if arg1.shape[along_axis] != arg2.shape[along_axis]:
                raise ValueError(f"Both must have the same axis {along_axis}")


def check_same_index(arg1, arg2):
    passed = arg1.index == arg2.index
    if isinstance(passed, bool):
        if passed:
            return
    else:
        if passed.all():
            return
    raise ValueError(f"Must have the same index")


def check_same_columns(arg1, arg2):
    passed = arg1.columns == arg2.columns
    if isinstance(passed, bool):
        if passed:
            return
    else:
        if passed.all():
            return
    raise ValueError(f"Must have the same columns")


def check_same_meta(arg1, arg2, check_dtype=True):
    check_same_type(arg1, arg2)
    check_same_shape(arg1, arg2)
    if is_pandas(arg1):
        check_same_index(arg1, arg2)
        if is_frame(arg1):
            check_same_columns(arg1, arg2)
        else:
            check_same_columns(to_2d(arg1), to_2d(arg2))
    if check_dtype:
        check_same_dtype(arg1, arg2)


def check_same(arg1, arg2):
    if is_pandas(arg1):
        if arg1.equals(arg2):
            return
    else:
        if np.array_equal(arg1, arg2):
            return
    raise ValueError(f"Must have the same type and values")


def check_level_exists(arg, level_name):
    if is_frame(arg):
        if level_name not in arg.columns.names:
            raise ValueError("Level not exists")


def check_level_not_exists(arg, level_name):
    if is_frame(arg):
        if level_name in arg.columns.names:
            raise ValueError("Level already exists")


# ############# Broadcasting ############# #


def to_1d(arg):
    """Reshape argument to one dimension."""
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        if arg.shape[1] == 1:
            if is_frame(arg):
                return arg.iloc[:, 0]
            return arg[:, 0]
    if arg.ndim == 1:
        return arg
    elif arg.ndim == 0:
        return arg.reshape((1,))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


def to_2d(arg, expand_axis=1):
    """Reshape argument to two dimensions."""
    if not is_array_like(arg):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        if is_series(arg):
            if expand_axis == 0:
                return pd.DataFrame(arg.values[None, :], columns=arg.index)
            elif expand_axis == 1:
                return arg.to_frame()
        return np.expand_dims(arg, expand_axis)
    elif arg.ndim == 0:
        return arg.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def tile_2d(arg, n, new_columns=None):
    arg = to_2d(arg)
    if is_frame(arg):
        return pd.concat([arg for i in range(n)], axis=1)
    return np.tile(arg, (1, n))


# You can change default broadcasting rules from code
# Useful for magic methods that cannot accept broadcasting parameters as kwargs
broadcasting_rules = dict(
    index_from='strict',
    columns_from='stack'
)


def broadcast_index(*args, index_from=None, is_columns=False, **kwargs):
    """Broadcast index/columns of all arguments."""
    args = list(args)
    new_index = None

    if index_from == 'default':
        index_from = broadcasting_rules[f"{'columns' if is_columns else 'index'}_from"]
    if index_from is not None:
        if isinstance(index_from, int):
            # Take index/columns of the object indexed by index_from
            if is_pandas(args[index_from]):
                if is_columns:
                    if is_frame(args[index_from]):
                        new_index = args[index_from].columns
                else:
                    new_index = args[index_from].index
        elif isinstance(index_from, str):
            if index_from == 'stack':
                # If pandas objects have different index/columns, stack them together for better tracking
                for i in range(len(args)):
                    if is_pandas(args[i]):
                        if is_columns:
                            if is_frame(args[i]) and args[i].shape[1] > 1:
                                if new_index is None:
                                    new_index = args[i].columns
                                new_index = stack_indexes(new_index, args[i].columns, **kwargs)
                        else:
                            if args[i].shape[0] > 1:
                                if new_index is None:
                                    new_index = args[i].index
                                new_index = stack_indexes(new_index, args[i].index, **kwargs)
            elif index_from == 'strict':
                # If pandas objects have different index/columns, raise an exception
                for i in range(len(args)):
                    if is_pandas(args[i]):
                        if is_columns:
                            if is_frame(args[i]):
                                if new_index is None:
                                    new_index = args[i].columns
                                if len(new_index) != len(args[i].columns) or (new_index != args[i].columns).any():
                                    raise ValueError("All pandas object must have the same columns")
                        else:
                            if new_index is None:
                                new_index = args[i].index
                            if len(new_index) != len(args[i].index) or (new_index != args[i].index).any():
                                raise ValueError("All pandas object must have the same index")
            else:
                raise ValueError(f"Invalid value for {'columns' if is_columns else 'index'}_from")
        else:
            raise ValueError(f"Invalid value for {'columns' if is_columns else 'index'}_from")
    return args, new_index


def wrap_broadcasted(old_arg, new_arg, is_pd=False, new_index=None, new_columns=None):
    """Transform newly broadcasted array to match the type of the original object."""
    # If the array expanded beyond new_index or new_columns
    if new_arg.ndim >= 1:
        if new_index is not None and new_arg.shape[0] > len(new_index):
            new_index = None  # basic range
    if new_arg.ndim == 2:
        if new_columns is not None and new_arg.shape[1] > len(new_columns):
            new_columns = None

    if is_pd:
        if is_pandas(old_arg):
            if new_index is None and old_arg.shape[0] == new_arg.shape[0]:
                new_index = old_arg.index
            if new_arg.ndim == 1:
                # Series -> Series
                return pd.Series(new_arg, index=new_index)
            if new_arg.ndim == 2:
                if is_frame(old_arg):
                    if new_columns is None and old_arg.shape[1] == new_arg.shape[1]:
                        new_columns = old_arg.columns
                    # DataFrame -> DataFrame
                    return pd.DataFrame(new_arg, index=new_index, columns=new_columns)
                # Series -> DataFrame
                return pd.DataFrame(new_arg, index=new_index, columns=new_columns)
        if new_arg.ndim == 1:
            # Other -> Series
            return pd.Series(new_arg, index=new_index)
        if new_arg.ndim == 2:
            # Other -> DataFrame
            return pd.DataFrame(new_arg, index=new_index, columns=new_columns)
    # Other -> Array
    return new_arg


def broadcast(*args, index_from='default', columns_from='default', **kwargs):
    """Bring multiple arguments to the same shape."""
    is_pd = False
    is_2d = False
    args = list(args)

    # Convert to np.ndarray object if not numpy or pandas
    for i in range(len(args)):
        if not is_array_like(args[i]):
            args[i] = np.asarray(args[i])
        if args[i].ndim > 1:
            is_2d = True
        if is_pandas(args[i]):
            is_pd = True

    # Decide on index and columns
    args, new_index = broadcast_index(*args, index_from=index_from, is_columns=False, **kwargs)
    args, new_columns = broadcast_index(*args, index_from=columns_from, is_columns=True, **kwargs)

    # Convert all pd.Series objects to pd.DataFrame
    if is_2d:
        for i in range(len(args)):
            if is_series(args[i]):
                args[i] = args[i].to_frame()

    # Perform broadcasting operation
    new_args = np.broadcast_arrays(*args, subok=True)

    # Bring arrays to their old types (e.g. array -> pandas)
    for i in range(len(new_args)):
        old_arg = args[i].copy()
        new_arg = new_args[i].copy()
        args[i] = wrap_broadcasted(old_arg, new_arg, is_pd=is_pd, new_index=new_index, new_columns=new_columns)
    return args


def broadcast_to(arg1, arg2, index_from='default', columns_from=1, **kwargs):
    """Bring first argument to the shape of second argument."""
    if not is_array_like(arg1):
        arg1 = np.asarray(arg1)
    if not is_array_like(arg2):
        arg2 = np.asarray(arg2)

    # Decide on index and columns
    args, new_index = broadcast_index(arg1, arg2, index_from=index_from, is_columns=False, **kwargs)
    args, new_columns = broadcast_index(arg1, arg2, index_from=columns_from, is_columns=True, **kwargs)

    # Convert all pd.Series objects to pd.DataFrame
    is_2d = arg1.ndim > 1 or arg2.ndim > 1
    is_pd = is_pandas(arg1) or is_pandas(arg2)
    if is_2d:
        if is_series(arg1):
            arg1 = arg1.to_frame()
        if is_series(arg2):
            arg2 = arg2.to_frame()

    # Perform broadcasting operation
    arg1_new = np.broadcast_to(arg1, arg2.shape, subok=True).copy()

    # Bring array to its old types (e.g. array -> pandas)
    return wrap_broadcasted(arg1, arg1_new, is_pd=is_pd, new_index=new_index, new_columns=new_columns)


def broadcast_to_array_of(arg1, arg2):
    """Bring first argument to the shape of an array of second argument."""
    arg1 = np.asarray(arg1)
    arg2 = np.asarray(arg2)
    if arg1.ndim == 0:
        arg1 = np.broadcast_to(arg1, arg2.shape, subok=True)[None, :]
    elif arg1.ndim == 1:
        arg1 = np.tile(arg1[:, None][:, None], (1, *arg2.shape))
    elif arg1.ndim == 2:
        if arg1.shape[1] != arg2.shape[0] or arg1.shape[2] != arg2.shape[1]:
            arg1 = np.broadcast_to(arg1, arg2.shape, subok=True)[None, :]
    return arg1.copy()  # deprecation warning


# ############# Indexing ############# #

def index_from_params(params, name=None):
    """Create index using params array."""
    check_ndim(params, (1, 3))

    if np.asarray(params).ndim == 1:
        # frame-wise params
        return pd.Index(params, name=name)
    else:
        # element-wise params per frame
        if np.array_equal(np.min(params, axis=(1, 2)), np.max(params, axis=(1, 2))):
            return pd.Index(params[:, 0, 0], name=name)
        else:
            return pd.Index(['mix_%d' % i for i in range(params.shape[0])], name=name)


def stack_indexes(index1, index2, drop_duplicates=True, **kwargs):
    """Stack indexes."""
    check_type(index1, pd.Index)
    check_type(index2, pd.Index)
    check_same_shape(index1, index2)

    tuples1 = index1.to_numpy()
    tuples2 = index2.to_numpy()

    names = []
    if isinstance(index1, pd.MultiIndex):
        tuples1 = list(zip(*tuples1))
        names.extend(index1.names)
    else:
        tuples1 = [tuples1]
        names.append(index1.name)
    if isinstance(index2, pd.MultiIndex):
        tuples2 = list(zip(*tuples2))
        names.extend(index2.names)
    else:
        tuples2 = [tuples2]
        names.append(index2.name)

    tuples = list(zip(*(tuples1 + tuples2)))
    multiindex = pd.MultiIndex.from_tuples(tuples, names=names)
    if drop_duplicates:
        multiindex = drop_duplicate_levels(multiindex, **kwargs)
    return multiindex


def combine_indexes(index1, index2, **kwargs):
    """Combine indexes using Cartesian product."""
    check_type(index1, pd.Index)
    check_type(index2, pd.Index)

    if len(index1) == 1:
        return index2
    elif len(index2) == 1:
        return index1

    tuples1 = np.repeat(index1.to_numpy(), len(index2))
    tuples2 = np.tile(index2.to_numpy(), len(index1))

    if isinstance(index1, pd.MultiIndex):
        index1 = pd.MultiIndex.from_tuples(tuples1, names=index1.names)
    else:
        index1 = pd.Index(tuples1, name=index1.name)
    if isinstance(index2, pd.MultiIndex):
        index2 = pd.MultiIndex.from_tuples(tuples2, names=index2.names)
    else:
        index2 = pd.Index(tuples2, name=index2.name)

    return stack_indexes(index1, index2, **kwargs)


def broadcast_combine_frames(df1, df2, drop_conflict=False, **kwargs):
    """Broadcast dataframes using Cartesian product."""
    check_type(df1, pd.DataFrame)
    check_type(df2, pd.DataFrame)

    values1 = np.repeat(df1.to_numpy(), len(df2.columns), axis=1)
    values2 = np.tile(df2.to_numpy(), len(df1.columns))
    columns = combine_indexes(df1.columns, df2.columns, **kwargs)
    df1 = pd.DataFrame(values1, index=df1.index, columns=columns)
    df2 = pd.DataFrame(values2, index=df2.index, columns=columns)
    if drop_conflict:
        keep_mask, columns = drop_conflict_values(columns, **kwargs)
        df1 = df1.loc[:, keep_mask]
        df2 = df2.loc[:, keep_mask]
        df1.columns = columns
        df2.columns = columns
    return df1, df2


def drop_conflict_values(index, **kwargs):
    """Drop index tuples that have conflicting values (different values with same level names)."""
    if not isinstance(index, pd.MultiIndex) and isinstance(index, pd.Index):
        return index
    check_type(index, pd.MultiIndex)
    if len(index.names) == len(set(index.names)):
        return index

    keep_mask = np.full(len(index), True)
    for name in set(index.names):
        name_values = []
        for i, name2 in enumerate(index.names):
            if name == name2:
                name_values.append(index.get_level_values(i).to_numpy().tolist())
        name_values = np.array(name_values)
        keep_mask &= np.sum(np.diff(name_values, axis=0), axis=0) == 0

    new_index = pd.MultiIndex.from_tuples(index.to_numpy()[keep_mask], names=index.names)
    return keep_mask, drop_duplicate_levels(new_index, **kwargs)


def drop_duplicate_levels(index, keep='last'):
    """Drop duplicate levels with the same name and values."""
    if not isinstance(index, pd.MultiIndex) and isinstance(index, pd.Index):
        return index
    check_type(index, pd.MultiIndex)

    levels = []
    levels_to_drop = []
    if keep == 'first':
        r = range(0, len(index.levels))
    elif keep == 'last':
        r = range(len(index.levels)-1, -1, -1)  # loop backwards
    for i in r:
        level = (index.levels[i].name, tuple(index.get_level_values(i).to_numpy().tolist()))
        if level not in levels:
            levels.append(level)
        else:
            levels_to_drop.append(i)
    return index.droplevel(levels_to_drop)


def drop_duplicate_columns(arg):
    """Drop duplicate columns with the same name."""
    check_type(arg, (pd.Series, pd.DataFrame))

    if is_frame(arg):
        return arg.loc[:, ~arg.columns.duplicated()]
    return arg


def clean_columns(arg, drop_levels=True, drop_columns=True, **kwargs):
    """Remove redundant columns."""
    check_type(arg, (pd.Series, pd.DataFrame))
    arg = to_2d(arg)

    if drop_levels:
        arg.columns = drop_duplicate_levels(arg.columns, **kwargs)
    if drop_columns:
        arg = drop_duplicate_columns(arg)
    if isinstance(arg.columns, pd.MultiIndex):
        levels_to_drop = []
        for i in range(len(arg.columns.levels)):
            if len(arg.columns.get_level_values(i).unique()) == 1:
                levels_to_drop.append(i)
        if len(levels_to_drop) == len(arg.columns.levels):
            arg = arg.iloc[:, 0]
        else:
            arg.columns = arg.columns.droplevel(levels_to_drop)
    if is_frame(arg):
        if isinstance(arg.columns, pd.Index):
            if len(arg.columns.unique()) == 1:
                arg = arg.iloc[:, 0]
    return arg


def rename_levels(index, name_dict):
    """Rename index/column levels."""
    for k, v in name_dict.items():
        if isinstance(index, pd.MultiIndex):
            if k in index.names:
                index = index.rename(v, level=k)
        else:
            if index.name == k:
                index.name = v
    return index


def unstack_to_array(arg):
    """Reshape object based on multi-index into a multi-dimensional array."""
    check_type(arg, (pd.Series, pd.DataFrame))
    sr = to_1d(arg)

    vals_idx_list = []
    for i in range(len(sr.index.levels)):
        vals = sr.index.get_level_values(i).to_numpy()
        unique_vals = np.unique(vals)
        idx_map = dict(zip(unique_vals, range(len(unique_vals))))
        vals_idx = list(map(lambda x: idx_map[x], vals))
        vals_idx_list.append(vals_idx)

    a = np.full(list(map(len, sr.index.levels)), np.nan)
    a[tuple(zip(vals_idx_list))] = sr.values
    return a


def make_symmetric(arg):
    """Make object symmetric along the diagonal."""
    check_type(arg, (pd.Series, pd.DataFrame))
    df = to_2d(arg)

    unique_index = np.unique(np.concatenate((df.columns, df.index)))
    df_out = pd.DataFrame(index=unique_index, columns=unique_index)
    df_out.loc[:, :] = df
    df_out[df_out.isnull()] = df.transpose()
    return df_out


def unstack_to_df(arg, symmetric=False):
    """Reshape object based on multi-index into dataframe."""
    check_type(arg, (pd.Series, pd.DataFrame))
    sr = to_1d(arg)

    index = sr.index.levels[0]
    columns = sr.index.levels[1]
    df = pd.DataFrame(unstack_to_array(sr), index=index, columns=columns)
    if symmetric:
        return make_symmetric(df)
    return df


# ############# Class decorators ############# #


def add_safe_nb_methods(*nb_funcs):
    def wrapper(cls):
        """Wrap numba functions as methods."""
        for nb_func in nb_funcs:
            def get_default_args(func):
                return {
                    k: v.default
                    for k, v in signature(func).parameters.items()
                    if v.default is not Parameter.empty
                }
            default_kwargs = get_default_args(nb_func)

            def array_operation(self, *args, nb_func=nb_func, default_kwargs=default_kwargs, **kwargs):
                if '_1d' in nb_func.__name__:
                    return self.wrap_array(nb_func(self.to_1d_array(), *args, **{**default_kwargs, **kwargs}))
                else:
                    # We work natively on 2d arrays
                    return self.wrap_array(nb_func(self.to_2d_array(), *args, **{**default_kwargs, **kwargs}))
            setattr(cls, nb_func.__name__.replace('_1d', '').replace('_nb', ''), array_operation)
        return cls
    return wrapper


def copy_func(f):
    """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
    g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                           argdefs=f.__defaults__,
                           closure=f.__closure__)
    g = update_wrapper(g, f)
    g.__kwdefaults__ = f.__kwdefaults__
    return g


def add_indexing(indexing_func):
    def wrapper(cls):
        """Add iloc, loc and plain brackets indexing to a class.

        Each indexing operation is forwarded to the underlying pandas objects and
        a new instance of the class with new pandas objects is created using the indexing_func."""

        class iLoc:
            def __init__(self, obj):
                self.obj = obj

            def __getitem__(self, key):
                return indexing_func(self.obj, lambda x: x.iloc.__getitem__(key))

        class Loc:
            def __init__(self, obj):
                self.obj = obj

            def __getitem__(self, key):
                return indexing_func(self.obj, lambda x: x.loc.__getitem__(key))

        @property
        def iloc(self):
            return self._iloc

        @property
        def loc(self):
            return self._loc

        def xs(self, *args, **kwargs):
            return indexing_func(self, lambda x: x.xs(*args, **kwargs))

        def __getitem__(self, key):
            return indexing_func(self, lambda x: x.__getitem__(key))

        orig_init_method = copy_func(cls.__init__)

        def __init__(self, *args, **kwargs):
            orig_init_method(self, *args, **kwargs)
            self._iloc = iLoc(self)
            self._loc = Loc(self)

        setattr(cls, '__init__', __init__)
        setattr(cls, '__getitem__', __getitem__)
        setattr(cls, 'iloc', iloc)
        setattr(cls, 'loc', loc)
        setattr(cls, 'xs', xs)
        return cls
    return wrapper


def loc_mapper(mapper, like_df, loc_pandas_func):
    """Mapper requires some pre and post-processing."""
    mapper = broadcast_to(mapper.to_frame().transpose(), like_df, index_from=1, columns_from=1)
    mapper = loc_pandas_func(mapper)
    if is_frame(mapper):
        mapper = pd.Series(mapper.values[0], index=mapper.columns)
    elif is_series(mapper):
        mapper = pd.Series([mapper.values[0]], index=[mapper.name])
    return mapper


def add_param_indexing(param_name, indexing_func):
    def wrapper(cls):
        """Add loc indexing of params to a class. Requires a mapper."""

        class ParamLoc:
            def __init__(self, obj, param_mapper, clean_columns=True, **clean_kwargs):
                check_type(param_mapper, pd.Series)

                self.obj = obj
                self.param_mapper = param_mapper
                self.clean_columns = clean_columns
                self.clean_kwargs = clean_kwargs

            def __call__(self, clean_columns=None, **clean_kwargs):
                if clean_columns is None:
                    clean_columns = self.clean_columns
                return self.__class__(self.obj, self.param_mapper, clean_columns=clean_columns, **clean_kwargs)

            def __getitem__(self, key):
                def loc_pandas_func(obj):
                    nonlocal key
                    if self.param_mapper.dtype == 'O':
                        # If params are objects, we must cast them to string first
                        param_mapper = self.param_mapper.astype(str)
                        # We must also cast the key to string
                        if isinstance(key, slice):
                            start = str(key.start) if key.start is not None else None
                            stop = str(key.stop) if key.stop is not None else None
                            key = slice(start, stop, key.step)
                        elif isinstance(key, list):
                            key = list(map(str, key))
                        else:
                            # Tuples, objects, etc.
                            key = str(key)
                    else:
                        param_mapper = self.param_mapper
                    # Use pandas to perform indexing
                    param_mapper = pd.Series(np.arange(len(param_mapper.index)), index=param_mapper.values)
                    indexes = param_mapper.loc.__getitem__(key)
                    if isinstance(indexes, pd.Series):
                        indexes = indexes.values
                    obj = obj.iloc[:, indexes]
                    if self.clean_columns:
                        obj = clean_columns(obj, **self.clean_kwargs)
                    return obj

                return indexing_func(self.obj, loc_pandas_func)

        @property
        def param_loc(self):
            return getattr(self, f'_{param_name}_loc')

        orig_init_method = copy_func(cls.__init__)

        def __init__(self, *args, **kwargs):
            orig_init_method(self, *args, **kwargs)
            mapper = getattr(self, f'{param_name}_mapper')
            setattr(self, f'_{param_name}_loc', ParamLoc(self, mapper))

        setattr(cls, '__init__', __init__)
        setattr(cls, f'{param_name}_loc', param_loc)
        return cls
    return wrapper

# ############# Property decorators ############# #


def cached_property(func):
    @wraps(func)
    def wrapper_decorator(*args, **kwargs):
        """Cache property to avoid recalculating it again and again."""
        obj = args[0]
        attr_name = '_' + func.__name__
        if hasattr(obj, attr_name):
            return getattr(obj, attr_name)
        else:
            to_be_cached = func(*args, **kwargs)
            setattr(obj, attr_name, to_be_cached)
            return to_be_cached
    return property(wrapper_decorator)


# ############# Custom accessors ############# #


class Base_Accessor():
    def __init__(self, obj):
        self._obj = obj._obj  # access pandas object
        self._validate(self._obj)

    dtype = None

    @classmethod
    def _validate(cls, obj):
        pass

    def validate(self):
        # Don't override it, just call it for the object to be instantiated
        pass

    @classmethod
    def empty(cls, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def empty_like(cls, *args, **kwargs):
        raise NotImplementedError

    def to_1d_array(self):
        return to_1d(self._obj.values)

    def to_2d_array(self):
        return to_2d(self._obj.values)

    def wrap_array(self, *args, **kwargs):
        raise NotImplementedError

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    def combine_with(self, *others, np_combine_func=None, **kwargs):
        check_not_none(np_combine_func)
        others = list(others)
        for i in range(len(others)):
            if isinstance(others[i], Base_Accessor):
                others[i] = others[i]._obj
        broadcasted = broadcast(self._obj, *others, **kwargs)

        def multi_np_combine_func(*broadcasted):
            result = None
            for i in range(1, len(broadcasted)):
                if result is None:
                    result = np_combine_func(broadcasted[i-1], broadcasted[i])
                else:
                    result = np_combine_func(result, broadcasted[i])
            return result

        return broadcasted[0].vbt.wrap_array(multi_np_combine_func(*list(map(np.asarray, broadcasted))))

    # Comparison operators
    def __eq__(self, other): return self.combine_with(other, np_combine_func=np.equal)
    def __ne__(self, other): return self.combine_with(other, np_combine_func=np.not_equal)
    def __lt__(self, other): return self.combine_with(other, np_combine_func=np.less)
    def __gt__(self, other): return self.combine_with(other, np_combine_func=np.greater)
    def __le__(self, other): return self.combine_with(other, np_combine_func=np.less_equal)
    def __ge__(self, other): return self.combine_with(other, np_combine_func=np.greater_equal)

    # Binary operators
    def __add__(self, other): return self.combine_with(other, np_combine_func=np.add)
    def __sub__(self, other): return self.combine_with(other, np_combine_func=np.subtract)
    def __mul__(self, other): return self.combine_with(other, np_combine_func=np.multiply)
    def __div__(self, other): return self.combine_with(other, np_combine_func=np.divide)
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rdiv__ = __div__

    # Boolean operators
    def __and__(self, other): return self.combine_with(other, np_combine_func=np.logical_and)
    def __or__(self, other): return self.combine_with(other, np_combine_func=np.logical_or)
    def __xor__(self, other): return self.combine_with(other, np_combine_func=np.logical_xor)


class Base_DFAccessor(Base_Accessor):

    @classmethod
    def _validate(cls, obj):
        check_type(obj, pd.DataFrame)

    @classmethod
    def empty(cls, shape, fill_value=np.nan, index=None, columns=None):
        return pd.DataFrame(
            np.full(shape, fill_value),
            index=index,
            columns=columns,
            dtype=cls.dtype)

    @classmethod
    def empty_like(cls, df, fill_value=np.nan):
        cls._validate(df)

        return cls.empty(
            df.shape,
            fill_value=fill_value,
            index=df.index,
            columns=df.columns)

    def wrap_array(self, a, index=None, columns=None):
        if index is None:
            index = self._obj.index
        if columns is None:
            columns = self._obj.columns
        # dtype should be set on array level
        return pd.DataFrame(a, index=index, columns=columns)


class Base_SRAccessor(Base_Accessor):
    # series is just a dataframe with one column
    # this way we don't have to define our custom functions for working with 1d data
    @classmethod
    def _validate(cls, obj):
        check_type(obj, pd.Series)

    @classmethod
    def empty(cls, size, fill_value=np.nan, index=None, name=None):
        return pd.Series(
            np.full(size, fill_value),
            index=index,
            name=name,
            dtype=cls.dtype)

    @classmethod
    def empty_like(cls, sr, fill_value=np.nan):
        cls._validate(sr)

        return cls.empty(
            sr.shape,
            fill_value=fill_value,
            index=sr.index,
            name=sr.name)

    def wrap_array(self, a, index=None, columns=None, name=None):
        if index is None:
            index = self._obj.index
        if name is None:
            name = self._obj.name
        # dtype should be set on array level
        if a.ndim == 1:
            return pd.Series(a, index=index, name=name)
        if a.shape[1] == 1:
            return pd.Series(a[:, 0], index=index, name=name)
        else:
            return pd.DataFrame(a, index=index, columns=columns)
