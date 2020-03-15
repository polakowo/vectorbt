import numpy as np
import pandas as pd
from functools import wraps, reduce, update_wrapper
from inspect import signature, Parameter
import types

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
    if isinstance(arg, pd.DataFrame):
        if (arg.dtypes != dtype).any():
            raise ValueError(f"Data type must be {dtype} for all columns")
    else:
        if arg.dtype != dtype:
            raise ValueError(f"Data type must be {dtype}")


def check_same_type(arg1, arg2):
    if type(arg1) != type(arg2):
        raise ValueError(f"Must have the same type")


def check_same_dtype(arg1, arg2):
    if isinstance(arg1, pd.DataFrame) and not isinstance(arg2, pd.DataFrame):
        if (arg1.dtypes == arg2.dtype).all():
            return
    elif not isinstance(arg1, pd.DataFrame) and isinstance(arg2, pd.DataFrame):
        if (arg2.dtypes == arg1.dtype).all():
            return
    elif isinstance(arg1, pd.DataFrame) and isinstance(arg2, pd.DataFrame):
        if (arg1.dtypes == arg2.dtypes).all():
            return
    else:
        if arg1.dtype == arg2.dtype:
            return
    raise ValueError(f"Data types do not match")


def check_ndim(arg, ndims):
    if not isinstance(arg, (np.ndarray, pd.Series, pd.DataFrame)):
        arg = np.asarray(arg)
    if isinstance(ndims, tuple):
        if arg.ndim not in ndims:
            raise ValueError(f"Number of dimensions must be one of {ndims}")
    else:
        if arg.ndim != ndims:
            raise ValueError(f"Number of dimensions must be {ndims}")


def check_same_shape(arg1, arg2, along_axis=None):
    if not isinstance(arg1, (np.ndarray, pd.Series, pd.DataFrame)):
        arg1 = np.asarray(arg1)
    if not isinstance(arg2, (np.ndarray, pd.Series, pd.DataFrame)):
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
    if isinstance(arg1, (pd.Series, pd.DataFrame)):
        check_same_index(arg1, arg2)
        if isinstance(arg1, pd.DataFrame):
            check_same_columns(arg1, arg2)
        else:
            check_same_columns(to_2d(arg1), to_2d(arg2))
    if check_dtype:
        check_same_dtype(arg1, arg2)


def check_same(arg1, arg2):
    if isinstance(arg1, (pd.Series, pd.DataFrame)):
        if arg1.equals(arg2):
            return
    else:
        if np.array_equal(arg1, arg2):
            return
    raise ValueError(f"Must have the same type and values")


# ############# Broadcasting ############# #


def to_1d(arg):
    """Reshape argument to one dimension."""
    if not isinstance(arg, (np.ndarray, pd.Series, pd.DataFrame)):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        if arg.shape[1] == 1:
            if isinstance(arg, pd.DataFrame):
                return arg.iloc[:, 0]
            return arg[:, 0]
    if arg.ndim == 1:
        return arg
    elif arg.ndim == 0:
        return arg.reshape((1,))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 1 dimension")


def to_2d(arg, expand_axis=1):
    """Reshape argument to two dimensions."""
    if not isinstance(arg, (np.ndarray, pd.Series, pd.DataFrame)):
        arg = np.asarray(arg)
    if arg.ndim == 2:
        return arg
    elif arg.ndim == 1:
        if isinstance(arg, pd.Series):
            if expand_axis == 0:
                return pd.DataFrame(arg.values[None, :], columns=arg.index)
            elif expand_axis == 1:
                return arg.to_frame()
        return np.expand_dims(arg, expand_axis)
    elif arg.ndim == 0:
        return arg.reshape((1, 1))
    raise ValueError(f"Cannot reshape a {arg.ndim}-dimensional array to 2 dimensions")


def _broadcast_index(old_index, new_size):
    """Broadcast index/columns."""
    if isinstance(old_index[0], str):
        return ['%s_%d' % (old_index[0], i) for i in range(new_size)]
    else:
        return None


def _wrap_broadcasted(old_arg, new_arg, new_columns=None):
    """Transform newly broadcasted array to match the type of the original object."""
    if old_arg.shape == new_arg.shape:
        return old_arg
    if isinstance(old_arg, pd.DataFrame):
        return pd.DataFrame(new_arg, index=old_arg.index, columns=new_columns)
    return new_arg


def broadcast_to(arg1, arg2):
    """Bring the first argument to the shape of the second argument."""
    # All pandas objects must have the same index
    if isinstance(arg1, (pd.Series, pd.DataFrame)) and isinstance(arg2, (pd.Series, pd.DataFrame)):
        check_same_index(arg1, arg2)

    is_1d = True
    if not isinstance(arg1, (np.ndarray, pd.Series, pd.DataFrame)):
        arg1 = np.asarray(arg1)
    if not isinstance(arg2, (np.ndarray, pd.Series, pd.DataFrame)):
        arg2 = np.asarray(arg2)
    if arg2.ndim > 1:
        is_1d = False
    new_columns = None
    if not is_1d:
        if isinstance(arg1, pd.Series):
            arg1 = arg1.to_frame()
        if isinstance(arg1, pd.DataFrame) and isinstance(arg2, pd.DataFrame):
            if len(arg2.columns) > len(arg1.columns):
                new_columns = arg2.columns
    arg1_new = np.broadcast_to(arg1, arg2.shape, subok=True).copy()
    return _wrap_broadcasted(arg1, arg1_new, new_columns=new_columns)


def broadcast(*args):
    """Bring multiple arguments to the same shape."""
    # All pandas objects must have the same index
    pd_args = [arg for arg in args if isinstance(arg, (pd.Series, pd.DataFrame))]
    if len(pd_args) > 1:
        for i in range(1, len(pd_args)):
            check_same_index(pd_args[i-1], pd_args[i])

    is_1d = True
    args = list(args)
    # Convert to np.ndarray object if not numpy or pandas
    for i in range(len(args)):
        if not isinstance(args[i], (np.ndarray, pd.Series, pd.DataFrame)):
            args[i] = np.asarray(args[i])
    # Check if we need to switch to 2d
    for arg in args:
        if arg.ndim > 1:
            is_1d = False
            break
    # If in 2d mode, convert all pd.Series objects to pd.DataFrame
    new_columns = None
    if not is_1d:
        for i in range(len(args)):
            if isinstance(args[i], pd.Series):
                args[i] = args[i].to_frame()
            if isinstance(args[i], pd.DataFrame):
                if new_columns is None:
                    new_columns = args[i].columns
                if len(args[i].columns) > len(new_columns):
                    new_columns = args[i].columns
    # Perform broadcasting operation
    new_args = np.broadcast_arrays(*args, subok=True)
    # Bring arrays to their old types (e.g. array -> pandas)
    for i in range(len(new_args)):
        old_arg = args[i].copy()
        new_arg = new_args[i].copy()
        args[i] = _wrap_broadcasted(old_arg, new_arg, new_columns=new_columns)
    return args


def broadcast_to_array_of(arg1, arg2):
    """Bring the first argument to the shape of an array of the second argument."""
    if arg1 is not None and arg2 is not None:
        if not isinstance(arg1, np.ndarray):
            arg1 = np.asarray(arg1)
        if not isinstance(arg2, np.ndarray):
            arg2 = np.asarray(arg2)
        if arg1.ndim == 0:
            arg1 = np.broadcast_to(arg1, arg2.shape, subok=True)[None, :]
        elif arg1.ndim == 1:
            arg1 = np.tile(arg1[:, None][:, None], (1, *arg2.shape))
        elif arg1.ndim == 2:
            if arg1.shape[1] != arg2.shape[0] or arg1.shape[2] != arg2.shape[1]:
                arg1 = np.broadcast_to(arg1, arg2.shape, subok=True)[None, :]
        return arg1.copy()  # deprecation warning
    return None

def broadcast_and_combine(*args, combine_func=None):
    """Broadcast multiple arguments and apply a combiner on each successive pair."""
    check_not_none(combine_func)

    broadcasted = broadcast(*args)
    result = None
    for i in range(1, len(broadcasted)):
        if result is None:
            result = combine_func(broadcasted[i-1], broadcasted[i])
        else:
            result = combine_func(result, broadcasted[i])
    return result

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


def add_indexing_methods(indexing_func):
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

        def __getitem__(self, key):
            return indexing_func(self, lambda x: x.__getitem__(key))

        def copy_func(f):
            """Based on http://stackoverflow.com/a/6528148/190597 (Glenn Maynard)"""
            g = types.FunctionType(f.__code__, f.__globals__, name=f.__name__,
                                   argdefs=f.__defaults__,
                                   closure=f.__closure__)
            g = update_wrapper(g, f)
            g.__kwdefaults__ = f.__kwdefaults__
            return g

        orig_init_method = copy_func(cls.__init__)

        def __init__(self, *args, **kwargs):
            orig_init_method(self, *args, **kwargs)
            self._iloc = iLoc(self)
            self._loc = Loc(self)

        setattr(cls, '__init__', __init__)
        setattr(cls, '__getitem__', __getitem__)
        setattr(cls, 'iloc', iloc)
        setattr(cls, 'loc', loc)
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

    def tile(self, n, new_columns=None):
        return self.wrap_array(np.tile(self.to_2d_array(), (1, n)), columns=new_columns)

    def broadcast_to(self, other):
        return broadcast_to(self._obj, other)

    def broadcast(self, *others):
        return broadcast(self._obj, *others)

    def broadcast_and_combine(self, *others, combine_func=None):
        return broadcast_and_combine(self._obj, *others, combine_func=combine_func)

    def plot(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def index_from_params(cls, params, name=None):
        # Checks and preprocessing
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

    @classmethod
    def stack_indexes(cls, index1, index2):
        """Stack indexes."""
        # Checks and preprocessing
        check_type(index1, pd.Index)
        check_type(index2, pd.Index)
        check_same_shape(index1, index2)

        tuples1 = index1.to_numpy()
        tuples2 = index2.to_numpy()

        if isinstance(tuples1[0], tuple):
            tuples1 = list(zip(*tuples1))
        else:
            tuples1 = [tuples1.tolist()]
        if isinstance(tuples2[0], tuple):
            tuples2 = list(zip(*tuples2))
        else:
            tuples2 = [tuples2.tolist()]
        tuples = list(zip(*(tuples1 + tuples2)))

        names = []
        if isinstance(index1, pd.MultiIndex):
            names.extend(index1.names)
        else:
            names.append(index1.name)
        if isinstance(index2, pd.MultiIndex):
            names.extend(index2.names)
        else:
            names.append(index2.name)

        return pd.MultiIndex.from_tuples(tuples, names=names)

    @classmethod
    def combine_indexes(cls, index1, index2):
        """Combine indexes using Cartesian product."""
        # Checks and preprocessing
        check_type(index1, pd.Index)
        check_type(index2, pd.Index)

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

        return cls.stack_indexes(index1, index2)

    def stack_columns(self, columns):
        """Stack columns on top of object's columns."""
        # Checks and preprocessing
        df = to_2d(self._obj)
        check_type(columns, pd.Index)

        return self.stack_indexes(columns, df.columns)

    def combine_columns(self, columns):
        """Stack columns on top of object's columns using a cartesian product."""
        # Checks and preprocessing
        df = to_2d(self._obj)
        check_type(columns, pd.Index)

        if len(columns) == 1:
            return df.columns
        elif len(df.columns) == 1:
            return columns
        else:
            return self.combine_indexes(columns, df.columns)

    def unstack_to_array(self):
        """Reshape object based on multi-index into a multi-dimensional array."""
        # Checks and preprocessing
        sr = to_1d(self._obj)

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

    def make_symmetric(self):
        """Make object symmetric across the diagonal."""
        # Checks and preprocessing
        df = to_2d(self._obj)

        unique_index = np.unique(np.concatenate((df.columns, df.index)))
        df_out = pd.DataFrame(index=unique_index, columns=unique_index)
        df_out.loc[:, :] = df
        df_out[df_out.isnull()] = df.transpose()
        return df_out

    def unstack_to_df(self, symmetric=False):
        """Reshape object based on multi-index into dataframe."""
        # Checks and preprocessing
        sr = to_1d(self._obj)

        index = sr.index.levels[0]
        columns = sr.index.levels[1]
        df = pd.DataFrame(self.unstack_to_array(), index=index, columns=columns)
        if symmetric:
            df = df.vbt.make_symmetric()
        return df


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
