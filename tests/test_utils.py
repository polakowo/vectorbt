import numpy as np
import pandas as pd
from numba import njit
import pytest

from vectorbt import defaults
from vectorbt.utils import checks, config, decorators, math, array

from tests.utils import hash


# ############# config.py ############# #

class TestConfig:
    def test_config(self):
        conf = config.Config({'a': 0, 'b': {'c': 1}}, frozen=False)
        conf['b']['d'] = 2

        conf = config.Config({'a': 0, 'b': {'c': 1}}, frozen=True)
        conf['a'] = 2

        with pytest.raises(Exception) as e_info:
            conf['d'] = 2

        # go deeper
        conf['b']['c'] = 2

        with pytest.raises(Exception) as e_info:
            conf['b']['d'] = 2

    def test_merge_kwargs(self):
        assert config.merge_kwargs({'a': 1}, {'b': 2}) == {'a': 1, 'b': 2}
        assert config.merge_kwargs({'a': 1}, {'a': 2}) == {'a': 2}
        assert config.merge_kwargs({'a': {'b': 2}}, {'a': {'c': 3}}) == {'a': {'b': 2, 'c': 3}}
        assert config.merge_kwargs({'a': {'b': 2}}, {'a': {'b': 3}}) == {'a': {'b': 3}}


# ############# decorators.py ############# #

class TestDecorators:
    def test_class_or_instancemethod(self):
        class G:
            @decorators.class_or_instancemethod
            def g(self_or_cls):
                if isinstance(self_or_cls, type):
                    return True  # class
                return False  # instance

        assert G.g()
        assert not G().g()

    def test_custom_property(self):
        class G:
            @decorators.custom_property(some='key')
            def cache_me(self): return np.random.uniform()

        assert 'some' in G.cache_me.kwargs
        assert G.cache_me.kwargs['some'] == 'key'

    def test_custom_method(self):
        class G:
            @decorators.custom_method(some='key')
            def cache_me(self): return np.random.uniform()

        assert 'some' in G.cache_me.kwargs
        assert G.cache_me.kwargs['some'] == 'key'

    def test_cached_property(self):
        class G:
            @decorators.cached_property(some='key')
            def cache_me(self): return np.random.uniform()

        assert 'some' in G.cache_me.kwargs
        assert G.cache_me.kwargs['some'] == 'key'

        class G:
            @decorators.cached_property
            def cache_me(self): return np.random.uniform()

        g = G()

        # general caching
        cached_number = g.cache_me
        assert g.cache_me == cached_number

        # clear_cache method
        G.cache_me.clear_cache(g)
        cached_number2 = g.cache_me
        assert cached_number2 != cached_number
        assert g.cache_me == cached_number2

        # disabled locally
        G.cache_me.disabled = True
        cached_number3 = g.cache_me
        assert cached_number3 != cached_number2
        assert g.cache_me != cached_number3
        G.cache_me.disabled = False

        # disabled globally
        defaults.caching = False
        cached_number4 = g.cache_me
        assert cached_number4 != cached_number3
        assert g.cache_me != cached_number4
        defaults.caching = True

    def test_cached_method(self):
        class G:
            @decorators.cached_method(some='key')
            def cache_me(self): return np.random.uniform()

        assert 'some' in G.cache_me.kwargs
        assert G.cache_me.kwargs['some'] == 'key'

        class G:
            @decorators.cached_method
            def cache_me(self, b=10): return np.random.uniform() * 10

        g = G()

        # general caching
        cached_number = g.cache_me()
        assert g.cache_me() == cached_number

        # clear_cache method
        G.cache_me.clear_cache(g)
        cached_number2 = g.cache_me()
        assert cached_number2 != cached_number
        assert g.cache_me() == cached_number2

        # disabled locally
        G.cache_me.disabled = True
        cached_number3 = g.cache_me()
        assert cached_number3 != cached_number2
        assert g.cache_me() != cached_number3
        G.cache_me.disabled = False

        # disabled globally
        defaults.caching = False
        cached_number4 = g.cache_me()
        assert cached_number4 != cached_number3
        assert g.cache_me() != cached_number4
        defaults.caching = True

        # disabled by non-hashable args
        cached_number5 = g.cache_me(b=np.zeros(1))
        assert cached_number5 != cached_number4
        assert g.cache_me(b=np.zeros(1)) != cached_number5

    def test_traverse_attr_kwargs(self):
        class A:
            @decorators.custom_property(some_key=0)
            def a(self): pass

        class B:
            @decorators.cached_property(some_key=0, child_cls=A)
            def a(self): pass

            @decorators.custom_method(some_key=1)
            def b(self): pass

        class C:
            @decorators.cached_method(some_key=0, child_cls=B)
            def b(self): pass

            @decorators.custom_property(some_key=1)
            def c(self): pass

        assert hash(str(decorators.traverse_attr_kwargs(C))) == 16728515581653529580
        assert hash(str(decorators.traverse_attr_kwargs(C, key='some_key'))) == 16728515581653529580
        assert hash(str(decorators.traverse_attr_kwargs(C, key='some_key', value=1))) == 703070484833749378
        assert hash(str(decorators.traverse_attr_kwargs(C, key='some_key', value=(0, 1)))) == 16728515581653529580


# ############# checks.py ############# #

class TestChecks:
    def test_is_pandas(self):
        assert not checks.is_pandas(0)
        assert not checks.is_pandas(np.array([0]))
        assert checks.is_pandas(pd.Series([1, 2, 3]))
        assert checks.is_pandas(pd.DataFrame([1, 2, 3]))

    def test_is_series(self):
        assert not checks.is_series(0)
        assert not checks.is_series(np.array([0]))
        assert checks.is_series(pd.Series([1, 2, 3]))
        assert not checks.is_series(pd.DataFrame([1, 2, 3]))

    def test_is_frame(self):
        assert not checks.is_frame(0)
        assert not checks.is_frame(np.array([0]))
        assert not checks.is_frame(pd.Series([1, 2, 3]))
        assert checks.is_frame(pd.DataFrame([1, 2, 3]))

    def test_is_array(self):
        assert not checks.is_array(0)
        assert checks.is_array(np.array([0]))
        assert checks.is_array(pd.Series([1, 2, 3]))
        assert checks.is_array(pd.DataFrame([1, 2, 3]))

    def test_is_numba_func(self):
        def test_func(x):
            return x

        @njit
        def test_func_nb(x):
            return x

        assert not checks.is_numba_func(test_func)
        assert checks.is_numba_func(test_func_nb)

    def test_is_hashable(self):
        assert checks.is_hashable(2)
        assert not checks.is_hashable(np.asarray(2))

    def test_is_index_equal(self):
        assert checks.is_index_equal(
            pd.Index([0]),
            pd.Index([0])
        )
        assert not checks.is_index_equal(
            pd.Index([0]),
            pd.Index([1])
        )
        assert not checks.is_index_equal(
            pd.Index([0], name='name'),
            pd.Index([0])
        )
        assert not checks.is_index_equal(
            pd.MultiIndex.from_arrays([[0], [1]]),
            pd.Index([0])
        )
        assert checks.is_index_equal(
            pd.MultiIndex.from_arrays([[0], [1]]),
            pd.MultiIndex.from_arrays([[0], [1]])
        )
        assert checks.is_index_equal(
            pd.MultiIndex.from_arrays([[0], [1]], names=['name1', 'name2']),
            pd.MultiIndex.from_arrays([[0], [1]], names=['name1', 'name2'])
        )
        assert not checks.is_index_equal(
            pd.MultiIndex.from_arrays([[0], [1]], names=['name1', 'name2']),
            pd.MultiIndex.from_arrays([[0], [1]], names=['name3', 'name4'])
        )

    def test_is_default_index(self):
        assert checks.is_default_index(pd.DataFrame([[1, 2, 3]]).columns)
        assert checks.is_default_index(pd.Series([1, 2, 3]).to_frame().columns)
        assert checks.is_default_index(pd.Index([0, 1, 2]))
        assert not checks.is_default_index(pd.Index([0, 1, 2], name='name'))

    def test_is_equal(self):
        assert checks.is_equal(np.arange(3), np.arange(3), np.array_equal)
        assert not checks.is_equal(np.arange(3), None, np.array_equal)
        assert not checks.is_equal(None, np.arange(3), np.array_equal)
        assert checks.is_equal(None, None, np.array_equal)

    def test_assert_value_in(self):
        checks.assert_value_in(0, (0, 1))
        with pytest.raises(Exception) as e_info:
            checks.assert_value_in(2, (0, 1))

    def test_assert_numba_func(self):
        def test_func(x):
            return x

        @njit
        def test_func_nb(x):
            return x

        checks.assert_numba_func(test_func_nb)
        with pytest.raises(Exception) as e_info:
            checks.assert_numba_func(test_func)

    def test_assert_not_none(self):
        checks.assert_not_none(0)
        with pytest.raises(Exception) as e_info:
            checks.assert_not_none(None)

    def test_assert_type(self):
        checks.assert_type(0, int)
        checks.assert_type(np.zeros(1), (np.ndarray, pd.Series))
        checks.assert_type(pd.Series([1, 2, 3]), (np.ndarray, pd.Series))
        with pytest.raises(Exception) as e_info:
            checks.assert_type(pd.DataFrame([1, 2, 3]), (np.ndarray, pd.Series))

    def test_assert_subclass(self):
        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        checks.assert_subclass(B, A)
        checks.assert_subclass(C, B)
        checks.assert_subclass(C, A)
        with pytest.raises(Exception) as e_info:
            checks.assert_subclass(A, B)

    def test_assert_same_type(self):
        checks.assert_same_type(0, 1)
        checks.assert_same_type(np.zeros(1), np.empty(1))
        with pytest.raises(Exception) as e_info:
            checks.assert_type(0, np.zeros(1))

    def test_assert_dtype(self):
        checks.assert_dtype(np.zeros(1), np.float)
        checks.assert_dtype(pd.Series([1, 2, 3]), np.int)
        checks.assert_dtype(pd.DataFrame({'a': [1, 2], 'b': [3, 4]}), np.int)
        with pytest.raises(Exception) as e_info:
            checks.assert_dtype(pd.DataFrame({'a': [1, 2], 'b': [3., 4.]}), np.int)

    def test_assert_subdtype(self):
        checks.assert_subdtype([0], np.number)
        checks.assert_subdtype(np.array([1, 2, 3]), np.number)
        checks.assert_subdtype(pd.DataFrame({'a': [1, 2], 'b': [3., 4.]}), np.number)
        with pytest.raises(Exception) as e_info:
            checks.assert_subdtype(np.array([1, 2, 3]), np.float)
        with pytest.raises(Exception) as e_info:
            checks.assert_subdtype(pd.DataFrame({'a': [1, 2], 'b': [3., 4.]}), np.float)

    def test_assert_same_dtype(self):
        checks.assert_same_dtype([1], [1, 1, 1])
        checks.assert_same_dtype(pd.Series([1, 2, 3]), pd.DataFrame([[1, 2, 3]]))
        checks.assert_same_dtype(pd.DataFrame([[1, 2, 3.]]), pd.DataFrame([[1, 2, 3.]]))
        with pytest.raises(Exception) as e_info:
            checks.assert_same_dtype(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[1, 2, 3.]]))

    def test_assert_ndim(self):
        checks.assert_ndim(0, 0)
        checks.assert_ndim(np.zeros(1), 1)
        checks.assert_ndim(pd.Series([1, 2, 3]), (1, 2))
        checks.assert_ndim(pd.DataFrame([1, 2, 3]), (1, 2))
        with pytest.raises(Exception) as e_info:
            checks.assert_ndim(np.zeros((3, 3, 3)), (1, 2))

    def test_assert_same_len(self):
        checks.assert_same_len([[1]], [[2]])
        checks.assert_same_len([[1]], [[2, 3]])
        with pytest.raises(Exception) as e_info:
            checks.assert_same_len([[1]], [[2], [3]])

    def test_assert_same_shape(self):
        checks.assert_same_shape(0, 1)
        checks.assert_same_shape([1, 2, 3], np.asarray([1, 2, 3]))
        checks.assert_same_shape([1, 2, 3], pd.Series([1, 2, 3]))
        checks.assert_same_shape(np.zeros((3, 3)), pd.Series([1, 2, 3]), axis=0)
        checks.assert_same_shape(np.zeros((2, 3)), pd.Series([1, 2, 3]), axis=(1, 0))
        with pytest.raises(Exception) as e_info:
            checks.assert_same_shape(np.zeros((2, 3)), pd.Series([1, 2, 3]), axis=(0, 1))

    def test_assert_same_index(self):
        index = ['a', 'b', 'c']
        checks.assert_same_index(pd.Series([1, 2, 3], index=index), pd.DataFrame([1, 2, 3], index=index))
        with pytest.raises(Exception) as e_info:
            checks.assert_same_index(pd.Series([1, 2, 3]), pd.DataFrame([1, 2, 3], index=index))

    def test_assert_same_columns(self):
        columns = ['a', 'b', 'c']
        checks.assert_same_columns(pd.DataFrame([[1, 2, 3]], columns=columns), pd.DataFrame([[1, 2, 3]], columns=columns))
        with pytest.raises(Exception) as e_info:
            checks.assert_same_columns(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[1, 2, 3]], columns=columns))

    def test_assert_same_meta(self):
        index = ['x', 'y', 'z']
        columns = ['a', 'b', 'c']
        checks.assert_same_meta(np.array([1, 2, 3]), np.array([1, 2, 3]))
        checks.assert_same_meta(pd.Series([1, 2, 3], index=index), pd.Series([1, 2, 3], index=index))
        checks.assert_same_meta(pd.DataFrame([[1, 2, 3]], columns=columns), pd.DataFrame([[1, 2, 3]], columns=columns))
        with pytest.raises(Exception) as e_info:
            checks.assert_same_meta(pd.Series([1, 2]), pd.DataFrame([1, 2]))

        with pytest.raises(Exception) as e_info:
            checks.assert_same_meta(pd.DataFrame([1, 2]), pd.DataFrame([1, 2, 3]))

        with pytest.raises(Exception) as e_info:
            checks.assert_same_meta(pd.DataFrame([1, 2, 3]), pd.DataFrame([1, 2, 3], index=index))

        with pytest.raises(Exception) as e_info:
            checks.assert_same_meta(pd.DataFrame([[1, 2, 3]]), pd.DataFrame([[1, 2, 3]], columns=columns))

    def test_assert_same(self):
        index = ['x', 'y', 'z']
        columns = ['a', 'b', 'c']
        checks.assert_same(np.array([1, 2, 3]), np.array([1, 2, 3]))
        checks.assert_same(pd.Series([1, 2, 3], index=index), pd.Series([1, 2, 3], index=index))
        checks.assert_same(pd.DataFrame([[1, 2, 3]], columns=columns), pd.DataFrame([[1, 2, 3]], columns=columns))
        with pytest.raises(Exception) as e_info:
            checks.assert_same(np.array([1, 2]), np.array([1, 2, 3]))

    def test_assert_level_not_exists(self):
        i = pd.Index(['x', 'y', 'z'], name='i')
        multi_i = pd.MultiIndex.from_arrays([['x', 'y', 'z'], ['x2', 'y2', 'z2']], names=['i', 'i2'])
        checks.assert_level_not_exists(i, 'i2')
        checks.assert_level_not_exists(multi_i, 'i3')
        with pytest.raises(Exception) as e_info:
            checks.assert_level_not_exists(i, 'i')
            checks.assert_level_not_exists(multi_i, 'i')


# ############# math.py ############# #

class TestMath:
    @pytest.mark.parametrize(
        "test_func",
        [math.is_close, math.is_close_nb],
    )
    def test_is_close(self, test_func):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert test_func(a, a)
        assert test_func(a, b)
        assert test_func(-a, -b)
        assert not test_func(-a, b)
        assert not test_func(a, -b)
        assert test_func(1e10 + a, 1e10 + b)

        # test np.nan
        assert not test_func(np.nan, b)
        assert not test_func(a, np.nan)

        # test np.inf
        assert not test_func(np.inf, b)
        assert not test_func(a, np.inf)
        assert not test_func(-np.inf, b)
        assert not test_func(a, -np.inf)
        assert not test_func(-np.inf, -np.inf)
        assert not test_func(np.inf, np.inf)
        assert not test_func(-np.inf, np.inf)

    @pytest.mark.parametrize(
        "test_func",
        [math.is_close_or_less, math.is_close_or_less_nb],
    )
    def test_is_close_or_less(self, test_func):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert test_func(a, a)
        assert test_func(a, b)
        assert test_func(-a, -b)
        assert test_func(-a, b)
        assert not test_func(a, -b)
        assert test_func(1e10 + a, 1e10 + b)

        # test np.nan
        assert not test_func(np.nan, b)
        assert not test_func(a, np.nan)

        # test np.inf
        assert not test_func(np.inf, b)
        assert test_func(a, np.inf)
        assert test_func(-np.inf, b)
        assert not test_func(a, -np.inf)
        assert not test_func(-np.inf, -np.inf)
        assert not test_func(np.inf, np.inf)
        assert test_func(-np.inf, np.inf)

    @pytest.mark.parametrize(
        "test_func",
        [math.is_less, math.is_less_nb],
    )
    def test_is_less(self, test_func):
        a = 0.3
        b = 0.1 + 0.2

        # test scalar
        assert not test_func(a, a)
        assert not test_func(a, b)
        assert not test_func(-a, -b)
        assert test_func(-a, b)
        assert not test_func(a, -b)
        assert not test_func(1e10 + a, 1e10 + b)

        # test np.nan
        assert not test_func(np.nan, b)
        assert not test_func(a, np.nan)

        # test np.inf
        assert not test_func(np.inf, b)
        assert test_func(a, np.inf)
        assert test_func(-np.inf, b)
        assert not test_func(a, -np.inf)
        assert not test_func(-np.inf, -np.inf)
        assert not test_func(np.inf, np.inf)
        assert test_func(-np.inf, np.inf)


# ############# array.py ############# #

class TestArray:
    def test_is_sorted(self):
        assert array.is_sorted(np.array([0, 1, 2, 3, 4]))
        assert array.is_sorted(np.array([0, 1]))
        assert array.is_sorted(np.array([0]))
        assert not array.is_sorted(np.array([1, 0]))
        assert not array.is_sorted(np.array([0, 1, 2, 4, 3]))
        # nb
        assert array.is_sorted_nb(np.array([0, 1, 2, 3, 4]))
        assert array.is_sorted_nb(np.array([0, 1]))
        assert array.is_sorted_nb(np.array([0]))
        assert not array.is_sorted_nb(np.array([1, 0]))
        assert not array.is_sorted_nb(np.array([0, 1, 2, 4, 3]))