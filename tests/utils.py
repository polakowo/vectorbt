import hashlib

import numpy as np
import pandas as pd

SIZE_GRANULARITY_CASES = [
    (1.0000000006, 1e-9, 1.0),  # 1e9 lots + 0.6 lot
    (0.9999999996, 1e-9, 0.999999999),  # 0.4 lot short of 1e9 lots
    (1 + 6e-13, 1e-12, 1.0),  # 1e12 lots + 0.6 lot
    (1 - 2e-13, 1e-12, 0.999999999999),  # 0.2 lot short of 1e12 lots
    (0.001 * (1 - 1e-10), 0.001, np.nan),  # A real shortfall must not trade.
    (0.1 * (1 - 1e-10), 0.1, np.nan),
    (1000 * (1 - 1e-10), 1000.0, np.nan),
    (0.0999, 0.1, np.nan),
    (np.nextafter(0.001, 0), 0.001, np.nextafter(0.001, 0)),  # One ULP short fills the request.
    (np.nextafter(0.1, 0), 0.1, np.nextafter(0.1, 0)),
    (np.nextafter(1000.0, 0), 1000.0, np.nextafter(1000.0, 0)),
]


def make_index_representations(kind, representation, name="key"):
    """Build equivalent flat and wrapped indexes for alignment tests."""
    if kind == "string":
        keys = pd.Index(["A", "B", "C"], name=name)
    elif kind == "datetime":
        keys = pd.date_range("2020", periods=3, name=name)
    else:
        keys = pd.timedelta_range("0D", periods=3, name=name)
    if representation == "categorical":
        wrapped = pd.CategoricalIndex(keys, name=name)
    else:
        wrapped = pd.MultiIndex.from_arrays([keys])
    return keys, wrapped


def integer_diff(values, n):
    """Use Python integers as an overflow-free oracle for discrete differences."""
    expected = np.full(values.shape, np.nan)
    for row in range(len(values)):
        previous = row - n
        if 0 <= previous < len(values):
            for col in range(values.shape[1]):
                expected[row, col] = int(values[row, col]) - int(values[previous, col])
    return expected


# non-randomized hash function
hash = lambda s: int(hashlib.sha512(s.encode("utf-8")).hexdigest()[:16], 16)


def isclose(a, b, rel_tol=1e-06, abs_tol=0.0):
    if np.isnan(a) or np.isnan(b):
        return bool(np.isnan(a) and np.isnan(b))
    if np.isinf(a) or np.isinf(b):
        return bool(a == b)
    return bool(abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol))


def record_arrays_close(x, y):
    for field in x.dtype.names:
        np.testing.assert_allclose(x[field], y[field], rtol=1e-06)
