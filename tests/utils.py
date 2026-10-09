import hashlib

import numpy as np

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
