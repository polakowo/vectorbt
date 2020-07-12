import numpy as np
import hashlib

seed = 42

day_dt = np.timedelta64(86400000000000)

# non-randomized hash function
hash = lambda s: int(hashlib.sha512(s.encode('utf-8')).hexdigest()[:16], 16)


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    if np.isnan(a) == np.isnan(b):
        return True
    if np.isinf(a) == np.isinf(b):
        return True
    if a == b:
        return True
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)
