import numpy as np


def rshift(a, n=1, fill=0):
    """Shift to the right."""
    rolled = np.roll(a, n)
    rolled[:n] = fill
    return rolled


def lshift(a, n=1, fill=0):
    """Shift to the left."""
    rolled = np.roll(a, -n)
    rolled[-n:] = fill
    return rolled


def ffill(a, fill=0):
    """Fill zeros with the last non-zero value."""
    prev = np.arange(len(a))
    prev[a == 0] = fill
    prev = np.maximum.accumulate(prev)
    return a[prev]


def rolling(a, window=None):
    """Rolling window over the array.
    Produce the pandas result when min_periods=1."""
    if window is None: # expanding
        window = a.shape[0]
    a = np.insert(a.astype(float), 0, np.full(window-1, np.nan))
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    strided = np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    return strided  # raw output


def rolling_mean(a, window=None):
    return np.nanmean(rolling(a, window=window), axis=1)

def rolling_std(a, window=None):
    return np.nanstd(rolling(a, window=window), axis=1)

def rolling_max(a, window=None):
    return np.nanmax(rolling(a, window=window), axis=1)


def ewma(a, window=None):
    """Exponential weighted moving average.
    Produce the pandas result when min_periods=1 and adjust=False.

    https://stackoverflow.com/a/42926270
    """
    if window is None: # expanding
        window = a.shape[0]
    alpha = 2 / (window + 1.0)
    alpha_rev = 1 - alpha
    n = a.shape[0]
    pows = alpha_rev ** (np.arange(n + 1))
    scale_arr = 1 / pows[:-1]
    offset = a[0] * pows[1:]
    pw0 = alpha * alpha_rev ** (n-1)
    mult = a * pw0 * scale_arr
    cumsums = mult.cumsum()
    out = offset + cumsums * scale_arr[::-1]
    return out

def pct_change(a):
    """pd.pct_change in NumPy."""
    return np.insert(np.diff(a) / a[:-1], 0, 0)
