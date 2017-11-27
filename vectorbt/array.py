import numpy as np

def ffill(arr):
    # Forward fill
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]

def pct_change(arr):
    # Percent change
    return np.insert(np.diff(arr) / arr[:-1], 0, 0)

def fshift(arr, n, fill):
    # Forward shift
    return np.concatenate([np.array([fill] * n), arr[:-n]], axis=0)
