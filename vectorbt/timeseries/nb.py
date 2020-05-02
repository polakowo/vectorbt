"""Numba-compiled 1D and 2D functions for working with time series.

!!! note
    Input can be of any data type, but output is always `numpy.float64`.

    `vectorbt` treats matrices as first-level citizens. Functions that work exclusively on 
    1D arrays have suffix `_1d`. All other functions work on 2D arrays only. 
    Data is processed in pandas fashion, that is, along index (axis 0).
    
    Rolling functions by default have `minp` set to the window size."""

from numba import njit, f8, i8, b1, optional
import numpy as np


@njit(cache=True)
def prepend_1d_nb(a, n, value):
    """Prepend `value` to `a` `n` times."""
    result = np.empty(a.shape[0]+n, dtype=f8)
    result[:n] = value
    result[n:] = a
    return result


@njit(cache=True)
def prepend_nb(a, n, value):
    """2D version of `prepend_1d_nb`."""
    result = np.empty((a.shape[0]+n, a.shape[1]), dtype=f8)
    result[:n, :] = value
    result[n:, :] = a
    return result


@njit(cache=True)
def set_by_mask_1d_nb(a, mask, value):
    """Set each element in `a` to `value` by boolean mask `mask`."""
    result = a.astype(f8)
    result[mask] = value
    return result


@njit(cache=True)
def set_by_mask_nb(a, mask, value):
    """2D version of `set_by_mask_1d_nb`."""
    result = a.astype(f8)
    for col in range(result.shape[1]):
        result[mask[:, col], col] = value
    return result


@njit(cache=True)
def set_by_mask_mult_1d_nb(a, mask, values):
    """Set each element in `a` to the corresponding element in `values` by boolean mask `mask`.

    `values` must be of the same shape as in `a`."""
    result = a.astype(f8)
    result[mask] = values[mask]
    return result


@njit(cache=True)
def set_by_mask_mult_nb(a, mask, values):
    """2D version of `set_by_mask_mult_1d_nb`."""
    result = a.astype(f8)
    for col in range(result.shape[1]):
        result[mask[:, col], col] = values[mask[:, col], col]
    return result


@njit(cache=True)
def fillna_1d_nb(a, value):
    """Replace NaNs in `a` with `value`.

    Numba equivalent to `pd.Series(a).fillna(value)`."""
    return set_by_mask_1d_nb(a, np.isnan(a), value)


@njit(cache=True)
def fillna_nb(a, value):
    """2D version of `fillna_1d_nb`."""
    return set_by_mask_nb(a, np.isnan(a), value)


@njit(cache=True)
def fshift_1d_nb(a, n):
    """Shift forward `a` by `n` positions.

    Numba equivalent to `pd.Series(a).shift(value)`."""
    result = np.empty_like(a, dtype=f8)
    result[:n] = np.nan
    result[n:] = a[:-n]
    return result


@njit(cache=True)
def fshift_nb(a, n):
    """2D version of `fshift_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    result[:n, :] = np.nan
    result[n:, :] = a[:-n, :]
    return result


@njit(cache=True)
def diff_1d_nb(a):
    """Calculate the 1-th discrete difference of `a`.

    Numba equivalent to `pd.Series(a).diff()`."""
    result = np.empty_like(a, dtype=f8)
    result[0] = np.nan
    result[1:] = a[1:] - a[:-1]
    return result


@njit(cache=True)
def diff_nb(a):
    """2D version of `diff_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    result[0, :] = np.nan
    result[1:, :] = a[1:, :] - a[:-1, :]
    return result


@njit(cache=True)
def pct_change_1d_nb(a):
    """Calculate the percentage change of `a`.

    Numba equivalent to `pd.Series(a).pct_change()`."""
    result = np.empty_like(a, dtype=f8)
    result[0] = np.nan
    result[1:] = a[1:] / a[:-1] - 1
    return result


@njit(cache=True)
def pct_change_nb(a):
    """2D version of `pct_change_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    result[0, :] = np.nan
    result[1:, :] = a[1:, :] / a[:-1, :] - 1
    return result


@njit(cache=True)
def ffill_1d_nb(a):
    """Fill NaNs in `a` by propagating last valid observation forward.

    Numba equivalent to `pd.Series(a).fillna(method='ffill')`."""
    result = np.empty_like(a, dtype=f8)
    lastval = a[0]
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            result[i] = lastval
        else:
            result[i] = a[i]
            lastval = result[i]
    return result


@njit(cache=True)
def ffill_nb(a):
    """2D version of `ffill_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = ffill_1d_nb(a[:, col])
    return result


@njit(cache=True)
def cumsum_1d_nb(a):
    """Calculate cumulative sum of `a`.

    Numba equivalent to `pd.Series(a).cumsum()`."""
    result = np.empty_like(a, dtype=f8)
    cumsum = 0
    for i in range(a.shape[0]):
        if ~np.isnan(a[i]):
            cumsum += a[i]
            result[i] = cumsum
        else:
            result[i] = np.nan
    return result


@njit(cache=True)
def cumsum_nb(a):
    """2D version of `cumsum_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = cumsum_1d_nb(a[:, col])
    return result


@njit(cache=True)
def cumprod_1d_nb(a):
    """Calculate cumulative product of `a`.

    Numba equivalent to `pd.Series(a).cumprod()`."""
    result = np.empty_like(a, dtype=f8)
    cumprod = 1
    for i in range(a.shape[0]):
        if ~np.isnan(a[i]):
            cumprod *= a[i]
            result[i] = cumprod
        else:
            result[i] = np.nan
    return result


@njit(cache=True)
def cumprod_nb(a):
    """2D version of `cumprod_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = cumprod_1d_nb(a[:, col])
    return result


@njit(cache=True)
def nanmax_cube_nb(a):
    """Calculate `nanmax` on a cube `a` by reducing the axis 0."""
    result = np.empty((a.shape[1], a.shape[2]), dtype=f8)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            result[i, j] = np.nanmax(a[:, i, j])
    return result

# ############# Rolling functions ############# #


@njit(cache=True)
def rolling_window_1d_nb(a, window):
    """Roll a window over `a` of size `window`.

    Creates a matrix of rolled windows with first axis being window size."""
    width = a.shape[0] - window + 1
    result = np.empty((window, width), dtype=f8)
    for col in range(result.shape[1]):
        result[:, col] = a[col:col+window]
    return result


@njit(cache=True)
def rolling_window_nb(a, window):
    """2D version of `rolling_window_1d_nb`.

    Creates a cube of rolled windows with first axis being columns and second axis being window size."""
    result = np.empty((a.shape[1], window, a.shape[0] - window + 1), dtype=f8)
    for col in range(a.shape[1]):
        result[col, :, :] = rolling_window_1d_nb(a[:, col], window)
    return result


@njit(cache=True)
def rolling_min_1d_nb(a, window, minp=None):
    """Calculate rolling min over `a`.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).min()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    result = np.empty_like(a, dtype=f8)
    for i in range(a.shape[0]):
        minv = a[i]
        cnt = 0
        for j in range(max(i-window+1, 0), i+1):
            if np.isnan(a[j]):
                continue
            if np.isnan(minv) or a[j] < minv:
                minv = a[j]
            cnt += 1
        if cnt < minp:
            result[i] = np.nan
        else:
            result[i] = minv
    return result


@njit(cache=True)
def rolling_min_nb(a, window, minp=None):
    """2D version of `rolling_min_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = rolling_min_1d_nb(a[:, col], window, minp=minp)
    return result


@njit(cache=True)
def rolling_max_1d_nb(a, window, minp=None):
    """Calculate rolling max over `a`.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).max()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    result = np.empty_like(a, dtype=f8)
    for i in range(a.shape[0]):
        maxv = a[i]
        cnt = 0
        for j in range(max(i-window+1, 0), i+1):
            if np.isnan(a[j]):
                continue
            if np.isnan(maxv) or a[j] > maxv:
                maxv = a[j]
            cnt += 1
        if cnt < minp:
            result[i] = np.nan
        else:
            result[i] = maxv
    return result


@njit(cache=True)
def rolling_max_nb(a, window, minp=None):
    """2D version of `rolling_max_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = rolling_max_1d_nb(a[:, col], window, minp=minp)
    return result


@njit(cache=True)
def rolling_mean_1d_nb(a, window, minp=None):
    """Calculate rolling mean over `a`.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).mean()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    result = np.empty_like(a, dtype=f8)
    cumsum_arr = np.zeros_like(a)
    cumsum = 0
    nancnt_arr = np.zeros_like(a)
    nancnt = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            nancnt = nancnt + 1
        else:
            cumsum = cumsum + a[i]
        nancnt_arr[i] = nancnt
        cumsum_arr[i] = cumsum
        if i < window:
            window_len = i + 1 - nancnt
            window_cumsum = cumsum
        else:
            window_len = window - (nancnt - nancnt_arr[i-window])
            window_cumsum = cumsum - cumsum_arr[i-window]
        if window_len < minp:
            result[i] = np.nan
        else:
            result[i] = window_cumsum / window_len
    return result


@njit(cache=True)
def rolling_mean_nb(a, window, minp=None):
    """2D version of `rolling_mean_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = rolling_mean_1d_nb(a[:, col], window, minp=minp)
    return result


@njit(cache=True)
def rolling_std_1d_nb(a, window, minp=None):
    """Calculate rolling standard deviation over `a`.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).std()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    if minp == 1:
        minp = 2
    result = np.empty_like(a, dtype=f8)
    cumsum_arr = np.zeros_like(a)
    cumsum = 0
    cumsum_sq_arr = np.zeros_like(a)
    cumsum_sq = 0
    nancnt_arr = np.zeros_like(a)
    nancnt = 0
    mean = 0
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            nancnt = nancnt + 1
        else:
            cumsum = cumsum + a[i]
            cumsum_sq = cumsum_sq + a[i] ** 2
        nancnt_arr[i] = nancnt
        cumsum_arr[i] = cumsum
        cumsum_sq_arr[i] = cumsum_sq
        if i < window:
            window_len = i + 1 - nancnt
            window_cumsum = cumsum
            window_cumsum_sq = cumsum_sq
        else:
            window_len = window - (nancnt - nancnt_arr[i-window])
            window_cumsum = cumsum - cumsum_arr[i-window]
            window_cumsum_sq = cumsum_sq - cumsum_sq_arr[i-window]
        if window_len < minp:
            result[i] = np.nan
        else:
            mean = window_cumsum / window_len
            result[i] = np.sqrt(np.abs(window_cumsum_sq - 2 * window_cumsum *
                                       mean + window_len * mean ** 2) / (window_len - 1))
    return result


@njit(cache=True)
def rolling_std_nb(a, window, minp=None):
    """2D version of `rolling_std_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = rolling_std_1d_nb(a[:, col], window, minp=minp)
    return result


@njit(cache=True)
def ewm_mean_1d_nb(a, span, minp=None):
    """Calculate exponential weighted average over `a`.

    Numba equivalent to `pd.Series(a).ewm(span=span, min_periods=minp).mean()`.

    Adaptation of `pandas._libs.window.aggregations.window_aggregations.ewma` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    N = len(a)
    result = np.empty(N, dtype=f8)
    if N == 0:
        return result
    com = (span - 1) / 2.0
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1.
    weighted_avg = a[0]
    is_observation = (weighted_avg == weighted_avg)
    nobs = int(is_observation)
    result[0] = weighted_avg if (nobs >= minp) else np.nan
    old_wt = 1.

    for i in range(1, N):
        cur = a[i]
        is_observation = (cur == cur)
        nobs += is_observation
        if weighted_avg == weighted_avg:
            old_wt *= old_wt_factor
            if is_observation:
                # avoid numerical errors on constant series
                if weighted_avg != cur:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (old_wt + new_wt)
                old_wt += new_wt
        elif is_observation:
            weighted_avg = cur
        result[i] = weighted_avg if (nobs >= minp) else np.nan
    return result


@njit(cache=True)
def ewm_mean_nb(a, span, minp=None):
    """2D version of `ewm_mean_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = ewm_mean_1d_nb(a[:, col], span, minp=minp)
    return result


@njit(cache=True)
def ewm_std_1d_nb(a, span, minp=None):
    """Calculate exponential weighted standard deviation over `a`.

    Numba equivalent to `pd.Series(a).ewm(span=span, min_periods=minp).std()`.

    Adaptation of `pandas._libs.window.aggregations.window_aggregations.ewmcov` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    N = len(a)
    result = np.empty(N, dtype=f8)
    if N == 0:
        return result
    com = (span - 1) / 2.0
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1.
    mean_x = a[0]
    mean_y = a[0]
    is_observation = ((mean_x == mean_x) and (mean_y == mean_y))
    nobs = int(is_observation)
    if not is_observation:
        mean_x = np.nan
        mean_y = np.nan
    result[0] = np.nan
    cov = 0.
    sum_wt = 1.
    sum_wt2 = 1.
    old_wt = 1.

    for i in range(1, N):
        cur_x = a[i]
        cur_y = a[i]
        is_observation = ((cur_x == cur_x) and (cur_y == cur_y))
        nobs += is_observation
        if mean_x == mean_x:
            sum_wt *= old_wt_factor
            sum_wt2 *= (old_wt_factor * old_wt_factor)
            old_wt *= old_wt_factor
            if is_observation:
                old_mean_x = mean_x
                old_mean_y = mean_y

                # avoid numerical errors on constant series
                if mean_x != cur_x:
                    mean_x = ((old_wt * old_mean_x) +
                              (new_wt * cur_x)) / (old_wt + new_wt)

                # avoid numerical errors on constant series
                if mean_y != cur_y:
                    mean_y = ((old_wt * old_mean_y) +
                              (new_wt * cur_y)) / (old_wt + new_wt)
                cov = ((old_wt * (cov + ((old_mean_x - mean_x) *
                                         (old_mean_y - mean_y)))) +
                       (new_wt * ((cur_x - mean_x) *
                                  (cur_y - mean_y)))) / (old_wt + new_wt)
                sum_wt += new_wt
                sum_wt2 += (new_wt * new_wt)
                old_wt += new_wt
        elif is_observation:
            mean_x = cur_x
            mean_y = cur_y

        if nobs >= minp:
            numerator = sum_wt * sum_wt
            denominator = numerator - sum_wt2
            if (denominator > 0.):
                result[i] = ((numerator / denominator) * cov)
            else:
                result[i] = np.nan
        else:
            result[i] = np.nan
    return np.sqrt(result)


@njit(cache=True)
def ewm_std_nb(a, span, minp=None):
    """2D version of `ewm_std_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = ewm_std_1d_nb(a[:, col], span, minp=minp)
    return result

# ############# Expanding functions ############# #


@njit(cache=True)
def expanding_min_1d_nb(a, minp=1):
    """Calculate expanding min over `a`.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).min()`."""
    result = np.empty_like(a, dtype=f8)
    minv = a[0]
    cnt = 0
    for i in range(a.shape[0]):
        if np.isnan(minv) or a[i] < minv:
            minv = a[i]
        if ~np.isnan(a[i]):
            cnt += 1
        if cnt < minp:
            result[i] = np.nan
        else:
            result[i] = minv
    return result


@njit(cache=True)
def expanding_min_nb(a, minp=1):
    """2D version of `expanding_min_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = expanding_min_1d_nb(a[:, col], minp=minp)
    return result


@njit(cache=True)
def expanding_max_1d_nb(a, minp=1):
    """Calculate expanding max over `a`.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).max()`."""
    result = np.empty_like(a, dtype=f8)
    maxv = a[0]
    cnt = 0
    for i in range(a.shape[0]):
        if np.isnan(maxv) or a[i] > maxv:
            maxv = a[i]
        if ~np.isnan(a[i]):
            cnt += 1
        if cnt < minp:
            result[i] = np.nan
        else:
            result[i] = maxv
    return result


@njit(cache=True)
def expanding_max_nb(a, minp=1):
    """2D version of `expanding_max_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = expanding_max_1d_nb(a[:, col], minp=minp)
    return result


@njit(cache=True)
def expanding_mean_1d_nb(a, minp=1):
    """Calculate expanding mean over `a`.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).mean()`."""
    return rolling_mean_1d_nb(a, a.shape[0], minp=minp)


@njit(cache=True)
def expanding_mean_nb(a, minp=1):
    """2D version of `expanding_mean_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = expanding_mean_1d_nb(a[:, col], minp=minp)
    return result


@njit(cache=True)
def expanding_std_1d_nb(a, minp=1):
    """Calculate expanding standard deviation over `a`.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).std()`."""
    return rolling_std_1d_nb(a, a.shape[0], minp=minp)


@njit(cache=True)
def expanding_std_nb(a, minp=1):
    """2D version of `expanding_std_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = expanding_std_1d_nb(a[:, col], minp=minp)
    return result

# ############# Apply functions (no caching) ############# #


@njit
def rolling_apply_1d_nb(a, window, apply_func_nb):
    """Provide rolling window calculations over `a` using `apply_func_nb`.

    Numba equivalent to `pd.Series(a).rolling(window).apply(apply_func_nb, raw=True)`."""
    result = np.empty_like(a, dtype=f8)
    for i in range(a.shape[0]):
        window_a = a[max(0, i+1-window):i+1]
        result[i] = apply_func_nb(window_a)
    return result


@njit
def rolling_apply_nb(a, window, apply_func_nb, on_2d=False):
    """2D version of `rolling_apply_1d_nb`.

    If `on_2d` is `True`, will roll over all columns as matrix, otherwise over each column individually."""
    result = np.empty_like(a, dtype=f8)
    if on_2d:
        for i in range(a.shape[0]):
            window_a = a[max(0, i+1-window):i+1, :]
            result[i, :] = apply_func_nb(window_a)
    else:
        for col in range(a.shape[1]):
            result[:, col] = rolling_apply_1d_nb(a[:, col], window, apply_func_nb)
    return result


@njit
def expanding_apply_1d_nb(a, apply_func_nb):
    """Provide expanding window calculations over `a` using `apply_func_nb`.

    Numba equivalent to `pd.Series(a).expanding().apply(apply_func_nb, raw=True)`."""
    result = np.empty_like(a, dtype=f8)
    for i in range(a.shape[0]):
        result[i] = apply_func_nb(a[0:i+1])
    return result


@njit
def expanding_apply_nb(a, apply_func_nb, on_2d=False):
    """2D version of `expanding_apply_1d_nb`.

    If `on_2d` is `True`, will expand over all columns as matrix, otherwise over each column individually."""
    result = np.empty_like(a, dtype=f8)
    if on_2d:
        for i in range(a.shape[0]):
            result[i, :] = apply_func_nb(a[0:i+1, :])
    else:
        for col in range(a.shape[1]):
            result[:, col] = expanding_apply_1d_nb(a[:, col], apply_func_nb)
    return result


@njit
def groupby_apply_1d_nb(a, b, apply_func_nb):
    """Apply function `apply_func_nb` to `a` by each unique value in `b`.

    Numba equivalent to `pd.Series(a).groupby(b).apply(apply_func_nb, raw=True)`.

    `b` must have the same shape as `a`. 
    Returns the groups and the result."""
    groups = np.unique(b)
    group_idxs = []
    for i in range(len(groups)):
        idx_lst = []
        for j in range(len(b)):
            if b[j] == groups[i]:
                idx_lst.append(j)
        group_idxs.append(np.asarray(idx_lst))
    result = np.empty(len(groups), dtype=f8)
    for i, idxs in enumerate(group_idxs):
        result[i] = apply_func_nb(a[idxs])
    return groups, result


@njit
def groupby_apply_nb(a, b, apply_func_nb, on_2d=False):
    """2D version of `groupby_apply_1d_nb`."""
    groups = np.unique(b)
    group_idxs = []
    for i in range(len(groups)):
        idx_lst = []
        for j in range(len(b)):
            if b[j] == groups[i]:
                idx_lst.append(j)
        group_idxs.append(np.asarray(idx_lst))
    result = np.empty((len(groups), a.shape[1]), dtype=f8)
    for i, idxs in enumerate(group_idxs):
        if on_2d:
            result[i, :] = apply_func_nb(a[idxs, :])
        else:
            for col in range(a.shape[1]):
                result[i, col] = apply_func_nb(a[idxs, col])
    return groups, result


@njit
def apply_by_mask_1d_nb(a, mask, apply_func_nb):
    """Apply function `apply_func_nb` to `a` by boolean `mask`. Used for resample-and-apply.

    `mask` must be a 2D boolean array of shape `(any, a.shape[0])`. For each row in `mask`,
    select the masked elements from `a` and perform calculation on them. The result will
    have the shape `(mask.shape[0],)`."""
    result = np.empty(mask.shape[0], dtype=f8)
    for i in range(mask.shape[0]):
        result[i] = apply_func_nb(a[mask[i, :]])
    return result


@njit
def apply_by_mask_nb(a, mask, apply_func_nb, on_2d=False):
    """2D version of `apply_by_mask_1d_nb`.

    If `on_2d` is `True`, will apply to all columns as matrix, otherwise to each column individually."""
    result = np.empty((mask.shape[0], a.shape[1]), dtype=f8)
    if on_2d:
        for i in range(mask.shape[0]):
            result[i, :] = apply_func_nb(a[mask[i, :], :])
    else:
        for col in range(a.shape[1]):
            result[:, col] = apply_by_mask_1d_nb(a[:, col], mask, apply_func_nb)
    return result
