"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    Input arrays can be of any data type, but most output arrays are `numpy.float64`.
    
    Rolling functions with `minp=None` have `min_periods` set to the window size.
    
    All functions passed as argument must be Numba-compiled."""

from numba import njit, f8
import numpy as np


@njit(cache=True)
def prepend_1d_nb(a, n, value):
    """Prepend value `n` times."""
    result = np.empty(a.shape[0] + n, dtype=f8)
    result[:n] = value
    result[n:] = a
    return result


@njit(cache=True)
def prepend_nb(a, n, value):
    """2-dim version of `prepend_1d_nb`."""
    result = np.empty((a.shape[0] + n, a.shape[1]), dtype=f8)
    result[:n, :] = value
    result[n:, :] = a
    return result


@njit(cache=True)
def set_by_mask_1d_nb(a, mask, value):
    """Set each element to a value by boolean mask."""
    result = a.astype(f8)
    result[mask] = value
    return result


@njit(cache=True)
def set_by_mask_nb(a, mask, value):
    """2-dim version of `set_by_mask_1d_nb`."""
    result = a.astype(f8)
    for col in range(a.shape[1]):
        result[mask[:, col], col] = value
    return result


@njit(cache=True)
def set_by_mask_mult_1d_nb(a, mask, values):
    """Set each element in one array to the corresponding element in another by boolean mask.

    `values` must be of the same shape as in `a`."""
    result = a.astype(f8)
    result[mask] = values[mask]
    return result


@njit(cache=True)
def set_by_mask_mult_nb(a, mask, values):
    """2-dim version of `set_by_mask_mult_1d_nb`."""
    result = a.astype(f8)
    for col in range(a.shape[1]):
        result[mask[:, col], col] = values[mask[:, col], col]
    return result


@njit(cache=True)
def fillna_1d_nb(a, value):
    """Replace NaNs with value.

    Numba equivalent to `pd.Series(a).fillna(value)`."""
    return set_by_mask_1d_nb(a, np.isnan(a), value)


@njit(cache=True)
def fillna_nb(a, value):
    """2-dim version of `fillna_1d_nb`."""
    return set_by_mask_nb(a, np.isnan(a), value)


@njit(cache=True)
def fshift_1d_nb(a, n):
    """Shift forward by `n` positions.

    Numba equivalent to `pd.Series(a).shift(value)`."""
    result = np.empty_like(a, dtype=f8)
    result[:n] = np.nan
    result[n:] = a[:-n]
    return result


@njit(cache=True)
def fshift_nb(a, n):
    """2-dim version of `fshift_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    result[:n, :] = np.nan
    result[n:, :] = a[:-n, :]
    return result


@njit(cache=True)
def diff_1d_nb(a):
    """Return the 1-th discrete difference.

    Numba equivalent to `pd.Series(a).diff()`."""
    result = np.empty_like(a, dtype=f8)
    result[0] = np.nan
    result[1:] = a[1:] - a[:-1]
    return result


@njit(cache=True)
def diff_nb(a):
    """2-dim version of `diff_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    result[0, :] = np.nan
    result[1:, :] = a[1:, :] - a[:-1, :]
    return result


@njit(cache=True)
def pct_change_1d_nb(a):
    """Return the percentage change.

    Numba equivalent to `pd.Series(a).pct_change()`."""
    result = np.empty_like(a, dtype=f8)
    result[0] = np.nan
    result[1:] = a[1:] / a[:-1] - 1
    return result


@njit(cache=True)
def pct_change_nb(a):
    """2-dim version of `pct_change_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    result[0, :] = np.nan
    result[1:, :] = a[1:, :] / a[:-1, :] - 1
    return result


@njit(cache=True)
def ffill_1d_nb(a):
    """Fill NaNs by propagating last valid observation forward.

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
    """2-dim version of `ffill_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = ffill_1d_nb(a[:, col])
    return result


@njit(cache=True)
def product_1d_nb(a):
    """Return product.

    Numba equivalent to `pd.Series(a).prod()`."""
    result = a[0]
    for i in range(1, a.shape[0]):
        if ~np.isnan(a[i]):
            if np.isnan(result):
                result = a[i]
            else:
                result *= a[i]
    return result


@njit(cache=True)
def product_nb(a):
    """2-dim version of `product_1d_nb`."""
    result = np.empty(a.shape[1], dtype=f8)
    for col in range(a.shape[1]):
        result[col] = product_1d_nb(a[:, col])
    return result


@njit(cache=True)
def cumsum_1d_nb(a):
    """Return cumulative sum.

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
    """2-dim version of `cumsum_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = cumsum_1d_nb(a[:, col])
    return result


@njit(cache=True)
def cumprod_1d_nb(a):
    """Return cumulative product.

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
    """2-dim version of `cumprod_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = cumprod_1d_nb(a[:, col])
    return result


@njit(cache=True)
def nancnt_nb(a):
    """Compute count while ignoring NaNs."""
    result = np.empty(a.shape[1], dtype=f8)
    for col in range(a.shape[1]):
        result[col] = np.sum(~np.isnan(a[:, col]))
    return result


@njit(cache=True)
def nansum_nb(a):
    """Compute sum while ignoring NaNs."""
    result = np.empty(a.shape[1], dtype=f8)
    for col in range(a.shape[1]):
        result[col] = np.nansum(a[:, col])
    return result


@njit(cache=True)
def nanmin_nb(a):
    """Compute minimum while ignoring NaNs."""
    result = np.empty(a.shape[1], dtype=f8)
    for col in range(a.shape[1]):
        result[col] = np.nanmin(a[:, col])
    return result


@njit(cache=True)
def nanmax_nb(a):
    """Compute maximum while ignoring NaNs."""
    result = np.empty(a.shape[1], dtype=f8)
    for col in range(a.shape[1]):
        result[col] = np.nanmax(a[:, col])
    return result


@njit(cache=True)
def nanmean_nb(a):
    """Compute mean while ignoring NaNs."""
    result = np.empty(a.shape[1], dtype=f8)
    for col in range(a.shape[1]):
        result[col] = np.nanmean(a[:, col])
    return result


@njit(cache=True)
def nanstd_1d_nb(a, ddof=1):
    """Compute the standard deviation while ignoring NaNs."""
    cnt = a.shape[0] - np.count_nonzero(np.isnan(a))
    rcount = max(cnt - ddof, 0)
    if rcount == 0:
        return np.nan
    else:
        return np.sqrt(np.nanvar(a) * cnt / rcount)


@njit(cache=True)
def nanstd_nb(a, ddof=1):
    """2-dim version of `nanstd_1d_nb`."""
    result = np.empty(a.shape[1], dtype=f8)
    for col in range(a.shape[1]):
        result[col] = nanstd_1d_nb(a[:, col], ddof=ddof)
    return result


# ############# Rolling functions ############# #


@njit(cache=True)
def rolling_window_1d_nb(a, window):
    """Roll a window.

    Creates a matrix of rolled windows with first axis being window size."""
    width = a.shape[0] - window + 1
    result = np.empty((window, width), dtype=f8)
    for col in range(width):
        result[:, col] = a[col:col + window]
    return result


@njit(cache=True)
def rolling_window_nb(a, window):
    """2-dim version of `rolling_window_1d_nb`.

    Creates a cube of rolled windows with first axis being columns and second axis being window size."""
    result = np.empty((a.shape[1], window, a.shape[0] - window + 1), dtype=f8)
    for col in range(a.shape[1]):
        result[col, :, :] = rolling_window_1d_nb(a[:, col], window)
    return result


@njit(cache=True)
def rolling_min_1d_nb(a, window, minp=None):
    """Return rolling min.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).min()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise Exception("minp must be <= window")
    result = np.empty_like(a, dtype=f8)
    for i in range(a.shape[0]):
        minv = a[i]
        cnt = 0
        for j in range(max(i - window + 1, 0), i + 1):
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
    """2-dim version of `rolling_min_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = rolling_min_1d_nb(a[:, col], window, minp=minp)
    return result


@njit(cache=True)
def rolling_max_1d_nb(a, window, minp=None):
    """Return rolling max.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).max()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise Exception("minp must be <= window")
    result = np.empty_like(a, dtype=f8)
    for i in range(a.shape[0]):
        maxv = a[i]
        cnt = 0
        for j in range(max(i - window + 1, 0), i + 1):
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
    """2-dim version of `rolling_max_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = rolling_max_1d_nb(a[:, col], window, minp=minp)
    return result


@njit(cache=True)
def rolling_mean_1d_nb(a, window, minp=None):
    """Return rolling mean.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).mean()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise Exception("minp must be <= window")
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
            window_len = window - (nancnt - nancnt_arr[i - window])
            window_cumsum = cumsum - cumsum_arr[i - window]
        if window_len < minp:
            result[i] = np.nan
        else:
            result[i] = window_cumsum / window_len
    return result


@njit(cache=True)
def rolling_mean_nb(a, window, minp=None):
    """2-dim version of `rolling_mean_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = rolling_mean_1d_nb(a[:, col], window, minp=minp)
    return result


@njit(cache=True)
def rolling_std_1d_nb(a, window, minp=None):
    """Return rolling standard deviation.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).std()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise Exception("minp must be <= window")
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
            window_len = window - (nancnt - nancnt_arr[i - window])
            window_cumsum = cumsum - cumsum_arr[i - window]
            window_cumsum_sq = cumsum_sq - cumsum_sq_arr[i - window]
        if window_len < minp:
            result[i] = np.nan
        else:
            mean = window_cumsum / window_len
            result[i] = np.sqrt(np.abs(window_cumsum_sq - 2 * window_cumsum *
                                       mean + window_len * mean ** 2) / (window_len - 1))
    return result


@njit(cache=True)
def rolling_std_nb(a, window, minp=None):
    """2-dim version of `rolling_std_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = rolling_std_1d_nb(a[:, col], window, minp=minp)
    return result


@njit(cache=True)
def ewm_mean_1d_nb(a, span, minp=None):
    """Return exponential weighted average.

    Numba equivalent to `pd.Series(a).ewm(span=span, min_periods=minp).mean()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewma` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise Exception("minp must be <= span")
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
    """2-dim version of `ewm_mean_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = ewm_mean_1d_nb(a[:, col], span, minp=minp)
    return result


@njit(cache=True)
def ewm_std_1d_nb(a, span, minp=None):
    """Return exponential weighted standard deviation.

    Numba equivalent to `pd.Series(a).ewm(span=span, min_periods=minp).std()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewmcov` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise Exception("minp must be <= span")
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
            if denominator > 0.:
                result[i] = ((numerator / denominator) * cov)
            else:
                result[i] = np.nan
        else:
            result[i] = np.nan
    return np.sqrt(result)


@njit(cache=True)
def ewm_std_nb(a, span, minp=None):
    """2-dim version of `ewm_std_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = ewm_std_1d_nb(a[:, col], span, minp=minp)
    return result


# ############# Expanding functions ############# #


@njit(cache=True)
def expanding_min_1d_nb(a, minp=1):
    """Return expanding min.

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
    """2-dim version of `expanding_min_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = expanding_min_1d_nb(a[:, col], minp=minp)
    return result


@njit(cache=True)
def expanding_max_1d_nb(a, minp=1):
    """Return expanding max.

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
    """2-dim version of `expanding_max_1d_nb`."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        result[:, col] = expanding_max_1d_nb(a[:, col], minp=minp)
    return result


@njit(cache=True)
def expanding_mean_1d_nb(a, minp=1):
    """Return expanding mean.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).mean()`."""
    return rolling_mean_1d_nb(a, a.shape[0], minp=minp)


@njit(cache=True)
def expanding_mean_nb(a, minp=1):
    """2-dim version of `expanding_mean_1d_nb`."""
    return rolling_mean_nb(a, a.shape[0], minp=minp)


@njit(cache=True)
def expanding_std_1d_nb(a, minp=1):
    """Return expanding standard deviation.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).std()`."""
    return rolling_std_1d_nb(a, a.shape[0], minp=minp)


@njit(cache=True)
def expanding_std_nb(a, minp=1):
    """2-dim version of `expanding_std_1d_nb`."""
    return rolling_std_nb(a, a.shape[0], minp=minp)


# ############# Apply functions ############# #


@njit
def rolling_apply_nb(a, window, apply_func_nb, *args):
    """Provide rolling window calculations.

    `apply_func_nb` must accept index of the current column, index of the current row, 
    the array, and `*args`. Must return a single value."""
    result = np.empty_like(a, dtype=f8)
    for col in range(a.shape[1]):
        for i in range(a.shape[0]):
            window_a = a[max(0, i + 1 - window):i + 1, col]
            result[i, col] = apply_func_nb(col, i, window_a, *args)
    return result


@njit
def rolling_apply_matrix_nb(a, window, apply_func_nb, *args):
    """`rolling_apply_nb` with `apply_func_nb` being applied on all columns at once.

    `apply_func_nb` must accept index of the current row, the 2-dim array, and `*args`. 
    Must return a single value or an array of shape `a.shape[1]`."""
    result = np.empty_like(a, dtype=f8)
    for i in range(a.shape[0]):
        window_a = a[max(0, i + 1 - window):i + 1, :]
        result[i, :] = apply_func_nb(i, window_a, *args)
    return result


@njit
def expanding_apply_nb(a, apply_func_nb, *args):
    """Expanding version of `rolling_apply_nb`."""
    return rolling_apply_nb(a, a.shape[0], apply_func_nb, *args)


@njit
def expanding_apply_matrix_nb(a, apply_func_nb, *args):
    """Expanding version of `rolling_apply_matrix_nb`."""
    return rolling_apply_matrix_nb(a, a.shape[0], apply_func_nb, *args)


@njit
def groupby_apply_nb(a, groups, apply_func_nb, *args):
    """Provide group-by calculations.

    `groups` must be a dictionary, where each key is an index that points to an element in the new array 
    where a group-by result will be stored, while the value should be an array of indices in `a`
    to apply `apply_func_nb` on.

    `apply_func_nb` must accept index of the current column, indices of the current group, 
    the array, and `*args`. Must return a single value."""
    result = np.empty((len(groups), a.shape[1]), dtype=f8)
    for col in range(a.shape[1]):
        for i, idxs in groups.items():
            result[i, col] = apply_func_nb(col, idxs, a[idxs, col], *args)
    return result


@njit
def groupby_apply_matrix_nb(a, groups, apply_func_nb, *args):
    """`groupby_apply_nb` with `apply_func_nb` being applied on all columns at once.

    `apply_func_nb` must accept indices of the current group, the 2-dim array, and `*args`. 
    Must return a single value or an array of shape `a.shape[1]`."""
    result = np.empty((len(groups), a.shape[1]), dtype=f8)
    for i, idxs in groups.items():
        result[i, :] = apply_func_nb(idxs, a[idxs, :], *args)
    return result


# ############# Map, filter and reduce ############# #


@njit
def applymap_nb(a, map_func_nb, *args):
    """Map non-NA elements elementwise using `map_func_nb`.

    `map_func_nb` must accept index of the current column, index of the current element, 
    the element itself, and `*args`. Must return an array of same size."""
    result = np.full_like(a, np.nan, dtype=f8)

    for col in range(result.shape[1]):
        idxs = np.flatnonzero(~np.isnan(a[:, col]))
        for i in idxs:
            result[i, col] = map_func_nb(col, i, a[i, col], *args)
    return result


@njit
def filter_nb(a, filter_func_nb, *args):
    """Filter non-NA elements elementwise using `filter_func_nb`. 
    The filtered out elements will become NA.

    `filter_func_nb` must accept index of the current column, index of the current element, 
    the element itself, and `*args`. Must return a boolean value."""
    result = a.astype(f8)

    for col in range(result.shape[1]):
        idxs = np.flatnonzero(~np.isnan(a[:, col]))
        for i in idxs:
            if ~filter_func_nb(col, i, a[i, col], *args):
                result[i, col] = np.nan
    return result


@njit
def apply_and_reduce_nb(a, apply_func_nb, reduce_func_nb, *args):
    """Apply `apply_func_nb` on each column and reduce into a single value using `reduce_func_nb`.

    `apply_func_nb` must accept index of the current column, the column itself, and `*args`. 
    Must return an array.

    `reduce_func_nb` must accept index of the current column, the array of results from 
    `apply_func_nb` for that column, and `*args`. Must return a single value."""
    result = np.full(a.shape[1], np.nan, dtype=f8)

    for col in range(a.shape[1]):
        mapped = apply_func_nb(col, a[:, col], *args)
        result[col] = reduce_func_nb(col, mapped, *args)
    return result


@njit
def reduce_nb(a, reduce_func_nb, *args):
    """Reduce each column into a single value using `reduce_func_nb`.

    `reduce_func_nb` must accept index of the current column, the array, and `*args`. 
    Must return a single value."""
    result = np.full(a.shape[1], np.nan, dtype=f8)

    for col in range(a.shape[1]):
        result[col] = reduce_func_nb(col, a[:, col], *args)
    return result


@njit
def reduce_to_array_nb(a, reduce_func_nb, *args):
    """Reduce each column into an array of values using `reduce_func_nb`.

    `reduce_func_nb` same as for `reduce_nb` but must return an array.

    !!! note
        Output of `reduce_func_nb` must be strictly homogeneous."""
    result_inited = False
    for col in range(a.shape[1]):
        col_result = reduce_func_nb(col, a[:, col], *args)
        if not result_inited:
            result = np.full((col_result.shape[0], a.shape[1]), np.nan, dtype=f8)
            result_inited = True
        result[:, col] = col_result

    return result


@njit(cache=True)
def nst_reduce_nb(col, a, n, *args):
    if n >= a.shape[0]:
        raise ValueError("index is out of bounds")
    return a[n]


min_reduce_nb = njit(cache=True)(lambda col, a, *args: np.nanmin(a))
max_reduce_nb = njit(cache=True)(lambda col, a, *args: np.nanmax(a))
mean_reduce_nb = njit(cache=True)(lambda col, a, *args: np.nanmean(a))
median_reduce_nb = njit(cache=True)(lambda col, a, *args: np.nanmedian(a))
sum_reduce_nb = njit(cache=True)(lambda col, a, *args: np.nansum(a))
count_reduce_nb = njit(cache=True)(lambda col, a, *args: np.sum(~np.isnan(a)))


@njit(cache=True)
def std_reduce_nb(col, a, ddof, *args):
    return nanstd_1d_nb(a, ddof=ddof)


@njit(cache=True)
def describe_reduce_nb(col, a, perc, ddof, *args):
    """Return descriptive statistics.

    Numba equivalent to `pd.Series(a).describe(perc)`."""
    a = a[~np.isnan(a)]
    result = np.empty(5 + len(perc), dtype=f8)
    result[0] = len(a)
    if len(a) > 0:
        result[1] = np.mean(a)
        result[2] = nanstd_1d_nb(a, ddof=ddof)
        result[3] = np.min(a)
        result[4:-1] = np.percentile(a, perc * 100)
        result[4 + len(perc)] = np.max(a)
    else:
        result[1:] = np.nan
    return result


@njit(cache=True)
def argmin_reduce_nb(col, a, *args):
    a = np.copy(a)
    mask = np.isnan(a)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    a[mask] = np.inf
    return np.argmin(a)


@njit(cache=True)
def argmax_reduce_nb(col, a, *args):
    a = np.copy(a)
    mask = np.isnan(a)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    a[mask] = -np.inf
    return np.argmax(a)
