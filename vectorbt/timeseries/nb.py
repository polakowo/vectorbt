from numba import njit, f8, i8, b1
import numpy as np


@njit(f8[:](f8[:], i8, f8), cache=True)
def prepend_1d_nb(a, n, value):
    """Prepend n values to the array."""
    b = np.full(a.shape[0]+n, value)
    b[n:] = a
    return b


@njit(f8[:, :](f8[:, :], i8, f8), cache=True)
def prepend_nb(a, n, value):
    b = np.full((a.shape[0]+n, a.shape[1]), value)
    b[n:, :] = a
    return b


@njit(f8[:, :](f8[:], i8), cache=True)
def rolling_window_1d_nb(a, window):
    """Stack and return all windows rolled over the array."""
    width = a.shape[0] - window + 1
    b = np.empty((window, width))
    for col in range(b.shape[1]):
        b[:, col] = a[col:col+window]
    return b


@njit(f8[:, :, :](f8[:, :], i8), cache=True)
def rolling_window_nb(a, window):
    b = np.empty((a.shape[1], window, a.shape[0] - window + 1))
    for col in range(a.shape[1]):
        b[col, :, :] = rolling_window_1d_nb(a[:, col], window)
    return b

# Functions below have shape in = shape out


@njit(f8[:](f8[:], b1[:], f8), cache=True)
def set_by_mask_1d_nb(a, mask, value):
    """Set value by boolean mask."""
    b = a.copy()
    b[mask] = value
    return b


@njit(f8[:, :](f8[:, :], b1[:, :], f8), cache=True)
def set_by_mask_nb(a, mask, value):
    b = a.copy()
    for col in range(b.shape[1]):
        b[mask[:, col], col] = value
    return b


@njit(f8[:](f8[:], b1[:], f8[:]), cache=True)
def set_by_mask_mult_1d_nb(a, mask, values):
    """Set values by boolean mask."""
    b = a.copy()
    b[mask] = values[mask]
    return b


@njit(f8[:, :](f8[:, :], b1[:, :], f8[:, :]), cache=True)
def set_by_mask_mult_nb(a, mask, values):
    b = a.copy()
    for col in range(b.shape[1]):
        b[mask[:, col], col] = values[mask[:, col], col]
    return b


@njit(f8[:](f8[:], f8), cache=True)
def fillna_1d_nb(a, value):
    """Fill NaNs with value."""
    return set_by_mask_1d_nb(a, np.isnan(a), value)


@njit(f8[:, :](f8[:, :], f8), cache=True)
def fillna_nb(a, value):
    return set_by_mask_nb(a, np.isnan(a), value)


@njit(f8[:](f8[:], i8), cache=True)
def fshift_1d_nb(a, n):
    """Shift forward by n."""
    b = np.full_like(a, np.nan)
    b[n:] = a[:-n]
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def fshift_nb(a, n):
    b = np.full_like(a, np.nan)
    b[n:, :] = a[:-n, :]
    return b


@njit(f8[:](f8[:]), cache=True)
def diff_1d_nb(a):
    """Calculate the 1-th discrete difference."""
    b = np.full_like(a, np.nan)
    b[1:] = a[1:] - a[:-1]
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def diff_nb(a):
    b = np.full_like(a, np.nan)
    b[1:, :] = a[1:, :] - a[:-1, :]
    return b


@njit(f8[:](f8[:]), cache=True)
def pct_change_1d_nb(a):
    """Compute the percentage change."""
    b = np.full_like(a, np.nan)
    b[1:] = a[1:] / a[:-1] - 1
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def pct_change_nb(a):
    b = np.full_like(a, np.nan)
    b[1:, :] = a[1:, :] / a[:-1, :] - 1
    return b


@njit(f8[:](f8[:]), cache=True)
def ffill_1d_nb(a):
    """Fill NaNs with the last value."""
    b = np.full_like(a, np.nan)
    maxval = a[0]
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            b[i] = maxval
        else:
            b[i] = a[i]
            maxval = b[i]
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def ffill_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = ffill_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:]), cache=True)
def cumsum_1d_nb(a):
    """Cumulative sum."""
    b = np.full_like(a, np.nan)
    cumsum = 0
    for i in range(a.shape[0]):
        if ~np.isnan(a[i]):
            cumsum += a[i]
            b[i] = cumsum
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def cumsum_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = cumsum_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:]), cache=True)
def cumprod_1d_nb(a):
    """Cumulative product."""
    b = np.full_like(a, np.nan)
    cumprod = 1
    for i in range(a.shape[0]):
        if ~np.isnan(a[i]):
            cumprod *= a[i]
            b[i] = cumprod
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def cumprod_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = cumprod_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:], i8), cache=True)
def rolling_mean_1d_nb(a, window):
    """Rolling mean."""
    b = np.full_like(a, np.nan)
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
        if i < window - 1:
            continue
        if i < window:
            if nancnt > 0:
                continue
            window_len = i + 1
            window_cumsum = cumsum
        else:
            if nancnt - nancnt_arr[i-window] > 0:
                continue
            window_len = window
            window_cumsum = cumsum - cumsum_arr[i-window]
        if window_len == 0:
            continue
        b[i] = window_cumsum / window_len
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def rolling_mean_nb(a, window):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = rolling_mean_1d_nb(a[:, col], window)
    return b


@njit(f8[:](f8[:], i8), cache=True)
def rolling_std_1d_nb(a, window):
    """Rolling std for ddof = 0."""
    b = np.full_like(a, np.nan)
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
        if i < window - 1:
            continue
        if i < window:
            if nancnt > 0:
                continue
            window_len = i + 1
            window_cumsum = cumsum
            window_cumsum_sq = cumsum_sq
        else:
            if nancnt - nancnt_arr[i-window] > 0:
                continue
            window_len = window - (nancnt - nancnt_arr[i-window])
            window_cumsum = cumsum - cumsum_arr[i-window]
            window_cumsum_sq = cumsum_sq - cumsum_sq_arr[i-window]
        if window_len == 0:
            continue
        mean = window_cumsum / window_len
        b[i] = np.sqrt(np.abs(window_cumsum_sq - 2 * window_cumsum * mean + window_len * mean ** 2) / (window_len - 1))
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def rolling_std_nb(a, window):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = rolling_std_1d_nb(a[:, col], window)
    return b


@njit(f8[:](f8[:], i8), cache=True)
def rolling_min_1d_nb(a, window):
    """Rolling min."""
    b = np.empty_like(a)
    for i in range(a.shape[0]):
        minv = a[i]
        for j in range(max(i-window+1, 0), i+1):
            if np.isnan(a[j]):
                minv = np.nan
                continue
            if a[j] < minv:
                minv = a[j]
        b[i] = minv
    b[:(window-1)] = np.nan
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def rolling_min_nb(a, window):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = rolling_min_1d_nb(a[:, col], window)
    return b


@njit(f8[:](f8[:], i8), cache=True)
def rolling_max_1d_nb(a, window):
    """Rolling max."""
    b = np.empty_like(a)
    for i in range(a.shape[0]):
        maxv = a[i]
        for j in range(max(i-window+1, 0), i+1):
            if np.isnan(a[j]):
                maxv = np.nan
                continue
            if a[j] > maxv:
                maxv = a[j]
        b[i] = maxv
    b[:(window-1)] = np.nan
    return b


@njit(f8[:, :](f8[:, :], i8), cache=True)
def rolling_max_nb(a, window):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = rolling_max_1d_nb(a[:, col], window)
    return b


@njit(f8[:](f8[:]), cache=True)
def expanding_max_1d_nb(a):
    """Expanding max."""
    b = np.empty_like(a)
    maxv = np.nan
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            if np.isnan(maxv):
                b[i] = np.nan
                continue
        else:
            if np.isnan(maxv) or a[i] > maxv:
                maxv = a[i]
        b[i] = maxv
    return b


@njit(f8[:, :](f8[:, :]), cache=True)
def expanding_max_nb(a):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = expanding_max_1d_nb(a[:, col])
    return b


@njit(f8[:](f8[:], i8), cache=True)
def ewm_mean_1d_nb(vals, span):
    """Adaptation of pandas._libs.window.aggregations.window_aggregations.ewma with default params."""
    N = len(vals)
    output = np.empty(N, dtype=f8)
    if N == 0:
        return output
    com = (span - 1) / 2.0
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1.
    weighted_avg = vals[0]
    is_observation = (weighted_avg == weighted_avg)
    output[0] = weighted_avg
    old_wt = 1.

    for i in range(1, N):
        cur = vals[i]
        is_observation = (cur == cur)
        if weighted_avg == weighted_avg:
            old_wt *= old_wt_factor
            if is_observation:
                # avoid numerical errors on constant series
                if weighted_avg != cur:
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (old_wt + new_wt)
                old_wt += new_wt
        elif is_observation:
            weighted_avg = cur
        output[i] = weighted_avg
    output[:(span-1)] = np.nan
    return output


@njit(f8[:, :](f8[:, :], i8), cache=True)
def ewm_mean_nb(a, span):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = ewm_mean_1d_nb(a[:, col], span)
    return b


@njit(f8[:](f8[:], i8), cache=True)
def ewm_std_1d_nb(vals, span):
    """Adaptation of pandas._libs.window.aggregations.window_aggregations.ewmcov with default params."""
    N = len(vals)
    output = np.empty(N, dtype=f8)
    if N == 0:
        return output
    com = (span - 1) / 2.0
    minp = 1.
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1.
    mean_x = vals[0]
    mean_y = vals[0]
    is_observation = ((mean_x == mean_x) and (mean_y == mean_y))
    nobs = int(is_observation)
    if not is_observation:
        mean_x = np.nan
        mean_y = np.nan
    output[0] = np.nan
    cov = 0.
    sum_wt = 1.
    sum_wt2 = 1.
    old_wt = 1.

    for i in range(1, N):
        cur_x = vals[i]
        cur_y = vals[i]
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
                output[i] = ((numerator / denominator) * cov)
            else:
                output[i] = np.nan
        else:
            output[i] = np.nan

    output[:(span-1)] = np.nan
    return np.sqrt(output)


@njit(f8[:, :](f8[:, :], i8), cache=True)
def ewm_std_nb(a, span):
    b = np.empty_like(a)
    for col in range(a.shape[1]):
        b[:, col] = ewm_std_1d_nb(a[:, col], span)
    return b


@njit
def rolling_apply_1d_nb(a, window, apply_func_nb):
    """Rolling apply."""
    b = np.empty_like(a)
    for i in range(a.shape[0]):
        b[i] = apply_func_nb(a[max(0, i+1-window):i+1])
    b[:(window-1)] = np.nan
    return b


@njit
def rolling_apply_nb(a, window, apply_func_nb, on_2d=False):
    """Use on_2d=True to pass rolling matrix to the apply_func_nb."""
    b = np.empty_like(a)
    if on_2d:
        for i in range(a.shape[0]):
            b[i, :] = apply_func_nb(a[max(0, i+1-window):i+1, :])
            b[:(window-1)] = np.nan
    else:
        for col in range(a.shape[1]):
            b[:, col] = rolling_apply_1d_nb(a[:, col], window, apply_func_nb)
    return b


@njit
def expanding_apply_1d_nb(a, apply_func_nb):
    """Expanding apply."""
    b = np.empty_like(a)
    for i in range(a.shape[0]):
        b[i] = apply_func_nb(a[0:i+1])
    return b


@njit
def expanding_apply_nb(a, apply_func_nb, on_2d=False):
    b = np.empty_like(a)
    if on_2d:
        for i in range(a.shape[0]):
            b[i, :] = apply_func_nb(a[0:i+1, :])
    else:
        for col in range(a.shape[1]):
            b[:, col] = expanding_apply_1d_nb(a[:, col], apply_func_nb)
    return b


@njit
def groupby_apply_1d_nb(a, by_b, apply_func_nb):
    """Group array by another array and apply a function."""
    groups = np.unique(by_b)
    group_idxs = []
    for i in range(len(groups)):
        idx_lst = []
        for j in range(len(by_b)):
            if by_b[j] == groups[i]:
                idx_lst.append(j)
        group_idxs.append(np.asarray(idx_lst))
    b = np.empty(len(groups))
    for i, idxs in enumerate(group_idxs):
        b[i] = apply_func_nb(a[idxs])
    return groups, b


@njit
def groupby_apply_nb(a, by_b, apply_func_nb, on_2d=False):
    groups = np.unique(by_b)
    group_idxs = []
    for i in range(len(groups)):
        idx_lst = []
        for j in range(len(by_b)):
            if by_b[j] == groups[i]:
                idx_lst.append(j)
        group_idxs.append(np.asarray(idx_lst))
    b = np.empty((len(groups), a.shape[1]))
    for i, idxs in enumerate(group_idxs):
        if on_2d:
            b[i, :] = apply_func_nb(a[idxs, :])
        else:
            for col in range(a.shape[1]):
                b[i, col] = apply_func_nb(a[idxs, col])
    return groups, b


@njit
def apply_by_mask_1d_nb(a, mask, apply_func_nb):
    """Apply a function on masked array."""
    b = np.empty(mask.shape[0])
    for i in range(mask.shape[0]):
        b[i] = apply_func_nb(a[mask[i, :]])
    return b


@njit
def apply_by_mask_nb(a, mask, apply_func_nb, on_2d=False):
    b = np.empty((mask.shape[0], a.shape[1]))
    if on_2d:
        for i in range(mask.shape[0]):
            b[i, :] = apply_func_nb(a[mask[i, :], :])
    else:
        for col in range(a.shape[1]):
            b[:, col] = apply_by_mask_1d_nb(a[:, col], mask, apply_func_nb)
    return b


@njit(f8[:, :](f8[:, :, :]), cache=True)
def nanmax_cube_nb(a):
    b = np.empty((a.shape[1], a.shape[2]), dtype=a.dtype)
    for i in range(a.shape[1]):
        for j in range(a.shape[2]):
            b[i, j] = np.nanmax(a[:, i, j])
    return b
