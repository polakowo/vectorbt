"""Numba-compiled 1-dim and 2-dim functions.

!!! note
    vectorbt treats matrices as first-class citizens and expects input arrays to be
    2-dim, unless function has suffix `_1d` or is meant to be input to another function. 
    Data is processed along index (axis 0).
    
    Input arrays can be of any data type, but most output arrays are `np.float64`.
    
    Rolling functions with `minp=None` have `min_periods` set to the window size.
    
    All functions passed as argument should be Numba-compiled."""

from numba import njit
import numpy as np

from vectorbt.generic.enums import DrawdownStatus, drawdown_dt


@njit(cache=True)
def shuffle_1d_nb(a, seed=None):
    """Shuffle each column in `a`.

    Specify `seed` to make output deterministic."""
    if seed is not None:
        np.random.seed(seed)
    return np.random.permutation(a)


@njit(cache=True)
def shuffle_nb(a, seed=None):
    """2-dim version of `shuffle_1d_nb`."""
    if seed is not None:
        np.random.seed(seed)
    out = np.empty_like(a, dtype=a.dtype)

    for col in range(a.shape[1]):
        out[:, col] = np.random.permutation(a[:, col])
    return out


@njit(cache=True)
def prepend_1d_nb(a, n, value):
    """Prepend value `n` times."""
    out = np.empty(a.shape[0] + n, dtype=np.float_)
    out[:n] = value
    out[n:] = a
    return out


@njit(cache=True)
def prepend_nb(a, n, value):
    """2-dim version of `prepend_1d_nb`."""
    out = np.empty((a.shape[0] + n, a.shape[1]), dtype=np.float_)
    out[:n, :] = value
    out[n:, :] = a
    return out


@njit(cache=True)
def set_by_mask_1d_nb(a, mask, value):
    """Set each element to a value by boolean mask."""
    out = a.astype(np.float_)
    out[mask] = value
    return out


@njit(cache=True)
def set_by_mask_nb(a, mask, value):
    """2-dim version of `set_by_mask_1d_nb`."""
    out = a.astype(np.float_)
    for col in range(a.shape[1]):
        out[mask[:, col], col] = value
    return out


@njit(cache=True)
def set_by_mask_mult_1d_nb(a, mask, values):
    """Set each element in one array to the corresponding element in another by boolean mask.

    `values` should be of the same shape as in `a`."""
    out = a.astype(np.float_)
    out[mask] = values[mask]
    return out


@njit(cache=True)
def set_by_mask_mult_nb(a, mask, values):
    """2-dim version of `set_by_mask_mult_1d_nb`."""
    out = a.astype(np.float_)
    for col in range(a.shape[1]):
        out[mask[:, col], col] = values[mask[:, col], col]
    return out


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
    out = np.empty_like(a, dtype=np.float_)
    out[:n] = np.nan
    out[n:] = a[:-n]
    return out


@njit(cache=True)
def fshift_nb(a, n):
    """2-dim version of `fshift_1d_nb`."""
    return fshift_1d_nb(a, n)


@njit(cache=True)
def diff_1d_nb(a):
    """Return the 1-th discrete difference.

    Numba equivalent to `pd.Series(a).diff()`."""
    out = np.empty_like(a, dtype=np.float_)
    out[0] = np.nan
    out[1:] = a[1:] - a[:-1]
    return out


@njit(cache=True)
def diff_nb(a):
    """2-dim version of `diff_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    out[0, :] = np.nan
    out[1:, :] = a[1:, :] - a[:-1, :]
    return out


@njit(cache=True)
def pct_change_1d_nb(a):
    """Return the percentage change.

    Numba equivalent to `pd.Series(a).pct_change()`."""
    out = np.empty_like(a, dtype=np.float_)
    out[0] = np.nan
    out[1:] = a[1:] / a[:-1] - 1
    return out


@njit(cache=True)
def pct_change_nb(a):
    """2-dim version of `pct_change_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    out[0, :] = np.nan
    out[1:, :] = a[1:, :] / a[:-1, :] - 1
    return out


@njit(cache=True)
def ffill_1d_nb(a):
    """Fill NaNs by propagating last valid observation forward.

    Numba equivalent to `pd.Series(a).fillna(method='ffill')`."""
    out = np.empty_like(a, dtype=np.float_)
    lastval = a[0]
    for i in range(a.shape[0]):
        if np.isnan(a[i]):
            out[i] = lastval
        else:
            lastval = out[i] = a[i]
    return out


@njit(cache=True)
def ffill_nb(a):
    """2-dim version of `ffill_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = ffill_1d_nb(a[:, col])
    return out


@njit(cache=True)
def product_1d_nb(a):
    """Return product.

    Numba equivalent to `pd.Series(a).prod()`."""
    out = a[0]
    for i in range(1, a.shape[0]):
        if ~np.isnan(a[i]):
            if np.isnan(out):
                out = a[i]
            else:
                out *= a[i]
    return out


@njit(cache=True)
def product_nb(a):
    """2-dim version of `product_1d_nb`."""
    out = np.empty(a.shape[1], dtype=np.float_)
    for col in range(a.shape[1]):
        out[col] = product_1d_nb(a[:, col])
    return out


@njit(cache=True)
def cumsum_1d_nb(a):
    """Return cumulative sum.

    Numba equivalent to `pd.Series(a).cumsum()`."""
    out = np.empty_like(a, dtype=np.float_)
    cumsum = 0
    for i in range(a.shape[0]):
        if ~np.isnan(a[i]):
            cumsum += a[i]
            out[i] = cumsum
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def cumsum_nb(a):
    """2-dim version of `cumsum_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = cumsum_1d_nb(a[:, col])
    return out


@njit(cache=True)
def cumprod_1d_nb(a):
    """Return cumulative product.

    Numba equivalent to `pd.Series(a).cumprod()`."""
    out = np.empty_like(a, dtype=np.float_)
    cumprod = 1
    for i in range(a.shape[0]):
        if ~np.isnan(a[i]):
            cumprod *= a[i]
            out[i] = cumprod
        else:
            out[i] = np.nan
    return out


@njit(cache=True)
def cumprod_nb(a):
    """2-dim version of `cumprod_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = cumprod_1d_nb(a[:, col])
    return out


@njit(cache=True)
def nancnt_nb(a):
    """Compute count while ignoring NaNs."""
    out = np.empty(a.shape[1], dtype=np.float_)
    for col in range(a.shape[1]):
        out[col] = np.sum(~np.isnan(a[:, col]))
    return out


@njit(cache=True)
def nansum_nb(a):
    """Compute sum while ignoring NaNs."""
    out = np.empty(a.shape[1], dtype=np.float_)
    for col in range(a.shape[1]):
        out[col] = np.nansum(a[:, col])
    return out


@njit(cache=True)
def nanmin_nb(a):
    """Compute minimum while ignoring NaNs."""
    out = np.empty(a.shape[1], dtype=np.float_)
    for col in range(a.shape[1]):
        out[col] = np.nanmin(a[:, col])
    return out


@njit(cache=True)
def nanmax_nb(a):
    """Compute maximum while ignoring NaNs."""
    out = np.empty(a.shape[1], dtype=np.float_)
    for col in range(a.shape[1]):
        out[col] = np.nanmax(a[:, col])
    return out


@njit(cache=True)
def nanmean_nb(a):
    """Compute mean while ignoring NaNs."""
    out = np.empty(a.shape[1], dtype=np.float_)
    for col in range(a.shape[1]):
        out[col] = np.nanmean(a[:, col])
    return out


@njit(cache=True)
def nanmedian_nb(a):
    """Compute median while ignoring NaNs."""
    out = np.empty(a.shape[1], dtype=np.float_)
    for col in range(a.shape[1]):
        out[col] = np.nanmedian(a[:, col])
    return out


@njit(cache=True)
def nanstd_1d_nb(a, ddof=0):
    """Compute the standard deviation while ignoring NaNs."""
    cnt = a.shape[0] - np.count_nonzero(np.isnan(a))
    rcount = max(cnt - ddof, 0)
    if rcount == 0:
        return np.nan
    else:
        return np.sqrt(np.nanvar(a) * cnt / rcount)


@njit(cache=True)
def nanstd_nb(a, ddof=0):
    """2-dim version of `nanstd_1d_nb`."""
    out = np.empty(a.shape[1], dtype=np.float_)
    for col in range(a.shape[1]):
        out[col] = nanstd_1d_nb(a[:, col], ddof=ddof)
    return out


# ############# Range functions ############# #


@njit(cache=True)
def concat_ranges_1d_nb(a, start_idxs, end_idxs):
    """Roll a window.

    For each index pair from `start_idxs` and `end_idxs`, slices `a` along index and concatenates."""
    out = np.empty((end_idxs[0] - start_idxs[0], start_idxs.shape[0]), dtype=a.dtype)
    for idx in range(start_idxs.shape[0]):
        out[:, idx] = a[start_idxs[idx]:end_idxs[idx]]
    return out


@njit(cache=True)
def concat_ranges_nb(a, start_idxs, end_idxs):
    """2-dim version of `range_1d_nb`."""
    out = np.empty((end_idxs[0] - start_idxs[0], start_idxs.shape[0] * a.shape[1]), dtype=a.dtype)
    for col in range(a.shape[1]):
        out[:, col * start_idxs.shape[0]:(col + 1) * start_idxs.shape[0]] = \
            concat_ranges_1d_nb(a[:, col], start_idxs, end_idxs)
    return out


# ############# Rolling functions ############# #


@njit(cache=True)
def rolling_min_1d_nb(a, window, minp=None):
    """Return rolling min.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).min()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(a, dtype=np.float_)
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
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@njit(cache=True)
def rolling_min_nb(a, window, minp=None):
    """2-dim version of `rolling_min_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = rolling_min_1d_nb(a[:, col], window, minp=minp)
    return out


@njit(cache=True)
def rolling_max_1d_nb(a, window, minp=None):
    """Return rolling max.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).max()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(a, dtype=np.float_)
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
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@njit(cache=True)
def rolling_max_nb(a, window, minp=None):
    """2-dim version of `rolling_max_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = rolling_max_1d_nb(a[:, col], window, minp=minp)
    return out


@njit(cache=True)
def rolling_mean_1d_nb(a, window, minp=None):
    """Return rolling mean.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).mean()`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(a, dtype=np.float_)
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
            out[i] = np.nan
        else:
            out[i] = window_cumsum / window_len
    return out


@njit(cache=True)
def rolling_mean_nb(a, window, minp=None):
    """2-dim version of `rolling_mean_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = rolling_mean_1d_nb(a[:, col], window, minp=minp)
    return out


@njit(cache=True)
def rolling_std_1d_nb(a, window, minp=None, ddof=0):
    """Return rolling standard deviation.

    Numba equivalent to `pd.Series(a).rolling(window, min_periods=minp).std(ddof=ddof)`."""
    if minp is None:
        minp = window
    if minp > window:
        raise ValueError("minp must be <= window")
    out = np.empty_like(a, dtype=np.float_)
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
        if window_len < minp or window_len == ddof:
            out[i] = np.nan
        else:
            mean = window_cumsum / window_len
            out[i] = np.sqrt(np.abs(window_cumsum_sq - 2 * window_cumsum *
                                    mean + window_len * mean ** 2) / (window_len - ddof))
    return out


@njit(cache=True)
def rolling_std_nb(a, window, minp=None, ddof=0):
    """2-dim version of `rolling_std_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = rolling_std_1d_nb(a[:, col], window, minp=minp, ddof=ddof)
    return out


@njit(cache=True)
def ewm_mean_1d_nb(a, span, minp=None, adjust=False):
    """Return exponential weighted average.

    Numba equivalent to `pd.Series(a).ewm(span=span, min_periods=minp, adjust=adjust).mean()`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewma` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    N = len(a)
    out = np.empty(N, dtype=np.float_)
    if N == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1. if adjust else alpha
    weighted_avg = a[0]
    is_observation = (weighted_avg == weighted_avg)
    nobs = int(is_observation)
    out[0] = weighted_avg if (nobs >= minp) else np.nan
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
                if adjust:
                    old_wt += new_wt
                else:
                    old_wt = 1.
        elif is_observation:
            weighted_avg = cur
        out[i] = weighted_avg if (nobs >= minp) else np.nan
    return out


@njit(cache=True)
def ewm_mean_nb(a, span, minp=None, adjust=False):
    """2-dim version of `ewm_mean_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = ewm_mean_1d_nb(a[:, col], span, minp=minp, adjust=adjust)
    return out


@njit(cache=True)
def ewm_std_1d_nb(a, span, minp=None, adjust=False, ddof=0):
    """Return exponential weighted standard deviation.

    Numba equivalent to `pd.Series(a).ewm(span=span, min_periods=minp).std(ddof=ddof)`.

    Adaptation of `pd._libs.window.aggregations.window_aggregations.ewmcov` with default arguments."""
    if minp is None:
        minp = span
    if minp > span:
        raise ValueError("minp must be <= span")
    N = len(a)
    out = np.empty(N, dtype=np.float_)
    if N == 0:
        return out
    com = (span - 1) / 2.0
    alpha = 1. / (1. + com)
    old_wt_factor = 1. - alpha
    new_wt = 1. if adjust else alpha
    mean_x = a[0]
    mean_y = a[0]
    is_observation = ((mean_x == mean_x) and (mean_y == mean_y))
    nobs = int(is_observation)
    if not is_observation:
        mean_x = np.nan
        mean_y = np.nan
    out[0] = np.nan
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
                if not adjust:
                    sum_wt /= old_wt
                    sum_wt2 /= (old_wt * old_wt)
                    old_wt = 1.
        elif is_observation:
            mean_x = cur_x
            mean_y = cur_y

        if nobs >= minp:
            numerator = sum_wt * sum_wt
            denominator = numerator - sum_wt2
            if denominator > 0.:
                out[i] = ((numerator / denominator) * cov)
            else:
                out[i] = np.nan
        else:
            out[i] = np.nan
    return np.sqrt(out)


@njit(cache=True)
def ewm_std_nb(a, span, minp=None, adjust=False, ddof=0):
    """2-dim version of `ewm_std_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = ewm_std_1d_nb(a[:, col], span, minp=minp, adjust=adjust, ddof=ddof)
    return out


# ############# Expanding functions ############# #


@njit(cache=True)
def expanding_min_1d_nb(a, minp=1):
    """Return expanding min.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).min()`."""
    out = np.empty_like(a, dtype=np.float_)
    minv = a[0]
    cnt = 0
    for i in range(a.shape[0]):
        if np.isnan(minv) or a[i] < minv:
            minv = a[i]
        if ~np.isnan(a[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = minv
    return out


@njit(cache=True)
def expanding_min_nb(a, minp=1):
    """2-dim version of `expanding_min_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = expanding_min_1d_nb(a[:, col], minp=minp)
    return out


@njit(cache=True)
def expanding_max_1d_nb(a, minp=1):
    """Return expanding max.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).max()`."""
    out = np.empty_like(a, dtype=np.float_)
    maxv = a[0]
    cnt = 0
    for i in range(a.shape[0]):
        if np.isnan(maxv) or a[i] > maxv:
            maxv = a[i]
        if ~np.isnan(a[i]):
            cnt += 1
        if cnt < minp:
            out[i] = np.nan
        else:
            out[i] = maxv
    return out


@njit(cache=True)
def expanding_max_nb(a, minp=1):
    """2-dim version of `expanding_max_1d_nb`."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        out[:, col] = expanding_max_1d_nb(a[:, col], minp=minp)
    return out


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
def expanding_std_1d_nb(a, minp=1, ddof=0):
    """Return expanding standard deviation.

    Numba equivalent to `pd.Series(a).expanding(min_periods=minp).std(ddof=ddof)`."""
    return rolling_std_1d_nb(a, a.shape[0], minp=minp, ddof=ddof)


@njit(cache=True)
def expanding_std_nb(a, minp=1, ddof=0):
    """2-dim version of `expanding_std_1d_nb`."""
    return rolling_std_nb(a, a.shape[0], minp=minp, ddof=ddof)


# ############# Apply functions ############# #


@njit
def rolling_apply_nb(a, window, apply_func_nb, *args):
    """Provide rolling window calculations.

    `apply_func_nb` should accept index of the row, index of the column,
    the array, and `*args`. Should return a single value."""
    out = np.empty_like(a, dtype=np.float_)
    for col in range(a.shape[1]):
        for i in range(a.shape[0]):
            window_a = a[max(0, i + 1 - window):i + 1, col]
            out[i, col] = apply_func_nb(i, col, window_a, *args)
    return out


@njit
def rolling_apply_matrix_nb(a, window, apply_func_nb, *args):
    """`rolling_apply_nb` with `apply_func_nb` being applied on all columns at once.

    `apply_func_nb` should accept index of the row, the 2-dim array, and `*args`.
    Should return a single value or an array of shape `a.shape[1]`."""
    out = np.empty_like(a, dtype=np.float_)
    for i in range(a.shape[0]):
        window_a = a[max(0, i + 1 - window):i + 1, :]
        out[i, :] = apply_func_nb(i, window_a, *args)
    return out


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

    `groups` should be a dictionary, where each key is an index that points to an element in the new array
    where a group-by result will be stored, while the value should be an array of indices in `a`
    to apply `apply_func_nb` on.

    `apply_func_nb` should accept indices of the group, index of the column,
    the array, and `*args`. Should return a single value."""
    out = np.empty((len(groups), a.shape[1]), dtype=np.float_)
    for col in range(a.shape[1]):
        for i, idxs in groups.items():
            out[i, col] = apply_func_nb(idxs, col, a[idxs, col], *args)
    return out


@njit
def groupby_apply_matrix_nb(a, groups, apply_func_nb, *args):
    """`groupby_apply_nb` with `apply_func_nb` being applied on all columns at once.

    `apply_func_nb` should accept indices of the group, the 2-dim array, and `*args`.
    Should return a single value or an array of shape `a.shape[1]`."""
    out = np.empty((len(groups), a.shape[1]), dtype=np.float_)
    for i, idxs in groups.items():
        out[i, :] = apply_func_nb(idxs, a[idxs, :], *args)
    return out


# ############# Map, filter and reduce ############# #


@njit
def applymap_nb(a, map_func_nb, *args):
    """Map non-NA elements elementwise using `map_func_nb`.

    `map_func_nb` should accept index of the row, index of the column,
    the element itself, and `*args`. Should return an array of same size."""
    out = np.full_like(a, np.nan, dtype=np.float_)

    for col in range(out.shape[1]):
        idxs = np.flatnonzero(~np.isnan(a[:, col]))
        for i in idxs:
            out[i, col] = map_func_nb(i, col, a[i, col], *args)
    return out


@njit
def filter_nb(a, filter_func_nb, *args):
    """Filter non-NA elements elementwise using `filter_func_nb`. 
    The filtered out elements will become NA.

    `filter_func_nb` should accept index of the row, index of the column,
    the element itself, and `*args`. Should return a boolean value."""
    out = a.astype(np.float_)

    for col in range(out.shape[1]):
        idxs = np.flatnonzero(~np.isnan(a[:, col]))
        for i in idxs:
            if not filter_func_nb(i, col, a[i, col], *args):
                out[i, col] = np.nan
    return out


@njit
def apply_and_reduce_nb(a, apply_func_nb, apply_args, reduce_func_nb, reduce_args):
    """Apply `apply_func_nb` on each column and reduce into a single value using `reduce_func_nb`.

    `apply_func_nb` should accept index of the column, the column itself, and `*apply_args`.
    Should return an array.

    `reduce_func_nb` should accept index of the column, the array of results from
    `apply_func_nb` for that column, and `*reduce_args`. Should return a single value."""
    out = np.full(a.shape[1], np.nan, dtype=np.float_)

    for col in range(a.shape[1]):
        mapped = apply_func_nb(col, a[:, col], *apply_args)
        out[col] = reduce_func_nb(col, mapped, *reduce_args)
    return out


@njit
def reduce_nb(a, reduce_func_nb, *args):
    """Reduce each column into a single value using `reduce_func_nb`.

    `reduce_func_nb` should accept index of the column, the array, and `*args`.
    Should return a single value."""
    out = np.full(a.shape[1], np.nan, dtype=np.float_)

    for col in range(a.shape[1]):
        out[col] = reduce_func_nb(col, a[:, col], *args)
    return out


@njit
def reduce_to_array_nb(a, reduce_func_nb, *args):
    """Reduce each column into an array of values using `reduce_func_nb`.

    `reduce_func_nb` same as for `reduce_nb` but should return an array.

    !!! note
        Output of `reduce_func_nb` should be strictly homogeneous."""
    out_inited = False
    for col in range(a.shape[1]):
        col_out = reduce_func_nb(col, a[:, col], *args)
        if not out_inited:
            out = np.full((col_out.shape[0], a.shape[1]), np.nan, dtype=np.float_)
            out_inited = True
        out[:, col] = col_out

    return out


@njit
def reduce_grouped_nb(a, group_lens, reduce_func_nb, *args):
    """Reduce each group of columns into a single value using `reduce_func_nb`.

    `reduce_func_nb` should accept index of the group, the array, and `*args`.
    Should return a single value."""
    out = np.empty(len(group_lens), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        out[group] = reduce_func_nb(group, a[:, from_col:to_col], *args)
        from_col = to_col
    return out


@njit(cache=True)
def flatten_forder_nb(a):
    """Flatten `a` in F order."""
    out = np.empty(a.shape[0] * a.shape[1], dtype=a.dtype)
    for col in range(a.shape[1]):
        out[col * a.shape[0]:(col + 1) * a.shape[0]] = a[:, col]
    return out


@njit
def flat_reduce_grouped_nb(a, group_lens, in_c_order, reduce_func_nb, *args):
    """Same as `reduce_grouped_nb` but passes flattened array."""
    out = np.empty(len(group_lens), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        if in_c_order:
            out[group] = reduce_func_nb(group, a[:, from_col:to_col].flatten(), *args)
        else:
            out[group] = reduce_func_nb(group, flatten_forder_nb(a[:, from_col:to_col]), *args)
        from_col = to_col
    return out


@njit
def reduce_grouped_to_array_nb(a, group_lens, reduce_func_nb, *args):
    """Reduce each group of columns into an array of values using `reduce_func_nb`.

    `reduce_func_nb` same as for `reduce_grouped_nb` but should return an array.

    !!! note
        Output of `reduce_func_nb` should be strictly homogeneous."""
    out_inited = False
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_out = reduce_func_nb(group, a[:, from_col:to_col], *args)
        if not out_inited:
            out = np.full((group_out.shape[0], len(group_lens)), np.nan, dtype=np.float_)
            out_inited = True
        out[:, group] = group_out
        from_col = to_col
    return out


@njit
def flat_reduce_grouped_to_array_nb(a, group_lens, in_c_order, reduce_func_nb, *args):
    """Same as `reduce_grouped_to_array_nb` but passes flattened 1D array."""
    out_inited = False
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        if in_c_order:
            group_out = reduce_func_nb(group, a[:, from_col:to_col].flatten(), *args)
        else:
            group_out = reduce_func_nb(group, flatten_forder_nb(a[:, from_col:to_col]), *args)
        if not out_inited:
            out = np.full((group_out.shape[0], len(group_lens)), np.nan, dtype=np.float_)
            out_inited = True
        out[:, group] = group_out
        from_col = to_col
    return out


@njit
def squeeze_grouped_nb(a, group_lens, reduce_func_nb, *args):
    """Squeeze each group of columns into a single column using `reduce_func_nb`.

    `reduce_func_nb` should accept index of the row, index of the group,
    the array, and `*args`. Should return a single value."""
    out = np.empty((a.shape[0], len(group_lens)), dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        for i in range(a.shape[0]):
            out[i, group] = reduce_func_nb(i, group, a[i, from_col:to_col], *args)
        from_col = to_col
    return out


# ############# Reshaping ############# #

@njit(cache=True)
def flatten_grouped_nb(a, group_lens, in_c_order):
    """Flatten each group of columns."""
    out = np.full((a.shape[0] * np.max(group_lens), len(group_lens)), np.nan, dtype=np.float_)
    from_col = 0
    for group in range(len(group_lens)):
        to_col = from_col + group_lens[group]
        group_len = to_col - from_col
        for k in range(group_len):
            if in_c_order:
                out[k::np.max(group_lens), group] = a[:, from_col + k]
            else:
                out[k * a.shape[0]:(k + 1) * a.shape[0], group] = a[:, from_col + k]
        from_col = to_col
    return out


# ############# Reducers ############# #


@njit(cache=True)
def nst_reduce_nb(col, a, n, *args):
    """Return nst element."""
    if n >= a.shape[0]:
        raise ValueError("index is out of bounds")
    return a[n]


@njit(cache=True)
def min_reduce_nb(col, a, *args):
    """Return min (ignores NaNs)."""
    return np.nanmin(a)


@njit(cache=True)
def max_reduce_nb(col, a, *args):
    """Return max (ignores NaNs)."""
    return np.nanmax(a)


@njit(cache=True)
def mean_reduce_nb(col, a, *args):
    """Return mean (ignores NaNs)."""
    return np.nanmean(a)


@njit(cache=True)
def median_reduce_nb(col, a, *args):
    """Return median (ignores NaNs)."""
    return np.nanmedian(a)


@njit(cache=True)
def sum_reduce_nb(col, a, *args):
    """Return sum (ignores NaNs)."""
    return np.nansum(a)


@njit(cache=True)
def count_reduce_nb(col, a, *args):
    """Return count (ignores NaNs)."""
    return np.sum(~np.isnan(a))


@njit(cache=True)
def std_reduce_nb(col, a, ddof, *args):
    """Return std (ignores NaNs)."""
    return nanstd_1d_nb(a, ddof=ddof)


@njit(cache=True)
def argmin_reduce_nb(col, a, *args):
    """Return position of min."""
    a = np.copy(a)
    mask = np.isnan(a)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    a[mask] = np.inf
    return np.argmin(a)


@njit(cache=True)
def argmax_reduce_nb(col, a, *args):
    """Return position of max."""
    a = np.copy(a)
    mask = np.isnan(a)
    if np.all(mask):
        raise ValueError("All-NaN slice encountered")
    a[mask] = -np.inf
    return np.argmax(a)


@njit(cache=True)
def describe_reduce_nb(col, a, perc, ddof, *args):
    """Return descriptive statistics (ignores NaNs).

    Numba equivalent to `pd.Series(a).describe(perc)`."""
    a = a[~np.isnan(a)]
    out = np.empty(5 + len(perc), dtype=np.float_)
    out[0] = len(a)
    if len(a) > 0:
        out[1] = np.mean(a)
        out[2] = nanstd_1d_nb(a, ddof=ddof)
        out[3] = np.min(a)
        out[4:-1] = np.percentile(a, perc * 100)
        out[4 + len(perc)] = np.max(a)
    else:
        out[1:] = np.nan
    return out


# ############# Drawdowns ############# #

@njit(cache=True)
def find_drawdowns_nb(ts):
    """Find drawdows and store their information as records to an array.

    ## Example

    Find drawdowns in time series:
    ```python-repl
    >>> import numpy as np
    >>> import pandas as pd
    >>> from vectorbt.generic.nb import find_drawdowns_nb

    >>> ts = np.asarray([
    ...     [1, 5, 1, 3],
    ...     [2, 4, 2, 2],
    ...     [3, 3, 3, 1],
    ...     [4, 2, 2, 2],
    ...     [5, 1, 1, 3]
    ... ])
    >>> records = find_drawdowns_nb(ts)

    >>> pd.DataFrame.from_records(records)
       id  col  start_idx  valley_idx  end_idx  status
    0   0    1          0           4        4       0
    1   1    2          2           4        4       0
    2   2    3          0           2        4       1
    ```
    """
    out = np.empty(ts.shape[0] * ts.shape[1], dtype=drawdown_dt)
    ridx = 0

    for col in range(ts.shape[1]):
        drawdown_started = False
        peak_idx = np.nan
        valley_idx = np.nan
        peak_val = ts[0, col]
        valley_val = ts[0, col]
        store_drawdown = False
        status = -1

        for i in range(ts.shape[0]):
            cur_val = ts[i, col]

            if not np.isnan(cur_val):
                if np.isnan(peak_val) or cur_val >= peak_val:
                    # Value increased
                    if not drawdown_started:
                        # If not running, register new peak
                        peak_val = cur_val
                        peak_idx = i
                    else:
                        # If running, potential recovery
                        if cur_val >= peak_val:
                            drawdown_started = False
                            store_drawdown = True
                            status = DrawdownStatus.Recovered
                else:
                    # Value decreased
                    if not drawdown_started:
                        # If not running, start new drawdown
                        drawdown_started = True
                        valley_val = cur_val
                        valley_idx = i
                    else:
                        # If running, potential valley
                        if cur_val < valley_val:
                            valley_val = cur_val
                            valley_idx = i

                if i == ts.shape[0] - 1 and drawdown_started:
                    # If still running, mark for save
                    drawdown_started = False
                    store_drawdown = True
                    status = DrawdownStatus.Active

                if store_drawdown:
                    # Save drawdown to the records
                    out[ridx]['id'] = ridx
                    out[ridx]['col'] = col
                    out[ridx]['start_idx'] = peak_idx
                    out[ridx]['valley_idx'] = valley_idx
                    out[ridx]['end_idx'] = i
                    out[ridx]['status'] = status
                    ridx += 1

                    # Reset running vars for a new drawdown
                    peak_idx = i
                    valley_idx = i
                    peak_val = cur_val
                    valley_val = cur_val
                    store_drawdown = False
                    status = -1

    return out[:ridx]


@njit(cache=True)
def dd_start_value_map_nb(record, ts):
    """`map_func_nb` that returns start value of a drawdown."""
    return ts[record['start_idx'], record['col']]


@njit(cache=True)
def dd_valley_value_map_nb(record, ts):
    """`map_func_nb` that returns valley value of a drawdown."""
    return ts[record['valley_idx'], record['col']]


@njit(cache=True)
def dd_end_value_map_nb(record, ts):
    """`map_func_nb` that returns end value of a drawdown.

    This can be either recovery value or last value of an active drawdown."""
    return ts[record['end_idx'], record['col']]


@njit(cache=True)
def dd_drawdown_map_nb(record, ts):
    """`map_func_nb` that returns drawdown value of a drawdown."""
    valley_val = dd_valley_value_map_nb(record, ts)
    start_val = dd_start_value_map_nb(record, ts)
    return (valley_val - start_val) / start_val


@njit(cache=True)
def dd_duration_map_nb(record):
    """`map_func_nb` that returns total duration of a drawdown."""
    return record['end_idx'] - record['start_idx']


@njit(cache=True)
def dd_ptv_duration_map_nb(record):
    """`map_func_nb` that returns duration of the peak-to-valley (PtV) phase."""
    return record['valley_idx'] - record['start_idx']


@njit(cache=True)
def dd_vtr_duration_map_nb(record):
    """`map_func_nb` that returns duration of the valley-to-recovery (VtR) phase."""
    return record['end_idx'] - record['valley_idx']


@njit(cache=True)
def dd_vtr_duration_ratio_map_nb(record):
    """`map_func_nb` that returns ratio of VtR duration to total duration."""
    return dd_vtr_duration_map_nb(record) / dd_duration_map_nb(record)


@njit(cache=True)
def dd_recovery_return_map_nb(record, ts):
    """`map_func_nb` that returns recovery return of a drawdown."""
    end_val = dd_end_value_map_nb(record, ts)
    valley_val = dd_valley_value_map_nb(record, ts)
    return (end_val - valley_val) / valley_val
