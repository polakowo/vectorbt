// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use ndarray::{Array2, ArrayView2};
use numpy::{Element, PyArray1, PyArray2, PyArrayDescr, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

const RANGE_OPEN: i64 = 0;
const RANGE_CLOSED: i64 = 1;
const DRAWDOWN_ACTIVE: i64 = 0;
const DRAWDOWN_RECOVERED: i64 = 1;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct RangeRecord {
    id: i64,
    col: i64,
    start_idx: i64,
    end_idx: i64,
    status: i64,
}

unsafe impl Element for RangeRecord {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        py.import_bound("vectorbt.generic.enums")
            .and_then(|m| m.getattr("range_dt"))
            .and_then(|dt| Ok(dt.downcast_into::<PyArrayDescr>()?))
            .expect("vectorbt.generic.enums.range_dt must be a NumPy dtype")
    }

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        *self
    }

    fn vec_from_slice(_py: Python<'_>, slc: &[Self]) -> Vec<Self> {
        slc.to_vec()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DrawdownRecord {
    id: i64,
    col: i64,
    peak_idx: i64,
    start_idx: i64,
    valley_idx: i64,
    end_idx: i64,
    peak_val: f64,
    valley_val: f64,
    end_val: f64,
    status: i64,
}

unsafe impl Element for DrawdownRecord {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        py.import_bound("vectorbt.generic.enums")
            .and_then(|m| m.getattr("drawdown_dt"))
            .and_then(|dt| Ok(dt.downcast_into::<PyArrayDescr>()?))
            .expect("vectorbt.generic.enums.drawdown_dt must be a NumPy dtype")
    }

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        *self
    }

    fn vec_from_slice(_py: Python<'_>, slc: &[Self]) -> Vec<Self> {
        slc.to_vec()
    }
}

fn validate_window(minp: usize, window: usize, name: &str) -> PyResult<()> {
    if minp > window {
        return Err(PyValueError::new_err(format!("minp must be <= {name}")));
    }
    Ok(())
}

fn check_bounds(n: isize, len: usize) -> PyResult<()> {
    if (n < 0 && n.unsigned_abs() > len) || (n >= 0 && n as usize >= len) {
        return Err(PyValueError::new_err("index is out of bounds"));
    }
    Ok(())
}

fn normalize_index(n: isize, len: usize) -> usize {
    if n >= 0 {
        n as usize
    } else {
        len - n.unsigned_abs()
    }
}

fn validate_group_lens(group_lens: &[i64]) -> PyResult<Vec<usize>> {
    let mut out = Vec::with_capacity(group_lens.len());
    for &group_len in group_lens {
        if group_len < 0 {
            return Err(PyValueError::new_err("group_lens must be non-negative"));
        }
        out.push(group_len as usize);
    }
    Ok(out)
}

fn broadcast_len2(len1: usize, len2: usize) -> PyResult<usize> {
    if len1 == len2 || len2 == 1 {
        Ok(len1)
    } else if len1 == 1 {
        Ok(len2)
    } else {
        Err(PyValueError::new_err(
            "operands could not be broadcast together",
        ))
    }
}

fn broadcast_len3(len1: usize, len2: usize, len3: usize) -> PyResult<usize> {
    if len1 == 0 || len2 == 0 || len3 == 0 {
        if (len1 == 0 || len1 == 1) && (len2 == 0 || len2 == 1) && (len3 == 0 || len3 == 1) {
            Ok(0)
        } else {
            Err(PyValueError::new_err(
                "operands could not be broadcast together",
            ))
        }
    } else {
        let out_len = len1.max(len2).max(len3);
        if (len1 == out_len || len1 == 1)
            && (len2 == out_len || len2 == 1)
            && (len3 == out_len || len3 == 1)
        {
            Ok(out_len)
        } else {
            Err(PyValueError::new_err(
                "operands could not be broadcast together",
            ))
        }
    }
}

fn broadcast_get<T: Copy>(values: &[T], i: usize) -> T {
    if values.len() == 1 {
        values[0]
    } else {
        values[i]
    }
}

fn apply_2d_by_col<F>(a: ArrayView2<'_, f64>, mut kernel: F) -> Array2<f64>
where
    F: FnMut(&[f64]) -> Vec<f64>,
{
    let (nrows, ncols) = a.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let mut col_buf = vec![0.0f64; nrows];
    for col in 0..ncols {
        for (i, &v) in a.column(col).iter().enumerate() {
            col_buf[i] = v;
        }
        let col_result = kernel(&col_buf);
        let mut out_col = out.column_mut(col);
        for (i, &v) in col_result.iter().enumerate() {
            out_col[i] = v;
        }
    }
    out
}

fn apply_2d_by_col_inplace<F>(a: ArrayView2<'_, f64>, mut kernel: F) -> Array2<f64>
where
    F: FnMut(&[f64], &mut [f64]),
{
    let (nrows, ncols) = a.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let mut col_buf = vec![0.0f64; nrows];
    let mut out_buf = vec![0.0f64; nrows];
    for col in 0..ncols {
        for (i, &v) in a.column(col).iter().enumerate() {
            col_buf[i] = v;
        }
        kernel(&col_buf, &mut out_buf);
        let mut out_col = out.column_mut(col);
        for (i, &v) in out_buf.iter().enumerate() {
            out_col[i] = v;
        }
    }
    out
}

fn reduce_2d_by_col<F>(a: ArrayView2<'_, f64>, mut kernel: F) -> Vec<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let (nrows, ncols) = a.dim();
    let mut out = vec![f64::NAN; ncols];
    let mut col_buf = vec![0.0f64; nrows];
    for col in 0..ncols {
        for (i, &v) in a.column(col).iter().enumerate() {
            col_buf[i] = v;
        }
        out[col] = kernel(&col_buf);
    }
    out
}

fn set_by_mask_1d(a: &[f64], mask: &[bool], value: f64) -> Vec<f64> {
    a.iter()
        .zip(mask.iter())
        .map(|(&v, &m)| if m { value } else { v })
        .collect()
}

fn set_by_mask_mult_1d(a: &[f64], mask: &[bool], values: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(mask.iter())
        .zip(values.iter())
        .map(|((&v, &m), &new_v)| if m { new_v } else { v })
        .collect()
}

fn fillna_1d(a: &[f64], value: f64) -> Vec<f64> {
    a.iter()
        .map(|&v| if v.is_nan() { value } else { v })
        .collect()
}

fn fillna_1d_into(a: &[f64], out: &mut [f64], value: f64) {
    for (i, &v) in a.iter().enumerate() {
        out[i] = if v.is_nan() { value } else { v };
    }
}

fn bshift_1d(a: &[f64], n: usize, fill_value: f64) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![fill_value; len];
    bshift_1d_into(a, &mut out, n, fill_value);
    out
}

fn bshift_1d_into(a: &[f64], out: &mut [f64], n: usize, fill_value: f64) {
    let len = a.len();
    for i in 0..len {
        if i + n < len {
            out[i] = a[i + n];
        } else {
            out[i] = fill_value;
        }
    }
}

fn fshift_1d(a: &[f64], n: usize, fill_value: f64) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![fill_value; len];
    fshift_1d_into(a, &mut out, n, fill_value);
    out
}

fn fshift_1d_into(a: &[f64], out: &mut [f64], n: usize, fill_value: f64) {
    let len = a.len();
    for i in 0..n.min(len) {
        out[i] = fill_value;
    }
    for i in n..len {
        out[i] = a[i - n];
    }
}

fn diff_1d(a: &[f64], n: usize) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![f64::NAN; len];
    diff_1d_into(a, &mut out, n);
    out
}

fn diff_1d_into(a: &[f64], out: &mut [f64], n: usize) {
    let len = a.len();
    for i in 0..n.min(len) {
        out[i] = f64::NAN;
    }
    for i in n..len {
        out[i] = a[i] - a[i - n];
    }
}

fn pct_change_1d(a: &[f64], n: usize) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![f64::NAN; len];
    pct_change_1d_into(a, &mut out, n);
    out
}

fn pct_change_1d_into(a: &[f64], out: &mut [f64], n: usize) {
    let len = a.len();
    for i in 0..n.min(len) {
        out[i] = f64::NAN;
    }
    for i in n..len {
        out[i] = a[i] / a[i - n] - 1.0;
    }
}

fn bfill_1d(a: &[f64]) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![f64::NAN; len];
    bfill_1d_into(a, &mut out);
    out
}

fn bfill_1d_into(a: &[f64], out: &mut [f64]) {
    let len = a.len();
    if len == 0 {
        return;
    }
    let mut lastval = a[len - 1];
    for i in (0..len).rev() {
        if a[i].is_nan() {
            out[i] = lastval;
        } else {
            lastval = a[i];
            out[i] = a[i];
        }
    }
}

fn ffill_1d(a: &[f64]) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![f64::NAN; len];
    ffill_1d_into(a, &mut out);
    out
}

fn ffill_1d_into(a: &[f64], out: &mut [f64]) {
    let len = a.len();
    if len == 0 {
        return;
    }
    let mut lastval = a[0];
    for i in 0..len {
        if a[i].is_nan() {
            out[i] = lastval;
        } else {
            lastval = a[i];
            out[i] = a[i];
        }
    }
}

fn nancumsum_1d_into(a: &[f64], out: &mut [f64]) {
    let mut sum = 0.0;
    for (i, &v) in a.iter().enumerate() {
        if !v.is_nan() {
            sum += v;
        }
        out[i] = sum;
    }
}

fn nancumprod_1d_into(a: &[f64], out: &mut [f64]) {
    let mut prod = 1.0;
    for (i, &v) in a.iter().enumerate() {
        if !v.is_nan() {
            prod *= v;
        }
        out[i] = prod;
    }
}

fn nanprod_1d(a: &[f64]) -> f64 {
    let mut out = 1.0;
    for &v in a {
        if !v.is_nan() {
            out *= v;
        }
    }
    out
}

fn nansum_1d(a: &[f64]) -> f64 {
    a.iter().filter(|v| !v.is_nan()).sum()
}

fn nancnt_1d(a: &[f64]) -> i64 {
    a.iter().filter(|v| !v.is_nan()).count() as i64
}

fn nanmin_1d(a: &[f64]) -> f64 {
    let mut iter = a.iter().copied();
    let mut out = loop {
        match iter.next() {
            Some(v) if !v.is_nan() => break v,
            Some(_) => continue,
            None => return f64::NAN,
        }
    };
    for v in iter {
        if v < out {
            out = v;
        }
    }
    out
}

fn nanmax_1d(a: &[f64]) -> f64 {
    let mut iter = a.iter().copied();
    let mut out = loop {
        match iter.next() {
            Some(v) if !v.is_nan() => break v,
            Some(_) => continue,
            None => return f64::NAN,
        }
    };
    for v in iter {
        if v > out {
            out = v;
        }
    }
    out
}

fn nanmean_1d(a: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut cnt = 0usize;
    for &v in a {
        if !v.is_nan() {
            sum += v;
            cnt += 1;
        }
    }
    if cnt == 0 {
        f64::NAN
    } else {
        sum / cnt as f64
    }
}

fn nanmedian_1d(a: &[f64]) -> f64 {
    let mut vals: Vec<f64> = a.iter().copied().filter(|v| !v.is_nan()).collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    let n = vals.len();
    let mid = n / 2;
    vals.select_nth_unstable_by(mid, |x, y| x.partial_cmp(y).unwrap());
    if n % 2 == 0 {
        let lower_max = vals[..mid]
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        (lower_max + vals[mid]) / 2.0
    } else {
        vals[mid]
    }
}

fn nanstd_1d(a: &[f64], ddof: usize) -> f64 {
    let mean = nanmean_1d(a);
    if mean.is_nan() {
        return f64::NAN;
    }
    let mut cnt = 0usize;
    let mut sq = 0.0;
    for &v in a {
        if !v.is_nan() {
            cnt += 1;
            let diff = v - mean;
            sq += diff * diff;
        }
    }
    if cnt <= ddof {
        f64::NAN
    } else {
        (sq / (cnt - ddof) as f64).sqrt()
    }
}

fn rolling_min_1d(a: &[f64], window: usize, minp: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let mut minv = a[i];
        let mut cnt = 0usize;
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        for j in start..=i {
            if a[j].is_nan() {
                continue;
            }
            if minv.is_nan() || a[j] < minv {
                minv = a[j];
            }
            cnt += 1;
        }
        if cnt >= minp {
            out[i] = minv;
        }
    }
    out
}

fn rolling_max_1d(a: &[f64], window: usize, minp: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    for i in 0..n {
        let mut maxv = a[i];
        let mut cnt = 0usize;
        let start = if i + 1 >= window { i + 1 - window } else { 0 };
        for j in start..=i {
            if a[j].is_nan() {
                continue;
            }
            if maxv.is_nan() || a[j] > maxv {
                maxv = a[j];
            }
            cnt += 1;
        }
        if cnt >= minp {
            out[i] = maxv;
        }
    }
    out
}

fn rolling_mean_1d(a: &[f64], window: usize, minp: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    let mut cumsum = 0.0f64;
    let mut nancnt = 0usize;
    let mut cumsum_arr = vec![0.0f64; n];
    let mut nancnt_arr = vec![0usize; n];
    for i in 0..n {
        if a[i].is_nan() {
            nancnt += 1;
        } else {
            cumsum += a[i];
        }
        nancnt_arr[i] = nancnt;
        cumsum_arr[i] = cumsum;
        let (window_len, window_cumsum) = if i < window {
            (i + 1 - nancnt, cumsum)
        } else {
            (
                window - (nancnt - nancnt_arr[i - window]),
                cumsum - cumsum_arr[i - window],
            )
        };
        if window_len >= minp {
            out[i] = window_cumsum / window_len as f64;
        }
    }
    out
}

fn rolling_std_1d(a: &[f64], window: usize, minp: usize, ddof: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    let mut cumsum = 0.0f64;
    let mut cumsum_sq = 0.0f64;
    let mut nancnt = 0usize;
    let mut cumsum_arr = vec![0.0f64; n];
    let mut cumsum_sq_arr = vec![0.0f64; n];
    let mut nancnt_arr = vec![0usize; n];
    for i in 0..n {
        if a[i].is_nan() {
            nancnt += 1;
        } else {
            cumsum += a[i];
            cumsum_sq += a[i] * a[i];
        }
        nancnt_arr[i] = nancnt;
        cumsum_arr[i] = cumsum;
        cumsum_sq_arr[i] = cumsum_sq;
        let (window_len, window_cumsum, window_cumsum_sq) = if i < window {
            (i + 1 - nancnt, cumsum, cumsum_sq)
        } else {
            (
                window - (nancnt - nancnt_arr[i - window]),
                cumsum - cumsum_arr[i - window],
                cumsum_sq - cumsum_sq_arr[i - window],
            )
        };
        if window_len >= minp && window_len > ddof {
            let mean = window_cumsum / window_len as f64;
            let variance = (window_cumsum_sq - 2.0 * window_cumsum * mean
                + window_len as f64 * mean * mean)
                / (window_len - ddof) as f64;
            out[i] = variance.abs().sqrt();
        }
    }
    out
}

fn ewm_mean_1d(a: &[f64], span: usize, minp: usize, adjust: bool) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let com = (span as f64 - 1.0) / 2.0;
    let alpha = 1.0 / (1.0 + com);
    let old_wt_factor = 1.0 - alpha;
    let new_wt = if adjust { 1.0 } else { alpha };
    let mut weighted_avg = a[0];
    let mut nobs = if weighted_avg == weighted_avg {
        1usize
    } else {
        0usize
    };
    if nobs >= minp {
        out[0] = weighted_avg;
    }
    let mut old_wt = 1.0;
    for i in 1..n {
        let cur = a[i];
        let is_observation = cur == cur;
        if is_observation {
            nobs += 1;
        }
        if weighted_avg == weighted_avg {
            old_wt *= old_wt_factor;
            if is_observation {
                if weighted_avg != cur {
                    weighted_avg = ((old_wt * weighted_avg) + (new_wt * cur)) / (old_wt + new_wt);
                }
                if adjust {
                    old_wt += new_wt;
                } else {
                    old_wt = 1.0;
                }
            }
        } else if is_observation {
            weighted_avg = cur;
        }
        if nobs >= minp {
            out[i] = weighted_avg;
        }
    }
    out
}

fn ewm_std_1d(a: &[f64], span: usize, minp: usize, adjust: bool, _ddof: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let com = (span as f64 - 1.0) / 2.0;
    let alpha = 1.0 / (1.0 + com);
    let old_wt_factor = 1.0 - alpha;
    let new_wt = if adjust { 1.0 } else { alpha };
    let mut mean_x = a[0];
    let mut mean_y = a[0];
    let is_observation = mean_x == mean_x && mean_y == mean_y;
    let mut nobs = if is_observation { 1usize } else { 0usize };
    if !is_observation {
        mean_x = f64::NAN;
        mean_y = f64::NAN;
    }
    let mut cov = 0.0;
    let mut sum_wt = 1.0;
    let mut sum_wt2 = 1.0;
    let mut old_wt = 1.0;
    for i in 1..n {
        let cur_x = a[i];
        let cur_y = a[i];
        let is_observation = cur_x == cur_x && cur_y == cur_y;
        if is_observation {
            nobs += 1;
        }
        if mean_x == mean_x {
            sum_wt *= old_wt_factor;
            sum_wt2 *= old_wt_factor * old_wt_factor;
            old_wt *= old_wt_factor;
            if is_observation {
                let old_mean_x = mean_x;
                let old_mean_y = mean_y;
                if mean_x != cur_x {
                    mean_x = ((old_wt * old_mean_x) + (new_wt * cur_x)) / (old_wt + new_wt);
                }
                if mean_y != cur_y {
                    mean_y = ((old_wt * old_mean_y) + (new_wt * cur_y)) / (old_wt + new_wt);
                }
                cov = ((old_wt * (cov + ((old_mean_x - mean_x) * (old_mean_y - mean_y))))
                    + (new_wt * ((cur_x - mean_x) * (cur_y - mean_y))))
                    / (old_wt + new_wt);
                sum_wt += new_wt;
                sum_wt2 += new_wt * new_wt;
                old_wt += new_wt;
                if !adjust {
                    sum_wt /= old_wt;
                    sum_wt2 /= old_wt * old_wt;
                    old_wt = 1.0;
                }
            }
        } else if is_observation {
            mean_x = cur_x;
            mean_y = cur_y;
        }
        if nobs >= minp {
            let numerator = sum_wt * sum_wt;
            let denominator = numerator - sum_wt2;
            if denominator > 0.0 {
                out[i] = ((numerator / denominator) * cov).sqrt();
            }
        }
    }
    out
}

fn expanding_min_1d(a: &[f64], minp: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let mut minv = a[0];
    let mut cnt = 0usize;
    for i in 0..n {
        if minv.is_nan() || a[i] < minv {
            minv = a[i];
        }
        if !a[i].is_nan() {
            cnt += 1;
        }
        if cnt >= minp {
            out[i] = minv;
        }
    }
    out
}

fn expanding_max_1d(a: &[f64], minp: usize) -> Vec<f64> {
    let n = a.len();
    let mut out = vec![f64::NAN; n];
    if n == 0 {
        return out;
    }
    let mut maxv = a[0];
    let mut cnt = 0usize;
    for i in 0..n {
        if maxv.is_nan() || a[i] > maxv {
            maxv = a[i];
        }
        if !a[i].is_nan() {
            cnt += 1;
        }
        if cnt >= minp {
            out[i] = maxv;
        }
    }
    out
}

fn shuffle_1d_with_rng<R: Rng + ?Sized>(a: &[f64], rng: &mut R) -> Vec<f64> {
    let mut out = a.to_vec();
    out.shuffle(rng);
    out
}

fn shuffle_1d(a: &[f64], seed: Option<u64>) -> Vec<f64> {
    match seed {
        Some(seed) => {
            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            shuffle_1d_with_rng(a, &mut rng)
        }
        None => {
            let mut rng = rand::thread_rng();
            shuffle_1d_with_rng(a, &mut rng)
        }
    }
}

fn flatten_forder(a: ArrayView2<'_, f64>) -> Vec<f64> {
    let (nrows, ncols) = a.dim();
    let mut out = Vec::with_capacity(nrows * ncols);
    for col in 0..ncols {
        out.extend(a.column(col).iter().copied());
    }
    out
}

fn flatten_grouped(a: ArrayView2<'_, f64>, group_lens: &[usize], in_c_order: bool) -> Array2<f64> {
    let (nrows, _) = a.dim();
    let max_group_len = group_lens.iter().copied().max().unwrap_or(0);
    let mut out = Array2::<f64>::from_elem((nrows * max_group_len, group_lens.len()), f64::NAN);
    let mut from_col = 0usize;
    for (group, &group_len) in group_lens.iter().enumerate() {
        for k in 0..group_len {
            let col_view = a.column(from_col + k);
            for row in 0..nrows {
                let out_row = if in_c_order {
                    k + row * max_group_len
                } else {
                    k * nrows + row
                };
                out[[out_row, group]] = col_view[row];
            }
        }
        from_col += group_len;
    }
    out
}

fn flatten_uniform_grouped(
    a: ArrayView2<'_, f64>,
    group_lens: &[usize],
    in_c_order: bool,
) -> Array2<f64> {
    flatten_grouped(a, group_lens, in_c_order)
}

fn min_squeeze_1d(a: &[f64]) -> f64 {
    nanmin_1d(a)
}

fn max_squeeze_1d(a: &[f64]) -> f64 {
    nanmax_1d(a)
}

fn sum_squeeze_1d(a: &[f64]) -> f64 {
    nansum_1d(a)
}

fn any_squeeze_1d(a: &[f64]) -> bool {
    a.iter().any(|&v| v != 0.0)
}

fn argmin_reduce_1d(a: &[f64]) -> PyResult<i64> {
    let mut best_idx = 0usize;
    let mut best_val = f64::NAN;
    for (i, &v) in a.iter().enumerate() {
        if !v.is_nan() {
            best_idx = i;
            best_val = v;
            break;
        }
    }
    if best_val.is_nan() {
        return Err(PyValueError::new_err("All-NaN slice encountered"));
    }
    for (i, &v) in a.iter().enumerate().skip(best_idx + 1) {
        if v < best_val {
            best_idx = i;
            best_val = v;
        }
    }
    Ok(best_idx as i64)
}

fn argmax_reduce_1d(a: &[f64]) -> PyResult<i64> {
    let mut best_idx = 0usize;
    let mut best_val = f64::NAN;
    for (i, &v) in a.iter().enumerate() {
        if !v.is_nan() {
            best_idx = i;
            best_val = v;
            break;
        }
    }
    if best_val.is_nan() {
        return Err(PyValueError::new_err("All-NaN slice encountered"));
    }
    for (i, &v) in a.iter().enumerate().skip(best_idx + 1) {
        if v > best_val {
            best_idx = i;
            best_val = v;
        }
    }
    Ok(best_idx as i64)
}

fn percentile_sorted(vals: &[f64], q: f64) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    if vals.len() == 1 {
        return vals[0];
    }
    let rank = (q / 100.0) * (vals.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        vals[lo]
    } else {
        vals[lo] * (hi as f64 - rank) + vals[hi] * (rank - lo as f64)
    }
}

fn describe_reduce_1d(a: &[f64], perc: &[f64], ddof: usize) -> Vec<f64> {
    let mut vals: Vec<f64> = a.iter().copied().filter(|v| !v.is_nan()).collect();
    vals.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut out = vec![f64::NAN; 5 + perc.len()];
    out[0] = vals.len() as f64;
    if vals.is_empty() {
        return out;
    }
    out[1] = vals.iter().sum::<f64>() / vals.len() as f64;
    out[2] = nanstd_1d(&vals, ddof);
    out[3] = vals[0];
    for (i, &p) in perc.iter().enumerate() {
        out[4 + i] = percentile_sorted(&vals, p * 100.0);
    }
    out[4 + perc.len()] = vals[vals.len() - 1];
    out
}

fn value_counts(
    codes: ArrayView2<'_, i64>,
    n_uniques: usize,
    group_lens: &[usize],
) -> PyResult<Array2<i64>> {
    let (nrows, ncols) = codes.dim();
    let mut out = Array2::<i64>::zeros((n_uniques, group_lens.len()));
    let mut from_col = 0usize;
    for (group, &group_len) in group_lens.iter().enumerate() {
        let to_col = from_col + group_len;
        if to_col > ncols {
            return Err(PyValueError::new_err("group_lens exceed number of columns"));
        }
        for col in from_col..to_col {
            let col_view = codes.column(col);
            for row in 0..nrows {
                let code = col_view[row];
                if code < 0 || code as usize >= n_uniques {
                    return Err(PyValueError::new_err("codes must be in [0, n_uniques)"));
                }
                out[[code as usize, group]] += 1;
            }
        }
        from_col = to_col;
    }
    Ok(out)
}

fn find_ranges(a: ArrayView2<'_, f64>, gap_value: f64) -> Vec<RangeRecord> {
    let (nrows, ncols) = a.dim();
    let mut out = Vec::<RangeRecord>::with_capacity(nrows * ncols);
    let mut ridx = 0i64;
    for col in 0..ncols {
        let col_view = a.column(col);
        let mut range_started = false;
        let mut start_idx = -1i64;
        let mut end_idx = -1i64;
        let mut store_record = false;
        let mut status = -1i64;
        for i in 0..nrows {
            let cur_val = col_view[i];
            if cur_val == gap_value || (cur_val.is_nan() && gap_value.is_nan()) {
                if range_started {
                    end_idx = i as i64;
                    range_started = false;
                    store_record = true;
                    status = RANGE_CLOSED;
                }
            } else if !range_started {
                start_idx = i as i64;
                range_started = true;
            }
            if i == nrows - 1 && range_started {
                end_idx = (nrows - 1) as i64;
                range_started = false;
                store_record = true;
                status = RANGE_OPEN;
            }
            if store_record {
                out.push(RangeRecord {
                    id: ridx,
                    col: col as i64,
                    start_idx,
                    end_idx,
                    status,
                });
                ridx += 1;
                store_record = false;
            }
        }
    }
    out
}

fn range_duration(start_idx: &[i64], end_idx: &[i64], status: &[i64]) -> Vec<i64> {
    let mut out = vec![0i64; start_idx.len()];
    for i in 0..out.len() {
        if status[i] == RANGE_OPEN {
            out[i] = end_idx[i] - start_idx[i] + 1;
        } else {
            out[i] = end_idx[i] - start_idx[i];
        }
    }
    out
}

fn range_coverage(
    start_idx: &[i64],
    end_idx: &[i64],
    status: &[i64],
    col_idxs: &[i64],
    col_lens: &[i64],
    index_lens: &[i64],
    overlapping: bool,
    normalize: bool,
) -> PyResult<Vec<f64>> {
    let mut out = vec![f64::NAN; col_lens.len()];
    let mut col_start = 0usize;
    for col in 0..col_lens.len() {
        let col_len = col_lens[col];
        if col_len < 0 || index_lens[col] < 0 {
            return Err(PyValueError::new_err(
                "col_lens and index_lens must be non-negative",
            ));
        }
        if col_len == 0 {
            continue;
        }
        let index_len = index_lens[col] as usize;
        let mut temp = vec![0i64; index_len];
        for j in col_start..col_start + col_len as usize {
            let ridx = col_idxs[j] as usize;
            let start = start_idx[ridx] as usize;
            let mut end = end_idx[ridx] as usize;
            if status[ridx] == RANGE_OPEN {
                end += 1;
            }
            for k in start..end {
                temp[k] += 1;
            }
        }
        if overlapping {
            let over = temp.iter().filter(|&&v| v > 1).count() as f64;
            if normalize {
                let covered = temp.iter().filter(|&&v| v > 0).count() as f64;
                out[col] = over / covered;
            } else {
                out[col] = over;
            }
        } else {
            let covered = temp.iter().filter(|&&v| v > 0).count() as f64;
            if normalize {
                out[col] = covered / index_len as f64;
            } else {
                out[col] = covered;
            }
        }
        col_start += col_len as usize;
    }
    Ok(out)
}

fn ranges_to_mask(
    start_idx: &[i64],
    end_idx: &[i64],
    status: &[i64],
    col_idxs: &[i64],
    col_lens: &[i64],
    index_len: usize,
) -> PyResult<Array2<bool>> {
    let mut out = Array2::<bool>::from_elem((index_len, col_lens.len()), false);
    let mut col_start = 0usize;
    for col in 0..col_lens.len() {
        let col_len = col_lens[col];
        if col_len < 0 {
            return Err(PyValueError::new_err("col_lens must be non-negative"));
        }
        for j in col_start..col_start + col_len as usize {
            let ridx = col_idxs[j] as usize;
            let start = start_idx[ridx] as usize;
            let mut end = end_idx[ridx] as usize;
            if status[ridx] == RANGE_OPEN {
                end += 1;
            }
            for k in start..end {
                out[[k, col]] = true;
            }
        }
        col_start += col_len as usize;
    }
    Ok(out)
}

fn get_drawdowns(a: ArrayView2<'_, f64>) -> Vec<DrawdownRecord> {
    let (nrows, ncols) = a.dim();
    let mut out = Vec::<DrawdownRecord>::with_capacity(nrows * ncols);
    if nrows == 0 {
        return out;
    }
    let mut ddidx = 0i64;
    for col in 0..ncols {
        let col_view = a.column(col);
        let mut drawdown_started = false;
        let mut peak_idx = -1i64;
        let mut valley_idx = -1i64;
        let mut peak_val = col_view[0];
        let mut valley_val = col_view[0];
        let mut store_record = false;
        let mut status = -1i64;
        for i in 0..nrows {
            let cur_val = col_view[i];
            if !cur_val.is_nan() {
                if peak_val.is_nan() || cur_val >= peak_val {
                    if !drawdown_started {
                        peak_val = cur_val;
                        peak_idx = i as i64;
                    } else if cur_val >= peak_val {
                        drawdown_started = false;
                        store_record = true;
                        status = DRAWDOWN_RECOVERED;
                    }
                } else if !drawdown_started {
                    drawdown_started = true;
                    valley_val = cur_val;
                    valley_idx = i as i64;
                } else if cur_val < valley_val {
                    valley_val = cur_val;
                    valley_idx = i as i64;
                }
                if i == nrows - 1 && drawdown_started {
                    drawdown_started = false;
                    store_record = true;
                    status = DRAWDOWN_ACTIVE;
                }
                if store_record {
                    out.push(DrawdownRecord {
                        id: ddidx,
                        col: col as i64,
                        peak_idx,
                        start_idx: peak_idx + 1,
                        valley_idx,
                        end_idx: i as i64,
                        peak_val,
                        valley_val,
                        end_val: cur_val,
                        status,
                    });
                    ddidx += 1;
                    peak_idx = i as i64;
                    valley_idx = i as i64;
                    peak_val = cur_val;
                    valley_val = cur_val;
                    store_record = false;
                    status = -1;
                }
            }
        }
    }
    out
}

fn crossed_above_1d(arr1: &[f64], arr2: &[f64], wait: usize) -> Vec<bool> {
    let mut out = vec![false; arr1.len()];
    let mut was_below = false;
    let mut crossed_ago = -1i64;
    for i in 0..arr1.len() {
        if arr1[i].is_nan() || arr2[i].is_nan() {
            crossed_ago = -1;
            was_below = false;
        } else if arr1[i] > arr2[i] {
            if was_below {
                crossed_ago += 1;
                out[i] = crossed_ago == wait as i64;
            }
        } else if arr1[i] == arr2[i] {
            crossed_ago = -1;
        } else {
            crossed_ago = -1;
            was_below = true;
        }
    }
    out
}

macro_rules! export_1d_array {
    ($name:ident, $kernel:ident $(, $arg:ident : $argty:ty)*) => {
        #[pyfunction]
        pub fn $name<'py>(
            py: Python<'py>,
            a: PyReadonlyArray1<'py, f64>,
            $($arg: $argty,)*
        ) -> PyResult<Bound<'py, PyArray1<f64>>> {
            let a_slice = a.as_slice()?;
            let result = py.allow_threads(|| $kernel(a_slice, $($arg),*));
            Ok(PyArray1::from_vec_bound(py, result))
        }
    };
}

#[pyfunction]
pub fn set_by_mask_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    mask: PyReadonlyArray1<'py, bool>,
    value: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let mask_slice = mask.as_slice()?;
    let result = py.allow_threads(|| set_by_mask_1d(a_slice, mask_slice, value));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn set_by_mask_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    value: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let mask_arr = mask.as_array();
    let (nrows, ncols) = a_arr.dim();
    let result = py.allow_threads(|| {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        for row in 0..nrows {
            for col in 0..ncols {
                out[[row, col]] = if mask_arr[[row, col]] {
                    value
                } else {
                    a_arr[[row, col]]
                };
            }
        }
        out
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn set_by_mask_mult_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    mask: PyReadonlyArray1<'py, bool>,
    values: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let mask_slice = mask.as_slice()?;
    let values_slice = values.as_slice()?;
    let result = py.allow_threads(|| set_by_mask_mult_1d(a_slice, mask_slice, values_slice));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn set_by_mask_mult_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    mask: PyReadonlyArray2<'py, bool>,
    values: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let mask_arr = mask.as_array();
    let values_arr = values.as_array();
    let (nrows, ncols) = a_arr.dim();
    let result = py.allow_threads(|| {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        for row in 0..nrows {
            for col in 0..ncols {
                out[[row, col]] = if mask_arr[[row, col]] {
                    values_arr[[row, col]]
                } else {
                    a_arr[[row, col]]
                };
            }
        }
        out
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(fillna_1d_rs, fillna_1d, value: f64);

#[pyfunction]
pub fn fillna_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    value: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| {
        apply_2d_by_col_inplace(a_arr, |col, out| fillna_1d_into(col, out, value))
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(bshift_1d_rs, bshift_1d, n: usize, fill_value: f64);

#[pyfunction]
pub fn bshift_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    n: usize,
    fill_value: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| {
        apply_2d_by_col_inplace(a_arr, |col, out| bshift_1d_into(col, out, n, fill_value))
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(fshift_1d_rs, fshift_1d, n: usize, fill_value: f64);

#[pyfunction]
pub fn fshift_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    n: usize,
    fill_value: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| {
        apply_2d_by_col_inplace(a_arr, |col, out| fshift_1d_into(col, out, n, fill_value))
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(diff_1d_rs, diff_1d, n: usize);

#[pyfunction]
pub fn diff_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col_inplace(a_arr, |col, out| diff_1d_into(col, out, n)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(pct_change_1d_rs, pct_change_1d, n: usize);

#[pyfunction]
pub fn pct_change_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| {
        apply_2d_by_col_inplace(a_arr, |col, out| pct_change_1d_into(col, out, n))
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(bfill_1d_rs, bfill_1d);

#[pyfunction]
pub fn bfill_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col_inplace(a_arr, |col, out| bfill_1d_into(col, out)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(ffill_1d_rs, ffill_1d);

#[pyfunction]
pub fn ffill_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col_inplace(a_arr, |col, out| ffill_1d_into(col, out)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn nanprod_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| reduce_2d_by_col(a_arr, nanprod_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn nancumsum_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col_inplace(a_arr, |col, out| nancumsum_1d_into(col, out)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn nancumprod_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py
        .allow_threads(|| apply_2d_by_col_inplace(a_arr, |col, out| nancumprod_1d_into(col, out)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn nansum_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| reduce_2d_by_col(a_arr, nansum_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn nancnt_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a_arr = a.as_array();
    let (nrows, ncols) = a_arr.dim();
    let result = py.allow_threads(|| {
        let mut out = vec![0i64; ncols];
        let mut col_buf = vec![0.0f64; nrows];
        for col in 0..ncols {
            for (i, &v) in a_arr.column(col).iter().enumerate() {
                col_buf[i] = v;
            }
            out[col] = nancnt_1d(&col_buf);
        }
        out
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn nanmin_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| reduce_2d_by_col(a_arr, nanmin_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn nanmax_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| reduce_2d_by_col(a_arr, nanmax_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn nanmean_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| reduce_2d_by_col(a_arr, nanmean_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn nanmedian_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| reduce_2d_by_col(a_arr, nanmedian_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn nanstd_1d_rs(a: PyReadonlyArray1<'_, f64>, ddof: usize) -> PyResult<f64> {
    Ok(nanstd_1d(a.as_slice()?, ddof))
}

#[pyfunction]
pub fn nanstd_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| reduce_2d_by_col(a_arr, |col| nanstd_1d(col, ddof)));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None))]
pub fn rolling_min_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| rolling_min_1d(a_slice, window, minp));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None))]
pub fn rolling_min_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col(a_arr, |col| rolling_min_1d(col, window, minp)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None))]
pub fn rolling_max_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| rolling_max_1d(a_slice, window, minp));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None))]
pub fn rolling_max_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col(a_arr, |col| rolling_max_1d(col, window, minp)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None))]
pub fn rolling_mean_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| rolling_mean_1d(a_slice, window, minp));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None))]
pub fn rolling_mean_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col(a_arr, |col| rolling_mean_1d(col, window, minp)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None, ddof=0))]
pub fn rolling_std_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| rolling_std_1d(a_slice, window, minp, ddof));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None, ddof=0))]
pub fn rolling_std_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col(a_arr, |col| rolling_std_1d(col, window, minp, ddof)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, span, minp=0, adjust=false))]
pub fn ewm_mean_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    span: usize,
    minp: usize,
    adjust: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_window(minp, span, "span")?;
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| ewm_mean_1d(a_slice, span, minp, adjust));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, span, minp=0, adjust=false))]
pub fn ewm_mean_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    span: usize,
    minp: usize,
    adjust: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_window(minp, span, "span")?;
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col(a_arr, |col| ewm_mean_1d(col, span, minp, adjust)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, span, minp=0, adjust=false, ddof=0))]
pub fn ewm_std_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    span: usize,
    minp: usize,
    adjust: bool,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    validate_window(minp, span, "span")?;
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| ewm_std_1d(a_slice, span, minp, adjust, ddof));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, span, minp=0, adjust=false, ddof=0))]
pub fn ewm_std_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    span: usize,
    minp: usize,
    adjust: bool,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    validate_window(minp, span, "span")?;
    let a_arr = a.as_array();
    let result = py
        .allow_threads(|| apply_2d_by_col(a_arr, |col| ewm_std_1d(col, span, minp, adjust, ddof)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(expanding_min_1d_rs, expanding_min_1d, minp: usize);

#[pyfunction]
pub fn expanding_min_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    minp: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| apply_2d_by_col(a_arr, |col| expanding_min_1d(col, minp)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

export_1d_array!(expanding_max_1d_rs, expanding_max_1d, minp: usize);

#[pyfunction]
pub fn expanding_max_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    minp: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| apply_2d_by_col(a_arr, |col| expanding_max_1d(col, minp)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn expanding_mean_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    minp: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| rolling_mean_1d(a_slice, a_slice.len(), minp));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn expanding_mean_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    minp: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result =
        py.allow_threads(|| apply_2d_by_col(a_arr, |col| rolling_mean_1d(col, col.len(), minp)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn expanding_std_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    minp: usize,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| rolling_std_1d(a_slice, a_slice.len(), minp, ddof));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn expanding_std_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    minp: usize,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py
        .allow_threads(|| apply_2d_by_col(a_arr, |col| rolling_std_1d(col, col.len(), minp, ddof)));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, seed=None))]
pub fn shuffle_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| shuffle_1d(a_slice, seed));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, seed=None))]
pub fn shuffle_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| {
        let (nrows, ncols) = a_arr.dim();
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        let mut col_buf = vec![0.0f64; nrows];
        match seed {
            Some(seed) => {
                let mut rng = ChaCha8Rng::seed_from_u64(seed);
                for col in 0..ncols {
                    for (i, &v) in a_arr.column(col).iter().enumerate() {
                        col_buf[i] = v;
                    }
                    let col_result = shuffle_1d_with_rng(&col_buf, &mut rng);
                    let mut out_col = out.column_mut(col);
                    for (i, &v) in col_result.iter().enumerate() {
                        out_col[i] = v;
                    }
                }
            }
            None => {
                let mut rng = rand::thread_rng();
                for col in 0..ncols {
                    for (i, &v) in a_arr.column(col).iter().enumerate() {
                        col_buf[i] = v;
                    }
                    let col_result = shuffle_1d_with_rng(&col_buf, &mut rng);
                    let mut out_col = out.column_mut(col);
                    for (i, &v) in col_result.iter().enumerate() {
                        out_col[i] = v;
                    }
                }
            }
        }
        out
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn flatten_forder_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| flatten_forder(a_arr));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn flatten_grouped_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    in_c_order: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let group_lens_vec = validate_group_lens(group_lens.as_slice()?)?;
    let result = py.allow_threads(|| flatten_grouped(a_arr, &group_lens_vec, in_c_order));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn flatten_uniform_grouped_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    in_c_order: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let group_lens_vec = validate_group_lens(group_lens.as_slice()?)?;
    let result = py.allow_threads(|| flatten_uniform_grouped(a_arr, &group_lens_vec, in_c_order));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn nth_reduce_rs(a: PyReadonlyArray1<'_, f64>, n: isize) -> PyResult<f64> {
    let a_slice = a.as_slice()?;
    check_bounds(n, a_slice.len())?;
    Ok(a_slice[normalize_index(n, a_slice.len())])
}

#[pyfunction]
pub fn nth_index_reduce_rs(a: PyReadonlyArray1<'_, f64>, n: isize) -> PyResult<i64> {
    let a_slice = a.as_slice()?;
    check_bounds(n, a_slice.len())?;
    Ok(normalize_index(n, a_slice.len()) as i64)
}

#[pyfunction]
pub fn min_reduce_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(nanmin_1d(a.as_slice()?))
}

#[pyfunction]
pub fn max_reduce_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(nanmax_1d(a.as_slice()?))
}

#[pyfunction]
pub fn mean_reduce_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(nanmean_1d(a.as_slice()?))
}

#[pyfunction]
pub fn median_reduce_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(nanmedian_1d(a.as_slice()?))
}

#[pyfunction]
pub fn std_reduce_rs(a: PyReadonlyArray1<'_, f64>, ddof: usize) -> PyResult<f64> {
    Ok(nanstd_1d(a.as_slice()?, ddof))
}

#[pyfunction]
pub fn sum_reduce_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(nansum_1d(a.as_slice()?))
}

#[pyfunction]
pub fn count_reduce_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<i64> {
    Ok(nancnt_1d(a.as_slice()?))
}

#[pyfunction]
pub fn argmin_reduce_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<i64> {
    argmin_reduce_1d(a.as_slice()?)
}

#[pyfunction]
pub fn argmax_reduce_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<i64> {
    argmax_reduce_1d(a.as_slice()?)
}

#[pyfunction]
pub fn describe_reduce_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    perc: PyReadonlyArray1<'py, f64>,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let perc_slice = perc.as_slice()?;
    let result = py.allow_threads(|| describe_reduce_1d(a_slice, perc_slice, ddof));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn value_counts_rs<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray2<'py, i64>,
    n_uniques: usize,
    group_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let codes_arr = codes.as_array();
    let group_lens_vec = validate_group_lens(group_lens.as_slice()?)?;
    let result = py.allow_threads(|| value_counts(codes_arr, n_uniques, &group_lens_vec))?;
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn min_squeeze_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(min_squeeze_1d(a.as_slice()?))
}

#[pyfunction]
pub fn max_squeeze_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(max_squeeze_1d(a.as_slice()?))
}

#[pyfunction]
pub fn sum_squeeze_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
    Ok(sum_squeeze_1d(a.as_slice()?))
}

#[pyfunction]
pub fn any_squeeze_rs(a: PyReadonlyArray1<'_, f64>) -> PyResult<bool> {
    Ok(any_squeeze_1d(a.as_slice()?))
}

#[pyfunction]
pub fn find_ranges_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    gap_value: f64,
) -> PyResult<Bound<'py, PyArray1<RangeRecord>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| find_ranges(a_arr, gap_value));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn range_duration_rs<'py>(
    py: Python<'py>,
    start_idx: PyReadonlyArray1<'py, i64>,
    end_idx: PyReadonlyArray1<'py, i64>,
    status: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let start_idx_slice = start_idx.as_slice()?;
    let end_idx_slice = end_idx.as_slice()?;
    let status_slice = status.as_slice()?;
    let result = py.allow_threads(|| range_duration(start_idx_slice, end_idx_slice, status_slice));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn range_coverage_rs<'py>(
    py: Python<'py>,
    start_idx: PyReadonlyArray1<'py, i64>,
    end_idx: PyReadonlyArray1<'py, i64>,
    status: PyReadonlyArray1<'py, i64>,
    col_map: (PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>),
    index_lens: PyReadonlyArray1<'py, i64>,
    overlapping: bool,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let start_idx_slice = start_idx.as_slice()?;
    let end_idx_slice = end_idx.as_slice()?;
    let status_slice = status.as_slice()?;
    let col_idxs_slice = col_map.0.as_slice()?;
    let col_lens_slice = col_map.1.as_slice()?;
    let index_lens_slice = index_lens.as_slice()?;
    let result = py.allow_threads(|| {
        range_coverage(
            start_idx_slice,
            end_idx_slice,
            status_slice,
            col_idxs_slice,
            col_lens_slice,
            index_lens_slice,
            overlapping,
            normalize,
        )
    })?;
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn ranges_to_mask_rs<'py>(
    py: Python<'py>,
    start_idx: PyReadonlyArray1<'py, i64>,
    end_idx: PyReadonlyArray1<'py, i64>,
    status: PyReadonlyArray1<'py, i64>,
    col_map: (PyReadonlyArray1<'py, i64>, PyReadonlyArray1<'py, i64>),
    index_len: usize,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let start_idx_slice = start_idx.as_slice()?;
    let end_idx_slice = end_idx.as_slice()?;
    let status_slice = status.as_slice()?;
    let col_idxs_slice = col_map.0.as_slice()?;
    let col_lens_slice = col_map.1.as_slice()?;
    let result = py.allow_threads(|| {
        ranges_to_mask(
            start_idx_slice,
            end_idx_slice,
            status_slice,
            col_idxs_slice,
            col_lens_slice,
            index_len,
        )
    })?;
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn get_drawdowns_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<DrawdownRecord>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| get_drawdowns(a_arr));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn dd_drawdown_rs<'py>(
    py: Python<'py>,
    peak_val: PyReadonlyArray1<'py, f64>,
    valley_val: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let peak_val_slice = peak_val.as_slice()?;
    let valley_val_slice = valley_val.as_slice()?;
    let out_len = broadcast_len2(peak_val_slice.len(), valley_val_slice.len())?;
    let result = py.allow_threads(|| {
        (0..out_len)
            .map(|i| {
                let peak = broadcast_get(peak_val_slice, i);
                let valley = broadcast_get(valley_val_slice, i);
                (valley - peak) / peak
            })
            .collect::<Vec<f64>>()
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn dd_decline_duration_rs<'py>(
    py: Python<'py>,
    start_idx: PyReadonlyArray1<'py, i64>,
    valley_idx: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let start_idx_slice = start_idx.as_slice()?;
    let valley_idx_slice = valley_idx.as_slice()?;
    let out_len = broadcast_len2(start_idx_slice.len(), valley_idx_slice.len())?;
    let result = py.allow_threads(|| {
        (0..out_len)
            .map(|i| {
                let start = broadcast_get(start_idx_slice, i);
                let valley = broadcast_get(valley_idx_slice, i);
                valley - start + 1
            })
            .collect::<Vec<i64>>()
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn dd_recovery_duration_rs<'py>(
    py: Python<'py>,
    valley_idx: PyReadonlyArray1<'py, i64>,
    end_idx: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let valley_idx_slice = valley_idx.as_slice()?;
    let end_idx_slice = end_idx.as_slice()?;
    let out_len = broadcast_len2(valley_idx_slice.len(), end_idx_slice.len())?;
    let result = py.allow_threads(|| {
        (0..out_len)
            .map(|i| {
                let valley = broadcast_get(valley_idx_slice, i);
                let end = broadcast_get(end_idx_slice, i);
                end - valley
            })
            .collect::<Vec<i64>>()
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn dd_recovery_duration_ratio_rs<'py>(
    py: Python<'py>,
    start_idx: PyReadonlyArray1<'py, i64>,
    valley_idx: PyReadonlyArray1<'py, i64>,
    end_idx: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let start_idx_slice = start_idx.as_slice()?;
    let valley_idx_slice = valley_idx.as_slice()?;
    let end_idx_slice = end_idx.as_slice()?;
    let out_len = broadcast_len3(
        start_idx_slice.len(),
        valley_idx_slice.len(),
        end_idx_slice.len(),
    )?;
    let result = py.allow_threads(|| {
        (0..out_len)
            .map(|i| {
                let start = broadcast_get(start_idx_slice, i);
                let valley = broadcast_get(valley_idx_slice, i);
                let end = broadcast_get(end_idx_slice, i);
                (end - valley) as f64 / (valley - start + 1) as f64
            })
            .collect::<Vec<f64>>()
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn dd_recovery_return_rs<'py>(
    py: Python<'py>,
    valley_val: PyReadonlyArray1<'py, f64>,
    end_val: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let valley_val_slice = valley_val.as_slice()?;
    let end_val_slice = end_val.as_slice()?;
    let out_len = broadcast_len2(valley_val_slice.len(), end_val_slice.len())?;
    let result = py.allow_threads(|| {
        (0..out_len)
            .map(|i| {
                let valley = broadcast_get(valley_val_slice, i);
                let end = broadcast_get(end_val_slice, i);
                (end - valley) / valley
            })
            .collect::<Vec<f64>>()
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn crossed_above_1d_rs<'py>(
    py: Python<'py>,
    arr1: PyReadonlyArray1<'py, f64>,
    arr2: PyReadonlyArray1<'py, f64>,
    wait: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let arr1_slice = arr1.as_slice()?;
    let arr2_slice = arr2.as_slice()?;
    let result = py.allow_threads(|| crossed_above_1d(arr1_slice, arr2_slice, wait));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn crossed_above_rs<'py>(
    py: Python<'py>,
    arr1: PyReadonlyArray2<'py, f64>,
    arr2: PyReadonlyArray2<'py, f64>,
    wait: usize,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let arr1_arr = arr1.as_array();
    let arr2_arr = arr2.as_array();
    let (nrows, ncols) = arr1_arr.dim();
    let result = py.allow_threads(|| {
        let mut out = Array2::<bool>::from_elem((nrows, ncols), false);
        for col in 0..ncols {
            let col1 = arr1_arr.column(col);
            let col2 = arr2_arr.column(col);
            let mut was_below = false;
            let mut crossed_ago = -1i64;
            for i in 0..nrows {
                let v1 = col1[i];
                let v2 = col2[i];
                if v1.is_nan() || v2.is_nan() {
                    crossed_ago = -1;
                    was_below = false;
                } else if v1 > v2 {
                    if was_below {
                        crossed_ago += 1;
                        out[[i, col]] = crossed_ago == wait as i64;
                    }
                } else if v1 < v2 {
                    crossed_ago = -1;
                    was_below = true;
                } else {
                    crossed_ago = -1;
                }
            }
        }
        out
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(set_by_mask_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(set_by_mask_rs, m)?)?;
    m.add_function(wrap_pyfunction!(set_by_mask_mult_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(set_by_mask_mult_rs, m)?)?;
    m.add_function(wrap_pyfunction!(fillna_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(fillna_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bshift_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bshift_rs, m)?)?;
    m.add_function(wrap_pyfunction!(fshift_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(fshift_rs, m)?)?;
    m.add_function(wrap_pyfunction!(diff_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(diff_rs, m)?)?;
    m.add_function(wrap_pyfunction!(pct_change_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(pct_change_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bfill_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bfill_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ffill_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ffill_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nanprod_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nancumsum_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nancumprod_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nansum_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nancnt_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nanmin_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nanmax_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nanmean_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nanmedian_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nanstd_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nanstd_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_min_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_min_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_max_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_max_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ewm_mean_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ewm_mean_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ewm_std_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ewm_std_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_min_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_min_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_max_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_max_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_mean_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_mean_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_std_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expanding_std_rs, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_rs, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_forder_rs, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_grouped_rs, m)?)?;
    m.add_function(wrap_pyfunction!(flatten_uniform_grouped_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nth_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nth_index_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(min_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(max_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(mean_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(median_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(std_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sum_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(count_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(argmin_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(argmax_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(describe_reduce_rs, m)?)?;
    m.add_function(wrap_pyfunction!(value_counts_rs, m)?)?;
    m.add_function(wrap_pyfunction!(min_squeeze_rs, m)?)?;
    m.add_function(wrap_pyfunction!(max_squeeze_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sum_squeeze_rs, m)?)?;
    m.add_function(wrap_pyfunction!(any_squeeze_rs, m)?)?;
    m.add_function(wrap_pyfunction!(find_ranges_rs, m)?)?;
    m.add_function(wrap_pyfunction!(range_duration_rs, m)?)?;
    m.add_function(wrap_pyfunction!(range_coverage_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ranges_to_mask_rs, m)?)?;
    m.add_function(wrap_pyfunction!(get_drawdowns_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dd_drawdown_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dd_decline_duration_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dd_recovery_duration_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dd_recovery_duration_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(dd_recovery_return_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crossed_above_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(crossed_above_rs, m)?)?;
    Ok(())
}
