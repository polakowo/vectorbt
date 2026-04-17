// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use crate::generic::{
    apply_2d_by_col, bshift_1d_into, ewm_mean_1d, ewm_std_1d, rolling_max_1d, rolling_mean_1d,
    rolling_min_1d, rolling_std_1d,
};
use ndarray::{Array2, ArrayView2};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const TREND_MODE_BINARY: i64 = 0;
const TREND_MODE_BINARY_CONT: i64 = 1;
const TREND_MODE_BINARY_CONT_SAT: i64 = 2;
const TREND_MODE_PCT_CHANGE: i64 = 3;
const TREND_MODE_PCT_CHANGE_NORM: i64 = 4;

fn reverse_vec(mut v: Vec<f64>) -> Vec<f64> {
    v.reverse();
    v
}

fn validate_matching_shape(
    name: &str,
    shape: (usize, usize),
    other: (usize, usize),
) -> PyResult<()> {
    if shape != other {
        return Err(PyValueError::new_err(format!(
            "`{name}` must have the same shape as `close`"
        )));
    }
    Ok(())
}

fn future_mean_apply_rolling_c(
    close: ArrayView2<'_, f64>,
    window: usize,
    wait: usize,
) -> Option<Array2<f64>> {
    let (nrows, ncols) = close.dim();
    let src = close.as_slice()?;
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let dst = out.as_slice_mut().expect("owned array must be sliceable");
    let mut sums = vec![0.0f64; ncols];
    let mut counts = vec![0usize; ncols];

    for i in (0..nrows).rev() {
        let start = i + wait;
        if start < nrows {
            let row_start = start * ncols;
            for col in 0..ncols {
                let v = src[row_start + col];
                if !v.is_nan() {
                    sums[col] += v;
                    counts[col] += 1;
                }
            }
        }
        let end = start + window;
        if end < nrows {
            let row_start = end * ncols;
            for col in 0..ncols {
                let v = src[row_start + col];
                if !v.is_nan() {
                    sums[col] -= v;
                    counts[col] -= 1;
                }
            }
        }
        let out_start = i * ncols;
        for col in 0..ncols {
            if counts[col] >= window {
                dst[out_start + col] = sums[col] / counts[col] as f64;
            }
        }
    }
    Some(out)
}

fn future_std_apply_rolling_c(
    close: ArrayView2<'_, f64>,
    window: usize,
    wait: usize,
    ddof: usize,
) -> Option<Array2<f64>> {
    let (nrows, ncols) = close.dim();
    let src = close.as_slice()?;
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let dst = out.as_slice_mut().expect("owned array must be sliceable");
    let mut sums = vec![0.0f64; ncols];
    let mut sums_sq = vec![0.0f64; ncols];
    let mut counts = vec![0usize; ncols];

    for i in (0..nrows).rev() {
        let start = i + wait;
        if start < nrows {
            let row_start = start * ncols;
            for col in 0..ncols {
                let v = src[row_start + col];
                if !v.is_nan() {
                    sums[col] += v;
                    sums_sq[col] += v * v;
                    counts[col] += 1;
                }
            }
        }
        let end = start + window;
        if end < nrows {
            let row_start = end * ncols;
            for col in 0..ncols {
                let v = src[row_start + col];
                if !v.is_nan() {
                    sums[col] -= v;
                    sums_sq[col] -= v * v;
                    counts[col] -= 1;
                }
            }
        }
        let out_start = i * ncols;
        for col in 0..ncols {
            let cnt = counts[col];
            if cnt >= window && cnt > ddof {
                let mean = sums[col] / cnt as f64;
                let variance = (sums_sq[col] - 2.0 * sums[col] * mean + cnt as f64 * mean * mean)
                    / (cnt - ddof) as f64;
                dst[out_start + col] = variance.abs().sqrt();
            }
        }
    }
    Some(out)
}

fn mean_labels_apply_rolling_c(
    close: ArrayView2<'_, f64>,
    window: usize,
    wait: usize,
) -> Option<Array2<f64>> {
    let (nrows, ncols) = close.dim();
    let src = close.as_slice()?;
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let dst = out.as_slice_mut().expect("owned array must be sliceable");
    let mut sums = vec![0.0f64; ncols];
    let mut counts = vec![0usize; ncols];

    for i in (0..nrows).rev() {
        let start = i + wait;
        if start < nrows {
            let row_start = start * ncols;
            for col in 0..ncols {
                let v = src[row_start + col];
                if !v.is_nan() {
                    sums[col] += v;
                    counts[col] += 1;
                }
            }
        }
        let end = start + window;
        if end < nrows {
            let row_start = end * ncols;
            for col in 0..ncols {
                let v = src[row_start + col];
                if !v.is_nan() {
                    sums[col] -= v;
                    counts[col] -= 1;
                }
            }
        }
        let out_start = i * ncols;
        for col in 0..ncols {
            if counts[col] >= window {
                let cur = src[out_start + col];
                dst[out_start + col] = (sums[col] / counts[col] as f64 - cur) / cur;
            }
        }
    }
    Some(out)
}

fn future_min_max_apply_rolling_c(
    close: ArrayView2<'_, f64>,
    window: usize,
    wait: usize,
    is_min: bool,
) -> Option<Array2<f64>> {
    let (nrows, ncols) = close.dim();
    let src = close.as_slice()?;
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let dst = out.as_slice_mut().expect("owned array must be sliceable");

    for i in 0..nrows {
        let start = i + wait;
        let end = start + window;
        if end > nrows {
            continue;
        }
        let out_start = i * ncols;
        for col in 0..ncols {
            let mut value = if is_min {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            let mut valid = true;
            for row in start..end {
                let v = src[row * ncols + col];
                if v.is_nan() {
                    valid = false;
                    break;
                }
                if is_min {
                    if v < value {
                        value = v;
                    }
                } else if v > value {
                    value = v;
                }
            }
            if valid {
                dst[out_start + col] = value;
            }
        }
    }
    Some(out)
}

pub(crate) fn future_mean_apply(
    close: ArrayView2<'_, f64>,
    window: usize,
    ewm: bool,
    wait: usize,
    adjust: bool,
) -> Array2<f64> {
    if !ewm {
        if let Some(out) = future_mean_apply_rolling_c(close, window, wait) {
            return out;
        }
    }
    let base = apply_2d_by_col(close, |col| {
        let mut rev: Vec<f64> = col.iter().rev().copied().collect();
        let out = if ewm {
            ewm_mean_1d(&rev, window, window, adjust)
        } else {
            rolling_mean_1d(&rev, window, window)
        };
        rev = reverse_vec(out);
        rev
    });
    if wait > 0 {
        apply_2d_by_col(base.view(), |col| {
            let mut out = vec![f64::NAN; col.len()];
            bshift_1d_into(col, &mut out, wait, f64::NAN);
            out
        })
    } else {
        base
    }
}

pub(crate) fn future_std_apply(
    close: ArrayView2<'_, f64>,
    window: usize,
    ewm: bool,
    wait: usize,
    adjust: bool,
    ddof: usize,
) -> Array2<f64> {
    if !ewm {
        if let Some(out) = future_std_apply_rolling_c(close, window, wait, ddof) {
            return out;
        }
    }
    let base = apply_2d_by_col(close, |col| {
        let rev: Vec<f64> = col.iter().rev().copied().collect();
        let out = if ewm {
            ewm_std_1d(&rev, window, window, adjust, ddof)
        } else {
            rolling_std_1d(&rev, window, window, ddof)
        };
        reverse_vec(out)
    });
    if wait > 0 {
        apply_2d_by_col(base.view(), |col| {
            let mut out = vec![f64::NAN; col.len()];
            bshift_1d_into(col, &mut out, wait, f64::NAN);
            out
        })
    } else {
        base
    }
}

pub(crate) fn future_min_apply(
    close: ArrayView2<'_, f64>,
    window: usize,
    wait: usize,
) -> Array2<f64> {
    if let Some(out) = future_min_max_apply_rolling_c(close, window, wait, true) {
        return out;
    }
    let base = apply_2d_by_col(close, |col| {
        let rev: Vec<f64> = col.iter().rev().copied().collect();
        let out = rolling_min_1d(&rev, window, window);
        reverse_vec(out)
    });
    if wait > 0 {
        apply_2d_by_col(base.view(), |col| {
            let mut out = vec![f64::NAN; col.len()];
            bshift_1d_into(col, &mut out, wait, f64::NAN);
            out
        })
    } else {
        base
    }
}

pub(crate) fn future_max_apply(
    close: ArrayView2<'_, f64>,
    window: usize,
    wait: usize,
) -> Array2<f64> {
    if let Some(out) = future_min_max_apply_rolling_c(close, window, wait, false) {
        return out;
    }
    let base = apply_2d_by_col(close, |col| {
        let rev: Vec<f64> = col.iter().rev().copied().collect();
        let out = rolling_max_1d(&rev, window, window);
        reverse_vec(out)
    });
    if wait > 0 {
        apply_2d_by_col(base.view(), |col| {
            let mut out = vec![f64::NAN; col.len()];
            bshift_1d_into(col, &mut out, wait, f64::NAN);
            out
        })
    } else {
        base
    }
}

pub(crate) fn fixed_labels_apply(close: ArrayView2<'_, f64>, n: usize) -> Array2<f64> {
    let (nrows, ncols) = close.dim();
    if let Some(src) = close.as_slice() {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        let dst = out.as_slice_mut().expect("owned array must be sliceable");
        for i in 0..nrows {
            let row_start = i * ncols;
            if i + n < nrows {
                let future_start = (i + n) * ncols;
                for col in 0..ncols {
                    let cur = src[row_start + col];
                    dst[row_start + col] = (src[future_start + col] - cur) / cur;
                }
            } else {
                for col in 0..ncols {
                    let cur = src[row_start + col];
                    dst[row_start + col] = (f64::NAN - cur) / cur;
                }
            }
        }
        return out;
    }
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    for col in 0..ncols {
        for i in 0..nrows {
            let cur = close[[i, col]];
            let future = if i + n < nrows {
                close[[i + n, col]]
            } else {
                f64::NAN
            };
            out[[i, col]] = (future - cur) / cur;
        }
    }
    out
}

pub(crate) fn mean_labels_apply(
    close: ArrayView2<'_, f64>,
    window: usize,
    ewm: bool,
    wait: usize,
    adjust: bool,
) -> Array2<f64> {
    if !ewm {
        if let Some(out) = mean_labels_apply_rolling_c(close, window, wait) {
            return out;
        }
    }
    let future = future_mean_apply(close, window, ewm, wait, adjust);
    let (nrows, ncols) = close.dim();
    if let (Some(close_src), Some(future_src)) = (close.as_slice(), future.as_slice()) {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        let dst = out.as_slice_mut().expect("owned array must be sliceable");
        for i in 0..close_src.len() {
            let cur = close_src[i];
            dst[i] = (future_src[i] - cur) / cur;
        }
        return out;
    }
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    for col in 0..ncols {
        for i in 0..nrows {
            let cur = close[[i, col]];
            out[[i, col]] = (future[[i, col]] - cur) / cur;
        }
    }
    out
}

pub(crate) fn local_extrema_apply(
    close: ArrayView2<'_, f64>,
    pos_th: ArrayView2<'_, f64>,
    neg_th: ArrayView2<'_, f64>,
) -> PyResult<Array2<i64>> {
    let (nrows, ncols) = close.dim();
    let mut out = Array2::<i64>::from_elem((nrows, ncols), 0);

    for col in 0..ncols {
        let mut prev_i: usize = 0;
        let mut direction: i64 = 0;

        for i in 1..nrows {
            let _pos_th = pos_th[[prev_i, col]].abs();
            let _neg_th = neg_th[[prev_i, col]].abs();
            if _pos_th == 0.0 {
                return Err(PyValueError::new_err("Positive threshold cannot be 0"));
            }
            if _neg_th == 0.0 {
                return Err(PyValueError::new_err("Negative threshold cannot be 0"));
            }

            if direction == 1 {
                if close[[i, col]] < close[[prev_i, col]] {
                    prev_i = i;
                } else if close[[i, col]] >= close[[prev_i, col]] * (1.0 + _pos_th) {
                    out[[prev_i, col]] = -1;
                    prev_i = i;
                    direction = -1;
                }
            } else if direction == -1 {
                if close[[i, col]] > close[[prev_i, col]] {
                    prev_i = i;
                } else if close[[i, col]] <= close[[prev_i, col]] * (1.0 - _neg_th) {
                    out[[prev_i, col]] = 1;
                    prev_i = i;
                    direction = 1;
                }
            } else {
                if close[[i, col]] >= close[[prev_i, col]] * (1.0 + _pos_th) {
                    out[[prev_i, col]] = -1;
                    prev_i = i;
                    direction = -1;
                } else if close[[i, col]] <= close[[prev_i, col]] * (1.0 - _neg_th) {
                    out[[prev_i, col]] = 1;
                    prev_i = i;
                    direction = 1;
                }
            }

            if i == nrows - 1 && direction != 0 {
                out[[prev_i, col]] = -direction;
            }
        }
    }
    Ok(out)
}

pub(crate) fn bn_trend_labels(
    close: ArrayView2<'_, f64>,
    local_extrema: ArrayView2<'_, i64>,
) -> Array2<f64> {
    let (nrows, ncols) = close.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);

    for col in 0..ncols {
        let mut prev_i_opt: Option<usize> = None;
        for next_i in 0..nrows {
            if local_extrema[[next_i, col]] == 0 {
                continue;
            }
            let Some(prev_i) = prev_i_opt else {
                prev_i_opt = Some(next_i);
                continue;
            };
            let fill = if close[[next_i, col]] > close[[prev_i, col]] {
                1.0
            } else {
                0.0
            };
            for i in prev_i..next_i {
                out[[i, col]] = fill;
            }
            prev_i_opt = Some(next_i);
        }
    }
    out
}

fn slice_min_max(close: ArrayView2<'_, f64>, col: usize, start: usize, end: usize) -> (f64, f64) {
    let mut _min = f64::INFINITY;
    let mut _max = f64::NEG_INFINITY;
    for i in start..=end {
        let v = close[[i, col]];
        if v.is_nan() {
            return (f64::NAN, f64::NAN);
        }
        if v < _min {
            _min = v;
        }
        if v > _max {
            _max = v;
        }
    }
    (_min, _max)
}

pub(crate) fn bn_cont_trend_labels(
    close: ArrayView2<'_, f64>,
    local_extrema: ArrayView2<'_, i64>,
) -> Array2<f64> {
    let (nrows, ncols) = close.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);

    for col in 0..ncols {
        let mut prev_i_opt: Option<usize> = None;
        for next_i in 0..nrows {
            if local_extrema[[next_i, col]] == 0 {
                continue;
            }
            let Some(prev_i) = prev_i_opt else {
                prev_i_opt = Some(next_i);
                continue;
            };
            let (_min, _max) = slice_min_max(close, col, prev_i, next_i);
            let range = _max - _min;
            for i in prev_i..next_i {
                out[[i, col]] = 1.0 - (close[[i, col]] - _min) / range;
            }
            prev_i_opt = Some(next_i);
        }
    }
    out
}

pub(crate) fn bn_cont_sat_trend_labels(
    close: ArrayView2<'_, f64>,
    local_extrema: ArrayView2<'_, i64>,
    pos_th: ArrayView2<'_, f64>,
    neg_th: ArrayView2<'_, f64>,
) -> PyResult<Array2<f64>> {
    let (nrows, ncols) = close.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);

    for col in 0..ncols {
        let mut prev_i_opt: Option<usize> = None;
        for next_i in 0..nrows {
            if local_extrema[[next_i, col]] == 0 {
                continue;
            }
            let Some(prev_i) = prev_i_opt else {
                prev_i_opt = Some(next_i);
                continue;
            };

            let _pos_th = pos_th[[prev_i, col]].abs();
            let _neg_th = neg_th[[prev_i, col]].abs();
            if _pos_th == 0.0 {
                return Err(PyValueError::new_err("Positive threshold cannot be 0"));
            }
            if _neg_th == 0.0 {
                return Err(PyValueError::new_err("Negative threshold cannot be 0"));
            }

            let (_min, _max) = slice_min_max(close, col, prev_i, next_i);

            let going_up = close[[next_i, col]] > close[[prev_i, col]];
            for i in prev_i..next_i {
                let c = close[[i, col]];
                if going_up {
                    let _start = _max / (1.0 + _pos_th);
                    let _end = _min * (1.0 + _pos_th);
                    if _max >= _end && c <= _start {
                        out[[i, col]] = 1.0;
                    } else {
                        out[[i, col]] = 1.0 - (c - _start) / (_max - _start);
                    }
                } else {
                    let _start = _min / (1.0 - _neg_th);
                    let _end = _max * (1.0 - _neg_th);
                    if _min <= _end && c >= _start {
                        out[[i, col]] = 0.0;
                    } else {
                        out[[i, col]] = 1.0 - (c - _min) / (_start - _min);
                    }
                }
            }
            prev_i_opt = Some(next_i);
        }
    }
    Ok(out)
}

pub(crate) fn pct_trend_labels(
    close: ArrayView2<'_, f64>,
    local_extrema: ArrayView2<'_, i64>,
    normalize: bool,
) -> Array2<f64> {
    let (nrows, ncols) = close.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);

    for col in 0..ncols {
        let mut prev_i_opt: Option<usize> = None;
        for next_i in 0..nrows {
            if local_extrema[[next_i, col]] == 0 {
                continue;
            }
            let Some(prev_i) = prev_i_opt else {
                prev_i_opt = Some(next_i);
                continue;
            };
            let going_up = close[[next_i, col]] > close[[prev_i, col]];
            for i in prev_i..next_i {
                let c = close[[i, col]];
                let next_c = close[[next_i, col]];
                out[[i, col]] = if going_up && normalize {
                    (next_c - c) / next_c
                } else {
                    (next_c - c) / c
                };
            }
            prev_i_opt = Some(next_i);
        }
    }
    out
}

pub(crate) fn trend_labels_apply(
    close: ArrayView2<'_, f64>,
    pos_th: ArrayView2<'_, f64>,
    neg_th: ArrayView2<'_, f64>,
    mode: i64,
) -> PyResult<Array2<f64>> {
    let local_extrema = local_extrema_apply(close, pos_th, neg_th)?;
    let out = match mode {
        TREND_MODE_BINARY => bn_trend_labels(close, local_extrema.view()),
        TREND_MODE_BINARY_CONT => bn_cont_trend_labels(close, local_extrema.view()),
        TREND_MODE_BINARY_CONT_SAT => {
            bn_cont_sat_trend_labels(close, local_extrema.view(), pos_th, neg_th)?
        }
        TREND_MODE_PCT_CHANGE => pct_trend_labels(close, local_extrema.view(), false),
        TREND_MODE_PCT_CHANGE_NORM => pct_trend_labels(close, local_extrema.view(), true),
        _ => return Err(PyValueError::new_err("Trend mode is not recognized")),
    };
    Ok(out)
}

pub(crate) fn breakout_labels(
    close: ArrayView2<'_, f64>,
    window: usize,
    pos_th: ArrayView2<'_, f64>,
    neg_th: ArrayView2<'_, f64>,
    wait: usize,
) -> Array2<f64> {
    let (nrows, ncols) = close.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), 0.0);

    for col in 0..ncols {
        for i in 0..nrows {
            let _pos_th = pos_th[[i, col]].abs();
            let _neg_th = neg_th[[i, col]].abs();
            let start = i + wait;
            let end = (i + window + wait).min(nrows);
            if start >= end {
                continue;
            }
            for j in start..end {
                if _pos_th > 0.0 && close[[j, col]] >= close[[i, col]] * (1.0 + _pos_th) {
                    out[[i, col]] = 1.0;
                    break;
                }
                if _neg_th > 0.0 && close[[j, col]] <= close[[i, col]] * (1.0 - _neg_th) {
                    out[[i, col]] = -1.0;
                    break;
                }
            }
        }
    }
    out
}

#[pyfunction]
#[pyo3(signature = (close, window, ewm, wait=1, adjust=false))]
pub fn future_mean_apply_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    wait: usize,
    adjust: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let result = py.allow_threads(|| future_mean_apply(close_arr, window, ewm, wait, adjust));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (close, window, ewm, wait=1, adjust=false, ddof=0))]
pub fn future_std_apply_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    wait: usize,
    adjust: bool,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let result = py.allow_threads(|| future_std_apply(close_arr, window, ewm, wait, adjust, ddof));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (close, window, wait=1))]
pub fn future_min_apply_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    window: usize,
    wait: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let result = py.allow_threads(|| future_min_apply(close_arr, window, wait));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (close, window, wait=1))]
pub fn future_max_apply_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    window: usize,
    wait: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let result = py.allow_threads(|| future_max_apply(close_arr, window, wait));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn fixed_labels_apply_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let result = py.allow_threads(|| fixed_labels_apply(close_arr, n));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (close, window, ewm, wait=1, adjust=false))]
pub fn mean_labels_apply_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    wait: usize,
    adjust: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let result = py.allow_threads(|| mean_labels_apply(close_arr, window, ewm, wait, adjust));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn local_extrema_apply_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    pos_th: PyReadonlyArray2<'py, f64>,
    neg_th: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let close_arr = close.as_array();
    let pos_arr = pos_th.as_array();
    let neg_arr = neg_th.as_array();
    validate_matching_shape("pos_th", close_arr.dim(), pos_arr.dim())?;
    validate_matching_shape("neg_th", close_arr.dim(), neg_arr.dim())?;
    let result = py.allow_threads(|| local_extrema_apply(close_arr, pos_arr, neg_arr));
    Ok(PyArray2::from_owned_array_bound(py, result?))
}

#[pyfunction]
pub fn bn_trend_labels_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    local_extrema: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let le_arr = local_extrema.as_array();
    validate_matching_shape("local_extrema", close_arr.dim(), le_arr.dim())?;
    let result = py.allow_threads(|| bn_trend_labels(close_arr, le_arr));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn bn_cont_trend_labels_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    local_extrema: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let le_arr = local_extrema.as_array();
    validate_matching_shape("local_extrema", close_arr.dim(), le_arr.dim())?;
    let result = py.allow_threads(|| bn_cont_trend_labels(close_arr, le_arr));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn bn_cont_sat_trend_labels_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    local_extrema: PyReadonlyArray2<'py, i64>,
    pos_th: PyReadonlyArray2<'py, f64>,
    neg_th: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let le_arr = local_extrema.as_array();
    let pos_arr = pos_th.as_array();
    let neg_arr = neg_th.as_array();
    validate_matching_shape("local_extrema", close_arr.dim(), le_arr.dim())?;
    validate_matching_shape("pos_th", close_arr.dim(), pos_arr.dim())?;
    validate_matching_shape("neg_th", close_arr.dim(), neg_arr.dim())?;
    let result = py.allow_threads(|| bn_cont_sat_trend_labels(close_arr, le_arr, pos_arr, neg_arr));
    Ok(PyArray2::from_owned_array_bound(py, result?))
}

#[pyfunction]
pub fn pct_trend_labels_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    local_extrema: PyReadonlyArray2<'py, i64>,
    normalize: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let le_arr = local_extrema.as_array();
    validate_matching_shape("local_extrema", close_arr.dim(), le_arr.dim())?;
    let result = py.allow_threads(|| pct_trend_labels(close_arr, le_arr, normalize));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn trend_labels_apply_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    pos_th: PyReadonlyArray2<'py, f64>,
    neg_th: PyReadonlyArray2<'py, f64>,
    mode: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let pos_arr = pos_th.as_array();
    let neg_arr = neg_th.as_array();
    validate_matching_shape("pos_th", close_arr.dim(), pos_arr.dim())?;
    validate_matching_shape("neg_th", close_arr.dim(), neg_arr.dim())?;
    let result = py.allow_threads(|| trend_labels_apply(close_arr, pos_arr, neg_arr, mode));
    Ok(PyArray2::from_owned_array_bound(py, result?))
}

#[pyfunction]
#[pyo3(signature = (close, window, pos_th, neg_th, wait=1))]
pub fn breakout_labels_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    window: usize,
    pos_th: PyReadonlyArray2<'py, f64>,
    neg_th: PyReadonlyArray2<'py, f64>,
    wait: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let close_arr = close.as_array();
    let pos_arr = pos_th.as_array();
    let neg_arr = neg_th.as_array();
    validate_matching_shape("pos_th", close_arr.dim(), pos_arr.dim())?;
    validate_matching_shape("neg_th", close_arr.dim(), neg_arr.dim())?;
    let result = py.allow_threads(|| breakout_labels(close_arr, window, pos_arr, neg_arr, wait));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(future_mean_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(future_std_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(future_min_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(future_max_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(fixed_labels_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(mean_labels_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(local_extrema_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bn_trend_labels_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bn_cont_trend_labels_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bn_cont_sat_trend_labels_rs, m)?)?;
    m.add_function(wrap_pyfunction!(pct_trend_labels_rs, m)?)?;
    m.add_function(wrap_pyfunction!(trend_labels_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(breakout_labels_rs, m)?)?;
    Ok(())
}
