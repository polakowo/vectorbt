// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Validate rolling parameters, matching the numba contract.
fn validate_rolling(minp: usize, window: usize) -> PyResult<()> {
    if minp > window {
        return Err(PyValueError::new_err("minp must be <= window"));
    }
    Ok(())
}

// ======================== 1D kernels ========================

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

        if window_len < minp || window_len == ddof {
            // leave as NAN
        } else {
            let mean = window_cumsum / window_len as f64;
            let variance = (window_cumsum_sq - 2.0 * window_cumsum * mean
                + window_len as f64 * mean * mean)
                / (window_len - ddof) as f64;
            out[i] = variance.abs().sqrt();
        }
    }
    out
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

fn diff_1d(a: &[f64], n: usize) -> Vec<f64> {
    let len = a.len();
    let mut out = vec![f64::NAN; len];
    for i in n..len {
        out[i] = a[i] - a[i - n];
    }
    out
}

// ======================== PyO3 exports: 1D ========================

#[pyfunction]
#[pyo3(signature = (a, window, minp=None))]
pub fn rolling_mean_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let minp = minp.unwrap_or(window);
    validate_rolling(minp, window)?;
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
    let a_arr = a.as_array();
    let (nrows, ncols) = a_arr.dim();
    let minp = minp.unwrap_or(window);
    validate_rolling(minp, window)?;
    let result = py.allow_threads(|| {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        for col in 0..ncols {
            let col_slice: Vec<f64> = (0..nrows).map(|r| a_arr[[r, col]]).collect();
            let col_result = rolling_mean_1d(&col_slice, window, minp);
            for r in 0..nrows {
                out[[r, col]] = col_result[r];
            }
        }
        out
    });
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
    let a_slice = a.as_slice()?;
    let minp = minp.unwrap_or(window);
    validate_rolling(minp, window)?;
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
    let a_arr = a.as_array();
    let (nrows, ncols) = a_arr.dim();
    let minp = minp.unwrap_or(window);
    validate_rolling(minp, window)?;
    let result = py.allow_threads(|| {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        for col in 0..ncols {
            let col_slice: Vec<f64> = (0..nrows).map(|r| a_arr[[r, col]]).collect();
            let col_result = rolling_std_1d(&col_slice, window, minp, ddof);
            for r in 0..nrows {
                out[[r, col]] = col_result[r];
            }
        }
        out
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, minp=None))]
pub fn rolling_min_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let minp = minp.unwrap_or(window);
    validate_rolling(minp, window)?;
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
    let a_arr = a.as_array();
    let (nrows, ncols) = a_arr.dim();
    let minp = minp.unwrap_or(window);
    validate_rolling(minp, window)?;
    let result = py.allow_threads(|| {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        for col in 0..ncols {
            let col_slice: Vec<f64> = (0..nrows).map(|r| a_arr[[r, col]]).collect();
            let col_result = rolling_min_1d(&col_slice, window, minp);
            for r in 0..nrows {
                out[[r, col]] = col_result[r];
            }
        }
        out
    });
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
    let a_slice = a.as_slice()?;
    let minp = minp.unwrap_or(window);
    validate_rolling(minp, window)?;
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
    let a_arr = a.as_array();
    let (nrows, ncols) = a_arr.dim();
    let minp = minp.unwrap_or(window);
    validate_rolling(minp, window)?;
    let result = py.allow_threads(|| {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        for col in 0..ncols {
            let col_slice: Vec<f64> = (0..nrows).map(|r| a_arr[[r, col]]).collect();
            let col_result = rolling_max_1d(&col_slice, window, minp);
            for r in 0..nrows {
                out[[r, col]] = col_result[r];
            }
        }
        out
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, n=1))]
pub fn diff_1d_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_slice = a.as_slice()?;
    let result = py.allow_threads(|| diff_1d(a_slice, n));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, n=1))]
pub fn diff_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let (nrows, ncols) = a_arr.dim();
    let result = py.allow_threads(|| {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        for col in 0..ncols {
            let col_slice: Vec<f64> = (0..nrows).map(|r| a_arr[[r, col]]).collect();
            let col_result = diff_1d(&col_slice, n);
            for r in 0..nrows {
                out[[r, col]] = col_result[r];
            }
        }
        out
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

// ======================== Registration ========================

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rolling_mean_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_mean_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_std_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_min_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_min_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_max_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_max_rs, m)?)?;
    m.add_function(wrap_pyfunction!(diff_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(diff_rs, m)?)?;
    Ok(())
}
