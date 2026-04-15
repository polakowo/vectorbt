// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use crate::generic::{
    apply_2d_by_col, apply_2d_by_col_inplace, array1_as_slice_cow, diff_1d_into, ewm_mean_1d,
    ewm_mean_2d_c, ewm_std_1d, ewm_std_2d_c, nancumsum_1d_into, rolling_max_1d, rolling_mean_1d,
    rolling_min_1d, rolling_std_1d,
};
use ndarray::{Array2, ArrayView2, Zip};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

pub(crate) fn tuple_hash(py: Python<'_>, window: usize, ewm: bool) -> PyResult<isize> {
    let key = PyTuple::new_bound(py, [window.into_py(py), ewm.into_py(py)]);
    key.hash()
}

pub(crate) fn validate_param_lengths(names: &str, expected: usize, actual: usize) -> PyResult<()> {
    if expected != actual {
        return Err(PyValueError::new_err(format!(
            "{names} must have the same length"
        )));
    }
    Ok(())
}

pub(crate) fn ma_2d(a: ArrayView2<'_, f64>, window: usize, ewm: bool, adjust: bool) -> Array2<f64> {
    if ewm {
        if a.is_standard_layout() {
            return ewm_mean_2d_c(a, window, window, adjust);
        }
        return apply_2d_by_col(a, |col| ewm_mean_1d(col, window, window, adjust));
    }
    apply_2d_by_col(a, |col| rolling_mean_1d(col, window, window))
}

pub(crate) fn mstd_2d(
    a: ArrayView2<'_, f64>,
    window: usize,
    ewm: bool,
    adjust: bool,
    ddof: usize,
) -> Array2<f64> {
    if ewm {
        if a.is_standard_layout() {
            return ewm_std_2d_c(a, window, window, adjust, ddof);
        }
        return apply_2d_by_col(a, |col| ewm_std_1d(col, window, window, adjust, ddof));
    }
    apply_2d_by_col(a, |col| rolling_std_1d(col, window, window, ddof))
}

pub(crate) fn diff_2d(a: ArrayView2<'_, f64>) -> Array2<f64> {
    apply_2d_by_col_inplace(a, |col, out| diff_1d_into(col, out, 1))
}

pub(crate) fn build_ma_cache<'py>(
    py: Python<'py>,
    close: ArrayView2<'_, f64>,
    windows: &[usize],
    ewms: &[bool],
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    validate_param_lengths("windows and ewms", windows.len(), ewms.len())?;
    let cache_dict = PyDict::new_bound(py);
    for i in 0..windows.len() {
        let h = tuple_hash(py, windows[i], ewms[i])?;
        if !cache_dict.contains(h)? {
            let result = py.allow_threads(|| ma_2d(close, windows[i], ewms[i], adjust));
            cache_dict.set_item(h, PyArray2::from_owned_array_bound(py, result))?;
        }
    }
    Ok(cache_dict)
}

pub(crate) fn build_mstd_cache<'py>(
    py: Python<'py>,
    close: ArrayView2<'_, f64>,
    windows: &[usize],
    ewms: &[bool],
    adjust: bool,
    ddof: usize,
) -> PyResult<Bound<'py, PyDict>> {
    validate_param_lengths("windows and ewms", windows.len(), ewms.len())?;
    let cache_dict = PyDict::new_bound(py);
    for i in 0..windows.len() {
        let h = tuple_hash(py, windows[i], ewms[i])?;
        if !cache_dict.contains(h)? {
            let result = py.allow_threads(|| mstd_2d(close, windows[i], ewms[i], adjust, ddof));
            cache_dict.set_item(h, PyArray2::from_owned_array_bound(py, result))?;
        }
    }
    Ok(cache_dict)
}

#[pyfunction]
#[pyo3(signature = (a, window, ewm, adjust=false))]
pub fn ma_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    adjust: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| ma_2d(a_arr, window, ewm, adjust));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, window, ewm, adjust=false, ddof=0))]
pub fn mstd_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    adjust: bool,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| mstd_2d(a_arr, window, ewm, adjust, ddof));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn ma_cache_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    windows: Vec<usize>,
    ewms: Vec<bool>,
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    build_ma_cache(py, close.as_array(), &windows, &ewms, adjust)
}

#[pyfunction]
pub fn ma_apply_rs<'py>(
    py: Python<'py>,
    _close: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    _adjust: bool,
    cache_dict: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let h = tuple_hash(py, window, ewm)?;
    Ok(cache_dict.get_item(h)?.into_py(py))
}

#[pyfunction]
pub fn mstd_cache_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    windows: Vec<usize>,
    ewms: Vec<bool>,
    adjust: bool,
    ddof: usize,
) -> PyResult<Bound<'py, PyDict>> {
    build_mstd_cache(py, close.as_array(), &windows, &ewms, adjust, ddof)
}

#[pyfunction]
pub fn mstd_apply_rs<'py>(
    py: Python<'py>,
    _close: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    _adjust: bool,
    _ddof: usize,
    cache_dict: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    let h = tuple_hash(py, window, ewm)?;
    Ok(cache_dict.get_item(h)?.into_py(py))
}

#[pyfunction]
pub fn bb_cache_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    windows: Vec<usize>,
    ewms: Vec<bool>,
    _alphas: Vec<f64>,
    adjust: bool,
    ddof: usize,
) -> PyResult<(Bound<'py, PyDict>, Bound<'py, PyDict>)> {
    let close_arr = close.as_array();
    Ok((
        build_ma_cache(py, close_arr, &windows, &ewms, adjust)?,
        build_mstd_cache(py, close_arr, &windows, &ewms, adjust, ddof)?,
    ))
}

#[pyfunction]
pub fn bb_apply_rs<'py>(
    py: Python<'py>,
    _close: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    alpha: f64,
    _adjust: bool,
    _ddof: usize,
    ma_cache_dict: &Bound<'py, PyAny>,
    mstd_cache_dict: &Bound<'py, PyAny>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let h = tuple_hash(py, window, ewm)?;
    let ma_item = ma_cache_dict.get_item(h)?;
    let mstd_item = mstd_cache_dict.get_item(h)?;
    let ma_arr = ma_item.extract::<PyReadonlyArray2<'py, f64>>()?;
    let mstd_arr = mstd_item.extract::<PyReadonlyArray2<'py, f64>>()?;
    let ma_view = ma_arr.as_array();
    let mstd_view = mstd_arr.as_array();
    let mut upper = Array2::<f64>::from_elem(ma_view.dim(), f64::NAN);
    let mut lower = Array2::<f64>::from_elem(ma_view.dim(), f64::NAN);
    let mut ma = Array2::<f64>::from_elem(ma_view.dim(), f64::NAN);
    Zip::from(&mut ma)
        .and(&mut upper)
        .and(&mut lower)
        .and(&ma_view)
        .and(&mstd_view)
        .for_each(|ma_out, upper_out, lower_out, &ma_v, &mstd_v| {
            *ma_out = ma_v;
            *upper_out = ma_v + alpha * mstd_v;
            *lower_out = ma_v - alpha * mstd_v;
        });
    Ok((
        PyArray2::from_owned_array_bound(py, ma),
        PyArray2::from_owned_array_bound(py, upper),
        PyArray2::from_owned_array_bound(py, lower),
    ))
}

#[pyfunction]
pub fn rsi_cache_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    windows: Vec<usize>,
    ewms: Vec<bool>,
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    validate_param_lengths("windows and ewms", windows.len(), ewms.len())?;
    let close_arr = close.as_array();
    let delta = py.allow_threads(|| diff_2d(close_arr));
    let mut up = delta.clone();
    let mut down = delta;
    up.mapv_inplace(|v| if v < 0.0 { 0.0 } else { v });
    down.mapv_inplace(|v| if v > 0.0 { 0.0 } else { v.abs() });

    let cache_dict = PyDict::new_bound(py);
    for i in 0..windows.len() {
        let h = tuple_hash(py, windows[i], ewms[i])?;
        if !cache_dict.contains(h)? {
            let roll_up = py.allow_threads(|| ma_2d(up.view(), windows[i], ewms[i], adjust));
            let roll_down = py.allow_threads(|| ma_2d(down.view(), windows[i], ewms[i], adjust));
            cache_dict.set_item(
                h,
                (
                    PyArray2::from_owned_array_bound(py, roll_up),
                    PyArray2::from_owned_array_bound(py, roll_down),
                ),
            )?;
        }
    }
    Ok(cache_dict)
}

#[pyfunction]
pub fn rsi_apply_rs<'py>(
    py: Python<'py>,
    _close: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    _adjust: bool,
    cache_dict: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let h = tuple_hash(py, window, ewm)?;
    let item = cache_dict.get_item(h)?;
    let tuple = item.downcast::<PyTuple>()?;
    let roll_up_arr = tuple.get_item(0)?.extract::<PyReadonlyArray2<'py, f64>>()?;
    let roll_down_arr = tuple.get_item(1)?.extract::<PyReadonlyArray2<'py, f64>>()?;
    let roll_up = roll_up_arr.as_array();
    let roll_down = roll_down_arr.as_array();
    let mut out = Array2::<f64>::from_elem(roll_up.dim(), f64::NAN);
    Zip::from(&mut out)
        .and(&roll_up)
        .and(&roll_down)
        .for_each(|out_v, &up, &down| {
            let rs = up / down;
            *out_v = 100.0 - 100.0 / (1.0 + rs);
        });
    Ok(PyArray2::from_owned_array_bound(py, out))
}

#[pyfunction]
pub fn stoch_cache_rs<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    _close: PyReadonlyArray2<'py, f64>,
    k_windows: Vec<usize>,
    _d_windows: Vec<usize>,
    _d_ewms: Vec<bool>,
    _adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let high_arr = high.as_array();
    let low_arr = low.as_array();
    if high_arr.dim() != low_arr.dim() {
        return Err(PyValueError::new_err(
            "high and low must have the same shape",
        ));
    }
    let cache_dict = PyDict::new_bound(py);
    for &k_window in &k_windows {
        let h = k_window as isize;
        if !cache_dict.contains(h)? {
            let roll_min = py.allow_threads(|| {
                apply_2d_by_col(low_arr, |col| rolling_min_1d(col, k_window, k_window))
            });
            let roll_max = py.allow_threads(|| {
                apply_2d_by_col(high_arr, |col| rolling_max_1d(col, k_window, k_window))
            });
            cache_dict.set_item(
                h,
                (
                    PyArray2::from_owned_array_bound(py, roll_min),
                    PyArray2::from_owned_array_bound(py, roll_max),
                ),
            )?;
        }
    }
    Ok(cache_dict)
}

#[pyfunction]
pub fn stoch_apply_rs<'py>(
    py: Python<'py>,
    _high: PyReadonlyArray2<'py, f64>,
    _low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    k_window: usize,
    d_window: usize,
    d_ewm: bool,
    adjust: bool,
    cache_dict: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let item = cache_dict.get_item(k_window as isize)?;
    let tuple = item.downcast::<PyTuple>()?;
    let roll_min_arr = tuple.get_item(0)?.extract::<PyReadonlyArray2<'py, f64>>()?;
    let roll_max_arr = tuple.get_item(1)?.extract::<PyReadonlyArray2<'py, f64>>()?;
    let roll_min = roll_min_arr.as_array();
    let roll_max = roll_max_arr.as_array();
    let close_arr = close.as_array();
    if close_arr.dim() != roll_min.dim() {
        return Err(PyValueError::new_err(
            "close and cached arrays must have the same shape",
        ));
    }
    let mut percent_k = Array2::<f64>::from_elem(close_arr.dim(), f64::NAN);
    Zip::from(&mut percent_k)
        .and(&close_arr)
        .and(&roll_min)
        .and(&roll_max)
        .for_each(|out_v, &close_v, &min_v, &max_v| {
            *out_v = 100.0 * (close_v - min_v) / (max_v - min_v);
        });
    let percent_d = py.allow_threads(|| ma_2d(percent_k.view(), d_window, d_ewm, adjust));
    Ok((
        PyArray2::from_owned_array_bound(py, percent_k),
        PyArray2::from_owned_array_bound(py, percent_d),
    ))
}

#[pyfunction]
pub fn macd_cache_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    fast_windows: Vec<usize>,
    slow_windows: Vec<usize>,
    _signal_windows: Vec<usize>,
    macd_ewms: Vec<bool>,
    _signal_ewms: Vec<bool>,
    adjust: bool,
) -> PyResult<Bound<'py, PyDict>> {
    validate_param_lengths(
        "fast_windows and macd_ewms",
        fast_windows.len(),
        macd_ewms.len(),
    )?;
    validate_param_lengths(
        "slow_windows and macd_ewms",
        slow_windows.len(),
        macd_ewms.len(),
    )?;
    let mut windows = fast_windows;
    windows.extend(slow_windows);
    let mut ewms = macd_ewms.clone();
    ewms.extend(macd_ewms);
    build_ma_cache(py, close.as_array(), &windows, &ewms, adjust)
}

#[pyfunction]
pub fn macd_apply_rs<'py>(
    py: Python<'py>,
    _close: PyReadonlyArray2<'py, f64>,
    fast_window: usize,
    slow_window: usize,
    signal_window: usize,
    macd_ewm: bool,
    signal_ewm: bool,
    adjust: bool,
    cache_dict: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let fast_h = tuple_hash(py, fast_window, macd_ewm)?;
    let slow_h = tuple_hash(py, slow_window, macd_ewm)?;
    let fast_item = cache_dict.get_item(fast_h)?;
    let slow_item = cache_dict.get_item(slow_h)?;
    let fast_ma_arr = fast_item.extract::<PyReadonlyArray2<'py, f64>>()?;
    let slow_ma_arr = slow_item.extract::<PyReadonlyArray2<'py, f64>>()?;
    let fast_ma = fast_ma_arr.as_array();
    let slow_ma = slow_ma_arr.as_array();
    let mut macd_ts = Array2::<f64>::from_elem(fast_ma.dim(), f64::NAN);
    Zip::from(&mut macd_ts)
        .and(&fast_ma)
        .and(&slow_ma)
        .for_each(|out_v, &fast_v, &slow_v| {
            *out_v = fast_v - slow_v;
        });
    let signal_ts = py.allow_threads(|| ma_2d(macd_ts.view(), signal_window, signal_ewm, adjust));
    Ok((
        PyArray2::from_owned_array_bound(py, macd_ts),
        PyArray2::from_owned_array_bound(py, signal_ts),
    ))
}

pub(crate) fn max3_numba(a: f64, b: f64, c: f64) -> f64 {
    let mut out = a;
    if b > out {
        out = b;
    }
    if c > out {
        out = c;
    }
    out
}

pub(crate) fn true_range_1d(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    let len = high.len();
    let mut out = vec![f64::NAN; len];
    true_range_1d_into(high, low, close, &mut out);
    out
}

pub(crate) fn true_range_1d_into(high: &[f64], low: &[f64], close: &[f64], out: &mut [f64]) {
    let len = high.len();
    if len == 0 {
        return;
    }
    out[0] = high[0] - low[0];
    for i in 1..len {
        let hl = high[i] - low[i];
        let hc = (high[i] - close[i - 1]).abs();
        let lc = (low[i] - close[i - 1]).abs();
        out[i] = max3_numba(hl, hc, lc);
    }
}

pub(crate) fn true_range_2d(
    high: ArrayView2<'_, f64>,
    low: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
) -> Array2<f64> {
    let (nrows, ncols) = high.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let mut h_buf = vec![0.0f64; nrows];
    let mut l_buf = vec![0.0f64; nrows];
    let mut c_buf = vec![0.0f64; nrows];
    let mut out_buf = vec![0.0f64; nrows];
    for col in 0..ncols {
        for (i, &v) in high.column(col).iter().enumerate() {
            h_buf[i] = v;
        }
        for (i, &v) in low.column(col).iter().enumerate() {
            l_buf[i] = v;
        }
        for (i, &v) in close.column(col).iter().enumerate() {
            c_buf[i] = v;
        }
        true_range_1d_into(&h_buf, &l_buf, &c_buf, &mut out_buf);
        let mut out_col = out.column_mut(col);
        for (i, &v) in out_buf.iter().enumerate() {
            out_col[i] = v;
        }
    }
    out
}

#[pyfunction]
pub fn true_range_1d_rs<'py>(
    py: Python<'py>,
    high: PyReadonlyArray1<'py, f64>,
    low: PyReadonlyArray1<'py, f64>,
    close: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let h_cow = array1_as_slice_cow(&high);
    let l_cow = array1_as_slice_cow(&low);
    let c_cow = array1_as_slice_cow(&close);
    let h = h_cow.as_ref();
    let l = l_cow.as_ref();
    let c = c_cow.as_ref();
    if h.len() != l.len() || h.len() != c.len() {
        return Err(PyValueError::new_err(
            "high, low, and close must have the same length",
        ));
    }
    let result = py.allow_threads(|| true_range_1d(h, l, c));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn true_range_rs<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let h_arr = high.as_array();
    let l_arr = low.as_array();
    let c_arr = close.as_array();
    if h_arr.dim() != l_arr.dim() || h_arr.dim() != c_arr.dim() {
        return Err(PyValueError::new_err(
            "high, low, and close must have the same shape",
        ));
    }
    let result = py.allow_threads(|| true_range_2d(h_arr, l_arr, c_arr));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn atr_cache_rs<'py>(
    py: Python<'py>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    windows: Vec<usize>,
    ewms: Vec<bool>,
    adjust: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyDict>)> {
    validate_param_lengths("windows and ewms", windows.len(), ewms.len())?;
    let h_arr = high.as_array();
    let l_arr = low.as_array();
    let c_arr = close.as_array();
    if h_arr.dim() != l_arr.dim() || h_arr.dim() != c_arr.dim() {
        return Err(PyValueError::new_err(
            "high, low, and close must have the same shape",
        ));
    }
    let tr = py.allow_threads(|| true_range_2d(h_arr, l_arr, c_arr));
    let cache_dict = PyDict::new_bound(py);
    for i in 0..windows.len() {
        let h = tuple_hash(py, windows[i], ewms[i])?;
        if !cache_dict.contains(h)? {
            let result = py.allow_threads(|| ma_2d(tr.view(), windows[i], ewms[i], adjust));
            cache_dict.set_item(h, PyArray2::from_owned_array_bound(py, result))?;
        }
    }
    Ok((PyArray2::from_owned_array_bound(py, tr), cache_dict))
}

#[pyfunction]
pub fn atr_apply_rs<'py>(
    py: Python<'py>,
    _high: PyReadonlyArray2<'py, f64>,
    _low: PyReadonlyArray2<'py, f64>,
    _close: PyReadonlyArray2<'py, f64>,
    window: usize,
    ewm: bool,
    _adjust: bool,
    tr: &Bound<'py, PyAny>,
    cache_dict: &Bound<'py, PyAny>,
) -> PyResult<(PyObject, PyObject)> {
    let h = tuple_hash(py, window, ewm)?;
    Ok((tr.clone().into_py(py), cache_dict.get_item(h)?.into_py(py)))
}

pub(crate) fn obv_custom_1d_into(close: &[f64], volume: &[f64], out: &mut [f64]) {
    let len = close.len();
    if len == 0 {
        return;
    }
    let mut signed_volume = vec![0.0f64; len];
    for i in 0..len {
        signed_volume[i] = if i > 0 && close[i] < close[i - 1] {
            -volume[i]
        } else {
            volume[i]
        };
    }
    nancumsum_1d_into(&signed_volume, out);
}

#[pyfunction]
pub fn obv_custom_rs<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    volume_ts: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let c_arr = close.as_array();
    let v_arr = volume_ts.as_array();
    if c_arr.dim() != v_arr.dim() {
        return Err(PyValueError::new_err(
            "close and volume_ts must have the same shape",
        ));
    }
    let (nrows, ncols) = c_arr.dim();
    let result = py.allow_threads(|| {
        let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
        let mut c_buf = vec![0.0f64; nrows];
        let mut v_buf = vec![0.0f64; nrows];
        let mut out_buf = vec![0.0f64; nrows];
        for col in 0..ncols {
            for (i, &val) in c_arr.column(col).iter().enumerate() {
                c_buf[i] = val;
            }
            for (i, &val) in v_arr.column(col).iter().enumerate() {
                v_buf[i] = val;
            }
            obv_custom_1d_into(&c_buf, &v_buf, &mut out_buf);
            let mut out_col = out.column_mut(col);
            for (i, &val) in out_buf.iter().enumerate() {
                out_col[i] = val;
            }
        }
        out
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ma_rs, m)?)?;
    m.add_function(wrap_pyfunction!(mstd_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ma_cache_rs, m)?)?;
    m.add_function(wrap_pyfunction!(ma_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(mstd_cache_rs, m)?)?;
    m.add_function(wrap_pyfunction!(mstd_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bb_cache_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bb_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rsi_cache_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rsi_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(stoch_cache_rs, m)?)?;
    m.add_function(wrap_pyfunction!(stoch_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(macd_cache_rs, m)?)?;
    m.add_function(wrap_pyfunction!(macd_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(true_range_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(true_range_rs, m)?)?;
    m.add_function(wrap_pyfunction!(atr_cache_rs, m)?)?;
    m.add_function(wrap_pyfunction!(atr_apply_rs, m)?)?;
    m.add_function(wrap_pyfunction!(obv_custom_rs, m)?)?;
    Ok(())
}
