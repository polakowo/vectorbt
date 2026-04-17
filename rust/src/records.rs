// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use ndarray::Array2;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyTuple};

use crate::generic::array1_as_slice_cow;

// ############# Indexing #############

fn col_range(col_arr: &[i64], n_cols: usize) -> Result<Vec<i64>, String> {
    let mut out = vec![-1i64; n_cols * 2];
    let mut last_col: i64 = -1;

    for r in 0..col_arr.len() {
        let col = col_arr[r];
        if col < last_col {
            return Err("col_arr must be in ascending order".to_string());
        }
        if col != last_col {
            if last_col != -1 {
                out[last_col as usize * 2 + 1] = r as i64;
            }
            out[col as usize * 2] = r as i64;
            last_col = col;
        }
        if r == col_arr.len() - 1 {
            out[col as usize * 2 + 1] = (r + 1) as i64;
        }
    }
    Ok(out)
}

#[pyfunction]
pub fn col_range_rs<'py>(
    py: Python<'py>,
    col_arr: PyReadonlyArray1<'py, i64>,
    n_cols: usize,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let col_arr_cow = array1_as_slice_cow(&col_arr);
    let result = py
        .allow_threads(|| col_range(col_arr_cow.as_ref(), n_cols))
        .map_err(PyValueError::new_err)?;
    let arr = Array2::from_shape_vec((n_cols, 2), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

fn col_range_select(
    col_range: &[i64],
    new_cols: &[i64],
) -> (Vec<i64>, Vec<i64>) {
    // First pass: compute total count
    let mut new_n: usize = 0;
    for &nc in new_cols {
        let c = nc as usize;
        let from_r = col_range[c * 2];
        let to_r = col_range[c * 2 + 1];
        if from_r != -1 && to_r != -1 {
            new_n += (to_r - from_r) as usize;
        }
    }

    let mut indices_out = vec![0i64; new_n];
    let mut col_arr_out = vec![0i64; new_n];
    let mut j: usize = 0;

    for (c_idx, &nc) in new_cols.iter().enumerate() {
        let c = nc as usize;
        let from_r = col_range[c * 2];
        let to_r = col_range[c * 2 + 1];
        if from_r == -1 || to_r == -1 {
            continue;
        }
        let count = (to_r - from_r) as usize;
        for k in 0..count {
            indices_out[j + k] = from_r + k as i64;
            col_arr_out[j + k] = c_idx as i64;
        }
        j += count;
    }
    (indices_out, col_arr_out)
}

#[pyfunction]
pub fn col_range_select_rs<'py>(
    py: Python<'py>,
    col_range: PyReadonlyArray2<'py, i64>,
    new_cols: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let cr_arr = col_range.as_array();
    // Flatten to contiguous slice
    let cr_flat: Vec<i64> = cr_arr.iter().copied().collect();
    let new_cols_cow = array1_as_slice_cow(&new_cols);
    let (indices, col_arr) =
        py.allow_threads(|| col_range_select(&cr_flat, new_cols_cow.as_ref()));
    Ok((
        PyArray1::from_vec_bound(py, indices),
        PyArray1::from_vec_bound(py, col_arr),
    ))
}

fn col_map(col_arr: &[i64], n_cols: usize) -> (Vec<i64>, Vec<i64>) {
    let mut col_lens = vec![0i64; n_cols];
    for &col in col_arr {
        col_lens[col as usize] += 1;
    }

    let mut col_start_idxs = vec![0i64; n_cols];
    for i in 1..n_cols {
        col_start_idxs[i] = col_start_idxs[i - 1] + col_lens[i - 1];
    }

    let mut col_idxs_out = vec![0i64; col_arr.len()];
    let mut col_i = vec![0i64; n_cols];
    for r in 0..col_arr.len() {
        let col = col_arr[r] as usize;
        col_idxs_out[(col_start_idxs[col] + col_i[col]) as usize] = r as i64;
        col_i[col] += 1;
    }

    (col_idxs_out, col_lens)
}

#[pyfunction]
pub fn col_map_rs<'py>(
    py: Python<'py>,
    col_arr: PyReadonlyArray1<'py, i64>,
    n_cols: usize,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let col_arr_cow = array1_as_slice_cow(&col_arr);
    let (col_idxs, col_lens) = py.allow_threads(|| col_map(col_arr_cow.as_ref(), n_cols));
    Ok((
        PyArray1::from_vec_bound(py, col_idxs),
        PyArray1::from_vec_bound(py, col_lens),
    ))
}

fn col_map_select(
    col_idxs: &[i64],
    col_lens: &[i64],
    new_cols: &[i64],
) -> (Vec<i64>, Vec<i64>) {
    let n_cols = col_lens.len();

    // Compute col_start_idxs
    let mut col_start_idxs = vec![0i64; n_cols];
    for i in 1..n_cols {
        col_start_idxs[i] = col_start_idxs[i - 1] + col_lens[i - 1];
    }

    // Total count
    let mut total_count: usize = 0;
    for &nc in new_cols {
        total_count += col_lens[nc as usize] as usize;
    }

    let mut idxs_out = vec![0i64; total_count];
    let mut col_arr_out = vec![0i64; total_count];
    let mut j: usize = 0;

    for (new_col_i, &new_col) in new_cols.iter().enumerate() {
        let col_len = col_lens[new_col as usize] as usize;
        if col_len == 0 {
            continue;
        }
        let col_start_idx = col_start_idxs[new_col as usize] as usize;
        for k in 0..col_len {
            idxs_out[j + k] = col_idxs[col_start_idx + k];
            col_arr_out[j + k] = new_col_i as i64;
        }
        j += col_len;
    }
    (idxs_out, col_arr_out)
}

#[pyfunction]
pub fn col_map_select_rs<'py>(
    py: Python<'py>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
    new_cols: PyReadonlyArray1<'py, i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    let col_idxs_cow = array1_as_slice_cow(&col_idxs);
    let col_lens_cow = array1_as_slice_cow(&col_lens);
    let new_cols_cow = array1_as_slice_cow(&new_cols);
    let (idxs, col_arr) = py.allow_threads(|| {
        col_map_select(
            col_idxs_cow.as_ref(),
            col_lens_cow.as_ref(),
            new_cols_cow.as_ref(),
        )
    });
    Ok((
        PyArray1::from_vec_bound(py, idxs),
        PyArray1::from_vec_bound(py, col_arr),
    ))
}

/// Byte offset of the `col` field in all vectorbt record dtypes.
/// All record dtypes start with (id: i64, col: i64, ...) so col is at offset 8.
const COL_FIELD_OFFSET: usize = 8;

/// Helper: create an empty numpy array with the given dtype and length.
fn numpy_empty<'py>(
    py: Python<'py>,
    n: usize,
    dtype: &Bound<'py, pyo3::PyAny>,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let np = py.import_bound("numpy")?;
    let args = PyTuple::new_bound(py, &[n.into_py(py)]);
    np.call_method("empty", (args, ), Some(&[("dtype", dtype)].into_py_dict_bound(py)))
}

/// Get the data pointer and itemsize from an untyped numpy array.
///
/// # Safety
/// The caller must ensure the array is contiguous and the returned pointer
/// is only used while the array is alive.
unsafe fn array_raw_parts(arr: &Bound<'_, pyo3::PyAny>) -> PyResult<(*mut u8, usize, usize)> {
    let arr_obj = arr.as_array_ptr() as *mut numpy::npyffi::PyArrayObject;
    let data = (*arr_obj).data as *mut u8;
    let dtype = arr.getattr("dtype")?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    let n: usize = arr.len()?;
    Ok((data, itemsize, n))
}

trait AsArrayPtr {
    fn as_array_ptr(&self) -> *mut pyo3::ffi::PyObject;
}

impl<'py> AsArrayPtr for Bound<'py, pyo3::PyAny> {
    fn as_array_ptr(&self) -> *mut pyo3::ffi::PyObject {
        self.as_ptr()
    }
}

#[pyfunction]
pub fn record_col_range_select_rs<'py>(
    py: Python<'py>,
    records: Bound<'py, pyo3::PyAny>,
    col_range: PyReadonlyArray2<'py, i64>,
    new_cols: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let cr_arr = col_range.as_array();
    let new_cols_cow = array1_as_slice_cow(&new_cols);

    // Get raw parts from input records
    let (src_data, itemsize, _src_n) = unsafe { array_raw_parts(&records)? };

    // Compute total output count
    let mut new_n: usize = 0;
    for &nc in new_cols_cow.as_ref() {
        let c = nc as usize;
        let from_r = cr_arr[[c, 0]];
        let to_r = cr_arr[[c, 1]];
        if from_r != -1 && to_r != -1 {
            new_n += (to_r - from_r) as usize;
        }
    }

    // Create output array with same dtype
    let dtype = records.getattr("dtype")?;
    let out = numpy_empty(py, new_n, &dtype)?;
    let (dst_data, _, _) = unsafe { array_raw_parts(&out)? };

    // Copy records and patch col field
    let mut j: usize = 0;
    for (c_idx, &nc) in new_cols_cow.as_ref().iter().enumerate() {
        let c = nc as usize;
        let from_r = cr_arr[[c, 0]];
        let to_r = cr_arr[[c, 1]];
        if from_r == -1 || to_r == -1 {
            continue;
        }
        let count = (to_r - from_r) as usize;
        unsafe {
            // Copy all records in this column range
            std::ptr::copy_nonoverlapping(
                src_data.add(from_r as usize * itemsize),
                dst_data.add(j * itemsize),
                count * itemsize,
            );
            // Patch col field for each copied record
            let new_col = c_idx as i64;
            for k in 0..count {
                let col_ptr = dst_data.add((j + k) * itemsize + COL_FIELD_OFFSET) as *mut i64;
                *col_ptr = new_col;
            }
        }
        j += count;
    }

    Ok(out)
}

#[pyfunction]
pub fn record_col_map_select_rs<'py>(
    py: Python<'py>,
    records: Bound<'py, pyo3::PyAny>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
    new_cols: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, pyo3::PyAny>> {
    let col_idxs_cow = array1_as_slice_cow(&col_idxs);
    let col_lens_cow = array1_as_slice_cow(&col_lens);
    let new_cols_cow = array1_as_slice_cow(&new_cols);
    let col_idxs_s = col_idxs_cow.as_ref();
    let col_lens_s = col_lens_cow.as_ref();

    // Get raw parts from input records
    let (src_data, itemsize, _src_n) = unsafe { array_raw_parts(&records)? };

    // Compute col_start_idxs and total count
    let n_cols = col_lens_s.len();
    let mut col_start_idxs = vec![0usize; n_cols];
    for i in 1..n_cols {
        col_start_idxs[i] = col_start_idxs[i - 1] + col_lens_s[i - 1] as usize;
    }
    let mut total_count: usize = 0;
    for &nc in new_cols_cow.as_ref() {
        total_count += col_lens_s[nc as usize] as usize;
    }

    // Create output array with same dtype
    let dtype = records.getattr("dtype")?;
    let out = numpy_empty(py, total_count, &dtype)?;
    let (dst_data, _, _) = unsafe { array_raw_parts(&out)? };

    // Copy records and patch col field
    let mut j: usize = 0;
    for (new_col_i, &new_col) in new_cols_cow.as_ref().iter().enumerate() {
        let col_len = col_lens_s[new_col as usize] as usize;
        if col_len == 0 {
            continue;
        }
        let col_start_idx = col_start_idxs[new_col as usize];
        let new_col_val = new_col_i as i64;
        for k in 0..col_len {
            let src_record_idx = col_idxs_s[col_start_idx + k] as usize;
            unsafe {
                // Copy one record
                std::ptr::copy_nonoverlapping(
                    src_data.add(src_record_idx * itemsize),
                    dst_data.add(j * itemsize),
                    itemsize,
                );
                // Patch col field
                let col_ptr = dst_data.add(j * itemsize + COL_FIELD_OFFSET) as *mut i64;
                *col_ptr = new_col_val;
            }
            j += 1;
        }
    }

    Ok(out)
}

// ############# Sorting #############

fn is_col_sorted(col_arr: &[i64]) -> bool {
    for i in 0..col_arr.len().saturating_sub(1) {
        if col_arr[i + 1] < col_arr[i] {
            return false;
        }
    }
    true
}

#[pyfunction]
pub fn is_col_sorted_rs<'py>(
    py: Python<'py>,
    col_arr: PyReadonlyArray1<'py, i64>,
) -> PyResult<bool> {
    let col_arr_cow = array1_as_slice_cow(&col_arr);
    Ok(py.allow_threads(|| is_col_sorted(col_arr_cow.as_ref())))
}

fn is_col_idx_sorted(col_arr: &[i64], id_arr: &[i64]) -> bool {
    for i in 0..col_arr.len().saturating_sub(1) {
        if col_arr[i + 1] < col_arr[i] {
            return false;
        }
        if col_arr[i + 1] == col_arr[i] && id_arr[i + 1] < id_arr[i] {
            return false;
        }
    }
    true
}

#[pyfunction]
pub fn is_col_idx_sorted_rs<'py>(
    py: Python<'py>,
    col_arr: PyReadonlyArray1<'py, i64>,
    id_arr: PyReadonlyArray1<'py, i64>,
) -> PyResult<bool> {
    let col_arr_cow = array1_as_slice_cow(&col_arr);
    let id_arr_cow = array1_as_slice_cow(&id_arr);
    Ok(py.allow_threads(|| is_col_idx_sorted(col_arr_cow.as_ref(), id_arr_cow.as_ref())))
}

// ############# Expansion #############

fn is_mapped_expandable(col_arr: &[i64], idx_arr: &[i64], nrows: usize, ncols: usize) -> bool {
    let mut seen = vec![false; nrows * ncols];
    for i in 0..col_arr.len() {
        let pos = idx_arr[i] as usize * ncols + col_arr[i] as usize;
        if seen[pos] {
            return false;
        }
        seen[pos] = true;
    }
    true
}

#[pyfunction]
pub fn is_mapped_expandable_rs<'py>(
    py: Python<'py>,
    col_arr: PyReadonlyArray1<'py, i64>,
    idx_arr: PyReadonlyArray1<'py, i64>,
    target_shape: (usize, usize),
) -> PyResult<bool> {
    let col_arr_cow = array1_as_slice_cow(&col_arr);
    let idx_arr_cow = array1_as_slice_cow(&idx_arr);
    let (nrows, ncols) = target_shape;
    Ok(py.allow_threads(|| {
        is_mapped_expandable(col_arr_cow.as_ref(), idx_arr_cow.as_ref(), nrows, ncols)
    }))
}

fn expand_mapped(
    mapped_arr: &[f64],
    col_arr: &[i64],
    idx_arr: &[i64],
    nrows: usize,
    ncols: usize,
    fill_value: f64,
) -> Array2<f64> {
    let mut out = Array2::<f64>::from_elem((nrows, ncols), fill_value);
    for r in 0..mapped_arr.len() {
        out[[idx_arr[r] as usize, col_arr[r] as usize]] = mapped_arr[r];
    }
    out
}

#[pyfunction]
pub fn expand_mapped_rs<'py>(
    py: Python<'py>,
    mapped_arr: PyReadonlyArray1<'py, f64>,
    col_arr: PyReadonlyArray1<'py, i64>,
    idx_arr: PyReadonlyArray1<'py, i64>,
    target_shape: (usize, usize),
    fill_value: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mapped_cow = array1_as_slice_cow(&mapped_arr);
    let col_arr_cow = array1_as_slice_cow(&col_arr);
    let idx_arr_cow = array1_as_slice_cow(&idx_arr);
    let (nrows, ncols) = target_shape;
    let result = py.allow_threads(|| {
        expand_mapped(
            mapped_cow.as_ref(),
            col_arr_cow.as_ref(),
            idx_arr_cow.as_ref(),
            nrows,
            ncols,
            fill_value,
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

fn stack_expand_mapped(
    mapped_arr: &[f64],
    col_idxs: &[i64],
    col_lens: &[i64],
    fill_value: f64,
) -> Array2<f64> {
    let n_cols = col_lens.len();

    // Compute col_start_idxs and max_len
    let mut col_start_idxs = vec![0usize; n_cols];
    let mut max_len: usize = 0;
    for i in 0..n_cols {
        if i > 0 {
            col_start_idxs[i] = col_start_idxs[i - 1] + col_lens[i - 1] as usize;
        }
        let cl = col_lens[i] as usize;
        if cl > max_len {
            max_len = cl;
        }
    }

    let mut out = Array2::<f64>::from_elem((max_len, n_cols), fill_value);
    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let col_start_idx = col_start_idxs[col];
        for k in 0..col_len {
            out[[k, col]] = mapped_arr[col_idxs[col_start_idx + k] as usize];
        }
    }
    out
}

#[pyfunction]
pub fn stack_expand_mapped_rs<'py>(
    py: Python<'py>,
    mapped_arr: PyReadonlyArray1<'py, f64>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
    fill_value: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mapped_cow = array1_as_slice_cow(&mapped_arr);
    let col_idxs_cow = array1_as_slice_cow(&col_idxs);
    let col_lens_cow = array1_as_slice_cow(&col_lens);
    let result = py.allow_threads(|| {
        stack_expand_mapped(
            mapped_cow.as_ref(),
            col_idxs_cow.as_ref(),
            col_lens_cow.as_ref(),
            fill_value,
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

// ############# Reducing #############

fn mapped_value_counts(
    codes: &[i64],
    n_uniques: usize,
    col_idxs: &[i64],
    col_lens: &[i64],
) -> Array2<i64> {
    let n_cols = col_lens.len();

    let mut col_start_idxs = vec![0usize; n_cols];
    for i in 1..n_cols {
        col_start_idxs[i] = col_start_idxs[i - 1] + col_lens[i - 1] as usize;
    }

    let mut out = Array2::<i64>::zeros((n_uniques, n_cols));
    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let col_start_idx = col_start_idxs[col];
        for c in 0..col_len {
            let code = codes[col_idxs[col_start_idx + c] as usize] as usize;
            out[[code, col]] += 1;
        }
    }
    out
}

#[pyfunction]
pub fn mapped_value_counts_rs<'py>(
    py: Python<'py>,
    codes: PyReadonlyArray1<'py, i64>,
    n_uniques: usize,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let codes_cow = array1_as_slice_cow(&codes);
    let col_idxs_cow = array1_as_slice_cow(&col_idxs);
    let col_lens_cow = array1_as_slice_cow(&col_lens);
    let result = py.allow_threads(|| {
        mapped_value_counts(
            codes_cow.as_ref(),
            n_uniques,
            col_idxs_cow.as_ref(),
            col_lens_cow.as_ref(),
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

// ############# Mapping #############

fn top_n_mapped_mask(
    mapped_arr: &[f64],
    col_idxs: &[i64],
    col_lens: &[i64],
    n: usize,
) -> Vec<bool> {
    let total_len = mapped_arr.len();
    let n_cols = col_lens.len();
    let mut out = vec![false; total_len];

    let mut col_start_idxs = vec![0usize; n_cols];
    for i in 1..n_cols {
        col_start_idxs[i] = col_start_idxs[i - 1] + col_lens[i - 1] as usize;
    }

    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let col_start_idx = col_start_idxs[col];

        // Collect (value, global_index) for this column
        let mut items: Vec<(f64, usize)> = Vec::with_capacity(col_len);
        for k in 0..col_len {
            let global_idx = col_idxs[col_start_idx + k] as usize;
            items.push((mapped_arr[global_idx], global_idx));
        }

        // Sort ascending; top N = last N after sort
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let start = if col_len > n { col_len - n } else { 0 };
        for k in start..col_len {
            out[items[k].1] = true;
        }
    }
    out
}

#[pyfunction]
pub fn top_n_mapped_mask_rs<'py>(
    py: Python<'py>,
    mapped_arr: PyReadonlyArray1<'py, f64>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let mapped_cow = array1_as_slice_cow(&mapped_arr);
    let col_idxs_cow = array1_as_slice_cow(&col_idxs);
    let col_lens_cow = array1_as_slice_cow(&col_lens);
    let result = py.allow_threads(|| {
        top_n_mapped_mask(
            mapped_cow.as_ref(),
            col_idxs_cow.as_ref(),
            col_lens_cow.as_ref(),
            n,
        )
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

fn bottom_n_mapped_mask(
    mapped_arr: &[f64],
    col_idxs: &[i64],
    col_lens: &[i64],
    n: usize,
) -> Vec<bool> {
    let total_len = mapped_arr.len();
    let n_cols = col_lens.len();
    let mut out = vec![false; total_len];

    let mut col_start_idxs = vec![0usize; n_cols];
    for i in 1..n_cols {
        col_start_idxs[i] = col_start_idxs[i - 1] + col_lens[i - 1] as usize;
    }

    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let col_start_idx = col_start_idxs[col];

        // Collect (value, global_index) for this column
        let mut items: Vec<(f64, usize)> = Vec::with_capacity(col_len);
        for k in 0..col_len {
            let global_idx = col_idxs[col_start_idx + k] as usize;
            items.push((mapped_arr[global_idx], global_idx));
        }

        // Sort ascending; bottom N = first N after sort
        items.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        let take = if col_len < n { col_len } else { n };
        for k in 0..take {
            out[items[k].1] = true;
        }
    }
    out
}

#[pyfunction]
pub fn bottom_n_mapped_mask_rs<'py>(
    py: Python<'py>,
    mapped_arr: PyReadonlyArray1<'py, f64>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
    n: usize,
) -> PyResult<Bound<'py, PyArray1<bool>>> {
    let mapped_cow = array1_as_slice_cow(&mapped_arr);
    let col_idxs_cow = array1_as_slice_cow(&col_idxs);
    let col_lens_cow = array1_as_slice_cow(&col_lens);
    let result = py.allow_threads(|| {
        bottom_n_mapped_mask(
            mapped_cow.as_ref(),
            col_idxs_cow.as_ref(),
            col_lens_cow.as_ref(),
            n,
        )
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

// ############# Registration #############

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(col_range_rs, m)?)?;
    m.add_function(wrap_pyfunction!(col_range_select_rs, m)?)?;
    m.add_function(wrap_pyfunction!(col_map_rs, m)?)?;
    m.add_function(wrap_pyfunction!(col_map_select_rs, m)?)?;
    m.add_function(wrap_pyfunction!(record_col_range_select_rs, m)?)?;
    m.add_function(wrap_pyfunction!(record_col_map_select_rs, m)?)?;
    m.add_function(wrap_pyfunction!(is_col_sorted_rs, m)?)?;
    m.add_function(wrap_pyfunction!(is_col_idx_sorted_rs, m)?)?;
    m.add_function(wrap_pyfunction!(is_mapped_expandable_rs, m)?)?;
    m.add_function(wrap_pyfunction!(expand_mapped_rs, m)?)?;
    m.add_function(wrap_pyfunction!(stack_expand_mapped_rs, m)?)?;
    m.add_function(wrap_pyfunction!(mapped_value_counts_rs, m)?)?;
    m.add_function(wrap_pyfunction!(top_n_mapped_mask_rs, m)?)?;
    m.add_function(wrap_pyfunction!(bottom_n_mapped_mask_rs, m)?)?;
    Ok(())
}
