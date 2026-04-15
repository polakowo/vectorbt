// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use crate::generic::{array1_as_slice_cow, RangeRecord, RANGE_CLOSED, RANGE_OPEN};
use ndarray::{Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyZeroDivisionError;
use pyo3::prelude::*;

pub(crate) fn clean_enex_1d(
    entries: &[bool],
    exits: &[bool],
    entry_first: bool,
) -> (Vec<bool>, Vec<bool>) {
    let mut entries_out = vec![false; entries.len()];
    let mut exits_out = vec![false; exits.len()];
    let mut phase = -1i64;

    for i in 0..entries.len() {
        if entries[i] && exits[i] {
            continue;
        }
        if entries[i] && (phase == -1 || phase == 0) {
            phase = 1;
            entries_out[i] = true;
        }
        if exits[i] && ((!entry_first && phase == -1) || phase == 1) {
            phase = 0;
            exits_out[i] = true;
        }
    }

    (entries_out, exits_out)
}

pub(crate) fn clean_enex_2d(
    entries: ArrayView2<'_, bool>,
    exits: ArrayView2<'_, bool>,
    entry_first: bool,
) -> (Array2<bool>, Array2<bool>) {
    let (nrows, ncols) = entries.dim();
    let mut entries_out = Array2::<bool>::from_elem((nrows, ncols), false);
    let mut exits_out = Array2::<bool>::from_elem((nrows, ncols), false);
    let mut entries_buf = vec![false; nrows];
    let mut exits_buf = vec![false; nrows];

    for col in 0..ncols {
        for row in 0..nrows {
            entries_buf[row] = entries[[row, col]];
            exits_buf[row] = exits[[row, col]];
        }
        let (col_entries, col_exits) = clean_enex_1d(&entries_buf, &exits_buf, entry_first);
        for row in 0..nrows {
            entries_out[[row, col]] = col_entries[row];
            exits_out[[row, col]] = col_exits[row];
        }
    }

    (entries_out, exits_out)
}

pub(crate) fn between_ranges(a: ArrayView2<'_, bool>) -> Vec<RangeRecord> {
    let (nrows, ncols) = a.dim();
    let mut out = Vec::<RangeRecord>::with_capacity(nrows * ncols);
    let mut ridx = 0i64;

    for col in 0..ncols {
        let mut prev_idx = -1i64;
        for row in 0..nrows {
            if a[[row, col]] {
                if prev_idx >= 0 {
                    out.push(RangeRecord::new(
                        ridx,
                        col as i64,
                        prev_idx,
                        row as i64,
                        RANGE_CLOSED,
                    ));
                    ridx += 1;
                }
                prev_idx = row as i64;
            }
        }
    }

    out
}

pub(crate) fn between_two_ranges(
    a: ArrayView2<'_, bool>,
    b: ArrayView2<'_, bool>,
    from_other: bool,
) -> Vec<RangeRecord> {
    let (nrows, ncols) = a.dim();
    let mut out = Vec::<RangeRecord>::with_capacity(nrows * ncols);
    let mut ridx = 0i64;

    for col in 0..ncols {
        let mut a_idxs = Vec::<i64>::new();
        let mut b_idxs = Vec::<i64>::new();
        for row in 0..nrows {
            if a[[row, col]] {
                a_idxs.push(row as i64);
            }
            if b[[row, col]] {
                b_idxs.push(row as i64);
            }
        }
        if a_idxs.is_empty() || b_idxs.is_empty() {
            continue;
        }
        if from_other {
            let mut a_pos = 0usize;
            for &to_i in &b_idxs {
                while a_pos + 1 < a_idxs.len() && a_idxs[a_pos + 1] <= to_i {
                    a_pos += 1;
                }
                if a_idxs[a_pos] <= to_i {
                    out.push(RangeRecord::new(
                        ridx,
                        col as i64,
                        a_idxs[a_pos],
                        to_i,
                        RANGE_CLOSED,
                    ));
                    ridx += 1;
                }
            }
        } else {
            let mut b_pos = 0usize;
            for &from_i in &a_idxs {
                while b_pos < b_idxs.len() && b_idxs[b_pos] < from_i {
                    b_pos += 1;
                }
                if b_pos < b_idxs.len() {
                    out.push(RangeRecord::new(
                        ridx,
                        col as i64,
                        from_i,
                        b_idxs[b_pos],
                        RANGE_CLOSED,
                    ));
                    ridx += 1;
                }
            }
        }
    }

    out
}

pub(crate) fn partition_ranges(a: ArrayView2<'_, bool>) -> Vec<RangeRecord> {
    let (nrows, ncols) = a.dim();
    let mut out = Vec::<RangeRecord>::with_capacity(nrows * ncols);
    let mut ridx = 0i64;

    for col in 0..ncols {
        let mut is_partition = false;
        let mut from_i = -1i64;
        for row in 0..nrows {
            if a[[row, col]] {
                if !is_partition {
                    from_i = row as i64;
                }
                is_partition = true;
            } else if is_partition {
                out.push(RangeRecord::new(
                    ridx,
                    col as i64,
                    from_i,
                    row as i64,
                    RANGE_CLOSED,
                ));
                ridx += 1;
                is_partition = false;
            }
            if row == nrows - 1 && is_partition {
                out.push(RangeRecord::new(
                    ridx,
                    col as i64,
                    from_i,
                    (nrows - 1) as i64,
                    RANGE_OPEN,
                ));
                ridx += 1;
            }
        }
    }

    out
}

pub(crate) fn between_partition_ranges(a: ArrayView2<'_, bool>) -> Vec<RangeRecord> {
    let (nrows, ncols) = a.dim();
    let mut out = Vec::<RangeRecord>::with_capacity(nrows * ncols);
    let mut ridx = 0i64;

    for col in 0..ncols {
        let mut is_partition = false;
        let mut from_i = -1i64;
        for row in 0..nrows {
            if a[[row, col]] {
                if !is_partition && from_i != -1 {
                    out.push(RangeRecord::new(
                        ridx,
                        col as i64,
                        from_i,
                        row as i64,
                        RANGE_CLOSED,
                    ));
                    ridx += 1;
                }
                is_partition = true;
                from_i = row as i64;
            } else {
                is_partition = false;
            }
        }
    }

    out
}

pub(crate) fn sig_pos_rank(
    a: ArrayView2<'_, bool>,
    reset_by: Option<ArrayView2<'_, bool>>,
    after_false: bool,
    allow_gaps: bool,
) -> Array2<i64> {
    let (nrows, ncols) = a.dim();
    let mut out = Array2::<i64>::from_elem((nrows, ncols), -1);
    let mut sig_pos_temp = vec![-1i64; ncols];

    for col in 0..ncols {
        let mut reset_i = 0usize;
        let mut prev_part_end_i = -1i64;
        let mut part_start_i = -1i64;
        let mut in_partition = false;
        let mut false_seen = !after_false;
        for row in 0..nrows {
            if let Some(reset) = reset_by {
                if reset[[row, col]] {
                    reset_i = row;
                }
            }
            if a[[row, col]] && !after_false || (a[[row, col]] && false_seen) {
                if !in_partition {
                    part_start_i = row as i64;
                }
                in_partition = true;
                if reset_i as i64 > prev_part_end_i && reset_i.max(part_start_i as usize) == row {
                    sig_pos_temp[col] = -1;
                } else if !allow_gaps && part_start_i == row as i64 {
                    sig_pos_temp[col] = -1;
                }
                sig_pos_temp[col] += 1;
                out[[row, col]] = sig_pos_temp[col];
            } else if !a[[row, col]] {
                if in_partition {
                    prev_part_end_i = row as i64 - 1;
                }
                in_partition = false;
                false_seen = true;
            }
        }
    }

    out
}

pub(crate) fn part_pos_rank(
    a: ArrayView2<'_, bool>,
    reset_by: Option<ArrayView2<'_, bool>>,
    after_false: bool,
) -> Array2<i64> {
    let (nrows, ncols) = a.dim();
    let mut out = Array2::<i64>::from_elem((nrows, ncols), -1);
    let mut part_pos_temp = vec![-1i64; ncols];

    for col in 0..ncols {
        let mut reset_i = 0usize;
        let mut prev_part_end_i = -1i64;
        let mut part_start_i = -1i64;
        let mut in_partition = false;
        let mut false_seen = !after_false;
        for row in 0..nrows {
            if let Some(reset) = reset_by {
                if reset[[row, col]] {
                    reset_i = row;
                }
            }
            if a[[row, col]] && !after_false || (a[[row, col]] && false_seen) {
                if !in_partition {
                    part_start_i = row as i64;
                }
                in_partition = true;
                if reset_i as i64 > prev_part_end_i && reset_i.max(part_start_i as usize) == row {
                    part_pos_temp[col] = 0;
                } else if part_start_i == row as i64 {
                    part_pos_temp[col] += 1;
                }
                out[[row, col]] = part_pos_temp[col];
            } else if !a[[row, col]] {
                if in_partition {
                    prev_part_end_i = row as i64 - 1;
                }
                in_partition = false;
                false_seen = true;
            }
        }
    }

    out
}

pub(crate) fn nth_index_1d(a: &[bool], n: i64) -> i64 {
    if n >= 0 {
        let mut found = -1i64;
        for (i, &v) in a.iter().enumerate() {
            if v {
                found += 1;
                if found == n {
                    return i as i64;
                }
            }
        }
    } else {
        let mut found = 0i64;
        for i in (0..a.len()).rev() {
            if a[i] {
                found -= 1;
                if found == n {
                    return i as i64;
                }
            }
        }
    }
    -1
}

pub(crate) fn nth_index(a: ArrayView2<'_, bool>, n: i64) -> Vec<i64> {
    let (nrows, ncols) = a.dim();
    let mut out = vec![-1i64; ncols];
    let mut col_buf = vec![false; nrows];

    for col in 0..ncols {
        for row in 0..nrows {
            col_buf[row] = a[[row, col]];
        }
        out[col] = nth_index_1d(&col_buf, n);
    }

    out
}

pub(crate) fn norm_avg_index_1d(a: &[bool]) -> f64 {
    let mut sum = 0.0f64;
    let mut cnt = 0usize;
    for (i, &v) in a.iter().enumerate() {
        if v {
            sum += i as f64;
            cnt += 1;
        }
    }
    let mean_index = sum / cnt as f64;
    (2.0 * mean_index / (a.len() as f64 - 1.0)) - 1.0
}

pub(crate) fn norm_avg_index(a: ArrayView2<'_, bool>) -> Vec<f64> {
    let (nrows, ncols) = a.dim();
    let mut out = vec![f64::NAN; ncols];
    let mut col_buf = vec![false; nrows];

    for col in 0..ncols {
        for row in 0..nrows {
            col_buf[row] = a[[row, col]];
        }
        out[col] = norm_avg_index_1d(&col_buf);
    }

    out
}

#[pyfunction]
pub fn clean_enex_1d_rs<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray1<'py, bool>,
    exits: PyReadonlyArray1<'py, bool>,
    entry_first: bool,
) -> PyResult<(Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<bool>>)> {
    let entries_cow = array1_as_slice_cow(&entries);
    let exits_cow = array1_as_slice_cow(&exits);
    let (entries_out, exits_out) =
        py.allow_threads(|| clean_enex_1d(entries_cow.as_ref(), exits_cow.as_ref(), entry_first));
    Ok((
        PyArray1::from_vec_bound(py, entries_out),
        PyArray1::from_vec_bound(py, exits_out),
    ))
}

#[pyfunction]
pub fn clean_enex_rs<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'py, bool>,
    exits: PyReadonlyArray2<'py, bool>,
    entry_first: bool,
) -> PyResult<(Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>)> {
    let entries_arr = entries.as_array();
    let exits_arr = exits.as_array();
    let (entries_out, exits_out) =
        py.allow_threads(|| clean_enex_2d(entries_arr, exits_arr, entry_first));
    Ok((
        PyArray2::from_owned_array_bound(py, entries_out),
        PyArray2::from_owned_array_bound(py, exits_out),
    ))
}

#[pyfunction]
pub fn between_ranges_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<RangeRecord>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| between_ranges(a_arr));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, b, from_other=false))]
pub fn between_two_ranges_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, bool>,
    b: PyReadonlyArray2<'py, bool>,
    from_other: bool,
) -> PyResult<Bound<'py, PyArray1<RangeRecord>>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();
    let result = py.allow_threads(|| between_two_ranges(a_arr, b_arr, from_other));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn partition_ranges_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<RangeRecord>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| partition_ranges(a_arr));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn between_partition_ranges_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<RangeRecord>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| between_partition_ranges(a_arr));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, reset_by=None, after_false=false, allow_gaps=false))]
pub fn sig_pos_rank_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, bool>,
    reset_by: Option<PyReadonlyArray2<'py, bool>>,
    after_false: bool,
    allow_gaps: bool,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let a_arr = a.as_array();
    let reset_arr = reset_by.as_ref().map(|x| x.as_array());
    let result = py.allow_threads(|| sig_pos_rank(a_arr, reset_arr, after_false, allow_gaps));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (a, reset_by=None, after_false=false))]
pub fn part_pos_rank_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, bool>,
    reset_by: Option<PyReadonlyArray2<'py, bool>>,
    after_false: bool,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let a_arr = a.as_array();
    let reset_arr = reset_by.as_ref().map(|x| x.as_array());
    let result = py.allow_threads(|| part_pos_rank(a_arr, reset_arr, after_false));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn nth_index_1d_rs(a: PyReadonlyArray1<'_, bool>, n: i64) -> PyResult<i64> {
    let a_cow = array1_as_slice_cow(&a);
    Ok(nth_index_1d(a_cow.as_ref(), n))
}

#[pyfunction]
pub fn nth_index_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, bool>,
    n: i64,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let a_arr = a.as_array();
    let result = py.allow_threads(|| nth_index(a_arr, n));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn norm_avg_index_1d_rs(a: PyReadonlyArray1<'_, bool>) -> PyResult<f64> {
    let a_cow = array1_as_slice_cow(&a);
    let a_slice = a_cow.as_ref();
    if a_slice.len() <= 1 || !a_slice.iter().any(|&v| v) {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    Ok(norm_avg_index_1d(a_slice))
}

#[pyfunction]
pub fn norm_avg_index_rs<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, bool>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let a_arr = a.as_array();
    if a_arr.nrows() <= 1 {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    for col in 0..a_arr.ncols() {
        if !a_arr.column(col).iter().any(|&v| v) {
            return Err(PyZeroDivisionError::new_err("division by zero"));
        }
    }
    let result = py.allow_threads(|| norm_avg_index(a_arr));
    Ok(PyArray1::from_vec_bound(py, result))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(clean_enex_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(clean_enex_rs, m)?)?;
    m.add_function(wrap_pyfunction!(between_ranges_rs, m)?)?;
    m.add_function(wrap_pyfunction!(between_two_ranges_rs, m)?)?;
    m.add_function(wrap_pyfunction!(partition_ranges_rs, m)?)?;
    m.add_function(wrap_pyfunction!(between_partition_ranges_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sig_pos_rank_rs, m)?)?;
    m.add_function(wrap_pyfunction!(part_pos_rank_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nth_index_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(nth_index_rs, m)?)?;
    m.add_function(wrap_pyfunction!(norm_avg_index_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(norm_avg_index_rs, m)?)?;
    Ok(())
}
