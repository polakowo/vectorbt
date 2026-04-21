// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use crate::generic::{array1_as_slice_cow, FlexArray, RangeRecord, RANGE_CLOSED, RANGE_OPEN};
use ndarray::{Array2, ArrayView2};
use numpy::{
    PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn, PyReadwriteArray2,
};
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use rand::seq::index::sample;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

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

    for col in 0..ncols {
        let mut phase = -1i64;
        for row in 0..nrows {
            let entry = entries[[row, col]];
            let exit = exits[[row, col]];
            if entry && exit {
                continue;
            }
            if entry && (phase == -1 || phase == 0) {
                phase = 1;
                entries_out[[row, col]] = true;
            }
            if exit && ((!entry_first && phase == -1) || phase == 1) {
                phase = 0;
                exits_out[[row, col]] = true;
            }
        }
    }

    (entries_out, exits_out)
}

pub(crate) fn generate_rand<R: Rng + ?Sized>(
    nrows: usize,
    ncols: usize,
    n: &[i64],
    rng: &mut R,
) -> Array2<bool> {
    let mut out = Array2::<bool>::from_elem((nrows, ncols), false);
    for col in 0..ncols {
        let size = n[col].min(nrows as i64) as usize;
        if size == 0 {
            continue;
        }
        let indices = sample(rng, nrows, size);
        for idx in indices.iter() {
            out[[idx, col]] = true;
        }
    }
    out
}

pub(crate) fn generate_rand_by_prob<R: Rng + ?Sized>(
    nrows: usize,
    ncols: usize,
    prob: &FlexArray<'_, f64>,
    pick_first: bool,
    rng: &mut R,
) -> Array2<bool> {
    let mut out = Array2::<bool>::from_elem((nrows, ncols), false);
    if let Some((prob_src, prob_cols)) = prob.as_full_2d() {
        for col in 0..ncols {
            for row in 0..nrows {
                if rng.gen::<f64>() < unsafe { *prob_src.get_unchecked(row * prob_cols + col) } {
                    out[[row, col]] = true;
                    if pick_first {
                        break;
                    }
                }
            }
        }
        return out;
    }
    for col in 0..ncols {
        for row in 0..nrows {
            if rng.gen::<f64>() < prob.get(row, col) {
                out[[row, col]] = true;
                if pick_first {
                    break;
                }
            }
        }
    }
    out
}

fn next_true_in_col(
    a: ArrayView2<'_, bool>,
    col: usize,
    start: usize,
    nrows: usize,
) -> Option<usize> {
    (start..nrows).find(|&row| a[[row, col]])
}

pub(crate) fn generate_rand_ex<R: Rng + ?Sized>(
    entries: ArrayView2<'_, bool>,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    rng: &mut R,
) -> Array2<bool> {
    let (nrows, ncols) = entries.dim();
    let mut exits = Array2::<bool>::from_elem((nrows, ncols), false);
    for col in 0..ncols {
        let mut last_exit_i: i64 = -1;
        let mut entry_i_opt = next_true_in_col(entries, col, 0, nrows);
        while let Some(entry_i) = entry_i_opt {
            let next_entry_i = next_true_in_col(entries, col, entry_i + 1, nrows);
            if !(skip_until_exit && (entry_i as i64) <= last_exit_i) {
                let from_i = entry_i + wait;
                let to_i = if until_next {
                    next_entry_i.unwrap_or(nrows)
                } else {
                    nrows
                };
                if to_i > from_i {
                    let exit_i = from_i + rng.gen_range(0..to_i - from_i);
                    exits[[exit_i, col]] = true;
                    last_exit_i = exit_i as i64;
                }
            }
            entry_i_opt = next_entry_i;
        }
    }
    exits
}

pub(crate) fn generate_rand_ex_by_prob<R: Rng + ?Sized>(
    entries: ArrayView2<'_, bool>,
    prob: &FlexArray<'_, f64>,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    rng: &mut R,
) -> Array2<bool> {
    let (nrows, ncols) = entries.dim();
    let mut exits = Array2::<bool>::from_elem((nrows, ncols), false);
    if let Some((prob_src, prob_cols)) = prob.as_full_2d() {
        for col in 0..ncols {
            let mut last_exit_i: i64 = -1;
            let mut entry_i_opt = next_true_in_col(entries, col, 0, nrows);
            while let Some(entry_i) = entry_i_opt {
                let next_entry_i = next_true_in_col(entries, col, entry_i + 1, nrows);
                if !(skip_until_exit && (entry_i as i64) <= last_exit_i) {
                    let from_i = entry_i + wait;
                    let to_i = if until_next {
                        next_entry_i.unwrap_or(nrows)
                    } else {
                        nrows
                    };
                    if to_i > from_i {
                        for idx in from_i..to_i {
                            if rng.gen::<f64>()
                                < unsafe { *prob_src.get_unchecked(idx * prob_cols + col) }
                            {
                                exits[[idx, col]] = true;
                                last_exit_i = idx as i64;
                                break;
                            }
                        }
                    }
                }
                entry_i_opt = next_entry_i;
            }
        }
        return exits;
    }

    for col in 0..ncols {
        let mut last_exit_i: i64 = -1;
        let mut entry_i_opt = next_true_in_col(entries, col, 0, nrows);
        while let Some(entry_i) = entry_i_opt {
            let next_entry_i = next_true_in_col(entries, col, entry_i + 1, nrows);
            if !(skip_until_exit && (entry_i as i64) <= last_exit_i) {
                let from_i = entry_i + wait;
                let to_i = if until_next {
                    next_entry_i.unwrap_or(nrows)
                } else {
                    nrows
                };
                if to_i > from_i {
                    for idx in from_i..to_i {
                        if rng.gen::<f64>() < prob.get(idx, col) {
                            exits[[idx, col]] = true;
                            last_exit_i = idx as i64;
                            break;
                        }
                    }
                }
            }
            entry_i_opt = next_entry_i;
        }
    }
    exits
}

fn uniform_summing_to_one<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Vec<f64> {
    let mut rand_floats = vec![0.0f64; n + 1];
    rand_floats[0] = 0.0;
    rand_floats[1] = 1.0;
    for slot in rand_floats.iter_mut().take(n + 1).skip(2) {
        *slot = rng.gen::<f64>();
    }
    rand_floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut diffs = vec![0.0f64; n];
    for i in 0..n {
        diffs[i] = rand_floats[i + 1] - rand_floats[i];
    }
    diffs
}

fn rescale_float_to_int<R: Rng + ?Sized>(
    floats: &[f64],
    int_range: (f64, f64),
    total: f64,
    rng: &mut R,
) -> Vec<i64> {
    let delta = int_range.1 - int_range.0;
    let mut ints: Vec<i64> = floats
        .iter()
        .map(|&f| (delta * f + int_range.0).floor() as i64)
        .collect();
    let sum: i64 = ints.iter().sum();
    let leftover = total as i64 - sum;
    if leftover > 0 && !ints.is_empty() {
        for _ in 0..leftover {
            let idx = rng.gen_range(0..ints.len());
            ints[idx] += 1;
        }
    }
    ints
}

pub(crate) fn generate_rand_enex<R: Rng + ?Sized>(
    nrows: usize,
    ncols: usize,
    n: &[i64],
    entry_wait: usize,
    exit_wait: usize,
    rng: &mut R,
) -> PyResult<(Array2<bool>, Array2<bool>)> {
    if entry_wait == 0 && exit_wait == 0 {
        return Err(PyValueError::new_err(
            "entry_wait and exit_wait cannot be both 0",
        ));
    }
    let mut entries = Array2::<bool>::from_elem((nrows, ncols), false);
    let mut exits = Array2::<bool>::from_elem((nrows, ncols), false);

    if entry_wait == 1 && exit_wait == 1 {
        let doubled: Vec<i64> = n.iter().map(|&x| x * 2).collect();
        let both = generate_rand(nrows, ncols, &doubled, rng);
        for col in 0..ncols {
            let mut k = 0usize;
            for row in 0..nrows {
                if both[[row, col]] {
                    if k % 2 == 0 {
                        entries[[row, col]] = true;
                    } else {
                        exits[[row, col]] = true;
                    }
                    k += 1;
                }
            }
        }
    } else {
        for col in 0..ncols {
            let col_n = n[col];
            if col_n == 1 {
                if nrows <= exit_wait {
                    return Err(PyValueError::new_err("Cannot fit entry before exit_wait"));
                }
                let entry_idx = rng.gen_range(0..(nrows - exit_wait));
                entries[[entry_idx, col]] = true;
            } else if col_n > 1 {
                let col_n_usize = col_n as usize;
                let min_range = entry_wait + exit_wait;
                let min_total_range = min_range * (col_n_usize - 1);
                if nrows < min_total_range + exit_wait + 1 {
                    return Err(PyValueError::new_err(
                        "Cannot take a larger sample than population",
                    ));
                }
                let max_free_space = nrows - min_total_range - 1;
                let free_space_pre = max_free_space.min(3 * nrows / (col_n_usize + 1));
                let free_space = if free_space_pre > exit_wait {
                    free_space_pre - exit_wait
                } else {
                    0
                };

                let rand_floats = uniform_summing_to_one(6, rng);
                let chosen_spaces = rescale_float_to_int(
                    &rand_floats,
                    (0.0, free_space as f64),
                    free_space as f64,
                    rng,
                );
                let first_idx = chosen_spaces[0] as usize;
                let tail_sum: i64 = chosen_spaces[chosen_spaces.len() - 2..].iter().sum();
                let last_idx = nrows as i64 - tail_sum - exit_wait as i64 - 1;
                let last_idx = last_idx as usize;

                let total_range = last_idx as i64 - first_idx as i64;
                let max_range = total_range - (col_n as i64 - 2) * min_range as i64;

                let rand_floats = uniform_summing_to_one(col_n_usize - 1, rng);
                let chosen_ranges = rescale_float_to_int(
                    &rand_floats,
                    (min_range as f64, max_range as f64),
                    total_range as f64,
                    rng,
                );

                let mut entry_idx = first_idx;
                entries[[entry_idx, col]] = true;
                for &r in &chosen_ranges {
                    entry_idx += r as usize;
                    entries[[entry_idx, col]] = true;
                }
            }
        }
        // Generate exits
        for col in 0..ncols {
            let mut entry_i_opt = next_true_in_col(entries.view(), col, 0, nrows);
            while let Some(entry_idx) = entry_i_opt {
                let next_entry_i = next_true_in_col(entries.view(), col, entry_idx + 1, nrows);
                let entry_i = entry_idx + exit_wait;
                let exit_i = if let Some(next_idx) = next_entry_i {
                    next_idx - entry_wait
                } else {
                    nrows - 1
                };
                let i = rng.gen_range(0..(exit_i - entry_i + 1));
                exits[[entry_i + i, col]] = true;
                entry_i_opt = next_entry_i;
            }
        }
    }
    Ok((entries, exits))
}

pub(crate) fn generate_rand_enex_by_prob<'a, R: Rng + ?Sized>(
    nrows: usize,
    ncols: usize,
    entry_prob: &FlexArray<'a, f64>,
    exit_prob: &FlexArray<'a, f64>,
    entry_wait: usize,
    exit_wait: usize,
    entry_pick_first: bool,
    exit_pick_first: bool,
    rng: &mut R,
) -> PyResult<(Array2<bool>, Array2<bool>)> {
    let mut entries = Array2::<bool>::from_elem((nrows, ncols), false);
    let mut exits = Array2::<bool>::from_elem((nrows, ncols), false);
    if let (Some((entry_src, entry_cols)), Some((exit_src, exit_cols))) =
        (entry_prob.as_full_2d(), exit_prob.as_full_2d())
    {
        for col in 0..ncols {
            let mut prev_prev_i: i64 = -2;
            let mut prev_i: i64 = -1;
            let mut i = 0usize;
            loop {
                let is_entry = i % 2 == 0;
                let from_i_raw: i64 = if is_entry {
                    if i == 0 {
                        0
                    } else {
                        prev_i + entry_wait as i64
                    }
                } else {
                    prev_i + exit_wait as i64
                };
                if from_i_raw >= nrows as i64 {
                    break;
                }
                let from_i = from_i_raw as usize;
                let to_i = nrows;
                let (prob_src, prob_cols, pick_first) = if is_entry {
                    (entry_src, entry_cols, entry_pick_first)
                } else {
                    (exit_src, exit_cols, exit_pick_first)
                };
                let mut first_i: Option<usize> = None;
                let mut last_i: Option<usize> = None;
                let mut hits: Vec<usize> = Vec::new();
                for idx in from_i..to_i {
                    if rng.gen::<f64>()
                        < unsafe { *prob_src.get_unchecked(idx * prob_cols + col) }
                    {
                        if first_i.is_none() {
                            first_i = Some(idx);
                        }
                        if !pick_first {
                            hits.push(idx);
                        }
                        last_i = Some(idx);
                        if pick_first {
                            break;
                        }
                    }
                }
                let first_i = match first_i {
                    Some(v) => v,
                    None => break,
                };
                if first_i as i64 == prev_i && prev_i == prev_prev_i {
                    return Err(PyValueError::new_err("Infinite loop detected"));
                }
                let target: &mut Array2<bool> = if is_entry { &mut entries } else { &mut exits };
                if pick_first {
                    target[[first_i, col]] = true;
                    prev_prev_i = prev_i;
                    prev_i = first_i as i64;
                } else {
                    for idx in &hits {
                        target[[*idx, col]] = true;
                    }
                    prev_prev_i = prev_i;
                    prev_i = last_i.unwrap() as i64;
                }
                i += 1;
            }
        }
        return Ok((entries, exits));
    }

    for col in 0..ncols {
        let mut prev_prev_i: i64 = -2;
        let mut prev_i: i64 = -1;
        let mut i = 0usize;
        loop {
            let is_entry = i % 2 == 0;
            let from_i_raw: i64 = if is_entry {
                if i == 0 {
                    0
                } else {
                    prev_i + entry_wait as i64
                }
            } else {
                prev_i + exit_wait as i64
            };
            if from_i_raw >= nrows as i64 {
                break;
            }
            let from_i = from_i_raw as usize;
            let to_i = nrows;
            let (prob_view, pick_first) = if is_entry {
                (entry_prob, entry_pick_first)
            } else {
                (exit_prob, exit_pick_first)
            };
            let mut first_i: Option<usize> = None;
            let mut last_i: Option<usize> = None;
            let mut hits: Vec<usize> = Vec::new();
            for idx in from_i..to_i {
                if rng.gen::<f64>() < prob_view.get(idx, col) {
                    if first_i.is_none() {
                        first_i = Some(idx);
                    }
                    if !pick_first {
                        hits.push(idx);
                    }
                    last_i = Some(idx);
                    if pick_first {
                        break;
                    }
                }
            }
            let first_i = match first_i {
                Some(v) => v,
                None => break,
            };
            if first_i as i64 == prev_i && prev_i == prev_prev_i {
                return Err(PyValueError::new_err("Infinite loop detected"));
            }
            let target: &mut Array2<bool> = if is_entry { &mut entries } else { &mut exits };
            if pick_first {
                target[[first_i, col]] = true;
                prev_prev_i = prev_i;
                prev_i = first_i as i64;
            } else {
                for idx in &hits {
                    target[[*idx, col]] = true;
                }
                prev_prev_i = prev_i;
                prev_i = last_i.unwrap() as i64;
            }
            i += 1;
        }
    }

    Ok((entries, exits))
}

/// Scan for the first stop hit within a bar range, mirroring `stop_choice_nb`.
///
/// Returns the first index where the stop price is crossed, or `None` if no hit.
/// When `pick_first` is false, this still returns the first hit; callers that
/// need all hits should iterate manually. The algorithm tracks the trailing
/// extremes across the range using `init_ts` seeded at `from_i - wait`.
fn stop_choice_first(
    from_i: usize,
    to_i: usize,
    col: usize,
    ts: ArrayView2<'_, f64>,
    stop: &FlexArray<'_, f64>,
    trailing: &FlexArray<'_, bool>,
    wait: usize,
) -> Option<usize> {
    let init_i = from_i as i64 - wait as i64;
    let init_i = if init_i < 0 { 0usize } else { init_i as usize };
    let init_ts = ts[[init_i, col]];
    let init_stop = stop.get(init_i, col);
    let init_trailing = trailing.get(init_i, col);
    let mut max_high = init_ts;
    let mut min_low = init_ts;

    for i in from_i..to_i {
        let curr_stop_price = if !init_stop.is_nan() {
            if init_trailing {
                if init_stop >= 0.0 {
                    min_low * (1.0 + init_stop.abs())
                } else {
                    max_high * (1.0 - init_stop.abs())
                }
            } else {
                init_ts * (1.0 + init_stop)
            }
        } else {
            f64::NAN
        };
        let curr_ts = ts[[i, col]];
        if !init_stop.is_nan() {
            let exit_signal = if init_stop >= 0.0 {
                curr_ts >= curr_stop_price
            } else {
                curr_ts <= curr_stop_price
            };
            if exit_signal {
                return Some(i);
            }
        }
        if init_trailing {
            if curr_ts < min_low {
                min_low = curr_ts;
            } else if curr_ts > max_high {
                max_high = curr_ts;
            }
        }
    }
    None
}

/// Collect **all** stop hits within `[from_i, to_i)`, for the `pick_first=false` path.
///
/// Unlike `stop_choice_first`, this walks the whole range and appends every bar
/// whose price crosses the current stop threshold (re-evaluated each bar).
fn stop_choice_all(
    from_i: usize,
    to_i: usize,
    col: usize,
    ts: ArrayView2<'_, f64>,
    stop: &FlexArray<'_, f64>,
    trailing: &FlexArray<'_, bool>,
    wait: usize,
) -> Vec<usize> {
    let init_i = from_i as i64 - wait as i64;
    let init_i = if init_i < 0 { 0usize } else { init_i as usize };
    let init_ts = ts[[init_i, col]];
    let init_stop = stop.get(init_i, col);
    let init_trailing = trailing.get(init_i, col);
    let mut max_high = init_ts;
    let mut min_low = init_ts;
    let mut out = Vec::new();

    for i in from_i..to_i {
        let curr_stop_price = if !init_stop.is_nan() {
            if init_trailing {
                if init_stop >= 0.0 {
                    min_low * (1.0 + init_stop.abs())
                } else {
                    max_high * (1.0 - init_stop.abs())
                }
            } else {
                init_ts * (1.0 + init_stop)
            }
        } else {
            f64::NAN
        };
        let curr_ts = ts[[i, col]];
        if !init_stop.is_nan() {
            let exit_signal = if init_stop >= 0.0 {
                curr_ts >= curr_stop_price
            } else {
                curr_ts <= curr_stop_price
            };
            if exit_signal {
                out.push(i);
            }
        }
        if init_trailing {
            if curr_ts < min_low {
                min_low = curr_ts;
            } else if curr_ts > max_high {
                max_high = curr_ts;
            }
        }
    }
    out
}

pub(crate) fn generate_stop_ex(
    entries: ArrayView2<'_, bool>,
    ts: ArrayView2<'_, f64>,
    stop: &FlexArray<'_, f64>,
    trailing: &FlexArray<'_, bool>,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    pick_first: bool,
) -> Array2<bool> {
    let (nrows, ncols) = entries.dim();
    let mut exits = Array2::<bool>::from_elem((nrows, ncols), false);

    for col in 0..ncols {
        let mut last_exit_i: i64 = -1;
        let mut entry_i_opt = next_true_in_col(entries, col, 0, nrows);
        while let Some(entry_i) = entry_i_opt {
            let next_entry_i = next_true_in_col(entries, col, entry_i + 1, nrows);
            if !(skip_until_exit && (entry_i as i64) <= last_exit_i) {
                let from_i = entry_i + wait;
                let to_i = if until_next {
                    next_entry_i.unwrap_or(nrows)
                } else {
                    nrows
                };
                if to_i > from_i {
                    if pick_first {
                        if let Some(idx) =
                            stop_choice_first(from_i, to_i, col, ts, stop, trailing, wait)
                        {
                            exits[[idx, col]] = true;
                            last_exit_i = idx as i64;
                        }
                    } else {
                        let hits = stop_choice_all(from_i, to_i, col, ts, stop, trailing, wait);
                        if !hits.is_empty() {
                            for &idx in &hits {
                                exits[[idx, col]] = true;
                            }
                            last_exit_i = *hits.last().unwrap() as i64;
                        }
                    }
                }
            }
            entry_i_opt = next_entry_i;
        }
    }
    exits
}

pub(crate) fn generate_stop_enex(
    entries: ArrayView2<'_, bool>,
    ts: ArrayView2<'_, f64>,
    stop: &FlexArray<'_, f64>,
    trailing: &FlexArray<'_, bool>,
    entry_wait: usize,
    exit_wait: usize,
    pick_first: bool,
) -> PyResult<(Array2<bool>, Array2<bool>)> {
    if entry_wait == 0 && exit_wait == 0 {
        return Err(PyValueError::new_err(
            "entry_wait and exit_wait cannot be both 0",
        ));
    }
    let (nrows, ncols) = entries.dim();
    let mut new_entries = Array2::<bool>::from_elem((nrows, ncols), false);
    let mut exits = Array2::<bool>::from_elem((nrows, ncols), false);

    for col in 0..ncols {
        let mut prev_prev_i: i64 = -2;
        let mut prev_i: i64 = -1;
        let mut i = 0usize;
        loop {
            let is_entry = i % 2 == 0;
            let from_i_raw: i64 = if is_entry {
                if i == 0 {
                    0
                } else {
                    prev_i + entry_wait as i64
                }
            } else {
                prev_i + exit_wait as i64
            };
            if from_i_raw >= nrows as i64 {
                break;
            }
            let from_i = from_i_raw.max(0) as usize;
            let to_i = nrows;

            let first_i: Option<usize> = if is_entry {
                // first_choice_nb: first `true` in entries[from_i..to_i, col]
                (from_i..to_i).find(|&r| entries[[r, col]])
            } else {
                stop_choice_first(from_i, to_i, col, ts, stop, trailing, exit_wait)
            };
            let first_i = match first_i {
                Some(v) => v,
                None => break,
            };
            if first_i as i64 == prev_i && prev_i == prev_prev_i {
                return Err(PyValueError::new_err("Infinite loop detected"));
            }

            let pick_first_here = if is_entry { true } else { pick_first };
            let target: &mut Array2<bool> = if is_entry {
                &mut new_entries
            } else {
                &mut exits
            };

            if pick_first_here {
                if first_i >= to_i {
                    return Err(PyValueError::new_err("First index is out of bounds"));
                }
                target[[first_i, col]] = true;
                prev_prev_i = prev_i;
                prev_i = first_i as i64;
            } else {
                // Only possible on the exit path when pick_first=false
                let hits = stop_choice_all(from_i, to_i, col, ts, stop, trailing, exit_wait);
                if hits.is_empty() {
                    break;
                }
                let last_i = *hits.last().unwrap();
                if last_i >= to_i {
                    return Err(PyValueError::new_err("Last index is out of bounds"));
                }
                for &idx in &hits {
                    target[[idx, col]] = true;
                }
                prev_prev_i = prev_i;
                prev_i = last_i as i64;
            }
            i += 1;
        }
    }
    Ok((new_entries, exits))
}

// Stop type codes matching `vectorbt.signals.enums.StopType`.
const STOP_TYPE_STOP_LOSS: i64 = 0;
const STOP_TYPE_TRAIL_STOP: i64 = 1;
const STOP_TYPE_TAKE_PROFIT: i64 = 2;

/// OHLC stop-choice scan mirroring `ohlc_stop_choice_nb`.
///
/// Walks bars in `[from_i, to_i)`, evaluating (trailing) stop-loss and take-profit
/// levels against the full OHLC bar, tracking trailing extremes. Writes the hit
/// price and stop-type code into `stop_price_out`/`stop_type_out` at each exit
/// bar, and returns the list of exit indices. When `pick_first=true`, returns
/// after the first hit.
fn ohlc_stop_choice(
    from_i: usize,
    to_i: usize,
    col: usize,
    open: ArrayView2<'_, f64>,
    high: ArrayView2<'_, f64>,
    low: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
    stop_price_out: &mut ndarray::ArrayViewMut2<'_, f64>,
    stop_type_out: &mut ndarray::ArrayViewMut2<'_, i64>,
    sl_stop: &FlexArray<'_, f64>,
    sl_trail: &FlexArray<'_, bool>,
    tp_stop: &FlexArray<'_, f64>,
    reverse: &FlexArray<'_, bool>,
    is_open_safe: bool,
    wait: usize,
    pick_first: bool,
) -> PyResult<Vec<usize>> {
    let init_i_i: i64 = from_i as i64 - wait as i64;
    let init_i = if init_i_i < 0 {
        0usize
    } else {
        init_i_i as usize
    };
    let init_open = open[[init_i, col]];
    let init_sl_stop = sl_stop.get(init_i, col);
    if !init_sl_stop.is_nan() && init_sl_stop < 0.0 {
        return Err(PyValueError::new_err("Stop value must be 0 or greater"));
    }
    let init_sl_trail = sl_trail.get(init_i, col);
    let init_tp_stop = tp_stop.get(init_i, col);
    if !init_tp_stop.is_nan() && init_tp_stop < 0.0 {
        return Err(PyValueError::new_err("Stop value must be 0 or greater"));
    }
    let init_reverse = reverse.get(init_i, col);
    let mut max_p = init_open;
    let mut min_p = init_open;
    let mut out = Vec::new();

    for i in from_i..to_i {
        let mut curr_open = open[[i, col]];
        let _high = high[[i, col]];
        let _low = low[[i, col]];
        let curr_close = close[[i, col]];
        if curr_open.is_nan() {
            curr_open = curr_close;
        }
        let curr_low_raw = if _low.is_nan() {
            curr_open.min(curr_close)
        } else {
            _low
        };
        let curr_high_raw = if _high.is_nan() {
            curr_open.max(curr_close)
        } else {
            _high
        };

        let curr_sl_stop_price = if !init_sl_stop.is_nan() {
            if init_sl_trail {
                if init_reverse {
                    min_p * (1.0 + init_sl_stop)
                } else {
                    max_p * (1.0 - init_sl_stop)
                }
            } else if init_reverse {
                init_open * (1.0 + init_sl_stop)
            } else {
                init_open * (1.0 - init_sl_stop)
            }
        } else {
            f64::NAN
        };
        let curr_tp_stop_price = if !init_tp_stop.is_nan() {
            if init_reverse {
                init_open * (1.0 - init_tp_stop)
            } else {
                init_open * (1.0 + init_tp_stop)
            }
        } else {
            f64::NAN
        };

        // is_open_safe means open happens at or before the bar open price,
        // so we can use the full bar range; otherwise only close is safe.
        let (curr_high, curr_low) = if i > init_i || is_open_safe {
            (curr_high_raw, curr_low_raw)
        } else {
            (curr_close, curr_close)
        };

        let mut exit_signal = false;
        if !init_sl_stop.is_nan() {
            let hit = if !init_reverse {
                curr_low <= curr_sl_stop_price
            } else {
                curr_high >= curr_sl_stop_price
            };
            if hit {
                exit_signal = true;
                stop_price_out[[i, col]] = curr_sl_stop_price;
                stop_type_out[[i, col]] = if init_sl_trail {
                    STOP_TYPE_TRAIL_STOP
                } else {
                    STOP_TYPE_STOP_LOSS
                };
            }
        }
        if !exit_signal && !init_tp_stop.is_nan() {
            let hit = if !init_reverse {
                curr_high >= curr_tp_stop_price
            } else {
                curr_low <= curr_tp_stop_price
            };
            if hit {
                exit_signal = true;
                stop_price_out[[i, col]] = curr_tp_stop_price;
                stop_type_out[[i, col]] = STOP_TYPE_TAKE_PROFIT;
            }
        }
        if exit_signal {
            out.push(i);
            if pick_first {
                return Ok(out);
            }
        }
        if init_sl_trail {
            if curr_low < min_p {
                min_p = curr_low;
            }
            if curr_high > max_p {
                max_p = curr_high;
            }
        }
    }
    Ok(out)
}

pub(crate) fn generate_ohlc_stop_ex(
    entries: ArrayView2<'_, bool>,
    open: ArrayView2<'_, f64>,
    high: ArrayView2<'_, f64>,
    low: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
    stop_price_out: &mut ndarray::ArrayViewMut2<'_, f64>,
    stop_type_out: &mut ndarray::ArrayViewMut2<'_, i64>,
    sl_stop: &FlexArray<'_, f64>,
    sl_trail: &FlexArray<'_, bool>,
    tp_stop: &FlexArray<'_, f64>,
    reverse: &FlexArray<'_, bool>,
    is_open_safe: bool,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    pick_first: bool,
) -> PyResult<Array2<bool>> {
    let (nrows, ncols) = entries.dim();
    let mut exits = Array2::<bool>::from_elem((nrows, ncols), false);

    for col in 0..ncols {
        let mut last_exit_i: i64 = -1;
        let mut entry_i_opt = next_true_in_col(entries, col, 0, nrows);
        while let Some(entry_i) = entry_i_opt {
            let next_entry_i = next_true_in_col(entries, col, entry_i + 1, nrows);
            if !(skip_until_exit && (entry_i as i64) <= last_exit_i) {
                let from_i = entry_i + wait;
                let to_i = if until_next {
                    next_entry_i.unwrap_or(nrows)
                } else {
                    nrows
                };
                if to_i > from_i {
                    let hits = ohlc_stop_choice(
                        from_i,
                        to_i,
                        col,
                        open,
                        high,
                        low,
                        close,
                        stop_price_out,
                        stop_type_out,
                        sl_stop,
                        sl_trail,
                        tp_stop,
                        reverse,
                        is_open_safe,
                        wait,
                        pick_first,
                    )?;
                    if hits.is_empty() {
                        entry_i_opt = next_entry_i;
                        continue;
                    }
                    if pick_first {
                        exits[[hits[0], col]] = true;
                        last_exit_i = hits[0] as i64;
                    } else {
                        for &idx in &hits {
                            exits[[idx, col]] = true;
                        }
                        last_exit_i = *hits.last().unwrap() as i64;
                    }
                }
            }
            entry_i_opt = next_entry_i;
        }
    }
    Ok(exits)
}

pub(crate) fn generate_ohlc_stop_enex(
    entries: ArrayView2<'_, bool>,
    open: ArrayView2<'_, f64>,
    high: ArrayView2<'_, f64>,
    low: ArrayView2<'_, f64>,
    close: ArrayView2<'_, f64>,
    stop_price_out: &mut ndarray::ArrayViewMut2<'_, f64>,
    stop_type_out: &mut ndarray::ArrayViewMut2<'_, i64>,
    sl_stop: &FlexArray<'_, f64>,
    sl_trail: &FlexArray<'_, bool>,
    tp_stop: &FlexArray<'_, f64>,
    reverse: &FlexArray<'_, bool>,
    is_open_safe: bool,
    entry_wait: usize,
    exit_wait: usize,
    pick_first: bool,
) -> PyResult<(Array2<bool>, Array2<bool>)> {
    if entry_wait == 0 && exit_wait == 0 {
        return Err(PyValueError::new_err(
            "entry_wait and exit_wait cannot be both 0",
        ));
    }
    let (nrows, ncols) = entries.dim();
    let mut new_entries = Array2::<bool>::from_elem((nrows, ncols), false);
    let mut exits = Array2::<bool>::from_elem((nrows, ncols), false);

    for col in 0..ncols {
        let mut prev_prev_i: i64 = -2;
        let mut prev_i: i64 = -1;
        let mut i = 0usize;
        loop {
            let is_entry = i % 2 == 0;
            let from_i_raw: i64 = if is_entry {
                if i == 0 {
                    0
                } else {
                    prev_i + entry_wait as i64
                }
            } else {
                prev_i + exit_wait as i64
            };
            if from_i_raw >= nrows as i64 {
                break;
            }
            let from_i = from_i_raw.max(0) as usize;
            let to_i = nrows;

            if is_entry {
                // first_choice_nb: first `true` in entries[from_i..to_i, col]
                let first_i = match (from_i..to_i).find(|&r| entries[[r, col]]) {
                    Some(v) => v,
                    None => break,
                };
                if first_i as i64 == prev_i && prev_i == prev_prev_i {
                    return Err(PyValueError::new_err("Infinite loop detected"));
                }
                if first_i >= to_i {
                    return Err(PyValueError::new_err("First index is out of bounds"));
                }
                new_entries[[first_i, col]] = true;
                prev_prev_i = prev_i;
                prev_i = first_i as i64;
            } else {
                let hits = ohlc_stop_choice(
                    from_i,
                    to_i,
                    col,
                    open,
                    high,
                    low,
                    close,
                    stop_price_out,
                    stop_type_out,
                    sl_stop,
                    sl_trail,
                    tp_stop,
                    reverse,
                    is_open_safe,
                    exit_wait,
                    pick_first,
                )?;
                if hits.is_empty() {
                    break;
                }
                let first_i = hits[0];
                if first_i as i64 == prev_i && prev_i == prev_prev_i {
                    return Err(PyValueError::new_err("Infinite loop detected"));
                }
                if pick_first {
                    if first_i >= to_i {
                        return Err(PyValueError::new_err("First index is out of bounds"));
                    }
                    exits[[first_i, col]] = true;
                    prev_prev_i = prev_i;
                    prev_i = first_i as i64;
                } else {
                    let last_i = *hits.last().unwrap();
                    if last_i >= to_i {
                        return Err(PyValueError::new_err("Last index is out of bounds"));
                    }
                    for &idx in &hits {
                        exits[[idx, col]] = true;
                    }
                    prev_prev_i = prev_i;
                    prev_i = last_i as i64;
                }
            }
            i += 1;
        }
    }
    Ok((new_entries, exits))
}

pub(crate) fn between_ranges(a: ArrayView2<'_, bool>) -> Vec<RangeRecord> {
    let (nrows, ncols) = a.dim();
    let mut out = Vec::<RangeRecord>::with_capacity(ncols);
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
    let mut out = Vec::<RangeRecord>::with_capacity(ncols);
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
    let mut out = Vec::<RangeRecord>::with_capacity(ncols);
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
    let mut out = Vec::<RangeRecord>::with_capacity(ncols);
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

    for col in 0..ncols {
        let mut sig_pos_temp = -1i64;
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
            let signal = a[[row, col]];
            if signal && (!after_false || false_seen) {
                if !in_partition {
                    part_start_i = row as i64;
                }
                in_partition = true;
                if reset_i as i64 > prev_part_end_i && reset_i.max(part_start_i as usize) == row {
                    sig_pos_temp = -1;
                } else if !allow_gaps && part_start_i == row as i64 {
                    sig_pos_temp = -1;
                }
                sig_pos_temp += 1;
                out[[row, col]] = sig_pos_temp;
            } else if !signal {
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

    for col in 0..ncols {
        let mut part_pos_temp = -1i64;
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
            let signal = a[[row, col]];
            if signal && (!after_false || false_seen) {
                if !in_partition {
                    part_start_i = row as i64;
                }
                in_partition = true;
                if reset_i as i64 > prev_part_end_i && reset_i.max(part_start_i as usize) == row {
                    part_pos_temp = 0;
                } else if part_start_i == row as i64 {
                    part_pos_temp += 1;
                }
                out[[row, col]] = part_pos_temp;
            } else if !signal {
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

    for col in 0..ncols {
        if n >= 0 {
            let mut found = -1i64;
            for row in 0..nrows {
                if a[[row, col]] {
                    found += 1;
                    if found == n {
                        out[col] = row as i64;
                        break;
                    }
                }
            }
        } else {
            let mut found = 0i64;
            for row in (0..nrows).rev() {
                if a[[row, col]] {
                    found -= 1;
                    if found == n {
                        out[col] = row as i64;
                        break;
                    }
                }
            }
        }
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

    if let Some(slice) = a.as_slice() {
        let mut sums = vec![0.0f64; ncols];
        let mut counts = vec![0usize; ncols];
        for row in 0..nrows {
            let base = row * ncols;
            for col in 0..ncols {
                if slice[base + col] {
                    sums[col] += row as f64;
                    counts[col] += 1;
                }
            }
        }
        for col in 0..ncols {
            if counts[col] > 0 {
                let mean_index = sums[col] / counts[col] as f64;
                out[col] = (2.0 * mean_index / (nrows as f64 - 1.0)) - 1.0;
            }
        }
        return out;
    }

    for col in 0..ncols {
        let mut sum = 0.0f64;
        let mut cnt = 0usize;
        for (row, &signal) in a.column(col).iter().enumerate() {
            if signal {
                sum += row as f64;
                cnt += 1;
            }
        }
        if cnt > 0 {
            let mean_index = sum / cnt as f64;
            out[col] = (2.0 * mean_index / (nrows as f64 - 1.0)) - 1.0;
        }
    }

    out
}

fn make_rng(seed: Option<u64>) -> ChaCha8Rng {
    match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::from_entropy(),
    }
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
#[pyo3(signature = (nrows, ncols, n, seed=None))]
pub fn generate_rand_rs<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    n: PyReadonlyArray1<'py, i64>,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let n_cow = array1_as_slice_cow(&n);
    let n_slice = n_cow.as_ref();
    if n_slice.len() != ncols {
        return Err(PyValueError::new_err("n must have length equal to ncols"));
    }
    let mut rng = make_rng(seed);
    let result = py.allow_threads(|| generate_rand(nrows, ncols, n_slice, &mut rng));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (nrows, ncols, prob, pick_first, seed=None, flex_2d=true))]
pub fn generate_rand_by_prob_rs<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    prob: PyReadonlyArrayDyn<'py, f64>,
    pick_first: bool,
    seed: Option<u64>,
    flex_2d: bool,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let prob_flex = FlexArray::from_pyarray("prob", &prob, nrows, ncols, flex_2d)?;
    let mut rng = make_rng(seed);
    let result =
        py.allow_threads(|| generate_rand_by_prob(nrows, ncols, &prob_flex, pick_first, &mut rng));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (entries, wait, until_next, skip_until_exit, seed=None))]
pub fn generate_rand_ex_rs<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'py, bool>,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    seed: Option<u64>,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let entries_arr = entries.as_array();
    let mut rng = make_rng(seed);
    let result = py.allow_threads(|| {
        generate_rand_ex(entries_arr, wait, until_next, skip_until_exit, &mut rng)
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (entries, prob, wait, until_next, skip_until_exit, seed=None, flex_2d=true))]
pub fn generate_rand_ex_by_prob_rs<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'py, bool>,
    prob: PyReadonlyArrayDyn<'py, f64>,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    seed: Option<u64>,
    flex_2d: bool,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let entries_arr = entries.as_array();
    let (nrows, ncols) = entries_arr.dim();
    let prob_flex = FlexArray::from_pyarray("prob", &prob, nrows, ncols, flex_2d)?;
    let mut rng = make_rng(seed);
    let result = py.allow_threads(|| {
        generate_rand_ex_by_prob(
            entries_arr,
            &prob_flex,
            wait,
            until_next,
            skip_until_exit,
            &mut rng,
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (nrows, ncols, n, entry_wait, exit_wait, seed=None))]
pub fn generate_rand_enex_rs<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    n: PyReadonlyArray1<'py, i64>,
    entry_wait: usize,
    exit_wait: usize,
    seed: Option<u64>,
) -> PyResult<(Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>)> {
    let n_cow = array1_as_slice_cow(&n);
    let n_slice = n_cow.as_ref();
    if n_slice.len() != ncols {
        return Err(PyValueError::new_err("n must have length equal to ncols"));
    }
    let mut rng = make_rng(seed);
    let (entries, exits) = py.allow_threads(|| {
        generate_rand_enex(nrows, ncols, n_slice, entry_wait, exit_wait, &mut rng)
    })?;
    Ok((
        PyArray2::from_owned_array_bound(py, entries),
        PyArray2::from_owned_array_bound(py, exits),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    nrows,
    ncols,
    entry_prob,
    exit_prob,
    entry_wait,
    exit_wait,
    entry_pick_first,
    exit_pick_first,
    seed=None,
    flex_2d=true,
))]
pub fn generate_rand_enex_by_prob_rs<'py>(
    py: Python<'py>,
    nrows: usize,
    ncols: usize,
    entry_prob: PyReadonlyArrayDyn<'py, f64>,
    exit_prob: PyReadonlyArrayDyn<'py, f64>,
    entry_wait: usize,
    exit_wait: usize,
    entry_pick_first: bool,
    exit_pick_first: bool,
    seed: Option<u64>,
    flex_2d: bool,
) -> PyResult<(Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>)> {
    if entry_wait == 0 && exit_wait == 0 {
        return Err(PyValueError::new_err(
            "entry_wait and exit_wait cannot be both 0",
        ));
    }
    let entry_prob_flex =
        FlexArray::from_pyarray("entry_prob", &entry_prob, nrows, ncols, flex_2d)?;
    let exit_prob_flex =
        FlexArray::from_pyarray("exit_prob", &exit_prob, nrows, ncols, flex_2d)?;
    let mut rng = make_rng(seed);
    let (entries, exits) = py.allow_threads(|| {
        generate_rand_enex_by_prob(
            nrows,
            ncols,
            &entry_prob_flex,
            &exit_prob_flex,
            entry_wait,
            exit_wait,
            entry_pick_first,
            exit_pick_first,
            &mut rng,
        )
    })?;
    Ok((
        PyArray2::from_owned_array_bound(py, entries),
        PyArray2::from_owned_array_bound(py, exits),
    ))
}

#[pyfunction]
#[pyo3(signature = (entries, ts, stop, trailing, wait, until_next, skip_until_exit, pick_first, flex_2d=true))]
pub fn generate_stop_ex_rs<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'py, bool>,
    ts: PyReadonlyArray2<'py, f64>,
    stop: PyReadonlyArrayDyn<'py, f64>,
    trailing: PyReadonlyArrayDyn<'py, bool>,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    pick_first: bool,
    flex_2d: bool,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let entries_arr = entries.as_array();
    let ts_arr = ts.as_array();
    let shape = entries_arr.dim();
    if ts_arr.dim() != shape {
        return Err(PyValueError::new_err("ts must match entries shape"));
    }
    let stop_flex = FlexArray::from_pyarray("stop", &stop, shape.0, shape.1, flex_2d)?;
    let trailing_flex = FlexArray::from_pyarray("trailing", &trailing, shape.0, shape.1, flex_2d)?;
    let result = py.allow_threads(|| {
        generate_stop_ex(
            entries_arr,
            ts_arr,
            &stop_flex,
            &trailing_flex,
            wait,
            until_next,
            skip_until_exit,
            pick_first,
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (entries, ts, stop, trailing, entry_wait, exit_wait, pick_first, flex_2d=true))]
pub fn generate_stop_enex_rs<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'py, bool>,
    ts: PyReadonlyArray2<'py, f64>,
    stop: PyReadonlyArrayDyn<'py, f64>,
    trailing: PyReadonlyArrayDyn<'py, bool>,
    entry_wait: usize,
    exit_wait: usize,
    pick_first: bool,
    flex_2d: bool,
) -> PyResult<(Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>)> {
    let entries_arr = entries.as_array();
    let ts_arr = ts.as_array();
    let shape = entries_arr.dim();
    if ts_arr.dim() != shape {
        return Err(PyValueError::new_err("ts must match entries shape"));
    }
    let stop_flex = FlexArray::from_pyarray("stop", &stop, shape.0, shape.1, flex_2d)?;
    let trailing_flex = FlexArray::from_pyarray("trailing", &trailing, shape.0, shape.1, flex_2d)?;
    let (new_entries, exits) = py.allow_threads(|| {
        generate_stop_enex(
            entries_arr,
            ts_arr,
            &stop_flex,
            &trailing_flex,
            entry_wait,
            exit_wait,
            pick_first,
        )
    })?;
    Ok((
        PyArray2::from_owned_array_bound(py, new_entries),
        PyArray2::from_owned_array_bound(py, exits),
    ))
}

#[pyfunction]
#[pyo3(signature = (
    entries,
    open,
    high,
    low,
    close,
    stop_price_out,
    stop_type_out,
    sl_stop,
    sl_trail,
    tp_stop,
    reverse,
    is_open_safe,
    wait,
    until_next,
    skip_until_exit,
    pick_first,
    flex_2d=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn generate_ohlc_stop_ex_rs<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'py, bool>,
    open: PyReadonlyArray2<'py, f64>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    mut stop_price_out: PyReadwriteArray2<'py, f64>,
    mut stop_type_out: PyReadwriteArray2<'py, i64>,
    sl_stop: PyReadonlyArrayDyn<'py, f64>,
    sl_trail: PyReadonlyArrayDyn<'py, bool>,
    tp_stop: PyReadonlyArrayDyn<'py, f64>,
    reverse: PyReadonlyArrayDyn<'py, bool>,
    is_open_safe: bool,
    wait: usize,
    until_next: bool,
    skip_until_exit: bool,
    pick_first: bool,
    flex_2d: bool,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let entries_arr = entries.as_array();
    let open_arr = open.as_array();
    let high_arr = high.as_array();
    let low_arr = low.as_array();
    let close_arr = close.as_array();
    let shape = entries_arr.dim();
    if open_arr.dim() != shape
        || high_arr.dim() != shape
        || low_arr.dim() != shape
        || close_arr.dim() != shape
    {
        return Err(PyValueError::new_err("OHLC inputs must match entries shape"));
    }
    let sl_stop_flex = FlexArray::from_pyarray("sl_stop", &sl_stop, shape.0, shape.1, flex_2d)?;
    let sl_trail_flex = FlexArray::from_pyarray("sl_trail", &sl_trail, shape.0, shape.1, flex_2d)?;
    let tp_stop_flex = FlexArray::from_pyarray("tp_stop", &tp_stop, shape.0, shape.1, flex_2d)?;
    let reverse_flex = FlexArray::from_pyarray("reverse", &reverse, shape.0, shape.1, flex_2d)?;
    let mut stop_price_view = stop_price_out.as_array_mut();
    let mut stop_type_view = stop_type_out.as_array_mut();
    if stop_price_view.dim() != shape || stop_type_view.dim() != shape {
        return Err(PyValueError::new_err(
            "Output arrays must match entries shape",
        ));
    }
    let result = py.allow_threads(|| {
        generate_ohlc_stop_ex(
            entries_arr,
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            &mut stop_price_view,
            &mut stop_type_view,
            &sl_stop_flex,
            &sl_trail_flex,
            &tp_stop_flex,
            &reverse_flex,
            is_open_safe,
            wait,
            until_next,
            skip_until_exit,
            pick_first,
        )
    })?;
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (
    entries,
    open,
    high,
    low,
    close,
    stop_price_out,
    stop_type_out,
    sl_stop,
    sl_trail,
    tp_stop,
    reverse,
    is_open_safe,
    entry_wait,
    exit_wait,
    pick_first,
    flex_2d=true,
))]
#[allow(clippy::too_many_arguments)]
pub fn generate_ohlc_stop_enex_rs<'py>(
    py: Python<'py>,
    entries: PyReadonlyArray2<'py, bool>,
    open: PyReadonlyArray2<'py, f64>,
    high: PyReadonlyArray2<'py, f64>,
    low: PyReadonlyArray2<'py, f64>,
    close: PyReadonlyArray2<'py, f64>,
    mut stop_price_out: PyReadwriteArray2<'py, f64>,
    mut stop_type_out: PyReadwriteArray2<'py, i64>,
    sl_stop: PyReadonlyArrayDyn<'py, f64>,
    sl_trail: PyReadonlyArrayDyn<'py, bool>,
    tp_stop: PyReadonlyArrayDyn<'py, f64>,
    reverse: PyReadonlyArrayDyn<'py, bool>,
    is_open_safe: bool,
    entry_wait: usize,
    exit_wait: usize,
    pick_first: bool,
    flex_2d: bool,
) -> PyResult<(Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<bool>>)> {
    let entries_arr = entries.as_array();
    let open_arr = open.as_array();
    let high_arr = high.as_array();
    let low_arr = low.as_array();
    let close_arr = close.as_array();
    let shape = entries_arr.dim();
    if open_arr.dim() != shape
        || high_arr.dim() != shape
        || low_arr.dim() != shape
        || close_arr.dim() != shape
    {
        return Err(PyValueError::new_err("OHLC inputs must match entries shape"));
    }
    let sl_stop_flex = FlexArray::from_pyarray("sl_stop", &sl_stop, shape.0, shape.1, flex_2d)?;
    let sl_trail_flex = FlexArray::from_pyarray("sl_trail", &sl_trail, shape.0, shape.1, flex_2d)?;
    let tp_stop_flex = FlexArray::from_pyarray("tp_stop", &tp_stop, shape.0, shape.1, flex_2d)?;
    let reverse_flex = FlexArray::from_pyarray("reverse", &reverse, shape.0, shape.1, flex_2d)?;
    let mut stop_price_view = stop_price_out.as_array_mut();
    let mut stop_type_view = stop_type_out.as_array_mut();
    if stop_price_view.dim() != shape || stop_type_view.dim() != shape {
        return Err(PyValueError::new_err(
            "Output arrays must match entries shape",
        ));
    }
    let (new_entries, exits) = py.allow_threads(|| {
        generate_ohlc_stop_enex(
            entries_arr,
            open_arr,
            high_arr,
            low_arr,
            close_arr,
            &mut stop_price_view,
            &mut stop_type_view,
            &sl_stop_flex,
            &sl_trail_flex,
            &tp_stop_flex,
            &reverse_flex,
            is_open_safe,
            entry_wait,
            exit_wait,
            pick_first,
        )
    })?;
    Ok((
        PyArray2::from_owned_array_bound(py, new_entries),
        PyArray2::from_owned_array_bound(py, exits),
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
    m.add_function(wrap_pyfunction!(generate_rand_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_rand_by_prob_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_rand_ex_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_rand_ex_by_prob_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_rand_enex_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_rand_enex_by_prob_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_stop_ex_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_stop_enex_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ohlc_stop_ex_rs, m)?)?;
    m.add_function(wrap_pyfunction!(generate_ohlc_stop_enex_rs, m)?)?;
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
