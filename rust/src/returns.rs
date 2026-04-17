// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use crate::generic::{
    apply_2d_by_col, array1_as_slice_cow, nanstd_1d, reduce_2d_by_col, validate_window,
};
use ndarray::{Array2, ArrayView2};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::{PyValueError, PyZeroDivisionError};
use pyo3::prelude::*;
use std::cmp::Ordering;

fn validate_same_shape_2d(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    name: &str,
) -> PyResult<()> {
    if a.dim() != b.dim() {
        return Err(PyValueError::new_err(format!(
            "{name} must have the same shape as input"
        )));
    }
    Ok(())
}

fn reduce_pair_2d_by_col<F>(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    mut kernel: F,
) -> Vec<f64>
where
    F: FnMut(&[f64], &[f64]) -> f64,
{
    let (nrows, ncols) = a.dim();
    let mut out = vec![f64::NAN; ncols];
    let mut a_buf = vec![0.0f64; nrows];
    let mut b_buf = vec![0.0f64; nrows];
    for col in 0..ncols {
        for (i, &v) in a.column(col).iter().enumerate() {
            a_buf[i] = v;
        }
        for (i, &v) in b.column(col).iter().enumerate() {
            b_buf[i] = v;
        }
        out[col] = kernel(&a_buf, &b_buf);
    }
    out
}

fn rolling_apply_2d_by_col<F>(
    a: ArrayView2<'_, f64>,
    window: usize,
    minp: usize,
    mut kernel: F,
) -> Array2<f64>
where
    F: FnMut(&[f64]) -> f64,
{
    let (nrows, ncols) = a.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let mut col_buf = vec![0.0f64; nrows];
    let mut nancnt_arr = vec![0usize; nrows];

    for col in 0..ncols {
        for (i, &value) in a.column(col).iter().enumerate() {
            col_buf[i] = value;
        }

        let mut nancnt = 0usize;
        for i in 0..nrows {
            if col_buf[i].is_nan() {
                nancnt += 1;
            }
            nancnt_arr[i] = nancnt;
            let valid_cnt = if i < window {
                i + 1 - nancnt
            } else {
                window - (nancnt - nancnt_arr[i - window])
            };
            if valid_cnt >= minp {
                let start = (i + 1).saturating_sub(window);
                out[[i, col]] = kernel(&col_buf[start..i + 1]);
            }
        }
    }

    out
}

fn rolling_apply_pair_2d_by_col<F>(
    a: ArrayView2<'_, f64>,
    b: ArrayView2<'_, f64>,
    window: usize,
    minp: usize,
    mut kernel: F,
) -> Array2<f64>
where
    F: FnMut(&[f64], &[f64]) -> f64,
{
    let (nrows, ncols) = a.dim();
    let mut out = Array2::<f64>::from_elem((nrows, ncols), f64::NAN);
    let mut a_buf = vec![0.0f64; nrows];
    let mut b_buf = vec![0.0f64; nrows];
    let mut nancnt_arr = vec![0usize; nrows];

    for col in 0..ncols {
        for (i, &value) in a.column(col).iter().enumerate() {
            a_buf[i] = value;
        }
        for (i, &value) in b.column(col).iter().enumerate() {
            b_buf[i] = value;
        }

        let mut nancnt = 0usize;
        for i in 0..nrows {
            if a_buf[i].is_nan() {
                nancnt += 1;
            }
            nancnt_arr[i] = nancnt;
            let valid_cnt = if i < window {
                i + 1 - nancnt
            } else {
                window - (nancnt - nancnt_arr[i - window])
            };
            if valid_cnt >= minp {
                let start = (i + 1).saturating_sub(window);
                out[[i, col]] = kernel(&a_buf[start..i + 1], &b_buf[start..i + 1]);
            }
        }
    }

    out
}

fn mean_strict(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut sum = 0.0;
    for &value in values {
        if value.is_nan() {
            return f64::NAN;
        }
        sum += value;
    }
    sum / values.len() as f64
}

fn percentile_unsorted(vals: &mut [f64], q: f64) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    if vals.len() == 1 {
        return vals[0];
    }
    let rank = (q / 100.0) * (vals.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    vals.select_nth_unstable_by(hi, |left, right| left.partial_cmp(right).unwrap());
    let hi_value = vals[hi];
    if lo == hi {
        hi_value
    } else {
        vals[..hi].select_nth_unstable_by(lo, |left, right| left.partial_cmp(right).unwrap());
        vals[lo] * (hi as f64 - rank) + hi_value * (rank - lo as f64)
    }
}

fn nanmean_shifted(values: &[f64], shift: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &value in values {
        let adj_value = value - shift;
        if !adj_value.is_nan() {
            sum += adj_value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn nanstd_shifted(values: &[f64], shift: f64, ddof: usize) -> f64 {
    let mean = nanmean_shifted(values, shift);
    if mean.is_nan() {
        return f64::NAN;
    }
    let mut count = 0usize;
    let mut sq = 0.0;
    for &value in values {
        let adj_value = value - shift;
        if !adj_value.is_nan() {
            count += 1;
            let diff = adj_value - mean;
            sq += diff * diff;
        }
    }
    if count <= ddof {
        f64::NAN
    } else {
        (sq / (count - ddof) as f64).sqrt()
    }
}

fn nanmean_pair_diff(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (&left, &right) in a.iter().zip(b.iter()) {
        let value = left - right;
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

fn nanstd_pair_diff(a: &[f64], b: &[f64], ddof: usize) -> f64 {
    let mean = nanmean_pair_diff(a, b);
    if mean.is_nan() {
        return f64::NAN;
    }
    let mut count = 0usize;
    let mut sq = 0.0;
    for (&left, &right) in a.iter().zip(b.iter()) {
        let value = left - right;
        if !value.is_nan() {
            count += 1;
            let diff = value - mean;
            sq += diff * diff;
        }
    }
    if count <= ddof {
        f64::NAN
    } else {
        (sq / (count - ddof) as f64).sqrt()
    }
}

fn nanmean_alpha_series(returns: &[f64], benchmark_rets: &[f64], beta: f64, risk_free: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for (&ret, &benchmark_ret) in returns.iter().zip(benchmark_rets.iter()) {
        let value = (ret - risk_free) - beta * (benchmark_ret - risk_free);
        if !value.is_nan() {
            sum += value;
            count += 1;
        }
    }
    if count == 0 {
        f64::NAN
    } else {
        sum / count as f64
    }
}

#[inline(always)]
pub(crate) fn get_return(input_value: f64, output_value: f64) -> f64 {
    if input_value == 0.0 {
        if output_value == 0.0 {
            return 0.0;
        }
        if output_value.is_nan() {
            return f64::NAN;
        }
        return f64::INFINITY.copysign(output_value);
    }
    let mut return_value = (output_value - input_value) / input_value;
    if input_value < 0.0 {
        return_value *= -1.0;
    }
    return_value
}

pub(crate) fn returns_1d(value: &[f64], init_value: f64) -> Vec<f64> {
    let mut out = vec![0.0f64; value.len()];
    let mut input_value = init_value;
    for (i, &output_value) in value.iter().enumerate() {
        out[i] = get_return(input_value, output_value);
        input_value = output_value;
    }
    out
}

pub(crate) fn returns_2d(value: ArrayView2<'_, f64>, init_value: &[f64]) -> Array2<f64> {
    let (nrows, ncols) = value.dim();
    if let Some(src) = value.as_slice() {
        let mut out = Array2::<f64>::zeros((nrows, ncols));
        let dst = out.as_slice_mut().expect("owned array must be sliceable");
        let mut input_values = init_value.to_vec();
        for row in 0..nrows {
            let row_start = row * ncols;
            for col in 0..ncols {
                let output_value = src[row_start + col];
                dst[row_start + col] = get_return(input_values[col], output_value);
                input_values[col] = output_value;
            }
        }
        return out;
    }
    let mut out = Array2::<f64>::zeros((nrows, ncols));
    for col in 0..ncols {
        let mut input_value = init_value[col];
        for row in 0..nrows {
            let output_value = value[[row, col]];
            out[[row, col]] = get_return(input_value, output_value);
            input_value = output_value;
        }
    }
    out
}

pub(crate) fn cum_returns_1d(returns: &[f64], start_value: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(returns.len());
    let mut cumprod = 1.0;
    for &ret in returns {
        if !ret.is_nan() {
            cumprod *= ret + 1.0;
        }
        out.push(if start_value == 0.0 {
            cumprod - 1.0
        } else {
            cumprod * start_value
        });
    }
    out
}

pub(crate) fn cum_returns_2d(returns: ArrayView2<'_, f64>, start_value: f64) -> Array2<f64> {
    let (nrows, ncols) = returns.dim();
    if let Some(src) = returns.as_slice() {
        let mut out = Array2::<f64>::zeros((nrows, ncols));
        let dst = out.as_slice_mut().expect("owned array must be sliceable");
        let mut cumprods = vec![1.0f64; ncols];
        for row in 0..nrows {
            let row_start = row * ncols;
            for col in 0..ncols {
                let ret = src[row_start + col];
                if !ret.is_nan() {
                    cumprods[col] *= ret + 1.0;
                }
                dst[row_start + col] = if start_value == 0.0 {
                    cumprods[col] - 1.0
                } else {
                    cumprods[col] * start_value
                };
            }
        }
        return out;
    }
    apply_2d_by_col(returns, |col| cum_returns_1d(col, start_value))
}

pub(crate) fn cum_returns_final_1d(returns: &[f64], start_value: f64) -> f64 {
    let mut out = 1.0;
    for &ret in returns {
        if !ret.is_nan() {
            out *= ret + 1.0;
        }
    }
    if start_value == 0.0 {
        out - 1.0
    } else {
        out * start_value
    }
}

pub(crate) fn cum_returns_final_2d(returns: ArrayView2<'_, f64>, start_value: f64) -> Vec<f64> {
    let (_, ncols) = returns.dim();
    if let Some(src) = returns.as_slice() {
        let mut out = vec![1.0f64; ncols];
        for row in src.chunks_exact(ncols) {
            for (col, &ret) in row.iter().enumerate() {
                if !ret.is_nan() {
                    out[col] *= ret + 1.0;
                }
            }
        }
        if start_value == 0.0 {
            for value in &mut out {
                *value -= 1.0;
            }
        } else {
            for value in &mut out {
                *value *= start_value;
            }
        }
        return out;
    }
    reduce_2d_by_col(returns, |col| cum_returns_final_1d(col, start_value))
}

pub(crate) fn annualized_return_1d(returns: &[f64], ann_factor: f64) -> f64 {
    let end_value = cum_returns_final_1d(returns, 1.0);
    end_value.powf(ann_factor / returns.len() as f64) - 1.0
}

pub(crate) fn annualized_return_2d(returns: ArrayView2<'_, f64>, ann_factor: f64) -> Vec<f64> {
    let (nrows, ncols) = returns.dim();
    if let Some(src) = returns.as_slice() {
        let mut products = vec![1.0f64; ncols];
        for row in src.chunks_exact(ncols) {
            for (col, &ret) in row.iter().enumerate() {
                if !ret.is_nan() {
                    products[col] *= ret + 1.0;
                }
            }
        }
        return products
            .iter()
            .map(|&product| annualized_return_from_product(product, nrows, ann_factor))
            .collect();
    }
    reduce_2d_by_col(returns, |col| annualized_return_1d(col, ann_factor))
}

fn annualized_return_from_product(product: f64, len: usize, ann_factor: f64) -> f64 {
    product.powf(ann_factor / len as f64) - 1.0
}

pub(crate) fn annualized_volatility_1d(
    returns: &[f64],
    ann_factor: f64,
    levy_alpha: f64,
    ddof: usize,
) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    nanstd_1d(returns, ddof) * ann_factor.powf(1.0 / levy_alpha)
}

pub(crate) fn annualized_volatility_2d(
    returns: ArrayView2<'_, f64>,
    ann_factor: f64,
    levy_alpha: f64,
    ddof: usize,
) -> Vec<f64> {
    let (nrows, ncols) = returns.dim();
    if let Some(src) = returns.as_slice() {
        let mut sums = vec![0.0f64; ncols];
        let mut counts = vec![0usize; ncols];
        for row in src.chunks_exact(ncols) {
            for (col, &ret) in row.iter().enumerate() {
                if !ret.is_nan() {
                    sums[col] += ret;
                    counts[col] += 1;
                }
            }
        }
        let mut sq = vec![0.0f64; ncols];
        for row in src.chunks_exact(ncols) {
            for (col, &ret) in row.iter().enumerate() {
                if !ret.is_nan() {
                    let diff = ret - sums[col] / counts[col] as f64;
                    sq[col] += diff * diff;
                }
            }
        }
        let ann_scale = ann_factor.powf(1.0 / levy_alpha);
        let mut out = vec![f64::NAN; ncols];
        for col in 0..ncols {
            if nrows >= 2 && counts[col] > ddof {
                out[col] = (sq[col] / (counts[col] - ddof) as f64).sqrt() * ann_scale;
            }
        }
        return out;
    }
    reduce_2d_by_col(returns, |col| {
        annualized_volatility_1d(col, ann_factor, levy_alpha, ddof)
    })
}

pub(crate) fn drawdown_1d(returns: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(returns.len());
    let mut cum_return = 100.0;
    let mut max_return = f64::NAN;
    for &ret in returns {
        if !ret.is_nan() {
            cum_return *= ret + 1.0;
        }
        if max_return.is_nan() || cum_return > max_return {
            max_return = cum_return;
        }
        out.push(cum_return / max_return - 1.0);
    }
    out
}

pub(crate) fn max_drawdown_1d(returns: &[f64]) -> f64 {
    let mut out = f64::INFINITY;
    let mut cum_return = 100.0;
    let mut max_return = f64::NAN;
    for &ret in returns {
        if !ret.is_nan() {
            cum_return *= ret + 1.0;
        }
        if max_return.is_nan() || cum_return > max_return {
            max_return = cum_return;
        }
        let drawdown = cum_return / max_return - 1.0;
        if drawdown.is_nan() {
            return f64::NAN;
        }
        if drawdown < out {
            out = drawdown;
        }
    }
    out
}

pub(crate) fn calmar_ratio_1d(returns: &[f64], ann_factor: f64) -> f64 {
    let max_drawdown = max_drawdown_1d(returns);
    if max_drawdown == 0.0 {
        return f64::NAN;
    }
    let annualized_return = annualized_return_1d(returns, ann_factor);
    annualized_return / max_drawdown.abs()
}

pub(crate) fn omega_ratio_1d(
    returns: &[f64],
    ann_factor: f64,
    risk_free: f64,
    required_return: f64,
) -> f64 {
    let return_threshold = if ann_factor == 1.0 {
        required_return
    } else if ann_factor <= -1.0 {
        return f64::NAN;
    } else {
        (1.0 + required_return).powf(1.0 / ann_factor) - 1.0
    };
    let mut numer = 0.0;
    let mut denom = 0.0;
    for &ret in returns {
        let adj = ret - risk_free - return_threshold;
        if adj > 0.0 {
            numer += adj;
        } else if adj < 0.0 {
            denom -= adj;
        }
    }
    if denom == 0.0 {
        return f64::INFINITY;
    }
    numer / denom
}

pub(crate) fn sharpe_ratio_1d(
    returns: &[f64],
    ann_factor: f64,
    risk_free: f64,
    ddof: usize,
) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let mean = nanmean_shifted(returns, risk_free);
    let std = nanstd_shifted(returns, risk_free, ddof);
    if std == 0.0 {
        return f64::INFINITY;
    }
    mean / std * ann_factor.sqrt()
}

pub(crate) fn sharpe_ratio_2d(
    returns: ArrayView2<'_, f64>,
    ann_factor: f64,
    risk_free: f64,
    ddof: usize,
) -> Vec<f64> {
    let (nrows, ncols) = returns.dim();
    if let Some(src) = returns.as_slice() {
        let mut sums = vec![0.0f64; ncols];
        let mut counts = vec![0usize; ncols];
        for row in src.chunks_exact(ncols) {
            for (col, &ret) in row.iter().enumerate() {
                let adj_ret = ret - risk_free;
                if !adj_ret.is_nan() {
                    sums[col] += adj_ret;
                    counts[col] += 1;
                }
            }
        }
        let mut sq = vec![0.0f64; ncols];
        for row in src.chunks_exact(ncols) {
            for (col, &ret) in row.iter().enumerate() {
                let adj_ret = ret - risk_free;
                if !adj_ret.is_nan() {
                    let diff = adj_ret - sums[col] / counts[col] as f64;
                    sq[col] += diff * diff;
                }
            }
        }
        let ann_sqrt = ann_factor.sqrt();
        let mut out = vec![f64::NAN; ncols];
        for col in 0..ncols {
            if nrows >= 2 && counts[col] > ddof {
                let mean = sums[col] / counts[col] as f64;
                let std = (sq[col] / (counts[col] - ddof) as f64).sqrt();
                out[col] = if std == 0.0 {
                    f64::INFINITY
                } else {
                    mean / std * ann_sqrt
                };
            }
        }
        return out;
    }
    reduce_2d_by_col(returns, |col| {
        sharpe_ratio_1d(col, ann_factor, risk_free, ddof)
    })
}

pub(crate) fn downside_risk_1d(returns: &[f64], ann_factor: f64, required_return: f64) -> f64 {
    let mut sum = 0.0;
    let mut count = 0usize;
    for &ret in returns {
        let mut adj_ret = ret - required_return;
        if adj_ret > 0.0 {
            adj_ret = 0.0;
        }
        if !adj_ret.is_nan() {
            sum += adj_ret * adj_ret;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    (sum / count as f64).sqrt() * ann_factor.sqrt()
}

pub(crate) fn sortino_ratio_1d(returns: &[f64], ann_factor: f64, required_return: f64) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let average_annualized_return = nanmean_shifted(returns, required_return) * ann_factor;
    let downside_risk = downside_risk_1d(returns, ann_factor, required_return);
    if downside_risk == 0.0 {
        return f64::INFINITY;
    }
    average_annualized_return / downside_risk
}

pub(crate) fn information_ratio_1d(returns: &[f64], benchmark_rets: &[f64], ddof: usize) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let mean = nanmean_pair_diff(returns, benchmark_rets);
    let std = nanstd_pair_diff(returns, benchmark_rets, ddof);
    if std == 0.0 {
        return f64::INFINITY;
    }
    mean / std
}

pub(crate) fn information_ratio_2d(
    returns: ArrayView2<'_, f64>,
    benchmark_rets: ArrayView2<'_, f64>,
    ddof: usize,
) -> Vec<f64> {
    let (nrows, ncols) = returns.dim();
    if let (Some(returns_src), Some(benchmark_src)) =
        (returns.as_slice(), benchmark_rets.as_slice())
    {
        let mut sums = vec![0.0f64; ncols];
        let mut counts = vec![0usize; ncols];
        for (returns_row, benchmark_row) in returns_src
            .chunks_exact(ncols)
            .zip(benchmark_src.chunks_exact(ncols))
        {
            for col in 0..ncols {
                let active_return = returns_row[col] - benchmark_row[col];
                if !active_return.is_nan() {
                    sums[col] += active_return;
                    counts[col] += 1;
                }
            }
        }
        let mut sq = vec![0.0f64; ncols];
        for (returns_row, benchmark_row) in returns_src
            .chunks_exact(ncols)
            .zip(benchmark_src.chunks_exact(ncols))
        {
            for col in 0..ncols {
                let active_return = returns_row[col] - benchmark_row[col];
                if !active_return.is_nan() {
                    let diff = active_return - sums[col] / counts[col] as f64;
                    sq[col] += diff * diff;
                }
            }
        }
        let mut out = vec![f64::NAN; ncols];
        for col in 0..ncols {
            if nrows >= 2 && counts[col] > ddof {
                let mean = sums[col] / counts[col] as f64;
                let std = (sq[col] / (counts[col] - ddof) as f64).sqrt();
                out[col] = if std == 0.0 {
                    f64::INFINITY
                } else {
                    mean / std
                };
            }
        }
        return out;
    }
    reduce_pair_2d_by_col(returns, benchmark_rets, |col, benchmark_col| {
        information_ratio_1d(col, benchmark_col, ddof)
    })
}

pub(crate) fn beta_1d(returns: &[f64], benchmark_rets: &[f64]) -> f64 {
    if benchmark_rets.len() < 2 {
        return f64::NAN;
    }

    let mut independent_sum = 0.0;
    let mut independent_count = 0usize;
    for (&ret, &benchmark_ret) in returns.iter().zip(benchmark_rets.iter()) {
        if !ret.is_nan() && !benchmark_ret.is_nan() {
            independent_sum += benchmark_ret;
            independent_count += 1;
        }
    }
    let independent_mean = if independent_count == 0 {
        f64::NAN
    } else {
        independent_sum / independent_count as f64
    };

    let mut cov_sum = 0.0;
    let mut cov_count = 0usize;
    let mut var_sum = 0.0;
    let mut var_count = 0usize;
    for (&ret, &benchmark_ret) in returns.iter().zip(benchmark_rets.iter()) {
        let independent_value = if ret.is_nan() {
            f64::NAN
        } else {
            benchmark_ret
        };
        let ind_residual = independent_value - independent_mean;
        let covariance = ind_residual * ret;
        if !covariance.is_nan() {
            cov_sum += covariance;
            cov_count += 1;
        }
        let variance = ind_residual * ind_residual;
        if !variance.is_nan() {
            var_sum += variance;
            var_count += 1;
        }
    }

    let covariances = if cov_count == 0 {
        f64::NAN
    } else {
        cov_sum / cov_count as f64
    };
    let mut ind_variances = if var_count == 0 {
        f64::NAN
    } else {
        var_sum / var_count as f64
    };
    if ind_variances < 1.0e-30 {
        ind_variances = f64::NAN;
    }
    if ind_variances == 0.0 {
        return f64::INFINITY;
    }
    covariances / ind_variances
}

pub(crate) fn alpha_1d(
    returns: &[f64],
    benchmark_rets: &[f64],
    ann_factor: f64,
    risk_free: f64,
) -> f64 {
    if returns.len() < 2 {
        return f64::NAN;
    }
    let beta = beta_1d(returns, benchmark_rets);
    (nanmean_alpha_series(returns, benchmark_rets, beta, risk_free) + 1.0).powf(ann_factor) - 1.0
}

pub(crate) fn tail_ratio_1d(returns: &[f64]) -> f64 {
    let mut vals: Vec<f64> = returns
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    let perc_95 = percentile_unsorted(&mut vals, 95.0).abs();
    let perc_5 = percentile_unsorted(&mut vals, 5.0).abs();
    if perc_5 == 0.0 {
        return f64::INFINITY;
    }
    perc_95 / perc_5
}

pub(crate) fn value_at_risk_1d(returns: &[f64], cutoff: f64) -> f64 {
    let mut vals: Vec<f64> = returns
        .iter()
        .copied()
        .filter(|value| !value.is_nan())
        .collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    percentile_unsorted(&mut vals, 100.0 * cutoff)
}

pub(crate) fn cond_value_at_risk_1d(returns: &[f64], cutoff: f64) -> f64 {
    if returns.is_empty() {
        return f64::NAN;
    }
    let mut vals = returns.to_vec();
    let cutoff_index = ((vals.len() - 1) as f64 * cutoff) as usize;
    vals.select_nth_unstable_by(cutoff_index, |left, right| {
        match (left.is_nan(), right.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            (false, true) => Ordering::Less,
            (false, false) => left.partial_cmp(right).unwrap(),
        }
    });
    mean_strict(&vals[..cutoff_index + 1])
}

pub(crate) fn capture_1d(returns: &[f64], benchmark_rets: &[f64], ann_factor: f64) -> f64 {
    let mut returns_prod = 1.0;
    let mut benchmark_prod = 1.0;
    for (&ret, &benchmark_ret) in returns.iter().zip(benchmark_rets.iter()) {
        if !ret.is_nan() {
            returns_prod *= ret + 1.0;
        }
        if !benchmark_ret.is_nan() {
            benchmark_prod *= benchmark_ret + 1.0;
        }
    }
    let annualized_return1 =
        annualized_return_from_product(returns_prod, returns.len(), ann_factor);
    let annualized_return2 =
        annualized_return_from_product(benchmark_prod, benchmark_rets.len(), ann_factor);
    if annualized_return2 == 0.0 {
        return f64::INFINITY;
    }
    annualized_return1 / annualized_return2
}

pub(crate) fn up_capture_1d(returns: &[f64], benchmark_rets: &[f64], ann_factor: f64) -> f64 {
    let mut returns_prod = 1.0;
    let mut benchmark_prod = 1.0;
    let mut count = 0usize;
    for (&ret, &benchmark_ret) in returns.iter().zip(benchmark_rets.iter()) {
        if benchmark_ret > 0.0 {
            if !ret.is_nan() {
                returns_prod *= ret + 1.0;
            }
            benchmark_prod *= benchmark_ret + 1.0;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    let annualized_return1 = annualized_return_from_product(returns_prod, count, ann_factor);
    let annualized_return2 = annualized_return_from_product(benchmark_prod, count, ann_factor);
    if annualized_return2 == 0.0 {
        return f64::INFINITY;
    }
    annualized_return1 / annualized_return2
}

pub(crate) fn down_capture_1d(returns: &[f64], benchmark_rets: &[f64], ann_factor: f64) -> f64 {
    let mut returns_prod = 1.0;
    let mut benchmark_prod = 1.0;
    let mut count = 0usize;
    for (&ret, &benchmark_ret) in returns.iter().zip(benchmark_rets.iter()) {
        if benchmark_ret < 0.0 {
            if !ret.is_nan() {
                returns_prod *= ret + 1.0;
            }
            benchmark_prod *= benchmark_ret + 1.0;
            count += 1;
        }
    }
    if count == 0 {
        return f64::NAN;
    }
    let annualized_return1 = annualized_return_from_product(returns_prod, count, ann_factor);
    let annualized_return2 = annualized_return_from_product(benchmark_prod, count, ann_factor);
    if annualized_return2 == 0.0 {
        return f64::INFINITY;
    }
    annualized_return1 / annualized_return2
}

pub(crate) fn capture_2d(
    returns: ArrayView2<'_, f64>,
    benchmark_rets: ArrayView2<'_, f64>,
    ann_factor: f64,
) -> Vec<f64> {
    let (nrows, ncols) = returns.dim();
    if let (Some(returns_src), Some(benchmark_src)) =
        (returns.as_slice(), benchmark_rets.as_slice())
    {
        let mut returns_prod = vec![1.0f64; ncols];
        let mut benchmark_prod = vec![1.0f64; ncols];
        for row in 0..nrows {
            let row_start = row * ncols;
            for col in 0..ncols {
                let ret = returns_src[row_start + col];
                let benchmark_ret = benchmark_src[row_start + col];
                if !ret.is_nan() {
                    returns_prod[col] *= ret + 1.0;
                }
                if !benchmark_ret.is_nan() {
                    benchmark_prod[col] *= benchmark_ret + 1.0;
                }
            }
        }
        let mut out = vec![f64::NAN; ncols];
        for col in 0..ncols {
            let annualized_return1 =
                annualized_return_from_product(returns_prod[col], nrows, ann_factor);
            let annualized_return2 =
                annualized_return_from_product(benchmark_prod[col], nrows, ann_factor);
            out[col] = if annualized_return2 == 0.0 {
                f64::INFINITY
            } else {
                annualized_return1 / annualized_return2
            };
        }
        return out;
    }
    reduce_pair_2d_by_col(returns, benchmark_rets, |col, benchmark_col| {
        capture_1d(col, benchmark_col, ann_factor)
    })
}

pub(crate) fn filtered_capture_2d(
    returns: ArrayView2<'_, f64>,
    benchmark_rets: ArrayView2<'_, f64>,
    ann_factor: f64,
    up: bool,
) -> Vec<f64> {
    let (_, ncols) = returns.dim();
    if let (Some(returns_src), Some(benchmark_src)) =
        (returns.as_slice(), benchmark_rets.as_slice())
    {
        let mut returns_prod = vec![1.0f64; ncols];
        let mut benchmark_prod = vec![1.0f64; ncols];
        let mut counts = vec![0usize; ncols];
        for (ret_row, benchmark_row) in returns_src
            .chunks_exact(ncols)
            .zip(benchmark_src.chunks_exact(ncols))
        {
            for col in 0..ncols {
                let benchmark_ret = benchmark_row[col];
                if (up && benchmark_ret > 0.0) || (!up && benchmark_ret < 0.0) {
                    let ret = ret_row[col];
                    if !ret.is_nan() {
                        returns_prod[col] *= ret + 1.0;
                    }
                    benchmark_prod[col] *= benchmark_ret + 1.0;
                    counts[col] += 1;
                }
            }
        }
        let mut out = vec![f64::NAN; ncols];
        for col in 0..ncols {
            let count = counts[col];
            if count > 0 {
                let annualized_return1 =
                    annualized_return_from_product(returns_prod[col], count, ann_factor);
                let annualized_return2 =
                    annualized_return_from_product(benchmark_prod[col], count, ann_factor);
                out[col] = if annualized_return2 == 0.0 {
                    f64::INFINITY
                } else {
                    annualized_return1 / annualized_return2
                };
            }
        }
        return out;
    }
    if up {
        reduce_pair_2d_by_col(returns, benchmark_rets, |col, benchmark_col| {
            up_capture_1d(col, benchmark_col, ann_factor)
        })
    } else {
        reduce_pair_2d_by_col(returns, benchmark_rets, |col, benchmark_col| {
            down_capture_1d(col, benchmark_col, ann_factor)
        })
    }
}

#[pyfunction]
pub fn get_return_rs(input_value: f64, output_value: f64) -> PyResult<f64> {
    Ok(get_return(input_value, output_value))
}

#[pyfunction]
pub fn returns_1d_rs<'py>(
    py: Python<'py>,
    value: PyReadonlyArray1<'py, f64>,
    init_value: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let value_cow = array1_as_slice_cow(&value);
    let result = py.allow_threads(|| returns_1d(value_cow.as_ref(), init_value));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn returns_rs<'py>(
    py: Python<'py>,
    value: PyReadonlyArray2<'py, f64>,
    init_value: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let value_arr = value.as_array();
    let init_value_cow = array1_as_slice_cow(&init_value);
    if init_value_cow.len() != value_arr.dim().1 {
        return Err(PyValueError::new_err(
            "init_value must match the number of columns in value",
        ));
    }
    let result = py.allow_threads(|| returns_2d(value_arr, init_value_cow.as_ref()));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn cum_returns_1d_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray1<'py, f64>,
    start_value: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_cow = array1_as_slice_cow(&returns);
    let result = py.allow_threads(|| cum_returns_1d(returns_cow.as_ref(), start_value));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn cum_returns_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    start_value: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| cum_returns_2d(returns_arr, start_value));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, start_value=0.0))]
pub fn cum_returns_final_1d_rs(
    returns: PyReadonlyArray1<'_, f64>,
    start_value: f64,
) -> PyResult<f64> {
    let returns_cow = array1_as_slice_cow(&returns);
    Ok(cum_returns_final_1d(returns_cow.as_ref(), start_value))
}

#[pyfunction]
#[pyo3(signature = (returns, start_value=0.0))]
pub fn cum_returns_final_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    start_value: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| cum_returns_final_2d(returns_arr, start_value));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn annualized_return_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    if returns_arr.dim().0 == 0 {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    let result = py.allow_threads(|| annualized_return_2d(returns_arr, ann_factor));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, ann_factor, levy_alpha=2.0, ddof=1))]
pub fn annualized_volatility_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
    levy_alpha: f64,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let result =
        py.allow_threads(|| annualized_volatility_2d(returns_arr, ann_factor, levy_alpha, ddof));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn drawdown_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| apply_2d_by_col(returns_arr, drawdown_1d));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
pub fn max_drawdown_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    if returns_arr.dim().0 == 0 {
        return Err(PyValueError::new_err(
            "zero-size array to reduction operation minimum which has no identity",
        ));
    }
    let result = py.allow_threads(|| reduce_2d_by_col(returns_arr, max_drawdown_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn calmar_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    if returns_arr.dim().0 == 0 {
        return Err(PyValueError::new_err(
            "zero-size array to reduction operation minimum which has no identity",
        ));
    }
    let result =
        py.allow_threads(|| reduce_2d_by_col(returns_arr, |col| calmar_ratio_1d(col, ann_factor)));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, ann_factor, risk_free=0.0, required_return=0.0))]
pub fn omega_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
    risk_free: f64,
    required_return: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        reduce_2d_by_col(returns_arr, |col| {
            omega_ratio_1d(col, ann_factor, risk_free, required_return)
        })
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, ann_factor, risk_free=0.0, ddof=1))]
pub fn sharpe_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
    risk_free: f64,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| sharpe_ratio_2d(returns_arr, ann_factor, risk_free, ddof));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, ann_factor, required_return=0.0))]
pub fn downside_risk_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
    required_return: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        reduce_2d_by_col(returns_arr, |col| {
            downside_risk_1d(col, ann_factor, required_return)
        })
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, ann_factor, required_return=0.0))]
pub fn sortino_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
    required_return: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        reduce_2d_by_col(returns_arr, |col| {
            sortino_ratio_1d(col, ann_factor, required_return)
        })
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_rets, ddof=1))]
pub fn information_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| information_ratio_2d(returns_arr, benchmark_rets_arr, ddof));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn beta_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result =
        py.allow_threads(|| reduce_pair_2d_by_col(returns_arr, benchmark_rets_arr, beta_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, benchmark_rets, ann_factor, risk_free=0.0))]
pub fn alpha_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
    risk_free: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| {
        reduce_pair_2d_by_col(returns_arr, benchmark_rets_arr, |col, benchmark_col| {
            alpha_1d(col, benchmark_col, ann_factor, risk_free)
        })
    });
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn tail_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| reduce_2d_by_col(returns_arr, tail_ratio_1d));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, cutoff=0.05))]
pub fn value_at_risk_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let result =
        py.allow_threads(|| reduce_2d_by_col(returns_arr, |col| value_at_risk_1d(col, cutoff)));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, cutoff=0.05))]
pub fn cond_value_at_risk_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    if returns_arr.dim().0 == 0 {
        return Err(PyZeroDivisionError::new_err("division by zero"));
    }
    let result = py
        .allow_threads(|| reduce_2d_by_col(returns_arr, |col| cond_value_at_risk_1d(col, cutoff)));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn capture_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| capture_2d(returns_arr, benchmark_rets_arr, ann_factor));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn up_capture_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result =
        py.allow_threads(|| filtered_capture_2d(returns_arr, benchmark_rets_arr, ann_factor, true));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
pub fn down_capture_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py
        .allow_threads(|| filtered_capture_2d(returns_arr, benchmark_rets_arr, ann_factor, false));
    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp=None, start_value=0.0))]
pub fn rolling_cum_returns_final_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    start_value: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            cum_returns_final_1d(col, start_value)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, ann_factor))]
pub fn rolling_annualized_return_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            annualized_return_1d(col, ann_factor)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, ann_factor, levy_alpha=2.0, ddof=1))]
pub fn rolling_annualized_volatility_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ann_factor: f64,
    levy_alpha: f64,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            annualized_volatility_1d(col, ann_factor, levy_alpha, ddof)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp=None))]
pub fn rolling_max_drawdown_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result =
        py.allow_threads(|| rolling_apply_2d_by_col(returns_arr, window, minp, max_drawdown_1d));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, ann_factor))]
pub fn rolling_calmar_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            calmar_ratio_1d(col, ann_factor)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, ann_factor, risk_free=0.0, required_return=0.0))]
pub fn rolling_omega_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ann_factor: f64,
    risk_free: f64,
    required_return: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            omega_ratio_1d(col, ann_factor, risk_free, required_return)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, ann_factor, risk_free=0.0, ddof=1))]
pub fn rolling_sharpe_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ann_factor: f64,
    risk_free: f64,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            sharpe_ratio_1d(col, ann_factor, risk_free, ddof)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, ann_factor, required_return=0.0))]
pub fn rolling_downside_risk_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ann_factor: f64,
    required_return: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            downside_risk_1d(col, ann_factor, required_return)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, ann_factor, required_return=0.0))]
pub fn rolling_sortino_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    ann_factor: f64,
    required_return: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            sortino_ratio_1d(col, ann_factor, required_return)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, benchmark_rets, ddof=1))]
pub fn rolling_information_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ddof: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| {
        rolling_apply_pair_2d_by_col(
            returns_arr,
            benchmark_rets_arr,
            window,
            minp,
            |col, benchmark_col| information_ratio_1d(col, benchmark_col, ddof),
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, benchmark_rets))]
pub fn rolling_beta_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| {
        rolling_apply_pair_2d_by_col(returns_arr, benchmark_rets_arr, window, minp, beta_1d)
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, benchmark_rets, ann_factor, risk_free=0.0))]
pub fn rolling_alpha_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
    risk_free: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| {
        rolling_apply_pair_2d_by_col(
            returns_arr,
            benchmark_rets_arr,
            window,
            minp,
            |col, benchmark_col| alpha_1d(col, benchmark_col, ann_factor, risk_free),
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp=None))]
pub fn rolling_tail_ratio_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result =
        py.allow_threads(|| rolling_apply_2d_by_col(returns_arr, window, minp, tail_ratio_1d));
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, cutoff=0.05))]
pub fn rolling_value_at_risk_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            value_at_risk_1d(col, cutoff)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, cutoff=0.05))]
pub fn rolling_cond_value_at_risk_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let result = py.allow_threads(|| {
        rolling_apply_2d_by_col(returns_arr, window, minp, |col| {
            cond_value_at_risk_1d(col, cutoff)
        })
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, benchmark_rets, ann_factor))]
pub fn rolling_capture_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| {
        rolling_apply_pair_2d_by_col(
            returns_arr,
            benchmark_rets_arr,
            window,
            minp,
            |col, benchmark_col| capture_1d(col, benchmark_col, ann_factor),
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, benchmark_rets, ann_factor))]
pub fn rolling_up_capture_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| {
        rolling_apply_pair_2d_by_col(
            returns_arr,
            benchmark_rets_arr,
            window,
            minp,
            |col, benchmark_col| up_capture_1d(col, benchmark_col, ann_factor),
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

#[pyfunction]
#[pyo3(signature = (returns, window, minp, benchmark_rets, ann_factor))]
pub fn rolling_down_capture_rs<'py>(
    py: Python<'py>,
    returns: PyReadonlyArray2<'py, f64>,
    window: usize,
    minp: Option<usize>,
    benchmark_rets: PyReadonlyArray2<'py, f64>,
    ann_factor: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let minp = minp.unwrap_or(window);
    validate_window(minp, window, "window")?;
    let returns_arr = returns.as_array();
    let benchmark_rets_arr = benchmark_rets.as_array();
    validate_same_shape_2d(returns_arr, benchmark_rets_arr, "benchmark_rets")?;
    let result = py.allow_threads(|| {
        rolling_apply_pair_2d_by_col(
            returns_arr,
            benchmark_rets_arr,
            window,
            minp,
            |col, benchmark_col| down_capture_1d(col, benchmark_col, ann_factor),
        )
    });
    Ok(PyArray2::from_owned_array_bound(py, result))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_return_rs, m)?)?;
    m.add_function(wrap_pyfunction!(returns_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(returns_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cum_returns_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cum_returns_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cum_returns_final_1d_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cum_returns_final_rs, m)?)?;
    m.add_function(wrap_pyfunction!(annualized_return_rs, m)?)?;
    m.add_function(wrap_pyfunction!(annualized_volatility_rs, m)?)?;
    m.add_function(wrap_pyfunction!(drawdown_rs, m)?)?;
    m.add_function(wrap_pyfunction!(max_drawdown_rs, m)?)?;
    m.add_function(wrap_pyfunction!(calmar_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(omega_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sharpe_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(downside_risk_rs, m)?)?;
    m.add_function(wrap_pyfunction!(sortino_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(information_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(beta_rs, m)?)?;
    m.add_function(wrap_pyfunction!(alpha_rs, m)?)?;
    m.add_function(wrap_pyfunction!(tail_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(value_at_risk_rs, m)?)?;
    m.add_function(wrap_pyfunction!(cond_value_at_risk_rs, m)?)?;
    m.add_function(wrap_pyfunction!(capture_rs, m)?)?;
    m.add_function(wrap_pyfunction!(up_capture_rs, m)?)?;
    m.add_function(wrap_pyfunction!(down_capture_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_cum_returns_final_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_annualized_return_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_annualized_volatility_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_max_drawdown_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_calmar_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_omega_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_sharpe_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_downside_risk_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_sortino_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_information_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_beta_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_alpha_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_tail_ratio_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_value_at_risk_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_cond_value_at_risk_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_capture_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_up_capture_rs, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_down_capture_rs, m)?)?;
    Ok(())
}
