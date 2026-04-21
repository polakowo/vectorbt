// Copyright (c) 2017-2026 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

#[allow(dead_code)]
use ndarray::Array2;
use numpy::{
    Element, PyArray1, PyArray2, PyArrayDescr, PyReadonlyArray1, PyReadonlyArray2,
    PyReadonlyArrayDyn, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::borrow::Cow;

/// Encode a raw pointer as usize so it can cross allow_threads boundaries.
/// Safety: the caller must ensure the pointed-to memory outlives the closure.
#[inline(always)]
fn ptr_to_usize(p: *mut u8) -> usize {
    p as usize
}

#[inline(always)]
fn usize_to_ptr(v: usize) -> *const u8 {
    v as *const u8
}

#[inline(always)]
fn usize_to_mut_ptr(v: usize) -> *mut u8 {
    v as *mut u8
}

#[inline(always)]
fn uninit_f64_vec(len: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(len);
    unsafe {
        out.set_len(len);
    }
    out
}

use crate::generic::{array1_as_slice_cow, array2_as_slice_cow, FlexArray};
use crate::records::{array_raw_parts, numpy_empty};

// ############# Math utilities ############# //

const REL_TOL: f64 = 1e-9;
const ABS_TOL: f64 = 1e-12;

#[inline(always)]
fn is_close(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() || b.is_infinite() {
        return false;
    }
    if a == b {
        return true;
    }
    (a - b).abs() <= ABS_TOL.max(REL_TOL * a.abs().max(b.abs()))
}

#[inline(always)]
fn is_close_or_less(a: f64, b: f64) -> bool {
    if is_close(a, b) {
        return true;
    }
    a < b
}

#[inline(always)]
fn is_less(a: f64, b: f64) -> bool {
    if is_close(a, b) {
        return false;
    }
    a < b
}

#[inline(always)]
fn is_addition_zero(a: f64, b: f64) -> bool {
    if a.signum() != b.signum() {
        is_close(a.abs(), b.abs())
    } else {
        is_close(a + b, 0.0)
    }
}

#[inline(always)]
fn add(a: f64, b: f64) -> f64 {
    if is_addition_zero(a, b) {
        0.0
    } else {
        a + b
    }
}

#[inline(always)]
fn get_return(input_value: f64, output_value: f64) -> f64 {
    if input_value == 0.0 {
        if output_value == 0.0 {
            return 0.0;
        }
        return f64::INFINITY * output_value.signum();
    }
    let mut r = (output_value - input_value) / input_value;
    if input_value < 0.0 {
        r *= -1.0;
    }
    r
}

fn col_start_idxs_usize(col_lens: &[i64]) -> Vec<usize> {
    let mut out = Vec::with_capacity(col_lens.len());
    let mut cumsum = 0usize;
    for &col_len in col_lens {
        out.push(cumsum);
        cumsum += col_len as usize;
    }
    out
}

// ############# Enum constants ############# //

// SizeType
const SIZE_TYPE_AMOUNT: i64 = 0;
const SIZE_TYPE_VALUE: i64 = 1;
const SIZE_TYPE_PERCENT: i64 = 2;
const SIZE_TYPE_TARGET_AMOUNT: i64 = 3;
const SIZE_TYPE_TARGET_VALUE: i64 = 4;
const SIZE_TYPE_TARGET_PERCENT: i64 = 5;

// Direction
const DIRECTION_LONG_ONLY: i64 = 0;
const DIRECTION_SHORT_ONLY: i64 = 1;
const DIRECTION_BOTH: i64 = 2;

// OrderStatus
const ORDER_STATUS_FILLED: i64 = 0;
const ORDER_STATUS_IGNORED: i64 = 1;
const ORDER_STATUS_REJECTED: i64 = 2;

// OrderSide
const ORDER_SIDE_BUY: i64 = 0;
const ORDER_SIDE_SELL: i64 = 1;

// OrderStatusInfo
const ORDER_STATUS_INFO_SIZE_NAN: i64 = 0;
const ORDER_STATUS_INFO_PRICE_NAN: i64 = 1;
const ORDER_STATUS_INFO_VAL_PRICE_NAN: i64 = 2;
const ORDER_STATUS_INFO_VALUE_NAN: i64 = 3;
const ORDER_STATUS_INFO_VALUE_ZERO_NEG: i64 = 4;
const ORDER_STATUS_INFO_SIZE_ZERO: i64 = 5;
const ORDER_STATUS_INFO_NO_CASH_SHORT: i64 = 6;
const ORDER_STATUS_INFO_NO_CASH_LONG: i64 = 7;
const ORDER_STATUS_INFO_NO_OPEN_POSITION: i64 = 8;
const ORDER_STATUS_INFO_MAX_SIZE_EXCEEDED: i64 = 9;
const ORDER_STATUS_INFO_RANDOM_EVENT: i64 = 10;
const ORDER_STATUS_INFO_CANT_COVER_FEES: i64 = 11;
const ORDER_STATUS_INFO_MIN_SIZE_NOT_REACHED: i64 = 12;
const ORDER_STATUS_INFO_PARTIAL_FILL: i64 = 13;

// TradeDirection
const TRADE_DIRECTION_LONG: i64 = 0;
const TRADE_DIRECTION_SHORT: i64 = 1;

// TradeStatus
const TRADE_STATUS_OPEN: i64 = 0;
const TRADE_STATUS_CLOSED: i64 = 1;

// CallSeqType
const CALL_SEQ_DEFAULT: i64 = 0;
const CALL_SEQ_REVERSED: i64 = 1;
const CALL_SEQ_RANDOM: i64 = 2;

// AccumulationMode
const ACCUMULATION_DISABLED: i64 = 0;
const ACCUMULATION_BOTH: i64 = 1;
const ACCUMULATION_ADD_ONLY: i64 = 2;
const ACCUMULATION_REMOVE_ONLY: i64 = 3;

// ConflictMode
const CONFLICT_IGNORE: i64 = 0;
const CONFLICT_ENTRY: i64 = 1;
const CONFLICT_EXIT: i64 = 2;
const CONFLICT_ADJACENT: i64 = 3;
const CONFLICT_OPPOSITE: i64 = 4;

// DirectionConflictMode
const DIR_CONFLICT_IGNORE: i64 = 0;
const DIR_CONFLICT_LONG: i64 = 1;
const DIR_CONFLICT_SHORT: i64 = 2;
const DIR_CONFLICT_ADJACENT: i64 = 3;
const DIR_CONFLICT_OPPOSITE: i64 = 4;

// OppositeEntryMode
const OPPOSITE_ENTRY_IGNORE: i64 = 0;
const OPPOSITE_ENTRY_CLOSE: i64 = 1;
const OPPOSITE_ENTRY_CLOSE_REDUCE: i64 = 2;
const OPPOSITE_ENTRY_REVERSE: i64 = 3;
const OPPOSITE_ENTRY_REVERSE_REDUCE: i64 = 4;

// StopEntryPrice
const STOP_ENTRY_VAL_PRICE: i64 = 0;
const STOP_ENTRY_PRICE: i64 = 1;
const STOP_ENTRY_FILL_PRICE: i64 = 2;
const STOP_ENTRY_CLOSE: i64 = 3;

// StopExitPrice
const STOP_EXIT_STOP_LIMIT: i64 = 0;
const STOP_EXIT_STOP_MARKET: i64 = 1;
const STOP_EXIT_PRICE: i64 = 2;
const STOP_EXIT_CLOSE: i64 = 3;

// StopExitMode
const STOP_MODE_CLOSE: i64 = 0;
const STOP_MODE_CLOSE_REDUCE: i64 = 1;
const STOP_MODE_REVERSE: i64 = 2;
const STOP_MODE_REVERSE_REDUCE: i64 = 3;

// StopUpdateMode
const STOP_UPDATE_KEEP: i64 = 0;
const STOP_UPDATE_OVERRIDE: i64 = 1;
const STOP_UPDATE_OVERRIDE_NAN: i64 = 2;

// ############# Core structs ############# //

#[pyclass(get_all, set_all)]
#[derive(Clone, Copy, Debug)]
pub struct Order {
    pub size: f64,
    pub price: f64,
    pub size_type: i64,
    pub direction: i64,
    pub fees: f64,
    pub fixed_fees: f64,
    pub slippage: f64,
    pub min_size: f64,
    pub max_size: f64,
    pub size_granularity: f64,
    pub reject_prob: f64,
    pub lock_cash: bool,
    pub allow_partial: bool,
    pub raise_reject: bool,
    pub log: bool,
}

#[pymethods]
impl Order {
    #[new]
    #[pyo3(signature = (
        size = f64::NAN,
        price = f64::INFINITY,
        size_type = SIZE_TYPE_AMOUNT,
        direction = DIRECTION_BOTH,
        fees = 0.0,
        fixed_fees = 0.0,
        slippage = 0.0,
        min_size = 0.0,
        max_size = f64::INFINITY,
        size_granularity = f64::NAN,
        reject_prob = 0.0,
        lock_cash = false,
        allow_partial = true,
        raise_reject = false,
        log = false,
    ))]
    fn new(
        size: f64,
        price: f64,
        size_type: i64,
        direction: i64,
        fees: f64,
        fixed_fees: f64,
        slippage: f64,
        min_size: f64,
        max_size: f64,
        size_granularity: f64,
        reject_prob: f64,
        lock_cash: bool,
        allow_partial: bool,
        raise_reject: bool,
        log: bool,
    ) -> Self {
        Order {
            size,
            price,
            size_type,
            direction,
            fees,
            fixed_fees,
            slippage,
            min_size,
            max_size,
            size_granularity,
            reject_prob,
            lock_cash,
            allow_partial,
            raise_reject,
            log,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "Order(size={}, price={}, size_type={}, direction={}, fees={}, fixed_fees={}, \
             slippage={}, min_size={}, max_size={}, size_granularity={}, reject_prob={}, \
             lock_cash={}, allow_partial={}, raise_reject={}, log={})",
            self.size,
            self.price,
            self.size_type,
            self.direction,
            self.fees,
            self.fixed_fees,
            self.slippage,
            self.min_size,
            self.max_size,
            self.size_granularity,
            self.reject_prob,
            self.lock_cash,
            self.allow_partial,
            self.raise_reject,
            self.log,
        )
    }
}

impl Default for Order {
    fn default() -> Self {
        Order {
            size: f64::NAN,
            price: f64::INFINITY,
            size_type: SIZE_TYPE_AMOUNT,
            direction: DIRECTION_BOTH,
            fees: 0.0,
            fixed_fees: 0.0,
            slippage: 0.0,
            min_size: 0.0,
            max_size: f64::INFINITY,
            size_granularity: f64::NAN,
            reject_prob: 0.0,
            lock_cash: false,
            allow_partial: true,
            raise_reject: false,
            log: false,
        }
    }
}

/// A "no order" sentinel.
const NO_ORDER: Order = Order {
    size: f64::NAN,
    price: f64::NAN,
    size_type: -1,
    direction: -1,
    fees: f64::NAN,
    fixed_fees: f64::NAN,
    slippage: f64::NAN,
    min_size: f64::NAN,
    max_size: f64::NAN,
    size_granularity: f64::NAN,
    reject_prob: f64::NAN,
    lock_cash: false,
    allow_partial: false,
    raise_reject: false,
    log: false,
};

#[pyclass(get_all)]
#[derive(Clone, Copy, Debug)]
pub struct OrderResult {
    pub size: f64,
    pub price: f64,
    pub fees: f64,
    pub side: i64,
    pub status: i64,
    pub status_info: i64,
}

#[pymethods]
impl OrderResult {
    #[new]
    fn new(size: f64, price: f64, fees: f64, side: i64, status: i64, status_info: i64) -> Self {
        OrderResult {
            size,
            price,
            fees,
            side,
            status,
            status_info,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OrderResult(size={}, price={}, fees={}, side={}, status={}, status_info={})",
            self.size, self.price, self.fees, self.side, self.status, self.status_info,
        )
    }
}

#[pyclass(get_all, set_all)]
#[derive(Clone, Copy, Debug)]
pub struct ExecuteOrderState {
    pub cash: f64,
    pub position: f64,
    pub debt: f64,
    pub free_cash: f64,
}

#[pymethods]
impl ExecuteOrderState {
    #[new]
    fn new(cash: f64, position: f64, debt: f64, free_cash: f64) -> Self {
        ExecuteOrderState {
            cash,
            position,
            debt,
            free_cash,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ExecuteOrderState(cash={}, position={}, debt={}, free_cash={})",
            self.cash, self.position, self.debt, self.free_cash,
        )
    }
}

#[pyclass(get_all, set_all)]
#[derive(Clone, Copy, Debug)]
pub struct ProcessOrderState {
    pub cash: f64,
    pub position: f64,
    pub debt: f64,
    pub free_cash: f64,
    pub val_price: f64,
    pub value: f64,
    pub oidx: i64,
    pub lidx: i64,
}

#[pymethods]
impl ProcessOrderState {
    #[new]
    fn new(
        cash: f64,
        position: f64,
        debt: f64,
        free_cash: f64,
        val_price: f64,
        value: f64,
        oidx: i64,
        lidx: i64,
    ) -> Self {
        ProcessOrderState {
            cash,
            position,
            debt,
            free_cash,
            val_price,
            value,
            oidx,
            lidx,
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ProcessOrderState(cash={}, position={}, debt={}, free_cash={}, \
             val_price={}, value={}, oidx={}, lidx={})",
            self.cash,
            self.position,
            self.debt,
            self.free_cash,
            self.val_price,
            self.value,
            self.oidx,
            self.lidx,
        )
    }
}

// ############# Record structs ############# //

#[repr(C)]
#[derive(Clone, Copy)]
pub struct OrderRecord {
    pub id: i64,
    pub col: i64,
    pub idx: i64,
    pub size: f64,
    pub price: f64,
    pub fees: f64,
    pub side: i64,
}

unsafe impl Element for OrderRecord {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        py.import_bound("vectorbt.portfolio.enums")
            .and_then(|m| m.getattr("order_dt"))
            .and_then(|dt| Ok(dt.downcast_into::<PyArrayDescr>()?))
            .expect("vectorbt.portfolio.enums.order_dt must be a NumPy dtype")
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
pub struct TradeRecord {
    pub id: i64,
    pub col: i64,
    pub size: f64,
    pub entry_idx: i64,
    pub entry_price: f64,
    pub entry_fees: f64,
    pub exit_idx: i64,
    pub exit_price: f64,
    pub exit_fees: f64,
    pub pnl: f64,
    pub return_: f64,
    pub direction: i64,
    pub status: i64,
    pub parent_id: i64,
}

unsafe impl Element for TradeRecord {
    const IS_COPY: bool = true;

    fn get_dtype_bound(py: Python<'_>) -> Bound<'_, PyArrayDescr> {
        py.import_bound("vectorbt.portfolio.enums")
            .and_then(|m| m.getattr("trade_dt"))
            .and_then(|dt| Ok(dt.downcast_into::<PyArrayDescr>()?))
            .expect("vectorbt.portfolio.enums.trade_dt must be a NumPy dtype")
    }

    fn clone_ref(&self, _py: Python<'_>) -> Self {
        *self
    }

    fn vec_from_slice(_py: Python<'_>, slc: &[Self]) -> Vec<Self> {
        slc.to_vec()
    }
}

// ############# Helpers for record field access ############# //

enum PortfolioSimError {
    ValueError(&'static str),
    RejectedOrder(&'static str),
    OrderRecordsOutOfRange,
    LogRecordsOutOfRange,
}

fn portfolio_sim_error_to_pyerr(py: Python<'_>, err: PortfolioSimError) -> PyErr {
    match err {
        PortfolioSimError::ValueError(msg) => PyValueError::new_err(msg),
        PortfolioSimError::RejectedOrder(msg) => {
            match py
                .import_bound("vectorbt.portfolio.enums")
                .and_then(|m| m.getattr("RejectedOrderError"))
                .and_then(|exc_type| exc_type.call1((msg,)))
            {
                Ok(exc) => PyErr::from_value_bound(exc),
                Err(_) => PyValueError::new_err(msg),
            }
        }
        PortfolioSimError::OrderRecordsOutOfRange => {
            PyIndexError::new_err("order_records index out of range. Set a higher max_orders.")
        }
        PortfolioSimError::LogRecordsOutOfRange => {
            PyIndexError::new_err("log_records index out of range. Set a higher max_logs.")
        }
    }
}

struct RecordFieldOffsets {
    id: usize,
    col: usize,
    idx: usize,
    size: usize,
    price: usize,
    fees: usize,
    side: usize,
    itemsize: usize,
}

fn order_record_offsets(records: &Bound<'_, pyo3::PyAny>) -> PyResult<RecordFieldOffsets> {
    let dtype = records.getattr("dtype")?;
    let fields = dtype.getattr("fields")?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    Ok(RecordFieldOffsets {
        id: fields.get_item("id")?.get_item(1)?.extract()?,
        col: fields.get_item("col")?.get_item(1)?.extract()?,
        idx: fields.get_item("idx")?.get_item(1)?.extract()?,
        size: fields.get_item("size")?.get_item(1)?.extract()?,
        price: fields.get_item("price")?.get_item(1)?.extract()?,
        fees: fields.get_item("fees")?.get_item(1)?.extract()?,
        side: fields.get_item("side")?.get_item(1)?.extract()?,
        itemsize,
    })
}

struct TradeFieldOffsets {
    id: usize,
    col: usize,
    size: usize,
    entry_idx: usize,
    entry_price: usize,
    entry_fees: usize,
    exit_idx: usize,
    exit_price: usize,
    exit_fees: usize,
    pnl: usize,
    return_: usize,
    direction: usize,
    status: usize,
    parent_id: usize,
    itemsize: usize,
}

fn trade_record_offsets(records: &Bound<'_, pyo3::PyAny>) -> PyResult<TradeFieldOffsets> {
    let dtype = records.getattr("dtype")?;
    let fields = dtype.getattr("fields")?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    Ok(TradeFieldOffsets {
        id: fields.get_item("id")?.get_item(1)?.extract()?,
        col: fields.get_item("col")?.get_item(1)?.extract()?,
        size: fields.get_item("size")?.get_item(1)?.extract()?,
        entry_idx: fields.get_item("entry_idx")?.get_item(1)?.extract()?,
        entry_price: fields.get_item("entry_price")?.get_item(1)?.extract()?,
        entry_fees: fields.get_item("entry_fees")?.get_item(1)?.extract()?,
        exit_idx: fields.get_item("exit_idx")?.get_item(1)?.extract()?,
        exit_price: fields.get_item("exit_price")?.get_item(1)?.extract()?,
        exit_fees: fields.get_item("exit_fees")?.get_item(1)?.extract()?,
        pnl: fields.get_item("pnl")?.get_item(1)?.extract()?,
        return_: fields.get_item("return")?.get_item(1)?.extract()?,
        direction: fields.get_item("direction")?.get_item(1)?.extract()?,
        status: fields.get_item("status")?.get_item(1)?.extract()?,
        parent_id: fields.get_item("parent_id")?.get_item(1)?.extract()?,
        itemsize,
    })
}

/// Log record field offsets — extracted at runtime because log_dt has bool fields
/// whose alignment depends on the platform.
struct LogFieldOffsets {
    id: usize,
    group: usize,
    col: usize,
    idx: usize,
    cash: usize,
    position: usize,
    debt: usize,
    free_cash: usize,
    val_price: usize,
    value: usize,
    req_size: usize,
    req_price: usize,
    req_size_type: usize,
    req_direction: usize,
    req_fees: usize,
    req_fixed_fees: usize,
    req_slippage: usize,
    req_min_size: usize,
    req_max_size: usize,
    req_size_granularity: usize,
    req_reject_prob: usize,
    req_lock_cash: usize,
    req_allow_partial: usize,
    req_raise_reject: usize,
    req_log: usize,
    new_cash: usize,
    new_position: usize,
    new_debt: usize,
    new_free_cash: usize,
    new_val_price: usize,
    new_value: usize,
    res_size: usize,
    res_price: usize,
    res_fees: usize,
    res_side: usize,
    res_status: usize,
    res_status_info: usize,
    order_id: usize,
    itemsize: usize,
}

fn log_record_offsets(records: &Bound<'_, pyo3::PyAny>) -> PyResult<LogFieldOffsets> {
    let dtype = records.getattr("dtype")?;
    let fields = dtype.getattr("fields")?;
    let itemsize: usize = dtype.getattr("itemsize")?.extract()?;
    Ok(LogFieldOffsets {
        id: fields.get_item("id")?.get_item(1)?.extract()?,
        group: fields.get_item("group")?.get_item(1)?.extract()?,
        col: fields.get_item("col")?.get_item(1)?.extract()?,
        idx: fields.get_item("idx")?.get_item(1)?.extract()?,
        cash: fields.get_item("cash")?.get_item(1)?.extract()?,
        position: fields.get_item("position")?.get_item(1)?.extract()?,
        debt: fields.get_item("debt")?.get_item(1)?.extract()?,
        free_cash: fields.get_item("free_cash")?.get_item(1)?.extract()?,
        val_price: fields.get_item("val_price")?.get_item(1)?.extract()?,
        value: fields.get_item("value")?.get_item(1)?.extract()?,
        req_size: fields.get_item("req_size")?.get_item(1)?.extract()?,
        req_price: fields.get_item("req_price")?.get_item(1)?.extract()?,
        req_size_type: fields.get_item("req_size_type")?.get_item(1)?.extract()?,
        req_direction: fields.get_item("req_direction")?.get_item(1)?.extract()?,
        req_fees: fields.get_item("req_fees")?.get_item(1)?.extract()?,
        req_fixed_fees: fields.get_item("req_fixed_fees")?.get_item(1)?.extract()?,
        req_slippage: fields.get_item("req_slippage")?.get_item(1)?.extract()?,
        req_min_size: fields.get_item("req_min_size")?.get_item(1)?.extract()?,
        req_max_size: fields.get_item("req_max_size")?.get_item(1)?.extract()?,
        req_size_granularity: fields
            .get_item("req_size_granularity")?
            .get_item(1)?
            .extract()?,
        req_reject_prob: fields.get_item("req_reject_prob")?.get_item(1)?.extract()?,
        req_lock_cash: fields.get_item("req_lock_cash")?.get_item(1)?.extract()?,
        req_allow_partial: fields
            .get_item("req_allow_partial")?
            .get_item(1)?
            .extract()?,
        req_raise_reject: fields
            .get_item("req_raise_reject")?
            .get_item(1)?
            .extract()?,
        req_log: fields.get_item("req_log")?.get_item(1)?.extract()?,
        new_cash: fields.get_item("new_cash")?.get_item(1)?.extract()?,
        new_position: fields.get_item("new_position")?.get_item(1)?.extract()?,
        new_debt: fields.get_item("new_debt")?.get_item(1)?.extract()?,
        new_free_cash: fields.get_item("new_free_cash")?.get_item(1)?.extract()?,
        new_val_price: fields.get_item("new_val_price")?.get_item(1)?.extract()?,
        new_value: fields.get_item("new_value")?.get_item(1)?.extract()?,
        res_size: fields.get_item("res_size")?.get_item(1)?.extract()?,
        res_price: fields.get_item("res_price")?.get_item(1)?.extract()?,
        res_fees: fields.get_item("res_fees")?.get_item(1)?.extract()?,
        res_side: fields.get_item("res_side")?.get_item(1)?.extract()?,
        res_status: fields.get_item("res_status")?.get_item(1)?.extract()?,
        res_status_info: fields.get_item("res_status_info")?.get_item(1)?.extract()?,
        order_id: fields.get_item("order_id")?.get_item(1)?.extract()?,
        itemsize,
    })
}

// ############# Insertion sort ############# //

/// In-place insertion sort (argsort). Very fast for small arrays.
fn insert_argsort(a: &mut [f64], idx: &mut [i64]) {
    for j in 1..a.len() {
        let a_j = a[j];
        let i_j = idx[j];
        let mut i = j as isize - 1;
        while i >= 0 && (a[i as usize] > a_j || a[i as usize].is_nan()) {
            a[(i + 1) as usize] = a[i as usize];
            idx[(i + 1) as usize] = idx[i as usize];
            i -= 1;
        }
        a[(i + 1) as usize] = a_j;
        idx[(i + 1) as usize] = i_j;
    }
}

// ############# Core order execution (internal) ############# //

fn order_not_filled(status: i64, status_info: i64) -> OrderResult {
    OrderResult {
        size: f64::NAN,
        price: f64::NAN,
        fees: f64::NAN,
        side: -1,
        status,
        status_info,
    }
}

fn buy(
    exec_state: ExecuteOrderState,
    size: f64,
    price: f64,
    direction: i64,
    fees: f64,
    fixed_fees: f64,
    slippage: f64,
    min_size: f64,
    max_size: f64,
    size_granularity: f64,
    lock_cash: bool,
    allow_partial: bool,
    percent: f64,
) -> Result<(ExecuteOrderState, OrderResult), PortfolioSimError> {
    // Get price adjusted with slippage
    let adj_price = price * (1.0 + slippage);

    // Set cash limit
    let mut cash_limit = if lock_cash {
        if exec_state.position >= 0.0 {
            exec_state.free_cash
        } else {
            let cover_req_cash = exec_state.position.abs() * adj_price * (1.0 + fees) + fixed_fees;
            let cover_free_cash = add(
                exec_state.free_cash + 2.0 * exec_state.debt,
                -cover_req_cash,
            );
            if cover_free_cash > 0.0 {
                exec_state.free_cash + 2.0 * exec_state.debt
            } else if cover_free_cash < 0.0 {
                let avg_entry_price = exec_state.debt / exec_state.position.abs();
                let max_short_size = (exec_state.free_cash - fixed_fees)
                    / (adj_price * (1.0 + fees) - 2.0 * avg_entry_price);
                max_short_size * adj_price * (1.0 + fees) + fixed_fees
            } else {
                exec_state.cash
            }
        }
    } else {
        exec_state.cash
    };
    cash_limit = cash_limit.min(exec_state.cash);
    if !percent.is_nan() {
        cash_limit = cash_limit.min(percent * cash_limit);
    }

    if direction == DIRECTION_LONG_ONLY || direction == DIRECTION_BOTH {
        if cash_limit == 0.0 {
            return Ok((
                exec_state,
                order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_NO_CASH_LONG),
            ));
        }
        if size.is_infinite() && cash_limit.is_infinite() {
            return Err(PortfolioSimError::ValueError(
                "Attempt to go in long direction infinitely",
            ));
        }
    } else if exec_state.position == 0.0 {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_NO_OPEN_POSITION),
        ));
    }

    // Get optimal order size
    let mut adj_size = if direction == DIRECTION_SHORT_ONLY {
        (-exec_state.position).min(size)
    } else {
        size
    };

    if adj_size == 0.0 {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_IGNORED, ORDER_STATUS_INFO_SIZE_ZERO),
        ));
    }

    if adj_size > max_size {
        if !allow_partial {
            return Ok((
                exec_state,
                order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_MAX_SIZE_EXCEEDED),
            ));
        }
        adj_size = max_size;
    }

    // Adjust granularity
    if !size_granularity.is_nan() {
        adj_size = (adj_size / size_granularity).floor() * size_granularity;
    }

    // Get cash required
    let req_cash = adj_size * adj_price;
    let req_fees = req_cash * fees + fixed_fees;
    let total_req_cash = req_cash + req_fees;

    let (final_size, fees_paid, final_req_cash);
    if is_close_or_less(total_req_cash, cash_limit) {
        final_size = adj_size;
        fees_paid = req_fees;
        final_req_cash = total_req_cash;
    } else {
        // Insufficient cash
        let max_req_cash = add(cash_limit, -fixed_fees) / (1.0 + fees);
        if max_req_cash <= 0.0 {
            return Ok((
                exec_state,
                order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_CANT_COVER_FEES),
            ));
        }

        let max_acq_size = max_req_cash / adj_price;

        if !size_granularity.is_nan() {
            final_size = (max_acq_size / size_granularity).floor() * size_granularity;
            let new_req_cash = final_size * adj_price;
            fees_paid = new_req_cash * fees + fixed_fees;
            final_req_cash = new_req_cash + fees_paid;
        } else {
            final_size = max_acq_size;
            fees_paid = cash_limit - max_req_cash;
            final_req_cash = cash_limit;
        }
    }

    if is_close(adj_size, 0.0) {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_IGNORED, ORDER_STATUS_INFO_SIZE_ZERO),
        ));
    }

    if is_less(final_size, min_size) {
        return Ok((
            exec_state,
            order_not_filled(
                ORDER_STATUS_REJECTED,
                ORDER_STATUS_INFO_MIN_SIZE_NOT_REACHED,
            ),
        ));
    }

    if size.is_finite() && is_less(final_size, size) && !allow_partial {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_PARTIAL_FILL),
        ));
    }

    // Update state
    let new_cash = add(exec_state.cash, -final_req_cash);
    let new_position = add(exec_state.position, final_size);

    let (new_debt, new_free_cash);
    if exec_state.position < 0.0 {
        let short_size = if new_position < 0.0 {
            final_size
        } else {
            exec_state.position.abs()
        };
        let avg_entry_price = exec_state.debt / exec_state.position.abs();
        let debt_diff = short_size * avg_entry_price;
        new_debt = add(exec_state.debt, -debt_diff);
        new_free_cash = add(exec_state.free_cash + 2.0 * debt_diff, -final_req_cash);
    } else {
        new_debt = exec_state.debt;
        new_free_cash = add(exec_state.free_cash, -final_req_cash);
    }

    let result = OrderResult {
        size: final_size,
        price: adj_price,
        fees: fees_paid,
        side: ORDER_SIDE_BUY,
        status: ORDER_STATUS_FILLED,
        status_info: -1,
    };
    let new_state = ExecuteOrderState {
        cash: new_cash,
        position: new_position,
        debt: new_debt,
        free_cash: new_free_cash,
    };
    Ok((new_state, result))
}

fn sell(
    exec_state: ExecuteOrderState,
    size: f64,
    price: f64,
    direction: i64,
    fees: f64,
    fixed_fees: f64,
    slippage: f64,
    min_size: f64,
    max_size: f64,
    size_granularity: f64,
    lock_cash: bool,
    allow_partial: bool,
    percent: f64,
) -> Result<(ExecuteOrderState, OrderResult), PortfolioSimError> {
    // Get price adjusted with slippage
    let adj_price = price * (1.0 - slippage);

    // Get optimal order size
    let mut size_limit;
    let mut percent_val = percent;

    if direction == DIRECTION_LONG_ONLY {
        size_limit = exec_state.position.min(size);
    } else if lock_cash || (size.is_infinite() && !percent_val.is_nan()) {
        // Get the maximum size that can be (short) sold
        let long_size = exec_state.position.max(0.0);
        let long_cash = long_size * adj_price * (1.0 - fees);
        let total_free_cash = add(exec_state.free_cash, long_cash);

        let max_size_limit;
        if total_free_cash <= 0.0 {
            if exec_state.position <= 0.0 {
                return Ok((
                    exec_state,
                    order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_NO_CASH_SHORT),
                ));
            }
            max_size_limit = long_size;
        } else {
            let max_short_size = add(total_free_cash, -fixed_fees) / (adj_price * (1.0 + fees));
            max_size_limit = add(long_size, max_short_size);
            if max_size_limit <= 0.0 {
                return Ok((
                    exec_state,
                    order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_CANT_COVER_FEES),
                ));
            }
        }

        if lock_cash {
            if size.is_infinite() && !percent_val.is_nan() {
                size_limit = (percent_val * max_size_limit).min(max_size_limit);
                percent_val = f64::NAN;
            } else if !percent_val.is_nan() {
                size_limit = (percent_val * size).min(max_size_limit);
                percent_val = f64::NAN;
            } else {
                size_limit = size.min(max_size_limit);
            }
        } else {
            // size.is_infinite() && !percent_val.is_nan()
            size_limit = max_size_limit;
        }
    } else {
        size_limit = size;
    }

    if !percent_val.is_nan() {
        size_limit = percent_val * size_limit;
    }

    if size_limit > max_size {
        if !allow_partial {
            return Ok((
                exec_state,
                order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_MAX_SIZE_EXCEEDED),
            ));
        }
        size_limit = max_size;
    }

    if direction == DIRECTION_SHORT_ONLY || direction == DIRECTION_BOTH {
        if size_limit.is_infinite() {
            return Err(PortfolioSimError::ValueError(
                "Attempt to go in short direction infinitely",
            ));
        }
    } else if exec_state.position == 0.0 {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_NO_OPEN_POSITION),
        ));
    }

    // Adjust granularity
    if !size_granularity.is_nan() {
        size_limit = (size_limit / size_granularity).floor() * size_granularity;
    }

    if is_close(size_limit, 0.0) {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_IGNORED, ORDER_STATUS_INFO_SIZE_ZERO),
        ));
    }

    if is_less(size_limit, min_size) {
        return Ok((
            exec_state,
            order_not_filled(
                ORDER_STATUS_REJECTED,
                ORDER_STATUS_INFO_MIN_SIZE_NOT_REACHED,
            ),
        ));
    }

    if size.is_finite() && is_less(size_limit, size) && !allow_partial {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_PARTIAL_FILL),
        ));
    }

    // Get acquired cash
    let acq_cash = size_limit * adj_price;
    let fees_paid = acq_cash * fees + fixed_fees;
    let final_acq_cash = add(acq_cash, -fees_paid);
    if final_acq_cash < 0.0 {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_CANT_COVER_FEES),
        ));
    }

    // Update state
    let new_cash = exec_state.cash + final_acq_cash;
    let new_position = add(exec_state.position, -size_limit);

    let (new_debt, new_free_cash);
    if new_position < 0.0 {
        let short_size = if exec_state.position < 0.0 {
            size_limit
        } else {
            new_position.abs()
        };
        let short_value = short_size * adj_price;
        new_debt = exec_state.debt + short_value;
        let free_cash_diff = add(final_acq_cash, -2.0 * short_value);
        new_free_cash = add(exec_state.free_cash, free_cash_diff);
    } else {
        new_debt = exec_state.debt;
        new_free_cash = exec_state.free_cash + final_acq_cash;
    }

    let result = OrderResult {
        size: size_limit,
        price: adj_price,
        fees: fees_paid,
        side: ORDER_SIDE_SELL,
        status: ORDER_STATUS_FILLED,
        status_info: -1,
    };
    let new_state = ExecuteOrderState {
        cash: new_cash,
        position: new_position,
        debt: new_debt,
        free_cash: new_free_cash,
    };
    Ok((new_state, result))
}

fn execute_order(
    state: &ProcessOrderState,
    order: &Order,
    rng: &mut impl Rng,
) -> Result<(ExecuteOrderState, OrderResult), PortfolioSimError> {
    // Numerical stability
    let mut cash = state.cash;
    if is_close(cash, 0.0) {
        cash = 0.0;
    }
    let mut position = state.position;
    if is_close(position, 0.0) {
        position = 0.0;
    }
    let mut debt = state.debt;
    if is_close(debt, 0.0) {
        debt = 0.0;
    }
    let mut free_cash = state.free_cash;
    if is_close(free_cash, 0.0) {
        free_cash = 0.0;
    }
    let mut val_price = state.val_price;
    if is_close(val_price, 0.0) {
        val_price = 0.0;
    }
    let mut value = state.value;
    if is_close(value, 0.0) {
        value = 0.0;
    }

    let exec_state = ExecuteOrderState {
        cash,
        position,
        debt,
        free_cash,
    };

    // Ignore order
    if order.size.is_nan() {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_IGNORED, ORDER_STATUS_INFO_SIZE_NAN),
        ));
    }
    if order.price.is_nan() {
        return Ok((
            exec_state,
            order_not_filled(ORDER_STATUS_IGNORED, ORDER_STATUS_INFO_PRICE_NAN),
        ));
    }

    // Check execution state
    if cash.is_nan() || cash < 0.0 {
        return Err(PortfolioSimError::ValueError(
            "cash cannot be NaN and must be greater than 0",
        ));
    }
    if !position.is_finite() {
        return Err(PortfolioSimError::ValueError("position must be finite"));
    }
    if !debt.is_finite() || debt < 0.0 {
        return Err(PortfolioSimError::ValueError(
            "debt must be finite and 0 or greater",
        ));
    }
    if free_cash.is_nan() {
        return Err(PortfolioSimError::ValueError("free_cash cannot be NaN"));
    }

    // Check order
    if !order.price.is_finite() || order.price <= 0.0 {
        return Err(PortfolioSimError::ValueError(
            "order.price must be finite and greater than 0",
        ));
    }
    if order.size_type < 0 || order.size_type >= 6 {
        return Err(PortfolioSimError::ValueError("order.size_type is invalid"));
    }
    if order.direction < 0 || order.direction >= 3 {
        return Err(PortfolioSimError::ValueError("order.direction is invalid"));
    }
    if order.direction == DIRECTION_LONG_ONLY && position < 0.0 {
        return Err(PortfolioSimError::ValueError(
            "position is negative but order.direction is Direction.LongOnly",
        ));
    }
    if order.direction == DIRECTION_SHORT_ONLY && position > 0.0 {
        return Err(PortfolioSimError::ValueError(
            "position is positive but order.direction is Direction.ShortOnly",
        ));
    }
    if !order.fees.is_finite() {
        return Err(PortfolioSimError::ValueError("order.fees must be finite"));
    }
    if !order.fixed_fees.is_finite() {
        return Err(PortfolioSimError::ValueError(
            "order.fixed_fees must be finite",
        ));
    }
    if !order.slippage.is_finite() || order.slippage < 0.0 {
        return Err(PortfolioSimError::ValueError(
            "order.slippage must be finite and 0 or greater",
        ));
    }
    if !order.min_size.is_finite() || order.min_size < 0.0 {
        return Err(PortfolioSimError::ValueError(
            "order.min_size must be finite and 0 or greater",
        ));
    }
    if order.max_size.is_nan() || order.max_size <= 0.0 {
        return Err(PortfolioSimError::ValueError(
            "order.max_size must be greater than 0",
        ));
    }
    if order.size_granularity.is_infinite()
        || (!order.size_granularity.is_nan() && order.size_granularity <= 0.0)
    {
        return Err(PortfolioSimError::ValueError(
            "order.size_granularity must be either NaN or finite and greater than 0",
        ));
    }
    if !order.reject_prob.is_finite() || order.reject_prob < 0.0 || order.reject_prob > 1.0 {
        return Err(PortfolioSimError::ValueError(
            "order.reject_prob must be between 0 and 1",
        ));
    }

    let mut order_size = order.size;
    let mut order_size_type = order.size_type;

    if order.direction == DIRECTION_SHORT_ONLY {
        order_size *= -1.0;
    }

    if order_size_type == SIZE_TYPE_TARGET_PERCENT {
        if value.is_nan() {
            return Ok((
                exec_state,
                order_not_filled(ORDER_STATUS_IGNORED, ORDER_STATUS_INFO_VALUE_NAN),
            ));
        }
        if value <= 0.0 {
            return Ok((
                exec_state,
                order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_VALUE_ZERO_NEG),
            ));
        }
        order_size *= value;
        order_size_type = SIZE_TYPE_TARGET_VALUE;
    }

    if order_size_type == SIZE_TYPE_VALUE || order_size_type == SIZE_TYPE_TARGET_VALUE {
        if val_price.is_nan() {
            return Ok((
                exec_state,
                order_not_filled(ORDER_STATUS_IGNORED, ORDER_STATUS_INFO_VAL_PRICE_NAN),
            ));
        }
        if val_price.is_infinite() || val_price <= 0.0 {
            return Err(PortfolioSimError::ValueError(
                "val_price_now must be finite and greater than 0",
            ));
        }
        order_size /= val_price;
        if order_size_type == SIZE_TYPE_VALUE {
            order_size_type = SIZE_TYPE_AMOUNT;
        } else {
            order_size_type = SIZE_TYPE_TARGET_AMOUNT;
        }
    }

    if order_size_type == SIZE_TYPE_TARGET_AMOUNT {
        order_size -= position;
        order_size_type = SIZE_TYPE_AMOUNT;
    }

    if order_size_type == SIZE_TYPE_AMOUNT {
        if order.direction == DIRECTION_SHORT_ONLY || order.direction == DIRECTION_BOTH {
            if order_size < 0.0 && order_size.is_infinite() {
                order_size = -1.0;
                order_size_type = SIZE_TYPE_PERCENT;
            }
        }
    }

    let mut percent = f64::NAN;
    if order_size_type == SIZE_TYPE_PERCENT {
        percent = order_size.abs();
        order_size = order_size.signum() * f64::INFINITY;
    }

    let (new_exec_state, order_result) = if order_size > 0.0 {
        buy(
            exec_state,
            order_size,
            order.price,
            order.direction,
            order.fees,
            order.fixed_fees,
            order.slippage,
            order.min_size,
            order.max_size,
            order.size_granularity,
            order.lock_cash,
            order.allow_partial,
            percent,
        )?
    } else {
        sell(
            exec_state,
            -order_size,
            order.price,
            order.direction,
            order.fees,
            order.fixed_fees,
            order.slippage,
            order.min_size,
            order.max_size,
            order.size_granularity,
            order.lock_cash,
            order.allow_partial,
            percent,
        )?
    };

    if order.reject_prob > 0.0 {
        if rng.gen::<f64>() < order.reject_prob {
            return Ok((
                exec_state,
                order_not_filled(ORDER_STATUS_REJECTED, ORDER_STATUS_INFO_RANDOM_EVENT),
            ));
        }
    }

    Ok((new_exec_state, order_result))
}

fn update_value(
    cash_before: f64,
    cash_now: f64,
    position_before: f64,
    position_now: f64,
    val_price_before: f64,
    price: f64,
    value_before: f64,
) -> (f64, f64) {
    let val_price_now = price;
    let cash_flow = cash_now - cash_before;
    let asset_value_before = if position_before != 0.0 {
        position_before * val_price_before
    } else {
        0.0
    };
    let asset_value_now = if position_now != 0.0 {
        position_now * val_price_now
    } else {
        0.0
    };
    let asset_value_diff = asset_value_now - asset_value_before;
    let value_now = value_before + cash_flow + asset_value_diff;
    (val_price_now, value_now)
}

/// Fill an order record into a pre-allocated OrderRecord buffer.
#[inline]
fn fill_order_record(
    records: &mut [OrderRecord],
    idx: usize,
    i: i64,
    col: i64,
    order_result: &OrderResult,
) {
    records[idx] = OrderRecord {
        id: idx as i64,
        col,
        idx: i,
        size: order_result.size,
        price: order_result.price,
        fees: order_result.fees,
        side: order_result.side,
    };
}

/// Fill a log record using raw pointer writes (because log_dt has mixed bool/f64/i64 fields).
#[inline]
unsafe fn fill_log_record_raw(
    log_data: *mut u8,
    offsets: &LogFieldOffsets,
    record_idx: usize,
    i: i64,
    col: i64,
    group: i64,
    state: &ProcessOrderState,
    order: &Order,
    new_cash: f64,
    new_position: f64,
    new_debt: f64,
    new_free_cash: f64,
    new_val_price: f64,
    new_value: f64,
    order_result: &OrderResult,
    order_id: i64,
) {
    let base = log_data.add(record_idx * offsets.itemsize);
    *(base.add(offsets.id) as *mut i64) = record_idx as i64;
    *(base.add(offsets.group) as *mut i64) = group;
    *(base.add(offsets.col) as *mut i64) = col;
    *(base.add(offsets.idx) as *mut i64) = i;
    *(base.add(offsets.cash) as *mut f64) = state.cash;
    *(base.add(offsets.position) as *mut f64) = state.position;
    *(base.add(offsets.debt) as *mut f64) = state.debt;
    *(base.add(offsets.free_cash) as *mut f64) = state.free_cash;
    *(base.add(offsets.val_price) as *mut f64) = state.val_price;
    *(base.add(offsets.value) as *mut f64) = state.value;
    *(base.add(offsets.req_size) as *mut f64) = order.size;
    *(base.add(offsets.req_price) as *mut f64) = order.price;
    *(base.add(offsets.req_size_type) as *mut i64) = order.size_type;
    *(base.add(offsets.req_direction) as *mut i64) = order.direction;
    *(base.add(offsets.req_fees) as *mut f64) = order.fees;
    *(base.add(offsets.req_fixed_fees) as *mut f64) = order.fixed_fees;
    *(base.add(offsets.req_slippage) as *mut f64) = order.slippage;
    *(base.add(offsets.req_min_size) as *mut f64) = order.min_size;
    *(base.add(offsets.req_max_size) as *mut f64) = order.max_size;
    *(base.add(offsets.req_size_granularity) as *mut f64) = order.size_granularity;
    *(base.add(offsets.req_reject_prob) as *mut f64) = order.reject_prob;
    *(base.add(offsets.req_lock_cash) as *mut u8) = order.lock_cash as u8;
    *(base.add(offsets.req_allow_partial) as *mut u8) = order.allow_partial as u8;
    *(base.add(offsets.req_raise_reject) as *mut u8) = order.raise_reject as u8;
    *(base.add(offsets.req_log) as *mut u8) = order.log as u8;
    *(base.add(offsets.new_cash) as *mut f64) = new_cash;
    *(base.add(offsets.new_position) as *mut f64) = new_position;
    *(base.add(offsets.new_debt) as *mut f64) = new_debt;
    *(base.add(offsets.new_free_cash) as *mut f64) = new_free_cash;
    *(base.add(offsets.new_val_price) as *mut f64) = new_val_price;
    *(base.add(offsets.new_value) as *mut f64) = new_value;
    *(base.add(offsets.res_size) as *mut f64) = order_result.size;
    *(base.add(offsets.res_price) as *mut f64) = order_result.price;
    *(base.add(offsets.res_fees) as *mut f64) = order_result.fees;
    *(base.add(offsets.res_side) as *mut i64) = order_result.side;
    *(base.add(offsets.res_status) as *mut i64) = order_result.status;
    *(base.add(offsets.res_status_info) as *mut i64) = order_result.status_info;
    *(base.add(offsets.order_id) as *mut i64) = order_id;
}

/// Process an order: execute it, fill records, and return updated state.
fn process_order(
    i: i64,
    col: i64,
    group: i64,
    state: &ProcessOrderState,
    do_update_value: bool,
    order: &Order,
    order_records: &mut Vec<OrderRecord>,
    log_data: Option<(*mut u8, &LogFieldOffsets, usize)>,
    rng: &mut impl Rng,
) -> Result<(OrderResult, ProcessOrderState), PortfolioSimError> {
    let (exec_state, order_result) = execute_order(state, order, rng)?;

    // Raise if rejected
    let is_rejected = order_result.status == ORDER_STATUS_REJECTED;
    if is_rejected && order.raise_reject {
        return Err(PortfolioSimError::RejectedOrder(rejected_order_message(
            &order_result,
        )));
    }

    // Update value
    let is_filled = order_result.status == ORDER_STATUS_FILLED;
    let (new_val_price, new_value) = if is_filled && do_update_value {
        update_value(
            state.cash,
            exec_state.cash,
            state.position,
            exec_state.position,
            state.val_price,
            order_result.price,
            state.value,
        )
    } else {
        (state.val_price, state.value)
    };

    let mut new_oidx = state.oidx;
    if is_filled {
        let oidx = state.oidx as usize;
        if oidx >= order_records.capacity() {
            return Err(PortfolioSimError::OrderRecordsOutOfRange);
        }
        if oidx >= order_records.len() {
            order_records.push(OrderRecord {
                id: 0,
                col: 0,
                idx: 0,
                size: 0.0,
                price: 0.0,
                fees: 0.0,
                side: 0,
            });
        }
        fill_order_record(order_records, oidx, i, col, &order_result);
        new_oidx += 1;
    }

    let mut new_lidx = state.lidx;
    if order.log {
        if let Some((log_ptr, log_offsets, log_capacity)) = log_data {
            let lidx = state.lidx as usize;
            if lidx >= log_capacity {
                return Err(PortfolioSimError::LogRecordsOutOfRange);
            }
            unsafe {
                fill_log_record_raw(
                    log_ptr,
                    log_offsets,
                    lidx,
                    i,
                    col,
                    group,
                    state,
                    order,
                    exec_state.cash,
                    exec_state.position,
                    exec_state.debt,
                    exec_state.free_cash,
                    new_val_price,
                    new_value,
                    &order_result,
                    if is_filled { state.oidx } else { -1 },
                );
            }
            new_lidx += 1;
        }
    }

    let new_state = ProcessOrderState {
        cash: exec_state.cash,
        position: exec_state.position,
        debt: exec_state.debt,
        free_cash: exec_state.free_cash,
        val_price: new_val_price,
        value: new_value,
        oidx: new_oidx,
        lidx: new_lidx,
    };

    Ok((order_result, new_state))
}

fn rejected_order_message(order_result: &OrderResult) -> &'static str {
    match order_result.status_info {
        ORDER_STATUS_INFO_SIZE_NAN => "Size is NaN",
        ORDER_STATUS_INFO_PRICE_NAN => "Price is NaN",
        ORDER_STATUS_INFO_VAL_PRICE_NAN => "Asset valuation price is NaN",
        ORDER_STATUS_INFO_VALUE_NAN => "Asset/group value is NaN",
        ORDER_STATUS_INFO_VALUE_ZERO_NEG => "Asset/group value is zero or negative",
        ORDER_STATUS_INFO_SIZE_ZERO => "Size is zero",
        ORDER_STATUS_INFO_NO_CASH_SHORT => "Not enough cash to short",
        ORDER_STATUS_INFO_NO_CASH_LONG => "Not enough cash to long",
        ORDER_STATUS_INFO_NO_OPEN_POSITION => "No open position to reduce/close",
        ORDER_STATUS_INFO_MAX_SIZE_EXCEEDED => "Size is greater than maximum allowed",
        ORDER_STATUS_INFO_RANDOM_EVENT => "Random event happened",
        ORDER_STATUS_INFO_CANT_COVER_FEES => "Not enough cash to cover fees",
        ORDER_STATUS_INFO_MIN_SIZE_NOT_REACHED => "Final size is less than minimum allowed",
        ORDER_STATUS_INFO_PARTIAL_FILL => "Final size is less than requested",
        _ => "Rejected order",
    }
}

// ############# Approximate order value ############# //

fn approx_order_value(
    size: f64,
    size_type: i64,
    direction: i64,
    cash_now: f64,
    position: f64,
    free_cash: f64,
    val_price: f64,
    value: f64,
) -> f64 {
    let order_value;
    let asset_value_now = position * val_price;
    let mut size_now = size;
    if direction == DIRECTION_SHORT_ONLY {
        size_now *= -1.0;
    }
    if size_type == SIZE_TYPE_AMOUNT {
        order_value = size_now * val_price;
    } else if size_type == SIZE_TYPE_VALUE {
        order_value = size_now;
    } else if size_type == SIZE_TYPE_PERCENT {
        if size_now >= 0.0 {
            order_value = size_now * cash_now;
        } else if direction == DIRECTION_LONG_ONLY {
            order_value = size_now * asset_value_now;
        } else {
            order_value = size_now * (2.0 * asset_value_now.max(0.0) + free_cash.max(0.0));
        }
    } else if size_type == SIZE_TYPE_TARGET_AMOUNT {
        order_value = size_now * val_price - asset_value_now;
    } else if size_type == SIZE_TYPE_TARGET_VALUE {
        order_value = size_now - asset_value_now;
    } else if size_type == SIZE_TYPE_TARGET_PERCENT {
        order_value = size_now * value - asset_value_now;
    } else {
        order_value = f64::NAN;
    }
    order_value
}

// ############# Group value ############# //

fn get_group_value(
    from_col: usize,
    to_col: usize,
    cash_now: f64,
    last_position: &[f64],
    last_val_price: &[f64],
) -> f64 {
    let mut value = cash_now;
    for col in from_col..to_col {
        if last_position[col] != 0.0 {
            value += last_position[col] * last_val_price[col];
        }
    }
    value
}

// ############# PyO3 exports: Core order functions ############# //

#[pyfunction]
#[pyo3(name = "order_not_filled_rs")]
pub fn order_not_filled_py(status: i64, status_info: i64) -> OrderResult {
    order_not_filled(status, status_info)
}

#[pyfunction]
#[pyo3(name = "buy_rs")]
#[pyo3(signature = (
    exec_state,
    size,
    price,
    direction = DIRECTION_BOTH,
    fees = 0.0,
    fixed_fees = 0.0,
    slippage = 0.0,
    min_size = 0.0,
    max_size = f64::INFINITY,
    size_granularity = f64::NAN,
    lock_cash = false,
    allow_partial = true,
    percent = f64::NAN,
))]
pub fn buy_py(
    py: Python<'_>,
    exec_state: ExecuteOrderState,
    size: f64,
    price: f64,
    direction: i64,
    fees: f64,
    fixed_fees: f64,
    slippage: f64,
    min_size: f64,
    max_size: f64,
    size_granularity: f64,
    lock_cash: bool,
    allow_partial: bool,
    percent: f64,
) -> PyResult<(ExecuteOrderState, OrderResult)> {
    buy(
        exec_state,
        size,
        price,
        direction,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        lock_cash,
        allow_partial,
        percent,
    )
    .map_err(|err| portfolio_sim_error_to_pyerr(py, err))
}

#[pyfunction]
#[pyo3(name = "sell_rs")]
#[pyo3(signature = (
    exec_state,
    size,
    price,
    direction = DIRECTION_BOTH,
    fees = 0.0,
    fixed_fees = 0.0,
    slippage = 0.0,
    min_size = 0.0,
    max_size = f64::INFINITY,
    size_granularity = f64::NAN,
    lock_cash = false,
    allow_partial = true,
    percent = f64::NAN,
))]
pub fn sell_py(
    py: Python<'_>,
    exec_state: ExecuteOrderState,
    size: f64,
    price: f64,
    direction: i64,
    fees: f64,
    fixed_fees: f64,
    slippage: f64,
    min_size: f64,
    max_size: f64,
    size_granularity: f64,
    lock_cash: bool,
    allow_partial: bool,
    percent: f64,
) -> PyResult<(ExecuteOrderState, OrderResult)> {
    sell(
        exec_state,
        size,
        price,
        direction,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        lock_cash,
        allow_partial,
        percent,
    )
    .map_err(|err| portfolio_sim_error_to_pyerr(py, err))
}

#[pyfunction]
#[pyo3(name = "execute_order_rs")]
pub fn execute_order_py(
    py: Python<'_>,
    state: ProcessOrderState,
    order: Order,
) -> PyResult<(ExecuteOrderState, OrderResult)> {
    let mut rng = rand::thread_rng();
    execute_order(&state, &order, &mut rng).map_err(|err| portfolio_sim_error_to_pyerr(py, err))
}

#[pyfunction]
#[pyo3(name = "raise_rejected_order_rs")]
pub fn raise_rejected_order_py(py: Python<'_>, order_result: OrderResult) -> PyResult<()> {
    Err(portfolio_sim_error_to_pyerr(
        py,
        PortfolioSimError::RejectedOrder(rejected_order_message(&order_result)),
    ))
}

#[pyfunction]
#[pyo3(name = "update_value_rs")]
pub fn update_value_py(
    cash_before: f64,
    cash_now: f64,
    position_before: f64,
    position_now: f64,
    val_price_before: f64,
    price: f64,
    value_before: f64,
) -> (f64, f64) {
    update_value(
        cash_before,
        cash_now,
        position_before,
        position_now,
        val_price_before,
        price,
        value_before,
    )
}

#[pyfunction]
#[pyo3(name = "order_rs")]
#[pyo3(signature = (
    size = f64::NAN,
    price = f64::INFINITY,
    size_type = SIZE_TYPE_AMOUNT,
    direction = DIRECTION_BOTH,
    fees = 0.0,
    fixed_fees = 0.0,
    slippage = 0.0,
    min_size = 0.0,
    max_size = f64::INFINITY,
    size_granularity = f64::NAN,
    reject_prob = 0.0,
    lock_cash = false,
    allow_partial = true,
    raise_reject = false,
    log = false,
))]
pub fn order_nb_py(
    size: f64,
    price: f64,
    size_type: i64,
    direction: i64,
    fees: f64,
    fixed_fees: f64,
    slippage: f64,
    min_size: f64,
    max_size: f64,
    size_granularity: f64,
    reject_prob: f64,
    lock_cash: bool,
    allow_partial: bool,
    raise_reject: bool,
    log: bool,
) -> Order {
    Order {
        size,
        price,
        size_type,
        direction,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        reject_prob,
        lock_cash,
        allow_partial,
        raise_reject,
        log,
    }
}

#[pyfunction]
#[pyo3(name = "close_position_rs")]
#[pyo3(signature = (
    price = f64::INFINITY,
    fees = 0.0,
    fixed_fees = 0.0,
    slippage = 0.0,
    min_size = 0.0,
    max_size = f64::INFINITY,
    size_granularity = f64::NAN,
    reject_prob = 0.0,
    lock_cash = false,
    allow_partial = true,
    raise_reject = false,
    log = false,
))]
pub fn close_position_py(
    price: f64,
    fees: f64,
    fixed_fees: f64,
    slippage: f64,
    min_size: f64,
    max_size: f64,
    size_granularity: f64,
    reject_prob: f64,
    lock_cash: bool,
    allow_partial: bool,
    raise_reject: bool,
    log: bool,
) -> Order {
    Order {
        size: 0.0,
        price,
        size_type: SIZE_TYPE_TARGET_AMOUNT,
        direction: DIRECTION_BOTH,
        fees,
        fixed_fees,
        slippage,
        min_size,
        max_size,
        size_granularity,
        reject_prob,
        lock_cash,
        allow_partial,
        raise_reject,
        log,
    }
}

#[pyfunction]
#[pyo3(name = "order_nothing_rs")]
pub fn order_nothing_py() -> Order {
    NO_ORDER
}

// ############# PyO3 exports: Validation & call sequence ############# //

#[pyfunction]
#[pyo3(name = "check_group_lens_rs")]
pub fn check_group_lens_py(group_lens: PyReadonlyArray1<'_, i64>, n_cols: usize) -> PyResult<()> {
    let gl = array1_as_slice_cow(&group_lens);
    let mut total: i64 = 0;
    for &g in gl.as_ref() {
        total += g;
    }
    if total as usize != n_cols {
        return Err(PyValueError::new_err(
            "group_lens has incorrect total number of columns",
        ));
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "check_group_init_cash_rs")]
pub fn check_group_init_cash_py(
    group_lens: PyReadonlyArray1<'_, i64>,
    n_cols: usize,
    init_cash: PyReadonlyArray1<'_, f64>,
    cash_sharing: bool,
) -> PyResult<()> {
    let gl = array1_as_slice_cow(&group_lens);
    let ic = array1_as_slice_cow(&init_cash);
    if cash_sharing {
        if ic.len() != gl.len() {
            return Err(PyValueError::new_err(
                "If cash sharing is enabled, init_cash must match the number of groups",
            ));
        }
    } else {
        if ic.len() != n_cols {
            return Err(PyValueError::new_err(
                "If cash sharing is disabled, init_cash must match the number of columns",
            ));
        }
    }
    Ok(())
}

#[pyfunction]
#[pyo3(name = "is_grouped_rs")]
pub fn is_grouped_py(group_lens: PyReadonlyArray1<'_, i64>) -> bool {
    let gl = array1_as_slice_cow(&group_lens);
    gl.iter().any(|&g| g > 1)
}

#[pyfunction]
#[pyo3(name = "shuffle_call_seq_rs")]
#[pyo3(signature = (call_seq, group_lens, seed=None))]
pub fn shuffle_call_seq_py<'py>(
    py: Python<'py>,
    mut call_seq: numpy::PyReadwriteArray2<'py, i64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    seed: Option<u64>,
) -> PyResult<()> {
    let gl = array1_as_slice_cow(&group_lens);
    let gl_usize = validate_group_lens_raw(&gl).map_err(PyValueError::new_err)?;
    let nrows = call_seq.shape()[0];
    let ncols = call_seq.shape()[1];

    let data = call_seq
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("call_seq must be contiguous"))?;

    let mut rng: Box<dyn rand::RngCore> = match seed {
        Some(s) => Box::new(ChaCha8Rng::seed_from_u64(s)),
        None => Box::new(rand::thread_rng()),
    };

    let mut from_col: usize = 0;
    for group in 0..gl_usize.len() {
        let group_len = gl_usize[group];
        for i in 0..nrows {
            let start = i * ncols + from_col;
            let slice = &mut data[start..start + group_len];
            // Fisher-Yates shuffle
            for j in (1..slice.len()).rev() {
                let k = rng.gen_range(0..=j);
                slice.swap(j, k);
            }
        }
        from_col += group_len;
    }

    Ok(())
}

#[pyfunction]
#[pyo3(name = "build_call_seq_rs")]
pub fn build_call_seq_py<'py>(
    py: Python<'py>,
    target_shape: (usize, usize),
    group_lens: PyReadonlyArray1<'py, i64>,
    call_seq_type: i64,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let gl = array1_as_slice_cow(&group_lens);
    let (nrows, ncols) = target_shape;

    let result = py
        .allow_threads(|| {
            let gl_usize = validate_group_lens_raw(&gl)?;
            let mut out = vec![0i64; nrows * ncols];
            let mut rng = rand::thread_rng();
            let mut from_col: usize = 0;
            for group in 0..gl_usize.len() {
                let group_len = gl_usize[group];
                let to_col = from_col + group_len;
                for i in 0..nrows {
                    for k in 0..group_len {
                        let idx = i * ncols + from_col + k;
                        if call_seq_type == CALL_SEQ_REVERSED {
                            out[idx] = (group_len - 1 - k) as i64;
                        } else {
                            out[idx] = k as i64;
                        }
                    }
                    if call_seq_type == CALL_SEQ_RANDOM {
                        let start = i * ncols + from_col;
                        let slice = &mut out[start..start + group_len];
                        for j in (1..slice.len()).rev() {
                            let k = rng.gen_range(0..=j);
                            slice.swap(j, k);
                        }
                    }
                }
                from_col = to_col;
            }
            Ok::<_, String>(out)
        })
        .map_err(PyValueError::new_err)?;

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "get_group_value_rs")]
pub fn get_group_value_py(
    from_col: usize,
    to_col: usize,
    cash_now: f64,
    last_position: PyReadonlyArray1<'_, f64>,
    last_val_price: PyReadonlyArray1<'_, f64>,
) -> f64 {
    let pos = array1_as_slice_cow(&last_position);
    let vp = array1_as_slice_cow(&last_val_price);
    get_group_value(from_col, to_col, cash_now, pos.as_ref(), vp.as_ref())
}

#[pyfunction]
#[pyo3(name = "approx_order_value_rs")]
pub fn approx_order_value_py(
    size: f64,
    size_type: i64,
    direction: i64,
    cash_now: f64,
    position: f64,
    free_cash: f64,
    val_price: f64,
    value: f64,
) -> f64 {
    approx_order_value(
        size, size_type, direction, cash_now, position, free_cash, val_price, value,
    )
}

fn validate_group_lens_raw(gl: &[i64]) -> Result<Vec<usize>, String> {
    let mut out = Vec::with_capacity(gl.len());
    for &g in gl {
        if g < 0 {
            return Err("group_lens must be non-negative".to_string());
        }
        out.push(g as usize);
    }
    Ok(out)
}

// ############# PyO3 exports: Scalar helpers ############# //

#[pyfunction]
#[pyo3(name = "simulate_from_orders_rs")]
#[pyo3(signature = (
    target_shape,
    group_lens,
    init_cash,
    call_seq,
    size,
    price,
    size_type,
    direction,
    fees,
    fixed_fees,
    slippage,
    min_size,
    max_size,
    size_granularity,
    reject_prob,
    lock_cash,
    allow_partial,
    raise_reject,
    log,
    val_price,
    close,
    auto_call_seq = false,
    ffill_val_price = true,
    update_value = false,
    max_orders = None,
    max_logs = 0,
    seed = None,
    flex_2d = true,
))]
pub fn simulate_from_orders_py<'py>(
    py: Python<'py>,
    target_shape: (usize, usize),
    group_lens: PyReadonlyArray1<'py, i64>,
    init_cash: PyReadonlyArray1<'py, f64>,
    mut call_seq: numpy::PyReadwriteArray2<'py, i64>,
    // Pre-broadcast 2D arrays
    size: PyReadonlyArrayDyn<'py, f64>,
    price: PyReadonlyArrayDyn<'py, f64>,
    size_type: PyReadonlyArrayDyn<'py, i64>,
    direction: PyReadonlyArrayDyn<'py, i64>,
    fees: PyReadonlyArrayDyn<'py, f64>,
    fixed_fees: PyReadonlyArrayDyn<'py, f64>,
    slippage: PyReadonlyArrayDyn<'py, f64>,
    min_size: PyReadonlyArrayDyn<'py, f64>,
    max_size: PyReadonlyArrayDyn<'py, f64>,
    size_granularity: PyReadonlyArrayDyn<'py, f64>,
    reject_prob: PyReadonlyArrayDyn<'py, f64>,
    lock_cash: PyReadonlyArrayDyn<'py, bool>,
    allow_partial: PyReadonlyArrayDyn<'py, bool>,
    raise_reject: PyReadonlyArrayDyn<'py, bool>,
    log: PyReadonlyArrayDyn<'py, bool>,
    val_price: PyReadonlyArrayDyn<'py, f64>,
    close: PyReadonlyArrayDyn<'py, f64>,
    auto_call_seq: bool,
    ffill_val_price: bool,
    update_value: bool,
    max_orders: Option<usize>,
    max_logs: usize,
    seed: Option<u64>,
    flex_2d: bool,
) -> PyResult<(Bound<'py, PyArray1<OrderRecord>>, Bound<'py, pyo3::PyAny>)> {
    let (nrows, ncols) = target_shape;

    // Extract slices
    let gl_cow = array1_as_slice_cow(&group_lens);
    let ic_cow = array1_as_slice_cow(&init_cash);
    let cs_cow = {
        let cs_view = call_seq.as_array();
        if cs_view.is_standard_layout() {
            Cow::Borrowed(call_seq.as_slice()?)
        } else {
            Cow::Owned(cs_view.iter().copied().collect())
        }
    };
    let size_flex = FlexArray::from_pyarray("size", &size, nrows, ncols, flex_2d)?;
    let price_flex = FlexArray::from_pyarray("price", &price, nrows, ncols, flex_2d)?;
    let st_flex = FlexArray::from_pyarray("size_type", &size_type, nrows, ncols, flex_2d)?;
    let dir_flex = FlexArray::from_pyarray("direction", &direction, nrows, ncols, flex_2d)?;
    let fees_flex = FlexArray::from_pyarray("fees", &fees, nrows, ncols, flex_2d)?;
    let ff_flex = FlexArray::from_pyarray("fixed_fees", &fixed_fees, nrows, ncols, flex_2d)?;
    let slip_flex = FlexArray::from_pyarray("slippage", &slippage, nrows, ncols, flex_2d)?;
    let mins_flex = FlexArray::from_pyarray("min_size", &min_size, nrows, ncols, flex_2d)?;
    let maxs_flex = FlexArray::from_pyarray("max_size", &max_size, nrows, ncols, flex_2d)?;
    let sg_flex = FlexArray::from_pyarray("size_granularity", &size_granularity, nrows, ncols, flex_2d)?;
    let rp_flex = FlexArray::from_pyarray("reject_prob", &reject_prob, nrows, ncols, flex_2d)?;
    let lc_flex = FlexArray::from_pyarray("lock_cash", &lock_cash, nrows, ncols, flex_2d)?;
    let ap_flex = FlexArray::from_pyarray("allow_partial", &allow_partial, nrows, ncols, flex_2d)?;
    let rr_flex = FlexArray::from_pyarray("raise_reject", &raise_reject, nrows, ncols, flex_2d)?;
    let log_flex = FlexArray::from_pyarray("log", &log, nrows, ncols, flex_2d)?;
    let vp_flex = FlexArray::from_pyarray("val_price", &val_price, nrows, ncols, flex_2d)?;
    let close_flex = FlexArray::from_pyarray("close", &close, nrows, ncols, flex_2d)?;

    // Allocate log records via numpy (because of mixed types)
    let cash_sharing = gl_cow.iter().any(|&g| g > 1);
    let total_cols: i64 = gl_cow.iter().sum();
    if total_cols != ncols as i64 {
        return Err(PyValueError::new_err(
            "group_lens has incorrect total number of columns",
        ));
    }
    if cash_sharing {
        if ic_cow.len() != gl_cow.len() {
            return Err(PyValueError::new_err(
                "If cash sharing is enabled, init_cash must match the number of groups",
            ));
        }
    } else if ic_cow.len() != ncols {
        return Err(PyValueError::new_err(
            "If cash sharing is disabled, init_cash must match the number of columns",
        ));
    }

    let log_dt = py
        .import_bound("vectorbt.portfolio.enums")?
        .getattr("log_dt")?;
    let effective_max_logs = if max_logs == 0 { 1 } else { max_logs };
    let log_arr = numpy_empty(py, effective_max_logs, &log_dt)?;
    let log_offsets = log_record_offsets(&log_arr)?;
    let (log_data, _log_itemsize, _) = unsafe { array_raw_parts(&log_arr)? };
    let log_data_usize = ptr_to_usize(log_data);

    let max_ord = max_orders.unwrap_or(nrows * ncols);

    // Run the simulation
    let (order_records, lidx, call_seq_out) = py
        .allow_threads(|| {
            if !cash_sharing && !auto_call_seq {
                let (order_records, lidx) = simulate_from_orders_non_shared_inner(
                    nrows,
                    ncols,
                    ic_cow.as_ref(),
                    &size_flex,
                    &price_flex,
                    &st_flex,
                    &dir_flex,
                    &fees_flex,
                    &ff_flex,
                    &slip_flex,
                    &mins_flex,
                    &maxs_flex,
                    &sg_flex,
                    &rp_flex,
                    &lc_flex,
                    &ap_flex,
                    &rr_flex,
                    &log_flex,
                    &vp_flex,
                    &close_flex,
                    ffill_val_price,
                    update_value,
                    max_ord,
                    effective_max_logs,
                    usize_to_mut_ptr(log_data_usize),
                    &log_offsets,
                    seed,
                )?;
                return Ok((order_records, lidx, None));
            }
            simulate_from_orders_inner(
                nrows,
                ncols,
                gl_cow.as_ref(),
                ic_cow.as_ref(),
                cs_cow.as_ref(),
                &size_flex,
                &price_flex,
                &st_flex,
                &dir_flex,
                &fees_flex,
                &ff_flex,
                &slip_flex,
                &mins_flex,
                &maxs_flex,
                &sg_flex,
                &rp_flex,
                &lc_flex,
                &ap_flex,
                &rr_flex,
                &log_flex,
                &vp_flex,
                &close_flex,
                auto_call_seq,
                ffill_val_price,
                update_value,
                max_ord,
                effective_max_logs,
                usize_to_mut_ptr(log_data_usize),
                &log_offsets,
                seed,
            )
        })
        .map_err(|err| portfolio_sim_error_to_pyerr(py, err))?;
    drop(cs_cow);

    if let Some(call_seq_out) = call_seq_out {
        let mut cs_view = call_seq.as_array_mut();
        for i in 0..nrows {
            for col in 0..ncols {
                cs_view[(i, col)] = call_seq_out[i * ncols + col];
            }
        }
    }

    let order_arr = PyArray1::from_vec_bound(py, order_records);

    // Slice log array to actual count
    let log_arr_sliced = if lidx < effective_max_logs {
        log_arr.call_method1(
            "__getitem__",
            (pyo3::types::PySlice::new_bound(py, 0, lidx as isize, 1),),
        )?
    } else {
        log_arr
    };

    Ok((order_arr, log_arr_sliced))
}

fn get_trade_stats(
    size: f64,
    entry_price: f64,
    entry_fees: f64,
    exit_price: f64,
    exit_fees: f64,
    direction: i64,
) -> (f64, f64) {
    let entry_val = size * entry_price;
    let exit_val = size * exit_price;
    let mut val_diff = add(exit_val, -entry_val);
    if val_diff != 0.0 && direction == TRADE_DIRECTION_SHORT {
        val_diff *= -1.0;
    }
    let pnl = val_diff - entry_fees - exit_fees;
    let ret = pnl / entry_val;
    (pnl, ret)
}
#[inline(always)]
fn read_signals(
    entry: bool,
    exit: bool,
    dir: i64,
    long_entry: bool,
    long_exit: bool,
    short_entry: bool,
    short_exit: bool,
) -> (bool, bool, bool, bool) {
    if entry || exit {
        if dir == DIRECTION_LONG_ONLY {
            (
                entry || long_entry,
                exit || long_exit,
                short_entry,
                short_exit,
            )
        } else if dir == DIRECTION_SHORT_ONLY {
            (
                long_entry,
                long_exit,
                entry || short_entry,
                exit || short_exit,
            )
        } else {
            // DIRECTION_BOTH
            (
                entry || long_entry,
                long_exit,
                exit || short_entry,
                short_exit,
            )
        }
    } else {
        (long_entry, long_exit, short_entry, short_exit)
    }
}

// ############# simulate_from_signals_rs ############# //

#[pyfunction]
#[pyo3(name = "simulate_from_signals_rs")]
#[pyo3(signature = (
    target_shape,
    group_lens,
    init_cash,
    call_seq,
    entries,
    exits,
    direction,
    long_entries,
    long_exits,
    short_entries,
    short_exits,
    size,
    price,
    size_type,
    fees,
    fixed_fees,
    slippage,
    min_size,
    max_size,
    size_granularity,
    reject_prob,
    lock_cash,
    allow_partial,
    raise_reject,
    log,
    accumulate,
    upon_long_conflict,
    upon_short_conflict,
    upon_dir_conflict,
    upon_opposite_entry,
    val_price,
    open,
    high,
    low,
    close,
    sl_stop,
    sl_trail,
    tp_stop,
    stop_entry_price,
    stop_exit_price,
    upon_stop_exit,
    upon_stop_update,
    use_stops = true,
    auto_call_seq = false,
    ffill_val_price = true,
    update_value = false,
    max_orders = None,
    max_logs = 0,
    seed = None,
    flex_2d = true,
))]
pub fn simulate_from_signals_py<'py>(
    py: Python<'py>,
    target_shape: (usize, usize),
    group_lens: PyReadonlyArray1<'py, i64>,
    init_cash: PyReadonlyArray1<'py, f64>,
    mut call_seq: numpy::PyReadwriteArray2<'py, i64>,
    // Signal arrays - pre-broadcast 2D
    entries: PyReadonlyArrayDyn<'py, bool>,
    exits: PyReadonlyArrayDyn<'py, bool>,
    direction: PyReadonlyArrayDyn<'py, i64>,
    long_entries: PyReadonlyArrayDyn<'py, bool>,
    long_exits: PyReadonlyArrayDyn<'py, bool>,
    short_entries: PyReadonlyArrayDyn<'py, bool>,
    short_exits: PyReadonlyArrayDyn<'py, bool>,
    // Order params
    size: PyReadonlyArrayDyn<'py, f64>,
    price: PyReadonlyArrayDyn<'py, f64>,
    size_type: PyReadonlyArrayDyn<'py, i64>,
    fees: PyReadonlyArrayDyn<'py, f64>,
    fixed_fees: PyReadonlyArrayDyn<'py, f64>,
    slippage: PyReadonlyArrayDyn<'py, f64>,
    min_size: PyReadonlyArrayDyn<'py, f64>,
    max_size: PyReadonlyArrayDyn<'py, f64>,
    size_granularity: PyReadonlyArrayDyn<'py, f64>,
    reject_prob: PyReadonlyArrayDyn<'py, f64>,
    lock_cash: PyReadonlyArrayDyn<'py, bool>,
    allow_partial: PyReadonlyArrayDyn<'py, bool>,
    raise_reject: PyReadonlyArrayDyn<'py, bool>,
    log: PyReadonlyArrayDyn<'py, bool>,
    // Signal conflict/resolution params
    accumulate: PyReadonlyArrayDyn<'py, i64>,
    upon_long_conflict: PyReadonlyArrayDyn<'py, i64>,
    upon_short_conflict: PyReadonlyArrayDyn<'py, i64>,
    upon_dir_conflict: PyReadonlyArrayDyn<'py, i64>,
    upon_opposite_entry: PyReadonlyArrayDyn<'py, i64>,
    // Price/valuation
    val_price: PyReadonlyArrayDyn<'py, f64>,
    open: PyReadonlyArrayDyn<'py, f64>,
    high: PyReadonlyArrayDyn<'py, f64>,
    low: PyReadonlyArrayDyn<'py, f64>,
    close: PyReadonlyArrayDyn<'py, f64>,
    // Stop params
    sl_stop: PyReadonlyArrayDyn<'py, f64>,
    sl_trail: PyReadonlyArrayDyn<'py, bool>,
    tp_stop: PyReadonlyArrayDyn<'py, f64>,
    stop_entry_price: PyReadonlyArrayDyn<'py, i64>,
    stop_exit_price: PyReadonlyArrayDyn<'py, i64>,
    upon_stop_exit: PyReadonlyArrayDyn<'py, i64>,
    upon_stop_update: PyReadonlyArrayDyn<'py, i64>,
    // Scalar flags
    use_stops: bool,
    auto_call_seq: bool,
    ffill_val_price: bool,
    update_value: bool,
    max_orders: Option<usize>,
    max_logs: usize,
    seed: Option<u64>,
    flex_2d: bool,
) -> PyResult<(Bound<'py, PyArray1<OrderRecord>>, Bound<'py, pyo3::PyAny>)> {
    let (nrows, ncols) = target_shape;

    let gl_cow = array1_as_slice_cow(&group_lens);
    let ic_cow = array1_as_slice_cow(&init_cash);
    let cs_cow = {
        let cs_view = call_seq.as_array();
        if cs_view.is_standard_layout() {
            Cow::Borrowed(call_seq.as_slice().unwrap())
        } else {
            Cow::Owned(cs_view.iter().copied().collect())
        }
    };

    let entries_flex = FlexArray::from_pyarray("entries", &entries, nrows, ncols, flex_2d)?;
    let exits_flex = FlexArray::from_pyarray("exits", &exits, nrows, ncols, flex_2d)?;
    let dir_flex = FlexArray::from_pyarray("direction", &direction, nrows, ncols, flex_2d)?;
    let le_flex = FlexArray::from_pyarray("long_entries", &long_entries, nrows, ncols, flex_2d)?;
    let lx_flex = FlexArray::from_pyarray("long_exits", &long_exits, nrows, ncols, flex_2d)?;
    let se_flex = FlexArray::from_pyarray("short_entries", &short_entries, nrows, ncols, flex_2d)?;
    let sx_flex = FlexArray::from_pyarray("short_exits", &short_exits, nrows, ncols, flex_2d)?;
    let size_flex = FlexArray::from_pyarray("size", &size, nrows, ncols, flex_2d)?;
    let price_flex = FlexArray::from_pyarray("price", &price, nrows, ncols, flex_2d)?;
    let st_flex = FlexArray::from_pyarray("size_type", &size_type, nrows, ncols, flex_2d)?;
    let fees_flex = FlexArray::from_pyarray("fees", &fees, nrows, ncols, flex_2d)?;
    let ff_flex = FlexArray::from_pyarray("fixed_fees", &fixed_fees, nrows, ncols, flex_2d)?;
    let slip_flex = FlexArray::from_pyarray("slippage", &slippage, nrows, ncols, flex_2d)?;
    let mins_flex = FlexArray::from_pyarray("min_size", &min_size, nrows, ncols, flex_2d)?;
    let maxs_flex = FlexArray::from_pyarray("max_size", &max_size, nrows, ncols, flex_2d)?;
    let sg_flex = FlexArray::from_pyarray("size_granularity", &size_granularity, nrows, ncols, flex_2d)?;
    let rp_flex = FlexArray::from_pyarray("reject_prob", &reject_prob, nrows, ncols, flex_2d)?;
    let lc_flex = FlexArray::from_pyarray("lock_cash", &lock_cash, nrows, ncols, flex_2d)?;
    let ap_flex = FlexArray::from_pyarray("allow_partial", &allow_partial, nrows, ncols, flex_2d)?;
    let rr_flex = FlexArray::from_pyarray("raise_reject", &raise_reject, nrows, ncols, flex_2d)?;
    let log_flex = FlexArray::from_pyarray("log", &log, nrows, ncols, flex_2d)?;
    let acc_flex = FlexArray::from_pyarray("accumulate", &accumulate, nrows, ncols, flex_2d)?;
    let ulc_flex = FlexArray::from_pyarray("upon_long_conflict", &upon_long_conflict, nrows, ncols, flex_2d)?;
    let usc_flex = FlexArray::from_pyarray("upon_short_conflict", &upon_short_conflict, nrows, ncols, flex_2d)?;
    let udc_flex = FlexArray::from_pyarray("upon_dir_conflict", &upon_dir_conflict, nrows, ncols, flex_2d)?;
    let uoe_flex = FlexArray::from_pyarray("upon_opposite_entry", &upon_opposite_entry, nrows, ncols, flex_2d)?;
    let vp_flex = FlexArray::from_pyarray("val_price", &val_price, nrows, ncols, flex_2d)?;
    let open_flex = FlexArray::from_pyarray("open", &open, nrows, ncols, flex_2d)?;
    let high_flex = FlexArray::from_pyarray("high", &high, nrows, ncols, flex_2d)?;
    let low_flex = FlexArray::from_pyarray("low", &low, nrows, ncols, flex_2d)?;
    let close_flex = FlexArray::from_pyarray("close", &close, nrows, ncols, flex_2d)?;
    let sls_flex = FlexArray::from_pyarray("sl_stop", &sl_stop, nrows, ncols, flex_2d)?;
    let slt_flex = FlexArray::from_pyarray("sl_trail", &sl_trail, nrows, ncols, flex_2d)?;
    let tps_flex = FlexArray::from_pyarray("tp_stop", &tp_stop, nrows, ncols, flex_2d)?;
    let sep_flex = FlexArray::from_pyarray("stop_entry_price", &stop_entry_price, nrows, ncols, flex_2d)?;
    let sxp_flex = FlexArray::from_pyarray("stop_exit_price", &stop_exit_price, nrows, ncols, flex_2d)?;
    let use_flex = FlexArray::from_pyarray("upon_stop_exit", &upon_stop_exit, nrows, ncols, flex_2d)?;
    let usu_flex = FlexArray::from_pyarray("upon_stop_update", &upon_stop_update, nrows, ncols, flex_2d)?;

    // Validate
    let cash_sharing = gl_cow.iter().any(|&g| g > 1);
    let total_cols: i64 = gl_cow.iter().sum();
    if total_cols != ncols as i64 {
        return Err(PyValueError::new_err(
            "group_lens has incorrect total number of columns",
        ));
    }
    if cash_sharing {
        if ic_cow.len() != gl_cow.len() {
            return Err(PyValueError::new_err(
                "If cash sharing is enabled, init_cash must match the number of groups",
            ));
        }
    } else if ic_cow.len() != ncols {
        return Err(PyValueError::new_err(
            "If cash sharing is disabled, init_cash must match the number of columns",
        ));
    }

    let log_dt = py
        .import_bound("vectorbt.portfolio.enums")?
        .getattr("log_dt")?;
    let effective_max_logs = if max_logs == 0 { 1 } else { max_logs };
    let log_arr = numpy_empty(py, effective_max_logs, &log_dt)?;
    let log_offsets = log_record_offsets(&log_arr)?;
    let (log_data, _log_itemsize, _) = unsafe { array_raw_parts(&log_arr)? };
    let log_data_usize = ptr_to_usize(log_data);

    let max_ord = max_orders.unwrap_or(nrows * ncols);

    let (order_records, lidx, call_seq_out) = py
        .allow_threads(|| {
            if !cash_sharing && !auto_call_seq {
                let (order_records, lidx) = simulate_from_signals_non_shared_inner(
                    nrows,
                    ncols,
                    ic_cow.as_ref(),
                    &entries_flex,
                    &exits_flex,
                    &dir_flex,
                    &le_flex,
                    &lx_flex,
                    &se_flex,
                    &sx_flex,
                    &size_flex,
                    &price_flex,
                    &st_flex,
                    &fees_flex,
                    &ff_flex,
                    &slip_flex,
                    &mins_flex,
                    &maxs_flex,
                    &sg_flex,
                    &rp_flex,
                    &lc_flex,
                    &ap_flex,
                    &rr_flex,
                    &log_flex,
                    &acc_flex,
                    &ulc_flex,
                    &usc_flex,
                    &udc_flex,
                    &uoe_flex,
                    &vp_flex,
                    &open_flex,
                    &high_flex,
                    &low_flex,
                    &close_flex,
                    &sls_flex,
                    &slt_flex,
                    &tps_flex,
                    &sep_flex,
                    &sxp_flex,
                    &use_flex,
                    &usu_flex,
                    use_stops,
                    ffill_val_price,
                    update_value,
                    max_ord,
                    effective_max_logs,
                    usize_to_mut_ptr(log_data_usize),
                    &log_offsets,
                    seed,
                )?;
                return Ok((order_records, lidx, None));
            }
            simulate_from_signals_inner(
                nrows,
                ncols,
                gl_cow.as_ref(),
                ic_cow.as_ref(),
                cs_cow.as_ref(),
                &entries_flex,
                &exits_flex,
                &dir_flex,
                &le_flex,
                &lx_flex,
                &se_flex,
                &sx_flex,
                &size_flex,
                &price_flex,
                &st_flex,
                &fees_flex,
                &ff_flex,
                &slip_flex,
                &mins_flex,
                &maxs_flex,
                &sg_flex,
                &rp_flex,
                &lc_flex,
                &ap_flex,
                &rr_flex,
                &log_flex,
                &acc_flex,
                &ulc_flex,
                &usc_flex,
                &udc_flex,
                &uoe_flex,
                &vp_flex,
                &open_flex,
                &high_flex,
                &low_flex,
                &close_flex,
                &sls_flex,
                &slt_flex,
                &tps_flex,
                &sep_flex,
                &sxp_flex,
                &use_flex,
                &usu_flex,
                use_stops,
                auto_call_seq,
                ffill_val_price,
                update_value,
                max_ord,
                effective_max_logs,
                usize_to_mut_ptr(log_data_usize),
                &log_offsets,
                seed,
            )
        })
        .map_err(|err| portfolio_sim_error_to_pyerr(py, err))?;
    drop(cs_cow);

    if let Some(call_seq_out) = call_seq_out {
        let mut cs_view = call_seq.as_array_mut();
        for i in 0..nrows {
            for col in 0..ncols {
                cs_view[(i, col)] = call_seq_out[i * ncols + col];
            }
        }
    }

    let order_arr = PyArray1::from_vec_bound(py, order_records);

    let log_arr_sliced = if lidx < effective_max_logs {
        log_arr.call_method1(
            "__getitem__",
            (pyo3::types::PySlice::new_bound(py, 0, lidx as isize, 1),),
        )?
    } else {
        log_arr
    };

    Ok((order_arr, log_arr_sliced))
}

fn get_long_size(position_before: f64, position_now: f64) -> f64 {
    if position_before <= 0.0 && position_now <= 0.0 {
        return 0.0;
    }
    if position_before >= 0.0 && position_now < 0.0 {
        return -position_before;
    }
    if position_before < 0.0 && position_now >= 0.0 {
        return position_now;
    }
    add(position_now, -position_before)
}

#[pyfunction]
#[pyo3(name = "get_trade_stats_rs")]
pub fn get_trade_stats_py(
    size: f64,
    entry_price: f64,
    entry_fees: f64,
    exit_price: f64,
    exit_fees: f64,
    direction: i64,
) -> (f64, f64) {
    get_trade_stats(
        size,
        entry_price,
        entry_fees,
        exit_price,
        exit_fees,
        direction,
    )
}

fn get_short_size(position_before: f64, position_now: f64) -> f64 {
    if position_before >= 0.0 && position_now >= 0.0 {
        return 0.0;
    }
    if position_before >= 0.0 && position_now < 0.0 {
        return -position_now;
    }
    if position_before < 0.0 && position_now >= 0.0 {
        return position_before;
    }
    add(position_before, -position_now)
}

#[pyfunction]
#[pyo3(name = "get_entry_trades_rs")]
pub fn get_entry_trades_py<'py>(
    py: Python<'py>,
    order_records: Bound<'py, pyo3::PyAny>,
    close: PyReadonlyArray2<'py, f64>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<TradeRecord>>> {
    let offsets = order_record_offsets(&order_records)?;
    let (src_data, _itemsize, n_records) = unsafe { array_raw_parts(&order_records)? };
    let src_send = ptr_to_usize(src_data);
    let ci_cow = array1_as_slice_cow(&col_idxs);
    let cl_cow = array1_as_slice_cow(&col_lens);
    let nrows = close.shape()[0];
    let ncols = close.shape()[1];
    let close_cow = array2_as_slice_cow(&close);

    let trades = py
        .allow_threads(|| {
            get_entry_trades_inner(
                usize_to_ptr(src_send),
                &offsets,
                ci_cow.as_ref(),
                cl_cow.as_ref(),
                close_cow.as_ref(),
                nrows,
                ncols,
                n_records,
            )
        })
        .map_err(|err| portfolio_sim_error_to_pyerr(py, err))?;

    Ok(PyArray1::from_vec_bound(py, trades))
}

fn get_free_cash_diff(
    position_before: f64,
    position_now: f64,
    debt_now: f64,
    price: f64,
    fees: f64,
) -> (f64, f64) {
    let size = add(position_now, -position_before);
    let final_cash = -size * price - fees;
    if is_close(size, 0.0) {
        return (debt_now, 0.0);
    }
    if size > 0.0 {
        if position_before < 0.0 {
            let short_size = if position_now < 0.0 {
                size.abs()
            } else {
                position_before.abs()
            };
            let avg_entry_price = debt_now / position_before.abs();
            let debt_diff = short_size * avg_entry_price;
            let new_debt = add(debt_now, -debt_diff);
            let free_cash_diff = add(2.0 * debt_diff, final_cash);
            (new_debt, free_cash_diff)
        } else {
            (debt_now, final_cash)
        }
    } else {
        if position_now < 0.0 {
            let short_size = if position_before < 0.0 {
                size.abs()
            } else {
                position_now.abs()
            };
            let short_value = short_size * price;
            let new_debt = debt_now + short_value;
            let free_cash_diff = add(final_cash, -2.0 * short_value);
            (new_debt, free_cash_diff)
        } else {
            (debt_now, final_cash)
        }
    }
}

// ############# simulate_from_orders_rs ############# //

#[pyfunction]
#[pyo3(name = "get_exit_trades_rs")]
pub fn get_exit_trades_py<'py>(
    py: Python<'py>,
    order_records: Bound<'py, pyo3::PyAny>,
    close: PyReadonlyArray2<'py, f64>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<TradeRecord>>> {
    let offsets = order_record_offsets(&order_records)?;
    let (src_data, _itemsize, n_records) = unsafe { array_raw_parts(&order_records)? };
    let src_send = ptr_to_usize(src_data);
    let ci_cow = array1_as_slice_cow(&col_idxs);
    let cl_cow = array1_as_slice_cow(&col_lens);
    let nrows = close.shape()[0];
    let ncols = close.shape()[1];
    let close_cow = array2_as_slice_cow(&close);

    let trades = py
        .allow_threads(|| {
            get_exit_trades_inner(
                usize_to_ptr(src_send),
                &offsets,
                ci_cow.as_ref(),
                cl_cow.as_ref(),
                close_cow.as_ref(),
                nrows,
                ncols,
                n_records,
            )
        })
        .map_err(|err| portfolio_sim_error_to_pyerr(py, err))?;

    Ok(PyArray1::from_vec_bound(py, trades))
}

fn simulate_from_orders_non_shared_inner(
    nrows: usize,
    ncols: usize,
    init_cash: &[f64],
    size_s: &FlexArray<'_, f64>,
    price_s: &FlexArray<'_, f64>,
    size_type_s: &FlexArray<'_, i64>,
    direction_s: &FlexArray<'_, i64>,
    fees_s: &FlexArray<'_, f64>,
    fixed_fees_s: &FlexArray<'_, f64>,
    slippage_s: &FlexArray<'_, f64>,
    min_size_s: &FlexArray<'_, f64>,
    max_size_s: &FlexArray<'_, f64>,
    size_granularity_s: &FlexArray<'_, f64>,
    reject_prob_s: &FlexArray<'_, f64>,
    lock_cash_s: &FlexArray<'_, bool>,
    allow_partial_s: &FlexArray<'_, bool>,
    raise_reject_s: &FlexArray<'_, bool>,
    log_s: &FlexArray<'_, bool>,
    val_price_s: &FlexArray<'_, f64>,
    close_s: &FlexArray<'_, f64>,
    ffill_val_price: bool,
    do_update_value: bool,
    max_orders: usize,
    max_logs: usize,
    log_data: *mut u8,
    log_offsets: &LogFieldOffsets,
    seed: Option<u64>,
) -> Result<(Vec<OrderRecord>, usize), PortfolioSimError> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::seed_from_u64(rand::random()),
    };

    let mut order_records: Vec<OrderRecord> = Vec::with_capacity(max_orders.min(nrows * ncols));
    let mut oidx: i64 = 0;
    let mut lidx: i64 = 0;

    for col in 0..ncols {
        let mut cash_now = init_cash[col];
        let mut free_cash_now = init_cash[col];
        let mut last_position = 0.0f64;
        let mut last_debt = 0.0f64;
        let mut last_val_price = f64::NAN;

        for i in 0..nrows {
            let mut order_price = price_s.get(i, col);
            if order_price.is_infinite() {
                if order_price > 0.0 {
                    order_price = close_s.get(i, col);
                } else if i > 0 {
                    order_price = close_s.get(i - 1, col);
                } else {
                    order_price = f64::NAN;
                }
            }

            let mut val_price_now = val_price_s.get(i, col);
            if val_price_now.is_infinite() {
                if val_price_now > 0.0 {
                    val_price_now = order_price;
                } else if i > 0 {
                    val_price_now = close_s.get(i - 1, col);
                } else {
                    val_price_now = f64::NAN;
                }
            }
            if !val_price_now.is_nan() || !ffill_val_price {
                last_val_price = val_price_now;
            }

            let mut value_now = cash_now;
            if last_position != 0.0 {
                value_now += last_position * last_val_price;
            }

            let order = Order {
                size: size_s.get(i, col),
                price: order_price,
                size_type: size_type_s.get(i, col),
                direction: direction_s.get(i, col),
                fees: fees_s.get(i, col),
                fixed_fees: fixed_fees_s.get(i, col),
                slippage: slippage_s.get(i, col),
                min_size: min_size_s.get(i, col),
                max_size: max_size_s.get(i, col),
                size_granularity: size_granularity_s.get(i, col),
                reject_prob: reject_prob_s.get(i, col),
                lock_cash: lock_cash_s.get(i, col),
                allow_partial: allow_partial_s.get(i, col),
                raise_reject: raise_reject_s.get(i, col),
                log: log_s.get(i, col),
            };

            let state = ProcessOrderState {
                cash: cash_now,
                position: last_position,
                debt: last_debt,
                free_cash: free_cash_now,
                val_price: last_val_price,
                value: value_now,
                oidx,
                lidx,
            };

            let log_info = if max_logs > 0 {
                Some((log_data, log_offsets as &LogFieldOffsets, max_logs))
            } else {
                None
            };

            let (_order_result, new_state) = process_order(
                i as i64,
                col as i64,
                col as i64,
                &state,
                do_update_value,
                &order,
                &mut order_records,
                log_info,
                &mut rng,
            )?;

            cash_now = new_state.cash;
            free_cash_now = new_state.free_cash;
            oidx = new_state.oidx;
            lidx = new_state.lidx;
            last_position = new_state.position;
            last_debt = new_state.debt;
            if !new_state.val_price.is_nan() || !ffill_val_price {
                last_val_price = new_state.val_price;
            }
        }
    }

    order_records.truncate(oidx as usize);
    Ok((order_records, lidx as usize))
}

/// Inner simulation function that runs without GIL.
fn simulate_from_orders_inner(
    nrows: usize,
    ncols: usize,
    group_lens: &[i64],
    init_cash: &[f64],
    call_seq: &[i64],
    size_s: &FlexArray<'_, f64>,
    price_s: &FlexArray<'_, f64>,
    size_type_s: &FlexArray<'_, i64>,
    direction_s: &FlexArray<'_, i64>,
    fees_s: &FlexArray<'_, f64>,
    fixed_fees_s: &FlexArray<'_, f64>,
    slippage_s: &FlexArray<'_, f64>,
    min_size_s: &FlexArray<'_, f64>,
    max_size_s: &FlexArray<'_, f64>,
    size_granularity_s: &FlexArray<'_, f64>,
    reject_prob_s: &FlexArray<'_, f64>,
    lock_cash_s: &FlexArray<'_, bool>,
    allow_partial_s: &FlexArray<'_, bool>,
    raise_reject_s: &FlexArray<'_, bool>,
    log_s: &FlexArray<'_, bool>,
    val_price_s: &FlexArray<'_, f64>,
    close_s: &FlexArray<'_, f64>,
    auto_call_seq: bool,
    ffill_val_price: bool,
    do_update_value: bool,
    max_orders: usize,
    max_logs: usize,
    log_data: *mut u8,
    log_offsets: &LogFieldOffsets,
    seed: Option<u64>,
) -> Result<(Vec<OrderRecord>, usize, Option<Vec<i64>>), PortfolioSimError> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::seed_from_u64(rand::random()),
    };

    let mut order_records: Vec<OrderRecord> = Vec::with_capacity(max_orders.min(nrows * ncols));
    let mut last_position = vec![0.0f64; ncols];
    let mut last_debt = vec![0.0f64; ncols];
    let mut last_val_price = vec![f64::NAN; ncols];
    let mut order_price_arr = vec![f64::NAN; ncols];
    let cash_sharing_global = group_lens.iter().any(|&g| g > 1);
    let mut temp_order_value = if cash_sharing_global && auto_call_seq {
        vec![0.0f64; ncols]
    } else {
        Vec::new()
    };
    let mut call_seq_mut = if auto_call_seq {
        Some(call_seq.to_vec())
    } else {
        None
    };

    let mut oidx: i64 = 0;
    let mut lidx: i64 = 0;

    let mut from_col: usize = 0;
    for group in 0..group_lens.len() {
        let group_len = group_lens[group] as usize;
        let to_col = from_col + group_len;
        let mut cash_now = init_cash[group];
        let mut free_cash_now = init_cash[group];

        for i in 0..nrows {
            // Phase 1: resolve prices for each column in this group
            for k in 0..group_len {
                let col = from_col + k;

                // Resolve order price
                let mut _price = price_s.get(i, col);
                if _price.is_infinite() {
                    if _price > 0.0 {
                        _price = close_s.get(i, col);
                    } else if i > 0 {
                        _price = close_s.get(i - 1, col);
                    } else {
                        _price = f64::NAN;
                    }
                }
                order_price_arr[col] = _price;

                // Resolve val price
                let mut _val_price = val_price_s.get(i, col);
                if _val_price.is_infinite() {
                    if _val_price > 0.0 {
                        _val_price = _price;
                    } else if i > 0 {
                        _val_price = close_s.get(i - 1, col);
                    } else {
                        _val_price = f64::NAN;
                    }
                }
                if !_val_price.is_nan() || !ffill_val_price {
                    last_val_price[col] = _val_price;
                }
            }

            // Phase 2: Calculate group value and auto sort if needed
            let mut value_now = 0.0f64;
            if cash_sharing_global {
                value_now = cash_now;
                for k in 0..group_len {
                    let col = from_col + k;
                    if last_position[col] != 0.0 {
                        value_now += last_position[col] * last_val_price[col];
                    }
                }

                if auto_call_seq {
                    for k in 0..group_len {
                        let col = from_col + k;
                        temp_order_value[k] = approx_order_value(
                            size_s.get(i, col),
                            size_type_s.get(i, col),
                            direction_s.get(i, col),
                            cash_now,
                            last_position[col],
                            free_cash_now,
                            last_val_price[col],
                            value_now,
                        );
                    }

                    // Sort call_seq by order value
                    let cs_start = i * ncols + from_col;
                    let call_seq_mut_ref = call_seq_mut
                        .as_mut()
                        .expect("call_seq must be mutable when auto_call_seq is enabled");
                    insert_argsort(
                        &mut temp_order_value[..group_len],
                        &mut call_seq_mut_ref[cs_start..cs_start + group_len],
                    );
                }
            }

            // Phase 3: Process orders
            for k in 0..group_len {
                let mut col = from_col + k;
                if cash_sharing_global {
                    let call_seq_ref = if let Some(ref call_seq_mut_ref) = call_seq_mut {
                        call_seq_mut_ref.as_slice()
                    } else {
                        call_seq
                    };
                    let col_i = call_seq_ref[i * ncols + col] as usize;
                    if col_i >= group_len {
                        return Err(PortfolioSimError::ValueError(
                            "Call index exceeds bounds of the group",
                        ));
                    }
                    col = from_col + col_i;
                }

                let position_now = last_position[col];
                let debt_now = last_debt[col];
                let val_price_now = last_val_price[col];
                if !cash_sharing_global {
                    value_now = cash_now;
                    if position_now != 0.0 {
                        value_now += position_now * val_price_now;
                    }
                }

                let order = Order {
                    size: size_s.get(i, col),
                    price: order_price_arr[col],
                    size_type: size_type_s.get(i, col),
                    direction: direction_s.get(i, col),
                    fees: fees_s.get(i, col),
                    fixed_fees: fixed_fees_s.get(i, col),
                    slippage: slippage_s.get(i, col),
                    min_size: min_size_s.get(i, col),
                    max_size: max_size_s.get(i, col),
                    size_granularity: size_granularity_s.get(i, col),
                    reject_prob: reject_prob_s.get(i, col),
                    lock_cash: lock_cash_s.get(i, col),
                    allow_partial: allow_partial_s.get(i, col),
                    raise_reject: raise_reject_s.get(i, col),
                    log: log_s.get(i, col),
                };

                let state = ProcessOrderState {
                    cash: cash_now,
                    position: position_now,
                    debt: debt_now,
                    free_cash: free_cash_now,
                    val_price: val_price_now,
                    value: value_now,
                    oidx,
                    lidx,
                };

                let log_info = if max_logs > 0 {
                    Some((log_data, log_offsets as &LogFieldOffsets, max_logs))
                } else {
                    None
                };

                let (_order_result, new_state) = process_order(
                    i as i64,
                    col as i64,
                    group as i64,
                    &state,
                    do_update_value,
                    &order,
                    &mut order_records,
                    log_info,
                    &mut rng,
                )?;

                cash_now = new_state.cash;
                free_cash_now = new_state.free_cash;
                oidx = new_state.oidx;
                lidx = new_state.lidx;

                last_position[col] = new_state.position;
                last_debt[col] = new_state.debt;
                if !new_state.val_price.is_nan() || !ffill_val_price {
                    last_val_price[col] = new_state.val_price;
                }

                if cash_sharing_global {
                    value_now = new_state.value;
                }
            }

            // Reset per-column cash when not sharing
            if !cash_sharing_global {
                // In non-sharing mode, each column gets its own cash from init_cash[col]
                // But actually the simulation maintains running cash_now per group,
                // and when not sharing, each column IS its own group.
                // So cash_now carries forward correctly.
            }
        }

        from_col = to_col;
    }

    order_records.truncate(oidx as usize);
    Ok((order_records, lidx as usize, call_seq_mut))
}

// ############# Signal processing helpers ############# //

/// Generate stop signal and change accumulation if needed.
fn generate_stop_signal(
    position_now: f64,
    upon_stop_exit: i64,
    accumulate: i64,
) -> (bool, bool, bool, bool, i64) {
    let mut is_long_entry = false;
    let mut is_long_exit = false;
    let mut is_short_entry = false;
    let mut is_short_exit = false;
    let mut accumulate = accumulate;
    if position_now > 0.0 {
        if upon_stop_exit == STOP_MODE_CLOSE {
            is_long_exit = true;
            accumulate = ACCUMULATION_DISABLED;
        } else if upon_stop_exit == STOP_MODE_CLOSE_REDUCE {
            is_long_exit = true;
        } else if upon_stop_exit == STOP_MODE_REVERSE {
            is_short_entry = true;
            accumulate = ACCUMULATION_DISABLED;
        } else {
            is_short_entry = true;
        }
    } else if position_now < 0.0 {
        if upon_stop_exit == STOP_MODE_CLOSE {
            is_short_exit = true;
            accumulate = ACCUMULATION_DISABLED;
        } else if upon_stop_exit == STOP_MODE_CLOSE_REDUCE {
            is_short_exit = true;
        } else if upon_stop_exit == STOP_MODE_REVERSE {
            is_long_entry = true;
            accumulate = ACCUMULATION_DISABLED;
        } else {
            is_long_entry = true;
        }
    }
    (
        is_long_entry,
        is_long_exit,
        is_short_entry,
        is_short_exit,
        accumulate,
    )
}

/// Resolve price and slippage of a stop order.
fn resolve_stop_price_and_slippage(
    stop_price: f64,
    price: f64,
    close: f64,
    slippage: f64,
    stop_exit_price: i64,
) -> (f64, f64) {
    if stop_exit_price == STOP_EXIT_STOP_MARKET {
        (stop_price, slippage)
    } else if stop_exit_price == STOP_EXIT_STOP_LIMIT {
        (stop_price, 0.0)
    } else if stop_exit_price == STOP_EXIT_CLOSE {
        (close, slippage)
    } else {
        (price, slippage)
    }
}

/// Resolve any conflict between an entry and an exit.
fn resolve_signal_conflict(
    position_now: f64,
    mut is_entry: bool,
    mut is_exit: bool,
    direction: i64,
    conflict_mode: i64,
) -> (bool, bool) {
    if is_entry && is_exit {
        if conflict_mode == CONFLICT_ENTRY {
            is_exit = false;
        } else if conflict_mode == CONFLICT_EXIT {
            is_entry = false;
        } else if conflict_mode == CONFLICT_ADJACENT {
            if position_now == 0.0 {
                is_entry = false;
                is_exit = false;
            } else if direction == DIRECTION_BOTH {
                if position_now > 0.0 {
                    is_exit = false;
                } else if position_now < 0.0 {
                    is_entry = false;
                }
            } else {
                is_exit = false;
            }
        } else if conflict_mode == CONFLICT_OPPOSITE {
            if position_now == 0.0 {
                is_entry = false;
                is_exit = false;
            } else if direction == DIRECTION_BOTH {
                if position_now > 0.0 {
                    is_entry = false;
                } else if position_now < 0.0 {
                    is_exit = false;
                }
            } else {
                is_entry = false;
            }
        } else {
            // Ignore
            is_entry = false;
            is_exit = false;
        }
    }
    (is_entry, is_exit)
}

/// Resolve any direction conflict between a long entry and a short entry.
fn resolve_dir_conflict(
    position_now: f64,
    mut is_long_entry: bool,
    mut is_short_entry: bool,
    upon_dir_conflict: i64,
) -> (bool, bool) {
    if is_long_entry && is_short_entry {
        if upon_dir_conflict == DIR_CONFLICT_LONG {
            is_short_entry = false;
        } else if upon_dir_conflict == DIR_CONFLICT_SHORT {
            is_long_entry = false;
        } else if upon_dir_conflict == DIR_CONFLICT_ADJACENT {
            if position_now > 0.0 {
                is_short_entry = false;
            } else if position_now < 0.0 {
                is_long_entry = false;
            } else {
                is_long_entry = false;
                is_short_entry = false;
            }
        } else if upon_dir_conflict == DIR_CONFLICT_OPPOSITE {
            if position_now > 0.0 {
                is_long_entry = false;
            } else if position_now < 0.0 {
                is_short_entry = false;
            } else {
                is_long_entry = false;
                is_short_entry = false;
            }
        } else {
            // Ignore
            is_long_entry = false;
            is_short_entry = false;
        }
    }
    (is_long_entry, is_short_entry)
}

/// Resolve opposite entry.
fn resolve_opposite_entry(
    position_now: f64,
    mut is_long_entry: bool,
    mut is_long_exit: bool,
    mut is_short_entry: bool,
    mut is_short_exit: bool,
    upon_opposite_entry: i64,
    mut accumulate: i64,
) -> (bool, bool, bool, bool, i64) {
    if position_now > 0.0 && is_short_entry {
        if upon_opposite_entry == OPPOSITE_ENTRY_IGNORE {
            is_short_entry = false;
        } else if upon_opposite_entry == OPPOSITE_ENTRY_CLOSE {
            is_short_entry = false;
            is_long_exit = true;
            accumulate = ACCUMULATION_DISABLED;
        } else if upon_opposite_entry == OPPOSITE_ENTRY_CLOSE_REDUCE {
            is_short_entry = false;
            is_long_exit = true;
        } else if upon_opposite_entry == OPPOSITE_ENTRY_REVERSE {
            accumulate = ACCUMULATION_DISABLED;
        }
    }
    if position_now < 0.0 && is_long_entry {
        if upon_opposite_entry == OPPOSITE_ENTRY_IGNORE {
            is_long_entry = false;
        } else if upon_opposite_entry == OPPOSITE_ENTRY_CLOSE {
            is_long_entry = false;
            is_short_exit = true;
            accumulate = ACCUMULATION_DISABLED;
        } else if upon_opposite_entry == OPPOSITE_ENTRY_CLOSE_REDUCE {
            is_long_entry = false;
            is_short_exit = true;
        } else if upon_opposite_entry == OPPOSITE_ENTRY_REVERSE {
            accumulate = ACCUMULATION_DISABLED;
        }
    }
    (
        is_long_entry,
        is_long_exit,
        is_short_entry,
        is_short_exit,
        accumulate,
    )
}

/// Translate direction-aware signals into size, size type, and direction.
fn signals_to_size(
    position_now: f64,
    is_long_entry: bool,
    is_long_exit: bool,
    is_short_entry: bool,
    is_short_exit: bool,
    size: f64,
    mut size_type: i64,
    accumulate: i64,
    val_price_now: f64,
) -> Result<(f64, i64, i64), PortfolioSimError> {
    if size_type != SIZE_TYPE_AMOUNT
        && size_type != SIZE_TYPE_VALUE
        && size_type != SIZE_TYPE_PERCENT
    {
        return Err(PortfolioSimError::RejectedOrder(
            "Only SizeType.Amount, SizeType.Value, and SizeType.Percent are supported",
        ));
    }
    let mut order_size: f64 = 0.0;
    let mut direction = DIRECTION_BOTH;
    let abs_position_now = position_now.abs();
    if is_less(size, 0.0) {
        return Err(PortfolioSimError::RejectedOrder(
            "Negative size is not allowed. You must express direction using signals.",
        ));
    }

    if position_now > 0.0 {
        if is_short_entry {
            if accumulate == ACCUMULATION_BOTH || accumulate == ACCUMULATION_REMOVE_ONLY {
                order_size = -size;
            } else {
                order_size = -abs_position_now;
                if !size.is_nan() {
                    if size_type == SIZE_TYPE_PERCENT {
                        return Err(PortfolioSimError::RejectedOrder(
                            "SizeType.Percent does not support position reversal using signals",
                        ));
                    }
                    if size_type == SIZE_TYPE_VALUE {
                        order_size -= size / val_price_now;
                    } else {
                        order_size -= size;
                    }
                }
                size_type = SIZE_TYPE_AMOUNT;
            }
        } else if is_long_exit {
            direction = DIRECTION_LONG_ONLY;
            if accumulate == ACCUMULATION_BOTH || accumulate == ACCUMULATION_REMOVE_ONLY {
                order_size = -size;
            } else {
                order_size = -abs_position_now;
                size_type = SIZE_TYPE_AMOUNT;
            }
        } else if is_long_entry {
            direction = DIRECTION_LONG_ONLY;
            if accumulate == ACCUMULATION_BOTH || accumulate == ACCUMULATION_ADD_ONLY {
                order_size = size;
            }
        }
    } else if position_now < 0.0 {
        if is_long_entry {
            if accumulate == ACCUMULATION_BOTH || accumulate == ACCUMULATION_REMOVE_ONLY {
                order_size = size;
            } else {
                order_size = abs_position_now;
                if !size.is_nan() {
                    if size_type == SIZE_TYPE_PERCENT {
                        return Err(PortfolioSimError::RejectedOrder(
                            "SizeType.Percent does not support position reversal using signals",
                        ));
                    }
                    if size_type == SIZE_TYPE_VALUE {
                        order_size += size / val_price_now;
                    } else {
                        order_size += size;
                    }
                }
                size_type = SIZE_TYPE_AMOUNT;
            }
        } else if is_short_exit {
            direction = DIRECTION_SHORT_ONLY;
            if accumulate == ACCUMULATION_BOTH || accumulate == ACCUMULATION_REMOVE_ONLY {
                order_size = size;
            } else {
                order_size = abs_position_now;
                size_type = SIZE_TYPE_AMOUNT;
            }
        } else if is_short_entry {
            direction = DIRECTION_SHORT_ONLY;
            if accumulate == ACCUMULATION_BOTH || accumulate == ACCUMULATION_ADD_ONLY {
                order_size = -size;
            }
        }
    } else {
        if is_long_entry {
            order_size = size;
        } else if is_short_entry {
            order_size = -size;
        }
    }

    Ok((order_size, size_type, direction))
}

/// Get stop price. If hit before open, returns open.
fn get_stop_price_rs(
    position_now: f64,
    stop_price: f64,
    stop: f64,
    open: f64,
    low: f64,
    high: f64,
    hit_below: bool,
) -> Result<f64, PortfolioSimError> {
    if stop < 0.0 {
        return Err(PortfolioSimError::RejectedOrder(
            "Stop value must be 0 or greater",
        ));
    }
    if (position_now > 0.0 && hit_below) || (position_now < 0.0 && !hit_below) {
        let sp = stop_price * (1.0 - stop);
        if open <= sp {
            return Ok(open);
        }
        if low <= sp && sp <= high {
            return Ok(sp);
        }
        return Ok(f64::NAN);
    }
    if (position_now < 0.0 && hit_below) || (position_now > 0.0 && !hit_below) {
        let sp = stop_price * (1.0 + stop);
        if sp <= open {
            return Ok(open);
        }
        if low <= sp && sp <= high {
            return Ok(sp);
        }
        return Ok(f64::NAN);
    }
    Ok(f64::NAN)
}

/// Whether to update stop.
fn should_update_stop(stop: f64, upon_stop_update: i64) -> bool {
    if upon_stop_update == STOP_UPDATE_OVERRIDE || upon_stop_update == STOP_UPDATE_OVERRIDE_NAN {
        if !stop.is_nan() || upon_stop_update == STOP_UPDATE_OVERRIDE_NAN {
            return true;
        }
    }
    false
}

/// Read signals from arrays and merge direction mode with LS mode.

#[pyfunction]
#[pyo3(name = "trade_winning_streak_rs")]
pub fn trade_winning_streak_py<'py>(
    py: Python<'py>,
    records: Bound<'py, pyo3::PyAny>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let toffsets = trade_record_offsets(&records)?;
    let (src_data, _itemsize, n) = unsafe { array_raw_parts(&records)? };
    let src_send = ptr_to_usize(src_data);

    let result = py.allow_threads(|| {
        let src_data = usize_to_ptr(src_send);
        let mut out = vec![0i64; n];
        let mut curr_rank: i64 = 0;
        for i in 0..n {
            let base = unsafe { src_data.add(i * toffsets.itemsize) };
            let pnl = unsafe { *(base.add(toffsets.pnl) as *const f64) };
            if pnl > 0.0 {
                curr_rank += 1;
            } else {
                curr_rank = 0;
            }
            out[i] = curr_rank;
        }
        out
    });

    Ok(PyArray1::from_vec_bound(py, result))
}

/// Process signals for a single (i, col) and return (is_long_entry, is_long_exit, is_short_entry, is_short_exit, accumulate, price, slippage).
/// Also updates stop trailing state in-place.

#[pyfunction]
#[pyo3(name = "trade_losing_streak_rs")]
pub fn trade_losing_streak_py<'py>(
    py: Python<'py>,
    records: Bound<'py, pyo3::PyAny>,
) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let toffsets = trade_record_offsets(&records)?;
    let (src_data, _itemsize, n) = unsafe { array_raw_parts(&records)? };
    let src_send = ptr_to_usize(src_data);

    let result = py.allow_threads(|| {
        let src_data = usize_to_ptr(src_send);
        let mut out = vec![0i64; n];
        let mut curr_rank: i64 = 0;
        for i in 0..n {
            let base = unsafe { src_data.add(i * toffsets.itemsize) };
            let pnl = unsafe { *(base.add(toffsets.pnl) as *const f64) };
            if pnl < 0.0 {
                curr_rank += 1;
            } else {
                curr_rank = 0;
            }
            out[i] = curr_rank;
        }
        out
    });

    Ok(PyArray1::from_vec_bound(py, result))
}

fn asset_flow_inner(
    src_data: *const u8,
    offsets: &RecordFieldOffsets,
    col_idxs: &[i64],
    col_lens: &[i64],
    nrows: usize,
    ncols: usize,
    direction: i64,
) -> Result<Vec<f64>, PortfolioSimError> {
    let n_cols = col_lens.len();
    let col_start_idxs = col_start_idxs_usize(col_lens);
    let mut out = vec![0.0f64; nrows * ncols];

    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let mut last_id: i64 = -1;
        let mut position_now: f64 = 0.0;

        for c in 0..col_len {
            let oidx = col_idxs[col_start_idxs[col] + c] as usize;
            let base = unsafe { src_data.add(oidx * offsets.itemsize) };
            let id = unsafe { *(base.add(offsets.id) as *const i64) };
            let i = unsafe { *(base.add(offsets.idx) as *const i64) } as usize;
            let side = unsafe { *(base.add(offsets.side) as *const i64) };
            let mut size = unsafe { *(base.add(offsets.size) as *const f64) };

            if id < last_id {
                return Err(PortfolioSimError::ValueError(
                    "id must come in ascending order per column",
                ));
            }
            last_id = id;

            if side == ORDER_SIDE_SELL {
                size *= -1.0;
            }
            let new_position_now = add(position_now, size);
            let asset_flow = if direction == DIRECTION_LONG_ONLY {
                get_long_size(position_now, new_position_now)
            } else if direction == DIRECTION_SHORT_ONLY {
                get_short_size(position_now, new_position_now)
            } else {
                size
            };
            out[i * ncols + col] = add(out[i * ncols + col], asset_flow);
            position_now = new_position_now;
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "get_positions_rs")]
pub fn get_positions_py<'py>(
    py: Python<'py>,
    trade_records: Bound<'py, pyo3::PyAny>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<TradeRecord>>> {
    let toffsets = trade_record_offsets(&trade_records)?;
    let (src_data, _itemsize, n_records) = unsafe { array_raw_parts(&trade_records)? };
    let src_send = ptr_to_usize(src_data);
    let ci_cow = array1_as_slice_cow(&col_idxs);
    let cl_cow = array1_as_slice_cow(&col_lens);

    let positions = py
        .allow_threads(|| {
            get_positions_inner(
                usize_to_ptr(src_send),
                &toffsets,
                ci_cow.as_ref(),
                cl_cow.as_ref(),
                n_records,
            )
        })
        .map_err(|err| portfolio_sim_error_to_pyerr(py, err))?;

    Ok(PyArray1::from_vec_bound(py, positions))
}

// ############# Post-simulation: Cash flow ############# //

#[pyfunction]
#[pyo3(name = "get_long_size_rs")]
pub fn get_long_size_py(position_before: f64, position_now: f64) -> f64 {
    get_long_size(position_before, position_now)
}

fn cash_flow_inner(
    src_data: *const u8,
    offsets: &RecordFieldOffsets,
    col_idxs: &[i64],
    col_lens: &[i64],
    nrows: usize,
    ncols: usize,
    free: bool,
) -> Result<Vec<f64>, PortfolioSimError> {
    let n_cols = col_lens.len();
    let col_start_idxs = col_start_idxs_usize(col_lens);
    let mut out = vec![0.0f64; nrows * ncols];

    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let mut last_id: i64 = -1;
        let mut position_now: f64 = 0.0;
        let mut debt_now: f64 = 0.0;

        if !free {
            for c in 0..col_len {
                let oidx = col_idxs[col_start_idxs[col] + c] as usize;
                let base = unsafe { src_data.add(oidx * offsets.itemsize) };
                let id = unsafe { *(base.add(offsets.id) as *const i64) };
                let i = unsafe { *(base.add(offsets.idx) as *const i64) } as usize;
                let side = unsafe { *(base.add(offsets.side) as *const i64) };
                let mut size = unsafe { *(base.add(offsets.size) as *const f64) };
                let price = unsafe { *(base.add(offsets.price) as *const f64) };
                let fees = unsafe { *(base.add(offsets.fees) as *const f64) };

                if id < last_id {
                    return Err(PortfolioSimError::ValueError(
                        "id must come in ascending order per column",
                    ));
                }
                last_id = id;

                if side == ORDER_SIDE_SELL {
                    size *= -1.0;
                }
                let cash_flow = -size * price - fees;
                out[i * ncols + col] = add(out[i * ncols + col], cash_flow);
            }
            continue;
        }

        for c in 0..col_len {
            let oidx = col_idxs[col_start_idxs[col] + c] as usize;
            let base = unsafe { src_data.add(oidx * offsets.itemsize) };
            let id = unsafe { *(base.add(offsets.id) as *const i64) };
            let i = unsafe { *(base.add(offsets.idx) as *const i64) } as usize;
            let side = unsafe { *(base.add(offsets.side) as *const i64) };
            let mut size = unsafe { *(base.add(offsets.size) as *const f64) };
            let price = unsafe { *(base.add(offsets.price) as *const f64) };
            let fees = unsafe { *(base.add(offsets.fees) as *const f64) };

            if id < last_id {
                return Err(PortfolioSimError::ValueError(
                    "id must come in ascending order per column",
                ));
            }
            last_id = id;

            if side == ORDER_SIDE_SELL {
                size *= -1.0;
            }
            let new_position_now = add(position_now, size);
            let (new_debt, cash_flow) =
                get_free_cash_diff(position_now, new_position_now, debt_now, price, fees);
            debt_now = new_debt;
            out[i * ncols + col] = add(out[i * ncols + col], cash_flow);
            position_now = new_position_now;
        }
    }
    Ok(out)
}

#[pyfunction]
#[pyo3(name = "get_short_size_rs")]
pub fn get_short_size_py(position_before: f64, position_now: f64) -> f64 {
    get_short_size(position_before, position_now)
}

fn sum_grouped_inner(
    a: &[f64],
    group_lens: &[i64],
    nrows: usize,
    ncols: usize,
    n_groups: usize,
) -> Vec<f64> {
    let group_starts = col_start_idxs_usize(group_lens);
    let mut out = uninit_f64_vec(nrows * n_groups);
    for group in 0..n_groups {
        let from_col = group_starts[group];
        let to_col = from_col + group_lens[group] as usize;
        for i in 0..nrows {
            let mut sum = 0.0f64;
            for col in from_col..to_col {
                sum += a[i * ncols + col];
            }
            out[i * n_groups + group] = sum;
        }
    }
    out
}
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn process_signals_at(
    i: usize,
    col: usize,
    position_now: f64,
    order_price: f64,
    slippage_val: f64,
    // Signal arrays
    entries_s: &FlexArray<'_, bool>,
    exits_s: &FlexArray<'_, bool>,
    direction_s: &FlexArray<'_, i64>,
    long_entries_s: &FlexArray<'_, bool>,
    long_exits_s: &FlexArray<'_, bool>,
    short_entries_s: &FlexArray<'_, bool>,
    short_exits_s: &FlexArray<'_, bool>,
    // Conflict params
    accumulate_s: &FlexArray<'_, i64>,
    upon_long_conflict_s: &FlexArray<'_, i64>,
    upon_short_conflict_s: &FlexArray<'_, i64>,
    upon_dir_conflict_s: &FlexArray<'_, i64>,
    upon_opposite_entry_s: &FlexArray<'_, i64>,
    // OHLC
    open_s: &FlexArray<'_, f64>,
    high_s: &FlexArray<'_, f64>,
    low_s: &FlexArray<'_, f64>,
    close_s: &FlexArray<'_, f64>,
    // Stop params
    upon_stop_exit_s: &FlexArray<'_, i64>,
    stop_exit_price_s: &FlexArray<'_, i64>,
    // Stop state (mutable)
    use_stops: bool,
    sl_curr_stop: f64,
    sl_curr_price: f64,
    sl_curr_trail: bool,
    sl_curr_i: &mut i64,
    sl_curr_price_mut: &mut f64,
    tp_curr_stop: f64,
    tp_init_price: f64,
) -> Result<(bool, bool, bool, bool, i64, f64, f64), PortfolioSimError> {
    let mut price_out = order_price;
    let mut slippage_out = slippage_val;
    let mut accumulate_val = accumulate_s.get(i, col);

    // Check stops
    let mut stop_price = f64::NAN;
    if use_stops {
        if !sl_curr_stop.is_nan() || !tp_curr_stop.is_nan() {
            let open_val = open_s.get(i, col);
            let high_val = high_s.get(i, col);
            let low_val = low_s.get(i, col);
            let close_val = close_s.get(i, col);
            let o = if open_val.is_nan() {
                close_val
            } else {
                open_val
            };
            let l = if low_val.is_nan() {
                o.min(close_val)
            } else {
                low_val
            };
            let h = if high_val.is_nan() {
                o.max(close_val)
            } else {
                high_val
            };

            if !sl_curr_stop.is_nan() {
                stop_price =
                    get_stop_price_rs(position_now, sl_curr_price, sl_curr_stop, o, l, h, true)?;
            }
            if stop_price.is_nan() && !tp_curr_stop.is_nan() {
                stop_price =
                    get_stop_price_rs(position_now, tp_init_price, tp_curr_stop, o, l, h, false)?;
            }

            // Update trailing stop
            if !sl_curr_stop.is_nan() && sl_curr_trail {
                if position_now > 0.0 {
                    if h > *sl_curr_price_mut {
                        *sl_curr_i = i as i64;
                        *sl_curr_price_mut = h;
                    }
                } else if position_now < 0.0 {
                    if l < *sl_curr_price_mut {
                        *sl_curr_i = i as i64;
                        *sl_curr_price_mut = l;
                    }
                }
            }
        }
    }

    let (mut is_long_entry, mut is_long_exit, mut is_short_entry, mut is_short_exit);

    if use_stops && !stop_price.is_nan() {
        let upon_stop = upon_stop_exit_s.get(i, col);
        let result = generate_stop_signal(position_now, upon_stop, accumulate_val);
        is_long_entry = result.0;
        is_long_exit = result.1;
        is_short_entry = result.2;
        is_short_exit = result.3;
        accumulate_val = result.4;

        let close_val = close_s.get(i, col);
        let sxp = stop_exit_price_s.get(i, col);
        let (p, s) =
            resolve_stop_price_and_slippage(stop_price, price_out, close_val, slippage_out, sxp);
        price_out = p;
        slippage_out = s;
    } else {
        let signals = read_signals(
            entries_s.get(i, col),
            exits_s.get(i, col),
            direction_s.get(i, col),
            long_entries_s.get(i, col),
            long_exits_s.get(i, col),
            short_entries_s.get(i, col),
            short_exits_s.get(i, col),
        );
        is_long_entry = signals.0;
        is_long_exit = signals.1;
        is_short_entry = signals.2;
        is_short_exit = signals.3;

        // Resolve signal conflicts
        if is_long_entry || is_short_entry {
            let r = resolve_signal_conflict(
                position_now,
                is_long_entry,
                is_long_exit,
                DIRECTION_LONG_ONLY,
                upon_long_conflict_s.get(i, col),
            );
            is_long_entry = r.0;
            is_long_exit = r.1;
            let r = resolve_signal_conflict(
                position_now,
                is_short_entry,
                is_short_exit,
                DIRECTION_SHORT_ONLY,
                upon_short_conflict_s.get(i, col),
            );
            is_short_entry = r.0;
            is_short_exit = r.1;

            let r = resolve_dir_conflict(
                position_now,
                is_long_entry,
                is_short_entry,
                upon_dir_conflict_s.get(i, col),
            );
            is_long_entry = r.0;
            is_short_entry = r.1;

            let r = resolve_opposite_entry(
                position_now,
                is_long_entry,
                is_long_exit,
                is_short_entry,
                is_short_exit,
                upon_opposite_entry_s.get(i, col),
                accumulate_val,
            );
            is_long_entry = r.0;
            is_long_exit = r.1;
            is_short_entry = r.2;
            is_short_exit = r.3;
            accumulate_val = r.4;
        }
    }

    Ok((
        is_long_entry,
        is_long_exit,
        is_short_entry,
        is_short_exit,
        accumulate_val,
        price_out,
        slippage_out,
    ))
}

fn simulate_from_signals_non_shared_inner(
    nrows: usize,
    ncols: usize,
    init_cash: &[f64],
    entries_s: &FlexArray<'_, bool>,
    exits_s: &FlexArray<'_, bool>,
    direction_s: &FlexArray<'_, i64>,
    long_entries_s: &FlexArray<'_, bool>,
    long_exits_s: &FlexArray<'_, bool>,
    short_entries_s: &FlexArray<'_, bool>,
    short_exits_s: &FlexArray<'_, bool>,
    size_s: &FlexArray<'_, f64>,
    price_s: &FlexArray<'_, f64>,
    size_type_s: &FlexArray<'_, i64>,
    fees_s: &FlexArray<'_, f64>,
    fixed_fees_s: &FlexArray<'_, f64>,
    slippage_s: &FlexArray<'_, f64>,
    min_size_s: &FlexArray<'_, f64>,
    max_size_s: &FlexArray<'_, f64>,
    size_granularity_s: &FlexArray<'_, f64>,
    reject_prob_s: &FlexArray<'_, f64>,
    lock_cash_s: &FlexArray<'_, bool>,
    allow_partial_s: &FlexArray<'_, bool>,
    raise_reject_s: &FlexArray<'_, bool>,
    log_s: &FlexArray<'_, bool>,
    accumulate_s: &FlexArray<'_, i64>,
    upon_long_conflict_s: &FlexArray<'_, i64>,
    upon_short_conflict_s: &FlexArray<'_, i64>,
    upon_dir_conflict_s: &FlexArray<'_, i64>,
    upon_opposite_entry_s: &FlexArray<'_, i64>,
    val_price_s: &FlexArray<'_, f64>,
    open_s: &FlexArray<'_, f64>,
    high_s: &FlexArray<'_, f64>,
    low_s: &FlexArray<'_, f64>,
    close_s: &FlexArray<'_, f64>,
    sl_stop_s: &FlexArray<'_, f64>,
    sl_trail_s: &FlexArray<'_, bool>,
    tp_stop_s: &FlexArray<'_, f64>,
    stop_entry_price_s: &FlexArray<'_, i64>,
    stop_exit_price_s: &FlexArray<'_, i64>,
    upon_stop_exit_s: &FlexArray<'_, i64>,
    upon_stop_update_s: &FlexArray<'_, i64>,
    use_stops: bool,
    ffill_val_price: bool,
    do_update_value: bool,
    max_orders: usize,
    max_logs: usize,
    log_data: *mut u8,
    log_offsets: &LogFieldOffsets,
    seed: Option<u64>,
) -> Result<(Vec<OrderRecord>, usize), PortfolioSimError> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::seed_from_u64(rand::random()),
    };

    let mut order_records: Vec<OrderRecord> = Vec::with_capacity(max_orders.min(nrows * ncols));
    let mut oidx: i64 = 0;
    let mut lidx: i64 = 0;

    for col in 0..ncols {
        let mut cash_now = init_cash[col];
        let mut free_cash_now = init_cash[col];
        let mut last_position = 0.0f64;
        let mut last_debt = 0.0f64;
        let mut last_val_price = f64::NAN;

        // Stop state per column
        let mut sl_init_i: i64 = -1;
        let mut sl_init_price = f64::NAN;
        let mut sl_curr_i: i64 = -1;
        let mut sl_curr_price = f64::NAN;
        let mut sl_curr_stop = f64::NAN;
        let mut sl_curr_trail = false;
        let mut tp_init_i: i64 = -1;
        let mut tp_init_price = f64::NAN;
        let mut tp_curr_stop = f64::NAN;

        for i in 0..nrows {
            // Resolve order price
            let mut order_price = price_s.get(i, col);
            if order_price.is_infinite() {
                if order_price > 0.0 {
                    order_price = close_s.get(i, col);
                } else {
                    let open_val = open_s.get(i, col);
                    if !open_val.is_nan() {
                        order_price = open_val;
                    } else if i > 0 {
                        order_price = close_s.get(i - 1, col);
                    } else {
                        order_price = f64::NAN;
                    }
                }
            }

            // Resolve valuation price
            let mut val_price_now = val_price_s.get(i, col);
            if val_price_now.is_infinite() {
                if val_price_now > 0.0 {
                    val_price_now = order_price;
                } else if i > 0 {
                    val_price_now = close_s.get(i - 1, col);
                } else {
                    val_price_now = f64::NAN;
                }
            }
            if !val_price_now.is_nan() || !ffill_val_price {
                last_val_price = val_price_now;
            }

            let position_now = last_position;
            let slippage_val = slippage_s.get(i, col);

            // Process signals
            let (
                is_long_entry,
                is_long_exit,
                is_short_entry,
                is_short_exit,
                accumulate_val,
                final_price,
                final_slippage,
            ) = process_signals_at(
                i,
                col,
                position_now,
                order_price,
                slippage_val,
                entries_s,
                exits_s,
                direction_s,
                long_entries_s,
                long_exits_s,
                short_entries_s,
                short_exits_s,
                accumulate_s,
                upon_long_conflict_s,
                upon_short_conflict_s,
                upon_dir_conflict_s,
                upon_opposite_entry_s,
                open_s,
                high_s,
                low_s,
                close_s,
                upon_stop_exit_s,
                stop_exit_price_s,
                use_stops,
                sl_curr_stop,
                sl_curr_price,
                sl_curr_trail,
                &mut sl_curr_i,
                &mut sl_curr_price,
                tp_curr_stop,
                tp_init_price,
            )?;

            // Convert signals to size
            let (mut order_size, final_size_type, final_direction) = signals_to_size(
                last_position,
                is_long_entry,
                is_long_exit,
                is_short_entry,
                is_short_exit,
                size_s.get(i, col),
                size_type_s.get(i, col),
                accumulate_val,
                last_val_price,
            )?;

            if order_size != 0.0 {
                // Apply ShortOnly size sign flip
                if order_size > 0.0 {
                    if final_direction == DIRECTION_SHORT_ONLY {
                        order_size *= -1.0;
                    }
                } else if final_direction == DIRECTION_SHORT_ONLY {
                    order_size *= -1.0;
                }

                let order = Order {
                    size: order_size,
                    price: final_price,
                    size_type: final_size_type,
                    direction: final_direction,
                    fees: fees_s.get(i, col),
                    fixed_fees: fixed_fees_s.get(i, col),
                    slippage: final_slippage,
                    min_size: min_size_s.get(i, col),
                    max_size: max_size_s.get(i, col),
                    size_granularity: size_granularity_s.get(i, col),
                    reject_prob: reject_prob_s.get(i, col),
                    lock_cash: lock_cash_s.get(i, col),
                    allow_partial: allow_partial_s.get(i, col),
                    raise_reject: raise_reject_s.get(i, col),
                    log: log_s.get(i, col),
                };

                let mut value_now = cash_now;
                if last_position != 0.0 {
                    value_now += last_position * last_val_price;
                }

                let state = ProcessOrderState {
                    cash: cash_now,
                    position: last_position,
                    debt: last_debt,
                    free_cash: free_cash_now,
                    val_price: last_val_price,
                    value: value_now,
                    oidx,
                    lidx,
                };

                let log_info = if max_logs > 0 {
                    Some((log_data, log_offsets as &LogFieldOffsets, max_logs))
                } else {
                    None
                };

                let (order_result, new_state) = process_order(
                    i as i64,
                    col as i64,
                    col as i64,
                    &state,
                    do_update_value,
                    &order,
                    &mut order_records,
                    log_info,
                    &mut rng,
                )?;

                cash_now = new_state.cash;
                free_cash_now = new_state.free_cash;
                oidx = new_state.oidx;
                lidx = new_state.lidx;
                last_position = new_state.position;
                last_debt = new_state.debt;
                if !new_state.val_price.is_nan() || !ffill_val_price {
                    last_val_price = new_state.val_price;
                }

                // Update stop state after fill
                if use_stops && order_result.status == ORDER_STATUS_FILLED {
                    if last_position == 0.0 {
                        sl_curr_i = -1;
                        sl_init_i = -1;
                        sl_curr_price = f64::NAN;
                        sl_init_price = f64::NAN;
                        sl_curr_stop = f64::NAN;
                        sl_curr_trail = false;
                        tp_init_i = -1;
                        tp_init_price = f64::NAN;
                        tp_curr_stop = f64::NAN;
                    } else {
                        let sep = stop_entry_price_s.get(i, col);
                        let new_init_price = if sep == STOP_ENTRY_VAL_PRICE {
                            new_state.val_price
                        } else if sep == STOP_ENTRY_PRICE {
                            order.price
                        } else if sep == STOP_ENTRY_FILL_PRICE {
                            order_result.price
                        } else {
                            close_s.get(i, col)
                        };
                        let usu = upon_stop_update_s.get(i, col);
                        let new_sl_stop = sl_stop_s.get(i, col);
                        let new_sl_trail = sl_trail_s.get(i, col);
                        let new_tp_stop = tp_stop_s.get(i, col);

                        if state.position == 0.0
                            || last_position.signum() != state.position.signum()
                        {
                            sl_curr_i = i as i64;
                            sl_init_i = i as i64;
                            sl_curr_price = new_init_price;
                            sl_init_price = new_init_price;
                            sl_curr_stop = new_sl_stop;
                            sl_curr_trail = new_sl_trail;
                            tp_init_i = i as i64;
                            tp_init_price = new_init_price;
                            tp_curr_stop = new_tp_stop;
                        } else if last_position.abs() > state.position.abs() {
                            if should_update_stop(new_sl_stop, usu) {
                                sl_curr_i = i as i64;
                                sl_init_i = i as i64;
                                sl_curr_price = new_init_price;
                                sl_init_price = new_init_price;
                                sl_curr_stop = new_sl_stop;
                                sl_curr_trail = new_sl_trail;
                            }
                            if should_update_stop(new_tp_stop, usu) {
                                tp_init_i = i as i64;
                                tp_init_price = new_init_price;
                                tp_curr_stop = new_tp_stop;
                            }
                        }
                    }
                }
            }
        }
    }

    order_records.truncate(oidx as usize);
    Ok((order_records, lidx as usize))
}

/// Inner simulation function for shared cash / auto call seq.
#[allow(clippy::too_many_arguments)]
fn simulate_from_signals_inner(
    nrows: usize,
    ncols: usize,
    group_lens: &[i64],
    init_cash: &[f64],
    call_seq: &[i64],
    entries_s: &FlexArray<'_, bool>,
    exits_s: &FlexArray<'_, bool>,
    direction_s: &FlexArray<'_, i64>,
    long_entries_s: &FlexArray<'_, bool>,
    long_exits_s: &FlexArray<'_, bool>,
    short_entries_s: &FlexArray<'_, bool>,
    short_exits_s: &FlexArray<'_, bool>,
    size_s: &FlexArray<'_, f64>,
    price_s: &FlexArray<'_, f64>,
    size_type_s: &FlexArray<'_, i64>,
    fees_s: &FlexArray<'_, f64>,
    fixed_fees_s: &FlexArray<'_, f64>,
    slippage_s: &FlexArray<'_, f64>,
    min_size_s: &FlexArray<'_, f64>,
    max_size_s: &FlexArray<'_, f64>,
    size_granularity_s: &FlexArray<'_, f64>,
    reject_prob_s: &FlexArray<'_, f64>,
    lock_cash_s: &FlexArray<'_, bool>,
    allow_partial_s: &FlexArray<'_, bool>,
    raise_reject_s: &FlexArray<'_, bool>,
    log_s: &FlexArray<'_, bool>,
    accumulate_s: &FlexArray<'_, i64>,
    upon_long_conflict_s: &FlexArray<'_, i64>,
    upon_short_conflict_s: &FlexArray<'_, i64>,
    upon_dir_conflict_s: &FlexArray<'_, i64>,
    upon_opposite_entry_s: &FlexArray<'_, i64>,
    val_price_s: &FlexArray<'_, f64>,
    open_s: &FlexArray<'_, f64>,
    high_s: &FlexArray<'_, f64>,
    low_s: &FlexArray<'_, f64>,
    close_s: &FlexArray<'_, f64>,
    sl_stop_s: &FlexArray<'_, f64>,
    sl_trail_s: &FlexArray<'_, bool>,
    tp_stop_s: &FlexArray<'_, f64>,
    stop_entry_price_s: &FlexArray<'_, i64>,
    stop_exit_price_s: &FlexArray<'_, i64>,
    upon_stop_exit_s: &FlexArray<'_, i64>,
    upon_stop_update_s: &FlexArray<'_, i64>,
    use_stops: bool,
    auto_call_seq: bool,
    ffill_val_price: bool,
    do_update_value: bool,
    max_orders: usize,
    max_logs: usize,
    log_data: *mut u8,
    log_offsets: &LogFieldOffsets,
    seed: Option<u64>,
) -> Result<(Vec<OrderRecord>, usize, Option<Vec<i64>>), PortfolioSimError> {
    let mut rng = match seed {
        Some(s) => ChaCha8Rng::seed_from_u64(s),
        None => ChaCha8Rng::seed_from_u64(rand::random()),
    };

    let mut order_records: Vec<OrderRecord> = Vec::with_capacity(max_orders.min(nrows * ncols));
    let mut last_position = vec![0.0f64; ncols];
    let mut last_debt = vec![0.0f64; ncols];
    let mut last_val_price = vec![f64::NAN; ncols];
    let mut price_arr = vec![f64::NAN; ncols];
    let mut size_arr = vec![0.0f64; ncols];
    let mut size_type_arr = vec![0i64; ncols];
    let mut slippage_arr = vec![0.0f64; ncols];
    let mut direction_arr = vec![0i64; ncols];

    // Stop tracking arrays
    let mut sl_init_i_arr = vec![-1i64; ncols];
    let mut sl_init_price_arr = vec![f64::NAN; ncols];
    let mut sl_curr_i_arr = vec![-1i64; ncols];
    let mut sl_curr_price_arr = vec![f64::NAN; ncols];
    let mut sl_curr_stop_arr = vec![f64::NAN; ncols];
    let mut sl_curr_trail_arr = vec![false; ncols];
    let mut tp_init_i_arr = vec![-1i64; ncols];
    let mut tp_init_price_arr = vec![f64::NAN; ncols];
    let mut tp_curr_stop_arr = vec![f64::NAN; ncols];

    let cash_sharing_global = group_lens.iter().any(|&g| g > 1);
    let mut temp_order_value = if cash_sharing_global && auto_call_seq {
        vec![0.0f64; ncols]
    } else {
        Vec::new()
    };
    let mut call_seq_mut = if auto_call_seq {
        Some(call_seq.to_vec())
    } else {
        None
    };

    let mut oidx: i64 = 0;
    let mut lidx: i64 = 0;

    let mut from_col: usize = 0;
    for group in 0..group_lens.len() {
        let group_len = group_lens[group] as usize;
        let to_col = from_col + group_len;
        let mut cash_now = init_cash[group];
        let mut free_cash_now = init_cash[group];

        for i in 0..nrows {
            // Phase 1: Resolve prices
            for k in 0..group_len {
                let col = from_col + k;

                let mut order_price = price_s.get(i, col);
                if order_price.is_infinite() {
                    if order_price > 0.0 {
                        order_price = close_s.get(i, col);
                    } else {
                        let open_val = open_s.get(i, col);
                        if !open_val.is_nan() {
                            order_price = open_val;
                        } else if i > 0 {
                            order_price = close_s.get(i - 1, col);
                        } else {
                            order_price = f64::NAN;
                        }
                    }
                }

                let mut val_price_now = val_price_s.get(i, col);
                if val_price_now.is_infinite() {
                    if val_price_now > 0.0 {
                        val_price_now = order_price;
                    } else if i > 0 {
                        val_price_now = close_s.get(i - 1, col);
                    } else {
                        val_price_now = f64::NAN;
                    }
                }
                if !val_price_now.is_nan() || !ffill_val_price {
                    last_val_price[col] = val_price_now;
                }
                price_arr[col] = order_price;
            }

            // Phase 2: Process signals for each column
            for k in 0..group_len {
                let col = from_col + k;
                let position_now = last_position[col];
                let slippage_val = slippage_s.get(i, col);

                let (
                    is_long_entry,
                    is_long_exit,
                    is_short_entry,
                    is_short_exit,
                    accumulate_val,
                    final_price,
                    final_slippage,
                ) = process_signals_at(
                    i,
                    col,
                    position_now,
                    price_arr[col],
                    slippage_val,
                    entries_s,
                    exits_s,
                    direction_s,
                    long_entries_s,
                    long_exits_s,
                    short_entries_s,
                    short_exits_s,
                    accumulate_s,
                    upon_long_conflict_s,
                    upon_short_conflict_s,
                    upon_dir_conflict_s,
                    upon_opposite_entry_s,
                    open_s,
                    high_s,
                    low_s,
                    close_s,
                    upon_stop_exit_s,
                    stop_exit_price_s,
                    use_stops,
                    sl_curr_stop_arr[col],
                    sl_curr_price_arr[col],
                    sl_curr_trail_arr[col],
                    &mut sl_curr_i_arr[col],
                    &mut sl_curr_price_arr[col],
                    tp_curr_stop_arr[col],
                    tp_init_price_arr[col],
                )?;

                let (order_size, final_size_type, final_direction) = signals_to_size(
                    last_position[col],
                    is_long_entry,
                    is_long_exit,
                    is_short_entry,
                    is_short_exit,
                    size_s.get(i, col),
                    size_type_s.get(i, col),
                    accumulate_val,
                    last_val_price[col],
                )?;

                price_arr[col] = final_price;
                slippage_arr[col] = final_slippage;
                size_arr[col] = order_size;
                size_type_arr[col] = final_size_type;
                direction_arr[col] = final_direction;

                if cash_sharing_global && auto_call_seq {
                    if order_size == 0.0 {
                        temp_order_value[k] = 0.0;
                    } else {
                        if final_size_type == SIZE_TYPE_AMOUNT {
                            temp_order_value[k] = order_size * last_val_price[col];
                        } else if final_size_type == SIZE_TYPE_VALUE {
                            temp_order_value[k] = order_size;
                        } else {
                            // Percent
                            if order_size >= 0.0 {
                                temp_order_value[k] = order_size * cash_now;
                            } else {
                                let asset_value_now = last_position[col] * last_val_price[col];
                                if final_direction == DIRECTION_LONG_ONLY {
                                    temp_order_value[k] = order_size * asset_value_now;
                                } else {
                                    let max_exposure =
                                        2.0 * asset_value_now.max(0.0) + free_cash_now.max(0.0);
                                    temp_order_value[k] = order_size * max_exposure;
                                }
                            }
                        }
                    }
                }
            }

            // Phase 2.5: Sort and compute value
            if cash_sharing_global {
                if auto_call_seq {
                    let cs = call_seq_mut.as_mut().unwrap();
                    insert_argsort(
                        &mut temp_order_value[..group_len],
                        &mut cs[i * ncols + from_col..i * ncols + to_col],
                    );
                }

                // Skipped: value_now computed below per column
            }
            let mut value_now = cash_now;
            if cash_sharing_global {
                for k in 0..group_len {
                    let col = from_col + k;
                    if last_position[col] != 0.0 {
                        value_now += last_position[col] * last_val_price[col];
                    }
                }
            }

            // Phase 3: Execute orders
            for k in 0..group_len {
                let col = if cash_sharing_global {
                    let cs = call_seq_mut.as_ref().map_or(call_seq, |v| v.as_slice());
                    let col_i = cs[i * ncols + from_col + k] as usize;
                    if col_i >= group_len {
                        return Err(PortfolioSimError::RejectedOrder(
                            "Call index exceeds bounds of the group",
                        ));
                    }
                    from_col + col_i
                } else {
                    from_col + k
                };

                let position_now = last_position[col];
                let debt_now = last_debt[col];
                let val_price_now = last_val_price[col];
                if !cash_sharing_global {
                    value_now = cash_now;
                    if position_now != 0.0 {
                        value_now += position_now * val_price_now;
                    }
                }

                let mut order_size = size_arr[col];
                let final_direction = direction_arr[col];

                if order_size != 0.0 {
                    if order_size > 0.0 {
                        if final_direction == DIRECTION_SHORT_ONLY {
                            order_size *= -1.0;
                        }
                    } else if final_direction == DIRECTION_SHORT_ONLY {
                        order_size *= -1.0;
                    }

                    let order = Order {
                        size: order_size,
                        price: price_arr[col],
                        size_type: size_type_arr[col],
                        direction: final_direction,
                        fees: fees_s.get(i, col),
                        fixed_fees: fixed_fees_s.get(i, col),
                        slippage: slippage_arr[col],
                        min_size: min_size_s.get(i, col),
                        max_size: max_size_s.get(i, col),
                        size_granularity: size_granularity_s.get(i, col),
                        reject_prob: reject_prob_s.get(i, col),
                        lock_cash: lock_cash_s.get(i, col),
                        allow_partial: allow_partial_s.get(i, col),
                        raise_reject: raise_reject_s.get(i, col),
                        log: log_s.get(i, col),
                    };

                    let state = ProcessOrderState {
                        cash: cash_now,
                        position: position_now,
                        debt: debt_now,
                        free_cash: free_cash_now,
                        val_price: val_price_now,
                        value: value_now,
                        oidx,
                        lidx,
                    };

                    let log_info = if max_logs > 0 {
                        Some((log_data, log_offsets as &LogFieldOffsets, max_logs))
                    } else {
                        None
                    };

                    let (order_result, new_state) = process_order(
                        i as i64,
                        col as i64,
                        group as i64,
                        &state,
                        do_update_value,
                        &order,
                        &mut order_records,
                        log_info,
                        &mut rng,
                    )?;

                    cash_now = new_state.cash;
                    free_cash_now = new_state.free_cash;
                    oidx = new_state.oidx;
                    lidx = new_state.lidx;
                    let new_position = new_state.position;
                    last_position[col] = new_position;
                    last_debt[col] = new_state.debt;
                    if !new_state.val_price.is_nan() || !ffill_val_price {
                        last_val_price[col] = new_state.val_price;
                    }
                    value_now = new_state.value;

                    // Update stop state
                    if use_stops && order_result.status == ORDER_STATUS_FILLED {
                        if new_position == 0.0 {
                            sl_curr_i_arr[col] = -1;
                            sl_init_i_arr[col] = -1;
                            sl_curr_price_arr[col] = f64::NAN;
                            sl_init_price_arr[col] = f64::NAN;
                            sl_curr_stop_arr[col] = f64::NAN;
                            sl_curr_trail_arr[col] = false;
                            tp_init_i_arr[col] = -1;
                            tp_init_price_arr[col] = f64::NAN;
                            tp_curr_stop_arr[col] = f64::NAN;
                        } else {
                            let sep = stop_entry_price_s.get(i, col);
                            let new_init_price = if sep == STOP_ENTRY_VAL_PRICE {
                                new_state.val_price
                            } else if sep == STOP_ENTRY_PRICE {
                                order.price
                            } else if sep == STOP_ENTRY_FILL_PRICE {
                                order_result.price
                            } else {
                                close_s.get(i, col)
                            };
                            let usu = upon_stop_update_s.get(i, col);
                            let new_sl_stop = sl_stop_s.get(i, col);
                            let new_sl_trail = sl_trail_s.get(i, col);
                            let new_tp_stop = tp_stop_s.get(i, col);

                            if position_now == 0.0 || new_position.signum() != position_now.signum()
                            {
                                sl_curr_i_arr[col] = i as i64;
                                sl_init_i_arr[col] = i as i64;
                                sl_curr_price_arr[col] = new_init_price;
                                sl_init_price_arr[col] = new_init_price;
                                sl_curr_stop_arr[col] = new_sl_stop;
                                sl_curr_trail_arr[col] = new_sl_trail;
                                tp_init_i_arr[col] = i as i64;
                                tp_init_price_arr[col] = new_init_price;
                                tp_curr_stop_arr[col] = new_tp_stop;
                            } else if new_position.abs() > position_now.abs() {
                                if should_update_stop(new_sl_stop, usu) {
                                    sl_curr_i_arr[col] = i as i64;
                                    sl_init_i_arr[col] = i as i64;
                                    sl_curr_price_arr[col] = new_init_price;
                                    sl_init_price_arr[col] = new_init_price;
                                    sl_curr_stop_arr[col] = new_sl_stop;
                                    sl_curr_trail_arr[col] = new_sl_trail;
                                }
                                if should_update_stop(new_tp_stop, usu) {
                                    tp_init_i_arr[col] = i as i64;
                                    tp_init_price_arr[col] = new_init_price;
                                    tp_curr_stop_arr[col] = new_tp_stop;
                                }
                            }
                        }
                    }
                } else {
                    // No order - still update last state
                    last_position[col] = position_now;
                    last_debt[col] = debt_now;
                }
            }
        }

        from_col = to_col;
    }

    order_records.truncate(oidx as usize);
    Ok((order_records, lidx as usize, call_seq_mut))
}

// ############# Post-simulation: Asset flow ############# //

#[pyfunction]
#[pyo3(name = "asset_flow_rs")]
pub fn asset_flow_py<'py>(
    py: Python<'py>,
    order_records: Bound<'py, pyo3::PyAny>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
    target_shape: (usize, usize),
    direction: i64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let offsets = order_record_offsets(&order_records)?;
    let (src_data, _itemsize, _n) = unsafe { array_raw_parts(&order_records)? };
    let src_send = ptr_to_usize(src_data);
    let ci_cow = array1_as_slice_cow(&col_idxs);
    let cl_cow = array1_as_slice_cow(&col_lens);
    let (nrows, ncols) = target_shape;

    let result = py
        .allow_threads(|| {
            asset_flow_inner(
                usize_to_ptr(src_send),
                &offsets,
                ci_cow.as_ref(),
                cl_cow.as_ref(),
                nrows,
                ncols,
                direction,
            )
        })
        .map_err(|err| portfolio_sim_error_to_pyerr(py, err))?;

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "assets_rs")]
pub fn assets_py<'py>(
    py: Python<'py>,
    asset_flow: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let af = asset_flow.as_array();
    let nrows = af.nrows();
    let ncols = af.ncols();
    let af_cow = array2_as_slice_cow(&asset_flow);

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * ncols);
        for col in 0..ncols {
            let mut position_now = 0.0f64;
            for i in 0..nrows {
                position_now = add(position_now, af_cow[i * ncols + col]);
                out[i * ncols + col] = position_now;
            }
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "get_free_cash_diff_rs")]
pub fn get_free_cash_diff_py(
    position_before: f64,
    position_now: f64,
    debt_now: f64,
    price: f64,
    fees: f64,
) -> (f64, f64) {
    get_free_cash_diff(position_before, position_now, debt_now, price, fees)
}

#[pyfunction]
#[pyo3(name = "cash_flow_rs")]
pub fn cash_flow_py<'py>(
    py: Python<'py>,
    order_records: Bound<'py, pyo3::PyAny>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
    target_shape: (usize, usize),
    free: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let offsets = order_record_offsets(&order_records)?;
    let (src_data, _itemsize, _n) = unsafe { array_raw_parts(&order_records)? };
    let src_send = ptr_to_usize(src_data);
    let ci_cow = array1_as_slice_cow(&col_idxs);
    let cl_cow = array1_as_slice_cow(&col_lens);
    let (nrows, ncols) = target_shape;

    let result = py
        .allow_threads(|| {
            cash_flow_inner(
                usize_to_ptr(src_send),
                &offsets,
                ci_cow.as_ref(),
                cl_cow.as_ref(),
                nrows,
                ncols,
                free,
            )
        })
        .map_err(|err| portfolio_sim_error_to_pyerr(py, err))?;

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "sum_grouped_rs")]
pub fn sum_grouped_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = a.shape()[0];
    let ncols = a.shape()[1];
    let a_cow = array2_as_slice_cow(&a);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let n_groups = gl_cow.len();

    let result = py.allow_threads(|| {
        sum_grouped_inner(a_cow.as_ref(), gl_cow.as_ref(), nrows, ncols, n_groups)
    });

    let arr = Array2::from_shape_vec((nrows, n_groups), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "cash_flow_grouped_rs")]
pub fn cash_flow_grouped_py<'py>(
    py: Python<'py>,
    cash_flow: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = cash_flow.shape()[0];
    let ncols = cash_flow.shape()[1];
    let cf_cow = array2_as_slice_cow(&cash_flow);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let n_groups = gl_cow.len();

    let result = py.allow_threads(|| {
        sum_grouped_inner(cf_cow.as_ref(), gl_cow.as_ref(), nrows, ncols, n_groups)
    });

    let arr = Array2::from_shape_vec((nrows, n_groups), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

// ############# Post-simulation: Performance metrics ############# //

#[pyfunction]
#[pyo3(name = "init_cash_grouped_rs")]
pub fn init_cash_grouped_py<'py>(
    py: Python<'py>,
    init_cash: PyReadonlyArray1<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    cash_sharing: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ic_cow = array1_as_slice_cow(&init_cash);
    let gl_cow = array1_as_slice_cow(&group_lens);

    if cash_sharing {
        let out: Vec<f64> = ic_cow.to_vec();
        return Ok(PyArray1::from_vec_bound(py, out));
    }

    let mut out = vec![0.0f64; gl_cow.len()];
    let mut from_col: usize = 0;
    for group in 0..gl_cow.len() {
        let to_col = from_col + gl_cow[group] as usize;
        let mut cash_sum = 0.0f64;
        for col in from_col..to_col {
            cash_sum += ic_cow[col];
        }
        out[group] = cash_sum;
        from_col = to_col;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
#[pyo3(name = "init_cash_rs")]
pub fn init_cash_py<'py>(
    py: Python<'py>,
    init_cash: PyReadonlyArray1<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    cash_sharing: bool,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let ic_cow = array1_as_slice_cow(&init_cash);
    let gl_cow = array1_as_slice_cow(&group_lens);

    if !cash_sharing {
        let out: Vec<f64> = ic_cow.to_vec();
        return Ok(PyArray1::from_vec_bound(py, out));
    }

    // When cash sharing, expand group init cash to per-column
    let total_cols: usize = gl_cow.iter().map(|&g| g as usize).sum();
    let mut out = vec![f64::NAN; total_cols];
    let mut col: usize = 0;
    for group in 0..gl_cow.len() {
        out[col] = ic_cow[group];
        // ffill
        let group_len = gl_cow[group] as usize;
        for k in 1..group_len {
            out[col + k] = ic_cow[group];
        }
        col += group_len;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
#[pyo3(name = "cash_rs")]
pub fn cash_py<'py>(
    py: Python<'py>,
    cash_flow: PyReadonlyArray2<'py, f64>,
    init_cash: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = cash_flow.shape()[0];
    let ncols = cash_flow.shape()[1];
    let cf_cow = array2_as_slice_cow(&cash_flow);
    let ic_cow = array1_as_slice_cow(&init_cash);

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * ncols);
        for col in 0..ncols {
            for i in 0..nrows {
                let cash_now = if i == 0 {
                    ic_cow[col]
                } else {
                    out[(i - 1) * ncols + col]
                };
                out[i * ncols + col] = add(cash_now, cf_cow[i * ncols + col]);
            }
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "cash_in_sim_order_rs")]
pub fn cash_in_sim_order_py<'py>(
    py: Python<'py>,
    cash_flow: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    init_cash_grouped: PyReadonlyArray1<'py, f64>,
    call_seq: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = cash_flow.shape()[0];
    let ncols = cash_flow.shape()[1];
    let cf_cow = array2_as_slice_cow(&cash_flow);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let ic_cow = array1_as_slice_cow(&init_cash_grouped);
    let cs_cow = array2_as_slice_cow(&call_seq);

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * ncols);
        let mut from_col: usize = 0;
        for group in 0..gl_cow.len() {
            let group_len = gl_cow[group] as usize;
            let to_col = from_col + group_len;
            let mut cash_now = ic_cow[group];
            for i in 0..nrows {
                for k in 0..group_len {
                    let col = from_col + cs_cow[i * ncols + from_col + k] as usize;
                    cash_now = add(cash_now, cf_cow[i * ncols + col]);
                    out[i * ncols + col] = cash_now;
                }
            }
            from_col = to_col;
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "cash_grouped_rs")]
pub fn cash_grouped_py<'py>(
    py: Python<'py>,
    target_shape: (usize, usize),
    cash_flow_grouped: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    init_cash_grouped: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = cash_flow_grouped.shape()[0];
    let n_groups = cash_flow_grouped.shape()[1];
    let cfg_cow = array2_as_slice_cow(&cash_flow_grouped);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let ic_cow = array1_as_slice_cow(&init_cash_grouped);

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * n_groups);
        for group in 0..gl_cow.len() {
            let mut cash_now = ic_cow[group];
            for i in 0..nrows {
                cash_now = add(cash_now, cfg_cow[i * n_groups + group]);
                out[i * n_groups + group] = cash_now;
            }
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, n_groups), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "asset_value_rs")]
pub fn asset_value_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    assets: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = close.shape()[0];
    let ncols = close.shape()[1];
    let c_cow = array2_as_slice_cow(&close);
    let a_cow = array2_as_slice_cow(&assets);

    let result = py.allow_threads(|| {
        c_cow
            .iter()
            .zip(a_cow.iter())
            .map(|(close, assets)| close * assets)
            .collect::<Vec<f64>>()
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "asset_value_grouped_rs")]
pub fn asset_value_grouped_py<'py>(
    py: Python<'py>,
    asset_value: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = asset_value.shape()[0];
    let ncols = asset_value.shape()[1];
    let av_cow = array2_as_slice_cow(&asset_value);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let n_groups = gl_cow.len();

    let result = py.allow_threads(|| {
        sum_grouped_inner(av_cow.as_ref(), gl_cow.as_ref(), nrows, ncols, n_groups)
    });

    let arr = Array2::from_shape_vec((nrows, n_groups), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "value_in_sim_order_rs")]
pub fn value_in_sim_order_py<'py>(
    py: Python<'py>,
    cash: PyReadonlyArray2<'py, f64>,
    asset_value: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    call_seq: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = cash.shape()[0];
    let ncols = cash.shape()[1];
    let c_cow = array2_as_slice_cow(&cash);
    let av_cow = array2_as_slice_cow(&asset_value);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let cs_cow = array2_as_slice_cow(&call_seq);

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * ncols);
        let mut from_col: usize = 0;
        for group in 0..gl_cow.len() {
            let group_len = gl_cow[group] as usize;
            let to_col = from_col + group_len;
            let mut asset_value_now = 0.0f64;
            let mut since_last_nan = group_len;
            for j in 0..(nrows * group_len) {
                let i = j / group_len;
                let col = from_col + cs_cow[i * ncols + from_col + j % group_len] as usize;
                if j >= group_len {
                    let last_j = j - group_len;
                    let last_i = last_j / group_len;
                    let last_col =
                        from_col + cs_cow[last_i * ncols + from_col + last_j % group_len] as usize;
                    if !av_cow[last_i * ncols + last_col].is_nan() {
                        asset_value_now -= av_cow[last_i * ncols + last_col];
                    }
                }
                if av_cow[i * ncols + col].is_nan() {
                    since_last_nan = 0;
                } else {
                    asset_value_now += av_cow[i * ncols + col];
                }
                if since_last_nan < group_len {
                    out[i * ncols + col] = f64::NAN;
                } else {
                    out[i * ncols + col] = c_cow[i * ncols + col] + asset_value_now;
                }
                since_last_nan += 1;
            }
            from_col = to_col;
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "value_rs")]
pub fn value_py<'py>(
    py: Python<'py>,
    cash: PyReadonlyArray2<'py, f64>,
    asset_value: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = cash.shape()[0];
    let ncols = cash.shape()[1];
    let c_cow = array2_as_slice_cow(&cash);
    let av_cow = array2_as_slice_cow(&asset_value);

    let result = py.allow_threads(|| {
        c_cow
            .iter()
            .zip(av_cow.iter())
            .map(|(cash, asset_value)| cash + asset_value)
            .collect::<Vec<f64>>()
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "total_profit_rs")]
pub fn total_profit_py<'py>(
    py: Python<'py>,
    target_shape: (usize, usize),
    close: PyReadonlyArray2<'py, f64>,
    order_records: Bound<'py, pyo3::PyAny>,
    col_idxs: PyReadonlyArray1<'py, i64>,
    col_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let offsets = order_record_offsets(&order_records)?;
    let (src_data, _itemsize, _n) = unsafe { array_raw_parts(&order_records)? };
    let src_send = ptr_to_usize(src_data);
    let (nrows, ncols) = target_shape;
    let ci_cow = array1_as_slice_cow(&col_idxs);
    let cl_cow = array1_as_slice_cow(&col_lens);
    let close_cow = array2_as_slice_cow(&close);

    let result = py
        .allow_threads(|| {
            let src_data = usize_to_ptr(src_send);
            let n_cols = cl_cow.len();
            let col_start_idxs = col_start_idxs_usize(cl_cow.as_ref());

            let mut assets = vec![0.0f64; ncols];
            let mut cash = vec![0.0f64; ncols];
            let mut zero_mask = vec![false; ncols];

            for col in 0..n_cols {
                let col_len = cl_cow[col] as usize;
                if col_len == 0 {
                    zero_mask[col] = true;
                    continue;
                }
                let mut last_id: i64 = -1;

                for c in 0..col_len {
                    let oidx = ci_cow[col_start_idxs[col] + c] as usize;
                    let base = unsafe { src_data.add(oidx * offsets.itemsize) };
                    let id = unsafe { *(base.add(offsets.id) as *const i64) };
                    let side = unsafe { *(base.add(offsets.side) as *const i64) };
                    let size = unsafe { *(base.add(offsets.size) as *const f64) };
                    let price = unsafe { *(base.add(offsets.price) as *const f64) };
                    let fees = unsafe { *(base.add(offsets.fees) as *const f64) };

                    if id < last_id {
                        return Err(PortfolioSimError::ValueError(
                            "id must come in ascending order per column",
                        ));
                    }
                    last_id = id;

                    if side == ORDER_SIDE_BUY {
                        assets[col] = add(assets[col], size);
                        let order_cash = size * price + fees;
                        cash[col] = add(cash[col], -order_cash);
                    } else {
                        assets[col] = add(assets[col], -size);
                        let order_cash = size * price - fees;
                        cash[col] = add(cash[col], order_cash);
                    }
                }
            }

            let mut out = vec![0.0f64; ncols];
            for col in 0..ncols {
                if zero_mask[col] {
                    out[col] = 0.0;
                } else {
                    out[col] = cash[col] + assets[col] * close_cow[(nrows - 1) * ncols + col];
                }
            }
            Ok(out)
        })
        .map_err(|err| portfolio_sim_error_to_pyerr(py, err))?;

    Ok(PyArray1::from_vec_bound(py, result))
}

#[pyfunction]
#[pyo3(name = "total_profit_grouped_rs")]
pub fn total_profit_grouped_py<'py>(
    py: Python<'py>,
    total_profit: PyReadonlyArray1<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tp_cow = array1_as_slice_cow(&total_profit);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let group_starts = col_start_idxs_usize(gl_cow.as_ref());

    let mut out = vec![0.0f64; gl_cow.len()];
    for group in 0..gl_cow.len() {
        let from_col = group_starts[group];
        let to_col = from_col + gl_cow[group] as usize;
        let mut sum = 0.0f64;
        for col in from_col..to_col {
            sum += tp_cow[col];
        }
        out[group] = sum;
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
#[pyo3(name = "final_value_rs")]
pub fn final_value_py<'py>(
    py: Python<'py>,
    total_profit: PyReadonlyArray1<'py, f64>,
    init_cash: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tp_cow = array1_as_slice_cow(&total_profit);
    let ic_cow = array1_as_slice_cow(&init_cash);
    let out: Vec<f64> = tp_cow
        .iter()
        .zip(ic_cow.iter())
        .map(|(t, i)| t + i)
        .collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
#[pyo3(name = "total_return_rs")]
pub fn total_return_py<'py>(
    py: Python<'py>,
    total_profit: PyReadonlyArray1<'py, f64>,
    init_cash: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let tp_cow = array1_as_slice_cow(&total_profit);
    let ic_cow = array1_as_slice_cow(&init_cash);
    let out: Vec<f64> = tp_cow
        .iter()
        .zip(ic_cow.iter())
        .map(|(t, i)| t / i)
        .collect();
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
#[pyo3(name = "returns_in_sim_order_rs")]
pub fn returns_in_sim_order_py<'py>(
    py: Python<'py>,
    value_iso: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    init_cash_grouped: PyReadonlyArray1<'py, f64>,
    call_seq: PyReadonlyArray2<'py, i64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = value_iso.shape()[0];
    let ncols = value_iso.shape()[1];
    let v_cow = array2_as_slice_cow(&value_iso);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let ic_cow = array1_as_slice_cow(&init_cash_grouped);
    let cs_cow = array2_as_slice_cow(&call_seq);

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * ncols);
        let mut from_col: usize = 0;
        for group in 0..gl_cow.len() {
            let group_len = gl_cow[group] as usize;
            let mut input_value = ic_cow[group];
            for j in 0..(nrows * group_len) {
                let i = j / group_len;
                let col = from_col + cs_cow[i * ncols + from_col + j % group_len] as usize;
                let output_value = v_cow[i * ncols + col];
                out[i * ncols + col] = get_return(input_value, output_value);
                input_value = output_value;
            }
            from_col += group_len;
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

// ############# Trade record aggregation ############# //

#[pyfunction]
#[pyo3(name = "asset_returns_rs")]
pub fn asset_returns_py<'py>(
    py: Python<'py>,
    cash_flow: PyReadonlyArray2<'py, f64>,
    asset_value: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = cash_flow.shape()[0];
    let ncols = cash_flow.shape()[1];
    let cf_cow = array2_as_slice_cow(&cash_flow);
    let av_cow = array2_as_slice_cow(&asset_value);

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * ncols);
        for col in 0..ncols {
            for i in 0..nrows {
                let input_value = if i == 0 {
                    0.0
                } else {
                    av_cow[(i - 1) * ncols + col]
                };
                let output_value = av_cow[i * ncols + col] + cf_cow[i * ncols + col];
                out[i * ncols + col] = get_return(input_value, output_value);
            }
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

fn get_entry_trades_inner(
    src_data: *const u8,
    offsets: &RecordFieldOffsets,
    col_idxs: &[i64],
    col_lens: &[i64],
    close: &[f64],
    nrows: usize,
    ncols: usize,
    n_records: usize,
) -> Result<Vec<TradeRecord>, PortfolioSimError> {
    let n_cols = col_lens.len();
    let col_start_idxs = col_start_idxs_usize(col_lens);

    let mut records: Vec<TradeRecord> = Vec::with_capacity(n_records);
    let mut parent_id: i64 = -1;

    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let mut last_id: i64 = -1;
        let mut in_position = false;
        let mut direction: i64 = 0;
        let mut entry_size_sum: f64 = 0.0;
        let mut entry_gross_sum: f64 = 0.0;
        let mut entry_fees_sum: f64 = 0.0;
        let mut exit_size_sum: f64 = 0.0;
        let mut exit_gross_sum: f64 = 0.0;
        let mut exit_fees_sum: f64 = 0.0;
        let mut first_c: usize = 0;
        let mut first_entry_size: f64 = 0.0;
        let mut first_entry_fees: f64 = 0.0;

        for c in 0..col_len {
            let oidx = col_idxs[col_start_idxs[col] + c] as usize;
            let base = unsafe { src_data.add(oidx * offsets.itemsize) };
            let id = unsafe { *(base.add(offsets.id) as *const i64) };
            let order_idx = unsafe { *(base.add(offsets.idx) as *const i64) };
            let order_size = unsafe { *(base.add(offsets.size) as *const f64) };
            let order_price = unsafe { *(base.add(offsets.price) as *const f64) };
            let order_fees = unsafe { *(base.add(offsets.fees) as *const f64) };
            let order_side = unsafe { *(base.add(offsets.side) as *const i64) };

            if id < last_id {
                return Err(PortfolioSimError::ValueError(
                    "id must come in ascending order per column",
                ));
            }
            last_id = id;
            if order_size <= 0.0 {
                return Err(PortfolioSimError::ValueError("size must be greater than 0"));
            }
            if order_price <= 0.0 {
                return Err(PortfolioSimError::ValueError(
                    "price must be greater than 0",
                ));
            }

            if !in_position {
                first_c = c;
                in_position = true;
                parent_id += 1;
                direction = if order_side == ORDER_SIDE_BUY {
                    TRADE_DIRECTION_LONG
                } else {
                    TRADE_DIRECTION_SHORT
                };
                entry_size_sum = 0.0;
                entry_gross_sum = 0.0;
                entry_fees_sum = 0.0;
                exit_size_sum = 0.0;
                exit_gross_sum = 0.0;
                exit_fees_sum = 0.0;
                first_entry_size = order_size;
                first_entry_fees = order_fees;
            }

            let is_entry = (direction == TRADE_DIRECTION_LONG && order_side == ORDER_SIDE_BUY)
                || (direction == TRADE_DIRECTION_SHORT && order_side == ORDER_SIDE_SELL);
            let is_exit = (direction == TRADE_DIRECTION_LONG && order_side == ORDER_SIDE_SELL)
                || (direction == TRADE_DIRECTION_SHORT && order_side == ORDER_SIDE_BUY);

            if is_entry {
                entry_size_sum += order_size;
                entry_gross_sum += order_size * order_price;
                entry_fees_sum += order_fees;
            } else if is_exit {
                if is_close(exit_size_sum + order_size, entry_size_sum) {
                    // Position closed exactly
                    in_position = false;
                    exit_size_sum = entry_size_sum;
                    exit_gross_sum += order_size * order_price;
                    exit_fees_sum += order_fees;

                    fill_entry_trades_in_position(
                        src_data,
                        offsets,
                        col_idxs,
                        &col_start_idxs,
                        col,
                        first_c,
                        c,
                        first_entry_size,
                        first_entry_fees,
                        order_idx,
                        exit_size_sum,
                        exit_gross_sum,
                        exit_fees_sum,
                        direction,
                        TRADE_STATUS_CLOSED,
                        parent_id,
                        &mut records,
                    );
                } else if is_less(exit_size_sum + order_size, entry_size_sum) {
                    // Position decreased
                    exit_size_sum += order_size;
                    exit_gross_sum += order_size * order_price;
                    exit_fees_sum += order_fees;
                } else {
                    // Position closed and reversed
                    in_position = false;
                    let remaining_size = add(entry_size_sum, -exit_size_sum);
                    exit_size_sum = entry_size_sum;
                    exit_gross_sum += remaining_size * order_price;
                    exit_fees_sum += remaining_size / order_size * order_fees;

                    fill_entry_trades_in_position(
                        src_data,
                        offsets,
                        col_idxs,
                        &col_start_idxs,
                        col,
                        first_c,
                        c,
                        first_entry_size,
                        first_entry_fees,
                        order_idx,
                        exit_size_sum,
                        exit_gross_sum,
                        exit_fees_sum,
                        direction,
                        TRADE_STATUS_CLOSED,
                        parent_id,
                        &mut records,
                    );

                    // New position opened
                    first_c = c;
                    in_position = true;
                    parent_id += 1;
                    direction = if order_side == ORDER_SIDE_BUY {
                        TRADE_DIRECTION_LONG
                    } else {
                        TRADE_DIRECTION_SHORT
                    };
                    entry_size_sum = add(order_size, -remaining_size);
                    entry_gross_sum = entry_size_sum * order_price;
                    entry_fees_sum = entry_size_sum / order_size * order_fees;
                    first_entry_size = entry_size_sum;
                    first_entry_fees = entry_fees_sum;
                    exit_size_sum = 0.0;
                    exit_gross_sum = 0.0;
                    exit_fees_sum = 0.0;
                }
            }
        }

        if in_position && is_less(exit_size_sum, entry_size_sum) {
            // Position hasn't been closed
            let remaining_size = add(entry_size_sum, -exit_size_sum);
            exit_size_sum = entry_size_sum;
            exit_gross_sum += remaining_size * close[(nrows - 1) * ncols + col];

            fill_entry_trades_in_position(
                src_data,
                offsets,
                col_idxs,
                &col_start_idxs,
                col,
                first_c,
                col_len - 1,
                first_entry_size,
                first_entry_fees,
                (nrows - 1) as i64,
                exit_size_sum,
                exit_gross_sum,
                exit_fees_sum,
                direction,
                TRADE_STATUS_OPEN,
                parent_id,
                &mut records,
            );
        }
    }

    Ok(records)
}

fn fill_entry_trades_in_position(
    src_data: *const u8,
    offsets: &RecordFieldOffsets,
    col_idxs: &[i64],
    col_start_idxs: &[usize],
    col: usize,
    first_c: usize,
    last_c: usize,
    first_entry_size: f64,
    first_entry_fees: f64,
    exit_idx: i64,
    exit_size_sum: f64,
    exit_gross_sum: f64,
    exit_fees_sum: f64,
    direction: i64,
    status: i64,
    parent_id: i64,
    records: &mut Vec<TradeRecord>,
) {
    let exit_price = exit_gross_sum / exit_size_sum;

    for c in first_c..=last_c {
        let oidx = col_idxs[col_start_idxs[col] + c] as usize;
        let base = unsafe { src_data.add(oidx * offsets.itemsize) };
        let order_side = unsafe { *(base.add(offsets.side) as *const i64) };

        // Ignore exit orders
        if (direction == TRADE_DIRECTION_LONG && order_side == ORDER_SIDE_SELL)
            || (direction == TRADE_DIRECTION_SHORT && order_side == ORDER_SIDE_BUY)
        {
            continue;
        }

        let (entry_size, entry_fees) = if c == first_c {
            (first_entry_size, first_entry_fees)
        } else {
            let size = unsafe { *(base.add(offsets.size) as *const f64) };
            let fees = unsafe { *(base.add(offsets.fees) as *const f64) };
            (size, fees)
        };

        let size_fraction = entry_size / exit_size_sum;
        let exit_fees = size_fraction * exit_fees_sum;
        let entry_price = unsafe { *(base.add(offsets.price) as *const f64) };
        let entry_idx = unsafe { *(base.add(offsets.idx) as *const i64) };

        let (pnl, ret) = get_trade_stats(
            entry_size,
            entry_price,
            entry_fees,
            exit_price,
            exit_fees,
            direction,
        );
        let tidx = records.len();
        records.push(TradeRecord {
            id: tidx as i64,
            col: col as i64,
            size: entry_size,
            entry_idx,
            entry_price,
            entry_fees,
            exit_idx,
            exit_price,
            exit_fees,
            pnl,
            return_: ret,
            direction,
            status,
            parent_id,
        });
    }
}

#[pyfunction]
#[pyo3(name = "benchmark_value_rs")]
pub fn benchmark_value_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    init_cash: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = close.shape()[0];
    let ncols = close.shape()[1];
    let c_cow = array2_as_slice_cow(&close);
    let ic_cow = array1_as_slice_cow(&init_cash);

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * ncols);
        if nrows == 0 {
            return out;
        }
        let mut factors = Vec::with_capacity(ncols);
        for col in 0..ncols {
            factors.push(ic_cow[col] / c_cow[col]);
        }
        for i in 0..nrows {
            let row_offset = i * ncols;
            for col in 0..ncols {
                out[row_offset + col] = c_cow[row_offset + col] * factors[col];
            }
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

fn get_exit_trades_inner(
    src_data: *const u8,
    offsets: &RecordFieldOffsets,
    col_idxs: &[i64],
    col_lens: &[i64],
    close: &[f64],
    nrows: usize,
    ncols: usize,
    n_records: usize,
) -> Result<Vec<TradeRecord>, PortfolioSimError> {
    let n_cols = col_lens.len();
    let col_start_idxs = col_start_idxs_usize(col_lens);

    let mut records: Vec<TradeRecord> = Vec::with_capacity(n_records);
    let mut parent_id: i64 = -1;

    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let mut last_id: i64 = -1;
        let mut in_position = false;
        let mut direction: i64 = 0;
        let mut entry_idx: i64 = 0;
        let mut entry_size_sum: f64 = 0.0;
        let mut entry_gross_sum: f64 = 0.0;
        let mut entry_fees_sum: f64 = 0.0;

        for c in 0..col_len {
            let oidx = col_idxs[col_start_idxs[col] + c] as usize;
            let base = unsafe { src_data.add(oidx * offsets.itemsize) };
            let id = unsafe { *(base.add(offsets.id) as *const i64) };
            let i = unsafe { *(base.add(offsets.idx) as *const i64) };
            let order_size = unsafe { *(base.add(offsets.size) as *const f64) };
            let order_price = unsafe { *(base.add(offsets.price) as *const f64) };
            let order_fees = unsafe { *(base.add(offsets.fees) as *const f64) };
            let order_side = unsafe { *(base.add(offsets.side) as *const i64) };

            if id < last_id {
                return Err(PortfolioSimError::ValueError(
                    "id must come in ascending order per column",
                ));
            }
            last_id = id;
            if order_size <= 0.0 {
                return Err(PortfolioSimError::ValueError("size must be greater than 0"));
            }
            if order_price <= 0.0 {
                return Err(PortfolioSimError::ValueError(
                    "price must be greater than 0",
                ));
            }

            if !in_position {
                in_position = true;
                entry_idx = i;
                direction = if order_side == ORDER_SIDE_BUY {
                    TRADE_DIRECTION_LONG
                } else {
                    TRADE_DIRECTION_SHORT
                };
                parent_id += 1;
                entry_size_sum = 0.0;
                entry_gross_sum = 0.0;
                entry_fees_sum = 0.0;
            }

            let is_entry = (direction == TRADE_DIRECTION_LONG && order_side == ORDER_SIDE_BUY)
                || (direction == TRADE_DIRECTION_SHORT && order_side == ORDER_SIDE_SELL);
            let is_exit = (direction == TRADE_DIRECTION_LONG && order_side == ORDER_SIDE_SELL)
                || (direction == TRADE_DIRECTION_SHORT && order_side == ORDER_SIDE_BUY);

            if is_entry {
                entry_size_sum += order_size;
                entry_gross_sum += order_size * order_price;
                entry_fees_sum += order_fees;
            } else if is_exit {
                if is_close_or_less(order_size, entry_size_sum) {
                    let exit_size = if is_close(order_size, entry_size_sum) {
                        entry_size_sum
                    } else {
                        order_size
                    };
                    let exit_price = order_price;
                    let exit_fees = order_fees;
                    let exit_idx = i;

                    let entry_price = entry_gross_sum / entry_size_sum;
                    let size_fraction = exit_size / entry_size_sum;
                    let entry_fees = size_fraction * entry_fees_sum;

                    let (pnl, ret) = get_trade_stats(
                        exit_size,
                        entry_price,
                        entry_fees,
                        exit_price,
                        exit_fees,
                        direction,
                    );

                    let tidx = records.len();
                    records.push(TradeRecord {
                        id: tidx as i64,
                        col: col as i64,
                        size: exit_size,
                        entry_idx,
                        entry_price,
                        entry_fees,
                        exit_idx,
                        exit_price,
                        exit_fees,
                        pnl,
                        return_: ret,
                        direction,
                        status: TRADE_STATUS_CLOSED,
                        parent_id,
                    });

                    if is_close(order_size, entry_size_sum) {
                        in_position = false;
                    } else {
                        let size_fraction = (entry_size_sum - order_size) / entry_size_sum;
                        entry_size_sum *= size_fraction;
                        entry_gross_sum *= size_fraction;
                        entry_fees_sum *= size_fraction;
                    }
                } else {
                    // Trade reversed
                    let cl_exit_size = entry_size_sum;
                    let cl_exit_price = order_price;
                    let cl_exit_fees = cl_exit_size / order_size * order_fees;
                    let cl_exit_idx = i;

                    let entry_price = entry_gross_sum / entry_size_sum;
                    let size_fraction = cl_exit_size / entry_size_sum;
                    let entry_fees = size_fraction * entry_fees_sum;

                    let (pnl, ret) = get_trade_stats(
                        cl_exit_size,
                        entry_price,
                        entry_fees,
                        cl_exit_price,
                        cl_exit_fees,
                        direction,
                    );

                    let tidx = records.len();
                    records.push(TradeRecord {
                        id: tidx as i64,
                        col: col as i64,
                        size: cl_exit_size,
                        entry_idx,
                        entry_price,
                        entry_fees,
                        exit_idx: cl_exit_idx,
                        exit_price: cl_exit_price,
                        exit_fees: cl_exit_fees,
                        pnl,
                        return_: ret,
                        direction,
                        status: TRADE_STATUS_CLOSED,
                        parent_id,
                    });

                    // Open new trade
                    entry_size_sum = order_size - cl_exit_size;
                    entry_gross_sum = entry_size_sum * order_price;
                    entry_fees_sum = order_fees - cl_exit_fees;
                    entry_idx = i;
                    direction = if direction == TRADE_DIRECTION_LONG {
                        TRADE_DIRECTION_SHORT
                    } else {
                        TRADE_DIRECTION_LONG
                    };
                    parent_id += 1;
                }
            }
        }

        if in_position && is_less(-entry_size_sum, 0.0) {
            let exit_size = entry_size_sum;
            let exit_price = close[(nrows - 1) * ncols + col];
            let exit_fees = 0.0;
            let exit_idx = (nrows - 1) as i64;

            let entry_price = entry_gross_sum / entry_size_sum;
            let size_fraction = exit_size / entry_size_sum;
            let entry_fees = size_fraction * entry_fees_sum;

            let (pnl, ret) = get_trade_stats(
                exit_size,
                entry_price,
                entry_fees,
                exit_price,
                exit_fees,
                direction,
            );

            let tidx = records.len();
            records.push(TradeRecord {
                id: tidx as i64,
                col: col as i64,
                size: exit_size,
                entry_idx,
                entry_price,
                entry_fees,
                exit_idx,
                exit_price,
                exit_fees,
                pnl,
                return_: ret,
                direction,
                status: TRADE_STATUS_OPEN,
                parent_id,
            });
        }
    }

    Ok(records)
}

#[pyfunction]
#[pyo3(name = "benchmark_value_grouped_rs")]
pub fn benchmark_value_grouped_py<'py>(
    py: Python<'py>,
    close: PyReadonlyArray2<'py, f64>,
    group_lens: PyReadonlyArray1<'py, i64>,
    init_cash_grouped: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = close.shape()[0];
    let ncols = close.shape()[1];
    let c_cow = array2_as_slice_cow(&close);
    let gl_cow = array1_as_slice_cow(&group_lens);
    let ic_cow = array1_as_slice_cow(&init_cash_grouped);
    let n_groups = gl_cow.len();
    let group_starts = col_start_idxs_usize(gl_cow.as_ref());

    let result = py.allow_threads(|| {
        let mut out = uninit_f64_vec(nrows * n_groups);
        for group in 0..n_groups {
            let from_col = group_starts[group];
            let group_len = gl_cow[group] as usize;
            let to_col = from_col + group_len;
            let col_init_cash = ic_cow[group] / group_len as f64;
            for i in 0..nrows {
                let mut sum = 0.0f64;
                for col in from_col..to_col {
                    sum += c_cow[i * ncols + col] / c_cow[col]; // close[i,col] / close[0,col]
                }
                out[i * n_groups + group] = col_init_cash * sum;
            }
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, n_groups), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

#[pyfunction]
#[pyo3(name = "total_benchmark_return_rs")]
pub fn total_benchmark_return_py<'py>(
    py: Python<'py>,
    benchmark_value: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let nrows = benchmark_value.shape()[0];
    let ncols = benchmark_value.shape()[1];
    let bv_cow = array2_as_slice_cow(&benchmark_value);

    let mut out = vec![0.0f64; ncols];
    for col in 0..ncols {
        out[col] = get_return(bv_cow[col], bv_cow[(nrows - 1) * ncols + col]);
    }
    Ok(PyArray1::from_vec_bound(py, out))
}

#[pyfunction]
#[pyo3(name = "gross_exposure_rs")]
pub fn gross_exposure_py<'py>(
    py: Python<'py>,
    asset_value: PyReadonlyArray2<'py, f64>,
    cash: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = asset_value.shape()[0];
    let ncols = asset_value.shape()[1];
    let av_cow = array2_as_slice_cow(&asset_value);
    let c_cow = array2_as_slice_cow(&cash);

    let result = py.allow_threads(|| {
        let mut out = Vec::with_capacity(nrows * ncols);
        for idx in 0..nrows * ncols {
            let denom = add(av_cow[idx], c_cow[idx]);
            out.push(if denom == 0.0 {
                0.0
            } else {
                av_cow[idx] / denom
            });
        }
        out
    });

    let arr = Array2::from_shape_vec((nrows, ncols), result).unwrap();
    Ok(PyArray2::from_owned_array_bound(py, arr))
}

fn get_positions_inner(
    src_data: *const u8,
    offsets: &TradeFieldOffsets,
    col_idxs: &[i64],
    col_lens: &[i64],
    n_records: usize,
) -> Result<Vec<TradeRecord>, PortfolioSimError> {
    let n_cols = col_lens.len();
    let mut col_start_idxs = vec![0i64; n_cols];
    let mut cumsum: i64 = 0;
    for c in 0..n_cols {
        col_start_idxs[c] = cumsum;
        cumsum += col_lens[c];
    }

    let mut records: Vec<TradeRecord> = Vec::with_capacity(n_records);

    for col in 0..n_cols {
        let col_len = col_lens[col] as usize;
        if col_len == 0 {
            continue;
        }
        let mut last_id: i64 = -1;
        let mut last_position_id: i64 = -1;
        let mut from_tidx: usize = 0;

        for c in 0..col_len {
            let tidx = col_idxs[(col_start_idxs[col] + c as i64) as usize] as usize;
            let base = unsafe { src_data.add(tidx * offsets.itemsize) };
            let id = unsafe { *(base.add(offsets.id) as *const i64) };
            let parent_id = unsafe { *(base.add(offsets.parent_id) as *const i64) };

            if id < last_id {
                return Err(PortfolioSimError::ValueError(
                    "id must come in ascending order per column",
                ));
            }
            last_id = id;

            if parent_id != last_position_id {
                if last_position_id != -1 {
                    // Aggregate trades from from_tidx to tidx
                    let pidx = records.len();
                    let pos = aggregate_position(
                        src_data,
                        offsets,
                        col_idxs,
                        col_start_idxs[col] as usize,
                        from_tidx,
                        c,
                        col,
                        pidx,
                        n_records,
                    );
                    records.push(pos);
                }
                from_tidx = c;
                last_position_id = parent_id;
            }
        }

        // Handle last position
        let pidx = records.len();
        let last_tidx_c = col_len;
        let pos = aggregate_position(
            src_data,
            offsets,
            col_idxs,
            col_start_idxs[col] as usize,
            from_tidx,
            last_tidx_c,
            col,
            pidx,
            n_records,
        );
        records.push(pos);
    }

    Ok(records)
}

fn aggregate_position(
    src_data: *const u8,
    offsets: &TradeFieldOffsets,
    col_idxs: &[i64],
    col_start: usize,
    from_c: usize,
    to_c: usize,
    col: usize,
    pidx: usize,
    n_records: usize,
) -> TradeRecord {
    let count = to_c - from_c;
    if count == 1 {
        // Speed up: copy single trade record
        let tidx = col_idxs[col_start + from_c] as usize;
        let base = unsafe { src_data.add(tidx * offsets.itemsize) };
        let mut rec = read_trade_record(base, offsets);
        rec.id = pidx as i64;
        rec.parent_id = pidx as i64;
        return rec;
    }

    // Aggregate multiple trades
    let mut total_size = 0.0f64;
    let mut entry_idx: i64 = 0;
    let mut exit_idx: i64 = 0;
    let mut total_entry_fees = 0.0f64;
    let mut total_exit_fees = 0.0f64;
    let mut direction: i64 = 0;
    let mut status: i64 = 0;

    // Weighted sums for prices
    let mut weighted_entry_price = 0.0f64;
    let mut weighted_exit_price = 0.0f64;

    for c in from_c..to_c {
        let tidx = col_idxs[col_start + c] as usize;
        let base = unsafe { src_data.add(tidx * offsets.itemsize) };
        let size = unsafe { *(base.add(offsets.size) as *const f64) };
        let ep = unsafe { *(base.add(offsets.entry_price) as *const f64) };
        let ef = unsafe { *(base.add(offsets.entry_fees) as *const f64) };
        let xp = unsafe { *(base.add(offsets.exit_price) as *const f64) };
        let xf = unsafe { *(base.add(offsets.exit_fees) as *const f64) };

        total_size += size;
        weighted_entry_price += size * ep;
        weighted_exit_price += size * xp;
        total_entry_fees += ef;
        total_exit_fees += xf;

        if c == from_c {
            entry_idx = unsafe { *(base.add(offsets.entry_idx) as *const i64) };
        }
        if c == to_c - 1 {
            exit_idx = unsafe { *(base.add(offsets.exit_idx) as *const i64) };
            direction = unsafe { *(base.add(offsets.direction) as *const i64) };
            status = unsafe { *(base.add(offsets.status) as *const i64) };
        }
    }

    let entry_price = weighted_entry_price / total_size;
    let exit_price = weighted_exit_price / total_size;
    let (pnl, ret) = get_trade_stats(
        total_size,
        entry_price,
        total_entry_fees,
        exit_price,
        total_exit_fees,
        direction,
    );

    TradeRecord {
        id: pidx as i64,
        col: col as i64,
        size: total_size,
        entry_idx,
        entry_price,
        entry_fees: total_entry_fees,
        exit_idx,
        exit_price,
        exit_fees: total_exit_fees,
        pnl,
        return_: ret,
        direction,
        status,
        parent_id: pidx as i64,
    }
}

fn read_trade_record(base: *const u8, offsets: &TradeFieldOffsets) -> TradeRecord {
    unsafe {
        TradeRecord {
            id: *(base.add(offsets.id) as *const i64),
            col: *(base.add(offsets.col) as *const i64),
            size: *(base.add(offsets.size) as *const f64),
            entry_idx: *(base.add(offsets.entry_idx) as *const i64),
            entry_price: *(base.add(offsets.entry_price) as *const f64),
            entry_fees: *(base.add(offsets.entry_fees) as *const f64),
            exit_idx: *(base.add(offsets.exit_idx) as *const i64),
            exit_price: *(base.add(offsets.exit_price) as *const f64),
            exit_fees: *(base.add(offsets.exit_fees) as *const f64),
            pnl: *(base.add(offsets.pnl) as *const f64),
            return_: *(base.add(offsets.return_) as *const f64),
            direction: *(base.add(offsets.direction) as *const i64),
            status: *(base.add(offsets.status) as *const i64),
            parent_id: *(base.add(offsets.parent_id) as *const i64),
        }
    }
}

// ############# Module registration ############# //

pub fn register(m: &Bound<'_, pyo3::types::PyModule>) -> PyResult<()> {
    // Classes
    m.add_class::<Order>()?;
    m.add_class::<OrderResult>()?;
    m.add_class::<ExecuteOrderState>()?;
    m.add_class::<ProcessOrderState>()?;

    // Core order functions
    m.add_function(wrap_pyfunction!(order_not_filled_py, m)?)?;
    m.add_function(wrap_pyfunction!(buy_py, m)?)?;
    m.add_function(wrap_pyfunction!(sell_py, m)?)?;
    m.add_function(wrap_pyfunction!(execute_order_py, m)?)?;
    m.add_function(wrap_pyfunction!(raise_rejected_order_py, m)?)?;
    m.add_function(wrap_pyfunction!(update_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(order_nb_py, m)?)?;
    m.add_function(wrap_pyfunction!(close_position_py, m)?)?;
    m.add_function(wrap_pyfunction!(order_nothing_py, m)?)?;

    // Validation & call sequence
    m.add_function(wrap_pyfunction!(check_group_lens_py, m)?)?;
    m.add_function(wrap_pyfunction!(check_group_init_cash_py, m)?)?;
    m.add_function(wrap_pyfunction!(is_grouped_py, m)?)?;
    m.add_function(wrap_pyfunction!(shuffle_call_seq_py, m)?)?;
    m.add_function(wrap_pyfunction!(build_call_seq_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_group_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(approx_order_value_py, m)?)?;

    // Scalar helpers
    m.add_function(wrap_pyfunction!(simulate_from_orders_py, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_from_signals_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_trade_stats_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_entry_trades_py, m)?)?;

    // Simulation
    m.add_function(wrap_pyfunction!(get_exit_trades_py, m)?)?;
    m.add_function(wrap_pyfunction!(trade_winning_streak_py, m)?)?;

    // Asset flow
    m.add_function(wrap_pyfunction!(trade_losing_streak_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_positions_py, m)?)?;

    // Cash flow
    m.add_function(wrap_pyfunction!(get_long_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_short_size_py, m)?)?;
    m.add_function(wrap_pyfunction!(asset_flow_py, m)?)?;
    m.add_function(wrap_pyfunction!(assets_py, m)?)?;
    m.add_function(wrap_pyfunction!(get_free_cash_diff_py, m)?)?;
    m.add_function(wrap_pyfunction!(cash_flow_py, m)?)?;
    m.add_function(wrap_pyfunction!(sum_grouped_py, m)?)?;
    m.add_function(wrap_pyfunction!(cash_flow_grouped_py, m)?)?;

    // Performance metrics
    m.add_function(wrap_pyfunction!(init_cash_grouped_py, m)?)?;
    m.add_function(wrap_pyfunction!(init_cash_py, m)?)?;
    m.add_function(wrap_pyfunction!(cash_py, m)?)?;
    m.add_function(wrap_pyfunction!(cash_in_sim_order_py, m)?)?;
    m.add_function(wrap_pyfunction!(cash_grouped_py, m)?)?;
    m.add_function(wrap_pyfunction!(asset_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(asset_value_grouped_py, m)?)?;
    m.add_function(wrap_pyfunction!(value_in_sim_order_py, m)?)?;
    m.add_function(wrap_pyfunction!(value_py, m)?)?;
    m.add_function(wrap_pyfunction!(total_profit_py, m)?)?;
    m.add_function(wrap_pyfunction!(total_profit_grouped_py, m)?)?;
    m.add_function(wrap_pyfunction!(final_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(total_return_py, m)?)?;
    m.add_function(wrap_pyfunction!(returns_in_sim_order_py, m)?)?;

    // Trade/position records
    m.add_function(wrap_pyfunction!(asset_returns_py, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_value_py, m)?)?;
    m.add_function(wrap_pyfunction!(benchmark_value_grouped_py, m)?)?;
    m.add_function(wrap_pyfunction!(total_benchmark_return_py, m)?)?;
    m.add_function(wrap_pyfunction!(gross_exposure_py, m)?)?;

    Ok(())
}
