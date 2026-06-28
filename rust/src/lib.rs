// Copyright (c) 2017-2026 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;

mod generic;
mod indicators;
mod labels;
mod portfolio;
mod records;
mod returns;
mod signals;

#[pymodule]
fn vectorbt_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let generic = PyModule::new_bound(m.py(), "generic")?;
    generic::register(&generic)?;
    m.add_submodule(&generic)?;

    let indicators = PyModule::new_bound(m.py(), "indicators")?;
    indicators::register(&indicators)?;
    m.add_submodule(&indicators)?;

    let labels = PyModule::new_bound(m.py(), "labels")?;
    labels::register(&labels)?;
    m.add_submodule(&labels)?;

    let returns = PyModule::new_bound(m.py(), "returns")?;
    returns::register(&returns)?;
    m.add_submodule(&returns)?;

    let records = PyModule::new_bound(m.py(), "records")?;
    records::register(&records)?;
    m.add_submodule(&records)?;

    let portfolio = PyModule::new_bound(m.py(), "portfolio")?;
    portfolio::register(&portfolio)?;
    m.add_submodule(&portfolio)?;

    let signals = PyModule::new_bound(m.py(), "signals")?;
    signals::register(&signals)?;
    m.add_submodule(&signals)?;

    // Register submodules in sys.modules so `from vectorbt_rust.<mod> import ...` works
    let sys = m.py().import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("vectorbt_rust.generic", generic)?;
    modules.set_item("vectorbt_rust.indicators", indicators)?;
    modules.set_item("vectorbt_rust.labels", labels)?;
    modules.set_item("vectorbt_rust.portfolio", portfolio)?;
    modules.set_item("vectorbt_rust.records", records)?;
    modules.set_item("vectorbt_rust.returns", returns)?;
    modules.set_item("vectorbt_rust.signals", signals)?;

    Ok(())
}
