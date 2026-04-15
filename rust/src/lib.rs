// Copyright (c) 2021 Oleg Polakow. All rights reserved.
// This code is licensed under Apache 2.0 with Commons Clause license (see LICENSE.md for details)

use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;

mod generic;
mod indicators;

#[pymodule]
fn vectorbt_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    let generic = PyModule::new_bound(m.py(), "generic")?;
    generic::register(&generic)?;
    m.add_submodule(&generic)?;

    let indicators = PyModule::new_bound(m.py(), "indicators")?;
    indicators::register(&indicators)?;
    m.add_submodule(&indicators)?;

    // Register submodules in sys.modules so `from vectorbt_rust.<mod> import ...` works
    let sys = m.py().import_bound("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("vectorbt_rust.generic", generic)?;
    modules.set_item("vectorbt_rust.indicators", indicators)?;

    Ok(())
}
