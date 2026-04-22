# Rust engine

This directory contains the optional Rust engine for vectorbt. It builds the
`vectorbt-rust` Python extension package, exposed to Python as `vectorbt_rust`.

Rust is an acceleration engine. The canonical semantics still live in the
Numba implementations under `vectorbt/<subpackage>/nb.py`; Rust kernels are
called through engine-neutral dispatch wrappers under
`vectorbt/<subpackage>/dispatch.py`.

> [!NOTE]
> Most users should not import `vectorbt_rust` directly. Prefer vectorbt's public
> APIs and pass `engine="rust"` or set the global engine.

## Installation

Install the Rust extension together with vectorbt:

```bash
pip install "vectorbt[rust]"
```

Or install the Rust extension package directly:

```bash
pip install vectorbt-rust
```

`vectorbt-rust` must be version-compatible with `vectorbt`. The engine resolver
compares the major/minor version prefix and treats mismatches as unavailable.

### Building from source

From the repository root:

```bash
python -m pip install -U pip maturin
python -m pip install -e ".[test]"
python -m maturin develop --manifest-path rust/Cargo.toml --release
```

You can also build from inside this directory:

```bash
cd rust
python -m maturin develop --release
```

The release profile enables LTO, one codegen unit, `opt-level = 3`, and symbol
stripping. Use release builds for benchmarks; debug builds are not representative.

## Usage

The shared resolver lives in `vectorbt/_engine.py`.

Per call:

```python
import vectorbt as vbt

out = vbt.MA.run(close, window=20, engine="rust")
```

Globally:

```python
import vectorbt as vbt

vbt.settings["engine"] = "rust"
```

Supported engine values:

- `auto`: Use Rust when it is installed, version-compatible, and the specific
  call is supported; otherwise fall back to Numba.
- `numba`: Force the Numba implementation.
- `rust`: Force Rust and raise an actionable error if Rust is unavailable or the
  call is unsupported.

Randomized functions are special: `auto` keeps using Numba to preserve legacy
NumPy/Numba random streams. Use `engine="rust"` explicitly when you want the
Rust random implementation.

## Compatibility

The current Rust engine targets NumPy arrays and deterministic, array-oriented
kernels. Some public Python APIs still intentionally resolve to Numba when Rust
cannot preserve behavior, such as callback-accepting functions or unsupported
input combinations.

## Testing

Run the engine-focused tests from the repository root:

```bash
pytest tests/test_engine.py
```

Run the full test suite:

```bash
pytest
```

Tests that require Rust are skipped when `vectorbt-rust` is not installed or not
version-compatible. To force those paths locally, install the extension with
`maturin develop` first.

## Benchmarks

The benchmark scripts live in [benchmarks](../benchmarks).

Generate markdown benchmark matrices:

```bash
python benchmarks/bench_matrix.py
```

Use release builds for any benchmark numbers you intend to publish.

## New kernels

New Rust kernels should follow this process:

1. Treat Numba as the reference.
2. Implement the Rust kernel with the same argument order and return shape.
3. Register the PyO3 function in the Rust submodule and wire new submodules in `src/lib.rs`.
4. Add or update the dispatch wrapper.
5. Add parity, fallback, explicit-error, and layout-sensitive tests.
6. Add benchmark cases once parity is stable.

Keep changes narrow and mechanical. Do not import Rust from `nb.py`, and do not
make public callers import `vectorbt_rust` directly.
