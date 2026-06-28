---
title: Contributing
---

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## Development setup

Install VectorBT from the repository in editable mode:

```bash
pip uninstall vectorbt
git clone https://github.com/polakowo/vectorbt.git
cd vectorbt
pip install -e .
```

## Running tests

Run the full Python test suite:

```bash
pytest
```

Run the Rust test suite:

```bash
cd rust
cargo test
```

Make sure to update tests as appropriate when submitting changes.

## Code style

Follow the conventions already established in the codebase. The project makes heavy use of NumPy, Numba, and pandas patterns. When in doubt, look at similar code nearby and match its style.

## Pull requests

1. Open an issue first for major changes to discuss the approach.
2. Keep PRs focused on a single concern.
3. Update or add tests for any changed behavior.
4. Make sure `pytest` passes before submitting.

## Rust contributions

The optional Rust engine lives in `rust/` with its own `Cargo.toml` and test suite. If you modify Rust code, run `cargo test` from within the `rust/` directory to verify your changes.

## Building docs

To preview the documentation locally:

```bash
cd docs
mkdocs serve
```

Then open `http://127.0.0.1:8000` in your browser.
