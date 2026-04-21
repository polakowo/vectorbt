# Benchmarks

This directory contains benchmark tooling for comparing Numba kernels with the
optional Rust engine.

The benchmarks are correctness-aware microbenchmarks. They exercise paired Numba
and Rust functions with deterministic inputs, optionally verify output parity,
and emit either CSV or generated markdown reports.

> [!NOTE]
> Generated reports are benchmark output, not source-of-truth behavior. Recreate
> them after meaningful Rust, dispatch, or benchmark-case changes.

## Requirements

Install vectorbt with test dependencies and the Rust extension:

```bash
python -m pip install -e ".[test]"
python -m maturin develop --manifest-path rust/Cargo.toml --release
```

The benchmark runner exits if `vectorbt-rust` is not installed or not
version-compatible.

Use a release Rust build for benchmark results. Debug extension builds include
extra overhead and are not representative.

## Single run

From the repository root:

```bash
python benchmarks/bench_engine.py --rows 5000 --cols 50 --check
```

The default output is CSV:

```text
function,numba_s,rust_s,speedup
generic.fillna,8.88e-06,2.75e-06,3.23
```

Useful options:

- `--rows` and `--cols` control the generated 2D input shape.
- `--window` controls rolling and indicator windows. Default: `20`.
- `--nan-ratio` controls the fraction of generated NaNs. Default: `0.05`.
- `--repeat` controls measured repetitions. Default: `5`.
- `--warmup` controls untimed warmup calls. Default: `2`.
- `--seed` controls deterministic benchmark input. Default: `42`.
- `--pattern` runs only cases whose name contains the given substring.
- `--check` verifies Rust and Numba output parity before timing.

Example targeted run:

```bash
python benchmarks/bench_engine.py \
  --rows 10000 \
  --cols 10 \
  --pattern signals.generate_ohlc_stop \
  --check
```

## Layout modes

Three layout modes for 1D column inputs are available:

- `view`: pass strided column views. This is the default and closest to common
  vectorbt usage.
- `contiguous`: pass contiguous 1D arrays. This is a best-case kernel baseline.
- `copy-included`: copy non-contiguous 1D arrays inside each timed call, so copy
  overhead is included in the measurement.

Example:

```bash
python benchmarks/bench_engine.py --layout contiguous --check
```

When `copy-included` is requested for a case without non-contiguous 1D inputs,
the runner treats it like `view` to avoid adding meaningless copy overhead.

## Suites

Benchmark cases are tagged and can be filtered by suite:

- `core`: default. Excludes scalar, O(1), fixed-input, cache-lookup, metadata,
  and explicitly extended-only cases.
- `extended`: includes all available cases.

Example:

```bash
python benchmarks/bench_engine.py --suite extended --check
```

Use `core` for headline matrices and `extended` for deeper investigation.

## Markdown reports

Generate all three report files:

```bash
python benchmarks/bench_matrix.py
```

By default this writes:

- `benchmarks/BENCHMARKS.md`
- `benchmarks/BENCHMARKS_NUMBA.md`
- `benchmarks/BENCHMARKS_RUST.md`

Choose a different speedup output path:

```bash
python benchmarks/bench_matrix.py --output benchmarks/BENCHMARKS_LOCAL.md
```

Companion files are derived from the output stem:

- `BENCHMARKS_LOCAL_NUMBA.md`
- `BENCHMARKS_LOCAL_RUST.md`

## Interpretation

`speedup = numba_s / rust_s`.

- Values above `1.00x` mean Rust was faster for that case.
- Values below `1.00x` mean Numba was faster for that case.
- Absolute runtime reports are often more useful for tiny kernels, where a large
  speedup can still be only nanoseconds or microseconds.
- Best-of-repeat timing is used after warmup, so results emphasize steady-state
  kernel cost rather than cold-start effects.

Benchmark numbers are sensitive to CPU, OS scheduling, Python version, NumPy
version, Rust compiler, and whether the extension was built in release mode.
Record those details when publishing or comparing results across machines.

## New benchmark cases

Add cases in `make_cases` in `bench_engine.py` after the Rust implementation and
dispatch tests are stable.

Keep benchmark cases:

- deterministic
- representative of public dispatch behavior
- cheap enough to run across the full matrix
- explicit about cases where parity cannot be exact
- tagged when they should be excluded from the `core` suite

After adding or changing cases, run at least one targeted checked benchmark:

```bash
python benchmarks/bench_engine.py --pattern <subpackage-or-function> --check
```

Then regenerate the matrix reports:

```bash
python benchmarks/bench_matrix.py
```
