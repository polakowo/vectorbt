"""Run bench_backend.py across multiple configurations and emit a markdown table.

Usage:
    python benchmarks/bench_matrix.py
"""

import csv
import io
import subprocess
import sys
from pathlib import Path

BENCH_SCRIPT = str(Path(__file__).with_name("bench_backend.py"))

CONFIGS = [
    {"rows": 100, "cols": 1, "label": "100x1"},
    {"rows": 1_000, "cols": 1, "label": "1Kx1"},
    {"rows": 10_000, "cols": 1, "label": "10Kx1"},
    {"rows": 100_000, "cols": 1, "label": "100Kx1"},
    {"rows": 100, "cols": 10, "label": "100x10"},
    {"rows": 1_000, "cols": 10, "label": "1Kx10"},
    {"rows": 10_000, "cols": 10, "label": "10Kx10"},
    {"rows": 100_000, "cols": 10, "label": "100Kx10"},
    {"rows": 1_000, "cols": 100, "label": "1Kx100"},
    {"rows": 10_000, "cols": 100, "label": "10Kx100"},
]

REPEAT = 5
WARMUP = 2
WINDOW = 20
SEED = 42


def run_config(rows: int, cols: int) -> dict[str, float]:
    """Run benchmark for one configuration, return {func_name: speedup}."""
    cmd = [
        sys.executable,
        BENCH_SCRIPT,
        "--rows",
        str(rows),
        "--cols",
        str(cols),
        "--window",
        str(WINDOW),
        "--repeat",
        str(REPEAT),
        "--warmup",
        str(WARMUP),
        "--seed",
        str(SEED),
        "--check",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED rows={rows} cols={cols}: {result.stderr}", file=sys.stderr)
        return {}
    speedups = {}
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        speedups[row["function"]] = float(row["speedup"])
    return speedups


def main() -> None:
    all_results: dict[str, dict[str, float]] = {}  # {label: {func: speedup}}
    all_funcs: list[str] = []

    for cfg in CONFIGS:
        label = cfg["label"]
        print(f"Running {label} ...", file=sys.stderr)
        speedups = run_config(cfg["rows"], cfg["cols"])
        all_results[label] = speedups
        for fn in speedups:
            if fn not in all_funcs:
                all_funcs.append(fn)

    labels = [cfg["label"] for cfg in CONFIGS]
    col_widths = [max(len(label), 6) for label in labels]
    func_width = max((len(fn) for fn in all_funcs), default=20)

    lines = []
    header = (
        f"| {'Function':<{func_width}} |" + "|".join(f" {label:>{w}} " for label, w in zip(labels, col_widths)) + "|"
    )
    sep = f"|{'-' * (func_width + 2)}|" + "|".join(f"{'-' * (w + 2)}" for w in col_widths) + "|"
    lines.append(header)
    lines.append(sep)

    for fn in all_funcs:
        row = f"| {fn:<{func_width}} |"
        for label, w in zip(labels, col_widths):
            val = all_results[label].get(fn)
            if val is None:
                cell = "-"
            else:
                cell = f"{val:.2f}x"
            row += f" {cell:>{w}} |"
        lines.append(row)

    table = "\n".join(lines)
    print(table)

    out_path = Path(__file__).with_name("BENCHMARKS.md")
    with open(out_path, "w") as f:
        f.write("# Rust vs Numba Speedup Matrix\n\n")
        f.write("Each cell shows **Rust speedup** over Numba (higher = Rust is faster).\n\n")
        f.write(f"- Window: {WINDOW}, NaN ratio: 5%, Repeat: {REPEAT}, Seed: {SEED}\n")
        f.write("- Includes generic kernels and indicator-level `indicator.*` ports\n")
        f.write(f"- Values >1.00x mean Rust is faster; <1.00x mean Numba is faster\n\n")
        f.write(table + "\n")

    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
