"""Run bench_backend.py across multiple configurations and emit a markdown table.

Usage:
    python benchmarks/bench_matrix.py
"""

import csv
import io
import statistics
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

STAT_NAMES = ("count", "min", "median", "mean", "max")


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


def calc_stats(values: list[float]) -> dict[str, float]:
    """Calculate descriptive statistics for benchmark speedups."""
    if len(values) == 0:
        return {}
    return {
        "count": float(len(values)),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def format_stat(name: str, value: float) -> str:
    """Format one statistic value."""
    if name == "count":
        return str(int(value))
    return f"{value:.2f}x"


def make_overall_table(stats: dict[str, float]) -> str:
    """Create a compact markdown table for overall statistics."""
    stat_width = max(len("Statistic"), *(len(name) for name in STAT_NAMES))
    value_width = max(len("Value"), *(len(format_stat(name, stats[name])) for name in STAT_NAMES if name in stats))
    lines = [
        f"| {'Statistic':<{stat_width}} | {'Value':>{value_width}} |",
        f"|{'-' * (stat_width + 2)}|{'-' * (value_width + 2)}|",
    ]
    for name in STAT_NAMES:
        value = stats.get(name)
        if value is not None:
            lines.append(f"| {name:<{stat_width}} | {format_stat(name, value):>{value_width}} |")
    return "\n".join(lines)


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
    stat_labels = [f"stats.{name}" for name in STAT_NAMES]
    func_width = max([*(len(fn) for fn in all_funcs), *(len(label) for label in stat_labels)], default=20)
    per_config_stats = {label: calc_stats(list(all_results[label].values())) for label in labels}
    overall_stats = calc_stats([value for label in labels for value in all_results[label].values()])
    col_widths = []
    for label in labels:
        value_widths = [len(f"{value:.2f}x") for value in all_results[label].values()]
        value_widths.extend(len(format_stat(name, value)) for name, value in per_config_stats[label].items())
        col_widths.append(max(len(label), *value_widths, 6))

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

    lines.append(sep)
    for name, stat_label in zip(STAT_NAMES, stat_labels):
        row = f"| {stat_label:<{func_width}} |"
        for label, w in zip(labels, col_widths):
            value = per_config_stats[label].get(name)
            cell = "-" if value is None else format_stat(name, value)
            row += f" {cell:>{w}} |"
        lines.append(row)

    table = "\n".join(lines)
    overall_table = make_overall_table(overall_stats)
    print(table)
    print()
    print(overall_table)

    out_path = Path(__file__).with_name("BENCHMARKS.md")
    with open(out_path, "w") as f:
        f.write("# Rust vs Numba Speedup Matrix\n\n")
        f.write("Each cell shows **Rust speedup** over Numba (higher = Rust is faster).\n\n")
        f.write(f"- Window: {WINDOW}, NaN ratio: 5%, Repeat: {REPEAT}, Seed: {SEED}\n")
        f.write("- Includes `generic.*`, `indicators.*`, `signals.*`, and `labels.*` ports\n")
        f.write("- Values >1.00x mean Rust is faster; <1.00x mean Numba is faster\n")
        f.write("- Statistics are computed from the speedup scores in this matrix\n\n")
        f.write(table + "\n")
        f.write("\n## Overall Statistics\n\n")
        f.write(overall_table + "\n")

    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
