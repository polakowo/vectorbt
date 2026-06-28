"""Run bench_engine.py across multiple configurations and emit markdown tables.

Usage:
    python benchmarks/bench_matrix.py
"""

import argparse
import csv
import io
import statistics
import subprocess
import sys
from pathlib import Path

BENCH_SCRIPT = str(Path(__file__).with_name("bench_engine.py"))
LAYOUT_CHOICES = ("contiguous", "view", "copy-included")
DEFAULT_LAYOUT = "view"
SUITE_CHOICES = ("core", "extended")
DEFAULT_SUITE = "core"

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


def run_config(rows: int, cols: int, layout: str, suite: str) -> dict[str, dict[str, float]]:
    """Run benchmark for one configuration, return metrics by function name."""
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
        "--layout",
        layout,
        "--suite",
        suite,
        "--check",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED rows={rows} cols={cols}: {result.stderr}", file=sys.stderr)
        return {}
    results = {}
    reader = csv.DictReader(io.StringIO(result.stdout))
    for row in reader:
        results[row["function"]] = {
            "numba_s": float(row["numba_s"]),
            "rust_s": float(row["rust_s"]),
            "speedup": float(row["speedup"]),
        }
    return results


def calc_stats(values: list[float]) -> dict[str, float]:
    """Calculate descriptive statistics for benchmark values."""
    if len(values) == 0:
        return {}
    return {
        "count": float(len(values)),
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def format_duration(seconds: float) -> str:
    """Format seconds using a compact duration unit."""
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} us"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def format_metric_value(metric: str, value: float) -> str:
    """Format one matrix value."""
    if metric == "speedup":
        return f"{value:.2f}x"
    return format_duration(value)


def format_stat(name: str, metric: str, value: float) -> str:
    """Format one statistic value."""
    if name == "count":
        return str(int(value))
    return format_metric_value(metric, value)


def make_overall_table(stats: dict[str, float], metric: str) -> str:
    """Create a compact markdown table for overall statistics."""
    stat_width = max(len("Statistic"), *(len(name) for name in STAT_NAMES))
    value_width = max(
        len("Value"),
        *(len(format_stat(name, metric, stats[name])) for name in STAT_NAMES if name in stats),
    )
    lines = [
        f"| {'Statistic':<{stat_width}} | {'Value':>{value_width}} |",
        f"|{'-' * (stat_width + 2)}|{'-' * (value_width + 2)}|",
    ]
    for name in STAT_NAMES:
        value = stats.get(name)
        if value is not None:
            lines.append(f"| {name:<{stat_width}} | {format_stat(name, metric, value):>{value_width}} |")
    return "\n".join(lines)


def make_per_config_table(stats_by_label: dict[str, dict[str, float]], labels: list[str], metric: str) -> str:
    """Create a markdown table for statistics computed per configuration."""
    stat_width = max(len("Statistic"), *(len(name) for name in STAT_NAMES))
    col_widths = []
    for label in labels:
        value_widths = [
            len(format_stat(name, metric, value))
            for name, value in stats_by_label[label].items()
        ]
        col_widths.append(max(len(label), *value_widths, 6))

    lines = [
        f"| {'Statistic':<{stat_width}} |"
        + "|".join(f" {label:>{w}} " for label, w in zip(labels, col_widths))
        + "|",
        f"|{'-' * (stat_width + 2)}|"
        + "|".join(f"{'-' * (w + 2)}" for w in col_widths)
        + "|",
    ]
    for name in STAT_NAMES:
        row = f"| {name:<{stat_width}} |"
        for label, w in zip(labels, col_widths):
            value = stats_by_label[label].get(name)
            cell = "-" if value is None else format_stat(name, metric, value)
            row += f" {cell:>{w}} |"
        lines.append(row)
    return "\n".join(lines)


def make_matrix(
    labels: list[str],
    all_funcs: list[str],
    all_results: dict[str, dict[str, dict[str, float]]],
    metric: str,
) -> tuple[str, str, str]:
    """Create a markdown matrix plus per-config and overall statistics tables."""
    func_width = max([*(len(fn) for fn in all_funcs)], default=20)
    per_config_stats = {
        label: calc_stats([values[metric] for values in all_results[label].values()]) for label in labels
    }
    overall_stats = calc_stats(
        [values[metric] for label in labels for values in all_results[label].values()]
    )
    col_widths = []
    for label in labels:
        value_widths = [len(format_metric_value(metric, values[metric])) for values in all_results[label].values()]
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
            values = all_results[label].get(fn)
            cell = "-" if values is None else format_metric_value(metric, values[metric])
            row += f" {cell:>{w}} |"
        lines.append(row)

    return (
        "\n".join(lines),
        make_per_config_table(per_config_stats, labels, metric),
        make_overall_table(overall_stats, metric),
    )


def companion_output_path(output_path: Path, suffix: str) -> Path:
    """Return a companion matrix path derived from the speedup output path."""
    return output_path.with_name(f"{output_path.stem}_{suffix}{output_path.suffix}")


def write_matrix_file(
    path: Path,
    title: str,
    description: str,
    notes: list[str],
    table: str,
    per_config_table: str,
    overall_table: str,
    layout: str,
    suite: str,
) -> None:
    """Write one documented benchmark matrix."""
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write(description + "\n\n")
        f.write(
            f"- Window: {WINDOW}, NaN ratio: 5%, Repeat: {REPEAT}, Seed: {SEED}, Layout: {layout}, Suite: {suite}\n"
        )
        for note in notes:
            f.write(f"- {note}\n")
        f.write("\n")
        f.write(table + "\n")
        f.write("\n## Per-Config Statistics\n\n")
        f.write(per_config_table + "\n")
        f.write("\n## Overall Statistics\n\n")
        f.write(overall_table + "\n")


def write_report_files(
    output_path: Path,
    labels: list[str],
    all_funcs: list[str],
    all_results: dict[str, dict[str, dict[str, float]]],
    layout: str,
    suite: str,
) -> dict[str, Path]:
    """Write speedup, Numba runtime, and Rust runtime matrix reports."""
    reports = {
        "speedup": {
            "path": output_path,
            "title": "Rust vs Numba Speedup Matrix",
            "description": "Each cell shows **Rust speedup** over Numba (higher = Rust is faster).",
            "notes": [
                "Values >1.00x mean Rust is faster; <1.00x mean Numba is faster",
                "Statistics are computed from the speedup scores in this matrix",
            ],
        },
        "numba_s": {
            "path": companion_output_path(output_path, "NUMBA"),
            "title": "Numba Absolute Runtime Matrix",
            "description": "Each cell shows the absolute Numba execution time for one benchmark call.",
            "notes": [
                "Lower values are faster",
                "Runtime is the best measured call time after warmup, formatted by duration unit",
                "Statistics are computed from the Numba runtimes in this matrix",
            ],
        },
        "rust_s": {
            "path": companion_output_path(output_path, "RUST"),
            "title": "Rust Absolute Runtime Matrix",
            "description": "Each cell shows the absolute Rust execution time for one benchmark call.",
            "notes": [
                "Lower values are faster",
                "Runtime is the best measured call time after warmup, formatted by duration unit",
                "Statistics are computed from the Rust runtimes in this matrix",
            ],
        },
    }

    written: dict[str, Path] = {}
    for metric, report in reports.items():
        table, per_config_table, overall_table = make_matrix(labels, all_funcs, all_results, metric)
        write_matrix_file(
            report["path"],
            report["title"],
            report["description"],
            report["notes"],
            table,
            per_config_table,
            overall_table,
            layout,
            suite,
        )
        written[metric] = report["path"]
    return written


def main() -> None:
    """Run all benchmark configurations and write the markdown matrix."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        choices=LAYOUT_CHOICES,
        default=DEFAULT_LAYOUT,
        help="Benchmark layout. Defaults to 'view', which passes strided 1D column views.",
    )
    parser.add_argument(
        "--suite",
        choices=SUITE_CHOICES,
        default=DEFAULT_SUITE,
        help="'core' excludes scalar/O(1)/metadata/cache-lookup cases; 'extended' includes all benchmark cases.",
    )
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("BENCHMARKS.md"))
    args = parser.parse_args()

    all_results: dict[str, dict[str, dict[str, float]]] = {}
    all_funcs: list[str] = []

    for cfg in CONFIGS:
        label = cfg["label"]
        print(f"Running {label} ...", file=sys.stderr)
        results = run_config(cfg["rows"], cfg["cols"], args.layout, args.suite)
        all_results[label] = results
        for fn in results:
            if fn not in all_funcs:
                all_funcs.append(fn)

    labels = [cfg["label"] for cfg in CONFIGS]
    written = write_report_files(args.output, labels, all_funcs, all_results, args.layout, args.suite)
    table, per_config_table, overall_table = make_matrix(labels, all_funcs, all_results, "speedup")
    print(table)
    print()
    print(per_config_table)
    print()
    print(overall_table)
    for path in written.values():
        print(f"Written to {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
