import importlib.util
from pathlib import Path


def load_bench_matrix():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_matrix.py"
    spec = importlib.util.spec_from_file_location("bench_matrix", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_bench_engine():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_engine.py"
    spec = importlib.util.spec_from_file_location("bench_engine", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_filter_cases_by_suite_keeps_core_clean_and_extended_complete():
    bench_engine = load_bench_engine()

    core_case = bench_engine.BenchmarkCase("core", len, ([],), len, ([],))
    extended_case = bench_engine.BenchmarkCase("extended", len, ([],), len, ([],), tags=("o1",))

    assert bench_engine.filter_cases_by_suite([core_case, extended_case], "core") == [core_case]
    assert bench_engine.filter_cases_by_suite([core_case, extended_case], "extended") == [
        core_case,
        extended_case,
    ]


def test_write_report_files_emits_speedup_numba_and_rust_matrices(tmp_path):
    bench_matrix = load_bench_matrix()
    labels = ["100x1"]
    funcs = ["generic.fillna"]
    results = {
        "100x1": {
            "generic.fillna": {
                "numba_s": 0.000002,
                "rust_s": 0.000001,
                "speedup": 2.0,
            }
        }
    }

    written = bench_matrix.write_report_files(
        tmp_path / "BENCHMARKS.md",
        labels,
        funcs,
        results,
        "view",
        "core",
    )

    speedup_text = written["speedup"].read_text()
    numba_text = written["numba_s"].read_text()
    rust_text = written["rust_s"].read_text()

    assert "# Rust vs Numba Speedup Matrix" in speedup_text
    assert "Each cell shows **Rust speedup** over Numba" in speedup_text
    assert "Suite: core" in speedup_text
    assert "2.00x" in speedup_text

    assert "# Numba Absolute Runtime Matrix" in numba_text
    assert "Each cell shows the absolute Numba execution time" in numba_text
    assert "2.00 us" in numba_text

    assert "# Rust Absolute Runtime Matrix" in rust_text
    assert "Each cell shows the absolute Rust execution time" in rust_text
    assert "1.00 us" in rust_text
