#!/usr/bin/env python3

import csv
import os
import subprocess
import sys
from pathlib import Path


SEQUENCE_LENGTHS = [16, 32, 64, 128, 256]
NUM_PAIRS_VALUES = [10, 100, 1000]
REPETITIONS = 3
RANDOM_SEED = 42
MATCH_SCORE = 2
MISMATCH_PENALTY = -1
GAP_PENALTY = -2
MEMORY_MODE = "rolling"

BENCHMARK_CSV_PATH = Path("benchmarks/smith_waterman_cpu_benchmark_results.csv")

CSV_FIELDNAMES = [
    "algorithm",
    "num_pairs",
    "sequence_length",
    "total_cells_computed",
    "cpu_time_ms",
    "average_time_per_pair_ms",
    "cells_per_second",
    "match_score",
    "mismatch_penalty",
    "gap_penalty",
    "memory_mode",
    "validation_status",
]


def executable_path(path: Path) -> Path:
    if os.name == "nt":
        return path.with_suffix(".exe")
    return path


def run_command(command: list[str], project_root: Path, *, check: bool) -> subprocess.CompletedProcess[str]:
    print("Running:", " ".join(str(part) for part in command))
    completed_process = subprocess.run(
        command,
        cwd=project_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed_process.stdout:
        print(completed_process.stdout, end="")
    if check and completed_process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {completed_process.returncode}: "
            f"{' '.join(str(part) for part in command)}"
        )
    return completed_process


def parse_key_value_output(output: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key:
            values[key] = value
    return values


def parse_float(values: dict[str, str], key: str) -> float:
    value = values.get(key)
    if value is None:
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def compile_program(project_root: Path) -> Path:
    binary_path = executable_path(project_root / "smith_waterman_cpu")
    run_command(
        [
            "g++",
            "src/smith_waterman_cpu.cpp",
            "-O3",
            "-std=c++17",
            "-I",
            "src/common",
            "-o",
            str(binary_path),
        ],
        project_root,
        check=True,
    )
    return binary_path


def build_row(num_pairs: int, sequence_length: int, values: dict[str, str]) -> dict[str, str | int | float]:
    cpu_time_ms = parse_float(values, "CPU_TIME_MS")
    total_cells_computed = num_pairs * (sequence_length + 1) * (sequence_length + 1)
    return {
        "algorithm": values.get("ALGORITHM", "smith_waterman_cpu"),
        "num_pairs": num_pairs,
        "sequence_length": sequence_length,
        "total_cells_computed": total_cells_computed,
        "cpu_time_ms": cpu_time_ms,
        "average_time_per_pair_ms": safe_divide(cpu_time_ms, float(num_pairs)),
        "cells_per_second": safe_divide(float(total_cells_computed), cpu_time_ms / 1000.0),
        "match_score": MATCH_SCORE,
        "mismatch_penalty": MISMATCH_PENALTY,
        "gap_penalty": GAP_PENALTY,
        "memory_mode": values.get("MEMORY_MODE", MEMORY_MODE),
        "validation_status": values.get("VALIDATION_STATUS", "UNKNOWN"),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    dataset_directory = project_root / "data" / "synthetic"
    result_directory = project_root / "results" / "smith_waterman"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)

    binary_path = compile_program(project_root)
    benchmark_rows: list[dict[str, str | int | float]] = []
    failed_workloads: list[str] = []

    for sequence_length in SEQUENCE_LENGTHS:
        for num_pairs in NUM_PAIRS_VALUES:
            dataset_path = dataset_directory / f"sw_synthetic_pairs_len{sequence_length}_n{num_pairs}.txt"
            output_path = result_directory / f"smith_waterman_cpu_len{sequence_length}_n{num_pairs}.csv"

            run_command(
                [
                    sys.executable,
                    "scripts/generate_synthetic_dataset.py",
                    "--num-pairs",
                    str(num_pairs),
                    "--sequence-length",
                    str(sequence_length),
                    "--output",
                    str(dataset_path),
                    "--seed",
                    str(RANDOM_SEED),
                ],
                project_root,
                check=True,
            )

            completed_process = run_command(
                [
                    str(binary_path),
                    str(dataset_path),
                    str(output_path),
                    "--match",
                    str(MATCH_SCORE),
                    "--mismatch",
                    str(MISMATCH_PENALTY),
                    "--gap",
                    str(GAP_PENALTY),
                    "--repetitions",
                    str(REPETITIONS),
                    "--memory-mode",
                    MEMORY_MODE,
                ],
                project_root,
                check=False,
            )
            values = parse_key_value_output(completed_process.stdout)
            benchmark_rows.append(build_row(num_pairs, sequence_length, values))
            if completed_process.returncode != 0 or values.get("VALIDATION_STATUS") != "PASSED":
                failed_workloads.append(f"length={sequence_length}, pairs={num_pairs}")

    BENCHMARK_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with (project_root / BENCHMARK_CSV_PATH).open("w", encoding="utf-8", newline="") as benchmark_file:
        writer = csv.DictWriter(benchmark_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"Smith-Waterman CPU benchmark results saved to: {project_root / BENCHMARK_CSV_PATH}")
    if failed_workloads:
        print(f"Error: validation failed for workloads: {', '.join(failed_workloads)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
