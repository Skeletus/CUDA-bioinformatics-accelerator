#!/usr/bin/env python3

import csv
import os
import subprocess
import sys
from pathlib import Path


SEQUENCE_LENGTHS = [8, 16, 32, 64]
NUM_PAIRS_VALUES = [10, 100, 1000]
REPETITIONS = 5
RANDOM_SEED = 42
MATCH_SCORE = 2
MISMATCH_PENALTY = -1
GAP_PENALTY = -2
IMPLEMENTATION = "wavefront"

BENCHMARK_CSV_PATH = Path("benchmarks/needleman_wunsch_gpu_benchmark_results.csv")

CSV_FIELDNAMES = [
    "algorithm",
    "implementation",
    "num_pairs",
    "sequence_length",
    "total_cells_computed",
    "cpu_time_ms",
    "gpu_kernel_time_ms",
    "gpu_total_time_ms",
    "h2d_copy_time_ms",
    "d2h_copy_time_ms",
    "cpu_reference_time_ms",
    "validation_time_ms",
    "kernel_speedup",
    "total_speedup",
    "cells_per_second_cpu",
    "cells_per_second_gpu_kernel",
    "cells_per_second_gpu_total",
    "match_score",
    "mismatch_penalty",
    "gap_penalty",
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


def compile_cpu(project_root: Path) -> Path:
    binary_path = executable_path(project_root / "needleman_wunsch_cpu")
    run_command(
        [
            "g++",
            "src/needleman_wunsch_cpu.cpp",
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


def compile_gpu(project_root: Path) -> Path:
    binary_path = executable_path(project_root / "needleman_wunsch_gpu")
    run_command(
        [
            "nvcc",
            "src/needleman_wunsch_gpu.cu",
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


def build_row(
    num_pairs: int,
    sequence_length: int,
    cpu_values: dict[str, str],
    gpu_values: dict[str, str],
) -> dict[str, str | int | float]:
    cpu_time_ms = parse_float(cpu_values, "CPU_TIME_MS")
    gpu_kernel_time_ms = parse_float(gpu_values, "GPU_KERNEL_TIME_MS")
    gpu_total_time_ms = parse_float(gpu_values, "GPU_TOTAL_TIME_MS")
    h2d_copy_time_ms = parse_float(gpu_values, "H2D_COPY_TIME_MS")
    d2h_copy_time_ms = parse_float(gpu_values, "D2H_COPY_TIME_MS")
    cpu_reference_time_ms = parse_float(gpu_values, "CPU_REFERENCE_TIME_MS")
    validation_time_ms = parse_float(gpu_values, "VALIDATION_TIME_MS")
    total_cells_computed = num_pairs * (sequence_length + 1) * (sequence_length + 1)

    return {
        "algorithm": "needleman_wunsch_gpu",
        "implementation": gpu_values.get("IMPLEMENTATION", IMPLEMENTATION),
        "num_pairs": num_pairs,
        "sequence_length": sequence_length,
        "total_cells_computed": total_cells_computed,
        "cpu_time_ms": cpu_time_ms,
        "gpu_kernel_time_ms": gpu_kernel_time_ms,
        "gpu_total_time_ms": gpu_total_time_ms,
        "h2d_copy_time_ms": h2d_copy_time_ms,
        "d2h_copy_time_ms": d2h_copy_time_ms,
        "cpu_reference_time_ms": cpu_reference_time_ms,
        "validation_time_ms": validation_time_ms,
        "kernel_speedup": safe_divide(cpu_time_ms, gpu_kernel_time_ms),
        "total_speedup": safe_divide(cpu_time_ms, gpu_total_time_ms),
        "cells_per_second_cpu": safe_divide(float(total_cells_computed), cpu_time_ms / 1000.0),
        "cells_per_second_gpu_kernel": safe_divide(float(total_cells_computed), gpu_kernel_time_ms / 1000.0),
        "cells_per_second_gpu_total": safe_divide(float(total_cells_computed), gpu_total_time_ms / 1000.0),
        "match_score": MATCH_SCORE,
        "mismatch_penalty": MISMATCH_PENALTY,
        "gap_penalty": GAP_PENALTY,
        "validation_status": gpu_values.get("VALIDATION_STATUS", "UNKNOWN"),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    dataset_directory = project_root / "data" / "synthetic"
    result_directory = project_root / "results" / "needleman_wunsch"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)

    cpu_binary_path = compile_cpu(project_root)
    gpu_binary_path = compile_gpu(project_root)

    benchmark_rows: list[dict[str, str | int | float]] = []
    failed_workloads: list[str] = []

    for sequence_length in SEQUENCE_LENGTHS:
        for num_pairs in NUM_PAIRS_VALUES:
            dataset_path = dataset_directory / f"nw_gpu_synthetic_pairs_len{sequence_length}_n{num_pairs}.txt"
            cpu_output_path = result_directory / f"needleman_wunsch_cpu_len{sequence_length}_n{num_pairs}.csv"
            gpu_output_path = result_directory / f"needleman_wunsch_gpu_len{sequence_length}_n{num_pairs}.csv"

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

            cpu_process = run_command(
                [
                    str(cpu_binary_path),
                    str(dataset_path),
                    str(cpu_output_path),
                    "--match",
                    str(MATCH_SCORE),
                    "--mismatch",
                    str(MISMATCH_PENALTY),
                    "--gap",
                    str(GAP_PENALTY),
                    "--repetitions",
                    str(REPETITIONS),
                    "--memory-mode",
                    "rolling",
                ],
                project_root,
                check=False,
            )
            gpu_process = run_command(
                [
                    str(gpu_binary_path),
                    str(dataset_path),
                    str(gpu_output_path),
                    "--match",
                    str(MATCH_SCORE),
                    "--mismatch",
                    str(MISMATCH_PENALTY),
                    "--gap",
                    str(GAP_PENALTY),
                    "--repetitions",
                    str(REPETITIONS),
                    "--implementation",
                    IMPLEMENTATION,
                    "--summary-only",
                ],
                project_root,
                check=False,
            )

            cpu_values = parse_key_value_output(cpu_process.stdout)
            gpu_values = parse_key_value_output(gpu_process.stdout)
            benchmark_rows.append(build_row(num_pairs, sequence_length, cpu_values, gpu_values))

            if (
                cpu_process.returncode != 0
                or gpu_process.returncode != 0
                or gpu_values.get("VALIDATION_STATUS") != "PASSED"
            ):
                failed_workloads.append(f"length={sequence_length}, pairs={num_pairs}")

    benchmark_csv_path = project_root / BENCHMARK_CSV_PATH
    benchmark_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_csv_path.open("w", encoding="utf-8", newline="") as benchmark_file:
        writer = csv.DictWriter(benchmark_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"Needleman-Wunsch GPU benchmark results saved to: {benchmark_csv_path}")
    if failed_workloads:
        print(f"Error: validation failed for workloads: {', '.join(failed_workloads)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
