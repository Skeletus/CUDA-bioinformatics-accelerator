#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SEQUENCE_LENGTHS = [8, 16, 32, 64]
NUM_PAIRS_VALUES = [10, 100, 1000, 5000]
BATCH_SIZES = [256, 1024]
NUM_STREAMS_VALUES = [1, 2, 4]
REPETITIONS = 5
RANDOM_SEED = 42
MATCH_SCORE = 2
MISMATCH_PENALTY = -1
GAP_PENALTY = -2

QUICK_SEQUENCE_LENGTHS = [16, 64]
QUICK_NUM_PAIRS_VALUES = [100, 1000]
QUICK_BATCH_SIZES = [1024]
QUICK_NUM_STREAMS_VALUES = [1, 2]

BENCHMARK_CSV_PATH = Path("benchmarks/needleman_wunsch_gpu_optimized_benchmark_results.csv")

CSV_FIELDNAMES = [
    "algorithm",
    "sequence_length",
    "num_pairs",
    "batch_size",
    "num_streams",
    "total_cells_computed",
    "cpu_time_ms",
    "baseline_gpu_kernel_time_ms",
    "baseline_gpu_total_time_ms",
    "optimized_h2d_copy_time_ms",
    "optimized_gpu_kernel_time_ms",
    "optimized_d2h_copy_time_ms",
    "optimized_gpu_pipeline_time_ms",
    "optimized_end_to_end_time_ms",
    "pipeline_speedup_optimized_vs_baseline",
    "end_to_end_speedup_optimized_vs_baseline",
    "kernel_speedup_vs_cpu",
    "pipeline_speedup_vs_cpu",
    "end_to_end_speedup_vs_cpu",
    "use_pinned_memory",
    "use_cuda_malloc_async",
    "use_cuda_streams",
    "cuda_malloc_async_supported",
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


def parse_bool(values: dict[str, str], key: str) -> str:
    return values.get(key, "false").lower()


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


def compile_baseline_gpu(project_root: Path) -> Path:
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


def compile_optimized_gpu(project_root: Path) -> Path:
    binary_path = executable_path(project_root / "needleman_wunsch_gpu_optimized")
    run_command(
        [
            "nvcc",
            "src/needleman_wunsch_gpu_optimized.cu",
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
    batch_size: int,
    num_streams: int,
    cpu_values: dict[str, str],
    baseline_values: dict[str, str],
    optimized_values: dict[str, str],
) -> dict[str, str | int | float]:
    cpu_time_ms = parse_float(cpu_values, "CPU_TIME_MS")
    baseline_gpu_kernel_time_ms = parse_float(baseline_values, "GPU_KERNEL_TIME_MS")
    baseline_gpu_total_time_ms = parse_float(baseline_values, "GPU_TOTAL_TIME_MS")
    optimized_h2d_copy_time_ms = parse_float(optimized_values, "H2D_COPY_TIME_MS")
    optimized_gpu_kernel_time_ms = parse_float(optimized_values, "GPU_KERNEL_TIME_MS")
    optimized_d2h_copy_time_ms = parse_float(optimized_values, "D2H_COPY_TIME_MS")
    optimized_gpu_pipeline_time_ms = parse_float(optimized_values, "GPU_PIPELINE_TIME_MS")
    optimized_end_to_end_time_ms = parse_float(optimized_values, "END_TO_END_TIME_MS")
    total_cells_computed = num_pairs * (sequence_length + 1) * (sequence_length + 1)

    return {
        "algorithm": "needleman_wunsch_gpu_optimized",
        "sequence_length": sequence_length,
        "num_pairs": num_pairs,
        "batch_size": batch_size,
        "num_streams": num_streams,
        "total_cells_computed": total_cells_computed,
        "cpu_time_ms": cpu_time_ms,
        "baseline_gpu_kernel_time_ms": baseline_gpu_kernel_time_ms,
        "baseline_gpu_total_time_ms": baseline_gpu_total_time_ms,
        "optimized_h2d_copy_time_ms": optimized_h2d_copy_time_ms,
        "optimized_gpu_kernel_time_ms": optimized_gpu_kernel_time_ms,
        "optimized_d2h_copy_time_ms": optimized_d2h_copy_time_ms,
        "optimized_gpu_pipeline_time_ms": optimized_gpu_pipeline_time_ms,
        "optimized_end_to_end_time_ms": optimized_end_to_end_time_ms,
        "pipeline_speedup_optimized_vs_baseline": safe_divide(
            baseline_gpu_total_time_ms,
            optimized_gpu_pipeline_time_ms,
        ),
        "end_to_end_speedup_optimized_vs_baseline": safe_divide(
            baseline_gpu_total_time_ms,
            optimized_end_to_end_time_ms,
        ),
        "kernel_speedup_vs_cpu": safe_divide(cpu_time_ms, optimized_gpu_kernel_time_ms),
        "pipeline_speedup_vs_cpu": safe_divide(cpu_time_ms, optimized_gpu_pipeline_time_ms),
        "end_to_end_speedup_vs_cpu": safe_divide(cpu_time_ms, optimized_end_to_end_time_ms),
        "use_pinned_memory": parse_bool(optimized_values, "USE_PINNED_MEMORY"),
        "use_cuda_malloc_async": parse_bool(optimized_values, "USE_CUDA_MALLOC_ASYNC"),
        "use_cuda_streams": parse_bool(optimized_values, "USE_CUDA_STREAMS"),
        "cuda_malloc_async_supported": parse_bool(optimized_values, "CUDA_MALLOC_ASYNC_SUPPORTED"),
        "validation_status": optimized_values.get("VALIDATION_STATUS", "UNKNOWN"),
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark baseline and optimized Needleman-Wunsch CUDA pipelines.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a smaller benchmark matrix for faster Google Colab validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    sequence_lengths = QUICK_SEQUENCE_LENGTHS if args.quick else SEQUENCE_LENGTHS
    num_pairs_values = QUICK_NUM_PAIRS_VALUES if args.quick else NUM_PAIRS_VALUES
    batch_sizes = QUICK_BATCH_SIZES if args.quick else BATCH_SIZES
    num_streams_values = QUICK_NUM_STREAMS_VALUES if args.quick else NUM_STREAMS_VALUES

    project_root = Path(__file__).resolve().parents[1]
    dataset_directory = project_root / "data" / "synthetic"
    result_directory = project_root / "results" / "needleman_wunsch"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)

    cpu_binary_path = compile_cpu(project_root)
    baseline_gpu_binary_path = compile_baseline_gpu(project_root)
    optimized_gpu_binary_path = compile_optimized_gpu(project_root)

    benchmark_rows: list[dict[str, str | int | float]] = []
    failed_workloads: list[str] = []

    for sequence_length in sequence_lengths:
        for num_pairs in num_pairs_values:
            dataset_path = dataset_directory / f"nw_gpu_optimized_pairs_len{sequence_length}_n{num_pairs}.txt"
            cpu_output_path = result_directory / f"needleman_wunsch_cpu_len{sequence_length}_n{num_pairs}.csv"
            baseline_output_path = result_directory / f"needleman_wunsch_gpu_len{sequence_length}_n{num_pairs}.csv"

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
            baseline_process = run_command(
                [
                    str(baseline_gpu_binary_path),
                    str(dataset_path),
                    str(baseline_output_path),
                    "--match",
                    str(MATCH_SCORE),
                    "--mismatch",
                    str(MISMATCH_PENALTY),
                    "--gap",
                    str(GAP_PENALTY),
                    "--repetitions",
                    str(REPETITIONS),
                    "--implementation",
                    "wavefront",
                    "--summary-only",
                ],
                project_root,
                check=False,
            )

            cpu_values = parse_key_value_output(cpu_process.stdout)
            baseline_values = parse_key_value_output(baseline_process.stdout)

            for batch_size in batch_sizes:
                for num_streams in num_streams_values:
                    optimized_output_path = (
                        result_directory
                        / f"needleman_wunsch_gpu_optimized_len{sequence_length}_n{num_pairs}"
                        f"_batch{batch_size}_streams{num_streams}.csv"
                    )
                    optimized_process = run_command(
                        [
                            str(optimized_gpu_binary_path),
                            str(dataset_path),
                            str(optimized_output_path),
                            "--match",
                            str(MATCH_SCORE),
                            "--mismatch",
                            str(MISMATCH_PENALTY),
                            "--gap",
                            str(GAP_PENALTY),
                            "--repetitions",
                            str(REPETITIONS),
                            "--batch-size",
                            str(batch_size),
                            "--num-streams",
                            str(num_streams),
                            "--summary-only",
                        ],
                        project_root,
                        check=False,
                    )
                    optimized_values = parse_key_value_output(optimized_process.stdout)
                    benchmark_rows.append(
                        build_row(
                            num_pairs,
                            sequence_length,
                            batch_size,
                            num_streams,
                            cpu_values,
                            baseline_values,
                            optimized_values,
                        )
                    )

                    if (
                        cpu_process.returncode != 0
                        or baseline_process.returncode != 0
                        or optimized_process.returncode != 0
                        or baseline_values.get("VALIDATION_STATUS") != "PASSED"
                        or optimized_values.get("VALIDATION_STATUS") != "PASSED"
                    ):
                        failed_workloads.append(
                            f"length={sequence_length}, pairs={num_pairs}, "
                            f"batch={batch_size}, streams={num_streams}"
                        )

    benchmark_csv_path = project_root / BENCHMARK_CSV_PATH
    benchmark_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_csv_path.open("w", encoding="utf-8", newline="") as benchmark_file:
        writer = csv.DictWriter(benchmark_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"Needleman-Wunsch optimized GPU benchmark results saved to: {benchmark_csv_path}")
    if failed_workloads:
        print(f"Error: validation failed for workloads: {', '.join(failed_workloads)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
