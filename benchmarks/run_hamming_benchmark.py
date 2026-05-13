#!/usr/bin/env python3

import csv
import os
import re
import subprocess
import sys
from pathlib import Path


SEQUENCE_LENGTHS = [64, 128, 256]
NUM_PAIR_VALUES = [1000, 10000, 100000]


def executable_path(path: Path) -> Path:
    if os.name == "nt":
        return path.with_suffix(".exe")
    return path


def run_command(command: list[str], project_root: Path) -> subprocess.CompletedProcess[str]:
    print("Running:", " ".join(str(part) for part in command))
    return subprocess.run(
        command,
        cwd=project_root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def parse_float_metric(output: str, metric_name: str) -> float | None:
    pattern = rf"{re.escape(metric_name)}:\s+([0-9]+(?:\.[0-9]+)?)\s+ms"
    match = re.search(pattern, output)
    if match is None:
        return None
    return float(match.group(1))


def parse_validation_status(output: str) -> str:
    match = re.search(r"Validation status:\s+([A-Z]+)", output)
    if match is None:
        return "UNKNOWN"
    return match.group(1)


def compile_programs(project_root: Path, binary_directory: Path) -> tuple[Path, Path]:
    binary_directory.mkdir(parents=True, exist_ok=True)
    cpu_binary = executable_path(binary_directory / "hamming_cpu")
    gpu_binary = executable_path(binary_directory / "hamming_gpu")

    run_command(
        [
            "g++",
            "src/hamming_cpu.cpp",
            "-O3",
            "-std=c++17",
            "-I",
            "src/common",
            "-o",
            str(cpu_binary),
        ],
        project_root,
    )
    run_command(
        [
            "nvcc",
            "src/hamming_gpu.cu",
            "-O3",
            "-std=c++17",
            "-I",
            "src/common",
            "-o",
            str(gpu_binary),
        ],
        project_root,
    )

    return cpu_binary, gpu_binary


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    binary_directory = project_root / "benchmarks" / "bin"
    dataset_directory = project_root / "data" / "synthetic"
    result_directory = project_root / "results" / "hamming"
    benchmark_csv_path = project_root / "benchmarks" / "hamming_benchmark_results.csv"

    dataset_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)

    cpu_binary, gpu_binary = compile_programs(project_root, binary_directory)

    benchmark_rows: list[dict[str, str | int | float | None]] = []
    for sequence_length in SEQUENCE_LENGTHS:
        for num_pairs in NUM_PAIR_VALUES:
            dataset_path = dataset_directory / f"synthetic_pairs_len{sequence_length}_n{num_pairs}.txt"
            cpu_output_path = result_directory / f"hamming_cpu_len{sequence_length}_n{num_pairs}.csv"
            gpu_output_path = result_directory / f"hamming_gpu_len{sequence_length}_n{num_pairs}.csv"

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
                    "42",
                ],
                project_root,
            )

            cpu_run = run_command([str(cpu_binary), str(dataset_path), str(cpu_output_path)], project_root)
            gpu_run = run_command([str(gpu_binary), str(dataset_path), str(gpu_output_path)], project_root)

            benchmark_rows.append(
                {
                    "sequence_length": sequence_length,
                    "num_pairs": num_pairs,
                    "cpu_time_ms": parse_float_metric(cpu_run.stdout, "CPU time"),
                    "gpu_kernel_time_ms": parse_float_metric(gpu_run.stdout, "GPU kernel time"),
                    "gpu_total_time_ms": parse_float_metric(gpu_run.stdout, "GPU total time"),
                    "validation_status": parse_validation_status(gpu_run.stdout),
                }
            )

    with benchmark_csv_path.open("w", newline="", encoding="utf-8") as benchmark_file:
        fieldnames = [
            "sequence_length",
            "num_pairs",
            "cpu_time_ms",
            "gpu_kernel_time_ms",
            "gpu_total_time_ms",
            "validation_status",
        ]
        writer = csv.DictWriter(benchmark_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"Benchmark results saved to: {benchmark_csv_path}")


if __name__ == "__main__":
    main()
