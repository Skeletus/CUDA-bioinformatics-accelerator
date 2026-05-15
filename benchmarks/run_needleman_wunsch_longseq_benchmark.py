#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


SEQUENCE_LENGTHS = [64, 128, 256, 512]
NUM_PAIRS_VALUES = [10, 100, 1000]
IMPLEMENTATIONS = ["global_matrix", "rolling_diagonal", "tiled_wavefront"]
REPETITIONS = 3

QUICK_SEQUENCE_LENGTHS = [64, 128]
QUICK_NUM_PAIRS_VALUES = [10, 100]
QUICK_IMPLEMENTATIONS = ["global_matrix", "rolling_diagonal"]
QUICK_REPETITIONS = 2

RANDOM_SEED = 42
MATCH_SCORE = 2
MISMATCH_PENALTY = -1
GAP_PENALTY = -2
SHARED_MEMORY_SEQUENCE_LIMIT = 64

BENCHMARK_CSV_PATH = Path("benchmarks/needleman_wunsch_longseq_benchmark_results.csv")

CSV_FIELDNAMES = [
    "algorithm",
    "implementation",
    "implementation_status",
    "num_pairs",
    "sequence_length",
    "total_cells_computed",
    "cpu_time_ms",
    "gpu_kernel_time_ms",
    "gpu_total_time_ms",
    "h2d_copy_time_ms",
    "d2h_copy_time_ms",
    "kernel_speedup_vs_cpu",
    "total_speedup_vs_cpu",
    "gpu_kernel_cells_per_second",
    "gpu_total_cells_per_second",
    "dp_memory_mode",
    "dp_memory_bytes",
    "max_supported_sequence_length",
    "validation_status",
    "notes",
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


def parse_int(values: dict[str, str], key: str) -> int:
    value = values.get(key)
    if value is None:
        return 0
    try:
        return int(float(value))
    except ValueError:
        return 0


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def total_cells(num_pairs: int, sequence_length: int) -> int:
    return num_pairs * (sequence_length + 1) * (sequence_length + 1)


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


def compile_longseq_gpu(project_root: Path) -> Path:
    binary_path = executable_path(project_root / "needleman_wunsch_gpu_longseq")
    run_command(
        [
            "nvcc",
            "src/needleman_wunsch_gpu_longseq.cu",
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


def build_cpu_row(
    num_pairs: int,
    sequence_length: int,
    cpu_values: dict[str, str],
) -> dict[str, str | int | float]:
    cpu_time_ms = parse_float(cpu_values, "CPU_TIME_MS")
    return {
        "algorithm": "needleman_wunsch_cpu",
        "implementation": cpu_values.get("MEMORY_MODE", "rolling"),
        "implementation_status": "STABLE",
        "num_pairs": num_pairs,
        "sequence_length": sequence_length,
        "total_cells_computed": total_cells(num_pairs, sequence_length),
        "cpu_time_ms": cpu_time_ms,
        "gpu_kernel_time_ms": 0.0,
        "gpu_total_time_ms": 0.0,
        "h2d_copy_time_ms": 0.0,
        "d2h_copy_time_ms": 0.0,
        "kernel_speedup_vs_cpu": 1.0,
        "total_speedup_vs_cpu": 1.0,
        "gpu_kernel_cells_per_second": 0.0,
        "gpu_total_cells_per_second": 0.0,
        "dp_memory_mode": "cpu_rolling_rows",
        "dp_memory_bytes": 0,
        "max_supported_sequence_length": "",
        "validation_status": cpu_values.get("VALIDATION_STATUS", "UNKNOWN"),
        "notes": "CPU reference row.",
    }


def build_gpu_row(
    algorithm: str,
    implementation: str,
    implementation_status: str,
    num_pairs: int,
    sequence_length: int,
    cpu_time_ms: float,
    gpu_values: dict[str, str],
    *,
    dp_memory_mode: str,
    dp_memory_bytes: int,
    max_supported_sequence_length: int | str,
    notes: str,
) -> dict[str, str | int | float]:
    gpu_kernel_time_ms = parse_float(gpu_values, "GPU_KERNEL_TIME_MS")
    gpu_total_time_ms = parse_float(gpu_values, "GPU_TOTAL_TIME_MS")
    h2d_copy_time_ms = parse_float(gpu_values, "H2D_COPY_TIME_MS")
    d2h_copy_time_ms = parse_float(gpu_values, "D2H_COPY_TIME_MS")
    cells = total_cells(num_pairs, sequence_length)
    return {
        "algorithm": algorithm,
        "implementation": implementation,
        "implementation_status": implementation_status,
        "num_pairs": num_pairs,
        "sequence_length": sequence_length,
        "total_cells_computed": cells,
        "cpu_time_ms": cpu_time_ms,
        "gpu_kernel_time_ms": gpu_kernel_time_ms,
        "gpu_total_time_ms": gpu_total_time_ms,
        "h2d_copy_time_ms": h2d_copy_time_ms,
        "d2h_copy_time_ms": d2h_copy_time_ms,
        "kernel_speedup_vs_cpu": safe_divide(cpu_time_ms, gpu_kernel_time_ms),
        "total_speedup_vs_cpu": safe_divide(cpu_time_ms, gpu_total_time_ms),
        "gpu_kernel_cells_per_second": parse_float(
            gpu_values,
            "GPU_KERNEL_CELLS_PER_SECOND",
        )
        or safe_divide(float(cells), gpu_kernel_time_ms / 1000.0),
        "gpu_total_cells_per_second": parse_float(
            gpu_values,
            "GPU_TOTAL_CELLS_PER_SECOND",
        )
        or safe_divide(float(cells), gpu_total_time_ms / 1000.0),
        "dp_memory_mode": dp_memory_mode,
        "dp_memory_bytes": dp_memory_bytes,
        "max_supported_sequence_length": max_supported_sequence_length,
        "validation_status": gpu_values.get("VALIDATION_STATUS", "UNKNOWN"),
        "notes": notes,
    }


def build_unsupported_row(
    algorithm: str,
    implementation: str,
    implementation_status: str,
    num_pairs: int,
    sequence_length: int,
    cpu_time_ms: float,
    *,
    dp_memory_mode: str,
    max_supported_sequence_length: int | str,
    notes: str,
) -> dict[str, str | int | float]:
    return {
        "algorithm": algorithm,
        "implementation": implementation,
        "implementation_status": implementation_status,
        "num_pairs": num_pairs,
        "sequence_length": sequence_length,
        "total_cells_computed": total_cells(num_pairs, sequence_length),
        "cpu_time_ms": cpu_time_ms,
        "gpu_kernel_time_ms": 0.0,
        "gpu_total_time_ms": 0.0,
        "h2d_copy_time_ms": 0.0,
        "d2h_copy_time_ms": 0.0,
        "kernel_speedup_vs_cpu": 0.0,
        "total_speedup_vs_cpu": 0.0,
        "gpu_kernel_cells_per_second": 0.0,
        "gpu_total_cells_per_second": 0.0,
        "dp_memory_mode": dp_memory_mode,
        "dp_memory_bytes": 0,
        "max_supported_sequence_length": max_supported_sequence_length,
        "validation_status": "UNSUPPORTED",
        "notes": notes,
    }


def first_error_line(output: str) -> str:
    for line in output.splitlines():
        if line.strip():
            return line.strip()
    return "Command failed before producing output."


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Needleman-Wunsch long-sequence GPU implementations.")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run the smaller validation-oriented benchmark matrix.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    sequence_lengths = QUICK_SEQUENCE_LENGTHS if args.quick else SEQUENCE_LENGTHS
    num_pairs_values = QUICK_NUM_PAIRS_VALUES if args.quick else NUM_PAIRS_VALUES
    implementations = QUICK_IMPLEMENTATIONS if args.quick else IMPLEMENTATIONS
    repetitions = QUICK_REPETITIONS if args.quick else REPETITIONS

    project_root = Path(__file__).resolve().parents[1]
    dataset_directory = project_root / "data" / "synthetic"
    result_directory = project_root / "results" / "needleman_wunsch"
    dataset_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)

    cpu_binary_path = compile_cpu(project_root)
    baseline_gpu_binary_path = compile_baseline_gpu(project_root)
    optimized_gpu_binary_path = compile_optimized_gpu(project_root)
    longseq_gpu_binary_path = compile_longseq_gpu(project_root)

    benchmark_rows: list[dict[str, str | int | float]] = []

    for sequence_length in sequence_lengths:
        for num_pairs in num_pairs_values:
            dataset_path = dataset_directory / f"nw_longseq_pairs_len{sequence_length}_n{num_pairs}.txt"
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

            cpu_output_path = result_directory / f"needleman_wunsch_cpu_longseq_len{sequence_length}_n{num_pairs}.csv"
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
                    str(repetitions),
                    "--memory-mode",
                    "rolling",
                ],
                project_root,
                check=False,
            )
            cpu_values = parse_key_value_output(cpu_process.stdout)
            cpu_time_ms = parse_float(cpu_values, "CPU_TIME_MS")
            benchmark_rows.append(build_cpu_row(num_pairs, sequence_length, cpu_values))

            if sequence_length <= SHARED_MEMORY_SEQUENCE_LIMIT:
                baseline_output_path = (
                    result_directory / f"needleman_wunsch_gpu_longseq_compare_len{sequence_length}_n{num_pairs}.csv"
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
                        str(repetitions),
                        "--implementation",
                        "wavefront",
                        "--summary-only",
                    ],
                    project_root,
                    check=False,
                )
                baseline_values = parse_key_value_output(baseline_process.stdout)
                benchmark_rows.append(
                    build_gpu_row(
                        "needleman_wunsch_gpu",
                        "wavefront",
                        "STABLE",
                        num_pairs,
                        sequence_length,
                        cpu_time_ms,
                        baseline_values,
                        dp_memory_mode="shared",
                        dp_memory_bytes=(sequence_length + 1) * (sequence_length + 1) * 4,
                        max_supported_sequence_length=SHARED_MEMORY_SEQUENCE_LIMIT,
                        notes="Shared-memory Phase 10 prototype.",
                    )
                )

                optimized_output_path = (
                    result_directory
                    / f"needleman_wunsch_gpu_optimized_longseq_compare_len{sequence_length}_n{num_pairs}.csv"
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
                        str(repetitions),
                        "--summary-only",
                    ],
                    project_root,
                    check=False,
                )
                optimized_values = parse_key_value_output(optimized_process.stdout)
                benchmark_rows.append(
                    build_gpu_row(
                        "needleman_wunsch_gpu_optimized",
                        optimized_values.get("IMPLEMENTATION", "wavefront_streamed"),
                        "STABLE",
                        num_pairs,
                        sequence_length,
                        cpu_time_ms,
                        {
                            **optimized_values,
                            "GPU_TOTAL_TIME_MS": optimized_values.get("GPU_PIPELINE_TIME_MS", "0"),
                        },
                        dp_memory_mode="shared",
                        dp_memory_bytes=(sequence_length + 1) * (sequence_length + 1) * 4,
                        max_supported_sequence_length=SHARED_MEMORY_SEQUENCE_LIMIT,
                        notes="Phase 10.1 shared-memory pipeline optimization.",
                    )
                )
            else:
                unsupported_note = (
                    "Shared-memory prototype supports sequence lengths up to 64 in the current implementation."
                )
                benchmark_rows.append(
                    build_unsupported_row(
                        "needleman_wunsch_gpu",
                        "wavefront",
                        "STABLE",
                        num_pairs,
                        sequence_length,
                        cpu_time_ms,
                        dp_memory_mode="shared",
                        max_supported_sequence_length=SHARED_MEMORY_SEQUENCE_LIMIT,
                        notes=unsupported_note,
                    )
                )
                benchmark_rows.append(
                    build_unsupported_row(
                        "needleman_wunsch_gpu_optimized",
                        "wavefront_streamed",
                        "STABLE",
                        num_pairs,
                        sequence_length,
                        cpu_time_ms,
                        dp_memory_mode="shared",
                        max_supported_sequence_length=SHARED_MEMORY_SEQUENCE_LIMIT,
                        notes=unsupported_note,
                    )
                )

            for implementation in implementations:
                longseq_output_path = (
                    result_directory
                    / f"needleman_wunsch_gpu_longseq_{implementation}_len{sequence_length}_n{num_pairs}.csv"
                )
                command = [
                    str(longseq_gpu_binary_path),
                    str(dataset_path),
                    str(longseq_output_path),
                    "--match",
                    str(MATCH_SCORE),
                    "--mismatch",
                    str(MISMATCH_PENALTY),
                    "--gap",
                    str(GAP_PENALTY),
                    "--repetitions",
                    str(repetitions),
                    "--implementation",
                    implementation,
                    "--summary-only",
                ]
                if implementation == "tiled_wavefront":
                    command.extend(["--tile-size", "16"])
                longseq_process = run_command(command, project_root, check=False)
                longseq_values = parse_key_value_output(longseq_process.stdout)

                if longseq_process.returncode != 0:
                    benchmark_rows.append(
                        build_unsupported_row(
                            "needleman_wunsch_gpu_longseq",
                            implementation,
                            "EXPERIMENTAL" if implementation == "tiled_wavefront" else "STABLE",
                            num_pairs,
                            sequence_length,
                            cpu_time_ms,
                            dp_memory_mode=longseq_values.get(
                                "DP_MEMORY_MODE",
                                "rolling_diagonal" if implementation == "rolling_diagonal" else "global",
                            ),
                            max_supported_sequence_length=parse_int(
                                longseq_values,
                                "MAX_SUPPORTED_SEQUENCE_LENGTH",
                            )
                            or 1024,
                            notes=first_error_line(longseq_process.stdout),
                        )
                    )
                    continue

                benchmark_rows.append(
                    build_gpu_row(
                        "needleman_wunsch_gpu_longseq",
                        implementation,
                        longseq_values.get("IMPLEMENTATION_STATUS", "UNKNOWN"),
                        num_pairs,
                        sequence_length,
                        cpu_time_ms,
                        longseq_values,
                        dp_memory_mode=longseq_values.get("DP_MEMORY_MODE", ""),
                        dp_memory_bytes=parse_int(longseq_values, "DP_MEMORY_BYTES"),
                        max_supported_sequence_length=parse_int(
                            longseq_values,
                            "MAX_SUPPORTED_SEQUENCE_LENGTH",
                        ),
                        notes="",
                    )
                )

    benchmark_csv_path = project_root / BENCHMARK_CSV_PATH
    benchmark_csv_path.parent.mkdir(parents=True, exist_ok=True)
    with benchmark_csv_path.open("w", encoding="utf-8", newline="") as benchmark_file:
        writer = csv.DictWriter(benchmark_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"Needleman-Wunsch long-sequence benchmark results saved to: {benchmark_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
