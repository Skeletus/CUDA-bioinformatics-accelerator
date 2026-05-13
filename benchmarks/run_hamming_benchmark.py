#!/usr/bin/env python3

import csv
import os
import subprocess
import sys
from pathlib import Path


# Reduce these lists if the Colab runtime, disk, or memory budget is limited.
SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024]
NUM_PAIRS_VALUES = [1000, 10000, 100000, 1000000]
REPETITIONS = 5
RANDOM_SEED = 42

# Existing benchmark artifacts are overwritten by default so repeated runs are reproducible.
OVERWRITE_EXISTING_FILES = True
AUTO_COMPILE_EXECUTABLES = True
STOP_ON_FAILURE = False


CSV_FIELDNAMES = [
    "algorithm",
    "num_pairs",
    "sequence_length",
    "total_bases_compared",
    "cpu_time_ms",
    "gpu_kernel_time_ms",
    "gpu_total_time_ms",
    "kernel_speedup",
    "total_speedup",
    "cpu_pairs_per_second",
    "gpu_kernel_pairs_per_second",
    "gpu_total_pairs_per_second",
    "cpu_bases_per_second",
    "gpu_kernel_bases_per_second",
    "gpu_total_bases_per_second",
    "passed",
]


def executable_path(path: Path) -> Path:
    if os.name == "nt":
        return path.with_suffix(".exe")
    return path


def relative_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.relative_to(project_root))
    except ValueError:
        return str(path)


def prepare_output_path(path: Path, project_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not OVERWRITE_EXISTING_FILES:
            raise FileExistsError(
                f"Refusing to overwrite existing file: {relative_path(path, project_root)}. "
                "Set OVERWRITE_EXISTING_FILES = True to replace generated benchmark artifacts."
            )
        print(f"Overwriting existing file: {relative_path(path, project_root)}")


def run_command(
    command: list[str],
    project_root: Path,
    *,
    check: bool,
) -> subprocess.CompletedProcess[str]:
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


def rate_per_second(count: int, time_ms: float) -> float:
    return safe_divide(float(count), time_ms / 1000.0)


def validate_result_files(cpu_output_path: Path, gpu_output_path: Path) -> bool:
    if not cpu_output_path.exists() or not gpu_output_path.exists():
        return False

    with cpu_output_path.open("r", encoding="utf-8") as cpu_file, gpu_output_path.open(
        "r", encoding="utf-8"
    ) as gpu_file:
        for cpu_line, gpu_line in zip(cpu_file, gpu_file):
            if cpu_line != gpu_line:
                return False
        return cpu_file.readline() == "" and gpu_file.readline() == ""


def compile_programs(project_root: Path) -> tuple[Path, Path]:
    cpu_binary = executable_path(project_root / "hamming_cpu")
    gpu_binary = executable_path(project_root / "hamming_gpu")

    if not AUTO_COMPILE_EXECUTABLES and cpu_binary.exists() and gpu_binary.exists():
        return cpu_binary, gpu_binary

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
        check=True,
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
        check=True,
    )

    return cpu_binary, gpu_binary


def build_benchmark_row(
    *,
    num_pairs: int,
    sequence_length: int,
    cpu_time_ms: float,
    gpu_kernel_time_ms: float,
    gpu_total_time_ms: float,
    passed: bool,
) -> dict[str, str | int | float]:
    total_bases_compared = num_pairs * sequence_length

    return {
        "algorithm": "hamming",
        "num_pairs": num_pairs,
        "sequence_length": sequence_length,
        "total_bases_compared": total_bases_compared,
        "cpu_time_ms": cpu_time_ms,
        "gpu_kernel_time_ms": gpu_kernel_time_ms,
        "gpu_total_time_ms": gpu_total_time_ms,
        "kernel_speedup": safe_divide(cpu_time_ms, gpu_kernel_time_ms),
        "total_speedup": safe_divide(cpu_time_ms, gpu_total_time_ms),
        "cpu_pairs_per_second": rate_per_second(num_pairs, cpu_time_ms),
        "gpu_kernel_pairs_per_second": rate_per_second(num_pairs, gpu_kernel_time_ms),
        "gpu_total_pairs_per_second": rate_per_second(num_pairs, gpu_total_time_ms),
        "cpu_bases_per_second": rate_per_second(total_bases_compared, cpu_time_ms),
        "gpu_kernel_bases_per_second": rate_per_second(total_bases_compared, gpu_kernel_time_ms),
        "gpu_total_bases_per_second": rate_per_second(total_bases_compared, gpu_total_time_ms),
        "passed": str(passed).lower(),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_directory = project_root / "data" / "synthetic"
    result_directory = project_root / "results" / "hamming"
    benchmark_csv_path = project_root / "benchmarks" / "hamming_benchmark_results.csv"

    dataset_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)
    prepare_output_path(benchmark_csv_path, project_root)

    print("Benchmark configuration:")
    print(f"  sequence_lengths={SEQUENCE_LENGTHS}")
    print(f"  num_pairs_values={NUM_PAIRS_VALUES}")
    print(f"  repetitions={REPETITIONS}")
    print(f"  overwrite_existing_files={OVERWRITE_EXISTING_FILES}")

    cpu_binary, gpu_binary = compile_programs(project_root)

    benchmark_rows: list[dict[str, str | int | float]] = []
    for sequence_length in SEQUENCE_LENGTHS:
        for num_pairs in NUM_PAIRS_VALUES:
            dataset_path = dataset_directory / f"synthetic_pairs_len{sequence_length}_n{num_pairs}.txt"
            cpu_output_path = result_directory / f"hamming_cpu_len{sequence_length}_n{num_pairs}.csv"
            gpu_output_path = result_directory / f"hamming_gpu_len{sequence_length}_n{num_pairs}.csv"

            prepare_output_path(dataset_path, project_root)
            prepare_output_path(cpu_output_path, project_root)
            prepare_output_path(gpu_output_path, project_root)

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

            cpu_run = run_command(
                [str(cpu_binary), str(dataset_path), str(cpu_output_path), "--repetitions", str(REPETITIONS)],
                project_root,
                check=False,
            )
            cpu_values = parse_key_value_output(cpu_run.stdout)
            cpu_time_ms = parse_float(cpu_values, "CPU_TIME_MS")

            if cpu_run.returncode != 0:
                print(f"CPU benchmark failed for length={sequence_length}, pairs={num_pairs}.")
                benchmark_rows.append(
                    build_benchmark_row(
                        num_pairs=num_pairs,
                        sequence_length=sequence_length,
                        cpu_time_ms=cpu_time_ms,
                        gpu_kernel_time_ms=0.0,
                        gpu_total_time_ms=0.0,
                        passed=False,
                    )
                )
                if STOP_ON_FAILURE:
                    break
                continue

            gpu_run = run_command(
                [str(gpu_binary), str(dataset_path), str(gpu_output_path), "--repetitions", str(REPETITIONS)],
                project_root,
                check=False,
            )
            gpu_values = parse_key_value_output(gpu_run.stdout)
            gpu_kernel_time_ms = parse_float(gpu_values, "GPU_KERNEL_TIME_MS")
            gpu_total_time_ms = parse_float(gpu_values, "GPU_TOTAL_TIME_MS")
            validation_status = gpu_values.get("VALIDATION_STATUS", "UNKNOWN")

            output_files_match = validate_result_files(cpu_output_path, gpu_output_path)
            passed = gpu_run.returncode == 0 and validation_status == "PASSED" and output_files_match

            if not passed:
                print(
                    "Validation failed or benchmark command failed for "
                    f"length={sequence_length}, pairs={num_pairs}. "
                    f"validation_status={validation_status}, output_files_match={output_files_match}"
                )
                if STOP_ON_FAILURE:
                    benchmark_rows.append(
                        build_benchmark_row(
                            num_pairs=num_pairs,
                            sequence_length=sequence_length,
                            cpu_time_ms=cpu_time_ms,
                            gpu_kernel_time_ms=gpu_kernel_time_ms,
                            gpu_total_time_ms=gpu_total_time_ms,
                            passed=False,
                        )
                    )
                    break

            benchmark_rows.append(
                build_benchmark_row(
                    num_pairs=num_pairs,
                    sequence_length=sequence_length,
                    cpu_time_ms=cpu_time_ms,
                    gpu_kernel_time_ms=gpu_kernel_time_ms,
                    gpu_total_time_ms=gpu_total_time_ms,
                    passed=passed,
                )
            )

    with benchmark_csv_path.open("w", newline="", encoding="utf-8") as benchmark_file:
        writer = csv.DictWriter(benchmark_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"Benchmark results saved to: {benchmark_csv_path}")


if __name__ == "__main__":
    main()
