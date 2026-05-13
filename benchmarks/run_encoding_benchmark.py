#!/usr/bin/env python3

import csv
import os
import subprocess
import sys
from pathlib import Path


SEQUENCE_LENGTHS = [64, 128, 256, 512, 1024]
NUM_PAIRS_VALUES = [1000, 10000, 100000, 1000000]
REPETITIONS = 5
RANDOM_SEED = 42

OVERWRITE_EXISTING_FILES = True
AUTO_COMPILE_EXECUTABLES = True
STOP_ON_FAILURE = False


CSV_FIELDNAMES = [
    "num_pairs",
    "sequence_length",
    "total_bases_compared",
    "char_gpu_kernel_time_ms",
    "char_gpu_total_time_ms",
    "encoded_file_read_time_ms",
    "encoded_input_validation_time_ms",
    "encoded_encoding_time_ms",
    "encoded_host_allocation_time_ms",
    "encoded_device_allocation_time_ms",
    "encoded_h2d_copy_time_ms",
    "encoded_gpu_kernel_time_ms",
    "encoded_d2h_copy_time_ms",
    "encoded_cpu_reference_time_ms",
    "encoded_validation_time_ms",
    "encoded_csv_write_time_ms",
    "encoded_gpu_total_time_ms",
    "encoded_end_to_end_time_ms",
    "encoding_time_ms",
    "kernel_speedup_encoded_vs_char",
    "total_speedup_encoded_vs_char",
    "end_to_end_speedup_encoded_vs_char",
    "encoded_passed",
    "char_passed",
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


def validate_result_files(first_output_path: Path, second_output_path: Path) -> bool:
    if not first_output_path.exists() or not second_output_path.exists():
        return False

    with first_output_path.open("r", encoding="utf-8") as first_file, second_output_path.open(
        "r", encoding="utf-8"
    ) as second_file:
        for first_line, second_line in zip(first_file, second_file):
            if first_line != second_line:
                return False
        return first_file.readline() == "" and second_file.readline() == ""


def compile_programs(project_root: Path) -> tuple[Path, Path]:
    char_gpu_binary = executable_path(project_root / "hamming_gpu")
    encoded_gpu_binary = executable_path(project_root / "hamming_gpu_encoded")

    if not AUTO_COMPILE_EXECUTABLES and char_gpu_binary.exists() and encoded_gpu_binary.exists():
        return char_gpu_binary, encoded_gpu_binary

    run_command(
        [
            "nvcc",
            "src/hamming_gpu.cu",
            "-O3",
            "-std=c++17",
            "-I",
            "src/common",
            "-o",
            str(char_gpu_binary),
        ],
        project_root,
        check=True,
    )
    run_command(
        [
            "nvcc",
            "src/hamming_gpu_encoded.cu",
            "-O3",
            "-std=c++17",
            "-I",
            "src/common",
            "-o",
            str(encoded_gpu_binary),
        ],
        project_root,
        check=True,
    )

    return char_gpu_binary, encoded_gpu_binary


def build_benchmark_row(
    *,
    num_pairs: int,
    sequence_length: int,
    char_gpu_kernel_time_ms: float,
    char_gpu_total_time_ms: float,
    encoded_file_read_time_ms: float,
    encoded_input_validation_time_ms: float,
    encoded_encoding_time_ms: float,
    encoded_host_allocation_time_ms: float,
    encoded_device_allocation_time_ms: float,
    encoded_h2d_copy_time_ms: float,
    encoded_gpu_kernel_time_ms: float,
    encoded_d2h_copy_time_ms: float,
    encoded_cpu_reference_time_ms: float,
    encoded_validation_time_ms: float,
    encoded_csv_write_time_ms: float,
    encoded_gpu_total_time_ms: float,
    encoded_end_to_end_time_ms: float,
    encoding_time_ms: float,
    char_passed: bool,
    encoded_passed: bool,
) -> dict[str, str | int | float]:
    total_bases_compared = num_pairs * sequence_length

    return {
        "num_pairs": num_pairs,
        "sequence_length": sequence_length,
        "total_bases_compared": total_bases_compared,
        "char_gpu_kernel_time_ms": char_gpu_kernel_time_ms,
        "char_gpu_total_time_ms": char_gpu_total_time_ms,
        "encoded_file_read_time_ms": encoded_file_read_time_ms,
        "encoded_input_validation_time_ms": encoded_input_validation_time_ms,
        "encoded_encoding_time_ms": encoded_encoding_time_ms,
        "encoded_host_allocation_time_ms": encoded_host_allocation_time_ms,
        "encoded_device_allocation_time_ms": encoded_device_allocation_time_ms,
        "encoded_h2d_copy_time_ms": encoded_h2d_copy_time_ms,
        "encoded_gpu_kernel_time_ms": encoded_gpu_kernel_time_ms,
        "encoded_d2h_copy_time_ms": encoded_d2h_copy_time_ms,
        "encoded_cpu_reference_time_ms": encoded_cpu_reference_time_ms,
        "encoded_validation_time_ms": encoded_validation_time_ms,
        "encoded_csv_write_time_ms": encoded_csv_write_time_ms,
        "encoded_gpu_total_time_ms": encoded_gpu_total_time_ms,
        "encoded_end_to_end_time_ms": encoded_end_to_end_time_ms,
        "encoding_time_ms": encoding_time_ms,
        "kernel_speedup_encoded_vs_char": safe_divide(char_gpu_kernel_time_ms, encoded_gpu_kernel_time_ms),
        "total_speedup_encoded_vs_char": safe_divide(char_gpu_total_time_ms, encoded_gpu_total_time_ms),
        "end_to_end_speedup_encoded_vs_char": safe_divide(char_gpu_total_time_ms, encoded_end_to_end_time_ms),
        "encoded_passed": str(encoded_passed).lower(),
        "char_passed": str(char_passed).lower(),
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    dataset_directory = project_root / "data" / "synthetic"
    result_directory = project_root / "results" / "hamming"
    benchmark_csv_path = project_root / "benchmarks" / "dna_encoding_benchmark_results.csv"

    dataset_directory.mkdir(parents=True, exist_ok=True)
    result_directory.mkdir(parents=True, exist_ok=True)
    prepare_output_path(benchmark_csv_path, project_root)

    print("DNA encoding benchmark configuration:")
    print(f"  sequence_lengths={SEQUENCE_LENGTHS}")
    print(f"  num_pairs_values={NUM_PAIRS_VALUES}")
    print(f"  repetitions={REPETITIONS}")
    print(f"  overwrite_existing_files={OVERWRITE_EXISTING_FILES}")

    char_gpu_binary, encoded_gpu_binary = compile_programs(project_root)

    benchmark_rows: list[dict[str, str | int | float]] = []
    for sequence_length in SEQUENCE_LENGTHS:
        for num_pairs in NUM_PAIRS_VALUES:
            dataset_path = dataset_directory / f"synthetic_pairs_len{sequence_length}_n{num_pairs}.txt"
            char_output_path = result_directory / f"hamming_gpu_char_len{sequence_length}_n{num_pairs}.csv"
            encoded_output_path = result_directory / f"hamming_gpu_encoded_len{sequence_length}_n{num_pairs}.csv"

            prepare_output_path(dataset_path, project_root)
            prepare_output_path(char_output_path, project_root)
            prepare_output_path(encoded_output_path, project_root)

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

            char_run = run_command(
                [str(char_gpu_binary), str(dataset_path), str(char_output_path), "--repetitions", str(REPETITIONS)],
                project_root,
                check=False,
            )
            char_values = parse_key_value_output(char_run.stdout)
            char_gpu_kernel_time_ms = parse_float(char_values, "GPU_KERNEL_TIME_MS")
            char_gpu_total_time_ms = parse_float(char_values, "GPU_TOTAL_TIME_MS")
            char_validation_status = char_values.get("VALIDATION_STATUS", "UNKNOWN")
            char_passed = char_run.returncode == 0 and char_validation_status == "PASSED"

            encoded_run = run_command(
                [
                    str(encoded_gpu_binary),
                    str(dataset_path),
                    str(encoded_output_path),
                    "--repetitions",
                    str(REPETITIONS),
                ],
                project_root,
                check=False,
            )
            encoded_values = parse_key_value_output(encoded_run.stdout)
            encoded_file_read_time_ms = parse_float(encoded_values, "FILE_READ_TIME_MS")
            encoded_input_validation_time_ms = parse_float(encoded_values, "INPUT_VALIDATION_TIME_MS")
            encoded_encoding_time_ms = parse_float(encoded_values, "ENCODING_TIME_MS")
            encoded_host_allocation_time_ms = parse_float(encoded_values, "HOST_ALLOCATION_TIME_MS")
            encoded_device_allocation_time_ms = parse_float(encoded_values, "DEVICE_ALLOCATION_TIME_MS")
            encoded_h2d_copy_time_ms = parse_float(encoded_values, "H2D_COPY_TIME_MS")
            encoded_gpu_kernel_time_ms = parse_float(encoded_values, "GPU_KERNEL_TIME_MS")
            encoded_d2h_copy_time_ms = parse_float(encoded_values, "D2H_COPY_TIME_MS")
            encoded_cpu_reference_time_ms = parse_float(encoded_values, "CPU_REFERENCE_TIME_MS")
            encoded_validation_time_ms = parse_float(encoded_values, "VALIDATION_TIME_MS")
            encoded_csv_write_time_ms = parse_float(encoded_values, "CSV_WRITE_TIME_MS")
            encoded_gpu_total_time_ms = parse_float(encoded_values, "GPU_TOTAL_TIME_MS")
            encoded_end_to_end_time_ms = parse_float(encoded_values, "END_TO_END_TIME_MS")
            encoding_time_ms = encoded_encoding_time_ms
            encoded_validation_status = encoded_values.get("VALIDATION_STATUS", "UNKNOWN")

            output_files_match = validate_result_files(char_output_path, encoded_output_path)
            encoded_passed = (
                encoded_run.returncode == 0
                and encoded_validation_status == "PASSED"
                and output_files_match
            )

            if not char_passed or not encoded_passed:
                print(
                    "Validation failed or benchmark command failed for "
                    f"length={sequence_length}, pairs={num_pairs}. "
                    f"char_status={char_validation_status}, "
                    f"encoded_status={encoded_validation_status}, "
                    f"output_files_match={output_files_match}"
                )
                if STOP_ON_FAILURE:
                    benchmark_rows.append(
                        build_benchmark_row(
                            num_pairs=num_pairs,
                            sequence_length=sequence_length,
                            char_gpu_kernel_time_ms=char_gpu_kernel_time_ms,
                            char_gpu_total_time_ms=char_gpu_total_time_ms,
                            encoded_file_read_time_ms=encoded_file_read_time_ms,
                            encoded_input_validation_time_ms=encoded_input_validation_time_ms,
                            encoded_encoding_time_ms=encoded_encoding_time_ms,
                            encoded_host_allocation_time_ms=encoded_host_allocation_time_ms,
                            encoded_device_allocation_time_ms=encoded_device_allocation_time_ms,
                            encoded_h2d_copy_time_ms=encoded_h2d_copy_time_ms,
                            encoded_gpu_kernel_time_ms=encoded_gpu_kernel_time_ms,
                            encoded_d2h_copy_time_ms=encoded_d2h_copy_time_ms,
                            encoded_cpu_reference_time_ms=encoded_cpu_reference_time_ms,
                            encoded_validation_time_ms=encoded_validation_time_ms,
                            encoded_csv_write_time_ms=encoded_csv_write_time_ms,
                            encoded_gpu_total_time_ms=encoded_gpu_total_time_ms,
                            encoded_end_to_end_time_ms=encoded_end_to_end_time_ms,
                            encoding_time_ms=encoding_time_ms,
                            char_passed=char_passed,
                            encoded_passed=encoded_passed,
                        )
                    )
                    break

            benchmark_rows.append(
                build_benchmark_row(
                    num_pairs=num_pairs,
                    sequence_length=sequence_length,
                    char_gpu_kernel_time_ms=char_gpu_kernel_time_ms,
                    char_gpu_total_time_ms=char_gpu_total_time_ms,
                    encoded_file_read_time_ms=encoded_file_read_time_ms,
                    encoded_input_validation_time_ms=encoded_input_validation_time_ms,
                    encoded_encoding_time_ms=encoded_encoding_time_ms,
                    encoded_host_allocation_time_ms=encoded_host_allocation_time_ms,
                    encoded_device_allocation_time_ms=encoded_device_allocation_time_ms,
                    encoded_h2d_copy_time_ms=encoded_h2d_copy_time_ms,
                    encoded_gpu_kernel_time_ms=encoded_gpu_kernel_time_ms,
                    encoded_d2h_copy_time_ms=encoded_d2h_copy_time_ms,
                    encoded_cpu_reference_time_ms=encoded_cpu_reference_time_ms,
                    encoded_validation_time_ms=encoded_validation_time_ms,
                    encoded_csv_write_time_ms=encoded_csv_write_time_ms,
                    encoded_gpu_total_time_ms=encoded_gpu_total_time_ms,
                    encoded_end_to_end_time_ms=encoded_end_to_end_time_ms,
                    encoding_time_ms=encoding_time_ms,
                    char_passed=char_passed,
                    encoded_passed=encoded_passed,
                )
            )

    with benchmark_csv_path.open("w", newline="", encoding="utf-8") as benchmark_file:
        writer = csv.DictWriter(benchmark_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(benchmark_rows)

    print(f"DNA encoding benchmark results saved to: {benchmark_csv_path}")


if __name__ == "__main__":
    main()
