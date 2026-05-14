#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DATASET_NAME = "sars-cov-2"
DEFAULT_FASTA_PATH = Path("data/raw/sars_cov_2_NC_045512_2.fasta")
DEFAULT_PAIRING_MODES = ["adjacent", "sampled", "all_vs_all", "mutated_queries"]
DEFAULT_OUTPUT_PATH = Path("benchmarks/encoded_optimized_benchmark_results.csv")


CSV_FIELDNAMES = [
    "pairing_mode",
    "number_of_pairs",
    "sequence_length",
    "total_bases_compared",
    "baseline_encoded_kernel_time_ms",
    "baseline_encoded_gpu_total_time_ms",
    "baseline_encoded_end_to_end_time_ms",
    "optimized_encoded_kernel_time_ms",
    "optimized_h2d_copy_time_ms",
    "optimized_d2h_copy_time_ms",
    "optimized_gpu_pipeline_time_ms",
    "optimized_setup_time_ms",
    "optimized_end_to_end_time_ms",
    "optimized_encoding_time_ms",
    "optimized_encoded_cache_time_ms",
    "optimized_flat_buffer_build_time_ms",
    "optimized_validation_time_ms",
    "optimized_csv_write_time_ms",
    "unique_sequence_count",
    "cache_hit_count",
    "cache_miss_count",
    "cache_hit_rate",
    "kernel_speedup_optimized_vs_baseline",
    "gpu_pipeline_speedup_optimized_vs_baseline",
    "end_to_end_speedup_optimized_vs_baseline",
    "indexed_mode",
    "unique_fragment_bytes",
    "pair_index_bytes",
    "flat_pair_bytes_avoided",
    "validation_status",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and optimized encoded CUDA Hamming pipelines.")
    parser.add_argument("--window-size", type=int, default=128, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride.")
    parser.add_argument(
        "--pairing-modes",
        nargs="+",
        default=DEFAULT_PAIRING_MODES,
        choices=DEFAULT_PAIRING_MODES,
        help="Pairing modes to benchmark.",
    )
    parser.add_argument("--pairs-per-fragment", type=int, default=64, help="Pairs per fragment for sampled workloads.")
    parser.add_argument("--max-pairs", type=int, default=1_000_000, help="Maximum generated pairs per mode.")
    parser.add_argument("--mutation-rate", type=float, default=0.05, help="Mutation rate for mutated_queries mode.")
    parser.add_argument("--repetitions", type=int, default=5, help="Measured repetitions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible pair generation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output benchmark CSV path.")
    return parser.parse_args()


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


def compile_programs(project_root: Path) -> tuple[Path, Path]:
    baseline_binary = executable_path(project_root / "hamming_gpu_encoded")
    optimized_binary = executable_path(project_root / "hamming_gpu_encoded_optimized")

    run_command(
        [
            "nvcc",
            "src/hamming_gpu_encoded.cu",
            "-O3",
            "-std=c++17",
            "-I",
            "src/common",
            "-o",
            str(baseline_binary),
        ],
        project_root,
        check=True,
    )
    run_command(
        [
            "nvcc",
            "src/hamming_gpu_encoded_optimized.cu",
            "-O3",
            "-std=c++17",
            "-I",
            "src/common",
            "-o",
            str(optimized_binary),
        ],
        project_root,
        check=True,
    )
    return baseline_binary, optimized_binary


def ensure_dataset(project_root: Path) -> None:
    if (project_root / DEFAULT_FASTA_PATH).exists():
        return
    run_command(
        [
            sys.executable,
            "scripts/download_datasets.py",
            "--dataset",
            DATASET_NAME,
            "--output",
            str(DEFAULT_FASTA_PATH),
        ],
        project_root,
        check=True,
    )


def generate_pairs(arguments: argparse.Namespace, project_root: Path, pairing_mode: str) -> Path:
    fragment_csv_path = project_root / f"data/processed/sars_cov_2_fragments_{arguments.window_size}.csv"
    fragment_txt_path = project_root / f"data/sars_cov_2_fragments_{arguments.window_size}.txt"
    pair_path = (
        project_root
        / f"data/processed/sars_cov_2_pairs_{arguments.window_size}_stride_{arguments.stride}_{pairing_mode}.txt"
    )

    run_command(
        [
            sys.executable,
            "scripts/fragment_fasta.py",
            "--input",
            str(DEFAULT_FASTA_PATH),
            "--output-csv",
            str(fragment_csv_path.relative_to(project_root)),
            "--output-txt",
            str(fragment_txt_path.relative_to(project_root)),
            "--output-pairs",
            str(pair_path.relative_to(project_root)),
            "--window-size",
            str(arguments.window_size),
            "--stride",
            str(arguments.stride),
            "--pairing-mode",
            pairing_mode,
            "--pairs-per-fragment",
            str(arguments.pairs_per_fragment),
            "--max-pairs",
            str(arguments.max_pairs),
            "--mutation-rate",
            str(arguments.mutation_rate),
            "--seed",
            str(arguments.seed),
            "--skip-ambiguous",
        ],
        project_root,
        check=True,
    )
    return pair_path


def build_row(
    pairing_mode: str,
    baseline_values: dict[str, str],
    optimized_values: dict[str, str],
    baseline_returncode: int,
    optimized_returncode: int,
) -> dict[str, str | int | float]:
    baseline_kernel_time_ms = parse_float(baseline_values, "GPU_KERNEL_TIME_MS")
    baseline_gpu_total_time_ms = parse_float(baseline_values, "GPU_TOTAL_TIME_MS")
    baseline_end_to_end_time_ms = parse_float(baseline_values, "END_TO_END_TIME_MS")
    optimized_kernel_time_ms = parse_float(optimized_values, "GPU_KERNEL_TIME_MS")
    optimized_gpu_pipeline_time_ms = parse_float(optimized_values, "GPU_PIPELINE_TIME_MS")
    optimized_end_to_end_time_ms = parse_float(optimized_values, "END_TO_END_TIME_MS")

    baseline_status = baseline_values.get("VALIDATION_STATUS", "UNKNOWN")
    optimized_status = optimized_values.get("VALIDATION_STATUS", "UNKNOWN")
    validation_status = (
        "PASSED"
        if baseline_returncode == 0
        and optimized_returncode == 0
        and baseline_status == "PASSED"
        and optimized_status == "PASSED"
        else "FAILED"
    )

    return {
        "pairing_mode": pairing_mode,
        "number_of_pairs": parse_int(optimized_values, "NUMBER_OF_PAIRS"),
        "sequence_length": parse_int(optimized_values, "SEQUENCE_LENGTH"),
        "total_bases_compared": parse_int(optimized_values, "TOTAL_BASES_COMPARED"),
        "baseline_encoded_kernel_time_ms": baseline_kernel_time_ms,
        "baseline_encoded_gpu_total_time_ms": baseline_gpu_total_time_ms,
        "baseline_encoded_end_to_end_time_ms": baseline_end_to_end_time_ms,
        "optimized_encoded_kernel_time_ms": optimized_kernel_time_ms,
        "optimized_h2d_copy_time_ms": parse_float(optimized_values, "H2D_COPY_TIME_MS"),
        "optimized_d2h_copy_time_ms": parse_float(optimized_values, "D2H_COPY_TIME_MS"),
        "optimized_gpu_pipeline_time_ms": optimized_gpu_pipeline_time_ms,
        "optimized_setup_time_ms": parse_float(optimized_values, "SETUP_TIME_MS"),
        "optimized_end_to_end_time_ms": optimized_end_to_end_time_ms,
        "optimized_encoding_time_ms": parse_float(optimized_values, "ENCODING_TIME_MS"),
        "optimized_encoded_cache_time_ms": parse_float(optimized_values, "ENCODED_CACHE_TIME_MS"),
        "optimized_flat_buffer_build_time_ms": parse_float(optimized_values, "FLAT_BUFFER_BUILD_TIME_MS"),
        "optimized_validation_time_ms": parse_float(optimized_values, "VALIDATION_TIME_MS"),
        "optimized_csv_write_time_ms": parse_float(optimized_values, "CSV_WRITE_TIME_MS"),
        "unique_sequence_count": parse_int(optimized_values, "UNIQUE_SEQUENCE_COUNT"),
        "cache_hit_count": parse_int(optimized_values, "CACHE_HIT_COUNT"),
        "cache_miss_count": parse_int(optimized_values, "CACHE_MISS_COUNT"),
        "cache_hit_rate": parse_float(optimized_values, "CACHE_HIT_RATE"),
        "kernel_speedup_optimized_vs_baseline": safe_divide(baseline_kernel_time_ms, optimized_kernel_time_ms),
        "gpu_pipeline_speedup_optimized_vs_baseline": safe_divide(
            baseline_gpu_total_time_ms,
            optimized_gpu_pipeline_time_ms,
        ),
        "end_to_end_speedup_optimized_vs_baseline": safe_divide(
            baseline_end_to_end_time_ms,
            optimized_end_to_end_time_ms,
        ),
        "indexed_mode": optimized_values.get("INDEXED_MODE", "false"),
        "unique_fragment_bytes": parse_int(optimized_values, "UNIQUE_FRAGMENT_BYTES"),
        "pair_index_bytes": parse_int(optimized_values, "PAIR_INDEX_BYTES"),
        "flat_pair_bytes_avoided": parse_int(optimized_values, "FLAT_PAIR_BYTES_AVOIDED"),
        "validation_status": validation_status,
    }


def main() -> int:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    result_directory = project_root / "results" / "hamming"
    result_directory.mkdir(parents=True, exist_ok=True)

    ensure_dataset(project_root)
    baseline_binary, optimized_binary = compile_programs(project_root)

    rows: list[dict[str, str | int | float]] = []
    failed_modes: list[str] = []

    for pairing_mode in arguments.pairing_modes:
        pair_path = generate_pairs(arguments, project_root, pairing_mode)
        baseline_output_path = result_directory / f"hamming_gpu_encoded_baseline_{pairing_mode}.csv"
        optimized_output_path = result_directory / f"hamming_gpu_encoded_optimized_{pairing_mode}.csv"

        baseline_run = run_command(
            [
                str(baseline_binary),
                str(pair_path),
                str(baseline_output_path),
                "--repetitions",
                str(arguments.repetitions),
            ],
            project_root,
            check=False,
        )
        optimized_run = run_command(
            [
                str(optimized_binary),
                str(pair_path),
                str(optimized_output_path),
                "--repetitions",
                str(arguments.repetitions),
                "--summary-only",
            ],
            project_root,
            check=False,
        )

        baseline_values = parse_key_value_output(baseline_run.stdout)
        optimized_values = parse_key_value_output(optimized_run.stdout)
        rows.append(
            build_row(
                pairing_mode,
                baseline_values,
                optimized_values,
                baseline_run.returncode,
                optimized_run.returncode,
            )
        )

        if baseline_run.returncode != 0 or optimized_run.returncode != 0:
            failed_modes.append(pairing_mode)

    output_path = project_root / arguments.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Encoded optimized benchmark results saved to: {output_path}")

    if failed_modes:
        print(f"Error: optimized encoded benchmark failed for modes: {', '.join(failed_modes)}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
