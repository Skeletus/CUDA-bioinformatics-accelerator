#!/usr/bin/env python3

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


DATASET_NAME = "sars-cov-2"
ACCESSION = "NC_045512.2"
DEFAULT_FASTA_PATH = Path("data/raw/sars_cov_2_NC_045512_2.fasta")
DEFAULT_FRAGMENT_CSV_PATH = Path("data/processed/sars_cov_2_fragments_128.csv")
DEFAULT_FRAGMENT_TXT_PATH = Path("data/sars_cov_2_fragments_128.txt")
DEFAULT_BENCHMARK_CSV_PATH = Path("benchmarks/real_dataset_hamming_benchmark_results.csv")
PAIRING_MODES = ("adjacent", "all_vs_all", "sampled", "mutated_queries")


CSV_FIELDNAMES = [
    "dataset_name",
    "accession",
    "genome_length",
    "window_size",
    "stride",
    "pairing_mode",
    "pairs_per_fragment",
    "max_pairs",
    "mutation_rate",
    "seed",
    "number_of_fragments",
    "number_of_pairs",
    "sequence_length",
    "total_bases_compared",
    "cpu_time_ms",
    "char_gpu_kernel_time_ms",
    "char_gpu_total_time_ms",
    "encoded_gpu_kernel_time_ms",
    "encoded_gpu_total_time_ms",
    "char_kernel_speedup",
    "char_total_speedup",
    "encoded_kernel_speedup",
    "encoded_total_speedup",
    "char_passed",
    "encoded_passed",
    "pair_generation_truncated",
    "total_bases_mutated",
    "observed_mutation_rate",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Hamming Distance on real SARS-CoV-2 fragments.")
    parser.add_argument("--window-size", type=int, default=128, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride.")
    parser.add_argument(
        "--pairing-mode",
        choices=PAIRING_MODES,
        default="adjacent",
        help="Real dataset pair generation mode.",
    )
    parser.add_argument(
        "--pairs-per-fragment",
        type=int,
        default=64,
        help="Number of sampled or mutated pairs generated per source fragment.",
    )
    parser.add_argument("--max-pairs", type=int, default=1_000_000, help="Maximum number of generated pairs.")
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.05,
        help="Per-base mutation probability for mutated_queries mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible pair generation.")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of benchmark repetitions.")
    parser.add_argument("--output", type=Path, default=DEFAULT_BENCHMARK_CSV_PATH, help="Output benchmark CSV path.")
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


def files_match(first_path: Path, second_path: Path) -> bool:
    if not first_path.exists() or not second_path.exists():
        return False
    with first_path.open("r", encoding="utf-8") as first_file, second_path.open(
        "r", encoding="utf-8"
    ) as second_file:
        for first_line, second_line in zip(first_file, second_file):
            if first_line != second_line:
                return False
        return first_file.readline() == "" and second_file.readline() == ""


def compile_programs(project_root: Path) -> tuple[Path, Path, Path | None]:
    cpu_binary = executable_path(project_root / "hamming_cpu")
    char_gpu_binary = executable_path(project_root / "hamming_gpu")
    encoded_gpu_binary = executable_path(project_root / "hamming_gpu_encoded")

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
            str(char_gpu_binary),
        ],
        project_root,
        check=True,
    )

    if (project_root / "src/hamming_gpu_encoded.cu").exists():
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
        return cpu_binary, char_gpu_binary, encoded_gpu_binary

    return cpu_binary, char_gpu_binary, None


def write_benchmark_csv(row: dict[str, str | int | float], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerow(row)


def build_row(
    *,
    genome_length: int,
    window_size: int,
    stride: int,
    pairing_mode: str,
    pairs_per_fragment: int,
    max_pairs: int,
    mutation_rate: float,
    seed: int,
    number_of_fragments: int,
    number_of_pairs: int,
    sequence_length: int,
    cpu_time_ms: float,
    char_gpu_kernel_time_ms: float,
    char_gpu_total_time_ms: float,
    encoded_gpu_kernel_time_ms: float,
    encoded_gpu_total_time_ms: float,
    char_passed: bool,
    encoded_passed: bool,
    pair_generation_truncated: bool,
    total_bases_mutated: int,
    observed_mutation_rate: float,
) -> dict[str, str | int | float]:
    total_bases_compared = number_of_pairs * sequence_length
    return {
        "dataset_name": DATASET_NAME,
        "accession": ACCESSION,
        "genome_length": genome_length,
        "window_size": window_size,
        "stride": stride,
        "pairing_mode": pairing_mode,
        "pairs_per_fragment": pairs_per_fragment,
        "max_pairs": max_pairs,
        "mutation_rate": mutation_rate,
        "seed": seed,
        "number_of_fragments": number_of_fragments,
        "number_of_pairs": number_of_pairs,
        "sequence_length": sequence_length,
        "total_bases_compared": total_bases_compared,
        "cpu_time_ms": cpu_time_ms,
        "char_gpu_kernel_time_ms": char_gpu_kernel_time_ms,
        "char_gpu_total_time_ms": char_gpu_total_time_ms,
        "encoded_gpu_kernel_time_ms": encoded_gpu_kernel_time_ms,
        "encoded_gpu_total_time_ms": encoded_gpu_total_time_ms,
        "char_kernel_speedup": safe_divide(cpu_time_ms, char_gpu_kernel_time_ms),
        "char_total_speedup": safe_divide(cpu_time_ms, char_gpu_total_time_ms),
        "encoded_kernel_speedup": safe_divide(cpu_time_ms, encoded_gpu_kernel_time_ms),
        "encoded_total_speedup": safe_divide(cpu_time_ms, encoded_gpu_total_time_ms),
        "char_passed": str(char_passed).lower(),
        "encoded_passed": str(encoded_passed).lower(),
        "pair_generation_truncated": str(pair_generation_truncated).lower(),
        "total_bases_mutated": total_bases_mutated,
        "observed_mutation_rate": observed_mutation_rate,
    }


def main() -> int:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    result_directory = project_root / "results" / "hamming"
    result_directory.mkdir(parents=True, exist_ok=True)

    fragment_csv_path = project_root / f"data/processed/sars_cov_2_fragments_{arguments.window_size}.csv"
    fragment_txt_path = project_root / f"data/sars_cov_2_fragments_{arguments.window_size}.txt"
    pair_path = (
        project_root
        / f"data/processed/sars_cov_2_pairs_{arguments.window_size}_stride_{arguments.stride}_{arguments.pairing_mode}.txt"
    )
    cpu_output_path = result_directory / f"real_dataset_hamming_cpu_{arguments.pairing_mode}.csv"
    char_output_path = result_directory / f"real_dataset_hamming_gpu_char_{arguments.pairing_mode}.csv"
    encoded_output_path = result_directory / f"real_dataset_hamming_gpu_encoded_{arguments.pairing_mode}.csv"

    if not (project_root / DEFAULT_FASTA_PATH).exists():
        download_run = run_command(
            [
                sys.executable,
                "scripts/download_datasets.py",
                "--dataset",
                DATASET_NAME,
                "--output",
                str(DEFAULT_FASTA_PATH),
            ],
            project_root,
            check=False,
        )
        if download_run.returncode != 0:
            print("Error: failed to download the SARS-CoV-2 FASTA file.")
            return 1

    fragment_run = run_command(
        [
            sys.executable,
            "scripts/fragment_fasta.py",
            "--input",
            str(DEFAULT_FASTA_PATH),
            "--output-csv",
            str(fragment_csv_path),
            "--output-txt",
            str(fragment_txt_path),
            "--output-pairs",
            str(pair_path),
            "--window-size",
            str(arguments.window_size),
            "--stride",
            str(arguments.stride),
            "--pairing-mode",
            arguments.pairing_mode,
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
    fragment_values = parse_key_value_output(fragment_run.stdout)
    genome_length = parse_int(fragment_values, "GENOME_LENGTH")
    number_of_fragments = parse_int(fragment_values, "NUMBER_OF_FRAGMENTS")
    number_of_pairs = parse_int(fragment_values, "NUMBER_OF_PAIRS")
    pair_generation_truncated = fragment_values.get("PAIR_GENERATION_TRUNCATED", "false").lower() == "true"
    total_bases_mutated = parse_int(fragment_values, "TOTAL_BASES_MUTATED")
    observed_mutation_rate = parse_float(fragment_values, "OBSERVED_MUTATION_RATE")

    cpu_binary, char_gpu_binary, encoded_gpu_binary = compile_programs(project_root)

    cpu_run = run_command(
        [str(cpu_binary), str(pair_path), str(cpu_output_path), "--repetitions", str(arguments.repetitions)],
        project_root,
        check=False,
    )
    cpu_values = parse_key_value_output(cpu_run.stdout)
    cpu_time_ms = parse_float(cpu_values, "CPU_TIME_MS")
    sequence_length = parse_int(cpu_values, "SEQUENCE_LENGTH") or arguments.window_size

    if cpu_run.returncode != 0:
        print("Error: CPU benchmark failed, so GPU correctness cannot be validated.")
        row = build_row(
            genome_length=genome_length,
            window_size=arguments.window_size,
            stride=arguments.stride,
            pairing_mode=arguments.pairing_mode,
            pairs_per_fragment=arguments.pairs_per_fragment,
            max_pairs=arguments.max_pairs,
            mutation_rate=arguments.mutation_rate,
            seed=arguments.seed,
            number_of_fragments=number_of_fragments,
            number_of_pairs=number_of_pairs,
            sequence_length=sequence_length,
            cpu_time_ms=cpu_time_ms,
            char_gpu_kernel_time_ms=0.0,
            char_gpu_total_time_ms=0.0,
            encoded_gpu_kernel_time_ms=0.0,
            encoded_gpu_total_time_ms=0.0,
            char_passed=False,
            encoded_passed=False,
            pair_generation_truncated=pair_generation_truncated,
            total_bases_mutated=total_bases_mutated,
            observed_mutation_rate=observed_mutation_rate,
        )
        write_benchmark_csv(row, project_root / arguments.output)
        return 1

    char_run = run_command(
        [str(char_gpu_binary), str(pair_path), str(char_output_path), "--repetitions", str(arguments.repetitions)],
        project_root,
        check=False,
    )
    char_values = parse_key_value_output(char_run.stdout)
    char_gpu_kernel_time_ms = parse_float(char_values, "GPU_KERNEL_TIME_MS")
    char_gpu_total_time_ms = parse_float(char_values, "GPU_TOTAL_TIME_MS")
    char_validation_status = char_values.get("VALIDATION_STATUS", "UNKNOWN")
    char_passed = (
        char_run.returncode == 0
        and char_validation_status == "PASSED"
        and files_match(cpu_output_path, char_output_path)
    )

    encoded_gpu_kernel_time_ms = 0.0
    encoded_gpu_total_time_ms = 0.0
    encoded_passed = False
    if encoded_gpu_binary is not None:
        encoded_run = run_command(
            [
                str(encoded_gpu_binary),
                str(pair_path),
                str(encoded_output_path),
                "--repetitions",
                str(arguments.repetitions),
            ],
            project_root,
            check=False,
        )
        encoded_values = parse_key_value_output(encoded_run.stdout)
        encoded_gpu_kernel_time_ms = parse_float(encoded_values, "GPU_KERNEL_TIME_MS")
        encoded_gpu_total_time_ms = parse_float(encoded_values, "GPU_TOTAL_TIME_MS")
        encoded_validation_status = encoded_values.get("VALIDATION_STATUS", "UNKNOWN")
        encoded_passed = (
            encoded_run.returncode == 0
            and encoded_validation_status == "PASSED"
            and files_match(cpu_output_path, encoded_output_path)
        )

    if not char_passed:
        print("Error: char-based GPU results did not match CPU reference results.")
    if not encoded_passed:
        print("Error: encoded GPU results did not match CPU reference results.")

    row = build_row(
        genome_length=genome_length,
        window_size=arguments.window_size,
        stride=arguments.stride,
        pairing_mode=arguments.pairing_mode,
        pairs_per_fragment=arguments.pairs_per_fragment,
        max_pairs=arguments.max_pairs,
        mutation_rate=arguments.mutation_rate,
        seed=arguments.seed,
        number_of_fragments=number_of_fragments,
        number_of_pairs=number_of_pairs,
        sequence_length=sequence_length,
        cpu_time_ms=cpu_time_ms,
        char_gpu_kernel_time_ms=char_gpu_kernel_time_ms,
        char_gpu_total_time_ms=char_gpu_total_time_ms,
        encoded_gpu_kernel_time_ms=encoded_gpu_kernel_time_ms,
        encoded_gpu_total_time_ms=encoded_gpu_total_time_ms,
        char_passed=char_passed,
        encoded_passed=encoded_passed,
        pair_generation_truncated=pair_generation_truncated,
        total_bases_mutated=total_bases_mutated,
        observed_mutation_rate=observed_mutation_rate,
    )
    write_benchmark_csv(row, project_root / arguments.output)
    print(f"Benchmark results saved to: {project_root / arguments.output}")
    return 0 if char_passed and encoded_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
