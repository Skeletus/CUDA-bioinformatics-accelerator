#!/usr/bin/env python3

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from run_real_dataset_benchmark import CSV_FIELDNAMES


DEFAULT_OUTPUT_PATH = Path("benchmarks/real_dataset_pairing_modes_benchmark_results.csv")
DEFAULT_PAIRING_MODES = ["adjacent", "sampled", "all_vs_all", "mutated_queries"]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark real dataset Hamming workloads across pairing modes.")
    parser.add_argument("--window-size", type=int, default=128, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride.")
    parser.add_argument(
        "--pairing-modes",
        nargs="+",
        default=DEFAULT_PAIRING_MODES,
        choices=DEFAULT_PAIRING_MODES,
        help="Pairing modes to benchmark.",
    )
    parser.add_argument(
        "--pairs-per-fragment",
        type=int,
        default=64,
        help="Number of sampled or mutated pairs generated per source fragment.",
    )
    parser.add_argument("--mutation-rate", type=float, default=0.05, help="Mutation rate for mutated_queries mode.")
    parser.add_argument("--max-pairs", type=int, default=1_000_000, help="Maximum pairs generated per mode.")
    parser.add_argument("--repetitions", type=int, default=5, help="Number of benchmark repetitions.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible pair generation.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Output combined benchmark CSV path.")
    return parser.parse_args()


def run_command(command: list[str], project_root: Path) -> subprocess.CompletedProcess[str]:
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
    return completed_process


def read_single_row(csv_path: Path) -> dict[str, str]:
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 1:
        raise ValueError(f"Expected exactly one row in {csv_path}, found {len(rows)}.")
    return rows[0]


def write_combined_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    arguments = parse_arguments()
    project_root = Path(__file__).resolve().parents[1]
    temporary_directory = project_root / "benchmarks" / ".pairing_mode_runs"
    temporary_directory.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    failed_modes: list[str] = []

    for pairing_mode in arguments.pairing_modes:
        mode_output = temporary_directory / f"real_dataset_{pairing_mode}.csv"
        completed_process = run_command(
            [
                sys.executable,
                "benchmarks/run_real_dataset_benchmark.py",
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
                "--repetitions",
                str(arguments.repetitions),
                "--seed",
                str(arguments.seed),
                "--output",
                str(mode_output.relative_to(project_root)),
            ],
            project_root,
        )
        if mode_output.exists():
            rows.append(read_single_row(mode_output))
        if completed_process.returncode != 0:
            failed_modes.append(pairing_mode)

    if rows:
        write_combined_csv(rows, project_root / arguments.output)
        print(f"Pairing mode benchmark results saved to: {project_root / arguments.output}")
    else:
        print("Error: no benchmark rows were generated.")
        return 1

    if failed_modes:
        print(f"Error: one or more pairing mode benchmarks failed: {', '.join(failed_modes)}")
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
