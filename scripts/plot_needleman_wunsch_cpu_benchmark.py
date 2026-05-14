#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/needleman_wunsch_cpu_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/needleman_wunsch_cpu")


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0.0") or 0.0)
    except ValueError:
        return 0.0


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Needleman-Wunsch benchmark CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Needleman-Wunsch benchmark CSV has no data rows: {csv_path}")
    return rows


def workload_labels(rows: list[dict[str, str]]) -> list[str]:
    return [f"len {row.get('sequence_length')}\npairs {row.get('num_pairs')}" for row in rows]


def save_bar_chart(
    rows: list[dict[str, str]],
    title: str,
    y_label: str,
    column_name: str,
    output_path: Path,
    *,
    use_log_scale: bool = False,
) -> None:
    labels = workload_labels(rows)
    x_values = list(range(len(rows)))
    values = [parse_float(row, column_name) for row in rows]

    plt.figure(figsize=(14, 7))
    plt.bar(x_values, values)
    plt.title(title)
    plt.xlabel("Workload")
    plt.ylabel(y_label)
    plt.xticks(x_values, labels, rotation=45, ha="right")
    if use_log_scale and all(value > 0.0 for value in values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_complexity_growth_chart(rows: list[dict[str, str]], output_path: Path) -> None:
    x_values = [parse_float(row, "total_cells_computed") for row in rows]
    y_values = [parse_float(row, "cpu_time_ms") for row in rows]

    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values)
    plt.title("Needleman-Wunsch CPU Complexity Growth")
    plt.xlabel("Total DP cells computed")
    plt.ylabel("CPU time (ms)")
    if all(value > 0.0 for value in x_values) and all(value > 0.0 for value in y_values):
        plt.xscale("log")
        plt.yscale("log")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows(BENCHMARK_CSV_PATH)
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_bar_chart(
        rows,
        "Needleman-Wunsch CPU Time by Workload",
        "CPU time (ms)",
        "cpu_time_ms",
        CHART_OUTPUT_DIRECTORY / "nw_cpu_time_by_workload.png",
        use_log_scale=True,
    )
    save_bar_chart(
        rows,
        "Needleman-Wunsch CPU Cells per Second",
        "DP cells per second",
        "cells_per_second",
        CHART_OUTPUT_DIRECTORY / "nw_cells_per_second.png",
        use_log_scale=True,
    )
    save_bar_chart(
        rows,
        "Needleman-Wunsch Average Time per Pair",
        "Average time per pair (ms)",
        "average_time_per_pair_ms",
        CHART_OUTPUT_DIRECTORY / "nw_average_time_per_pair.png",
        use_log_scale=True,
    )
    save_complexity_growth_chart(
        rows,
        CHART_OUTPUT_DIRECTORY / "nw_complexity_growth.png",
    )

    print(f"Needleman-Wunsch CPU benchmark charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
