#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/needleman_wunsch_gpu_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/needleman_wunsch_gpu")


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0.0") or 0.0)
    except ValueError:
        return 0.0


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Needleman-Wunsch GPU benchmark CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Needleman-Wunsch GPU benchmark CSV has no data rows: {csv_path}")
    return rows


def workload_labels(rows: list[dict[str, str]]) -> list[str]:
    return [f"len {row.get('sequence_length')}\npairs {row.get('num_pairs')}" for row in rows]


def save_grouped_time_chart(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = workload_labels(rows)
    x_values = list(range(len(rows)))
    width = 0.35
    cpu_values = [parse_float(row, "cpu_time_ms") for row in rows]
    gpu_values = [parse_float(row, "gpu_total_time_ms") for row in rows]

    plt.figure(figsize=(14, 7))
    plt.bar([x - width / 2 for x in x_values], cpu_values, width, label="CPU")
    plt.bar([x + width / 2 for x in x_values], gpu_values, width, label="GPU total")
    plt.title("Needleman-Wunsch CPU vs GPU Total Time")
    plt.xlabel("Workload")
    plt.ylabel("Time (ms)")
    plt.xticks(x_values, labels, rotation=45, ha="right")
    if all(value > 0.0 for value in cpu_values + gpu_values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


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


def save_copy_kernel_breakdown(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = workload_labels(rows)
    x_values = list(range(len(rows)))
    h2d_values = [parse_float(row, "h2d_copy_time_ms") for row in rows]
    kernel_values = [parse_float(row, "gpu_kernel_time_ms") for row in rows]
    d2h_values = [parse_float(row, "d2h_copy_time_ms") for row in rows]

    plt.figure(figsize=(14, 7))
    plt.bar(x_values, h2d_values, label="H2D copy")
    plt.bar(x_values, kernel_values, bottom=h2d_values, label="Kernel")
    d2h_bottom = [h2d_values[index] + kernel_values[index] for index in range(len(rows))]
    plt.bar(x_values, d2h_values, bottom=d2h_bottom, label="D2H copy")
    plt.title("Needleman-Wunsch GPU H2D, Kernel, and D2H Breakdown")
    plt.xlabel("Workload")
    plt.ylabel("Time (ms)")
    plt.xticks(x_values, labels, rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_scaling_by_sequence_length(rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row.get("num_pairs", ""), []).append(row)

    plt.figure(figsize=(10, 6))
    for num_pairs, group_rows in sorted(grouped.items(), key=lambda item: int(item[0] or 0)):
        sorted_rows = sorted(group_rows, key=lambda row: int(row.get("sequence_length", "0") or 0))
        x_values = [parse_float(row, "sequence_length") for row in sorted_rows]
        y_values = [parse_float(row, "gpu_total_time_ms") for row in sorted_rows]
        plt.plot(x_values, y_values, marker="o", label=f"{num_pairs} pairs")
    plt.title("Needleman-Wunsch GPU Scaling by Sequence Length")
    plt.xlabel("Sequence length")
    plt.ylabel("GPU total time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows(BENCHMARK_CSV_PATH)
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_grouped_time_chart(rows, CHART_OUTPUT_DIRECTORY / "nw_gpu_cpu_vs_gpu_time.png")
    save_bar_chart(
        rows,
        "Needleman-Wunsch GPU Kernel Speedup",
        "Speedup vs CPU",
        "kernel_speedup",
        CHART_OUTPUT_DIRECTORY / "nw_gpu_kernel_speedup.png",
    )
    save_bar_chart(
        rows,
        "Needleman-Wunsch GPU Total Speedup",
        "Speedup vs CPU",
        "total_speedup",
        CHART_OUTPUT_DIRECTORY / "nw_gpu_total_speedup.png",
    )
    save_bar_chart(
        rows,
        "Needleman-Wunsch GPU Total Cells per Second",
        "DP cells per second",
        "cells_per_second_gpu_total",
        CHART_OUTPUT_DIRECTORY / "nw_gpu_cells_per_second.png",
        use_log_scale=True,
    )
    save_copy_kernel_breakdown(
        rows,
        CHART_OUTPUT_DIRECTORY / "nw_gpu_h2d_kernel_d2h_breakdown.png",
    )
    save_scaling_by_sequence_length(
        rows,
        CHART_OUTPUT_DIRECTORY / "nw_gpu_scaling_by_sequence_length.png",
    )

    print(f"Needleman-Wunsch GPU benchmark charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
