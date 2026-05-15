#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/needleman_wunsch_gpu_optimized_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/needleman_wunsch_gpu_optimized")


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0.0") or 0.0)
    except ValueError:
        return 0.0


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Needleman-Wunsch optimized benchmark CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Needleman-Wunsch optimized benchmark CSV has no data rows: {csv_path}")
    return rows


def workload_labels(rows: list[dict[str, str]]) -> list[str]:
    return [
        f"len {row.get('sequence_length')}\npairs {row.get('num_pairs')}\n"
        f"batch {row.get('batch_size')}\nstreams {row.get('num_streams')}"
        for row in rows
    ]


def save_optimized_vs_baseline_pipeline(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = workload_labels(rows)
    x_values = list(range(len(rows)))
    width = 0.35
    baseline_values = [parse_float(row, "baseline_gpu_total_time_ms") for row in rows]
    optimized_values = [parse_float(row, "optimized_gpu_pipeline_time_ms") for row in rows]

    plt.figure(figsize=(16, 8))
    plt.bar([x - width / 2 for x in x_values], baseline_values, width, label="Baseline GPU total")
    plt.bar([x + width / 2 for x in x_values], optimized_values, width, label="Optimized GPU pipeline")
    plt.title("Needleman-Wunsch Optimized Pipeline vs Baseline GPU")
    plt.xlabel("Workload")
    plt.ylabel("Time (ms)")
    plt.xticks(x_values, labels, rotation=45, ha="right")
    if all(value > 0.0 for value in baseline_values + optimized_values):
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

    plt.figure(figsize=(16, 8))
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


def save_streams_comparison(rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("sequence_length", ""), row.get("num_pairs", ""), row.get("batch_size", ""))
        grouped.setdefault(key, []).append(row)

    plt.figure(figsize=(11, 7))
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple(int(value or 0) for value in item[0])):
        sorted_rows = sorted(group_rows, key=lambda row: int(row.get("num_streams", "0") or 0))
        x_values = [parse_float(row, "num_streams") for row in sorted_rows]
        y_values = [parse_float(row, "optimized_gpu_pipeline_time_ms") for row in sorted_rows]
        label = f"len {key[0]}, pairs {key[1]}, batch {key[2]}"
        plt.plot(x_values, y_values, marker="o", label=label)
    plt.title("Needleman-Wunsch Optimized Pipeline by Stream Count")
    plt.xlabel("Number of CUDA streams")
    plt.ylabel("Optimized GPU pipeline time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_batch_size_comparison(rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = {}
    for row in rows:
        key = (row.get("sequence_length", ""), row.get("num_pairs", ""), row.get("num_streams", ""))
        grouped.setdefault(key, []).append(row)

    plt.figure(figsize=(11, 7))
    for key, group_rows in sorted(grouped.items(), key=lambda item: tuple(int(value or 0) for value in item[0])):
        sorted_rows = sorted(group_rows, key=lambda row: int(row.get("batch_size", "0") or 0))
        x_values = [parse_float(row, "batch_size") for row in sorted_rows]
        y_values = [parse_float(row, "optimized_gpu_pipeline_time_ms") for row in sorted_rows]
        label = f"len {key[0]}, pairs {key[1]}, streams {key[2]}"
        plt.plot(x_values, y_values, marker="o", label=label)
    plt.title("Needleman-Wunsch Optimized Pipeline by Batch Size")
    plt.xlabel("Batch size")
    plt.ylabel("Optimized GPU pipeline time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_h2d_kernel_d2h_breakdown(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = workload_labels(rows)
    x_values = list(range(len(rows)))
    h2d_values = [parse_float(row, "optimized_h2d_copy_time_ms") for row in rows]
    kernel_values = [parse_float(row, "optimized_gpu_kernel_time_ms") for row in rows]
    d2h_values = [parse_float(row, "optimized_d2h_copy_time_ms") for row in rows]

    plt.figure(figsize=(16, 8))
    plt.bar(x_values, h2d_values, label="H2D copy")
    plt.bar(x_values, kernel_values, bottom=h2d_values, label="Kernel")
    d2h_bottom = [h2d_values[index] + kernel_values[index] for index in range(len(rows))]
    plt.bar(x_values, d2h_values, bottom=d2h_bottom, label="D2H copy")
    plt.title("Needleman-Wunsch Optimized H2D, Kernel, and D2H Breakdown")
    plt.xlabel("Workload")
    plt.ylabel("Time (ms)")
    plt.xticks(x_values, labels, rotation=45, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_cells_per_second(rows: list[dict[str, str]], output_path: Path) -> None:
    labels = workload_labels(rows)
    x_values = list(range(len(rows)))
    values = []
    for row in rows:
        cells = parse_float(row, "total_cells_computed")
        pipeline_time_ms = parse_float(row, "optimized_gpu_pipeline_time_ms")
        values.append(0.0 if pipeline_time_ms <= 0.0 else cells / (pipeline_time_ms / 1000.0))

    plt.figure(figsize=(16, 8))
    plt.bar(x_values, values)
    plt.title("Needleman-Wunsch Optimized GPU Cells per Second")
    plt.xlabel("Workload")
    plt.ylabel("DP cells per second")
    plt.xticks(x_values, labels, rotation=45, ha="right")
    if all(value > 0.0 for value in values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows(BENCHMARK_CSV_PATH)
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_optimized_vs_baseline_pipeline(
        rows,
        CHART_OUTPUT_DIRECTORY / "nw_optimized_vs_baseline_gpu_pipeline.png",
    )
    save_bar_chart(
        rows,
        "Needleman-Wunsch Optimized Pipeline Speedup",
        "Speedup vs baseline GPU",
        "pipeline_speedup_optimized_vs_baseline",
        CHART_OUTPUT_DIRECTORY / "nw_optimized_pipeline_speedup.png",
    )
    save_bar_chart(
        rows,
        "Needleman-Wunsch Optimized End-to-End Speedup",
        "Speedup vs baseline GPU",
        "end_to_end_speedup_optimized_vs_baseline",
        CHART_OUTPUT_DIRECTORY / "nw_optimized_end_to_end_speedup.png",
    )
    save_streams_comparison(
        rows,
        CHART_OUTPUT_DIRECTORY / "nw_optimized_streams_comparison.png",
    )
    save_batch_size_comparison(
        rows,
        CHART_OUTPUT_DIRECTORY / "nw_optimized_batch_size_comparison.png",
    )
    save_h2d_kernel_d2h_breakdown(
        rows,
        CHART_OUTPUT_DIRECTORY / "nw_optimized_h2d_kernel_d2h_breakdown.png",
    )
    save_cells_per_second(
        rows,
        CHART_OUTPUT_DIRECTORY / "nw_optimized_cells_per_second.png",
    )

    print(f"Needleman-Wunsch optimized benchmark charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
