#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/indexed_graphs_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/indexed_graphs")


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0.0") or 0.0)
    except ValueError:
        return 0.0


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Benchmark CSV has no data rows: {csv_path}")
    return rows


def mode_labels(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("pairing_mode", "unknown") for row in rows]


def save_grouped_bar_chart(
    rows: list[dict[str, str]],
    title: str,
    y_label: str,
    series: list[tuple[str, str]],
    output_path: Path,
    *,
    use_log_scale: bool = False,
) -> None:
    labels = mode_labels(rows)
    x_values = list(range(len(rows)))
    bar_width = 0.8 / max(1, len(series))
    all_values: list[float] = []

    plt.figure(figsize=(12, 7))
    for series_index, (label, column_name) in enumerate(series):
        offsets = [x_value + (series_index - (len(series) - 1) / 2.0) * bar_width for x_value in x_values]
        values = [parse_float(row, column_name) for row in rows]
        all_values.extend(values)
        plt.bar(offsets, values, width=bar_width, label=label)

    plt.title(title)
    plt.xlabel("Pairing mode")
    plt.ylabel(y_label)
    plt.xticks(x_values, labels, rotation=20, ha="right")
    if use_log_scale and all(value > 0.0 for value in all_values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_single_bar_chart(
    rows: list[dict[str, str]],
    title: str,
    y_label: str,
    column_name: str,
    output_path: Path,
) -> None:
    labels = mode_labels(rows)
    x_values = list(range(len(rows)))
    values = [parse_float(row, column_name) for row in rows]

    plt.figure(figsize=(12, 7))
    plt.bar(x_values, values)
    plt.title(title)
    plt.xlabel("Pairing mode")
    plt.ylabel(y_label)
    plt.xticks(x_values, labels, rotation=20, ha="right")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows(BENCHMARK_CSV_PATH)
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_grouped_bar_chart(
        rows,
        "Indexed vs Flat Encoded GPU Pipeline Time",
        "Time (ms)",
        [
            ("Flat optimized", "flat_optimized_gpu_pipeline_time_ms"),
            ("Indexed graphs", "indexed_graphs_gpu_pipeline_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "indexed_vs_flat_gpu_pipeline_time.png",
        use_log_scale=True,
    )

    save_grouped_bar_chart(
        rows,
        "Indexed vs Flat Encoded End-to-end Time",
        "Time (ms)",
        [
            ("Flat optimized", "flat_optimized_end_to_end_time_ms"),
            ("Indexed graphs", "indexed_graphs_end_to_end_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "indexed_vs_flat_end_to_end_time.png",
        use_log_scale=True,
    )

    save_single_bar_chart(
        rows,
        "Indexed Pipeline Speedup over Flat Optimized",
        "Speedup",
        "pipeline_speedup_indexed_vs_flat",
        CHART_OUTPUT_DIRECTORY / "indexed_pipeline_speedup.png",
    )

    save_single_bar_chart(
        rows,
        "Indexed End-to-end Speedup over Flat Optimized",
        "Speedup",
        "end_to_end_speedup_indexed_vs_flat",
        CHART_OUTPUT_DIRECTORY / "indexed_end_to_end_speedup.png",
    )

    save_single_bar_chart(
        rows,
        "Indexed Transfer Reduction Ratio",
        "Flat bytes / indexed bytes",
        "transfer_reduction_ratio",
        CHART_OUTPUT_DIRECTORY / "transfer_reduction_ratio.png",
    )

    save_grouped_bar_chart(
        rows,
        "Indexed Representation Memory Footprint",
        "Bytes",
        [
            ("Flat pair bytes equivalent", "flat_pair_bytes_equivalent"),
            ("Indexed representation bytes", "indexed_representation_bytes"),
            ("Result bytes", "result_bytes"),
        ],
        CHART_OUTPUT_DIRECTORY / "indexed_memory_footprint.png",
        use_log_scale=True,
    )

    save_grouped_bar_chart(
        rows,
        "CUDA Graphs Timing Breakdown",
        "Time (ms)",
        [
            ("Graph creation", "graph_creation_time_ms"),
            ("Graph instantiation", "graph_instantiation_time_ms"),
            ("Graph average execution", "graph_average_execution_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "cuda_graphs_timing_breakdown.png",
        use_log_scale=True,
    )

    print(f"Indexed CUDA Graphs benchmark charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
