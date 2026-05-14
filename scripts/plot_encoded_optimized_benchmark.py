#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/encoded_optimized_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/encoded_optimized")


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


def save_stacked_bar_chart(
    rows: list[dict[str, str]],
    title: str,
    y_label: str,
    series: list[tuple[str, str]],
    output_path: Path,
) -> None:
    labels = mode_labels(rows)
    x_values = list(range(len(rows)))
    bottoms = [0.0 for _ in rows]

    plt.figure(figsize=(12, 7))
    for label, column_name in series:
        values = [parse_float(row, column_name) for row in rows]
        plt.bar(x_values, values, bottom=bottoms, label=label)
        bottoms = [bottom + value for bottom, value in zip(bottoms, values)]

    plt.title(title)
    plt.xlabel("Pairing mode")
    plt.ylabel(y_label)
    plt.xticks(x_values, labels, rotation=20, ha="right")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows(BENCHMARK_CSV_PATH)
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_grouped_bar_chart(
        rows,
        "Optimized vs Baseline Encoded End-to-end Time",
        "Time (ms)",
        [
            ("Baseline encoded", "baseline_encoded_end_to_end_time_ms"),
            ("Optimized encoded", "optimized_end_to_end_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "optimized_vs_baseline_end_to_end.png",
        use_log_scale=True,
    )

    save_grouped_bar_chart(
        rows,
        "Optimized vs Baseline Encoded GPU Pipeline Time",
        "Time (ms)",
        [
            ("Baseline encoded GPU total", "baseline_encoded_gpu_total_time_ms"),
            ("Optimized encoded pipeline", "optimized_gpu_pipeline_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "optimized_vs_baseline_gpu_pipeline.png",
        use_log_scale=True,
    )

    save_grouped_bar_chart(
        rows,
        "Optimized H2D and D2H Copy Breakdown",
        "Time (ms)",
        [
            ("H2D copy", "optimized_h2d_copy_time_ms"),
            ("D2H copy", "optimized_d2h_copy_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "optimized_h2d_d2h_breakdown.png",
        use_log_scale=True,
    )

    save_grouped_bar_chart(
        rows,
        "Optimized Encoded Cache Hit Rate",
        "Cache hit rate",
        [("Cache hit rate", "cache_hit_rate")],
        CHART_OUTPUT_DIRECTORY / "optimized_cache_hit_rate.png",
    )

    save_stacked_bar_chart(
        rows,
        "Optimized Setup Time vs GPU Pipeline Time",
        "Time (ms)",
        [
            ("Setup", "optimized_setup_time_ms"),
            ("GPU pipeline", "optimized_gpu_pipeline_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "optimized_setup_vs_pipeline.png",
    )

    save_grouped_bar_chart(
        rows,
        "Optimized Encoded Speedup by Pairing Mode",
        "Speedup over baseline",
        [
            ("Kernel", "kernel_speedup_optimized_vs_baseline"),
            ("GPU pipeline", "gpu_pipeline_speedup_optimized_vs_baseline"),
            ("End-to-end", "end_to_end_speedup_optimized_vs_baseline"),
        ],
        CHART_OUTPUT_DIRECTORY / "optimized_speedup_by_pairing_mode.png",
    )

    print(f"Encoded optimized benchmark charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
