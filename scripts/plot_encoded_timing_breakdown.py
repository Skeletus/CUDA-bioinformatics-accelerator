#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/real_dataset_pairing_modes_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/encoded_timing_breakdown")


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


def save_stacked_bar_chart(
    rows: list[dict[str, str]],
    components: list[tuple[str, str]],
    title: str,
    y_label: str,
    output_path: Path,
) -> None:
    labels = mode_labels(rows)
    x_values = list(range(len(rows)))
    bottoms = [0.0 for _ in rows]

    plt.figure(figsize=(12, 7))
    for label, column_name in components:
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


def add_other_end_to_end_column(rows: list[dict[str, str]]) -> None:
    measured_columns = [
        "encoded_encoding_time_ms",
        "encoded_h2d_copy_time_ms",
        "encoded_gpu_kernel_time_ms",
        "encoded_d2h_copy_time_ms",
        "encoded_validation_time_ms",
        "encoded_csv_write_time_ms",
    ]
    for row in rows:
        end_to_end_time_ms = parse_float(row, "encoded_end_to_end_time_ms")
        measured_time_ms = sum(parse_float(row, column_name) for column_name in measured_columns)
        row["encoded_other_end_to_end_time_ms"] = str(max(0.0, end_to_end_time_ms - measured_time_ms))


def add_non_kernel_overhead_column(rows: list[dict[str, str]]) -> None:
    for row in rows:
        end_to_end_time_ms = parse_float(row, "encoded_end_to_end_time_ms")
        kernel_time_ms = parse_float(row, "encoded_gpu_kernel_time_ms")
        row["encoded_non_kernel_overhead_time_ms"] = str(max(0.0, end_to_end_time_ms - kernel_time_ms))


def main() -> None:
    rows = load_rows(BENCHMARK_CSV_PATH)
    add_other_end_to_end_column(rows)
    add_non_kernel_overhead_column(rows)
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_stacked_bar_chart(
        rows,
        [
            ("Encoding", "encoded_encoding_time_ms"),
            ("H2D copy", "encoded_h2d_copy_time_ms"),
            ("Kernel", "encoded_gpu_kernel_time_ms"),
            ("D2H copy", "encoded_d2h_copy_time_ms"),
            ("Validation", "encoded_validation_time_ms"),
            ("CSV write", "encoded_csv_write_time_ms"),
            ("Other end-to-end", "encoded_other_end_to_end_time_ms"),
        ],
        "Encoded End-to-end Timing Breakdown by Pairing Mode",
        "Time (ms)",
        CHART_OUTPUT_DIRECTORY / "encoded_timing_breakdown_by_mode.png",
    )

    save_stacked_bar_chart(
        rows,
        [
            ("Device allocation", "encoded_device_allocation_time_ms"),
            ("H2D copy", "encoded_h2d_copy_time_ms"),
            ("Kernel", "encoded_gpu_kernel_time_ms"),
            ("D2H copy", "encoded_d2h_copy_time_ms"),
        ],
        "Encoded GPU Pipeline Breakdown by Pairing Mode",
        "Time (ms)",
        CHART_OUTPUT_DIRECTORY / "encoded_gpu_pipeline_breakdown_by_mode.png",
    )

    save_stacked_bar_chart(
        rows,
        [
            ("Kernel", "encoded_gpu_kernel_time_ms"),
            ("Non-kernel overhead", "encoded_non_kernel_overhead_time_ms"),
        ],
        "Encoded Kernel Time vs Non-kernel Overhead by Pairing Mode",
        "Time (ms)",
        CHART_OUTPUT_DIRECTORY / "encoded_overhead_vs_kernel_by_mode.png",
    )

    print(f"Encoded timing breakdown charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
