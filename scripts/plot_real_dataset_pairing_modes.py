#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/real_dataset_pairing_modes_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/real_dataset_pairing_modes")


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError):
        return 0.0


def load_benchmark_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Benchmark CSV not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))

    if not rows:
        raise ValueError(f"Benchmark CSV has no data rows: {csv_path}")

    return rows


def mode_labels(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("pairing_mode", "unknown") for row in rows]


def save_bar_chart(
    rows: list[dict[str, str]],
    title: str,
    x_label: str,
    y_label: str,
    series: list[tuple[str, str]],
    output_path: Path,
    *,
    use_log_scale: bool = False,
) -> None:
    labels = mode_labels(rows)
    x_values = list(range(len(rows)))
    bar_width = 0.8 / max(1, len(series))

    plt.figure(figsize=(11, 6))
    all_values: list[float] = []
    for series_index, (label, column_name) in enumerate(series):
        offsets = [x_value + (series_index - (len(series) - 1) / 2.0) * bar_width for x_value in x_values]
        values = [parse_float(row, column_name) for row in rows]
        all_values.extend(values)
        plt.bar(offsets, values, width=bar_width, label=label)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(x_values, labels, rotation=20, ha="right")
    if use_log_scale and all(value > 0.0 for value in all_values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    if len(series) > 1:
        plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def add_throughput_columns(rows: list[dict[str, str]]) -> None:
    for row in rows:
        total_bases = parse_float(row, "total_bases_compared")
        for source_column, output_column in [
            ("cpu_time_ms", "cpu_bases_per_second"),
            ("char_gpu_kernel_time_ms", "char_kernel_bases_per_second"),
            ("char_gpu_total_time_ms", "char_total_bases_per_second"),
            ("encoded_gpu_kernel_time_ms", "encoded_kernel_bases_per_second"),
            ("encoded_gpu_total_time_ms", "encoded_total_bases_per_second"),
        ]:
            time_ms = parse_float(row, source_column)
            row[output_column] = str(total_bases / (time_ms / 1000.0)) if time_ms > 0.0 else "0.0"


def main() -> None:
    rows = load_benchmark_rows(BENCHMARK_CSV_PATH)
    add_throughput_columns(rows)
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_bar_chart(
        rows,
        "Real Dataset Pairing Modes CPU vs GPU Time",
        "Pairing mode",
        "Average time (ms)",
        [
            ("CPU", "cpu_time_ms"),
            ("Char GPU kernel", "char_gpu_kernel_time_ms"),
            ("Char GPU total", "char_gpu_total_time_ms"),
            ("Encoded GPU kernel", "encoded_gpu_kernel_time_ms"),
            ("Encoded GPU total", "encoded_gpu_total_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "pairing_modes_cpu_vs_gpu_time.png",
        use_log_scale=True,
    )
    save_bar_chart(
        rows,
        "Real Dataset Pairing Modes Kernel Speedup",
        "Pairing mode",
        "Speedup over CPU",
        [
            ("Char kernel speedup", "char_kernel_speedup"),
            ("Encoded kernel speedup", "encoded_kernel_speedup"),
        ],
        CHART_OUTPUT_DIRECTORY / "pairing_modes_kernel_speedup.png",
    )
    save_bar_chart(
        rows,
        "Real Dataset Pairing Modes Total Speedup",
        "Pairing mode",
        "Speedup over CPU",
        [
            ("Char total speedup", "char_total_speedup"),
            ("Encoded total speedup", "encoded_total_speedup"),
        ],
        CHART_OUTPUT_DIRECTORY / "pairing_modes_total_speedup.png",
    )
    save_bar_chart(
        rows,
        "Real Dataset Pairing Modes Throughput",
        "Pairing mode",
        "Bases compared per second",
        [
            ("CPU", "cpu_bases_per_second"),
            ("Char GPU kernel", "char_kernel_bases_per_second"),
            ("Char GPU total", "char_total_bases_per_second"),
            ("Encoded GPU kernel", "encoded_kernel_bases_per_second"),
            ("Encoded GPU total", "encoded_total_bases_per_second"),
        ],
        CHART_OUTPUT_DIRECTORY / "pairing_modes_throughput.png",
        use_log_scale=True,
    )
    save_bar_chart(
        rows,
        "Number of Pairs by Pairing Mode",
        "Pairing mode",
        "Number of pairs",
        [("Pairs", "number_of_pairs")],
        CHART_OUTPUT_DIRECTORY / "number_of_pairs_by_mode.png",
        use_log_scale=True,
    )

    print(f"Real dataset pairing mode charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
