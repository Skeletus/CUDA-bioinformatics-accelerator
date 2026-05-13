#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/real_dataset_hamming_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/real_dataset")


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


def workload_labels(rows: list[dict[str, str]]) -> list[str]:
    labels = []
    for row in rows:
        dataset_name = row.get("dataset_name", "dataset")
        window_size = int(parse_float(row, "window_size"))
        stride = int(parse_float(row, "stride"))
        labels.append(f"{dataset_name}\nwin {window_size}\nstride {stride}")
    return labels


def save_bar_chart(
    rows: list[dict[str, str]],
    title: str,
    y_label: str,
    series: list[tuple[str, str]],
    output_path: Path,
    *,
    use_log_scale: bool = False,
) -> None:
    labels = workload_labels(rows)
    x_values = list(range(len(rows)))
    bar_width = 0.8 / max(1, len(series))

    plt.figure(figsize=(10, 6))
    all_values: list[float] = []
    for series_index, (label, column_name) in enumerate(series):
        offsets = [x_value + (series_index - (len(series) - 1) / 2.0) * bar_width for x_value in x_values]
        values = [parse_float(row, column_name) for row in rows]
        all_values.extend(values)
        plt.bar(offsets, values, width=bar_width, label=label)

    plt.title(title)
    plt.xlabel("Real dataset workload")
    plt.ylabel(y_label)
    plt.xticks(x_values, labels)
    if use_log_scale and all(value > 0.0 for value in all_values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
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
        "Real Dataset CPU vs GPU Time",
        "Average time (ms)",
        [
            ("CPU", "cpu_time_ms"),
            ("Char GPU kernel", "char_gpu_kernel_time_ms"),
            ("Char GPU total", "char_gpu_total_time_ms"),
            ("Encoded GPU kernel", "encoded_gpu_kernel_time_ms"),
            ("Encoded GPU total", "encoded_gpu_total_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "real_dataset_cpu_vs_gpu_time.png",
        use_log_scale=True,
    )
    save_bar_chart(
        rows,
        "Real Dataset Kernel Speedup",
        "Speedup over CPU",
        [
            ("Char kernel speedup", "char_kernel_speedup"),
            ("Encoded kernel speedup", "encoded_kernel_speedup"),
        ],
        CHART_OUTPUT_DIRECTORY / "real_dataset_kernel_speedup.png",
    )
    save_bar_chart(
        rows,
        "Real Dataset Total Speedup",
        "Speedup over CPU",
        [
            ("Char total speedup", "char_total_speedup"),
            ("Encoded total speedup", "encoded_total_speedup"),
        ],
        CHART_OUTPUT_DIRECTORY / "real_dataset_total_speedup.png",
    )
    save_bar_chart(
        rows,
        "Real Dataset Throughput",
        "Bases compared per second",
        [
            ("CPU", "cpu_bases_per_second"),
            ("Char GPU kernel", "char_kernel_bases_per_second"),
            ("Char GPU total", "char_total_bases_per_second"),
            ("Encoded GPU kernel", "encoded_kernel_bases_per_second"),
            ("Encoded GPU total", "encoded_total_bases_per_second"),
        ],
        CHART_OUTPUT_DIRECTORY / "real_dataset_throughput.png",
        use_log_scale=True,
    )

    print(f"Real dataset benchmark charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
