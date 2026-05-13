#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/dna_encoding_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/encoding")


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
        num_pairs = int(float(row["num_pairs"]))
        sequence_length = int(float(row["sequence_length"]))
        labels.append(f"{num_pairs:,}\nlen {sequence_length}")
    return labels


def configure_axes(title: str, y_label: str, labels: list[str], values: list[float]) -> None:
    plt.title(title)
    plt.xlabel("Workload")
    plt.ylabel(y_label)
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    if values and all(value > 0.0 for value in values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()


def save_line_chart(
    rows: list[dict[str, str]],
    title: str,
    y_label: str,
    series: list[tuple[str, str]],
    output_path: Path,
    *,
    use_log_scale: bool = True,
) -> None:
    labels = workload_labels(rows)
    x_values = list(range(len(rows)))
    all_values: list[float] = []

    plt.figure(figsize=(12, 6))
    for label, column_name in series:
        y_values = [parse_float(row, column_name) for row in rows]
        all_values.extend(y_values)
        plt.plot(x_values, y_values, marker="o", label=label)

    if not use_log_scale:
        all_values = []
    configure_axes(title, y_label, labels, all_values)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rows = load_benchmark_rows(BENCHMARK_CSV_PATH)
    rows.sort(key=lambda row: (int(float(row["sequence_length"])), int(float(row["num_pairs"]))))
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_line_chart(
        rows,
        "Char vs Encoded GPU Kernel Time",
        "Average kernel time (ms)",
        [
            ("Char GPU kernel", "char_gpu_kernel_time_ms"),
            ("Encoded GPU kernel", "encoded_gpu_kernel_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "char_vs_encoded_kernel_time.png",
    )
    save_line_chart(
        rows,
        "Char vs Encoded GPU Total Time",
        "Average total time (ms)",
        [
            ("Char GPU total", "char_gpu_total_time_ms"),
            ("Encoded GPU total", "encoded_gpu_total_time_ms"),
        ],
        CHART_OUTPUT_DIRECTORY / "char_vs_encoded_total_time.png",
    )
    save_line_chart(
        rows,
        "Encoded Kernel Speedup over Char Kernel",
        "Speedup",
        [("Kernel speedup", "kernel_speedup_encoded_vs_char")],
        CHART_OUTPUT_DIRECTORY / "encoded_kernel_speedup.png",
        use_log_scale=False,
    )
    save_line_chart(
        rows,
        "Encoded Total Speedup over Char Total",
        "Speedup",
        [("Total speedup", "total_speedup_encoded_vs_char")],
        CHART_OUTPUT_DIRECTORY / "encoded_total_speedup.png",
        use_log_scale=False,
    )
    save_line_chart(
        rows,
        "CPU Encoding Overhead",
        "Encoding time (ms)",
        [("Encoding time", "encoding_time_ms")],
        CHART_OUTPUT_DIRECTORY / "encoding_overhead.png",
    )

    print(f"Encoding benchmark charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
