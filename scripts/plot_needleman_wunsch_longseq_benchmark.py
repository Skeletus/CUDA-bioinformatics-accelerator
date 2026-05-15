#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


BENCHMARK_CSV_PATH = Path("benchmarks/needleman_wunsch_longseq_benchmark_results.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/needleman_wunsch_longseq")


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0.0") or 0.0)
    except ValueError:
        return 0.0


def parse_int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "0") or 0))
    except ValueError:
        return 0


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Needleman-Wunsch long-sequence benchmark CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"Needleman-Wunsch long-sequence benchmark CSV has no data rows: {csv_path}")
    return rows


def gpu_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("algorithm") != "needleman_wunsch_cpu"
        and row.get("validation_status") not in {"UNSUPPORTED", ""}
    ]


def workload_label(row: dict[str, str]) -> str:
    return (
        f"{row.get('algorithm', '').replace('needleman_wunsch_', '')}\n"
        f"{row.get('implementation')}\n"
        f"len {row.get('sequence_length')}, pairs {row.get('num_pairs')}"
    )


def save_cpu_vs_gpu_total_time(rows: list[dict[str, str]], output_path: Path) -> None:
    filtered_rows = gpu_rows(rows)
    labels = [workload_label(row) for row in filtered_rows]
    x_values = list(range(len(filtered_rows)))
    width = 0.35
    cpu_values = [parse_float(row, "cpu_time_ms") for row in filtered_rows]
    gpu_values = [parse_float(row, "gpu_total_time_ms") for row in filtered_rows]

    plt.figure(figsize=(18, 8))
    plt.bar([value - width / 2 for value in x_values], cpu_values, width, label="CPU reference")
    plt.bar([value + width / 2 for value in x_values], gpu_values, width, label="GPU total")
    plt.title("Needleman-Wunsch Long-Sequence CPU vs GPU Total Time")
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


def save_metric_bar_chart(
    rows: list[dict[str, str]],
    title: str,
    y_label: str,
    column_name: str,
    output_path: Path,
    *,
    use_log_scale: bool = False,
) -> None:
    filtered_rows = gpu_rows(rows)
    labels = [workload_label(row) for row in filtered_rows]
    x_values = list(range(len(filtered_rows)))
    values = [parse_float(row, column_name) for row in filtered_rows]

    plt.figure(figsize=(18, 8))
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


def save_dp_memory_usage(rows: list[dict[str, str]], output_path: Path) -> None:
    filtered_rows = gpu_rows(rows)
    labels = [workload_label(row) for row in filtered_rows]
    x_values = list(range(len(filtered_rows)))
    values = [parse_float(row, "dp_memory_bytes") for row in filtered_rows]

    plt.figure(figsize=(18, 8))
    plt.bar(x_values, values)
    plt.title("Needleman-Wunsch Long-Sequence DP Memory Usage")
    plt.xlabel("Workload")
    plt.ylabel("DP memory bytes")
    plt.xticks(x_values, labels, rotation=45, ha="right")
    if all(value > 0.0 for value in values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_implementation_comparison(rows: list[dict[str, str]], output_path: Path) -> None:
    filtered_rows = [
        row
        for row in rows
        if row.get("algorithm") == "needleman_wunsch_gpu_longseq"
        and row.get("validation_status") not in {"UNSUPPORTED", ""}
    ]
    grouped: dict[tuple[int, int], list[dict[str, str]]] = {}
    for row in filtered_rows:
        key = (parse_int(row, "sequence_length"), parse_int(row, "num_pairs"))
        grouped.setdefault(key, []).append(row)

    plt.figure(figsize=(11, 7))
    for key, group_rows in sorted(grouped.items()):
        sorted_rows = sorted(group_rows, key=lambda row: row.get("implementation", ""))
        x_values = [row.get("implementation", "") for row in sorted_rows]
        y_values = [parse_float(row, "gpu_total_time_ms") for row in sorted_rows]
        plt.plot(x_values, y_values, marker="o", label=f"len {key[0]}, pairs {key[1]}")
    plt.title("Needleman-Wunsch Long-Sequence Implementation Comparison")
    plt.xlabel("Implementation")
    plt.ylabel("GPU total time (ms)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_supported_lengths(rows: list[dict[str, str]], output_path: Path) -> None:
    grouped: dict[str, dict[int, int]] = {}
    for row in rows:
        label = f"{row.get('algorithm')}:{row.get('implementation')}"
        sequence_length = parse_int(row, "sequence_length")
        supported = 0 if row.get("validation_status") == "UNSUPPORTED" else 1
        supported_by_length = grouped.setdefault(label, {})
        supported_by_length[sequence_length] = max(
            supported,
            supported_by_length.get(sequence_length, 0),
        )

    sequence_lengths = sorted({parse_int(row, "sequence_length") for row in rows})
    labels = sorted(grouped)
    plt.figure(figsize=(14, 7))
    for label in labels:
        values = [grouped[label].get(length, 0) for length in sequence_lengths]
        plt.plot(sequence_lengths, values, marker="o", label=label)
    plt.title("Needleman-Wunsch Supported Sequence Lengths")
    plt.xlabel("Sequence length")
    plt.ylabel("Supported")
    plt.yticks([0, 1], ["No", "Yes"])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows(BENCHMARK_CSV_PATH)
    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_cpu_vs_gpu_total_time(rows, CHART_OUTPUT_DIRECTORY / "nw_longseq_cpu_vs_gpu_total_time.png")
    save_metric_bar_chart(
        rows,
        "Needleman-Wunsch Long-Sequence Kernel Speedup",
        "Speedup vs CPU",
        "kernel_speedup_vs_cpu",
        CHART_OUTPUT_DIRECTORY / "nw_longseq_kernel_speedup.png",
    )
    save_metric_bar_chart(
        rows,
        "Needleman-Wunsch Long-Sequence Total Speedup",
        "Speedup vs CPU",
        "total_speedup_vs_cpu",
        CHART_OUTPUT_DIRECTORY / "nw_longseq_total_speedup.png",
    )
    save_metric_bar_chart(
        rows,
        "Needleman-Wunsch Long-Sequence GPU Cells per Second",
        "DP cells per second",
        "gpu_total_cells_per_second",
        CHART_OUTPUT_DIRECTORY / "nw_longseq_cells_per_second.png",
        use_log_scale=True,
    )
    save_dp_memory_usage(rows, CHART_OUTPUT_DIRECTORY / "nw_longseq_dp_memory_usage.png")
    save_implementation_comparison(rows, CHART_OUTPUT_DIRECTORY / "nw_longseq_implementation_comparison.png")
    save_supported_lengths(rows, CHART_OUTPUT_DIRECTORY / "nw_longseq_supported_lengths.png")

    print(f"Needleman-Wunsch long-sequence benchmark charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
