#!/usr/bin/env python3

import csv
from pathlib import Path

import matplotlib.pyplot as plt


SUMMARY_CSV_PATH = Path("benchmarks/phase7_final_summary.csv")
CHART_OUTPUT_DIRECTORY = Path("assets/benchmark_charts/phase7_final_summary")


def parse_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0.0") or 0.0)
    except ValueError:
        return 0.0


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Phase 7 summary CSV not found: {csv_path}")
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def mode_labels(rows: list[dict[str, str]]) -> list[str]:
    return [row.get("pairing_mode", "unknown") for row in rows]


def add_value_labels(x_values: list[float], values: list[float], *, precision: int = 2) -> None:
    for x_value, value in zip(x_values, values):
        label = f"{value:.{precision}f}"
        plt.text(x_value, value, label, ha="center", va="bottom", fontsize=9)


def save_empty_chart(title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.title(title)
    plt.text(0.5, 0.5, "No Phase 7 summary rows available", ha="center", va="center")
    plt.axis("off")
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
    if not rows:
        save_empty_chart(title, output_path)
        return

    labels = mode_labels(rows)
    x_values = list(range(len(rows)))
    values = [parse_float(row, column_name) for row in rows]

    plt.figure(figsize=(11, 6))
    plt.bar(x_values, values)
    plt.title(title)
    plt.xlabel("Pairing mode")
    plt.ylabel(y_label)
    plt.xticks(x_values, labels, rotation=20, ha="right")
    if use_log_scale and all(value > 0.0 for value in values):
        plt.yscale("log")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    if not use_log_scale:
        add_value_labels([float(x) for x in x_values], values)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_grouped_speedup_chart(rows: list[dict[str, str]], output_path: Path) -> None:
    title = "Pipeline Speedup vs End-to-End Speedup"
    if not rows:
        save_empty_chart(title, output_path)
        return

    labels = mode_labels(rows)
    x_values = list(range(len(rows)))
    bar_width = 0.4
    pipeline_values = [parse_float(row, "pipeline_speedup_indexed_vs_flat") for row in rows]
    end_to_end_values = [parse_float(row, "end_to_end_speedup_indexed_vs_flat") for row in rows]
    left_offsets = [x - bar_width / 2.0 for x in x_values]
    right_offsets = [x + bar_width / 2.0 for x in x_values]

    plt.figure(figsize=(11, 6))
    plt.bar(left_offsets, pipeline_values, width=bar_width, label="Pipeline speedup")
    plt.bar(right_offsets, end_to_end_values, width=bar_width, label="End-to-end speedup")
    plt.title(title)
    plt.xlabel("Pairing mode")
    plt.ylabel("Speedup")
    plt.xticks(x_values, labels, rotation=20, ha="right")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_validation_status_chart(rows: list[dict[str, str]], output_path: Path) -> None:
    title = "Phase 7 Validation Status by Pairing Mode"
    if not rows:
        save_empty_chart(title, output_path)
        return

    labels = mode_labels(rows)
    status_values = [1.0 if row.get("validation_status", "").upper() == "PASSED" else 0.0 for row in rows]
    status_labels = [row.get("validation_status", "UNKNOWN") for row in rows]
    x_values = list(range(len(rows)))

    plt.figure(figsize=(11, 5))
    plt.bar(x_values, status_values)
    for x_value, status_label in zip(x_values, status_labels):
        plt.text(x_value, 0.5, status_label, ha="center", va="center", fontsize=10)
    plt.title(title)
    plt.xlabel("Pairing mode")
    plt.ylabel("Validation passed")
    plt.yticks([0, 1], ["Failed or unknown", "Passed"])
    plt.xticks(x_values, labels, rotation=20, ha="right")
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_table_like_chart(rows: list[dict[str, str]], title: str, output_path: Path, columns: list[str]) -> None:
    if not rows:
        save_empty_chart(title, output_path)
        return

    table_rows = [[row.get(column, "") for column in columns] for row in rows]
    plt.figure(figsize=(13, 5))
    plt.title(title)
    plt.axis("off")
    table = plt.table(
        cellText=table_rows,
        colLabels=[column.replace("_", " ").title() for column in columns],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_bottleneck_summary_chart(rows: list[dict[str, str]], output_path: Path) -> None:
    title = "Phase 7 Bottleneck Summary"
    if not rows:
        save_empty_chart(title, output_path)
        return

    average_pipeline_speedup = sum(parse_float(row, "pipeline_speedup_indexed_vs_flat") for row in rows) / len(rows)
    average_end_to_end_speedup = sum(parse_float(row, "end_to_end_speedup_indexed_vs_flat") for row in rows) / len(rows)
    labels = ["GPU pipeline improved", "End-to-end remains limited", "Next bottleneck: CPU preprocessing"]
    values = [average_pipeline_speedup, average_end_to_end_speedup, max(0.0, average_pipeline_speedup - average_end_to_end_speedup)]

    plt.figure(figsize=(11, 6))
    x_values = list(range(len(labels)))
    plt.bar(x_values, values)
    plt.title(title)
    plt.ylabel("Relative signal")
    plt.xticks(x_values, labels, rotation=15, ha="right")
    plt.grid(True, which="both", axis="y", alpha=0.3)
    add_value_labels([float(x) for x in x_values], values)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    rows = load_rows(SUMMARY_CSV_PATH)
    if not rows:
        print(f"Warning: no rows found in {SUMMARY_CSV_PATH}. Placeholder charts will be generated.")

    CHART_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    save_bar_chart(
        rows,
        "Phase 7 GPU Pipeline Speedup by Pairing Mode",
        "Pipeline speedup indexed vs flat",
        "pipeline_speedup_indexed_vs_flat",
        CHART_OUTPUT_DIRECTORY / "phase7_pipeline_speedup.png",
    )
    save_bar_chart(
        rows,
        "Phase 7 End-to-End Speedup by Pairing Mode",
        "End-to-end speedup indexed vs flat",
        "end_to_end_speedup_indexed_vs_flat",
        CHART_OUTPUT_DIRECTORY / "phase7_end_to_end_speedup.png",
    )
    save_bar_chart(
        rows,
        "Phase 7 Transfer Footprint Reduction by Pairing Mode",
        "Flat bytes / indexed bytes",
        "transfer_reduction_ratio",
        CHART_OUTPUT_DIRECTORY / "phase7_transfer_reduction_ratio.png",
    )
    save_grouped_speedup_chart(
        rows,
        CHART_OUTPUT_DIRECTORY / "phase7_pipeline_vs_end_to_end_speedup.png",
    )
    save_bar_chart(
        rows,
        "Number of Sequence Pairs by Pairing Mode",
        "Number of pairs",
        "number_of_pairs",
        CHART_OUTPUT_DIRECTORY / "phase7_number_of_pairs.png",
        use_log_scale=True,
    )
    save_validation_status_chart(
        rows,
        CHART_OUTPUT_DIRECTORY / "phase7_validation_status.png",
    )
    save_table_like_chart(
        rows,
        "Phase 7 Best Modes Summary",
        CHART_OUTPUT_DIRECTORY / "phase7_best_modes_summary.png",
        ["pairing_mode", "recommended_role", "interpretation"],
    )
    save_bottleneck_summary_chart(
        rows,
        CHART_OUTPUT_DIRECTORY / "phase7_bottleneck_summary.png",
    )

    print(f"Phase 7 final summary charts saved to: {CHART_OUTPUT_DIRECTORY}")


if __name__ == "__main__":
    main()
