#!/usr/bin/env python3

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_DNA_BASES = {"A", "C", "G", "T"}
DEFAULT_INPUT_PATH = Path("data/raw/sars_cov_2_NC_045512_2.fasta")
DEFAULT_OUTPUT_CSV_PATH = Path("data/processed/sars_cov_2_fragments_128.csv")
DEFAULT_OUTPUT_TXT_PATH = Path("data/sars_cov_2_fragments_128.txt")
DEFAULT_OUTPUT_PAIRS_PATH = Path("data/processed/sars_cov_2_pairs_128_stride_32.txt")


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    start: int
    end: int
    sequence: str


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a FASTA genome and generate fixed-length DNA fragments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input FASTA path.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV_PATH, help="Output fragment CSV path.")
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT_PATH, help="Output fragment text path.")
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=DEFAULT_OUTPUT_PAIRS_PATH,
        help="Output adjacent sequence pair path.",
    )
    parser.add_argument("--window-size", type=int, default=128, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride.")
    parser.add_argument(
        "--skip-ambiguous",
        action="store_true",
        help="Skip fragments that contain bases outside A, C, G, and T.",
    )
    return parser.parse_args()


def read_fasta(input_path: Path) -> tuple[str, str]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input FASTA file not found: {input_path}")

    header = ""
    sequence_lines: list[str] = []

    with input_path.open("r", encoding="utf-8") as input_file:
        for line_number, raw_line in enumerate(input_file, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header:
                    raise ValueError("Only single-record FASTA files are supported in Phase 7.")
                header = line[1:].strip()
                continue
            if not header:
                raise ValueError(f"Found sequence data before a FASTA header at line {line_number}.")
            sequence_lines.append(line.upper())

    if not header:
        raise ValueError("FASTA header was not found.")
    if not sequence_lines:
        raise ValueError("FASTA sequence data was not found.")

    return header, "".join(sequence_lines)


def unsupported_bases(sequence: str) -> set[str]:
    return {base for base in sequence if base not in SUPPORTED_DNA_BASES}


def generate_fragments(sequence: str, window_size: int, stride: int, skip_ambiguous: bool) -> list[Fragment]:
    if window_size <= 0:
        raise ValueError("--window-size must be greater than zero.")
    if stride <= 0:
        raise ValueError("--stride must be greater than zero.")
    if len(sequence) < window_size:
        raise ValueError("Genome sequence is shorter than the requested window size.")

    invalid_bases = unsupported_bases(sequence)
    if invalid_bases and not skip_ambiguous:
        invalid_bases_text = ",".join(sorted(invalid_bases))
        raise ValueError(
            "FASTA contains bases outside A, C, G, and T. "
            f"Unsupported bases: {invalid_bases_text}. "
            "Use --skip-ambiguous to skip affected fragments."
        )

    fragments: list[Fragment] = []
    fragment_id = 0
    for start in range(0, len(sequence) - window_size + 1, stride):
        end = start + window_size
        fragment_sequence = sequence[start:end]
        if unsupported_bases(fragment_sequence):
            continue
        fragments.append(
            Fragment(
                fragment_id=fragment_id,
                start=start,
                end=end,
                sequence=fragment_sequence,
            )
        )
        fragment_id += 1

    if not fragments:
        raise ValueError("No fragments were generated. Check the window size, stride, and ambiguous bases.")

    return fragments


def write_fragment_csv(fragments: list[Fragment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(["fragment_id", "start", "end", "sequence"])
        for fragment in fragments:
            writer.writerow([fragment.fragment_id, fragment.start, fragment.end, fragment.sequence])


def write_fragment_text(fragments: list[Fragment], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for fragment in fragments:
            output_file.write(f"{fragment.sequence}\n")


def write_adjacent_pairs(fragments: list[Fragment], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    number_of_pairs = max(0, len(fragments) - 1)
    with output_path.open("w", encoding="utf-8") as output_file:
        for pair_index in range(number_of_pairs):
            output_file.write(f"{fragments[pair_index].sequence} {fragments[pair_index + 1].sequence}\n")
    return number_of_pairs


def main() -> None:
    arguments = parse_arguments()
    header, sequence = read_fasta(arguments.input)
    fragments = generate_fragments(
        sequence=sequence,
        window_size=arguments.window_size,
        stride=arguments.stride,
        skip_ambiguous=arguments.skip_ambiguous,
    )

    write_fragment_csv(fragments, arguments.output_csv)
    write_fragment_text(fragments, arguments.output_txt)
    number_of_pairs = write_adjacent_pairs(fragments, arguments.output_pairs)

    print(f"FASTA header: {header}")
    print(f"Total genome length: {len(sequence)}")
    print(f"Window size: {arguments.window_size}")
    print(f"Stride: {arguments.stride}")
    print(f"Number of fragments generated: {len(fragments)}")
    print(f"Output CSV path: {arguments.output_csv}")
    print(f"Output TXT path: {arguments.output_txt}")
    print(f"Output pair path: {arguments.output_pairs}")
    print(f"FASTA_HEADER={header}")
    print(f"GENOME_LENGTH={len(sequence)}")
    print(f"WINDOW_SIZE={arguments.window_size}")
    print(f"STRIDE={arguments.stride}")
    print(f"NUMBER_OF_FRAGMENTS={len(fragments)}")
    print(f"OUTPUT_CSV={arguments.output_csv}")
    print(f"OUTPUT_TXT={arguments.output_txt}")
    print(f"OUTPUT_PAIRS={arguments.output_pairs}")
    print(f"NUMBER_OF_PAIRS={number_of_pairs}")
    print("FRAGMENTATION_STATUS=SUCCESS")


if __name__ == "__main__":
    main()
