#!/usr/bin/env python3

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_DNA_BASES = {"A", "C", "G", "T"}
PAIRING_MODES = ("adjacent", "all_vs_all", "sampled", "mutated_queries")
DEFAULT_INPUT_PATH = Path("data/raw/sars_cov_2_NC_045512_2.fasta")
DEFAULT_OUTPUT_CSV_PATH = Path("data/processed/sars_cov_2_fragments_128.csv")
DEFAULT_OUTPUT_TXT_PATH = Path("data/sars_cov_2_fragments_128.txt")


@dataclass(frozen=True)
class Fragment:
    fragment_id: int
    start: int
    end: int
    sequence: str


@dataclass
class PairGenerationStats:
    number_of_pairs: int
    truncated: bool = False
    total_bases_processed: int = 0
    total_bases_mutated: int = 0

    @property
    def observed_mutation_rate(self) -> float:
        if self.total_bases_processed == 0:
            return 0.0
        return self.total_bases_mutated / self.total_bases_processed


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parse a FASTA genome and generate fixed-length DNA fragments.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH, help="Input FASTA path.")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV_PATH, help="Output fragment CSV path.")
    parser.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT_PATH, help="Output fragment text path.")
    parser.add_argument(
        "--output-pairs",
        type=Path,
        default=None,
        help="Output sequence pair path. Defaults to a mode-specific path under data/processed.",
    )
    parser.add_argument("--window-size", type=int, default=128, help="Sliding window size.")
    parser.add_argument("--stride", type=int, default=32, help="Sliding window stride.")
    parser.add_argument(
        "--pairing-mode",
        choices=PAIRING_MODES,
        default="adjacent",
        help="Pair generation mode.",
    )
    parser.add_argument(
        "--pairs-per-fragment",
        type=int,
        default=None,
        help="Number of sampled or mutated pairs generated per source fragment.",
    )
    parser.add_argument("--max-pairs", type=int, default=1_000_000, help="Maximum number of pairs to write.")
    parser.add_argument(
        "--mutation-rate",
        type=float,
        default=0.05,
        help="Per-base mutation probability for mutated_queries mode.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible pair generation.")
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


def default_pair_path(window_size: int, stride: int, pairing_mode: str) -> Path:
    return Path(f"data/processed/sars_cov_2_pairs_{window_size}_stride_{stride}_{pairing_mode}.txt")


def validate_pair_arguments(pairs_per_fragment: int, max_pairs: int, mutation_rate: float) -> None:
    if pairs_per_fragment <= 0:
        raise ValueError("--pairs-per-fragment must be greater than zero.")
    if max_pairs <= 0:
        raise ValueError("--max-pairs must be greater than zero.")
    if mutation_rate < 0.0 or mutation_rate > 1.0:
        raise ValueError("--mutation-rate must be between 0.0 and 1.0.")


def write_adjacent_pairs_limited(fragments: list[Fragment], output_path: Path, max_pairs: int) -> PairGenerationStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_pairs = max(0, len(fragments) - 1)
    number_of_pairs = min(expected_pairs, max_pairs)
    with output_path.open("w", encoding="utf-8") as output_file:
        for pair_index in range(number_of_pairs):
            output_file.write(f"{fragments[pair_index].sequence} {fragments[pair_index + 1].sequence}\n")
    return PairGenerationStats(number_of_pairs=number_of_pairs, truncated=expected_pairs > max_pairs)


def write_all_vs_all_pairs(fragments: list[Fragment], output_path: Path, max_pairs: int) -> PairGenerationStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    expected_pairs = len(fragments) * max(0, len(fragments) - 1)
    number_of_pairs = 0

    with output_path.open("w", encoding="utf-8") as output_file:
        for fragment_i in fragments:
            for fragment_j in fragments:
                if fragment_i.fragment_id == fragment_j.fragment_id:
                    continue
                if number_of_pairs >= max_pairs:
                    return PairGenerationStats(number_of_pairs=number_of_pairs, truncated=True)
                output_file.write(f"{fragment_i.sequence} {fragment_j.sequence}\n")
                number_of_pairs += 1

    return PairGenerationStats(number_of_pairs=number_of_pairs, truncated=expected_pairs > max_pairs)


def write_sampled_pairs(
    fragments: list[Fragment],
    output_path: Path,
    pairs_per_fragment: int,
    max_pairs: int,
    seed: int,
) -> PairGenerationStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(seed)
    number_of_pairs = 0
    available_per_fragment = max(0, len(fragments) - 1)
    expected_pairs = len(fragments) * min(pairs_per_fragment, available_per_fragment)

    with output_path.open("w", encoding="utf-8") as output_file:
        for source_index, source_fragment in enumerate(fragments):
            sample_count = min(pairs_per_fragment, available_per_fragment)
            candidate_indices = [index for index in range(len(fragments)) if index != source_index]
            sampled_indices = random_generator.sample(candidate_indices, sample_count)
            for target_index in sampled_indices:
                if number_of_pairs >= max_pairs:
                    return PairGenerationStats(number_of_pairs=number_of_pairs, truncated=True)
                output_file.write(f"{source_fragment.sequence} {fragments[target_index].sequence}\n")
                number_of_pairs += 1

    return PairGenerationStats(number_of_pairs=number_of_pairs, truncated=expected_pairs > max_pairs)


def mutate_sequence(sequence: str, mutation_rate: float, random_generator: random.Random) -> tuple[str, int]:
    mutated_bases: list[str] = []
    number_of_mutations = 0
    for base in sequence:
        if random_generator.random() < mutation_rate:
            replacement_options = [candidate for candidate in sorted(SUPPORTED_DNA_BASES) if candidate != base]
            mutated_bases.append(random_generator.choice(replacement_options))
            number_of_mutations += 1
        else:
            mutated_bases.append(base)
    return "".join(mutated_bases), number_of_mutations


def write_mutated_query_pairs(
    fragments: list[Fragment],
    output_path: Path,
    pairs_per_fragment: int,
    max_pairs: int,
    mutation_rate: float,
    seed: int,
) -> PairGenerationStats:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random_generator = random.Random(seed)
    number_of_pairs = 0
    total_bases_processed = 0
    total_bases_mutated = 0
    expected_pairs = len(fragments) * pairs_per_fragment

    with output_path.open("w", encoding="utf-8") as output_file:
        for fragment in fragments:
            for _ in range(pairs_per_fragment):
                if number_of_pairs >= max_pairs:
                    return PairGenerationStats(
                        number_of_pairs=number_of_pairs,
                        truncated=True,
                        total_bases_processed=total_bases_processed,
                        total_bases_mutated=total_bases_mutated,
                    )
                mutated_sequence, mutation_count = mutate_sequence(
                    fragment.sequence,
                    mutation_rate,
                    random_generator,
                )
                output_file.write(f"{fragment.sequence} {mutated_sequence}\n")
                number_of_pairs += 1
                total_bases_processed += len(fragment.sequence)
                total_bases_mutated += mutation_count

    return PairGenerationStats(
        number_of_pairs=number_of_pairs,
        truncated=expected_pairs > max_pairs,
        total_bases_processed=total_bases_processed,
        total_bases_mutated=total_bases_mutated,
    )


def write_pairs(
    fragments: list[Fragment],
    output_path: Path,
    pairing_mode: str,
    pairs_per_fragment: int,
    max_pairs: int,
    mutation_rate: float,
    seed: int,
) -> PairGenerationStats:
    if pairing_mode == "adjacent":
        return write_adjacent_pairs_limited(fragments, output_path, max_pairs)
    if pairing_mode == "all_vs_all":
        return write_all_vs_all_pairs(fragments, output_path, max_pairs)
    if pairing_mode == "sampled":
        return write_sampled_pairs(fragments, output_path, pairs_per_fragment, max_pairs, seed)
    if pairing_mode == "mutated_queries":
        return write_mutated_query_pairs(fragments, output_path, pairs_per_fragment, max_pairs, mutation_rate, seed)
    raise ValueError(f"Unsupported pairing mode: {pairing_mode}")


def main() -> None:
    arguments = parse_arguments()
    pairs_per_fragment = arguments.pairs_per_fragment
    if pairs_per_fragment is None:
        pairs_per_fragment = 1 if arguments.pairing_mode == "mutated_queries" else 64
    validate_pair_arguments(pairs_per_fragment, arguments.max_pairs, arguments.mutation_rate)
    output_pairs = arguments.output_pairs or default_pair_path(
        arguments.window_size,
        arguments.stride,
        arguments.pairing_mode,
    )
    header, sequence = read_fasta(arguments.input)
    fragments = generate_fragments(
        sequence=sequence,
        window_size=arguments.window_size,
        stride=arguments.stride,
        skip_ambiguous=arguments.skip_ambiguous,
    )

    write_fragment_csv(fragments, arguments.output_csv)
    write_fragment_text(fragments, arguments.output_txt)
    pair_stats = write_pairs(
        fragments=fragments,
        output_path=output_pairs,
        pairing_mode=arguments.pairing_mode,
        pairs_per_fragment=pairs_per_fragment,
        max_pairs=arguments.max_pairs,
        mutation_rate=arguments.mutation_rate,
        seed=arguments.seed,
    )

    print(f"FASTA header: {header}")
    print(f"Total genome length: {len(sequence)}")
    print(f"Window size: {arguments.window_size}")
    print(f"Stride: {arguments.stride}")
    print(f"Pairing mode: {arguments.pairing_mode}")
    print(f"Number of fragments generated: {len(fragments)}")
    print(f"Output CSV path: {arguments.output_csv}")
    print(f"Output TXT path: {arguments.output_txt}")
    print(f"Output pair path: {output_pairs}")
    if pair_stats.truncated:
        print(f"Warning: pair generation reached --max-pairs={arguments.max_pairs}; output was truncated.")
    print(f"FASTA_HEADER={header}")
    print(f"GENOME_LENGTH={len(sequence)}")
    print(f"WINDOW_SIZE={arguments.window_size}")
    print(f"STRIDE={arguments.stride}")
    print(f"PAIRING_MODE={arguments.pairing_mode}")
    print(f"PAIRS_PER_FRAGMENT={pairs_per_fragment}")
    print(f"MAX_PAIRS={arguments.max_pairs}")
    print(f"MUTATION_RATE={arguments.mutation_rate}")
    print(f"SEED={arguments.seed}")
    print(f"NUMBER_OF_FRAGMENTS={len(fragments)}")
    print(f"OUTPUT_CSV={arguments.output_csv}")
    print(f"OUTPUT_TXT={arguments.output_txt}")
    print(f"OUTPUT_PAIRS={output_pairs}")
    print(f"NUMBER_OF_PAIRS={pair_stats.number_of_pairs}")
    print(f"PAIR_GENERATION_TRUNCATED={str(pair_stats.truncated).lower()}")
    if arguments.pairing_mode == "mutated_queries":
        print(f"TOTAL_BASES_PROCESSED={pair_stats.total_bases_processed}")
        print(f"TOTAL_BASES_MUTATED={pair_stats.total_bases_mutated}")
        print(f"OBSERVED_MUTATION_RATE={pair_stats.observed_mutation_rate:.8f}")
    print("PAIR_GENERATION_STATUS=SUCCESS")
    print("FRAGMENTATION_STATUS=SUCCESS")


if __name__ == "__main__":
    main()
