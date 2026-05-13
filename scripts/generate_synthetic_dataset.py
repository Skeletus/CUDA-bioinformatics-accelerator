#!/usr/bin/env python3

import argparse
import random
from pathlib import Path


DNA_ALPHABET = "ACGT"


def generate_sequence(sequence_length: int, random_generator: random.Random) -> str:
    return "".join(random_generator.choice(DNA_ALPHABET) for _ in range(sequence_length))


def generate_dataset(num_pairs: int, sequence_length: int, output_path: Path, seed: int) -> None:
    random_generator = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as output_file:
        for _ in range(num_pairs):
            first_sequence = generate_sequence(sequence_length, random_generator)
            second_sequence = generate_sequence(sequence_length, random_generator)
            output_file.write(f"{first_sequence} {second_sequence}\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic fixed-length DNA sequence pairs.")
    parser.add_argument("--num-pairs", type=int, required=True, help="Number of sequence pairs to generate.")
    parser.add_argument("--sequence-length", type=int, required=True, help="Length of each DNA sequence.")
    parser.add_argument("--output", type=Path, required=True, help="Output dataset path.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible generation.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.num_pairs <= 0:
        raise ValueError("--num-pairs must be greater than zero.")
    if arguments.sequence_length <= 0:
        raise ValueError("--sequence-length must be greater than zero.")

    generate_dataset(
        num_pairs=arguments.num_pairs,
        sequence_length=arguments.sequence_length,
        output_path=arguments.output,
        seed=arguments.seed,
    )

    print(f"Generated {arguments.num_pairs} pairs")
    print(f"Sequence length: {arguments.sequence_length}")
    print(f"Output path: {arguments.output}")


if __name__ == "__main__":
    main()
