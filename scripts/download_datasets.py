#!/usr/bin/env python3

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path


SARS_COV_2_ACCESSION = "NC_045512.2"
SARS_COV_2_DATASET_NAME = "sars-cov-2"
SARS_COV_2_FASTA_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    "?db=nuccore&id=NC_045512.2&rettype=fasta&retmode=text"
)
DEFAULT_OUTPUT_PATH = Path("data/raw/sars_cov_2_NC_045512_2.fasta")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download real genomic datasets used by this project.")
    parser.add_argument(
        "--dataset",
        choices=[SARS_COV_2_DATASET_NAME],
        default=SARS_COV_2_DATASET_NAME,
        help="Dataset to download.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output FASTA path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing output file.",
    )
    return parser.parse_args()


def print_dataset_metadata(output_path: Path, status: str) -> None:
    print(f"DATASET_NAME={SARS_COV_2_DATASET_NAME}")
    print(f"ACCESSION={SARS_COV_2_ACCESSION}")
    print(f"OUTPUT_PATH={output_path}")
    print(f"DOWNLOAD_STATUS={status}")


def download_text_file(url: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "CUDA-Bioinformatic/phase-7"})

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            content = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"HTTP error while downloading FASTA: {error.code} {error.reason}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Network error while downloading FASTA: {error.reason}") from error
    except TimeoutError as error:
        raise RuntimeError("Timed out while downloading FASTA.") from error

    if not content.startswith(">"):
        raise RuntimeError("Downloaded content does not look like FASTA data.")
    if SARS_COV_2_ACCESSION not in content.splitlines()[0]:
        raise RuntimeError("Downloaded FASTA header does not contain the expected accession.")

    output_path.write_text(content, encoding="utf-8")


def main() -> int:
    arguments = parse_arguments()

    if arguments.output.exists() and not arguments.force:
        print_dataset_metadata(arguments.output, "ALREADY_EXISTS")
        return 0

    try:
        download_text_file(SARS_COV_2_FASTA_URL, arguments.output)
    except RuntimeError as error:
        print(f"Error: {error}", file=sys.stderr)
        print_dataset_metadata(arguments.output, "FAILED")
        return 1

    print_dataset_metadata(arguments.output, "SUCCESS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
