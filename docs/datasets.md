# Real Dataset Integration

Phase 7 adds real genomic data to the CUDA Bioinformatics Accelerator. Earlier phases use synthetic DNA pairs because they are easy to control and useful for correctness testing. Real genomic fragments make the benchmark closer to a biological workflow while keeping the algorithmic scope limited to fixed-length Hamming Distance.

## Dataset

The first real dataset is the SARS-CoV-2 reference genome:

```text
Dataset: SARS-CoV-2 reference genome
NCBI RefSeq accession: NC_045512.2
Format: FASTA
```

The download script saves the raw FASTA file to:

```text
data/raw/sars_cov_2_NC_045512_2.fasta
```

## FASTA Parsing

The Phase 7 parser reads a single-record FASTA file. It stores the first header line without the leading `>` character, removes blank lines, concatenates all sequence lines, and converts the sequence to uppercase.

Only these DNA bases are supported by default:

```text
A, C, G, T
```

If the genome contains ambiguous bases such as `N`, the fragmentation script can either reject the input with a clear error or skip fragments containing unsupported bases when `--skip-ambiguous` is provided.

## Sliding Window Fragmentation

The genome is split into fixed-length fragments with a sliding window.

```text
window size = number of bases in each fragment
stride      = number of bases to move the window after each fragment
fragment    = one fixed-length genome window
```

For the default Phase 7 settings:

```text
Window size: 128
Stride: 32
```

The first windows are:

```text
fragment 0: bases 0..127
fragment 1: bases 32..159
fragment 2: bases 64..191
```

Fixed-length fragments are used because the current CPU and CUDA Hamming implementations require both sequences in each pair to have the same length.

## Adjacent Pairs

The pair generator compares neighboring genome windows:

```text
fragment_0 fragment_1
fragment_1 fragment_2
fragment_2 fragment_3
```

This creates real genomic sequence pairs while preserving the existing pair-based input format:

```text
sequenceA sequenceB
sequenceA sequenceB
```

The same pair file can be passed directly to:

```text
hamming_cpu
hamming_gpu
hamming_gpu_encoded
```

## Commands

Download the SARS-CoV-2 FASTA file:

```bash
python scripts/download_datasets.py \
  --dataset sars-cov-2 \
  --output data/raw/sars_cov_2_NC_045512_2.fasta
```

Generate fragments and adjacent pairs:

```bash
python scripts/fragment_fasta.py \
  --input data/raw/sars_cov_2_NC_045512_2.fasta \
  --output-csv data/processed/sars_cov_2_fragments_128.csv \
  --output-txt data/sars_cov_2_fragments_128.txt \
  --output-pairs data/processed/sars_cov_2_pairs_128_stride_32.txt \
  --window-size 128 \
  --stride 32 \
  --skip-ambiguous
```

Run the real dataset benchmark:

```bash
python benchmarks/run_real_dataset_benchmark.py \
  --window-size 128 \
  --stride 32 \
  --repetitions 5
```

Generate charts:

```bash
python scripts/plot_real_dataset_benchmark.py
```

## Output Files

Raw FASTA:

```text
data/raw/sars_cov_2_NC_045512_2.fasta
```

Fragment CSV:

```text
data/processed/sars_cov_2_fragments_128.csv
```

Fragment text file:

```text
data/sars_cov_2_fragments_128.txt
```

Adjacent pair file:

```text
data/processed/sars_cov_2_pairs_128_stride_32.txt
```

Benchmark CSV:

```text
benchmarks/real_dataset_hamming_benchmark_results.csv
```

Benchmark charts:

```text
assets/benchmark_charts/real_dataset/
```

## Current Limitations

- Only `A`, `C`, `G`, and `T` bases are supported by default.
- Ambiguous bases are rejected unless `--skip-ambiguous` is used.
- Fragmentation uses only fixed-length windows.
- Pair generation uses a simple adjacent-fragment strategy.
- Phase 7 still uses Hamming Distance, not full sequence alignment.

## Future Work

- Compare each query fragment against a reference database.
- Support ambiguous base `N`.
- Support variable-length sequences.
- Use Smith-Waterman for more realistic local alignment.
