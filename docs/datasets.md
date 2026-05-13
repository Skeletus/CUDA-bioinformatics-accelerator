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

## Real Dataset Pair Generation Modes

Phase 7 now supports several pair generation modes. Each mode writes the same Hamming-compatible pair format:

```text
sequenceA sequenceB
sequenceA sequenceB
```

The pair files can be passed directly to:

```text
hamming_cpu
hamming_gpu
hamming_gpu_encoded
```

### adjacent

Adjacent mode compares neighboring genome windows:

```text
fragment_0 fragment_1
fragment_1 fragment_2
fragment_2 fragment_3
```

This mode is biologically intuitive because nearby windows come from neighboring genome positions. It is useful as a baseline, but it creates only `number_of_fragments - 1` pairs. For the SARS-CoV-2 reference genome with a 128-base window and stride 32, that workload is too small for meaningful GPU scalability analysis.

### all_vs_all

All-vs-all mode compares every fragment against every other fragment while excluding self-pairs:

```text
fragment_i fragment_j
i != j
```

For `N` fragments, this creates `N * (N - 1)` possible pairs. This is much larger than adjacent mode and is better for GPU scalability studies. Use `--max-pairs` to prevent runaway output sizes.

### sampled

Sampled mode randomly samples a fixed number of target fragments for each source fragment. The default is:

```text
--pairs-per-fragment 64
```

For each source fragment, the generator samples unique targets and never pairs a fragment with itself. Pair `(i, j)` and pair `(j, i)` can both appear. This mode is the recommended default for meaningful GPU benchmarking because it provides a larger workload than adjacent mode without expanding as aggressively as all-vs-all mode.

### mutated_queries

Mutated queries mode uses real SARS-CoV-2 fragments as sources and creates synthetic mutated copies as query targets. For each base, the generator mutates with probability `--mutation-rate`. Mutations use only `A`, `C`, `G`, and `T`, and a base is never replaced with itself.

This mode is useful for correctness and similarity experiments because the expected difference is controlled by the mutation rate. When `--pairs-per-fragment` is omitted in `fragment_fasta.py`, it generates one mutated pair per fragment. It can generate multiple mutated copies per fragment when `--pairs-per-fragment` is greater than 1.

### Reproducibility and Size Limits

The `--seed` argument makes sampled and mutated workloads reproducible. The same input FASTA, window size, stride, pairing mode, and seed produce the same pair file.

The `--max-pairs` argument caps generated pairs for large modes. If generation reaches the cap, the script stops writing pairs, prints a warning, and reports:

```text
PAIR_GENERATION_TRUNCATED=true
```

This keeps all-vs-all and sampled workloads practical in Google Colab.

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
  --output-pairs data/processed/sars_cov_2_pairs_128_stride_32_adjacent.txt \
  --window-size 128 \
  --stride 32 \
  --pairing-mode adjacent \
  --skip-ambiguous
```

Generate sampled pairs, the recommended default for GPU benchmarking:

```bash
python scripts/fragment_fasta.py \
  --input data/raw/sars_cov_2_NC_045512_2.fasta \
  --output-csv data/processed/sars_cov_2_fragments_128.csv \
  --output-txt data/sars_cov_2_fragments_128.txt \
  --output-pairs data/processed/sars_cov_2_pairs_128_stride_32_sampled.txt \
  --window-size 128 \
  --stride 32 \
  --pairing-mode sampled \
  --pairs-per-fragment 64 \
  --max-pairs 1000000 \
  --seed 42 \
  --skip-ambiguous
```

Run single-mode real dataset benchmarks:

```bash
python benchmarks/run_real_dataset_benchmark.py \
  --window-size 128 \
  --stride 32 \
  --pairing-mode adjacent \
  --repetitions 5

python benchmarks/run_real_dataset_benchmark.py \
  --window-size 128 \
  --stride 32 \
  --pairing-mode sampled \
  --pairs-per-fragment 64 \
  --max-pairs 1000000 \
  --seed 42 \
  --repetitions 5

python benchmarks/run_real_dataset_benchmark.py \
  --window-size 128 \
  --stride 32 \
  --pairing-mode all_vs_all \
  --max-pairs 1000000 \
  --repetitions 5

python benchmarks/run_real_dataset_benchmark.py \
  --window-size 128 \
  --stride 32 \
  --pairing-mode mutated_queries \
  --pairs-per-fragment 4 \
  --mutation-rate 0.05 \
  --max-pairs 1000000 \
  --seed 42 \
  --repetitions 5
```

Run all pairing modes:

```bash
python benchmarks/run_real_dataset_pairing_modes_benchmark.py
```

Generate charts:

```bash
python scripts/plot_real_dataset_benchmark.py
python scripts/plot_real_dataset_pairing_modes.py
python scripts/plot_encoded_timing_breakdown.py
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

Mode-specific pair files:

```text
data/processed/sars_cov_2_pairs_128_stride_32_adjacent.txt
data/processed/sars_cov_2_pairs_128_stride_32_all_vs_all.txt
data/processed/sars_cov_2_pairs_128_stride_32_sampled.txt
data/processed/sars_cov_2_pairs_128_stride_32_mutated_queries.txt
```

Benchmark CSV:

```text
benchmarks/real_dataset_hamming_benchmark_results.csv
benchmarks/real_dataset_pairing_modes_benchmark_results.csv
```

Benchmark charts:

```text
assets/benchmark_charts/real_dataset/
assets/benchmark_charts/real_dataset_pairing_modes/
assets/benchmark_charts/encoded_timing_breakdown/
```

## Encoded Timing Breakdown

The encoded GPU benchmark now reports detailed timing fields for real dataset runs. This was added because previous results showed good encoded kernel-level performance, but poor encoded total runtime. The breakdown makes it possible to see whether the overhead comes from CPU-side encoding, device allocation, memory transfers, validation, CSV writing, or other pipeline work.

The key timing definitions are:

```text
GPU_KERNEL_TIME_MS = average CUDA kernel time only
GPU_TOTAL_TIME_MS = DEVICE_ALLOCATION_TIME_MS + H2D_COPY_TIME_MS + GPU_KERNEL_TIME_MS + D2H_COPY_TIME_MS
END_TO_END_TIME_MS = full executable pipeline time
```

`GPU_TOTAL_TIME_MS` is intentionally limited to the GPU device pipeline. It does not include FASTA pair file reading, CPU encoding, CPU reference computation, validation, or CSV writing. `END_TO_END_TIME_MS` includes all of those phases and is the right number for full executable cost.

The real dataset benchmark CSV includes encoded timing columns such as:

```text
encoded_encoding_time_ms
encoded_h2d_copy_time_ms
encoded_gpu_kernel_time_ms
encoded_d2h_copy_time_ms
encoded_validation_time_ms
encoded_csv_write_time_ms
encoded_gpu_total_time_ms
encoded_end_to_end_time_ms
```

Generate the breakdown charts with:

```bash
python scripts/plot_encoded_timing_breakdown.py
```

The charts are saved to:

```text
assets/benchmark_charts/encoded_timing_breakdown/
```

This instrumentation should be used before optimizing the encoded path. It helps evaluate future changes such as GPU-side encoding, reusing encoded arrays across multiple kernels, storing encoded datasets on disk, adding 2-bit packing, batching with CUDA streams, and eventually feeding encoded data into Smith-Waterman.

## Current Limitations

- Only `A`, `C`, `G`, and `T` bases are supported by default.
- Ambiguous bases are rejected unless `--skip-ambiguous` is used.
- Fragmentation uses only fixed-length windows.
- Phase 7 still uses Hamming Distance, not full sequence alignment.
- Needleman-Wunsch, Smith-Waterman, and 2-bit packing are intentionally not part of Phase 7.

## Future Work

- Compare each query fragment against a reference database.
- Support ambiguous base `N`.
- Support variable-length sequences.
- Use Smith-Waterman for more realistic local alignment.
