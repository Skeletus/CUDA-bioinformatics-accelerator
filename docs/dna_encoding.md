# DNA Encoding Optimization

Phase 6 introduces a numeric DNA representation for fixed-length sequence pairs.
The encoded representation is used by a new CUDA Hamming Distance implementation
and benchmarked against the existing char-based CUDA implementation.

## Encoding

DNA uses a four-base alphabet, so each base can be represented by a small integer:

```text
A = 0
C = 1
G = 2
T = 3
```

The current implementation stores each encoded base as `uint8_t`. Uppercase and
lowercase input bases are accepted. Invalid bases are rejected with a clear error
message instead of being silently converted.

## Why uint8_t Encoding Helps

In this phase, `char` and `uint8_t` both use 1 byte per base, so memory size does
not decrease yet. The value of this phase is structural:

* DNA bases are represented as compact numeric symbols instead of ASCII values.
* GPU comparisons operate on explicit biological codes rather than text bytes.
* The code path is ready for future 2-bit packing.
* Alignment algorithms can reuse the encoded values for scoring matrix lookups.

ASCII comparison checks values such as `A = 65`, `C = 67`, `G = 71`, and `T = 84`.
Encoded comparison checks values `0`, `1`, `2`, and `3`. For Hamming Distance the
result is the same for valid uppercase datasets, but the encoded representation is
cleaner for later bioinformatics algorithms.

## Current Scope

Phase 6 still uses fixed-length pair-based text datasets:

```text
ACGTACGT ACGTTCGT
GGGGAAAA GGGAAAAT
TTTTCCCC TTATCCCA
```

The phase does not implement:

* 2-bit packing.
* Ambiguous bases such as `N`.
* Variable-length sequence pairs.
* Needleman-Wunsch.
* Smith-Waterman.

## Encoded Hamming CUDA Flow

`src/hamming_gpu_encoded.cu` performs the following steps:

1. Read pair-based text input.
2. Validate all DNA bases.
3. Convert sequence A and sequence B from `char` arrays to flat `uint8_t` arrays.
4. Compute a CPU encoded Hamming Distance baseline.
5. Allocate GPU memory with `cudaMalloc`.
6. Copy encoded arrays with `cudaMemcpy`.
7. Launch one CUDA thread per sequence pair.
8. Copy distances back to the host.
9. Validate every GPU distance against the CPU baseline.
10. Save `pair_id,distance,similarity` results to CSV.

If validation fails, the program prints the first mismatching pair ID, the CPU
distance, and the GPU distance. The program exits with a non-zero status.

## Compile and Run

From the repository root:

```bash
g++ src/dna_encoding.cpp -O3 -std=c++17 -I src/common -o dna_encoding

nvcc src/hamming_gpu.cu -O3 -std=c++17 -I src/common -o hamming_gpu

nvcc src/hamming_gpu_encoded.cu -O3 -std=c++17 -I src/common -o hamming_gpu_encoded

python scripts/generate_synthetic_dataset.py \
  --num-pairs 10000 \
  --sequence-length 128 \
  --output data/synthetic/synthetic_pairs_128.txt \
  --seed 42

./dna_encoding data/synthetic/synthetic_pairs_128.txt

./hamming_gpu_encoded \
  data/synthetic/synthetic_pairs_128.txt \
  results/hamming/hamming_gpu_encoded_results.csv \
  --repetitions 5
```

The encoded CUDA executable prints machine-readable lines:

```text
FILE_READ_TIME_MS=...
INPUT_VALIDATION_TIME_MS=...
ENCODING_TIME_MS=...
HOST_ALLOCATION_TIME_MS=...
DEVICE_ALLOCATION_TIME_MS=...
H2D_COPY_TIME_MS=...
GPU_KERNEL_TIME_MS=...
D2H_COPY_TIME_MS=...
CPU_REFERENCE_TIME_MS=...
VALIDATION_TIME_MS=...
CSV_WRITE_TIME_MS=...
GPU_TOTAL_TIME_MS=...
END_TO_END_TIME_MS=...
NUMBER_OF_PAIRS=...
SEQUENCE_LENGTH=...
TOTAL_BASES_COMPARED=...
VALIDATION_STATUS=PASSED
OUTPUT_PATH=...
```

## Encoded Timing Breakdown

The encoded GPU implementation now reports the major phases of the executable pipeline separately. This instrumentation was added because the encoded kernel can be fast while the encoded total runtime still looks poor. Without a breakdown, CPU-side encoding, memory copies, validation, and CSV writing can be mistaken for kernel cost.

Timing fields are defined as follows:

* `GPU_KERNEL_TIME_MS`: average CUDA kernel execution time measured with CUDA events. The warm-up launch is not included.
* `GPU_TOTAL_TIME_MS`: device pipeline time, defined as `DEVICE_ALLOCATION_TIME_MS + H2D_COPY_TIME_MS + GPU_KERNEL_TIME_MS + D2H_COPY_TIME_MS`.
* `END_TO_END_TIME_MS`: full executable pipeline time, including file read, input validation, CPU encoding, host allocation, device allocation, memory copies, kernel execution, CPU reference computation, validation, and CSV writing.

The current encoded path still converts text bases to `uint8_t` values on the CPU for every run. For small and medium workloads, that preprocessing can dominate the full executable time even when the CUDA kernel itself is efficient. Validation and CSV writing are also intentionally measured separately because they are correctness and reporting costs, not GPU kernel performance.

Structural checks needed while loading the pair file, such as equal sequence lengths, are part of the file-read phase. `INPUT_VALIDATION_TIME_MS` measures the dedicated validation pass over loaded bases and final dataset shape checks.

This measurement should guide optimization work before algorithm changes. Useful next steps include moving encoding to the GPU, reusing encoded data across multiple kernels, storing encoded datasets on disk, adding 2-bit packing, batching work with CUDA streams, and later applying the encoded representation to Smith-Waterman or other alignment kernels.

## Benchmark Methodology

The Phase 6 benchmark compares:

* `src/hamming_gpu.cu`: char-based CUDA Hamming Distance.
* `src/hamming_gpu_encoded.cu`: encoded `uint8_t` CUDA Hamming Distance.

The default benchmark matrix is:

```text
sequence_lengths = [64, 128, 256, 512, 1024]
num_pairs_values = [1000, 10000, 100000, 1000000]
repetitions = 5
```

The benchmark script generates synthetic datasets, compiles both CUDA programs,
runs both implementations on identical inputs, parses their `KEY=value` output,
and saves:

```text
benchmarks/dna_encoding_benchmark_results.csv
```

Run it with:

```bash
python benchmarks/run_encoding_benchmark.py
```

Required benchmark columns include kernel time, total time, encoding overhead,
speedup, validation status, and total bases compared.

## Charts

Generate Phase 6 charts with:

```bash
python scripts/plot_encoding_benchmarks.py
```

Charts are saved to:

```text
assets/benchmark_charts/encoding/
```

Generated files:

```text
char_vs_encoded_kernel_time.png
char_vs_encoded_total_time.png
encoded_kernel_speedup.png
encoded_total_speedup.png
encoding_overhead.png
```

## Limitations

The current encoding is intentionally simple:

* It still uses 1 byte per base.
* It rejects ambiguous DNA bases such as `N`.
* It assumes fixed-length sequence pairs.
* It stores sequence A and sequence B as separate flat arrays.

## Future Work

The next memory-focused step is 2-bit packing, where four DNA bases can be stored
in one byte. Later phases can also add ambiguous-base handling, variable-length
sequence support, and scoring matrix based alignment algorithms.
