# Needleman-Wunsch CUDA Implementation

## Overview

Phase 10 adds a CUDA prototype for Needleman-Wunsch global sequence alignment. The goal is correctness, measurable timing, and a clear introduction to dynamic programming on the GPU before implementing Smith-Waterman CUDA.

The implementation compares fixed-length sequence pairs from the existing pair-based text format and validates every GPU score against the Phase 8 CPU Needleman-Wunsch reference.

Default scoring:

```text
Match: +2
Mismatch: -1
Gap: -2
```

## Why Needleman-Wunsch Is Harder Than Hamming Distance

Hamming Distance compares equal-length sequences position by position. Each pair can be processed independently with a simple loop, so GPU parallelism is straightforward.

Needleman-Wunsch uses dynamic programming. Each DP cell depends on neighboring cells:

```text
top
left
top-left
```

Because of these dependencies, the full matrix cannot be computed in one fully independent parallel pass.

## CPU Reference

The CUDA executable uses the reusable CPU implementation in:

```text
src/common/needleman_wunsch.h
```

The CPU score function is used as the correctness reference. GPU scores must match CPU scores exactly unless validation is explicitly skipped.

## CUDA Implementation Strategy

The prototype supports two implementation modes:

```text
baseline
wavefront
```

The baseline mode computes one sequence pair per block using a single thread for the DP recurrence. This mode is simple and useful as a GPU control path.

The wavefront mode computes one sequence pair per block and parallelizes cells along anti-diagonals. This is the main Phase 10 implementation.

## Wavefront Parallelism

Cells on the same anti-diagonal do not depend on each other. Once the previous anti-diagonal has been computed, the current anti-diagonal can be processed in parallel.

For a matrix with sequence lengths `m` and `n`, the kernel computes diagonals from:

```text
diagonal = 2 to m + n
```

Each CUDA thread computes one or more cells on the current anti-diagonal. A block-level synchronization barrier is used between diagonals.

## Anti-Diagonal Computation

For a given diagonal:

```text
row + col = diagonal
```

The valid row range is:

```text
startRow = max(1, diagonal - sequenceLengthB)
endRow   = min(sequenceLengthA, diagonal - 1)
```

Each thread maps a cell offset to:

```text
row = startRow + cellOffset
col = diagonal - row
```

Then the standard Needleman-Wunsch recurrence is applied.

## Shared Memory Usage

The DP matrix is stored in shared memory for each CUDA block. This keeps the first prototype simple and avoids allocating one global DP matrix per sequence pair.

The current prototype supports sequence lengths up to:

```text
MAX_SUPPORTED_SEQUENCE_LENGTH = 64
```

A 65 x 65 integer DP matrix uses about 16.9 KB of shared memory per block. Larger sequence lengths require a different strategy, such as global memory DP buffers or tiled wavefront computation.

## Synchronization with __syncthreads()

The wavefront kernel uses `__syncthreads()` after initialization and after each anti-diagonal. This ensures all cells required by the next diagonal are complete before any thread continues.

This synchronization is block-local, which is why the prototype uses one CUDA block per sequence pair.

## Memory Layout

Input sequences use the existing flat fixed-length layout:

```text
sequenceA_flat = [pair0_A][pair1_A][pair2_A]...
sequenceB_flat = [pair0_B][pair1_B][pair2_B]...
```

The offset is:

```cpp
int sequenceOffset = pairIndex * sequenceLength;
```

The output array stores one final global alignment score per pair:

```text
scores[pairIndex] = dp[m][n]
```

Traceback matrices and aligned strings are not generated in Phase 10.

## Timing Methodology

The CUDA executable reports:

```text
H2D_COPY_TIME_MS
GPU_KERNEL_TIME_MS
D2H_COPY_TIME_MS
GPU_TOTAL_TIME_MS
CPU_REFERENCE_TIME_MS
VALIDATION_TIME_MS
```

`GPU_KERNEL_TIME_MS` is measured with CUDA events and averaged across repetitions.

`GPU_TOTAL_TIME_MS` is defined as:

```text
H2D_COPY_TIME_MS + GPU_KERNEL_TIME_MS + D2H_COPY_TIME_MS
```

CPU reference and validation time are reported separately and are not included in `GPU_TOTAL_TIME_MS`.

## Correctness Validation

Validation is enabled by default. For every sequence pair:

```text
GPU score == CPU score
```

If validation fails, the executable prints:

```text
VALIDATION_STATUS=FAILED
FIRST_MISMATCH_PAIR_ID=...
CPU_SCORE=...
GPU_SCORE=...
```

If validation is explicitly skipped with `--skip-validation`, the executable prints:

```text
VALIDATION_STATUS=SKIPPED
```

## Benchmark Methodology

The benchmark script is:

```text
benchmarks/run_needleman_wunsch_gpu_benchmark.py
```

Default workloads:

```text
sequence_lengths = [8, 16, 32, 64]
num_pairs_values = [10, 100, 1000]
repetitions = 5
```

The benchmark compiles the CPU and GPU executables, generates synthetic fixed-length DNA datasets, runs both implementations, parses machine-readable metrics, validates GPU correctness, and saves:

```text
benchmarks/needleman_wunsch_gpu_benchmark_results.csv
```

## Results

Generate benchmark charts with:

```bash
python scripts/plot_needleman_wunsch_gpu_benchmark.py
```

Charts are saved to:

```text
assets/benchmark_charts/needleman_wunsch_gpu/
```

Generated charts:

```text
nw_gpu_cpu_vs_gpu_time.png
nw_gpu_kernel_speedup.png
nw_gpu_total_speedup.png
nw_gpu_cells_per_second.png
nw_gpu_h2d_kernel_d2h_breakdown.png
nw_gpu_scaling_by_sequence_length.png
```

## Limitations

Phase 10 is a correctness-first CUDA prototype. Current limitations:

* Fixed-length sequence pairs per input file.
* Shared-memory sequence length limit of 64 bases per sequence side.
* Score-only output.
* No traceback reconstruction.
* No CUDA streams batching.
* No 2-bit packing.
* Not optimized for very large sequence lengths.

## Future Work

Future work includes:

* Global memory DP buffers for longer sequences.
* Tiled wavefront computation.
* Rolling diagonal memory layouts.
* Batched processing across many sequence pairs.
* CUDA streams for overlapping transfers and computation.
* Smith-Waterman CUDA using the same wavefront structure with local-alignment reset behavior.
* Real genomic fragment alignment.
* Multi-GPU batch alignment.
