# Smith-Waterman CUDA Implementation

## Overview

Phase 11 adds a CUDA prototype for Smith-Waterman local sequence alignment. The implementation computes one maximum local alignment score per fixed-length sequence pair, validates every GPU result against the CPU reference, and reports timing for host-device copies and kernel execution.

## Relationship to Needleman-Wunsch CUDA

Smith-Waterman uses the same dynamic programming dependency structure as Needleman-Wunsch:

```text
top
left
top-left
```

Because cells on the same anti-diagonal do not depend on each other, wavefront parallelism remains applicable.

## Why Smith-Waterman Is Different

Needleman-Wunsch:

* performs global alignment
* initializes the first row and column with gap penalties
* returns `dp[m][n]`

Smith-Waterman:

* performs local alignment
* initializes the first row and column to zero
* resets negative scores to zero
* returns the maximum value anywhere in the DP matrix

## CPU Reference

The CUDA executable reuses the Phase 9 reference function in:

```text
src/common/smith_waterman.h
```

The reusable API is:

```cpp
int computeSmithWatermanScore(
    const std::string& sequenceA,
    const std::string& sequenceB,
    int matchScore,
    int mismatchPenalty,
    int gapPenalty
);
```

## CUDA Implementation Strategy

The prototype supports:

```text
baseline
wavefront
```

Both modes use one CUDA block per sequence pair and a shared-memory DP matrix. The baseline mode computes each pair with one thread and exists as a simple control path. The wavefront mode parallelizes cells along anti-diagonals.

## Wavefront Parallelism

For each pair, the kernel processes DP anti-diagonals after initializing the zero-valued borders. Cells on a given diagonal can run in parallel because their dependencies lie only on earlier diagonals.

## Anti-Diagonal Computation

For the interior DP cells, the kernel evaluates diagonals from:

```text
diagonal = 2 to sequenceLengthA + sequenceLengthB
```

For each diagonal:

```text
startRow = max(1, diagonal - sequenceLengthB)
endRow   = min(sequenceLengthA, diagonal - 1)
```

Each thread maps a cell offset to:

```text
row = startRow + cellOffset
col = diagonal - row
```

## Shared Memory Usage

The first GPU prototype stores one full DP matrix per block in shared memory. The supported maximum is:

```text
MAX_SUPPORTED_SEQUENCE_LENGTH = 64
DP_MEMORY_MODE = shared
```

For sequence length 64, the main DP matrix uses approximately:

```text
65 x 65 x 4 bytes ≈ 16.9 KB
```

The wavefront mode also reserves a small shared-memory array for per-thread maxima before reduction.

## Local Alignment Reset Behavior

Smith-Waterman applies:

```text
dp[i][j] = max(0, scoreDiagonal, scoreUp, scoreLeft)
```

The zero candidate is what allows alignments to restart locally instead of forcing the entire sequence prefix to contribute to the score.

## Maximum Score Reduction

Unlike Needleman-Wunsch, Smith-Waterman does not return the bottom-right DP cell. It returns the largest cell value anywhere in the matrix.

In the wavefront kernel, each thread tracks the largest cell it computed. After all diagonals finish, those thread-local maxima are written to shared memory and reduced to one block-level maximum score:

```text
scores[pairIndex] = maximum local alignment score
```

## Synchronization with __syncthreads()

The wavefront kernel uses `__syncthreads()` after border initialization, after each anti-diagonal, and during the block-level maximum reduction. This ensures that all predecessor cells are complete before the next diagonal begins and that all local maxima are visible before reduction.

## Memory Layout

Inputs use a flat fixed-length layout:

```text
sequenceA_flat = [pair0_A][pair1_A][pair2_A]...
sequenceB_flat = [pair0_B][pair1_B][pair2_B]...
```

The offset for a pair is:

```cpp
int sequenceOffset = pairIndex * sequenceLength;
```

The output array stores one maximum local alignment score per pair.

## Timing Methodology

The executable reports:

```text
H2D_COPY_TIME_MS
GPU_KERNEL_TIME_MS
D2H_COPY_TIME_MS
GPU_TOTAL_TIME_MS
CPU_REFERENCE_TIME_MS
VALIDATION_TIME_MS
```

`GPU_KERNEL_TIME_MS` is measured with CUDA events. Host-side copy timing uses `std::chrono`.

```text
GPU_TOTAL_TIME_MS = H2D_COPY_TIME_MS + GPU_KERNEL_TIME_MS + D2H_COPY_TIME_MS
```

## Correctness Validation

Validation is enabled by default. For every pair:

```text
GPU score == CPU Smith-Waterman score
```

If validation fails, the executable reports the first mismatching pair ID and both scores, then exits with a non-zero status.

## Benchmark Methodology

The benchmark script is:

```text
benchmarks/run_smith_waterman_gpu_benchmark.py
```

Default workloads:

```text
sequence_lengths = [8, 16, 32, 64]
num_pairs_values = [10, 100, 1000]
repetitions = 5
```

Quick mode uses:

```text
sequence_lengths = [16, 32]
num_pairs_values = [10, 100]
repetitions = 3
```

## Results

Benchmark results are written to:

```text
benchmarks/smith_waterman_gpu_benchmark_results.csv
```

Generate charts with:

```bash
python scripts/plot_smith_waterman_gpu_benchmark.py
```

Charts are saved to:

```text
assets/benchmark_charts/smith_waterman_gpu/
```

## Limitations

Current limitations:

* fixed-length sequence pairs
* shared-memory sequence-length limit of 64
* score-only output
* no traceback
* no long-sequence rolling-diagonal Smith-Waterman version yet
* no CUDA streams batching yet
* not optimized for very large biological datasets yet

## Future Work

Future work includes:

* Smith-Waterman rolling diagonal
* tiled wavefront
* CUDA streams batching
* FASTA query-vs-reference mode
* 2-bit DNA encoding
* traceback reconstruction
* larger real genomic datasets
