# Needleman-Wunsch CUDA Longer Sequence Support

## Overview

Phase 10.2 extends the CUDA Needleman-Wunsch work beyond the shared-memory-only prototype. The new executable preserves the existing scoring recurrence and adds three memory-architecture modes:

```text
global_matrix
rolling_diagonal
tiled_wavefront
```

The goal is to support longer sequences while keeping correctness visible and benchmarkable before attempting more aggressive optimization.

## Why the Shared-Memory Prototype Was Limited

The original Phase 10 kernel stores one full DP matrix per CUDA block in shared memory. For sequence length `n`, that matrix grows as:

```text
(n + 1) x (n + 1)
```

Because shared memory is limited per block, a full matrix becomes impractical as `n` grows. A 64-base sequence already needs a `65 x 65` integer matrix. Larger lengths quickly exceed the comfortable range for the original prototype even though the recurrence itself is unchanged.

## Global Memory DP Matrix

The `global_matrix` mode keeps the same anti-diagonal wavefront idea but stores each pair's full DP matrix in global memory instead of shared memory.

This expands the feasible sequence length because global memory is much larger than per-block shared memory. The tradeoff is that global memory is slower, so this mode is best understood as a correctness-first baseline for longer sequences rather than a final high-performance design.

## Rolling Diagonal Approach

The `rolling_diagonal` mode avoids storing the full DP matrix when only the final alignment score is required.

For wavefront computation, each anti-diagonal only needs:

```text
previous_previous_diagonal
previous_diagonal
current_diagonal
```

That reduces memory from quadratic growth to linear growth with sequence length. It is the most practical Phase 10.2 mode for longer score-only workloads because it keeps the recurrence correct while using far less DP storage.

## Tiled Wavefront Prototype

The `tiled_wavefront` mode divides the matrix into square tiles and processes tile anti-diagonals in dependency order. Each tile then computes its internal cells in local wavefront order.

This is the architectural direction needed for scalable larger matrices because it opens the door to locality-aware processing and broader scheduling strategies. In Phase 10.2, the mode is intentionally marked:

```text
IMPLEMENTATION_STATUS=EXPERIMENTAL
```

The prototype is correct-first and prepares the design space for later optimization work.

## Memory Complexity

For equal-length sequences of length `n`:

```text
global_matrix:      O(n^2)
rolling_diagonal:   O(n)
tiled_wavefront:    O(n^2) in the current prototype
```

The current tiled prototype still stores a full global matrix so tile dependencies remain straightforward to validate.

## Time Complexity

All three modes preserve the standard Needleman-Wunsch dynamic programming work:

```text
O(m x n)
```

Changing memory architecture changes storage cost and memory-access behavior, but not the asymptotic recurrence cost.

## CUDA Synchronization Challenges

Wavefront dynamic programming is harder than embarrassingly parallel kernels because dependencies must be respected across anti-diagonals.

The main synchronization concerns are:

* waiting between anti-diagonals inside a block
* managing tile dependency order
* avoiding races while rotating diagonal buffers
* preventing later work from reading DP state before predecessor state is complete

As tiled implementations scale beyond the current prototype, global synchronization across tiles becomes one of the central design challenges.

## Benchmark Methodology

The benchmark script is:

```text
benchmarks/run_needleman_wunsch_longseq_benchmark.py
```

Default matrix:

```text
sequence_lengths = [64, 128, 256, 512]
num_pairs_values = [10, 100, 1000]
implementations = ["global_matrix", "rolling_diagonal", "tiled_wavefront"]
repetitions = 3
```

Quick mode:

```text
sequence_lengths = [64, 128]
num_pairs_values = [10, 100]
implementations = ["global_matrix", "rolling_diagonal"]
repetitions = 2
```

The benchmark records unsupported shared-memory workloads explicitly rather than hiding them.

## Results

Benchmark results are written to:

```text
benchmarks/needleman_wunsch_longseq_benchmark_results.csv
```

Charts are generated with:

```bash
python scripts/plot_needleman_wunsch_longseq_benchmark.py
```

and saved to:

```text
assets/benchmark_charts/needleman_wunsch_longseq/
```

The results section should be updated with measured Google Colab T4 data after the benchmark is executed.

## Correctness Validation

Validation is enabled by default. Every GPU score is compared against the reusable CPU reference from:

```text
src/common/needleman_wunsch.h
```

If the first mismatch appears, the executable reports the pair ID, CPU score, and GPU score, then exits with a non-zero status.

## Limitations

The current Phase 10.2 implementation still has important limitations:

* Fixed-length sequence pairs per input file.
* Score-only output with no traceback.
* `global_matrix` and the current `tiled_wavefront` prototype use quadratic memory.
* The tiled mode is correct-first and experimental rather than aggressively optimized.
* Large batch scheduling is still simple.
* Variable-length sequence support is not implemented yet.
* Global synchronization across tiles remains difficult for more advanced tiling schemes.
* Memory bandwidth becomes increasingly important as matrices grow.

## Future Work

Before moving to Smith-Waterman CUDA, the next useful steps are:

* strengthen tiled-wavefront scheduling
* evaluate larger tiles and tile-local shared-memory staging
* combine longer-sequence support with the Phase 10.1 pipeline improvements
* add variable-length sequence handling
* prepare traceback-aware storage strategies
* study larger FASTA-derived workloads
* decide whether rolling diagonals or tiled matrices should become the main path for future Smith-Waterman work
