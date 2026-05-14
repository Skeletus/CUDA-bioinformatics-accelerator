# Needleman-Wunsch CPU Implementation

## Overview

Phase 8 adds a CPU implementation of the Needleman-Wunsch global alignment algorithm. This phase establishes a correct dynamic programming baseline before future CUDA acceleration work.

The implementation reads pair-based sequence files, computes one global alignment score per pair, writes CSV results, and reports benchmark metrics.

## Why Global Alignment Matters

Global alignment compares two complete sequences from start to end. This is useful when the full length of both sequences should be aligned, including substitutions, insertions, and deletions.

Needleman-Wunsch is more biologically expressive than Hamming Distance because it supports gaps. It is also more computationally expensive, which makes it a strong candidate for future GPU acceleration.

## Difference Between Hamming Distance and Needleman-Wunsch

Hamming Distance:

* compares equal-length sequences position by position.
* does not support gaps.
* runs in `O(n)` time.

Needleman-Wunsch:

* supports insertions and deletions through gaps.
* aligns whole sequences globally.
* runs in `O(m * n)` time for sequence lengths `m` and `n`.

## Dynamic Programming Matrix

For sequences:

```text
A length = m
B length = n
```

Needleman-Wunsch builds a matrix of size:

```text
(m + 1) x (n + 1)
```

The first row and first column represent aligning prefixes against gaps.

## Scoring Scheme

Default scores:

```text
Match: +2
Mismatch: -1
Gap: -2
```

These values can be changed with:

```bash
--match 2
--mismatch -1
--gap -2
```

## Algorithm Steps

Initialization:

```text
dp[0][0] = 0
dp[i][0] = i * gapPenalty
dp[0][j] = j * gapPenalty
```

Transition:

```text
scoreDiagonal = dp[i - 1][j - 1] + matchOrMismatch
scoreUp       = dp[i - 1][j] + gapPenalty
scoreLeft     = dp[i][j - 1] + gapPenalty
dp[i][j]      = max(scoreDiagonal, scoreUp, scoreLeft)
```

Final score:

```text
dp[m][n]
```

Phase 8 returns only the alignment score. It does not reconstruct the aligned sequences yet.

## Time Complexity

Needleman-Wunsch computes every DP cell once:

```text
O(m * n)
```

For `N` sequence pairs of length `L`, the benchmark computes approximately:

```text
N * (L + 1) * (L + 1)
```

DP cells.

## Space Complexity

The full matrix implementation uses:

```text
O(m * n)
```

space.

The rolling-row implementation uses:

```text
O(n)
```

space by keeping only the previous and current DP rows. The executable defaults to:

```text
--memory-mode rolling
```

The full matrix version is still available for clarity:

```text
--memory-mode full
```

## Implementation Details

Reusable functions are implemented in:

```text
src/common/needleman_wunsch.h
```

The command-line executable is:

```text
src/needleman_wunsch_cpu.cpp
```

Compile:

```bash
g++ src/needleman_wunsch_cpu.cpp \
  -O3 \
  -std=c++17 \
  -I src/common \
  -o needleman_wunsch_cpu
```

Run:

```bash
./needleman_wunsch_cpu \
  data/synthetic/synthetic_pairs_128.txt \
  results/needleman_wunsch/needleman_wunsch_cpu_results.csv \
  --repetitions 3
```

## Validation Examples

The test executable validates:

* perfect match: `ACGT` vs `ACGT`, expected score `8`.
* one mismatch: `ACGT` vs `ACCT`, expected score `5`.
* one deletion: `ACGT` vs `AGT`, expected score `4`.
* empty vs non-empty: empty string vs `ACGT`, expected score `-8`.
* a classic global alignment example: `GATTACA` vs `GCATGCU`, expected score `2` with the default scoring scheme.

Run tests:

```bash
g++ tests/test_needleman_wunsch.cpp \
  -O3 \
  -std=c++17 \
  -I src/common \
  -o test_needleman_wunsch

./test_needleman_wunsch
```

## Benchmark Methodology

The benchmark script is:

```text
benchmarks/run_needleman_wunsch_cpu_benchmark.py
```

Default workloads:

```text
sequence_lengths = [16, 32, 64, 128, 256]
num_pairs_values = [10, 100, 1000]
repetitions = 3
```

The benchmark avoids huge workloads because Needleman-Wunsch is quadratic per pair.

Output:

```text
benchmarks/needleman_wunsch_cpu_benchmark_results.csv
```

## Results

Generate charts with:

```bash
python scripts/plot_needleman_wunsch_cpu_benchmark.py
```

Charts are saved to:

```text
assets/benchmark_charts/needleman_wunsch_cpu/
```

Generated charts:

```text
nw_cpu_time_by_workload.png
nw_cells_per_second.png
nw_average_time_per_pair.png
nw_complexity_growth.png
```

## Limitations

Phase 8 is CPU-only. It does not implement:

* CUDA Needleman-Wunsch.
* GPU wavefront parallelism.
* Smith-Waterman.
* traceback reconstruction.
* affine gap penalties.
* ambiguous base scoring.

## Future Work

Future work includes:

* CUDA implementation.
* Wavefront parallelism.
* Shared memory tiling.
* Smith-Waterman local alignment.
* Real genomic fragment alignment.
* Further rolling-row optimization and batching.
* Traceback reconstruction.
