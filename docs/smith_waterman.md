# Smith-Waterman CPU Implementation

## Overview

Phase 9 adds a CPU implementation of the Smith-Waterman local alignment algorithm. This phase establishes a correct local alignment baseline before future CUDA acceleration.

The implementation reads pair-based sequence files, computes one maximum local alignment score per pair, writes CSV results, and reports benchmark metrics.

## Why Local Alignment Matters

Local alignment finds the best matching region between two sequences. This is useful when only part of a sequence is expected to match, such as local similarity search across larger genomic regions.

Smith-Waterman is more biologically flexible than Hamming Distance because it supports gaps and does not require full-sequence alignment.

## Difference Between Hamming Distance, Needleman-Wunsch, and Smith-Waterman

Hamming Distance:

* compares equal-length sequences position by position.
* does not support gaps.
* runs in `O(n)` time.

Needleman-Wunsch:

* performs global alignment.
* aligns the full sequence end-to-end.
* supports gaps.
* usually uses `dp[m][n]` as the final score.
* runs in `O(m * n)` time.

Smith-Waterman:

* performs local alignment.
* finds the best matching subsequence region.
* supports insertions and deletions through gaps.
* resets negative DP values to zero.
* uses the maximum value anywhere in the DP matrix as the final score.
* runs in `O(m * n)` time.

## Dynamic Programming Matrix

For sequences:

```text
A length = m
B length = n
```

Smith-Waterman builds a matrix of size:

```text
(m + 1) x (n + 1)
```

The first row and first column are initialized to zero so local alignments can start anywhere.

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
dp[0][j] = 0
dp[i][0] = 0
```

Transition:

```text
scoreDiagonal = dp[i - 1][j - 1] + matchOrMismatch
scoreUp       = dp[i - 1][j] + gapPenalty
scoreLeft     = dp[i][j - 1] + gapPenalty
dp[i][j]      = max(0, scoreDiagonal, scoreUp, scoreLeft)
```

The final score is:

```text
maxScore = maximum value in the entire DP matrix
```

Phase 9 returns only the maximum local alignment score. It does not reconstruct aligned subsequences yet.

## Time Complexity

Smith-Waterman computes every DP cell once:

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

space. The executable defaults to:

```text
--memory-mode rolling
```

The full matrix version is available for clarity:

```text
--memory-mode full
```

## Implementation Details

Reusable functions are implemented in:

```text
src/common/smith_waterman.h
```

The command-line executable is:

```text
src/smith_waterman_cpu.cpp
```

Compile:

```bash
g++ src/smith_waterman_cpu.cpp \
  -O3 \
  -std=c++17 \
  -I src/common \
  -o smith_waterman_cpu
```

Run:

```bash
./smith_waterman_cpu \
  data/synthetic/synthetic_pairs_128.txt \
  results/smith_waterman/smith_waterman_cpu_results.csv \
  --repetitions 3
```

## Validation Examples

The test executable validates:

* perfect match: `ACGT` vs `ACGT`, expected score `8`.
* local match inside a longer sequence: `ACGT` vs `TTACGTAA`, expected score `8`.
* no useful match: `AAAA` vs `TTTT`, expected score `0`.
* partial local match: `GATTACA` vs `TACA`, expected score `8`.
* one mismatch: `ACGT` vs `ACCT`, expected score `5`.
* empty sequence: empty string vs `ACGT`, expected score `0`.

Run tests:

```bash
g++ tests/test_smith_waterman.cpp \
  -O3 \
  -std=c++17 \
  -I src/common \
  -o test_smith_waterman

./test_smith_waterman
```

## Benchmark Methodology

The benchmark script is:

```text
benchmarks/run_smith_waterman_cpu_benchmark.py
```

Default workloads:

```text
sequence_lengths = [16, 32, 64, 128, 256]
num_pairs_values = [10, 100, 1000]
repetitions = 3
```

The benchmark avoids huge workloads because Smith-Waterman is quadratic per pair.

Output:

```text
benchmarks/smith_waterman_cpu_benchmark_results.csv
```

## Results

Generate charts with:

```bash
python scripts/plot_smith_waterman_cpu_benchmark.py
```

Charts are saved to:

```text
assets/benchmark_charts/smith_waterman_cpu/
```

Generated charts:

```text
sw_cpu_time_by_workload.png
sw_cells_per_second.png
sw_average_time_per_pair.png
sw_complexity_growth.png
```

## Limitations

Phase 9 is CPU-only. It does not implement:

* CUDA Smith-Waterman.
* GPU wavefront parallelism.
* 2-bit packing.
* traceback reconstruction.
* affine gap penalties.
* banded Smith-Waterman.
* ambiguous base scoring.

## Future Work

Future work includes:

* CUDA implementation.
* Wavefront parallelism.
* Shared memory tiling.
* Real genomic fragment alignment.
* Rolling-row batching improvements.
* Traceback reconstruction.
* Banded Smith-Waterman.
* Batched Smith-Waterman GPU.
