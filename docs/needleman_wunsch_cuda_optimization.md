# Needleman-Wunsch CUDA Pipeline Optimization

## Overview

Phase 10.1 improves the execution pipeline around the existing Needleman-Wunsch CUDA wavefront prototype without changing the dynamic programming recurrence. The optimized executable keeps the same shared-memory score-only kernel and adds pinned host memory, reusable device buffers, optional `cudaMallocAsync`, CUDA streams, batched execution, and a more detailed timing breakdown.

## Baseline CUDA Implementation

The Phase 10 baseline implementation processes fixed-length sequence pairs with one CUDA block per pair. The wavefront kernel evaluates anti-diagonals in shared memory and validates every GPU score against the CPU Needleman-Wunsch reference.

That implementation is intentionally correctness-first. It copies the whole dataset at once, allocates device memory before execution, and uses a simple synchronous transfer flow.

## Why Pipeline Optimization Matters

Once the recurrence is correct, throughput is influenced by more than kernel execution alone. Allocation cost, host-memory type, transfer scheduling, and repeated movement of large datasets can dominate small workloads or reduce scaling for larger workloads.

Phase 10.1 focuses on those pipeline costs while preserving the current algorithm and sequence-length limit.

## Pinned Host Memory

Pageable host memory can require the CUDA runtime to stage data through temporary pinned buffers before DMA transfers reach the GPU. That extra work can slow CPU-GPU transfers and prevents the cleanest use of asynchronous copies.

Pinned, or page-locked, host memory is allocated with `cudaMallocHost`. Because those pages cannot be moved by the operating system while the transfer is in flight, `cudaMemcpyAsync` can transfer directly between host and device more efficiently.

The optimized executable allocates pinned buffers for:

```text
hostSequenceA
hostSequenceB
hostScores
```

The implementation allocates one host buffer set per CUDA stream so a later batch does not overwrite memory still being used by an in-flight transfer.

## cudaMallocAsync and Memory Pooling

Repeated allocation and deallocation are expensive operations, especially if they happen inside the measured compute path. The optimized implementation avoids that pattern.

When supported by the CUDA runtime and device, Phase 10.1 uses `cudaMallocAsync` and `cudaFreeAsync`. These APIs work with CUDA memory pools and can reduce allocator overhead compared with repeated synchronous allocation patterns.

The executable also implements a simple reusable pool design:

```text
one reusable deviceSequenceA buffer per stream
one reusable deviceSequenceB buffer per stream
one reusable deviceScores buffer per stream
```

Each buffer is sized for the maximum configured batch and reused across all repetitions and compatible batches. If async allocation is not supported, the executable falls back to `cudaMalloc` and reports that decision explicitly.

## CUDA Streams

CUDA streams provide independent queues of asynchronous work. In Phase 10.1, each batch follows the same sequence:

```text
H2D copy -> wavefront kernel -> D2H copy
```

Using multiple streams allows batches to be rotated across queues:

```text
batch 0 -> stream 0
batch 1 -> stream 1
batch 2 -> stream 0
batch 3 -> stream 1
```

This structure enables overlap opportunities between transfers and compute for batched workloads. Correctness still requires careful synchronization: before a stream-local host buffer is reused, that stream is synchronized and its prior output is harvested.

## Batched Execution

The optimized executable reads all sequence pairs, validates the fixed-length requirement once, and then processes them in configurable batches. Batching prevents large datasets from requiring one monolithic host-to-device copy and makes the stream pipeline practical.

The default batch size is:

```text
1024
```

The final batch may contain fewer pairs than the configured batch size. Device buffers are still reused across all batches where possible.

## Timing Methodology

The executable reports:

```text
FILE_READ_TIME_MS
INPUT_VALIDATION_TIME_MS
PINNED_HOST_ALLOCATION_TIME_MS
DEVICE_ALLOCATION_TIME_MS
H2D_COPY_TIME_MS
GPU_KERNEL_TIME_MS
D2H_COPY_TIME_MS
GPU_PIPELINE_TIME_MS
CPU_REFERENCE_TIME_MS
VALIDATION_TIME_MS
CSV_WRITE_TIME_MS
DEVICE_FREE_TIME_MS
PINNED_HOST_FREE_TIME_MS
END_TO_END_TIME_MS
```

`H2D_COPY_TIME_MS`, `GPU_KERNEL_TIME_MS`, and `D2H_COPY_TIME_MS` are averaged across repetitions. Stage timing uses CUDA events on the stream that owns each batch.

```text
GPU_PIPELINE_TIME_MS = H2D_COPY_TIME_MS + GPU_KERNEL_TIME_MS + D2H_COPY_TIME_MS
```

`END_TO_END_TIME_MS` is wall-clock time for the full executable path, including file read, validation, allocation, GPU work, CPU reference computation, CSV writing, and cleanup.

The executable also reports:

```text
TOTAL_CELLS_COMPUTED
GPU_TOTAL_CELLS_PER_SECOND
GPU_KERNEL_CELLS_PER_SECOND
```

## Correctness Validation

Validation remains enabled by default. Every GPU score is compared against the reusable CPU Needleman-Wunsch reference from:

```text
src/common/needleman_wunsch.h
```

If a mismatch occurs, the executable reports the first mismatching pair ID and both scores, then exits with a non-zero status.

## Benchmark Methodology

The comparison benchmark is:

```text
benchmarks/run_needleman_wunsch_gpu_optimized_benchmark.py
```

It compares:

```text
needleman_wunsch_cpu
needleman_wunsch_gpu
needleman_wunsch_gpu_optimized
```

Default matrix:

```text
sequence_lengths = [8, 16, 32, 64]
num_pairs_values = [10, 100, 1000, 5000]
batch_sizes = [256, 1024]
num_streams_values = [1, 2, 4]
repetitions = 5
```

For faster Colab validation, use:

```bash
python benchmarks/run_needleman_wunsch_gpu_optimized_benchmark.py --quick
```

## Results

Benchmark results are written to:

```text
benchmarks/needleman_wunsch_gpu_optimized_benchmark_results.csv
```

Charts are generated with:

```bash
python scripts/plot_needleman_wunsch_gpu_optimized_benchmark.py
```

and saved to:

```text
assets/benchmark_charts/needleman_wunsch_gpu_optimized/
```

The results section should be updated with measured Colab T4 observations after the benchmark is run.

## Limitations

Phase 10.1 improves the execution pipeline, but longer sequence support requires a different DP memory architecture such as global-memory DP buffers, rolling diagonals, or tiled wavefront.

Current limitations remain:

* Fixed-length sequence pairs per input file.
* Shared-memory sequence-length limit of 64 bases per sequence side.
* Score-only output.
* No traceback reconstruction.
* No Smith-Waterman CUDA implementation yet.
* Stream overlap is intentionally simple and correctness-first rather than fully aggressive.

## Future Work

Next steps include:

* Phase 10.2: global memory DP support.
* Rolling diagonals.
* Tiled wavefront.
* CUDA Graphs for repeated launch patterns.
* Larger FASTA workloads.
* Smith-Waterman CUDA.
