# Encoded Pipeline Optimization

Phase 7 timing breakdowns showed that the encoded CUDA Hamming kernel is not the main bottleneck for real dataset workloads. The larger costs are device allocation, host-to-device transfers, repeated CPU-side encoding, duplicated flat pair buffers, CSV writing, and other end-to-end overhead.

This phase adds a separate optimized executable:

```text
src/hamming_gpu_encoded_optimized.cu
```

The original encoded executable remains available for baseline comparisons.

## Bottlenecks

The pair-based real dataset files duplicate sequence text for every comparison:

```text
SEQUENCE_A SEQUENCE_B
SEQUENCE_A SEQUENCE_B
```

This is simple and compatible with the existing Hamming programs, but it causes repeated CPU-side parsing and encoding. In `sampled` and `all_vs_all` modes, the same fragments are reused many times, so encoding every pair independently wastes work.

The previous encoded pipeline also made it hard to separate setup costs from operational GPU costs. `cudaMalloc` can be expensive because it interacts with the CUDA runtime and device memory allocator. Paying that cost in the measured path can hide the true kernel and transfer behavior.

## Optimized Pipeline

The optimized executable keeps the same input format, but changes the internal pipeline:

1. Read pair-based input.
2. Build a cache of unique DNA sequences.
3. Encode each unique sequence once.
4. Allocate pinned host buffers for flat encoded pair arrays and distances.
5. Build flat encoded buffers from the encoded cache.
6. Allocate GPU buffers once through a simple memory pool.
7. Reuse those buffers across measured repetitions.
8. Copy pinned host buffers to device.
9. Launch one encoded Hamming kernel.
10. Copy distances back to pinned host memory.
11. Validate against CPU reference results.
12. Optionally write per-pair CSV output.

## Pinned Host Memory

The optimized executable uses `cudaMallocHost` for:

```text
hostEncodedSequenceA
hostEncodedSequenceB
hostDistances
```

Pinned, or page-locked, host memory lets CUDA perform host-device transfers more efficiently than pageable memory. This can reduce H2D and D2H copy time, especially for large pair buffers.

Pinned allocation is measured separately:

```text
PINNED_HOST_ALLOCATION_TIME_MS=...
```

Pinned memory is released with `cudaFreeHost`.

## Memory Pool

The optimized executable includes a small `GpuMemoryPool` class that owns:

```text
deviceSequenceA
deviceSequenceB
deviceDistances
```

The pool allocates these buffers once, reuses them for all measured repetitions, and frees them once at the end. This keeps `cudaMalloc` out of the repeated measurement loop while still reporting allocation cost:

```text
DEVICE_ALLOCATION_TIME_MS=...
DEVICE_FREE_TIME_MS=...
```

## Encoded Data Cache

The cache maps each unique DNA sequence to an index:

```text
sequenceToCacheIndex
encodedSequenceCache
pairSequenceAIndex
pairSequenceBIndex
```

Each unique sequence is encoded once. Pair buffers are then reconstructed from cached encoded sequences. This reduces repeated CPU-side encoding and reports:

```text
UNIQUE_SEQUENCE_COUNT=...
CACHE_HIT_COUNT=...
CACHE_MISS_COUNT=...
CACHE_HIT_RATE=...
ENCODED_CACHE_TIME_MS=...
FLAT_BUFFER_BUILD_TIME_MS=...
```

This helps most when real fragments are reused many times, such as `sampled` and `all_vs_all` modes.

## Timing Definitions

The optimized executable separates setup, GPU pipeline, and end-to-end time:

```text
SETUP_TIME_MS =
  PINNED_HOST_ALLOCATION_TIME_MS +
  DEVICE_ALLOCATION_TIME_MS +
  ENCODED_CACHE_TIME_MS
```

```text
GPU_PIPELINE_TIME_MS =
  H2D_COPY_TIME_MS +
  GPU_KERNEL_TIME_MS +
  D2H_COPY_TIME_MS
```

```text
END_TO_END_TIME_MS =
  complete program runtime including file read, validation, encoding/cache,
  setup, flat buffer construction, copies, kernel, CPU reference validation,
  optional CSV writing, and cleanup
```

`--summary-only` disables per-pair CSV writing so performance benchmarks are not distorted by disk I/O. Validation still runs unless `--skip-validation` is explicitly passed.

## Kernel Fusion Opportunities

The current encoded Hamming pipeline already uses a single Hamming kernel for the measured computation. There are no extra initialization, postprocessing, or similarity kernels to fuse in the optimized path. For this phase, the correct optimization focus is memory allocation, memory transfer, encoded caching, and output overhead.

Future kernels that compute additional metrics could fuse distance and summary reductions, but that is not necessary for the current fixed-length Hamming workload.

## Benchmark Methodology

Run:

```bash
python benchmarks/run_encoded_optimized_benchmark.py
```

The benchmark compares:

```text
hamming_gpu_encoded
hamming_gpu_encoded_optimized
```

across:

```text
adjacent
sampled
all_vs_all
mutated_queries
```

Results are saved to:

```text
benchmarks/encoded_optimized_benchmark_results.csv
```

Charts are generated with:

```bash
python scripts/plot_encoded_optimized_benchmark.py
```

and saved to:

```text
assets/benchmark_charts/encoded_optimized/
```

## Results Interpretation

The most useful comparisons are:

* `optimized_gpu_pipeline_time_ms` versus `baseline_encoded_gpu_total_time_ms`
* `optimized_end_to_end_time_ms` versus `baseline_encoded_end_to_end_time_ms`
* cache hit rate by pairing mode
* H2D and D2H copy time after pinned memory
* setup time versus repeated GPU pipeline time

If end-to-end speedup is low but GPU pipeline speedup improves, the remaining bottleneck is outside the repeated GPU work.

## Limitations

The optimized implementation still builds flat encoded pair buffers after caching. This preserves compatibility and keeps the implementation readable, but it still duplicates pair data before transfer.

The optimized implementation does not implement:

* index-based GPU pair representation
* GPU-side encoding
* 2-bit packing
* CUDA streams
* batching
* `cudaMallocAsync`
* persistent kernels
* Needleman-Wunsch
* Smith-Waterman

## Future Work

The highest-value next step is an index-based pair representation. Instead of transferring duplicated flat pair buffers, the GPU would receive unique encoded fragments plus two pair-index arrays. This can reduce H2D copy size for `sampled` and `all_vs_all` modes.

Other future optimizations include GPU-side encoding, storing encoded datasets on disk, 2-bit packing, CUDA streams, batching, `cudaMallocAsync`, persistent kernels, and later Smith-Waterman kernels.
