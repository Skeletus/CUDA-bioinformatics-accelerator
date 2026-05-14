# Indexed CUDA Graphs Optimization

This phase adds a new encoded Hamming implementation:

```text
src/hamming_gpu_encoded_indexed_graphs.cu
```

It keeps the existing pair-based text input format, but changes the GPU representation to reduce duplicated sequence transfers and repeated launch orchestration.

## Why Flat Pair Buffers Are Wasteful

The flat encoded pipeline builds two arrays:

```text
sequenceA_flat = [pair0_A][pair1_A][pair2_A]...
sequenceB_flat = [pair0_B][pair1_B][pair2_B]...
```

This is simple, but it duplicates fragments every time a fragment appears in a pair. In `sampled` and `all_vs_all` modes, each real genomic fragment appears many times, so the same encoded sequence is copied to the GPU repeatedly.

## Index-based Representation

The indexed representation stores each unique encoded fragment once:

```text
uniqueEncodedFragments = [fragment0][fragment1][fragment2]...
pairIndexA = [0, 0, 1, 2, ...]
pairIndexB = [1, 2, 2, 3, ...]
```

The CUDA kernel reads `pairIndexA[pair]` and `pairIndexB[pair]`, computes offsets into `uniqueEncodedFragments`, and compares those two fragments.

This reduces H2D traffic when many pairs reuse the same fragments. The executable reports:

```text
UNIQUE_FRAGMENT_BYTES
PAIR_INDEX_BYTES
RESULT_BYTES
FLAT_PAIR_BYTES_EQUIVALENT
INDEXED_REPRESENTATION_BYTES
FLAT_PAIR_BYTES_AVOIDED
TRANSFER_REDUCTION_RATIO
```

`TRANSFER_REDUCTION_RATIO` is `flat bytes / indexed bytes`. Values above `1.0` mean the indexed representation transfers less input data than the flat pair representation.

## Pinned Host Memory

The indexed implementation uses pinned host memory for:

```text
hostUniqueEncodedFragments
hostPairIndexA
hostPairIndexB
hostDistances
```

Pinned memory improves host-device transfer behavior and is freed with `cudaFreeHost`.

## cudaMallocAsync

`cudaMallocAsync` allocates device memory from a CUDA memory pool on a stream. It can reduce allocation overhead and improve allocator behavior compared with repeated synchronous `cudaMalloc` calls.

The executable enables `cudaMallocAsync` by default when the runtime and device support CUDA memory pools. If unsupported, it falls back to `cudaMalloc` and prints:

```text
CUDA_MALLOC_ASYNC_SUPPORTED=false
USE_CUDA_MALLOC_ASYNC=false
FALLBACK_MODE=true
```

## CUDA Graphs

CUDA Graphs capture a repeated sequence of GPU work:

```text
H2D copies
indexed Hamming kernel
D2H copy
```

After graph instantiation, repeated launches can reduce CPU-side orchestration overhead. CUDA Graphs help with repeated copy/kernel/copy execution and launch overhead. They do not reduce data volume; the index-based representation does that.

If CUDA Graphs are unsupported or disabled, the executable falls back to the normal stream path and prints:

```text
CUDA_GRAPHS_SUPPORTED=false
USE_CUDA_GRAPHS=false
FALLBACK_MODE=true
```

## Timing Definitions

The executable reports:

```text
SETUP_TIME_MS =
  pinned host allocation +
  device allocation +
  graph creation +
  graph instantiation
```

```text
GPU_PIPELINE_TIME_MS =
  H2D_COPY_TIME_MS +
  GPU_KERNEL_TIME_MS +
  D2H_COPY_TIME_MS
```

When CUDA Graphs are enabled:

```text
GPU_PIPELINE_TIME_MS = GRAPH_AVERAGE_EXECUTION_TIME_MS
```

`END_TO_END_TIME_MS` is the complete executable runtime, including file read, validation, unique sequence detection, encoding, setup, graph or stream execution, CPU reference computation, validation, optional CSV writing, and cleanup.

## Benchmark Methodology

Run:

```bash
python benchmarks/run_indexed_graphs_benchmark.py
```

The benchmark compares:

```text
hamming_gpu_encoded_optimized
hamming_gpu_encoded_indexed_graphs
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
benchmarks/indexed_graphs_benchmark_results.csv
```

Generate charts with:

```bash
python scripts/plot_indexed_graphs_benchmark.py
```

Charts are saved to:

```text
assets/benchmark_charts/indexed_graphs/
```

## Results Interpretation

The indexed representation should help most when fragments are reused many times. `sampled` and `all_vs_all` should benefit more than `adjacent`, because adjacent pairs already have less repeated comparison structure.

Key columns:

* `transfer_reduction_ratio`: how much input transfer volume was reduced.
* `pipeline_speedup_indexed_vs_flat`: flat optimized GPU pipeline time divided by indexed pipeline time.
* `end_to_end_speedup_indexed_vs_flat`: flat optimized end-to-end time divided by indexed end-to-end time.
* `use_cuda_graphs`: whether graph execution was actually used.
* `use_cuda_malloc_async`: whether async memory allocation was actually used.

## Limitations

This phase still uses `uint8_t` per base. It does not implement 2-bit packing, Needleman-Wunsch, or Smith-Waterman. It still reads pair-based text files, so file I/O and unique sequence detection remain CPU-side costs.

## Future Work

Future work includes GPU-side encoding, 2-bit packing, CUDA streams for batched execution, larger genomes such as E. coli, Smith-Waterman, and index-based dynamic programming kernels.
