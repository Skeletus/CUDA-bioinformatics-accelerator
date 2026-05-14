# Phase 5 Benchmark Report

## Scope

Phase 5 measures the performance of the Hamming Distance CPU baseline against the manually implemented CUDA Hamming Distance kernel. The benchmark uses synthetic fixed-length DNA sequence pairs and records CPU time, GPU kernel time, GPU total time, speedups, pair throughput, base throughput, and correctness status.

The default workload matrix is:

```text
sequence_lengths = [64, 128, 256, 512, 1024]
num_pairs_values = [1000, 10000, 100000, 1000000]
```

These values are configurable near the top of `benchmarks/run_hamming_benchmark.py` so the workload can be reduced when a Colab runtime has limited time, memory, or disk space.

## Timing Definitions

CPU time is the average time required by the C++ CPU implementation to compute Hamming Distance for every sequence pair.

GPU kernel time is measured with CUDA events around only the CUDA kernel launch. It does not include host-to-device copies, device-to-host copies, file I/O, or validation.

GPU total time is the average elapsed host-side time for a measured GPU run that includes host-to-device copies, kernel execution, and device-to-host copies. This value is usually more representative of end-to-end accelerator cost for the current implementation.

For the encoded GPU implementation, Phase 7 now reports a more detailed timing breakdown:

```text
FILE_READ_TIME_MS
INPUT_VALIDATION_TIME_MS
ENCODING_TIME_MS
HOST_ALLOCATION_TIME_MS
DEVICE_ALLOCATION_TIME_MS
H2D_COPY_TIME_MS
GPU_KERNEL_TIME_MS
D2H_COPY_TIME_MS
CPU_REFERENCE_TIME_MS
VALIDATION_TIME_MS
CSV_WRITE_TIME_MS
GPU_TOTAL_TIME_MS
END_TO_END_TIME_MS
```

For encoded runs, `GPU_TOTAL_TIME_MS` is defined as:

```text
DEVICE_ALLOCATION_TIME_MS + H2D_COPY_TIME_MS + GPU_KERNEL_TIME_MS + D2H_COPY_TIME_MS
```

`END_TO_END_TIME_MS` is the full executable pipeline time. It includes file reading, input validation, CPU-side encoding, host allocation, device allocation, memory copies, kernel execution, CPU reference computation, validation, and CSV writing.

This distinction matters because encoded kernel time can be good while encoded end-to-end time is poor. CPU-side encoding, correctness validation, and CSV writing are useful and necessary for the benchmark, but they should not be confused with CUDA kernel performance.

## Small and Large Workloads

Small workloads may be faster on CPU because GPU execution has fixed overheads: memory allocation, data transfers, kernel launch latency, synchronization, and CUDA runtime setup. When the number of sequence pairs is small, there may not be enough parallel work to amortize those costs.

Larger workloads are expected to use the GPU more effectively. Hamming Distance is data-parallel across sequence pairs, so increasing the number of pairs gives the CUDA kernel more independent work and improves occupancy and throughput potential.

## Warm-Up

The GPU executable launches one warm-up kernel before measured timing. The warm-up initializes CUDA context behavior and removes first-launch overhead from the reported kernel time. The warm-up result is not used for the final benchmark result, and the code synchronizes after warm-up with CUDA error checking before measured runs begin.

## Repeated Measurements

Both CPU and GPU executables support measured repetitions:

```bash
./hamming_cpu input_path output_path --repetitions 5
./hamming_gpu input_path output_path --repetitions 5
```

If `--repetitions` is omitted, both executables use `5`. The benchmark reports average times to reduce noise from transient runtime variation. Minimum times are also printed by the executables for inspection.

## Derived Metrics

The benchmark CSV includes:

- `total_bases_compared`: `num_pairs * sequence_length`
- `kernel_speedup`: `cpu_time_ms / gpu_kernel_time_ms`
- `total_speedup`: `cpu_time_ms / gpu_total_time_ms`
- `cpu_pairs_per_second`: CPU pair throughput
- `gpu_kernel_pairs_per_second`: GPU kernel-only pair throughput
- `gpu_total_pairs_per_second`: GPU end-to-end pair throughput
- `cpu_bases_per_second`: CPU base comparison throughput
- `gpu_kernel_bases_per_second`: GPU kernel-only base comparison throughput
- `gpu_total_bases_per_second`: GPU end-to-end base comparison throughput

Division by zero is handled safely by reporting `0.0` for the affected derived metric.

For real dataset encoded benchmarks, the CSV also distinguishes:

- `encoded_kernel_speedup`: `cpu_time_ms / encoded_gpu_kernel_time_ms`
- `encoded_total_speedup`: `cpu_time_ms / encoded_gpu_total_time_ms`
- `encoded_end_to_end_speedup`: `cpu_time_ms / encoded_end_to_end_time_ms`

`encoded_total_speedup` compares CPU time against the encoded GPU device pipeline. `encoded_end_to_end_speedup` compares CPU time against the full encoded executable pipeline, including preprocessing and validation.

## Correctness Validation

Correctness validation is preserved. The GPU executable computes a CPU reference internally and reports `VALIDATION_STATUS=PASSED` only when every GPU distance matches the CPU reference. The benchmark script also compares the CPU and GPU result CSV files. The final `passed` column is `true` only when the GPU executable passes validation, both commands complete successfully, and the output files match.

If validation fails, the benchmark prints a clear failure message, marks the row as failed, and continues to the next workload unless `STOP_ON_FAILURE` is set to `True` in `benchmarks/run_hamming_benchmark.py`.

## Running the Benchmark

In Google Colab, compile and run:

```bash
!g++ src/hamming_cpu.cpp -O3 -std=c++17 -I src/common -o hamming_cpu
!nvcc src/hamming_gpu.cu -O3 -std=c++17 -I src/common -o hamming_gpu
!python benchmarks/run_hamming_benchmark.py
!python scripts/plot_benchmarks.py
```

The benchmark script can also compile the executables automatically when `AUTO_COMPILE_EXECUTABLES = True`.

## Outputs

Benchmark data is saved to:

```text
benchmarks/hamming_benchmark_results.csv
```

CPU and GPU result files are saved to:

```text
results/hamming/
```

Synthetic datasets are saved to:

```text
data/synthetic/
```

Benchmark charts are saved to:

```text
assets/benchmark_charts/
```

Generated charts:

```text
cpu_vs_gpu_total_time.png
cpu_vs_gpu_kernel_time.png
kernel_speedup_by_workload.png
total_speedup_by_workload.png
bases_per_second.png
```

Encoded timing breakdown charts for Phase 7 real dataset pairing modes are generated with:

```bash
python scripts/plot_encoded_timing_breakdown.py
```

Charts are saved to:

```text
assets/benchmark_charts/encoded_timing_breakdown/
```

Generated charts:

```text
encoded_timing_breakdown_by_mode.png
encoded_gpu_pipeline_breakdown_by_mode.png
encoded_overhead_vs_kernel_by_mode.png
```

These charts make it easier to decide whether future work should focus on GPU-side encoding, encoded data reuse, encoded on-disk datasets, 2-bit packing, CUDA streams, or later alignment algorithms such as Smith-Waterman.

## Optimized Encoded Pipeline Benchmark

Phase 7 adds a separate optimized encoded benchmark:

```bash
python benchmarks/run_encoded_optimized_benchmark.py
```

It compares the baseline `hamming_gpu_encoded` executable against `hamming_gpu_encoded_optimized` across the real dataset pairing modes. The optimized executable uses pinned host memory, one-time reusable device allocation, encoded unique-sequence caching, and `--summary-only` output to reduce non-kernel overhead during benchmark runs.

Results are saved to:

```text
benchmarks/encoded_optimized_benchmark_results.csv
```

Generated charts:

```bash
python scripts/plot_encoded_optimized_benchmark.py
```

```text
assets/benchmark_charts/encoded_optimized/
```

The optimized benchmark separates:

- setup time
- H2D copy time
- kernel time
- D2H copy time
- GPU pipeline time
- end-to-end time
- cache hit rate

This benchmark should be used to decide whether the next optimization should target transfer volume, encoded data reuse, output cost, or a future index-based GPU representation.

## Indexed CUDA Graphs Benchmark

The indexed CUDA Graphs benchmark compares the optimized flat encoded implementation against:

```text
hamming_gpu_encoded_indexed_graphs
```

Run:

```bash
python benchmarks/run_indexed_graphs_benchmark.py
python scripts/plot_indexed_graphs_benchmark.py
```

Results are saved to:

```text
benchmarks/indexed_graphs_benchmark_results.csv
assets/benchmark_charts/indexed_graphs/
```

The benchmark reports transfer reduction, graph support, `cudaMallocAsync` support, setup time, GPU pipeline time, end-to-end time, and validation status. CUDA Graphs reduce repeated H2D/kernel/D2H orchestration overhead, while the indexed representation reduces transferred input bytes.
