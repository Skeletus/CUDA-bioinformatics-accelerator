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
