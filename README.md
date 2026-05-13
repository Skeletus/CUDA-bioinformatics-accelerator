# CUDA Bioinformatics Accelerator

CUDA Bioinformatics Accelerator is a GPU-accelerated sequence comparison and alignment engine focused on processing large batches of DNA sequence pairs efficiently. The project is designed to demonstrate real CUDA programming, GPU memory optimization, parallel algorithm design, benchmarking discipline, and bioinformatics-oriented high-performance computing.

The goal is not to build a generic bioinformatics tool with a graphical interface. The goal is to build a technically strong CUDA/HPC project that can compare many DNA sequence fragments in parallel and report similarity or alignment scores faster than a CPU baseline.

All source code, variable names, function names, comments, documentation strings, commit messages, and technical notes must be written in English.

---

## 1. Project Objective

The main objective is to build a CUDA-accelerated engine for massive DNA sequence comparison.

The project starts with simple fixed-length sequence comparison using Hamming Distance and evolves toward more advanced bioinformatics algorithms such as Needleman-Wunsch and Smith-Waterman.

The project must clearly demonstrate:

- CUDA kernel development.
- CPU vs GPU benchmarking.
- Correctness validation.
- Efficient memory layout.
- Batched sequence processing.
- GPU memory management.
- Coalesced memory access.
- CUDA events for timing.
- Optional CUDA streams for overlapping transfers and computation.
- Shared memory and tiling for dynamic programming alignment.
- Professional GitHub documentation.

---

## 2. High-Level Description

DNA sequences can be represented as strings over a four-character alphabet:

```text
A, C, G, T
````

Example:

```text
ACGTACGTAGCTAGCT
```

The project compares many pairs of DNA sequences and produces similarity or alignment scores.

Example input:

```text
Sequence A: ACGTACGT
Sequence B: ACGTTCGT
```

Example output:

```text
Hamming distance: 1
Similarity: 87.5%
```

At scale, the tool should process thousands or millions of sequence pairs in batches.

---

## 3. Target User

The intended user is a developer, student, researcher, or engineer who wants to compare DNA sequences using GPU acceleration.

The first version of the project should work as a command-line tool or notebook-driven benchmark.

Example future CLI usage:

```bash
./cuda-bioaligner \
  --input data/synthetic_pairs.txt \
  --algorithm hamming \
  --output results/hamming_results.csv
```

Expected output file:

```csv
pair_id,distance,similarity
0,12,90.625
1,8,93.75
2,44,65.625
```

---

## 4. Core Use Case

The core use case is:

1. Load or generate DNA sequence pairs.
2. Store the sequence data in a GPU-friendly flat memory layout.
3. Copy the sequence data from CPU memory to GPU memory.
4. Launch CUDA kernels to compare many sequence pairs in parallel.
5. Copy the results back from GPU memory to CPU memory.
6. Validate GPU results against a CPU baseline.
7. Benchmark CPU time, GPU kernel time, and total GPU time.
8. Export results and benchmark metrics.

---

## 5. Architecture Overview

The project architecture should be organized around a CPU orchestration layer and a GPU computation layer.

```text
+--------------------------------------------------+
|                  Host / CPU                      |
|--------------------------------------------------|
| - Dataset generation                             |
| - FASTA or text parsing                          |
| - Input validation                               |
| - CPU baseline algorithms                        |
| - GPU memory allocation                          |
| - Kernel launch configuration                    |
| - Result validation                              |
| - Benchmark reporting                            |
+-------------------------+------------------------+
                          |
                          | cudaMemcpy
                          v
+--------------------------------------------------+
|                 Device / GPU                     |
|--------------------------------------------------|
| - Batched Hamming Distance kernel                |
| - Encoded DNA comparison kernel                  |
| - Needleman-Wunsch kernel                        |
| - Smith-Waterman wavefront kernel                |
| - Optional reductions                            |
| - Optional stream-based batch processing         |
+--------------------------------------------------+
                          |
                          | cudaMemcpy
                          v
+--------------------------------------------------+
|                    Results                       |
|--------------------------------------------------|
| - Distance scores                                |
| - Similarity percentages                         |
| - Alignment scores                               |
| - Benchmark CSV files                            |
| - Correctness reports                            |
+--------------------------------------------------+
```

---

## 6. Programming Language and Code Style Requirements

All programming must be done in English.

This includes:

* Variable names.
* Function names.
* Class names.
* File names.
* Comments.
* Error messages.
* Documentation.
* Commit messages.

Example of correct naming:

```cpp
int sequenceLength;
int numberOfPairs;
char* deviceSequenceA;
char* deviceSequenceB;
int* deviceDistances;
```

Avoid Spanish or mixed-language identifiers.

Incorrect:

```cpp
int longitudSecuencia;
char* secuenciaGpu;
```

Correct:

```cpp
int sequenceLength;
char* deviceSequence;
```

Code should be simple, readable, and well-structured. Prefer clarity first, then optimize.

---

## 7. Recommended Tech Stack

The project should be compatible with Google Colab and local CUDA environments when possible.

Recommended stack:

```text
CUDA C/C++
C++17 or C++20
Python for dataset generation and plotting
Google Colab
CMake optional for local builds
Matplotlib for benchmark charts
Nsight Compute optional for profiling
Nsight Systems optional for timeline profiling
```

The initial implementation should work in Google Colab using `nvcc`.

---

## 8. Dataset Strategy

The project should use three dataset levels.

### 8.1 Synthetic DNA Dataset

Synthetic datasets are used for correctness testing and early benchmarking.

Example:

```text
10,000 pairs of DNA sequences
Fixed sequence length: 128
Alphabet: A, C, G, T
```

Purpose:

* Validate the pipeline.
* Test CPU and GPU correctness.
* Control sequence length.
* Generate repeatable benchmarks.

Example synthetic pair:

```text
ACGTACGTACGT TCGTACGTACGA
```

---

### 8.2 Real Dataset: SARS-CoV-2 Reference Genome

The first real biological dataset should be the SARS-CoV-2 reference genome:

```text
NCBI RefSeq accession: NC_045512.2
```

Purpose:

* Use a real biological sequence.
* Generate realistic genomic fragments.
* Keep dataset size manageable for Google Colab.
* Benchmark sequence comparison on real genomic data.

Planned preprocessing:

```text
Input genome sequence
→ sliding window fragmentation
→ fixed-length DNA fragments
→ pair generation
→ GPU comparison
```

Example window settings:

```text
Window size: 128, 256, 512
Stride: 32 or 64
```

---

### 8.3 Larger Real Dataset: E. coli K-12 MG1655

For scalability benchmarking, the project may use:

```text
NCBI RefSeq accession: NC_000913.3
Organism: Escherichia coli K-12 MG1655
```

Purpose:

* Generate larger numbers of fragments.
* Test scalability.
* Measure throughput on larger workloads.
* Compare performance across sequence lengths and batch sizes.

This dataset should not be required for the first implementation phase.

---

## 9. Input Formats

The project should support simple text input first.

### 9.1 Pair-Based Text Format

Initial input format:

```text
ACGTACGT ACGTTCGT
GGGGAAAA GGGAAAAT
TTTTCCCC TTATCCCA
```

Each line contains two DNA sequences of the same length.

This format is ideal for Hamming Distance.

---

### 9.2 FASTA Format

Later phases should support FASTA input.

Example:

```text
>sequence_1
ACGTACGTACGTACGT

>sequence_2
ACGTTCGTACGTACGA
```

For benchmarking, FASTA sequences can be fragmented into fixed-size windows.

---

## 10. Output Formats

The project should output results as CSV files.

### 10.1 Hamming Distance Output

```csv
pair_id,distance,similarity
0,1,87.5
1,4,50.0
2,2,75.0
```

### 10.2 Alignment Score Output

For Needleman-Wunsch or Smith-Waterman:

```csv
pair_id,alignment_score,sequence_length_a,sequence_length_b
0,42,128,128
1,37,128,128
2,91,256,256
```

### 10.3 Benchmark Output

```csv
algorithm,num_pairs,sequence_length,cpu_time_ms,gpu_kernel_time_ms,gpu_total_time_ms,speedup_kernel,speedup_total
hamming,10000,128,12.4,0.3,1.1,41.3,11.2
```

---

## 11. Project Phases

The project should be implemented incrementally.

---

### Phase 1: CUDA Environment and Minimal GPU Validation

Goal:

Verify that CUDA is working in Google Colab.

Tasks:

* Check GPU with `nvidia-smi`.
* Check CUDA compiler with `nvcc --version`.
* Compile a simple CUDA program.
* Run a simple CUDA kernel.
* Validate `cudaMalloc`, `cudaMemcpy`, kernel launch, and `cudaFree`.

Deliverables:

```text
src/hello_cuda.cu
src/vector_add.cu
docs/environment_setup.md
```

---

### Phase 2: Synthetic Dataset Generator

Goal:

Generate synthetic DNA sequence pairs.

Tasks:

* Create Python script to generate random DNA sequence pairs.
* Support configurable number of pairs.
* Support configurable sequence length.
* Save output as text file.
* Ensure reproducibility using a random seed.

Deliverables:

```text
scripts/generate_synthetic_dataset.py
data/synthetic_pairs_128.txt
data/synthetic_pairs_256.txt
```

Required features:

```bash
python scripts/generate_synthetic_dataset.py \
  --num-pairs 10000 \
  --sequence-length 128 \
  --output data/synthetic_pairs_128.txt \
  --seed 42
```

---

### Phase 3: CPU Hamming Distance Baseline

Goal:

Implement a correct CPU baseline for fixed-length sequence comparison.

Tasks:

* Load pair-based text dataset.
* Compute Hamming Distance for each pair.
* Compute similarity percentage.
* Save results to CSV.
* Measure CPU execution time.

Deliverables:

```text
src/hamming_cpu.cpp
results/hamming_cpu_results.csv
benchmarks/cpu_hamming_benchmark.csv
```

Correctness rule:

For two sequences of equal length:

```text
distance = number of positions where sequenceA[i] != sequenceB[i]
similarity = 100 * (1 - distance / sequenceLength)
```

---

### Phase 4: GPU Hamming Distance Kernel

Goal:

Implement the first real CUDA bioinformatics kernel.

Tasks:

* Store sequence pairs in flat arrays.
* Allocate GPU memory.
* Copy sequence data from CPU to GPU.
* Launch one CUDA thread per sequence pair.
* Compute Hamming Distance on GPU.
* Copy results back to CPU.
* Validate against CPU output.
* Measure GPU kernel time with CUDA events.

Deliverables:

```text
src/hamming_gpu.cu
results/hamming_gpu_results.csv
benchmarks/gpu_hamming_benchmark.csv
```

Expected kernel behavior:

```text
Thread 0 computes pair 0
Thread 1 computes pair 1
Thread 2 computes pair 2
...
```

Important CUDA concepts:

* `__global__` kernel.
* `threadIdx.x`.
* `blockIdx.x`.
* `blockDim.x`.
* `cudaMalloc`.
* `cudaMemcpy`.
* `cudaDeviceSynchronize`.
* CUDA events.
* Bounds checking.

---

### Phase 5: CPU vs GPU Benchmark Suite

Goal:

Create a reproducible benchmark suite.

Tasks:

* Benchmark different sequence lengths.
* Benchmark different numbers of pairs.
* Compare CPU time vs GPU kernel time.
* Compare CPU time vs total GPU time including memory transfers.
* Export benchmark tables.
* Plot benchmark charts.

Test matrix:

```text
Sequence lengths: 64, 128, 256, 512, 1024
Number of pairs: 1,000; 10,000; 100,000; 1,000,000
```

Deliverables:

```text
benchmarks/run_hamming_benchmark.py
benchmarks/hamming_benchmark_results.csv
notebooks/benchmark_analysis.ipynb
docs/benchmark_report.md
```

Metrics:

```text
CPU time in milliseconds
GPU kernel time in milliseconds
GPU total time in milliseconds
Speedup vs CPU
Pairs processed per second
Bases compared per second
```

Current Phase 5 status:

* The Hamming benchmark suite supports the default workload matrix from 1,000 to 1,000,000 sequence pairs and sequence lengths from 64 to 1024.
* CPU and GPU executables support `--repetitions` with a default of `5`.
* The CUDA benchmark performs a warm-up kernel launch before measured kernel timing.
* Benchmark output uses machine-readable `KEY=value` lines and exports derived throughput and speedup metrics.
* GPU correctness validation is preserved, and the benchmark CSV marks `passed=true` only when validation and output comparison succeed.
* Benchmark charts are generated with `python scripts/plot_benchmarks.py` and saved in `assets/benchmark_charts/`.

---

### Phase 6: DNA Encoding Optimization

Goal:

Improve memory efficiency by encoding DNA bases as integers.

Initial encoding:

```text
A = 0
C = 1
G = 2
T = 3
```

Tasks:

* Implement CPU-side encoding from char to `uint8_t`.
* Modify GPU kernel to compare encoded sequences.
* Benchmark char-based vs encoded representation.
* Validate correctness.

Deliverables:

```text
src/dna_encoding.cpp
src/hamming_gpu_encoded.cu
docs/dna_encoding.md
notebooks/02_dna_encoding_optimization.ipynb
benchmarks/run_encoding_benchmark.py
scripts/plot_encoding_benchmarks.py
```

Future optional optimization:

Pack four DNA bases into one byte using 2-bit encoding.

Current Phase 6 status:

* DNA encoding helpers map `A=0`, `C=1`, `G=2`, and `T=3` using `uint8_t`.
* Invalid DNA bases are rejected during validation or encoding.
* The encoded CUDA Hamming implementation validates every result against a CPU encoded baseline.
* The char-based and encoded CUDA implementations can be benchmarked on the same synthetic datasets.
* Benchmark results are saved to `benchmarks/dna_encoding_benchmark_results.csv`.
* Encoding benchmark charts are saved to `assets/benchmark_charts/encoding/`.
* Phase 6 still uses fixed-length sequence pairs and does not implement 2-bit packing.

Phase 6 Colab commands:

```bash
!g++ src/dna_encoding.cpp -O3 -std=c++17 -I src/common -o dna_encoding

!nvcc src/hamming_gpu.cu -O3 -std=c++17 -I src/common -o hamming_gpu

!nvcc src/hamming_gpu_encoded.cu -O3 -std=c++17 -I src/common -o hamming_gpu_encoded

!python scripts/generate_synthetic_dataset.py \
  --num-pairs 10000 \
  --sequence-length 128 \
  --output data/synthetic/synthetic_pairs_128.txt \
  --seed 42

!./dna_encoding \
  data/synthetic/synthetic_pairs_128.txt

!./hamming_gpu_encoded \
  data/synthetic/synthetic_pairs_128.txt \
  results/hamming/hamming_gpu_encoded_results.csv \
  --repetitions 5

!python benchmarks/run_encoding_benchmark.py

!python scripts/plot_encoding_benchmarks.py
```

Additional documentation:

```text
docs/dna_encoding.md
```

---

### Phase 7: Real Dataset Integration

Goal:

Use real genomic data.

Tasks:

* Download or load SARS-CoV-2 reference genome `NC_045512.2`.
* Parse FASTA format.
* Generate fixed-length fragments using sliding windows.
* Create sequence pairs from fragments.
* Run Hamming Distance benchmarks.
* Save results.

Deliverables:

```text
scripts/download_datasets.py
scripts/fragment_fasta.py
data/sars_cov_2_fragments_128.txt
docs/datasets.md
notebooks/03_real_dataset_integration.ipynb
```

Fragmentation example:

```text
Window size: 128
Stride: 32
```

Current Phase 7 status:

* The project downloads the SARS-CoV-2 reference genome `NC_045512.2` as FASTA.
* The FASTA parser reads the header, concatenates sequence lines, and validates supported DNA bases.
* Sliding window fragmentation generates fixed-length real genomic fragments.
* Adjacent real genomic fragments are converted into pair-based Hamming input.
* CPU, char-based GPU, and encoded GPU Hamming implementations can be benchmarked on real genomic pairs.
* Benchmark results are saved to `benchmarks/real_dataset_hamming_benchmark_results.csv`.
* Real dataset charts are saved to `assets/benchmark_charts/real_dataset/`.
* Phase 7 still uses Hamming Distance and does not implement Needleman-Wunsch, Smith-Waterman, or 2-bit packing.

Phase 7 Colab commands:

```bash
!python scripts/download_datasets.py \
  --dataset sars-cov-2 \
  --output data/raw/sars_cov_2_NC_045512_2.fasta

!python scripts/fragment_fasta.py \
  --input data/raw/sars_cov_2_NC_045512_2.fasta \
  --output-csv data/processed/sars_cov_2_fragments_128.csv \
  --output-txt data/sars_cov_2_fragments_128.txt \
  --output-pairs data/processed/sars_cov_2_pairs_128_stride_32.txt \
  --window-size 128 \
  --stride 32 \
  --skip-ambiguous

!g++ src/hamming_cpu.cpp -O3 -std=c++17 -I src/common -o hamming_cpu

!nvcc src/hamming_gpu.cu -O3 -std=c++17 -I src/common -o hamming_gpu

!nvcc src/hamming_gpu_encoded.cu -O3 -std=c++17 -I src/common -o hamming_gpu_encoded

!python benchmarks/run_real_dataset_benchmark.py \
  --window-size 128 \
  --stride 32 \
  --repetitions 5

!python scripts/plot_real_dataset_benchmark.py
```

---

### Phase 8: Needleman-Wunsch CPU Implementation

Goal:

Implement global sequence alignment on CPU.

Tasks:

* Implement Needleman-Wunsch dynamic programming.
* Support match score, mismatch penalty, and gap penalty.
* Return alignment score.
* Validate on small examples.
* Benchmark CPU performance.

Deliverables:

```text
src/needleman_wunsch_cpu.cpp
tests/test_needleman_wunsch.cpp
docs/needleman_wunsch.md
```

Default scoring:

```text
Match: +2
Mismatch: -1
Gap: -2
```

---

### Phase 9: Smith-Waterman CPU Implementation

Goal:

Implement local sequence alignment on CPU.

Tasks:

* Implement Smith-Waterman dynamic programming.
* Support match score, mismatch penalty, and gap penalty.
* Return maximum local alignment score.
* Validate on small examples.
* Benchmark CPU performance.

Deliverables:

```text
src/smith_waterman_cpu.cpp
tests/test_smith_waterman.cpp
docs/smith_waterman.md
```

Default scoring:

```text
Match: +2
Mismatch: -1
Gap: -2
```

---

### Phase 10: Smith-Waterman CUDA Prototype

Goal:

Implement a CUDA version of Smith-Waterman.

Tasks:

* Start with one block per sequence pair.
* Use wavefront parallelism.
* Use shared memory where appropriate.
* Support fixed-length sequences first.
* Validate against CPU Smith-Waterman.
* Benchmark performance.

Deliverables:

```text
src/smith_waterman_gpu.cu
docs/smith_waterman_cuda.md
benchmarks/smith_waterman_benchmark.csv
```

Important CUDA concepts:

* Dynamic programming on GPU.
* Wavefront parallelism.
* Shared memory.
* Synchronization within blocks using `__syncthreads()`.
* Register pressure.
* Occupancy considerations.
* Memory bandwidth.

---

### Phase 11: Batched Processing and CUDA Streams

Goal:

Improve throughput for larger datasets.

Tasks:

* Split input into batches.
* Use pinned host memory.
* Use multiple CUDA streams.
* Overlap host-to-device copies, kernel execution, and device-to-host copies.
* Benchmark stream-based pipeline vs non-stream pipeline.

Deliverables:

```text
src/batched_hamming_streams.cu
src/batched_smith_waterman_streams.cu
docs/cuda_streams.md
```

Expected pipeline:

```text
Batch N+1: copy to GPU
Batch N: compute on GPU
Batch N-1: copy results back to CPU
```

---

### Phase 12: Final Documentation and Portfolio Polish

Goal:

Make the project recruiter-friendly and AI-agent-readable.

Tasks:

* Write final README.
* Add architecture diagrams.
* Add benchmark charts.
* Add Nsight profiling screenshots if available.
* Add explanation of memory layout.
* Add explanation of kernel design.
* Add correctness validation report.
* Add limitations and future work.

Deliverables:

```text
README.md
docs/architecture.md
docs/memory_layout.md
docs/profiling.md
docs/final_report.md
assets/benchmark_charts/
assets/demo_outputs/
```

---

## 12. Suggested Repository Structure

```text
cuda-bioinformatics-accelerator/
│
├── README.md
│
├── src/
│   ├── hello_cuda.cu
│   ├── vector_add.cu
│   ├── hamming_cpu.cpp
│   ├── hamming_gpu.cu
│   ├── hamming_gpu_encoded.cu
│   ├── dna_encoding.cpp
│   ├── needleman_wunsch_cpu.cpp
│   ├── smith_waterman_cpu.cpp
│   ├── smith_waterman_gpu.cu
│   └── common/
│       ├── cuda_utils.h
│       ├── timer.h
│       ├── dna_utils.h
│       └── file_io.h
│
├── scripts/
│   ├── generate_synthetic_dataset.py
│   ├── download_datasets.py
│   ├── fragment_fasta.py
│   ├── run_hamming_benchmarks.py
│   └── plot_benchmarks.py
│
├── data/
│   ├── synthetic/
│   ├── raw/
│   └── processed/
│
├── results/
│   ├── hamming/
│   ├── needleman_wunsch/
│   └── smith_waterman/
│
├── benchmarks/
│   ├── hamming_benchmark_results.csv
│   ├── smith_waterman_benchmark_results.csv
│   └── benchmark_summary.md
│
├── notebooks/
│   ├── 01_cuda_environment_setup.ipynb
│   ├── 02_hamming_benchmark_analysis.ipynb
│   └── 03_real_dataset_analysis.ipynb
│
├── docs/
│   ├── environment_setup.md
│   ├── architecture.md
│   ├── datasets.md
│   ├── memory_layout.md
│   ├── cuda_kernels.md
│   ├── dna_encoding.md
│   ├── smith_waterman_cuda.md
│   ├── cuda_streams.md
│   └── profiling.md
│
├── tests/
│   ├── test_hamming.cpp
│   ├── test_dna_encoding.cpp
│   ├── test_needleman_wunsch.cpp
│   └── test_smith_waterman.cpp
│
├── assets/
│   ├── benchmark_charts/
│   ├── architecture_diagrams/
│   └── profiling_screenshots/
│
├── CMakeLists.txt
└── LICENSE
```

---

## 13. CUDA Coding Requirements

All CUDA code must follow these rules.

### 13.1 Error Checking

Every CUDA API call should be checked.

Use a helper macro like:

```cpp
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = call;                                           \
        if (error != cudaSuccess) {                                         \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__   \
                      << " -> " << cudaGetErrorString(error) << std::endl; \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                  \
    } while (0)
```

### 13.2 Kernel Bounds Checking

Every kernel must check array bounds.

Example:

```cpp
int pairIndex = blockIdx.x * blockDim.x + threadIdx.x;

if (pairIndex >= numberOfPairs) {
    return;
}
```

### 13.3 Naming Convention

Use clear English names.

Good:

```cpp
numberOfPairs
sequenceLength
hostSequenceA
deviceSequenceA
deviceDistances
computeHammingDistanceKernel
```

Bad:

```cpp
n
seq
datos
distanciasGpu
```

### 13.4 Memory Naming Convention

Use prefixes to distinguish memory location.

```cpp
hostSequenceA
hostSequenceB
hostDistances

deviceSequenceA
deviceSequenceB
deviceDistances
```

or:

```cpp
h_sequenceA
h_sequenceB
h_distances

d_sequenceA
d_sequenceB
d_distances
```

Either style is acceptable, but the project must be consistent.

### 13.5 Timing

Use:

* `std::chrono` for CPU timing.
* CUDA events for GPU kernel timing.
* Optional total GPU timing including memory copies.

---

## 14. Memory Layout Requirements

The first GPU implementation should use a flat fixed-length layout.

Example:

```text
Sequence 0: ACGT
Sequence 1: TGCA
Sequence 2: AAAA
```

Stored as:

```text
ACGTTGCAAAAA
```

The offset is:

```cpp
int offset = pairIndex * sequenceLength;
```

Access:

```cpp
char baseA = sequenceA[offset + baseIndex];
char baseB = sequenceB[offset + baseIndex];
```

This layout is simple and GPU-friendly.

---

## 15. Correctness Requirements

Every GPU algorithm must be validated against a CPU baseline.

Required validation:

```text
For every pair:
GPU result == CPU result
```

If results mismatch, the program must report:

```text
pair_id
CPU result
GPU result
input sequence A
input sequence B
```

The project must prioritize correctness before optimization.

---

## 16. Benchmark Requirements

Benchmarks must include at least:

```text
Number of sequence pairs
Sequence length
CPU time
GPU kernel time
GPU total time
Speedup using kernel time
Speedup using total time
Pairs processed per second
Bases compared per second
GPU model
CUDA version
```

Example benchmark table:

| Algorithm |  Pairs | Length | CPU Time (ms) | GPU Kernel (ms) | GPU Total (ms) | Kernel Speedup | Total Speedup |
| --------- | -----: | -----: | ------------: | --------------: | -------------: | -------------: | ------------: |
| Hamming   | 10,000 |    128 |          12.4 |             0.3 |            1.1 |          41.3x |         11.2x |

---

## 17. Profiling Goals

The project should eventually include profiling notes.

Tools:

```text
Nsight Compute
Nsight Systems
nvidia-smi
CUDA events
```

Metrics to discuss:

```text
Achieved occupancy
Global memory throughput
Global load efficiency
Global store efficiency
Warp execution efficiency
Branch efficiency
Kernel duration
Memory transfer overhead
```

---

## 18. Important CUDA Concepts Used in This Project

This project is expected to demonstrate:

* Host vs device memory.
* CUDA kernels.
* Threads, blocks, and grids.
* Global memory.
* Shared memory.
* Constant memory.
* CUDA events.
* CUDA streams.
* Pinned memory.
* Coalesced memory access.
* Occupancy awareness.
* Warp divergence awareness.
* Batched processing.
* Dynamic programming on GPU.
* Wavefront parallelism.
* CPU/GPU correctness validation.

---

## 19. Development Guidelines for AI Coding Agents

An AI agent working on this repository must follow these rules:

1. All code must be written in English.
2. All variable names must be descriptive.
3. All comments must be in English.
4. Do not skip the CPU baseline.
5. Do not optimize before correctness is verified.
6. Do not introduce complex abstractions too early.
7. Start with fixed-length sequences before variable-length sequences.
8. Start with Hamming Distance before Smith-Waterman.
9. Always include CUDA error checking.
10. Always include benchmark output.
11. Always validate GPU results against CPU results.
12. Keep the project compatible with Google Colab whenever possible.
13. Prefer simple `nvcc` compilation first.
14. Add CMake later only if necessary.
15. Avoid using high-level GPU libraries for the core algorithms unless explicitly stated.
16. CUDA kernels should be implemented manually.
17. The README and docs should explain why each optimization exists.
18. Each phase should be implemented as an independent milestone.

---

## 20. Initial Milestone Checklist

The first milestone is complete when the repository can:

* Generate synthetic DNA pairs.
* Run CPU Hamming Distance.
* Run GPU Hamming Distance.
* Validate GPU results against CPU results.
* Export benchmark results.
* Run in Google Colab.
* Show CPU vs GPU timing.

Minimum required files for Milestone 1:

```text
scripts/generate_synthetic_dataset.py
src/hamming_cpu.cpp
src/hamming_gpu.cu
src/common/cuda_utils.h
benchmarks/run_hamming_benchmarks.py
README.md
```

---

## 21. Future Work

Possible future improvements:

* Add 2-bit DNA packing.
* Add variable-length sequence support.
* Add FASTA parser.
* Add Needleman-Wunsch GPU implementation.
* Add Smith-Waterman GPU implementation.
* Add shared-memory tiled dynamic programming.
* Add CUDA streams for batch processing.
* Add pinned memory for faster transfers.
* Add multi-GPU support.
* Add Python bindings with pybind11.
* Add PyTorch extension version for research workflows.
* Add Nsight profiling reports.
* Add GitHub Actions for CPU-only tests.

---

## 22. Final Project Vision

The final version of this project should be presentable as:

```text
A CUDA-accelerated bioinformatics sequence alignment engine that compares large batches of genomic fragments using manually implemented GPU kernels, CPU baselines, correctness validation, and detailed performance benchmarks.
```

The project should communicate strong skills in:

```text
CUDA programming
HPC systems engineering
Parallel algorithm design
GPU memory optimization
Benchmarking
Scientific computing
Bioinformatics fundamentals
C++ engineering
Performance profiling
```

This repository should be built as a serious portfolio project for GPU engineering, HPC, AI infrastructure, scientific computing, and systems programming roles.
