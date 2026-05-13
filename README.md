# CUDA Bioinformatics Accelerator

CUDA Bioinformatics Accelerator is a CUDA/C++ portfolio project for batched DNA sequence comparison. The long-term goal is to build a GPU-accelerated sequence comparison and alignment engine, starting with simple and correct baselines before adding more advanced algorithms.

All code, variable names, function names, comments, documentation, and commit-style notes are written in English.

## Milestone 1: Hamming Distance Baseline

Milestone 1 compares fixed-length DNA sequence pairs using Hamming Distance.

Implemented features:

- Synthetic DNA pair generation with reproducible seeds.
- CPU Hamming Distance baseline.
- CUDA Hamming Distance kernel with one thread per sequence pair.
- GPU result validation against CPU results.
- CSV output with pair id, distance, and similarity percentage.
- Basic CPU/GPU benchmark script for Google Colab.

This milestone intentionally does not implement Needleman-Wunsch or Smith-Waterman.

## Project Structure

```text
.
├── README.md
├── src/
│   ├── common/
│   │   ├── cuda_utils.h
│   │   ├── dna_utils.h
│   │   └── timer.h
│   ├── hamming_cpu.cpp
│   └── hamming_gpu.cu
├── scripts/
│   └── generate_synthetic_dataset.py
├── data/
│   └── synthetic/
├── results/
│   └── hamming/
├── benchmarks/
│   └── run_hamming_benchmark.py
└── notebooks/
    └── 01_colab_setup_and_hamming_baseline.ipynb
```

## Google Colab Quick Start

Check the GPU and CUDA compiler:

```bash
!nvidia-smi
!nvcc --version
```

Generate a synthetic dataset:

```bash
python scripts/generate_synthetic_dataset.py \
  --num-pairs 10000 \
  --sequence-length 128 \
  --output data/synthetic/synthetic_pairs_128.txt \
  --seed 42
```

Compile the CPU and GPU programs:

```bash
g++ src/hamming_cpu.cpp -O3 -std=c++17 -I src/common -o hamming_cpu
nvcc src/hamming_gpu.cu -O3 -std=c++17 -I src/common -o hamming_gpu
```

Run the CPU baseline:

```bash
./hamming_cpu \
  data/synthetic/synthetic_pairs_128.txt \
  results/hamming/hamming_cpu_results.csv
```

Run the GPU baseline:

```bash
./hamming_gpu \
  data/synthetic/synthetic_pairs_128.txt \
  results/hamming/hamming_gpu_results.csv
```

The GPU program validates every GPU distance against a CPU-computed reference before writing results.

## Benchmark

Run the benchmark matrix:

```bash
python benchmarks/run_hamming_benchmark.py
```

The benchmark generates datasets for sequence lengths `64`, `128`, and `256`, with `1000`, `10000`, and `100000` pairs. Results are saved to:

```text
benchmarks/hamming_benchmark_results.csv
```

## Dataset Format

Each line contains two DNA sequences separated by one space:

```text
ACGTACGT ACGTTCGT
```

Milestone 1 assumes fixed-length pairs and uses the alphabet `A`, `C`, `G`, and `T`.
