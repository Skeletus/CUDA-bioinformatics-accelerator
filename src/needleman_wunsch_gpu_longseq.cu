#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "needleman_wunsch.h"

namespace {

constexpr int DEFAULT_MAX_SEQUENCE_LENGTH = 1024;
constexpr int THREADS_PER_BLOCK = 128;

struct SequencePair {
    std::string first_sequence;
    std::string second_sequence;
};

struct ProgramOptions {
    std::string input_path;
    std::string output_path;
    int match_score = 2;
    int mismatch_penalty = -1;
    int gap_penalty = -2;
    int repetitions = 5;
    int max_sequence_length = DEFAULT_MAX_SEQUENCE_LENGTH;
    int tile_size = 16;
    std::string implementation = "rolling_diagonal";
    bool summary_only = false;
    bool skip_validation = false;
};

struct Dataset {
    std::vector<SequencePair> pairs;
    std::vector<char> flat_first_sequences;
    std::vector<char> flat_second_sequences;
    int number_of_pairs = 0;
    int sequence_length_a = 0;
    int sequence_length_b = 0;
};

struct ValidationResult {
    std::string status = "SKIPPED";
    int first_mismatch_pair_id = -1;
    int cpu_score = 0;
    int gpu_score = 0;
};

struct DeviceBuffers {
    char* sequence_a = nullptr;
    char* sequence_b = nullptr;
    int* scores = nullptr;
    int* dp = nullptr;
};

double elapsed_milliseconds(std::chrono::high_resolution_clock::time_point start_time,
                            std::chrono::high_resolution_clock::time_point end_time) {
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

std::runtime_error cuda_runtime_error(cudaError_t status, const std::string& context) {
    return std::runtime_error(context + ": " + cudaGetErrorString(status));
}

void throw_if_cuda_error(cudaError_t status, const std::string& context) {
    if (status != cudaSuccess) {
        throw cuda_runtime_error(status, context);
    }
}

__device__ int device_max3(int first_value, int second_value, int third_value) {
    const int larger_pair_value = first_value > second_value ? first_value : second_value;
    return larger_pair_value > third_value ? larger_pair_value : third_value;
}

__global__ void needlemanWunschGlobalMatrixKernel(const char* sequenceA,
                                                  const char* sequenceB,
                                                  int* scores,
                                                  int* dpMatrices,
                                                  int numberOfPairs,
                                                  int sequenceLengthA,
                                                  int sequenceLengthB,
                                                  int matchScore,
                                                  int mismatchPenalty,
                                                  int gapPenalty) {
    const int pairIndex = blockIdx.x;
    if (pairIndex >= numberOfPairs) {
        return;
    }

    const int cols = sequenceLengthB + 1;
    const int matrixCells = (sequenceLengthA + 1) * cols;
    int* dp = dpMatrices + pairIndex * matrixCells;

    for (int row = threadIdx.x; row <= sequenceLengthA; row += blockDim.x) {
        dp[row * cols] = row * gapPenalty;
    }
    for (int col = threadIdx.x; col <= sequenceLengthB; col += blockDim.x) {
        dp[col] = col * gapPenalty;
    }
    __syncthreads();

    const int offsetA = pairIndex * sequenceLengthA;
    const int offsetB = pairIndex * sequenceLengthB;
    for (int diagonal = 2; diagonal <= sequenceLengthA + sequenceLengthB; ++diagonal) {
        const int startRow = (diagonal - sequenceLengthB) > 1 ? (diagonal - sequenceLengthB) : 1;
        const int endRow = sequenceLengthA < (diagonal - 1) ? sequenceLengthA : (diagonal - 1);
        const int cellsOnDiagonal = endRow - startRow + 1;

        for (int cellOffset = threadIdx.x; cellOffset < cellsOnDiagonal; cellOffset += blockDim.x) {
            const int row = startRow + cellOffset;
            const int col = diagonal - row;
            const int substitutionScore =
                sequenceA[offsetA + row - 1] == sequenceB[offsetB + col - 1] ? matchScore : mismatchPenalty;
            const int diagonalScore = dp[(row - 1) * cols + (col - 1)] + substitutionScore;
            const int upScore = dp[(row - 1) * cols + col] + gapPenalty;
            const int leftScore = dp[row * cols + (col - 1)] + gapPenalty;
            dp[row * cols + col] = device_max3(diagonalScore, upScore, leftScore);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores[pairIndex] = dp[sequenceLengthA * cols + sequenceLengthB];
    }
}

__global__ void needlemanWunschRollingDiagonalKernel(const char* sequenceA,
                                                     const char* sequenceB,
                                                     int* scores,
                                                     int* diagonalBuffers,
                                                     int numberOfPairs,
                                                     int sequenceLengthA,
                                                     int sequenceLengthB,
                                                     int matchScore,
                                                     int mismatchPenalty,
                                                     int gapPenalty) {
    const int pairIndex = blockIdx.x;
    if (pairIndex >= numberOfPairs) {
        return;
    }

    const int diagonalLength = sequenceLengthA + 1;
    int* previousPreviousDiagonal = diagonalBuffers + pairIndex * 3 * diagonalLength;
    int* previousDiagonal = previousPreviousDiagonal + diagonalLength;
    int* currentDiagonal = previousDiagonal + diagonalLength;

    if (threadIdx.x == 0) {
        previousDiagonal[0] = 0;
    }
    __syncthreads();

    const int offsetA = pairIndex * sequenceLengthA;
    const int offsetB = pairIndex * sequenceLengthB;
    for (int diagonal = 1; diagonal <= sequenceLengthA + sequenceLengthB; ++diagonal) {
        const int startRow = (diagonal - sequenceLengthB) > 0 ? (diagonal - sequenceLengthB) : 0;
        const int endRow = sequenceLengthA < diagonal ? sequenceLengthA : diagonal;
        const int cellsOnDiagonal = endRow - startRow + 1;

        for (int cellOffset = threadIdx.x; cellOffset < cellsOnDiagonal; cellOffset += blockDim.x) {
            const int row = startRow + cellOffset;
            const int col = diagonal - row;
            if (row == 0) {
                currentDiagonal[row] = col * gapPenalty;
            } else if (col == 0) {
                currentDiagonal[row] = row * gapPenalty;
            } else {
                const int substitutionScore =
                    sequenceA[offsetA + row - 1] == sequenceB[offsetB + col - 1] ? matchScore : mismatchPenalty;
                const int diagonalScore = previousPreviousDiagonal[row - 1] + substitutionScore;
                const int upScore = previousDiagonal[row - 1] + gapPenalty;
                const int leftScore = previousDiagonal[row] + gapPenalty;
                currentDiagonal[row] = device_max3(diagonalScore, upScore, leftScore);
            }
        }
        __syncthreads();

        int* temporaryDiagonal = previousPreviousDiagonal;
        previousPreviousDiagonal = previousDiagonal;
        previousDiagonal = currentDiagonal;
        currentDiagonal = temporaryDiagonal;
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores[pairIndex] = previousDiagonal[sequenceLengthA];
    }
}

__global__ void needlemanWunschTiledWavefrontKernel(const char* sequenceA,
                                                    const char* sequenceB,
                                                    int* scores,
                                                    int* dpMatrices,
                                                    int numberOfPairs,
                                                    int sequenceLengthA,
                                                    int sequenceLengthB,
                                                    int matchScore,
                                                    int mismatchPenalty,
                                                    int gapPenalty,
                                                    int tileSize) {
    const int pairIndex = blockIdx.x;
    if (pairIndex >= numberOfPairs) {
        return;
    }

    const int cols = sequenceLengthB + 1;
    const int matrixCells = (sequenceLengthA + 1) * cols;
    int* dp = dpMatrices + pairIndex * matrixCells;

    for (int row = threadIdx.x; row <= sequenceLengthA; row += blockDim.x) {
        dp[row * cols] = row * gapPenalty;
    }
    for (int col = threadIdx.x; col <= sequenceLengthB; col += blockDim.x) {
        dp[col] = col * gapPenalty;
    }
    __syncthreads();

    const int offsetA = pairIndex * sequenceLengthA;
    const int offsetB = pairIndex * sequenceLengthB;
    const int tileRows = (sequenceLengthA + tileSize - 1) / tileSize;
    const int tileCols = (sequenceLengthB + tileSize - 1) / tileSize;

    for (int tileDiagonal = 0; tileDiagonal <= tileRows + tileCols - 2; ++tileDiagonal) {
        const int startTileRow = (tileDiagonal - tileCols + 1) > 0 ? (tileDiagonal - tileCols + 1) : 0;
        const int endTileRow = (tileRows - 1) < tileDiagonal ? (tileRows - 1) : tileDiagonal;

        for (int tileRow = startTileRow; tileRow <= endTileRow; ++tileRow) {
            const int tileCol = tileDiagonal - tileRow;
            const int rowStart = tileRow * tileSize + 1;
            const int rowEnd = ((tileRow + 1) * tileSize) < sequenceLengthA
                                   ? ((tileRow + 1) * tileSize)
                                   : sequenceLengthA;
            const int colStart = tileCol * tileSize + 1;
            const int colEnd = ((tileCol + 1) * tileSize) < sequenceLengthB
                                   ? ((tileCol + 1) * tileSize)
                                   : sequenceLengthB;
            const int tileHeight = rowEnd - rowStart + 1;
            const int tileWidth = colEnd - colStart + 1;

            for (int localDiagonal = 0; localDiagonal <= tileHeight + tileWidth - 2; ++localDiagonal) {
                const int localStartRow =
                    (localDiagonal - tileWidth + 1) > 0 ? (localDiagonal - tileWidth + 1) : 0;
                const int localEndRow = (tileHeight - 1) < localDiagonal ? (tileHeight - 1) : localDiagonal;
                const int cellsOnDiagonal = localEndRow - localStartRow + 1;

                for (int cellOffset = threadIdx.x; cellOffset < cellsOnDiagonal; cellOffset += blockDim.x) {
                    const int localRow = localStartRow + cellOffset;
                    const int localCol = localDiagonal - localRow;
                    const int row = rowStart + localRow;
                    const int col = colStart + localCol;
                    const int substitutionScore =
                        sequenceA[offsetA + row - 1] == sequenceB[offsetB + col - 1] ? matchScore : mismatchPenalty;
                    const int diagonalScore = dp[(row - 1) * cols + (col - 1)] + substitutionScore;
                    const int upScore = dp[(row - 1) * cols + col] + gapPenalty;
                    const int leftScore = dp[row * cols + (col - 1)] + gapPenalty;
                    dp[row * cols + col] = device_max3(diagonalScore, upScore, leftScore);
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        scores[pairIndex] = dp[sequenceLengthA * cols + sequenceLengthB];
    }
}

ProgramOptions parse_arguments(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: " + std::string(argv[0]) +
            " <input_dataset> <output_csv> [--match N] [--mismatch N] [--gap N] "
            "[--repetitions N] [--implementation global_matrix|rolling_diagonal|tiled_wavefront] "
            "[--summary-only] [--write-results] [--skip-validation] "
            "[--max-sequence-length N] [--tile-size N]");
    }

    ProgramOptions options;
    options.input_path = argv[1];
    options.output_path = argv[2];

    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--match") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--match requires a value.");
            }
            options.match_score = std::stoi(argv[++index]);
        } else if (argument == "--mismatch") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--mismatch requires a value.");
            }
            options.mismatch_penalty = std::stoi(argv[++index]);
        } else if (argument == "--gap") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--gap requires a value.");
            }
            options.gap_penalty = std::stoi(argv[++index]);
        } else if (argument == "--repetitions") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--repetitions requires a value.");
            }
            options.repetitions = std::stoi(argv[++index]);
            if (options.repetitions <= 0) {
                throw std::runtime_error("--repetitions must be greater than zero.");
            }
        } else if (argument == "--implementation") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--implementation requires a value.");
            }
            options.implementation = argv[++index];
            if (options.implementation != "global_matrix" &&
                options.implementation != "rolling_diagonal" &&
                options.implementation != "tiled_wavefront") {
                throw std::runtime_error(
                    "--implementation must be global_matrix, rolling_diagonal, or tiled_wavefront.");
            }
        } else if (argument == "--summary-only") {
            options.summary_only = true;
        } else if (argument == "--write-results") {
            options.summary_only = false;
        } else if (argument == "--skip-validation") {
            options.skip_validation = true;
        } else if (argument == "--max-sequence-length") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--max-sequence-length requires a value.");
            }
            options.max_sequence_length = std::stoi(argv[++index]);
            if (options.max_sequence_length <= 0) {
                throw std::runtime_error("--max-sequence-length must be greater than zero.");
            }
        } else if (argument == "--tile-size") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--tile-size requires a value.");
            }
            options.tile_size = std::stoi(argv[++index]);
            if (options.tile_size <= 0) {
                throw std::runtime_error("--tile-size must be greater than zero.");
            }
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    return options;
}

std::vector<SequencePair> read_sequence_pairs(const std::string& input_path) {
    std::ifstream input_file(input_path);
    if (!input_file) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    std::vector<SequencePair> sequence_pairs;
    std::string first_sequence;
    std::string second_sequence;
    while (input_file >> first_sequence >> second_sequence) {
        if (first_sequence.empty() || second_sequence.empty()) {
            throw std::runtime_error("Input sequence pairs must not contain empty sequences.");
        }
        sequence_pairs.push_back({first_sequence, second_sequence});
    }

    if (sequence_pairs.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }
    return sequence_pairs;
}

Dataset validate_and_flatten_dataset(std::vector<SequencePair> sequence_pairs, const ProgramOptions& options) {
    Dataset dataset;
    dataset.pairs = std::move(sequence_pairs);

    if (dataset.pairs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Number of pairs exceeds supported integer range.");
    }

    const std::size_t first_length = dataset.pairs.front().first_sequence.size();
    const std::size_t second_length = dataset.pairs.front().second_sequence.size();
    if (first_length > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        second_length > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Sequence length exceeds supported integer range.");
    }

    dataset.sequence_length_a = static_cast<int>(first_length);
    dataset.sequence_length_b = static_cast<int>(second_length);
    if (dataset.sequence_length_a > options.max_sequence_length ||
        dataset.sequence_length_b > options.max_sequence_length) {
        throw std::runtime_error(
            "ERROR: Sequence length exceeds supported limit for implementation " + options.implementation + ".");
    }

    dataset.flat_first_sequences.reserve(dataset.pairs.size() * first_length);
    dataset.flat_second_sequences.reserve(dataset.pairs.size() * second_length);
    for (const SequencePair& pair : dataset.pairs) {
        if (pair.first_sequence.size() != first_length || pair.second_sequence.size() != second_length) {
            throw std::runtime_error("ERROR: Phase 10.2 currently requires fixed-length sequence pairs.");
        }
        dataset.flat_first_sequences.insert(
            dataset.flat_first_sequences.end(),
            pair.first_sequence.begin(),
            pair.first_sequence.end());
        dataset.flat_second_sequences.insert(
            dataset.flat_second_sequences.end(),
            pair.second_sequence.begin(),
            pair.second_sequence.end());
    }

    dataset.number_of_pairs = static_cast<int>(dataset.pairs.size());
    return dataset;
}

std::vector<int> compute_cpu_reference_scores(const Dataset& dataset, const ProgramOptions& options) {
    std::vector<int> cpu_scores;
    cpu_scores.reserve(dataset.pairs.size());
    for (const SequencePair& pair : dataset.pairs) {
        cpu_scores.push_back(needleman_wunsch::computeNeedlemanWunschScore(
            pair.first_sequence,
            pair.second_sequence,
            options.match_score,
            options.mismatch_penalty,
            options.gap_penalty));
    }
    return cpu_scores;
}

ValidationResult validate_results(const std::vector<int>& cpu_scores, const std::vector<int>& gpu_scores) {
    ValidationResult result;
    result.status = "PASSED";
    for (std::size_t pair_index = 0; pair_index < cpu_scores.size(); ++pair_index) {
        if (cpu_scores[pair_index] != gpu_scores[pair_index]) {
            result.status = "FAILED";
            result.first_mismatch_pair_id = static_cast<int>(pair_index);
            result.cpu_score = cpu_scores[pair_index];
            result.gpu_score = gpu_scores[pair_index];
            return result;
        }
    }
    return result;
}

void write_results_csv(const std::string& output_path,
                       const Dataset& dataset,
                       const std::vector<int>& cpu_scores,
                       const std::vector<int>& gpu_scores,
                       const ValidationResult& validation_result,
                       bool summary_only) {
    const std::filesystem::path output_file_path(output_path);
    if (output_file_path.has_parent_path()) {
        std::filesystem::create_directories(output_file_path.parent_path());
    }

    std::ofstream output_file(output_path);
    if (!output_file) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    if (summary_only) {
        output_file << "metric,value\n";
        output_file << "number_of_pairs," << dataset.number_of_pairs << "\n";
        output_file << "sequence_length_a," << dataset.sequence_length_a << "\n";
        output_file << "sequence_length_b," << dataset.sequence_length_b << "\n";
        output_file << "validation_status," << validation_result.status << "\n";
        return;
    }

    output_file << "pair_id,sequence_length_a,sequence_length_b,cpu_score,gpu_score,match_status\n";
    for (std::size_t pair_index = 0; pair_index < gpu_scores.size(); ++pair_index) {
        const std::string match_status =
            validation_result.status == "SKIPPED" ? "SKIPPED" :
            (cpu_scores[pair_index] == gpu_scores[pair_index] ? "PASSED" : "FAILED");
        output_file << pair_index << ","
                    << dataset.sequence_length_a << ","
                    << dataset.sequence_length_b << ",";
        if (validation_result.status == "SKIPPED") {
            output_file << ",";
        } else {
            output_file << cpu_scores[pair_index] << ",";
        }
        output_file << gpu_scores[pair_index] << ","
                    << match_status << "\n";
    }
}

std::size_t compute_dp_memory_bytes(const Dataset& dataset, const ProgramOptions& options) {
    if (options.implementation == "rolling_diagonal") {
        return static_cast<std::size_t>(dataset.number_of_pairs) *
               3ULL *
               static_cast<std::size_t>(dataset.sequence_length_a + 1) *
               sizeof(int);
    }

    return static_cast<std::size_t>(dataset.number_of_pairs) *
           static_cast<std::size_t>(dataset.sequence_length_a + 1) *
           static_cast<std::size_t>(dataset.sequence_length_b + 1) *
           sizeof(int);
}

std::string implementation_status(const ProgramOptions& options) {
    return options.implementation == "tiled_wavefront" ? "EXPERIMENTAL" : "STABLE";
}

std::string dp_memory_mode(const ProgramOptions& options) {
    if (options.implementation == "rolling_diagonal") {
        return "rolling_diagonal";
    }
    return "global";
}

void ensure_available_device_memory(std::size_t required_bytes) {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    throw_if_cuda_error(cudaMemGetInfo(&free_bytes, &total_bytes), "Failed to query CUDA memory");
    if (required_bytes > free_bytes) {
        throw std::runtime_error("ERROR: Required DP memory exceeds currently available device memory.");
    }
}

void allocate_device_buffers(DeviceBuffers& buffers,
                             const Dataset& dataset,
                             std::size_t dp_memory_bytes) {
    const std::size_t first_sequence_bytes = dataset.flat_first_sequences.size() * sizeof(char);
    const std::size_t second_sequence_bytes = dataset.flat_second_sequences.size() * sizeof(char);
    const std::size_t score_bytes = static_cast<std::size_t>(dataset.number_of_pairs) * sizeof(int);

    throw_if_cuda_error(
        cudaMalloc(reinterpret_cast<void**>(&buffers.sequence_a), first_sequence_bytes),
        "cudaMalloc failed for sequence A");
    throw_if_cuda_error(
        cudaMalloc(reinterpret_cast<void**>(&buffers.sequence_b), second_sequence_bytes),
        "cudaMalloc failed for sequence B");
    throw_if_cuda_error(
        cudaMalloc(reinterpret_cast<void**>(&buffers.scores), score_bytes),
        "cudaMalloc failed for scores");
    throw_if_cuda_error(
        cudaMalloc(reinterpret_cast<void**>(&buffers.dp), dp_memory_bytes),
        "cudaMalloc failed for DP memory");
}

void free_device_buffers(DeviceBuffers& buffers) {
    if (buffers.sequence_a != nullptr) {
        throw_if_cuda_error(cudaFree(buffers.sequence_a), "cudaFree failed for sequence A");
        buffers.sequence_a = nullptr;
    }
    if (buffers.sequence_b != nullptr) {
        throw_if_cuda_error(cudaFree(buffers.sequence_b), "cudaFree failed for sequence B");
        buffers.sequence_b = nullptr;
    }
    if (buffers.scores != nullptr) {
        throw_if_cuda_error(cudaFree(buffers.scores), "cudaFree failed for scores");
        buffers.scores = nullptr;
    }
    if (buffers.dp != nullptr) {
        throw_if_cuda_error(cudaFree(buffers.dp), "cudaFree failed for DP memory");
        buffers.dp = nullptr;
    }
}

void best_effort_free_device_buffers(DeviceBuffers& buffers) {
    if (buffers.sequence_a != nullptr) {
        cudaFree(buffers.sequence_a);
        buffers.sequence_a = nullptr;
    }
    if (buffers.sequence_b != nullptr) {
        cudaFree(buffers.sequence_b);
        buffers.sequence_b = nullptr;
    }
    if (buffers.scores != nullptr) {
        cudaFree(buffers.scores);
        buffers.scores = nullptr;
    }
    if (buffers.dp != nullptr) {
        cudaFree(buffers.dp);
        buffers.dp = nullptr;
    }
}

void launch_kernel(const ProgramOptions& options,
                   const Dataset& dataset,
                   const DeviceBuffers& buffers) {
    if (options.implementation == "global_matrix") {
        needlemanWunschGlobalMatrixKernel<<<dataset.number_of_pairs, THREADS_PER_BLOCK>>>(
            buffers.sequence_a,
            buffers.sequence_b,
            buffers.scores,
            buffers.dp,
            dataset.number_of_pairs,
            dataset.sequence_length_a,
            dataset.sequence_length_b,
            options.match_score,
            options.mismatch_penalty,
            options.gap_penalty);
    } else if (options.implementation == "rolling_diagonal") {
        needlemanWunschRollingDiagonalKernel<<<dataset.number_of_pairs, THREADS_PER_BLOCK>>>(
            buffers.sequence_a,
            buffers.sequence_b,
            buffers.scores,
            buffers.dp,
            dataset.number_of_pairs,
            dataset.sequence_length_a,
            dataset.sequence_length_b,
            options.match_score,
            options.mismatch_penalty,
            options.gap_penalty);
    } else {
        needlemanWunschTiledWavefrontKernel<<<dataset.number_of_pairs, THREADS_PER_BLOCK>>>(
            buffers.sequence_a,
            buffers.sequence_b,
            buffers.scores,
            buffers.dp,
            dataset.number_of_pairs,
            dataset.sequence_length_a,
            dataset.sequence_length_b,
            options.match_score,
            options.mismatch_penalty,
            options.gap_penalty,
            options.tile_size);
    }
    throw_if_cuda_error(cudaGetLastError(), "Needleman-Wunsch long-sequence kernel launch failed");
}

double safe_cells_per_second(long long total_cells, double milliseconds) {
    if (milliseconds <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(total_cells) / (milliseconds / 1000.0);
}

}  // namespace

int main(int argc, char** argv) {
    const auto end_to_end_start_time = std::chrono::high_resolution_clock::now();
    DeviceBuffers device_buffers;
    cudaEvent_t kernel_start_event = nullptr;
    cudaEvent_t kernel_stop_event = nullptr;

    try {
        const ProgramOptions options = parse_arguments(argc, argv);

        const auto file_read_start_time = std::chrono::high_resolution_clock::now();
        std::vector<SequencePair> sequence_pairs = read_sequence_pairs(options.input_path);
        const auto file_read_end_time = std::chrono::high_resolution_clock::now();

        const auto input_validation_start_time = std::chrono::high_resolution_clock::now();
        const Dataset dataset = validate_and_flatten_dataset(std::move(sequence_pairs), options);
        const auto input_validation_end_time = std::chrono::high_resolution_clock::now();

        const std::size_t dp_memory_bytes = compute_dp_memory_bytes(dataset, options);
        ensure_available_device_memory(dp_memory_bytes);
        allocate_device_buffers(device_buffers, dataset, dp_memory_bytes);
        throw_if_cuda_error(cudaEventCreate(&kernel_start_event), "Failed to create kernel start event");
        throw_if_cuda_error(cudaEventCreate(&kernel_stop_event), "Failed to create kernel stop event");

        std::vector<int> gpu_scores(static_cast<std::size_t>(dataset.number_of_pairs), 0);
        const std::size_t first_sequence_bytes = dataset.flat_first_sequences.size() * sizeof(char);
        const std::size_t second_sequence_bytes = dataset.flat_second_sequences.size() * sizeof(char);
        const std::size_t score_bytes = gpu_scores.size() * sizeof(int);

        double total_h2d_copy_time_ms = 0.0;
        double total_kernel_time_ms = 0.0;
        double total_d2h_copy_time_ms = 0.0;
        for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            const auto h2d_start_time = std::chrono::high_resolution_clock::now();
            throw_if_cuda_error(
                cudaMemcpy(
                    device_buffers.sequence_a,
                    dataset.flat_first_sequences.data(),
                    first_sequence_bytes,
                    cudaMemcpyHostToDevice),
                "H2D copy failed for sequence A");
            throw_if_cuda_error(
                cudaMemcpy(
                    device_buffers.sequence_b,
                    dataset.flat_second_sequences.data(),
                    second_sequence_bytes,
                    cudaMemcpyHostToDevice),
                "H2D copy failed for sequence B");
            throw_if_cuda_error(cudaDeviceSynchronize(), "Failed to synchronize after H2D copy");
            const auto h2d_end_time = std::chrono::high_resolution_clock::now();
            total_h2d_copy_time_ms += elapsed_milliseconds(h2d_start_time, h2d_end_time);

            throw_if_cuda_error(cudaEventRecord(kernel_start_event), "Failed to record kernel start event");
            launch_kernel(options, dataset, device_buffers);
            throw_if_cuda_error(cudaEventRecord(kernel_stop_event), "Failed to record kernel stop event");
            throw_if_cuda_error(cudaEventSynchronize(kernel_stop_event), "Failed to synchronize kernel stop event");
            float kernel_time_ms = 0.0f;
            throw_if_cuda_error(
                cudaEventElapsedTime(&kernel_time_ms, kernel_start_event, kernel_stop_event),
                "Failed to measure kernel time");
            total_kernel_time_ms += static_cast<double>(kernel_time_ms);

            const auto d2h_start_time = std::chrono::high_resolution_clock::now();
            throw_if_cuda_error(
                cudaMemcpy(gpu_scores.data(), device_buffers.scores, score_bytes, cudaMemcpyDeviceToHost),
                "D2H copy failed for scores");
            throw_if_cuda_error(cudaDeviceSynchronize(), "Failed to synchronize after D2H copy");
            const auto d2h_end_time = std::chrono::high_resolution_clock::now();
            total_d2h_copy_time_ms += elapsed_milliseconds(d2h_start_time, d2h_end_time);
        }

        const double average_h2d_copy_time_ms =
            total_h2d_copy_time_ms / static_cast<double>(options.repetitions);
        const double average_kernel_time_ms =
            total_kernel_time_ms / static_cast<double>(options.repetitions);
        const double average_d2h_copy_time_ms =
            total_d2h_copy_time_ms / static_cast<double>(options.repetitions);
        const double average_gpu_total_time_ms =
            average_h2d_copy_time_ms + average_kernel_time_ms + average_d2h_copy_time_ms;

        std::vector<int> cpu_scores;
        double cpu_reference_time_ms = 0.0;
        if (!options.skip_validation) {
            const auto cpu_reference_start_time = std::chrono::high_resolution_clock::now();
            cpu_scores = compute_cpu_reference_scores(dataset, options);
            const auto cpu_reference_end_time = std::chrono::high_resolution_clock::now();
            cpu_reference_time_ms = elapsed_milliseconds(cpu_reference_start_time, cpu_reference_end_time);
        } else {
            cpu_scores.assign(static_cast<std::size_t>(dataset.number_of_pairs), 0);
        }

        ValidationResult validation_result;
        double validation_time_ms = 0.0;
        if (options.skip_validation) {
            validation_result.status = "SKIPPED";
        } else {
            const auto validation_start_time = std::chrono::high_resolution_clock::now();
            validation_result = validate_results(cpu_scores, gpu_scores);
            const auto validation_end_time = std::chrono::high_resolution_clock::now();
            validation_time_ms = elapsed_milliseconds(validation_start_time, validation_end_time);
        }

        const auto csv_write_start_time = std::chrono::high_resolution_clock::now();
        write_results_csv(
            options.output_path,
            dataset,
            cpu_scores,
            gpu_scores,
            validation_result,
            options.summary_only);
        const auto csv_write_end_time = std::chrono::high_resolution_clock::now();

        if (kernel_start_event != nullptr) {
            throw_if_cuda_error(cudaEventDestroy(kernel_start_event), "Failed to destroy kernel start event");
            kernel_start_event = nullptr;
        }
        if (kernel_stop_event != nullptr) {
            throw_if_cuda_error(cudaEventDestroy(kernel_stop_event), "Failed to destroy kernel stop event");
            kernel_stop_event = nullptr;
        }
        free_device_buffers(device_buffers);

        const long long total_cells_computed =
            static_cast<long long>(dataset.number_of_pairs) *
            static_cast<long long>(dataset.sequence_length_a + 1) *
            static_cast<long long>(dataset.sequence_length_b + 1);
        const auto end_to_end_end_time = std::chrono::high_resolution_clock::now();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "ALGORITHM=needleman_wunsch_gpu_longseq\n";
        std::cout << "IMPLEMENTATION=" << options.implementation << "\n";
        std::cout << "IMPLEMENTATION_STATUS=" << implementation_status(options) << "\n";
        std::cout << "NUMBER_OF_PAIRS=" << dataset.number_of_pairs << "\n";
        std::cout << "SEQUENCE_LENGTH_A=" << dataset.sequence_length_a << "\n";
        std::cout << "SEQUENCE_LENGTH_B=" << dataset.sequence_length_b << "\n";
        std::cout << "MATCH_SCORE=" << options.match_score << "\n";
        std::cout << "MISMATCH_PENALTY=" << options.mismatch_penalty << "\n";
        std::cout << "GAP_PENALTY=" << options.gap_penalty << "\n";
        std::cout << "SUMMARY_ONLY=" << (options.summary_only ? "true" : "false") << "\n";
        std::cout << "SUPPORTED_SEQUENCE_LENGTH=true\n";
        std::cout << "FILE_READ_TIME_MS=" << elapsed_milliseconds(file_read_start_time, file_read_end_time) << "\n";
        std::cout << "INPUT_VALIDATION_TIME_MS="
                  << elapsed_milliseconds(input_validation_start_time, input_validation_end_time) << "\n";
        std::cout << "H2D_COPY_TIME_MS=" << average_h2d_copy_time_ms << "\n";
        std::cout << "GPU_KERNEL_TIME_MS=" << average_kernel_time_ms << "\n";
        std::cout << "D2H_COPY_TIME_MS=" << average_d2h_copy_time_ms << "\n";
        std::cout << "GPU_TOTAL_TIME_MS=" << average_gpu_total_time_ms << "\n";
        std::cout << "CPU_REFERENCE_TIME_MS=" << cpu_reference_time_ms << "\n";
        std::cout << "VALIDATION_TIME_MS=" << validation_time_ms << "\n";
        std::cout << "CSV_WRITE_TIME_MS=" << elapsed_milliseconds(csv_write_start_time, csv_write_end_time) << "\n";
        std::cout << "END_TO_END_TIME_MS="
                  << elapsed_milliseconds(end_to_end_start_time, end_to_end_end_time) << "\n";
        std::cout << "TOTAL_CELLS_COMPUTED=" << total_cells_computed << "\n";
        std::cout << "GPU_KERNEL_CELLS_PER_SECOND="
                  << safe_cells_per_second(total_cells_computed, average_kernel_time_ms) << "\n";
        std::cout << "GPU_TOTAL_CELLS_PER_SECOND="
                  << safe_cells_per_second(total_cells_computed, average_gpu_total_time_ms) << "\n";
        std::cout << "DP_MEMORY_MODE=" << dp_memory_mode(options) << "\n";
        std::cout << "DP_MEMORY_BYTES=" << dp_memory_bytes << "\n";
        std::cout << "MAX_SUPPORTED_SEQUENCE_LENGTH=" << options.max_sequence_length << "\n";
        if (options.implementation == "rolling_diagonal") {
            std::cout << "ROLLING_BUFFER_COUNT=3\n";
        }
        if (options.implementation == "tiled_wavefront") {
            std::cout << "TILE_SIZE=" << options.tile_size << "\n";
        }
        std::cout << "VALIDATION_STATUS=" << validation_result.status << "\n";
        if (validation_result.status == "FAILED") {
            std::cout << "FIRST_MISMATCH_PAIR_ID=" << validation_result.first_mismatch_pair_id << "\n";
            std::cout << "CPU_SCORE=" << validation_result.cpu_score << "\n";
            std::cout << "GPU_SCORE=" << validation_result.gpu_score << "\n";
        }
        std::cout << "OUTPUT_PATH=" << options.output_path << "\n";

        return validation_result.status == "FAILED" ? 2 : 0;
    } catch (const std::exception& error) {
        if (kernel_start_event != nullptr) {
            cudaEventDestroy(kernel_start_event);
        }
        if (kernel_stop_event != nullptr) {
            cudaEventDestroy(kernel_stop_event);
        }
        best_effort_free_device_buffers(device_buffers);
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
