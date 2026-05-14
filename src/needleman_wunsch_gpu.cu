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

#include "cuda_utils.h"
#include "needleman_wunsch.h"

namespace {

constexpr int MAX_SUPPORTED_SEQUENCE_LENGTH = 64;
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
    std::string implementation = "wavefront";
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

double elapsed_milliseconds(std::chrono::high_resolution_clock::time_point start_time,
                            std::chrono::high_resolution_clock::time_point end_time) {
    return std::chrono::duration<double, std::milli>(end_time - start_time).count();
}

__device__ int device_max3(int first_value, int second_value, int third_value) {
    const int larger_pair_value = first_value > second_value ? first_value : second_value;
    return larger_pair_value > third_value ? larger_pair_value : third_value;
}

__global__ void needlemanWunschBaselineKernel(const char* sequenceA,
                                              const char* sequenceB,
                                              int* scores,
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

    extern __shared__ int dp[];
    const int cols = sequenceLengthB + 1;

    if (threadIdx.x == 0) {
        dp[0] = 0;
        for (int row = 1; row <= sequenceLengthA; ++row) {
            dp[row * cols] = row * gapPenalty;
        }
        for (int col = 1; col <= sequenceLengthB; ++col) {
            dp[col] = col * gapPenalty;
        }

        const int offsetA = pairIndex * sequenceLengthA;
        const int offsetB = pairIndex * sequenceLengthB;
        for (int row = 1; row <= sequenceLengthA; ++row) {
            for (int col = 1; col <= sequenceLengthB; ++col) {
                const int substitutionScore =
                    sequenceA[offsetA + row - 1] == sequenceB[offsetB + col - 1] ? matchScore : mismatchPenalty;
                const int diagonalScore = dp[(row - 1) * cols + (col - 1)] + substitutionScore;
                const int upScore = dp[(row - 1) * cols + col] + gapPenalty;
                const int leftScore = dp[row * cols + (col - 1)] + gapPenalty;
                dp[row * cols + col] = device_max3(diagonalScore, upScore, leftScore);
            }
        }

        scores[pairIndex] = dp[sequenceLengthA * cols + sequenceLengthB];
    }
}

__global__ void needlemanWunschWavefrontKernel(const char* sequenceA,
                                               const char* sequenceB,
                                               int* scores,
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

    extern __shared__ int dp[];
    const int cols = sequenceLengthB + 1;

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

ProgramOptions parse_arguments(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: " + std::string(argv[0]) +
            " <input_dataset> <output_csv> [--match N] [--mismatch N] [--gap N] "
            "[--repetitions N] [--implementation baseline|wavefront] [--summary-only] [--skip-validation]");
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
            if (options.implementation != "baseline" && options.implementation != "wavefront") {
                throw std::runtime_error("--implementation must be baseline or wavefront.");
            }
        } else if (argument == "--summary-only") {
            options.summary_only = true;
        } else if (argument == "--skip-validation") {
            options.skip_validation = true;
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    return options;
}

Dataset read_dataset(const std::string& input_path) {
    std::ifstream input_file(input_path);
    if (!input_file) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    Dataset dataset;
    std::string first_sequence;
    std::string second_sequence;
    bool has_sequence_lengths = false;

    while (input_file >> first_sequence >> second_sequence) {
        if (first_sequence.empty() || second_sequence.empty()) {
            throw std::runtime_error("Input sequence pairs must not contain empty sequences.");
        }
        if (!has_sequence_lengths) {
            if (first_sequence.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                second_sequence.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Sequence length exceeds supported integer range.");
            }
            dataset.sequence_length_a = static_cast<int>(first_sequence.size());
            dataset.sequence_length_b = static_cast<int>(second_sequence.size());
            has_sequence_lengths = true;
        } else if (first_sequence.size() != static_cast<std::size_t>(dataset.sequence_length_a) ||
                   second_sequence.size() != static_cast<std::size_t>(dataset.sequence_length_b)) {
            throw std::runtime_error("ERROR: Phase 10 GPU implementation currently requires fixed-length sequence pairs.");
        }

        dataset.flat_first_sequences.insert(
            dataset.flat_first_sequences.end(),
            first_sequence.begin(),
            first_sequence.end());
        dataset.flat_second_sequences.insert(
            dataset.flat_second_sequences.end(),
            second_sequence.begin(),
            second_sequence.end());
        dataset.pairs.push_back({first_sequence, second_sequence});
    }

    if (dataset.pairs.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }
    if (dataset.pairs.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Number of pairs exceeds supported integer range.");
    }
    if (dataset.sequence_length_a > MAX_SUPPORTED_SEQUENCE_LENGTH ||
        dataset.sequence_length_b > MAX_SUPPORTED_SEQUENCE_LENGTH) {
        throw std::runtime_error("ERROR: Sequence length exceeds MAX_SEQUENCE_LENGTH for the current CUDA prototype.");
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

void launch_kernel(const ProgramOptions& options,
                   const Dataset& dataset,
                   const char* device_first_sequences,
                   const char* device_second_sequences,
                   int* device_scores,
                   cudaEvent_t kernel_start_event,
                   cudaEvent_t kernel_stop_event,
                   float& kernel_time_ms) {
    const std::size_t shared_memory_bytes =
        static_cast<std::size_t>(dataset.sequence_length_a + 1) *
        static_cast<std::size_t>(dataset.sequence_length_b + 1) *
        sizeof(int);

    CUDA_CHECK(cudaEventRecord(kernel_start_event));
    if (options.implementation == "baseline") {
        needlemanWunschBaselineKernel<<<dataset.number_of_pairs, THREADS_PER_BLOCK, shared_memory_bytes>>>(
            device_first_sequences,
            device_second_sequences,
            device_scores,
            dataset.number_of_pairs,
            dataset.sequence_length_a,
            dataset.sequence_length_b,
            options.match_score,
            options.mismatch_penalty,
            options.gap_penalty);
    } else {
        needlemanWunschWavefrontKernel<<<dataset.number_of_pairs, THREADS_PER_BLOCK, shared_memory_bytes>>>(
            device_first_sequences,
            device_second_sequences,
            device_scores,
            dataset.number_of_pairs,
            dataset.sequence_length_a,
            dataset.sequence_length_b,
            options.match_score,
            options.mismatch_penalty,
            options.gap_penalty);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(kernel_stop_event));
    CUDA_CHECK(cudaEventSynchronize(kernel_stop_event));
    CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, kernel_start_event, kernel_stop_event));
}

}  // namespace

int main(int argc, char** argv) {
    char* device_first_sequences = nullptr;
    char* device_second_sequences = nullptr;
    int* device_scores = nullptr;
    cudaEvent_t kernel_start_event = nullptr;
    cudaEvent_t kernel_stop_event = nullptr;

    try {
        const ProgramOptions options = parse_arguments(argc, argv);
        const Dataset dataset = read_dataset(options.input_path);

        std::cout << "Needleman-Wunsch CUDA prototype started.\n";
        std::cout << "Implementation: " << options.implementation << "\n";
        std::cout << "MAX_SUPPORTED_SEQUENCE_LENGTH=" << MAX_SUPPORTED_SEQUENCE_LENGTH << "\n";

        std::vector<int> cpu_scores;
        double cpu_reference_time_ms = 0.0;
        if (!options.skip_validation) {
            const auto cpu_start_time = std::chrono::high_resolution_clock::now();
            cpu_scores = compute_cpu_reference_scores(dataset, options);
            const auto cpu_end_time = std::chrono::high_resolution_clock::now();
            cpu_reference_time_ms = elapsed_milliseconds(cpu_start_time, cpu_end_time);
        } else {
            cpu_scores.assign(static_cast<std::size_t>(dataset.number_of_pairs), 0);
        }

        std::vector<int> gpu_scores(static_cast<std::size_t>(dataset.number_of_pairs), 0);
        const std::size_t first_sequence_bytes = dataset.flat_first_sequences.size() * sizeof(char);
        const std::size_t second_sequence_bytes = dataset.flat_second_sequences.size() * sizeof(char);
        const std::size_t score_bytes = gpu_scores.size() * sizeof(int);

        CUDA_CHECK(cudaMalloc(&device_first_sequences, first_sequence_bytes));
        CUDA_CHECK(cudaMalloc(&device_second_sequences, second_sequence_bytes));
        CUDA_CHECK(cudaMalloc(&device_scores, score_bytes));
        CUDA_CHECK(cudaEventCreate(&kernel_start_event));
        CUDA_CHECK(cudaEventCreate(&kernel_stop_event));

        CUDA_CHECK(cudaMemcpy(
            device_first_sequences,
            dataset.flat_first_sequences.data(),
            first_sequence_bytes,
            cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(
            device_second_sequences,
            dataset.flat_second_sequences.data(),
            second_sequence_bytes,
            cudaMemcpyHostToDevice));
        float warmup_kernel_time_ms = 0.0f;
        launch_kernel(
            options,
            dataset,
            device_first_sequences,
            device_second_sequences,
            device_scores,
            kernel_start_event,
            kernel_stop_event,
            warmup_kernel_time_ms);
        CUDA_CHECK(cudaDeviceSynchronize());

        double total_h2d_copy_time_ms = 0.0;
        double total_kernel_time_ms = 0.0;
        double total_d2h_copy_time_ms = 0.0;

        for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            const auto h2d_start_time = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpy(
                device_first_sequences,
                dataset.flat_first_sequences.data(),
                first_sequence_bytes,
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(
                device_second_sequences,
                dataset.flat_second_sequences.data(),
                second_sequence_bytes,
                cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto h2d_end_time = std::chrono::high_resolution_clock::now();
            total_h2d_copy_time_ms += elapsed_milliseconds(h2d_start_time, h2d_end_time);

            float repetition_kernel_time_ms = 0.0f;
            launch_kernel(
                options,
                dataset,
                device_first_sequences,
                device_second_sequences,
                device_scores,
                kernel_start_event,
                kernel_stop_event,
                repetition_kernel_time_ms);
            total_kernel_time_ms += static_cast<double>(repetition_kernel_time_ms);

            const auto d2h_start_time = std::chrono::high_resolution_clock::now();
            CUDA_CHECK(cudaMemcpy(gpu_scores.data(), device_scores, score_bytes, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto d2h_end_time = std::chrono::high_resolution_clock::now();
            total_d2h_copy_time_ms += elapsed_milliseconds(d2h_start_time, d2h_end_time);
        }

        const double average_h2d_copy_time_ms = total_h2d_copy_time_ms / static_cast<double>(options.repetitions);
        const double average_kernel_time_ms = total_kernel_time_ms / static_cast<double>(options.repetitions);
        const double average_d2h_copy_time_ms = total_d2h_copy_time_ms / static_cast<double>(options.repetitions);
        const double average_gpu_total_time_ms =
            average_h2d_copy_time_ms + average_kernel_time_ms + average_d2h_copy_time_ms;

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

        write_results_csv(
            options.output_path,
            dataset,
            cpu_scores,
            gpu_scores,
            validation_result,
            options.summary_only);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "ALGORITHM=needleman_wunsch_gpu\n";
        std::cout << "IMPLEMENTATION=" << options.implementation << "\n";
        std::cout << "NUMBER_OF_PAIRS=" << dataset.number_of_pairs << "\n";
        std::cout << "SEQUENCE_LENGTH_A=" << dataset.sequence_length_a << "\n";
        std::cout << "SEQUENCE_LENGTH_B=" << dataset.sequence_length_b << "\n";
        std::cout << "MATCH_SCORE=" << options.match_score << "\n";
        std::cout << "MISMATCH_PENALTY=" << options.mismatch_penalty << "\n";
        std::cout << "GAP_PENALTY=" << options.gap_penalty << "\n";
        std::cout << "GPU_KERNEL_TIME_MS=" << average_kernel_time_ms << "\n";
        std::cout << "GPU_TOTAL_TIME_MS=" << average_gpu_total_time_ms << "\n";
        std::cout << "H2D_COPY_TIME_MS=" << average_h2d_copy_time_ms << "\n";
        std::cout << "D2H_COPY_TIME_MS=" << average_d2h_copy_time_ms << "\n";
        std::cout << "CPU_REFERENCE_TIME_MS=" << cpu_reference_time_ms << "\n";
        std::cout << "VALIDATION_TIME_MS=" << validation_time_ms << "\n";
        std::cout << "VALIDATION_STATUS=" << validation_result.status << "\n";
        if (validation_result.status == "FAILED") {
            std::cout << "FIRST_MISMATCH_PAIR_ID=" << validation_result.first_mismatch_pair_id << "\n";
            std::cout << "CPU_SCORE=" << validation_result.cpu_score << "\n";
            std::cout << "GPU_SCORE=" << validation_result.gpu_score << "\n";
        }
        std::cout << "OUTPUT_PATH=" << options.output_path << "\n";

        CUDA_CHECK(cudaEventDestroy(kernel_start_event));
        CUDA_CHECK(cudaEventDestroy(kernel_stop_event));
        CUDA_CHECK(cudaFree(device_first_sequences));
        CUDA_CHECK(cudaFree(device_second_sequences));
        CUDA_CHECK(cudaFree(device_scores));

        return validation_result.status == "FAILED" ? 2 : 0;
    } catch (const std::exception& error) {
        if (kernel_start_event != nullptr) {
            cudaEventDestroy(kernel_start_event);
        }
        if (kernel_stop_event != nullptr) {
            cudaEventDestroy(kernel_stop_event);
        }
        if (device_first_sequences != nullptr) {
            cudaFree(device_first_sequences);
        }
        if (device_second_sequences != nullptr) {
            cudaFree(device_second_sequences);
        }
        if (device_scores != nullptr) {
            cudaFree(device_scores);
        }

        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
