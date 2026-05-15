#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

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
    int batch_size = 1024;
    int num_streams = 2;
    bool summary_only = false;
    bool skip_validation = false;
    bool prefer_cuda_malloc_async = true;
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

struct StreamContext {
    cudaStream_t stream = nullptr;
    char* host_sequence_a = nullptr;
    char* host_sequence_b = nullptr;
    int* host_scores = nullptr;
    char* device_sequence_a = nullptr;
    char* device_sequence_b = nullptr;
    int* device_scores = nullptr;
    cudaEvent_t h2d_start_event = nullptr;
    cudaEvent_t h2d_stop_event = nullptr;
    cudaEvent_t kernel_start_event = nullptr;
    cudaEvent_t kernel_stop_event = nullptr;
    cudaEvent_t d2h_start_event = nullptr;
    cudaEvent_t d2h_stop_event = nullptr;
    bool has_pending_batch = false;
    int pending_pair_offset = 0;
    int pending_pair_count = 0;
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
            "[--repetitions N] [--summary-only] [--write-results] [--skip-validation] "
            "[--batch-size N] [--num-streams N] [--use-cuda-malloc-async] "
            "[--disable-cuda-malloc-async]");
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
        } else if (argument == "--summary-only") {
            options.summary_only = true;
        } else if (argument == "--write-results") {
            options.summary_only = false;
        } else if (argument == "--skip-validation") {
            options.skip_validation = true;
        } else if (argument == "--batch-size") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--batch-size requires a value.");
            }
            options.batch_size = std::stoi(argv[++index]);
            if (options.batch_size <= 0) {
                throw std::runtime_error("--batch-size must be greater than zero.");
            }
        } else if (argument == "--num-streams") {
            if (index + 1 >= argc) {
                throw std::runtime_error("--num-streams requires a value.");
            }
            options.num_streams = std::stoi(argv[++index]);
            if (options.num_streams <= 0) {
                throw std::runtime_error("--num-streams must be greater than zero.");
            }
        } else if (argument == "--use-cuda-malloc-async") {
            options.prefer_cuda_malloc_async = true;
        } else if (argument == "--disable-cuda-malloc-async") {
            options.prefer_cuda_malloc_async = false;
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

Dataset validate_and_flatten_dataset(std::vector<SequencePair> sequence_pairs) {
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
    if (dataset.sequence_length_a > MAX_SUPPORTED_SEQUENCE_LENGTH ||
        dataset.sequence_length_b > MAX_SUPPORTED_SEQUENCE_LENGTH) {
        throw std::runtime_error(
            "ERROR: Sequence length exceeds MAX_SEQUENCE_LENGTH for the current CUDA prototype.");
    }

    dataset.flat_first_sequences.reserve(dataset.pairs.size() * first_length);
    dataset.flat_second_sequences.reserve(dataset.pairs.size() * second_length);
    for (const SequencePair& pair : dataset.pairs) {
        if (pair.first_sequence.size() != first_length || pair.second_sequence.size() != second_length) {
            throw std::runtime_error(
                "ERROR: Phase 10.1 GPU implementation currently requires fixed-length sequence pairs.");
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

bool cuda_malloc_async_supported() {
#if CUDART_VERSION >= 11020
    int device_id = 0;
    throw_if_cuda_error(cudaGetDevice(&device_id), "Failed to query current CUDA device");
    int memory_pools_supported = 0;
    throw_if_cuda_error(
        cudaDeviceGetAttribute(&memory_pools_supported, cudaDevAttrMemoryPoolsSupported, device_id),
        "Failed to query CUDA memory pool support");
    return memory_pools_supported != 0;
#else
    return false;
#endif
}

void create_stream_resources(std::vector<StreamContext>& stream_contexts) {
    for (StreamContext& context : stream_contexts) {
        throw_if_cuda_error(cudaStreamCreate(&context.stream), "Failed to create CUDA stream");
        throw_if_cuda_error(cudaEventCreate(&context.h2d_start_event), "Failed to create H2D start event");
        throw_if_cuda_error(cudaEventCreate(&context.h2d_stop_event), "Failed to create H2D stop event");
        throw_if_cuda_error(cudaEventCreate(&context.kernel_start_event), "Failed to create kernel start event");
        throw_if_cuda_error(cudaEventCreate(&context.kernel_stop_event), "Failed to create kernel stop event");
        throw_if_cuda_error(cudaEventCreate(&context.d2h_start_event), "Failed to create D2H start event");
        throw_if_cuda_error(cudaEventCreate(&context.d2h_stop_event), "Failed to create D2H stop event");
    }
}

void allocate_pinned_host_buffers(std::vector<StreamContext>& stream_contexts,
                                  std::size_t first_sequence_bytes,
                                  std::size_t second_sequence_bytes,
                                  std::size_t score_bytes) {
    for (StreamContext& context : stream_contexts) {
        throw_if_cuda_error(
            cudaMallocHost(reinterpret_cast<void**>(&context.host_sequence_a), first_sequence_bytes),
            "Pinned host allocation failed for hostSequenceA");
        throw_if_cuda_error(
            cudaMallocHost(reinterpret_cast<void**>(&context.host_sequence_b), second_sequence_bytes),
            "Pinned host allocation failed for hostSequenceB");
        throw_if_cuda_error(
            cudaMallocHost(reinterpret_cast<void**>(&context.host_scores), score_bytes),
            "Pinned host allocation failed for hostScores");
    }
}

void allocate_device_buffers(std::vector<StreamContext>& stream_contexts,
                             std::size_t first_sequence_bytes,
                             std::size_t second_sequence_bytes,
                             std::size_t score_bytes,
                             bool use_cuda_malloc_async) {
    for (StreamContext& context : stream_contexts) {
#if CUDART_VERSION >= 11020
        if (use_cuda_malloc_async) {
            throw_if_cuda_error(
                cudaMallocAsync(reinterpret_cast<void**>(&context.device_sequence_a), first_sequence_bytes, context.stream),
                "cudaMallocAsync failed for deviceSequenceA");
            throw_if_cuda_error(
                cudaMallocAsync(reinterpret_cast<void**>(&context.device_sequence_b), second_sequence_bytes, context.stream),
                "cudaMallocAsync failed for deviceSequenceB");
            throw_if_cuda_error(
                cudaMallocAsync(reinterpret_cast<void**>(&context.device_scores), score_bytes, context.stream),
                "cudaMallocAsync failed for deviceScores");
            continue;
        }
#endif
        throw_if_cuda_error(
            cudaMalloc(reinterpret_cast<void**>(&context.device_sequence_a), first_sequence_bytes),
            "cudaMalloc failed for deviceSequenceA");
        throw_if_cuda_error(
            cudaMalloc(reinterpret_cast<void**>(&context.device_sequence_b), second_sequence_bytes),
            "cudaMalloc failed for deviceSequenceB");
        throw_if_cuda_error(
            cudaMalloc(reinterpret_cast<void**>(&context.device_scores), score_bytes),
            "cudaMalloc failed for deviceScores");
    }

    if (use_cuda_malloc_async) {
        for (StreamContext& context : stream_contexts) {
            throw_if_cuda_error(cudaStreamSynchronize(context.stream), "Failed to synchronize async allocations");
        }
    }
}

void copy_batch_to_pinned_buffers(const Dataset& dataset,
                                  int pair_offset,
                                  int pair_count,
                                  StreamContext& context) {
    const std::size_t first_sequence_offset =
        static_cast<std::size_t>(pair_offset) * static_cast<std::size_t>(dataset.sequence_length_a);
    const std::size_t second_sequence_offset =
        static_cast<std::size_t>(pair_offset) * static_cast<std::size_t>(dataset.sequence_length_b);
    const std::size_t first_sequence_bytes =
        static_cast<std::size_t>(pair_count) * static_cast<std::size_t>(dataset.sequence_length_a) * sizeof(char);
    const std::size_t second_sequence_bytes =
        static_cast<std::size_t>(pair_count) * static_cast<std::size_t>(dataset.sequence_length_b) * sizeof(char);

    std::memcpy(
        context.host_sequence_a,
        dataset.flat_first_sequences.data() + first_sequence_offset,
        first_sequence_bytes);
    std::memcpy(
        context.host_sequence_b,
        dataset.flat_second_sequences.data() + second_sequence_offset,
        second_sequence_bytes);
}

void launch_batch(const ProgramOptions& options,
                  const Dataset& dataset,
                  int pair_offset,
                  int pair_count,
                  StreamContext& context) {
    const std::size_t first_sequence_bytes =
        static_cast<std::size_t>(pair_count) * static_cast<std::size_t>(dataset.sequence_length_a) * sizeof(char);
    const std::size_t second_sequence_bytes =
        static_cast<std::size_t>(pair_count) * static_cast<std::size_t>(dataset.sequence_length_b) * sizeof(char);
    const std::size_t score_bytes = static_cast<std::size_t>(pair_count) * sizeof(int);
    const std::size_t shared_memory_bytes =
        static_cast<std::size_t>(dataset.sequence_length_a + 1) *
        static_cast<std::size_t>(dataset.sequence_length_b + 1) *
        sizeof(int);

    throw_if_cuda_error(cudaEventRecord(context.h2d_start_event, context.stream), "Failed to record H2D start event");
    throw_if_cuda_error(
        cudaMemcpyAsync(
            context.device_sequence_a,
            context.host_sequence_a,
            first_sequence_bytes,
            cudaMemcpyHostToDevice,
            context.stream),
        "Async H2D copy failed for sequence A");
    throw_if_cuda_error(
        cudaMemcpyAsync(
            context.device_sequence_b,
            context.host_sequence_b,
            second_sequence_bytes,
            cudaMemcpyHostToDevice,
            context.stream),
        "Async H2D copy failed for sequence B");
    throw_if_cuda_error(cudaEventRecord(context.h2d_stop_event, context.stream), "Failed to record H2D stop event");

    throw_if_cuda_error(
        cudaEventRecord(context.kernel_start_event, context.stream),
        "Failed to record kernel start event");
    needlemanWunschWavefrontKernel<<<pair_count, THREADS_PER_BLOCK, shared_memory_bytes, context.stream>>>(
        context.device_sequence_a,
        context.device_sequence_b,
        context.device_scores,
        pair_count,
        dataset.sequence_length_a,
        dataset.sequence_length_b,
        options.match_score,
        options.mismatch_penalty,
        options.gap_penalty);
    throw_if_cuda_error(cudaGetLastError(), "Needleman-Wunsch kernel launch failed");
    throw_if_cuda_error(
        cudaEventRecord(context.kernel_stop_event, context.stream),
        "Failed to record kernel stop event");

    throw_if_cuda_error(cudaEventRecord(context.d2h_start_event, context.stream), "Failed to record D2H start event");
    throw_if_cuda_error(
        cudaMemcpyAsync(
            context.host_scores,
            context.device_scores,
            score_bytes,
            cudaMemcpyDeviceToHost,
            context.stream),
        "Async D2H copy failed for scores");
    throw_if_cuda_error(cudaEventRecord(context.d2h_stop_event, context.stream), "Failed to record D2H stop event");

    context.has_pending_batch = true;
    context.pending_pair_offset = pair_offset;
    context.pending_pair_count = pair_count;
}

void harvest_pending_batch(StreamContext& context,
                           std::vector<int>& gpu_scores,
                           double& total_h2d_copy_time_ms,
                           double& total_kernel_time_ms,
                           double& total_d2h_copy_time_ms) {
    if (!context.has_pending_batch) {
        return;
    }

    throw_if_cuda_error(cudaStreamSynchronize(context.stream), "Failed to synchronize CUDA stream");

    float h2d_copy_time_ms = 0.0f;
    float kernel_time_ms = 0.0f;
    float d2h_copy_time_ms = 0.0f;
    throw_if_cuda_error(
        cudaEventElapsedTime(&h2d_copy_time_ms, context.h2d_start_event, context.h2d_stop_event),
        "Failed to measure H2D copy time");
    throw_if_cuda_error(
        cudaEventElapsedTime(&kernel_time_ms, context.kernel_start_event, context.kernel_stop_event),
        "Failed to measure kernel time");
    throw_if_cuda_error(
        cudaEventElapsedTime(&d2h_copy_time_ms, context.d2h_start_event, context.d2h_stop_event),
        "Failed to measure D2H copy time");

    total_h2d_copy_time_ms += static_cast<double>(h2d_copy_time_ms);
    total_kernel_time_ms += static_cast<double>(kernel_time_ms);
    total_d2h_copy_time_ms += static_cast<double>(d2h_copy_time_ms);

    std::memcpy(
        gpu_scores.data() + context.pending_pair_offset,
        context.host_scores,
        static_cast<std::size_t>(context.pending_pair_count) * sizeof(int));
    context.has_pending_batch = false;
}

void run_gpu_pipeline_once(const ProgramOptions& options,
                           const Dataset& dataset,
                           std::vector<StreamContext>& stream_contexts,
                           std::vector<int>& gpu_scores,
                           double& total_h2d_copy_time_ms,
                           double& total_kernel_time_ms,
                           double& total_d2h_copy_time_ms) {
    const int number_of_batches =
        (dataset.number_of_pairs + options.batch_size - 1) / options.batch_size;

    for (int batch_index = 0; batch_index < number_of_batches; ++batch_index) {
        StreamContext& context = stream_contexts[batch_index % stream_contexts.size()];
        harvest_pending_batch(
            context,
            gpu_scores,
            total_h2d_copy_time_ms,
            total_kernel_time_ms,
            total_d2h_copy_time_ms);

        const int pair_offset = batch_index * options.batch_size;
        const int pair_count = std::min(options.batch_size, dataset.number_of_pairs - pair_offset);
        copy_batch_to_pinned_buffers(dataset, pair_offset, pair_count, context);
        launch_batch(options, dataset, pair_offset, pair_count, context);
    }

    for (StreamContext& context : stream_contexts) {
        harvest_pending_batch(
            context,
            gpu_scores,
            total_h2d_copy_time_ms,
            total_kernel_time_ms,
            total_d2h_copy_time_ms);
    }
}

void free_device_buffers(std::vector<StreamContext>& stream_contexts, bool use_cuda_malloc_async) {
    for (StreamContext& context : stream_contexts) {
#if CUDART_VERSION >= 11020
        if (use_cuda_malloc_async) {
            if (context.device_sequence_a != nullptr) {
                throw_if_cuda_error(cudaFreeAsync(context.device_sequence_a, context.stream), "cudaFreeAsync failed for deviceSequenceA");
                context.device_sequence_a = nullptr;
            }
            if (context.device_sequence_b != nullptr) {
                throw_if_cuda_error(cudaFreeAsync(context.device_sequence_b, context.stream), "cudaFreeAsync failed for deviceSequenceB");
                context.device_sequence_b = nullptr;
            }
            if (context.device_scores != nullptr) {
                throw_if_cuda_error(cudaFreeAsync(context.device_scores, context.stream), "cudaFreeAsync failed for deviceScores");
                context.device_scores = nullptr;
            }
            continue;
        }
#endif
        if (context.device_sequence_a != nullptr) {
            throw_if_cuda_error(cudaFree(context.device_sequence_a), "cudaFree failed for deviceSequenceA");
            context.device_sequence_a = nullptr;
        }
        if (context.device_sequence_b != nullptr) {
            throw_if_cuda_error(cudaFree(context.device_sequence_b), "cudaFree failed for deviceSequenceB");
            context.device_sequence_b = nullptr;
        }
        if (context.device_scores != nullptr) {
            throw_if_cuda_error(cudaFree(context.device_scores), "cudaFree failed for deviceScores");
            context.device_scores = nullptr;
        }
    }

    if (use_cuda_malloc_async) {
        for (StreamContext& context : stream_contexts) {
            throw_if_cuda_error(cudaStreamSynchronize(context.stream), "Failed to synchronize async frees");
        }
    }
}

void free_pinned_host_buffers(std::vector<StreamContext>& stream_contexts) {
    for (StreamContext& context : stream_contexts) {
        if (context.host_sequence_a != nullptr) {
            throw_if_cuda_error(cudaFreeHost(context.host_sequence_a), "cudaFreeHost failed for hostSequenceA");
            context.host_sequence_a = nullptr;
        }
        if (context.host_sequence_b != nullptr) {
            throw_if_cuda_error(cudaFreeHost(context.host_sequence_b), "cudaFreeHost failed for hostSequenceB");
            context.host_sequence_b = nullptr;
        }
        if (context.host_scores != nullptr) {
            throw_if_cuda_error(cudaFreeHost(context.host_scores), "cudaFreeHost failed for hostScores");
            context.host_scores = nullptr;
        }
    }
}

void destroy_stream_resources(std::vector<StreamContext>& stream_contexts) {
    for (StreamContext& context : stream_contexts) {
        if (context.h2d_start_event != nullptr) {
            cudaEventDestroy(context.h2d_start_event);
            context.h2d_start_event = nullptr;
        }
        if (context.h2d_stop_event != nullptr) {
            cudaEventDestroy(context.h2d_stop_event);
            context.h2d_stop_event = nullptr;
        }
        if (context.kernel_start_event != nullptr) {
            cudaEventDestroy(context.kernel_start_event);
            context.kernel_start_event = nullptr;
        }
        if (context.kernel_stop_event != nullptr) {
            cudaEventDestroy(context.kernel_stop_event);
            context.kernel_stop_event = nullptr;
        }
        if (context.d2h_start_event != nullptr) {
            cudaEventDestroy(context.d2h_start_event);
            context.d2h_start_event = nullptr;
        }
        if (context.d2h_stop_event != nullptr) {
            cudaEventDestroy(context.d2h_stop_event);
            context.d2h_stop_event = nullptr;
        }
        if (context.stream != nullptr) {
            cudaStreamDestroy(context.stream);
            context.stream = nullptr;
        }
    }
}

void best_effort_cleanup(std::vector<StreamContext>& stream_contexts, bool use_cuda_malloc_async) {
    for (StreamContext& context : stream_contexts) {
        if (context.stream != nullptr) {
            cudaStreamSynchronize(context.stream);
        }
    }

    for (StreamContext& context : stream_contexts) {
#if CUDART_VERSION >= 11020
        if (use_cuda_malloc_async && context.stream != nullptr) {
            if (context.device_sequence_a != nullptr) {
                cudaFreeAsync(context.device_sequence_a, context.stream);
                context.device_sequence_a = nullptr;
            }
            if (context.device_sequence_b != nullptr) {
                cudaFreeAsync(context.device_sequence_b, context.stream);
                context.device_sequence_b = nullptr;
            }
            if (context.device_scores != nullptr) {
                cudaFreeAsync(context.device_scores, context.stream);
                context.device_scores = nullptr;
            }
            cudaStreamSynchronize(context.stream);
        } else
#endif
        {
            if (context.device_sequence_a != nullptr) {
                cudaFree(context.device_sequence_a);
                context.device_sequence_a = nullptr;
            }
            if (context.device_sequence_b != nullptr) {
                cudaFree(context.device_sequence_b);
                context.device_sequence_b = nullptr;
            }
            if (context.device_scores != nullptr) {
                cudaFree(context.device_scores);
                context.device_scores = nullptr;
            }
        }

        if (context.host_sequence_a != nullptr) {
            cudaFreeHost(context.host_sequence_a);
            context.host_sequence_a = nullptr;
        }
        if (context.host_sequence_b != nullptr) {
            cudaFreeHost(context.host_sequence_b);
            context.host_sequence_b = nullptr;
        }
        if (context.host_scores != nullptr) {
            cudaFreeHost(context.host_scores);
            context.host_scores = nullptr;
        }
    }

    destroy_stream_resources(stream_contexts);
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
    std::vector<StreamContext> stream_contexts;
    bool use_cuda_malloc_async = false;

    try {
        const ProgramOptions options = parse_arguments(argc, argv);

        const auto file_read_start_time = std::chrono::high_resolution_clock::now();
        std::vector<SequencePair> sequence_pairs = read_sequence_pairs(options.input_path);
        const auto file_read_end_time = std::chrono::high_resolution_clock::now();

        const auto input_validation_start_time = std::chrono::high_resolution_clock::now();
        const Dataset dataset = validate_and_flatten_dataset(std::move(sequence_pairs));
        const auto input_validation_end_time = std::chrono::high_resolution_clock::now();

        const int maximum_batch_pairs = std::min(options.batch_size, dataset.number_of_pairs);
        const int number_of_batches =
            (dataset.number_of_pairs + options.batch_size - 1) / options.batch_size;
        const std::size_t maximum_first_sequence_bytes =
            static_cast<std::size_t>(maximum_batch_pairs) *
            static_cast<std::size_t>(dataset.sequence_length_a) *
            sizeof(char);
        const std::size_t maximum_second_sequence_bytes =
            static_cast<std::size_t>(maximum_batch_pairs) *
            static_cast<std::size_t>(dataset.sequence_length_b) *
            sizeof(char);
        const std::size_t maximum_score_bytes =
            static_cast<std::size_t>(maximum_batch_pairs) * sizeof(int);

        const bool async_supported = cuda_malloc_async_supported();
        use_cuda_malloc_async = async_supported && options.prefer_cuda_malloc_async;
        stream_contexts.resize(static_cast<std::size_t>(options.num_streams));
        create_stream_resources(stream_contexts);

        const auto pinned_allocation_start_time = std::chrono::high_resolution_clock::now();
        allocate_pinned_host_buffers(
            stream_contexts,
            maximum_first_sequence_bytes,
            maximum_second_sequence_bytes,
            maximum_score_bytes);
        const auto pinned_allocation_end_time = std::chrono::high_resolution_clock::now();

        const auto device_allocation_start_time = std::chrono::high_resolution_clock::now();
        allocate_device_buffers(
            stream_contexts,
            maximum_first_sequence_bytes,
            maximum_second_sequence_bytes,
            maximum_score_bytes,
            use_cuda_malloc_async);
        const auto device_allocation_end_time = std::chrono::high_resolution_clock::now();

        std::vector<int> gpu_scores(static_cast<std::size_t>(dataset.number_of_pairs), 0);
        double total_h2d_copy_time_ms = 0.0;
        double total_kernel_time_ms = 0.0;
        double total_d2h_copy_time_ms = 0.0;
        for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            run_gpu_pipeline_once(
                options,
                dataset,
                stream_contexts,
                gpu_scores,
                total_h2d_copy_time_ms,
                total_kernel_time_ms,
                total_d2h_copy_time_ms);
        }

        const double average_h2d_copy_time_ms =
            total_h2d_copy_time_ms / static_cast<double>(options.repetitions);
        const double average_kernel_time_ms =
            total_kernel_time_ms / static_cast<double>(options.repetitions);
        const double average_d2h_copy_time_ms =
            total_d2h_copy_time_ms / static_cast<double>(options.repetitions);
        const double average_gpu_pipeline_time_ms =
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

        const auto device_free_start_time = std::chrono::high_resolution_clock::now();
        free_device_buffers(stream_contexts, use_cuda_malloc_async);
        const auto device_free_end_time = std::chrono::high_resolution_clock::now();

        const auto pinned_free_start_time = std::chrono::high_resolution_clock::now();
        free_pinned_host_buffers(stream_contexts);
        const auto pinned_free_end_time = std::chrono::high_resolution_clock::now();
        destroy_stream_resources(stream_contexts);

        const long long total_cells_computed =
            static_cast<long long>(dataset.number_of_pairs) *
            static_cast<long long>(dataset.sequence_length_a + 1) *
            static_cast<long long>(dataset.sequence_length_b + 1);
        const auto end_to_end_end_time = std::chrono::high_resolution_clock::now();

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "ALGORITHM=needleman_wunsch_gpu_optimized\n";
        std::cout << "IMPLEMENTATION=wavefront_streamed\n";
        std::cout << "NUMBER_OF_PAIRS=" << dataset.number_of_pairs << "\n";
        std::cout << "SEQUENCE_LENGTH_A=" << dataset.sequence_length_a << "\n";
        std::cout << "SEQUENCE_LENGTH_B=" << dataset.sequence_length_b << "\n";
        std::cout << "MATCH_SCORE=" << options.match_score << "\n";
        std::cout << "MISMATCH_PENALTY=" << options.mismatch_penalty << "\n";
        std::cout << "GAP_PENALTY=" << options.gap_penalty << "\n";
        std::cout << "MAX_SUPPORTED_SEQUENCE_LENGTH=" << MAX_SUPPORTED_SEQUENCE_LENGTH << "\n";
        std::cout << "USE_PINNED_MEMORY=true\n";
        std::cout << "CUDA_MALLOC_ASYNC_SUPPORTED=" << (async_supported ? "true" : "false") << "\n";
        std::cout << "USE_CUDA_MALLOC_ASYNC=" << (use_cuda_malloc_async ? "true" : "false") << "\n";
        if (!async_supported) {
            std::cout << "FALLBACK_DEVICE_ALLOCATION=true\n";
        }
        std::cout << "USE_CUDA_STREAMS=true\n";
        std::cout << "BATCH_SIZE=" << options.batch_size << "\n";
        std::cout << "NUMBER_OF_BATCHES=" << number_of_batches << "\n";
        std::cout << "NUM_STREAMS=" << options.num_streams << "\n";
        std::cout << "STREAMS_CREATED=" << stream_contexts.size() << "\n";
        std::cout << "SUMMARY_ONLY=" << (options.summary_only ? "true" : "false") << "\n";
        std::cout << "FILE_READ_TIME_MS=" << elapsed_milliseconds(file_read_start_time, file_read_end_time) << "\n";
        std::cout << "INPUT_VALIDATION_TIME_MS="
                  << elapsed_milliseconds(input_validation_start_time, input_validation_end_time) << "\n";
        std::cout << "PINNED_HOST_ALLOCATION_TIME_MS="
                  << elapsed_milliseconds(pinned_allocation_start_time, pinned_allocation_end_time) << "\n";
        std::cout << "DEVICE_ALLOCATION_TIME_MS="
                  << elapsed_milliseconds(device_allocation_start_time, device_allocation_end_time) << "\n";
        std::cout << "H2D_COPY_TIME_MS=" << average_h2d_copy_time_ms << "\n";
        std::cout << "GPU_KERNEL_TIME_MS=" << average_kernel_time_ms << "\n";
        std::cout << "D2H_COPY_TIME_MS=" << average_d2h_copy_time_ms << "\n";
        std::cout << "GPU_PIPELINE_TIME_MS=" << average_gpu_pipeline_time_ms << "\n";
        std::cout << "CPU_REFERENCE_TIME_MS=" << cpu_reference_time_ms << "\n";
        std::cout << "VALIDATION_TIME_MS=" << validation_time_ms << "\n";
        std::cout << "CSV_WRITE_TIME_MS=" << elapsed_milliseconds(csv_write_start_time, csv_write_end_time) << "\n";
        std::cout << "DEVICE_FREE_TIME_MS="
                  << elapsed_milliseconds(device_free_start_time, device_free_end_time) << "\n";
        std::cout << "PINNED_HOST_FREE_TIME_MS="
                  << elapsed_milliseconds(pinned_free_start_time, pinned_free_end_time) << "\n";
        std::cout << "END_TO_END_TIME_MS="
                  << elapsed_milliseconds(end_to_end_start_time, end_to_end_end_time) << "\n";
        std::cout << "TOTAL_CELLS_COMPUTED=" << total_cells_computed << "\n";
        std::cout << "GPU_TOTAL_CELLS_PER_SECOND="
                  << safe_cells_per_second(total_cells_computed, average_gpu_pipeline_time_ms) << "\n";
        std::cout << "GPU_KERNEL_CELLS_PER_SECOND="
                  << safe_cells_per_second(total_cells_computed, average_kernel_time_ms) << "\n";
        std::cout << "VALIDATION_STATUS=" << validation_result.status << "\n";
        if (validation_result.status == "FAILED") {
            std::cout << "FIRST_MISMATCH_PAIR_ID=" << validation_result.first_mismatch_pair_id << "\n";
            std::cout << "CPU_SCORE=" << validation_result.cpu_score << "\n";
            std::cout << "GPU_SCORE=" << validation_result.gpu_score << "\n";
        }
        std::cout << "OUTPUT_PATH=" << options.output_path << "\n";

        return validation_result.status == "FAILED" ? 2 : 0;
    } catch (const std::exception& error) {
        best_effort_cleanup(stream_contexts, use_cuda_malloc_async);
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }
}
