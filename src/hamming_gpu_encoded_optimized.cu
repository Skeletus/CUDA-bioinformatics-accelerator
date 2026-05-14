#include <cuda_runtime.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_utils.h"
#include "dna_utils.h"
#include "timer.h"

__global__ void hammingEncodedOptimizedKernel(const std::uint8_t* sequence_a,
                                              const std::uint8_t* sequence_b,
                                              int* distances,
                                              int number_of_pairs,
                                              int sequence_length) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_index >= number_of_pairs) {
        return;
    }

    const int offset = pair_index * sequence_length;
    int distance = 0;
    for (int base_index = 0; base_index < sequence_length; ++base_index) {
        if (sequence_a[offset + base_index] != sequence_b[offset + base_index]) {
            ++distance;
        }
    }

    distances[pair_index] = distance;
}

struct ProgramOptions {
    std::string input_path;
    std::string output_path;
    int repetitions = 5;
    bool summary_only = false;
    bool skip_validation = false;
};

struct CachedDataset {
    std::vector<std::string> unique_sequences;
    std::vector<int> pair_sequence_a_indices;
    std::vector<int> pair_sequence_b_indices;
    int number_of_pairs = 0;
    int sequence_length = 0;
    std::size_t cache_hit_count = 0;
    std::size_t cache_miss_count = 0;
};

struct PinnedHostBuffers {
    std::uint8_t* encoded_sequence_a = nullptr;
    std::uint8_t* encoded_sequence_b = nullptr;
    int* distances = nullptr;

    void allocate(std::size_t sequence_bytes, std::size_t result_bytes) {
        cudaError_t status = cudaMallocHost(reinterpret_cast<void**>(&encoded_sequence_a), sequence_bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned host memory for sequence A: " +
                                     std::string(cudaGetErrorString(status)));
        }
        status = cudaMallocHost(reinterpret_cast<void**>(&encoded_sequence_b), sequence_bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned host memory for sequence B: " +
                                     std::string(cudaGetErrorString(status)));
        }
        status = cudaMallocHost(reinterpret_cast<void**>(&distances), result_bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned host memory for result distances: " +
                                     std::string(cudaGetErrorString(status)));
        }
    }

    void release() {
        if (encoded_sequence_a != nullptr) {
            cudaFreeHost(encoded_sequence_a);
            encoded_sequence_a = nullptr;
        }
        if (encoded_sequence_b != nullptr) {
            cudaFreeHost(encoded_sequence_b);
            encoded_sequence_b = nullptr;
        }
        if (distances != nullptr) {
            cudaFreeHost(distances);
            distances = nullptr;
        }
    }
};

class GpuMemoryPool {
public:
    std::uint8_t* device_sequence_a = nullptr;
    std::uint8_t* device_sequence_b = nullptr;
    int* device_distances = nullptr;

    void allocate(std::size_t sequence_bytes, std::size_t result_bytes) {
        CUDA_CHECK(cudaMalloc(&device_sequence_a, sequence_bytes));
        CUDA_CHECK(cudaMalloc(&device_sequence_b, sequence_bytes));
        CUDA_CHECK(cudaMalloc(&device_distances, result_bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void release() {
        if (device_sequence_a != nullptr) {
            cudaFree(device_sequence_a);
            device_sequence_a = nullptr;
        }
        if (device_sequence_b != nullptr) {
            cudaFree(device_sequence_b);
            device_sequence_b = nullptr;
        }
        if (device_distances != nullptr) {
            cudaFree(device_distances);
            device_distances = nullptr;
        }
    }
};

struct ValidationResult {
    bool passed = true;
    std::size_t first_mismatch_pair_id = 0;
    int cpu_distance = 0;
    int gpu_distance = 0;
};

struct TimingBreakdown {
    double file_read_time_ms = 0.0;
    double input_validation_time_ms = 0.0;
    double encoded_cache_time_ms = 0.0;
    double encoding_time_ms = 0.0;
    double flat_buffer_build_time_ms = 0.0;
    double pinned_host_allocation_time_ms = 0.0;
    double device_allocation_time_ms = 0.0;
    double h2d_copy_time_ms = 0.0;
    double gpu_kernel_time_ms = 0.0;
    double d2h_copy_time_ms = 0.0;
    double cpu_reference_time_ms = 0.0;
    double validation_time_ms = 0.0;
    double csv_write_time_ms = 0.0;
    double device_free_time_ms = 0.0;
    double setup_time_ms = 0.0;
    double gpu_pipeline_time_ms = 0.0;
    double end_to_end_time_ms = 0.0;
};

ProgramOptions parse_arguments(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: " + std::string(argv[0]) +
            " <input_dataset> <output_csv> [--repetitions N] [--summary-only] [--write-results] [--skip-validation]");
    }

    ProgramOptions options;
    options.input_path = argv[1];
    options.output_path = argv[2];

    for (int index = 3; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--repetitions") {
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
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    return options;
}

int get_or_create_sequence_index(CachedDataset& dataset,
                                 std::unordered_map<std::string, int>& sequence_to_cache_index,
                                 const std::string& sequence) {
    const auto existing_sequence = sequence_to_cache_index.find(sequence);
    if (existing_sequence != sequence_to_cache_index.end()) {
        ++dataset.cache_hit_count;
        return existing_sequence->second;
    }

    if (dataset.unique_sequences.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Unique sequence count exceeds supported integer range.");
    }
    const int cache_index = static_cast<int>(dataset.unique_sequences.size());
    dataset.unique_sequences.push_back(sequence);
    sequence_to_cache_index.emplace(dataset.unique_sequences.back(), cache_index);
    ++dataset.cache_miss_count;
    return cache_index;
}

CachedDataset read_dataset_with_cache(const std::string& input_path) {
    std::ifstream input_file(input_path);
    if (!input_file) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    CachedDataset dataset;
    std::unordered_map<std::string, int> sequence_to_cache_index;
    std::string first_sequence;
    std::string second_sequence;
    bool has_sequence_length = false;

    while (input_file >> first_sequence >> second_sequence) {
        if (first_sequence.size() != second_sequence.size()) {
            throw std::runtime_error("Found a sequence pair with unequal lengths.");
        }
        if (!has_sequence_length) {
            if (first_sequence.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::runtime_error("Sequence length exceeds supported integer range.");
            }
            dataset.sequence_length = static_cast<int>(first_sequence.size());
            has_sequence_length = true;
        } else if (first_sequence.size() != static_cast<std::size_t>(dataset.sequence_length)) {
            throw std::runtime_error("Found sequence pairs with different fixed lengths.");
        }

        const int first_index = get_or_create_sequence_index(dataset, sequence_to_cache_index, first_sequence);
        const int second_index = get_or_create_sequence_index(dataset, sequence_to_cache_index, second_sequence);
        dataset.pair_sequence_a_indices.push_back(first_index);
        dataset.pair_sequence_b_indices.push_back(second_index);
    }

    if (dataset.pair_sequence_a_indices.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }
    if (dataset.pair_sequence_a_indices.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Number of pairs exceeds supported integer range.");
    }

    dataset.number_of_pairs = static_cast<int>(dataset.pair_sequence_a_indices.size());
    return dataset;
}

void validate_cached_dataset(const CachedDataset& dataset) {
    if (dataset.number_of_pairs <= 0 || dataset.sequence_length <= 0) {
        throw std::runtime_error("Input dataset is empty.");
    }
    if (dataset.pair_sequence_a_indices.size() != dataset.pair_sequence_b_indices.size()) {
        throw std::runtime_error("Pair index vectors have inconsistent sizes.");
    }

    for (const std::string& sequence : dataset.unique_sequences) {
        if (sequence.size() != static_cast<std::size_t>(dataset.sequence_length)) {
            throw std::runtime_error("Cached sequences have inconsistent lengths.");
        }
        if (!dna::validateDnaSequence(sequence)) {
            throw std::runtime_error("Found an invalid DNA sequence. Allowed bases are A, C, G, and T.");
        }
    }
}

std::vector<std::vector<std::uint8_t>> build_encoded_cache(const CachedDataset& dataset) {
    std::vector<std::vector<std::uint8_t>> encoded_cache;
    encoded_cache.reserve(dataset.unique_sequences.size());
    for (const std::string& sequence : dataset.unique_sequences) {
        encoded_cache.push_back(dna::encodeDnaSequence(sequence));
    }
    return encoded_cache;
}

void build_flat_pinned_buffers(const CachedDataset& dataset,
                               const std::vector<std::vector<std::uint8_t>>& encoded_cache,
                               PinnedHostBuffers& host_buffers) {
    for (int pair_index = 0; pair_index < dataset.number_of_pairs; ++pair_index) {
        const int first_index = dataset.pair_sequence_a_indices[static_cast<std::size_t>(pair_index)];
        const int second_index = dataset.pair_sequence_b_indices[static_cast<std::size_t>(pair_index)];
        const std::vector<std::uint8_t>& first_sequence = encoded_cache[static_cast<std::size_t>(first_index)];
        const std::vector<std::uint8_t>& second_sequence = encoded_cache[static_cast<std::size_t>(second_index)];
        const std::size_t output_offset = static_cast<std::size_t>(pair_index) *
                                          static_cast<std::size_t>(dataset.sequence_length);

        for (int base_index = 0; base_index < dataset.sequence_length; ++base_index) {
            host_buffers.encoded_sequence_a[output_offset + static_cast<std::size_t>(base_index)] =
                first_sequence[static_cast<std::size_t>(base_index)];
            host_buffers.encoded_sequence_b[output_offset + static_cast<std::size_t>(base_index)] =
                second_sequence[static_cast<std::size_t>(base_index)];
        }
    }
}

std::vector<int> compute_cpu_distances(const CachedDataset& dataset,
                                       const std::vector<std::vector<std::uint8_t>>& encoded_cache) {
    std::vector<int> cpu_distances(static_cast<std::size_t>(dataset.number_of_pairs));

    for (int pair_index = 0; pair_index < dataset.number_of_pairs; ++pair_index) {
        const int first_index = dataset.pair_sequence_a_indices[static_cast<std::size_t>(pair_index)];
        const int second_index = dataset.pair_sequence_b_indices[static_cast<std::size_t>(pair_index)];
        const std::vector<std::uint8_t>& first_sequence = encoded_cache[static_cast<std::size_t>(first_index)];
        const std::vector<std::uint8_t>& second_sequence = encoded_cache[static_cast<std::size_t>(second_index)];

        int distance = 0;
        for (int base_index = 0; base_index < dataset.sequence_length; ++base_index) {
            if (first_sequence[static_cast<std::size_t>(base_index)] !=
                second_sequence[static_cast<std::size_t>(base_index)]) {
                ++distance;
            }
        }
        cpu_distances[static_cast<std::size_t>(pair_index)] = distance;
    }

    return cpu_distances;
}

ValidationResult validate_results(const int* gpu_distances, const std::vector<int>& cpu_distances) {
    ValidationResult result;
    for (std::size_t pair_index = 0; pair_index < cpu_distances.size(); ++pair_index) {
        if (gpu_distances[pair_index] != cpu_distances[pair_index]) {
            result.passed = false;
            result.first_mismatch_pair_id = pair_index;
            result.cpu_distance = cpu_distances[pair_index];
            result.gpu_distance = gpu_distances[pair_index];
            return result;
        }
    }
    return result;
}

void write_results_csv(const std::string& output_path,
                       const int* distances,
                       int number_of_pairs,
                       int sequence_length) {
    const std::filesystem::path output_file_path(output_path);
    if (output_file_path.has_parent_path()) {
        std::filesystem::create_directories(output_file_path.parent_path());
    }

    std::ofstream output_file(output_path);
    if (!output_file) {
        throw std::runtime_error("Failed to open output file: " + output_path);
    }

    output_file << "pair_id,distance,similarity\n";
    output_file << std::fixed << std::setprecision(6);
    for (int pair_index = 0; pair_index < number_of_pairs; ++pair_index) {
        const double similarity = 100.0 * (1.0 - static_cast<double>(distances[pair_index]) /
                                                     static_cast<double>(sequence_length));
        output_file << pair_index << "," << distances[pair_index] << "," << similarity << "\n";
    }
}

int main(int argc, char** argv) {
    PinnedHostBuffers host_buffers;
    GpuMemoryPool gpu_memory_pool;
    cudaEvent_t kernel_start_event = nullptr;
    cudaEvent_t kernel_stop_event = nullptr;

    try {
        CpuTimer end_to_end_timer;
        TimingBreakdown timings;
        const ProgramOptions options = parse_arguments(argc, argv);

        CpuTimer file_read_timer;
        CachedDataset dataset = read_dataset_with_cache(options.input_path);
        timings.file_read_time_ms = file_read_timer.elapsed_milliseconds();

        CpuTimer input_validation_timer;
        validate_cached_dataset(dataset);
        timings.input_validation_time_ms = input_validation_timer.elapsed_milliseconds();

        CpuTimer encoded_cache_timer;
        const std::vector<std::vector<std::uint8_t>> encoded_cache = build_encoded_cache(dataset);
        timings.encoded_cache_time_ms = encoded_cache_timer.elapsed_milliseconds();
        timings.encoding_time_ms = timings.encoded_cache_time_ms;

        const std::size_t sequence_bytes = static_cast<std::size_t>(dataset.number_of_pairs) *
                                           static_cast<std::size_t>(dataset.sequence_length) *
                                           sizeof(std::uint8_t);
        const std::size_t result_bytes = static_cast<std::size_t>(dataset.number_of_pairs) * sizeof(int);

        CpuTimer pinned_allocation_timer;
        host_buffers.allocate(sequence_bytes, result_bytes);
        timings.pinned_host_allocation_time_ms = pinned_allocation_timer.elapsed_milliseconds();

        CpuTimer flat_buffer_timer;
        build_flat_pinned_buffers(dataset, encoded_cache, host_buffers);
        timings.flat_buffer_build_time_ms = flat_buffer_timer.elapsed_milliseconds();

        CpuTimer cpu_reference_timer;
        const std::vector<int> cpu_distances = compute_cpu_distances(dataset, encoded_cache);
        timings.cpu_reference_time_ms = cpu_reference_timer.elapsed_milliseconds();

        CpuTimer device_allocation_timer;
        gpu_memory_pool.allocate(sequence_bytes, result_bytes);
        timings.device_allocation_time_ms = device_allocation_timer.elapsed_milliseconds();

        timings.setup_time_ms = timings.pinned_host_allocation_time_ms +
                                timings.device_allocation_time_ms +
                                timings.encoded_cache_time_ms;

        CUDA_CHECK(cudaEventCreate(&kernel_start_event));
        CUDA_CHECK(cudaEventCreate(&kernel_stop_event));

        CUDA_CHECK(cudaMemcpy(gpu_memory_pool.device_sequence_a,
                              host_buffers.encoded_sequence_a,
                              sequence_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(gpu_memory_pool.device_sequence_b,
                              host_buffers.encoded_sequence_b,
                              sequence_bytes,
                              cudaMemcpyHostToDevice));

        const int threads_per_block = 256;
        const int blocks_per_grid = (dataset.number_of_pairs + threads_per_block - 1) / threads_per_block;

        hammingEncodedOptimizedKernel<<<blocks_per_grid, threads_per_block>>>(
            gpu_memory_pool.device_sequence_a,
            gpu_memory_pool.device_sequence_b,
            gpu_memory_pool.device_distances,
            dataset.number_of_pairs,
            dataset.sequence_length);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        double total_h2d_copy_time_ms = 0.0;
        double total_kernel_time_ms = 0.0;
        double total_d2h_copy_time_ms = 0.0;

        for (int repetition = 0; repetition < options.repetitions; ++repetition) {
            CpuTimer h2d_copy_timer;
            CUDA_CHECK(cudaMemcpy(gpu_memory_pool.device_sequence_a,
                                  host_buffers.encoded_sequence_a,
                                  sequence_bytes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(gpu_memory_pool.device_sequence_b,
                                  host_buffers.encoded_sequence_b,
                                  sequence_bytes,
                                  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());
            total_h2d_copy_time_ms += h2d_copy_timer.elapsed_milliseconds();

            CUDA_CHECK(cudaEventRecord(kernel_start_event));
            hammingEncodedOptimizedKernel<<<blocks_per_grid, threads_per_block>>>(
                gpu_memory_pool.device_sequence_a,
                gpu_memory_pool.device_sequence_b,
                gpu_memory_pool.device_distances,
                dataset.number_of_pairs,
                dataset.sequence_length);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(kernel_stop_event));
            CUDA_CHECK(cudaEventSynchronize(kernel_stop_event));

            float repetition_kernel_time_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&repetition_kernel_time_ms, kernel_start_event, kernel_stop_event));
            total_kernel_time_ms += static_cast<double>(repetition_kernel_time_ms);

            CpuTimer d2h_copy_timer;
            CUDA_CHECK(cudaMemcpy(host_buffers.distances,
                                  gpu_memory_pool.device_distances,
                                  result_bytes,
                                  cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaDeviceSynchronize());
            total_d2h_copy_time_ms += d2h_copy_timer.elapsed_milliseconds();
        }

        timings.h2d_copy_time_ms = total_h2d_copy_time_ms / static_cast<double>(options.repetitions);
        timings.gpu_kernel_time_ms = total_kernel_time_ms / static_cast<double>(options.repetitions);
        timings.d2h_copy_time_ms = total_d2h_copy_time_ms / static_cast<double>(options.repetitions);
        timings.gpu_pipeline_time_ms = timings.h2d_copy_time_ms +
                                       timings.gpu_kernel_time_ms +
                                       timings.d2h_copy_time_ms;

        ValidationResult validation;
        if (!options.skip_validation) {
            CpuTimer validation_timer;
            validation = validate_results(host_buffers.distances, cpu_distances);
            timings.validation_time_ms = validation_timer.elapsed_milliseconds();
        }

        if (!validation.passed) {
            std::cerr << "Error: optimized encoded GPU validation failed against CPU reference results.\n";
            std::cerr << "First mismatching pair ID: " << validation.first_mismatch_pair_id << "\n";
            std::cerr << "CPU distance: " << validation.cpu_distance << "\n";
            std::cerr << "GPU distance: " << validation.gpu_distance << "\n";
        }

        if (!options.summary_only) {
            CpuTimer csv_write_timer;
            write_results_csv(options.output_path, host_buffers.distances, dataset.number_of_pairs, dataset.sequence_length);
            timings.csv_write_time_ms = csv_write_timer.elapsed_milliseconds();
        }

        CpuTimer device_free_timer;
        if (kernel_start_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(kernel_start_event));
            kernel_start_event = nullptr;
        }
        if (kernel_stop_event != nullptr) {
            CUDA_CHECK(cudaEventDestroy(kernel_stop_event));
            kernel_stop_event = nullptr;
        }
        gpu_memory_pool.release();
        CUDA_CHECK(cudaDeviceSynchronize());
        timings.device_free_time_ms = device_free_timer.elapsed_milliseconds();

        host_buffers.release();
        timings.end_to_end_time_ms = end_to_end_timer.elapsed_milliseconds();

        const std::size_t total_bases_compared = static_cast<std::size_t>(dataset.number_of_pairs) *
                                                 static_cast<std::size_t>(dataset.sequence_length);
        const std::size_t flat_pair_bytes = sequence_bytes * 2;
        const double cache_hit_rate = (dataset.cache_hit_count + dataset.cache_miss_count) == 0
                                          ? 0.0
                                          : static_cast<double>(dataset.cache_hit_count) /
                                                static_cast<double>(dataset.cache_hit_count + dataset.cache_miss_count);

        std::cout << "Number of pairs: " << dataset.number_of_pairs << "\n";
        std::cout << "Sequence length: " << dataset.sequence_length << "\n";
        std::cout << "Unique sequence count: " << dataset.unique_sequences.size() << "\n";
        std::cout << "Summary only: " << (options.summary_only ? "true" : "false") << "\n";
        std::cout << "Validation status: "
                  << (options.skip_validation ? "SKIPPED" : (validation.passed ? "PASSED" : "FAILED")) << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "FILE_READ_TIME_MS=" << timings.file_read_time_ms << "\n";
        std::cout << "INPUT_VALIDATION_TIME_MS=" << timings.input_validation_time_ms << "\n";
        std::cout << "ENCODING_TIME_MS=" << timings.encoding_time_ms << "\n";
        std::cout << "ENCODED_CACHE_TIME_MS=" << timings.encoded_cache_time_ms << "\n";
        std::cout << "FLAT_BUFFER_BUILD_TIME_MS=" << timings.flat_buffer_build_time_ms << "\n";
        std::cout << "PINNED_HOST_ALLOCATION_TIME_MS=" << timings.pinned_host_allocation_time_ms << "\n";
        std::cout << "DEVICE_ALLOCATION_TIME_MS=" << timings.device_allocation_time_ms << "\n";
        std::cout << "H2D_COPY_TIME_MS=" << timings.h2d_copy_time_ms << "\n";
        std::cout << "GPU_KERNEL_TIME_MS=" << timings.gpu_kernel_time_ms << "\n";
        std::cout << "D2H_COPY_TIME_MS=" << timings.d2h_copy_time_ms << "\n";
        std::cout << "CPU_REFERENCE_TIME_MS=" << timings.cpu_reference_time_ms << "\n";
        std::cout << "VALIDATION_TIME_MS=" << timings.validation_time_ms << "\n";
        std::cout << "CSV_WRITE_TIME_MS=" << timings.csv_write_time_ms << "\n";
        std::cout << "DEVICE_FREE_TIME_MS=" << timings.device_free_time_ms << "\n";
        std::cout << "SETUP_TIME_MS=" << timings.setup_time_ms << "\n";
        std::cout << "GPU_PIPELINE_TIME_MS=" << timings.gpu_pipeline_time_ms << "\n";
        std::cout << "GPU_TOTAL_TIME_MS=" << timings.gpu_pipeline_time_ms << "\n";
        std::cout << "END_TO_END_TIME_MS=" << timings.end_to_end_time_ms << "\n";
        std::cout << "NUMBER_OF_PAIRS=" << dataset.number_of_pairs << "\n";
        std::cout << "SEQUENCE_LENGTH=" << dataset.sequence_length << "\n";
        std::cout << "TOTAL_BASES_COMPARED=" << total_bases_compared << "\n";
        std::cout << "UNIQUE_SEQUENCE_COUNT=" << dataset.unique_sequences.size() << "\n";
        std::cout << "CACHE_HIT_COUNT=" << dataset.cache_hit_count << "\n";
        std::cout << "CACHE_MISS_COUNT=" << dataset.cache_miss_count << "\n";
        std::cout << "CACHE_HIT_RATE=" << cache_hit_rate << "\n";
        std::cout << "PINNED_HOST_MEMORY=true\n";
        std::cout << "MEMORY_POOL_REUSED=true\n";
        std::cout << "SUMMARY_ONLY=" << (options.summary_only ? "true" : "false") << "\n";
        std::cout << "WRITE_RESULTS=" << (!options.summary_only ? "true" : "false") << "\n";
        std::cout << "INDEXED_MODE=false\n";
        std::cout << "UNIQUE_FRAGMENT_BYTES=0\n";
        std::cout << "PAIR_INDEX_BYTES=0\n";
        std::cout << "FLAT_PAIR_BYTES_AVOIDED=0\n";
        std::cout << "FLAT_PAIR_BYTES=" << flat_pair_bytes << "\n";
        std::cout << "REPETITIONS=" << options.repetitions << "\n";
        std::cout << "VALIDATION_STATUS="
                  << (options.skip_validation ? "SKIPPED" : (validation.passed ? "PASSED" : "FAILED")) << "\n";
        if (!validation.passed) {
            std::cout << "FIRST_MISMATCH_PAIR_ID=" << validation.first_mismatch_pair_id << "\n";
            std::cout << "CPU_DISTANCE=" << validation.cpu_distance << "\n";
            std::cout << "GPU_DISTANCE=" << validation.gpu_distance << "\n";
        }
        std::cout << "OUTPUT_PATH=" << options.output_path << "\n";

        if (!options.skip_validation && !validation.passed) {
            return 2;
        }
    } catch (const std::exception& error) {
        if (kernel_start_event != nullptr) {
            cudaEventDestroy(kernel_start_event);
        }
        if (kernel_stop_event != nullptr) {
            cudaEventDestroy(kernel_stop_event);
        }
        gpu_memory_pool.release();
        host_buffers.release();

        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
