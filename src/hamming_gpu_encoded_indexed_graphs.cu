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

__global__ void hammingIndexedEncodedKernel(const std::uint8_t* unique_encoded_fragments,
                                            const int* pair_index_a,
                                            const int* pair_index_b,
                                            int* distances,
                                            int number_of_pairs,
                                            int sequence_length) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_index >= number_of_pairs) {
        return;
    }

    const int fragment_a = pair_index_a[pair_index];
    const int fragment_b = pair_index_b[pair_index];
    const int offset_a = fragment_a * sequence_length;
    const int offset_b = fragment_b * sequence_length;

    int distance = 0;
    for (int base_index = 0; base_index < sequence_length; ++base_index) {
        if (unique_encoded_fragments[offset_a + base_index] !=
            unique_encoded_fragments[offset_b + base_index]) {
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
    bool request_cuda_graphs = true;
    bool request_cuda_malloc_async = true;
};

struct TimingBreakdown {
    double file_read_time_ms = 0.0;
    double input_validation_time_ms = 0.0;
    double unique_sequence_build_time_ms = 0.0;
    double encoding_time_ms = 0.0;
    double index_array_build_time_ms = 0.0;
    double pinned_host_allocation_time_ms = 0.0;
    double device_allocation_time_ms = 0.0;
    double async_device_allocation_time_ms = 0.0;
    double h2d_copy_time_ms = 0.0;
    double gpu_kernel_time_ms = 0.0;
    double d2h_copy_time_ms = 0.0;
    double gpu_pipeline_time_ms = 0.0;
    double graph_creation_time_ms = 0.0;
    double graph_instantiation_time_ms = 0.0;
    double graph_execution_time_ms = 0.0;
    double graph_average_execution_time_ms = 0.0;
    double cpu_reference_time_ms = 0.0;
    double validation_time_ms = 0.0;
    double csv_write_time_ms = 0.0;
    double device_free_time_ms = 0.0;
    double async_device_free_time_ms = 0.0;
    double setup_time_ms = 0.0;
    double end_to_end_time_ms = 0.0;
};

struct IndexedDataset {
    std::vector<std::string> unique_sequences;
    std::vector<int> pair_index_a;
    std::vector<int> pair_index_b;
    int number_of_pairs = 0;
    int sequence_length = 0;
    std::size_t cache_hit_count = 0;
    std::size_t cache_miss_count = 0;
};

struct PinnedHostBuffers {
    std::uint8_t* unique_encoded_fragments = nullptr;
    int* pair_index_a = nullptr;
    int* pair_index_b = nullptr;
    int* distances = nullptr;

    void allocate(std::size_t unique_fragment_bytes, std::size_t pair_index_bytes, std::size_t result_bytes) {
        cudaError_t status = cudaMallocHost(reinterpret_cast<void**>(&unique_encoded_fragments), unique_fragment_bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned host memory for unique encoded fragments: " +
                                     std::string(cudaGetErrorString(status)));
        }
        status = cudaMallocHost(reinterpret_cast<void**>(&pair_index_a), pair_index_bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned host memory for pairIndexA: " +
                                     std::string(cudaGetErrorString(status)));
        }
        status = cudaMallocHost(reinterpret_cast<void**>(&pair_index_b), pair_index_bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned host memory for pairIndexB: " +
                                     std::string(cudaGetErrorString(status)));
        }
        status = cudaMallocHost(reinterpret_cast<void**>(&distances), result_bytes);
        if (status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned host memory for distances: " +
                                     std::string(cudaGetErrorString(status)));
        }
    }

    void release() {
        if (unique_encoded_fragments != nullptr) {
            cudaFreeHost(unique_encoded_fragments);
            unique_encoded_fragments = nullptr;
        }
        if (pair_index_a != nullptr) {
            cudaFreeHost(pair_index_a);
            pair_index_a = nullptr;
        }
        if (pair_index_b != nullptr) {
            cudaFreeHost(pair_index_b);
            pair_index_b = nullptr;
        }
        if (distances != nullptr) {
            cudaFreeHost(distances);
            distances = nullptr;
        }
    }
};

struct DeviceBuffers {
    std::uint8_t* unique_encoded_fragments = nullptr;
    int* pair_index_a = nullptr;
    int* pair_index_b = nullptr;
    int* distances = nullptr;

    void allocate(std::size_t unique_fragment_bytes,
                  std::size_t pair_index_bytes,
                  std::size_t result_bytes,
                  cudaStream_t stream,
                  bool use_cuda_malloc_async) {
#if CUDART_VERSION >= 11020
        if (use_cuda_malloc_async) {
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&unique_encoded_fragments), unique_fragment_bytes, stream));
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&pair_index_a), pair_index_bytes, stream));
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&pair_index_b), pair_index_bytes, stream));
            CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&distances), result_bytes, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            return;
        }
#else
        (void)stream;
        (void)use_cuda_malloc_async;
#endif
        CUDA_CHECK(cudaMalloc(&unique_encoded_fragments, unique_fragment_bytes));
        CUDA_CHECK(cudaMalloc(&pair_index_a, pair_index_bytes));
        CUDA_CHECK(cudaMalloc(&pair_index_b, pair_index_bytes));
        CUDA_CHECK(cudaMalloc(&distances, result_bytes));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    void release(cudaStream_t stream, bool use_cuda_malloc_async) {
#if CUDART_VERSION >= 11020
        if (use_cuda_malloc_async) {
            if (unique_encoded_fragments != nullptr) {
                cudaFreeAsync(unique_encoded_fragments, stream);
                unique_encoded_fragments = nullptr;
            }
            if (pair_index_a != nullptr) {
                cudaFreeAsync(pair_index_a, stream);
                pair_index_a = nullptr;
            }
            if (pair_index_b != nullptr) {
                cudaFreeAsync(pair_index_b, stream);
                pair_index_b = nullptr;
            }
            if (distances != nullptr) {
                cudaFreeAsync(distances, stream);
                distances = nullptr;
            }
            cudaStreamSynchronize(stream);
            return;
        }
#else
        (void)stream;
        (void)use_cuda_malloc_async;
#endif
        if (unique_encoded_fragments != nullptr) {
            cudaFree(unique_encoded_fragments);
            unique_encoded_fragments = nullptr;
        }
        if (pair_index_a != nullptr) {
            cudaFree(pair_index_a);
            pair_index_a = nullptr;
        }
        if (pair_index_b != nullptr) {
            cudaFree(pair_index_b);
            pair_index_b = nullptr;
        }
        if (distances != nullptr) {
            cudaFree(distances);
            distances = nullptr;
        }
    }
};

struct ValidationResult {
    bool passed = true;
    std::size_t first_mismatch_pair_id = 0;
    int cpu_distance = 0;
    int gpu_distance = 0;
};

ProgramOptions parse_arguments(int argc, char** argv) {
    if (argc < 3) {
        throw std::runtime_error(
            "Usage: " + std::string(argv[0]) +
            " <input_dataset> <output_csv> [--repetitions N] [--summary-only] [--write-results] "
            "[--skip-validation] [--use-cuda-graphs] [--disable-cuda-graphs] "
            "[--use-cuda-malloc-async] [--disable-cuda-malloc-async]");
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
        } else if (argument == "--use-cuda-graphs") {
            options.request_cuda_graphs = true;
        } else if (argument == "--disable-cuda-graphs") {
            options.request_cuda_graphs = false;
        } else if (argument == "--use-cuda-malloc-async") {
            options.request_cuda_malloc_async = true;
        } else if (argument == "--disable-cuda-malloc-async") {
            options.request_cuda_malloc_async = false;
        } else {
            throw std::runtime_error("Unknown argument: " + argument);
        }
    }

    return options;
}

bool runtime_supports_cuda_graphs() {
#if CUDART_VERSION >= 10000
    int runtime_version = 0;
    cudaError_t status = cudaRuntimeGetVersion(&runtime_version);
    return status == cudaSuccess && runtime_version >= 10000;
#else
    return false;
#endif
}

bool runtime_supports_cuda_malloc_async() {
#if CUDART_VERSION >= 11020
    int runtime_version = 0;
    if (cudaRuntimeGetVersion(&runtime_version) != cudaSuccess || runtime_version < 11020) {
        return false;
    }
    int device_id = 0;
    if (cudaGetDevice(&device_id) != cudaSuccess) {
        return false;
    }
    int memory_pools_supported = 0;
    if (cudaDeviceGetAttribute(&memory_pools_supported, cudaDevAttrMemoryPoolsSupported, device_id) != cudaSuccess) {
        return false;
    }
    return memory_pools_supported != 0;
#else
    return false;
#endif
}

int get_or_create_sequence_index(IndexedDataset& dataset,
                                 std::unordered_map<std::string, int>& sequence_to_index,
                                 const std::string& sequence,
                                 TimingBreakdown& timings) {
    CpuTimer unique_timer;
    const auto existing_sequence = sequence_to_index.find(sequence);
    if (existing_sequence != sequence_to_index.end()) {
        ++dataset.cache_hit_count;
        timings.unique_sequence_build_time_ms += unique_timer.elapsed_milliseconds();
        return existing_sequence->second;
    }

    if (dataset.unique_sequences.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Unique sequence count exceeds supported integer range.");
    }
    const int cache_index = static_cast<int>(dataset.unique_sequences.size());
    dataset.unique_sequences.push_back(sequence);
    sequence_to_index.emplace(dataset.unique_sequences.back(), cache_index);
    ++dataset.cache_miss_count;
    timings.unique_sequence_build_time_ms += unique_timer.elapsed_milliseconds();
    return cache_index;
}

IndexedDataset read_dataset(const std::string& input_path, TimingBreakdown& timings) {
    std::ifstream input_file(input_path);
    if (!input_file) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    IndexedDataset dataset;
    std::unordered_map<std::string, int> sequence_to_index;
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

        const int first_index = get_or_create_sequence_index(dataset, sequence_to_index, first_sequence, timings);
        const int second_index = get_or_create_sequence_index(dataset, sequence_to_index, second_sequence, timings);

        CpuTimer index_timer;
        dataset.pair_index_a.push_back(first_index);
        dataset.pair_index_b.push_back(second_index);
        timings.index_array_build_time_ms += index_timer.elapsed_milliseconds();
    }

    if (dataset.pair_index_a.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }
    if (dataset.pair_index_a.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Number of pairs exceeds supported integer range.");
    }

    dataset.number_of_pairs = static_cast<int>(dataset.pair_index_a.size());
    return dataset;
}

void validate_dataset(const IndexedDataset& dataset) {
    if (dataset.number_of_pairs <= 0 || dataset.sequence_length <= 0) {
        throw std::runtime_error("Input dataset is empty.");
    }
    if (dataset.pair_index_a.size() != dataset.pair_index_b.size()) {
        throw std::runtime_error("Pair index arrays have inconsistent sizes.");
    }
    for (const std::string& sequence : dataset.unique_sequences) {
        if (sequence.size() != static_cast<std::size_t>(dataset.sequence_length)) {
            throw std::runtime_error("Unique sequences have inconsistent lengths.");
        }
        if (!dna::validateDnaSequence(sequence)) {
            throw std::runtime_error("Found an invalid DNA sequence. Allowed bases are A, C, G, and T.");
        }
    }
}

void fill_pinned_buffers(const IndexedDataset& dataset, PinnedHostBuffers& host_buffers) {
    for (std::size_t sequence_index = 0; sequence_index < dataset.unique_sequences.size(); ++sequence_index) {
        const std::string& sequence = dataset.unique_sequences[sequence_index];
        const std::size_t output_offset = sequence_index * static_cast<std::size_t>(dataset.sequence_length);
        for (int base_index = 0; base_index < dataset.sequence_length; ++base_index) {
            host_buffers.unique_encoded_fragments[output_offset + static_cast<std::size_t>(base_index)] =
                dna::encodeDnaBase(sequence[static_cast<std::size_t>(base_index)]);
        }
    }
    for (int pair_index = 0; pair_index < dataset.number_of_pairs; ++pair_index) {
        host_buffers.pair_index_a[pair_index] = dataset.pair_index_a[static_cast<std::size_t>(pair_index)];
        host_buffers.pair_index_b[pair_index] = dataset.pair_index_b[static_cast<std::size_t>(pair_index)];
    }
}

std::vector<int> compute_cpu_reference(const IndexedDataset& dataset, const PinnedHostBuffers& host_buffers) {
    std::vector<int> distances(static_cast<std::size_t>(dataset.number_of_pairs));
    for (int pair_index = 0; pair_index < dataset.number_of_pairs; ++pair_index) {
        const int fragment_a = host_buffers.pair_index_a[pair_index];
        const int fragment_b = host_buffers.pair_index_b[pair_index];
        const int offset_a = fragment_a * dataset.sequence_length;
        const int offset_b = fragment_b * dataset.sequence_length;
        int distance = 0;
        for (int base_index = 0; base_index < dataset.sequence_length; ++base_index) {
            if (host_buffers.unique_encoded_fragments[offset_a + base_index] !=
                host_buffers.unique_encoded_fragments[offset_b + base_index]) {
                ++distance;
            }
        }
        distances[static_cast<std::size_t>(pair_index)] = distance;
    }
    return distances;
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

void copy_h2d_async(const PinnedHostBuffers& host_buffers,
                    const DeviceBuffers& device_buffers,
                    std::size_t unique_fragment_bytes,
                    std::size_t pair_index_bytes,
                    cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(device_buffers.unique_encoded_fragments,
                               host_buffers.unique_encoded_fragments,
                               unique_fragment_bytes,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(device_buffers.pair_index_a,
                               host_buffers.pair_index_a,
                               pair_index_bytes,
                               cudaMemcpyHostToDevice,
                               stream));
    CUDA_CHECK(cudaMemcpyAsync(device_buffers.pair_index_b,
                               host_buffers.pair_index_b,
                               pair_index_bytes,
                               cudaMemcpyHostToDevice,
                               stream));
}

void launch_indexed_kernel(const DeviceBuffers& device_buffers,
                           int number_of_pairs,
                           int sequence_length,
                           cudaStream_t stream) {
    constexpr int threads_per_block = 256;
    const int blocks_per_grid = (number_of_pairs + threads_per_block - 1) / threads_per_block;
    hammingIndexedEncodedKernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        device_buffers.unique_encoded_fragments,
        device_buffers.pair_index_a,
        device_buffers.pair_index_b,
        device_buffers.distances,
        number_of_pairs,
        sequence_length);
    CUDA_CHECK(cudaGetLastError());
}

void copy_d2h_async(PinnedHostBuffers& host_buffers,
                    const DeviceBuffers& device_buffers,
                    std::size_t result_bytes,
                    cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(host_buffers.distances,
                               device_buffers.distances,
                               result_bytes,
                               cudaMemcpyDeviceToHost,
                               stream));
}

void run_stream_pipeline(const IndexedDataset& dataset,
                         PinnedHostBuffers& host_buffers,
                         const DeviceBuffers& device_buffers,
                         std::size_t unique_fragment_bytes,
                         std::size_t pair_index_bytes,
                         std::size_t result_bytes,
                         int repetitions,
                         cudaStream_t stream,
                         TimingBreakdown& timings) {
    double total_h2d_time_ms = 0.0;
    double total_kernel_time_ms = 0.0;
    double total_d2h_time_ms = 0.0;
    cudaEvent_t kernel_start_event = nullptr;
    cudaEvent_t kernel_stop_event = nullptr;
    CUDA_CHECK(cudaEventCreate(&kernel_start_event));
    CUDA_CHECK(cudaEventCreate(&kernel_stop_event));

    for (int repetition = 0; repetition < repetitions; ++repetition) {
        CpuTimer h2d_timer;
        copy_h2d_async(host_buffers, device_buffers, unique_fragment_bytes, pair_index_bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        total_h2d_time_ms += h2d_timer.elapsed_milliseconds();

        CUDA_CHECK(cudaEventRecord(kernel_start_event, stream));
        launch_indexed_kernel(device_buffers, dataset.number_of_pairs, dataset.sequence_length, stream);
        CUDA_CHECK(cudaEventRecord(kernel_stop_event, stream));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop_event));
        float kernel_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, kernel_start_event, kernel_stop_event));
        total_kernel_time_ms += static_cast<double>(kernel_time_ms);

        CpuTimer d2h_timer;
        copy_d2h_async(host_buffers, device_buffers, result_bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        total_d2h_time_ms += d2h_timer.elapsed_milliseconds();
    }

    CUDA_CHECK(cudaEventDestroy(kernel_start_event));
    CUDA_CHECK(cudaEventDestroy(kernel_stop_event));
    timings.h2d_copy_time_ms = total_h2d_time_ms / static_cast<double>(repetitions);
    timings.gpu_kernel_time_ms = total_kernel_time_ms / static_cast<double>(repetitions);
    timings.d2h_copy_time_ms = total_d2h_time_ms / static_cast<double>(repetitions);
    timings.gpu_pipeline_time_ms = timings.h2d_copy_time_ms + timings.gpu_kernel_time_ms + timings.d2h_copy_time_ms;
}

void measure_component_estimates(const IndexedDataset& dataset,
                                 PinnedHostBuffers& host_buffers,
                                 const DeviceBuffers& device_buffers,
                                 std::size_t unique_fragment_bytes,
                                 std::size_t pair_index_bytes,
                                 std::size_t result_bytes,
                                 cudaStream_t stream,
                                 TimingBreakdown& timings) {
    run_stream_pipeline(dataset,
                        host_buffers,
                        device_buffers,
                        unique_fragment_bytes,
                        pair_index_bytes,
                        result_bytes,
                        1,
                        stream,
                        timings);
}

void run_graph_pipeline(const IndexedDataset& dataset,
                        PinnedHostBuffers& host_buffers,
                        const DeviceBuffers& device_buffers,
                        std::size_t unique_fragment_bytes,
                        std::size_t pair_index_bytes,
                        std::size_t result_bytes,
                        int repetitions,
                        cudaStream_t stream,
                        TimingBreakdown& timings) {
#if CUDART_VERSION >= 10000
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    CpuTimer graph_creation_timer;
    CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    copy_h2d_async(host_buffers, device_buffers, unique_fragment_bytes, pair_index_bytes, stream);
    launch_indexed_kernel(device_buffers, dataset.number_of_pairs, dataset.sequence_length, stream);
    copy_d2h_async(host_buffers, device_buffers, result_bytes, stream);
    CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    timings.graph_creation_time_ms = graph_creation_timer.elapsed_milliseconds();

    CpuTimer graph_instantiation_timer;
#if CUDART_VERSION >= 12000
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, 0));
#else
    CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
#endif
    timings.graph_instantiation_time_ms = graph_instantiation_timer.elapsed_milliseconds();

    CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    double total_graph_execution_time_ms = 0.0;
    for (int repetition = 0; repetition < repetitions; ++repetition) {
        CpuTimer graph_execution_timer;
        CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        total_graph_execution_time_ms += graph_execution_timer.elapsed_milliseconds();
    }

    timings.graph_execution_time_ms = total_graph_execution_time_ms;
    timings.graph_average_execution_time_ms = total_graph_execution_time_ms / static_cast<double>(repetitions);
    timings.gpu_pipeline_time_ms = timings.graph_average_execution_time_ms;

    CUDA_CHECK(cudaGraphExecDestroy(graph_exec));
    CUDA_CHECK(cudaGraphDestroy(graph));
#else
    (void)dataset;
    (void)host_buffers;
    (void)device_buffers;
    (void)unique_fragment_bytes;
    (void)pair_index_bytes;
    (void)result_bytes;
    (void)repetitions;
    (void)stream;
    (void)timings;
    throw std::runtime_error("CUDA Graphs are not available in this CUDA runtime.");
#endif
}

int main(int argc, char** argv) {
    PinnedHostBuffers host_buffers;
    DeviceBuffers device_buffers;
    cudaStream_t stream = nullptr;
    bool cleanup_uses_cuda_malloc_async = false;

    try {
        CpuTimer end_to_end_timer;
        TimingBreakdown timings;
        const ProgramOptions options = parse_arguments(argc, argv);

        const bool cuda_graphs_supported = runtime_supports_cuda_graphs();
        const bool cuda_malloc_async_supported = runtime_supports_cuda_malloc_async();
        const bool use_cuda_graphs = options.request_cuda_graphs && cuda_graphs_supported;
        const bool use_cuda_malloc_async = options.request_cuda_malloc_async && cuda_malloc_async_supported;
        cleanup_uses_cuda_malloc_async = use_cuda_malloc_async;
        const bool fallback_mode = !use_cuda_graphs || !use_cuda_malloc_async;

        CUDA_CHECK(cudaStreamCreate(&stream));

        CpuTimer file_read_timer;
        IndexedDataset dataset = read_dataset(options.input_path, timings);
        timings.file_read_time_ms = file_read_timer.elapsed_milliseconds();

        CpuTimer input_validation_timer;
        validate_dataset(dataset);
        timings.input_validation_time_ms = input_validation_timer.elapsed_milliseconds();

        const std::size_t unique_fragment_bytes = dataset.unique_sequences.size() *
                                                  static_cast<std::size_t>(dataset.sequence_length) *
                                                  sizeof(std::uint8_t);
        const std::size_t pair_index_bytes = static_cast<std::size_t>(dataset.number_of_pairs) * sizeof(int);
        const std::size_t result_bytes = static_cast<std::size_t>(dataset.number_of_pairs) * sizeof(int);
        const std::size_t flat_pair_bytes_equivalent = static_cast<std::size_t>(dataset.number_of_pairs) *
                                                       static_cast<std::size_t>(dataset.sequence_length) *
                                                       2 * sizeof(std::uint8_t);
        const std::size_t indexed_representation_bytes = unique_fragment_bytes + 2 * pair_index_bytes;
        const std::size_t flat_pair_bytes_avoided =
            flat_pair_bytes_equivalent > indexed_representation_bytes
                ? flat_pair_bytes_equivalent - indexed_representation_bytes
                : 0;
        const double transfer_reduction_ratio =
            indexed_representation_bytes == 0
                ? 0.0
                : static_cast<double>(flat_pair_bytes_equivalent) /
                      static_cast<double>(indexed_representation_bytes);
        const double cache_hit_rate =
            (dataset.cache_hit_count + dataset.cache_miss_count) == 0
                ? 0.0
                : static_cast<double>(dataset.cache_hit_count) /
                      static_cast<double>(dataset.cache_hit_count + dataset.cache_miss_count);

        CpuTimer pinned_allocation_timer;
        host_buffers.allocate(unique_fragment_bytes, pair_index_bytes, result_bytes);
        timings.pinned_host_allocation_time_ms = pinned_allocation_timer.elapsed_milliseconds();

        CpuTimer encoding_timer;
        fill_pinned_buffers(dataset, host_buffers);
        timings.encoding_time_ms = encoding_timer.elapsed_milliseconds();

        CpuTimer cpu_reference_timer;
        const std::vector<int> cpu_distances = compute_cpu_reference(dataset, host_buffers);
        timings.cpu_reference_time_ms = cpu_reference_timer.elapsed_milliseconds();

        CpuTimer device_allocation_timer;
        device_buffers.allocate(unique_fragment_bytes, pair_index_bytes, result_bytes, stream, use_cuda_malloc_async);
        if (use_cuda_malloc_async) {
            timings.async_device_allocation_time_ms = device_allocation_timer.elapsed_milliseconds();
            timings.device_allocation_time_ms = timings.async_device_allocation_time_ms;
        } else {
            timings.device_allocation_time_ms = device_allocation_timer.elapsed_milliseconds();
        }

        copy_h2d_async(host_buffers, device_buffers, unique_fragment_bytes, pair_index_bytes, stream);
        launch_indexed_kernel(device_buffers, dataset.number_of_pairs, dataset.sequence_length, stream);
        copy_d2h_async(host_buffers, device_buffers, result_bytes, stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (use_cuda_graphs) {
            measure_component_estimates(dataset,
                                        host_buffers,
                                        device_buffers,
                                        unique_fragment_bytes,
                                        pair_index_bytes,
                                        result_bytes,
                                        stream,
                                        timings);
            run_graph_pipeline(dataset,
                               host_buffers,
                               device_buffers,
                               unique_fragment_bytes,
                               pair_index_bytes,
                               result_bytes,
                               options.repetitions,
                               stream,
                               timings);
        } else {
            run_stream_pipeline(dataset,
                                host_buffers,
                                device_buffers,
                                unique_fragment_bytes,
                                pair_index_bytes,
                                result_bytes,
                                options.repetitions,
                                stream,
                                timings);
        }

        ValidationResult validation;
        if (!options.skip_validation) {
            CpuTimer validation_timer;
            validation = validate_results(host_buffers.distances, cpu_distances);
            timings.validation_time_ms = validation_timer.elapsed_milliseconds();
        }

        if (!validation.passed) {
            std::cerr << "Error: indexed encoded GPU validation failed against CPU reference results.\n";
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
        device_buffers.release(stream, use_cuda_malloc_async);
        if (use_cuda_malloc_async) {
            timings.async_device_free_time_ms = device_free_timer.elapsed_milliseconds();
            timings.device_free_time_ms = timings.async_device_free_time_ms;
        } else {
            timings.device_free_time_ms = device_free_timer.elapsed_milliseconds();
        }

        host_buffers.release();
        CUDA_CHECK(cudaStreamDestroy(stream));
        stream = nullptr;

        timings.setup_time_ms = timings.pinned_host_allocation_time_ms +
                                timings.device_allocation_time_ms +
                                timings.graph_creation_time_ms +
                                timings.graph_instantiation_time_ms;
        timings.end_to_end_time_ms = end_to_end_timer.elapsed_milliseconds();

        const std::size_t total_bases_compared = static_cast<std::size_t>(dataset.number_of_pairs) *
                                                 static_cast<std::size_t>(dataset.sequence_length);

        std::cout << "Number of pairs: " << dataset.number_of_pairs << "\n";
        std::cout << "Sequence length: " << dataset.sequence_length << "\n";
        std::cout << "Unique sequence count: " << dataset.unique_sequences.size() << "\n";
        std::cout << "CUDA Graphs supported: " << (cuda_graphs_supported ? "true" : "false") << "\n";
        std::cout << "cudaMallocAsync supported: " << (cuda_malloc_async_supported ? "true" : "false") << "\n";
        std::cout << "Fallback mode: " << (fallback_mode ? "true" : "false") << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "FILE_READ_TIME_MS=" << timings.file_read_time_ms << "\n";
        std::cout << "INPUT_VALIDATION_TIME_MS=" << timings.input_validation_time_ms << "\n";
        std::cout << "UNIQUE_SEQUENCE_BUILD_TIME_MS=" << timings.unique_sequence_build_time_ms << "\n";
        std::cout << "ENCODING_TIME_MS=" << timings.encoding_time_ms << "\n";
        std::cout << "INDEX_ARRAY_BUILD_TIME_MS=" << timings.index_array_build_time_ms << "\n";
        std::cout << "PINNED_HOST_ALLOCATION_TIME_MS=" << timings.pinned_host_allocation_time_ms << "\n";
        std::cout << "DEVICE_ALLOCATION_TIME_MS=" << timings.device_allocation_time_ms << "\n";
        std::cout << "ASYNC_DEVICE_ALLOCATION_TIME_MS=" << timings.async_device_allocation_time_ms << "\n";
        std::cout << "H2D_COPY_TIME_MS=" << timings.h2d_copy_time_ms << "\n";
        std::cout << "GPU_KERNEL_TIME_MS=" << timings.gpu_kernel_time_ms << "\n";
        std::cout << "D2H_COPY_TIME_MS=" << timings.d2h_copy_time_ms << "\n";
        std::cout << "GPU_PIPELINE_TIME_MS=" << timings.gpu_pipeline_time_ms << "\n";
        std::cout << "GPU_TOTAL_TIME_MS=" << timings.gpu_pipeline_time_ms << "\n";
        std::cout << "GRAPH_CREATION_TIME_MS=" << timings.graph_creation_time_ms << "\n";
        std::cout << "GRAPH_INSTANTIATION_TIME_MS=" << timings.graph_instantiation_time_ms << "\n";
        std::cout << "GRAPH_EXECUTION_TIME_MS=" << timings.graph_execution_time_ms << "\n";
        std::cout << "GRAPH_AVERAGE_EXECUTION_TIME_MS=" << timings.graph_average_execution_time_ms << "\n";
        std::cout << "CPU_REFERENCE_TIME_MS=" << timings.cpu_reference_time_ms << "\n";
        std::cout << "VALIDATION_TIME_MS=" << timings.validation_time_ms << "\n";
        std::cout << "CSV_WRITE_TIME_MS=" << timings.csv_write_time_ms << "\n";
        std::cout << "DEVICE_FREE_TIME_MS=" << timings.device_free_time_ms << "\n";
        std::cout << "ASYNC_DEVICE_FREE_TIME_MS=" << timings.async_device_free_time_ms << "\n";
        std::cout << "SETUP_TIME_MS=" << timings.setup_time_ms << "\n";
        std::cout << "END_TO_END_TIME_MS=" << timings.end_to_end_time_ms << "\n";
        std::cout << "UNIQUE_SEQUENCE_COUNT=" << dataset.unique_sequences.size() << "\n";
        std::cout << "NUMBER_OF_PAIRS=" << dataset.number_of_pairs << "\n";
        std::cout << "SEQUENCE_LENGTH=" << dataset.sequence_length << "\n";
        std::cout << "TOTAL_BASES_COMPARED=" << total_bases_compared << "\n";
        std::cout << "UNIQUE_FRAGMENT_BYTES=" << unique_fragment_bytes << "\n";
        std::cout << "PAIR_INDEX_BYTES=" << pair_index_bytes * 2 << "\n";
        std::cout << "RESULT_BYTES=" << result_bytes << "\n";
        std::cout << "FLAT_PAIR_BYTES_EQUIVALENT=" << flat_pair_bytes_equivalent << "\n";
        std::cout << "INDEXED_REPRESENTATION_BYTES=" << indexed_representation_bytes << "\n";
        std::cout << "FLAT_PAIR_BYTES_AVOIDED=" << flat_pair_bytes_avoided << "\n";
        std::cout << "TRANSFER_REDUCTION_RATIO=" << transfer_reduction_ratio << "\n";
        std::cout << "CACHE_HIT_COUNT=" << dataset.cache_hit_count << "\n";
        std::cout << "CACHE_MISS_COUNT=" << dataset.cache_miss_count << "\n";
        std::cout << "CACHE_HIT_RATE=" << cache_hit_rate << "\n";
        std::cout << "CUDA_GRAPHS_SUPPORTED=" << (cuda_graphs_supported ? "true" : "false") << "\n";
        std::cout << "USE_CUDA_GRAPHS=" << (use_cuda_graphs ? "true" : "false") << "\n";
        std::cout << "CUDA_MALLOC_ASYNC_SUPPORTED=" << (cuda_malloc_async_supported ? "true" : "false") << "\n";
        std::cout << "USE_CUDA_MALLOC_ASYNC=" << (use_cuda_malloc_async ? "true" : "false") << "\n";
        std::cout << "FALLBACK_MODE=" << (fallback_mode ? "true" : "false") << "\n";
        std::cout << "SUMMARY_ONLY=" << (options.summary_only ? "true" : "false") << "\n";
        std::cout << "INDEXED_MODE=true\n";
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
        device_buffers.release(stream, cleanup_uses_cuda_malloc_async);
        host_buffers.release();
        if (stream != nullptr) {
            cudaStreamDestroy(stream);
        }
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
