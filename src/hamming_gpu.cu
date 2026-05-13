#include <cuda_runtime.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_utils.h"
#include "dna_utils.h"
#include "timer.h"

__global__ void hamming_distance_kernel(const char* first_sequences,
                                        const char* second_sequences,
                                        int* distances,
                                        int num_pairs,
                                        int sequence_length) {
    const int pair_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_index >= num_pairs) {
        return;
    }

    int distance = 0;
    const int sequence_offset = pair_index * sequence_length;
    for (int base_index = 0; base_index < sequence_length; ++base_index) {
        if (first_sequences[sequence_offset + base_index] != second_sequences[sequence_offset + base_index]) {
            ++distance;
        }
    }

    distances[pair_index] = distance;
}

struct Dataset {
    std::vector<char> first_sequences;
    std::vector<char> second_sequences;
    std::vector<int> cpu_distances;
    int num_pairs = 0;
    int sequence_length = 0;
};

Dataset read_dataset(const std::string& input_path) {
    std::ifstream input_file(input_path);
    if (!input_file) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    Dataset dataset;
    std::string first_sequence;
    std::string second_sequence;
    bool has_sequence_length = false;

    while (input_file >> first_sequence >> second_sequence) {
        if (first_sequence.size() != second_sequence.size()) {
            throw std::runtime_error("Found a sequence pair with unequal lengths.");
        }
        if (!dna::is_valid_dna_sequence(first_sequence) || !dna::is_valid_dna_sequence(second_sequence)) {
            throw std::runtime_error("Found an invalid DNA sequence. Allowed bases are A, C, G, and T.");
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

        dataset.first_sequences.insert(dataset.first_sequences.end(), first_sequence.begin(), first_sequence.end());
        dataset.second_sequences.insert(dataset.second_sequences.end(), second_sequence.begin(), second_sequence.end());
        dataset.cpu_distances.push_back(static_cast<int>(dna::hamming_distance(first_sequence, second_sequence)));
    }

    if (dataset.cpu_distances.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }
    if (dataset.cpu_distances.size() > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Number of pairs exceeds supported integer range.");
    }

    dataset.num_pairs = static_cast<int>(dataset.cpu_distances.size());
    return dataset;
}

void write_results_csv(const std::string& output_path,
                       const std::vector<int>& distances,
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
    for (std::size_t pair_index = 0; pair_index < distances.size(); ++pair_index) {
        const double similarity = 100.0 * (1.0 - static_cast<double>(distances[pair_index]) /
                                                     static_cast<double>(sequence_length));
        output_file << pair_index << "," << distances[pair_index] << "," << similarity << "\n";
    }
}

bool validate_results(const std::vector<int>& gpu_distances, const std::vector<int>& cpu_distances) {
    if (gpu_distances.size() != cpu_distances.size()) {
        return false;
    }

    for (std::size_t pair_index = 0; pair_index < gpu_distances.size(); ++pair_index) {
        if (gpu_distances[pair_index] != cpu_distances[pair_index]) {
            return false;
        }
    }

    return true;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dataset> <output_csv>\n";
        return 1;
    }

    char* device_first_sequences = nullptr;
    char* device_second_sequences = nullptr;
    int* device_distances = nullptr;
    cudaEvent_t kernel_start_event = nullptr;
    cudaEvent_t kernel_stop_event = nullptr;

    try {
        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        const Dataset dataset = read_dataset(input_path);

        std::vector<int> gpu_distances(dataset.num_pairs);
        const std::size_t sequence_bytes = dataset.first_sequences.size() * sizeof(char);
        const std::size_t distance_bytes = gpu_distances.size() * sizeof(int);

        CUDA_CHECK(cudaMalloc(&device_first_sequences, sequence_bytes));
        CUDA_CHECK(cudaMalloc(&device_second_sequences, sequence_bytes));
        CUDA_CHECK(cudaMalloc(&device_distances, distance_bytes));
        CUDA_CHECK(cudaEventCreate(&kernel_start_event));
        CUDA_CHECK(cudaEventCreate(&kernel_stop_event));

        CpuTimer total_timer;
        CUDA_CHECK(cudaMemcpy(device_first_sequences, dataset.first_sequences.data(), sequence_bytes,
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_second_sequences, dataset.second_sequences.data(), sequence_bytes,
                              cudaMemcpyHostToDevice));

        const int threads_per_block = 256;
        const int blocks_per_grid = (dataset.num_pairs + threads_per_block - 1) / threads_per_block;

        CUDA_CHECK(cudaEventRecord(kernel_start_event));
        hamming_distance_kernel<<<blocks_per_grid, threads_per_block>>>(
            device_first_sequences,
            device_second_sequences,
            device_distances,
            dataset.num_pairs,
            dataset.sequence_length);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(kernel_stop_event));
        CUDA_CHECK(cudaEventSynchronize(kernel_stop_event));

        float kernel_time_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_time_ms, kernel_start_event, kernel_stop_event));

        CUDA_CHECK(cudaMemcpy(gpu_distances.data(), device_distances, distance_bytes, cudaMemcpyDeviceToHost));
        const double total_time_ms = total_timer.elapsed_milliseconds();

        const bool validation_passed = validate_results(gpu_distances, dataset.cpu_distances);
        if (!validation_passed) {
            throw std::runtime_error("GPU validation failed against CPU reference results.");
        }

        write_results_csv(output_path, gpu_distances, dataset.sequence_length);

        std::cout << "Number of pairs: " << dataset.num_pairs << "\n";
        std::cout << "Sequence length: " << dataset.sequence_length << "\n";
        std::cout << "GPU kernel time: " << std::fixed << std::setprecision(3) << kernel_time_ms << " ms\n";
        std::cout << "GPU total time: " << std::fixed << std::setprecision(3) << total_time_ms << " ms\n";
        std::cout << "Validation status: PASSED\n";
        std::cout << "Output path: " << output_path << "\n";

        CUDA_CHECK(cudaEventDestroy(kernel_start_event));
        CUDA_CHECK(cudaEventDestroy(kernel_stop_event));
        CUDA_CHECK(cudaFree(device_first_sequences));
        CUDA_CHECK(cudaFree(device_second_sequences));
        CUDA_CHECK(cudaFree(device_distances));
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
        if (device_distances != nullptr) {
            cudaFree(device_distances);
        }

        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
