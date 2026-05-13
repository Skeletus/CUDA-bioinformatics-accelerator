#include <cuda_runtime.h>

#include <cstdint>
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

__global__ void hammingEncodedKernel(const std::uint8_t* sequence_a,
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

struct Dataset {
    std::vector<char> first_sequences_char;
    std::vector<char> second_sequences_char;
    int number_of_pairs = 0;
    int sequence_length = 0;
};

struct ValidationResult {
    bool passed = true;
    std::size_t first_mismatch_pair_id = 0;
    int cpu_distance = 0;
    int gpu_distance = 0;
};

int parse_repetitions(int argc, char** argv) {
    constexpr int default_repetitions = 5;
    if (argc == 3) {
        return default_repetitions;
    }
    if (argc == 5 && std::string(argv[3]) == "--repetitions") {
        const int repetitions = std::stoi(argv[4]);
        if (repetitions <= 0) {
            throw std::runtime_error("--repetitions must be greater than zero.");
        }
        return repetitions;
    }

    throw std::runtime_error("Usage: " + std::string(argv[0]) +
                             " <input_dataset> <output_csv> [--repetitions N]");
}

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
        if (!dna::validateDnaSequence(first_sequence) || !dna::validateDnaSequence(second_sequence)) {
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

        dataset.first_sequences_char.insert(dataset.first_sequences_char.end(),
                                            first_sequence.begin(),
                                            first_sequence.end());
        dataset.second_sequences_char.insert(dataset.second_sequences_char.end(),
                                             second_sequence.begin(),
                                             second_sequence.end());
    }

    if (dataset.first_sequences_char.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }

    const std::size_t number_of_pairs = dataset.first_sequences_char.size() /
                                        static_cast<std::size_t>(dataset.sequence_length);
    if (number_of_pairs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Number of pairs exceeds supported integer range.");
    }

    dataset.number_of_pairs = static_cast<int>(number_of_pairs);
    return dataset;
}

std::vector<int> compute_cpu_distances(const std::vector<std::uint8_t>& sequence_a,
                                       const std::vector<std::uint8_t>& sequence_b,
                                       int number_of_pairs,
                                       int sequence_length) {
    std::vector<int> distances(static_cast<std::size_t>(number_of_pairs));

    for (int pair_index = 0; pair_index < number_of_pairs; ++pair_index) {
        const int offset = pair_index * sequence_length;
        int distance = 0;
        for (int base_index = 0; base_index < sequence_length; ++base_index) {
            if (sequence_a[offset + base_index] != sequence_b[offset + base_index]) {
                ++distance;
            }
        }
        distances[static_cast<std::size_t>(pair_index)] = distance;
    }

    return distances;
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

ValidationResult validate_results(const std::vector<int>& gpu_distances,
                                  const std::vector<int>& cpu_distances) {
    ValidationResult result;
    if (gpu_distances.size() != cpu_distances.size()) {
        result.passed = false;
        result.cpu_distance = static_cast<int>(cpu_distances.size());
        result.gpu_distance = static_cast<int>(gpu_distances.size());
        return result;
    }

    for (std::size_t pair_index = 0; pair_index < gpu_distances.size(); ++pair_index) {
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

int main(int argc, char** argv) {
    std::uint8_t* device_sequence_a = nullptr;
    std::uint8_t* device_sequence_b = nullptr;
    int* device_distances = nullptr;
    cudaEvent_t kernel_start_event = nullptr;
    cudaEvent_t kernel_stop_event = nullptr;

    try {
        const int repetitions = parse_repetitions(argc, argv);
        const std::string input_path = argv[1];
        const std::string output_path = argv[2];
        const Dataset dataset = read_dataset(input_path);

        CpuTimer encoding_timer;
        const std::vector<std::uint8_t> encoded_sequence_a =
            dna::encodeFlatDnaSequences(dataset.first_sequences_char);
        const std::vector<std::uint8_t> encoded_sequence_b =
            dna::encodeFlatDnaSequences(dataset.second_sequences_char);
        const double encoding_time_ms = encoding_timer.elapsed_milliseconds();

        const std::vector<int> cpu_distances = compute_cpu_distances(
            encoded_sequence_a,
            encoded_sequence_b,
            dataset.number_of_pairs,
            dataset.sequence_length);
        std::vector<int> gpu_distances(static_cast<std::size_t>(dataset.number_of_pairs));

        const std::size_t sequence_bytes = encoded_sequence_a.size() * sizeof(std::uint8_t);
        const std::size_t distance_bytes = gpu_distances.size() * sizeof(int);

        CUDA_CHECK(cudaMalloc(&device_sequence_a, sequence_bytes));
        CUDA_CHECK(cudaMalloc(&device_sequence_b, sequence_bytes));
        CUDA_CHECK(cudaMalloc(&device_distances, distance_bytes));
        CUDA_CHECK(cudaEventCreate(&kernel_start_event));
        CUDA_CHECK(cudaEventCreate(&kernel_stop_event));

        CUDA_CHECK(cudaMemcpy(device_sequence_a, encoded_sequence_a.data(), sequence_bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(device_sequence_b, encoded_sequence_b.data(), sequence_bytes, cudaMemcpyHostToDevice));

        const int threads_per_block = 256;
        const int blocks_per_grid = (dataset.number_of_pairs + threads_per_block - 1) / threads_per_block;

        hammingEncodedKernel<<<blocks_per_grid, threads_per_block>>>(
            device_sequence_a,
            device_sequence_b,
            device_distances,
            dataset.number_of_pairs,
            dataset.sequence_length);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        double total_kernel_time_ms = 0.0;
        double total_transfer_compute_time_ms = 0.0;
        float minimum_kernel_time_ms = std::numeric_limits<float>::max();
        double minimum_transfer_compute_time_ms = std::numeric_limits<double>::max();

        for (int repetition = 0; repetition < repetitions; ++repetition) {
            CpuTimer total_timer;
            CUDA_CHECK(cudaMemcpy(device_sequence_a, encoded_sequence_a.data(), sequence_bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(device_sequence_b, encoded_sequence_b.data(), sequence_bytes, cudaMemcpyHostToDevice));

            CUDA_CHECK(cudaEventRecord(kernel_start_event));
            hammingEncodedKernel<<<blocks_per_grid, threads_per_block>>>(
                device_sequence_a,
                device_sequence_b,
                device_distances,
                dataset.number_of_pairs,
                dataset.sequence_length);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(kernel_stop_event));
            CUDA_CHECK(cudaEventSynchronize(kernel_stop_event));

            float repetition_kernel_time_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&repetition_kernel_time_ms, kernel_start_event, kernel_stop_event));

            CUDA_CHECK(cudaMemcpy(gpu_distances.data(), device_distances, distance_bytes, cudaMemcpyDeviceToHost));
            const double repetition_transfer_compute_time_ms = total_timer.elapsed_milliseconds();

            total_kernel_time_ms += static_cast<double>(repetition_kernel_time_ms);
            total_transfer_compute_time_ms += repetition_transfer_compute_time_ms;
            if (repetition_kernel_time_ms < minimum_kernel_time_ms) {
                minimum_kernel_time_ms = repetition_kernel_time_ms;
            }
            if (repetition_transfer_compute_time_ms < minimum_transfer_compute_time_ms) {
                minimum_transfer_compute_time_ms = repetition_transfer_compute_time_ms;
            }
        }

        const double average_kernel_time_ms = total_kernel_time_ms / static_cast<double>(repetitions);
        const double average_transfer_compute_time_ms =
            total_transfer_compute_time_ms / static_cast<double>(repetitions);
        const double average_gpu_total_time_ms = encoding_time_ms + average_transfer_compute_time_ms;
        const double minimum_gpu_total_time_ms = encoding_time_ms + minimum_transfer_compute_time_ms;

        const ValidationResult validation = validate_results(gpu_distances, cpu_distances);
        if (!validation.passed) {
            std::cerr << "Error: Encoded GPU validation failed against CPU reference results.\n";
            std::cerr << "First mismatching pair ID: " << validation.first_mismatch_pair_id << "\n";
            std::cerr << "CPU distance: " << validation.cpu_distance << "\n";
            std::cerr << "GPU distance: " << validation.gpu_distance << "\n";
        }

        write_results_csv(output_path, gpu_distances, dataset.sequence_length);

        std::cout << "Number of pairs: " << dataset.number_of_pairs << "\n";
        std::cout << "Sequence length: " << dataset.sequence_length << "\n";
        std::cout << "Encoding time: " << std::fixed << std::setprecision(3) << encoding_time_ms << " ms\n";
        std::cout << "GPU average kernel time: " << std::fixed << std::setprecision(3) << average_kernel_time_ms
                  << " ms\n";
        std::cout << "GPU minimum kernel time: " << std::fixed << std::setprecision(3) << minimum_kernel_time_ms
                  << " ms\n";
        std::cout << "GPU average total time: " << std::fixed << std::setprecision(3) << average_gpu_total_time_ms
                  << " ms\n";
        std::cout << "GPU minimum total time: " << std::fixed << std::setprecision(3) << minimum_gpu_total_time_ms
                  << " ms\n";
        std::cout << "Repetitions: " << repetitions << "\n";
        std::cout << "Validation status: " << (validation.passed ? "PASSED" : "FAILED") << "\n";
        std::cout << "Output path: " << output_path << "\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "ENCODING_TIME_MS=" << encoding_time_ms << "\n";
        std::cout << "GPU_KERNEL_TIME_MS=" << average_kernel_time_ms << "\n";
        std::cout << "GPU_KERNEL_MIN_TIME_MS=" << minimum_kernel_time_ms << "\n";
        std::cout << "GPU_TRANSFER_COMPUTE_TIME_MS=" << average_transfer_compute_time_ms << "\n";
        std::cout << "GPU_TOTAL_TIME_MS=" << average_gpu_total_time_ms << "\n";
        std::cout << "GPU_TOTAL_MIN_TIME_MS=" << minimum_gpu_total_time_ms << "\n";
        std::cout << "NUMBER_OF_PAIRS=" << dataset.number_of_pairs << "\n";
        std::cout << "SEQUENCE_LENGTH=" << dataset.sequence_length << "\n";
        std::cout << "REPETITIONS=" << repetitions << "\n";
        std::cout << "VALIDATION_STATUS=" << (validation.passed ? "PASSED" : "FAILED") << "\n";
        if (!validation.passed) {
            std::cout << "FIRST_MISMATCH_PAIR_ID=" << validation.first_mismatch_pair_id << "\n";
            std::cout << "CPU_DISTANCE=" << validation.cpu_distance << "\n";
            std::cout << "GPU_DISTANCE=" << validation.gpu_distance << "\n";
        }
        std::cout << "OUTPUT_PATH=" << output_path << "\n";

        CUDA_CHECK(cudaEventDestroy(kernel_start_event));
        CUDA_CHECK(cudaEventDestroy(kernel_stop_event));
        CUDA_CHECK(cudaFree(device_sequence_a));
        CUDA_CHECK(cudaFree(device_sequence_b));
        CUDA_CHECK(cudaFree(device_distances));

        if (!validation.passed) {
            return 2;
        }
    } catch (const std::exception& error) {
        if (kernel_start_event != nullptr) {
            cudaEventDestroy(kernel_start_event);
        }
        if (kernel_stop_event != nullptr) {
            cudaEventDestroy(kernel_stop_event);
        }
        if (device_sequence_a != nullptr) {
            cudaFree(device_sequence_a);
        }
        if (device_sequence_b != nullptr) {
            cudaFree(device_sequence_b);
        }
        if (device_distances != nullptr) {
            cudaFree(device_distances);
        }

        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
