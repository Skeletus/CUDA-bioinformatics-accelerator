#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "dna_utils.h"
#include "timer.h"

struct SequencePair {
    std::string first_sequence;
    std::string second_sequence;
};

std::vector<SequencePair> read_sequence_pairs(const std::string& input_path, std::size_t& sequence_length) {
    std::ifstream input_file(input_path);
    if (!input_file) {
        throw std::runtime_error("Failed to open input file: " + input_path);
    }

    std::vector<SequencePair> sequence_pairs;
    std::string first_sequence;
    std::string second_sequence;
    bool has_sequence_length = false;
    sequence_length = 0;

    while (input_file >> first_sequence >> second_sequence) {
        if (first_sequence.size() != second_sequence.size()) {
            throw std::runtime_error("Found a sequence pair with unequal lengths.");
        }
        if (!dna::is_valid_dna_sequence(first_sequence) || !dna::is_valid_dna_sequence(second_sequence)) {
            throw std::runtime_error("Found an invalid DNA sequence. Allowed bases are A, C, G, and T.");
        }
        if (!has_sequence_length) {
            sequence_length = first_sequence.size();
            has_sequence_length = true;
        } else if (first_sequence.size() != sequence_length) {
            throw std::runtime_error("Found sequence pairs with different fixed lengths.");
        }

        sequence_pairs.push_back({first_sequence, second_sequence});
    }

    if (sequence_pairs.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }

    return sequence_pairs;
}

void write_results_csv(const std::string& output_path,
                       const std::vector<std::size_t>& distances,
                       std::size_t sequence_length) {
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

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dataset> <output_csv>\n";
        return 1;
    }

    try {
        const std::string input_path = argv[1];
        const std::string output_path = argv[2];

        std::size_t sequence_length = 0;
        const std::vector<SequencePair> sequence_pairs = read_sequence_pairs(input_path, sequence_length);
        std::vector<std::size_t> distances(sequence_pairs.size());

        CpuTimer timer;
        for (std::size_t pair_index = 0; pair_index < sequence_pairs.size(); ++pair_index) {
            distances[pair_index] = dna::hamming_distance(sequence_pairs[pair_index].first_sequence,
                                                          sequence_pairs[pair_index].second_sequence);
        }
        const double cpu_time_ms = timer.elapsed_milliseconds();

        write_results_csv(output_path, distances, sequence_length);

        std::cout << "Number of pairs: " << sequence_pairs.size() << "\n";
        std::cout << "Sequence length: " << sequence_length << "\n";
        std::cout << "CPU time: " << std::fixed << std::setprecision(3) << cpu_time_ms << " ms\n";
        std::cout << "Output path: " << output_path << "\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
