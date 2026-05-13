#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "dna_utils.h"

struct Dataset {
    std::vector<char> first_sequences;
    std::vector<char> second_sequences;
    int number_of_pairs = 0;
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

        dataset.first_sequences.insert(dataset.first_sequences.end(), first_sequence.begin(), first_sequence.end());
        dataset.second_sequences.insert(dataset.second_sequences.end(), second_sequence.begin(), second_sequence.end());
    }

    if (dataset.first_sequences.empty()) {
        throw std::runtime_error("Input dataset is empty.");
    }

    const std::size_t number_of_pairs = dataset.first_sequences.size() /
                                        static_cast<std::size_t>(dataset.sequence_length);
    if (number_of_pairs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("Number of pairs exceeds supported integer range.");
    }

    dataset.number_of_pairs = static_cast<int>(number_of_pairs);
    return dataset;
}

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            throw std::runtime_error("Usage: " + std::string(argv[0]) + " <input_dataset>");
        }

        const std::string input_path = argv[1];
        const Dataset dataset = read_dataset(input_path);

        const std::vector<std::uint8_t> encoded_first_sequences =
            dna::encodeFlatDnaSequences(dataset.first_sequences);
        const std::vector<std::uint8_t> encoded_second_sequences =
            dna::encodeFlatDnaSequences(dataset.second_sequences);

        const std::size_t raw_char_memory_bytes =
            (dataset.first_sequences.size() + dataset.second_sequences.size()) * sizeof(char);
        const std::size_t encoded_memory_bytes =
            (encoded_first_sequences.size() + encoded_second_sequences.size()) * sizeof(std::uint8_t);

        std::cout << "DNA encoding summary\n";
        std::cout << "Number of pairs: " << dataset.number_of_pairs << "\n";
        std::cout << "Sequence length: " << dataset.sequence_length << "\n";
        std::cout << "Raw char memory size in bytes: " << raw_char_memory_bytes << "\n";
        std::cout << "Encoded uint8_t memory size in bytes: " << encoded_memory_bytes << "\n";
        std::cout << "Encoding succeeded: true\n";
        std::cout << "Note: char and uint8_t both use 1 byte per base in this phase.\n";
        std::cout << "The numeric representation prepares the project for future 2-bit packing and scoring matrices.\n";
        std::cout << "NUMBER_OF_PAIRS=" << dataset.number_of_pairs << "\n";
        std::cout << "SEQUENCE_LENGTH=" << dataset.sequence_length << "\n";
        std::cout << "RAW_CHAR_MEMORY_BYTES=" << raw_char_memory_bytes << "\n";
        std::cout << "ENCODED_UINT8_MEMORY_BYTES=" << encoded_memory_bytes << "\n";
        std::cout << "ENCODING_STATUS=PASSED\n";
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        std::cout << "ENCODING_STATUS=FAILED\n";
        return 1;
    }

    return 0;
}
