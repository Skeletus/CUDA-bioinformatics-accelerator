#ifndef DNA_UTILS_H
#define DNA_UTILS_H

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dna {

inline bool isValidDnaBase(char base) {
    return base == 'A' || base == 'a' || base == 'C' || base == 'c' ||
           base == 'G' || base == 'g' || base == 'T' || base == 't';
}

inline bool is_valid_dna_base(char base) {
    return isValidDnaBase(base);
}

inline std::uint8_t encodeDnaBase(char base) {
    switch (base) {
        case 'A':
        case 'a':
            return 0;
        case 'C':
        case 'c':
            return 1;
        case 'G':
        case 'g':
            return 2;
        case 'T':
        case 't':
            return 3;
        default:
            throw std::invalid_argument("Invalid DNA base. Allowed bases are A, C, G, and T.");
    }
}

inline bool validateDnaSequence(const std::string& sequence) {
    for (char base : sequence) {
        if (!isValidDnaBase(base)) {
            return false;
        }
    }
    return true;
}

inline bool is_valid_dna_sequence(const std::string& sequence) {
    return validateDnaSequence(sequence);
}

inline std::vector<std::uint8_t> encodeDnaSequence(const std::string& sequence) {
    std::vector<std::uint8_t> encoded_sequence;
    encoded_sequence.reserve(sequence.size());

    for (char base : sequence) {
        encoded_sequence.push_back(encodeDnaBase(base));
    }

    return encoded_sequence;
}

inline std::vector<std::uint8_t> encodeFlatDnaSequences(const std::vector<char>& flat_sequences) {
    std::vector<std::uint8_t> encoded_sequences;
    encoded_sequences.reserve(flat_sequences.size());

    for (char base : flat_sequences) {
        encoded_sequences.push_back(encodeDnaBase(base));
    }

    return encoded_sequences;
}

inline std::size_t hamming_distance(const std::string& first_sequence,
                                    const std::string& second_sequence) {
    if (first_sequence.size() != second_sequence.size()) {
        throw std::invalid_argument("Hamming Distance requires equal-length sequences.");
    }

    std::size_t distance = 0;
    for (std::size_t index = 0; index < first_sequence.size(); ++index) {
        if (first_sequence[index] != second_sequence[index]) {
            ++distance;
        }
    }
    return distance;
}

}  // namespace dna

#endif  // DNA_UTILS_H
