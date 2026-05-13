#ifndef DNA_UTILS_H
#define DNA_UTILS_H

#include <cstddef>
#include <stdexcept>
#include <string>

namespace dna {

inline bool is_valid_dna_base(char base) {
    return base == 'A' || base == 'C' || base == 'G' || base == 'T';
}

inline bool is_valid_dna_sequence(const std::string& sequence) {
    for (char base : sequence) {
        if (!is_valid_dna_base(base)) {
            return false;
        }
    }
    return true;
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
