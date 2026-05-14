#ifndef SMITH_WATERMAN_H
#define SMITH_WATERMAN_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace smith_waterman {

struct ScoringScheme {
    int match_score = 2;
    int mismatch_penalty = -1;
    int gap_penalty = -2;
};

inline int substitution_score(char first_base, char second_base, const ScoringScheme& scoring) {
    return first_base == second_base ? scoring.match_score : scoring.mismatch_penalty;
}

inline int score_full_matrix(const std::string& first_sequence,
                             const std::string& second_sequence,
                             const ScoringScheme& scoring) {
    const std::size_t rows = first_sequence.size() + 1;
    const std::size_t cols = second_sequence.size() + 1;
    std::vector<int> dp(rows * cols, 0);
    int max_score = 0;

    const auto index = [cols](std::size_t row, std::size_t col) {
        return row * cols + col;
    };

    for (std::size_t row = 1; row < rows; ++row) {
        for (std::size_t col = 1; col < cols; ++col) {
            const int diagonal_score = dp[index(row - 1, col - 1)] +
                                       substitution_score(
                                           first_sequence[row - 1],
                                           second_sequence[col - 1],
                                           scoring);
            const int up_score = dp[index(row - 1, col)] + scoring.gap_penalty;
            const int left_score = dp[index(row, col - 1)] + scoring.gap_penalty;
            const int cell_score = std::max({0, diagonal_score, up_score, left_score});
            dp[index(row, col)] = cell_score;
            max_score = std::max(max_score, cell_score);
        }
    }

    return max_score;
}

inline int score_rolling_rows(const std::string& first_sequence,
                              const std::string& second_sequence,
                              const ScoringScheme& scoring) {
    const std::size_t cols = second_sequence.size() + 1;
    std::vector<int> previous_row(cols, 0);
    std::vector<int> current_row(cols, 0);
    int max_score = 0;

    for (std::size_t row = 1; row <= first_sequence.size(); ++row) {
        current_row[0] = 0;
        for (std::size_t col = 1; col < cols; ++col) {
            const int diagonal_score = previous_row[col - 1] +
                                       substitution_score(
                                           first_sequence[row - 1],
                                           second_sequence[col - 1],
                                           scoring);
            const int up_score = previous_row[col] + scoring.gap_penalty;
            const int left_score = current_row[col - 1] + scoring.gap_penalty;
            const int cell_score = std::max({0, diagonal_score, up_score, left_score});
            current_row[col] = cell_score;
            max_score = std::max(max_score, cell_score);
        }
        previous_row.swap(current_row);
        std::fill(current_row.begin(), current_row.end(), 0);
    }

    return max_score;
}

inline int score(const std::string& first_sequence,
                 const std::string& second_sequence,
                 const ScoringScheme& scoring,
                 const std::string& memory_mode) {
    if (memory_mode == "full") {
        return score_full_matrix(first_sequence, second_sequence, scoring);
    }
    if (memory_mode == "rolling") {
        return score_rolling_rows(first_sequence, second_sequence, scoring);
    }
    throw std::invalid_argument("Unsupported memory mode. Use full or rolling.");
}

inline int computeSmithWatermanScore(const std::string& sequence_a,
                                     const std::string& sequence_b,
                                     int match_score,
                                     int mismatch_penalty,
                                     int gap_penalty) {
    return score_rolling_rows(sequence_a, sequence_b, {match_score, mismatch_penalty, gap_penalty});
}

}  // namespace smith_waterman

#endif  // SMITH_WATERMAN_H
