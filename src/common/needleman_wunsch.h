#ifndef NEEDLEMAN_WUNSCH_H
#define NEEDLEMAN_WUNSCH_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

namespace needleman_wunsch {

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

    const auto index = [cols](std::size_t row, std::size_t col) {
        return row * cols + col;
    };

    for (std::size_t row = 1; row < rows; ++row) {
        dp[index(row, 0)] = dp[index(row - 1, 0)] + scoring.gap_penalty;
    }
    for (std::size_t col = 1; col < cols; ++col) {
        dp[index(0, col)] = dp[index(0, col - 1)] + scoring.gap_penalty;
    }

    for (std::size_t row = 1; row < rows; ++row) {
        for (std::size_t col = 1; col < cols; ++col) {
            const int diagonal_score = dp[index(row - 1, col - 1)] +
                                       substitution_score(
                                           first_sequence[row - 1],
                                           second_sequence[col - 1],
                                           scoring);
            const int up_score = dp[index(row - 1, col)] + scoring.gap_penalty;
            const int left_score = dp[index(row, col - 1)] + scoring.gap_penalty;
            dp[index(row, col)] = std::max({diagonal_score, up_score, left_score});
        }
    }

    return dp[index(rows - 1, cols - 1)];
}

inline int score_rolling_rows(const std::string& first_sequence,
                              const std::string& second_sequence,
                              const ScoringScheme& scoring) {
    const std::size_t cols = second_sequence.size() + 1;
    std::vector<int> previous_row(cols, 0);
    std::vector<int> current_row(cols, 0);

    for (std::size_t col = 1; col < cols; ++col) {
        previous_row[col] = previous_row[col - 1] + scoring.gap_penalty;
    }

    for (std::size_t row = 1; row <= first_sequence.size(); ++row) {
        current_row[0] = previous_row[0] + scoring.gap_penalty;
        for (std::size_t col = 1; col < cols; ++col) {
            const int diagonal_score = previous_row[col - 1] +
                                       substitution_score(
                                           first_sequence[row - 1],
                                           second_sequence[col - 1],
                                           scoring);
            const int up_score = previous_row[col] + scoring.gap_penalty;
            const int left_score = current_row[col - 1] + scoring.gap_penalty;
            current_row[col] = std::max({diagonal_score, up_score, left_score});
        }
        previous_row.swap(current_row);
    }

    return previous_row[cols - 1];
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

inline int computeNeedlemanWunschScore(const std::string& sequenceA,
                                       const std::string& sequenceB,
                                       int matchScore,
                                       int mismatchPenalty,
                                       int gapPenalty) {
    const ScoringScheme scoring{matchScore, mismatchPenalty, gapPenalty};
    return score_rolling_rows(sequenceA, sequenceB, scoring);
}

}  // namespace needleman_wunsch

#endif  // NEEDLEMAN_WUNSCH_H
