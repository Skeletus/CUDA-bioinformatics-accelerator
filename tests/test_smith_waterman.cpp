#include <iostream>
#include <string>

#include "smith_waterman.h"

bool run_test_case(const std::string& test_case_name,
                   const std::string& first_sequence,
                   const std::string& second_sequence,
                   int expected_score,
                   const smith_waterman::ScoringScheme& scoring) {
    const int full_score = smith_waterman::score_full_matrix(first_sequence, second_sequence, scoring);
    const int rolling_score = smith_waterman::score_rolling_rows(first_sequence, second_sequence, scoring);
    const bool passed = full_score == expected_score && rolling_score == expected_score;

    std::cout << "TEST_CASE=" << test_case_name
              << " STATUS=" << (passed ? "PASSED" : "FAILED")
              << " EXPECTED_SCORE=" << expected_score
              << " FULL_SCORE=" << full_score
              << " ROLLING_SCORE=" << rolling_score
              << "\n";

    return passed;
}

int main() {
    const smith_waterman::ScoringScheme scoring{2, -1, -2};

    bool all_tests_passed = true;
    all_tests_passed &= run_test_case("perfect_match", "ACGT", "ACGT", 8, scoring);
    all_tests_passed &= run_test_case("local_match_inside_longer_sequence", "ACGT", "TTACGTAA", 8, scoring);
    all_tests_passed &= run_test_case("no_useful_match", "AAAA", "TTTT", 0, scoring);
    all_tests_passed &= run_test_case("partial_local_match", "GATTACA", "TACA", 8, scoring);
    all_tests_passed &= run_test_case("one_mismatch", "ACGT", "ACCT", 5, scoring);
    all_tests_passed &= run_test_case("empty_sequence", "", "ACGT", 0, scoring);

    std::cout << "ALL_TESTS_PASSED=" << (all_tests_passed ? "true" : "false") << "\n";
    return all_tests_passed ? 0 : 1;
}
