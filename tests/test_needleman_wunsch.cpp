#include <iostream>
#include <string>

#include "needleman_wunsch.h"

bool run_test_case(const std::string& test_case_name,
                   const std::string& first_sequence,
                   const std::string& second_sequence,
                   int expected_score,
                   const needleman_wunsch::ScoringScheme& scoring) {
    const int full_score = needleman_wunsch::score_full_matrix(first_sequence, second_sequence, scoring);
    const int rolling_score = needleman_wunsch::score_rolling_rows(first_sequence, second_sequence, scoring);
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
    const needleman_wunsch::ScoringScheme scoring{2, -1, -2};

    bool all_tests_passed = true;
    all_tests_passed &= run_test_case("perfect_match", "ACGT", "ACGT", 8, scoring);
    all_tests_passed &= run_test_case("one_mismatch", "ACGT", "ACCT", 5, scoring);
    all_tests_passed &= run_test_case("one_deletion", "ACGT", "AGT", 4, scoring);
    all_tests_passed &= run_test_case("empty_vs_non_empty", "", "ACGT", -8, scoring);
    all_tests_passed &= run_test_case("classic_global_alignment", "GATTACA", "GCATGCU", 2, scoring);

    std::cout << "ALL_TESTS_PASSED=" << (all_tests_passed ? "true" : "false") << "\n";
    return all_tests_passed ? 0 : 1;
}
