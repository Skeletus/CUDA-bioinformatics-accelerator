#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string executable_path() {
#ifdef _WIN32
    return ".\\needleman_wunsch_gpu.exe";
#else
    return "./needleman_wunsch_gpu";
#endif
}

void write_dataset(const std::filesystem::path& dataset_path, const std::string& contents) {
    if (dataset_path.has_parent_path()) {
        std::filesystem::create_directories(dataset_path.parent_path());
    }
    std::ofstream dataset_file(dataset_path);
    if (!dataset_file) {
        throw std::runtime_error("Failed to create test dataset: " + dataset_path.string());
    }
    dataset_file << contents;
}

bool run_gpu_validation_case(const std::string& test_case_name, const std::string& dataset_contents) {
    const std::filesystem::path input_path = std::filesystem::path("tests") /
                                             ("tmp_" + test_case_name + "_pairs.txt");
    const std::filesystem::path output_path = std::filesystem::path("results") /
                                              "needleman_wunsch" /
                                              ("tmp_" + test_case_name + "_gpu_results.csv");

    write_dataset(input_path, dataset_contents);
    std::filesystem::create_directories(output_path.parent_path());

    const std::string command =
        executable_path() + " " +
        input_path.string() + " " +
        output_path.string() +
        " --repetitions 2 --implementation wavefront --summary-only";

    const int exit_code = std::system(command.c_str());
    const bool passed = exit_code == 0;
    std::cout << "TEST_CASE=" << test_case_name
              << " STATUS=" << (passed ? "PASSED" : "FAILED") << "\n";
    return passed;
}

}  // namespace

int main() {
    try {
        bool all_tests_passed = true;
        all_tests_passed = run_gpu_validation_case("perfect_match_gpu", "ACGT ACGT\n") && all_tests_passed;
        all_tests_passed = run_gpu_validation_case("one_mismatch_gpu", "ACGT ACCT\n") && all_tests_passed;
        all_tests_passed = run_gpu_validation_case("one_deletion_gpu", "ACGT AGT\n") && all_tests_passed;
        all_tests_passed = run_gpu_validation_case("short_fixed_length_pair_gpu", "ACGT TGCA\n") && all_tests_passed;
        all_tests_passed = run_gpu_validation_case(
                               "multiple_pairs_gpu",
                               "ACGT ACGT\n"
                               "ACGT ACCT\n"
                               "TGCA TGCA\n") &&
                           all_tests_passed;

        std::cout << "ALL_TESTS_PASSED=" << (all_tests_passed ? "true" : "false") << "\n";
        return all_tests_passed ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "Error: " << error.what() << "\n";
        std::cout << "ALL_TESTS_PASSED=false\n";
        return 1;
    }
}
